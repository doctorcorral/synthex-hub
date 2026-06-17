defmodule Server.Experiments do
  @moduledoc """
  Data-access layer for `Server.Experiment` rows.

  Experiments are now *sessions* — the policy itself lives on
  `Server.EnvPolicy` (the lineage, owned per `(env_name, config_sig)`).
  This module focuses on the session lifecycle (create, start, mark
  terminal) and on the per-(env, sig) dashboard summary that joins
  sessions to their lineage.

  Three consumers:

    * `Server.Workers.Experiment*` Oban workers, which read/write
      session state via the helpers here.

    * Read APIs: the router (POST submission, GET status) and the
      landing page (rolling summary, incidents banner).

    * `Server.AggregateBroker`, which fetches lightweight session
      records to render the streaming SSE feed.
  """

  import Ecto.Query
  alias Server.{AggregateBroker, EnvPolicies, EnvPolicy, Experiment,
                PolicyVersion, Repo, SystemEvent}
  alias Server.EnvPolicy.ConfigSig

  # Heartbeat dead → controller Oban job has crashed/disappeared.
  # Surfaces as `stalled` on the dashboard.
  @stalled_threshold_seconds 300

  # Default rollout count per candidate when the experiment config
  # doesn't specify one — mirrors
  # `Server.Workers.ExperimentBootstrap.config_to_opts/1`. Used to
  # normalize the *summed* reward stored in `Experiment.best_reward`
  # (one number per candidate = Σ over n_episodes) back into a
  # per-episode mean for the dashboard.
  @default_n_episodes 30

  # Heartbeat fresh AND there's an in-flight wave AND no chunks have
  # completed in this long → workers stopped pulling work even though
  # the controller wants more. Surfaces as `idle` on the dashboard.
  @idle_threshold_seconds 5 * 60

  # First few minutes after a wave is dispatched there will legitimately
  # be no chunk completions yet.
  @boot_grace_seconds 5 * 60

  # Heartbeat fresh AND chunks ARE flowing but the swarm is so far
  # below the workload's bit granularity that no bit will be accepted
  # for ages. Surfaces as `slow` on the dashboard.
  @slow_after_seconds 60 * 60

  # ── Submission ──────────────────────────────────────────────

  @doc """
  Create a new experiment session and enqueue its bootstrap job.

  Per the env_policies promotion: two submissions for the same
  `env_name` with DIFFERENT `config_sig`s run in parallel (different
  lineages); two with the SAME `config_sig` collide on the partial
  unique index `experiments_one_active_per_env_policy`.

  We compute the sig client-side (without touching env_policies yet —
  bootstrap does the upsert) and pre-check for an active session on
  the same lineage to fail fast with a clean 409 rather than waiting
  for the partial unique index to bite, because the index can't
  conflict until bootstrap has set the experiment row's
  `env_policy_id`.

  Returns `{:ok, experiment}` or `{:error, reason}`.
  """
  def create(attrs, opts \\ []) do
    submitter = Keyword.get(opts, :submitter)

    attrs = normalize_create_attrs(attrs, submitter)

    case validate_create_attrs(attrs) do
      :ok ->
        config = Map.get(attrs, "config") || %{}
        {sig, _canonical} = ConfigSig.sig_for_config(config)
        env_name = Map.get(attrs, "env_name")

        if active_for_sig?(env_name, sig) do
          {:error, :already_running}
        else
          do_insert_experiment(attrs)
        end

      {:error, reason} ->
        {:error, reason}
    end
  end

  # The experiment row's `env_policy_id` is NOT NULL — so resolve
  # (or create) the lineage row SYNCHRONOUSLY here, before inserting.
  # If multiple submissions for the same `(env_name, config_sig)`
  # land simultaneously and BOTH pass `active_for_sig?` (because the
  # lineage doesn't exist yet), `EnvPolicies.upsert_for_submission/2`
  # collapses them onto a single row via the unique index — and the
  # `experiments_one_active_per_env_policy` partial unique index
  # then kills the second experiment insert.
  defp do_insert_experiment(attrs) do
    env_attrs = %{
      env_name: Map.get(attrs, "env_name"),
      env_key: Map.get(attrs, "env_key"),
      config: Map.get(attrs, "config") || %{}
    }

    Repo.transaction(fn ->
      case Server.EnvPolicies.upsert_for_submission(env_attrs) do
        {:ok, env_policy, _state} ->
          attrs_with_lineage = Map.put(attrs, "env_policy_id", env_policy.id)
          insert_experiment_with_lineage(attrs_with_lineage)

        {:error, reason} ->
          Repo.rollback({:env_policy_failed, reason})
      end
    end)
    |> case do
      {:ok, experiment} -> {:ok, experiment}
      {:error, {:experiment_conflict, _}} -> {:error, :already_running}
      {:error, reason} -> {:error, reason}
    end
  end

  defp insert_experiment_with_lineage(attrs) do
    changeset = Experiment.changeset(%Experiment{}, attrs)

    case Repo.insert(changeset) do
      {:ok, experiment} ->
        case Server.Workers.ExperimentBootstrap.new(%{"experiment_id" => experiment.id})
             |> Oban.insert() do
          {:ok, _job} ->
            log_event!("info", "master",
              "experiment submitted: #{experiment.env_name} (#{experiment.id})",
              env_name: experiment.env_name,
              experiment_id: experiment.id
            )

            experiment

          {:error, reason} ->
            Repo.rollback({:enqueue_failed, reason})
        end

      {:error, %Ecto.Changeset{errors: errors}} ->
        # The partial unique index `experiments_one_active_per_env_policy`
        # fires here under the racy double-submission case described
        # above. Surface a clean :already_running rather than a raw
        # changeset.
        if Keyword.has_key?(errors, :env_policy_id) do
          Repo.rollback({:experiment_conflict, errors})
        else
          Repo.rollback({:invalid, errors})
        end
    end
  end

  # Pre-check: is there an active (pending/running) session for this
  # `(env_name, sig)`? We resolve the lineage row first; if it exists
  # AND has an active session attached, reject. If it doesn't exist
  # OR exists but has no active session, the partial unique index
  # picks up the slack post-bootstrap.
  defp active_for_sig?(env_name, sig) do
    case Repo.get_by(EnvPolicy, env_name: env_name, config_sig: sig) do
      nil ->
        false

      %EnvPolicy{id: env_policy_id} ->
        from(e in Experiment,
          where:
            e.env_policy_id == ^env_policy_id and
              e.status in ["pending", "running"]
        )
        |> Repo.exists?()
    end
  end

  defp normalize_create_attrs(attrs, submitter) do
    %{
      "env_key" => to_string(Map.get(attrs, "env_key") || Map.get(attrs, :env_key) || ""),
      "env_name" => to_string(Map.get(attrs, "env_name") || Map.get(attrs, :env_name) || ""),
      "config" => Map.get(attrs, "config") || Map.get(attrs, :config) || %{},
      "submitter" => submitter,
      "status" => "pending"
    }
  end

  defp validate_create_attrs(%{"env_key" => env_key, "env_name" => env_name})
       when env_key == "" or env_name == "" do
    {:error, :missing_env}
  end

  defp validate_create_attrs(%{"env_key" => env_key}) do
    Code.ensure_loaded(Synthex.Gym.Mujoco)

    Synthex.Gym.Mujoco.known_envs()
    |> Enum.find(fn atom -> Atom.to_string(atom) == env_key end)
    |> case do
      nil -> {:error, {:unknown_env, env_key}}
      _atom -> :ok
    end
  end

  # ── CRUD ────────────────────────────────────────────────────

  @doc "Fetch one experiment by UUID."
  def get(id) when is_binary(id) do
    case Repo.get(Experiment, id) do
      nil -> {:error, :not_found}
      exp -> {:ok, exp}
    end
  end

  @doc """
  Update an experiment's session state. Used by workers to
  checkpoint after CEGAR-step advances and to transition status.
  Note: this no longer touches predicates / policy_version — those
  live on the env_policy and are mutated only via `Server.CommitGate`.
  """
  def update_state(%Experiment{} = exp, attrs) do
    exp
    |> Experiment.changeset(attrs)
    |> Repo.update()
  end

  @doc """
  Bump an experiment's `updated_at` without changing any state. Used
  as a paused-experiment heartbeat by the scorer's poll loop: while a
  run waits for a capable worker to (re)appear, this keeps the
  controller-liveness clock fresh so `OrphanReaper` and the dashboard
  treat it as alive-and-waiting rather than a crashed/stalled master.
  Cheap: a single indexed UPDATE, no struct load.
  """
  @spec touch(String.t()) :: :ok
  def touch(experiment_id) when is_binary(experiment_id) do
    from(e in Experiment, where: e.id == ^experiment_id)
    |> Repo.update_all(set: [updated_at: DateTime.utc_now()])

    :ok
  rescue
    _ -> :ok
  end

  def touch(_), do: :ok

  @doc "Transition an experiment to completed."
  def mark_completed(%Experiment{} = exp) do
    update_state(exp, %{
      "status" => "completed",
      "completed_at" => DateTime.utc_now()
    })
  end

  @doc "Transition an experiment to failed with an error message."
  def mark_failed(%Experiment{} = exp, error_message) do
    result =
      update_state(exp, %{
        "status" => "failed",
        "completed_at" => DateTime.utc_now(),
        "error" => error_message
      })

    reap_in_flight(exp, "failed")
    result
  end

  @doc "Transition an experiment to cancelled (operator-initiated)."
  def mark_cancelled(%Experiment{} = exp, reason \\ nil) do
    result =
      update_state(exp, %{
        "status" => "cancelled",
        "completed_at" => DateTime.utc_now(),
        "error" => reason
      })

    reap_in_flight(exp, "cancelled")
    result
  end

  # Immediately cancel the experiment's outstanding chunk jobs on a
  # terminal transition. Without this, a cancelled/failed experiment's
  # `available` chunks sit claimable until the OrphanReaper cron runs
  # (every 2 min) — wasting worker cycles on dead work and leaving
  # thousands of `available` rows pressuring Postgrex. The reaper
  # remains as the defense-in-depth safety net; this just closes the
  # latency gap on the common path. Best-effort: a failure here must
  # never block the state transition.
  defp reap_in_flight(%Experiment{id: id, env_name: env_name}, status) do
    case Server.Queue.cancel_experiment_in_flight_batches(id) do
      {:ok, %{batches: 0, jobs: 0}} ->
        :ok

      {:ok, %{batches: n_batches, jobs: n_jobs}} ->
        log_event!(
          "warn",
          "master",
          "reaped #{n_jobs} chunk job(s) across #{n_batches} batch(es) on #{status}: #{env_name}",
          env_name: env_name,
          experiment_id: id,
          metadata: %{"cancelled_jobs" => n_jobs, "cancelled_batches" => n_batches, "trigger" => status}
        )

      _ ->
        :ok
    end
  rescue
    _ -> :ok
  end

  @doc "List experiments newest first."
  def list(limit \\ 50) do
    from(e in Experiment,
      order_by: [desc: e.inserted_at],
      limit: ^limit
    )
    |> Repo.all()
  end

  # ── Public dashboard ────────────────────────────────────────

  @doc """
  Per-lineage summary for the public landing page. One row per
  `(env_name, config_sig)` — a card.

  Each card surfaces:

    * The lineage's all-time best reward (env_policy.best_reward),
      the canonical "current best" since the lineage IS the policy.
      No more "ALL-TIME vs current run" gymnastics — they're the
      same number.

    * The active session attached to the lineage (if any), plus its
      progress and health.

    * The most recent terminal session on the lineage, with status
      and per-session delta, for context on what the last attempt did.

  Returns `[%{env_name, config_sig, status, active, last_run, ...}, ...]`.
  """
  def summary do
    env_policies = EnvPolicies.list_all()

    # Pre-fetch all sessions, group by env_policy_id once.
    by_lineage =
      Repo.all(
        from e in Experiment,
          where: not is_nil(e.env_policy_id),
          order_by: [desc: e.inserted_at]
      )
      |> Enum.group_by(& &1.env_policy_id)

    env_policies
    |> Enum.map(&render_lineage(&1, Map.get(by_lineage, &1.id, [])))
    |> Enum.sort_by(fn row -> {row.env_name, row.config_sig} end)
  end

  defp render_lineage(%EnvPolicy{} = env_policy, sessions) do
    active = Enum.find(sessions, &(&1.status in ["pending", "running"]))
    completed = Enum.filter(sessions, &(&1.status == "completed"))
    latest_completed = List.first(completed)

    last_run = Enum.find(sessions, &(&1.status not in ["pending", "running"]))

    %{
      env_name: env_policy.env_name,
      env_key: env_policy.env_key,
      env_policy_id: env_policy.id,
      config_sig: env_policy.config_sig,
      config_summary: ConfigSig.summary(env_policy.config_data || %{}),
      config_data: env_policy.config_data,
      policy_version: env_policy.policy_version,
      n_episodes: env_policy.n_episodes || @default_n_episodes,
      # Canonical, cross-run-comparable performance: per-episode mean on
      # the fixed held-out validation block (already per-episode; no
      # normalization). nil until the lineage runs a step under a
      # validation-aware controller.
      validation_avg: env_policy.validation_avg,
      validation_version: env_policy.validation_version,
      validation_stale: validation_stale?(env_policy),
      # Tail/robustness of the held-out block (worst-10% mean etc.), or
      # nil if the worker can't emit per-seed returns yet.
      validation_tail: env_policy.validation_tail,
      # `best_reward` is a TRAINING high-water-mark (sum over per-round
      # scoring seeds, max-selected). Kept for forensics/back-compat,
      # but `validation_avg` is the honest headline. Normalize to per-ep.
      best_reward: normalize_reward(env_policy.best_reward, env_policy.n_episodes || @default_n_episodes),
      best_reward_sum: env_policy.best_reward,
      baseline_reward: normalize_reward(env_policy.baseline_reward, env_policy.n_episodes || @default_n_episodes),
      baseline_reward_sum: env_policy.baseline_reward,
      lineage_first_seen_at: env_policy.first_seen_at,
      lineage_updated_at: env_policy.updated_at,
      status:
        cond do
          active -> active.status
          last_run -> last_run.status
          true -> "history"
        end,
      active: render_active(active, env_policy),
      latest: render_latest(latest_completed, env_policy),
      last_run: render_last_run(last_run, env_policy),
      completed_count: length(completed),
      total_count: length(sessions)
    }
  end

  defp render_active(nil, _env_policy), do: nil

  defp render_active(%Experiment{} = exp, %EnvPolicy{} = env_policy) do
    config = exp.config || %{}
    max_iters = get_int(config, "max_iters", 5)
    cegar_rounds = get_int(config, "cegar_rounds", 3)
    bits_per_dim = get_int(config, "bits_per_dim", 3)
    n_episodes = n_episodes_for(exp)

    n_bits =
      try do
        env_atom = Server.Workers.ExperimentBootstrap.decode_env_key(exp.env_key)
        cfg = Synthex.Gym.Mujoco.env_config(env_atom)
        bits_per_dim * cfg.n_action_dims
      rescue
        _ -> nil
      end

    total_iters = cegar_rounds * max_iters
    done_iters = max(0, (exp.current_cegar_iter - 1) * max_iters + (exp.current_iter - 1))
    progress = if total_iters > 0, do: done_iters / total_iters, else: 0.0

    flow = AggregateBroker.experiment_flow(exp.id)
    {health, polled_ago} = compute_health(exp, flow)

    %{
      experiment_id: exp.id,
      status: exp.status,
      cegar_iter: exp.current_cegar_iter,
      total_cegar_iters: cegar_rounds,
      iter: exp.current_iter,
      total_iters: max_iters,
      n_bits: n_bits,
      accepted_count: exp.accepted_count,
      # Per-episode means for display. Raw sums alongside for forensic
      # / API consumers. Session-scoped values; for the lineage's
      # all-time best look at the top-level `best_reward`.
      best_reward: normalize_reward(exp.best_reward, n_episodes),
      baseline_reward: normalize_reward(exp.baseline_reward, n_episodes),
      best_reward_sum: exp.best_reward,
      baseline_reward_sum: exp.baseline_reward,
      n_episodes: n_episodes,
      progress: progress,
      started_at: exp.started_at || exp.inserted_at,
      elapsed_seconds: elapsed_seconds(exp.started_at || exp.inserted_at),
      health: health,
      heartbeat_seconds_ago: polled_ago,
      chunks_per_min: flow && flow.chunks_per_min,
      chunks_done: flow && flow.chunks_done,
      chunks_total: flow && flow.chunks_total,
      chunks_pending: flow && flow.chunks_pending,
      n_active_bits: flow && flow.n_active_bits,
      eta_wave_seconds: eta_wave_seconds(flow, n_bits),
      wave_dispatched_chunks: flow && flow.wave_dispatched_chunks,
      wave_done_chunks: flow && flow.wave_done_chunks,
      wave_total_chunks_estimate: wave_total_chunks_estimate(flow, n_bits),
      # Lineage version is the source of truth — surface it on the active
      # session's payload so the dashboard's "v=N" reflects the lineage,
      # not a session-scoped counter.
      policy_version: env_policy.policy_version,
      latest_commits: render_commits(latest_commits_for_lineage(env_policy.id, 5), n_episodes)
    }
  end

  defp eta_wave_seconds(nil, _n_bits), do: nil

  defp eta_wave_seconds(%{chunks_per_min: rate} = flow, n_bits)
       when is_integer(rate) and rate > 0 do
    # `wave_total_chunks_estimate/2` returns nil before any bit-
    # batches are dispatched (collect_states phase, or the brief
    # window between wave start and the first per-bit chunk landing
    # in the DB) — we can't compute a useful ETA yet.
    case wave_total_chunks_estimate(flow, n_bits) do
      nil ->
        nil

      total_estimate when is_integer(total_estimate) ->
        done = flow.wave_done_chunks || 0
        pending = max(total_estimate - done, 0)

        cond do
          pending == 0 -> nil
          true -> div(pending * 60, rate)
        end
    end
  end

  defp eta_wave_seconds(_, _), do: nil

  defp wave_total_chunks_estimate(nil, _n_bits), do: nil

  defp wave_total_chunks_estimate(%{wave_dispatched_chunks: dispatched, wave_dispatched_bits: n_disp},
                                  n_bits)
       when is_integer(dispatched) and dispatched > 0 and is_integer(n_disp) and n_disp > 0 do
    per_bit = dispatched / n_disp

    case n_bits do
      n when is_integer(n) and n > n_disp ->
        dispatched + round((n - n_disp) * per_bit)

      _ ->
        dispatched
    end
  end

  defp wave_total_chunks_estimate(_flow, _n_bits), do: nil

  defp render_last_run(nil, _env_policy), do: nil

  defp render_last_run(%Experiment{} = exp, _env_policy) do
    n_episodes = n_episodes_for(exp)

    best_mean = normalize_reward(exp.best_reward, n_episodes)
    baseline_mean = normalize_reward(exp.baseline_reward, n_episodes)

    delta =
      cond do
        is_number(best_mean) and is_number(baseline_mean) ->
          best_mean - baseline_mean

        true ->
          nil
      end

    %{
      experiment_id: exp.id,
      status: exp.status,
      best_reward: best_mean,
      baseline_reward: baseline_mean,
      best_reward_sum: exp.best_reward,
      baseline_reward_sum: exp.baseline_reward,
      n_episodes: n_episodes,
      delta: delta,
      accepted_count: exp.accepted_count || 0,
      completed_at: exp.completed_at,
      error: exp.error
    }
  end

  defp render_latest(nil, _env_policy), do: nil

  defp render_latest(%Experiment{} = exp, _env_policy) do
    n_episodes = n_episodes_for(exp)

    best_mean = normalize_reward(exp.best_reward, n_episodes)
    baseline_mean = normalize_reward(exp.baseline_reward, n_episodes)

    delta =
      cond do
        is_number(best_mean) and is_number(baseline_mean) ->
          best_mean - baseline_mean

        true ->
          nil
      end

    %{
      experiment_id: exp.id,
      best_reward: best_mean,
      baseline_reward: baseline_mean,
      best_reward_sum: exp.best_reward,
      baseline_reward_sum: exp.baseline_reward,
      n_episodes: n_episodes,
      delta: delta,
      accepted_count: exp.accepted_count,
      completed_at: exp.completed_at
    }
  end

  defp compute_health(%Experiment{status: "pending"} = exp, _flow) do
    age = elapsed_seconds(exp.inserted_at) || 0

    if age >= @stalled_threshold_seconds,
      do: {"idle", nil},
      else: {"healthy", nil}
  end

  defp compute_health(%Experiment{status: "running", updated_at: updated_at} = exp, flow)
       when not is_nil(updated_at) do
    secs = DateTime.diff(DateTime.utc_now(), updated_at, :second)
    elapsed = elapsed_seconds(exp.started_at || exp.inserted_at) || 0

    health =
      cond do
        secs > @stalled_threshold_seconds ->
          "stalled"

        elapsed > @boot_grace_seconds and chunks_stuck?(flow) ->
          # Distinguish a paused run (no live worker can claim this
          # adapter's chunks — e.g. a mujoco_warp run while the GPU box
          # is offline; it will resume on its own when one rejoins)
          # from a genuine idle swarm (capable workers present but not
          # pulling). Paused is expected, not an incident.
          if paused_for_missing_worker?(exp), do: "paused", else: "idle"

        (exp.accepted_count || 0) == 0 and elapsed > @slow_after_seconds and chunks_flowing?(flow) ->
          "slow"

        true ->
          "healthy"
      end

    {health, secs}
  end

  defp compute_health(_exp, _flow), do: {"unknown", nil}

  # True when the experiment's physics adapter has no live worker, so
  # its queued chunks can't be claimed and the run is waiting (paused),
  # not failing. Only consulted for an already-stuck running experiment,
  # so this is a bounded, rare workers-table check.
  defp paused_for_missing_worker?(%Experiment{config: config}) do
    adapter =
      case is_map(config) && (Map.get(config, "adapter") || Map.get(config, :adapter)) do
        a when is_binary(a) and a != "" -> a
        _ -> "mujoco"
      end

    not Server.Queue.adapter_has_live_worker?(adapter)
  rescue
    _ -> false
  end

  defp chunks_flowing?(nil), do: false

  defp chunks_flowing?(%{chunks_per_min: rate, last_progress_at: last})
       when is_integer(rate) and rate > 0,
       do: not stale_progress?(last)

  defp chunks_flowing?(%{last_progress_at: last}), do: not stale_progress?(last)

  defp chunks_stuck?(nil), do: false
  defp chunks_stuck?(%{chunks_pending: pending}) when not is_integer(pending) or pending <= 0,
    do: false

  defp chunks_stuck?(%{last_progress_at: nil, chunks_pending: pending}) when pending > 0, do: true

  defp chunks_stuck?(%{last_progress_at: last}), do: stale_progress?(last)

  defp stale_progress?(nil), do: true

  defp stale_progress?(%DateTime{} = last) do
    DateTime.diff(DateTime.utc_now(), last, :second) > @idle_threshold_seconds
  end

  defp elapsed_seconds(nil), do: nil
  defp elapsed_seconds(%DateTime{} = dt), do: DateTime.diff(DateTime.utc_now(), dt, :second)

  defp get_int(map, key, default) when is_map(map) do
    case Map.get(map, key, default) do
      n when is_integer(n) -> n
      n when is_float(n) -> trunc(n)
      _ -> default
    end
  end

  # ── System events ──────────────────────────────────────────

  @doc """
  Insert a system_event row. Raises on changeset failure — incidents
  must never be silently dropped. Returns the inserted event.
  """
  def log_event!(level, source, message, opts \\ []) do
    attrs = %{
      "level" => level,
      "source" => source,
      "message" => message,
      "env_name" => Keyword.get(opts, :env_name),
      "experiment_id" => Keyword.get(opts, :experiment_id),
      "metadata" => Keyword.get(opts, :metadata, %{})
    }

    %SystemEvent{}
    |> SystemEvent.changeset(attrs)
    |> Repo.insert!()
  end

  @doc """
  Recent system events for the public banner. Filters to
  `level in (warn, error)` over the last `hours` hours.
  """
  def recent_incidents(hours \\ 24, limit \\ 50) do
    cutoff = DateTime.add(DateTime.utc_now(), -hours * 3600, :second)

    from(e in SystemEvent,
      where: e.level in ["warn", "error"] and e.inserted_at >= ^cutoff,
      order_by: [desc: e.inserted_at],
      limit: ^limit,
      select: %{
        id: e.id,
        level: e.level,
        source: e.source,
        message: e.message,
        env_name: e.env_name,
        experiment_id: e.experiment_id,
        inserted_at: e.inserted_at
      }
    )
    |> Repo.all()
  end

  # ── Streaming CEGAR commit log ────────────────────────────────

  @doc """
  Most-recent commit-gate accepts for a lineage, newest first. Used
  by:

    * `render_active/2` — to surface the last few commits on the
      dashboard card.
    * `Server.AggregateBroker` — to fan-out commit events via SSE.

  The `predicates` blob is NOT returned (it's the full policy state,
  too heavy for a frequent SSE tick). Use `Server.EnvPolicies.get/1`
  + `policy_versions` joins for full audit replay.
  """
  def latest_commits_for_lineage(env_policy_id, limit \\ 5) do
    from(v in PolicyVersion,
      where: v.env_policy_id == ^env_policy_id,
      order_by: [desc: v.version],
      limit: ^limit,
      select: %{
        version: v.version,
        bit_idx: v.bit_idx,
        prev_reward: v.prev_reward,
        new_reward: v.new_reward,
        worker_id: v.worker_id,
        committed_at: v.inserted_at,
        committed_by_experiment_id: v.experiment_id,
        metadata: v.metadata
      }
    )
    |> Repo.all()
  end

  defp render_commits(rows, n_episodes) do
    Enum.map(rows, fn row ->
      prev_mean = normalize_reward(row.prev_reward, n_episodes)
      new_mean = normalize_reward(row.new_reward, n_episodes)

      delta =
        cond do
          is_number(new_mean) and is_number(prev_mean) ->
            new_mean - prev_mean

          true ->
            nil
        end

      %{
        version: row.version,
        bit_idx: row.bit_idx,
        prev_reward: prev_mean,
        new_reward: new_mean,
        prev_reward_sum: row.prev_reward,
        new_reward_sum: row.new_reward,
        n_episodes: n_episodes,
        delta: delta,
        committed_at: row.committed_at,
        committed_seconds_ago: elapsed_seconds(row.committed_at),
        worker_id: row.worker_id,
        committed_by_experiment_id: row.committed_by_experiment_id,
        cegar_iter: Map.get(row.metadata || %{}, "cegar_iter"),
        wave: Map.get(row.metadata || %{}, "wave")
      }
    end)
  end

  # ── Reward normalization helpers ────────────────────────────

  @doc """
  Per-experiment `n_episodes` (rollouts per candidate), used to
  convert summed candidate rewards into per-episode means.

  Falls back to `@default_n_episodes` (30) when the config doesn't
  set one. Guards against zero/negative.
  """
  def n_episodes_for(%Experiment{config: config}) do
    case get_int(config || %{}, "n_episodes", @default_n_episodes) do
      n when is_integer(n) and n > 0 -> n
      _ -> @default_n_episodes
    end
  end

  @doc """
  Convert a summed reward (one number per candidate = Σ over
  n_episodes) into a per-episode mean. Returns `nil` unchanged so
  callers can pipe optional fields through without guarding.
  """
  def normalize_reward(nil, _n_episodes), do: nil
  def normalize_reward(_value, n_episodes) when n_episodes <= 0, do: nil
  def normalize_reward(value, n_episodes) when is_number(value), do: value / n_episodes
  def normalize_reward(_value, _n_episodes), do: nil

  # `validation_avg` lags `policy_version` if a commit landed after the
  # last step-end validation measurement. Flag it so the dashboard can
  # mark a held-out number as not-yet-reflecting the very latest bit.
  defp validation_stale?(%EnvPolicy{validation_avg: nil}), do: true

  defp validation_stale?(%EnvPolicy{validation_version: v, policy_version: pv})
       when is_integer(v) and is_integer(pv),
       do: v < pv

  defp validation_stale?(_), do: true
end
