defmodule Server.Experiments do
  @moduledoc """
  Data-access layer for `Server.Experiment` rows.

  Everything the master-loop Oban workers and the public landing page
  need to read or write about an experiment lives here. Pulled out
  into its own module to keep `Server.Queue` focused on the
  batch/chunk lifecycle.

  Two consumers:

    * The `Server.Workers.Experiment*` Oban workers, which checkpoint
      after every accepted bit via `update_state/2` and `mark_*` —
      a crashed/retried worker resumes from the persisted state.

    * Read APIs: the router (POST submission, GET status) and the
      landing page (rolling summary, incidents banner).
  """

  import Ecto.Query
  alias Server.{Experiment, Repo, SystemEvent}

  @stalled_threshold_seconds 300

  # An iter that has been alive (heartbeat fresh) but accepted ZERO
  # bits for this long is almost certainly bottlenecked on something
  # — typically an undersized worker swarm, an unowned-batch backlog,
  # or a misconfigured CEGAR run. Distinct from `stalled` (heartbeat
  # dead → master crashed); this fires when the master is healthy
  # but the WORK isn't moving.
  @no_progress_threshold_seconds 60 * 60

  # ── Submission ──────────────────────────────────────────────

  @doc """
  Create a new experiment and enqueue its bootstrap job. Rejects if
  there's already a pending/running row for the same env (one
  experiment at a time per env — operators can launch a second
  Ant run only after the first finishes or fails). The unique
  index `experiments_one_active_per_env` enforces this at the
  database level so a racing submit can't slip through.

  Returns `{:ok, experiment}` or `{:error, reason}`.
  """
  def create(attrs, opts \\ []) do
    submitter = Keyword.get(opts, :submitter)

    attrs = normalize_create_attrs(attrs, submitter)

    case validate_create_attrs(attrs) do
      :ok ->
        Repo.transaction(fn ->
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
              if active_conflict?(errors) do
                Repo.rollback(:already_running)
              else
                Repo.rollback({:invalid, errors})
              end
          end
        end)
        |> case do
          {:ok, experiment} -> {:ok, experiment}
          {:error, reason} -> {:error, reason}
        end

      {:error, reason} ->
        {:error, reason}
    end
  end

  # Did the insert fail because of the partial unique index on
  # active (pending/running) experiments per env? Errors come back
  # as `{message, opts}` tuples from `unique_constraint/3`.
  defp active_conflict?(errors) do
    case Keyword.get(errors, :env_name) do
      {_msg, opts} ->
        Keyword.get(opts, :constraint_name) == "experiments_one_active_per_env" or
          Keyword.get(opts, :constraint) == :unique

      _ ->
        false
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
    # Round-trip via Atom.to_string/1 rather than to_existing_atom —
    # safer on a cold BEAM where Synthex.Gym.Mujoco's `:ant` etc.
    # haven't been added to the atom table yet by any caller.
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
  Update an experiment's state. Used by workers to checkpoint
  after every accepted bit and after every iter advance.
  """
  def update_state(%Experiment{} = exp, attrs) do
    exp
    |> Experiment.changeset(attrs)
    |> Repo.update()
  end

  @doc """
  Bump the cached `best_reward`/`accepted_count` atomically. Called
  after each accepted bit. Avoids a read-modify-write race with
  concurrent reads from the public dashboard.
  """
  def record_acceptance(%Experiment{id: id}, reward) do
    sql = """
    UPDATE experiments
    SET best_reward = GREATEST(COALESCE(best_reward, $1::float8), $1::float8),
        accepted_count = accepted_count + 1,
        updated_at = now()
    WHERE id = $2
    """

    Repo.query!(sql, [reward * 1.0, Ecto.UUID.dump!(id)])
    :ok
  end

  @doc """
  Transition an experiment to "running" with baseline reward set.
  Called by `ExperimentBootstrap` once initial validation succeeds.
  """
  def mark_running(%Experiment{} = exp, baseline_reward) do
    update_state(exp, %{
      "status" => "running",
      "started_at" => exp.started_at || DateTime.utc_now(),
      "baseline_reward" => baseline_reward
    })
  end

  @doc "Transition an experiment to completed."
  def mark_completed(%Experiment{} = exp) do
    update_state(exp, %{
      "status" => "completed",
      "completed_at" => DateTime.utc_now()
    })
  end

  @doc "Transition an experiment to failed with an error message."
  def mark_failed(%Experiment{} = exp, error_message) do
    update_state(exp, %{
      "status" => "failed",
      "completed_at" => DateTime.utc_now(),
      "error" => error_message
    })
  end

  @doc "Transition an experiment to cancelled (operator-initiated)."
  def mark_cancelled(%Experiment{} = exp, reason \\ nil) do
    update_state(exp, %{
      "status" => "cancelled",
      "completed_at" => DateTime.utc_now(),
      "error" => reason
    })
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
  Per-environment summary for the public landing page.

  Reads directly from the `experiments` table — no more
  reverse-engineering experiment shape from per-bit Batch rows
  (which always lagged reality and routinely showed the wrong
  liveness state). Each row carries the canonical CEGAR
  progress, current best reward, and health.

  Returns `[%{env_name, status, active, latest, ...}, ...]`.
  """
  def summary do
    experiments = Repo.all(from e in Experiment, order_by: [desc: e.inserted_at])

    by_env = Enum.group_by(experiments, & &1.env_name)

    Enum.map(by_env, fn {env_name, exps} ->
      active = Enum.find(exps, &(&1.status in ["pending", "running"]))
      completed = Enum.filter(exps, &(&1.status == "completed"))
      latest_completed = List.first(completed)

      best_completed =
        completed
        |> Enum.map(& &1.best_reward)
        |> Enum.reject(&is_nil/1)
        |> case do
          [] -> nil
          xs -> Enum.max(xs)
        end

      %{
        env_name: env_name,
        status: cond do
          active -> active.status
          latest_completed -> "completed"
          true -> "history"
        end,
        active: render_active(active),
        latest: render_latest(latest_completed),
        best_reward: best_completed,
        completed_count: length(completed),
        total_count: length(exps)
      }
    end)
    |> Enum.sort_by(& &1.env_name)
  end

  defp render_active(nil), do: nil

  defp render_active(%Experiment{} = exp) do
    config = exp.config || %{}
    max_iters = get_int(config, "max_iters", 5)
    cegar_rounds = get_int(config, "cegar_rounds", 3)
    bits_per_dim = get_int(config, "bits_per_dim", 3)

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

    {health, polled_ago} = compute_health(exp)

    %{
      experiment_id: exp.id,
      status: exp.status,
      cegar_iter: exp.current_cegar_iter,
      total_cegar_iters: cegar_rounds,
      iter: exp.current_iter,
      total_iters: max_iters,
      bits_done: length(exp.bit_progress || []),
      n_bits: n_bits,
      accepted_count: exp.accepted_count,
      best_reward: exp.best_reward,
      baseline_reward: exp.baseline_reward,
      progress: progress,
      started_at: exp.started_at || exp.inserted_at,
      elapsed_seconds: elapsed_seconds(exp.started_at || exp.inserted_at),
      health: health,
      heartbeat_seconds_ago: polled_ago
    }
  end

  defp render_latest(nil), do: nil

  defp render_latest(%Experiment{} = exp) do
    delta =
      cond do
        is_number(exp.best_reward) and is_number(exp.baseline_reward) ->
          exp.best_reward - exp.baseline_reward

        true ->
          nil
      end

    %{
      experiment_id: exp.id,
      best_reward: exp.best_reward,
      baseline_reward: exp.baseline_reward,
      delta: delta,
      accepted_count: exp.accepted_count,
      completed_at: exp.completed_at
    }
  end

  # Liveness for an experiment. With the master loop now running as
  # Oban jobs on the hub, "live" means the most recent
  # `Server.Workers.Experiment*` heartbeat is fresh. We derive
  # heartbeat from the experiment row's `updated_at` (every checkpoint
  # bumps it) plus the existence of an unstarted/executing Oban job
  # in queue `:master` for this experiment.
  defp compute_health(%Experiment{status: "pending"} = exp) do
    age = elapsed_seconds(exp.inserted_at) || 0

    if age >= @stalled_threshold_seconds,
      do: {"no_progress", nil},
      else: {"healthy", nil}
  end

  defp compute_health(%Experiment{status: "running", updated_at: updated_at} = exp)
       when not is_nil(updated_at) do
    secs = DateTime.diff(DateTime.utc_now(), updated_at, :second)
    elapsed = elapsed_seconds(exp.started_at || exp.inserted_at) || 0
    bits_done = length(exp.bit_progress || [])

    health =
      cond do
        secs > @stalled_threshold_seconds -> "stalled"
        bits_done == 0 and elapsed > @no_progress_threshold_seconds -> "no_progress"
        true -> "healthy"
      end

    {health, secs}
  end

  defp compute_health(_exp), do: {"unknown", nil}

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
end
