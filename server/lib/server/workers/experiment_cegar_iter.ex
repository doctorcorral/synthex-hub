defmodule Server.Workers.ExperimentCegarIter do
  @moduledoc """
  One CEGAR iteration. Dispatches all `n_bits` bit searches
  concurrently against the same frozen baseline policy (Jacobi
  coordinate descent), composes the accepted deltas at iter end,
  then enqueues either the next iter or `ExperimentComplete`.

  See `docs/streaming-cegar.md` §Layer 2 for the design rationale.
  The synchronous bit-by-bit (Gauss-Seidel) version it replaces
  serialized the swarm to one bit's batch at a time; the Jacobi
  variant saturates the swarm with `n_bits` independent batches in
  flight at once, all evaluated against the iter's starting policy
  `v_i`. `v_{i+1}` is `v_i` with every accepted bit delta applied.

  ## Resume on crash

  The iter is idempotent at the `(experiment_id, cegar_iter, iter)`
  level. Per-bit results are not persisted; on retry every bit is
  re-dispatched against the same baseline.

    * Oban retries (max_attempts = 3, exponential backoff).
    * The retry re-collects states / rebuilds features (cheap
      relative to bit searches), and re-issues all bit batches.
      Stale batches from the previous attempt are abandoned and
      cleared up by `Server.Jobs.OrphanReaper` via
      `master_polled_at`.

  After three exhausted attempts the failure handler in
  `Server.ObanFailureHandler` marks the experiment as `failed` and
  records an `error`-level `system_event`. The landing page picks
  that up as a banner.

  ## Heartbeat

  Iter jobs can take many hours on slow envs, and a single bit
  search can churn for minutes without producing an acceptance.
  Two things need a regular pulse:

    * Oban's `Lifeline` plugin reaps "executing" jobs whose
      `attempted_at` falls behind `rescue_after`. We bump
      `attempted_at` to keep the iter alive in Oban.
    * The public dashboard derives `health` from
      `experiments.updated_at` (see `Experiments.compute_health/1`).
      Without a beat the dashboard mislabels a healthy iter as
      `stalled` whenever a bit search runs longer than the
      stalled-threshold.

  A single `Task` updates both rows every
  `@heartbeat_interval_ms`; it dies on `perform/1` exit.
  """

  use Oban.Worker,
    queue: :master,
    max_attempts: 3,
    # Only one ACTIVE iter job per experiment at a time. We
    # intentionally OMIT `:completed` from `states`: when an iter
    # finishes successfully it enqueues the next iter for the same
    # experiment, and we don't want the just-completed sibling
    # blocking that. `:infinity` period means "look at all jobs in
    # the listed states" without a lookback window.
    unique: [
      keys: [:experiment_id],
      states: [:available, :scheduled, :executing, :retryable],
      period: :infinity
    ]

  require Logger
  import Ecto.Query

  alias Server.{Experiments, Experiment, Repo}
  alias Server.Workers.ExperimentBootstrap
  alias Synthex.Core.PrettyPrint
  alias Synthex.Gym.Mujoco

  @heartbeat_interval_ms 60_000

  @impl Oban.Worker
  def perform(%Oban.Job{id: job_id, args: %{"experiment_id" => exp_id}}) do
    case Experiments.get(exp_id) do
      {:error, :not_found} ->
        {:discard, "experiment not found"}

      {:ok, %Experiment{status: status}} when status != "running" ->
        Logger.info("[CegarIter] experiment #{exp_id} is #{status}; exiting")
        :ok

      {:ok, %Experiment{} = exp} ->
        heartbeat = start_heartbeat(job_id, exp.id)

        try do
          do_iter(exp)
        after
          stop_heartbeat(heartbeat)
        end
    end
  end

  defp do_iter(%Experiment{} = exp) do
    env_key = ExperimentBootstrap.decode_env_key(exp.env_key)
    ctx = ExperimentBootstrap.build_context(env_key, exp.config, exp.id)

    preds = decode_predicates(exp.predicates)
    cegar_iter = exp.current_cegar_iter
    iter = exp.current_iter

    Logger.info(
      "[CegarIter] #{exp.env_name} CEGAR #{cegar_iter}/#{ctx.cegar_rounds} " <>
        "iter #{iter}/#{ctx.max_iters} (Jacobi parallel-bit dispatch)"
    )

    # collect_states + features are deterministic given (preds, ctx).
    # Cheap relative to the bit searches that follow (~minutes vs
    # ~hours), so we just re-run on every iter attempt instead of
    # persisting potentially-multi-MB feature blobs in jsonb.
    {states, _} = Mujoco.collect_states(preds, ctx)
    features = Mujoco.build_features(states, ctx)

    Logger.info("[CegarIter] #{length(features)} features built")

    bit_shuffle = ensure_bit_shuffle(exp, ctx, cegar_iter, iter)
    seeds = Mujoco.seeds_for(cegar_iter, iter, ctx)

    Logger.info(
      "[CegarIter] dispatching #{length(bit_shuffle)} bit batches concurrently " <>
        "against the same baseline"
    )

    # Jacobi semantics: every bit's batch evaluates candidates against
    # the SAME baseline `preds`. The accepted deltas (one per bit at
    # most) are composed at iter end. See
    # `docs/streaming-cegar.md` §Layer 2 — the parallel form is what
    # HybridSynthesis §11.1 actually describes; the prior serial form
    # was an implementation artefact.
    results =
      bit_shuffle
      |> Task.async_stream(
        fn bit_idx ->
          run_one_bit_jacobi(exp, ctx, preds, bit_idx, features, seeds, cegar_iter, iter)
        end,
        max_concurrency: max(length(bit_shuffle), 1),
        timeout: :infinity,
        ordered: false
      )
      |> Enum.flat_map(fn
        {:ok, result} ->
          [result]

        {:exit, reason} ->
          Logger.warning("[CegarIter] bit task exited unexpectedly: #{inspect(reason)}")
          []
      end)

    accepted = Enum.filter(results, fn {_bit, accepted?, _, _} -> accepted? end)
    accepted_in_iter = length(accepted)

    # Apply accepted deltas in bit-index order so the checkpoint is
    # deterministic regardless of task completion order. Order doesn't
    # affect correctness (each delta replaces a distinct slot).
    final_preds =
      accepted
      |> Enum.sort_by(fn {bit, _, _, _} -> bit end)
      |> Enum.reduce(preds, fn {bit, _, new_pred, _}, acc ->
        List.replace_at(acc, bit, new_pred)
      end)

    Logger.info(
      "[CegarIter] #{exp.env_name} iter done: #{accepted_in_iter}/#{ctx.n_bits} bits accepted"
    )

    val_seeds = Mujoco.validation_seeds()
    {val_total, val_survived} = Mujoco.validate(final_preds, val_seeds, ctx)
    val_avg = val_total / length(val_seeds)

    # One snapshot push per iter (not per bit): under Jacobi the
    # intermediate per-bit snapshots aren't physically meaningful —
    # only the iter-end policy is. We tag it with the most impactful
    # accepted bit and the full-policy validation reward, which is
    # what the dashboard surfaces.
    if accepted_in_iter > 0 do
      {top_bit, _, _, _} = Enum.max_by(accepted, fn {_, _, _, r} -> r end)
      :ok = push_policy_snapshot(exp, ctx, final_preds, top_bit, cegar_iter, iter, val_avg)
    end

    {:ok, _} = checkpoint(exp.id, final_preds, bit_shuffle)

    Experiments.log_event!(
      "info",
      "master",
      "iter done: #{exp.env_name} CEGAR #{cegar_iter}/#{ctx.cegar_rounds} " <>
        "iter #{iter}/#{ctx.max_iters} avg=#{Float.round(val_avg, 2)} " <>
        "(#{accepted_in_iter}/#{ctx.n_bits} accepted, jacobi)",
      env_name: exp.env_name,
      experiment_id: exp.id,
      metadata: %{
        "cegar_iter" => cegar_iter,
        "iter" => iter,
        "validation_avg" => val_avg,
        "validation_survived" => val_survived,
        "accepted_in_iter" => accepted_in_iter,
        "dispatch_mode" => "jacobi"
      }
    )

    advance_or_complete(exp, ctx, final_preds, val_avg)
  end

  # First-attempt path persists the bit shuffle so any subsequent
  # attempt picks up the SAME ordering. We also reset `bit_progress`
  # at iter start (under Jacobi, per-attempt progress is not carried
  # forward — see module doc).
  defp ensure_bit_shuffle(%Experiment{} = exp, ctx, cegar_iter, iter) do
    case exp.bit_shuffle do
      [] ->
        shuffle = Mujoco.shuffle_bits(ctx.n_bits, shuffle_seed(exp.id, cegar_iter, iter))
        {:ok, _} = Experiments.update_state(exp, %{"bit_shuffle" => shuffle, "bit_progress" => []})
        shuffle

      existing ->
        {:ok, _} = Experiments.update_state(exp, %{"bit_progress" => []})
        existing
    end
  end

  defp run_one_bit_jacobi(exp, ctx, baseline_preds, bit_idx, features, seeds, cegar_iter, iter) do
    outcome =
      try do
        Mujoco.optimize_bit(baseline_preds, bit_idx, features, ctx, seeds)
      rescue
        err ->
          Logger.warning(
            "[CegarIter] bit #{bit_idx} crashed: #{Exception.message(err)}\n" <>
              Exception.format_stacktrace(__STACKTRACE__)
          )

          :no_improvement
      end

    # Bump bit_progress on completion (accepted or not) so the
    # dashboard's `bits_done` counter advances as bits drain. The
    # update is atomic at the DB level so concurrent bit tasks don't
    # clobber each other's appends.
    :ok = append_bit_progress(exp.id, bit_idx)

    case outcome do
      :no_improvement ->
        {bit_idx, false, nil, nil}

      {:improved, new_pred, reward} ->
        # The "preview" policy passed to telemetry is the baseline
        # with this bit alone flipped — accurate for the per-bit
        # telemetry contract; the global iter-end checkpoint applies
        # all accepted deltas together.
        preview = List.replace_at(baseline_preds, bit_idx, new_pred)
        Mujoco.emit_bit_accepted(ctx, cegar_iter, iter, bit_idx, reward, preview)
        :ok = Experiments.record_acceptance(exp, reward)
        {bit_idx, true, new_pred, reward}
    end
  end

  defp append_bit_progress(exp_id, bit_idx) do
    sql = """
    UPDATE experiments
    SET bit_progress =
          CASE
            WHEN $1 = ANY(bit_progress) THEN bit_progress
            ELSE array_append(bit_progress, $1::integer)
          END,
        updated_at = now()
    WHERE id = $2
    """

    Repo.query!(sql, [bit_idx, Ecto.UUID.dump!(exp_id)])
    :ok
  end

  defp push_policy_snapshot(exp, ctx, preds, bit_idx, cegar_iter, iter, reward) do
    code =
      PrettyPrint.to_python(preds,
        bits_per_dim: ctx.bits_per_dim,
        n_action_dims: ctx.n_action_dims,
        action_range: ctx.cfg.action_range,
        action_dim_names: ctx.cfg.action_dim_names
      )

    attrs = %{
      "env_name" => exp.env_name,
      "bit_predicates" => %{"preds" => Enum.map(preds, &PrettyPrint.to_json_term/1)},
      "policy_code" => code,
      "code_language" => "python",
      "n_bits" => ctx.n_bits,
      "target_bit" => bit_idx,
      "cegar_iter" => cegar_iter,
      "iter" => iter,
      "best_reward" => reward,
      "baseline_reward" => exp.baseline_reward
    }

    case Server.Queue.upsert_policy_snapshot(attrs, submitter: exp.submitter) do
      {:ok, _snapshot} ->
        :ok

      {:error, reason} ->
        Logger.warning("[CegarIter] snapshot push failed: #{inspect(reason)}")
        :ok
    end
  end

  defp checkpoint(exp_id, preds, bit_progress) do
    encoded = Enum.map(preds, &PrettyPrint.to_json_term/1)

    sql = """
    UPDATE experiments
    SET predicates = $1, bit_progress = $2, updated_at = now()
    WHERE id = $3
    """

    Repo.query!(sql, [%{"preds" => encoded}, bit_progress, Ecto.UUID.dump!(exp_id)])
    {:ok, :checkpointed}
  end

  defp advance_or_complete(%Experiment{} = exp, ctx, final_preds, _val_avg) do
    {next_cegar, next_iter, done?} =
      cond do
        exp.current_iter < ctx.max_iters ->
          {exp.current_cegar_iter, exp.current_iter + 1, false}

        exp.current_cegar_iter < ctx.cegar_rounds ->
          {exp.current_cegar_iter + 1, 1, false}

        true ->
          {exp.current_cegar_iter, exp.current_iter, true}
      end

    encoded = %{"preds" => Enum.map(final_preds, &PrettyPrint.to_json_term/1)}

    if done? do
      {:ok, _} =
        Experiments.update_state(exp, %{
          "predicates" => encoded,
          "bit_shuffle" => [],
          "bit_progress" => []
        })

      Server.Workers.ExperimentComplete.new(%{"experiment_id" => exp.id})
      |> Oban.insert!()

      :ok
    else
      {:ok, _} =
        Experiments.update_state(exp, %{
          "predicates" => encoded,
          "current_cegar_iter" => next_cegar,
          "current_iter" => next_iter,
          "bit_shuffle" => [],
          "bit_progress" => []
        })

      __MODULE__.new(%{"experiment_id" => exp.id})
      |> Oban.insert!()

      :ok
    end
  end

  # ── Helpers ─────────────────────────────────────────────────

  defp decode_predicates(%{"preds" => list}) when is_list(list),
    do: Enum.map(list, &PrettyPrint.from_json_term/1)

  defp decode_predicates(_), do: []

  defp shuffle_seed(exp_id, cegar_iter, iter) do
    # Deterministic seed per (experiment, cegar_iter, iter) so a
    # crashed worker's retry shuffles the bits IDENTICALLY to the
    # original attempt. Without this, a partial bit_progress would
    # paired with a fresh shuffle would skip the wrong bits.
    :erlang.phash2({exp_id, cegar_iter, iter})
  end

  # Pulse both Oban (so Lifeline doesn't reap a slow iter) and the
  # experiment row (so the dashboard's `heartbeat_seconds_ago` reflects
  # real liveness, not just bit-acceptance events).
  defp start_heartbeat(job_id, exp_id) do
    Task.start_link(fn -> heartbeat_loop(job_id, exp_id) end)
  end

  defp heartbeat_loop(job_id, exp_id) do
    receive do
      :stop -> :ok
    after
      @heartbeat_interval_ms ->
        beat(job_id, exp_id)
        heartbeat_loop(job_id, exp_id)
    end
  end

  defp beat(job_id, exp_id) do
    try do
      from(j in Oban.Job, where: j.id == ^job_id)
      |> Repo.update_all(set: [attempted_at: DateTime.utc_now()])

      from(e in Experiment, where: e.id == ^exp_id)
      |> Repo.update_all(set: [updated_at: DateTime.utc_now()])
    rescue
      _ -> :ok
    end
  end

  defp stop_heartbeat({:ok, pid}) when is_pid(pid), do: Process.exit(pid, :kill)
  defp stop_heartbeat(_), do: :ok
end
