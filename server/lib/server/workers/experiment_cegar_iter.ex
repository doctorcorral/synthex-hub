defmodule Server.Workers.ExperimentCegarIter do
  @moduledoc """
  One CEGAR iteration. Runs all `n_bits` bit searches for a single
  `(cegar_iter, iter)` pair, then enqueues either the next iter or
  `ExperimentComplete`.

  ## Resume on crash

  Every accepted bit triggers an immediate UPDATE of the experiment
  row's `predicates` and `bit_progress`. If this worker crashes or
  is killed:

    * Oban retries (max_attempts = 3, exponential backoff).
    * The retry re-collects states / rebuilds features (cheap
      relative to bit searches), then iterates over the bit shuffle
      MINUS `bit_progress`. Already-accepted bits never re-run, so
      no wasted compute.

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
        "iter #{iter}/#{ctx.max_iters}"
    )

    # collect_states + features are deterministic given (preds, ctx).
    # Cheap relative to the bit searches that follow (~minutes vs
    # ~hours), so we just re-run on every iter attempt instead of
    # persisting potentially-multi-MB feature blobs in jsonb.
    {states, _} = Synthex.Gym.Mujoco.collect_states(preds, ctx)
    features = Synthex.Gym.Mujoco.build_features(states, ctx)

    Logger.info("[CegarIter] #{length(features)} features built")

    # Bit shuffle: persisted on first attempt so a retry reuses the
    # SAME order. Together with `bit_progress`, this means a retry
    # picks up exactly where we left off — no bit ever runs twice.
    bit_shuffle =
      case exp.bit_shuffle do
        [] ->
          shuffle = Synthex.Gym.Mujoco.shuffle_bits(ctx.n_bits, shuffle_seed(exp.id, cegar_iter, iter))
          {:ok, _} = Experiments.update_state(exp, %{"bit_shuffle" => shuffle})
          shuffle

        existing ->
          existing
      end

    seeds = Synthex.Gym.Mujoco.seeds_for(cegar_iter, iter, ctx)
    progress = MapSet.new(exp.bit_progress || [])

    {final_preds, _final_progress, accepted_in_iter} =
      Enum.reduce(bit_shuffle, {preds, progress, 0}, fn bit_idx, {cur_preds, prog, accepted} ->
        if MapSet.member?(prog, bit_idx) do
          {cur_preds, prog, accepted}
        else
          {next_preds, was_accepted} = run_one_bit(exp, ctx, cur_preds, bit_idx, features, seeds, cegar_iter, iter)
          new_prog = MapSet.put(prog, bit_idx)

          {:ok, _} =
            checkpoint(exp.id, next_preds, MapSet.to_list(new_prog))

          {next_preds, new_prog, if(was_accepted, do: accepted + 1, else: accepted)}
        end
      end)

    Logger.info(
      "[CegarIter] #{exp.env_name} iter done: #{accepted_in_iter}/#{ctx.n_bits} bits accepted"
    )

    # Optional per-iter validation. Mirrors the `validate/2` call
    # in `Synthex.Gym.Mujoco.solve/2` so we get the same per-iter
    # log line ("avg=X/ep") in the server logs as the laptop master
    # would have produced.
    val_seeds = Synthex.Gym.Mujoco.validation_seeds()
    {val_total, val_survived} = Synthex.Gym.Mujoco.validate(final_preds, val_seeds, ctx)
    val_avg = val_total / length(val_seeds)

    Experiments.log_event!(
      "info",
      "master",
      "iter done: #{exp.env_name} CEGAR #{cegar_iter}/#{ctx.cegar_rounds} " <>
        "iter #{iter}/#{ctx.max_iters} avg=#{Float.round(val_avg, 2)} " <>
        "(#{accepted_in_iter}/#{ctx.n_bits} accepted)",
      env_name: exp.env_name,
      experiment_id: exp.id,
      metadata: %{
        "cegar_iter" => cegar_iter,
        "iter" => iter,
        "validation_avg" => val_avg,
        "validation_survived" => val_survived,
        "accepted_in_iter" => accepted_in_iter
      }
    )

    advance_or_complete(exp, ctx, final_preds, val_avg)
  end

  defp run_one_bit(exp, ctx, preds, bit_idx, features, seeds, cegar_iter, iter) do
    case Synthex.Gym.Mujoco.optimize_bit(preds, bit_idx, features, ctx, seeds) do
      :no_improvement ->
        {preds, false}

      {:improved, new_pred, reward} ->
        updated = List.replace_at(preds, bit_idx, new_pred)
        Synthex.Gym.Mujoco.emit_bit_accepted(ctx, cegar_iter, iter, bit_idx, reward, updated)

        :ok = push_policy_snapshot(exp, ctx, updated, bit_idx, cegar_iter, iter, reward)
        :ok = Experiments.record_acceptance(exp, reward)

        {updated, true}
    end
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
