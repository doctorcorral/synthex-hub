defmodule Server.Workers.ExperimentController do
  @moduledoc """
  Streaming-CEGAR controller. One long-lived Oban job per running
  experiment that drives the synthesis loop end-to-end. See
  `docs/streaming-cegar.md` §Layer 3 for the algorithm.

  ## Algorithm

  The controller advances one CEGAR-step per `perform/1` invocation
  and self-enqueues for the next step when this one saturates. A
  step is the unit between feature-pool refreshes:

      1. Read current `(predicates, policy_version)` from the
         attached `Server.EnvPolicy` (the lineage).
      2. `collect_states` + `build_features` against current preds.
      3. Dispatch *passes* until saturation:
           - Each pass walks the open bits **sequentially**: for
             every bit, the controller re-reads the env_policy row
             (so it observes the latest `(predicates, policy_version)`
             reflecting any commit from a prior bit in this same
             pass), then dispatches one Synthex batch via
             `Mujoco.optimize_bit`. The result is handed to
             `Server.CommitGate.attempt_commit/1` with
             `evaluated_at_version: V_current` — that bit's own
             freshly-read version, not a stale wave-start snapshot.
             The gate atomically:
                * commits if the reward beats the env_policy's
                  running best by `acceptance_epsilon`, bumping
                  `env_policy.policy_version` and inserting a
                  `policy_versions` audit row,
                * rejects as `:no_improvement` otherwise.
             `:stale` rejections are impossible in the sequential
             path by construction: we always pass the version we
             just read.
           - A bit is "settled" at the current version if it was
             evaluated against that version and produced no
             improvement. Once any later commit advances the
             lineage version, all settled bits become open again.
      4. Saturation = a pass with zero successful commits. At this
         point no further commit is possible against the current
         feature pool.
      5. Validate the final policy on `validation_seeds()`, log
         step-done event, advance `current_cegar_iter`.
      6. If more rounds remain → self-enqueue. Otherwise enqueue
         `ExperimentComplete`.

  ## Source of truth for predicates

  All predicate reads go through `Server.EnvPolicies.for_experiment/1`,
  which joins through the experiment's `env_policy_id` FK. The
  experiment row no longer stores predicates — every accepted commit
  writes through the gate to the env_policy row, and the lineage
  survives session crashes.

  ## Why sequential, not Jacobi parallel

  See git history / `docs/streaming-cegar.md` §Layer 3. The
  short version: parallel dispatch with a wave-start `v_start`
  snapshot wastes ~N-1 evaluations per wave on stale rejections.
  Sequential dispatch evaluates every bit against the current
  policy, so every successful commit raises the bar for the next
  bit and the swarm's compute lands as commits rather than
  discarded results. Operators with envs whose per-bit chunk count
  is smaller than the swarm size can opt back into parallel
  dispatch via `bit_concurrency > 1` in the experiment config; the
  parallel path is preserved exactly for that case.

  ## Retry semantics

  Oban retries the same `perform/1` up to `max_attempts: 3` with
  exponential backoff. On retry the controller re-reads the
  env_policy row, which already reflects every commit accepted so
  far (`policy_version`, `predicates`). The step is therefore
  idempotent across retries — a crashed step at version `V` resumes
  by re-collecting states under `V` and dispatching open bits, with
  the gate's version check ensuring no stale work slips in.

  After three exhausted attempts `Server.ObanFailureHandler` marks
  the experiment as `failed` and records an `error`-level
  `system_event`. The env_policy persists — the next submission for
  the same `(env, sig)` inherits the lineage automatically.
  """

  use Oban.Worker,
    queue: :master,
    max_attempts: 3,
    unique: [
      keys: [:experiment_id],
      states: [:available, :scheduled, :executing, :retryable],
      period: :infinity
    ]

  require Logger
  import Ecto.Query

  alias Server.{CommitGate, EnvPolicies, EnvPolicy, Experiment, Experiments,
                Queue, Repo}
  alias Server.Workers.{ExperimentBootstrap, ExperimentComplete}
  alias Synthex.Core.PrettyPrint
  alias Synthex.Gym.Mujoco

  @heartbeat_interval_ms 60_000

  # Defensive cap on waves per step. Saturation usually happens in
  # 1–2 waves (most bits report no_improvement); a much higher cap
  # would mean every-bit-interacts-with-every-bit, which by §11.1's
  # near-independence assumption shouldn't happen in practice.
  @max_waves_per_step 6

  # Cap on how many bits are in-flight at once. Default 1 (sequential
  # evaluation; see module doc). Workers fair-share chunks across
  # whatever batches happen to be in flight, so swarm throughput is
  # invariant to this knob as long as one bit's chunk count saturates
  # the swarm. Operators with envs whose per-bit chunk count is
  # smaller than the swarm size can raise `bit_concurrency` in the
  # config; the parallel path falls back to the wave-snapshot
  # version for the commit gate.
  @default_bit_concurrency 1

  @impl Oban.Worker
  def perform(%Oban.Job{id: job_id, args: %{"experiment_id" => exp_id} = args}) do
    case Experiments.get(exp_id) do
      {:error, :not_found} ->
        {:discard, "experiment not found"}

      {:ok, %Experiment{status: status}} when status != "running" ->
        Logger.info("[Controller] experiment #{exp_id} is #{status}; exiting")
        :ok

      {:ok, %Experiment{} = exp} ->
        heartbeat = start_heartbeat(job_id, exp.id)

        try do
          # Cancel any in-flight batches left behind by a previous
          # attempt of this controller job. Without this sweep the
          # swarm's fair-share `claim_chunk` keeps feeding the dead
          # batches to workers indefinitely — nothing is polling
          # their results, so every completed chunk is wasted CPU.
          sweep_orphan_batches!(exp.id)

          run_step(exp, args)
        after
          stop_heartbeat(heartbeat)
        end
    end
  end

  defp sweep_orphan_batches!(exp_id) do
    case Queue.cancel_experiment_in_flight_batches(exp_id) do
      {:ok, %{batches: 0}} ->
        :ok

      {:ok, %{batches: n_batches, jobs: n_jobs}} ->
        Logger.info(
          "[Controller] swept #{n_batches} orphan in-flight batches " <>
            "(#{n_jobs} pending chunk jobs cancelled) on perform/1 entry"
        )

        Experiments.log_event!(
          "warn",
          "master",
          "controller resumed after crash/deploy; " <>
            "cancelled #{n_batches} orphan batch(es) from prior attempt " <>
            "(#{n_jobs} pending chunks freed)",
          experiment_id: exp_id,
          metadata: %{"orphan_batches" => n_batches, "orphan_chunks" => n_jobs}
        )

        :ok

      {:error, reason} ->
        Logger.warning("[Controller] orphan sweep failed: #{inspect(reason)}")
        :ok
    end
  end

  # ── One CEGAR step ──────────────────────────────────────────

  defp run_step(%Experiment{} = exp, _args) do
    env_key = ExperimentBootstrap.decode_env_key(exp.env_key)
    ctx = ExperimentBootstrap.build_context(env_key, exp.config, exp.id)
    cegar_iter = exp.current_cegar_iter
    epsilon = acceptance_epsilon(exp.config)
    bit_concurrency = bit_concurrency(exp.config)

    mode = if bit_concurrency <= 1, do: "sequential", else: "parallel"

    {:ok, env_policy_start} = EnvPolicies.for_experiment(exp)

    Logger.info(
      "[Controller] #{exp.env_name} step #{cegar_iter}/#{ctx.cegar_rounds} " <>
        "(#{mode}, ε=#{epsilon}, bit_concurrency=#{bit_concurrency}, " <>
        "lineage v=#{env_policy_start.policy_version})"
    )

    # collect_states + features against the CURRENT predicates. We
    # do this at every step start because the previous step's
    # commits changed the policy, so trajectories differ. The cost
    # is one big batch up front; the savings come from subsequent
    # waves.
    preds_at_start = decode_predicates(env_policy_start.predicates)
    {states, _} = Mujoco.collect_states(preds_at_start, ctx)
    features = Mujoco.build_features(states, ctx)
    Logger.info("[Controller] #{length(features)} features built")

    seeds = Mujoco.seeds_for(cegar_iter, 1, ctx)
    initial_state = %{committed: MapSet.new(), settled_at: %{}}

    final_state =
      Enum.reduce_while(
        1..@max_waves_per_step,
        initial_state,
        fn wave_num, acc ->
          case dispatch_wave(
                 exp.id,
                 ctx,
                 features,
                 seeds,
                 cegar_iter,
                 epsilon,
                 bit_concurrency,
                 wave_num,
                 acc
               ) do
            {:saturated, next_acc} -> {:halt, next_acc}
            {:continue, next_acc} -> {:cont, next_acc}
            {:experiment_finished, next_acc} -> {:halt, next_acc}
          end
        end
      )

    # Re-read the env_policy so we have the latest predicates
    # (after all wave commits) for validation + the step-done log.
    {:ok, after_step} = Experiments.get(exp.id)
    {:ok, env_policy_after} = EnvPolicies.for_experiment(after_step)

    final_preds = decode_predicates(env_policy_after.predicates)
    accepted_in_step = MapSet.size(final_state.committed)

    val_seeds = Mujoco.validation_seeds()
    {val_total, val_survived} = Mujoco.validate(final_preds, val_seeds, ctx)
    val_avg = val_total / length(val_seeds)

    Logger.info(
      "[Controller] #{exp.env_name} step done: " <>
        "#{accepted_in_step} commits, v=#{env_policy_after.policy_version}, " <>
        "validation avg=#{Float.round(val_avg, 2)}"
    )

    Experiments.log_event!(
      "info",
      "master",
      "step done: #{exp.env_name} step #{cegar_iter}/#{ctx.cegar_rounds} " <>
        "v=#{env_policy_after.policy_version} avg=#{Float.round(val_avg, 2)} " <>
        "(#{accepted_in_step} commits)",
      env_name: exp.env_name,
      experiment_id: exp.id,
      metadata: %{
        "cegar_iter" => cegar_iter,
        "policy_version" => env_policy_after.policy_version,
        "validation_avg" => val_avg,
        "validation_survived" => val_survived,
        "accepted_in_step" => accepted_in_step,
        "dispatch_mode" =>
          if(bit_concurrency(exp.config) <= 1, do: "sequential", else: "parallel")
      }
    )

    advance_or_complete(after_step, ctx, val_avg)
  end

  # ── One pass over the open bits ─────────────────────────────

  defp dispatch_wave(exp_id, ctx, features, seeds, cegar_iter, epsilon, concurrency, wave_num, acc) do
    {:ok, exp_before} = Experiments.get(exp_id)

    cond do
      exp_before.status != "running" ->
        Logger.info(
          "[Controller] experiment #{exp_id} status=#{exp_before.status}; aborting waves"
        )

        {:experiment_finished, acc}

      true ->
        {:ok, env_policy_before} = EnvPolicies.for_experiment(exp_before)
        v_start = env_policy_before.policy_version
        open = open_bits(ctx.n_bits, v_start, acc)

        case open do
          [] ->
            Logger.info("[Controller] pass #{wave_num}: no open bits (saturated)")
            {:saturated, acc}

          bits ->
            effective_conc = min(length(bits), concurrency)

            if effective_conc <= 1 do
              run_pass_sequential(
                bits,
                exp_id,
                ctx,
                features,
                seeds,
                cegar_iter,
                epsilon,
                wave_num,
                acc
              )
            else
              run_pass_parallel(
                bits,
                exp_id,
                ctx,
                features,
                seeds,
                cegar_iter,
                epsilon,
                effective_conc,
                env_policy_before,
                wave_num,
                acc
              )
            end
        end
    end
  end

  # Sequential pass: walk bits one at a time, re-read the env_policy
  # row before each bit so we evaluate against the latest policy
  # version. This is the path that makes the strict-monotonicity
  # commit gate cheap — every result we feed it is by-construction
  # version-fresh, so no evaluation is wasted as `:stale`.
  defp run_pass_sequential(bits, exp_id, ctx, features, seeds, cegar_iter, epsilon, wave_num, acc) do
    {:ok, exp_before} = Experiments.get(exp_id)
    {:ok, env_policy_at_start} = EnvPolicies.for_experiment(exp_before)

    Logger.info(
      "[Controller] pass #{wave_num}: walking #{length(bits)} bits sequentially at v=#{env_policy_at_start.policy_version}"
    )

    {{n_commit, n_stale, n_no_imp, n_finished}, acc1} =
      Enum.reduce(bits, {{0, 0, 0, 0}, acc}, fn bit_idx, {{nc, ns, ni, nf}, a} ->
        if nf > 0 do
          {{nc, ns, ni, nf}, a}
        else
          evaluate_one_bit_sequential(
            bit_idx,
            {nc, ns, ni, nf},
            a,
            exp_id,
            ctx,
            features,
            seeds,
            cegar_iter,
            epsilon,
            wave_num
          )
        end
      end)

    Logger.info(
      "[Controller] pass #{wave_num} done: " <>
        "#{n_commit} commits, #{n_stale} stale, #{n_no_imp} no-improvement"
    )

    cond do
      n_finished > 0 -> {:experiment_finished, acc1}
      n_commit == 0 -> {:saturated, acc1}
      true -> {:continue, acc1}
    end
  end

  defp evaluate_one_bit_sequential(
         bit_idx,
         {nc, ns, ni, nf},
         a,
         exp_id,
         ctx,
         features,
         seeds,
         cegar_iter,
         epsilon,
         wave_num
       ) do
    case with_db_retry(fn -> Experiments.get(exp_id) end) do
      {:error, :not_found} ->
        Logger.info("[Controller] bit #{bit_idx}: experiment gone, halting pass")
        {{nc, ns, ni, nf + 1}, a}

      {:ok, %Experiment{status: status}} when status != "running" ->
        Logger.info(
          "[Controller] bit #{bit_idx}: experiment status=#{status}, halting pass"
        )

        {{nc, ns, ni, nf + 1}, a}

      {:ok, %Experiment{} = exp_now} ->
        {:ok, env_policy_now} = with_db_retry(fn -> EnvPolicies.for_experiment(exp_now) end)
        v_current = env_policy_now.policy_version
        preds_now = decode_predicates(env_policy_now.predicates)

        case evaluate_bit(preds_now, bit_idx, features, ctx, seeds) do
          :no_improvement ->
            {{nc, ns, ni + 1, nf},
             %{a | settled_at: Map.put(a.settled_at, bit_idx, v_current)}}

          {:improved, candidate, reward} ->
            apply_commit(
              {bit_idx, candidate, reward},
              {nc, ns, ni, nf},
              a,
              exp_id,
              v_current,
              cegar_iter,
              ctx,
              epsilon,
              wave_num
            )
        end
    end
  end

  # Parallel path — retained for envs where `bit_concurrency > 1`
  # is configured. Same wave-snapshot behavior as before the
  # env_policy refactor; the snapshot is now of the lineage row, not
  # the experiment row, but otherwise unchanged.
  defp run_pass_parallel(
         bits,
         exp_id,
         ctx,
         features,
         seeds,
         cegar_iter,
         epsilon,
         concurrency,
         %EnvPolicy{} = env_policy_at_start,
         wave_num,
         acc
       ) do
    v_start = env_policy_at_start.policy_version
    preds = decode_predicates(env_policy_at_start.predicates)

    Logger.info(
      "[Controller] pass #{wave_num}: dispatching #{length(bits)} bits in parallel at v=#{v_start} " <>
        "(in-flight cap=#{concurrency})"
    )

    outcomes =
      bits
      |> Task.async_stream(
        fn bit_idx ->
          {bit_idx, evaluate_bit(preds, bit_idx, features, ctx, seeds)}
        end,
        max_concurrency: concurrency,
        timeout: :infinity,
        ordered: false
      )
      |> Enum.flat_map(fn
        {:ok, result} ->
          [result]

        {:exit, reason} ->
          Logger.warning("[Controller] bit task exited: #{inspect(reason)}")
          []
      end)

    handle_outcomes(outcomes, exp_id, v_start, cegar_iter, ctx, epsilon, wave_num, acc)
  end

  defp evaluate_bit(preds, bit_idx, features, ctx, seeds) do
    try do
      Mujoco.optimize_bit(preds, bit_idx, features, ctx, seeds)
    rescue
      err ->
        Logger.warning(
          "[Controller] bit #{bit_idx} crashed: #{Exception.message(err)}\n" <>
            Exception.format_stacktrace(__STACKTRACE__)
        )

        :no_improvement
    end
  end

  defp handle_outcomes(outcomes, exp_id, v_start, cegar_iter, ctx, epsilon, wave_num, acc) do
    {{n_commit, n_stale, n_no_imp, n_finished}, acc1} =
      Enum.reduce(outcomes, {{0, 0, 0, 0}, acc}, fn {bit_idx, outcome}, {{nc, ns, ni, nf}, a} ->
        case outcome do
          :no_improvement ->
            {{nc, ns, ni + 1, nf}, %{a | settled_at: Map.put(a.settled_at, bit_idx, v_start)}}

          {:improved, candidate, reward} ->
            apply_commit(
              {bit_idx, candidate, reward},
              {nc, ns, ni, nf},
              a,
              exp_id,
              v_start,
              cegar_iter,
              ctx,
              epsilon,
              wave_num
            )
        end
      end)

    Logger.info(
      "[Controller] pass #{wave_num} done (parallel): " <>
        "#{n_commit} commits, #{n_stale} stale, #{n_no_imp} no-improvement"
    )

    cond do
      n_finished > 0 -> {:experiment_finished, acc1}
      n_commit == 0 and n_stale == 0 -> {:saturated, acc1}
      true -> {:continue, acc1}
    end
  end

  defp apply_commit(
         {bit_idx, candidate, reward},
         {nc, ns, ni, nf},
         a,
         exp_id,
         v_start,
         cegar_iter,
         ctx,
         epsilon,
         wave_num
       ) do
    attempt =
      CommitGate.attempt_commit(
        experiment_id: exp_id,
        bit_idx: bit_idx,
        candidate_term: PrettyPrint.to_json_term(candidate),
        reward: reward,
        evaluated_at_version: v_start,
        acceptance_epsilon: epsilon,
        metadata: %{
          "cegar_iter" => cegar_iter,
          "wave" => wave_num
        }
      )

    case attempt do
      {:committed, new_version, _env_policy, _exp} ->
        :ok =
          push_policy_snapshot(exp_id, ctx, bit_idx, cegar_iter, new_version, reward)

        {{nc + 1, ns, ni, nf}, %{a | committed: MapSet.put(a.committed, bit_idx)}}

      {:rejected, :stale} ->
        {{nc, ns + 1, ni, nf}, %{a | settled_at: Map.delete(a.settled_at, bit_idx)}}

      {:rejected, :no_improvement} ->
        {{nc, ns, ni + 1, nf}, %{a | settled_at: Map.put(a.settled_at, bit_idx, v_start)}}

      {:rejected, :experiment_not_running} ->
        {{nc, ns, ni, nf + 1}, a}

      {:error, reason} ->
        Logger.warning(
          "[Controller] commit gate error for bit #{bit_idx}: #{inspect(reason)}"
        )

        {{nc, ns, ni, nf}, a}
    end
  end

  defp open_bits(n_bits, v_current, %{committed: committed, settled_at: settled_at}) do
    Enum.reject(0..(n_bits - 1), fn bit ->
      MapSet.member?(committed, bit) or Map.get(settled_at, bit) == v_current
    end)
  end

  # ── Step end ────────────────────────────────────────────────

  defp advance_or_complete(%Experiment{} = exp, ctx, _val_avg) do
    {next_cegar, done?} =
      if exp.current_cegar_iter < ctx.cegar_rounds do
        {exp.current_cegar_iter + 1, false}
      else
        {exp.current_cegar_iter, true}
      end

    if done? do
      ExperimentComplete.new(%{"experiment_id" => exp.id})
      |> Oban.insert!()

      :ok
    else
      {:ok, _} =
        Experiments.update_state(exp, %{
          "current_cegar_iter" => next_cegar,
          "current_iter" => 1
        })

      enqueue_next_step!(exp.id)
      :ok
    end
  end

  # Self-enqueue MUST bypass `unique:` because our own job row is
  # still in `:executing` state at the moment we insert the
  # successor — Oban's unique check matches it and silently returns
  # *us* instead of inserting a new job. The current `perform/1`
  # then exits normally, the job transitions to `:completed`, and
  # there's no successor left to run. The experiment hangs at
  # status=running with no controller, exactly the "no checkpoint
  # for Nm — controller Oban job may have crashed" stall pattern we
  # hit on session 0a2db7c5.
  #
  # Disabling uniqueness only on this specific insert is safe:
  #   * Bootstrap's initial controller enqueue still uses `unique:`
  #     (no prior controller exists when it fires, so it inserts
  #     cleanly).
  #   * Manual/operator re-enqueues still use `unique:` and dedupe
  #     against this controller or its scheduled successor.
  #   * Two controllers can't actually run concurrently for the same
  #     experiment: by the time the new job becomes `:available`,
  #     this `perform/1` has returned and our job is `:completed`.
  defp enqueue_next_step!(exp_id) do
    __MODULE__.new(%{"experiment_id" => exp_id}, unique: false)
    |> Oban.insert!()
  end

  # ── DB helpers ──────────────────────────────────────────────

  defp push_policy_snapshot(exp_id, ctx, bit_idx, cegar_iter, version, reward) do
    {:ok, exp} = Experiments.get(exp_id)
    {:ok, env_policy} = EnvPolicies.for_experiment(exp)
    preds = decode_predicates(env_policy.predicates)

    code =
      PrettyPrint.to_python(preds,
        bits_per_dim: ctx.bits_per_dim,
        n_action_dims: ctx.n_action_dims,
        action_range: ctx.cfg.action_range,
        action_dim_names: ctx.cfg.action_dim_names
      )

    attrs = %{
      "env_policy_id" => env_policy.id,
      "env_name" => exp.env_name,
      "bit_predicates" => %{"preds" => Enum.map(preds, &PrettyPrint.to_json_term/1)},
      "policy_code" => code,
      "code_language" => "python",
      "n_bits" => ctx.n_bits,
      "target_bit" => bit_idx,
      "cegar_iter" => cegar_iter,
      "iter" => version,
      "best_reward" => reward,
      "baseline_reward" => env_policy.baseline_reward
    }

    case Server.Queue.upsert_policy_snapshot(attrs, submitter: exp.submitter) do
      {:ok, _snapshot} ->
        :ok

      {:error, reason} ->
        Logger.warning("[Controller] snapshot push failed: #{inspect(reason)}")
        :ok
    end
  end

  defp decode_predicates(%{"preds" => list}) when is_list(list),
    do: Enum.map(list, &PrettyPrint.from_json_term/1)

  defp decode_predicates(_), do: []

  # Epsilon defaults to 0.0 (any strict improvement counts). Operators
  # can raise it via the experiment config to filter rollout noise on
  # high-variance envs.
  defp acceptance_epsilon(%{"acceptance_epsilon" => v}) when is_number(v), do: v * 1.0
  defp acceptance_epsilon(_), do: 0.0

  # Transient pool blips (DBConnection.ConnectionError "request was
  # dropped from queue") shouldn't burn through Oban's 3-attempt
  # budget — the bit loop fires hundreds of small reads per CEGAR
  # pass, and even a brief saturation during a wave dispatch
  # surge can crash the whole controller mid-stride. Retry the
  # read up to 4 times with exponential backoff (~250ms → 4s
  # cumulative) before propagating; that's enough to ride out
  # the queue_interval window without hard-crashing.
  #
  # Only wrap small idempotent reads. Repo.transaction blocks
  # MUST NOT be retried this way — they hold a checked-out
  # connection across the retry, defeating the purpose.
  @retry_backoffs_ms [250, 500, 1500, 3000]

  defp with_db_retry(fun) do
    with_db_retry(fun, @retry_backoffs_ms)
  end

  defp with_db_retry(fun, []) do
    fun.()
  end

  defp with_db_retry(fun, [backoff | rest]) do
    try do
      fun.()
    rescue
      err in DBConnection.ConnectionError ->
        Logger.warning(
          "[Controller] transient DB pool error, retrying in #{backoff}ms " <>
            "(#{length(rest)} more attempts): #{Exception.message(err)}"
        )

        Process.sleep(backoff)
        with_db_retry(fun, rest)
    end
  end

  defp bit_concurrency(%{"bit_concurrency" => v}) when is_integer(v) and v > 0, do: v
  defp bit_concurrency(_), do: @default_bit_concurrency

  # ── Heartbeat ───────────────────────────────────────────────

  # The heartbeat is a long-lived `receive` loop that bumps
  # `oban_jobs.attempted_at` + `experiments.updated_at` every minute
  # so Oban Lifeline and `Server.Experiments.compute_health/2` can
  # tell a live controller apart from an orphaned one. We start it
  # *linked* so a crashed heartbeat takes the controller down (the
  # alternative — silent heartbeat death with the controller
  # plowing on — would let the dashboard mislabel the run as
  # `stalled` and Lifeline rescue a still-alive job). The matching
  # `stop_heartbeat/1` must therefore terminate it WITHOUT relying
  # on a kill signal that would propagate back through the link.
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

  # Unlink first (so a stray late exit can't take us down), then
  # send `:stop` which the loop's `receive` handles cleanly. Was
  # previously `Process.exit(pid, :kill)`, which propagated `:killed`
  # back through the link and crashed the controller right after every
  # successful step (the c9c415a1 family of failures).
  defp stop_heartbeat({:ok, pid}) when is_pid(pid) do
    Process.unlink(pid)
    send(pid, :stop)
    :ok
  end

  defp stop_heartbeat(_), do: :ok
end
