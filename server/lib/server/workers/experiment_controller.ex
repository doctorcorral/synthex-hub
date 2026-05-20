defmodule Server.Workers.ExperimentController do
  @moduledoc """
  Streaming-CEGAR controller. One long-lived Oban job per running
  experiment that drives the synthesis loop end-to-end. Replaces
  `Server.Workers.ExperimentCegarIter` (Layer 2's Jacobi-iter
  master) with a continuous, version-gated commit pipeline
  (`docs/streaming-cegar.md` §Layer 3).

  ## Algorithm

  The controller advances one CEGAR-step per `perform/1` invocation
  and self-enqueues for the next step when this one saturates. A
  step is the unit between feature-pool refreshes:

      1. Read current `(predicates, policy_version)` from the
         experiment row.
      2. `collect_states` + `build_features` against current preds.
         Reset `best_reward_per_bit` to `{}` (per-step pool).
      3. Dispatch *passes* until saturation:
           - Each pass walks the open bits **sequentially**: for
             every bit, the controller re-reads the experiment row
             (so it observes the latest `(predicates, policy_version)`
             reflecting any commit from a prior bit in this same
             pass), then dispatches one Synthex batch via
             `Mujoco.optimize_bit`. The result is handed to
             `Server.CommitGate.attempt_commit/1` with
             `evaluated_at_version: V_current` — that bit's own
             freshly-read version, not a stale wave-start snapshot.
             The gate atomically:
                * commits if the reward beats the running best by
                  `acceptance_epsilon`, bumping `policy_version` and
                  inserting a `policy_versions` audit row,
                * rejects as `:no_improvement` otherwise.
             `:stale` rejections are now impossible by construction:
             we always pass the version we just read.
           - "Still-open" bit definition: a bit is considered
             settled-at-this-version if it was evaluated against the
             current `V` *and* its candidate didn't clear the gate.
             Once any later commit advances `V`, all settled bits
             become open again — their no_improvement verdict at the
             old `V` is no longer authoritative because the policy
             has changed.
      4. Saturation = a pass with zero successful commits. At this
         point no further commit is possible against the current
         feature pool.
      5. Validate the final policy on `validation_seeds()`, log
         iter-done event, advance `current_cegar_iter`.
      6. If more rounds remain → self-enqueue. Otherwise enqueue
         `ExperimentComplete`.

  ## Why sequential, not Jacobi parallel

  The earlier design dispatched all open bits in parallel via
  `Task.async_stream`, evaluating every one against the same frozen
  `v_start = exp.policy_version` of the wave. That looked like
  swarm-friendly parallelism but produced a pathology: at end-of-wave
  `handle_outcomes` walked the outcomes serially and called the
  commit gate once per `:improved` entry; the **first** call bumped
  `policy_version` to `v_start + 1`, and every subsequent call in
  the same wave was rejected as `:stale` (correctly — those
  evaluations were against the old baseline, and applying multiple
  independently-measured improvements would violate strict
  monotonicity). Net effect: at most one commit per wave, with
  `N - 1` evaluations discarded as sunk cost on each wave. For an
  18-bit experiment on HalfCheetah, ~94 % of swarm-side compute was
  thrown away.

  Sequentializing fixes this without giving up correctness or swarm
  utilization. Swarm throughput is bounded by `chunks_per_min`, not
  by how many master-side bit tasks are in flight: any single
  in-flight batch with > workers chunks already saturates the swarm
  (HalfCheetah's per-bit batches have ~1320 chunks vs. swarm size in
  single digits; Ant's are ~220 k). So we lose nothing by holding
  one batch at a time, and we gain that every bit is evaluated
  against the **actually-current** policy — every successful commit
  raises the bar for the next bit, and stale rejections vanish
  entirely. Best case (near-independent bits, the §11.1 assumption
  CSHRLSynthesis makes explicit): the pass commits up to N bits in
  N · per_bit_chunks / C wall-clock. Worst case (highly
  interdependent bits): same N² scaling the old Jacobi design had
  — never worse.

  Operators with envs whose per-bit chunk count is smaller than the
  swarm size can still opt back into parallel dispatch via a
  `bit_concurrency > 1` setting in the experiment config; the
  parallel path is preserved exactly for that case.

  ## Why one step per Oban job

    * Per-step Lifeline / retry / timeout accounting is uniform
      across all environments. A `Humanoid-v5` step's wall-clock is
      bounded by `max_waves × max_bit_search_time`, which is
      predictable.
    * Heartbeats span the step's lifetime; crash recovery on retry
      just re-runs the current step from scratch (states will be
      slightly different due to rollout stochasticity, but the
      version-gated commit pipeline guarantees no regression).
    * Dashboards can report progress per step without scraping
      a long-lived process state.

  ## Retry semantics

  Oban retries the same `perform/1` up to `max_attempts: 3` with
  exponential backoff. On retry the controller re-reads the
  experiment row, which already reflects every commit accepted so
  far (`policy_version`, `predicates`). The step is therefore
  idempotent across retries — a crashed step at version `V` resumes
  by re-collecting states under `V` and dispatching open bits, with
  the gate's version check ensuring no stale work slips in.

  After three exhausted attempts `Server.ObanFailureHandler` marks
  the experiment as `failed` and records an `error`-level
  `system_event` so the landing page picks up the incident.
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

  alias Server.{CommitGate, Experiment, Experiments, Queue, Repo}
  alias Server.Workers.{ExperimentBootstrap, ExperimentComplete}
  alias Synthex.Core.PrettyPrint
  alias Synthex.Gym.Mujoco

  @heartbeat_interval_ms 60_000

  # Defensive cap on waves per step. Saturation usually happens in
  # 1–2 waves (most bits report no_improvement); a much higher cap
  # would mean every-bit-interacts-with-every-bit, which by §11.1's
  # near-independence assumption shouldn't happen in practice. Acts
  # as a circuit-breaker if the cap is hit, the controller advances
  # the CEGAR step rather than spinning.
  @max_waves_per_step 6

  # Cap on how many bits are in-flight at once. Default is 1
  # (sequential evaluation; see the module doc's "Why sequential,
  # not Jacobi parallel" section). Workers fair-share chunks across
  # whatever batches happen to be in flight, so swarm throughput is
  # invariant to this knob *as long as* the in-flight batches'
  # combined chunk count saturates the swarm — easy for any
  # realistic env where one bit's batch alone is hundreds of chunks
  # vs. a single-digit-to-low-hundreds worker swarm.
  #
  # Operators with envs whose per-bit chunk count is small enough
  # that one bit can't keep the swarm busy (toy envs, very large
  # swarms) can raise `bit_concurrency` in the experiment config.
  # The parallel path falls back to the old wave-snapshot version
  # for the commit gate, which means it still has the stale-tail
  # cost — so prefer sequential unless you have a concrete reason.
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
          # attempt of this controller job (Lifeline rescue, BEAM
          # restart, deploy, OOM-kill, …). Without this sweep the
          # swarm's fair-share `claim_chunk` keeps feeding the dead
          # batches to workers indefinitely — nothing is polling
          # their results, so every completed chunk is wasted CPU.
          #
          # Safe because:
          #   * Oban's `unique: [keys: [:experiment_id]]` guarantees
          #     no concurrent `ExperimentController` instance exists
          #     for the same experiment, so every in-flight batch
          #     at perform/1 entry is necessarily from a prior dead
          #     attempt.
          #   * If perform/1 returns normally, all its batches are
          #     already in a terminal state, so the next attempt's
          #     sweep is a no-op.
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

    Logger.info(
      "[Controller] #{exp.env_name} step #{cegar_iter}/#{ctx.cegar_rounds} " <>
        "(#{mode}, ε=#{epsilon}, bit_concurrency=#{bit_concurrency})"
    )

    # collect_states + features against the CURRENT predicates. We
    # do this at every step start because the previous step's
    # commits changed the policy, so trajectories differ. The
    # cost is one big batch up front; the savings come from the
    # subsequent waves.
    preds_at_start = decode_predicates(exp.predicates)
    {states, _} = Mujoco.collect_states(preds_at_start, ctx)
    features = Mujoco.build_features(states, ctx)
    Logger.info("[Controller] #{length(features)} features built")

    # Per-step pool snapshot — `best_reward_per_bit` is a passive
    # dashboard aggregate, reset at step start because the feature
    # pool changed.
    :ok = reset_pool_snapshot(exp.id)

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

    # Re-read the experiment so we have the latest predicates
    # (after all wave commits) for validation + the iter-done log.
    {:ok, after_step} = Experiments.get(exp.id)
    final_preds = decode_predicates(after_step.predicates)
    accepted_in_step = MapSet.size(final_state.committed)

    val_seeds = Mujoco.validation_seeds()
    {val_total, val_survived} = Mujoco.validate(final_preds, val_seeds, ctx)
    val_avg = val_total / length(val_seeds)

    Logger.info(
      "[Controller] #{exp.env_name} step done: " <>
        "#{accepted_in_step} commits, v=#{after_step.policy_version}, " <>
        "validation avg=#{Float.round(val_avg, 2)}"
    )

    Experiments.log_event!(
      "info",
      "master",
      "step done: #{exp.env_name} step #{cegar_iter}/#{ctx.cegar_rounds} " <>
        "v=#{after_step.policy_version} avg=#{Float.round(val_avg, 2)} " <>
        "(#{accepted_in_step} commits)",
      env_name: exp.env_name,
      experiment_id: exp.id,
      metadata: %{
        "cegar_iter" => cegar_iter,
        "policy_version" => after_step.policy_version,
        "validation_avg" => val_avg,
        "validation_survived" => val_survived,
        "accepted_in_step" => accepted_in_step,
        # `dispatch_mode` records the controller's bit-evaluation
        # strategy so audit replays can tell sequential commits
        # (every bit against `v_current`) apart from the legacy
        # parallel commits (all bits against `v_start`).
        "dispatch_mode" =>
          if(bit_concurrency(exp.config) <= 1, do: "sequential", else: "parallel")
      }
    )

    advance_or_complete(after_step, ctx, val_avg)
  end

  # ── One pass over the open bits ─────────────────────────────
  #
  # Despite the legacy name `dispatch_wave`, this is a single
  # *pass* over the open bits. With `bit_concurrency = 1` (the
  # default), the pass walks bits sequentially and the commit gate
  # observes a fresh `v_current` for each one — see the module
  # doc's "Why sequential, not Jacobi parallel" section. The `wave`
  # tag survives in log lines and `system_event` metadata for
  # continuity with the existing audit trail.

  defp dispatch_wave(exp_id, ctx, features, seeds, cegar_iter, epsilon, concurrency, wave_num, acc) do
    {:ok, exp_before} = Experiments.get(exp_id)

    cond do
      exp_before.status != "running" ->
        Logger.info(
          "[Controller] experiment #{exp_id} status=#{exp_before.status}; aborting waves"
        )

        {:experiment_finished, acc}

      true ->
        v_start = exp_before.policy_version
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
                v_start,
                wave_num,
                acc
              )
            end
        end
    end
  end

  # Sequential pass: walk bits one at a time, re-read the experiment
  # row before each bit so we evaluate against the latest policy
  # version. This is the path that makes the strict-monotonicity
  # commit gate cheap — every result we feed it is by-construction
  # version-fresh, so no evaluation is wasted as `:stale`.
  defp run_pass_sequential(bits, exp_id, ctx, features, seeds, cegar_iter, epsilon, wave_num, acc) do
    {:ok, exp_at_start} = Experiments.get(exp_id)
    v_start = exp_at_start.policy_version

    Logger.info(
      "[Controller] pass #{wave_num}: walking #{length(bits)} bits sequentially at v=#{v_start}"
    )

    {{n_commit, n_stale, n_no_imp, n_finished}, acc1} =
      Enum.reduce(bits, {{0, 0, 0, 0}, acc}, fn bit_idx, {{nc, ns, ni, nf}, a} ->
        # Short-circuit cheaply if a prior bit's commit-gate path
        # observed that the experiment moved out of `running` (e.g.
        # an operator cancellation landed mid-pass). Avoids issuing
        # more rollouts to a halted experiment.
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
      n_finished > 0 ->
        {:experiment_finished, acc1}

      n_commit == 0 ->
        # No commits this pass — every still-open bit was evaluated
        # against its own freshly-read version and produced nothing
        # better. The step has saturated against this feature pool.
        # (Note: with the sequential path `n_stale` is always 0;
        # we keep it in the tuple to stay structurally compatible
        # with the parallel path's `handle_outcomes/8`.)
        {:saturated, acc1}

      true ->
        {:continue, acc1}
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
    case Experiments.get(exp_id) do
      {:error, :not_found} ->
        Logger.info("[Controller] bit #{bit_idx}: experiment gone, halting pass")
        {{nc, ns, ni, nf + 1}, a}

      {:ok, %Experiment{status: status}} when status != "running" ->
        Logger.info(
          "[Controller] bit #{bit_idx}: experiment status=#{status}, halting pass"
        )

        {{nc, ns, ni, nf + 1}, a}

      {:ok, %Experiment{} = exp_now} ->
        v_current = exp_now.policy_version
        preds_now = decode_predicates(exp_now.predicates)

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
  # is configured. Same behavior as before the sequential rewrite:
  # one wave-start snapshot of `v_start` for all bits, end-of-pass
  # serial walk of outcomes through the commit gate, stale-tail
  # cost included. Prefer the sequential path unless you have a
  # tiny-env reason to keep multiple bits in flight.
  defp run_pass_parallel(
         bits,
         exp_id,
         ctx,
         features,
         seeds,
         cegar_iter,
         epsilon,
         concurrency,
         v_start,
         wave_num,
         acc
       ) do
    {:ok, exp_before} = Experiments.get(exp_id)
    preds = decode_predicates(exp_before.predicates)

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

  # `evaluate_bit/5` is a thin wrapper around `optimize_bit` that
  # returns the same shape regardless of outcome — keeps the
  # commit-gate dispatch logic uniform.
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

  # Walks the per-bit outcomes, calling the commit gate for each
  # improved candidate. Threads the running accumulator (committed +
  # settled_at) through the reduce so each gate response can update
  # state immutably. Returns `{:saturated | :continue |
  # :experiment_finished, new_acc}` for the wave loop.
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
      n_finished > 0 ->
        {:experiment_finished, acc1}

      n_commit == 0 and n_stale == 0 ->
        # No commits this pass AND no stale rejections → every open
        # bit was evaluated against the current version and produced
        # nothing better. The step has saturated against this
        # feature pool.
        {:saturated, acc1}

      true ->
        {:continue, acc1}
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
      {:committed, new_version, _} ->
        :ok =
          push_policy_snapshot(exp_id, ctx, bit_idx, cegar_iter, new_version, reward)

        {{nc + 1, ns, ni, nf}, %{a | committed: MapSet.put(a.committed, bit_idx)}}

      {:rejected, :stale} ->
        # A sibling's commit invalidated this evaluation. Drop any
        # prior settled-at marker so the bit re-dispatches against
        # the new version on the next wave.
        {{nc, ns + 1, ni, nf}, %{a | settled_at: Map.delete(a.settled_at, bit_idx)}}

      {:rejected, :no_improvement} ->
        # Candidate beat its iter-seed baseline but failed the
        # policy-level monotonicity check (probably rollout noise).
        # Mark settled at this version.
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

  # A bit is open if:
  #   * it has not already committed in this step (committed set),
  #   * AND its last settled-at-version is older than the current
  #     wave's starting version (i.e. its no_improvement verdict is
  #     no longer authoritative because the policy advanced).
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
          # Clear the legacy iter machinery; the streaming controller
          # doesn't use it but other readers (dashboard, tests) check
          # these fields. Empty arrays match the post-iter state.
          "current_iter" => 1,
          "bit_shuffle" => [],
          "bit_progress" => []
        })

      __MODULE__.new(%{"experiment_id" => exp.id})
      |> Oban.insert!()

      :ok
    end
  end

  # ── DB helpers ──────────────────────────────────────────────

  defp reset_pool_snapshot(exp_id) do
    sql = """
    UPDATE experiments
    SET best_reward_per_bit = '{}'::jsonb, updated_at = now()
    WHERE id = $1
    """

    Repo.query!(sql, [Ecto.UUID.dump!(exp_id)])
    :ok
  end

  defp push_policy_snapshot(exp_id, ctx, bit_idx, cegar_iter, version, reward) do
    {:ok, exp} = Experiments.get(exp_id)
    preds = decode_predicates(exp.predicates)

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
      "iter" => version,
      "best_reward" => reward,
      "baseline_reward" => exp.baseline_reward
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

  # In-flight bit-batch cap per wave. See `@default_bit_concurrency`.
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

  # PREVIOUSLY: `Process.exit(pid, :kill)`. That was the root cause
  # of the controller-killed-on-step-end family of failures (e.g.
  # HalfCheetah c9c415a1 finishing step 1/3 with 19 committed bits
  # at +2117 mean reward, "step done" logged at 02:05:28.652, and
  # 40 ms later Oban records `EXIT from #PID<...> killed` on attempt
  # 3 — exhausted retries, experiment failed). The `:kill` exit
  # reason is untrappable; the heartbeat dies with `:killed`, and
  # because we started it via `Task.start_link/1` (intentionally —
  # see `start_heartbeat/2`'s doc), the EXIT signal propagates back
  # to the controller. The controller doesn't trap exits, so it
  # dies too — with reason `:killed`, which Oban records as a
  # failure and counts against `max_attempts`. After three clean
  # step-ends in a row the controller is permanently `discarded`
  # and `Server.ObanFailureHandler` marks the experiment `failed`.
  # The work itself is fine — every commit landed atomically via
  # `Server.CommitGate` before this kill — but the master loop
  # bails before it can advance to the next CEGAR step.
  #
  # Fix: unlink first (so even a stray late exit can't take us
  # down), then send the already-supported `:stop` message which
  # the loop's `receive` handles by returning `:ok` cleanly. Idempotent
  # against a dead pid (silent `send` to a dead pid is fine; the
  # `unlink` is a no-op for a non-link).
  defp stop_heartbeat({:ok, pid}) when is_pid(pid) do
    Process.unlink(pid)
    send(pid, :stop)
    :ok
  end

  defp stop_heartbeat(_), do: :ok
end
