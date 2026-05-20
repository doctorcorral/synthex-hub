defmodule Server.CommitGate do
  @moduledoc """
  The atomic commit gate for the versioned streaming-CEGAR controller
  (`docs/streaming-cegar.md` §Layer 3 / "Step 1" of the deployment
  plan).

  This module is the **only** path that mutates an env_policy's
  predicates. Every accepted improvement goes through
  `attempt_commit/1`. The gate enforces three invariants atomically:

    1. **Version-fresh evaluation**: the candidate must have been
       evaluated against the predicates of the current
       `env_policy.policy_version`. If a concurrent commit bumped
       the version between dispatch and result, the result is
       *stale* and silently discarded.

    2. **Monotone policy improvement**: the candidate's reward must
       exceed `env_policy.best_reward` by at least
       `acceptance_epsilon`. Under version-reject (1), the candidate
       was evaluated against the *current* policy's seeds, so this
       comparison is on the same seed set as the running best —
       strict policy-level monotonicity follows by construction.

    3. **Atomic bump + audit**: on success the env_policy row's
       predicates, `policy_version`, `best_reward` are bumped, the
       session row's `best_reward`/`accepted_count` cache is
       updated, and a `policy_versions` audit row is inserted, in a
       single transaction. Either everything commits or nothing
       does.

  ## Why env_policy ownership

  Before env_policies, predicates and policy_version lived on the
  experiment row. A session crash (Oban retries exhausted,
  cancellation, OOM) destroyed the policy lineage; the next
  submission for the same env started from `falsep` and rediscovered
  the same bits. The promotion of policy ownership to env_policies
  (one row per `(env_name, config_sig)`) fixes this: the lineage
  survives every session, every commit is permanent, and the
  next submission inherits the policy automatically.

  The `FOR UPDATE` lock is now on the env_policy row, not the
  experiment row. Two simultaneous bit-result arrivals — from the
  same session or, in principle, from two sessions writing to the
  same lineage — serialize at the env_policy level so the
  monotonicity invariant holds globally for the lineage.
  """

  import Ecto.Query

  alias Server.{EnvPolicies, EnvPolicy, Experiment, Experiments,
                PolicyVersion, Repo}

  @typedoc """
  Reasons the gate can reject a commit. None are errors — they're
  expected outcomes the controller handles routinely.
  """
  @type rejection ::
          :stale
          | :no_improvement
          | :experiment_not_running

  @typedoc """
  A successful commit returns the new lineage version plus the
  updated env_policy + experiment so the caller can advance its
  in-memory baseline without re-reading.
  """
  @type commit_ok :: {:committed, pos_integer(), EnvPolicy.t(), Experiment.t()}

  @doc """
  Attempt to commit a single bit's improvement.

  ## Arguments (keyword list)

    * `:experiment_id`       — the session attributing the commit.
                                The gate looks up its env_policy_id.
    * `:bit_idx`             — which bit slot the candidate replaces.
    * `:candidate_term`      — the predicate, in `PrettyPrint.to_json_term/1`
                               form (already serialized).
    * `:reward`              — the candidate's mean reward as measured
                               against `evaluated_at_version`'s
                               predicates.
    * `:evaluated_at_version` — the `policy_version` the worker saw
                               when it produced this candidate. The
                               gate compares against the env_policy's
                               current version under `FOR UPDATE`.
    * `:acceptance_epsilon`  — optional. Minimum margin by which the
                               candidate must beat the env_policy's
                               running best reward (default `0.0`)
                               to count as strict monotone progress.
                               Set higher to filter rollout noise.
    * `:worker_id`           — optional. Foreign key into `workers`
                               recorded on the audit row.
    * `:metadata`            — optional. Forensic blob stored in the
                               audit row; the gate doesn't read it.

  ## Returns

    * `{:committed, new_version, %EnvPolicy{}, %Experiment{}}` — lineage
      predicates updated, version bumped, audit row inserted, session
      stats bumped.
    * `{:rejected, :stale}` — evaluated-at version is older than the
      lineage's current. The result is silently discarded.
    * `{:rejected, :no_improvement}` — reward did not exceed the
      lineage's current best by `acceptance_epsilon`.
    * `{:rejected, :experiment_not_running}` — the experiment moved
      to a terminal state while the candidate was in flight. The
      controller will treat this as a terminal signal.
  """
  @spec attempt_commit(keyword()) ::
          commit_ok()
          | {:rejected, rejection()}
          | {:error, term()}
  def attempt_commit(opts) do
    experiment_id = Keyword.fetch!(opts, :experiment_id)
    bit_idx = Keyword.fetch!(opts, :bit_idx)
    candidate_term = Keyword.fetch!(opts, :candidate_term)
    reward = Keyword.fetch!(opts, :reward)
    evaluated_at_version = Keyword.fetch!(opts, :evaluated_at_version)
    epsilon = Keyword.get(opts, :acceptance_epsilon, 0.0)
    worker_id = Keyword.get(opts, :worker_id)
    metadata = Keyword.get(opts, :metadata, %{})

    Repo.transaction(fn ->
      case lock_session(experiment_id) do
        nil ->
          Repo.rollback({:rejected, :experiment_not_running})

        %Experiment{status: status} when status != "running" ->
          Repo.rollback({:rejected, :experiment_not_running})

        %Experiment{env_policy_id: nil} ->
          # Shouldn't happen post-bootstrap; treat as terminal so the
          # controller bails out cleanly rather than spinning.
          Repo.rollback({:rejected, :experiment_not_running})

        %Experiment{env_policy_id: env_policy_id} = exp ->
          case EnvPolicies.lock_for_update(env_policy_id) do
            nil ->
              # Shouldn't happen — env_policies are never deleted in
              # normal flow. Treat as a transient error so the
              # controller can decide whether to retry.
              Repo.rollback({:error, :env_policy_missing})

            %EnvPolicy{} = env_policy ->
              do_commit(env_policy, exp,
                bit_idx: bit_idx,
                candidate_term: candidate_term,
                reward: reward,
                evaluated_at_version: evaluated_at_version,
                acceptance_epsilon: epsilon,
                worker_id: worker_id,
                metadata: metadata
              )
          end
      end
    end)
    |> case do
      {:ok, {:committed, _, _, _} = ok} -> ok
      {:error, {:rejected, _} = rej} -> rej
      {:error, other} -> {:error, other}
    end
  end

  # ── Internals ───────────────────────────────────────────────────

  # The session row is locked first (cheap, narrow FOR UPDATE) so a
  # status change racing with a commit (operator cancel, complete
  # transition) doesn't slip past the running-check below. The lineage
  # FOR UPDATE follows inside `do_commit` once we've confirmed the
  # session is still alive.
  defp lock_session(experiment_id) do
    from(e in Experiment,
      where: e.id == ^experiment_id,
      lock: "FOR UPDATE"
    )
    |> Repo.one()
  end

  defp do_commit(%EnvPolicy{} = env_policy, %Experiment{} = exp, opts) do
    bit_idx = Keyword.fetch!(opts, :bit_idx)
    candidate_term = Keyword.fetch!(opts, :candidate_term)
    reward = Keyword.fetch!(opts, :reward) * 1.0
    evaluated_at_version = Keyword.fetch!(opts, :evaluated_at_version)
    epsilon = Keyword.fetch!(opts, :acceptance_epsilon)
    worker_id = Keyword.get(opts, :worker_id)
    metadata = Keyword.fetch!(opts, :metadata)

    cond do
      evaluated_at_version != env_policy.policy_version ->
        Repo.rollback({:rejected, :stale})

      not improves_policy?(env_policy.best_reward, reward, epsilon) ->
        Repo.rollback({:rejected, :no_improvement})

      true ->
        new_version = env_policy.policy_version + 1
        new_predicates = replace_pred(env_policy.predicates, bit_idx, candidate_term)
        prev_reward = env_policy.best_reward
        session_n_eps = Experiments.n_episodes_for(exp)

        {:ok, updated_env_policy} =
          env_policy
          |> Ecto.Changeset.change(
            predicates: new_predicates,
            policy_version: new_version,
            best_reward: reward,
            # n_episodes is calibration for `best_reward`: if a later
            # session re-measures the lineage under a different
            # n_episodes, the dashboard needs to know which calibration
            # the stored sum-domain `best_reward` was measured under.
            n_episodes: session_n_eps,
            last_committed_by_experiment_id: exp.id
          )
          |> Repo.update()

        # Session-scoped cache. `experiments.best_reward` is the
        # session's monotonically increasing best; updating it here in
        # the same transaction keeps the dashboard's per-session
        # delta-vs-baseline display in lockstep with the lineage state.
        new_session_best =
          case exp.best_reward do
            nil -> reward
            current when current >= reward -> current
            _ -> reward
          end

        {:ok, updated_exp} =
          exp
          |> Ecto.Changeset.change(
            best_reward: new_session_best,
            accepted_count: (exp.accepted_count || 0) + 1
          )
          |> Repo.update()

        # Stamp the calibration into the audit row so an audit replay
        # can convert the stored sum-domain rewards back into
        # per-episode means even if the env_policy's n_episodes
        # later changes (e.g. a future session uses a different
        # n_episodes). The lineage's current n_episodes is the
        # session's; older audit rows preserve theirs.
        audit_metadata =
          metadata
          |> Map.put_new("n_episodes_at_commit", session_n_eps)

        {:ok, _audit} =
          %PolicyVersion{}
          |> PolicyVersion.changeset(%{
            "env_policy_id" => env_policy.id,
            "experiment_id" => exp.id,
            "version" => new_version,
            "predicates" => new_predicates,
            "bit_idx" => bit_idx,
            "prev_reward" => prev_reward,
            "new_reward" => reward,
            "worker_id" => worker_id,
            "metadata" => audit_metadata
          })
          |> Repo.insert()

        {:committed, new_version, updated_env_policy, updated_exp}
    end
  end

  # First commit on a lineage with no measured baseline yet: trivially
  # an improvement. After `EnvPolicies.ensure_baseline/3` runs during
  # bootstrap, `best_reward` is non-nil and this path is the normal
  # strict-improvement check.
  defp improves_policy?(nil, _reward, _epsilon), do: true
  defp improves_policy?(best, reward, epsilon), do: reward > best + epsilon

  # `predicates` is `%{"preds" => [encoded_term, ...]}`. Replacing
  # one slot preserves the encoding contract round-tripped through
  # `Synthex.Core.PrettyPrint.to_json_term/1` /
  # `from_json_term/1`.
  defp replace_pred(%{"preds" => list} = predicates, bit_idx, encoded)
       when is_list(list) and bit_idx < length(list) do
    Map.put(predicates, "preds", List.replace_at(list, bit_idx, encoded))
  end

  defp replace_pred(predicates, _bit_idx, _encoded) do
    raise ArgumentError,
          "Server.CommitGate.replace_pred/3: bit_idx out of range " <>
            "for #{inspect(predicates)}"
  end
end
