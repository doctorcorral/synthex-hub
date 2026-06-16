defmodule Server.EnvPolicies do
  @moduledoc """
  Data-access layer for `Server.EnvPolicy` rows.

  The env_policy is the long-lived policy artefact owned per
  `(env_name, config_sig)`. Experiments attach to it, write commits
  through `Server.CommitGate` to it, and detach (complete / fail /
  cancel) — the artefact survives.

  Three entry points matter:

    * `upsert_for_submission/2` — bootstrap finds or creates the env_policy
      for the experiment's `(env_name, config_sig)`. Returns the row +
      a `:created | :existing` tag so the caller knows whether to use
      the empty-policy baseline (created) or the lineage's stored
      `best_reward` (existing).

    * `for_experiment/1` — controller reads the current predicates +
      policy_version via the experiment's `env_policy_id`. Replaces
      the old `experiment.predicates` / `experiment.policy_version`
      reads.

    * `lock_for_commit/1` — commit gate takes `FOR UPDATE` on the
      lineage row inside its transaction so two concurrent commits
      from different experiments (or different bit dispatches within
      one experiment) can't both bump `policy_version`.

  No experiment-lifecycle logic lives here; that's in
  `Server.Experiments`. This module knows about lineages only.
  """

  import Ecto.Query
  alias Server.{EnvPolicy, Experiment, PolicyVersion, Repo}
  alias Server.EnvPolicy.ConfigSig

  @doc """
  Find-or-create the env_policy for `(env_name, env_key, config)`.
  The config_sig is derived from `config` via `ConfigSig.sig_for_config/1`.

  Returns `{:ok, env_policy, :created | :existing}`.

  Race-safe under concurrent first-time submissions for the same
  `(env_name, sig)`: the table's `unique_index(:env_policies,
  [:env_name, :config_sig])` makes simultaneous inserts collide;
  we catch the conflict and fall back to a read so the loser
  still gets a valid row.
  """
  @spec upsert_for_submission(map(), keyword()) ::
          {:ok, EnvPolicy.t(), :created | :existing}
  def upsert_for_submission(attrs, _opts \\ []) do
    env_name = Map.fetch!(attrs, :env_name)
    env_key = Map.fetch!(attrs, :env_key)
    config = Map.get(attrs, :config) || %{}

    {sig, canonical} = ConfigSig.sig_for_config(config)

    case Repo.get_by(EnvPolicy, env_name: env_name, config_sig: sig) do
      %EnvPolicy{} = existing ->
        {:ok, existing, :existing}

      nil ->
        attrs = %{
          "env_name" => env_name,
          "env_key" => env_key,
          "config_sig" => sig,
          "config_data" => canonical,
          "predicates" => %{"preds" => []},
          "policy_version" => 0,
          "best_reward" => nil,
          "baseline_reward" => nil,
          "n_episodes" => nil,
          "first_seen_at" => DateTime.utc_now()
        }

        case %EnvPolicy{} |> EnvPolicy.changeset(attrs) |> Repo.insert() do
          {:ok, env_policy} ->
            {:ok, env_policy, :created}

          {:error, %Ecto.Changeset{errors: errors}} ->
            # Concurrent submission inserted the row a microsecond
            # before us — fall back to the now-existing row. Anything
            # else is a real validation failure that we surface.
            if Keyword.has_key?(errors, :config_sig) or
                 Keyword.has_key?(errors, :env_name) do
              case Repo.get_by(EnvPolicy, env_name: env_name, config_sig: sig) do
                %EnvPolicy{} = row -> {:ok, row, :existing}
                _ -> {:error, errors}
              end
            else
              {:error, errors}
            end
        end
    end
  end

  @doc """
  Fetch the env_policy currently attached to an experiment. Used by
  the controller's per-step / per-bit reads of the live predicates +
  policy_version.

  Raises if the experiment has no env_policy_id (bootstrap is
  supposed to set it before transitioning the experiment to
  `running`; an experiment in `running` with `nil` env_policy_id
  is a contract violation worth crashing on).
  """
  @spec for_experiment(Experiment.t() | binary()) ::
          {:ok, EnvPolicy.t()} | {:error, :not_found}
  def for_experiment(%Experiment{env_policy_id: nil} = exp) do
    raise "Experiment #{exp.id} has no env_policy_id — bootstrap contract violated"
  end

  def for_experiment(%Experiment{env_policy_id: id}), do: get(id)

  def for_experiment(experiment_id) when is_binary(experiment_id) do
    from(p in EnvPolicy,
      join: e in Experiment,
      on: e.env_policy_id == p.id,
      where: e.id == ^experiment_id,
      select: p
    )
    |> Repo.one()
    |> case do
      nil -> {:error, :not_found}
      env_policy -> {:ok, env_policy}
    end
  end

  @doc "Fetch one env_policy by id."
  @spec get(binary()) :: {:ok, EnvPolicy.t()} | {:error, :not_found}
  def get(id) when is_binary(id) do
    case Repo.get(EnvPolicy, id) do
      nil -> {:error, :not_found}
      env_policy -> {:ok, env_policy}
    end
  end

  @doc """
  Take a `FOR UPDATE` lock on the env_policy row identified by `id`.
  Caller must already be inside a `Repo.transaction/1`. Returns the
  locked row, or `nil` if it disappeared between read and lock
  (shouldn't happen — env_policies are never deleted in normal flow).
  """
  @spec lock_for_update(binary()) :: EnvPolicy.t() | nil
  def lock_for_update(id) when is_binary(id) do
    from(p in EnvPolicy, where: p.id == ^id, lock: "FOR UPDATE")
    |> Repo.one()
  end

  @doc """
  Set the env_policy's `baseline_reward`/`n_episodes` if not yet
  measured. Called by bootstrap after evaluating the empty-policy
  baseline on a freshly-created env_policy. Idempotent: a non-nil
  `baseline_reward` is preserved (only the first session ever
  measures it for a lineage).
  """
  @spec ensure_baseline(EnvPolicy.t(), float(), pos_integer()) :: {:ok, EnvPolicy.t()}
  def ensure_baseline(%EnvPolicy{baseline_reward: nil} = env_policy, baseline, n_episodes) do
    env_policy
    |> EnvPolicy.changeset(%{
      "baseline_reward" => baseline,
      "n_episodes" => n_episodes,
      # Pre-commit, the "best_reward" of the lineage is the empty
      # policy's reward. The commit gate's monotonicity check uses
      # this as the floor for the first commit; without it, the
      # first commit's improves_policy? would short-circuit on `nil`
      # and accept any reward (which is the OLD behavior — fine
      # when there were no prior commits, but we now want a real
      # floor so a freshly-bootstrapped experiment whose baseline
      # is somehow stronger than its first candidate doesn't
      # mistakenly commit a regression).
      "best_reward" => baseline
    })
    |> Repo.update()
  end

  def ensure_baseline(%EnvPolicy{} = env_policy, _baseline, _n_episodes) do
    {:ok, env_policy}
  end

  @doc """
  Undo a step's commits by restoring the lineage to a prior predicate
  vector. Used by the controller's step-level validation guard: the
  commit gate enforces monotonicity on the small per-round *training*
  seed block, which can overfit and regress the held-out *validation*
  average. When a step's net effect regresses validation, the
  controller calls this to roll the lineage back to `start`.

  The version is bumped *forward* (not decremented): we append a revert
  version carrying `start`'s predicates so the version space stays
  monotonic, the `(env_policy_id, version)` audit key never collides,
  and the revert itself is recorded in `policy_versions` for forensics.
  The session's cached `best_reward` is reset to the restored value.
  """
  @spec revert_step(EnvPolicy.t(), EnvPolicy.t(), Experiment.t(), float(), float()) ::
          {:ok, EnvPolicy.t()}
  def revert_step(%EnvPolicy{} = current, %EnvPolicy{} = start, %Experiment{} = exp, val_before, val_after) do
    Repo.transaction(fn ->
      new_version = current.policy_version + 1

      {:ok, reverted} =
        current
        |> Ecto.Changeset.change(
          predicates: start.predicates,
          policy_version: new_version,
          best_reward: start.best_reward,
          n_episodes: start.n_episodes
        )
        |> Repo.update()

      {:ok, _audit} =
        %PolicyVersion{}
        |> PolicyVersion.changeset(%{
          "env_policy_id" => current.id,
          "experiment_id" => exp.id,
          "version" => new_version,
          "predicates" => start.predicates,
          "bit_idx" => -1,
          "prev_reward" => current.best_reward,
          "new_reward" => start.best_reward || 0.0,
          "metadata" => %{
            "reason" => "validation_guard_revert",
            "validation_before" => val_before,
            "validation_after" => val_after,
            "reverted_to_version" => start.policy_version
          }
        })
        |> Repo.insert()

      {:ok, _exp} =
        exp
        |> Ecto.Changeset.change(best_reward: start.best_reward)
        |> Repo.update()

      reverted
    end)
  end

  @doc """
  Record the lineage's held-out validation average. Called by the
  controller at the end of each CEGAR step with the per-episode mean on
  the fixed validation seed block and the `policy_version` it was
  measured at. This is the canonical, cross-run-comparable performance
  metric (vs the training-block high-water-mark `best_reward`).

  Best-effort: a failure here must never derail the step. Stamped
  outside the commit transaction since it reflects the whole step's net
  policy, not a single bit commit.
  """
  @spec record_validation(EnvPolicy.t(), float(), non_neg_integer(), map() | nil) :: :ok
  def record_validation(env_policy, val_avg, version, tail \\ nil)

  def record_validation(%EnvPolicy{} = env_policy, val_avg, version, tail)
      when is_number(val_avg) and is_integer(version) do
    changes = %{validation_avg: val_avg * 1.0, validation_version: version}
    changes = if is_map(tail), do: Map.put(changes, :validation_tail, tail), else: changes

    env_policy
    |> Ecto.Changeset.change(changes)
    |> Repo.update()

    :ok
  rescue
    _ -> :ok
  end

  def record_validation(_, _, _, _), do: :ok

  @doc "List env_policies for an env_name, newest-committed-first."
  def list_for_env(env_name) do
    from(p in EnvPolicy,
      where: p.env_name == ^env_name,
      order_by: [desc: p.policy_version, desc: p.updated_at]
    )
    |> Repo.all()
  end

  @doc "List all env_policies, for the dashboard summary."
  def list_all do
    from(p in EnvPolicy, order_by: [asc: p.env_name, desc: p.updated_at])
    |> Repo.all()
  end
end
