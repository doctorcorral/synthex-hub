defmodule Server.EnvPolicy do
  @moduledoc """
  Persistent policy artefact owned per `(env_name, config_sig)`.

  This is the source of truth for the synthesized policy. Experiments
  are *sessions* that operate on an env_policy: a fresh submission
  either creates a new env_policy (if no row exists for that
  `(env_name, sig)` pair) or attaches itself to an existing one and
  inherits all of its accepted commits. When the experiment ends —
  cleanly, with errors, or by cancellation — the env_policy keeps the
  predicates that were committed during its life. The next
  submission for the same `(env, sig)` pair picks up from there. No
  accepted compute is ever lost.

  ## Why decouple policy from experiment

  Before this module existed, predicates and `policy_version` lived
  on the `experiments` row. When the controller crashed
  unrecoverably (Oban exhausted attempts, operator cancellation,
  master self-killing bug from the heartbeat-link issue, ...), the
  policy died with it: the next submission for the same env_name
  started from `falsep` and rediscovered the same 19 bits again
  before the operator noticed. That's the wasted-compute pattern this
  module exists to eliminate.

  Now the policy is a long-lived row that experiments *attach to*.
  An experiment failure is a SESSION failure, not a POLICY failure;
  the resubmission inherits the policy and continues from the
  current `policy_version`.

  ## Identity

  `(env_name, config_sig)` is the unique key.

    * `env_name` distinguishes Ant from HalfCheetah etc.
    * `config_sig` (see `Server.EnvPolicy.ConfigSig`) distinguishes
      `bits_per_dim=3` from `bits_per_dim=4` for the same env, plus
      a handful of other policy-shape-determining knobs.

  This lets the same env have multiple distinct policy lineages
  running in parallel for direct comparison. The dashboard surfaces
  each lineage as its own card.

  ## Concurrency

  Multiple experiments CAN exist simultaneously for the same env if
  they differ in `config_sig`. They cannot for the SAME `config_sig` —
  that's enforced by the `experiments_one_active_per_env_policy`
  partial unique index — because two parallel writers to the same
  predicates+version stream would defeat the monotonicity invariant.
  `Server.CommitGate` takes `FOR UPDATE` on the env_policy row to
  serialize commits regardless.
  """

  use Ecto.Schema
  import Ecto.Changeset

  @primary_key {:id, Ecto.UUID, autogenerate: true}
  @foreign_key_type Ecto.UUID

  schema "env_policies" do
    field :env_name, :string
    field :env_key, :string

    # 16-hex-char hash over the canonicalized policy-shape sub-config.
    # See `Server.EnvPolicy.ConfigSig`. Together with `env_name`
    # forms the natural key.
    field :config_sig, :string

    # The canonical map that produced `config_sig`. Stored so the
    # dashboard can render a human-readable subtitle without having
    # to chase back through `last_committed_by_experiment_id`'s
    # config (which might have been pruned).
    field :config_data, :map, default: %{}

    # The policy itself. Same JSON shape as the legacy
    # `experiments.predicates`: `%{"preds" => [encoded_term, ...]}`.
    field :predicates, :map, default: %{"preds" => []}

    # Monotonically increasing counter, bumped by `Server.CommitGate`
    # on every accepted improvement. Promoted from `experiments` —
    # one global version per `(env, sig)` lineage now, not one per
    # experiment row.
    field :policy_version, :integer, default: 0

    # `best_reward` is the reward of the current predicates under the
    # validation seed set, in the same sum-domain units the rest of
    # the system uses (per-seed sum across `n_episodes`). Divide by
    # `n_episodes` for per-episode mean display.
    #
    # `baseline_reward` is the reward of the empty (`falsep`)
    # predicates from the first-ever experiment in this lineage —
    # i.e., the "from scratch" floor used to compute the lineage's
    # total contribution. Kept constant for the life of the row so
    # the dashboard's "+Δ vs baseline" reads as cumulative progress
    # rather than per-session progress.
    field :best_reward, :float
    field :baseline_reward, :float

    # The n_episodes the rewards above were measured under. Used by
    # the API to convert sum-domain rewards to per-episode means.
    # Different experiments in the same lineage may use different
    # n_episodes values; we store the one that recorded `best_reward`
    # because that's what `best_reward` is calibrated against.
    field :n_episodes, :integer

    field :first_seen_at, :utc_datetime_usec

    belongs_to :last_committed_by_experiment, Server.Experiment,
      foreign_key: :last_committed_by_experiment_id,
      type: Ecto.UUID

    has_many :experiments, Server.Experiment
    has_many :policy_versions, Server.PolicyVersion

    timestamps(type: :utc_datetime_usec)
  end

  @castable ~w(env_name env_key config_sig config_data
               predicates policy_version best_reward baseline_reward
               n_episodes first_seen_at last_committed_by_experiment_id)a

  def changeset(env_policy, attrs) do
    env_policy
    |> cast(attrs, @castable)
    |> validate_required([:env_name, :env_key, :config_sig])
    |> unique_constraint([:env_name, :config_sig], name: :env_policies_env_name_config_sig_index)
  end
end
