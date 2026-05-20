defmodule Server.Experiment do
  @moduledoc """
  A CEGAR synthesis SESSION. One row per `POST /api/master/experiments`
  submission. The session attaches to a long-lived
  `Server.EnvPolicy` row (the *lineage*) which is the source of truth
  for the synthesized predicates and the running `policy_version`.

  ## Lineage / session split

  Before `env_policies`, an experiment row owned both the session
  state AND the policy state. When a session crashed/cancelled/failed
  the predicates went with it — the next submission for the same env
  started from `falsep` and rediscovered the same bits again.

  Now:

    * `Server.EnvPolicy` owns `predicates`, `policy_version`,
      `best_reward`, `baseline_reward` — survives every experiment
      session for its `(env_name, config_sig)`.

    * `Server.Experiment` owns the session-scoped fields below: which
      CEGAR step we're on, when we started, how many bits this session
      accepted, whether the session is still running. All policy
      state is fetched via the `env_policy_id` FK.

  This module is therefore a session log. `experiments.best_reward`
  is the highest reward observed *during this session* (>=
  env_policy.best_reward at session start by construction, since the
  commit gate's monotonicity guards reject regressions);
  `experiments.baseline_reward` is what the session started from
  (== env_policy.best_reward at submit time, or the empty-policy
  reward when this session was the lineage's first).

  ## State machine

      pending  ──bootstrap──→ running ──last_step_done──→ completed
         ↓                       ↓
       (fail / cancel)         (fail / cancel)
         ↓                       ↓
       cancelled               failed
  """

  use Ecto.Schema
  import Ecto.Changeset

  @primary_key {:id, Ecto.UUID, autogenerate: true}
  @foreign_key_type Ecto.UUID
  schema "experiments" do
    field :env_name, :string
    field :env_key, :string
    field :submitter, :string

    field :config, :map, default: %{}
    field :status, :string, default: "pending"

    # CEGAR step within the configured `cegar_rounds`. The streaming
    # controller advances this on each step's saturation.
    field :current_cegar_iter, :integer, default: 0
    field :current_iter, :integer, default: 0

    # Session-scoped reward summary. `baseline_reward` is the reward
    # of the policy this session started from (inherited from
    # env_policy, or empty-policy reward if the session was the
    # lineage's first). `best_reward` is the highest reward the
    # commit gate accepted during this session — by the
    # monotonicity invariant it is always >= baseline_reward and
    # always >= env_policy.best_reward at session start.
    field :baseline_reward, :float
    field :best_reward, :float
    field :accepted_count, :integer, default: 0

    field :started_at, :utc_datetime_usec
    field :completed_at, :utc_datetime_usec
    field :error, :string

    # The lineage this session writes into. Set by `ExperimentBootstrap`
    # (upserting an env_policies row if one doesn't yet exist for
    # the submission's `(env_name, config_sig)`). After bootstrap
    # this is immutable.
    belongs_to :env_policy, Server.EnvPolicy, type: Ecto.UUID

    timestamps(type: :utc_datetime_usec)
  end

  @castable ~w(env_name env_key submitter config status
               current_cegar_iter current_iter
               baseline_reward best_reward accepted_count
               started_at completed_at error env_policy_id)a

  def changeset(experiment, attrs) do
    experiment
    |> cast(attrs, @castable)
    |> validate_required([:env_name, :env_key])
    |> validate_inclusion(:status, ~w(pending running completed failed cancelled))
    # Matches the partial unique index in the migration. The
    # constraint surface uses `env_policy_id` because a lineage is
    # the natural "one active at a time" boundary now — two
    # submissions for the same env with DIFFERENT configs (different
    # bits_per_dim etc.) get different env_policy_ids and run in
    # parallel; two submissions with the SAME config share an
    # env_policy_id and conflict.
    |> unique_constraint(:env_policy_id, name: :experiments_one_active_per_env_policy)
  end
end
