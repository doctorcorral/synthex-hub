defmodule Server.PolicyVersion do
  @moduledoc """
  Append-only audit log of every accepted commit produced by
  `Server.CommitGate`. One row per `policy_version` bump in an
  env_policy's lineage; the lineage's history can be replayed in full
  by folding these in order.

  ## Keying

  Each row is keyed by `(env_policy_id, version)` — the lineage owns
  the version space, not the session that triggered the commit. Two
  experiments writing to the same env_policy can never produce two
  rows with the same version (the commit gate's `FOR UPDATE` on the
  env_policy row makes the bump strictly serial).

  `experiment_id` is retained as attribution metadata: which session
  triggered this commit. Nullable because deleting an experiment
  shouldn't erase the lineage's audit history.

  ## Used by

    * The dashboard's commit-log strip (last N versions for a lineage).
    * The reward-over-time sparkline (`new_reward` series).
    * Post-hoc analysis: which worker triggered which improvement,
      what the acceptance epsilon was, etc. — `metadata` is the
      forensic blob.

  Insertion happens **inside the same transaction** as the env_policy
  row update in `Server.CommitGate.attempt_commit/1`, so the audit
  log and the live policy state never disagree.
  """

  use Ecto.Schema
  import Ecto.Changeset

  @primary_key {:id, :id, autogenerate: true}
  @foreign_key_type Ecto.UUID

  schema "policy_versions" do
    belongs_to :env_policy, Server.EnvPolicy, type: Ecto.UUID
    belongs_to :experiment, Server.Experiment, type: Ecto.UUID

    field :version, :integer
    field :predicates, :map
    field :bit_idx, :integer
    field :prev_reward, :float
    field :new_reward, :float
    field :worker_id, :string
    field :metadata, :map, default: %{}

    timestamps(type: :utc_datetime_usec, updated_at: false)
  end

  @castable ~w(env_policy_id experiment_id version predicates bit_idx
               prev_reward new_reward worker_id metadata)a

  def changeset(version, attrs) do
    version
    |> cast(attrs, @castable)
    |> validate_required([:env_policy_id, :version, :predicates, :bit_idx, :new_reward])
    |> unique_constraint([:env_policy_id, :version])
    |> foreign_key_constraint(:env_policy_id)
    |> foreign_key_constraint(:experiment_id)
  end
end
