defmodule Server.Repo.Migrations.AddPolicyVersioning do
  use Ecto.Migration

  @moduledoc """
  Step 1 of `docs/streaming-cegar.md` §Layer 3 — versioned commit gate.

  Adds:

    * `experiments.policy_version` — monotonically increasing counter,
      bumped by the commit gate on every accepted improvement. The
      gate's `FOR UPDATE` + `WHERE policy_version = $evaluated_at`
      makes stale results trivial to reject without explicit locks at
      the worker level.

    * `experiments.best_reward_per_bit` — per-bit best reward seen so
      far at the current policy_version. The commit gate compares a
      candidate's reward against `best_reward_per_bit[bit_idx]` (with
      `acceptance_epsilon`) before committing. Stored as `jsonb` of
      `{"<bit_idx>": float}` so we don't need a separate table for
      what amounts to an in-memory pool snapshot.

    * `policy_versions` — audit log. One row per accepted commit.
      Records the bit that flipped, the previous and new rewards, the
      committing worker (nullable; the master is technically the
      committer but we record the worker whose evaluation triggered
      it). Indexed on `(experiment_id, version)` and
      `(experiment_id, committed_at)` so the dashboard's commit-log
      strip + reward-over-time chart are cheap to render.

  Forward-compatible with the legacy `ExperimentCegarIter` flow at
  the time of this migration: it ignored `policy_version` (writes
  never touched it), and the column defaulted to 0.
  `ExperimentController` (Step 2) drives these columns; the
  `experiments.policy_version` / `best_reward_per_bit` columns are
  later dropped by the env_policies promotion migration.
  """

  def up do
    alter table(:experiments) do
      add :policy_version, :integer, null: false, default: 0
      # JSON shape: `{"0": -2820.4, "1": -2950.1, ...}` keyed by bit
      # index as string. `nil` (column-level) means no bit has been
      # scored yet; an empty `{}` means "scored but no bit reached
      # acceptance" — distinct meanings.
      add :best_reward_per_bit, :map, default: %{}
    end

    create table(:policy_versions) do
      add :experiment_id,
          references(:experiments, type: :uuid, on_delete: :delete_all),
          null: false

      # Monotonically increasing within an experiment. Starts at 1
      # (version 0 == "no commits yet, predicates are baseline").
      add :version, :integer, null: false

      add :predicates, :map, null: false
      add :bit_idx, :integer, null: false
      add :prev_reward, :float
      add :new_reward, :float, null: false

      # Workers table uses a string primary key (the worker's
      # self-chosen install ID), so the FK is :string too.
      add :worker_id,
          references(:workers, type: :string, on_delete: :nilify_all)

      # Anonymous metadata (acceptance_epsilon used, time-since-prev,
      # which validation seed window, etc.) — keeps the table itself
      # narrow without losing forensic info.
      add :metadata, :map, null: false, default: %{}

      timestamps(type: :utc_datetime_usec, updated_at: false)
    end

    # (experiment_id, version) is the natural primary lookup key —
    # used by the audit-log replay test (re-derive predicates by
    # folding versions) and by the dashboard SSE feed.
    create unique_index(:policy_versions, [:experiment_id, :version])
    create index(:policy_versions, [:experiment_id, :inserted_at])
  end

  def down do
    drop table(:policy_versions)

    alter table(:experiments) do
      remove :best_reward_per_bit
      remove :policy_version
    end
  end
end
