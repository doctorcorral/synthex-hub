defmodule Server.Repo.Migrations.CreateBatchContributions do
  @moduledoc """
  Per-(batch, worker) contribution counters. Driven by `complete_chunk/3`
  on every successful chunk submission. `display_name` is denormalized
  from `workers.name` at submission time; this lets the leaderboard
  GROUP BY display_name so all rows with name `"anonymous"` aggregate
  into a single bucket.

  No FK constraints on purpose: we want contributions to survive even
  if a worker row is later deleted, and we want chunk submissions to
  succeed for transient/unregistered workers.
  """
  use Ecto.Migration

  def change do
    create table(:batch_contributions, primary_key: false) do
      add :batch_id, :string, primary_key: true, null: false
      add :worker_id, :string, primary_key: true, null: false
      add :display_name, :string, null: false
      add :chunks_completed, :integer, default: 0, null: false
      add :candidates_evaluated, :bigint, default: 0, null: false
      add :first_chunk_at, :utc_datetime_usec, null: false
      add :last_chunk_at, :utc_datetime_usec, null: false
    end

    create index(:batch_contributions, [:display_name])
    create index(:batch_contributions, [:batch_id])
    create index(:batch_contributions, [:last_chunk_at])
  end
end
