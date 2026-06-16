defmodule Server.Repo.Migrations.CreateChunkResults do
  use Ecto.Migration

  # Narrow home for per-chunk evaluation results, decoupled from
  # `oban_jobs`.
  #
  # Previously a completed chunk stored its result items inside
  # `oban_jobs.args["results"]` — the SAME wide jsonb row that also
  # carries the chunk's input `candidates` and `params`. Two problems:
  #
  #   1. `fetch_batch_chunks/1` dragged every completed chunk's full
  #      `args` jsonb (candidates + params + results) back in ONE query
  #      to extract just `results`. For a 10^4-chunk batch that's a
  #      multi-GB read that blows the statement timeout.
  #   2. Results lived on a row owned by Oban's lifecycle — the Pruner
  #      deletes completed `oban_jobs` after a few hours, so a batch
  #      that outlived the prune window lost its results.
  #
  # This table stores ONLY `(batch_id, chunk_index, results)`, written
  # once on chunk completion and read back by keyset pagination over
  # `chunk_index`. The unique `(batch_id, chunk_index)` index doubles
  # as the pagination/scan index (leading column `batch_id`) and as the
  # idempotency guard for duplicate chunk submissions.
  def change do
    create table(:chunk_results, primary_key: false) do
      add :batch_id, :string, null: false
      add :chunk_index, :integer, null: false
      add :results, :map, null: false, default: fragment("'[]'::jsonb")
      add :inserted_at, :utc_datetime_usec, null: false, default: fragment("now()")
    end

    create unique_index(:chunk_results, [:batch_id, :chunk_index])
  end
end
