defmodule Server.Repo.Migrations.AddBatchScoreSummary do
  @moduledoc """
  Cache per-batch reward aggregates so the public landing page can
  show a per-environment leaderboard without scanning every chunk's
  results JSON on each request.

    * `best_reward`     — max reward across all candidates in this
                          batch's chunks. NULL until the first chunk
                          with rewards arrives, or for non-
                          `score_bit` batches like `collect_states`.
    * `baseline_reward` — for `score_bit` batches, the reward of the
                          prepended baseline candidate (chunk_index
                          0, item 0). NULL otherwise.

  Backfilled in-place from existing `results` arrays on completed
  `score_bit` batches so the dashboard isn't blank on first deploy.
  """
  use Ecto.Migration

  def up do
    alter table(:batches) do
      add :best_reward, :float
      add :baseline_reward, :float
    end

    flush()

    # Backfill: max reward across all chunk items for each completed
    # score_bit batch.
    execute("""
    UPDATE batches AS b
    SET best_reward = sub.best_reward
    FROM (
      SELECT
        b2.id AS batch_id,
        MAX((item.value->>'reward')::float8) AS best_reward
      FROM batches b2
      CROSS JOIN LATERAL UNNEST(b2.results) AS chunk
      CROSS JOIN LATERAL JSONB_ARRAY_ELEMENTS(chunk->'items') AS item
      WHERE b2.cmd = 'score_bit'
        AND b2.status = 'completed'
        AND COALESCE(ARRAY_LENGTH(b2.results, 1), 0) > 0
        AND (item.value ? 'reward')
      GROUP BY b2.id
    ) AS sub
    WHERE b.id = sub.batch_id
    """)

    # Baseline: chunk_index = 0, item 0 (master prepends it before
    # submission in `Synthex.Hub.Client.score_bit/3`).
    execute("""
    UPDATE batches AS b
    SET baseline_reward = sub.baseline_reward
    FROM (
      SELECT
        b2.id AS batch_id,
        ((chunk->'items')->0->>'reward')::float8 AS baseline_reward
      FROM batches b2
      CROSS JOIN LATERAL UNNEST(b2.results) AS chunk
      WHERE b2.cmd = 'score_bit'
        AND b2.status = 'completed'
        AND (chunk->>'chunk_index')::int = 0
    ) AS sub
    WHERE b.id = sub.batch_id
    """)

    create index(:batches, [:env_name, :status])
  end

  def down do
    drop index(:batches, [:env_name, :status])

    alter table(:batches) do
      remove :best_reward
      remove :baseline_reward
    end
  end
end
