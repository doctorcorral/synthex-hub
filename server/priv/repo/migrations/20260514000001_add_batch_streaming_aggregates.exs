defmodule Server.Repo.Migrations.AddBatchStreamingAggregates do
  @moduledoc """
  Layer 1b of the streaming-cegar architecture
  (see `docs/streaming-cegar.md`). Adds running per-batch aggregates
  that are updated atomically on every chunk submission, so the
  dashboard and SSE feeds can render live progress without waiting
  for the batch to finalize.

    * `n_results`        — count of individual candidate results
                            rolled into the aggregates. NOT the same
                            as `completed_chunks`; this counts per-
                            candidate evaluations, not per-chunk.
    * `sum_reward`       — running sum of all numeric rewards (for
                            mean computation: mean = sum_reward /
                            n_results).
    * `sum_sq_reward`    — running sum of squares (for stddev:
                            var = sum_sq_reward / n_results - mean²,
                            stddev = sqrt(var)).
    * `last_result_at`   — wall-clock timestamp of the most recent
                            aggregate update. Used to compute live
                            "candidates per minute" via diffing the
                            previous snapshot in `MetricsBroker`.

  Backfilled in-place for existing batches so the dashboard isn't
  blank on first deploy.
  """
  use Ecto.Migration

  def up do
    alter table(:batches) do
      add :n_results, :integer, null: false, default: 0
      add :sum_reward, :float
      add :sum_sq_reward, :float
      add :last_result_at, :utc_datetime_usec
    end

    flush()

    # Backfill from each batch's results array. Same UNNEST/JSONB
    # pattern as the original best_reward backfill — counts rewardful
    # items, sums them, sums their squares.
    execute("""
    UPDATE batches AS b
    SET
      n_results     = COALESCE(sub.n_results, 0),
      sum_reward    = sub.sum_reward,
      sum_sq_reward = sub.sum_sq_reward,
      last_result_at = b.completed_at
    FROM (
      SELECT
        b2.id AS batch_id,
        COUNT(*) FILTER (WHERE item.value ? 'reward') AS n_results,
        SUM((item.value->>'reward')::float8) AS sum_reward,
        SUM(POWER((item.value->>'reward')::float8, 2)) AS sum_sq_reward
      FROM batches b2
      CROSS JOIN LATERAL UNNEST(b2.results) AS chunk
      CROSS JOIN LATERAL JSONB_ARRAY_ELEMENTS(chunk->'items') AS item
      WHERE b2.cmd = 'score_bit'
        AND COALESCE(ARRAY_LENGTH(b2.results, 1), 0) > 0
        AND (item.value ? 'reward')
      GROUP BY b2.id
    ) AS sub
    WHERE b.id = sub.batch_id
    """)

    # Streaming clients filter by recent activity; index supports that
    # without a full scan.
    create index(:batches, [:status, :last_result_at])
  end

  def down do
    drop index(:batches, [:status, :last_result_at])

    alter table(:batches) do
      remove :last_result_at
      remove :sum_sq_reward
      remove :sum_reward
      remove :n_results
    end
  end
end
