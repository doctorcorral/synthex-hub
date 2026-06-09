defmodule Server.Repo.Migrations.ClearBatchResultsBloat do
  use Ecto.Migration

  @moduledoc """
  One-shot reclaim of the `batches.results` jsonb-array column.

  Background — the egress catastrophe this fixes:

  `batches.results` is a jsonb-array column that, before this PR's
  companion application change, got `array_append`ed on every chunk
  completion. Each push:

    * rewrote the full TOASTed array (Postgres can't HOT-share a
      varlena column across MVCC tuple versions), so writes were
      O(chunks_so_far) bytes;
    * was followed by a `Repo.get!(Batch, ...)` that pulled the
      bloated row back over the wire to decide if completion had
      been reached.

  Net effect: O(N²) traffic in *both* directions for a batch of N
  chunks. The orphaned Ant `collect_states` batch with 19k chunks
  alone projected to multi-TB Postgres↔app traffic — and we paid
  for every byte through Neon's metered network. We've now stopped
  populating this column entirely; per-chunk items live solely on
  `oban_jobs.args["results"]`, where the pruner reclaims them
  after 7 days.

  This migration empties the column on every existing batch
  (`results = '{}'`). The data is redundant — every chunk's items
  are also on the corresponding `oban_jobs` row — so we lose
  nothing.

  ## Storage reclaim

  We deliberately do NOT run `VACUUM FULL` here, because it takes
  an `ACCESS EXCLUSIVE` lock that would block every query against
  `batches` (and stall AggregateBroker / public-status reads) for
  the duration of the rewrite. The plain UPDATE only marks the
  old TOAST chunks dead; autovacuum will reclaim them in the
  background under normal `vacuum_cost_limit` budget over the
  next hour or so. If you need the space back immediately, run
  `VACUUM (FULL, ANALYZE) batches` from an rpc session during a
  maintenance window — but the egress fix this migration enables
  is itself the dominant cost item, not the storage.
  """

  def up do
    execute("UPDATE batches SET results = '{}' WHERE array_length(results, 1) > 0")
  end

  def down do
    # Re-inflating the column is impossible — the data is gone
    # (and the application no longer writes to it anyway). The
    # column itself is kept as a no-op for backward read
    # compatibility; no schema change to roll back.
    :ok
  end
end
