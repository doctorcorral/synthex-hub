defmodule Server.Repo.Migrations.AddObanJobsChunkIdIndex do
  use Ecto.Migration

  @moduledoc """
  Index `oban_jobs (args->>'chunk_id')` so chunk completion is O(log n).

  ## The bug this fixes

  `Server.Queue.complete_chunk/3` locates the chunk's Oban job with

      from j in Oban.Job,
        where: fragment("? ->> 'chunk_id' = ?", j.args, ^chunk_id),
        lock: "FOR UPDATE"

  There was no index on the `args->>'chunk_id'` JSON expression, so
  every single result submission did a **sequential scan of the whole
  `oban_jobs` table under `FOR UPDATE`**. While the table was small
  this was invisible. It stopped being small: a worker stuck in a
  failed-submit retry loop (each cycle spawns and then cancels chunk
  jobs) bloated `oban_jobs` to ~265k rows, ~249k of them `cancelled`
  and still inside the pruner's 7-day window. At that size the seq
  scan exceeded the worker's 30s HTTP receive timeout, so EVERY
  submit failed with `Req.TransportError :timeout` — which in turn
  spawned more retry/cancel churn. Claim and heartbeat stayed fast
  the whole time (they hit Oban's indexed `state`/`queue` columns),
  which is exactly why the hang looked selective and got
  misattributed to the database tier.

  With this expression index the lookup is an index scan regardless
  of table size (measured: 30s+ → ~3ms on the bloated table), so
  completion latency no longer depends on how much terminal-state
  job history is lying around.

  ## Companion change

  This PR also lowers `Oban.Plugins.Pruner` `max_age` (see
  `config/config.exs`) so terminal jobs don't accumulate to that
  scale again. The index is the correctness fix; the shorter
  retention is defense-in-depth and Neon storage/egress hygiene.

  ## Idempotency

  `create_if_not_exists` because the index was already applied by
  hand on production to unblock a live run; on a fresh database the
  table is tiny and the build is instant, so a plain (non-concurrent)
  create is safe and avoids `CONCURRENTLY`'s fragility over a
  transaction-pooled connection.
  """

  def up do
    create_if_not_exists index(:oban_jobs, ["(args->>'chunk_id')"],
                            name: :oban_jobs_chunk_id_idx
                          )
  end

  def down do
    drop_if_exists index(:oban_jobs, ["(args->>'chunk_id')"], name: :oban_jobs_chunk_id_idx)
  end
end
