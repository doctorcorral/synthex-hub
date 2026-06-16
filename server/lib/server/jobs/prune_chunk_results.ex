defmodule Server.Jobs.PruneChunkResults do
  @moduledoc """
  Periodic Oban cron job: delete aged-out rows from `chunk_results`.

  `chunk_results` is the narrow home for per-chunk evaluation items
  (see `Server.Queue.fetch_batch_chunks/1`). The controller reads a
  batch's items exactly once, at end-of-batch, within minutes of
  completion — after that the rows are dead weight. Unlike the
  `oban_jobs` rows they replaced, nothing prunes them automatically, so
  without this job the table would grow without bound and reintroduce
  the very bloat #2 set out to remove.

  We delete by age rather than by batch terminal-state so a single
  cheap, index-free-friendly statement covers everything. The window
  (6h, double the Oban Pruner's 3h) leaves a generous margin for a slow
  or retried controller to still fetch results before they're swept.
  """
  use Oban.Worker, queue: :system, max_attempts: 1

  import Ecto.Query
  require Logger
  alias Server.Repo

  @max_age_seconds 60 * 60 * 6

  @impl Oban.Worker
  def perform(_job) do
    max_age = Application.get_env(:server, :chunk_results_max_age_secs, @max_age_seconds)
    cutoff = DateTime.add(DateTime.utc_now(), -max_age, :second)

    {deleted, _} =
      from(c in "chunk_results", where: c.inserted_at < ^cutoff)
      |> Repo.delete_all(timeout: 60_000)

    if deleted > 0 do
      Logger.info("[PruneChunkResults] deleted #{deleted} aged chunk_results rows (>#{max_age}s)")
    end

    :ok
  end
end
