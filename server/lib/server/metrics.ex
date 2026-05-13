defmodule Server.Metrics do
  @moduledoc """
  Read-only "load" snapshot of the hub for the public SSE stream.

  Six cheap aggregates — each a single-row Postgres aggregate over
  small index-friendly tables. Computed once per second by
  `Server.MetricsBroker` and broadcast over Server-Sent Events;
  individual SSE clients never query the DB directly.

  Fields:

    * `workers_active` — `WorkerNode.status == "active"` count
    * `cores_total`    — sum of `WorkerNode.pool_size` for active workers
    * `chunks_executing` — Oban jobs currently being processed
    * `chunks_pending`   — Oban jobs waiting in queue (available + scheduled + retryable)
    * `evals_total`      — all-time sum of `WorkerNode.candidates_evaluated`
    * `active_batches`   — Batches in `pending` or `running` state
  """

  import Ecto.Query
  alias Server.{Repo, Batch, WorkerNode}

  @pending_states ~w(available scheduled retryable)
  @executing_states ~w(executing)

  @spec snapshot() :: map()
  def snapshot do
    %{
      workers_active: workers_active(),
      cores_total: cores_total(),
      chunks_executing: oban_count(@executing_states),
      chunks_pending: oban_count(@pending_states),
      evals_total: evals_total(),
      active_batches: active_batches()
    }
  end

  defp workers_active do
    from(w in WorkerNode, where: w.status == "active")
    |> Repo.aggregate(:count, :id)
  end

  defp cores_total do
    (from(w in WorkerNode, where: w.status == "active")
     |> Repo.aggregate(:sum, :pool_size)) || 0
  end

  defp oban_count(states) do
    from(j in Oban.Job, where: j.state in ^states)
    |> Repo.aggregate(:count, :id)
  end

  defp evals_total do
    Repo.aggregate(WorkerNode, :sum, :candidates_evaluated) || 0
  end

  defp active_batches do
    from(b in Batch, where: b.status in ["pending", "running"])
    |> Repo.aggregate(:count, :id)
  end
end
