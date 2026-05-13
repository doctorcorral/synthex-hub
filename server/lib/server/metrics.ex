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

  # Oban v2.18's `oban_jobs.state` column is a plain `:string` —
  # Ecto-cast-checks fail loudly if we pass atoms here, so we use
  # the canonical state strings exactly as Oban writes them.
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
    from(w in WorkerNode, where: w.status == "active")
    |> Repo.aggregate(:sum, :pool_size)
    |> to_int()
  end

  defp oban_count(states) do
    from(j in Oban.Job, where: j.state in ^states)
    |> Repo.aggregate(:count, :id)
  end

  defp evals_total do
    Repo.aggregate(WorkerNode, :sum, :candidates_evaluated)
    |> to_int()
  end

  # Postgres' `SUM(integer)` returns numeric/bigint; Ecto sometimes
  # decodes that as `%Decimal{}` (depending on adapter version), and
  # downstream consumers — JSON encoding, arithmetic against ints in
  # the broker's rolling window — choke on the mixed types.
  # Normalize to a plain integer here so the rest of the system
  # never has to think about it.
  defp to_int(nil), do: 0
  defp to_int(%Decimal{} = d), do: Decimal.to_integer(d)
  defp to_int(n) when is_integer(n), do: n
  defp to_int(n) when is_float(n), do: trunc(n)

  defp active_batches do
    from(b in Batch, where: b.status in ["pending", "running"])
    |> Repo.aggregate(:count, :id)
  end
end
