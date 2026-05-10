defmodule Server.Queue do
  @moduledoc """
  Oban-backed batch and chunk queue with HTTP-pull semantics.

  ## Lifecycle

      master submits batch
        ↓
      `submit_batch/1` (transactional)
        - INSERT batch row
        - INSERT N Oban.Job rows (queue: :chunks, state: "available")

      external worker polls `/api/worker/jobs/request`
        ↓
      `claim_chunk/1` (transactional, SKIP LOCKED)
        - SELECT one available chunk FOR UPDATE SKIP LOCKED
        - UPDATE state = "executing", attempted_at = now(), attempted_by = [worker_id]
        - return chunk payload

      worker submits results to `/api/worker/jobs/submit`
        ↓
      `complete_chunk/3` (transactional)
        - if state != "executing": idempotent no-op
        - UPDATE state = "completed", completed_at = now()
        - increment batch.completed_chunks; if last, finalize batch

      worker dies / segfaults / never returns
        ↓
      Oban.Plugins.Lifeline (every 60s by default) reaps any
      "executing" job whose attempted_at > rescue_after threshold,
      moves it back to "available" → next worker picks it up
  """

  import Ecto.Query
  alias Server.{Batch, BatchContribution, Repo, BrokerWorker, WorkerNode}

  @default_chunk_size 100

  # ── Master API ──────────────────────────────────────────────

  @doc """
  Submits a batch. Splits `candidates` into chunks of `chunk_size`,
  inserts each chunk as an Oban job, and returns `{:ok, batch}`.
  """
  def submit_batch(payload, opts \\ []) do
    submitter = Keyword.get(opts, :submitter)
    candidates = Map.get(payload, "candidates") || Map.get(payload, :candidates) || []
    chunk_size = Map.get(payload, "chunk_size", default_chunk_size())
    chunks = Enum.chunk_every(candidates, chunk_size)
    total_chunks = length(chunks)

    batch_id = "batch_#{:erlang.system_time(:millisecond)}_#{:erlang.unique_integer([:positive])}"

    Repo.transaction(fn ->
      {:ok, batch} =
        %Batch{}
        |> Batch.changeset(%{
          id: batch_id,
          name: Map.get(payload, "name"),
          env_name: Map.get(payload, "env_name", "unknown"),
          cmd: Map.get(payload, "cmd", "score_bit"),
          payload: scrub_payload(payload),
          total_chunks: total_chunks,
          status: if(total_chunks == 0, do: "completed", else: "pending"),
          submitter: submitter
        })
        |> Repo.insert()

      # Build all chunk Oban jobs as changesets and insert them in one
      # multi-VALUES INSERT. With ~1000+ chunks, the previous one-job-
      # per-Oban.insert approach took >15s of round-trips against
      # Neon and timed out the DB connection on large submissions.
      params = Map.drop(payload, ["candidates", "chunk_size", "name"])

      job_changesets =
        Enum.with_index(chunks, fn chunk, idx ->
          chunk_id = "#{batch_id}_chunk_#{idx}"

          args = %{
            "chunk_id" => chunk_id,
            "batch_id" => batch_id,
            "chunk_index" => idx,
            "cmd" => batch.cmd,
            "env_name" => batch.env_name,
            "candidates" => chunk,
            "params" => params
          }

          BrokerWorker.new(args, meta: %{"chunk_id" => chunk_id, "batch_id" => batch_id})
        end)

      if total_chunks > 0 do
        {_count, _jobs} = Oban.insert_all(job_changesets)
      end

      if total_chunks == 0 do
        Repo.update!(Batch.changeset(batch, %{completed_at: DateTime.utc_now()}))
      end

      batch
    end)
  end

  @doc "Look up a batch by id, with progress and (if completed) aggregated results."
  def get_batch(batch_id) do
    case Repo.get(Batch, batch_id) do
      nil -> {:error, :not_found}
      batch -> {:ok, batch}
    end
  end

  @doc "List recent batches, newest first."
  def list_batches(limit \\ 50) do
    Batch
    |> order_by(desc: :inserted_at)
    |> limit(^limit)
    |> Repo.all()
  end

  # ── Worker API ──────────────────────────────────────────────

  @doc """
  Atomically claim the oldest available chunk. Marks the Oban job
  as `executing` with the worker_id recorded; Oban Lifeline will
  rescue it if no completion arrives within `rescue_after`.
  Returns `{:ok, payload}` or `:empty`.
  """
  def claim_chunk(worker_id) do
    Repo.transaction(fn ->
      job =
        from(j in Oban.Job,
          where: j.state == "available" and j.queue == "chunks",
          order_by: [asc: j.priority, asc: j.scheduled_at, asc: j.id],
          limit: 1,
          lock: "FOR UPDATE SKIP LOCKED"
        )
        |> Repo.one()

      if job do
        now = DateTime.utc_now()

        {:ok, claimed} =
          Repo.update(
            Ecto.Changeset.change(job,
              state: "executing",
              attempted_at: now,
              attempted_by: [worker_id]
            )
          )

        chunk_payload_for_worker(claimed)
      else
        Repo.rollback(:empty)
      end
    end)
    |> case do
      {:ok, payload} -> {:ok, payload}
      {:error, :empty} -> :empty
    end
  end

  @doc """
  Mark a chunk completed (idempotent). Stores the per-chunk results,
  increments the batch's completed_chunks, and finalizes the batch
  when the last chunk lands.
  """
  def complete_chunk(chunk_id, results, worker_id) do
    Repo.transaction(fn ->
      job =
        from(j in Oban.Job,
          where: fragment("? ->> 'chunk_id' = ?", j.args, ^chunk_id),
          lock: "FOR UPDATE"
        )
        |> Repo.one()

      cond do
        is_nil(job) ->
          Repo.rollback(:not_found)

        job.state == "completed" ->
          # idempotent: a duplicate submission, ignore
          :already_completed

        true ->
          batch_id = job.args["batch_id"]
          updated_args = Map.put(job.args, "results", results)
          now = DateTime.utc_now()

          {:ok, _} =
            Repo.update(
              Ecto.Changeset.change(job,
                state: "completed",
                completed_at: now,
                args: updated_args
              )
            )

          {1, _} =
            from(b in Batch,
              where: b.id == ^batch_id,
              update: [
                inc: [completed_chunks: 1],
                push: [results: ^%{"chunk_index" => job.args["chunk_index"], "items" => results}]
              ]
            )
            |> Repo.update_all([])

          batch = Repo.get!(Batch, batch_id)

          if batch.completed_chunks >= batch.total_chunks do
            Repo.update!(
              Batch.changeset(batch, %{status: "completed", completed_at: now})
            )
          end

          if worker_id do
            from(w in WorkerNode, where: w.id == ^worker_id)
            |> Repo.update_all(inc: [jobs_completed: 1, candidates_evaluated: length(results)])

            record_contribution(batch_id, worker_id, results, now)
          end

          :ok
      end
    end)
  end

  # Upsert one row per (batch_id, worker_id). On first chunk: create
  # with counters = 1 / length(results). On subsequent chunks for the
  # same pair: increment counters and bump last_chunk_at. We always
  # refresh `display_name` from WorkerNode so renames take effect on
  # the next chunk.
  defp record_contribution(batch_id, worker_id, results, now) do
    display_name = lookup_display_name(worker_id)
    n = length(results)

    %BatchContribution{}
    |> BatchContribution.changeset(%{
      batch_id: batch_id,
      worker_id: worker_id,
      display_name: display_name,
      chunks_completed: 1,
      candidates_evaluated: n,
      first_chunk_at: now,
      last_chunk_at: now
    })
    |> Repo.insert(
      on_conflict: [
        inc: [chunks_completed: 1, candidates_evaluated: n],
        set: [last_chunk_at: now, display_name: display_name]
      ],
      conflict_target: [:batch_id, :worker_id]
    )
  end

  defp lookup_display_name(worker_id) do
    case Repo.get(WorkerNode, worker_id) do
      %WorkerNode{name: name} when is_binary(name) and byte_size(name) > 0 -> name
      _ -> "anonymous"
    end
  end

  # ── Worker registration ─────────────────────────────────────

  def register_worker(attrs) do
    now = DateTime.utc_now()

    attrs =
      attrs
      |> Map.put_new("registered_at", now)
      |> Map.put("last_heartbeat_at", now)
      |> Map.put("status", "active")

    case Repo.get(WorkerNode, attrs["id"]) do
      nil -> %WorkerNode{}
      existing -> existing
    end
    |> WorkerNode.changeset(attrs)
    |> Repo.insert_or_update()
  end

  def heartbeat(worker_id) do
    from(w in WorkerNode, where: w.id == ^worker_id)
    |> Repo.update_all(set: [last_heartbeat_at: DateTime.utc_now(), status: "active"])
    |> case do
      {0, _} -> {:error, :unknown_worker}
      {_, _} -> :ok
    end
  end

  def list_active_workers(timeout_secs \\ 120) do
    cutoff = DateTime.add(DateTime.utc_now(), -timeout_secs, :second)

    from(w in WorkerNode,
      where: w.last_heartbeat_at >= ^cutoff,
      order_by: [desc: w.last_heartbeat_at]
    )
    |> Repo.all()
  end

  def mark_inactive_workers(timeout_secs) do
    cutoff = DateTime.add(DateTime.utc_now(), -timeout_secs, :second)

    from(w in WorkerNode,
      where: w.last_heartbeat_at < ^cutoff and w.status == "active"
    )
    |> Repo.update_all(set: [status: "inactive"])
  end

  # ── Status ──────────────────────────────────────────────────

  def status do
    %{
      workers:
        from(w in WorkerNode, where: w.status == "active") |> Repo.aggregate(:count, :id),
      total_cores:
        from(w in WorkerNode, where: w.status == "active")
        |> Repo.aggregate(:sum, :pool_size) || 0,
      jobs_available:
        from(j in Oban.Job, where: j.state == "available" and j.queue == "chunks")
        |> Repo.aggregate(:count, :id),
      jobs_executing:
        from(j in Oban.Job, where: j.state == "executing" and j.queue == "chunks")
        |> Repo.aggregate(:count, :id),
      jobs_completed:
        from(j in Oban.Job, where: j.state == "completed" and j.queue == "chunks")
        |> Repo.aggregate(:count, :id),
      candidates_evaluated: candidates_evaluated(),
      batches_pending:
        from(b in Batch, where: b.status in ["pending", "running"]) |> Repo.aggregate(:count, :id),
      batches_completed:
        from(b in Batch, where: b.status == "completed") |> Repo.aggregate(:count, :id)
    }
  end

  @doc """
  Aggregate stats safe for unauthenticated public consumption — drives
  the landing-page counters at synthex.fit. No raw IDs, payloads, or
  worker hostnames; just totals.
  """
  def public_status do
    workers_count =
      from(w in WorkerNode, where: w.status == "active")
      |> Repo.aggregate(:count, :id)

    total_cores =
      (from(w in WorkerNode, where: w.status == "active")
       |> Repo.aggregate(:sum, :pool_size)) || 0

    experiments_completed =
      from(b in Batch, where: b.status == "completed")
      |> Repo.aggregate(:count, :id)

    %{
      active_workers: workers_count,
      total_cores: total_cores,
      candidates_evaluated: candidates_evaluated(),
      experiments_completed: experiments_completed
    }
  end

  # Persistent running total — survives Oban's pruning of completed
  # jobs. Incremented by length(results) on every chunk submission.
  defp candidates_evaluated do
    Repo.aggregate(WorkerNode, :sum, :candidates_evaluated) || 0
  end

  # ── Leaderboard ─────────────────────────────────────────────

  @doc """
  All-time top contributors, grouped by `display_name`. All
  workers whose name is `"anonymous"` collapse into a single row.
  Returns `[%{name, candidates_evaluated, chunks_completed,
  experiments_contributed, last_seen_at}, ...]`.
  """
  def leaderboard(opts \\ []) do
    limit = Keyword.get(opts, :limit, 20)

    from(c in BatchContribution,
      group_by: c.display_name,
      select: %{
        name: c.display_name,
        candidates_evaluated: sum(c.candidates_evaluated),
        chunks_completed: sum(c.chunks_completed),
        experiments_contributed: count(c.batch_id, :distinct),
        last_seen_at: max(c.last_chunk_at)
      },
      order_by: [desc: sum(c.candidates_evaluated)],
      limit: ^limit
    )
    |> Repo.all()
  end

  @doc """
  Per-batch contributors, grouped by `display_name`. Useful for
  showing "who helped run THIS experiment" once a batch is in flight
  or completed.
  """
  def batch_contributors(batch_id, opts \\ []) do
    limit = Keyword.get(opts, :limit, 50)

    from(c in BatchContribution,
      where: c.batch_id == ^batch_id,
      group_by: c.display_name,
      select: %{
        name: c.display_name,
        candidates_evaluated: sum(c.candidates_evaluated),
        chunks_completed: sum(c.chunks_completed),
        first_chunk_at: min(c.first_chunk_at),
        last_chunk_at: max(c.last_chunk_at)
      },
      order_by: [desc: sum(c.candidates_evaluated)],
      limit: ^limit
    )
    |> Repo.all()
  end

  # ── Internals ───────────────────────────────────────────────

  defp chunk_payload_for_worker(job) do
    Map.merge(job.args, %{
      "oban_job_id" => job.id,
      "attempt" => job.attempt,
      "max_attempts" => job.max_attempts
    })
  end

  defp scrub_payload(payload) do
    payload
    |> Map.drop(["candidates"])
    |> Map.put("n_candidates", length(Map.get(payload, "candidates", [])))
  end

  defp default_chunk_size do
    Application.get_env(:server, :default_chunk_size, @default_chunk_size)
  end
end
