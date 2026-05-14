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
  alias Server.{Batch, BatchContribution, PolicySnapshot, Repo, BrokerWorker, WorkerNode}

  # Smaller chunks = more frequent worker contributions = more
  # granular dashboard updates and per-chunk resilience on worker
  # failure. Each chunk evaluates `chunk_size` candidates × `n_episodes`
  # rollouts, so 10 candidates at the typical n_episodes=30 = 300
  # episodes ≈ 30 s of work per chunk on one core. See
  # docs/streaming-cegar.md (Layer 1).
  @default_chunk_size 10

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
          submitter: submitter,
          experiment_id: extract_experiment_id(payload)
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

      # Postgres caps a single statement at 65535 bind parameters.
      # An Oban.Job carries ~30 columns, so we keep each insert_all
      # call comfortably under that with batches of 500 jobs.
      if total_chunks > 0 do
        job_changesets
        |> Enum.chunk_every(500)
        |> Enum.each(fn batch_chunks ->
          _jobs = Oban.insert_all(batch_chunks)
        end)
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

  @doc """
  Lightweight batch progress lookup — selects only small columns,
  never the (potentially many-MB) `results` jsonb-array. Used by
  master polling to avoid streaming every accumulated chunk's
  payload back across the wire on every poll.
  """
  def get_batch_progress(batch_id) do
    query =
      from(b in Batch,
        where: b.id == ^batch_id,
        select: %{
          id: b.id,
          name: b.name,
          env_name: b.env_name,
          cmd: b.cmd,
          status: b.status,
          total_chunks: b.total_chunks,
          completed_chunks: b.completed_chunks,
          best_reward: b.best_reward,
          baseline_reward: b.baseline_reward,
          inserted_at: b.inserted_at,
          completed_at: b.completed_at
        }
      )

    case Repo.one(query) do
      nil -> {:error, :not_found}
      row -> {:ok, row}
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
  Atomically claim a chunk, fair-sharing across concurrently-active
  batches. Marks the Oban job as `executing` with the worker_id
  recorded; Oban Lifeline will rescue it if no completion arrives
  within `rescue_after`. Returns `{:ok, payload}` or `:empty`.

  ## Fairness

  We rank each batch's available chunks by `scheduled_at` (oldest
  first) using `ROW_NUMBER() OVER (PARTITION BY batch_id ...)`, then
  pick the chunk whose rank-within-its-batch is smallest. That
  interleaves batches naturally: with a 7338-chunk batch already
  in flight, a new 1000-chunk batch immediately gets every other
  claim instead of waiting in line. Priority is still honored as
  the primary tiebreaker so an opt-in "high-priority" batch can
  jump ahead.
  """
  def claim_chunk(worker_id) do
    # Within the same rank-within-batch, break ties RANDOMLY across
    # batches so concurrent batches get equal-share of worker time
    # rather than letting whichever batch was submitted first
    # consume all worker capacity (older `scheduled_at` would
    # always win otherwise).
    sql = """
    WITH ranked AS (
      SELECT
        id,
        priority,
        ROW_NUMBER() OVER (
          PARTITION BY (args->>'batch_id')
          ORDER BY scheduled_at ASC, id ASC
        ) AS rn
      FROM oban_jobs
      WHERE state = 'available' AND queue = 'chunks'
    )
    SELECT j.id
    FROM oban_jobs j
    JOIN ranked r ON r.id = j.id
    WHERE j.state = 'available' AND j.queue = 'chunks'
    ORDER BY j.priority ASC, r.rn ASC, random()
    LIMIT 1
    FOR UPDATE OF j SKIP LOCKED
    """

    Repo.transaction(fn ->
      case Repo.query!(sql) do
        %Postgrex.Result{rows: [[job_id]]} ->
          job = Repo.get!(Oban.Job, job_id)
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

        %Postgrex.Result{rows: []} ->
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
          chunk_index = job.args["chunk_index"]
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
                push: [results: ^%{"chunk_index" => chunk_index, "items" => results}]
              ]
            )
            |> Repo.update_all([])

          # Refresh the cached reward summary. We do this in a
          # separate UPDATE because conditional GREATEST/COALESCE
          # arithmetic doesn't fit cleanly in the Ecto `inc/push`
          # form above. Cheap (single row, indexed by PK) and runs
          # inside the same transaction.
          maybe_update_score_summary(batch_id, job.args["cmd"], chunk_index, results)

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

  # Roll the new chunk's rewards into the batch's cached aggregates.
  #
  #   * `best_reward`     — running max across all chunks. Updated
  #     atomically via `GREATEST(COALESCE(best_reward, x), x)` so
  #     concurrent chunk submissions can't lose updates.
  #   * `baseline_reward` — for `score_bit` batches, set once when
  #     chunk_index == 0 lands. The master prepends the baseline
  #     candidate to its candidate list, so item 0 of chunk 0 IS
  #     the baseline reward.
  #
  # Layer 1b additions (docs/streaming-cegar.md):
  #
  #   * `n_results`       — running count of candidate evaluations
  #     rolled in, incremented by the number of rewardful items in
  #     this chunk.
  #   * `sum_reward`      — running sum, used together with `n_results`
  #     to derive the live mean for SSE / dashboard.
  #   * `sum_sq_reward`   — running sum of squares for live stddev.
  #   * `last_result_at`  — wall-clock of this update, used by
  #     `MetricsBroker` to compute "candidates / min" rates.
  #
  # All five columns are updated in one atomic SQL statement so
  # concurrent chunk submits can't desync them. No-ops cleanly for
  # `collect_states` results (no `reward` field → counts/sums = 0).
  defp maybe_update_score_summary(batch_id, cmd, chunk_index, results) do
    chunk_max = chunk_max_reward(results)
    {chunk_n, chunk_sum, chunk_sum_sq} = chunk_reward_aggregates(results)

    baseline_reward =
      if cmd == "score_bit" and chunk_index == 0 do
        case results do
          [%{"reward" => r} | _] when is_number(r) -> r * 1.0
          _ -> nil
        end
      end

    cond do
      is_nil(chunk_max) and is_nil(baseline_reward) and chunk_n == 0 ->
        :ok

      true ->
        # Single UPDATE; arguments that are NULL leave their column
        # untouched. Atomic against concurrent chunk submissions.
        sql = """
        UPDATE batches
        SET
          best_reward = CASE
            WHEN $1::float8 IS NULL THEN best_reward
            ELSE GREATEST(COALESCE(best_reward, $1::float8), $1::float8)
          END,
          baseline_reward = COALESCE(baseline_reward, $2::float8),
          n_results     = COALESCE(n_results, 0)     + $4::integer,
          sum_reward    = COALESCE(sum_reward, 0.0)  + $5::float8,
          sum_sq_reward = COALESCE(sum_sq_reward, 0.0) + $6::float8,
          last_result_at = CASE
            WHEN $4::integer = 0 THEN last_result_at
            ELSE $7::timestamptz
          END
        WHERE id = $3
        """

        Repo.query!(sql, [
          chunk_max,
          baseline_reward,
          batch_id,
          chunk_n,
          chunk_sum,
          chunk_sum_sq,
          DateTime.utc_now()
        ])

        :ok
    end
  end

  # Max reward over a chunk's items. Items lacking a numeric
  # `reward` (e.g. `collect_states` returns `states`/`success`)
  # contribute nothing, and an all-rewardless chunk yields `nil`.
  defp chunk_max_reward(results) when is_list(results) do
    Enum.reduce(results, nil, fn
      %{"reward" => r}, best when is_number(r) ->
        if is_nil(best) or r > best, do: r * 1.0, else: best

      _, best ->
        best
    end)
  end

  defp chunk_max_reward(_), do: nil

  # Count / sum / sum-of-squares over a chunk's rewards. Single pass
  # so we don't traverse the list three times. Items without a
  # numeric `reward` are ignored (e.g. `collect_states` results
  # carry `states` / `success` instead).
  defp chunk_reward_aggregates(results) when is_list(results) do
    Enum.reduce(results, {0, 0.0, 0.0}, fn
      %{"reward" => r}, {n, s, sq} when is_number(r) ->
        rf = r * 1.0
        {n + 1, s + rf, sq + rf * rf}

      _, acc ->
        acc
    end)
  end

  defp chunk_reward_aggregates(_), do: {0, 0.0, 0.0}

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

  @doc """
  Per-environment summary for the public landing page.

  Returns a list of `%{env_name, active, best_reward, latest, ...}`
  rows, one per env that has either an in-flight batch or a
  completed `score_bit` batch in history. Drives the "Active
  experiments" widget at https://synthex.fit.

  Shape:

      [
        %{
          env_name: "Ant-v5",
          active: %{
            batch_id: "...",
            name: "...",
            cmd: "score_bit",
            target_bit: 4,
            total_chunks: 7338,
            completed_chunks: 164,
            progress: 0.022,
            current_best_reward: -42.5,
            started_at: ~U[...],
            elapsed_seconds: 9000
          },                              # nil if no batch in flight
          best_reward: -38.2,             # all-time max across env's score_bit batches
          latest: %{                      # most recent COMPLETED score_bit
            batch_id: "...",
            best_reward: -38.2,
            baseline_reward: -52.0,
            delta: 13.8,
            target_bit: 3,
            completed_at: ~U[...]
          },                              # nil if no completed batches yet
          completed_batches: 12,
          total_batches: 13               # active + completed (any cmd)
        },
        ...
      ]

  Cached upstream (HTTP cache-control 15s).
  """
  def experiments_summary do
    # 1) Per-env aggregates over completed score_bit batches
    # `Synthex.Hub.Client.score_bit/3` splits large candidate lists
    # into N HTTP sub-batches named "<base>-part0", ..., "<base>-partK".
    # A logical CEGAR bit search corresponds to ONE such <base>, not
    # to a single Batch row — so naive counts here would tally each
    # sub-batch separately and (worse) declare bit-search complete
    # the moment a small trailing part finishes. We aggregate by
    # stripped name and only count a logical group as "complete"
    # when ALL its sub-batches are complete.
    history = compute_history_per_env()

    # 2) Per-env count of all batches (any status, any cmd) — useful
    #    so we surface envs that exist but haven't completed a
    #    score_bit batch yet.
    totals =
      from(b in Batch,
        group_by: b.env_name,
        select: {b.env_name, count(b.id)}
      )
      |> Repo.all()
      |> Map.new()

    # 3) Latest completed score_bit batch per env. There are only a
    #    handful of envs (1 per ongoing experiment), so a per-env
    #    fetch is cheaper than a window-function gymnastic in Ecto.
    latest_by_env =
      Map.keys(history)
      |> Enum.reduce(%{}, fn env, acc ->
        case latest_completed_score_bit(env) do
          nil -> acc
          batch -> Map.put(acc, env, batch)
        end
      end)

    # 4) Active (pending or running) batches. Group sub-batches by
    #    logical name (stripping `-partN` suffix added by
    #    `Synthex.Hub.Client.score_bit/3` when it splits a candidate
    #    list across multiple HTTP submits) so the landing page
    #    shows ONE row per logical score_bit, with aggregated
    #    chunk counts AND aggregated liveness.
    #
    #    Why this matters: without grouping, the dashboard always
    #    picks the newest sub-batch, which by construction is the
    #    LAST one the master will get around to polling — so its
    #    master_polled_at stays NULL for hours and the card flips
    #    to "stalled" even though the master is healthily working
    #    its way through earlier sub-batches.
    actives =
      from(b in Batch,
        where: b.status in ["pending", "running"],
        order_by: [asc: b.inserted_at]
      )
      |> Repo.all()

    active_by_env =
      actives
      |> Enum.group_by(& &1.env_name)
      |> Map.new(fn {env, env_batches} ->
        {env, aggregate_active_subbatches(env_batches)}
      end)

    # 5) Stitch.
    env_names =
      [Map.keys(history), Map.keys(active_by_env), Map.keys(totals)]
      |> List.flatten()
      |> Enum.uniq()
      |> Enum.sort()

    now = DateTime.utc_now()

    Enum.map(env_names, fn env ->
      hist = Map.get(history, env)
      active = Map.get(active_by_env, env)
      latest = Map.get(latest_by_env, env)

      %{
        env_name: env,
        active: render_active(active, now),
        best_reward: hist && hist.best_reward,
        latest: render_latest(latest),
        completed_batches: (hist && hist.completed_batches) || 0,
        total_batches: Map.get(totals, env, 0)
      }
    end)
  end

  defp render_active(nil, _now), do: nil

  defp render_active(%{} = agg, now) do
    progress =
      if agg.total_chunks > 0,
        do: agg.completed_chunks / agg.total_chunks,
        else: 0.0

    elapsed =
      case agg.inserted_at do
        %DateTime{} = t -> DateTime.diff(now, t, :second)
        _ -> nil
      end

    {health, polled_ago} = compute_health_from(agg.master_polled_at, agg.inserted_at, now)

    %{
      batch_id: agg.id,
      name: agg.name,
      cmd: agg.cmd,
      target_bit: agg.target_bit,
      total_chunks: agg.total_chunks,
      completed_chunks: agg.completed_chunks,
      progress: progress,
      current_best_reward: agg.best_reward,
      baseline_reward: agg.baseline_reward,
      started_at: agg.inserted_at,
      elapsed_seconds: elapsed,
      health: health,
      master_polled_seconds_ago: polled_ago,
      master_polled_at: agg.master_polled_at,
      subbatch_count: agg.subbatch_count
    }
  end

  # Group active batches by logical name (strip `-partN`) and
  # collapse each group into a single aggregate that mirrors the
  # %Batch{} fields render_active/2 needs. For non-split batches
  # the group has size 1 and aggregation is a no-op.
  defp aggregate_active_subbatches(batches) do
    batches
    |> Enum.group_by(fn b -> strip_part_suffix(b.name) end)
    |> Map.values()
    |> Enum.max_by(&group_priority/1)
    |> aggregate_group()
  end

  defp strip_part_suffix(name) when is_binary(name) do
    Regex.replace(~r/-part\d+$/, name, "")
  end

  defp strip_part_suffix(other), do: other

  # Choose which logical batch to surface per env. Preference:
  #   1. groups with ANY recent master_polled_at (alive master)
  #   2. groups inserted most recently (newest experiment)
  defp group_priority(batches) do
    max_polled =
      batches
      |> Enum.map(& &1.master_polled_at)
      |> Enum.reject(&is_nil/1)
      |> case do
        [] -> ~U[1970-01-01 00:00:00.000000Z]
        xs -> Enum.max(xs, DateTime)
      end

    max_inserted =
      batches
      |> Enum.map(& &1.inserted_at)
      |> Enum.max(DateTime)

    {max_polled, max_inserted}
  end

  defp aggregate_group(batches) do
    # Representative = oldest sub-batch (sub-batch 0): the one the
    # master submitted and polls first, and the one that carries
    # the right `cmd`, `payload`, `started_at` semantics. We then
    # overwrite chunk counts and poll timestamps with cross-group
    # aggregates so the dashboard reflects the WHOLE logical batch.
    rep = Enum.min_by(batches, & &1.inserted_at, DateTime)

    total_chunks = Enum.reduce(batches, 0, &(&1.total_chunks + &2))
    completed = Enum.reduce(batches, 0, &(&1.completed_chunks + &2))

    best_reward =
      batches
      |> Enum.map(& &1.best_reward)
      |> Enum.reject(&is_nil/1)
      |> case do
        [] -> nil
        xs -> Enum.max(xs)
      end

    max_polled =
      batches
      |> Enum.map(& &1.master_polled_at)
      |> Enum.reject(&is_nil/1)
      |> case do
        [] -> nil
        xs -> Enum.max(xs, DateTime)
      end

    %{
      id: rep.id,
      name: strip_part_suffix(rep.name),
      cmd: rep.cmd,
      target_bit: get_in(rep.payload, ["target_bit"]),
      total_chunks: total_chunks,
      completed_chunks: completed,
      best_reward: best_reward,
      baseline_reward: rep.baseline_reward,
      inserted_at: rep.inserted_at,
      master_polled_at: max_polled,
      subbatch_count: length(batches)
    }
  end

  # Liveness derived from the master's poll cadence. Masters call
  # `GET /api/master/batches/:id` every few seconds while a batch is
  # in flight; the router stamps `master_polled_at` on each call.
  #
  #   :healthy     — polled within @stalled_threshold_seconds
  #   :stalled     — last poll older than the threshold
  #   :no_poll_yet — batch exists but has never been polled AND has
  #                  been around for at least @stalled_threshold_seconds
  #                  (so we don't flap on freshly-created batches that
  #                  haven't had a chance to be polled once)
  @stalled_threshold_seconds 300

  defp compute_health_from(nil, inserted_at, now) do
    age = DateTime.diff(now, inserted_at, :second)
    if age >= @stalled_threshold_seconds,
      do: {"no_poll_yet", nil},
      else: {"healthy", nil}
  end

  defp compute_health_from(%DateTime{} = polled_at, _inserted_at, now) do
    secs = DateTime.diff(now, polled_at, :second)
    health = if secs > @stalled_threshold_seconds, do: "stalled", else: "healthy"
    {health, secs}
  end

  defp render_latest(nil), do: nil

  defp render_latest(%{} = b) do
    delta =
      cond do
        is_number(b.best_reward) and is_number(b.baseline_reward) ->
          b.best_reward - b.baseline_reward

        true ->
          nil
      end

    %{
      batch_id: b.id,
      name: b.name,
      target_bit: get_in(b.payload || %{}, ["target_bit"]),
      best_reward: b.best_reward,
      baseline_reward: b.baseline_reward,
      delta: delta,
      completed_at: b.completed_at
    }
  end

  defp latest_completed_score_bit(env_name) do
    # Same logical-batch aggregation as `compute_history_per_env/0`:
    # collapse `<base>-partN` sub-batches into one logical batch and
    # only count those whose ALL parts have status="completed".
    # Return a synthesized struct with aggregated chunk counts and
    # the max best_reward across the group — that's the actual
    # "best candidate found during this bit search", not just one
    # tail sub-batch's score.
    rows =
      from(b in Batch,
        where:
          b.cmd == "score_bit" and b.env_name == ^env_name,
        select: %{
          status: b.status,
          name: b.name,
          best_reward: b.best_reward,
          baseline_reward: b.baseline_reward,
          completed_at: b.completed_at,
          inserted_at: b.inserted_at,
          payload: b.payload,
          id: b.id,
          total_chunks: b.total_chunks,
          completed_chunks: b.completed_chunks
        }
      )
      |> Repo.all()

    rows
    |> Enum.group_by(fn r -> strip_part_suffix(r.name) end)
    |> Enum.filter(fn {_logical, parts} ->
      Enum.all?(parts, &(&1.status == "completed"))
    end)
    |> case do
      [] ->
        nil

      groups ->
        groups
        |> Enum.map(fn {_logical, parts} -> aggregate_completed_group(parts) end)
        |> Enum.max_by(fn agg -> agg.completed_at end, DateTime, fn -> nil end)
    end
  end

  defp aggregate_completed_group(parts) do
    rep = Enum.min_by(parts, & &1.inserted_at, DateTime)
    best = parts |> Enum.map(& &1.best_reward) |> Enum.reject(&is_nil/1) |> safe_max()

    last_completed =
      parts |> Enum.map(& &1.completed_at) |> Enum.reject(&is_nil/1) |> safe_max_dt()

    %{
      id: rep.id,
      name: strip_part_suffix(rep.name),
      best_reward: best,
      baseline_reward: rep.baseline_reward,
      completed_at: last_completed,
      inserted_at: rep.inserted_at,
      payload: rep.payload,
      total_chunks: Enum.reduce(parts, 0, &(&1.total_chunks + &2)),
      completed_chunks: Enum.reduce(parts, 0, &(&1.completed_chunks + &2))
    }
  end

  defp compute_history_per_env do
    from(b in Batch,
      where: b.cmd == "score_bit",
      select: %{
        env_name: b.env_name,
        status: b.status,
        name: b.name,
        best_reward: b.best_reward
      }
    )
    |> Repo.all()
    |> Enum.group_by(& &1.env_name)
    |> Map.new(fn {env, env_rows} ->
      completed_logical =
        env_rows
        |> Enum.group_by(fn r -> strip_part_suffix(r.name) end)
        |> Enum.filter(fn {_logical, parts} ->
          Enum.all?(parts, &(&1.status == "completed"))
        end)

      best =
        completed_logical
        |> Enum.flat_map(fn {_logical, parts} ->
          Enum.map(parts, & &1.best_reward)
        end)
        |> Enum.reject(&is_nil/1)
        |> safe_max()

      {env,
       %{
         env_name: env,
         best_reward: best,
         completed_batches: length(completed_logical)
       }}
    end)
  end

  defp safe_max([]), do: nil
  defp safe_max(xs), do: Enum.max(xs)

  defp safe_max_dt([]), do: nil
  defp safe_max_dt(xs), do: Enum.max(xs, DateTime)

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

  # Pull `experiment_id` out of the master's submit payload so the
  # `batches.experiment_id` FK is set on every batch the new
  # Oban-master refactor spawns. Legacy (laptop-driven) masters
  # don't include this key, in which case the FK stays NULL and
  # the batch behaves exactly as before.
  defp extract_experiment_id(payload) do
    case Map.get(payload, "experiment_id") do
      id when is_binary(id) and byte_size(id) > 0 -> id
      _ -> nil
    end
  end

  defp default_chunk_size do
    Application.get_env(:server, :default_chunk_size, @default_chunk_size)
  end

  @doc """
  Stamp `master_polled_at = now()` on a batch. Called from the
  router on every master poll so the landing page can tell when a
  master has gone silent (process crashed, network died, ...) even
  while the batch is still marked active.

  Returns `:ok` regardless — failure to update the timestamp must
  never block the poll response.
  """
  def touch_master_poll(batch_id) when is_binary(batch_id) do
    try do
      from(b in Batch, where: b.id == ^batch_id)
      |> Repo.update_all(set: [master_polled_at: DateTime.utc_now()])
      :ok
    rescue
      _ -> :ok
    end
  end

  # ── Policy snapshots ────────────────────────────────────────

  @doc """
  Upsert the latest policy snapshot for an environment. Called by
  masters via `POST /api/master/policy-snapshots` whenever the
  CEGAR loop accepts a new bit.
  """
  def upsert_policy_snapshot(attrs, opts \\ []) do
    submitter = Keyword.get(opts, :submitter)

    attrs =
      attrs
      |> Map.put_new("submitter", submitter)
      |> stringify_top_level_keys()

    existing =
      case Map.get(attrs, "env_name") do
        nil -> %PolicySnapshot{}
        env -> Repo.get(PolicySnapshot, env) || %PolicySnapshot{}
      end

    existing
    |> PolicySnapshot.changeset(attrs)
    |> Repo.insert_or_update()
  end

  @doc "Latest snapshot for an env, or `{:error, :not_found}`."
  def get_policy_snapshot(env_name) do
    case Repo.get(PolicySnapshot, env_name) do
      nil -> {:error, :not_found}
      snapshot -> {:ok, snapshot}
    end
  end

  @doc """
  All snapshots, newest-first. Used for a future "all current
  policies" overview endpoint.
  """
  def list_policy_snapshots(limit \\ 50) do
    from(s in PolicySnapshot,
      order_by: [desc: s.updated_at],
      limit: ^limit
    )
    |> Repo.all()
  end

  # Plug.Parsers decodes JSON keys as strings; PolicySnapshot.changeset
  # uses `cast/3` which accepts string or atom keys, but we
  # normalize defensively so atom-keyed callers (tests, scripts)
  # work too.
  defp stringify_top_level_keys(map) when is_map(map) do
    Map.new(map, fn
      {k, v} when is_atom(k) -> {Atom.to_string(k), v}
      kv -> kv
    end)
  end
end
