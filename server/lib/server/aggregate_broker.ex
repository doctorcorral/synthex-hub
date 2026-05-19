defmodule Server.AggregateBroker do
  @moduledoc """
  Caches per-active-experiment streaming aggregates for the SSE
  feed at `/api/public-status/stream/aggregates`.

  Layer 1c of `docs/streaming-cegar.md`. Sister of
  `Server.MetricsBroker`: same lifecycle (one tick per second, ETS
  cache for read-concurrent SSE clients), but the cached payload is
  per-experiment rather than global.

  ## What's in a snapshot

  Reward fields are *per-episode means* (the candidate score Σ over
  n_episodes rollouts divided by n_episodes). Raw sums live alongside
  under `*_sum` keys for API consumers who want the exact stored
  value. See `Server.Experiments.normalize_reward/2`.

      %{
        experiments: [
          %{
            experiment_id: "…",
            env_name: "Ant-v5",
            n_episodes: 30,
            active_bit: %{
              batch_id: "…",
              target_bit: 5,
              cmd: "score_bit",
              n_results: 318,
              mean: -99.3,          # per episode
              stddev: 4.8,
              best_reward: -94.0,
              baseline_reward: -100.1,
              best_reward_sum: -2820.4,  # raw sum across 30 eps
              completed_chunks: 32,
              total_chunks: 60,
              progress: 0.533,
              results_per_min: 47        # rolling 60-s window
            }
          }
        ],
        ts: "2026-05-14T03:04:05Z"
      }

  ## Why a broker

  Each SSE client polling once/second would otherwise hit Postgres
  every second per viewer; one broker per node keeps DB load O(1)
  in viewer count. The rolling per-batch rate also has to live
  somewhere — we keep a `:queue` of `{ts_ms, n_results}` samples per
  in-flight batch, trimmed to the last `@window_secs` seconds.

  ## Per-experiment flow stats

  Beyond the per-bit `active_bit` payload above, we also compute
  per-experiment flow aggregates and cache them under a separate
  ETS key. These are consumed by `Server.Experiments.compute_health/2`
  to produce honest dashboard health labels — "slow" (chunks
  flowing but no bit yet) vs "idle" (chunks stopped flowing) vs
  "healthy". See `experiment_flow/1` below.

  We sum across ALL in-flight batches for the experiment, not just
  the latest one: Jacobi parallel-bit dispatch keeps N bits open
  concurrently, so the swarm's per-experiment throughput is the
  sum across those N batches' rolling rates.
  """

  use GenServer
  require Logger

  import Ecto.Query

  alias Server.{Batch, Experiment, Experiments, PolicyVersion, Repo}

  @table :server_aggregate_cache
  @refresh_ms 1_000
  @window_secs 60

  # ── public API ──────────────────────────────────────────────────

  def start_link(_opts) do
    GenServer.start_link(__MODULE__, [], name: __MODULE__)
  end

  @doc "Latest cached snapshot (nil before first refresh)."
  def latest do
    case :ets.whereis(@table) do
      :undefined ->
        nil

      _ ->
        case :ets.lookup(@table, :snapshot) do
          [{:snapshot, snap}] -> snap
          [] -> nil
        end
    end
  end

  @doc """
  Per-experiment chunk flow stats — the swarm's collective
  throughput on this experiment summed across all in-flight bits,
  plus wave-level totals (completed + in-flight) for honest ETA
  projection.

  Returns `nil` if the broker has no row for the experiment (either
  it hasn't refreshed yet, or the experiment has no in-flight
  batches right now — e.g. between waves while the controller
  collects states and builds features).

  Shape:

      %{
        chunks_done: 1_828,          # currently in-flight batches only
        chunks_total: 293_488,       # currently in-flight batches only
        chunks_pending: 291_660,
        chunks_per_min: 7,           # rolling 60-s window, summed across bits
        last_progress_at: ~U[…],     # max(batches.last_result_at)
        n_active_bits: 4,            # in-flight batches under this experiment
        # ── wave-scoped totals, for honest "wave completion" ETA ──
        wave_dispatched_chunks: 23_500,  # sum(total_chunks) across all batches
                                         # inserted since the most recent commit
                                         # (or experiment start if no commits).
        wave_done_chunks: 22_900,        # sum(completed_chunks) over the same set
        wave_dispatched_bits: 18          # distinct target_bits seen in this wave
      }

  Why wave-scoped totals matter: each bit in `optimize_bit` issues
  TWO `score_bit` batches (depth-0 atomic search + depth-1
  compound refine), and `Task.async_stream` keeps only `bit_concurrency`
  bits in flight at once. The "in-flight only" totals cycle every
  few hours as bit-groups churn through D0/D1, so an ETA based on
  them oscillates between minutes and hours forever even though
  the true wall-clock to wave completion is days. Wave-scoped
  totals stay monotone within a wave and let `compute_health/2`
  project an honest ETA.
  """
  def experiment_flow(experiment_id) when is_binary(experiment_id) do
    case :ets.whereis(@table) do
      :undefined ->
        nil

      _ ->
        case :ets.lookup(@table, {:flow, experiment_id}) do
          [{_, flow}] -> flow
          [] -> nil
        end
    end
  end

  def experiment_flow(_), do: nil

  # ── GenServer ───────────────────────────────────────────────────

  @impl true
  def init(_) do
    :ets.new(@table, [:named_table, :public, read_concurrency: true])
    Process.send_after(self(), :refresh, 100)
    # `rings`: per-batch rolling `n_results` window (existing).
    # `exp_rings`: per-experiment rolling window of summed completed_chunks
    #   across in-flight batches, used to derive `chunks_per_min`.
    # `prev_flow_keys`: experiment_ids we wrote on the previous tick so
    #   we can evict stale rows whose experiments are no longer active.
    {:ok, %{rings: %{}, exp_rings: %{}, prev_flow_keys: MapSet.new()}}
  end

  @impl true
  def handle_info(:refresh, state) do
    new_state =
      try do
        do_refresh(state)
      rescue
        err ->
          Logger.warning(
            "[AggregateBroker] refresh failed: #{Exception.message(err)}\n" <>
              Exception.format_stacktrace(__STACKTRACE__)
          )

          state
      end

    Process.send_after(self(), :refresh, @refresh_ms)
    {:noreply, new_state}
  end

  # ── internals ───────────────────────────────────────────────────

  defp do_refresh(state) do
    now_ms = System.system_time(:millisecond)
    rows = fetch_active_rows()

    {experiment_payloads, new_rings} =
      Enum.map_reduce(rows, state.rings, fn row, rings_acc ->
        {payload, updated_rings} = render_row(row, rings_acc, now_ms)
        {payload, updated_rings}
      end)

    # Drop rings for batches we no longer track — keeps memory bounded
    # to the active set, not all-time.
    active_batch_ids =
      rows
      |> Enum.map(& &1.batch_id)
      |> MapSet.new()

    pruned_rings =
      new_rings
      |> Enum.filter(fn {batch_id, _ring} -> MapSet.member?(active_batch_ids, batch_id) end)
      |> Map.new()

    snapshot = %{
      experiments: experiment_payloads,
      ts: DateTime.utc_now() |> DateTime.to_iso8601()
    }

    :ets.insert(@table, {:snapshot, snapshot})

    # Per-experiment flow stats — summed across all in-flight bits,
    # plus wave-scoped totals (completed + in-flight batches since
    # the last commit, see `experiment_flow/1`). The wave-scoped
    # totals feed an honest "wave completion" ETA on the dashboard
    # so the user doesn't see "ETA 2h" for days on end.
    inflight_rows = fetch_all_active_batches()
    wave_rows = fetch_wave_batches(inflight_rows)

    {new_exp_rings, current_flow_keys} =
      refresh_flow_cache(inflight_rows, wave_rows, state.exp_rings, now_ms)

    # Evict ETS rows for experiments that disappeared from the active
    # set since the previous tick — otherwise stale flow stats would
    # linger forever for cancelled / completed experiments and trick
    # `compute_health/2` into reporting old throughput numbers.
    stale_keys = MapSet.difference(state.prev_flow_keys, current_flow_keys)
    Enum.each(stale_keys, fn exp_id -> :ets.delete(@table, {:flow, exp_id}) end)

    %{state | rings: pruned_rings, exp_rings: new_exp_rings, prev_flow_keys: current_flow_keys}
  end

  # Same scope as `fetch_active_rows` but returns ALL in-flight
  # batches per experiment, not just the latest. Per-bit aggregates
  # use the latest (one card shows one in-flight bit), but per-experiment
  # flow stats need to sum across every bit currently dispatched —
  # Jacobi keeps N batches open simultaneously.
  defp fetch_all_active_batches do
    from(b in Batch,
      join: e in Experiment,
      on: e.id == b.experiment_id,
      where: b.status in ["pending", "running"] and e.status == "running",
      select: %{
        experiment_id: b.experiment_id,
        batch_id: b.id,
        completed_chunks: b.completed_chunks,
        total_chunks: b.total_chunks,
        last_result_at: b.last_result_at
      }
    )
    |> Repo.all()
  end

  # All `score_bit` batches for each experiment that has in-flight
  # work, restricted to "the current wave": batches inserted since
  # the most recent commit (or since the experiment started if no
  # commits yet). Includes status='completed' batches — they're the
  # work already done in this wave and we need them to compute
  # honest wave-completion ETAs.
  #
  # The wave boundary is `max(experiment.started_at,
  # latest_policy_version.inserted_at)`. We compute it once per
  # active experiment rather than join inline because policy_versions
  # is keyed by (experiment_id, version) and the per-experiment max
  # is cheap to derive in Elixir from a single fetch.
  defp fetch_wave_batches(inflight_rows) do
    active_exp_ids =
      inflight_rows
      |> Enum.map(& &1.experiment_id)
      |> Enum.uniq()

    case active_exp_ids do
      [] ->
        []

      ids ->
        cutoffs = wave_cutoffs(ids)

        cutoff_pairs =
          ids
          |> Enum.flat_map(fn id ->
            case Map.get(cutoffs, id) do
              nil -> []
              ts -> [{id, ts}]
            end
          end)

        case cutoff_pairs do
          [] ->
            []

          pairs ->
            # One query: all score_bit batches whose (experiment_id,
            # inserted_at) is at or after this experiment's wave cutoff.
            # Postgres handles the per-experiment cutoff via a values
            # CTE in the WHERE clause via Ecto fragments.
            exp_ids = Enum.map(pairs, fn {id, _} -> id end)
            cutoff_map = Map.new(pairs)

            from(b in Batch,
              where:
                b.experiment_id in ^exp_ids and
                  b.cmd == "score_bit",
              select: %{
                experiment_id: b.experiment_id,
                batch_id: b.id,
                completed_chunks: b.completed_chunks,
                total_chunks: b.total_chunks,
                status: b.status,
                inserted_at: b.inserted_at,
                target_bit:
                  fragment("(?->>'target_bit')::int", b.payload)
              }
            )
            |> Repo.all()
            |> Enum.filter(fn row ->
              case Map.get(cutoff_map, row.experiment_id) do
                nil -> false
                cutoff -> ndt_geq?(row.inserted_at, cutoff)
              end
            end)
        end
    end
  end

  # Per-experiment wave cutoff: max(experiment.started_at OR
  # inserted_at, max(policy_versions.inserted_at)). Returns a map of
  # experiment_id => NaiveDateTime. Experiments not yet started fall
  # back to inserted_at. We do this in two roundtrips (experiments,
  # policy_versions) which is fine — broker ticks once per second
  # and these tables are small.
  defp wave_cutoffs(experiment_ids) do
    exp_starts =
      from(e in Experiment,
        where: e.id in ^experiment_ids,
        select: {e.id, e.started_at, e.inserted_at}
      )
      |> Repo.all()
      |> Map.new(fn {id, started_at, inserted_at} ->
        {id, started_at || inserted_at}
      end)

    last_commits =
      from(v in PolicyVersion,
        where: v.experiment_id in ^experiment_ids,
        group_by: v.experiment_id,
        select: {v.experiment_id, max(v.inserted_at)}
      )
      |> Repo.all()
      |> Map.new()

    Enum.reduce(experiment_ids, %{}, fn id, acc ->
      start_ts = Map.get(exp_starts, id)
      commit_ts = Map.get(last_commits, id)

      case {start_ts, commit_ts} do
        {nil, nil} -> acc
        {ts, nil} -> Map.put(acc, id, ts)
        {nil, ts} -> Map.put(acc, id, ts)
        {a, b} -> Map.put(acc, id, ndt_max(a, b))
      end
    end)
  end

  defp ndt_max(a, b) do
    if ndt_geq?(a, b), do: a, else: b
  end

  # Compare two timestamps that may arrive as DateTime or NaiveDateTime
  # depending on whether the field is `:utc_datetime_usec` (batches.inserted_at,
  # most policy_versions) or `:naive_datetime` (experiments.started_at).
  # Normalize to NaiveDateTime for the comparison — they're all UTC here.
  defp ndt_geq?(%DateTime{} = a, b), do: ndt_geq?(DateTime.to_naive(a), b)
  defp ndt_geq?(a, %DateTime{} = b), do: ndt_geq?(a, DateTime.to_naive(b))

  defp ndt_geq?(%NaiveDateTime{} = a, %NaiveDateTime{} = b) do
    case NaiveDateTime.compare(a, b) do
      :lt -> false
      _ -> true
    end
  end

  defp refresh_flow_cache(inflight_rows, wave_rows, exp_rings, now_ms) do
    inflight_grouped = Enum.group_by(inflight_rows, & &1.experiment_id)
    wave_grouped = Enum.group_by(wave_rows, & &1.experiment_id)

    Enum.reduce(inflight_grouped, {exp_rings, MapSet.new()}, fn {exp_id, exp_rows},
                                                                {rings_acc, keys_acc} ->
      done = exp_rows |> Enum.map(&(&1.completed_chunks || 0)) |> Enum.sum()
      total = exp_rows |> Enum.map(&(&1.total_chunks || 0)) |> Enum.sum()
      pending = max(total - done, 0)
      n_active_bits = length(exp_rows)

      last_progress_at =
        exp_rows
        |> Enum.map(& &1.last_result_at)
        |> Enum.reject(&is_nil/1)
        |> case do
          [] -> nil
          ts_list -> Enum.max(ts_list, DateTime)
        end

      # Wave-scoped aggregates: completed + in-flight `score_bit`
      # batches in this experiment's current wave. See
      # `fetch_wave_batches/1` for the definition of "wave".
      wave_rows_for_exp = Map.get(wave_grouped, exp_id, [])

      wave_dispatched_chunks =
        wave_rows_for_exp |> Enum.map(&(&1.total_chunks || 0)) |> Enum.sum()

      wave_done_chunks =
        wave_rows_for_exp |> Enum.map(&(&1.completed_chunks || 0)) |> Enum.sum()

      wave_dispatched_bits =
        wave_rows_for_exp
        |> Enum.map(& &1.target_bit)
        |> Enum.reject(&is_nil/1)
        |> Enum.uniq()
        |> length()

      # Reset the ring when `done` drops below the prior sample —
      # that's the wave-boundary case: the previous wave completed,
      # a new wave was dispatched, and `done` (summed across
      # in-flight batches) just snapped back to zero. Without this
      # the rate would briefly read negative.
      prior_ring = Map.get(rings_acc, exp_id, :queue.new())
      base_ring =
        case :queue.peek_r(prior_ring) do
          {:value, {_, prev_done}} when prev_done > done -> :queue.new()
          _ -> prior_ring
        end

      ring =
        base_ring
        |> push_ring(now_ms, done)
        |> trim_ring(now_ms)

      rate = rate_per_minute(ring, now_ms, done)

      flow = %{
        chunks_done: done,
        chunks_total: total,
        chunks_pending: pending,
        chunks_per_min: rate,
        last_progress_at: last_progress_at,
        n_active_bits: n_active_bits,
        wave_dispatched_chunks: wave_dispatched_chunks,
        wave_done_chunks: wave_done_chunks,
        wave_dispatched_bits: wave_dispatched_bits
      }

      :ets.insert(@table, {{:flow, exp_id}, flow})

      {Map.put(rings_acc, exp_id, ring), MapSet.put(keys_acc, exp_id)}
    end)
    |> then(fn {rings, keys} ->
      # Also prune in-memory exp_rings for experiments that fell out
      # of the active set, mirroring what we do for per-batch `rings`.
      pruned = rings |> Enum.filter(fn {k, _} -> MapSet.member?(keys, k) end) |> Map.new()
      {pruned, keys}
    end)
  end

  # The in-flight batch for each running experiment is the most recent
  # not-yet-completed `score_bit` (or `collect_states`) row pointing
  # at that experiment. If there isn't one, the experiment doesn't
  # appear in the snapshot — the dashboard already shows experiment
  # status via the existing `/api/public-status/experiments` route.
  defp fetch_active_rows do
    sub =
      from(b in Batch,
        where: not is_nil(b.experiment_id) and b.status in ["pending", "running"],
        select: %{
          experiment_id: b.experiment_id,
          batch_id: b.id,
          inserted_at: b.inserted_at
        }
      )

    latest_per_exp =
      from(s in subquery(sub),
        group_by: s.experiment_id,
        select: %{experiment_id: s.experiment_id, latest_inserted_at: max(s.inserted_at)}
      )

    from(b in Batch,
      join: l in subquery(latest_per_exp),
      on:
        l.experiment_id == b.experiment_id and
          l.latest_inserted_at == b.inserted_at,
      join: e in Experiment,
      on: e.id == b.experiment_id,
      where: b.status in ["pending", "running"] and e.status == "running",
      select: %{
        experiment_id: b.experiment_id,
        env_name: e.env_name,
        # Carry the experiment record through so `render_row/3` can
        # ask `Experiments.n_episodes_for/1` for this experiment's
        # rollout count and normalize summed rewards into per-episode
        # means for the dashboard / SSE consumers.
        experiment: e,
        batch_id: b.id,
        cmd: b.cmd,
        payload: b.payload,
        n_results: b.n_results,
        sum_reward: b.sum_reward,
        sum_sq_reward: b.sum_sq_reward,
        best_reward: b.best_reward,
        baseline_reward: b.baseline_reward,
        completed_chunks: b.completed_chunks,
        total_chunks: b.total_chunks,
        policy_version: e.policy_version
      }
    )
    |> Repo.all()
  end

  defp render_row(row, rings, now_ms) do
    n = row.n_results || 0
    sum = row.sum_reward || 0.0
    sum_sq = row.sum_sq_reward || 0.0

    # Per-candidate stats are computed in the *summed* domain (each
    # candidate's reward is itself Σ over n_episodes rollouts in the
    # oracle). The dashboard speaks in per-episode means, so divide
    # by n_episodes after computing variance to get honest μ and σ
    # that match the per-episode best/baseline shown above. Stddev
    # scales linearly with the divisor so this is just `σ / k`.
    n_episodes = Experiments.n_episodes_for(row.experiment)

    sum_mean = if n > 0, do: sum / n, else: nil

    sum_stddev =
      if n > 1 and not is_nil(sum_mean) do
        var = sum_sq / n - sum_mean * sum_mean
        if var > 0, do: :math.sqrt(var), else: 0.0
      end

    mean = Experiments.normalize_reward(sum_mean, n_episodes)
    stddev = Experiments.normalize_reward(sum_stddev, n_episodes)

    progress =
      cond do
        is_integer(row.total_chunks) and row.total_chunks > 0 ->
          (row.completed_chunks || 0) / row.total_chunks

        true ->
          0.0
      end

    {rate, new_rings} = update_ring_and_rate(rings, row.batch_id, n, now_ms)

    target_bit =
      case row.payload do
        %{"target_bit" => b} when is_integer(b) -> b
        _ -> nil
      end

    # Streaming-CEGAR §Layer 3 surface: piggyback the most recent
    # commits on each active-experiment frame so the dashboard
    # can light up a "v=N — bit 3 +2.1" strip without opening a
    # second SSE stream. Two reads per refresh per active row;
    # `latest_commits/2` is indexed on (experiment_id, version).
    commits = Experiments.latest_commits(row.experiment_id, 5)

    payload = %{
      experiment_id: row.experiment_id,
      env_name: row.env_name,
      n_episodes: n_episodes,
      active_bit: %{
        batch_id: row.batch_id,
        target_bit: target_bit,
        cmd: row.cmd,
        n_results: n,
        mean: mean,
        stddev: stddev,
        best_reward: Experiments.normalize_reward(row.best_reward, n_episodes),
        baseline_reward: Experiments.normalize_reward(row.baseline_reward, n_episodes),
        # Raw summed forms alongside, for API consumers who want the
        # exact value as stored.
        best_reward_sum: row.best_reward,
        baseline_reward_sum: row.baseline_reward,
        mean_sum: sum_mean,
        stddev_sum: sum_stddev,
        completed_chunks: row.completed_chunks || 0,
        total_chunks: row.total_chunks || 0,
        progress: progress,
        results_per_min: rate
      },
      policy_version: row.policy_version,
      latest_commits:
        Enum.map(commits, fn c ->
          prev_mean = Experiments.normalize_reward(c.prev_reward, n_episodes)
          new_mean = Experiments.normalize_reward(c.new_reward, n_episodes)

          %{
            version: c.version,
            bit_idx: c.bit_idx,
            prev_reward: prev_mean,
            new_reward: new_mean,
            prev_reward_sum: c.prev_reward,
            new_reward_sum: c.new_reward,
            delta:
              if(is_number(new_mean) and is_number(prev_mean),
                do: new_mean - prev_mean,
                else: nil
              ),
            committed_at: c.committed_at
          }
        end)
    }

    {payload, new_rings}
  end

  # Rolling 60-s window: push the current `n_results` sample, trim
  # anything older than the window, derive the rate from (newest -
  # oldest) over elapsed time. Same shape as `MetricsBroker`'s
  # rate-per-minute, just keyed by batch_id.
  defp update_ring_and_rate(rings, batch_id, n_results, now_ms) do
    ring =
      rings
      |> Map.get(batch_id, :queue.new())
      |> push_ring(now_ms, n_results)
      |> trim_ring(now_ms)

    rate = rate_per_minute(ring, now_ms, n_results)

    {rate, Map.put(rings, batch_id, ring)}
  end

  defp push_ring(ring, ts, n), do: :queue.in({ts, n}, ring)

  defp trim_ring(ring, now_ms) do
    cutoff = now_ms - @window_secs * 1000

    case :queue.peek(ring) do
      {:value, {ts, _}} when ts < cutoff ->
        {{:value, _}, rest} = :queue.out(ring)
        trim_ring(rest, now_ms)

      _ ->
        ring
    end
  end

  defp rate_per_minute(ring, now_ms, latest_total) do
    case :queue.peek(ring) do
      {:value, {oldest_ts, oldest_total}} when oldest_ts < now_ms ->
        elapsed_s = (now_ms - oldest_ts) / 1000

        if elapsed_s > 0 do
          delta = latest_total - oldest_total
          round(delta * 60 / elapsed_s)
        else
          0
        end

      _ ->
        0
    end
  end
end
