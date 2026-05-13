defmodule Server.MetricsBroker do
  @moduledoc """
  Refreshes a single `Server.Metrics.snapshot/0` once per second and
  caches it in a public ETS table for SSE consumers.

  Two reasons not to let each SSE client query the DB directly:

    1. Postgres egress — every SSE client polling once/sec would
       multiply DB round-trips by the number of viewers. One
       broker per node keeps it O(1).
    2. Rolling rate — `evals_per_minute` needs a window of past
       samples; that state lives here, not in clients.

  Layout:

    * State: a `:queue` of `{ts_ms, evals_total}` samples, trimmed
      to the last `@window_secs` seconds on every tick.
    * ETS: `Server.MetricsBroker.Cache` (public, read-concurrent),
      one row `{:snapshot, map}` updated every tick.

  Read with `Server.MetricsBroker.latest/0` — returns `nil` until
  the first refresh lands (~100 ms after node boot).
  """

  use GenServer
  require Logger

  @table :server_metrics_cache
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

  # ── GenServer ───────────────────────────────────────────────────

  @impl true
  def init(_) do
    :ets.new(@table, [:named_table, :public, read_concurrency: true])
    Process.send_after(self(), :refresh, 100)
    {:ok, %{ring: :queue.new()}}
  end

  @impl true
  def handle_info(:refresh, state) do
    new_state =
      try do
        do_refresh(state)
      rescue
        err ->
          Logger.warning("[MetricsBroker] refresh failed: #{inspect(err)}")
          state
      end

    Process.send_after(self(), :refresh, @refresh_ms)
    {:noreply, new_state}
  end

  # ── internals ───────────────────────────────────────────────────

  defp do_refresh(state) do
    raw = Server.Metrics.snapshot()
    now_ms = System.system_time(:millisecond)

    ring =
      state.ring
      |> :queue.in({now_ms, raw.evals_total})
      |> trim_ring(now_ms)

    rate_per_min = rate_per_minute_from(ring, now_ms, raw.evals_total)

    snapshot =
      raw
      |> Map.put(:evals_per_minute, rate_per_min)
      |> Map.put(:ts, DateTime.utc_now() |> DateTime.to_iso8601())

    :ets.insert(@table, {:snapshot, snapshot})
    %{state | ring: ring}
  end

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

  defp rate_per_minute_from(ring, now_ms, latest_total) do
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
