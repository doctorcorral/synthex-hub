defmodule Server.SSEStream do
  @moduledoc """
  Minimal Server-Sent Events handler for the public load stream.

  Wire format per the SSE spec (`text/event-stream`): each message
  is `"data: <json>\\n\\n"`. Browsers' native `EventSource` handles
  reconnection automatically on connection drop, so we deliberately
  cap each open stream at `@max_duration_ms` and exit cleanly — this
  keeps proxy keepalive budgets predictable and prevents long-lived
  Bandit processes from accumulating during weekly traffic spikes.

  Notes:

    * `Process.sleep/1` blocks ONE Bandit process per connection; in
      BEAM that's a few KB each, so hundreds of concurrent watchers
      cost ~MBs at most.
    * We push from `Server.MetricsBroker.latest/0` — never the DB —
      so per-client cost is constant regardless of viewer count.
    * `x-accel-buffering: no` is a Nginx hint; Fly's proxy already
      streams chunked responses, but the header is harmless
      elsewhere and helps if anyone reverse-proxies the site.
  """

  import Plug.Conn

  @event_interval_ms 1_000
  @max_duration_ms 5 * 60 * 1_000

  def serve(conn) do
    conn =
      conn
      |> put_resp_header("content-type", "text/event-stream")
      |> put_resp_header("cache-control", "no-cache, no-transform")
      |> put_resp_header("connection", "keep-alive")
      |> put_resp_header("x-accel-buffering", "no")
      |> put_resp_header("access-control-allow-origin", "*")
      |> send_chunked(200)

    # Send one frame immediately so the client gets data on connect
    # without waiting a full tick.
    started = System.monotonic_time(:millisecond)
    stream_loop(conn, started)
  end

  defp stream_loop(conn, started_at) do
    payload = encode_event(Server.MetricsBroker.latest())

    case chunk(conn, payload) do
      {:ok, conn} ->
        if System.monotonic_time(:millisecond) - started_at > @max_duration_ms do
          conn
        else
          Process.sleep(@event_interval_ms)
          stream_loop(conn, started_at)
        end

      {:error, _reason} ->
        conn
    end
  end

  defp encode_event(nil), do: "data: {\"loading\":true}\n\n"

  defp encode_event(snap) when is_map(snap) do
    case Jason.encode(snap) do
      {:ok, json} -> "data: " <> json <> "\n\n"
      {:error, _} -> "data: {\"error\":\"encode\"}\n\n"
    end
  end
end
