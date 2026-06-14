defmodule Worker.Pipeline do
  @moduledoc """
  Broadway pipeline that turns the worker into a streaming
  candidate-evaluator.

  Topology:

      Worker.Pipeline.Producer  (one GenStage producer)
        │  emits one Broadway.Message per candidate, tagged with
        │  chunk_id, total, and the chunk-level params
        │
        ▼
      processors :default  (concurrency = pool_size, max_demand = 1)
        │  each call:
        │    1. Worker.PortPool.with_port/2  → checkout a Python port
        │    2. Worker.PythonPort.score/3    → score one candidate
        │    3. Worker.ChunkAggregator.add_result/2 with the result
        │  port is auto-returned (with_port wraps in try/after)
        │
        ▼
      (no batchers)
        Submission to the hub is performed by ChunkAggregator once
        all of a chunk's candidates have been scored. This avoids
        Broadway's fixed batch_size/batch_timeout, which doesn't fit
        variable-sized chunks.

  Why `max_demand: 1`: each candidate is heavy (a full Gymnasium
  rollout), so we want demand-pull granularity at message level. With
  a higher max_demand a single processor would buffer multiple
  messages while holding a port, blocking the rest of the pool.
  """
  use Broadway
  require Logger

  def start_link(_opts) do
    pool_size = Application.get_env(:worker, :pool_size, 1)

    producer_opts =
      [module: {Worker.Pipeline.Producer, []}, concurrency: 1]
      |> maybe_put_rate_limiting()

    Broadway.start_link(__MODULE__,
      name: __MODULE__,
      producer: producer_opts,
      processors: [
        default: [concurrency: pool_size, max_demand: 1, min_demand: 0]
      ]
      # No batchers — see moduledoc.
    )
  end

  defp maybe_put_rate_limiting(opts) do
    case Application.get_env(:worker, :rate_limit) do
      %{allowed: a, interval: i} ->
        Keyword.put(opts, :rate_limiting, allowed_messages: a, interval: i)

      _ ->
        opts
    end
  end

  # ── Broadway callbacks ──────────────────────────────────────

  @impl Broadway
  def handle_message(_processor, %Broadway.Message{data: %{batched: true}} = message, _context) do
    handle_batched(message)
  end

  def handle_message(_processor, %Broadway.Message{} = message, _context) do
    %{candidate: candidate, idx: idx} = message.data

    chunk_id = message.metadata.chunk_id
    base_params = message.metadata.base_params
    timeout = score_timeout()

    started = System.monotonic_time()

    result =
      try do
        Worker.PortPool.with_port(timeout, fn port ->
          payload = build_payload(base_params, [candidate])

          case Worker.PythonPort.score(port, payload, timeout) do
            {:ok, [first | _]} ->
              first |> Map.put("idx", idx)

            {:ok, []} ->
              %{"idx" => idx, "error" => "empty result from oracle"}

            {:error, reason} ->
              %{"idx" => idx, "error" => inspect(reason)}
          end
        end)
      rescue
        e -> %{"idx" => idx, "error" => "#{inspect(e.__struct__)}: #{Exception.message(e)}"}
      catch
        :exit, reason -> %{"idx" => idx, "error" => "exit: #{inspect(reason)}"}
      end

    duration = System.monotonic_time() - started

    :telemetry.execute(
      [:synthex_hub, :worker, :candidate, :scored],
      %{duration: duration},
      %{chunk_id: chunk_id, idx: idx, success: not Map.has_key?(result, "error")}
    )

    Worker.ChunkAggregator.add_result(chunk_id, result)
    Broadway.Message.put_data(message, result)
  end

  # Score an entire chunk in one oracle call (batched adapters). The
  # oracle returns one result per candidate, in candidate order; we fan
  # exactly `total` results back into the aggregator (padding with errors
  # if the oracle returned fewer) so the chunk always completes.
  defp handle_batched(message) do
    candidates = message.data.candidates
    chunk_id = message.metadata.chunk_id
    total = message.metadata.total
    base_params = message.metadata.base_params
    timeout = score_timeout()

    started = System.monotonic_time()

    outcome =
      try do
        Worker.PortPool.with_port(timeout, fn port ->
          payload = build_payload(base_params, candidates)

          case Worker.PythonPort.score(port, payload, timeout) do
            {:ok, list} when is_list(list) -> {:ok, list}
            {:ok, other} -> {:error, "unexpected oracle shape: #{inspect(other)}"}
            {:error, reason} -> {:error, inspect(reason)}
          end
        end)
      rescue
        e -> {:error, "#{inspect(e.__struct__)}: #{Exception.message(e)}"}
      catch
        :exit, reason -> {:error, "exit: #{inspect(reason)}"}
      end

    duration = System.monotonic_time() - started

    results =
      case outcome do
        {:ok, list} ->
          for idx <- 0..(total - 1) do
            case Enum.at(list, idx) do
              %{} = r -> Map.put(r, "idx", idx)
              _ -> %{"idx" => idx, "error" => "missing result from oracle"}
            end
          end

        {:error, reason} ->
          for idx <- 0..(total - 1), do: %{"idx" => idx, "error" => reason}
      end

    Enum.each(results, fn r -> Worker.ChunkAggregator.add_result(chunk_id, r) end)

    :telemetry.execute(
      [:synthex_hub, :worker, :chunk, :batched_scored],
      %{duration: duration, candidates: total},
      %{chunk_id: chunk_id, success: match?({:ok, _}, outcome)}
    )

    Broadway.Message.put_data(message, %{"batched" => true, "n" => total})
  end

  @impl Broadway
  def handle_failed(messages, _context) do
    # Broadway only calls this when handle_message itself raises beyond
    # our rescue/catch. Guard the invariant: aggregator must still see
    # a result for every message of the chunk, otherwise the chunk
    # would never complete.
    Enum.each(messages, fn message ->
      chunk_id = message.metadata[:chunk_id]
      reason = inspect(message.status)

      cond do
        is_nil(chunk_id) ->
          :ok

        # A batched message owns its whole chunk: fan one error per
        # expected result so the chunk still reaches `total` and submits.
        match?(%{batched: true}, message.data) ->
          total = message.metadata[:total] || 0

          for idx <- 0..(total - 1)//1 do
            Worker.ChunkAggregator.add_result(chunk_id, %{
              "idx" => idx,
              "error" => "broadway_failed: #{reason}"
            })
          end

        true ->
          idx = message.data[:idx] || -1

          Worker.ChunkAggregator.add_result(chunk_id, %{
            "idx" => idx,
            "error" => "broadway_failed: #{reason}"
          })
      end
    end)

    messages
  end

  defp build_payload(base_params, candidates) when is_list(candidates) do
    base_params
    |> Map.put_new("cmd", "score_bit")
    |> Map.put("candidates", candidates)
  end

  # Borrowers must be willing to wait at least as long as a port may be
  # busy on a single job, plus the grace the port itself uses to kill +
  # recycle a wedged interpreter. Otherwise a checkout could give up
  # while the one port (pool_size=1, GPU) is mid-recycle.
  defp score_timeout do
    Application.get_env(:worker, :job_timeout_ms, 5 * 60 * 1000) + 60_000
  end
end
