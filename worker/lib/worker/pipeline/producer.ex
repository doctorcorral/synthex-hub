defmodule Worker.Pipeline.Producer do
  @moduledoc """
  Custom GenStage producer for the Broadway worker pipeline.

  Pulls chunks from the hub via HTTP and emits one `Broadway.Message`
  per *candidate*. Backpressure is honoured: we only pull a new chunk
  when downstream demand exceeds our buffer, and we only have one
  HTTP poll in flight at a time.

  On graceful shutdown (`prepare_for_draining/1`), we stop polling so
  Broadway can drain the in-flight messages without claiming new work.

  Notes on chunk lifecycle:

    * Before emitting messages for a chunk, we call
      `Worker.ChunkAggregator.register_chunk/3`. The aggregator owns
      the per-chunk %{total, results} state and does the final HTTP
      submission once all candidates report back.
    * If the producer crashes mid-chunk, partially-claimed chunks are
      rescued by Oban Lifeline on the server side after the lease
      expires. No client-side recovery needed.
  """
  use GenStage
  require Logger

  @behaviour Broadway.Producer

  @ack_ref __MODULE__

  defmodule State do
    @moduledoc false
    defstruct buffer: [],
              demand: 0,
              poll_in_flight?: false,
              poll_ref: nil,
              draining?: false
  end

  # ── Broadway.Producer callbacks ─────────────────────────────

  @impl Broadway.Producer
  def prepare_for_start(_module, broadway_opts) do
    {[], broadway_opts}
  end

  @impl Broadway.Producer
  def prepare_for_draining(state) do
    Logger.info("[Producer] draining; will stop polling for new chunks")
    {:noreply, [], %{state | draining?: true}}
  end

  # ── GenStage callbacks ──────────────────────────────────────

  @impl GenStage
  def init(_opts) do
    schedule_poll(0)
    {:producer, %State{poll_in_flight?: true}}
  end

  @impl GenStage
  def handle_demand(incoming, state) when incoming > 0 do
    state = %{state | demand: state.demand + incoming}
    deliver(state)
  end

  @impl GenStage
  def handle_info(:poll, %State{draining?: true} = state) do
    {:noreply, [], %{state | poll_in_flight?: false}}
  end

  def handle_info(:poll, state) do
    state = %{state | poll_in_flight?: false, poll_ref: nil}

    case poll_chunk() do
      {:ok, chunk} ->
        :telemetry.execute(
          [:synthex_hub, :worker, :chunk, :claimed],
          %{candidates: chunk |> Map.get("candidates", []) |> length()},
          %{chunk_id: chunk["chunk_id"], env_name: chunk["env_name"]}
        )

        new_messages = chunk_to_messages(chunk)
        state = %{state | buffer: state.buffer ++ new_messages}
        deliver(state)

      :empty ->
        :telemetry.execute([:synthex_hub, :worker, :poll, :empty], %{count: 1}, %{})
        schedule_poll_into(state, Application.get_env(:worker, :poll_interval_ms, 2_000))

      {:error, reason} ->
        Logger.warning("[Producer] poll failed: #{inspect(reason)}; backing off")
        :telemetry.execute([:synthex_hub, :worker, :poll, :error], %{count: 1}, %{})
        schedule_poll_into(state, 5_000)
    end
  end

  def handle_info(_other, state), do: {:noreply, [], state}

  # ── Internals ───────────────────────────────────────────────

  defp deliver(%State{demand: 0} = state) do
    {:noreply, [], state}
  end

  defp deliver(%State{buffer: [], draining?: true} = state) do
    {:noreply, [], state}
  end

  defp deliver(%State{buffer: []} = state) do
    {:noreply, [], maybe_schedule_poll(state, 0)}
  end

  defp deliver(%State{buffer: buffer, demand: demand} = state) do
    {to_send, rest} = Enum.split(buffer, demand)
    new_state = %{state | buffer: rest, demand: demand - length(to_send)}

    new_state =
      if rest == [] and new_state.demand > 0 do
        maybe_schedule_poll(new_state, 0)
      else
        new_state
      end

    {:noreply, to_send, new_state}
  end

  defp maybe_schedule_poll(%State{poll_in_flight?: true} = state, _delay), do: state

  defp maybe_schedule_poll(state, delay) do
    schedule_poll(delay)
    %{state | poll_in_flight?: true}
  end

  defp schedule_poll_into(state, delay) do
    {:noreply, [], maybe_schedule_poll(state, delay)}
  end

  defp schedule_poll(delay), do: Process.send_after(self(), :poll, delay)

  defp poll_chunk do
    worker_id = Application.get_env(:worker, :worker_id)

    case Worker.HttpClient.get("/worker/jobs/request?worker_id=#{worker_id}") do
      {:ok, %{status: 204}} ->
        :empty

      {:ok, %{status: 200, body: body}} ->
        {:ok, decode_body(body)}

      {:ok, %{status: 401}} ->
        {:error, :unauthorized}

      {:ok, %{status: status}} ->
        {:error, {:http, status}}

      {:error, reason} ->
        {:error, reason}
    end
  rescue
    e -> {:error, {:exception, Exception.message(e)}}
  end

  defp chunk_to_messages(chunk) do
    chunk_id = chunk["chunk_id"]
    candidates = chunk["candidates"] || []
    total = length(candidates)

    Worker.ChunkAggregator.register_chunk(chunk_id, total, chunk)

    base_params =
      chunk
      |> Map.drop(["candidates", "oban_job_id", "attempt", "max_attempts", "chunk_id"])
      |> Map.merge(chunk["params"] || %{})

    if batched_adapter?(chunk["adapter"]) and total > 0 do
      # Batched adapters (e.g. mujoco_warp) score the WHOLE chunk in a
      # single oracle call — one big GPU launch of chunk_size×episodes
      # worlds — instead of one ~episodes-world launch per candidate.
      # We emit a single message carrying every candidate; the processor
      # fans the N results back into the aggregator (registered total=N).
      [
        %Broadway.Message{
          data: %{batched: true, candidates: candidates},
          metadata: %{chunk_id: chunk_id, total: total, base_params: base_params},
          acknowledger: {__MODULE__, @ack_ref, %{chunk_id: chunk_id, idx: :batch}}
        }
      ]
    else
      # CPU swarm: one message per candidate so the whole port pool
      # scores candidates in parallel across cores.
      candidates
      |> Enum.with_index()
      |> Enum.map(fn {candidate, idx} ->
        %Broadway.Message{
          data: %{candidate: candidate, idx: idx},
          metadata: %{
            chunk_id: chunk_id,
            total: total,
            base_params: base_params
          },
          acknowledger: {__MODULE__, @ack_ref, %{chunk_id: chunk_id, idx: idx}}
        }
      end)
    end
  end

  # Adapters whose oracle evaluates a chunk's candidates as one batched
  # call. Configurable so a new GPU adapter can opt in without code
  # changes; defaults to mujoco_warp.
  defp batched_adapter?(adapter) when is_binary(adapter) do
    adapter in Application.get_env(:worker, :batched_adapters, ["mujoco_warp"])
  end

  defp batched_adapter?(_), do: false

  defp decode_body(body) when is_map(body), do: body
  defp decode_body(body) when is_binary(body), do: Jason.decode!(body)

  # ── Acknowledger ────────────────────────────────────────────

  @behaviour Broadway.Acknowledger

  @impl Broadway.Acknowledger
  def ack(@ack_ref, _successful, []), do: :ok

  def ack(@ack_ref, _successful, failed) do
    # Failed messages already had their failure recorded in
    # ChunkAggregator (via handle_failed/2 in the pipeline).
    # Nothing more to do; Oban Lifeline rescues abandoned chunks.
    Enum.each(failed, fn msg ->
      Logger.debug("[Producer.ack] failed: #{inspect(msg.metadata)}")
    end)

    :ok
  end
end
