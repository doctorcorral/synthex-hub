defmodule Worker.PythonPort do
  @moduledoc """
  One persistent Python interpreter behind an Erlang Port.

  Crash & wedge safety:
    - On port `:exit_status` we reply `{:error, :port_crashed}` to every
      pending caller and stop. The supervisor restarts us → fresh Python.
    - On Python exception, the oracle script writes a JSON `{"error": ...}`
      response so the call returns immediately rather than timing out.
    - Each in-flight job arms a **watchdog timer**. If the oracle does not
      answer within `:job_timeout_ms`, we treat the Python process as
      wedged (stuck in a compute/GPU-bound launch that ignores stdin),
      `SIGKILL` its OS pid, reply `{:error, :job_timeout}` to the caller,
      and stop so the supervisor spawns a fresh interpreter. This is what
      makes a slow/hung chunk self-heal in seconds instead of needing a
      human to restart the container.

  Why kill the OS pid and not just `Port.close/1`: closing the port only
  shuts the pipe's stdin. A Python process busy inside a long numpy loop
  or a CUDA launch is not reading stdin, so it keeps running (and keeps
  the GPU/CPU pinned) after the port closes. Only an explicit signal
  reclaims it.
  """
  use GenServer
  require Logger

  # Fallback when :job_timeout_ms is unset. Generous enough never to fire
  # on a legitimate rollout, tight enough to reclaim a true wedge quickly.
  @default_job_timeout_ms 5 * 60 * 1000

  def start_link(opts) do
    name = Keyword.fetch!(opts, :name)
    GenServer.start_link(__MODULE__, opts, name: name)
  end

  @doc """
  Synchronously score a payload (must include a `cmd` field).

  The caller's `GenServer.call` deadline is derived from the job timeout
  plus a grace margin, so the in-process watchdog (which kills + recycles
  the port) is always the authority that fires first. A caller therefore
  gets a clean `{:error, :job_timeout}` rather than an exit.
  """
  def score(name, payload, _timeout \\ nil) do
    call_timeout = job_timeout_ms() + 30_000

    try do
      GenServer.call(name, {:score, payload}, call_timeout)
    catch
      :exit, reason -> {:error, {:port_unavailable, reason}}
    end
  end

  @impl true
  def init(opts) do
    Process.flag(:trap_exit, true)
    state = %{port: nil, os_pid: nil, pending: %{}, buffer: "", opts: opts}
    {:ok, open_port(state)}
  end

  defp open_port(state) do
    python = Application.get_env(:worker, :python_executable, "python3")
    script = Application.get_env(:worker, :oracle_script)
    exe = System.find_executable(python) || raise "python executable not found: #{python}"

    Logger.info("[#{label(state)}] starting #{python} #{script}")

    port =
      Port.open({:spawn_executable, exe}, [
        :binary,
        :stream,
        :use_stdio,
        :exit_status,
        args: ["-u", script]
      ])

    os_pid =
      case Port.info(port, :os_pid) do
        {:os_pid, pid} -> pid
        _ -> nil
      end

    %{state | port: port, os_pid: os_pid, buffer: "", pending: %{}}
  end

  @impl true
  def handle_call({:score, payload}, from, state) do
    id = System.unique_integer([:positive])
    payload = Map.put(payload, "job_id", id)

    case Jason.encode(payload) do
      {:ok, json} ->
        Port.command(state.port, json <> "\n")
        timer = Process.send_after(self(), {:watchdog, id}, job_timeout_ms())
        {:noreply, %{state | pending: Map.put(state.pending, id, {from, timer})}}

      {:error, err} ->
        {:reply, {:error, {:encode_failed, err}}, state}
    end
  end

  @impl true
  def handle_info({port, {:data, data}}, %{port: port} = state) do
    new_buffer = state.buffer <> data
    {lines, remaining} = split_lines(new_buffer)

    pending =
      Enum.reduce(lines, state.pending, fn line, acc ->
        deliver_line(line, acc)
      end)

    {:noreply, %{state | pending: pending, buffer: remaining}}
  end

  def handle_info({:watchdog, id}, state) do
    case Map.pop(state.pending, id) do
      {nil, _} ->
        # Already answered; stale timer.
        {:noreply, state}

      {{from, _timer}, remaining} ->
        Logger.error(
          "[#{label(state)}] job #{id} exceeded #{job_timeout_ms()}ms — " <>
            "SIGKILL python (os_pid=#{inspect(state.os_pid)}) and recycling port"
        )

        kill_os_process(state.os_pid)
        GenServer.reply(from, {:error, :job_timeout})

        # Fail any other in-flight callers too — the interpreter is gone.
        Enum.each(remaining, fn {_id, {other_from, t}} ->
          cancel_timer(t)
          GenServer.reply(other_from, {:error, :job_timeout})
        end)

        {:stop, {:shutdown, :job_timeout}, %{state | pending: %{}}}
    end
  end

  def handle_info({port, {:exit_status, status}}, %{port: port} = state) do
    Logger.warning("[#{label(state)}] python port exited (status=#{status}); restarting subtree")

    Enum.each(state.pending, fn {_id, {from, t}} ->
      cancel_timer(t)
      GenServer.reply(from, {:error, {:port_crashed, status}})
    end)

    {:stop, {:shutdown, {:port_crashed, status}}, %{state | pending: %{}}}
  end

  def handle_info(msg, state) do
    Logger.debug("[#{label(state)}] unexpected message: #{inspect(msg)}")
    {:noreply, state}
  end

  @impl true
  def terminate(_reason, state) do
    # Make sure we never leak a compute-bound Python child on shutdown.
    kill_os_process(Map.get(state, :os_pid))

    case Map.get(state, :port) do
      port when is_port(port) ->
        try do
          Port.close(port)
        rescue
          _ -> :ok
        end

      _ ->
        :ok
    end

    :ok
  end

  # ── Internals ───────────────────────────────────────────────

  defp deliver_line(line, pending) do
    case Jason.decode(line) do
      {:ok, %{"job_id" => id} = response} ->
        case Map.pop(pending, id) do
          {nil, acc} ->
            Logger.warning("orphan response for job_id=#{id}")
            acc

          {{from, timer}, acc} ->
            cancel_timer(timer)

            reply =
              cond do
                Map.has_key?(response, "error") -> {:error, response["error"]}
                true -> {:ok, response["results"] || []}
              end

            GenServer.reply(from, reply)
            acc
        end

      {:ok, _other} ->
        Logger.warning("python emitted JSON without job_id: #{line}")
        pending

      {:error, _} ->
        Logger.warning("python emitted non-JSON line: #{line}")
        pending
    end
  end

  defp kill_os_process(nil), do: :ok

  defp kill_os_process(os_pid) when is_integer(os_pid) do
    case System.find_executable("kill") do
      nil -> :ok
      kill -> System.cmd(kill, ["-9", Integer.to_string(os_pid)], stderr_to_stdout: true)
    end

    :ok
  rescue
    _ -> :ok
  end

  defp cancel_timer(nil), do: :ok
  defp cancel_timer(ref), do: Process.cancel_timer(ref)

  defp job_timeout_ms,
    do: Application.get_env(:worker, :job_timeout_ms, @default_job_timeout_ms)

  defp split_lines(buffer), do: split_lines(buffer, [])

  defp split_lines(buffer, acc) do
    case :binary.split(buffer, "\n") do
      [line, rest] -> split_lines(rest, [line | acc])
      [rest] -> {Enum.reverse(acc), rest}
    end
  end

  defp label(state) do
    case Keyword.get(state.opts, :name, __MODULE__) do
      {:via, _, {_, key}} -> "PythonPort:#{inspect(key)}"
      atom when is_atom(atom) -> "PythonPort:#{inspect(atom)}"
      other -> "PythonPort:#{inspect(other)}"
    end
  end
end
