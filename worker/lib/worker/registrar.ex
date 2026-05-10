defmodule Worker.Registrar do
  @moduledoc """
  Long-running task that registers this worker with the hub on startup
  and pings `/api/worker/heartbeat` on a fixed interval. The hub uses
  heartbeats to compute "active workers" and to drive Lifeline-based
  rescue when a worker disappears.
  """
  use GenServer
  require Logger

  def start_link(_opts), do: GenServer.start_link(__MODULE__, [], name: __MODULE__)

  @impl true
  def init(_) do
    Process.send_after(self(), :register, 0)
    {:ok, %{registered: false}}
  end

  @impl true
  def handle_info(:register, state) do
    case do_register() do
      :ok ->
        interval = Application.get_env(:worker, :heartbeat_interval_ms, 30_000)
        Process.send_after(self(), :heartbeat, interval)
        {:noreply, %{state | registered: true}}

      :retry ->
        Process.send_after(self(), :register, 5_000)
        {:noreply, state}
    end
  end

  def handle_info(:heartbeat, state) do
    interval = Application.get_env(:worker, :heartbeat_interval_ms, 30_000)

    case Worker.HttpClient.post("/worker/heartbeat", %{worker_id: worker_id()}) do
      {:ok, %{status: 200}} ->
        :ok

      {:ok, %{status: 404}} ->
        Logger.warning("[Registrar] hub forgot us; re-registering")
        send(self(), :register)

      other ->
        Logger.debug("[Registrar] heartbeat: #{inspect(other)}")
    end

    Process.send_after(self(), :heartbeat, interval)
    {:noreply, state}
  end

  defp do_register do
    payload = %{
      id: worker_id(),
      name: display_name(),
      hostname: Application.get_env(:worker, :hostname),
      pool_size: Application.get_env(:worker, :pool_size, 1),
      version: Application.spec(:worker, :vsn) |> to_string(),
      metadata: %{
        elixir: System.version(),
        otp: System.otp_release()
      }
    }

    case Worker.HttpClient.post("/worker/register", payload) do
      {:ok, %{status: status}} when status in [200, 201] ->
        Logger.info(
          "[Registrar] registered worker_id=#{worker_id()} display_name=#{display_name()}"
        )

        :ok

      other ->
        Logger.warning("[Registrar] register failed: #{inspect(other)}; retrying")
        :retry
    end
  end

  defp worker_id, do: Application.get_env(:worker, :worker_id)
  defp display_name, do: Application.get_env(:worker, :display_name, "anonymous")
end
