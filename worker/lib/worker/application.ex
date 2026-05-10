defmodule Worker.Application do
  use Application
  require Logger

  @impl true
  def start(_type, _args) do
    log_startup()
    Worker.Telemetry.attach()

    children = [
      Worker.PortSupervisor,
      Worker.PortPool,
      {Task.Supervisor, name: Worker.SubmitTaskSupervisor},
      Worker.ChunkAggregator,
      Worker.Pipeline,
      Worker.Registrar
    ]

    # rest_for_one: if PortPool dies, Pipeline must restart too
    # because its handle_message callbacks depend on it. Same for
    # PortSupervisor → everything below.
    opts = [strategy: :rest_for_one, name: Worker.Supervisor]
    Supervisor.start_link(children, opts)
  end

  defp log_startup do
    cfg = Application.get_all_env(:worker)
    Logger.info("Synthex Hub worker starting:")
    Logger.info("  worker_id    = #{cfg[:worker_id]}  (stable internal UUID)")
    Logger.info("  display_name = #{cfg[:display_name]}  (shown on leaderboard)")
    Logger.info("  hostname     = #{cfg[:hostname]}")
    Logger.info("  server_url   = #{cfg[:server_url]}")
    Logger.info("  pool_size    = #{cfg[:pool_size]}")

    case cfg[:api_token] do
      t when is_binary(t) and byte_size(t) > 0 ->
        Logger.info("  api_token    = configured (#{byte_size(t)} bytes)")

      _ ->
        Logger.info("  api_token    = anonymous (no Authorization header sent)")
    end

    if cfg[:display_name] in [nil, "", "anonymous"] do
      Logger.info(
        "  Tip: set WORKER_NAME=<your-handle> to appear on the public leaderboard at /api/public-status/leaderboard"
      )
    end
  end
end
