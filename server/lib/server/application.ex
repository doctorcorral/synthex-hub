defmodule Server.Application do
  use Application
  require Logger

  @impl true
  def start(_type, _args) do
    port = Application.get_env(:server, :port, 4000)

    children = [
      Server.Repo,
      {Oban, Application.fetch_env!(:server, Oban)},
      {Bandit, plug: Server.Router, port: port}
    ]

    opts = [strategy: :one_for_one, name: Server.Supervisor]

    case Supervisor.start_link(children, opts) do
      {:ok, pid} ->
        Logger.info("Synthex Hub server listening on :#{port}")
        log_auth_status()
        {:ok, pid}

      err ->
        err
    end
  end

  defp log_auth_status do
    case Application.get_env(:server, :api_token) do
      token when is_binary(token) and byte_size(token) > 0 ->
        Logger.info(
          "API auth: master-only (Bearer token, #{byte_size(token)} bytes). " <>
            "/api/worker/* and /api/public-status are open; /api/master/* and /api/status require the token."
        )

      _ ->
        Logger.warning(
          "API auth: fully open (no API_TOKEN set). Anyone can submit batches. " <>
            "Set fly secrets set API_TOKEN=… for production."
        )
    end
  end
end
