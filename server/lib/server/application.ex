defmodule Server.Application do
  use Application
  require Logger

  @impl true
  def start(_type, _args) do
    port = Application.get_env(:server, :port, 4000)

    children = [
      Server.Repo,
      {Oban, Application.fetch_env!(:server, Oban)},
      {Bandit,
       plug: Server.Router,
       port: port,
       # Workers submit large `collect_states` payloads (multi-megabyte
       # trajectories). Bandit's default request_line / header limits
       # are fine; we explicitly bump body-side timeouts here so a slow
       # upload across the public internet doesn't time out at the
       # acceptor before Plug.Parsers (in Server.Router) can read it.
       http_options: [
         max_request_line_length: 16_384,
         max_header_length: 16_384,
         max_requests: 1000
       ],
       thousand_island_options: [
         read_timeout: 60_000
       ]}
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
