defmodule Server.Application do
  use Application
  require Logger

  @impl true
  def start(_type, _args) do
    port = Application.get_env(:server, :port, 4000)

    children = [
      Server.Repo,
      {Oban, Application.fetch_env!(:server, Oban)},
      # MetricsBroker caches one Server.Metrics.snapshot/0 per
      # second for the SSE load stream. Must start AFTER the Repo
      # (it queries on init) and BEFORE Bandit (so the route is
      # answerable as soon as Bandit binds the port).
      Server.MetricsBroker,
      # AggregateBroker mirrors MetricsBroker but caches per-active-
      # batch streaming aggregates (n, mean, stddev, rate) for the
      # /api/public-status/stream/aggregates SSE feed. Same start-order
      # constraints as MetricsBroker: after Repo, before Bandit.
      Server.AggregateBroker,
      {Bandit, plug: Server.Router, port: port}
    ]

    opts = [strategy: :one_for_one, name: Server.Supervisor]

    case Supervisor.start_link(children, opts) do
      {:ok, pid} ->
        Logger.info("Synthex Hub server listening on :#{port}")
        log_auth_status()

        # Telemetry-based incident recording. Attached AFTER the
        # supervisor is up so the Repo is available when failed
        # jobs try to write a system_event. Idempotent across
        # restarts because :telemetry.attach/4 returns
        # `{:error, :already_exists}` on duplicates, which we
        # gracefully ignore in the handler module.
        case Server.ObanFailureHandler.attach() do
          :ok -> :ok
          {:error, :already_exists} -> :ok
          other -> Logger.warning("ObanFailureHandler attach: #{inspect(other)}")
        end

        rescue_orphan_executing_jobs()

        {:ok, pid}

      err ->
        err
    end
  end

  # Self-heal: if the previous node died mid-step (deploy, crash, OOM),
  # the controller row sits in `:executing` until Lifeline notices, which
  # can be up to `rescue_after` (currently 5 min). The streaming
  # controller's heartbeat ticks every 60 s, so any executing master
  # job whose `attempted_at` is older than a couple of beats can't
  # possibly have a live owner — reset it to `:available` so Oban
  # picks it up immediately. The `unique` constraint on
  # ExperimentController still protects against double-running.
  defp rescue_orphan_executing_jobs do
    import Ecto.Query

    cutoff = DateTime.add(DateTime.utc_now(), -180, :second)

    {n, _} =
      from(j in Oban.Job,
        where: j.queue == "master" and j.state == "executing" and j.attempted_at < ^cutoff
      )
      |> Server.Repo.update_all(set: [state: "available"])

    if n > 0 do
      Logger.warning(
        "Rescued #{n} orphan executing master job(s) on boot (attempted_at older than 180s)."
      )
    end

    :ok
  rescue
    error ->
      Logger.warning("Orphan rescue failed: #{inspect(error)}")
      :ok
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
