import Config

# WORKER_ID is the stable internal identifier — a UUID set by
# scripts/entrypoint.sh on first start and persisted to a Docker
# volume so it survives restarts. We fall back to a synthesized id
# only when running outside Docker (mix run locally).
worker_internal_id =
  System.get_env("WORKER_ID") ||
    "w-#{:inet.gethostname() |> elem(1)}-#{:erlang.system_time(:second)}"

# WORKER_NAME is the human handle shown on the public leaderboard.
# All workers whose name is "anonymous" aggregate into a single row
# server-side, so opting out of recognition is just leaving this
# empty (the default).
display_name =
  case System.get_env("WORKER_NAME") do
    nil -> "anonymous"
    "" -> "anonymous"
    name -> name
  end

pool_size =
  case System.get_env("POOL_SIZE") do
    nil -> System.schedulers_online()
    val -> max(1, String.to_integer(val))
  end

oracle_script =
  System.get_env("ORACLE_SCRIPT") ||
    Path.expand("../environments/gymnasium/oracle_port.py", Path.dirname(__ENV__.file))

config :worker,
  server_url: System.get_env("SERVER_URL", "http://localhost:4000/api"),
  api_token: System.get_env("API_TOKEN"),
  worker_id: worker_internal_id,
  display_name: display_name,
  hostname: System.get_env("HOSTNAME", to_string(elem(:inet.gethostname(), 1))),
  pool_size: pool_size,
  python_executable: System.get_env("PYTHON", "python3"),
  oracle_script: oracle_script,
  poll_interval_ms: String.to_integer(System.get_env("POLL_INTERVAL_MS", "2000")),
  heartbeat_interval_ms: String.to_integer(System.get_env("HEARTBEAT_INTERVAL_MS", "30000")),
  request_timeout_ms: String.to_integer(System.get_env("REQUEST_TIMEOUT_MS", "30000"))
