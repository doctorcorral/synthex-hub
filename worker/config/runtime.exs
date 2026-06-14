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

# WORKER_CAPABILITIES is a comma-separated, preference-ordered list
# of physics adapters this worker can run, e.g. "mujoco_warp,mujoco"
# for a CUDA box that prefers Warp but will fall back to plain
# MuJoCo. Defaults to "mujoco" — the CPU swarm. The hub uses this
# to route chunks (hard filter on membership, soft sort on order),
# so a CPU worker must NOT advertise mujoco_warp (it can't run it),
# and the chosen ORACLE_SCRIPT must actually implement every adapter
# listed here.
capabilities =
  case System.get_env("WORKER_CAPABILITIES") do
    nil ->
      ["mujoco"]

    "" ->
      ["mujoco"]

    csv ->
      csv
      |> String.split(",", trim: true)
      |> Enum.map(&String.trim/1)
      |> Enum.reject(&(&1 == ""))
      |> case do
        [] -> ["mujoco"]
        list -> list
      end
  end

config :worker,
  capabilities: capabilities,
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
  request_timeout_ms: String.to_integer(System.get_env("REQUEST_TIMEOUT_MS", "30000")),
  # Per-job watchdog: if the oracle doesn't answer within this window the
  # worker SIGKILLs the (presumed wedged) Python process and recycles a
  # fresh interpreter, so a slow/hung chunk self-heals without a manual
  # container restart. Generous by default; lower it on fast CPU swarms.
  job_timeout_ms: String.to_integer(System.get_env("JOB_TIMEOUT_MS", "300000"))
