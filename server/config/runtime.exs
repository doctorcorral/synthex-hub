import Config

if config_env() == :prod do
  database_url =
    System.get_env("DATABASE_URL") ||
      raise """
      environment variable DATABASE_URL is missing.
      Example: postgres://USER:PASS@HOST:5432/DB
      """

  # Default to ssl: true in prod — every hosted Postgres (Neon,
  # Supabase, Fly, RDS) requires it. Override with DATABASE_SSL=false
  # if you're behind a private network where TLS would just add cost.
  ssl_enabled = System.get_env("DATABASE_SSL", "true") == "true"

  ssl_opts =
    if ssl_enabled do
      [verify: :verify_none]
    else
      false
    end

  config :server, Server.Repo,
    url: database_url,
    # Pool sizing history:
    #   10 → saturated almost immediately
    #   30 → held up with 1–2 active experiments
    #   60 → currently default, comfortable headroom for 3
    #        concurrent CEGAR controllers + AggregateBroker +
    #        Oban polling + a swarm of worker chunk submissions.
    #
    # Postgres on Neon comfortably serves >100 concurrent backends;
    # the bottleneck is always our app-side pool, never the
    # database. Crank POOL_SIZE higher (via Fly secret) if you
    # find DBConnection.ConnectionError reappearing — it's free as
    # long as we stay under the Postgres max_connections.
    #
    # queue_target / queue_interval govern Ecto's pool-overload
    # heuristic: if checkouts wait longer than queue_target for
    # more than half of a queue_interval window, the pool starts
    # DROPPING requests with a ConnectionError. Old values
    # (1_000ms / 5_000ms) were tuned for user-facing latency —
    # but we're a master-and-workers backend, none of these
    # callers are humans. Better to wait 5s for a connection than
    # to crash an Oban controller mid-CEGAR-pass and burn through
    # its max_attempts on a transient blip.
    pool_size: String.to_integer(System.get_env("POOL_SIZE", "60")),
    queue_target: String.to_integer(System.get_env("DB_QUEUE_TARGET_MS", "5000")),
    queue_interval: String.to_integer(System.get_env("DB_QUEUE_INTERVAL_MS", "30000")),
    timeout: 60_000,
    ssl: ssl_enabled,
    ssl_opts: ssl_opts
end

# Auth token shared with workers. Required in prod, optional in dev.
config :server,
  api_token: System.get_env("API_TOKEN"),
  port: String.to_integer(System.get_env("PORT", "4000")),
  default_chunk_size: String.to_integer(System.get_env("DEFAULT_CHUNK_SIZE", "10")),
  worker_heartbeat_timeout_secs:
    String.to_integer(System.get_env("WORKER_HEARTBEAT_TIMEOUT_SECS", "120"))
