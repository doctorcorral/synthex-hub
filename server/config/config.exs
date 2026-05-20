import Config

config :server,
  ecto_repos: [Server.Repo]

# Oban queues:
#
#   :chunks  — HTTP-pulled by external workers (concurrency 0,
#              paused). Oban handles persistence, lease/retry/prune.
#   :master  — Server.Workers.Experiment* run the entire CEGAR
#              synthesis loop here, in-process, with checkpoint
#              persistence per accepted bit. ONE master job per
#              experiment at a time (enforced by uniqueness),
#              and these jobs can run for many hours.
#   :system  — short-lived housekeeping (ReapWorkers, ...).
#
# `Lifeline.rescue_after` is the orphan-detection deadline. The
# master worker heartbeats `attempted_at` every 60 s (see
# `Server.Workers.ExperimentController`), so a HEALTHY step stays
# well inside any reasonable bound regardless of how long the
# underlying bit search takes. With a 60 s heartbeat, 5 min gives
# tolerance for transient DB hiccups while keeping orphan recovery
# fast after a crash or deploy. Previously this was 60 min, which
# left orphaned iters stuck "executing" for an hour after each
# deploy.
config :server, Oban,
  repo: Server.Repo,
  queues: [chunks: [limit: 1, paused: true], master: 4, system: 5],
  plugins: [
    {Oban.Plugins.Pruner, max_age: 60 * 60 * 24 * 7},
    {Oban.Plugins.Lifeline, rescue_after: :timer.minutes(5)},
    {Oban.Plugins.Cron, crontab: [
      {"*/5 * * * *", Server.Jobs.ReapWorkers},
      {"*/2 * * * *", Server.Jobs.OrphanReaper}
    ]}
  ]

import_config "#{config_env()}.exs"
