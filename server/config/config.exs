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
    # 1 day, was 7. Chunk jobs churn fast (a stuck worker can spawn
    # and cancel thousands per hour), and at ~250k terminal rows the
    # unindexed `complete_chunk` lookup seq-scanned past the worker's
    # 30s HTTP timeout (see migration 20260611000001). Per-chunk
    # results on `oban_jobs.args` are consumed by end-of-batch
    # readers within minutes of completion, so a day of retention is
    # far more than enough — and keeps Neon storage/egress in check.
    {Oban.Plugins.Pruner, max_age: 60 * 60 * 24},
    {Oban.Plugins.Lifeline, rescue_after: :timer.minutes(5)},
    {Oban.Plugins.Cron, crontab: [
      {"*/5 * * * *", Server.Jobs.ReapWorkers},
      {"*/2 * * * *", Server.Jobs.OrphanReaper},
      # Keep claim_chunk's planner stats fresh against bursty
      # available-chunk counts; without this a stale "1 available
      # row" estimate makes claim_chunk pick an O(N²) plan and hang.
      {"*/3 * * * *", Server.Jobs.AnalyzeObanJobs}
    ]}
  ]

import_config "#{config_env()}.exs"
