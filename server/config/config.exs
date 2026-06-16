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
    # Per-chunk results on `oban_jobs.args` are consumed by
    # end-of-batch readers within MINUTES of completion, so terminal
    # chunk rows have no value after that — they're pure bloat that
    # slows every COUNT(available)/claim SKIP LOCKED/insert against the
    # table (the contention behind the 57014 statement timeouts). With
    # the per-bit pool now bounded (synthex `max_candidates`), chunk
    # churn is far lower, but short retention compounds the win. 3h
    # (was 24h) keeps a generous margin for a slow controller to fetch
    # results while keeping the table small.
    {Oban.Plugins.Pruner, max_age: 60 * 60 * 3},
    {Oban.Plugins.Lifeline, rescue_after: :timer.minutes(5)},
    {Oban.Plugins.Cron, crontab: [
      {"*/5 * * * *", Server.Jobs.ReapWorkers},
      {"*/2 * * * *", Server.Jobs.OrphanReaper},
      # Keep claim_chunk's planner stats fresh against bursty
      # available-chunk counts; without this a stale "1 available
      # row" estimate makes claim_chunk pick an O(N²) plan and hang.
      {"*/3 * * * *", Server.Jobs.AnalyzeObanJobs},
      # Sweep consumed per-chunk results (narrow chunk_results table)
      # that nothing prunes otherwise. Hourly is plenty — rows live ~6h.
      {"0 * * * *", Server.Jobs.PruneChunkResults}
    ]}
  ]

import_config "#{config_env()}.exs"
