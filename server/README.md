# Synthex Hub Server

The orchestrator. Elixir + Bandit + Postgres + [Oban](https://hexdocs.pm/oban).

## What it does

Master clients submit a batch — `{env_name, cmd, candidates, ...}`.
The server splits `candidates` into chunks of `chunk_size` and inserts
each chunk as an `Oban.Job` row in Postgres. Then it sits and waits.

Workers HTTP-poll `/api/worker/jobs/request`. The server runs

```sql
SELECT id, args, ... FROM oban_jobs
WHERE state = 'available' AND queue = 'chunks'
ORDER BY priority, scheduled_at, id
LIMIT 1
FOR UPDATE SKIP LOCKED
```

inside a transaction, marks the row `executing` with the worker_id,
and returns the chunk payload. When the worker submits results via
`/api/worker/jobs/submit`, the row flips to `completed`, the batch's
`completed_chunks` is incremented, and (if it was the last chunk) the
batch's `status` flips to `completed` with aggregated results.

If a worker dies mid-chunk, Oban's
[`Lifeline` plugin](https://hexdocs.pm/oban/Oban.Plugins.Lifeline.html)
moves the orphaned `executing` row back to `available` after
`rescue_after` (15 minutes by default — tune in `config/config.exs`),
and the next worker picks it up. Per-chunk retries (max 5) are also
free.

The `chunks` Oban queue runs at concurrency 0 so Oban itself never
tries to execute jobs in-process.

## Setup

You need Elixir 1.18+ and a Postgres 13+.

```bash
docker compose up -d postgres        # local dev Postgres on :5432
mix deps.get
mix ecto.create
mix ecto.migrate
API_TOKEN=$(openssl rand -hex 32) mix run --no-halt
```

## Configuration

All runtime config is via environment variables:

| Variable                         | Default                                                            | Notes                                          |
|----------------------------------|--------------------------------------------------------------------|------------------------------------------------|
| `DATABASE_URL`                   | `postgres://postgres:postgres@localhost:5432/synthex_hub_dev` (dev)| Required in `MIX_ENV=prod`                     |
| `API_TOKEN`                      | (unset → all routes open)                                          | Master-only auth. When set, gates `/api/master/*` and `/api/status`. `/api/worker/*` is always open. |
| `PORT`                           | `4000`                                                             |                                                |
| `POOL_SIZE`                      | `10`                                                               | DB pool size                                   |
| `DEFAULT_CHUNK_SIZE`             | `100`                                                              | Used when master payload omits `chunk_size`    |
| `WORKER_HEARTBEAT_TIMEOUT_SECS`  | `120`                                                              | Workers older than this are marked inactive    |
| `DATABASE_SSL`                   | `false`                                                            | Set `true` for managed Postgres                |

## Production release

For one-off testing:

```bash
MIX_ENV=prod mix release
DATABASE_URL=postgres://...      \
API_TOKEN=$(cat /secrets/api_token) \
_build/prod/rel/server/bin/server eval "Server.Release.migrate()"
_build/prod/rel/server/bin/server start
```

For real deployment with Postgres + TLS / public domain, use the
Docker stacks in [`../deploy/`](../deploy/) (Cloudflare Tunnel or
VPS+Caddy). They wire up migrations, healthchecks, restart policies,
and TLS termination automatically.

## API

**Master-only auth.** Endpoints split into three classes:

* **Public** (always open): `/`, `/install`, `/install.sh`, `/health`,
  `/api/public-status`. The landing page + installer + aggregate
  counters live here.
* **Worker** (always open): `/api/worker/*`. Anyone can connect a
  worker — no token needed. Result-trust is handled at the
  experiment layer (multi-worker consensus, outlier detection on
  reward distributions).
* **Master** (gated by `API_TOKEN`): `/api/master/*` and the
  detailed `/api/status`. Only the master script submits batches,
  so only the master needs the token.

Send the master token as `Authorization: Bearer $API_TOKEN`. For
curl debugging you can also pass `?token=$API_TOKEN`.

### Health & ops

| Method | Path                            | Auth   | Notes                                                |
|--------|---------------------------------|--------|------------------------------------------------------|
| GET    | `/health`                       | open   | returns version                                      |
| GET    | `/api/public-status`            | open   | aggregate stats: workers / cores / candidates / experiments |
| GET    | `/api/status`                   | master | detailed: per-worker pool sizes, queue depth, etc.   |

### Worker endpoints (always open — no token)

| Method | Path                            | Notes                                                        |
|--------|---------------------------------|--------------------------------------------------------------|
| POST   | `/api/worker/register`          | Body: `{id, name, hostname, pool_size, version, metadata}`. Idempotent (upsert). |
| POST   | `/api/worker/heartbeat`         | Body: `{worker_id}`. Bumps `last_heartbeat_at`.              |
| GET    | `/api/worker/jobs/request?worker_id=…` | 200 with chunk payload, or 204 if queue is empty.            |
| POST   | `/api/worker/jobs/submit`       | Body: `{chunk_id, worker_id, results}`. Idempotent (no-op if already completed). |

### Master endpoints (require `API_TOKEN`)

| Method | Path                                | Notes                                                |
|--------|-------------------------------------|------------------------------------------------------|
| POST   | `/api/master/batches`               | Body: see schema below. 201 with `{batch_id, total_chunks}`. |
| GET    | `/api/master/batches/:batch_id`     | Returns progress and (when complete) aggregated results.    |
| GET    | `/api/master/batches?limit=…`       | Lists recent batches for ops dashboards.             |

#### Batch payload

```json
{
  "env_name": "Humanoid-v5",
  "cmd": "score_bit",
  "name": "humanoid-round-7-bit-0",
  "bits_per_dim": 3,
  "max_steps": 1000,
  "target_bit": 0,
  "bit_predicates": ["falsep", "falsep", ...],
  "seeds": [42, 123, 777],
  "chunk_size": 50,
  "candidates": [["feat", ["axis", 0, 0.05]], ...]
}
```

`cmd` defaults to `"score_bit"`. New commands plug into the same
chunking + leasing infrastructure — just add a dispatch case in
`worker/environments/gymnasium/oracle_port.py` and the workers will
handle it.

## Why Oban (vs. a custom queue)

* `oban_jobs` already has the columns we want: `state`, `attempt`,
  `attempted_at`, `attempted_by`, `discarded_at`, with the right
  indexes. We get that table for free instead of designing one.
* `Oban.Plugins.Lifeline` rescues stuck `executing` rows on a
  schedule. Don't have to write or test reaping logic.
* `Oban.Plugins.Pruner` keeps the table from growing unboundedly.
* `Oban.Plugins.Cron` runs the `Server.Jobs.ReapWorkers` task every
  5 minutes to mark dead workers as inactive.
* If we ever want a built-in dashboard, [Oban Web](https://hexdocs.pm/oban_web)
  drops in.

The chunk queue runs at concurrency 0 because we deliberately *don't*
want Oban to execute the chunks itself — they're for external HTTP
workers. Oban becomes pure persistence + scheduling infrastructure.

## Schema

* `batches` — one row per master submission. Tracks status,
  total/completed chunk counts, aggregated results.
* `oban_jobs` — one row per chunk. Args contain `{batch_id, chunk_id,
  chunk_index, cmd, env_name, candidates, params}`.
* `workers` — registered compute nodes; bumped on heartbeat.
