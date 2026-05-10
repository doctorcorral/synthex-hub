# Synthex Hub: Distributed Coinductive Synthesis

Synthex Hub is the distributed orchestration engine for **Coinductive
Symmetric Homomorphism Reinforcement Learning (CSHRL)**.

While deep RL needs centralized GPU clusters, CSHRL evaluates discrete
Boolean predicate boundaries — making policy synthesis *embarrassingly
parallel*. Synthex Hub separates the search from the simulation, so you
can drive a single CEGAR loop with an elastic mesh of compute nodes
contributed by collaborators.

---

## Donate compute (one line)

Got Docker and a few free cores? Help us tackle MuJoCo Humanoid:

```bash
curl -fsSL https://synthex.fit/install | sh
```

No token, no signup. Workers are anonymous; only batch
*submission* is authenticated. The script will:

1. verify Docker is installed and running,
2. build the worker image directly from
   `https://github.com/doctorcorral/synthex-hub` using Docker's remote-build
   feature (`docker build https://….git#main:worker`) — no clone,
   no registry login, no separate publishing step,
3. start a worker bound to `https://synthex.fit/api` with one Python
   port per CPU core,
4. print the next steps (`docker logs -f synthex-worker`,
   `docker stop synthex-worker`).

To update, friends just re-run the one-liner — it rebuilds from the
tip of `main` (the Docker layer cache makes re-runs fast).

Override defaults with environment variables:

```bash
POOL_SIZE=4 WORKER_NAME=alice-mbp \
  sh -c "$(curl -fsSL https://synthex.fit/install)"

# Or pin to a specific branch / fork:
BUILD_FROM=https://github.com/myfork/synthex-hub.git#feature:worker \
  sh -c "$(curl -fsSL https://synthex.fit/install)"

# Or skip the build and use a pre-baked registry image
# (for operators who want to publish; see deploy/publish-worker.sh):
IMAGE=ghcr.io/doctorcorral/synthex-worker:latest \
  sh -c "$(curl -fsSL https://synthex.fit/install)"
```

Live cluster status: <https://synthex.fit/> (workers / cores /
candidates evaluated / experiments completed). The landing page is
served by the hub itself; the installer is at
[`server/priv/static/install.sh`](server/priv/static/install.sh).

> Operators only: if the rebuild-on-every-run UX bothers you, you
> *can* pre-build and push a multi-arch image with
> [`./deploy/publish-worker.sh`](deploy/publish-worker.sh) and tell
> friends to set `IMAGE=…`. But the default — build from the public
> repo — has zero operator burden.

---

## Architecture

```
                     ┌────────────────────────────┐
   master script ───►│  hub server (synthex.fit)  │◄───┐
   (synthex CLI)     │                            │    │
                     │  • Postgres (durable)      │    │  HTTP poll-pull
                     │  • Oban queue + Lifeline   │    │  + Bearer token
                     │  • leases, retries, dedup  │    │
                     └────────────────────────────┘    │
                                                       │
                       ┌─────────────┬─────────────┬───┴─────────┐
                       ▼             ▼             ▼             ▼
                 ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐
                 │ worker  │   │ worker  │   │ worker  │   │ worker  │
                 │ (alice) │   │ (bob)   │   │ (carol) │   │ (dave)  │
                 └────┬────┘   └────┬────┘   └────┬────┘   └────┬────┘
                      │N ports     │N ports     │N ports     │N ports
                      ▼            ▼            ▼            ▼
                  python3      python3      python3      python3
                  (gym/MuJoCo) (gym/MuJoCo) (gym/MuJoCo) (gym/MuJoCo)
```

* **Server** is Elixir + Postgres + [Oban](https://github.com/oban-bg/oban).
  Chunks are stored as Oban jobs so they get persistence, lease-based
  rescue, retries, idempotency, and pruning out of the box. The `chunks`
  Oban queue runs at concurrency 0 because chunks are pulled by external
  HTTP workers, not executed in-process.
* **Worker** is an Elixir [Broadway](https://hexdocs.pm/broadway)
  pipeline. A custom GenStage producer HTTP-pulls chunks from the
  hub and emits one Broadway message per *candidate*; processors
  (one per CPU core) check out a Python port from the pool and score
  one candidate at a time; a sibling `ChunkAggregator` GenServer
  tracks per-chunk completion and submits results back to the hub
  once all candidates of a chunk report. Backpressure, telemetry,
  graceful shutdown, and per-port crash isolation come for free.
* **Auth** is master-only. `/api/master/*` (batch submission, batch
  reads) and `/api/status` require a Bearer token (`API_TOKEN`);
  `/api/worker/*` is open so anyone can donate compute. Result
  poisoning is mitigated at the *experiment* layer (multi-worker
  consensus, outlier detection on reward distributions) rather than
  at HTTP. See [`server/lib/server/auth.ex`](server/lib/server/auth.ex)
  for the full route table.

## End-to-end quickstart (local)

### 1. Start Postgres + the hub server

```bash
cd server
docker compose up -d postgres
mix deps.get
mix ecto.setup        # create db + run migrations
API_TOKEN=$(openssl rand -hex 32) mix run --no-halt
```

The server listens on `:4000`; remember the token.

### 2. Run a worker (native)

```bash
cd worker
mix deps.get
SERVER_URL=http://localhost:4000/api \
API_TOKEN=<the-token-from-step-1>     \
POOL_SIZE=4                            \
WORKER_NAME=$(hostname)-dev            \
mix run --no-halt
```

You should see `[Registrar] registered as ...` followed by
`[Pool] starting 4 python port(s)`.

### 2'. Or run a worker (Docker — what your friends will use)

```bash
cd worker
docker build -t synthex-worker .
docker run --rm \
  -e SERVER_URL=https://synthex.fit/api \
  -e API_TOKEN=<the-token>              \
  -e POOL_SIZE=8                        \
  -e WORKER_NAME=alice-mbp              \
  synthex-worker
```

### 3. Submit a batch from a master script

```python
import os, requests, time

SERVER_URL = os.environ.get("SYNTHEX_HUB_URL", "https://synthex.fit/api")
TOKEN      = os.environ["SYNTHEX_HUB_TOKEN"]
HEADERS    = {"Authorization": f"Bearer {TOKEN}"}

payload = {
    "env_name": "Humanoid-v5",
    "cmd": "score_bit",
    "bits_per_dim": 3,
    "max_steps": 1000,
    "target_bit": 0,
    "bit_predicates": ["falsep"] * (17 * 3),
    "seeds": [42, 123, 777],
    "chunk_size": 50,
    "candidates": [["feat", ["axis", 0, i * 0.05]] for i in range(2000)],
}

batch_id = requests.post(
    f"{SERVER_URL}/master/batches", json=payload, headers=HEADERS
).json()["batch_id"]

while True:
    s = requests.get(f"{SERVER_URL}/master/batches/{batch_id}", headers=HEADERS).json()
    print(f"{s['status']:>10} {s['progress']*100:6.1f}%  ({s['completed_chunks']}/{s['total_chunks']})")
    if s["status"] == "completed":
        flat = [r for chunk in s["results"] for r in chunk["items"]]
        best = max(flat, key=lambda r: r["reward"])
        print(f"best idx={best['idx']} reward={best['reward']}")
        break
    time.sleep(5)
```

Watch active workers and queue depth at any time:

```bash
curl -H "Authorization: Bearer $TOKEN" https://synthex.fit/api/status | jq
```

## Deploying at synthex.fit

Three production paths in [`deploy/`](deploy/):

* [`deploy/fly/`](deploy/fly/) ⭐ — Fly.io + Neon Postgres. Always-on,
  ~$0–10/mo, ~15 min from zero. First-class Elixir tooling
  (`fly logs`, `fly ssh console -C "/app/bin/server remote"` for an
  IEx into production).
* [`deploy/cloudflared/`](deploy/cloudflared/) — laptop-hosted, $0,
  no VPS. Postgres + server + cloudflared in one
  `docker compose up -d`. Best if you have a box that's always on
  and want zero recurring bills.
* [`deploy/vps-caddy/`](deploy/vps-caddy/) — VPS + Caddy + WireGuard
  for full control of the TLS edge.

The hub itself is *very* small (peak ~5% of one shared vCPU,
~250 MB RAM under a 32-worker swarm) — the bottleneck is always
Python on the workers, not orchestration. See
[`deploy/README.md`](deploy/README.md) for the decision tree and
sizing breakdown.

## Tackling MuJoCo Humanoid

Humanoid is `n_action_dims=17`. With `bits_per_dim=3` that's 51 bit
predicates × ~thousands of candidates per CEGAR round × dozens of
seeds × 1000 steps. A single laptop won't get there in a week — that's
exactly why the Hub exists.

Realistic worker math:
* an M2 / Ryzen 7 with 8 cores ≈ 50–100 candidate-seconds/sec
* 4 friends × 8 cores each ≈ 200–400 candidate-seconds/sec
* a CEGAR round with 1000 candidates × 5 seeds × 200 steps ≈
  one round / 5–10 minutes across the cluster

Once the Hub is up at `synthex.fit`, give friends the token + this
Docker incantation and you're a 32-core swarm by tonight.

## What the refactor solved (vs. v0.1)

Server side (Oban):

* **Persistence**: hub state survives restarts — chunks live in
  Postgres, not a `GenServer`.
* **Idempotent submission**: duplicate `submit` for the same
  `chunk_id` is a no-op.
* **Worker timeouts**: Oban Lifeline rescues any chunk whose lease
  expired, no manual reaping required.
* **Auth**: shared Bearer token enforced on every `/api/*` route.
* **Worker registration & heartbeat**: `/api/status` shows who's
  connected, total cores, queue depth.

Worker side (Broadway):

* **Streaming pipeline**: a Broadway pipeline turns a chunk into a
  stream of per-candidate messages. Producer pulls only when
  processors demand more (real backpressure); processors stay 100%
  utilized across chunk boundaries (no idle gap between chunks).
* **Per-port crash isolation**: one Python segfault no longer kills
  the worker; supervisor restarts that port only. The borrower
  process holding a port whose monitor fires gets the port
  auto-checked back into the pool.
* **Graceful shutdown**: Broadway's `prepare_for_draining/1` stops
  the producer from claiming new chunks; in-flight messages drain;
  only then does the worker exit.
* **Bounded-retry submission**: completed chunks ride a
  `Task.Supervisor` with exponential backoff; if all retries fail,
  Oban Lifeline rescues the chunk on the server side anyway.
* **Telemetry**: per-stage `:telemetry` events for chunks, candidates,
  polls, and Broadway's own processor events. Drop in your favourite
  exporter.
* **Always-reply Python protocol**: oracles return JSON errors instead
  of dying on stderr, so the Elixir port never hangs.

## See also

* [`server/README.md`](server/README.md) — server API reference,
  schema, and ops.
* [`worker/README.md`](worker/README.md) — worker config, Docker,
  adding new environments.
