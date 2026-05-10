# Synthex Hub Worker

The compute node. A Broadway pipeline that pulls chunks from the hub,
fans candidates out across a pool of Python ports, and submits
results back when each chunk completes.

## Architecture

```
                    ┌──────────────────────────────┐
                    │  Worker.Pipeline.Producer    │
                    │  (custom GenStage producer)  │
                    │                              │
                    │  • HTTP-pulls chunks from    │
                    │    /api/worker/jobs/request  │
                    │  • Emits one Broadway.Message│
                    │    per CANDIDATE             │
                    │  • Backpressure: only pulls  │
                    │    when downstream demands   │
                    │  • prepare_for_draining/1    │
                    │    stops polling on shutdown │
                    └──────────┬───────────────────┘
                               │ Broadway.Message
                               │ {data: candidate, metadata: {chunk_id, total, ...}}
                               ▼
                    ┌──────────────────────────────┐
                    │  Worker.Pipeline (Broadway)  │
                    │  processors :default         │
                    │  concurrency = pool_size     │
                    │  max_demand = 1              │
                    └──────────┬───────────────────┘
                               │  for each message:
                               │    1. Worker.PortPool.with_port/2
                               │    2. Worker.PythonPort.score/3
                               │    3. ChunkAggregator.add_result/2
                               │
                ┌──────────────┼──────────────┬──────────────┐
                ▼              ▼              ▼              ▼
           PythonPort 0   PythonPort 1   PythonPort 2 ... PythonPort N-1
              │               │              │                │
              ▼               ▼              ▼                ▼
           python3 oracle_port.py  (one persistent interpreter per port)

                               │
                               ▼
                    ┌──────────────────────────────┐
                    │  Worker.ChunkAggregator      │
                    │                              │
                    │  • per-chunk state:          │
                    │    %{total, results}         │
                    │  • on full chunk →           │
                    │    Task.Supervisor + retries │
                    │    POST /worker/jobs/submit  │
                    └──────────────────────────────┘
```

### Why Broadway here?

Broadway gives us, in ~150 lines of pipeline code:

* **Backpressure** — the producer only HTTP-polls the hub when the
  processor pool has capacity. No more "claim chunks then sit on
  them while the ports are busy".
* **Per-port utilization** — `max_demand: 1` on processors means
  every Python port is always working on one candidate. No idle gaps
  at chunk boundaries.
* **Crash isolation** — a Python segfault crashes one port; the
  supervisor restarts it; in-flight messages on other ports keep
  flowing. The borrower process holding a port that crashes has its
  monitor fire and the port is auto-checked back in.
* **Graceful shutdown** — Broadway's `prepare_for_draining/1`
  callback stops the producer from claiming new chunks; in-flight
  messages drain cleanly; only then does the worker exit.
* **Telemetry** — every Broadway processor emits standard
  `[:broadway, ...]` events. We additionally emit our own
  `[:synthex_hub, :worker, :*]` events (see `Worker.Telemetry`)
  for chunk-level visibility (claimed / completed / submitted).
* **Retries** — at the candidate level via `handle_failed/2`, at
  the chunk-submission level via `ChunkAggregator`'s retry loop.

### Why the aggregator is *not* a Broadway batcher

Broadway's batchers flush on `batch_size` OR `batch_timeout`, both
fixed at compile time. Chunks have variable sizes (the last chunk of
a master submission may be partial). Either we'd flush partial chunks
prematurely, or we'd wait the full timeout on every last chunk. A
sibling GenServer keyed by `chunk_id` with a known `total` (carried
in message metadata) handles this in tens of lines, with no
correctness compromises.

### Component table

| Module                          | Role                                                      |
|---------------------------------|-----------------------------------------------------------|
| `Worker.PortSupervisor`         | Supervises N `Worker.PythonPort` children + Registry      |
| `Worker.PythonPort`             | One persistent Python interpreter behind a Port           |
| `Worker.PortPool`               | `checkout/checkin` GenServer, monitors borrowers          |
| `Worker.Pipeline.Producer`      | GenStage producer; HTTP-polls; emits `Broadway.Message`s  |
| `Worker.Pipeline`               | Broadway pipeline; `handle_message/3` scores one candidate |
| `Worker.ChunkAggregator`        | Per-chunk completion + bounded-retry HTTP submit          |
| `Worker.Registrar`              | Registers + heartbeats with the hub                       |
| `Worker.HttpClient`             | Req wrapper: base URL + Bearer token + timeout            |
| `Worker.Telemetry`              | Attaches log handlers to Broadway / our events            |

### Supervision strategy

`rest_for_one`: if `PortPool` dies, `Pipeline` must restart too
(its `handle_message/3` borrows from the pool). If `PortSupervisor`
dies, the whole subtree restarts.

```
Worker.Supervisor (rest_for_one)
├── Worker.PortSupervisor              (one_for_one)
│   ├── Worker.PortRegistry
│   ├── Worker.PythonPort {:port, 0}
│   ├── Worker.PythonPort {:port, 1}
│   └── … N total
├── Worker.PortPool                    (single GenServer)
├── Worker.SubmitTaskSupervisor        (Task.Supervisor)
├── Worker.ChunkAggregator             (single GenServer)
├── Worker.Pipeline                    (Broadway → owns its own subtree)
│   ├── Producer
│   └── Processors × pool_size
└── Worker.Registrar                   (single GenServer)
```

## Donating compute (Docker)

The friction-free path is the [public installer](https://synthex.fit/install):

```bash
curl -fsSL https://synthex.fit/install | sh
```

If you'd rather invoke Docker yourself:

```bash
docker run --rm \
  -e SERVER_URL=https://synthex.fit/api \
  -e WORKER_NAME=$(hostname)            \
  -e POOL_SIZE=8                        \
  doctorcorral/synthex-worker:latest
```

No token required — workers are anonymous. Or persistent:

```bash
SERVER_URL=https://synthex.fit/api docker compose up -d
```

When you stop the worker, Broadway gracefully drains in-flight
messages before exiting. The hub marks you `inactive` within ~2
minutes; any chunk you didn't get to submit is rescued by Oban
Lifeline and reassigned to another worker.

## Running natively (development)

```bash
cd environments/gymnasium && pip install -r requirements.txt && cd ../..
SERVER_URL=http://localhost:4000/api POOL_SIZE=4 \
  WORKER_NAME=$(hostname)-dev mix run --no-halt
```

## Configuration (env vars)

| Variable                | Default                                     | Notes                                                |
|-------------------------|---------------------------------------------|------------------------------------------------------|
| `SERVER_URL`            | `http://localhost:4000/api`                 | Trailing `/api` is required.                         |
| `API_TOKEN`             | (unset)                                     | Optional. Workers are anonymous on public hubs; only set if your hub requires worker auth. |
| `POOL_SIZE`             | `System.schedulers_online()`                | One Python interpreter per pool slot.                |
| `WORKER_NAME`           | `<hostname>-<unix-time>`                    | Stable across restarts is better.                    |
| `HOSTNAME`              | `:inet.gethostname()`                       | Reported in `/api/status`.                           |
| `PYTHON`                | `python3`                                   | Path / name of the python executable.                |
| `ORACLE_SCRIPT`         | `<root>/environments/gymnasium/oracle_port.py` | Override to point at a different oracle.          |
| `POLL_INTERVAL_MS`      | `2000`                                      | Sleep when the hub queue is empty.                   |
| `HEARTBEAT_INTERVAL_MS` | `30000`                                     |                                                      |
| `REQUEST_TIMEOUT_MS`    | `30000`                                     | Per-HTTP-call timeout (Req's `receive_timeout`).     |

## Hooking up your own metrics

Every meaningful event the worker emits goes through `:telemetry`:

```elixir
:telemetry.attach_many(
  "my-metrics",
  [
    [:synthex_hub, :worker, :chunk, :claimed],
    [:synthex_hub, :worker, :chunk, :completed],
    [:synthex_hub, :worker, :chunk, :submitted],
    [:synthex_hub, :worker, :chunk, :submit_failed],
    [:synthex_hub, :worker, :candidate, :scored],
    [:synthex_hub, :worker, :poll, :empty],
    [:synthex_hub, :worker, :poll, :error],
    # plus all of [:broadway, ...]
  ],
  &MyMetrics.handle_event/4,
  nil
)
```

Drop in `telemetry_metrics_prometheus`, OpenTelemetry, or
`telemetry_poller` as fits your ops stack.

## Adding new environments

For Gymnasium environments, add an entry to `ENV_CONFIGS` in
`environments/gymnasium/oracle_port.py`. For a fundamentally
different simulator:

1. Create `environments/<your-sim>/oracle_port.py` that obeys the
   protocol below.
2. Set `ORACLE_SCRIPT` env var on the affected workers.

### Oracle protocol

* Read one JSON object per line from `stdin`.
* Each input has at least `{"job_id": int, "cmd": str}` plus
  command-specific fields.
* Write one JSON object per line to `stdout`. **Always** include the
  original `job_id`. On success: `{"job_id": id, "results": [...]}`.
  On failure: `{"job_id": id, "error": "..."}`. Never just write to
  stderr — the Elixir port is waiting on a JSON line.

## Smoke test

```bash
# Terminal A — the hub:
cd ../server && API_TOKEN=foo PORT=4000 mix run --no-halt

# Terminal B — submit a tiny batch via curl, then start a worker:
SERVER_URL=http://localhost:4000/api API_TOKEN=foo POOL_SIZE=2 \
  mix run --no-halt
```

For a more direct port-only test (no pool, no pipeline), bypass
everything:

```bash
mix run test_ant_poc.exs
```
