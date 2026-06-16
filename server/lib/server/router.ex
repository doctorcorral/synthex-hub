defmodule Server.Router do
  use Plug.Router
  require Logger

  plug Plug.Logger, log: :info

  # Serve everything under priv/static at the root URL — landing page
  # (index.html), worker installer (install.sh), and the static
  # showcase page (showcase/). Cached for an hour; bust by redeploying.
  plug Plug.Static,
    at: "/",
    from: {:server, "priv/static"},
    only: ~w(index.html install.sh favicon.ico robots.txt assets showcase),
    cache_control_for_etags: "public, max-age=3600"

  plug :match

  # Body limit: workers submitting `collect_states` chunks can return
  # tens of megabytes of trajectories at a time (e.g. Ant: 1000 steps
  # × 105 obs floats × N seeds). Default is 8 MB, which we'd routinely
  # blow through. Cap at 64 MB; chunks larger than that should be
  # subdivided rather than handled here.
  plug Plug.Parsers,
    parsers: [:json],
    pass: ["application/json"],
    json_decoder: Jason,
    length: 64_000_000,
    read_length: 1_000_000,
    read_timeout: 60_000

  plug Server.Auth
  plug :dispatch

  # ── Landing page & installer (unauthenticated) ──────────────

  get "/" do
    serve_static(conn, "index.html", "text/html; charset=utf-8")
  end

  # Showcase: hand-curated static page with policy videos. Both
  # `/showcase` and `/showcase/` resolve to the index file (Plug.Router
  # normalizes trailing slashes, so a single route covers both).
  # Everything under `/showcase/*` is served verbatim by Plug.Static
  # above. The HTML carries `<base href="/showcase/">` so relative
  # asset paths (e.g. `videos/foo.mp4`) always resolve under
  # /showcase/ regardless of which URL form the visitor lands on.
  get "/showcase" do
    serve_static(conn, "showcase/index.html", "text/html; charset=utf-8")
  end

  # `curl https://synthex.fit/install | sh` — no extension, served as
  # a shell script so it pipes straight into sh.
  get "/install" do
    serve_static(conn, "install.sh", "text/x-shellscript; charset=utf-8")
  end

  get "/install.sh" do
    serve_static(conn, "install.sh", "text/x-shellscript; charset=utf-8")
  end

  # ── Health & ops ────────────────────────────────────────────

  get "/health" do
    send_json(conn, 200, %{status: "ok", version: Application.spec(:server, :vsn) |> to_string()})
  end

  get "/api/public-status" do
    conn
    |> put_public_headers()
    |> send_json(200, Server.Queue.public_status())
  end

  # Live load stream — Server-Sent Events, one frame per second
  # carrying %Server.Metrics.snapshot/0 fields. Public, CORS-open,
  # cheap (every client reads from the MetricsBroker ETS cache, no
  # DB queries per connection). EventSource clients will reconnect
  # automatically when this hits its 5-min duration cap.
  get "/api/public-status/stream" do
    Server.SSEStream.serve(conn, latest_fn: &Server.MetricsBroker.latest/0)
  end

  # Per-experiment / per-active-batch streaming aggregates. Layer 1c
  # of docs/streaming-cegar.md. Pushes live mean / stddev /
  # rate-per-min for whatever batch is currently in flight on each
  # active experiment, every second.
  get "/api/public-status/stream/aggregates" do
    Server.SSEStream.serve(conn, latest_fn: &Server.AggregateBroker.latest/0)
  end

  # CORS preflight for browsers fetching public-status from any
  # origin (showcase pages, embeds, etc.).
  options "/api/public-status" do
    send_cors_preflight(conn)
  end

  options "/api/public-status/experiments" do
    send_cors_preflight(conn)
  end

  options "/api/public-status/leaderboard" do
    send_cors_preflight(conn)
  end

  options "/api/public-status/batches/:_batch_id/contributors" do
    send_cors_preflight(conn)
  end

  # Per-environment view of in-flight + recently-completed
  # experiments. Reads from the `experiments` table (the canonical
  # CEGAR-run record), not from per-bit Batch rows. One row per
  # env: status, progress (cegar_iter/iter), accepted bit count,
  # best reward, baseline, health.
  get "/api/public-status/experiments" do
    conn
    |> put_public_headers()
    |> send_json(200, %{experiments: Server.Experiments.summary()})
  end

  # Recent system incidents — anything that would otherwise silently
  # break the swarm. Drives the red banner on the landing page.
  # 24h window so a brief outage stays visible long enough that
  # someone notices.
  get "/api/public-status/incidents" do
    conn
    |> put_public_headers()
    |> send_json(200, %{incidents: Server.Experiments.recent_incidents()})
  end

  options "/api/public-status/incidents" do
    send_cors_preflight(conn)
  end

  # All-time top contributors. Anonymous workers (`name == "anonymous"`)
  # collapse into a single bucket. Cached briefly so the landing page
  # can poll without hammering Postgres.
  get "/api/public-status/leaderboard" do
    limit = parse_limit(conn.params["limit"], default: 20, max: 100)

    conn
    |> put_public_headers()
    |> send_json(200, %{contributors: Server.Queue.leaderboard(limit: limit)})
  end

  # Per-experiment contributors. Public so a master can share a batch_id
  # link with their friends and they can see who-did-what.
  get "/api/public-status/batches/:batch_id/contributors" do
    limit = parse_limit(conn.params["limit"], default: 50, max: 500)

    conn
    |> put_public_headers()
    |> send_json(200, %{
      batch_id: batch_id,
      contributors: Server.Queue.batch_contributors(batch_id, limit: limit)
    })
  end

  # Latest policy snapshot for a lineage. Public; CORS-enabled so
  # the showcase / landing page can fetch from any origin.
  #
  # Keyed by env_policy_id (UUID) — see Server.PolicySnapshot's
  # moduledoc for why per-lineage is the natural grain (two
  # configs on the same env_name are independent lineages and
  # have independent snapshots).
  get "/api/public-status/policies/:env_policy_id" do
    case Server.Queue.get_policy_snapshot(env_policy_id) do
      {:ok, snapshot} ->
        conn
        |> put_public_headers()
        |> send_json(200, render_snapshot(snapshot))

      {:error, :not_found} ->
        conn
        |> put_public_headers()
        |> send_json(404, %{error: "snapshot_not_found", env_policy_id: env_policy_id})
    end
  end

  options "/api/public-status/policies/:_env_policy_id" do
    send_cors_preflight(conn)
  end

  get "/api/status" do
    send_json(conn, 200, Server.Queue.status())
  end

  # ── Worker endpoints ────────────────────────────────────────

  post "/api/worker/register" do
    attrs = conn.body_params

    case Server.Queue.register_worker(attrs) do
      {:ok, worker} ->
        send_json(conn, 200, %{
          worker_id: worker.id,
          heartbeat_interval_secs: 30
        })

      {:error, changeset} ->
        send_json(conn, 422, %{
          error: "invalid_worker",
          details: format_errors(changeset)
        })
    end
  end

  post "/api/worker/heartbeat" do
    worker_id = conn.body_params["worker_id"] || conn.body_params["id"]

    case Server.Queue.heartbeat(worker_id) do
      :ok -> send_json(conn, 200, %{status: "ok"})
      {:error, :unknown_worker} -> send_json(conn, 404, %{error: "unknown_worker"})
    end
  end

  get "/api/worker/jobs/request" do
    worker_id = conn.params["worker_id"] || conn.params["id"]

    case Server.Queue.claim_chunk(worker_id || "anonymous") do
      :empty -> send_json(conn, 204, %{})
      {:ok, payload} -> send_json(conn, 200, payload)
    end
  end

  post "/api/worker/jobs/submit" do
    chunk_id = conn.body_params["chunk_id"] || conn.body_params["job_id"]
    results = conn.body_params["results"] || []
    worker_id = conn.body_params["worker_id"]

    case Server.Queue.complete_chunk(chunk_id, results, worker_id) do
      {:ok, _} -> send_json(conn, 200, %{status: "ok"})
      {:error, :not_found} -> send_json(conn, 404, %{error: "unknown_chunk"})
    end
  end

  # ── Master endpoints ────────────────────────────────────────

  # Submit a new experiment. The hub spawns an Oban-supervised
  # master loop that runs CEGAR end-to-end on the server, checkpointing
  # after every accepted bit, and the resulting policy is stored on
  # the long-lived `env_policies` row for the submission's
  # `(env_name, config_sig)`. Subsequent submissions inherit the
  # lineage automatically.
  #
  # Body:
  #
  #     {
  #       "env_key": "ant",
  #       "env_name": "Ant-v5",
  #       "config": { "bits_per_dim": 3, "depth": 1, ... }
  #     }
  #
  # Rejects with 409 if an experiment for the same
  # `(env_name, config_sig)` lineage is already active. Two
  # submissions for the same env_name with DIFFERENT policy-shape
  # configs (different bits_per_dim etc.) get separate lineages and
  # run in parallel.
  post "/api/master/experiments" do
    submitter = List.first(get_req_header(conn, "x-submitter"))

    case Server.Experiments.create(conn.body_params, submitter: submitter) do
      {:ok, exp} ->
        send_json(conn, 201, %{
          id: exp.id,
          env_name: exp.env_name,
          env_key: exp.env_key,
          status: exp.status,
          submitter: exp.submitter
        })

      {:error, :missing_env} ->
        send_json(conn, 422, %{
          error: "missing_env",
          message: "Body must include env_key (e.g. \"ant\") and env_name (e.g. \"Ant-v5\")"
        })

      {:error, {:unknown_env, env_key}} ->
        known = Synthex.Gym.Mujoco.known_envs() |> Enum.map(&Atom.to_string/1) |> Enum.sort()

        send_json(conn, 422, %{
          error: "unknown_env",
          env_key: env_key,
          known_envs: known
        })

      {:error, :already_running} ->
        send_json(conn, 409, %{
          error: "already_running",
          message:
            "An experiment for this (env_name, config) lineage is already pending/running. " <>
              "Submissions for the same env with a DIFFERENT policy-shape config " <>
              "(bits_per_dim, depth, feature_types, max_coeff, tridiag_*) are accepted in parallel."
        })

      {:error, reason} ->
        send_json(conn, 500, %{error: "create_failed", reason: inspect(reason)})
    end
  end

  get "/api/master/experiments" do
    limit = parse_limit(conn.params["limit"], default: 50, max: 200)

    experiments =
      Server.Experiments.list(limit)
      |> Enum.map(fn e ->
        %{
          id: e.id,
          env_name: e.env_name,
          env_key: e.env_key,
          status: e.status,
          baseline_reward: e.baseline_reward,
          best_reward: e.best_reward,
          accepted_count: e.accepted_count,
          current_cegar_iter: e.current_cegar_iter,
          current_iter: e.current_iter,
          inserted_at: e.inserted_at,
          completed_at: e.completed_at,
          error: e.error,
          submitter: e.submitter
        }
      end)

    send_json(conn, 200, %{experiments: experiments})
  end

  get "/api/master/experiments/:id" do
    case Server.Experiments.get(id) do
      {:ok, e} ->
        env_policy =
          case e.env_policy_id do
            nil ->
              nil

            _ ->
              case Server.EnvPolicies.for_experiment(e) do
                {:ok, ep} -> ep
                _ -> nil
              end
          end

        body = %{
          id: e.id,
          env_name: e.env_name,
          env_key: e.env_key,
          status: e.status,
          config: e.config,
          baseline_reward: e.baseline_reward,
          best_reward: e.best_reward,
          accepted_count: e.accepted_count,
          current_cegar_iter: e.current_cegar_iter,
          current_iter: e.current_iter,
          inserted_at: e.inserted_at,
          started_at: e.started_at,
          completed_at: e.completed_at,
          error: e.error,
          submitter: e.submitter,
          env_policy_id: e.env_policy_id,
          env_policy:
            env_policy &&
              %{
                id: env_policy.id,
                env_name: env_policy.env_name,
                config_sig: env_policy.config_sig,
                config_data: env_policy.config_data,
                policy_version: env_policy.policy_version,
                predicates: env_policy.predicates,
                validation_avg: env_policy.validation_avg,
                validation_version: env_policy.validation_version,
                validation_tail: env_policy.validation_tail,
                best_reward: env_policy.best_reward,
                baseline_reward: env_policy.baseline_reward,
                n_episodes: env_policy.n_episodes,
                first_seen_at: env_policy.first_seen_at,
                updated_at: env_policy.updated_at
              }
        }

        send_json(conn, 200, body)

      {:error, :not_found} ->
        send_json(conn, 404, %{error: "experiment_not_found"})
    end
  end

  # Cancel an experiment. Idempotent: cancelling an already-finished
  # one is a no-op. The OrphanReaper will clean up any in-flight
  # chunks within 2 minutes.
  post "/api/master/experiments/:id/cancel" do
    case Server.Experiments.get(id) do
      {:ok, e} ->
        reason = Map.get(conn.body_params, "reason", "cancelled by operator")

        case e.status do
          s when s in ["pending", "running"] ->
            {:ok, _} = Server.Experiments.mark_cancelled(e, reason)
            Server.Experiments.log_event!(
              "warn",
              "master",
              "experiment cancelled by operator: #{e.env_name} — #{reason}",
              env_name: e.env_name,
              experiment_id: e.id
            )
            send_json(conn, 200, %{status: "cancelled"})

          other ->
            send_json(conn, 200, %{status: other, message: "already #{other}"})
        end

      {:error, :not_found} ->
        send_json(conn, 404, %{error: "experiment_not_found"})
    end
  end

  post "/api/master/batches" do
    payload = conn.body_params
    submitter = List.first(get_req_header(conn, "x-submitter"))

    case Server.Queue.submit_batch(payload, submitter: submitter) do
      {:ok, batch} ->
        send_json(conn, 201, %{
          batch_id: batch.id,
          total_chunks: batch.total_chunks,
          status: batch.status
        })

      {:error, reason} ->
        send_json(conn, 422, %{error: "submit_failed", reason: inspect(reason)})
    end
  end

  # Default: a SLIM response with progress and reward summary only.
  # `?include_results=1` opts into the heavy `results` array — the
  # master fetches that exactly once on completion, so we don't
  # restream multi-MB of accumulated chunk payloads on every poll.
  get "/api/master/batches/:batch_id" do
    if include_results?(conn.params["include_results"]) do
      send_full_batch(conn, batch_id)
    else
      send_batch_progress(conn, batch_id)
    end
  end

  # Upsert the latest policy snapshot for an env. The master calls
  # this from its telemetry handler on each accepted CEGAR step.
  # One row per `env_name`; subsequent posts overwrite. Returns the
  # canonical snapshot so the caller can confirm what's stored.
  post "/api/master/policy-snapshots" do
    payload = conn.body_params
    submitter = List.first(get_req_header(conn, "x-submitter"))

    case Server.Queue.upsert_policy_snapshot(payload, submitter: submitter) do
      {:ok, snapshot} ->
        send_json(conn, 200, render_snapshot(snapshot))

      {:error, changeset} ->
        send_json(conn, 422, %{
          error: "invalid_snapshot",
          details: format_errors(changeset)
        })
    end
  end

  get "/api/master/batches" do
    limit =
      case Integer.parse(conn.params["limit"] || "50") do
        {n, _} when n > 0 and n <= 500 -> n
        _ -> 50
      end

    batches =
      Server.Queue.list_batches(limit)
      |> Enum.map(fn b ->
        %{
          batch_id: b.id,
          name: b.name,
          env_name: b.env_name,
          status: b.status,
          total_chunks: b.total_chunks,
          completed_chunks: b.completed_chunks,
          inserted_at: b.inserted_at
        }
      end)

    send_json(conn, 200, %{batches: batches})
  end

  match _ do
    send_json(conn, 404, %{error: "not_found"})
  end

  # ── Helpers ─────────────────────────────────────────────────

  defp include_results?(value) do
    case value do
      "1" -> true
      "true" -> true
      "TRUE" -> true
      _ -> false
    end
  end

  defp send_batch_progress(conn, batch_id) do
    Server.Queue.touch_master_poll(batch_id)

    case Server.Queue.get_batch_progress(batch_id) do
      {:ok, row} ->
        send_json(conn, 200, %{
          batch_id: row.id,
          name: row.name,
          env_name: row.env_name,
          cmd: row.cmd,
          status: row.status,
          total_chunks: row.total_chunks,
          completed_chunks: row.completed_chunks,
          progress:
            if(row.total_chunks > 0,
              do: row.completed_chunks / row.total_chunks,
              else: 1.0
            ),
          best_reward: row.best_reward,
          baseline_reward: row.baseline_reward,
          inserted_at: row.inserted_at,
          completed_at: row.completed_at
        })

      {:error, :not_found} ->
        send_json(conn, 404, %{error: "batch_not_found"})
    end
  end

  defp send_full_batch(conn, batch_id) do
    Server.Queue.touch_master_poll(batch_id)

    # Two-phase: lightweight progress lookup for the metadata, then
    # per-chunk items pulled from oban_jobs.args["results"]. Used to
    # serve everything from a single `Repo.get(Batch)` that included
    # the `Batch.results` column — but that column is no longer
    # populated (see `Server.Queue.fetch_batch_chunks/1`'s docs for
    # the O(N²) egress story). The wire shape is preserved: callers
    # still get `results: [%{"chunk_index" => i, "items" => [...]}]`
    # sorted by chunk_index, so `Synthex.Hub.Client` keeps working
    # unchanged.
    with {:ok, progress} <- Server.Queue.get_batch_progress(batch_id),
         {:ok, chunks} <- Server.Queue.fetch_batch_chunks(batch_id) do
      send_json(conn, 200, %{
        batch_id: progress.id,
        name: progress.name,
        env_name: progress.env_name,
        cmd: progress.cmd,
        status: progress.status,
        total_chunks: progress.total_chunks,
        completed_chunks: progress.completed_chunks,
        progress:
          if(progress.total_chunks > 0,
            do: progress.completed_chunks / progress.total_chunks,
            else: 1.0
          ),
        best_reward: progress.best_reward,
        baseline_reward: progress.baseline_reward,
        results: chunks,
        inserted_at: progress.inserted_at,
        completed_at: progress.completed_at
      })
    else
      {:error, :not_found} ->
        send_json(conn, 404, %{error: "batch_not_found"})
    end
  end

  defp serve_static(conn, filename, content_type) do
    path = Path.join(:code.priv_dir(:server) |> to_string(), Path.join("static", filename))

    case File.read(path) do
      {:ok, body} ->
        conn
        |> put_resp_content_type(content_type)
        |> put_resp_header("cache-control", "public, max-age=300")
        |> send_resp(200, body)

      {:error, _} ->
        send_json(conn, 404, %{error: "not_found"})
    end
  end

  defp send_json(conn, status, body) do
    conn
    |> put_resp_content_type("application/json")
    |> send_resp(status, Jason.encode!(body))
  end

  # Public read-only endpoints: allow cross-origin reads (showcase
  # pages, embeds, hackathon dashboards) and cache briefly so the
  # database isn't hammered by polling browsers.
  defp put_public_headers(conn) do
    conn
    |> put_resp_header("cache-control", "public, max-age=15")
    |> put_resp_header("access-control-allow-origin", "*")
    |> put_resp_header("access-control-allow-methods", "GET, OPTIONS")
  end

  defp send_cors_preflight(conn) do
    conn
    |> put_resp_header("access-control-allow-origin", "*")
    |> put_resp_header("access-control-allow-methods", "GET, OPTIONS")
    |> put_resp_header("access-control-max-age", "86400")
    |> send_resp(204, "")
  end

  defp render_snapshot(%Server.PolicySnapshot{} = s) do
    %{
      env_policy_id: s.env_policy_id,
      env_name: s.env_name,
      policy_code: s.policy_code,
      code_language: s.code_language,
      bit_predicates: s.bit_predicates,
      n_bits: s.n_bits,
      target_bit: s.target_bit,
      cegar_iter: s.cegar_iter,
      iter: s.iter,
      best_reward: s.best_reward,
      baseline_reward: s.baseline_reward,
      batch_id: s.batch_id,
      submitter: s.submitter,
      updated_at: s.updated_at
    }
  end

  defp parse_limit(value, opts) do
    default = Keyword.fetch!(opts, :default)
    max = Keyword.fetch!(opts, :max)

    case value && Integer.parse(value) do
      {n, _} when n > 0 and n <= max -> n
      _ -> default
    end
  end

  defp format_errors(changeset) do
    Ecto.Changeset.traverse_errors(changeset, fn {msg, opts} ->
      Regex.replace(~r"%{(\w+)}", msg, fn _, key ->
        opts |> Keyword.get(String.to_existing_atom(key), key) |> to_string()
      end)
    end)
  end
end
