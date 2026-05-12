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

  # Showcase: hand-curated static page with policy videos.
  # `/showcase` and `/showcase/` resolve to the index file; everything
  # under `/showcase/*` is served verbatim by Plug.Static above.
  get "/showcase" do
    serve_static(conn, "showcase/index.html", "text/html; charset=utf-8")
  end

  get "/showcase/" do
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
  # experiments. Each env collapses its own batch history into a
  # single "achieved score" so the landing page can show what the
  # swarm has actually learned, not just job throughput.
  get "/api/public-status/experiments" do
    conn
    |> put_public_headers()
    |> send_json(200, %{experiments: Server.Queue.experiments_summary()})
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

  # Latest policy snapshot for an env. Public; CORS-enabled so
  # the showcase / landing page can fetch from any origin.
  get "/api/public-status/policies/:env_name" do
    case Server.Queue.get_policy_snapshot(env_name) do
      {:ok, snapshot} ->
        conn
        |> put_public_headers()
        |> send_json(200, render_snapshot(snapshot))

      {:error, :not_found} ->
        conn
        |> put_public_headers()
        |> send_json(404, %{error: "snapshot_not_found", env_name: env_name})
    end
  end

  options "/api/public-status/policies/:_env_name" do
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
    case Server.Queue.get_batch(batch_id) do
      {:ok, batch} ->
        send_json(conn, 200, %{
          batch_id: batch.id,
          name: batch.name,
          env_name: batch.env_name,
          cmd: batch.cmd,
          status: batch.status,
          total_chunks: batch.total_chunks,
          completed_chunks: batch.completed_chunks,
          progress:
            if(batch.total_chunks > 0,
              do: batch.completed_chunks / batch.total_chunks,
              else: 1.0
            ),
          best_reward: batch.best_reward,
          baseline_reward: batch.baseline_reward,
          results: batch.results,
          inserted_at: batch.inserted_at,
          completed_at: batch.completed_at
        })

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
