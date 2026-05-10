defmodule Server.Auth do
  @moduledoc """
  Master-only Bearer-token auth.

  Routes split into three classes:

    1. **Public** (always open):
         /, /index.html, /install, /install.sh, /favicon.ico,
         /robots.txt, /health, /api/public-status

    2. **Worker** (always open — friends donate compute without
       needing to be authenticated):
         /api/worker/*

    3. **Master** (gated by `API_TOKEN`):
         /api/master/*  + /api/status (the detailed view)

  When `API_TOKEN` is unset/empty, all routes are open — useful for
  local dev. In production, set the env var via `fly secrets set
  API_TOKEN=…`.

  ## Threat model

  The token protects against:

    * batch-submission DoS (fill the queue with bogus work)
    * reading other people's batch payloads / aggregated results

  It does **not** protect against:

    * fake workers submitting poisoned results — those routes are
      intentionally open. Defend at the experiment layer (require
      multi-worker consensus, statistical outlier detection on
      reward distributions). The hub already records `worker_id` on
      every chunk completion so audits are possible after the fact.
  """
  import Plug.Conn

  # Routes that need no auth ever — landing page, installer, health
  # checks, anything under the public-status namespace (cluster
  # counters, leaderboards, per-batch contributor breakdowns).
  @public_paths ~w(
    /
    /health
    /install
    /install.sh
    /index.html
    /favicon.ico
    /robots.txt
  )

  # Prefixes that are unconditionally open. `/api/public-status` covers
  # the bare counter endpoint AND the nested leaderboard/contributor
  # endpoints under it. Worker endpoints are open so friends can donate
  # compute without being authenticated.
  @public_prefixes ["/api/public-status"]
  @worker_prefix "/api/worker"

  def init(opts), do: opts

  def call(%Plug.Conn{request_path: path} = conn, _opts) do
    cond do
      path in @public_paths ->
        conn

      Enum.any?(@public_prefixes, &String.starts_with?(path, &1)) ->
        conn

      String.starts_with?(path, @worker_prefix) ->
        conn

      # Everything else (master submission, master batch reads,
      # /api/status) requires the token if one is configured.
      true ->
        gate(conn)
    end
  end

  defp gate(conn) do
    case configured_token() do
      nil -> conn
      "" -> conn
      expected -> if valid?(conn, expected), do: conn, else: deny(conn)
    end
  end

  defp deny(conn) do
    conn
    |> put_resp_content_type("application/json")
    |> send_resp(401, Jason.encode!(%{error: "unauthorized"}))
    |> halt()
  end

  defp valid?(conn, expected) do
    presented =
      get_req_header(conn, "authorization")
      |> List.first()
      |> case do
        "Bearer " <> token -> token
        "bearer " <> token -> token
        _ -> nil
      end

    presented = presented || conn.params["token"]

    is_binary(presented) and Plug.Crypto.secure_compare(presented, expected)
  end

  defp configured_token do
    Application.get_env(:server, :api_token)
  end
end
