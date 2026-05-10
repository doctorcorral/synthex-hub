defmodule Synthex.Hub.Client do
  @moduledoc """
  HTTP client for talking to a [Synthex Hub](https://synthex.fit) from
  a master driver. Two responsibilities:

    1. **Submit a batch**: POSTs the candidates + per-candidate
       parameters to `/api/master/batches`. Returns the assigned
       `batch_id` and the chunk count.

    2. **Wait for completion**: polls `/api/master/batches/:id`
       until `status == "completed"`, then returns aggregated
       per-candidate results in submission order.

  Auth: master endpoints are gated by a Bearer token (`API_TOKEN`
  on the hub side). Workers are anonymous.
  """

  require Logger

  @default_url "https://synthex.fit/api"
  @default_chunk_size 100
  @default_poll_interval_ms 5_000
  @default_request_timeout_ms 30_000
  @default_max_wait_ms 24 * 60 * 60 * 1000

  @type t :: %__MODULE__{
          base_url: String.t(),
          token: String.t() | nil,
          chunk_size: pos_integer(),
          poll_interval_ms: pos_integer(),
          request_timeout_ms: pos_integer(),
          max_wait_ms: pos_integer()
        }

  defstruct base_url: @default_url,
            token: nil,
            chunk_size: @default_chunk_size,
            poll_interval_ms: @default_poll_interval_ms,
            request_timeout_ms: @default_request_timeout_ms,
            max_wait_ms: @default_max_wait_ms

  @doc """
  Build a client. Reads `SYNTHEX_HUB_URL` / `SYNTHEX_HUB_TOKEN` env
  vars as defaults; opts override.
  """
  def new(opts \\ []) do
    %__MODULE__{
      base_url:
        Keyword.get(opts, :url) || System.get_env("SYNTHEX_HUB_URL") || @default_url,
      token: Keyword.get(opts, :token) || System.get_env("SYNTHEX_HUB_TOKEN"),
      chunk_size: Keyword.get(opts, :chunk_size, @default_chunk_size),
      poll_interval_ms: Keyword.get(opts, :poll_interval_ms, @default_poll_interval_ms),
      request_timeout_ms: Keyword.get(opts, :request_timeout_ms, @default_request_timeout_ms),
      max_wait_ms: Keyword.get(opts, :max_wait_ms, @default_max_wait_ms)
    }
  end

  @doc """
  Submit a `score_bit` batch and block until it's completed.

  Returns `{:ok, %{scores: [...], baseline_reward: float, batch_id: string}}`,
  where `scores` is the per-candidate result list in **the same
  order** as `payload["candidates"]`, and `baseline_reward` is the
  reward obtained by leaving `bit_predicates[target_bit]` untouched.

  We compute baseline by prepending `bit_predicates[target_bit]` to
  the candidate list before submission and stripping it back off
  after results arrive. This is one extra evaluation per CEGAR
  iteration — negligible against the thousands of real candidates.
  """
  def score_bit(%__MODULE__{} = client, payload, opts \\ []) do
    name = Keyword.get(opts, :batch_name, "synthex-#{:erlang.system_time(:millisecond)}")
    target_bit = payload["target_bit"]
    bit_preds = payload["bit_predicates"]
    baseline_pred = Enum.at(bit_preds, target_bit)

    augmented_candidates = [baseline_pred | payload["candidates"]]

    body =
      payload
      |> Map.put("name", name)
      |> Map.put("chunk_size", client.chunk_size)
      |> Map.put("candidates", augmented_candidates)

    with {:ok, batch_id, total_chunks} <- submit_batch(client, body),
         _ =
           Logger.info(
             "[Hub] batch #{batch_id} submitted: #{total_chunks} chunks, " <>
               "#{length(augmented_candidates)} candidates (incl. baseline)"
           ),
         {:ok, results} <- await_batch(client, batch_id) do
      flat =
        results
        |> Enum.sort_by(fn chunk -> chunk["chunk_index"] end)
        |> Enum.flat_map(fn chunk -> chunk["items"] end)

      case flat do
        [baseline | rest] ->
          scores =
            rest
            |> Enum.with_index()
            |> Enum.map(fn {item, global_idx} -> Map.put(item, "idx", global_idx) end)

          {:ok,
           %{
             scores: scores,
             baseline_reward: Map.get(baseline, "reward", 0.0),
             batch_id: batch_id
           }}

        [] ->
          {:error, "batch #{batch_id} completed with no results"}
      end
    end
  end

  @doc """
  Submit a `collect_states` batch (one job-unit per seed) and block
  until completion.

  Returns `{:ok, %{states: [[float]], n_landings: int, n_episodes: int,
  batch_id: string}}` — the same shape that `Synthex.Gym.Mujoco`'s
  `collect_states/2` historically produced via local Python, just
  rebuilt by concatenating per-seed states from the worker swarm.

  `payload` MUST contain `env_name`, `bit_predicates`, `bits_per_dim`,
  `seeds` (a list of ints), and `max_steps`.
  """
  def collect_states(%__MODULE__{} = client, payload, opts \\ []) do
    name =
      Keyword.get(
        opts,
        :batch_name,
        "synthex-collect-#{:erlang.system_time(:millisecond)}"
      )

    seeds = payload["seeds"] || []

    if seeds == [] do
      {:error, "collect_states: payload[\"seeds\"] must be a non-empty list"}
    else
      body =
        payload
        |> Map.put("name", name)
        |> Map.put("cmd", "collect_states")
        |> Map.put("chunk_size", client.chunk_size)
        |> Map.put("candidates", seeds)
        |> Map.delete("seeds")

      with {:ok, batch_id, total_chunks} <- submit_batch(client, body),
           _ =
             Logger.info(
               "[Hub] collect_states batch #{batch_id}: " <>
                 "#{total_chunks} chunks, #{length(seeds)} seeds"
             ),
           {:ok, results} <- await_batch(client, batch_id) do
        items =
          results
          |> Enum.sort_by(fn chunk -> chunk["chunk_index"] end)
          |> Enum.flat_map(fn chunk -> chunk["items"] end)

        states =
          Enum.flat_map(items, fn item -> Map.get(item, "states", []) end)

        n_landings =
          Enum.count(items, fn item -> Map.get(item, "success", false) end)

        {:ok,
         %{
           states: states,
           n_landings: n_landings,
           n_episodes: length(items),
           batch_id: batch_id
         }}
      end
    end
  end

  @doc "Submit a batch payload. Returns `{:ok, batch_id, total_chunks}`."
  def submit_batch(%__MODULE__{} = client, payload) do
    case Req.post(url(client, "/master/batches"),
           headers: auth_headers(client),
           json: payload,
           receive_timeout: client.request_timeout_ms,
           retry: false
         ) do
      {:ok, %{status: 201, body: %{"batch_id" => id, "total_chunks" => n}}} ->
        {:ok, id, n}

      {:ok, %{status: 401}} ->
        {:error,
         "401 unauthorized — set SYNTHEX_HUB_TOKEN to the hub's master token"}

      {:ok, %{status: status, body: body}} ->
        {:error, "submit failed (HTTP #{status}): #{inspect(body)}"}

      {:error, reason} ->
        {:error, "submit failed: #{inspect(reason)}"}
    end
  end

  @doc """
  Block until the batch completes. Returns `{:ok, results}`.

  Polls a SLIM status endpoint (no `results` array) so a multi-hour
  run doesn't restream the same accumulated chunk payloads back to
  the master on every 5-second tick. When the status flips to
  `completed`, makes one final request with `?include_results=1`
  to actually fetch the chunk results.
  """
  def await_batch(%__MODULE__{} = client, batch_id) do
    deadline = System.monotonic_time(:millisecond) + client.max_wait_ms
    poll_loop(client, batch_id, deadline, _last_progress = -1)
  end

  defp poll_loop(client, batch_id, deadline, last_progress) do
    if System.monotonic_time(:millisecond) > deadline do
      {:error, "max_wait_ms exceeded for batch #{batch_id}"}
    else
      case fetch_batch(client, batch_id, include_results: false) do
        {:ok, %{"status" => "completed"}} ->
          # One heavy fetch, only once: pull the full results.
          case fetch_batch(client, batch_id, include_results: true) do
            {:ok, %{"results" => results}} -> {:ok, results || []}
            {:ok, _} -> {:ok, []}
            {:error, reason} -> {:error, "completed but failed to fetch results: #{reason}"}
          end

        {:ok, %{"status" => "failed"} = body} ->
          {:error, "batch #{batch_id} failed: #{inspect(body)}"}

        {:ok, %{} = body} ->
          progress = body["progress"] || 0.0
          completed = body["completed_chunks"] || 0
          total = body["total_chunks"] || 0

          if progress != last_progress do
            Logger.info(
              "[Hub] batch #{batch_id}: #{Float.round((progress || 0.0) * 100, 1)}%  (#{completed}/#{total} chunks)"
            )
          end

          Process.sleep(client.poll_interval_ms)
          poll_loop(client, batch_id, deadline, progress)

        {:error, reason} ->
          # Transient HTTP errors shouldn't kill a multi-hour run;
          # log and retry until the deadline.
          Logger.warning("[Hub] poll error: #{inspect(reason)} (will retry)")
          Process.sleep(client.poll_interval_ms)
          poll_loop(client, batch_id, deadline, last_progress)
      end
    end
  end

  defp fetch_batch(%__MODULE__{} = client, batch_id, opts) do
    qs = if Keyword.get(opts, :include_results, false), do: "?include_results=1", else: ""

    case Req.get(url(client, "/master/batches/#{batch_id}#{qs}"),
           headers: auth_headers(client),
           receive_timeout: client.request_timeout_ms,
           retry: false
         ) do
      {:ok, %{status: 200, body: body}} -> {:ok, body}
      {:ok, %{status: status, body: body}} -> {:error, "HTTP #{status}: #{inspect(body)}"}
      {:error, reason} -> {:error, reason}
    end
  end

  @doc """
  Get unauthenticated cluster status. Returns `{:ok, %{active_workers,
  total_cores, candidates_evaluated, experiments_completed}}`.
  Useful for "is anyone connected?" checks before submitting a batch.
  """
  def public_status(%__MODULE__{} = client) do
    case Req.get(url(client, "/public-status"),
           receive_timeout: client.request_timeout_ms,
           retry: false
         ) do
      {:ok, %{status: 200, body: body}} -> {:ok, body}
      {:ok, %{status: s, body: b}} -> {:error, "HTTP #{s}: #{inspect(b)}"}
      {:error, reason} -> {:error, reason}
    end
  end

  @doc """
  Get the public per-batch contributor list (display names, no IDs).
  Returns `{:ok, [%{name, candidates_evaluated, chunks_completed, ...}]}`.
  """
  def batch_contributors(%__MODULE__{} = client, batch_id) do
    case Req.get(url(client, "/public-status/batches/#{batch_id}/contributors"),
           receive_timeout: client.request_timeout_ms,
           retry: false
         ) do
      {:ok, %{status: 200, body: %{"contributors" => list}}} -> {:ok, list}
      {:ok, %{status: s, body: b}} -> {:error, "HTTP #{s}: #{inspect(b)}"}
      {:error, reason} -> {:error, reason}
    end
  end

  defp url(%__MODULE__{base_url: base}, path) do
    String.trim_trailing(base, "/") <> path
  end

  defp auth_headers(%__MODULE__{token: token})
       when is_binary(token) and byte_size(token) > 0 do
    [{"authorization", "Bearer #{token}"}]
  end

  defp auth_headers(_), do: []
end
