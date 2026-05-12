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

  # 30s was fine for slim polls but catastrophic for batch submits:
  # uploading tens of MB of candidates over consumer broadband
  # routinely takes longer, and a transport timeout silently kills
  # the master mid-CEGAR-step. 5 min is comfortable headroom; the
  # actual submit ceiling is bounded by the server's 64 MB body
  # cap and the per-submit candidate cap below.
  @default_request_timeout_ms 300_000

  # Hard upper bound on candidates per `submit_batch` HTTP POST.
  # Beyond this we automatically split into multiple sub-batches
  # client-side and aggregate the results — see `score_bit/3`.
  # 50_000 keeps the JSON body comfortably under the server's
  # 64 MB Plug.Parsers cap (each candidate ~50 B + payload
  # wrapper) with plenty of margin for nested predicate ADTs.
  @default_max_candidates_per_submit 50_000

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

    max_per_submit =
      Keyword.get(opts, :max_candidates_per_submit, @default_max_candidates_per_submit)

    target_bit = payload["target_bit"]
    bit_preds = payload["bit_predicates"]
    baseline_pred = Enum.at(bit_preds, target_bit)

    # Prepend baseline so it always lands as the FIRST result item
    # across the entire (possibly multi-sub-batch) submission.
    augmented_candidates = [baseline_pred | payload["candidates"]]
    n_total = length(augmented_candidates)

    # Auto-chunk: high-dim envs (Humanoid: ~5.8M features) blow past
    # any sane HTTP body cap when shipped as a single POST. We split
    # into sub-batches of <= max_per_submit candidates, submit each
    # in sequence, then concatenate per-chunk results back into one
    # logical "score_bit" response.
    groups =
      augmented_candidates
      |> Enum.chunk_every(max_per_submit)
      |> Enum.with_index()

    n_sub = length(groups)

    if n_sub > 1 do
      Logger.info(
        "[Hub] score_bit: #{n_total} candidates > #{max_per_submit}/submit; " <>
          "splitting into #{n_sub} sub-batches"
      )
    end

    with {:ok, sub_batches} <- submit_score_bit_subbatches(client, payload, name, groups, n_sub),
         {:ok, items_by_idx} <- await_score_bit_subbatches(client, sub_batches) do
      flat =
        items_by_idx
        |> Enum.sort_by(fn {sub_idx, _items} -> sub_idx end)
        |> Enum.flat_map(fn {_sub_idx, items} -> items end)

      case flat do
        [baseline | rest] ->
          scores =
            rest
            |> Enum.with_index()
            |> Enum.map(fn {item, global_idx} -> Map.put(item, "idx", global_idx) end)

          [{first_batch_id, _, _} | _] = sub_batches

          {:ok,
           %{
             scores: scores,
             baseline_reward: Map.get(baseline, "reward", 0.0),
             batch_id: first_batch_id
           }}

        [] ->
          [{first_batch_id, _, _} | _] = sub_batches
          {:error, "batch #{first_batch_id} completed with no results"}
      end
    end
  end

  defp submit_score_bit_subbatches(client, payload, name, groups, n_sub) do
    Enum.reduce_while(groups, {:ok, []}, fn {candidates_chunk, sub_idx}, {:ok, acc} ->
      sub_name = if n_sub == 1, do: name, else: "#{name}-part#{sub_idx}"

      body =
        payload
        |> Map.put("name", sub_name)
        |> Map.put("chunk_size", client.chunk_size)
        |> Map.put("candidates", candidates_chunk)

      case submit_batch(client, body) do
        {:ok, batch_id, total_chunks} ->
          if n_sub > 1 do
            Logger.info(
              "[Hub]  sub-batch #{sub_idx + 1}/#{n_sub} submitted: #{batch_id} " <>
                "(#{length(candidates_chunk)} candidates, #{total_chunks} chunks)"
            )
          else
            Logger.info(
              "[Hub] batch #{batch_id} submitted: #{total_chunks} chunks, " <>
                "#{length(candidates_chunk)} candidates (incl. baseline)"
            )
          end

          {:cont, {:ok, [{batch_id, total_chunks, sub_idx} | acc]}}

        {:error, _} = err ->
          {:halt, err}
      end
    end)
    |> case do
      {:ok, list} -> {:ok, Enum.reverse(list)}
      err -> err
    end
  end

  defp await_score_bit_subbatches(client, sub_batches) do
    Enum.reduce_while(sub_batches, {:ok, []}, fn {batch_id, _total, sub_idx}, {:ok, acc} ->
      case await_batch(client, batch_id) do
        {:ok, results} ->
          items =
            results
            |> Enum.sort_by(fn chunk -> chunk["chunk_index"] end)
            |> Enum.flat_map(fn chunk -> chunk["items"] end)

          {:cont, {:ok, [{sub_idx, items} | acc]}}

        {:error, _} = err ->
          {:halt, err}
      end
    end)
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
           # `receive_timeout` is the overall request budget in Req:
           # connection + send + first response byte. Big POSTs to
           # `/master/batches` can take a minute or more to upload,
           # so we give the whole thing the full request_timeout_ms.
           receive_timeout: client.request_timeout_ms,
           pool_timeout: 30_000,
           connect_options: [timeout: 30_000],
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

  @doc """
  Push the master's latest policy state to the hub. Master-auth
  endpoint; upserts on `env_name`.

  `attrs` MUST include `env_name`. Recommended other fields:
  `bit_predicates` (JSON-safe form via
  `Synthex.Core.PrettyPrint.to_json_term/1`), `policy_code`,
  `n_bits`, `target_bit`, `cegar_iter`, `iter`, `best_reward`,
  `baseline_reward`, `batch_id`.

  Returns `{:ok, snapshot}` on success, `{:error, reason}` on
  HTTP / transport failure. Intentionally a "fire and check"
  call — masters should NOT crash on a snapshot push failure.
  """
  def push_policy_snapshot(%__MODULE__{} = client, attrs) when is_map(attrs) do
    case Req.post(url(client, "/master/policy-snapshots"),
           headers: auth_headers(client),
           json: attrs,
           receive_timeout: client.request_timeout_ms,
           retry: false
         ) do
      {:ok, %{status: 200, body: body}} -> {:ok, body}
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
