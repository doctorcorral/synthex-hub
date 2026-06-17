defmodule Server.LocalScorer do
  @moduledoc """
  In-process implementation of the `Synthex.Scoring` contract for
  the server-hosted Oban CEGAR master. Functionally equivalent to
  `Synthex.Hub.Scorer` but bypasses the master ⇄ Bandit HTTP loop
  entirely — when the master and the hub are the same BEAM (the
  always-true case post `docs/streaming-cegar.md` §Layer 3),
  shipping multi-megabyte candidate payloads through Bandit +
  Plug.Parsers is pure waste: same data, twice in memory, twice on
  the heap.

  ## Path comparison

      Hub.Scorer:
        master proc ─→ JSON encode (~8 MB for Ant score_bit) ─→ Req
          └→ Bandit ─→ Plug.Parsers JSON decode (~15 MB term)
              └→ Server.Router ─→ Server.Queue.submit_batch
                  └→ DB writes ┄→ batches + Oban.Job rows
        ※ peak transient heap (both sides): 50–200 MB per batch
          for the Ant tridiag pool. With 8 parallel-bit waves, hit
          ~2 GB and OOM-kill the Fly machine.

      LocalScorer:
        master proc ─→ Server.Queue.submit_batch (direct call)
            └→ DB writes ┄→ batches + Oban.Job rows
        ※ peak transient heap: 5–20 MB per batch. The Erlang term
          is built once, by the master, and consumed directly by
          Postgrex.

  ## Polling

  Same liveness model as `Synthex.Hub.Client.await_batch`: poll
  batch progress, fail if no chunk has completed within
  `stall_timeout_ms`, raise on hard wall-clock deadline. Lower
  poll interval (default 500 ms vs. 5 s for HTTP) because the
  in-process `Server.Queue.get_batch_progress/1` is a one-shot
  indexed SELECT with no network cost.

  ## Why not Phoenix.PubSub

  Server.Queue has no PubSub topic for batch completion yet.
  Polling the DB at 500 ms is already cheaper than the HTTP poll
  loop this replaces. A future PR can add a broadcast on batch
  completion without changing this module's public shape.

  ## Drop-in compatibility

  Returns the same string-keyed result maps as `Synthex.Hub.Scorer`
  (`"scores"` / `"baseline_reward"` / `"baseline_landings"` for
  `score_bit`; `"states"` / `"n_landings"` / `"n_episodes"` for
  `collect_states`) so `Synthex.Gym.Mujoco.call_scorer!/2` works
  unchanged.
  """

  require Logger

  alias Server.Queue

  @default_chunk_size 10
  @default_collect_chunk_size 4
  @default_state_stride 10
  @default_poll_interval_ms 500
  # Stall = no new completed chunk in this long. Master waits
  # indefinitely on slow workers but bails on a truly silent swarm.
  # 2 h matches `Synthex.Hub.Client`'s default and is plenty even
  # for the worst Humanoid chunk on a single core.
  @default_stall_timeout_ms 2 * 60 * 60 * 1000
  @default_max_wait_ms 30 * 24 * 60 * 60 * 1000

  @type t :: (map() -> {:ok, map()} | {:error, term()})

  @doc """
  Build a scorer closure. Options mirror `Synthex.Hub.Scorer.new/1`
  for drop-in compatibility, minus the HTTP-specific ones (`:url`,
  `:token`, `:request_timeout_ms`, `:max_candidates_per_submit`).
  """
  @spec new(keyword()) :: t()
  def new(opts) do
    env_key = Keyword.fetch!(opts, :env_key)

    state = %{
      env_key: env_key,
      # Physics adapter tag stamped onto every batch this scorer
      # submits; the hub routes chunks to workers advertising a
      # matching capability. Defaults to "mujoco" (the CPU swarm).
      adapter: Keyword.get(opts, :adapter) || "mujoco",
      chunk_size: Keyword.get(opts, :chunk_size, @default_chunk_size),
      collect_chunk_size:
        Keyword.get(opts, :collect_states_chunk_size, @default_collect_chunk_size),
      state_stride: Keyword.get(opts, :state_stride, @default_state_stride),
      submitter: Keyword.get(opts, :submitter),
      experiment_id: Keyword.get(opts, :experiment_id),
      batch_prefix:
        Keyword.get(
          opts,
          :batch_name_prefix,
          "#{env_key}-#{:erlang.system_time(:second)}"
        ),
      poll_interval_ms: Keyword.get(opts, :poll_interval_ms, @default_poll_interval_ms),
      stall_timeout_ms: Keyword.get(opts, :stall_timeout_ms, @default_stall_timeout_ms),
      max_wait_ms: Keyword.get(opts, :max_wait_ms, @default_max_wait_ms),
      fallback:
        Keyword.get(opts, :fallback) ||
          fn req ->
            {:error,
             "Server.LocalScorer received an unsupported cmd: " <>
               inspect(Map.get(req, "cmd"))}
          end
    }

    fn request -> dispatch(request, state) end
  end

  # ── Dispatch ────────────────────────────────────────────────

  defp dispatch(%{"cmd" => "score_bit"} = request, state) do
    target_bit = request["target_bit"]
    bit_preds = request["bit_predicates"]
    baseline_pred = Enum.at(bit_preds, target_bit)
    candidates = request["candidates"] || []

    # Prepend the baseline so it lands as the FIRST result item.
    # This mirrors `Synthex.Hub.Client.score_bit`'s contract — the
    # master gets back N+1 evaluations and we peel off the first
    # one as the "leave bit_predicates[target_bit] untouched"
    # baseline reward.
    augmented = [baseline_pred | candidates]

    batch_name = "#{state.batch_prefix}-bit#{target_bit}"

    payload =
      request
      |> Map.put("name", batch_name)
      |> Map.put("chunk_size", state.chunk_size)
      |> Map.put("candidates", augmented)
      |> Map.put("adapter", state.adapter)
      |> maybe_put_experiment_id(state.experiment_id)

    case Queue.submit_batch(payload, submitter: state.submitter) do
      {:ok, batch} ->
        Logger.info(
          "[LocalScorer] score_bit batch #{batch.id} submitted: " <>
            "#{batch.total_chunks} chunks, #{length(augmented)} candidates"
        )

        case await_batch_items(batch.id, state) do
          {:ok, items} ->
            result = unpack_score_bit_items(items, batch.id)
            debug_score_bit(result, request, state, batch)
            result

          {:error, _} = err ->
            err
        end

      {:error, reason} ->
        {:error, "submit_batch failed: #{inspect(reason)}"}
    end
  end

  defp dispatch(%{"cmd" => "collect_states"} = request, state) do
    seeds = request["seeds"] || []

    if seeds == [] do
      {:error, "collect_states: payload[\"seeds\"] must be non-empty"}
    else
      batch_name = "#{state.batch_prefix}-collect"

      payload =
        request
        |> Map.put("name", batch_name)
        |> Map.put("cmd", "collect_states")
        |> Map.put("chunk_size", state.collect_chunk_size)
        |> Map.put("candidates", seeds)
        |> Map.delete("seeds")
        |> Map.put_new("state_stride", state.state_stride)
        |> Map.put("adapter", state.adapter)
        |> maybe_put_experiment_id(state.experiment_id)

      case Queue.submit_batch(payload, submitter: state.submitter) do
        {:ok, batch} ->
          Logger.info(
            "[LocalScorer] collect_states batch #{batch.id}: " <>
              "#{batch.total_chunks} chunks, #{length(seeds)} seeds"
          )

          case await_batch_items(batch.id, state) do
            {:ok, items} ->
              states = Enum.flat_map(items, fn item -> Map.get(item, "states", []) end)
              n_landings = Enum.count(items, fn item -> Map.get(item, "success", false) end)

              {:ok,
               %{
                 "states" => states,
                 "n_landings" => n_landings,
                 "n_episodes" => length(items)
               }}

            {:error, _} = err ->
              err
          end

        {:error, reason} ->
          {:error, "submit_batch failed: #{inspect(reason)}"}
      end
    end
  end

  defp dispatch(%{"cmd" => "eval_regret"} = request, state) do
    seeds = request["seeds"] || []

    if seeds == [] do
      {:error, "eval_regret: payload[\"seeds\"] must be non-empty"}
    else
      payload =
        request
        |> Map.put("name", "#{state.batch_prefix}-regret")
        |> Map.put("cmd", "eval_regret")
        |> Map.put("chunk_size", state.collect_chunk_size)
        |> Map.put("candidates", seeds)
        |> Map.delete("seeds")
        |> Map.put("adapter", state.adapter)
        |> maybe_put_experiment_id(state.experiment_id)

      case Queue.submit_batch(payload, submitter: state.submitter) do
        {:ok, batch} ->
          Logger.info(
            "[LocalScorer] eval_regret batch #{batch.id}: " <>
              "#{batch.total_chunks} chunks, #{length(seeds)} seeds"
          )

          case await_batch_items(batch.id, state) do
            {:ok, items} ->
              regrets =
                items
                |> Enum.reject(fn item -> Map.has_key?(item, "error") end)
                |> Enum.sort_by(fn item -> Map.get(item, "idx", 0) end)

              {:ok, %{"regrets" => regrets}}

            {:error, _} = err ->
              err
          end

        {:error, reason} ->
          {:error, "submit_batch failed: #{inspect(reason)}"}
      end
    end
  end

  defp dispatch(request, %{fallback: fallback}), do: fallback.(request)

  # ── Helpers ────────────────────────────────────────────────

  defp unpack_score_bit_items([], batch_id) do
    {:error, "score_bit batch #{batch_id} completed with zero items"}
  end

  defp unpack_score_bit_items([baseline | rest], _batch_id) do
    # Re-index `rest` with a stable global index so callers (Mujoco)
    # can map scored items back to their candidate position.
    scores =
      rest
      |> Enum.with_index()
      |> Enum.map(fn {item, idx} -> Map.put(item, "idx", idx) end)

    {:ok,
     %{
       "scores" => scores,
       "baseline_reward" => Map.get(baseline, "reward", 0.0),
       "baseline_landings" => Map.get(baseline, "landings", 0),
       # Present only when the request set `want_per_seed` AND the worker
       # is new enough to honour it (validation tail metrics). `nil`
       # otherwise; callers fall back to mean-only.
       "baseline_per_seed" => Map.get(baseline, "per_seed")
     }}
  end

  # ── Temporary CEGAR-acceptance instrumentation ──────────────────
  # Records the exact `baseline_reward` + best-candidate reward the
  # master's `optimize_bit` receives for this `score_bit` call. Paired
  # with the controller's `cegar-debug` verdict events, this isolates
  # whether a null result comes from an inflated baseline, the
  # candidate scoring, or the commit gate. Never crashes the scorer.
  defp debug_score_bit({:ok, %{"scores" => scores, "baseline_reward" => base}}, request, state, batch) do
    rewards = Enum.map(scores, fn s -> Map.get(s, "reward", 0.0) end)
    best = Enum.max(rewards, fn -> nil end)

    Server.Experiments.log_event!(
      "info",
      "cegar-debug",
      "score_bit bit=#{request["target_bit"]} batch=#{batch.name} " <>
        "baseline=#{inspect(base)} best_cand=#{inspect(best)} n_cands=#{length(scores)}",
      env_name: request["env_name"],
      experiment_id: state.experiment_id,
      metadata: %{
        "phase" => "score_bit",
        "target_bit" => request["target_bit"],
        "baseline_reward" => base,
        "best_candidate_reward" => best,
        "n_candidates" => length(scores)
      }
    )
  rescue
    e -> Logger.warning("[LocalScorer] debug_score_bit failed: #{inspect(e)}")
  end

  defp debug_score_bit(_other, _request, _state, _batch), do: :ok

  defp maybe_put_experiment_id(payload, nil), do: payload

  defp maybe_put_experiment_id(payload, id) when is_binary(id),
    do: Map.put(payload, "experiment_id", id)

  # ── Await + poll ────────────────────────────────────────────

  # After this much apparent no-progress we start asking whether the
  # batch's adapter even has a live worker. Short enough to surface a
  # pause quickly, long enough to avoid querying the workers table on
  # every healthy poll.
  @pause_grace_ms 30_000

  # While paused (no capable worker online) we refresh the experiment's
  # liveness heartbeat at most this often, so a returning worker and the
  # dashboard both see the run as alive-and-waiting, not dead.
  @pause_heartbeat_ms 30_000

  # Poll cadence while paused. Slower than the active poll interval so a
  # multi-hour GPU outage doesn't probe the workers table every 500 ms;
  # 5 s is still responsive enough to resume promptly when one rejoins.
  @paused_poll_interval_ms 5_000

  defp await_batch_items(batch_id, state) do
    now = System.monotonic_time(:millisecond)

    s = %{
      hard_deadline: now + state.max_wait_ms,
      # Initialize the stall clock at "now"; we accept the first
      # chunk completion as the baseline progress event.
      last_progress_at: now,
      last_completed: -1,
      poll_interval_ms: state.poll_interval_ms,
      stall_timeout_ms: state.stall_timeout_ms,
      last_logged_progress: -1.0,
      # Pause bookkeeping. `adapter`/`experiment_id` drive the
      # "is there a worker that can run this?" check; `paused_since`
      # gates the resume log; `last_pause_heartbeat` throttles the
      # liveness touch.
      adapter: state.adapter,
      experiment_id: state.experiment_id,
      paused_since: nil,
      last_pause_heartbeat: 0
    }

    poll_loop(batch_id, s)
  end

  defp poll_loop(batch_id, s) do
    now = System.monotonic_time(:millisecond)
    no_progress_ms = now - s.last_progress_at

    cond do
      now > s.hard_deadline ->
        {:error, "batch #{batch_id} exceeded hard wall-clock deadline"}

      # PAUSED, not stalled: progress has been idle past the grace
      # window AND no live worker advertises this batch's adapter (e.g.
      # a mujoco_warp run while the GPU box is offline). Don't burn the
      # stall clock — freeze it, keep a liveness heartbeat, and wait.
      # The chunks stay `available`; when a capable worker rejoins it
      # claims them and progress resumes on its own.
      no_progress_ms > @pause_grace_ms and not Queue.adapter_has_live_worker?(s.adapter) ->
        s2 = note_paused(s, batch_id, now)
        Process.sleep(max(s.poll_interval_ms, @paused_poll_interval_ms))
        # Park the stall clock just past the grace boundary: the next
        # iteration still re-checks worker liveness (stays "paused"),
        # but the elapsed-no-progress can never crawl toward
        # stall_timeout while we wait. When a capable worker rejoins,
        # this branch stops matching and the parked clock leaves a full
        # stall window for real chunk completion.
        poll_loop(batch_id, %{s2 | last_progress_at: now - @pause_grace_ms - 1})

      no_progress_ms > s.stall_timeout_ms ->
        {:error,
         "batch #{batch_id} stalled (no chunk progress in " <>
           "#{div(s.stall_timeout_ms, 1000)}s)"}

      true ->
        case Queue.get_batch_progress(batch_id) do
          {:error, :not_found} ->
            {:error, "batch #{batch_id} not found"}

          {:ok, progress} ->
            case progress.status do
              "completed" ->
                fetch_completed_items(batch_id)

              "failed" ->
                {:error, "batch #{batch_id} failed"}

              "cancelled" ->
                {:error, "batch #{batch_id} cancelled"}

              _ ->
                s2 = s |> maybe_log_resume(batch_id) |> maybe_advance_progress_clock(progress, now, batch_id)
                Process.sleep(s.poll_interval_ms)
                poll_loop(batch_id, s2)
            end
        end
    end
  end

  # Heartbeat + log a paused experiment. Throttled to @pause_heartbeat_ms
  # so a long GPU outage doesn't spam the event log or the DB.
  defp note_paused(s, batch_id, now) do
    if now - s.last_pause_heartbeat >= @pause_heartbeat_ms do
      if is_nil(s.paused_since) do
        Logger.warning(
          "[LocalScorer] batch #{batch_id} PAUSED: no live '#{s.adapter}' worker; " <>
            "holding (chunks remain queued, will resume when one rejoins)"
        )

        log_pause_event(s, batch_id)
      end

      Server.Experiments.touch(s.experiment_id)

      %{s | paused_since: s.paused_since || now, last_pause_heartbeat: now}
    else
      %{s | paused_since: s.paused_since || now}
    end
  end

  # Emit a one-shot incident when a run first pauses so it shows on the
  # dashboard's event feed. Best-effort — never let logging break the loop.
  defp log_pause_event(%{experiment_id: id} = s, _batch_id) when is_binary(id) do
    Server.Experiments.log_event!(
      "warn",
      "master",
      "experiment paused: waiting for a '#{s.adapter}' worker to come online",
      experiment_id: id,
      metadata: %{"adapter" => s.adapter}
    )
  rescue
    _ -> :ok
  end

  defp log_pause_event(_s, _batch_id), do: :ok

  defp maybe_log_resume(%{paused_since: nil} = s, _batch_id), do: s

  defp maybe_log_resume(s, batch_id) do
    Logger.info("[LocalScorer] batch #{batch_id} RESUMED: a '#{s.adapter}' worker is back online")
    %{s | paused_since: nil, last_pause_heartbeat: 0}
  end

  defp maybe_advance_progress_clock(s, progress, now, batch_id) do
    completed = progress.completed_chunks || 0
    total = progress.total_chunks || 1

    s =
      if completed > s.last_completed do
        progress_pct = completed / max(total, 1)
        # Throttle the per-poll log; only emit on >=5% jumps so
        # we don't spam at high chunk counts.
        s =
          if progress_pct >= s.last_logged_progress + 0.05 do
            Logger.info(
              "[LocalScorer] batch #{batch_id}: " <>
                "#{completed}/#{total} chunks " <>
                "(#{Float.round(progress_pct * 100, 1)}%)"
            )

            %{s | last_logged_progress: progress_pct}
          else
            s
          end

        %{s | last_completed: completed, last_progress_at: now}
      else
        s
      end

    s
  end

  defp fetch_completed_items(batch_id) do
    # Items live on oban_jobs.args["results"] (one row per chunk),
    # not on Batch.results (which is no longer populated — see
    # `Server.Queue.fetch_batch_chunks/1` for the why). This is a
    # single indexed read at end-of-batch instead of the old
    # O(N²) push-and-readback dance.
    case Queue.fetch_batch_chunks(batch_id) do
      {:ok, chunks} ->
        {:ok, Enum.flat_map(chunks, fn c -> c["items"] || [] end)}
    end
  end
end
