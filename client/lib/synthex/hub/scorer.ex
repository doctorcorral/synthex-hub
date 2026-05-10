defmodule Synthex.Hub.Scorer do
  @moduledoc """
  A `Synthex.Scoring` implementation that distributes ALL oracle calls
  to a [Synthex Hub](https://synthex.fit). The master is a thin Elixir
  coordinator: no Python, no Gymnasium, no MuJoCo — every state
  collection, candidate scoring, and validation episode runs on the
  worker swarm.

  ## Supported commands

    * `score_bit`      — distributed: one chunk == K candidates × N seeds
    * `collect_states` — distributed: one chunk == K seeds (rollouts)

  Both go through `Synthex.Hub.Client`'s submit-and-poll path on top of
  the same Oban-backed batch queue, so they reuse all the same
  retries / lifelines / contributor accounting.

  ## Usage

      scorer =
        Synthex.Hub.Scorer.new(
          env_key: :ant,
          url: "https://synthex.fit/api",
          token: System.fetch_env!("SYNTHEX_HUB_TOKEN")
        )

      Synthex.Gym.Mujoco.solve(:ant, scorer: scorer, ...)

  Defaults are read from `SYNTHEX_HUB_URL` / `SYNTHEX_HUB_TOKEN` env
  vars, so for the canonical hub at synthex.fit you can usually just
  call `Synthex.Hub.Scorer.new(env_key: :ant)`.

  ## Custom fallback

  Pass `:fallback` to override the scorer used for any command the hub
  scorer does NOT know how to dispatch. Defaults to a function that
  raises — keeping the master Python-free by construction. Tests
  override this with stubs.
  """

  alias Synthex.Hub.Client

  @doc """
  Build a scorer closure suitable for `Synthex.Gym.Mujoco.solve/2`'s
  `scorer:` opt.

  ## Options

    * `:env_key` — required. Tagged onto each batch for accounting;
      passed to the fallback if you provide one.
    * `:url`, `:token` — passed through to `Synthex.Hub.Client.new/1`.
    * `:chunk_size`, `:poll_interval_ms`, `:request_timeout_ms`,
      `:max_wait_ms` — all forwarded to `Synthex.Hub.Client`.
    * `:batch_name_prefix` — prepended to each batch's auto-generated
      name (used for grouping in the hub's UI).
    * `:collect_states_chunk_size` — independent chunk size for the
      `collect_states` command (defaults to 4 seeds per chunk, since
      one rollout episode is much heavier than scoring one candidate).
    * `:state_stride` — keep every Nth state from each rollout
      (default: 10). Only the master sees the resulting states (used
      for feature generation), and the full per-step trajectory is
      almost never needed. Lower values send more data through the
      hub; set to 1 to disable subsampling.
    * `:fallback` — a `Synthex.Scoring.t()` invoked for any command
      the hub doesn't know how to handle. Defaults to a function that
      raises.
  """
  @spec new(keyword()) :: Synthex.Scoring.t()
  def new(opts) do
    env_key = Keyword.fetch!(opts, :env_key)

    client =
      Client.new(
        url: Keyword.get(opts, :url),
        token: Keyword.get(opts, :token),
        chunk_size: Keyword.get(opts, :chunk_size, 100),
        poll_interval_ms: Keyword.get(opts, :poll_interval_ms, 5_000),
        request_timeout_ms: Keyword.get(opts, :request_timeout_ms, 30_000),
        max_wait_ms: Keyword.get(opts, :max_wait_ms, 24 * 60 * 60 * 1000)
      )

    collect_chunk_size = Keyword.get(opts, :collect_states_chunk_size, 4)

    collect_client =
      Client.new(
        url: Keyword.get(opts, :url),
        token: Keyword.get(opts, :token),
        chunk_size: collect_chunk_size,
        poll_interval_ms: Keyword.get(opts, :poll_interval_ms, 5_000),
        request_timeout_ms: Keyword.get(opts, :request_timeout_ms, 30_000),
        max_wait_ms: Keyword.get(opts, :max_wait_ms, 24 * 60 * 60 * 1000)
      )

    fallback =
      Keyword.get(opts, :fallback) ||
        fn request ->
          {:error,
           "Synthex.Hub.Scorer received an unsupported command " <>
             "(#{inspect(Map.get(request, "cmd"))}). Pass `:fallback` to " <>
             "Synthex.Hub.Scorer.new/1 if you need local execution."}
        end

    batch_prefix =
      Keyword.get(
        opts,
        :batch_name_prefix,
        "#{env_key}-#{:erlang.system_time(:second)}"
      )

    state_stride = Keyword.get(opts, :state_stride, 10)

    state = %{
      score_client: client,
      collect_client: collect_client,
      fallback: fallback,
      batch_prefix: batch_prefix,
      state_stride: state_stride
    }

    fn request -> dispatch(request, state) end
  end

  defp dispatch(%{"cmd" => "score_bit"} = request, %{
         score_client: client,
         batch_prefix: batch_prefix
       }) do
    target_bit = request["target_bit"]
    batch_name = "#{batch_prefix}-bit#{target_bit}"

    case Client.score_bit(client, request, batch_name: batch_name) do
      {:ok, %{scores: scores, baseline_reward: baseline}} ->
        {:ok,
         %{
           "scores" => scores,
           "baseline_reward" => baseline,
           "baseline_landings" => 0
         }}

      {:error, reason} ->
        {:error, "Synthex.Hub.Scorer: score_bit batch failed: #{reason}"}
    end
  end

  defp dispatch(%{"cmd" => "collect_states"} = request, %{
         collect_client: client,
         batch_prefix: batch_prefix,
         state_stride: state_stride
       }) do
    batch_name = "#{batch_prefix}-collect"

    request = Map.put_new(request, "state_stride", state_stride)

    case Client.collect_states(client, request, batch_name: batch_name) do
      {:ok, %{states: states, n_landings: n_landings, n_episodes: n_episodes}} ->
        {:ok,
         %{
           "states" => states,
           "n_landings" => n_landings,
           "n_episodes" => n_episodes
         }}

      {:error, reason} ->
        {:error, "Synthex.Hub.Scorer: collect_states batch failed: #{reason}"}
    end
  end

  defp dispatch(request, %{fallback: fallback}) do
    fallback.(request)
  end
end
