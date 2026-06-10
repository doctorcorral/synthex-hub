defmodule Server.Workers.ExperimentBootstrap do
  @moduledoc """
  First job in an experiment's lifecycle. Runs exactly once.

  Responsibilities:

    1. Read the experiment row.
    2. Look up (or create) the `Server.EnvPolicy` for this
       experiment's `(env_name, config_sig)`. The env_policy is the
       source of truth for the synthesized predicates and the
       lineage's `policy_version` — see `Server.EnvPolicy`'s
       moduledoc for the lineage / session split.
    3. Inherit the env_policy's predicates as this session's
       starting state:
         * `:existing` (lineage already has a winner) — start from
           the lineage's committed predicates at its current
           `policy_version`. The session's `baseline_reward` is the
           lineage's `best_reward` (what we're starting from);
           commits only land if they improve on it.
         * `:created` (lineage is fresh) — start from `falsep`,
           evaluate the empty-policy baseline, write it into the
           env_policy via `EnvPolicies.ensure_baseline/3`.
    4. Persist on the experiment row: `env_policy_id`, the inherited
       (or freshly-measured) baseline, `current_cegar_iter=1`,
       `current_iter=1`, status → `running`.
    5. Enqueue the first `ExperimentController` job.

  ## Retry safety

  If steps 1-4 succeed but step 5 fails, Oban retries with the row
  already initialized — the next attempt sees `status == "running"`
  and just re-enqueues the controller. `upsert_for_submission/2` is
  idempotent: a fresh env_policy is only inserted if one doesn't
  already exist for the `(env_name, sig)` pair.
  """

  use Oban.Worker, queue: :master, max_attempts: 3
  require Logger

  alias Server.{EnvPolicies, Experiment, Experiments}
  alias Synthex.Core.PrettyPrint

  @impl Oban.Worker
  def perform(%Oban.Job{args: %{"experiment_id" => id}}) do
    case Experiments.get(id) do
      {:error, :not_found} ->
        Logger.warning("[Bootstrap] experiment #{id} not found; discarding")
        {:discard, "experiment not found"}

      {:ok, %Experiment{status: status}} when status in ["completed", "failed", "cancelled"] ->
        Logger.info("[Bootstrap] experiment #{id} already #{status}; nothing to do")
        :ok

      {:ok, %Experiment{status: "running"} = exp} ->
        # Re-entry after a partial success in a previous attempt:
        # the row was initialized, but enqueueing the controller
        # failed. Just enqueue and exit.
        Logger.info("[Bootstrap] experiment #{id} already running; (re-)enqueueing controller")
        enqueue_first_step(exp.id)

      {:ok, %Experiment{status: "pending"} = exp} ->
        try do
          bootstrap(exp)
        catch
          {:duplicate_session, _id} ->
            # Sibling session won the active-lineage race; bootstrap
            # transitioned us to `cancelled` already. Return `:ok` so
            # Oban marks this job complete rather than retrying.
            :ok
        end
    end
  end

  defp bootstrap(%Experiment{} = exp) do
    Logger.info("[Bootstrap] starting #{exp.env_name} (#{exp.id})")

    env_key = decode_env_key(exp.env_key)
    ctx = build_context(env_key, exp.config, exp.id)
    n_episodes = ctx.n_episodes

    # Locate or create the env_policy lineage for this (env, sig).
    {:ok, env_policy, status} =
      EnvPolicies.upsert_for_submission(%{
        env_name: exp.env_name,
        env_key: exp.env_key,
        config: exp.config || %{}
      })

    {predicates, session_baseline, env_policy} =
      if status == :existing and has_committed_predicates?(env_policy) do
        # Lineage has at least one accepted commit. Start from its
        # current policy; baseline (for the session's
        # delta-vs-baseline display) is what the lineage achieved
        # before we arrived.
        preds = decode_predicates(env_policy.predicates)

        baseline =
          env_policy.best_reward ||
            measure_baseline(preds, ctx)

        {:ok, env_policy} = EnvPolicies.ensure_baseline(env_policy, baseline, n_episodes)

        Logger.info(
          "[Bootstrap] #{exp.env_name} inheriting lineage v=#{env_policy.policy_version} " <>
            "(#{length(preds)} bits, baseline=#{Float.round((baseline || 0.0) / n_episodes, 2)}/ep)"
        )

        {preds, baseline, env_policy}
      else
        # Fresh lineage — first ever submission for this (env, sig)
        # (or an existing lineage that has no predicates yet because
        # we backfilled it empty). Evaluate the empty-policy baseline
        # and seed env_policies with it.
        initial_preds = Synthex.Gym.Mujoco.initial_predicates(ctx)
        baseline = measure_baseline(initial_preds, ctx)

        encoded = %{"preds" => Enum.map(initial_preds, &PrettyPrint.to_json_term/1)}

        {:ok, env_policy} =
          env_policy
          |> Ecto.Changeset.change(predicates: encoded)
          |> Server.Repo.update()

        {:ok, env_policy} = EnvPolicies.ensure_baseline(env_policy, baseline, n_episodes)

        Logger.info(
          "[Bootstrap] #{exp.env_name} fresh lineage; " <>
            "baseline=#{Float.round(baseline / n_episodes, 2)}/ep"
        )

        {initial_preds, baseline, env_policy}
      end

    case Experiments.update_state(exp, %{
           "env_policy_id" => env_policy.id,
           "status" => "running",
           "started_at" => DateTime.utc_now(),
           "current_cegar_iter" => 1,
           "current_iter" => 1,
           "baseline_reward" => session_baseline,
           "best_reward" => session_baseline
         }) do
      {:ok, _updated} ->
        :ok

      {:error, %Ecto.Changeset{errors: errors}} = err ->
        # Most likely cause: the partial unique index
        # `experiments_one_active_per_env_policy` fired because
        # another session for this (env, sig) raced us through
        # bootstrap and got there first. Cancel this duplicate
        # session cleanly rather than retrying forever.
        if Keyword.has_key?(errors, :env_policy_id) do
          Logger.info(
            "[Bootstrap] #{exp.env_name} duplicate submission for active lineage; cancelling #{exp.id}"
          )

          {:ok, _} =
            Experiments.mark_cancelled(
              exp,
              "duplicate submission — another session for this (env, config) lineage is already active"
            )

          Experiments.log_event!(
            "warn",
            "master",
            "duplicate submission cancelled: #{exp.env_name} (#{exp.id}) — " <>
              "another session for the same lineage was already active",
            env_name: exp.env_name,
            experiment_id: exp.id,
            metadata: %{"env_policy_id" => env_policy.id}
          )

          throw({:duplicate_session, exp.id})
        else
          # Some other validation failure — surface it via Oban retry.
          raise "ExperimentBootstrap update_state failed: #{inspect(err)}"
        end
    end

    Experiments.log_event!(
      "info",
      "master",
      "bootstrap complete: #{exp.env_name} v=#{env_policy.policy_version} " <>
        "baseline=#{Float.round((session_baseline || 0.0) / n_episodes, 2)}/ep " <>
        "(#{length(predicates)} bits inherited)",
      env_name: exp.env_name,
      experiment_id: exp.id,
      metadata: %{
        "baseline_reward" => session_baseline,
        "env_policy_id" => env_policy.id,
        "inherited_policy_version" => env_policy.policy_version,
        "inherited_accepted_bits" => length(predicates)
      }
    )

    enqueue_first_step(exp.id)
  end

  defp measure_baseline(preds, ctx) do
    val_seeds = Synthex.Gym.Mujoco.validation_seeds()
    {total_reward, _survived} = Synthex.Gym.Mujoco.validate(preds, val_seeds, ctx)
    total_reward / length(val_seeds)
  end

  defp has_committed_predicates?(%{predicates: %{"preds" => list}}) when is_list(list) and list != [],
    do: true

  defp has_committed_predicates?(_), do: false

  defp decode_predicates(%{"preds" => list}) when is_list(list),
    do: Enum.map(list, &PrettyPrint.from_json_term/1)

  defp decode_predicates(_), do: []

  # Streaming CEGAR controller (`docs/streaming-cegar.md` §Layer 3 /
  # "Step 2" of the deployment plan).
  defp enqueue_first_step(experiment_id) do
    case Server.Workers.ExperimentController.new(%{"experiment_id" => experiment_id})
         |> Oban.insert() do
      {:ok, _job} -> :ok
      {:error, reason} -> {:error, reason}
    end
  end

  # ── Helpers shared with the controller ──────────────────────

  @doc """
  Build a Synthex.Gym.Mujoco context from an experiment config,
  pointing the scorer at the local hub. The scorer goes through
  `Synthex.Hub.Scorer` so all of `collect_states`/`score_bit`
  still hit the same Postgres-backed batch queue — we're just
  the master, not a special-snowflake in-process driver.

  `experiment_id` is included in the batch-name prefix so workers
  and operators can correlate Batch rows back to their experiment.
  """
  def build_context(env_key, config, experiment_id) when is_atom(env_key) do
    opts = config_to_opts(config)
    scorer = build_local_scorer(env_key, experiment_id, config)
    Synthex.Gym.Mujoco.init_context(env_key, Keyword.put(opts, :scorer, scorer))
  end

  # Scorer wiring. The controller and bootstrap workers run in the
  # same BEAM as Server.Queue, so we use `Server.LocalScorer` which
  # calls `submit_batch/2` directly — no JSON encode + HTTP + decode
  # + parse loop between the master and the hub. This eliminates
  # 50–70% of the master's per-batch transient heap and is the
  # difference between fitting in our 4 GB Fly machines and OOMing
  # on Ant's tridiag pool (≈ 300 K candidates per `score_bit`).
  defp build_local_scorer(env_key, experiment_id, config) do
    adapter = get_adapter(config)
    {default_chunk, default_collect} = default_chunk_sizes(adapter)
    chunk_size = get_int(config, "chunk_size", default_chunk)
    collect_chunk_size = get_int(config, "collect_states_chunk_size", default_collect)
    state_stride = get_int(config, "state_stride", 10)
    poll_interval_ms = get_int(config, "poll_interval_ms", 500)

    Server.LocalScorer.new(
      env_key: env_key,
      adapter: adapter,
      chunk_size: chunk_size,
      collect_states_chunk_size: collect_chunk_size,
      state_stride: state_stride,
      poll_interval_ms: poll_interval_ms,
      batch_name_prefix: "exp-#{experiment_id_short(experiment_id)}",
      experiment_id: experiment_id
    )
  end

  # Physics adapter for this experiment's chunks. Defaults to
  # "mujoco" so existing experiments route to the CPU swarm; a
  # Warp experiment sets `"adapter": "mujoco_warp"` in its config.
  # Not part of config_sig — the Warp env is a distinct env_name
  # (a "different legit environment"), so its lineage is already
  # separated by name; the adapter tag is purely for routing.
  defp get_adapter(config) do
    case Map.get(config, "adapter") || Map.get(config, :adapter) do
      a when is_binary(a) and a != "" -> a
      _ -> "mujoco"
    end
  end

  # Adapter-aware {score_bit, collect_states} chunk-size defaults.
  # The two adapters want OPPOSITE granularity, and the wrong default
  # silently kills throughput:
  #   * mujoco (CPU swarm): one Python rollout loop per candidate, so
  #     SMALL chunks keep per-chunk wall-time sane and load-balance
  #     across many workers.
  #   * mujoco_warp (GPU): a whole chunk is ONE batched launch of
  #     `chunk_size * n_episodes` worlds, so it must be LARGE or the
  #     GPU starves (a size-10 chunk = 300 worlds ≈ CPU-tied, 0% util).
  # Explicit `chunk_size` / `collect_states_chunk_size` in the config
  # always override these.
  defp default_chunk_sizes("mujoco_warp"), do: {128, 128}
  defp default_chunk_sizes(_), do: {10, 4}

  defp experiment_id_short(id) when is_binary(id), do: String.slice(id, 0, 8)
  defp experiment_id_short(_), do: "unknown"

  defp config_to_opts(%{} = config) do
    [
      bits_per_dim: get_int(config, "bits_per_dim", 3),
      depth: get_int(config, "depth", 1),
      max_coeff: get_int(config, "max_coeff", 5),
      tridiag_max_coeff: get_int(config, "tridiag_max_coeff", 2),
      tridiag_dims: tridiag_range(Map.get(config, "tridiag_dims")),
      n_episodes: get_int(config, "n_episodes", 30),
      top_k: get_int(config, "top_k", 20),
      max_iters: get_int(config, "max_iters", 5),
      cegar_rounds: get_int(config, "cegar_rounds", 3),
      max_steps: get_int(config, "max_steps", 1000),
      feature_types: feature_types(Map.get(config, "feature_types"))
    ]
  end

  defp tridiag_range(nil), do: nil
  defp tridiag_range([lo, hi]) when is_integer(lo) and is_integer(hi), do: lo..hi
  defp tridiag_range(%{"lo" => lo, "hi" => hi}), do: lo..hi
  defp tridiag_range(_), do: nil

  defp feature_types(nil), do: nil

  # Feature class names are a closed set; whitelist them so we don't
  # depend on Synthex.Gym.Oracle's atoms being in the BEAM atom
  # table when an Oban job first runs.
  @feature_atoms %{
    "axis" => :axis,
    "diag" => :diag,
    "sq_diag" => :sq_diag,
    "prod" => :prod,
    "tridiag" => :tridiag,
    "sin_axis" => :sin_axis,
    "cos_axis" => :cos_axis
  }

  defp feature_types(list) when is_list(list) do
    Enum.map(list, fn
      s when is_binary(s) ->
        Map.get(@feature_atoms, s) ||
          raise ArgumentError, "unknown feature type: #{inspect(s)}"

      a when is_atom(a) ->
        a
    end)
  end

  defp get_int(map, key, default) do
    case Map.get(map, key, default) do
      n when is_integer(n) -> n
      n when is_float(n) -> trunc(n)
      _ -> default
    end
  end

  @doc """
  Atom-safe env_key decoder. We round-trip every atom in
  `Synthex.Gym.Mujoco.known_envs/0` through `Atom.to_string/1`
  and pick the match — avoiding `String.to_existing_atom/1`,
  which races with module loading on the very first Oban job
  after a release boot.
  """
  def decode_env_key(env_key) when is_binary(env_key) do
    Code.ensure_loaded(Synthex.Gym.Mujoco)

    Enum.find(Synthex.Gym.Mujoco.known_envs(), fn atom ->
      Atom.to_string(atom) == env_key
    end) || raise ArgumentError, "unknown env_key: #{inspect(env_key)}"
  end
end
