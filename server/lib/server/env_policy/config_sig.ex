defmodule Server.EnvPolicy.ConfigSig do
  @moduledoc """
  Canonicalize the policy-shape-determining subset of an experiment
  config and hash it into a stable signature.

  The signature is the `(env_name, config_sig)` half of the
  `Server.EnvPolicy` identity. Two submissions with the same env_name
  AND the same `config_sig` share a policy lineage and inherit each
  other's commits. Two submissions that differ in any policy-shape
  field fork into independent lineages — a v=19 policy under
  `bits_per_dim=3` is not even structurally meaningful as a starting
  point for `bits_per_dim=4`, so we must not merge them.

  ## Policy-shape fields

  These knobs change what a predicate IS (its tree shape, its leaf
  vocabulary, its acceptable coefficient range) or how many of them
  there are (n_bits = bits_per_dim × n_action_dims). Changing any
  of them invalidates cross-experiment policy inheritance:

    * `bits_per_dim`       — n_bits per action dimension
    * `depth`              — predicate compositional depth
    * `feature_types`      — which feature classes are emittable
    * `max_coeff`          — coefficient range for non-tridiag features
    * `tridiag_max_coeff`  — coefficient range for tridiag features
    * `tridiag_dims`       — which action-dim pairs participate in
                              tridiag features (nil = none/default)

  ## Excluded (search procedure / evaluation regime, not shape)

    * `top_k`, `max_iters`, `cegar_rounds` — search budget
    * `n_episodes`, `max_steps`            — evaluation regime; reward
                                              magnitudes change but the
                                              policy data type doesn't,
                                              so the lineage is shared
                                              and the commit gate
                                              naturally rejects regressions
    * `chunk_size`, `collect_states_chunk_size`,
      `state_stride`, `poll_interval_ms`   — pure infra knobs

  If a future need emerges to fork lineages on e.g. `max_steps`
  (latency-bounded eval), add it to `@policy_shape_keys` — old
  signatures don't change because we always read from the field's
  default when absent, but new submissions under the new key fork
  cleanly.
  """

  # Fields that participate in the signature. Order doesn't matter
  # for the hash (we canonicalize the map before hashing) but matters
  # for `canonicalize/1`'s deterministic output.
  @policy_shape_keys [
    "bits_per_dim",
    "depth",
    "feature_types",
    "max_coeff",
    "tridiag_max_coeff",
    "tridiag_dims"
  ]

  # Mirrors `Server.Workers.ExperimentBootstrap.config_to_opts/1`
  # defaults so submissions that omit a key get the SAME sig as a
  # submission that spells the default out explicitly. Drift between
  # these defaults and the bootstrap defaults silently forks lineages —
  # keep them in lockstep.
  @defaults %{
    "bits_per_dim" => 3,
    "depth" => 1,
    "feature_types" => nil,
    "max_coeff" => 5,
    "tridiag_max_coeff" => 2,
    "tridiag_dims" => nil,
    # Counterexample-source strategy. NOT a policy-*shape* field — it
    # doesn't change what a predicate is — but it changes the training
    # distribution, so an adversarially-trained policy and a
    # random-trained one are different experiments that must NOT inherit
    # each other's commits. Forked in `stable_encode/1` in a
    # backward-compatible way: the default (`random`) is omitted from the
    # digest, so every pre-existing signature is unchanged.
    "verifier" => "random",
    # Replicate index. NOT a policy-shape field — it only shifts the
    # training-seed draw — but two replicates must be INDEPENDENT
    # lineages (they must not inherit each other's commits) or a sweep
    # collapses to one shared policy. Forked backward-compatibly in
    # `stable_encode/1`: the default (0) is omitted from the digest so
    # every pre-existing signature is byte-identical.
    "run_seed" => 0,
    # Per-bit candidate proposer. NOT a policy-shape field — it doesn't
    # change what a predicate is — but it changes HOW candidates are
    # found, so a GA-searched policy and an enumerate-searched one are
    # different experiments that must NOT inherit each other's commits.
    # Forked backward-compatibly in `stable_encode/1`: the default
    # (`enumerate`) is omitted from the digest so every pre-existing
    # signature is byte-identical.
    "proposer" => "enumerate",
    # Per-bit candidate fitness / ranking objective. NOT a policy-shape
    # field — the predicate form is unchanged — but successor-ranked and
    # episode-ranked policies are different experiments. Forked backward-
    # compatibly in `stable_encode/1`: the default (`episode`) is omitted.
    "scorer" => "episode",
    # Structured continuation for successor rollout lookaheads (e.g. a
    # fixed reference gait cycle standing in for V*). NOT a policy-shape
    # field, but it changes the ranking objective fundamentally — a
    # cycle-continuation policy must not inherit incumbent-continuation
    # commits or vice versa. Forked backward-compatibly in
    # `stable_encode/1`: the default (nil, incumbent continuation) is
    # omitted from the digest.
    "successor_continuation" => nil,
    # Per-experiment gym.make kwargs merged over the env_key's registry
    # entry (the reward-annealing knob, e.g. healthy_reward=0.1). Two
    # runs under different reward definitions are different experiments
    # that must NOT inherit each other's commits. Forked backward-
    # compatibly in `stable_encode/1`: the default (nil) is omitted.
    "env_kwargs" => nil,
    # Commit acceptance order: "episode" (greedy return monotonicity)
    # vs "successor" (coinductive dominance). Different acceptance
    # orders accept different commit sequences — separate lineages.
    # Forked backward-compatibly: the default ("episode") is omitted.
    "commit_gate" => "episode",
    # Observation augmentation (markovization): policies over different
    # observation spaces are incomparable — separate lineages. Forked
    # backward-compatibly: the default (nil) is omitted.
    "obs_augmentation" => nil
  }

  @feature_canonical %{
    "axis" => "axis",
    "diag" => "diag",
    "sq_diag" => "sq_diag",
    "prod" => "prod",
    "tridiag" => "tridiag",
    "sin_axis" => "sin_axis",
    "cos_axis" => "cos_axis",
    "wavelet_box" => "wavelet_box",
    "wavelet_ricker" => "wavelet_ricker"
  }

  @doc """
  Canonical form of the policy-shape sub-config. Always a map keyed
  by string with normalized values:

    * integers stay integers (floats truncated to int)
    * `feature_types` sorted, atoms stringified, unknowns rejected
    * `tridiag_dims` normalized to a sorted two-element list or nil
    * missing keys filled from `@defaults`

  Two configs canonicalize-equal iff they belong to the same policy
  lineage. The serialized canonical form is what `hash/1` digests.
  """
  @spec canonicalize(map()) :: map()
  def canonicalize(config) when is_map(config) do
    base =
      @policy_shape_keys
      |> Enum.map(fn key ->
        raw = Map.get(config, key, Map.get(config, String.to_atom(key), Map.get(@defaults, key)))
        {key, normalize_value(key, raw)}
      end)
      |> Map.new()

    verifier_raw = Map.get(config, "verifier", Map.get(config, :verifier, @defaults["verifier"]))

    run_seed_raw =
      Map.get(config, "run_seed", Map.get(config, :run_seed, @defaults["run_seed"]))

    proposer_raw =
      Map.get(config, "proposer", Map.get(config, :proposer, @defaults["proposer"]))

    scorer_raw =
      Map.get(config, "scorer", Map.get(config, :scorer, @defaults["scorer"]))

    continuation_raw =
      Map.get(
        config,
        "successor_continuation",
        Map.get(config, :successor_continuation, @defaults["successor_continuation"])
      )

    env_kwargs_raw =
      Map.get(config, "env_kwargs", Map.get(config, :env_kwargs, @defaults["env_kwargs"]))

    commit_gate_raw =
      Map.get(config, "commit_gate", Map.get(config, :commit_gate, @defaults["commit_gate"]))

    obs_aug_raw =
      Map.get(
        config,
        "obs_augmentation",
        Map.get(config, :obs_augmentation, @defaults["obs_augmentation"])
      )

    base
    |> Map.put("verifier", canonical_verifier(verifier_raw))
    |> Map.put("run_seed", canonical_run_seed(run_seed_raw))
    |> Map.put("proposer", canonical_proposer(proposer_raw))
    |> Map.put("scorer", canonical_scorer(scorer_raw))
    |> Map.put("successor_continuation", canonical_continuation(continuation_raw))
    |> Map.put("env_kwargs", canonical_env_kwargs(env_kwargs_raw))
    |> Map.put("commit_gate", canonical_commit_gate(commit_gate_raw))
    |> Map.put("obs_augmentation", canonical_obs_aug(obs_aug_raw))
  end

  def canonicalize(_), do: canonicalize(%{})

  @doc """
  Stable 16-hex-char signature over a *canonicalized* sub-config.
  Use `sig_for_config/1` if you have a raw config; this lower-level
  entry point exists for the migration's backfill which works on
  already-canonicalized maps.

  16 hex chars = 64 bits — collision-free at the scale of
  human-submitted experiments by many orders of magnitude.
  """
  @spec hash(map()) :: String.t()
  def hash(canonical) when is_map(canonical) do
    canonical
    |> stable_encode()
    |> then(&:crypto.hash(:sha256, &1))
    |> Base.encode16(case: :lower)
    |> String.slice(0, 16)
  end

  @doc """
  Convenience: canonicalize a raw config and return both the
  canonical map and its hash. Bootstrap and the submit endpoint
  call this once per experiment.
  """
  @spec sig_for_config(map()) :: {String.t(), map()}
  def sig_for_config(config) do
    canonical = canonicalize(config)
    {hash(canonical), canonical}
  end

  # JSON-shaped string with keys in @policy_shape_keys order. Two
  # configs canonicalize-equal iff they produce the same string, so
  # the order MUST be deterministic — Map iteration order in BEAM
  # is implementation-defined and changes with internal representation
  # (flat-map vs. hashed-map transition at 32 keys). We hand-build
  # the object string to dodge that entirely.
  #
  # feature_types/tridiag_dims values are already canonicalized to
  # sorted lists by normalize_value, so a recursive sort isn't needed.
  defp stable_encode(map) do
    body =
      @policy_shape_keys
      |> Enum.map(fn k ->
        "\"#{k}\":" <> Jason.encode!(Map.get(map, k))
      end)
      |> Enum.join(",")

    # Backward-compatible lineage fork on the verifier strategy. The
    # default (`random`) is omitted entirely, so the digest is
    # byte-for-byte identical to every signature minted before the
    # verifier seam existed — no lineage is orphaned. A non-default
    # verifier (e.g. `ga_qd`) is appended, forking it into its own
    # lineage so an adversarial run starts from baseline rather than
    # inheriting a random-trained policy.
    verifier_suffix =
      case Map.get(map, "verifier", "random") do
        v when v in [nil, "random"] -> ""
        v -> ",\"verifier\":" <> Jason.encode!(v)
      end

    # Same backward-compatible fork as the verifier: run_seed 0 (the
    # default) contributes nothing to the digest, so existing signatures
    # are unchanged; a non-zero run_seed appends and forks a fresh
    # replicate lineage.
    run_seed_suffix =
      case Map.get(map, "run_seed", 0) do
        s when s in [nil, 0] -> ""
        s -> ",\"run_seed\":" <> Jason.encode!(s)
      end

    # Same backward-compatible fork as verifier/run_seed: the default
    # proposer (`enumerate`) contributes nothing, so pre-existing
    # signatures are byte-identical; `ga` appends and forks a fresh
    # lineage so a genetic-search policy never inherits enumerate commits.
    proposer_suffix =
      case Map.get(map, "proposer", "enumerate") do
        p when p in [nil, "enumerate"] -> ""
        p -> ",\"proposer\":" <> Jason.encode!(p)
      end

    scorer_suffix =
      case Map.get(map, "scorer", "episode") do
        s when s in [nil, "episode"] -> ""
        s -> ",\"scorer\":" <> Jason.encode!(s)
      end

    # Fork on the successor continuation. Hand-built key order (type,
    # hold, actions) keeps the digest deterministic; the actions list is
    # part of the identity — two different reference cycles are two
    # different ranking objectives and must be separate lineages.
    continuation_suffix =
      case Map.get(map, "successor_continuation") do
        %{"type" => type, "hold" => hold, "actions" => actions} ->
          ",\"successor_continuation\":{\"type\":" <>
            Jason.encode!(type) <>
            ",\"hold\":" <>
            Jason.encode!(hold) <>
            ",\"actions\":" <> Jason.encode!(actions) <> "}"

        _ ->
          ""
      end

    # Fork on per-experiment env kwargs. Keys sorted so the digest is
    # deterministic regardless of map construction order.
    env_kwargs_suffix =
      case Map.get(map, "env_kwargs") do
        kw when is_map(kw) and map_size(kw) > 0 ->
          inner =
            kw
            |> Enum.sort_by(fn {k, _} -> k end)
            |> Enum.map(fn {k, v} -> Jason.encode!(k) <> ":" <> Jason.encode!(v) end)
            |> Enum.join(",")

          ",\"env_kwargs\":{" <> inner <> "}"

        _ ->
          ""
      end

    commit_gate_suffix =
      case Map.get(map, "commit_gate", "episode") do
        g when g in [nil, "episode"] -> ""
        g -> ",\"commit_gate\":" <> Jason.encode!(g)
      end

    obs_aug_suffix =
      case Map.get(map, "obs_augmentation") do
        aug when is_map(aug) and map_size(aug) > 0 ->
          encoded =
            aug
            |> Enum.sort_by(fn {k, _} -> k end)
            |> Enum.map(fn {k, v} -> Jason.encode!(k) <> ":" <> Jason.encode!(v) end)
            |> Enum.join(",")

          ",\"obs_augmentation\":{" <> encoded <> "}"

        _ ->
          ""
      end

    "{" <>
      body <>
      verifier_suffix <>
      run_seed_suffix <>
      proposer_suffix <>
      scorer_suffix <>
      continuation_suffix <>
      env_kwargs_suffix <> commit_gate_suffix <> obs_aug_suffix <> "}"
  end

  # `nil` is a valid value (means "use Synthex's default feature set"
  # or "no tridiag dims"); preserve it through the sig so a submission
  # that explicitly omits feature_types and one that explicitly sets
  # feature_types=nil collapse to the same lineage.
  defp normalize_value(_key, nil), do: nil

  defp normalize_value("feature_types", list) when is_list(list) do
    list
    |> Enum.map(&canonical_feature/1)
    |> Enum.sort()
    |> Enum.uniq()
  end

  defp normalize_value("tridiag_dims", [lo, hi]) when is_integer(lo) and is_integer(hi),
    do: [lo, hi]

  defp normalize_value("tridiag_dims", %{"lo" => lo, "hi" => hi})
       when is_integer(lo) and is_integer(hi),
       do: [lo, hi]

  defp normalize_value("tridiag_dims", %{lo: lo, hi: hi})
       when is_integer(lo) and is_integer(hi),
       do: [lo, hi]

  defp normalize_value(_key, n) when is_integer(n), do: n
  defp normalize_value(_key, n) when is_float(n), do: trunc(n)
  defp normalize_value(_key, n) when is_binary(n), do: n
  defp normalize_value(_key, n) when is_atom(n), do: Atom.to_string(n)
  defp normalize_value(_key, n) when is_list(n), do: n

  defp canonical_feature(f) when is_binary(f) do
    Map.get(@feature_canonical, f) ||
      raise ArgumentError, "unknown feature type in config: #{inspect(f)}"
  end

  defp canonical_feature(f) when is_atom(f), do: canonical_feature(Atom.to_string(f))

  defp canonical_verifier(v) when v in ["ga_qd", :ga_qd], do: "ga_qd"
  defp canonical_verifier(_), do: "random"

  defp canonical_continuation(%{} = cont) do
    type = Map.get(cont, "type") || Map.get(cont, :type)
    actions = Map.get(cont, "actions") || Map.get(cont, :actions)
    hold = Map.get(cont, "hold") || Map.get(cont, :hold) || 1

    if to_string(type) == "cycle" and is_list(actions) and actions != [] do
      %{
        "type" => "cycle",
        "hold" => canonical_int(hold),
        "actions" => Enum.map(actions, fn row -> Enum.map(row, &(&1 * 1.0)) end)
      }
    else
      nil
    end
  end

  defp canonical_continuation(_), do: nil

  # String-key the kwargs and float-normalize numbers so e.g. 0.1 and
  # 0.10 (or atom vs string keys) collapse to one lineage.
  defp canonical_env_kwargs(%{} = kw) when map_size(kw) > 0 do
    Map.new(kw, fn {k, v} ->
      {to_string(k), if(is_number(v), do: v * 1.0, else: v)}
    end)
  end

  defp canonical_env_kwargs(_), do: nil

  defp canonical_commit_gate(g) when g in ["successor", :successor], do: "successor"
  defp canonical_commit_gate(g) when g in ["unfolding", :unfolding], do: "unfolding"
  defp canonical_commit_gate(_), do: "episode"

  # String-key and normalize the augmentation spec: prev_obs to int,
  # prev_action to bool, clock to a list of ints (order preserved —
  # it determines appended dim order). Empty/degenerate specs collapse
  # to nil so `{}` hashes identically to absent.
  defp canonical_obs_aug(aug) when is_map(aug) do
    k =
      case Map.get(aug, "prev_obs", Map.get(aug, :prev_obs)) do
        n when is_integer(n) and n > 0 -> n
        n when is_float(n) and n >= 1 -> trunc(n)
        _ -> 0
      end

    pa = Map.get(aug, "prev_action", Map.get(aug, :prev_action)) in [true, "true"]

    clock =
      case Map.get(aug, "clock", Map.get(aug, :clock)) do
        l when is_list(l) ->
          l
          |> Enum.map(fn
            p when is_integer(p) -> p
            p when is_float(p) -> trunc(p)
            _ -> 0
          end)
          |> Enum.filter(&(&1 > 0))

        _ ->
          []
      end

    out = %{}
    out = if k > 0, do: Map.put(out, "prev_obs", k), else: out
    out = if pa, do: Map.put(out, "prev_action", true), else: out
    out = if clock != [], do: Map.put(out, "clock", clock), else: out

    if map_size(out) > 0, do: out, else: nil
  end

  defp canonical_obs_aug(_), do: nil

  defp canonical_int(n) when is_integer(n), do: n
  defp canonical_int(n) when is_float(n), do: trunc(n)
  defp canonical_int(_), do: 1

  defp canonical_run_seed(n) when is_integer(n) and n >= 0, do: n
  defp canonical_run_seed(n) when is_float(n) and n >= 0, do: trunc(n)

  defp canonical_run_seed(n) when is_binary(n) do
    case Integer.parse(n) do
      {i, _} when i >= 0 -> i
      _ -> 0
    end
  end

  defp canonical_run_seed(_), do: 0

  defp canonical_proposer(p) when p in ["ga", :ga], do: "ga"
  defp canonical_proposer(_), do: "enumerate"

  defp canonical_scorer(s) when s in ["successor", :successor], do: "successor"
  defp canonical_scorer(_), do: "episode"

  @doc """
  Short human label for surfacing on the dashboard. e.g.
  `"b=3 · d=1 · f=axis,diag,prod"`. Reads the canonical form so the
  label is stable across raw-config syntactic variants.
  """
  @spec summary(map()) :: String.t()
  def summary(canonical) when is_map(canonical) do
    bits = Map.get(canonical, "bits_per_dim")
    depth = Map.get(canonical, "depth")
    feats = Map.get(canonical, "feature_types")
    max_coeff = Map.get(canonical, "max_coeff")
    tridiag = Map.get(canonical, "tridiag_dims")
    verifier = Map.get(canonical, "verifier")
    run_seed = Map.get(canonical, "run_seed", 0)
    proposer = Map.get(canonical, "proposer", "enumerate")
    scorer = Map.get(canonical, "scorer", "episode")

    parts = [
      bits && "b=#{bits}",
      depth && "d=#{depth}",
      feats && "f=#{Enum.join(feats, ",")}",
      max_coeff && max_coeff != @defaults["max_coeff"] && "c=#{max_coeff}",
      tridiag && "t=#{Enum.join(tridiag, ":")}",
      run_seed && run_seed != 0 && "r#{run_seed}",
      # Surface the counterexample strategy on the card. The default
      # (`random`) is omitted so existing cards are unchanged — its
      # absence means standard CEGAR; presence of a verifier badge means
      # an adversarial source is driving the search.
      verifier && verifier != @defaults["verifier"] && verifier_badge(verifier),
      # Surface the proposer (default `enumerate` omitted): a "GA" badge
      # means the genetic composition search drove candidate generation.
      proposer && proposer != @defaults["proposer"] && proposer_badge(proposer),
      scorer && scorer != @defaults["scorer"] && scorer_badge(scorer)
    ]

    parts
    |> Enum.filter(& &1)
    |> Enum.join(" · ")
  end

  defp verifier_badge("ga_qd"), do: "GAQD"
  defp verifier_badge(v), do: "verifier=#{v}"

  defp proposer_badge("ga"), do: "GA"
  defp proposer_badge(p), do: "proposer=#{p}"

  defp scorer_badge("successor"), do: "SUCC"
  defp scorer_badge(s), do: "scorer=#{s}"

  @doc "Fields whose change forks a new policy lineage. Exposed for tests."
  def policy_shape_keys, do: @policy_shape_keys

  @doc "Default values used when a key is omitted from the raw config."
  def defaults, do: @defaults
end
