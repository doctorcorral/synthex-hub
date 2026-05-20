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
    "tridiag_dims" => nil
  }

  @feature_canonical %{
    "axis" => "axis",
    "diag" => "diag",
    "sq_diag" => "sq_diag",
    "prod" => "prod",
    "tridiag" => "tridiag"
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
    @policy_shape_keys
    |> Enum.map(fn key ->
      raw = Map.get(config, key, Map.get(config, String.to_atom(key), Map.get(@defaults, key)))
      {key, normalize_value(key, raw)}
    end)
    |> Map.new()
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

  # JSON with keys sorted top-level. The canonical map only has
  # @policy_shape_keys at the top so a one-level sort is enough; we
  # avoid recursive sort because feature_types/tridiag_dims are
  # already normalized to deterministic forms by normalize_value.
  defp stable_encode(map) do
    pairs =
      @policy_shape_keys
      |> Enum.map(fn k -> {k, Map.get(map, k)} end)

    Jason.encode!(pairs)
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

    parts = [
      bits && "b=#{bits}",
      depth && "d=#{depth}",
      feats && "f=#{Enum.join(feats, ",")}",
      max_coeff && max_coeff != @defaults["max_coeff"] && "c=#{max_coeff}",
      tridiag && "t=#{Enum.join(tridiag, ":")}"
    ]

    parts
    |> Enum.filter(& &1)
    |> Enum.join(" · ")
  end

  @doc "Fields whose change forks a new policy lineage. Exposed for tests."
  def policy_shape_keys, do: @policy_shape_keys

  @doc "Default values used when a key is omitted from the raw config."
  def defaults, do: @defaults
end
