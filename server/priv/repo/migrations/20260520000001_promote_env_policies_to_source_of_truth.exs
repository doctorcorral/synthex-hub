defmodule Server.Repo.Migrations.PromoteEnvPoliciesToSourceOfTruth do
  use Ecto.Migration

  @moduledoc """
  Decouples the synthesized policy from the experiment that produced
  it. After this migration:

    * The `env_policies` table is the source of truth for predicates,
      policy_version, best_reward, and baseline_reward.
    * `experiments` is a session log — what one CEGAR run did to its
      env_policy. The `predicates` / `policy_version` /
      `best_reward_per_bit` columns are dropped from `experiments`.
    * `policy_versions` audit rows are keyed by `env_policy_id` (the
      lineage) rather than `experiment_id` (the session). The
      committing experiment is retained as attribution metadata.
    * Multiple experiments can run in parallel for the same `env_name`
      provided they differ in `config_sig`. Same-sig double-submits
      remain forbidden by partial unique index.

  ## Backfill

  For each unique `(env_name, config_sig)` discovered across existing
  experiments rows, an env_policies row is created. The "winner"
  experiment per group — the one whose `(accepted_count, best_reward)`
  is highest — donates its `predicates`, `policy_version`, and
  `best_reward`. The first-ever experiment in the group (by
  `inserted_at`) donates `baseline_reward`. Every experiment row in
  the group has its `env_policy_id` set to the new row.

  Existing `policy_versions` rows are linked to env_policies by
  joining through their `experiment_id` → `experiments.env_policy_id`.

  ## Reversibility

  Down is intentionally not provided. The data restructure is
  fundamentally one-way (we don't keep enough state to reconstruct
  the per-experiment `(predicates, policy_version)` columns after
  multiple experiments in the same lineage merge into a single
  env_policy row). If a rollback is ever needed, restore from a
  pre-migration backup.
  """

  alias Server.EnvPolicy.ConfigSig

  def up do
    # ── 1. Create env_policies ────────────────────────────────
    create table(:env_policies, primary_key: false) do
      add :id, :uuid, primary_key: true
      add :env_name, :string, null: false
      add :env_key, :string, null: false
      add :config_sig, :string, null: false
      add :config_data, :map, null: false, default: %{}

      add :predicates, :map, null: false, default: %{"preds" => []}
      add :policy_version, :integer, null: false, default: 0

      add :best_reward, :float
      add :baseline_reward, :float
      add :n_episodes, :integer

      add :last_committed_by_experiment_id,
          references(:experiments, type: :uuid, on_delete: :nilify_all)

      add :first_seen_at, :utc_datetime_usec, null: false, default: fragment("now()")

      timestamps(type: :utc_datetime_usec)
    end

    create unique_index(:env_policies, [:env_name, :config_sig])
    create index(:env_policies, [:env_name])
    create index(:env_policies, [:last_committed_by_experiment_id])

    # ── 2. experiments.env_policy_id (nullable until backfill) ─
    alter table(:experiments) do
      add :env_policy_id, references(:env_policies, type: :uuid, on_delete: :restrict)
    end

    create index(:experiments, [:env_policy_id])

    flush()

    # ── 3. Backfill env_policies + experiments.env_policy_id ──
    backfill_env_policies()

    # ── 4. Make env_policy_id NOT NULL ────────────────────────
    execute("ALTER TABLE experiments ALTER COLUMN env_policy_id SET NOT NULL")

    # ── 5. policy_versions: add env_policy_id; rekey unique ──
    alter table(:policy_versions) do
      add :env_policy_id, references(:env_policies, type: :uuid, on_delete: :delete_all)
    end

    flush()

    execute("""
    UPDATE policy_versions pv
    SET env_policy_id = e.env_policy_id
    FROM experiments e
    WHERE pv.experiment_id = e.id
    """)

    execute("ALTER TABLE policy_versions ALTER COLUMN env_policy_id SET NOT NULL")

    # Before rekeying the unique index, renumber `version` per
    # env_policy lineage. The pre-migration contract was
    # (experiment_id, version) unique, so each experiment's audit
    # entries started at version=1. After folding multiple
    # experiments into one lineage, multiple version=1 rows
    # collide on the new (env_policy_id, version) key.
    #
    # Reassign with ROW_NUMBER() OVER (PARTITION BY env_policy_id
    # ORDER BY inserted_at, version) so each lineage gets a
    # contiguous 1..N sequence aligned with commit chronology.
    # Within the same inserted_at tick, the original `version`
    # field breaks ties (it was already monotone per-experiment).
    drop unique_index(:policy_versions, [:experiment_id, :version])

    execute("""
    WITH renum AS (
      SELECT id,
             ROW_NUMBER() OVER (
               PARTITION BY env_policy_id
               ORDER BY inserted_at, version, id
             ) AS new_version
      FROM policy_versions
    )
    UPDATE policy_versions pv
    SET version = renum.new_version
    FROM renum
    WHERE pv.id = renum.id
    """)

    # Realign env_policies.policy_version to the renumbered max so
    # future commits via Server.CommitGate (which increments
    # env_policy.policy_version) don't collide with historical
    # audit-log entries. If a lineage has zero audit rows (no
    # commits ever landed for it), leave policy_version at the
    # default 0 — its env_policy.policy_version was inherited
    # from a winner experiment that itself had no audit entries.
    execute("""
    UPDATE env_policies ep
    SET policy_version = sub.max_v
    FROM (
      SELECT env_policy_id, MAX(version) AS max_v
      FROM policy_versions
      GROUP BY env_policy_id
    ) sub
    WHERE ep.id = sub.env_policy_id
    """)

    create unique_index(:policy_versions, [:env_policy_id, :version])
    create index(:policy_versions, [:env_policy_id, :inserted_at])

    # `experiment_id` is retained on policy_versions as attribution
    # metadata: "which session triggered this commit". Its existing
    # `on_delete: :delete_all` is too aggressive now — if a session
    # gets deleted, its versions are part of the env_policy's
    # history and must survive. Relax to nilify_all (which requires
    # the column to be nullable, so drop NOT NULL too).
    drop constraint(:policy_versions, "policy_versions_experiment_id_fkey")
    execute("ALTER TABLE policy_versions ALTER COLUMN experiment_id DROP NOT NULL")

    alter table(:policy_versions) do
      modify :experiment_id, references(:experiments, type: :uuid, on_delete: :nilify_all)
    end

    # ── 6. Drop now-redundant experiment columns ──────────────
    #
    # `predicates`, `policy_version`, `best_reward_per_bit` are
    # promoted to env_policies (the lineage owns them now).
    #
    # `bit_shuffle`, `bit_progress` are vestigial state from the
    # Jacobi-iter `ExperimentCegarIter` master that the streaming
    # controller replaced — the controller writes them as empty
    # arrays on advance for transitional compatibility but never
    # reads them. Removing now to avoid carrying dead schema
    # forward; the dashboard's `bits_done` is replaced by
    # `accepted_count` (already used) plus the in-flight per-bit
    # work surfaced via `AggregateBroker.experiment_flow/1`.
    alter table(:experiments) do
      remove :predicates
      remove :policy_version
      remove :best_reward_per_bit
      remove :bit_shuffle
      remove :bit_progress
    end

    # ── 7. Swap the active-row uniqueness from env to env_policy ─
    drop unique_index(:experiments, [:env_name], name: :experiments_one_active_per_env)

    create unique_index(:experiments, [:env_policy_id],
             where: "status IN ('pending','running')",
             name: :experiments_one_active_per_env_policy
           )
  end

  def down do
    raise Ecto.MigrationError,
      message:
        "PromoteEnvPoliciesToSourceOfTruth is irreversible. Restore from backup if needed."
  end

  # ── Backfill helpers ──────────────────────────────────────────

  # Walks every experiment, computes its (env_name, config_sig),
  # groups them, picks a winner per group, inserts env_policies,
  # and sets env_policy_id on each experiment. Run inside the
  # migration transaction.
  defp backfill_env_policies do
    repo = repo()

    # NOTE: select raw UUIDs (not ::text) so Postgrex returns them as
    # 16-byte binaries — that's the only form `$N::uuid` parameters
    # accept on the way back into INSERT / UPDATE statements. The
    # earlier draft used `::text` and bounced off Postgrex's strict
    # pre-encode validator with "expected a binary of 16 bytes, got
    # '<uuid-string>'".
    %Postgrex.Result{rows: rows} =
      repo.query!("""
      SELECT id, env_name, env_key, config,
             predicates, policy_version,
             baseline_reward, best_reward, accepted_count,
             inserted_at
      FROM experiments
      ORDER BY inserted_at ASC
      """)

    grouped =
      rows
      |> Enum.map(&row_to_map/1)
      |> Enum.group_by(fn r ->
        sig = compute_sig(r.config || %{})
        {r.env_name, sig}
      end)

    Enum.each(grouped, fn {{env_name, sig}, exps} ->
      insert_env_policy_for_group(env_name, sig, exps)
    end)
  end

  defp row_to_map([id, env_name, env_key, config, predicates, policy_version,
                   baseline_reward, best_reward, accepted_count, inserted_at]) do
    %{
      id: id,
      env_name: env_name,
      env_key: env_key,
      config: config,
      predicates: predicates,
      policy_version: policy_version,
      baseline_reward: baseline_reward,
      best_reward: best_reward,
      accepted_count: accepted_count,
      inserted_at: inserted_at
    }
  end

  defp compute_sig(config) do
    {sig, _canonical} = ConfigSig.sig_for_config(config)
    sig
  end

  defp insert_env_policy_for_group(env_name, sig, exps) do
    repo = repo()

    canonical = ConfigSig.canonicalize(List.first(exps).config || %{})

    # First experiment in the group (by inserted_at, already sorted)
    # donates the baseline. If multiple sessions exist, the oldest
    # one's baseline IS the lineage's "from scratch" floor.
    first = List.first(exps)
    baseline = first.baseline_reward

    # Winner experiment donates the predicates+version+best.
    # Tie-break: higher accepted_count, then higher best_reward,
    # then most recent inserted_at.
    winner =
      Enum.max_by(exps, fn e ->
        # inserted_at can come back as DateTime or NaiveDateTime
        # depending on column type / Postgrex codec — use the
        # erlang term ordering which works for both.
        {e.accepted_count || 0, e.best_reward || -1.0e15, e.inserted_at}
      end)

    {predicates_json, version, best_reward, n_eps, env_key} =
      {
        winner.predicates || %{"preds" => []},
        winner.policy_version || 0,
        winner.best_reward,
        get_in(winner.config || %{}, ["n_episodes"]) || 30,
        winner.env_key
      }

    # NOTE: do NOT pre-encode maps with Jason.encode! before passing
    # them to a jsonb parameter — Postgrex's default jsonb codec
    # invokes Jason.encode! itself, so a pre-encoded string ends up
    # double-encoded as a jsonb STRING value (e.g. the column reads
    # back as `"{...}"` instead of `{...}`). Hand Postgrex the raw
    # map and let it serialize.
    %Postgrex.Result{rows: [[env_policy_id]]} =
      repo.query!(
        """
        INSERT INTO env_policies
          (id, env_name, env_key, config_sig, config_data,
           predicates, policy_version, best_reward, baseline_reward,
           n_episodes, last_committed_by_experiment_id,
           first_seen_at, inserted_at, updated_at)
        VALUES
          (gen_random_uuid(), $1, $2, $3, $4, $5, $6,
           $7, $8, $9, $10, now(), now(), now())
        RETURNING id
        """,
        [
          env_name,
          env_key,
          sig,
          canonical,
          predicates_json,
          version,
          best_reward,
          baseline,
          n_eps,
          if(winner.accepted_count && winner.accepted_count > 0, do: winner.id, else: nil)
        ]
      )

    # Backfill experiments.env_policy_id for every row in the group.
    exp_ids = Enum.map(exps, & &1.id)

    repo.query!(
      """
      UPDATE experiments
      SET env_policy_id = $1::uuid
      WHERE id = ANY($2::uuid[])
      """,
      [env_policy_id, exp_ids]
    )
  end
end
