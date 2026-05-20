defmodule Server.Repo.Migrations.RepairEnvPoliciesJsonb do
  use Ecto.Migration

  @moduledoc """
  Repairs env_policies.config_data and env_policies.predicates that
  were stored as JSONB-encoded strings by the initial backfill in
  20260520000001 (it pre-encoded maps with Jason.encode! before
  passing them to a $N::jsonb parameter, and Postgrex's default
  jsonb codec re-encoded the string, producing `"{...}"` instead
  of `{...}`).

  Detection: `jsonb_typeof = 'string'` flags the corrupt rows.
  Repair: `(col #>> '{}')::jsonb` extracts the inner text and
  reparses it as JSON.

  Idempotent: rows where jsonb_typeof is already 'object' are
  untouched, so re-running the migration is safe.
  """

  def up do
    execute("""
    UPDATE env_policies
    SET config_data = (config_data #>> '{}')::jsonb
    WHERE jsonb_typeof(config_data) = 'string'
    """)

    execute("""
    UPDATE env_policies
    SET predicates = (predicates #>> '{}')::jsonb
    WHERE jsonb_typeof(predicates) = 'string'
    """)
  end

  def down do
    # No down: re-stringifying healthy JSONB would itself be a
    # corruption, and there's no business reason to revert.
    :ok
  end
end
