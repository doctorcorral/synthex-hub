defmodule Server.Repo.Migrations.AddEnvPolicyValidationAvg do
  use Ecto.Migration

  # Promote held-out validation to a first-class lineage metric.
  #
  # `best_reward` is a TRAINING-block high-water-mark (the single best
  # candidate's sum over the per-round scoring seeds, max-selected over
  # thousands of noisy candidates). It is NOT comparable across runs or
  # lineages and routinely overstates true performance. `validation_avg`
  # is the per-episode mean of the current predicates on the FIXED
  # held-out validation seed block — the honest, comparable number the
  # dashboard should headline. Written by the controller each step
  # (after the validation guard decision). Nullable: pre-existing
  # lineages have none until their next step runs.
  def change do
    alter table(:env_policies) do
      add :validation_avg, :float
      add :validation_version, :integer
    end
  end
end
