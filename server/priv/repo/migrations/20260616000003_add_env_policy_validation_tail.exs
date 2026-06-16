defmodule Server.Repo.Migrations.AddEnvPolicyValidationTail do
  use Ecto.Migration

  # Tail / robustness statistics of the held-out validation block,
  # alongside the mean (`validation_avg`).
  #
  # Mean held-out reward is an average-case instrument: it's close to
  # blind to a policy whose value (or whose damage) is in the tail —
  # rare states where it fails badly. This jsonb captures the worst end
  # of the per-seed return distribution computed over the SAME
  # validation rollouts: `%{"cvar10" => mean of worst 10%, "worst" =>
  # min, "p10" => 10th percentile, "mean" => mean, "n" => n_seeds}`.
  #
  # Nullable: stays empty for lineages whose worker is too old to emit
  # per-seed returns, or that haven't run a step since this shipped.
  def change do
    alter table(:env_policies) do
      add :validation_tail, :map
    end
  end
end
