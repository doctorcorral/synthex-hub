defmodule Server.Workers.ExperimentComplete do
  @moduledoc """
  Final job in an experiment's lifecycle. Runs once when the
  `ExperimentController` has advanced through all configured
  `cegar_rounds`.

  Responsibilities:

    1. Read the env_policy attached to this experiment and validate
       its predicates against `Synthex.Gym.Mujoco.validation_seeds/0`.
    2. Persist the final reward as the session's `best_reward` (a
       snapshot of the lineage state at the end of this session).
    3. Flip `status → completed` and stamp `completed_at`.
    4. Emit a `"info"` `system_event` noting the session finished.

  The env_policy itself is not mutated here — the lineage's state is
  whatever the commit gate accepted during the session. Future
  sessions inherit from there.

  Idempotent: if it runs twice it just re-validates and re-stamps;
  no duplicate side effects.
  """

  use Oban.Worker, queue: :master, max_attempts: 3
  require Logger

  alias Server.{EnvPolicies, Experiments, Experiment}
  alias Server.Workers.ExperimentBootstrap
  alias Synthex.Core.PrettyPrint

  @impl Oban.Worker
  def perform(%Oban.Job{args: %{"experiment_id" => id}}) do
    case Experiments.get(id) do
      {:error, :not_found} ->
        {:discard, "experiment not found"}

      {:ok, %Experiment{status: status}} when status in ["completed", "failed", "cancelled"] ->
        :ok

      {:ok, %Experiment{} = exp} ->
        finalize(exp)
    end
  end

  defp finalize(%Experiment{} = exp) do
    env_key = ExperimentBootstrap.decode_env_key(exp.env_key)
    # build_context/3 already derives the correct ctx.env_name (the
    # warp variant included) from env_key + config — same as the
    # controller's run_step. Passing exp.env_name here called a
    # nonexistent build_context/4 and crashed finalize/1.
    ctx = ExperimentBootstrap.build_context(env_key, exp.config, exp.id)

    {:ok, env_policy} = EnvPolicies.for_experiment(exp)
    preds = decode_predicates(env_policy.predicates)

    val_seeds = Synthex.Gym.Mujoco.validation_seeds()
    {total_reward, survived} = Synthex.Gym.Mujoco.validate(preds, val_seeds, ctx)
    final_avg = total_reward / length(val_seeds)
    n_episodes = ctx.n_episodes

    Logger.info(
      "[Complete] #{exp.env_name} final validation: avg=#{Float.round(final_avg / n_episodes, 2)}/ep " <>
        "survived=#{survived}/#{length(val_seeds)} (lineage v=#{env_policy.policy_version})"
    )

    {:ok, _exp} =
      Experiments.update_state(exp, %{
        "status" => "completed",
        "completed_at" => DateTime.utc_now(),
        "best_reward" => max(exp.best_reward || final_avg, final_avg)
      })

    Experiments.log_event!(
      "info",
      "master",
      "experiment completed: #{exp.env_name} final_avg=#{Float.round(final_avg / n_episodes, 2)}/ep " <>
        "(session baseline #{Float.round((exp.baseline_reward || 0.0) / n_episodes, 2)}/ep, " <>
        "#{exp.accepted_count} bits accepted, lineage v=#{env_policy.policy_version})",
      env_name: exp.env_name,
      experiment_id: exp.id,
      metadata: %{
        "final_avg" => final_avg,
        "survived" => survived,
        "baseline_reward" => exp.baseline_reward,
        "accepted_count" => exp.accepted_count,
        "env_policy_id" => env_policy.id,
        "lineage_policy_version" => env_policy.policy_version
      }
    )

    :ok
  end

  defp decode_predicates(%{"preds" => list}) when is_list(list),
    do: Enum.map(list, &PrettyPrint.from_json_term/1)

  defp decode_predicates(_), do: []
end
