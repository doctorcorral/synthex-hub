defmodule Server.ObanFailureHandler do
  @moduledoc """
  Telemetry handler for Oban job failures. Catches the two failure
  modes that actually matter for surfacing experiments incidents:

    * `[:oban, :job, :exception]` with `state == :discarded`
      (max_attempts exhausted)
    * `[:oban, :job, :exception]` with state `:failure` (a retry
      attempt failed, but more attempts remain — we log these at
      info level so a trail exists)

  On terminal failure of an `ExperimentBootstrap` / `ExperimentController` /
  `ExperimentComplete` job, the handler:

    1. Marks the corresponding experiment `failed` and records the
       error string on the row.
    2. Inserts a `level=error` `system_event` so the landing page's
       incident banner picks it up.

  Failures of other workers (e.g. `ReapWorkers`) are logged but don't
  touch experiments.
  """

  require Logger
  alias Server.{Experiments, Experiment}

  @events [
    [:oban, :job, :exception]
  ]

  @master_workers ~w(
    Server.Workers.ExperimentBootstrap
    Server.Workers.ExperimentController
    Server.Workers.ExperimentComplete
  )

  @doc """
  Attach the handler globally. Called once from
  `Server.Application.start/2` after the Repo and Oban have come up.
  """
  def attach do
    :telemetry.attach_many(
      "server-oban-failure-handler",
      @events,
      &__MODULE__.handle_event/4,
      %{}
    )
  end

  @doc false
  def handle_event([:oban, :job, :exception], measurements, metadata, _config) do
    job = metadata.job
    state = Map.get(metadata, :state) || :failure

    cond do
      # Oban emits :discard when a job exhausts max_attempts.
      # That's the terminal failure we want to surface as an
      # incident — everything else (transient :failure on a
      # retryable attempt) is logged at warn but doesn't trip
      # the landing page banner.
      state == :discard ->
        handle_discarded(job, metadata, measurements)

      state == :failure ->
        Logger.warning(
          "[Oban] job #{job.id} (#{job.worker}) attempt #{job.attempt}/#{job.max_attempts} failed: " <>
            inspect(metadata[:reason])
        )

      true ->
        :ok
    end
  end

  defp handle_discarded(job, metadata, _measurements) do
    reason = format_reason(metadata)
    Logger.error("[Oban] job #{job.id} (#{job.worker}) discarded: #{reason}")

    if job.worker in @master_workers do
      mark_experiment_failed(job, reason)
    else
      # Non-master worker failure: still surface as an incident so we
      # don't lose visibility, but don't try to update experiments.
      Experiments.log_event!(
        "error",
        "oban",
        "#{job.worker} job exhausted retries: #{reason}",
        metadata: %{
          "worker" => job.worker,
          "job_id" => job.id,
          "args" => job.args
        }
      )
    end
  end

  defp mark_experiment_failed(job, reason) do
    case Map.get(job.args, "experiment_id") do
      nil ->
        Logger.warning("[Oban] master worker #{job.worker} discarded without experiment_id")

      exp_id ->
        case Experiments.get(exp_id) do
          {:ok, %Experiment{status: status} = exp} when status not in ["completed", "cancelled"] ->
            error_msg =
              "#{worker_short_name(job.worker)} failed after #{job.attempt} attempts: #{reason}"

            {:ok, _} = Experiments.mark_failed(exp, error_msg)

            Experiments.log_event!(
              "error",
              "oban",
              "experiment failed: #{exp.env_name} — #{error_msg}",
              env_name: exp.env_name,
              experiment_id: exp.id,
              metadata: %{
                "worker" => job.worker,
                "job_id" => job.id,
                "attempts" => job.attempt
              }
            )

          {:ok, exp} ->
            Logger.info(
              "[Oban] master job for already-#{exp.status} experiment #{exp_id} discarded; ignoring"
            )

          {:error, :not_found} ->
            Logger.warning("[Oban] master job referenced unknown experiment #{exp_id}")
        end
    end
  end

  defp format_reason(%{reason: reason}) when not is_nil(reason) do
    reason |> inspect() |> String.slice(0, 500)
  end

  defp format_reason(%{kind: kind, error: error, stacktrace: _st}) do
    "#{kind}: #{inspect(error) |> String.slice(0, 400)}"
  end

  defp format_reason(_), do: "unknown failure"

  defp worker_short_name("Server.Workers.ExperimentBootstrap"), do: "bootstrap"
  defp worker_short_name("Server.Workers.ExperimentController"), do: "controller"
  defp worker_short_name("Server.Workers.ExperimentComplete"), do: "complete"
  defp worker_short_name(other), do: other
end
