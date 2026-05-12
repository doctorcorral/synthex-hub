defmodule Server.Repo.Migrations.AddBatchMasterPolledAt do
  @moduledoc """
  Track when a batch's master last polled its status. Masters call
  `GET /api/master/batches/:id` every few seconds while waiting for
  chunks to complete; if this timestamp goes stale (e.g. > 5 min in
  the past) while the batch is still in flight, the master is
  presumed dead and the landing page surfaces a "stalled" indicator.

  Without this signal the landing page conflated "master crashed
  9 hours ago" with "experiment is just slow" — both showed
  `progress < 100%` and a growing `elapsed_seconds`.
  """
  use Ecto.Migration

  def change do
    alter table(:batches) do
      add :master_polled_at, :utc_datetime_usec
    end

    create index(:batches, [:master_polled_at])
  end
end
