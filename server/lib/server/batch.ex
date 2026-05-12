defmodule Server.Batch do
  @moduledoc """
  A synthesis experiment submitted by a master.
  Tracks progress; results aggregate as chunks complete.
  """
  use Ecto.Schema
  import Ecto.Changeset

  @primary_key {:id, :string, autogenerate: false}
  schema "batches" do
    field :name, :string
    field :env_name, :string
    field :cmd, :string, default: "score_bit"
    field :payload, :map, default: %{}
    field :total_chunks, :integer, default: 0
    field :completed_chunks, :integer, default: 0
    field :status, :string, default: "pending"
    field :results, {:array, :map}, default: []
    field :submitter, :string
    field :completed_at, :utc_datetime_usec
    field :ttl_at, :utc_datetime_usec

    # Reward aggregates, maintained as chunks complete. Cached so the
    # public landing page can render a per-environment leaderboard
    # without scanning the (potentially many-MB) `results` array on
    # every request. NULL while no rewards have arrived yet, or for
    # non-`score_bit` batches.
    field :best_reward, :float
    field :baseline_reward, :float

    # Stamped on every `GET /api/master/batches/:id` so we can tell
    # whether the master is alive. Used by `experiments_summary/0`
    # to compute the `health` flag surfaced on the landing page.
    field :master_polled_at, :utc_datetime_usec

    timestamps(type: :utc_datetime_usec)
  end

  def changeset(batch, attrs) do
    batch
    |> cast(attrs, [
      :id,
      :name,
      :env_name,
      :cmd,
      :payload,
      :total_chunks,
      :completed_chunks,
      :status,
      :results,
      :submitter,
      :completed_at,
      :ttl_at,
      :best_reward,
      :baseline_reward,
      :master_polled_at
    ])
    |> validate_required([:id, :env_name, :cmd, :total_chunks])
    |> validate_inclusion(:status, ~w(pending running completed failed cancelled))
  end
end
