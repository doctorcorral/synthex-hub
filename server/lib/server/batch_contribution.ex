defmodule Server.BatchContribution do
  @moduledoc """
  Per-(batch, worker) work attribution. One row per worker that
  helped on a given batch. `display_name` is denormalized from
  `WorkerNode.name` so that all anonymous workers (whose name is
  literally `"anonymous"`) collapse into one leaderboard row when
  we `GROUP BY display_name`.
  """
  use Ecto.Schema
  import Ecto.Changeset

  @primary_key false
  schema "batch_contributions" do
    field :batch_id, :string, primary_key: true
    field :worker_id, :string, primary_key: true
    field :display_name, :string
    field :chunks_completed, :integer, default: 0
    field :candidates_evaluated, :integer, default: 0
    field :first_chunk_at, :utc_datetime_usec
    field :last_chunk_at, :utc_datetime_usec
  end

  def changeset(contribution, attrs) do
    contribution
    |> cast(attrs, [
      :batch_id,
      :worker_id,
      :display_name,
      :chunks_completed,
      :candidates_evaluated,
      :first_chunk_at,
      :last_chunk_at
    ])
    |> validate_required([
      :batch_id,
      :worker_id,
      :display_name,
      :first_chunk_at,
      :last_chunk_at
    ])
  end
end
