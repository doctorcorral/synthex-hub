defmodule Server.MixProject do
  use Mix.Project

  def project do
    [
      app: :server,
      version: "0.2.0",
      elixir: "~> 1.18",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      aliases: aliases(),
      releases: releases()
    ]
  end

  defp releases do
    [
      server: [
        include_executables_for: [:unix],
        applications: [runtime_tools: :permanent],
        steps: [:assemble, :tar]
      ]
    ]
  end

  def application do
    [
      extra_applications: [:logger],
      mod: {Server.Application, []}
    ]
  end

  defp deps do
    [
      {:bandit, "~> 1.0"},
      {:plug, "~> 1.15"},
      {:jason, "~> 1.4"},
      {:ecto_sql, "~> 3.12"},
      {:postgrex, "~> 0.19"},
      {:oban, "~> 2.18"},
      # The CEGAR synthesis primitives (`Synthex.Gym.Mujoco.optimize_bit/5`
      # etc.) live in the synthex repo. Server.Workers.* drives them
      # from inside Oban jobs, so the entire master loop is a
      # supervised, checkpointable process on the hub rather than a
      # one-shot script on someone's laptop. `synthex_hub_client`
      # carries the HTTP client and the `Synthex.Hub.Scorer` adapter;
      # we configure it to talk to localhost so we reuse all the
      # existing batch/chunk machinery without going through the
      # public proxy.
      #
      # Both deps default to public git refs so the Docker build
      # (whose context is `server/` only) can fetch them. Set
      # SYNTHEX_PATH / SYNTHEX_HUB_CLIENT_PATH to local checkouts
      # for live iteration against unpushed branches.
      synthex_dep(),
      synthex_hub_client_dep()
    ]
  end

  defp synthex_dep do
    case System.get_env("SYNTHEX_PATH") do
      path when is_binary(path) and path != "" ->
        {:synthex, path: path, override: true}

      _ ->
        ref = System.get_env("SYNTHEX_GIT_REF", "main")
        {:synthex, git: "https://github.com/doctorcorral/synthex.git", ref: ref, override: true}
    end
  end

  defp synthex_hub_client_dep do
    case System.get_env("SYNTHEX_HUB_CLIENT_PATH") do
      path when is_binary(path) and path != "" ->
        {:synthex_hub_client, path: path}

      _ ->
        ref = System.get_env("SYNTHEX_HUB_GIT_REF", "main")
        {:synthex_hub_client,
         git: "https://github.com/doctorcorral/synthex-hub.git",
         sparse: "client",
         ref: ref}
    end
  end

  defp aliases do
    [
      "ecto.setup": ["ecto.create", "ecto.migrate"],
      "ecto.reset": ["ecto.drop", "ecto.setup"]
    ]
  end
end
