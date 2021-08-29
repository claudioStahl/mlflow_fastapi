defmodule MlflowFastapiEx.MixProject do
  use Mix.Project

  def project do
    [
      app: :mlflow_fastapi_ex,
      version: "0.1.0",
      elixir: "~> 1.12",
      start_permanent: true,
      deps: deps()
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger],
      mod: {MlflowFastapiEx.Application, []}
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:erlport, "~> 0.10.1"},
      {:poolboy, "~> 1.5"}
    ]
  end
end
