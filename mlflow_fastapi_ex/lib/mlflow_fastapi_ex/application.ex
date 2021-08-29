defmodule MlflowFastapiEx.Application do
  # See https://hexdocs.pm/elixir/Application.html
  # for more information on OTP Applications
  @moduledoc false

  use Application

  @impl true
  def start(_type, _args) do
    IO.puts("node=#{node()}")

    children = [
      # Starts a worker by calling: MlflowFastapiEx.Worker.start_link(arg)
      # {MlflowFastapiEx.Worker, arg}
      {Task.Supervisor, name: MlflowFastapiEx.RemoteTaskSupervisor}
    ]

    # See https://hexdocs.pm/elixir/Supervisor.html
    # for other strategies and supported options
    opts = [strategy: :one_for_one, name: MlflowFastapiEx.Supervisor]
    Supervisor.start_link(children, opts)
  end
end
