# MlflowFastapiEx

## Installation

If [available in Hex](https://hex.pm/docs/publish), the package can be installed
by adding `mlflow_fastapi_ex` to your list of dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:mlflow_fastapi_ex, "~> 0.1.0"}
  ]
end
```

Documentation can be generated with [ExDoc](https://github.com/elixir-lang/ex_doc)
and published on [HexDocs](https://hexdocs.pm). Once published, the docs can
be found at [https://hexdocs.pm/mlflow_fastapi_ex](https://hexdocs.pm/mlflow_fastapi_ex).

## Development

Server A

    iex --sname servera --cookie cookiex -S mix

Server B

    iex --sname serverb --cookie cookiex -S mix
    Node.connect :"servera@claudio-Latitude-5400"
    Node.spawn(:"servera@claudio-Latitude-5400", MlflowFastapiEx.hello/0)
    {MlflowFastapiEx.RemoteTaskSupervisor, :"servera@claudio-Latitude-5400"} |> Task.Supervisor.async(MlflowFastapiEx, :hello, []) |> Task.await()
