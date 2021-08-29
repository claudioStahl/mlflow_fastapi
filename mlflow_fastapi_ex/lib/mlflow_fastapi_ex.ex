defmodule MlflowFastapiEx do
  @moduledoc """
  Documentation for `MlflowFastapiEx`.
  """

  @doc """
  Hello world.

  ## Examples

      iex> MlflowFastapiEx.hello()
      :world

  """
  def hello do
    :world
  end

  def predict do
    pypath = '/home/claudio/Workplaces/Stone/mlflow_fastapi/mlflow_fastapi:'
    pybinpath = '/home/claudio/anaconda3/envs/mlflow_fastapi/bin/python'
    {:ok, python_pid} = :python.start(python_path: pypath, python: pybinpath)

    content_type = "application/json; format=pandas-split"
    payload = ~s({"columns":["alcohol", "chlorides", "citric acid", "density", "fixed acidity", "free sulfur dioxide", "pH", "residual sugar", "sulphates", "total sulfur dioxide", "volatile acidity"],"data":[[12.8, 0.029, 0.48, 0.98, 6.2, 29, 3.33, 1.2, 0.39, 75, 0.66]]})

    result =
      :python.call(
        python_pid,
        :elixir_wrapper,
        :ex_parse_and_predict,
        [content_type, payload]
      )
  end
end
