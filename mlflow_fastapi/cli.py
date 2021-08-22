import click
import subprocess
import os

from mlflow_fastapi.prepare_model import do_prepare_model

@click.group()
def cli():
    pass

@cli.command()
@click.option(
    "--host",
    "-h",
    metavar="HOST",
    default="127.0.0.1",
    help="The network address to listen on (default: 127.0.0.1). "
    "Use 0.0.0.0 to bind to all addresses if you want to access the tracking "
    "server from other machines.",
)
@click.option("--port", "-p", default=8000, help="The port to listen on (default: 5000).")
@click.option(
    "--workers",
    "-w",
    default=4,
    help="Number of gunicorn worker processes to handle requests (default: 4).",
)
def serve_model(host, port, workers):
    """
    Serve a model saved with MLflow by launching a webserver on the specified host and port.
    The command supports models with the ``python_function`` or ``crate`` (R Function) flavor.
    For information about the input data formats accepted by the webserver, see the following
    documentation: https://www.mlflow.org/docs/latest/models.html#built-in-deployment-tools.

    You can make requests to ``POST /invocations`` in pandas split- or record-oriented formats.

    Example:

    .. code-block:: bash

        $ mlflow models serve -m runs:/my-run-id/model-path &

        $ curl http://127.0.0.1:5000/invocations -H 'Content-Type: application/json' -d '{
            "columns": ["a", "b", "c"],
            "data": [[1, 2, 3], [4, 5, 6]]
        }'
    """

    command = (
        "uvicorn --host={host} --port={port} --workers={workers}"
        " mlflow_fastapi.main:app"
    ).format(host=host, port=port, workers=workers)

    command_env = os.environ.copy()

    if os.name != "nt":
      subprocess.Popen(["bash", "-c", command], env=command_env).wait()
    else:
      subprocess.Popen(command, env=command_env).wait()

@cli.command()
@click.option(
    "--model-uri",
    "-m",
    default=None,
    metavar="URI",
    required=True,
    help="URI to the model. A local path, a 'runs:/' URI, or a"
    " remote storage URI (e.g., an 's3://' URI). For more information"
    " about supported remote URIs for model artifacts, see"
    " https://mlflow.org/docs/latest/tracking.html"
    "#artifact-stores",
)
def prepare_model(model_uri):
  do_prepare_model(model_uri)
  click.echo("Success")

if __name__ == '__main__':
    cli()
