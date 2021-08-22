import click
import subprocess
import os

from mlflow_fastapi.prepare_model import do_prepare_model

@click.group()
def cli():
    pass

@cli.command()
def serve_model():
    """Example script."""

    command = "uvicorn mlflow_fastapi.main:app"

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
