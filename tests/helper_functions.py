import os
import random
import math
import numpy as np
from unittest import mock

import requests
import string
import time
import signal
import socket
import subprocess
import uuid
import sys
import yaml
import importlib_metadata

import pandas as pd
import pytest

import mlflow
import mlflow.pyfunc.scoring_server as pyfunc_scoring_server
import mlflow.pyfunc
from mlflow.types import DataType
from mlflow.utils.file_utils import read_yaml, write_yaml


LOCALHOST = "127.0.0.1"
_CONDA_ENV_FILE_NAME = "conda.yaml"
_REQUIREMENTS_FILE_NAME = "requirements.txt"
_CONSTRAINTS_FILE_NAME = "constraints.txt"


def shuffle_pdf(pdf):
    cols = list(pdf.columns)
    random.shuffle(cols)
    return pdf[cols]


def _strip_dev_version_suffix(version):
    return re.sub(r"(\.?)dev.*", "", version)


def _get_installed_version(package, module=None):
    """
    Obtains the installed package version using `importlib_metadata.version`. If it fails, use
    `__import__(module or package).__version__`.
    """
    try:
        version = importlib_metadata.version(package)
    except importlib_metadata.PackageNotFoundError:
        # Note `importlib_metadata.version(package)` is not necessarily equal to
        # `__import__(package).__version__`. See the example for pytorch below.
        #
        # Example
        # -------
        # $ pip install torch==1.9.0
        # $ python -c "import torch; print(torch.__version__)"
        # 1.9.0+cu102
        # $ python -c "import importlib_metadata; print(importlib_metadata.version('torch'))"
        # 1.9.0
        version = __import__(module or package).__version__

    # In Databricks, strip a dev version suffix for pyspark (e.g. '3.1.2.dev0' -> '3.1.2')
    # and make it installable from PyPI.
    if package == "pyspark" and is_in_databricks_runtime():
        version = _strip_dev_version_suffix(version)

    return version


# TODO: This would be much simpler if artifact_repo.download_artifacts could take the absolute path
# or no path.
def _download_artifact_from_uri(artifact_uri, output_path=None):
    """
    :param artifact_uri: The *absolute* URI of the artifact to download.
    :param output_path: The local filesystem path to which to download the artifact. If unspecified,
                        a local output path will be created.
    """
    if os.path.exists(artifact_uri):
        artifact_uri = path_to_local_file_uri(artifact_uri)
    parsed_uri = urllib.parse.urlparse(str(artifact_uri))
    prefix = ""
    if parsed_uri.scheme and not parsed_uri.path.startswith("/"):
        # relative path is a special case, urllib does not reconstruct it properly
        prefix = parsed_uri.scheme + ":"
        parsed_uri = parsed_uri._replace(scheme="")

    # For models:/ URIs, it doesn't make sense to initialize a ModelsArtifactRepository with only
    # the model name portion of the URI, then call download_artifacts with the version info.
    if ModelsArtifactRepository.is_models_uri(artifact_uri):
        root_uri = artifact_uri
        artifact_path = ""
    else:
        artifact_path = posixpath.basename(parsed_uri.path)
        parsed_uri = parsed_uri._replace(
            path=posixpath.dirname(parsed_uri.path))
        root_uri = prefix + urllib.parse.urlunparse(parsed_uri)

    return get_artifact_repository(artifact_uri=root_uri).download_artifacts(
        artifact_path=artifact_path, dst_path=output_path
    )


def _get_pip_deps(conda_env):
    """
    :return: The pip dependencies from the conda env
    """
    if conda_env is not None:
        for dep in conda_env["dependencies"]:
            if _is_pip_deps(dep):
                return dep["pip"]
    return []


def get_safe_port():
    """Returns an ephemeral port that is very likely to be free to bind to."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((LOCALHOST, 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def random_int(lo=1, hi=1e10):
    return random.randint(lo, hi)


def random_str(size=10, chars=string.ascii_uppercase + string.digits):
    return "".join(random.choice(chars) for _ in range(size))


def random_file(ext):
    return "temp_test_%d.%s" % (random_int(), ext)


def score_model_in_sagemaker_docker_container(
    model_uri,
    data,
    content_type,
    flavor=mlflow.pyfunc.FLAVOR_NAME,
    activity_polling_timeout_seconds=500,
):
    """
    :param model_uri: URI to the model to be served.
    :param data: The data to send to the docker container for testing. This is either a
                 Pandas dataframe or string of the format specified by `content_type`.
    :param content_type: The type of the data to send to the docker container for testing. This is
                         one of `mlflow.pyfunc.scoring_server.CONTENT_TYPES`.
    :param flavor: Model flavor to be deployed.
    :param activity_polling_timeout_seconds: The amount of time, in seconds, to wait before
                                             declaring the scoring process to have failed.
    """
    env = dict(os.environ)
    env.update(LC_ALL="en_US.UTF-8", LANG="en_US.UTF-8")
    proc = _start_scoring_proc(
        cmd=["mlflow", "sagemaker", "run-local", "-m",
             model_uri, "-p", "5000", "-f", flavor],
        env=env,
    )
    return _evaluate_scoring_proc(proc, 5000, data, content_type, activity_polling_timeout_seconds)


def pyfunc_build_image(model_uri, extra_args=None):
    """
    Builds a docker image containing the specified model, returning the name of the image.
    :param model_uri: URI of model, e.g. runs:/some-run-id/run-relative/path/to/model
    :param extra_args: List of extra args to pass to `mlflow models build-docker` command
    """
    name = uuid.uuid4().hex
    cmd = ["mlflow", "models", "build-docker", "-m", model_uri, "-n", name]
    if extra_args:
        cmd += extra_args
    p = subprocess.Popen(cmd,)
    assert p.wait() == 0, "Failed to build docker image to serve model from %s" % model_uri
    return name


def pyfunc_serve_from_docker_image(image_name, host_port, extra_args=None):
    """
    Serves a model from a docker container, exposing it as an endpoint at the specified port
    on the host machine. Returns a handle (Popen object) to the server process.
    """
    env = dict(os.environ)
    env.update(LC_ALL="en_US.UTF-8", LANG="en_US.UTF-8")
    scoring_cmd = ["docker", "run", "-p", "%s:8080" % host_port, image_name]
    if extra_args is not None:
        scoring_cmd += extra_args
    return _start_scoring_proc(cmd=scoring_cmd, env=env)


def pyfunc_serve_from_docker_image_with_env_override(
    image_name, host_port, gunicorn_opts, extra_args=None
):
    """
    Serves a model from a docker container, exposing it as an endpoint at the specified port
    on the host machine. Returns a handle (Popen object) to the server process.
    """
    env = dict(os.environ)
    env.update(LC_ALL="en_US.UTF-8", LANG="en_US.UTF-8")
    scoring_cmd = [
        "docker",
        "run",
        "-e",
        "GUNICORN_CMD_ARGS=%s" % gunicorn_opts,
        "-p",
        "%s:8080" % host_port,
        image_name,
    ]
    if extra_args is not None:
        scoring_cmd += extra_args
    return _start_scoring_proc(cmd=scoring_cmd, env=env)


def exec_prepare_model(model_uri, stdout=sys.stdout):
    env = dict(os.environ)
    env.update(LC_ALL="en_US.UTF-8", LANG="en_US.UTF-8")
    env.update(MLFLOW_TRACKING_URI=mlflow.get_tracking_uri())
    env.update(MLFLOW_HOME=_get_mlflow_home())
    env.update(MLFLOW_FASTAPI_HOME=os.getcwd())

    prepare_cmd = [
        "mlflow_fastapi",
        "prepare-model",
        "-m",
        model_uri
    ]
    _start_scoring_proc(cmd=prepare_cmd, env=env, stdout=stdout, stderr=stdout)


def pyfunc_serve_and_score_model(
    model_uri,
    data,
    content_type,
    activity_polling_timeout_seconds=500,
    extra_args=None,
    stdout=sys.stdout,
):
    """
    :param model_uri: URI to the model to be served.
    :param data: The data to send to the pyfunc server for testing. This is either a
                 Pandas dataframe or string of the format specified by `content_type`.
    :param content_type: The type of the data to send to the pyfunc server for testing. This is
                         one of `mlflow.pyfunc.scoring_server.CONTENT_TYPES`.
    :param activity_polling_timeout_seconds: The amount of time, in seconds, to wait before
                                             declaring the scoring process to have failed.
    :param extra_args: A list of extra arguments to pass to the pyfunc scoring server command. For
                       example, passing ``extra_args=["--no-conda"]`` will pass the ``--no-conda``
                       flag to the scoring server to ensure that conda environment activation
                       is skipped.
    """
    env = dict(os.environ)
    env.update(LC_ALL="en_US.UTF-8", LANG="en_US.UTF-8")
    env.update(MLFLOW_TRACKING_URI=mlflow.get_tracking_uri())
    env.update(MLFLOW_HOME=_get_mlflow_home())
    env.update(MLFLOW_FASTAPI_HOME=os.getcwd())
    port = get_safe_port()

    scoring_cmd = [
        "mlflow_fastapi",
        "serve-model",
        "-p",
        str(port),
    ]
    if extra_args is not None:
        scoring_cmd += extra_args
    proc = _start_scoring_proc(
        cmd=scoring_cmd, env=env, stdout=stdout, stderr=stdout)
    return _evaluate_scoring_proc(proc, port, data, content_type, activity_polling_timeout_seconds)


def _get_mlflow_home():
    """
    :return: The path to the MLflow installation root directory
    """
    mlflow_module_path = os.path.dirname(os.path.abspath(mlflow.__file__))
    # The MLflow root directory is one level about the mlflow module location
    return os.path.join(mlflow_module_path, os.pardir)


def _start_scoring_proc(cmd, env, stdout=sys.stdout, stderr=sys.stderr):
    if os.name != "nt":
        return subprocess.Popen(
            cmd,
            stdout=stdout,
            stderr=stderr,
            universal_newlines=True,
            env=env,
            # Assign the scoring process to a process group. All child processes of the
            # scoring process will be assigned to this group as well. This allows child
            # processes of the scoring process to be terminated successfully
            preexec_fn=os.setsid,
        )
    else:
        return subprocess.Popen(
            cmd,
            stdout=stdout,
            stderr=stderr,
            universal_newlines=True,
            env=env,
            # On Windows, `os.setsid` and `preexec_fn` are unavailable
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )


class RestEndpoint:
    def __init__(self, proc, port, activity_polling_timeout_seconds=250):
        self._proc = proc
        self._port = port
        self._activity_polling_timeout_seconds = activity_polling_timeout_seconds

    def __enter__(self):
        for i in range(0, int(self._activity_polling_timeout_seconds / 5)):
            assert self._proc.poll() is None, "scoring process died"
            time.sleep(5)
            # noinspection PyBroadException
            try:
                ping_status = requests.get(
                    url="http://localhost:%d/ping" % self._port)
                print("connection attempt", i,
                      "server is up! ping status", ping_status)
                if ping_status.status_code == 200:
                    break
            except Exception:
                print("connection attempt", i, "failed, server is not up yet")
        if ping_status.status_code != 200:
            raise Exception("ping failed, server is not happy")
        print("server up, ping status", ping_status)
        return self

    def __exit__(self, tp, val, traceback):
        if self._proc.poll() is None:
            # Terminate the process group containing the scoring process.
            # This will terminate all child processes of the scoring process
            if os.name != "nt":
                pgrp = os.getpgid(self._proc.pid)
                os.killpg(pgrp, signal.SIGTERM)
            else:
                # https://stackoverflow.com/questions/47016723/windows-equivalent-for-spawning-and-killing-separate-process-group-in-python-3  # noqa
                self._proc.send_signal(signal.CTRL_BREAK_EVENT)
                self._proc.kill()

    def invoke(self, data, content_type):
        if type(data) == pd.DataFrame:
            if content_type == pyfunc_scoring_server.CONTENT_TYPE_JSON_RECORDS_ORIENTED:
                data = data.to_json(orient="records")
            elif (
                content_type == pyfunc_scoring_server.CONTENT_TYPE_JSON
                or content_type == pyfunc_scoring_server.CONTENT_TYPE_JSON_SPLIT_ORIENTED
            ):
                data = data.to_json(orient="split")
            elif content_type == pyfunc_scoring_server.CONTENT_TYPE_CSV:
                data = data.to_csv(index=False)
            else:
                raise Exception(
                    "Unexpected content type for Pandas dataframe input %s" % content_type
                )
        response = requests.post(
            url="http://localhost:%d/invocations" % self._port,
            data=data,
            headers={"Content-Type": content_type},
        )
        return response


def _evaluate_scoring_proc(proc, port, data, content_type, activity_polling_timeout_seconds=250):
    """
    :param activity_polling_timeout_seconds: The amount of time, in seconds, to wait before
                                             declaring the scoring process to have failed.
    """
    with RestEndpoint(proc, port, activity_polling_timeout_seconds) as endpoint:
        return endpoint.invoke(data, content_type)


@pytest.fixture
def pandas_df_with_all_types():
    pdf = pd.DataFrame(
        {
            "boolean": [True, False, True],
            "integer": np.array([1, 2, 3], np.int32),
            "long": np.array([1, 2, 3], np.int64),
            "float": np.array([math.pi, 2 * math.pi, 3 * math.pi], np.float32),
            "double": [math.pi, 2 * math.pi, 3 * math.pi],
            "binary": [bytearray([1, 2, 3]), bytearray([4, 5, 6]), bytearray([7, 8, 9])],
            "datetime": [
                np.datetime64("2021-01-01 00:00:00"),
                np.datetime64("2021-02-02 00:00:00"),
                np.datetime64("2021-03-03 12:00:00"),
            ],
        }
    )
    pdf["string"] = pd.Series(
        ["a", "b", "c"], dtype=DataType.string.to_pandas())
    return pdf


@pytest.fixture(scope="module", autouse=True)
def set_boto_credentials():
    os.environ["AWS_ACCESS_KEY_ID"] = "NotARealAccessKey"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "NotARealSecretAccessKey"
    os.environ["AWS_SESSION_TOKEN"] = "NotARealSessionToken"


@pytest.fixture
def mock_s3_bucket():
    """
    Creates a mock S3 bucket using moto

    :return: The name of the mock bucket
    """
    import boto3
    import moto

    with moto.mock_s3():
        bucket_name = "mock-bucket"
        s3_client = boto3.client("s3")
        s3_client.create_bucket(Bucket=bucket_name)
        yield bucket_name


class safe_edit_yaml(object):
    def __init__(self, root, file_name, edit_func):
        self._root = root
        self._file_name = file_name
        self._edit_func = edit_func
        self._original = read_yaml(root, file_name)

    def __enter__(self):
        new_dict = self._edit_func(self._original.copy())
        write_yaml(self._root, self._file_name, new_dict, overwrite=True)

    def __exit__(self, *args):
        write_yaml(self._root, self._file_name, self._original, overwrite=True)


def create_mock_response(status_code, text):
    """
    Create a mock resposne object with the status_code and text

    :param: status_code int HTTP status code
    :param: text message from the response
    :reutrn: mock HTTP Response
    """
    response = mock.MagicMock()
    response.status_code = status_code
    response.text = text
    return response


def _read_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _read_lines(path):
    with open(path, "r") as f:
        return f.read().splitlines()


def _compare_conda_env_requirements(env_path, req_path):
    assert os.path.exists(req_path)
    custom_env_parsed = _read_yaml(env_path)
    requirements = _read_lines(req_path)
    assert _get_pip_deps(custom_env_parsed) == requirements


# TODO: Set `strict` to True for `pip_requirements` tests to strictly check its behavior.
def _assert_pip_requirements(model_uri, requirements, constraints=None, strict=False):
    """
    Loads the pip requirements (and optionally constraints) from `model_uri` and compares them
    to `requirements` (and `constraints`).

    If `strict` is True, evaluate `set(requirements) == set(loaded_requirements)`.
    Otherwise, evaluate `set(requirements) <= set(loaded_requirements)`.
    """
    local_path = _download_artifact_from_uri(model_uri)
    txt_reqs = _read_lines(os.path.join(local_path, _REQUIREMENTS_FILE_NAME))
    conda_reqs = _get_pip_deps(_read_yaml(
        os.path.join(local_path, _CONDA_ENV_FILE_NAME)))
    compare_func = set.__eq__ if strict else set.__le__
    requirements = set(requirements)
    assert compare_func(requirements, set(txt_reqs))
    assert compare_func(requirements, set(conda_reqs))

    if constraints is not None:
        assert f"-c {_CONSTRAINTS_FILE_NAME}" in txt_reqs
        assert f"-c {_CONSTRAINTS_FILE_NAME}" in conda_reqs
        cons = _read_lines(os.path.join(local_path, _CONSTRAINTS_FILE_NAME))
        assert compare_func(set(constraints), set(cons))


def _is_available_on_pypi(package, version=None, module=None):
    """
    Returns True if the specified package version is available on PyPI.

    :param package: The name of the package.
    :param version: The version of the package. If None, defaults to the installed version.
    :param module: The name of the top-level module provided by the package . For example,
                   if `package` is 'scikit-learn', `module` should be 'sklearn'. If None, defaults
                   to `package`.
    """
    resp = requests.get("https://pypi.python.org/pypi/{}/json".format(package))
    if not resp.ok:
        return False

    version = version or _get_installed_version(module or package)
    dist_files = resp.json()["releases"].get(version)
    return (
        dist_files is not None  # specified version exists
        and (len(dist_files) > 0)  # at least one distribution file exists
        # specified version is not yanked
        and not dist_files[0].get("yanked", False)
    )


def _is_importable(module_name):
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False


def disable_prevent_infer_pip_requirements_fallback_if(condition):
    def decorator(f):
        return pytest.mark.disable_prevent_infer_pip_requirements_fallback(f) if condition else f

    return decorator
