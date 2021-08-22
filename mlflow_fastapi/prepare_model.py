from mlflow.models import FlavorBackend
from mlflow.models.docker_utils import _build_image, DISABLE_ENV_CREATION
from mlflow.pyfunc import ENV, scoring_server

from mlflow.utils.conda import get_or_create_conda_env, get_conda_bin_executable, get_conda_command
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.file_utils import path_to_local_file_uri
from mlflow.version import VERSION


from collections import OrderedDict
import json
import logging
import numpy as np
import os
import pandas as pd
import sys
import traceback
import shutil

# NB: We need to be careful what we import form mlflow here. Scoring server is used from within
# model's conda environment. The version of mlflow doing the serving (outside) and the version of
# mlflow in the model's conda environment (inside) can differ. We should therefore keep mlflow
# dependencies to the minimum here.
# ALl of the mlfow dependencies below need to be backwards compatible.
from mlflow.exceptions import MlflowException
from mlflow.types import Schema
from mlflow.utils import reraise
from mlflow.utils.proto_json_utils import (
    NumpyEncoder,
    _dataframe_from_json,
    _get_jsonable_obj,
    parse_tf_serving_input,
)

try:
    from mlflow.pyfunc import load_model, PyFuncModel
except ImportError:
    from mlflow.pyfunc import load_pyfunc as load_model
from mlflow.protos.databricks_pb2 import MALFORMED_REQUEST, BAD_REQUEST
from mlflow.server.handlers import catch_mlflow_exception

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

_SERVER_MODEL_PATH = "__pyfunc_model_path__"

PREDICTIONS_WRAPPER_ATTR_NAME_ENV_KEY = "PREDICTIONS_WRAPPER_ATTR_NAME"

_logger = logging.getLogger(__name__)

# model_uri = "models:/ElasticnetWineModel/Staging"


def do_prepare_model(model_uri):
    model_path = os.getcwd() + "/model"
    # home_path = os.environ['MLFLOW_FASTAPI_HOME']
    # model_path = home_path + "/model"

    if os.path.exists(model_path):
        shutil.rmtree(model_path)

    local_path = _download_artifact_from_uri(model_uri)
    shutil.copytree(local_path, model_path)
