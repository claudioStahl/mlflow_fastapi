from typing import Optional
from fastapi import FastAPI, Response, Request, HTTPException, status


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

from mlflow_fastapi.parsers import infer_and_parse_json_input, parse_json_input, parse_csv_input, parse_split_oriented_json_input_to_numpy

_SERVER_MODEL_PATH = "__pyfunc_model_path__"

CONTENT_TYPE_CSV = "text/csv"
CONTENT_TYPE_JSON = "application/json"
CONTENT_TYPE_JSON_RECORDS_ORIENTED = "application/json; format=pandas-records"
CONTENT_TYPE_JSON_SPLIT_ORIENTED = "application/json; format=pandas-split"
CONTENT_TYPE_JSON_SPLIT_NUMPY = "application/json-numpy-split"

CONTENT_TYPES = [
    CONTENT_TYPE_CSV,
    CONTENT_TYPE_JSON,
    CONTENT_TYPE_JSON_RECORDS_ORIENTED,
    CONTENT_TYPE_JSON_SPLIT_ORIENTED,
    CONTENT_TYPE_JSON_SPLIT_NUMPY,
]

PREDICTIONS_WRAPPER_ATTR_NAME_ENV_KEY = "PREDICTIONS_WRAPPER_ATTR_NAME"

_logger = logging.getLogger(__name__)


def predict(model, data):
    try:
        return model.predict(data)
    except ValueError as err:
        raise HTTPException(status_code=422, detail=('{}'.format(err)))


model_path = os.getcwd() + "/model"

# NB: Absolute windows paths do not work with mlflow apis, use file uri to ensure
# platform compatibility.
model_uri = path_to_local_file_uri(model_path)
print(model_uri)

model = load_model(model_uri)
input_schema = model.metadata.get_input_schema()

app = FastAPI()


@app.get("/ping")
def ping(response: Response):
    health = model is not None
    response.status_code = 200 if health else 404
    return "pong"


@app.post("/invocations")
async def invocations(request: Request):
    content_type = request.headers['content-type']
    body = await request.body()

    if content_type == CONTENT_TYPE_CSV:
        data = parse_csv_input(csv_input=body)
    elif content_type == CONTENT_TYPE_JSON:
        data = infer_and_parse_json_input(body, input_schema)
    elif content_type == CONTENT_TYPE_JSON_SPLIT_ORIENTED:
        data = parse_json_input(
            json_input=body,
            orient="split",
            schema=input_schema,
        )
    elif content_type == CONTENT_TYPE_JSON_RECORDS_ORIENTED:
        data = parse_json_input(
            json_input=body,
            orient="records",
            schema=input_schema,
        )
    elif content_type == CONTENT_TYPE_JSON_SPLIT_NUMPY:
        data = parse_split_oriented_json_input_to_numpy(body)
    else:
        raise HTTPException(status_code=415, detail=(
            "This predictor only supports the following content types,"
            " {supported_content_types}. Got '{received_content_type}'.".format(
                supported_content_types=CONTENT_TYPES,
                received_content_type=content_type,
            )
        ))

    raw_predictions = predict(model, data)
    predictions = _get_jsonable_obj(raw_predictions, pandas_orient="records")

    return predictions
