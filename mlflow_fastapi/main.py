from typing import Optional
from fastapi import FastAPI, Response, Request, status


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



def infer_and_parse_json_input(json_input, schema: Schema = None):
    """
    :param json_input: A JSON-formatted string representation of TF serving input or a Pandas
                       DataFrame, or a stream containing such a string representation.
    :param schema: Optional schema specification to be used during parsing.
    """
    try:
        decoded_input = json.loads(json_input)
    except json.decoder.JSONDecodeError:
        _handle_serving_error(
            error_message=(
                "Failed to parse input from JSON. Ensure that input is a valid JSON"
                " formatted string."
            ),
            error_code=MALFORMED_REQUEST,
        )

    if isinstance(decoded_input, list):
        return parse_json_input(json_input=json_input, orient="records", schema=schema)
    elif isinstance(decoded_input, dict):
        if "instances" in decoded_input or "inputs" in decoded_input:
            try:
                return parse_tf_serving_input(decoded_input, schema=schema)
            except MlflowException as ex:
                _handle_serving_error(
                    error_message=(ex.message), error_code=MALFORMED_REQUEST,
                )
        else:
            return parse_json_input(json_input=json_input, orient="split", schema=schema)
    else:
        _handle_serving_error(
            error_message=(
                "Failed to parse input from JSON. Ensure that input is a valid JSON"
                " list or dictionary."
            ),
            error_code=MALFORMED_REQUEST,
        )


def parse_json_input(json_input, orient="split", schema: Schema = None):
    """
    :param json_input: A JSON-formatted string representation of a Pandas DataFrame, or a stream
                       containing such a string representation.
    :param orient: The Pandas DataFrame orientation of the JSON input. This is either 'split'
                   or 'records'.
    :param schema: Optional schema specification to be used during parsing.
    """

    try:
        return _dataframe_from_json(json_input, pandas_orient=orient, schema=schema)
    except Exception:
        _handle_serving_error(
            error_message=(
                "Failed to parse input as a Pandas DataFrame. Ensure that the input is"
                " a valid JSON-formatted Pandas DataFrame with the `{orient}` orient"
                " produced using the `pandas.DataFrame.to_json(..., orient='{orient}')`"
                " method.".format(orient=orient)
            ),
            error_code=MALFORMED_REQUEST,
        )


def parse_csv_input(csv_input):
    """
    :param csv_input: A CSV-formatted string representation of a Pandas DataFrame, or a stream
                      containing such a string representation.
    """

    try:
        return pd.read_csv(csv_input)
    except Exception:
        _handle_serving_error(
            error_message=(
                "Failed to parse input as a Pandas DataFrame. Ensure that the input is"
                " a valid CSV-formatted Pandas DataFrame produced using the"
                " `pandas.DataFrame.to_csv()` method."
            ),
            error_code=MALFORMED_REQUEST,
        )


def parse_split_oriented_json_input_to_numpy(json_input):
    """
    :param json_input: A JSON-formatted string representation of a Pandas DataFrame with split
                       orient, or a stream containing such a string representation.
    """

    try:
        json_input_list = json.loads(json_input, object_pairs_hook=OrderedDict)
        return pd.DataFrame(
            index=json_input_list["index"],
            data=np.array(json_input_list["data"], dtype=object),
            columns=json_input_list["columns"],
        ).infer_objects()
    except Exception:
        _handle_serving_error(
            error_message=(
                "Failed to parse input as a Numpy. Ensure that the input is"
                " a valid JSON-formatted Pandas DataFrame with the split orient"
                " produced using the `pandas.DataFrame.to_json(..., orient='split')`"
                " method."
            ),
            error_code=MALFORMED_REQUEST,
        )

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
    print(content_type)
    print(body)

    if content_type == CONTENT_TYPE_CSV:
        # data = flask.request.data.decode("utf-8")
        # csv_input = StringIO(body)
        data = parse_csv_input(csv_input=body)
    elif content_type == CONTENT_TYPE_JSON:
        # json_str = flask.request.data.decode("utf-8")
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
        return {}

    raw_predictions = model.predict(data)
    predictions = _get_jsonable_obj(raw_predictions, pandas_orient="records")

    return predictions
