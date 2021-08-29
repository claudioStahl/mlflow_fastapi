import logging
import os

# NB: We need to be careful what we import form mlflow here. Scoring server is used from within
# model's conda environment. The version of mlflow doing the serving (outside) and the version of
# mlflow in the model's conda environment (inside) can differ. We should therefore keep mlflow
# dependencies to the minimum here.
# ALl of the mlfow dependencies below need to be backwards compatible.
from mlflow.utils.proto_json_utils import _get_jsonable_obj

try:
    from mlflow.pyfunc import load_model
except ImportError:
    from mlflow.pyfunc import load_pyfunc as load_model

from fastapi import FastAPI, Response, Request, HTTPException
from fastapi.responses import ORJSONResponse

from mlflow.utils.file_utils import path_to_local_file_uri    

from mlflow_fastapi.parsers import infer_and_parse_json_input, parse_json_input, parse_csv_input, parse_split_oriented_json_input_to_numpy

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


app = FastAPI()        

model_path = os.getcwd() + "/model"
model_uri = path_to_local_file_uri(model_path)

model = load_model(model_uri)
input_schema = model.metadata.get_input_schema()


@app.get("/ping", response_class=ORJSONResponse)
def ping(response: Response):
    health = model is not None
    response.status_code = 200 if health else 404
    return "pong"


@app.post("/invocations", response_class=ORJSONResponse)
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
