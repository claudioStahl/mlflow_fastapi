from fastapi import HTTPException

from collections import OrderedDict
import json
import numpy as np
import pandas as pd

# NB: We need to be careful what we import form mlflow here. Scoring server is used from within
# model's conda environment. The version of mlflow doing the serving (outside) and the version of
# mlflow in the model's conda environment (inside) can differ. We should therefore keep mlflow
# dependencies to the minimum here.
# ALl of the mlfow dependencies below need to be backwards compatible.
from mlflow.exceptions import MlflowException
from mlflow.types import Schema
from mlflow.utils.proto_json_utils import _dataframe_from_json, parse_tf_serving_input

def infer_and_parse_json_input(json_input, schema: Schema = None):
    """
    :param json_input: A JSON-formatted string representation of TF serving input or a Pandas
                       DataFrame, or a stream containing such a string representation.
    :param schema: Optional schema specification to be used during parsing.
    """
    try:
        decoded_input = json.loads(json_input)
    except json.decoder.JSONDecodeError:
        raise HTTPException(status_code=400, detail=(
            "Failed to parse input from JSON. Ensure that input is a valid JSON"
            " formatted string."
        ))

    if isinstance(decoded_input, list):
        return parse_json_input(json_input=json_input, orient="records", schema=schema)
    elif isinstance(decoded_input, dict):
        if "instances" in decoded_input or "inputs" in decoded_input:
            try:
                return parse_tf_serving_input(decoded_input, schema=schema)
            except MlflowException as ex:
                raise HTTPException(status_code=400, detail=(ex.message))
        else:
            return parse_json_input(json_input=json_input, orient="split", schema=schema)
    else:
        raise HTTPException(status_code=400, detail=(
            "Failed to parse input from JSON. Ensure that input is a valid JSON"
            " list or dictionary."
        ))


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
        raise HTTPException(status_code=400, detail=(
            "Failed to parse input as a Pandas DataFrame. Ensure that the input is"
            " a valid JSON-formatted Pandas DataFrame with the `{orient}` orient"
            " produced using the `pandas.DataFrame.to_json(..., orient='{orient}')`"
            " method.".format(orient=orient)
        ))


def parse_csv_input(csv_input):
    """
    :param csv_input: A CSV-formatted string representation of a Pandas DataFrame, or a stream
                      containing such a string representation.
    """

    try:
        return pd.read_csv(csv_input)
    except Exception:
        raise HTTPException(status_code=400, detail=(
            "Failed to parse input as a Pandas DataFrame. Ensure that the input is"
            " a valid CSV-formatted Pandas DataFrame produced using the"
            " `pandas.DataFrame.to_csv()` method."
        ))


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
        raise HTTPException(status_code=400, detail=(
            "Failed to parse input as a Numpy. Ensure that the input is"
            " a valid JSON-formatted Pandas DataFrame with the split orient"
            " produced using the `pandas.DataFrame.to_json(..., orient='split')`"
            " method."
        ))