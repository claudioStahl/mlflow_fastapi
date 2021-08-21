import json
import math
import numpy as np
import os
import pandas as pd
from collections import namedtuple, OrderedDict

from keras.models import Model
from keras.layers import Dense, Input, Concatenate
from keras.optimizers import SGD
import pytest
import random
import sklearn.datasets as datasets
import sklearn.neighbors as knn

from mlflow.exceptions import MlflowException
import mlflow.pyfunc.scoring_server as pyfunc_scoring_server
import mlflow.sklearn
from mlflow.models import ModelSignature, infer_signature
from mlflow.protos.databricks_pb2 import ErrorCode, MALFORMED_REQUEST, BAD_REQUEST
from mlflow.pyfunc import PythonModel
from mlflow.types import Schema, ColSpec, DataType
from mlflow.utils.file_utils import TempDir
from mlflow.utils.proto_json_utils import NumpyEncoder

from fastapi import FastAPI
from fastapi.testclient import TestClient
from mlflow_fastapi.main import app

from tests.helper_functions import pyfunc_serve_and_score_model, random_int, random_str

ModelWithData = namedtuple("ModelWithData", ["model", "inference_data"])

client = TestClient(app)


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
    pdf["string"] = pd.Series(["a", "b", "c"], dtype=DataType.string.to_pandas())
    return pdf


@pytest.fixture(scope="session")
def sklearn_model():
    iris = datasets.load_iris()
    X = iris.data[:, :2]  # we only take the first two features.
    y = iris.target
    knn_model = knn.KNeighborsClassifier()
    knn_model.fit(X, y)
    return ModelWithData(model=knn_model, inference_data=X)


@pytest.fixture(scope="session")
def keras_model():
    iris = datasets.load_iris()
    data = pd.DataFrame(
        data=np.c_[iris["data"], iris["target"]], columns=iris["feature_names"] + ["target"]
    )
    y = data["target"]
    X = data.drop("target", axis=1).values
    input_a = Input(shape=(2,), name="a")
    input_b = Input(shape=(2,), name="b")
    output = Dense(1)(Dense(3, input_dim=4)(Concatenate()([input_a, input_b])))
    model = Model(inputs=[input_a, input_b], outputs=output)
    model.compile(loss="mean_squared_error", optimizer=SGD())
    model.fit([X[:, :2], X[:, -2:]], y)
    return ModelWithData(model=model, inference_data=X)


@pytest.fixture
def model_path(tmpdir):
    return str(os.path.join(tmpdir.strpath, "model"))


def test_read_main():
    response = client.get("/ping")
    assert response.status_code == 200
    assert response.json() == "pong"
