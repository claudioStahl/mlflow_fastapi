import json
import numpy as np
import os
import pandas as pd
from collections import namedtuple

from keras.models import Model
from keras.layers import Dense, Input, Concatenate
from keras.optimizers import SGD
import pytest
import sklearn.datasets as datasets
import sklearn.neighbors as knn

import mlflow.pyfunc.scoring_server as pyfunc_scoring_server
import mlflow.sklearn
from mlflow.models import ModelSignature
from mlflow.pyfunc import PythonModel
from mlflow.types import Schema, ColSpec
from mlflow.utils.proto_json_utils import NumpyEncoder

from tests.helper_functions import pyfunc_serve_and_score_model, exec_prepare_model, shuffle_pdf, pandas_df_with_all_types

ModelWithData = namedtuple("ModelWithData", ["model", "inference_data"])


@pytest.fixture(scope="session")
def sklearn_model():
    iris = datasets.load_iris()
    X = iris.data[:, :2]  # we only take the first two features.
    y = iris.target
    knn_model = knn.KNeighborsClassifier()
    knn_model.fit(X, y)
    return ModelWithData(model=knn_model, inference_data=X)


@pytest.fixture(scope="session")
def sklearn_model_resp():
    return [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1,
        2, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 1, 2, 2,
        2, 2, 1, 1, 2, 2, 2, 2, 1, 2, 1, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 1, 2, 2,
        2, 1
    ]


@pytest.fixture(scope="session")
def keras_model():
    iris = datasets.load_iris()
    data = pd.DataFrame(
        data=np.c_[iris["data"], iris["target"]
                   ], columns=iris["feature_names"] + ["target"]
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


def test_scoring_server_responds_to_invalid_json_input_with_stacktrace_and_error_code(
    sklearn_model, model_path
):
    mlflow.sklearn.save_model(sk_model=sklearn_model.model, path=model_path)

    model_uri = os.path.abspath(model_path)
    exec_prepare_model(model_uri)

    incorrect_json_content = json.dumps({"not": "a serialized dataframe"})
    response = pyfunc_serve_and_score_model(
        model_uri,
        data=incorrect_json_content,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON_SPLIT_ORIENTED,
    )
    assert response.status_code == 400
    response_json = json.loads(response.content)
    assert "detail" in response_json

    incorrect_json_content = json.dumps("not a dict or a list")
    response = pyfunc_serve_and_score_model(
        model_uri,
        data=incorrect_json_content,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
    )
    assert response.status_code == 400
    response_json = json.loads(response.content)
    assert "detail" in response_json


def test_scoring_server_responds_to_malformed_json_input_with_stacktrace_and_error_code(
    sklearn_model, model_path
):
    mlflow.sklearn.save_model(sk_model=sklearn_model.model, path=model_path)

    model_uri = os.path.abspath(model_path)
    exec_prepare_model(model_uri)

    malformed_json_content = "this is,,,, not valid json"
    response = pyfunc_serve_and_score_model(
        model_uri,
        data=malformed_json_content,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON_SPLIT_ORIENTED,
    )
    assert response.status_code == 400
    response_json = json.loads(response.content)
    assert "detail" in response_json


def test_scoring_server_responds_to_invalid_pandas_input_format_with_stacktrace_and_error_code(
    sklearn_model, model_path
):
    mlflow.sklearn.save_model(sk_model=sklearn_model.model, path=model_path)

    model_uri = os.path.abspath(model_path)
    exec_prepare_model(model_uri)

    # The pyfunc scoring server expects a serialized Pandas Dataframe in `split` or `records`
    # format; passing a serialized Dataframe in `table` format should yield a readable error
    pandas_table_content = pd.DataFrame(
        sklearn_model.inference_data).to_json(orient="table")
    response = pyfunc_serve_and_score_model(
        model_uri,
        data=pandas_table_content,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON_SPLIT_ORIENTED,
    )
    assert response.status_code == 400
    response_json = json.loads(response.content)
    assert "detail" in response_json


def test_scoring_server_responds_to_incompatible_inference_dataframe_with_stacktrace_and_error_code(
    sklearn_model, model_path
):
    mlflow.sklearn.save_model(sk_model=sklearn_model.model, path=model_path)
    incompatible_df = pd.DataFrame(np.array(range(10)))

    model_uri = os.path.abspath(model_path)
    exec_prepare_model(model_uri)

    response = pyfunc_serve_and_score_model(
        model_uri,
        data=incompatible_df,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON_SPLIT_ORIENTED,
    )
    assert response.status_code == 422
    response_json = json.loads(response.content)
    assert "detail" in response_json


def test_scoring_server_successfully_evaluates_correct_dataframes_with_pandas_records_orientation(
    sklearn_model, sklearn_model_resp, model_path
):
    mlflow.sklearn.save_model(sk_model=sklearn_model.model, path=model_path)

    model_uri = os.path.abspath(model_path)
    exec_prepare_model(model_uri)

    pandas_record_content = pd.DataFrame(
        sklearn_model.inference_data).to_json(orient="records")
    response_records_content_type = pyfunc_serve_and_score_model(
        model_uri,
        data=pandas_record_content,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
    )
    assert response_records_content_type.status_code == 200
    assert response_records_content_type.json() == sklearn_model_resp

    response_records_content_type = pyfunc_serve_and_score_model(
        model_uri,
        data=pandas_record_content,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON_RECORDS_ORIENTED,
    )
    assert response_records_content_type.status_code == 200
    assert response_records_content_type.json() == sklearn_model_resp


def test_scoring_server_successfully_evaluates_correct_dataframes_with_pandas_split_orientation(
    sklearn_model, sklearn_model_resp, model_path
):
    mlflow.sklearn.save_model(sk_model=sklearn_model.model, path=model_path)

    model_uri = os.path.abspath(model_path)
    exec_prepare_model(model_uri)

    pandas_split_content = pd.DataFrame(
        sklearn_model.inference_data).to_json(orient="split")

    response_default_content_type = pyfunc_serve_and_score_model(
        model_uri,
        data=pandas_split_content,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
    )
    assert response_default_content_type.status_code == 200
    assert response_default_content_type.json() == sklearn_model_resp

    response = pyfunc_serve_and_score_model(
        model_uri,
        data=pandas_split_content,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON_SPLIT_ORIENTED,
    )
    assert response.status_code == 200
    assert response.json() == sklearn_model_resp


def test_scoring_server_successfully_evaluates_correct_split_to_numpy(
        sklearn_model, sklearn_model_resp, model_path
):
    mlflow.sklearn.save_model(sk_model=sklearn_model.model, path=model_path)

    model_uri = os.path.abspath(model_path)
    exec_prepare_model(model_uri)

    pandas_split_content = pd.DataFrame(
        sklearn_model.inference_data).to_json(orient="split")
    response_records_content_type = pyfunc_serve_and_score_model(
        model_uri,
        data=pandas_split_content,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON_SPLIT_NUMPY,
    )
    assert response_records_content_type.status_code == 200
    assert response_records_content_type.json() == sklearn_model_resp


def test_scoring_server_responds_to_invalid_content_type_request_with_unsupported_content_type_code(
    sklearn_model, model_path
):
    mlflow.sklearn.save_model(sk_model=sklearn_model.model, path=model_path)

    model_uri = os.path.abspath(model_path)
    exec_prepare_model(model_uri)

    pandas_split_content = pd.DataFrame(
        sklearn_model.inference_data).to_json(orient="split")
    response = pyfunc_serve_and_score_model(
        model_uri,
        data=pandas_split_content,
        content_type="not_a_supported_content_type",
    )
    assert response.status_code == 415


def test_scoring_server_successfully_evaluates_correct_tf_serving_sklearn(
    sklearn_model, sklearn_model_resp, model_path
):
    mlflow.sklearn.save_model(sk_model=sklearn_model.model, path=model_path)

    model_uri = os.path.abspath(model_path)
    exec_prepare_model(model_uri)

    inp_dict = {"instances": sklearn_model.inference_data.tolist()}
    response_records_content_type = pyfunc_serve_and_score_model(
        model_uri,
        data=json.dumps(inp_dict),
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
    )
    assert response_records_content_type.status_code == 200
    assert response_records_content_type.json() == sklearn_model_resp


def test_scoring_server_successfully_evaluates_correct_tf_serving_keras_instances(
    keras_model, model_path
):
    mlflow.keras.save_model(keras_model.model, model_path)

    model_uri = os.path.abspath(model_path)
    exec_prepare_model(model_uri)

    inp_dict = {
        "instances": [
            {"a": a.tolist(), "b": b.tolist()}
            for (a, b) in zip(keras_model.inference_data[:, :2], keras_model.inference_data[:, -2:])
        ]
    }
    response = pyfunc_serve_and_score_model(
        model_uri,
        data=json.dumps(inp_dict),
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
    )
    assert response.status_code == 200

    response_json = response.json()
    assert len(response_json) == 150
    assert len(response_json[0]) == 1


def test_scoring_server_successfully_evaluates_correct_tf_serving_keras_inputs(
    keras_model, model_path
):
    mlflow.keras.save_model(keras_model.model, model_path)

    model_uri = os.path.abspath(model_path)
    exec_prepare_model(model_uri)

    inp_dict = {
        "inputs": {
            "a": keras_model.inference_data[:, :2].tolist(),
            "b": keras_model.inference_data[:, -2:].tolist(),
        }
    }
    response = pyfunc_serve_and_score_model(
        model_uri,
        data=json.dumps(inp_dict),
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
    )
    assert response.status_code == 200

    response_json = response.json()
    assert len(response_json) == 150
    assert len(response_json[0]) == 1


def test_serving_model_with_schema(pandas_df_with_all_types):
    class TestModel(PythonModel):
        def predict(self, context, model_input):
            return [[k, str(v)] for k, v in model_input.dtypes.items()]

    schema = Schema([ColSpec(c, c) for c in pandas_df_with_all_types.columns])
    df = shuffle_pdf(pandas_df_with_all_types)

    with mlflow.start_run() as run:
        mlflow.pyfunc.log_model(
            "model", python_model=TestModel(), signature=ModelSignature(schema)
        )

        model_uri = "runs:/{}/model".format(run.info.run_id)
        exec_prepare_model(model_uri)

        response = pyfunc_serve_and_score_model(
            model_uri,
            data=json.dumps(df.to_dict(orient="split"), cls=NumpyEncoder),
            content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON_SPLIT_ORIENTED
        )

        response_json = json.loads(response.content)
        assert response_json == [
            [k, str(v)] for k, v in pandas_df_with_all_types.dtypes.items()]

        response = pyfunc_serve_and_score_model(
            model_uri,
            data=json.dumps(pandas_df_with_all_types.to_dict(
                orient="records"), cls=NumpyEncoder),
            content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON_RECORDS_ORIENTED
        )

        response_json = json.loads(response.content)
        assert response_json == [
            [k, str(v)] for k, v in pandas_df_with_all_types.dtypes.items()]


def test_parse_json_input_including_path():
    class TestModel(PythonModel):
        def predict(self, context, model_input):
            return 1

    with mlflow.start_run() as run:
        mlflow.pyfunc.log_model("model", python_model=TestModel())

        pandas_split_content = pd.DataFrame(
            {
                "url": ["http://foo.com", "https://bar.com"],
                "bad_protocol": ["aaa://bbb", "address:/path"],
            }
        ).to_json(orient="split")

        model_uri = "runs:/{}/model".format(run.info.run_id)
        exec_prepare_model(model_uri)

        response_records_content_type = pyfunc_serve_and_score_model(
            model_uri,
            data=pandas_split_content,
            content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        )
        assert response_records_content_type.status_code == 200
