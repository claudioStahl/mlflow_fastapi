import json
import pytest
import numpy as np
import pandas as pd
from collections import OrderedDict
from fastapi import HTTPException

from mlflow.models import infer_signature
from mlflow.types import Schema, ColSpec
from mlflow.utils.proto_json_utils import NumpyEncoder

from mlflow_fastapi.parsers import infer_and_parse_json_input, parse_json_input, parse_csv_input, parse_split_oriented_json_input_to_numpy

from tests.helper_functions import random_int, random_str, shuffle_pdf, pandas_df_with_all_types


def test_parse_json_input_records_oriented():
    size = 2
    data = {
        "col_m": [random_int(0, 1000) for _ in range(size)],
        "col_z": [random_str(4) for _ in range(size)],
        "col_a": [random_int() for _ in range(size)],
    }
    p1 = pd.DataFrame.from_dict(data)
    p2 = parse_json_input(p1.to_json(orient="records"), orient="records")
    # "records" orient may shuffle column ordering. Hence comparing each column Series
    for col in data.keys():
        assert all(p1[col] == p2[col])


def test_parse_json_input_split_oriented():
    size = 200
    data = {
        "col_m": [random_int(0, 1000) for _ in range(size)],
        "col_z": [random_str(4) for _ in range(size)],
        "col_a": [random_int() for _ in range(size)],
    }
    p1 = pd.DataFrame.from_dict(data)
    p2 = parse_json_input(p1.to_json(orient="split"), orient="split")
    assert all(p1 == p2)


def test_records_oriented_json_to_df():
    # test that datatype for "zip" column is not converted to "int64"
    jstr = (
        "["
        '{"zip":"95120","cost":10.45,"score":8},'
        '{"zip":"95128","cost":23.0,"score":0},'
        '{"zip":"95128","cost":12.1,"score":10}'
        "]"
    )
    df = parse_json_input(jstr, orient="records")

    assert set(df.columns) == {"zip", "cost", "score"}
    assert set(str(dt) for dt in df.dtypes) == {"object", "float64", "int64"}


def test_split_oriented_json_to_df():
    # test that datatype for "zip" column is not converted to "int64"
    jstr = (
        '{"columns":["zip","cost","count"],"index":[0,1,2],'
        '"data":[["95120",10.45,-8],["95128",23.0,-1],["95128",12.1,1000]]}'
    )
    df = parse_json_input(jstr, orient="split")

    assert set(df.columns) == {"zip", "cost", "count"}
    assert set(str(dt) for dt in df.dtypes) == {"object", "float64", "int64"}


def test_parse_with_schema(pandas_df_with_all_types):
    schema = Schema([ColSpec(c, c) for c in pandas_df_with_all_types.columns])
    df = shuffle_pdf(pandas_df_with_all_types)
    json_str = json.dumps(df.to_dict(orient="split"), cls=NumpyEncoder)
    df = parse_json_input(json_str, orient="split", schema=schema)
    assert schema == infer_signature(df[schema.input_names()]).inputs

    json_str = json.dumps(df.to_dict(orient="records"), cls=NumpyEncoder)
    df = parse_json_input(json_str, orient="records", schema=schema)
    assert schema == infer_signature(df[schema.input_names()]).inputs

    # The current behavior with pandas json parse with type hints is weird. In some cases, the
    # types are forced ignoting overflow and loss of precision:

    bad_df = """{
      "columns":["bad_integer", "bad_float", "bad_string", "bad_boolean"],
      "data":[
        [9007199254740991.0, 1.1,                1, 1.5],
        [9007199254740992.0, 9007199254740992.0, 2, 0],
        [9007199254740994.0, 3.3,                3, "some arbitrary string"]
      ]
    }"""
    schema = Schema(
        [
            ColSpec("integer", "bad_integer"),
            ColSpec("float", "bad_float"),
            ColSpec("float", "good_float"),
            ColSpec("string", "bad_string"),
            ColSpec("boolean", "bad_boolean"),
        ]
    )
    df = parse_json_input(bad_df, orient="split", schema=schema)

    # Unfortunately, the current behavior of pandas parse is to force numbers to int32 even if
    # they don't fit:
    assert df["bad_integer"].dtype == np.int32
    assert all(df["bad_integer"] == [-2147483648, -2147483648, -2147483648])

    # The same goes for floats:
    assert df["bad_float"].dtype == np.float32
    assert all(df["bad_float"] == np.array(
        [1.1, 9007199254740992, 3.3], dtype=np.float32))

    # However bad string is recognized as int64:
    assert all(df["bad_string"] == np.array([1, 2, 3], dtype=np.object))

    # Boolean is forced - zero and empty string is false, everything else is true:
    assert df["bad_boolean"].dtype == np.bool
    assert all(df["bad_boolean"] == [True, False, True])


def test_parse_json_input_with_invalid_values():
    with pytest.raises(HTTPException) as ex:
        parse_json_input(json.dumps('"just a string"'))
    assert ex.value.status_code == 400
    assert ex.value.detail == (
        "Failed to parse input as a Pandas DataFrame. Ensure that the input is"
        " a valid JSON-formatted Pandas DataFrame with the `{orient}` orient"
        " produced using the `pandas.DataFrame.to_json(..., orient='{orient}')`"
        " method.".format(orient="split")
    )

    jstr = ('{"zip2":1.3,"cost":12.1,"score":10}')
    with pytest.raises(HTTPException):
        print(parse_json_input(jstr, orient="records"))


    jstr = (
        '{"columns":["zip","cost","count"],"index":[0,1,2],'
        '"data":[["95120",10.45,-8, 1],["95128",23.0,-1],["95128",12.1,1000]]}'
    )
    with pytest.raises(HTTPException):
        parse_json_input(jstr, orient="split")


def test_infer_and_parse_json_input():
    size = 20
    # input is correctly recognized as list, and parsed as pd df with orient 'records'
    data = {
        "col_m": [random_int(0, 1000) for _ in range(size)],
        "col_z": [random_str(4) for _ in range(size)],
        "col_a": [random_int() for _ in range(size)],
    }
    p1 = pd.DataFrame.from_dict(data)
    p2 = infer_and_parse_json_input(p1.to_json(orient="records"))
    assert all(p1 == p2)

    # input is correctly recognized as a dict, and parsed as pd df with orient 'split'
    data = {
        "col_m": [random_int(0, 1000) for _ in range(size)],
        "col_z": [random_str(4) for _ in range(size)],
        "col_a": [random_int() for _ in range(size)],
    }
    p1 = pd.DataFrame.from_dict(data)
    p2 = infer_and_parse_json_input(p1.to_json(orient="split"))
    assert all(p1 == p2)

    # input is correctly recognized as tf serving input
    arr = [
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        [[3, 2, 1], [6, 5, 4], [9, 8, 7]],
    ]
    tfserving_input = {"instances": arr}
    result = infer_and_parse_json_input(json.dumps(tfserving_input))
    assert result.shape == (2, 3, 3)
    assert (result == np.array(arr)).all()

    # input is unrecognized JSON input
    with pytest.raises(HTTPException) as ex:
        infer_and_parse_json_input(json.dumps('"just a string"'))
    assert (
        "Failed to parse input from JSON. Ensure that input is a valid JSON"
        " list or dictionary." in str(ex)
    )

    # input is not json str
    with pytest.raises(HTTPException) as ex:
        infer_and_parse_json_input("(not a json string)")
    assert (
        "Failed to parse input from JSON. Ensure that input is a valid JSON"
        " formatted string." in str(ex)
    )


def test_split_oriented_json_to_numpy_array():
    # test that datatype for "zip" column is not converted to "int64"
    jstr = (
        '{"columns":["zip","cost","count"],"index":[0,1,2],'
        '"data":[["95120",10.45,-8],["95128",23.0,-1],["95128",12.1,1000]]}'
    )
    df = parse_split_oriented_json_input_to_numpy(jstr)

    assert set(df.columns) == {"zip", "cost", "count"}
    assert set(str(dt) for dt in df.dtypes) == {"object", "float64", "int64"}


def test_parse_json_input_split_oriented_to_numpy_array():
    size = 200
    data = OrderedDict(
        [
            ("col_m", [random_int(0, 1000) for _ in range(size)]),
            ("col_z", [random_str(4) for _ in range(size)]),
            ("col_a", [random_int() for _ in range(size)]),
        ]
    )
    p0 = pd.DataFrame.from_dict(data)
    np_array = np.array(
        [[a, b, c] for a, b, c in zip(data["col_m"], data["col_z"], data["col_a"])], dtype=object
    )
    p1 = pd.DataFrame(np_array).infer_objects()
    p2 = parse_split_oriented_json_input_to_numpy(
        p0.to_json(orient="split"))
    np.testing.assert_array_equal(p1, p2)
