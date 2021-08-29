from mlflow_fastapi.model_wrapper import parse_and_predict


def ex_parse_and_predict(content_type, body):
    return parse_and_predict(content_type.decode(), body.decode())
