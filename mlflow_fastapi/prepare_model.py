import logging
import os
import shutil

from mlflow.tracking.artifact_utils import _download_artifact_from_uri

_logger = logging.getLogger(__name__)

# model_uri = "models:/ElasticnetWineModel/Staging"


def do_prepare_model(model_uri):
    model_path = os.getcwd() + "/model"

    if os.path.exists(model_path):
        shutil.rmtree(model_path)

    local_path = _download_artifact_from_uri(model_uri)
    shutil.copytree(local_path, model_path)
