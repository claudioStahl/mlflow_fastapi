import logging
import os
import shutil

from mlflow.tracking.artifact_utils import _download_artifact_from_uri

_logger = logging.getLogger(__name__)

model_path = os.getcwd() + "/model"
tmp_model_path = os.getcwd() + "/tmp/model_tmp"
trash_model_path = os.getcwd() + "/tmp/model_trash"


def do_prepare_model(model_uri):
    local_path = _download_artifact_from_uri(model_uri)
    shutil.copytree(local_path, tmp_model_path)

    if os.path.exists(model_path):
        os.rename(model_path, trash_model_path)
    
    os.rename(tmp_model_path, model_path)

    if os.path.exists(trash_model_path):
        shutil.rmtree(trash_model_path)
