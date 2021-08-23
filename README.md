# Requirements

Create env

    conda env create -f environment.yml
    
Active env

    conda activate mlflow_fastapi


# Development

Start MLflow

    mlflow server --backend-store-uri postgresql://postgres:postgres@localhost:5432/mlflow --default-artifact-root /home/claudio/Workplaces/Stone/mlflow/tmp/mlruns --host 0.0.0.0

Export the MLflow host

    export MLFLOW_TRACKING_URI='http://localhost:5000'

Train the model

    mlflow run git@github.com:mlflow/mlflow-example.git -P alpha=0.5

Register the created experiment as an registered model on MLflow UI. And change the state to 'Staging'.

Install project to use the CLI

    pip install --editable .

Prepare model to fastapi

    mlflow_fastapi prepare-model -m "models:/SampleModel/Staging"

Start the fastapi

    mlflow_fastapi serve-model

Call the fastapi

```sh
curl -X POST -H "Content-Type:application/json; format=pandas-split" --data '{"columns":["alcohol", "chlorides", "citric acid", "density", "fixed acidity", "free sulfur dioxide", "pH", "residual sugar", "sulphates", "total sulfur dioxide", "volatile acidity"],"data":[[12.8, 0.029, 0.48, 0.98, 6.2, 29, 3.33, 1.2, 0.39, 75, 0.66]]}' http://127.0.0.1:8000/invocations
```


# Testing

Run the pytest

    pytest

