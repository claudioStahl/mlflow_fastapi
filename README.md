# Install env

```sh
conda env create -f environment.yml
conda activate mlflow_fastapi
```

# Update env

```sh
conda env update -f environment.yml
```

# Dev

```sh
export MLFLOW_TRACKING_URI='http://localhost:5000'
uvicorn mlflow_fastapi.main:app --reload
```

# Curl

```sh
curl -X POST -H "Content-Type:application/json; format=pandas-split" --data '{"columns":["alcohol", "chlorides", "citric acid", "density", "fixed acidity", "free sulfur dioxide", "pH", "residual sugar", "sulphates", "total sulfur dioxide", "volatile acidity"],"data":[[12.8, 0.029, 0.48, 0.98, 6.2, 29, 3.33, 1.2, 0.39, 75, 0.66]]}' http://127.0.0.1:8000/invocations
```
