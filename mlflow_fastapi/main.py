import logging

from fastapi import FastAPI, WebSocket, Response, Request, HTTPException
from fastapi.responses import ORJSONResponse

from mlflow_fastapi.model_wrapper import parse_and_predict, model

_logger = logging.getLogger(__name__)

app = FastAPI()


@app.get("/ping", response_class=ORJSONResponse)
def ping(response: Response):
    health = model is not None
    response.status_code = 200 if health else 404
    return "pong"


@app.post("/invocations", response_class=ORJSONResponse)
async def invocations(request: Request):
    content_type = request.headers['content-type']
    body = await request.body()
    return parse_and_predict(content_type, body)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        received = await websocket.receive_json()
        result = parse_and_predict(received['type'], received['content'])
        await websocket.send_json(result)
