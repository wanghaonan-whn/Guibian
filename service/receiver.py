import os

import zmq
import uvicorn
import toml
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from pathlib import Path

app = FastAPI()
project_root = os.path.dirname(os.path.dirname(__file__))
config = toml.load(os.path.join(project_root, "config/config.toml"))

@app.post("/upload")
async def upload_image(request: Request, file: UploadFile = File(...), path: str = Form(...)):
    socket = getattr(request.app.state, "socket", None)
    if socket is None:
        raise HTTPException(status_code=503, detail="socket not initialized")

    data = await file.read()

    suffix = Path(path).suffix.lower()
    if suffix == ".txt":
        txt_save_path = Path(os.path.join(config["path"]["results_path"], path))
        txt_save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(txt_save_path, "wb") as f:
            f.write(data)
        return {"code": 0, "msg": "txt saved"}

    # image send to ZMQ
    msg = {"data": data, "path": path}
    socket.send_pyobj(msg)

    return {"code": 0, "msg": "received"}


def run_receiver():
    # ZeroMQ PUSH
    context = zmq.Context()
    socket = context.socket(zmq.PUSH)

    # worker 接收地址
    socket.bind("tcp://*:5555")
    app.state.socket = socket
    app.state.zmq_context = context

    uvicorn.run(app, host="0.0.0.0", port=8000, access_log=False)
