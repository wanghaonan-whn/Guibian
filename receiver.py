import cv2
import zmq
import numpy as np
from fastapi import FastAPI, UploadFile, File
import uvicorn

app = FastAPI()

# ZeroMQ PUSH
context = zmq.Context()
socket = context.socket(zmq.PUSH)

# worker 接收地址
socket.bind("tcp://*:5555")


@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):

    data = await file.read()

    # bytes → numpy
    numpy_array = np.frombuffer(data, np.uint8)

    # decode image
    image = cv2.imdecode(numpy_array, cv2.IMREAD_GRAYSCALE)

    if image is None:
        return {"code": -1, "msg": "decode failed"}

    # 发送给 worker
    socket.send_pyobj(image)

    return {"code": 0, "msg": "received"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)