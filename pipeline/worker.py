import zmq
import cv2
import numpy as np
from splitter import split_exposure


def worker():

    context = zmq.Context()

    socket = context.socket(zmq.PULL)
    socket.connect("tcp://localhost:5555")

    print("Worker started...")

    while True:

        image = socket.recv_pyobj()

        high, low = split_exposure(image)

        print("receive image:", image.shape)
        print("high exposure:", high.shape)
        print("low exposure:", low.shape)

        # TODO
        # 在这里调用你的算法
        # algorithm(high, low)