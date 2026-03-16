from typing import List, Tuple

import torch
import zmq
import toml
from loguru import logger
from mef.model import E2EMEF
from mef.select_device import DeviceManager
from pipeline.splitter import split_exposure
from pipeline.warmup import warmup_shapes_u8

DEVICE_TYPE_MAP = {
    "nvidia": "cuda",
    "ascend": "npu",
}


def worker(config_path: str, device_id: str, rank: int) -> None:
    config = toml.load(config_path)
    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.connect("tcp://localhost:5555")

    logger.info("Worker started, waiting for images...")

    # select device
    device_type = DEVICE_TYPE_MAP[config["device_type"]]
    device_manager = DeviceManager(
        device_config=device_id,
        backend=device_type
    )
    device, device_str = device_manager.setup_device()
    logger.info(f"Worker {rank} using device {device_str}")

    # init model
    model = E2EMEF(config=config, is_guided=config["model"]["is_guided"])
    model.load_checkpoint(config["model"]["checkpoint"])
    model = model.to(device)
    model.eval()
    torch.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = True

    # -------- warmup --------
    try:
        warmup_shapes = config.get("warmup_shapes", [[2000, 2048]])
        warmup_shapes = [tuple(s) for s in warmup_shapes]
        warmup_shapes_u8(
            model,
            device,
            shapes=warmup_shapes,
            low_size=config["model"]["low_size"],
            iters=5,
            logger=logger
        )
    except Exception as e:
        logger.warning("[warmup] failed: %s", e)

    # init camera
    flip_cams = [x.strip() for x in config["camera"]["flip_cams"].split(",") if x.strip()]

    while True:
        image = socket.recv_pyobj()
        high, low = split_exposure(image)
