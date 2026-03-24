import os
import cv2
import numpy as np
import torch
import zmq
import toml
from PIL import Image
from loguru import logger

from mef.datasets import preprocess_mefnet, YCbCrToRGB, tensor_to_uint8_hwc, resize_by_height
from mef.model import E2EMEF
from mef.select_device import DeviceManager
from pipeline.warmup import warmup_shapes_u8

DEVICE_TYPE_MAP = {
    "nvidia": "gpu",
    "ascend": "npu",
}


def worker(config_path: str, device_id: str, rank: int) -> None:
    config = toml.load(config_path)

    context = zmq.Context().instance()
    socket = context.socket(zmq.PULL)
    socket.setsockopt(zmq.RCVHWM, 5000)
    socket.connect("tcp://localhost:5555")
    logger.info(f"Worker {rank} started, waiting for images...")

    # select device
    device_type = DEVICE_TYPE_MAP[config["model"]["device_type"]]
    device_manager = DeviceManager(device_config=device_id, backend=device_type)

    device, device_str = device_manager.init_backend().setup_device(rank)
    logger.info(f"Worker {rank} using device {device_str}")

    # init model
    model = E2EMEF(config=config, is_guided=config["model"]["is_guided"])
    model.load_checkpoint(config["model"]["checkpoint"])
    # print(device)
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
            device=device,
            shapes_hw=warmup_shapes,
            low_size=config["model"]["low_size"],
            iters=5,
            device_manager=device_manager,
            rank=rank,
        )
    except Exception as e:
        logger.warning(f"[warmup] failed: {e}")

    # init camera
    flip_cams = [str(x) for x in config["camera"]["flip_cams"]]

    while True:
        message = socket.recv_pyobj()

        data = message["data"]
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
        path = message["path"]

        I_he, I_le, Cb_f, Cr_f = preprocess_mefnet(image, device)

        O_he, W_he = model(I_le, I_he)
        O_hr_RGB = YCbCrToRGB()(torch.cat((O_he.detach().cpu(), Cb_f.detach().cpu(), Cr_f.detach().cpu()), dim=1))
        t = O_hr_RGB[0].contiguous()
        hwc = tensor_to_uint8_hwc(t)

        channel = path.split("/")[-2]
        if str(channel) in flip_cams:
            hwc = cv2.flip(hwc, 0)

        out_path = os.path.join(config["path"]["results_path"], path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        Image.fromarray(hwc).save(out_path, format="JPEG", compress_level=95)
        logger.success(f"{path} saved， pid: {rank}")

        out_scale_path = os.path.join(config["path"]["results_scale_path"], path)
        os.makedirs(os.path.dirname(out_scale_path), exist_ok=True)
        channel = path.split("/")[-2]
        if channel == "16":
            image_scale = resize_by_height(hwc, config["scale"]["target_height_16"])
        else:
            image_scale = resize_by_height(hwc, config["scale"]["target_height"])
        Image.fromarray(image_scale).save(out_scale_path, format="JPEG", quality=95)
