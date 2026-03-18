import torch
from typing import List, Tuple
from loguru import logger
from torch import nn
from mef.select_device import DeviceManager


@torch.inference_mode()
def warmup_shapes_u8(
        model: nn.Module,
        device: torch.device,
        shapes_hw: List[Tuple[int, int]],
        low_size: int,
        iters: int,
        device_manager: DeviceManager,
        rank: int,
):
    model.eval()
    logger.debug(f"[warmup] begin shapes={shapes_hw} iters={iters}")

    for (H, W) in shapes_hw:
        x_hr_u8 = torch.zeros((2, 1, H, W), dtype=torch.uint8, device=device)
        scale = low_size / float(min(H, W))
        lh = max(1, int(round(H * scale)))
        lw = max(1, int(round(W * scale)))
        x_lr_u8 = torch.zeros((2, 1, lh, lw), dtype=torch.uint8, device=device)

        for i in range(max(1, iters)):
            x_hr = x_hr_u8.to(torch.float32).div_(255.0)
            x_lr = x_lr_u8.to(torch.float32).div_(255.0)
            y, _ = model(x_lr, x_hr)

            u8 = (y.clamp(0, 1) * 255.0).to(torch.uint8)
            _ = u8[0, 0].contiguous().cpu().numpy()
            if i == 0:
                logger.debug(f"[warmup] shape=({H, W}) lr=({lh, lw}) ok")

    logger.debug("[warmup] done")
