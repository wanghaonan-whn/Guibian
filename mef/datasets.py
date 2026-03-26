import cv2
import numpy as np
import torch

EPS = 1e-8
high_size = 2048
low_size = 128


def resize_min_edge_if_both_gt(img: np.ndarray, size: int, interp=cv2.INTER_LINEAR) -> np.ndarray:
    h, w = img.shape[:2]
    if h > size and w > size:
        scale = size / float(min(h, w))
        nh = max(1, int(round(h * scale)))
        nw = max(1, int(round(w * scale)))
        return cv2.resize(img, (nw, nh), interpolation=interp)
    return img


def resize_min_edge_always(img: np.ndarray, size: int, interp=cv2.INTER_LINEAR) -> np.ndarray:
    h, w = img.shape[:2]
    scale = size / float(min(h, w))
    nh = max(1, int(round(h * scale)))
    nw = max(1, int(round(w * scale)))
    return cv2.resize(img, (nw, nh), interpolation=interp)


def split_exposure(image: np.ndarray):
    """Line-scan camera high/low exposure split."""
    low = image[0::2]
    high = image[1::2]
    return high, low


def to_1chw_u8(img: np.ndarray) -> torch.Tensor:
    if img.dtype != np.uint8:
        img = img.astype(np.uint8, copy=False)
    return torch.from_numpy(np.ascontiguousarray(img)).unsqueeze(0)


def preprocess_mefnet(image: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """Build uint8 grayscale tensors on CPU; caller moves/casts on device."""
    high, low = split_exposure(image)

    high_hr = resize_min_edge_if_both_gt(high, high_size, interp=cv2.INTER_LINEAR)
    low_hr = resize_min_edge_if_both_gt(low, high_size, interp=cv2.INTER_LINEAR)
    high_lr = resize_min_edge_always(high, low_size, interp=cv2.INTER_LINEAR)
    low_lr = resize_min_edge_always(low, low_size, interp=cv2.INTER_LINEAR)

    i_he_u8 = torch.stack([to_1chw_u8(high_hr), to_1chw_u8(low_hr)], 0).contiguous()
    i_le_u8 = torch.stack([to_1chw_u8(high_lr), to_1chw_u8(low_lr)], 0).contiguous()
    return i_he_u8, i_le_u8


def tensor_to_uint8_hwc(img: torch.Tensor, expand_gray_to_rgb: bool = True) -> np.ndarray:
    """
    img: [1,H,W] or [3,H,W] tensor on CPU or device.
    If the tensor is float, quantization runs on the current device before copying to CPU.
    """
    if img.ndim != 3:
        raise ValueError(f"Expected a 3D tensor, got shape={tuple(img.shape)}")

    if img.dtype != torch.uint8:
        img = img.clamp(0, 1).mul(255).to(torch.uint8)

    if img.shape[0] == 1:
        if expand_gray_to_rgb:
            img = img.expand(3, *img.shape[1:])
        else:
            return img[0].contiguous().cpu().numpy()
    elif img.shape[0] != 3:
        raise ValueError(f"Expected 1 or 3 channels, got shape={tuple(img.shape)}")

    return img.permute(1, 2, 0).contiguous().cpu().numpy()


def resize_by_height(img: np.ndarray, target_height: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = target_height / h
    new_w = int(w * scale)
    return cv2.resize(img, (new_w, target_height))
