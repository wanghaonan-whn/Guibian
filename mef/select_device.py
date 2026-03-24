import torch
from typing import Tuple
from collections.abc import Sequence

from traits.trait_types import self


def parse_device_list(device_config, device_prefix: str) -> list[str]:
    if device_config is None:
        return [f"{device_prefix}:0"]

    devices = []
    if isinstance(device_config, str):
        items = device_config.split(",")
    elif isinstance(device_config, Sequence) and not isinstance(device_config, (str, bytes)):
        items = device_config
    else:
        items = [device_config]

    for item in items:
        item = str(item).strip()
        if not item:
            continue
        if ":" in item:
            devices.append(item)
        else:
            devices.append(f"{device_prefix}:{item}")
    return devices or [f"{device_prefix}:0"]


class Nvidia:
    def __init__(self, device_config):
        self.device_config = device_config

    @staticmethod
    def get_device_list(config) -> list[str]:
        return parse_device_list(config, "cuda")

    def setup_device(self, rank: int) -> Tuple[torch.device, str]:
        device_list = self.get_device_list(self.device_config)
        device_str = device_list[rank % len(device_list)]
        print(device_list)
        return torch.device(device_str), device_str


class Ascend:
    def __init__(self, device_config, _torch_npu=None) -> None:
        self._torch_npu = _torch_npu
        self.device_config = device_config

    def get_torch_npu(self):
        if self._torch_npu is None:
            import torch_npu as _tn
            self._torch_npu = _tn
        return self._torch_npu

    def setup_device(self, rank: int) -> Tuple[torch.device, str]:
        device_list = self.get_device_list(self.device_config)

        device_str = device_list[rank % len(device_list)]
        idx = int(device_str.split(":")[1])
        self.get_torch_npu().npu.set_device(idx)

        return torch.device(device_str), device_str

    @staticmethod
    def get_device_list(config) -> list[str]:
        return parse_device_list(config, "npu")


class DeviceManager:
    def __init__(self, device_config=None, backend: str | None = None):
        self.device_config = device_config
        self.backend = backend
        self.backend_impl = self.init_backend()

    def init_backend(self):
        if self.backend == "gpu":
            return Nvidia(self.device_config)
        if self.backend == "npu":
            return Ascend(self.device_config)
        return None

    def get_device_list(self) -> list[str]:
        """ return a list of available devices """
        return self.backend_impl.get_device_list(self.device_config)
