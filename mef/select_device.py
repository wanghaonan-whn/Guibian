import torch
from typing import Tuple
from collections.abc import Sequence


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
    @staticmethod
    def get_device_list(config: dict) -> list[str]:
        return parse_device_list(config, "cuda")

    @staticmethod
    def setup_device(device_str: str) -> Tuple[torch.device, str]:
        device = torch.device(device_str)
        torch.cuda.set_device(device)
        return device, device_str


class Ascend:
    def __init__(self, _torch_npu=None) -> None:
        self._torch_npu = _torch_npu

    def get_torch_npu(self):
        if self._torch_npu is None:
            import torch_npu as _tn
            self._torch_npu = _tn
        return self._torch_npu

    def setup_npu(self, device_str: str) -> Tuple[torch.device, str]:
        if ":" in device_str:
            idx = int(device_str.split(":")[1])
        else:
            idx = 0
            device_str = "npu:0"
        self.get_torch_npu().npu.set_device(idx)
        return torch.device(device_str), device_str

    @staticmethod
    def get_device_list(config: dict) -> list[str]:

        return parse_device_list(config, "npu")


class DeviceManager:
    def __init__(self, device_config=None, backend: str | None = None):
        if backend not in ("cuda", "npu"):
            raise ValueError("backend must be 'cuda' or 'npu'")
        self.device_config = device_config
        self.backend = backend
        self.backend_impl = self._init_backend()

    def _init_backend(self):
        if self.backend == "cuda":
            return Nvidia()
        if self.backend == "npu":
            return Ascend()
        return None

    def get_device_list(self) -> list[str]:
        """ return a list of available devices """
        return self.backend_impl.get_device_list(self.device_config)

    def setup_device(self, device_str: str | None = None) -> Tuple[torch.device, str]:
        """ init device """
        if device_str is None:
            device_str = self.get_device_list()[0]
        return self.backend_impl.setup_device(device_str)
