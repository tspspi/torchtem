from __future__ import annotations

import warnings
from numbers import Number
from types import ModuleType

import numpy as np
import scipy
import scipy.ndimage
import torch

import tspi.torchtem.config as config

try:
    import cupy as cp  # type: ignore
except Exception:
    cp = None


def check_cupy_is_installed() -> None:
    if cp is None:
        raise RuntimeError("CuPy is not installed, GPU calculations disabled")


def validate_device(device: str | None = None) -> str:
    if device is None:
        configured = config.get("device")
        assert isinstance(configured, str)
        return configured
    device = device.lower()
    if device not in {"cpu", "gpu", "cuda"}:
        raise ValueError(f"Unsupported device {device}")
    return "gpu" if device == "cuda" else device


def get_array_module(
    x: ModuleType | np.ndarray | torch.Tensor | str | None = None,
) -> ModuleType:
    if x is None:
        return get_array_module(config.get("device"))
    if isinstance(x, str):
        if x.lower() in {"numpy", "cpu"}:
            return np
        if x.lower() in {"torch", "gpu", "cuda"}:
            return torch
    if isinstance(x, np.ndarray) or isinstance(x, Number) or x is np:
        return np
    if isinstance(x, torch.Tensor) or x is torch:
        return torch
    if cp is not None and isinstance(x, cp.ndarray):
        return cp
    raise ValueError(f"array module specification {x} not recognized")


def device_name_from_array_module(xp: ModuleType) -> str:
    if xp is np:
        return "cpu"
    if xp is torch or (cp is not None and xp is cp):
        return "gpu"
    raise ValueError(f"array module must be NumPy, Torch, or CuPy, not {xp}")


def get_scipy_module(x: ModuleType | np.ndarray | torch.Tensor | str | None = None):
    xp = get_array_module(x)
    if xp is np:
        return scipy
    if xp is torch:
        return torch
    if cp is not None and xp is cp:
        import cupyx.scipy  # type: ignore

        return cupyx.scipy
    raise ValueError(f"array module must be NumPy, Torch, or CuPy, not {xp}")


def get_ndimage_module(
    x: ModuleType | np.ndarray | torch.Tensor | str | None = None,
) -> ModuleType:
    xp = get_array_module(x)
    if xp is np:
        return scipy.ndimage
    if xp is torch:
        return torch
    if cp is not None and xp is cp:
        import cupyx.scipy.ndimage as cupyx_ndimage  # type: ignore

        return cupyx_ndimage
    raise RuntimeError("Invalid array module")


def asnumpy(array):
    if isinstance(array, np.ndarray):
        return array
    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    if cp is not None and isinstance(array, cp.ndarray):
        return cp.asnumpy(array)
    return np.asarray(array)


def copy_to_device(array, device: str = "cpu"):
    validated = validate_device(device)
    if isinstance(array, torch.Tensor):
        if validated == "cpu":
            return array.cpu()
        if not torch.cuda.is_available():
            warnings.warn("CUDA requested but not available; leaving tensor on current device")
            return array
        return array.cuda()
    if isinstance(array, np.ndarray):
        if validated == "cpu":
            return array
        return torch.as_tensor(array, device="cuda" if torch.cuda.is_available() else "cpu")
    return array


def ensure_cuda_cluster():
    raise RuntimeError("Torch backend does not use dask-cuda clusters")


def is_gpu_dask_client(client) -> bool:
    return False
