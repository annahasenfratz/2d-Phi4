from __future__ import annotations

from pathlib import Path

import numpy as np

from .kernel_io import KernelSpec


def load_configs(path: Path, max_configs: int | None = None) -> np.ndarray:
    with np.load(path) as data:
        for key in ("phi", "configs", "arr_0"):
            if key in data.files:
                arr = data[key]
                break
        else:
            arr = data[data.files[0]]
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[1] != arr.shape[2]:
        raise ValueError(f"expected configs with shape (N,L,L), got {arr.shape} from {path}")
    if max_configs is not None:
        arr = arr[:max_configs]
    return arr


def apply_kernel(phi: np.ndarray, kernel: KernelSpec) -> np.ndarray:
    arr = np.asarray(phi, dtype=np.float64)
    matrix = np.asarray(kernel.matrix, dtype=np.float64)
    radius = matrix.shape[0] // 2
    out = np.zeros_like(arr, dtype=np.float64)
    for i, dx in enumerate(range(-radius, radius + 1)):
        for j, dy in enumerate(range(-radius, radius + 1)):
            w = matrix[i, j]
            if w == 0.0:
                continue
            out += w * np.roll(np.roll(arr, -dx, axis=1), -dy, axis=2)
    return out


def block_configs(phi: np.ndarray, kernel: KernelSpec, block_factor: int | None = None) -> np.ndarray:
    factor = int(block_factor if block_factor is not None else kernel.scale_factor)
    if factor <= 0:
        raise ValueError("block_factor must be positive")
    smooth = apply_kernel(phi, kernel)
    return smooth[:, 0::factor, 0::factor]
