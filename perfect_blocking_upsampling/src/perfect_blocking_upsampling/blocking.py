"""Factor-two blocking utilities shared by the wrapped-flow MIT drivers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


ETA_SCALE = 2.0**0.125


def json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(type(obj).__name__)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")


def load_phi(path: Path) -> np.ndarray:
    with np.load(path) as z:
        arr = z["phi"] if "phi" in z.files else z[z.files[0]]
    return np.asarray(arr, dtype=np.float32)


def load_kernel_matrix(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not bool(data.get("kernel_coefficients_include_eta_scale", False)):
        raise ValueError(f"kernel is not marked eta-included: {path}")
    eta_scale = float(data.get("eta_scale_numeric", ETA_SCALE))
    if "matrix" in data:
        mat = np.asarray(data["matrix"], dtype=np.float64)
    elif "base_matrix_before_eta_scale" in data:
        mat = eta_scale * np.asarray(data["base_matrix_before_eta_scale"], dtype=np.float64)
    else:
        raise ValueError(f"kernel JSON has no matrix field: {path}")
    if not np.isclose(float(mat.sum()), eta_scale, atol=1.0e-10, rtol=0.0):
        raise ValueError(f"kernel sum {float(mat.sum()):.17g} != eta scale {eta_scale:.17g}: {path}")
    return mat, data


def apply_kernel(phi: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    arr = np.asarray(phi, dtype=np.float64)
    out = np.zeros_like(arr)
    radius = kernel.shape[0] // 2
    for y in range(kernel.shape[0]):
        for x in range(kernel.shape[1]):
            weight = float(kernel[y, x])
            if weight:
                out += weight * np.roll(np.roll(arr, y - radius, axis=-2), x - radius, axis=-1)
    return out.astype(np.float32)


def kernel_fft(kernel: np.ndarray, lattice_size: int) -> np.ndarray:
    weights = np.zeros((lattice_size, lattice_size), dtype=np.float64)
    radius = kernel.shape[0] // 2
    for y in range(kernel.shape[0]):
        for x in range(kernel.shape[1]):
            weights[(y - radius) % lattice_size, (x - radius) % lattice_size] += kernel[y, x]
    return np.fft.fft2(weights)


def inverse_kernel(psi: np.ndarray, kernel: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    transform = kernel_fft(kernel, int(psi.shape[-1]))
    abs_transform = np.abs(transform)
    phi_complex = np.fft.ifft2(np.fft.fft2(psi.astype(np.float64), axes=(-2, -1)) / transform[None], axes=(-2, -1))
    return phi_complex.real.astype(np.float32), {
        "min_abs_K": float(abs_transform.min()),
        "max_abs_K": float(abs_transform.max()),
        "max_invK": float(np.max(1.0 / np.maximum(abs_transform, 1.0e-300))),
        "condition_number": float(abs_transform.max() / max(abs_transform.min(), 1.0e-300)),
        "max_inverse_imag": float(np.max(np.abs(phi_complex.imag))),
    }


def assemble_psi(coarse: np.ndarray, detail: np.ndarray) -> np.ndarray:
    n, lattice_size, _ = coarse.shape
    psi = np.empty((n, 2 * lattice_size, 2 * lattice_size), dtype=np.float32)
    psi[:, 0::2, 0::2] = coarse
    psi[:, 0::2, 1::2] = detail[:, 0]
    psi[:, 1::2, 0::2] = detail[:, 1]
    psi[:, 1::2, 1::2] = detail[:, 2]
    return psi
