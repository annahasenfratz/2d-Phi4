from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ORBIT_OFFSETS: dict[str, list[tuple[int, int]]] = {
    "00": [(0, 0)],
    "10": [(1, 0), (-1, 0), (0, 1), (0, -1)],
    "11": [(1, 1), (1, -1), (-1, 1), (-1, -1)],
    "20": [(2, 0), (-2, 0), (0, 2), (0, -2)],
    "21": [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)],
    "22": [(2, 2), (2, -2), (-2, 2), (-2, -2)],
    "30": [(3, 0), (-3, 0), (0, 3), (0, -3)],
    "31": [(3, 1), (3, -1), (-3, 1), (-3, -1), (1, 3), (1, -3), (-1, 3), (-1, -3)],
}


@dataclass(frozen=True)
class KernelSpec:
    name: str
    type: str
    eta: float
    scale_factor: int
    normalization: str = "sum_to_one"
    orbits: dict[str, float] | None = None
    stencil: list[dict[str, float]] | None = None
    matrix: list[list[float]] | None = None
    eta_scale: float | None = None
    kernel_coefficients_include_eta_scale: bool = False


def load_kernel(path: str | Path) -> tuple[KernelSpec, dict[str, Any]]:
    data = json.loads(Path(path).read_text())
    eta = float(data["eta"])
    eta_scale = float(data.get("eta_scale_numeric", data.get("eta_scale", 2.0 ** (eta / 2.0))))
    spec = KernelSpec(
        name=str(data["name"]),
        type=str(data["type"]),
        eta=eta,
        eta_scale=eta_scale,
        scale_factor=int(data.get("scale_factor", 2)),
        normalization=str(data.get("normalization", "sum_to_one")),
        orbits={str(k): float(v) for k, v in data.get("orbits", {}).items()} if data.get("orbits") is not None else None,
        stencil=data.get("stencil"),
        matrix=data.get("matrix"),
        kernel_coefficients_include_eta_scale=bool(data.get("kernel_coefficients_include_eta_scale", False)),
    )
    stencil = kernel_stencil_from_spec(spec)
    expected_sum = eta_scale if spec.kernel_coefficients_include_eta_scale else 1.0
    actual_sum = float(stencil.sum())
    if spec.kernel_coefficients_include_eta_scale and not np.isclose(actual_sum, expected_sum, rtol=1.0e-12, atol=1.0e-12):
        raise ValueError(
            f"{path} declares eta-included coefficients but sum(K)={actual_sum:.17g}, "
            f"expected eta_scale={expected_sum:.17g}"
        )
    print(
        "loaded kernel "
        f"{path}: include_eta_scale={spec.kernel_coefficients_include_eta_scale}, eta={spec.eta:.17g}, "
        f"eta_scale={spec.eta_scale:.17g}, sum(K)={actual_sum:.17g}"
    )
    return spec, data


def kernel_stencil_from_spec(spec: KernelSpec) -> np.ndarray:
    if spec.type == "orbit_kernel":
        if spec.orbits is None:
            raise ValueError("orbit kernel requires orbits")
        stencil = np.zeros((7, 7), dtype=np.float64)
        center = 3
        for orbit, weight in spec.orbits.items():
            for dx, dy in ORBIT_OFFSETS[orbit]:
                stencil[center + dx, center + dy] += weight
        return stencil
    if spec.type == "explicit_stencil":
        if spec.stencil is None:
            raise ValueError("explicit stencil kernel requires stencil")
        max_off = max(max(abs(int(s["dx"])), abs(int(s["dy"]))) for s in spec.stencil)
        stencil = np.zeros((2 * max_off + 1, 2 * max_off + 1), dtype=np.float64)
        center = max_off
        for s in spec.stencil:
            stencil[center + int(s["dx"]), center + int(s["dy"])] += float(s["weight"])
        return stencil
    if spec.type == "matrix":
        if spec.matrix is None:
            raise ValueError("matrix kernel requires matrix")
        mat = np.asarray(spec.matrix, dtype=np.float64)
        if mat.ndim != 2 or mat.shape[0] != mat.shape[1] or mat.shape[0] % 2 != 1:
            raise ValueError(f"matrix kernel must be an odd square matrix, got {mat.shape}")
        return mat
    raise ValueError(f"unknown kernel type {spec.type}")


def normalize_kernel(stencil: np.ndarray) -> np.ndarray:
    total = float(stencil.sum())
    if not np.isfinite(total) or abs(total) < 1.0e-300:
        raise ValueError("kernel normalization is degenerate")
    return stencil / total


def kernel_fft(stencil: np.ndarray, L: int, eta: float) -> np.ndarray:
    w = np.zeros((L, L), dtype=np.float64)
    k = stencil.shape[0] // 2
    for i in range(stencil.shape[0]):
        for j in range(stencil.shape[1]):
            w[i - k, j - k] = stencil[i, j]
    return (2.0 ** (eta / 2.0)) * np.fft.fft2(w)


def kernel_fft_from_spec(stencil: np.ndarray, L: int, spec: KernelSpec) -> np.ndarray:
    w = np.zeros((L, L), dtype=np.float64)
    k = stencil.shape[0] // 2
    for i in range(stencil.shape[0]):
        for j in range(stencil.shape[1]):
            w[i - k, j - k] = stencil[i, j]
    scale = 1.0 if spec.kernel_coefficients_include_eta_scale else eta_scale_for_spec(spec)
    return scale * np.fft.fft2(w)


def eta_scale_for_spec(spec: KernelSpec) -> float:
    return float(spec.eta_scale if spec.eta_scale is not None else 2.0 ** (spec.eta / 2.0))


def apply_kernel(phi: np.ndarray, spec: KernelSpec) -> np.ndarray:
    arr = np.asarray(phi, dtype=np.float64)
    stencil = kernel_stencil_from_spec(spec)
    if not spec.kernel_coefficients_include_eta_scale:
        stencil = normalize_kernel(stencil)
    out = np.zeros_like(arr, dtype=np.float64)
    k = stencil.shape[0] // 2
    for i in range(stencil.shape[0]):
        for j in range(stencil.shape[1]):
            weight = stencil[i, j]
            if weight == 0:
                continue
            out += weight * np.roll(np.roll(arr, i - k, axis=-2), j - k, axis=-1)
    scale = 1.0 if spec.kernel_coefficients_include_eta_scale else eta_scale_for_spec(spec)
    return (scale * out).astype(np.float32)


def inverse_kernel(psi: np.ndarray, spec: KernelSpec) -> tuple[np.ndarray, dict[str, float]]:
    arr = np.asarray(psi, dtype=np.float64)
    L = int(arr.shape[-1])
    stencil = kernel_stencil_from_spec(spec)
    if not spec.kernel_coefficients_include_eta_scale:
        stencil = normalize_kernel(stencil)
    kt = kernel_fft_from_spec(stencil, L, spec)
    abs_kt = np.abs(kt)
    phi_c = np.fft.ifft2(np.fft.fft2(arr, axes=(-2, -1)) / kt[None, :, :], axes=(-2, -1))
    idx = np.unravel_index(int(np.argmin(abs_kt)), abs_kt.shape)
    info = {
        "min_abs_Keta_tilde": float(abs_kt[idx]),
        "max_abs_Keta_tilde": float(abs_kt.max()),
        "condition_number_abs": float(abs_kt.max() / max(abs_kt[idx], 1.0e-300)),
        "max_inverse_ifft_imag": float(np.max(np.abs(phi_c.imag))),
    }
    return phi_c.real.astype(np.float32), info
