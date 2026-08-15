from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class KernelSpec:
    name: str
    type: str
    eta: float
    eta_scale: float
    scale_factor: int
    normalization: str
    matrix: np.ndarray
    metadata: dict[str, Any]
    kernel_coefficients_include_eta_scale: bool


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def orbit_matrix(orbits: dict[str, float]) -> np.ndarray:
    matrix = np.zeros((5, 5), dtype=np.float64)
    offsets = range(-2, 3)
    for i, dx in enumerate(offsets):
        for j, dy in enumerate(offsets):
            key = f"{abs(dx)}{abs(dy)}"
            key_rev = f"{abs(dy)}{abs(dx)}"
            if key in orbits:
                matrix[i, j] = float(orbits[key])
            elif key_rev in orbits:
                matrix[i, j] = float(orbits[key_rev])
            else:
                raise ValueError(f"missing orbit weight for offset ({dx},{dy})")
    return matrix


def load_kernel(path: Path, *, apply_eta_scale: bool = False) -> KernelSpec:
    data = json.loads(path.read_text())
    if data.get("type") == "orbit_kernel":
        matrix = orbit_matrix({str(k): float(v) for k, v in data["orbits"].items()})
    elif "matrix" in data:
        matrix = np.asarray(data["matrix"], dtype=np.float64)
    else:
        raise ValueError(f"unsupported kernel format in {path}")
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"kernel matrix must be square, got {matrix.shape}")
    eta = float(data.get("eta", 0.0))
    eta_scale = float(data.get("eta_scale_numeric", data.get("eta_scale", 2.0 ** (eta / 2.0))))
    include_eta = bool(data.get("kernel_coefficients_include_eta_scale", False))
    explicit_include_flag = "kernel_coefficients_include_eta_scale" in data
    if include_eta:
        expected_sum = eta_scale
        actual_sum = float(matrix.sum())
        if not np.isclose(actual_sum, expected_sum, rtol=1.0e-12, atol=1.0e-12):
            raise ValueError(
                f"{path} declares eta-included coefficients but sum(K)={actual_sum:.17g}, "
                f"expected eta_scale={expected_sum:.17g}"
            )
    elif explicit_include_flag:
        if not apply_eta_scale:
            raise ValueError(
                f"{path} declares kernel_coefficients_include_eta_scale=false. "
                "Pass --apply-eta-scale to multiply the base-normalized kernel explicitly."
            )
        s = float(matrix.sum())
        if abs(s) <= 1.0e-300:
            raise ValueError(f"kernel normalization is degenerate in {path}")
        matrix = eta_scale * matrix / s
    elif data.get("normalization") == "sum_to_one":
        s = float(matrix.sum())
        if abs(s) > 1.0e-300:
            matrix = matrix / s
    print(
        "loaded kernel "
        f"{path}: include_eta_scale={include_eta}, eta={eta:.17g}, "
        f"eta_scale={eta_scale:.17g}, sum(K)={float(matrix.sum()):.17g}"
    )
    return KernelSpec(
        name=str(data.get("name", path.stem)),
        type=str(data.get("type", "matrix")),
        eta=eta,
        eta_scale=eta_scale,
        scale_factor=int(data.get("scale_factor", data.get("block_factor", 2))),
        normalization=str(data.get("normalization", "")),
        matrix=matrix,
        metadata=data,
        kernel_coefficients_include_eta_scale=include_eta,
    )


def write_kernel_text(kernel: KernelSpec, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"name: {kernel.name}",
        f"type: {kernel.type}",
        f"eta: {kernel.eta}",
        f"eta_scale: {kernel.eta_scale:.17g}",
        f"scale_factor: {kernel.scale_factor}",
        f"normalization: {kernel.normalization}",
        f"kernel_coefficients_include_eta_scale: {kernel.kernel_coefficients_include_eta_scale}",
        f"matrix_shape: {kernel.matrix.shape[0]}x{kernel.matrix.shape[1]}",
        f"matrix_sum: {float(kernel.matrix.sum()):.17g}",
        "",
        "matrix rows correspond to dx=-2,-1,0,1,2 and columns to dy=-2,-1,0,1,2 for 5x5 kernels:",
    ]
    for row in kernel.matrix:
        lines.append(" ".join(f"{x:.17g}" for x in row))
    path.write_text("\n".join(lines) + "\n")


def copy_kernel_with_metadata(src: Path, dst_json: Path, dst_txt: Path) -> dict[str, Any]:
    dst_json.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(src.read_text())
    dst_json.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    kernel = load_kernel(dst_json)
    write_kernel_text(kernel, dst_txt)
    return {
        "source": str(src),
        "destination_json": str(dst_json),
        "destination_txt": str(dst_txt),
        "sha256": sha256_file(dst_json),
        "kernel_name": kernel.name,
    }
