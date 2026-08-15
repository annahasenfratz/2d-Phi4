from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def load_kernel_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"kernel JSON must be a mapping: {path}")
    return data


def kernel_matrix(data: dict[str, Any]) -> np.ndarray:
    for key in ("kernel", "coefficients", "matrix", "kernel_matrix"):
        if key in data:
            arr = np.asarray(data[key], dtype=float)
            if arr.ndim == 2:
                return arr
    if "orbits" in data and isinstance(data["orbits"], dict):
        orbits = {tuple(int(c) for c in key): float(value) for key, value in data["orbits"].items()}
        radius = max(max(key) for key in orbits)
        arr = np.zeros((2 * radius + 1, 2 * radius + 1), dtype=float)
        for ix, dx in enumerate(range(-radius, radius + 1)):
            for iy, dy in enumerate(range(-radius, radius + 1)):
                key = tuple(sorted((abs(dx), abs(dy)), reverse=True))
                arr[ix, iy] = orbits.get(key, 0.0)
        return arr
    raise ValueError("kernel JSON does not contain a 2D kernel matrix")


def validate_eta_included(path: Path, expected_eta_scale: float, atol: float = 1.0e-10) -> dict[str, Any]:
    data = load_kernel_json(path)
    coeffs = kernel_matrix(data)
    include_eta = bool(data.get("kernel_coefficients_include_eta_scale", False))
    total = float(np.sum(coeffs))
    if not include_eta:
        raise ValueError(f"kernel is not marked eta-included: {path}")
    if not np.isclose(total, expected_eta_scale, atol=atol, rtol=0.0):
        raise ValueError(f"kernel sum {total:.17g} != eta_scale {expected_eta_scale:.17g}: {path}")
    return {"path": str(path), "sum": total, "eta_scale": expected_eta_scale, "shape": list(coeffs.shape)}
