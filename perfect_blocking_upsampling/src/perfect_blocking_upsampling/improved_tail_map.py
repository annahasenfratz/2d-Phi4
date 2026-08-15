"""Frozen deterministic tail maps for wrapped-flow sweep-zero initialization."""
from __future__ import annotations

import numpy as np


def parameters(
    coarse_lattice_size: int,
    *,
    base_scale: float | None = None,
    tail_strength: float | None = None,
    tail_width: float = 2.0,
) -> tuple[float, float, float]:
    """Return the validated factor-two map parameters for the coarse volume."""
    defaults = {16: (1.02, 0.15), 32: (1.03, 0.21)}
    if coarse_lattice_size not in defaults:
        raise ValueError("improved tail-map initialization is validated for L16->L32 and L32->L64 only")
    default_scale, default_strength = defaults[coarse_lattice_size]
    return (
        float(default_scale if base_scale is None else base_scale),
        float(default_strength if tail_strength is None else tail_strength),
        float(tail_width),
    )


def apply(detail: np.ndarray, detail_scale: np.ndarray, *, base_scale: float, tail_strength: float, tail_width: float) -> np.ndarray:
    """Apply an invertible sitewise detail map in physical detail coordinates."""
    scale = np.asarray(detail_scale, dtype=np.float64).reshape(1, 3, 1, 1)
    u = np.asarray(detail, dtype=np.float64) / scale
    tanh_u = np.tanh(u / tail_width)
    mapped = base_scale * (u - tail_strength * tanh_u**3)
    derivative = base_scale * (1.0 - (3.0 * tail_strength / tail_width) * tanh_u**2 * (1.0 - tanh_u**2))
    if not np.all(np.isfinite(mapped)) or not np.all(derivative > 0.0):
        raise ValueError("improved tail map is nonfinite or non-invertible")
    return (mapped * scale).astype(np.float32)
