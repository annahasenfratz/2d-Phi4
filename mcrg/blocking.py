"""Blocking adapters.  The perfect-kernel path delegates to production code."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_KERNEL = ROOT / "perfect_blocking/perfect_blocking_lam1p0/kernels/selected_for_upscaling/current_final_7x7_no33_nn_constrained_eta_included.json"
PRODUCTION_KERNEL_5X5 = ROOT / "perfect_blocking/perfect_blocking_lam1p0/kernels/final/chosen_kernel.json"
PB_ROOT = ROOT / "perfect_blocking"
if str(PB_ROOT) not in sys.path:
    sys.path.insert(0, str(PB_ROOT))

from scripts.common.blocking import block_configs as production_block_configs  # noqa: E402
from scripts.common.kernel_io import load_kernel as production_load_kernel  # noqa: E402


def perfect_kernel(path: Path = PRODUCTION_KERNEL):
    """Load the production eta-included 7x7 kernel without reimplementing it."""
    return production_load_kernel(path)


def perfect_block(phi: np.ndarray, path: Path = PRODUCTION_KERNEL) -> np.ndarray:
    return production_block_configs(np.asarray(phi), perfect_kernel(path))


def perfect5_block(phi: np.ndarray, path: Path = PRODUCTION_KERNEL_5X5) -> np.ndarray:
    """Production support-balanced eta-included 5x5 kernel, applied unchanged."""
    return production_block_configs(np.asarray(phi), perfect_kernel(path))


def perfect_zero_momentum_response(path: Path = PRODUCTION_KERNEL) -> float:
    return float(perfect_kernel(path).matrix.sum())


def average_block(phi: np.ndarray, normalization: str = "matched", perfect_kernel_path: Path = PRODUCTION_KERNEL) -> np.ndarray:
    """Periodic 2x2 blocks anchored at (2i,2j), unlike centered odd-width kernels."""
    arr = np.asarray(phi, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[-1] != arr.shape[-2] or arr.shape[-1] % 2:
        raise ValueError(f"expected (N,L,L) with even L, got {arr.shape}")
    out = (arr[:, 0::2, 0::2] + arr[:, 1::2, 0::2] + arr[:, 0::2, 1::2] + arr[:, 1::2, 1::2]) / 4.0
    if normalization == "literal":
        return out
    if normalization == "matched":
        return perfect_zero_momentum_response(perfect_kernel_path) * out
    raise ValueError("normalization must be 'literal' or 'matched'")


def block(phi: np.ndarray, kernel: str, average_normalization: str = "matched", kernel_path: Path = PRODUCTION_KERNEL) -> np.ndarray:
    if kernel == "average":
        return average_block(phi, average_normalization, kernel_path)
    if kernel == "perfect7":
        return perfect_block(phi, kernel_path)
    if kernel == "perfect5":
        return perfect5_block(phi, PRODUCTION_KERNEL_5X5)
    raise ValueError(f"unknown kernel {kernel}")
