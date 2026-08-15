"""Volume-scalable calibrated empirical joint-detail initializer."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from .kernels import apply_kernel, inverse_kernel, load_kernel


@dataclass(frozen=True)
class EmpiricalUpscalerConfig:
    donor_ensemble_path: Path
    donor_start: int
    donor_stop: int
    kernel_path: Path
    kernel_sha256: str
    k: int
    beta: float
    radial_mean_scale: float
    radial_sigma_coefficient: float
    reblocking_tolerance: float


def load_config(path: str | Path, project_root: Path) -> EmpiricalUpscalerConfig:
    raw = json.loads(Path(path).read_text())
    resolve = lambda value: Path(value) if Path(value).is_absolute() else project_root / value
    kernel = resolve(raw["kernel_path"])
    if hashlib.sha256(kernel.read_bytes()).hexdigest() != raw["kernel_sha256"]:
        raise ValueError("unexpected empirical upscaler kernel hash")
    if not raw.get("kernel_coefficients_include_eta_scale") or raw.get("eta_extra_multiplier"):
        raise ValueError("empirical upscaler requires eta-included kernel with no extra multiplier")
    return EmpiricalUpscalerConfig(resolve(raw["donor_ensemble_path"]), int(raw["donor_source_indices"]["start"]), int(raw["donor_source_indices"]["stop"]), kernel, raw["kernel_sha256"], int(raw["k"]), float(raw["beta"]), float(raw["radial_mean_scale"]), float(raw["radial_sigma_coefficient"]), float(raw["reblocking_tolerance"]))


def _load_phi(path: Path) -> np.ndarray:
    with np.load(path) as payload:
        return np.asarray(payload["phi"] if "phi" in payload.files else payload[payload.files[0]], dtype=np.float32)


def _metadata(n: int, lc: int) -> np.ndarray:
    return np.asarray([(i, y, x) for i in range(n) for y in range(0, lc, 2) for x in range(0, lc, 2)], dtype=np.int32)


def _features(coarse: np.ndarray, meta: np.ndarray) -> np.ndarray:
    lc = coarse.shape[1]
    return np.asarray([[coarse[i, (y + dy) % lc, (x + dx) % lc] for dy in range(-3, 4) for dx in range(-3, 4)] for i, y, x in meta], dtype=np.float32)


def _detail_vectors(detail: np.ndarray, meta: np.ndarray) -> np.ndarray:
    return np.asarray([detail[i, :, y:y + 2, x:x + 2].reshape(12) for i, y, x in meta], dtype=np.float32)


def _assemble(coarse: np.ndarray, detail: np.ndarray) -> np.ndarray:
    psi = np.empty((len(coarse), 2 * coarse.shape[1], 2 * coarse.shape[2]), dtype=np.float32)
    psi[:, 0::2, 0::2] = coarse; psi[:, 0::2, 1::2] = detail[:, 0]; psi[:, 1::2, 0::2] = detail[:, 1]; psi[:, 1::2, 1::2] = detail[:, 2]
    return psi


def apply_radial_calibration(
    detail: np.ndarray,
    z: np.ndarray,
    *,
    mean_scale: float,
    sigma_coefficient: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the fixed-volume radial calibration without changing D11."""
    array = np.asarray(detail, dtype=np.float32)
    if array.ndim != 4 or array.shape[1] != 3 or array.shape[2] != array.shape[3]:
        raise ValueError(f"expected detail shape (C,3,Lc,Lc), got {array.shape}")
    latent = np.asarray(z, dtype=np.float32)
    if latent.shape != (array.shape[0],):
        raise ValueError(f"expected one radial latent per chain, got {latent.shape}")
    gamma = np.float32(mean_scale) * np.exp(np.float32(sigma_coefficient / array.shape[2]) * latent)
    calibrated = array.copy()
    calibrated[:, :2] *= gamma[:, None, None, None]
    return calibrated, gamma


class EmpiricalJointDetailUpscaler:
    def __init__(self, config: EmpiricalUpscalerConfig):
        self.config = config
        self.kernel, _ = load_kernel(config.kernel_path)
        donor = _load_phi(config.donor_ensemble_path)[config.donor_start:config.donor_stop + 1]
        self.donor_count = len(donor)
        psi = apply_kernel(donor, self.kernel)
        coarse = psi[:, 0::2, 0::2]; detail = np.stack((psi[:, 0::2, 1::2], psi[:, 1::2, 0::2], psi[:, 1::2, 1::2]), axis=1)
        meta = _metadata(len(donor), coarse.shape[1]); raw_h = _features(coarse, meta)
        self.hmean = raw_h.mean(0); self.hstd = raw_h.std(0) + 1.e-6
        self.donor_h = (raw_h - self.hmean) / self.hstd; self.donor_d = _detail_vectors(detail, meta); self.tree = cKDTree(self.donor_h)

    def sample(self, coarse: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        c = np.asarray(coarse, dtype=np.float32)
        if c.ndim != 3 or c.shape[1] != c.shape[2] or c.shape[1] % 2:
            raise ValueError(f"expected (N,Lc,Lc) with even Lc, got {c.shape}")
        meta = _metadata(len(c), c.shape[1]); h = (_features(c, meta) - self.hmean) / self.hstd
        distance, index = self.tree.query(h, k=self.config.k); tau = float(np.quantile(distance[:, 0], .25)); scale = self.config.beta * (self.donor_d.std(0) + 1.e-6)
        blocks = np.empty((len(meta), 12), dtype=np.float32); chosen = np.empty(len(meta), dtype=np.int64)
        for row in range(len(meta)):
            weight = np.exp(-(distance[row] ** 2 - distance[row, 0] ** 2) / (2. * tau * tau)); weight /= weight.sum()
            pick = int(rng.choice(self.config.k, p=weight)); chosen[row] = index[row, pick]
            blocks[row] = self.donor_d[chosen[row]] + rng.normal(size=12).astype(np.float32) * scale
        detail = np.zeros((len(c), 3, c.shape[1], c.shape[2]), dtype=np.float32)
        for row, (chain, y, x) in enumerate(meta): detail[chain, :, y:y + 2, x:x + 2] = blocks[row].reshape(3, 2, 2)
        z = rng.normal(size=len(c)).astype(np.float32)
        detail, gamma = apply_radial_calibration(
            detail,
            z,
            mean_scale=self.config.radial_mean_scale,
            sigma_coefficient=self.config.radial_sigma_coefficient,
        )
        psi = _assemble(c, detail); fine, inverse_info = inverse_kernel(psi, self.kernel); reblock = float(np.max(np.abs(apply_kernel(fine, self.kernel)[:, 0::2, 0::2] - c)))
        if reblock > self.config.reblocking_tolerance: raise RuntimeError(f"empirical initializer reblocking error {reblock}")
        return fine, detail, z, {
            "initializer_type": "calibrated_empirical_joint_2x2",
            "empirical_donor_count": self.donor_count,
            "empirical_k": self.config.k,
            "empirical_tau": tau,
            "empirical_beta": self.config.beta,
            "empirical_tiling_offset": [0, 0],
            "radial_mean_scale": self.config.radial_mean_scale,
            "radial_sigma_coefficient": self.config.radial_sigma_coefficient,
            "radial_sigma": self.config.radial_sigma_coefficient / c.shape[1],
            "radial_gamma_mean": float(gamma.mean()),
            "radial_gamma_std": float(gamma.std()),
            "reblocking_error": reblock,
            "inverse": inverse_info,
            "selected_donor_blocks": chosen,
        }
