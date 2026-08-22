"""Covariances, stable Swendsen solves, eigensystems, and bootstrap."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class RGResult:
    A: np.ndarray
    B: np.ndarray
    T: np.ndarray
    eigenvalues: np.ndarray
    right_eigenvectors: np.ndarray
    left_eigenvectors: np.ndarray
    singular_values: np.ndarray
    a_eigenvalues: np.ndarray
    condition_number: float
    svd_rtol: float


def connected_covariance(x: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
    """Sample connected covariance with observations on axis zero."""
    x = np.asarray(x, dtype=np.float64)
    y = x if y is None else np.asarray(y, dtype=np.float64)
    if x.shape[0] != y.shape[0] or x.shape[0] < 2:
        raise ValueError("covariance requires matching arrays with at least two observations")
    return (x - x.mean(0)).T @ (y - y.mean(0)) / (x.shape[0] - 1)


def solve_rg(fine: np.ndarray, coarse: np.ndarray, svd_rtol: float = 1e-10, scales: np.ndarray | None = None) -> RGResult:
    """Solve A T=B by SVD, never by explicit inverse.

    ``scales`` is a fixed operator-basis scaling shared by both levels.
    """
    if fine.shape != coarse.shape:
        raise ValueError(f"operator arrays must match, got {fine.shape} and {coarse.shape}")
    d = np.ones(fine.shape[1]) if scales is None else np.asarray(scales, dtype=np.float64)
    if d.shape != (fine.shape[1],) or np.any(d == 0):
        raise ValueError("scales must be one nonzero factor per operator")
    # Same D at both levels: T' = D^-1 T D, a similarity transformation.
    f, c = fine * d, coarse * d
    A = connected_covariance(c)
    B = connected_covariance(c, f)
    u, s, vh = np.linalg.svd(A, full_matrices=False)
    cutoff = svd_rtol * s[0] if s.size else 0.0
    sinv = np.where(s > cutoff, 1.0 / s, 0.0)
    T_scaled = (vh.T * sinv) @ u.T @ B
    T = (d[:, None] * T_scaled) / d[None, :]
    values, right = np.linalg.eig(T)
    left_values, left_raw = np.linalg.eig(T.T)
    left = np.empty(right.shape, dtype=np.complex128)
    for i, value in enumerate(values):
        left[:, i] = left_raw[:, np.argmin(np.abs(left_values - value))]
    return RGResult(A, B, T, values, right, left, s, np.linalg.eigvalsh(A), float(s[0] / max(s[-1], np.finfo(float).tiny)), svd_rtol)


def solve_rg_common_transform(fine: np.ndarray, coarse: np.ndarray, transform: np.ndarray, svd_rtol: float = 1e-10) -> RGResult:
    """Common invertible operator change ``S'=W S`` at both RG levels.

    This is a similarity transformation of T; W may be recomputed per bootstrap
    sample, but it must be applied identically to its paired levels.
    """
    w = np.asarray(transform, dtype=np.float64)
    if w.shape != (fine.shape[1], fine.shape[1]):
        raise ValueError("common transform has wrong shape")
    transformed = solve_rg(fine @ w.T, coarse @ w.T, svd_rtol)
    invwt = np.linalg.inv(w.T)
    transformed.T = w.T @ transformed.T @ invwt
    values, right = np.linalg.eig(transformed.T)
    left_values, left_raw = np.linalg.eig(transformed.T.T)
    left = np.empty(right.shape, dtype=np.complex128)
    for i, value in enumerate(values): left[:, i] = left_raw[:, np.argmin(abs(left_values-value))]
    transformed.eigenvalues, transformed.right_eigenvectors, transformed.left_eigenvectors = values, right, left
    return transformed


def leading_real(result: RGResult) -> tuple[float, int]:
    vals = result.eigenvalues
    candidates = np.where((np.abs(vals.imag) < 1e-7 * np.maximum(1, np.abs(vals.real))) & (vals.real > 0))[0]
    if not len(candidates):
        raise ValueError(f"no positive real RG eigenvalue: {vals}")
    i = candidates[np.argmax(vals[candidates].real)]
    return float(vals[i].real), int(i)


def exponents(lambda_t: float | None = None, lambda_h: float | None = None) -> dict[str, float]:
    out: dict[str, float] = {}
    if lambda_t is not None and lambda_t > 1:
        out["nu"] = float(np.log(2.0) / np.log(lambda_t))
    if lambda_h is not None and lambda_h > 0:
        yh = float(np.log(lambda_h) / np.log(2.0))
        out.update(y_h=yh, eta=float(4.0 - 2.0 * yh))
    return out


def bootstrap_rg(fine: np.ndarray, coarse: np.ndarray, svd_rtol: float, n_bootstrap: int, seed: int) -> list[RGResult]:
    rng = np.random.default_rng(seed)
    n = fine.shape[0]
    # One index vector per replicate: each fine/coarse pair is derived from the
    # same original configuration and must remain paired in resampling.
    return [solve_rg(fine[idx], coarse[idx], svd_rtol) for idx in (rng.integers(n, size=n) for _ in range(n_bootstrap))]
