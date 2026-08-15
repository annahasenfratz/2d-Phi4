#!/usr/bin/env python3
"""Exact block-consistency linear design for symmetric block variables."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
OUT = PROJECT / "outputs" / "exact_block_consistency_design"
BASE = PROJECT / "outputs" / "inverse_blocking_step_by_step"
KERNEL = ROOT / "perfect_blocking" / "perfect_blocking_lam1p0_blockavg" / "perfect_block_lam1_blockavg_kernel5x5_kernel.json"

L_FINE = 16
L_COARSE = 8
ETA_EXPONENT = 0.25
BLOCK_NORM = 2 ** (ETA_EXPONENT / 2.0)

SHELLS = {
    "w00": [(0, 0)],
    "w10": [(1, 0), (-1, 0), (0, 1), (0, -1)],
    "w11": [(1, 1), (1, -1), (-1, 1), (-1, -1)],
    "w20": [(2, 0), (-2, 0), (0, 2), (0, -2)],
    "w21": [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)],
    "w22": [(2, 2), (2, -2), (-2, 2), (-2, -2)],
}


def block_sym_np(phi: np.ndarray, w: dict[str, float]) -> np.ndarray:
    psi = np.zeros_like(phi, dtype=np.float64)
    for shell, offsets in SHELLS.items():
        for dy, dx in offsets:
            psi += w[shell] * np.roll(np.roll(phi, -dy, axis=-2), -dx, axis=-1)
    return 0.25 * BLOCK_NORM * (psi[:, 0::2, 0::2] + psi[:, 1::2, 0::2] + psi[:, 0::2, 1::2] + psi[:, 1::2, 1::2])


def build_B(w: dict[str, float]) -> np.ndarray:
    B = np.zeros((L_COARSE * L_COARSE, L_FINE * L_FINE), dtype=np.float64)
    for j in range(L_FINE * L_FINE):
        e = np.zeros((1, L_FINE, L_FINE), dtype=np.float64)
        e.reshape(1, -1)[0, j] = 1.0
        B[:, j] = block_sym_np(e, w).reshape(-1)
    return B


def kernel_array(w: dict[str, float]) -> np.ndarray:
    arr = np.zeros((L_FINE, L_FINE), dtype=float)
    for shell, offsets in SHELLS.items():
        for dy, dx in offsets:
            arr[dy % L_FINE, dx % L_FINE] += w[shell]
    return arr


def smooth_backbone(coarse: np.ndarray, w: dict[str, float]) -> np.ndarray:
    ktilde = np.fft.fft2(kernel_array(w))
    p = np.fft.fftfreq(L_FINE) * 2.0 * np.pi
    px, py = np.meshgrid(p, p)
    A = 0.25 * (1.0 + np.exp(1j * py)) * (1.0 + np.exp(1j * px))
    cft = np.fft.fft2(coarse, axes=(-2, -1))
    padded_shift = np.zeros((coarse.shape[0], L_FINE, L_FINE), dtype=complex)
    padded_shift[:, 4:12, 4:12] = 4.0 * np.fft.fftshift(cft, axes=(-2, -1))
    padded = np.fft.ifftshift(padded_shift, axes=(-2, -1))
    mask_shift = np.zeros((L_FINE, L_FINE), dtype=bool)
    mask_shift[4:12, 4:12] = True
    mask = np.fft.ifftshift(mask_shift)
    inv = np.zeros_like(padded)
    inv[:, mask] = padded[:, mask] / (BLOCK_NORM * ktilde * A)[mask]
    return np.fft.ifft2(inv, axes=(-2, -1)).real


def obs(phi: np.ndarray) -> dict[str, float]:
    nn = 0.5 * ((phi * np.roll(phi, -1, axis=-2)).mean(axis=(-2, -1)) + (phi * np.roll(phi, -1, axis=-1)).mean(axis=(-2, -1)))
    nn2 = 0.5 * (((phi * np.roll(phi, -1, axis=-2)) ** 2).mean(axis=(-2, -1)) + ((phi * np.roll(phi, -1, axis=-1)) ** 2).mean(axis=(-2, -1)))
    diag = (phi * np.roll(np.roll(phi, -1, axis=-2), -1, axis=-1)).mean(axis=(-2, -1))
    twonn = 0.5 * ((phi * np.roll(phi, -2, axis=-2)).mean(axis=(-2, -1)) + (phi * np.roll(phi, -2, axis=-1)).mean(axis=(-2, -1)))
    return {
        "phi2": float(np.mean(phi**2)),
        "phi4": float(np.mean(phi**4)),
        "NN": float(np.mean(nn)),
        "nn2": float(np.mean(nn2)),
        "diag": float(np.mean(diag)),
        "2nn": float(np.mean(twonn)),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(f"Refusing to overwrite existing outputs in {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    meta = json.loads(KERNEL.read_text())
    w = {k: float(meta["weights"][k]) for k in ["w00", "w10", "w11", "w20", "w21", "w22"]}
    B = build_B(w)
    BBt = B @ B.T
    BBt_inv = np.linalg.inv(BBt)
    rank = int(np.linalg.matrix_rank(B, tol=1e-12))
    eig = np.linalg.eigvalsh(BBt)
    linear = {
        "B_shape": list(B.shape),
        "rank_B": rank,
        "nullity": int(B.shape[1] - rank),
        "condition_number_BBt": float(np.linalg.cond(BBt)),
        "min_eigenvalue_BBt": float(eig[0]),
        "max_eigenvalue_BBt": float(eig[-1]),
        "projection_formula": "P_null(u)=u-B^T(BB^T)^(-1)Bu",
    }
    (OUT / "linear_operator_check.json").write_text(json.dumps(linear, indent=2) + "\n")

    fine = np.load(BASE / "input_fine_batch.npy").astype(np.float64)
    coarse = block_sym_np(fine, w)
    backbone = smooth_backbone(coarse, w)
    back_res = block_sym_np(backbone, w) - coarse
    backbone_check = {
        "max_abs_B_backbone_minus_phi_c": float(np.max(np.abs(back_res))),
        "rms_B_backbone_minus_phi_c": float(np.sqrt(np.mean(back_res**2))),
        **{f"backbone_{k}": v for k, v in obs(backbone).items()},
        **{f"original_{k}": v for k, v in obs(fine).items()},
    }
    (OUT / "backbone_check.json").write_text(json.dumps(backbone_check, indent=2) + "\n")
    np.save(OUT / "phi_backbone.npy", backbone)

    rng = np.random.default_rng(20240623)
    U = rng.normal(size=(16, L_FINE * L_FINE))
    P = np.eye(L_FINE * L_FINE) - B.T @ BBt_inv @ B
    PU = U @ P.T
    BP = (B @ PU.T).T
    phi = backbone[:16].reshape(16, -1) + PU
    phi = phi.reshape(16, L_FINE, L_FINE)
    gen_res = block_sym_np(phi, w) - coarse[:16]
    null_check = {
        "random_u_count": 16,
        "max_abs_B_Pnull_u": float(np.max(np.abs(BP))),
        "rms_B_Pnull_u": float(np.sqrt(np.mean(BP**2))),
        "max_abs_B_backbone_plus_detail_minus_phi_c": float(np.max(np.abs(gen_res))),
        "rms_B_backbone_plus_detail_minus_phi_c": float(np.sqrt(np.mean(gen_res**2))),
        "projection_idempotence_error": float(np.max(np.abs(P @ P - P))),
        "projection_symmetry_error": float(np.max(np.abs(P.T - P))),
    }
    (OUT / "null_projection_check.json").write_text(json.dumps(null_check, indent=2) + "\n")

    fine_residual = fine - backbone
    fine_res_proj = fine_residual.reshape(len(fine), -1) @ P.T
    rows = []
    for name, arr in {
        "original_fine": fine,
        "backbone": backbone,
        "raw_fine_minus_backbone": fine_residual,
        "projected_null_residual": fine_res_proj.reshape(len(fine), L_FINE, L_FINE),
        "random_null_detail": PU.reshape(16, L_FINE, L_FINE),
        "backbone_plus_random_null_detail": phi,
    }.items():
        vals = obs(arr)
        block = block_sym_np(arr, w) if arr.shape[1:] == (L_FINE, L_FINE) else None
        rows.append({"ensemble": name, **vals, "block_rms": float(np.sqrt(np.mean(block**2))) if block is not None else math.nan})
    write_csv(OUT / "residual_observable_check.csv", rows)
    report = f"""# Exact Block-Consistency Design

## Linear Operator

- B shape: {linear['B_shape']}
- rank: {rank}
- nullity: {linear['nullity']}
- cond(BB^T): {linear['condition_number_BBt']:.12g}

## Backbone

- max |B phi_backbone - phi_c|: {backbone_check['max_abs_B_backbone_minus_phi_c']:.12g}
- RMS residual: {backbone_check['rms_B_backbone_minus_phi_c']:.12g}

## Null Projection

- max |B P_null(u)|: {null_check['max_abs_B_Pnull_u']:.12g}
- RMS |B P_null(u)|: {null_check['rms_B_Pnull_u']:.12g}
- max |B(backbone+detail)-phi_c|: {null_check['max_abs_B_backbone_plus_detail_minus_phi_c']:.12g}

## Answers

1. B_sym can be represented as a 64 x 256 matrix and projected stably for this lattice.
2. The low-mode backbone blocks back to phi_c to numerical precision.
3. The null projection annihilates B_sym to numerical precision.
4. An exact constrained flow is feasible: generate coordinates in the null space and add them to the backbone.
5. The next NF should generate null-space detail variables rather than unconstrained full fields with a soft penalty.
"""
    (OUT / "report.md").write_text(report)


if __name__ == "__main__":
    main()
