#!/usr/bin/env python3
"""Optimize a fixed-eta D4-symmetric 5x5 real-space blocking kernel.

Target:
    lambda = 0.022, kappa = 0.2705, L32 -> L16

Convention:
    psi = 2^(eta/2) K phi, eta = 0.25, sum K = 1
    blocked field = psi[:, 0::2, 0::2]
    no four-sublattice average
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize


REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "perfect_blocking" / "perfect_blocking_lam0p022_kappa0p2705_fixedeta"
FINE_PATH = REPO / "phi4_phase-diagram" / "ensembles" / "lam0p022_kappa0p271_L32_embedded_wolff_sign_cluster_plus_radial_heatbath" / "configs.npz"
DIRECT_PATH = REPO / "phi4_phase-diagram" / "ensembles" / "lam0p022_kappa0p271_L16_embedded_wolff_sign_cluster_plus_radial_heatbath" / "configs.npz"

LAMBDA_VALUE = 0.022
KAPPA = 0.2705
L_FINE = 32
L_COARSE = 16
ETA = 0.25
ETA_SCALE = float(2.0 ** (ETA / 2.0))
SEED = 20262705

OPERATORS_DOP = ["m2", "m4", "nn", "diag", "2nn", "nn2", "diag2", "2nn2", "action_density"]
OPERATORS_REPORT = [
    "m_mean",
    "abs_m",
    "m2",
    "m4",
    "binder",
    "susceptibility",
    "xi_over_L",
    "phi2_density",
    "phi4_density",
    "nn",
    "diag",
    "2nn",
    "nn2",
    "diag2",
    "2nn2",
    "action_density",
]


def load_phi(path: Path) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    if "phi" not in data:
        raise KeyError(f"{path} missing phi array")
    phi = data["phi"].astype(np.float64, copy=False)
    if phi.ndim != 3:
        raise ValueError(f"{path} expected shape (N,L,L), got {phi.shape}")
    return phi


def roll(phi: np.ndarray, dy: int, dx: int) -> np.ndarray:
    return np.roll(np.roll(phi, dy, axis=1), dx, axis=2)


def kernel_matrix(w00: float, w10: float, w11: float, w20: float, w21: float, w22: float) -> np.ndarray:
    return np.array(
        [
            [w22, w21, w20, w21, w22],
            [w21, w11, w10, w11, w21],
            [w20, w10, w00, w10, w20],
            [w21, w11, w10, w11, w21],
            [w22, w21, w20, w21, w22],
        ],
        dtype=np.float64,
    )


def weights_from_x(x: np.ndarray) -> tuple[float, float, float, float, float, float]:
    w10, w11, w20, w21, w22 = [float(v) for v in x]
    w00 = 1.0 - 4.0 * w10 - 4.0 * w11 - 4.0 * w20 - 8.0 * w21 - 4.0 * w22
    return w00, w10, w11, w20, w21, w22


def block_5x5(phi: np.ndarray, w00: float, w10: float, w11: float, w20: float, w21: float, w22: float) -> np.ndarray:
    shell10 = roll(phi, 1, 0) + roll(phi, -1, 0) + roll(phi, 0, 1) + roll(phi, 0, -1)
    shell11 = roll(phi, 1, 1) + roll(phi, 1, -1) + roll(phi, -1, 1) + roll(phi, -1, -1)
    shell20 = roll(phi, 2, 0) + roll(phi, -2, 0) + roll(phi, 0, 2) + roll(phi, 0, -2)
    shell21 = (
        roll(phi, 2, 1)
        + roll(phi, 2, -1)
        + roll(phi, -2, 1)
        + roll(phi, -2, -1)
        + roll(phi, 1, 2)
        + roll(phi, 1, -2)
        + roll(phi, -1, 2)
        + roll(phi, -1, -2)
    )
    shell22 = roll(phi, 2, 2) + roll(phi, 2, -2) + roll(phi, -2, 2) + roll(phi, -2, -2)
    psi = ETA_SCALE * (w00 * phi + w10 * shell10 + w11 * shell11 + w20 * shell20 + w21 * shell21 + w22 * shell22)
    return psi[:, 0::2, 0::2]


def action_density(phi: np.ndarray) -> np.ndarray:
    onsite = ((1.0 - 2.0 * LAMBDA_VALUE) * phi**2 + LAMBDA_VALUE * phi**4).sum(axis=(1, 2))
    hop = -2.0 * KAPPA * phi * (np.roll(phi, -1, axis=1) + np.roll(phi, -1, axis=2))
    return (onsite + hop.sum(axis=(1, 2))) / (phi.shape[1] * phi.shape[2])


def xi_over_l(phi: np.ndarray) -> float:
    volume = phi.shape[1] * phi.shape[2]
    m = phi.mean(axis=(1, 2))
    susceptibility = float(volume * (np.mean(m**2) - np.mean(m) ** 2))
    centered = phi - phi.mean(axis=(1, 2), keepdims=True)
    fft = np.fft.fftn(centered, axes=(1, 2))
    power = (fft.real**2 + fft.imag**2) / volume
    f_mean = float(0.5 * (np.mean(power[:, 1, 0]) + np.mean(power[:, 0, 1])))
    if f_mean <= 0.0 or susceptibility <= f_mean:
        return float("nan")
    xi = (1.0 / (2.0 * np.sin(np.pi / phi.shape[1]))) * math.sqrt(susceptibility / f_mean - 1.0)
    return float(xi / phi.shape[1])


def per_config_observables(phi: np.ndarray) -> dict[str, np.ndarray]:
    m = phi.mean(axis=(1, 2))
    nn_y = np.mean(phi * np.roll(phi, -1, axis=1), axis=(1, 2))
    nn_x = np.mean(phi * np.roll(phi, -1, axis=2), axis=(1, 2))
    diag_a = np.mean(phi * roll(phi, -1, -1), axis=(1, 2))
    diag_b = np.mean(phi * roll(phi, -1, 1), axis=(1, 2))
    two_y = np.mean(phi * np.roll(phi, -2, axis=1), axis=(1, 2))
    two_x = np.mean(phi * np.roll(phi, -2, axis=2), axis=(1, 2))
    nn = 0.5 * (nn_y + nn_x)
    diag = 0.5 * (diag_a + diag_b)
    two_nn = 0.5 * (two_y + two_x)
    return {
        "m": m,
        "abs_m": np.abs(m),
        "m2": m**2,
        "m4": m**4,
        "phi2_density": np.mean(phi**2, axis=(1, 2)),
        "phi4_density": np.mean(phi**4, axis=(1, 2)),
        "nn": nn,
        "diag": diag,
        "2nn": two_nn,
        "nn2": nn**2,
        "diag2": diag**2,
        "2nn2": two_nn**2,
        "action_density": action_density(phi),
    }


def summary_from_phi(phi: np.ndarray) -> dict[str, float]:
    obs = per_config_observables(phi)
    m = obs["m"]
    m2 = float(np.mean(obs["m2"]))
    m4 = float(np.mean(obs["m4"]))
    volume = phi.shape[1] * phi.shape[2]
    susceptibility = float(volume * (np.mean(m**2) - np.mean(m) ** 2))
    out = {
        "count": int(len(phi)),
        "m_mean": float(np.mean(m)),
        "abs_m": float(np.mean(obs["abs_m"])),
        "m2": m2,
        "m4": m4,
        "binder": float(1.0 - m4 / (3.0 * max(m2, 1e-16) ** 2)),
        "susceptibility": susceptibility,
        "xi_over_L": xi_over_l(phi),
    }
    for key in ["phi2_density", "phi4_density", "nn", "diag", "2nn", "nn2", "diag2", "2nn2", "action_density"]:
        out[key] = float(np.mean(obs[key]))
    return out


def make_boot_indices(n: int, n_boot: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(n_boot, n), endpoint=False)


def bootstrap_summaries(phi: np.ndarray, boot_indices: np.ndarray) -> list[dict[str, float]]:
    return [summary_from_phi(phi[idx]) for idx in boot_indices]


def with_errors(summary: dict[str, float], reps: list[dict[str, float]]) -> dict[str, float]:
    out = dict(summary)
    for key in OPERATORS_REPORT:
        arr = np.array([r[key] for r in reps], dtype=np.float64)
        out[f"{key}_err"] = float(np.nanstd(arr, ddof=1))
    return out


def zscore(delta: float, err_a: float, err_b: float) -> float:
    denom = math.sqrt(err_a**2 + err_b**2)
    return float(delta / denom) if denom > 0 else 0.0


def kernel_fourier_min_abs(w: tuple[float, float, float, float, float, float], size: int = 32) -> dict[str, float]:
    mat = np.zeros((size, size), dtype=np.float64)
    stencil = kernel_matrix(*w)
    for iy, dy in enumerate(range(-2, 3)):
        for ix, dx in enumerate(range(-2, 3)):
            mat[dy % size, dx % size] = stencil[iy, ix]
    kt = np.fft.fft2(mat)
    return {
        "min_abs_Ktilde": float(np.min(np.abs(kt))),
        "max_abs_Ktilde": float(np.max(np.abs(kt))),
        "min_abs_Keta_tilde": float(ETA_SCALE * np.min(np.abs(kt))),
        "max_abs_Keta_tilde": float(ETA_SCALE * np.max(np.abs(kt))),
    }


def evaluate_kernel(
    fine: np.ndarray,
    direct_summary: dict[str, float],
    direct_boot: list[dict[str, float]],
    boot_indices: np.ndarray,
    x: np.ndarray,
    eps: float,
) -> dict[str, Any] | None:
    weights = weights_from_x(x)
    if not np.all(np.isfinite(weights)):
        return None
    if max(abs(v) for v in weights) > 1.5:
        return None
    norm = weights[0] + 4.0 * weights[1] + 4.0 * weights[2] + 4.0 * weights[3] + 8.0 * weights[4] + 4.0 * weights[5]
    if abs(norm - 1.0) > 1e-10:
        return None
    blocked = block_5x5(fine, *weights)
    if not np.all(np.isfinite(blocked)) or np.std(blocked) < 1e-8 or np.max(np.abs(blocked)) > 80.0:
        return None
    blocked_summary = summary_from_phi(blocked)
    blocked_boot = bootstrap_summaries(blocked, boot_indices)
    delta = np.array([blocked_summary[k] - direct_summary[k] for k in OPERATORS_DOP], dtype=np.float64)
    delta_boot = np.array(
        [[bb[k] - db[k] for k in OPERATORS_DOP] for bb, db in zip(blocked_boot, direct_boot)],
        dtype=np.float64,
    )
    cov = np.cov(delta_boot, rowvar=False, ddof=1)
    if not np.all(np.isfinite(cov)):
        return None
    tr = float(np.trace(cov))
    reg = cov + eps * tr / len(OPERATORS_DOP) * np.eye(len(OPERATORS_DOP))
    try:
        cinv = np.linalg.inv(reg)
    except np.linalg.LinAlgError:
        cinv = np.linalg.pinv(reg, rcond=1e-12)
    dop = float(0.5 * delta @ cinv @ delta)
    z_local = [
        zscore(blocked_summary[k] - direct_summary[k], blocked_summary.get(f"{k}_err", 0.0), direct_summary.get(f"{k}_err", 0.0))
        for k in OPERATORS_DOP
    ]
    ir_z = [
        zscore(blocked_summary[k] - direct_summary[k], blocked_summary.get(f"{k}_err", 0.0), direct_summary.get(f"{k}_err", 0.0))
        for k in ["binder", "xi_over_L"]
    ]
    return {
        "weights": weights,
        "blocked_summary": blocked_summary,
        "direct_summary": direct_summary,
        "D_op": dop,
        "covariance": cov,
        "covariance_reg": reg,
        "cov_contrib": 0.5 * delta * (cinv @ delta),
        "local_rms_z": float(np.sqrt(np.nanmean(np.square(z_local)))),
        "ir_rms_z": float(np.sqrt(np.nanmean(np.square(ir_z)))),
        "condition_number_C": float(np.linalg.cond(cov)),
        "condition_number_Creg": float(np.linalg.cond(reg)),
    }


def dop_summary_from_obs(obs: dict[str, np.ndarray]) -> dict[str, float]:
    return {key: float(np.mean(obs[key])) for key in OPERATORS_DOP}


def dop_bootstrap_from_obs(obs: dict[str, np.ndarray], boot_indices: np.ndarray) -> np.ndarray:
    out = np.empty((len(boot_indices), len(OPERATORS_DOP)), dtype=np.float64)
    for b, idx in enumerate(boot_indices):
        for j, key in enumerate(OPERATORS_DOP):
            out[b, j] = float(np.mean(obs[key][idx]))
    return out


def evaluate_kernel_fast(
    fine: np.ndarray,
    direct_dop_summary: dict[str, float],
    direct_dop_boot: np.ndarray,
    boot_indices: np.ndarray,
    x: np.ndarray,
    eps: float,
) -> dict[str, Any] | None:
    weights = weights_from_x(x)
    if not np.all(np.isfinite(weights)):
        return None
    if max(abs(v) for v in weights) > 1.5:
        return None
    norm = weights[0] + 4.0 * weights[1] + 4.0 * weights[2] + 4.0 * weights[3] + 8.0 * weights[4] + 4.0 * weights[5]
    if abs(norm - 1.0) > 1e-10:
        return None
    blocked = block_5x5(fine, *weights)
    if not np.all(np.isfinite(blocked)) or np.std(blocked) < 1e-8 or np.max(np.abs(blocked)) > 80.0:
        return None
    obs = per_config_observables(blocked)
    blocked_dop_summary = dop_summary_from_obs(obs)
    blocked_boot = dop_bootstrap_from_obs(obs, boot_indices)
    delta = np.array([blocked_dop_summary[k] - direct_dop_summary[k] for k in OPERATORS_DOP], dtype=np.float64)
    delta_boot = blocked_boot - direct_dop_boot
    cov = np.cov(delta_boot, rowvar=False, ddof=1)
    if not np.all(np.isfinite(cov)):
        return None
    tr = float(np.trace(cov))
    reg = cov + eps * tr / len(OPERATORS_DOP) * np.eye(len(OPERATORS_DOP))
    try:
        cinv = np.linalg.inv(reg)
    except np.linalg.LinAlgError:
        cinv = np.linalg.pinv(reg, rcond=1e-12)
    dop = float(0.5 * delta @ cinv @ delta)
    err_delta = np.nanstd(delta_boot, axis=0, ddof=1)
    z_local = np.divide(delta, err_delta, out=np.zeros_like(delta), where=err_delta > 0)
    return {
        "weights": weights,
        "D_op": dop,
        "local_rms_z": float(np.sqrt(np.nanmean(np.square(z_local)))),
        "condition_number_C": float(np.linalg.cond(cov)),
        "condition_number_Creg": float(np.linalg.cond(reg)),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare-n", type=int, default=1000)
    parser.add_argument("--boot-obj", type=int, default=48)
    parser.add_argument("--boot-final", type=int, default=160)
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--maxiter", type=int, default=44)
    parser.add_argument("--maxfev", type=int, default=320)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fine = load_phi(FINE_PATH)
    direct = load_phi(DIRECT_PATH)
    n = min(args.compare_n, len(fine), len(direct))
    fine = fine[:n]
    direct = direct[:n]

    boot_obj = make_boot_indices(n, args.boot_obj, SEED + 1)
    boot_final = make_boot_indices(n, args.boot_final, SEED + 2)
    direct_boot_obj = bootstrap_summaries(direct, boot_obj)
    direct_summary_obj = with_errors(summary_from_phi(direct), direct_boot_obj)
    direct_obs_obj = per_config_observables(direct)
    direct_dop_summary_obj = dop_summary_from_obs(direct_obs_obj)
    direct_dop_boot_obj = dop_bootstrap_from_obs(direct_obs_obj, boot_obj)
    direct_boot_final = bootstrap_summaries(direct, boot_final)
    direct_summary_final = with_errors(summary_from_phi(direct), direct_boot_final)

    starts: list[tuple[str, tuple[float, float, float, float, float]]] = [
        ("identity", (0.0, 0.0, 0.0, 0.0, 0.0)),
        ("lambda1_5x5_etafit", (0.02707808, 0.01396044, 0.01438420, 0.00239109, -0.00696931)),
        ("smooth_positive", (0.06, 0.02, 0.015, 0.004, 0.0)),
        ("edge_heavy", (0.10, -0.01, 0.0, 0.0, 0.0)),
        ("broad_mild", (0.04, 0.018, 0.018, 0.006, 0.001)),
    ]
    rng = np.random.default_rng(SEED)
    for i in range(8):
        starts.append(
            (
                f"random_{i+1}",
                (
                    float(rng.uniform(-0.04, 0.13)),
                    float(rng.uniform(-0.04, 0.08)),
                    float(rng.uniform(-0.03, 0.07)),
                    float(rng.uniform(-0.02, 0.035)),
                    float(rng.uniform(-0.025, 0.025)),
                ),
            )
        )

    cache: dict[tuple[float, float, float, float, float], float] = {}
    eval_cache: dict[tuple[float, float, float, float, float], dict[str, Any]] = {}

    def objective(x: np.ndarray) -> float:
        key = tuple(round(float(v), 10) for v in x)
        if key in cache:
            return cache[key]
        result = evaluate_kernel_fast(fine, direct_dop_summary_obj, direct_dop_boot_obj, boot_obj, x, args.eps)
        if result is None:
            cache[key] = 1e12
            return 1e12
        cache[key] = float(result["D_op"])
        eval_cache[key] = result
        return cache[key]

    bounds = [(-0.18, 0.28), (-0.16, 0.16), (-0.12, 0.18), (-0.08, 0.08), (-0.08, 0.08)]
    log_rows: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for label, x0 in starts:
        res = minimize(
            objective,
            np.array(x0, dtype=np.float64),
            method="Powell",
            bounds=bounds,
            options={"maxiter": args.maxiter, "maxfev": args.maxfev, "xtol": 5e-4, "ftol": 1e-4, "disp": False},
        )
        x = np.array(res.x, dtype=np.float64)
        result = evaluate_kernel_fast(fine, direct_dop_summary_obj, direct_dop_boot_obj, boot_obj, x, args.eps)
        if result is None:
            continue
        w = result["weights"]
        row = {
            "start_label": label,
            "success": bool(res.success),
            "message": str(res.message),
            "nfev": int(res.nfev),
            "nit": int(res.nit),
            "final_fun": float(res.fun),
            "w00": w[0],
            "w10": w[1],
            "w11": w[2],
            "w20": w[3],
            "w21": w[4],
            "w22": w[5],
            "D_op": result["D_op"],
            "local_rms_z": result["local_rms_z"],
            "ir_rms_z": "",
            "condition_number_C": result["condition_number_C"],
            "condition_number_Creg": result["condition_number_Creg"],
        }
        log_rows.append(row)
        candidates.append({"label": label, "x": x, **result})

    if not candidates:
        raise RuntimeError("no valid kernel candidates")
    best = min(candidates, key=lambda r: float(r["D_op"]))
    best_final = evaluate_kernel(fine, direct_summary_final, direct_boot_final, boot_final, np.array(best["x"]), args.eps)
    if best_final is None:
        raise RuntimeError("final evaluation failed")
    best_weights = best_final["weights"]
    best_blocked = block_5x5(fine, *best_weights)
    blocked_summary_final = with_errors(summary_from_phi(best_blocked), bootstrap_summaries(best_blocked, boot_final))
    best_final["blocked_summary"] = blocked_summary_final
    best_final["direct_summary"] = direct_summary_final

    write_csv(
        OUT_DIR / "optimization_log.csv",
        log_rows,
        [
            "start_label",
            "success",
            "message",
            "nfev",
            "nit",
            "final_fun",
            "w00",
            "w10",
            "w11",
            "w20",
            "w21",
            "w22",
            "D_op",
            "local_rms_z",
            "ir_rms_z",
            "condition_number_C",
            "condition_number_Creg",
        ],
    )

    op_rows: list[dict[str, Any]] = []
    contrib = np.asarray(best_final["cov_contrib"], dtype=np.float64)
    for i, op in enumerate(OPERATORS_REPORT):
        delta = blocked_summary_final[op] - direct_summary_final[op]
        op_rows.append(
            {
                "operator": op,
                "direct_L16_value": direct_summary_final[op],
                "direct_L16_error": direct_summary_final.get(f"{op}_err", 0.0),
                "blocked_L32_to_L16_value": blocked_summary_final[op],
                "blocked_L32_to_L16_error": blocked_summary_final.get(f"{op}_err", 0.0),
                "difference": delta,
                "z_score": zscore(delta, blocked_summary_final.get(f"{op}_err", 0.0), direct_summary_final.get(f"{op}_err", 0.0)),
                "included_in_D_op": op in OPERATORS_DOP,
                "covariance_contribution": float(contrib[OPERATORS_DOP.index(op)]) if op in OPERATORS_DOP else "",
            }
        )
    write_csv(
        OUT_DIR / "operator_matching_table.csv",
        op_rows,
        [
            "operator",
            "direct_L16_value",
            "direct_L16_error",
            "blocked_L32_to_L16_value",
            "blocked_L32_to_L16_error",
            "difference",
            "z_score",
            "included_in_D_op",
            "covariance_contribution",
        ],
    )

    np.savez_compressed(
        OUT_DIR / "blocked_lam0p022_kappa0p2705_L32_to_L16_kernel5x5_fixedeta.npz",
        phi=best_blocked.astype(np.float32),
        source_fine_path=np.array(str(FINE_PATH)),
        source_reference_path=np.array(str(DIRECT_PATH)),
        lambda_value=np.array(LAMBDA_VALUE, dtype=np.float32),
        kappa=np.array(KAPPA, dtype=np.float32),
        eta=np.array(ETA, dtype=np.float32),
        eta_scale=np.array(ETA_SCALE, dtype=np.float32),
        weights=np.array(best_weights, dtype=np.float32),
        kernel_matrix=kernel_matrix(*best_weights).astype(np.float32),
        convention=np.array("psi = 2^(eta/2) K phi; blocked field = even-even sublattice; no four-sublattice average"),
    )

    fft_stats = kernel_fourier_min_abs(best_weights, size=L_FINE)
    coeff = {
        "lambda": LAMBDA_VALUE,
        "kappa_cr": KAPPA,
        "eta": ETA,
        "eta_fixed": True,
        "eta_scale": ETA_SCALE,
        "normalization": "sum K = 1 before eta scaling",
        "convention": "psi = 2^(eta/2) K phi; blocked field = psi[:,0::2,0::2]; no four-sublattice average",
        "weights_shells": {
            "w00": best_weights[0],
            "w10": best_weights[1],
            "w11": best_weights[2],
            "w20": best_weights[3],
            "w21": best_weights[4],
            "w22": best_weights[5],
        },
        "normalization_check": best_weights[0] + 4.0 * best_weights[1] + 4.0 * best_weights[2] + 4.0 * best_weights[3] + 8.0 * best_weights[4] + 4.0 * best_weights[5],
        "kernel_5x5": kernel_matrix(*best_weights),
        "eta_scaled_kernel_5x5": ETA_SCALE * kernel_matrix(*best_weights),
        **fft_stats,
    }
    (OUT_DIR / "kernel_coefficients.json").write_text(json.dumps(to_jsonable(coeff), indent=2) + "\n")

    summary = {
        "status": "complete",
        "selected_kernel_file": str(OUT_DIR / "kernel_coefficients.json"),
        "source_fine_ensemble": str(FINE_PATH),
        "source_direct_ensemble": str(DIRECT_PATH),
        "lambda": LAMBDA_VALUE,
        "kappa_cr": KAPPA,
        "L_fine": L_FINE,
        "L_coarse": L_COARSE,
        "compare_n": n,
        "eta": ETA,
        "eta_fixed": True,
        "eta_scale": ETA_SCALE,
        "convention": coeff["convention"],
        "best_start_label": best["label"],
        "D_op": best_final["D_op"],
        "local_rms_z": best_final["local_rms_z"],
        "ir_rms_z": best_final["ir_rms_z"],
        "condition_number_C": best_final["condition_number_C"],
        "condition_number_Creg": best_final["condition_number_Creg"],
        "weights_shells": coeff["weights_shells"],
        "normalization_check": coeff["normalization_check"],
        "fft_stats": fft_stats,
        "direct_summary": direct_summary_final,
        "blocked_summary": blocked_summary_final,
    }
    (OUT_DIR / "kernel5x5_summary.json").write_text(json.dumps(to_jsonable(summary), indent=2) + "\n")

    report = [
        "# Lambda=0.022 fixed-eta 5x5 kernel optimization",
        "",
        "## Convention",
        "- Real-space 5x5 D4-symmetric kernel.",
        "- `sum K = 1`.",
        f"- `eta = {ETA}` fixed, so `2^(eta/2) = {ETA_SCALE:.16f}`.",
        "- Map: `psi = 2^(eta/2) K phi` on the full periodic L32 lattice.",
        "- Blocked field: `psi[:,0::2,0::2]`; no four-sublattice average.",
        "",
        "## Inputs",
        f"- Fine L32: `{FINE_PATH}`",
        f"- Direct L16: `{DIRECT_PATH}`",
        f"- compare_n: `{n}`",
        "",
        "## Selected Kernel",
        f"- best start: `{best['label']}`",
        f"- D_op: `{float(best_final['D_op']):.6g}`",
        f"- local RMS z: `{float(best_final['local_rms_z']):.6g}`",
        f"- IR RMS z: `{float(best_final['ir_rms_z']):.6g}`",
        f"- min |Ktilde| on L32 grid: `{fft_stats['min_abs_Ktilde']:.8g}`",
        f"- min |K_eta_tilde| on L32 grid: `{fft_stats['min_abs_Keta_tilde']:.8g}`",
        "",
        "Shell weights:",
        f"- w00 = `{best_weights[0]:.12g}`",
        f"- w10 = `{best_weights[1]:.12g}`",
        f"- w11 = `{best_weights[2]:.12g}`",
        f"- w20 = `{best_weights[3]:.12g}`",
        f"- w21 = `{best_weights[4]:.12g}`",
        f"- w22 = `{best_weights[5]:.12g}`",
        f"- normalization check = `{coeff['normalization_check']:.16g}`",
        "",
        "## Files",
        "- `kernel5x5_summary.json`",
        "- `kernel_coefficients.json`",
        "- `optimization_log.csv`",
        "- `operator_matching_table.csv`",
        "- `blocked_lam0p022_kappa0p2705_L32_to_L16_kernel5x5_fixedeta.npz`",
        "",
        "The selected kernel is the fixed-eta real-space even-even kernel for the lambda=0.022 branch.",
    ]
    (OUT_DIR / "report.md").write_text("\n".join(report) + "\n")
    print(json.dumps({"status": "complete", "out_dir": str(OUT_DIR), "D_op": float(best_final["D_op"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
