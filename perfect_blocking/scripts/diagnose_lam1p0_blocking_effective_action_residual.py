#!/usr/bin/env python3
"""Rank local operators missing from the L32->L16 blocked marginal.

Fits a regularized density-ratio classifier between direct L16 fields and
L32 fields blocked with a supplied kernel.  The coefficients identify the
operator directions that remain distinguishable after blocking; they are a
diagnostic, not yet a modification of the MH coarse action.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "perfect_blocking/scripts"))
from run_lam1p0_7x7_kernel_search import block, observable_arrays  # noqa: E402

DIRECT = ROOT / "data/configs_phi4_2d/lam1p0_kappac0p340301_L16/configs.npz"
FINE = ROOT / "data/configs_phi4_2d/lam1p0_kappac0p340301_L32/configs.npz"
KERNEL = ROOT / "perfect_blocking/perfect_blocking_lam1p0/kernels/candidates/colleague_paper_objective_5x5_eta_included.json"
OUT = ROOT / "perfect_blocking/perfect_blocking_lam1p0/tests/archive_superseded_kernel_explorations_20260818/colleague5_effective_action_residual"


def local_features(phi: np.ndarray) -> dict[str, np.ndarray]:
    o = observable_arrays(phi)
    px, py = np.roll(phi, -1, axis=1), np.roll(phi, -1, axis=2)
    bond_x, bond_y = phi * px, phi * py
    vals = {
        "action_density": o["action_density"], "local_kurtosis_ratio": o["local_kurtosis_ratio"],
        "phi2": o["phi2"], "phi4": o["phi4"],
        "phi6": np.mean(phi**6, axis=(1, 2)), "phi8": np.mean(phi**8, axis=(1, 2)),
        "NN": o["NN"], "diag": o["diag"], "2nn": o["2nn"],
        "m": o["m"], "m2": o["m2"], "m4": o["m4"], "G_pmin": o["G_pmin_avg"], "G_pmin_avg": o["G_pmin_avg"],
        "bond_sq": 0.5 * np.mean(bond_x**2 + bond_y**2, axis=(1, 2)),
        "plaquette4": np.mean(phi * px * py * np.roll(px, -1, axis=2), axis=(1, 2)),
        "phi3_neighbor": 0.5 * np.mean(phi**3 * (px + py), axis=(1, 2)),
        # Ensemble-level products capture the correlation directions used in
        # the highcorr kernel objective.
        "phi2_x_NN": o["phi2"] * o["NN"],
        "phi2_x_phi4": o["phi2"] * o["phi4"],
        "NN_x_diag": o["NN"] * o["diag"],
        "action_x_kurtosis": o["action_density"] * o["local_kurtosis_ratio"],
    }
    return vals


def fit_logistic(x: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    # Intercept is intentionally unpenalized; all features are standardized.
    def fun(theta: np.ndarray) -> tuple[float, np.ndarray]:
        z = theta[0] + x @ theta[1:]
        p = expit(z)
        loss = np.mean(np.logaddexp(0.0, z) - y * z) + 0.5 * ridge * np.dot(theta[1:], theta[1:])
        grad = np.empty_like(theta)
        grad[0] = np.mean(p - y)
        grad[1:] = x.T @ (p - y) / len(y) + ridge * theta[1:]
        return float(loss), grad
    result = minimize(fun, np.zeros(x.shape[1] + 1), jac=True, method="L-BFGS-B", options={"maxiter": 300})
    if not result.success:
        raise RuntimeError(result.message)
    return result.x


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", type=Path, default=KERNEL)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--ridge", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=2026080713)
    args = ap.parse_args()
    kernel = args.kernel if args.kernel.is_absolute() else ROOT / args.kernel
    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    with np.load(DIRECT) as z: direct_phi = np.asarray(z["phi"], dtype=np.float64)
    with np.load(FINE) as z: fine_phi = np.asarray(z["phi"], dtype=np.float64)
    matrix = np.asarray(json.loads(kernel.read_text())["matrix"], dtype=np.float64)
    rng = np.random.default_rng(args.seed)
    blocked_phi = block(fine_phi[rng.permutation(len(fine_phi))[:len(direct_phi)]], matrix)
    d, b = local_features(direct_phi), local_features(blocked_phi)
    names = list(d)
    x = np.vstack([np.column_stack([d[k] for k in names]), np.column_stack([b[k] for k in names])])
    y = np.r_[np.zeros(len(direct_phi)), np.ones(len(blocked_phi))]
    perm = rng.permutation(len(y)); split = int(.70 * len(y)); train, test = perm[:split], perm[split:]
    mean, std = x[train].mean(0), x[train].std(0, ddof=1)
    std = np.maximum(std, 1e-12)
    theta = fit_logistic((x[train] - mean) / std, y[train], args.ridge)
    score = theta[0] + (x[test] - mean) @ (theta[1:] / std)
    # AUC via Mann-Whitney ranks, with no optional dependency.
    order = np.argsort(score); ranks = np.empty(len(order), float); ranks[order] = np.arange(1, len(order) + 1)
    n1 = int(y[test].sum()); n0 = len(y[test]) - n1
    auc = (ranks[y[test] == 1].sum() - n1 * (n1 + 1) / 2) / (n0 * n1)
    rows = []
    for i, name in enumerate(names):
        rows.append({"operator": name, "direct_mean": float(d[name].mean()), "blocked_mean": float(b[name].mean()),
                     "direct_std": float(d[name].std(ddof=1)), "blocked_std": float(b[name].std(ddof=1)),
                     "standardized_log_density_ratio_coefficient": float(theta[i + 1]),
                     "abs_coefficient": float(abs(theta[i + 1]))})
    rows.sort(key=lambda r: r["abs_coefficient"], reverse=True)
    write_csv(out / "ranked_residual_operators.csv", rows)
    fig, ax = plt.subplots(figsize=(8.0, 5.2), constrained_layout=True)
    show = rows[:14][::-1]
    ax.barh([r["operator"] for r in show], [r["standardized_log_density_ratio_coefficient"] for r in show], color="tab:purple")
    ax.axvline(0, color="black", lw=.8); ax.set_xlabel("standardized density-ratio coefficient (blocked / direct)")
    ax.set_title(f"Residual blocked-action directions; held-out AUC = {auc:.3f}")
    ax.tick_params(direction="in", top=True, right=True)
    fig.savefig(out / "ranked_residual_operators.pdf", bbox_inches="tight")
    (out / "summary.json").write_text(json.dumps({"kernel": str(kernel), "n_per_class": len(direct_phi), "ridge": args.ridge, "held_out_auc": float(auc), "operators": names}, indent=2) + "\n")
    print(json.dumps({"out_dir": str(out), "held_out_auc": float(auc), "top_operators": rows[:6]}, indent=2))


if __name__ == "__main__":
    main()
