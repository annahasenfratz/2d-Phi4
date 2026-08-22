#!/usr/bin/env python3
"""Fit and validate a compact local correction to the L16 coarse action.

For balanced direct-L16 / blocked-L32 samples, logistic regression estimates
``log[p_blocked(c) / p_direct(c)]``.  Thus a direct L16 configuration is
reweighted to the corrected coarse action by ``w = exp(log_ratio)``.  The
fit and validation configurations are disjoint.
"""
from __future__ import annotations

import csv
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "perfect_blocking/scripts"))
from run_lam1p0_7x7_kernel_search import block, observable_arrays  # noqa: E402
from diagnose_lam1p0_blocking_effective_action_residual import local_features, write_csv  # noqa: E402

DIRECT = ROOT / "data/configs_phi4_2d/lam1p0_kappac0p340301_L16/configs.npz"
FINE = ROOT / "data/configs_phi4_2d/lam1p0_kappac0p340301_L32/configs.npz"
KERNEL = ROOT / "perfect_blocking/perfect_blocking_lam1p0/kernels/candidates/colleague_paper_objective_5x5_eta_included.json"
OUT = ROOT / "perfect_blocking/perfect_blocking_lam1p0/tests/archive_superseded_kernel_explorations_20260818/colleague5_corrected_coarse_action_reweight"

# A deliberately compact, translationally invariant local action basis.  It
# contains the original lattice terms plus the leading residual interactions.
ACTION_BASIS = ["phi2", "phi4", "NN", "diag", "2nn", "phi6", "bond_sq", "phi3_neighbor", "plaquette4"]
TWO_QUARTIC_BASIS = ["bond_sq", "phi3_neighbor"]
REPORT = ["action_density", "phi2", "phi4", "local_kurtosis_ratio", "NN", "diag", "2nn", "m2", "m4", "G_pmin_avg", *ACTION_BASIS]


def weighted_mean_std(x: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    m = float(np.sum(w * x))
    return m, float(np.sqrt(np.sum(w * (x - m) ** 2)))


def rg_invariants(values: dict[str, np.ndarray], indices: np.ndarray, weights: np.ndarray | None = None) -> dict[str, float]:
    if weights is None:
        weights = np.full(len(indices), 1.0 / len(indices))
    m = float(np.sum(weights * values["m"][indices]))
    m2 = float(np.sum(weights * values["m2"][indices]))
    m4 = float(np.sum(weights * values["m4"][indices]))
    gp = float(np.sum(weights * values["G_pmin_avg"][indices]))
    g0 = 16**2 * max(m2 - m*m, 0.0)
    arg = g0 / gp - 1.0 if gp > 0 else float("nan")
    xi = np.sqrt(arg) / (2.0 * 16 * np.sin(np.pi / 16)) if arg > 0 else float("nan")
    return {"Binder_U4": 1.0 - m4 / max(3.0 * m2*m2, 1e-300), "xi_over_L": float(xi), "chi": g0}


def bootstrap_rg_invariants(values: dict[str, np.ndarray], indices: np.ndarray, weights: np.ndarray | None, rng: np.random.Generator, n_boot: int = 500) -> dict[str, float]:
    point = rg_invariants(values, indices, weights)
    draws = np.empty((n_boot, 3), float)
    for b in range(n_boot):
        take = rng.integers(0, len(indices), size=len(indices))
        boot_weights = None if weights is None else weights[take] / np.sum(weights[take])
        inv = rg_invariants(values, indices[take], boot_weights)
        draws[b] = [inv["Binder_U4"], inv["xi_over_L"], inv["chi"]]
    return {**point, "Binder_U4_bootstrap_se": float(draws[:,0].std(ddof=1)),
            "xi_over_L_bootstrap_se": float(draws[:,1].std(ddof=1)),
            "chi_bootstrap_se": float(draws[:,2].std(ddof=1))}


def fit_exponential_tilt(z_direct: np.ndarray, z_blocked: np.ndarray, ridge: float = 1e-4) -> np.ndarray:
    """Fit log w=theta.z so weighted direct moments match blocked moments."""
    target = z_blocked.mean(axis=0)
    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        logw = z_direct @ theta
        logz = logsumexp(logw) - np.log(len(logw))
        w = np.exp(logw - logsumexp(logw))
        loss = logz - target @ theta + .5 * ridge * np.dot(theta, theta)
        grad = w @ z_direct - target + ridge * theta
        return float(loss), grad
    result = minimize(objective, np.zeros(z_direct.shape[1]), jac=True, method="L-BFGS-B", options={"maxiter": 1000})
    if not result.success:
        raise RuntimeError(result.message)
    return result.x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ridge", type=float, default=1e-4)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--basis", choices=("full", "two_quartic"), default="full")
    ap.add_argument("--kernel", type=Path, default=KERNEL)
    args = ap.parse_args()
    out = args.out if args.out.is_absolute() else ROOT / args.out
    kernel = args.kernel if args.kernel.is_absolute() else ROOT / args.kernel
    out.mkdir(parents=True, exist_ok=True)
    action_basis = ACTION_BASIS if args.basis == "full" else TWO_QUARTIC_BASIS
    rng = np.random.default_rng(2026080714)
    with np.load(DIRECT) as z: direct_phi = np.asarray(z["phi"], dtype=np.float64)
    with np.load(FINE) as z: fine_phi = np.asarray(z["phi"], dtype=np.float64)
    matrix = np.asarray(json.loads(kernel.read_text())["matrix"], dtype=np.float64)
    blocked_phi = block(fine_phi[rng.permutation(len(fine_phi))[:len(direct_phi)]], matrix)
    direct, blocked = local_features(direct_phi), local_features(blocked_phi)
    x0 = np.column_stack([direct[k] for k in action_basis])
    x1 = np.column_stack([blocked[k] for k in action_basis])
    n = len(x0); ix0, ix1 = rng.permutation(n), rng.permutation(n)
    n_train = int(.70 * n)
    train0, test0, train1, test1 = ix0[:n_train], ix0[n_train:], ix1[:n_train], ix1[n_train:]
    train_x = np.vstack([x0[train0], x1[train1]])
    train_y = np.r_[np.zeros(n_train), np.ones(n_train)]
    mean, std = train_x.mean(0), np.maximum(train_x.std(0, ddof=1), 1e-12)
    z0_train = (x0[train0] - mean) / std
    z1_train = (x1[train1] - mean) / std
    theta = fit_exponential_tilt(z0_train, z1_train, ridge=args.ridge)
    raw_logw = (x0[test0] - mean) @ (theta / std)
    logw = raw_logw - raw_logw.max()
    w = np.exp(logw); w /= w.sum()
    ess = 1.0 / np.sum(w * w)
    rows = []
    seen: set[str] = set()
    for key in REPORT:
        if key in seen: continue
        seen.add(key)
        a, b = direct[key][test0], blocked[key][test1]
        wm, ws = weighted_mean_std(a, w)
        rows.append({"operator": key, "direct_mean": float(a.mean()), "blocked_mean": float(b.mean()),
                     "reweighted_direct_mean": wm, "direct_std": float(a.std(ddof=1)),
                     "blocked_std": float(b.std(ddof=1)), "reweighted_direct_std": ws,
                     "unweighted_delta": float(a.mean() - b.mean()), "reweighted_delta": float(wm - b.mean())})
    write_csv(out / "heldout_reweighted_operator_comparison.csv", rows)
    # Features are densities.  If the correction is written as
    # Delta S = V sum_i g_i O_i, then log w=-Delta S and g_i carries one
    # additional factor 1/V relative to the fitted log-weight coefficient.
    volume = 16 * 16
    coeff_rows = [{"operator": k, "standardized_log_ratio_coefficient": float(theta[i]),
                   "log_weight_coefficient_for_density": float(theta[i] / std[i]),
                   "action_density_coefficient_deltaS": float(-theta[i] / (volume * std[i]))}
                  for i, k in enumerate(action_basis)]
    coeff_rows.sort(key=lambda r: abs(r["standardized_log_ratio_coefficient"]), reverse=True)
    write_csv(out / "corrected_coarse_action_coefficients.csv", coeff_rows)
    # Correlation validation for the five directions previously emphasized.
    pairs = [("phi2", "NN"), ("phi2", "phi4"), ("phi2", "local_kurtosis_ratio"), ("NN", "diag"), ("action_density", "local_kurtosis_ratio")]
    corr_rows = []
    for a, b in pairs:
        xa, xb = direct[a][test0], direct[b][test0]
        ma, mb = np.sum(w * xa), np.sum(w * xb)
        rho_w = np.sum(w * (xa-ma) * (xb-mb)) / np.sqrt(np.sum(w*(xa-ma)**2) * np.sum(w*(xb-mb)**2))
        corr_rows.append({"pair": f"{a} x {b}", "rho_direct": float(np.corrcoef(xa, xb)[0,1]),
                          "rho_blocked": float(np.corrcoef(blocked[a][test1], blocked[b][test1])[0,1]),
                          "rho_reweighted_direct": float(rho_w)})
    write_csv(out / "heldout_reweighted_correlations.csv", corr_rows)
    invariant_rows = []
    for label, values, indices, weights in [
        ("direct", direct, test0, None),
        ("blocked", blocked, test1, None),
        ("reweighted_direct", direct, test0, w),
    ]:
        invariant_rows.append({"ensemble": label, **rg_invariants(values, indices, weights)})
    write_csv(out / "heldout_rg_invariants.csv", invariant_rows)
    bootstrap_rng = np.random.default_rng(2026080715)
    invariant_bootstrap_rows = []
    for label, values, indices, weights in [
        ("direct", direct, test0, None),
        ("blocked", blocked, test1, None),
        ("reweighted_direct", direct, test0, w),
    ]:
        invariant_bootstrap_rows.append({"ensemble": label, **bootstrap_rg_invariants(values, indices, weights, bootstrap_rng)})
    write_csv(out / "heldout_rg_invariants_bootstrap.csv", invariant_bootstrap_rows)
    fig, ax = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)
    top = coeff_rows[::-1]
    ax.barh([r["operator"] for r in top], [r["standardized_log_ratio_coefficient"] for r in top], color="tab:green")
    ax.axvline(0, color="black", lw=.8); ax.set_xlabel("standardized log[p(blocked)/p(direct)] coefficient")
    ax.set_title(f"Corrected coarse action; held-out ESS/N = {ess / len(test0):.3f}")
    ax.tick_params(direction="in", top=True, right=True)
    fig.savefig(out / "corrected_coarse_action_coefficients.pdf", bbox_inches="tight")
    summary = {"kernel": str(kernel), "action_basis": action_basis, "n_train_per_class": n_train,
               "n_test_per_class": len(test0), "heldout_ess": float(ess), "heldout_ess_fraction": float(ess/len(test0)),
               "log_weight_span": float(np.ptp(raw_logw)), "ridge": args.ridge,
               "fit": "exponential_tilt_moment_matching"}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"out_dir": str(out), **summary, "top_coefficients": coeff_rows[:5]}, indent=2))


if __name__ == "__main__":
    main()
