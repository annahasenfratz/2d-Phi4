#!/usr/bin/env python3
r"""Infer a local approximation to the L32->L16 blocked action.

For direct coarse configurations distributed as p_c \propto exp(-S_c), and
native fine configurations blocked to fields distributed as p_b, this fits

    S_b = S_c + Delta S,        Delta S = V sum_i g_i O_i.

The fit minimizes D_KL(p_b || p_c exp(-Delta S)/Z), equivalently matching the
blocked means of the action-basis operators to reweighted direct-L16 means.
The direct and blocked samples are split before fitting; the final test set is
never used to determine the couplings.  The additive constant in Delta S is
unidentifiable and intentionally omitted.
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
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "perfect_blocking"))
from scripts.common.blocking import block_configs, load_configs  # noqa: E402
from scripts.common.kernel_io import load_kernel  # noqa: E402
from scripts.run_lam1p0_7x7_kernel_search import observable_arrays  # noqa: E402

DIRECT_DEFAULT = ROOT / "data/configs_phi4_2d/lam1p0_kappac0p340301_L16/configs.npz"
FINE_DEFAULT = ROOT / "data/configs_phi4_2d/lam1p0_kappac0p340301_L32/configs.npz"
KERNEL_DEFAULT = ROOT / "perfect_blocking/perfect_blocking_lam1p0/kernels/candidates/allL16_chi2_R3_soft_conditioned_7x7_eta_included.json"
OUT_DEFAULT = ROOT / "perfect_blocking/perfect_blocking_lam1p0/tests/softcond7_blocked_action_relative_entropy"

# These are local, extensive-action densities.  The first six include the
# usual phi^4 action and its leading local extensions; the last four are the
# simplest Z2-even mixed interactions generated under blocking.
ACTION_BASIS = [
    "phi2", "phi4", "phi6", "NN", "diag", "2nn",
    "bond_sq", "phi3_neighbor", "phi2_neighbor", "plaquette4",
]
REPORT_BASIS = [
    "action_density", "phi2", "phi4", "phi6", "phi8",
    "local_kurtosis_ratio", "NN", "diag", "2nn", "3nn",
    "bond_sq", "phi3_neighbor", "phi2_neighbor", "plaquette4",
    "m2", "m4", "G_pmin_avg",
]
# Candidate operators for a *screened* extension of the action basis.  They
# are not included in the default fit: each must earn its place on a disjoint
# validation split.
EXTENSION_CANDIDATES = [
    "phi8", "3nn", "diag_bond_sq", "2nn_bond_sq", "diag_phi3_neighbor",
    "2nn_phi3_neighbor", "diag_phi2_neighbor", "2nn_phi2_neighbor", "rectangle4",
]
HIGH_FIELD_DIAGONAL_CANDIDATES = [
    "diag_phi4_neighbor", "diag_phi3phi3", "diag_phi2_bond_sq",
]
MULTISITE_CANDIDATES = ["corner211", "corner411", "corner222", "plaquette6", "corr_21", "corr_22", "corr_31"]


def local_action_features(phi: np.ndarray) -> dict[str, np.ndarray]:
    """Translation- and D4-averaged local densities for one ensemble."""
    out = dict(observable_arrays(phi))
    px, py = np.roll(phi, -1, axis=1), np.roll(phi, -1, axis=2)
    p2x, p2y = np.roll(phi, -2, axis=1), np.roll(phi, -2, axis=2)
    dplus, dminus = np.roll(np.roll(phi, -1, axis=1), -1, axis=2), np.roll(np.roll(phi, -1, axis=1), 1, axis=2)
    mx, my = np.roll(phi, 1, axis=1), np.roll(phi, 1, axis=2)
    corner = 0.25 * (px*py + px*my + mx*py + mx*my)
    def orbit_corr(a: int, b: int) -> np.ndarray:
        shifts = {(a,b), (a,-b), (-a,b), (-a,-b), (b,a), (b,-a), (-b,a), (-b,-a)}
        return sum(phi * np.roll(np.roll(phi, -dx, axis=1), -dy, axis=2) for dx,dy in shifts) / len(shifts)
    p3x, p3y = np.roll(phi, -3, axis=1), np.roll(phi, -3, axis=2)
    bond_x, bond_y = phi * px, phi * py
    out.update({
        "phi6": np.mean(phi**6, axis=(1, 2)),
        "phi8": np.mean(phi**8, axis=(1, 2)),
        "3nn": 0.5 * np.mean(phi * p3x + phi * p3y, axis=(1, 2)),
        "bond_sq": 0.5 * np.mean(bond_x**2 + bond_y**2, axis=(1, 2)),
        # The translation sum symmetrizes phi_x^3 phi_y automatically.
        "phi3_neighbor": 0.5 * np.mean(phi**3 * (px + py), axis=(1, 2)),
        "phi2_neighbor": 0.5 * np.mean((phi**2 + px**2) * phi * px + (phi**2 + py**2) * phi * py, axis=(1, 2)),
        "plaquette4": np.mean(phi * px * py * np.roll(px, -1, axis=2), axis=(1, 2)),
        "diag_bond_sq": 0.5 * np.mean((phi * dplus)**2 + (phi * dminus)**2, axis=(1, 2)),
        "2nn_bond_sq": 0.5 * np.mean((phi * p2x)**2 + (phi * p2y)**2, axis=(1, 2)),
        "diag_phi3_neighbor": 0.5 * np.mean(phi**3 * (dplus + dminus), axis=(1, 2)),
        "2nn_phi3_neighbor": 0.5 * np.mean(phi**3 * (p2x + p2y), axis=(1, 2)),
        "diag_phi2_neighbor": 0.5 * np.mean((phi**2 + dplus**2) * phi * dplus + (phi**2 + dminus**2) * phi * dminus, axis=(1, 2)),
        "2nn_phi2_neighbor": 0.5 * np.mean((phi**2 + p2x**2) * phi * p2x + (phi**2 + p2y**2) * phi * p2y, axis=(1, 2)),
        "rectangle4": 0.5 * np.mean(
            phi * p2x * py * np.roll(p2x, -1, axis=2)
            + phi * px * p2y * np.roll(p2y, -1, axis=1), axis=(1, 2)),
        # Independent Z2-even degree-six diagonal interactions.  These target
        # the remaining phi6--diagonal correlation residuals directly.
        "diag_phi4_neighbor": 0.5 * np.mean(phi**5 * (dplus + dminus), axis=(1, 2)),
        "diag_phi3phi3": 0.5 * np.mean((phi * dplus)**3 + (phi * dminus)**3, axis=(1, 2)),
        "diag_phi2_bond_sq": 0.5 * np.mean((phi**2 + dplus**2) * (phi * dplus)**2 + (phi**2 + dminus**2) * (phi * dminus)**2, axis=(1, 2)),
        # D4-averaged three-site corners and a field-decorated plaquette.
        "corner211": np.mean(phi**2 * corner, axis=(1, 2)),
        "corner411": np.mean(phi**4 * corner, axis=(1, 2)),
        "corner222": np.mean(phi**2 * 0.25 * (px**2*py**2 + px**2*my**2 + mx**2*py**2 + mx**2*my**2), axis=(1, 2)),
        "plaquette6": np.mean(phi * px * py * np.roll(px, -1, axis=2) * (phi**2 + px**2 + py**2 + np.roll(px, -1, axis=2)**2) / 4.0, axis=(1, 2)),
        "corr_21": np.mean(orbit_corr(2, 1), axis=(1, 2)),
        "corr_22": np.mean(orbit_corr(2, 2), axis=(1, 2)),
        "corr_31": np.mean(orbit_corr(3, 1), axis=(1, 2)),
    })
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def normalized_weights(x: np.ndarray, alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    logw = x @ alpha
    logw -= logsumexp(logw)
    return np.exp(logw), logw


def fit_tilt(x_direct: np.ndarray, target: np.ndarray, ridge: float) -> np.ndarray:
    """Minimize relative entropy in standardized feature coordinates."""
    def objective(alpha: np.ndarray) -> tuple[float, np.ndarray]:
        logw = x_direct @ alpha
        norm = logsumexp(logw) - np.log(len(logw))
        w = np.exp(logw - logsumexp(logw))
        # alpha is log[p_b/p_c] in standardized coordinates.
        return (
            float(norm - alpha @ target + 0.5 * ridge * (alpha @ alpha)),
            w @ x_direct - target + ridge * alpha,
        )
    result = minimize(objective, np.zeros(x_direct.shape[1]), jac=True,
                      method="L-BFGS-B", options={"maxiter": 1000, "ftol": 1e-12})
    if not result.success:
        raise RuntimeError(f"relative-entropy fit failed: {result.message}")
    return np.asarray(result.x, float)


def effective_sample_size(w: np.ndarray) -> float:
    return float(1.0 / np.sum(w * w))


def weighted_mean_se(x: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    mean = float(w @ x)
    var = float(w @ (x - mean)**2)
    return mean, float(np.sqrt(var / max(effective_sample_size(w), 1.0)))


def fit_with_scaling(direct: np.ndarray, blocked: np.ndarray, ridge: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stacked = np.vstack([direct, blocked])
    center = stacked.mean(axis=0)
    scale = np.maximum(stacked.std(axis=0, ddof=1), 1e-12)
    alpha = fit_tilt((direct - center) / scale, ((blocked - center) / scale).mean(axis=0), ridge)
    return alpha, center, scale


def bootstrap_coefficients(direct: np.ndarray, blocked: np.ndarray, ridge: float,
                           n_boot: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    draws = np.empty((n_boot, direct.shape[1]), float)
    for b in range(n_boot):
        xd = direct[rng.integers(len(direct), size=len(direct))]
        xb = blocked[rng.integers(len(blocked), size=len(blocked))]
        alpha, center, scale = fit_with_scaling(xd, xb, ridge)
        draws[b] = alpha / scale
    return draws


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--direct", type=Path, default=DIRECT_DEFAULT)
    ap.add_argument("--fine", type=Path, default=FINE_DEFAULT)
    ap.add_argument("--kernel", type=Path, default=KERNEL_DEFAULT)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--n-configs", type=int, default=5000)
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--n-validation", type=int, default=1000)
    ap.add_argument("--n-bootstrap", type=int, default=250)
    ap.add_argument("--ridge", type=float, default=1e-4)
    ap.add_argument("--min-ess-fraction", type=float, default=0.10)
    ap.add_argument("--extra-action-operators", default="", help="comma-separated screened additions to the default action basis")
    ap.add_argument("--seed", type=int, default=20260809)
    args = ap.parse_args()
    if args.n_train + args.n_validation >= args.n_configs:
        raise ValueError("n_train + n_validation must leave a nonempty held-out test set")

    extras = [name for name in args.extra_action_operators.split(",") if name]
    allowed_extensions = set(EXTENSION_CANDIDATES) | set(HIGH_FIELD_DIAGONAL_CANDIDATES) | set(MULTISITE_CANDIDATES)
    unknown = set(extras) - allowed_extensions
    if unknown:
        raise ValueError(f"unknown extension operators: {sorted(unknown)}")
    action_basis = list(ACTION_BASIS) + extras
    report_basis = list(dict.fromkeys(REPORT_BASIS + extras))
    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    direct_path = args.direct if args.direct.is_absolute() else ROOT / args.direct
    fine_path = args.fine if args.fine.is_absolute() else ROOT / args.fine
    kernel_path = args.kernel if args.kernel.is_absolute() else ROOT / args.kernel
    rng = np.random.default_rng(args.seed)
    direct_phi = load_configs(direct_path)
    fine_phi = load_configs(fine_path)
    n = min(args.n_configs, len(direct_phi), len(fine_phi))
    if n < args.n_train + args.n_validation + 1:
        raise ValueError(f"only {n} configurations available for requested split")
    direct_phi = direct_phi[rng.permutation(len(direct_phi))[:n]]
    fine_phi = fine_phi[rng.permutation(len(fine_phi))[:n]]
    kernel = load_kernel(kernel_path)
    blocked_phi = block_configs(fine_phi, kernel)
    direct = local_action_features(direct_phi)
    blocked = local_action_features(blocked_phi)
    volume = direct_phi.shape[1] ** 2
    x_direct = np.column_stack([direct[k] for k in action_basis])
    x_blocked = np.column_stack([blocked[k] for k in action_basis])

    split_train = args.n_train
    split_val = split_train + args.n_validation
    train, val, test = np.arange(split_train), np.arange(split_train, split_val), np.arange(split_val, n)
    alpha_train, center_train, scale_train = fit_with_scaling(x_direct[train], x_blocked[train], args.ridge)
    # Refit after validation allocation is fixed; the test configurations remain untouched.
    fit = np.r_[train, val]
    alpha, center, scale = fit_with_scaling(x_direct[fit], x_blocked[fit], args.ridge)
    beta = alpha / scale                 # log[p_b/p_c] coefficient for each density
    delta_g = -beta / volume             # Delta S = V sum delta_g O
    boot_beta = bootstrap_coefficients(x_direct[fit], x_blocked[fit], args.ridge, args.n_bootstrap, args.seed + 1)
    boot_delta_g = -boot_beta / volume

    z_test = (x_direct[test] - center) / scale
    w_test, logw_test = normalized_weights(z_test, alpha)
    ess = effective_sample_size(w_test)
    warnings: list[str] = []
    if ess / len(test) < args.min_ess_fraction:
        warnings.append("held-out effective sample size is below requested guard; one-step reweighting has poor overlap")

    coeff_rows = []
    for i, name in enumerate(action_basis):
        coeff_rows.append({
            "operator": name,
            "deltaS_density_coefficient": float(delta_g[i]),
            "deltaS_density_bootstrap_se": float(boot_delta_g[:, i].std(ddof=1)),
            "log_ratio_coefficient_for_density": float(beta[i]),
            "log_ratio_bootstrap_se": float(boot_beta[:, i].std(ddof=1)),
            "standardized_log_ratio_coefficient": float(alpha[i]),
        })
    coeff_rows.sort(key=lambda row: abs(float(row["standardized_log_ratio_coefficient"])), reverse=True)
    write_csv(out / "blocked_action_coefficients.csv", coeff_rows)

    report_rows = []
    for name in report_basis:
        xd, xb = direct[name][test], blocked[name][test]
        reweighted_mean, reweighted_se = weighted_mean_se(xd, w_test)
        report_rows.append({
            "operator": name,
            "direct_mean": float(xd.mean()), "direct_se": float(xd.std(ddof=1) / np.sqrt(len(xd))),
            "blocked_mean": float(xb.mean()), "blocked_se": float(xb.std(ddof=1) / np.sqrt(len(xb))),
            "reweighted_mean": reweighted_mean, "reweighted_naive_se": reweighted_se,
            "direct_minus_blocked": float(xd.mean() - xb.mean()),
            "reweighted_minus_blocked": float(reweighted_mean - xb.mean()),
        })
    write_csv(out / "heldout_operator_comparison.csv", report_rows)

    # The fit basis should agree by construction on the independent test set;
    # the broad report basis tests what the local ansatz predicts beyond it.
    fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
    shown = coeff_rows[::-1]
    ax.errorbar([r["deltaS_density_coefficient"] for r in shown], range(len(shown)),
                xerr=[r["deltaS_density_bootstrap_se"] for r in shown], fmt="o", color="C3", capsize=2)
    ax.axvline(0.0, color="black", lw=0.8)
    ax.set_yticks(range(len(shown)), [str(r["operator"]) for r in shown])
    ax.set_xlabel(r"$g_i$ in $\Delta S=V\sum_i g_i O_i$")
    ax.set_title(f"Blocked-action relative-entropy fit; held-out ESS/N = {ess / len(test):.3f}")
    ax.tick_params(direction="in", top=True, right=True)
    fig.savefig(out / "blocked_action_coefficients.pdf", bbox_inches="tight")
    plt.close(fig)

    summary = {
        "method": "relative_entropy_exponential_tilt",
        "meaning": "p_blocked/p_direct = exp[-DeltaS]/Z; DeltaS = V sum_i g_i O_i",
        "kernel": str(kernel_path), "direct": str(direct_path), "fine": str(fine_path),
        "action_basis": action_basis, "report_basis": report_basis,
        "n_train": len(train), "n_validation": len(val), "n_fit": len(fit), "n_test": len(test),
        "n_bootstrap": args.n_bootstrap, "ridge": args.ridge,
        "heldout_ess": ess, "heldout_ess_fraction": ess / len(test),
        "heldout_log_weight_span": float(np.ptp(logw_test)), "warnings": warnings,
        "train_only_standardized_coefficients": alpha_train.tolist(),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"out_dir": str(out), "heldout_ess_fraction": ess / len(test),
                      "warnings": warnings, "top_terms": coeff_rows[:5]}, indent=2))


if __name__ == "__main__":
    main()
