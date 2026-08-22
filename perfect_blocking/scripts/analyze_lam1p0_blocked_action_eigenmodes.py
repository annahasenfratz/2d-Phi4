#!/usr/bin/env python3
"""Diagonalize and cross-validate the relative-entropy blocked-action fit.

The relative-entropy Hessian is the covariance of standardized local action
operators under the reweighted direct ensemble.  Its eigenvectors are the
independently constrained action combinations.  A train/validation/test rank
scan decides whether the small-eigenvalue directions are supported by data or
are merely overfitting.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[2]
FIT_SCRIPT = ROOT / "perfect_blocking/scripts/fit_lam1p0_blocked_action_relative_entropy.py"
DEFAULT_OUT = ROOT / "perfect_blocking/perfect_blocking_lam1p0/tests/archive_superseded_kernel_explorations_20260818/softcond7_blocked_action_relative_entropy"


def load_fit_module():
    spec = importlib.util.spec_from_file_location("blocked_action_fit", FIT_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {FIT_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def objective(z_direct: np.ndarray, z_blocked: np.ndarray, alpha: np.ndarray) -> float:
    return float(logsumexp(z_direct @ alpha) - np.log(len(z_direct)) - z_blocked.mean(axis=0) @ alpha)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    out = args.out if args.out.is_absolute() else ROOT / args.out
    summary = json.loads((out / "summary.json").read_text())
    coefficients = pd.read_csv(out / "blocked_action_coefficients.csv")
    fit = load_fit_module()
    names = fit.ACTION_BASIS
    alpha_by_name = dict(zip(coefficients["operator"], coefficients["standardized_log_ratio_coefficient"]))
    alpha_full = np.asarray([alpha_by_name[name] for name in names], float)

    rng = np.random.default_rng(20260809)
    n = int(summary["n_train"] + summary["n_validation"] + summary["n_test"])
    direct_phi = fit.load_configs(Path(summary["direct"]))
    fine_phi = fit.load_configs(Path(summary["fine"]))
    direct_phi = direct_phi[rng.permutation(len(direct_phi))[:n]]
    fine_phi = fine_phi[rng.permutation(len(fine_phi))[:n]]
    blocked_phi = fit.block_configs(fine_phi, fit.load_kernel(Path(summary["kernel"])))
    direct = fit.local_action_features(direct_phi)
    blocked = fit.local_action_features(blocked_phi)
    x_direct = np.column_stack([direct[name] for name in names])
    x_blocked = np.column_stack([blocked[name] for name in names])
    n_train, n_val = int(summary["n_train"]), int(summary["n_validation"])
    train = np.arange(n_train); val = np.arange(n_train, n_train + n_val); test = np.arange(n_train + n_val, n)
    fit_indices = np.r_[train, val]
    stacked = np.vstack([x_direct[fit_indices], x_blocked[fit_indices]])
    center = stacked.mean(axis=0)
    scale = np.maximum(stacked.std(axis=0, ddof=1), 1e-12)
    z_direct, z_blocked = (x_direct - center) / scale, (x_blocked - center) / scale

    w_fit, _ = fit.normalized_weights(z_direct[fit_indices], alpha_full)
    mean_fit = w_fit @ z_direct[fit_indices]
    hessian = (z_direct[fit_indices] - mean_fit).T @ ((z_direct[fit_indices] - mean_fit) * w_fit[:, None])
    hessian += float(summary["ridge"]) * np.eye(len(names))
    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]

    mode_rows: list[dict[str, object]] = []
    for i, (value, vector) in enumerate(zip(eigenvalues, eigenvectors.T), start=1):
        row: dict[str, object] = {
            "mode": i, "eigenvalue": float(value),
            "condition_relative_to_max": float(eigenvalues[0] / value),
            "full_fit_projection": float(vector @ alpha_full),
            "approx_single_ensemble_se": float(np.sqrt(1.0 / (len(fit_indices) * value))),
        }
        row.update({name: float(vector[j]) for j, name in enumerate(names)})
        mode_rows.append(row)
    pd.DataFrame(mode_rows).to_csv(out / "relative_entropy_hessian_eigenmodes.csv", index=False)

    scan_rows = []
    for rank in range(len(names) + 1):
        if rank == 0:
            alpha_train = np.zeros(len(names)); alpha_final = alpha_train
        else:
            a_train = fit.fit_tilt(z_direct[train] @ eigenvectors[:, :rank],
                                    (z_blocked[train] @ eigenvectors[:, :rank]).mean(axis=0),
                                    float(summary["ridge"]))
            alpha_train = eigenvectors[:, :rank] @ a_train
            a_final = fit.fit_tilt(z_direct[fit_indices] @ eigenvectors[:, :rank],
                                    (z_blocked[fit_indices] @ eigenvectors[:, :rank]).mean(axis=0),
                                    float(summary["ridge"]))
            alpha_final = eigenvectors[:, :rank] @ a_final
        weights_test, _ = fit.normalized_weights(z_direct[test], alpha_final)
        scan_rows.append({
            "rank": rank,
            "validation_objective": objective(z_direct[val], z_blocked[val], alpha_train),
            "heldout_test_objective": objective(z_direct[test], z_blocked[test], alpha_final),
            "heldout_ess_fraction": fit.effective_sample_size(weights_test) / len(test),
        })
    pd.DataFrame(scan_rows).to_csv(out / "relative_entropy_rank_scan.csv", index=False)

    # A rank prefix is useful for the spectrum, but it does not answer whether
    # a particular low-stiffness direction carries real distributional signal.
    # This leave-one-mode-out table does; choose reductions on validation, then
    # use the test column only as a final confirmation.
    def fit_score(keep: np.ndarray, fit_indices: np.ndarray, eval_indices: np.ndarray) -> float:
        coeff = fit.fit_tilt(z_direct[fit_indices] @ eigenvectors[:, keep],
                             (z_blocked[fit_indices] @ eigenvectors[:, keep]).mean(axis=0),
                             float(summary["ridge"]))
        return objective(z_direct[eval_indices], z_blocked[eval_indices], eigenvectors[:, keep] @ coeff)
    full_modes = np.arange(len(names))
    full_validation = fit_score(full_modes, train, val)
    full_test = fit_score(full_modes, fit_indices, test)
    leave_rows = []
    for dropped in range(len(names)):
        keep = np.asarray([mode for mode in full_modes if mode != dropped])
        leave_rows.append({
            "dropped_mode": dropped + 1,
            "eigenvalue": float(eigenvalues[dropped]),
            "validation_objective_increase": fit_score(keep, train, val) - full_validation,
            "heldout_test_objective_increase": fit_score(keep, fit_indices, test) - full_test,
        })
    pd.DataFrame(leave_rows).to_csv(out / "relative_entropy_leave_one_mode_out.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8), constrained_layout=True)
    axes[0].semilogy(np.arange(1, len(names) + 1), eigenvalues, "o-")
    axes[0].set(xlabel="eigenmode rank", ylabel="Hessian eigenvalue", title="Relative-entropy stiffness spectrum")
    axes[1].plot([r["rank"] for r in scan_rows], [r["validation_objective"] for r in scan_rows], "o-", label="validation")
    axes[1].plot([r["rank"] for r in scan_rows], [r["heldout_test_objective"] for r in scan_rows], "s-", label="held-out test")
    axes[1].set(xlabel="retained eigenmodes", ylabel="relative-entropy objective difference", title="Rank selection")
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.tick_params(direction="in", top=True, right=True)
    fig.savefig(out / "relative_entropy_eigenmode_rank_scan.pdf", bbox_inches="tight")

    print(json.dumps({"out": str(out), "condition_number": float(eigenvalues[0] / eigenvalues[-1]),
                      "best_validation_rank": int(min(scan_rows, key=lambda r: r["validation_objective"])["rank"]),
                      "best_test_rank": int(min(scan_rows, key=lambda r: r["heldout_test_objective"])["rank"])}, indent=2))


if __name__ == "__main__":
    main()
