#!/usr/bin/env python3
r"""Fit the strict two-coupling phi^4 action to Ethan-blocked L32 fields.

The model is exactly S_lat[c | lambda_prime, kappa_prime] in the stored
blocked-field convention.  No field rescaling is permitted.  Direct native
L16 configurations at (lambda,kappa)=(1,0.340301) are importance reweighted;
the fit minimizes D_KL(p_blocked || p_reweighted_direct).  A held-out test
partition checks both reweighting overlap and residual operator mismatch.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "perfect_blocking"))
from scripts.common.blocking import block_configs, load_configs  # noqa: E402
from scripts.common.kernel_io import load_kernel  # noqa: E402
from scripts.run_lam1p0_7x7_kernel_search import observable_arrays  # noqa: E402

LAMBDA0 = 1.0
KAPPA0 = 0.340301
DIRECT = ROOT / "data/configs_phi4_2d/lam1p0_kappac0p340301_L16/configs.npz"
FINE = ROOT / "data/configs_phi4_2d/lam1p0_kappac0p340301_L32/configs.npz"
KERNEL = ROOT / "perfect_blocking/perfect_blocking_lam1p0/kernels/selected_for_upscaling/ethan_7x7_paper_objective_eta_included.json"
OUT = ROOT / "perfect_blocking/perfect_blocking_lam1p0/tests/intermediate/ethan7_strict_two_coupling_blocked_action_20260820"


def cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--direct", type=Path, default=DIRECT)
    p.add_argument("--fine", type=Path, default=FINE)
    p.add_argument("--kernel", type=Path, default=KERNEL)
    p.add_argument("--n-direct-fit", type=int, default=4000)
    p.add_argument("--n-blocked-fit", type=int, default=8000)
    p.add_argument("--n-direct-test", type=int, default=1000)
    p.add_argument("--n-blocked-test", type=int, default=2000)
    p.add_argument("--n-bootstrap", type=int, default=500)
    p.add_argument("--seed", type=int, default=20260820)
    return p.parse_args()


def features(phi: np.ndarray) -> np.ndarray:
    """[phi2, phi4, NN], all action densities in the canonical convention."""
    p2 = np.mean(phi * phi, axis=(1, 2))
    p4 = np.mean(phi**4, axis=(1, 2))
    nn = 0.5 * np.mean(
        phi * np.roll(phi, -1, axis=1) + phi * np.roll(phi, -1, axis=2), axis=(1, 2)
    )
    return np.column_stack((p2, p4, nn))


def action_density(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    lam, kap = params
    return (1.0 - 2.0 * lam) * x[:, 0] + lam * x[:, 1] - 4.0 * kap * x[:, 2]


def weights_from_delta(delta: np.ndarray) -> np.ndarray:
    return np.exp(-delta - logsumexp(-delta))


def fit_params(x_direct: np.ndarray, x_blocked: np.ndarray, volume: int) -> tuple[np.ndarray, float, float]:
    base = action_density(x_direct, np.array([LAMBDA0, KAPPA0]))

    def objective(params: np.ndarray) -> float:
        delta_direct = volume * (action_density(x_direct, params) - base)
        delta_blocked = volume * (action_density(x_blocked, params) - action_density(x_blocked, np.array([LAMBDA0, KAPPA0])))
        return float(logsumexp(-delta_direct) - np.log(len(delta_direct)) + delta_blocked.mean())

    result = minimize(
        objective, np.array([LAMBDA0, KAPPA0]), method="L-BFGS-B",
        bounds=[(0.6, 1.4), (0.30, 0.38)],
        options={"maxiter": 1000, "ftol": 1.0e-13, "gtol": 1.0e-9},
    )
    # Bootstrap resamples occasionally make the finite-difference L-BFGS line
    # search report a precision-loss warning despite a finite near-optimum.
    # A bounded Powell retry is robust in this two-dimensional problem.
    if not result.success:
        retry = minimize(
            objective, result.x, method="Powell", bounds=[(0.6, 1.4), (0.30, 0.38)],
            options={"maxiter": 400, "xtol": 1.0e-9, "ftol": 1.0e-12},
        )
        if not retry.success:
            raise RuntimeError(f"L-BFGS-B: {result.message}; Powell: {retry.message}")
        result = retry
    delta = volume * (action_density(x_direct, result.x) - base)
    w = weights_from_delta(delta)
    return np.asarray(result.x, float), float(result.fun), float(1.0 / np.sum(w * w))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    a = cli(); a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    direct = load_configs(a.direct)
    fine = load_configs(a.fine)
    blocked = block_configs(fine, load_kernel(a.kernel))
    if len(direct) < a.n_direct_fit + a.n_direct_test or len(blocked) < a.n_blocked_fit + a.n_blocked_test:
        raise ValueError("requested fit/test partitions exceed available configurations")
    direct = direct[rng.permutation(len(direct))]
    blocked = blocked[rng.permutation(len(blocked))]
    direct_fit, direct_test = direct[:a.n_direct_fit], direct[a.n_direct_fit:a.n_direct_fit + a.n_direct_test]
    blocked_fit, blocked_test = blocked[:a.n_blocked_fit], blocked[a.n_blocked_fit:a.n_blocked_fit + a.n_blocked_test]
    x_fit, y_fit = features(direct_fit), features(blocked_fit)
    volume = direct.shape[1] ** 2
    params, objective, ess_fit = fit_params(x_fit, y_fit, volume)

    # Independent bootstrap partitions quantify the sampling uncertainty of
    # this KL projection; they do not make the two-coupling ansatz exact.
    draws = np.empty((a.n_bootstrap, 2), float)
    for ib in range(a.n_bootstrap):
        xd = x_fit[rng.integers(len(x_fit), size=len(x_fit))]
        yb = y_fit[rng.integers(len(y_fit), size=len(y_fit))]
        draws[ib], _, _ = fit_params(xd, yb, volume)
    np.save(a.out / "bootstrap_parameters.npy", draws)

    x_test, y_test = features(direct_test), features(blocked_test)
    delta_test = volume * (action_density(x_test, params) - action_density(x_test, np.array([LAMBDA0, KAPPA0])))
    w_test = weights_from_delta(delta_test)
    ess_test = float(1.0 / np.sum(w_test * w_test))
    direct_obs, blocked_obs = observable_arrays(direct_test), observable_arrays(blocked_test)
    rows: list[dict[str, object]] = []
    for name in ("action_density", "phi2", "phi4", "local_kurtosis_ratio", "NN", "2nn", "diag", "m2", "m4", "G_pmin_avg"):
        val = direct_obs[name]
        rw_mean = float(w_test @ val)
        rw_var = float(w_test @ (val - rw_mean) ** 2)
        rows.append({
            "operator": name,
            "blocked_mean": float(blocked_obs[name].mean()),
            "blocked_se": float(blocked_obs[name].std(ddof=1) / np.sqrt(len(blocked_obs[name]))),
            "direct_mean": float(val.mean()),
            "reweighted_mean": rw_mean,
            "reweighted_naive_se": float(np.sqrt(rw_var / ess_test)),
        })
    write_csv(a.out / "heldout_operator_means.csv", rows)
    result = {
        "ansatz": "S_b[c] = S_lat[c | lambda_prime, kappa_prime]; no field rescaling",
        "source_kernel": str(a.kernel), "lambda_prime": float(params[0]), "kappa_prime": float(params[1]),
        "lambda_bootstrap_se": float(draws[:, 0].std(ddof=1)), "kappa_bootstrap_se": float(draws[:, 1].std(ddof=1)),
        "fit_relative_entropy_objective": objective,
        "fit_ess": ess_fit, "fit_ess_fraction": ess_fit / len(x_fit),
        "heldout_ess": ess_test, "heldout_ess_fraction": ess_test / len(x_test),
        "n_direct_fit": len(x_fit), "n_blocked_fit": len(y_fit), "n_direct_test": len(x_test), "n_blocked_test": len(y_test),
        "bootstrap_count": a.n_bootstrap,
    }
    (a.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
