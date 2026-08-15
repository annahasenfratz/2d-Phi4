#!/usr/bin/env python3
"""Stage 4A: reweight the direct L16 ensemble in a diagonal coupling."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "perfect_blocking/scripts"))
from run_coarse_action_reweight_kappa_lambda import (  # noqa: E402
    BLOCKED,
    DIRECT,
    KAPPA0,
    LAMBDA0,
    evaluate,
    load_csv,
    trial_observables,
    write_rows,
)

OUT = ROOT / "perfect_blocking/perfect_blocking_lam1p0/coarse_action_reweight_kappa_lambda_20260720"
DIRECT_CONFIGS = ROOT / "data/configs_phi4_2d/lam1p0_kappac0p340301_L16/configs.npz"


def diagonal_two_orientation_density(phi: np.ndarray) -> np.ndarray:
    """Return O_diag/V with each +(+x,+y) and +(+x,-y) bond counted once."""
    a = np.asarray(phi, dtype=float)
    return 0.5 * (
        np.mean(a * np.roll(np.roll(a, -1, axis=1), -1, axis=2), axis=(1, 2))
        + np.mean(a * np.roll(np.roll(a, -1, axis=1), 1, axis=2), axis=(1, 2))
    )


def evaluate_diag(d, b, diag, kdiag):
    target = trial_observables(b, KAPPA0, LAMBDA0)
    trial = trial_observables(d, KAPPA0, LAMBDA0)
    trial["diag_two_orientation"] = diag
    trial["action_density"] = trial["action_density"] - 2.0 * kdiag * diag
    logw = -2.0 * kdiag * diag * 256.0
    logw -= np.max(logw)
    w = np.exp(logw)
    w /= w.sum()
    out = {"kappa_NN": KAPPA0, "kappa_diag": kdiag, "delta_kappa_diag": kdiag, "lambda": LAMBDA0, "ESS": 1.0 / np.sum(w*w), "ESS_fraction": 1.0 / (np.sum(w*w)*len(w)), "max_log_weight_shift": float(np.max(logw)-np.min(logw))}
    keys = ["PC1", "PC2", "action_density", "phi2", "phi4", "local_kurtosis_ratio", "NN", "diag", "2nn", "G_pmin_x_cfg", "G_pmin_y_cfg"]
    for key in keys:
        x = trial[key]
        y = target[key]
        mu = float(np.sum(w*x)); sd = float(np.sqrt(np.sum(w*(x-mu)**2)))
        bm = float(np.mean(y)); bsd = float(np.std(y))
        out[f"{key}_mean"] = mu; out[f"{key}_std"] = sd
        out[f"{key}_mean_shift_blocked_sigma"] = (mu-bm)/max(bsd, 1e-300)
        out[f"{key}_std_ratio"] = sd/max(bsd, 1e-300)
        from run_coarse_action_reweight_kappa_lambda import hist_metrics, weighted_ks, weighted_quantile
        out[f"{key}_KS"] = weighted_ks(x, w, y)
        out[f"{key}_TV"], out[f"{key}_JS"], out[f"{key}_overlap"] = hist_metrics(x, w, y)
        for q in [0.05, 0.50, 0.95, 0.99]:
            out[f"{key}_q{int(q*100):02d}"] = weighted_quantile(x, w, q)
    return out


def main() -> None:
    out = OUT
    out.mkdir(parents=True, exist_ok=True)
    d, b = load_csv(DIRECT), load_csv(BLOCKED)
    phi = np.load(DIRECT_CONFIGS)["phi"]
    diag = diagonal_two_orientation_density(phi)
    rows = [evaluate_diag(d, b, diag, kd) for kd in [0.0, -0.0001, 0.0001, -0.0002, 0.0002, -0.0005, 0.0005, -0.001, 0.001]]
    write_rows(out / "stage4a_kappa_diag_scan.csv", rows)
    (out / "stage4a_convention.json").write_text(json.dumps({"operator": "O_diag=sum_x[c_x c_(x+ex+ey)+c_x c_(x+ex-ey)]", "density": "0.5*(mean forward +diagonal + mean forward -diagonal)", "action_term": "-2*kappa_diag*O_diag", "direct_config_source": str(DIRECT_CONFIGS), "each_undirected_bond_once": True}, indent=2) + "\n")
    print(f"stage 4A complete: {len(rows)} points, N_direct={len(diag)}, flush=complete", flush=True)


if __name__ == "__main__":
    main()
