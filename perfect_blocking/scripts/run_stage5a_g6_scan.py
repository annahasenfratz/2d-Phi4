#!/usr/bin/env python3
"""Stage 5A: reweight the direct L16 ensemble in a local c^6 operator."""

from __future__ import annotations

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
    hist_metrics,
    load_csv,
    trial_observables,
    weighted_ks,
    weighted_quantile,
    write_rows,
)

OUT = ROOT / "perfect_blocking/perfect_blocking_lam1p0/coarse_action_reweight_kappa_lambda_20260720"
DIRECT_CONFIGS = ROOT / "data/configs_phi4_2d/lam1p0_kappac0p340301_L16/configs.npz"


def evaluate_g6(d, b, o6_density, g6):
    trial = trial_observables(d, KAPPA0, LAMBDA0)
    target = trial_observables(b, KAPPA0, LAMBDA0)
    trial["action_density"] = trial["action_density"] + g6 * o6_density
    logw = -g6 * o6_density * 256.0
    logw -= np.max(logw)
    raw = np.exp(logw)
    w = raw / raw.sum()
    out = {"g6": g6, "ESS": 1.0 / np.sum(w*w), "ESS_fraction": 1.0 / (np.sum(w*w)*len(w)), "max_normalized_weight": float(np.max(w)), "weight_cv": float(np.std(w) / np.mean(w)), "max_log_weight_shift": float(np.max(logw)-np.min(logw))}
    for key in ["PC1", "PC2", "action_density", "phi2", "phi4", "local_kurtosis_ratio", "NN", "diag", "2nn", "G_pmin_x_cfg", "G_pmin_y_cfg"]:
        x, y = trial[key], target[key]
        mu = float(np.sum(w*x)); sd = float(np.sqrt(np.sum(w*(x-mu)**2)))
        bm, bsd = float(np.mean(y)), float(np.std(y))
        out[f"{key}_mean"] = mu; out[f"{key}_std"] = sd
        out[f"{key}_mean_shift_blocked_sigma"] = (mu-bm)/max(bsd, 1e-300)
        out[f"{key}_std_ratio"] = sd/max(bsd, 1e-300)
        out[f"{key}_KS"] = weighted_ks(x, w, y)
        out[f"{key}_TV"], out[f"{key}_JS"], out[f"{key}_overlap"] = hist_metrics(x, w, y)
        for q in [0.05, 0.50, 0.95, 0.99]:
            out[f"{key}_q{int(q*100):02d}"] = weighted_quantile(x, w, q)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d, b = load_csv(DIRECT), load_csv(BLOCKED)
    phi = np.load(DIRECT_CONFIGS)["phi"]
    o6_density = np.mean(phi**6, axis=(1, 2))
    coeffs = [0.0, -1e-5, 1e-5, -2e-5, 2e-5, -5e-5, 5e-5, -1e-4, 1e-4, -2e-4, 2e-4]
    rows = [evaluate_g6(d, b, o6_density, g) for g in coeffs]
    write_rows(OUT / "stage5a_g6_scan.csv", rows)
    (OUT / "stage5a_convention.json").write_text(json.dumps({"operator": "O6=sum_x c_x^6", "density": "mean(c^6)", "action_term": "+g6*O6", "weight": "exp(-g6*O6)", "direct_config_source": str(DIRECT_CONFIGS), "volume": 256}, indent=2) + "\n")
    print(f"stage 5A complete: {len(rows)} points, N_direct={len(o6_density)}, flush=complete", flush=True)


if __name__ == "__main__":
    main()
