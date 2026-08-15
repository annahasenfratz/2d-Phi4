#!/usr/bin/env python3
"""Stage 5B: reweight the direct L16 ensemble in O22=sum nearest bonds c^2 c'^2."""

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


def o22_density(phi: np.ndarray) -> np.ndarray:
    """Return (sum forward NN bonds c^2 c'^2)/V, with each bond once."""
    a2 = np.asarray(phi, dtype=float) ** 2
    return np.mean(a2 * np.roll(a2, -1, axis=1) + a2 * np.roll(a2, -1, axis=2), axis=(1, 2))


def evaluate_g22(d, b, op_density, g22):
    trial = trial_observables(d, KAPPA0, LAMBDA0)
    target = trial_observables(b, KAPPA0, LAMBDA0)
    target["action_density"] = b["action_density"]
    # The stored NN observable is the average of x/y bonds; the action uses both.
    trial["action_density"] = (1.0 - 2.0 * LAMBDA0) * d["phi2"] + LAMBDA0 * d["phi4"] - 4.0 * KAPPA0 * d["NN"] + g22 * op_density
    logw = -g22 * op_density * 256.0
    logw -= np.max(logw)
    raw = np.exp(logw)
    w = raw / raw.sum()
    out = {"g22": g22, "ESS": 1.0 / np.sum(w*w), "ESS_fraction": 1.0 / (np.sum(w*w)*len(w)), "max_normalized_weight": float(np.max(w)), "weight_cv": float(np.std(w)/np.mean(w)), "max_log_weight_shift": float(np.max(logw)-np.min(logw))}
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
    op = o22_density(phi)
    initial = [0.0, -1e-5, 1e-5, -2e-5, 2e-5, -5e-5, 5e-5, -1e-4, 1e-4, -2e-4, 2e-4, -5e-4, 5e-4]
    rows = [evaluate_g22(d, b, op, g) for g in initial]
    if min(r["ESS_fraction"] for r in rows) > 0.99:
        rows.extend(evaluate_g22(d, b, op, g) for g in [-1e-3, 1e-3])
    rows.sort(key=lambda r: r["g22"])
    write_rows(OUT / "stage5b_g22_scan.csv", rows)
    (OUT / "stage5b_convention.json").write_text(json.dumps({"operator": "O22=sum_x[c_x^2 c_(x+ex)^2+c_x^2 c_(x+ey)^2]", "density": "mean(c^2*c_shift_x^2+c^2*c_shift_y^2)", "action_term": "+g22*O22", "weight": "exp(-g22*O22)", "direct_config_source": str(DIRECT_CONFIGS), "volume": 256, "o22_mean": float(np.mean(op)), "o22_std": float(np.std(op)), "o22_min": float(np.min(op)), "o22_max": float(np.max(op)), "extended_to_pm_1e-3": bool(min(r["ESS_fraction"] for r in rows if abs(r["g22"]) <= 5e-4) > 0.99)}, indent=2) + "\n")
    print(f"stage 5B complete: {len(rows)} points, O22 std={np.std(op):.8g}, range=[{np.min(op):.8g},{np.max(op):.8g}], flush=complete", flush=True)


if __name__ == "__main__":
    main()
