#!/usr/bin/env python3
"""Regenerate the same-split held-out comparison for ETA-FREE-ETHAN7-LAM1."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "perfect_blocking"), str(ROOT / "perfect_blocking/scripts")]
from scripts.common.blocking import load_configs
import run_lam1p0_ethan_7x7_free_eta as driver

RAW = ROOT / "perfect_blocking/perfect_blocking_lam1p0/tests/intermediate/ethan_7x7_free_eta_mu30_N5_train9000_test1000_20260820"


def rows_for(label: str, direct: dict[str, np.ndarray], fine: np.ndarray, kernel: np.ndarray, pair: str) -> list[dict[str, object]]:
    rows = driver.report_rows(direct, driver.operators(driver.block(fine, kernel)))
    return [{"volume_pair": pair, "kernel": label, **row} for row in rows]


def main() -> None:
    result = json.loads((RAW / "result.json").read_text())
    source = json.loads(Path(result["source_kernel"]).read_text())
    ids16 = np.asarray(result["split"]["direct_L16_indices"], dtype=int)
    ids32 = np.asarray(result["split"]["fine_L32_indices"], dtype=int)
    ids64 = np.asarray(result["split"]["fine_L64_indices"], dtype=int)
    direct16 = load_configs(ROOT / "data/configs_phi4_2d/lam1p0_kappac0p340301_L16/configs.npz")
    fine32 = load_configs(ROOT / "data/configs_phi4_2d/lam1p0_kappac0p340301_L32/configs.npz")
    fine64 = load_configs(ROOT / "data/configs_phi4_2d/lam1p0_kappac0p340301_L64/configs.npz")
    direct_l16 = driver.operators(direct16[ids16[4000:]])
    direct_l32 = driver.operators(fine32[ids32[9000:]])
    heldout_l32 = fine32[ids32[9000:]]
    heldout_l64 = fine64[ids64]
    all_rows: list[dict[str, object]] = []
    for label, kernel in (("Ethan fixed eta=0.25", np.asarray(source["matrix"])), ("free eta=0.2518658", np.asarray(result["matrix"]))):
        all_rows += rows_for(label, direct_l16, heldout_l32, kernel, "L32_to_L16")
        all_rows += rows_for(label, direct_l32, heldout_l64, kernel, "L64_to_L32")
    pd.DataFrame(all_rows).to_csv(OUT / "observables.csv", index=False)
    pd.DataFrame([{
        "kernel": "Ethan fixed eta=0.25", "eta": 0.25, "eta_scale": float(np.asarray(source["matrix"]).sum()),
        "fit_chi2": 1.6443611669205966, "inverse_tail_fraction": 0.1806934003333773,
        "objective": 7.065163176921915, "min_K": 0.5437241326652574,
        "max_K": 1.2439044206652576, "condition_number": 2.287749146920916,
        "max_abs_K_inverse": 1.8391679528700409,
    }, {
        "kernel": "free eta=0.2518658", "eta": result["eta"], "eta_scale": result["eta_scale_numeric"],
        "fit_chi2": 0.7003509421089624, "inverse_tail_fraction": result["inverse_tail_fraction"],
        "objective": result["best_fit"]["objective"], "min_K": result["momentum_stability"]["min_K"],
        "max_K": result["momentum_stability"]["max_K"], "condition_number": result["condition_number"],
        "max_abs_K_inverse": result["momentum_stability"]["max_inverse_K"],
    }]).to_csv(OUT / "optimization_summary.csv", index=False)


if __name__ == "__main__":
    main()
