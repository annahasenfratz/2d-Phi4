#!/usr/bin/env python3
"""Common N=5000 global-A/R audit for L16->L32 alternating-KL iterations 4/5."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "perfect_blocking_upsampling"
AUDIT = PKG / "scripts/audit_lam1p0_softcond7_global_ar.py"
RUN4 = PKG / "runs/lam1p0/training/lam1p0_L16to32_alternatingKL_highcorr5_r1_continue/flow4"
K4 = ROOT / "perfect_blocking/perfect_blocking_lam1p0/kernels/candidates/alternatingKL_L16to32_highcorr5_r1_continue/iteration4.json"
RUN5 = PKG / "runs/lam1p0/training/lam1p0_L16to32_alternatingKL_highcorr5_r1_iteration5/flow5"
K5 = ROOT / "perfect_blocking/perfect_blocking_lam1p0/kernels/candidates/alternatingKL_L16to32_highcorr5_r1_iteration5/iteration5.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260812)
    args = parser.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    pairs = [
        ("iteration4", RUN4 / "stage_oo/checkpoints/checkpoint_best_nll.pt", K4),
        ("iteration5", RUN5 / "stage_oo/checkpoints/checkpoint_best_nll.pt", K5),
    ]
    rows = []
    for label, checkpoint, kernel in pairs:
        subprocess.run([
            sys.executable, "-B", str(AUDIT), "--checkpoint", str(checkpoint),
            "--kernel", str(kernel), "--out", str(out / label),
            "--n", str(args.n), "--seed", str(args.seed),
        ], check=True)
        row = json.loads((out / label / "summary.json").read_text())
        row["label"] = label
        rows.append(row)
    keys = ["label", "logw_std", "logw_q05", "logw_q95", "ess_over_n",
            "stationary_independence_MH_acceptance_proxy", "logdet_K"]
    with (out / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows([{key: row[key] for key in keys} for row in rows])
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
