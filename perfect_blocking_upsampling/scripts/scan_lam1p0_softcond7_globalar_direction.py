#!/usr/bin/env python3
"""Guarded frozen-flow line scan along the globally promising 7x7 direction."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
BASE_KERNEL = ROOT / (
    "perfect_blocking/perfect_blocking_lam1p0/kernels/candidates/"
    "allL16_chi2_R3_soft_conditioned_7x7_eta_included.json"
)
BASE_CHECKPOINT = ROOT / (
    "perfect_blocking_upsampling/runs/lam1p0/training/"
    "lam1p0_L16to32_softcond7_pureNLL_N5000_20260808T230212Z/"
    "stage_oo/checkpoints/checkpoint_best_nll.pt"
)
AUDIT = ROOT / "perfect_blocking_upsampling/scripts/audit_lam1p0_softcond7_global_ar.py"
ORBITS = ((3, 1), (3, 2), (2, 1))


def orbit_mask(radius: int, orbit: tuple[int, int]) -> np.ndarray:
    coordinates = np.arange(-radius, radius + 1)
    xx, yy = np.meshgrid(coordinates, coordinates, indexing="ij")
    return (
        (np.maximum(np.abs(xx), np.abs(yy)) == orbit[0])
        & (np.minimum(np.abs(xx), np.abs(yy)) == orbit[1])
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epsilons", type=float, nargs="+", default=[0.00025, 0.0005, 0.001, 0.002])
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--min-k", type=float, default=0.35)
    parser.add_argument("--max-condition", type=float, default=3.0)
    args = parser.parse_args()
    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    base = json.loads(BASE_KERNEL.read_text())
    matrix = np.asarray(base["matrix"], dtype=float)
    radius = matrix.shape[0] // 2
    masks = [orbit_mask(radius, orbit) for orbit in ORBITS]
    multiplicities = [int(mask.sum()) for mask in masks]
    rows: list[dict[str, object]] = []

    for epsilon in args.epsilons:
        candidate = matrix.copy()
        for mask in masks:
            candidate[mask] -= epsilon
        # This compensates the change in every D4 orbit, preserving Khat(0).
        candidate[radius, radius] += epsilon * sum(multiplicities)

        spectrum = np.abs(np.fft.fft2(candidate, s=(32, 32)))
        min_k = float(spectrum.min())
        condition = float(spectrum.max() / min_k)
        row: dict[str, object] = {
            "epsilon": epsilon,
            "min_abs_K": min_k,
            "max_abs_Kinv": 1.0 / min_k,
            "condition": condition,
        }
        label = f"eps{epsilon:.5f}".replace(".", "p")
        if min_k < args.min_k or condition > args.max_condition:
            row["status"] = "rejected_guard"
            rows.append(row)
            continue

        spec = dict(base)
        spec["name"] = f"softcond7_globalar_direction_{label}"
        spec["matrix"] = candidate.tolist()
        spec["global_ar_line_scan"] = {
            "orbits_minus_epsilon": ["31", "32", "21"],
            "epsilon": epsilon,
            "min_abs_K": min_k,
            "condition": condition,
        }
        kernel_path = out / f"kernel_{label}.json"
        kernel_path.write_text(json.dumps(spec, indent=2) + "\n")
        case_out = out / f"case_{label}"
        command = [
            sys.executable, "-B", str(AUDIT),
            "--checkpoint", str(BASE_CHECKPOINT),
            "--kernel", str(kernel_path),
            "--out", str(case_out),
            "--n", str(args.n),
            "--seed", str(args.seed),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            row.update(status="failed", error=result.stderr[-1000:])
        else:
            summary = json.loads((case_out / "summary.json").read_text())
            row.update(
                status="ok",
                logw_std=summary["logw_std"],
                ess_over_n=summary["ess_over_n"],
                stationary_acceptance=summary["stationary_independence_MH_acceptance_proxy"],
            )
        rows.append(row)

    columns = sorted({column for row in rows for column in row})
    with (out / "line_scan.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
