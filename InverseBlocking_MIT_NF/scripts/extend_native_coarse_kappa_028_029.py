#!/usr/bin/env python3
"""Generate extended native coarse ensembles for kappa_c=0.28 and 0.29."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT / "scripts"))

from pilot_utils import generate_coarse_ensemble  # type: ignore


OUT = PROJECT / "outputs" / "coarse_distribution_calibration" / "generated_native_extended"
KAPPAS = [0.28, 0.29]
LAMBDA = 1.0
L = 8
N_SAMPLES = 4096
THERMAL_SWEEPS = 2000
SKIP_SWEEPS = 16
PROPOSAL_WIDTH = 0.8
SEED_BASE = 20250624


def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def tag(kappa: float) -> str:
    return f"kappa{kappa:.2f}".replace(".", "p")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summaries = []
    for i, kappa in enumerate(KAPPAS):
        stem = f"native_coarse_lam1_{tag(kappa)}_L8_extended"
        cfg_path = OUT / f"{stem}.npy"
        summary_path = OUT / f"{stem}_summary.json"
        history_path = OUT / f"{stem}_history.csv"
        if cfg_path.exists() and summary_path.exists():
            summary = json.loads(summary_path.read_text())
            summary["status"] = "reused_existing"
        else:
            cfgs, summary, history = generate_coarse_ensemble(
                L=L,
                kappa=kappa,
                lam=LAMBDA,
                n_samples=N_SAMPLES,
                thermal_sweeps=THERMAL_SWEEPS,
                skip_sweeps=SKIP_SWEEPS,
                proposal_width=PROPOSAL_WIDTH,
                seed=SEED_BASE + i * 1000 + int(round(kappa * 1000)),
            )
            import numpy as np

            np.save(cfg_path, cfgs)
            write_csv(history_path, history)
            summary["status"] = "generated"
            summary["production_quality"] = False
            summary["quality_note"] = "Extended diagnostic native coarse chain; autocorrelation not measured."
            summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        summary["configs_path"] = str(cfg_path.resolve())
        summary["history_path"] = str(history_path.resolve())
        summary["summary_path"] = str(summary_path.resolve())
        summaries.append(summary)
    (OUT / "extended_generation_summary.json").write_text(json.dumps({"runs": summaries}, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
