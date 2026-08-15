#!/usr/bin/env python3
"""5000-config confirmation plots for the lambda=1.0 Pareto diagnostic kernel."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.blocking import load_configs  # noqa: E402
from scripts.common.histogram_compare import metrics, plot_histogram  # noqa: E402
from scripts.common.kernel_io import load_kernel  # noqa: E402
from scripts.run_lam1p0_7x7_kernel_search import ETA_SCALE, block, momentum_extrema, observable_arrays  # noqa: E402


LAM_ROOT = PROJECT_ROOT / "perfect_blocking/perfect_blocking_lam1p0"
DIRECT = PROJECT_ROOT / "data/configs_phi4_2d/lam1p0_kappac0p340301_L16/configs.npz"
FINE = PROJECT_ROOT / "data/configs_phi4_2d/lam1p0_kappac0p340301_L32/configs.npz"
KERNEL = LAM_ROOT / "kernels/candidates/redo_phi2_phi4_pareto_2000/best_pareto_redo_phi2_phi4_eta_included.json"
OUT = LAM_ROOT / "tests/final/final_kernel_confirmation_direct_L16_vs_blocked_L32_phi2_phi4_pareto_5000"

OBS = [
    "action_density",
    "phi2",
    "phi4",
    "local_kurtosis_ratio",
    "NN",
    "2nn",
    "diag",
    "m",
    "m2",
    "m4",
    "Binder_U4_from_averages",
    "xi_over_L",
    "G_00",
    "G_10",
    "G_01",
    "G_pmin_avg",
]
PLOT_OBS = ["phi2", "phi4", "local_kurtosis_ratio", "action_density", "NN", "G_pmin_avg"]
N_CONFIGS = 5000


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    plot_dir = OUT / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    spec = load_kernel(KERNEL)
    matrix = spec.matrix
    metadata = spec.metadata
    direct = load_configs(DIRECT)[:N_CONFIGS]
    fine = load_configs(FINE)[:N_CONFIGS]
    direct_obs = observable_arrays(direct)
    blocked_obs = observable_arrays(block(fine, matrix))

    rows: list[dict[str, Any]] = []
    for obs in OBS:
        rows.append({"candidate": "best_pareto_redo_phi2_phi4_eta_included", "observable": obs, **metrics(direct_obs[obs], blocked_obs[obs], bins=90)})
    write_csv(OUT / "histogram_metrics_5000.csv", rows)

    for obs in PLOT_OBS:
        plot_histogram(
            direct_obs[obs],
            blocked_obs[obs],
            observable=obs,
            out_pdf=plot_dir / f"best_pareto_redo_phi2_phi4_5000_{obs}.pdf",
            bins=90,
            label_a="direct native L16",
            label_b="blocked native L32->L16, Pareto diagnostic kernel",
        )

    mom = momentum_extrema(matrix, grid=512)
    write_csv(
        OUT / "momentum_stability.csv",
        [
            {
                "candidate": "best_pareto_redo_phi2_phi4_eta_included",
                **mom,
                "condition_number": mom["max_K"] / mom["min_K"],
            }
        ],
    )
    write_csv(
        OUT / "kernel_sum_eta_check.csv",
        [
            {
                "candidate": "best_pareto_redo_phi2_phi4_eta_included",
                "kernel_path": str(KERNEL),
                "sum_K": float(np.sum(matrix)),
                "eta_scale": ETA_SCALE,
                "kernel_coefficients_include_eta_scale": bool(metadata.get("kernel_coefficients_include_eta_scale", False)),
                "no_extra_eta_multiplier": True,
                "n_direct": int(direct.shape[0]),
                "n_blocked": int(fine.shape[0]),
            }
        ],
    )
    np.savetxt(OUT / "eta_included_matrix.txt", matrix, fmt="%.16e")
    np.savetxt(OUT / "base_unit_sum_matrix.txt", matrix / ETA_SCALE, fmt="%.16e")

    key = {row["observable"]: row for row in rows}
    lines = [
        "# 5000-Config Confirmation: Pareto Diagnostic Kernel",
        "",
        f"Kernel: `{KERNEL}`",
        f"Direct configs: `{DIRECT}` first `{N_CONFIGS}`.",
        f"Blocked configs: `{FINE}` first `{N_CONFIGS}`.",
        "",
        "This is a diagnostic rerun only. The kernel is not promoted.",
        "",
        "| observable | shift | std ratio | KS | JS | TV | W1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for obs in ["action_density", "phi2", "phi4", "local_kurtosis_ratio", "NN", "G_pmin_avg"]:
        row = key[obs]
        lines.append(
            f"| `{obs}` | {float(row['standardized_mean_shift']):.5f} | {float(row['std_ratio_a_over_b']):.5f} | "
            f"{float(row['ks_statistic']):.5f} | {float(row['jensen_shannon']):.5f} | "
            f"{float(row['total_variation']):.5f} | {float(row['wasserstein_1']):.5f} |"
        )
    lines.extend(
        [
            "",
            "## Eta and Momentum",
            "",
            f"- `sum(K) = {float(np.sum(matrix)):.16g}`",
            f"- `eta_scale = {ETA_SCALE:.16g}`",
            f"- `kernel_coefficients_include_eta_scale = {bool(metadata.get('kernel_coefficients_include_eta_scale', False))}`",
            f"- `min K(p) = {mom['min_K']:.8g}`",
            f"- `max 1/K(p) = {mom['max_inverse_K']:.8g}`",
            "",
            "## Plots",
            "",
        ]
    )
    for obs in PLOT_OBS:
        lines.append(f"- `{plot_dir / f'best_pareto_redo_phi2_phi4_5000_{obs}.pdf'}`")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")

    print(json.dumps({"out": str(OUT), "kernel": str(KERNEL), "n_configs": N_CONFIGS, "sum_K": float(np.sum(matrix))}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
