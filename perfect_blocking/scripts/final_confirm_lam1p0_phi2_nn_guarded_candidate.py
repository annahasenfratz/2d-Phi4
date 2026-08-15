#!/usr/bin/env python3
"""Final confirmation for the lambda=1.0 phi2-priority NN-guarded 7x7 candidate."""

from __future__ import annotations

import csv
import json
import math
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
OUT = LAM_ROOT / "tests/final/final_kernel_confirmation_phi2_nn_guarded_7x7"
DIRECT = PROJECT_ROOT / "data/configs_phi4_2d/lam1p0_kappac0p340301_L16/configs.npz"
FINE = PROJECT_ROOT / "data/configs_phi4_2d/lam1p0_kappac0p340301_L32/configs.npz"
KERNELS = {
    "current_final_nn_constrained_7x7": LAM_ROOT / "kernels/final/chosen_kernel.json",
    "retrained_5x5": LAM_ROOT / "kernels/candidates/systematic_training/best_retrained_5x5_full_objective_eta_included.json",
    "candidate_phi2_nn_guarded_7x7": LAM_ROOT / "kernels/candidates/7x7_full_retraining_phi2_nn_guarded/best_7x7_full_retraining_controlled_eta_included.json",
}
KEY_OBS = ["action_density", "phi2", "phi4", "local_kurtosis_ratio", "NN", "2nn", "diag", "m2", "m4", "G_pmin_avg"]
ALL_OBS = [
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


def load_kernel_matrix(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    data = json.loads(path.read_text())
    if "matrix" in data and bool(data.get("kernel_coefficients_include_eta_scale")):
        return np.asarray(data["matrix"], dtype=np.float64), data
    spec = load_kernel(path)
    return spec.matrix, spec.metadata


def l16_grid_extrema(matrix: np.ndarray, L: int = 16) -> dict[str, float]:
    radius = matrix.shape[0] // 2
    coords = np.arange(-radius, radius + 1)
    vals = []
    for nx in range(L):
        px = 2.0 * np.pi * nx / L
        if px >= np.pi:
            px -= 2.0 * np.pi
        ex = np.exp(1j * px * coords)
        for ny in range(L):
            py = 2.0 * np.pi * ny / L
            if py >= np.pi:
                py -= 2.0 * np.pi
            ey = np.exp(1j * py * coords)
            val = float(np.real_if_close(np.sum(matrix * ex[:, None] * ey[None, :])).real)
            vals.append(val)
    arr = np.asarray(vals, dtype=np.float64)
    inv = 1.0 / arr
    return {
        "l16_min_K": float(arr.min()),
        "l16_max_K": float(arr.max()),
        "l16_min_inverse_K": float(inv.min()),
        "l16_max_inverse_K": float(inv.max()),
        "l16_condition_number": float(arr.max() / arr.min()),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "plots").mkdir(exist_ok=True)
    direct = load_configs(DIRECT)
    fine = load_configs(FINE)
    direct_obs = observable_arrays(direct)

    matrices: dict[str, np.ndarray] = {}
    metadata: dict[str, dict[str, Any]] = {}
    blocked_obs: dict[str, dict[str, np.ndarray]] = {}
    full_rows: list[dict[str, Any]] = []
    momentum_rows: list[dict[str, Any]] = []
    eta_rows: list[dict[str, Any]] = []
    outer_rows: list[dict[str, Any]] = []

    for name, path in KERNELS.items():
        matrix, meta = load_kernel_matrix(path)
        matrices[name] = matrix
        metadata[name] = meta
        blocked_obs[name] = observable_arrays(block(fine, matrix))
        for obs in ALL_OBS:
            full_rows.append({"candidate": name, "observable": obs, **metrics(direct_obs[obs], blocked_obs[name][obs], bins=50)})
        dense = momentum_extrema(matrix, grid=1024)
        l16 = l16_grid_extrema(matrix)
        momentum_rows.append({"candidate": name, **dense, **l16, "dense_condition_number": dense["max_K"] / dense["min_K"]})
        eta_rows.append(
            {
                "candidate": name,
                "kernel_path": str(path),
                "sum_K": float(matrix.sum()),
                "eta_scale": ETA_SCALE,
                "sum_matches_eta_scale": math.isclose(float(matrix.sum()), ETA_SCALE, rel_tol=1e-12, abs_tol=1e-12),
                "kernel_coefficients_include_eta_scale": bool(meta.get("kernel_coefficients_include_eta_scale")),
                "no_extra_eta_factor_applied": True,
            }
        )
        base_classes = meta.get("base_orbit_classes_before_eta_scale", {})
        outer_rows.append(
            {
                "candidate": name,
                "K30_base": base_classes.get("30", 0.0),
                "K31_base": base_classes.get("31", 0.0),
                "K32_base": base_classes.get("32", 0.0),
                "K33_base": base_classes.get("33", 0.0),
            }
        )

    write_csv(OUT / "full_metrics_old_retrained_7x7.csv", full_rows)
    write_csv(OUT / "momentum_stability.csv", momentum_rows)
    write_csv(OUT / "eta_convention_check.csv", eta_rows)
    write_csv(OUT / "outer_shell_coefficients.csv", outer_rows)

    # Split and bootstrap metrics.
    n = len(direct)
    rng = np.random.default_rng(20260719)
    split_defs = {
        "first_half": np.arange(0, n // 2),
        "second_half": np.arange(n // 2, n),
        "random_2500_seed20260719": rng.choice(n, size=n // 2, replace=False),
    }
    split_rows: list[dict[str, Any]] = []
    for split, idx in split_defs.items():
        for name in KERNELS:
            for obs in KEY_OBS:
                split_rows.append({"sample": split, "candidate": name, "observable": obs, **metrics(direct_obs[obs][idx], blocked_obs[name][obs][idx], bins=50)})

    boot_summary_rows: list[dict[str, Any]] = []
    boot_raw: list[dict[str, Any]] = []
    accum: dict[tuple[str, str], dict[str, list[float]]] = {}
    for b in range(160):
        idx = rng.integers(0, n, size=n)
        for name in KERNELS:
            for obs in KEY_OBS:
                row = metrics(direct_obs[obs][idx], blocked_obs[name][obs][idx], bins=50)
                raw = {"sample": f"bootstrap_{b:03d}", "candidate": name, "observable": obs, **row}
                boot_raw.append(raw)
                key = (name, obs)
                acc = accum.setdefault(key, {k: [] for k in ["ks_statistic", "jensen_shannon", "total_variation", "standardized_mean_shift", "std_ratio_a_over_b"]})
                for metric_name in acc:
                    acc[metric_name].append(float(row[metric_name]))
    for (name, obs), values in accum.items():
        out: dict[str, Any] = {"candidate": name, "observable": obs}
        for metric_name, vals in values.items():
            arr = np.asarray(vals, dtype=np.float64)
            out[f"{metric_name}_mean"] = float(np.mean(arr))
            out[f"{metric_name}_std"] = float(np.std(arr, ddof=1))
            out[f"{metric_name}_q05"] = float(np.quantile(arr, 0.05))
            out[f"{metric_name}_q95"] = float(np.quantile(arr, 0.95))
        boot_summary_rows.append(out)
    write_csv(OUT / "split_metrics.csv", split_rows)
    write_csv(OUT / "bootstrap_metrics_raw.csv", boot_raw)
    write_csv(OUT / "bootstrap_metric_uncertainties.csv", boot_summary_rows)

    for name in KERNELS:
        for obs in KEY_OBS:
            plot_histogram(
                direct_obs[obs],
                blocked_obs[name][obs],
                obs,
                OUT / "plots" / f"{name}_{obs}.pdf",
                label_a="direct L16",
                label_b=f"{name} blocked L32->L16",
            )

    # Decision report.
    rows_by = {(r["candidate"], r["observable"]): r for r in full_rows}
    mom_by = {r["candidate"]: r for r in momentum_rows}
    boot_by = {(r["candidate"], r["observable"]): r for r in boot_summary_rows}

    def val(candidate: str, obs: str, key: str) -> float:
        return float(rows_by[(candidate, obs)][key])

    current = "current_final_nn_constrained_7x7"
    candidate = "candidate_phi2_nn_guarded_7x7"
    decision_checks = {
        "phi2_improves_vs_current_final": val(candidate, "phi2", "ks_statistic") < val(current, "phi2", "ks_statistic"),
        "phi4_within_relaxed_guardrail": val(candidate, "phi4", "ks_statistic") <= 0.030,
        "local_kurtosis_improves_vs_current_final": val(candidate, "local_kurtosis_ratio", "ks_statistic") < val(current, "local_kurtosis_ratio", "ks_statistic"),
        "action_density_within_guardrail": val(candidate, "action_density", "ks_statistic") <= 0.035,
        "NN_improves_vs_current_final": val(candidate, "NN", "ks_statistic") < val(current, "NN", "ks_statistic"),
        "m2_comparable": val(candidate, "m2", "ks_statistic") <= 0.025,
        "m4_comparable": val(candidate, "m4", "ks_statistic") <= 0.025,
        "G_pmin_comparable": val(candidate, "G_pmin_avg", "ks_statistic") <= 0.035,
        "dense_K_positive": float(mom_by[candidate]["min_K"]) > 0.0,
        "max_invK_bounded": float(mom_by[candidate]["max_inverse_K"]) <= 1.6,
    }
    passes = all(decision_checks.values())

    lines = [
        "# Final Confirmation: Lambda 1.0 Phi2-Priority NN-Guarded 7x7 Candidate",
        "",
        "Compared current final NN-constrained 7x7, retrained 5x5, and the phi2-priority NN-guarded 7x7 candidate on identical L16/L32 ensembles.",
        "",
        "## Full-Sample Metrics",
        "",
        "| observable | current final KS | retrained KS | candidate KS | current final JS | retrained JS | candidate JS |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for obs in KEY_OBS:
        lines.append(
            f"| {obs} | {val(current, obs, 'ks_statistic'):.6g} | {val('retrained_5x5', obs, 'ks_statistic'):.6g} | {val(candidate, obs, 'ks_statistic'):.6g} | "
            f"{val(current, obs, 'jensen_shannon'):.6g} | {val('retrained_5x5', obs, 'jensen_shannon'):.6g} | {val(candidate, obs, 'jensen_shannon'):.6g} |"
        )
    lines.extend(["", "## Bootstrap 90% Intervals", "", "| candidate | observable | KS mean | KS 5%-95% | JS mean | JS 5%-95% |", "|---|---|---:|---:|---:|---:|"])
    for name in KERNELS:
        for obs in KEY_OBS:
            b = boot_by[(name, obs)]
            lines.append(
                f"| {name} | {obs} | {float(b['ks_statistic_mean']):.6g} | {float(b['ks_statistic_q05']):.6g}-{float(b['ks_statistic_q95']):.6g} | "
                f"{float(b['jensen_shannon_mean']):.6g} | {float(b['jensen_shannon_q05']):.6g}-{float(b['jensen_shannon_q95']):.6g} |"
            )
    lines.extend(["", "## Momentum And Eta", "", "| candidate | sum K | min K | max K | max 1/K | L16 max 1/K | eta included |", "|---|---:|---:|---:|---:|---:|---|"])
    eta_by = {r["candidate"]: r for r in eta_rows}
    for name in KERNELS:
        m = mom_by[name]
        e = eta_by[name]
        lines.append(
            f"| {name} | {float(e['sum_K']):.17g} | {float(m['min_K']):.6g} | {float(m['max_K']):.6g} | {float(m['max_inverse_K']):.6g} | {float(m['l16_max_inverse_K']):.6g} | {e['kernel_coefficients_include_eta_scale']} |"
        )
    lines.extend(["", "## Decision Checks", ""])
    for key, ok in decision_checks.items():
        lines.append(f"- `{key}`: `{ok}`")
    lines.extend(
        [
            "",
            f"Passes configured promotion evidence checks: `{passes}`",
            "",
            "No promotion was performed.",
            "",
            "## Prepared Promotion Plan",
            "",
            "If explicitly approved later:",
            "",
            "1. Copy `perfect_blocking/perfect_blocking_lam1p0/kernels/final/chosen_kernel.json` to an archive filename under `kernels/final/`.",
            "2. Copy `perfect_blocking/perfect_blocking_lam1p0/kernels/candidates/7x7_full_retraining_phi2_nn_guarded/best_7x7_full_retraining_controlled_eta_included.json` to `kernels/final/chosen_kernel.json`.",
            "3. Regenerate `kernels/final/chosen_kernel.txt`.",
            "4. Update `kernels/final/README.md` and `perfect_blocking_lam1p0/README.md` marking the current final NN-constrained 7x7 as superseded.",
        ]
    )
    (OUT / "final_confirmation_report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"out": str(OUT), "passes": passes}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
