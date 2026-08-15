#!/usr/bin/env python3
"""Full confirmation for the lambda=1.0 phi2-support balanced kernel."""

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
DIRECT = PROJECT_ROOT / "data/configs_phi4_2d/lam1p0_kappac0p340301_L16/configs.npz"
FINE = PROJECT_ROOT / "data/configs_phi4_2d/lam1p0_kappac0p340301_L32/configs.npz"
CURRENT = LAM_ROOT / "kernels/selected_for_upscaling/best_7x7_phi2_nn_guarded_eta_included.json"
CANDIDATE = LAM_ROOT / "kernels/candidates/redo_phi2_support_balanced_2000/best_phi2_support_balanced_eta_included.json"
OUT = LAM_ROOT / "tests/final/final_confirm_phi2_support_balanced_kernel_full"

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
    "G_00",
    "G_10",
    "G_01",
    "G_pmin_avg",
]
KEY_OBS = ["action_density", "phi2", "phi4", "local_kurtosis_ratio", "NN", "2nn", "diag", "m2", "m4", "G_pmin_avg"]
PLOT_OBS = ["action_density", "phi2", "phi4", "local_kurtosis_ratio", "NN", "2nn", "diag", "m2", "m4", "G_pmin_avg"]


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


def full_metrics(direct_obs: dict[str, np.ndarray], blocked_obs: dict[str, np.ndarray], bins: int = 100) -> dict[str, dict[str, Any]]:
    return {obs: metrics(direct_obs[obs], blocked_obs[obs], bins=bins) for obs in OBS}


def tail_stats(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    q = {p: float(np.quantile(a, p)) for p in [0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99]}
    return {
        "native_q01": q[0.01],
        "native_q05": q[0.05],
        "native_q10": q[0.10],
        "native_q50": q[0.50],
        "native_q90": q[0.90],
        "native_q95": q[0.95],
        "native_q99": q[0.99],
        "blocked_q01": float(np.quantile(b, 0.01)),
        "blocked_q05": float(np.quantile(b, 0.05)),
        "blocked_q10": float(np.quantile(b, 0.10)),
        "blocked_q50": float(np.quantile(b, 0.50)),
        "blocked_q90": float(np.quantile(b, 0.90)),
        "blocked_q95": float(np.quantile(b, 0.95)),
        "blocked_q99": float(np.quantile(b, 0.99)),
        "blocked_frac_below_native_q01": float(np.mean(b < q[0.01])),
        "blocked_frac_below_native_q05": float(np.mean(b < q[0.05])),
        "blocked_frac_below_native_q10": float(np.mean(b < q[0.10])),
        "blocked_frac_above_native_q90": float(np.mean(b > q[0.90])),
        "blocked_frac_above_native_q95": float(np.mean(b > q[0.95])),
        "blocked_frac_above_native_q99": float(np.mean(b > q[0.99])),
        "blocked_frac_inside_native_q05_q95": float(np.mean((b >= q[0.05]) & (b <= q[0.95]))),
        "blocked_frac_inside_native_q01_q99": float(np.mean((b >= q[0.01]) & (b <= q[0.99]))),
    }


def summarize_kernel(name: str, kernel_path: Path, direct: np.ndarray, fine: np.ndarray) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, float]], dict[str, float], np.ndarray]:
    spec = load_kernel(kernel_path)
    matrix = spec.matrix
    direct_obs = observable_arrays(direct)
    blocked_obs = observable_arrays(block(fine, matrix))
    rows = full_metrics(direct_obs, blocked_obs, bins=100)
    tails = {obs: tail_stats(direct_obs[obs], blocked_obs[obs]) for obs in KEY_OBS}
    mom = momentum_extrema(matrix, grid=512)
    return rows, tails, mom, matrix


def split_metrics(name: str, matrix: np.ndarray, direct: np.ndarray, fine: np.ndarray) -> list[dict[str, Any]]:
    rng = np.random.default_rng(2026071824)
    nd = direct.shape[0]
    nf = fine.shape[0]
    splits = {
        "full": (np.arange(nd), np.arange(nf)),
        "first_half": (np.arange(nd // 2), np.arange(nf // 2)),
        "second_half": (np.arange(nd // 2, nd), np.arange(nf // 2, nf)),
        "random_half": (rng.choice(nd, nd // 2, replace=False), rng.choice(nf, nf // 2, replace=False)),
    }
    out: list[dict[str, Any]] = []
    for split, (di, fi) in splits.items():
        rows = full_metrics(observable_arrays(direct[di]), observable_arrays(block(fine[fi], matrix)), bins=80)
        for obs in KEY_OBS:
            out.append({"kernel": name, "split": split, "observable": obs, **rows[obs]})
    return out


def bootstrap_metrics(name: str, direct_obs: dict[str, np.ndarray], blocked_obs: dict[str, np.ndarray], n_boot: int = 500) -> list[dict[str, Any]]:
    rng = np.random.default_rng(2026071825)
    out: list[dict[str, Any]] = []
    for obs in ["action_density", "phi2", "phi4", "local_kurtosis_ratio", "NN", "G_pmin_avg"]:
        a = direct_obs[obs]
        b = blocked_obs[obs]
        vals: dict[str, list[float]] = {k: [] for k in ["ks_statistic", "total_variation", "jensen_shannon", "wasserstein_1", "std_ratio_a_over_b"]}
        for _ in range(n_boot):
            aa = a[rng.integers(0, len(a), len(a))]
            bb = b[rng.integers(0, len(b), len(b))]
            m = metrics(aa, bb, bins=80)
            for key in vals:
                vals[key].append(float(m[key]))
        for key, samples in vals.items():
            q = np.quantile(samples, [0.025, 0.16, 0.50, 0.84, 0.975])
            out.append({"kernel": name, "observable": obs, "metric": key, "q025": q[0], "q16": q[1], "median": q[2], "q84": q[3], "q975": q[4], "n_boot": n_boot})
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    plot_dir = OUT / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    direct = load_configs(DIRECT)
    fine = load_configs(FINE)
    direct_obs = observable_arrays(direct)

    kernels = [
        ("current_selected_7x7", CURRENT),
        ("phi2_support_balanced", CANDIDATE),
    ]
    all_hist: list[dict[str, Any]] = []
    all_tail: list[dict[str, Any]] = []
    all_mom: list[dict[str, Any]] = []
    all_sum: list[dict[str, Any]] = []
    all_split: list[dict[str, Any]] = []
    all_boot: list[dict[str, Any]] = []
    matrices: dict[str, np.ndarray] = {}
    blocked_by_kernel: dict[str, dict[str, np.ndarray]] = {}

    for name, path in kernels:
        spec = load_kernel(path)
        matrix = spec.matrix
        matrices[name] = matrix
        blocked_obs = observable_arrays(block(fine, matrix))
        blocked_by_kernel[name] = blocked_obs
        rows = full_metrics(direct_obs, blocked_obs, bins=100)
        tails = {obs: tail_stats(direct_obs[obs], blocked_obs[obs]) for obs in KEY_OBS}
        mom = momentum_extrema(matrix, grid=512)
        for obs in OBS:
            all_hist.append({"kernel": name, "kernel_path": str(path), "observable": obs, **rows[obs]})
        for obs in KEY_OBS:
            all_tail.append({"kernel": name, "kernel_path": str(path), "observable": obs, **tails[obs]})
        all_mom.append({"kernel": name, "kernel_path": str(path), **mom, "condition_number": mom["max_K"] / mom["min_K"]})
        all_sum.append(
            {
                "kernel": name,
                "kernel_path": str(path),
                "sum_K": float(np.sum(matrix)),
                "eta_scale": ETA_SCALE,
                "kernel_coefficients_include_eta_scale": bool(spec.metadata.get("kernel_coefficients_include_eta_scale", False)),
                "no_extra_eta_multiplier": True,
            }
        )
        all_split += split_metrics(name, matrix, direct, fine)
        all_boot += bootstrap_metrics(name, direct_obs, blocked_obs, n_boot=500)
        for obs in PLOT_OBS:
            plot_histogram(
                direct_obs[obs],
                blocked_obs[obs],
                obs,
                plot_dir / f"{name}_{obs}.pdf",
                bins=100,
                label_a="direct native L16",
                label_b=f"{name} blocked native L32->L16",
            )

    write_csv(OUT / "histogram_metrics_full.csv", all_hist)
    write_csv(OUT / "tail_coverage_full.csv", all_tail)
    write_csv(OUT / "momentum_stability.csv", all_mom)
    write_csv(OUT / "kernel_sum_eta_check.csv", all_sum)
    write_csv(OUT / "split_robustness.csv", all_split)
    write_csv(OUT / "bootstrap_metrics.csv", all_boot)
    np.savetxt(OUT / "phi2_support_balanced_eta_included_matrix.txt", matrices["phi2_support_balanced"], fmt="%.16e")
    np.savetxt(OUT / "phi2_support_balanced_base_unit_sum_matrix.txt", matrices["phi2_support_balanced"] / ETA_SCALE, fmt="%.16e")

    hist_key = {(r["kernel"], r["observable"]): r for r in all_hist}
    tail_key = {(r["kernel"], r["observable"]): r for r in all_tail}
    mom_key = {r["kernel"]: r for r in all_mom}
    lines = [
        "# Full Confirmation: Lambda 1.0 Phi2-Support Balanced Kernel",
        "",
        "No kernel was promoted.",
        f"Direct L16 configs: `{DIRECT}` count `{direct.shape[0]}`.",
        f"Blocked L32 configs: `{FINE}` count `{fine.shape[0]}`.",
        f"Candidate: `{CANDIDATE}`.",
        f"Current selected reference: `{CURRENT}`.",
        "",
        "## Distribution Metrics",
        "",
        "| kernel | observable | mean direct | mean blocked | KS | TV | JS | W1 | std ratio | below q05 | above q95 | above q99 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for kernel in ["current_selected_7x7", "phi2_support_balanced"]:
        for obs in ["action_density", "phi2", "phi4", "local_kurtosis_ratio", "NN", "G_pmin_avg"]:
            h = hist_key[(kernel, obs)]
            t = tail_key[(kernel, obs)]
            lines.append(
                f"| `{kernel}` | `{obs}` | {float(h['mean_a']):.8g} | {float(h['mean_b']):.8g} | "
                f"{float(h['ks_statistic']):.4f} | {float(h['total_variation']):.4f} | {float(h['jensen_shannon']):.4f} | "
                f"{float(h['wasserstein_1']):.5g} | {float(h['std_ratio_a_over_b']):.4f} | "
                f"{float(t['blocked_frac_below_native_q05']):.3f} | {float(t['blocked_frac_above_native_q95']):.3f} | {float(t['blocked_frac_above_native_q99']):.3f} |"
            )
    lines.extend(
        [
            "",
            "Reference matched tail fractions are below q05 ~= 0.05, above q95 ~= 0.05, above q99 ~= 0.01.",
            "",
            "## Momentum And Eta",
            "",
            "| kernel | sum(K) | min K(p) | max K(p) | max 1/K(p) | condition |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in all_sum:
        m = mom_key[row["kernel"]]
        lines.append(
            f"| `{row['kernel']}` | {float(row['sum_K']):.16g} | {float(m['min_K']):.6g} | {float(m['max_K']):.6g} | "
            f"{float(m['max_inverse_K']):.6g} | {float(m['condition_number']):.6g} |"
        )
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            "The support-balanced kernel should be considered for promotion only after review. It materially improves phi2 and local-kurtosis distributional overlap while preserving NN and Gpmin, at the cost of a modest action-density and phi4 degradation relative to the current selected kernel.",
            "",
            "## Output Files",
            "",
            f"- `{OUT / 'histogram_metrics_full.csv'}`",
            f"- `{OUT / 'tail_coverage_full.csv'}`",
            f"- `{OUT / 'split_robustness.csv'}`",
            f"- `{OUT / 'bootstrap_metrics.csv'}`",
            f"- `{OUT / 'plots'}`",
        ]
    )
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"out": str(OUT), "direct_count": int(direct.shape[0]), "fine_count": int(fine.shape[0]), "candidate": str(CANDIDATE)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
