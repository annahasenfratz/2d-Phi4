#!/usr/bin/env python3
"""Summarize a combined factor-two coordinate-MH measurement directory."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp

PROJECT_ROOT = Path(__file__).resolve().parents[2]
import sys

sys.path[:0] = [str(PROJECT_ROOT / "perfect_blocking_upsampling" / "src"), str(PROJECT_ROOT / "perfect_blocking_upsampling" / "scripts")]

from perfect_blocking_upsampling.actions import ActionSpec  # noqa: E402
from perfect_blocking_upsampling.blocking import load_phi  # noqa: E402
from run_lam1p0_l16to32_rqspline_zeroshot import per_config_observables  # noqa: E402


OBSERVABLES = ("action_density", "phi2", "phi4", "kurtosis", "NN", "diag", "2nn", "m2", "m4", "G_pmin_avg")
REPORT_SWEEPS = (0, 50, 100, 200, 400)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def ips_tau(series: np.ndarray, max_lag: int = 30) -> tuple[float, float]:
    x = np.asarray(series, dtype=np.float64)
    x -= x.mean()
    variance = np.mean(x * x)
    if not np.isfinite(variance) or variance <= 1.0e-30:
        return float("nan"), float("nan")
    tau = 0.5
    lag_one = float("nan")
    for lag in range(1, min(max_lag, len(x) // 2) + 1):
        rho = float(np.dot(x[:-lag], x[lag:]) / ((len(x) - lag) * variance))
        if lag == 1:
            lag_one = rho
        if not np.isfinite(rho) or rho <= 0.0:
            break
        tau += rho
    return tau, lag_one


def row_values(rows: list[dict[str, str]], observable: str) -> np.ndarray:
    if observable == "kurtosis":
        return np.asarray([float(row["phi4"]) / float(row["phi2"]) ** 2 for row in rows], dtype=np.float64)
    if observable == "G_pmin_avg":
        return np.asarray([(float(row["G_pmin_x_cfg"]) + float(row["G_pmin_y_cfg"])) / 2.0 for row in rows], dtype=np.float64)
    return np.asarray([float(row[observable]) for row in rows], dtype=np.float64)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("combined_run", type=Path)
    parser.add_argument("--native-source", type=Path, default=None)
    args = parser.parse_args()
    run = args.combined_run.resolve()
    manifest = json.loads((run / "combined_manifest.json").read_text(encoding="utf-8"))
    rows = read_csv(run / "observables" / "per_sweep_observables.csv")
    inventory = read_csv(run / "source_index_inventory.csv")
    first_config = json.loads((Path(manifest["combined_runs"][0]) / "run_config.json").read_text(encoding="utf-8"))
    lc, lf = int(first_config["L_c"]), int(first_config["L_f"])
    native_source = args.native_source or Path(first_config["native_reference_source"])
    if not native_source.is_absolute():
        native_source = PROJECT_ROOT / native_source

    native_index = {int(row["chain_id"]): int(row["source_index"]) for row in inventory if row["role"] == "native_reference"}
    chain_ids = sorted(native_index)
    native_all = load_phi(native_source)
    native_phi = native_all[[native_index[chain] for chain in chain_ids]]
    native_obs, native_g = per_config_observables(native_phi, ActionSpec("phi4_nn", 1.0, 0.340301))
    native = {**{key: native_obs[key] for key in ("action_density", "phi2", "phi4", "NN", "diag", "2nn", "m2", "m4")},
              "kurtosis": native_obs["phi4"] / native_obs["phi2"] ** 2,
              "G_pmin_avg": native_g["G_pmin_avg"]}

    run_info: list[dict[str, object]] = []
    chain_to_group: dict[int, str] = {}
    for source_run, offset in manifest["chain_id_offsets"].items():
        config = json.loads((Path(source_run) / "run_config.json").read_text(encoding="utf-8"))
        status = json.loads((Path(source_run) / "status.json").read_text(encoding="utf-8"))
        group = f"coarse={config['coarse_sigma']:.3g},detail={config['detail_sigma']:.3g}"
        for chain in range(int(offset), int(offset) + 128):
            chain_to_group[chain] = group
        run_info.append({
            "run": Path(source_run).name,
            "source_start": config["initial_start_index"],
            "coarse_sigma": config["coarse_sigma"],
            "detail_sigma": config["detail_sigma"],
            "coarse_acceptance": status["coordinate_acceptance"]["coarse"],
            "d01_acceptance": status["coordinate_acceptance"]["d01"],
            "d10_acceptance": status["coordinate_acceptance"]["d10"],
            "d11_acceptance": status["coordinate_acceptance"]["d11"],
        })
    write_csv(run / "run_inventory_summary.csv", run_info)

    by_sweep: dict[int, list[dict[str, str]]] = defaultdict(list)
    by_chain: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_sweep[int(row["sweep"])].append(row)
        by_chain[int(row["chain_id"])].append(row)

    metrics: list[dict[str, object]] = []
    for sweep in REPORT_SWEEPS:
        measured = sorted(by_sweep[sweep], key=lambda row: int(row["chain_id"]))
        for observable in OBSERVABLES:
            x, y = row_values(measured, observable), native[observable]
            z = (x.mean() - y.mean()) / np.sqrt(x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y))
            metrics.append({
                "sweep": sweep, "observable": observable,
                "mean": x.mean(), "native_mean": y.mean(), "mean_z": z,
                "width_ratio": x.std(ddof=1) / y.std(ddof=1),
                "ks": ks_2samp(x, y).statistic,
            })
    write_csv(run / "sweep_native_comparison_summary.csv", metrics)

    auto_rows: list[dict[str, object]] = []
    for observable in OBSERVABLES:
        by_group: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for chain in chain_ids:
            series_rows = sorted((row for row in by_chain[chain] if int(row["sweep"]) >= 100), key=lambda row: int(row["sweep"]))
            tau, lag1 = ips_tau(row_values(series_rows, observable))
            by_group[chain_to_group[chain]].append((tau * 5.0, lag1))
        for group, values in by_group.items():
            a = np.asarray(values, dtype=np.float64)
            auto_rows.append({"observable": observable, "step_group": group, "n_chains": len(a),
                              "tau_int_sweeps_mean": np.nanmean(a[:, 0]), "tau_int_sweeps_median": np.nanmedian(a[:, 0]),
                              "lag1_mean": np.nanmean(a[:, 1]), "window": "post_sweep_100", "saved_cadence": 5})
    write_csv(run / "autocorrelation_summary.csv", auto_rows)

    def metric(sweep: int, observable: str) -> dict[str, object]:
        return next(row for row in metrics if row["sweep"] == sweep and row["observable"] == observable)

    lines = [
        f"# L{lc} to L{lf} Coordinate-MH Summary",
        "",
        "## Scope",
        "",
        f"- {len(run_info)} independent 128-chain runs, for {len(chain_ids)} chains total; all completed through sweep 400.",
        "- State updates are physical-coordinate coarse-plus-detail Metropolis updates; the wrapped flow initializes sweep 0 only.",
        "- `DIVIDE=2`: each recorded sweep visits all four residue classes once.",
        f"- Native comparison uses the matching {len(chain_ids)} L{lf} source indices from `{native_source.relative_to(PROJECT_ROOT)}`, not a different-volume ensemble. This is essential because volume-averaged observables have volume-dependent widths.",
        "",
        "## Step Sizes and Acceptance",
        "",
        "| Runs | coarse sigma | detail sigma | coarse acceptance | d01 | d10 | d11 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for info in sorted(run_info, key=lambda row: int(row["source_start"])):
        lines.append(f"| cfg{info['source_start']} | {info['coarse_sigma']:.3g} | {info['detail_sigma']:.3g} | {info['coarse_acceptance']:.3f} | {info['d01_acceptance']:.3f} | {info['d10_acceptance']:.3f} | {info['d11_acceptance']:.3f} |")
    groups = defaultdict(int)
    for info in run_info:
        groups[f"({info['coarse_sigma']:.3g}, {info['detail_sigma']:.3g})"] += 1
    group_text = ", ".join(f"{count} runs at `(coarse_sigma, detail_sigma)={label}`" for label, count in sorted(groups.items()))
    lines += [
        "",
        f"Step-size groups: {group_text}. Different reversible step sizes target the same distribution, but their acceptance and autocorrelation should be compared only within a step group.",
        "",
        f"## Native L{lf} Agreement",
        "",
        "The table reports mean difference in combined standard-error units, width ratio, and two-sample KS statistic.",
        "",
        "| Sweep | Observable | generated mean | native mean | mean z | width ratio | KS |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for sweep in (0, 50, 100, 400):
        for observable in ("action_density", "phi2", "phi4", "NN", "m2", "G_pmin_avg"):
            item = metric(sweep, observable)
            lines.append(f"| {sweep} | {observable} | {item['mean']:.6g} | {item['native_mean']:.6g} | {item['mean_z']:+.2f} | {item['width_ratio']:.3f} | {item['ks']:.3f} |")
    lines += [
        "",
        "Interpretation should use the sweep-resolved table rather than a comparison to a different-volume native ensemble. The post-100 autocorrelation table below quantifies the remaining serial correlation after the visible initialization transient.",
        "",
        "## Post-100 Autocorrelation",
        "",
        "Integrated autocorrelation times use a per-chain initial-positive-sequence estimate from measurements saved every five sweeps. They are reported in recorded sweeps. Because step sizes differ between the tuning and production-matched runs, compare only within a step group.",
        "",
        "| Observable | step group | chains | mean tau_int | median tau_int | mean lag-1 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in auto_rows:
        lines.append(f"| {row['observable']} | {row['step_group']} | {row['n_chains']} | {row['tau_int_sweeps_mean']:.1f} | {row['tau_int_sweeps_median']:.1f} | {row['lag1_mean']:.3f} |")
    lines += [
        "",
        "## Recommended Use",
        "",
        "- Choose burn-in from the sweep-resolved native comparison and the post-burn autocorrelation estimates for this volume and step group.",
        "- Treat persistent long-distance offsets separately from local-observable relaxation; more burn-in is not automatically the appropriate remedy.",
        "- For histogram comparisons, filter to a single saved sweep, normally sweep 100 or 400; do not histogram all 0–400 rows together.",
    ]
    (run / f"L{lc}toL{lf}_coordinate_mh_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
