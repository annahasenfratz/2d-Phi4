#!/usr/bin/env python3
"""Cluster-bootstrap a combined coordinate-MH ensemble against native data.

The combined coordinate-MH directory retains source_run for each chain.  Those
runs, rather than arbitrary consecutive chunks of chains, are the independent
bootstrap units.  Native configurations are grouped into consecutive blocks to
respect their Markov ordering.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "perfect_blocking_upsampling" / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "perfect_blocking_upsampling" / "scripts"))

from analyze_coarse_detail_run_partial import OBS_KEYS, aggregate_measurements  # noqa: E402


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def groups_from_run_inventory(rows: list[dict[str, str]], inventory: list[dict[str, str]]) -> list[list[dict[str, str]]]:
    source_run_by_chain = {int(r["chain_id"]): r["source_run"] for r in inventory}
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(source_run_by_chain[int(row["chain_id"])], []).append(row)
    return list(groups.values())


def consecutive_groups(rows: list[dict[str, str]], size: int) -> list[list[dict[str, str]]]:
    n = len(rows) // size
    if n < 2:
        raise ValueError(f"bin size {size} leaves fewer than two groups")
    return [rows[i * size : (i + 1) * size] for i in range(n)]


RAW_KEYS = ("m", "m2", "m4", "phi2", "phi4", "NN", "2nn", "diag", "action_density", "G_pmin_x_cfg", "G_pmin_y_cfg")


def derived_from_means(means: dict[str, np.ndarray], L: int) -> dict[str, np.ndarray]:
    """Apply the analysis aggregate convention to scalar or vector means."""
    m = means["m"]
    m2 = means["m2"]
    m4 = means["m4"]
    phi2 = means["phi2"]
    phi4 = means["phi4"]
    gpx = means["G_pmin_x_cfg"]
    gpy = means["G_pmin_y_cfg"]
    gp = 0.5 * (gpx + gpy)
    chi = float(L * L) * np.maximum(m2 - m * m, 0.0)
    sqrt_arg = chi / gp - 1.0
    xi = np.where(
        sqrt_arg > 0.0,
        np.sqrt(sqrt_arg) / (2.0 * L * np.sin(np.pi / L)),
        np.nan,
    )
    binder = 1.0 - m4 / np.maximum(3.0 * m2 * m2, 1.0e-300)
    local_kurtosis = phi4 / np.maximum(phi2 * phi2, 1.0e-300)
    return {
        "Binder_U4_from_averages": binder,
        "Binder_U4": binder,
        "xi_over_L": xi,
        "chi": chi,
        "susceptibility_connected": chi,
        "m2": m2,
        "m4": m4,
        "phi2": phi2,
        "phi4": phi4,
        "NN": means["NN"],
        "2nn": means["2nn"],
        "diag": means["diag"],
        "action_density": means["action_density"],
        "local_kurtosis_ratio_from_averages": local_kurtosis,
    }


def bootstrap(groups: list[list[dict[str, str]]], nboot: int, seed: int) -> tuple[dict[str, float], dict[str, float]]:
    if len(groups) < 2:
        raise ValueError("need at least two bootstrap groups")
    point = aggregate_measurements([row for group in groups for row in group])
    L = int(float(groups[0][0]["L"]))
    group_counts = np.asarray([len(group) for group in groups], dtype=np.float64)
    group_sums = np.asarray(
        [[sum(float(row[key]) for row in group) for key in RAW_KEYS] for group in groups],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    keys = list(OBS_KEYS)
    n_groups = len(groups)
    chosen = rng.integers(0, n_groups, size=(nboot, n_groups))
    total_counts = group_counts[chosen].sum(axis=1)
    total_sums = group_sums[chosen].sum(axis=1)
    means = {key: total_sums[:, i] / total_counts for i, key in enumerate(RAW_KEYS)}
    samples = derived_from_means(means, L)
    errors = {key: float(np.nanstd(samples[key], ddof=1)) for key in keys}
    return point, errors


def atomic_write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("combined_run", type=Path)
    parser.add_argument("native_csv", type=Path)
    parser.add_argument("--sweep", type=int, default=400)
    parser.add_argument("--native-block-size", type=int, default=50)
    parser.add_argument("--nboot", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260723)
    args = parser.parse_args()

    run = args.combined_run.resolve()
    rows = [row for row in read_rows(run / "observables" / "per_sweep_observables.csv") if int(row["sweep"]) == args.sweep]
    inventory = read_rows(run / "source_index_inventory.csv")
    native_rows = read_rows(args.native_csv.resolve())
    upscaled_groups = groups_from_run_inventory(rows, inventory)
    native_groups = consecutive_groups(native_rows, args.native_block_size)
    upscaled, upscaled_err = bootstrap(upscaled_groups, args.nboot, args.seed)
    native, native_err = bootstrap(native_groups, args.nboot, args.seed + 1)

    output = run / "partial_analysis"
    output.mkdir(parents=True, exist_ok=True)
    comparison: list[dict[str, Any]] = []
    for key in OBS_KEYS:
        combined_error = float(np.hypot(native_err[key], upscaled_err[key]))
        delta = upscaled[key] - native[key]
        comparison.append({
            "observable": key,
            "native_mean": native[key],
            "native_cluster_bootstrap_error": native_err[key],
            "upscaled_mean": upscaled[key],
            "upscaled_run_cluster_bootstrap_error": upscaled_err[key],
            "difference_upscaled_minus_native": delta,
            "combined_error": combined_error,
            "combined_z": delta / combined_error if combined_error else float("nan"),
        })
    result_csv = output / f"run_cluster_bootstrap_vs_native_sweep{args.sweep}.csv"
    atomic_write_csv(result_csv, comparison)
    metadata = {
        "combined_run": str(run), "native_csv": str(args.native_csv.resolve()), "sweep": args.sweep,
        "n_bootstrap_replicates": args.nboot, "seed": args.seed,
        "upscaled_independent_run_clusters": len(upscaled_groups),
        "upscaled_chain_count": len(rows),
        "native_consecutive_block_size": args.native_block_size,
        "native_independent_blocks": len(native_groups),
        "native_configs_used": len(native_groups) * args.native_block_size,
        "native_configs_discarded": len(native_rows) % args.native_block_size,
    }
    (output / f"run_cluster_bootstrap_vs_native_sweep{args.sweep}.json").write_text(
        json.dumps({"metadata": metadata, "comparison": comparison}, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))
    for row in comparison:
        print(f"{row['observable']:38s} z={row['combined_z']:+.3f}  delta={row['difference_upscaled_minus_native']:+.8g}")


if __name__ == "__main__":
    main()
