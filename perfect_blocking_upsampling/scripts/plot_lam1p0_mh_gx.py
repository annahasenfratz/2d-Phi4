#!/usr/bin/env python3
"""Plot the axis two-point function from saved lambda=1 MH restart states."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def axis_correlator(fields: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Connected C(x,0), translationally averaged per configuration."""
    arr = np.asarray(fields, dtype=np.float64)
    n, length, _ = arr.shape
    ft = np.fft.fft2(arr, axes=(1, 2))
    # ifft(|phi_k|^2)/V is the spatial average at each displacement.
    per_config = np.fft.ifft2(np.abs(ft) ** 2, axes=(1, 2)).real / (length * length)
    per_config -= arr.mean() ** 2
    values = per_config[:, : length // 2 + 1, 0]
    return values.mean(axis=0), values.std(axis=0, ddof=1) / np.sqrt(n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("combined_run", type=Path, help="Combined output directory with combined_manifest.json")
    args = ap.parse_args()
    run = args.combined_run.resolve()
    manifest = json.loads((run / "combined_manifest.json").read_text())
    cfg = yaml.safe_load((run / "run_config.yaml").read_text())

    fields_by_sweep: dict[int, list[np.ndarray]] = {}
    for relative in manifest["source_runs"]:
        source = PROJECT_ROOT / relative
        with np.load(source / "checkpoints" / "checkpoint_latest.npz", allow_pickle=True) as state:
            meta = json.loads(str(state["meta"].item()))
            checkpoint_sweep = int(meta["completed_sweeps"])
            fields_by_sweep.setdefault(checkpoint_sweep, []).append(state["phi_current"].astype(np.float64))
    native_path = PROJECT_ROOT / cfg["fine_config_source"]
    with np.load(native_path) as native_file:
        native = native_file["phi"].astype(np.float64)
    native_mean, native_se = axis_correlator(native)
    x = np.arange(len(native_mean))

    analysis = run / "analysis"
    analysis.mkdir(exist_ok=True)
    sweep_tag = "_".join(f"sweep{sweep}" for sweep in sorted(fields_by_sweep))
    out_csv = analysis / f"Gx_axis_{sweep_tag}_vs_native.csv"
    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["series", "checkpoint_sweep", "x", "mean", "se"])
        writer.writeheader()
        writer.writerows({"series": "direct_native", "checkpoint_sweep": "", "x": int(r), "mean": a, "se": b}
                         for r, a, b in zip(x, native_mean, native_se))
        for checkpoint_sweep, groups in sorted(fields_by_sweep.items()):
            mh_mean, mh_se = axis_correlator(np.concatenate(groups, axis=0))
            writer.writerows({"series": "MH", "checkpoint_sweep": checkpoint_sweep, "x": int(r), "mean": a, "se": b}
                             for r, a, b in zip(x, mh_mean, mh_se))

    fig, ax = plt.subplots(figsize=(6.1, 4.1), constrained_layout=True)
    ax.errorbar(x, native_mean, yerr=native_se, fmt="o-", color="black", ms=3.5, lw=1.4, capsize=2,
                label=f"direct native L32 (N={len(native)})")
    for color, (checkpoint_sweep, groups) in zip(["C3", "C0", "C2", "C4"], sorted(fields_by_sweep.items())):
        mh = np.concatenate(groups, axis=0)
        mh_mean, mh_se = axis_correlator(mh)
        ax.errorbar(x, mh_mean, yerr=mh_se, fmt="s-", color=color, ms=3.5, lw=1.4, capsize=2,
                    label=f"MH checkpoint sweep {checkpoint_sweep} (N={len(mh)})")
    ax.set_yscale("log")
    ax.set_xlabel(r"axis separation $x$")
    ax.set_ylabel(r"connected $G(x,0)$")
    ax.set_xticks(x)
    ax.tick_params(direction="in", top=True, right=True)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("L16-to-L32 MH two-point function")
    fig.savefig(analysis / f"Gx_axis_{sweep_tag}_vs_native.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
