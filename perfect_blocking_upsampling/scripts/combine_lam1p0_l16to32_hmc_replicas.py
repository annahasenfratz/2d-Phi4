#!/usr/bin/env python3
"""Combine the two independent N=5000 L16->L32 HMC replicas per flow.

The original run directories remain immutable.  This script writes merged
per-configuration histories with an explicit replica identifier and pooled
N=10000 mean/SEM histories for the highcorr-5x5 and Ethan-7x7 comparisons.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RUNS = {
    "Highcorr 5x5": [
        ROOT / "perfect_blocking_upsampling/outputs/fine_hmc_lam1p0/global/L16toL32_highcorr5_pureNLL_directnative_N5000_S100_tau2_n28_eps2over28_20260818",
        ROOT / "perfect_blocking_upsampling/outputs/fine_hmc_lam1p0/global/L16toL32_highcorr5_pureNLL_directnative_replica2_N5000_S100_tau2_n28_eps2over28_20260818",
    ],
    "Ethan 7x7": [
        ROOT / "perfect_blocking_upsampling/outputs/fine_hmc_lam1p0/global/L16toL32_ethan7_fresh_pureNLL_directnative_N5000_S100_tau2_n28_eps2over28_20260818",
        ROOT / "perfect_blocking_upsampling/outputs/fine_hmc_lam1p0/global/L16toL32_ethan7_fresh_pureNLL_directnative_replica2_N5000_S100_tau2_n28_eps2over28_20260818",
    ],
}
STYLE = {"Highcorr 5x5": "#d73027", "Ethan 7x7": "#4575b4"}
PRIMARY = [("action_density", "action density"), ("phi2", r"$\phi^2$"), ("phi4", r"$\phi^4$"), ("local_kurtosis_ratio", r"$\langle\phi^4\rangle/\langle\phi^2\rangle^2$")]
SECONDARY = [("NN", "NN"), ("2nn", "2nn"), ("diag", "diag"), ("m2", r"$m^2$"), ("m4", r"$m^4$"), ("G_pmin_avg", r"$G(p_{\min})$")]
REF = {
    "action_density": -0.5545391886945952, "phi2": 0.8308959655815319,
    "phi4": 1.0548743233116198, "local_kurtosis_ratio": 1.5279069033365993,
    "NN": 0.5719330434120697, "2nn": 0.4823092137949688,
    "diag": 0.5141745628913674, "m2": 0.35823825750750404,
    "m4": 0.15088896263582652, "G_pmin_avg": 11.358745520092183,
}


def merge_rows(paths: list[Path], filename: str) -> pd.DataFrame:
    frames = []
    for replica, run in enumerate(paths, start=1):
        d = pd.read_csv(run / "observables" / filename)
        d.insert(0, "replica_id", replica)
        d.insert(1, "combined_chain_id", d["chain_id"].astype(np.int64) + (replica - 1) * 5000)
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def save_history_plot(summary: pd.DataFrame, items, shape: tuple[int, int], output: Path) -> None:
    fig, axes = plt.subplots(*shape, figsize=(5.5 * shape[1], 3.9 * shape[0]), sharex=True, constrained_layout=True)
    axes = np.ravel(axes)
    for ax, (key, title) in zip(axes, items):
        ax.axhline(REF[key], color="0.2", linestyle="--", label="direct native L32")
        for label in RUNS:
            d = summary[(summary.flow == label) & (summary.observable == key)].sort_values("sweep")
            ax.errorbar(d.sweep, d["mean"], yerr=d["sem"], marker="o", markersize=3.5, capsize=1.8, linewidth=1.7, color=STYLE[label], label=label)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("fine-field HMC sweep")
        ax.set_ylabel("pooled N=10000 mean")
        ax.grid(alpha=.18)
    for ax in axes[len(items):]:
        ax.remove()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(.5, 1.04))
    fig.suptitle("Independent-replica L16→L32 rethermalization", fontsize=16, y=1.09)
    fig.savefig(output.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=ROOT / "perfect_blocking_upsampling/outputs/flow_comparisons_lam1p0/L16toL32_highcorr5_ethan7_combined_N10000_20260819")
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for label, paths in RUNS.items():
        if any(not (p / "status.json").exists() for p in paths):
            raise FileNotFoundError(f"missing completed replica for {label}")
        tag = "highcorr5" if label.startswith("Highcorr") else "ethan7"
        flow_dir = a.out_dir / tag
        obs_dir = flow_dir / "observables"
        obs_dir.mkdir(parents=True, exist_ok=True)
        main_df = merge_rows(paths, "main_per_sweep_measurements.csv")
        g_df = merge_rows(paths, "Gk_per_sweep_measurements.csv")
        main_df["local_kurtosis_ratio"] = main_df["phi4"] / np.maximum(main_df["phi2"] ** 2, 1.0e-300)
        main_df.to_csv(obs_dir / "main_per_sweep_measurements_combined_N10000.csv", index=False)
        g_df.to_csv(obs_dir / "Gk_per_sweep_measurements_combined_N10000.csv", index=False)
        for key, _ in PRIMARY + SECONDARY:
            source = g_df if key == "G_pmin_avg" else main_df
            for sweep, values in source.groupby("sweep")[key]:
                v = values.to_numpy(float)
                summary.append({"flow": label, "observable": key, "sweep": int(sweep), "n": len(v), "mean": v.mean(), "std": v.std(ddof=1), "sem": v.std(ddof=1) / np.sqrt(len(v)), "native_mean": REF[key], "shift_from_native": v.mean() - REF[key]})
    summary = pd.DataFrame(summary)
    summary.to_csv(a.out_dir / "combined_time_history_summary_N10000.csv", index=False)

    save_history_plot(summary, PRIMARY, (2, 2), a.out_dir / "combined_primary_history_N10000")
    save_history_plot(summary, SECONDARY, (2, 3), a.out_dir / "combined_secondary_history_N10000")
    print(a.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
