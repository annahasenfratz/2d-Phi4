#!/usr/bin/env python3
"""Compare the common N=5000 L16->L32 HMC rethermalization histories.

The three runs differ only in the sweep-zero flow/kernel initialization.  All
subsequent updates are identical full-field fine HMC, tau=2.  The output CSV
contains ensemble means and standard errors at every saved sweep; the figures
overlay all three histories and the independent direct-L32 reference.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "perfect_blocking_upsampling" / "src"), str(ROOT / "perfect_blocking_upsampling" / "scripts")]

from perfect_blocking_upsampling.actions import ActionSpec  # noqa: E402
from train_lam1p0_flow_detail_pilot import load_phi, per_config_rows  # noqa: E402


RUNS = {
    "Alternating-KL 5x5": ROOT / "perfect_blocking_upsampling/outputs/fine_hmc_lam1p0/global/L16toL32_alternatingKL5_directnative_N5000_S100_tau2_n28_eps2over28_20260818",
    "Highcorr 5x5": ROOT / "perfect_blocking_upsampling/outputs/fine_hmc_lam1p0/global/L16toL32_highcorr5_pureNLL_directnative_N5000_S100_tau2_n28_eps2over28_20260818",
    "Ethan 7x7": ROOT / "perfect_blocking_upsampling/outputs/fine_hmc_lam1p0/global/L16toL32_ethan7_fresh_pureNLL_directnative_N5000_S100_tau2_n28_eps2over28_20260818",
}
STYLE = {
    "Alternating-KL 5x5": "#2ca25f",
    "Highcorr 5x5": "#d73027",
    "Ethan 7x7": "#4575b4",
}
PRIMARY = [
    ("action_density", "action density"),
    ("phi2", r"$\phi^2$"),
    ("phi4", r"$\phi^4$"),
    ("local_kurtosis_ratio", r"$\langle\phi^4\rangle/\langle\phi^2\rangle^2$"),
]
SECONDARY = [
    ("NN", "NN"), ("2nn", "2nn"), ("diag", "diag"),
    ("m2", r"$m^2$"), ("m4", r"$m^4$"), ("G_pmin_avg", r"$G(p_{\min})$"),
]


def reference(n: int) -> dict[str, tuple[float, float]]:
    """Independent native-L32 means and iid standard errors for first n fields."""
    fields = load_phi(ROOT / "data/configs_phi4_2d/lam1p0_kappac0p340301_L32/configs.npz")[:n].astype(np.float32)
    main, g = per_config_rows(fields, ActionSpec("phi4_nn", 1.0, 0.340301), "native_L32")
    refs: dict[str, tuple[float, float]] = {}
    for key, rows in [(key, main) for key, _ in PRIMARY + SECONDARY if key != "G_pmin_avg"] + [("G_pmin_avg", g)]:
        values = np.asarray([r[key] for r in rows], dtype=float)
        refs[key] = (float(values.mean()), float(values.std(ddof=1) / np.sqrt(len(values))))
    return refs


def history(label: str, run: Path) -> pd.DataFrame:
    main = pd.read_csv(run / "observables" / "main_per_sweep_measurements.csv")
    g = pd.read_csv(run / "observables" / "Gk_per_sweep_measurements.csv")
    # This derived per-configuration observable is present in the sweep-zero
    # audit but not in the HMC measurement CSV.
    main["local_kurtosis_ratio"] = main["phi4"] / np.maximum(main["phi2"] ** 2, 1.0e-300)
    rows = []
    for key, _ in PRIMARY + SECONDARY:
        source = g if key == "G_pmin_avg" else main
        for sweep, values in source.groupby("sweep")[key]:
            values = values.to_numpy(dtype=float)
            rows.append({
                "flow": label, "sweep": int(sweep), "observable": key,
                "n": len(values), "mean": values.mean(),
                "std": values.std(ddof=1), "sem": values.std(ddof=1) / np.sqrt(len(values)),
            })
    return pd.DataFrame(rows)


def plot_group(df: pd.DataFrame, refs: dict[str, tuple[float, float]], items, path: Path, title: str, shape: tuple[int, int]) -> None:
    fig, axes = plt.subplots(*shape, figsize=(5.8 * shape[1], 3.9 * shape[0]), sharex=True, constrained_layout=True)
    axes = np.ravel(axes)
    for ax, (key, xlabel) in zip(axes, items):
        ref, ref_sem = refs[key]
        ax.axhspan(ref - ref_sem, ref + ref_sem, color="0.3", alpha=0.12, zorder=0)
        ax.axhline(ref, color="0.2", linewidth=1.25, linestyle="--", label="direct native L32")
        for label in RUNS:
            d = df[(df.flow == label) & (df.observable == key)].sort_values("sweep")
            ax.errorbar(d.sweep, d["mean"], yerr=d["sem"], color=STYLE[label], marker="o", markersize=3.4,
                        linewidth=1.6, capsize=1.8, label=label)
        ax.set_title(xlabel, fontsize=14)
        ax.set_xlabel("fine-field HMC sweep")
        ax.set_ylabel("ensemble mean")
        ax.grid(alpha=0.18)
    for ax in axes[len(items):]:
        ax.remove()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle(title, fontsize=16, y=1.08)
    fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=ROOT / "perfect_blocking_upsampling/outputs/flow_comparisons_lam1p0/L16toL32_threeflow_hmc_20260818")
    ap.add_argument("--native-count", type=int, default=10000, help="Number of direct native L32 fields used for the reference band.")
    args = ap.parse_args()
    for path in RUNS.values():
        if not (path / "status.json").exists():
            raise FileNotFoundError(f"missing completed run: {path}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.concat([history(label, path) for label, path in RUNS.items()], ignore_index=True)
    refs = reference(args.native_count)
    df["native_mean"] = df.observable.map(lambda x: refs[x][0])
    df["native_sem"] = df.observable.map(lambda x: refs[x][1])
    df["mean_shift_from_native"] = df["mean"] - df["native_mean"]
    df.to_csv(args.out_dir / "time_history_summary.csv", index=False)
    pd.DataFrame([
        {"observable": k, "native_mean": v[0], "native_sem": v[1], "n": args.native_count}
        for k, v in refs.items()
    ]).to_csv(args.out_dir / "native_reference_summary.csv", index=False)
    plot_group(df, refs, PRIMARY, args.out_dir / "time_history_primary", "L16→L32: thermalization of three flow initializers", (2, 2))
    plot_group(df, refs, SECONDARY, args.out_dir / "time_history_secondary", "L16→L32: thermalization of three flow initializers", (2, 3))
    print(args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
