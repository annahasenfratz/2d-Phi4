#!/usr/bin/env python3
# %% [markdown]
# # Lambda=1 perfect-blocking observable overlays
#
# Run this file directly, or open it in JupyterLab and execute its `# %%` cells.
# It overlays direct-generated L16 observables with L32 fields blocked to L16
# using the specified 5x5 kernel.  The default inputs reproduce the latest
# 500-configuration, local-loss-plus-correct-m2 optimization study.

# %%
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
os.environ.setdefault("MPLCONFIGDIR", str((PROJECT_ROOT / "perfect_blocking/logs/mplconfig").resolve()))

import matplotlib.pyplot as plt
import numpy as np

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.blocking import block_configs, load_configs  # noqa: E402
from scripts.common.histogram_compare import LABELS, metrics  # noqa: E402
from scripts.common.kernel_io import load_kernel  # noqa: E402
from scripts.run_lam1p0_7x7_kernel_search import observable_arrays  # noqa: E402


DIRECT_DEFAULT = PROJECT_ROOT / "data/configs_phi4_2d/lam1p0_kappac0p340301_L16/configs.npz"
FINE_DEFAULT = PROJECT_ROOT / "data/configs_phi4_2d/lam1p0_kappac0p340301_L32/configs.npz"
KERNEL_DEFAULT = PROJECT_ROOT / "perfect_blocking/perfect_blocking_lam1p0/kernels/candidates/local_loss_plus_correct_m2_train500_20260803/best_phi2_support_balanced_eta_included.json"
OUTPUT_DEFAULT = PROJECT_ROOT / "perfect_blocking/perfect_blocking_lam1p0/tests/intermediate/local_loss_plus_correct_m2_train500_20260803/overlay_histograms"

# m4 and the correlators are validation-only in the optimization, but all are
# included here so the plots expose any tradeoffs directly.
OBSERVABLES = [
    "phi2",
    "phi4",
    "local_kurtosis_ratio",
    "action_density",
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

DISPLAY_LABELS = {
    "local_kurtosis_ratio": r"$\langle\phi^4\rangle/\langle\phi^2\rangle^2$",
    "action_density": "action density",
    "NN": "nearest-neighbor bond",
    "2nn": "distance-two bond",
    "diag": "diagonal bond",
    "m": r"$m=V^{-1}\sum_x\phi_x$",
    **LABELS,
}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot direct-L16 and blocked-L32-to-L16 observable overlays.")
    p.add_argument("--direct-configs", type=Path, default=DIRECT_DEFAULT)
    p.add_argument("--fine-configs", type=Path, default=FINE_DEFAULT)
    p.add_argument("--kernel", type=Path, default=KERNEL_DEFAULT)
    p.add_argument("--n-configs", type=int, default=500)
    p.add_argument("--bins", type=int, default=40)
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DEFAULT)
    return p


def finite_pair(direct: np.ndarray, blocked: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    direct = np.asarray(direct, dtype=np.float64)
    blocked = np.asarray(blocked, dtype=np.float64)
    direct = direct[np.isfinite(direct)]
    blocked = blocked[np.isfinite(blocked)]
    if not len(direct) or not len(blocked):
        raise ValueError("observable has no finite samples")
    lo, hi = float(min(direct.min(), blocked.min())), float(max(direct.max(), blocked.max()))
    if math.isclose(lo, hi):
        lo, hi = lo - 0.5, hi + 0.5
    pad = 0.025 * (hi - lo)
    return direct, blocked, (lo - pad, hi + pad)


def overlay(ax: plt.Axes, direct: np.ndarray, blocked: np.ndarray, observable: str, bins: int) -> dict[str, float]:
    direct, blocked, interval = finite_pair(direct, blocked)
    ax.hist(blocked, bins=bins, range=interval, density=True, histtype="stepfilled", alpha=0.32, color="#1f77b4", label="L32 blocked → L16")
    ax.hist(direct, bins=bins, range=interval, density=True, histtype="step", linewidth=1.7, color="#d62728", label="direct-generated L16")
    summary = metrics(direct, blocked, bins=bins)
    ax.set_title(DISPLAY_LABELS.get(observable, observable), fontsize=10)
    ax.set_ylabel("density")
    ax.grid(alpha=0.18, linewidth=0.6)
    ax.text(0.98, 0.96, f"KS={summary['ks_statistic']:.3f}\nstd ratio={summary['std_ratio_a_over_b']:.3f}", transform=ax.transAxes, ha="right", va="top", fontsize=8, bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.78, "edgecolor": "0.75"})
    return summary


def main() -> None:
    args = parser().parse_args()
    if args.n_configs <= 0:
        raise ValueError("--n-configs must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    direct_fields = load_configs(args.direct_configs, args.n_configs)
    fine_fields = load_configs(args.fine_configs, args.n_configs)
    if len(direct_fields) != args.n_configs or len(fine_fields) != args.n_configs:
        raise RuntimeError(f"need {args.n_configs} configurations; found direct={len(direct_fields)}, fine={len(fine_fields)}")
    kernel = load_kernel(args.kernel)
    blocked_fields = block_configs(fine_fields, kernel)
    if blocked_fields.shape[1:] != direct_fields.shape[1:]:
        raise ValueError(f"blocked lattice {blocked_fields.shape[1:]} does not match direct lattice {direct_fields.shape[1:]}")

    direct_obs = observable_arrays(direct_fields)
    blocked_obs = observable_arrays(blocked_fields)
    summary_rows: list[dict[str, float | str]] = []

    # %% Individual publication-quality overlays
    for observable in OBSERVABLES:
        fig, ax = plt.subplots(figsize=(5.4, 3.6), constrained_layout=True)
        summary = overlay(ax, direct_obs[observable], blocked_obs[observable], observable, args.bins)
        ax.set_xlabel(DISPLAY_LABELS.get(observable, observable))
        ax.legend(frameon=False, fontsize=8)
        fig.savefig(args.output_dir / f"overlay_{observable}.pdf")
        plt.close(fig)
        summary_rows.append({"observable": observable, **summary})

    # %% Compact overview for JupyterLab and manuscript review
    ncols = 3
    nrows = math.ceil(len(OBSERVABLES) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.05 * nrows), constrained_layout=True)
    for ax, observable in zip(axes.flat, OBSERVABLES):
        overlay(ax, direct_obs[observable], blocked_obs[observable], observable, args.bins)
        ax.set_xlabel(DISPLAY_LABELS.get(observable, observable), fontsize=9)
    for ax in axes.flat[len(OBSERVABLES):]:
        ax.set_visible(False)
    axes.flat[0].legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle("λ=1: direct-generated L16 vs. L32 blocked to L16", fontsize=13)
    fig.savefig(args.output_dir / "all_observables_overlay.pdf")
    plt.close(fig)

    with (args.output_dir / "histogram_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"wrote {len(OBSERVABLES)} individual PDFs and all_observables_overlay.pdf to {args.output_dir}")


if __name__ == "__main__":
    main()
