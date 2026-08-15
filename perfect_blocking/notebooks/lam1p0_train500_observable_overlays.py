# %% [markdown]
# # λ=1 direct-L16 versus blocked-L32→L16 observable overlays
#
# Copy each `# %%` block below into a separate JupyterLab cell, in order.
# The analysis uses the 500 configurations and kernel from the
# `local_loss_plus_correct_m2_train500_20260803` study.

# %% Imports and fixed input files
from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Image, display
from scipy import stats

# This explicit path makes the notebook independent of the JupyterLab launch
# directory.  Change only this line if the project is moved again.
PROJECT_ROOT = Path("/Users/anna/Work/Research/Normalizing-flow/Inverse_RG/phi4_inverse_blocking")
if not (PROJECT_ROOT / "perfect_blocking").is_dir():
    # Fallback for a moved checkout launched from within the project tree.
    PROJECT_ROOT = next(
        (candidate for candidate in (Path.cwd().resolve(), *Path.cwd().resolve().parents)
         if (candidate / "perfect_blocking").is_dir()),
        None,
    )
if PROJECT_ROOT is None:
    raise RuntimeError("Set PROJECT_ROOT to the directory containing perfect_blocking/.")

PERFECT_BLOCKING_ROOT = PROJECT_ROOT / "perfect_blocking"
if str(PERFECT_BLOCKING_ROOT) not in sys.path:
    sys.path.insert(0, str(PERFECT_BLOCKING_ROOT))

from scripts.common.blocking import block_configs, load_configs
from scripts.common.kernel_io import load_kernel

N_CONFIGS = 500
DIRECT_L16_FILE = PROJECT_ROOT / "data/configs_phi4_2d/lam1p0_kappac0p340301_L16/configs.npz"
FINE_L32_FILE = PROJECT_ROOT / "data/configs_phi4_2d/lam1p0_kappac0p340301_L32/configs.npz"
KERNEL_FILE = PROJECT_ROOT / "perfect_blocking/perfect_blocking_lam1p0/kernels/candidates/local_loss_plus_correct_m2_train500_20260803/best_phi2_support_balanced_eta_included.json"
PLOT_DIR = PROJECT_ROOT / "perfect_blocking/perfect_blocking_lam1p0/tests/intermediate/local_loss_plus_correct_m2_train500_20260803/jupyter_overlays"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

OBSERVABLES = [
    "phi2", "phi4", "local_kurtosis_ratio", "action_density",
    "NN", "2nn", "diag", "m", "m2", "m4",
    "G_00", "G_10", "G_01", "G_pmin_avg",
]

DISPLAY_LABELS = {
    "local_kurtosis_ratio": r"$\langle\phi^4\rangle/\langle\phi^2\rangle^2$",
    "action_density": "action density",
    "NN": "nearest-neighbor bond",
    "2nn": "distance-two bond",
    "diag": "diagonal bond",
    "m": r"$m=V^{-1}\sum_x\phi_x$",
    "phi2": r"$\phi^2$",
    "phi4": r"$\phi^4$",
    "m2": r"$m^2$",
    "m4": r"$m^4$",
    "G_00": r"$G(0,0)$",
    "G_10": r"$G(2\pi/L,0)$",
    "G_01": r"$G(0,2\pi/L)$",
    "G_pmin_avg": r"$G(p_{\min})$",
}


# %% Reusable functions
def finite_pair(direct_values: np.ndarray, blocked_values: np.ndarray):
    """Return finite direct/blocked samples and a shared padded histogram range."""
    direct_values = np.asarray(direct_values, dtype=float)
    blocked_values = np.asarray(blocked_values, dtype=float)
    direct_values = direct_values[np.isfinite(direct_values)]
    blocked_values = blocked_values[np.isfinite(blocked_values)]
    if len(direct_values) == 0 or len(blocked_values) == 0:
        raise ValueError("No finite values available for this observable.")
    lo = min(direct_values.min(), blocked_values.min())
    hi = max(direct_values.max(), blocked_values.max())
    if math.isclose(lo, hi):
        lo, hi = lo - 0.5, hi + 0.5
    pad = 0.025 * (hi - lo)
    return direct_values, blocked_values, (lo - pad, hi + pad)


def observable_arrays(fields: np.ndarray, *, lam: float = 1.0, kappa: float = 0.340301) -> dict[str, np.ndarray]:
    """Per-configuration observables, kept local so Jupyter retains its inline backend."""
    arr = np.asarray(fields, dtype=np.float64)
    n, L, _ = arr.shape
    volume = L * L
    m = arr.mean(axis=(1, 2))
    phi2 = np.mean(arr**2, axis=(1, 2))
    phi4 = np.mean(arr**4, axis=(1, 2))
    nn = 0.5 * (
        np.mean(arr * np.roll(arr, -1, axis=1), axis=(1, 2))
        + np.mean(arr * np.roll(arr, -1, axis=2), axis=(1, 2))
    )
    twonn = 0.5 * (
        np.mean(arr * np.roll(arr, -2, axis=1), axis=(1, 2))
        + np.mean(arr * np.roll(arr, -2, axis=2), axis=(1, 2))
    )
    diag = np.mean(arr * np.roll(np.roll(arr, -1, axis=1), -1, axis=2), axis=(1, 2))
    action_density = np.mean(
        (1.0 - 2.0 * lam) * arr**2 + lam * arr**4
        - 2.0 * kappa * (arr * np.roll(arr, -1, axis=1) + arr * np.roll(arr, -1, axis=2)),
        axis=(1, 2),
    )
    phase = np.exp(2j * np.pi * np.arange(L) / L)
    phi_x = np.tensordot(arr, phase, axes=([1], [0])).sum(axis=1)
    phi_y = np.tensordot(arr, phase, axes=([2], [0])).sum(axis=1)
    g10 = np.abs(phi_x) ** 2 / float(volume)
    g01 = np.abs(phi_y) ** 2 / float(volume)
    return {
        "phi2": phi2,
        "phi4": phi4,
        "local_kurtosis_ratio": phi4 / np.maximum(phi2**2, 1.0e-300),
        "action_density": action_density,
        "NN": nn,
        "2nn": twonn,
        "diag": diag,
        "m": m,
        "m2": m**2,
        "m4": m**4,
        "G_00": volume * m**2,
        "G_10": g10,
        "G_01": g01,
        "G_pmin_avg": 0.5 * (g10 + g01),
    }


def histogram_metrics(direct_values: np.ndarray, blocked_values: np.ndarray, *, bins: int) -> dict[str, float]:
    """Distribution-agreement metrics, implemented locally to keep Jupyter interactive."""
    direct_values, blocked_values, value_range = finite_pair(direct_values, blocked_values)
    direct_counts, edges = np.histogram(direct_values, bins=bins, range=value_range)
    blocked_counts, _ = np.histogram(blocked_values, bins=edges)
    p = direct_counts / direct_counts.sum()
    q = blocked_counts / blocked_counts.sum()
    midpoint = 0.5 * (p + q)
    kl = lambda a, b: np.sum(a[a > 0] * np.log(a[a > 0] / b[a > 0]))
    pooled_std = np.sqrt(0.5 * (np.var(direct_values, ddof=1) + np.var(blocked_values, ddof=1)))
    return {
        "n_direct": len(direct_values),
        "n_blocked": len(blocked_values),
        "direct_mean": float(np.mean(direct_values)),
        "blocked_mean": float(np.mean(blocked_values)),
        "direct_std": float(np.std(direct_values, ddof=1)),
        "blocked_std": float(np.std(blocked_values, ddof=1)),
        "standardized_mean_shift": float((np.mean(blocked_values) - np.mean(direct_values)) / pooled_std),
        "std_ratio_direct_over_blocked": float(np.std(direct_values, ddof=1) / np.std(blocked_values, ddof=1)),
        "total_variation": float(0.5 * np.abs(p - q).sum()),
        "jensen_shannon": float(0.5 * kl(p, midpoint) + 0.5 * kl(q, midpoint)),
        "wasserstein_1": float(stats.wasserstein_distance(direct_values, blocked_values)),
        "ks_statistic": float(stats.ks_2samp(direct_values, blocked_values).statistic),
        "ks_pvalue": float(stats.ks_2samp(direct_values, blocked_values).pvalue),
    }


def plot_overlay(direct_values, blocked_values, observable, *, bins=40, ax=None):
    """Overlay direct-generated L16 and blocked L32→L16 histograms on `ax`."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5.4, 3.6), constrained_layout=True)
    direct_values, blocked_values, value_range = finite_pair(direct_values, blocked_values)
    ax.hist(
        blocked_values, bins=bins, range=value_range, density=True,
        histtype="stepfilled", alpha=0.32, color="#1f77b4",
        label="native L32 blocked → L16",
    )
    ax.hist(
        direct_values, bins=bins, range=value_range, density=True,
        histtype="step", linewidth=1.7, color="#d62728",
        label="direct-generated L16",
    )
    result = histogram_metrics(direct_values, blocked_values, bins=bins)
    ax.axvline(
        result["direct_mean"], color="#d62728", linestyle="--", linewidth=1.35,
        label=fr"direct mean = {result['direct_mean']:.5g}",
    )
    ax.axvline(
        result["blocked_mean"], color="#1f77b4", linestyle="--", linewidth=1.35,
        label=fr"blocked mean = {result['blocked_mean']:.5g}",
    )
    ax.set_title(DISPLAY_LABELS.get(observable, observable))
    ax.set_xlabel(DISPLAY_LABELS.get(observable, observable))
    ax.set_ylabel("density")
    ax.grid(alpha=0.18, linewidth=0.6)
    # Keep this opposite the legend (which is explicitly upper-right below).
    ax.text(
        0.02, 0.96,
        f"KS = {result['ks_statistic']:.3f}\nstd ratio = {result['std_ratio_direct_over_blocked']:.3f}",
        transform=ax.transAxes, ha="left", va="top", fontsize=8,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.8, "edgecolor": "0.75"},
    )
    return ax, result


def save_overlay(direct_obs, blocked_obs, observable, *, bins=40, output_dir=PLOT_DIR):
    """Create one publication-quality PDF and return its metric dictionary."""
    fig, ax = plt.subplots(figsize=(5.4, 3.6), constrained_layout=True)
    _, result = plot_overlay(direct_obs[observable], blocked_obs[observable], observable, bins=bins, ax=ax)
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    fig.savefig(output_dir / f"overlay_{observable}.pdf")
    png_path = output_dir / f"overlay_{observable}.png"
    fig.savefig(png_path, dpi=180)
    plt.close(fig)
    display(Image(filename=str(png_path)))
    return {"observable": observable, **result}


# %% Load the given data files and form the blocked L32→L16 comparison ensemble
direct_fields = load_configs(DIRECT_L16_FILE, N_CONFIGS)
fine_fields = load_configs(FINE_L32_FILE, N_CONFIGS)
kernel = load_kernel(KERNEL_FILE)
blocked_fields = block_configs(fine_fields, kernel)

assert direct_fields.shape == blocked_fields.shape, (direct_fields.shape, blocked_fields.shape)
direct_observables = observable_arrays(direct_fields)
blocked_observables = observable_arrays(blocked_fields)

print(f"Direct fields:  {direct_fields.shape} from {DIRECT_L16_FILE}")
print(f"Blocked fields: {blocked_fields.shape} from {FINE_L32_FILE}")
print(f"Kernel: {KERNEL_FILE.name}")


# %% One reusable analysis call: select any observable and inspect/save its overlay
observable = "m2"  # Change, for example, to "phi2", "m4", "G_pmin_avg", or "NN".
single_metrics = save_overlay(direct_observables, blocked_observables, observable, bins=40)
pd.DataFrame([single_metrics])


# %% Analyse every observable: one PDF per observable plus a sortable metrics table
all_metrics = [
    save_overlay(direct_observables, blocked_observables, observable, bins=40)
    for observable in OBSERVABLES
]
metrics_table = pd.DataFrame(all_metrics).sort_values("ks_statistic", ascending=False)
metrics_table.to_csv(PLOT_DIR / "histogram_metrics.csv", index=False)
metrics_table


# %% Compact all-observable figure, also saved as a PDF
ncols = 3
nrows = math.ceil(len(OBSERVABLES) / ncols)
fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.05 * nrows), constrained_layout=True)
for ax, observable in zip(axes.flat, OBSERVABLES):
    plot_overlay(direct_observables[observable], blocked_observables[observable], observable, bins=40, ax=ax)
for ax in axes.flat[len(OBSERVABLES):]:
    ax.set_visible(False)
axes.flat[0].legend(loc="upper right", frameon=False, fontsize=8)
fig.suptitle("λ=1: direct-generated L16 vs. native L32 blocked to L16", fontsize=13)
fig.savefig(PLOT_DIR / "all_observables_overlay.pdf")
overview_png = PLOT_DIR / "all_observables_overlay.png"
fig.savefig(overview_png, dpi=180)
plt.close(fig)
display(Image(filename=str(overview_png)))
