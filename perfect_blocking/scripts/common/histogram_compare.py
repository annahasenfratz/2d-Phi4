from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path("perfect_blocking/logs/mplconfig").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


LABELS = {
    "phi2": r"$\phi^2$",
    "phi4": r"$\phi^4$",
    "m2": r"$m^2$",
    "m4": r"$m^4$",
    "Binder_U4_from_averages": r"$U_4$",
    "xi_over_L": r"$\xi/L$",
    "G_00": r"$G(0,0)$",
    "G_10": r"$G(2\pi/L,0)$",
    "G_01": r"$G(0,2\pi/L)$",
    "G_pmin_avg": r"$G(p_{\min})$",
}


def finite(x: np.ndarray) -> np.ndarray:
    return x[np.isfinite(x)]


def metrics(a: np.ndarray, b: np.ndarray, bins: int = 50) -> dict[str, Any]:
    a = finite(np.asarray(a, dtype=np.float64))
    b = finite(np.asarray(b, dtype=np.float64))
    lo = float(min(a.min(), b.min()))
    hi = float(max(a.max(), b.max()))
    if math.isclose(lo, hi):
        lo -= 0.5
        hi += 0.5
    hist_a, edges = np.histogram(a, bins=bins, range=(lo, hi), density=False)
    hist_b, _ = np.histogram(b, bins=edges, density=False)
    pa = hist_a / max(hist_a.sum(), 1)
    pb = hist_b / max(hist_b.sum(), 1)
    total_variation = 0.5 * float(np.sum(np.abs(pa - pb)))
    m = 0.5 * (pa + pb)
    js = 0.5 * _kl(pa, m) + 0.5 * _kl(pb, m)
    ks = stats.ks_2samp(a, b)
    std_a = float(np.std(a, ddof=1)) if len(a) > 1 else float("nan")
    std_b = float(np.std(b, ddof=1)) if len(b) > 1 else float("nan")
    eps = 1.0e-12
    pooled = math.sqrt(0.5 * (std_a * std_a + std_b * std_b)) if np.isfinite(std_a) and np.isfinite(std_b) else float("nan")
    return {
        "n_a": int(len(a)),
        "n_b": int(len(b)),
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "std_a": std_a,
        "std_b": std_b,
        "mean_difference_b_minus_a": float(np.mean(b) - np.mean(a)),
        "standardized_mean_shift": float((np.mean(b) - np.mean(a)) / pooled) if pooled > eps else float("nan"),
        "std_ratio_a_over_b": float(std_a / std_b) if std_b > eps else float("nan"),
        "total_variation": total_variation,
        "jensen_shannon": float(js),
        "wasserstein_1": float(stats.wasserstein_distance(a, b)),
        "ks_statistic": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
    }


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    mask = p > 0.0
    return float(np.sum(p[mask] * np.log(p[mask] / np.maximum(q[mask], 1.0e-300))))


def plot_histogram(
    a: np.ndarray,
    b: np.ndarray,
    observable: str,
    out_pdf: Path,
    bins: int = 50,
    label_a: str = "direct-generated L16",
    label_b: str = "native L32 blocked->L16",
) -> None:
    a = finite(np.asarray(a, dtype=np.float64))
    b = finite(np.asarray(b, dtype=np.float64))
    lo = float(min(a.min(), b.min()))
    hi = float(max(a.max(), b.max()))
    if math.isclose(lo, hi):
        lo -= 0.5
        hi += 0.5
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.2, 3.4), constrained_layout=True)
    ax.hist(
        b,
        bins=bins,
        range=(lo, hi),
        density=True,
        histtype="stepfilled",
        alpha=0.35,
        color="#1f77b4",
        edgecolor="#1f77b4",
        linewidth=1.5,
        label=label_b,
    )
    ax.hist(
        a,
        bins=bins,
        range=(lo, hi),
        density=True,
        histtype="step",
        color="#d62728",
        linewidth=1.6,
        label=label_a,
    )
    ax.set_xlabel(LABELS.get(observable, observable))
    ax.set_ylabel("density")
    ax.legend(frameon=False, fontsize=8)
    ax.tick_params(direction="in", top=True, right=True)
    fig.savefig(out_pdf)
    plt.close(fig)
