#!/usr/bin/env python3
"""Plot direct-L32 and alternating-KL L16->L32 proposal distributions."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "perfect_blocking_upsampling/src"), str(ROOT / "perfect_blocking_upsampling/scripts")]

from perfect_blocking_upsampling.actions import action_total
from perfect_blocking_upsampling.io import ActionSpec
from train_lam1p0_flow_detail_pilot import load_phi


BASE = ROOT / "perfect_blocking_upsampling/outputs/global_ar_lam1p0/L16to32_alternatingKL_highcorr5_r1_iterations2to4"
NATIVE = ROOT / "data/configs_phi4_2d/lam1p0_kappac0p340301_L32/configs.npz"
ACTION = ActionSpec("phi4_nn", 1.0, 0.340301)
LABELS = {
    "action_density": "action density",
    "phi2": r"$\langle\phi^2\rangle_{\rm cfg}$",
    "phi4": r"$\langle\phi^4\rangle_{\rm cfg}$",
    "NN": "NN",
    "local_kurtosis_ratio": r"$\langle\phi^4\rangle_{\rm cfg}/\langle\phi^2\rangle_{\rm cfg}^2$",
}
COLORS = {"iteration2": "#d62728", "iteration3": "#2ca02c", "iteration4": "#9467bd"}


def native_observables(phi: np.ndarray) -> dict[str, np.ndarray]:
    phi = np.asarray(phi, dtype=np.float64)
    phi2 = (phi * phi).mean(axis=(1, 2))
    phi4 = (phi**4).mean(axis=(1, 2))
    return {
        "action_density": action_total(phi, ACTION) / (phi.shape[1] * phi.shape[2]),
        "phi2": phi2,
        "phi4": phi4,
        "NN": 0.5 * (phi * np.roll(phi, -1, 1) + phi * np.roll(phi, -1, 2)).mean(axis=(1, 2)),
        "local_kurtosis_ratio": phi4 / np.maximum(phi2 * phi2, 1.0e-300),
    }


def main() -> None:
    native = native_observables(load_phi(NATIVE)[:5000])
    proposals = {label: np.load(BASE / label / "observable_samples.npz") for label in COLORS}
    names = list(LABELS)
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.2), constrained_layout=True)
    for ax, name in zip(axes.flat, names):
        arrays = [native[name]] + [proposals[label][name] for label in COLORS]
        low = min(np.quantile(x, 0.002) for x in arrays)
        high = max(np.quantile(x, 0.998) for x in arrays)
        pad = 0.04 * (high - low) if high > low else 1.0
        bins = np.linspace(low - pad, high + pad, 55)
        ax.hist(native[name], bins=bins, density=True, histtype="stepfilled", alpha=0.25,
                color="black", label="direct L32")
        for label, color in COLORS.items():
            values = proposals[label][name]
            ax.hist(values, bins=bins, density=True, histtype="step", linewidth=1.65,
                    color=color, label=label.replace("iteration", "iteration "))
        ax.axvline(native[name].mean(), color="black", linestyle="--", linewidth=1.1)
        ax.set_xlabel(LABELS[name])
        ax.set_ylabel("density")
        ax.tick_params(direction="in", top=True, right=True)
    axes.flat[-1].axis("off")
    axes.flat[0].legend(frameon=False, fontsize=9)
    fig.suptitle("L16→L32 proposals: direct L32 versus alternating-KL iterations", fontsize=14)
    out = BASE / "physical_distributions_iterations2to4.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()
