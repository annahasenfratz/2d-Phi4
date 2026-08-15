#!/usr/bin/env python3
"""Plot the transverse-zero-momentum correlator on direct critical ensembles."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "perfect_blocking_upsampling/outputs/hmc_upscale_chain_lam1p0/analysis"
VOLUMES = (32, 64, 128)
BIN_SIZE = 20


def periodic_one_state(x: np.ndarray, amplitude: float, mass: float, L: int) -> np.ndarray:
    return amplitude * (np.exp(-mass * x) + np.exp(-mass * (L - x)))


def correlator_samples(phi: np.ndarray) -> np.ndarray:
    """C(x) = <L^-1 sum_x0 P(x0+x)P(x0)>, P(x)=sum_y phi(x,y)."""
    plane = phi.sum(axis=2, dtype=np.float64)
    # Remove only the ensemble one-point value; do not remove each configuration's
    # p=(0,0) fluctuation, which is part of the p_y=0 correlator.
    plane -= plane.mean()
    modes = np.fft.fft(plane, axis=1)
    return np.fft.ifft(modes * modes.conj(), axis=1).real / plane.shape[1]


def jackknife_error(bin_values: np.ndarray) -> np.ndarray:
    nbin = len(bin_values)
    jk = (bin_values.sum(axis=0, keepdims=True) - bin_values) / (nbin - 1)
    return np.sqrt((nbin - 1) / nbin * ((jk - jk.mean(axis=0)) ** 2).sum(axis=0))


def load(L: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float]]:
    phi = np.load(ROOT / f"data/configs_phi4_2d/lam1p0_kappac0p340301_L{L}/configs.npz")["phi"]
    samples = correlator_samples(phi)
    nbin = len(samples) // BIN_SIZE
    bins = samples[: nbin * BIN_SIZE].reshape(nbin, BIN_SIZE, L).mean(axis=1)
    mean, error = bins.mean(axis=0), jackknife_error(bins)
    xfit = np.arange(L // 4, L // 2 + 1)
    parameters, _ = curve_fit(
        lambda x, A, m: periodic_one_state(x, A, m, L),
        xfit, mean[xfit], sigma=error[xfit], absolute_sigma=True,
        p0=(mean[L // 4], 1.0 / L), bounds=([0.0, 0.0], [np.inf, 1.0]),
    )
    return np.arange(L // 2 + 1), mean[: L // 2 + 1], error[: L // 2 + 1], tuple(parameters)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig, (ax_raw, ax_scaled) = plt.subplots(1, 2, figsize=(11.2, 4.25), constrained_layout=True)
    colors = {32: "#d62728", 64: "#1f77b4", 128: "#009e73"}
    summary = []
    for L in VOLUMES:
        x, mean, error, (amplitude, mass) = load(L)
        color = colors[L]
        points = np.arange(0, len(x), max(1, L // 32))
        ax_raw.errorbar(x[points], mean[points], yerr=error[points], fmt="o", ms=3,
                        capsize=1.5, color=color, label=fr"$L={L}$")
        xline = np.linspace(L / 4, L / 2, 250)
        ax_raw.plot(xline, periodic_one_state(xline, amplitude, mass, L), color=color, lw=1.35, ls="--")
        normalized, normalized_error = mean / mean[0], error / mean[0]
        ax_scaled.errorbar(x[points] / L, normalized[points], yerr=normalized_error[points],
                           fmt="o", ms=3, capsize=1.5, color=color, label=fr"$L={L}$")
        ax_scaled.plot(xline / L, periodic_one_state(xline, amplitude, mass, L) / mean[0],
                       color=color, lw=1.35, ls="--")
        summary.append((L, mass, mass * L))

    ax_raw.set_yscale("log")
    ax_raw.set_xlabel(r"separation $x$")
    ax_raw.set_ylabel(r"$G_{p_y=0}(x)$")
    ax_raw.set_title(r"Direct native: transverse-zero-momentum correlator")
    ax_scaled.set_yscale("log")
    ax_scaled.set_xlabel(r"$x/L$")
    ax_scaled.set_ylabel(r"$G_{p_y=0}(x)/G_{p_y=0}(0)$")
    ax_scaled.set_title(r"Finite-size collapse")
    for ax in (ax_raw, ax_scaled):
        ax.tick_params(direction="in", top=True, right=True)
        ax.grid(alpha=0.2, lw=0.6)
        ax.legend(frameon=False)
    fig.savefig(OUT / "py0_correlator_direct_L32_L64_L128.pdf", bbox_inches="tight")
    print("fit range: L/4 <= x <= L/2")
    for L, mass, mass_L in summary:
        print(f"L={L}: m={mass:.7f}, mL={mass_L:.5f}")


if __name__ == "__main__":
    main()
