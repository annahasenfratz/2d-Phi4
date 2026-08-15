#!/usr/bin/env python3
"""Plot HMC rethermalization versus 1/sweep across the L32--L256 cascade."""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "perfect_blocking_upsampling"
sys.path[:0] = [str(PKG / "src"), str(PKG / "scripts")]

from perfect_blocking_upsampling.io import ActionSpec
from run_lam1p0_l16to32_rqspline_zeroshot import per_config_observables


OUTPUT = PKG / "outputs/hmc_upscale_chain_lam1p0/analysis/cascade_sweeps_k0340301.pdf"
STAGES = {
    "L16→L32": {
        "L": 32, "color": "C0",
        "path": PKG / "outputs/hmc_upscale_chain_lam1p0/L8toL64/L8toL64_N1500_start0_HMCtherm100_100_100_r1/levels/L16toL32/observables/main_per_sweep_measurements.csv",
    },
    "L32→L64": {
        "L": 64, "color": "C1",
        "path": PKG / "outputs/hmc_upscale_chain_lam1p0/L8toL64/L8toL64_N1500_start0_HMCtherm100_100_100_r1/levels/L32toL64/observables/main_per_sweep_measurements.csv",
    },
    "L64→L128": {
        "L": 128, "color": "C2",
        "path": PKG / "outputs/hmc_upscale_chain_lam1p0/L8toL128/L8toL128_N1500_start0_HMCtherm100_100_100_100_d2_r3/levels/L64toL128/observables/main_per_sweep_measurements.csv",
    },
    "L128→L256": {
        "L": 256, "color": "C3",
        "path": PKG / "outputs/hmc_upscale_chain_lam1p0/L8toL256/L8toL256_N1500_start0_HMCtherm100_100_100_200_100_d2_r2/levels/L128toL256/observables/main_per_sweep_measurements.csv",
    },
}
# Matches the reference plot: 1/sweep = 0.5, 0.1, 0.04, 0.01.
# Use every saved measurement.  The stages have different output cadences
# (one or five sweeps), so this displays their full rethermalization bands.
SELECT_SWEEPS: tuple[int, ...] = ()
MAX_SWEEP = 100
SMOOTH_WINDOW = 5
LOCAL = ("action_density", "phi2", "local_kurtosis_ratio", "NN")
XI_2ND_OVER_L_ISING = 0.9050488292


def derived(frame: pd.DataFrame, *, n_boot: int = 250, seed: int = 20260813) -> dict[str, tuple[float, float]]:
    """Chain-bootstrap summary for ensemble observables."""
    x = frame[["m", "m2", "m4", "G_pmin_x_cfg", "G_pmin_y_cfg"]].to_numpy(float)
    L = int(frame["L"].iloc[0])
    def eval_means(means: np.ndarray) -> np.ndarray:
        m, m2, m4, gx, gy = means.T
        chi = L**2 * (m2 - m**2)
        gp = 0.5 * (gx + gy)
        ratio = chi / gp - 1.0
        # xi = sqrt(G(0)/G(p_min)-1)/(2 sin(pi/L)); divide by L once.
        xi_over_L = np.where(ratio > 0, np.sqrt(ratio) / (2 * L * np.sin(np.pi / L)), np.nan)
        binder = 1.0 - m4 / (3.0 * m2**2)
        return np.column_stack((chi / L**(2.0 - 0.25), gp / L**(2.0 - 0.25), xi_over_L, binder))
    point = eval_means(x.mean(0, keepdims=True))[0]
    rng = np.random.default_rng(seed)
    w = rng.multinomial(len(x), np.full(len(x), 1 / len(x)), size=n_boot) / len(x)
    boot = eval_means(w @ x)
    return {name: (float(point[i]), float(np.nanstd(boot[:, i], ddof=1)))
            for i, name in enumerate(("chi_scaled", "G_pmin_scaled", "xi_over_L", "Binder_U4"))}


def measure_native(L: int) -> dict[str, tuple[float, float]]:
    path = ROOT / f"data/configs_phi4_2d/lam1p0_kappac0p340301_L{L}/configs.npz"
    if not path.exists():
        return {}
    with np.load(path) as payload:
        phi = payload["phi"].astype(np.float32)
    action = ActionSpec("phi4_nn", 1.0, 0.340301)
    rows = []
    for lo in range(0, len(phi), 25):
        obs, g = per_config_observables(phi[lo:lo + 25], action)
        rows.append(pd.DataFrame({**obs, "G_pmin_x_cfg": g["G_10"], "G_pmin_y_cfg": g["G_01"], "L": L}))
    frame = pd.concat(rows, ignore_index=True)
    out = {key: (float(frame[key].mean()), float(frame[key].std(ddof=1) / np.sqrt(len(frame)))) for key in LOCAL}
    out.update(derived(frame, seed=1000 + L))
    return out


def summaries(path: Path) -> dict[str, pd.DataFrame]:
    df = pd.read_csv(path)
    df["local_kurtosis_ratio"] = df["phi4"] / np.maximum(df["phi2"] ** 2, 1.0e-300)
    available = set(df.sweep.unique())
    sweeps = ([s for s in SELECT_SWEEPS if s in available and s <= MAX_SWEEP]
              if SELECT_SWEEPS else sorted(s for s in available if 0 < s <= MAX_SWEEP))
    out = {key: [] for key in (*LOCAL, "chi_scaled", "G_pmin_scaled", "xi_over_L", "Binder_U4")}
    for sweep in sweeps:
        f = df[df.sweep.eq(sweep)]
        for key in LOCAL:
            out[key].append((sweep, f[key].mean(), f[key].std(ddof=1) / np.sqrt(len(f))))
        d = derived(f, seed=2000 + sweep)
        for key, (mean, se) in d.items():
            out[key].append((sweep, mean, se))
    return {key: pd.DataFrame(rows, columns=("sweep", "mean", "se")) for key, rows in out.items()}


def main() -> None:
    labels = {
        "action_density": "action density", "phi2": r"$\langle\phi^2\rangle$",
        "local_kurtosis_ratio": r"$\langle\phi^4/(\phi^2)^2\rangle$", "NN": r"nearest neighbour",
        "chi_scaled": r"$\chi/L^{2-\eta}$", "G_pmin_scaled": r"$G(p_\min)/L^{2-\eta}$",
        "xi_over_L": r"$\xi/L$", "Binder_U4": r"Binder cumulant $U_4$",
    }
    metrics = (*LOCAL, "chi_scaled", "G_pmin_scaled", "xi_over_L", "Binder_U4")
    native = {L: measure_native(L) for L in (32, 64, 128)}
    data = {name: summaries(cfg["path"]) for name, cfg in STAGES.items() if cfg["path"].exists()}
    fig, axes = plt.subplots(2, 4, figsize=(15.2, 7.2), constrained_layout=True)
    for ax, metric in zip(axes.flat, metrics):
        for name, cfg in STAGES.items():
            if name not in data or metric not in data[name]:
                continue
            d = data[name][metric]
            if d.empty:
                continue
            # Smooth in sweep order (not in the transformed x coordinate).
            smooth = d.copy()
            smooth["mean"] = smooth["mean"].rolling(SMOOTH_WINDOW, center=True, min_periods=1).mean()
            smooth["se"] = smooth["se"].rolling(SMOOTH_WINDOW, center=True, min_periods=1).mean()
            x = 1.0 / smooth["sweep"].to_numpy(float)
            # Center line plus a continuous one-SE uncertainty band.
            ax.plot(x, smooth["mean"], "-", lw=1.55, color=cfg["color"], label=name)
            ax.fill_between(x, smooth["mean"] - smooth["se"], smooth["mean"] + smooth["se"],
                            color=cfg["color"], alpha=.20, linewidth=0)
            ref = native.get(cfg["L"], {}).get(metric)
            if ref is not None:
                mean, se = ref
                ax.axhline(mean, color=cfg["color"], ls="--", lw=1.15, alpha=.78)
                native_band = ax.axhspan(mean - se, mean + se, facecolor="none",
                                        edgecolor=cfg["color"], hatch="///", linewidth=0,
                                        alpha=.28)
        if metric == "xi_over_L":
            ax.axhline(XI_2ND_OVER_L_ISING, color="black", ls=":", lw=1.35,
                       label=r"2D Ising $\xi_{\rm 2nd}/L=0.90505$")
            ax.legend(frameon=False, fontsize=8, loc="best")
        ax.set_xscale("log")
        ax.set_xlim(0.008, 0.65)
        ax.set_xticks([0.01, 0.04, 0.1, 0.5])
        ax.set_xticklabels(["0.01", "0.04", "0.1", "0.5"])
        ax.set_xlabel(r"$1/\mathrm{sweep}$")
        ax.set_title(labels[metric])
        ax.tick_params(direction="in", top=True, right=True)
        ax.grid(alpha=.22)
    axes.flat[0].legend(frameon=False, fontsize=9, loc="best")
    fig.suptitle(r"Flow-upscaling cascade versus rethermalization budget, $\kappa=0.340301$"
                 "\nDashed line and band: direct native ensemble at the same fine lattice")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, bbox_inches="tight")
    print(OUTPUT)


if __name__ == "__main__":
    main()
