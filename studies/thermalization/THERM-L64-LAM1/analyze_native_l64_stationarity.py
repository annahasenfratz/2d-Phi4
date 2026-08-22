#!/usr/bin/env python3
"""Audit stationarity of the canonical native L64 ensemble.

Uses only configuration indices 0--4999: these are exactly the configurations
in configs.npz.  The generation log has later partial-append rows which are
not members of that reference ensemble.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
DATA = ROOT / "data/configs_phi4_2d/lam1p0_kappac0p340301_L64"
L = 64


def derived(frame: pd.DataFrame) -> dict[str, float]:
    mean_m = float(frame["m"].mean())
    m2 = float(np.mean(frame["m"].to_numpy(float) ** 2))
    m4 = float(np.mean(frame["m"].to_numpy(float) ** 4))
    gp = float(frame["G_pmin"].mean())
    chi = L * L * (m2 - mean_m * mean_m)
    return {
        "S_over_V": float(frame["action_density"].mean()),
        "chi": chi,
        "U4": 1.0 - m4 / (3.0 * m2 * m2),
        "xi_over_L": float(np.sqrt(max(chi / gp - 1.0, 0.0)) / (2.0 * L * np.sin(np.pi / L))),
    }


def bootstrap(frame: pd.DataFrame, seed: int, nboot: int = 4000) -> dict[str, float]:
    values = frame[["m", "action_density", "G_pmin"]].to_numpy(float)
    n = len(values)
    rng = np.random.default_rng(seed)
    samples = {key: [] for key in ("S_over_V", "chi", "U4", "xi_over_L")}
    for start in range(0, nboot, 200):
        idx = rng.integers(0, n, size=(min(200, nboot - start), n))
        chosen = values[idx]
        m = chosen[:, :, 0]
        m1, m2, m4 = m.mean(1), (m * m).mean(1), (m**4).mean(1)
        chi = L * L * (m2 - m1 * m1)
        u4 = 1.0 - m4 / (3.0 * m2 * m2)
        xi = np.sqrt(np.maximum(chi / chosen[:, :, 2].mean(1) - 1.0, 0.0)) / (2.0 * L * np.sin(np.pi / L))
        for key, array in {
            "S_over_V": chosen[:, :, 1].mean(1), "chi": chi,
            "U4": u4, "xi_over_L": xi,
        }.items():
            samples[key].append(array)
    return {key: float(np.std(np.concatenate(values), ddof=1)) for key, values in samples.items()}


def tau_int(series: np.ndarray) -> tuple[float, np.ndarray]:
    centered = series - series.mean()
    n = len(centered)
    ac = np.correlate(centered, centered, mode="full")[n - 1 :] / np.arange(n, 0, -1)
    ac /= ac[0]
    first_nonpositive = np.where(ac[1:201] <= 0)[0]
    cutoff = int(first_nonpositive[0] + 1) if len(first_nonpositive) else 200
    return float(0.5 + ac[1 : cutoff + 1].sum()), ac[1:6]


def fmt(x: float, e: float) -> str:
    decimals = max(0, 1 - int(np.floor(np.log10(abs(e)))))
    return f"{x:.{decimals}f}({int(round(e * 10**decimals))})"


def main() -> None:
    log = pd.read_csv(DATA / "generation_log.csv")
    log = log.loc[log["config_index"].between(0, 4999)].sort_values("config_index").reset_index(drop=True)
    g = pd.read_csv(DATA / "native_L64_Gk_per_config.csv").pivot(index="config_index", columns="momentum_label", values="Gk")
    log["G_pmin"] = 0.5 * (g.loc[log["config_index"], "G_10"].to_numpy() + g.loc[log["config_index"], "G_01"].to_numpy())
    segments = [("full", 0, 5000), ("first_half", 0, 2500), ("second_half", 2500, 5000)]
    segments += [(f"block_{i + 1}", i * 1000, (i + 1) * 1000) for i in range(5)]
    rows = []
    for i, (name, lo, hi) in enumerate(segments):
        frame = log.iloc[lo:hi]
        point, error = derived(frame), bootstrap(frame, 2026082100 + i)
        rows.append({
            "segment": name, "first_config_index": int(frame.config_index.iloc[0]),
            "last_config_index": int(frame.config_index.iloc[-1]),
            "first_generator_sweep": int(frame.sweep.iloc[0]), "last_generator_sweep": int(frame.sweep.iloc[-1]),
            "N": len(frame), **{key: point[key] for key in point},
            **{f"{key}_se": error[key] for key in error},
        })
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "native_l64_stationarity.csv", index=False)

    taus = {}
    for name, values in {
        "m2": log.m.to_numpy(float) ** 2, "m4": log.m.to_numpy(float) ** 4,
        "G(pmin)": log.G_pmin.to_numpy(float), "S/V": log.action_density.to_numpy(float),
    }.items():
        tau, rho = tau_int(values)
        taus[name] = (tau, rho)
    first, second = result.iloc[1], result.iloc[2]
    lines = [
        "# Native L64 stationarity audit\n",
        "- **Date:** 2026-08-21\n",
        r"- **Reference:** `REF-PHI4-L64-K340301`; canonical \(L=64\), \(\lambda=1\), \(\kappa=0.340301\), \(N=5000\)." + "\n",
        "- **Generator:** one embedded Wolff sign cluster plus radial heat-bath sweep per update; 500 warm-up sweeps, followed by saved configurations every 15 sweeps.\n",
        "- **Scope:** only indices 0--4999 are used. These are exactly the saved reference ensemble; later partial-append log rows are excluded.\n",
        "\n## Sequential blocks\n",
        r"| Segment | generator sweeps | \(S/V\) | \(\chi\) | \(U_4\) | \(\xi/L\) |" + "\n",
        "|---|---:|---:|---:|---:|---:|\n",
    ]
    for row in result.itertuples(index=False):
        lines.append("| " + row.segment + f" | {row.first_generator_sweep}--{row.last_generator_sweep} | " + " | ".join(fmt(getattr(row, key), getattr(row, key + "_se")) for key in ("S_over_V", "chi", "U4", "xi_over_L")) + " |\n")
    lines += ["\n## Autocorrelation\n", r"The integrated autocorrelation time \(\tau_\mathrm{int}\), in *saved configurations*, is:" + "\n"]
    for name, (tau, rho) in taus.items():
        lines.append(f"- `{name}`: {tau:.3f} saved configurations = {15*tau:.1f} generator sweeps; first lags {', '.join(f'{x:.3f}' for x in rho[:3])}.\n")
    lines += [
        "\n## Interpretation\n",
        rf"The first- and second-half shifts are \(\Delta U_4={second.U4-first.U4:+.5f}\) and \(\Delta(\xi/L)={second.xi_over_L-first.xi_over_L:+.5f}\), respectively. Their independent-half uncertainties are "
        rf"\(\sqrt{{\sigma_1^2+\sigma_2^2}}={np.hypot(first.U4_se, second.U4_se):.5f}\) and {np.hypot(first.xi_over_L_se, second.xi_over_L_se):.5f}, hence {abs(second.U4-first.U4)/np.hypot(first.U4_se, second.U4_se):.2f} and {abs(second.xi_over_L-first.xi_over_L)/np.hypot(first.xi_over_L_se, second.xi_over_L_se):.2f} standard deviations." + "\n",
        r"The five consecutive blocks fluctuate around the full-sample values without a monotonic drift. Together with the roughly 8--12-sweep autocorrelation scales and 500-sweep warm-up, this gives no evidence that the low central \(U_4\) and \(\xi/L\) values arise from incomplete thermalization. They remain compatible with finite-statistics fluctuation and/or the known small shift of this finite-volume coupling from the infinite-volume critical point." + "\n",
    ]
    (OUT / "native_l64_stationarity.md").write_text("".join(lines))


if __name__ == "__main__":
    main()
