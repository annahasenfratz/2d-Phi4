#!/usr/bin/env python3
"""Build the requested L32 -> L64 HMC/Wolff thermalization tables.

All measurements are recomputed from the retained per-configuration histories.
The susceptibility is connected, chi = L^2(<m^2>-<m>^2).  Uncertainties are
configuration-bootstrap standard deviations (4,000 replicates); the two HMC
off-critical replicas are concatenated before resampling.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
UPSAMPLING = ROOT / "perfect_blocking_upsampling"
sys.path[:0] = [str(UPSAMPLING / "src"), str(UPSAMPLING / "scripts")]

from perfect_blocking_upsampling.io import ActionSpec
from run_lam1p0_l16to32_rqspline_zeroshot import per_config_observables


L = 64
OBSERVABLES = ("action_density", "phi2", "phi4", "NN", "2nn", "diag", "chi", "U4", "xi_over_L")
LABELS = {
    "action_density": r"$S/V$", "phi2": r"$\langle\phi^2\rangle$",
    "phi4": r"$\langle\phi^4\rangle$", "NN": r"$\mathrm{NN}$",
    "2nn": r"$2\mathrm{nn}$", "diag": r"$\mathrm{diag}$",
    "chi": r"$\chi$", "U4": r"$U_4$", "xi_over_L": r"$\xi/L$",
}
RUNS = {
    "Wolff, $\\kappa_c=0.340301$": [
        ROOT / "perfect_blocking_upsampling/outputs/wolff_rethermalization_lam1p0/L64_ethan7_nativeL32_sweep0_N5000_kc0p340301_kf0p340301_radial_wolff_fixed4_r1",
    ],
    "Wolff, $\\kappa_c=0.340100$": [
        ROOT / "perfect_blocking_upsampling/outputs/wolff_rethermalization_lam1p0/L64_ethan7_L32k0p340100_replicas_N3000_sweep0_kc0p340100_kf0p340301_radial_wolff_fixed4_r1",
    ],
    "HMC, $\\kappa_c=0.340301$": [
        ROOT / "perfect_blocking_upsampling/outputs/fine_hmc_lam1p0/global/L32toL64_ethan7_nativeL32_sweep0_kf0p340301_N5000_S50_tau2_n36_eps2over36_r1",
    ],
    "HMC, $\\kappa_c=0.340100$": [
        ROOT / "perfect_blocking_upsampling/outputs/fine_hmc_lam1p0/global/L32k0p340100_toL64_ethan7_sweep0_kf0p340301_N1500_S100_tau2_n36_eps2over36_r1",
        ROOT / "perfect_blocking_upsampling/outputs/fine_hmc_lam1p0/global/L32k0p340100_r2_toL64_ethan7_sweep0_kf0p340301_N1500_S100_tau2_n36_eps2over36_r1",
    ],
}
SWEEPS = {
    "Wolff, $\\kappa_c=0.340301$": (0, 10, 20, 50, 100),
    "Wolff, $\\kappa_c=0.340100$": (0, 10, 20, 50, 100),
    "HMC, $\\kappa_c=0.340301$": (0, 50, 100, 250, 500),
    "HMC, $\\kappa_c=0.340100$": (0, 50, 100, 250, 500),
}


def add_derived(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["G_pmin"] = 0.5 * (out["G_pmin_x_cfg"] + out["G_pmin_y_cfg"])
    return out


def calculate(frame: pd.DataFrame, seed: int, nboot: int = 4000) -> tuple[dict[str, float], dict[str, float]]:
    a = frame["action_density"].to_numpy(float)
    values = {key: frame[key].to_numpy(float) for key in ("phi2", "phi4", "NN", "2nn", "diag")}
    m = frame["m"].to_numpy(float)
    m2 = frame["m2"].to_numpy(float)
    m4 = frame["m4"].to_numpy(float)
    gp = frame["G_pmin"].to_numpy(float)

    def derive(indices: np.ndarray | None = None) -> dict[str, np.ndarray | float]:
        if indices is None:
            means = {key: float(np.mean(v)) for key, v in values.items()}
            mean_a, mean_m, mean_m2, mean_m4, mean_gp = map(float, (a.mean(), m.mean(), m2.mean(), m4.mean(), gp.mean()))
        else:
            means = {key: v[indices].mean(axis=1) for key, v in values.items()}
            mean_a, mean_m, mean_m2, mean_m4, mean_gp = (v[indices].mean(axis=1) for v in (a, m, m2, m4, gp))
        chi = L * L * (mean_m2 - mean_m * mean_m)
        u4 = 1.0 - mean_m4 / (3.0 * mean_m2 * mean_m2)
        xi = np.sqrt(np.maximum(chi / mean_gp - 1.0, 0.0)) / (2.0 * L * np.sin(np.pi / L))
        return {"action_density": mean_a, **means, "chi": chi, "U4": u4, "xi_over_L": xi}

    point = derive()
    rng = np.random.default_rng(seed)
    n = len(frame)
    pieces: dict[str, list[np.ndarray]] = {key: [] for key in OBSERVABLES}
    # Batching avoids a large 4000 x N index array at N=5000.
    for _ in range(0, nboot, 200):
        idx = rng.integers(0, n, size=(min(200, nboot - _), n))
        boot = derive(idx)
        for key in OBSERVABLES:
            pieces[key].append(np.asarray(boot[key], dtype=float))
    errors = {key: float(np.std(np.concatenate(pieces[key]), ddof=1)) for key in OBSERVABLES}
    return point, errors


def fmt(value: float, error: float) -> str:
    # PDG-style parenthetical errors, retaining enough digits for these data.
    if error == 0 or not np.isfinite(error):
        return f"{value:.6g}"
    exponent = int(np.floor(np.log10(abs(error))))
    decimals = max(0, 1 - exponent)
    scale = 10 ** decimals
    return f"{value:.{decimals}f}({int(round(error * scale))})"


def native_frame() -> pd.DataFrame:
    path = ROOT / "data/configs_phi4_2d/lam1p0_kappac0p340301_L64/configs.npz"
    phi = np.load(path)["phi"].astype(np.float32)
    observables, g = per_config_observables(phi, ActionSpec("phi4_nn", 1.0, 0.340301))
    result = pd.DataFrame(observables)
    result["G_pmin_x_cfg"] = g["G_10"]
    result["G_pmin_y_cfg"] = g["G_01"]
    return add_derived(result)


def run_frame(paths: list[Path], sweep: int) -> pd.DataFrame:
    parts = []
    for path in paths:
        table = pd.read_csv(path / "observables/main_per_sweep_measurements.csv")
        part = table.loc[table["sweep"] == sweep]
        if part.empty:
            raise ValueError(f"{path}: missing sweep {sweep}")
        parts.append(part)
    return add_derived(pd.concat(parts, ignore_index=True))


def main() -> None:
    outdir = Path(__file__).resolve().parent
    native, native_se = calculate(native_frame(), seed=2026082101)
    records: list[dict[str, object]] = []
    for key in OBSERVABLES:
        records.append({"method": "Native L64", "kappa_c": "0.340301", "kappa_f": "0.340301", "sweep": "native", "N": 5000, "observable": key, "mean": native[key], "se": native_se[key], "uncertainty_method": "configuration bootstrap, 4000 replicates"})
    for run_name, paths in RUNS.items():
        kc = "0.340100" if "340100" in run_name else "0.340301"
        for sweep in SWEEPS[run_name]:
            point, error = calculate(run_frame(paths, sweep), seed=2026082200 + sweep + (100 if "HMC" in run_name else 0) + (10 if kc.endswith("100") else 0))
            n = sum(len(pd.read_csv(path / "observables/main_per_sweep_measurements.csv", nrows=1)) * 0 + (1500 if "N1500" in path.name else 5000 if "N5000" in path.name else 3000) for path in paths)
            for obs in OBSERVABLES:
                records.append({"method": run_name.split(",")[0], "kappa_c": kc, "kappa_f": "0.340301", "sweep": sweep, "N": n, "observable": obs, "mean": point[obs], "se": error[obs], "uncertainty_method": "configuration bootstrap, 4000 replicates; HMC off-source replicas concatenated" if len(paths) == 2 else "configuration bootstrap, 4000 replicates"})
    pd.DataFrame(records).to_csv(outdir / "requested_l32_to_l64_table_data.csv", index=False)

    rows = pd.DataFrame(records)
    text = [
        "# Requested L32 → L64 thermalization comparison\n",
        r"All values below were recomputed from the raw per-configuration histories.  Here \(\chi=L^2(\langle m^2\rangle-\langle m\rangle^2)\) is connected. Errors are independent configuration-bootstrap standard deviations (4,000 replicates)." + "\n",
        "Column I uses the $\\kappa_c=0.340301$ source and column II the $\\kappa_c=0.340100$ source; both have $\\kappa_f=0.340301$.  A native value is printed only in the common sweep-zero row for each observable.\n",
    ]
    tex = ["% Auto-generated by build_requested_comparison.py; do not hand-edit.\n"]
    wolff_sweeps, hmc_sweeps = (0, 10, 20, 50, 100), (0, 50, 100, 250, 500)
    all_sweeps = (0, 10, 20, 50, 100, 250, 500)

    def entry(method: str, kc: str, sweep: int, obs: str) -> str:
        row = rows[(rows.method == method) & (rows.kappa_c == kc) & (rows.sweep.astype(str) == str(sweep)) & (rows.observable == obs)]
        return "" if row.empty else fmt(row.iloc[0]["mean"], row.iloc[0]["se"])

    def emit_table(title: str, observables: tuple[str, ...], label: str) -> None:
        text.append(f"## {title}\n")
        text.append("| Observable | Native L64 | Wolff sweep | Upscaled Wolff I | Upscaled Wolff II | HMC sweep | Upscaled HMC I | Upscaled HMC II |\n")
        text.append("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        tex.extend([
            "\\begin{table*}[t]", "\\centering",
            "\\caption{L32 $\\to$ L64 rethermalization at $\\kappa_f=0.340301$. Columns I and II use source $\\kappa_c=0.340301$ and $0.340100$, respectively. A native value is shown only at sweep zero. Errors are configuration-bootstrap standard deviations.}",
            "\\scriptsize", "\\begin{ruledtabular}", "\\begin{tabular}{lccccccc}",
            "Observable & Native L64 & Wolff sweep & Upscaled Wolff I & Upscaled Wolff II & HMC sweep & Upscaled HMC I & Upscaled HMC II " + r"\\", "\\hline",
        ])
        for obs in observables:
            native_row = rows[(rows.method == "Native L64") & (rows.observable == obs)].iloc[0]
            for i, sweep in enumerate(all_sweeps):
                native_value = fmt(native_row["mean"], native_row["se"]) if i == 0 else ""
                w1 = entry("Wolff", "0.340301", sweep, obs) if sweep in wolff_sweeps else ""
                w2 = entry("Wolff", "0.340100", sweep, obs) if sweep in wolff_sweeps else ""
                h1 = entry("HMC", "0.340301", sweep, obs) if sweep in hmc_sweeps else ""
                h2 = entry("HMC", "0.340100", sweep, obs) if sweep in hmc_sweeps else ""
                label_obs = LABELS[obs] if i == 0 else ""
                ws = str(sweep) if sweep in wolff_sweeps else ""
                hs = str(sweep) if sweep in hmc_sweeps else ""
                values = (label_obs, native_value, ws, w1, w2, hs, h1, h2)
                text.append("| " + " | ".join(values) + " |\n")
                tex.append(" & ".join(values) + r" \\")
            tex.append("\\hline")
            text.append("\n")
        tex.extend(["\\end{tabular}", "\\end{ruledtabular}", f"\\label{{{label}}}", "\\end{table*}", ""])
        text.append("\n")

    emit_table("Local / short-distance observables", ("action_density", "phi2", "phi4", "NN", "2nn", "diag"), "tab:therm_l64_common_local")
    emit_table("Long-distance observables", ("chi", "U4", "xi_over_L"), "tab:therm_l64_common_long")
    (outdir / "requested_l32_to_l64_table.md").write_text("\n".join(text))
    (outdir / "requested_l32_to_l64_table.tex").write_text("\n".join(tex))


if __name__ == "__main__":
    main()
