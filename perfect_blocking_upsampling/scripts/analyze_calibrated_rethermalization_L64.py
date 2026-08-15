#!/usr/bin/env python3
"""Create fixed-bin L64 relaxation plots and stationarity tables."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "perfect_blocking_upsampling/runs/lam1p0/calibrated_empirical_fine_rethermalization_20260721"
PLOTS = OUT / "plots_L64_extension"
PLOTS.mkdir(exist_ok=True)
SAVES = (0, 1, 2, 5, 10, 20, 30, 50, 75, 100, 150, 200)
HIST_SWEEPS = (0, 10, 20, 50, 100, 200)
PLOT_OBS = ("action_density", "phi2", "phi4", "NN", "m2", "G_pmin_avg")


def observables(phi: np.ndarray) -> dict[str, np.ndarray]:
    x = np.asarray(phi, dtype=np.float64)
    phi2 = np.mean(x * x, axis=(1, 2))
    phi4 = np.mean(x**4, axis=(1, 2))
    nn = .5 * np.mean(x * np.roll(x, -1, 1) + x * np.roll(x, -1, 2), axis=(1, 2))
    fft = np.fft.fft2(x, axes=(1, 2))
    return {
        "action_density": 1. - 2. * phi2 + phi4 - 4. * .340301 * nn,
        "phi2": phi2, "phi4": phi4, "NN": nn,
        "m2": np.mean(x, axis=(1, 2)) ** 2,
        "G_pmin_avg": .5 * (np.abs(fft[:, 1, 0])**2 + np.abs(fft[:, 0, 1])**2) / (x.shape[1]**2),
    }


def exponential(s: np.ndarray, inf: float, amplitude: float, tau: float) -> np.ndarray:
    return inf + amplitude * np.exp(-s / tau)


def main() -> None:
    snapshots = {s: np.load(OUT / f"states_L64_sweep{s:03d}.npz") for s in HIST_SWEEPS}
    reference = observables(snapshots[0]["reference_native"])
    cal = {s: observables(snapshots[s]["calibrated"]) for s in HIST_SWEEPS}
    metrics = list(csv.DictReader((OUT / "metrics_L64_extended.csv").open()))

    for name in PLOT_OBS:
        values = np.concatenate([reference[name]] + [cal[s][name] for s in HIST_SWEEPS])
        lo, hi = np.quantile(values, [.001, .999])
        padding = .05 * max(hi - lo, 1.e-12)
        bins = np.linspace(lo - padding, hi + padding, 61)
        fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.5), constrained_layout=True)
        for axis, sweep in zip(axes.flat, HIST_SWEEPS):
            axis.hist(reference[name], bins=bins, density=True, histtype="step", lw=1.5, label="native")
            axis.hist(cal[sweep][name], bins=bins, density=True, histtype="step", lw=1.5, label=f"calibrated s={sweep}")
            axis.set_title(f"sweep {sweep}")
            axis.set_xlabel(name)
            axis.set_ylabel("density")
            if sweep == 0:
                axis.legend(frameon=False, fontsize=8)
        fig.suptitle(f"L64 native vs calibrated empirical initializer: {name}")
        fig.savefig(PLOTS / f"hist_{name}_L64.pdf")
        fig.savefig(PLOTS / f"hist_{name}_L64.png", dpi=180)
        plt.close(fig)

    rows = []
    fit_rows = []
    x = np.asarray(SAVES, dtype=float)
    for name in ("phi2", "phi4", "action_density", "NN"):
        cal_rows = [r for r in metrics if r["ensemble"] == "calibrated" and r["observable"] == name]
        cal_rows.sort(key=lambda r: int(r["sweep"]))
        width = np.asarray([float(r["width_ratio"]) for r in cal_rows])
        for left, right in ((20, 50), (50, 100), (100, 150), (150, 200)):
            lval = width[list(SAVES).index(left)]
            rval = width[list(SAVES).index(right)]
            rows.append({"observable": name, "window_start": left, "window_end": right, "width_start": lval, "width_end": rval, "width_change": rval - lval})
        try:
            fit, covariance = curve_fit(exponential, x, width, p0=(1., width[0] - 1., 20.), bounds=([.5, -2., .1], [1.5, 2., 1000.]), maxfev=50000)
            error = np.sqrt(np.diag(covariance))
            fit_rows.append({"observable": name, "fit_inf": fit[0], "fit_amplitude": fit[1], "tau_sweeps": fit[2], "tau_se": error[2], "fit_status": "single-exponential diagnostic only"})
        except Exception as exc:
            fit_rows.append({"observable": name, "fit_status": f"not fit: {exc}"})

    def write(path: Path, records: list[dict]) -> None:
        fields = list(dict.fromkeys(k for r in records for k in r))
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(records)
    write(OUT / "width_stationarity_windows_L64.csv", rows)
    write(OUT / "relaxation_fits_L64.csv", fit_rows)

    fig, axes = plt.subplots(2, 2, figsize=(10, 6.8), constrained_layout=True)
    for axis, name in zip(axes.flat, ("phi2", "phi4", "action_density", "NN")):
        cal_rows = sorted((r for r in metrics if r["ensemble"] == "calibrated" and r["observable"] == name), key=lambda r: int(r["sweep"]))
        sw = np.asarray([int(r["sweep"]) for r in cal_rows])
        width = np.asarray([float(r["width_ratio"]) for r in cal_rows])
        axis.plot(sw, width, "o-", label="calibrated")
        axis.axhline(1., color="black", lw=.8, ls="--", label="native width")
        axis.set_title(name); axis.set_xlabel("full-field sweeps"); axis.set_ylabel("width ratio")
        axis.legend(frameon=False, fontsize=8)
    fig.savefig(PLOTS / "width_evolution_L64.pdf")
    fig.savefig(PLOTS / "width_evolution_L64.png", dpi=180)
    plt.close(fig)

    native_rows = [r for r in metrics if r["ensemble"] == "native_control" and r["observable"] in ("phi2", "phi4", "action_density", "NN")]
    write(OUT / "native_control_stationarity_L64.csv", native_rows)
    summary = [
        "# L64 Extended Fine Rethermalization",
        "",
        "The extension deterministically replayed the original sweep-20 states before continuing with the same checkerboard full-field Metropolis update (step size 0.5, seed 1062).",
        "",
        "The single-exponential fits are descriptive only: finite-sample width estimates are visibly nonmonotonic after approximately sweep 30.",
        "",
        "Key fixed-bin histogram panels are in `plots_L64_extension/`.",
    ]
    (OUT / "L64_extension_summary.md").write_text("\n".join(summary) + "\n")


if __name__ == "__main__":
    main()
