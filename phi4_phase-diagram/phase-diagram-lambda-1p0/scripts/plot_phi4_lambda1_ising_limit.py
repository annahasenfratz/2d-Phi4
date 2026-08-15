#!/usr/bin/env python3
"""Summarize phi4 lambda=1 Ising-limit sanity scan."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {16: "#1f77b4", 24: "#d62728", 32: "#2ca02c"}
STYLES = {0.33: "-", 0.335: "--", 0.34: ":"}


def read_rows(path: Path) -> list[dict[str, float]]:
    rows = []
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            rows.append({k: float(v) for k, v in row.items()})
    return rows


def group(rows: list[dict[str, float]]) -> dict[tuple[int, float], list[dict[str, float]]]:
    out = defaultdict(list)
    for row in rows:
        out[(int(row["L"]), row["kappa0"])].append(row)
    for key in out:
        out[key].sort(key=lambda r: r["kappa"])
    return dict(out)


def selected(curve: list[dict[str, float]], obs: str, min_ess: float) -> list[dict[str, float]]:
    return [r for r in curve if r["ess_over_n"] >= min_ess and np.isfinite(r[obs])]


def peak(curve: list[dict[str, float]], obs: str, min_ess: float) -> dict[str, float | str]:
    pts = selected(curve, obs, min_ess)
    imax = int(np.argmax([r[obs] for r in pts]))
    if imax == 0 or imax == len(pts) - 1:
        r = pts[imax]
        return {"kappa_peak": float(r["kappa"]), "peak_value": float(r[obs]), "method": "max_point"}
    local = pts[imax - 1 : imax + 2]
    k = np.array([r["kappa"] for r in local])
    y = np.array([r[obs] for r in local])
    a, b, c = np.polyfit(k, y, 2)
    if a >= 0:
        r = pts[imax]
        return {"kappa_peak": float(r["kappa"]), "peak_value": float(r[obs]), "method": "max_point"}
    kp = -b / (2 * a)
    return {"kappa_peak": float(kp), "peak_value": float(a * kp * kp + b * kp + c), "method": "quadratic_3pt"}


def crossing(curve_a: list[dict[str, float]], curve_b: list[dict[str, float]], min_ess: float) -> dict[str, float | str]:
    a = {round(r["kappa"], 10): r["binder_u4"] for r in selected(curve_a, "binder_u4", min_ess)}
    b = {round(r["kappa"], 10): r["binder_u4"] for r in selected(curve_b, "binder_u4", min_ess)}
    grid = sorted(set(a).intersection(b))
    diffs = [(k, a[k] - b[k]) for k in grid]
    for (k0, d0), (k1, d1) in zip(diffs, diffs[1:]):
        if d0 * d1 < 0:
            kc = k0 - d0 * (k1 - k0) / (d1 - d0)
            return {"kappa_crossing": float(kc), "method": "linear", "bracket": [k0, k1]}
    k_best, d_best = min(diffs, key=lambda kd: abs(kd[1]))
    return {"kappa_crossing": float(k_best), "method": "closest_grid_no_sign_change", "delta": float(d_best)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--broad", type=Path, default=Path("outputs/phi4_lambda1_cluster_l16_l24_l32_curves.csv"))
    parser.add_argument("--refined", type=Path, default=Path("outputs/phi4_lambda1_cluster_l16_l24_l32_refined_curves.csv"))
    parser.add_argument("--output-png", type=Path, default=Path("outputs/phi4_lambda1_l16_l24_l32_chi_binder.png"))
    parser.add_argument("--output-pdf", type=Path, default=Path("outputs/phi4_lambda1_l16_l24_l32_chi_binder.pdf"))
    parser.add_argument("--output-json", type=Path, default=Path("outputs/phi4_lambda1_l16_l24_l32_chi_binder.json"))
    parser.add_argument("--min-ess", type=float, default=0.30)
    args = parser.parse_args()

    refined = group(read_rows(args.refined))
    broad = group(read_rows(args.broad))

    peak_rows = []
    for L in [16, 24, 32]:
        entries = []
        for k0 in [0.33, 0.335]:
            p = peak(refined[(L, k0)], "susceptibility_abs_centered", args.min_ess)
            entries.append({"L": L, "kappa0": k0, **p})
        peak_rows.extend(entries)

    averaged = {}
    for L in [16, 24, 32]:
        entries = [r for r in peak_rows if r["L"] == L]
        averaged[str(L)] = {
            "kappa_peak_mean": float(np.mean([r["kappa_peak"] for r in entries])),
            "kappa_peak_half_spread": float(0.5 * (max(r["kappa_peak"] for r in entries) - min(r["kappa_peak"] for r in entries))),
            "chi_peak_mean": float(np.mean([r["peak_value"] for r in entries])),
            "chi_peak_half_spread": float(0.5 * (max(r["peak_value"] for r in entries) - min(r["peak_value"] for r in entries))),
        }

    L_arr = np.array([16.0, 24.0, 32.0])
    chi_arr = np.array([averaged[str(int(L))]["chi_peak_mean"] for L in L_arr])
    expo, log_amp = np.polyfit(np.log(L_arr), np.log(chi_arr), 1)

    crossings = {}
    for k0, source in [(0.335, refined), (0.34, broad)]:
        crossings[str(k0)] = {
            "16-24": crossing(source[(16, k0)], source[(24, k0)], args.min_ess),
            "24-32": crossing(source[(24, k0)], source[(32, k0)], args.min_ess),
            "16-32": crossing(source[(16, k0)], source[(32, k0)], args.min_ess),
        }

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.9), constrained_layout=True)
    ax = axes[0]
    for (L, k0), curve in sorted(refined.items()):
        pts = selected(curve, "susceptibility_abs_centered", args.min_ess)
        ax.plot(
            [r["kappa"] for r in pts],
            [r["susceptibility_abs_centered"] for r in pts],
            color=COLORS[L],
            linestyle=STYLES[round(k0, 3)],
            linewidth=1.9,
            label=f"L={L}, k0={k0:.3f}",
        )
    text = "\n".join(
        f"L={L}: {averaged[str(L)]['kappa_peak_mean']:.5f}" for L in [16, 24, 32]
    )
    ax.text(0.02, 0.98, "mean chi peaks\n" + text, transform=ax.transAxes, va="top", fontsize=9, bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "#cccccc"})
    ax.set_title(r"$\lambda=1$ phi4 $\chi_{|m|}$")
    ax.set_xlabel(r"$\kappa$")
    ax.set_ylabel(r"$V(\langle m^2\rangle-\langle |m|\rangle^2)$")
    ax.grid(True, color="#dddddd")
    ax.legend(fontsize=8, ncols=2)

    ax = axes[1]
    for L in [16, 24, 32]:
        pts = selected(broad[(L, 0.34)], "binder_u4", args.min_ess)
        ax.plot([r["kappa"] for r in pts], [r["binder_u4"] for r in pts], "o-", color=COLORS[L], linewidth=2.0, markersize=2.6, label=f"L={L}, k0=0.340")
    for c in crossings["0.34"].values():
        if c["method"] == "linear":
            ax.axvline(c["kappa_crossing"], color="#777777", linestyle=":", linewidth=1.0)
    cross_text = "\n".join(
        f"{pair}: {c['kappa_crossing']:.6f}" for pair, c in crossings["0.34"].items()
    )
    ax.text(0.02, 0.98, "Binder crossings\n" + cross_text, transform=ax.transAxes, va="top", fontsize=9, bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "#cccccc"})
    ax.set_title(r"$\lambda=1$ phi4 Binder, $k_0=0.340$")
    ax.set_xlabel(r"$\kappa$")
    ax.set_ylabel(r"$U_4$")
    ax.grid(True, color="#dddddd")
    ax.legend(fontsize=9)

    fig.suptitle(r"2D phi4 Ising-limit sanity check, $\lambda=1$", fontsize=13)
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, dpi=180)
    fig.savefig(args.output_pdf)

    out = {
        "inputs": {"broad": str(args.broad), "refined": str(args.refined)},
        "min_ess": args.min_ess,
        "peak_rows": peak_rows,
        "averaged_peaks": averaged,
        "height_power_fit": {"exponent": float(expo), "amplitude": float(np.exp(log_amp))},
        "binder_crossings": crossings,
    }
    args.output_json.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(json.dumps({"output_png": str(args.output_png), "output_pdf": str(args.output_pdf), "output_json": str(args.output_json)}, indent=2))


if __name__ == "__main__":
    main()
