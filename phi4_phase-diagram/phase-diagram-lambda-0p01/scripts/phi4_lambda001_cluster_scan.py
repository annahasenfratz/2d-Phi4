#!/usr/bin/env python3
"""Standalone 2D phi4 cluster/reweighting scan.

Uses local Metropolis sweeps for amplitudes plus embedded Wolff sign-cluster
updates at fixed amplitudes.  At fixed lambda, kappa reweighting uses
exp[2 (kappa-kappa0) B], B=sum_x,mu phi_x phi_{x+mu}.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ChainStats:
    local_accepts: int = 0
    local_trials: int = 0
    cluster_calls: int = 0
    cluster_sites: int = 0

    def as_dict(self) -> dict[str, float | int]:
        return {
            "local_acceptance": self.local_accepts / self.local_trials if self.local_trials else float("nan"),
            "cluster_calls": self.cluster_calls,
            "mean_cluster_size": self.cluster_sites / self.cluster_calls if self.cluster_calls else float("nan"),
        }


def parse_ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def parse_floats(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def onsite(x: np.ndarray | float, lam: float) -> np.ndarray | float:
    x = np.asarray(x)
    return (1.0 - 2.0 * lam) * x * x + lam * x**4


def metropolis_sweep(phi: np.ndarray, kappa: float, lam: float, width: float, rng: np.random.Generator) -> tuple[int, int]:
    L = phi.shape[0]
    xx, yy = np.indices(phi.shape)
    accepts = 0
    trials = 0
    for parity in (0, 1):
        mask = (xx + yy) % 2 == parity
        old = phi[mask]
        prop = old + width * rng.normal(size=old.shape)
        neigh = (
            np.roll(phi, 1, axis=0)
            + np.roll(phi, -1, axis=0)
            + np.roll(phi, 1, axis=1)
            + np.roll(phi, -1, axis=1)
        )[mask]
        delta = onsite(prop, lam) - onsite(old, lam) - 2.0 * kappa * (prop - old) * neigh
        acc = np.log(rng.random(size=old.shape)) < -delta
        updated = old.copy()
        updated[acc] = prop[acc]
        phi[mask] = updated
        accepts += int(np.sum(acc))
        trials += int(mask.sum())
    return accepts, trials


def wolff_sign_cluster(phi: np.ndarray, kappa: float, rng: np.random.Generator) -> int:
    L = phi.shape[0]
    seed = (int(rng.integers(L)), int(rng.integers(L)))
    sign = phi[seed] >= 0.0
    in_cluster = np.zeros((L, L), dtype=bool)
    in_cluster[seed] = True
    stack = [seed]
    while stack:
        x, y = stack.pop()
        amp = abs(phi[x, y])
        for nx, ny in ((x + 1) % L, y), ((x - 1) % L, y), (x, (y + 1) % L), (x, (y - 1) % L):
            if in_cluster[nx, ny] or ((phi[nx, ny] >= 0.0) != sign):
                continue
            p_add = 1.0 - np.exp(-4.0 * kappa * amp * abs(phi[nx, ny]))
            if rng.random() < p_add:
                in_cluster[nx, ny] = True
                stack.append((nx, ny))
    phi[in_cluster] *= -1.0
    return int(np.sum(in_cluster))


def hybrid_samples(
    L: int,
    kappa: float,
    lam: float,
    samples: int,
    thermal_sweeps: int,
    skip_sweeps: int,
    clusters_per_sweep: int,
    proposal_width: float,
    seed: int,
) -> tuple[np.ndarray, ChainStats]:
    rng = np.random.default_rng(seed)
    phi = rng.normal(size=(L, L))
    stats = ChainStats()

    def sweep() -> None:
        a, t = metropolis_sweep(phi, kappa, lam, proposal_width, rng)
        stats.local_accepts += a
        stats.local_trials += t
        for _ in range(clusters_per_sweep):
            stats.cluster_sites += wolff_sign_cluster(phi, kappa, rng)
            stats.cluster_calls += 1

    for _ in range(thermal_sweeps):
        sweep()
    out = np.empty((samples, L, L), dtype=float)
    for i in range(samples):
        for _ in range(skip_sweeps):
            sweep()
        out[i] = phi
    return out, stats


def magnetization(configs: np.ndarray) -> np.ndarray:
    return np.mean(configs, axis=(1, 2))


def sector_counts(configs: np.ndarray) -> dict[str, int]:
    m = magnetization(configs)
    return {
        "positive": int(np.sum(m > 0.0)),
        "negative": int(np.sum(m < 0.0)),
        "zero": int(np.sum(m == 0.0)),
        "sign_flips": int(np.sum(np.signbit(m[1:]) != np.signbit(m[:-1]))) if len(m) > 1 else 0,
    }


def bond_sum(configs: np.ndarray) -> np.ndarray:
    return np.sum(configs * np.roll(configs, -1, axis=1), axis=(1, 2)) + np.sum(
        configs * np.roll(configs, -1, axis=2), axis=(1, 2)
    )


def reweight_curve(configs: np.ndarray, kappa0: float, grid: np.ndarray) -> list[dict[str, float]]:
    L = configs.shape[1]
    volume = L * L
    m = magnetization(configs)
    m2 = m * m
    m4 = m2 * m2
    abs_m = np.abs(m)
    bonds = bond_sum(configs)
    rows = []
    for kappa in grid:
        logw = 2.0 * (float(kappa) - kappa0) * bonds
        logw -= np.max(logw)
        w = np.exp(logw)
        wsum = np.sum(w)
        ess = wsum * wsum / np.sum(w * w)
        mean_m = float(np.sum(w * m) / wsum)
        mean_abs = float(np.sum(w * abs_m) / wsum)
        mean_m2 = float(np.sum(w * m2) / wsum)
        mean_m4 = float(np.sum(w * m4) / wsum)
        rows.append(
            {
                "kappa": float(kappa),
                "ess_over_n": float(ess / len(m)),
                "m_mean": mean_m,
                "abs_m_mean": mean_abs,
                "m2_mean": mean_m2,
                "m4_mean": mean_m4,
                "binder_u4": float(1.0 - mean_m4 / (3.0 * mean_m2 * mean_m2)),
                "susceptibility": float(volume * (mean_m2 - mean_m * mean_m)),
                "susceptibility_abs_centered": float(volume * (mean_m2 - mean_abs * mean_abs)),
            }
        )
    return rows


def peak(rows: list[dict[str, float]], obs: str, min_ess: float) -> dict[str, float | str]:
    pts = [r for r in rows if r["ess_over_n"] >= min_ess and np.isfinite(r[obs])]
    imax = int(np.argmax([r[obs] for r in pts]))
    if imax == 0 or imax == len(pts) - 1:
        r = pts[imax]
        return {"kappa_peak": r["kappa"], "peak_value": r[obs], "method": "max_point"}
    local = pts[imax - 1 : imax + 2]
    k = np.array([r["kappa"] for r in local])
    y = np.array([r[obs] for r in local])
    a, b, c = np.polyfit(k, y, 2)
    if a >= 0:
        r = pts[imax]
        return {"kappa_peak": r["kappa"], "peak_value": r[obs], "method": "max_point"}
    kp = -b / (2.0 * a)
    return {"kappa_peak": float(kp), "peak_value": float(a * kp * kp + b * kp + c), "method": "quadratic_3pt"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda", dest="lam", type=float, default=1.0)
    parser.add_argument("--Ls", default="16,24,32")
    parser.add_argument("--centers", default="0.330,0.335")
    parser.add_argument("--window", type=float, default=0.015)
    parser.add_argument("--step", type=float, default=0.0005)
    parser.add_argument("--samples", type=int, default=8192)
    parser.add_argument("--thermal-sweeps", type=int, default=3000)
    parser.add_argument("--skip-sweeps", type=int, default=4)
    parser.add_argument("--clusters-per-sweep", type=int, default=1)
    parser.add_argument("--proposal-width", type=float, default=0.8)
    parser.add_argument("--min-ess", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=97531)
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/phi4_lambda1_cluster_l16_l24_l32_refined_curves.csv"))
    parser.add_argument("--output-json", type=Path, default=Path("outputs/phi4_lambda1_cluster_l16_l24_l32_refined_summary.json"))
    args = parser.parse_args()

    all_rows = []
    summaries = []
    for L in parse_ints(args.Ls):
        for kappa0 in parse_floats(args.centers):
            configs, stats = hybrid_samples(
                L=L,
                kappa=kappa0,
                lam=args.lam,
                samples=args.samples,
                thermal_sweeps=args.thermal_sweeps,
                skip_sweeps=args.skip_sweeps,
                clusters_per_sweep=args.clusters_per_sweep,
                proposal_width=args.proposal_width,
                seed=args.seed + 1000 * L + int(round(100000 * kappa0)),
            )
            grid = np.arange(kappa0 - args.window, kappa0 + args.window + 0.5 * args.step, args.step)
            curve = reweight_curve(configs, kappa0, grid)
            for row in curve:
                all_rows.append({"L": L, "lambda": args.lam, "kappa0": kappa0, "n_cfg": args.samples, **row})
            pk = peak(curve, "susceptibility_abs_centered", args.min_ess)
            sec = sector_counts(configs)
            summaries.append({"L": L, "kappa0": kappa0, "n_cfg": args.samples, **sec, **stats.as_dict(), "chi_abs_peak_kappa": pk["kappa_peak"], "chi_abs_peak_value": pk["peak_value"], "chi_abs_peak_method": pk["method"]})
            print(
                f"L={L:2d} k0={kappa0:.4f} U4@k0={min(curve, key=lambda r: abs(r['kappa']-kappa0))['binder_u4']:.4f} "
                f"chi_abs_peak={float(pk['kappa_peak']):.6f} flips={sec['sign_flips']} "
                f"acc={stats.as_dict()['local_acceptance']:.3f} csize={stats.as_dict()['mean_cluster_size']:.1f}",
                flush=True,
            )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    args.output_json.write_text(
        json.dumps(
            {
                "args": vars(args) | {"output_csv": str(args.output_csv), "output_json": str(args.output_json)},
                "summaries": summaries,
                "notes": ["Standalone lambda=1 phi4 cluster/reweighting scan."],
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(json.dumps({"output_csv": str(args.output_csv), "output_json": str(args.output_json)}, indent=2))


if __name__ == "__main__":
    main()
