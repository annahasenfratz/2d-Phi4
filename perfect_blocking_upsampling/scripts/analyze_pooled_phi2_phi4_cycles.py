#!/usr/bin/env python3
"""Audit pooled phi2/phi4 relaxation and tails for the two L16->L32 runs."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / "perfect_blocking_upsampling/outputs/controlled_patch_lam1p0/coarse_detail_L16to32"
RUNS = [
    RUN_ROOT / "prod_cd_bL32_RQS_cfg2000_N500_Pc16x1_Pd16x16_kf0p340301_kc0p340301",
    RUN_ROOT / "prod_cd_bL32_RQS_cfg2500_N500_Pc16x1_Pd16x16_kf0p340301_kc0p340301",
]
NATIVE = ROOT / "data/configs_phi4_2d/lam1p0_kappac0p340301_L32/native_L32_all_observables_per_config.csv"
OUT = RUN_ROOT / "pooled_cfg2000_cfg2500_phi2_phi4_cycle_tail_covariance.md"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def q(values: np.ndarray, p: float) -> float:
    return float(np.quantile(values, p))


def fmt(x: float) -> str:
    return f"{x:.9g}"


def main() -> None:
    # Keep chain order within each run so cycle differences are paired.
    pooled: dict[int, np.ndarray] = {}
    paired: dict[tuple[int, int], np.ndarray] = {}
    for run in RUNS:
        rows = read_csv(run / "observables/main_per_sweep_measurements.csv")
        by_sweep: dict[int, np.ndarray] = {}
        for sweep in (200, 250, 300):
            selected = [r for r in rows if int(r["sweep"]) == sweep]
            selected.sort(key=lambda r: int(r["chain_id"]))
            by_sweep[sweep] = np.asarray([[f(r, "phi2"), f(r, "phi4")] for r in selected])
        for sweep, values in by_sweep.items():
            pooled[sweep] = values if sweep not in pooled else np.vstack((pooled[sweep], values))
        for a, b in ((200, 250), (250, 300)):
            delta = by_sweep[b] - by_sweep[a]
            paired[(a, b)] = delta if (a, b) not in paired else np.vstack((paired[(a, b)], delta))

    native_rows = read_csv(NATIVE)
    native = {
        key: np.asarray([f(r, key) for r in native_rows], dtype=np.float64)
        for key in ("phi2", "phi4", "local_kurtosis_ratio")
    }

    lines = [
        "# Pooled phi2/phi4 cycle, covariance, and tail audit",
        "",
        "Pooled data are the 500 chains from each of the cfg2000 and cfg2500 runs (1000 chains total). Cycle changes use paired chains within each run.",
        "",
        "## Cycle means",
        "",
        "| sweep | n | phi2 mean | phi2 SE | phi4 mean | phi4 SE |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    means: dict[int, np.ndarray] = {}
    for sweep in (200, 250, 300):
        values = pooled[sweep]
        means[sweep] = values.mean(axis=0)
        se = values.std(axis=0, ddof=1) / math.sqrt(len(values))
        lines.append(f"| {sweep} | {len(values)} | {fmt(means[sweep][0])} | {fmt(se[0])} | {fmt(means[sweep][1])} | {fmt(se[1])} |")

    lines += [
        "",
        "## Paired cycle changes",
        "",
        "The SE below is the paired-chain SE of the change, not the larger independent-sample SE.",
        "",
        "| change | mean delta phi2 | paired SE | mean delta phi4 | paired SE | corr(delta phi2, delta phi4) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for a, b in ((200, 250), (250, 300)):
        delta = paired[(a, b)]
        dmean = delta.mean(axis=0)
        dse = delta.std(axis=0, ddof=1) / math.sqrt(len(delta))
        corr = np.corrcoef(delta.T)[0, 1]
        lines.append(f"| {a}->{b} | {fmt(dmean[0])} | {fmt(dse[0])} | {fmt(dmean[1])} | {fmt(dse[1])} | {fmt(corr)} |")

    values = pooled[300]
    cov = np.cov(values.T, ddof=1)
    corr = np.corrcoef(values.T)[0, 1]
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    # Fix signs for readable coefficients: the first component has positive phi2 coefficient.
    if eigvecs[0, 0] < 0:
        eigvecs[:, 0] *= -1
    if eigvecs[0, 1] < 0:
        eigvecs[:, 1] *= -1
    centered = values - values.mean(axis=0)
    pc = centered @ eigvecs
    native_mean = np.asarray([native["phi2"].mean(), native["phi4"].mean()])
    drift = means[300] - native_mean
    drift_norm = np.linalg.norm(drift)
    drift_unit = drift / drift_norm
    drift_orth = np.asarray([-drift_unit[1], drift_unit[0]])

    lines += [
        "",
        "## Sweep-300 covariance and principal combinations",
        "",
        f"Pooled covariance matrix (rows/columns [phi2, phi4]): `[[{fmt(cov[0, 0])}, {fmt(cov[0, 1])}], [{fmt(cov[1, 0])}, {fmt(cov[1, 1])}]]`.",
        f"Pearson correlation: `{fmt(corr)}`.",
        "",
        "| combination | alpha (phi2) | beta (phi4) | variance | SD |",
        "|---|---:|---:|---:|---:|",
        f"| covariance PC1, X | {fmt(eigvecs[0, 0])} | {fmt(eigvecs[1, 0])} | {fmt(eigvals[0])} | {fmt(math.sqrt(eigvals[0]))} |",
        f"| covariance PC2, Y | {fmt(eigvecs[0, 1])} | {fmt(eigvecs[1, 1])} | {fmt(eigvals[1])} | {fmt(math.sqrt(eigvals[1]))} |",
        "",
        f"PC1 variance fraction: `{fmt(eigvals[0] / eigvals.sum())}`.",
        f"Sweep-300 generated-minus-native drift vector: `[{fmt(drift[0])}, {fmt(drift[1])}]`; drift-unit combination coefficients: `[{fmt(drift_unit[0])}, {fmt(drift_unit[1])}]`; orthogonal coefficients: `[{fmt(drift_orth[0])}, {fmt(drift_orth[1])}]`.",
        f"Drift projections (generated minus native mean): along drift `{fmt(np.dot(drift, drift_unit))}`, orthogonal `{fmt(np.dot(drift, drift_orth))}`.",
        f"Pooled sweep-300 PC1 and PC2 SDs: `{fmt(pc[:, 0].std(ddof=1))}`, `{fmt(pc[:, 1].std(ddof=1))}`.",
        "",
        "## Sweep-300 tails",
        "",
        "Local kurtosis is computed per configuration as `phi4 / phi2^2`.",
        "",
        "| observable | source | q05 | q95 | q99 | mean | SD |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    tail_specs = [("phi2", (0.05, 0.95)), ("phi4", (0.05, 0.95, 0.99)), ("local_kurtosis_ratio", (0.05, 0.95, 0.99))]
    generated = {
        "phi2": values[:, 0],
        "phi4": values[:, 1],
        "local_kurtosis_ratio": values[:, 1] / np.maximum(values[:, 0] ** 2, 1e-300),
    }
    for key, probs in tail_specs:
        for label, arr in (("pooled flow sweep 300", generated[key]), ("native L32", native[key])):
            qs = [fmt(q(arr, p)) for p in probs]
            while len(qs) < 3:
                qs.append("")
            lines.append(f"| {key} | {label} | {qs[0]} | {qs[1]} | {qs[2]} | {fmt(arr.mean())} | {fmt(arr.std(ddof=1))} |")

    lines += [
        "",
        "## Interpretation",
        "",
        "The cycle table should be read together with the paired changes: a residual mode is supported when both observables move coherently in the same direction and the 250->300 change remains non-negligible relative to its paired SE.",
        "The covariance PCs distinguish a shared radial fluctuation (PC1) from the orthogonal shape/composition direction (PC2). Tail rows show whether the sweep-300 KS differences are primarily location shifts or include tail-shape changes.",
        "",
    ]
    OUT.write_text("\n".join(lines))
    print(OUT)


if __name__ == "__main__":
    main()
