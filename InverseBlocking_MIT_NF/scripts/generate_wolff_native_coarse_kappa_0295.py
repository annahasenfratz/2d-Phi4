#!/usr/bin/env python3
"""Generate a native L=8 lambda=1, kappa=0.295 coarse ensemble.

This uses the same update structure as the phase-diagram cluster scans:
local Metropolis amplitude sweeps plus embedded Wolff sign-cluster flips.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "outputs" / "coarse_distribution_calibration" / "generated_native_wolff"

L = 8
LAMBDA = 1.0
KAPPA = 0.295
N_SAMPLES = 4096
THERMAL_SWEEPS = 2000
SKIP_SWEEPS = 16
CLUSTERS_PER_SWEEP = 1
PROPOSAL_WIDTH = 0.8
SEED = 20252950


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


def onsite(x: np.ndarray, lam: float) -> np.ndarray:
    return (1.0 - 2.0 * lam) * x * x + lam * x**4


def metropolis_sweep(phi: np.ndarray, kappa: float, lam: float, width: float, rng: np.random.Generator) -> tuple[int, int]:
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
    """Embedded Wolff sign-cluster update at fixed amplitudes."""

    Lx = phi.shape[0]
    seed = (int(rng.integers(Lx)), int(rng.integers(Lx)))
    sign = phi[seed] >= 0.0
    in_cluster = np.zeros((Lx, Lx), dtype=bool)
    in_cluster[seed] = True
    stack = [seed]
    while stack:
        x, y = stack.pop()
        amp = abs(phi[x, y])
        for nx, ny in ((x + 1) % Lx, y), ((x - 1) % Lx, y), (x, (y + 1) % Lx), (x, (y - 1) % Lx):
            if in_cluster[nx, ny] or ((phi[nx, ny] >= 0.0) != sign):
                continue
            p_add = 1.0 - np.exp(-4.0 * kappa * amp * abs(phi[nx, ny]))
            if rng.random() < p_add:
                in_cluster[nx, ny] = True
                stack.append((nx, ny))
    phi[in_cluster] *= -1.0
    return int(np.sum(in_cluster))


def action_density(configs: np.ndarray, kappa: float, lam: float) -> np.ndarray:
    nn = (
        configs * np.roll(configs, -1, axis=1)
        + configs * np.roll(configs, -1, axis=2)
    ).mean(axis=(1, 2))
    onsite_density = ((1.0 - 2.0 * lam) * configs**2 + lam * configs**4).mean(axis=(1, 2))
    return onsite_density - 2.0 * kappa * nn


def observables(configs: np.ndarray, kappa: float, lam: float) -> dict[str, float]:
    m = configs.mean(axis=(1, 2))
    nn = 0.5 * (
        (configs * np.roll(configs, -1, axis=1)).mean(axis=(1, 2))
        + (configs * np.roll(configs, -1, axis=2)).mean(axis=(1, 2))
    )
    return {
        "mean_m": float(np.mean(m)),
        "mean_abs_m": float(np.mean(np.abs(m))),
        "mean_phi2": float(np.mean(configs**2)),
        "mean_phi4": float(np.mean(configs**4)),
        "mean_nn_bond": float(np.mean(nn)),
        "mean_action_density": float(np.mean(action_density(configs, kappa, lam))),
        "std_m": float(np.std(m)),
    }


def write_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    stem = "native_coarse_lam1_kappa0p295_L8_wolff"
    cfg_path = OUT / f"{stem}.npy"
    summary_path = OUT / f"{stem}_summary.json"
    history_path = OUT / f"{stem}_history.csv"

    if cfg_path.exists() and summary_path.exists():
        print(f"reusing existing {cfg_path}")
        return

    rng = np.random.default_rng(SEED)
    phi = rng.normal(size=(L, L))
    stats = ChainStats()
    history: list[dict[str, float | int]] = []

    def sweep() -> None:
        accepts, trials = metropolis_sweep(phi, KAPPA, LAMBDA, PROPOSAL_WIDTH, rng)
        stats.local_accepts += accepts
        stats.local_trials += trials
        for _ in range(CLUSTERS_PER_SWEEP):
            stats.cluster_sites += wolff_sign_cluster(phi, KAPPA, rng)
            stats.cluster_calls += 1

    for sweep_idx in range(THERMAL_SWEEPS):
        sweep()
        if (sweep_idx + 1) % 250 == 0:
            history.append(
                {
                    "stage": 0,
                    "sweep": sweep_idx + 1,
                    "sample": -1,
                    "m": float(phi.mean()),
                    "phi2": float(np.mean(phi**2)),
                    "phi4": float(np.mean(phi**4)),
                }
            )

    configs = np.empty((N_SAMPLES, L, L), dtype=np.float64)
    for i in range(N_SAMPLES):
        for _ in range(SKIP_SWEEPS):
            sweep()
        configs[i] = phi
        if (i + 1) % 128 == 0:
            history.append(
                {
                    "stage": 1,
                    "sweep": THERMAL_SWEEPS + (i + 1) * SKIP_SWEEPS,
                    "sample": i + 1,
                    "m": float(phi.mean()),
                    "phi2": float(np.mean(phi**2)),
                    "phi4": float(np.mean(phi**4)),
                }
            )

    np.save(cfg_path, configs.astype(np.float32))
    write_csv(history_path, history)
    summary = {
        "algorithm": "local Metropolis amplitude sweeps plus embedded Wolff sign-cluster flips",
        "source_algorithm_reference": "phi4_phase-diagram lambda=1 cluster scan update",
        "lambda": LAMBDA,
        "kappa": KAPPA,
        "L": L,
        "n_samples": N_SAMPLES,
        "thermal_sweeps": THERMAL_SWEEPS,
        "skip_sweeps": SKIP_SWEEPS,
        "clusters_per_sweep": CLUSTERS_PER_SWEEP,
        "proposal_width": PROPOSAL_WIDTH,
        "seed": SEED,
        "configs_path": str(cfg_path.resolve()),
        "history_path": str(history_path.resolve()),
        "production_quality": False,
        "quality_note": "Diagnostic native coarse chain; autocorrelation not measured.",
        **stats.as_dict(),
        **observables(configs, KAPPA, LAMBDA),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
