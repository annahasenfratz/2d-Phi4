#!/usr/bin/env python3
"""Deterministically extend the frozen L64 calibrated-initializer relaxation test."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "perfect_blocking_upsampling/scripts"))
sys.path.insert(0, str(ROOT / "perfect_blocking_upsampling/src"))

from perfect_blocking_upsampling.actions import ActionSpec, action_total
from perfect_blocking_upsampling.observables import second_moment_components
from run_native_l32_metropolis import metropolis_sweep, StreamingCsv

RUN = ROOT / "perfect_blocking_upsampling/runs/lam1p0/fresh_disjoint_calibrated_empirical_validation_20260721"
OUT = ROOT / "perfect_blocking_upsampling/runs/lam1p0/calibrated_empirical_fine_rethermalization_20260721"
SAVES = (0, 1, 2, 5, 10, 20, 30, 50, 75, 100, 150, 200)
HIST_SWEEPS = (0, 10, 20, 50, 100, 200)
OBSERVABLES = ("action_density", "phi2", "phi4", "local_kurtosis", "NN", "diag", "2nn", "m2", "m4", "G_pmin_avg")


def atomic_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def config_observables(phi: np.ndarray, action: ActionSpec) -> dict[str, np.ndarray]:
    field = np.asarray(phi, dtype=np.float64)
    phi2 = np.mean(field**2, axis=(1, 2))
    phi4 = np.mean(field**4, axis=(1, 2))
    fft = np.fft.fft2(field, axes=(1, 2))
    gp = 0.5 * (np.abs(fft[:, 1, 0]) ** 2 + np.abs(fft[:, 0, 1]) ** 2) / (field.shape[1] ** 2)
    # Match the established rethermalization observable convention exactly:
    # NN and 2nn are orientation-averaged, while action_density uses the
    # project-local observable normalization (not ActionSpec.action_density).
    nn = 0.5 * np.mean(field * np.roll(field, -1, 1) + field * np.roll(field, -1, 2), axis=(1, 2))
    two_nn = 0.5 * np.mean(field * np.roll(field, -2, 1) + field * np.roll(field, -2, 2), axis=(1, 2))
    return {
        "action_density": 1.0 - 2.0 * phi2 + phi4 - 4.0 * action.kappa * nn,
        "phi2": phi2,
        "phi4": phi4,
        "local_kurtosis": phi4 / np.maximum(phi2 * phi2, 1.0e-15),
        "NN": nn,
        "diag": np.mean(field * np.roll(np.roll(field, -1, 1), -1, 2), axis=(1, 2)),
        "2nn": two_nn,
        "m2": np.mean(field, axis=(1, 2)) ** 2,
        "m4": np.mean(field, axis=(1, 2)) ** 4,
        "G_pmin_avg": gp,
    }


def ensemble_scalars(phi: np.ndarray) -> dict[str, float]:
    field = np.asarray(phi, dtype=np.float64)
    m = np.mean(field, axis=(1, 2))
    m2 = float(np.mean(m * m))
    m4 = float(np.mean(m**4))
    moment = second_moment_components(field)
    return {
        "Binder": float(1.0 - m4 / (3.0 * m2 * m2)) if m2 > 0 else float("nan"),
        "chi": float(field.shape[1] ** 2 * (m2 - np.mean(m) ** 2)),
        "xi_over_L": float(moment["xi_over_L"]),
    }


def overlap(a: np.ndarray, b: np.ndarray) -> float:
    lo, hi = min(float(a.min()), float(b.min())), max(float(a.max()), float(b.max()))
    bins = np.linspace(lo, hi, 81) if hi > lo else np.array([lo - 1.0, hi + 1.0])
    ha, _ = np.histogram(a, bins=bins, density=True)
    hb, _ = np.histogram(b, bins=bins, density=True)
    return float(np.minimum(ha, hb).sum() * (bins[1] - bins[0]))


def compare(reference: dict[str, np.ndarray], current: dict[str, np.ndarray], sweep: int, ensemble: str, at20: dict[str, np.ndarray] | None) -> list[dict]:
    rows: list[dict] = []
    for name in OBSERVABLES:
        native, value = reference[name], current[name]
        std = float(np.std(native, ddof=1))
        paired = value - native
        row = {
            "sweep": sweep,
            "ensemble": ensemble,
            "observable": name,
            "native_mean": float(np.mean(native)),
            "value_mean": float(np.mean(value)),
            "mean_shift_native_sigma": float((np.mean(value) - np.mean(native)) / max(std, 1.0e-15)),
            "width_ratio": float(np.std(value, ddof=1) / max(std, 1.0e-15)),
            "KS": float(ks_2samp(native, value).statistic),
            "overlap": overlap(native, value),
            "W1": float(wasserstein_distance(native, value)),
            "q01_coverage": float(np.mean((value >= np.quantile(native, .01)) & (value <= np.quantile(native, .99)))),
            "q05_coverage": float(np.mean((value >= np.quantile(native, .05)) & (value <= np.quantile(native, .95)))),
            "q10_coverage": float(np.mean((value >= np.quantile(native, .10)) & (value <= np.quantile(native, .90)))),
            "q90_coverage": float(np.mean(value >= np.quantile(native, .90))),
            "q95_coverage": float(np.mean(value >= np.quantile(native, .95))),
            "q99_coverage": float(np.mean(value >= np.quantile(native, .99))),
            "paired_change_sweep0_mean": float(np.mean(paired)),
            "paired_change_sweep0_std": float(np.std(paired, ddof=1)),
        }
        if at20 is not None:
            change20 = value - at20[name]
            row.update({"paired_change_sweep20_mean": float(np.mean(change20)), "paired_change_sweep20_std": float(np.std(change20, ddof=1))})
        rows.append(row)
    return rows


def bootstrap_widths(reference: dict[str, np.ndarray], snapshots: dict[int, dict[str, np.ndarray]], seed: int = 7351) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for name in ("phi2", "phi4", "action_density", "NN"):
        native = reference[name]
        n = len(native)
        indices = rng.integers(0, n, size=(1000, n))
        ref_std = np.std(native[indices], axis=1, ddof=1)
        for sweep, obs in snapshots.items():
            ratios = np.std(obs[name][indices], axis=1, ddof=1) / ref_std
            rows.append({"observable": name, "sweep": sweep, "width_ratio": float(np.std(obs[name], ddof=1) / np.std(native, ddof=1)), "bootstrap_mean": float(np.mean(ratios)), "bootstrap_se": float(np.std(ratios, ddof=1)), "ci_low": float(np.quantile(ratios, .025)), "ci_high": float(np.quantile(ratios, .975))})
    return rows


def verify_recovery(saved_metrics: Path, reference: dict[str, np.ndarray], recovered: dict[str, np.ndarray]) -> list[dict]:
    legacy = list(csv.DictReader(saved_metrics.open()))
    rows = []
    by_name = {row["observable"]: row for row in legacy if row["sweep"] == "20" and row["ensemble"] == "calibrated"}
    for name in OBSERVABLES:
        legacy_name = "local_kurtosis_ratio" if name == "local_kurtosis" else name
        expected = float(by_name[legacy_name]["generated_mean"])
        actual = float(np.mean(recovered[name]))
        rows.append({"observable": name, "saved_sweep20_mean": expected, "recovered_sweep20_mean": actual, "absolute_difference": abs(actual - expected)})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-chains", type=int, default=100)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    payload = np.load(RUN / "paired_fields_L64.npz")
    calibrated = payload["calibrated"][: args.n_chains].copy()
    native = payload["native"][: args.n_chains].copy()
    reference = native.copy()
    action = ActionSpec("phi4_nn", 1.0, 0.340301)
    rng = np.random.default_rng(998 + 64)
    update_path = OUT / "updates_L64_extension.csv"
    writer = StreamingCsv(update_path, ["sweep", "pass", "update_order", "parity", "sites_touched", "attempts", "accepted", "acceptance", "DeltaS_mean", "DeltaS_std", "DeltaS_min", "DeltaS_max", "log_accept_mean", "log_accept_std", "elapsed_sec"])
    ref_obs = config_observables(reference, action)
    snapshots: dict[int, dict[str, np.ndarray]] = {}
    metric_rows: list[dict] = []
    check_rows: list[dict] = []
    recovered_checked = False
    for sweep in range(201):
        if sweep in SAVES:
            cal_obs, native_obs = config_observables(calibrated, action), config_observables(native, action)
            snapshots[sweep] = cal_obs
            metric_rows.extend(compare(ref_obs, cal_obs, sweep, "calibrated", snapshots.get(20)))
            metric_rows.extend(compare(ref_obs, native_obs, sweep, "native_control", None))
            for label, field in (("calibrated", calibrated), ("native_control", native)):
                for name, value in ensemble_scalars(field).items():
                    metric_rows.append({"sweep": sweep, "ensemble": label, "observable": name, "value_mean": value})
            if sweep in HIST_SWEEPS:
                np.savez_compressed(OUT / f"states_L64_sweep{sweep:03d}.npz", calibrated=calibrated, native_control=native, reference_native=reference)
        if sweep == 20 and not recovered_checked:
            check_rows = verify_recovery(OUT / "metrics_L64.csv", ref_obs, snapshots[20])
            # The original metrics path used float32 Torch observables; this
            # extension uses float64 NumPy for the new diagnostics.
            if max(row["absolute_difference"] for row in check_rows) > 2.0e-6:
                raise RuntimeError(f"deterministic sweep-20 recovery failed: {check_rows}")
            recovered_checked = True
        if sweep < 200:
            calibrated, _ = metropolis_sweep(calibrated, action, sweep + 1, 1, .5, "checkerboard", rng, writer)
            native, _ = metropolis_sweep(native, action, sweep + 1, 1, .5, "checkerboard", rng, writer)
    writer.close()
    atomic_csv(OUT / "metrics_L64_extended.csv", metric_rows)
    atomic_csv(OUT / "sweep20_recovery_validation.csv", check_rows)
    atomic_csv(OUT / "width_bootstrap_L64.csv", bootstrap_widths(ref_obs, snapshots))
    atomic_csv(OUT / "state_observables_L64_extended.csv", [
        {"sweep": sweep, "chain_id": chain, **{name: float(obs[name][chain]) for name in OBSERVABLES}}
        for sweep, obs in snapshots.items() for chain in range(args.n_chains)
    ])
    (OUT / "L64_extension_manifest.json").write_text(json.dumps({"n_chains": args.n_chains, "saves": SAVES, "histogram_sweeps": HIST_SWEEPS, "update": {"algorithm": "checkerboard", "step_size": .5, "passes": 1, "seed": 1062}, "initializer": "frozen calibrated empirical proposal", "reconstructed_sweeps_0_to_20": True, "sweep20_recovery_verified": True}, indent=2) + "\n")


if __name__ == "__main__":
    main()
