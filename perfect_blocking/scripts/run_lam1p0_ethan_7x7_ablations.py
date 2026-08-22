#!/usr/bin/env python3
"""Ablate Ethan's 7x7 kernel objective on a fixed L32->L16 HMC split.

Each arm is refit from the supplied eta-included Ethan kernel.  The 1,000
configuration test partitions are never used in fitting or start selection;
the L64->L32 evaluation is an independent cross-volume check.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "perfect_blocking"))
from scripts.common.blocking import load_configs  # noqa: E402
from scripts.run_lam1p0_7x7_kernel_search import (  # noqa: E402
    CLASS_MULT, ETA_SCALE, block, full_metrics, matrix_from_classes,
    momentum_extrema, observable_arrays,
)

LAM = ROOT / "perfect_blocking/perfect_blocking_lam1p0"
ORBIT_KEYS = ("00", "10", "11", "20", "21", "22", "30", "31", "32", "33")
FIT_OPERATORS = ("phi2", "phi4", "phi6", "NN", "2nn", "diag", "m2", "G21", "G22", "G30", "G31")
REPORT_OPERATORS = ("action_density", "phi2", "phi4", "phi6", "local_kurtosis_ratio", "NN", "2nn", "diag", "m2", "m4", "G_pmin_avg", "G21", "G22", "G30", "G31")


def cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=LAM / "tests/intermediate/ethan_7x7_objective_ablations")
    p.add_argument("--seed", type=int, default=20260817)
    p.add_argument("--n-train", type=int, default=9000)
    p.add_argument("--n-test", type=int, default=1000)
    p.add_argument("--n-direct-train", type=int, default=4000,
                   help="native L16 reference configurations used in fitting")
    p.add_argument("--starts", type=int, default=4)
    p.add_argument("--maxiter", type=int, default=160)
    return p.parse_args()


def normalize(c: dict[str, float]) -> dict[str, float]:
    out = {k: float(c.get(k, 0.0)) for k in ORBIT_KEYS}
    out["00"] = 1.0 - sum(CLASS_MULT[k] * out[k] for k in ORBIT_KEYS if k != "00")
    return out


def orbit_correlation(phi: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """D4-orbit average of phi(x) phi(x+r), for r=(dx,dy)."""
    offsets = set()
    for sx in (-1, 1):
        for sy in (-1, 1):
            offsets.add((sx * dx, sy * dy))
            offsets.add((sx * dy, sy * dx))
    return np.mean([np.mean(phi * np.roll(np.roll(phi, -x, 1), -y, 2), axis=(1, 2))
                    for x, y in offsets], axis=0)


def operators(phi: np.ndarray) -> dict[str, np.ndarray]:
    out = observable_arrays(phi)
    out["phi6"] = np.mean(phi**6, axis=(1, 2))
    for dx, dy in ((2, 1), (2, 2), (3, 0), (3, 1)):
        out[f"G{dx}{dy}"] = orbit_correlation(phi, dx, dy)
    return out


def inverse_tail_fraction(matrix: np.ndarray, box: int, grid: int = 128) -> float:
    """Absolute inverse-kernel mass outside the centered box, with periodic tails."""
    radius = matrix.shape[0] // 2
    kernel = np.zeros((grid, grid), dtype=float)
    for iy, y in enumerate(range(-radius, radius + 1)):
        for ix, x in enumerate(range(-radius, radius + 1)):
            kernel[y % grid, x % grid] = matrix[iy, ix]
    inverse = np.fft.ifft2(1.0 / np.fft.fft2(kernel)).real
    distance = np.minimum(np.arange(grid), grid - np.arange(grid))
    xx, yy = np.meshgrid(distance, distance, indexing="ij")
    outside = (xx > (box - 1) // 2) | (yy > (box - 1) // 2)
    return float(np.abs(inverse[outside]).sum() / np.abs(inverse).sum())


def rows_for_report(direct: dict[str, np.ndarray], blocked: dict[str, np.ndarray]) -> list[dict[str, float | str]]:
    rows = []
    for name in REPORT_OPERATORS:
        a, b = direct[name], blocked[name]
        combined = np.r_[a, b]
        bins = np.histogram_bin_edges(combined, bins=70)
        cdf_a = np.cumsum(np.histogram(a, bins=bins)[0]) / len(a)
        cdf_b = np.cumsum(np.histogram(b, bins=bins)[0]) / len(b)
        rows.append({"operator": name, "direct_mean": float(np.mean(a)), "blocked_mean": float(np.mean(b)),
                     "std_ratio_direct_over_blocked": float(np.std(a, ddof=1) / np.std(b, ddof=1)),
                     "ks": float(np.max(np.abs(cdf_a - cdf_b)))})
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    a = cli(); a.out.mkdir(parents=True, exist_ok=True)
    direct_all = load_configs(ROOT / "data/configs_phi4_2d/lam1p0_kappac0p340301_L16/configs.npz")
    fine_all = load_configs(ROOT / "data/configs_phi4_2d/lam1p0_kappac0p340301_L32/configs.npz")
    fine64_all = load_configs(ROOT / "data/configs_phi4_2d/lam1p0_kappac0p340301_L64/configs.npz")
    fine_needed = a.n_train + a.n_test
    direct_needed = a.n_direct_train + a.n_test
    if len(direct_all) < direct_needed or len(fine_all) < fine_needed or len(fine64_all) < a.n_test:
        raise ValueError(f"Need {direct_needed} L16, {fine_needed} L32, and {a.n_test} L64 configurations; found L16={len(direct_all)}, L32={len(fine_all)}, L64={len(fine64_all)}")
    rng = np.random.default_rng(a.seed)
    id16, id32 = rng.permutation(len(direct_all))[:direct_needed], rng.permutation(len(fine_all))[:fine_needed]
    id64 = rng.permutation(len(fine64_all))[:a.n_test]
    d_train, d_test = direct_all[id16[:a.n_direct_train]], direct_all[id16[a.n_direct_train:]],
    f_train, f_test = fine_all[id32[:a.n_train]], fine_all[id32[a.n_train:]],
    direct_train, direct_test = operators(d_train), operators(d_test)
    direct32_cross, fine64_cross = operators(fine_all[id32[a.n_train:]]), fine64_all[id64]

    # The provided coefficients have sum 2^(eta/2), so convert them to the
    # normalized base kernel used as the free parameterization here.
    ethan_included = {"00": .888641822, "10": .010508374, "11": -.066254410, "20": .037116968,
                      "21": .022657096, "22": -.001649881, "30": .020222068, "31": .007299892,
                      "32": -.003848342, "33": -.001693935}
    start = normalize({k: v / ETA_SCALE for k, v in ethan_included.items()})
    free_keys = ORBIT_KEYS[1:]

    arms = {
        "full_mu30_N5": {"operators": FIT_OPERATORS, "mu": 30.0, "box": 5},
        "no_phi6_mu30_N5": {"operators": tuple(k for k in FIT_OPERATORS if k != "phi6"), "mu": 30.0, "box": 5},
        "no_extraG_mu30_N5": {"operators": tuple(k for k in FIT_OPERATORS if not k.startswith("G")), "mu": 30.0, "box": 5},
        "full_mu0_N5": {"operators": FIT_OPERATORS, "mu": 0.0, "box": 5},
        "full_mu30_N7": {"operators": FIT_OPERATORS, "mu": 30.0, "box": 7},
    }
    summary = {"split": {"seed": a.seed, "n_fine_train": a.n_train, "n_direct_train": a.n_direct_train, "n_test": a.n_test,
                           "direct_L16_indices": id16.tolist(), "fine_L32_indices": id32.tolist(),
                           "fine_L64_cross_volume_indices": id64.tolist()},
               "reference_kernel": "Ethan coefficients supplied in conversation; eta scale included",
               "objective": "sum of squared standardized mean residuals plus mu times absolute inverse-kernel mass outside box",
               "arms": {}}

    for arm, settings in arms.items():
        arm_out = a.out / arm; arm_out.mkdir(exist_ok=True)
        names = settings["operators"]
        # The denominator estimates the uncertainty of a difference of means.
        ref_mean = np.array([np.mean(direct_train[k]) for k in names])
        ref_var = np.array([np.var(direct_train[k], ddof=1) / len(d_train) for k in names])

        def unpack(x: np.ndarray) -> dict[str, float]:
            return normalize({k: float(v) for k, v in zip(free_keys, x)})

        def objective(x: np.ndarray) -> float:
            classes = unpack(x); matrix = ETA_SCALE * matrix_from_classes(classes)
            momentum = momentum_extrema(matrix, grid=128)
            if momentum["min_K"] <= .30:
                return 1.e10 + 1.e8 * (.30 - momentum["min_K"])
            blocked_train = operators(block(f_train, matrix))
            mean = np.array([np.mean(blocked_train[k]) for k in names])
            var = np.array([np.var(blocked_train[k], ddof=1) / len(f_train) for k in names])
            residual = (mean - ref_mean) / np.sqrt(np.maximum(ref_var + var, 1.e-300))
            return float(residual @ residual + settings["mu"] * inverse_tail_fraction(matrix, settings["box"]))

        x0 = np.array([start[k] for k in free_keys])
        starts = [x0]
        for _ in range(a.starts - 1):
            starts.append(x0 + rng.normal(0.0, .002, size=len(x0)))
        fits = []
        for i, initial in enumerate(starts):
            fit = minimize(objective, initial, method="Powell", bounds=[(-.12, .12)] * len(initial),
                           options={"maxiter": a.maxiter})
            record = {"start": i, "objective": float(fit.fun), "nfev": int(fit.nfev), "success": bool(fit.success),
                      "message": str(fit.message), "classes": unpack(fit.x)}
            fits.append(record)
            (arm_out / "progress.json").write_text(json.dumps(fits, indent=2) + "\n")
            print(f"{arm}: completed start {i + 1}/{len(starts)}; objective={fit.fun:.6g}", flush=True)
        best = min(fits, key=lambda r: r["objective"])
        matrix = ETA_SCALE * matrix_from_classes(best["classes"])
        test_blocked = operators(block(f_test, matrix))
        report = rows_for_report(direct_test, test_blocked)
        write_csv(arm_out / "heldout_L32_to_L16_metrics.csv", report)
        cross_blocked = operators(block(fine64_cross, matrix))
        cross_report = rows_for_report(direct32_cross, cross_blocked)
        write_csv(arm_out / "heldout_L64_to_L32_metrics.csv", cross_report)
        stability = momentum_extrema(matrix, grid=1024)
        result = {"arm": arm, "settings": settings, "best": best, "matrix": matrix.tolist(),
                  "base_orbit_classes_before_eta_scale": best["classes"], "eta_scale_numeric": ETA_SCALE,
                  "momentum_stability": stability, "condition_number": stability["max_K"] / stability["min_K"],
                  "inverse_tail_fraction": inverse_tail_fraction(matrix, settings["box"]),
                  "test_metrics_L32_to_L16": report, "cross_volume_metrics_L64_to_L32": cross_report}
        (arm_out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        summary["arms"][arm] = {"result": str(arm_out / "result.json"), "objective": best["objective"],
                                "condition_number": result["condition_number"], "inverse_tail_fraction": result["inverse_tail_fraction"]}
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(a.out)


if __name__ == "__main__":
    main()
