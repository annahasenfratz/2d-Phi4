#!/usr/bin/env python3
"""Refit Ethan's D4-symmetric 7x7 kernel with its normalization free.

The shape is parameterized by a unit-sum base kernel \bar K, while eta is a
separate fitted coordinate: K = 2**(eta/2) \bar K.  This starts exactly from
Ethan's supplied eta-included kernel (eta=0.25) and uses his stated L32->L16
operator objective and inverse-locality penalty (mu=30 outside a 5x5 box).

This is deliberately a non-promoting search.  The held-out L32->L16 and
cross-volume L64->L32 reports are written alongside the fitted kernel.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "perfect_blocking"))
from scripts.common.blocking import load_configs  # noqa: E402
from scripts.run_lam1p0_7x7_kernel_search import (  # noqa: E402
    CLASS_MULT, block, matrix_from_classes, momentum_extrema, observable_arrays,
)

LAM = ROOT / "perfect_blocking/perfect_blocking_lam1p0"
ORBIT_KEYS = ("00", "10", "11", "20", "21", "22", "30", "31", "32", "33")
FREE_KEYS = ORBIT_KEYS[1:]
FIT_OPERATORS = ("phi2", "phi4", "phi6", "NN", "2nn", "diag", "m2", "G21", "G22", "G30", "G31")
REPORT_OPERATORS = (
    "action_density", "phi2", "phi4", "phi6", "local_kurtosis_ratio", "NN", "2nn",
    "diag", "m2", "m4", "G_pmin_avg", "G21", "G22", "G30", "G31",
)
ETHAN_KERNEL = LAM / "kernels/selected_for_upscaling/ethan_7x7_paper_objective_eta_included.json"


def cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20260820)
    p.add_argument("--n-train", type=int, default=9000)
    p.add_argument("--n-test", type=int, default=1000)
    p.add_argument("--n-direct-train", type=int, default=4000)
    p.add_argument("--starts", type=int, default=4)
    p.add_argument("--maxiter", type=int, default=160)
    p.add_argument("--eta-min", type=float, default=0.0)
    p.add_argument("--eta-max", type=float, default=0.5)
    p.add_argument("--mu", type=float, default=30.0)
    p.add_argument("--locality-box", type=int, default=5)
    p.add_argument("--basis-batch-size", type=int, default=250,
                   help="L32 configurations per cached-basis / observable batch")
    p.add_argument("--kernel", type=Path, default=ETHAN_KERNEL)
    return p.parse_args()


def normalize(values: dict[str, float]) -> dict[str, float]:
    out = {k: float(values.get(k, 0.0)) for k in ORBIT_KEYS}
    out["00"] = 1.0 - sum(CLASS_MULT[k] * out[k] for k in FREE_KEYS)
    return out


def orbit_correlation(phi: np.ndarray, dx: int, dy: int) -> np.ndarray:
    offsets: set[tuple[int, int]] = set()
    for sx in (-1, 1):
        for sy in (-1, 1):
            offsets.add((sx * dx, sy * dy))
            offsets.add((sx * dy, sy * dx))
    return np.mean([
        np.mean(phi * np.roll(np.roll(phi, -x, 1), -y, 2), axis=(1, 2))
        for x, y in offsets
    ], axis=0)


def operators(phi: np.ndarray) -> dict[str, np.ndarray]:
    out = observable_arrays(phi)
    out["phi6"] = np.mean(phi**6, axis=(1, 2))
    for dx, dy in ((2, 1), (2, 2), (3, 0), (3, 1)):
        out[f"G{dx}{dy}"] = orbit_correlation(phi, dx, dy)
    return out


def inverse_tail_fraction(matrix: np.ndarray, box: int, grid: int = 128) -> float:
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


def quick_momentum_extrema(matrix: np.ndarray, grid: int = 128) -> dict[str, float]:
    """Fast dense-grid guard for the optimizer; final reporting uses grid=1024."""
    values = np.fft.fft2(np.fft.ifftshift(matrix), s=(grid, grid)).real
    minimum, maximum = float(values.min()), float(values.max())
    inv = 1.0 / values
    return {
        "min_K": minimum, "max_K": maximum,
        "min_inverse_K": float(inv.min()), "max_inverse_K": float(inv.max()),
    }


def report_rows(direct: dict[str, np.ndarray], blocked: dict[str, np.ndarray]) -> list[dict[str, float | str]]:
    rows = []
    for name in REPORT_OPERATORS:
        a, b = direct[name], blocked[name]
        edges = np.histogram_bin_edges(np.r_[a, b], bins=70)
        cdf_a = np.cumsum(np.histogram(a, bins=edges)[0]) / len(a)
        cdf_b = np.cumsum(np.histogram(b, bins=edges)[0]) / len(b)
        rows.append({
            "operator": name, "direct_mean": float(a.mean()), "blocked_mean": float(b.mean()),
            "std_ratio_direct_over_blocked": float(a.std(ddof=1) / b.std(ddof=1)),
            "ks": float(np.max(np.abs(cdf_a - cdf_b))),
        })
    return rows


def write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    a = cli()
    if not (a.eta_min <= 0.25 <= a.eta_max):
        raise ValueError("eta bounds must contain Ethan's starting eta=0.25")
    if a.locality_box % 2 != 1:
        raise ValueError("--locality-box must be odd")
    a.out.mkdir(parents=True, exist_ok=True)
    direct_all = load_configs(ROOT / "data/configs_phi4_2d/lam1p0_kappac0p340301_L16/configs.npz")
    fine32_all = load_configs(ROOT / "data/configs_phi4_2d/lam1p0_kappac0p340301_L32/configs.npz")
    fine64_all = load_configs(ROOT / "data/configs_phi4_2d/lam1p0_kappac0p340301_L64/configs.npz")
    if (len(direct_all) < a.n_direct_train + a.n_test
            or len(fine32_all) < a.n_train + a.n_test
            or len(fine64_all) < a.n_test):
        raise ValueError("insufficient configurations for requested train/test split")

    rng = np.random.default_rng(a.seed)
    ids16 = rng.permutation(len(direct_all))[:a.n_direct_train + a.n_test]
    ids32 = rng.permutation(len(fine32_all))[:a.n_train + a.n_test]
    ids64 = rng.permutation(len(fine64_all))[:a.n_test]
    d_train, d_test = direct_all[ids16[:a.n_direct_train]], direct_all[ids16[a.n_direct_train:]]
    f_train, f_test = fine32_all[ids32[:a.n_train]], fine32_all[ids32[a.n_train:]]
    direct_train, direct_test = operators(d_train), operators(d_test)
    direct32_cross = operators(fine32_all[ids32[a.n_train:]])

    supplied = json.loads(a.kernel.read_text())
    supplied_matrix = np.asarray(supplied["matrix"], dtype=float)
    eta0 = float(supplied.get("eta", 0.25))
    scale0 = 2.0 ** (eta0 / 2.0)
    base0 = supplied_matrix / scale0
    start = normalize({k: float(base0[3 + int(k[0]), 3 + int(k[1])]) for k in ORBIT_KEYS})
    x0 = np.array([start[k] for k in FREE_KEYS] + [eta0], dtype=float)
    ref_mean = np.array([direct_train[k].mean() for k in FIT_OPERATORS])
    ref_var = np.array([direct_train[k].var(ddof=1) / len(d_train) for k in FIT_OPERATORS])

    # Blocking is linear in every D4 orbit coefficient.  The old driver
    # repeatedly applied all 49 shifts to 9,000 L32 fields at every Powell
    # evaluation.  Cache the ten unit-orbit blocked fields once instead; an
    # evaluation now forms one weighted sum on the L16 fields before measuring
    # the nonlinear observables.
    cache_path = a.out / "orbit_blocking_bases_float32.npy"
    cache_marker = a.out / "orbit_blocking_bases_metadata.json"
    basis_shape = (len(ORBIT_KEYS), len(f_train), f_train.shape[1] // 2, f_train.shape[2] // 2)
    cached = False
    if cache_path.exists() and cache_marker.exists():
        try:
            marker = json.loads(cache_marker.read_text())
            cached = tuple(marker.get("shape", ())) == basis_shape
        except (json.JSONDecodeError, OSError):
            cached = False
    if cached:
        print("reusing cached ten D4-orbit blocking bases", flush=True)
        basis = np.load(cache_path, mmap_mode="r")
    else:
        print("precomputing ten D4-orbit blocking bases in chunks", flush=True)
        basis = open_memmap(cache_path, mode="w+", dtype=np.float32, shape=basis_shape)
        units = []
        for key in ORBIT_KEYS:
            unit = {name: 0.0 for name in ORBIT_KEYS}
            unit[key] = 1.0
            units.append(matrix_from_classes(unit))
        for lo in range(0, len(f_train), a.basis_batch_size):
            hi = min(lo + a.basis_batch_size, len(f_train))
            fine_batch = f_train[lo:hi]
            for i, unit_matrix in enumerate(units):
                basis[i, lo:hi] = block(fine_batch, unit_matrix).astype(np.float32)
            basis.flush()
            (a.out / "basis_progress.json").write_text(json.dumps({
                "status": "building", "completed": hi, "total": len(f_train),
                "batch_size": a.basis_batch_size,
            }, indent=2) + "\n")
            print(f"cached bases through {hi}/{len(f_train)} L32 configurations", flush=True)
        cache_marker.write_text(json.dumps({"shape": basis_shape, "dtype": "float32", "completed": True}, indent=2) + "\n")
        (a.out / "basis_progress.json").unlink(missing_ok=True)
        basis = np.load(cache_path, mmap_mode="r")
    (a.out / "run_metadata.json").write_text(json.dumps({
        "status": "running", "optimization": "Powell with cached D4-orbit blocking bases",
        "n_train": len(f_train), "n_orbit_bases": len(ORBIT_KEYS),
        "eta_bounds": [a.eta_min, a.eta_max], "mu": a.mu, "locality_box": a.locality_box,
    }, indent=2) + "\n")

    def unpack(x: np.ndarray) -> tuple[dict[str, float], float, np.ndarray]:
        classes = normalize({k: float(v) for k, v in zip(FREE_KEYS, x[:-1])})
        eta = float(x[-1])
        matrix = (2.0 ** (eta / 2.0)) * matrix_from_classes(classes)
        return classes, eta, matrix

    evaluation_count = 0
    cache: dict[tuple[float, ...], float] = {}

    def objective(x: np.ndarray) -> float:
        nonlocal evaluation_count
        key = tuple(np.round(x, 12))
        if key in cache:
            return cache[key]
        evaluation_count += 1
        classes, eta, matrix = unpack(x)
        mom = quick_momentum_extrema(matrix, grid=128)
        if not np.isfinite(mom["min_K"]) or mom["min_K"] <= 0.30:
            value = 1.0e10 + 1.0e8 * max(0.0, 0.30 - mom["min_K"])
        else:
            coeff = (2.0 ** (eta / 2.0)) * np.array([classes[k] for k in ORBIT_KEYS])
            sums = np.zeros(len(FIT_OPERATORS), dtype=float)
            sums2 = np.zeros(len(FIT_OPERATORS), dtype=float)
            for lo in range(0, len(f_train), a.basis_batch_size):
                hi = min(lo + a.basis_batch_size, len(f_train))
                blocked_field = np.tensordot(coeff, basis[:, lo:hi], axes=(0, 0))
                blocked = operators(blocked_field)
                for i, name in enumerate(FIT_OPERATORS):
                    values = blocked[name]
                    sums[i] += values.sum()
                    sums2[i] += np.square(values).sum()
            mean = sums / len(f_train)
            var = (sums2 - len(f_train) * mean * mean) / (len(f_train) - 1)
            var /= len(f_train)
            residual = (mean - ref_mean) / np.sqrt(np.maximum(ref_var + var, 1.0e-300))
            value = float(residual @ residual + a.mu * inverse_tail_fraction(matrix, a.locality_box))
        cache[key] = float(value)
        if evaluation_count == 1 or evaluation_count % 20 == 0:
            (a.out / "inflight.json").write_text(json.dumps({
                "status": "running", "evaluation": evaluation_count, "objective": float(value),
                "eta": eta, "eta_scale_numeric": float(2.0 ** (eta / 2.0)),
                "base_orbit_classes": classes, "min_K": mom["min_K"],
            }, indent=2) + "\n")
            print(f"evaluation {evaluation_count}: eta={eta:.7g}, objective={value:.7g}", flush=True)
        return float(value)

    starts = [x0]
    for _ in range(a.starts - 1):
        trial = x0.copy()
        trial[:-1] += rng.normal(0.0, 0.002, size=len(FREE_KEYS))
        trial[-1] += rng.normal(0.0, 0.015)
        trial[-1] = np.clip(trial[-1], a.eta_min, a.eta_max)
        starts.append(trial)
    progress: list[dict[str, object]] = []
    bounds = [(-0.12, 0.12)] * len(FREE_KEYS) + [(a.eta_min, a.eta_max)]
    for i, initial in enumerate(starts):
        fit = minimize(objective, initial, method="Powell", bounds=bounds,
                       options={"maxiter": a.maxiter, "xtol": 1.0e-5, "ftol": 1.0e-5})
        classes, eta, matrix = unpack(fit.x)
        record: dict[str, object] = {
            "start": i, "objective": float(fit.fun), "nfev": int(fit.nfev),
            "success": bool(fit.success), "message": str(fit.message),
            "base_orbit_classes": classes, "eta": eta,
            "eta_scale_numeric": float(2.0 ** (eta / 2.0)),
        }
        progress.append(record)
        (a.out / "progress.json").write_text(json.dumps(progress, indent=2) + "\n")
        print(f"completed start {i + 1}/{len(starts)}: eta={eta:.7g}, objective={fit.fun:.7g}", flush=True)
    best = min(progress, key=lambda r: float(r["objective"]))
    classes = dict(best["base_orbit_classes"])
    eta = float(best["eta"])
    matrix = (2.0 ** (eta / 2.0)) * matrix_from_classes(classes)
    stability = momentum_extrema(matrix, grid=1024)
    test = report_rows(direct_test, operators(block(f_test, matrix)))
    cross = report_rows(direct32_cross, operators(block(fine64_all[ids64], matrix)))
    write_csv(a.out / "heldout_L32_to_L16_metrics.csv", test)
    write_csv(a.out / "heldout_L64_to_L32_metrics.csv", cross)
    result = {
        "name": "ethan_7x7_free_eta_nonpromoting", "description": __doc__.splitlines()[0],
        "source_kernel": str(a.kernel), "eta": eta, "eta_scale_numeric": float(2.0 ** (eta / 2.0)),
        "kernel_coefficients_include_eta_scale": True,
        "normalization": "sum(K)=2^(eta/2), with eta fitted", "matrix": matrix.tolist(),
        "base_orbit_classes_before_eta_scale": classes,
        "orbit_coefficients_eta_included": {k: float(matrix[3 + int(k[0]), 3 + int(k[1])]) for k in ORBIT_KEYS},
        "fit_operators": FIT_OPERATORS,
        "objective": "sum squared standardized mean residuals + mu * inverse tail fraction",
        "mu": a.mu, "inverse_locality_box": a.locality_box,
        "eta_bounds": [a.eta_min, a.eta_max], "momentum_stability": stability,
        "condition_number": float(stability["max_K"] / stability["min_K"]),
        "inverse_tail_fraction": inverse_tail_fraction(matrix, a.locality_box),
        "best_fit": best,
        "split": {"seed": a.seed, "direct_L16_indices": ids16.tolist(), "fine_L32_indices": ids32.tolist(), "fine_L64_indices": ids64.tolist()},
    }
    (a.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    (a.out / "kernel_eta_included.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": "completed", "eta": eta, "out": str(a.out)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
