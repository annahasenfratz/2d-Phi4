#!/usr/bin/env python3
"""Targeted lambda=1.0 Pareto kernel search for phi2/local-kurtosis matching.

This is a non-promotion search.  It compares direct native L16 configurations
against native L32 configurations blocked to L16 using eta-included candidate
kernels, with phi2/phi4/local-kurtosis treated as primary constrained targets.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.blocking import load_configs  # noqa: E402
from scripts.common.histogram_compare import metrics, plot_histogram  # noqa: E402
from scripts.common.kernel_io import load_kernel  # noqa: E402
from scripts.run_lam1p0_7x7_kernel_search import ETA_SCALE, block, momentum_extrema, observable_arrays  # noqa: E402


LAM_ROOT = PROJECT_ROOT / "perfect_blocking/perfect_blocking_lam1p0"
DIRECT = PROJECT_ROOT / "data/configs_phi4_2d/lam1p0_kappac0p340301_L16/configs.npz"
FINE = PROJECT_ROOT / "data/configs_phi4_2d/lam1p0_kappac0p340301_L32/configs.npz"

OUT = LAM_ROOT / "tests/intermediate/redo_kernel_phi2_phi4_pareto_2000"
CAND_DIR = LAM_ROOT / "kernels/candidates/redo_phi2_phi4_pareto_2000"
FINAL = LAM_ROOT / "tests/final/final_kernel_confirmation_direct_L16_vs_blocked_L32_phi2_phi4_pareto_2000"
BASELINE_FAILURE = LAM_ROOT / "tests/final/direct_L16_vs_blocked_L32_baseline_failure/baseline_failure_histogram_metrics.csv"

SELECTED = LAM_ROOT / "kernels/selected_for_upscaling/best_7x7_phi2_nn_guarded_eta_included.json"
CURRENT_FINAL = LAM_ROOT / "kernels/final/chosen_kernel.json"
RETRAINED_5X5 = LAM_ROOT / "kernels/selected_for_upscaling/best_5x5_retrained_full_objective_eta_included.json"
NN_CONSTRAINED = LAM_ROOT / "kernels/candidates/7x7_no33_nn_constrained/best_7x7_no33_nn_constrained_eta_included.json"
PHI2_NN_GUARDED = LAM_ROOT / "kernels/candidates/7x7_full_retraining_phi2_nn_guarded/best_7x7_full_retraining_controlled_eta_included.json"
PREVIOUS_REDO = LAM_ROOT / "kernels/candidates/redo_phi2_phi4/best_redo_phi2_phi4_eta_included.json"

OBS = [
    "action_density",
    "phi2",
    "phi4",
    "local_kurtosis_ratio",
    "NN",
    "2nn",
    "diag",
    "m",
    "m2",
    "m4",
    "Binder_U4_from_averages",
    "xi_over_L",
    "G_00",
    "G_10",
    "G_01",
    "G_pmin_avg",
]
KEY_OBS = ["action_density", "phi2", "phi4", "local_kurtosis_ratio", "NN", "2nn", "diag", "m2", "m4", "G_pmin_avg"]
PLOT_OBS = ["phi2", "phi4", "local_kurtosis_ratio", "action_density", "NN", "G_pmin_avg"]

SEARCH_N_CONFIGS = 2000
SUBSET_N = 1000
RNG_SEED = 2026071817
MAX_INV_SOFT = 1.45
MAX_INV_HARD = 1.60


def orbit_keys(radius: int) -> list[str]:
    return [f"{a}{b}" for a in range(radius + 1) for b in range(a + 1)]


def orbit_mult(radius: int) -> dict[str, int]:
    mult: dict[str, int] = {}
    for a in range(radius + 1):
        for b in range(a + 1):
            if a == 0 and b == 0:
                m = 1
            elif b == 0 or a == b:
                m = 4
            else:
                m = 8
            mult[f"{a}{b}"] = m
    return mult


def matrix_from_classes(classes: dict[str, float], radius: int) -> np.ndarray:
    matrix = np.zeros((2 * radius + 1, 2 * radius + 1), dtype=np.float64)
    for ix, dx in enumerate(range(-radius, radius + 1)):
        for iy, dy in enumerate(range(-radius, radius + 1)):
            a, b = sorted((abs(dx), abs(dy)), reverse=True)
            matrix[ix, iy] = float(classes.get(f"{a}{b}", 0.0))
    return matrix


def classes_from_matrix(matrix: np.ndarray, target_radius: int | None = None) -> dict[str, float]:
    source_radius = matrix.shape[0] // 2
    radius = source_radius if target_radius is None else target_radius
    out = {key: 0.0 for key in orbit_keys(radius)}
    for a in range(min(source_radius, radius) + 1):
        for b in range(a + 1):
            out[f"{a}{b}"] = float(matrix[source_radius + a, source_radius + b])
    return out


def normalize(classes: dict[str, float], radius: int, zero_keys: set[str] | None = None) -> dict[str, float]:
    zero_keys = zero_keys or set()
    out = {key: float(classes.get(key, 0.0)) for key in orbit_keys(radius)}
    for key in zero_keys:
        out[key] = 0.0
    mult = orbit_mult(radius)
    rest = sum(mult[key] * val for key, val in out.items() if key != "00")
    out["00"] = 1.0 - rest
    return out


def eta_matrix_from_json(path: Path, target_radius: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    data = json.loads(path.read_text())
    if "matrix" in data:
        matrix = np.asarray(data["matrix"], dtype=np.float64)
        meta = data
    else:
        spec = load_kernel(path)
        matrix = spec.matrix
        meta = spec.metadata
    if target_radius is not None and matrix.shape[0] != 2 * target_radius + 1:
        source_radius = matrix.shape[0] // 2
        if source_radius <= target_radius:
            out = np.zeros((2 * target_radius + 1, 2 * target_radius + 1), dtype=np.float64)
            lo = target_radius - source_radius
            out[lo : lo + matrix.shape[0], lo : lo + matrix.shape[1]] = matrix
            matrix = out
        else:
            lo = source_radius - target_radius
            hi = lo + 2 * target_radius + 1
            matrix = matrix[lo:hi, lo:hi]
    return matrix, meta


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def scalar_score(row: dict[str, Any]) -> float:
    std_ratio = float(row["std_ratio_a_over_b"])
    width = 0.0 if not np.isfinite(std_ratio) or std_ratio <= 0 else abs(math.log(std_ratio))
    return (
        float(row["ks_statistic"])
        + 1.5 * float(row["total_variation"])
        + 15.0 * float(row["jensen_shannon"])
        + 0.35 * abs(float(row["standardized_mean_shift"]))
        + 0.20 * width
    )


def full_metrics(direct_obs: dict[str, np.ndarray], blocked_obs: dict[str, np.ndarray], bins: int = 70) -> dict[str, dict[str, Any]]:
    return {obs: metrics(direct_obs[obs], blocked_obs[obs], bins=bins) for obs in OBS}


class BasisEvaluator:
    def __init__(self, fine: np.ndarray, radius: int):
        self.radius = radius
        self.keys = orbit_keys(radius)
        self.basis: dict[str, np.ndarray] = {}
        for key in self.keys:
            classes = {k: 0.0 for k in self.keys}
            classes[key] = 1.0
            self.basis[key] = block(fine, ETA_SCALE * matrix_from_classes(classes, radius))

    def blocked(self, classes: dict[str, float]) -> np.ndarray:
        out: np.ndarray | None = None
        for key, value in classes.items():
            if value == 0.0:
                continue
            term = value * self.basis[key]
            out = term.copy() if out is None else out + term
        if out is None:
            raise ValueError("empty kernel")
        return out


def objective(rows: dict[str, dict[str, Any]], mom: dict[str, float], classes: dict[str, float], radius: int, selected: dict[str, dict[str, Any]]) -> float:
    # Primary score heavily favors phi2 and local-kurtosis while retaining phi4 as
    # a hard protected shape target rather than allowing a phi4-only optimum.
    weights = {
        "phi2": 28.0,
        "local_kurtosis_ratio": 35.0,
        "phi4": 12.0,
        "action_density": 10.0,
        "NN": 10.0,
        "2nn": 4.0,
        "diag": 4.0,
        "m2": 4.0,
        "m4": 4.0,
        "G_pmin_avg": 7.0,
    }
    val = sum(weights[obs] * scalar_score(rows[obs]) for obs in weights)
    # Constrained penalties.  These are intentionally stronger than the first
    # broad screen: the goal is to map the feasible frontier, not reward a local
    # observable win that breaks action/NN.
    hard = {
        "phi2": 0.060,
        "local_kurtosis_ratio": 0.190,
        "phi4": 0.032,
        "action_density": 0.040,
        "NN": 0.035,
        "2nn": 0.040,
        "diag": 0.040,
        "m2": max(0.030, float(selected["m2"]["ks_statistic"]) + 0.004),
        "m4": max(0.030, float(selected["m4"]["ks_statistic"]) + 0.004),
        "G_pmin_avg": 0.025,
    }
    for obs, limit in hard.items():
        excess = max(0.0, float(rows[obs]["ks_statistic"]) - limit)
        w = 1.8e6 if obs in {"phi2", "local_kurtosis_ratio"} else 9.0e5
        val += w * excess * excess
    shift_limits = {"phi2": 0.12, "local_kurtosis_ratio": 0.45, "action_density": 0.08, "NN": 0.08, "phi4": 0.08}
    for obs, limit in shift_limits.items():
        excess = max(0.0, abs(float(rows[obs]["standardized_mean_shift"])) - limit)
        val += 2.0e5 * excess * excess
    val += 5.0e6 * max(0.0, -float(mom["min_K"])) ** 2
    val += 5.0e5 * max(0.0, float(mom["max_inverse_K"]) - MAX_INV_HARD) ** 2
    val += 6.0e4 * max(0.0, float(mom["max_inverse_K"]) - MAX_INV_SOFT) ** 2
    mult = orbit_mult(radius)
    for key, coeff in classes.items():
        if key == "00":
            continue
        a = int(key[0])
        if a >= 3:
            val += 250.0 * mult[key] * coeff * coeff
        if a >= 4:
            val += 2500.0 * mult[key] * coeff * coeff
    return float(val)


def category(rows: dict[str, dict[str, Any]], mom: dict[str, float]) -> str:
    if (
        float(rows["phi2"]["ks_statistic"]) < 0.060
        and float(rows["local_kurtosis_ratio"]["ks_statistic"]) < 0.190
        and abs(float(rows["phi2"]["standardized_mean_shift"])) < 0.12
        and abs(float(rows["local_kurtosis_ratio"]["standardized_mean_shift"])) < 0.45
        and float(rows["action_density"]["ks_statistic"]) <= 0.040
        and float(rows["NN"]["ks_statistic"]) <= 0.035
        and float(rows["G_pmin_avg"]["ks_statistic"]) <= 0.025
        and float(rows["phi4"]["ks_statistic"]) <= 0.032
        and float(mom["min_K"]) > 0.0
        and float(mom["max_inverse_K"]) <= MAX_INV_HARD
    ):
        return "promotion-worthy"
    if float(mom["min_K"]) > 0.0 and float(mom["max_inverse_K"]) <= MAX_INV_HARD:
        return "pareto"
    return "rejected-momentum"


def eval_classes(
    name: str,
    family: str,
    classes: dict[str, float],
    radius: int,
    direct_obs: dict[str, np.ndarray],
    evaluator: BasisEvaluator,
    selected_rows: dict[str, dict[str, Any]],
    bins: int,
    momentum_grid: int,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, float], np.ndarray]:
    base = matrix_from_classes(classes, radius)
    matrix = ETA_SCALE * base
    rows = full_metrics(direct_obs, observable_arrays(evaluator.blocked(classes)), bins=bins)
    mom = momentum_extrema(matrix, grid=momentum_grid)
    rec: dict[str, Any] = {
        "candidate": name,
        "family": family,
        "radius": radius,
        "sum_base": float(base.sum()),
        "sum_K": float(matrix.sum()),
        "eta_scale": ETA_SCALE,
        "kernel_coefficients_include_eta_scale": True,
        "objective_score": objective(rows, mom, classes, radius, selected_rows),
        "category": category(rows, mom),
        "min_K": mom["min_K"],
        "max_K": mom["max_K"],
        "min_inverse_K": mom["min_inverse_K"],
        "max_inverse_K": mom["max_inverse_K"],
        "condition_number": mom["max_K"] / mom["min_K"],
    }
    for obs in KEY_OBS:
        rec[f"{obs}_KS"] = rows[obs]["ks_statistic"]
        rec[f"{obs}_JS"] = rows[obs]["jensen_shannon"]
        rec[f"{obs}_TV"] = rows[obs]["total_variation"]
        rec[f"{obs}_W1"] = rows[obs]["wasserstein_1"]
        rec[f"{obs}_shift"] = rows[obs]["standardized_mean_shift"]
        rec[f"{obs}_std_ratio"] = rows[obs]["std_ratio_a_over_b"]
    return rec, rows, mom, matrix


def random_perturb(rng: np.random.Generator, center: dict[str, float], radius: int, scale: float, variables: list[str], zero_keys: set[str]) -> dict[str, float]:
    cls = dict(center)
    for key in variables:
        # Larger steps on the short-distance core, smaller steps on outer shells.
        shell = int(key[0])
        local_scale = scale / (1.0 + 0.45 * max(0, shell - 2))
        cls[key] = float(cls.get(key, 0.0) + rng.normal(0.0, local_scale))
    return normalize(cls, radius, zero_keys)


def optimize_center(
    name: str,
    family: str,
    center: dict[str, float],
    radius: int,
    variables: list[str],
    zero_keys: set[str],
    direct_obs: dict[str, np.ndarray],
    evaluator: BasisEvaluator,
    selected_rows: dict[str, dict[str, Any]],
    maxiter: int,
) -> tuple[dict[str, float], dict[str, Any]]:
    x0 = np.array([center.get(key, 0.0) for key in variables], dtype=np.float64)
    bounds = []
    for key in variables:
        shell = int(key[0])
        lim = 0.10 if shell <= 2 else 0.045 if shell == 3 else 0.018
        bounds.append((-lim, lim))
    cache: dict[tuple[float, ...], float] = {}

    def obj(x: np.ndarray) -> float:
        cache_key = tuple(np.round(x, 9))
        if cache_key in cache:
            return cache[cache_key]
        cls = dict(center)
        for key, val in zip(variables, x):
            cls[key] = float(val)
        cls = normalize(cls, radius, zero_keys)
        rec, _rows, _mom, _mat = eval_classes(name, family, cls, radius, direct_obs, evaluator, selected_rows, bins=46, momentum_grid=96)
        val = float(rec["objective_score"])
        cache[cache_key] = val
        return val

    result = minimize(
        obj,
        x0,
        method="Powell",
        bounds=bounds,
        options={"maxiter": maxiter, "xtol": 5.0e-5, "ftol": 5.0e-5, "disp": False},
    )
    cls = dict(center)
    for key, val in zip(variables, result.x):
        cls[key] = float(val)
    return normalize(cls, radius, zero_keys), {"success": bool(result.success), "fun": float(result.fun), "nfev": int(result.nfev), "nit": int(result.nit), "message": str(result.message)}


def load_start(path: Path, radius: int, zero_keys: set[str]) -> dict[str, float] | None:
    if not path.exists():
        return None
    matrix, _meta = eta_matrix_from_json(path, target_radius=radius)
    return normalize(classes_from_matrix(matrix / ETA_SCALE, radius), radius, zero_keys)


def split_rows(name: str, classes: dict[str, float], radius: int, direct: np.ndarray, fine: np.ndarray) -> list[dict[str, Any]]:
    matrix = ETA_SCALE * matrix_from_classes(classes, radius)
    rng = np.random.default_rng(2026071818)
    n_direct = direct.shape[0]
    n_fine = fine.shape[0]
    masks = {
        "full": (np.arange(n_direct), np.arange(n_fine)),
        "first_half": (np.arange(n_direct // 2), np.arange(n_fine // 2)),
        "second_half": (np.arange(n_direct // 2, n_direct), np.arange(n_fine // 2, n_fine)),
        "random_half": (rng.choice(n_direct, n_direct // 2, replace=False), rng.choice(n_fine, n_fine // 2, replace=False)),
    }
    out: list[dict[str, Any]] = []
    for split, (di, fi) in masks.items():
        rows = full_metrics(observable_arrays(direct[di]), observable_arrays(block(fine[fi], matrix)), bins=60)
        for obs in ["phi2", "phi4", "local_kurtosis_ratio", "action_density", "NN", "m2", "m4", "G_pmin_avg"]:
            out.append({"candidate": name, "split": split, "observable": obs, **rows[obs]})
    return out


def bootstrap_rows(name: str, direct_obs: dict[str, np.ndarray], blocked_obs: dict[str, np.ndarray], n_boot: int = 250) -> list[dict[str, Any]]:
    rng = np.random.default_rng(2026071819)
    out: list[dict[str, Any]] = []
    for obs in ["phi2", "phi4", "local_kurtosis_ratio", "action_density", "NN", "G_pmin_avg"]:
        a = direct_obs[obs]
        b = blocked_obs[obs]
        vals: dict[str, list[float]] = {k: [] for k in ["standardized_mean_shift", "std_ratio_a_over_b", "ks_statistic", "wasserstein_1"]}
        for _ in range(n_boot):
            aa = a[rng.integers(0, len(a), len(a))]
            bb = b[rng.integers(0, len(b), len(b))]
            m = metrics(aa, bb, bins=60)
            for key in vals:
                vals[key].append(float(m[key]))
        for key, samples in vals.items():
            q = np.quantile(samples, [0.16, 0.5, 0.84])
            out.append({"candidate": name, "observable": obs, "metric": key, "q16": q[0], "median": q[1], "q84": q[2], "n_boot": n_boot})
    return out


def write_kernel(name: str, family: str, classes: dict[str, float], radius: int, rec: dict[str, Any], rows: dict[str, dict[str, Any]], mom: dict[str, float]) -> Path:
    CAND_DIR.mkdir(parents=True, exist_ok=True)
    base = matrix_from_classes(classes, radius)
    matrix = ETA_SCALE * base
    payload = {
        "name": name,
        "type": "matrix",
        "matrix": matrix.tolist(),
        "base_matrix_before_eta_scale": base.tolist(),
        "base_orbit_classes_before_eta_scale": classes,
        "lambda": 1.0,
        "kappa_f": 0.340301,
        "kappa_c": 0.340301,
        "eta": 0.25,
        "block_factor": 2,
        "eta_scale": "2^0.125",
        "eta_scale_numeric": ETA_SCALE,
        "kernel_coefficients_include_eta_scale": True,
        "base_kernel_sum_before_eta_scale": float(base.sum()),
        "final_kernel_sum_after_eta_scale": float(matrix.sum()),
        "normalization": "eta_included_sum_to_eta_scale",
        "convention": "stored coefficients include eta_scale; do not multiply again on application",
        "family": family,
        "selection_metrics": rec,
        "momentum_stability": mom,
        "histogram_metrics": {obs: rows[obs] for obs in KEY_OBS},
    }
    path = CAND_DIR / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (CAND_DIR / f"{name}.txt").write_text("\n".join(" ".join(f"{v: .16e}" for v in row) for row in matrix) + "\n")
    return path


def candidate_pass_vector(row: dict[str, Any]) -> dict[str, bool]:
    return {
        "phi2_improved": float(row["phi2_KS"]) < 0.0708 and abs(float(row["phi2_shift"])) < 0.154,
        "kurtosis_improved": float(row["local_kurtosis_ratio_KS"]) < 0.2195 and abs(float(row["local_kurtosis_ratio_shift"])) < 0.566,
        "phi2_target": float(row["phi2_KS"]) < 0.060,
        "kurtosis_target": float(row["local_kurtosis_ratio_KS"]) < 0.190,
        "action_guard": float(row["action_density_KS"]) <= 0.040,
        "NN_guard": float(row["NN_KS"]) <= 0.035,
        "G_guard": float(row["G_pmin_avg_KS"]) <= 0.025,
        "phi4_guard": float(row["phi4_KS"]) <= 0.032,
        "conditioning_guard": float(row["min_K"]) > 0.0 and float(row["max_inverse_K"]) <= MAX_INV_HARD,
    }


def frontier_label(row: dict[str, Any], best_by: dict[str, str]) -> str:
    labels = [label for label, cand in best_by.items() if cand == row["candidate"]]
    return ";".join(labels)


def read_baseline_failure_table() -> list[dict[str, Any]]:
    if not BASELINE_FAILURE.exists():
        return []
    with BASELINE_FAILURE.open() as f:
        rows = list(csv.DictReader(f))
    keep = {"action_density", "phi2", "phi4", "local_kurtosis_ratio", "NN", "G_pmin_avg"}
    return [r for r in rows if r.get("observable") in keep]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    CAND_DIR.mkdir(parents=True, exist_ok=True)
    FINAL.mkdir(parents=True, exist_ok=True)
    (OUT / "plots").mkdir(parents=True, exist_ok=True)
    (FINAL / "plots").mkdir(parents=True, exist_ok=True)

    direct = load_configs(DIRECT)[:SEARCH_N_CONFIGS]
    fine = load_configs(FINE)[:SEARCH_N_CONFIGS]
    direct_obs = observable_arrays(direct)
    subset_direct = direct[:SUBSET_N]
    subset_fine = fine[:SUBSET_N]
    subset_direct_obs = observable_arrays(subset_direct)

    selected_matrix, _ = eta_matrix_from_json(SELECTED, target_radius=3)
    selected_classes = normalize(classes_from_matrix(selected_matrix / ETA_SCALE, 3), 3, set())
    selected_rows = full_metrics(direct_obs, observable_arrays(block(fine, selected_matrix)), bins=70)

    baseline_paths = [
        ("baseline_selected_upscaling_7x7", "baseline", SELECTED),
        ("baseline_current_final", "baseline", CURRENT_FINAL),
        ("baseline_retrained_5x5", "baseline", RETRAINED_5X5),
        ("baseline_nn_constrained_7x7", "baseline", NN_CONSTRAINED),
        ("baseline_phi2_nn_guarded_7x7", "baseline", PHI2_NN_GUARDED),
        ("previous_redo_best", "baseline", PREVIOUS_REDO),
    ]

    rng = np.random.default_rng(RNG_SEED)
    evaluators_subset = {r: BasisEvaluator(subset_fine, r) for r in [2, 3, 4]}
    evaluators_full = {r: BasisEvaluator(fine, r) for r in [2, 3, 4]}

    evaluated: list[tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, float], dict[str, float], int, np.ndarray]] = []
    starts_by_radius: dict[int, list[tuple[str, dict[str, float]]]] = {2: [], 3: [], 4: []}
    for name, family, path in baseline_paths:
        for radius in [2, 3, 4]:
            zero_keys = {"33"} if radius == 3 and "no33" in name else set()
            cls = load_start(path, radius, zero_keys)
            if cls is None:
                continue
            starts_by_radius[radius].append((name, cls))
        if path.exists():
            radius = 3
            cls = load_start(path, radius, set()) or selected_classes
            rec, rows, mom, mat = eval_classes(name, family, cls, radius, direct_obs, evaluators_full[radius], selected_rows, bins=70, momentum_grid=384)
            rec["category"] = "baseline"
            evaluated.append((rec, rows, mom, cls, radius, mat))

    families = [
        ("targeted_5x5", 2, [k for k in orbit_keys(2) if k != "00"], set(), 0.0035, 160, 4, 10),
        ("targeted_7x7_no33", 3, [k for k in orbit_keys(3) if k not in {"00", "33"}], {"33"}, 0.0024, 220, 5, 12),
        ("targeted_7x7_with33", 3, [k for k in orbit_keys(3) if k != "00"], set(), 0.0021, 180, 4, 10),
        ("targeted_9x9_stable", 4, [k for k in orbit_keys(4) if k != "00"], set(), 0.0008, 90, 2, 7),
    ]

    stage_rows: list[dict[str, Any]] = []
    screen: list[tuple[float, str, str, dict[str, float], int]] = []
    for family, radius, variables, zero_keys, scale, n_random, n_refine, maxiter in families:
        starts: list[tuple[str, dict[str, float]]] = []
        for start_name, cls in starts_by_radius[radius]:
            starts.append((start_name, normalize(cls, radius, zero_keys)))
        if radius == 3:
            starts.append(("selected_no33_projection", normalize(selected_classes, radius, zero_keys)))
        starts.append(("local_delta_center", normalize({}, radius, zero_keys)))

        family_screen: list[tuple[float, str, str, dict[str, float], int]] = []
        for start_idx, (start_name, center) in enumerate(starts):
            # Add the center itself and several targeted scales around it.
            rec, _rows, _mom, _mat = eval_classes(
                f"center_{family}_{start_idx:02d}_{start_name}",
                family,
                center,
                radius,
                subset_direct_obs,
                evaluators_subset[radius],
                selected_rows,
                bins=46,
                momentum_grid=96,
            )
            item = (float(rec["objective_score"]), rec["candidate"], family, center, radius)
            screen.append(item)
            family_screen.append(item)
            stage_rows.append({"stage": "center_subset", "start": start_name, **rec})
            for i in range(n_random):
                step_scale = scale * (0.45 if i < n_random // 3 else 1.0 if i < 2 * n_random // 3 else 1.8)
                cls = random_perturb(rng, center, radius, step_scale, variables, zero_keys)
                rec, _rows, _mom, _mat = eval_classes(
                    f"screen_{family}_{start_idx:02d}_{i:04d}",
                    family,
                    cls,
                    radius,
                    subset_direct_obs,
                    evaluators_subset[radius],
                    selected_rows,
                    bins=46,
                    momentum_grid=96,
                )
                item = (float(rec["objective_score"]), rec["candidate"], family, cls, radius)
                screen.append(item)
                family_screen.append(item)
                stage_rows.append({"stage": "random_subset", "start": start_name, **rec})

        family_screen.sort(key=lambda x: x[0])
        for j, (_score, _name, _fam, center, _rad) in enumerate(family_screen[:n_refine]):
            opt_cls, info = optimize_center(
                f"opt_{family}_{j:02d}",
                family,
                center,
                radius,
                variables,
                zero_keys,
                subset_direct_obs,
                evaluators_subset[radius],
                selected_rows,
                maxiter=maxiter,
            )
            rec, _rows, _mom, _mat = eval_classes(
                f"opt_{family}_{j:02d}",
                family,
                opt_cls,
                radius,
                subset_direct_obs,
                evaluators_subset[radius],
                selected_rows,
                bins=46,
                momentum_grid=128,
            )
            item = (float(rec["objective_score"]), rec["candidate"], family, opt_cls, radius)
            screen.append(item)
            stage_rows.append({"stage": "powell_subset", **rec, **{f"opt_{k}": v for k, v in info.items()}})

    screen.sort(key=lambda x: x[0])

    # Ensure that the full evaluation contains scalar-best, primary-target best,
    # and guardrail-preserving representatives.
    selected_specs: list[tuple[float, str, str, dict[str, float], int]] = []
    seen: set[str] = set()

    def add_spec(spec: tuple[float, str, str, dict[str, float], int]) -> None:
        if spec[1] not in seen:
            selected_specs.append(spec)
            seen.add(spec[1])

    for spec in screen[:36]:
        add_spec(spec)
    for key in ["phi2_KS", "local_kurtosis_ratio_KS", "NN_KS", "action_density_KS", "G_pmin_avg_KS"]:
        rows = [r for r in stage_rows if r.get("stage") in {"random_subset", "powell_subset", "center_subset"} and "candidate" in r]
        rows.sort(key=lambda r, k=key: float(r.get(k, 999.0)))
        wanted = {r["candidate"] for r in rows[:10]}
        for spec in screen:
            if spec[1] in wanted:
                add_spec(spec)
    for spec in screen:
        if len(selected_specs) >= 85:
            break
        add_spec(spec)

    for _score, name, family, cls, radius in selected_specs:
        rec, rows, mom, mat = eval_classes(name.replace("screen_", "full_").replace("center_", "full_center_").replace("opt_", "full_opt_"), family, cls, radius, direct_obs, evaluators_full[radius], selected_rows, bins=70, momentum_grid=384)
        evaluated.append((rec, rows, mom, cls, radius, mat))

    # Final categories are assigned against full metrics.
    candidate_rows = [x[0] for x in evaluated]
    for row in candidate_rows:
        for key, ok in candidate_pass_vector(row).items():
            row[key] = ok

    # Pareto/frontier representatives requested by the user.
    non_baseline = [row for row in candidate_rows if row["category"] != "baseline"]
    guard_ok = [row for row in non_baseline if row["conditioning_guard"]]
    primary_ok = [row for row in guard_ok if row["phi2_improved"] and row["kurtosis_improved"]]
    protected_ok = [row for row in primary_ok if row["action_guard"] and row["NN_guard"] and row["G_guard"]]
    best_by: dict[str, str] = {}
    pools = {
        "scalar_best": guard_ok,
        "phi2_best": guard_ok,
        "kurtosis_best": guard_ok,
        "phi2_kurtosis_best": primary_ok or guard_ok,
        "protected_best": protected_ok or primary_ok or guard_ok,
    }
    sorters = {
        "scalar_best": lambda r: float(r["objective_score"]),
        "phi2_best": lambda r: (float(r["phi2_KS"]), abs(float(r["phi2_shift"]))),
        "kurtosis_best": lambda r: (float(r["local_kurtosis_ratio_KS"]), abs(float(r["local_kurtosis_ratio_shift"]))),
        "phi2_kurtosis_best": lambda r: (float(r["phi2_KS"]) + float(r["local_kurtosis_ratio_KS"]), abs(float(r["phi2_shift"])) + abs(float(r["local_kurtosis_ratio_shift"]))),
        "protected_best": lambda r: (
            0 if r.get("action_guard") and r.get("NN_guard") and r.get("G_guard") else 1,
            float(r["phi2_KS"]) + float(r["local_kurtosis_ratio_KS"]) + float(r["action_density_KS"]) + float(r["NN_KS"]),
        ),
    }
    for label, pool in pools.items():
        if pool:
            best_by[label] = min(pool, key=sorters[label])["candidate"]
    for row in candidate_rows:
        row["frontier_label"] = frontier_label(row, best_by)

    candidate_rows.sort(key=lambda r: (r["category"] != "promotion-worthy", float(r["objective_score"])))

    hist_rows: list[dict[str, Any]] = []
    mom_rows: list[dict[str, Any]] = []
    sum_rows: list[dict[str, Any]] = []
    coeff_rows: list[dict[str, Any]] = []
    matrix_by_candidate: dict[str, np.ndarray] = {}
    classes_by_candidate: dict[str, tuple[dict[str, float], int]] = {}
    rows_by_candidate: dict[str, dict[str, dict[str, Any]]] = {}
    mom_by_candidate: dict[str, dict[str, float]] = {}
    for rec, rows, mom, cls, radius, mat in evaluated:
        matrix_by_candidate[rec["candidate"]] = mat
        classes_by_candidate[rec["candidate"]] = (cls, radius)
        rows_by_candidate[rec["candidate"]] = rows
        mom_by_candidate[rec["candidate"]] = mom
        for obs in OBS:
            hist_rows.append({"candidate": rec["candidate"], "family": rec["family"], "category": rec["category"], "frontier_label": rec.get("frontier_label", ""), "observable": obs, **rows[obs]})
        mom_rows.append({"candidate": rec["candidate"], "family": rec["family"], "category": rec["category"], "frontier_label": rec.get("frontier_label", ""), **mom, "condition_number": mom["max_K"] / mom["min_K"]})
        sum_rows.append({"candidate": rec["candidate"], "family": rec["family"], "radius": radius, "sum_base": float((mat / ETA_SCALE).sum()), "sum_K": float(mat.sum()), "eta_scale": ETA_SCALE, "kernel_coefficients_include_eta_scale": True})
        for key, val in sorted(cls.items()):
            coeff_rows.append({"candidate": rec["candidate"], "family": rec["family"], "class": key, "base_coefficient": val, "eta_included_coefficient": ETA_SCALE * val})

    write_csv(OUT / "baseline_failure_table_copied.csv", read_baseline_failure_table())
    write_csv(OUT / "stage_subset_screen.csv", stage_rows)
    write_csv(OUT / "candidate_scores.csv", candidate_rows)
    write_csv(OUT / "full_histogram_metrics.csv", hist_rows)
    write_csv(OUT / "histogram_distance_table.csv", hist_rows)
    write_csv(OUT / "momentum_stability.csv", mom_rows)
    write_csv(OUT / "candidate_kernel_sums.csv", sum_rows)
    write_csv(OUT / "orbit_coefficients_by_candidate.csv", coeff_rows)

    frontier_candidates = [cand for cand in dict.fromkeys(best_by.values())]
    write_csv(OUT / "pareto_frontier_candidates.csv", [row for row in candidate_rows if row["candidate"] in frontier_candidates])

    saved_paths: dict[str, str] = {}
    for cand in frontier_candidates:
        row = next(r for r in candidate_rows if r["candidate"] == cand)
        cls, radius = classes_by_candidate[cand]
        path = write_kernel(f"{cand}_eta_included", str(row["family"]), cls, radius, row, rows_by_candidate[cand], mom_by_candidate[cand])
        saved_paths[cand] = str(path)

    # Pick the scalar best among promotion-worthy if any; otherwise use protected
    # frontier representative. This is diagnostic only, not a promotion.
    promotion = [r for r in candidate_rows if r["category"] == "promotion-worthy"]
    if promotion:
        chosen = min(promotion, key=lambda r: float(r["objective_score"]))
    elif "protected_best" in best_by:
        chosen = next(r for r in candidate_rows if r["candidate"] == best_by["protected_best"])
    else:
        chosen = candidate_rows[0]

    chosen_name = str(chosen["candidate"])
    chosen_cls, chosen_radius = classes_by_candidate[chosen_name]
    chosen_rows = rows_by_candidate[chosen_name]
    chosen_mom = mom_by_candidate[chosen_name]
    chosen_path = write_kernel("best_pareto_redo_phi2_phi4_eta_included", str(chosen["family"]), chosen_cls, chosen_radius, chosen, chosen_rows, chosen_mom)
    chosen_mat = matrix_by_candidate[chosen_name]
    np.savetxt(FINAL / "best_eta_included_matrix.txt", chosen_mat, fmt="%.16e")
    np.savetxt(FINAL / "best_base_unit_sum_matrix.txt", chosen_mat / ETA_SCALE, fmt="%.16e")
    (CAND_DIR / "best_pareto_redo_phi2_phi4_eta_included.txt").write_text("\n".join(" ".join(f"{v: .16e}" for v in row) for row in chosen_mat) + "\n")

    blocked_chosen_obs = observable_arrays(evaluators_full[chosen_radius].blocked(chosen_cls))
    for obs in PLOT_OBS:
        plot_histogram(direct_obs[obs], blocked_chosen_obs[obs], obs, OUT / "plots" / f"best_pareto_redo_phi2_phi4_{obs}.pdf", bins=70, label_a="direct L16", label_b="candidate blocked L32->L16")
        plot_histogram(direct_obs[obs], blocked_chosen_obs[obs], obs, FINAL / "plots" / f"best_pareto_redo_phi2_phi4_{obs}.pdf", bins=70, label_a="direct L16", label_b="candidate blocked L32->L16")

    split: list[dict[str, Any]] = []
    split += split_rows("baseline_selected_upscaling_7x7", selected_classes, 3, direct, fine)
    for cand in frontier_candidates + [chosen_name]:
        cls, radius = classes_by_candidate[cand]
        split += split_rows(cand, cls, radius, direct, fine)
    write_csv(OUT / "split_robustness.csv", split)
    write_csv(FINAL / "split_robustness.csv", split)

    boot = bootstrap_rows(chosen_name, direct_obs, blocked_chosen_obs, n_boot=250)
    write_csv(OUT / "bootstrap_metrics.csv", boot)
    write_csv(FINAL / "bootstrap_metrics.csv", boot)
    write_csv(FINAL / "final_candidate_histogram_metrics.csv", [{"candidate": chosen_name, "family": chosen["family"], "category": chosen["category"], "observable": obs, **chosen_rows[obs]} for obs in OBS])
    write_csv(FINAL / "momentum_stability.csv", [{"candidate": chosen_name, **chosen_mom, "condition_number": chosen_mom["max_K"] / chosen_mom["min_K"]}])
    write_csv(FINAL / "candidate_kernel_sums.csv", [{"candidate": chosen_name, "sum_base": float((chosen_mat / ETA_SCALE).sum()), "sum_K": float(chosen_mat.sum()), "eta_scale": ETA_SCALE, "kernel_coefficients_include_eta_scale": True}])

    conflict_intrinsic = not any(r["category"] == "promotion-worthy" for r in candidate_rows)
    lines = [
        "# Lambda 1.0 Targeted 2000-Config Phi2/Phi4 Pareto Kernel Search",
        "",
        "No kernel has been promoted. Flow training remains paused.",
        "",
        f"Direct sample: `{DIRECT}` first `{SEARCH_N_CONFIGS}` configs.",
        f"Blocked sample: `{FINE}` first `{SEARCH_N_CONFIGS}` configs.",
        f"Current selected kernel reference: `{SELECTED}`.",
        f"Candidate directory: `{CAND_DIR}`.",
        "",
        "## Baseline Failure Table",
        "",
        "| observable | shift | KS | JS | TV | W1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in read_baseline_failure_table():
        lines.append(
            f"| {row['observable']} | {float(row['standardized_mean_shift']):.4f} | {float(row['ks_statistic']):.4f} | "
            f"{float(row['jensen_shannon']):.4f} | {float(row['total_variation']):.4f} | {float(row['wasserstein_1']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Search Families Tried",
            "",
            "- Full symmetric 5x5.",
            "- Full symmetric 7x7 no-corner.",
            "- Full symmetric 7x7 with corner.",
            "- 7x7 starts from the current selected kernel and retained 5x5/7x7 candidates, with all coefficients allowed to move.",
            "- Stable exploratory 9x9 local family.",
            "",
            f"Subset candidates evaluated: `{len(stage_rows)}`.",
            f"Full 2000-config candidates evaluated, including baselines: `{len(candidate_rows)}`.",
            "",
            "## Pareto Representatives",
            "",
            "| label | candidate | family | category | phi2 KS | phi2 shift | kurt KS | kurt shift | phi4 KS | action KS | NN KS | Gpmin KS | max 1/K |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for label, cand in best_by.items():
        row = next(r for r in candidate_rows if r["candidate"] == cand)
        lines.append(
            f"| {label} | {cand} | {row['family']} | {row['category']} | "
            f"{float(row['phi2_KS']):.4f} | {float(row['phi2_shift']):.4f} | "
            f"{float(row['local_kurtosis_ratio_KS']):.4f} | {float(row['local_kurtosis_ratio_shift']):.4f} | "
            f"{float(row['phi4_KS']):.4f} | {float(row['action_density_KS']):.4f} | "
            f"{float(row['NN_KS']):.4f} | {float(row['G_pmin_avg_KS']):.4f} | {float(row['max_inverse_K']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Diagnostic Best",
            "",
            f"Saved diagnostic best eta-included matrix: `{chosen_path}`.",
            f"Eta-included matrix text: `{CAND_DIR / 'best_pareto_redo_phi2_phi4_eta_included.txt'}`.",
            f"Base unit-sum matrix: `{FINAL / 'best_base_unit_sum_matrix.txt'}`.",
            f"Category: `{chosen['category']}`.",
            "",
            "## Recommendation",
            "",
        ]
    )
    if any(r["category"] == "promotion-worthy" for r in candidate_rows):
        lines.append("At least one candidate satisfies the promotion guardrails on this 2000-config search. Do not promote automatically; run a full confirmation before replacing the selected upscaling kernel.")
    else:
        lines.append("No promotion-worthy candidate was found. Keep the current selected kernel for now and treat the saved candidates as Pareto diagnostics.")
    if conflict_intrinsic:
        lines.append("Within this targeted 5x5/7x7/limited-9x9 pass, the conflict appears real: candidates that improve phi2/local-kurtosis enough tend to worsen action_density, NN, G_pmin, or fail the kurtosis target.")
    lines.append("")
    lines.append("## Output Files")
    lines.append("")
    for p in [
        OUT / "candidate_scores.csv",
        OUT / "pareto_frontier_candidates.csv",
        OUT / "full_histogram_metrics.csv",
        OUT / "momentum_stability.csv",
        OUT / "split_robustness.csv",
        OUT / "bootstrap_metrics.csv",
        FINAL / "final_candidate_histogram_metrics.csv",
    ]:
        lines.append(f"- `{p}`")
    report = "\n".join(lines) + "\n"
    (OUT / "recommendation.md").write_text(report)
    (OUT / "candidate_scores.md").write_text(report)
    (FINAL / "recommendation.md").write_text(report)

    print(json.dumps({"out": str(OUT), "final": str(FINAL), "candidate_dir": str(CAND_DIR), "chosen": str(chosen_path), "promotion_worthy": any(r["category"] == "promotion-worthy" for r in candidate_rows), "full_candidates": len(candidate_rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
