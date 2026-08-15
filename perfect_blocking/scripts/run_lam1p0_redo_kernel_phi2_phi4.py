#!/usr/bin/env python3
"""Redo lambda=1.0 kernel selection with phi2/phi4 coarse marginal matching."""

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
OUT = LAM_ROOT / "tests/intermediate/redo_kernel_phi2_phi4"
CAND_DIR = LAM_ROOT / "kernels/candidates/redo_phi2_phi4"
FINAL = LAM_ROOT / "tests/final/final_kernel_confirmation_direct_L16_vs_blocked_L32_phi2_phi4"

SELECTED = LAM_ROOT / "kernels/selected_for_upscaling/best_7x7_phi2_nn_guarded_eta_included.json"
CURRENT_FINAL = LAM_ROOT / "kernels/final/chosen_kernel.json"
RETRAINED_5X5 = LAM_ROOT / "kernels/selected_for_upscaling/best_5x5_retrained_full_objective_eta_included.json"
NN_CONSTRAINED = LAM_ROOT / "kernels/candidates/7x7_no33_nn_constrained/best_7x7_no33_nn_constrained_eta_included.json"
PHI2_NN_GUARDED = LAM_ROOT / "kernels/candidates/7x7_full_retraining_phi2_nn_guarded/best_7x7_full_retraining_controlled_eta_included.json"

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
PLOT_OBS = ["action_density", "phi2", "phi4", "local_kurtosis_ratio", "NN", "2nn", "diag", "m2", "m4", "G_pmin_avg"]
SEARCH_N_CONFIGS = 2000
SCREEN_RANDOM_PER_START = 45
REFINE_CENTERS_PER_FAMILY = 0
POWELL_MAXITER = 6
FULL_EVAL_TOP = 20
EXPLORATORY_9X9_RANDOM = 20


def orbit_keys(radius: int) -> list[str]:
    keys = []
    for a in range(radius + 1):
        for b in range(a + 1):
            keys.append(f"{a}{b}")
    return keys


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
    radius = target_radius if target_radius is not None else source_radius
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
    rest = sum(mult[k] * v for k, v in out.items() if k != "00")
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
        out = np.zeros((2 * target_radius + 1, 2 * target_radius + 1), dtype=np.float64)
        lo = target_radius - source_radius
        out[lo : lo + matrix.shape[0], lo : lo + matrix.shape[1]] = matrix
        matrix = out
    return matrix, meta


def scalar_score(row: dict[str, Any]) -> float:
    std_ratio = float(row["std_ratio_a_over_b"])
    log_width = 0.0 if not np.isfinite(std_ratio) or std_ratio <= 0 else abs(math.log(std_ratio))
    return (
        float(row["ks_statistic"])
        + 1.5 * float(row["total_variation"])
        + 15.0 * float(row["jensen_shannon"])
        + 0.35 * abs(float(row["standardized_mean_shift"]))
        + 0.20 * log_width
    )


def full_metrics(direct_obs: dict[str, np.ndarray], blocked_obs: dict[str, np.ndarray], bins: int = 70) -> dict[str, dict[str, Any]]:
    return {obs: metrics(direct_obs[obs], blocked_obs[obs], bins=bins) for obs in OBS}


def base_objective(rows: dict[str, dict[str, Any]]) -> float:
    weights = {
        "phi2": 10.0,
        "phi4": 10.0,
        "local_kurtosis_ratio": 12.0,
        "action_density": 4.0,
        "NN": 3.0,
        "2nn": 2.0,
        "diag": 2.0,
        "m2": 2.0,
        "m4": 2.0,
        "G_pmin_avg": 2.0,
    }
    return float(sum(weights.get(obs, 1.0) * scalar_score(rows[obs]) for obs in weights))


def guardrail_penalty(rows: dict[str, dict[str, Any]], mom: dict[str, float], classes: dict[str, float], radius: int, baseline: dict[str, dict[str, Any]]) -> float:
    limits = {
        "phi2": min(0.055, float(baseline["phi2"]["ks_statistic"])),
        "phi4": min(0.030, max(0.025, float(baseline["phi4"]["ks_statistic"]) + 0.008)),
        "local_kurtosis_ratio": min(0.180, float(baseline["local_kurtosis_ratio"]["ks_statistic"])),
        "action_density": max(0.038, float(baseline["action_density"]["ks_statistic"]) + 0.006),
        "NN": max(0.038, float(baseline["NN"]["ks_statistic"]) + 0.008),
        "2nn": 0.040,
        "diag": 0.040,
        "m2": 0.025,
        "m4": 0.025,
        "G_pmin_avg": 0.035,
    }
    penalty = 0.0
    for obs, limit in limits.items():
        excess = max(0.0, float(rows[obs]["ks_statistic"]) - limit)
        weight = 6.0e5 if obs in {"phi2", "phi4", "local_kurtosis_ratio"} else 1.2e5
        penalty += weight * excess * excess
    penalty += 1.0e5 * max(0.0, abs(float(rows["phi2"]["standardized_mean_shift"])) - 0.10) ** 2
    penalty += 1.0e5 * max(0.0, abs(float(rows["phi4"]["standardized_mean_shift"])) - 0.10) ** 2
    penalty += 1.0e5 * max(0.0, abs(float(rows["action_density"]["standardized_mean_shift"])) - 0.10) ** 2
    penalty += 2.0e6 * max(0.0, -float(mom["min_K"])) ** 2
    penalty += 2.0e5 * max(0.0, float(mom["max_inverse_K"]) - 1.6) ** 2
    mult = orbit_mult(radius)
    for key, value in classes.items():
        if key == "00":
            continue
        a = int(key[0])
        if a >= 3:
            penalty += 60.0 * mult[key] * value * value
        if a >= 4:
            penalty += 500.0 * mult[key] * value * value
    return float(penalty)


def total_score(rows: dict[str, dict[str, Any]], mom: dict[str, float], classes: dict[str, float], radius: int, baseline: dict[str, dict[str, Any]]) -> float:
    return base_objective(rows) + guardrail_penalty(rows, mom, classes, radius, baseline)


def candidate_category(rows: dict[str, dict[str, Any]], mom: dict[str, float], baseline: dict[str, dict[str, Any]]) -> str:
    if (
        float(rows["phi2"]["ks_statistic"]) < float(baseline["phi2"]["ks_statistic"])
        and float(rows["phi4"]["ks_statistic"]) <= max(0.030, float(baseline["phi4"]["ks_statistic"]) + 0.005)
        and float(rows["local_kurtosis_ratio"]["ks_statistic"]) < float(baseline["local_kurtosis_ratio"]["ks_statistic"])
        and float(rows["action_density"]["ks_statistic"]) <= max(0.040, float(baseline["action_density"]["ks_statistic"]) + 0.006)
        and float(rows["NN"]["ks_statistic"]) <= 0.040
        and float(rows["m2"]["ks_statistic"]) <= 0.025
        and float(rows["m4"]["ks_statistic"]) <= 0.025
        and float(rows["G_pmin_avg"]["ks_statistic"]) <= 0.035
        and float(mom["min_K"]) > 0
        and float(mom["max_inverse_K"]) <= 1.6
    ):
        return "promotion-worthy"
    if float(mom["min_K"]) > 0 and float(mom["max_inverse_K"]) <= 1.6:
        return "diagnostic-pareto"
    return "rejected-momentum"


def eval_classes(
    name: str,
    family: str,
    classes: dict[str, float],
    radius: int,
    direct_obs: dict[str, np.ndarray],
    fine: np.ndarray,
    baseline: dict[str, dict[str, Any]],
    momentum_grid: int,
    bins: int = 70,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, float], np.ndarray]:
    base_matrix = matrix_from_classes(classes, radius)
    matrix = ETA_SCALE * base_matrix
    blocked_obs = observable_arrays(block(fine, matrix))
    rows = full_metrics(direct_obs, blocked_obs, bins=bins)
    mom = momentum_extrema(matrix, grid=momentum_grid)
    rec: dict[str, Any] = {
        "candidate": name,
        "family": family,
        "radius": radius,
        "sum_base": float(base_matrix.sum()),
        "sum_K": float(matrix.sum()),
        "eta_scale": ETA_SCALE,
        "kernel_coefficients_include_eta_scale": True,
        "objective_score": total_score(rows, mom, classes, radius, baseline),
        "base_score": base_objective(rows),
        "category": candidate_category(rows, mom, baseline),
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


def random_perturb(rng: np.random.Generator, center: dict[str, float], radius: int, scale: float, variables: list[str], zero_keys: set[str]) -> dict[str, float]:
    cls = dict(center)
    for key in variables:
        cls[key] = float(cls.get(key, 0.0) + rng.normal(0.0, scale))
    return normalize(cls, radius, zero_keys)


def optimize_center(
    center: dict[str, float],
    radius: int,
    variables: list[str],
    zero_keys: set[str],
    direct_obs: dict[str, np.ndarray],
    fine: np.ndarray,
    baseline: dict[str, dict[str, Any]],
    maxiter: int,
) -> tuple[dict[str, float], dict[str, Any]]:
    x0 = np.array([center.get(k, 0.0) for k in variables], dtype=np.float64)
    bounds = [(-0.08, 0.12) for _ in variables]
    cache: dict[tuple[float, ...], float] = {}

    def obj(x: np.ndarray) -> float:
        key = tuple(np.round(x, 10))
        if key in cache:
            return cache[key]
        cls = dict(center)
        for name, value in zip(variables, x):
            cls[name] = float(value)
        cls = normalize(cls, radius, zero_keys)
        rec, _rows, _mom, _matrix = eval_classes("opt", "opt", cls, radius, direct_obs, fine, baseline, momentum_grid=96, bins=50)
        val = float(rec["objective_score"])
        cache[key] = val
        return val

    result = minimize(
        obj,
        x0,
        method="Powell",
        bounds=bounds,
        options={"maxiter": maxiter, "xtol": 7.0e-5, "ftol": 7.0e-5, "disp": False},
    )
    cls = dict(center)
    for name, value in zip(variables, result.x):
        cls[name] = float(value)
    return normalize(cls, radius, zero_keys), {"success": bool(result.success), "fun": float(result.fun), "nfev": int(result.nfev), "nit": int(result.nit), "message": str(result.message)}


def split_rows(name: str, classes: dict[str, float], radius: int, direct: np.ndarray, fine: np.ndarray) -> list[dict[str, Any]]:
    base = matrix_from_classes(classes, radius)
    matrix = ETA_SCALE * base
    rng = np.random.default_rng(2026071803)
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


def bootstrap_rows(name: str, rows: dict[str, dict[str, Any]], direct_obs: dict[str, np.ndarray], blocked_obs: dict[str, np.ndarray], n_boot: int = 250) -> list[dict[str, Any]]:
    rng = np.random.default_rng(2026071804)
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
            out.append({"candidate": name, "observable": obs, "metric": key, "q16": q[0], "median": q[1], "q84": q[2], "point": rows[obs][key], "n_boot": n_boot})
    return out


def write_kernel(name: str, family: str, classes: dict[str, float], radius: int, rec: dict[str, Any], rows: dict[str, dict[str, Any]], mom: dict[str, float]) -> Path:
    base = matrix_from_classes(classes, radius)
    matrix = ETA_SCALE * base
    path = CAND_DIR / f"{name}.json"
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
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    txt = CAND_DIR / f"{name}.txt"
    txt.write_text("\n".join(" ".join(f"{v: .16e}" for v in row) for row in matrix) + "\n")
    return path


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    FINAL.mkdir(parents=True, exist_ok=True)
    CAND_DIR.mkdir(parents=True, exist_ok=True)
    (OUT / "plots").mkdir(parents=True, exist_ok=True)
    (FINAL / "plots").mkdir(parents=True, exist_ok=True)

    direct = load_configs(DIRECT)[:SEARCH_N_CONFIGS]
    fine = load_configs(FINE)[:SEARCH_N_CONFIGS]
    direct_obs = observable_arrays(direct)
    subset_n = min(1200, SEARCH_N_CONFIGS)
    subset_direct = direct[:subset_n]
    subset_fine = fine[:subset_n]
    subset_direct_obs = observable_arrays(subset_direct)

    selected_matrix, _ = eta_matrix_from_json(SELECTED, target_radius=3)
    selected_rows = full_metrics(direct_obs, observable_arrays(block(fine, selected_matrix)), bins=70)

    baseline_paths = [
        ("baseline_selected_upscaling_7x7", "baseline", SELECTED),
        ("baseline_current_final_7x7", "baseline", CURRENT_FINAL),
        ("baseline_retrained_5x5", "baseline", RETRAINED_5X5),
        ("baseline_nn_constrained_7x7", "baseline", NN_CONSTRAINED),
        ("baseline_phi2_nn_guarded_7x7", "baseline", PHI2_NN_GUARDED),
    ]
    starts: list[tuple[str, int, dict[str, float]]] = []
    evaluated: list[tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, float], dict[str, float], int, np.ndarray]] = []
    for name, family, path in baseline_paths:
        if not path.exists():
            continue
        radius = 3
        matrix, _meta = eta_matrix_from_json(path, target_radius=radius)
        cls = normalize(classes_from_matrix(matrix / ETA_SCALE, target_radius=radius), radius, set())
        rec, rows, mom, mat = eval_classes(name, family, cls, radius, direct_obs, fine, selected_rows, momentum_grid=512)
        rec["category"] = "baseline"
        evaluated.append((rec, rows, mom, cls, radius, mat))
        starts.append((name, radius, cls))

    rng = np.random.default_rng(2026071805)
    screen: list[tuple[float, str, str, dict[str, float], int]] = []

    families = [
        ("full_5x5", 2, [k for k in orbit_keys(2) if k != "00"], set()),
        ("full_7x7_no33", 3, [k for k in orbit_keys(3) if k not in {"00", "33"}], {"33"}),
        ("full_7x7_with33", 3, [k for k in orbit_keys(3) if k != "00"], set()),
    ]
    stage_rows: list[dict[str, Any]] = []
    for family, radius, variables, zero_keys in families:
        family_starts: list[dict[str, float]] = []
        for _sname, _sr, scls in starts:
            family_starts.append(normalize(classes_from_matrix(matrix_from_classes(scls, _sr), target_radius=radius), radius, zero_keys))
        family_starts.append(normalize({}, radius, zero_keys))
        for start_idx, center in enumerate(family_starts):
            for i in range(SCREEN_RANDOM_PER_START):
                scale = 0.0015 if start_idx < len(family_starts) - 1 else 0.008
                cls = random_perturb(rng, center, radius, scale, variables, zero_keys)
                rec, _rows, mom, _mat = eval_classes(f"screen_{family}_{start_idx}_{i:04d}", family, cls, radius, subset_direct_obs, subset_fine, selected_rows, momentum_grid=96, bins=50)
                screen.append((float(rec["objective_score"]), rec["candidate"], family, cls, radius))
                stage_rows.append({"stage": "random_screen", **rec})
        screen.sort(key=lambda x: x[0])
        for center_idx, (_score, _name, _family, center, _radius) in enumerate([x for x in screen if x[2] == family][:REFINE_CENTERS_PER_FAMILY]):
            opt_cls, info = optimize_center(center, radius, variables, zero_keys, subset_direct_obs, subset_fine, selected_rows, maxiter=POWELL_MAXITER)
            rec, _rows, _mom, _mat = eval_classes(f"opt_{family}_{center_idx:02d}", family, opt_cls, radius, subset_direct_obs, subset_fine, selected_rows, momentum_grid=128, bins=50)
            screen.append((float(rec["objective_score"]), rec["candidate"], family, opt_cls, radius))
            stage_rows.append({"stage": "powell_subset", **rec, **{f"opt_{k}": v for k, v in info.items()}})

    # Optional small 9x9 shell search around the best 7x7 if the 7x7 screen has not
    # already found a clearly strong phi2/kurtosis improvement.
    best7 = min((x for x in screen if x[4] == 3), key=lambda x: x[0])
    radius = 4
    zero_keys = set()
    variables = [k for k in orbit_keys(4) if k != "00"]
    center9 = normalize(classes_from_matrix(matrix_from_classes(best7[3], 3), target_radius=4), radius, zero_keys)
    for i in range(EXPLORATORY_9X9_RANDOM):
        cls = random_perturb(rng, center9, radius, 0.00075, variables, zero_keys)
        rec, _rows, _mom, _mat = eval_classes(f"screen_9x9_exploratory_{i:04d}", "9x9_exploratory", cls, radius, subset_direct_obs, subset_fine, selected_rows, momentum_grid=96, bins=50)
        screen.append((float(rec["objective_score"]), rec["candidate"], "9x9_exploratory", cls, radius))
        stage_rows.append({"stage": "optional_9x9_screen", **rec})

    screen.sort(key=lambda x: x[0])
    full_specs = screen[:FULL_EVAL_TOP]
    for _score, name, family, cls, radius in full_specs:
        rec, rows, mom, mat = eval_classes(name.replace("screen_", "full_"), family, cls, radius, direct_obs, fine, selected_rows, momentum_grid=384, bins=70)
        evaluated.append((rec, rows, mom, cls, radius, mat))

    candidate_rows = [x[0] for x in evaluated]
    candidate_rows.sort(key=lambda r: (r["category"] != "promotion-worthy", r["objective_score"]))

    hist_rows: list[dict[str, Any]] = []
    mom_rows: list[dict[str, Any]] = []
    sum_rows: list[dict[str, Any]] = []
    matrix_rows: list[dict[str, Any]] = []
    for rec, rows, mom, cls, radius, mat in evaluated:
        for obs in OBS:
            hist_rows.append({"candidate": rec["candidate"], "family": rec["family"], "category": rec["category"], "observable": obs, **rows[obs]})
        mom_rows.append({"candidate": rec["candidate"], "family": rec["family"], "category": rec["category"], **mom, "condition_number": mom["max_K"] / mom["min_K"]})
        sum_rows.append({"candidate": rec["candidate"], "family": rec["family"], "radius": radius, "sum_base": float((mat / ETA_SCALE).sum()), "sum_K": float(mat.sum()), "eta_scale": ETA_SCALE, "kernel_coefficients_include_eta_scale": True})
        for key, value in sorted(cls.items()):
            matrix_rows.append({"candidate": rec["candidate"], "family": rec["family"], "class": key, "base_coefficient": value, "eta_included_coefficient": ETA_SCALE * value})

    write_csv(OUT / "stage_subset_screen.csv", stage_rows)
    write_csv(OUT / "candidate_scores.csv", candidate_rows)
    write_csv(OUT / "candidate_scores.md.csv", candidate_rows)
    write_csv(OUT / "full_histogram_metrics.csv", hist_rows)
    write_csv(OUT / "histogram_distance_table.csv", hist_rows)
    write_csv(OUT / "momentum_stability.csv", mom_rows)
    write_csv(OUT / "candidate_kernel_sums.csv", sum_rows)
    write_csv(OUT / "orbit_coefficients_by_candidate.csv", matrix_rows)

    best_tuple = next((x for x in sorted(evaluated, key=lambda y: y[0]["objective_score"]) if x[0]["category"] == "promotion-worthy"), None)
    if best_tuple is None:
        best_tuple = min(evaluated, key=lambda x: x[0]["objective_score"])
    best_rec, best_rows, best_mom, best_cls, best_radius, best_mat = best_tuple
    best_name = "best_redo_phi2_phi4_eta_included"
    best_path = write_kernel(best_name, str(best_rec["family"]), best_cls, best_radius, best_rec, best_rows, best_mom)

    blocked_best_obs = observable_arrays(block(fine, best_mat))
    for obs in PLOT_OBS:
        plot_histogram(direct_obs[obs], blocked_best_obs[obs], obs, OUT / "plots" / f"{best_name}_{obs}.pdf", bins=70, label_a="direct L16", label_b="best redo blocked L32->L16")
        plot_histogram(direct_obs[obs], blocked_best_obs[obs], obs, FINAL / "plots" / f"{best_name}_{obs}.pdf", bins=70, label_a="direct L16", label_b="best redo blocked L32->L16")

    split = split_rows(best_name, best_cls, best_radius, direct, fine)
    split += split_rows("baseline_selected_upscaling_7x7", normalize(classes_from_matrix(selected_matrix / ETA_SCALE, 3), 3, set()), 3, direct, fine)
    write_csv(OUT / "split_robustness.csv", split)
    write_csv(FINAL / "split_robustness.csv", split)
    boot = bootstrap_rows(best_name, best_rows, direct_obs, blocked_best_obs, n_boot=250)
    write_csv(OUT / "bootstrap_metrics.csv", boot)
    write_csv(FINAL / "bootstrap_metrics.csv", boot)

    final_hist = [{"candidate": best_name, "family": best_rec["family"], "category": best_rec["category"], "observable": obs, **best_rows[obs]} for obs in OBS]
    write_csv(FINAL / "final_candidate_histogram_metrics.csv", final_hist)
    write_csv(FINAL / "momentum_stability.csv", [{"candidate": best_name, **best_mom, "condition_number": best_mom["max_K"] / best_mom["min_K"]}])
    write_csv(FINAL / "candidate_kernel_sums.csv", [{"candidate": best_name, "sum_base": float((best_mat / ETA_SCALE).sum()), "sum_K": float(best_mat.sum()), "eta_scale": ETA_SCALE, "kernel_coefficients_include_eta_scale": True}])

    np.savetxt(FINAL / "best_eta_included_matrix.txt", best_mat, fmt="%.16e")
    np.savetxt(FINAL / "best_base_unit_sum_matrix.txt", best_mat / ETA_SCALE, fmt="%.16e")

    lines = [
        "# Lambda 1.0 Redo Kernel Search With Phi2/Phi4 Hard Targets",
        "",
        "Flow/upscaling training is paused because the lambda=1.0 coarse marginal still has a phi2/local-kurtosis mismatch between direct native L16 and native L32 blocked to L16.",
        "",
        f"Selected baseline kernel: `{SELECTED}`",
        f"Initial search sample size: `{SEARCH_N_CONFIGS}` direct L16 configs and `{SEARCH_N_CONFIGS}` native L32 configs blocked to L16.",
        f"Search breadth: `{SCREEN_RANDOM_PER_START}` random candidates per start, `{REFINE_CENTERS_PER_FAMILY}` Powell centers per family, maxiter `{POWELL_MAXITER}`, full evaluation top `{FULL_EVAL_TOP}`.",
        f"Best candidate saved: `{best_path}`",
        f"Best candidate category: `{best_rec['category']}`",
        "",
        "## Top candidates",
        "",
        "| candidate | family | category | score | action KS | phi2 KS | phi4 KS | kurtosis KS | NN KS | m2 KS | m4 KS | Gpmin KS | max 1/K |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in candidate_rows[:25]:
        lines.append(
            f"| {row['candidate']} | {row['family']} | {row['category']} | {float(row['objective_score']):.6g} | "
            f"{float(row['action_density_KS']):.5f} | {float(row['phi2_KS']):.5f} | {float(row['phi4_KS']):.5f} | "
            f"{float(row['local_kurtosis_ratio_KS']):.5f} | {float(row['NN_KS']):.5f} | {float(row['m2_KS']):.5f} | "
            f"{float(row['m4_KS']):.5f} | {float(row['G_pmin_avg_KS']):.5f} | {float(row['max_inverse_K']):.5f} |"
        )
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            "Do not promote automatically. Promote only after reviewing the final confirmation metrics, split robustness, and plots. If no promotion-worthy candidate appears, retain the current selected kernel and continue a broader Pareto search.",
            "",
            "## Documentation note",
            "",
            "The previous phi2-priority NN-guarded 7x7 was good relative to earlier tests, but direct native L16 vs blocked native L32->L16 still showed a phi2/local-kurtosis mismatch. This invalidated the flow-training starting point, so lambda=1.0 kernel training was reopened with phi2/phi4 coarse-marginal matching as hard targets.",
        ]
    )
    report = "\n".join(lines) + "\n"
    (OUT / "recommendation.md").write_text(report)
    (OUT / "candidate_scores.md").write_text(report)
    (FINAL / "recommendation.md").write_text(report)

    print(json.dumps({"out": str(OUT), "final": str(FINAL), "best_candidate": str(best_path), "category": best_rec["category"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
