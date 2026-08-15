#!/usr/bin/env python3
"""Support-balanced lambda=1.0 kernel search.

This pass deliberately avoids over-optimizing phi4 against phi2.  It ranks
candidates mainly by distribution shape, phi2 upper-tail coverage, blocked-width
coverage, and local-kurtosis shape while keeping action density and NN protected.
No candidate is promoted by this script.
"""

from __future__ import annotations

import argparse
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
from scripts.run_lam1p0_redo_kernel_phi2_phi4_pareto_2000 import (  # noqa: E402
    BasisEvaluator,
    classes_from_matrix,
    eta_matrix_from_json,
    matrix_from_classes,
    normalize,
    orbit_keys,
    orbit_mult,
    scalar_score,
)


LAM_ROOT = PROJECT_ROOT / "perfect_blocking/perfect_blocking_lam1p0"
DIRECT = PROJECT_ROOT / "data/configs_phi4_2d/lam1p0_kappac0p340301_L16/configs.npz"
FINE = PROJECT_ROOT / "data/configs_phi4_2d/lam1p0_kappac0p340301_L32/configs.npz"
OUT = LAM_ROOT / "tests/intermediate/redo_kernel_phi2_support_balanced_2000"
CAND_DIR = LAM_ROOT / "kernels/candidates/redo_phi2_support_balanced_2000"
FINAL = LAM_ROOT / "tests/final/final_kernel_confirmation_direct_L16_vs_blocked_L32_phi2_support_balanced_5000"

SELECTED = LAM_ROOT / "kernels/selected_for_upscaling/best_7x7_phi2_nn_guarded_eta_included.json"
PREV_REDO = LAM_ROOT / "kernels/candidates/redo_phi2_phi4/best_redo_phi2_phi4_eta_included.json"
PARETO_5X5 = LAM_ROOT / "kernels/candidates/redo_phi2_phi4_pareto_2000/best_pareto_redo_phi2_phi4_eta_included.json"
PARETO_7X7 = LAM_ROOT / "kernels/candidates/redo_phi2_phi4_pareto_2000/full_opt_targeted_7x7_no33_04_eta_included.json"
PHI2_BEST = LAM_ROOT / "kernels/candidates/redo_phi2_phi4_pareto_2000/full_targeted_7x7_no33_01_0149_eta_included.json"
RETRAINED_5X5 = LAM_ROOT / "kernels/selected_for_upscaling/best_5x5_retrained_full_objective_eta_included.json"

OBS = ["action_density", "phi2", "phi4", "local_kurtosis_ratio", "NN", "2nn", "diag", "m", "m2", "m4", "G_00", "G_10", "G_01", "G_pmin_avg"]
KEY_OBS = ["action_density", "phi2", "phi4", "local_kurtosis_ratio", "NN", "2nn", "diag", "m2", "m4", "G_pmin_avg"]
PLOT_OBS = ["phi2", "phi4", "local_kurtosis_ratio", "action_density", "NN", "G_pmin_avg"]
SEARCH_N = 2000
CONFIRM_N = 5000
SUBSET_N = 1000
PHI2_MEAN_MATCH_WEIGHT = 0.0
PHI2_LOWER_TAIL_MATCH_WEIGHT = 0.0
KURTOSIS_UPPER_TAIL_MATCH_WEIGHT = 0.0


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


def full_metrics(direct_obs: dict[str, np.ndarray], blocked_obs: dict[str, np.ndarray], bins: int = 80) -> dict[str, dict[str, Any]]:
    return {obs: metrics(direct_obs[obs], blocked_obs[obs], bins=bins) for obs in OBS}


def tail_stats(direct_obs: dict[str, np.ndarray], blocked_obs: dict[str, np.ndarray], obs: str) -> dict[str, float]:
    a = direct_obs[obs]
    b = blocked_obs[obs]
    q = {p: float(np.quantile(a, p)) for p in [0.01, 0.05, 0.10, 0.90, 0.95, 0.99]}
    return {
        "below_q01": float(np.mean(b < q[0.01])),
        "below_q05": float(np.mean(b < q[0.05])),
        "below_q10": float(np.mean(b < q[0.10])),
        "above_q90": float(np.mean(b > q[0.90])),
        "above_q95": float(np.mean(b > q[0.95])),
        "above_q99": float(np.mean(b > q[0.99])),
        "inside_q05_q95": float(np.mean((b >= q[0.05]) & (b <= q[0.95]))),
        "inside_q01_q99": float(np.mean((b >= q[0.01]) & (b <= q[0.99]))),
    }


def support_penalty(rows: dict[str, dict[str, Any]], tails: dict[str, dict[str, float]]) -> float:
    val = 0.0
    # We prefer not to miss regions.  For phi2, undercoverage of upper tail is
    # worse than overcoverage, and a slightly wide blocked distribution is better
    # than a narrow one.
    phi2 = rows["phi2"]
    phi2_std_ratio = float(phi2["std_ratio_a_over_b"])  # direct std / blocked std
    val += 450.0 * max(0.0, phi2_std_ratio - 1.02) ** 2
    val += 80.0 * max(0.0, 0.88 - phi2_std_ratio) ** 2
    val += 1400.0 * max(0.0, 0.050 - tails["phi2"]["above_q95"]) ** 2
    val += 2200.0 * max(0.0, 0.010 - tails["phi2"]["above_q99"]) ** 2
    # Overcoverage is allowed but not unlimited.
    val += 180.0 * max(0.0, tails["phi2"]["above_q95"] - 0.085) ** 2
    val += 260.0 * max(0.0, tails["phi2"]["above_q99"] - 0.030) ** 2
    # Action density must keep both tails present.  Do not force exact equality,
    # but reject obvious support gaps.
    val += 900.0 * max(0.0, 0.025 - tails["action_density"]["below_q05"]) ** 2
    val += 900.0 * max(0.0, 0.025 - tails["action_density"]["above_q95"]) ** 2
    val += 300.0 * max(0.0, tails["action_density"]["below_q05"] - 0.075) ** 2
    val += 300.0 * max(0.0, tails["action_density"]["above_q95"] - 0.075) ** 2
    # Phi4 is protected, but less aggressively than phi2/local-kurtosis.
    val += 220.0 * max(0.0, float(rows["phi4"]["ks_statistic"]) - 0.055) ** 2
    val += 180.0 * max(0.0, tails["phi4"]["above_q95"] - 0.085) ** 2
    return float(val)


def shape_score(rows: dict[str, dict[str, Any]], tails: dict[str, dict[str, float]], mom: dict[str, float], classes: dict[str, float], radius: int) -> float:
    def s(obs: str) -> float:
        r = rows[obs]
        std_ratio = float(r["std_ratio_a_over_b"])
        width = 0.0 if std_ratio <= 0 or not np.isfinite(std_ratio) else abs(math.log(std_ratio))
        return float(r["ks_statistic"]) + 1.15 * float(r["total_variation"]) + 9.0 * float(r["jensen_shannon"]) + 0.60 * width

    val = (
        28.0 * s("phi2")
        + 28.0 * s("local_kurtosis_ratio")
        + 7.0 * s("phi4")
        + 15.0 * s("action_density")
        + 10.0 * s("NN")
        + 4.0 * s("2nn")
        + 4.0 * s("diag")
        # The only global-magnetization moment in the fit is the correctly
        # normalized m2 = (V^{-1} sum_x phi_x)^2.  m4 and all G observables
        # are retained in the output tables and promotion validation only.
        + 3.0 * s("m2")
        + support_penalty(rows, tails)
    )
    # Optional targeted refinement: the baseline support-balanced objective
    # deliberately ignores mean shifts.  These terms are off by default and
    # can be enabled only for a measured phi2 mean/tail mismatch.
    val += PHI2_MEAN_MATCH_WEIGHT * float(rows["phi2"]["standardized_mean_shift"]) ** 2
    val += PHI2_LOWER_TAIL_MATCH_WEIGHT * (float(tails["phi2"]["below_q05"]) - 0.05) ** 2
    val += KURTOSIS_UPPER_TAIL_MATCH_WEIGHT * (
        (float(tails["local_kurtosis_ratio"]["above_q90"]) - 0.10) ** 2
        + (float(tails["local_kurtosis_ratio"]["above_q95"]) - 0.05) ** 2
    )
    # Soft guardrails.  Action is important, but a slightly wider distribution is
    # not treated as worse than a hard support gap.
    limits = {
        "action_density": 0.050,
        "NN": 0.040,
        "G_pmin_avg": 0.030,
        "phi2": 0.065,
        "local_kurtosis_ratio": 0.170,
        "phi4": 0.060,
    }
    for obs, lim in limits.items():
        val += 1.2e5 * max(0.0, float(rows[obs]["ks_statistic"]) - lim) ** 2
    val += 4.0e6 * max(0.0, -float(mom["min_K"])) ** 2
    val += 3.5e5 * max(0.0, float(mom["max_inverse_K"]) - 1.60) ** 2
    val += 4.0e4 * max(0.0, float(mom["max_inverse_K"]) - 1.48) ** 2
    mult = orbit_mult(radius)
    for key, value in classes.items():
        if key == "00":
            continue
        shell = int(key[0])
        if shell >= 3:
            val += 120.0 * mult[key] * value * value
        if shell >= 4:
            val += 1800.0 * mult[key] * value * value
    return float(val)


def load_start(path: Path, radius: int, zero_keys: set[str]) -> dict[str, float] | None:
    if not path.exists():
        return None
    matrix, _ = eta_matrix_from_json(path, target_radius=radius)
    return normalize(classes_from_matrix(matrix / ETA_SCALE, radius), radius, zero_keys)


def eval_candidate(
    name: str,
    family: str,
    classes: dict[str, float],
    radius: int,
    direct_obs: dict[str, np.ndarray],
    evaluator: BasisEvaluator,
    bins: int,
    momentum_grid: int,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, dict[str, float]], dict[str, float], np.ndarray]:
    blocked_obs = observable_arrays(evaluator.blocked(classes))
    rows = full_metrics(direct_obs, blocked_obs, bins=bins)
    tails = {obs: tail_stats(direct_obs, blocked_obs, obs) for obs in ["action_density", "phi2", "phi4", "local_kurtosis_ratio", "NN", "G_pmin_avg"]}
    matrix = ETA_SCALE * matrix_from_classes(classes, radius)
    mom = momentum_extrema(matrix, grid=momentum_grid)
    rec: dict[str, Any] = {
        "candidate": name,
        "family": family,
        "radius": radius,
        "objective_score": shape_score(rows, tails, mom, classes, radius),
        "sum_base": float(np.sum(matrix / ETA_SCALE)),
        "sum_K": float(np.sum(matrix)),
        "eta_scale": ETA_SCALE,
        "kernel_coefficients_include_eta_scale": True,
        "min_K": mom["min_K"],
        "max_K": mom["max_K"],
        "min_inverse_K": mom["min_inverse_K"],
        "max_inverse_K": mom["max_inverse_K"],
        "condition_number": mom["max_K"] / mom["min_K"],
    }
    for obs in KEY_OBS:
        rec[f"{obs}_KS"] = rows[obs]["ks_statistic"]
        rec[f"{obs}_TV"] = rows[obs]["total_variation"]
        rec[f"{obs}_JS"] = rows[obs]["jensen_shannon"]
        rec[f"{obs}_W1"] = rows[obs]["wasserstein_1"]
        rec[f"{obs}_std_ratio"] = rows[obs]["std_ratio_a_over_b"]
        rec[f"{obs}_shift"] = rows[obs]["standardized_mean_shift"]
    for obs, t in tails.items():
        for k, v in t.items():
            rec[f"{obs}_{k}"] = v
    rec["promotion_worthy"] = bool(
        rec["phi2_KS"] <= 0.065
        and rec["local_kurtosis_ratio_KS"] <= 0.170
        and rec["action_density_KS"] <= 0.050
        and rec["NN_KS"] <= 0.040
        and rec["G_pmin_avg_KS"] <= 0.030
        and rec["phi4_KS"] <= 0.060
        and rec["phi2_above_q95"] >= 0.045
        and rec["phi2_above_q99"] >= 0.008
        and rec["action_density_below_q05"] >= 0.025
        and rec["action_density_above_q95"] >= 0.025
        and mom["min_K"] > 0
        and mom["max_inverse_K"] <= 1.60
    )
    return rec, rows, tails, mom, matrix


def random_perturb(rng: np.random.Generator, center: dict[str, float], radius: int, variables: list[str], zero_keys: set[str], scale: float) -> dict[str, float]:
    cls = dict(center)
    for key in variables:
        shell = int(key[0])
        cls[key] = float(cls.get(key, 0.0) + rng.normal(0.0, scale / (1.0 + 0.5 * max(0, shell - 2))))
    return normalize(cls, radius, zero_keys)


def optimize_center(
    center: dict[str, float],
    radius: int,
    variables: list[str],
    zero_keys: set[str],
    direct_obs: dict[str, np.ndarray],
    evaluator: BasisEvaluator,
    maxiter: int,
) -> tuple[dict[str, float], dict[str, Any]]:
    x0 = np.asarray([center.get(k, 0.0) for k in variables], dtype=np.float64)
    bounds = []
    for key in variables:
        shell = int(key[0])
        lim = 0.11 if shell <= 2 else 0.050 if shell == 3 else 0.020
        bounds.append((-lim, lim))
    cache: dict[tuple[float, ...], float] = {}

    def obj(x: np.ndarray) -> float:
        key = tuple(np.round(x, 9))
        if key in cache:
            return cache[key]
        cls = dict(center)
        for k, v in zip(variables, x):
            cls[k] = float(v)
        cls = normalize(cls, radius, zero_keys)
        rec, *_ = eval_candidate("opt", "opt", cls, radius, direct_obs, evaluator, bins=46, momentum_grid=96)
        cache[key] = float(rec["objective_score"])
        return cache[key]

    result = minimize(obj, x0, method="Powell", bounds=bounds, options={"maxiter": maxiter, "xtol": 5e-5, "ftol": 5e-5, "disp": False})
    cls = dict(center)
    for k, v in zip(variables, result.x):
        cls[k] = float(v)
    return normalize(cls, radius, zero_keys), {"success": bool(result.success), "nfev": int(result.nfev), "nit": int(result.nit), "fun": float(result.fun), "message": str(result.message)}


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
        "base_kernel_sum_before_eta_scale": float(np.sum(base)),
        "final_kernel_sum_after_eta_scale": float(np.sum(matrix)),
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-n", type=int, default=SEARCH_N, help="Number of paired direct-L16 and blocked-L32 configurations used for optimization.")
    parser.add_argument("--confirm-n", type=int, default=CONFIRM_N, help="Configurations for the post-search diagnostic confirmation (0 disables it).")
    parser.add_argument("--only-5x5", action="store_true", help="Optimize only the D4-symmetric 5x5 (radius-2) family.")
    parser.add_argument("--start-kernel", type=Path, default=None, help="Use this kernel as the sole optimization start.")
    parser.add_argument("--out", type=Path, default=None, help="Directory for search diagnostics.")
    parser.add_argument("--candidate-dir", type=Path, default=None, help="Directory for newly written kernel candidates.")
    parser.add_argument("--final-out", type=Path, default=None, help="Directory for confirmation diagnostics.")
    parser.add_argument("--random-per-start", type=int, default=None, help="Override the random screen size per start.")
    parser.add_argument("--refine-count", type=int, default=None, help="Override the number of Powell refinements.")
    parser.add_argument("--maxiter", type=int, default=None, help="Override Powell's maximum iterations.")
    parser.add_argument("--phi2-mean-match-weight", type=float, default=0.0, help="Penalty weight for the squared standardized phi2 mean shift.")
    parser.add_argument("--phi2-lower-tail-match-weight", type=float, default=0.0, help="Penalty weight for matching blocked phi2 mass below the direct q05 to 0.05.")
    parser.add_argument("--kurtosis-upper-tail-match-weight", type=float, default=0.0, help="Penalty weight for matching blocked local-kurtosis mass above the direct q90/q95 to 0.10/0.05.")
    parser.add_argument("--seed", type=int, default=2026071821)
    args = parser.parse_args()
    if args.train_n <= 0:
        raise ValueError("--train-n must be positive")
    if args.confirm_n < 0:
        raise ValueError("--confirm-n must be non-negative")

    global OUT, CAND_DIR, FINAL, PHI2_MEAN_MATCH_WEIGHT, PHI2_LOWER_TAIL_MATCH_WEIGHT, KURTOSIS_UPPER_TAIL_MATCH_WEIGHT
    PHI2_MEAN_MATCH_WEIGHT = args.phi2_mean_match_weight
    PHI2_LOWER_TAIL_MATCH_WEIGHT = args.phi2_lower_tail_match_weight
    KURTOSIS_UPPER_TAIL_MATCH_WEIGHT = args.kurtosis_upper_tail_match_weight
    if args.out is not None:
        OUT = args.out
    if args.candidate_dir is not None:
        CAND_DIR = args.candidate_dir
    if args.final_out is not None:
        FINAL = args.final_out
    OUT.mkdir(parents=True, exist_ok=True)
    CAND_DIR.mkdir(parents=True, exist_ok=True)
    FINAL.mkdir(parents=True, exist_ok=True)
    (OUT / "plots").mkdir(parents=True, exist_ok=True)
    (FINAL / "plots").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    direct = load_configs(DIRECT)[:args.train_n]
    fine = load_configs(FINE)[:args.train_n]
    if len(direct) < args.train_n or len(fine) < args.train_n:
        raise RuntimeError(f"requested {args.train_n} training configurations, but only found direct={len(direct)}, fine={len(fine)}")
    direct_obs = observable_arrays(direct)
    subset_n = min(SUBSET_N, args.train_n)
    subset_direct_obs = observable_arrays(direct[:subset_n])
    radii = [2] if args.only_5x5 else [2, 3, 4]
    eval_subset = {r: BasisEvaluator(fine[:subset_n], r) for r in radii}
    eval_full = {r: BasisEvaluator(fine, r) for r in radii}

    starts_paths = [args.start_kernel] if args.start_kernel is not None else [SELECTED, PREV_REDO, PARETO_5X5, PARETO_7X7, PHI2_BEST, RETRAINED_5X5]
    stage: list[dict[str, Any]] = []
    screen: list[tuple[float, str, str, dict[str, float], int]] = []
    families = [
        ("support_balanced_5x5", 2, [k for k in orbit_keys(2) if k != "00"], set(), 0.0030, 170, 5, 10),
        ("support_balanced_7x7_no33", 3, [k for k in orbit_keys(3) if k not in {"00", "33"}], {"33"}, 0.0022, 260, 7, 12),
        ("support_balanced_7x7_with33", 3, [k for k in orbit_keys(3) if k != "00"], set(), 0.0019, 180, 4, 10),
        ("support_balanced_9x9_stable", 4, [k for k in orbit_keys(4) if k != "00"], set(), 0.00065, 80, 2, 6),
    ]
    if args.only_5x5:
        families = families[:1]
    for family, radius, variables, zero_keys, scale, n_random, n_refine, maxiter in families:
        if args.random_per_start is not None:
            n_random = args.random_per_start
        if args.refine_count is not None:
            n_refine = args.refine_count
        if args.maxiter is not None:
            maxiter = args.maxiter
        centers: list[tuple[str, dict[str, float]]] = []
        for path in starts_paths:
            cls = load_start(path, radius, zero_keys)
            if cls is not None:
                centers.append((path.stem, cls))
        centers.append(("local_delta_center", normalize({}, radius, zero_keys)))
        fam_screen: list[tuple[float, str, str, dict[str, float], int]] = []
        for ci, (cname, center) in enumerate(centers):
            candidates = [center]
            for i in range(n_random):
                step = scale * (0.5 if i < n_random // 3 else 1.0 if i < 2 * n_random // 3 else 1.8)
                candidates.append(random_perturb(rng, center, radius, variables, zero_keys, step))
            for i, cls in enumerate(candidates):
                rec, *_ = eval_candidate(f"screen_{family}_{ci:02d}_{i:04d}", family, cls, radius, subset_direct_obs, eval_subset[radius], bins=46, momentum_grid=96)
                rec["stage"] = "subset_screen"
                rec["start"] = cname
                stage.append(rec)
                item = (float(rec["objective_score"]), rec["candidate"], family, cls, radius)
                screen.append(item)
                fam_screen.append(item)
        fam_screen.sort(key=lambda x: x[0])
        for j, (_score, _name, _fam, center, _radius) in enumerate(fam_screen[:n_refine]):
            cls, info = optimize_center(center, radius, variables, zero_keys, subset_direct_obs, eval_subset[radius], maxiter)
            rec, *_ = eval_candidate(f"opt_{family}_{j:02d}", family, cls, radius, subset_direct_obs, eval_subset[radius], bins=46, momentum_grid=128)
            rec["stage"] = "subset_powell"
            for k, v in info.items():
                rec[f"opt_{k}"] = v
            stage.append(rec)
            screen.append((float(rec["objective_score"]), rec["candidate"], family, cls, radius))

    screen.sort(key=lambda x: x[0])
    full_specs = screen[:90]
    # Also force include the existing named starts for direct comparison.
    for path in starts_paths:
        if not path.exists():
            continue
        radius = 2 if args.only_5x5 else 3
        cls = load_start(path, radius, {"33"} if "no33" in path.name else set())
        if cls is not None:
            full_specs.append((0.0, f"baseline_{path.stem}", "baseline", cls, radius))

    evaluated: list[tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, dict[str, float]], dict[str, float], dict[str, float], int, np.ndarray]] = []
    seen: set[str] = set()
    for _score, name, family, cls, radius in full_specs:
        out_name = name.replace("screen_", "full_").replace("opt_", "full_opt_")
        if out_name in seen:
            continue
        seen.add(out_name)
        rec, rows, tails, mom, matrix = eval_candidate(out_name, family, cls, radius, direct_obs, eval_full[radius], bins=80, momentum_grid=384)
        evaluated.append((rec, rows, tails, mom, cls, radius, matrix))

    records = [x[0] for x in evaluated]
    records.sort(key=lambda r: (not bool(r["promotion_worthy"]), float(r["objective_score"])))
    write_csv(OUT / "candidate_scores.csv", records)
    write_csv(OUT / "stage_subset_screen.csv", stage)

    hist_rows: list[dict[str, Any]] = []
    tail_rows: list[dict[str, Any]] = []
    mom_rows: list[dict[str, Any]] = []
    for rec, rows, tails, mom, cls, radius, matrix in evaluated:
        for obs in OBS:
            hist_rows.append({"candidate": rec["candidate"], "family": rec["family"], "observable": obs, **rows[obs]})
        for obs, t in tails.items():
            tail_rows.append({"candidate": rec["candidate"], "family": rec["family"], "observable": obs, **t})
        mom_rows.append({"candidate": rec["candidate"], "family": rec["family"], **mom, "condition_number": mom["max_K"] / mom["min_K"]})
    write_csv(OUT / "full_histogram_metrics.csv", hist_rows)
    write_csv(OUT / "tail_coverage_metrics.csv", tail_rows)
    write_csv(OUT / "momentum_stability.csv", mom_rows)

    best_rec, best_rows, best_tails, best_mom, best_cls, best_radius, best_matrix = next(x for x in evaluated if x[0]["candidate"] == records[0]["candidate"])
    best_path = write_kernel("best_phi2_support_balanced_eta_included", best_rec["family"], best_cls, best_radius, best_rec, best_rows, best_mom)

    # Optional independent-size diagnostic confirmation for the selected best.
    if args.confirm_n:
        direct5 = load_configs(DIRECT)[:args.confirm_n]
        fine5 = load_configs(FINE)[:args.confirm_n]
        if len(direct5) < args.confirm_n or len(fine5) < args.confirm_n:
            raise RuntimeError(f"requested {args.confirm_n} confirmation configurations, but only found direct={len(direct5)}, fine={len(fine5)}")
    else:
        direct5 = fine5 = None
    if args.confirm_n:
        direct5_obs = observable_arrays(direct5)
        blocked5_obs = observable_arrays(block(fine5, best_matrix))
        confirm_rows = []
        confirm_tails = []
        for obs in OBS:
            confirm_rows.append({"candidate": "best_phi2_support_balanced_eta_included", "observable": obs, **metrics(direct5_obs[obs], blocked5_obs[obs], bins=90)})
        for obs in ["action_density", "phi2", "phi4", "local_kurtosis_ratio", "NN", "G_pmin_avg"]:
            confirm_tails.append({"candidate": "best_phi2_support_balanced_eta_included", "observable": obs, **tail_stats(direct5_obs, blocked5_obs, obs)})
        write_csv(FINAL / "histogram_metrics.csv", confirm_rows)
        write_csv(FINAL / "tail_coverage.csv", confirm_tails)
        write_csv(FINAL / "momentum_stability.csv", [{"candidate": "best_phi2_support_balanced_eta_included", **best_mom, "condition_number": best_mom["max_K"] / best_mom["min_K"]}])
        write_csv(FINAL / "kernel_sum_eta_check.csv", [{"candidate": "best_phi2_support_balanced_eta_included", "kernel_path": str(best_path), "sum_K": float(best_matrix.sum()), "eta_scale": ETA_SCALE, "kernel_coefficients_include_eta_scale": True, "n_direct": args.confirm_n, "n_blocked": args.confirm_n}])
        np.savetxt(FINAL / "best_eta_included_matrix.txt", best_matrix, fmt="%.16e")
        np.savetxt(FINAL / "best_base_unit_sum_matrix.txt", best_matrix / ETA_SCALE, fmt="%.16e")
        for obs in PLOT_OBS:
            plot_histogram(direct5_obs[obs], blocked5_obs[obs], obs, FINAL / "plots" / f"best_support_balanced_{obs}.pdf", bins=90, label_a="direct native L16", label_b="blocked L32->L16 support-balanced")
    else:
        confirm_rows = []
        confirm_tails = []

    key = {r["observable"]: r for r in confirm_rows}
    tkey = {r["observable"]: r for r in confirm_tails}
    lines = [
        "# Lambda 1.0 Phi2-Support Balanced Kernel Search",
        "",
        "No kernel was promoted.",
        f"Best diagnostic kernel: `{best_path}`",
        f"Search candidates fully evaluated on {args.train_n} configs: `{len(records)}`",
        "",
        "The objective de-emphasized exact phi4 matching, removed mean-shift from ranking, and rewarded phi2 support/width rather than narrow mean agreement.",
        "",
        f"## Best {args.confirm_n}-Config Confirmation" if args.confirm_n else "## Confirmation",
        "",
        "| observable | KS | TV | JS | std ratio | below q05 | above q95 | above q99 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for obs in (["action_density", "phi2", "phi4", "local_kurtosis_ratio", "NN", "G_pmin_avg"] if args.confirm_n else []):
        r = key[obs]
        t = tkey[obs]
        lines.append(f"| `{obs}` | {float(r['ks_statistic']):.4f} | {float(r['total_variation']):.4f} | {float(r['jensen_shannon']):.4f} | {float(r['std_ratio_a_over_b']):.4f} | {float(t['below_q05']):.3f} | {float(t['above_q95']):.3f} | {float(t['above_q99']):.3f} |")
    lines.extend([
        "",
        "Reference matched tail fractions are below q05 ~= 0.05, above q95 ~= 0.05, above q99 ~= 0.01.",
        "",
        f"## Top {args.train_n}-Config Candidates",
        "",
        "| candidate | promo? | family | score | phi2 KS | phi2 >q95/q99 | kurt KS | phi4 KS | action KS | action low5/high5 | NN KS | Gpmin KS |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for r in records[:12]:
        lines.append(
            f"| `{r['candidate']}` | {r['promotion_worthy']} | {r['family']} | {float(r['objective_score']):.3g} | "
            f"{float(r['phi2_KS']):.4f} | {float(r['phi2_above_q95']):.3f}/{float(r['phi2_above_q99']):.3f} | "
            f"{float(r['local_kurtosis_ratio_KS']):.4f} | {float(r['phi4_KS']):.4f} | {float(r['action_density_KS']):.4f} | "
            f"{float(r['action_density_below_q05']):.3f}/{float(r['action_density_above_q95']):.3f} | "
            f"{float(r['NN_KS']):.4f} | {float(r['G_pmin_avg_KS']):.4f} |"
        )
    lines.extend([
        "",
        "## Files",
        "",
        f"- `{OUT / 'candidate_scores.csv'}`",
        f"- `{OUT / 'full_histogram_metrics.csv'}`",
        f"- `{OUT / 'tail_coverage_metrics.csv'}`",
        f"- `{OUT / 'momentum_stability.csv'}`",
    ])
    if args.confirm_n:
        lines.extend([
            f"- `{FINAL / 'histogram_metrics.csv'}`",
            f"- `{FINAL / 'tail_coverage.csv'}`",
            f"- `{FINAL / 'plots'}`",
        ])
    report = "\n".join(lines) + "\n"
    (OUT / "recommendation.md").write_text(report)
    (FINAL / "summary.md").write_text(report)
    print(json.dumps({"out": str(OUT), "final": str(FINAL), "best": str(best_path), "promotion_worthy": bool(best_rec["promotion_worthy"]), "full_candidates": len(records)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
