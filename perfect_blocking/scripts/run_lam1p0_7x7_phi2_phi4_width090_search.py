#!/usr/bin/env python3
"""Constrained 7x7 no-corner lambda=1.0 width search.

Non-promotion study.  The goal is to test whether allowing a 7x7 stencil can
make blocked L32->L16 distributions narrower in phi2/phi4, with direct native
remaining a bit wider, while preserving action, local-kurtosis, NN and G(pmin).
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
from scripts.common.kernel_io import load_kernel, sha256_file  # noqa: E402
from scripts.run_lam1p0_7x7_kernel_search import ETA_SCALE, block, momentum_extrema, observable_arrays  # noqa: E402
from scripts.run_lam1p0_redo_kernel_phi2_phi4_pareto_2000 import (  # noqa: E402
    BasisEvaluator,
    classes_from_matrix,
    eta_matrix_from_json,
    matrix_from_classes,
    normalize,
    orbit_keys,
    orbit_mult,
)


LAM_ROOT = PROJECT_ROOT / "perfect_blocking/perfect_blocking_lam1p0"
DIRECT = PROJECT_ROOT / "data/configs_phi4_2d/lam1p0_kappac0p340301_L16/configs.npz"
FINE = PROJECT_ROOT / "data/configs_phi4_2d/lam1p0_kappac0p340301_L32/configs.npz"
CURRENT = LAM_ROOT / "kernels/final/chosen_kernel.json"
STARTS = [
    CURRENT,
    LAM_ROOT / "kernels/candidates/tweak_phi2_phi4_width090_20260719/best_phi2_phi4_width090_5x5_eta_included.json",
    LAM_ROOT / "kernels/candidates/redo_phi2_phi4_pareto_2000/full_opt_targeted_7x7_no33_04_eta_included.json",
    LAM_ROOT / "kernels/selected_for_upscaling/current_final_7x7_no33_nn_constrained_eta_included.json",
    LAM_ROOT / "kernels/selected_for_upscaling/best_7x7_phi2_nn_guarded_eta_included.json",
]
OUT = LAM_ROOT / "tests/intermediate/7x7_phi2_phi4_width090_L32_to_L16_20260719"
CAND_DIR = LAM_ROOT / "kernels/candidates/7x7_phi2_phi4_width090_20260719"
FINAL = LAM_ROOT / "tests/final/final_kernel_confirmation_direct_L16_vs_blocked_L32_7x7_phi2_phi4_width090_20260719"

OBS = ["action_density", "phi2", "phi4", "local_kurtosis_ratio", "NN", "2nn", "diag", "m", "m2", "m4", "G_00", "G_10", "G_01", "G_pmin_avg"]
KEY_OBS = ["action_density", "phi2", "phi4", "local_kurtosis_ratio", "NN", "2nn", "diag", "m2", "m4", "G_pmin_avg"]
PLOT_OBS = ["action_density", "phi2", "phi4", "local_kurtosis_ratio", "NN", "G_pmin_avg"]
SEARCH_N = 2000
CONFIRM_N = 5000
SUBSET_N = 1000
RADIUS = 3
ZERO_KEYS = {"33"}
RNG_SEED = 2026071904


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


def load_start(path: Path) -> tuple[str, dict[str, float]] | None:
    if not path.exists():
        return None
    matrix, _meta = eta_matrix_from_json(path, target_radius=RADIUS)
    classes = classes_from_matrix(matrix / ETA_SCALE, RADIUS)
    return path.stem, normalize(classes, RADIUS, ZERO_KEYS)


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
    }


def metric_score(row: dict[str, Any], mean_weight: float = 0.10, width_weight: float = 0.35) -> float:
    std_ratio = float(row["std_b"]) / float(row["std_a"])
    width = abs(math.log(std_ratio)) if std_ratio > 0.0 and np.isfinite(std_ratio) else 10.0
    return (
        float(row["ks_statistic"])
        + 1.0 * float(row["total_variation"])
        + 8.0 * float(row["jensen_shannon"])
        + mean_weight * abs(float(row["standardized_mean_shift"]))
        + width_weight * width
    )


def width_target(rows: dict[str, dict[str, Any]]) -> float:
    targets = {
        "phi2": (0.90, 0.86, 0.96, 260.0),
        "phi4": (0.90, 0.86, 0.96, 300.0),
        "action_density": (0.93, 0.87, 0.99, 120.0),
        "local_kurtosis_ratio": (0.93, 0.87, 0.99, 140.0),
        "NN": (0.96, 0.90, 1.03, 45.0),
    }
    val = 0.0
    for obs, (target, lo, hi, weight) in targets.items():
        ratio = float(rows[obs]["std_b"]) / float(rows[obs]["std_a"])
        val += weight * (ratio - target) ** 2
        val += 8.0 * weight * max(0.0, lo - ratio) ** 2
        val += 8.0 * weight * max(0.0, ratio - hi) ** 2
    return float(val)


def objective(rows: dict[str, dict[str, Any]], tails: dict[str, dict[str, float]], mom: dict[str, float], classes: dict[str, float]) -> float:
    val = (
        16.0 * metric_score(rows["action_density"], mean_weight=0.08, width_weight=0.55)
        + 18.0 * metric_score(rows["local_kurtosis_ratio"], mean_weight=0.10, width_weight=0.65)
        + 16.0 * metric_score(rows["phi2"], mean_weight=0.08, width_weight=0.55)
        + 16.0 * metric_score(rows["phi4"], mean_weight=0.06, width_weight=0.55)
        + 9.0 * metric_score(rows["NN"], mean_weight=0.12, width_weight=0.30)
        + 3.0 * metric_score(rows["2nn"], mean_weight=0.08, width_weight=0.20)
        + 3.0 * metric_score(rows["diag"], mean_weight=0.08, width_weight=0.20)
        + 5.0 * metric_score(rows["G_pmin_avg"], mean_weight=0.10, width_weight=0.20)
        + width_target(rows)
    )
    for obs in ["phi2", "phi4"]:
        val += 2200.0 * max(0.0, tails[obs]["above_q95"] - 0.045) ** 2
        val += 3000.0 * max(0.0, tails[obs]["above_q99"] - 0.013) ** 2
        val += 1300.0 * max(0.0, tails[obs]["below_q05"] - 0.050) ** 2
        val += 450.0 * max(0.0, 0.020 - tails[obs]["above_q95"]) ** 2
    for obs in ["action_density", "local_kurtosis_ratio"]:
        val += 1600.0 * max(0.0, 0.030 - tails[obs]["above_q95"]) ** 2
        val += 1200.0 * max(0.0, 0.030 - tails[obs]["below_q05"]) ** 2
        val += 500.0 * max(0.0, tails[obs]["above_q95"] - 0.065) ** 2
        val += 500.0 * max(0.0, tails[obs]["below_q05"] - 0.075) ** 2
    limits = {
        "action_density": 0.055,
        "local_kurtosis_ratio": 0.110,
        "phi2": 0.060,
        "phi4": 0.060,
        "NN": 0.035,
        "G_pmin_avg": 0.030,
    }
    for obs, limit in limits.items():
        val += 2.5e5 * max(0.0, float(rows[obs]["ks_statistic"]) - limit) ** 2
    val += 6.0e6 * max(0.0, -float(mom["min_K"])) ** 2
    val += 6.0e5 * max(0.0, float(mom["max_inverse_K"]) - 1.55) ** 2
    val += 3.0e4 * max(0.0, float(mom["max_inverse_K"]) - 1.45) ** 2
    mult = orbit_mult(RADIUS)
    for key, coeff in classes.items():
        if key == "00":
            continue
        shell = int(key[0])
        if shell >= 3:
            val += 600.0 * mult[key] * coeff * coeff
        else:
            val += 50.0 * mult[key] * coeff * coeff
    return float(val)


def evaluate(name: str, classes: dict[str, float], direct_obs: dict[str, np.ndarray], evaluator: BasisEvaluator, bins: int, momentum_grid: int) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, dict[str, float]], dict[str, float], np.ndarray]:
    blocked_obs = observable_arrays(evaluator.blocked(classes))
    rows = full_metrics(direct_obs, blocked_obs, bins=bins)
    tails = {obs: tail_stats(direct_obs, blocked_obs, obs) for obs in ["action_density", "phi2", "phi4", "local_kurtosis_ratio", "NN", "G_pmin_avg"]}
    matrix = ETA_SCALE * matrix_from_classes(classes, RADIUS)
    mom = momentum_extrema(matrix, grid=momentum_grid)
    rec: dict[str, Any] = {
        "candidate": name,
        "objective_score": objective(rows, tails, mom, classes),
        "sum_base": float(np.sum(matrix / ETA_SCALE)),
        "sum_K": float(np.sum(matrix)),
        "eta_scale": ETA_SCALE,
        "kernel_coefficients_include_eta_scale": True,
        **mom,
    }
    for obs in KEY_OBS:
        rec[f"{obs}_KS"] = rows[obs]["ks_statistic"]
        rec[f"{obs}_TV"] = rows[obs]["total_variation"]
        rec[f"{obs}_JS"] = rows[obs]["jensen_shannon"]
        rec[f"{obs}_W1"] = rows[obs]["wasserstein_1"]
        rec[f"{obs}_shift"] = rows[obs]["standardized_mean_shift"]
        rec[f"{obs}_std_ratio_blocked_over_direct"] = float(rows[obs]["std_b"]) / float(rows[obs]["std_a"])
    for obs, vals in tails.items():
        for k, v in vals.items():
            rec[f"{obs}_{k}"] = v
    rec["promotion_candidate"] = bool(
        rec["phi2_std_ratio_blocked_over_direct"] <= 0.96
        and rec["phi4_std_ratio_blocked_over_direct"] <= 0.96
        and rec["phi2_KS"] <= 0.060
        and rec["phi4_KS"] <= 0.060
        and rec["action_density_KS"] <= 0.055
        and rec["local_kurtosis_ratio_KS"] <= 0.110
        and rec["NN_KS"] <= 0.035
        and rec["G_pmin_avg_KS"] <= 0.030
        and rec["action_density_above_q95"] >= 0.030
        and rec["local_kurtosis_ratio_above_q95"] >= 0.030
        and mom["min_K"] > 0
        and mom["max_inverse_K"] <= 1.55
    )
    return rec, rows, tails, mom, matrix


def random_perturb(rng: np.random.Generator, center: dict[str, float], variables: list[str], scale: float) -> dict[str, float]:
    cls = dict(center)
    for key in variables:
        shell = int(key[0])
        local_scale = scale * (0.7 if shell >= 3 else 1.0)
        cls[key] = float(cls[key] + rng.normal(0.0, local_scale))
    return normalize(cls, RADIUS, ZERO_KEYS)


def optimize(center: dict[str, float], variables: list[str], direct_obs: dict[str, np.ndarray], evaluator: BasisEvaluator) -> tuple[dict[str, float], dict[str, Any]]:
    x0 = np.asarray([center[k] for k in variables], dtype=np.float64)
    bounds = []
    for key in variables:
        shell = int(key[0])
        lim = 0.12 if shell <= 2 else 0.030
        bounds.append((-lim, lim))
    cache: dict[tuple[float, ...], float] = {}

    def obj(x: np.ndarray) -> float:
        key = tuple(np.round(x, 9))
        if key in cache:
            return cache[key]
        cls = dict(center)
        for k, v in zip(variables, x):
            cls[k] = float(v)
        cls = normalize(cls, RADIUS, ZERO_KEYS)
        rec, *_ = evaluate("opt", cls, direct_obs, evaluator, bins=46, momentum_grid=128)
        cache[key] = float(rec["objective_score"])
        return cache[key]

    result = minimize(obj, x0, method="Powell", bounds=bounds, options={"maxiter": 20, "xtol": 5e-5, "ftol": 5e-5, "disp": False})
    out = dict(center)
    for k, v in zip(variables, result.x):
        out[k] = float(v)
    return normalize(out, RADIUS, ZERO_KEYS), {"success": bool(result.success), "nfev": int(result.nfev), "nit": int(result.nit), "fun": float(result.fun), "message": str(result.message)}


def write_kernel(name: str, classes: dict[str, float], rec: dict[str, Any], rows: dict[str, dict[str, Any]], tails: dict[str, dict[str, float]], mom: dict[str, float]) -> Path:
    CAND_DIR.mkdir(parents=True, exist_ok=True)
    base = matrix_from_classes(classes, RADIUS)
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
        "family": "7x7_no33_phi2_phi4_width090",
        "source_kernel": str(CURRENT),
        "source_kernel_sha256": sha256_file(CURRENT),
        "selection_metrics": rec,
        "momentum_stability": mom,
        "histogram_metrics": {obs: rows[obs] for obs in KEY_OBS},
        "tail_metrics": tails,
    }
    path = CAND_DIR / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    np.savetxt(CAND_DIR / f"{name}.txt", matrix, fmt="%.16e")
    np.savetxt(CAND_DIR / f"{name}_base_unit_sum.txt", base, fmt="%.16e")
    return path


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    CAND_DIR.mkdir(parents=True, exist_ok=True)
    FINAL.mkdir(parents=True, exist_ok=True)
    (FINAL / "plots").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)
    direct = load_configs(DIRECT)[:SEARCH_N]
    fine = load_configs(FINE)[:SEARCH_N]
    direct_obs = observable_arrays(direct)
    subset_direct_obs = observable_arrays(direct[:SUBSET_N])
    evaluator = BasisEvaluator(fine, RADIUS)
    subset_eval = BasisEvaluator(fine[:SUBSET_N], RADIUS)
    starts = [x for p in STARTS if (x := load_start(p)) is not None]
    variables = [k for k in orbit_keys(RADIUS) if k not in {"00", "33"}]

    candidates: list[tuple[str, dict[str, float]]] = []
    for start_name, start_cls in starts:
        candidates.append((start_name, start_cls))
        for scale, n in [(0.0007, 45), (0.0015, 55), (0.0030, 45)]:
            for i in range(n):
                candidates.append((f"{start_name}_rand_s{scale:g}_{i:04d}", random_perturb(rng, start_cls, variables, scale)))

    screen: list[tuple[float, str, dict[str, float]]] = []
    stage_rows: list[dict[str, Any]] = []
    for name, cls in candidates:
        rec, *_ = evaluate(name, cls, subset_direct_obs, subset_eval, bins=42, momentum_grid=96)
        rec["stage"] = "subset_screen"
        stage_rows.append(rec)
        screen.append((float(rec["objective_score"]), name, cls))
    screen.sort(key=lambda x: x[0])

    refined: list[tuple[float, str, dict[str, float]]] = []
    for j, (_score, name, cls) in enumerate(screen[:12]):
        opt_cls, info = optimize(cls, variables, subset_direct_obs, subset_eval)
        rec, *_ = evaluate(f"opt_{j:02d}_{name}", opt_cls, subset_direct_obs, subset_eval, bins=46, momentum_grid=128)
        rec["stage"] = "subset_powell"
        for k, v in info.items():
            rec[f"opt_{k}"] = v
        stage_rows.append(rec)
        refined.append((float(rec["objective_score"]), rec["candidate"], opt_cls))

    full_specs = screen[:30] + refined
    evaluated: list[tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, dict[str, float]], dict[str, float], dict[str, float], np.ndarray]] = []
    final_rows: list[dict[str, Any]] = []
    hist_rows: list[dict[str, Any]] = []
    tail_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for _score, name, cls in full_specs:
        if name in seen:
            continue
        seen.add(name)
        rec, rows, tails, mom, matrix = evaluate(name, cls, direct_obs, evaluator, bins=70, momentum_grid=256)
        final_rows.append(rec)
        evaluated.append((rec, rows, tails, mom, cls, matrix))
        for obs in OBS:
            hist_rows.append({"candidate": name, "observable": obs, **rows[obs]})
        for obs, vals in tails.items():
            tail_rows.append({"candidate": name, "observable": obs, **vals})
    final_rows.sort(key=lambda r: (not bool(r["promotion_candidate"]), float(r["objective_score"])))
    write_csv(OUT / "stage_subset_screen.csv", stage_rows)
    write_csv(OUT / "candidate_scores_2000.csv", final_rows)
    write_csv(OUT / "histogram_metrics_2000.csv", hist_rows)
    write_csv(OUT / "tail_coverage_2000.csv", tail_rows)

    best_name = final_rows[0]["candidate"]
    best_rec, best_rows, best_tails, best_mom, best_cls, best_matrix = next(x for x in evaluated if x[0]["candidate"] == best_name)
    best_path = write_kernel("best_7x7_no33_phi2_phi4_width090_eta_included", best_cls, best_rec, best_rows, best_tails, best_mom)

    direct5 = load_configs(DIRECT)[:CONFIRM_N]
    fine5 = load_configs(FINE)[:CONFIRM_N]
    direct5_obs = observable_arrays(direct5)
    eval5 = BasisEvaluator(fine5, RADIUS)
    current_cls = load_start(CURRENT)[1]  # type: ignore[index]
    confirm = [("current_final_5x5", current_cls), ("best_7x7_no33_phi2_phi4_width090_eta_included", best_cls)]
    confirm_rows: list[dict[str, Any]] = []
    confirm_tails: list[dict[str, Any]] = []
    for name, cls in confirm:
        bobs = observable_arrays(eval5.blocked(cls))
        for obs in OBS:
            confirm_rows.append({"candidate": name, "observable": obs, **metrics(direct5_obs[obs], bobs[obs], bins=90)})
        for obs in ["action_density", "phi2", "phi4", "local_kurtosis_ratio", "NN", "G_pmin_avg"]:
            confirm_tails.append({"candidate": name, "observable": obs, **tail_stats(direct5_obs, bobs, obs)})
        for obs in PLOT_OBS:
            plot_histogram(direct5_obs[obs], bobs[obs], obs, FINAL / "plots" / f"{name}_{obs}.pdf", bins=90, label_a="direct native L16", label_b=f"{name} blocked L32->L16")
    write_csv(FINAL / "histogram_metrics_5000.csv", confirm_rows)
    write_csv(FINAL / "tail_coverage_5000.csv", confirm_tails)
    write_csv(FINAL / "kernel_sum_eta_check.csv", [
        {"candidate": "current_final_5x5", "kernel_path": str(CURRENT), "sha256": sha256_file(CURRENT), "sum_K": float(load_kernel(CURRENT).matrix.sum()), "eta_scale": ETA_SCALE, "kernel_coefficients_include_eta_scale": True},
        {"candidate": "best_7x7_no33_phi2_phi4_width090_eta_included", "kernel_path": str(best_path), "sum_K": float(best_matrix.sum()), "eta_scale": ETA_SCALE, "kernel_coefficients_include_eta_scale": True, **best_mom},
    ])
    np.savetxt(FINAL / "best_7x7_no33_phi2_phi4_width090_eta_included_matrix.txt", best_matrix, fmt="%.16e")
    np.savetxt(FINAL / "best_7x7_no33_phi2_phi4_width090_base_unit_sum_matrix.txt", best_matrix / ETA_SCALE, fmt="%.16e")

    cr = {(r["candidate"], r["observable"]): r for r in confirm_rows}
    ct = {(r["candidate"], r["observable"]): r for r in confirm_tails}
    lines = [
        "# Lambda 1.0 7x7 Phi2/Phi4 Width 0.90 Search",
        "",
        "No kernel was promoted.  Candidate is confirmed against the current final 5x5 on 5000 L32->L16 blocked configs.",
        f"Best candidate: `{best_path}`",
        "",
        "| candidate | observable | shift | std ratio blocked/direct | KS | TV | JS | OVL | below q05 | above q95 | above q99 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cand in ["current_final_5x5", "best_7x7_no33_phi2_phi4_width090_eta_included"]:
        for obs in ["action_density", "phi2", "phi4", "local_kurtosis_ratio", "NN", "G_pmin_avg"]:
            r = cr[(cand, obs)]
            t = ct[(cand, obs)]
            std_ratio = float(r["std_b"]) / float(r["std_a"])
            lines.append(
                f"| `{cand}` | `{obs}` | {float(r['standardized_mean_shift']):.4f} | {std_ratio:.4f} | "
                f"{float(r['ks_statistic']):.4f} | {float(r['total_variation']):.4f} | {float(r['jensen_shannon']):.5f} | "
                f"{1.0 - float(r['total_variation']):.4f} | {float(t['below_q05']):.4f} | {float(t['above_q95']):.4f} | {float(t['above_q99']):.4f} |"
            )
    lines.extend([
        "",
        "Reference tail fractions are below q05 ~= 0.05, above q95 ~= 0.05, above q99 ~= 0.01.",
        "",
        "## Files",
        "",
        f"- `{OUT / 'candidate_scores_2000.csv'}`",
        f"- `{FINAL / 'histogram_metrics_5000.csv'}`",
        f"- `{FINAL / 'tail_coverage_5000.csv'}`",
        f"- `{FINAL / 'plots'}`",
    ])
    report = "\n".join(lines) + "\n"
    (OUT / "summary.md").write_text(report)
    (FINAL / "summary.md").write_text(report)
    print(json.dumps({"out": str(OUT), "final": str(FINAL), "best": str(best_path), "promotion_candidate": bool(best_rec["promotion_candidate"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
