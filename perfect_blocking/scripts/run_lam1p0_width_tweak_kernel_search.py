#!/usr/bin/env python3
"""Small lambda=1.0 kernel-width tweak search.

This is a non-promotion study.  It starts from the current final
support-balanced 5x5 kernel and makes small symmetric 5x5 perturbations.
The target is not to make the blocked ensemble wider than direct.  Instead,
direct native should remain a bit wider everywhere: action/local-kurtosis
support gaps are reduced, while phi2/phi4 are narrowed back toward direct.
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
    matrix_from_classes,
    normalize,
    orbit_keys,
    orbit_mult,
)


LAM_ROOT = PROJECT_ROOT / "perfect_blocking/perfect_blocking_lam1p0"
DIRECT = PROJECT_ROOT / "data/configs_phi4_2d/lam1p0_kappac0p340301_L16/configs.npz"
FINE = PROJECT_ROOT / "data/configs_phi4_2d/lam1p0_kappac0p340301_L32/configs.npz"
CURRENT = LAM_ROOT / "kernels/final/chosen_kernel.json"
OUT = LAM_ROOT / "tests/intermediate/tweak_kernel_action_kurtosis_width_L32_to_L16_20260719"
CAND_DIR = LAM_ROOT / "kernels/candidates/tweak_action_kurtosis_width_20260719"
FINAL = LAM_ROOT / "tests/final/final_kernel_confirmation_direct_L16_vs_blocked_L32_width_tweak_20260719"

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
    "G_00",
    "G_10",
    "G_01",
    "G_pmin_avg",
]
KEY_OBS = ["action_density", "phi2", "phi4", "local_kurtosis_ratio", "NN", "2nn", "diag", "m2", "m4", "G_pmin_avg"]
PLOT_OBS = ["action_density", "phi2", "phi4", "local_kurtosis_ratio", "NN", "G_pmin_avg"]
SEARCH_N = 2000
CONFIRM_N = 5000
RADIUS = 2
RNG_SEED = 2026071903


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
    }


def metric_score(row: dict[str, Any], *, mean_weight: float = 0.15, width_weight: float = 0.30) -> float:
    std_ratio = float(row["std_ratio_a_over_b"])
    width = 0.0 if std_ratio <= 0.0 or not np.isfinite(std_ratio) else abs(math.log(std_ratio))
    return (
        float(row["ks_statistic"])
        + 1.0 * float(row["total_variation"])
        + 8.0 * float(row["jensen_shannon"])
        + mean_weight * abs(float(row["standardized_mean_shift"]))
        + width_weight * width
    )


def width_target_penalty(rows: dict[str, dict[str, Any]]) -> float:
    # We want direct to remain a little broader.  Current action and
    # local-kurtosis are too narrow, while phi2 and phi4 are too wide.  The
    # target is therefore a blocked/direct std ratio near 0.95, with a loose
    # acceptable range rather than an exact equality.
    targets = {
        "action_density": (0.95, 0.90, 0.99, 85.0),
        "local_kurtosis_ratio": (0.95, 0.90, 0.99, 105.0),
        "phi2": (0.96, 0.91, 1.00, 95.0),
        "phi4": (0.96, 0.91, 1.00, 105.0),
        "NN": (0.96, 0.91, 1.02, 35.0),
    }
    val = 0.0
    for obs, (target, lo, hi, weight) in targets.items():
        ratio = float(rows[obs]["std_b"]) / float(rows[obs]["std_a"])
        val += weight * (ratio - target) ** 2
        val += 6.0 * weight * max(0.0, lo - ratio) ** 2
        val += 6.0 * weight * max(0.0, ratio - hi) ** 2
    return float(val)


def objective(rows: dict[str, dict[str, Any]], tails: dict[str, dict[str, float]], mom: dict[str, float], classes: dict[str, float]) -> float:
    # The current kernel already matches phi2/phi4/NN means well.  The objective
    # mainly aims for distribution support: direct slightly wider than blocked
    # for all local observables, without creating action/local-kurtosis support
    # holes.
    val = (
        18.0 * metric_score(rows["action_density"], mean_weight=0.08, width_weight=0.65)
        + 24.0 * metric_score(rows["local_kurtosis_ratio"], mean_weight=0.10, width_weight=0.75)
        + 8.0 * metric_score(rows["phi2"], mean_weight=0.10, width_weight=0.25)
        + 8.0 * metric_score(rows["phi4"], mean_weight=0.08, width_weight=0.18)
        + 9.0 * metric_score(rows["NN"], mean_weight=0.12, width_weight=0.25)
        + 3.0 * metric_score(rows["2nn"], mean_weight=0.08, width_weight=0.18)
        + 3.0 * metric_score(rows["diag"], mean_weight=0.08, width_weight=0.18)
        + 4.0 * metric_score(rows["G_pmin_avg"], mean_weight=0.10, width_weight=0.20)
        + width_target_penalty(rows)
    )

    # In direct-tail coordinates, a blocked distribution that is just slightly
    # narrower should occupy roughly 3.5-4.5% of the direct 5% tails.  Current
    # action/local-kurtosis high-tail coverage is too low; current phi2/phi4
    # tail coverage is too high.
    for obs, target_low, target_high in [
        ("action_density", 0.038, 0.038),
        ("local_kurtosis_ratio", 0.040, 0.038),
    ]:
        val += 1400.0 * max(0.0, target_low - tails[obs]["below_q05"]) ** 2
        val += 1800.0 * max(0.0, target_high - tails[obs]["above_q95"]) ** 2
        val += 1800.0 * max(0.0, 0.006 - tails[obs]["above_q99"]) ** 2
        val += 450.0 * max(0.0, tails[obs]["below_q05"] - 0.058) ** 2
        val += 450.0 * max(0.0, tails[obs]["above_q95"] - 0.058) ** 2

    for obs in ["phi2", "phi4"]:
        val += 900.0 * max(0.0, tails[obs]["below_q05"] - 0.052) ** 2
        val += 1300.0 * max(0.0, tails[obs]["above_q95"] - 0.052) ** 2
        val += 1900.0 * max(0.0, tails[obs]["above_q99"] - 0.017) ** 2
        val += 450.0 * max(0.0, 0.025 - tails[obs]["below_q05"]) ** 2
        val += 450.0 * max(0.0, 0.025 - tails[obs]["above_q95"]) ** 2

    # Protect already-good marginals.  Phi4 should narrow; do not let a
    # candidate solve action by making phi4 still wider.
    limits = {
        "action_density": 0.045,
        "local_kurtosis_ratio": 0.075,
        "phi2": 0.050,
        "phi4": 0.045,
        "NN": 0.030,
        "G_pmin_avg": 0.030,
    }
    for obs, lim in limits.items():
        val += 2.0e5 * max(0.0, float(rows[obs]["ks_statistic"]) - lim) ** 2

    val += 5.0e6 * max(0.0, -float(mom["min_K"])) ** 2
    val += 4.0e5 * max(0.0, float(mom["max_inverse_K"]) - 1.50) ** 2
    val += 2.0e4 * max(0.0, float(mom["max_inverse_K"]) - 1.40) ** 2
    mult = orbit_mult(RADIUS)
    for key, coeff in classes.items():
        if key != "00":
            val += 80.0 * mult[key] * coeff * coeff
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
        rec[f"{obs}_std_ratio_blocked_over_direct"] = float(rows[obs]["std_b"] / rows[obs]["std_a"])
    for obs, vals in tails.items():
        for k, v in vals.items():
            rec[f"{obs}_{k}"] = v
    rec["promotion_candidate"] = bool(
        rec["action_density_KS"] <= 0.045
        and rec["local_kurtosis_ratio_KS"] <= 0.075
        and rec["phi2_KS"] <= 0.050
        and rec["phi4_KS"] <= 0.045
        and rec["NN_KS"] <= 0.030
        and rec["G_pmin_avg_KS"] <= 0.030
        and rec["action_density_below_q05"] >= 0.035
        and rec["action_density_above_q95"] >= 0.035
        and rec["local_kurtosis_ratio_above_q95"] >= 0.035
        and rec["phi2_above_q95"] <= 0.055
        and rec["phi4_above_q95"] <= 0.055
        and rec["phi2_std_ratio_blocked_over_direct"] <= 1.01
        and rec["phi4_std_ratio_blocked_over_direct"] <= 1.01
        and mom["min_K"] > 0
        and mom["max_inverse_K"] <= 1.50
    )
    return rec, rows, tails, mom, matrix


def random_perturb(rng: np.random.Generator, center: dict[str, float], variables: list[str], scale: float) -> dict[str, float]:
    out = dict(center)
    for key in variables:
        out[key] = float(out[key] + rng.normal(0.0, scale))
    return normalize(out, RADIUS, set())


def optimize(center: dict[str, float], variables: list[str], direct_obs: dict[str, np.ndarray], evaluator: BasisEvaluator) -> tuple[dict[str, float], dict[str, Any]]:
    x0 = np.asarray([center[k] for k in variables], dtype=np.float64)
    bounds = [(-0.12, 0.12) for _ in variables]
    cache: dict[tuple[float, ...], float] = {}

    def obj(x: np.ndarray) -> float:
        key = tuple(np.round(x, 9))
        if key in cache:
            return cache[key]
        cls = dict(center)
        for k, v in zip(variables, x):
            cls[k] = float(v)
        cls = normalize(cls, RADIUS, set())
        rec, *_ = evaluate("opt", cls, direct_obs, evaluator, bins=52, momentum_grid=128)
        cache[key] = float(rec["objective_score"])
        return cache[key]

    result = minimize(obj, x0, method="Powell", bounds=bounds, options={"maxiter": 24, "xtol": 4e-5, "ftol": 4e-5, "disp": False})
    out = dict(center)
    for k, v in zip(variables, result.x):
        out[k] = float(v)
    return normalize(out, RADIUS, set()), {"success": bool(result.success), "nfev": int(result.nfev), "nit": int(result.nit), "fun": float(result.fun), "message": str(result.message)}


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
        "family": "width_tweak_5x5_from_final_chosen",
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
    (OUT / "plots").mkdir(parents=True, exist_ok=True)
    (FINAL / "plots").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)

    kernel = load_kernel(CURRENT)
    base = kernel.matrix / ETA_SCALE
    start = normalize(classes_from_matrix(base, RADIUS), RADIUS, set())
    variables = [k for k in orbit_keys(RADIUS) if k != "00"]

    direct = load_configs(DIRECT)[:SEARCH_N]
    fine = load_configs(FINE)[:SEARCH_N]
    direct_obs = observable_arrays(direct)
    evaluator = BasisEvaluator(fine, RADIUS)

    candidates: list[tuple[str, dict[str, float]]] = [("current_final_chosen", start)]
    for scale, n in [(0.0006, 80), (0.0012, 110), (0.0024, 100), (0.0040, 55)]:
        for i in range(n):
            candidates.append((f"rand_s{scale:g}_{i:04d}", random_perturb(rng, start, variables, scale)))

    screened: list[tuple[float, str, dict[str, float]]] = []
    stage_rows: list[dict[str, Any]] = []
    for name, cls in candidates:
        rec, *_ = evaluate(name, cls, direct_obs, evaluator, bins=52, momentum_grid=128)
        rec["stage"] = "screen_2000"
        stage_rows.append(rec)
        screened.append((float(rec["objective_score"]), name, cls))
    screened.sort(key=lambda x: x[0])

    refined: list[tuple[float, str, dict[str, float]]] = []
    for j, (_score, name, cls) in enumerate(screened[:10]):
        opt_cls, info = optimize(cls, variables, direct_obs, evaluator)
        rec, *_ = evaluate(f"opt_{j:02d}_{name}", opt_cls, direct_obs, evaluator, bins=64, momentum_grid=192)
        rec["stage"] = "powell_2000"
        for k, v in info.items():
            rec[f"opt_{k}"] = v
        stage_rows.append(rec)
        refined.append((float(rec["objective_score"]), rec["candidate"], opt_cls))

    full_specs = screened[:30] + refined
    final_rows: list[dict[str, Any]] = []
    hist_rows: list[dict[str, Any]] = []
    tail_rows: list[dict[str, Any]] = []
    evaluated: list[tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, dict[str, float]], dict[str, float], dict[str, float], np.ndarray]] = []
    seen: set[str] = set()
    for _score, name, cls in full_specs:
        if name in seen:
            continue
        seen.add(name)
        rec, rows, tails, mom, matrix = evaluate(name, cls, direct_obs, evaluator, bins=80, momentum_grid=384)
        final_rows.append(rec)
        evaluated.append((rec, rows, tails, mom, cls, matrix))
        for obs in OBS:
            hist_rows.append({"candidate": name, "observable": obs, **rows[obs]})
        for obs, vals in tails.items():
            tail_rows.append({"candidate": name, "observable": obs, **vals})

    final_rows.sort(key=lambda r: (not bool(r["promotion_candidate"]), float(r["objective_score"])))
    write_csv(OUT / "screen_and_refine_metrics.csv", stage_rows)
    write_csv(OUT / "candidate_scores_2000.csv", final_rows)
    write_csv(OUT / "histogram_metrics_2000.csv", hist_rows)
    write_csv(OUT / "tail_coverage_2000.csv", tail_rows)

    best_name = final_rows[0]["candidate"]
    best_rec, best_rows, best_tails, best_mom, best_cls, best_matrix = next(x for x in evaluated if x[0]["candidate"] == best_name)
    best_path = write_kernel("best_width_tweak_5x5_eta_included", best_cls, best_rec, best_rows, best_tails, best_mom)

    # 5000-config confirmation of current and best candidate.
    direct5 = load_configs(DIRECT)[:CONFIRM_N]
    fine5 = load_configs(FINE)[:CONFIRM_N]
    direct5_obs = observable_arrays(direct5)
    eval5 = BasisEvaluator(fine5, RADIUS)
    confirm: list[tuple[str, dict[str, float]]] = [("current_final_chosen", start), ("best_width_tweak_5x5_eta_included", best_cls)]
    confirm_rows: list[dict[str, Any]] = []
    confirm_tails: list[dict[str, Any]] = []
    for name, cls in confirm:
        blocked_obs = observable_arrays(eval5.blocked(cls))
        for obs in OBS:
            confirm_rows.append({"candidate": name, "observable": obs, **metrics(direct5_obs[obs], blocked_obs[obs], bins=90)})
        for obs in ["action_density", "phi2", "phi4", "local_kurtosis_ratio", "NN", "G_pmin_avg"]:
            confirm_tails.append({"candidate": name, "observable": obs, **tail_stats(direct5_obs, blocked_obs, obs)})
        for obs in PLOT_OBS:
            plot_histogram(direct5_obs[obs], blocked_obs[obs], obs, FINAL / "plots" / f"{name}_{obs}.pdf", bins=90, label_a="direct native L16", label_b=f"{name} blocked L32->L16")
    write_csv(FINAL / "histogram_metrics_5000.csv", confirm_rows)
    write_csv(FINAL / "tail_coverage_5000.csv", confirm_tails)
    write_csv(FINAL / "kernel_sum_eta_check.csv", [
        {
            "candidate": "current_final_chosen",
            "kernel_path": str(CURRENT),
            "sha256": sha256_file(CURRENT),
            "sum_K": float(kernel.matrix.sum()),
            "eta_scale": ETA_SCALE,
            "kernel_coefficients_include_eta_scale": bool(kernel.kernel_coefficients_include_eta_scale),
        },
        {
            "candidate": "best_width_tweak_5x5_eta_included",
            "kernel_path": str(best_path),
            "sum_K": float(best_matrix.sum()),
            "eta_scale": ETA_SCALE,
            "kernel_coefficients_include_eta_scale": True,
            **best_mom,
        },
    ])
    np.savetxt(FINAL / "best_width_tweak_eta_included_matrix.txt", best_matrix, fmt="%.16e")
    np.savetxt(FINAL / "best_width_tweak_base_unit_sum_matrix.txt", best_matrix / ETA_SCALE, fmt="%.16e")

    by_candidate = {(r["candidate"], r["observable"]): r for r in confirm_rows}
    by_tail = {(r["candidate"], r["observable"]): r for r in confirm_tails}
    lines = [
        "# Lambda 1.0 L32->L16 Width-Tweak Kernel Study",
        "",
        "No kernel was promoted.  This study starts from the current final 5x5 kernel and makes small local perturbations.  The target is direct native slightly wider than blocked for all local observables: action/local-kurtosis support gaps should shrink, while phi2/phi4 should narrow relative to the current kernel.",
        "",
        f"Current kernel: `{CURRENT}`",
        f"Current hash: `{sha256_file(CURRENT)}`",
        f"Best diagnostic candidate: `{best_path}`",
        "",
        "## 5000-Config Confirmation",
        "",
        "| candidate | observable | shift | std ratio blocked/direct | KS | TV | JS | OVL | below q05 | above q95 | above q99 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cand in ["current_final_chosen", "best_width_tweak_5x5_eta_included"]:
        for obs in ["action_density", "phi2", "phi4", "local_kurtosis_ratio", "NN", "G_pmin_avg"]:
            r = by_candidate[(cand, obs)]
            t = by_tail[(cand, obs)]
            std_ratio = float(r["std_b"]) / float(r["std_a"])
            ovl = 1.0 - float(r["total_variation"])
            lines.append(
                f"| `{cand}` | `{obs}` | {float(r['standardized_mean_shift']):.4f} | {std_ratio:.4f} | "
                f"{float(r['ks_statistic']):.4f} | {float(r['total_variation']):.4f} | {float(r['jensen_shannon']):.5f} | "
                f"{ovl:.4f} | {float(t['below_q05']):.4f} | {float(t['above_q95']):.4f} | {float(t['above_q99']):.4f} |"
            )
    lines.extend([
        "",
        "Reference tail fractions are below q05 ~= 0.05, above q95 ~= 0.05, and above q99 ~= 0.01.",
        "",
        "## Interpretation",
        "",
        "Promotion should require improvement in action/local-kurtosis support without losing phi2, phi4, NN, or G(pmin).  If the best candidate mostly trades one local observable for another, keep the current final kernel.",
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
