#!/usr/bin/env python3
"""Controlled lambda=1.0 7x7 perfect-blocking kernel search."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
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


LAM = 1.0
KAPPA = 0.340301
ETA = 0.25
ETA_SCALE = 2.0 ** (ETA / 2.0)
BLOCK_FACTOR = 2

DEFAULT_ROOT = PROJECT_ROOT / "perfect_blocking/perfect_blocking_lam1p0"
DEFAULT_FINE = PROJECT_ROOT / "data/configs_phi4_2d/lam1p0_kappac0p340301_L32/configs.npz"
DEFAULT_DIRECT = PROJECT_ROOT / "data/configs_phi4_2d/lam1p0_kappac0p340301_L16/configs.npz"
DEFAULT_FINAL = DEFAULT_ROOT / "kernels/final/chosen_kernel.json"
DEFAULT_STUDY = DEFAULT_ROOT / "tests/intermediate/7x7_kernel_search"
DEFAULT_CAND_DIR = DEFAULT_ROOT / "kernels/candidates/7x7_search"

OBSERVABLES = [
    "phi2",
    "phi4",
    "local_kurtosis_ratio",
    "NN",
    "2nn",
    "diag",
    "action_density",
    "m",
    "m2",
    "m4",
]
GK_OBSERVABLES = ["G_00", "G_10", "G_01", "G_pmin_avg"]
OBJECTIVE_WEIGHTS = {
    "phi2": 5.0,
    "local_kurtosis_ratio": 6.0,
    "phi4": 4.0,
    "action_density": 2.0,
    "NN": 1.0,
    "2nn": 1.0,
    "diag": 1.0,
    "m2": 2.0,
    "m4": 2.0,
    "G_pmin_avg": 2.0,
}

CLASS_MULT = {
    "00": 1,
    "10": 4,
    "11": 4,
    "20": 4,
    "21": 8,
    "22": 4,
    "30": 4,
    "31": 8,
    "32": 8,
    "33": 4,
}


@dataclass(frozen=True)
class Candidate:
    name: str
    family: str
    classes: dict[str, float]
    base_matrix: np.ndarray
    op_matrix: np.ndarray
    metadata: dict[str, Any]


def matrix_from_classes(classes: dict[str, float], radius: int = 3) -> np.ndarray:
    matrix = np.zeros((2 * radius + 1, 2 * radius + 1), dtype=np.float64)
    offsets = range(-radius, radius + 1)
    for i, dx in enumerate(offsets):
        for j, dy in enumerate(offsets):
            a, b = sorted((abs(dx), abs(dy)), reverse=True)
            key = f"{a}{b}"
            matrix[i, j] = float(classes.get(key, 0.0))
    return matrix


def classes_from_matrix(matrix: np.ndarray) -> dict[str, float]:
    radius = matrix.shape[0] // 2
    out: dict[str, float] = {}
    for dx in range(0, radius + 1):
        for dy in range(0, dx + 1):
            key = f"{dx}{dy}"
            out[key] = float(matrix[radius + dx, radius + dy])
    return out


def normalize_classes_with_dependent_center(params: dict[str, float]) -> dict[str, float]:
    classes = dict(params)
    rest = sum(CLASS_MULT[k] * v for k, v in classes.items() if k != "00")
    classes["00"] = 1.0 - rest
    return classes


def embed_current_5x5_base(path: Path) -> dict[str, float]:
    kernel = load_kernel(path)
    base5 = kernel.matrix / ETA_SCALE
    base7 = np.zeros((7, 7), dtype=np.float64)
    base7[1:6, 1:6] = base5
    return classes_from_matrix(base7)


def apply_kernel(phi: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    radius = matrix.shape[0] // 2
    out = np.zeros_like(phi, dtype=np.float64)
    for i, dx in enumerate(range(-radius, radius + 1)):
        for j, dy in enumerate(range(-radius, radius + 1)):
            w = matrix[i, j]
            if w != 0.0:
                out += w * np.roll(np.roll(phi, -dx, axis=1), -dy, axis=2)
    return out


def block(phi: np.ndarray, op_matrix: np.ndarray) -> np.ndarray:
    return apply_kernel(phi, op_matrix)[:, 0::BLOCK_FACTOR, 0::BLOCK_FACTOR]


def observable_arrays(phi: np.ndarray) -> dict[str, np.ndarray]:
    arr = np.asarray(phi, dtype=np.float64)
    n, L, _ = arr.shape
    volume = L * L
    m = arr.mean(axis=(1, 2))
    phi2 = np.mean(arr**2, axis=(1, 2))
    phi4 = np.mean(arr**4, axis=(1, 2))
    nn = 0.5 * (
        np.mean(arr * np.roll(arr, -1, axis=1), axis=(1, 2))
        + np.mean(arr * np.roll(arr, -1, axis=2), axis=(1, 2))
    )
    twonn = 0.5 * (
        np.mean(arr * np.roll(arr, -2, axis=1), axis=(1, 2))
        + np.mean(arr * np.roll(arr, -2, axis=2), axis=(1, 2))
    )
    diag = np.mean(arr * np.roll(np.roll(arr, -1, axis=1), -1, axis=2), axis=(1, 2))
    total_action = ((1.0 - 2.0 * LAM) * arr**2 + LAM * arr**4 - 2.0 * KAPPA * (
        arr * np.roll(arr, -1, axis=1) + arr * np.roll(arr, -1, axis=2)
    )).sum(axis=(1, 2))
    phase = np.exp(2j * np.pi * np.arange(L) / L)
    phi_x = np.tensordot(arr, phase, axes=([1], [0])).sum(axis=1)
    phi_y = np.tensordot(arr, phase, axes=([2], [0])).sum(axis=1)
    g10 = np.abs(phi_x) ** 2 / float(volume)
    g01 = np.abs(phi_y) ** 2 / float(volume)
    out = {
        "phi2": phi2,
        "phi4": phi4,
        "local_kurtosis_ratio": phi4 / np.maximum(phi2 * phi2, 1.0e-300),
        "NN": nn,
        "2nn": twonn,
        "diag": diag,
        "action_density": total_action / volume,
        "m": m,
        "m2": m * m,
        "m4": m**4,
        "G_00": volume * m * m,
        "G_10": g10,
        "G_01": g01,
        "G_pmin_avg": 0.5 * (g10 + g01),
    }
    mean_m = float(np.mean(m))
    mean_m2 = float(np.mean(out["m2"]))
    mean_m4 = float(np.mean(out["m4"]))
    binder = 1.0 - mean_m4 / max(3.0 * mean_m2 * mean_m2, 1.0e-300)
    gp = float(np.mean(out["G_pmin_avg"]))
    g0 = float(volume * max(mean_m2 - mean_m * mean_m, 0.0))
    sqrt_arg = g0 / gp - 1.0 if gp > 0.0 else float("nan")
    xi = (1.0 / (2.0 * L * math.sin(math.pi / L))) * math.sqrt(sqrt_arg) if sqrt_arg > 0 else float("nan")
    out["Binder_U4_from_averages"] = np.full(n, binder)
    out["xi_over_L"] = np.full(n, xi)
    return out


def momentum_extrema(matrix: np.ndarray, grid: int = 512) -> dict[str, float]:
    radius = matrix.shape[0] // 2
    coords = np.arange(-radius, radius + 1)
    ps = np.linspace(-np.pi, np.pi, grid, endpoint=False)
    min_k = float("inf")
    max_k = -float("inf")
    min_inv = float("inf")
    max_inv = -float("inf")
    min_k_px = min_k_py = max_k_px = max_k_py = 0.0
    for px in ps:
        ex = np.exp(1j * px * coords)
        ey = np.exp(1j * ps[:, None] * coords)
        vals = np.einsum("i,ij,nj->n", ex, matrix, ey)
        vals = np.real_if_close(vals, tol=1000).real
        inv = 1.0 / vals
        i_min = int(np.argmin(vals))
        i_max = int(np.argmax(vals))
        if float(vals[i_min]) < min_k:
            min_k = float(vals[i_min])
            min_k_px, min_k_py = float(px), float(ps[i_min])
        if float(vals[i_max]) > max_k:
            max_k = float(vals[i_max])
            max_k_px, max_k_py = float(px), float(ps[i_max])
        min_inv = min(min_inv, float(np.min(inv)))
        max_inv = max(max_inv, float(np.max(inv)))
    return {
        "min_K": min_k,
        "max_K": max_k,
        "min_inverse_K": min_inv,
        "max_inverse_K": max_inv,
        "min_K_px": min_k_px,
        "min_K_py": min_k_py,
        "max_K_px": max_k_px,
        "max_K_py": max_k_py,
    }


def quick_score(direct: dict[str, np.ndarray], blocked: dict[str, np.ndarray], mom: dict[str, float], outer: dict[str, float]) -> float:
    score = 0.0
    for obs, weight in OBJECTIVE_WEIGHTS.items():
        a = direct[obs]
        b = blocked[obs]
        pooled = math.sqrt(0.5 * (float(np.var(a, ddof=1)) + float(np.var(b, ddof=1))))
        mean_shift = abs((float(np.mean(b)) - float(np.mean(a))) / max(pooled, 1.0e-12))
        std_ratio = abs(math.log(max(float(np.std(a, ddof=1)), 1.0e-12) / max(float(np.std(b, ddof=1)), 1.0e-12)))
        score += weight * (mean_shift + 0.3 * std_ratio)
    if mom["min_K"] <= 0.0:
        score += 1.0e6 + 1.0e5 * abs(mom["min_K"])
    if mom["max_inverse_K"] > 2.0:
        score += 100.0 * (mom["max_inverse_K"] - 2.0) ** 2
    for key, value in outer.items():
        mult = CLASS_MULT[key]
        scale = 20.0 if key == "33" else 4.0
        score += scale * mult * value * value
    return float(score)


def full_metrics(direct: dict[str, np.ndarray], blocked: dict[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for obs in OBSERVABLES + ["Binder_U4_from_averages", "xi_over_L", *GK_OBSERVABLES]:
        out[obs] = metrics(direct[obs], blocked[obs], bins=50)
    return out


def ranking_score(metric_rows: dict[str, dict[str, Any]]) -> float:
    selected = ["phi2", "phi4", "local_kurtosis_ratio", "action_density", "NN", "m2", "m4", "G_pmin_avg"]
    weights = {"phi2": 4.0, "local_kurtosis_ratio": 5.0, "phi4": 3.0}
    total = 0.0
    for obs in selected:
        row = metric_rows[obs]
        weight = weights.get(obs, 1.0)
        total += weight * (
            abs(float(row["standardized_mean_shift"]))
            + float(row["total_variation"])
            + 10.0 * float(row["jensen_shannon"])
        )
    return float(total)


def make_candidate(name: str, family: str, classes: dict[str, float], metadata: dict[str, Any]) -> Candidate:
    base = matrix_from_classes(classes)
    op = ETA_SCALE * base
    return Candidate(name=name, family=family, classes=classes, base_matrix=base, op_matrix=op, metadata=metadata)


def optimize_family(
    family: str,
    start: dict[str, float],
    variables: list[str],
    fine_subset: np.ndarray,
    direct_subset_obs: dict[str, np.ndarray],
    maxiter: int,
    include_33_penalty: bool = False,
    momentum_grid: int = 64,
) -> tuple[dict[str, float], dict[str, Any]]:
    x0 = np.asarray([start.get(k, 0.0) for k in variables], dtype=np.float64)
    bounds = [(-0.12, 0.16) for _ in variables]

    cache: dict[tuple[float, ...], float] = {}

    def obj(x: np.ndarray) -> float:
        key = tuple(np.round(x, 12))
        if key in cache:
            return cache[key]
        params = dict(start)
        for name, value in zip(variables, x):
            params[name] = float(value)
        classes = normalize_classes_with_dependent_center(params)
        base = matrix_from_classes(classes)
        op = ETA_SCALE * base
        mom = momentum_extrema(op, grid=momentum_grid)
        blocked_obs = observable_arrays(block(fine_subset, op))
        outer = {k: classes.get(k, 0.0) for k in ("30", "31", "32", "33") if k in classes}
        if include_33_penalty:
            outer["33"] = classes.get("33", 0.0)
        val = quick_score(direct_subset_obs, blocked_obs, mom, outer)
        cache[key] = val
        return val

    result = minimize(
        obj,
        x0,
        method="Powell",
        bounds=bounds,
        options={"maxiter": maxiter, "xtol": 1.0e-4, "ftol": 1.0e-4, "disp": False},
    )
    params = dict(start)
    for name, value in zip(variables, result.x):
        params[name] = float(value)
    classes = normalize_classes_with_dependent_center(params)
    return classes, {
        "success": bool(result.success),
        "message": str(result.message),
        "fun": float(result.fun),
        "nfev": int(result.nfev),
        "nit": int(result.nit),
        "variables": variables,
    }


def write_candidate(candidate: Candidate, cand_dir: Path) -> None:
    cand_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "name": candidate.name,
        "type": "matrix",
        "matrix": candidate.op_matrix.tolist(),
        "base_matrix_before_eta_scale": candidate.base_matrix.tolist(),
        "base_orbit_classes_before_eta_scale": candidate.classes,
        "lambda": LAM,
        "kappa_f": KAPPA,
        "kappa_c": KAPPA,
        "eta": ETA,
        "block_factor": BLOCK_FACTOR,
        "scale_factor": BLOCK_FACTOR,
        "eta_scale": "2^0.125",
        "eta_scale_numeric": ETA_SCALE,
        "kernel_coefficients_include_eta_scale": True,
        "base_kernel_sum_before_eta_scale": float(candidate.base_matrix.sum()),
        "final_kernel_sum_after_eta_scale": float(candidate.op_matrix.sum()),
        "normalization": "eta_included_sum_to_eta_scale",
        "convention": "stored coefficients include eta_scale; do not multiply again on application",
        "family": candidate.family,
        "metadata": candidate.metadata,
    }
    json_path = cand_dir / f"{candidate.name}.json"
    json_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    lines = [
        f"name: {candidate.name}",
        f"family: {candidate.family}",
        f"eta_scale: {ETA_SCALE:.17g}",
        f"matrix_sum: {float(candidate.op_matrix.sum()):.17g}",
        "matrix:",
    ]
    for row in candidate.op_matrix:
        lines.append(" ".join(f"{x:.17g}" for x in row))
    (cand_dir / f"{candidate.name}.txt").write_text("\n".join(lines) + "\n")
    metadata = dict(candidate.metadata)
    metadata.update(
        {
            "candidate_json": str(json_path),
            "candidate_txt": str(cand_dir / f"{candidate.name}.txt"),
            "base_kernel_sum_before_eta_scale": float(candidate.base_matrix.sum()),
            "final_kernel_sum_after_eta_scale": float(candidate.op_matrix.sum()),
        }
    )
    (cand_dir / f"{candidate.name}_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def write_observable_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, default=DEFAULT_STUDY)
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_CAND_DIR)
    parser.add_argument("--fine-configs", type=Path, default=DEFAULT_FINE)
    parser.add_argument("--direct-configs", type=Path, default=DEFAULT_DIRECT)
    parser.add_argument("--current-kernel", type=Path, default=DEFAULT_FINAL)
    parser.add_argument("--opt-configs", type=int, default=1600)
    parser.add_argument("--maxiter", type=int, default=45)
    parser.add_argument("--include-full33", action="store_true", help="Run optional full 7x7 family with penalized (3,3).")
    parser.add_argument("--skip-full", action="store_true")
    args = parser.parse_args()

    args.study_dir.mkdir(parents=True, exist_ok=True)
    args.candidate_dir.mkdir(parents=True, exist_ok=True)
    (args.study_dir / "plots").mkdir(parents=True, exist_ok=True)

    fine_full = load_configs(args.fine_configs)
    direct_full = load_configs(args.direct_configs)
    fine_subset = fine_full[: args.opt_configs]
    direct_subset = direct_full[: args.opt_configs]
    direct_subset_obs = observable_arrays(direct_subset)
    direct_full_obs = observable_arrays(direct_full)

    start = embed_current_5x5_base(args.current_kernel)
    for key in ("30", "31", "32", "33"):
        start.setdefault(key, 0.0)

    candidates: list[Candidate] = []
    candidates.append(
        make_candidate(
            "embedded_5x5_control_eta_included",
            "embedded_5x5_baseline",
            normalize_classes_with_dependent_center(start),
            {"description": "current final 5x5 base kernel embedded in 7x7 with zero outer shell"},
        )
    )

    edge_vars = ["30", "31", "32"]
    edge_classes, edge_meta = optimize_family(
        "edge_outer_no_33",
        start,
        edge_vars,
        fine_subset,
        direct_subset_obs,
        args.maxiter,
    )
    candidates.append(make_candidate("edge_outer_no33_eta_included", "edge_outer_no_33", edge_classes, edge_meta))

    no_corner_vars = ["10", "11", "20", "21", "22", "30", "31", "32"]
    no_corner_classes, no_corner_meta = optimize_family(
        "full_7x7_no_33",
        start,
        no_corner_vars,
        fine_subset,
        direct_subset_obs,
        args.maxiter,
    )
    candidates.append(make_candidate("full_7x7_no33_eta_included", "full_7x7_no_33", no_corner_classes, no_corner_meta))

    if args.include_full33:
        full_vars = ["10", "11", "20", "21", "22", "30", "31", "32", "33"]
        full_classes, full_meta = optimize_family(
            "full_7x7_with_33_penalized",
            start,
            full_vars,
            fine_subset,
            direct_subset_obs,
            max(8, args.maxiter // 2),
            include_33_penalty=True,
        )
        candidates.append(
            make_candidate("full_7x7_with33_penalized_eta_included", "full_7x7_with_33_penalized", full_classes, full_meta)
        )

    score_rows: list[dict[str, Any]] = []
    stability_rows: list[dict[str, Any]] = []
    observable_rows: list[dict[str, Any]] = []
    metric_by_candidate: dict[str, dict[str, dict[str, Any]]] = {}

    for cand in candidates:
        write_candidate(cand, args.candidate_dir)
        mom = momentum_extrema(cand.op_matrix, grid=512)
        blocked_full = block(fine_full, cand.op_matrix)
        blocked_obs = observable_arrays(blocked_full)
        full = full_metrics(direct_full_obs, blocked_obs)
        metric_by_candidate[cand.name] = full
        score = ranking_score(full)
        score_rows.append(
            {
                "candidate": cand.name,
                "family": cand.family,
                "score_lower_is_better": score,
                "phi2_ks": full["phi2"]["ks_statistic"],
                "phi2_js": full["phi2"]["jensen_shannon"],
                "phi2_shift": full["phi2"]["standardized_mean_shift"],
                "local_kurtosis_ks": full["local_kurtosis_ratio"]["ks_statistic"],
                "local_kurtosis_js": full["local_kurtosis_ratio"]["jensen_shannon"],
                "phi4_ks": full["phi4"]["ks_statistic"],
                "action_density_ks": full["action_density"]["ks_statistic"],
                "m2_ks": full["m2"]["ks_statistic"],
                "m4_ks": full["m4"]["ks_statistic"],
                "G_pmin_avg_ks": full["G_pmin_avg"]["ks_statistic"],
                "max_inverse_K": mom["max_inverse_K"],
                "min_K": mom["min_K"],
                "outer_30": cand.classes.get("30", 0.0),
                "outer_31": cand.classes.get("31", 0.0),
                "outer_32": cand.classes.get("32", 0.0),
                "outer_33": cand.classes.get("33", 0.0),
            }
        )
        stability_rows.append({"candidate": cand.name, "family": cand.family, **mom})
        for obs, row in full.items():
            observable_rows.append({"candidate": cand.name, "family": cand.family, "observable": obs, **row})
        for obs in ["phi2", "phi4", "local_kurtosis_ratio", "action_density", "m2", "m4", "G_pmin_avg"]:
            plot_histogram(
                direct_full_obs[obs],
                blocked_obs[obs],
                observable=obs,
                out_pdf=args.study_dir / "plots" / f"{cand.name}_{obs}.pdf",
                label_a="direct L16",
                label_b=f"{cand.name} blocked L32->L16",
            )

    score_rows.sort(key=lambda r: float(r["score_lower_is_better"]))
    write_observable_csv(args.study_dir / "candidate_scores.csv", score_rows)
    write_observable_csv(args.study_dir / "momentum_stability.csv", stability_rows)
    write_observable_csv(args.study_dir / "observable_histogram_summary.csv", observable_rows)

    best = score_rows[0]["candidate"]
    best_src = args.candidate_dir / f"{best}.json"
    best_dst = args.candidate_dir / "best_7x7_candidate_eta_included.json"
    best_dst.write_text(best_src.read_text())

    lines = [
        "# Lambda 1.0 7x7 Kernel Search",
        "",
        "Candidate search branch for improving the lambda=1.0 perfect-blocking kernel without overwriting the current final 5x5 kernel.",
        "",
        "## Families Tested",
        "",
        "- `embedded_5x5_baseline`: current 5x5 base kernel embedded in 7x7, outer shell zero.",
        "- `edge_outer_no_33`: only `(3,0)`, `(3,1)`, `(3,2)` outer classes are varied; `(3,3)` excluded.",
        "- `full_7x7_no_33`: 5x5 classes and `(3,0)`, `(3,1)`, `(3,2)` varied; `(3,3)` excluded.",
        "- `full_7x7_with_33_penalized`: optional full 7x7 including `(3,3)`, with an explicit far-corner penalty. "
        + ("This optional family was run." if args.include_full33 else "This optional family was not run in this pass."),
        "",
        f"Best candidate by full-sample ranking score: `{best}`.",
        "",
        "## Candidate Scores",
        "",
        "| candidate | family | score | phi2 KS | phi2 JS | local kurtosis KS | action KS | m2 KS | m4 KS | Gpmin KS | max 1/K | min K |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in score_rows:
        lines.append(
            "| {candidate} | {family} | {score_lower_is_better:.6g} | {phi2_ks:.6g} | {phi2_js:.6g} | {local_kurtosis_ks:.6g} | {action_density_ks:.6g} | {m2_ks:.6g} | {m4_ks:.6g} | {G_pmin_avg_ks:.6g} | {max_inverse_K:.6g} | {min_K:.6g} |".format(
                **{k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in row.items()}
            )
        )
    baseline = next(r for r in score_rows if r["candidate"] == "embedded_5x5_control_eta_included")
    best_row = score_rows[0]
    phi2_improved = float(best_row["phi2_ks"]) < 0.75 * float(baseline["phi2_ks"]) or float(best_row["phi2_js"]) < 0.75 * float(baseline["phi2_js"])
    local_ok = float(best_row["local_kurtosis_ks"]) <= float(baseline["local_kurtosis_ks"])
    action_ok = float(best_row["action_density_ks"]) <= max(0.035, 1.15 * float(baseline["action_density_ks"]))
    stable = float(best_row["min_K"]) > 0.0 and float(best_row["max_inverse_K"]) < 2.0
    recommendation = (
        "consider promoting the best 7x7 candidate after review"
        if phi2_improved and local_ok and action_ok and stable
        else "keep the current 5x5 final kernel for now"
    )
    lines += [
        "",
        "## Recommendation",
        "",
        f"`{recommendation}`.",
        "",
        f"- phi2 improved by acceptance criterion: `{phi2_improved}`",
        f"- local kurtosis not worsened: `{local_ok}`",
        f"- action-density guardrail passed: `{action_ok}`",
        f"- momentum stability passed: `{stable}`",
        "",
        "The current final 5x5 kernel was not overwritten.",
    ]
    (args.study_dir / "README.md").write_text("\n".join(lines) + "\n")
    (args.study_dir / "candidate_scores.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"best": best, "recommendation": recommendation, "study_dir": str(args.study_dir)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
