#!/usr/bin/env python3
"""Controlled lambda=1.0 7x7 retraining with phi2 protected."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.blocking import load_configs  # noqa: E402
from scripts.common.histogram_compare import plot_histogram  # noqa: E402
from scripts.common.kernel_io import load_kernel  # noqa: E402
from scripts.run_lam1p0_7x7_kernel_search import (  # noqa: E402
    CLASS_MULT,
    ETA_SCALE,
    block,
    classes_from_matrix,
    full_metrics,
    make_candidate,
    matrix_from_classes,
    momentum_extrema,
    observable_arrays,
    write_candidate,
)


LAM_ROOT = PROJECT_ROOT / "perfect_blocking/perfect_blocking_lam1p0"
OUT = LAM_ROOT / "tests/intermediate/7x7_full_retraining_phi2_constrained"
CAND_DIR = LAM_ROOT / "kernels/candidates/7x7_full_retraining_phi2_constrained"
DIRECT = PROJECT_ROOT / "data/configs_phi4_2d/lam1p0_kappac0p340301_L16/configs.npz"
FINE = PROJECT_ROOT / "data/configs_phi4_2d/lam1p0_kappac0p340301_L32/configs.npz"
CURRENT_FINAL = LAM_ROOT / "kernels/final/chosen_kernel.json"
RETRAINED_5X5 = LAM_ROOT / "kernels/candidates/systematic_training/best_retrained_5x5_full_objective_eta_included.json"
UNCONSTRAINED_7X7 = LAM_ROOT / "kernels/candidates/7x7_from_retrained_5x5/best_7x7_from_retrained_5x5_eta_included.json"

NO33_KEYS = ["10", "11", "20", "21", "22", "30", "31", "32"]
FULL33_KEYS = [*NO33_KEYS, "33"]
KEY_OBS = ["action_density", "phi2", "phi4", "local_kurtosis_ratio", "NN", "2nn", "diag", "m2", "m4", "G_pmin_avg"]
PLOT_OBS = ["action_density", "phi2", "phi4", "local_kurtosis_ratio", "NN", "m2", "m4", "G_pmin_avg"]
ALL_OBS = [
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


def load_matrix(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    data = json.loads(path.read_text())
    if "matrix" in data and bool(data.get("kernel_coefficients_include_eta_scale")):
        return np.asarray(data["matrix"], dtype=np.float64), data
    spec = load_kernel(path)
    return spec.matrix, spec.metadata


def classes_from_eta_matrix(matrix: np.ndarray) -> dict[str, float]:
    classes = {}
    for key, value in classes_from_matrix(matrix / ETA_SCALE).items():
        classes[key] = float(value)
    for key in FULL33_KEYS:
        classes.setdefault(key, 0.0)
    return normalize(classes, include_33=True)


def normalize(classes: dict[str, float], include_33: bool) -> dict[str, float]:
    out = dict(classes)
    if not include_33:
        out["33"] = 0.0
    for key in FULL33_KEYS:
        out.setdefault(key, 0.0)
    rest = sum(CLASS_MULT[k] * float(v) for k, v in out.items() if k != "00")
    out["00"] = 1.0 - rest
    return out


def random_classes(rng: np.random.Generator, center: dict[str, float] | None, scale: float, include_33: bool) -> dict[str, float]:
    cls = dict(center or {})
    keys = FULL33_KEYS if include_33 else NO33_KEYS
    for key in keys:
        base = float(cls.get(key, 0.0))
        cls[key] = base + float(rng.normal(0.0, scale))
    if include_33:
        cls["33"] *= 0.25
    return normalize(cls, include_33=include_33)


def scalar(row: dict[str, Any]) -> float:
    return float(row["ks_statistic"]) + 2.0 * float(row["total_variation"]) + 20.0 * float(row["jensen_shannon"]) + 0.2 * abs(float(row["standardized_mean_shift"]))


def local_score(rows: dict[str, dict[str, Any]]) -> float:
    return 4.0 * scalar(rows["phi2"]) + 3.0 * scalar(rows["phi4"]) + 3.0 * scalar(rows["local_kurtosis_ratio"])


def guardrail_penalty(rows: dict[str, dict[str, Any]], mom: dict[str, float], cls: dict[str, float], reference: dict[str, dict[str, Any]]) -> float:
    penalty = 0.0
    for obs, limit in {
        "action_density": 0.035,
        "phi2": min(0.083, float(reference["phi2"]["ks_statistic"]) + 0.002),
        "NN": 0.035,
        "m2": 0.025,
        "m4": 0.025,
        "G_pmin_avg": 0.035,
    }.items():
        penalty += 8.0e4 * max(0.0, float(rows[obs]["ks_statistic"]) - limit) ** 2
    penalty += 8.0e4 * max(0.0, float(rows["action_density"]["jensen_shannon"]) - 0.005) ** 2
    penalty += 4.0e5 * max(0.0, float(rows["phi2"]["ks_statistic"]) - min(0.083, float(reference["phi2"]["ks_statistic"]) + 0.002)) ** 2
    penalty += 1.4e5 * max(0.0, float(rows["NN"]["ks_statistic"]) - 0.035) ** 2
    penalty += 1.2e6 * max(0.0, float(rows["NN"]["ks_statistic"]) - 0.040) ** 2
    penalty += 7.5e4 * max(0.0, mom["max_inverse_K"] - 1.6) ** 2
    penalty += 1.2e6 * max(0.0, -mom["min_K"]) ** 2
    outer_l2 = 4.0 * cls.get("30", 0.0) ** 2 + 8.0 * cls.get("31", 0.0) ** 2 + 8.0 * cls.get("32", 0.0) ** 2 + 30.0 * cls.get("33", 0.0) ** 2
    penalty += 100.0 * outer_l2
    return float(penalty)


def total_score(rows: dict[str, dict[str, Any]], mom: dict[str, float], cls: dict[str, float], reference: dict[str, dict[str, Any]]) -> float:
    return float(local_score(rows) + guardrail_penalty(rows, mom, cls, reference))


def category(rows: dict[str, dict[str, Any]], mom: dict[str, float], reference: dict[str, dict[str, Any]], candidate: str) -> str:
    if candidate.startswith("baseline_"):
        return "baseline"
    passes = (
        float(rows["action_density"]["ks_statistic"]) <= 0.035
        and float(rows["phi2"]["ks_statistic"]) <= min(0.083, float(reference["phi2"]["ks_statistic"]) + 0.002)
        and float(rows["NN"]["ks_statistic"]) <= 0.035
        and float(rows["m2"]["ks_statistic"]) <= 0.025
        and float(rows["m4"]["ks_statistic"]) <= 0.025
        and float(rows["G_pmin_avg"]["ks_statistic"]) <= 0.035
        and mom["min_K"] > 0.0
        and mom["max_inverse_K"] <= 1.6
    )
    improves = (
        float(rows["phi2"]["ks_statistic"]) < float(reference["phi2"]["ks_statistic"])
        and float(rows["phi4"]["ks_statistic"]) <= float(reference["phi4"]["ks_statistic"])
        and float(rows["local_kurtosis_ratio"]["ks_statistic"]) <= float(reference["local_kurtosis_ratio"]["ks_statistic"])
        and local_score(rows) < local_score(reference)
    )
    if passes and improves:
        return "promotion-worthy"
    if passes:
        return "valid but not better"
    return "diagnostic only"


def evaluate(name: str, family: str, cls: dict[str, float], direct_obs: dict[str, np.ndarray], fine: np.ndarray, reference: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, float], np.ndarray]:
    matrix = ETA_SCALE * matrix_from_classes(cls)
    rows = full_metrics(direct_obs, observable_arrays(block(fine, matrix)))
    mom = momentum_extrema(matrix, grid=512)
    rec: dict[str, Any] = {
        "candidate": name,
        "family": family,
        "sum_K": float(matrix.sum()),
        "sum_base": float((matrix / ETA_SCALE).sum()),
        "local_score": local_score(rows),
        "constrained_score": total_score(rows, mom, cls, reference),
        "min_K": mom["min_K"],
        "max_K": mom["max_K"],
        "min_inverse_K": mom["min_inverse_K"],
        "max_inverse_K": mom["max_inverse_K"],
        "condition_number": mom["max_K"] / mom["min_K"],
        "eta_included_sum_ok": math.isclose(float(matrix.sum()), ETA_SCALE, rel_tol=1.0e-12, abs_tol=1.0e-12),
    }
    for obs in KEY_OBS:
        rec[f"{obs}_KS"] = rows[obs]["ks_statistic"]
        rec[f"{obs}_JS"] = rows[obs]["jensen_shannon"]
        rec[f"{obs}_TV"] = rows[obs]["total_variation"]
        rec[f"{obs}_shift"] = rows[obs]["standardized_mean_shift"]
    return rec, rows, mom, matrix


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "plots").mkdir(parents=True, exist_ok=True)
    CAND_DIR.mkdir(parents=True, exist_ok=True)

    direct = load_configs(DIRECT)
    fine = load_configs(FINE)
    direct_obs = observable_arrays(direct)
    fine_sub = fine[:1400]
    direct_sub_obs = observable_arrays(direct[:1400])

    current_matrix, current_meta = load_matrix(CURRENT_FINAL)
    retrained_matrix, retrained_meta = load_matrix(RETRAINED_5X5)
    previous_matrix, previous_meta = load_matrix(UNCONSTRAINED_7X7)
    current_classes = classes_from_eta_matrix(current_matrix)
    retrained_classes = classes_from_eta_matrix(retrained_matrix)
    previous_classes = classes_from_eta_matrix(previous_matrix)
    current_reference = full_metrics(direct_obs, observable_arrays(block(fine, current_matrix)))
    current_reference_sub = full_metrics(direct_sub_obs, observable_arrays(block(fine_sub, current_matrix)))

    rng = np.random.default_rng(20260721)
    stage_rows: list[dict[str, Any]] = []

    broad_specs: list[tuple[str, str, dict[str, float], dict[str, Any]]] = []
    starts = [current_classes, retrained_classes, previous_classes, None]
    for start_idx, center in enumerate(starts):
        for i in range(260):
            scale = 0.0025 if center is not None else 0.010
            cls = random_classes(rng, center, scale, include_33=False)
            broad_specs.append((f"stage1_no33_start{start_idx}_{i:04d}", "stage1_broad_no33", cls, {"stage": 1, "start": start_idx, "scale": scale}))

    screen: list[tuple[float, str, str, dict[str, float], dict[str, Any]]] = []
    for name, family, cls, meta in broad_specs:
        matrix = ETA_SCALE * matrix_from_classes(cls)
        rows = full_metrics(direct_sub_obs, observable_arrays(block(fine_sub, matrix)))
        mom = momentum_extrema(matrix, grid=192)
        score = total_score(rows, mom, cls, current_reference_sub)
        screen.append((score, name, family, cls, meta))
        stage_rows.append({"stage": 1, "candidate": name, "family": family, "subset_score": score, "NN_KS": rows["NN"]["ks_statistic"], "phi2_KS": rows["phi2"]["ks_statistic"], "phi4_KS": rows["phi4"]["ks_statistic"], "kurt_KS": rows["local_kurtosis_ratio"]["ks_statistic"], "action_KS": rows["action_density"]["ks_statistic"], "max_inverse_K": mom["max_inverse_K"]})
    screen.sort(key=lambda item: item[0])

    local_specs: list[tuple[str, str, dict[str, float], dict[str, Any]]] = []
    for center_idx, (_score, _name, _family, center, _meta) in enumerate(screen[:30]):
        for i in range(28):
            scale = 0.0008 if i < 18 else 0.0016
            cls = random_classes(rng, center, scale, include_33=False)
            local_specs.append((f"stage2_no33_center{center_idx:02d}_{i:03d}", "stage2_local_no33", cls, {"stage": 2, "center": center_idx, "scale": scale}))

    local_screen: list[tuple[float, str, str, dict[str, float], dict[str, Any]]] = []
    for name, family, cls, meta in local_specs:
        matrix = ETA_SCALE * matrix_from_classes(cls)
        rows = full_metrics(direct_sub_obs, observable_arrays(block(fine_sub, matrix)))
        mom = momentum_extrema(matrix, grid=192)
        score = total_score(rows, mom, cls, current_reference_sub)
        local_screen.append((score, name, family, cls, meta))
        stage_rows.append({"stage": 2, "candidate": name, "family": family, "subset_score": score, "NN_KS": rows["NN"]["ks_statistic"], "phi2_KS": rows["phi2"]["ks_statistic"], "phi4_KS": rows["phi4"]["ks_statistic"], "kurt_KS": rows["local_kurtosis_ratio"]["ks_statistic"], "action_KS": rows["action_density"]["ks_statistic"], "max_inverse_K": mom["max_inverse_K"]})
    local_screen.sort(key=lambda item: item[0])

    full_specs = screen[:35] + local_screen[:85]
    evaluated: list[tuple[dict[str, Any], dict[str, float], dict[str, dict[str, Any]], dict[str, float], np.ndarray, dict[str, Any]]] = []
    baselines = [
        ("baseline_retrained_5x5", "baseline", retrained_classes, retrained_meta),
        ("baseline_current_final_nn_constrained_7x7", "baseline", current_classes, current_meta),
        ("baseline_previous_unconstrained_7x7", "baseline", previous_classes, previous_meta),
    ]
    for name, family, cls, meta in baselines + [(name, family, cls, meta) for _score, name, family, cls, meta in full_specs]:
        rec, rows, mom, matrix = evaluate(name, family, cls, direct_obs, fine, current_reference)
        rec["category"] = category(rows, mom, current_reference, name)
        evaluated.append((rec, cls, rows, mom, matrix, meta))

    no33_best = next((item for item in sorted(evaluated, key=lambda item: float(item[0]["constrained_score"])) if item[0]["category"] == "promotion-worthy"), None)
    full33_specs: list[tuple[float, str, str, dict[str, float], dict[str, Any]]] = []
    if no33_best is None:
        centers = [item[1] for item in sorted(evaluated, key=lambda item: float(item[0]["constrained_score"]))[:20]]
        for center_idx, center in enumerate(centers):
            for i in range(18):
                cls = random_classes(rng, center, 0.0009, include_33=True)
                matrix = ETA_SCALE * matrix_from_classes(cls)
                rows = full_metrics(direct_sub_obs, observable_arrays(block(fine_sub, matrix)))
                mom = momentum_extrema(matrix, grid=192)
                score = total_score(rows, mom, cls, current_reference_sub)
                full33_specs.append((score, f"stage3_full33_center{center_idx:02d}_{i:03d}", "stage3_full33_penalized", cls, {"stage": 3, "center": center_idx}))
                stage_rows.append({"stage": 3, "candidate": f"stage3_full33_center{center_idx:02d}_{i:03d}", "family": "stage3_full33_penalized", "subset_score": score, "NN_KS": rows["NN"]["ks_statistic"], "phi2_KS": rows["phi2"]["ks_statistic"], "phi4_KS": rows["phi4"]["ks_statistic"], "kurt_KS": rows["local_kurtosis_ratio"]["ks_statistic"], "action_KS": rows["action_density"]["ks_statistic"], "max_inverse_K": mom["max_inverse_K"]})
        full33_specs.sort(key=lambda item: item[0])
        for _score, name, family, cls, meta in full33_specs[:45]:
            rec, rows, mom, matrix = evaluate(name, family, cls, direct_obs, fine, current_reference)
            rec["category"] = category(rows, mom, current_reference, name)
            evaluated.append((rec, cls, rows, mom, matrix, meta))

    candidate_rows = [item[0] for item in evaluated]
    candidate_rows.sort(key=lambda row: (row["category"] != "promotion-worthy", float(row["constrained_score"])))
    hist_rows: list[dict[str, Any]] = []
    mom_rows: list[dict[str, Any]] = []
    outer_rows: list[dict[str, Any]] = []
    for rec, cls, rows, mom, _matrix, _meta in evaluated:
        for obs in ALL_OBS:
            hist_rows.append({"candidate": rec["candidate"], "family": rec["family"], "observable": obs, **rows[obs]})
        mom_rows.append({"candidate": rec["candidate"], "family": rec["family"], **mom, "condition_number": mom["max_K"] / mom["min_K"]})
        outer_rows.append(
            {
                "candidate": rec["candidate"],
                "family": rec["family"],
                "K30_base": cls.get("30", 0.0),
                "K31_base": cls.get("31", 0.0),
                "K32_base": cls.get("32", 0.0),
                "K33_base": cls.get("33", 0.0),
                "K30_eta_included": ETA_SCALE * cls.get("30", 0.0),
                "K31_eta_included": ETA_SCALE * cls.get("31", 0.0),
                "K32_eta_included": ETA_SCALE * cls.get("32", 0.0),
                "K33_eta_included": ETA_SCALE * cls.get("33", 0.0),
            }
        )
    write_csv(OUT / "stage_subset_screen.csv", stage_rows)
    write_csv(OUT / "candidate_scores.csv", candidate_rows)
    write_csv(OUT / "full_histogram_metrics.csv", hist_rows)
    write_csv(OUT / "momentum_stability.csv", mom_rows)
    write_csv(OUT / "outer_shell_coefficients.csv", outer_rows)

    best = next((item for item in sorted(evaluated, key=lambda item: float(item[0]["constrained_score"])) if item[0]["category"] == "promotion-worthy"), None)
    best_path = None
    best_name = "none"
    if best is not None:
        rec, cls, _rows, _mom, _matrix, meta = best
        best_name = str(rec["candidate"])
        cand = make_candidate(
            "best_7x7_full_retraining_controlled_eta_included",
            str(rec["family"]),
            cls,
            {
                "source_candidate": best_name,
            "selection": "best staged controlled from-scratch 7x7 candidate with phi2 protected as a hard guardrail",
                "baseline": str(CURRENT_FINAL),
                "K33_policy": "no33 in stages 1-2; stage 3 full33 only if no33 failed to beat current final",
                **{k: v for k, v in meta.items() if isinstance(v, (str, int, float, bool))},
            },
        )
        write_candidate(cand, CAND_DIR)
        best_path = CAND_DIR / "best_7x7_full_retraining_controlled_eta_included.json"

    plot_names = {"baseline_retrained_5x5", "baseline_current_final_nn_constrained_7x7"}
    if best is not None:
        plot_names.add(best_name)
    for rec, _cls, _rows, _mom, matrix, _meta in evaluated:
        if rec["candidate"] in plot_names:
            blocked_obs = observable_arrays(block(fine, matrix))
            for obs in PLOT_OBS:
                plot_histogram(
                    direct_obs[obs],
                    blocked_obs[obs],
                    obs,
                    OUT / "plots" / f"{rec['candidate']}_{obs}.pdf",
                    label_a="direct L16",
                    label_b=f"{rec['candidate']} blocked L32->L16",
                )

    lines = [
        "# Lambda 1.0 Controlled Full 7x7 Retraining With Phi2 Constraint",
        "",
        "This study retrains the symmetric 7x7 orbit classes without keeping the retrained 5x5 core fixed. Stages 1 and 2 exclude `(3,3)`; stage 3 allows penalized `(3,3)` only if no no-corner candidate beats the current final. This pass adds a hard phi2 guardrail: `phi2 KS <= min(0.083, current_final_phi2_KS + 0.002)`.",
        "",
        f"Best promotion-worthy candidate: `{best_name}`",
        f"Saved candidate path: `{best_path if best_path else 'none'}`",
        "",
        "| candidate | category | family | score | action KS | phi2 KS | phi4 KS | kurt KS | NN KS | 2nn KS | diag KS | m2 KS | m4 KS | Gpmin KS | max 1/K | K33 base |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    outer_by = {row["candidate"]: row for row in outer_rows}
    for row in candidate_rows[:30]:
        outer = outer_by[row["candidate"]]
        lines.append(
            f"| {row['candidate']} | {row['category']} | {row['family']} | {float(row['constrained_score']):.6g} | "
            f"{float(row['action_density_KS']):.6g} | {float(row['phi2_KS']):.6g} | {float(row['phi4_KS']):.6g} | "
            f"{float(row['local_kurtosis_ratio_KS']):.6g} | {float(row['NN_KS']):.6g} | {float(row['2nn_KS']):.6g} | "
            f"{float(row['diag_KS']):.6g} | {float(row['m2_KS']):.6g} | {float(row['m4_KS']):.6g} | "
            f"{float(row['G_pmin_avg_KS']):.6g} | {float(row['max_inverse_K']):.6g} | {float(outer['K33_base']):.6g} |"
        )
    if best is None:
        lines.extend(["", "No from-scratch 7x7 candidate beat the current NN-constrained final under the configured guardrails."])
    else:
        lines.extend(["", "A from-scratch 7x7 candidate beat the current final under the configured guardrails. It was saved but not promoted."])
    (OUT / "candidate_scores.md").write_text("\n".join(lines) + "\n")
    (OUT / "recommendation.md").write_text("\n".join(lines) + "\n")

    print(json.dumps({"out": str(OUT), "best_candidate": str(best_path) if best_path else None}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
