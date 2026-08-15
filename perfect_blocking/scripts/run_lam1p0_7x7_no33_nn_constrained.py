#!/usr/bin/env python3
"""NN-constrained lambda=1.0 7x7 no-corner refinement."""

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
from scripts.common.histogram_compare import metrics, plot_histogram  # noqa: E402
from scripts.common.kernel_io import load_kernel  # noqa: E402
from scripts.run_lam1p0_7x7_kernel_search import (  # noqa: E402
    CLASS_MULT,
    ETA_SCALE,
    Candidate,
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
OUT = LAM_ROOT / "tests/intermediate/7x7_no33_nn_constrained"
CAND_DIR = LAM_ROOT / "kernels/candidates/7x7_no33_nn_constrained"
DIRECT = PROJECT_ROOT / "data/configs_phi4_2d/lam1p0_kappac0p340301_L16/configs.npz"
FINE = PROJECT_ROOT / "data/configs_phi4_2d/lam1p0_kappac0p340301_L32/configs.npz"
OLD_FINAL = LAM_ROOT / "kernels/final/chosen_kernel.json"
RETRAINED = LAM_ROOT / "kernels/candidates/systematic_training/best_retrained_5x5_full_objective_eta_included.json"
PREVIOUS_7X7 = LAM_ROOT / "kernels/candidates/7x7_from_retrained_5x5/best_7x7_from_retrained_5x5_eta_included.json"

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


def base_classes_from_eta_matrix(matrix: np.ndarray) -> dict[str, float]:
    classes = classes_from_matrix(matrix / ETA_SCALE)
    for key in ("30", "31", "32", "33"):
        classes.setdefault(key, 0.0)
    classes["33"] = 0.0
    return normalize(classes)


def normalize(classes: dict[str, float]) -> dict[str, float]:
    out = dict(classes)
    out["33"] = 0.0
    rest = sum(CLASS_MULT[k] * v for k, v in out.items() if k != "00")
    out["00"] = 1.0 - rest
    return out


def scalar_score(row: dict[str, Any]) -> float:
    return (
        float(row["ks_statistic"])
        + 2.0 * float(row["total_variation"])
        + 20.0 * float(row["jensen_shannon"])
        + 0.2 * abs(float(row["standardized_mean_shift"]))
    )


def local_score(rows: dict[str, dict[str, Any]]) -> float:
    return 4.0 * scalar_score(rows["phi2"]) + 3.0 * scalar_score(rows["phi4"]) + 3.0 * scalar_score(rows["local_kurtosis_ratio"])


def constrained_score(rows: dict[str, dict[str, Any]], mom: dict[str, float], retrained_rows: dict[str, dict[str, Any]], base_matrix: np.ndarray, matrix: np.ndarray) -> float:
    score = local_score(rows)
    penalty = 0.0
    hard_limits = {
        "action_density": 0.035,
        "NN": 0.035,
        "m2": 0.025,
        "m4": 0.025,
        "G_pmin_avg": 0.035,
    }
    for obs, limit in hard_limits.items():
        penalty += 5.0e4 * max(0.0, float(rows[obs]["ks_statistic"]) - limit) ** 2
    penalty += 5.0e4 * max(0.0, float(rows["action_density"]["jensen_shannon"]) - 0.005) ** 2
    penalty += 1.2e5 * max(0.0, float(rows["NN"]["ks_statistic"]) - 0.035) ** 2
    penalty += 1.0e6 * max(0.0, float(rows["NN"]["ks_statistic"]) - 0.040) ** 2
    penalty += 5.0e4 * max(0.0, mom["max_inverse_K"] - 1.6) ** 2
    penalty += 1.0e6 * max(0.0, -mom["min_K"]) ** 2
    penalty += 800.0 * float(np.mean((matrix - base_matrix) ** 2))
    # Discourage candidates that lose the retrained 5x5's already good action sector.
    penalty += 2.5e4 * max(0.0, float(rows["action_density"]["ks_statistic"]) - float(retrained_rows["action_density"]["ks_statistic"]) - 0.005) ** 2
    return float(score + penalty)


def category(rows: dict[str, dict[str, Any]], mom: dict[str, float], retrained_rows: dict[str, dict[str, Any]], candidate_name: str) -> str:
    improves_local = (
        float(rows["phi2"]["ks_statistic"]) < float(retrained_rows["phi2"]["ks_statistic"])
        and float(rows["phi4"]["ks_statistic"]) < float(retrained_rows["phi4"]["ks_statistic"])
        and float(rows["local_kurtosis_ratio"]["ks_statistic"]) <= float(retrained_rows["local_kurtosis_ratio"]["ks_statistic"])
    )
    passes = (
        float(rows["NN"]["ks_statistic"]) <= 0.035
        and float(rows["action_density"]["ks_statistic"]) <= 0.035
        and float(rows["m2"]["ks_statistic"]) <= 0.025
        and float(rows["m4"]["ks_statistic"]) <= 0.025
        and float(rows["G_pmin_avg"]["ks_statistic"]) <= 0.035
        and mom["min_K"] > 0.0
        and mom["max_inverse_K"] <= 1.6
    )
    if candidate_name in {"old_final_5x5", "retrained_5x5", "previous_best_7x7_no33"}:
        return "baseline"
    if passes and improves_local:
        return "promotion-worthy"
    if improves_local:
        return "diagnostic only"
    return "not useful"


def evaluate(name: str, family: str, matrix: np.ndarray, direct_obs: dict[str, np.ndarray], fine: np.ndarray) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, float]]:
    blocked = observable_arrays(block(fine, matrix))
    rows = full_metrics(direct_obs, blocked)
    mom = momentum_extrema(matrix, grid=512)
    rec: dict[str, Any] = {
        "candidate": name,
        "family": family,
        "sum_K": float(matrix.sum()),
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
        rec[f"{obs}_shift"] = rows[obs]["standardized_mean_shift"]
    return rec, rows, mom


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "plots").mkdir(parents=True, exist_ok=True)
    CAND_DIR.mkdir(parents=True, exist_ok=True)

    direct = load_configs(DIRECT)
    fine = load_configs(FINE)
    direct_obs = observable_arrays(direct)

    old_matrix, old_meta = load_matrix(OLD_FINAL)
    retrained_matrix, retrained_meta = load_matrix(RETRAINED)
    previous_matrix, previous_meta = load_matrix(PREVIOUS_7X7)

    base_classes = base_classes_from_eta_matrix(retrained_matrix)
    previous_classes = base_classes_from_eta_matrix(previous_matrix)
    retrained_rows = full_metrics(direct_obs, observable_arrays(block(fine, retrained_matrix)))

    candidate_specs: list[tuple[str, str, dict[str, float], dict[str, Any]]] = []
    for alpha in np.linspace(0.0, 1.0, 101):
        cls = {key: (1.0 - alpha) * base_classes.get(key, 0.0) + alpha * previous_classes.get(key, 0.0) for key in set(base_classes) | set(previous_classes)}
        cls = normalize(cls)
        candidate_specs.append((f"blend_retrained_to_prev7x7_alpha_{alpha:.3f}".replace(".", "p"), "blend_scan", cls, {"blend_alpha": float(alpha)}))

    rng = np.random.default_rng(20260720)
    local_keys = ["10", "11", "20", "21", "22", "30", "31", "32"]
    for alpha, count in [(0.35, 180), (0.45, 260), (0.55, 180)]:
        center = {key: (1.0 - alpha) * base_classes.get(key, 0.0) + alpha * previous_classes.get(key, 0.0) for key in set(base_classes) | set(previous_classes)}
        center = normalize(center)
        for i in range(count):
            scale = 0.0006 if i < count // 2 else 0.0012
            cls = dict(center)
            for key in local_keys:
                cls[key] = float(cls.get(key, 0.0) + rng.normal(0.0, scale))
            cls = normalize(cls)
            candidate_specs.append((f"local_no33_alpha_{alpha:.2f}_{i:04d}".replace(".", "p"), "local_no33_perturbation", cls, {"center_alpha": alpha, "perturb_scale": scale}))

    # Screen local perturbations on a smaller subset; keep all blend points plus the best local perturbations.
    fine_sub = fine[:1400]
    direct_sub_obs = observable_arrays(direct[:1400])
    retrained_sub_rows = full_metrics(direct_sub_obs, observable_arrays(block(fine_sub, retrained_matrix)))
    screen_rows: list[tuple[float, str, str, dict[str, float], dict[str, Any]]] = []
    blend_rows: list[tuple[float, str, str, dict[str, float], dict[str, Any]]] = []
    for name, family, cls, meta in candidate_specs:
        matrix = ETA_SCALE * matrix_from_classes(cls)
        rows = full_metrics(direct_sub_obs, observable_arrays(block(fine_sub, matrix)))
        mom = momentum_extrema(matrix, grid=192)
        score = constrained_score(rows, mom, retrained_sub_rows, retrained_matrix, matrix)
        item = (score, name, family, cls, meta)
        if family == "blend_scan":
            blend_rows.append(item)
        else:
            screen_rows.append(item)
    screen_rows.sort(key=lambda item: item[0])

    full_specs = blend_rows + screen_rows[:120]
    evaluated: list[tuple[dict[str, Any], dict[str, float], np.ndarray, dict[str, dict[str, Any]], dict[str, float], dict[str, Any]]] = []
    baselines = [
        ("old_final_5x5", "baseline", base_classes_from_eta_matrix(old_matrix), old_meta),
        ("retrained_5x5", "baseline", base_classes, retrained_meta),
        ("previous_best_7x7_no33", "baseline", previous_classes, previous_meta),
    ]
    for name, family, cls, meta in baselines + [(name, family, cls, meta) for _, name, family, cls, meta in full_specs]:
        matrix = ETA_SCALE * matrix_from_classes(cls)
        rec, rows, mom = evaluate(name, family, matrix, direct_obs, fine)
        rec["constrained_score"] = constrained_score(rows, mom, retrained_rows, retrained_matrix, matrix)
        rec["local_score"] = local_score(rows)
        rec["category"] = category(rows, mom, retrained_rows, name)
        rec["eta_included_sum_ok"] = math.isclose(float(matrix.sum()), ETA_SCALE, rel_tol=1.0e-12, abs_tol=1.0e-12)
        evaluated.append((rec, cls, matrix, rows, mom, meta))

    candidate_rows = [item[0] for item in evaluated]
    candidate_rows.sort(key=lambda row: (row["category"] != "promotion-worthy", float(row["constrained_score"])))

    hist_rows: list[dict[str, Any]] = []
    mom_rows: list[dict[str, Any]] = []
    outer_rows: list[dict[str, Any]] = []
    for rec, cls, matrix, rows, mom, _meta in evaluated:
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

    blend_scan = [row for row in candidate_rows if row["family"] == "blend_scan"]
    write_csv(OUT / "candidate_scores.csv", candidate_rows)
    write_csv(OUT / "full_histogram_metrics.csv", hist_rows)
    write_csv(OUT / "momentum_stability.csv", mom_rows)
    write_csv(OUT / "outer_shell_coefficients.csv", outer_rows)
    write_csv(OUT / "blend_scan.csv", sorted(blend_scan, key=lambda row: str(row["candidate"])))

    best_tuple = next((item for item in sorted(evaluated, key=lambda item: float(item[0]["constrained_score"])) if item[0]["category"] == "promotion-worthy"), None)
    best_path = None
    best_name = "none"
    if best_tuple is not None:
        rec, cls, _matrix, _rows, _mom, meta = best_tuple
        best_name = str(rec["candidate"])
        cand = make_candidate(
            "best_7x7_no33_nn_constrained_eta_included",
            str(rec["family"]),
            cls,
            {
                "source_candidate": best_name,
                "selection": "best lambda=1.0 7x7 no-corner candidate with NN protected as a hard local-operator guardrail",
                "NN_guardrail": "promotion-worthy requires NN KS <= 0.035; NN KS > 0.040 is rejected",
                "baseline": str(RETRAINED),
                "previous_7x7": str(PREVIOUS_7X7),
                **{k: v for k, v in meta.items() if isinstance(v, (str, int, float, bool))},
            },
        )
        write_candidate(cand, CAND_DIR)
        src_json = CAND_DIR / "best_7x7_no33_nn_constrained_eta_included.json"
        src_txt = CAND_DIR / "best_7x7_no33_nn_constrained_eta_included.txt"
        src_meta = CAND_DIR / "best_7x7_no33_nn_constrained_eta_included_metadata.json"
        best_path = src_json
        # User requested the shorter txt/metadata names as well.
        (CAND_DIR / "best_7x7_no33_nn_constrained.txt").write_text(src_txt.read_text())
        (CAND_DIR / "best_7x7_no33_nn_constrained_metadata.json").write_text(src_meta.read_text())

    plot_candidates = {"retrained_5x5", "previous_best_7x7_no33"}
    if best_tuple is not None:
        plot_candidates.add(best_name)
    for rec, _cls, matrix, _rows, _mom, _meta in evaluated:
        if rec["candidate"] in plot_candidates:
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

    md_lines = [
        "# Lambda 1.0 7x7 no33 NN-Constrained Refinement",
        "",
        "This pass protects the nearest-neighbor local operator as a hard guardrail. The previous 7x7 is kept as diagnostic only if its NN KS remains above the guardrail.",
        "",
        "## Top Candidates",
        "",
        "| candidate | category | family | score | action KS | phi2 KS | phi4 KS | kurt KS | NN KS | 2nn KS | diag KS | m2 KS | m4 KS | Gpmin KS | max 1/K |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in candidate_rows[:25]:
        md_lines.append(
            f"| {row['candidate']} | {row['category']} | {row['family']} | {float(row['constrained_score']):.6g} | "
            f"{float(row['action_density_KS']):.6g} | {float(row['phi2_KS']):.6g} | {float(row['phi4_KS']):.6g} | "
            f"{float(row['local_kurtosis_ratio_KS']):.6g} | {float(row['NN_KS']):.6g} | {float(row['2nn_KS']):.6g} | "
            f"{float(row['diag_KS']):.6g} | {float(row['m2_KS']):.6g} | {float(row['m4_KS']):.6g} | "
            f"{float(row['G_pmin_avg_KS']):.6g} | {float(row['max_inverse_K']):.6g} |"
        )
    md_lines.extend(
        [
            "",
            "## Recommendation",
            "",
            f"Best promotion-worthy candidate: `{best_name}`",
            f"Saved candidate path: `{best_path if best_path else 'none'}`",
            "",
        ]
    )
    if best_tuple is None:
        md_lines.append("No NN-constrained 7x7 candidate passed the promotion-worthy guardrails. Keep the retrained 5x5, or run a broader Pareto search.")
    else:
        rec = best_tuple[0]
        md_lines.append(
            "A no-corner 7x7 candidate passed the NN guardrail while preserving much of the phi2/phi4/local-kurtosis improvement. "
            "It was not promoted."
        )
        md_lines.append("")
        md_lines.append(f"- eta-included sum(K): `{float(rec['sum_K']):.17g}`")
        md_lines.append(f"- max 1/K(p): `{float(rec['max_inverse_K']):.6g}`")
        md_lines.append(f"- NN KS: `{float(rec['NN_KS']):.6g}`")
    (OUT / "candidate_scores.md").write_text("\n".join(md_lines) + "\n")
    (OUT / "recommendation.md").write_text("\n".join(md_lines) + "\n")

    print(json.dumps({"out": str(OUT), "best_candidate": str(best_path) if best_path else None}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
