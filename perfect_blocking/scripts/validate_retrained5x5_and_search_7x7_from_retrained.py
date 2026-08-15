#!/usr/bin/env python3
"""Validate lambda=1.0 retrained 5x5 and search 7x7 refinements from it."""

from __future__ import annotations

import argparse
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
OLD_KERNEL = LAM_ROOT / "kernels/final/chosen_kernel.json"
RETRAINED_KERNEL = LAM_ROOT / "kernels/candidates/systematic_training/best_retrained_5x5_full_objective_eta_included.json"
FINE_CONFIGS = PROJECT_ROOT / "data/configs_phi4_2d/lam1p0_kappac0p340301_L32/configs.npz"
DIRECT_CONFIGS = PROJECT_ROOT / "data/configs_phi4_2d/lam1p0_kappac0p340301_L16/configs.npz"
VALIDATION_DIR = LAM_ROOT / "tests/intermediate/retrained_5x5_validation"
SEARCH_DIR = LAM_ROOT / "tests/intermediate/7x7_from_retrained_5x5"
CAND_DIR = LAM_ROOT / "kernels/candidates/7x7_from_retrained_5x5"

KEY_OBS = ["action_density", "phi2", "phi4", "local_kurtosis_ratio", "NN", "m2", "m4", "G_pmin_avg"]
TABLE_OBS = [
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


def kernel_matrix(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    data = json.loads(path.read_text())
    if "matrix" in data:
        matrix = np.asarray(data["matrix"], dtype=np.float64)
        if bool(data.get("kernel_coefficients_include_eta_scale")):
            return matrix, data
    kernel = load_kernel(path)
    return kernel.matrix, kernel.metadata


def scalar_score(row: dict[str, Any]) -> float:
    return abs(float(row["standardized_mean_shift"])) + float(row["total_variation"]) + 10.0 * float(row["jensen_shannon"])


def local_score(metric_rows: dict[str, dict[str, Any]]) -> float:
    return (
        4.0 * scalar_score(metric_rows["phi2"])
        + 3.0 * scalar_score(metric_rows["phi4"])
        + 5.0 * scalar_score(metric_rows["local_kurtosis_ratio"])
    )


def guardrail_ok(metric_rows: dict[str, dict[str, Any]], mom: dict[str, float], baseline: dict[str, dict[str, Any]], base_mom: dict[str, float]) -> bool:
    return (
        float(metric_rows["action_density"]["ks_statistic"]) <= max(0.035, float(baseline["action_density"]["ks_statistic"]) + 0.005)
        and float(metric_rows["action_density"]["jensen_shannon"]) <= max(0.005, float(baseline["action_density"]["jensen_shannon"]) + 0.001)
        and float(metric_rows["m2"]["ks_statistic"]) <= max(0.025, float(baseline["m2"]["ks_statistic"]) + 0.003)
        and float(metric_rows["m4"]["ks_statistic"]) <= max(0.025, float(baseline["m4"]["ks_statistic"]) + 0.003)
        and float(metric_rows["G_pmin_avg"]["ks_statistic"]) <= max(0.035, float(baseline["G_pmin_avg"]["ks_statistic"]) + 0.004)
        and mom["min_K"] > 0.0
        and mom["max_inverse_K"] <= max(1.6, base_mom["max_inverse_K"] + 0.1)
    )


def evaluate_matrix(name: str, matrix: np.ndarray, direct_obs: dict[str, np.ndarray], fine: np.ndarray) -> tuple[dict[str, dict[str, Any]], dict[str, float]]:
    blocked_obs = observable_arrays(block(fine, matrix))
    return full_metrics(direct_obs, blocked_obs), momentum_extrema(matrix, grid=512)


def metric_rows_for_candidate(candidate: str, family: str, metric_rows: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for obs in TABLE_OBS:
        rows.append({"candidate": candidate, "family": family, "observable": obs, **metric_rows[obs]})
    return rows


def validate_retrained(direct: np.ndarray, fine: np.ndarray, old_matrix: np.ndarray, retrained_matrix: np.ndarray, old_meta: dict[str, Any], retrained_meta: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, float], dict[str, float]]:
    out = VALIDATION_DIR
    (out / "plots").mkdir(parents=True, exist_ok=True)
    direct_obs = observable_arrays(direct)
    old_metrics, old_mom = evaluate_matrix("old_current_5x5", old_matrix, direct_obs, fine)
    new_metrics, new_mom = evaluate_matrix("retrained_5x5", retrained_matrix, direct_obs, fine)

    rows: list[dict[str, Any]] = []
    for candidate, family, metric_rows in [
        ("old_current_5x5", "current_final", old_metrics),
        ("retrained_5x5", "systematic_training", new_metrics),
    ]:
        rows.extend(metric_rows_for_candidate(candidate, family, metric_rows))
    write_csv(out / "full_metrics_old_vs_retrained.csv", rows)

    momentum_rows = [
        {
            "candidate": "old_current_5x5",
            "sum_K": float(old_matrix.sum()),
            "kernel_coefficients_include_eta_scale": bool(old_meta.get("kernel_coefficients_include_eta_scale")),
            "eta_scale": ETA_SCALE,
            **old_mom,
            "condition_number": old_mom["max_K"] / old_mom["min_K"],
        },
        {
            "candidate": "retrained_5x5",
            "sum_K": float(retrained_matrix.sum()),
            "kernel_coefficients_include_eta_scale": bool(retrained_meta.get("kernel_coefficients_include_eta_scale")),
            "eta_scale": ETA_SCALE,
            **new_mom,
            "condition_number": new_mom["max_K"] / new_mom["min_K"],
        },
    ]
    write_csv(out / "momentum_stability_old_vs_retrained.csv", momentum_rows)

    split_rows: list[dict[str, Any]] = []
    n = len(direct)
    split_defs = {
        "first_half": np.arange(0, n // 2),
        "second_half": np.arange(n // 2, n),
        "random_2500_seed20260716": np.random.default_rng(20260716).choice(n, size=n // 2, replace=False),
    }
    for split, idx in split_defs.items():
        d_obs = observable_arrays(direct[idx])
        for cname, matrix in [("old_current_5x5", old_matrix), ("retrained_5x5", retrained_matrix)]:
            mrows, _ = evaluate_matrix(cname, matrix, d_obs, fine[idx])
            for obs in KEY_OBS:
                split_rows.append({"split": split, "candidate": cname, "observable": obs, **mrows[obs]})

    # Bootstrap paired resamples for key metrics. Recompute metrics on blocked observables only, not blocking.
    old_blocked_obs = observable_arrays(block(fine, old_matrix))
    new_blocked_obs = observable_arrays(block(fine, retrained_matrix))
    rng = np.random.default_rng(20260717)
    for b in range(80):
        idx = rng.integers(0, n, size=n)
        for cname, bobs in [("old_current_5x5", old_blocked_obs), ("retrained_5x5", new_blocked_obs)]:
            for obs in KEY_OBS:
                row = metrics(direct_obs[obs][idx], bobs[obs][idx], bins=50)
                split_rows.append({"split": f"bootstrap_{b:03d}", "candidate": cname, "observable": obs, **row})
    write_csv(out / "bootstrap_or_split_metrics.csv", split_rows)

    for cname, matrix, bobs in [
        ("old_current_5x5", old_matrix, old_blocked_obs),
        ("retrained_5x5", retrained_matrix, new_blocked_obs),
    ]:
        for obs in KEY_OBS:
            plot_histogram(
                direct_obs[obs],
                bobs[obs],
                obs,
                out / "plots" / f"{cname}_{obs}.pdf",
                label_a="direct L16",
                label_b=f"{cname} blocked L32->L16",
            )

    report_lines = [
        "# Retrained 5x5 Validation",
        "",
        "The retrained 5x5 candidate is compared against the old current final 5x5 on the same lambda=1.0 L16/L32 ensembles.",
        "",
        "| observable | old KS | retrained KS | old JS | retrained JS | old shift | retrained shift |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for obs in KEY_OBS:
        report_lines.append(
            f"| {obs} | {float(old_metrics[obs]['ks_statistic']):.6g} | {float(new_metrics[obs]['ks_statistic']):.6g} | "
            f"{float(old_metrics[obs]['jensen_shannon']):.6g} | {float(new_metrics[obs]['jensen_shannon']):.6g} | "
            f"{float(old_metrics[obs]['standardized_mean_shift']):.6g} | {float(new_metrics[obs]['standardized_mean_shift']):.6g} |"
        )
    robust = (
        float(new_metrics["phi2"]["ks_statistic"]) < float(old_metrics["phi2"]["ks_statistic"])
        and float(new_metrics["phi4"]["ks_statistic"]) < float(old_metrics["phi4"]["ks_statistic"])
        and float(new_metrics["action_density"]["ks_statistic"]) <= max(0.035, float(old_metrics["action_density"]["ks_statistic"]) + 0.005)
        and new_mom["max_inverse_K"] <= 1.6
        and math.isclose(float(retrained_matrix.sum()), ETA_SCALE, rel_tol=1e-12, abs_tol=1e-12)
    )
    report_lines.extend(
        [
            "",
            "## Eta And Momentum",
            "",
            f"- old sum(K): `{float(old_matrix.sum()):.17g}`",
            f"- retrained sum(K): `{float(retrained_matrix.sum()):.17g}`",
            f"- eta_scale: `{ETA_SCALE:.17g}`",
            f"- retrained max 1/K(p): `{new_mom['max_inverse_K']:.17g}`",
            "",
            f"Robustly better under the configured checks: `{robust}`.",
        ]
    )
    (out / "retrained_5x5_validation_report.md").write_text("\n".join(report_lines) + "\n")
    return old_metrics, new_metrics, old_mom, new_mom


def classes_from_eta_matrix(matrix: np.ndarray) -> dict[str, float]:
    return classes_from_matrix(matrix / ETA_SCALE)


def search_7x7_from_retrained(direct: np.ndarray, fine: np.ndarray, retrained_matrix: np.ndarray, baseline_metrics: dict[str, dict[str, Any]], baseline_mom: dict[str, float]) -> None:
    out = SEARCH_DIR
    for rel in ["plots"]:
        (out / rel).mkdir(parents=True, exist_ok=True)
    CAND_DIR.mkdir(parents=True, exist_ok=True)

    direct_obs = observable_arrays(direct)
    base_classes = classes_from_eta_matrix(retrained_matrix)
    for key in ("30", "31", "32", "33"):
        base_classes.setdefault(key, 0.0)

    rng = np.random.default_rng(20260718)
    candidate_specs: list[tuple[str, str, dict[str, float], dict[str, Any]]] = []
    candidate_specs.append(("embedded_retrained_5x5", "embedded_retrained_5x5", dict(base_classes), {"description": "fallback baseline"}))

    # Edge-only no33 small deformations.
    for i in range(360):
        scale = 0.004 if i < 160 else 0.010
        cls = dict(base_classes)
        cls["30"] = float(rng.normal(0.0, scale))
        cls["31"] = float(rng.normal(0.0, scale))
        cls["32"] = float(rng.normal(0.0, scale))
        cls["33"] = 0.0
        rest = sum(CLASS_MULT[k] * v for k, v in cls.items() if k != "00")
        cls["00"] = 1.0 - rest
        candidate_specs.append((f"edge_no33_from_retrained_{i:04d}", "edge_only_no33", cls, {"outer_scale": scale}))

    # Full no33 small deformations around retrained 5x5.
    keys = ["10", "11", "20", "21", "22", "30", "31", "32"]
    for i in range(520):
        scale = 0.002 if i < 220 else 0.005
        cls = dict(base_classes)
        for key in keys:
            cls[key] = float(cls.get(key, 0.0) + rng.normal(0.0, scale))
        cls["33"] = 0.0
        rest = sum(CLASS_MULT[k] * v for k, v in cls.items() if k != "00")
        cls["00"] = 1.0 - rest
        candidate_specs.append((f"full_no33_from_retrained_{i:04d}", "full_no33", cls, {"perturb_scale": scale}))

    # Subset screen before full scoring.
    fine_sub = fine[:1200]
    direct_sub_obs = observable_arrays(direct[:1200])
    screen_rows: list[tuple[float, str, str, dict[str, float], dict[str, Any]]] = []
    for name, family, cls, meta in candidate_specs:
        matrix = ETA_SCALE * matrix_from_classes(cls)
        bobs = observable_arrays(block(fine_sub, matrix))
        mrows = full_metrics(direct_sub_obs, bobs)
        mom = momentum_extrema(matrix, grid=192)
        loc = local_score(mrows)
        penalty = 0.0
        for obs, key, tol in [
            ("action_density", "ks_statistic", 0.004),
            ("action_density", "jensen_shannon", 0.001),
            ("m2", "ks_statistic", 0.003),
            ("m4", "ks_statistic", 0.003),
            ("G_pmin_avg", "ks_statistic", 0.004),
        ]:
            base = float(baseline_metrics[obs][key])
            penalty += 2.0e4 * max(0.0, float(mrows[obs][key]) - base - tol) ** 2
        penalty += 2.0e4 * max(0.0, mom["max_inverse_K"] - 1.6) ** 2
        penalty += 1.0e6 * max(0.0, -mom["min_K"]) ** 2
        penalty += 600.0 * float(np.mean((matrix - retrained_matrix) ** 2))
        screen_rows.append((loc + penalty, name, family, cls, meta))
    screen_rows.sort(key=lambda x: x[0])

    full_candidates = screen_rows[:90]
    candidate_rows: list[dict[str, Any]] = []
    hist_rows: list[dict[str, Any]] = []
    mom_rows: list[dict[str, Any]] = []
    outer_rows: list[dict[str, Any]] = []
    sum_rows: list[dict[str, Any]] = []
    details: list[tuple[dict[str, Any], dict[str, float], np.ndarray, dict[str, dict[str, Any]], dict[str, float]]] = []

    for _, name, family, cls, meta in full_candidates:
        matrix = ETA_SCALE * matrix_from_classes(cls)
        mrows, mom = evaluate_matrix(name, matrix, direct_obs, fine)
        ok = guardrail_ok(mrows, mom, baseline_metrics, baseline_mom)
        row = {
            "candidate": name,
            "family": family,
            "acceptable": ok,
            "local_score": local_score(mrows),
            "local_score_delta_vs_retrained_5x5": local_score(mrows) - local_score(baseline_metrics),
            "phi2_KS": mrows["phi2"]["ks_statistic"],
            "phi2_JS": mrows["phi2"]["jensen_shannon"],
            "phi2_shift": mrows["phi2"]["standardized_mean_shift"],
            "local_kurtosis_KS": mrows["local_kurtosis_ratio"]["ks_statistic"],
            "local_kurtosis_JS": mrows["local_kurtosis_ratio"]["jensen_shannon"],
            "phi4_KS": mrows["phi4"]["ks_statistic"],
            "action_density_KS": mrows["action_density"]["ks_statistic"],
            "action_density_JS": mrows["action_density"]["jensen_shannon"],
            "m2_KS": mrows["m2"]["ks_statistic"],
            "m4_KS": mrows["m4"]["ks_statistic"],
            "G_pmin_avg_KS": mrows["G_pmin_avg"]["ks_statistic"],
            "min_K": mom["min_K"],
            "max_K": mom["max_K"],
            "min_inverse_K": mom["min_inverse_K"],
            "max_inverse_K": mom["max_inverse_K"],
            "condition_number": mom["max_K"] / mom["min_K"],
        }
        candidate_rows.append(row)
        details.append((row, cls, matrix, mrows, mom))
        mom_rows.append({"candidate": name, "family": family, **mom})
        outer_rows.append({"candidate": name, "family": family, "K30_base": cls.get("30", 0.0), "K31_base": cls.get("31", 0.0), "K32_base": cls.get("32", 0.0), "K33_base": cls.get("33", 0.0)})
        sum_rows.append({"candidate": name, "family": family, "sum_base": float((matrix / ETA_SCALE).sum()), "sum_operational": float(matrix.sum()), "eta_scale": ETA_SCALE})
        for obs in TABLE_OBS:
            hist_rows.append({"candidate": name, "family": family, "observable": obs, **mrows[obs]})

    candidate_rows.sort(key=lambda r: (not bool(r["acceptable"]), float(r["local_score"])))
    write_csv(out / "candidate_scores.csv", candidate_rows)
    write_csv(out / "full_histogram_metrics.csv", hist_rows)
    write_csv(out / "momentum_stability.csv", mom_rows)
    write_csv(out / "outer_shell_coefficients.csv", outer_rows)
    write_csv(out / "candidate_kernel_sums.csv", sum_rows)

    best = None
    for row, cls, matrix, mrows, mom in sorted(details, key=lambda x: float(x[0]["local_score"])):
        improves = (
            row["candidate"] != "embedded_retrained_5x5"
            and bool(row["acceptable"])
            and (
                float(row["phi2_KS"]) < float(baseline_metrics["phi2"]["ks_statistic"])
                or float(row["local_kurtosis_KS"]) < float(baseline_metrics["local_kurtosis_ratio"]["ks_statistic"])
                or float(row["phi4_KS"]) < float(baseline_metrics["phi4"]["ks_statistic"])
            )
        )
        if improves:
            best = (row, cls, matrix, mrows, mom)
            break

    if best is not None:
        row, cls, matrix, mrows, mom = best
        cand = make_candidate(
            "best_7x7_from_retrained_5x5_eta_included",
            str(row["family"]),
            cls,
            {
                "source_candidate": row["candidate"],
                "selection": "best acceptable 7x7 candidate initialized from retrained 5x5",
                "guardrails": "relative to retrained 5x5: action, m2, m4, Gpmin, positivity, max_invK",
            },
        )
        write_candidate(cand, CAND_DIR)
        best_path = CAND_DIR / "best_7x7_from_retrained_5x5_eta_included.json"
    else:
        best_path = None

    # Plot fallback and best if available.
    plot_names = ["embedded_retrained_5x5"]
    if best is not None:
        plot_names.append(str(best[0]["candidate"]))
    plotted = set()
    for row, cls, matrix, mrows, mom in details:
        if row["candidate"] in plot_names and row["candidate"] not in plotted:
            bobs = observable_arrays(block(fine, matrix))
            for obs in KEY_OBS:
                plot_histogram(
                    direct_obs[obs],
                    bobs[obs],
                    obs,
                    out / "plots" / f"{row['candidate']}_{obs}.pdf",
                    label_a="direct L16",
                    label_b=f"{row['candidate']} blocked L32->L16",
                )
            plotted.add(str(row["candidate"]))

    lines = [
        "# 7x7 From Retrained 5x5 Recommendation",
        "",
        "The embedded retrained 5x5 was included as the fallback baseline. Candidate ranking used the mature guardrails relative to that retrained 5x5.",
        "",
        "| candidate | family | acceptable | local score | d local | phi2 KS | kurt KS | phi4 KS | action KS | m2 KS | m4 KS | Gpmin KS | max 1/K |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in candidate_rows[:20]:
        lines.append(
            "| {candidate} | {family} | {acceptable} | {local_score:.6g} | {local_score_delta_vs_retrained_5x5:.6g} | {phi2_KS:.6g} | {local_kurtosis_KS:.6g} | {phi4_KS:.6g} | {action_density_KS:.6g} | {m2_KS:.6g} | {m4_KS:.6g} | {G_pmin_avg_KS:.6g} | {max_inverse_K:.6g} |".format(
                **row
            )
        )
    lines.extend(["", f"Best 7x7 candidate path: `{best_path if best_path else 'none'}`"])
    if best is None:
        lines.append("")
        lines.append("No 7x7 candidate beat the retrained 5x5 under the configured mature guardrails.")
    else:
        lines.append("")
        lines.append("A 7x7 candidate improved at least one local target while passing the configured guardrails. It was not promoted.")
    (out / "recommendation.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    SEARCH_DIR.mkdir(parents=True, exist_ok=True)

    direct = load_configs(DIRECT_CONFIGS)
    fine = load_configs(FINE_CONFIGS)
    old_matrix, old_meta = kernel_matrix(OLD_KERNEL)
    retrained_matrix, retrained_meta = kernel_matrix(RETRAINED_KERNEL)
    old_metrics, new_metrics, old_mom, new_mom = validate_retrained(direct, fine, old_matrix, retrained_matrix, old_meta, retrained_meta)
    search_7x7_from_retrained(direct, fine, retrained_matrix, new_metrics, new_mom)
    print(json.dumps({"validation_dir": str(VALIDATION_DIR), "search_dir": str(SEARCH_DIR)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
