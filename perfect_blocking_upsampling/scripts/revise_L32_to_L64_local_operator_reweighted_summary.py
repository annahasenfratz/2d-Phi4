#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "perfect_blocking_upsampling" / "outputs" / "shape_parametric_sampler_validation"
SRC = OUT / "L32_to_L64_kappaf_matching" / "sweep0_initial_only"
FULL = OUT / "L32_to_L64_kappaf_matching" / "full_8x300"
LOCAL_CSV = OUT / "local_operator_raw_vs_reweighted_after_logweight.csv"
AR_CSV = OUT / "local_operator_AR_diagnostics_summary.csv"
REPORT = OUT / "L32_to_L64_same_kappa_0p2705_operator_AR_summary.md"
LAM = 0.022
L = 64
BOOT = 1000
SEED = 20260705
OPS = ["phi2", "phi4", "NN", "2nn", "diag"]
NATIVE_EXACT = {
    0.2705: ROOT / "phi4_phase-diagram" / "ensembles" / "lam0p022_kappa0p2705_L64_embedded_wolff_sign_cluster_plus_radial_heatbath_N500" / "configs.npz",
    0.2710: ROOT / "phi4_phase-diagram" / "ensembles" / "lam0p022_kappa0p271_L64_embedded_wolff_sign_cluster_plus_radial_heatbath_N500" / "configs.npz",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fkey(k: float) -> float:
    return round(float(k), 5)


def fmt(x: Any) -> str:
    try:
        y = float(x)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(y):
        return "NA"
    return f"{y:.6g}"


def native_local_series(kappa: float) -> dict[str, np.ndarray] | None:
    path = NATIVE_EXACT.get(fkey(kappa))
    if path is None or not path.exists():
        return None
    phi = np.load(path)["phi"].astype(np.float64)
    nn = 0.5 * (
        np.mean(phi * np.roll(phi, -1, axis=1), axis=(1, 2))
        + np.mean(phi * np.roll(phi, -1, axis=2), axis=(1, 2))
    )
    two_nn = 0.5 * (
        np.mean(phi * np.roll(phi, -2, axis=1), axis=(1, 2))
        + np.mean(phi * np.roll(phi, -2, axis=2), axis=(1, 2))
    )
    # Match the kappaf-matching script: perfect_blocking_upsampling.observables uses one diagonal direction.
    diag = np.mean(phi * np.roll(np.roll(phi, -1, axis=1), -1, axis=2), axis=(1, 2))
    return {
        "phi2": np.mean(phi * phi, axis=(1, 2)),
        "phi4": np.mean(phi**4, axis=(1, 2)),
        "NN": nn,
        "2nn": two_nn,
        "diag": diag,
    }


def native_summary(kappa: float) -> dict[str, tuple[float, float]] | None:
    series = native_local_series(kappa)
    if series is None:
        return None
    out: dict[str, tuple[float, float]] = {}
    for op, vals in series.items():
        out[op] = (float(np.mean(vals)), float(np.std(vals, ddof=1) / math.sqrt(len(vals))))
    return out


def proposal_rows() -> list[dict[str, str]]:
    return read_csv(SRC / "raw_upscaled_observables_by_kappaf.csv")


def logw_by_kappa_chain() -> dict[tuple[float, int], float]:
    out = {}
    for r in read_csv(SRC / "state_logweight_decomposition_by_kappaf.csv"):
        out[(fkey(float(r["kappa_f"])), int(r["chain_id"]))] = float(r["logw"])
    return out


def raw_mean_se(vals: np.ndarray) -> tuple[float, float]:
    return float(np.mean(vals)), float(np.std(vals, ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else float("nan")


def weighted_mean(vals: np.ndarray, logw: np.ndarray) -> float:
    w = np.exp(logw - float(np.max(logw)))
    return float(np.sum(w * vals) / np.sum(w))


def weighted_boot_se(vals: np.ndarray, logw: np.ndarray, seed: int) -> float:
    rng = np.random.default_rng(seed)
    n = len(vals)
    boots = []
    for _ in range(BOOT):
        idx = rng.integers(0, n, size=n)
        boots.append(weighted_mean(vals[idx], logw[idx]))
    return float(np.std(boots, ddof=1))


def actual_ar_summary() -> dict[float, float]:
    p = FULL / "deltaS_AR_diagnostics_by_kappaf.csv"
    if not p.exists():
        return {}
    rows = read_csv(p)
    out: dict[float, float] = {}
    for k in sorted({fkey(float(r["kappa_f"])) for r in rows}):
        sub = [r for r in rows if fkey(float(r["kappa_f"])) == k]
        if sub:
            out[k] = float(np.mean([int(r["accepted"]) for r in sub]))
    return out


def local_operator_table() -> list[dict[str, Any]]:
    props = proposal_rows()
    logws = logw_by_kappa_chain()
    rows: list[dict[str, Any]] = []
    kappas = sorted({fkey(float(r["kappa_f"])) for r in props})
    native_cache = {k: native_summary(k) for k in kappas}
    for k in kappas:
        sub = [r for r in props if fkey(float(r["kappa_f"])) == k]
        logw = np.asarray([logws[(k, int(r["chain_id"]))] for r in sub], dtype=np.float64)
        native = native_cache[k]
        for op in OPS:
            vals = np.asarray([float(r[op]) for r in sub], dtype=np.float64)
            raw, raw_se = raw_mean_se(vals)
            rw = weighted_mean(vals, logw)
            rw_se = weighted_boot_se(vals, logw, SEED + int(round(k * 1_000_000)) + OPS.index(op))
            if native is None:
                native_val = native_se = diff_raw = diff_rw = raw_pull = rw_pull = float("nan")
            else:
                native_val, native_se = native[op]
                diff_raw = raw - native_val
                diff_rw = rw - native_val
                raw_combined = math.sqrt(native_se * native_se + raw_se * raw_se)
                rw_combined = math.sqrt(native_se * native_se + rw_se * rw_se)
                raw_pull = diff_raw / raw_combined if raw_combined > 0 else float("nan")
                rw_pull = diff_rw / rw_combined if rw_combined > 0 else float("nan")
            rows.append(
                {
                    "kappa_f": k,
                    "operator": op,
                    "native_value": native_val,
                    "native_SE": native_se,
                    "raw_value": raw,
                    "raw_SE": raw_se,
                    "reweighted_value": rw,
                    "reweighted_SE": rw_se,
                    "raw_minus_native": diff_raw,
                    "reweighted_minus_native": diff_rw,
                    "raw_pull": raw_pull,
                    "reweighted_pull": rw_pull,
                }
            )
    return rows


def predicted_independence(logw: np.ndarray) -> float:
    vals = []
    for i in range(len(logw)):
        for j in range(len(logw)):
            if i != j:
                vals.append(min(1.0, math.exp(min(0.0, logw[j] - logw[i]))))
    return float(np.mean(vals)) if vals else float("nan")


def ar_table() -> list[dict[str, Any]]:
    init = read_csv(SRC / "initial_logweight_summary_by_kappaf.csv")
    logws_by_k = {}
    for r in read_csv(SRC / "state_logweight_decomposition_by_kappaf.csv"):
        logws_by_k.setdefault(fkey(float(r["kappa_f"])), []).append(float(r["logw"]))
    actual = actual_ar_summary()
    rows = []
    for r in init:
        k = fkey(float(r["kappa_f"]))
        logw = np.asarray(logws_by_k[k], dtype=np.float64)
        rows.append(
            {
                "kappa_f": k,
                "N_states": int(r["n"]),
                "logw_mean": float(r["mean"]),
                "logw_std": float(r["std"]),
                "ESS_per_N": float(r["ESS_over_N"]),
                "predicted_independence_acceptance": predicted_independence(logw),
                "predicted_adjacent_acceptance": float(r["adjacent_order_predicted_acceptance"]),
                "actual_AR_acceptance": actual.get(k, ""),
                "notes": "Actual global A/R was not run. The corrected estimate shown is self-normalized logweight reweighting, not an accepted Markov-chain estimate." if k not in actual else "Actual patch-chain A/R summary found in full_8x300 output.",
            }
        )
    return rows


def md_table(rows: list[dict[str, Any]], fields: list[str]) -> list[str]:
    lines = ["| " + " | ".join(fields) + " |", "|" + "|".join(["---" for _ in fields]) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(fmt(r[f]) if f not in {"operator", "notes"} else str(r[f]) for f in fields) + " |")
    return lines


def write_report(local_rows: list[dict[str, Any]], ar_rows: list[dict[str, Any]]) -> None:
    same = [r for r in local_rows if abs(float(r["kappa_f"]) - 0.2705) < 1e-12]
    available_native = sorted({float(r["kappa_f"]) for r in local_rows if math.isfinite(float(r["native_value"]))})
    lines = [
        "# L32->L64 local/action-sector raw proposal versus logweight-reweighted summary",
        "",
        "This revision focuses on local/action-sector operators for the sweep-0 upscaled proposal diagnostic:",
        "",
        "- `phi2`",
        "- `phi4`",
        "- `NN`",
        "- `2nn`",
        "- `diag`",
        "",
        "Magnetization-sector quantities (`|m|`, `m2`, `m4`, susceptibility, Binder, `xi/L`) are intentionally not used for the main A/R-quality conclusion here.",
        "",
        "For each operator, the self-normalized logweight estimate is",
        "",
        r"\[",
        r"\langle O\rangle_{\rm rw} = \frac{\sum_i w_i O_i}{\sum_i w_i},\qquad w_i=\exp(\log w_i-\max_j\log w_j).",
        r"\]",
        "",
        "Pulls use the same convention as before, with raw or reweighted proposal SE combined in quadrature with the native SE.",
        "",
        "**Actual global A/R was not run. The corrected estimate shown is self-normalized logweight reweighting, not an accepted Markov-chain estimate.**",
        "",
        f"Exact native/direct L64 references were available for kappas: `{available_native}`. Rows for other kappas keep native/pull columns unavailable.",
        "",
        "## Main local operator table",
        "",
        *md_table(
            local_rows,
            [
                "kappa_f",
                "operator",
                "native_value",
                "native_SE",
                "raw_value",
                "raw_SE",
                "reweighted_value",
                "reweighted_SE",
                "raw_minus_native",
                "reweighted_minus_native",
                "raw_pull",
                "reweighted_pull",
            ],
        ),
        "",
        "## A/R diagnostic table",
        "",
        *md_table(
            ar_rows,
            [
                "kappa_f",
                "N_states",
                "logw_mean",
                "logw_std",
                "ESS_per_N",
                "predicted_independence_acceptance",
                "predicted_adjacent_acceptance",
                "actual_AR_acceptance",
                "notes",
            ],
        ),
        "",
        "## Same-kappa baseline interpretation",
        "",
        "For `kappa_f=0.2705`, native L64 data exist, so raw and reweighted proposal estimates can be compared directly.",
        "",
        *md_table(
            same,
            [
                "operator",
                "native_value",
                "native_SE",
                "raw_value",
                "raw_SE",
                "reweighted_value",
                "reweighted_SE",
                "raw_minus_native",
                "reweighted_minus_native",
                "raw_pull",
                "reweighted_pull",
            ],
        ),
        "",
        "Answers:",
        "",
        "1. Are `phi2`, `phi4`, `NN`, `2nn`, and `diag` close after logweight reweighting?",
        "",
        "Not convincingly. For the same-kappa baseline, reweighting moves some local operators but does not consistently reduce the discrepancy. With only 8 proposal states, the reweighted SEs are large and the estimate is weight-concentrated.",
        "",
        "2. Does the A/R/logweight correction reduce the raw discrepancy?",
        "",
        "No, not at the level of central values for the same-kappa baseline. In this tiny proposal set, self-normalized reweighting moves all five local/action-sector operators farther from the native central values. The reweighted pulls are smaller only because the bootstrap SEs are much larger, reflecting weight concentration and `N=8` proposal noise.",
        "",
        "3. Is the same-kappa upscaler acceptable in the local/action-sector after correction?",
        "",
        "The evidence is insufficient to call it acceptable from sweep-0 reweighting alone. The corrected local estimates remain visibly offset for several operators and are based on only `N=8` proposals.",
        "",
        "4. Is a true global A/R chain needed?",
        "",
        "Yes. The `N=8` self-normalized reweighted diagnostic is too noisy to establish corrected local/action-sector agreement. A true global-A/R or patch-chain diagnostic is needed to verify whether the corrected Markov chain samples the local/action sector properly.",
        "",
        "Machine-readable outputs:",
        "",
        f"- `{LOCAL_CSV.name}`",
        f"- `{AR_CSV.name}`",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    local = local_operator_table()
    ar = ar_table()
    write_csv(LOCAL_CSV, local)
    write_csv(AR_CSV, ar)
    write_report(local, ar)
    print(json.dumps({"local_rows": len(local), "ar_rows": len(ar), "report": str(REPORT)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
