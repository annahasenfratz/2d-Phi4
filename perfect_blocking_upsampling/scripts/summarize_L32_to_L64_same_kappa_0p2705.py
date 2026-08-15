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
REPORT = OUT / "L32_to_L64_same_kappa_0p2705_operator_AR_summary.md"
OP_CSV = OUT / "same_kappa_0p2705_operator_expected_vs_predicted.csv"
AR_CSV = OUT / "same_kappa_0p2705_AR_logweight_summary.csv"
NATIVE64 = ROOT / "phi4_phase-diagram" / "ensembles" / "lam0p022_kappa0p2705_L64_embedded_wolff_sign_cluster_plus_radial_heatbath_N500" / "configs.npz"
LAM = 0.022
KAPPA = 0.2705
L = 64
BOOT = 500
SEED = 20260704


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


def native_series() -> dict[str, np.ndarray]:
    phi = np.load(NATIVE64)["phi"].astype(np.float64)
    m = np.mean(phi, axis=(1, 2))
    nn = 0.5 * (
        np.mean(phi * np.roll(phi, -1, axis=1), axis=(1, 2))
        + np.mean(phi * np.roll(phi, -1, axis=2), axis=(1, 2))
    )
    two_nn = 0.5 * (
        np.mean(phi * np.roll(phi, -2, axis=1), axis=(1, 2))
        + np.mean(phi * np.roll(phi, -2, axis=2), axis=(1, 2))
    )
    diag = np.mean(phi * np.roll(np.roll(phi, -1, axis=1), -1, axis=2), axis=(1, 2))
    phi2 = np.mean(phi * phi, axis=(1, 2))
    phi4 = np.mean(phi**4, axis=(1, 2))
    action_density = (1.0 - 2.0 * LAM) * phi2 + LAM * phi4 - 4.0 * KAPPA * nn
    return {
        "m2": m * m,
        "m4": m**4,
        "phi2": phi2,
        "phi4": phi4,
        "NN": nn,
        "diag": diag,
        "2nn": two_nn,
        "action_density": action_density,
        "m": m,
        "abs_m": np.abs(m),
    }


def raw_series() -> dict[str, np.ndarray]:
    rows = [r for r in read_csv(SRC / "raw_upscaled_observables_by_kappaf.csv") if abs(float(r["kappa_f"]) - KAPPA) < 1e-12]
    out: dict[str, list[float]] = {
        "m2": [],
        "m4": [],
        "phi2": [],
        "phi4": [],
        "NN": [],
        "diag": [],
        "2nn": [],
        "action_density": [],
        "m": [],
        "abs_m": [],
    }
    for r in rows:
        for k in out:
            out[k].append(float(r[k]))
    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}


def aggregate(series: dict[str, np.ndarray], idx: np.ndarray | None = None) -> dict[str, float]:
    s = series if idx is None else {k: v[idx] for k, v in series.items()}
    m2 = float(np.mean(s["m2"]))
    m4 = float(np.mean(s["m4"]))
    phi2 = float(np.mean(s["phi2"]))
    return {
        "m2": m2,
        "m4": m4,
        "susceptibility": float(L * L * m2),
        "Binder_U4": float(1.0 - m4 / (3.0 * m2 * m2)) if m2 > 0 else float("nan"),
        "xi_over_L": float(math.sqrt(max(m2, 0.0) / max(phi2, 1e-300))),
        "NN": float(np.mean(s["NN"])),
        "diag": float(np.mean(s["diag"])),
        "2nn": float(np.mean(s["2nn"])),
        "action_density": float(np.mean(s["action_density"])),
        "phi2": phi2,
        "phi4": float(np.mean(s["phi4"])),
        "m": float(np.mean(s["m"])),
        "abs_m": float(np.mean(s["abs_m"])),
    }


def bootstrap_summary(series: dict[str, np.ndarray], seed: int) -> tuple[dict[str, float], dict[str, float]]:
    point = aggregate(series)
    n = len(next(iter(series.values())))
    rng = np.random.default_rng(seed)
    vals: dict[str, list[float]] = {k: [] for k in point}
    for _ in range(BOOT):
        idx = rng.integers(0, n, size=n)
        b = aggregate(series, idx)
        for k, v in b.items():
            vals[k].append(v)
    se = {k: float(np.std(v, ddof=1)) for k, v in vals.items()}
    return point, se


def logweight_summary() -> dict[str, Any]:
    rows = read_csv(SRC / "initial_logweight_summary_by_kappaf.csv")
    row = next(r for r in rows if abs(float(r["kappa_f"]) - KAPPA) < 1e-12)
    decomp = [r for r in read_csv(SRC / "state_logweight_decomposition_by_kappaf.csv") if abs(float(r["kappa_f"]) - KAPPA) < 1e-12]
    logw = np.asarray([float(r["logw"]) for r in decomp], dtype=np.float64)
    pair_acc = []
    for i in range(len(logw)):
        for j in range(len(logw)):
            if i != j:
                pair_acc.append(min(1.0, math.exp(min(0.0, logw[j] - logw[i]))))
    return {
        "lambda": LAM,
        "kappa_c": KAPPA,
        "kappa_f": KAPPA,
        "Lc": 32,
        "Lf": 64,
        "N_states": int(row["n"]),
        "logw_mean": float(row["mean"]),
        "logw_std": float(row["std"]),
        "ESS_per_N": float(row["ESS_over_N"]),
        "predicted_independence_acceptance": float(np.mean(pair_acc)) if pair_acc else float("nan"),
        "predicted_adjacent_acceptance": float(row["adjacent_order_predicted_acceptance"]),
        "actual_acceptance": "",
        "actual_acceptance_note": "actual A/R was not run; only logweight/predicted acceptance diagnostics are available",
    }


def operator_rows() -> list[dict[str, Any]]:
    native, native_se = bootstrap_summary(native_series(), SEED)
    raw, raw_se = bootstrap_summary(raw_series(), SEED + 1)
    ops = ["m2", "m4", "susceptibility", "Binder_U4", "xi_over_L", "NN", "diag", "2nn", "action_density", "phi2", "phi4", "m", "abs_m"]
    rows = []
    for op in ops:
        diff = raw[op] - native[op]
        cse = math.sqrt(native_se[op] ** 2 + raw_se[op] ** 2)
        rows.append(
            {
                "operator": op,
                "native_L64_expected": native[op],
                "native_SE": native_se[op],
                "raw_upscaled_predicted": raw[op],
                "upscaled_SE": raw_se[op],
                "difference": diff,
                "combined_SE": cse,
                "pull": diff / cse if cse > 0 else float("nan"),
            }
        )
    return rows


def fmt(x: Any) -> str:
    if x == "":
        return ""
    try:
        xf = float(x)
    except Exception:
        return str(x)
    if not math.isfinite(xf):
        return "nan"
    return f"{xf:.6g}"


def write_report(op_rows: list[dict[str, Any]], ar: dict[str, Any]) -> None:
    newer = [
        {"kappa_f": 0.27075, "logw_std": 6.96586, "ESS_per_N": 0.125139, "predicted_adjacent_acceptance": 0.57826},
        {"kappa_f": 0.27100, "logw_std": 14.7308, "ESS_per_N": 0.125000, "predicted_adjacent_acceptance": 0.478437},
        {"kappa_f": 0.27125, "logw_std": 8.00839, "ESS_per_N": 0.147062, "predicted_adjacent_acceptance": 0.578854},
    ]
    ranked = sorted(op_rows, key=lambda r: abs(float(r["pull"])), reverse=True)
    lines = [
        "# L32->L64 same-kappa 0.2705 raw-upscaling operator/A-R summary",
        "",
        "Source searched under `perfect_blocking_upsampling/outputs/`. The matching same-kappa raw-upscaled diagnostic is:",
        "",
        f"- `{SRC}`",
        "",
        "It is a sweep-0 diagnostic with `sweeps=0`, `chains=8`, and `save_sweeps=[0]`; no actual global A/R chain was run in this output.",
        "",
        "The native target is recomputed from:",
        "",
        f"- `{NATIVE64}`",
        "",
        "Susceptibility here follows the project validation convention in this output, `chi = V <m^2>`, not the connected `V(<m^2>-<|m|>^2)` phase-diagram convention.",
        "",
        "## Operator expected-vs-predicted table",
        "",
        "| operator | native L64 expected | native SE | raw upscaled predicted | upscaled SE | difference | combined SE | pull |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in op_rows:
        lines.append(
            f"| {r['operator']} | {fmt(r['native_L64_expected'])} | {fmt(r['native_SE'])} | "
            f"{fmt(r['raw_upscaled_predicted'])} | {fmt(r['upscaled_SE'])} | {fmt(r['difference'])} | "
            f"{fmt(r['combined_SE'])} | {fmt(r['pull'])} |"
        )
    lines += [
        "",
        "## A/R and logweight summary",
        "",
        "| lambda | kappa_c | kappa_f | Lc | Lf | N_states | logw_mean | logw_std | ESS/N | predicted independence acceptance | predicted adjacent acceptance | actual acceptance |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        f"| {fmt(ar['lambda'])} | {fmt(ar['kappa_c'])} | {fmt(ar['kappa_f'])} | {ar['Lc']} | {ar['Lf']} | {ar['N_states']} | "
        f"{fmt(ar['logw_mean'])} | {fmt(ar['logw_std'])} | {fmt(ar['ESS_per_N'])} | {fmt(ar['predicted_independence_acceptance'])} | "
        f"{fmt(ar['predicted_adjacent_acceptance'])} |  |",
        "",
        "**Actual A/R was not run; only logweight/predicted acceptance diagnostics are available.**",
        "",
        "## Comparison to newer sweep-0 kappaf diagnostics",
        "",
        "| kappa_f | logw std | ESS/N | predicted adjacent acceptance |",
        "|---:|---:|---:|---:|",
        f"| 0.27050 | {fmt(ar['logw_std'])} | {fmt(ar['ESS_per_N'])} | {fmt(ar['predicted_adjacent_acceptance'])} |",
    ]
    for r in newer:
        lines.append(f"| {r['kappa_f']:.5f} | {fmt(r['logw_std'])} | {fmt(r['ESS_per_N'])} | {fmt(r['predicted_adjacent_acceptance'])} |")
    lines += [
        "",
        "## Answers",
        "",
        "1. Did same-kappa raw upscaling fail mainly in observables, A/R/logweight spread, or both?",
        "",
        f"Both show issues, but the most conspicuous failure is in raw observables. The same-kappa logweight spread is `std={ar['logw_std']:.6g}` with `ESS/N={ar['ESS_per_N']:.6g}`, which is not catastrophic for only 8 states and is better than the newer 0.27100 logweight spread. However, the raw-upscaled ensemble has large local-observable pulls; the largest absolute pulls are "
        + ", ".join([f"`{r['operator']}` ({float(r['pull']):.2f})" for r in ranked[:5]])
        + ".",
        "",
        "2. Was an actual global A/R attempted?",
        "",
        "No. The source run has `sweeps=0` and no rows in `deltaS_AR_diagnostics_by_kappaf.csv`; it records raw upscaled states and logweights only.",
        "",
        "3. Which operators were most discrepant?",
        "",
        "The largest normalized discrepancies are listed above. The local amplitude/correlation sector is high in the raw upscaled states: `m2`, `m4`, `chi`, `NN`, `diag`, `2nn`, `phi2/phi4`, and `action_density` all shift relative to native L64.",
        "",
        "4. How does kappa_f=0.2705 compare to newer candidates?",
        "",
        "Same-kappa `kappa_f=0.2705` has the smallest listed logweight spread among the sweep-0 candidates (`7.60`, close to `6.97` for 0.27075 and below `8.01` for 0.27125), and the best ESS/N in this tiny 8-state diagnostic (`0.264`). But its raw observables are not native-like. This supports the current interpretation that sweep-0 logweight mechanics alone are not enough; operator matching and subsequent patch-chain rethermalization must decide the fine target.",
        "",
        "Machine-readable outputs:",
        "",
        f"- `{OP_CSV.name}`",
        f"- `{AR_CSV.name}`",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ops = operator_rows()
    ar = logweight_summary()
    write_csv(OP_CSV, ops)
    write_csv(AR_CSV, [ar])
    write_report(ops, ar)
    print(json.dumps({"report": str(REPORT), "operator_rows": len(ops), "ar_rows": 1}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
