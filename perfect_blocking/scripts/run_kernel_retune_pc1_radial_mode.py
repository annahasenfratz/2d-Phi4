#!/usr/bin/env python3
"""Retune the lambda=1.0 blocking kernel for the radial PC1 coarse marginal."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path("perfect_blocking/logs/mplconfig").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize, stats


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "perfect_blocking/perfect_blocking_lam1p0/kernel_retune_pc1_radial_mode_20260720"
KERNEL_PATH = ROOT / "perfect_blocking/perfect_blocking_lam1p0/kernels/final/chosen_kernel.json"
DIRECT_PATH = ROOT / "data/configs_phi4_2d/lam1p0_kappac0p340301_L16/configs.npz"
FINE_PATH = ROOT / "data/configs_phi4_2d/lam1p0_kappac0p340301_L32/configs.npz"
COARSE64_PATH = ROOT / "data/configs_phi4_2d/lam1p0_kappac0p340301_L64/configs.npz"
LAM = 1.0
KAPPA = 0.340301
ETA_SCALE = 2.0 ** 0.125
PC1 = np.asarray([0.4525, 0.8918], dtype=np.float64)
PC2 = np.asarray([0.8918, -0.4525], dtype=np.float64)
ORBIT_KEYS = ("00", "10", "11", "20", "21", "22")
ORBIT_MULT = {"00": 1, "10": 4, "11": 4, "20": 4, "21": 8, "22": 4}
OBS = [
    "action_density", "NN", "diag", "2nn", "phi2", "phi4",
    "local_kurtosis_ratio", "m2", "m4", "Binder_U4_from_averages",
    "xi_over_L", "G_pmin_avg", "PC1", "PC2",
]
GUARD_OBS = ["action_density", "NN", "diag", "2nn", "phi2", "phi4", "local_kurtosis_ratio", "m2", "m4", "Binder_U4_from_averages", "xi_over_L", "G_pmin_avg"]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_npz(path: Path) -> np.ndarray:
    with np.load(path) as data:
        return np.asarray(data["phi"], dtype=np.float64)


def orbit_classes(matrix: np.ndarray) -> dict[str, float]:
    r = matrix.shape[0] // 2
    out: dict[str, float] = {}
    for dx in range(r + 1):
        for dy in range(dx + 1):
            out[f"{dx}{dy}"] = float(matrix[r + dx, r + dy])
    return out


def matrix_from_classes(classes: dict[str, float], radius: int = 2) -> np.ndarray:
    out = np.zeros((2 * radius + 1, 2 * radius + 1), dtype=np.float64)
    for i, dx in enumerate(range(-radius, radius + 1)):
        for j, dy in enumerate(range(-radius, radius + 1)):
            out[i, j] = classes[f"{max(abs(dx), abs(dy))}{min(abs(dx), abs(dy))}"]
    return out


def normalize_classes(classes: dict[str, float]) -> dict[str, float]:
    out = dict(classes)
    out["00"] = 1.0 - sum(ORBIT_MULT[k] * out[k] for k in ORBIT_KEYS if k != "00")
    return out


def apply_kernel(phi: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    out = np.zeros_like(phi, dtype=np.float64)
    r = matrix.shape[0] // 2
    for i, dx in enumerate(range(-r, r + 1)):
        for j, dy in enumerate(range(-r, r + 1)):
            if matrix[i, j] != 0.0:
                out += matrix[i, j] * np.roll(np.roll(phi, -dx, axis=1), -dy, axis=2)
    return out


def block(phi: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return apply_kernel(phi, matrix)[:, ::2, ::2]


def observables(phi: np.ndarray) -> dict[str, np.ndarray]:
    arr = np.asarray(phi, dtype=np.float64)
    n, L, _ = arr.shape
    v = float(L * L)
    m = arr.mean(axis=(1, 2))
    p2 = np.mean(arr**2, axis=(1, 2))
    p4 = np.mean(arr**4, axis=(1, 2))
    nn = 0.5 * (np.mean(arr * np.roll(arr, -1, axis=1), axis=(1, 2)) + np.mean(arr * np.roll(arr, -1, axis=2), axis=(1, 2)))
    twonn = 0.5 * (np.mean(arr * np.roll(arr, -2, axis=1), axis=(1, 2)) + np.mean(arr * np.roll(arr, -2, axis=2), axis=(1, 2)))
    diag = np.mean(arr * np.roll(np.roll(arr, -1, axis=1), -1, axis=2), axis=(1, 2))
    action = ((1.0 - 2.0 * LAM) * arr**2 + LAM * arr**4 - 2.0 * KAPPA * (arr * np.roll(arr, -1, axis=1) + arr * np.roll(arr, -1, axis=2))).sum(axis=(1, 2)) / v
    phase = np.exp(2j * np.pi * np.arange(L) / L)
    gx = np.abs(np.tensordot(arr, phase, axes=([1], [0])).sum(axis=1)) ** 2 / v
    gy = np.abs(np.tensordot(arr, phase, axes=([2], [0])).sum(axis=1)) ** 2 / v
    m2, m4 = m**2, m**4
    binder = np.full(n, 1.0 - float(np.mean(m4)) / max(3.0 * float(np.mean(m2)) ** 2, 1e-300))
    gp = float(np.mean(0.5 * (gx + gy)))
    g0 = float(v * max(float(np.mean(m2)) - float(np.mean(m)) ** 2, 0.0))
    xi_arg = g0 / gp - 1.0 if gp > 0 else float("nan")
    xi = np.full(n, (1.0 / (2.0 * L * math.sin(math.pi / L))) * math.sqrt(xi_arg) if xi_arg > 0 else float("nan"))
    return {
        "action_density": action, "NN": nn, "diag": diag, "2nn": twonn,
        "phi2": p2, "phi4": p4, "local_kurtosis_ratio": p4 / np.maximum(p2 * p2, 1e-300),
        "m2": m2, "m4": m4, "Binder_U4_from_averages": binder, "xi_over_L": xi,
        "G_pmin_avg": 0.5 * (gx + gy), "PC1": PC1[0] * p2 + PC1[1] * p4,
        "PC2": PC2[0] * p2 + PC2[1] * p4,
    }


def finite(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x[np.isfinite(x)]


def metric(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    a, b = finite(a), finite(b)
    ks = stats.ks_2samp(a, b).statistic
    sa, sb = float(np.std(a, ddof=1)), float(np.std(b, ddof=1))
    ratio = sb / sa if sa > 1e-14 else (1.0 if sb <= 1e-14 else float("nan"))
    shift = (float(np.mean(b)) - float(np.mean(a))) / sa if sa > 1e-14 else 0.0
    return {
        "direct_mean": float(np.mean(a)), "blocked_mean": float(np.mean(b)),
        "mean_shift_raw": float(np.mean(b) - np.mean(a)),
        "direct_std": sa, "blocked_std": sb, "std_ratio_blocked_over_direct": ratio,
        "standardized_mean_shift": shift,
        "KS": float(ks), "W1": float(stats.wasserstein_distance(a, b)),
        "TV": float(0.5 * np.sum(np.abs(np.histogram(a, bins=80, range=(min(a.min(), b.min()), max(a.max(), b.max())))[0] / len(a) - np.histogram(b, bins=80, range=(min(a.min(), b.min()), max(a.max(), b.max())))[0] / len(b)))),
        "q05_direct": float(np.quantile(a, .05)), "q50_direct": float(np.quantile(a, .50)), "q95_direct": float(np.quantile(a, .95)), "q99_direct": float(np.quantile(a, .99)),
        "q05_blocked": float(np.quantile(b, .05)), "q50_blocked": float(np.quantile(b, .50)), "q95_blocked": float(np.quantile(b, .95)), "q99_blocked": float(np.quantile(b, .99)),
    }


def evaluate(name: str, classes: dict[str, float], direct: dict[str, np.ndarray], fine: np.ndarray, *, full: bool, l64: np.ndarray | None = None) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray]:
    matrix = ETA_SCALE * matrix_from_classes(classes)
    blocked = observables(block(fine, matrix))
    rows: list[dict[str, Any]] = []
    rec: dict[str, Any] = {"candidate": name, "family": "5x5_D4", "sum_base": float(np.sum(matrix / ETA_SCALE)), "sum_K": float(np.sum(matrix)), **momentum(matrix)}
    for key in OBS:
        m = metric(direct[key], blocked[key])
        rows.append({"candidate": name, "level": "L32toL16", "observable": key, **m})
        rec[f"{key}_shift"] = m["standardized_mean_shift"]
        rec[f"{key}_std_ratio"] = m["std_ratio_blocked_over_direct"]
        rec[f"{key}_KS"] = m["KS"]
        rec[f"{key}_W1"] = m["W1"]
    if l64 is not None:
        # Direct L32 is the first half of the native L32 ensemble for a matched-size guardrail.
        direct32 = observables(load_npz(FINE_PATH))
        blocked64 = observables(block(l64, matrix))
        for key in ("PC1", "PC2", "action_density", "NN", "phi2", "phi4", "local_kurtosis_ratio", "G_pmin_avg"):
            rows.append({"candidate": name, "level": "L64toL32", "observable": key, **metric(direct32[key], blocked64[key])})
    return rec, rows, matrix


def momentum(matrix: np.ndarray, grid: int = 256) -> dict[str, float]:
    r = matrix.shape[0] // 2
    coords = np.arange(-r, r + 1)
    ps = np.linspace(-np.pi, np.pi, grid, endpoint=False)
    vals_all = []
    for px in ps:
        ex = np.exp(1j * px * coords)
        ey = np.exp(1j * ps[:, None] * coords)
        vals_all.append(np.real(np.einsum("i,ij,nj->n", ex, matrix, ey)))
    vals = np.concatenate(vals_all)
    return {"min_K": float(vals.min()), "max_K": float(vals.max()), "min_inverse_K": float((1.0 / vals).min()), "max_inverse_K": float((1.0 / vals).max()), "condition_number": float(vals.max() / vals.min()) if vals.min() > 0 else float("inf")}


def guard_ok(rec: dict[str, Any], baseline: dict[str, Any]) -> bool:
    return bool(
        abs(rec["PC1_shift"]) < 0.03
        and abs(rec["PC1_std_ratio"] - 1.0) < 0.02
        and rec["PC1_KS"] < 0.04
        and abs(rec["PC2_shift"]) < 0.05
        and abs(rec["PC2_std_ratio"] - 1.0) < 0.05
        and abs(rec["action_density_shift"]) < 0.05
        and abs(rec["action_density_std_ratio"] - 1.0) < 0.03
        and rec["NN_KS"] <= baseline["NN_KS"] + 0.01
        and rec["local_kurtosis_ratio_KS"] <= baseline["local_kurtosis_ratio_KS"]
        and abs(rec["G_pmin_avg_shift"]) < 0.05
        and rec["min_K"] > 0.0
        and rec["max_inverse_K"] <= baseline["max_inverse_K"] * 1.02
    )


def score(rec: dict[str, Any], baseline: dict[str, Any]) -> float:
    return (
        100.0 * rec["PC1_shift"] ** 2
        + 60.0 * (rec["PC1_std_ratio"] - 1.0) ** 2
        + 20.0 * rec["PC1_KS"] ** 2
        + 15.0 * rec["PC2_shift"] ** 2
        + 10.0 * (rec["PC2_std_ratio"] - 1.0) ** 2
        + 10.0 * rec["action_density_shift"] ** 2
        + 10.0 * max(0.0, rec["NN_KS"] - baseline["NN_KS"]) ** 2
        + 10.0 * max(0.0, rec["local_kurtosis_ratio_KS"] - baseline["local_kurtosis_ratio_KS"]) ** 2
        + 5.0 * rec["G_pmin_avg_shift"] ** 2
    )


def save_kernel(path: Path, classes: dict[str, float], name: str, matrix: np.ndarray, mom: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"name": name, "family": "5x5_D4", "lambda": LAM, "kappa_f": KAPPA, "kappa_c": KAPPA, "eta": .25, "eta_scale_numeric": ETA_SCALE, "kernel_coefficients_include_eta_scale": True, "base_kernel_sum_before_eta_scale": 1.0, "final_kernel_sum_after_eta_scale": float(np.sum(matrix)), "base_orbit_classes_before_eta_scale": classes, "base_matrix_before_eta_scale": matrix.tolist(), "matrix": (ETA_SCALE * matrix).tolist(), "momentum_stability": mom}, indent=2) + "\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    direct_phi = load_npz(DIRECT_PATH)
    fine_phi = load_npz(FINE_PATH)
    l64_phi = load_npz(COARSE64_PATH)
    with KERNEL_PATH.open() as f:
        current_json = json.load(f)
    current_classes = normalize_classes({k: float(v) for k, v in current_json["base_orbit_classes_before_eta_scale"].items()})
    direct_full, fine_obs = observables(direct_phi), observables(fine_phi)
    direct_obs = direct_full
    # Current baseline on the fast subset and full ensemble.
    subset = min(2000, len(fine_phi), len(direct_phi))
    direct_fast = {k: v[:subset] for k, v in direct_obs.items()}
    baseline_rec, baseline_rows, baseline_matrix = evaluate("current_kernel", current_classes, direct_fast, fine_phi[:subset], full=False)
    baseline_rec["score"] = score(baseline_rec, baseline_rec)
    baseline_rec["guard_ok"] = guard_ok(baseline_rec, baseline_rec)
    full_baseline_rec, full_baseline_rows, _ = evaluate("current_kernel", current_classes, direct_obs, fine_phi, full=True, l64=l64_phi)
    write_csv(OUT / "candidate_metrics.csv", [])
    write_csv(OUT / "L32to16_full_validation.csv", full_baseline_rows)
    write_csv(OUT / "L64to32_validation.csv", [r for r in full_baseline_rows if r["level"] == "L64toL32"])

    rng = np.random.default_rng(2026072020)
    vars_ = list(ORBIT_KEYS[1:])
    candidates: list[tuple[str, dict[str, float]]] = [("current_kernel", current_classes)]
    # Coordinate response and local random exploration around the selected kernel.
    for key in vars_:
        for sign in (-1.0, 1.0):
            c = dict(current_classes)
            c[key] += sign * 0.003
            candidates.append((f"local_{key}_{'m' if sign < 0 else 'p'}003", normalize_classes(c)))
    for i in range(300):
        scale = float(rng.choice([0.0005, 0.001, 0.002, 0.003, 0.005, 0.008]))
        c = dict(current_classes)
        for key in vars_:
            c[key] += float(rng.normal(0.0, scale))
        candidates.append((f"random_{i:04d}_s{scale:g}", normalize_classes(c)))

    fast_records: list[dict[str, Any]] = []
    fast_metrics: list[dict[str, Any]] = []
    kernel_records: list[dict[str, Any]] = []
    for name, classes in candidates:
        rec, rows, matrix = evaluate(name, classes, direct_fast, fine_phi[:subset], full=False)
        rec["score"] = score(rec, baseline_rec)
        rec["guard_ok"] = guard_ok(rec, baseline_rec)
        rec.update({f"class_{k}": classes[k] for k in ORBIT_KEYS})
        fast_records.append(rec)
        fast_metrics.extend(rows)
        kernel_records.append({"candidate": name, **classes, "stored_sum": float(np.sum(ETA_SCALE * matrix_from_classes(classes)))})
    fast_records.sort(key=lambda r: (not r["guard_ok"], r["score"]))
    write_csv(OUT / "candidate_metrics.csv", fast_records)
    write_csv(OUT / "candidate_kernels.csv", kernel_records)
    write_csv(OUT / "guardrail_metrics.csv", [r for r in fast_metrics if r["observable"] in GUARD_OBS])

    # Finite-difference sensitivities of raw PC1 mean and standard deviation.
    sens: list[dict[str, Any]] = []
    eps = 0.001
    for key in vars_:
        cp, cm = dict(current_classes), dict(current_classes)
        cp[key] += eps
        cm[key] -= eps
        rp, _, _ = evaluate("plus", normalize_classes(cp), direct_fast, fine_phi[:subset], full=False)
        rm, _, _ = evaluate("minus", normalize_classes(cm), direct_fast, fine_phi[:subset], full=False)
        sens.append({"coefficient": key, "epsilon": eps, "d_PC1_mean_shift_per_base_coeff": (rp["PC1_shift"] - rm["PC1_shift"]) / (2 * eps), "d_PC1_std_ratio_per_base_coeff": (rp["PC1_std_ratio"] - rm["PC1_std_ratio"]) / (2 * eps), "d_PC2_mean_shift_per_base_coeff": (rp["PC2_shift"] - rm["PC2_shift"]) / (2 * eps)})
    write_csv(OUT / "kernel_sensitivity.csv", sens)

    # Full confirmation of current plus the best five fast candidates.
    shortlist = ["current_kernel"] + [r["candidate"] for r in fast_records if r["candidate"] != "current_kernel"][:5]
    class_map = {n: c for n, c in candidates}
    full_records: list[dict[str, Any]] = []
    full_rows: list[dict[str, Any]] = []
    full_kernels: list[dict[str, Any]] = []
    for name in shortlist:
        rec, rows, matrix = evaluate(name, class_map[name], direct_obs, fine_phi, full=True, l64=l64_phi)
        rec["score"] = score(rec, full_baseline_rec)
        rec["guard_ok"] = guard_ok(rec, full_baseline_rec)
        full_records.append(rec)
        full_rows.extend(rows)
        mom = momentum(matrix)
        save_kernel(OUT / ("best_candidate.json" if name == shortlist[1] else f"{name}.json"), class_map[name], name, matrix_from_classes(class_map[name]), mom)
        full_kernels.append({"candidate": name, **class_map[name], **mom})
    write_csv(OUT / "pc1_pc2_metrics.csv", [r for r in full_rows if r["observable"] in ("PC1", "PC2")])
    write_csv(OUT / "L32to16_full_validation.csv", [r for r in full_rows if r["level"] == "L32toL16"])
    write_csv(OUT / "L64to32_validation.csv", [r for r in full_rows if r["level"] == "L64toL32"])
    write_csv(OUT / "momentum_conditioning.csv", full_kernels)

    # Publication-quality comparison figures for PC1, PC2, action and kurtosis.
    best_name = full_records[1]["candidate"] if len(full_records) > 1 else "current_kernel"
    best_classes = class_map[best_name]
    best_matrix = ETA_SCALE * matrix_from_classes(best_classes)
    best_blocked = observables(block(fine_phi, best_matrix))
    current_blocked = observables(block(fine_phi, ETA_SCALE * matrix_from_classes(current_classes)))
    for key in ("PC1", "PC2", "action_density", "phi2", "phi4", "local_kurtosis_ratio", "NN", "G_pmin_avg"):
        lo = min(float(np.min(direct_obs[key])), float(np.min(current_blocked[key])), float(np.min(best_blocked[key])))
        hi = max(float(np.max(direct_obs[key])), float(np.max(current_blocked[key])), float(np.max(best_blocked[key])))
        fig, ax = plt.subplots(figsize=(5.2, 3.5), constrained_layout=True)
        ax.hist(direct_obs[key], bins=80, range=(lo, hi), density=True, histtype="step", lw=1.5, label="direct L16")
        ax.hist(current_blocked[key], bins=80, range=(lo, hi), density=True, histtype="step", lw=1.2, label="current blocked")
        ax.hist(best_blocked[key], bins=80, range=(lo, hi), density=True, histtype="step", lw=1.2, label="best candidate")
        ax.set_xlabel(key)
        ax.set_ylabel("density")
        ax.legend(frameon=False, fontsize=8)
        fig.savefig(OUT / "figures" / f"{key}_comparison.pdf")
        plt.close(fig)
    ps = np.linspace(-np.pi, np.pi, 512, endpoint=False)
    for mat, label in ((ETA_SCALE * matrix_from_classes(current_classes), "current"), (best_matrix, "best")):
        r = mat.shape[0] // 2
        coords = np.arange(-r, r + 1)
        vals = []
        for p in ps:
            e = np.exp(1j * p * coords)
            vals.append(float(np.real(e @ mat @ e)))
        plt.plot(ps, vals, label=label)
    plt.xlabel("p along diagonal")
    plt.ylabel("K(p)")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(OUT / "figures" / "momentum_conditioning.pdf")
    plt.close()

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=False, capture_output=True, text=True).stdout.strip()
    exact = f"../.venv/bin/python perfect_blocking/scripts/run_kernel_retune_pc1_radial_mode.py"
    (OUT / "exact_commands.txt").write_text(exact + "\n")
    (OUT / "source_commit.txt").write_text(commit + "\n")
    best = full_records[1] if len(full_records) > 1 else full_records[0]
    summary = [
        "# Lambda=1.0 PC1 radial-mode kernel retune",
        "",
        f"Current selected kernel: `{KERNEL_PATH.relative_to(ROOT)}`. All candidate kernels retain D4 symmetry, translational invariance, eta=0.25, stored eta scaling, and base sum 1.",
        "",
        "## Recommendation status",
        "",
        f"Fast scan candidates: `{len(candidates)}`. Full-confirmed candidates: `{len(full_records)}`.",
        f"Best full-confirmed candidate by guardrail-aware score: `{best['candidate']}`; guardrails pass: `{best.get('guard_ok')}`.",
        "The existing upscaling flow was not retrained or silently promoted to a new kernel. Compatibility remains a separate follow-up step after inspecting these blocking-only metrics.",
        "",
        "## Full-confirmation PC1/PC2 table",
        "",
        "| candidate | PC1 shift (native sigma) | PC1 std ratio | PC1 KS | PC2 shift | PC2 std ratio | PC2 KS | guardrails |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for rec in full_records:
        summary.append(f"| `{rec['candidate']}` | {rec['PC1_shift']:.6g} | {rec['PC1_std_ratio']:.6g} | {rec['PC1_KS']:.6g} | {rec['PC2_shift']:.6g} | {rec['PC2_std_ratio']:.6g} | {rec['PC2_KS']:.6g} | {rec['guard_ok']} |")
    summary += [
        "",
        "## Required interpretation",
        "",
        "1. The existing 5x5 family is considered able to remove the mismatch only if a full-confirmed candidate passes the PC1 target and all hard guardrails.",
        "2. `kernel_sensitivity.csv` gives local PC1/PC2 mean and width responses per base orbit coefficient.",
        "3. `L64to32_validation.csv` is the scale guardrail; a candidate that fixes L32->L16 but fails there is not promotable.",
        "4. No production kernel was overwritten. Existing-flow compatibility must be checked before any upscaling run uses a candidate.",
        "",
    ]
    (OUT / "summary.md").write_text("\n".join(summary))
    print(OUT)


if __name__ == "__main__":
    main()
