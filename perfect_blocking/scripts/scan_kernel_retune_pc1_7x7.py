#!/usr/bin/env python3
"""Modest 7x7 D4 expansion for the lambda=1.0 PC1 radial retune."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from run_kernel_retune_pc1_radial_mode import (
    COARSE64_PATH, DIRECT_PATH, FINE_PATH, ETA_SCALE, OUT,
    block, load_npz, metric, momentum, observables, PC1, PC2,
)

ROOT = Path(__file__).resolve().parents[2]
ORBIT_KEYS = ("00", "10", "11", "20", "21", "22", "30", "31", "32", "33")
MULT = {"00": 1, "10": 4, "11": 4, "20": 4, "21": 8, "22": 4, "30": 4, "31": 8, "32": 8, "33": 4}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def matrix(classes: dict[str, float]) -> np.ndarray:
    out = np.zeros((7, 7), dtype=np.float64)
    for i, dx in enumerate(range(-3, 4)):
        for j, dy in enumerate(range(-3, 4)):
            out[i, j] = classes[f"{max(abs(dx), abs(dy))}{min(abs(dx), abs(dy))}"]
    return out


def normalize(c: dict[str, float]) -> dict[str, float]:
    c = dict(c)
    c["00"] = 1.0 - sum(MULT[k] * c[k] for k in ORBIT_KEYS if k != "00")
    return c


def embed_current() -> dict[str, float]:
    with (ROOT / "perfect_blocking/perfect_blocking_lam1p0/kernels/final/chosen_kernel.json").open() as f:
        d = json.load(f)
    base5 = np.asarray(d["base_matrix_before_eta_scale"], dtype=np.float64)
    m = np.zeros((7, 7), dtype=np.float64)
    m[1:6, 1:6] = base5
    c = {k: 0.0 for k in ORBIT_KEYS}
    for dx in range(4):
        for dy in range(dx + 1):
            c[f"{dx}{dy}"] = float(m[3 + dx, 3 + dy])
    return normalize(c)


def eval_candidate(name: str, c: dict[str, float], direct: dict[str, np.ndarray], fine: np.ndarray, l64: np.ndarray | None = None):
    b = observables(block(fine, ETA_SCALE * matrix(c)))
    rec = {"candidate": name, **c, **momentum(ETA_SCALE * matrix(c))}
    rows = []
    for key in ("PC1", "PC2", "action_density", "NN", "diag", "2nn", "phi2", "phi4", "local_kurtosis_ratio", "m2", "m4", "G_pmin_avg"):
        m = metric(direct[key], b[key])
        rows.append({"candidate": name, "level": "L32toL16", "observable": key, **m})
        rec[f"{key}_shift"] = m["standardized_mean_shift"]
        rec[f"{key}_std_ratio"] = m["std_ratio_blocked_over_direct"]
        rec[f"{key}_KS"] = m["KS"]
    if l64 is not None:
        d32 = observables(load_npz(FINE_PATH))
        b64 = observables(block(l64, ETA_SCALE * matrix(c)))
        for key in ("PC1", "PC2", "action_density", "NN", "phi2", "phi4", "local_kurtosis_ratio", "G_pmin_avg"):
            rows.append({"candidate": name, "level": "L64toL32", "observable": key, **metric(d32[key], b64[key])})
    rec["score"] = (
        100 * rec["PC1_shift"] ** 2 + 100 * rec["PC2_shift"] ** 2
        + 50 * (rec["PC1_std_ratio"] - 1) ** 2 + 50 * (rec["PC2_std_ratio"] - 1) ** 2
        + 20 * rec["PC1_KS"] ** 2 + 10 * rec["PC2_KS"] ** 2
        + 20 * rec["action_density_shift"] ** 2
        + 10 * max(0, rec["NN_KS"] - 0.04) ** 2
        + 10 * max(0, rec["local_kurtosis_ratio_KS"] - 0.09) ** 2
    )
    rec["guard_ok"] = bool(
        abs(rec["PC1_shift"]) < .03 and abs(rec["PC1_std_ratio"] - 1) < .02 and rec["PC1_KS"] < .04
        and abs(rec["PC2_shift"]) < .05 and abs(rec["PC2_std_ratio"] - 1) < .05
        and abs(rec["action_density_shift"]) < .05 and abs(rec["action_density_std_ratio"] - 1) < .03
        and rec["NN_KS"] < .04 and rec["local_kurtosis_ratio_KS"] <= .09
        and abs(rec["G_pmin_avg_shift"]) < .05 and rec["min_K"] > 0 and rec["max_inverse_K"] <= 1.50
    )
    return rec, rows


def write(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = sorted({k for r in rows for k in r})
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def main() -> None:
    direct_phi, fine_phi, l64_phi = load_npz(DIRECT_PATH), load_npz(FINE_PATH), load_npz(COARSE64_PATH)
    direct = observables(direct_phi)
    fast_n = 2000
    direct_fast = {k: v[:fast_n] for k, v in direct.items()}
    base = embed_current()
    rng = np.random.default_rng(2026072021)
    candidates = [("expanded_7x7_embedded_current", base)]
    # Include previously recorded 7x7 candidates as external baselines.
    for p in [
        ROOT / "perfect_blocking/perfect_blocking_lam1p0/kernels/selected_for_upscaling/current_final_7x7_no33_nn_constrained_eta_included.json",
        ROOT / "perfect_blocking/perfect_blocking_lam1p0/kernels/candidates/7x7_phi2_phi4_width095_kurtguard_20260719/best_7x7_no33_phi2_phi4_width095_kurtguard_eta_included.json",
    ]:
        if p.exists():
            d = json.loads(p.read_text())
            candidates.append((d.get("name", p.stem), normalize({k: float(v) for k, v in d["base_orbit_classes_before_eta_scale"].items()})))
    vars_ = list(ORBIT_KEYS[1:])
    for i in range(300):
        scale = float(rng.choice([0.0001, 0.0002, 0.0004, 0.0007, 0.0012, 0.002]))
        c = dict(base)
        for k in vars_:
            c[k] += float(rng.normal(0, scale))
        candidates.append((f"expanded_random_{i:04d}_s{scale:g}", normalize(c)))
    fast_records, fast_rows = [], []
    for name, c in candidates:
        r, rows = eval_candidate(name, c, direct_fast, fine_phi[:fast_n])
        fast_records.append(r); fast_rows.extend(rows)
    fast_records.sort(key=lambda r: r["score"])
    write(OUT / "expanded_7x7_candidate_metrics.csv", fast_records)
    write(OUT / "expanded_7x7_candidate_observables.csv", fast_rows)
    class_map = {name: c for name, c in candidates}
    selected = [r["candidate"] for r in fast_records[:12]]
    full_records, full_rows = [], []
    for name in selected:
        r, rows = eval_candidate(name, class_map[name], direct, fine_phi, l64_phi)
        full_records.append(r); full_rows.extend(rows)
    full_records.sort(key=lambda r: (not r["guard_ok"], r["score"]))
    write(OUT / "expanded_7x7_full_metrics.csv", full_records)
    write(OUT / "expanded_7x7_full_observables.csv", full_rows)
    write(OUT / "expanded_7x7_L32to16_validation.csv", [r for r in full_rows if r["level"] == "L32toL16"])
    write(OUT / "expanded_7x7_L64to32_validation.csv", [r for r in full_rows if r["level"] == "L64toL32"])
    for i, r in enumerate(full_records[:5]):
        name = r["candidate"]
        p = OUT / ("expanded_7x7_best_candidate.json" if i == 0 else f"expanded_7x7_rank{i}_{name}.json")
        p.write_text(json.dumps({"name": name, "family": "7x7_D4", "lambda": 1.0, "kappa_f": .340301, "kappa_c": .340301, "eta": .25, "eta_scale_numeric": ETA_SCALE, "kernel_coefficients_include_eta_scale": True, "base_orbit_classes_before_eta_scale": class_map[name], "base_matrix_before_eta_scale": matrix(class_map[name]).tolist(), "matrix": (ETA_SCALE * matrix(class_map[name])).tolist(), "momentum_stability": momentum(ETA_SCALE * matrix(class_map[name]))}, indent=2) + "\n")
    lines = ["# Expanded 7x7 PC1 retune", "", "| rank | candidate | PC1 shift | PC1 std ratio | PC1 KS | PC2 shift | PC2 std ratio | action shift | NN KS | kurtosis KS | guardrails |", "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for i, r in enumerate(full_records, 1):
        lines.append(f"| {i} | `{r['candidate']}` | {r['PC1_shift']:.6g} | {r['PC1_std_ratio']:.6g} | {r['PC1_KS']:.6g} | {r['PC2_shift']:.6g} | {r['PC2_std_ratio']:.6g} | {r['action_density_shift']:.6g} | {r['NN_KS']:.6g} | {r['local_kurtosis_ratio_KS']:.6g} | {r['guard_ok']} |")
    (OUT / "expanded_7x7_summary.md").write_text("\n".join(lines) + "\n")
    print(OUT)


if __name__ == "__main__":
    main()
