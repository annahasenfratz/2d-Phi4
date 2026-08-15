#!/usr/bin/env python3
"""Broader 5x5 PC1 scan with relaxed but strictly positive Fourier conditioning."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from run_kernel_retune_pc1_radial_mode import (
    COARSE64_PATH, DIRECT_PATH, ETA_SCALE, FINE_PATH, OUT,
    evaluate, load_npz, matrix_from_classes, momentum, normalize_classes,
    observables, save_kernel,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = sorted({k for r in rows for k in r})
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def relaxed_ok(classes: dict[str, float]) -> tuple[bool, dict[str, float]]:
    m = ETA_SCALE * matrix_from_classes(classes)
    mom = momentum(m)
    # Relax the former safe range substantially, but never permit a zero/negative mode.
    return bool(mom["min_K"] > 0.20 and mom["max_inverse_K"] < 5.0), mom


def objective(rec: dict[str, float]) -> float:
    return (
        120.0 * rec["PC1_shift"] ** 2 + 120.0 * rec["PC2_shift"] ** 2
        + 80.0 * (rec["PC1_std_ratio"] - 1.0) ** 2 + 50.0 * (rec["PC2_std_ratio"] - 1.0) ** 2
        + 30.0 * rec["PC1_KS"] ** 2 + 15.0 * rec["PC2_KS"] ** 2
        + 25.0 * rec["action_density_shift"] ** 2 + 10.0 * (rec["action_density_std_ratio"] - 1.0) ** 2
        + 10.0 * rec["NN_KS"] ** 2 + 10.0 * rec["local_kurtosis_ratio_KS"] ** 2
    )


def main() -> None:
    out = OUT / "relaxed_conditioning"
    out.mkdir(parents=True, exist_ok=True)
    direct_phi, fine_phi, l64_phi = load_npz(DIRECT_PATH), load_npz(FINE_PATH), load_npz(COARSE64_PATH)
    direct = observables(direct_phi)
    nfast = 2000
    direct_fast = {k: v[:nfast] for k, v in direct.items()}
    current = json.loads((Path(__file__).resolve().parents[2] / "perfect_blocking/perfect_blocking_lam1p0/kernels/final/chosen_kernel.json").read_text())
    center = normalize_classes({k: float(v) for k, v in current["base_orbit_classes_before_eta_scale"].items()})
    variables = ["10", "11", "20", "21", "22"]
    rng = np.random.default_rng(2026072022)
    candidates = [("current_kernel", center)]
    for key in variables:
        for scale in (0.005, 0.01, 0.02, 0.04):
            for sign in (-1.0, 1.0):
                c = dict(center); c[key] += sign * scale; candidates.append((f"coord_{key}_{'m' if sign < 0 else 'p'}{scale:g}", normalize_classes(c)))
    for i in range(650):
        scale = float(rng.choice([0.005, 0.008, 0.012, 0.018, 0.025, 0.035, 0.05]))
        c = dict(center)
        for key in variables:
            c[key] += float(rng.normal(0.0, scale))
        candidates.append((f"relaxed_random_{i:04d}_s{scale:g}", normalize_classes(c)))
    fast_records, fast_rows, class_map = [], [], {}
    for name, c in candidates:
        ok, mom = relaxed_ok(c)
        if not ok:
            continue
        rec, rows, _ = evaluate(name, c, direct_fast, fine_phi[:nfast], full=False)
        rec.update({"conditioning_relaxed_ok": True, "conditioning_min_K": mom["min_K"], "conditioning_max_inverse_K": mom["max_inverse_K"], "relaxed_score": objective(rec)})
        rec.update({f"class_{k}": c[k] for k in ("00", "10", "11", "20", "21", "22")})
        fast_records.append(rec); fast_rows.extend(rows); class_map[name] = c
    fast_records.sort(key=lambda r: r["relaxed_score"])
    write_csv(out / "relaxed_candidate_metrics.csv", fast_records)
    write_csv(out / "relaxed_candidate_observables.csv", fast_rows)
    selected = [r["candidate"] for r in fast_records[:20]]
    full_records, full_rows = [], []
    for name in selected:
        rec, rows, mat = evaluate(name, class_map[name], direct, fine_phi, full=True, l64=l64_phi)
        rec["relaxed_score"] = objective(rec)
        rec["conditioning_relaxed_ok"] = True
        full_records.append(rec); full_rows.extend(rows)
    full_records.sort(key=lambda r: r["relaxed_score"])
    write_csv(out / "relaxed_full_metrics.csv", full_records)
    write_csv(out / "relaxed_full_observables.csv", full_rows)
    write_csv(out / "relaxed_L32to16_validation.csv", [r for r in full_rows if r["level"] == "L32toL16"])
    write_csv(out / "relaxed_L64to32_validation.csv", [r for r in full_rows if r["level"] == "L64toL32"])
    conditioning = []
    for r in full_records:
        conditioning.append({"candidate": r["candidate"], **{k: class_map[r["candidate"]][k] for k in ("00", "10", "11", "20", "21", "22")}, **{k: r[k] for k in ("min_K", "max_K", "min_inverse_K", "max_inverse_K", "condition_number")}})
    write_csv(out / "relaxed_momentum_conditioning.csv", conditioning)
    for i, r in enumerate(full_records[:5]):
        name = r["candidate"]; c = class_map[name]; mat = matrix_from_classes(c)
        save_kernel(out / ("relaxed_best_candidate.json" if i == 0 else f"relaxed_rank{i}_{name}.json"), c, name, mat, momentum(ETA_SCALE * mat))
    lines = [
        "# Relaxed-conditioning PC1 retune",
        "",
        "The search required `min K(p) > 0.20` and `max 1/K(p) < 5.0`; all retained candidates remain strictly invertible.",
        "",
        "| rank | candidate | PC1 shift | PC1 std ratio | PC1 KS | PC2 shift | PC2 std ratio | action shift | NN KS | kurtosis KS | min K | max 1/K |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for i, r in enumerate(full_records, 1):
        lines.append(f"| {i} | `{r['candidate']}` | {r['PC1_shift']:.6g} | {r['PC1_std_ratio']:.6g} | {r['PC1_KS']:.6g} | {r['PC2_shift']:.6g} | {r['PC2_std_ratio']:.6g} | {r['action_density_shift']:.6g} | {r['NN_KS']:.6g} | {r['local_kurtosis_ratio_KS']:.6g} | {r['min_K']:.6g} | {r['max_inverse_K']:.6g} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print(out)


if __name__ == "__main__":
    main()
