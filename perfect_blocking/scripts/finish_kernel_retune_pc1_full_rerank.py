#!/usr/bin/env python3
"""Full-ensemble rerank of the stored PC1 kernel scan."""

from __future__ import annotations

import csv
import json

import numpy as np

from run_kernel_retune_pc1_radial_mode import (
    COARSE64_PATH,
    DIRECT_PATH,
    FINE_PATH,
    ETA_SCALE,
    OUT,
    block,
    evaluate,
    guard_ok,
    load_npz,
    matrix_from_classes,
    momentum,
    normalize_classes,
    observables,
    save_kernel,
    score,
)


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    direct = observables(load_npz(DIRECT_PATH))
    fine = load_npz(FINE_PATH)
    l64 = load_npz(COARSE64_PATH)
    candidates = []
    for row in read_csv(OUT / "candidate_kernels.csv"):
        classes = normalize_classes({k: float(row[k]) for k in ("00", "10", "11", "20", "21", "22")})
        candidates.append((row["candidate"], classes))
    fast = {r["candidate"]: r for r in read_csv(OUT / "candidate_metrics.csv")}
    ordered = sorted(candidates, key=lambda x: (abs(float(fast[x[0]]["PC1_shift"])), float(fast[x[0]]["score"])))
    selected = []
    for name, classes in ordered:
        if name not in selected:
            selected.append(name)
        if len(selected) >= 15:
            break
    # Always include the explicit coordinate-response candidates and current baseline.
    for name, classes in candidates:
        if name == "current_kernel" or name.startswith("local_"):
            if name not in selected:
                selected.append(name)
    class_map = dict(candidates)
    full_records = []
    full_rows = []
    for name in selected:
        rec, rows, matrix = evaluate(name, class_map[name], direct, fine, full=True, l64=l64)
        full_records.append(rec)
        full_rows.extend(rows)
    baseline = next(r for r in full_records if r["candidate"] == "current_kernel")
    for rec in full_records:
        rec["score"] = score(rec, baseline)
        rec["guard_ok"] = guard_ok(rec, baseline)
    full_records.sort(key=lambda r: (not r["guard_ok"], r["score"]))
    with (OUT / "candidate_metrics_full.csv").open("w", newline="") as f:
        fields = sorted({k for r in full_records for k in r})
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(full_records)
    with (OUT / "candidate_metrics_full_observables.csv").open("w", newline="") as f:
        fields = sorted({k for r in full_rows for k in r})
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(full_rows)
    with (OUT / "pc1_pc2_metrics.csv").open("w", newline="") as f:
        selected_rows = [r for r in full_rows if r["observable"] in ("PC1", "PC2")]
        fields = sorted({k for r in selected_rows for k in r})
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(selected_rows)
    with (OUT / "L32to16_full_validation.csv").open("w", newline="") as f:
        selected_rows = [r for r in full_rows if r["level"] == "L32toL16"]
        fields = sorted({k for r in selected_rows for k in r})
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(selected_rows)
    with (OUT / "L64to32_validation.csv").open("w", newline="") as f:
        selected_rows = [r for r in full_rows if r["level"] == "L64toL32"]
        fields = sorted({k for r in selected_rows for k in r})
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(selected_rows)
    conditioning = []
    for rec in full_records:
        conditioning.append({"candidate": rec["candidate"], **{k: class_map[rec["candidate"]][k] for k in ("00", "10", "11", "20", "21", "22")}, **{k: rec[k] for k in ("min_K", "max_K", "min_inverse_K", "max_inverse_K", "condition_number")}})
    with (OUT / "momentum_conditioning.csv").open("w", newline="") as f:
        fields = sorted({k for r in conditioning for k in r})
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(conditioning)

    for i, rec in enumerate(full_records[:5]):
        name = rec["candidate"]
        mat = matrix_from_classes(class_map[name])
        save_kernel(OUT / ("best_candidate.json" if i == 0 else f"full_rank_{i}_{name}.json"), class_map[name], name, mat, momentum(ETA_SCALE * mat))

    lines = [
        "# Full-ensemble rerank",
        "",
        f"Candidates reranked on full direct L16 (`{len(next(iter(direct.values())))} configs`) and native L32 (`{len(fine)} configs`) ensembles.",
        "",
        "| rank | candidate | PC1 shift | PC1 std ratio | PC1 KS | PC2 shift | PC2 std ratio | action shift | NN KS | kurtosis KS | guardrails |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for i, r in enumerate(full_records, 1):
        lines.append(f"| {i} | `{r['candidate']}` | {r['PC1_shift']:.6g} | {r['PC1_std_ratio']:.6g} | {r['PC1_KS']:.6g} | {r['PC2_shift']:.6g} | {r['PC2_std_ratio']:.6g} | {r['action_density_shift']:.6g} | {r['NN_KS']:.6g} | {r['local_kurtosis_ratio_KS']:.6g} | {r['guard_ok']} |")
    lines += [
        "",
        "The rerank uses the same blocking-only guardrails and does not modify the production kernel or flow.",
        "",
    ]
    (OUT / "summary_full_rerank.md").write_text("\n".join(lines))
    print(OUT)


if __name__ == "__main__":
    main()
