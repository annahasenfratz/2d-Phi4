#!/usr/bin/env python3
from __future__ import annotations

import csv
import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path("perfect_blocking_upsampling/outputs/controlled_patch_lam0p2/flow_detail_rethermalization_L16to32_latest/runs")
PATTERN = "prod_cd_dL16_kc0p3235_*"
OUT_NAME = "combined_prod_cd_dL16_kc0p3235_N2768_mixed_Finit_to_sweep600_observables_20260715"

PER_CHAIN_CSVS = [
    "observables/main_per_sweep_measurements.csv",
    "observables/per_sweep_observables.csv",
    "observables/blocked_consistency.csv",
    "per_sweep_observables.csv",
    "blocked_consistency.csv",
]

PER_SWEEP_CSVS = [
    "observables/acceptance_history.csv",
    "acceptance_history.csv",
    "observables/ensemble_average_history.csv",
    "ensemble_average_history.csv",
]


def json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(type(obj).__name__)


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sweeps_for(run: Path, rel: str) -> set[int]:
    path = run / rel
    if not path.exists():
        return set()
    _fields, rows = read_csv(path)
    return {int(float(r["sweep"])) for r in rows if r.get("sweep", "")}


def chain_ids_for(run: Path, rel: str, sweeps: set[int]) -> list[int]:
    path = run / rel
    _fields, rows = read_csv(path)
    out = sorted({int(float(r["chain_id"])) for r in rows if int(float(r["sweep"])) in sweeps and r.get("chain_id", "")})
    return out


def chain_maps(run_dirs: list[Path], sweeps: set[int]) -> dict[str, dict[int, int]]:
    offset = 0
    maps: dict[str, dict[int, int]] = {}
    for run in run_dirs:
        ids = chain_ids_for(run, "observables/main_per_sweep_measurements.csv", sweeps)
        maps[run.name] = {old: offset + j for j, old in enumerate(ids)}
        offset += len(ids)
    return maps


def field_union(existing: list[str], fields: list[str]) -> list[str]:
    out = list(existing)
    for field in fields:
        if field not in out:
            out.append(field)
    return out


def combine_per_chain(out_dir: Path, rel: str, run_dirs: list[Path], sweeps: set[int], maps: dict[str, dict[int, int]]) -> int:
    extra = ["source_run_index", "source_run", "original_chain_id"]
    fieldnames: list[str] = []
    out_rows: list[dict[str, Any]] = []
    for run_index, run in enumerate(run_dirs):
        path = run / rel
        if not path.exists():
            continue
        fields, rows = read_csv(path)
        fieldnames = field_union(fieldnames, fields)
        cmap = maps[run.name]
        for row in rows:
            sweep = int(float(row["sweep"]))
            if sweep not in sweeps:
                continue
            old_chain = int(float(row["chain_id"]))
            if old_chain not in cmap:
                continue
            new = dict(row)
            new["source_run_index"] = run_index
            new["source_run"] = run.name
            new["original_chain_id"] = old_chain
            new["chain_id"] = cmap[old_chain]
            out_rows.append(new)
    if out_rows:
        write_csv(out_dir / rel, extra + [f for f in fieldnames if f not in extra], out_rows)
    return len(out_rows)


def combine_source_per_sweep(out_dir: Path, rel: str, run_dirs: list[Path], sweeps: set[int]) -> int:
    extra = ["source_run_index", "source_run"]
    fieldnames: list[str] = []
    out_rows: list[dict[str, Any]] = []
    for run_index, run in enumerate(run_dirs):
        path = run / rel
        if not path.exists():
            continue
        fields, rows = read_csv(path)
        fieldnames = field_union(fieldnames, fields)
        for row in rows:
            sweep = int(float(row["sweep"]))
            if sweep not in sweeps:
                continue
            new = dict(row)
            new["source_run_index"] = run_index
            new["source_run"] = run.name
            out_rows.append(new)
    if out_rows:
        name = Path(rel)
        out_rel = name.with_name("source_" + name.name)
        write_csv(out_dir / out_rel, extra + [f for f in fieldnames if f not in extra], out_rows)
    return len(out_rows)


def combine_acceptance(out_dir: Path, rel: str, run_dirs: list[Path], common_sweeps: list[int]) -> int:
    fieldnames: list[str] | None = None
    by_sweep: dict[int, list[dict[str, str]]] = {s: [] for s in common_sweeps}
    for run in run_dirs:
        path = run / rel
        if not path.exists():
            continue
        fields, rows = read_csv(path)
        fieldnames = fieldnames or fields
        for row in rows:
            sweep = int(float(row["sweep"]))
            if sweep in by_sweep:
                by_sweep[sweep].append(row)
    if not fieldnames:
        return 0
    sum_keys = [
        "coarse_proposals",
        "coarse_accepts",
        "coarse_proposals_cumulative",
        "coarse_accepts_cumulative",
        "detail_proposals",
        "detail_accepts",
        "detail_proposals_cumulative",
        "detail_accepts_cumulative",
        "conditional_flow_refreshes",
    ]
    out_rows: list[dict[str, Any]] = []
    for sweep in common_sweeps:
        rows = by_sweep[sweep]
        if not rows:
            continue
        out: dict[str, Any] = {k: rows[0].get(k, "") for k in fieldnames}
        out["sweep"] = sweep
        out["update_mode"] = "combined"
        for key in sum_keys:
            if key in fieldnames:
                out[key] = sum(float(r.get(key, "") or 0.0) for r in rows)
        for rate, acc_key, prop_key in [
            ("coarse_acceptance", "coarse_accepts", "coarse_proposals"),
            ("coarse_acceptance_cumulative", "coarse_accepts_cumulative", "coarse_proposals_cumulative"),
            ("detail_acceptance", "detail_accepts", "detail_proposals"),
            ("detail_acceptance_cumulative", "detail_accepts_cumulative", "detail_proposals_cumulative"),
        ]:
            if rate in fieldnames:
                props = float(out.get(prop_key, 0.0) or 0.0)
                acc = float(out.get(acc_key, 0.0) or 0.0)
                out[rate] = acc / props if props else math.nan
        out_rows.append(out)
    if out_rows:
        write_csv(out_dir / rel, fieldnames, out_rows)
    return len(out_rows)


def mirror_complete_observable_csvs(out_dir: Path) -> None:
    for src_rel, dst_rel in [
        ("observables/main_per_sweep_measurements.csv", "main_per_sweep_measurements.csv"),
        ("observables/per_sweep_observables.csv", "per_sweep_observables.csv"),
        ("observables/blocked_consistency.csv", "blocked_consistency.csv"),
        ("observables/acceptance_history.csv", "acceptance_history.csv"),
    ]:
        src = out_dir / src_rel
        if src.exists():
            dst = out_dir / dst_rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(src.read_bytes())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--pattern", default=PATTERN)
    ap.add_argument("--out-name", default=OUT_NAME)
    args = ap.parse_args()
    out_dir = args.root / args.out_name

    run_dirs = sorted(p for p in args.root.glob(args.pattern) if p.is_dir())
    run_dirs = [p for p in run_dirs if (p / "observables" / "main_per_sweep_measurements.csv").exists()]
    if not run_dirs:
        raise SystemExit(f"no matching runs with main observables under {args.root}")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"refusing to overwrite nonempty output dir: {out_dir}")
    (out_dir / "observables").mkdir(parents=True, exist_ok=True)
    main_sweep_sets = [sweeps_for(run, "observables/main_per_sweep_measurements.csv") for run in run_dirs]
    common_sweeps = sorted(set.intersection(*main_sweep_sets))
    if not common_sweeps:
        raise SystemExit("no common sweeps")
    common_sweeps = [s for s in common_sweeps if s <= max(common_sweeps)]
    common_set = set(common_sweeps)
    maps = chain_maps(run_dirs, common_set)
    counts_by_run = {run.name: len(maps[run.name]) for run in run_dirs}

    row_counts: dict[str, int] = {}
    for rel in PER_CHAIN_CSVS:
        row_counts[rel] = combine_per_chain(out_dir, rel, run_dirs, common_set, maps)
    for rel in PER_SWEEP_CSVS:
        if "acceptance_history.csv" in rel:
            row_counts[rel] = combine_acceptance(out_dir, rel, run_dirs, common_sweeps)
        row_counts["source_" + rel] = combine_source_per_sweep(out_dir, rel, run_dirs, common_set)
    mirror_complete_observable_csvs(out_dir)
    for rel in [
        "main_per_sweep_measurements.csv",
        "per_sweep_observables.csv",
        "blocked_consistency.csv",
        "acceptance_history.csv",
    ]:
        if (out_dir / rel).exists():
            with (out_dir / rel).open(newline="", encoding="utf-8") as f:
                row_counts[rel] = sum(1 for _ in csv.DictReader(f))

    manifest = {
        "status": "completed",
        "combined_run_name": out_dir.name,
        "source_runs": [str(p) for p in run_dirs],
        "source_run_names": [p.name for p in run_dirs],
        "n_source_runs": len(run_dirs),
        "common_sweeps": common_sweeps,
        "last_common_sweep": max(common_sweeps),
        "n_common_sweeps": len(common_sweeps),
        "chains_by_run": counts_by_run,
        "total_chains": int(sum(counts_by_run.values())),
        "row_counts": row_counts,
        "notes": "Observable CSVs only; no checkpoints copied or combined. Per-chain files reindex chain_id and retain original_chain_id/source_run provenance.",
    }
    (out_dir / "combined_observables_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=json_default, allow_nan=True) + "\n", encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "last_common_sweep": max(common_sweeps), "n_common_sweeps": len(common_sweeps), "total_chains": manifest["total_chains"], "row_counts": row_counts}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
