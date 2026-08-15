#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

RUN_NAMES = [
    "prod_cd_dL32_kc0p323124_N64_Finit50_Pc16x1_ec0p06_Pd16x20_jac_20260713_054808",
    "prod_cd_dL32_kc0p323124_N500_Finit50_Pc16x1_ec0p06_Pd16x20_jac_20260713_123942",
    "prod_cd_dL32_kc0p323124_N500_Finit50_Pc16x1_ec0p06_Pd16x20_jac_20260713_124321",
    "prod_cd_dL32_kc0p323124_N500_Finit50_Pc16x1_ec0p06_Pd16x20_jac_20260713_124454",
    "prod_cd_dL32_kc0p323124_N500_Finit50_Pc16x1_ec0p06_Pd16x20_jac_20260713_124648",
]

ROOT = Path("perfect_blocking_upsampling/outputs/controlled_patch_lam0p2/flow_detail_rethermalization_L32to64_latest/runs")
OUT_NAME = "combined_prod_cd_dL32_kc0p323124_N2064_Finit50_Pc16x1_ec0p06_Pd16x20_jac_to_sweep325_final_configs_20260714"
OUT = ROOT / OUT_NAME

PER_CHAIN_CSVS = [
    "per_sweep_observables.csv",
    "observables/per_sweep_observables.csv",
    "observables/main_per_sweep_measurements.csv",
    "blocked_consistency.csv",
    "observables/blocked_consistency.csv",
    "observables/reblocking_diagnostics.csv",
    "observables/coarse_weight_history.csv",
]

ACCEPTANCE_CSVS = ["acceptance_history.csv", "observables/acceptance_history.csv"]


def json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(type(obj).__name__)


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return list(r.fieldnames or []), list(r)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def sweep_from_npz(path: Path) -> int:
    return int(path.stem.replace("configs_sweep", ""))


def finite_mask_npz(path: Path) -> np.ndarray:
    with np.load(path) as z:
        n = len(z["phi"])
        mask = np.ones(n, dtype=bool)
        for key in ("phi", "psi"):
            if key in z.files:
                arr = z[key]
                mask &= np.isfinite(arr.reshape(n, -1)).all(axis=1)
        for key in ("source_native_L32_index", "source_coarse_action_nn", "source_coarse_log_weight_nn"):
            if key in z.files:
                arr = z[key]
                mask &= np.isfinite(arr.reshape(n, -1)).all(axis=1)
        return mask


def available_config_sweeps(run_dir: Path) -> set[int]:
    return {sweep_from_npz(p) for p in (run_dir / "checkpoints_or_final_configs").glob("configs_sweep*.npz")}


def valid_indices_for_run(run_dir: Path, sweep: int) -> np.ndarray:
    path = run_dir / "checkpoints_or_final_configs" / f"configs_sweep{sweep:03d}.npz"
    return np.flatnonzero(finite_mask_npz(path))


def combine_npz(run_dirs: list[Path], valid_by_run: dict[str, np.ndarray], sweeps: list[int]) -> dict[int, int]:
    rows_per_sweep: dict[int, int] = {}
    out_dir = OUT / "checkpoints_or_final_configs"
    out_dir.mkdir(parents=True, exist_ok=True)
    for sweep in [max(sweeps)]:
        payload: dict[str, list[np.ndarray]] = {}
        for run_dir in run_dirs:
            valid = valid_by_run[run_dir.name]
            path = run_dir / "checkpoints_or_final_configs" / f"configs_sweep{sweep:03d}.npz"
            with np.load(path) as z:
                n = len(z["phi"])
                for key in z.files:
                    arr = z[key]
                    if arr.shape[:1] == (n,):
                        piece = arr[valid]
                    else:
                        piece = arr
                    payload.setdefault(key, []).append(piece)
        combined: dict[str, np.ndarray] = {}
        for key, pieces in payload.items():
            if pieces[0].ndim >= 1 and all(p.shape[1:] == pieces[0].shape[1:] for p in pieces):
                combined[key] = np.concatenate(pieces, axis=0)
            else:
                combined[key] = pieces[0]
        rows_per_sweep[sweep] = int(len(combined["phi"]))
        np.savez(out_dir / f"configs_sweep{sweep:03d}.npz", **combined)
    return rows_per_sweep


def chain_maps(valid_by_run: dict[str, np.ndarray]) -> dict[str, dict[int, int]]:
    offset = 0
    out: dict[str, dict[int, int]] = {}
    for name in RUN_NAMES:
        valid = valid_by_run[name]
        out[name] = {int(old): int(offset + j) for j, old in enumerate(valid)}
        offset += len(valid)
    return out


def combine_per_chain_csv(rel: str, run_dirs: list[Path], maps: dict[str, dict[int, int]], sweeps: set[int]) -> None:
    fieldnames: list[str] | None = None
    out_rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        path = run_dir / rel
        if not path.exists():
            continue
        fields, rows = read_csv(path)
        if fieldnames is None:
            fieldnames = fields
        cmap = maps[run_dir.name]
        for row in rows:
            if "sweep" in row and int(float(row["sweep"])) not in sweeps:
                continue
            if "chain_id" not in row:
                continue
            old_chain = int(float(row["chain_id"]))
            if old_chain not in cmap:
                continue
            new = dict(row)
            new["chain_id"] = cmap[old_chain]
            out_rows.append(new)
    if fieldnames is not None:
        write_csv(OUT / rel, fieldnames, out_rows)


def combine_acceptance_csv(rel: str, run_dirs: list[Path], sweeps: list[int]) -> None:
    fieldnames: list[str] | None = None
    by_sweep: dict[int, list[dict[str, str]]] = {s: [] for s in sweeps}
    for run_dir in run_dirs:
        path = run_dir / rel
        if not path.exists():
            continue
        fields, rows = read_csv(path)
        if fieldnames is None:
            fieldnames = fields
        for row in rows:
            sweep = int(float(row["sweep"]))
            if sweep in by_sweep:
                by_sweep[sweep].append(row)
    if fieldnames is None:
        return
    proposal_keys = [
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
    for sweep in sweeps:
        rows = by_sweep[sweep]
        if not rows:
            continue
        out: dict[str, Any] = {k: rows[0].get(k, "") for k in fieldnames}
        out["sweep"] = sweep
        for key in proposal_keys:
            if key in fieldnames:
                out[key] = sum(float(r.get(key, "0") or 0.0) for r in rows)
        if "coarse_acceptance" in fieldnames:
            props = float(out.get("coarse_proposals", 0.0) or 0.0)
            acc = float(out.get("coarse_accepts", 0.0) or 0.0)
            out["coarse_acceptance"] = acc / props if props else float("nan")
        if "coarse_acceptance_cumulative" in fieldnames:
            props = float(out.get("coarse_proposals_cumulative", 0.0) or 0.0)
            acc = float(out.get("coarse_accepts_cumulative", 0.0) or 0.0)
            out["coarse_acceptance_cumulative"] = acc / props if props else float("nan")
        if "detail_acceptance" in fieldnames:
            props = float(out.get("detail_proposals", 0.0) or 0.0)
            acc = float(out.get("detail_accepts", 0.0) or 0.0)
            out["detail_acceptance"] = acc / props if props else float("nan")
        if "detail_acceptance_cumulative" in fieldnames:
            props = float(out.get("detail_proposals_cumulative", 0.0) or 0.0)
            acc = float(out.get("detail_accepts_cumulative", 0.0) or 0.0)
            out["detail_acceptance_cumulative"] = acc / props if props else float("nan")
        out_rows.append(out)
    write_csv(OUT / rel, fieldnames, out_rows)


def copy_reference_files(first: Path) -> None:
    for dirname in ("configs", "scripts", "manifests"):
        src = first / dirname
        dst = OUT / dirname
        if src.exists() and not dst.exists():
            shutil.copytree(src, dst)
    for rel in ("run_manifest.json",):
        src = first / rel
        if src.exists():
            shutil.copy2(src, OUT / rel)


def mirror_complete_observable_csvs() -> None:
    for src_rel, dst_rel in [
        ("observables/per_sweep_observables.csv", "per_sweep_observables.csv"),
        ("observables/blocked_consistency.csv", "blocked_consistency.csv"),
    ]:
        src = OUT / src_rel
        if src.exists():
            shutil.copy2(src, OUT / dst_rel)


def main() -> int:
    run_dirs = [ROOT / name for name in RUN_NAMES]
    missing = [str(p) for p in run_dirs if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing requested runs: {missing}")
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output dir: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    for sub in ("observables", "checkpoints_or_final_configs", "logs", "plots"):
        (OUT / sub).mkdir(exist_ok=True)

    sweep_sets = [available_config_sweeps(d) for d in run_dirs]
    common_sweeps = sorted(set.intersection(*sweep_sets))
    common_sweeps = [s for s in common_sweeps if s <= max(common_sweeps)]
    last_common = max(common_sweeps)
    valid_by_run = {d.name: valid_indices_for_run(d, last_common) for d in run_dirs}
    maps = chain_maps(valid_by_run)

    rows_per_sweep = combine_npz(run_dirs, valid_by_run, common_sweeps)
    for rel in PER_CHAIN_CSVS:
        combine_per_chain_csv(rel, run_dirs, maps, set(common_sweeps))
    for rel in ACCEPTANCE_CSVS:
        combine_acceptance_csv(rel, run_dirs, common_sweeps)
    mirror_complete_observable_csvs()
    copy_reference_files(run_dirs[0])

    summary = {
        "status": "completed",
        "combined_run_name": OUT.name,
        "source_runs": [str(d) for d in run_dirs],
        "common_sweeps": common_sweeps,
        "last_common_sweep": last_common,
        "valid_counts_by_run": {name: int(len(valid)) for name, valid in valid_by_run.items()},
        "total_valid_configs": int(sum(len(v) for v in valid_by_run.values())),
        "rows_per_sweep": rows_per_sweep,
        "skipped_counts_by_run": {
            name: int(max(available_config_sweeps(ROOT / name) and finite_mask_npz(ROOT / name / "checkpoints_or_final_configs" / f"configs_sweep{last_common:03d}.npz").shape[0] or 0, 0) - len(valid))
            for name, valid in valid_by_run.items()
        },
    }
    (OUT / "combined_manifest.json").write_text(json.dumps(summary, indent=2, default=json_default, allow_nan=True) + "\n", encoding="utf-8")
    (OUT / "progress.json").write_text(json.dumps({"status": "completed", "completed_sweeps": last_common, "completed_configs": summary["total_valid_configs"], "summary": summary}, indent=2, default=json_default, allow_nan=True) + "\n", encoding="utf-8")
    print(json.dumps({"out_dir": str(OUT), "last_common_sweep": last_common, "total_valid_configs": summary["total_valid_configs"], "valid_counts_by_run": summary["valid_counts_by_run"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
