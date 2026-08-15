#!/usr/bin/env python3
"""Combine four lambda=1.0 L32->L64 observable chains through sweep 300."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(
    "perfect_blocking_upsampling/outputs/controlled_patch_lam1p0/coarse_detail_L32to64"
)
RUNS = [
    ROOT / "prod_cd_bL64_RQS_cfg0_N250_Pc16x1_Pd16x16_kf0p340301_kc0p340301",
    ROOT / "prod_cd_bL64_RQS_cfg1000_N250_Pc16x1_Pd16x16_kf0p340301_kc0p340301",
    ROOT / "prod_cd_bL64_RQS_cfg2000_N250_Pc16x1_Pd16x16_kf0p340301_kc0p340301",
    ROOT / "prod_cd_bL64_RQS_cfg3000_N250_Pc16x1_Pd16x16_kf0p340301_kc0p340301",
]
OUT = ROOT / "combined_cfg0_cfg1000_cfg2000_cfg3000_N1000_to_sweep300"
MAX_SWEEP = 300
CHAIN_FILES = [
    "per_sweep_observables.csv",
    "main_per_sweep_measurements.csv",
    "Gk_per_sweep_measurements.csv",
]


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def combine_chain_file(name: str) -> tuple[int, set[int]]:
    output_rows: list[dict[str, str]] = []
    fields: list[str] | None = None
    chain_offset = 0
    sweeps: set[int] = set()

    for run in RUNS:
        run_fields, rows = read_csv(run / "observables" / name)
        if fields is None:
            fields = run_fields
        run_chain_ids = set()
        for row in rows:
            if "sweep" in row and int(float(row["sweep"])) > MAX_SWEEP:
                continue
            if "chain_id" not in row or row["chain_id"] == "":
                continue
            old_chain = int(float(row["chain_id"]))
            run_chain_ids.add(old_chain)
            new = dict(row)
            new["chain_id"] = str(old_chain + chain_offset)
            output_rows.append(new)
            if "sweep" in row and row["sweep"] != "":
                sweeps.add(int(float(row["sweep"])))
        if run_chain_ids:
            chain_offset += max(run_chain_ids) + 1

    if fields is None:
        raise ValueError(f"no fields found for {name}")
    write_csv(OUT / "observables" / name, fields, output_rows)
    return len(output_rows), sweeps


def main() -> None:
    missing = [str(run) for run in RUNS if not run.is_dir()]
    if missing:
        raise FileNotFoundError(missing)
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")

    (OUT / "observables").mkdir(parents=True)
    row_counts: dict[str, int] = {}
    sweep_sets: list[set[int]] = []
    for name in CHAIN_FILES:
        rows, sweeps = combine_chain_file(name)
        row_counts[name] = rows
        sweep_sets.append(sweeps)

    common_sweeps = sorted(set.intersection(*sweep_sets))
    manifest = {
        "status": "completed",
        "description": "Combined four lambda=1.0 L32->L64 observable chains through sweep 300.",
        "source_runs": [str(run) for run in RUNS],
        "n_source_runs": len(RUNS),
        "chains_per_run": 250,
        "total_chains": len(RUNS) * 250,
        "max_sweep_included": MAX_SWEEP,
        "common_measured_sweeps": common_sweeps,
        "row_counts": row_counts,
        "combined_files": CHAIN_FILES,
        "chain_id_range": [0, len(RUNS) * 250 - 1],
    }
    (OUT / "combined_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
