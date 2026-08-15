#!/usr/bin/env python3
"""Combine the lambda=1.0 L16->L32 observable chains."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path


ROOT = Path("perfect_blocking_upsampling/outputs/controlled_patch_lam1p0/coarse_detail_L16to32")
RUNS = [
    ROOT / "prod_cd_bL32_RQS_cfg0_N500_Pc16x1_Pd16x16_kf0p340301_kc0p340301",
    ROOT / "prod_cd_bL32_RQS_cfg500_N500_Pc16x1_Pd16x16_kf0p340301_kc0p340301",
    ROOT / "prod_cd_bL32_RQS_cfg1000_N500_Pc16x1_Pd16x16_kf0p340301_kc0p340301",
    ROOT / "prod_cd_bL32_RQS_cfg1500_N500_Pc16x1_Pd16x16_kf0p340301_kc0p340301",
    ROOT / "prod_cd_bL32_RQS_cfg2000_N500_Pc16x1_Pd16x16_kf0p340301_kc0p340301",
    ROOT / "prod_cd_bL32_RQS_cfg2500_N500_Pc16x1_Pd16x16_kf0p340301_kc0p340301",
    ROOT / "prod_cd_bL32_RQS_cfg3000_N500_Pc16x1_Pd16x16_kf0p340301_kc0p340301",
    ROOT / "prod_cd_bL32_RQS_cfg3500_N500_Pc16x1_Pd16x16_kf0p340301_kc0p340301",
]
OUT = ROOT / "combined_cfg0_cfg1000_cfg2000_cfg2500_cfg3000_N5000"
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


def combine_chain_file(name: str) -> int:
    output_rows: list[dict[str, str]] = []
    fields: list[str] | None = None
    chain_offset = 0
    for run in RUNS:
        run_fields, rows = read_csv(run / "observables" / name)
        if fields is None:
            fields = run_fields
        for row in rows:
            row = dict(row)
            if "chain_id" in row:
                row["chain_id"] = str(int(float(row["chain_id"])) + chain_offset)
            output_rows.append(row)
        chain_ids = [int(float(row["chain_id"])) for row in rows if row.get("chain_id", "") != ""]
        chain_offset += max(chain_ids) + 1 if chain_ids else 0
    assert fields is not None
    write_csv(OUT / "observables" / name, fields, output_rows)
    return len(output_rows)


def main() -> None:
    missing = [str(run) for run in RUNS if not run.is_dir()]
    if missing:
        raise FileNotFoundError(missing)
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    (OUT / "observables").mkdir(parents=True)
    for name in CHAIN_FILES:
        combine_chain_file(name)
    for name in ["native_L32_reference_summary.csv", "sweep_native_comparison.csv"]:
        source = RUNS[0] / "observables" / name
        if source.exists():
            shutil.copy2(source, OUT / "observables" / name)
    manifest = {
        "status": "completed",
        "description": "Combined all chains from eight lambda=1.0 L16->L32 runs; columns and sweep numbering preserved.",
        "source_runs": [str(run) for run in RUNS],
        "n_source_runs": len(RUNS),
        "chains_per_run": 500,
        "total_chains": len(RUNS) * 500,
        "measured_sweeps_per_run": 61,
        "chain_id_ranges": [[i * 500, (i + 1) * 500 - 1] for i in range(len(RUNS))],
        "combined_files": CHAIN_FILES,
    }
    (OUT / "combined_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
