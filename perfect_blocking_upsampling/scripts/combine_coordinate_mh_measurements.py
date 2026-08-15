#!/usr/bin/env python3
"""Combine completed coordinate-MH measurement CSVs without changing schemas."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


MEASUREMENT_FILES = (
    "per_sweep_observables.csv",
    "main_per_sweep_measurements.csv",
    "Gk_per_sweep_measurements.csv",
)


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or ()), list(reader)


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-sweep", type=int, default=None, help="Keep measurements through this saved sweep, inclusive.")
    parser.add_argument(
        "--dedupe-source-config-index",
        action="store_true",
        help="Keep the first run's chain for each source_config_index and drop overlapping chains from later runs.",
    )
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args()

    runs = [path.resolve() for path in args.runs]
    output = args.output_dir.resolve()
    observables = output / "observables"
    observables.mkdir(parents=True, exist_ok=True)

    selected_chains: dict[Path, set[int]] = {}
    dropped_duplicates: dict[Path, list[int]] = {}
    seen_source_indices: set[int] = set()
    for run in runs:
        fields, rows = read_csv(run / "observables" / "per_sweep_observables.csv")
        if "chain_id" not in fields or "source_config_index" not in fields or "sweep" not in fields:
            raise RuntimeError(f"missing chain_id/source_config_index/sweep in {run}")
        by_chain: dict[int, int] = {}
        for row in rows:
            if int(row["sweep"]) != 0:
                continue
            chain = int(row["chain_id"])
            source = int(row["source_config_index"])
            if chain in by_chain and by_chain[chain] != source:
                raise RuntimeError(f"inconsistent source_config_index for chain {chain} in {run}")
            by_chain[chain] = source
        if not by_chain:
            raise RuntimeError(f"no sweep-zero chains found in {run}")
        keep: set[int] = set()
        dropped: list[int] = []
        for chain, source in sorted(by_chain.items()):
            if args.dedupe_source_config_index and source in seen_source_indices:
                dropped.append(chain)
                continue
            keep.add(chain)
            seen_source_indices.add(source)
        selected_chains[run] = keep
        dropped_duplicates[run] = dropped

    offsets: dict[Path, int] = {}
    next_offset = 0
    for run in runs:
        fields, rows = read_csv(run / "observables" / "per_sweep_observables.csv")
        if "chain_id" not in fields:
            raise RuntimeError(f"missing chain_id in {run}")
        chain_ids = sorted({int(row["chain_id"]) for row in rows if int(row["chain_id"]) in selected_chains[run]})
        if chain_ids != sorted(selected_chains[run]):
            raise RuntimeError(f"noncontiguous chain IDs in {run}: {chain_ids[:5]}...")
        if args.max_sweep is not None:
            available = {int(row["sweep"]) for row in rows}
            if args.max_sweep not in available:
                raise RuntimeError(f"requested sweep {args.max_sweep} is not saved in {run}")
        offsets[run] = next_offset
        next_offset += len(chain_ids)

    for filename in MEASUREMENT_FILES:
        expected_fields: list[str] | None = None
        combined: list[dict[str, str]] = []
        for run in runs:
            fields, rows = read_csv(run / "observables" / filename)
            if expected_fields is None:
                expected_fields = fields
            elif fields != expected_fields:
                raise RuntimeError(f"schema mismatch in {run / 'observables' / filename}")
            for row in rows:
                if int(row["chain_id"]) not in selected_chains[run]:
                    continue
                if args.max_sweep is not None and int(row["sweep"]) > args.max_sweep:
                    continue
                row = row.copy()
                row["chain_id"] = str(int(row["chain_id"]) + offsets[run])
                combined.append(row)
        assert expected_fields is not None
        combined.sort(key=lambda row: (int(row["sweep"]), int(row["chain_id"])))
        write_csv(observables / filename, expected_fields, combined)

    inventory: list[dict[str, object]] = []
    for run in runs:
        _, rows = read_csv(run / "source_index_inventory.csv")
        for row in rows:
            if row.get("chain_id", "") and int(row["chain_id"]) in selected_chains[run]:
                row = row.copy()
                row["chain_id"] = str(int(row["chain_id"]) + offsets[run])
                inventory.append({"source_run": str(run), **row})
    inventory_fields = ["source_run", "role", "chain_id", "source_index"]
    write_csv(output / "source_index_inventory.csv", inventory_fields, inventory)  # type: ignore[arg-type]

    manifest = {
        "combined_runs": [str(run) for run in runs],
        "chain_id_offsets": {str(run): offsets[run] for run in runs},
        "combined_chain_count": next_offset,
        "measurement_files": list(MEASUREMENT_FILES),
        "chain_id_policy": "Each input run's chain IDs are offset to be unique; measurement schemas are otherwise unchanged.",
        "max_sweep": args.max_sweep,
        "dedupe_source_config_index": args.dedupe_source_config_index,
        "retained_chain_counts": {str(run): len(selected_chains[run]) for run in runs},
        "dropped_duplicate_chain_ids": {str(run): dropped_duplicates[run] for run in runs},
    }
    tmp = output / "combined_manifest.json.tmp"
    tmp.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    tmp.replace(output / "combined_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
