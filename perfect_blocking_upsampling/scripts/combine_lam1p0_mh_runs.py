#!/usr/bin/env python3
"""Combine completed lambda=1.0 MH run blocks into one analyzable run directory."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
UPSAMPLING = PROJECT_ROOT / "perfect_blocking_upsampling"
sys.path.insert(0, str(UPSAMPLING / "scripts"))

from run_lam0p2_flow_detail_rethermalization import aggregate_history  # noqa: E402


CHAIN_FILES = [
    "per_sweep_observables.csv",
    "main_per_sweep_measurements.csv",
    "Gk_per_sweep_measurements.csv",
]


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_rows(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def selected_rows(rows: list[dict[str, str]], max_sweep: int | None) -> list[dict[str, str]]:
    if max_sweep is None:
        return rows
    return [row for row in rows if int(float(row["sweep"])) <= max_sweep]


def combine_chain_file(runs: list[Path], output: Path, name: str, max_sweep: int | None) -> list[dict[str, str]]:
    fields: list[str] | None = None
    combined: list[dict[str, str]] = []
    chain_offset = 0
    for run in runs:
        run_fields, rows = read_rows(run / "observables" / name)
        rows = selected_rows(rows, max_sweep)
        if fields is None:
            fields = run_fields
        elif fields != run_fields:
            raise ValueError(f"schema mismatch in {run / 'observables' / name}")
        chain_ids = [int(float(row["chain_id"])) for row in rows if row.get("chain_id", "")]
        for row in rows:
            item = dict(row)
            if item.get("chain_id", ""):
                item["chain_id"] = str(int(float(item["chain_id"])) + chain_offset)
            combined.append(item)
        chain_offset += max(chain_ids) + 1 if chain_ids else 0
    if fields is None:
        raise ValueError(f"no rows for {name}")
    write_rows(output / "observables" / name, fields, combined)
    return combined


def combine_acceptance(runs: list[Path], output: Path, max_sweep: int | None) -> None:
    fields: list[str] | None = None
    by_sweep: dict[int, list[dict[str, str]]] = {}
    for run in runs:
        run_fields, rows = read_rows(run / "observables" / "acceptance_history.csv")
        fields = run_fields if fields is None else fields
        for row in selected_rows(rows, max_sweep):
            by_sweep.setdefault(int(float(row["sweep"])), []).append(row)
    if fields is None:
        return
    sum_columns = {"coarse_proposals", "coarse_accepts", "detail_proposals", "detail_accepts", "conditional_flow_refreshes"}
    output_rows: list[dict[str, Any]] = []
    coarse_prop_cum = coarse_acc_cum = detail_prop_cum = detail_acc_cum = 0
    for sweep, rows in sorted(by_sweep.items()):
        item: dict[str, Any] = {"sweep": sweep, "update_mode": rows[0]["update_mode"]}
        for key in sum_columns:
            item[key] = sum(int(float(row.get(key, 0) or 0)) for row in rows)
        coarse_prop_cum += item["coarse_proposals"]
        coarse_acc_cum += item["coarse_accepts"]
        detail_prop_cum += item["detail_proposals"]
        detail_acc_cum += item["detail_accepts"]
        item["coarse_acceptance"] = item["coarse_accepts"] / item["coarse_proposals"] if item["coarse_proposals"] else float("nan")
        item["detail_acceptance"] = item["detail_accepts"] / item["detail_proposals"] if item["detail_proposals"] else float("nan")
        item["coarse_proposals_cumulative"] = coarse_prop_cum
        item["coarse_accepts_cumulative"] = coarse_acc_cum
        item["detail_proposals_cumulative"] = detail_prop_cum
        item["detail_accepts_cumulative"] = detail_acc_cum
        item["coarse_acceptance_cumulative"] = coarse_acc_cum / coarse_prop_cum if coarse_prop_cum else float("nan")
        item["detail_acceptance_cumulative"] = detail_acc_cum / detail_prop_cum if detail_prop_cum else float("nan")
        output_rows.append(item)
    write_rows(output / "observables" / "acceptance_history.csv", list(output_rows[0]), output_rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True, help="Combined run directory, relative to the repository root.")
    ap.add_argument("--run", type=Path, action="append", required=True, help="Source run directory, relative to the repository root.")
    ap.add_argument("--max-sweep", type=int, default=None, help="Keep rows through this common completed sweep.")
    ap.add_argument("--allow-partial", action="store_true", help="Allow source runs that are not marked completed.")
    args = ap.parse_args()
    output = PROJECT_ROOT / args.output
    runs = [PROJECT_ROOT / run for run in args.run]
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    for run in runs:
        status = json.loads((run / "status.json").read_text())
        if status.get("status") != "completed" and not args.allow_partial:
            raise RuntimeError(f"source run is not completed: {run} ({status.get('status')})")

    main_rows: list[dict[str, str]] | None = None
    run_rows: list[dict[str, str]] | None = None
    for name in CHAIN_FILES:
        rows = combine_chain_file(runs, output, name, args.max_sweep)
        if name == "main_per_sweep_measurements.csv":
            main_rows = rows
        elif name == "per_sweep_observables.csv":
            run_rows = rows
    assert main_rows is not None and run_rows is not None
    history = aggregate_history(main_rows, run_rows)
    history_fields = list(history[0]) if history else []
    write_rows(output / "observables" / "ensemble_average_history.csv", history_fields, history)
    combine_acceptance(runs, output, args.max_sweep)

    for name in ["native_L32_reference_summary.csv"]:
        source = runs[0] / "observables" / name
        if source.exists():
            shutil.copy2(source, output / "observables" / name)

    source_statuses = [json.loads((run / "status.json").read_text()) for run in runs]
    source_configs = [yaml.safe_load((run / "run_config.yaml").read_text()) for run in runs]
    config = dict(source_configs[0])
    config["run_dir"] = str(output)
    config["n_chains"] = sum(
        int(status.get("n_chains", source_cfg["n_chains"]))
        for status, source_cfg in zip(source_statuses, source_configs)
    )
    config["start_index"] = min(
        int(status.get("start_index", source_cfg.get("start_index", 0)))
        for status, source_cfg in zip(source_statuses, source_configs)
    )
    config["coarse_source_selection"] = "combined_completed_source_blocks"
    if args.max_sweep is not None:
        config["n_sweeps"] = args.max_sweep
    with (output / "run_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    manifest = {
        "status": "completed",
        "description": "Combined lambda=1.0 flow-initialized fixed-detail delayed-acceptance MH runs.",
        "source_runs": [str(run.relative_to(PROJECT_ROOT)) for run in runs],
        "total_chains": config["n_chains"],
        "sweeps": args.max_sweep if args.max_sweep is not None else int(source_statuses[0]["current_sweep"]),
        "source_statuses": [status.get("status") for status in source_statuses],
        "chain_id_ranges": [[i * 500, (i + 1) * 500 - 1] for i in range(len(runs))],
        "combined_files": CHAIN_FILES + ["ensemble_average_history.csv", "acceptance_history.csv"],
    }
    (output / "combined_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output / "status.json").write_text(json.dumps({"status": "completed", "current_sweep": manifest["sweeps"], "n_chains": manifest["total_chains"], "run_dir": str(output)}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
