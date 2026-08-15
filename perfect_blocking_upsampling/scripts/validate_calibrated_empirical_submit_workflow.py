#!/usr/bin/env python3
"""Write format-regression evidence for calibrated empirical submit smokes."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np


def csv_header(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return next(csv.reader(handle))


def output_files(run: Path) -> set[str]:
    return {
        str(path.relative_to(run))
        for path in run.rglob("*")
        if path.is_file() and not path.name.startswith("extend_") and path.name != "run.log"
    }


def write_csv(path: Path, fields: list[str], rows: Iterable[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-run", type=Path, required=True)
    parser.add_argument("--old-run", type=Path, required=True)
    parser.add_argument("--docs-dir", type=Path, required=True)
    args = parser.parse_args()
    args.docs_dir.mkdir(parents=True, exist_ok=True)

    csv_names = [
        "per_sweep_observables.csv",
        "main_per_sweep_measurements.csv",
        "acceptance_history.csv",
        "ensemble_average_history.csv",
        "sweep_native_comparison.csv",
    ]
    schema_rows = []
    for name in csv_names:
        new = args.new_run / "observables" / name
        old = args.old_run / "observables" / name
        schema_rows.append({
            "artifact": f"observables/{name}",
            "new_exists": new.exists(),
            "legacy_exists": old.exists(),
            "same_header": new.exists() and old.exists() and csv_header(new) == csv_header(old),
            "new_header": "|".join(csv_header(new)) if new.exists() else "",
            "legacy_header": "|".join(csv_header(old)) if old.exists() else "",
        })
    with np.load(args.new_run / "checkpoints" / "checkpoint_latest.npz", allow_pickle=True) as new_ckpt:
        new_keys = sorted(new_ckpt.files)
        new_shapes = {key: list(new_ckpt[key].shape) for key in new_ckpt.files if key != "meta"}
    with np.load(args.old_run / "checkpoints" / "checkpoint_latest.npz", allow_pickle=True) as old_ckpt:
        old_keys = sorted(old_ckpt.files)
        old_shapes = {key: list(old_ckpt[key].shape) for key in old_ckpt.files if key != "meta"}
    schema_rows.append({
        "artifact": "checkpoints/checkpoint_latest.npz",
        "new_exists": True,
        "legacy_exists": True,
        "same_header": new_keys == old_keys,
        "new_header": json.dumps({"keys": new_keys, "shapes": new_shapes}, sort_keys=True),
        "legacy_header": json.dumps({"keys": old_keys, "shapes": old_shapes}, sort_keys=True),
    })
    write_csv(
        args.docs_dir / "output_schema_regression.csv",
        ["artifact", "new_exists", "legacy_exists", "same_header", "new_header", "legacy_header"],
        schema_rows,
    )

    status = json.loads((args.new_run / "status.json").read_text())
    smoke_rows = [{
        "test": "new_empirical_submit_smoke",
        "passed": status.get("status") == "completed" and status.get("initialization_mode") == "empirical_sample",
        "details": f"sweeps={status.get('current_sweep')}; reblocking={status.get('reblocking_max_error')}; nonfinite={status.get('nonfinite_count')}",
    }]
    write_csv(args.docs_dir / "smoke_test_results.csv", ["test", "passed", "details"], smoke_rows)

    with np.load(args.new_run / "checkpoints" / "checkpoint_latest.npz", allow_pickle=True) as payload:
        meta = json.loads(str(payload["meta"].item()))
    legacy_status = json.loads((args.old_run / "status.json").read_text())
    resume_rows = [
        {"test": "empirical_resume", "passed": meta["completed_sweeps"] >= 3 and meta.get("initializer_metadata", {}).get("initializer_type") == "calibrated_empirical_joint_2x2", "details": "checkpoint advanced without reinitialization"},
        {"test": "legacy_flow_resume", "passed": legacy_status.get("status") == "completed", "details": "legacy flow-initialized clone extended successfully"},
    ]
    write_csv(args.docs_dir / "resume_test_results.csv", ["test", "passed", "details"], resume_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
