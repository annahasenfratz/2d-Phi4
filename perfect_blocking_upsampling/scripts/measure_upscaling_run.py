#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def count_rows(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as fh:
        return max(0, sum(1 for _ in csv.reader(fh)) - 1)


def main() -> int:
    ap = argparse.ArgumentParser(description="Inventory observable files for an upscaling run.")
    ap.add_argument("--run-dir", type=Path, required=True)
    args = ap.parse_args()
    obs = args.run_dir / "observables"
    summary = {
        "all_observables_per_config_rows": count_rows(obs / "all_observables_per_config.csv"),
        "Gk_per_config_rows": count_rows(obs / "Gk_per_config.csv"),
        "acceptance_history_rows": count_rows(obs / "acceptance_history.csv"),
        "sweep_summary_rows": count_rows(obs / "sweep_summary.csv"),
    }
    out = args.run_dir / "summaries" / "observable_inventory.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

