#!/usr/bin/env python3
"""Split a paired native_/upscaled_ observable table into two matching CSVs."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--direct-output", required=True, type=Path)
    parser.add_argument("--comparison-output", required=True, type=Path)
    parser.add_argument("--direct-prefix", default="native_")
    parser.add_argument("--comparison-prefix", default="upscaled_")
    args = parser.parse_args()
    with args.input.open(newline="", encoding="utf-8") as handle:
        source = list(csv.DictReader(handle))
    if not source:
        raise ValueError(f"empty input CSV: {args.input}")

    def extract(prefix: str) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        for row in source:
            out = {"source_config_index": row["source_config_index"], "label": row["label"]}
            out.update({key[len(prefix):]: value for key, value in row.items() if key.startswith(prefix)})
            rows.append(out)
        return rows

    direct, comparison = extract(args.direct_prefix), extract(args.comparison_prefix)
    write_rows(args.direct_output, direct)
    write_rows(args.comparison_output, comparison)
    print(f"Wrote {args.direct_output}")
    print(f"Wrote {args.comparison_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
