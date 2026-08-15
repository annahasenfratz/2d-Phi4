#!/usr/bin/env python3
"""Create a compact Markdown summary from histogram quantification CSVs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-csv", type=Path, required=True)
    p.add_argument("--output-md", type=Path, required=True)
    p.add_argument("--title", default="Kernel Observable Comparison")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with args.input_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {args.title}", ""]
    lines.append("| observable | standardized_mean_shift | std_ratio_a_over_b | total_variation | jensen_shannon | wasserstein_1 | ks_statistic | ks_pvalue |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        lines.append(
            "| {observable} | {standardized_mean_shift:.6g} | {std_ratio_a_over_b:.6g} | {total_variation:.6g} | {jensen_shannon:.6g} | {wasserstein_1:.6g} | {ks_statistic:.6g} | {ks_pvalue:.6g} |".format(
                observable=row["observable"],
                standardized_mean_shift=float(row["standardized_mean_shift"]),
                std_ratio_a_over_b=float(row["std_ratio_a_over_b"]),
                total_variation=float(row["total_variation"]),
                jensen_shannon=float(row["jensen_shannon"]),
                wasserstein_1=float(row["wasserstein_1"]),
                ks_statistic=float(row["ks_statistic"]),
                ks_pvalue=float(row["ks_pvalue"]),
            )
        )
    args.output_md.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
