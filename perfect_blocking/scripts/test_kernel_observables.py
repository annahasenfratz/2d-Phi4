#!/usr/bin/env python3
"""Compare direct coarse and blocked-fine observable CSV files."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.histogram_compare import metrics, plot_histogram
from scripts.common.scan_utils import load_config


DEFAULT_OBSERVABLES = [
    "phi2",
    "phi4",
    "local_kurtosis_ratio",
    "NN",
    "2nn",
    "diag",
    "action_density",
    "m",
    "m2",
    "m4",
    "Binder_U4_from_averages",
    "xi_over_L",
]

DEFAULT_GK_OBSERVABLES = ["G_00", "G_10", "G_01", "G_pmin_avg"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path)
    p.add_argument("--direct-csv", type=Path)
    p.add_argument("--blocked-csv", type=Path)
    p.add_argument("--output-csv", type=Path, required=True)
    p.add_argument("--plot-dir", type=Path, required=True)
    p.add_argument("--gk-direct-csv", type=Path)
    p.add_argument("--gk-blocked-csv", type=Path)
    p.add_argument("--gk-output-csv", type=Path)
    p.add_argument("--bins", type=int)
    p.add_argument("--observables", nargs="*")
    p.add_argument("--gk-observables", nargs="*")
    p.add_argument("--label-a", default="direct-generated L16")
    p.add_argument("--label-b", default="native L32 blocked to L16")
    p.add_argument("--plot-prefix", default="histogram_compare")
    return p.parse_args()


def read_columns(path: Path) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, value in row.items():
                if key in {"config_index", "source"} or value in {"", None}:
                    continue
                try:
                    out.setdefault(key, []).append(float(value))
                except ValueError:
                    pass
    return out


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config) if args.config else {}
    direct_csv = args.direct_csv or Path(cfg.get("direct_observable_csv", ""))
    blocked_csv = args.blocked_csv or Path(cfg.get("blocked_observable_csv", ""))
    if not direct_csv or not blocked_csv:
        raise SystemExit("pass --direct-csv and --blocked-csv, or set them in config")
    bins = int(args.bins or cfg.get("histogram_bins", 50))
    observables = args.observables or cfg.get("observables") or DEFAULT_OBSERVABLES

    direct = read_columns(direct_csv)
    blocked = read_columns(blocked_csv)
    rows: list[dict[str, object]] = []
    args.plot_dir.mkdir(parents=True, exist_ok=True)
    for obs in observables:
        if obs not in direct or obs not in blocked:
            continue
        row = metrics(direct[obs], blocked[obs], bins=bins)
        row = {"observable": obs, **row}
        rows.append(row)
        plot_histogram(
            direct[obs],
            blocked[obs],
            observable=obs,
            out_pdf=args.plot_dir / f"{args.plot_prefix}_{obs}.pdf",
            bins=bins,
            label_a=args.label_a,
            label_b=args.label_b,
        )

    if not rows:
        raise SystemExit("no overlapping numeric observable columns found")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with args.output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    gk_direct_csv = args.gk_direct_csv or (Path(cfg["direct_gk_summary_csv"]) if cfg.get("direct_gk_summary_csv") else None)
    gk_blocked_csv = args.gk_blocked_csv or (Path(cfg["blocked_gk_summary_csv"]) if cfg.get("blocked_gk_summary_csv") else None)
    gk_output_csv = args.gk_output_csv or (Path(cfg["gk_comparison_csv"]) if cfg.get("gk_comparison_csv") else None)
    if gk_direct_csv and gk_blocked_csv:
        if gk_output_csv is None:
            gk_output_csv = args.output_csv.with_name(args.output_csv.stem + "_Gk.csv")
        gk_observables = args.gk_observables or cfg.get("gk_comparison_observables") or DEFAULT_GK_OBSERVABLES
        gk_direct = read_columns(gk_direct_csv)
        gk_blocked = read_columns(gk_blocked_csv)
        gk_rows: list[dict[str, object]] = []
        for obs in gk_observables:
            if obs not in gk_direct or obs not in gk_blocked:
                continue
            row = metrics(gk_direct[obs], gk_blocked[obs], bins=bins)
            row = {"observable": obs, **row}
            gk_rows.append(row)
            plot_histogram(
                gk_direct[obs],
                gk_blocked[obs],
                observable=obs,
                out_pdf=args.plot_dir / f"{args.plot_prefix}_{obs}.pdf",
                bins=bins,
                label_a=args.label_a,
                label_b=args.label_b,
            )
        if not gk_rows:
            raise SystemExit("G(k) comparison requested, but no overlapping G(k) columns found")
        gk_output_csv.parent.mkdir(parents=True, exist_ok=True)
        keys = list(gk_rows[0].keys())
        with gk_output_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(gk_rows)


if __name__ == "__main__":
    main()
