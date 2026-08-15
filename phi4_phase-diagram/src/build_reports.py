#!/usr/bin/env python3
"""Build organized phase-diagram reports from the dated phi4 scan outputs."""

from __future__ import annotations

import csv
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
DATA = DOCS / "data"
PLOTS = DOCS / "plots"


@dataclass(frozen=True)
class RunSpec:
    label: str
    lam: float
    folder: Path
    prefix: str
    note: str

    @property
    def slug(self) -> str:
        return self.label.replace(".", "p")

    @property
    def output_dir(self) -> Path:
        return self.folder / "outputs"

    @property
    def refined_csv(self) -> Path:
        return self.output_dir / f"{self.prefix}_cluster_l16_l24_l32_refined_curves.csv"

    @property
    def summary_json(self) -> Path:
        return self.output_dir / f"{self.prefix}_l16_l24_l32_chi_binder.json"

    @property
    def headline_plot(self) -> Path:
        return self.output_dir / f"{self.prefix}_l16_l24_l32_chi_binder.png"


RUNS = [
    RunSpec(
        label="1.0",
        lam=1.0,
        folder=ROOT / "2026-06-09-phase-diagram-lambda-1-2",
        prefix="phi4_lambda1",
        note="Cleanest finite-size scaling among the current scans; close to the Ising-limit expectation.",
    ),
    RunSpec(
        label="0.5",
        lam=0.5,
        folder=ROOT / "2026-06-09-phase-diagram-lambda-0p5-2",
        prefix="phi4_lambda05",
        note="Clean finite-size scaling; susceptibility peaks drift toward the Binder crossing region.",
    ),
    RunSpec(
        label="0.1",
        lam=0.1,
        folder=ROOT / "2026-06-09-phase-diagram-lambda-0p1-2",
        prefix="phi4_lambda01",
        note="Diagnostic rather than final: reweighting-center dependence is visible, especially at L=32.",
    ),
    RunSpec(
        label="0.01",
        lam=0.01,
        folder=ROOT / "2026-06-10-phase-diagram-lambda-0p01",
        prefix="phi4_lambda001",
        note="Weak-coupling diagnostic: Binder crossings are center-sensitive and scaling is not Ising-clean.",
    ),
]


def read_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def fnum(value: object, digits: int = 6) -> str:
    if value is None:
        return ""
    try:
        x = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(x):
        return "nan"
    if abs(x) >= 100:
        return f"{x:.3f}"
    if abs(x) >= 10:
        return f"{x:.4f}"
    return f"{x:.{digits}f}"


def markdown_table(headers: list[str], rows: Iterable[Iterable[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return "\n".join(lines)


def averaged_peak_rows(summary: dict) -> list[tuple[int, dict]]:
    peaks = summary["averaged_peaks"]
    return sorted((int(L), row) for L, row in peaks.items())


def all_crossings(summary: dict) -> list[tuple[float, str, dict]]:
    rows: list[tuple[float, str, dict]] = []
    for center, pairs in summary.get("binder_crossings", {}).items():
        match = re.search(r"[-+]?\d+(?:\.\d+)?", str(center))
        center_value = float(match.group(0)) if match else math.nan
        for pair, crossing in pairs.items():
            rows.append((center_value, pair, crossing))
    return sorted(rows, key=lambda row: (row[0], row[1]))


def linear_fit(xs: list[float], ys: list[float]) -> tuple[float, float]:
    n = len(xs)
    sx = sum(xs)
    sy = sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-15:
        return 0.0, sy / n
    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    return slope, intercept


def finite_size_kappa_estimate(summary: dict) -> dict[str, float]:
    """Estimate kappa_c by linear extrapolation of peak locations in 1/L.

    This is intentionally labeled as a finite-volume diagnostic.  Binder
    crossings remain the more reliable qualitative indicator in these scans.
    """

    rows = averaged_peak_rows(summary)
    xs = [1.0 / L for L, _ in rows]
    ys = [float(row["kappa_peak_mean"]) for _, row in rows]
    slope, intercept = linear_fit(xs, ys)
    residuals = [y - (slope * x + intercept) for x, y in zip(xs, ys)]
    rms = math.sqrt(sum(r * r for r in residuals) / len(residuals))
    return {"intercept": intercept, "slope": slope, "rms": rms}


def crossing_band(summary: dict, *, linear_only: bool = True) -> dict[str, float | int | None]:
    values = [
        float(crossing["kappa_crossing"])
        for _, _, crossing in all_crossings(summary)
        if "kappa_crossing" in crossing
        and (not linear_only or crossing.get("method") == "linear")
    ]
    if not values and linear_only:
        return crossing_band(summary, linear_only=False)
    if not values:
        return {"min": None, "max": None, "mean": None, "count": 0}
    return {
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
        "count": len(values),
    }


def copy_selected_columns(rows: list[dict[str, str]], path: Path) -> None:
    fields = [
        "lambda",
        "L",
        "kappa0",
        "kappa",
        "ess_over_n",
        "binder_u4",
        "susceptibility",
        "susceptibility_abs_centered",
        "abs_m_mean",
        "m2_mean",
        "m4_mean",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def rows_near_kappa(rows: list[dict[str, str]], kappa_c: float, radius: float = 0.003) -> list[dict[str, str]]:
    near = [
        row
        for row in rows
        if abs(float(row["kappa"]) - kappa_c) <= radius and float(row["ess_over_n"]) >= 0.3
    ]
    return sorted(near, key=lambda r: (int(r["L"]), float(r["kappa0"]), float(r["kappa"])))


def write_lambda_report(spec: RunSpec, summary: dict, rows: list[dict[str, str]]) -> dict[str, float | str]:
    band = crossing_band(summary)
    extrap = finite_size_kappa_estimate(summary)
    exponent = float(summary["height_power_fit"]["exponent"])
    data_path = DATA / f"lambda_{spec.slug}_refined_observables.csv"
    copy_selected_columns(rows, data_path)

    band_text = (
        f"{fnum(band['min'])}-{fnum(band['max'])}"
        if band["count"]
        else "not available"
    )
    preferred_kappa = float(band["mean"]) if band["mean"] is not None else extrap["intercept"]
    near_rows = rows_near_kappa(rows, preferred_kappa)

    peak_table = markdown_table(
        ["L", "kappa peak", "half-spread", "chi_abs peak"],
        [
            [
                L,
                fnum(row["kappa_peak_mean"]),
                fnum(row.get("kappa_peak_half_spread", "")),
                fnum(row["chi_peak_mean"]),
            ]
            for L, row in averaged_peak_rows(summary)
        ],
    )

    crossing_table = markdown_table(
        ["kappa0", "L pair", "kappa crossing", "method"],
        [
            [
                fnum(center),
                pair,
                fnum(crossing.get("kappa_crossing")),
                crossing.get("method", ""),
            ]
            for center, pair, crossing in all_crossings(summary)
        ],
    )

    near_table = markdown_table(
        [
            "L",
            "kappa0",
            "kappa",
            "ESS/N",
            "Binder U4",
            "chi",
            "chi_abs",
            "|m|",
        ],
        [
            [
                row["L"],
                fnum(row["kappa0"], 4),
                fnum(row["kappa"]),
                fnum(row["ess_over_n"], 4),
                fnum(row["binder_u4"]),
                fnum(row["susceptibility"]),
                fnum(row["susceptibility_abs_centered"]),
                fnum(row["abs_m_mean"]),
            ]
            for row in near_rows
        ],
    )

    full_table = markdown_table(
        [
            "L",
            "kappa0",
            "kappa",
            "ESS/N",
            "Binder U4",
            "chi",
            "chi_abs",
        ],
        [
            [
                row["L"],
                fnum(row["kappa0"], 4),
                fnum(row["kappa"]),
                fnum(row["ess_over_n"], 4),
                fnum(row["binder_u4"]),
                fnum(row["susceptibility"]),
                fnum(row["susceptibility_abs_centered"]),
            ]
            for row in sorted(rows, key=lambda r: (int(r["L"]), float(r["kappa0"]), float(r["kappa"])))
        ],
    )

    rel_plot = spec.headline_plot.relative_to(ROOT)
    rel_data = data_path.relative_to(ROOT)
    doc_data = data_path.relative_to(DOCS)
    text = f"""# Lambda = {spec.label}

Canonical run folder: `{spec.folder.name}`.

Headline plot: [`{rel_plot}`](../{rel_plot})

Machine-readable refined table: [`{rel_data}`]({doc_data})

## Interpretation

{spec.note}

Working critical-kappa diagnostics:

- Linear Binder crossing band: `{band_text}` from {band["count"]} pair/center entries.
- Mean linear Binder crossing: `{fnum(band["mean"])}`.
- Linear extrapolation of susceptibility peak locations versus `1/L`: `{fnum(extrap["intercept"])}` with RMS residual `{fnum(extrap["rms"])}`.
- Susceptibility peak-height fit: `chi_abs,max ~ L^p`, `p = {fnum(exponent)}`.

## Susceptibility Peaks

{peak_table}

## Binder Crossings

{crossing_table}

## Refined Observables Near The Crossing Band

This table keeps refined rows with `ESS/N >= 0.3` within `0.003` of the mean
Binder crossing.  The complete refined per-kappa table is included below and in
the CSV linked above.

{near_table}

## Complete Refined Per-Kappa Observables

{full_table}
"""
    out_path = DOCS / f"lambda_{spec.slug}.md"
    out_path.write_text(text)

    return {
        "lambda": spec.label,
        "kappa_binder_mean": fnum(band["mean"]),
        "kappa_binder_min": fnum(band["min"]),
        "kappa_binder_max": fnum(band["max"]),
        "kappa_peak_extrap": fnum(extrap["intercept"]),
        "peak_exponent": fnum(exponent),
        "report": out_path.name,
    }


def svg_polyline(points: list[tuple[float, float]], xscale, yscale, color: str) -> str:
    coords = " ".join(f"{xscale(x):.1f},{yscale(y):.1f}" for x, y in points)
    return f'<polyline fill="none" stroke="{color}" stroke-width="2.2" points="{coords}" />'


def svg_circles(points: list[tuple[float, float]], xscale, yscale, color: str) -> str:
    return "\n".join(
        f'<circle cx="{xscale(x):.1f}" cy="{yscale(y):.1f}" r="3.6" fill="{color}" />'
        for x, y in points
    )


def write_summary_svgs(run_data: list[tuple[RunSpec, dict]]) -> None:
    try:
        write_summary_plots_matplotlib(run_data)
        return
    except ImportError:
        pass

    width, height = 780, 440
    margin_l, margin_r, margin_t, margin_b = 70, 30, 34, 62
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    colors = ["#1b6ca8", "#c44536", "#2f8f46", "#7b4ab3"]

    lambdas = [spec.lam for spec, _ in run_data]
    log_lams = [math.log10(x) for x in lambdas]
    summaries = [summary for _, summary in run_data]
    bands = [crossing_band(summary) for summary in summaries]
    means = [float(band["mean"]) for band in bands]
    exponents = [float(summary["height_power_fit"]["exponent"]) for summary in summaries]

    def make_axis_svg(yvalues: list[float], title: str, ylabel: str, outfile: Path) -> None:
        xmin, xmax = min(log_lams), max(log_lams)
        ymin, ymax = min(yvalues), max(yvalues)
        pad = max((ymax - ymin) * 0.12, 0.002)
        ymin -= pad
        ymax += pad

        def xs(x: float) -> float:
            return margin_l + (x - xmin) / (xmax - xmin) * plot_w

        def ys(y: float) -> float:
            return margin_t + (ymax - y) / (ymax - ymin) * plot_h

        points = sorted(zip(log_lams, yvalues), key=lambda p: p[0])
        body = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="white" />',
            f'<text x="{width/2}" y="22" text-anchor="middle" font-family="Arial" font-size="16">{title}</text>',
            f'<line x1="{margin_l}" y1="{height-margin_b}" x2="{width-margin_r}" y2="{height-margin_b}" stroke="#222" />',
            f'<line x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{height-margin_b}" stroke="#222" />',
            f'<text x="{width/2}" y="{height-18}" text-anchor="middle" font-family="Arial" font-size="13">log10(lambda)</text>',
            f'<text x="18" y="{height/2}" text-anchor="middle" font-family="Arial" font-size="13" transform="rotate(-90 18 {height/2})">{ylabel}</text>',
        ]
        for tick in [-2, -1, 0]:
            x = xs(tick)
            body.append(f'<line x1="{x:.1f}" y1="{height-margin_b}" x2="{x:.1f}" y2="{height-margin_b+5}" stroke="#222" />')
            body.append(f'<text x="{x:.1f}" y="{height-margin_b+21}" text-anchor="middle" font-family="Arial" font-size="12">{tick}</text>')
        for i in range(5):
            yv = ymin + i * (ymax - ymin) / 4
            y = ys(yv)
            body.append(f'<line x1="{margin_l-5}" y1="{y:.1f}" x2="{margin_l}" y2="{y:.1f}" stroke="#222" />')
            body.append(f'<text x="{margin_l-9}" y="{y+4:.1f}" text-anchor="end" font-family="Arial" font-size="12">{fnum(yv, 3)}</text>')
        body.append(svg_polyline(points, xs, ys, "#333"))
        for idx, (spec, _) in enumerate(run_data):
            x = math.log10(spec.lam)
            y = yvalues[idx]
            color = colors[idx % len(colors)]
            body.append(svg_circles([(x, y)], xs, ys, color))
            body.append(f'<text x="{xs(x)+7:.1f}" y="{ys(y)-7:.1f}" font-family="Arial" font-size="12">lambda={spec.label}</text>')
        body.append("</svg>\n")
        outfile.write_text("\n".join(body))

    make_axis_svg(means, "Binder crossing estimate across lambda", "mean kappa crossing", PLOTS / "kappa_vs_lambda.svg")
    make_axis_svg(exponents, "Susceptibility peak-height exponent across lambda", "p in chi_abs,max ~ L^p", PLOTS / "peak_exponent_vs_lambda.svg")


def write_summary_plots_matplotlib(run_data: list[tuple[RunSpec, dict]]) -> None:
    mpl_config = ROOT / "work" / "mplconfig"
    mpl_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lambdas = [spec.lam for spec, _ in run_data]
    labels = [spec.label for spec, _ in run_data]
    log_lams = [math.log10(x) for x in lambdas]
    summaries = [summary for _, summary in run_data]
    bands = [crossing_band(summary) for summary in summaries]
    means = [float(band["mean"]) for band in bands]
    lows = [float(band["min"]) for band in bands]
    highs = [float(band["max"]) for band in bands]
    exponents = [float(summary["height_power_fit"]["exponent"]) for summary in summaries]

    order = sorted(range(len(log_lams)), key=lambda i: log_lams[i])

    def ordered(values: list[float]) -> list[float]:
        return [values[i] for i in order]

    x = ordered(log_lams)
    ordered_labels = [labels[i] for i in order]

    fig, ax = plt.subplots(figsize=(7.8, 4.4), constrained_layout=True)
    y = ordered(means)
    yerr = [
        [means[i] - lows[i] for i in order],
        [highs[i] - means[i] for i in order],
    ]
    ax.errorbar(x, y, yerr=yerr, marker="o", linewidth=1.8, capsize=4, color="#1b6ca8")
    for xi, yi, label in zip(x, y, ordered_labels):
        ax.annotate(f"lambda={label}", (xi, yi), xytext=(6, 5), textcoords="offset points", fontsize=9)
    ax.set_title("Binder crossing estimate across lambda")
    ax.set_xlabel("log10(lambda)")
    ax.set_ylabel("mean kappa crossing")
    ax.grid(True, alpha=0.25)
    fig.savefig(PLOTS / "kappa_vs_lambda.svg")
    fig.savefig(PLOTS / "kappa_vs_lambda.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.8, 4.4), constrained_layout=True)
    y = ordered(exponents)
    ax.plot(x, y, marker="o", linewidth=1.8, color="#c44536")
    ax.axhline(1.75, color="#333333", linestyle="--", linewidth=1.2, alpha=0.7, label="2D Ising 7/4")
    for xi, yi, label in zip(x, y, ordered_labels):
        ax.annotate(f"lambda={label}", (xi, yi), xytext=(6, 5), textcoords="offset points", fontsize=9)
    ax.set_title("Susceptibility peak-height exponent across lambda")
    ax.set_xlabel("log10(lambda)")
    ax.set_ylabel("p in chi_abs,max ~ L^p")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(PLOTS / "peak_exponent_vs_lambda.svg")
    fig.savefig(PLOTS / "peak_exponent_vs_lambda.png", dpi=180)
    plt.close(fig)


def write_index(summaries: list[dict[str, float | str]]) -> None:
    table = markdown_table(
        [
            "lambda",
            "linear Binder mean",
            "linear Binder band",
            "peak extrap.",
            "chi exponent p",
            "report",
        ],
        [
            [
                row["lambda"],
                row["kappa_binder_mean"],
                f"{row['kappa_binder_min']}-{row['kappa_binder_max']}",
                row["kappa_peak_extrap"],
                row["peak_exponent"],
                f"[`{row['report']}`]({row['report']})",
            ]
            for row in summaries
        ],
    )

    text = f"""# 2D phi4 kappa-lambda phase diagram notes

This directory collects the current finite-volume scans for the two-dimensional
phi4 model in the action convention

```text
S = sum_x [(1 - 2 lambda) phi_x^2 + lambda phi_x^4
           - 2 kappa sum_mu phi_x phi_{{x+mu}}]
```

The canonical data currently use `L = 16, 24, 32` and single-histogram
reweighting from the dated run folders.  The per-lambda reports include
susceptibilities, abs-centered susceptibilities, Binder cumulants, and ESS/N at
every refined kappa value.

## Current critical-kappa diagnostics

{table}

## Reading the table

- `linear Binder mean` and `linear Binder band` summarize stored Binder
  crossings where the two Binder curves actually bracket a crossing.  Per-lambda
  reports still list closest-grid/no-sign-change diagnostics separately.
- `peak extrap.` is a linear extrapolation of susceptibility peak positions in
  `1/L`; it is a finite-volume diagnostic, not a precision infinite-volume fit.
- `chi exponent p` comes from `chi_abs,max ~ L^p`.  The 2D Ising value is
  `gamma/nu = 7/4 = 1.75`; lambda `1.0` and `0.5` are close, while lambda `0.1`
  and `0.01` are not yet clean.

## Plots

- [`plots/kappa_vs_lambda.svg`](plots/kappa_vs_lambda.svg)
- [`plots/kappa_vs_lambda.png`](plots/kappa_vs_lambda.png)
- [`plots/peak_exponent_vs_lambda.svg`](plots/peak_exponent_vs_lambda.svg)
- [`plots/peak_exponent_vs_lambda.png`](plots/peak_exponent_vs_lambda.png)

## Data products

Machine-readable refined tables are in [`data/`](data/).  The original dated
run folders are left in place so the provenance of each plot and summary remains
visible.
"""
    (DOCS / "README.md").write_text(text)


def write_project_readme() -> None:
    text = """# phi4 phase diagram

Organized workspace for the two-dimensional lattice phi4 phase-structure scans
in the kappa-lambda plane.

- [`docs/`](docs/) contains the current summary reports, per-lambda observables,
  aggregate CSV files, and standard-library SVG plots.
- `src/build_reports.py` regenerates the organized reports from the dated run
  folders.
- The dated `2026-*` folders are retained as raw run provenance.
- `tests/` and `notebooks/` are placeholders for follow-up validation and
  exploration.

Install dependencies and regenerate reports with:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python src/build_reports.py
```
"""
    (ROOT / "README.md").write_text(text)


def main() -> None:
    DOCS.mkdir(exist_ok=True)
    DATA.mkdir(exist_ok=True)
    PLOTS.mkdir(exist_ok=True)

    summary_rows: list[dict[str, float | str]] = []
    run_data: list[tuple[RunSpec, dict]] = []
    for spec in RUNS:
        summary = read_json(spec.summary_json)
        rows = read_rows(spec.refined_csv)
        summary_rows.append(write_lambda_report(spec, summary, rows))
        run_data.append((spec, summary))

    write_summary_svgs(run_data)
    write_index(summary_rows)
    write_project_readme()


if __name__ == "__main__":
    main()
