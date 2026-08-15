#!/usr/bin/env python3
"""Summarize phi4 ensembles with both L=16 and L=32 scans."""

from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
DATA = DOCS / "data"
OUT_CSV = DOCS / "critical_ensembles.csv"
OUT_MD = DOCS / "critical_ensembles.md"


@dataclass(frozen=True)
class Source:
    summary_path: Path
    curve_path: Path | None
    kind: str


@dataclass(frozen=True)
class SampleSource:
    metadata_paths: tuple[Path, ...]
    kind: str = "raw_cluster_samples"


def fnum(value: object, digits: int = 6) -> str:
    if value is None:
        return ""
    try:
        x = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(x):
        return ""
    return f"{x:.{digits}f}"


def rel(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path.relative_to(ROOT.parent))


def read_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def read_csv(path: Path | None) -> list[dict[str, str]]:
    if path is None or not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def lambda_from_summary(summary: dict) -> float | None:
    args = summary.get("args", {})
    for key in ("lam", "lambda"):
        if key in args:
            return float(args[key])
    for row in summary.get("summaries", []):
        if "lambda" in row:
            return float(row["lambda"])
    return None


def lambda_from_path(path: Path) -> float | None:
    match = re.search(r"lambda[-_]?([0-9]+(?:p[0-9]+)?)", str(path))
    if not match:
        return None
    token = match.group(1).replace("p", ".")
    if token in {"001", "01", "05"}:
        token = {"001": "0.01", "01": "0.1", "05": "0.5"}[token]
    return float(token)


def curve_path_for(summary_path: Path, summary: dict) -> Path | None:
    inputs = summary.get("inputs", {})
    if isinstance(inputs, dict):
        refined = inputs.get("refined")
        if refined:
            path = summary_path.parent.parent / refined
            if path.exists():
                return path
            path = summary_path.parent / Path(refined).name
            if path.exists():
                return path

    output_csv = summary.get("args", {}).get("output_csv")
    if output_csv:
        path = ROOT.parent / output_csv
        if path.exists():
            return path
        path = summary_path.parent / Path(output_csv).name
        if path.exists():
            return path

    name = summary_path.name
    candidates: list[Path] = []
    if name.endswith("_chi_binder.json"):
        candidates.append(summary_path.with_name(name.replace("_chi_binder.json", "_cluster_l16_l24_l32_refined_curves.csv")))
    if name.endswith(".json"):
        candidates.append(summary_path.with_suffix(".csv"))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def discover_sources() -> list[Source]:
    sources: list[Source] = []
    for path in sorted(ROOT.rglob("*.json")):
        if "__pycache__" in path.parts:
            continue
        summary = read_json(path)
        if any(path.parent.glob("*_chi_binder.json")) and not path.name.endswith("_chi_binder.json"):
            continue
        if isinstance(summary.get("averaged_peaks"), dict):
            sizes = {int(L) for L in summary["averaged_peaks"]}
        elif isinstance(summary.get("summaries"), list):
            sizes = {int(row["L"]) for row in summary["summaries"] if "L" in row}
        else:
            continue
        if not {16, 32}.issubset(sizes):
            continue
        kind = "chi_binder" if "averaged_peaks" in summary else "scan_summary"
        sources.append(Source(path, curve_path_for(path, summary), kind))
    return sources


def discover_sample_sources() -> list[SampleSource]:
    sample_root = ROOT.parent / "perfect_blocking_multilevel" / "results" / "samples"
    if not sample_root.exists():
        return []

    grouped: dict[float, list[Path]] = defaultdict(list)
    for path in sorted(sample_root.glob("phi4_cluster_L*_lam*_n*_seed*.json")):
        try:
            meta = read_json(path)
            lam = float(meta["lambda"])
            L = int(meta["lattice_size"])
        except (KeyError, TypeError, ValueError):
            continue
        if L in {16, 32}:
            grouped[lam].append(path)

    sources: list[SampleSource] = []
    for paths in grouped.values():
        sizes = {int(read_json(path)["lattice_size"]) for path in paths}
        if {16, 32}.issubset(sizes):
            sources.append(SampleSource(tuple(paths)))
    return sources


def all_crossings(summary: dict) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for center, pairs in summary.get("binder_crossings", {}).items():
        match = re.search(r"[-+]?\d+(?:\.\d+)?", str(center))
        center_value = float(match.group(0)) if match else math.nan
        for pair, crossing in pairs.items():
            if "kappa_crossing" not in crossing:
                continue
            rows.append(
                {
                    "kappa0": center_value,
                    "pair": pair,
                    "kappa_crossing": float(crossing["kappa_crossing"]),
                    "method": crossing.get("method", ""),
                }
            )
    return rows


def binder_kappa_cr(summary: dict) -> tuple[float | None, str, str, int]:
    linear = [row for row in all_crossings(summary) if row["method"] == "linear"]
    rows = linear or all_crossings(summary)
    if not rows:
        return None, "", "", 0
    values = [float(row["kappa_crossing"]) for row in rows]
    return sum(values) / len(values), min(values), max(values), len(values)


def fit_peak_extrapolation(points: list[tuple[int, float]]) -> float | None:
    if len(points) < 2:
        return None
    xs = [1.0 / L for L, _ in points]
    ys = [kappa for _, kappa in points]
    n = len(points)
    sx = sum(xs)
    sy = sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-15:
        return None
    return (sy - ((n * sxy - sx * sy) / denom) * sx) / n


def peak_by_l(summary: dict) -> dict[int, tuple[float, float | None]]:
    if "averaged_peaks" in summary:
        return {
            int(L): (float(row["kappa_peak_mean"]), float(row.get("chi_peak_mean", "nan")))
            for L, row in summary["averaged_peaks"].items()
        }
    grouped: dict[int, list[tuple[float, float | None]]] = defaultdict(list)
    for row in summary.get("summaries", []):
        if "chi_abs_peak_kappa" not in row:
            continue
        grouped[int(row["L"])].append(
            (float(row["chi_abs_peak_kappa"]), float(row.get("chi_abs_peak_value", "nan")))
        )
    return {
        L: (
            sum(k for k, _ in rows) / len(rows),
            sum(v for _, v in rows if v is not None and not math.isnan(v)) / len(rows),
        )
        for L, rows in grouped.items()
    }


def interpolate(xs: list[float], ys: list[float], x: float) -> float | None:
    if not xs:
        return None
    ordered = sorted(zip(xs, ys))
    if x < ordered[0][0] or x > ordered[-1][0]:
        return None
    for (x0, y0), (x1, y1) in zip(ordered, ordered[1:]):
        if x0 <= x <= x1:
            if abs(x1 - x0) < 1e-15:
                return y0
            t = (x - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    if abs(x - ordered[-1][0]) < 1e-15:
        return ordered[-1][1]
    return None


def binder_at_kappa(rows: list[dict[str, str]], kappa: float | None) -> dict[int, float | None]:
    if kappa is None:
        return {}
    grouped: dict[tuple[int, float], list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        if "binder_u4" not in row:
            continue
        try:
            key = (int(row["L"]), float(row["kappa0"]))
            grouped[key].append((float(row["kappa"]), float(row["binder_u4"])))
        except (KeyError, ValueError):
            continue

    by_l: dict[int, list[float]] = defaultdict(list)
    for (L, _), points in grouped.items():
        value = interpolate([p[0] for p in points], [p[1] for p in points], kappa)
        if value is not None:
            by_l[L].append(value)
    return {L: sum(values) / len(values) for L, values in by_l.items()}


def diagnostics_path(meta_path: Path, meta: dict) -> Path:
    diag = meta.get("diagnostics_file")
    if diag:
        path = Path(str(diag).replace("/Users/anna/Work/Normalizing-flow/Inverse_RG", str(ROOT.parent)))
        if path.exists():
            return path
    return meta_path.with_name(meta_path.stem + "_diagnostics.npz")


def sample_observables(meta_path: Path) -> dict[str, float | int | str]:
    import numpy as np

    meta = read_json(meta_path)
    diag = np.load(diagnostics_path(meta_path, meta))
    m = diag["magnetization"].astype(float)
    L = int(meta["lattice_size"])
    volume = L * L
    m_mean = float(np.mean(m))
    m2 = float(np.mean(m * m))
    m4 = float(np.mean(m**4))
    abs_m = np.abs(m)
    return {
        "lambda": float(meta["lambda"]),
        "L": L,
        "kappa": float(meta["kappa"]),
        "binder_u4": 1.0 - m4 / (3.0 * m2 * m2),
        "susceptibility": volume * (m2 - m_mean * m_mean),
        "susceptibility_abs_centered": volume * (float(np.mean(abs_m * abs_m)) - float(np.mean(abs_m)) ** 2),
        "source": rel(meta_path),
    }


def direct_l16_l32_crossing(rows: list[dict[str, str]]) -> float | None:
    grouped: dict[float, dict[int, list[tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        try:
            grouped[float(row["kappa0"])][int(row["L"])].append((float(row["kappa"]), float(row["binder_u4"])))
        except (KeyError, ValueError):
            continue

    crossings: list[float] = []
    for by_l in grouped.values():
        if 16 not in by_l or 32 not in by_l:
            continue
        l16 = dict(by_l[16])
        l32 = dict(by_l[32])
        common = sorted(set(l16) & set(l32))
        diffs = [(k, l16[k] - l32[k]) for k in common]
        for (k0, d0), (k1, d1) in zip(diffs, diffs[1:]):
            if d0 == 0:
                crossings.append(k0)
            elif d0 * d1 < 0:
                crossings.append(k0 - d0 * (k1 - k0) / (d1 - d0))
    if crossings:
        return sum(crossings) / len(crossings)
    return None


def row_for_source(source: Source) -> dict[str, str]:
    summary = read_json(source.summary_path)
    curves = read_csv(source.curve_path)
    lam = lambda_from_summary(summary)
    if lam is None and curves:
        for key in ("lambda", "lambda4"):
            if key in curves[0]:
                lam = float(curves[0][key])
                break
    if lam is None:
        lam = lambda_from_path(source.summary_path)
    peaks = peak_by_l(summary)

    kappa_cr, band_min, band_max, crossing_count = binder_kappa_cr(summary)
    method = "linear_binder_mean"
    if kappa_cr is None:
        kappa_cr = direct_l16_l32_crossing(curves)
        method = "direct_L16_L32_binder_crossing" if kappa_cr is not None else "peak_1_over_L_extrapolation"
    if kappa_cr is None:
        kappa_cr = fit_peak_extrapolation(sorted((L, p[0]) for L, p in peaks.items()))

    binders = binder_at_kappa(curves, kappa_cr)
    return {
        "lambda": fnum(lam),
        "kappa_cr": fnum(kappa_cr),
        "kappa_cr_method": method,
        "binder_crossing_min": fnum(band_min),
        "binder_crossing_max": fnum(band_max),
        "binder_crossing_count": str(crossing_count),
        "binder_u4_L16_at_kappa_cr": fnum(binders.get(16)),
        "binder_u4_L24_at_kappa_cr": fnum(binders.get(24)),
        "binder_u4_L32_at_kappa_cr": fnum(binders.get(32)),
        "chi_abs_peak_kappa_L16": fnum(peaks.get(16, ("", ""))[0]),
        "chi_abs_peak_kappa_L24": fnum(peaks.get(24, ("", ""))[0]),
        "chi_abs_peak_kappa_L32": fnum(peaks.get(32, ("", ""))[0]),
        "chi_abs_peak_value_L16": fnum(peaks.get(16, ("", ""))[1]),
        "chi_abs_peak_value_L24": fnum(peaks.get(24, ("", ""))[1]),
        "chi_abs_peak_value_L32": fnum(peaks.get(32, ("", ""))[1]),
        "source_kind": source.kind,
        "source_summary": rel(source.summary_path),
        "source_curves": rel(source.curve_path),
    }


def row_for_sample_source(source: SampleSource) -> dict[str, str]:
    obs = [sample_observables(path) for path in source.metadata_paths]
    lam = float(obs[0]["lambda"])
    by_l: dict[int, list[dict[str, float | int | str]]] = defaultdict(list)
    for row in obs:
        by_l[int(row["L"])].append(row)

    peaks: dict[int, dict[str, float | int | str]] = {}
    for L, rows in by_l.items():
        peaks[L] = max(rows, key=lambda row: float(row["susceptibility_abs_centered"]))

    kappa_cr = fit_peak_extrapolation(
        sorted((L, float(row["kappa"])) for L, row in peaks.items() if L in {16, 32})
    )
    if kappa_cr is None and 32 in peaks:
        kappa_cr = float(peaks[32]["kappa"])

    source_list = ";".join(str(path.relative_to(ROOT.parent)) for path in source.metadata_paths)
    return {
        "lambda": fnum(lam),
        "kappa_cr": fnum(kappa_cr),
        "kappa_cr_method": "chi_abs_peak_1_over_L_extrapolation",
        "binder_crossing_min": "",
        "binder_crossing_max": "",
        "binder_crossing_count": "0",
        "binder_u4_L16_at_kappa_cr": "",
        "binder_u4_L24_at_kappa_cr": "",
        "binder_u4_L32_at_kappa_cr": "",
        "chi_abs_peak_kappa_L16": fnum(peaks.get(16, {}).get("kappa")),
        "chi_abs_peak_kappa_L24": fnum(peaks.get(24, {}).get("kappa")),
        "chi_abs_peak_kappa_L32": fnum(peaks.get(32, {}).get("kappa")),
        "chi_abs_peak_value_L16": fnum(peaks.get(16, {}).get("susceptibility_abs_centered")),
        "chi_abs_peak_value_L24": fnum(peaks.get(24, {}).get("susceptibility_abs_centered")),
        "chi_abs_peak_value_L32": fnum(peaks.get(32, {}).get("susceptibility_abs_centered")),
        "source_kind": source.kind,
        "source_summary": source_list,
        "source_curves": "",
    }


def markdown_table(headers: list[str], rows: Iterable[Iterable[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def write_outputs(rows: list[dict[str, str]]) -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    fields = [
        "lambda",
        "kappa_cr",
        "kappa_cr_method",
        "binder_crossing_min",
        "binder_crossing_max",
        "binder_crossing_count",
        "binder_u4_L16_at_kappa_cr",
        "binder_u4_L24_at_kappa_cr",
        "binder_u4_L32_at_kappa_cr",
        "chi_abs_peak_kappa_L16",
        "chi_abs_peak_kappa_L24",
        "chi_abs_peak_kappa_L32",
        "chi_abs_peak_value_L16",
        "chi_abs_peak_value_L24",
        "chi_abs_peak_value_L32",
        "source_kind",
        "source_summary",
        "source_curves",
    ]
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    table = markdown_table(
        [
            "lambda",
            "kappa_cr",
            "method",
            "Binder U4 L16",
            "Binder U4 L24",
            "Binder U4 L32",
            "chi peak kappa L16",
            "chi peak kappa L24",
            "chi peak kappa L32",
            "source",
        ],
        [
            [
                row["lambda"],
                row["kappa_cr"],
                row["kappa_cr_method"],
                row["binder_u4_L16_at_kappa_cr"],
                row["binder_u4_L24_at_kappa_cr"],
                row["binder_u4_L32_at_kappa_cr"],
                row["chi_abs_peak_kappa_L16"],
                row["chi_abs_peak_kappa_L24"],
                row["chi_abs_peak_kappa_L32"],
                source_link(row["source_summary"]),
            ]
            for row in rows
        ],
    )
    OUT_MD.write_text(
        "# Critical Ensemble Summary\n\n"
        "This file summarizes phi4 scan outputs that contain at least both `L=16` "
        "and `L=32` runs. The machine-readable version is "
        "[`critical_ensembles.csv`](critical_ensembles.csv).\n\n"
        "`kappa_cr` uses the mean of stored linear Binder crossings when those "
        "exist. If a summary has no stored crossings, the script attempts a "
        "direct `L=16`/`L=32` Binder crossing from the curve CSV, then falls back "
        "to a linear `1/L` extrapolation of the abs-centered susceptibility peak "
        "locations. For raw cluster sample sets, the same peak extrapolation is "
        "used when no Binder curves are available. Binder cumulants at `kappa_cr` "
        "are linear interpolations from available refined curves and averaged over "
        "reweighting centers for each volume.\n\n"
        f"{table}\n\n"
        "Regenerate with:\n\n"
        "```bash\n"
        "../.venv/bin/python -B phi4_phase-diagram/src/summarize_critical_ensembles.py\n"
        "```\n"
    )


def source_link(source_summary: str) -> str:
    parts = [part for part in source_summary.split(";") if part]
    if not parts:
        return ""
    if len(parts) == 1:
        return f"[`{parts[0]}`](../{parts[0]})"
    return f"{len(parts)} raw sample metadata files"


def main() -> None:
    rows = [row_for_source(source) for source in discover_sources()]
    rows.extend(row_for_sample_source(source) for source in discover_sample_sources())
    rows.sort(key=lambda row: (float(row["lambda"]), row["source_summary"]))
    write_outputs(rows)
    print(f"wrote {OUT_CSV.relative_to(ROOT)}")
    print(f"wrote {OUT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
