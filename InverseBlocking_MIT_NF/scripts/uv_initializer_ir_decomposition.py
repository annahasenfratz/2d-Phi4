#!/usr/bin/env python3
"""IR and blocking decomposition for empirical UV library initializers."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from empirical_uv_library_initializer import block_average_2x2, block_sym, kernel_weights


PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "outputs/paired_data_lam1_kappaf0p320"
SRC = PROJECT / "outputs/uv_library_source_comparison"
OUT = SRC / "ir_uv_decomposition"
BENCH = PROJECT / "outputs/inverse_blocking_proposal_benchmark_full"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def ir_metrics(phi: np.ndarray) -> dict[str, float]:
    arr = np.asarray(phi, dtype=np.float64)
    n, ly, lx = arr.shape
    volume = ly * lx
    m = arr.mean(axis=(-2, -1))
    m2 = float(np.mean(m**2))
    m4 = float(np.mean(m**4))
    b4 = m4 / (m2 * m2) if m2 > 0 else math.nan
    binder = 1.0 - b4 / 3.0 if math.isfinite(b4) else math.nan
    ft = np.fft.fftn(arr, axes=(-2, -1))
    s0 = float(np.mean(np.abs(ft[:, 0, 0]) ** 2) / volume)
    sx = float(np.mean(np.abs(ft[:, 1, 0]) ** 2) / volume)
    sy = float(np.mean(np.abs(ft[:, 0, 1]) ** 2) / volume)
    spmin = 0.5 * (sx + sy)
    xi = math.nan
    if spmin > 0 and s0 / spmin > 1.0:
        xi = float(0.5 / np.sin(np.pi / lx) * np.sqrt(s0 / spmin - 1.0))
    return {
        "Binder_U4": float(binder),
        "Binder_ratio_B4": float(b4),
        "xi": float(xi),
        "xi_over_L": float(xi / lx) if math.isfinite(xi) else math.nan,
        "S0": s0,
        "S_pmin": float(spmin),
        "S_pmin_x": sx,
        "S_pmin_y": sy,
        "mean_m": float(np.mean(m)),
        "abs_m": float(np.mean(np.abs(m))),
    }


def fourier_cross_terms(back: np.ndarray, delta: np.ndarray) -> dict[str, float]:
    b = np.asarray(back, dtype=np.float64)
    d = np.asarray(delta, dtype=np.float64)
    volume = b.shape[-1] * b.shape[-2]
    fb = np.fft.fftn(b, axes=(-2, -1))
    fd = np.fft.fftn(d, axes=(-2, -1))

    def cross(ky: int, kx: int) -> float:
        return float(np.mean(2.0 * np.real(np.conj(fb[:, ky, kx]) * fd[:, ky, kx])) / volume)

    def power(ky: int, kx: int) -> float:
        return float(np.mean(np.abs(fd[:, ky, kx]) ** 2) / volume)

    return {
        "delta_S0": power(0, 0),
        "delta_S_pmin": 0.5 * (power(1, 0) + power(0, 1)),
        "delta_S_pmin_x": power(1, 0),
        "delta_S_pmin_y": power(0, 1),
        "cross_2Re_back_star_delta_p0": cross(0, 0),
        "cross_2Re_back_star_delta_pmin": 0.5 * (cross(1, 0) + cross(0, 1)),
        "cross_2Re_back_star_delta_pmin_x": cross(1, 0),
        "cross_2Re_back_star_delta_pmin_y": cross(0, 1),
    }


def simple_block_stats(delta: np.ndarray) -> dict[str, float]:
    d = np.asarray(delta, dtype=np.float64)
    n, lf, _ = d.shape
    block_sum = d.reshape(n, lf // 2, 2, lf // 2, 2).sum(axis=(2, 4))
    block_avg = block_sum / 4.0
    return {
        "simple_delta_block_sum_mean": float(np.mean(block_sum)),
        "simple_delta_block_sum_rms": float(np.sqrt(np.mean(block_sum * block_sum))),
        "simple_delta_block_sum_max_abs": float(np.max(np.abs(block_sum))),
        "simple_delta_block_avg_rms": float(np.sqrt(np.mean(block_avg * block_avg))),
        "simple_delta_block_avg_max_abs": float(np.max(np.abs(block_avg))),
    }


def load_aligned_reference() -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    back = np.load(DATA / "backbone_configs.npy").astype(np.float64)
    coarse = np.load(DATA / "coarse_blocked_configs.npy").astype(np.float64)
    summary = BENCH / "summary.json"
    if summary.exists():
        selected = np.asarray(json.loads(summary.read_text()).get("selected_indices", []), dtype=int)
        if len(selected):
            return back[selected], coarse[selected], selected, "benchmark_selected_indices"
    return back, coarse, np.arange(len(back)), "canonical_order"


def source_arrays() -> list[dict[str, Any]]:
    back = np.load(DATA / "backbone_configs.npy").astype(np.float64)
    coarse = np.load(DATA / "coarse_blocked_configs.npy").astype(np.float64)
    rows = []
    for path in sorted(SRC.glob("initializer_from_*.npy")):
        rows.append(
            {
                "ensemble": path.stem,
                "path": path,
                "after": np.load(path).astype(np.float64),
                "back": back,
                "coarse": coarse,
                "alignment": "canonical_order",
            }
        )
    prev = PROJECT / "outputs/empirical_uv_library_initializer/haar_conditional_fine_block_average.npy"
    if prev.exists():
        rows.append(
            {
                "ensemble": "previous_fine_library_oracle_initializer",
                "path": prev,
                "after": np.load(prev).astype(np.float64),
                "back": back,
                "coarse": coarse,
                "alignment": "canonical_order",
            }
        )
    zero = SRC / "zero_sum_gaussian_backbone_avg_sigma0p15.npy"
    if zero.exists():
        rows.append(
            {
                "ensemble": "zero_sum_gaussian_backbone_avg_sigma0p15",
                "path": zero,
                "after": np.load(zero).astype(np.float64),
                "back": back,
                "coarse": coarse,
                "alignment": "canonical_order",
            }
        )
    # This baseline was not saved by source-comparison; use empirical initializer output if present.
    zero_emp = PROJECT / "outputs/empirical_uv_library_initializer/zero_sum_gaussian_on_backbone_blockavg_sigma0p15.npy"
    if zero_emp.exists() and not zero.exists():
        rows.append(
            {
                "ensemble": "zero_sum_gaussian_on_backbone_blockavg_sigma0p15",
                "path": zero_emp,
                "after": np.load(zero_emp).astype(np.float64),
                "back": back,
                "coarse": coarse,
                "alignment": "canonical_order",
            }
        )
    local = SRC / "exact_null_local_chunk_100_sweeps_reference.npy"
    if not local.exists():
        local = BENCH / "samples_sweeps_100.npy"
    if local.exists():
        local_back, local_coarse, selected, alignment = load_aligned_reference()
        rows.append(
            {
                "ensemble": "exact_null_local_chunk_100_sweeps_reference",
                "path": local,
                "after": np.load(local).astype(np.float64),
                "back": local_back,
                "coarse": local_coarse,
                "alignment": alignment,
            }
        )
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    w, block_norm = kernel_weights()
    rows = source_arrays()

    before_after_rows: list[dict[str, Any]] = []
    uv_rows: list[dict[str, Any]] = []
    blocking_rows: list[dict[str, Any]] = []
    map_rows: list[dict[str, Any]] = []

    for item in rows:
        label = item["ensemble"]
        after = item["after"]
        back = item["back"]
        coarse = item["coarse"]
        n = min(len(after), len(back), len(coarse))
        after = after[:n]
        back = back[:n]
        coarse = coarse[:n]
        delta = after - back

        before_ir = ir_metrics(back)
        after_ir = ir_metrics(after)
        before_after_rows.append(
            {
                "ensemble": label,
                "alignment": item["alignment"],
                "stage": "before_uv_backbone",
                **before_ir,
            }
        )
        before_after_rows.append(
            {
                "ensemble": label,
                "alignment": item["alignment"],
                "stage": "after_uv_initializer",
                **after_ir,
                "delta_Binder_U4_after_minus_before": after_ir["Binder_U4"] - before_ir["Binder_U4"],
                "delta_xi_over_L_after_minus_before": after_ir["xi_over_L"] - before_ir["xi_over_L"],
                "delta_S0_after_minus_before": after_ir["S0"] - before_ir["S0"],
                "delta_S_pmin_after_minus_before": after_ir["S_pmin"] - before_ir["S_pmin"],
            }
        )

        b_delta = block_sym(delta, w, block_norm)
        uv_rows.append(
            {
                "ensemble": label,
                "alignment": item["alignment"],
                **simple_block_stats(delta),
                "Bsym_delta_rms": float(np.sqrt(np.mean(b_delta * b_delta))),
                "Bsym_delta_max_abs": float(np.max(np.abs(b_delta))),
                **fourier_cross_terms(back, delta),
            }
        )

        simple_before = block_average_2x2(back)
        simple_after = block_average_2x2(after)
        bsym_before = block_sym(back, w, block_norm)
        bsym_after = block_sym(after, w, block_norm)
        for map_name, before_map, after_map in [
            ("simple_2x2_block_average", simple_before, simple_after),
            ("Bsym_blocking_map", bsym_before, bsym_after),
        ]:
            delta_map = after_map - before_map
            map_rows.append(
                {
                    "ensemble": label,
                    "alignment": item["alignment"],
                    "blocking_map": map_name,
                    "before_phi2": float(np.mean(before_map**2)),
                    "after_phi2": float(np.mean(after_map**2)),
                    "coarse_target_phi2": float(np.mean(coarse**2)),
                    "before_minus_coarse_rms": float(np.sqrt(np.mean((before_map - coarse) ** 2))),
                    "after_minus_coarse_rms": float(np.sqrt(np.mean((after_map - coarse) ** 2))),
                    "after_minus_before_rms": float(np.sqrt(np.mean(delta_map * delta_map))),
                    "after_minus_before_max_abs": float(np.max(np.abs(delta_map))),
                    "after_minus_before_mean": float(np.mean(delta_map)),
                }
            )

        blocking_rows.append(
            {
                "ensemble": label,
                "alignment": item["alignment"],
                "simple_block_avg_delta_rms": float(np.sqrt(np.mean((simple_after - simple_before) ** 2))),
                "simple_block_avg_after_minus_coarse_rms": float(np.sqrt(np.mean((simple_after - coarse) ** 2))),
                "Bsym_delta_rms": float(np.sqrt(np.mean((bsym_after - bsym_before) ** 2))),
                "Bsym_after_minus_coarse_rms": float(np.sqrt(np.mean((bsym_after - coarse) ** 2))),
                "Bsym_before_minus_coarse_rms": float(np.sqrt(np.mean((bsym_before - coarse) ** 2))),
            }
        )

    write_csv(OUT / "ir_before_after.csv", before_after_rows)
    write_csv(OUT / "uv_field_diagnostics.csv", uv_rows)
    write_csv(OUT / "blocking_map_comparison.csv", map_rows)
    write_csv(OUT / "blocking_residual_summary.csv", blocking_rows)

    summary = {
        "output": str(OUT),
        "n_ensembles": len(rows),
        "definitions": {
            "S0": "mean |FFT(phi)[0,0]|^2 / V",
            "S_pmin": "0.5*(S(2pi/L,0)+S(0,2pi/L))",
            "cross_term": "2 Re[FFT(backbone)^* FFT(delta_UV)] / V, ensemble averaged",
            "simple_block_sums": "sum of delta_UV over each 2x2 block",
            "Bsym_delta": "B_sym(delta_UV), using selected symmetric blockavg kernel",
        },
    }
    write_json(OUT / "summary.json", summary)

    def md_table(data: list[dict[str, Any]], cols: list[str]) -> str:
        lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
        for row in data:
            vals = []
            for col in cols:
                val = row.get(col, "")
                vals.append(f"{val:.6g}" if isinstance(val, float) else str(val))
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    after_rows = [r for r in before_after_rows if r["stage"] == "after_uv_initializer"]
    compact = []
    for r in after_rows:
        uv = next(u for u in uv_rows if u["ensemble"] == r["ensemble"])
        br = next(b for b in blocking_rows if b["ensemble"] == r["ensemble"])
        compact.append(
            {
                "ensemble": r["ensemble"],
                "Binder_U4": r["Binder_U4"],
                "xi_over_L": r["xi_over_L"],
                "S0": r["S0"],
                "S_pmin": r["S_pmin"],
                "delta_S_pmin": r["delta_S_pmin_after_minus_before"],
                "UV_S_pmin": uv["delta_S_pmin"],
                "cross_pmin": uv["cross_2Re_back_star_delta_pmin"],
                "simple_delta_rms": br["simple_block_avg_delta_rms"],
                "Bsym_delta_rms": br["Bsym_delta_rms"],
            }
        )
    report = f"""# IR / UV Decomposition For UV Initializers

This diagnostic decomposes each initializer as:

`phi_after = phi_back + delta_UV`

and compares the IR observables and blocking-map effects before and after adding UV details.

## After-UV Summary

{md_table(compact, ["ensemble", "Binder_U4", "xi_over_L", "S0", "S_pmin", "delta_S_pmin", "UV_S_pmin", "cross_pmin", "simple_delta_rms", "Bsym_delta_rms"])}

## Interpretation

- Haar initializers preserve the simple 2x2 block average of `phi_back`, so their simple block-average delta is near zero, but `B_sym(delta_UV)` is not zero.
- The exact-null 100-sweep reference is aligned to its benchmark selected coarse indices; its `B_sym_delta_rms` is roundoff-small because the constrained correction stays in the block-null space.
- `S(0)` is mostly unchanged because the UV constructions have negligible zero-mode delta.
- `S(pmin)` changes through the explicit UV power and the cross term `2 Re[phi_back^* delta_UV]`; see `uv_field_diagnostics.csv` for both components.
- The same comparison is saved for both simple 2x2 block averaging and the actual `B_sym` blocking map in `blocking_map_comparison.csv`.

## Output Files

- `ir_before_after.csv`
- `uv_field_diagnostics.csv`
- `blocking_map_comparison.csv`
- `blocking_residual_summary.csv`
- `summary.json`
"""
    (OUT / "report.md").write_text(report)
    print(json.dumps({"output": str(OUT), "n_ensembles": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
