#!/usr/bin/env python3
"""Canonical comparison of native kappa=0.295 to induced blocked-fine coarse data."""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT / "outputs" / "matplotlib_config"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if str(PROJECT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT / "scripts"))

from canonical_observable_audit import (  # type: ignore
    BOOT_SEED,
    INPUTS,
    LOW_MODES,
    N_BOOT,
    OBSOLETE_BLOCKED,
    aggregate_observables,
    bootstrap_errors,
    file_info,
    low_momentum_spectrum,
    write_csv,
)
from generated_coarse_backbone_ir_check import (  # type: ignore
    BLOCK_NORM,
    ETA_EXPONENT,
    KERNEL_META,
    block_sym,
    kernel_sum,
    load_kernel,
    upscale_backbone,
)


OUT = PROJECT / "outputs" / "kappa0p295_canonical_comparison"
HISTS = OUT / "histograms"
NATIVE_0295 = (
    PROJECT
    / "outputs"
    / "coarse_distribution_calibration"
    / "generated_native_wolff"
    / "native_coarse_lam1_kappa0p295_L8_wolff.npy"
)
NATIVE_0295_SUMMARY = NATIVE_0295.with_name("native_coarse_lam1_kappa0p295_L8_wolff_summary.json")


def load_json_if_exists(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {"json_read_error": str(path)}


def row_by_label(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    for row in rows:
        if row["ensemble"] == label:
            return row
    raise KeyError(label)


def distance_rows(
    rows: list[dict[str, Any]], err_rows: list[dict[str, Any]], reference: str, candidate: str, comparison: str
) -> list[dict[str, Any]]:
    ref = row_by_label(rows, reference)
    cand = row_by_label(rows, candidate)
    ref_e = row_by_label(err_rows, reference)
    cand_e = row_by_label(err_rows, candidate)
    ops = ["m", "abs_m", "phi2", "phi4", "NN", "nn2", "diag", "2nn", "Binder_U4", "Binder_ratio_B4", "xi", "xi_over_L"]
    out = []
    for op in ops:
        diff = float(cand[op]) - float(ref[op])
        err = math.sqrt(float(ref_e.get(f"{op}_err", math.nan)) ** 2 + float(cand_e.get(f"{op}_err", math.nan)) ** 2)
        out.append(
            {
                "comparison": comparison,
                "reference": reference,
                "candidate": candidate,
                "observable": op,
                "reference_value": float(ref[op]),
                "candidate_value": float(cand[op]),
                "difference_candidate_minus_reference": diff,
                "combined_bootstrap_error": err,
                "z_score": diff / err if err > 0 and math.isfinite(err) else math.nan,
                "relative_difference": diff / float(ref[op]) if abs(float(ref[op])) > 1.0e-30 else math.nan,
            }
        )
    return out


def plot_histograms(arrays: dict[str, np.ndarray]) -> None:
    HISTS.mkdir(parents=True, exist_ok=True)
    for kind, values_fn, xlabel in [
        ("magnetization", lambda a: a.mean(axis=(-2, -1)), "m"),
        ("site_phi", lambda a: a.reshape(-1), "phi"),
    ]:
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        for label, arr in arrays.items():
            vals = values_fn(arr)
            ax.hist(vals, bins=60, density=True, histtype="step", linewidth=1.4, label=label)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("density")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(HISTS / f"{kind}_overlay.pdf")
        fig.savefig(HISTS / f"{kind}_overlay.png", dpi=180)
        plt.close(fig)


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    HISTS.mkdir(parents=True, exist_ok=True)

    w, kernel_meta = load_kernel()
    fine = np.load(INPUTS["fine_16x16"]).astype(np.float64)
    blocked_coarse = np.load(INPUTS["blocked_coarse_8x8"]).astype(np.float64)
    blocked_backbone = np.load(INPUTS["backbone_16x16"]).astype(np.float64)
    native_coarse = np.load(NATIVE_0295).astype(np.float64)
    native_backbone, transfer = upscale_backbone(native_coarse, w)
    reblocked_native = block_sym(native_backbone, w)
    native_block_err = reblocked_native - native_coarse

    arrays = {
        "fine_16x16": fine,
        "blocked_fine_coarse_8x8": blocked_coarse,
        "native_kappa0p295_coarse_8x8": native_coarse,
        "blocked_fine_backbone_16x16": blocked_backbone,
        "native_kappa0p295_backbone_16x16": native_backbone,
    }
    lattice_sizes = {
        "fine_16x16": 16,
        "blocked_fine_coarse_8x8": 8,
        "native_kappa0p295_coarse_8x8": 8,
        "blocked_fine_backbone_16x16": 16,
        "native_kappa0p295_backbone_16x16": 16,
    }

    obs_rows = [aggregate_observables(arr, label, lattice_sizes[label]) for label, arr in arrays.items()]
    rng = np.random.default_rng(BOOT_SEED)
    err_rows = [bootstrap_errors(arr, label, lattice_sizes[label], rng) for label, arr in arrays.items()]
    spec_rows: list[dict[str, Any]] = []
    for label, arr in arrays.items():
        spec_rows.extend(low_momentum_spectrum(arr, label, lattice_sizes[label]))

    write_csv(OUT / "canonical_comparison.csv", obs_rows)
    write_csv(OUT / "canonical_comparison_with_errors.csv", err_rows)
    write_csv(
        OUT / "coarse_level_distance.csv",
        distance_rows(
            obs_rows,
            err_rows,
            "blocked_fine_coarse_8x8",
            "native_kappa0p295_coarse_8x8",
            "native kappa=0.295 coarse vs induced blocked-fine coarse",
        ),
    )
    write_csv(
        OUT / "backbone_level_distance.csv",
        distance_rows(
            obs_rows,
            err_rows,
            "blocked_fine_backbone_16x16",
            "native_kappa0p295_backbone_16x16",
            "native kappa=0.295 backbone vs blocked-fine backbone",
        ),
    )
    write_csv(OUT / "low_momentum_spectrum.csv", spec_rows)
    plot_histograms(arrays)

    provenance = {
        "audit_created_utc": __import__("datetime").datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "script": str(Path(__file__).resolve()),
        "kernel_metadata_path": str(KERNEL_META.resolve()),
        "kernel_original_source_path": kernel_meta.get("original_source_path"),
        "kernel_caveat": kernel_meta.get("caveat"),
        "weights": w,
        "K_sum_check": kernel_sum(w),
        "eta_exponent": ETA_EXPONENT,
        "block_norm": BLOCK_NORM,
        "transfer": transfer,
        "native_backbone_reblock_check": {
            "max_abs_error": float(np.max(np.abs(native_block_err))),
            "rms_error": float(np.sqrt(np.mean(native_block_err**2))),
            "mean_abs_error": float(np.mean(np.abs(native_block_err))),
        },
        "inputs": {
            "fine_16x16": file_info(INPUTS["fine_16x16"]),
            "blocked_fine_coarse_8x8": file_info(INPUTS["blocked_coarse_8x8"]),
            "blocked_fine_backbone_16x16": file_info(INPUTS["backbone_16x16"]),
            "native_kappa0p295_coarse_8x8": file_info(NATIVE_0295),
            "native_kappa0p295_summary": load_json_if_exists(NATIVE_0295_SUMMARY),
        },
        "obsolete_file_warning": {
            "path": str(OBSOLETE_BLOCKED.resolve()),
            "status": "obsolete_for_current_comparisons",
            "file_info": file_info(OBSOLETE_BLOCKED),
            "instruction": "Do not use this old 64-config blocked_fine_coarse.npy for current comparisons.",
        },
        "observable_implementation": {
            "source": "canonical_observable_audit.aggregate_observables and bootstrap_errors",
            "bootstrap_replicates": N_BOOT,
            "bootstrap_seed": BOOT_SEED,
        },
    }
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")

    headers = ["ensemble", "L", "N", "m", "abs_m", "phi2", "phi4", "NN", "nn2", "diag", "2nn", "Binder_U4", "Binder_ratio_B4", "xi", "xi_over_L", "F_pmin"]
    table = "\n".join(
        ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] + ["---:"] * (len(headers) - 1)) + "|"]
        + ["| " + " | ".join(fmt(row[h]) for h in headers) + " |" for row in obs_rows]
    )

    def val(label: str, op: str) -> float:
        return float(row_by_label(obs_rows, label)[op])

    coarse_diff_xi = val("native_kappa0p295_coarse_8x8", "xi_over_L") - val("blocked_fine_coarse_8x8", "xi_over_L")
    coarse_diff_binder = val("native_kappa0p295_coarse_8x8", "Binder_U4") - val("blocked_fine_coarse_8x8", "Binder_U4")
    coarse_diff_phi2 = val("native_kappa0p295_coarse_8x8", "phi2") - val("blocked_fine_coarse_8x8", "phi2")
    coarse_diff_nn2 = val("native_kappa0p295_coarse_8x8", "nn2") - val("blocked_fine_coarse_8x8", "nn2")
    back_diff_xi = val("native_kappa0p295_backbone_16x16", "xi_over_L") - val("blocked_fine_backbone_16x16", "xi_over_L")
    back_diff_binder = val("native_kappa0p295_backbone_16x16", "Binder_U4") - val("blocked_fine_backbone_16x16", "Binder_U4")

    report = f"""# Canonical kappa=0.295 Native-Coarse Comparison

## Inputs

- fine target: `{INPUTS['fine_16x16'].resolve()}`
- induced blocked-fine coarse: `{INPUTS['blocked_coarse_8x8'].resolve()}`
- induced blocked-fine backbone: `{INPUTS['backbone_16x16'].resolve()}`
- native kappa=0.295 Wolff coarse: `{NATIVE_0295.resolve()}`

The obsolete 64-config file `{OBSOLETE_BLOCKED.resolve()}` is not used except in `provenance.json`.

## Native Backbone Reblocking Check

- max error: `{np.max(np.abs(native_block_err)):.6g}`
- RMS error: `{np.sqrt(np.mean(native_block_err**2)):.6g}`
- mean absolute error: `{np.mean(np.abs(native_block_err)):.6g}`

## Canonical Comparison Table

{table}

## Report Questions

1. Does kappa=0.295 native coarse match the induced blocked-fine coarse distribution?

It is close in Binder but not fully matched. Binder differs by `{coarse_diff_binder:.6g}`, while xi/L differs by `{coarse_diff_xi:.6g}`. Local coarse observables are also offset: phi2 differs by `{coarse_diff_phi2:.6g}` and nn2 differs by `{coarse_diff_nn2:.6g}`.

2. Does its upscaled backbone match the blocked-fine backbone?

The native backbone reblocks to its own native coarse field at roundoff, so the inverse algebra is stable. Distributionally, its Binder differs from the blocked-fine backbone by `{back_diff_binder:.6g}` and xi/L differs by `{back_diff_xi:.6g}`.

3. Which observables agree: Binder, local moments, NN/nn2, xi/L?

Binder is the best agreement at kappa=0.295. xi/L is high relative to the induced blocked-fine coarse/backbone. Local moments and link observables are not identical either; see `coarse_level_distance.csv` and `backbone_level_distance.csv` for z-scores using bootstrap errors.

4. Is xi/L the only significant mismatch, or are local coarse observables also off?

xi/L is not the only mismatch. The native coarse ensemble also has visible offsets in phi2, phi4, NN, nn2, diag, and 2nn relative to the induced blocked-fine coarse ensemble.

5. Is kappa=0.295 good enough as a native coarse source for a pilot, with caveats?

It is useful as a diagnostic native source because Binder is close and it was generated with the embedded Wolff sign-cluster update. It should not be treated as the solved induced coarse distribution: xi/L and local coarse observables remain mismatched.

6. Should future inverse tests use blocked-fine coarse fields, native kappa=0.295, or both separately?

Use blocked-fine coarse fields for conditional inverse-map development and quantitative conclusions. Use native kappa=0.295 separately as a native-coarse stress test with explicit caveats.
"""
    (OUT / "report.md").write_text(report)


if __name__ == "__main__":
    main()
