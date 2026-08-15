#!/usr/bin/env python3
"""Heidelberg-style zero-sum 2x2 block-noise diagnostic."""

from __future__ import annotations

import argparse
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

from canonical_observable_audit import aggregate_observables, low_momentum_spectrum, write_csv  # type: ignore
from generated_coarse_backbone_ir_check import (  # type: ignore
    BLOCK_NORM,
    ETA_EXPONENT,
    KERNEL_META,
    block_sym,
    kernel_sum,
    load_kernel,
)


DATA = PROJECT / "outputs" / "paired_data_lam1_kappaf0p320"
OUT = PROJECT / "outputs" / "heidelberg_zero_sum_noise_test"
FINE = DATA / "fine_configs.npy"
COARSE = DATA / "coarse_blocked_configs.npy"
BACKBONE = DATA / "backbone_configs.npy"
KAPPA_F = 0.320
SEED = 20260625
SIGMAS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
LOW_MODES = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (0, 2)]


def action_components(phi: np.ndarray) -> dict[str, float]:
    nn = 0.5 * (
        (phi * np.roll(phi, -1, axis=-2)).mean(axis=(-2, -1))
        + (phi * np.roll(phi, -1, axis=-1)).mean(axis=(-2, -1))
    )
    phi2 = float(np.mean(phi**2))
    phi4 = float(np.mean(phi**4))
    hop = -4.0 * KAPPA_F * float(np.mean(nn))
    return {
        "action_hopping_density": hop,
        "action_phi2_density": -phi2,
        "action_phi4_density": phi4,
        "action_density": hop - phi2 + phi4,
    }


def obs_row(phi: np.ndarray, label: str, variant: str, sigma: float | str) -> dict[str, Any]:
    row = aggregate_observables(phi, label, 16)
    row.update(action_components(phi))
    row.update({"variant": variant, "sigma": sigma})
    return row


def repeat_coarse(coarse: np.ndarray) -> np.ndarray:
    return np.repeat(np.repeat(coarse, 2, axis=1), 2, axis=2)


def block_average_2x2(phi: np.ndarray) -> np.ndarray:
    return 0.25 * (
        phi[:, 0::2, 0::2]
        + phi[:, 1::2, 0::2]
        + phi[:, 0::2, 1::2]
        + phi[:, 1::2, 1::2]
    )


def repeat_block_average(phi: np.ndarray) -> np.ndarray:
    return repeat_coarse(block_average_2x2(phi))


def zero_sum_noise(rng: np.random.Generator, n: int, sigma: float) -> np.ndarray:
    eps = rng.normal(scale=sigma, size=(n, 8, 8, 2, 2))
    eps -= eps.mean(axis=(3, 4), keepdims=True)
    out = np.zeros((n, 16, 16), dtype=np.float64)
    out[:, 0::2, 0::2] = eps[:, :, :, 0, 0]
    out[:, 1::2, 0::2] = eps[:, :, :, 1, 0]
    out[:, 0::2, 1::2] = eps[:, :, :, 0, 1]
    out[:, 1::2, 1::2] = eps[:, :, :, 1, 1]
    return out


def block_average_residual_row(phi: np.ndarray, target_avg: np.ndarray, label: str, variant: str, sigma: float | str) -> dict[str, Any]:
    residual = block_average_2x2(phi) - target_avg
    return {
        "ensemble": label,
        "variant": variant,
        "sigma": sigma,
        "rms_simple_block_average_residual": float(np.sqrt(np.mean(residual**2))),
        "max_abs_simple_block_average_residual": float(np.max(np.abs(residual))),
        "mean_abs_simple_block_average_residual": float(np.mean(np.abs(residual))),
    }


def bsym_residual_row(phi: np.ndarray, coarse: np.ndarray, w: dict[str, float], label: str, variant: str, sigma: float | str) -> dict[str, Any]:
    residual = block_sym(phi, w) - coarse
    return {
        "ensemble": label,
        "variant": variant,
        "sigma": sigma,
        "rms_Bsym_residual": float(np.sqrt(np.mean(residual**2))),
        "max_abs_Bsym_residual": float(np.max(np.abs(residual))),
        "mean_abs_Bsym_residual": float(np.mean(np.abs(residual))),
        "relative_rms_Bsym_residual": float(np.sqrt(np.mean(residual**2)) / max(np.sqrt(np.mean(coarse**2)), 1.0e-30)),
    }


def shell_ids() -> np.ndarray:
    ky = np.fft.fftfreq(16) * 16
    kx = np.fft.fftfreq(16) * 16
    yy, xx = np.meshgrid(ky, kx, indexing="ij")
    return np.rint(xx**2 + yy**2).astype(int)


def shell_power_rows(phi: np.ndarray, label: str, variant: str, sigma: float | str) -> list[dict[str, Any]]:
    ft = np.fft.fft2(phi, axes=(-2, -1))
    shells = shell_ids()
    rows = []
    for sid in sorted(set(shells.ravel())):
        vals = ft[:, shells == sid]
        rows.append(
            {
                "ensemble": label,
                "variant": variant,
                "sigma": sigma,
                "shell_id": int(sid),
                "n_modes": int(np.sum(shells == sid)),
                "mean_power_per_mode": float(np.mean(np.abs(vals) ** 2) / (16 * 16)),
            }
        )
    return rows


def plot_scan_to_dir(rows: list[dict[str, Any]], y: str, filename: str, plots: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    for variant in ["coarse_repeat", "smooth_backbone", "backbone_blockavg_repeat"]:
        xs, ys = [], []
        for row in rows:
            if row["variant"] == variant:
                xs.append(float(row["sigma"]))
                ys.append(float(row[y]))
        if xs:
            order = np.argsort(xs)
            ax.plot(np.asarray(xs)[order], np.asarray(ys)[order], marker="o", label=variant)
    fine = next(row for row in rows if row["ensemble"] == "fine")
    ax.axhline(float(fine[y]), color="black", linestyle="--", linewidth=1.0, label="fine")
    ax.set_xlabel("sigma")
    ax.set_ylabel(y)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(plots / filename)
    plt.close(fig)


def plot_residual_to_dir(rows: list[dict[str, Any]], key: str, filename: str, plots: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    for variant in ["coarse_repeat", "smooth_backbone", "backbone_blockavg_repeat"]:
        xs, ys = [], []
        for row in rows:
            if row["variant"] == variant:
                xs.append(float(row["sigma"]))
                ys.append(float(row[key]))
        if xs:
            order = np.argsort(xs)
            ax.plot(np.asarray(xs)[order], np.asarray(ys)[order], marker="o", label=variant)
    ax.set_xlabel("sigma")
    ax.set_ylabel(key)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(plots / filename)
    plt.close(fig)


def plot_heatmaps_to_dir(examples: dict[str, np.ndarray], plots: Path) -> None:
    fig, axes = plt.subplots(1, len(examples), figsize=(3.1 * len(examples), 3.0), constrained_layout=True)
    if len(examples) == 1:
        axes = [axes]
    vmin = min(float(np.min(a[0])) for a in examples.values())
    vmax = max(float(np.max(a[0])) for a in examples.values())
    for ax, (name, arr) in zip(axes, examples.items()):
        im = ax.imshow(arr[0], cmap="coolwarm", vmin=vmin, vmax=vmax)
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes, fraction=0.025)
    fig.savefig(plots / "representative_heatmaps.pdf")
    fig.savefig(plots / "representative_heatmaps.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    out = args.output_dir
    plots = out / "plots"
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"Refusing to overwrite existing outputs in {out}")
    plots.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    w, kernel_meta = load_kernel()
    fine = np.load(FINE).astype(np.float64)
    coarse = np.load(COARSE).astype(np.float64)
    backbone = np.load(BACKBONE).astype(np.float64)

    bases = {
        "coarse_repeat": repeat_coarse(coarse),
        "smooth_backbone": backbone,
        "backbone_blockavg_repeat": repeat_block_average(backbone),
    }
    avg_targets = {
        "coarse_repeat": coarse,
        "smooth_backbone": block_average_2x2(backbone),
        "backbone_blockavg_repeat": block_average_2x2(backbone),
    }

    obs_rows: list[dict[str, Any]] = []
    avg_rows: list[dict[str, Any]] = []
    bsym_rows: list[dict[str, Any]] = []
    spec_rows: list[dict[str, Any]] = []
    shell_rows: list[dict[str, Any]] = []
    examples = {"fine": fine, "coarse_repeat_base": bases["coarse_repeat"], "backbone": backbone}

    references = {
        "fine": fine,
        "coarse_repeat_base": bases["coarse_repeat"],
        "backbone": backbone,
        "backbone_blockavg_repeat_base": bases["backbone_blockavg_repeat"],
    }
    for path, label in [
        (PROJECT / "outputs" / "inverse_blocking_proposal_benchmark_full" / "samples_sweeps_50.npy", "local_chunk_50"),
        (PROJECT / "outputs" / "inverse_blocking_proposal_benchmark_full" / "samples_sweeps_100.npy", "local_chunk_100"),
    ]:
        if path.exists():
            references[label] = np.load(path).astype(np.float64)

    for label, arr in references.items():
        obs_rows.append(obs_row(arr, label, "reference", "reference"))
        spec_rows.extend(low_momentum_spectrum(arr, label, 16))
        shell_rows.extend(shell_power_rows(arr, label, "reference", "reference"))
        if label != "fine":
            ref_coarse = coarse
            if label.startswith("local_chunk"):
                summary = json.loads((PROJECT / "outputs" / "inverse_blocking_proposal_benchmark_full" / "summary.json").read_text())
                ref_coarse = coarse[np.asarray(summary["selected_indices"], dtype=int)]
            bsym_rows.append(bsym_residual_row(arr, ref_coarse, w, label, "reference", "reference"))

    for variant, base in bases.items():
        target_avg = avg_targets[variant]
        for sigma in SIGMAS:
            phi = base + zero_sum_noise(rng, len(base), sigma)
            label = f"{variant}_sigma{sigma:g}"
            obs_rows.append(obs_row(phi, label, variant, sigma))
            avg_rows.append(block_average_residual_row(phi, target_avg, label, variant, sigma))
            bsym_rows.append(bsym_residual_row(phi, coarse, w, label, variant, sigma))
            spec_rows.extend(low_momentum_spectrum(phi, label, 16))
            shell_rows.extend(shell_power_rows(phi, label, variant, sigma))
            if variant == "coarse_repeat" and sigma in {0.25, 0.30}:
                examples[label] = phi
            if variant == "smooth_backbone" and sigma == 0.30:
                examples[label] = phi

    write_csv(out / "observable_scan.csv", obs_rows)
    write_csv(out / "block_average_residual_scan.csv", avg_rows)
    write_csv(out / "bsym_residual_scan.csv", bsym_rows)
    write_csv(out / "low_momentum_spectrum.csv", spec_rows)
    write_csv(out / "shell_power_scan.csv", shell_rows)

    for key, name in [
        ("phi2", "phi2_vs_sigma.pdf"),
        ("phi4", "phi4_vs_sigma.pdf"),
        ("nn2", "nn2_vs_sigma.pdf"),
        ("Binder_U4", "binder_vs_sigma.pdf"),
        ("xi_over_L", "xi_over_L_vs_sigma.pdf"),
        ("action_density", "action_density_vs_sigma.pdf"),
    ]:
        plot_scan_to_dir(obs_rows, key, name, plots)
    plot_residual_to_dir(avg_rows, "rms_simple_block_average_residual", "simple_block_average_residual_vs_sigma.pdf", plots)
    plot_residual_to_dir(bsym_rows, "rms_Bsym_residual", "Bsym_residual_vs_sigma.pdf", plots)
    plot_heatmaps_to_dir(examples, plots)

    fine_row = next(r for r in obs_rows if r["ensemble"] == "fine")
    candidates = [r for r in obs_rows if r["variant"] in bases]
    best = min(
        candidates,
        key=lambda r: abs(float(r["phi2"]) - float(fine_row["phi2"]))
        + abs(float(r["phi4"]) - float(fine_row["phi4"]))
        + abs(float(r["nn2"]) - float(fine_row["nn2"])),
    )
    best_bsym = next(r for r in bsym_rows if r["ensemble"] == best["ensemble"])
    best_avg = next(r for r in avg_rows if r["ensemble"] == best["ensemble"])
    local100 = next((r for r in obs_rows if r["ensemble"] == "local_chunk_100"), None)

    summary = {
        "canonical_inputs": {
            "fine": str(FINE.resolve()),
            "coarse": str(COARSE.resolve()),
            "backbone": str(BACKBONE.resolve()),
        },
        "kernel_metadata_path": str(KERNEL_META.resolve()),
        "kernel_original_source_path": kernel_meta.get("original_source_path"),
        "K_sum_check": kernel_sum(w),
        "eta_exponent": ETA_EXPONENT,
        "block_norm": BLOCK_NORM,
        "sigmas": SIGMAS,
        "seed": args.seed,
        "best_by_phi2_phi4_nn2": best,
        "best_simple_block_average_residual": best_avg,
        "best_Bsym_residual": best_bsym,
        "notes": [
            "Zero-sum noise preserves the simple 2x2 block average exactly relative to each variant's chosen block average.",
            "For smooth_backbone and backbone_blockavg_repeat variants, the simple average target is the 2x2 average of phi_back, not phi_c.",
            "The actual symmetric B_sym residual is measured separately and is not constrained by this construction.",
        ],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    def fmt(row: dict[str, Any]) -> str:
        return (
            f"| {row['ensemble']} | {row['variant']} | {row['sigma']} | {float(row['phi2']):.6g} | "
            f"{float(row['phi4']):.6g} | {float(row['nn2']):.6g} | {float(row['Binder_U4']):.6g} | "
            f"{float(row['xi_over_L']):.6g} | {float(row['action_density']):.6g} |"
        )

    table_rows = [
        fine_row,
        next(r for r in obs_rows if r["ensemble"] == "coarse_repeat_base"),
        next(r for r in obs_rows if r["ensemble"] == "backbone"),
        best,
    ]
    if local100 is not None:
        table_rows.append(local100)
    table = "\n".join(
        [
            "| ensemble | variant | sigma | phi2 | phi4 | nn2 | Binder U4 | xi/L | action density |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        + [fmt(r) for r in table_rows]
    )
    report = f"""# Heidelberg-Style Zero-Sum Block-Noise Diagnostic

## Setup

- fine configs: `{FINE.resolve()}`
- coarse configs: `{COARSE.resolve()}`
- backbone configs: `{BACKBONE.resolve()}`
- selected kernel metadata: `{KERNEL_META.resolve()}`
- K sum check: `{kernel_sum(w):.12g}`
- eta_exponent: `{ETA_EXPONENT}`
- block_norm: `{BLOCK_NORM:.12g}`

For each `2x2` block, Gaussian noise is drawn on four sites and its block mean is subtracted. This exactly preserves the simple `2x2` block average of the chosen base field.

## Main Table

{table}

Best row by simple `phi2/phi4/nn2` score:

- `{best['ensemble']}`
- simple block-average RMS residual: `{best_avg['rms_simple_block_average_residual']:.6g}`
- actual `B_sym` RMS residual: `{best_bsym['rms_Bsym_residual']:.6g}`

## Answers

1. Does zero-sum `2x2` Gaussian noise restore phi2, phi4, and nn2?

It can restore some UV/local power, but the scan should be judged against `observable_scan.csv`. The best simple local-moment row is `{best['ensemble']}` with phi2 `{float(best['phi2']):.6g}`, phi4 `{float(best['phi4']):.6g}`, and nn2 `{float(best['nn2']):.6g}` versus fine phi2 `{float(fine_row['phi2']):.6g}`, phi4 `{float(fine_row['phi4']):.6g}`, nn2 `{float(fine_row['nn2']):.6g}`.

2. Does it preserve Binder and xi/L?

It tends to preserve Binder and xi/L qualitatively because the block means and long-distance content of the base field dominate those observables, but the exact values depend on the base field. See `low_momentum_spectrum.csv`.

3. Does it preserve the simple block average exactly?

Yes. The simple block-average residuals are roundoff-small for every generated row by construction. See `block_average_residual_scan.csv`.

4. How badly does it violate the actual `B_sym` block map?

This construction does not enforce `B_sym`. The best local-moment row has `B_sym` RMS residual `{best_bsym['rms_Bsym_residual']:.6g}`. See `bsym_residual_scan.csv` for the full scan.

5. Which base field works best?

Use `observable_scan.csv` to compare variants. The chosen best-by-local-moment row is `{best['variant']}` at sigma `{best['sigma']}`. The repeated-coarse base starts with too much local amplitude in some operators, while the smooth backbone starts too smooth.

6. Is there a sigma range where local observables are close and `B_sym` residual is still acceptable?

There are sigma values that improve local observables, but `B_sym` residual is not automatically small. This is an initialization diagnostic, not an exact block-consistent sampler.

7. Is this promising for a later CNF/conditional flow?

As an initialization idea, yes: it is simple, local, and injects UV fluctuations without changing the simple block average. As a final sampler, no: a later CNF would still need to learn or correct the actual `B_sym` residual and non-Gaussian UV structure.

8. CNF training?

No CNF was trained in this task.
"""
    (out / "report.md").write_text(report)


if __name__ == "__main__":
    main()
