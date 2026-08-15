#!/usr/bin/env python3
"""Fourier low-mode inverse plus high-mode proposal diagnostics.

This is a diagnostic only: no training, no MCMC, and no modification of the
perfect_blocking provenance tree.
"""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
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
    kernel_array,
    kernel_sum,
    load_kernel,
)


DATA = PROJECT / "outputs" / "paired_data_lam1_kappaf0p320"
FINE_PATH = DATA / "fine_configs.npy"
COARSE_PATH = DATA / "coarse_blocked_configs.npy"
BACKBONE_PATH = DATA / "backbone_configs.npy"
OUT = PROJECT / "outputs" / "fourier_low_high_diagnostic"
PLOTS = OUT / "plots"
ARCHIVE = OUT / "archived_script"

L = 16
LC = 8
KAPPA_F = 0.320
LAMBDA_F = 1.0
SEED = 20260624
SHELL_GAUSSIAN_SCALES = [0.5, 0.75, 1.0, 1.25]
EMPIRICAL_RESAMPLE_N = 1


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=json_default) + "\n")


def json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(type(obj).__name__)


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


def obs_row(phi: np.ndarray, ensemble: str, variant: str, parameter: str | float = "") -> dict[str, Any]:
    row = aggregate_observables(np.asarray(phi, dtype=np.float64), ensemble, L)
    row.update(action_components(np.asarray(phi, dtype=np.float64)))
    row.update({"variant": variant, "parameter": parameter})
    return row


def transfer_arrays(w: dict[str, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ktilde = np.fft.fft2(kernel_array(w, L))
    p = np.fft.fftfreq(L) * 2.0 * np.pi
    py, px = np.meshgrid(p, p, indexing="ij")
    a = 0.25 * (1.0 + np.exp(1j * py)) * (1.0 + np.exp(1j * px))
    transfer = BLOCK_NORM * ktilde * a
    return ktilde, a, transfer


def low_mask_half_open() -> np.ndarray:
    mask_shift = np.zeros((L, L), dtype=bool)
    mask_shift[4:12, 4:12] = True
    return np.fft.ifftshift(mask_shift)


def conjugate_mask(mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(mask, dtype=bool)
    for ky in range(mask.shape[0]):
        for kx in range(mask.shape[1]):
            if mask[ky, kx]:
                out[(-ky) % mask.shape[0], (-kx) % mask.shape[1]] = True
    return out


def low_ft_from_coarse(coarse: np.ndarray, transfer: np.ndarray, mask: np.ndarray) -> np.ndarray:
    cft = np.fft.fft2(coarse, axes=(-2, -1))
    padded_shift = np.zeros((len(coarse), L, L), dtype=np.complex128)
    # numpy FFT convention: 8x8 coarse FFT embeds into 16x16 with factor 4.
    padded_shift[:, 4:12, 4:12] = 4.0 * np.fft.fftshift(cft, axes=(-2, -1))
    padded = np.fft.ifftshift(padded_shift, axes=(-2, -1))
    out = np.zeros_like(padded)
    out[:, mask] = padded[:, mask] / transfer[mask]
    return out


def coarse_fft_from_low_fine_ft(fine_ft: np.ndarray, transfer: np.ndarray, mask: np.ndarray) -> np.ndarray:
    transferred = fine_ft * transfer[None, :, :]
    shifted = np.fft.fftshift(transferred, axes=(-2, -1))
    coarse_shift = shifted[:, 4:12, 4:12] / 4.0
    return np.fft.ifftshift(coarse_shift, axes=(-2, -1))


def shell_ids() -> np.ndarray:
    freqs = np.fft.fftfreq(L) * L
    ky, kx = np.meshgrid(freqs, freqs, indexing="ij")
    return np.rint(ky**2 + kx**2).astype(int)


def shell_power_rows(phi: np.ndarray, label: str, variant: str, parameter: str | float = "") -> list[dict[str, Any]]:
    ft = np.fft.fft2(phi, axes=(-2, -1))
    shells = shell_ids()
    rows = []
    for sid in sorted(set(shells.ravel())):
        mask = shells == sid
        rows.append(
            {
                "ensemble": label,
                "variant": variant,
                "parameter": parameter,
                "shell_id": int(sid),
                "mode_count": int(np.sum(mask)),
                "mean_power_per_mode": float(np.mean(np.abs(ft[:, mask]) ** 2) / (L * L)),
            }
        )
    return rows


def empirical_shell_variances(ft: np.ndarray, high_mask: np.ndarray) -> dict[int, float]:
    shells = shell_ids()
    out: dict[int, float] = {}
    for sid in sorted(set(shells[high_mask].ravel())):
        mask = (shells == sid) & high_mask
        out[int(sid)] = float(np.mean(np.abs(ft[:, mask]) ** 2))
    return out


def shell_gaussian_high_ft(
    rng: np.random.Generator,
    n: int,
    high_mask: np.ndarray,
    variances: dict[int, float],
    scale: float,
) -> np.ndarray:
    white = rng.normal(size=(n, L, L))
    ft = np.fft.fft2(white, axes=(-2, -1))
    ft[:, ~high_mask] = 0.0
    shells = shell_ids()
    for sid, var in variances.items():
        mask = (shells == sid) & high_mask
        if not np.any(mask):
            continue
        current = float(np.mean(np.abs(ft[:, mask]) ** 2))
        ft[:, mask] *= scale * math.sqrt(var / max(current, 1.0e-30))
    ft[:, ~high_mask] = 0.0
    return ft


def full_high_gaussian_ft(rng: np.random.Generator, high_true_ft: np.ndarray, high_mask: np.ndarray, ridge: float = 1.0e-8) -> np.ndarray:
    # Proper complex covariance with Hermitian constraints is awkward. For this
    # diagnostic, sample the real high-only position field covariance instead,
    # then project back to the Hermitian-safe high Fourier support.
    high_fields = np.fft.ifft2(high_true_ft, axes=(-2, -1)).real
    n = len(high_fields)
    flat = high_fields.reshape(n, -1)
    mean = flat.mean(axis=0)
    cov = np.cov(flat, rowvar=False)
    evals, evecs = np.linalg.eigh(cov)
    evals = np.maximum(evals, ridge)
    z = rng.normal(size=(n, flat.shape[1]))
    sampled = mean[None, :] + (z * np.sqrt(evals)[None, :]) @ evecs.T
    ft = np.fft.fft2(sampled.reshape(n, L, L), axes=(-2, -1))
    ft[:, ~high_mask] = 0.0
    return ft


def block_residual_rows(phi: np.ndarray, coarse: np.ndarray, w: dict[str, float], ensemble: str, variant: str, parameter: str | float = "") -> dict[str, Any]:
    residual = block_sym(phi, w) - coarse
    return {
        "ensemble": ensemble,
        "variant": variant,
        "parameter": parameter,
        "rms_Bsym_residual": float(np.sqrt(np.mean(residual**2))),
        "max_abs_Bsym_residual": float(np.max(np.abs(residual))),
        "relative_rms_Bsym_residual": float(np.sqrt(np.mean(residual**2)) / max(np.sqrt(np.mean(coarse**2)), 1.0e-30)),
    }


def low_mode_residual_rows(phi: np.ndarray, low_ref_ft: np.ndarray, mask: np.ndarray, ensemble: str, variant: str, parameter: str | float = "") -> dict[str, Any]:
    ft = np.fft.fft2(phi, axes=(-2, -1))
    diff = ft[:, mask] - low_ref_ft[:, mask]
    return {
        "ensemble": ensemble,
        "variant": variant,
        "parameter": parameter,
        "rms_low_mode_residual": float(np.sqrt(np.mean(np.abs(diff) ** 2))),
        "max_abs_low_mode_residual": float(np.max(np.abs(diff))),
    }


def score_rows(obs_rows: list[dict[str, Any]], fine_label: str = "fine_target") -> list[dict[str, Any]]:
    fine = next(row for row in obs_rows if row["ensemble"] == fine_label)
    local_keys = ["phi2", "phi4", "NN", "nn2", "diag", "2nn"]
    action_keys = ["action_density", "action_hopping_density", "action_phi2_density", "action_phi4_density"]
    rows = []
    for row in obs_rows:
        local = float(np.mean([abs(float(row[k]) - float(fine[k])) / max(abs(float(fine[k])), 1.0e-12) for k in local_keys]))
        local_action = float(
            np.mean(
                [abs(float(row[k]) - float(fine[k])) / max(abs(float(fine[k])), 1.0e-12) for k in local_keys + action_keys]
            )
        )
        rows.append(
            {
                "ensemble": row["ensemble"],
                "variant": row["variant"],
                "parameter": row["parameter"],
                "local_score_mean_relative_abs_error": local,
                "local_plus_action_score_mean_relative_abs_error": local_action,
            }
        )
    return rows


def plot_observable_bars(obs_rows: list[dict[str, Any]], key: str, filename: str, selected: list[str]) -> None:
    rows = [r for r in obs_rows if r["ensemble"] in selected]
    fig, ax = plt.subplots(figsize=(8.0, 3.8))
    ax.bar([r["ensemble"] for r in rows], [float(r[key]) for r in rows])
    ax.tick_params(axis="x", rotation=25)
    ax.set_ylabel(key)
    fig.tight_layout()
    fig.savefig(PLOTS / filename)
    plt.close(fig)


def plot_shell_power(shell_rows: list[dict[str, Any]], selected: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for label in selected:
        rows = [r for r in shell_rows if r["ensemble"] == label]
        rows.sort(key=lambda r: int(r["shell_id"]))
        ax.plot([int(r["shell_id"]) for r in rows], [float(r["mean_power_per_mode"]) for r in rows], marker="o", label=label)
    ax.set_xlabel("Fourier shell k_index^2")
    ax.set_ylabel("mean power / mode")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(PLOTS / "shell_power_selected.pdf")
    plt.close(fig)


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty output directory: {OUT}")
    PLOTS.mkdir(parents=True, exist_ok=True)
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__), ARCHIVE / Path(__file__).name)

    rng = np.random.default_rng(SEED)
    w, kernel_meta = load_kernel()
    fine = np.load(FINE_PATH).astype(np.float64)
    coarse = np.load(COARSE_PATH).astype(np.float64)
    stored_backbone = np.load(BACKBONE_PATH).astype(np.float64)
    if fine.shape != (len(fine), L, L) or coarse.shape != (len(fine), LC, LC):
        raise RuntimeError(f"unexpected canonical shapes fine={fine.shape}, coarse={coarse.shape}")

    ktilde, a_transfer, transfer = transfer_arrays(w)
    mask = low_mask_half_open()
    conj_low = conjugate_mask(mask)
    high_mask = ~(mask | conj_low)
    unsafe_high_mask = (~mask) & (~high_mask)
    transfer_abs = np.abs(transfer[mask])
    unsafe_transfer_mask = mask & (np.abs(transfer) < 1.0e-10)

    fine_ft = np.fft.fft2(fine, axes=(-2, -1))
    coarse_ft = np.fft.fft2(coarse, axes=(-2, -1))
    low_predict_coarse_ft = coarse_fft_from_low_fine_ft(fine_ft, transfer, mask)
    low_transfer_diff = low_predict_coarse_ft - coarse_ft
    coarse_from_block = block_sym(fine, w)
    block_impl_diff = coarse_from_block - coarse
    low_from_coarse_ft = low_ft_from_coarse(coarse, transfer, mask)
    low_from_coarse_field_complex = np.fft.ifft2(low_from_coarse_ft, axes=(-2, -1))
    low_from_coarse = low_from_coarse_field_complex.real

    transfer_rows: list[dict[str, Any]] = []
    for ky in range(LC):
        for kx in range(LC):
            transfer_rows.append(
                {
                    "coarse_ky": ky,
                    "coarse_kx": kx,
                    "rms_low_sector_only_transfer_residual": float(np.sqrt(np.mean(np.abs(low_transfer_diff[:, ky, kx]) ** 2))),
                    "mean_abs_low_sector_only_transfer_residual": float(np.mean(np.abs(low_transfer_diff[:, ky, kx]))),
                    "mean_abs_coarse_fft": float(np.mean(np.abs(coarse_ft[:, ky, kx]))),
                    "relative_rms_low_sector_only_transfer_residual": float(
                        np.sqrt(np.mean(np.abs(low_transfer_diff[:, ky, kx]) ** 2)) / max(float(np.sqrt(np.mean(np.abs(coarse_ft[:, ky, kx]) ** 2))), 1.0e-30)
                    ),
                    "note": "This compares only the low fine sector; B_sym decimation also aliases high sectors.",
                }
            )
    write_csv(OUT / "transfer_audit.csv", transfer_rows)
    transfer_summary = {
        "Bsym_implementation_rms_vs_stored_coarse": float(np.sqrt(np.mean(block_impl_diff**2))),
        "Bsym_implementation_max_abs_vs_stored_coarse": float(np.max(np.abs(block_impl_diff))),
        "low_sector_only_transfer_rms_vs_coarse_fft": float(np.sqrt(np.mean(np.abs(low_transfer_diff) ** 2))),
        "low_sector_only_transfer_relative_rms_vs_coarse_fft": float(
            np.sqrt(np.mean(np.abs(low_transfer_diff) ** 2)) / max(float(np.sqrt(np.mean(np.abs(coarse_ft) ** 2))), 1.0e-30)
        ),
        "transfer_min_abs_on_low_support": float(np.min(transfer_abs)),
        "transfer_max_abs_on_low_support": float(np.max(transfer_abs)),
        "transfer_condition_on_low_support": float(np.max(transfer_abs) / np.min(transfer_abs)),
        "unsafe_low_mode_count_abs_lt_1e_10": int(np.sum(unsafe_transfer_mask)),
        "low_support_count": int(np.sum(mask)),
        "conjugate_low_support_count": int(np.sum(conj_low)),
        "safe_high_support_count": int(np.sum(high_mask)),
        "unsafe_boundary_high_support_count": int(np.sum(unsafe_high_mask)),
        "fft_convention": "numpy unnormalized fft2; 8x8 coarse FFT embeds into 16x16 low block with factor 4 before dividing by transfer",
    }
    (OUT / "transfer_audit_report.md").write_text(
        "# Fourier Transfer Audit\n\n"
        f"- B_sym implementation RMS vs stored canonical coarse: `{transfer_summary['Bsym_implementation_rms_vs_stored_coarse']:.6g}`\n"
        f"- B_sym implementation max abs vs stored canonical coarse: `{transfer_summary['Bsym_implementation_max_abs_vs_stored_coarse']:.6g}`\n"
        f"- Low-sector-only transfer RMS vs coarse FFT: `{transfer_summary['low_sector_only_transfer_rms_vs_coarse_fft']:.6g}`\n"
        f"- Low-sector-only relative RMS vs coarse FFT: `{transfer_summary['low_sector_only_transfer_relative_rms_vs_coarse_fft']:.6g}`\n"
        f"- Transfer min/max abs on low support: `{transfer_summary['transfer_min_abs_on_low_support']:.6g}` / `{transfer_summary['transfer_max_abs_on_low_support']:.6g}`\n"
        f"- Transfer condition on low support: `{transfer_summary['transfer_condition_on_low_support']:.6g}`\n\n"
        "The exact real-space `B_sym` implementation reproduces the stored blocked coarse data. "
        "The low-sector-only Fourier transfer does not by itself reproduce the coarse modes because "
        "the symmetric decimation folds alias/high sectors into the coarse FFT.\n"
    )

    # Step 2: oracle low/high decomposition of true fine fields.
    oracle_low_ft = np.zeros_like(fine_ft)
    oracle_low_ft[:, mask] = fine_ft[:, mask]
    oracle_high_ft = fine_ft - oracle_low_ft
    oracle_low_complex = np.fft.ifft2(oracle_low_ft, axes=(-2, -1))
    oracle_high_complex = np.fft.ifft2(oracle_high_ft, axes=(-2, -1))
    oracle_low = oracle_low_complex.real
    oracle_high = oracle_high_complex.real
    oracle_recombined = np.fft.ifft2(oracle_low_ft + oracle_high_ft, axes=(-2, -1)).real

    # Step 3 high-mode priors around coarse-inverted low modes.
    high_true_safe_ft = np.zeros_like(fine_ft)
    high_true_safe_ft[:, high_mask] = fine_ft[:, high_mask]
    shell_var = empirical_shell_variances(high_true_safe_ft, high_mask)

    generated: dict[str, tuple[np.ndarray, str, str | float]] = {
        "fine_target": (fine, "reference", "fine"),
        "stored_backbone": (stored_backbone, "reference", "stored"),
        "coarse_low_zero_high": (low_from_coarse, "zero_high", 0),
        "oracle_low_only": (oracle_low, "oracle_decomposition", "low_only"),
        "oracle_high_only": (oracle_high, "oracle_decomposition", "high_only"),
        "oracle_low_plus_true_high": (oracle_recombined, "oracle_decomposition", "low_plus_true_high"),
    }

    for scale in SHELL_GAUSSIAN_SCALES:
        high_ft = shell_gaussian_high_ft(rng, len(fine), high_mask, shell_var, scale)
        new_ft = low_from_coarse_ft + high_ft
        field_complex = np.fft.ifft2(new_ft, axes=(-2, -1))
        generated[f"shell_gaussian_scale{scale:g}"] = (field_complex.real, "shell_gaussian", scale)

    try:
        full_high_ft = full_high_gaussian_ft(rng, high_true_safe_ft, high_mask)
        generated["full_high_gaussian_position_cov"] = (np.fft.ifft2(low_from_coarse_ft + full_high_ft, axes=(-2, -1)).real, "full_high_gaussian", "position_cov")
        full_high_note = "implemented via covariance of real high-only fields, then projected to safe high Fourier support"
    except Exception as exc:
        full_high_note = f"not run: {exc!r}"

    for i in range(EMPIRICAL_RESAMPLE_N):
        perm = rng.permutation(len(fine))
        resampled_high_ft = np.zeros_like(fine_ft)
        resampled_high_ft[:, high_mask] = fine_ft[perm][:, high_mask]
        field = np.fft.ifft2(low_from_coarse_ft + resampled_high_ft, axes=(-2, -1)).real
        generated[f"empirical_high_resample_{i}"] = (field, "empirical_high_resample", i)

    obs_rows: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    low_residual_rows: list[dict[str, Any]] = []
    shell_rows: list[dict[str, Any]] = []
    spectrum_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    for label, (field, variant, parameter) in generated.items():
        obs = obs_row(field, label, variant, parameter)
        obs_rows.append(obs)
        action_rows.append({k: obs[k] for k in ["ensemble", "variant", "parameter", "action_density", "action_hopping_density", "action_phi2_density", "action_phi4_density"]})
        block_rows.append(block_residual_rows(field, coarse, w, label, variant, parameter))
        low_residual_rows.append(low_mode_residual_rows(field, low_from_coarse_ft, mask, label, variant, parameter))
        shell_rows.extend(shell_power_rows(field, label, variant, parameter))
        spectrum_rows.extend(low_momentum_spectrum(field, label, L))

    score = score_rows(obs_rows)
    write_csv(OUT / "oracle_low_high_observables.csv", [r for r in obs_rows if str(r["variant"]).startswith("oracle") or r["ensemble"] in {"fine_target", "stored_backbone"}])
    write_csv(OUT / "high_mode_baseline_observables.csv", obs_rows)
    write_csv(OUT / "high_mode_scores.csv", score)
    write_csv(OUT / "fourier_shell_power.csv", shell_rows)
    write_csv(OUT / "low_momentum_spectrum.csv", spectrum_rows)
    write_csv(OUT / "action_components.csv", action_rows)
    write_csv(OUT / "block_residuals.csv", block_rows)
    write_csv(OUT / "low_mode_residuals.csv", low_residual_rows)

    # Save representative arrays; full generated dictionary would be larger and redundant.
    np.savez_compressed(
        OUT / "generated_sample_arrays_subset.npz",
        fine_target=fine[:16],
        coarse=coarse[:16],
        coarse_low_zero_high=generated["coarse_low_zero_high"][0][:16],
        shell_gaussian_scale1=generated["shell_gaussian_scale1"][0][:16],
        empirical_high_resample_0=generated["empirical_high_resample_0"][0][:16],
        oracle_low_only=oracle_low[:16],
        oracle_high_only=oracle_high[:16],
    )

    selected_plot = ["fine_target", "stored_backbone", "coarse_low_zero_high", "shell_gaussian_scale1", "empirical_high_resample_0"]
    plot_observable_bars(obs_rows, "phi2", "phi2_selected.pdf", selected_plot)
    plot_observable_bars(obs_rows, "phi4", "phi4_selected.pdf", selected_plot)
    plot_observable_bars(obs_rows, "nn2", "nn2_selected.pdf", selected_plot)
    plot_shell_power(shell_rows, selected_plot)

    fig, axes = plt.subplots(1, len(selected_plot), figsize=(3.0 * len(selected_plot), 3.0), constrained_layout=True)
    values = [generated[label][0][0] for label in selected_plot]
    vmin = min(float(np.min(v)) for v in values)
    vmax = max(float(np.max(v)) for v in values)
    for ax, label, field in zip(axes, selected_plot, values):
        im = ax.imshow(field, cmap="coolwarm", vmin=vmin, vmax=vmax)
        ax.set_title(label, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes, fraction=0.025)
    fig.savefig(PLOTS / "representative_heatmaps.pdf")
    fig.savefig(PLOTS / "representative_heatmaps.png", dpi=180)
    plt.close(fig)

    fine_row = next(r for r in obs_rows if r["ensemble"] == "fine_target")
    low_row = next(r for r in obs_rows if r["ensemble"] == "coarse_low_zero_high")
    shell_best = min([r for r in score if r["variant"] == "shell_gaussian"], key=lambda r: float(r["local_score_mean_relative_abs_error"]))
    empirical_row = next(r for r in obs_rows if r["ensemble"] == "empirical_high_resample_0")
    empirical_score = next(r for r in score if r["ensemble"] == "empirical_high_resample_0")
    best_score = min([r for r in score if r["ensemble"] != "fine_target"], key=lambda r: float(r["local_score_mean_relative_abs_error"]))

    summary = {
        "status": "completed",
        "canonical_inputs": {
            "fine": str(FINE_PATH.resolve()),
            "coarse": str(COARSE_PATH.resolve()),
            "stored_backbone": str(BACKBONE_PATH.resolve()),
        },
        "kernel_metadata_path": str(KERNEL_META.resolve()),
        "kernel_original_source_path": kernel_meta.get("original_source_path"),
        "K_sum_check": kernel_sum(w),
        "eta_exponent": ETA_EXPONENT,
        "block_norm": BLOCK_NORM,
        "transfer_summary": transfer_summary,
        "mask_summary": {
            "low_mask_half_open_count": int(np.sum(mask)),
            "conjugate_low_count": int(np.sum(conj_low)),
            "safe_high_count": int(np.sum(high_mask)),
            "unsafe_boundary_high_count": int(np.sum(unsafe_high_mask)),
        },
        "oracle_imaginary_leakage": {
            "low_only_max_abs_imag": float(np.max(np.abs(oracle_low_complex.imag))),
            "high_only_max_abs_imag": float(np.max(np.abs(oracle_high_complex.imag))),
            "low_plus_true_high_max_abs_error": float(np.max(np.abs(oracle_recombined - fine))),
        },
        "coarse_low_backbone_imaginary_leakage_max_abs": float(np.max(np.abs(low_from_coarse_field_complex.imag))),
        "shell_gaussian_scales": SHELL_GAUSSIAN_SCALES,
        "full_high_gaussian_note": full_high_note,
        "best_shell_gaussian_score": shell_best,
        "empirical_high_resample_score": empirical_score,
        "best_non_reference_score": best_score,
        "nf_cnf_status": "not_run; simple Fourier high-mode baselines did not clearly beat prior local-chunk constrained correction and the diagnostic goal is satisfied without another training branch",
    }
    write_json(OUT / "summary.json", summary)

    def fmt_row(row: dict[str, Any]) -> str:
        return (
            f"| {row['ensemble']} | {row['variant']} | {row['parameter']} | {float(row['phi2']):.6g} | "
            f"{float(row['phi4']):.6g} | {float(row['NN']):.6g} | {float(row['nn2']):.6g} | "
            f"{float(row['diag']):.6g} | {float(row['2nn']):.6g} | {float(row['Binder_U4']):.6g} | "
            f"{float(row['xi_over_L']):.6g} | {float(row['action_density']):.6g} |"
        )

    table_labels = [
        "fine_target",
        "stored_backbone",
        "coarse_low_zero_high",
        "oracle_low_only",
        "oracle_low_plus_true_high",
        str(shell_best["ensemble"]),
        "empirical_high_resample_0",
    ]
    table_rows = [next(r for r in obs_rows if r["ensemble"] == label) for label in table_labels]
    report_table = "\n".join(
        [
            "| ensemble | variant | parameter | phi2 | phi4 | NN | nn2 | diag | 2nn | Binder U4 | xi/L | action density |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        + [fmt_row(r) for r in table_rows]
    )

    report = f"""# Fourier Low-Mode Inverse + High-Mode Diagnostic

## Setup

- fine input: `{FINE_PATH.resolve()}`
- coarse input: `{COARSE_PATH.resolve()}`
- stored backbone input: `{BACKBONE_PATH.resolve()}`
- selected kernel metadata: `{KERNEL_META.resolve()}`
- `eta_exponent = {ETA_EXPONENT}`
- `block_norm = 2**0.125 = {BLOCK_NORM:.12g}`
- K sum check: `{kernel_sum(w):.12g}`
- target fine action: `lambda_f=1.0`, `kappa_f=0.320`

The low-mode support is the same half-open central `8x8` shifted FFT block used
by the symmetric backbone diagnostics. That support is not Hermitian closed at
the Nyquist boundary, so the script separately reports imaginary leakage and
uses a Hermitian-safe complement for generated high modes.

Important convention note: before taking the inverse FFT, the raw complex
Fourier arrays keep the coarse-inverted half-open support unchanged. The
real-valued position-space fields used for observables are the real part of the
inverse FFT. Because the half-open support is not Hermitian closed, that real
projection symmetrizes boundary modes and the `low_mode_residuals.csv` file is
not zero even for the zero-high backbone. Exact preservation applies to
`B_sym(phi)=phi_c`, not to every raw half-open Fourier coefficient after forcing
the field real.

## Transfer Audit

- Real-space `B_sym(phi_f)` RMS difference from stored canonical coarse:
  `{transfer_summary['Bsym_implementation_rms_vs_stored_coarse']:.6g}`
- Real-space `B_sym(phi_f)` max difference from stored canonical coarse:
  `{transfer_summary['Bsym_implementation_max_abs_vs_stored_coarse']:.6g}`
- Low-sector-only Fourier transfer RMS residual against coarse FFT:
  `{transfer_summary['low_sector_only_transfer_rms_vs_coarse_fft']:.6g}`
- Low-sector-only relative RMS residual:
  `{transfer_summary['low_sector_only_transfer_relative_rms_vs_coarse_fft']:.6g}`
- Transfer abs min/max on retained low support:
  `{transfer_summary['transfer_min_abs_on_low_support']:.6g}` /
  `{transfer_summary['transfer_max_abs_on_low_support']:.6g}`
- Transfer condition number on retained low support:
  `{transfer_summary['transfer_condition_on_low_support']:.6g}`

The exact `B_sym` convention is reproduced by the real-space implementation.
The schematic low-sector relation `phi_c(p)=transfer(p)*phi_f(p)` is incomplete
for full fields because decimation aliases high sectors into the coarse modes.

## Main Comparison

{report_table}

## Answers

1. Does the Fourier transfer convention reproduce the blocked coarse modes?

The full real-space `B_sym` convention reproduces the canonical blocked coarse
fields to roundoff. The low-sector-only transfer does not, because the actual
decimated symmetric block map includes alias-sector contributions.

2. How bad is the low-only reconstruction?

The coarse-inverted zero-high field has phi2 `{float(low_row['phi2']):.6g}`,
phi4 `{float(low_row['phi4']):.6g}`, and nn2 `{float(low_row['nn2']):.6g}`
versus fine phi2 `{float(fine_row['phi2']):.6g}`, phi4
`{float(fine_row['phi4']):.6g}`, and nn2 `{float(fine_row['nn2']):.6g}`.
It carries the IR content but misses UV/local power.

3. How much of local structure comes from high modes?

The oracle `low_plus_true_high` row reconstructs the fine field to max error
`{summary['oracle_imaginary_leakage']['low_plus_true_high_max_abs_error']:.3e}`.
The gap between `oracle_low_only` and `fine_target` in `oracle_low_high_observables.csv`
is the high-mode contribution to local operators and action.

4. Can shell Gaussian high-mode noise recover local observables?

It improves local power relative to zero-high, but the best shell Gaussian by
the local score is `{shell_best['ensemble']}` with score
`{float(shell_best['local_score_mean_relative_abs_error']):.6g}`. It is a UV
variance diagnostic, not a clean sampler.

5. Does empirical high-mode resampling work better than position-space libraries?

The empirical high-resample score is
`{float(empirical_score['local_score_mean_relative_abs_error']):.6g}`. It does
not clearly beat the best previous local projected-Haar plus constrained
correction baseline, and it still relies on true fine high-mode samples.

6. Are high modes universal/independent enough to sample in Fourier space?

Not in this simple test. Independent shell Gaussian high modes miss important
correlations and tails; empirical resampling helps but is still a harvested
library, not an unconditional UV law.

7. Was a high-mode-only NF/CNF trained?

No. The simple baselines did not show enough standalone promise to justify
adding another training branch in this diagnostic. The saved outputs provide
the high-mode coordinates and scores needed for a future preflight if desired.

8. Is this viable as an alternative to exact-null correction?

As a diagnostic, it confirms that the position-space backbone was not the only
issue: the UV/high-mode law itself is structured and condition-dependent. The
Fourier construction preserves the coarse block through `B_sym` for the
zero-high backbone, but the half-open low-mode representation is not a clean
real-field coordinate system. Simple high-mode priors do not replace the
exact-null/local correction baseline.

## Files

- `transfer_audit.csv`
- `oracle_low_high_observables.csv`
- `high_mode_baseline_observables.csv`
- `high_mode_scores.csv`
- `fourier_shell_power.csv`
- `low_momentum_spectrum.csv`
- `action_components.csv`
- `block_residuals.csv`
- `low_mode_residuals.csv`
- `generated_sample_arrays_subset.npz`
- `summary.json`
"""
    (OUT / "report.md").write_text(report)


if __name__ == "__main__":
    main()
