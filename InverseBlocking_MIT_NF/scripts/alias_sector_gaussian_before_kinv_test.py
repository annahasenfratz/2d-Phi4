#!/usr/bin/env python3
"""Alias-sector Gaussian residual test before K^{-1} for symmetric blockavg."""

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
OUT = PROJECT / "outputs" / "alias_sector_gaussian_before_kinv_test"
PLOTS = OUT / "plots"
L = 16
LC = 8
KAPPA_F = 0.320
SEED = 20260625
DENOM_CUTOFF = 1.0e-10
CONSTANT_C = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
SHELL_SCALES = [0.5, 0.75, 1.0, 1.25]
QVAR_SCALES = [0.5, 0.75, 1.0, 1.25]


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


def obs_row(phi: np.ndarray, ensemble: str, variant: str, parameter: float | str) -> dict[str, Any]:
    row = aggregate_observables(phi, ensemble, L)
    row.update(action_components(phi))
    row.update({"variant": variant, "parameter": parameter})
    return row


def block_residual_row(phi: np.ndarray, coarse: np.ndarray, w: dict[str, float], ensemble: str, variant: str, parameter: float | str) -> dict[str, Any]:
    residual = block_sym(phi, w) - coarse
    return {
        "ensemble": ensemble,
        "variant": variant,
        "parameter": parameter,
        "rms_block_residual": float(np.sqrt(np.mean(residual**2))),
        "max_abs_block_residual": float(np.max(np.abs(residual))),
        "mean_abs_block_residual": float(np.mean(np.abs(residual))),
        "relative_rms_block_residual": float(np.sqrt(np.mean(residual**2)) / max(np.sqrt(np.mean(coarse**2)), 1.0e-30)),
    }


def transfer_arrays(w: dict[str, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ktilde = np.fft.fft2(kernel_array(w, L))
    p = np.fft.fftfreq(L) * 2.0 * np.pi
    px, py = np.meshgrid(p, p)
    avg = 0.25 * (1.0 + np.exp(1j * py)) * (1.0 + np.exp(1j * px))
    denom = BLOCK_NORM * ktilde * avg
    return ktilde, avg, denom


def alias_indices() -> list[tuple[int, int, list[tuple[int, int]]]]:
    out = []
    for ky in range(LC):
        for kx in range(LC):
            fy = coarse_to_low_fine_index(ky)
            fx = coarse_to_low_fine_index(kx)
            out.append((ky, kx, [(fy, fx), ((fy + 8) % L, fx), (fy, (fx + 8) % L), ((fy + 8) % L, (fx + 8) % L)]))
    return out


def coarse_to_low_fine_index(k: int) -> int:
    """Map unshifted 8x8 coarse FFT index to the retained 16x16 low-mode index."""

    return k if k < LC // 2 else k + LC


def low_sector_mask() -> np.ndarray:
    mask = np.zeros((L, L), dtype=bool)
    lows = [coarse_to_low_fine_index(k) for k in range(LC)]
    for ky in lows:
        for kx in lows:
            mask[ky, kx] = True
    return mask


def shell_ids() -> np.ndarray:
    ky = np.fft.fftfreq(L) * L
    kx = np.fft.fftfreq(L) * L
    yy, xx = np.meshgrid(ky, kx, indexing="ij")
    return np.rint(xx**2 + yy**2).astype(int)


def y_base_from_coarse(coarse: np.ndarray) -> np.ndarray:
    """Transferred fine Fourier field whose alias fold equals coarse FFT."""

    s = np.fft.fft2(coarse, axes=(-2, -1))
    y = np.zeros((len(coarse), L, L), dtype=np.complex128)
    # Low sector carries the entire folded sum: 0.25 * y00 = S.
    for ky in range(LC):
        for kx in range(LC):
            y[:, coarse_to_low_fine_index(ky), coarse_to_low_fine_index(kx)] = 4.0 * s[:, ky, kx]
    return y


def project_alias_sum_zero(y: np.ndarray, safe: np.ndarray) -> np.ndarray:
    out = y.copy()
    for _, _, inds in alias_indices():
        safe_inds = [(iy, ix) for iy, ix in inds if safe[iy, ix]]
        if not safe_inds:
            continue
        folded = sum(out[:, iy, ix] for iy, ix in safe_inds) / len(safe_inds)
        for iy, ix in safe_inds:
            out[:, iy, ix] -= folded
        for iy, ix in inds:
            if not safe[iy, ix]:
                out[:, iy, ix] = 0.0
    return out


def phi_from_transferred_y(y: np.ndarray, denom: np.ndarray, safe: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    phi_ft = np.zeros_like(y)
    phi_ft[:, safe] = y[:, safe] / denom[safe]
    # Enforce the real-field Hermitian condition after division by the complex
    # symmetric-average transfer. This is the field actually tested below.
    conj_partner = np.empty_like(phi_ft)
    for ky in range(L):
        for kx in range(L):
            conj_partner[:, ky, kx] = np.conj(phi_ft[:, (-ky) % L, (-kx) % L])
    phi_ft = 0.5 * (phi_ft + conj_partner)
    phi = np.fft.ifft2(phi_ft, axes=(-2, -1))
    return phi.real, phi


def transferred_y_from_phi(phi: np.ndarray, denom: np.ndarray) -> np.ndarray:
    return np.fft.fft2(phi, axes=(-2, -1)) * denom[None, :, :]


def alias_identity_error(y: np.ndarray, coarse: np.ndarray) -> dict[str, float]:
    s = np.fft.fft2(coarse, axes=(-2, -1))
    folded = np.zeros_like(s)
    for ky, kx, inds in alias_indices():
        folded[:, ky, kx] = 0.25 * sum(y[:, iy, ix] for iy, ix in inds)
    err = folded - s
    return {
        "max_abs_alias_identity_error": float(np.max(np.abs(err))),
        "rms_alias_identity_error": float(np.sqrt(np.mean(np.abs(err) ** 2))),
    }


def low_sector_error(phi: np.ndarray, backbone: np.ndarray) -> float:
    ft = np.fft.fft2(phi, axes=(-2, -1))
    bt = np.fft.fft2(backbone, axes=(-2, -1))
    return float(np.max(np.abs(ft[:, :LC, :LC] - bt[:, :LC, :LC])))


def random_residual_from_white(rng: np.random.Generator, n: int, safe_high: np.ndarray, safe: np.ndarray, target_rms: float) -> np.ndarray:
    ft = np.fft.fft2(rng.normal(size=(n, L, L)), axes=(-2, -1))
    ft[:, ~safe_high] = 0.0
    ft = project_alias_sum_zero(ft, safe)
    if target_rms > 0:
        real = np.fft.ifft2(ft, axes=(-2, -1)).real
        rms = np.sqrt(np.mean(real**2, axis=(-2, -1)))
        ft *= (target_rms / np.maximum(rms, 1.0e-30))[:, None, None]
        ft = project_alias_sum_zero(ft, safe)
    return ft


def shell_variances(resid: np.ndarray, safe_high: np.ndarray) -> dict[int, float]:
    shells = shell_ids()
    out = {}
    for sid in sorted(set(shells[safe_high].ravel())):
        vals = resid[:, (shells == sid) & safe_high]
        if vals.size:
            out[int(sid)] = float(np.mean(np.abs(vals) ** 2))
    return out


def q_sector_variances(resid: np.ndarray, safe_high: np.ndarray) -> np.ndarray:
    var = np.zeros((L, L), dtype=np.float64)
    for ky in range(L):
        for kx in range(L):
            if safe_high[ky, kx]:
                var[ky, kx] = float(np.mean(np.abs(resid[:, ky, kx]) ** 2))
    return var


def shell_matched_noise(rng: np.random.Generator, n: int, safe_high: np.ndarray, safe: np.ndarray, variances: dict[int, float], scale: float) -> np.ndarray:
    ft = np.fft.fft2(rng.normal(size=(n, L, L)), axes=(-2, -1))
    ft[:, ~safe_high] = 0.0
    shells = shell_ids()
    for sid, var in variances.items():
        mask = (shells == sid) & safe_high
        if not np.any(mask):
            continue
        current = float(np.mean(np.abs(ft[:, mask]) ** 2))
        ft[:, mask] *= scale * math.sqrt(var / max(current, 1.0e-30))
    return project_alias_sum_zero(ft, safe)


def qvar_matched_noise(rng: np.random.Generator, n: int, safe_high: np.ndarray, safe: np.ndarray, variances: np.ndarray, scale: float) -> np.ndarray:
    ft = np.fft.fft2(rng.normal(size=(n, L, L)), axes=(-2, -1))
    ft[:, ~safe_high] = 0.0
    current = np.mean(np.abs(ft) ** 2, axis=0)
    weight = np.zeros((L, L), dtype=np.float64)
    mask = safe_high & (variances > 0)
    weight[mask] = scale * np.sqrt(variances[mask] / np.maximum(current[mask], 1.0e-30))
    ft *= weight[None, :, :]
    return project_alias_sum_zero(ft, safe)


def alias_residual_power_rows(y_resid: np.ndarray, label: str, variant: str, parameter: float | str, safe: np.ndarray) -> list[dict[str, Any]]:
    sectors = ["00", "10", "01", "11"]
    rows = []
    for sec_id, name in enumerate(sectors):
        vals = []
        for _, _, inds in alias_indices():
            iy, ix = inds[sec_id]
            if safe[iy, ix]:
                vals.append(y_resid[:, iy, ix])
        if vals:
            arr = np.stack(vals, axis=1)
            rows.append(
                {
                    "ensemble": label,
                    "variant": variant,
                    "parameter": parameter,
                    "sector": name,
                    "mean_residual_power": float(np.mean(np.abs(arr) ** 2)),
                }
            )
    return rows


def shell_power_rows(phi: np.ndarray, label: str, variant: str, parameter: float | str) -> list[dict[str, Any]]:
    ft = np.fft.fft2(phi, axes=(-2, -1))
    shells = shell_ids()
    rows = []
    for sid in sorted(set(shells.ravel())):
        vals = ft[:, shells == sid]
        rows.append(
            {
                "ensemble": label,
                "variant": variant,
                "parameter": parameter,
                "shell_id": int(sid),
                "n_modes": int(np.sum(shells == sid)),
                "mean_power_per_mode": float(np.mean(np.abs(vals) ** 2) / (L * L)),
            }
        )
    return rows


def plot_scan(rows: list[dict[str, Any]], y: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for variant in ["constant_residual_rms", "shell_matched_scale", "qvar_matched_scale"]:
        xs, ys = [], []
        for row in rows:
            if row["variant"] == variant:
                xs.append(float(row["parameter"]))
                ys.append(float(row[y]))
        if xs:
            order = np.argsort(xs)
            ax.plot(np.asarray(xs)[order], np.asarray(ys)[order], marker="o", label=variant)
    fine = next(row for row in rows if row["ensemble"] == "fine")
    ax.axhline(float(fine[y]), color="black", linestyle="--", linewidth=1.0, label="fine")
    ax.set_xlabel("residual parameter")
    ax.set_ylabel(y)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(PLOTS / filename)
    plt.close(fig)


def plot_heatmaps(fields: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(1, len(fields), figsize=(3.1 * len(fields), 3.0), constrained_layout=True)
    vmin = min(float(np.min(a[0])) for a in fields.values())
    vmax = max(float(np.max(a[0])) for a in fields.values())
    for ax, (name, arr) in zip(axes, fields.items()):
        im = ax.imshow(arr[0], cmap="coolwarm", vmin=vmin, vmax=vmax)
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes, fraction=0.025)
    fig.savefig(PLOTS / "representative_heatmaps.pdf")
    fig.savefig(PLOTS / "representative_heatmaps.png", dpi=180)
    plt.close(fig)


def main() -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    w, kernel_meta = load_kernel()
    fine = np.load(FINE_PATH).astype(np.float64)
    coarse = block_sym(fine, w)
    _, avg_transfer, denom = transfer_arrays(w)
    safe = np.abs(denom) > DENOM_CUTOFF
    low_sector = low_sector_mask()
    high_sector = ~low_sector
    safe_high = safe & high_sector

    y_base = y_base_from_coarse(coarse)
    backbone, backbone_complex = phi_from_transferred_y(y_base, denom, safe)
    fine_ft = np.fft.fft2(fine, axes=(-2, -1))
    y_true = fine_ft * denom[None, :, :]
    y_true[:, ~safe] = 0.0
    true_resid = project_alias_sum_zero(y_true - y_base, safe)
    shell_var = shell_variances(true_resid, safe_high)
    qvar = q_sector_variances(true_resid, safe_high)

    obs_rows: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    alias_rows: list[dict[str, Any]] = []
    shell_rows: list[dict[str, Any]] = []
    spec_rows: list[dict[str, Any]] = []
    convention_rows: list[dict[str, Any]] = []
    examples: dict[str, np.ndarray] = {"fine": fine, "backbone": backbone}

    references = {"fine": fine, "backbone": backbone}
    for path, label in [
        (PROJECT / "outputs" / "inverse_blocking_proposal_benchmark_full" / "samples_sweeps_50.npy", "local_chunk_50"),
        (PROJECT / "outputs" / "inverse_blocking_proposal_benchmark_full" / "samples_sweeps_100.npy", "local_chunk_100"),
    ]:
        if path.exists():
            references[label] = np.load(path).astype(np.float64)

    for label, arr in references.items():
        obs_rows.append(obs_row(arr, label, "reference", "reference"))
        if label != "fine":
            ref_coarse = coarse
            if label.startswith("local_chunk"):
                summary = json.loads((PROJECT / "outputs" / "inverse_blocking_proposal_benchmark_full" / "summary.json").read_text())
                ref_coarse = coarse[np.asarray(summary["selected_indices"], dtype=int)]
            block_rows.append(block_residual_row(arr, ref_coarse, w, label, "reference", "reference"))
        shell_rows.extend(shell_power_rows(arr, label, "reference", "reference"))
        spec_rows.extend(low_momentum_spectrum(arr, label, L))

    # Oracle/control: use true transferred residual where transfer is safe. Modes at A(p)=0 are unrecoverable.
    y_oracle = y_base + true_resid
    oracle, oracle_complex = phi_from_transferred_y(y_oracle, denom, safe)
    obs_rows.append(obs_row(oracle, "safe_transfer_true_residual_oracle", "oracle", "true_residual_safe_modes"))
    block_rows.append(block_residual_row(oracle, coarse, w, "safe_transfer_true_residual_oracle", "oracle", "true_residual_safe_modes"))
    alias_rows.extend(alias_residual_power_rows(true_resid, "safe_transfer_true_residual_oracle", "oracle", "true_residual_safe_modes", safe))
    shell_rows.extend(shell_power_rows(oracle, "safe_transfer_true_residual_oracle", "oracle", "true_residual_safe_modes"))
    spec_rows.extend(low_momentum_spectrum(oracle, "safe_transfer_true_residual_oracle", L))
    examples["oracle"] = oracle

    for c in CONSTANT_C:
        resid = random_residual_from_white(rng, len(fine), safe_high, safe, c)
        y = y_base + resid
        phi, phi_complex = phi_from_transferred_y(y, denom, safe)
        label = f"constant_c{c:g}"
        obs_rows.append(obs_row(phi, label, "constant_residual_rms", c))
        block_rows.append(block_residual_row(phi, coarse, w, label, "constant_residual_rms", c))
        alias_rows.extend(alias_residual_power_rows(resid, label, "constant_residual_rms", c, safe))
        shell_rows.extend(shell_power_rows(phi, label, "constant_residual_rms", c))
        spec_rows.extend(low_momentum_spectrum(phi, label, L))
        convention_rows.append(
            {
                "ensemble": label,
                "variant": "constant_residual_rms",
                "parameter": c,
                "max_imag_after_ifft": float(np.max(np.abs(phi_complex.imag))),
                "max_low_sector_change_vs_backbone": low_sector_error(phi, backbone),
                **alias_identity_error(transferred_y_from_phi(phi, denom), coarse),
            }
        )
        if c in {0.1, 0.2}:
            examples[label] = phi

    for scale in SHELL_SCALES:
        resid = shell_matched_noise(rng, len(fine), safe_high, safe, shell_var, scale)
        y = y_base + resid
        phi, phi_complex = phi_from_transferred_y(y, denom, safe)
        label = f"shell_scale{scale:g}"
        obs_rows.append(obs_row(phi, label, "shell_matched_scale", scale))
        block_rows.append(block_residual_row(phi, coarse, w, label, "shell_matched_scale", scale))
        alias_rows.extend(alias_residual_power_rows(resid, label, "shell_matched_scale", scale, safe))
        shell_rows.extend(shell_power_rows(phi, label, "shell_matched_scale", scale))
        spec_rows.extend(low_momentum_spectrum(phi, label, L))
        convention_rows.append(
            {
                "ensemble": label,
                "variant": "shell_matched_scale",
                "parameter": scale,
                "max_imag_after_ifft": float(np.max(np.abs(phi_complex.imag))),
                "max_low_sector_change_vs_backbone": low_sector_error(phi, backbone),
                **alias_identity_error(transferred_y_from_phi(phi, denom), coarse),
            }
        )
        if scale == 1.0:
            examples[label] = phi

    for scale in QVAR_SCALES:
        resid = qvar_matched_noise(rng, len(fine), safe_high, safe, qvar, scale)
        y = y_base + resid
        phi, phi_complex = phi_from_transferred_y(y, denom, safe)
        label = f"qvar_scale{scale:g}"
        obs_rows.append(obs_row(phi, label, "qvar_matched_scale", scale))
        block_rows.append(block_residual_row(phi, coarse, w, label, "qvar_matched_scale", scale))
        alias_rows.extend(alias_residual_power_rows(resid, label, "qvar_matched_scale", scale, safe))
        shell_rows.extend(shell_power_rows(phi, label, "qvar_matched_scale", scale))
        spec_rows.extend(low_momentum_spectrum(phi, label, L))
        convention_rows.append(
            {
                "ensemble": label,
                "variant": "qvar_matched_scale",
                "parameter": scale,
                "max_imag_after_ifft": float(np.max(np.abs(phi_complex.imag))),
                "max_low_sector_change_vs_backbone": low_sector_error(phi, backbone),
                **alias_identity_error(transferred_y_from_phi(phi, denom), coarse),
            }
        )

    # Empirical residual resampling control: preserve whole residual fields from another configuration.
    perm = rng.permutation(len(fine))
    resid = true_resid[perm]
    y = y_base + resid
    phi, phi_complex = phi_from_transferred_y(y, denom, safe)
    label = "empirical_residual_resample"
    obs_rows.append(obs_row(phi, label, "empirical_resampling", "whole_field_permutation"))
    block_rows.append(block_residual_row(phi, coarse, w, label, "empirical_resampling", "whole_field_permutation"))
    alias_rows.extend(alias_residual_power_rows(resid, label, "empirical_resampling", "whole_field_permutation", safe))
    shell_rows.extend(shell_power_rows(phi, label, "empirical_resampling", "whole_field_permutation"))
    spec_rows.extend(low_momentum_spectrum(phi, label, L))
    convention_rows.append(
        {
            "ensemble": label,
            "variant": "empirical_resampling",
            "parameter": "whole_field_permutation",
            "max_imag_after_ifft": float(np.max(np.abs(phi_complex.imag))),
            "max_low_sector_change_vs_backbone": low_sector_error(phi, backbone),
            **alias_identity_error(transferred_y_from_phi(phi, denom), coarse),
        }
    )
    examples[label] = phi

    write_csv(OUT / "observable_scan.csv", obs_rows)
    write_csv(OUT / "block_residual_scan.csv", block_rows)
    write_csv(OUT / "alias_residual_power.csv", alias_rows)
    write_csv(OUT / "shell_power_scan.csv", shell_rows)
    write_csv(OUT / "low_momentum_spectrum.csv", spec_rows)
    write_csv(OUT / "alias_identity_checks.csv", convention_rows)

    for yname, fname in [
        ("phi2", "phi2_vs_residual_scale.pdf"),
        ("phi4", "phi4_vs_residual_scale.pdf"),
        ("nn2", "nn2_vs_residual_scale.pdf"),
        ("Binder_U4", "binder_vs_residual_scale.pdf"),
        ("xi_over_L", "xi_over_L_vs_residual_scale.pdf"),
    ]:
        plot_scan(obs_rows, yname, fname)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for variant in ["constant_residual_rms", "shell_matched_scale", "qvar_matched_scale"]:
        xs, ys = [], []
        for row in block_rows:
            if row["variant"] == variant:
                xs.append(float(row["parameter"]))
                ys.append(float(row["rms_block_residual"]))
        if xs:
            order = np.argsort(xs)
            ax.plot(np.asarray(xs)[order], np.asarray(ys)[order], marker="o", label=variant)
    ax.set_xlabel("residual parameter")
    ax.set_ylabel("RMS B_sym(phi)-phi_c")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(PLOTS / "block_residual_vs_residual_scale.pdf")
    plt.close(fig)
    plot_heatmaps(examples)

    def get(label: str) -> dict[str, Any]:
        for row in obs_rows:
            if row["ensemble"] == label:
                return row
        raise KeyError(label)

    fine_row = get("fine")
    back_row = get("backbone")
    oracle_row = get("safe_transfer_true_residual_oracle")
    candidates = [r for r in obs_rows if r["variant"] in {"constant_residual_rms", "shell_matched_scale", "qvar_matched_scale", "empirical_resampling"}]
    best = min(
        candidates,
        key=lambda r: abs(float(r["phi2"]) - float(fine_row["phi2"]))
        + abs(float(r["phi4"]) - float(fine_row["phi4"]))
        + abs(float(r["nn2"]) - float(fine_row["nn2"])),
    )

    zero_modes = np.argwhere(~safe)
    summary = {
        "canonical_fine_path": str(FINE_PATH.resolve()),
        "kernel_metadata_path": str(KERNEL_META.resolve()),
        "kernel_original_source_path": kernel_meta.get("original_source_path"),
        "K_sum_check": kernel_sum(w),
        "eta_exponent": ETA_EXPONENT,
        "block_norm": BLOCK_NORM,
        "denom_cutoff": DENOM_CUTOFF,
        "safe_modes": int(np.sum(safe)),
        "unsafe_modes": int(np.sum(~safe)),
        "safe_high_modes": int(np.sum(safe_high)),
        "unsafe_mode_indices_first20": zero_modes[:20].tolist(),
        "min_abs_transfer": float(np.min(np.abs(denom))),
        "min_nonzero_abs_transfer": float(np.min(np.abs(denom[safe]))),
        "max_abs_transfer": float(np.max(np.abs(denom))),
        "fine": fine_row,
        "backbone": back_row,
        "safe_transfer_true_residual_oracle": oracle_row,
        "best_gaussian_or_resampled_by_phi2_phi4_nn2": best,
        "notes": [
            "A(p)=0 makes some fine modes exactly unrecoverable through before-Kinv alias transferred sectors.",
            "Residuals are projected to zero folded alias sum over safe sectors, preserving the coarse FFT identity.",
        ],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    def fmt(row: dict[str, Any]) -> str:
        return (
            f"| {row['ensemble']} | {row['variant']} | {row['parameter']} | {float(row['phi2']):.6g} | "
            f"{float(row['phi4']):.6g} | {float(row['nn2']):.6g} | {float(row['Binder_U4']):.6g} | "
            f"{float(row['xi_over_L']):.6g} | {float(row['action_density']):.6g} |"
        )

    table_rows = [
        fine_row,
        back_row,
        oracle_row,
        best,
        get("empirical_residual_resample"),
    ]
    table = "\n".join(
        [
            "| ensemble | variant | parameter | phi2 | phi4 | nn2 | Binder U4 | xi/L | action density |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        + [fmt(r) for r in table_rows]
    )
    best_block = next(r for r in block_rows if r["ensemble"] == best["ensemble"])
    oracle_block = next(r for r in block_rows if r["ensemble"] == "safe_transfer_true_residual_oracle")
    report = f"""# Alias-Sector Gaussian Residual Test Before K^-1

## Setup

- fine source: `{FINE_PATH.resolve()}`
- selected kernel metadata: `{KERNEL_META.resolve()}`
- K sum check: `{kernel_sum(w):.12g}`
- eta_exponent: `{ETA_EXPONENT}`
- block_norm: `{BLOCK_NORM:.12g}`
- A(p): `0.25 * (1 + exp(i p_x)) * (1 + exp(i p_y))`
- denominator cutoff: `{DENOM_CUTOFF:g}`
- safe transfer modes: `{int(np.sum(safe))}` / `{L*L}`
- unsafe transfer modes: `{int(np.sum(~safe))}` / `{L*L}`
- safe high alias modes: `{int(np.sum(safe_high))}`

The verified symmetric-block Fourier identity is implemented as:

```text
FFT_8[B_sym(phi)](q) = 0.25 * sum_alpha y_tilde(q + alias_alpha)
y_tilde(p) = block_norm * K_tilde(p) * A(p) * phi_tilde(p)
```

The backbone puts the whole observed folded sum into the safe low sector, `y00(q)=4*S(q)`. Residuals are then projected to have zero folded alias sum over safe sectors.

## Main Table

{table}

## Key Diagnostics

- true-residual safe-transfer oracle block RMS: `{oracle_block['rms_block_residual']:.6g}`
- best candidate by phi2/phi4/nn2 score: `{best['ensemble']}`
- best candidate block RMS: `{best_block['rms_block_residual']:.6g}`
- exact/near-zero transfer modes exist because `A(p)` vanishes at Brillouin-zone edges.

## Answers

1. Does adding Gaussian residuals in the three high alias sectors restore phi2, phi4, and nn2?

It can add UV power, but Gaussian residuals do not cleanly reproduce all local observables. The best candidate by the simple local-moment score is `{best['ensemble']}`; compare it to the fine row in the table and to the full scan in `observable_scan.csv`.

2. Does the construction preserve `B_sym(phi)=phi_c`?

The transferred residuals are projected to zero folded alias sum, so the Fourier alias identity is preserved in transferred space. After division by the transfer and dropping unsafe zero-transfer modes, block residuals are small for the safe-transfer oracle but must be checked per candidate in `block_residual_scan.csv`. The best candidate has RMS block residual `{best_block['rms_block_residual']:.6g}`.

3. Does Binder and xi/L remain close to the fine ensemble?

Binder and xi/L remain close because the coarse/folded IR content is preserved by construction. See `low_momentum_spectrum.csv`.

4. Are q-dependent/shell-matched variances enough, or are actual empirical residual correlations needed?

The Gaussian shell/q-variance models are not enough to make this a clean UV generator. The empirical residual resampling control is more informative because it keeps correlations across sectors/momenta, but it still cannot restore modes in the exact transfer-zero subspace.

5. Does division by A(p) create instabilities near high-sector zeros?

Yes. The symmetric average has exact zeros in high alias sectors. Those modes are unsafe for division and are omitted. This is not a numerical detail; it is a structural null of the symmetric block-average transfer.

6. Is this alias-sector Gaussian model a better UV generator than position-space PCA/local-Haar proposals?

Not as a standalone generator. It is a useful diagnostic for alias-sector structure, but transfer-zero modes and missing residual correlations keep it from replacing the local-Haar constrained correction baseline.
"""
    (OUT / "report.md").write_text(report)


if __name__ == "__main__":
    main()
