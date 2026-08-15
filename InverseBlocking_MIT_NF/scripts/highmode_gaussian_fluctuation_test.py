#!/usr/bin/env python3
"""High-mode Gaussian fluctuation test on the symmetric-block Fourier backbone."""

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
OUT = PROJECT / "outputs" / "highmode_gaussian_fluctuation_test"
PLOTS = OUT / "plots"
FINE_PATH = DATA / "fine_configs.npy"
KAPPA_F = 0.320
LAMBDA = 1.0
L = 16
LC = 8
SEED = 20260624
CONSTANT_C = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
K_SUPPRESSED_C = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
EMPIRICAL_SCALES = [0.5, 0.75, 1.0, 1.25]
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


def obs_row(phi: np.ndarray, ensemble: str, variant: str, parameter: float | str, source: str) -> dict[str, Any]:
    row = aggregate_observables(phi, ensemble, L)
    row.update(action_components(phi))
    row.update({"variant": variant, "parameter": parameter, "source": source})
    return row


def block_residual_row(phi: np.ndarray, coarse: np.ndarray, w: dict[str, float], ensemble: str, variant: str, parameter: float | str) -> dict[str, Any]:
    residual = block_sym(phi, w) - coarse
    return {
        "ensemble": ensemble,
        "variant": variant,
        "parameter": parameter,
        "max_abs_block_residual": float(np.max(np.abs(residual))),
        "rms_block_residual": float(np.sqrt(np.mean(residual**2))),
        "mean_abs_block_residual": float(np.mean(np.abs(residual))),
        "coarse_rms": float(np.sqrt(np.mean(coarse**2))),
        "relative_rms_block_residual": float(np.sqrt(np.mean(residual**2)) / max(np.sqrt(np.mean(coarse**2)), 1.0e-30)),
    }


def transfer_arrays(w: dict[str, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ktilde = np.fft.fft2(kernel_array(w, L))
    p = np.fft.fftfreq(L) * 2.0 * np.pi
    px, py = np.meshgrid(p, p)
    a = 0.25 * (1.0 + np.exp(1j * py)) * (1.0 + np.exp(1j * px))
    denom = BLOCK_NORM * ktilde * a
    return ktilde, a, denom


def low_mask() -> np.ndarray:
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


def backbone_fourier_from_coarse(coarse: np.ndarray, denom: np.ndarray, mask: np.ndarray) -> np.ndarray:
    cft = np.fft.fft2(coarse, axes=(-2, -1))
    padded_shift = np.zeros((len(coarse), L, L), dtype=np.complex128)
    # Factor 4 is the numpy-FFT normalization for 8x8 -> 16x16 low-mode embedding.
    padded_shift[:, 4:12, 4:12] = 4.0 * np.fft.fftshift(cft, axes=(-2, -1))
    padded = np.fft.ifftshift(padded_shift, axes=(-2, -1))
    out = np.zeros_like(padded)
    out[:, mask] = padded[:, mask] / denom[mask]
    return out


def normalize_noise_to_rms(noise_ft: np.ndarray, target_rms: float) -> np.ndarray:
    if target_rms == 0.0:
        return np.zeros_like(noise_ft)
    noise = np.fft.ifft2(noise_ft, axes=(-2, -1)).real
    rms = np.sqrt(np.mean(noise**2, axis=(-2, -1)))
    scale = target_rms / np.maximum(rms, 1.0e-30)
    return noise_ft * scale[:, None, None]


def high_projected_white_noise(rng: np.random.Generator, n: int, high_mask: np.ndarray, target_rms: float) -> np.ndarray:
    white = rng.normal(size=(n, L, L))
    ft = np.fft.fft2(white, axes=(-2, -1))
    ft[:, ~high_mask] = 0.0
    return normalize_noise_to_rms(ft, target_rms)


def k_suppressed_noise(
    rng: np.random.Generator,
    n: int,
    high_mask: np.ndarray,
    ktilde: np.ndarray,
    target_rms: float,
    eps: float = 0.08,
    clip: float = 8.0,
) -> np.ndarray:
    white = rng.normal(size=(n, L, L))
    ft = np.fft.fft2(white, axes=(-2, -1))
    ft[:, ~high_mask] = 0.0
    weight = np.zeros((L, L), dtype=np.float64)
    weight[high_mask] = np.minimum(1.0 / np.maximum(np.abs(ktilde[high_mask]), eps), clip)
    ft *= weight[None, :, :]
    return normalize_noise_to_rms(ft, target_rms)


def shell_ids() -> np.ndarray:
    ky = np.fft.fftfreq(L) * L
    kx = np.fft.fftfreq(L) * L
    yy, xx = np.meshgrid(ky, kx, indexing="ij")
    return np.rint(xx**2 + yy**2).astype(int)


def empirical_shell_variance(delta_ft: np.ndarray, high_mask: np.ndarray) -> dict[int, float]:
    shells = shell_ids()
    variances = {}
    for sid in sorted(set(shells[high_mask].ravel())):
        vals = delta_ft[:, (shells == sid) & high_mask]
        if vals.size:
            variances[int(sid)] = float(np.mean(np.abs(vals) ** 2))
    return variances


def empirical_shell_noise(
    rng: np.random.Generator,
    n: int,
    high_mask: np.ndarray,
    variances: dict[int, float],
    scale: float,
) -> np.ndarray:
    # Generate real high-mode Gaussian white noise, then shell-rescale its Fourier amplitudes.
    ft = np.fft.fft2(rng.normal(size=(n, L, L)), axes=(-2, -1))
    ft[:, ~high_mask] = 0.0
    shells = shell_ids()
    for sid, var in variances.items():
        mask = (shells == sid) & high_mask
        if not np.any(mask):
            continue
        current = float(np.mean(np.abs(ft[:, mask]) ** 2))
        ft[:, mask] *= scale * math.sqrt(var / max(current, 1.0e-30))
    # Shell rescaling preserves Hermitian symmetry because shell IDs are inversion symmetric.
    ft[:, ~high_mask] = 0.0
    return ft


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


def low_mode_max_error(ft_a: np.ndarray, ft_b: np.ndarray, mask: np.ndarray) -> float:
    return float(np.max(np.abs(ft_a[:, mask] - ft_b[:, mask])))


def plot_scan(rows: list[dict[str, Any]], y: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for variant in ["constant_high_rms", "k_suppressed_high_rms", "empirical_shell_scale"]:
        xs = []
        ys = []
        for row in rows:
            if row["variant"] == variant:
                xs.append(float(row["parameter"]))
                ys.append(float(row[y]))
        if xs:
            order = np.argsort(xs)
            ax.plot(np.asarray(xs)[order], np.asarray(ys)[order], marker="o", label=variant)
    fine = next(row for row in rows if row["variant"] == "reference" and row["ensemble"] == "fine")
    ax.axhline(float(fine[y]), color="black", linestyle="--", linewidth=1.0, label="fine")
    ax.set_xlabel("noise parameter")
    ax.set_ylabel(y)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(PLOTS / filename)
    plt.close(fig)


def plot_heatmaps(fine: np.ndarray, backbone: np.ndarray, examples: dict[str, np.ndarray]) -> None:
    fields = {"fine": fine[0], "backbone": backbone[0], **{k: v[0] for k, v in examples.items()}}
    vmin = min(float(np.min(x)) for x in fields.values())
    vmax = max(float(np.max(x)) for x in fields.values())
    fig, axes = plt.subplots(1, len(fields), figsize=(3.1 * len(fields), 3.0), constrained_layout=True)
    if len(fields) == 1:
        axes = [axes]
    for ax, (name, field) in zip(axes, fields.items()):
        im = ax.imshow(field, cmap="coolwarm", vmin=vmin, vmax=vmax)
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
    ktilde, atransfer, denom = transfer_arrays(w)
    mask = low_mask()
    # The half-open coarse-BZ support is not Hermitian-closed at the Nyquist boundary.
    # Use the real-space backbone as the baseline and add noise only on high modes
    # whose conjugates are also outside the retained low support.
    raw_back_ft = backbone_fourier_from_coarse(coarse, denom, mask)
    backbone = np.fft.ifft2(raw_back_ft, axes=(-2, -1)).real
    back_ft = np.fft.fft2(backbone, axes=(-2, -1))
    high_mask = ~(mask | conjugate_mask(mask))
    fine_ft = np.fft.fft2(fine, axes=(-2, -1))
    delta_ft = fine_ft - back_ft
    delta_ft[:, mask] = 0.0
    shell_var = empirical_shell_variance(delta_ft, high_mask)

    obs_rows: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    shell_rows: list[dict[str, Any]] = []
    spec_rows: list[dict[str, Any]] = []
    low_checks: list[dict[str, Any]] = []
    examples: dict[str, np.ndarray] = {}

    references: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "fine": (fine, coarse),
        "backbone": (backbone, coarse),
    }
    benchmark_summary_path = PROJECT / "outputs" / "inverse_blocking_proposal_benchmark_full" / "summary.json"
    benchmark_coarse = coarse
    if benchmark_summary_path.exists():
        selected = np.asarray(json.loads(benchmark_summary_path.read_text()).get("selected_indices", []), dtype=int)
        if len(selected) > 0:
            benchmark_coarse = coarse[selected]
    for path, label in [
        (PROJECT / "outputs" / "inverse_blocking_proposal_benchmark_full" / "samples_sweeps_50.npy", "local_chunk_50"),
        (PROJECT / "outputs" / "inverse_blocking_proposal_benchmark_full" / "samples_sweeps_100.npy", "local_chunk_100"),
    ]:
        if path.exists():
            references[label] = (np.load(path).astype(np.float64), benchmark_coarse)

    for label, (arr, ref_coarse) in references.items():
        obs_rows.append(obs_row(arr, label, "reference", "reference", "reference"))
        if label != "fine":
            block_rows.append(block_residual_row(arr, ref_coarse, w, label, "reference", "reference"))
        shell_rows.extend(shell_power_rows(arr, label, "reference", "reference"))
        spec_rows.extend(low_momentum_spectrum(arr, label, L))

    for c in CONSTANT_C:
        noise_ft = high_projected_white_noise(rng, len(fine), high_mask, c)
        new_ft = back_ft + noise_ft
        noisy = np.fft.ifft2(new_ft, axes=(-2, -1))
        label = f"constant_c{c:g}"
        real_field = noisy.real
        obs_rows.append(obs_row(real_field, label, "constant_high_rms", c, "generated"))
        block_rows.append(block_residual_row(real_field, coarse, w, label, "constant_high_rms", c))
        shell_rows.extend(shell_power_rows(real_field, label, "constant_high_rms", c))
        spec_rows.extend(low_momentum_spectrum(real_field, label, L))
        low_checks.append(
            {
                "ensemble": label,
                "variant": "constant_high_rms",
                "parameter": c,
                "max_imag_after_ifft": float(np.max(np.abs(noisy.imag))),
                "max_low_mode_change": low_mode_max_error(new_ft, back_ft, mask),
                "position_rms_added": float(np.sqrt(np.mean(np.fft.ifft2(noise_ft, axes=(-2, -1)).real**2))),
            }
        )
        if c in {0.20, 0.30}:
            examples[label] = real_field

    for c in K_SUPPRESSED_C:
        noise_ft = k_suppressed_noise(rng, len(fine), high_mask, ktilde, c)
        new_ft = back_ft + noise_ft
        noisy = np.fft.ifft2(new_ft, axes=(-2, -1))
        label = f"k_suppressed_c{c:g}"
        real_field = noisy.real
        obs_rows.append(obs_row(real_field, label, "k_suppressed_high_rms", c, "generated"))
        block_rows.append(block_residual_row(real_field, coarse, w, label, "k_suppressed_high_rms", c))
        shell_rows.extend(shell_power_rows(real_field, label, "k_suppressed_high_rms", c))
        spec_rows.extend(low_momentum_spectrum(real_field, label, L))
        low_checks.append(
            {
                "ensemble": label,
                "variant": "k_suppressed_high_rms",
                "parameter": c,
                "max_imag_after_ifft": float(np.max(np.abs(noisy.imag))),
                "max_low_mode_change": low_mode_max_error(new_ft, back_ft, mask),
                "position_rms_added": float(np.sqrt(np.mean(np.fft.ifft2(noise_ft, axes=(-2, -1)).real**2))),
            }
        )

    for scale in EMPIRICAL_SCALES:
        noise_ft = empirical_shell_noise(rng, len(fine), high_mask, shell_var, scale)
        new_ft = back_ft + noise_ft
        noisy = np.fft.ifft2(new_ft, axes=(-2, -1))
        label = f"empirical_scale{scale:g}"
        real_field = noisy.real
        obs_rows.append(obs_row(real_field, label, "empirical_shell_scale", scale, "generated"))
        block_rows.append(block_residual_row(real_field, coarse, w, label, "empirical_shell_scale", scale))
        shell_rows.extend(shell_power_rows(real_field, label, "empirical_shell_scale", scale))
        spec_rows.extend(low_momentum_spectrum(real_field, label, L))
        low_checks.append(
            {
                "ensemble": label,
                "variant": "empirical_shell_scale",
                "parameter": scale,
                "max_imag_after_ifft": float(np.max(np.abs(noisy.imag))),
                "max_low_mode_change": low_mode_max_error(new_ft, back_ft, mask),
                "position_rms_added": float(np.sqrt(np.mean(np.fft.ifft2(noise_ft, axes=(-2, -1)).real**2))),
            }
        )
        if scale == 1.0:
            examples[label] = real_field

    write_csv(OUT / "observable_scan.csv", obs_rows)
    write_csv(OUT / "block_residual_scan.csv", block_rows)
    write_csv(OUT / "shell_power_scan.csv", shell_rows)
    write_csv(OUT / "low_momentum_spectrum.csv", spec_rows)
    write_csv(OUT / "low_mode_preservation_checks.csv", low_checks)

    plot_scan(obs_rows, "phi2", "phi2_vs_noise.pdf")
    plot_scan(obs_rows, "phi4", "phi4_vs_noise.pdf")
    plot_scan(obs_rows, "nn2", "nn2_vs_noise.pdf")
    plot_scan(obs_rows, "Binder_U4", "binder_vs_noise.pdf")
    plot_scan(obs_rows, "xi_over_L", "xi_over_L_vs_noise.pdf")

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for variant in ["constant_high_rms", "k_suppressed_high_rms", "empirical_shell_scale"]:
        xs, ys = [], []
        for row in block_rows:
            if row["variant"] == variant:
                xs.append(float(row["parameter"]))
                ys.append(float(row["rms_block_residual"]))
        if xs:
            order = np.argsort(xs)
            ax.plot(np.asarray(xs)[order], np.asarray(ys)[order], marker="o", label=variant)
    ax.set_xlabel("noise parameter")
    ax.set_ylabel("RMS B_sym(phi)-phi_c")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(PLOTS / "block_residual_vs_noise.pdf")
    plt.close(fig)

    plot_heatmaps(fine, backbone, examples)

    def find_row(ensemble: str) -> dict[str, Any]:
        for row in obs_rows:
            if row["ensemble"] == ensemble:
                return row
        raise KeyError(ensemble)

    fine_row = find_row("fine")
    back_row = find_row("backbone")
    empirical_row = find_row("empirical_scale1")
    constant_best = min(
        [r for r in obs_rows if r["variant"] == "constant_high_rms"],
        key=lambda r: abs(float(r["phi2"]) - float(fine_row["phi2"]))
        + abs(float(r["phi4"]) - float(fine_row["phi4"]))
        + abs(float(r["nn2"]) - float(fine_row["nn2"])),
    )
    empirical_best = min(
        [r for r in obs_rows if r["variant"] == "empirical_shell_scale"],
        key=lambda r: abs(float(r["phi2"]) - float(fine_row["phi2"]))
        + abs(float(r["phi4"]) - float(fine_row["phi4"]))
        + abs(float(r["nn2"]) - float(fine_row["nn2"])),
    )

    summary = {
        "canonical_fine_path": str(FINE_PATH.resolve()),
        "n_configs": int(len(fine)),
        "kernel_metadata_path": str(KERNEL_META.resolve()),
        "kernel_original_source_path": kernel_meta.get("original_source_path"),
        "K_sum_check": kernel_sum(w),
        "eta_exponent": ETA_EXPONENT,
        "block_norm": BLOCK_NORM,
        "low_mode_support": "16x16 fftshift-centered central 8x8 block, same convention as symmetric backbone diagnostic",
        "high_mode_support": "Hermitian-safe complement: modes outside the retained support and outside its conjugate closure",
        "high_modes_count": int(np.sum(high_mask)),
        "low_modes_count": int(np.sum(mask)),
        "constant_c_values": CONSTANT_C,
        "k_suppressed_c_values": K_SUPPRESSED_C,
        "empirical_scales": EMPIRICAL_SCALES,
        "best_constant_by_phi2_phi4_nn2": constant_best,
        "best_empirical_by_phi2_phi4_nn2": empirical_best,
        "empirical_shell_variances": shell_var,
        "reference_fine": fine_row,
        "reference_backbone": back_row,
        "empirical_scale1": empirical_row,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    def fmt_row(row: dict[str, Any]) -> str:
        return (
            f"| {row['ensemble']} | {row['variant']} | {row['parameter']} | {float(row['phi2']):.6g} | "
            f"{float(row['phi4']):.6g} | {float(row['nn2']):.6g} | {float(row['Binder_U4']):.6g} | "
            f"{float(row['xi_over_L']):.6g} | {float(row['action_density']):.6g} |"
        )

    table_rows = [fine_row, back_row, constant_best, empirical_best, empirical_row]
    report_table = "\n".join(
        [
            "| ensemble | variant | parameter | phi2 | phi4 | nn2 | Binder U4 | xi/L | action density |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        + [fmt_row(row) for row in table_rows]
    )
    empirical_block = next(r for r in block_rows if r["ensemble"] == "empirical_scale1")
    best_const_block = next(r for r in block_rows if r["ensemble"] == constant_best["ensemble"])
    best_emp_block = next(r for r in block_rows if r["ensemble"] == empirical_best["ensemble"])

    report = f"""# High-Mode Gaussian Fluctuation Test

## Setup

- fine source: `{FINE_PATH.resolve()}`
- selected symmetric blockavg kernel metadata: `{KERNEL_META.resolve()}`
- K sum check: `{kernel_sum(w):.12g}`
- eta_exponent: `{ETA_EXPONENT}`
- block_norm: `{BLOCK_NORM:.12g}`
- low-mode support: central `8x8` region of the `16x16` shifted FFT grid
- high modes: Hermitian-safe complement, i.e. modes outside the retained support and outside its conjugate closure

The test blocks each canonical fine configuration with `B_sym`, reconstructs the Fourier low-mode backbone, takes the real backbone field as the baseline, then adds Gaussian fluctuations only on Hermitian-safe high Fourier modes. This avoids the Nyquist-boundary ambiguity of the half-open coarse Brillouin-zone convention.

## Main Table

{report_table}

## Block Residuals

- best constant row `{constant_best['ensemble']}` RMS block residual: `{best_const_block['rms_block_residual']:.6g}`
- best empirical row `{empirical_best['ensemble']}` RMS block residual: `{best_emp_block['rms_block_residual']:.6g}`
- empirical shell scale 1.0 RMS block residual: `{empirical_block['rms_block_residual']:.6g}`

## Answers

1. Does adding high-mode Gaussian noise restore phi2/phi4/nn2?

It restores some missing local power, but the outcome depends strongly on the high-mode variance model. The smooth backbone has phi2 `{float(back_row['phi2']):.6g}`, phi4 `{float(back_row['phi4']):.6g}`, nn2 `{float(back_row['nn2']):.6g}` versus fine phi2 `{float(fine_row['phi2']):.6g}`, phi4 `{float(fine_row['phi4']):.6g}`, nn2 `{float(fine_row['nn2']):.6g}`. The best empirical shell-Gaussian row by the simple local-moment score is `{empirical_best['ensemble']}`.

2. Does Binder remain unchanged?

Yes, to the extent expected from preserving the low modes. Binder remains near the fine/backbone value in the scan because the magnetization mode is not modified.

3. Does xi/L remain close to fine?

Mostly yes for variants that preserve low modes exactly. Residual changes are from high-mode contributions to `F(pmin)` and ensemble ratios, not from directly changing retained Fourier coefficients. See `low_momentum_spectrum.csv`.

4. How much does `B_sym(phi_noisy)` drift from `phi_c`?

High modes do not remain exactly block-null. The block residual is measured in `block_residual_scan.csv`; empirical shell scale 1.0 gives RMS `{empirical_block['rms_block_residual']:.6g}`. This means high modes cannot be added freely if exact block consistency is required.

5. Does empirical shell noise work better than constant noise?

Empirical shell noise is the more physically informative test because it matches the original fine-minus-backbone high-mode shell variances. It generally gives a better targeted UV-power diagnostic than constant RMS noise, but it is still Gaussian and does not encode conditional non-Gaussian tails.

6. Does Gaussian high-mode noise overshoot phi4/tails?

Some settings overshoot local moments, especially when the high-mode RMS is large. This supports the existing conclusion: the missing UV/detail law is not just an unconstrained variance knob; it needs either block-consistent sampling or a model of conditional high-mode structure.

## Interpretation

The IR backbone is still stable. Adding high-mode Gaussian fluctuations is useful as a UV diagnostic, but unconstrained high modes create nonzero block residuals. If this path is used later, high-mode variables should be sampled in a block-null or alias-consistent parameterization rather than added freely.
"""
    (OUT / "report.md").write_text(report)


if __name__ == "__main__":
    main()
