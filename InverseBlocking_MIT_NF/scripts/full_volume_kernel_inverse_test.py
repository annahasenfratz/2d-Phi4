#!/usr/bin/env python3
"""Full-volume periodic K inverse sanity test without decimation."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
BASE = PROJECT / "outputs" / "inverse_blocking_step_by_step"
OUT = BASE / "full_volume_kernel_inverse_test"
KERNEL_META = PROJECT / "kernels" / "from_perfect_blocking_lam1p0" / "selected_kernel_metadata.json"

L = 16
ETA_EXPONENT = 0.25
B = 2
BLOCK_NORM = B ** (ETA_EXPONENT / 2.0)
NEAR_ZERO_TOL = 1.0e-12

SHELLS = {
    "w00": [(0, 0)],
    "w10": [(1, 0), (-1, 0), (0, 1), (0, -1)],
    "w11": [(1, 1), (1, -1), (-1, 1), (-1, -1)],
    "w20": [(2, 0), (-2, 0), (0, 2), (0, -2)],
    "w21": [
        (2, 1),
        (2, -1),
        (-2, 1),
        (-2, -1),
        (1, 2),
        (1, -2),
        (-1, 2),
        (-1, -2),
    ],
    "w22": [(2, 2), (2, -2), (-2, 2), (-2, -2)],
}


def kernel_array(weights: dict[str, float]) -> np.ndarray:
    arr = np.zeros((L, L), dtype=float)
    for shell, offsets in SHELLS.items():
        for dy, dx in offsets:
            arr[dy % L, dx % L] += weights[shell]
    return arr


def heatmap(data: np.ndarray, title: str, path: Path, cmap: str = "viridis") -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.4), constrained_layout=True)
    im = ax.imshow(data, origin="lower", cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("p_x index, fftshifted")
    ax.set_ylabel("p_y index, fftshifted")
    fig.colorbar(im, ax=ax)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def error_stats(recon_complex: np.ndarray, phi: np.ndarray) -> dict:
    recon = recon_complex.real
    diff = recon - phi
    rms = float(np.sqrt(np.mean(diff**2)))
    phi_rms = float(np.sqrt(np.mean(phi**2)))
    return {
        "max_abs_error": float(np.max(np.abs(diff))),
        "rms_error": rms,
        "relative_rms_error": rms / phi_rms if phi_rms > 0 else math.nan,
        "max_imaginary_part_after_inverse_fft": float(np.max(np.abs(recon_complex.imag))),
    }


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(f"Refusing to overwrite existing outputs in {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)

    meta = json.loads(KERNEL_META.read_text())
    weights = {k: float(meta["weights"][k]) for k in ["w00", "w10", "w11", "w20", "w21", "w22"]}
    phi = np.load(BASE / "input_fine_batch.npy")[0]
    karr = kernel_array(weights)
    ktilde = np.fft.fft2(karr)
    kt_abs_shift = np.abs(np.fft.fftshift(ktilde))
    p = np.fft.fftshift(np.fft.fftfreq(L)) * 2.0 * np.pi
    px, py = np.meshgrid(p, p)
    min_idx = np.unravel_index(np.argmin(kt_abs_shift), kt_abs_shift.shape)

    phi_tilde = np.fft.fft2(phi)
    psi_tilde = ktilde * phi_tilde
    psi_complex = np.fft.ifft2(psi_tilde)
    recon_tilde = psi_tilde / ktilde
    recon_complex = np.fft.ifft2(recon_tilde)

    psiB_tilde = BLOCK_NORM * ktilde * phi_tilde
    psiB_complex = np.fft.ifft2(psiB_tilde)
    reconB_tilde = psiB_tilde / (BLOCK_NORM * ktilde)
    reconB_complex = np.fft.ifft2(reconB_tilde)

    np.save(OUT / "original_phi.npy", phi)
    np.save(OUT / "blurred_K_phi.npy", psi_complex.real)
    np.save(OUT / "reconstructed_phi.npy", recon_complex.real)
    heatmap(kt_abs_shift, "|K_tilde|", OUT / "K_tilde_abs.png")
    heatmap(recon_complex.real - phi, "K-only reconstructed - original", OUT / "difference_heatmap.png", cmap="coolwarm")

    summary = {
        "input_config_index": 0,
        "kernel_metadata_file": str(KERNEL_META.resolve()),
        "eta_exponent": ETA_EXPONENT,
        "b": B,
        "block_norm": BLOCK_NORM,
        "kernel_sum": float(np.sum(karr)),
        "near_zero_tolerance": NEAR_ZERO_TOL,
        "min_abs_K_tilde": float(kt_abs_shift[min_idx]),
        "min_abs_K_tilde_shifted_index": [int(min_idx[0]), int(min_idx[1])],
        "min_abs_K_tilde_momentum": [float(py[min_idx]), float(px[min_idx])],
        "any_abs_K_tilde_near_zero": bool(np.any(np.abs(ktilde) < NEAR_ZERO_TOL)),
        "K_only": {
            **error_stats(recon_complex, phi),
            "max_imaginary_part_blurred_K_phi": float(np.max(np.abs(psi_complex.imag))),
        },
        "full_B": {
            **error_stats(reconB_complex, phi),
            "max_imaginary_part_blurred_B_phi": float(np.max(np.abs(psiB_complex.imag))),
            "max_abs_blurred_B_minus_block_norm_blurred_K": float(
                np.max(np.abs(psiB_complex.real - BLOCK_NORM * psi_complex.real))
            ),
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (OUT / "report.md").write_text(
        "# Full-Volume Kernel Inverse Test\n\n"
        "This test applies the 5x5 kernel as a full 16x16 periodic convolution and inverts it in Fourier space. "
        "It does not decimate, zero-pad, alias-tile, or fill missing sites.\n\n"
        "## Kernel\n\n"
        f"- kernel metadata: `{summary['kernel_metadata_file']}`\n"
        f"- sum K: {summary['kernel_sum']:.15g}\n"
        f"- eta_exponent: {ETA_EXPONENT}\n"
        f"- b: {B}\n"
        f"- block_norm = 2**0.125: {BLOCK_NORM:.15g}\n"
        f"- min |K_tilde|: {summary['min_abs_K_tilde']:.12g} at p={summary['min_abs_K_tilde_momentum']}\n"
        f"- any |K_tilde| near zero below {NEAR_ZERO_TOL:g}: {summary['any_abs_K_tilde_near_zero']}\n\n"
        "## Reconstruction Errors\n\n"
        "| test | max abs error | RMS error | relative RMS error | max inverse-FFT imaginary part |\n"
        "|---|---:|---:|---:|---:|\n"
        f"| K-only | {summary['K_only']['max_abs_error']:.12g} | {summary['K_only']['rms_error']:.12g} | "
        f"{summary['K_only']['relative_rms_error']:.12g} | {summary['K_only']['max_imaginary_part_after_inverse_fft']:.12g} |\n"
        f"| full B | {summary['full_B']['max_abs_error']:.12g} | {summary['full_B']['rms_error']:.12g} | "
        f"{summary['full_B']['relative_rms_error']:.12g} | {summary['full_B']['max_imaginary_part_after_inverse_fft']:.12g} |\n\n"
        "Passing this test only proves the full periodic convolution operator is invertible on the 16x16 grid. "
        "It does not prove the decimated blocking map is invertible, because decimation discards and aliases information.\n"
    )


if __name__ == "__main__":
    main()
