#!/usr/bin/env python3
"""Alias-sector decomposition diagnostic for decimated inverse blocking."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
BASE = PROJECT / "outputs" / "inverse_blocking_step_by_step"
OUT = BASE / "alias_sector_decomposition"
KERNEL_META = PROJECT / "kernels" / "from_perfect_blocking_lam1p0" / "selected_kernel_metadata.json"

L_FINE = 16
L_COARSE = 8
ETA_EXPONENT = 0.25
B = 2
BLOCK_NORM = B ** (ETA_EXPONENT / 2.0)

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

SECTORS = {
    "P00": (0, 0),
    "P10": (L_COARSE, 0),
    "P01": (0, L_COARSE),
    "P11": (L_COARSE, L_COARSE),
}


def fine_low_index(k: int) -> int:
    return k if k < L_COARSE // 2 else k + L_COARSE


def sector_indices() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    ky = np.array([fine_low_index(k) for k in range(L_COARSE)])
    kx = np.array([fine_low_index(k) for k in range(L_COARSE)])
    yy, xx = np.meshgrid(ky, kx, indexing="ij")
    out = {}
    for name, (dy, dx) in SECTORS.items():
        out[name] = ((yy + dy) % L_FINE, (xx + dx) % L_FINE)
    return out


def kernel_array(weights: dict[str, float]) -> np.ndarray:
    arr = np.zeros((L_FINE, L_FINE), dtype=float)
    for shell, offsets in SHELLS.items():
        for dy, dx in offsets:
            arr[dy % L_FINE, dx % L_FINE] += weights[shell]
    return arr


def obs(configs: np.ndarray) -> dict[str, float]:
    m_cfg = configs.mean(axis=(-2, -1))
    nn = 0.5 * (
        np.mean(configs * np.roll(configs, -1, axis=-2), axis=(-2, -1))
        + np.mean(configs * np.roll(configs, -1, axis=-1), axis=(-2, -1))
    )
    nn2 = 0.5 * (
        np.mean((configs * np.roll(configs, -1, axis=-2)) ** 2, axis=(-2, -1))
        + np.mean((configs * np.roll(configs, -1, axis=-1)) ** 2, axis=(-2, -1))
    )
    diag = np.mean(
        configs * np.roll(np.roll(configs, -1, axis=-2), -1, axis=-1), axis=(-2, -1)
    )
    twonn = 0.5 * (
        np.mean(configs * np.roll(configs, -2, axis=-2), axis=(-2, -1))
        + np.mean(configs * np.roll(configs, -2, axis=-1), axis=(-2, -1))
    )
    return {
        "m": float(np.mean(m_cfg)),
        "|m|": float(np.mean(np.abs(m_cfg))),
        "phi2": float(np.mean(configs**2)),
        "phi4": float(np.mean(configs**4)),
        "NN": float(np.mean(nn)),
        "nn2": float(np.mean(nn2)),
        "diag": float(np.mean(diag)),
        "2nn": float(np.mean(twonn)),
    }


def compact_even_even_obs(configs: np.ndarray) -> dict[str, float]:
    a = configs[:, 0::2, 0::2]
    nn2 = 0.5 * (
        np.mean((a * np.roll(a, -1, axis=-2)) ** 2)
        + np.mean((a * np.roll(a, -1, axis=-1)) ** 2)
    )
    return {
        "compact_even_even_phi2": float(np.mean(a**2)),
        "compact_even_even_phi4": float(np.mean(a**4)),
        "compact_even_even_nn2": float(nn2),
    }


def sector_power_rows(label: str, ft: np.ndarray, idx: dict[str, tuple[np.ndarray, np.ndarray]]) -> list[dict]:
    powers = {}
    for name, (yy, xx) in idx.items():
        powers[name] = float(np.mean(np.abs(ft[:, yy, xx]) ** 2))
    total = sum(powers.values())
    row = {
        "field": label,
        **powers,
        "Ptotal": total,
        "frac_P00": powers["P00"] / total,
        "frac_P10": powers["P10"] / total,
        "frac_P01": powers["P01"] / total,
        "frac_P11": powers["P11"] / total,
    }
    return [row]


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def reconstruct_variants(
    phi: np.ndarray,
    psi_tilde: np.ndarray,
    coarse_fft: np.ndarray,
    ktilde: np.ndarray,
    idx: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, np.ndarray]:
    n = phi.shape[0]
    low_fft = np.zeros((n, L_FINE, L_FINE), dtype=complex)
    oracle_fft = np.zeros_like(low_fft)
    equal_fft = np.zeros_like(low_fft)
    current_fft = np.zeros_like(low_fft)

    yy_low, xx_low = idx["P00"]
    low_fft[:, yy_low, xx_low] = psi_tilde[:, yy_low, xx_low] / (BLOCK_NORM * ktilde[yy_low, xx_low])

    for name, (yy, xx) in idx.items():
        oracle_fft[:, yy, xx] = psi_tilde[:, yy, xx] / (BLOCK_NORM * ktilde[yy, xx])
        # FFT_8[psi_even_even] = (1/4) sum_alias psi_tilde, so equal sector assignment sets each psi sector to coarse_fft.
        equal_fft[:, yy, xx] = coarse_fft / (BLOCK_NORM * ktilde[yy, xx])
        # Current inverse alias method divides by the low-sector kernel, then tiles the same phi coefficient into every alias sector.
        current_fft[:, yy, xx] = coarse_fft / (BLOCK_NORM * ktilde[yy_low, xx_low])

    return {
        "A_low_sector_only_from_original_psi": np.fft.ifft2(low_fft, axes=(-2, -1)).real,
        "B_all_sector_oracle_from_original_psi": np.fft.ifft2(oracle_fft, axes=(-2, -1)).real,
        "C_decimated_equal_psi_sector_distribution": np.fft.ifft2(equal_fft, axes=(-2, -1)).real,
        "D_current_inverse_alias_method": np.fft.ifft2(current_fft, axes=(-2, -1)).real,
    }


def reconstruction_rows(phi: np.ndarray, variants: dict[str, np.ndarray]) -> list[dict]:
    rows = []
    phi_rms = float(np.sqrt(np.mean(phi**2)))
    for name, arr in variants.items():
        diff = arr - phi
        row = {
            "variant": name,
            "max_abs_error": float(np.max(np.abs(diff))),
            "RMS_error": float(np.sqrt(np.mean(diff**2))),
            "relative_RMS_error": float(np.sqrt(np.mean(diff**2)) / phi_rms),
            **obs(arr),
            **compact_even_even_obs(arr),
        }
        rows.append(row)
    return rows


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(f"Refusing to overwrite existing outputs in {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)

    meta = json.loads(KERNEL_META.read_text())
    weights = {k: float(meta["weights"][k]) for k in ["w00", "w10", "w11", "w20", "w21", "w22"]}
    phi = np.load(BASE / "input_fine_batch.npy")
    ktilde = np.fft.fft2(kernel_array(weights))
    phi_tilde = np.fft.fft2(phi, axes=(-2, -1))
    psi_tilde = BLOCK_NORM * ktilde[None, :, :] * phi_tilde
    psi = np.fft.ifft2(psi_tilde, axes=(-2, -1)).real
    coarse = psi[:, 0::2, 0::2]
    coarse_fft = np.fft.fft2(coarse, axes=(-2, -1))

    idx = sector_indices()
    phi_power_rows = sector_power_rows("phi_before_K", phi_tilde, idx)
    psi_power_rows = sector_power_rows("psi_after_block_norm_K", psi_tilde, idx)
    write_csv(OUT / "sector_power_phi.csv", phi_power_rows)
    write_csv(OUT / "sector_power_after_K.csv", psi_power_rows)

    alias_sum = np.zeros_like(coarse_fft)
    for yy, xx in idx.values():
        alias_sum += psi_tilde[:, yy, xx]
    folded = 0.25 * alias_sum
    identity_err = folded - coarse_fft
    identity = {
        "numpy_fft_identity": "FFT_8[psi[0::2,0::2]] = 0.25 * sum_four_alias_sectors FFT_16[psi]",
        "max_abs_error": float(np.max(np.abs(identity_err))),
        "rms_error": float(np.sqrt(np.mean(np.abs(identity_err) ** 2))),
        "relative_rms_error": float(
            np.sqrt(np.mean(np.abs(identity_err) ** 2)) / np.sqrt(np.mean(np.abs(coarse_fft) ** 2))
        ),
        "normalization_factor_alias_sum_to_coarse_fft": 0.25,
        "block_norm": BLOCK_NORM,
        "eta_exponent": ETA_EXPONENT,
    }
    (OUT / "decimation_alias_identity_check.json").write_text(json.dumps(identity, indent=2) + "\n")

    variants = reconstruct_variants(phi, psi_tilde, coarse_fft, ktilde, idx)
    rows = reconstruction_rows(phi, variants)
    write_csv(OUT / "reconstruction_variant_comparison.csv", rows)

    original_compact = compact_even_even_obs(phi)
    row_by_name = {r["variant"]: r for r in rows}
    report = f"""# Alias-Sector Decomposition

This diagnostic decomposes the full-volume blocked field `psi = block_norm * K * phi` into the four fine-momentum alias sectors that fold together under even-even decimation. It does not train a normalizing flow.

## Decimation Identity

For numpy FFT conventions, the verified identity is:

`FFT_8[psi[0::2,0::2]] = 0.25 * sum_four_alias_sectors FFT_16[psi]`

- max absolute identity error: {identity['max_abs_error']:.12g}
- RMS identity error: {identity['rms_error']:.12g}
- relative RMS identity error: {identity['relative_rms_error']:.12g}

## Sector Power Fractions

Before K:

- low sector P00 fraction: {phi_power_rows[0]['frac_P00']:.12g}
- P10 fraction: {phi_power_rows[0]['frac_P10']:.12g}
- P01 fraction: {phi_power_rows[0]['frac_P01']:.12g}
- P11 fraction: {phi_power_rows[0]['frac_P11']:.12g}

After `block_norm * K`:

- low sector P00 fraction: {psi_power_rows[0]['frac_P00']:.12g}
- P10 fraction: {psi_power_rows[0]['frac_P10']:.12g}
- P01 fraction: {psi_power_rows[0]['frac_P01']:.12g}
- P11 fraction: {psi_power_rows[0]['frac_P11']:.12g}

## Reconstruction Variants

| variant | RMS error | relative RMS | phi2 | phi4 | NN | nn2 | compact ee phi2 | compact ee phi4 | compact ee nn2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
"""
    for row in rows:
        report += (
            f"| {row['variant']} | {row['RMS_error']:.12g} | {row['relative_RMS_error']:.12g} | "
            f"{row['phi2']:.12g} | {row['phi4']:.12g} | {row['NN']:.12g} | {row['nn2']:.12g} | "
            f"{row['compact_even_even_phi2']:.12g} | {row['compact_even_even_phi4']:.12g} | {row['compact_even_even_nn2']:.12g} |\n"
        )

    low = row_by_name["A_low_sector_only_from_original_psi"]
    oracle = row_by_name["B_all_sector_oracle_from_original_psi"]
    equal = row_by_name["C_decimated_equal_psi_sector_distribution"]
    current = row_by_name["D_current_inverse_alias_method"]
    report += f"""
## Questions

1. What fraction of power is in the low sector before applying K?

The low-sector fraction before K is {phi_power_rows[0]['frac_P00']:.12g}.

2. What fraction of power remains in the low sector after applying K?

After `block_norm*K`, the low-sector fraction is {psi_power_rows[0]['frac_P00']:.12g}. K suppresses the high alias sectors only partially; the three high sectors still carry {1.0 - psi_power_rows[0]['frac_P00']:.12g} of the blocked-field Fourier power.

3. How much alias-sector information is irretrievably lost by decimation?

Decimation keeps only the folded sum of the four sectors for each coarse momentum. The individual sector amplitudes are lost. The all-sector oracle reconstructs with relative RMS error {oracle['relative_RMS_error']:.12g}, while assumptions based on reduced sector information have much larger errors: low-sector-only {low['relative_RMS_error']:.12g}, equal sector distribution {equal['relative_RMS_error']:.12g}, and current inverse alias {current['relative_RMS_error']:.12g}.

4. Does the current inverse alias field correspond to a particular equal-distribution or low-sector assumption?

Yes. It uses the decimated coarse Fourier mode and divides by the low-sector `K_tilde(q)`, then tiles that same inferred low-sector fine-field coefficient into all four alias sectors. That is neither the true all-sector inverse nor the equal blocked-psi-sector distribution, because equal blocked-psi-sector distribution would divide each assigned sector by its own `K_tilde(p)`.

5. Which sector loss explains the suppressed even-even phi4 and nn2?

The original compact even-even moments are phi4 {original_compact['compact_even_even_phi4']:.12g} and nn2 {original_compact['compact_even_even_nn2']:.12g}. The current inverse alias gives compact even-even phi4 {current['compact_even_even_phi4']:.12g} and nn2 {current['compact_even_even_nn2']:.12g}. This suppression comes from replacing the unknown sector decomposition by a low-sector/tiled assumption after decimation; the lost high-sector phases and amplitudes encode local UV structure that contributes strongly to phi4 and nn2.

6. What should the conditional NF learn: the discarded alias sectors, local residuals, or both?

Both. The decimated field constrains only folded sector sums, so the conditional model must infer discarded alias-sector content. In position space this appears as local residual UV structure: within-block variation, odd-sublattice fluctuations, and corrections to high-moment and squared-link operators.
"""
    (OUT / "report.md").write_text(report)


if __name__ == "__main__":
    main()
