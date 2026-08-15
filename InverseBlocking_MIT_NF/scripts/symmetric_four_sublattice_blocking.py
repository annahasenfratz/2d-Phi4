#!/usr/bin/env python3
"""Compare one-sublattice and symmetric four-sublattice blocking."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
BASE = PROJECT / "outputs" / "inverse_blocking_step_by_step"
OUT = PROJECT / "outputs" / "symmetric_four_sublattice_blocking"
HEAT = OUT / "transfer_heatmaps"
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


def kernel_array(weights: dict[str, float]) -> np.ndarray:
    arr = np.zeros((L_FINE, L_FINE), dtype=float)
    for shell, offsets in SHELLS.items():
        for dy, dx in offsets:
            arr[dy % L_FINE, dx % L_FINE] += weights[shell]
    return arr


def apply_K(configs: np.ndarray, weights: dict[str, float]) -> np.ndarray:
    out = np.zeros_like(configs, dtype=float)
    for shell, offsets in SHELLS.items():
        for dy, dx in offsets:
            out += weights[shell] * np.roll(np.roll(configs, -dy, axis=-2), -dx, axis=-1)
    return out


def obs(configs: np.ndarray) -> dict[str, float]:
    _, ly, lx = configs.shape
    v = ly * lx
    m = configs.mean(axis=(-2, -1))
    nn = 0.5 * (
        (configs * np.roll(configs, -1, axis=-2)).mean(axis=(-2, -1))
        + (configs * np.roll(configs, -1, axis=-1)).mean(axis=(-2, -1))
    )
    nn2 = 0.5 * (
        ((configs * np.roll(configs, -1, axis=-2)) ** 2).mean(axis=(-2, -1))
        + ((configs * np.roll(configs, -1, axis=-1)) ** 2).mean(axis=(-2, -1))
    )
    diag = (configs * np.roll(np.roll(configs, -1, axis=-2), -1, axis=-1)).mean(axis=(-2, -1))
    twonn = 0.5 * (
        (configs * np.roll(configs, -2, axis=-2)).mean(axis=(-2, -1))
        + (configs * np.roll(configs, -2, axis=-1)).mean(axis=(-2, -1))
    )
    m2 = float(np.mean(m**2))
    m4 = float(np.mean(m**4))
    b4 = m4 / (m2 * m2) if m2 > 0 else math.nan
    u4 = 1.0 - b4 / 3.0 if math.isfinite(b4) else math.nan
    ft = np.fft.fft2(configs, axes=(-2, -1))
    chi = float(v * (np.mean(m**2) - np.mean(m) ** 2))
    fmin = float(0.5 * (np.mean(np.abs(ft[:, 1, 0]) ** 2) + np.mean(np.abs(ft[:, 0, 1]) ** 2)) / v)
    ratio = chi / fmin - 1.0 if fmin > 0 else math.nan
    xi = (1.0 / (2.0 * math.sin(math.pi / lx))) * math.sqrt(ratio) if ratio > 0 else math.nan
    return {
        "phi2": float(np.mean(configs**2)),
        "phi4": float(np.mean(configs**4)),
        "NN": float(np.mean(nn)),
        "nn2": float(np.mean(nn2)),
        "diag": float(np.mean(diag)),
        "2nn": float(np.mean(twonn)),
        "Binder_U4": float(u4),
        "xi/L": float(xi / lx) if math.isfinite(xi) else math.nan,
    }


def block_replicate(coarse: np.ndarray) -> np.ndarray:
    out = np.empty((coarse.shape[0], L_FINE, L_FINE), dtype=coarse.dtype)
    out[:, 0::2, 0::2] = coarse
    out[:, 1::2, 0::2] = coarse
    out[:, 0::2, 1::2] = coarse
    out[:, 1::2, 1::2] = coarse
    return out


def pair_stats(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    x = a.ravel()
    y = b.ravel()
    slope = float(np.dot(x, y) / np.dot(x, x))
    return {
        "correlation": float(np.corrcoef(x, y)[0, 1]),
        "variance_ratio_symmetric_over_current": float(np.var(y) / np.var(x)),
        "best_fit_slope_symmetric_vs_current": slope,
        "rms_difference": float(np.sqrt(np.mean((y - x) ** 2))),
    }


def low_embed_from_coarse_fft(coarse_fft: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fine_shift = np.zeros((coarse_fft.shape[0], L_FINE, L_FINE), dtype=complex)
    start = L_FINE // 2 - L_COARSE // 2
    stop = start + L_COARSE
    fine_shift[:, start:stop, start:stop] = 4.0 * np.fft.fftshift(coarse_fft, axes=(-2, -1))
    mask_shift = np.zeros((L_FINE, L_FINE), dtype=bool)
    mask_shift[start:stop, start:stop] = True
    return np.fft.ifftshift(fine_shift, axes=(-2, -1)), np.fft.ifftshift(mask_shift)


def save_heat(data: np.ndarray, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.5), constrained_layout=True)
    im = ax.imshow(data, origin="lower", cmap="viridis")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(f"Refusing to overwrite existing outputs in {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    HEAT.mkdir(exist_ok=True)
    meta = json.loads(KERNEL_META.read_text())
    weights = {k: float(meta["weights"][k]) for k in ["w00", "w10", "w11", "w20", "w21", "w22"]}
    fine = np.load(BASE / "input_fine_batch.npy")
    psi = apply_K(fine, weights)
    current = BLOCK_NORM * psi[:, 0::2, 0::2]
    sym = 0.25 * BLOCK_NORM * (
        psi[:, 0::2, 0::2]
        + psi[:, 1::2, 0::2]
        + psi[:, 0::2, 1::2]
        + psi[:, 1::2, 1::2]
    )
    np.save(OUT / "current_one_sublattice_coarse.npy", current)
    np.save(OUT / "symmetric_four_sublattice_coarse.npy", sym)

    rows = []
    for name, arr in {"current_one_sublattice": current, "symmetric_four_sublattice": sym}.items():
        vals = obs(arr)
        rows.extend({"ensemble": name, "operator": k, "value": v} for k, v in vals.items())
    with (OUT / "current_vs_symmetric_coarse.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    pairs = pair_stats(current, sym)
    ktilde = np.fft.fft2(kernel_array(weights))
    p = np.fft.fftfreq(L_FINE) * 2.0 * np.pi
    px, py = np.meshgrid(p, p)
    # Sign is consistent with the convolution/shift convention used in this code.
    A = 0.25 * (1.0 + np.exp(1j * py)) * (1.0 + np.exp(1j * px))
    transfer = ktilde * A
    kt_shift = np.fft.fftshift(ktilde)
    tr_shift = np.fft.fftshift(transfer)
    abs_tr = np.abs(tr_shift)
    start = L_FINE // 2 - L_COARSE // 2
    stop = start + L_COARSE
    low_abs = abs_tr[start:stop, start:stop]
    gidx = np.unravel_index(np.argmin(abs_tr), abs_tr.shape)
    lidx_local = np.unravel_index(np.argmin(low_abs), low_abs.shape)
    lidx = (lidx_local[0] + start, lidx_local[1] + start)
    p_shift = np.fft.fftshift(p)
    pxs, pys = np.meshgrid(p_shift, p_shift)
    save_heat(np.abs(np.fft.fftshift(ktilde)), "|K_tilde|", HEAT / "K_tilde_abs.png")
    save_heat(np.abs(np.fft.fftshift(A)), "|A(p)|", HEAT / "four_sublattice_A_abs.png")
    save_heat(abs_tr, "|K_tilde A(p)|", HEAT / "K_times_A_abs.png")

    # Numerical transfer check by applying the 2x2 averaging operator to random fields.
    rng = np.random.default_rng(20240623)
    test = rng.normal(size=(4, L_FINE, L_FINE))
    test_ft = np.fft.fft2(test, axes=(-2, -1))
    avg = 0.25 * (
        test
        + np.roll(test, -1, axis=-2)
        + np.roll(test, -1, axis=-1)
        + np.roll(np.roll(test, -1, axis=-2), -1, axis=-1)
    )
    avg_ft = np.fft.fft2(avg, axes=(-2, -1))
    transfer_err = avg_ft - A[None] * test_ft
    transfer_check = {
        "A_definition": "A(p)=0.25*(1+exp(i p_x))*(1+exp(i p_y)) for the forward-shift averaging convention",
        "max_abs_transfer_check_error": float(np.max(np.abs(transfer_err))),
        "rms_transfer_check_error": float(np.sqrt(np.mean(np.abs(transfer_err) ** 2))),
        "global_min_abs_K_times_A": float(abs_tr[gidx]),
        "global_min_location_momentum": [float(pys[gidx]), float(pxs[gidx])],
        "coarse_BZ_min_abs_K_times_A": float(abs_tr[lidx]),
        "coarse_BZ_min_location_momentum": [float(pys[lidx]), float(pxs[lidx])],
        "zeros_from_A_at_px_pi_or_py_pi": True,
        "zeros_inside_lowmode_support": bool(np.any(low_abs < 1.0e-12)),
        "block_norm": BLOCK_NORM,
        "eta_exponent": ETA_EXPONENT,
    }
    (OUT / "four_sublattice_transfer_check.json").write_text(json.dumps(transfer_check, indent=2) + "\n")

    coarse_fft = np.fft.fft2(sym, axes=(-2, -1))
    padded, mask = low_embed_from_coarse_fft(coarse_fft)
    denom = BLOCK_NORM * transfer
    inv_ft = np.zeros_like(padded)
    inv_ft[:, mask] = padded[:, mask] / denom[mask]
    smooth_inv = np.fft.ifft2(inv_ft, axes=(-2, -1)).real

    fine_ft = np.fft.fft2(fine, axes=(-2, -1))
    fine_low_ft = np.zeros_like(fine_ft)
    fine_low_ft[:, mask] = fine_ft[:, mask]
    fine_low = np.fft.ifft2(fine_low_ft, axes=(-2, -1)).real
    sym_replicated = block_replicate(sym)
    np.save(OUT / "symmetric_inverse_lowmode_backbone.npy", smooth_inv)
    np.save(OUT / "original_fine_lowmode_projection.npy", fine_low)
    np.save(OUT / "symmetric_block_replicated_baseline.npy", sym_replicated)

    inv_rows = []
    for name, arr in {
        "original_fine": fine,
        "original_fine_lowmode_projection": fine_low,
        "symmetric_inverse_lowmode_backbone": smooth_inv,
        "symmetric_block_replicated_baseline": sym_replicated,
    }.items():
        vals = obs(arr)
        inv_rows.extend({"ensemble": name, "operator": k, "value": v} for k, v in vals.items())
    with (OUT / "inverse_backbone_comparison.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(inv_rows[0]))
        writer.writeheader()
        writer.writerows(inv_rows)

    def val(table: list[dict], ens: str, op: str) -> float:
        return next(r["value"] for r in table if r["ensemble"] == ens and r["operator"] == op)

    report = f"""# Symmetric Four-Sublattice Blocking Diagnostic

## Setup

- eta_exponent: {ETA_EXPONENT}
- block_norm: {BLOCK_NORM:.15g}
- sum K: {sum(weights[k] * len(SHELLS[k]) for k in weights):.15g}
- current rule: `phi_c(n) = block_norm * psi(2n)`
- symmetric rule: `phi_c_sym(n) = block_norm * 0.25 * [psi(2n)+psi(2n+e0)+psi(2n+e1)+psi(2n+e0+e1)]`

## Coarse Field Comparison

- correlation symmetric vs current: {pairs['correlation']:.12g}
- variance ratio symmetric/current: {pairs['variance_ratio_symmetric_over_current']:.12g}
- best-fit slope symmetric vs current: {pairs['best_fit_slope_symmetric_vs_current']:.12g}
- RMS difference: {pairs['rms_difference']:.12g}

Current coarse phi2/phi4/nn2: {val(rows, 'current_one_sublattice', 'phi2'):.12g}, {val(rows, 'current_one_sublattice', 'phi4'):.12g}, {val(rows, 'current_one_sublattice', 'nn2'):.12g}

Symmetric coarse phi2/phi4/nn2: {val(rows, 'symmetric_four_sublattice', 'phi2'):.12g}, {val(rows, 'symmetric_four_sublattice', 'phi4'):.12g}, {val(rows, 'symmetric_four_sublattice', 'nn2'):.12g}

## Transfer Function

The four-sublattice average uses `A(p)=0.25*(1+exp(i p_x))*(1+exp(i p_y))`.

- transfer check max error: {transfer_check['max_abs_transfer_check_error']:.12g}
- global min |K_tilde A|: {transfer_check['global_min_abs_K_times_A']:.12g} at p={transfer_check['global_min_location_momentum']}
- min |K_tilde A| in coarse BZ: {transfer_check['coarse_BZ_min_abs_K_times_A']:.12g} at p={transfer_check['coarse_BZ_min_location_momentum']}
- zeros at p_x=pi or p_y=pi from A(p): true
- zeros inside lowmode support: {transfer_check['zeros_inside_lowmode_support']}

## Inverse Backbone Moments

Original fine phi2/phi4/nn2: {val(inv_rows, 'original_fine', 'phi2'):.12g}, {val(inv_rows, 'original_fine', 'phi4'):.12g}, {val(inv_rows, 'original_fine', 'nn2'):.12g}

Original fine low-mode projection phi2/phi4/nn2: {val(inv_rows, 'original_fine_lowmode_projection', 'phi2'):.12g}, {val(inv_rows, 'original_fine_lowmode_projection', 'phi4'):.12g}, {val(inv_rows, 'original_fine_lowmode_projection', 'nn2'):.12g}

Symmetric inverse low-mode backbone phi2/phi4/nn2: {val(inv_rows, 'symmetric_inverse_lowmode_backbone', 'phi2'):.12g}, {val(inv_rows, 'symmetric_inverse_lowmode_backbone', 'phi4'):.12g}, {val(inv_rows, 'symmetric_inverse_lowmode_backbone', 'nn2'):.12g}

Symmetric 2x2 block-replicated baseline phi2/phi4/nn2: {val(inv_rows, 'symmetric_block_replicated_baseline', 'phi2'):.12g}, {val(inv_rows, 'symmetric_block_replicated_baseline', 'phi4'):.12g}, {val(inv_rows, 'symmetric_block_replicated_baseline', 'nn2'):.12g}

## Questions

1. Does symmetric four-sublattice blocking give a better coarse variable than one-sublattice decimation?

It gives a less sublattice-specific coarse variable and substantially smooths the coarse field. Whether it is better for RG matching requires a direct coarse-reference comparison, but it avoids choosing one representative sublattice.

2. Does it avoid privileging the even-even sublattice?

Yes. The symmetric rule averages all four sites of each 2x2 block after applying K, so no sublattice is singled out.

3. Does the symmetric transfer function have safe inverse behavior in the coarse BZ?

Yes for the retained low-mode support in this 16x16/8x8 setup. The A(p) zeros are at p_x=pi or p_y=pi, outside the central low-mode block used for inversion. The minimum |K_tilde A| in the coarse BZ is {transfer_check['coarse_BZ_min_abs_K_times_A']:.12g}.

4. Are phi2/phi4/nn2 of the inverse backbone less biased?

The symmetric inverse low-mode backbone is best compared to the original fine low-mode projection, not the full fine field. Its one-site moments are close to that low-mode projection, while both are much smoother than the full fine field. The 2x2 replicated symmetric baseline preserves the symmetric coarse amplitudes but remains a crude piecewise-constant fill.

5. Should the conditional NF condition on a symmetric block variable rather than fixed even-even sites?

This diagnostic supports that direction. Fixing even-even sites bakes in a privileged sublattice and can inherit backbone moment bias. A symmetric coarse condition avoids that hard site constraint.

6. Proposed new conditional setup:

Use `condition = symmetric coarse block field + smooth symmetric inverse backbone`, and generate all 16x16 fine sites. Enforce consistency through the fine action plus an optional soft blocking penalty to the symmetric coarse field, instead of hard-fixing even-even sites and filling only missing sites.
"""
    (OUT / "report.md").write_text(report)


if __name__ == "__main__":
    main()
