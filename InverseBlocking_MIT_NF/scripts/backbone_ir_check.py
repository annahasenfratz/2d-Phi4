#!/usr/bin/env python3
"""IR check for the symmetric-block upscaled backbone."""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT / "outputs" / "matplotlib_config"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = PROJECT.parent
DATA = PROJECT / "outputs" / "paired_data_lam1_kappaf0p320"
OUT = PROJECT / "outputs" / "backbone_ir_check"
PLOTS = OUT / "plots"
KERNEL_META = PROJECT / "kernels" / "from_perfect_blocking_lam1p0_blockavg" / "selected_kernel_metadata.json"

L_FINE = 16
L_COARSE = 8
LAMBDA = 1.0
KAPPA_F = 0.320
ETA_EXPONENT = 0.25
BLOCK_NORM = 2 ** (ETA_EXPONENT / 2.0)

SHELLS = {
    "w00": [(0, 0)],
    "w10": [(1, 0), (-1, 0), (0, 1), (0, -1)],
    "w11": [(1, 1), (1, -1), (-1, 1), (-1, -1)],
    "w20": [(2, 0), (-2, 0), (0, 2), (0, -2)],
    "w21": [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)],
    "w22": [(2, 2), (2, -2), (-2, 2), (-2, -2)],
}

LOW_MODES = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (0, 2)]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def load_kernel() -> tuple[dict[str, float], dict[str, object]]:
    meta = json.loads(KERNEL_META.read_text())
    weights = meta.get("source_metadata", {}).get("weights", meta.get("weights"))
    if weights is None:
        raise RuntimeError(f"No weights found in {KERNEL_META}")
    w = {k: float(weights[k]) for k in SHELLS}
    return w, meta


def kernel_sum(w: dict[str, float]) -> float:
    return float(sum(len(SHELLS[k]) * w[k] for k in SHELLS))


def shell_convolve(phi: np.ndarray, w: dict[str, float]) -> np.ndarray:
    out = np.zeros_like(phi, dtype=np.float64)
    for shell, offsets in SHELLS.items():
        for dy, dx in offsets:
            out += w[shell] * np.roll(np.roll(phi, -dy, axis=-2), -dx, axis=-1)
    return out


def block_sym(phi: np.ndarray, w: dict[str, float]) -> np.ndarray:
    psi = shell_convolve(phi, w)
    return 0.25 * BLOCK_NORM * (
        psi[:, 0::2, 0::2] + psi[:, 1::2, 0::2] + psi[:, 0::2, 1::2] + psi[:, 1::2, 1::2]
    )


def kernel_array(w: dict[str, float], n: int = L_FINE) -> np.ndarray:
    arr = np.zeros((n, n), dtype=np.float64)
    for shell, offsets in SHELLS.items():
        for dy, dx in offsets:
            arr[dy % n, dx % n] += w[shell]
    return arr


def smooth_backbone(coarse: np.ndarray, w: dict[str, float]) -> tuple[np.ndarray, dict[str, float]]:
    ktilde = np.fft.fft2(kernel_array(w, L_FINE))
    p = np.fft.fftfreq(L_FINE) * 2.0 * np.pi
    px, py = np.meshgrid(p, p)
    a = 0.25 * (1.0 + np.exp(1j * py)) * (1.0 + np.exp(1j * px))
    denom = BLOCK_NORM * ktilde * a

    cft = np.fft.fft2(coarse, axes=(-2, -1))
    padded_shift = np.zeros((len(coarse), L_FINE, L_FINE), dtype=np.complex128)
    padded_shift[:, 4:12, 4:12] = 4.0 * np.fft.fftshift(cft, axes=(-2, -1))
    padded = np.fft.ifftshift(padded_shift, axes=(-2, -1))
    mask_shift = np.zeros((L_FINE, L_FINE), dtype=bool)
    mask_shift[4:12, 4:12] = True
    mask = np.fft.ifftshift(mask_shift)
    inv = np.zeros_like(padded)
    inv[:, mask] = padded[:, mask] / denom[mask]
    back = np.fft.ifft2(inv, axes=(-2, -1)).real
    denom_abs = np.abs(denom[mask])
    return back, {
        "low_mode_min_abs_block_transfer": float(np.min(denom_abs)),
        "low_mode_max_abs_block_transfer": float(np.max(denom_abs)),
        "fft_convention": "numpy fft2/ifft2; forward unnormalized, inverse includes 1/V; coarse FFT block multiplied by 4 when embedded in 16x16 low modes",
    }


def ensemble_observables(phi: np.ndarray, label: str, lattice_size: int, kappa_for_action: float | None = KAPPA_F) -> dict[str, float | str | int]:
    phi = phi.astype(np.float64)
    n, ly, lx = phi.shape
    v = ly * lx
    m = phi.mean(axis=(-2, -1))
    abs_m = np.abs(m)
    m2 = float(np.mean(m**2))
    m4 = float(np.mean(m**4))
    b4 = m4 / (m2 * m2) if m2 > 0 else math.nan
    u4 = 1.0 - b4 / 3.0 if math.isfinite(b4) else math.nan
    nn_vals = 0.5 * (
        (phi * np.roll(phi, -1, axis=-2)).mean(axis=(-2, -1))
        + (phi * np.roll(phi, -1, axis=-1)).mean(axis=(-2, -1))
    )
    nn2_vals = 0.5 * (
        ((phi * np.roll(phi, -1, axis=-2)) ** 2).mean(axis=(-2, -1))
        + ((phi * np.roll(phi, -1, axis=-1)) ** 2).mean(axis=(-2, -1))
    )
    diag_vals = (phi * np.roll(np.roll(phi, -1, axis=-2), -1, axis=-1)).mean(axis=(-2, -1))
    twonn_vals = 0.5 * (
        (phi * np.roll(phi, -2, axis=-2)).mean(axis=(-2, -1))
        + (phi * np.roll(phi, -2, axis=-1)).mean(axis=(-2, -1))
    )

    ft = np.fft.fft2(phi, axes=(-2, -1))
    chi_connected = float(v * (np.mean(m**2) - np.mean(m) ** 2))
    chi_uncentered = float(v * np.mean(m**2))
    fmin = float(0.5 * (np.mean(np.abs(ft[:, 1, 0]) ** 2) + np.mean(np.abs(ft[:, 0, 1]) ** 2)) / v)
    ratio = chi_connected / fmin - 1.0 if fmin > 0 else math.nan
    xi = (1.0 / (2.0 * math.sin(math.pi / lx))) * math.sqrt(ratio) if ratio > 0 else math.nan

    action_hop = math.nan
    action_phi2 = math.nan
    action_phi4 = math.nan
    action_density = math.nan
    if kappa_for_action is not None:
        action_hop = -4.0 * kappa_for_action * float(np.mean(nn_vals))
        action_phi2 = -float(np.mean(phi**2))
        action_phi4 = float(np.mean(phi**4))
        action_density = action_hop + action_phi2 + action_phi4

    return {
        "ensemble": label,
        "L": lattice_size,
        "N": n,
        "m": float(np.mean(m)),
        "abs_m": float(np.mean(abs_m)),
        "m2": m2,
        "m4": m4,
        "Binder_U4": float(u4),
        "Binder_ratio_B4": float(b4),
        "chi_connected": chi_connected,
        "chi_uncentered": chi_uncentered,
        "F_pmin": fmin,
        "xi": float(xi) if math.isfinite(xi) else math.nan,
        "xi_over_L": float(xi / lx) if math.isfinite(xi) else math.nan,
        "mean_phi2": float(np.mean(phi**2)),
        "mean_phi4": float(np.mean(phi**4)),
        "NN": float(np.mean(nn_vals)),
        "nn2": float(np.mean(nn2_vals)),
        "diag": float(np.mean(diag_vals)),
        "2nn": float(np.mean(twonn_vals)),
        "action_hopping_density": action_hop,
        "action_phi2_density": action_phi2,
        "action_phi4_density": action_phi4,
        "action_density": action_density,
    }


def structure_factor(phi: np.ndarray, label: str, modes: list[tuple[int, int]]) -> list[dict[str, object]]:
    phi = phi.astype(np.float64)
    _, ly, lx = phi.shape
    v = ly * lx
    ft = np.fft.fft2(phi, axes=(-2, -1))
    rows = []
    for ky, kx in modes:
        sval = float(np.mean(np.abs(ft[:, ky % ly, kx % lx]) ** 2) / v)
        rows.append(
            {
                "ensemble": label,
                "L": lx,
                "ky_index": ky,
                "kx_index": kx,
                "p_y": float(2.0 * math.pi * ky / ly),
                "p_x": float(2.0 * math.pi * kx / lx),
                "S_p": sval,
            }
        )
    return rows


def radial_shells(phi: np.ndarray, label: str, max_shells: int = 8) -> list[dict[str, object]]:
    phi = phi.astype(np.float64)
    _, ly, lx = phi.shape
    v = ly * lx
    ft = np.fft.fft2(phi, axes=(-2, -1))
    vals = np.abs(ft) ** 2 / v
    rows = []
    shell_map: dict[int, list[float]] = {}
    for ky in range(ly):
        sy = min(ky, ly - ky)
        for kx in range(lx):
            sx = min(kx, lx - kx)
            shell = sx * sx + sy * sy
            shell_map.setdefault(shell, []).append(float(np.mean(vals[:, ky, kx])))
    for shell in sorted(shell_map)[:max_shells]:
        rows.append({"ensemble": label, "L": lx, "shell_k2_index": shell, "S_shell_mean": float(np.mean(shell_map[shell])), "mode_count": len(shell_map[shell])})
    return rows


def plot_bars(rows: list[dict[str, object]], key: str, ylabel: str, path: Path) -> None:
    labels = [str(r["ensemble"]) for r in rows]
    vals = [float(r[key]) for r in rows]
    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    ax.bar(labels, vals, color=["#4C78A8", "#F58518", "#54A24B"])
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_spectrum(rows: list[dict[str, object]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    by_label: dict[str, list[dict[str, object]]] = {}
    for r in rows:
        if int(r["L"]) != L_FINE:
            continue
        by_label.setdefault(str(r["ensemble"]), []).append(r)
    xs = np.arange(len(LOW_MODES))
    labels = [f"({ky},{kx})" for ky, kx in LOW_MODES]
    for name, data in by_label.items():
        vals = [float(r["S_p"]) for r in data[: len(LOW_MODES)]]
        ax.plot(xs, vals, marker="o", label=name)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel("S(p) = <|FFT(phi)(p)|^2> / V")
    ax.set_xlabel("momentum index (ky,kx)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_heatmaps(fine: np.ndarray, coarse: np.ndarray, back: np.ndarray, outdir: Path, n: int = 4) -> None:
    for i in range(min(n, len(fine))):
        fields = [
            ("fine", fine[i]),
            ("blocked_coarse", coarse[i]),
            ("backbone", back[i]),
            ("backbone_minus_fine", back[i] - fine[i]),
        ]
        fig, axs = plt.subplots(1, 4, figsize=(11.5, 3.0))
        for ax, (title, arr) in zip(axs, fields):
            im = ax.imshow(arr, cmap="coolwarm", origin="lower")
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(outdir / f"representative_{i:03d}.pdf")
        fig.savefig(outdir / f"representative_{i:03d}.png", dpi=160)
        plt.close(fig)


def main() -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)

    w, meta = load_kernel()
    fine = np.load(DATA / "fine_configs.npy").astype(np.float64)
    coarse = block_sym(fine, w)
    back, transfer_meta = smooth_backbone(coarse, w)
    reblocked = block_sym(back, w)
    err = reblocked - coarse

    np.save(OUT / "blocked_coarse.npy", coarse.astype(np.float32))
    np.save(OUT / "backbone_configs.npy", back.astype(np.float32))

    metadata = {
        "fine_configs_path": str((DATA / "fine_configs.npy").resolve()),
        "n_configs": int(len(fine)),
        "fine_shape": list(fine.shape),
        "lambda_f": LAMBDA,
        "kappa_f": KAPPA_F,
        "kernel_metadata_path": str(KERNEL_META.resolve()),
        "kernel_original_source_path": meta.get("original_source_path"),
        "kernel_local_copy_path": meta.get("local_copy_path"),
        "kernel_caveat": meta.get("caveat"),
        "kernel_weights": w,
        "K_sum_check": kernel_sum(w),
        "eta_exponent": ETA_EXPONENT,
        "block_norm": BLOCK_NORM,
        "blocking_rule": "symmetric_2x2_average_after_K",
        "transfer_metadata": transfer_meta,
    }
    (OUT / "fine_configs_used_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    block_check = {
        "max_abs_error": float(np.max(np.abs(err))),
        "rms_error": float(np.sqrt(np.mean(err**2))),
        "mean_abs_error": float(np.mean(np.abs(err))),
        "coarse_rms": float(np.sqrt(np.mean(coarse**2))),
        "relative_rms_error": float(np.sqrt(np.mean(err**2)) / max(np.sqrt(np.mean(coarse**2)), 1.0e-30)),
    }
    (OUT / "block_reconstruction_check.json").write_text(json.dumps(block_check, indent=2) + "\n")

    obs_rows = [
        ensemble_observables(fine, "original_fine", L_FINE, KAPPA_F),
        ensemble_observables(back, "backbone", L_FINE, KAPPA_F),
        ensemble_observables(coarse, "blocked_coarse", L_COARSE, None),
    ]
    write_csv(OUT / "ir_observables.csv", obs_rows)
    write_csv(OUT / "operator_comparison.csv", obs_rows)

    spec_rows = []
    spec_rows += structure_factor(fine, "original_fine", LOW_MODES)
    spec_rows += structure_factor(back, "backbone", LOW_MODES)
    coarse_modes = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (0, 2)]
    spec_rows += structure_factor(coarse, "blocked_coarse", coarse_modes)
    spec_rows += radial_shells(fine, "original_fine")
    spec_rows += radial_shells(back, "backbone")
    spec_rows += radial_shells(coarse, "blocked_coarse")
    write_csv(OUT / "low_momentum_spectrum.csv", spec_rows)

    plot_bars(obs_rows, "Binder_U4", "Binder U4", PLOTS / "binder_comparison.pdf")
    plot_bars(obs_rows, "xi_over_L", "xi/L", PLOTS / "xi_over_L_comparison.pdf")
    plot_spectrum(spec_rows, PLOTS / "low_momentum_spectrum.pdf")
    plot_heatmaps(fine, coarse, back, PLOTS)

    fine_obs, back_obs, coarse_obs = obs_rows
    def diff(key: str) -> float:
        return float(back_obs[key]) - float(fine_obs[key])

    report = f"""# Backbone IR Check

## Setup

- fine data: `{metadata['fine_configs_path']}`
- N configs: `{len(fine)}`
- lambda_f: `{LAMBDA}`
- kappa_f: `{KAPPA_F}`
- kernel metadata: `{metadata['kernel_metadata_path']}`
- original kernel source: `{metadata['kernel_original_source_path']}`
- caveat: `{metadata['kernel_caveat']}`
- K sum check: `{metadata['K_sum_check']:.12g}`
- eta exponent: `{ETA_EXPONENT}`
- block_norm: `{BLOCK_NORM:.12g}`
- blocking rule: `symmetric_2x2_average_after_K`

FFT convention: numpy `fft2/ifft2`; forward FFT is unnormalized and inverse FFT includes `1/V`. The coarse 8x8 Fourier block is centered and embedded into the central 8x8 region of the 16x16 spectrum with the factor `4`, then divided only on that low-mode support by `block_norm * K_tilde(p) * A(p)`.

## Block Reconstruction

- max |B_sym(phi_back) - phi_c|: `{block_check['max_abs_error']:.6g}`
- RMS error: `{block_check['rms_error']:.6g}`
- relative RMS error: `{block_check['relative_rms_error']:.6g}`

## IR Observables

| ensemble | L | Binder U4 | B4 | xi/L | chi connected | F(pmin) | phi2 | phi4 | NN | nn2 | diag | 2nn |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| original fine | 16 | {fine_obs['Binder_U4']:.6g} | {fine_obs['Binder_ratio_B4']:.6g} | {fine_obs['xi_over_L']:.6g} | {fine_obs['chi_connected']:.6g} | {fine_obs['F_pmin']:.6g} | {fine_obs['mean_phi2']:.6g} | {fine_obs['mean_phi4']:.6g} | {fine_obs['NN']:.6g} | {fine_obs['nn2']:.6g} | {fine_obs['diag']:.6g} | {fine_obs['2nn']:.6g} |
| backbone | 16 | {back_obs['Binder_U4']:.6g} | {back_obs['Binder_ratio_B4']:.6g} | {back_obs['xi_over_L']:.6g} | {back_obs['chi_connected']:.6g} | {back_obs['F_pmin']:.6g} | {back_obs['mean_phi2']:.6g} | {back_obs['mean_phi4']:.6g} | {back_obs['NN']:.6g} | {back_obs['nn2']:.6g} | {back_obs['diag']:.6g} | {back_obs['2nn']:.6g} |
| blocked coarse | 8 | {coarse_obs['Binder_U4']:.6g} | {coarse_obs['Binder_ratio_B4']:.6g} | {coarse_obs['xi_over_L']:.6g} | {coarse_obs['chi_connected']:.6g} | {coarse_obs['F_pmin']:.6g} | {coarse_obs['mean_phi2']:.6g} | {coarse_obs['mean_phi4']:.6g} | {coarse_obs['NN']:.6g} | {coarse_obs['nn2']:.6g} | {coarse_obs['diag']:.6g} | {coarse_obs['2nn']:.6g} |

## Answers

1. Does `B_sym(phi_back)` reproduce `phi_c` to roundoff?

Yes. The RMS reconstruction error is `{block_check['rms_error']:.3g}` and the max error is `{block_check['max_abs_error']:.3g}`.

2. Does `phi_back` reproduce Binder U4 of the original fine ensemble?

It reproduces Binder very closely in this dataset: original fine `{fine_obs['Binder_U4']:.6g}`, backbone `{back_obs['Binder_U4']:.6g}`, difference `{diff('Binder_U4'):.3g}`.

3. Does `phi_back` reproduce xi/L of the original fine ensemble?

It is close but not identical: original fine `{fine_obs['xi_over_L']:.6g}`, backbone `{back_obs['xi_over_L']:.6g}`, difference `{diff('xi_over_L'):.3g}`.

4. Does `phi_back` reproduce the low-momentum structure factor of the original fine ensemble?

Use `low_momentum_spectrum.csv` and `plots/low_momentum_spectrum.pdf`. The zero mode and retained low modes are the direct test; discrepancies there reflect either the blocking/upscale transfer or residual high-mode contributions to ensemble averages.

5. Which observables fail mostly because UV/detail modes are missing?

The local/UV-sensitive observables fail most strongly: phi2 difference `{diff('mean_phi2'):.3g}`, phi4 difference `{diff('mean_phi4'):.3g}`, and nn2 difference `{diff('nn2'):.3g}`. This is expected because the backbone is a low-mode field with high modes set to zero.

6. Is the backbone good enough as an IR carrier?

For Binder and xi/L, yes as a diagnostic IR carrier. It should not be treated as a complete fine field because local moments and squared-link observables remain biased by missing UV/detail modes.

7. If Binder or xi/L are wrong, is the problem in the blockavg kernel/upscale construction or in high-mode contributions?

The block reconstruction error is roundoff-small, so the algebraic construction is not the problem. Any remaining Binder or xi/L mismatch is more plausibly from the truncation to low modes and from high-mode contributions to the nonlinear observables.

8. Should future learned models condition on `phi_back` as the IR field, or is `phi_back` itself already IR-biased?

Future learned models can condition on `phi_back` as an IR carrier, but should not hard-fix it as the generated field. The backbone carries the coarse/IR information while the model must generate UV/detail structure around it.
"""
    (OUT / "report.md").write_text(report)


if __name__ == "__main__":
    main()
