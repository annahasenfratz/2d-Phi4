#!/usr/bin/env python3
"""Step-by-step deterministic inverse-blocking diagnostic.

This script reads the finite-lambda perfect-blocking kernel as provenance only.
All inverse-blocking outputs are written inside InverseBlocking_MIT_NF.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
import traceback
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
OUT = PROJECT / "outputs" / "inverse_blocking_step_by_step"
PLOTS = OUT / "plots"
KERNEL_COPY_DIR = PROJECT / "kernels" / "from_perfect_blocking_lam1p0"

SOURCE_DIR_CANDIDATES = [
    ROOT / "perfect_blocking" / "perfect_blocking_lam1p0",
    ROOT / "perfect_blocking" / "perfect_blocking_lam1",
]
SUMMARY_NAME = "perfect_block_lam1_kernel5x5_KL_etafit_L32_to_L16_summary.json"
REPORT_NAME = "perfect_block_lam1_kernel5x5_KL_etafit_L32_to_L16_report.md"
FINE_CONFIGS = (
    PROJECT
    / "outputs"
    / "physics_diagnostics_kc030_kf032"
    / "reference_ensembles"
    / "fine"
    / "configs.npy"
)

LAMBDA = 1.0
KAPPA_CR_APPROX = 0.3402
ETA_EXPONENT = 0.25
B = 2
BLOCK_NORM = B ** (ETA_EXPONENT / 2.0)
L_FINE = 16
L_COARSE = 8
BATCH_N = 64
BOOTSTRAP_N = 256
RNG_SEED = 20240623
NEAR_ZERO_TOL = 1.0e-8

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


def write_error(exc: BaseException) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "error_log.txt").open("a") as f:
        f.write("\n=== failure ===\n")
        f.write("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))


def source_dir() -> Path:
    for path in SOURCE_DIR_CANDIDATES:
        if (path / SUMMARY_NAME).exists():
            return path
    raise FileNotFoundError("Could not find corrected eta-fit summary in perfect_blocking source tree")


def prepare_dirs() -> Path:
    if OUT.exists():
        existing = list(OUT.iterdir())
        if existing:
            names = "\n".join(str(p.relative_to(ROOT)) for p in sorted(existing))
            raise RuntimeError(f"Refusing to overwrite previous output directory contents:\n{names}")
    OUT.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(exist_ok=True)
    KERNEL_COPY_DIR.mkdir(parents=True, exist_ok=True)
    return source_dir()


def load_and_copy_kernel_metadata(src: Path) -> dict:
    summary_path = src / SUMMARY_NAME
    report_path = src / REPORT_NAME
    local_summary = KERNEL_COPY_DIR / SUMMARY_NAME
    local_report = KERNEL_COPY_DIR / REPORT_NAME
    shutil.copy2(summary_path, local_summary)
    if report_path.exists():
        shutil.copy2(report_path, local_report)

    with summary_path.open() as f:
        summary = json.load(f)
    best = summary["best_by_D_op"]
    weights = {k: float(best[k]) for k in ["w00", "w10", "w11", "w20", "w21", "w22"]}
    mult = {k: len(v) for k, v in SHELLS.items()}
    sum_k = float(sum(weights[k] * mult[k] for k in weights))
    stencil = np.zeros((5, 5), dtype=float)
    for shell, offsets in SHELLS.items():
        for dy, dx in offsets:
            stencil[dy + 2, dx + 2] = weights[shell]

    metadata = {
        "original_source_path": str(src.resolve()),
        "exact_file_copied": str(summary_path.resolve()),
        "local_copied_summary": str(local_summary.resolve()),
        "local_copied_report": str(local_report.resolve()) if report_path.exists() else None,
        "copy_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "eta_convention": "eta is exponent; block_norm = b**(eta_exponent/2); B = block_norm * K",
        "lambda": LAMBDA,
        "kappa_cr_approx": KAPPA_CR_APPROX,
        "eta": ETA_EXPONENT,
        "eta_exponent": ETA_EXPONENT,
        "b": B,
        "block_norm": BLOCK_NORM,
        "block_norm_numeric": BLOCK_NORM,
        "K_normalization": "K normalized so sum_x K = 1",
        "sum_x_K": sum_k,
        "renormalized_in_diagnostic": False,
        "weights": weights,
        "kernel_shells": {k: [list(x) for x in v] for k, v in SHELLS.items()},
        "shell_multiplicities": mult,
        "stencil_5x5_centered": stencil.tolist(),
    }
    if abs(sum_k - 1.0) > 1.0e-12:
        raise RuntimeError(f"K is not normalized to 1: sum_x K = {sum_k}")
    if abs(BLOCK_NORM - 2**0.125) > 1.0e-15:
        raise RuntimeError("Incorrect block_norm; expected 2**0.125")

    text = json.dumps(metadata, indent=2) + "\n"
    (KERNEL_COPY_DIR / "selected_kernel_metadata.json").write_text(text)
    (OUT / "kernel_metadata_check.json").write_text(text)
    return metadata


def kernel_array(weights: dict[str, float], n: int) -> np.ndarray:
    arr = np.zeros((n, n), dtype=float)
    for shell, offsets in SHELLS.items():
        for dy, dx in offsets:
            arr[dy % n, dx % n] += weights[shell]
    return arr


def apply_K(configs: np.ndarray, weights: dict[str, float]) -> np.ndarray:
    out = np.zeros_like(configs, dtype=float)
    for shell, offsets in SHELLS.items():
        for dy, dx in offsets:
            out += weights[shell] * np.roll(np.roll(configs, -dy, axis=-2), -dx, axis=-1)
    return out


def forward_block(fine: np.ndarray, weights: dict[str, float]) -> np.ndarray:
    psi = apply_K(fine, weights)
    return BLOCK_NORM * psi[:, 0::2, 0::2]


def save_heatmap(data: np.ndarray, title: str, path: Path, mark_bz: bool = True) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 4.6), constrained_layout=True)
    im = ax.imshow(data, origin="lower", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("p_x index, fftshifted")
    ax.set_ylabel("p_y index, fftshifted")
    if mark_bz:
        start = L_FINE // 2 - L_COARSE // 2 - 0.5
        ax.add_patch(plt.Rectangle((start, start), L_COARSE, L_COARSE, fill=False, ec="white", lw=1.5))
    fig.colorbar(im, ax=ax)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def kernel_fourier(ktilde: np.ndarray) -> dict:
    kt = np.fft.fftshift(ktilde)
    abs_kt = np.abs(kt)
    p = np.fft.fftshift(np.fft.fftfreq(L_FINE)) * 2 * np.pi
    px, py = np.meshgrid(p, p)
    start = L_FINE // 2 - L_COARSE // 2
    stop = start + L_COARSE
    low = abs_kt[start:stop, start:stop]
    gidx = np.unravel_index(np.argmin(abs_kt), abs_kt.shape)
    llocal = np.unravel_index(np.argmin(low), low.shape)
    lidx = (llocal[0] + start, llocal[1] + start)
    inv_clip = np.clip(1.0 / np.maximum(abs_kt, 1.0e-14), 0, np.quantile(1.0 / np.maximum(abs_kt, 1.0e-14), 0.98))

    save_heatmap(np.real(kt), "Re K_tilde(p)", OUT / "kernel_fourier_heatmap_real.png")
    save_heatmap(abs_kt, "|K_tilde(p)|", OUT / "kernel_fourier_heatmap_abs.png")
    save_heatmap(inv_clip, "clipped 1 / |K_tilde(p)|", OUT / "kernel_fourier_inverse_heatmap.png")

    summary = {
        "global_min_abs_K_tilde": float(abs_kt[gidx]),
        "global_min_location_shifted_index": [int(gidx[0]), int(gidx[1])],
        "global_min_location_momentum": [float(py[gidx]), float(px[gidx])],
        "coarse_BZ_min_abs_K_tilde": float(abs_kt[lidx]),
        "coarse_BZ_min_location_shifted_index": [int(lidx[0]), int(lidx[1])],
        "coarse_BZ_min_location_momentum": [float(py[lidx]), float(px[lidx])],
        "near_zero_tolerance": NEAR_ZERO_TOL,
        "near_zero_inside_coarse_BZ": bool(np.any(low < NEAR_ZERO_TOL)),
    }
    (OUT / "kernel_fourier_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (OUT / "kernel_fourier_summary.md").write_text(
        "# Kernel Fourier Summary\n\n"
        f"- global min |K_tilde|: {summary['global_min_abs_K_tilde']:.12g} at {summary['global_min_location_momentum']}\n"
        f"- coarse-BZ min |K_tilde|: {summary['coarse_BZ_min_abs_K_tilde']:.12g} at {summary['coarse_BZ_min_location_momentum']}\n"
        f"- near-zero inside coarse BZ: {summary['near_zero_inside_coarse_BZ']}\n"
        f"- block_norm: {BLOCK_NORM:.15g}\n"
    )
    return summary


def embed_low(coarse_fft: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fine = np.zeros((coarse_fft.shape[0], L_FINE, L_FINE), dtype=complex)
    start = L_FINE // 2 - L_COARSE // 2
    stop = start + L_COARSE
    fine_shift = np.zeros_like(fine)
    fine_shift[:, start:stop, start:stop] = 4.0 * np.fft.fftshift(coarse_fft, axes=(-2, -1))
    mask_shift = np.zeros((L_FINE, L_FINE), dtype=bool)
    mask_shift[start:stop, start:stop] = True
    return np.fft.ifftshift(fine_shift, axes=(-2, -1)), np.fft.ifftshift(mask_shift)


def fine_low_index(k: int) -> int:
    return k if k < L_COARSE // 2 else k + L_COARSE


def inverse_lowmode(coarse_fft: np.ndarray, ktilde: np.ndarray) -> tuple[np.ndarray, dict]:
    padded, mask = embed_low(coarse_fft)
    np.save(OUT / "coarse_fft_lowmode_padded.npy", padded)
    denom = BLOCK_NORM * ktilde
    inv_fft = np.zeros_like(padded)
    inv_fft[:, mask] = padded[:, mask] / denom[mask]
    field = np.fft.ifft2(inv_fft, axes=(-2, -1)).real
    recon = np.zeros_like(inv_fft)
    recon[:, mask] = denom[mask] * inv_fft[:, mask]
    err = recon[:, mask] - padded[:, mask]
    check = {
        "fft_padding_convention": "LxL coarse FFT block is multiplied by 4 when embedded in the 2Lx2L low-mode block",
        "inverse_divisor": "block_norm * K_tilde(p)",
        "inverse_applied_only_on_lowmode_support": True,
        "high_momentum_modes_zero": bool(np.all(inv_fft[:, ~mask] == 0)),
        "max_abs_reconstruction_error_on_padded_low_modes": float(np.max(np.abs(err))),
        "rms_reconstruction_error_on_padded_low_modes": float(np.sqrt(np.mean(np.abs(err) ** 2))),
        "has_nan_or_inf": bool(not np.isfinite(field).all()),
    }
    np.save(OUT / "inverse_kernel_lowmode_field.npy", field)
    (OUT / "inverse_kernel_lowmode_check.json").write_text(json.dumps(check, indent=2) + "\n")
    return field, check


def inverse_alias(coarse_fft: np.ndarray, ktilde: np.ndarray) -> tuple[np.ndarray, dict]:
    alias_fft = np.zeros((coarse_fft.shape[0], L_FINE, L_FINE), dtype=complex)
    for ky in range(L_COARSE):
        fy = fine_low_index(ky)
        for kx in range(L_COARSE):
            fx = fine_low_index(kx)
            corr = coarse_fft[:, ky, kx] / (BLOCK_NORM * ktilde[fy, fx])
            for ay in (0, L_COARSE):
                for ax in (0, L_COARSE):
                    alias_fft[:, (fy + ay) % L_FINE, (fx + ax) % L_FINE] = corr
    field = np.fft.ifft2(alias_fft, axes=(-2, -1)).real
    ee = float(np.max(np.abs(field[:, 0::2, 0::2])))
    oe = float(np.max(np.abs(field[:, 1::2, 0::2])))
    eo = float(np.max(np.abs(field[:, 0::2, 1::2])))
    oo = float(np.max(np.abs(field[:, 1::2, 1::2])))
    check = {
        "alias_tiling_convention": "K^{-1}-corrected LxL block tiled into four alias sectors with no extra factor",
        "max_abs_even_even": ee,
        "max_abs_odd_even": oe,
        "max_abs_even_odd": eo,
        "max_abs_odd_odd": oo,
        "odd_even_leakage_ratio": max(oe, eo, oo) / ee if ee > 0 else math.inf,
        "has_nan_or_inf": bool(not np.isfinite(field).all()),
    }
    np.save(OUT / "inverse_kernel_alias_field.npy", field)
    (OUT / "alias_support_check.json").write_text(json.dumps(check, indent=2) + "\n")
    heat = np.log10(np.abs(np.fft.fftshift(np.fft.fft2(field[0]))) + 1.0e-14)
    save_heatmap(heat, "log10 |FFT(alias field)|", OUT / "alias_fft_heatmap.png", mark_bz=False)
    (OUT / "alias_support_report.md").write_text(
        "# Alias Support Check\n\n"
        f"- max |even-even|: {ee:.12g}\n"
        f"- max |odd-even|: {oe:.12g}\n"
        f"- max |even-odd|: {eo:.12g}\n"
        f"- max |odd-odd|: {oo:.12g}\n"
        f"- odd/even leakage ratio: {check['odd_even_leakage_ratio']:.12g}\n"
    )
    return field, check


def neighbor_fill(alias: np.ndarray) -> tuple[np.ndarray, dict]:
    out = np.zeros_like(alias)
    ee = alias[:, 0::2, 0::2]
    out[:, 0::2, 0::2] = ee
    out[:, 1::2, 0::2] = 0.5 * (ee + np.roll(ee, -1, axis=-2))
    out[:, 0::2, 1::2] = 0.5 * (ee + np.roll(ee, -1, axis=-1))
    out[:, 1::2, 1::2] = 0.25 * (
        ee
        + np.roll(ee, -1, axis=-2)
        + np.roll(ee, -1, axis=-1)
        + np.roll(np.roll(ee, -1, axis=-2), -1, axis=-1)
    )
    check = {
        "max_abs_even_even_preservation_error": float(np.max(np.abs(out[:, 0::2, 0::2] - ee))),
        "has_nan_or_inf": bool(not np.isfinite(out).all()),
        "mean": float(np.mean(out)),
        "variance": float(np.var(out)),
    }
    np.save(OUT / "neighbor_filled_init.npy", out)
    (OUT / "fill_check.json").write_text(json.dumps(check, indent=2) + "\n")
    return out, check


def obs_values(configs: np.ndarray) -> dict[str, float]:
    n, ly, lx = configs.shape
    v = ly * lx
    m_cfg = configs.mean(axis=(-2, -1))
    nn_cfg = 0.5 * (
        (configs * np.roll(configs, -1, axis=-2)).mean(axis=(-2, -1))
        + (configs * np.roll(configs, -1, axis=-1)).mean(axis=(-2, -1))
    )
    diag_cfg = (configs * np.roll(np.roll(configs, -1, axis=-2), -1, axis=-1)).mean(axis=(-2, -1))
    twonn_cfg = 0.5 * (
        (configs * np.roll(configs, -2, axis=-2)).mean(axis=(-2, -1))
        + (configs * np.roll(configs, -2, axis=-1)).mean(axis=(-2, -1))
    )
    m2 = float(np.mean(m_cfg**2))
    m4 = float(np.mean(m_cfg**4))
    b4 = m4 / (m2 * m2) if m2 > 0 else math.nan
    u4 = 1.0 - b4 / 3.0 if math.isfinite(b4) else math.nan
    ft = np.fft.fft2(configs, axes=(-2, -1))
    chi = float(v * (np.mean(m_cfg**2) - np.mean(m_cfg) ** 2))
    fmin = float(0.5 * (np.mean(np.abs(ft[:, 1, 0]) ** 2) + np.mean(np.abs(ft[:, 0, 1]) ** 2)) / v)
    ratio = chi / fmin - 1.0 if fmin > 0 else math.nan
    xi = (1.0 / (2.0 * math.sin(math.pi / lx))) * math.sqrt(ratio) if ratio > 0 else math.nan
    return {
        "m": float(np.mean(m_cfg)),
        "|m|": float(np.mean(np.abs(m_cfg))),
        "phi2": float(np.mean(configs**2)),
        "phi4": float(np.mean(configs**4)),
        "NN": float(np.mean(nn_cfg)),
        "diag": float(np.mean(diag_cfg)),
        "2nn": float(np.mean(twonn_cfg)),
        "Binder_U4": float(u4),
        "Binder_B4": float(b4),
        "xi": float(xi) if math.isfinite(xi) else math.nan,
        "xi/L": float(xi / lx) if math.isfinite(xi) else math.nan,
    }


def bootstrap(configs: np.ndarray, rng: np.random.Generator) -> dict[str, dict[str, float]]:
    mean = obs_values(configs)
    vals = {k: [] for k in mean}
    for _ in range(BOOTSTRAP_N):
        idx = rng.integers(0, configs.shape[0], size=configs.shape[0])
        rep = obs_values(configs[idx])
        for key in vals:
            vals[key].append(rep[key])
    return {k: {"mean": mean[k], "error": float(np.nanstd(vals[k], ddof=1))} for k in mean}


def observable_tables(ensembles: dict[str, np.ndarray]) -> dict:
    rng = np.random.default_rng(RNG_SEED)
    summary = {name: bootstrap(arr, rng) for name, arr in ensembles.items()}
    base = summary["original_fine"]
    rows = []
    for name, vals in summary.items():
        for op, stat in vals.items():
            diff = stat["mean"] - base[op]["mean"]
            den = math.sqrt(stat["error"] ** 2 + base[op]["error"] ** 2)
            rows.append(
                {
                    "ensemble": name,
                    "L": int(ensembles[name].shape[-1]),
                    "operator": op,
                    "mean": stat["mean"],
                    "error": stat["error"],
                    "original_fine_mean": base[op]["mean"],
                    "original_fine_error": base[op]["error"],
                    "difference_vs_original_fine": diff,
                    "z_score_vs_original_fine": diff / den if den > 0 else math.nan,
                }
            )
    with (OUT / "operator_comparison.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    md = ["# Operator Comparison\n\nErrors are 256-bootstrap standard deviations.\n"]
    for name in ensembles:
        md += [f"\n## {name}\n", "| operator | mean | error | diff vs original fine | z |\n", "|---|---:|---:|---:|---:|\n"]
        for row in rows:
            if row["ensemble"] == name:
                md.append(
                    f"| {row['operator']} | {row['mean']:.8g} | {row['error']:.3g} | "
                    f"{row['difference_vs_original_fine']:.8g} | {row['z_score_vs_original_fine']:.3g} |\n"
                )
    (OUT / "operator_comparison.md").write_text("".join(md))
    (OUT / "observable_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return {"summary": summary, "rows": rows}


def plot_configs(fine: np.ndarray, coarse: np.ndarray, low: np.ndarray, alias: np.ndarray, filled: np.ndarray) -> None:
    for i in range(min(4, fine.shape[0])):
        panels = [
            ("original fine", fine[i]),
            ("blocked coarse", coarse[i]),
            ("low-mode inverse", low[i]),
            ("alias even-even", alias[i]),
            ("neighbor-filled", filled[i]),
            ("filled - original", filled[i] - fine[i]),
        ]
        fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.6), constrained_layout=True)
        for ax, (title, data) in zip(axes.ravel(), panels):
            im = ax.imshow(data, origin="lower", cmap="coolwarm")
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, shrink=0.78)
        fig.savefig(PLOTS / f"config_{i:02d}_diagnostic.pdf")
        fig.savefig(PLOTS / f"config_{i:02d}_diagnostic.png", dpi=180)
        plt.close(fig)


def report(metadata: dict, fine_source: dict, ksum: dict, low: dict, alias: dict, fill: dict, obs: dict) -> None:
    filled_rows = [r for r in obs["rows"] if r["ensemble"] == "neighbor_filled"]
    distorted = sorted(
        filled_rows,
        key=lambda r: abs(r["z_score_vs_original_fine"]) if math.isfinite(r["z_score_vs_original_fine"]) else -1,
        reverse=True,
    )[:5]
    inverse_safe = (not ksum["near_zero_inside_coarse_BZ"]) and ksum["coarse_BZ_min_abs_K_tilde"] > NEAR_ZERO_TOL and not low["has_nan_or_inf"]
    alias_ok = alias["odd_even_leakage_ratio"] < 1.0e-10
    fill_ok = fill["max_abs_even_even_preservation_error"] < 1.0e-12
    text = f"""# Deterministic Inverse-Blocking Diagnostic

This is an algebra and raw-observable diagnostic only. No normalizing flow was trained.

## Inputs

- project: `InverseBlocking_MIT_NF`
- kernel provenance source: `{metadata['original_source_path']}`
- copied kernel metadata: `{metadata['local_copied_summary']}`
- fine ensemble: `{fine_source['path']}`
- batch: {fine_source['batch_n']} configurations from source shape {fine_source['source_shape']}
- generated new configs: {fine_source['generated_new_configs']}

## Convention

- eta is the exponent: {ETA_EXPONENT}
- b: {B}
- block_norm = b**(eta/2) = 2**0.125 = {BLOCK_NORM:.15g}
- K is the normalized shape kernel, sum_x K = {metadata['sum_x_K']:.15g}
- B is the full blocking operator, B = block_norm * K

## Answers

- Did K normalize to 1? {'yes' if abs(metadata['sum_x_K'] - 1.0) < 1e-12 else 'no'}.
- Was block_norm = 2**0.125 used everywhere? yes, in forward blocking and inverse division by block_norm*K_tilde.
- What is min |K_tilde(p)| globally? {ksum['global_min_abs_K_tilde']:.12g} at p={ksum['global_min_location_momentum']}.
- What is min |K_tilde(p)| inside the coarse Brillouin zone? {ksum['coarse_BZ_min_abs_K_tilde']:.12g} at p={ksum['coarse_BZ_min_location_momentum']}.
- Is K^-1 safe on the low-mode support? {'yes for this diagnostic tolerance' if inverse_safe else 'not safely established'}.
- Does alias tiling produce even-even support? {'yes' if alias_ok else 'not exactly'}; odd/even leakage ratio = {alias['odd_even_leakage_ratio']:.12g}.
- Does neighbor filling preserve even-even values exactly? {'yes' if fill_ok else 'no'}; max error = {fill['max_abs_even_even_preservation_error']:.12g}.

## Most Distorted Neighbor-Filled Observables

| operator | original fine | neighbor-filled | difference | z |
|---|---:|---:|---:|---:|
"""
    for row in distorted:
        text += (
            f"| {row['operator']} | {row['original_fine_mean']:.8g} | {row['mean']:.8g} | "
            f"{row['difference_vs_original_fine']:.8g} | {row['z_score_vs_original_fine']:.3g} |\n"
        )
    text += """
## Interpretation

The deterministic initializer is plausible only as an algebraic starting condition for a later conditional NF if the low-mode inverse, alias support, and neighbor-fill constraints pass. It is not a physics-success claim. The raw observable mismatches above identify the UV/detail distortions left for conditional generative filling.
"""
    (OUT / "report.md").write_text(text)


def main() -> None:
    src = prepare_dirs()
    metadata = load_and_copy_kernel_metadata(src)
    fine_all = np.load(FINE_CONFIGS)
    if fine_all.ndim != 3 or fine_all.shape[1:] != (L_FINE, L_FINE):
        raise RuntimeError(f"Expected fine configs shaped (N,{L_FINE},{L_FINE}), got {fine_all.shape}")
    fine = np.asarray(fine_all[:BATCH_N], dtype=float)
    fine_source = {
        "path": str(FINE_CONFIGS.resolve()),
        "source_shape": list(fine_all.shape),
        "batch_n": int(fine.shape[0]),
        "lambda": LAMBDA,
        "kappa_f_inferred_from_path": 0.32,
        "generated_new_configs": False,
        "production_quality_claimed": False,
    }
    (OUT / "fine_ensemble_source.json").write_text(json.dumps(fine_source, indent=2) + "\n")
    np.save(OUT / "input_fine_batch.npy", fine)

    coarse = forward_block(fine, metadata["weights"])
    if coarse.shape != (BATCH_N, L_COARSE, L_COARSE) or not np.isfinite(coarse).all():
        raise RuntimeError("Forward-blocked coarse ensemble failed shape/finite checks")
    np.save(OUT / "forward_blocked_coarse.npy", coarse)

    ktilde = np.fft.fft2(kernel_array(metadata["weights"], L_FINE))
    ksum = kernel_fourier(ktilde)
    coarse_fft = np.fft.fft2(coarse, axes=(-2, -1))
    low_field, low_check = inverse_lowmode(coarse_fft, ktilde)
    alias_field, alias_check = inverse_alias(coarse_fft, ktilde)
    filled, fill_check = neighbor_fill(alias_field)

    obs = observable_tables(
        {
            "original_fine": fine,
            "forward_blocked_coarse": coarse,
            "lowmode_inverse": low_field,
            "alias_even_even": alias_field,
            "neighbor_filled": filled,
        }
    )
    plot_configs(fine, coarse, low_field, alias_field, filled)
    report(metadata, fine_source, ksum, low_check, alias_check, fill_check, obs)
    shutil.copy2(Path(__file__), OUT / Path(__file__).name)
    (OUT / "error_log.txt").write_text("No errors.\n")


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        write_error(exc)
        raise
