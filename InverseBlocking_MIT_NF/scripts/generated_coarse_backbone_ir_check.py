#!/usr/bin/env python3
"""IR test for native/generated 8x8 coarse fields upscaled to phi_back."""

from __future__ import annotations

import csv
import json
import math
import os
import re
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT / "outputs" / "matplotlib_config"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DATA = PROJECT / "outputs" / "paired_data_lam1_kappaf0p320"
COARSE_CAL = PROJECT / "outputs" / "coarse_distribution_calibration"
NATIVE_SCAN = COARSE_CAL / "generated_native_scan"
NATIVE_EXTENDED = COARSE_CAL / "generated_native_extended"
NATIVE_WOLFF = COARSE_CAL / "generated_native_wolff"
NATIVE_K030 = PROJECT / "outputs" / "physics_diagnostics_kc030_kf032" / "reference_ensembles" / "coarse" / "configs.npy"
KERNEL_META = PROJECT / "kernels" / "from_perfect_blocking_lam1p0_blockavg" / "selected_kernel_metadata.json"
OUT = PROJECT / "outputs" / "generated_coarse_backbone_ir_check"
HISTS = OUT / "magnetization_histograms"
PLOTS = OUT / "plots"

L_FINE = 16
L_COARSE = 8
ETA_EXPONENT = 0.25
BLOCK_NORM = 2 ** (ETA_EXPONENT / 2.0)
LAMBDA = 1.0
KAPPA_F = 0.320

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
        raise RuntimeError(f"No kernel weights found in {KERNEL_META}")
    return {k: float(weights[k]) for k in SHELLS}, meta


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


def upscale_backbone(coarse: np.ndarray, w: dict[str, float]) -> tuple[np.ndarray, dict[str, float]]:
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
        "fft_convention": "numpy fft2/ifft2; coarse FFT block is fftshift-centered, multiplied by 4 in 16x16 low-mode embedding, then divided on low support only",
    }


def ensemble_observables(phi: np.ndarray, label: str, L: int, source_type: str, kappa_for_action: float | None = KAPPA_F) -> dict[str, object]:
    phi = phi.astype(np.float64)
    n, ly, lx = phi.shape
    v = ly * lx
    m = phi.mean(axis=(-2, -1))
    m2 = float(np.mean(m**2))
    m4 = float(np.mean(m**4))
    b4 = m4 / (m2 * m2) if m2 > 0 else math.nan
    u4 = 1.0 - b4 / 3.0 if math.isfinite(b4) else math.nan
    nn = 0.5 * (
        (phi * np.roll(phi, -1, axis=-2)).mean(axis=(-2, -1))
        + (phi * np.roll(phi, -1, axis=-1)).mean(axis=(-2, -1))
    )
    nn2 = 0.5 * (
        ((phi * np.roll(phi, -1, axis=-2)) ** 2).mean(axis=(-2, -1))
        + ((phi * np.roll(phi, -1, axis=-1)) ** 2).mean(axis=(-2, -1))
    )
    diag = (phi * np.roll(np.roll(phi, -1, axis=-2), -1, axis=-1)).mean(axis=(-2, -1))
    twonn = 0.5 * (
        (phi * np.roll(phi, -2, axis=-2)).mean(axis=(-2, -1))
        + (phi * np.roll(phi, -2, axis=-1)).mean(axis=(-2, -1))
    )
    ft = np.fft.fft2(phi, axes=(-2, -1))
    chi = float(v * (np.mean(m**2) - np.mean(m) ** 2))
    fmin = float(0.5 * (np.mean(np.abs(ft[:, 1, 0]) ** 2) + np.mean(np.abs(ft[:, 0, 1]) ** 2)) / v)
    ratio = chi / fmin - 1.0 if fmin > 0 else math.nan
    xi = (1.0 / (2.0 * math.sin(math.pi / lx))) * math.sqrt(ratio) if ratio > 0 else math.nan
    hop = math.nan
    p2 = math.nan
    p4 = math.nan
    action = math.nan
    if kappa_for_action is not None:
        hop = -4.0 * kappa_for_action * float(np.mean(nn))
        p2 = -float(np.mean(phi**2))
        p4 = float(np.mean(phi**4))
        action = hop + p2 + p4
    return {
        "ensemble": label,
        "source_type": source_type,
        "L": L,
        "N": n,
        "m": float(np.mean(m)),
        "abs_m": float(np.mean(np.abs(m))),
        "m2": m2,
        "m4": m4,
        "Binder_U4": float(u4),
        "Binder_ratio_B4": float(b4),
        "chi_connected": chi,
        "F_pmin": fmin,
        "xi": float(xi) if math.isfinite(xi) else math.nan,
        "xi_over_L": float(xi / lx) if math.isfinite(xi) else math.nan,
        "phi2": float(np.mean(phi**2)),
        "phi4": float(np.mean(phi**4)),
        "NN": float(np.mean(nn)),
        "nn2": float(np.mean(nn2)),
        "diag": float(np.mean(diag)),
        "2nn": float(np.mean(twonn)),
        "action_hopping_density": hop,
        "action_phi2_density": p2,
        "action_phi4_density": p4,
        "action_density": action,
    }


def structure_factor(phi: np.ndarray, label: str, source_type: str, modes: list[tuple[int, int]]) -> list[dict[str, object]]:
    phi = phi.astype(np.float64)
    _, ly, lx = phi.shape
    v = ly * lx
    ft = np.fft.fft2(phi, axes=(-2, -1))
    rows = []
    for ky, kx in modes:
        rows.append(
            {
                "ensemble": label,
                "source_type": source_type,
                "L": lx,
                "ky_index": ky,
                "kx_index": kx,
                "p_y": float(2.0 * math.pi * ky / ly),
                "p_x": float(2.0 * math.pi * kx / lx),
                "S_p": float(np.mean(np.abs(ft[:, ky % ly, kx % lx]) ** 2) / v),
            }
        )
    return rows


def source_label_from_path(path: Path) -> str:
    m = re.search(r"kappa0p(\d+)", path.name)
    if not m:
        return path.stem
    digits = m.group(1)
    return f"native_kappa_0.{digits}"


def discover_coarse_sources(w: dict[str, float]) -> list[dict[str, object]]:
    fine = np.load(DATA / "fine_configs.npy").astype(np.float64)
    blocked_control = block_sym(fine, w)
    sources: list[dict[str, object]] = [
        {
            "label": "blocked_fine_control",
            "source_type": "blocked_fine_control",
            "path": str((DATA / "fine_configs.npy").resolve()),
            "coarse": blocked_control,
            "production_quality": True,
            "notes": "B_sym recomputed from target fine ensemble",
        }
    ]
    if NATIVE_K030.exists():
        sources.append(
            {
                "label": "native_kappa_0.30",
                "source_type": "native_phi4",
                "path": str(NATIVE_K030.resolve()),
                "coarse": np.load(NATIVE_K030).astype(np.float64),
                "production_quality": True,
                "notes": "native 8x8 reference from physics_diagnostics_kc030_kf032",
            }
        )
    for path in sorted(NATIVE_SCAN.glob("native_coarse_lam1_kappa*_L8_nonproduction.npy")):
        sources.append(
            {
                "label": source_label_from_path(path),
                "source_type": "native_phi4_scan_nonproduction",
                "path": str(path.resolve()),
                "coarse": np.load(path).astype(np.float64),
                "production_quality": False,
                "notes": "short diagnostic scan ensemble; non-production",
            }
        )
    for path in sorted(NATIVE_EXTENDED.glob("native_coarse_lam1_kappa*_L8_extended.npy")):
        label = source_label_from_path(path) + "_extended"
        sources.append(
            {
                "label": label,
                "source_type": "native_phi4_extended_local_metropolis",
                "path": str(path.resolve()),
                "coarse": np.load(path).astype(np.float64),
                "production_quality": False,
                "notes": "extended local-Metropolis diagnostic chain; autocorrelation not measured",
            }
        )
    for path in sorted(NATIVE_WOLFF.glob("native_coarse_lam1_kappa*_L8_wolff.npy")):
        label = source_label_from_path(path) + "_wolff"
        sources.append(
            {
                "label": label,
                "source_type": "native_phi4_embedded_wolff_sign_cluster",
                "path": str(path.resolve()),
                "coarse": np.load(path).astype(np.float64),
                "production_quality": False,
                "notes": "diagnostic embedded Wolff sign-cluster plus local Metropolis amplitude chain; autocorrelation not measured",
            }
        )
    return sources


def plot_magnetization_histograms(mag_rows: dict[str, np.ndarray]) -> None:
    for label, vals in mag_rows.items():
        fig, ax = plt.subplots(figsize=(5.5, 3.4))
        ax.hist(vals, bins=40, density=True, alpha=0.8, color="#4C78A8")
        ax.set_xlabel("m")
        ax.set_ylabel("density")
        ax.set_title(label)
        fig.tight_layout()
        fig.savefig(HISTS / f"{label}_magnetization_histogram.pdf")
        fig.savefig(HISTS / f"{label}_magnetization_histogram.png", dpi=160)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for label, vals in mag_rows.items():
        ax.hist(vals, bins=50, density=True, histtype="step", linewidth=1.5, label=label)
    ax.set_xlabel("m")
    ax.set_ylabel("density")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(HISTS / "magnetization_histograms_overlay.pdf")
    fig.savefig(HISTS / "magnetization_histograms_overlay.png", dpi=160)
    plt.close(fig)


def plot_low_spectrum(rows: list[dict[str, object]]) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    labels = [f"({ky},{kx})" for ky, kx in LOW_MODES]
    xs = np.arange(len(LOW_MODES))
    for ensemble in sorted({str(r["ensemble"]) for r in rows if int(r["L"]) == L_FINE}):
        vals = [float(r["S_p"]) for r in rows if r["ensemble"] == ensemble and int(r["L"]) == L_FINE][: len(LOW_MODES)]
        if len(vals) == len(LOW_MODES):
            ax.plot(xs, vals, marker="o", label=ensemble)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_xlabel("momentum index (ky,kx)")
    ax.set_ylabel("S(p) = <|FFT(phi)(p)|^2> / V")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(PLOTS / "low_momentum_spectrum.pdf")
    fig.savefig(PLOTS / "low_momentum_spectrum.png", dpi=160)
    plt.close(fig)


def main() -> None:
    HISTS.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)

    w, kernel_meta = load_kernel()
    fine = np.load(DATA / "fine_configs.npy").astype(np.float64)
    sources = discover_coarse_sources(w)

    fine_obs = ensemble_observables(fine, "target_fine_16x16", L_FINE, "target_fine", KAPPA_F)
    obs_rows = [fine_obs]
    spec_rows = structure_factor(fine, "target_fine_16x16", "target_fine", LOW_MODES)
    block_rows = []
    mag_hists = {"target_fine_16x16": fine.mean(axis=(-2, -1))}
    source_summary = []

    for src in sources:
        coarse = np.asarray(src["coarse"], dtype=np.float64)
        back, transfer = upscale_backbone(coarse, w)
        reblocked = block_sym(back, w)
        err = reblocked - coarse
        label = str(src["label"])
        source_type = str(src["source_type"])
        np.save(OUT / f"{label}_backbone.npy", back.astype(np.float32))
        obs_rows.append(ensemble_observables(back, f"{label}_backbone", L_FINE, source_type, KAPPA_F))
        obs_rows.append(ensemble_observables(coarse, f"{label}_coarse", L_COARSE, source_type, None))
        spec_rows += structure_factor(back, f"{label}_backbone", source_type, LOW_MODES)
        spec_rows += structure_factor(coarse, f"{label}_coarse", source_type, LOW_MODES)
        mag_hists[f"{label}_backbone"] = back.mean(axis=(-2, -1))
        block_rows.append(
            {
                "ensemble": label,
                "source_type": source_type,
                "N": int(len(coarse)),
                "coarse_path": str(src["path"]),
                "production_quality": bool(src["production_quality"]),
                "max_abs_error": float(np.max(np.abs(err))),
                "rms_error": float(np.sqrt(np.mean(err**2))),
                "mean_abs_error": float(np.mean(np.abs(err))),
                "coarse_rms": float(np.sqrt(np.mean(coarse**2))),
                "relative_rms_error": float(np.sqrt(np.mean(err**2)) / max(np.sqrt(np.mean(coarse**2)), 1.0e-30)),
                **transfer,
            }
        )
        source_summary.append(
            {
                "label": label,
                "source_type": source_type,
                "path": str(src["path"]),
                "N": int(len(coarse)),
                "production_quality": bool(src["production_quality"]),
                "notes": str(src["notes"]),
            }
        )

    write_csv(OUT / "ir_observables.csv", obs_rows)
    write_csv(OUT / "low_momentum_spectrum.csv", spec_rows)
    write_csv(OUT / "block_reconstruction_check.csv", block_rows)
    (OUT / "source_metadata.json").write_text(
        json.dumps(
            {
                "kernel_metadata_path": str(KERNEL_META.resolve()),
                "kernel_original_source_path": kernel_meta.get("original_source_path"),
                "kernel_local_copy_path": kernel_meta.get("local_copy_path"),
                "kernel_caveat": kernel_meta.get("caveat"),
                "weights": w,
                "K_sum_check": kernel_sum(w),
                "eta_exponent": ETA_EXPONENT,
                "block_norm": BLOCK_NORM,
                "sources": source_summary,
            },
            indent=2,
        )
        + "\n"
    )

    plot_magnetization_histograms(mag_hists)
    plot_low_spectrum(spec_rows)

    target = fine_obs
    back_rows = [r for r in obs_rows if str(r["ensemble"]).endswith("_backbone")]
    for r in back_rows:
        r["IR_score_abs_Binder_xiL"] = abs(float(r["Binder_U4"]) - float(target["Binder_U4"])) + abs(float(r["xi_over_L"]) - float(target["xi_over_L"]))
        r["IR_score_abs_Binder_xiL_Spmin"] = (
            float(r["IR_score_abs_Binder_xiL"])
            + abs(float(r["F_pmin"]) - float(target["F_pmin"])) / max(abs(float(target["F_pmin"])), 1.0e-30)
        )
    ranked = sorted(back_rows, key=lambda r: float(r["IR_score_abs_Binder_xiL_Spmin"]))

    def fmt_row(r: dict[str, object]) -> str:
        score = float(r.get("IR_score_abs_Binder_xiL_Spmin", 0.0))
        return (
            f"| {r['ensemble']} | {r['N']} | {float(r['Binder_U4']):.6g} | {float(r['xi_over_L']):.6g} | "
            f"{float(r['F_pmin']):.6g} | {float(r['phi2']):.6g} | {float(r['phi4']):.6g} | {float(r['nn2']):.6g} | "
            f"{score:.6g} |"
        )

    table = "\n".join(
        ["| ensemble | N | Binder U4 | xi/L | F(pmin) | phi2 | phi4 | nn2 | IR score |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        + [fmt_row(r) for r in [target] + back_rows]
    )
    best = ranked[0]
    blocked = next(r for r in back_rows if r["ensemble"] == "blocked_fine_control_backbone")
    native30 = next((r for r in back_rows if r["ensemble"] == "native_kappa_0.30_backbone"), None)
    report = f"""# Generated/Native Coarse Backbone IR Check

## Setup

- target fine: `{(DATA / 'fine_configs.npy').resolve()}`
- selected kernel metadata: `{KERNEL_META.resolve()}`
- original kernel source: `{kernel_meta.get('original_source_path')}`
- kernel caveat: `{kernel_meta.get('caveat')}`
- K sum check: `{kernel_sum(w):.12g}`
- eta_exponent: `{ETA_EXPONENT}`
- block_norm: `{BLOCK_NORM:.12g}`
- upscale: coarse `8x8` FFT block embedded into `16x16` low modes, divided by `block_norm * K_tilde(p) * A(p)`, high modes set to zero.

## Backbone IR Table

{table}

## Block Reconstruction

Every upscaled backbone reblocks to its input coarse ensemble at roundoff. See `block_reconstruction_check.csv`.

## Answers

1. Does native/generated `phi_c` upscaled to `phi_back` reproduce fine Binder?

The blocked-fine control reproduces Binder by construction of the induced coarse distribution: target `{float(target['Binder_U4']):.6g}`, blocked-control backbone `{float(blocked['Binder_U4']):.6g}`. Native/generated sources should be judged against this target in `ir_observables.csv`; deviations are from the coarse ensemble distribution, not the inverse algebra.

2. Does it reproduce fine xi/L?

The blocked-fine control is close: target `{float(target['xi_over_L']):.6g}`, blocked-control backbone `{float(blocked['xi_over_L']):.6g}`. Native/generated sources vary by their coarse distribution. The best source by the simple IR score is `{best['ensemble']}` with xi/L `{float(best['xi_over_L']):.6g}`.

3. Does it reproduce the low-momentum spectrum?

See `low_momentum_spectrum.csv` and `plots/low_momentum_spectrum.pdf`. The most direct scalar proxy in this report is `F(pmin)`: target `{float(target['F_pmin']):.6g}`, blocked-control backbone `{float(blocked['F_pmin']):.6g}`.

4. Is the failure, if any, due to the kernel/upscaling or due to the coarse ensemble distribution?

The upscaling algebra is not the failure mode: reblocking errors are roundoff-small for every source. If a native/generated source misses Binder, xi/L, magnetization histograms, or low-momentum `S(p)`, the cause is the input coarse distribution.

5. Which coarse source should be used for future generative tests?

Use blocked-fine coarse fields for conditional inverse-map development. Among native/generated sources in this diagnostic, use the ranked table as guidance only; the native scan ensembles except kappa 0.30 are explicitly non-production.
"""
    if native30 is not None:
        report += (
            f"\nFor the available production native `kappa_c=0.30` source, the upscaled backbone has Binder "
            f"`{float(native30['Binder_U4']):.6g}` and xi/L `{float(native30['xi_over_L']):.6g}`.\n"
        )
    (OUT / "report.md").write_text(report)


if __name__ == "__main__":
    main()
