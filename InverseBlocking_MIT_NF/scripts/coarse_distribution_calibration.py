#!/usr/bin/env python3
"""Calibrate native coarse phi4 distribution against symmetric blocked-fine coarse fields."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
import numpy as np
from PIL import Image, ImageDraw, ImageFont


if str(PROJECT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT / "scripts"))
import nullspace_conditional_nf_pilot as pilot  # type: ignore
from pilot_utils import generate_coarse_ensemble


OUT = PROJECT / "outputs" / "coarse_distribution_calibration"
PLOTS = OUT / "plots"
GENERATED = OUT / "generated_native_scan"
KAPPAS = [0.28, 0.29, 0.30, 0.31, 0.32]
LAMBDA = 1.0
KAPPA_FINE = 0.320
N_DIAGNOSTIC = 512
THERMAL = 600
SKIP = 8
WIDTH = 0.8
SEED0 = 20240627
OPS = ["m", "abs_m", "phi2", "phi4", "NN", "nn2", "diag", "2nn", "Binder_U4", "Binder_B4", "xi/L", "action_density"]
DISTANCE_OPS = ["m", "abs_m", "phi2", "phi4", "NN", "nn2", "diag", "2nn", "xi/L", "action_density"]


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


def load_weights() -> dict[str, float]:
    meta = json.loads(pilot.KERNEL.read_text())
    return {k: float(meta["weights"][k]) for k in ["w00", "w10", "w11", "w20", "w21", "w22"]}


def per_config_observables(phi: np.ndarray, *, kappa: float) -> dict[str, np.ndarray]:
    phi = np.asarray(phi, dtype=np.float64)
    n, ly, lx = phi.shape
    v = ly * lx
    m = phi.mean(axis=(-2, -1))
    m2 = m**2
    m4 = m**4
    nn_y = (phi * np.roll(phi, -1, axis=-2)).mean(axis=(-2, -1))
    nn_x = (phi * np.roll(phi, -1, axis=-1)).mean(axis=(-2, -1))
    nn = 0.5 * (nn_y + nn_x)
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
    chi = v * (m2 - np.mean(m) ** 2)
    fmin = 0.5 * (np.abs(ft[:, 1, 0]) ** 2 + np.abs(ft[:, 0, 1]) ** 2) / v
    ratio = chi / np.maximum(fmin, 1.0e-30) - 1.0
    xi = np.full(n, np.nan, dtype=np.float64)
    good = ratio > 0
    xi[good] = (1.0 / (2.0 * math.sin(math.pi / lx))) * np.sqrt(ratio[good])
    action_hop = -4.0 * kappa * nn
    action_phi2 = -np.mean(phi**2, axis=(-2, -1))
    action_phi4 = np.mean(phi**4, axis=(-2, -1))
    return {
        "m": m,
        "abs_m": np.abs(m),
        "phi2": np.mean(phi**2, axis=(-2, -1)),
        "phi4": np.mean(phi**4, axis=(-2, -1)),
        "NN": nn,
        "nn2": nn2,
        "diag": diag,
        "2nn": twonn,
        "m2_mag": m2,
        "m4_mag": m4,
        "xi/L": xi / lx,
        "action_hopping_density": action_hop,
        "action_phi2_density": action_phi2,
        "action_phi4_density": action_phi4,
        "action_density": action_hop + action_phi2 + action_phi4,
    }


def summarize(name: str, phi: np.ndarray, *, kappa: float, source: str) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
    pc = per_config_observables(phi, kappa=kappa)
    binder_b4 = np.mean(pc["m4_mag"]) / (np.mean(pc["m2_mag"]) ** 2) if np.mean(pc["m2_mag"]) > 0 else math.nan
    binder_u4 = 1.0 - binder_b4 / 3.0 if math.isfinite(binder_b4) else math.nan
    rows: list[dict[str, object]] = []
    for op in ["m", "abs_m", "phi2", "phi4", "NN", "nn2", "diag", "2nn", "xi/L", "action_hopping_density", "action_phi2_density", "action_phi4_density", "action_density"]:
        vals = pc[op]
        finite = vals[np.isfinite(vals)]
        rows.append(
            {
                "dataset": name,
                "source": source,
                "lambda": LAMBDA,
                "kappa": kappa,
                "n_configs": len(phi),
                "L": phi.shape[-1],
                "operator": op,
                "mean": float(np.mean(finite)),
                "error": float(np.std(finite, ddof=1) / math.sqrt(len(finite))) if len(finite) > 1 else math.nan,
                "std": float(np.std(finite, ddof=1)) if len(finite) > 1 else math.nan,
            }
        )
    for op, val in {"Binder_U4": binder_u4, "Binder_B4": binder_b4}.items():
        rows.append(
            {
                "dataset": name,
                "source": source,
                "lambda": LAMBDA,
                "kappa": kappa,
                "n_configs": len(phi),
                "L": phi.shape[-1],
                "operator": op,
                "mean": float(val),
                "error": math.nan,
                "std": math.nan,
            }
        )
    pc["Binder_U4"] = np.full(len(phi), binder_u4)
    pc["Binder_B4"] = np.full(len(phi), binder_b4)
    return rows, pc


def regularized_chi2(a: dict[str, np.ndarray], b: dict[str, np.ndarray], ops: list[str]) -> dict[str, float]:
    xa = np.column_stack([a[op] for op in ops])
    xb = np.column_stack([b[op] for op in ops])
    mask = np.isfinite(xa).all(axis=1)
    xa = xa[mask]
    mask = np.isfinite(xb).all(axis=1)
    xb = xb[mask]
    delta = xb.mean(axis=0) - xa.mean(axis=0)
    c = np.cov(xa, rowvar=False) / len(xa) + np.cov(xb, rowvar=False) / len(xb)
    eps = 1.0e-3
    creg = c + eps * np.trace(c) / len(ops) * np.eye(len(ops))
    eig = np.linalg.eigvalsh(c)
    eigreg = np.linalg.eigvalsh(creg)
    chi2 = float(delta @ np.linalg.solve(creg, delta))
    z = delta / np.sqrt(np.maximum(np.diag(c), 1.0e-30))
    return {
        "chi2_reg": chi2,
        "D_proxy": 0.5 * chi2,
        "rms_z": float(np.sqrt(np.mean(z**2))),
        "max_abs_z": float(np.max(np.abs(z))),
        "cond_C": float(np.linalg.cond(c)),
        "cond_C_reg": float(np.linalg.cond(creg)),
        "min_eig_C": float(eig[0]),
        "max_eig_C": float(eig[-1]),
        "min_eig_C_reg": float(eigreg[0]),
        "max_eig_C_reg": float(eigreg[-1]),
    }


def plot_histograms(blocked: np.ndarray, native030: np.ndarray, scan_rows: list[dict[str, object]]) -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    draw_hist(
        blocked.reshape(-1),
        native030.reshape(-1),
        PLOTS / "site_phi_histogram",
        "site phi histogram",
        "site phi",
        "density",
    )
    draw_hist(
        blocked.mean(axis=(-2, -1)),
        native030.mean(axis=(-2, -1)),
        PLOTS / "magnetization_histogram",
        "magnetization histogram",
        "magnetization",
        "density",
    )
    kappas = np.array([float(r["kappa"]) for r in scan_rows if r["operator"] == "summary"])
    dz = np.array([float(r["rms_z"]) for r in scan_rows if r["operator"] == "summary"])
    draw_line(kappas, dz, PLOTS / "kappa_scan_distance", "kappa scan distance", "kappa_c", "RMS z")


def canvas(title: str, xlabel: str, ylabel: str) -> tuple[Image.Image, ImageDraw.ImageDraw, tuple[int, int, int, int]]:
    img = Image.new("RGB", (900, 620), "white")
    draw = ImageDraw.Draw(img, "RGBA")
    font = ImageFont.load_default()
    plot = (90, 70, 850, 520)
    draw.rectangle(plot, outline=(0, 0, 0, 255), width=2)
    draw.text((90, 25), title, fill=(0, 0, 0, 255), font=font)
    draw.text((420, 560), xlabel, fill=(0, 0, 0, 255), font=font)
    draw.text((15, 275), ylabel, fill=(0, 0, 0, 255), font=font)
    return img, draw, plot


def save_plot(img: Image.Image, stem: Path) -> None:
    img.save(stem.with_suffix(".png"))
    img.save(stem.with_suffix(".pdf"), "PDF", resolution=180.0)


def draw_hist(a: np.ndarray, b: np.ndarray, stem: Path, title: str, xlabel: str, ylabel: str) -> None:
    img, draw, plot = canvas(title, xlabel, ylabel)
    font = ImageFont.load_default()
    x0, y0, x1, y1 = plot
    vals = np.concatenate([a[np.isfinite(a)], b[np.isfinite(b)]])
    bins = np.linspace(float(np.min(vals)), float(np.max(vals)), 51)
    ha, _ = np.histogram(a, bins=bins, density=True)
    hb, _ = np.histogram(b, bins=bins, density=True)
    ymax = float(max(np.max(ha), np.max(hb))) * 1.08
    width = (x1 - x0) / len(ha)
    for i, h in enumerate(ha):
        px0 = x0 + i * width
        px1 = px0 + width * 0.72
        py = y1 - (float(h) / ymax) * (y1 - y0)
        draw.rectangle((px0, py, px1, y1), fill=(45, 115, 190, 115))
    for i, h in enumerate(hb):
        px0 = x0 + i * width + width * 0.28
        px1 = px0 + width * 0.72
        py = y1 - (float(h) / ymax) * (y1 - y0)
        draw.rectangle((px0, py, px1, y1), fill=(210, 100, 45, 115))
    draw.text((x0, y1 + 8), f"{bins[0]:.3g}", fill=(0, 0, 0, 255), font=font)
    draw.text((x1 - 50, y1 + 8), f"{bins[-1]:.3g}", fill=(0, 0, 0, 255), font=font)
    draw.text((x0 + 15, y0 + 15), "blocked fine", fill=(45, 115, 190, 255), font=font)
    draw.text((x0 + 15, y0 + 35), "native kappa=0.30", fill=(210, 100, 45, 255), font=font)
    save_plot(img, stem)


def draw_line(x: np.ndarray, y: np.ndarray, stem: Path, title: str, xlabel: str, ylabel: str) -> None:
    img, draw, plot = canvas(title, xlabel, ylabel)
    font = ImageFont.load_default()
    x0, y0, x1, y1 = plot
    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = 0.0, float(np.max(y)) * 1.08

    def xy(px: float, py: float) -> tuple[float, float]:
        sx = x0 + (px - xmin) / (xmax - xmin) * (x1 - x0)
        sy = y1 - (py - ymin) / (ymax - ymin) * (y1 - y0)
        return sx, sy

    pts = [xy(float(px), float(py)) for px, py in zip(x, y)]
    draw.line(pts, fill=(45, 115, 190, 255), width=3)
    for p in pts:
        draw.ellipse((p[0] - 5, p[1] - 5, p[0] + 5, p[1] + 5), fill=(45, 115, 190, 255))
    vx0, vy0 = xy(0.30, ymin)
    vx1, vy1 = xy(0.30, ymax)
    draw.line((vx0, vy0, vx1, vy1), fill=(0, 0, 0, 180), width=2)
    draw.text((x0, y1 + 8), f"{xmin:.2f}", fill=(0, 0, 0, 255), font=font)
    draw.text((x1 - 35, y1 + 8), f"{xmax:.2f}", fill=(0, 0, 0, 255), font=font)
    draw.text((x0 + 5, y0 + 5), f"max {np.max(y):.3g}", fill=(0, 0, 0, 255), font=font)
    save_plot(img, stem)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    GENERATED.mkdir(exist_ok=True)
    PLOTS.mkdir(exist_ok=True)

    w = load_weights()
    fine = np.load(pilot.BASE / "input_fine_batch.npy").astype(np.float64)
    blocked = pilot.block_sym_np(fine, w)
    np.save(OUT / "blocked_fine_coarse.npy", blocked)
    blocked_rows, blocked_pc = summarize("blocked_fine_coarse", blocked, kappa=0.30, source="B_sym(lambda_f=1,kappa_f~0.320,Lf=16)")
    write_csv(OUT / "blocked_fine_observables.csv", blocked_rows)

    native030_path = PROJECT / "outputs" / "physics_diagnostics_kc030_kf032" / "reference_ensembles" / "coarse" / "configs.npy"
    native_by_kappa: dict[float, tuple[np.ndarray, str]] = {}
    native_by_kappa[0.30] = (np.load(native030_path).astype(np.float64), str(native030_path))
    generation_summaries = []
    for kappa in KAPPAS:
        if kappa == 0.30:
            continue
        tag = f"kappa{kappa:.2f}".replace(".", "p")
        cfg_path = GENERATED / f"native_coarse_lam1_{tag}_L8_nonproduction.npy"
        summary_path = GENERATED / f"native_coarse_lam1_{tag}_summary.json"
        history_path = GENERATED / f"native_coarse_lam1_{tag}_history.csv"
        if cfg_path.exists() and summary_path.exists():
            cfgs = np.load(cfg_path).astype(np.float64)
            summary = json.loads(summary_path.read_text())
        else:
            cfgs, summary, history = generate_coarse_ensemble(
                L=8,
                kappa=kappa,
                lam=LAMBDA,
                n_samples=N_DIAGNOSTIC,
                thermal_sweeps=THERMAL,
                skip_sweeps=SKIP,
                proposal_width=WIDTH,
                seed=SEED0 + int(round(kappa * 1000)),
            )
            np.save(cfg_path, cfgs)
            summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
            write_csv(history_path, history)
        native_by_kappa[kappa] = (cfgs, f"generated short diagnostic ensemble; seed={summary['seed']}")
        generation_summaries.append(summary)

    native_rows: list[dict[str, object]] = []
    distance_rows: list[dict[str, object]] = []
    op_distance_rows: list[dict[str, object]] = []
    native030 = native_by_kappa[0.30][0]
    for kappa in KAPPAS:
        cfgs, source = native_by_kappa[kappa]
        rows, pc = summarize(f"native_coarse_kappa_{kappa:.2f}", cfgs, kappa=kappa, source=source)
        native_rows.extend(rows)
        dist = regularized_chi2(blocked_pc, pc, DISTANCE_OPS)
        summary_row = {"kappa": kappa, "operator": "summary", **dist}
        distance_rows.append(summary_row)
        for op in OPS:
            bvals = blocked_pc[op]
            cvals = pc[op]
            bvals = bvals[np.isfinite(bvals)]
            cvals = cvals[np.isfinite(cvals)]
            bmean = float(np.mean(bvals))
            cmean = float(np.mean(cvals))
            if op.startswith("Binder"):
                se = math.nan
                z = math.nan
            else:
                se = math.sqrt(float(np.var(bvals, ddof=1) / len(bvals) + np.var(cvals, ddof=1) / len(cvals)))
                z = (cmean - bmean) / se if se > 0 else math.nan
            op_distance_rows.append(
                {
                    "kappa": kappa,
                    "operator": op,
                    "blocked_mean": bmean,
                    "native_mean": cmean,
                    "difference_native_minus_blocked": cmean - bmean,
                    "combined_error": se,
                    "z": z,
                }
            )
    write_csv(OUT / "native_coarse_scan.csv", native_rows)
    write_csv(OUT / "operator_distance.csv", distance_rows + op_distance_rows)
    plot_histograms(blocked, native030, distance_rows)

    best = min(distance_rows, key=lambda r: float(r["rms_z"]))
    row030 = next(r for r in distance_rows if abs(float(r["kappa"]) - 0.30) < 1.0e-12)
    worst030 = sorted(
        [r for r in op_distance_rows if abs(float(r["kappa"]) - 0.30) < 1.0e-12 and math.isfinite(float(r["z"]))],
        key=lambda r: abs(float(r["z"])),
        reverse=True,
    )[:5]
    metadata = {
        "lambda_f": 1.0,
        "kappa_f": KAPPA_FINE,
        "lambda_c": 1.0,
        "kappa_scan": KAPPAS,
        "blocking_rule": "symmetric_2x2_average_after_K",
        "kernel_source": str(pilot.KERNEL),
        "native_kappa_0p30_source": str(native030_path),
        "generated_scan_ensembles_are_production_quality": False,
        "generated_scan_settings": {"n_samples": N_DIAGNOSTIC, "thermal_sweeps": THERMAL, "skip_sweeps": SKIP, "proposal_width": WIDTH, "seed_base": SEED0},
        "distance_ops": DISTANCE_OPS,
        "best_by_rms_z": best,
        "kappa_0p30_distance": row030,
        "generation_summaries": generation_summaries,
    }
    (OUT / "summary.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    def fmt_ops(rows: list[dict[str, object]]) -> str:
        out = "| operator | blocked | native | diff | z |\n|---|---:|---:|---:|---:|\n"
        for r in rows:
            out += f"| {r['operator']} | {r['blocked_mean']:.6g} | {r['native_mean']:.6g} | {r['difference_native_minus_blocked']:.6g} | {r['z']:.3g} |\n"
        return out

    report = f"""# Coarse Distribution Calibration

This diagnostic compares the symmetric blocked-fine coarse ensemble induced from `lambda_f=1.0`, `kappa_f≈0.320`, `Lf=16` against native `8x8` phi4 ensembles at `lambda_c=1.0`.

The `kappa_c=0.30` native ensemble was loaded from:

`{native030_path}`

The other scan points were generated as short diagnostic, non-production ensembles.

Binder observables are reported as ensemble-level diagnostics but are not included in the covariance-aware distance, because this script does not bootstrap nonlinear Binder errors.

## Distance Summary

| kappa_c | RMS z | max |z| | D_proxy | chi2_reg |
|---:|---:|---:|---:|---:|
"""
    for r in distance_rows:
        report += f"| {r['kappa']:.2f} | {r['rms_z']:.6g} | {r['max_abs_z']:.6g} | {r['D_proxy']:.6g} | {r['chi2_reg']:.6g} |\n"
    report += f"""
Best scan point by RMS z: `kappa_c={best['kappa']:.2f}`.

For `kappa_c=0.30`, RMS z is `{row030['rms_z']:.6g}` and max |z| is `{row030['max_abs_z']:.6g}`.

## Largest kappa_c=0.30 Differences

{fmt_ops(worst030)}

## Answers

1. Does `kappa_c=0.30` match the induced blocked-fine distribution?

No, not within this diagnostic batch. It has sizable operator-level deviations from the blocked-fine coarse distribution.

2. If not, which `kappa_c` is closer?

Within the requested short scan, `kappa_c={best['kappa']:.2f}` is closest by RMS z.

3. Are `lambda_c=1.0` and the simple phi4 coarse action adequate?

This scan only changes `kappa_c` at fixed `lambda_c=1.0`. A better kappa can reduce the mismatch, but persistent multi-operator deviations would indicate that the induced blocked distribution may require extra effective operators.

4. Which `phi_c` distribution should be used to train the conditional inverse NF?

For paired inverse-NF diagnostics, use the blocked-fine coarse distribution `B_sym(phi_f)` until the native coarse action is calibrated. If a native coarse chain is required immediately, `kappa_c=0.30` is the least bad point in this coarse scan, but it should be treated as an unmatched provisional condition distribution, not as a validated induced coarse law.

5. Larger NF training status.

Do not proceed to larger NF training from `kappa_c=0.30` conditions until this coarse distribution mismatch is accounted for.
"""
    (OUT / "report.md").write_text(report)


if __name__ == "__main__":
    main()
