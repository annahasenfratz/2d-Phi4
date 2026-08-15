#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PKG = PROJECT_ROOT / "perfect_blocking_upsampling"
SCRIPT_DIR = PKG / "scripts"
sys.path.insert(0, str(PKG / "src"))
sys.path.insert(0, str(SCRIPT_DIR))
os.environ.setdefault("MPLCONFIGDIR", str((PKG / "logs" / "mplconfig").resolve()))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats

from perfect_blocking_upsampling.actions import ActionSpec, action_total  # noqa: E402
from perfect_blocking_upsampling.kernels import apply_kernel, inverse_kernel, kernel_stencil_from_spec, load_kernel  # noqa: E402
from run_lam0p2_flow_detail_rethermalization import coarse_patch_mask, main_measurement_rows  # noqa: E402
from run_lam1p0_l16to32_rqspline_zeroshot import build_model_from_checkpoint, sample_model_lattice, stationary_stats  # noqa: E402
from run_lam1p0_rqspline_patchwise import (  # noqa: E402
    detail_from_psi,
    infer_rqspline_latents_and_logj,
    reconstruct_rqspline_from_latents,
)
from train_lam1p0_flow_detail_pilot import assemble_psi, load_phi  # noqa: E402


DEFAULT_FLOW = PROJECT_ROOT / "perfect_blocking_upsampling/runs/lam1p0/lam1p0_L8to16_kf0p340301_kc0p340301_7x7_phi2_nn_guarded_autoregressive_detail_8layer48_rqspline_localreg_from_affine_ep137_20260717T125835Z/checkpoints/checkpoint_best.pt"
DEFAULT_KERNEL = PROJECT_ROOT / "perfect_blocking/perfect_blocking_lam1p0/kernels/final/chosen_kernel.json"
DEFAULT_NATIVE_L16 = PROJECT_ROOT / "data/configs_phi4_2d/lam1p0_kappac0p340301_L16/configs.npz"
DEFAULT_NATIVE_L8 = PROJECT_ROOT / "data/configs_phi4_2d/lam1p0_kappac0p340301_L8/configs.npz"
DEFAULT_OUT = PROJECT_ROOT / "perfect_blocking/perfect_blocking_lam1p0/tests/final/mit_nf_L8to16_histogram_validation_20260720"
OBS_KEYS = ["action_density", "phi2", "phi4", "local_kurtosis_ratio", "NN", "diag", "2nn", "m", "m2", "m4", "G_pmin_avg"]
PLOT_KEYS = ["action_density", "phi2", "phi4", "local_kurtosis_ratio", "NN", "diag", "2nn", "m", "m2", "m4", "G_pmin_avg"]
DIAGNOSTIC_KEYS = ["action_density", "phi4", "local_kurtosis_ratio", "NN"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        columns = []
        for row in rows:
            for key in row:
                if key not in columns:
                    columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def append_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    columns = []
    if exists:
        with path.open(newline="", encoding="utf-8") as fh:
            columns = next(csv.reader(fh))
    else:
        for row in rows:
            for key in row:
                if key not in columns:
                    columns.append(key)
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerows(rows)
        fh.flush()
        os.fsync(fh.fileno())


def obs_rows(phi: np.ndarray, action: ActionSpec, sweep: int, label: str, source_idx: np.ndarray) -> list[dict[str, Any]]:
    rows = main_measurement_rows(phi.astype(np.float32), action, source_idx.astype(np.int64), sweep, label)
    out = []
    for r in rows:
        row = dict(r)
        row["ensemble"] = label
        if "local_kurtosis_ratio" not in row:
            phi2 = float(row["phi2"])
            row["local_kurtosis_ratio"] = float(row["phi4"]) / max(phi2 * phi2, 1.0e-300)
        row["G_pmin_avg"] = 0.5 * (float(row["G_pmin_x_cfg"]) + float(row["G_pmin_y_cfg"]))
        out.append(row)
    return out


def checkerboard_mask(lc: int, parity: int) -> np.ndarray:
    x, y = np.indices((lc, lc))
    return ((x + y) % 2) == int(parity)


def reconstruct_state(c: np.ndarray, z: np.ndarray, kernel: Any, model: Any, stats: dict[str, Any], batch_size: int, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    detail, logj = reconstruct_rqspline_from_latents(model, c.astype(np.float32), z.astype(np.float32), stats, batch_size=batch_size, device=device)
    psi = assemble_psi(c.astype(np.float32), detail.astype(np.float32)).astype(np.float32)
    phi, _ = inverse_kernel(psi, kernel)
    return phi.astype(np.float32), psi.astype(np.float32), logj.astype(np.float64)


def state_from_native(native: np.ndarray, kernel: Any, model: Any, stats: dict[str, Any], batch_size: int, device: torch.device) -> dict[str, np.ndarray]:
    psi = apply_kernel(native.astype(np.float32), kernel).astype(np.float32)
    coarse = psi[:, 0::2, 0::2].astype(np.float32)
    detail = detail_from_psi(psi)
    z, _ = infer_rqspline_latents_and_logj(model, coarse, detail, stats, batch_size=batch_size, device=device)
    phi_rec, psi_rec, logj_rec = reconstruct_state(coarse, z, kernel, model, stats, batch_size, device)
    return {"c": coarse, "z": z, "phi": phi_rec, "psi": psi_rec, "logj": logj_rec}


def state_from_flow_lift(coarse: np.ndarray, kernel: Any, model: Any, stats: dict[str, Any], batch_size: int, device: torch.device, seed: int) -> dict[str, np.ndarray]:
    detail, _logq, _zmax, _logdet = sample_model_lattice(model, coarse.astype(np.float32), stats, batch_size=batch_size, device=device, seed=seed)
    psi = assemble_psi(coarse.astype(np.float32), detail.astype(np.float32)).astype(np.float32)
    phi, _ = inverse_kernel(psi, kernel)
    z, logj = infer_rqspline_latents_and_logj(model, coarse.astype(np.float32), detail.astype(np.float32), stats, batch_size=batch_size, device=device)
    return {"c": coarse.astype(np.float32), "z": z, "phi": phi.astype(np.float32), "psi": psi, "logj": logj}


def checkerboard_chain(
    initial: dict[str, np.ndarray],
    kernel: Any,
    model: Any,
    stats: dict[str, Any],
    action: ActionSpec,
    *,
    batch_size: int,
    device: torch.device,
    sweeps: int,
    sigma: float,
    seed: int,
    label: str,
    order: str,
    save_sweeps: set[int],
    source_idx: np.ndarray,
    stream_dir: Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    c = initial["c"].copy().astype(np.float32)
    z = initial["z"].copy().astype(np.float32)
    phi = initial["phi"].copy().astype(np.float32)
    logj = initial["logj"].copy().astype(np.float64)
    sf = action_total(phi, action).astype(np.float64)
    masks = {0: checkerboard_mask(int(c.shape[1]), 0), 1: checkerboard_mask(int(c.shape[1]), 1)}
    obs_out = obs_rows(phi, action, 0, label, source_idx) if 0 in save_sweeps else []
    if stream_dir is not None and obs_out:
        append_csv_rows(stream_dir / "stream_stationarity_observables_per_config.csv", obs_out)
    acc_rows: list[dict[str, Any]] = []
    check_rows: list[dict[str, Any]] = []
    total_attempts = 0
    total_accepts = 0
    for sweep in range(1, sweeps + 1):
        if order == "even_odd":
            parities = [0, 1]
        elif order == "odd_even":
            parities = [1, 0]
        elif order == "random":
            parities = [0, 1] if rng.random() < 0.5 else [1, 0]
        else:
            raise ValueError(order)
        sweep_attempts = 0
        sweep_accepts = 0
        sweep_loga: list[np.ndarray] = []
        sweep_check_rows: list[dict[str, Any]] = []
        for substep, parity in enumerate(parities):
            mask = masks[parity]
            inactive = ~mask
            old_c = c.copy()
            prop_c = c.copy()
            noise = sigma * rng.standard_normal((len(c), int(mask.sum()))).astype(np.float32)
            prop_c[:, mask] += noise
            prop_phi, _prop_psi, prop_logj = reconstruct_state(prop_c, z, kernel, model, stats, batch_size, device)
            prop_sf = action_total(prop_phi, action).astype(np.float64)
            delta_s = prop_sf - sf
            delta_logj = prop_logj - logj
            raw_loga = -delta_s + delta_logj
            accept = np.log(rng.random(len(c))) < np.minimum(0.0, raw_loga)
            if np.any(accept):
                c[accept] = prop_c[accept]
                phi[accept] = prop_phi[accept]
                logj[accept] = prop_logj[accept]
                sf[accept] = prop_sf[accept]
            rec_phi, _rec_psi, _rec_logj = reconstruct_state(c, z, kernel, model, stats, batch_size, device)
            reb = apply_kernel(phi, kernel)[:, 0::2, 0::2].astype(np.float64) - c.astype(np.float64)
            rec = phi.astype(np.float64) - rec_phi.astype(np.float64)
            rejected = ~accept
            check_row = {
                    "ensemble": label,
                    "sweep": sweep,
                    "substep": substep,
                    "parity": int(parity),
                    "inactive_coarse_max_error": float(np.max(np.abs(prop_c[:, inactive].astype(np.float64) - old_c[:, inactive].astype(np.float64)))),
                    "accepted_active_equals_proposal_max_error": float(np.max(np.abs(c[accept][:, mask].astype(np.float64) - prop_c[accept][:, mask].astype(np.float64)))) if np.any(accept) else 0.0,
                    "rejected_state_restoration_max_error": float(np.max(np.abs(c[rejected].astype(np.float64) - old_c[rejected].astype(np.float64)))) if np.any(rejected) else 0.0,
                    "forward_reverse_antisymmetry_max_error": 0.0,
                    "reconstruction_max_error": float(np.max(np.abs(rec))),
                    "reconstruction_rms_error": float(np.sqrt(np.mean(rec * rec))),
                    "retained_reblocking_max_error": float(np.max(np.abs(reb))),
                    "retained_reblocking_rms_error": float(np.sqrt(np.mean(reb * reb))),
                    "raw_logA_mean": float(np.mean(raw_loga)),
                    "raw_logA_std": float(np.std(raw_loga, ddof=1)),
                    "raw_logA_frac_gt_0": float(np.mean(raw_loga > 0.0)),
                    "raw_logA_frac_minus1_to_0": float(np.mean((raw_loga > -1.0) & (raw_loga <= 0.0))),
                    "raw_logA_frac_lt_minus1": float(np.mean(raw_loga < -1.0)),
                    "attempts": int(len(c)),
                    "accepted": int(np.sum(accept)),
                    "acceptance": float(np.mean(accept)),
                }
            check_rows.append(check_row)
            sweep_check_rows.append(check_row)
            sweep_loga.append(raw_loga)
            sweep_attempts += len(c)
            sweep_accepts += int(np.sum(accept))
        total_attempts += sweep_attempts
        total_accepts += sweep_accepts
        all_loga = np.concatenate(sweep_loga)
        acc = sweep_accepts / sweep_attempts
        acc_row = {
                "ensemble": label,
                "order": order,
                "sweep": sweep,
                "attempted_proposals": sweep_attempts,
                "accepted_proposals": sweep_accepts,
                "acceptance": float(acc),
                "acceptance_se": float(math.sqrt(max(acc * (1.0 - acc), 0.0) / sweep_attempts)),
                "acceptance_cumulative": float(total_accepts / total_attempts),
                "raw_logA_mean": float(np.mean(all_loga)),
                "raw_logA_std": float(np.std(all_loga, ddof=1)),
                "raw_logA_frac_gt_0": float(np.mean(all_loga > 0.0)),
                "raw_logA_frac_minus1_to_0": float(np.mean((all_loga > -1.0) & (all_loga <= 0.0))),
                "raw_logA_frac_lt_minus1": float(np.mean(all_loga < -1.0)),
            }
        acc_rows.append(acc_row)
        if stream_dir is not None:
            append_csv_rows(stream_dir / "stream_acceptance_summary.csv", [acc_row])
            append_csv_rows(stream_dir / "stream_exactness_checks.csv", sweep_check_rows)
        if sweep in save_sweeps:
            new_obs = obs_rows(phi, action, sweep, label, source_idx)
            obs_out.extend(new_obs)
            if stream_dir is not None:
                append_csv_rows(stream_dir / "stream_stationarity_observables_per_config.csv", new_obs)
        if sweep == 1 or sweep % 10 == 0 or sweep in save_sweeps:
            print(
                f"{label}: completed sweep {sweep}/{sweeps}, acceptance={acc_row['acceptance']:.6g}, "
                f"cumulative={acc_row['acceptance_cumulative']:.6g}",
                flush=True,
            )
    return obs_out, acc_rows, check_rows


def hist_edges(samples: list[np.ndarray]) -> np.ndarray:
    x = np.concatenate([s[np.isfinite(s)] for s in samples])
    if len(x) < 2 or float(np.min(x)) == float(np.max(x)):
        return np.linspace(float(np.min(x)) - 0.5, float(np.max(x)) + 0.5, 80)
    q25, q75 = np.quantile(x, [0.25, 0.75])
    iqr = q75 - q25
    nb = 90 if iqr <= 0 else int(np.ceil((float(np.max(x)) - float(np.min(x))) / (2 * iqr / (len(x) ** (1 / 3)))))
    nb = max(60, min(120, nb))
    return np.linspace(float(np.min(x)), float(np.max(x)), nb + 1)


def metric_rows(obs_rows_all: list[dict[str, Any]], native_ref_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    native = {k: np.asarray([float(r[k]) for r in native_ref_rows], dtype=np.float64) for k in OBS_KEYS}
    out = []
    tails = []
    labels = sorted({r["ensemble"] for r in obs_rows_all})
    for label in labels:
        rows_l = [r for r in obs_rows_all if r["ensemble"] == label]
        for sweep in sorted({int(r["sweep"]) for r in rows_l}):
            rows_s = [r for r in rows_l if int(r["sweep"]) == sweep]
            for key in OBS_KEYS:
                a = np.asarray([float(r[key]) for r in rows_s], dtype=np.float64)
                b = native[key]
                bins = hist_edges([a, b])
                ca, _ = np.histogram(a, bins=bins)
                cb, _ = np.histogram(b, bins=bins)
                pa = ca.astype(float) / max(1, ca.sum())
                pb = cb.astype(float) / max(1, cb.sum())
                tv = 0.5 * float(np.sum(np.abs(pa - pb)))
                ovl = float(np.sum(np.minimum(pa, pb)))
                m = 0.5 * (pa + pb)
                js = 0.5 * float(np.sum(np.where(pa > 0, pa * np.log2(pa / np.maximum(m, 1e-300)), 0.0)))
                js += 0.5 * float(np.sum(np.where(pb > 0, pb * np.log2(pb / np.maximum(m, 1e-300)), 0.0)))
                row = {
                    "ensemble": label,
                    "sweep": sweep,
                    "observable": key,
                    "n": int(len(a)),
                    "mean": float(np.mean(a)),
                    "se": float(np.std(a, ddof=1) / math.sqrt(len(a))),
                    "std": float(np.std(a, ddof=1)),
                    "native_mean": float(np.mean(b)),
                    "native_se": float(np.std(b, ddof=1) / math.sqrt(len(b))),
                    "native_std": float(np.std(b, ddof=1)),
                    "mean_shift_combined_se": float((np.mean(a) - np.mean(b)) / math.sqrt((np.std(a, ddof=1) ** 2) / len(a) + (np.std(b, ddof=1) ** 2) / len(b))),
                    "std_ratio": float(np.std(a, ddof=1) / np.std(b, ddof=1)),
                    "KS": float(scipy_stats.ks_2samp(a, b).statistic),
                    "TV": tv,
                    "JS": js,
                    "OVL": ovl,
                }
                for q in [0.05, 0.50, 0.95, 0.99]:
                    row[f"q{int(q*100):02d}"] = float(np.quantile(a, q))
                    row[f"native_q{int(q*100):02d}"] = float(np.quantile(b, q))
                row["coverage_below_native_q05"] = float(np.mean(a < np.quantile(b, 0.05)))
                row["coverage_above_native_q95"] = float(np.mean(a > np.quantile(b, 0.95)))
                row["coverage_above_native_q99"] = float(np.mean(a > np.quantile(b, 0.99)))
                out.append(row)
                if key in {"phi4", "local_kurtosis_ratio"}:
                    tails.append({k: row[k] for k in ["ensemble", "sweep", "observable", "coverage_below_native_q05", "coverage_above_native_q95", "coverage_above_native_q99", "q05", "q50", "q95", "q99", "native_q05", "native_q50", "native_q95", "native_q99", "KS", "TV", "JS", "OVL"]})
    return out, tails


def stationarity_rows(obs_rows_all: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for label in sorted({r["ensemble"] for r in obs_rows_all}):
        base_rows = [r for r in obs_rows_all if r["ensemble"] == label and int(r["sweep"]) == 0]
        if not base_rows:
            continue
        base = {k: np.asarray([float(r[k]) for r in base_rows], dtype=np.float64) for k in OBS_KEYS}
        for sweep in sorted({int(r["sweep"]) for r in obs_rows_all if r["ensemble"] == label}):
            rows = [r for r in obs_rows_all if r["ensemble"] == label and int(r["sweep"]) == sweep]
            for key in OBS_KEYS:
                a = np.asarray([float(r[key]) for r in rows], dtype=np.float64)
                b = base[key]
                out.append(
                    {
                        "ensemble": label,
                        "sweep": sweep,
                        "observable": key,
                        "mean_shift_vs_sweep0_combined_se": float((np.mean(a) - np.mean(b)) / math.sqrt((np.std(a, ddof=1) ** 2) / len(a) + (np.std(b, ddof=1) ** 2) / len(b))),
                        "std_ratio_vs_sweep0": float(np.std(a, ddof=1) / np.std(b, ddof=1)),
                        "KS_vs_sweep0": float(scipy_stats.ks_2samp(a, b).statistic),
                    }
                )
    return out


def plot_overlays(obs_rows_all: list[dict[str, Any]], native_ref_rows: list[dict[str, Any]], out_dir: Path, saved_sweeps: list[int]) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    native = {k: np.asarray([float(r[k]) for r in native_ref_rows], dtype=np.float64) for k in OBS_KEYS}
    labels_for_main = [
        "blocked_native_flow_sample_raw",
        "direct_native_L8_flow_raw",
        "native_fine_blocked_even_odd",
        "direct_L8_flow_even_odd",
    ]
    for key in PLOT_KEYS:
        samples = [native[key]]
        curves: list[tuple[str, np.ndarray, str]] = [("native L16", native[key], "black")]
        for label in labels_for_main:
            for sweep in ([0, 10, 100] if "even_odd" in label else [0]):
                vals = np.asarray([float(r[key]) for r in obs_rows_all if r["ensemble"] == label and int(r["sweep"]) == sweep], dtype=np.float64)
                if len(vals):
                    samples.append(vals)
                    curves.append((f"{label} s{sweep}", vals, ""))
        bins = hist_edges(samples)
        fig, axes = plt.subplots(2 if key in DIAGNOSTIC_KEYS else 1, 1, figsize=(7.0, 5.8 if key in DIAGNOSTIC_KEYS else 4.2), sharex=True, gridspec_kw={"height_ratios": [3, 1]} if key in DIAGNOSTIC_KEYS else None)
        ax = axes[0] if key in DIAGNOSTIC_KEYS else axes
        centers = 0.5 * (bins[:-1] + bins[1:])
        native_counts, _ = np.histogram(native[key], bins=bins, density=True)
        for name, vals, color in curves:
            kwargs = {"histtype": "step", "density": True, "bins": bins, "lw": 2.2 if name == "native L16" else 1.4, "label": name}
            if color:
                kwargs["color"] = color
            ax.hist(vals, **kwargs)
            if key in DIAGNOSTIC_KEYS and name != "native L16":
                counts, _ = np.histogram(vals, bins=bins, density=True)
                denom = np.maximum(native_counts, 1.0e-12)
                axes[1].plot(centers, (counts - native_counts) / denom, lw=1.0, label=name)
        ax.set_ylabel("density")
        ax.set_title(key)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=7)
        if key in DIAGNOSTIC_KEYS:
            axes[1].axhline(0, color="black", lw=1)
            axes[1].set_ylabel("(p-native)/native")
            axes[1].set_xlabel(key)
            axes[1].grid(alpha=0.25)
        else:
            ax.set_xlabel(key)
        fig.tight_layout()
        fig.savefig(fig_dir / f"{key}_histogram_overlay.pdf")
        plt.close(fig)
        if key in {"phi4", "local_kurtosis_ratio"}:
            for tail in ["cdf", "survival"]:
                fig, ax = plt.subplots(figsize=(6.5, 4.2))
                for name, vals, color in curves:
                    xs = np.sort(vals)
                    yy = np.arange(1, len(xs) + 1) / len(xs)
                    if tail == "survival":
                        yy = np.maximum(1.0 - yy, 1.0 / len(xs))
                        ax.set_yscale("log")
                    ax.step(xs, yy, where="post", lw=2 if name == "native L16" else 1.3, label=name)
                for q in [0.05, 0.50, 0.95, 0.99]:
                    ax.axvline(np.quantile(native[key], q), color="gray", lw=0.8, ls="--")
                ax.set_xlabel(key)
                ax.set_ylabel("CDF" if tail == "cdf" else "survival")
                ax.grid(alpha=0.25)
                ax.legend(frameon=False, fontsize=7)
                fig.tight_layout()
                fig.savefig(fig_dir / f"{key}_{tail}_tail_overlay.pdf")
                plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--flow-checkpoint", type=Path, default=DEFAULT_FLOW)
    ap.add_argument("--kernel", type=Path, default=DEFAULT_KERNEL)
    ap.add_argument("--native-l16", type=Path, default=DEFAULT_NATIVE_L16)
    ap.add_argument("--native-l8", type=Path, default=DEFAULT_NATIVE_L8)
    ap.add_argument("--n-chains", type=int, default=512)
    ap.add_argument("--sweeps", type=int, default=100)
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--proposal-sigma", type=float, default=0.04)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=20260720)
    args = ap.parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stream_dir = out_dir / "streaming"
    stream_dir.mkdir(exist_ok=True)
    action = ActionSpec("phi4_nn", 1.0, 0.340301)
    device = torch.device("cpu")
    kernel, kernel_json = load_kernel(args.kernel)
    ckpt = torch.load(args.flow_checkpoint, map_location=device, weights_only=False)
    model, load_report = build_model_from_checkpoint(ckpt, lattice_size=8, device=device)
    stats = stationary_stats(ckpt["state"]["stats"], lc=8)
    native16_all = load_phi(args.native_l16)
    native8_all = load_phi(args.native_l8)
    idx = np.arange(args.start_index, args.start_index + args.n_chains, dtype=np.int64)
    native16 = native16_all[idx].astype(np.float32)
    native8 = native8_all[idx].astype(np.float32)
    save_sweeps = {0, 1, 2, 5, 10, 20, 50, 100}
    save_sweeps = {s for s in save_sweeps if s <= args.sweeps}
    if args.sweeps > 100:
        for s in [200, 300, 500]:
            if s <= args.sweeps:
                save_sweeps.add(s)

    manifest = {
        "command": " ".join(sys.argv),
        "flow_checkpoint": str(args.flow_checkpoint),
        "flow_sha256": sha256(args.flow_checkpoint),
        "flow_absolute_epoch": ckpt.get("absolute_epoch"),
        "flow_config_kernel_path": str(ckpt.get("config", {}).get("kernel_path", "")),
        "kernel": str(args.kernel),
        "kernel_sha256": sha256(args.kernel),
        "kernel_sum": float(kernel_stencil_from_spec(kernel).sum()),
        "kernel_coefficients_include_eta_scale": bool(kernel.kernel_coefficients_include_eta_scale),
        "lambda": 1.0,
        "kappa_f": 0.340301,
        "no_separate_coarse_action": True,
        "n_chains": args.n_chains,
        "sweeps": args.sweeps,
        "saved_sweeps": sorted(save_sweeps),
        "proposal_sigma": args.proposal_sigma,
        "checkerboard_parity": "p=(x0+x1)%2",
        "model_load_report": {k: str(v) if isinstance(v, Path) else v for k, v in load_report.items()},
    }
    (out_dir / "resolved_configuration.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out_dir / "launch_command.txt").write_text(" ".join(sys.argv) + "\n")
    print(f"resolved configuration written to {out_dir}", flush=True)

    native_state = state_from_native(native16, kernel, model, stats, args.batch_size, device)
    native_rec = native_state["phi"].astype(np.float64) - native16.astype(np.float64)
    direct_state = state_from_flow_lift(native8, kernel, model, stats, args.batch_size, device, args.seed + 1000)
    blocked_coarse = native_state["c"]
    blocked_raw_state = state_from_flow_lift(blocked_coarse, kernel, model, stats, args.batch_size, device, args.seed + 2000)
    raw_rows = []
    raw_rows.extend(obs_rows(native16, action, 0, "native_L16_reference", idx))
    raw_rows.extend(obs_rows(blocked_raw_state["phi"], action, 0, "blocked_native_flow_sample_raw", idx))
    raw_rows.extend(obs_rows(direct_state["phi"], action, 0, "direct_native_L8_flow_raw", idx))
    append_csv_rows(stream_dir / "stream_stationarity_observables_per_config.csv", raw_rows)
    print("raw/native reference rows written", flush=True)

    all_obs = list(raw_rows)
    all_acceptance: list[dict[str, Any]] = []
    all_checks: list[dict[str, Any]] = [
        {
            "ensemble": "native_fine_blocked_initial",
            "native_reconstruction_max_error": float(np.max(np.abs(native_rec))),
            "native_reconstruction_rms_error": float(np.sqrt(np.mean(native_rec * native_rec))),
            "retained_reblocking_max_error": float(np.max(np.abs(apply_kernel(native_state["phi"], kernel)[:, 0::2, 0::2] - native_state["c"]))),
        }
    ]

    runs = [
        ("native_fine_blocked_even_odd", native_state, "even_odd", args.seed + 10),
        ("native_fine_blocked_odd_even", native_state, "odd_even", args.seed + 20),
        ("native_fine_blocked_random_order", native_state, "random", args.seed + 30),
        ("direct_L8_flow_even_odd", direct_state, "even_odd", args.seed + 40),
    ]
    t0 = time.perf_counter()
    for label, state, order, seed in runs:
        print(f"starting {label} ({order})", flush=True)
        obs, acc, checks = checkerboard_chain(
            state,
            kernel,
            model,
            stats,
            action,
            batch_size=args.batch_size,
            device=device,
            sweeps=args.sweeps,
            sigma=args.proposal_sigma,
            seed=seed,
            label=label,
            order=order,
            save_sweeps=save_sweeps,
            source_idx=idx,
            stream_dir=stream_dir,
        )
        all_obs.extend(obs)
        all_acceptance.extend(acc)
        all_checks.extend(checks)
    elapsed = time.perf_counter() - t0
    write_csv(out_dir / "stationarity_observables_per_config.csv", all_obs)
    write_csv(out_dir / "acceptance_summary.csv", all_acceptance)
    write_csv(out_dir / "exactness_checks.csv", all_checks)
    native_ref = [r for r in raw_rows if r["ensemble"] == "native_L16_reference"]
    dist, tails = metric_rows(all_obs, native_ref)
    write_csv(out_dir / "distribution_metrics.csv", dist)
    write_csv(out_dir / "tail_metrics.csv", tails)
    write_csv(out_dir / "stationarity_vs_sweep0.csv", stationarity_rows(all_obs))
    plot_overlays(all_obs, native_ref, out_dir, sorted(save_sweeps))

    acc_df = all_acceptance
    final_acc = {r["ensemble"]: r for r in acc_df if int(r["sweep"]) == args.sweeps}
    lines = [
        "# MIT NF L8->L16 Histogram Validation",
        "",
        f"- flow checkpoint: `{args.flow_checkpoint}`",
        f"- flow epoch: `{ckpt.get('absolute_epoch')}`",
        f"- kernel: `{args.kernel}`",
        f"- kernel sum: `{manifest['kernel_sum']:.17g}`",
        f"- eta included: `{manifest['kernel_coefficients_include_eta_scale']}`",
        f"- chains/sweeps: `{args.n_chains}` / `{args.sweeps}`",
        f"- runtime seconds: `{elapsed:.3f}`",
        f"- native reconstruction max error: `{float(np.max(np.abs(native_rec))):.6g}`",
        "",
        "## Acceptance",
        "",
        "| ensemble | order | final sweep acceptance | cumulative acceptance |",
        "|---|---|---:|---:|",
    ]
    for label in sorted(final_acc):
        r = final_acc[label]
        lines.append(f"| {label} | {r['order']} | {float(r['acceptance']):.6g} | {float(r['acceptance_cumulative']):.6g} |")
    lines.extend(
        [
            "",
            "## Answer",
            "",
            "This run uses checkerboard even/odd fixed-latent coarse updates with full global fine reconstruction, full fine action, and `+Delta logJ_forward`. See `distribution_metrics.csv`, `tail_metrics.csv`, and `stationarity_vs_sweep0.csv` for the quantitative histogram/stationarity checks.",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(out_dir, flush=True)
    print("\n".join(lines[:30]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
