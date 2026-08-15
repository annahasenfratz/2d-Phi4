#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import platform
import random
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PKG = PROJECT_ROOT / "perfect_blocking_upsampling"
SCRIPT_DIR = PKG / "scripts"
sys.path.insert(0, str(PKG / "src"))
sys.path.insert(0, str(SCRIPT_DIR))
os.environ.setdefault("MPLCONFIGDIR", str((PKG / "logs" / "mplconfig").resolve()))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

from perfect_blocking_upsampling.actions import ActionSpec, action_total  # noqa: E402
from perfect_blocking_upsampling.kernels import load_kernel as load_kernel_spec  # noqa: E402
from run_lam0p2_flow_detail_rethermalization import PATCH_HISTORY_FIELDS  # noqa: E402
from run_lam0p2_residual_flow_patch_chain import StreamingCsv, patch_correct  # noqa: E402
from run_lam1p0_l16to32_rqspline_zeroshot import (  # noqa: E402
    AffineARDetailFlow,
    ResidualSplineARDetailFlow,
    SplineARDetailFlow,
    build_model_from_checkpoint,
    log_prob_model_lattice,
    sample_model_lattice,
)
from train_lam1p0_autoregressive_detail_flow import DEFAULT_WEIGHTS, parse_weights  # noqa: E402
from train_lam1p0_flow_detail_localreg import torch_inverse_kernel, torch_kernel_fft, torch_observables  # noqa: E402
from train_lam1p0_flow_detail_pilot import (  # noqa: E402
    ETA_SCALE,
    apply_kernel,
    assemble_psi,
    inverse_kernel,
    load_kernel_matrix,
    load_phi,
    per_config_rows,
    split_pairs,
    write_csv,
    write_json,
)


OBS_KEYS = ["action_density", "phi2", "phi4", "local_kurtosis_ratio", "NN", "2nn", "diag", "m2", "m4", "G_pmin_avg"]


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def json_clean(obj: Any) -> Any:
    return json.loads(json.dumps(obj, default=str))


def rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python_random": random.getstate(),
        "numpy_random": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def scalar_sector_stats(coarse: np.ndarray, detail: np.ndarray, train_idx: np.ndarray) -> dict[str, Any]:
    c = coarse[train_idx].astype(np.float64)
    d = detail[train_idx].astype(np.float64)
    return {
        "coarse_mean": float(np.mean(c)),
        "coarse_std": float(max(np.std(c), 1.0e-6)),
        "detail_mean": np.mean(d, axis=(0, 2, 3)).astype(np.float32),
        "detail_std": np.maximum(np.std(d, axis=(0, 2, 3)), 1.0e-6).astype(np.float32),
        "method": "matched native L32 blocked-to-L16 training statistics: scalar coarse mean/std and per-sector detail mean/std",
        "lattice_size": 16,
    }


def standardize_arrays(coarse: np.ndarray, detail: np.ndarray, stats: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    c = ((coarse - float(stats["coarse_mean"])) / float(stats["coarse_std"])).astype(np.float32)
    mean = np.asarray(stats["detail_mean"], dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.asarray(stats["detail_std"], dtype=np.float32).reshape(1, 3, 1, 1)
    d = ((detail - mean) / std).astype(np.float32)
    return c, d


def normalize_stats_metadata(stats: dict[str, Any]) -> dict[str, Any]:
    out = dict(stats)
    for key in ["detail_mean", "detail_std"]:
        value = out.get(key)
        if isinstance(value, str):
            out[key] = np.fromstring(value.strip().strip("[]"), sep=" ", dtype=np.float32)
        else:
            out[key] = np.asarray(value, dtype=np.float32)
    out["coarse_mean"] = float(out["coarse_mean"])
    out["coarse_std"] = float(out["coarse_std"])
    return out


def physical_log_jac_const(stats: dict[str, Any], lc: int) -> float:
    return -float(lc * lc * np.sum(np.log(np.asarray(stats["detail_std"], dtype=np.float64).reshape(3))))


def native_targets(phi: np.ndarray, idx: np.ndarray) -> dict[str, dict[str, float]]:
    action = ActionSpec("phi4_nn", 1.0, 0.340301)
    rows, grows = per_config_rows(phi[idx], action, "native_L32")
    vals: dict[str, np.ndarray] = {}
    for key in ["action_density", "phi2", "phi4", "local_kurtosis_ratio", "NN", "2nn", "diag", "m2", "m4"]:
        vals[key] = np.asarray([float(r[key]) for r in rows], dtype=np.float64)
    vals["G_pmin_avg"] = np.asarray([float(r["G_pmin_avg"]) for r in grows], dtype=np.float64)
    return {k: {"mean": float(np.mean(v)), "std": float(max(np.std(v, ddof=1), 1.0e-6))} for k, v in vals.items()}


def native_phi2_support_targets(phi: np.ndarray, idx: np.ndarray) -> dict[str, float]:
    action = ActionSpec("phi4_nn", 1.0, 0.340301)
    rows, _grows = per_config_rows(phi[idx], action, "native_L32")
    vals = np.asarray([float(r["phi2"]) for r in rows], dtype=np.float64)
    std = float(max(np.std(vals, ddof=1), 1.0e-6))
    return {
        "mean": float(np.mean(vals)),
        "std": std,
        "q90": float(np.quantile(vals, 0.90)),
        "q95": float(np.quantile(vals, 0.95)),
        "q99": float(np.quantile(vals, 0.99)),
        "tail_fraction_q90": 0.10,
        "tail_fraction_q95": 0.05,
        "tail_fraction_q99": 0.01,
        "smooth_tail_width": float(max(0.05 * std, 1.0e-4)),
    }


def native_observable_support_targets(phi: np.ndarray, idx: np.ndarray, observable: str) -> dict[str, float]:
    action = ActionSpec("phi4_nn", 1.0, 0.340301)
    rows, grows = per_config_rows(phi[idx], action, "native_L32")
    if observable == "G_pmin_avg":
        vals = np.asarray([float(r["G_pmin_avg"]) for r in grows], dtype=np.float64)
    else:
        vals = np.asarray([float(r[observable]) for r in rows], dtype=np.float64)
    std = float(max(np.std(vals, ddof=1), 1.0e-6))
    return {
        "mean": float(np.mean(vals)),
        "std": std,
        "q01": float(np.quantile(vals, 0.01)),
        "q05": float(np.quantile(vals, 0.05)),
        "q10": float(np.quantile(vals, 0.10)),
        "q90": float(np.quantile(vals, 0.90)),
        "q95": float(np.quantile(vals, 0.95)),
        "q99": float(np.quantile(vals, 0.99)),
        "tail_fraction_q01": 0.01,
        "tail_fraction_q05": 0.05,
        "tail_fraction_q10": 0.10,
        "tail_fraction_q90": 0.10,
        "tail_fraction_q95": 0.05,
        "tail_fraction_q99": 0.01,
        "smooth_tail_width": float(max(0.05 * std, 1.0e-4)),
    }


def native_observable_shape_targets(phi: np.ndarray, idx: np.ndarray, observable: str) -> dict[str, float]:
    action = ActionSpec("phi4_nn", 1.0, 0.340301)
    rows, grows = per_config_rows(phi[idx], action, "native_L32")
    if observable == "G_pmin_avg":
        vals = np.asarray([float(r["G_pmin_avg"]) for r in grows], dtype=np.float64)
    else:
        vals = np.asarray([float(r[observable]) for r in rows], dtype=np.float64)
    std = float(max(np.std(vals, ddof=1), 1.0e-6))
    return {
        "mean": float(np.mean(vals)),
        "std": std,
        "q05": float(np.quantile(vals, 0.05)),
        "q25": float(np.quantile(vals, 0.25)),
        "q50": float(np.quantile(vals, 0.50)),
        "q75": float(np.quantile(vals, 0.75)),
        "q90": float(np.quantile(vals, 0.90)),
        "q95": float(np.quantile(vals, 0.95)),
        "q99": float(np.quantile(vals, 0.99)),
    }


def finite(x: np.ndarray) -> np.ndarray:
    y = np.asarray(x, dtype=np.float64)
    return y[np.isfinite(y)]


def ks_stat(a: np.ndarray, b: np.ndarray) -> float:
    x = np.sort(finite(a))
    y = np.sort(finite(b))
    grid = np.sort(np.concatenate([x, y]))
    if len(grid) == 0:
        return float("nan")
    return float(np.max(np.abs(np.searchsorted(x, grid, side="right") / len(x) - np.searchsorted(y, grid, side="right") / len(y))))


def wasserstein_1(a: np.ndarray, b: np.ndarray) -> float:
    x = np.sort(finite(a))
    y = np.sort(finite(b))
    n = min(len(x), len(y))
    if n == 0:
        return float("nan")
    q = (np.arange(n) + 0.5) / n
    return float(np.mean(np.abs(np.quantile(x, q) - np.quantile(y, q))))


def hist_score(native: np.ndarray, sample: np.ndarray, bins: int = 80) -> dict[str, float]:
    a = finite(native)
    b = finite(sample)
    lo = float(min(np.min(a), np.min(b)))
    hi = float(max(np.max(a), np.max(b)))
    if math.isclose(lo, hi):
        lo -= 0.5
        hi += 0.5
    ha, edges = np.histogram(a, bins=bins, range=(lo, hi))
    hb, _ = np.histogram(b, bins=edges)
    pa = ha / max(float(ha.sum()), 1.0)
    pb = hb / max(float(hb.sum()), 1.0)
    dx = np.diff(edges)
    m = 0.5 * (pa + pb)
    js = 0.0
    mask = pa > 0
    js += 0.5 * float(np.sum(pa[mask] * np.log(pa[mask] / np.maximum(m[mask], 1.0e-300))))
    mask = pb > 0
    js += 0.5 * float(np.sum(pb[mask] * np.log(pb[mask] / np.maximum(m[mask], 1.0e-300))))
    overlap = float(np.sum(np.minimum(pa / dx, pb / dx) * dx))
    return {
        "native_mean": float(np.mean(a)),
        "native_std": float(np.std(a, ddof=1)),
        "sample_mean": float(np.mean(b)),
        "sample_std": float(np.std(b, ddof=1)),
        "mean_shift_native_sigma": float((np.mean(b) - np.mean(a)) / max(np.std(a, ddof=1), 1.0e-300)),
        "std_ratio": float(np.std(b, ddof=1) / max(np.std(a, ddof=1), 1.0e-300)),
        "ks_statistic": ks_stat(a, b),
        "wasserstein_1": wasserstein_1(a, b),
        "jensen_shannon": js,
        "histogram_overlap_coefficient": overlap,
    }


def phi2_support_score(native: np.ndarray, sample: np.ndarray) -> dict[str, float]:
    a = finite(native)
    b = finite(sample)
    qn = {q: float(np.quantile(a, q)) for q in [0.90, 0.95, 0.99]}
    qs = {q: float(np.quantile(b, q)) for q in [0.90, 0.95, 0.99]}
    return {
        "native_q90": qn[0.90],
        "native_q95": qn[0.95],
        "native_q99": qn[0.99],
        "sample_q90": qs[0.90],
        "sample_q95": qs[0.95],
        "sample_q99": qs[0.99],
        "q90_ratio": qs[0.90] / max(qn[0.90], 1.0e-300),
        "q95_ratio": qs[0.95] / max(qn[0.95], 1.0e-300),
        "q99_ratio": qs[0.99] / max(qn[0.99], 1.0e-300),
        "frac_above_native_q90": float(np.mean(b > qn[0.90])),
        "frac_above_native_q95": float(np.mean(b > qn[0.95])),
        "frac_above_native_q99": float(np.mean(b > qn[0.99])),
        "max_phi2": float(np.max(b)),
    }


def support_score(native: np.ndarray, sample: np.ndarray, prefix: str = "") -> dict[str, float]:
    a = finite(native)
    b = finite(sample)
    qlist = [0.01, 0.05, 0.10, 0.90, 0.95, 0.99]
    qn = {q: float(np.quantile(a, q)) for q in qlist}
    qs = {q: float(np.quantile(b, q)) for q in qlist}
    out = {
        "sample_min": float(np.min(b)),
        "sample_max": float(np.max(b)),
        "frac_below_native_q01": float(np.mean(b < qn[0.01])),
        "frac_below_native_q05": float(np.mean(b < qn[0.05])),
        "frac_below_native_q10": float(np.mean(b < qn[0.10])),
        "frac_above_native_q90": float(np.mean(b > qn[0.90])),
        "frac_above_native_q95": float(np.mean(b > qn[0.95])),
        "frac_above_native_q99": float(np.mean(b > qn[0.99])),
    }
    for q in qlist:
        tag = f"q{int(round(100*q)):02d}"
        out[f"native_{tag}"] = qn[q]
        out[f"sample_{tag}"] = qs[q]
        out[f"{tag}_difference"] = qs[q] - qn[q]
        out[f"{tag}_ratio"] = qs[q] / max(abs(qn[q]), 1.0e-300)
    if prefix:
        return {f"{prefix}_{k}": v for k, v in out.items()}
    return out


def shape_score(native: np.ndarray, sample: np.ndarray, prefix: str = "") -> dict[str, float]:
    a = finite(native)
    b = finite(sample)
    out = {
        "sample_min": float(np.min(b)),
        "sample_max": float(np.max(b)),
    }
    for q in [0.05, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]:
        nq = float(np.quantile(a, q))
        sq = float(np.quantile(b, q))
        tag = f"q{int(round(100*q)):02d}"
        out[f"native_{tag}"] = nq
        out[f"sample_{tag}"] = sq
        out[f"{tag}_difference"] = sq - nq
        out[f"{tag}_ratio"] = sq / max(abs(nq), 1.0e-300)
    for q in [0.05, 0.10]:
        thr = float(np.quantile(a, q))
        out[f"frac_below_native_q{int(round(100*q)):02d}"] = float(np.mean(b < thr))
    for q in [0.90, 0.95, 0.99]:
        thr = float(np.quantile(a, q))
        out[f"frac_above_native_q{int(round(100*q)):02d}"] = float(np.mean(b > thr))
    if prefix:
        return {f"{prefix}_{k}": v for k, v in out.items()}
    return out


def per_cfg(phi: np.ndarray, action: ActionSpec) -> dict[str, np.ndarray]:
    rows, grows = per_config_rows(phi, action, "sample")
    out: dict[str, np.ndarray] = {}
    for key in ["action_density", "phi2", "phi4", "local_kurtosis_ratio", "NN", "2nn", "diag", "m2", "m4"]:
        out[key] = np.asarray([float(r[key]) for r in rows], dtype=np.float64)
    out["G_pmin_avg"] = np.asarray([float(r["G_pmin_avg"]) for r in grows], dtype=np.float64)
    return out


def raw_histogram_metrics(model: Any, coarse: np.ndarray, native_phi: np.ndarray, stats: dict[str, Any], kernel: np.ndarray, args: argparse.Namespace, epoch: int, label: str) -> list[dict[str, Any]]:
    n = min(args.raw_eval_count, len(coarse), len(native_phi))
    detail, logq, zmax, logdet = sample_model_lattice(model, coarse[:n], stats, batch_size=args.batch_size, device=torch.device(args.device), seed=args.random_seed + 30000 + epoch)
    phi, _ = inverse_kernel(assemble_psi(coarse[:n], detail), kernel)
    action = ActionSpec("phi4_nn", 1.0, 0.340301)
    native_obs = per_cfg(native_phi[:n], action)
    sample_obs = per_cfg(phi, action)
    rows: list[dict[str, Any]] = []
    for key in OBS_KEYS:
        row = {"epoch": epoch, "checkpoint": label, "observable": key, "n": n, **hist_score(native_obs[key], sample_obs[key])}
        if key == "phi2":
            row.update(phi2_support_score(native_obs[key], sample_obs[key]))
        if key in {"action_density", "phi4"}:
            row.update(support_score(native_obs[key], sample_obs[key]))
        if key == "local_kurtosis_ratio":
            row.update(shape_score(native_obs[key], sample_obs[key]))
        rows.append(row)
    rows.append({"epoch": epoch, "checkpoint": label, "observable": "diagnostic_nonfinite_count", "n": n, "sample_mean": int(np.sum(~np.isfinite(phi)) + np.sum(~np.isfinite(detail))), "sample_std": 0.0})
    rows.append({"epoch": epoch, "checkpoint": label, "observable": "diagnostic_max_abs_z", "n": n, "sample_mean": float(np.max(zmax)), "sample_std": float(np.std(zmax))})
    rows.append({"epoch": epoch, "checkpoint": label, "observable": "diagnostic_logdet", "n": n, "sample_mean": float(np.mean(logdet)), "sample_std": float(np.std(logdet, ddof=1))})
    rows.append({"epoch": epoch, "checkpoint": label, "observable": "diagnostic_logq", "n": n, "sample_mean": float(np.mean(logq)), "sample_std": float(np.std(logq, ddof=1))})
    return rows


def tail_summary(delta_s: np.ndarray, loga: np.ndarray, accepted_by_sweep: list[np.ndarray]) -> dict[str, Any]:
    acc = np.stack(accepted_by_sweep) if accepted_by_sweep else np.zeros((0, 0), dtype=bool)
    longest = 0
    for chain in range(acc.shape[1] if acc.ndim == 2 else 0):
        cur = 0
        for ok in acc[:, chain]:
            cur = 0 if ok else cur + 1
            longest = max(longest, cur)
    return {
        "DeltaS_mean": float(np.mean(delta_s)),
        "DeltaS_std": float(np.std(delta_s, ddof=1)),
        "DeltaS_median": float(np.quantile(delta_s, 0.50)),
        "DeltaS_p90": float(np.quantile(delta_s, 0.90)),
        "DeltaS_p95": float(np.quantile(delta_s, 0.95)),
        "DeltaS_p99": float(np.quantile(delta_s, 0.99)),
        "logA_mean": float(np.mean(loga)),
        "logA_std": float(np.std(loga, ddof=1)),
        "logA_p01": float(np.quantile(loga, 0.01)),
        "logA_p05": float(np.quantile(loga, 0.05)),
        "logA_p10": float(np.quantile(loga, 0.10)),
        "frac_logA_lt_minus5": float(np.mean(loga < -5.0)),
        "frac_logA_lt_minus10": float(np.mean(loga < -10.0)),
        "frac_logA_lt_minus20": float(np.mean(loga < -20.0)),
        "longest_rejection_streak": int(longest),
    }


def global_independence_diag(model: Any, coarse: np.ndarray, native_phi: np.ndarray, native_detail: np.ndarray, stats: dict[str, Any], kernel: np.ndarray, args: argparse.Namespace, epoch: int, label: str) -> dict[str, Any]:
    n = min(args.global_chains, len(coarse), len(native_phi), len(native_detail))
    device = torch.device(args.device)
    c = coarse[:n].copy()
    phi = native_phi[:n].copy()
    current_s = action_total(phi, ActionSpec("phi4_nn", 1.0, 0.340301)).astype(np.float64)
    current_logq = log_prob_model_lattice(model, c, native_detail[:n], stats, batch_size=args.batch_size, device=device)
    rng = np.random.default_rng(args.random_seed + 40000 + epoch)
    accepted_total = 0
    ds_vals: list[np.ndarray] = []
    la_vals: list[np.ndarray] = []
    accepted_by_sweep: list[np.ndarray] = []
    for sweep in range(1, args.global_sweeps + 1):
        prop_detail, prop_logq, _z, _ld = sample_model_lattice(model, c, stats, batch_size=args.batch_size, device=device, seed=args.random_seed + 41000 + 1000 * epoch + sweep)
        prop_phi, _ = inverse_kernel(assemble_psi(c, prop_detail), kernel)
        prop_s = action_total(prop_phi, ActionSpec("phi4_nn", 1.0, 0.340301)).astype(np.float64)
        delta_s = prop_s - current_s
        loga = -delta_s + current_logq - prop_logq
        accept = np.log(rng.random(n)) < np.minimum(loga, 0.0)
        if np.any(accept):
            phi[accept] = prop_phi[accept]
            current_s[accept] = prop_s[accept]
            current_logq[accept] = prop_logq[accept]
        accepted_total += int(np.sum(accept))
        ds_vals.append(delta_s)
        la_vals.append(loga)
        accepted_by_sweep.append(accept)
    ds = np.concatenate(ds_vals)
    la = np.concatenate(la_vals)
    row = {
        "epoch": epoch,
        "checkpoint": label,
        "diagnostic_type": "global_fixed_coarse_independence",
        "number_of_chains": n,
        "number_of_sweeps": args.global_sweeps,
        "acceptance_denominator": "chains * sweeps full-field proposals",
        "attempts": int(n * args.global_sweeps),
        "accepted": int(accepted_total),
        "acceptance": float(accepted_total / max(n * args.global_sweeps, 1)),
    }
    row.update(tail_summary(ds, la, accepted_by_sweep))
    return row


def local_patch_diag(
    model: Any,
    coarse: np.ndarray,
    stats: dict[str, Any],
    kernel_spec: Any,
    kernel_matrix: np.ndarray,
    args: argparse.Namespace,
    run: Path,
    epoch: int,
    label: str,
) -> dict[str, Any]:
    n = min(args.local_chains, len(coarse))
    device = torch.device(args.device)
    detail, _logq, zmax, _logdet = sample_model_lattice(model, coarse[:n], stats, batch_size=args.batch_size, device=device, seed=args.random_seed + 50000 + epoch)
    psi = assemble_psi(coarse[:n], detail).astype(np.float32)
    action = ActionSpec("phi4_nn", 1.0, 0.340301)
    rng = np.random.default_rng(args.random_seed + 51000 + epoch)
    attempted = 0
    accepted = 0
    loga_means = []
    ds_means = []
    last_phi = None
    patch_dir = run / "logs" / "local_patch_diagnostics"
    patch_dir.mkdir(parents=True, exist_ok=True)
    pc_args = argparse.Namespace(
        disable_coarse_updates=True,
        detail_passes=int(args.local_detail_passes),
        fine_proposal_sigma=float(args.local_fine_proposal_sigma),
        fine_patch_size=int(args.local_detail_patch_size),
        passes=0,
        proposal_sigma=0.0,
        coarse_patch_size=int(args.local_detail_patch_size),
        global_sweep=0,
        verbose_patch_log=False,
    )
    for sweep in range(1, args.local_sweeps + 1):
        pc_args.global_sweep = sweep
        writer = StreamingCsv(patch_dir / f"{label}_epoch{epoch:04d}_sweep{sweep:03d}.csv", PATCH_HISTORY_FIELDS)
        last_phi, psi, meta = patch_correct(psi, kernel_spec, action, pc_args, writer, rng)
        writer.close()
        attempted += int(meta.get("detail_update_config_attempts", 0))
        accepted += int(meta.get("detail_update_accepts", 0))
        loga_means.append(float(meta.get("latent_log_accept_mean", float("nan"))))
        ds_means.append(float(meta.get("detail_deltaS_mean", float("nan"))))
    if last_phi is None:
        last_phi, _ = inverse_kernel(psi, kernel_matrix)
    reb = apply_kernel(last_phi, kernel_matrix)[:, 0::2, 0::2] - coarse[:n]
    return {
        "epoch": epoch,
        "checkpoint": label,
        "diagnostic_type": "local_patchwise_detail",
        "number_of_chains": n,
        "number_of_sweeps": args.local_sweeps,
        "patch_size": args.local_detail_patch_size,
        "detail_passes": args.local_detail_passes,
        "acceptance_denominator": "local detail patch config-updates attempted by lambda=0.2 patch_correct",
        "attempts": int(attempted),
        "accepted": int(accepted),
        "acceptance": float(accepted / max(attempted, 1)),
        "DeltaS_mean": float(np.nanmean(ds_means)),
        "logA_mean": float(np.nanmean(loga_means)),
        "reblocking_max_error": float(np.max(np.abs(reb))),
        "nonfinite_count": int(np.sum(~np.isfinite(last_phi)) + np.sum(~np.isfinite(psi))),
        "max_abs_z_initial": float(np.max(zmax)),
    }


def save_checkpoint(
    path: Path,
    model: Any,
    opt: torch.optim.Optimizer,
    args: argparse.Namespace,
    state: dict[str, Any],
    history: list[dict[str, Any]],
    epoch: int,
    best_nll: float,
    bad: int,
    weights: dict[str, float],
    source_ckpt: dict[str, Any],
    checkpoint_metadata: dict[str, Any] | None = None,
) -> None:
    source_config = source_ckpt.get("config", {})
    cfg = {
        "resume_checkpoint": source_config.get("resume_checkpoint"),
        "initialization_mode": getattr(args, "initialization_mode", "transferred"),
        "source_rqspline_checkpoint": str(args.source_checkpoint),
        "layers": int(args.layers),
        "hidden_channels": int(args.hidden_channels),
        "conv_kernel_size": int(args.conv_kernel_size),
        "log_scale_bound": float(args.log_scale_bound),
        "num_bins": int(args.num_bins),
        "tail_bound": float(args.tail_bound),
        "min_bin_width": float(args.min_bin_width),
        "min_bin_height": float(args.min_bin_height),
        "min_derivative": float(args.min_derivative),
        "mode": "lam1p0_L16to32_matched_pair_rqspline_finetune",
        "phi2_tail_guard": bool(getattr(args, "phi2_tail_guard", False)),
        "phi2_support_weight": (
            None
            if getattr(args, "phi2_support_weight", None) is None
            else float(getattr(args, "phi2_support_weight"))
        ),
        "local_kurtosis_shape_guard": bool(getattr(args, "local_kurtosis_shape_guard", False)),
        "local_kurtosis_shape_weight": (
            None
            if getattr(args, "local_kurtosis_shape_weight", None) is None
            else float(getattr(args, "local_kurtosis_shape_weight"))
        ),
        "two_sided_tail_guard": bool(getattr(args, "two_sided_tail_guard", False)),
        "normalization_metadata": None if getattr(args, "normalization_metadata", None) is None else str(getattr(args, "normalization_metadata")),
    }
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": opt.state_dict(),
            "scheduler_state": None,
            "epoch": int(epoch),
            "absolute_epoch": int(epoch),
            "best_validation_nll": float(best_nll),
            "patience_counter": int(bad),
            "config": cfg,
            "state": state,
            "history": history,
            "observable_weights": weights,
            "architecture": {
                "factorization": "q(d01|coarse) q(d10|coarse,d01) q(d11|coarse,d01,d10)",
                "layers_per_stage": int(args.layers),
                "hidden_channels": int(args.hidden_channels),
                "conv_kernel_size": int(args.conv_kernel_size),
                "parameter_count": int(sum(p.numel() for p in model.parameters())),
            },
            "spline_settings": {
                "num_bins": int(args.num_bins),
                "tail_bound": float(args.tail_bound),
                "min_bin_width": float(args.min_bin_width),
                "min_bin_height": float(args.min_bin_height),
                "min_derivative": float(args.min_derivative),
                "tails": "linear",
            },
            "rng_state": rng_state(),
            "checkpoint_metadata": checkpoint_metadata or {},
        },
        path,
    )


def raw_metric_lookup(raw_rows: list[dict[str, Any]], epoch: int) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for row in raw_rows:
        if int(row.get("epoch", -999999)) != int(epoch):
            continue
        obs = str(row.get("observable", ""))
        out[obs] = row
    return out


def phi2_support_rows_from_raw(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in raw_rows:
        if row.get("observable") != "phi2":
            continue
        rows.append(
            {
                "epoch": row.get("epoch"),
                "checkpoint": row.get("checkpoint"),
                "n": row.get("n"),
                "mean_shift_native_sigma": row.get("mean_shift_native_sigma"),
                "std_ratio": row.get("std_ratio"),
                "ks_statistic": row.get("ks_statistic"),
                "histogram_overlap_coefficient": row.get("histogram_overlap_coefficient"),
                "wasserstein_1": row.get("wasserstein_1"),
                "native_q90": row.get("native_q90"),
                "native_q95": row.get("native_q95"),
                "native_q99": row.get("native_q99"),
                "sample_q90": row.get("sample_q90"),
                "sample_q95": row.get("sample_q95"),
                "sample_q99": row.get("sample_q99"),
                "q90_ratio": row.get("q90_ratio"),
                "q95_ratio": row.get("q95_ratio"),
                "q99_ratio": row.get("q99_ratio"),
                "frac_above_native_q90": row.get("frac_above_native_q90"),
                "frac_above_native_q95": row.get("frac_above_native_q95"),
                "frac_above_native_q99": row.get("frac_above_native_q99"),
                "max_phi2": row.get("max_phi2"),
            }
        )
    return rows


def kurtosis_support_rows_from_raw(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = [
        "epoch",
        "checkpoint",
        "n",
        "mean_shift_native_sigma",
        "std_ratio",
        "ks_statistic",
        "histogram_overlap_coefficient",
        "wasserstein_1",
        "sample_min",
        "sample_max",
        "native_q05",
        "sample_q05",
        "native_q25",
        "sample_q25",
        "native_q50",
        "sample_q50",
        "native_q75",
        "sample_q75",
        "native_q90",
        "sample_q90",
        "native_q95",
        "sample_q95",
        "native_q99",
        "sample_q99",
        "frac_below_native_q05",
        "frac_below_native_q10",
        "frac_above_native_q90",
        "frac_above_native_q95",
        "frac_above_native_q99",
    ]
    rows = []
    for row in raw_rows:
        if row.get("observable") != "local_kurtosis_ratio":
            continue
        rows.append({field: row.get(field) for field in fields})
    return rows


def patch_eligibility(
    gd: dict[str, Any],
    ld: dict[str, Any],
    raw_rows: list[dict[str, Any]],
    source_local_acceptance: float,
    epoch: int,
) -> tuple[bool, list[str]]:
    raw = raw_metric_lookup(raw_rows, epoch)
    reasons: list[str] = []
    nonfinite = raw.get("diagnostic_nonfinite_count", {}).get("sample_mean", 0.0)
    if float(nonfinite) != 0.0 or int(ld.get("nonfinite_count", 0)) != 0:
        reasons.append("nonfinite outputs")
    if float(ld.get("reblocking_max_error", float("inf"))) > 1.0e-5:
        reasons.append(f"reblocking_max_error={ld.get('reblocking_max_error')}")
    if float(ld.get("acceptance", 0.0)) < float(source_local_acceptance) - 0.02:
        reasons.append(f"local_acceptance={ld.get('acceptance')} below source tolerance")
    checks = [
        ("action_density", "mean_shift_native_sigma", 0.30),
        ("phi4", "mean_shift_native_sigma", 0.30),
        ("local_kurtosis_ratio", "mean_shift_native_sigma", 0.35),
        ("action_density", "ks_statistic", 0.13),
        ("phi4", "ks_statistic", 0.13),
        ("local_kurtosis_ratio", "ks_statistic", 0.15),
    ]
    for obs, field, limit in checks:
        val = raw.get(obs, {}).get(field, float("nan"))
        if not np.isfinite(float(val)):
            reasons.append(f"{obs}.{field} missing")
        elif "shift" in field and abs(float(val)) >= limit:
            reasons.append(f"{obs}.{field}={float(val):.6g} >= {limit}")
        elif "shift" not in field and float(val) >= limit:
            reasons.append(f"{obs}.{field}={float(val):.6g} >= {limit}")
    if getattr(patch_eligibility, "require_phi2_support", False):
        phi2 = raw.get("phi2", {})
        phi2_checks = [
            ("mean_shift_native_sigma", 0.20, "abs_lt"),
            ("ks_statistic", 0.10, "lt"),
            ("q90_ratio", (0.97, 1.03), "range"),
            ("q95_ratio", (0.95, 1.05), "range"),
            ("q99_ratio", (0.90, 1.10), "range"),
            ("frac_above_native_q95", 0.04, "ge"),
            ("frac_above_native_q99", 0.005, "ge"),
        ]
        for field, limit, mode in phi2_checks:
            val = phi2.get(field, float("nan"))
            if not np.isfinite(float(val)):
                reasons.append(f"phi2.{field} missing")
            elif mode == "abs_lt" and abs(float(val)) >= float(limit):
                reasons.append(f"phi2.{field}={float(val):.6g} outside abs<{limit}")
            elif mode == "lt" and float(val) >= float(limit):
                reasons.append(f"phi2.{field}={float(val):.6g} >= {limit}")
            elif mode == "ge" and float(val) < float(limit):
                reasons.append(f"phi2.{field}={float(val):.6g} < {limit}")
            elif mode == "range":
                lo, hi = limit
                if not (float(lo) <= float(val) <= float(hi)):
                    reasons.append(f"phi2.{field}={float(val):.6g} outside [{lo}, {hi}]")
    if getattr(patch_eligibility, "require_balanced_support", False):
        action = raw.get("action_density", {})
        phi4 = raw.get("phi4", {})
        balanced_checks = [
            ("action_density", action, "mean_shift_native_sigma", 0.20, "abs_lt"),
            ("action_density", action, "ks_statistic", 0.10, "lt"),
            ("action_density", action, "std_ratio", (0.95, 1.05), "range"),
            ("action_density", action, "frac_below_native_q05", (0.035, 0.065), "range"),
            ("action_density", action, "frac_above_native_q95", (0.035, 0.065), "range"),
            ("phi4", phi4, "mean_shift_native_sigma", 0.20, "abs_lt"),
            ("phi4", phi4, "ks_statistic", 0.10, "lt"),
            ("phi4", phi4, "q95_ratio", (0.95, 1.05), "range"),
            ("phi4", phi4, "q99_ratio", (0.90, 1.10), "range"),
            ("NN", raw.get("NN", {}), "ks_statistic", 0.10, "lt"),
        ]
        for obs, source, field, limit, mode in balanced_checks:
            val = source.get(field, float("nan"))
            if not np.isfinite(float(val)):
                reasons.append(f"{obs}.{field} missing")
            elif mode == "abs_lt" and abs(float(val)) >= float(limit):
                reasons.append(f"{obs}.{field}={float(val):.6g} outside abs<{limit}")
            elif mode == "lt" and float(val) >= float(limit):
                reasons.append(f"{obs}.{field}={float(val):.6g} >= {limit}")
            elif mode == "range":
                lo, hi = limit
                if not (float(lo) <= float(val) <= float(hi)):
                    reasons.append(f"{obs}.{field}={float(val):.6g} outside [{lo}, {hi}]")
    return (len(reasons) == 0), reasons


def patch_score(gd: dict[str, Any]) -> float:
    return float(gd["DeltaS_std"]) + 15.0 * float(gd["frac_logA_lt_minus10"]) + 50.0 * float(gd["frac_logA_lt_minus20"])


def validation_nll(model: Any, loader: DataLoader, stats: dict[str, Any], args: argparse.Namespace) -> float:
    model.eval()
    total = 0.0
    count = 0
    lj = physical_log_jac_const(stats, int(args.coarse_lattice))
    with torch.no_grad():
        for cb, db in loader:
            cb = cb.to(args.device)
            db = db.to(args.device)
            lp = model.log_prob(cb, db) + lj
            total += float((-lp).sum().detach().cpu())
            count += int(len(cb))
    return total / max(count, 1)


def phi2_support_loss(
    phi: torch.Tensor,
    targets: dict[str, float],
    weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    phi2_cfg = (phi * phi).mean(dim=(1, 2))
    std_eps = float(targets["std"])
    mean = phi2_cfg.mean()
    std = torch.sqrt(torch.clamp(torch.mean((phi2_cfg - mean) ** 2), min=1.0e-12))
    q90 = torch.quantile(phi2_cfg, 0.90)
    q95 = torch.quantile(phi2_cfg, 0.95)
    q99 = torch.quantile(phi2_cfg, 0.99)
    smooth = float(targets["smooth_tail_width"])
    f90 = torch.sigmoid((phi2_cfg - float(targets["q90"])) / smooth).mean()
    f95 = torch.sigmoid((phi2_cfg - float(targets["q95"])) / smooth).mean()
    f99 = torch.sigmoid((phi2_cfg - float(targets["q99"])) / smooth).mean()

    def z(x: torch.Tensor, target: float) -> torch.Tensor:
        return (x - float(target)) / std_eps

    target_q90 = torch.as_tensor(targets["q90"], dtype=phi.dtype, device=phi.device)
    target_q95 = torch.as_tensor(targets["q95"], dtype=phi.dtype, device=phi.device)
    target_q99 = torch.as_tensor(targets["q99"], dtype=phi.dtype, device=phi.device)

    terms = {
        "phi2_support_mean": z(mean, targets["mean"]) ** 2,
        "phi2_support_std": z(std, targets["std"]) ** 2,
        "phi2_support_q90_under": torch.relu((target_q90 - q90) / std_eps) ** 2,
        "phi2_support_q95_under": torch.relu((target_q95 - q95) / std_eps) ** 2,
        "phi2_support_q99_under": torch.relu((target_q99 - q99) / std_eps) ** 2,
        "phi2_support_tail_q90_under": torch.relu(torch.as_tensor(targets["tail_fraction_q90"], dtype=phi.dtype, device=phi.device) - f90) ** 2,
        "phi2_support_tail_q95_under": torch.relu(torch.as_tensor(targets["tail_fraction_q95"], dtype=phi.dtype, device=phi.device) - f95) ** 2,
        "phi2_support_tail_q99_under": torch.relu(torch.as_tensor(targets["tail_fraction_q99"], dtype=phi.dtype, device=phi.device) - f99) ** 2,
    }
    loss = (
        terms["phi2_support_mean"]
        + terms["phi2_support_std"]
        + 0.5 * terms["phi2_support_q90_under"]
        + terms["phi2_support_q95_under"]
        + 2.0 * terms["phi2_support_q99_under"]
        + 2.0 * terms["phi2_support_tail_q90_under"]
        + 2.0 * terms["phi2_support_tail_q95_under"]
        + 2.0 * terms["phi2_support_tail_q99_under"]
    )
    log = {key: float(val.detach().cpu()) for key, val in terms.items()}
    log.update(
        {
            "phi2_batch_mean": float(mean.detach().cpu()),
            "phi2_batch_std": float(std.detach().cpu()),
            "phi2_batch_q90": float(q90.detach().cpu()),
            "phi2_batch_q95": float(q95.detach().cpu()),
            "phi2_batch_q99": float(q99.detach().cpu()),
            "phi2_batch_tail_q90": float(f90.detach().cpu()),
            "phi2_batch_tail_q95": float(f95.detach().cpu()),
            "phi2_batch_tail_q99": float(f99.detach().cpu()),
            "phi2_support_loss_unweighted": float(loss.detach().cpu()),
            "phi2_support_loss_weighted": float((float(weight) * loss).detach().cpu()),
        }
    )
    return float(weight) * loss, log


def observable_support_loss(
    values: torch.Tensor,
    targets: dict[str, float],
    weight: float,
    name: str,
    mode: str,
    std_loss_weight: float = 1.0,
    quantile_loss_weight: float = 1.0,
    occupancy_loss_weight: float = 2.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    std_eps = float(targets["std"])
    mean = values.mean()
    std = torch.sqrt(torch.clamp(torch.mean((values - mean) ** 2), min=1.0e-12))
    q01 = torch.quantile(values, 0.01)
    q05 = torch.quantile(values, 0.05)
    q10 = torch.quantile(values, 0.10)
    q90 = torch.quantile(values, 0.90)
    q95 = torch.quantile(values, 0.95)
    q99 = torch.quantile(values, 0.99)
    smooth = float(targets["smooth_tail_width"])

    low01 = torch.sigmoid((float(targets["q01"]) - values) / smooth).mean()
    low05 = torch.sigmoid((float(targets["q05"]) - values) / smooth).mean()
    low10 = torch.sigmoid((float(targets["q10"]) - values) / smooth).mean()
    high90 = torch.sigmoid((values - float(targets["q90"])) / smooth).mean()
    high95 = torch.sigmoid((values - float(targets["q95"])) / smooth).mean()
    high99 = torch.sigmoid((values - float(targets["q99"])) / smooth).mean()

    def norm_sq(x: torch.Tensor, target: float) -> torch.Tensor:
        return ((x - float(target)) / std_eps) ** 2

    terms: dict[str, torch.Tensor] = {
        f"{name}_support_mean": norm_sq(mean, targets["mean"]),
        f"{name}_support_std": norm_sq(std, targets["std"]),
    }
    if mode == "two_sided":
        for tag, q, target in [
            ("q01", q01, "q01"),
            ("q05", q05, "q05"),
            ("q10", q10, "q10"),
            ("q90", q90, "q90"),
            ("q95", q95, "q95"),
            ("q99", q99, "q99"),
        ]:
            terms[f"{name}_support_{tag}"] = norm_sq(q, targets[target])
        terms[f"{name}_support_low_q05_occ"] = (low05 - float(targets["tail_fraction_q05"])) ** 2
        terms[f"{name}_support_high_q95_occ"] = (high95 - float(targets["tail_fraction_q95"])) ** 2
        loss = (
            terms[f"{name}_support_mean"]
            + float(std_loss_weight) * terms[f"{name}_support_std"]
            + float(quantile_loss_weight) * 0.5 * terms[f"{name}_support_q01"]
            + float(quantile_loss_weight) * terms[f"{name}_support_q05"]
            + float(quantile_loss_weight) * 0.5 * terms[f"{name}_support_q10"]
            + float(quantile_loss_weight) * 0.5 * terms[f"{name}_support_q90"]
            + float(quantile_loss_weight) * terms[f"{name}_support_q95"]
            + float(quantile_loss_weight) * 0.5 * terms[f"{name}_support_q99"]
            + float(occupancy_loss_weight) * terms[f"{name}_support_low_q05_occ"]
            + float(occupancy_loss_weight) * terms[f"{name}_support_high_q95_occ"]
        )
    elif mode == "upper_symmetric":
        for tag, q, target in [("q90", q90, "q90"), ("q95", q95, "q95"), ("q99", q99, "q99")]:
            terms[f"{name}_support_{tag}"] = norm_sq(q, targets[target])
        terms[f"{name}_support_high_q90_occ"] = (high90 - float(targets["tail_fraction_q90"])) ** 2
        terms[f"{name}_support_high_q95_occ"] = (high95 - float(targets["tail_fraction_q95"])) ** 2
        terms[f"{name}_support_high_q99_occ"] = (high99 - float(targets["tail_fraction_q99"])) ** 2
        loss = (
            terms[f"{name}_support_mean"]
            + terms[f"{name}_support_std"]
            + 0.5 * terms[f"{name}_support_q90"]
            + terms[f"{name}_support_q95"]
            + terms[f"{name}_support_q99"]
            + terms[f"{name}_support_high_q90_occ"]
            + 2.0 * terms[f"{name}_support_high_q95_occ"]
            + 2.0 * terms[f"{name}_support_high_q99_occ"]
        )
    else:
        raise ValueError(f"unknown support loss mode: {mode}")

    log = {key: float(val.detach().cpu()) for key, val in terms.items()}
    log.update(
        {
            f"{name}_batch_mean": float(mean.detach().cpu()),
            f"{name}_batch_std": float(std.detach().cpu()),
            f"{name}_batch_q01": float(q01.detach().cpu()),
            f"{name}_batch_q05": float(q05.detach().cpu()),
            f"{name}_batch_q10": float(q10.detach().cpu()),
            f"{name}_batch_q90": float(q90.detach().cpu()),
            f"{name}_batch_q95": float(q95.detach().cpu()),
            f"{name}_batch_q99": float(q99.detach().cpu()),
            f"{name}_batch_low_q05": float(low05.detach().cpu()),
            f"{name}_batch_high_q95": float(high95.detach().cpu()),
            f"{name}_support_loss_unweighted": float(loss.detach().cpu()),
            f"{name}_support_loss_weighted": float((float(weight) * loss).detach().cpu()),
        }
    )
    return float(weight) * loss, log


def proposal_low_tail_coverage_loss(
    values: torch.Tensor,
    targets: dict[str, float],
    weight: float,
    name: str,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Require proposal coverage below native q01/q05, without penalizing excess."""
    smooth = float(targets["smooth_tail_width"])
    std = max(float(targets["std"]), 1.0e-6)
    q01 = torch.quantile(values, 0.01)
    q05 = torch.quantile(values, 0.05)
    mass01 = torch.sigmoid((float(targets["q01"]) - values) / smooth).mean()
    mass05 = torch.sigmoid((float(targets["q05"]) - values) / smooth).mean()
    q01_deficit = torch.relu((q01 - float(targets["q01"])) / std) ** 2
    q05_deficit = torch.relu((q05 - float(targets["q05"])) / std) ** 2
    mass01_deficit = torch.relu(float(targets["tail_fraction_q01"]) - mass01) ** 2
    mass05_deficit = torch.relu(float(targets["tail_fraction_q05"]) - mass05) ** 2
    loss = 0.5 * q01_deficit + q05_deficit + 2.0 * mass01_deficit + 2.0 * mass05_deficit
    log = {
        f"{name}_proposal_lowtail_q01": float(q01.detach().cpu()),
        f"{name}_proposal_lowtail_q05": float(q05.detach().cpu()),
        f"{name}_proposal_lowtail_mass_q01": float(mass01.detach().cpu()),
        f"{name}_proposal_lowtail_mass_q05": float(mass05.detach().cpu()),
        f"{name}_proposal_lowtail_loss_unweighted": float(loss.detach().cpu()),
        f"{name}_proposal_lowtail_loss_weighted": float((float(weight) * loss).detach().cpu()),
    }
    return float(weight) * loss, log


def local_kurtosis_shape_loss(
    phi: torch.Tensor,
    targets: dict[str, float],
    weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    phi2_cfg = (phi * phi).mean(dim=(1, 2))
    phi4_cfg = (phi**4).mean(dim=(1, 2))
    values = phi4_cfg / torch.clamp(phi2_cfg * phi2_cfg, min=1.0e-12)
    return local_kurtosis_shape_loss_from_values(values, targets, weight)


def local_kurtosis_shape_loss_from_values(
    values: torch.Tensor,
    targets: dict[str, float],
    weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    std_eps = float(targets["std"])
    mean = values.mean()
    std = torch.sqrt(torch.clamp(torch.mean((values - mean) ** 2), min=1.0e-12))

    def norm_sq(x: torch.Tensor, target: float) -> torch.Tensor:
        return ((x - float(target)) / std_eps) ** 2

    temperature = max(std_eps * 0.05, 1.0e-4)

    def soft_cdf_at(threshold: float) -> torch.Tensor:
        th = torch.as_tensor(float(threshold), dtype=values.dtype, device=values.device)
        return torch.sigmoid((th - values) / temperature).mean()

    # Match the native CDF at selected native quantiles. This is the cheap
    # differentiable proxy for quantile matching used in training; hard batch
    # quantiles are still logged below with gradients disabled.
    cdf25 = soft_cdf_at(targets["q25"])
    cdf50 = soft_cdf_at(targets["q50"])
    cdf75 = soft_cdf_at(targets["q75"])
    cdf90 = soft_cdf_at(targets["q90"])
    cdf95 = soft_cdf_at(targets["q95"])

    terms = {
        "local_kurtosis_shape_mean": norm_sq(mean, targets["mean"]),
        "local_kurtosis_shape_std": norm_sq(std, targets["std"]),
        "local_kurtosis_shape_q25": (cdf25 - 0.25) ** 2,
        "local_kurtosis_shape_q50": (cdf50 - 0.50) ** 2,
        "local_kurtosis_shape_q75": (cdf75 - 0.75) ** 2,
        "local_kurtosis_shape_q90": (cdf90 - 0.90) ** 2,
        "local_kurtosis_shape_q95": (cdf95 - 0.95) ** 2,
    }
    loss = (
        terms["local_kurtosis_shape_mean"]
        + terms["local_kurtosis_shape_std"]
        + 0.5 * terms["local_kurtosis_shape_q25"]
        + 0.75 * terms["local_kurtosis_shape_q50"]
        + 0.5 * terms["local_kurtosis_shape_q75"]
        + 0.5 * terms["local_kurtosis_shape_q90"]
        + 0.75 * terms["local_kurtosis_shape_q95"]
    )
    with torch.no_grad():
        q25 = torch.quantile(values, 0.25)
        q50 = torch.quantile(values, 0.50)
        q75 = torch.quantile(values, 0.75)
        q90 = torch.quantile(values, 0.90)
        q95 = torch.quantile(values, 0.95)
    log = {key: float(val.detach().cpu()) for key, val in terms.items()}
    log.update(
        {
            "local_kurtosis_batch_mean": float(mean.detach().cpu()),
            "local_kurtosis_batch_std": float(std.detach().cpu()),
            "local_kurtosis_batch_q25": float(q25.detach().cpu()),
            "local_kurtosis_batch_q50": float(q50.detach().cpu()),
            "local_kurtosis_batch_q75": float(q75.detach().cpu()),
            "local_kurtosis_batch_q90": float(q90.detach().cpu()),
            "local_kurtosis_batch_q95": float(q95.detach().cpu()),
            "local_kurtosis_cdf_q25": float(cdf25.detach().cpu()),
            "local_kurtosis_cdf_q50": float(cdf50.detach().cpu()),
            "local_kurtosis_cdf_q75": float(cdf75.detach().cpu()),
            "local_kurtosis_cdf_q90": float(cdf90.detach().cpu()),
            "local_kurtosis_cdf_q95": float(cdf95.detach().cpu()),
            "local_kurtosis_cdf_temperature": float(temperature),
            "local_kurtosis_shape_loss_unweighted": float(loss.detach().cpu()),
            "local_kurtosis_shape_loss_weighted": float((float(weight) * loss).detach().cpu()),
        }
    )
    return float(weight) * loss, log


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--source-checkpoint", type=Path, default=Path("perfect_blocking_upsampling/runs/lam1p0/lam1p0_L8to16_kf0p340301_kc0p340301_7x7_phi2_nn_guarded_autoregressive_detail_8layer48_rqspline_localreg_from_affine_ep137_20260717T125835Z/checkpoints/checkpoint_best.pt"))
    ap.add_argument("--fine-config-source", type=Path, default=Path("data/configs_phi4_2d/lam1p0_kappac0p340301_L32/configs.npz"))
    ap.add_argument("--coarse-lattice", type=int, default=16, help="Coarse lattice L_c; the matched fine source must have L_f=2L_c.")
    ap.add_argument("--kernel-path", type=Path, default=Path("perfect_blocking/perfect_blocking_lam1p0/kernels/selected_for_upscaling/best_7x7_phi2_nn_guarded_eta_included.json"))
    ap.add_argument("--initialization-mode", choices=["transferred", "fresh"], default="transferred")
    ap.add_argument("--source-start-index", type=int, default=0)
    ap.add_argument("--total-count", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--train-stage", choices=["all", "eo", "oe", "oo"], default="all", help="Freeze the other autoregressive detail stages.")
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5.0e-5)
    ap.add_argument("--weight-decay", type=float, default=1.0e-5)
    ap.add_argument("--grad-clip", type=float, default=10.0)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--hidden-channels", type=int, default=48)
    ap.add_argument("--conv-kernel-size", type=int, default=3)
    ap.add_argument("--log-scale-bound", type=float, default=0.75)
    ap.add_argument("--num-bins", type=int, default=8)
    ap.add_argument("--tail-bound", type=float, default=6.0)
    ap.add_argument("--min-bin-width", type=float, default=1.0e-3)
    ap.add_argument("--min-bin-height", type=float, default=1.0e-3)
    ap.add_argument("--min-derivative", type=float, default=1.0e-3)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--raw-eval-count", type=int, default=500)
    ap.add_argument("--global-chains", type=int, default=64)
    ap.add_argument("--global-sweeps", type=int, default=100)
    ap.add_argument("--local-chains", type=int, default=64)
    ap.add_argument("--local-sweeps", type=int, default=25)
    ap.add_argument("--local-detail-patch-size", type=int, default=16)
    ap.add_argument("--local-detail-passes", type=int, default=10)
    ap.add_argument("--local-fine-proposal-sigma", type=float, default=0.04)
    ap.add_argument("--obs-weights", default="")
    ap.add_argument("--phi2-tail-guard", action="store_true", help="Add differentiable phi2 mean/std/upper-tail support penalties and require phi2 tail coverage for best-patch selection.")
    ap.add_argument("--phi2-support-weight", type=float, default=None, help="Override the phi2 observable weight used to scale the phi2 support penalty.")
    ap.add_argument("--phi2-support-scale", type=float, default=1.0, help="Scale the phi2 support penalty weight after selecting it from --phi2-support-weight or obs weight.")
    ap.add_argument("--balanced-support-guard", action="store_true", help="Add action-density and phi4 distribution-support losses and require balanced support eligibility.")
    ap.add_argument("--two-sided-tail-guard", action="store_true", help="Add two-sided low/high-tail support losses for action_density, phi2, and phi4.")
    ap.add_argument("--action-support-weight", type=float, default=None)
    ap.add_argument("--phi4-support-weight", type=float, default=None)
    ap.add_argument("--action-std-match-weight", type=float, default=0.0, help="Additional direct action-density variance-match weight; zero disables it.")
    ap.add_argument("--phi4-std-match-weight", type=float, default=0.0, help="Additional direct phi4 variance-match weight; zero disables it.")
    ap.add_argument("--proposal-action-lowtail-weight", type=float, default=0.0, help="One-sided action proposal-coverage weight below native q01/q05.")
    ap.add_argument("--proposal-kurtosis-lowtail-weight", type=float, default=0.0, help="One-sided local-kurtosis proposal-coverage weight below native q01/q05.")
    ap.add_argument("--tail-stratified-train", action="store_true", help="Oversample native configurations in the low action-density or low local-kurtosis tails during conditional-NLL training.")
    ap.add_argument("--tail-stratified-quantile", type=float, default=0.10, help="Low-tail quantile used for each observable in --tail-stratified-train.")
    ap.add_argument("--tail-stratified-tail-fraction", type=float, default=0.40, help="Fraction of draws devoted to the union of the selected low-tail configurations.")
    ap.add_argument("--proposal-phi4-min-std-ratio", type=float, default=0.0, help="Require proposal phi4 std to reach this multiple of native std; zero disables it.")
    ap.add_argument("--proposal-phi4-min-std-weight", type=float, default=0.0, help="Weight for the one-sided phi4 minimum-width loss.")
    ap.add_argument("--tail-guard-std-weight", type=float, default=0.25, help="Relative weight for std matching in --two-sided-tail-guard; keep modest to prefer coverage over narrow matching.")
    ap.add_argument("--tail-guard-quantile-weight", type=float, default=1.0)
    ap.add_argument("--tail-guard-occupancy-weight", type=float, default=3.0)
    ap.add_argument("--tail-guard-low-occupancy-weight", type=float, default=None, help="Optional override for low-tail occupancy terms in --two-sided-tail-guard.")
    ap.add_argument("--tail-guard-high-occupancy-weight", type=float, default=None, help="Optional override for high-tail occupancy terms in --two-sided-tail-guard.")
    ap.add_argument("--local-kurtosis-shape-guard", action="store_true", help="Add local-kurtosis mean/std/quantile shape matching without enabling broader support guards.")
    ap.add_argument("--local-kurtosis-shape-weight", type=float, default=None)
    ap.add_argument("--normalization-metadata", type=Path, default=None, help="Reuse existing coarse/detail normalization metadata instead of recomputing from this subset.")
    ap.add_argument("--random-seed", type=int, default=2026071721)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--train-count", type=int, default=4000)
    ap.add_argument("--val-count", type=int, default=500)
    ap.add_argument("--test-count", type=int, default=500)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--stop-after-eval-epoch", type=int, default=None)
    ap.add_argument("--exact-eval-every", type=int, default=None, help="Run global/local exact diagnostics at this cadence; raw diagnostics still follow --eval-every.")
    args = ap.parse_args()
    if args.coarse_lattice <= 0:
        raise ValueError("--coarse-lattice must be positive")
    fine_lattice = 2 * int(args.coarse_lattice)
    if not 0.0 < args.tail_stratified_quantile < 0.5:
        raise ValueError("--tail-stratified-quantile must lie in (0, 0.5)")
    if not 0.0 < args.tail_stratified_tail_fraction < 1.0:
        raise ValueError("--tail-stratified-tail-fraction must lie in (0, 1)")
    if args.smoke:
        args.epochs = min(args.epochs, 1)
        args.train_count = min(args.train_count, 32)
        args.val_count = min(args.val_count, 8)
        args.test_count = min(args.test_count, 8)
        args.raw_eval_count = min(args.raw_eval_count, 8)
        args.global_chains = min(args.global_chains, 4)
        args.global_sweeps = min(args.global_sweeps, 2)
        args.local_chains = min(args.local_chains, 4)
        args.local_sweeps = min(args.local_sweeps, 1)

    run = args.run_dir
    for sub in ["logs", "checkpoints", "observables", "plots", "summaries", "debug"]:
        (run / sub).mkdir(parents=True, exist_ok=True)
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    args.device = str(args.device)
    device = torch.device(args.device)
    weights = parse_weights(args.obs_weights) if args.obs_weights else dict(DEFAULT_WEIGHTS)
    patch_eligibility.require_phi2_support = bool(args.phi2_tail_guard)
    patch_eligibility.require_balanced_support = bool(args.balanced_support_guard)

    kernel, kernel_json = load_kernel_matrix(args.kernel_path)
    kernel_spec, _kernel_spec_json = load_kernel_spec(PROJECT_ROOT / args.kernel_path)
    if not bool(kernel_json.get("kernel_coefficients_include_eta_scale", False)) or not np.isclose(float(kernel.sum()), ETA_SCALE, atol=1.0e-10):
        raise RuntimeError(f"bad kernel eta convention for {args.kernel_path}: sum={float(kernel.sum())}")
    phi32_all = load_phi(args.fine_config_source)
    start_index = int(args.source_start_index)
    if start_index < 0 or start_index >= len(phi32_all):
        raise ValueError(f"bad --source-start-index {start_index} for {len(phi32_all)} configs")
    total_count = len(phi32_all) - start_index if args.total_count is None else int(args.total_count)
    if total_count <= 0:
        raise ValueError("--total-count must be positive")
    stop_index = min(start_index + total_count, len(phi32_all))
    source_indices = np.arange(start_index, stop_index, dtype=np.int64)
    phi32 = phi32_all[source_indices]
    if phi32.shape[1:] != (fine_lattice, fine_lattice):
        raise ValueError(f"fine source has shape {phi32.shape}; expected lattice {fine_lattice}")
    pairs = split_pairs(phi32, kernel)
    reb_native = apply_kernel(phi32[: min(16, len(phi32))], kernel)[:, 0::2, 0::2] - pairs["coarse"][: min(16, len(phi32))]

    n = len(phi32)
    rng = np.random.default_rng(args.random_seed)
    idx = rng.permutation(n)
    if n < 5000:
        print(json.dumps({"warning": "using fewer than requested 5000 matched pairs", "available": n}), flush=True)
    n_train = min(args.train_count, int(0.8 * n))
    n_val = min(args.val_count, n - n_train)
    n_test = min(args.test_count, n - n_train - n_val)
    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val : n_train + n_val + n_test]
    if args.normalization_metadata is not None:
        stats = normalize_stats_metadata(json.loads((PROJECT_ROOT / args.normalization_metadata).read_text()))
        stats["method"] = f"reused normalization metadata from {args.normalization_metadata}"
    else:
        stats = scalar_sector_stats(pairs["coarse"], pairs["detail"], train_idx)
    coarse_std, detail_std = standardize_arrays(pairs["coarse"], pairs["detail"], stats)
    state = {
        "stats": stats,
        "train_idx": train_idx.astype(np.int64),
        "val_idx": val_idx.astype(np.int64),
        "test_idx": test_idx.astype(np.int64),
        "source_indices": source_indices.astype(np.int64),
        "kernel_path": str(args.kernel_path),
        "data_source": str(args.fine_config_source),
    }

    if args.initialization_mode == "fresh":
        source_ckpt = {
            "config": {
                "resume_checkpoint": None,
                "initialization_mode": "fresh",
                "layers": args.layers,
                "hidden_channels": args.hidden_channels,
                "conv_kernel_size": args.conv_kernel_size,
                "log_scale_bound": args.log_scale_bound,
                "num_bins": args.num_bins,
                "tail_bound": args.tail_bound,
                "min_bin_width": args.min_bin_width,
                "min_bin_height": args.min_bin_height,
                "min_derivative": args.min_derivative,
            },
            "epoch": 0,
        }
        affine = AffineARDetailFlow(
            lattice_size=int(args.coarse_lattice),
            layers=int(args.layers),
            hidden=int(args.hidden_channels),
            kernel_size=int(args.conv_kernel_size),
            log_scale_bound=float(args.log_scale_bound),
        )
        spline = SplineARDetailFlow(
            lattice_size=int(args.coarse_lattice),
            layers=int(args.layers),
            hidden=int(args.hidden_channels),
            kernel_size=int(args.conv_kernel_size),
            num_bins=int(args.num_bins),
            tail_bound=float(args.tail_bound),
            min_bin_width=float(args.min_bin_width),
            min_bin_height=float(args.min_bin_height),
            min_derivative=float(args.min_derivative),
        )
        model = ResidualSplineARDetailFlow(affine, spline).to(device)
        load_report = {
            "load_strict": True,
            "missing_keys": [],
            "unexpected_keys": [],
            "source_resume_checkpoint": None,
            "source_spline_settings": {
                "num_bins": int(args.num_bins),
                "tail_bound": float(args.tail_bound),
                "min_bin_width": float(args.min_bin_width),
                "min_bin_height": float(args.min_bin_height),
                "min_derivative": float(args.min_derivative),
                "tails": "linear",
            },
            "stripped_duplicate_alias_keys": 0,
            "initialization_mode": "fresh_identity_initialized_couplings",
        }
    else:
        source_ckpt = torch.load(PROJECT_ROOT / args.source_checkpoint, map_location=device, weights_only=False)
        model, load_report = build_model_from_checkpoint(source_ckpt, lattice_size=int(args.coarse_lattice), device=device)
    # Both the affine base and spline are stored as three parallel ModuleLists.
    # Their stages are 0=d01/eo, 1=d10/oe, 2=d11/oo.
    stage_tokens = {"eo": ("flows.0.",), "oe": ("flows.1.",), "oo": ("flows.2.",)}
    if args.train_stage != "all":
        tokens = stage_tokens[args.train_stage]
        trainable = []
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(any(token in name.lower() for token in tokens))
            if parameter.requires_grad:
                trainable.append(name)
        if not trainable:
            raise RuntimeError(f"no parameters matched --train-stage={args.train_stage}; inspect model parameter names")
        print(json.dumps({"train_stage": args.train_stage, "trainable_parameters": trainable}), flush=True)
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=args.weight_decay)
    optimizer_resume_report = {"attempted": False, "restored": False, "reason": "source checkpoint has no optimizer_state"}
    # A checkpoint trained with all three sectors has an optimizer state whose
    # parameter-group layout differs from a staged (EO/OE/OO-only) optimizer.
    # Reusing it can attach moments to the wrong tensors.  Transfer weights but
    # deliberately start fresh Adam moments for every single-stage run.
    if args.train_stage != "all":
        optimizer_resume_report = {"attempted": False, "restored": False, "reason": "fresh optimizer required for a frozen single-stage parameter set"}
    elif args.initialization_mode == "transferred" and isinstance(source_ckpt.get("optimizer_state"), dict) and source_ckpt.get("optimizer_state"):
        optimizer_resume_report = {"attempted": True, "restored": False, "reason": ""}
        try:
            opt.load_state_dict(source_ckpt["optimizer_state"])
            for group in opt.param_groups:
                group["lr"] = float(args.lr)
                group["weight_decay"] = float(args.weight_decay)
            optimizer_resume_report = {"attempted": True, "restored": True, "reason": "optimizer state loaded; lr and weight_decay reset from branch args"}
        except Exception as exc:
            optimizer_resume_report = {"attempted": True, "restored": False, "reason": repr(exc)}
    train_ds = TensorDataset(torch.from_numpy(coarse_std[train_idx]), torch.from_numpy(detail_std[train_idx]), torch.from_numpy(pairs["coarse"][train_idx]))
    val_ds = TensorDataset(torch.from_numpy(coarse_std[val_idx]), torch.from_numpy(detail_std[val_idx]))
    tail_stratification: dict[str, Any] = {"enabled": False}
    if args.tail_stratified_train:
        action = ActionSpec("phi4_nn", 1.0, 0.340301)
        train_rows, _train_global_rows = per_config_rows(phi32[train_idx], action, "native_L32_train")
        action_values = np.asarray([float(row["action_density"]) for row in train_rows], dtype=np.float64)
        kurtosis_values = np.asarray([float(row["local_kurtosis_ratio"]) for row in train_rows], dtype=np.float64)
        action_cut = float(np.quantile(action_values, args.tail_stratified_quantile))
        kurtosis_cut = float(np.quantile(kurtosis_values, args.tail_stratified_quantile))
        tail_mask = (action_values <= action_cut) | (kurtosis_values <= kurtosis_cut)
        n_tail = int(np.count_nonzero(tail_mask))
        n_bulk = int(len(tail_mask) - n_tail)
        if n_tail == 0 or n_bulk == 0:
            raise RuntimeError("tail stratification produced an empty tail or bulk set")
        tail_weight = (float(args.tail_stratified_tail_fraction) * n_bulk) / ((1.0 - float(args.tail_stratified_tail_fraction)) * n_tail)
        sample_weights = np.ones(len(tail_mask), dtype=np.float64)
        sample_weights[tail_mask] = tail_weight
        sampler = WeightedRandomSampler(torch.as_tensor(sample_weights, dtype=torch.double), num_samples=len(train_ds), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler)
        tail_stratification = {
            "enabled": True,
            "observables": ["action_density", "local_kurtosis_ratio"],
            "low_quantile": float(args.tail_stratified_quantile),
            "target_tail_draw_fraction": float(args.tail_stratified_tail_fraction),
            "action_cut": action_cut,
            "kurtosis_cut": kurtosis_cut,
            "tail_count": n_tail,
            "bulk_count": n_bulk,
            "tail_sampling_weight": float(tail_weight),
        }
        print(json.dumps({"tail_stratification": tail_stratification}), flush=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    state["tail_stratification"] = tail_stratification
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    targets = native_targets(phi32, val_idx)
    phi2_targets = native_phi2_support_targets(phi32, val_idx)
    phi2_twosided_support_targets = native_observable_support_targets(phi32, val_idx, "phi2")
    action_support_targets = native_observable_support_targets(phi32, val_idx, "action_density")
    phi4_support_targets = native_observable_support_targets(phi32, val_idx, "phi4")
    kurtosis_support_targets = native_observable_support_targets(phi32, val_idx, "local_kurtosis_ratio")
    local_kurtosis_shape_targets = native_observable_shape_targets(phi32, val_idx, "local_kurtosis_ratio")
    kt = torch_kernel_fft(kernel, fine_lattice, device)
    d_mean = torch.tensor(np.asarray(stats["detail_mean"]).reshape(1, 3, 1, 1), dtype=torch.float32, device=device)
    d_std_t = torch.tensor(np.asarray(stats["detail_std"]).reshape(1, 3, 1, 1), dtype=torch.float32, device=device)

    def obs_penalty(cb: torch.Tensor, coarse_phys: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        active_weights = {key: float(weight) for key, weight in weights.items() if abs(float(weight)) > 0.0}
        d_samp, _logq, _z, _ld = model.sample(cb)
        detail_phys = d_samp * d_std_t + d_mean
        psi = torch.empty((len(cb), fine_lattice, fine_lattice), dtype=detail_phys.dtype, device=device)
        psi[:, 0::2, 0::2] = coarse_phys
        psi[:, 0::2, 1::2] = detail_phys[:, 0]
        psi[:, 1::2, 0::2] = detail_phys[:, 1]
        psi[:, 1::2, 1::2] = detail_phys[:, 2]
        phi = torch_inverse_kernel(psi, kt)
        phi2_cfg = (phi * phi).mean(dim=(1, 2))
        phi4_cfg = (phi**4).mean(dim=(1, 2))
        nn_cfg = 0.5 * (
            (phi * torch.roll(phi, shifts=-1, dims=1)).mean(dim=(1, 2))
            + (phi * torch.roll(phi, shifts=-1, dims=2)).mean(dim=(1, 2))
        )
        local_kurtosis_cfg = phi4_cfg / torch.clamp(phi2_cfg * phi2_cfg, min=1.0e-12)
        action_cfg = (1.0 - 2.0 * 1.0) * phi2_cfg + phi4_cfg - 4.0 * 0.340301 * nn_cfg
        obs = {
            "action_density": action_cfg.mean(),
            "phi2": phi2_cfg.mean(),
            "phi4": phi4_cfg.mean(),
            "local_kurtosis_ratio": local_kurtosis_cfg.mean(),
            "NN": nn_cfg.mean(),
        }
        missing_keys = set(active_weights).difference(obs)
        if missing_keys:
            if "2nn" in missing_keys:
                obs["2nn"] = 0.5 * (
                    (phi * torch.roll(phi, shifts=-2, dims=1)).mean(dim=(1, 2))
                    + (phi * torch.roll(phi, shifts=-2, dims=2)).mean(dim=(1, 2))
                ).mean()
            if "diag" in missing_keys:
                obs["diag"] = (phi * torch.roll(torch.roll(phi, shifts=-1, dims=1), shifts=-1, dims=2)).mean(dim=(1, 2)).mean()
            if "m2" in missing_keys or "m4" in missing_keys:
                m_cfg = phi.mean(dim=(1, 2))
                if "m2" in missing_keys:
                    obs["m2"] = (m_cfg * m_cfg).mean()
                if "m4" in missing_keys:
                    obs["m4"] = (m_cfg**4).mean()
            if "G_pmin_avg" in missing_keys:
                fft = torch.fft.fft2(phi.to(torch.complex64), dim=(1, 2))
                volume = float(phi.shape[1] * phi.shape[2])
                obs["G_pmin_avg"] = (
                    0.5 * ((torch.abs(fft[:, 1, 0]) ** 2) / volume + (torch.abs(fft[:, 0, 1]) ** 2) / volume)
                ).real.mean()
        loss = cb.new_tensor(0.0)
        zvals: dict[str, float] = {}
        for key, weight in active_weights.items():
            z = (obs[key] - float(targets[key]["mean"])) / float(targets[key]["std"])
            loss = loss + float(weight) * z * z
            zvals[f"z_{key}"] = float(z.detach().cpu())
            zvals[f"loss_component_{key}"] = float((float(weight) * z * z).detach().cpu())
        # Separate variance terms can move phi4 and action toward their own
        # targets even when they need corrections in opposite directions.
        for name, values, target, weight in [
            ("action_density", action_cfg, action_support_targets, float(args.action_std_match_weight)),
            ("phi4", phi4_cfg, phi4_support_targets, float(args.phi4_std_match_weight)),
        ]:
            if weight <= 0.0:
                continue
            batch_std = torch.std(values, unbiased=False)
            z_std = (batch_std - float(target["std"])) / max(float(target["std"]), 1.0e-6)
            std_loss = weight * z_std * z_std
            loss = loss + std_loss
            zvals[f"{name}_explicit_std"] = float(batch_std.detach().cpu())
            zvals[f"{name}_explicit_std_z"] = float(z_std.detach().cpu())
            zvals[f"{name}_explicit_std_loss"] = float(std_loss.detach().cpu())
        phi4_min_ratio = float(args.proposal_phi4_min_std_ratio)
        phi4_min_weight = float(args.proposal_phi4_min_std_weight)
        if phi4_min_ratio > 0.0 and phi4_min_weight > 0.0:
            phi4_std = torch.std(phi4_cfg, unbiased=False)
            phi4_floor = phi4_min_ratio * float(phi4_support_targets["std"])
            phi4_deficit = torch.relu((phi4_floor - phi4_std) / max(float(phi4_support_targets["std"]), 1.0e-6)) ** 2
            phi4_floor_loss = phi4_min_weight * phi4_deficit
            loss = loss + phi4_floor_loss
            zvals["phi4_proposal_min_std"] = float(phi4_std.detach().cpu())
            zvals["phi4_proposal_min_std_ratio"] = float(phi4_std.detach().cpu()) / float(phi4_support_targets["std"])
            zvals["phi4_proposal_min_std_loss"] = float(phi4_floor_loss.detach().cpu())
        action_coverage_loss, action_coverage_log = proposal_low_tail_coverage_loss(
            action_cfg, action_support_targets, float(args.proposal_action_lowtail_weight), "action_density"
        )
        kurtosis_coverage_loss, kurtosis_coverage_log = proposal_low_tail_coverage_loss(
            local_kurtosis_cfg, kurtosis_support_targets, float(args.proposal_kurtosis_lowtail_weight), "local_kurtosis"
        )
        loss = loss + action_coverage_loss + kurtosis_coverage_loss
        zvals.update(action_coverage_log)
        zvals.update(kurtosis_coverage_log)
        if args.phi2_tail_guard:
            support_weight = float(args.phi2_support_weight if args.phi2_support_weight is not None else weights.get("phi2", 0.0)) * float(args.phi2_support_scale)
            support_loss, support_log = phi2_support_loss(phi, phi2_targets, support_weight)
            loss = loss + support_loss
            zvals.update(support_log)
        if args.balanced_support_guard:
            action_weight = float(args.action_support_weight if args.action_support_weight is not None else weights.get("action_density", 0.0))
            phi4_weight = float(args.phi4_support_weight if args.phi4_support_weight is not None else weights.get("phi4", 0.0))
            action_loss, action_log = observable_support_loss(action_cfg, action_support_targets, action_weight, "action_density", "two_sided")
            phi4_loss, phi4_log = observable_support_loss(phi4_cfg, phi4_support_targets, phi4_weight, "phi4", "upper_symmetric")
            loss = loss + action_loss + phi4_loss
            zvals.update(action_log)
            zvals.update(phi4_log)
        if args.two_sided_tail_guard:
            action_weight = float(args.action_support_weight if args.action_support_weight is not None else weights.get("action_density", 0.0))
            phi2_weight = float(args.phi2_support_weight if args.phi2_support_weight is not None else weights.get("phi2", 0.0)) * float(args.phi2_support_scale)
            phi4_weight = float(args.phi4_support_weight if args.phi4_support_weight is not None else weights.get("phi4", 0.0))
            action_loss, action_log = observable_support_loss(
                action_cfg,
                action_support_targets,
                action_weight,
                "action_density_twosided",
                "two_sided",
                std_loss_weight=float(args.tail_guard_std_weight),
                quantile_loss_weight=float(args.tail_guard_quantile_weight),
                occupancy_loss_weight=float(args.tail_guard_occupancy_weight),
            )
            phi2_loss, phi2_log = observable_support_loss(
                phi2_cfg,
                phi2_twosided_support_targets,
                phi2_weight,
                "phi2_twosided",
                "two_sided",
                std_loss_weight=float(args.tail_guard_std_weight),
                quantile_loss_weight=float(args.tail_guard_quantile_weight),
                occupancy_loss_weight=float(args.tail_guard_occupancy_weight),
            )
            phi4_loss, phi4_log = observable_support_loss(
                phi4_cfg,
                phi4_support_targets,
                phi4_weight,
                "phi4_twosided",
                "two_sided",
                std_loss_weight=float(args.tail_guard_std_weight),
                quantile_loss_weight=float(args.tail_guard_quantile_weight),
                occupancy_loss_weight=float(args.tail_guard_occupancy_weight),
            )
            loss = loss + action_loss + phi2_loss + phi4_loss
            low_occ_weight = float(
                args.tail_guard_low_occupancy_weight
                if args.tail_guard_low_occupancy_weight is not None
                else args.tail_guard_occupancy_weight
            )
            high_occ_weight = float(
                args.tail_guard_high_occupancy_weight
                if args.tail_guard_high_occupancy_weight is not None
                else args.tail_guard_occupancy_weight
            )
            if low_occ_weight != float(args.tail_guard_occupancy_weight) or high_occ_weight != float(args.tail_guard_occupancy_weight):
                # Small directional correction on top of the symmetric support loss.
                # This lets a diagnostic run emphasize missing lower support without
                # changing the baseline support-loss convention used above.
                smooth_action = float(action_support_targets["smooth_tail_width"])
                smooth_phi2 = float(phi2_twosided_support_targets["smooth_tail_width"])
                smooth_phi4 = float(phi4_support_targets["smooth_tail_width"])
                action_low = torch.sigmoid((float(action_support_targets["q05"]) - action_cfg) / smooth_action).mean()
                action_high = torch.sigmoid((action_cfg - float(action_support_targets["q95"])) / smooth_action).mean()
                phi2_low = torch.sigmoid((float(phi2_twosided_support_targets["q05"]) - phi2_cfg) / smooth_phi2).mean()
                phi2_high = torch.sigmoid((phi2_cfg - float(phi2_twosided_support_targets["q95"])) / smooth_phi2).mean()
                phi4_low = torch.sigmoid((float(phi4_support_targets["q05"]) - phi4_cfg) / smooth_phi4).mean()
                phi4_high = torch.sigmoid((phi4_cfg - float(phi4_support_targets["q95"])) / smooth_phi4).mean()
                base_occ = float(args.tail_guard_occupancy_weight)
                occ_correction = cb.new_tensor(0.0)
                for name, low, high, target, weight in [
                    ("action_density_twosided", action_low, action_high, action_weight, action_weight),
                    ("phi2_twosided", phi2_low, phi2_high, phi2_weight, phi2_weight),
                    ("phi4_twosided", phi4_low, phi4_high, phi4_weight, phi4_weight),
                ]:
                    low_term = (low - 0.05) ** 2
                    high_term = (high - 0.05) ** 2
                    occ_correction = occ_correction + float(weight) * (
                        (low_occ_weight - base_occ) * low_term + (high_occ_weight - base_occ) * high_term
                    )
                    zvals[f"{name}_extra_low_q05_occ_loss"] = float((float(weight) * (low_occ_weight - base_occ) * low_term).detach().cpu())
                    zvals[f"{name}_extra_high_q95_occ_loss"] = float((float(weight) * (high_occ_weight - base_occ) * high_term).detach().cpu())
                loss = loss + occ_correction
                zvals["twosided_extra_occupancy_loss_weighted"] = float(occ_correction.detach().cpu())
            zvals.update(action_log)
            zvals.update(phi2_log)
            zvals.update(phi4_log)
        if args.local_kurtosis_shape_guard:
            kurt_weight = float(args.local_kurtosis_shape_weight if args.local_kurtosis_shape_weight is not None else weights.get("local_kurtosis_ratio", 0.0))
            kurt_loss, kurt_log = local_kurtosis_shape_loss_from_values(local_kurtosis_cfg, local_kurtosis_shape_targets, kurt_weight)
            loss = loss + kurt_loss
            zvals.update(kurt_log)
        return loss, zvals

    init_val = validation_nll(model, val_loader, stats, args)
    init_global = global_independence_diag(model, pairs["coarse"][test_idx], phi32[test_idx], pairs["detail"][test_idx], stats, kernel, args, 0, "source_zero_shot")
    init_local = local_patch_diag(model, pairs["coarse"][test_idx], stats, kernel_spec, kernel, args, run, 0, "source_zero_shot")
    init_raw = raw_histogram_metrics(model, pairs["coarse"][test_idx], phi32[test_idx], stats, kernel, args, 0, "source_zero_shot")

    history: list[dict[str, Any]] = []
    global_rows = [init_global]
    local_rows = [init_local]
    raw_rows = init_raw
    best_nll = init_val
    best_patch_score = patch_score(init_global)
    best_nll_epoch = 0
    best_patch_epoch = 0
    best_nll_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_patch_state = copy.deepcopy(best_nll_state)
    best_nll_opt = copy.deepcopy(opt.state_dict())
    best_patch_opt = copy.deepcopy(opt.state_dict())
    best_patch_eligibility = {"epoch": 0, "eligible": True, "reasons": [], "patch_score": best_patch_score}
    eligibility_rows: list[dict[str, Any]] = [
        {
            "epoch": 0,
            "checkpoint": "source_zero_shot",
            "eligible": True,
            "reasons": "",
            "patch_score": best_patch_score,
        }
    ]
    write_csv(run / "observables" / "global_independence_diagnostics.csv", global_rows)
    write_csv(run / "observables" / "local_patch_diagnostics.csv", local_rows)
    write_csv(run / "observables" / "raw_histogram_metrics.csv", raw_rows)
    write_csv(run / "observables" / "phi2_support_metrics.csv", phi2_support_rows_from_raw(raw_rows))
    write_csv(run / "observables" / "kurtosis_support_metrics.csv", kurtosis_support_rows_from_raw(raw_rows))
    write_csv(run / "observables" / "patch_eligibility.csv", eligibility_rows)
    bad = 0

    print(json.dumps({"epoch": 0, "validation_nll": init_val, "global_acceptance": init_global["acceptance"], "global_DeltaS_std": init_global["DeltaS_std"], "local_acceptance": init_local["acceptance"]}), flush=True)
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = {"loss": 0.0, "nll": 0.0, "obs": 0.0, "count": 0}
        last_z: dict[str, float] = {}
        lj = physical_log_jac_const(stats, int(args.coarse_lattice))
        for cb, db, coarse_phys in train_loader:
            cb = cb.to(device)
            db = db.to(device)
            coarse_phys = coarse_phys.to(device)
            opt.zero_grad(set_to_none=True)
            nll = -(model.log_prob(cb, db) + lj).mean()
            op, zvals = obs_penalty(cb, coarse_phys)
            loss = nll + op
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            bs = int(len(cb))
            totals["loss"] += float(loss.detach().cpu()) * bs
            totals["nll"] += float(nll.detach().cpu()) * bs
            totals["obs"] += float(op.detach().cpu()) * bs
            totals["count"] += bs
            last_z = zvals
        val = validation_nll(model, val_loader, stats, args)
        row = {
            "epoch": epoch,
            "train_loss": totals["loss"] / max(totals["count"], 1),
            "train_nll": totals["nll"] / max(totals["count"], 1),
            "train_observable_penalty": totals["obs"] / max(totals["count"], 1),
            "validation_nll": val,
            **last_z,
        }
        history.append(row)
        if val < best_nll - 1.0e-8:
            best_nll = val
            best_nll_epoch = epoch
            best_nll_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_nll_opt = copy.deepcopy(opt.state_dict())
            bad = 0
        else:
            bad += 1
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            exact_every = args.eval_every if args.exact_eval_every is None else int(args.exact_eval_every)
            exact_due = epoch % exact_every == 0 or epoch == args.epochs
            gd = (
                global_independence_diag(model, pairs["coarse"][test_idx], phi32[test_idx], pairs["detail"][test_idx], stats, kernel, args, epoch, f"epoch_{epoch:04d}")
                if exact_due
                else global_rows[-1]
            )
            ld = (
                local_patch_diag(model, pairs["coarse"][test_idx], stats, kernel_spec, kernel, args, run, epoch, f"epoch_{epoch:04d}")
                if exact_due
                else local_rows[-1]
            )
            rm = raw_histogram_metrics(model, pairs["coarse"][test_idx], phi32[test_idx], stats, kernel, args, epoch, f"epoch_{epoch:04d}")
            if exact_due:
                global_rows.append(gd)
                local_rows.append(ld)
            raw_rows.extend(rm)
            eligible, reasons = patch_eligibility(gd, ld, raw_rows, init_local["acceptance"], epoch)
            score = patch_score(gd)
            eligibility_rows.append(
                {
                    "epoch": epoch,
                    "checkpoint": f"epoch_{epoch:04d}",
                    "eligible": bool(eligible),
                    "reasons": "; ".join(reasons),
                    "patch_score": score,
                    "exact_diagnostics_evaluated": bool(exact_due),
                }
            )
            if exact_due and eligible and score < best_patch_score:
                best_patch_score = score
                best_patch_epoch = epoch
                best_patch_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_patch_opt = copy.deepcopy(opt.state_dict())
                best_patch_eligibility = {"epoch": epoch, "eligible": True, "reasons": [], "patch_score": score}
            write_csv(run / "observables" / "global_independence_diagnostics.csv", global_rows)
            write_csv(run / "observables" / "local_patch_diagnostics.csv", local_rows)
            write_csv(run / "observables" / "raw_histogram_metrics.csv", raw_rows)
            write_csv(run / "observables" / "phi2_support_metrics.csv", phi2_support_rows_from_raw(raw_rows))
            write_csv(run / "observables" / "kurtosis_support_metrics.csv", kurtosis_support_rows_from_raw(raw_rows))
            write_csv(run / "observables" / "patch_eligibility.csv", eligibility_rows)
            save_checkpoint(
                run / "checkpoints" / f"checkpoint_epoch{epoch:03d}.pt",
                model,
                opt,
                args,
                state,
                history,
                epoch,
                best_nll,
                bad,
                weights,
                source_ckpt,
                {"patch_eligibility": {"eligible": bool(eligible), "reasons": reasons, "patch_score": score}},
            )
        write_csv(run / "observables" / "training_history.csv", history)
        write_csv(run / "observables" / "validation_nll.csv", [{"epoch": r["epoch"], "validation_nll": r["validation_nll"]} for r in history])
        save_checkpoint(run / "checkpoints" / "checkpoint_latest.pt", model, opt, args, state, history, epoch, best_nll, bad, weights, source_ckpt)
        if val <= best_nll + 1.0e-12:
            save_checkpoint(run / "checkpoints" / "checkpoint_best_nll.pt", model, opt, args, state, history, epoch, best_nll, bad, weights, source_ckpt, {"selection": "best_nll"})
        if best_patch_epoch == epoch:
            save_checkpoint(run / "checkpoints" / "checkpoint_best_patch.pt", model, opt, args, state, history, epoch, best_nll, bad, weights, source_ckpt, {"selection": "best_patch", "patch_eligibility": best_patch_eligibility})
        write_json(run / "status.json", {"status": "running", "epoch": epoch, "best_nll_epoch": best_nll_epoch, "best_patch_epoch": best_patch_epoch, "best_validation_nll": best_nll, "elapsed_seconds": time.time() - start_time})
        print(json.dumps(row), flush=True)
        if args.stop_after_eval_epoch is not None and epoch >= int(args.stop_after_eval_epoch) and (epoch % args.eval_every == 0 or epoch == args.epochs):
            write_json(run / "status.json", {"status": "stopped_after_eval_epoch", "epoch": epoch, "best_nll_epoch": best_nll_epoch, "best_patch_epoch": best_patch_epoch, "best_validation_nll": best_nll, "elapsed_seconds": time.time() - start_time})
            break
        if bad >= args.patience:
            break

    model.load_state_dict(best_nll_state)
    opt.load_state_dict(best_nll_opt)
    save_checkpoint(run / "checkpoints" / "checkpoint_best_nll.pt", model, opt, args, state, history, best_nll_epoch, best_nll, bad, weights, source_ckpt, {"selection": "best_nll"})
    nll_smoke = torch.load(run / "checkpoints" / "checkpoint_best_nll.pt", map_location=device, weights_only=False)
    _model_smoke, smoke_report = build_model_from_checkpoint(nll_smoke, lattice_size=int(args.coarse_lattice), device=device)
    model.load_state_dict(best_patch_state)
    opt.load_state_dict(best_patch_opt)
    save_checkpoint(run / "checkpoints" / "checkpoint_best_patch.pt", model, opt, args, state, history, best_patch_epoch, best_nll, bad, weights, source_ckpt, {"selection": "best_patch", "patch_eligibility": best_patch_eligibility})
    patch_smoke = torch.load(run / "checkpoints" / "checkpoint_best_patch.pt", map_location=device, weights_only=False)
    _model_smoke2, smoke_report2 = build_model_from_checkpoint(patch_smoke, lattice_size=int(args.coarse_lattice), device=device)

    comparison = [
        {"checkpoint": "source_zero_shot", "epoch": 0, "checkpoint_path": str(args.source_checkpoint), "validation_nll": init_val},
        {"checkpoint": "best_nll", "epoch": best_nll_epoch, "checkpoint_path": str(run / "checkpoints" / "checkpoint_best_nll.pt"), "validation_nll": best_nll},
        {"checkpoint": "best_patch", "epoch": best_patch_epoch, "checkpoint_path": str(run / "checkpoints" / "checkpoint_best_patch.pt"), "validation_nll": next((r["validation_nll"] for r in history if r["epoch"] == best_patch_epoch), init_val)},
    ]
    write_csv(run / "observables" / "checkpoint_comparison.csv", comparison)

    import yaml

    cfg = {
        "lambda": 1.0,
        "kappa_f": 0.340301,
        "kappa_c": 0.340301,
        "eta": 0.25,
        "eta_scale_numeric": ETA_SCALE,
        "block_factor": 2,
        "L_c": int(args.coarse_lattice),
        "L_f": int(fine_lattice),
        "fine_config_source": str(args.fine_config_source),
        "matched_pair_source": "native L32 blocked to L16 with selected eta-included kernel",
        "kernel_path": str(args.kernel_path),
        "kernel_coefficients_include_eta_scale": True,
        "kernel_sum": float(kernel.sum()),
        "source_checkpoint": None if args.initialization_mode == "fresh" else str(args.source_checkpoint),
        "source_checkpoint_epoch": int(source_ckpt.get("epoch", 0)),
        "initialization_mode": args.initialization_mode,
        "optimizer_resume_report": optimizer_resume_report,
        "source_checkpoint_load_report": json_clean(load_report),
        "architecture": {
            "factorization": "q(d01|coarse) q(d10|coarse,d01) q(d11|coarse,d01,d10)",
            "layers_per_stage": args.layers,
            "hidden_channels": args.hidden_channels,
            "conv_kernel_size": args.conv_kernel_size,
            "rqspline_bins": args.num_bins,
            "tail_bound": args.tail_bound,
            "parameter_count": int(sum(p.numel() for p in model.parameters())),
        },
        "objective": {"NLL": 1.0, "observable_weights": weights},
        "phi2_tail_guard": {
            "enabled": bool(args.phi2_tail_guard),
            "support_weight": float(args.phi2_support_weight if args.phi2_support_weight is not None else weights.get("phi2", 0.0)) * float(args.phi2_support_scale),
            "support_scale": float(args.phi2_support_scale),
            "native_validation_targets": phi2_targets,
            "eligibility_required": bool(args.phi2_tail_guard),
        },
        "balanced_support_guard": {
            "enabled": bool(args.balanced_support_guard),
            "action_support_weight": float(args.action_support_weight if args.action_support_weight is not None else weights.get("action_density", 0.0)),
            "phi4_support_weight": float(args.phi4_support_weight if args.phi4_support_weight is not None else weights.get("phi4", 0.0)),
            "action_std_match_weight": float(args.action_std_match_weight),
            "phi4_std_match_weight": float(args.phi4_std_match_weight),
            "proposal_action_lowtail_weight": float(args.proposal_action_lowtail_weight),
            "proposal_kurtosis_lowtail_weight": float(args.proposal_kurtosis_lowtail_weight),
            "proposal_phi4_min_std_ratio": float(args.proposal_phi4_min_std_ratio),
            "proposal_phi4_min_std_weight": float(args.proposal_phi4_min_std_weight),
            "action_support_targets": action_support_targets,
            "phi4_support_targets": phi4_support_targets,
        },
        "two_sided_tail_guard": {
            "enabled": bool(args.two_sided_tail_guard),
            "action_support_weight": float(args.action_support_weight if args.action_support_weight is not None else weights.get("action_density", 0.0)),
            "phi2_support_weight": float(args.phi2_support_weight if args.phi2_support_weight is not None else weights.get("phi2", 0.0)) * float(args.phi2_support_scale),
            "phi4_support_weight": float(args.phi4_support_weight if args.phi4_support_weight is not None else weights.get("phi4", 0.0)),
            "std_loss_weight": float(args.tail_guard_std_weight),
            "quantile_loss_weight": float(args.tail_guard_quantile_weight),
            "occupancy_loss_weight": float(args.tail_guard_occupancy_weight),
            "low_occupancy_loss_weight": (
                None if args.tail_guard_low_occupancy_weight is None else float(args.tail_guard_low_occupancy_weight)
            ),
            "high_occupancy_loss_weight": (
                None if args.tail_guard_high_occupancy_weight is None else float(args.tail_guard_high_occupancy_weight)
            ),
            "note": "two-sided low/high-tail coverage guard; std matching is intentionally mild",
            "phi2_support_targets": phi2_twosided_support_targets,
            "action_support_targets": action_support_targets,
            "phi4_support_targets": phi4_support_targets,
        },
        "local_kurtosis_shape_guard": {
            "enabled": bool(args.local_kurtosis_shape_guard),
            "support_weight": float(args.local_kurtosis_shape_weight if args.local_kurtosis_shape_weight is not None else weights.get("local_kurtosis_ratio", 0.0)),
            "shape_targets": local_kurtosis_shape_targets,
        },
        "split": {
            "seed": args.random_seed,
            "source_start_index": start_index,
            "source_stop_index_exclusive": int(stop_index),
            "source_indices_count": int(len(source_indices)),
            "n_total": n,
            "n_train": len(train_idx),
            "n_validation": len(val_idx),
            "n_test": len(test_idx),
        },
    }
    (run / "run_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    write_json(run / "submit_manifest.txt", {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "command": " ".join(sys.argv), "git_commit": git_commit(), "host": socket.gethostname(), "platform": platform.platform()})
    write_json(run / "kernel_metadata.json", {"kernel_path": str(args.kernel_path), "kernel_sum": float(kernel.sum()), "eta_scale": ETA_SCALE, "kernel_coefficients_include_eta_scale": bool(kernel_json.get("kernel_coefficients_include_eta_scale", False)), "kernel_json": json_clean(kernel_json)})
    write_json(run / "dataset_split.json", {"seed": args.random_seed, "source_start_index": start_index, "source_stop_index_exclusive": int(stop_index), "source_indices": source_indices.tolist(), "train_local_indices": train_idx.astype(int).tolist(), "validation_local_indices": val_idx.astype(int).tolist(), "test_local_indices": test_idx.astype(int).tolist(), "train_source_indices": source_indices[train_idx].astype(int).tolist(), "validation_source_indices": source_indices[val_idx].astype(int).tolist(), "test_source_indices": source_indices[test_idx].astype(int).tolist()})
    write_json(run / "normalization_metadata.json", json_clean(stats))
    write_json(run / "debug" / "initialization_diagnostics.json", {"kernel_sum": float(kernel.sum()), "eta_scale": ETA_SCALE, "reblocking_native_pair_max_error": float(np.max(np.abs(reb_native))), "source_checkpoint_load_report": load_report, "optimizer_resume_report": optimizer_resume_report, "strict_reload_best_nll": smoke_report, "strict_reload_best_patch": smoke_report2, "source_checkpoint": None if args.initialization_mode == "fresh" else str(args.source_checkpoint), "source_checkpoint_epoch": int(source_ckpt.get("epoch", 0)), "initialization_mode": args.initialization_mode, "aborted_125627Z_not_used": True, "phi2_tail_guard": bool(args.phi2_tail_guard), "phi2_support_targets": phi2_targets, "balanced_support_guard": bool(args.balanced_support_guard), "action_support_targets": action_support_targets, "phi4_support_targets": phi4_support_targets, "local_kurtosis_shape_guard": bool(args.local_kurtosis_shape_guard), "local_kurtosis_shape_targets": local_kurtosis_shape_targets})

    prod_cfg = run / "summaries" / "prepared_patchwise_detail_only_config_note.md"
    prod_cfg.write_text(
        "\n".join(
            [
                "# Prepared Production Config",
                "",
                "Use the selected checkpoint below as `FLOW_CHECKPOINT`/`flow_checkpoint` in `submit_flow_detail_coarse_detail` after reviewing diagnostics.",
                "",
                f"- best NLL checkpoint: `{run / 'checkpoints' / 'checkpoint_best_nll.pt'}`",
                f"- best patch checkpoint: `{run / 'checkpoints' / 'checkpoint_best_patch.pt'}`",
                "",
                "No production chain was launched by this training script.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Lambda=1.0 L16->L32 RQ-Spline Fine-Tune",
        "",
        f"- run directory: `{run}`",
        f"- initialization mode: `{args.initialization_mode}`",
        f"- source checkpoint: `{None if args.initialization_mode == 'fresh' else args.source_checkpoint}`",
        "- source run ending in `20260717T125627Z` was not used.",
        f"- matched pairs: `{n}` native L32 configs blocked to L16",
        f"- split: train `{len(train_idx)}`, validation `{len(val_idx)}`, test `{len(test_idx)}`",
        f"- kernel sum: `{float(kernel.sum()):.17g}`",
        f"- best NLL epoch: `{best_nll_epoch}`",
        f"- best patch epoch: `{best_patch_epoch}`",
        f"- best validation NLL: `{best_nll:.8g}`",
        f"- phi2 tail guard: `{bool(args.phi2_tail_guard)}`",
        f"- balanced support guard: `{bool(args.balanced_support_guard)}`",
        f"- local kurtosis shape guard: `{bool(args.local_kurtosis_shape_guard)}`",
        "",
        "## Checkpoints",
        "",
        f"- best NLL: `{run / 'checkpoints' / 'checkpoint_best_nll.pt'}`",
        f"- best patch: `{run / 'checkpoints' / 'checkpoint_best_patch.pt'}`",
        f"- latest: `{run / 'checkpoints' / 'checkpoint_latest.pt'}`",
        "",
        "## Diagnostics",
        "",
        f"- global independence: `{run / 'observables' / 'global_independence_diagnostics.csv'}`",
        f"- local patch: `{run / 'observables' / 'local_patch_diagnostics.csv'}`",
        f"- raw histograms: `{run / 'observables' / 'raw_histogram_metrics.csv'}`",
        f"- phi2 support metrics: `{run / 'observables' / 'phi2_support_metrics.csv'}`",
        f"- local-kurtosis support metrics: `{run / 'observables' / 'kurtosis_support_metrics.csv'}`",
    ]
    (run / "summaries" / "run_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (run / "summaries" / "checkpoint_comparison.md").write_text(
        "\n".join(["# Checkpoint Comparison", "", *(f"- {r['checkpoint']}: epoch {r['epoch']}, validation NLL {r['validation_nll']}" for r in comparison)]) + "\n",
        encoding="utf-8",
    )
    write_json(run / "status.json", {"status": "completed", "best_nll_epoch": best_nll_epoch, "best_patch_epoch": best_patch_epoch, "best_nll_checkpoint": str(run / "checkpoints" / "checkpoint_best_nll.pt"), "best_patch_checkpoint": str(run / "checkpoints" / "checkpoint_best_patch.pt"), "epochs_completed": len(history), "validation_plateau": bad >= args.patience, "summary": str(run / "summaries" / "run_summary.md")})
    print(json.dumps({"status": "completed", "run_dir": str(run), "best_nll_epoch": best_nll_epoch, "best_patch_epoch": best_patch_epoch, "best_validation_nll": best_nll}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
