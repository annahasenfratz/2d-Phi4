#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
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
from run_lam0p2_flow_detail_rethermalization import coarse_patch_mask  # noqa: E402
from run_lam0p2_residual_flow_patch_chain import fine_detail_mask, patches_per_pass  # noqa: E402
from run_lam1p0_l16to32_rqspline_zeroshot import build_model_from_checkpoint, per_config_observables, sample_model_lattice, stationary_stats  # noqa: E402
from run_lam1p0_rqspline_patchwise import infer_rqspline_latents_and_logj  # noqa: E402
from train_lam1p0_flow_detail_pilot import assemble_psi, load_phi  # noqa: E402


DEFAULT_FLOW = PROJECT_ROOT / "perfect_blocking_upsampling/runs/lam1p0/lam1p0_L8to16_kf0p340301_kc0p340301_7x7_phi2_nn_guarded_autoregressive_detail_8layer48_rqspline_localreg_from_affine_ep137_20260717T125835Z/checkpoints/checkpoint_best.pt"
DEFAULT_KERNEL = PROJECT_ROOT / "perfect_blocking/perfect_blocking_lam1p0/kernels/final/chosen_kernel.json"
DEFAULT_NATIVE_L8 = PROJECT_ROOT / "data/configs_phi4_2d/lam1p0_kappac0p340301_L8/configs.npz"
DEFAULT_NATIVE_L16 = PROJECT_ROOT / "data/configs_phi4_2d/lam1p0_kappac0p340301_L16/configs.npz"
DEFAULT_OUT = PROJECT_ROOT / "perfect_blocking/perfect_blocking_lam1p0/tests/final/intended_two_stage_coarse_patch_L8to16_small_test_20260720"
OBS = [
    "action_density",
    "total_action",
    "phi2",
    "phi4",
    "local_kurtosis_ratio",
    "NN",
    "diag",
    "2nn",
    "m",
    "abs_m",
    "m2",
    "m4",
    "Binder_U4_proxy",
    "susceptibility_proxy",
    "G_00",
    "G_10",
    "G_01",
    "G_pmin_avg",
]
PLOT_OBS = ["action_density", "phi2", "phi4", "local_kurtosis_ratio", "NN"]


def default_saved_cycles(cycles: int) -> set[int]:
    saved = {0}
    saved.update(range(1, min(cycles, 20) + 1))
    saved.update(range(25, min(cycles, 100) + 1, 5))
    saved.update(range(110, cycles + 1, 10))
    saved.add(cycles)
    return {c for c in saved if 0 <= c <= cycles}


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
        fh.flush()
        os.fsync(fh.fileno())


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())


def split_detail(psi: np.ndarray) -> np.ndarray:
    return np.stack([psi[:, 0::2, 1::2], psi[:, 1::2, 0::2], psi[:, 1::2, 1::2]], axis=1).astype(np.float32)


def build_state(psi: np.ndarray, kernel: Any, action_f: ActionSpec, model: Any, stats: dict[str, Any], batch_size: int, device: torch.device) -> dict[str, np.ndarray]:
    coarse = psi[:, 0::2, 0::2].astype(np.float32)
    detail = split_detail(psi)
    phi, _ = inverse_kernel(psi.astype(np.float32), kernel)
    sf = action_total(phi, action_f).astype(np.float64)
    _z, logj = infer_rqspline_latents_and_logj(model, coarse, detail, stats, batch_size=batch_size, device=device)
    blocked = apply_kernel(phi, kernel)[:, 0::2, 0::2]
    return {
        "psi": psi.astype(np.float32),
        "coarse": coarse.astype(np.float32),
        "detail": detail.astype(np.float32),
        "phi": phi.astype(np.float32),
        "S_f": sf,
        "logJ": logj.astype(np.float64),
        "reblock": np.max(np.abs(blocked.astype(np.float64) - coarse.astype(np.float64)), axis=(1, 2)),
        "nonfinite": np.sum(~np.isfinite(phi).reshape(len(phi), -1), axis=1),
    }


def update_state_from_accepted(current: dict[str, np.ndarray], prop: dict[str, np.ndarray], accept: np.ndarray) -> None:
    for key in current:
        current[key][accept] = prop[key][accept]


def append_observables(rows: list[dict[str, Any]], method: str, patch_size: int, cycle: int, state: dict[str, np.ndarray], source_idx: np.ndarray, coarse_action: np.ndarray) -> None:
    obs, g = per_config_observables(state["phi"], ActionSpec("phi4_nn", 1.0, 0.340301))
    obs = dict(obs)
    obs.update(g)
    obs["abs_m"] = np.abs(obs["m"])
    obs["Binder_U4_proxy"] = 1.0 - obs["m4"] / np.maximum(3.0 * obs["m2"] * obs["m2"], 1.0e-300)
    obs["susceptibility_proxy"] = 16 * 16 * obs["m2"]
    for chain in range(len(state["phi"])):
        row = {
            "method": method,
            "coarse_patch_size": patch_size,
            "chain_id": chain,
            "cycle": cycle,
            "source_config_index": int(source_idx[chain]),
            "coarse_action": float(coarse_action[chain]),
            "fine_action": float(state["S_f"][chain]),
            "fine_action_density": float(state["S_f"][chain] / (16 * 16)),
            "logJ": float(state["logJ"][chain]),
            "reblocking_max_error": float(state["reblock"][chain]),
            "nonfinite_count": int(state["nonfinite"][chain]),
        }
        for key in OBS:
            row[key] = float(obs[key][chain])
        rows.append(row)


def detail_sweeps(
    current: dict[str, np.ndarray],
    *,
    method: str,
    patch_size_c: int,
    cycle: int,
    kernel: Any,
    action_f: ActionSpec,
    model: Any,
    stats: dict[str, Any],
    batch_size: int,
    device: torch.device,
    rng: np.random.Generator,
    detail_passes: int,
    detail_patch_size: int,
    detail_sigma: float,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    psi = current["psi"].copy()
    old_coarse = psi[:, 0::2, 0::2].copy()
    phi = current["phi"].copy()
    current_sf = current["S_f"].copy()
    n_patch = patches_per_pass(16, detail_patch_size)
    for dpass in range(detail_passes):
        for patch_idx in range(n_patch):
            x0 = int(rng.integers(0, 16))
            y0 = int(rng.integers(0, 16))
            mask = fine_detail_mask(16, x0, y0, detail_patch_size)
            prop_psi = psi.copy()
            noise = detail_sigma * rng.standard_normal((len(psi), int(mask.sum()))).astype(np.float32)
            prop_psi[:, mask] += noise
            prop_psi[:, 0::2, 0::2] = old_coarse
            prop_phi, _ = inverse_kernel(prop_psi, kernel)
            prop_sf = action_total(prop_phi, action_f).astype(np.float64)
            delta_sf = prop_sf - current_sf
            logR = -delta_sf
            accept = np.log(rng.random(len(psi))) < np.minimum(0.0, logR)
            restore = np.zeros(len(psi), dtype=np.float64)
            if np.any(~accept):
                restore[~accept] = np.max(np.abs(psi[~accept] - current["psi"][~accept]), axis=(1, 2))
            if np.any(accept):
                psi[accept] = prop_psi[accept]
                phi[accept] = prop_phi[accept]
                current_sf[accept] = prop_sf[accept]
            rows.append(
                {
                    "method": method,
                    "coarse_patch_size": patch_size_c,
                    "cycle": cycle,
                    "detail_pass": dpass,
                    "patch_index": patch_idx,
                    "patch_x": x0,
                    "patch_y": y0,
                    "attempts": len(psi),
                    "accepted": int(np.sum(accept)),
                    "acceptance": float(np.mean(accept)),
                    "DeltaSf_mean": float(np.mean(delta_sf)),
                    "DeltaSf_std": float(np.std(delta_sf, ddof=1)),
                    "logR_mean": float(np.mean(logR)),
                    "logR_std": float(np.std(logR, ddof=1)),
                    "coarse_unchanged_error": float(np.max(np.abs(psi[:, 0::2, 0::2] - old_coarse))),
                    "restore_error_if_rejected": float(np.max(restore)),
                }
            )
    new_state = build_state(psi, kernel, action_f, model, stats, batch_size, device)
    return new_state, rows


def one_coarse_attempt(
    current: dict[str, np.ndarray],
    current_sc: np.ndarray,
    *,
    method: str,
    patch_size: int,
    cycle: int,
    kernel: Any,
    action_c: ActionSpec,
    action_f: ActionSpec,
    model: Any,
    stats: dict[str, Any],
    batch_size: int,
    device: torch.device,
    rng: np.random.Generator,
    sigma_c: float,
) -> tuple[dict[str, np.ndarray], np.ndarray, list[dict[str, Any]], list[dict[str, Any]]]:
    n = len(current["psi"])
    rows1: list[dict[str, Any]] = []
    rows2: list[dict[str, Any]] = []
    old = {key: val.copy() for key, val in current.items()}
    old_sc = current_sc.copy()
    x0 = int(rng.integers(0, 8))
    y0 = int(rng.integers(0, 8))
    mask = coarse_patch_mask(8, x0, y0, patch_size)
    prop_coarse = current["coarse"].copy()
    noise = sigma_c * rng.standard_normal((n, int(mask.sum()))).astype(np.float32)
    prop_coarse[:, mask] += noise
    prop_sc = action_total(prop_coarse, action_c).astype(np.float64)
    delta_sc = prop_sc - current_sc
    log_stage1 = -delta_sc
    stage1_accept = np.ones(n, dtype=bool)
    if method == "intended":
        stage1_accept = np.log(rng.random(n)) < np.minimum(0.0, log_stage1)
    detail = current["detail"].copy()
    prop_psi = assemble_psi(prop_coarse, detail).astype(np.float32)
    prop = build_state(prop_psi, kernel, action_f, model, stats, batch_size, device)
    delta_sf = prop["S_f"] - current["S_f"]
    delta_logj = prop["logJ"] - current["logJ"]
    if method == "intended":
        logR = -delta_sf + delta_sc + delta_logj
    elif method == "existing":
        logR = -delta_sf + delta_logj
    else:
        raise ValueError(method)
    stage2_accept = np.zeros(n, dtype=bool)
    active = stage1_accept
    if np.any(active):
        stage2_accept[active] = np.log(rng.random(np.sum(active))) < np.minimum(0.0, logR[active])
    total_accept = stage1_accept & stage2_accept
    new_current = {key: val.copy() for key, val in current.items()}
    new_sc = current_sc.copy()
    update_state_from_accepted(new_current, prop, total_accept)
    new_sc[total_accept] = prop_sc[total_accept]
    for chain in range(n):
        log_t_forward = min(0.0, float(-delta_sc[chain]))
        log_t_reverse = min(0.0, float(delta_sc[chain]))
        log_tc_ratio = log_t_reverse - log_t_forward
        fine_reverse = -float(logR[chain])
        rows1.append(
            {
                "method": method,
                "coarse_patch_size": patch_size,
                "cycle": cycle,
                "chain_id": chain,
                "patch_x": x0,
                "patch_y": y0,
                "sigma_c": sigma_c,
                "stage1_attempted": 1,
                "stage1_accepted": int(stage1_accept[chain]),
                "DeltaSc": float(delta_sc[chain]),
                "log_stage1_raw": float(log_stage1[chain]),
                "log_T_reverse_over_forward": log_tc_ratio,
                "coarse_reversibility_residual": float(log_tc_ratio - delta_sc[chain]),
                "patch_l2": float(np.sqrt(np.sum(noise[chain].astype(np.float64) ** 2))),
            }
        )
        rows2.append(
            {
                "method": method,
                "coarse_patch_size": patch_size,
                "cycle": cycle,
                "chain_id": chain,
                "stage1_accepted": int(stage1_accept[chain]),
                "stage2_attempted": int(stage1_accept[chain]),
                "stage2_accepted": int(stage2_accept[chain]),
                "total_coarse_accepted": int(total_accept[chain]),
                "minus_DeltaSf": float(-delta_sf[chain]),
                "DeltaSc": float(delta_sc[chain]),
                "DeltaLogJ": float(delta_logj[chain]),
                "logR_fine_correction": float(logR[chain]),
                "reverse_logR_fine_correction": fine_reverse,
                "fine_correction_antisymmetry_residual": float(logR[chain] + fine_reverse),
                "old_Sf": float(old["S_f"][chain]),
                "proposed_Sf": float(prop["S_f"][chain]),
                "old_Sc": float(old_sc[chain]),
                "proposed_Sc": float(prop_sc[chain]),
                "old_logJ": float(old["logJ"][chain]),
                "proposed_logJ": float(prop["logJ"][chain]),
                "details_unchanged_during_coarse": float(np.max(np.abs(prop["detail"][chain] - old["detail"][chain]))),
                "restore_error_if_rejected": float(np.max(np.abs(new_current["psi"][chain] - old["psi"][chain]))) if not total_accept[chain] else 0.0,
                "reblocking_max_error": float(new_current["reblock"][chain]),
                "nonfinite_count": int(new_current["nonfinite"][chain]),
            }
        )
    return new_current, new_sc, rows1, rows2


def run_variant(
    *,
    method: str,
    patch_size: int,
    init_psi: np.ndarray,
    source_idx: np.ndarray,
    kernel: Any,
    action_c: ActionSpec,
    action_f: ActionSpec,
    model: Any,
    stats: dict[str, Any],
    batch_size: int,
    device: torch.device,
    seed: int,
    cycles: int,
    sigma_c: float,
    detail_passes: int,
    detail_patch_size: int,
    detail_sigma: float,
    save_cycles: set[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    current = build_state(init_psi.copy(), kernel, action_f, model, stats, batch_size, device)
    current_sc = action_total(current["coarse"], action_c).astype(np.float64)
    stage1_rows: list[dict[str, Any]] = []
    stage2_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    obs_rows: list[dict[str, Any]] = []
    if 0 in save_cycles:
        append_observables(obs_rows, method, patch_size, 0, current, source_idx, current_sc)
    for cycle in range(1, cycles + 1):
        current, current_sc, s1, s2 = one_coarse_attempt(
            current,
            current_sc,
            method=method,
            patch_size=patch_size,
            cycle=cycle,
            kernel=kernel,
            action_c=action_c,
            action_f=action_f,
            model=model,
            stats=stats,
            batch_size=batch_size,
            device=device,
            rng=rng,
            sigma_c=sigma_c,
        )
        stage1_rows.extend(s1)
        stage2_rows.extend(s2)
        current, drows = detail_sweeps(
            current,
            method=method,
            patch_size_c=patch_size,
            cycle=cycle,
            kernel=kernel,
            action_f=action_f,
            model=model,
            stats=stats,
            batch_size=batch_size,
            device=device,
            rng=rng,
            detail_passes=detail_passes,
            detail_patch_size=detail_patch_size,
            detail_sigma=detail_sigma,
        )
        current_sc = action_total(current["coarse"], action_c).astype(np.float64)
        detail_rows.extend(drows)
        if cycle in save_cycles:
            append_observables(obs_rows, method, patch_size, cycle, current, source_idx, current_sc)
        print(f"P{patch_size} {method} cycle {cycle}: coarse accepted {sum(r['total_coarse_accepted'] for r in s2)}/{len(s2)} detail accepted {sum(r['accepted'] for r in drows)}/{sum(r['attempts'] for r in drows)}", flush=True)
    return stage1_rows, stage2_rows, detail_rows, obs_rows


def summarize_variant(stage1: list[dict[str, Any]], stage2: list[dict[str, Any]], detail: list[dict[str, Any]], method: str, patch_size: int) -> dict[str, Any]:
    s1 = np.asarray([int(r["stage1_accepted"]) for r in stage1], dtype=np.float64)
    s2_attempt = np.asarray([int(r["stage2_attempted"]) for r in stage2], dtype=np.float64)
    s2_acc = np.asarray([int(r["stage2_accepted"]) for r in stage2 if int(r["stage2_attempted"])], dtype=np.float64)
    total = np.asarray([int(r["total_coarse_accepted"]) for r in stage2], dtype=np.float64)
    d_acc = sum(int(r["accepted"]) for r in detail)
    d_att = sum(int(r["attempts"]) for r in detail)
    out = {
        "method": method,
        "coarse_patch_size": patch_size,
        "stage1_attempts": len(stage1),
        "stage1_acceptance": float(np.mean(s1)) if len(s1) else float("nan"),
        "stage2_attempts": int(np.sum(s2_attempt)),
        "stage2_acceptance_conditional": float(np.mean(s2_acc)) if len(s2_acc) else float("nan"),
        "total_coarse_acceptance": float(np.mean(total)) if len(total) else float("nan"),
        "detail_attempts": d_att,
        "detail_acceptance": float(d_acc / d_att) if d_att else float("nan"),
        "max_coarse_reversibility_residual": float(np.max(np.abs([float(r["coarse_reversibility_residual"]) for r in stage1]))) if stage1 else float("nan"),
        "max_fine_antisymmetry_residual": float(np.max(np.abs([float(r["fine_correction_antisymmetry_residual"]) for r in stage2]))) if stage2 else float("nan"),
        "max_details_changed_during_coarse": float(np.max(np.abs([float(r["details_unchanged_during_coarse"]) for r in stage2]))) if stage2 else float("nan"),
        "max_rejected_restore_error": float(np.max(np.abs([float(r["restore_error_if_rejected"]) for r in stage2]))) if stage2 else float("nan"),
        "max_reblocking_error": float(np.max(np.abs([float(r["reblocking_max_error"]) for r in stage2]))) if stage2 else float("nan"),
        "nonfinite_count": int(np.sum([int(r["nonfinite_count"]) for r in stage2])),
    }
    for key, col in [("DeltaSc", "DeltaSc"), ("DeltaSf", "minus_DeltaSf"), ("DeltaLogJ", "DeltaLogJ"), ("logR", "logR_fine_correction")]:
        vals = np.asarray([float(r[col]) for r in stage2], dtype=np.float64)
        if key == "DeltaSf":
            vals = -vals
        out[f"{key}_mean"] = float(np.mean(vals))
        out[f"{key}_std"] = float(np.std(vals, ddof=1))
        out[f"{key}_min"] = float(np.min(vals))
        out[f"{key}_max"] = float(np.max(vals))
    return out


def histogram_edges(samples: list[np.ndarray]) -> np.ndarray:
    vals = np.concatenate([np.asarray(s, dtype=np.float64)[np.isfinite(s)] for s in samples])
    lo = float(np.min(vals))
    hi = float(np.max(vals))
    if lo == hi:
        return np.linspace(lo - 0.5, hi + 0.5, 61)
    return np.linspace(lo, hi, 81)


def distribution_metrics(vals: np.ndarray, native: np.ndarray, bins: np.ndarray) -> dict[str, float]:
    vals = np.asarray(vals, dtype=np.float64)
    native = np.asarray(native, dtype=np.float64)
    widths = np.diff(bins)
    hp, _ = np.histogram(vals, bins=bins, density=False)
    hn, _ = np.histogram(native, bins=bins, density=False)
    pp = hp.astype(np.float64) / max(float(np.sum(hp)), 1.0)
    pn = hn.astype(np.float64) / max(float(np.sum(hn)), 1.0)
    tv = 0.5 * float(np.sum(np.abs(pp - pn)))
    mix = 0.5 * (pp + pn)
    mask_p = pp > 0
    mask_n = pn > 0
    js = 0.5 * float(np.sum(pp[mask_p] * np.log(pp[mask_p] / mix[mask_p]))) + 0.5 * float(np.sum(pn[mask_n] * np.log(pn[mask_n] / mix[mask_n])))
    dp = pp / np.maximum(widths, 1.0e-300)
    dn = pn / np.maximum(widths, 1.0e-300)
    ovl = float(np.sum(np.minimum(dp, dn) * widths))
    se_vals = float(np.std(vals, ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else float("nan")
    se_native = float(np.std(native, ddof=1) / math.sqrt(len(native))) if len(native) > 1 else float("nan")
    combined = math.sqrt(se_vals * se_vals + se_native * se_native) if np.isfinite(se_vals) and np.isfinite(se_native) else float("nan")
    native_std = float(np.std(native, ddof=1))
    qn = {f"native_q{q:02d}": float(np.percentile(native, q)) for q in [5, 50, 95, 99]}
    qv = {f"q{q:02d}": float(np.percentile(vals, q)) for q in [5, 50, 95, 99]}
    return {
        "n": int(len(vals)),
        "native_n": int(len(native)),
        "mean": float(np.mean(vals)),
        "stderr": se_vals,
        "native_mean": float(np.mean(native)),
        "native_stderr": se_native,
        "std": float(np.std(vals, ddof=1)),
        "native_std": native_std,
        "mean_shift_combined_se": float((np.mean(vals) - np.mean(native)) / combined) if combined and combined > 0 else float("nan"),
        "mean_shift_native_sigma": float((np.mean(vals) - np.mean(native)) / native_std) if native_std > 0 else float("nan"),
        "std_ratio": float(np.std(vals, ddof=1) / native_std) if native_std > 0 else float("nan"),
        "KS": float(scipy_stats.ks_2samp(vals, native).statistic),
        "TV": tv,
        "JS": js,
        "overlap": ovl,
        "W1": float(scipy_stats.wasserstein_distance(vals, native)),
        "coverage_below_native_q05": float(np.mean(vals < qn["native_q05"])),
        "coverage_above_native_q95": float(np.mean(vals > qn["native_q95"])),
        "coverage_above_native_q99": float(np.mean(vals > qn["native_q99"])),
        **qv,
        **qn,
    }


def native_observable_arrays(native16: np.ndarray) -> dict[str, np.ndarray]:
    obs, g = per_config_observables(native16.astype(np.float32), ActionSpec("phi4_nn", 1.0, 0.340301))
    out = dict(obs)
    out.update(g)
    out["abs_m"] = np.abs(out["m"])
    out["Binder_U4_proxy"] = 1.0 - out["m4"] / np.maximum(3.0 * out["m2"] * out["m2"], 1.0e-300)
    out["susceptibility_proxy"] = 16 * 16 * out["m2"]
    return out


def metrics_and_plots(out_dir: Path, obs_rows: list[dict[str, Any]], native16: np.ndarray, saved_cycles: set[int]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    native_obs = native_observable_arrays(native16)
    methods = sorted({r["method"] for r in obs_rows})
    patch_sizes = sorted({int(r["coarse_patch_size"]) for r in obs_rows})
    dist_rows: list[dict[str, Any]] = []
    time_rows: list[dict[str, Any]] = []
    tail_rows: list[dict[str, Any]] = []
    for patch_size in patch_sizes:
        cycles_present = sorted({int(r["cycle"]) for r in obs_rows if int(r["coarse_patch_size"]) == patch_size})
        for method in methods:
            for cycle in cycles_present:
                current_rows = [r for r in obs_rows if r["method"] == method and int(r["coarse_patch_size"]) == patch_size and int(r["cycle"]) == cycle]
                if not current_rows:
                    continue
                for obs in OBS:
                    vals = np.asarray([float(r[obs]) for r in current_rows], dtype=np.float64)
                    bins = histogram_edges([native_obs[obs], vals])
                    row = {
                        "method": method,
                        "coarse_patch_size": patch_size,
                        "cycle": cycle,
                        "observable": obs,
                    }
                    row.update(distribution_metrics(vals, native_obs[obs], bins))
                    dist_rows.append(row)
                    time_rows.append({k: row[k] for k in ["method", "coarse_patch_size", "cycle", "observable", "n", "mean", "stderr", "native_mean", "native_stderr", "mean_shift_combined_se", "mean_shift_native_sigma", "std_ratio"]})
                    if obs in PLOT_OBS:
                        tail_rows.append({k: row[k] for k in ["method", "coarse_patch_size", "cycle", "observable", "coverage_below_native_q05", "coverage_above_native_q95", "coverage_above_native_q99", "q05", "q50", "q95", "q99", "native_q05", "native_q50", "native_q95", "native_q99"]})
        hist_cycles = [c for c in [0, 20, 50, 100, 200, 300] if c in cycles_present]
        for obs in PLOT_OBS:
            samples = [native_obs[obs]]
            for method in methods:
                for cycle in hist_cycles:
                    samples.append(np.asarray([float(r[obs]) for r in obs_rows if r["method"] == method and int(r["coarse_patch_size"]) == patch_size and int(r["cycle"]) == cycle], dtype=np.float64))
            bins = histogram_edges(samples)
            fig, ax = plt.subplots(figsize=(7.2, 4.7))
            ax.hist(native_obs[obs], bins=bins, density=True, histtype="step", lw=2.4, color="black", label="native L16")
            colors = {"existing": "tab:blue", "intended": "tab:orange"}
            styles = {0: "--", 20: "-", 50: "-", 100: "-", 200: "-", 300: "-"}
            for method in methods:
                for cycle in hist_cycles:
                    vals = np.asarray([float(r[obs]) for r in obs_rows if r["method"] == method and int(r["coarse_patch_size"]) == patch_size and int(r["cycle"]) == cycle], dtype=np.float64)
                    alpha = 0.55 if cycle not in {0, 300} else 0.95
                    lw = 1.3 if cycle not in {0, 300} else 2.0
                    ax.hist(vals, bins=bins, density=True, histtype="step", lw=lw, ls=styles.get(cycle, "-"), alpha=alpha, color=colors.get(method), label=f"{method} c{cycle}")
            ax.set_xlabel(obs)
            ax.set_ylabel("density")
            ax.grid(alpha=0.25)
            ax.legend(frameon=False, fontsize=7, ncol=2)
            fig.tight_layout()
            fig.savefig(fig_dir / f"P{patch_size}_{obs}_hist_cycles.pdf")
            plt.close(fig)
        fig, axes = plt.subplots(3, 2, figsize=(10.5, 9.0), sharex=True)
        axes = axes.ravel()
        for ax, obs in zip(axes, PLOT_OBS):
            for method in methods:
                means = []
                cycles = sorted({int(r["cycle"]) for r in obs_rows if r["method"] == method and int(r["coarse_patch_size"]) == patch_size})
                for cyc in cycles:
                    vals = [float(r[obs]) for r in obs_rows if r["method"] == method and int(r["coarse_patch_size"]) == patch_size and int(r["cycle"]) == cyc]
                    means.append(float(np.mean(vals)))
                ax.plot(cycles, means, marker="o", ms=2.5, lw=1.2, label=method)
            ax.axhline(float(np.mean(native_obs[obs])), color="black", lw=1.0, alpha=0.7)
            ax.set_ylabel(obs)
            ax.grid(alpha=0.25)
        axes[-1].axis("off")
        axes[0].legend(frameon=False, fontsize=8)
        for ax in axes[-2:]:
            ax.set_xlabel("cycle")
        fig.tight_layout()
        fig.savefig(fig_dir / f"P{patch_size}_observable_evolution_existing_vs_intended.pdf")
        plt.close(fig)
        fig, axes = plt.subplots(3, 2, figsize=(10.5, 9.0), sharex=True)
        axes = axes.ravel()
        for ax, obs in zip(axes, PLOT_OBS):
            for method in methods:
                cycles = sorted({int(r["cycle"]) for r in obs_rows if r["method"] == method and int(r["coarse_patch_size"]) == patch_size})
                shifts = []
                for cyc in cycles:
                    vals = np.asarray([float(r[obs]) for r in obs_rows if r["method"] == method and int(r["coarse_patch_size"]) == patch_size and int(r["cycle"]) == cyc], dtype=np.float64)
                    shifts.append(float((np.mean(vals) - np.mean(native_obs[obs])) / np.std(native_obs[obs], ddof=1)))
                ax.plot(cycles, shifts, marker="o", ms=2.5, lw=1.2, label=method)
            ax.axhline(0.0, color="black", lw=1.0, alpha=0.7)
            ax.set_ylabel(obs)
            ax.grid(alpha=0.25)
        axes[-1].axis("off")
        axes[0].legend(frameon=False, fontsize=8)
        for ax in axes[-2:]:
            ax.set_xlabel("cycle")
        fig.tight_layout()
        fig.savefig(fig_dir / f"P{patch_size}_distance_from_native_existing_vs_intended.pdf")
        plt.close(fig)
    return time_rows, dist_rows, tail_rows


def observable_change_rows(obs_rows: list[dict[str, Any]], native16: np.ndarray) -> list[dict[str, Any]]:
    native_obs = native_observable_arrays(native16)
    methods = sorted({r["method"] for r in obs_rows})
    patch_sizes = sorted({int(r["coarse_patch_size"]) for r in obs_rows})
    cycles = sorted({int(r["cycle"]) for r in obs_rows})
    rows: list[dict[str, Any]] = []
    for method in methods:
        for patch_size in patch_sizes:
            base_rows = [r for r in obs_rows if r["method"] == method and int(r["coarse_patch_size"]) == patch_size and int(r["cycle"]) == 0]
            if not base_rows:
                continue
            for obs in OBS:
                native_mean = float(np.mean(native_obs[obs]))
                native_std = float(np.std(native_obs[obs], ddof=1))
                initial_vals = np.asarray([float(r[obs]) for r in base_rows], dtype=np.float64)
                initial_mean = float(np.mean(initial_vals))
                initial_shift = float((initial_mean - native_mean) / max(native_std, 1.0e-300))
                for cycle in cycles:
                    current_rows = [r for r in obs_rows if r["method"] == method and int(r["coarse_patch_size"]) == patch_size and int(r["cycle"]) == cycle]
                    if not current_rows:
                        continue
                    vals = np.asarray([float(r[obs]) for r in current_rows], dtype=np.float64)
                    current_mean = float(np.mean(vals))
                    current_shift = float((current_mean - native_mean) / max(native_std, 1.0e-300))
                    rows.append(
                        {
                            "method": method,
                            "coarse_patch_size": patch_size,
                            "cycle": cycle,
                            "observable": obs,
                            "native_mean": native_mean,
                            "native_std": native_std,
                            "initial_mean": initial_mean,
                            "current_mean": current_mean,
                            "delta_from_cycle0": float(current_mean - initial_mean),
                            "initial_shift_native_sigma": initial_shift,
                            "current_shift_native_sigma": current_shift,
                            "shift_change_native_sigma": float(current_shift - initial_shift),
                            "abs_shift_improvement": float(abs(initial_shift) - abs(current_shift)),
                            "current_stderr": float(np.std(vals, ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else float("nan"),
                        }
                    )
    return rows


def late_window_rows(obs_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    windows = [(100, 150), (150, 200), (200, 250), (250, 300)]
    methods = sorted({r["method"] for r in obs_rows})
    patch_sizes = sorted({int(r["coarse_patch_size"]) for r in obs_rows})
    rows: list[dict[str, Any]] = []
    for method in methods:
        for patch_size in patch_sizes:
            for obs in PLOT_OBS:
                vals_by_window: list[tuple[tuple[int, int], np.ndarray]] = []
                for lo, hi in windows:
                    vals = np.asarray(
                        [
                            float(r[obs])
                            for r in obs_rows
                            if r["method"] == method
                            and int(r["coarse_patch_size"]) == patch_size
                            and lo <= int(r["cycle"]) <= hi
                        ],
                        dtype=np.float64,
                    )
                    if len(vals):
                        vals_by_window.append(((lo, hi), vals))
                for i in range(1, len(vals_by_window)):
                    (lo0, hi0), v0 = vals_by_window[i - 1]
                    (lo1, hi1), v1 = vals_by_window[i]
                    se = math.sqrt(np.var(v0, ddof=1) / len(v0) + np.var(v1, ddof=1) / len(v1))
                    rows.append(
                        {
                            "method": method,
                            "coarse_patch_size": patch_size,
                            "observable": obs,
                            "window_a": f"{lo0}-{hi0}",
                            "window_b": f"{lo1}-{hi1}",
                            "mean_a": float(np.mean(v0)),
                            "mean_b": float(np.mean(v1)),
                            "mean_difference": float(np.mean(v1) - np.mean(v0)),
                            "difference_combined_se": float((np.mean(v1) - np.mean(v0)) / se) if se > 0 else float("nan"),
                            "KS": float(scipy_stats.ks_2samp(v0, v1).statistic),
                        }
                    )
    return rows


def acceptance_plots(out_dir: Path, stage1: list[dict[str, Any]], stage2: list[dict[str, Any]], detail: list[dict[str, Any]]) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    methods = sorted({r["method"] for r in stage2})
    patch_sizes = sorted({int(r["coarse_patch_size"]) for r in stage2})
    for patch_size in patch_sizes:
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        for method in methods:
            cycles = sorted({int(r["cycle"]) for r in stage2 if r["method"] == method and int(r["coarse_patch_size"]) == patch_size})
            stage1_acc = []
            stage2_acc = []
            total_acc = []
            detail_acc = []
            for cycle in cycles:
                s1 = [int(r["stage1_accepted"]) for r in stage1 if r["method"] == method and int(r["coarse_patch_size"]) == patch_size and int(r["cycle"]) == cycle]
                s2 = [int(r["stage2_accepted"]) for r in stage2 if r["method"] == method and int(r["coarse_patch_size"]) == patch_size and int(r["cycle"]) == cycle and int(r["stage2_attempted"])]
                tot = [int(r["total_coarse_accepted"]) for r in stage2 if r["method"] == method and int(r["coarse_patch_size"]) == patch_size and int(r["cycle"]) == cycle]
                dacc = sum(int(r["accepted"]) for r in detail if r["method"] == method and int(r["coarse_patch_size"]) == patch_size and int(r["cycle"]) == cycle)
                datt = sum(int(r["attempts"]) for r in detail if r["method"] == method and int(r["coarse_patch_size"]) == patch_size and int(r["cycle"]) == cycle)
                stage1_acc.append(float(np.mean(s1)) if s1 else float("nan"))
                stage2_acc.append(float(np.mean(s2)) if s2 else float("nan"))
                total_acc.append(float(np.mean(tot)) if tot else float("nan"))
                detail_acc.append(float(dacc / datt) if datt else float("nan"))
            ax.plot(cycles, stage1_acc, lw=1.2, label=f"{method} stage1")
            ax.plot(cycles, stage2_acc, lw=1.2, label=f"{method} stage2|stage1")
            ax.plot(cycles, total_acc, lw=1.8, label=f"{method} total coarse")
            ax.plot(cycles, detail_acc, lw=1.2, label=f"{method} detail")
        ax.set_xlabel("cycle")
        ax.set_ylabel("acceptance")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(fig_dir / f"P{patch_size}_acceptance_vs_cycle.pdf")
        plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--flow-checkpoint", type=Path, default=DEFAULT_FLOW)
    ap.add_argument("--kernel", type=Path, default=DEFAULT_KERNEL)
    ap.add_argument("--native-l8", type=Path, default=DEFAULT_NATIVE_L8)
    ap.add_argument("--native-l16", type=Path, default=DEFAULT_NATIVE_L16)
    ap.add_argument("--n-chains", type=int, default=32)
    ap.add_argument("--cycles", type=int, default=20)
    ap.add_argument("--coarse-patch-sizes", default="2,4")
    ap.add_argument("--methods", default="existing,intended", help="Comma-separated methods: existing,intended")
    ap.add_argument("--coarse-sigma", type=float, default=0.06)
    ap.add_argument("--detail-passes", type=int, default=5)
    ap.add_argument("--detail-patch-size", type=int, default=8)
    ap.add_argument("--detail-sigma", type=float, default=0.04)
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20260720)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"writing to {out_dir}", flush=True)

    device = torch.device("cpu")
    action_c = ActionSpec("phi4_nn", 1.0, 0.340301)
    action_f = ActionSpec("phi4_nn", 1.0, 0.340301)
    kernel, _ = load_kernel(args.kernel)
    ckpt = torch.load(args.flow_checkpoint, map_location=device, weights_only=False)
    model, load_report = build_model_from_checkpoint(ckpt, lattice_size=8, device=device)
    stats = stationary_stats(ckpt["state"]["stats"], lc=8)
    native_l8 = load_phi(args.native_l8)
    native_l16 = load_phi(args.native_l16)
    source_idx = np.arange(args.start_index, args.start_index + args.n_chains, dtype=np.int64)
    coarse = native_l8[source_idx].astype(np.float32)
    detail, _logq, _zmax, _logdet = sample_model_lattice(model, coarse, stats, batch_size=args.batch_size, device=device, seed=args.seed + 17)
    init_psi = assemble_psi(coarse, detail).astype(np.float32)
    init_phi, _ = inverse_kernel(init_psi, kernel)
    init_reblock = float(np.max(np.abs(apply_kernel(init_phi, kernel)[:, 0::2, 0::2].astype(np.float64) - coarse.astype(np.float64))))
    if init_reblock > 5.0e-6:
        raise RuntimeError(f"initial reblocking failed: {init_reblock}")
    print(f"initialized {args.n_chains} L8->L16 chains; reblock={init_reblock:.3e}", flush=True)

    patch_sizes = [int(x) for x in args.coarse_patch_sizes.split(",") if x.strip()]
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    for method in methods:
        if method not in {"existing", "intended"}:
            raise ValueError(f"unknown method {method}")
    save_cycles = default_saved_cycles(args.cycles)
    all_stage1: list[dict[str, Any]] = []
    all_stage2: list[dict[str, Any]] = []
    all_detail: list[dict[str, Any]] = []
    all_obs: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for patch_size in patch_sizes:
        for method in methods:
            seed = args.seed + 10000 * patch_size + (0 if method == "existing" else 1000)
            s1, s2, drows, orows = run_variant(
                method=method,
                patch_size=patch_size,
                init_psi=init_psi,
                source_idx=source_idx,
                kernel=kernel,
                action_c=action_c,
                action_f=action_f,
                model=model,
                stats=stats,
                batch_size=args.batch_size,
                device=device,
                seed=seed,
                cycles=args.cycles,
                sigma_c=args.coarse_sigma,
                detail_passes=args.detail_passes,
                detail_patch_size=args.detail_patch_size,
                detail_sigma=args.detail_sigma,
                save_cycles=save_cycles,
            )
            all_stage1.extend(s1)
            all_stage2.extend(s2)
            all_detail.extend(drows)
            all_obs.extend(orows)
            summary_rows.append(summarize_variant(s1, s2, drows, method, patch_size))

    write_csv(out_dir / "stage1_coarse_attempts.csv", all_stage1)
    write_csv(out_dir / "stage2_fine_corrections.csv", all_stage2)
    write_csv(out_dir / "detail_attempts.csv", all_detail)
    write_csv(out_dir / "chain_observables.csv", all_obs)
    write_csv(out_dir / "acceptance_summary.csv", summary_rows)
    change_rows = observable_change_rows(all_obs, native_l16)
    write_csv(out_dir / "observable_change_from_initial.csv", change_rows)
    time_rows, dist_rows, tail_rows = metrics_and_plots(out_dir, all_obs, native_l16, save_cycles)
    write_csv(out_dir / "observable_time_series_summary.csv", time_rows)
    write_csv(out_dir / "observable_distance_from_native.csv", time_rows)
    write_csv(out_dir / "distribution_metrics_by_cycle.csv", dist_rows)
    write_csv(out_dir / "tail_metrics_by_cycle.csv", tail_rows)
    write_csv(out_dir / "late_window_stationarity.csv", late_window_rows(all_obs))
    write_csv(out_dir / "existing_vs_intended_metrics.csv", [r for r in dist_rows if int(r["cycle"]) == min(args.cycles, 20)])
    acceptance_plots(out_dir, all_stage1, all_stage2, all_detail)

    manifest = {
        "command": " ".join(sys.argv),
        "flow_checkpoint": str(args.flow_checkpoint),
        "flow_sha256": sha256(args.flow_checkpoint),
        "flow_epoch": ckpt.get("absolute_epoch", ckpt.get("epoch")),
        "kernel": str(args.kernel),
        "kernel_sha256": sha256(args.kernel),
        "kernel_sum": float(kernel_stencil_from_spec(kernel).sum()),
        "kernel_coefficients_include_eta_scale": bool(kernel.kernel_coefficients_include_eta_scale),
        "native_l8": str(args.native_l8),
        "native_l16": str(args.native_l16),
        "source_indices": source_idx.tolist(),
        "coarse_patch_sizes": patch_sizes,
        "methods": methods,
        "saved_cycles": sorted(save_cycles),
        "coarse_sigma": args.coarse_sigma,
        "detail_passes": args.detail_passes,
        "detail_patch_size": args.detail_patch_size,
        "detail_sigma": args.detail_sigma,
        "load_report": {k: str(v) for k, v in load_report.items()},
    }
    write_text(out_dir / "run_manifest.json", json.dumps(manifest, indent=2) + "\n")

    lines = [
        "# Intended Two-Stage Coarse Patch L8->L16 Small Test",
        "",
        "This is a guarded test path only. Production code was not modified.",
        "",
        f"- chains: `{args.n_chains}`",
        f"- cycles: `{args.cycles}`",
        f"- coarse patch sizes: `{patch_sizes}`",
        f"- methods: `{methods}`",
        f"- coarse sigma: `{args.coarse_sigma}`",
        f"- detail schedule per cycle: `{args.detail_passes}` physical-detail passes",
        f"- detail patch size/sigma: `{args.detail_patch_size}` / `{args.detail_sigma}`",
        f"- initial reblocking max error: `{init_reblock:.6g}`",
        f"- flow checkpoint: `{args.flow_checkpoint}`",
        f"- kernel: `{args.kernel}`",
        "",
        "## Acceptance Summary",
        "",
        "| method | Pc | stage1 acc | stage2 conditional acc | total coarse acc | detail acc | max coarse reversibility residual | max fine antisym residual | max reblock | nonfinite |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['method']} | {row['coarse_patch_size']} | {row['stage1_acceptance']:.6g} | {row['stage2_acceptance_conditional']:.6g} | {row['total_coarse_acceptance']:.6g} | {row['detail_acceptance']:.6g} | {row['max_coarse_reversibility_residual']:.3e} | {row['max_fine_antisymmetry_residual']:.3e} | {row['max_reblocking_error']:.3e} | {row['nonfinite_count']} |"
        )
    lines.extend(["", "## Log-Ratio Component Summary", "", "| method | Pc | DeltaSc mean/std/min/max | DeltaSf mean/std/min/max | DeltaLogJ mean/std/min/max | logR mean/std/min/max |", "|---|---:|---:|---:|---:|---:|"])
    for row in summary_rows:
        lines.append(
            f"| {row['method']} | {row['coarse_patch_size']} | {row['DeltaSc_mean']:.4g}/{row['DeltaSc_std']:.4g}/{row['DeltaSc_min']:.4g}/{row['DeltaSc_max']:.4g} | {row['DeltaSf_mean']:.4g}/{row['DeltaSf_std']:.4g}/{row['DeltaSf_min']:.4g}/{row['DeltaSf_max']:.4g} | {row['DeltaLogJ_mean']:.4g}/{row['DeltaLogJ_std']:.4g}/{row['DeltaLogJ_min']:.4g}/{row['DeltaLogJ_max']:.4g} | {row['logR_mean']:.4g}/{row['logR_std']:.4g}/{row['logR_min']:.4g}/{row['logR_max']:.4g} |"
        )
    lines.extend(["", f"## Cycle-{args.cycles} Observable Change From Initial", "", f"| method | Pc | observable | initial shift | cycle{args.cycles} shift | shift change | abs-shift improvement |", "|---|---:|---|---:|---:|---:|---:|"])
    for row in change_rows:
        if int(row["cycle"]) == args.cycles and row["observable"] in ["action_density", "phi2", "phi4", "local_kurtosis_ratio", "NN"]:
            lines.append(
                f"| {row['method']} | {row['coarse_patch_size']} | {row['observable']} | {row['initial_shift_native_sigma']:.4g} | {row['current_shift_native_sigma']:.4g} | {row['shift_change_native_sigma']:.4g} | {row['abs_shift_improvement']:.4g} |"
            )
    lines.extend(["", f"## Cycle-{args.cycles} Distance From Native", "", "| method | Pc | observable | mean | native mean | shift/native sigma | KS | TV | JS | overlap | q05/q50/q95/q99 |", "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for row in dist_rows:
        if int(row["cycle"]) == args.cycles and row["observable"] in PLOT_OBS:
            lines.append(
                f"| {row['method']} | {row['coarse_patch_size']} | {row['observable']} | {row['mean']:.6g} | {row['native_mean']:.6g} | {row['mean_shift_native_sigma']:.4g} | {row['KS']:.4g} | {row['TV']:.4g} | {row['JS']:.4g} | {row['overlap']:.4g} | {row['q05']:.4g}/{row['q50']:.4g}/{row['q95']:.4g}/{row['q99']:.4g} |"
            )
    write_text(out_dir / "summary.md", "\n".join(lines) + "\n")
    print(out_dir, flush=True)
    print("\n".join(lines[:24]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
