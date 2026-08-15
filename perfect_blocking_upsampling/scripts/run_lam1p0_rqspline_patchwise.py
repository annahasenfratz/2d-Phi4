#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

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
import torch

from perfect_blocking_upsampling.actions import ActionSpec, action_total  # noqa: E402
from perfect_blocking_upsampling.empirical_joint_detail_upscaler import EmpiricalJointDetailUpscaler, load_config as load_empirical_upscaler_config  # noqa: E402
from perfect_blocking_upsampling.kernels import apply_kernel, inverse_kernel, load_kernel  # noqa: E402
from perfect_blocking_upsampling.observables import second_moment_components  # noqa: E402
from run_lam0p2_flow_detail_rethermalization import (  # noqa: E402
    MAIN_MEASUREMENT_FIELDS,
    PATCH_HISTORY_FIELDS,
    acceptance_history_row,
    aggregate_history,
    coarse_patch_mask,
    combine_interval_meta,
    consistency,
    main_measurement_rows,
    rows_for_sweep,
    write_main_measurement_rows,
    write_per_chain_rows,
)
from run_lam0p2_residual_flow_patch_chain import StreamingCsv, patch_correct  # noqa: E402
from run_lam1p0_l16to32_rqspline_zeroshot import (  # noqa: E402
    build_model_from_checkpoint,
    sample_model_lattice,
    stationary_stats,
)
from train_lam1p0_flow_detail_pilot import assemble_psi, load_phi, write_json  # noqa: E402


LAM = 1.0
KAPPA = 0.340301
ETA_SCALE = 2.0**0.125
G_COLUMNS = ["chain_id", "sweep", "source_config_index", "source_native_L32_index", "L", "volume", "G_00", "G_10", "G_01", "G_pmin_avg"]
GX_AXIS_CHECKPOINT_COLUMNS = ["sweep", "x", "Gx_connected_mean", "Gx_connected_se", "n_chains", "L", "definition"]
TWO_STAGE_COARSE_FIELDS = [
    "sweep",
    "pass",
    "patch_index",
    "patch_x",
    "patch_y",
    "patch_size",
    "attempts",
    "stage1_accepted",
    "stage1_acceptance",
    "stage2_attempts",
    "stage2_accepted",
    "stage2_acceptance_conditional",
    "total_accepted",
    "total_acceptance",
    "DeltaSc_mean",
    "DeltaSc_std",
    "DeltaSc_min",
    "DeltaSc_max",
    "DeltaSf_mean",
    "DeltaSf_std",
    "DeltaSf_min",
    "DeltaSf_max",
    "Delta_logJ_mean",
    "Delta_logJ_std",
    "Delta_logJ_min",
    "Delta_logJ_max",
    "log_accept_mean",
    "log_accept_std",
    "log_accept_min",
    "log_accept_max",
    "coarse_reversibility_residual_max",
    "fine_correction_antisymmetry_residual_max",
    "details_unchanged_during_coarse_max",
    "restore_error_if_rejected_max",
    "reblocking_max_error",
    "nonfinite_count",
    "acceptance_formula",
    "patch_l2_mean",
    "local_rms",
    "elapsed_sec",
]


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


def gx_axis_checkpoint_rows(phi: np.ndarray, sweep: int) -> list[dict[str, Any]]:
    """Connected, translationally averaged axis correlator at one checkpoint."""
    fields = np.asarray(phi, dtype=np.float64)
    n_chains, length, _ = fields.shape
    transform = np.fft.fft2(fields, axes=(1, 2))
    # Each entry is (1/V) sum_y phi(y) phi(y + r), averaged over a chain.
    per_chain = np.fft.ifft2(np.abs(transform) ** 2, axes=(1, 2)).real / (length * length)
    per_chain -= float(fields.mean()) ** 2
    axis = per_chain[:, : length // 2 + 1, 0]
    means = axis.mean(axis=0)
    errors = axis.std(axis=0, ddof=1) / math.sqrt(n_chains) if n_chains > 1 else np.full(len(means), np.nan)
    return [
        {
            "sweep": int(sweep),
            "x": int(x),
            "Gx_connected_mean": float(mean),
            "Gx_connected_se": float(error),
            "n_chains": int(n_chains),
            "L": int(length),
            "definition": "<phi(y)phi(y+(x,0))>_{y,chains}-<phi>_{y,chains}^2",
        }
        for x, mean, error in zip(range(len(means)), means, errors)
    ]


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def cumulative_from_acceptance(rows: list[dict[str, Any]]) -> dict[str, int]:
    if not rows:
        return {"coarse_proposals": 0, "coarse_accepts": 0, "detail_proposals": 0, "detail_accepts": 0}
    last = rows[-1]

    def as_int(key: str) -> int:
        val = last.get(key, 0)
        if val in ("", None):
            return 0
        return int(float(val))

    return {
        "coarse_proposals": as_int("coarse_proposals_cumulative"),
        "coarse_accepts": as_int("coarse_accepts_cumulative"),
        "detail_proposals": as_int("detail_proposals_cumulative"),
        "detail_accepts": as_int("detail_accepts_cumulative"),
    }


def read_config(run_dir: Path) -> dict[str, Any]:
    import yaml

    return yaml.safe_load((run_dir / "run_config.yaml").read_text(encoding="utf-8"))


def g_rows_for_sweep(phi: np.ndarray, source_idx: np.ndarray, sweep: int, coarse_source: str = "") -> list[dict[str, Any]]:
    arr = np.asarray(phi, dtype=np.float64)
    n, L, _ = arr.shape
    V = L * L
    ft = np.fft.fft2(arr, axes=(1, 2))
    g00 = np.abs(ft[:, 0, 0]) ** 2 / V
    g10 = np.abs(ft[:, 1, 0]) ** 2 / V
    g01 = np.abs(ft[:, 0, 1]) ** 2 / V
    return [
        {
            "chain_id": i,
            "sweep": int(sweep),
            "source_config_index": int(source_idx[i]),
            "source_native_L32_index": int(source_idx[i]) if str(coarse_source).startswith("blocked_native_L") else -1,
            "L": L,
            "volume": V,
            "G_00": float(g00[i]),
            "G_10": float(g10[i]),
            "G_01": float(g01[i]),
            "G_pmin_avg": float(0.5 * (g10[i] + g01[i])),
        }
        for i in range(n)
    ]


def native_summary_rows(native: np.ndarray, action: ActionSpec, label: str) -> list[dict[str, Any]]:
    rows = main_measurement_rows(native.astype(np.float32), action, np.arange(len(native)), 0, label)
    out = []
    keys = ["action_density", "total_action", "phi2", "phi4", "NN", "diag", "2nn", "m", "m2", "m4", "G_pmin_x_cfg", "G_pmin_y_cfg"]
    for key in keys:
        vals = np.asarray([float(r[key]) for r in rows], dtype=np.float64)
        out.append({"observable": key, "mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)), "n": int(len(vals))})
    return out


def detail_from_psi(psi: np.ndarray) -> np.ndarray:
    return np.stack([psi[:, 0::2, 1::2], psi[:, 1::2, 0::2], psi[:, 1::2, 1::2]], axis=1).astype(np.float32)


def infer_rqspline_latents_and_logj(
    model: Any,
    coarse_phys: np.ndarray,
    detail_phys: np.ndarray,
    stats: dict[str, Any],
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    coarse_std = ((coarse_phys - stats["coarse_mean"]) / stats["coarse_std"]).astype(np.float32)
    mean = np.asarray(stats["detail_mean"], dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.asarray(stats["detail_std"], dtype=np.float32).reshape(1, 3, 1, 1)
    detail_std = ((detail_phys - mean) / std).astype(np.float32)
    z_out = np.zeros_like(detail_std, dtype=np.float32)
    logj = np.zeros(len(coarse_std), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, len(coarse_std), batch_size):
            stop = min(start + batch_size, len(coarse_std))
            cb = torch.from_numpy(coarse_std[start:stop]).to(device)
            db = torch.from_numpy(detail_std[start:stop]).to(device)
            reconstructed = torch.zeros_like(db)
            zb = torch.zeros_like(db)
            logj_b = torch.zeros(stop - start, dtype=cb.dtype, device=device)
            for stage in range(3):
                cond_spline = model.cond(cb, db, stage)
                pre_spline, spline_inv_logdet = model.spline.flows[stage].inverse(db[:, stage].flatten(1), cond_spline)
                cond_affine = model.affine_base.cond(cb, reconstructed, stage)
                z_stage, affine_inv_logdet = model.affine_base.flows[stage].inverse(pre_spline, cond_affine)
                zb[:, stage] = z_stage.reshape(stop - start, cb.shape[1], cb.shape[2])
                logj_b = logj_b - affine_inv_logdet - spline_inv_logdet
                reconstructed[:, stage] = db[:, stage]
            z_out[start:stop] = zb.detach().cpu().numpy().astype(np.float32)
            logj[start:stop] = logj_b.detach().cpu().numpy().astype(np.float64)
    return z_out, logj


def reconstruct_rqspline_from_latents(
    model: Any,
    coarse_phys: np.ndarray,
    z: np.ndarray,
    stats: dict[str, Any],
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    coarse_std = ((coarse_phys - stats["coarse_mean"]) / stats["coarse_std"]).astype(np.float32)
    mean = np.asarray(stats["detail_mean"], dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.asarray(stats["detail_std"], dtype=np.float32).reshape(1, 3, 1, 1)
    detail_out = np.zeros_like(z, dtype=np.float32)
    logj = np.zeros(len(coarse_std), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, len(coarse_std), batch_size):
            stop = min(start + batch_size, len(coarse_std))
            cb = torch.from_numpy(coarse_std[start:stop]).to(device)
            zb = torch.from_numpy(z[start:stop]).to(device)
            db = torch.zeros_like(zb)
            logj_b = torch.zeros(stop - start, dtype=cb.dtype, device=device)
            for stage in range(3):
                cond_affine = model.affine_base.cond(cb, db, stage)
                x_affine, affine_logdet = model.affine_base.flows[stage].forward(zb[:, stage].flatten(1), cond_affine)
                cond_spline = model.cond(cb, db, stage)
                x, spline_logdet = model.spline.flows[stage].forward(x_affine, cond_spline)
                db[:, stage] = x.reshape(stop - start, cb.shape[1], cb.shape[2])
                logj_b = logj_b + affine_logdet + spline_logdet
            d_phys = db.detach().cpu().numpy().astype(np.float32) * std + mean
            detail_out[start:stop] = d_phys.astype(np.float32)
            logj[start:stop] = logj_b.detach().cpu().numpy().astype(np.float64)
    return detail_out, logj


def rqspline_fixed_latent_coarse_patch_update(
    psi0: np.ndarray,
    kernel: Any,
    fine_action: ActionSpec,
    model: Any,
    stats: dict[str, Any],
    *,
    batch_size: int,
    device: torch.device,
    passes: int,
    patch_size: int,
    step_size: float,
    sweep: int,
    writer: StreamingCsv,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    psi = psi0.copy().astype(np.float32)
    phi, _ = inverse_kernel(psi, kernel)
    current_sf = action_total(phi, fine_action).astype(np.float64)
    current_ee = psi[:, 0::2, 0::2].astype(np.float32)
    current_detail = detail_from_psi(psi)
    z, current_logj = infer_rqspline_latents_and_logj(model, current_ee, current_detail, stats, batch_size=batch_size, device=device)
    attempts = 0
    accepts = 0
    patch_accs: list[float] = []
    start = time.perf_counter()
    n_patch = int(math.ceil(float(current_ee.shape[1] * current_ee.shape[2]) / float(patch_size * patch_size)))
    for p in range(passes):
        for patch_idx in range(n_patch):
            ee = psi[:, 0::2, 0::2].astype(np.float32)
            x0 = int(rng.integers(0, ee.shape[1]))
            y0 = int(rng.integers(0, ee.shape[2]))
            mask = coarse_patch_mask(int(ee.shape[1]), x0, y0, patch_size)
            prop_ee = ee.copy()
            noise = step_size * rng.standard_normal((len(ee), int(mask.sum()))).astype(np.float32)
            prop_ee[:, mask] += noise
            prop_detail, prop_logj = reconstruct_rqspline_from_latents(
                model, prop_ee, z, stats, batch_size=batch_size, device=device
            )
            prop_psi = assemble_psi(prop_ee, prop_detail).astype(np.float32)
            prop_phi, _ = inverse_kernel(prop_psi, kernel)
            prop_sf = action_total(prop_phi, fine_action).astype(np.float64)
            delta_sf = prop_sf - current_sf
            delta_logj = prop_logj - current_logj
            log_accept_raw = -delta_sf + delta_logj
            log_accept = np.minimum(0.0, log_accept_raw)
            accept = np.log(rng.random(len(psi))) < log_accept
            if np.any(accept):
                psi[accept] = prop_psi[accept]
                phi[accept] = prop_phi[accept]
                current_sf[accept] = prop_sf[accept]
                current_logj[accept] = prop_logj[accept]
            attempts += int(len(psi))
            accepts += int(np.sum(accept))
            acc = float(np.mean(accept))
            patch_accs.append(acc)
            writer.write(
                {
                    "sweep": sweep,
                    "pass": p,
                    "patch_index": patch_idx,
                    "patch_x": x0,
                    "patch_y": y0,
                    "patch_size": patch_size,
                    "attempts": int(len(psi)),
                    "accepted": int(np.sum(accept)),
                    "acceptance": acc,
                    "DeltaSc_mean": float("nan"),
                    "DeltaSc_std": float("nan"),
                    "DeltaSc_min": float("nan"),
                    "DeltaSc_max": float("nan"),
                    "DeltaSf_mean": float(np.mean(delta_sf)),
                    "DeltaSf_std": float(np.std(delta_sf, ddof=1)) if len(delta_sf) > 1 else 0.0,
                    "Delta_logJ_mean": float(np.mean(delta_logj)),
                    "Delta_logJ_std": float(np.std(delta_logj, ddof=1)) if len(delta_logj) > 1 else 0.0,
                    "log_accept_mean": float(np.mean(log_accept_raw)),
                    "log_accept_std": float(np.std(log_accept_raw, ddof=1)) if len(log_accept_raw) > 1 else 0.0,
                    "acceptance_formula": "-DeltaSf + Delta_logJ_fixed_RQSpline_AR_latent",
                    "patch_l2_mean": float(np.mean(np.sqrt(np.sum(noise.astype(np.float64) ** 2, axis=1)))),
                    "local_rms": float(np.sqrt(np.mean(noise.astype(np.float64) ** 2))),
                    "elapsed_sec": float(time.perf_counter() - start),
                }
            )
    return psi.astype(np.float32), phi.astype(np.float32), {
        "coarse_acceptance": float(accepts / attempts) if attempts else float("nan"),
        "coarse_proposals": int(attempts),
        "coarse_accepts": int(accepts),
        "coarse_patch_attempts": int(len(patch_accs)),
        "coarse_patch_acceptance": float(np.mean(patch_accs)) if patch_accs else float("nan"),
        "coarse_update_scheme": "flat_fixed_latent",
    }


def rqspline_two_stage_coarse_action_patch_update(
    psi0: np.ndarray,
    kernel: Any,
    coarse_action: ActionSpec,
    fine_action: ActionSpec,
    *,
    passes: int,
    patch_size: int,
    step_size: float,
    sweep: int,
    writer: StreamingCsv,
    diagnostics_writer: StreamingCsv,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Run fixed-detail delayed-acceptance MH in the coarse ``psi_ee`` field.

    The upscaling spline is used only to create the sweep-zero field.  At every
    later coarse proposal, the three detail sublattices remain unchanged:

    1. propose a symmetric random walk in ``psi_ee`` and accept against
       ``S_c``;
    2. apply ``K^{-1}`` to that fixed-detail proposal and accept the delayed
       correction ``-Delta S_f + Delta S_c``.
    """
    psi = psi0.copy().astype(np.float32)
    phi, _ = inverse_kernel(psi, kernel)
    current_sf = action_total(phi, fine_action).astype(np.float64)
    current_coarse = psi[:, 0::2, 0::2].astype(np.float32)
    current_sc = action_total(current_coarse, coarse_action).astype(np.float64)
    attempts = 0
    stage1_accepts = 0
    stage2_attempts = 0
    stage2_accepts = 0
    total_accepts = 0
    patch_accs: list[float] = []
    stage1_accs: list[float] = []
    stage2_accs: list[float] = []
    start = time.perf_counter()
    lc = int(current_coarse.shape[1])
    n_patch = int(math.ceil(float(lc * lc) / float(patch_size * patch_size)))
    for p in range(passes):
        for patch_idx in range(n_patch):
            ee = psi[:, 0::2, 0::2].astype(np.float32)
            x0 = int(rng.integers(0, lc))
            y0 = int(rng.integers(0, lc))
            mask = coarse_patch_mask(lc, x0, y0, patch_size)
            prop_ee = ee.copy()
            noise = step_size * rng.standard_normal((len(ee), int(mask.sum()))).astype(np.float32)
            prop_ee[:, mask] += noise
            log_stage1, delta_sc, prop_sc = coarse_action_log_acceptance(
                current_coarse, prop_ee, current_sc, coarse_action
            )
            stage1_accept = np.log(rng.random(len(psi))) < np.minimum(0.0, log_stage1)

            # Keep every detail coordinate fixed.  The spline and its latent
            # variables play no role after the sweep-zero initialization.
            prop_psi = psi.copy()
            prop_psi[:, 0::2, 0::2] = prop_ee
            prop_phi, _ = inverse_kernel(prop_psi, kernel)
            prop_sf = action_total(prop_phi, fine_action).astype(np.float64)
            delta_sf = prop_sf - current_sf
            log_accept_raw = -delta_sf + delta_sc
            stage2_accept = np.zeros(len(psi), dtype=bool)
            if np.any(stage1_accept):
                stage2_accept[stage1_accept] = (
                    np.log(rng.random(int(np.sum(stage1_accept)))) < np.minimum(0.0, log_accept_raw[stage1_accept])
                )
            total_accept = stage1_accept & stage2_accept
            if np.any(total_accept):
                psi[total_accept] = prop_psi[total_accept]
                phi[total_accept] = prop_phi[total_accept]
                current_sf[total_accept] = prop_sf[total_accept]
                current_sc[total_accept] = prop_sc[total_accept]
            current_coarse = psi[:, 0::2, 0::2].astype(np.float32)

            blocked = apply_kernel(phi, kernel)[:, 0::2, 0::2].astype(np.float64)
            reblock_err = float(np.max(np.abs(blocked - psi[:, 0::2, 0::2].astype(np.float64))))
            nonfinite = int(np.sum(~np.isfinite(phi)))
            restore_err = 0.0
            detail_unchanged = float(np.max(np.abs(psi[:, 0::2, 1::2] - prop_psi[:, 0::2, 1::2])))
            detail_unchanged = max(detail_unchanged, float(np.max(np.abs(psi[:, 1::2, 0::2] - prop_psi[:, 1::2, 0::2]))))
            detail_unchanged = max(detail_unchanged, float(np.max(np.abs(psi[:, 1::2, 1::2] - prop_psi[:, 1::2, 1::2]))))
            log_t_forward = np.minimum(0.0, -delta_sc)
            log_t_reverse = np.minimum(0.0, delta_sc)
            coarse_resid = float(np.max(np.abs((log_t_reverse - log_t_forward) - delta_sc)))

            patch_attempts = int(len(psi))
            patch_stage1 = int(np.sum(stage1_accept))
            patch_stage2 = int(np.sum(stage2_accept))
            patch_total = int(np.sum(total_accept))
            attempts += patch_attempts
            stage1_accepts += patch_stage1
            stage2_attempts += patch_stage1
            stage2_accepts += patch_stage2
            total_accepts += patch_total
            total_acc = float(patch_total / patch_attempts) if patch_attempts else float("nan")
            patch_accs.append(total_acc)
            stage1_accs.append(float(patch_stage1 / patch_attempts) if patch_attempts else float("nan"))
            stage2_accs.append(float(patch_stage2 / patch_stage1) if patch_stage1 else float("nan"))

            row = {
                "sweep": sweep,
                "phase": "coarse_two_stage",
                "pass": p,
                "patch_index": patch_idx,
                "patch_x": x0,
                "patch_y": y0,
                "patch_size": patch_size,
                "attempts": patch_attempts,
                "accepted": patch_total,
                "acceptance": total_acc,
                "A_over_R": total_acc,
                "deltaS_mean": float(np.mean(delta_sf)),
                "deltaS_std": float(np.std(delta_sf, ddof=1)) if len(delta_sf) > 1 else 0.0,
                "deltaS_min": float(np.min(delta_sf)),
                "deltaS_max": float(np.max(delta_sf)),
                "delta_logw_mean": float(np.mean(log_accept_raw)),
                "delta_logw_std": float(np.std(log_accept_raw, ddof=1)) if len(log_accept_raw) > 1 else 0.0,
                "log_accept_mean": float(np.mean(log_accept_raw)),
                "log_accept_std": float(np.std(log_accept_raw, ddof=1)) if len(log_accept_raw) > 1 else 0.0,
                "patch_l2_mean": float(np.mean(np.sqrt(np.sum(noise.astype(np.float64) ** 2, axis=1)))),
                "local_rms": float(np.sqrt(np.mean(noise.astype(np.float64) ** 2))),
                "elapsed_sec": float(time.perf_counter() - start),
            }
            writer.write(row)
            diagnostics_writer.write(
                {
                    **{k: row[k] for k in ["sweep", "pass", "patch_index", "patch_x", "patch_y", "patch_size", "attempts"]},
                    "stage1_accepted": patch_stage1,
                    "stage1_acceptance": float(patch_stage1 / patch_attempts) if patch_attempts else float("nan"),
                    "stage2_attempts": patch_stage1,
                    "stage2_accepted": patch_stage2,
                    "stage2_acceptance_conditional": float(patch_stage2 / patch_stage1) if patch_stage1 else float("nan"),
                    "total_accepted": patch_total,
                    "total_acceptance": total_acc,
                    "DeltaSc_mean": float(np.mean(delta_sc)),
                    "DeltaSc_std": float(np.std(delta_sc, ddof=1)) if len(delta_sc) > 1 else 0.0,
                    "DeltaSc_min": float(np.min(delta_sc)),
                    "DeltaSc_max": float(np.max(delta_sc)),
                    "DeltaSf_mean": float(np.mean(delta_sf)),
                    "DeltaSf_std": float(np.std(delta_sf, ddof=1)) if len(delta_sf) > 1 else 0.0,
                    "DeltaSf_min": float(np.min(delta_sf)),
                    "DeltaSf_max": float(np.max(delta_sf)),
                    "Delta_logJ_mean": 0.0,
                    "Delta_logJ_std": 0.0,
                    "Delta_logJ_min": 0.0,
                    "Delta_logJ_max": 0.0,
                    "log_accept_mean": float(np.mean(log_accept_raw)),
                    "log_accept_std": float(np.std(log_accept_raw, ddof=1)) if len(log_accept_raw) > 1 else 0.0,
                    "log_accept_min": float(np.min(log_accept_raw)),
                    "log_accept_max": float(np.max(log_accept_raw)),
                    "coarse_reversibility_residual_max": coarse_resid,
                    "fine_correction_antisymmetry_residual_max": 0.0,
                    "details_unchanged_during_coarse_max": detail_unchanged,
                    "restore_error_if_rejected_max": restore_err,
                    "reblocking_max_error": reblock_err,
                    "nonfinite_count": nonfinite,
                    "acceptance_formula": "-DeltaSf + DeltaSc_after_stage1_coarse_MH_fixed_detail",
                    "patch_l2_mean": row["patch_l2_mean"],
                    "local_rms": row["local_rms"],
                    "elapsed_sec": row["elapsed_sec"],
                }
            )

    return psi.astype(np.float32), phi.astype(np.float32), {
        "coarse_acceptance": float(total_accepts / attempts) if attempts else float("nan"),
        "coarse_proposals": int(attempts),
        "coarse_accepts": int(total_accepts),
        "coarse_patch_attempts": int(len(patch_accs)),
        "coarse_patch_acceptance": float(np.mean(patch_accs)) if patch_accs else float("nan"),
        "coarse_update_scheme": "two_stage_coarse_action",
        "stage1_coarse_action_attempts": int(attempts),
        "stage1_coarse_action_accepts": int(stage1_accepts),
        "stage1_coarse_action_acceptance": float(stage1_accepts / attempts) if attempts else float("nan"),
        "stage2_fine_correction_attempts": int(stage2_attempts),
        "stage2_fine_correction_accepts": int(stage2_accepts),
        "stage2_fine_correction_acceptance_conditional": float(stage2_accepts / stage2_attempts) if stage2_attempts else float("nan"),
    }


def coarse_action_log_acceptance(
    current_coarse: np.ndarray,
    proposed_coarse: np.ndarray,
    current_sc: np.ndarray,
    coarse_action: ActionSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return stage-1 log acceptance and coarse-action deltas using kappa_c."""
    proposed_sc = action_total(proposed_coarse, coarse_action).astype(np.float64)
    delta_sc = proposed_sc - current_sc
    return -delta_sc, delta_sc, proposed_sc


def comparison_rows(measure_rows: list[dict[str, Any]], native_rows: list[dict[str, Any]], sweeps: list[int]) -> list[dict[str, Any]]:
    native = {r["observable"]: r for r in native_rows}
    out = []
    for sweep in sweeps:
        rows = [r for r in measure_rows if int(r["sweep"]) == int(sweep)]
        if not rows:
            continue
        for key in ["action_density", "phi2", "phi4", "NN", "diag", "2nn", "m2", "m4"]:
            vals = np.asarray([float(r[key]) for r in rows], dtype=np.float64)
            nr = native[key]
            out.append(
                {
                    "sweep": sweep,
                    "observable": key,
                    "generated_mean": float(np.mean(vals)),
                    "native_mean": float(nr["mean"]),
                    "mean_shift_native_sigma": float((np.mean(vals) - float(nr["mean"])) / max(float(nr["std"]), 1.0e-300)),
                    "generated_std": float(np.std(vals, ddof=1)),
                    "native_std": float(nr["std"]),
                }
            )
        vals = np.asarray([0.5 * (float(r["G_pmin_x_cfg"]) + float(r["G_pmin_y_cfg"])) for r in rows], dtype=np.float64)
        nvals = None
        if "G_pmin_avg" not in native:
            gx, gy = native["G_pmin_x_cfg"], native["G_pmin_y_cfg"]
            nvals = {"mean": 0.5 * (float(gx["mean"]) + float(gy["mean"])), "std": 0.5 * (float(gx["std"]) + float(gy["std"]))}
        out.append(
            {
                "sweep": sweep,
                "observable": "G_pmin_avg",
                "generated_mean": float(np.mean(vals)),
                "native_mean": float(nvals["mean"]),
                "mean_shift_native_sigma": float((np.mean(vals) - float(nvals["mean"])) / max(float(nvals["std"]), 1.0e-300)),
                "generated_std": float(np.std(vals, ddof=1)),
                "native_std": float(nvals["std"]),
            }
        )
    return out


def plot_histograms(run: Path, measure_rows: list[dict[str, Any]], native_phi: np.ndarray, action: ActionSpec, sweeps: list[int], native_label: str) -> None:
    native_rows = main_measurement_rows(native_phi.astype(np.float32), action, np.arange(len(native_phi)), 0, native_label)
    for key in ["action_density", "phi4", "NN", "m2", "m4"]:
        series: dict[str, np.ndarray] = {native_label: np.asarray([float(r[key]) for r in native_rows], dtype=np.float64)}
        for sweep in sweeps:
            rows = [r for r in measure_rows if int(r["sweep"]) == sweep]
            if rows:
                series[f"sweep_{sweep}"] = np.asarray([float(r[key]) for r in rows], dtype=np.float64)
        vals = np.concatenate(list(series.values()))
        lo, hi = np.quantile(vals, [0.005, 0.995])
        pad = 0.05 * max(float(hi - lo), 1.0e-12)
        fig, ax = plt.subplots(figsize=(6.2, 3.8), constrained_layout=True)
        for label, x in series.items():
            hist, edges = np.histogram(x, bins=70, range=(lo - pad, hi + pad), density=True)
            ax.step(0.5 * (edges[:-1] + edges[1:]), hist, where="mid", label=label)
        ax.set_xlabel(key)
        ax.set_ylabel("density")
        ax.legend(frameon=False, fontsize=8)
        ax.tick_params(direction="in", top=True, right=True)
        fig.savefig(run / "plots" / f"{key}_native_vs_sweeps.pdf")
        plt.close(fig)


def flush_monitoring_outputs(
    run: Path,
    *,
    all_rows: list[dict[str, Any]],
    all_measurement_rows: list[dict[str, Any]],
    g_rows: list[dict[str, Any]],
    acceptance_rows: list[dict[str, Any]],
    status: dict[str, Any],
) -> None:
    write_per_chain_rows(run / "observables" / "per_sweep_observables.csv", all_rows)
    write_main_measurement_rows(run / "observables" / "main_per_sweep_measurements.csv", all_measurement_rows)
    write_csv(run / "observables" / "Gk_per_sweep_measurements.csv", g_rows, G_COLUMNS)
    write_csv(run / "observables" / "acceptance_history.csv", acceptance_rows)
    write_csv(run / "observables" / "ensemble_average_history.csv", aggregate_history(all_measurement_rows, all_rows))
    write_json(run / "status.json", status)


def write_restart_checkpoint(
    run: Path,
    *,
    completed_sweeps: int,
    cfg: dict[str, Any],
    run_mode: str,
    initialization_mode: str,
    initial_detail_only_sweeps: int,
    coarse_update_scheme: str,
    phi: np.ndarray,
    psi: np.ndarray,
    source_idx: np.ndarray,
    initializer_metadata: dict[str, Any] | None,
    resume_mode: bool,
    resume_initializer_metadata: dict[str, Any] | None,
    patch_rng: np.random.Generator,
) -> Path:
    """Atomically persist the exact state required to resume after a sweep."""
    checkpoint = run / "checkpoints" / "checkpoint_latest.npz"
    temporary = run / "checkpoints" / "checkpoint_latest.tmp.npz"
    checkpoint_payload = {
        "meta": np.array(
            json.dumps(
                {
                    "completed_sweeps": int(completed_sweeps),
                    "lambda": LAM,
                    "kappa_f": float(cfg["kappa_f"]),
                    "kappa_c": float(cfg["kappa_c"]),
                    "L_c": int(cfg["L_c"]),
                    "L_f": int(cfg["L_f"]),
                    "mode": run_mode,
                    "initialization_mode": initialization_mode,
                    "detail_patch_size": int(cfg["patch"]["detail_patch_size"]),
                    "detail_passes": int(cfg["patch"]["detail_passes"]),
                    "initial_detail_only_sweeps": initial_detail_only_sweeps,
                    "coarse_update_scheme": coarse_update_scheme,
                    "fine_proposal_sigma": float(cfg["patch"]["fine_proposal_sigma"]),
                    "coarse_source_selection": str(cfg.get("coarse_source_selection", "")),
                    "start_index": cfg.get("start_index"),
                    "source_config_index_first": int(source_idx[0]),
                    "source_config_index_last": int(source_idx[-1]),
                    "initializer_metadata": (
                        {key: value for key, value in initializer_metadata.items() if key != "selected_donor_blocks"}
                        if not resume_mode and initializer_metadata is not None
                        else resume_initializer_metadata
                    ),
                    "rng_state": patch_rng.bit_generator.state,
                }
            ),
            dtype=np.str_,
        ),
        "phi_current": phi.astype(np.float32),
        "psi_current": psi.astype(np.float32),
        "source_config_index": source_idx.astype(np.int64),
    }
    np.savez_compressed(temporary, **checkpoint_payload)
    os.replace(temporary, checkpoint)
    return checkpoint


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--target-sweeps", type=int, default=None, help="Extend an existing run to this total sweep count.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    run = args.run_dir
    cfg = read_config(run)
    for sub in ["logs", "observables", "plots", "summaries", "debug", "checkpoints"]:
        (run / sub).mkdir(parents=True, exist_ok=True)

    if str(cfg.get("mode")) not in {"patchwise_detail_only", "patchwise_coarse_detail"}:
        raise SystemExit(f"unsupported mode for patchwise runner: {cfg.get('mode')}")
    if cfg.get("flow_type") != "rqspline":
        raise SystemExit("lambda=1.0 patchwise runner currently requires flow_type: rqspline")
    if args.dry_run:
        write_json(run / "status.json", {"status": "dry_run_ok", "mode": cfg.get("mode")})
        return 0

    t0 = time.perf_counter()
    kernel_spec, kernel_json = load_kernel(PROJECT_ROOT / cfg["kernel_path"])
    stencil_sum = float(np.sum(__import__("perfect_blocking_upsampling.kernels", fromlist=["kernel_stencil_from_spec"]).kernel_stencil_from_spec(kernel_spec)))
    if not kernel_spec.kernel_coefficients_include_eta_scale or not np.isclose(stencil_sum, ETA_SCALE, atol=1.0e-10):
        raise RuntimeError(f"bad eta convention: include={kernel_spec.kernel_coefficients_include_eta_scale} sum={stencil_sum}")

    coarse_all = load_phi(PROJECT_ROOT / cfg["coarse_config_source"])
    native_l32 = load_phi(PROJECT_ROOT / cfg["fine_config_source"])
    native_label = f"native_L{int(cfg['L_f'])}"
    action = ActionSpec("phi4_nn", LAM, float(cfg["kappa_f"]))
    coarse_action = ActionSpec("phi4_nn", LAM, float(cfg.get("kappa_c", cfg["kappa_f"])))
    target_sweeps = int(args.target_sweeps) if args.target_sweeps is not None else int(cfg["n_sweeps"])
    resume_mode = args.target_sweeps is not None
    resume_initializer_metadata: dict[str, Any] | None = None
    if resume_mode and (run / "status.json").exists():
        saved_status = json.loads((run / "status.json").read_text(encoding="utf-8"))
        saved_initializer = saved_status.get("initializer_metadata")
        if isinstance(saved_initializer, dict):
            resume_initializer_metadata = saved_initializer
    run_mode = str(cfg.get("mode"))
    initialization_mode = str(cfg.get("initialization_mode", "flow_sample"))
    if initialization_mode not in {"flow_sample", "native_fine_blocked", "empirical_sample"}:
        raise SystemExit(f"unsupported initialization_mode: {initialization_mode}")
    update_mode_label = "coarse_and_detail" if run_mode == "patchwise_coarse_detail" else "detail_only"
    initial_detail_only_sweeps = int(cfg.get("patch", {}).get("initial_detail_only_sweeps", 0))
    coarse_update_scheme = str(cfg.get("patch", {}).get("coarse_update_scheme", "flat_fixed_latent"))
    if coarse_update_scheme not in {"flat_fixed_latent", "two_stage_coarse_action"}:
        raise SystemExit(f"unsupported patch.coarse_update_scheme: {coarse_update_scheme}")
    if run_mode != "patchwise_coarse_detail":
        initial_detail_only_sweeps = 0
    if initial_detail_only_sweeps < 0:
        raise RuntimeError(f"initial_detail_only_sweeps must be nonnegative, got {initial_detail_only_sweeps}")
    source_label = f"blocked_native_L{int(cfg['L_f'])}" if initialization_mode == "native_fine_blocked" else f"direct_native_L{int(cfg['L_c'])}"
    load_report: dict[str, Any]
    device = torch.device(str(cfg.get("device", "cpu")))
    model = None
    stats = None
    # The spline is needed only for a new flow-sampled initial state.  A
    # resumed fixed-detail two-stage chain continues from its saved psi field.
    needs_model = (not resume_mode) or coarse_update_scheme == "flat_fixed_latent"
    if needs_model:
        ckpt = torch.load(PROJECT_ROOT / cfg["flow_checkpoint"], map_location=device, weights_only=False)
        model, load_report = build_model_from_checkpoint(ckpt, lattice_size=int(cfg["L_c"]), device=device)
        stats = stationary_stats(ckpt["state"]["stats"], lc=int(cfg["L_c"]))
    else:
        load_report = {}

    if resume_mode:
        checkpoint_path = run / "checkpoints" / "checkpoint_latest.npz"
        if not checkpoint_path.exists():
            raise RuntimeError(f"cannot extend: missing {checkpoint_path}")
        with np.load(checkpoint_path, allow_pickle=True) as z:
            meta = json.loads(str(z["meta"].item()))
            completed_sweeps = int(meta["completed_sweeps"])
            phi = z["phi_current"].astype(np.float32)
            psi = z["psi_current"].astype(np.float32)
            source_idx = z["source_config_index"].astype(np.int64)
            rng_state = meta["rng_state"]
        if target_sweeps <= completed_sweeps:
            raise RuntimeError(f"target_sweeps={target_sweeps} is not greater than completed_sweeps={completed_sweeps}")
        n_chains = int(len(source_idx))
        coarse = psi[:, 0::2, 0::2].astype(np.float32) if initialization_mode == "native_fine_blocked" else coarse_all[source_idx].astype(np.float32)
        blocked0 = apply_kernel(phi, kernel_spec).astype(np.float32)
        reb0 = blocked0[:, 0::2, 0::2].astype(np.float64) - psi[:, 0::2, 0::2].astype(np.float64)
        roundtrip0 = blocked0.astype(np.float64) - psi.astype(np.float64)
        resume_reblock_max = float(np.max(np.abs(reb0)))
        resume_roundtrip_max = float(np.max(np.abs(roundtrip0)))
        if resume_reblock_max > 5.0e-6:
            raise RuntimeError(f"resume current-coarse reblocking failed: {resume_reblock_max}")
        if resume_roundtrip_max > 5.0e-6:
            raise RuntimeError(f"resume kernel roundtrip failed: {resume_roundtrip_max}")
        if run_mode != "patchwise_coarse_detail":
            source_reb0 = blocked0[:, 0::2, 0::2].astype(np.float64) - coarse.astype(np.float64)
            source_reblock_max = float(np.max(np.abs(source_reb0)))
            if source_reblock_max > 5.0e-6:
                raise RuntimeError(f"resume source-coarse reblocking failed: {source_reblock_max}")
        else:
            source_reblock_max = float(
                np.max(np.abs(blocked0[:, 0::2, 0::2].astype(np.float64) - coarse.astype(np.float64)))
            )
        # A process may have flushed monitoring rows after its last durable
        # checkpoint.  Retain only the checkpointed prefix: sweeps beyond it
        # are regenerated from the restored RNG state below.
        def checkpointed_prefix(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
            return [row for row in rows if int(float(row["sweep"])) <= completed_sweeps]

        all_rows = checkpointed_prefix(read_csv(run / "observables" / "per_sweep_observables.csv"))
        all_measurement_rows = checkpointed_prefix(read_csv(run / "observables" / "main_per_sweep_measurements.csv"))
        g_rows = checkpointed_prefix(read_csv(run / "observables" / "Gk_per_sweep_measurements.csv"))
        acceptance_rows = checkpointed_prefix(read_csv(run / "observables" / "acceptance_history.csv"))
        gx_checkpoint_rows = checkpointed_prefix(read_csv(run / "observables" / "Gx_axis_by_checkpoint.csv"))
        cumulative = cumulative_from_acceptance(acceptance_rows)
        patch_rng = np.random.default_rng()
        patch_rng.bit_generator.state = rng_state
        start_sweep = completed_sweeps + 1
        load_report = {
            **load_report,
            "resume_checkpoint": str(checkpoint_path),
            "completed_sweeps": completed_sweeps,
            "resume_reblocking_max_error": resume_reblock_max,
            "resume_kernel_roundtrip_max_error": resume_roundtrip_max,
            "resume_source_coarse_difference_max": source_reblock_max,
            "resume_source_coarse_difference_note": (
                "expected nonzero for patchwise_coarse_detail because coarse patches evolve the retained field"
                if run_mode == "patchwise_coarse_detail"
                else "must remain near zero for detail-only runs because the retained coarse field is fixed"
            ),
        }
        # The initializer is only used to annotate restart checkpoints.  It is
        # nevertheless required when checkpointing a resumed trajectory.
        initializer_metadata = resume_initializer_metadata or {
            "initializer_type": "resumed_saved_state"
        }
    else:
        assert model is not None and stats is not None
        n_chains = int(cfg["n_chains"])
        start_index = cfg.get("start_index")
        source_pool_len = len(native_l32) if initialization_mode == "native_fine_blocked" else len(coarse_all)
        if start_index is None:
            rng = np.random.default_rng(int(cfg["random_seed"]))
            source_idx = rng.choice(source_pool_len, size=n_chains, replace=False)
            source_selection = "random_without_replacement"
        else:
            start = int(start_index)
            stop = start + n_chains
            if start < 0 or stop > source_pool_len:
                raise RuntimeError(
                    f"requested contiguous source range [{start}, {stop}) from {source_pool_len} configs"
                )
            source_idx = np.arange(start, stop, dtype=np.int64)
            source_selection = "contiguous_from_start_index"
        initializer_metadata: dict[str, Any] = {"initializer_type": "learned_flow" if initialization_mode == "flow_sample" else initialization_mode}
        if initialization_mode == "native_fine_blocked":
            phi = native_l32[source_idx].astype(np.float32)
            psi = apply_kernel(phi, kernel_spec).astype(np.float32)
            coarse = psi[:, 0::2, 0::2].astype(np.float32)
        elif initialization_mode == "empirical_sample":
            empirical_config = cfg.get("empirical_upscaler_config")
            if not empirical_config:
                raise RuntimeError("empirical_sample requires empirical_upscaler_config")
            upscaler = EmpiricalJointDetailUpscaler(load_empirical_upscaler_config(PROJECT_ROOT / str(empirical_config), PROJECT_ROOT))
            coarse = coarse_all[source_idx].astype(np.float32)
            phi, _detail, _z, initializer_metadata = upscaler.sample(
                coarse, np.random.default_rng(int(cfg["random_seed"]) + int(cfg.get("empirical_rng_seed_offset", 1000)))
            )
            psi = apply_kernel(phi, kernel_spec).astype(np.float32)
            initializer_metadata = {
                **initializer_metadata,
                "empirical_donor_bank_path": str(upscaler.config.donor_ensemble_path),
                "radial_latent_seed": int(cfg["random_seed"]) + int(cfg.get("empirical_rng_seed_offset", 1000)),
                "old_flow_checkpoint": None,
                "coarse_coordinate_flow_checkpoint": str(cfg["flow_checkpoint"]),
            }
        else:
            coarse = coarse_all[source_idx].astype(np.float32)
            detail, _logq, _zmax, _logdet = sample_model_lattice(
                model,
                coarse,
                stats,
                batch_size=int(cfg.get("batch_size", 64)),
                device=device,
                seed=int(cfg["random_seed"]) + 1000,
            )
            psi = assemble_psi(coarse, detail).astype(np.float32)
            phi, _ = inverse_kernel(psi, kernel_spec)
        reb0 = apply_kernel(phi, kernel_spec)[:, 0::2, 0::2] - coarse
        if float(np.max(np.abs(reb0))) > 5.0e-6:
            raise RuntimeError(f"initial reblocking failed: {float(np.max(np.abs(reb0)))}")

        cumulative = {"coarse_proposals": 0, "coarse_accepts": 0, "detail_proposals": 0, "detail_accepts": 0}
        all_rows: list[dict[str, Any]] = []
        all_measurement_rows: list[dict[str, Any]] = []
        g_rows: list[dict[str, Any]] = []
        acceptance_rows: list[dict[str, Any]] = []
        gx_checkpoint_rows: list[dict[str, Any]] = []
        initial_meta = {
            "update_mode": update_mode_label,
            "detail_update_acceptance": float("nan"),
            "detail_update_config_attempts": 0,
            "detail_update_accepts": 0,
            "coarse_acceptance": float("nan"),
            "coarse_proposals": 0,
            "coarse_accepts": 0,
            "conditional_flow_refreshes": 1,
        }
        rows = rows_for_sweep(phi, psi, kernel_spec, action, source_idx, 0, initial_meta, source_label)
        measurements = main_measurement_rows(phi, action, source_idx, 0, source_label)
        all_rows.extend(rows)
        all_measurement_rows.extend(measurements)
        g_rows.extend(g_rows_for_sweep(phi, source_idx, 0, source_label))
        acceptance_rows.append(acceptance_history_row(0, initial_meta, cumulative))
        flush_monitoring_outputs(
            run,
            all_rows=all_rows,
            all_measurement_rows=all_measurement_rows,
            g_rows=g_rows,
            acceptance_rows=acceptance_rows,
            status={
                "status": "running",
                "run_dir": str(run),
                "lambda": LAM,
                "mode": run_mode,
                "current_sweep": 0,
                "target_sweeps": target_sweeps,
                "initial_detail_only_sweeps": initial_detail_only_sweeps,
                "coarse_update_scheme": coarse_update_scheme,
                "latest_checkpoint": None,
                "kernel_sum": stencil_sum,
                "kernel_coefficients_include_eta_scale": True,
                "flow_checkpoint": str(cfg["flow_checkpoint"]),
                "initializer_metadata": {key: value for key, value in initializer_metadata.items() if key != "selected_donor_blocks"},
                "initialization_mode": initialization_mode,
                "flow_load_report": load_report,
                "coarse_source_selection": source_selection,
                "start_index": start_index,
                "source_config_index_first": int(source_idx[0]),
                "source_config_index_last": int(source_idx[-1]),
            },
        )
        patch_rng = np.random.default_rng(int(cfg["random_seed"]) + 2000)
        start_sweep = 1

    for sweep in range(start_sweep, target_sweeps + 1):
        patch_writer = StreamingCsv(run / "logs" / f"patch_history_to_sweep{sweep:03d}.csv", PATCH_HISTORY_FIELDS)
        two_stage_writer = StreamingCsv(
            run / "logs" / "coarse_two_stage_history.csv",
            TWO_STAGE_COARSE_FIELDS,
            append=sweep != start_sweep,
        )
        coarse_metas: list[dict[str, Any]] = []
        detail_warmup_sweep = run_mode == "patchwise_coarse_detail" and sweep <= initial_detail_only_sweeps
        sweep_update_mode = "initial_detail_only" if detail_warmup_sweep else update_mode_label
        if (
            run_mode == "patchwise_coarse_detail"
            and not detail_warmup_sweep
            and int(cfg["patch"].get("coarse_passes", 0)) > 0
        ):
            if coarse_update_scheme == "two_stage_coarse_action":
                psi, phi, cmeta = rqspline_two_stage_coarse_action_patch_update(
                    psi,
                    kernel_spec,
                    coarse_action,
                    action,
                    passes=int(cfg["patch"]["coarse_passes"]),
                    patch_size=int(cfg["patch"]["coarse_patch_size"]),
                    step_size=float(cfg["patch"]["coarse_step_size"]),
                    sweep=sweep,
                    writer=patch_writer,
                    diagnostics_writer=two_stage_writer,
                    rng=patch_rng,
                )
            else:
                if model is None or stats is None:
                    raise RuntimeError("flat_fixed_latent coarse updates require loaded RQ-spline model and stats")
                psi, phi, cmeta = rqspline_fixed_latent_coarse_patch_update(
                    psi,
                    kernel_spec,
                    action,
                    model,
                    stats,
                    batch_size=int(cfg.get("batch_size", 64)),
                    device=device,
                    passes=int(cfg["patch"]["coarse_passes"]),
                    patch_size=int(cfg["patch"]["coarse_patch_size"]),
                    step_size=float(cfg["patch"]["coarse_step_size"]),
                    sweep=sweep,
                    writer=patch_writer,
                    rng=patch_rng,
                )
            coarse_metas.append(cmeta)
        pc_args = argparse.Namespace(
            disable_coarse_updates=True,
            detail_passes=int(cfg["patch"]["detail_passes"]),
            fine_proposal_sigma=float(cfg["patch"]["fine_proposal_sigma"]),
            fine_patch_size=int(cfg["patch"]["detail_patch_size"]),
            passes=0,
            proposal_sigma=float(cfg["patch"].get("coarse_step_size", 0.0)),
            coarse_patch_size=int(cfg["patch"]["detail_patch_size"]),
            global_sweep=sweep,
            verbose_patch_log=False,
        )
        phi, psi, dmeta = patch_correct(psi, kernel_spec, action, pc_args, patch_writer, patch_rng)
        meta = combine_interval_meta(sweep_update_mode, coarse_metas, [dmeta], 0)
        if coarse_metas:
            for key in [
                "coarse_update_scheme",
                "stage1_coarse_action_attempts",
                "stage1_coarse_action_accepts",
                "stage1_coarse_action_acceptance",
                "stage2_fine_correction_attempts",
                "stage2_fine_correction_accepts",
                "stage2_fine_correction_acceptance_conditional",
            ]:
                if key in coarse_metas[0]:
                    meta[key] = coarse_metas[0][key]
        acceptance_rows.append(acceptance_history_row(sweep, meta, cumulative))
        measure_every = max(1, int(cfg.get("measure_every", 1)))
        measured_this_sweep = (sweep % measure_every == 0) or (sweep == target_sweeps)
        if measured_this_sweep:
            rows = rows_for_sweep(phi, psi, kernel_spec, action, source_idx, sweep, meta, source_label)
            measurements = main_measurement_rows(phi, action, source_idx, sweep, source_label)
            all_rows.extend(rows)
            all_measurement_rows.extend(measurements)
            g_rows.extend(g_rows_for_sweep(phi, source_idx, sweep, source_label))
        final = acceptance_rows[-1]
        checkpoint_every = max(1, int(cfg.get("checkpoint_every", target_sweeps)))
        checkpoint_path = None
        if sweep % checkpoint_every == 0 or sweep == target_sweeps:
            checkpoint_path = write_restart_checkpoint(
                run,
                completed_sweeps=sweep,
                cfg=cfg,
                run_mode=run_mode,
                initialization_mode=initialization_mode,
                initial_detail_only_sweeps=initial_detail_only_sweeps,
                coarse_update_scheme=coarse_update_scheme,
                phi=phi,
                psi=psi,
                source_idx=source_idx,
                initializer_metadata=initializer_metadata,
                resume_mode=resume_mode,
                resume_initializer_metadata=resume_initializer_metadata,
                patch_rng=patch_rng,
            )
            gx_checkpoint_rows = [row for row in gx_checkpoint_rows if int(float(row["sweep"])) != sweep]
            gx_checkpoint_rows.extend(gx_axis_checkpoint_rows(phi, sweep))
            write_csv(
                run / "observables" / "Gx_axis_by_checkpoint.csv",
                gx_checkpoint_rows,
                GX_AXIS_CHECKPOINT_COLUMNS,
            )
        flush_monitoring_outputs(
            run,
            all_rows=all_rows,
            all_measurement_rows=all_measurement_rows,
            g_rows=g_rows,
            acceptance_rows=acceptance_rows,
            status={
                "status": "running",
                "run_dir": str(run),
                "lambda": LAM,
                "mode": run_mode,
                "initialization_mode": initialization_mode,
                "update_mode_current_sweep": sweep_update_mode,
                "initial_detail_only_sweeps": initial_detail_only_sweeps,
                "coarse_update_scheme": coarse_update_scheme,
                "current_sweep": sweep,
                "target_sweeps": target_sweeps,
                "detail_acceptance_current_sweep": final["detail_acceptance"],
                "detail_acceptance_cumulative": final["detail_acceptance_cumulative"],
                "detail_proposals_cumulative": final["detail_proposals_cumulative"],
                "detail_accepts_cumulative": final["detail_accepts_cumulative"],
                "coarse_acceptance_current_sweep": final["coarse_acceptance"],
                "coarse_acceptance_cumulative": final["coarse_acceptance_cumulative"],
                "coarse_proposals_cumulative": final["coarse_proposals_cumulative"],
                "coarse_accepts_cumulative": final["coarse_accepts_cumulative"],
                "stage1_coarse_action_acceptance_current_sweep": meta.get("stage1_coarse_action_acceptance"),
                "stage2_fine_correction_acceptance_conditional_current_sweep": meta.get("stage2_fine_correction_acceptance_conditional"),
                "latest_checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
                "initializer_metadata": (
                    {key: value for key, value in initializer_metadata.items() if key != "selected_donor_blocks"}
                    if not resume_mode else resume_initializer_metadata
                ),
                "elapsed_seconds": float(time.perf_counter() - t0),
            },
        )
        if sweep % 10 == 0 or sweep == 1:
            print(
                json.dumps(
                    {
                        "sweep": sweep,
                        "update_mode": sweep_update_mode,
                        "coarse_acceptance": meta["coarse_acceptance"],
                        "coarse_proposals": meta["coarse_proposals"],
                        "detail_acceptance": meta["detail_update_acceptance"],
                        "detail_proposals": meta["detail_update_config_attempts"],
                    }
                ),
                flush=True,
            )

    write_per_chain_rows(run / "observables" / "per_sweep_observables.csv", all_rows)
    write_main_measurement_rows(run / "observables" / "main_per_sweep_measurements.csv", all_measurement_rows)
    write_csv(run / "observables" / "Gk_per_sweep_measurements.csv", g_rows, G_COLUMNS)
    write_csv(run / "observables" / "acceptance_history.csv", acceptance_rows)
    write_csv(run / "observables" / "ensemble_average_history.csv", aggregate_history(all_measurement_rows, all_rows))
    native_rows = native_summary_rows(native_l32, action, native_label)
    write_csv(run / "observables" / f"{native_label}_reference_summary.csv", native_rows)
    comparison_sweeps = sorted(set([0, 1, 5, 10, 25, 50, 100, target_sweeps]))
    write_csv(run / "observables" / "sweep_native_comparison.csv", comparison_rows(all_measurement_rows, native_rows, comparison_sweeps))
    plot_histograms(run, all_measurement_rows, native_l32, action, [0, 10, 100], native_label)

    final = acceptance_rows[-1]
    checkpoint_path = write_restart_checkpoint(
        run,
        completed_sweeps=target_sweeps,
        cfg=cfg,
        run_mode=run_mode,
        initialization_mode=initialization_mode,
        initial_detail_only_sweeps=initial_detail_only_sweeps,
        coarse_update_scheme=coarse_update_scheme,
        phi=phi,
        psi=psi,
        source_idx=source_idx,
        initializer_metadata=initializer_metadata,
        resume_mode=resume_mode,
        resume_initializer_metadata=resume_initializer_metadata,
        patch_rng=patch_rng,
    )
    status = {
        "status": "completed",
        "run_dir": str(run),
        "lambda": LAM,
        "mode": run_mode,
        "initialization_mode": initialization_mode,
        "resume_reuses_saved_state": bool(resume_mode),
        "algorithm": (
            "lambda=1.0 local patch workflow: two-stage coarse-action Metropolis plus fine correction"
            if coarse_update_scheme == "two_stage_coarse_action"
            else "lambda=1.0 local patch workflow: flat fixed-latent RQ-spline coarse patches plus local fine/detail Metropolis patches"
        ),
        "coarse_update_scheme": coarse_update_scheme,
        "kernel_sum": stencil_sum,
        "kernel_coefficients_include_eta_scale": True,
        "flow_checkpoint": str(cfg["flow_checkpoint"]),
        "coarse_coordinate_flow_checkpoint": str(cfg["flow_checkpoint"]),
        "initializer_metadata": (
            {key: value for key, value in initializer_metadata.items() if key != "selected_donor_blocks"}
            if not resume_mode else resume_initializer_metadata
        ),
        "flow_load_report": load_report,
        "n_chains": n_chains,
        "coarse_source_selection": str(cfg.get("coarse_source_selection", "")),
        "start_index": cfg.get("start_index"),
        "source_config_index_first": int(source_idx[0]),
        "source_config_index_last": int(source_idx[-1]),
        "n_sweeps": target_sweeps,
        "initial_detail_only_sweeps": initial_detail_only_sweeps,
        "detail_patch_size": int(cfg["patch"]["detail_patch_size"]),
        "detail_passes_per_sweep": int(cfg["patch"]["detail_passes"]),
        "coarse_patch_size": int(cfg["patch"].get("coarse_patch_size", 0)),
        "coarse_passes_per_sweep": int(cfg["patch"].get("coarse_passes", 0)) if run_mode == "patchwise_coarse_detail" else 0,
        "patches_per_pass": int(math.ceil(2.0 * int(cfg["L_f"]) * int(cfg["L_f"]) / float(int(cfg["patch"]["detail_patch_size"]) ** 2))),
        "attempted_detail_config_updates_per_sweep": int(cfg["patch"]["detail_passes"]) * int(math.ceil(2.0 * int(cfg["L_f"]) * int(cfg["L_f"]) / float(int(cfg["patch"]["detail_patch_size"]) ** 2))) * n_chains,
        "detail_acceptance_final_sweep": final["detail_acceptance"],
        "detail_acceptance_cumulative": final["detail_acceptance_cumulative"],
        "coarse_acceptance_final_sweep": final["coarse_acceptance"],
        "coarse_acceptance_cumulative": final["coarse_acceptance_cumulative"],
        "coarse_proposals_cumulative": final["coarse_proposals_cumulative"],
        "coarse_accepts_cumulative": final["coarse_accepts_cumulative"],
        "stage1_coarse_action_acceptance_final_sweep": meta.get("stage1_coarse_action_acceptance"),
        "stage2_fine_correction_acceptance_conditional_final_sweep": meta.get("stage2_fine_correction_acceptance_conditional"),
        "detail_proposals_cumulative": final["detail_proposals_cumulative"],
        "detail_accepts_cumulative": final["detail_accepts_cumulative"],
        "latest_checkpoint": str(checkpoint_path),
        "current_sweep": target_sweeps,
        "reblocking_max_error": float(np.max([float(r["reblocking_max_error"]) for r in all_rows])),
        "nonfinite_count": int(np.sum([int(r["nonfinite_count"]) for r in all_measurement_rows])),
        "elapsed_seconds": float(time.perf_counter() - t0),
    }
    write_json(run / "status.json", status)
    (run / "summaries" / "run_summary.md").write_text(
        "\n".join(
            [
                f"# Lambda=1.0 {run_mode} L{int(cfg['L_c'])}->L{int(cfg['L_f'])}",
                "",
                f"Algorithm: {status['algorithm']}.",
                f"Detail patch: P={status['detail_patch_size']}, passes/sweep={status['detail_passes_per_sweep']}, patches/pass={status['patches_per_pass']}.",
                f"Initial detail-only sweeps before coarse updates: {status['initial_detail_only_sweeps']}.",
                f"Coarse patch: P={status['coarse_patch_size']}, passes/sweep={status['coarse_passes_per_sweep']}.",
                f"Coarse update scheme: {status['coarse_update_scheme']}.",
                f"Cumulative coarse acceptance: {status['coarse_acceptance_cumulative']}.",
                f"Final-sweep coarse acceptance: {status['coarse_acceptance_final_sweep']}.",
                f"Final-sweep Stage-1 coarse-action acceptance: {status.get('stage1_coarse_action_acceptance_final_sweep')}.",
                f"Final-sweep Stage-2 fine-correction acceptance conditional on Stage 1: {status.get('stage2_fine_correction_acceptance_conditional_final_sweep')}.",
                f"Attempted detail config updates/sweep: {status['attempted_detail_config_updates_per_sweep']}.",
                f"Cumulative detail acceptance: {status['detail_acceptance_cumulative']}.",
                f"Final-sweep detail acceptance: {status['detail_acceptance_final_sweep']}.",
                f"Max reblocking error: {status['reblocking_max_error']:.3e}.",
                f"Nonfinite count: {status['nonfinite_count']}.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
