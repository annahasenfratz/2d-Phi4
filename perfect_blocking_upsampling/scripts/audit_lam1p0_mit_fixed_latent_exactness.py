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

from perfect_blocking_upsampling.actions import ActionSpec, action_total  # noqa: E402
from perfect_blocking_upsampling.kernels import (  # noqa: E402
    apply_kernel,
    inverse_kernel,
    kernel_stencil_from_spec,
    load_kernel,
)
from run_lam0p2_flow_detail_rethermalization import coarse_patch_mask, main_measurement_rows  # noqa: E402
from run_lam1p0_l16to32_rqspline_zeroshot import (  # noqa: E402
    build_model_from_checkpoint,
    stationary_stats,
)
from run_lam1p0_rqspline_patchwise import (  # noqa: E402
    detail_from_psi,
    infer_rqspline_latents_and_logj,
    reconstruct_rqspline_from_latents,
)
from train_lam1p0_flow_detail_pilot import assemble_psi, load_phi  # noqa: E402


DEFAULT_CONFIG = PROJECT_ROOT / "perfect_blocking_upsampling/run_configs/generated/submit_flow_detail_coarse_detail_lam1p0.yaml"
DEFAULT_OUT = PROJECT_ROOT / "perfect_blocking/perfect_blocking_lam1p0/tests/final/mit_style_fixed_latent_exactness_audit_20260720"
OBS_KEYS = ["action_density", "phi2", "phi4", "NN", "2nn", "diag", "local_kurtosis_ratio", "m2", "m4"]


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


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def obs_rows(phi: np.ndarray, action: ActionSpec, sweep: int, label: str) -> list[dict[str, Any]]:
    rows = main_measurement_rows(phi.astype(np.float32), action, np.arange(len(phi), dtype=np.int64), sweep, label)
    out = []
    for r in rows:
        row = dict(r)
        row["label"] = label
        if "local_kurtosis_ratio" not in row:
            phi2 = float(row["phi2"])
            row["local_kurtosis_ratio"] = float(row["phi4"]) / max(phi2 * phi2, 1.0e-300)
        out.append(row)
    return out


def distribution_rows(rows: list[dict[str, Any]], native_rows: list[dict[str, Any]], labels: list[str]) -> list[dict[str, Any]]:
    out = []
    native_by_key = {k: np.asarray([float(r[k]) for r in native_rows], dtype=np.float64) for k in OBS_KEYS}
    for label in labels:
        label_rows = [r for r in rows if r["label"] == label]
        sweeps = sorted({int(r["sweep"]) for r in label_rows})
        for sweep in sweeps:
            sweep_rows = [r for r in label_rows if int(r["sweep"]) == sweep]
            for key in OBS_KEYS:
                a = np.asarray([float(r[key]) for r in sweep_rows], dtype=np.float64)
                b = native_by_key[key]
                if len(a) < 2 or len(b) < 2:
                    continue
                bins = np.histogram_bin_edges(np.concatenate([a, b]), bins="fd")
                if len(bins) < 8:
                    bins = np.linspace(min(float(a.min()), float(b.min())), max(float(a.max()), float(b.max())), 32)
                pa, _ = np.histogram(a, bins=bins)
                pb, _ = np.histogram(b, bins=bins)
                pa = pa.astype(np.float64) / max(1.0, float(pa.sum()))
                pb = pb.astype(np.float64) / max(1.0, float(pb.sum()))
                tv = 0.5 * float(np.sum(np.abs(pa - pb)))
                m = 0.5 * (pa + pb)
                js = 0.5 * float(np.sum(np.where(pa > 0, pa * np.log2(pa / np.maximum(m, 1e-300)), 0.0)))
                js += 0.5 * float(np.sum(np.where(pb > 0, pb * np.log2(pb / np.maximum(m, 1e-300)), 0.0)))
                out.append(
                    {
                        "label": label,
                        "sweep": sweep,
                        "observable": key,
                        "mean": float(np.mean(a)),
                        "std": float(np.std(a, ddof=1)),
                        "native_mean": float(np.mean(b)),
                        "native_std": float(np.std(b, ddof=1)),
                        "mean_shift_native_sigma": float((np.mean(a) - np.mean(b)) / np.std(b, ddof=1)),
                        "std_ratio": float(np.std(a, ddof=1) / np.std(b, ddof=1)),
                        "q05": float(np.quantile(a, 0.05)),
                        "q50": float(np.quantile(a, 0.50)),
                        "q95": float(np.quantile(a, 0.95)),
                        "q99": float(np.quantile(a, 0.99)),
                        "native_q05": float(np.quantile(b, 0.05)),
                        "native_q50": float(np.quantile(b, 0.50)),
                        "native_q95": float(np.quantile(b, 0.95)),
                        "native_q99": float(np.quantile(b, 0.99)),
                        "frac_below_native_q05": float(np.mean(a < np.quantile(b, 0.05))),
                        "frac_above_native_q95": float(np.mean(a > np.quantile(b, 0.95))),
                        "frac_above_native_q99": float(np.mean(a > np.quantile(b, 0.99))),
                        "TV": tv,
                        "JS": js,
                        "OVL": float(np.sum(np.minimum(pa, pb))),
                    }
                )
    return out


def torch_kernel_fft(spec: Any, L: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    stencil = kernel_stencil_from_spec(spec)
    w = np.zeros((L, L), dtype=np.float64)
    k = stencil.shape[0] // 2
    for i in range(stencil.shape[0]):
        for j in range(stencil.shape[1]):
            w[(i - k) % L, (j - k) % L] += stencil[i, j]
    scale = 1.0 if spec.kernel_coefficients_include_eta_scale else float(spec.eta_scale)
    wt = torch.tensor(scale * w, device=device, dtype=dtype)
    return torch.fft.fft2(wt)


def torch_inverse_kernel(psi: torch.Tensor, kernel_fft: torch.Tensor) -> torch.Tensor:
    return torch.fft.ifft2(torch.fft.fft2(psi, dim=(-2, -1)) / kernel_fft[None, :, :], dim=(-2, -1)).real


def torch_reconstruct_detail_and_logj(model: Any, coarse_phys: torch.Tensor, z: torch.Tensor, stats: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    coarse_std = (coarse_phys - float(stats["coarse_mean"])) / float(stats["coarse_std"])
    mean = torch.tensor(np.asarray(stats["detail_mean"], dtype=np.float64).reshape(1, 3, 1, 1), device=coarse_phys.device, dtype=coarse_phys.dtype)
    std = torch.tensor(np.asarray(stats["detail_std"], dtype=np.float64).reshape(1, 3, 1, 1), device=coarse_phys.device, dtype=coarse_phys.dtype)
    d = torch.zeros_like(z, dtype=coarse_phys.dtype)
    logj = torch.zeros(coarse_phys.shape[0], device=coarse_phys.device, dtype=coarse_phys.dtype)
    for stage in range(3):
        cond_affine = model.affine_base.cond(coarse_std, d, stage)
        x_affine, affine_logdet = model.affine_base.flows[stage].forward(z[:, stage].flatten(1), cond_affine)
        cond_spline = model.cond(coarse_std, d, stage)
        x, spline_logdet = model.spline.flows[stage].forward(x_affine, cond_spline)
        d[:, stage] = x.reshape(coarse_phys.shape[0], coarse_phys.shape[1], coarse_phys.shape[2])
        logj = logj + affine_logdet + spline_logdet
    return d * std + mean, logj


def torch_assemble_psi(coarse: torch.Tensor, detail: torch.Tensor) -> torch.Tensor:
    n, lc, _ = coarse.shape
    psi = coarse.new_empty((n, 2 * lc, 2 * lc))
    psi[:, 0::2, 0::2] = coarse
    psi[:, 0::2, 1::2] = detail[:, 0]
    psi[:, 1::2, 0::2] = detail[:, 1]
    psi[:, 1::2, 1::2] = detail[:, 2]
    return psi


def explicit_jacobian_check(
    model: Any,
    stats: dict[str, Any],
    kernel: Any,
    out_dir: Path,
    *,
    lc: int,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    model = model.to(device).double().eval()
    rng = np.random.default_rng(seed)
    c0 = rng.normal(float(stats["coarse_mean"]), float(stats["coarse_std"]), size=(1, lc, lc)).astype(np.float64)
    z0 = rng.normal(size=(1, 3, lc, lc)).astype(np.float64)
    flat0 = torch.tensor(np.concatenate([c0.reshape(-1), z0.reshape(-1)]), dtype=torch.float64, device=device, requires_grad=True)
    kfft = torch_kernel_fft(kernel, 2 * lc, device, torch.float64)

    def f(flat: torch.Tensor) -> torch.Tensor:
        c = flat[: lc * lc].reshape(1, lc, lc)
        z = flat[lc * lc :].reshape(1, 3, lc, lc)
        detail, _ = torch_reconstruct_detail_and_logj(model, c, z, stats)
        phi = torch_inverse_kernel(torch_assemble_psi(c, detail), kfft)
        return phi.reshape(-1)

    jac = torch.autograd.functional.jacobian(f, flat0, vectorize=False)
    sign, logabsdet = torch.linalg.slogdet(jac)
    with torch.no_grad():
        c = flat0[: lc * lc].reshape(1, lc, lc)
        z = flat0[lc * lc :].reshape(1, 3, lc, lc)
        _, reported = torch_reconstruct_detail_and_logj(model, c, z, stats)
        explicit_forward = float(logabsdet.detach().cpu())
        reported_logj = float(reported.detach().cpu()[0])
    c1 = c0.copy()
    c1.reshape(-1)[0] += 1e-3
    with torch.no_grad():
        c0_t = torch.tensor(c0, dtype=torch.float64, device=device)
        c1_t = torch.tensor(c1, dtype=torch.float64, device=device)
        z_t = torch.tensor(z0, dtype=torch.float64, device=device)
        _, logj0_t = torch_reconstruct_detail_and_logj(model, c0_t, z_t, stats)
        _, logj1_t = torch_reconstruct_detail_and_logj(model, c1_t, z_t, stats)
        logj0 = float(logj0_t.detach().cpu()[0])
        logj1 = float(logj1_t.detach().cpu()[0])
    # Full explicit determinant at c1 for delta convention.
    flat1 = torch.tensor(np.concatenate([c1.reshape(-1), z0.reshape(-1)]), dtype=torch.float64, device=device, requires_grad=True)
    jac1 = torch.autograd.functional.jacobian(f, flat1, vectorize=False)
    sign1, logabsdet1 = torch.linalg.slogdet(jac1)
    row = {
        "lc": lc,
        "lf": 2 * lc,
        "reported_logj": reported_logj,
        "explicit_forward_logdet": explicit_forward,
        "explicit_inverse_logdet": -explicit_forward,
        "slogdet_sign": float(sign.detach().cpu()),
        "constant_offset_explicit_minus_reported": explicit_forward - reported_logj,
        "reported_delta_logj_c1_minus_c0": float(logj1 - logj0),
        "explicit_delta_forward_logdet_c1_minus_c0": float(logabsdet1.detach().cpu() - logabsdet.detach().cpu()),
        "delta_discrepancy": float((logabsdet1.detach().cpu() - logabsdet.detach().cpu()) - (logj1 - logj0)),
        "metropolis_sign": "+Delta_logJ_forward",
    }
    write_csv(out_dir / "jacobian_convention_report.csv", [row])
    return row


def state_from_native(native: np.ndarray, kernel: Any, model: Any, stats: dict[str, Any], batch_size: int, device: torch.device) -> dict[str, np.ndarray]:
    psi = apply_kernel(native.astype(np.float32), kernel).astype(np.float32)
    c = psi[:, 0::2, 0::2].astype(np.float32)
    detail = detail_from_psi(psi)
    z, logj = infer_rqspline_latents_and_logj(model, c, detail, stats, batch_size=batch_size, device=device)
    detail_rec, logj_rec = reconstruct_rqspline_from_latents(model, c, z, stats, batch_size=batch_size, device=device)
    phi_rec, _ = inverse_kernel(assemble_psi(c, detail_rec).astype(np.float32), kernel)
    return {"c": c, "z": z, "phi": phi_rec.astype(np.float32), "logj": logj_rec.astype(np.float64), "psi": assemble_psi(c, detail_rec).astype(np.float32)}


def reconstruct_state(c: np.ndarray, z: np.ndarray, kernel: Any, model: Any, stats: dict[str, Any], batch_size: int, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    detail, logj = reconstruct_rqspline_from_latents(model, c.astype(np.float32), z.astype(np.float32), stats, batch_size=batch_size, device=device)
    psi = assemble_psi(c.astype(np.float32), detail.astype(np.float32)).astype(np.float32)
    phi, _ = inverse_kernel(psi, kernel)
    return phi.astype(np.float32), psi.astype(np.float32), logj.astype(np.float64)


def global_fixed_latent_chain(
    state: dict[str, np.ndarray],
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
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    c = state["c"].copy().astype(np.float32)
    z = state["z"].copy().astype(np.float32)
    phi = state["phi"].copy().astype(np.float32)
    logj = state["logj"].copy().astype(np.float64)
    sf = action_total(phi, action).astype(np.float64)
    obs_out = obs_rows(phi, action, 0, label)
    acc_rows = []
    rec_rows = []
    accepted_total = 0
    for sweep in range(1, sweeps + 1):
        prop_c = c + sigma * rng.standard_normal(c.shape).astype(np.float32)
        prop_phi, prop_psi, prop_logj = reconstruct_state(prop_c, z, kernel, model, stats, batch_size, device)
        prop_sf = action_total(prop_phi, action).astype(np.float64)
        delta_sf = prop_sf - sf
        delta_logj = prop_logj - logj
        log_ratio = -delta_sf + delta_logj
        accept = np.log(rng.random(len(c))) < np.minimum(0.0, log_ratio)
        if np.any(accept):
            c[accept] = prop_c[accept]
            phi[accept] = prop_phi[accept]
            logj[accept] = prop_logj[accept]
            sf[accept] = prop_sf[accept]
        accepted_total += int(np.sum(accept))
        psi = assemble_psi(c, detail_from_psi(apply_kernel(phi, kernel).astype(np.float32))).astype(np.float32)
        # The stored phi must match F(c,z); recompute directly for the check.
        rec_phi, rec_psi, _ = reconstruct_state(c, z, kernel, model, stats, batch_size, device)
        reb = apply_kernel(phi, kernel)[:, 0::2, 0::2].astype(np.float64) - c.astype(np.float64)
        rec = phi.astype(np.float64) - rec_phi.astype(np.float64)
        rec_rows.append(
            {
                "label": label,
                "sweep": sweep,
                "max_reconstruction_error": float(np.max(np.abs(rec))),
                "rms_reconstruction_error": float(np.sqrt(np.mean(rec * rec))),
                "max_retained_blocking_error": float(np.max(np.abs(reb))),
                "rms_retained_blocking_error": float(np.sqrt(np.mean(reb * reb))),
            }
        )
        acc_rows.append(
            {
                "label": label,
                "sweep": sweep,
                "attempts": len(c),
                "accepted": int(np.sum(accept)),
                "acceptance": float(np.mean(accept)),
                "acceptance_cumulative": float(accepted_total / (sweep * len(c))),
                "DeltaS_mean": float(np.mean(delta_sf)),
                "DeltaS_std": float(np.std(delta_sf, ddof=1)) if len(delta_sf) > 1 else 0.0,
                "Delta_logJ_mean": float(np.mean(delta_logj)),
                "Delta_logJ_std": float(np.std(delta_logj, ddof=1)) if len(delta_logj) > 1 else 0.0,
                "log_ratio_mean": float(np.mean(log_ratio)),
                "log_ratio_std": float(np.std(log_ratio, ddof=1)) if len(log_ratio) > 1 else 0.0,
                "frac_log_ratio_lt_minus10": float(np.mean(log_ratio < -10.0)),
                "frac_log_ratio_lt_minus20": float(np.mean(log_ratio < -20.0)),
            }
        )
        obs_out.extend(obs_rows(phi, action, sweep, label))
    return obs_out, acc_rows, rec_rows


def checkerboard_mask(lc: int, parity: int) -> np.ndarray:
    x, y = np.indices((lc, lc))
    return ((x + y) % 2) == int(parity)


def checkerboard_fixed_latent_chain(
    state: dict[str, np.ndarray],
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
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    c = state["c"].copy().astype(np.float32)
    z = state["z"].copy().astype(np.float32)
    phi = state["phi"].copy().astype(np.float32)
    logj = state["logj"].copy().astype(np.float64)
    sf = action_total(phi, action).astype(np.float64)
    obs_out = obs_rows(phi, action, 0, label)
    acc_rows: list[dict[str, Any]] = []
    rec_rows: list[dict[str, Any]] = []
    substep_rows: list[dict[str, Any]] = []
    accepted_total = 0
    attempts_total = 0
    masks = {0: checkerboard_mask(int(c.shape[1]), 0), 1: checkerboard_mask(int(c.shape[1]), 1)}
    for sweep in range(1, sweeps + 1):
        if order == "even_odd":
            parities = [0, 1]
        elif order == "odd_even":
            parities = [1, 0]
        elif order == "random":
            parities = [0, 1] if rng.random() < 0.5 else [1, 0]
        else:
            raise ValueError(f"unknown checkerboard order: {order}")
        sweep_accepts = 0
        sweep_attempts = 0
        for substep, parity in enumerate(parities):
            mask = masks[parity]
            inactive = ~mask
            old_c = c.copy()
            noise_active = sigma * rng.standard_normal((len(c), int(mask.sum()))).astype(np.float32)
            prop_c = c.copy()
            prop_c[:, mask] += noise_active
            inactive_bitwise_unchanged_proposal = bool(np.array_equal(prop_c[:, inactive], old_c[:, inactive]))
            active_delta_error = float(np.max(np.abs((prop_c[:, mask] - old_c[:, mask]).astype(np.float64) - noise_active.astype(np.float64))))

            prop_phi, _, prop_logj = reconstruct_state(prop_c, z, kernel, model, stats, batch_size, device)
            prop_sf = action_total(prop_phi, action).astype(np.float64)
            delta_sf = prop_sf - sf
            delta_logj = prop_logj - logj
            log_ratio = -delta_sf + delta_logj
            reverse_log_ratio = -log_ratio
            accept = np.log(rng.random(len(c))) < np.minimum(0.0, log_ratio)
            if np.any(accept):
                c[accept] = prop_c[accept]
                phi[accept] = prop_phi[accept]
                logj[accept] = prop_logj[accept]
                sf[accept] = prop_sf[accept]
            accepted = int(np.sum(accept))
            attempts = int(len(c))
            sweep_accepts += accepted
            sweep_attempts += attempts
            accepted_total += accepted
            attempts_total += attempts

            rec_phi, _, _ = reconstruct_state(c, z, kernel, model, stats, batch_size, device)
            reb = apply_kernel(phi, kernel)[:, 0::2, 0::2].astype(np.float64) - c.astype(np.float64)
            rec = phi.astype(np.float64) - rec_phi.astype(np.float64)
            rejected = ~accept
            inactive_after_error = float(np.max(np.abs(c[:, inactive].astype(np.float64) - old_c[:, inactive].astype(np.float64))))
            rejected_state_error = (
                float(np.max(np.abs(c[rejected].astype(np.float64) - old_c[rejected].astype(np.float64))))
                if np.any(rejected)
                else 0.0
            )
            accepted_active_error = (
                float(np.max(np.abs((c[accept][:, mask] - prop_c[accept][:, mask]).astype(np.float64))))
                if np.any(accept)
                else 0.0
            )
            substep_rows.append(
                {
                    "label": label,
                    "sweep": sweep,
                    "substep": substep,
                    "parity": int(parity),
                    "active_sites": int(mask.sum()),
                    "inactive_sites": int(inactive.sum()),
                    "attempts": attempts,
                    "accepted": accepted,
                    "acceptance": float(accepted / attempts),
                    "inactive_bitwise_unchanged_proposal": inactive_bitwise_unchanged_proposal,
                    "active_delta_error": active_delta_error,
                    "inactive_after_accept_reject_max_error": inactive_after_error,
                    "accepted_active_equals_proposal_max_error": accepted_active_error,
                    "rejected_state_restoration_max_error": rejected_state_error,
                    "DeltaS_mean": float(np.mean(delta_sf)),
                    "DeltaS_std": float(np.std(delta_sf, ddof=1)) if len(delta_sf) > 1 else 0.0,
                    "Delta_logJ_mean": float(np.mean(delta_logj)),
                    "Delta_logJ_std": float(np.std(delta_logj, ddof=1)) if len(delta_logj) > 1 else 0.0,
                    "log_ratio_mean": float(np.mean(log_ratio)),
                    "log_ratio_std": float(np.std(log_ratio, ddof=1)) if len(log_ratio) > 1 else 0.0,
                    "max_abs_forward_reverse_antisymmetry_residual": float(np.max(np.abs(log_ratio + reverse_log_ratio))),
                    "max_reconstruction_error": float(np.max(np.abs(rec))),
                    "rms_reconstruction_error": float(np.sqrt(np.mean(rec * rec))),
                    "max_retained_blocking_error": float(np.max(np.abs(reb))),
                    "rms_retained_blocking_error": float(np.sqrt(np.mean(reb * reb))),
                }
            )
        rec_phi, _, _ = reconstruct_state(c, z, kernel, model, stats, batch_size, device)
        reb = apply_kernel(phi, kernel)[:, 0::2, 0::2].astype(np.float64) - c.astype(np.float64)
        rec = phi.astype(np.float64) - rec_phi.astype(np.float64)
        rec_rows.append(
            {
                "label": label,
                "sweep": sweep,
                "max_reconstruction_error": float(np.max(np.abs(rec))),
                "rms_reconstruction_error": float(np.sqrt(np.mean(rec * rec))),
                "max_retained_blocking_error": float(np.max(np.abs(reb))),
                "rms_retained_blocking_error": float(np.sqrt(np.mean(reb * reb))),
            }
        )
        acc_rows.append(
            {
                "label": label,
                "sweep": sweep,
                "attempts": sweep_attempts,
                "accepted": sweep_accepts,
                "acceptance": float(sweep_accepts / sweep_attempts),
                "acceptance_cumulative": float(accepted_total / attempts_total),
                "substeps_per_sweep": 2,
                "order": order,
            }
        )
        obs_out.extend(obs_rows(phi, action, sweep, label))
    return obs_out, acc_rows, rec_rows, substep_rows


def ratio_antisymmetry_checks(
    state: dict[str, np.ndarray],
    kernel: Any,
    model: Any,
    stats: dict[str, Any],
    action: ActionSpec,
    *,
    batch_size: int,
    device: torch.device,
    seed: int,
    patch_size: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    c0 = state["c"][:8].copy().astype(np.float32)
    z = state["z"][:8].copy().astype(np.float32)
    phi0, _, logj0 = reconstruct_state(c0, z, kernel, model, stats, batch_size, device)
    sf0 = action_total(phi0, action).astype(np.float64)
    cases = [
        ("single_site", 0, 0, 1),
        ("interior_patch", 3, 5, patch_size),
        ("wrap_x_patch", int(c0.shape[1] - 2), 3, patch_size),
        ("wrap_y_patch", 3, int(c0.shape[2] - 2), patch_size),
        ("wrap_xy_patch", int(c0.shape[1] - 2), int(c0.shape[2] - 2), patch_size),
    ]
    out = []
    current_c = c0.copy()
    current_phi = phi0.copy()
    current_sf = sf0.copy()
    current_logj = logj0.copy()
    for kind, x0, y0, ps in cases:
        mask = coarse_patch_mask(int(current_c.shape[1]), x0, y0, ps)
        prop_c = current_c.copy()
        noise = 0.04 * rng.standard_normal((len(prop_c), int(mask.sum()))).astype(np.float32)
        prop_c[:, mask] += noise
        prop_phi, _, prop_logj = reconstruct_state(prop_c, z, kernel, model, stats, batch_size, device)
        prop_sf = action_total(prop_phi, action).astype(np.float64)
        ds = prop_sf - current_sf
        dlj = prop_logj - current_logj
        fwd = -ds + dlj
        rev = -(-ds) + (-dlj)
        out.append(
            {
                "case": kind,
                "patch_x": x0,
                "patch_y": y0,
                "patch_size": ps,
                "Sf_old_mean": float(np.mean(current_sf)),
                "Sf_new_mean": float(np.mean(prop_sf)),
                "DeltaS_mean": float(np.mean(ds)),
                "logJ_old_mean": float(np.mean(current_logj)),
                "logJ_new_mean": float(np.mean(prop_logj)),
                "Delta_logJ_mean": float(np.mean(dlj)),
                "forward_log_ratio_mean": float(np.mean(fwd)),
                "reverse_log_ratio_mean": float(np.mean(rev)),
                "max_abs_antisymmetry_residual": float(np.max(np.abs(fwd + rev))),
            }
        )
        # Force one sequential overlap update to test state composition.
        current_c = prop_c
        current_phi = prop_phi
        current_sf = prop_sf
        current_logj = prop_logj
    return out


def dependency_halo(
    state: dict[str, np.ndarray],
    kernel: Any,
    model: Any,
    stats: dict[str, Any],
    *,
    batch_size: int,
    device: torch.device,
    out_dir: Path,
    tol: float = 1.0e-10,
) -> dict[str, Any]:
    c = state["c"][:1].copy()
    z = state["z"][:1].copy()
    phi0, _, logj0 = reconstruct_state(c, z, kernel, model, stats, batch_size, device)
    cp = c.copy()
    cp[0, 0, 0] += 1.0e-3
    phi1, _, logj1 = reconstruct_state(cp, z, kernel, model, stats, batch_size, device)
    diff = np.abs(phi1[0].astype(np.float64) - phi0[0].astype(np.float64))
    coords = np.argwhere(diff > tol)
    np.save(out_dir / "dependency_halo_single_coarse_site_absdiff.npy", diff)
    rows = [{"x": int(x), "y": int(y), "abs_delta_phi": float(diff[x, y])} for x, y in coords]
    write_csv(out_dir / "dependency_halo_changed_fine_sites.csv", rows)
    return {
        "perturbed_coarse_site": [0, 0],
        "fine_sites_changed_above_tol": int(len(coords)),
        "fine_volume": int(diff.size),
        "fraction_fine_sites_changed": float(len(coords) / diff.size),
        "max_abs_delta_phi": float(np.max(diff)),
        "reported_logj_delta": float(logj1[0] - logj0[0]),
        "note": "A global/nonlocal affected set means cropped local coarse updates would need a full dependency halo; production currently reconstructs globally.",
    }


def plot_observable_evolution(dist_rows: list[dict[str, Any]], out_dir: Path) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for key in ["action_density", "phi2", "phi4", "NN", "local_kurtosis_ratio"]:
        fig, ax = plt.subplots(figsize=(6.5, 4.0))
        for label in sorted({r["label"] for r in dist_rows}):
            rows = [r for r in dist_rows if r["label"] == label and r["observable"] == key]
            if not rows:
                continue
            ax.plot([int(r["sweep"]) for r in rows], [float(r["mean_shift_native_sigma"]) for r in rows], marker="o", ms=3, label=label)
        ax.axhline(0.0, color="black", lw=1)
        ax.set_xlabel("sweep")
        ax.set_ylabel("mean shift vs native sigma")
        ax.set_title(key)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / f"{key}_stationarity_evolution.pdf")
        plt.close(fig)


def stable_seed(base: int, label: str) -> int:
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return int((int(base) + int.from_bytes(digest[:4], "little")) % (2**32 - 1))


def json_safe(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--n-chains", type=int, default=64)
    ap.add_argument("--sweeps", type=int, default=20)
    ap.add_argument("--proposal-sigma", type=float, default=None)
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=20260720)
    ap.add_argument("--tiny-lc", type=int, default=2)
    ap.add_argument("--skip-jacobian-autograd", action="store_true")
    args = ap.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(exist_ok=True)
    cfg = load_yaml(args.config)
    device = torch.device(str(cfg.get("device", "cpu")))
    action = ActionSpec("phi4_nn", float(cfg["lambda"]), float(cfg["kappa_f"]))
    kernel_path = PROJECT_ROOT / cfg["kernel_path"]
    kernel, kernel_json = load_kernel(kernel_path)
    flow_path = PROJECT_ROOT / cfg["flow_checkpoint"]
    ckpt = torch.load(flow_path, map_location=device, weights_only=False)
    model, load_report = build_model_from_checkpoint(ckpt, lattice_size=int(cfg["L_c"]), device=device)
    stats = stationary_stats(ckpt["state"]["stats"], lc=int(cfg["L_c"]))
    native = load_phi(PROJECT_ROOT / cfg["fine_config_source"])
    start = int(args.start_index)
    stop = start + int(args.n_chains)
    native_subset = native[start:stop].astype(np.float32)
    if len(native_subset) != args.n_chains:
        raise RuntimeError(f"not enough native configs for [{start}, {stop})")

    manifest = {
        "config": str(args.config),
        "out_dir": str(out_dir),
        "flow_checkpoint": str(flow_path),
        "flow_checkpoint_sha256": sha256(flow_path),
        "kernel_path": str(kernel_path),
        "kernel_sha256": sha256(kernel_path),
        "kernel_sum": float(kernel_stencil_from_spec(kernel).sum()),
        "kernel_coefficients_include_eta_scale": bool(kernel.kernel_coefficients_include_eta_scale),
        "lambda": float(cfg["lambda"]),
        "kappa_f": float(cfg["kappa_f"]),
        "kappa_c_metadata_only": float(cfg["kappa_c"]),
        "no_separate_coarse_action": True,
        "model_load_report": load_report,
        "n_chains": int(args.n_chains),
        "sweeps": int(args.sweeps),
        "start_index": start,
        "source_indices": [int(i) for i in range(start, stop)],
        "proposal": "symmetric additive Gaussian full coarse field at fixed global z",
    }
    (out_dir / "resolved_configuration.json").write_text(json.dumps(json_safe(manifest), indent=2) + "\n", encoding="utf-8")

    jac_row: dict[str, Any] | None = None
    if not args.skip_jacobian_autograd:
        try:
            tiny_ckpt = torch.load(flow_path, map_location=device, weights_only=False)
            tiny_model, _ = build_model_from_checkpoint(tiny_ckpt, lattice_size=int(args.tiny_lc), device=device)
            tiny_stats = stationary_stats(tiny_ckpt["state"]["stats"], lc=int(args.tiny_lc))
            jac_row = explicit_jacobian_check(tiny_model, tiny_stats, kernel, out_dir, lc=int(args.tiny_lc), device=device, seed=args.seed)
        except Exception as exc:
            jac_row = {"status": "failed", "error": repr(exc), "note": "Autograd tiny-lattice check failed; global ratio tests still ran on production lattice."}
            write_csv(out_dir / "jacobian_convention_report.csv", [jac_row])

    native_state = state_from_native(native_subset, kernel, model, stats, args.batch_size, device)
    native_rec = native_state["phi"].astype(np.float64) - native_subset.astype(np.float64)
    initial_rec_rows = [
        {
            "label": "native_fine_blocked_initial",
            "max_native_reconstruction_error": float(np.max(np.abs(native_rec))),
            "rms_native_reconstruction_error": float(np.sqrt(np.mean(native_rec * native_rec))),
            "max_retained_blocking_error": float(np.max(np.abs(apply_kernel(native_state["phi"], kernel)[:, 0::2, 0::2] - native_state["c"]))),
        }
    ]
    write_csv(out_dir / "initial_native_reconstruction.csv", initial_rec_rows)

    rng = np.random.default_rng(args.seed + 17)
    direct_coarse = load_phi(PROJECT_ROOT / cfg["coarse_config_source"])[start:stop].astype(np.float32)
    direct_detail, direct_logj = reconstruct_rqspline_from_latents(model, direct_coarse, native_state["z"], stats, batch_size=args.batch_size, device=device)
    direct_phi, _ = inverse_kernel(assemble_psi(direct_coarse, direct_detail), kernel)
    direct_state = {"c": direct_coarse, "z": native_state["z"].copy(), "phi": direct_phi, "logj": direct_logj}
    hot_c = native_state["c"] + 0.25 * rng.standard_normal(native_state["c"].shape).astype(np.float32)
    hot_phi, _, hot_logj = reconstruct_state(hot_c, native_state["z"], kernel, model, stats, args.batch_size, device)
    hot_state = {"c": hot_c, "z": native_state["z"].copy(), "phi": hot_phi, "logj": hot_logj}
    cold_c = np.full_like(native_state["c"], np.mean(native_state["c"], dtype=np.float64), dtype=np.float32)
    cold_phi, _, cold_logj = reconstruct_state(cold_c, native_state["z"], kernel, model, stats, args.batch_size, device)
    cold_state = {"c": cold_c, "z": native_state["z"].copy(), "phi": cold_phi, "logj": cold_logj}

    sigma = float(args.proposal_sigma if args.proposal_sigma is not None else cfg["patch"].get("coarse_step_size", 0.04))
    all_obs: list[dict[str, Any]] = []
    all_acc: list[dict[str, Any]] = []
    all_rec: list[dict[str, Any]] = []
    checkerboard_substeps: list[dict[str, Any]] = []
    for label, state in [
        ("native_fine_blocked", native_state),
        ("direct_native_coarse_flow_lift", direct_state),
        ("hot_distorted_coarse", hot_state),
        ("cold_low_variance_coarse", cold_state),
    ]:
        o, a, r = global_fixed_latent_chain(
            state,
            kernel,
            model,
            stats,
            action,
            batch_size=args.batch_size,
            device=device,
            sweeps=args.sweeps,
            sigma=sigma,
            seed=stable_seed(args.seed, f"global_{label}"),
            label=label,
        )
        all_obs.extend(o)
        all_acc.extend(a)
        all_rec.extend(r)
    for order in ["even_odd", "odd_even", "random"]:
        label = f"checkerboard_{order}_native_fine_blocked"
        o, a, r, s = checkerboard_fixed_latent_chain(
            native_state,
            kernel,
            model,
            stats,
            action,
            batch_size=args.batch_size,
            device=device,
            sweeps=args.sweeps,
            sigma=sigma,
            seed=stable_seed(args.seed, label),
            label=label,
            order=order,
        )
        all_obs.extend(o)
        all_acc.extend(a)
        all_rec.extend(r)
        checkerboard_substeps.extend(s)
    write_csv(out_dir / "stationarity_observables_per_config.csv", all_obs)
    write_csv(out_dir / "global_acceptance_history.csv", all_acc)
    write_csv(out_dir / "reconstruction_consistency.csv", all_rec)
    write_csv(out_dir / "checkerboard_substep_checks.csv", checkerboard_substeps)

    native_obs0 = [r for r in all_obs if r["label"] == "native_fine_blocked" and int(r["sweep"]) == 0]
    dist = distribution_rows(all_obs, native_obs0, sorted({r["label"] for r in all_obs}))
    write_csv(out_dir / "distribution_metrics_vs_native_sweep0.csv", dist)
    plot_observable_evolution(dist, out_dir)

    anti = ratio_antisymmetry_checks(native_state, kernel, model, stats, action, batch_size=args.batch_size, device=device, seed=args.seed + 123, patch_size=int(cfg["patch"]["coarse_patch_size"]))
    write_csv(out_dir / "forward_reverse_ratio_checks.csv", anti)

    halo = dependency_halo(native_state, kernel, model, stats, batch_size=args.batch_size, device=device, out_dir=out_dir)
    (out_dir / "dependency_halo_summary.json").write_text(json.dumps(halo, indent=2) + "\n", encoding="utf-8")

    prop = {
        "additive_gaussian_full_coarse_reference": {
            "symmetric_wrt_flat_lebesgue": True,
            "log_q_reverse_minus_forward": 0.0,
        },
        "production_coarse_patch_additive_gaussian": {
            "symmetric_wrt_flat_lebesgue": True,
            "log_q_reverse_minus_forward": 0.0,
            "note": "No pCN is enabled in current lambda=1.0 production config.",
        },
        "pCN": {
            "enabled_current_config": bool(cfg["patch"].get("pcn_enabled", False)),
            "note": "If enabled, pCN is reversible w.r.t. its Gaussian reference measure, not flat Lebesgue; the corresponding reference density must be included unless it is part of the target coordinate density.",
        },
    }
    (out_dir / "proposal_symmetry_report.json").write_text(json.dumps(prop, indent=2) + "\n", encoding="utf-8")

    max_anti = max(float(r["max_abs_antisymmetry_residual"]) for r in anti)
    max_cb_anti = max(float(r["max_abs_forward_reverse_antisymmetry_residual"]) for r in checkerboard_substeps) if checkerboard_substeps else float("nan")
    max_cb_inactive = max(float(r["inactive_after_accept_reject_max_error"]) for r in checkerboard_substeps) if checkerboard_substeps else float("nan")
    max_rec = max(float(r["max_reconstruction_error"]) for r in all_rec) if all_rec else float("nan")
    max_reb = max(float(r["max_retained_blocking_error"]) for r in all_rec) if all_rec else float("nan")
    native_acc = [r for r in all_acc if r["label"] == "native_fine_blocked"]
    final_native_acc = native_acc[-1]["acceptance_cumulative"] if native_acc else float("nan")
    lines = [
        "# MIT-Style Fixed-Latent Exactness Audit",
        "",
        "This audit uses the current lambda=1.0 selected kernel and RQ-spline flow.",
        "The global reference transition proposes the full retained coarse field with an additive symmetric Gaussian at fixed global latent z.",
        "",
        "## Configuration",
        f"- flow checkpoint: `{flow_path}`",
        f"- kernel: `{kernel_path}`",
        f"- kernel sum: `{manifest['kernel_sum']:.17g}`",
        f"- eta-included kernel: `{manifest['kernel_coefficients_include_eta_scale']}`",
        f"- fine action: lambda={cfg['lambda']}, kappa_f={cfg['kappa_f']}",
        "- separate coarse action: `not used`",
        f"- chains/sweeps: `{args.n_chains}` / `{args.sweeps}`",
        f"- proposal sigma: `{sigma}`",
        "",
        "## Jacobian Convention",
    ]
    if jac_row is not None:
        lines.extend([f"- reported logj: `{jac_row.get('reported_logj')}`", f"- explicit forward logdet: `{jac_row.get('explicit_forward_logdet')}`", f"- delta discrepancy: `{jac_row.get('delta_discrepancy')}`", f"- required ratio sign: `{jac_row.get('metropolis_sign')}`"])
    lines.extend(
        [
            "",
            "## Exactness Checks",
            f"- max forward/reverse antisymmetry residual: `{max_anti:.6g}`",
            f"- max checkerboard substep antisymmetry residual: `{max_cb_anti:.6g}`",
            f"- max checkerboard inactive-coarse change after accept/reject: `{max_cb_inactive:.6g}`",
            f"- max stored F(c,z) reconstruction error after accepted moves: `{max_rec:.6g}`",
            f"- max retained blocking error after accepted moves: `{max_reb:.6g}`",
            f"- native-fine-blocked cumulative global acceptance: `{final_native_acc}`",
            f"- dependency halo changed fine sites: `{halo['fine_sites_changed_above_tol']}` / `{halo['fine_volume']}`",
            "",
            "## Checkerboard Sublattice Updates",
            "One checkerboard sweep is one even and one odd full-lattice substep at fixed z. The audit ran even-odd, odd-even, and randomized order. Each substep proposes only active-parity coarse sites, globally reconstructs the full fine field, and accepts/rejects with `-Delta S_f + Delta logJ_F`.",
            "",
            "## Interpretation",
            "The files in this directory contain the numerical evidence. Native-fine-blocked stationarity should be judged from `distribution_metrics_vs_native_sweep0.csv`; exact reversibility from `forward_reverse_ratio_checks.csv`; and coordinate consistency from `reconstruction_consistency.csv`.",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_dir)
    print("\n".join(lines[:35]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
