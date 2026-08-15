"""Latent/detail coordinate maps for the retained wrapped RQ-spline flow."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch


def detail_from_psi(psi: np.ndarray) -> np.ndarray:
    return np.stack([psi[:, 0::2, 1::2], psi[:, 1::2, 0::2], psi[:, 1::2, 1::2]], axis=1).astype(np.float32)


def infer_rqspline_latents_and_logj(model: Any, coarse_phys: np.ndarray, detail_phys: np.ndarray, stats: dict[str, Any], *, batch_size: int, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    coarse_std = ((coarse_phys - stats["coarse_mean"]) / stats["coarse_std"]).astype(np.float32)
    mean = np.asarray(stats["detail_mean"], dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.asarray(stats["detail_std"], dtype=np.float32).reshape(1, 3, 1, 1)
    detail_std = ((detail_phys - mean) / std).astype(np.float32)
    z_out = np.zeros_like(detail_std, dtype=np.float32)
    logj = np.zeros(len(coarse_std), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, len(coarse_std), batch_size):
            stop = min(start + batch_size, len(coarse_std))
            coarse_batch = torch.from_numpy(coarse_std[start:stop]).to(device)
            detail_batch = torch.from_numpy(detail_std[start:stop]).to(device)
            reconstructed = torch.zeros_like(detail_batch)
            z_batch = torch.zeros_like(detail_batch)
            logj_batch = torch.zeros(stop - start, dtype=coarse_batch.dtype, device=device)
            for stage in range(3):
                spline_cond = model.cond(coarse_batch, detail_batch, stage)
                pre_spline, spline_inv_logdet = model.spline.flows[stage].inverse(detail_batch[:, stage].flatten(1), spline_cond)
                affine_cond = model.affine_base.cond(coarse_batch, reconstructed, stage)
                z_stage, affine_inv_logdet = model.affine_base.flows[stage].inverse(pre_spline, affine_cond)
                z_batch[:, stage] = z_stage.reshape(stop - start, coarse_batch.shape[1], coarse_batch.shape[2])
                logj_batch -= affine_inv_logdet + spline_inv_logdet
                reconstructed[:, stage] = detail_batch[:, stage]
            z_out[start:stop] = z_batch.cpu().numpy().astype(np.float32)
            logj[start:stop] = logj_batch.cpu().numpy().astype(np.float64)
    return z_out, logj


def reconstruct_rqspline_from_latents(model: Any, coarse_phys: np.ndarray, z: np.ndarray, stats: dict[str, Any], *, batch_size: int, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    coarse_std = ((coarse_phys - stats["coarse_mean"]) / stats["coarse_std"]).astype(np.float32)
    mean = np.asarray(stats["detail_mean"], dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.asarray(stats["detail_std"], dtype=np.float32).reshape(1, 3, 1, 1)
    detail_out = np.zeros_like(z, dtype=np.float32)
    logj = np.zeros(len(coarse_std), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, len(coarse_std), batch_size):
            stop = min(start + batch_size, len(coarse_std))
            coarse_batch = torch.from_numpy(coarse_std[start:stop]).to(device)
            z_batch = torch.from_numpy(z[start:stop]).to(device)
            detail_batch = torch.zeros_like(z_batch)
            logj_batch = torch.zeros(stop - start, dtype=coarse_batch.dtype, device=device)
            for stage in range(3):
                affine_cond = model.affine_base.cond(coarse_batch, detail_batch, stage)
                affine_out, affine_logdet = model.affine_base.flows[stage].forward(z_batch[:, stage].flatten(1), affine_cond)
                spline_cond = model.cond(coarse_batch, detail_batch, stage)
                detail_stage, spline_logdet = model.spline.flows[stage].forward(affine_out, spline_cond)
                detail_batch[:, stage] = detail_stage.reshape(stop - start, coarse_batch.shape[1], coarse_batch.shape[2])
                logj_batch += affine_logdet + spline_logdet
            detail_out[start:stop] = detail_batch.cpu().numpy().astype(np.float32) * std + mean
            logj[start:stop] = logj_batch.cpu().numpy().astype(np.float64)
    return detail_out, logj
