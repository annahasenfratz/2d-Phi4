#!/usr/bin/env python3
"""Preflight scaffold for symmetric-block conditional NF.

No training is run. This checks an all-sites conditional affine flow with
condition channels from a symmetric coarse field and smooth inverse backbone.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn


PROJECT = Path(__file__).resolve().parents[1]
BASE = PROJECT / "outputs" / "inverse_blocking_step_by_step"
SYM = PROJECT / "outputs" / "symmetric_four_sublattice_blocking"
OUT = PROJECT / "outputs" / "symmetric_conditional_nf_scaffold"
KERNEL_META = PROJECT / "kernels" / "from_perfect_blocking_lam1p0" / "selected_kernel_metadata.json"

L_FINE = 16
L_COARSE = 8
ETA_EXPONENT = 0.25
B = 2
BLOCK_NORM = B ** (ETA_EXPONENT / 2.0)
LAMBDA = 1.0
KAPPA_F = 0.320
LAMBDA_BLOCK = 1.0
SEED = 20240623

SHELLS = {
    "w00": [(0, 0)],
    "w10": [(1, 0), (-1, 0), (0, 1), (0, -1)],
    "w11": [(1, 1), (1, -1), (-1, 1), (-1, -1)],
    "w20": [(2, 0), (-2, 0), (0, 2), (0, -2)],
    "w21": [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)],
    "w22": [(2, 2), (2, -2), (-2, 2), (-2, -2)],
}


def shell_convolve_np(phi: np.ndarray, w: dict[str, float]) -> np.ndarray:
    out = np.zeros_like(phi, dtype=np.float64)
    for shell, offsets in SHELLS.items():
        for dy, dx in offsets:
            out += w[shell] * np.roll(np.roll(phi, -dy, axis=-2), -dx, axis=-1)
    return out


def block_sym_np(phi: np.ndarray, w: dict[str, float]) -> np.ndarray:
    psi = shell_convolve_np(phi, w)
    return 0.25 * BLOCK_NORM * (
        psi[:, 0::2, 0::2] + psi[:, 1::2, 0::2] + psi[:, 0::2, 1::2] + psi[:, 1::2, 1::2]
    )


def block_sym_torch(phi: torch.Tensor, w: dict[str, float]) -> torch.Tensor:
    psi = torch.zeros_like(phi)
    for shell, offsets in SHELLS.items():
        for dy, dx in offsets:
            psi = psi + float(w[shell]) * torch.roll(torch.roll(phi, -dy, dims=-2), -dx, dims=-1)
    return 0.25 * BLOCK_NORM * (
        psi[:, 0::2, 0::2] + psi[:, 1::2, 0::2] + psi[:, 0::2, 1::2] + psi[:, 1::2, 1::2]
    )


def fine_action(phi: torch.Tensor) -> torch.Tensor:
    nn = 0.5 * (
        (phi * torch.roll(phi, -1, dims=-2)).mean(dim=(-2, -1))
        + (phi * torch.roll(phi, -1, dims=-1)).mean(dim=(-2, -1))
    )
    phi2 = (phi**2).mean(dim=(-2, -1))
    phi4 = (phi**4).mean(dim=(-2, -1))
    density = -4.0 * KAPPA_F * nn + (1.0 - 2.0 * LAMBDA) * phi2 + LAMBDA * phi4
    return density * (L_FINE * L_FINE)


class Coupling(nn.Module):
    def __init__(self, dim: int, cond_dim: int, mask: torch.Tensor):
        super().__init__()
        self.register_buffer("mask", mask)
        self.net = nn.Sequential(
            nn.Linear(dim + cond_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 2 * dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def st(self, x: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.cat([x * self.mask, cond], dim=1)
        s, t = self.net(h).chunk(2, dim=1)
        active = 1.0 - self.mask
        return 0.5 * torch.tanh(s) * active, t * active

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        s, t = self.st(x, cond)
        y = x * self.mask + (1.0 - self.mask) * (x * torch.exp(s) + t)
        return y, s.sum(dim=1)

    def inverse(self, y: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        s, t = self.st(y, cond)
        x = y * self.mask + (1.0 - self.mask) * ((y - t) * torch.exp(-s))
        return x, -s.sum(dim=1)


class Flow(nn.Module):
    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        base = torch.arange(dim) % 2
        self.layers = nn.ModuleList(
            [Coupling(dim, cond_dim, (base if i % 2 == 0 else 1 - base).float()) for i in range(4)]
        )

    def forward(self, z: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = z
        ld = torch.zeros(z.shape[0], dtype=z.dtype)
        for layer in self.layers:
            x, d = layer(x, cond)
            ld = ld + d
        return x, ld

    def inverse(self, x: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = x
        ld = torch.zeros(x.shape[0], dtype=x.dtype)
        for layer in reversed(self.layers):
            z, d = layer.inverse(z, cond)
            ld = ld + d
        return z, ld


def main() -> None:
    torch.manual_seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    meta = json.loads(KERNEL_META.read_text())
    weights = {k: float(meta["weights"][k]) for k in ["w00", "w10", "w11", "w20", "w21", "w22"]}
    fine = np.load(BASE / "input_fine_batch.npy").astype(np.float32)
    if (SYM / "symmetric_four_sublattice_coarse.npy").exists() and (SYM / "symmetric_inverse_lowmode_backbone.npy").exists():
        coarse = np.load(SYM / "symmetric_four_sublattice_coarse.npy").astype(np.float32)
        backbone = np.load(SYM / "symmetric_inverse_lowmode_backbone.npy").astype(np.float32)
        kernel_source = "old one-sublattice provenance kernel; scaffold only pending blockavg reoptimized kernel"
    else:
        coarse = block_sym_np(fine.astype(np.float64), weights).astype(np.float32)
        backbone = np.zeros_like(fine)
        kernel_source = "fallback old provenance kernel without precomputed symmetric backbone"
    coarse_broadcast = np.empty_like(fine)
    coarse_broadcast[:, 0::2, 0::2] = coarse
    coarse_broadcast[:, 1::2, 0::2] = coarse
    coarse_broadcast[:, 0::2, 1::2] = coarse
    coarse_broadcast[:, 1::2, 1::2] = coarse
    cond_np = np.stack([coarse_broadcast, backbone], axis=1).reshape(fine.shape[0], -1)
    dim = L_FINE * L_FINE
    cond = torch.tensor(cond_np[:8], dtype=torch.float32)
    flow = Flow(dim, cond.shape[1])
    z = torch.randn(8, dim)
    x, ld = flow(z, cond)
    z_back, inv_ld = flow.inverse(x, cond)
    phi = x.reshape(8, L_FINE, L_FINE)
    coarse_t = torch.tensor(coarse[:8], dtype=torch.float32)
    block_res = block_sym_torch(phi, weights) - coarse_t
    action = fine_action(phi)
    logp = -0.5 * (z**2 + math.log(2.0 * math.pi)).sum(dim=1)
    logq = logp - ld
    penalty = LAMBDA_BLOCK * (block_res**2).mean(dim=(-2, -1)) * (L_COARSE * L_COARSE)
    loss = logq + action + penalty
    summary = {
        "status": "preflight_only_no_training",
        "kernel_source": kernel_source,
        "eta_exponent": ETA_EXPONENT,
        "block_norm": BLOCK_NORM,
        "condition_channels": ["symmetric_coarse_broadcast_2x2", "smooth_inverse_lowmode_backbone"],
        "generated_variables": "all_16x16_fine_sites",
        "fixed_site_constraint": False,
        "shape_device_dtype": {
            "fine_shape": list(fine.shape),
            "coarse_shape": list(coarse.shape),
            "backbone_shape": list(backbone.shape),
            "condition_shape": list(cond_np.shape),
            "generated_dim": dim,
            "device": "cpu",
            "dtype": "float32",
        },
        "invertibility_max_abs_z_error": float(torch.max(torch.abs(z_back - z)).item()),
        "logJ_sign_max_abs_sum": float(torch.max(torch.abs(ld + inv_ld)).item()),
        "finite_action": bool(torch.isfinite(action).all().item()),
        "finite_logq": bool(torch.isfinite(logq).all().item()),
        "finite_loss": bool(torch.isfinite(loss).all().item()),
        "action_mean": float(action.mean().item()),
        "logq_mean": float(logq.mean().item()),
        "block_penalty_mean": float(penalty.mean().item()),
        "loss_mean": float(loss.mean().item()),
        "block_residual_rms": float(torch.sqrt(torch.mean(block_res**2)).item()),
        "block_residual_max_abs": float(torch.max(torch.abs(block_res)).item()),
        "condition_finite": bool(np.isfinite(cond_np).all()),
        "no_fixed_site_constraint_remains": True,
        "training_allowed": False,
    }
    (OUT / "preflight_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (OUT / "preflight_report.md").write_text(
        "# Symmetric Conditional NF Scaffold Preflight\n\n"
        "No training was run. This scaffold generates all 16x16 sites and uses a soft symmetric-block residual term only for the preflight loss check.\n\n"
        f"- invertibility max error: {summary['invertibility_max_abs_z_error']:.12g}\n"
        f"- logJ sign error: {summary['logJ_sign_max_abs_sum']:.12g}\n"
        f"- finite action/logq/loss: {summary['finite_action']}, {summary['finite_logq']}, {summary['finite_loss']}\n"
        f"- block residual RMS: {summary['block_residual_rms']:.12g}\n"
        f"- fixed-site constraint: {summary['fixed_site_constraint']}\n"
        f"- training allowed now: {summary['training_allowed']}\n"
    )


if __name__ == "__main__":
    main()
