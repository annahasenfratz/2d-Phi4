#!/usr/bin/env python3
"""Exact global independence-MH audit for an L8->L16 RQ-spline flow."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "perfect_blocking_upsampling"
sys.path[:0] = [str(PKG / "scripts")]

from train_lam1p0_autoregressive_detail_flow import ARDetailFlow
from train_lam1p0_rqspline_detail_flow import RQSplineARDetailFlow, ResidualSplineARDetailFlow
from train_lam1p0_flow_detail_pilot import load_phi


def load_kernel(path: Path) -> np.ndarray:
    matrix = np.asarray(json.loads(path.read_text())["matrix"], dtype=np.float64)
    if matrix.shape == (7, 7) and np.allclose(matrix[[0, -1], :], 0) and np.allclose(matrix[:, [0, -1]], 0):
        matrix = matrix[1:-1, 1:-1]
    if matrix.shape != (5, 5):
        raise ValueError(f"expected a 5x5 kernel, got {matrix.shape}")
    return matrix


def build_flow(checkpoint: Path, device: torch.device) -> tuple[ResidualSplineARDetailFlow, dict]:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    if ckpt.get("model_class") != "RQSplineARDetailFlow":
        raise ValueError(f"not an RQSpline checkpoint: {checkpoint}")
    cfg, spl = ckpt["config"], ckpt["spline_settings"]
    affine = ARDetailFlow(layers=int(cfg["layers"]), hidden=int(cfg["hidden_channels"]), kernel_size=int(cfg["conv_kernel_size"]), log_scale_bound=float(cfg["log_scale_bound"])).to(device)
    spline = RQSplineARDetailFlow(layers=int(cfg["layers"]), hidden=int(cfg["hidden_channels"]), kernel_size=int(cfg["conv_kernel_size"]), num_bins=int(spl["num_bins"]), tail_bound=float(spl["tail_bound"]), min_bin_width=float(spl["min_bin_width"]), min_bin_height=float(spl["min_bin_height"]), min_derivative=float(spl["min_derivative"])).to(device)
    model = ResidualSplineARDetailFlow(affine, spline).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["state"]["stats"]


def assemble_psi(coarse: torch.Tensor, detail: torch.Tensor) -> torch.Tensor:
    psi = torch.empty((len(coarse), 16, 16), dtype=coarse.dtype, device=coarse.device)
    psi[:, 0::2, 0::2] = coarse
    psi[:, 0::2, 1::2] = detail[:, 0]
    psi[:, 1::2, 0::2] = detail[:, 1]
    psi[:, 1::2, 1::2] = detail[:, 2]
    return psi


def action_total(phi: torch.Tensor) -> torch.Tensor:
    bonds = (phi * torch.roll(phi, -1, 1)).sum((1, 2)) + (phi * torch.roll(phi, -1, 2)).sum((1, 2))
    return -(phi * phi).sum((1, 2)) + (phi ** 4).sum((1, 2)) - 2.0 * 0.340301 * bonds


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--kernel", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=20260810)
    args = p.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    kernel = load_kernel(ROOT / args.kernel)
    padded = np.zeros((16, 16)); padded[:5, :5] = np.fft.ifftshift(kernel)
    khat = np.fft.fft2(padded).real
    if khat.min() <= 0:
        raise ValueError("kernel is not positive/invertible on the L16 Fourier grid")
    logdet = float(np.log(khat).sum())
    model, stats = build_flow(ROOT / args.checkpoint, device)
    coarse = load_phi(ROOT / "data/configs_phi4_2d/lam1p0_kappac0p340301_L8/configs.npz")[:args.n]
    cmean = torch.tensor(stats["coarse"]["mean"].reshape(1, 8, 8), dtype=torch.float32, device=device)
    cstd = torch.tensor(stats["coarse"]["std"].reshape(1, 8, 8), dtype=torch.float32, device=device)
    dmean = torch.tensor(stats["detail"]["mean"].reshape(1, 3, 8, 8), dtype=torch.float32, device=device)
    dstd = torch.tensor(stats["detail"]["std"].reshape(1, 3, 8, 8), dtype=torch.float32, device=device)
    kernel_t = torch.tensor(kernel, dtype=torch.float32, device=device)
    torch.manual_seed(args.seed)
    sf_all, sc_all, logq_all = [], [], []
    with torch.no_grad():
        for start in range(0, len(coarse), args.batch_size):
            c = torch.tensor(coarse[start:start + args.batch_size], dtype=torch.float32, device=device)
            dnorm, logq_norm, _, _ = model.sample((c - cmean) / cstd)
            d = dnorm * dstd + dmean
            psi = assemble_psi(c, d)
            pad = torch.zeros((16, 16), dtype=torch.float32, device=device)
            pad[:5, :5] = torch.fft.ifftshift(kernel_t)
            phi = torch.fft.ifft2(torch.fft.fft2(psi) / torch.fft.fft2(pad).real).real
            sf_all.append(action_total(phi).cpu().numpy())
            sc_all.append(action_total(c).cpu().numpy())
            logq_all.append((logq_norm - torch.log(dstd).sum()).cpu().numpy())
    sf, sc, logq = map(np.concatenate, (sf_all, sc_all, logq_all))
    logw = -sf + sc - logq - logdet
    weights = np.exp(logw - logw.max()); weights /= weights.sum()
    ess = float(1.0 / (weights @ weights) / len(weights))
    rng = np.random.default_rng(args.seed + 1)
    current = rng.choice(len(logw), size=200_000, p=weights)
    proposal = rng.integers(len(logw), size=200_000)
    acceptance = float(np.minimum(1.0, np.exp(np.minimum(logw[proposal] - logw[current], 0.0))).mean())
    summary = {"checkpoint": str(args.checkpoint), "kernel": str(args.kernel), "n": len(logw), "logdet_K": logdet, "min_K": float(khat.min()), "max_invK": float(1.0 / khat.min()), "logw_mean": float(logw.mean()), "logw_std": float(logw.std(ddof=1)), "logw_q01": float(np.quantile(logw, .01)), "logw_q99": float(np.quantile(logw, .99)), "ess_over_n": ess, "stationary_independence_MH_acceptance_proxy": acceptance}
    np.savez_compressed(out / "global_ar_samples.npz", logw=logw, Sf=sf, Sc=sc, logq_detail=logq)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
