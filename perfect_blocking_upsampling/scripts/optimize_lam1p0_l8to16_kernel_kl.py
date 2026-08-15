#!/usr/bin/env python3
"""Optimize the 5x5 D4 kernel with a frozen L8->L16 flow.

The objective is the variational KL (up to a K-independent constant), evaluated
on direct L8 coarse configurations and reparameterized flow samples:
    [S_f(K^-1 psi) - S_c(c) + log q(d|c) + log|det K|] / L_f^2.
Only the first and last terms affect the fixed-flow kernel gradient.
"""
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
sys.path[:0] = [str(PKG / "src"), str(PKG / "scripts")]

from perfect_blocking_upsampling.io import ActionSpec
from train_lam1p0_autoregressive_detail_flow import ARDetailFlow
from train_lam1p0_rqspline_detail_flow import RQSplineARDetailFlow, ResidualSplineARDetailFlow
from train_lam1p0_flow_detail_pilot import load_phi

ETA_SCALE = 2.0 ** 0.125
ORBITS = ((1, 0), (1, 1), (2, 0), (2, 1), (2, 2))


def load_kernel(path: Path) -> tuple[np.ndarray, dict]:
    raw = json.loads(path.read_text())
    matrix = np.asarray(raw["matrix"], dtype=np.float64)
    if matrix.shape == (5, 5):
        return matrix, raw
    if matrix.shape == (7, 7) and np.allclose(matrix[[0, -1], :], 0) and np.allclose(matrix[:, [0, -1]], 0):
        raw["matrix"] = matrix[1:-1, 1:-1].tolist()
        return matrix[1:-1, 1:-1], raw
    raise ValueError(f"expected a 5x5 kernel (or 7x7 with zero border), got {matrix.shape}")


def d4_mask(orbit: tuple[int, int]) -> torch.Tensor:
    coords = torch.arange(-2, 3)
    x, y = torch.meshgrid(coords, coords, indexing="ij")
    return ((torch.maximum(x.abs(), y.abs()) == orbit[0]) & (torch.minimum(x.abs(), y.abs()) == orbit[1]))


def assemble_psi(coarse: torch.Tensor, detail: torch.Tensor) -> torch.Tensor:
    psi = torch.empty((len(coarse), 16, 16), dtype=coarse.dtype, device=coarse.device)
    psi[:, 0::2, 0::2] = coarse
    psi[:, 0::2, 1::2] = detail[:, 0]
    psi[:, 1::2, 0::2] = detail[:, 1]
    psi[:, 1::2, 1::2] = detail[:, 2]
    return psi


def inverse_kernel(psi: torch.Tensor, kernel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    padded = torch.zeros((16, 16), dtype=psi.dtype, device=psi.device)
    padded[:5, :5] = torch.fft.ifftshift(kernel)
    khat = torch.fft.fft2(padded).real
    phi = torch.fft.ifft2(torch.fft.fft2(psi) / khat).real
    return phi, khat


def action_total(phi: torch.Tensor) -> torch.Tensor:
    phi2 = (phi * phi).sum(dim=(1, 2))
    phi4 = (phi ** 4).sum(dim=(1, 2))
    nn = 0.5 * (
        (phi * torch.roll(phi, -1, 1)).sum(dim=(1, 2))
        + (phi * torch.roll(phi, -1, 2)).sum(dim=(1, 2))
    )
    return -phi2 + phi4 - 4.0 * 0.340301 * nn


def build_flow(checkpoint: Path, device: torch.device) -> tuple[ResidualSplineARDetailFlow, dict]:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    if ckpt.get("model_class") != "RQSplineARDetailFlow":
        raise ValueError("the KL kernel optimizer expects an RQSplineARDetailFlow checkpoint")
    config = ckpt["config"]
    spline_cfg = ckpt["spline_settings"]
    affine = ARDetailFlow(
        layers=int(config["layers"]), hidden=int(config["hidden_channels"]),
        kernel_size=int(config["conv_kernel_size"]), log_scale_bound=float(config["log_scale_bound"]),
    ).to(device)
    spline = RQSplineARDetailFlow(
        layers=int(config["layers"]), hidden=int(config["hidden_channels"]),
        kernel_size=int(config["conv_kernel_size"]), num_bins=int(spline_cfg["num_bins"]),
        tail_bound=float(spline_cfg["tail_bound"]), min_bin_width=float(spline_cfg["min_bin_width"]),
        min_bin_height=float(spline_cfg["min_bin_height"]), min_derivative=float(spline_cfg["min_derivative"]),
    ).to(device)
    model = ResidualSplineARDetailFlow(affine, spline).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, ckpt["state"]["stats"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--kernel", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out-kernel", type=Path, required=True)
    p.add_argument("--out-summary", type=Path, required=True)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--n-coarse", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20260810)
    p.add_argument("--min-k", type=float, default=0.50)
    p.add_argument("--max-inv-k", type=float, default=2.0)
    p.add_argument("--guard-weight", type=float, default=1e5)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cpu")

    base, metadata = load_kernel(ROOT / args.kernel)
    base_t = torch.tensor(base, dtype=torch.float32, device=device)
    masks = [d4_mask(orbit).to(device) for orbit in ORBITS]
    multiplicities = torch.tensor([int(mask.sum()) for mask in masks], dtype=torch.float32, device=device)
    delta = torch.nn.Parameter(torch.zeros(len(ORBITS), dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam([delta], lr=args.lr)
    model, stats = build_flow(ROOT / args.checkpoint, device)
    coarse = load_phi(ROOT / "data/configs_phi4_2d/lam1p0_kappac0p340301_L8/configs.npz")[:args.n_coarse]
    coarse_t = torch.tensor(coarse, dtype=torch.float32, device=device)
    cmean = torch.tensor(stats["coarse"]["mean"].reshape(1, 8, 8), dtype=torch.float32, device=device)
    cstd = torch.tensor(stats["coarse"]["std"].reshape(1, 8, 8), dtype=torch.float32, device=device)
    dmean = torch.tensor(stats["detail"]["mean"].reshape(1, 3, 8, 8), dtype=torch.float32, device=device)
    dstd = torch.tensor(stats["detail"]["std"].reshape(1, 3, 8, 8), dtype=torch.float32, device=device)
    history: list[dict[str, float]] = []
    best: tuple[float, np.ndarray] | None = None

    for step in range(args.steps + 1):
        indices = torch.randint(len(coarse_t), (args.batch_size,), device=device)
        c_phys = coarse_t[indices]
        c_norm = (c_phys - cmean) / cstd
        with torch.no_grad():
            d_norm, _logq, _zmax, _logdet = model.sample(c_norm)
        d_phys = d_norm * dstd + dmean
        candidate = base_t.clone()
        for value, mask in zip(delta, masks):
            candidate = candidate + value * mask
        candidate[2, 2] = candidate[2, 2] - torch.sum(delta * multiplicities)
        phi, khat = inverse_kernel(assemble_psi(c_phys, d_phys), candidate)
        min_k = khat.min()
        max_inv = 1.0 / min_k
        logdet = torch.log(khat).sum()
        kl_per_site = (action_total(phi) - action_total(c_phys)).mean() / 256.0 + logdet / 256.0
        guard = args.guard_weight * (
            torch.relu(args.min_k - min_k) ** 2 + torch.relu(max_inv - args.max_inv_k) ** 2
        )
        loss = kl_per_site + guard
        row = {
            "step": float(step), "kl_per_site": float(kl_per_site.detach()),
            "loss": float(loss.detach()), "min_K": float(min_k.detach()),
            "max_invK": float(max_inv.detach()),
        }
        row.update({f"delta_{orbit[0]}{orbit[1]}": float(value.detach()) for orbit, value in zip(ORBITS, delta)})
        history.append(row)
        valid = bool(min_k.detach() >= args.min_k and max_inv.detach() <= args.max_inv_k)
        if valid and (best is None or row["kl_per_site"] < best[0]):
            best = (row["kl_per_site"], candidate.detach().cpu().numpy())
        if step == args.steps:
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    if best is None:
        raise RuntimeError("no candidate satisfied the inverse-kernel guard")
    # The optimization uses float32, but kernel JSON is consumed with an exact
    # eta-sum check. Restore that linear constraint in float64 on serialization.
    final_matrix = np.asarray(best[1], dtype=np.float64)
    final_matrix[2, 2] += ETA_SCALE - float(final_matrix.sum())
    result = dict(metadata)
    result["name"] = f"{metadata.get('name', 'kernel')}_kl_updated"
    result["matrix"] = final_matrix.tolist()
    result["kernel_coefficients_include_eta_scale"] = True
    result["alternating_kl_optimization"] = {
        "objective": "E_q[(S_f-S_c+log|det K|)/L_f^2]; fixed-flow terms omitted from gradient",
        "source_kernel": str(args.kernel), "source_checkpoint": str(args.checkpoint),
        "orbits": [f"{x}{y}" for x, y in ORBITS], "steps": args.steps,
        "best_kl_per_site": best[0], "min_K_guard": args.min_k, "max_invK_guard": args.max_inv_k,
        "history": history,
    }
    out_kernel = ROOT / args.out_kernel
    out_summary = ROOT / args.out_summary
    out_kernel.parent.mkdir(parents=True, exist_ok=True)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_kernel.write_text(json.dumps(result, indent=2) + "\n")
    out_summary.write_text(json.dumps(result["alternating_kl_optimization"], indent=2) + "\n")
    print(json.dumps({"out_kernel": str(out_kernel), **result["alternating_kl_optimization"]}, indent=2))


if __name__ == "__main__":
    main()
