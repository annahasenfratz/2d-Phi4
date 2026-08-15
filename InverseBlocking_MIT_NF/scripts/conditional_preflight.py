#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from pilot_utils import torch_from_numpy_configs, write_json

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from invblock_mit_nf.actions import Phi4Action, Phi4Params
from invblock_mit_nf.blocking import load_kernel_json, momentum_inverse_upscale_to_even_even
from invblock_mit_nf.conditional_flow import ConditionalPhi4Flow


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", type=Path, default=Path("outputs/conditional_preflight"))
    p.add_argument("--coarse-configs", type=Path, default=Path("outputs/coarse_preflight/coarse_configs.npy"))
    p.add_argument("--kernel", type=Path, default=Path("kernels/finite_lambda_kernel_template.json"))
    p.add_argument("--kappa-c", type=float, default=0.30)
    p.add_argument("--kappa-f", type=float, default=0.320)
    p.add_argument("--lam-c", type=float, default=1.0)
    p.add_argument("--lam-f", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    coarse = np.load(args.coarse_configs)
    if len(coarse) < args.batch_size:
        raise ValueError("not enough coarse configs for the requested batch size")
    coarse = coarse[: args.batch_size]
    coarse_t = torch_from_numpy_configs(coarse)
    kernel = load_kernel_json(str(ROOT / args.kernel))
    condition = momentum_inverse_upscale_to_even_even(coarse_t, kernel, Lf=16)

    flow = ConditionalPhi4Flow(L=16, n_layers=2, hidden=8).double()
    y, logq = flow.sample(args.batch_size, condition)
    z, rev_logdet = flow.reverse(y, condition)
    y_round, fwd_logdet = flow.forward(z, condition)

    fixed_violation = float(torch.max(torch.abs(y[:, 0::2, 0::2] - condition[:, 0::2, 0::2])).item())
    roundtrip_error = float(torch.max(torch.abs(y_round - y)).item())
    latent_roundtrip_error = float(torch.max(torch.abs(z - flow.reverse(y_round, condition)[0])).item())
    logdet_consistency = float(torch.max(torch.abs(fwd_logdet + rev_logdet)).item())

    action = Phi4Action(Phi4Params(kappa=args.kappa_f, lam=args.lam_f))
    S = action(y).detach().cpu().numpy()
    logq_np = logq.detach().cpu().numpy()
    loss = float(np.mean(logq_np + S))
    logw = -S - logq_np
    summary = {
        "kernel": str(args.kernel),
        "kappa_c": args.kappa_c,
        "kappa_f": args.kappa_f,
        "lam_c": args.lam_c,
        "lam_f": args.lam_f,
        "batch_size": args.batch_size,
        "fixed_site_violation": fixed_violation,
        "roundtrip_error": roundtrip_error,
        "latent_roundtrip_error": latent_roundtrip_error,
        "logdet_consistency": logdet_consistency,
        "mean_fine_action": float(np.mean(S)),
        "mean_logq": float(np.mean(logq_np)),
        "reverse_kl_estimate": loss,
        "logw_mean": float(np.mean(logw)),
        "logw_std": float(np.std(logw)),
        "finite": bool(np.isfinite(S).all() and np.isfinite(logq_np).all()),
        "even_even_fixed_exact": fixed_violation == 0.0,
    }
    write_json(outdir / "fixed_site_check.json", summary)
    report = (
        "# Conditional preflight\n\n"
        f"- kernel: `{args.kernel}`\n"
        f"- kappa_c = {args.kappa_c}\n"
        f"- kappa_f = {args.kappa_f}\n"
        f"- lambda_c = {args.lam_c}\n"
        f"- lambda_f = {args.lam_f}\n"
        f"- batch_size = {args.batch_size}\n"
        f"- fixed_site_violation = {fixed_violation:.3e}\n"
        f"- roundtrip_error = {roundtrip_error:.3e}\n"
        f"- latent_roundtrip_error = {latent_roundtrip_error:.3e}\n"
        f"- logdet_consistency = {logdet_consistency:.3e}\n"
        f"- mean_fine_action = {summary['mean_fine_action']:.6f}\n"
        f"- mean_logq = {summary['mean_logq']:.6f}\n"
        f"- reverse_kl_estimate = {loss:.6f}\n"
        f"- logw_std = {summary['logw_std']:.6f}\n"
        f"- finite = {summary['finite']}\n"
    )
    (outdir / "fixed_site_check.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
