#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from pilot_utils import observables_numpy, torch_from_numpy_configs, write_json

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from invblock_mit_nf.actions import Phi4Action, Phi4Params
from invblock_mit_nf.blocking import load_kernel_json, momentum_inverse_upscale_to_even_even
from invblock_mit_nf.conditional_flow import ConditionalPhi4Flow


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", type=Path, default=Path("outputs/smoke_test"))
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
    action = Phi4Action(Phi4Params(kappa=args.kappa_f, lam=args.lam_f))
    S = action(y)
    logw = -S - logq
    obs = [observables_numpy(sample.detach().cpu().numpy()) for sample in y]
    obs_mean = {k: float(np.mean([o[k] for o in obs])) for k in obs[0]}
    summary = {
        "lambda_c": args.lam_c,
        "lambda_f": args.lam_f,
        "kappa_c": args.kappa_c,
        "kappa_f": args.kappa_f,
        "kappa_cr": 0.3402,
        "kernel": str(args.kernel),
        "batch_size": args.batch_size,
        "even_even_fixed_exact": bool(torch.allclose(y[:, 0::2, 0::2], condition[:, 0::2, 0::2])),
        "mean_fine_action": float(S.mean().item()),
        "mean_logq": float(logq.mean().item()),
        "reverse_kl_estimate": float((logq + S).mean().item()),
        "logw_mean": float(logw.mean().item()),
        "logw_std": float(logw.std().item()),
        "finite": bool(torch.isfinite(S).all() and torch.isfinite(logq).all() and torch.isfinite(logw).all()),
        "observables_mean": obs_mean,
    }
    write_json(outdir / "smoke_summary.json", summary)
    report = (
        "# Smoke test\n\n"
        f"- lambda_c = {args.lam_c}\n"
        f"- lambda_f = {args.lam_f}\n"
        f"- kappa_c = {args.kappa_c}\n"
        f"- kappa_f = {args.kappa_f}\n"
        f"- kappa_cr = 0.3402\n"
        f"- kernel = `{args.kernel}`\n"
        f"- batch_size = {args.batch_size}\n"
        f"- even_even_fixed_exact = {summary['even_even_fixed_exact']}\n"
        f"- mean_fine_action = {summary['mean_fine_action']:.6f}\n"
        f"- mean_logq = {summary['mean_logq']:.6f}\n"
        f"- reverse_kl_estimate = {summary['reverse_kl_estimate']:.6f}\n"
        f"- logw_std = {summary['logw_std']:.6f}\n"
        f"- finite = {summary['finite']}\n"
    )
    (outdir / "smoke_report.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
