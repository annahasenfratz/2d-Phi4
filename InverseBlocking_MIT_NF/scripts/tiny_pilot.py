#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
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
from invblock_mit_nf.train_inverse_kl import reverse_kl_step


def save_checkpoint(path: Path, *, epoch: int, flow: torch.nn.Module, optimizer: torch.optim.Optimizer, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": flow.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "summary": summary,
        },
        path,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", type=Path, default=Path("outputs/tiny_pilot"))
    p.add_argument("--coarse-configs", type=Path, default=Path("outputs/coarse_preflight/coarse_configs.npy"))
    p.add_argument("--kernel", type=Path, default=Path("kernels/finite_lambda_kernel_template.json"))
    p.add_argument("--kappa-c", type=float, default=0.30)
    p.add_argument("--kappa-f", type=float, default=0.320)
    p.add_argument("--lam-c", type=float, default=1.0)
    p.add_argument("--lam-f", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--checkpoint-every", type=int, default=5)
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()

    outdir = args.outdir
    ckpt_dir = outdir / "checkpoints"
    outdir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    coarse = np.load(args.coarse_configs)
    if len(coarse) < args.batch_size:
        raise ValueError("not enough coarse configs for the requested batch size")
    coarse = coarse[: args.batch_size]
    coarse_t = torch_from_numpy_configs(coarse)
    kernel = load_kernel_json(str(ROOT / args.kernel))
    condition = momentum_inverse_upscale_to_even_even(coarse_t, kernel, Lf=16)

    flow = ConditionalPhi4Flow(L=16, n_layers=2, hidden=8).double()
    action = Phi4Action(Phi4Params(kappa=args.kappa_f, lam=args.lam_f))
    optimizer = torch.optim.Adam(flow.parameters(), lr=args.lr)

    history: list[dict[str, float]] = []
    for epoch in range(args.epochs):
        stats = reverse_kl_step(flow, action, optimizer, condition)
        with torch.no_grad():
            y, logq = flow.sample(args.batch_size, condition)
            S = action(y)
            logw = -S - logq
            obs = np.stack([observables_numpy(sample.detach().cpu().numpy()) for sample in y], axis=0)
            fixed_violation = float(torch.max(torch.abs(y[:, 0::2, 0::2] - condition[:, 0::2, 0::2])).item())
        row = {
            "epoch": epoch,
            "loss": stats["loss"],
            "S": stats["S"],
            "logq": stats["logq"],
            "ess_frac_batch": stats["ess_frac_batch"],
            "mean_fine_action": float(S.mean().item()),
            "mean_logq": float(logq.mean().item()),
            "reverse_kl_estimate": float((logq + S).mean().item()),
            "logw_mean": float(logw.mean().item()),
            "logw_std": float(logw.std().item()),
            "even_even_fixed_max_abs": fixed_violation,
            "finite": float(torch.isfinite(S).all() and torch.isfinite(logq).all() and torch.isfinite(logw).all()),
        }
        for key in ("mean_phi", "abs_mean_phi", "m2", "m4", "binder", "nn", "diag"):
            row[key] = float(np.mean([o[key] for o in obs]))
        history.append(row)
        if (epoch + 1) % args.checkpoint_every == 0 or epoch + 1 == args.epochs:
            save_checkpoint(ckpt_dir / f"epoch_{epoch+1:04d}.pt", epoch=epoch + 1, flow=flow, optimizer=optimizer, summary=row)

    history_path = outdir / "history.csv"
    with history_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    summary = {
        "lambda_c": args.lam_c,
        "lambda_f": args.lam_f,
        "kappa_c": args.kappa_c,
        "kappa_f": args.kappa_f,
        "kappa_cr": 0.3402,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "checkpoint_every": args.checkpoint_every,
        "final": history[-1],
        "device": "cpu",
        "finite": bool(all(bool(r["finite"]) for r in history)),
    }
    write_json(outdir / "summary.json", summary)
    report = (
        "# Tiny pilot\n\n"
        f"- lambda_c = {args.lam_c}\n"
        f"- lambda_f = {args.lam_f}\n"
        f"- kappa_c = {args.kappa_c}\n"
        f"- kappa_f = {args.kappa_f}\n"
        f"- epochs = {args.epochs}\n"
        f"- batch_size = {args.batch_size}\n"
        f"- lr = {args.lr}\n"
        f"- checkpoint_every = {args.checkpoint_every}\n"
        f"- final_loss = {history[-1]['loss']:.6f}\n"
        f"- final_logq = {history[-1]['logq']:.6f}\n"
        f"- final_S = {history[-1]['S']:.6f}\n"
        f"- final_even_even_fixed_max_abs = {history[-1]['even_even_fixed_max_abs']:.3e}\n"
        f"- final_logw_std = {history[-1]['logw_std']:.6f}\n"
        f"- finite = {summary['finite']}\n"
        f"- device = cpu\n"
    )
    (outdir / "report.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
