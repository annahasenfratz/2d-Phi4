#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pilot_utils import generate_coarse_ensemble, write_csv, write_json


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", type=Path, default=Path("outputs/coarse_preflight"))
    p.add_argument("--L", type=int, default=8)
    p.add_argument("--kappa", type=float, default=0.30)
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--n-samples", type=int, default=256)
    p.add_argument("--thermal-sweeps", type=int, default=400)
    p.add_argument("--skip-sweeps", type=int, default=8)
    p.add_argument("--proposal-width", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=12345)
    args = p.parse_args()

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    configs, summary, history = generate_coarse_ensemble(
        L=args.L,
        kappa=args.kappa,
        lam=args.lam,
        n_samples=args.n_samples,
        thermal_sweeps=args.thermal_sweeps,
        skip_sweeps=args.skip_sweeps,
        proposal_width=args.proposal_width,
        seed=args.seed,
    )
    np.save(outdir / "coarse_configs.npy", configs)
    write_csv(outdir / "coarse_history.csv", history)
    write_json(outdir / "coarse_summary.json", summary)
    report = (
        "# Coarse preflight\n\n"
        f"- L = {args.L}\n"
        f"- kappa = {args.kappa}\n"
        f"- lambda = {args.lam}\n"
        f"- n_samples = {args.n_samples}\n"
        f"- thermal_sweeps = {args.thermal_sweeps}\n"
        f"- skip_sweeps = {args.skip_sweeps}\n"
        f"- proposal_width = {args.proposal_width}\n"
        f"- local_acceptance = {summary['local_acceptance']:.6f}\n"
        f"- mean_action = {summary['mean_action']:.6f}\n"
        f"- mean_phi = {summary['mean_phi']:.6f}\n"
        f"- mean_abs_phi = {summary['mean_abs_phi']:.6f}\n"
        f"- mean_phi2 = {summary['mean_phi2']:.6f}\n"
        f"- mean_nn_bond = {summary['mean_nn_bond']:.6f}\n"
        f"- finite = {summary['finite']}\n"
    )
    (outdir / "coarse_report.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
