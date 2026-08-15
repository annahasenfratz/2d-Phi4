#!/usr/bin/env python3
"""Reverse-KL scalar detail-scale scan for the retained wrapped flow.

This diagnostic uses direct native-action coarse configurations only.  It does
not load target-volume native fine fields.  A sampled flow detail field d is
mapped to d_alpha = alpha d, so the conditional proposal density is exactly
log q_alpha(d_alpha|c) = log q_flow(d|c) - 3 L_c^2 log(alpha).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PKG = PROJECT_ROOT / "perfect_blocking_upsampling"
sys.path[:0] = [str(PKG / "src"), str(PKG / "scripts")]

from perfect_blocking_upsampling.actions import action_total  # noqa: E402
from perfect_blocking_upsampling.blocking import apply_kernel, assemble_psi, inverse_kernel, load_kernel_matrix, load_phi  # noqa: E402
from perfect_blocking_upsampling.io import ActionSpec  # noqa: E402
from run_lam1p0_l16to32_rqspline_zeroshot import (  # noqa: E402
    build_model_from_checkpoint,
    sample_model_lattice,
    stationary_stats,
)


def parse_alphas(text: str) -> list[float]:
    values = [float(value) for value in text.split(",") if value.strip()]
    if not values or any(value <= 0.0 for value in values):
        raise ValueError("--alphas must contain positive values")
    return values


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    ap = argparse.ArgumentParser(description="Reverse-KL scalar detail-scale scan without target fine configurations.")
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--coarse-source", type=Path, required=True)
    ap.add_argument("--flow-checkpoint", type=Path, required=True)
    ap.add_argument("--kernel-path", type=Path, required=True)
    ap.add_argument("--from-L", type=int, default=16)
    ap.add_argument("--to-L", type=int, default=32)
    ap.add_argument("--n-coarse", type=int, default=256)
    ap.add_argument("--coarse-start-index", type=int, default=0)
    ap.add_argument("--alphas", default="0.85,0.90,0.95,0.975,1.0,1.025,1.05,1.10,1.15")
    ap.add_argument("--seed", type=int, default=2026072401)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    if args.to_L != 2 * args.from_L:
        raise RuntimeError("only factor-two transfers are supported")
    alphas = parse_alphas(args.alphas)
    run = args.run_dir.resolve()
    (run / "observables").mkdir(parents=True, exist_ok=True)
    (run / "logs").mkdir(parents=True, exist_ok=True)
    write_json(run / "status.json", {
        "status": "running", "stage": "startup", "run_dir": str(run),
        "n_coarse": args.n_coarse, "alphas": alphas,
    })
    print(json.dumps({"run_dir": str(run), "status": "running", "stage": "startup"}), flush=True)
    for path in (args.coarse_source, args.flow_checkpoint, args.kernel_path):
        if not path.exists():
            raise FileNotFoundError(path)

    coarse_all = load_phi(args.coarse_source)
    if coarse_all.shape[1:] != (args.from_L, args.from_L):
        raise RuntimeError(f"expected L{args.from_L} coarse source, got {coarse_all.shape[1:]}")
    stop = args.coarse_start_index + args.n_coarse
    if stop > len(coarse_all):
        raise RuntimeError(f"requested coarse indices [{args.coarse_start_index}, {stop}) exceed source size {len(coarse_all)}")
    coarse_indices = np.arange(args.coarse_start_index, stop, dtype=np.int64)
    coarse = coarse_all[coarse_indices].astype(np.float32)
    kernel, kernel_metadata = load_kernel_matrix(args.kernel_path)
    if not bool(kernel_metadata.get("kernel_coefficients_include_eta_scale", False)):
        raise RuntimeError("kernel must include eta scale; no extra eta multiplier is allowed")
    device = torch.device(args.device)
    checkpoint = torch.load(args.flow_checkpoint, map_location=device, weights_only=False)
    model, load_report = build_model_from_checkpoint(checkpoint, args.from_L, device)
    stats = stationary_stats(checkpoint["state"]["stats"], args.from_L)
    detail, logq_flow, zmax, flow_logdet = sample_model_lattice(
        model, coarse, stats, batch_size=args.batch_size, device=device, seed=args.seed,
    )
    write_json(run / "progress.json", {"stage": "flow_samples_ready", "completed_alphas": 0, "total_alphas": len(alphas)})
    print(json.dumps({"stage": "flow_samples_ready", "n_coarse": args.n_coarse}), flush=True)
    action = ActionSpec("phi4_nn", 1.0, 0.340301)
    coarse_action = action_total(coarse, action).astype(np.float64)
    dim_detail = 3 * args.from_L * args.from_L
    rows: list[dict[str, object]] = []
    sample_rows: list[dict[str, object]] = []
    rng = np.random.default_rng(args.seed + 1)
    pairings = np.stack([rng.permutation(args.n_coarse) for _ in range(128)])
    for alpha in alphas:
        scaled_detail = (alpha * detail).astype(np.float32)
        psi = assemble_psi(coarse, scaled_detail)
        fine, _ = inverse_kernel(psi, kernel)
        fine_action = action_total(fine, action).astype(np.float64)
        logq = logq_flow - dim_detail * math.log(alpha)
        reverse_kl = fine_action + logq
        # Existing direct-MIT convention, up to fixed kernel constants:
        # log w = -S_f + S_c - log q(detail|coarse).
        log_weight = -fine_action + coarse_action - logq
        loga = log_weight[pairings] - log_weight[None, :]
        acceptance = np.minimum(1.0, np.exp(np.minimum(0.0, loga))).mean()
        reb_psi = apply_kernel(fine, kernel)
        reblock_error = np.max(np.abs(reb_psi[:, 0::2, 0::2] - coarse))
        rows.append({
            "alpha": alpha,
            "n_coarse": args.n_coarse,
            "reverse_kl_surrogate_mean": float(reverse_kl.mean()),
            "reverse_kl_surrogate_se": float(reverse_kl.std(ddof=1) / np.sqrt(args.n_coarse)),
            "fine_action_mean": float(fine_action.mean()),
            "fine_action_std": float(fine_action.std(ddof=1)),
            "logq_detail_mean": float(logq.mean()),
            "log_weight_mean": float(log_weight.mean()),
            "log_weight_std": float(log_weight.std(ddof=1)),
            "direct_mit_acceptance_surrogate": float(acceptance),
            "reblocking_max_error": float(reblock_error),
            "nonfinite_count": int(np.sum(~np.isfinite(fine))),
        })
        sample_rows.extend({
            "alpha": alpha, "source_coarse_index": int(coarse_indices[i]),
            "fine_action": float(fine_action[i]), "coarse_action": float(coarse_action[i]),
            "logq_detail": float(logq[i]), "reverse_kl_surrogate": float(reverse_kl[i]),
            "log_weight": float(log_weight[i]), "zmax": float(zmax[i]), "flow_logdet": float(flow_logdet[i]),
        } for i in range(args.n_coarse))
        # Persist each completed alpha so a monitored run exposes useful output.
        write_csv(run / "observables" / "alpha_scan.csv", rows)
        write_csv(run / "observables" / "per_sample_alpha_scan.csv", sample_rows)
        best_so_far = min(rows, key=lambda row: float(row["reverse_kl_surrogate_mean"]))
        write_json(run / "progress.json", {
            "stage": "alpha_scan", "completed_alphas": len(rows), "total_alphas": len(alphas),
            "last_alpha": alpha, "best_alpha_by_reverse_kl_so_far": best_so_far["alpha"],
        })
        print(json.dumps({"stage": "alpha_complete", "alpha": alpha,
                          "reverse_kl": rows[-1]["reverse_kl_surrogate_mean"],
                          "acceptance_surrogate": rows[-1]["direct_mit_acceptance_surrogate"],
                          "best_alpha": best_so_far["alpha"]}), flush=True)
    best = min(rows, key=lambda row: float(row["reverse_kl_surrogate_mean"]))
    (run / "coarse_source_indices.csv").write_text("source_coarse_index\n" + "\n".join(map(str, coarse_indices)) + "\n", encoding="utf-8")
    config = vars(args) | {
        "alphas": alphas,
        "objective": "E_q[S_f(phi_alpha) + log q_alpha(detail|coarse)]",
        "density_correction": "logq_alpha = logq_flow - 3*L_c^2*log(alpha)",
        "fine_native_configurations_used": False,
        "kernel_metadata": kernel_metadata,
        "flow_load_report": load_report,
        "best_alpha_by_reverse_kl": best["alpha"],
    }
    write_json(run / "run_config.json", config)
    (run / "summary.md").write_text(
        "# Reverse-KL Detail-Scale Pilot\n\n"
        f"- Direct L{args.from_L} coarse fields: {args.n_coarse}, indices [{args.coarse_start_index}, {stop}).\n"
        "- No target-volume native fine configurations were loaded or used.\n"
        f"- Best common detail scale by reverse-KL surrogate: `{best['alpha']}`.\n"
        f"- Its direct-MIT acceptance surrogate: `{best['direct_mit_acceptance_surrogate']:.6f}`.\n",
        encoding="utf-8",
    )
    write_json(run / "status.json", {"status": "completed", "run_dir": str(run), "best": best})
    print(json.dumps({"run_dir": str(run), "status": "completed", "best": best}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
