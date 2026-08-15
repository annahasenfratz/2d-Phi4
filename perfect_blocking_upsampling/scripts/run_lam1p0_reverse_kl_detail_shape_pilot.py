#!/usr/bin/env python3
"""Exact two-parameter detail-shape scan for a wrapped-flow proposal.

The map is applied in units of each sampled detail sector RMS:
  u -> a * [u - delta * tanh(u / b)^3].
It expands the bulk with ``a`` while compressing tails with ``delta``.  The
map is monotone for delta < 4 b / 3, preserves the blocked coarse field, and
has an explicit product Jacobian.
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
from run_lam1p0_l16to32_rqspline_zeroshot import build_model_from_checkpoint, sample_model_lattice, stationary_stats  # noqa: E402


def values(text: str) -> list[float]:
    out = [float(x) for x in text.split(",") if x.strip()]
    if not out:
        raise ValueError("empty scan value list")
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def write_json(path: Path, payload: dict[str, object]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def observables(phi: np.ndarray, action: ActionSpec) -> dict[str, float]:
    arr = np.asarray(phi, dtype=np.float64)
    phi2 = float(np.mean(arr * arr))
    phi4 = float(np.mean(arr**4))
    nn = float(0.5 * (np.mean(arr * np.roll(arr, -1, axis=1)) + np.mean(arr * np.roll(arr, -1, axis=2))))
    action_density = float(np.mean(action_total(arr, action) / (arr.shape[1] * arr.shape[2])))
    return {"action_density": action_density, "phi2": phi2, "phi4": phi4, "NN": nn, "kurtosis": phi4 / (phi2 * phi2)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--coarse-source", type=Path, required=True)
    ap.add_argument("--flow-checkpoint", type=Path, required=True)
    ap.add_argument("--kernel-path", type=Path, required=True)
    ap.add_argument("--from-L", type=int, default=16)
    ap.add_argument("--to-L", type=int, default=32)
    ap.add_argument("--n-coarse", type=int, default=256)
    ap.add_argument("--coarse-start-index", type=int, default=256)
    ap.add_argument("--scales", default="0.995,1.0,1.005,1.01,1.015")
    ap.add_argument("--tail-strengths", default="0.0,0.02,0.04,0.06,0.08")
    ap.add_argument("--tail-width", type=float, default=1.0)
    ap.add_argument(
        "--sectors", choices=("all", "d01", "d10", "d11", "d01d10"), default="all",
        help="Detail sectors receiving the shape map; unselected sectors are unchanged.",
    )
    ap.add_argument("--target-action-density", type=float, default=-0.55454)
    ap.add_argument("--target-phi2", type=float, default=0.83090)
    ap.add_argument("--target-phi4", type=float, default=1.05490)
    ap.add_argument("--target-kurtosis", type=float, default=1.52790)
    ap.add_argument("--seed", type=int, default=2026072402)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    if args.to_L != 2 * args.from_L or args.tail_width <= 0.0:
        raise RuntimeError("requires factor-two transfer and positive tail width")
    scales, strengths = values(args.scales), values(args.tail_strengths)
    if any(scale <= 0.0 for scale in scales) or any(delta < 0.0 or delta >= 4.0 * args.tail_width / 3.0 for delta in strengths):
        raise RuntimeError("invalid scale or tail strength; require scale>0 and 0<=delta<4*b/3")
    run = args.run_dir.resolve()
    (run / "observables").mkdir(parents=True, exist_ok=True)
    write_json(run / "status.json", {"status": "running", "stage": "startup", "run_dir": str(run)})
    print(json.dumps({"run_dir": str(run), "status": "running", "stage": "startup"}), flush=True)

    all_coarse = load_phi(args.coarse_source)
    stop = args.coarse_start_index + args.n_coarse
    if all_coarse.shape[1:] != (args.from_L, args.from_L) or stop > len(all_coarse):
        raise RuntimeError("coarse source does not satisfy requested lattice/index range")
    coarse = all_coarse[args.coarse_start_index:stop].astype(np.float32)
    kernel, kernel_metadata = load_kernel_matrix(args.kernel_path)
    if not bool(kernel_metadata.get("kernel_coefficients_include_eta_scale", False)):
        raise RuntimeError("kernel eta scale must already be included")
    device = torch.device(args.device)
    checkpoint = torch.load(args.flow_checkpoint, map_location=device, weights_only=False)
    model, load_report = build_model_from_checkpoint(checkpoint, args.from_L, device)
    detail, base_logq, zmax, flow_logdet = sample_model_lattice(
        model, coarse, stationary_stats(checkpoint["state"]["stats"], args.from_L),
        batch_size=args.batch_size, device=device, seed=args.seed,
    )
    action = ActionSpec("phi4_nn", 1.0, 0.340301)
    coarse_action = action_total(coarse, action).astype(np.float64)
    base_obs = observables(inverse_kernel(assemble_psi(coarse, detail), kernel)[0], action)
    sector_rms = np.sqrt(np.mean(detail.astype(np.float64) ** 2, axis=(0, 2, 3)))
    selected_sectors = {"all": (0, 1, 2), "d01": (0,), "d10": (1,), "d11": (2,), "d01d10": (0, 1)}[args.sectors]
    target_observables = {
        "action_density": args.target_action_density,
        "phi2": args.target_phi2,
        "phi4": args.target_phi4,
        "kurtosis": args.target_kurtosis,
    }
    target_delta = {key: target_observables[key] - base_obs[key] for key in target_observables}
    # Requested absolute targets are more stable than relative shifts when a
    # fresh direct-coarse sample has a slightly different baseline mean.
    target_tolerance = {"action_density": 0.002, "phi2": 0.001, "phi4": 0.002, "kurtosis": 0.005}
    dim = 3 * args.from_L * args.from_L
    rng = np.random.default_rng(args.seed + 1)
    pairings = np.stack([rng.permutation(args.n_coarse) for _ in range(128)])
    rows: list[dict[str, object]] = []
    write_json(run / "progress.json", {"stage": "flow_samples_ready", "completed": 0, "total": len(scales) * len(strengths)})
    print(json.dumps({"stage": "flow_samples_ready", "sector_rms": sector_rms.tolist(), "base": base_obs}), flush=True)
    for scale in scales:
        for delta in strengths:
            transformed = detail.copy()
            u = detail[:, selected_sectors] / sector_rms[np.asarray(selected_sectors)].reshape(1, len(selected_sectors), 1, 1)
            t = np.tanh(u / args.tail_width)
            transformed_u = scale * (u - delta * t**3)
            transformed[:, selected_sectors] = (transformed_u * sector_rms[np.asarray(selected_sectors)].reshape(1, len(selected_sectors), 1, 1)).astype(np.float32)
            deriv = scale * (1.0 - (3.0 * delta / args.tail_width) * t * t * (1.0 - t * t))
            if np.any(deriv <= 0.0):
                raise RuntimeError("nonpositive detail-map derivative")
            logq = base_logq - np.log(deriv).sum(axis=(1, 2, 3))
            fine, _ = inverse_kernel(assemble_psi(coarse, transformed), kernel)
            fine_action = action_total(fine, action).astype(np.float64)
            obs = observables(fine, action)
            delta_obs = {key: obs[key] - base_obs[key] for key in target_delta}
            match_score = sum(((obs[key] - target_observables[key]) / target_tolerance[key]) ** 2 for key in target_observables)
            log_weight = -fine_action + coarse_action - logq
            loga = log_weight[pairings] - log_weight[None, :]
            acceptance = float(np.minimum(1.0, np.exp(np.minimum(0.0, loga))).mean())
            reb = np.max(np.abs(apply_kernel(fine, kernel)[:, 0::2, 0::2] - coarse))
            row = {
                "sectors": args.sectors, "scale": scale, "tail_strength": delta, "tail_width": args.tail_width,
                "reverse_kl_surrogate_mean": float(np.mean(fine_action + logq)),
                "direct_mit_acceptance_surrogate": acceptance,
                "action_density": obs["action_density"], "phi2": obs["phi2"], "phi4": obs["phi4"], "NN": obs["NN"], "kurtosis": obs["kurtosis"],
                "delta_action_density": delta_obs["action_density"], "delta_phi2": delta_obs["phi2"], "delta_phi4": delta_obs["phi4"], "delta_kurtosis": delta_obs["kurtosis"],
                "target_match_score": float(match_score), "reblocking_max_error": float(reb), "nonfinite_count": int(np.sum(~np.isfinite(fine))),
            }
            rows.append(row)
            write_csv(run / "observables" / "shape_scan.csv", rows)
            best = min(rows, key=lambda item: float(item["target_match_score"]))
            write_json(run / "progress.json", {"stage": "shape_scan", "completed": len(rows), "total": len(scales) * len(strengths), "best": best})
            print(json.dumps({"stage": "point_complete", **row, "best_scale": best["scale"], "best_delta": best["tail_strength"]}), flush=True)
    best = min(rows, key=lambda item: float(item["target_match_score"]))
    config = vars(args) | {"sector_rms": sector_rms.tolist(), "base_observables": base_obs, "target_observables": target_observables, "target_delta": target_delta,
                            "selected_sectors": list(selected_sectors),
                            "map": "selected d_s -> rms_s * scale * [d_s/rms_s - delta*tanh((d_s/rms_s)/b)^3]",
                            "density_correction": "logq_new = logq_flow - sum log(map_derivative)",
                            "fine_native_configurations_used": False, "kernel_metadata": kernel_metadata, "flow_load_report": load_report,
                            "best_target_match": best}
    write_json(run / "run_config.json", config)
    write_json(run / "status.json", {"status": "completed", "run_dir": str(run), "best": best})
    print(json.dumps({"run_dir": str(run), "status": "completed", "best": best}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
