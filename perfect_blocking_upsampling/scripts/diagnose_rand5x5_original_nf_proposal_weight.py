#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PKG = PROJECT_ROOT / "perfect_blocking_upsampling"
SCRIPT_DIR = PKG / "scripts"
sys.path.insert(0, str(PKG / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

from perfect_blocking_upsampling.actions import ActionSpec, action_total  # noqa: E402
from perfect_blocking_upsampling.kernels import inverse_kernel, kernel_fft, kernel_stencil_from_spec, load_kernel, normalize_kernel  # noqa: E402
from run_lam0p2_conditional_gaussian_residual_L16 import residual_logdet  # noqa: E402
from run_lam0p2_residual_flow_patch_chain import (  # noqa: E402
    assemble_psi_generic,
    ar_condition,
    inverse_transform_residual_generic,
    load_initializer,
    predict_detail_generic,
    read_phi,
)

LAM = 0.2
KAPPA = 0.323124
DEFAULT_OUT = PKG / "outputs" / "controlled_patch_lam0p2" / "rand5x5_0084_original_nf_weight_diagnostic"
DEFAULT_COARSE = PKG / "outputs" / "lam0p2_kappa0p323124" / "native" / "L16" / "configs.npz"
DEFAULT_KERNEL = PKG / "outputs" / "controlled_patch_lam0p2" / "tail_aware_kernel_search_L16to32" / "rand5x5_0084_kernel.json"
DEFAULT_BASELINE = PKG / "outputs" / "controlled_patch_lam0p2" / "rand5x5_0084_residual_flow_pilot" / "checkpoints" / "checkpoint_best_residual_flow.pt"
DEFAULT_BEST = PKG / "outputs" / "controlled_patch_lam0p2" / "rand5x5_0084_tail_aware_ar_flow_pilot" / "width0p03_tail0p03" / "autoregressive_checkpoint.pt"


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=True, default=json_default) + "\n", encoding="utf-8")


def json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(type(obj).__name__)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def obs_per_config(phi: np.ndarray, action: ActionSpec) -> dict[str, np.ndarray]:
    arr = np.asarray(phi, dtype=np.float64)
    m = arr.mean(axis=(1, 2))
    m2 = m * m
    m4 = m2 * m2
    phi2 = np.mean(arr * arr, axis=(1, 2))
    phi4 = np.mean(arr**4, axis=(1, 2))
    nn = 0.5 * (
        np.mean(arr * np.roll(arr, -1, axis=1), axis=(1, 2))
        + np.mean(arr * np.roll(arr, -1, axis=2), axis=(1, 2))
    )
    twonn = 0.5 * (
        np.mean(arr * np.roll(arr, -2, axis=1), axis=(1, 2))
        + np.mean(arr * np.roll(arr, -2, axis=2), axis=(1, 2))
    )
    diag = np.mean(arr * np.roll(np.roll(arr, -1, axis=1), -1, axis=2), axis=(1, 2))
    return {
        "m": m,
        "m2": m2,
        "m4": m4,
        "phi2": phi2,
        "phi4": phi4,
        "NN": nn,
        "2nn": twonn,
        "diag": diag,
        "action_density": action_total(arr, action) / float(arr.shape[1] * arr.shape[2]),
    }


def logsumexp(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    m = float(np.max(x))
    return m + float(np.log(np.sum(np.exp(x - m))))


def ess_over_n(logw: np.ndarray) -> float:
    n = len(logw)
    if n == 0:
        return float("nan")
    return float(np.exp(2.0 * logsumexp(logw) - logsumexp(2.0 * logw) - math.log(n)))


def constrained_inverse_kernel_logjac(kernel: Any, fine_l: int) -> float:
    """Volume Jacobian from psi_detail coordinates to phi on fixed-psi_ee manifold.

    The map is rectangular: 3*(L/2)^2 detail coordinates -> L^2 fine
    coordinates.  The induced volume factor is sqrt(det(B^T B)), where
    B applies K^{-1} to a detail-basis vector with ee held fixed.
    """
    lc = fine_l // 2
    n_detail = 3 * lc * lc
    cols = np.empty((fine_l * fine_l, n_detail), dtype=np.float64)
    col = 0
    for sub in range(3):
        for x in range(lc):
            for y in range(lc):
                psi = np.zeros((1, fine_l, fine_l), dtype=np.float32)
                if sub == 0:
                    psi[0, 0::2, 1::2][x, y] = 1.0
                elif sub == 1:
                    psi[0, 1::2, 0::2][x, y] = 1.0
                else:
                    psi[0, 1::2, 1::2][x, y] = 1.0
                phi, _ = inverse_kernel(psi, kernel)
                cols[:, col] = phi.reshape(-1).astype(np.float64)
                col += 1
    singular_values = np.linalg.svd(cols, full_matrices=False, compute_uv=False)
    return float(np.sum(np.log(np.maximum(singular_values, 1.0e-300))))


def full_inverse_kernel_logdet(kernel: Any, fine_l: int) -> float:
    stencil = normalize_kernel(kernel_stencil_from_spec(kernel))
    kt = kernel_fft(stencil, fine_l, kernel.eta)
    return -float(np.sum(np.log(np.maximum(np.abs(kt), 1.0e-300))))


def sample_with_components(models: list[torch.nn.Module], predictor: dict[str, Any], stats: dict[str, Any], ee: np.ndarray, args: argparse.Namespace) -> dict[str, np.ndarray]:
    device = torch.device(args.device)
    lc = int(ee.shape[1])
    y_batches = []
    logpz_batches = []
    flow_logdet_batches = []
    z_norm2_batches = []
    torch.manual_seed(args.seed + 101)
    with torch.no_grad():
        for start in range(0, len(ee), args.batch_size):
            cond = torch.from_numpy(ee[start : start + args.batch_size].reshape(-1, lc * lc)).to(device)
            yb = torch.zeros((cond.shape[0], 3, lc, lc), dtype=cond.dtype, device=device)
            logpz = torch.zeros(cond.shape[0], dtype=cond.dtype, device=device)
            logdet = torch.zeros(cond.shape[0], dtype=cond.dtype, device=device)
            z_norm2 = torch.zeros(cond.shape[0], dtype=cond.dtype, device=device)
            for stage, model in enumerate(models):
                c = ar_condition(cond, yb, stage, lc)
                z = torch.randn((cond.shape[0], lc * lc), dtype=cond.dtype, device=device)
                sample, stage_logdet = model.forward(z, c)
                logpz = logpz - 0.5 * (z * z + math.log(2.0 * math.pi)).sum(dim=1)
                logdet = logdet + stage_logdet
                z_norm2 = z_norm2 + (z * z).sum(dim=1)
                yb[:, stage] = sample.reshape(cond.shape[0], lc, lc)
            y_batches.append(yb.detach().cpu().numpy().astype(np.float32))
            logpz_batches.append(logpz.detach().cpu().numpy().astype(np.float64))
            flow_logdet_batches.append(logdet.detach().cpu().numpy().astype(np.float64))
            z_norm2_batches.append(z_norm2.detach().cpu().numpy().astype(np.float64))
    y = np.concatenate(y_batches, axis=0)
    pred = predict_detail_generic(predictor, ee)
    detail = (pred + inverse_transform_residual_generic(y, stats)).astype(np.float32)
    psi = assemble_psi_generic(ee, detail)
    return {
        "y": y,
        "detail": detail,
        "psi": psi,
        "logpz": np.concatenate(logpz_batches),
        "flow_logdet": np.concatenate(flow_logdet_batches),
        "z_norm2": np.concatenate(z_norm2_batches),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coarse-ensemble", type=Path, default=DEFAULT_COARSE)
    ap.add_argument("--kernel-path", type=Path, default=DEFAULT_KERNEL)
    ap.add_argument("--baseline-checkpoint", type=Path, default=DEFAULT_BASELINE)
    ap.add_argument("--best-checkpoint", type=Path, default=DEFAULT_BEST)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--configs", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=2026070621)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-coupling-layers", type=int, default=8)
    ap.add_argument("--hidden-channels", type=int, default=32)
    ap.add_argument("--conv-kernel-size", type=int, default=5)
    ap.add_argument("--log-scale-bound", type=float, default=0.75)
    ap.add_argument("--skip-constrained-kernel-jacobian", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    # Reuse the existing initializer loader.  These attributes are consumed by it.
    args.initializer_kind = "ar"
    args.from_L = 16
    action_c = ActionSpec("phi4_nn", LAM, KAPPA)
    action_f = ActionSpec("phi4_nn", LAM, KAPPA)
    kernel, kernel_json = load_kernel(args.kernel_path)
    coarse_all = read_phi(args.coarse_ensemble, 16)
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(coarse_all), size=args.configs, replace=False)
    coarse = coarse_all[idx].astype(np.float32)
    models, predictor, stats = load_initializer(args)
    comp = sample_with_components(models, predictor, stats, coarse, args)
    phi_f, inv_info = inverse_kernel(comp["psi"], kernel)

    s_f = action_total(phi_f, action_f).astype(np.float64)
    s_c = action_total(coarse, action_c).astype(np.float64)
    obs = obs_per_config(phi_f, action_f)
    residual_jac = residual_logdet(stats)
    flow_logdet = comp["flow_logdet"]
    constrained_k_jac = 0.0 if args.skip_constrained_kernel_jacobian else constrained_inverse_kernel_logjac(kernel, 32)
    full_k_inv_logdet = full_inverse_kernel_logdet(kernel, 32)
    logabsdet_total = flow_logdet + residual_jac + constrained_k_jac
    logw = -s_f + s_c - comp["logpz"] + logabsdet_total

    rows = []
    for i in range(args.configs):
        rows.append(
            {
                "config_id": i,
                "source_native_L16_index": int(idx[i]),
                "S_f": float(s_f[i]),
                "S_c": float(s_c[i]),
                "logpz": float(comp["logpz"][i]),
                "logabsdetJ": float(logabsdet_total[i]),
                "logabsdetJ_flow_only": float(flow_logdet[i]),
                "logabsdetJ_residual_unwhiten": float(residual_jac),
                "logabsdetJ_Kinv_constrained": float(constrained_k_jac),
                "logabsdetJ_Kinv_full_constant": float(full_k_inv_logdet),
                "logw": float(logw[i]),
                "z_norm2": float(comp["z_norm2"][i]),
                "action_density_f": float(obs["action_density"][i]),
                "phi2": float(obs["phi2"][i]),
                "phi4": float(obs["phi4"][i]),
                "NN": float(obs["NN"][i]),
                "2nn": float(obs["2nn"][i]),
                "diag": float(obs["diag"][i]),
                "m": float(obs["m"][i]),
                "m2": float(obs["m2"][i]),
                "m4": float(obs["m4"][i]),
                "binder_contribution_m2": float(obs["m2"][i]),
                "binder_contribution_m4": float(obs["m4"][i]),
            }
        )
    per_config = args.out_dir / "rand5x5_original_nf_proposal_weights_per_config.csv"
    write_csv(per_config, rows)

    summary = {
        "status": "completed",
        "formula": "logw = -S_f(phi_f) + S_c(phi_c) - log p_z(z) + logabsdetJ, up to additive constant",
        "note": "logabsdetJ includes AR flow forward logdet, residual unwhitening logdet, and the constant constrained K^{-1} induced volume factor. The full K^{-1} determinant is reported separately but not used.",
        "configs": int(args.configs),
        "coarse_ensemble": str(args.coarse_ensemble),
        "kernel_path": str(args.kernel_path),
        "kernel": kernel_json,
        "baseline_checkpoint": str(args.baseline_checkpoint),
        "best_checkpoint": str(args.best_checkpoint),
        "source_native_L16_indices": idx.tolist(),
        "per_config_csv": str(per_config),
        "S_f": stats_dict(s_f),
        "S_c": stats_dict(s_c),
        "logpz": stats_dict(comp["logpz"]),
        "logabsdetJ": stats_dict(logabsdet_total),
        "logabsdetJ_flow_only": stats_dict(flow_logdet),
        "logabsdetJ_residual_unwhiten": float(residual_jac),
        "logabsdetJ_Kinv_constrained": float(constrained_k_jac),
        "logabsdetJ_Kinv_full_constant": float(full_k_inv_logdet),
        "logw": stats_dict(logw),
        "ESS_over_N": ess_over_n(logw),
        "max_logw_minus_median": float(np.max(logw) - np.median(logw)),
        "inverse_kernel_info": inv_info,
        "observables_mean": {k: float(np.mean(v)) for k, v in obs.items()},
        "nonfinite_count": int(
            (~np.isfinite(phi_f)).sum()
            + (~np.isfinite(comp["psi"])).sum()
            + (~np.isfinite(logw)).sum()
            + (~np.isfinite(comp["logpz"])).sum()
            + (~np.isfinite(logabsdet_total)).sum()
        ),
    }
    write_json(args.out_dir / "rand5x5_original_nf_proposal_weight_summary.json", summary)

    lines = [
        "# rand5x5_0084 Original NF Proposal Weight Diagnostic",
        "",
        f"- configs: `{args.configs}`",
        f"- coarse ensemble: `{args.coarse_ensemble}`",
        f"- per-config CSV: `{per_config}`",
        f"- logw mean/std/min/max: `{summary['logw']['mean']:.6g}` / `{summary['logw']['std']:.6g}` / `{summary['logw']['min']:.6g}` / `{summary['logw']['max']:.6g}`",
        f"- ESS/N: `{summary['ESS_over_N']:.6g}`",
        f"- max logw - median: `{summary['max_logw_minus_median']:.6g}`",
        f"- nonfinite count: `{summary['nonfinite_count']}`",
        "",
        "The constrained `K^{-1}` Jacobian is constant across configurations, so it shifts log weights but does not affect ESS.",
    ]
    (args.out_dir / "rand5x5_original_nf_proposal_weight_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "completed", "out_dir": str(args.out_dir), "ESS_over_N": summary["ESS_over_N"], "logw_std": summary["logw"]["std"]}, indent=2), flush=True)
    return 0


def stats_dict(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


if __name__ == "__main__":
    raise SystemExit(main())
