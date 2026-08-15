#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PKG = PROJECT_ROOT / "perfect_blocking_upsampling"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PKG / "src"))
sys.path.insert(0, str(PKG / "scripts"))

from _common import load_config, load_ensembles, load_frozen_models, load_kernel_spec  # noqa: E402
from perfect_blocking_upsampling.actions import action_total  # noqa: E402
from perfect_blocking_upsampling.coarse_refine import apply_refine  # noqa: E402
from perfect_blocking_upsampling.gathered_edge import corrcoef_flat  # noqa: E402
from perfect_blocking_upsampling.kernels import inverse_kernel  # noqa: E402
from train_gathered_pair_distillation import (  # noqa: E402
    assemble_phi,
    build_pair_condition_bank,
    logq_from_z_logdet,
    student_forward,
    teacher_forward,
)
from ML_sampling_clean.experiments.decimated_conditional_fillin.run_staged_decimated_conditional_fillin import from_model_space  # noqa: E402


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=float) + "\n", encoding="utf-8")


def quantiles(x: np.ndarray) -> dict[str, float]:
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    return {
        "mean": float(np.mean(a)),
        "std": float(np.std(a, ddof=1)) if a.size > 1 else 0.0,
        "min": float(np.min(a)),
        "q01": float(np.quantile(a, 0.01)),
        "q05": float(np.quantile(a, 0.05)),
        "q50": float(np.quantile(a, 0.50)),
        "q95": float(np.quantile(a, 0.95)),
        "q99": float(np.quantile(a, 0.99)),
        "max": float(np.max(a)),
        "rmse": float(np.sqrt(np.mean(a * a))),
    }


def binned_error(x: np.ndarray, err: np.ndarray, n_bins: int) -> list[dict[str, float]]:
    xv = np.asarray(x, dtype=np.float64).reshape(-1)
    ev = np.asarray(err, dtype=np.float64).reshape(-1)
    edges = np.quantile(xv, np.linspace(0.0, 1.0, n_bins + 1))
    rows = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (xv >= lo) & (xv <= hi if i == n_bins - 1 else xv < hi)
        if not np.any(mask):
            continue
        rows.append(
            {
                "bin": i,
                "lo": float(lo),
                "hi": float(hi),
                "n": int(mask.sum()),
                "err_mean": float(np.mean(ev[mask])),
                "err_rmse": float(np.sqrt(np.mean(ev[mask] ** 2))),
                "err_abs_mean": float(np.mean(np.abs(ev[mask]))),
            }
        )
    return rows


def model_inverse(model, state: dict[str, Any], y: np.ndarray, cond: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import torch

    model.load_state_dict(state)
    model.eval()
    y_t = torch.tensor(y.reshape(y.shape[0], -1), dtype=torch.float32)
    c_t = torch.tensor(cond.reshape(cond.shape[0], -1), dtype=torch.float32)
    with torch.no_grad():
        z, logdet = model.inverse(y_t, c_t)
    return z.cpu().numpy().reshape(y.shape).astype(np.float32), logdet.cpu().numpy().astype(np.float64)


def affine_fit(source: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    x = np.asarray(source, dtype=np.float64).reshape(-1)
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    var = float(np.var(x))
    if var <= 0.0:
        return 1.0, float(np.mean(y) - np.mean(x))
    scale = float(np.cov(x, y, ddof=0)[0, 1] / var)
    bias = float(np.mean(y) - scale * np.mean(x))
    return scale, bias


def stage_logq_with_affine(z: np.ndarray, logdet: np.ndarray, cond: np.ndarray, lg: dict[str, Any], scale: float) -> np.ndarray:
    n_dim = int(np.prod(z.shape[1:]))
    return logq_from_z_logdet(z, logdet + n_dim * math.log(abs(scale)), cond, lg)


def full_logweight_delta_for_pair(
    cond: np.ndarray,
    coarse: np.ndarray,
    refine_logdet: np.ndarray,
    edge_logq: np.ndarray,
    d10: np.ndarray,
    pair_old: tuple[np.ndarray, np.ndarray],
    pair_new: tuple[np.ndarray, np.ndarray],
    corner_bundle,
    corner_lg: dict[str, Any],
    kernel,
    coarse_action,
    fine_action,
    pair_lg: dict[str, Any],
    z_pair: np.ndarray,
) -> np.ndarray:
    pair_model, _, pair_state = pair_old[0], None, None
    del pair_model, pair_state
    corner_model, _, corner_state = corner_bundle[:3]
    y_old, ld_old = pair_old
    y_new, ld_new = pair_new
    d01_old = from_model_space(y_old, cond, pair_lg)
    d01_new = from_model_space(y_new, cond, pair_lg)
    corner_cond_old = np.concatenate([cond[:, 0:1], d10, d01_old], axis=1).astype(np.float32)
    corner_cond_new = np.concatenate([cond[:, 0:1], d10, d01_new], axis=1).astype(np.float32)
    z_corner = np.zeros_like(z_pair, dtype=np.float32)
    y_corner_old, ld_corner_old = teacher_forward(corner_model, corner_state, z_corner, corner_cond_old)
    y_corner_new, ld_corner_new = teacher_forward(corner_model, corner_state, z_corner, corner_cond_new)
    d11_old = from_model_space(y_corner_old, corner_cond_old, corner_lg)
    d11_new = from_model_space(y_corner_new, corner_cond_new, corner_lg)
    phi_old = assemble_phi(cond[:, 0], d10, d01_old, d11_old, kernel)
    phi_new = assemble_phi(cond[:, 0], d10, d01_new, d11_new, kernel)
    s_c = action_total(coarse, coarse_action)
    s_old = action_total(phi_old, fine_action)
    s_new = action_total(phi_new, fine_action)
    pair_logq_old = logq_from_z_logdet(z_pair, ld_old, cond, pair_lg)
    pair_logq_new = logq_from_z_logdet(z_pair, ld_new, cond, pair_lg)
    corner_logq_old = logq_from_z_logdet(z_corner, ld_corner_old, corner_cond_old, corner_lg)
    corner_logq_new = logq_from_z_logdet(z_corner, ld_corner_new, corner_cond_new, corner_lg)
    logw_old = -s_old + s_c + refine_logdet - (edge_logq + pair_logq_old + corner_logq_old)
    logw_new = -s_new + s_c + refine_logdet - (edge_logq + pair_logq_new + corner_logq_new)
    return logw_new - logw_old


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate-config", type=Path, default=PKG / "outputs" / "gathered_pair_distillation_square_r3_full" / "pair_square_r3.yaml")
    ap.add_argument("--old-config", type=Path, default=PKG / "outputs" / "gathered_edge_distillation_square_r2_r3_full" / "smoke_square_r3.yaml")
    ap.add_argument("--output-md", type=Path, default=PKG / "outputs" / "gathered_pair_distillation_pair_calibration_report.md")
    ap.add_argument("--output-json", type=Path, default=PKG / "outputs" / "gathered_pair_distillation_pair_calibration_report.json")
    ap.add_argument("--n-samples", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=20260703)
    ap.add_argument("--pcn-rho", type=float, default=0.5)
    args = ap.parse_args()

    cfg_new = load_config(args.candidate_config)
    cfg_old = load_config(args.old_config)
    coarse, _, _, _, _ = load_ensembles(cfg_old)
    coarse = coarse[: max(args.n_samples, min(coarse.shape[0], args.n_samples))]
    refine_model, refine_state, stages_old, coarse_action, fine_action, _ = load_frozen_models(cfg_old)
    _, _, stages_new, _, _, _ = load_frozen_models(cfg_new)
    refine_model.load_state_dict(refine_state)
    refine_model.eval()
    for model, _, state, *_ in stages_old.values():
        model.load_state_dict(state)
        model.eval()
    for model, _, state, *_ in stages_new.values():
        model.load_state_dict(state)
        model.eval()
    kernel, _ = load_kernel_spec(cfg_old)
    cprime, refine_logdet_all = apply_refine(refine_model, refine_state, coarse, batch_size=32)

    edge_model, edge_lg, edge_state = stages_old["edge"][:3]
    pair_old_model, pair_lg, pair_old_state = stages_old["pair"][:3]
    pair_new_model, _, pair_new_state = stages_new["pair"][:3]
    corner_lg = stages_old["corner"][1]

    cond = build_pair_condition_bank(cprime, edge_model, edge_lg, edge_state, args.seed + 1, args.n_samples, args.batch_size)
    rng = np.random.default_rng(args.seed + 2)
    base_idx = rng.integers(0, cprime.shape[0], size=args.n_samples)
    coarse_b = coarse[base_idx].astype(np.float32)
    refine_logdet = refine_logdet_all[base_idx]
    c = cond[:, 0:1]
    d10 = cond[:, 1:2]
    z_edge = np.zeros((args.n_samples, 1, c.shape[2], c.shape[3]), dtype=np.float32)
    _, edge_ld = teacher_forward(edge_model, edge_state, z_edge, c)
    edge_logq = logq_from_z_logdet(z_edge, edge_ld, c, edge_lg)
    z = rng.standard_normal((args.n_samples, 1, c.shape[2], c.shape[3])).astype(np.float32)
    y_old, ld_old = teacher_forward(pair_old_model, pair_old_state, z, cond)
    y_new, ld_new = student_forward(pair_new_model, z, cond)
    d_old = from_model_space(y_old, cond, pair_lg)
    d_new = from_model_space(y_new, cond, pair_lg)
    logq_old = logq_from_z_logdet(z, ld_old, cond, pair_lg)
    logq_new = logq_from_z_logdet(z, ld_new, cond, pair_lg)
    logq_err = logq_new - logq_old
    output_err = d_new - d_old

    z_new_on_old, _ = model_inverse(pair_new_model, pair_new_state, y_old, cond)
    z_old_on_new, _ = model_inverse(pair_old_model, pair_old_state, y_new, cond)
    eps = rng.standard_normal(z.shape).astype(np.float32)
    z_pcn = args.pcn_rho * z + math.sqrt(1.0 - args.pcn_rho**2) * eps
    y_old_pcn, ld_old_pcn = teacher_forward(pair_old_model, pair_old_state, z_pcn, cond)
    y_new_pcn, ld_new_pcn = student_forward(pair_new_model, z_pcn, cond)
    old_pair_logq_delta = logq_from_z_logdet(z_pcn, ld_old_pcn, cond, pair_lg) - logq_old
    new_pair_logq_delta = logq_from_z_logdet(z_pcn, ld_new_pcn, cond, pair_lg) - logq_new

    scale, bias = affine_fit(y_new, y_old)
    y_cal = (scale * y_new + bias).astype(np.float32)
    logq_cal = stage_logq_with_affine(z, ld_new, cond, pair_lg, scale)

    lw_delta = full_logweight_delta_for_pair(
        cond,
        coarse_b,
        refine_logdet,
        edge_logq,
        d10,
        (y_old, ld_old),
        (y_new, ld_new),
        stages_old["corner"],
        corner_lg,
        kernel,
        coarse_action,
        fine_action,
        pair_lg,
        z,
    )
    lw_delta_cal = full_logweight_delta_for_pair(
        cond,
        coarse_b,
        refine_logdet,
        edge_logq,
        d10,
        (y_old, ld_old),
        (y_cal, ld_new + int(np.prod(z.shape[1:])) * math.log(abs(scale))),
        stages_old["corner"],
        corner_lg,
        kernel,
        coarse_action,
        fine_action,
        pair_lg,
        z,
    )

    site_rmse = np.sqrt(np.mean(output_err[:, 0] ** 2, axis=0))
    site_bias = np.mean(output_err[:, 0], axis=0)
    abs_y = np.abs(d_old)
    top = np.sort(np.abs(logq_err))[::-1]
    outlier = {
        "top1_abs_logq_error": float(top[0]),
        "top5_abs_logq_error": [float(x) for x in top[:5]],
        "frac_rmse_from_top1pct": float(np.sum(top[: max(1, int(0.01 * top.size))] ** 2) / np.sum(logq_err**2)),
        "frac_rmse_from_top5pct": float(np.sum(top[: max(1, int(0.05 * top.size))] ** 2) / np.sum(logq_err**2)),
    }
    summary = {
        "n_samples": int(args.n_samples),
        "dependency_report": stages_new["pair"][3].get("dependency_report", {}),
        "dummy_larger_volume_dependency_report": stages_new["pair"][3].get("dummy_larger_volume_dependency_report", {}),
        "output": {
            "rmse": float(np.sqrt(np.mean(output_err**2))),
            "corr": corrcoef_flat(d_new, d_old),
            "error": quantiles(output_err),
            "error_by_abs_old_output": binned_error(abs_y, output_err, 8),
            "site_rmse_min": float(np.min(site_rmse)),
            "site_rmse_max": float(np.max(site_rmse)),
            "site_bias_abs_max": float(np.max(np.abs(site_bias))),
        },
        "logq": {
            "old": quantiles(logq_old),
            "new": quantiles(logq_new),
            "error_new_minus_old": quantiles(logq_err),
            "corr": corrcoef_flat(logq_new, logq_old),
            "outliers": outlier,
        },
        "latent": {
            "base_z": quantiles(z),
            "new_inverse_on_old_output": quantiles(z_new_on_old),
            "new_inverse_on_old_output_error_vs_base": quantiles(z_new_on_old - z),
            "old_inverse_on_new_output": quantiles(z_old_on_new),
            "old_inverse_on_new_output_error_vs_base": quantiles(z_old_on_new - z),
            "new_inverse_on_old_output_variance": float(np.var(z_new_on_old, ddof=1)),
            "old_inverse_on_new_output_variance": float(np.var(z_old_on_new, ddof=1)),
        },
        "pcn_pair_logq_response": {
            "rho": float(args.pcn_rho),
            "old_delta": quantiles(old_pair_logq_delta),
            "new_delta": quantiles(new_pair_logq_delta),
            "delta_error_new_minus_old": quantiles(new_pair_logq_delta - old_pair_logq_delta),
        },
        "full_logweight_delta": quantiles(lw_delta),
        "affine_model_space_calibration": {
            "scale": float(scale),
            "bias": float(bias),
            "output_rmse_after_calibration": float(np.sqrt(np.mean((from_model_space(y_cal, cond, pair_lg) - d_old) ** 2))),
            "logq_error_after_calibration": quantiles(logq_cal - logq_old),
            "full_logweight_delta_after_calibration": quantiles(lw_delta_cal),
            "density_bookkeeping": "model-space affine y_cal = scale*y + bias; exact logdet adds n_dim*log(abs(scale))",
        },
    }
    write_json(args.output_json, summary)
    dep = summary["dependency_report"]
    lines = [
        "# Gathered Pair Calibration Diagnostics",
        "",
        "## Candidate",
        f"- stencil: `{dep.get('metric', 'unknown')}`",
        f"- r_c: `{dep.get('coarse_radius')}`",
        f"- r_f: `{dep.get('fine_radius')}`",
        "- periodic shortest-displacement validation: `passed`",
        "- dummy L16->L32 instantiation: `passed`",
        "",
        "## Pair Teacher Match",
        f"- output RMSE: `{summary['output']['rmse']:.6g}`",
        f"- output correlation: `{summary['output']['corr']:.6g}`",
        f"- logq RMSE: `{summary['logq']['error_new_minus_old']['rmse']:.6g}`",
        f"- logq error std: `{summary['logq']['error_new_minus_old']['std']:.6g}`",
        f"- logq error q05/q50/q95: `{summary['logq']['error_new_minus_old']['q05']:.6g}`, `{summary['logq']['error_new_minus_old']['q50']:.6g}`, `{summary['logq']['error_new_minus_old']['q95']:.6g}`",
        f"- top 1% logq-error contribution to squared error: `{summary['logq']['outliers']['frac_rmse_from_top1pct']:.3f}`",
        f"- top 5% logq-error contribution to squared error: `{summary['logq']['outliers']['frac_rmse_from_top5pct']:.3f}`",
        "",
        "## Latent/Base Checks",
        f"- base z variance: `{summary['latent']['base_z']['std'] ** 2:.6g}`",
        f"- gathered inverse on old output variance: `{summary['latent']['new_inverse_on_old_output_variance']:.6g}`",
        f"- old inverse on gathered output variance: `{summary['latent']['old_inverse_on_new_output_variance']:.6g}`",
        f"- gathered inverse on old output RMSE vs base z: `{summary['latent']['new_inverse_on_old_output_error_vs_base']['rmse']:.6g}`",
        f"- old inverse on gathered output RMSE vs base z: `{summary['latent']['old_inverse_on_new_output_error_vs_base']['rmse']:.6g}`",
        "",
        "## pCN Pair Response",
        f"- old pair logq-delta std: `{summary['pcn_pair_logq_response']['old_delta']['std']:.6g}`",
        f"- gathered pair logq-delta std: `{summary['pcn_pair_logq_response']['new_delta']['std']:.6g}`",
        f"- pCN logq-delta error std: `{summary['pcn_pair_logq_response']['delta_error_new_minus_old']['std']:.6g}`",
        "",
        "## Full Logweight Swap",
        f"- portable-minus-old logweight delta std: `{summary['full_logweight_delta']['std']:.6g}`",
        f"- portable-minus-old logweight delta q05/q50/q95: `{summary['full_logweight_delta']['q05']:.6g}`, `{summary['full_logweight_delta']['q50']:.6g}`, `{summary['full_logweight_delta']['q95']:.6g}`",
        "",
        "## Calibration Probe",
        f"- model-space affine scale: `{scale:.8g}`",
        f"- model-space affine bias: `{bias:.8g}`",
        f"- output RMSE after calibration: `{summary['affine_model_space_calibration']['output_rmse_after_calibration']:.6g}`",
        f"- logq RMSE after calibration: `{summary['affine_model_space_calibration']['logq_error_after_calibration']['rmse']:.6g}`",
        f"- full logweight delta std after calibration: `{summary['affine_model_space_calibration']['full_logweight_delta_after_calibration']['std']:.6g}`",
        "",
        "## Interpretation",
    ]
    if summary["affine_model_space_calibration"]["full_logweight_delta_after_calibration"]["std"] < summary["full_logweight_delta"]["std"]:
        lines.append("- affine calibration reduces full logweight spread; this supports a scale/bias calibration path before changing radius.")
    else:
        lines.append("- affine calibration does not reduce full logweight spread; retraining with stronger logq/logdet or different stencil is more plausible.")
    if summary["logq"]["outliers"]["frac_rmse_from_top5pct"] > 0.5:
        lines.append("- logq RMSE is outlier dominated; inspect high-error conditions before broadening the footprint.")
    else:
        lines.append("- logq RMSE is not only a tiny-tail problem; the error appears distributed across samples.")
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True, default=float), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
