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
from train_gathered_pair_distillation import logq_from_z_logdet, student_forward, teacher_forward  # noqa: E402
from ML_sampling_clean.experiments.decimated_conditional_fillin.run_staged_decimated_conditional_fillin import from_model_space  # noqa: E402


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=float) + "\n", encoding="utf-8")


def quantiles(x: np.ndarray) -> dict[str, float]:
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    return {
        "mean": float(np.mean(a)),
        "std": float(np.std(a, ddof=1)) if a.size > 1 else 0.0,
        "rmse": float(np.sqrt(np.mean(a * a))),
        "min": float(np.min(a)),
        "q05": float(np.quantile(a, 0.05)),
        "q50": float(np.quantile(a, 0.50)),
        "q95": float(np.quantile(a, 0.95)),
        "max": float(np.max(a)),
    }


def corr(a: np.ndarray, b: np.ndarray) -> float:
    return corrcoef_flat(np.asarray(a), np.asarray(b))


def assemble_phi(cprime: np.ndarray, d10: np.ndarray, d01: np.ndarray, d11: np.ndarray, kernel) -> np.ndarray:
    psi = np.empty((cprime.shape[0], 2 * cprime.shape[1], 2 * cprime.shape[2]), dtype=np.float32)
    psi[:, 0::2, 0::2] = cprime
    psi[:, 1::2, 0::2] = d10[:, 0]
    psi[:, 0::2, 1::2] = d01[:, 0]
    psi[:, 1::2, 1::2] = d11[:, 0]
    phi, _ = inverse_kernel(psi, kernel)
    return phi


def stage_sample_from_z(model, state: dict[str, Any], z: np.ndarray, cond: np.ndarray, lg: dict[str, Any], portable: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if portable:
        y, ld = student_forward(model, z, cond)
    else:
        y, ld = teacher_forward(model, state, z, cond)
    x = from_model_space(y, cond, lg)
    logq = logq_from_z_logdet(z, ld, cond, lg)
    return y, x, ld, logq


def load_all(old_config: Path, candidate_config: Path):
    cfg_old = load_config(old_config)
    cfg_candidate = load_config(candidate_config)
    coarse, _, _, _, _ = load_ensembles(cfg_old)
    refine_model, refine_state, stages_old, coarse_action, fine_action, _ = load_frozen_models(cfg_old)
    _, _, stages_candidate, _, _, _ = load_frozen_models(cfg_candidate)
    refine_model.load_state_dict(refine_state)
    refine_model.eval()
    for model, _, state, *_ in stages_old.values():
        model.load_state_dict(state)
        model.eval()
    for model, _, state, *_ in stages_candidate.values():
        model.load_state_dict(state)
        model.eval()
    kernel, _ = load_kernel_spec(cfg_old)
    return cfg_old, cfg_candidate, coarse, refine_model, refine_state, stages_old, stages_candidate, coarse_action, fine_action, kernel


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-config", type=Path, default=PKG / "outputs" / "gathered_pair_structure_diagnostics" / "old_pair_procedural_masks.yaml")
    ap.add_argument("--candidate-config", type=Path, default=PKG / "outputs" / "procedural_corner_diagnostics" / "old_pair_corner_procedural_masks.yaml")
    ap.add_argument("--output-dir", type=Path, default=PKG / "outputs" / "procedural_corner_diagnostics" / "corner_audit")
    ap.add_argument("--n-samples", type=int, default=512)
    ap.add_argument("--seed", type=int, default=20260708)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cfg_old, cfg_candidate, coarse_all, refine_model, refine_state, stages_old, stages_candidate, coarse_action, fine_action, kernel = load_all(args.old_config, args.candidate_config)
    coarse = coarse_all[: args.n_samples].astype(np.float32)
    cprime, refine_logdet = apply_refine(refine_model, refine_state, coarse, batch_size=32)
    del refine_logdet
    rng = np.random.default_rng(args.seed)
    z_edge = rng.standard_normal((args.n_samples, 1, cprime.shape[1], cprime.shape[2])).astype(np.float32)
    z_pair = rng.standard_normal(z_edge.shape).astype(np.float32)
    z_corner = rng.standard_normal(z_edge.shape).astype(np.float32)

    edge_model, edge_lg, edge_state = stages_old["edge"][:3]
    pair_model, pair_lg, pair_state = stages_old["pair"][:3]
    corner_old_model, corner_lg, corner_old_state, corner_old_ckpt = stages_old["corner"][:4]
    corner_new_model, corner_new_lg, corner_new_state, corner_new_ckpt = stages_candidate["corner"][:4]

    _, d10, _, _ = stage_sample_from_z(edge_model, edge_state, z_edge, cprime[:, None].astype(np.float32), edge_lg)
    pair_cond = np.concatenate([cprime[:, None], d10], axis=1).astype(np.float32)
    _, d01, _, _ = stage_sample_from_z(pair_model, pair_state, z_pair, pair_cond, pair_lg)
    corner_cond = np.concatenate([cprime[:, None], d10, d01], axis=1).astype(np.float32)

    y_old, d11_old, ld_old, logq_old = stage_sample_from_z(corner_old_model, corner_old_state, z_corner, corner_cond, corner_lg)
    y_new, d11_new, ld_new, logq_new = stage_sample_from_z(corner_new_model, corner_new_state, z_corner, corner_cond, corner_new_lg, portable=True)

    phi_old = assemble_phi(cprime, d10, d01, d11_old, kernel)
    phi_new = assemble_phi(cprime, d10, d01, d11_new, kernel)
    s_old = action_total(phi_old, fine_action)
    s_new = action_total(phi_new, fine_action)
    delta_s = s_new - s_old
    delta_logq = logq_new - logq_old
    delta_logw = -delta_s - delta_logq
    output_delta = d11_new - d11_old
    phi_delta = phi_new - phi_old

    dep = corner_new_ckpt.get("dependency_report", {})
    dummy = corner_new_ckpt.get("dummy_larger_volume_dependency_report", {})
    summary = {
        "old_config": str(args.old_config),
        "candidate_config": str(args.candidate_config),
        "n_samples": int(args.n_samples),
        "corner_teacher_match": {
            "output_rmse": float(np.sqrt(np.mean(output_delta**2))),
            "output_corr": corr(d11_new, d11_old),
            "model_space_rmse": float(np.sqrt(np.mean((y_new - y_old) ** 2))),
            "logdet_rmse": float(np.sqrt(np.mean((ld_new - ld_old) ** 2))),
            "logq_rmse": float(np.sqrt(np.mean((logq_new - logq_old) ** 2))),
        },
        "reconstruction": {
            "phi_rmse": float(np.sqrt(np.mean(phi_delta**2))),
            "delta_S": quantiles(delta_s),
            "delta_logq": quantiles(delta_logq),
            "full_swap_delta_logw": quantiles(delta_logw),
        },
        "dependency_report": dep,
        "dummy_larger_volume_dependency_report": dummy,
        "convention": {
            "old_corner_flow_arch": corner_old_ckpt.get("config", {}).get("flow_arch", "legacy_stage_model"),
            "portable_corner_flow_arch": corner_new_ckpt.get("config", {}).get("flow_arch"),
            "conditioning_order": ["cprime", "accepted_edge_d10", "accepted_pair_d01"],
            "logweight_delta_formula": "-delta_S - delta_logq",
        },
    }
    write_json(args.output_dir / "corner_equivalence_audit_report.json", summary)
    lines = [
        "# Corner/Body Procedural-Mask Equivalence Audit",
        "",
        f"- output RMSE: `{summary['corner_teacher_match']['output_rmse']:.6g}`",
        f"- output correlation: `{summary['corner_teacher_match']['output_corr']:.6g}`",
        f"- logq RMSE: `{summary['corner_teacher_match']['logq_rmse']:.6g}`",
        f"- reconstructed phi RMSE: `{summary['reconstruction']['phi_rmse']:.6g}`",
        f"- delta S std: `{summary['reconstruction']['delta_S']['std']:.6g}`",
        f"- delta logq std: `{summary['reconstruction']['delta_logq']['std']:.6g}`",
        f"- full swap delta logw std: `{summary['reconstruction']['full_swap_delta_logw']['std']:.6g}`",
        f"- dependency r_c/r_f: `{dep.get('coarse_radius')}` / `{dep.get('fine_radius')}`",
        f"- dependency metric: `{dep.get('metric')}`",
        f"- dummy L16->L32 instantiation: `{'passed' if dummy else 'not_reported'}`",
        "",
        "This is a shape-parametric portability audit. It is not a strict finite-footprint proof for the old stacked circular-conv corner/body architecture.",
    ]
    (args.output_dir / "corner_equivalence_audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, default=float), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
