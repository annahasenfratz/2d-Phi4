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
from perfect_blocking_upsampling.actions import action_density, action_total  # noqa: E402
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
        "q01": float(np.quantile(a, 0.01)),
        "q05": float(np.quantile(a, 0.05)),
        "q50": float(np.quantile(a, 0.50)),
        "q95": float(np.quantile(a, 0.95)),
        "q99": float(np.quantile(a, 0.99)),
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


def load_all(old_config: Path, pair_config: Path):
    cfg_old = load_config(old_config)
    cfg_pair = load_config(pair_config)
    coarse, _, _, _, _ = load_ensembles(cfg_old)
    refine_model, refine_state, stages_old, coarse_action, fine_action, _ = load_frozen_models(cfg_old)
    _, _, stages_pair, _, _, _ = load_frozen_models(cfg_pair)
    refine_model.load_state_dict(refine_state)
    refine_model.eval()
    for model, _, state, *_ in stages_old.values():
        model.load_state_dict(state)
        model.eval()
    for model, _, state, *_ in stages_pair.values():
        model.load_state_dict(state)
        model.eval()
    kernel, _ = load_kernel_spec(cfg_old)
    return cfg_old, cfg_pair, coarse, refine_model, refine_state, stages_old, stages_pair, coarse_action, fine_action, kernel


def parity_stats(arr: np.ndarray) -> dict[str, dict[str, float]]:
    # arr is (N, H, W), report checkerboard parities on stage lattice.
    h, w = arr.shape[-2:]
    yy, xx = np.indices((h, w))
    out = {}
    for p in [0, 1]:
        mask = ((xx + yy) % 2) == p
        out[f"parity_{p}"] = quantiles(arr[:, mask])
    return out


def binned_by_amplitude(amp: np.ndarray, val: np.ndarray, n_bins: int = 8) -> list[dict[str, float]]:
    a = np.asarray(amp, dtype=np.float64).reshape(-1)
    v = np.asarray(val, dtype=np.float64).reshape(-1)
    edges = np.quantile(a, np.linspace(0.0, 1.0, n_bins + 1))
    rows = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (a >= lo) & (a <= hi if i == n_bins - 1 else a < hi)
        if not np.any(mask):
            continue
        rows.append({"bin": i, "lo": float(lo), "hi": float(hi), "n": int(mask.sum()), **quantiles(v[mask])})
    return rows


def make_state(
    cprime: np.ndarray,
    coarse: np.ndarray,
    refine_logdet: np.ndarray,
    d10: np.ndarray,
    edge_logq: np.ndarray,
    d01: np.ndarray,
    pair_logq: np.ndarray,
    corner_model,
    corner_state: dict[str, Any],
    corner_lg: dict[str, Any],
    z_corner: np.ndarray,
    kernel,
    coarse_action,
    fine_action,
) -> dict[str, np.ndarray]:
    corner_cond = np.concatenate([cprime[:, None], d10, d01], axis=1).astype(np.float32)
    y11, d11, ld11, corner_logq = stage_sample_from_z(corner_model, corner_state, z_corner, corner_cond, corner_lg)
    phi = assemble_phi(cprime, d10, d01, d11, kernel)
    sf = action_total(phi, fine_action)
    sc = action_total(coarse, coarse_action)
    logw = -sf + sc + refine_logdet - (edge_logq + pair_logq + corner_logq)
    return {
        "corner_cond": corner_cond,
        "y11": y11,
        "d11": d11,
        "corner_logq": corner_logq,
        "phi": phi,
        "sf": sf,
        "sc": sc,
        "logw": logw,
        "action_density": action_density(phi, fine_action),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-config", type=Path, default=PKG / "outputs" / "gathered_edge_distillation_square_r2_r3_full" / "smoke_square_r3.yaml")
    ap.add_argument("--pair-config", type=Path, default=PKG / "outputs" / "gathered_pair_distillation_square_r3_logdet0p01" / "pair_square_r3.yaml")
    ap.add_argument("--output-dir", type=Path, default=PKG / "outputs" / "gathered_pair_equivalence_audit")
    ap.add_argument("--n-samples", type=int, default=512)
    ap.add_argument("--seed", type=int, default=20260706)
    ap.add_argument("--pcn-rho", type=float, default=0.5)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cfg_old, cfg_pair, coarse_all, refine_model, refine_state, stages_old, stages_pair, coarse_action, fine_action, kernel = load_all(args.old_config, args.pair_config)
    coarse = coarse_all[: args.n_samples].astype(np.float32)
    cprime, refine_logdet = apply_refine(refine_model, refine_state, coarse, batch_size=32)
    rng = np.random.default_rng(args.seed)
    z_edge = rng.standard_normal((args.n_samples, 1, cprime.shape[1], cprime.shape[2])).astype(np.float32)
    z_corner = rng.standard_normal(z_edge.shape).astype(np.float32)

    edge_model, edge_lg, edge_state = stages_old["edge"][:3]
    pair_old_model, pair_lg, pair_old_state, pair_old_ckpt = stages_old["pair"][:4]
    pair_new_model, pair_new_lg, pair_new_state, pair_new_ckpt = stages_pair["pair"][:4]
    corner_model, corner_lg, corner_state = stages_old["corner"][:3]

    y_edge, d10, edge_ld, edge_logq = stage_sample_from_z(edge_model, edge_state, z_edge, cprime[:, None].astype(np.float32), edge_lg)
    pair_cond = np.concatenate([cprime[:, None], d10], axis=1).astype(np.float32)
    z_pair = rng.standard_normal(z_edge.shape).astype(np.float32)
    y_old, d01_old, ld_old, logq_old = stage_sample_from_z(pair_old_model, pair_old_state, z_pair, pair_cond, pair_lg)
    y_new, d01_new, ld_new, logq_new = stage_sample_from_z(pair_new_model, pair_new_state, z_pair, pair_cond, pair_new_lg, portable=True)

    state_a = make_state(cprime, coarse, refine_logdet, d10, edge_logq, d01_old, logq_old, corner_model, corner_state, corner_lg, z_corner, kernel, coarse_action, fine_action)
    state_b = make_state(cprime, coarse, refine_logdet, d10, edge_logq, d01_new, logq_new, corner_model, corner_state, corner_lg, z_corner, kernel, coarse_action, fine_action)
    state_c = make_state(cprime, coarse, refine_logdet, d10, edge_logq, d01_old, logq_new, corner_model, corner_state, corner_lg, z_corner, kernel, coarse_action, fine_action)
    state_d = make_state(cprime, coarse, refine_logdet, d10, edge_logq, d01_new, logq_old, corner_model, corner_state, corner_lg, z_corner, kernel, coarse_action, fine_action)

    delta_s = state_b["sf"] - state_a["sf"]
    delta_pair_logq = logq_new - logq_old
    delta_corner_logq = state_b["corner_logq"] - state_a["corner_logq"]
    delta_logq = delta_pair_logq + delta_corner_logq
    delta_logw = state_b["logw"] - state_a["logw"]
    recon_delta = state_b["phi"] - state_a["phi"]
    action_density_delta = state_b["action_density"] - state_a["action_density"]
    pair_output_delta = d01_new - d01_old

    # Pair-only pCN: keep cprime/edge/corner latent fixed; move only pair z.
    eps = rng.standard_normal(z_pair.shape).astype(np.float32)
    z_pair_pcn = args.pcn_rho * z_pair + math.sqrt(1.0 - args.pcn_rho**2) * eps
    _, d01_old_pcn, _, logq_old_pcn = stage_sample_from_z(pair_old_model, pair_old_state, z_pair_pcn, pair_cond, pair_lg)
    _, d01_new_pcn, _, logq_new_pcn = stage_sample_from_z(pair_new_model, pair_new_state, z_pair_pcn, pair_cond, pair_new_lg, portable=True)
    old_pcn = make_state(cprime, coarse, refine_logdet, d10, edge_logq, d01_old_pcn, logq_old_pcn, corner_model, corner_state, corner_lg, z_corner, kernel, coarse_action, fine_action)
    new_pcn = make_state(cprime, coarse, refine_logdet, d10, edge_logq, d01_new_pcn, logq_new_pcn, corner_model, corner_state, corner_lg, z_corner, kernel, coarse_action, fine_action)
    old_pair_pcn_delta = old_pcn["logw"] - state_a["logw"]
    new_pair_pcn_delta = new_pcn["logw"] - state_b["logw"]

    high = np.argsort(np.abs(delta_logw))[::-1][:10]
    high_rows = []
    for i in high:
        high_rows.append(
            {
                "sample": int(i),
                "delta_logw": float(delta_logw[i]),
                "delta_s": float(delta_s[i]),
                "delta_pair_logq": float(delta_pair_logq[i]),
                "delta_corner_logq": float(delta_corner_logq[i]),
                "pair_output_rmse": float(np.sqrt(np.mean(pair_output_delta[i] ** 2))),
                "phi_rmse": float(np.sqrt(np.mean(recon_delta[i] ** 2))),
            }
        )

    convention = {
        "old_pair_checkpoint_stage": pair_old_ckpt.get("stage", pair_old_ckpt.get("config", {}).get("stage")),
        "portable_pair_checkpoint_stage": pair_new_ckpt.get("stage", pair_new_ckpt.get("config", {}).get("stage")),
        "old_pair_flow_arch": pair_old_ckpt.get("config", {}).get("flow_arch", "legacy_stage_model"),
        "portable_pair_flow_arch": pair_new_ckpt.get("config", {}).get("flow_arch"),
        "old_pair_lg_sigma": np.asarray(pair_lg["sigma"]).tolist(),
        "portable_pair_lg_sigma": np.asarray(pair_new_lg["sigma"]).tolist(),
        "conditioning_order": ["cprime", "accepted_edge_d10"],
        "reconstruction_slots": {"cprime": "psi[0::2,0::2]", "edge_d10": "psi[1::2,0::2]", "pair_d01": "psi[0::2,1::2]", "corner_d11": "psi[1::2,1::2]"},
        "logq_formula": "log_base(z) - forward_logdet - local_gaussian_log_jacobian",
        "logweight_formula": "-S_fine(phi) + S_coarse(raw_coarse) + logdet_refine - logq_missing",
        "tensor_shapes": {"cprime": list(cprime.shape), "d10": list(d10.shape), "pair_cond": list(pair_cond.shape), "pair_output": list(d01_old.shape)},
    }

    teacher_forced = {
        "A_old_output_old_logq_std": float(np.std(state_a["logw"], ddof=1)),
        "B_portable_output_portable_logq_std": float(np.std(state_b["logw"], ddof=1)),
        "C_old_output_portable_logq_delta_vs_A": quantiles(state_c["logw"] - state_a["logw"]),
        "D_portable_output_old_logq_delta_vs_A": quantiles(state_d["logw"] - state_a["logw"]),
        "B_full_swap_delta_vs_A": quantiles(delta_logw),
    }

    action_logq = {
        "delta_logw": quantiles(delta_logw),
        "delta_S": quantiles(delta_s),
        "minus_delta_S": quantiles(-delta_s),
        "delta_pair_logq": quantiles(delta_pair_logq),
        "delta_corner_logq": quantiles(delta_corner_logq),
        "delta_total_logq": quantiles(delta_logq),
        "minus_delta_total_logq": quantiles(-delta_logq),
        "corr_deltaS_deltaLogq": corr(delta_s, delta_logq),
        "corr_minusDeltaS_minusDeltaLogq": corr(-delta_s, -delta_logq),
        "cov_deltaS_deltaLogq": float(np.cov(delta_s, delta_logq, ddof=1)[0, 1]),
    }

    local = {
        "pair_output_error": quantiles(pair_output_delta),
        "pair_output_error_parity": parity_stats(pair_output_delta[:, 0]),
        "pair_output_error_by_abs_old_pair": binned_by_amplitude(np.abs(d01_old), pair_output_delta),
        "reconstructed_phi_error": quantiles(recon_delta),
        "action_density_delta": quantiles(action_density_delta),
        "per_sample_abs_delta_logw_top10": high_rows,
        "top1pct_abs_delta_logw_fraction_of_square_error": float(np.sum(np.sort(np.abs(delta_logw))[::-1][: max(1, args.n_samples // 100)] ** 2) / np.sum(delta_logw**2)),
        "top5pct_abs_delta_logw_fraction_of_square_error": float(np.sum(np.sort(np.abs(delta_logw))[::-1][: max(1, args.n_samples // 20)] ** 2) / np.sum(delta_logw**2)),
    }

    pair_pcn = {
        "rho": float(args.pcn_rho),
        "old_pair_only_delta_logw": quantiles(old_pair_pcn_delta),
        "portable_pair_only_delta_logw": quantiles(new_pair_pcn_delta),
        "old_acceptance_estimate": float(np.mean(np.minimum(1.0, np.exp(np.minimum(0.0, old_pair_pcn_delta))))),
        "portable_acceptance_estimate": float(np.mean(np.minimum(1.0, np.exp(np.minimum(0.0, new_pair_pcn_delta))))),
        "old_pair_logq_delta": quantiles(logq_old_pcn - logq_old),
        "portable_pair_logq_delta": quantiles(logq_new_pcn - logq_new),
        "old_action_delta": quantiles(old_pcn["sf"] - state_a["sf"]),
        "portable_action_delta": quantiles(new_pcn["sf"] - state_b["sf"]),
    }

    # This intentionally reproduces the wrong reconstruction convention to catch whether older diagnostics were inflated.
    wrong_phi_old = assemble_phi(coarse, d10, d01_old, state_a["d11"], kernel)
    wrong_phi_new = assemble_phi(coarse, d10, d01_new, state_b["d11"], kernel)
    wrong_delta_s = action_total(wrong_phi_new, fine_action) - action_total(wrong_phi_old, fine_action)
    coordinate_check = {
        "correct_cprime_delta_logw_std": float(np.std(delta_logw, ddof=1)),
        "wrong_raw_coarse_reconstruction_delta_logw_std": float(np.std((-wrong_delta_s) - delta_logq, ddof=1)),
        "cprime_minus_raw_coarse_rmse": float(np.sqrt(np.mean((cprime - coarse) ** 2))),
    }

    summary = {
        "old_config": str(args.old_config),
        "pair_config": str(args.pair_config),
        "n_samples": int(args.n_samples),
        "convention": convention,
        "coordinate_check": coordinate_check,
        "pair_teacher_match": {
            "model_space_rmse": float(np.sqrt(np.mean((y_new - y_old) ** 2))),
            "output_rmse": float(np.sqrt(np.mean(pair_output_delta**2))),
            "output_corr": corr(d01_new, d01_old),
            "logq_rmse": float(np.sqrt(np.mean((logq_new - logq_old) ** 2))),
            "logdet_rmse": float(np.sqrt(np.mean((ld_new - ld_old) ** 2))),
        },
        "teacher_forced_reconstruction": teacher_forced,
        "action_vs_logq_decomposition": action_logq,
        "local_contribution_map": local,
        "pair_pcn_local_diagnostic": pair_pcn,
    }
    write_json(args.output_dir / "pair_equivalence_audit_report.json", summary)

    dominant = "output/action" if action_logq["delta_S"]["std"] > action_logq["delta_total_logq"]["std"] else "density/logq"
    lines = [
        "# Pair Stage Equivalence Audit",
        "",
        "## Coordinate Convention",
        f"- pair condition order: `{convention['conditioning_order']}`",
        f"- reconstruction slots: `{convention['reconstruction_slots']}`",
        f"- pair tensor shape: `{convention['pair_output'] if 'pair_output' in convention else convention['tensor_shapes']['pair_output']}`",
        f"- old flow arch: `{convention['old_pair_flow_arch']}`",
        f"- portable flow arch: `{convention['portable_pair_flow_arch']}`",
        f"- pair local Gaussian sigma old/new: `{convention['old_pair_lg_sigma']}` / `{convention['portable_pair_lg_sigma']}`",
        f"- correct cprime swap delta std: `{coordinate_check['correct_cprime_delta_logw_std']:.6g}`",
        f"- wrong raw-coarse reconstruction swap delta std: `{coordinate_check['wrong_raw_coarse_reconstruction_delta_logw_std']:.6g}`",
        f"- cprime minus raw coarse RMSE: `{coordinate_check['cprime_minus_raw_coarse_rmse']:.6g}`",
        "",
        "## Pair Teacher Match",
        f"- output RMSE: `{summary['pair_teacher_match']['output_rmse']:.6g}`",
        f"- output correlation: `{summary['pair_teacher_match']['output_corr']:.6g}`",
        f"- logq RMSE: `{summary['pair_teacher_match']['logq_rmse']:.6g}`",
        f"- logdet RMSE: `{summary['pair_teacher_match']['logdet_rmse']:.6g}`",
        "",
        "## Teacher-Forced Reconstruction",
        f"- B full swap delta std: `{teacher_forced['B_full_swap_delta_vs_A']['std']:.6g}`",
        f"- C old output + portable logq delta std: `{teacher_forced['C_old_output_portable_logq_delta_vs_A']['std']:.6g}`",
        f"- D portable output + old logq delta std: `{teacher_forced['D_portable_output_old_logq_delta_vs_A']['std']:.6g}`",
        "",
        "## Action vs Logq",
        f"- delta logw std: `{action_logq['delta_logw']['std']:.6g}`",
        f"- delta S std: `{action_logq['delta_S']['std']:.6g}`",
        f"- delta total logq std: `{action_logq['delta_total_logq']['std']:.6g}`",
        f"- delta pair logq std: `{action_logq['delta_pair_logq']['std']:.6g}`",
        f"- delta corner logq std: `{action_logq['delta_corner_logq']['std']:.6g}`",
        f"- corr(delta S, delta logq): `{action_logq['corr_deltaS_deltaLogq']:.6g}`",
        f"- dominant term: `{dominant}`",
        "",
        "## Local Structure",
        f"- pair output error std: `{local['pair_output_error']['std']:.6g}`",
        f"- reconstructed phi error std: `{local['reconstructed_phi_error']['std']:.6g}`",
        f"- action density delta std: `{local['action_density_delta']['std']:.6g}`",
        f"- top 1% abs-delta-logw square-error fraction: `{local['top1pct_abs_delta_logw_fraction_of_square_error']:.3f}`",
        f"- top 5% abs-delta-logw square-error fraction: `{local['top5pct_abs_delta_logw_fraction_of_square_error']:.3f}`",
        "",
        "## Pair pCN Local",
        f"- old pair-only delta logw std: `{pair_pcn['old_pair_only_delta_logw']['std']:.6g}`",
        f"- portable pair-only delta logw std: `{pair_pcn['portable_pair_only_delta_logw']['std']:.6g}`",
        f"- old acceptance estimate: `{pair_pcn['old_acceptance_estimate']:.6g}`",
        f"- portable acceptance estimate: `{pair_pcn['portable_acceptance_estimate']:.6g}`",
        "",
        "## Decision",
    ]
    if teacher_forced["D_portable_output_old_logq_delta_vs_A"]["std"] > teacher_forced["C_old_output_portable_logq_delta_vs_A"]["std"]:
        lines.append("- output/reconstruction error dominates over pair logq bookkeeping. Next fix should target high-impact pair output directions, not isolated logq calibration.")
    else:
        lines.append("- pair logq bookkeeping dominates over output/reconstruction. Next fix should target logdet/logq convention.")
    if coordinate_check["wrong_raw_coarse_reconstruction_delta_logw_std"] > 1.5 * coordinate_check["correct_cprime_delta_logw_std"]:
        lines.append("- raw-coarse-vs-cprime reconstruction can inflate diagnostics; future pair audits must use cprime in the missing-field reconstruction path.")
    lines.append("- do not train corner/body until the pair-stage issue above is resolved.")
    (args.output_dir / "pair_equivalence_audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True, default=float), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
