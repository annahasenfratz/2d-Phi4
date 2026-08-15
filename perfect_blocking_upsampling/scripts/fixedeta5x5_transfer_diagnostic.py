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
FINITE = PROJECT_ROOT / "InverseBlocking_lam0p022_finite_footprint_flow"
FROZEN = PROJECT_ROOT / "InverseBlocking_lam0p022_frozen"
sys.path.insert(0, str(PKG / "scripts"))
sys.path.insert(0, str(PKG / "src"))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(FINITE / "scripts"))
sys.path.insert(0, str(FROZEN / "scripts"))
sys.path.insert(0, str(PKG / "src"))
sys.path.insert(0, str(PKG / "scripts"))

from _common import load_config, load_ensembles, load_frozen_models, load_kernel_spec  # noqa: E402
from perfect_blocking_upsampling.actions import action_total  # noqa: E402
from perfect_blocking_upsampling.kernels import apply_kernel, inverse_kernel  # noqa: E402
from perfect_blocking_upsampling.observables import observables as ensemble_observables  # noqa: E402
from train_finite_footprint_flow import local_observables, write_json  # noqa: E402
from train_finite_footprint_transported_detail import (  # noqa: E402
    inner_patch_metropolis,
    patch_sites,
    random_origin_patch_schedule,
    schedule_preflight,
)
from ML_sampling_clean.experiments.decimated_conditional_fillin.run_staged_decimated_conditional_fillin import (  # noqa: E402
    from_model_space,
    log_jacobian,
)

LOG2PI = math.log(2.0 * math.pi)


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
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    if aa.size < 2 or float(np.std(aa)) == 0.0 or float(np.std(bb)) == 0.0:
        return float("nan")
    return float(np.corrcoef(aa, bb)[0, 1])


def load_bundle(path: Path) -> dict[str, Any]:
    cfg = load_config(path)
    coarse, fine, coarse_manifest, fine_manifest, paths = load_ensembles(cfg)
    refine_model, refine_state, stages, coarse_action, fine_action, _ = load_frozen_models(cfg)
    refine_model.load_state_dict(refine_state)
    refine_model.eval()
    for model, _, state, *_ in stages.values():
        model.load_state_dict(state)
        model.eval()
    kernel, kernel_json = load_kernel_spec(cfg)
    return {
        "config": cfg,
        "coarse": coarse,
        "fine": fine,
        "coarse_manifest": coarse_manifest,
        "fine_manifest": fine_manifest,
        "paths": paths,
        "refine_model": refine_model,
        "stages": stages,
        "coarse_action": coarse_action,
        "fine_action": fine_action,
        "kernel": kernel,
        "kernel_json": kernel_json,
    }


def apply_refine_loaded(model, u: np.ndarray, batch_size: int = 64) -> tuple[np.ndarray, np.ndarray]:
    import torch

    outs = []
    logdets = []
    model.eval()
    with torch.no_grad():
        for start in range(0, u.shape[0], batch_size):
            ub_np = u[start : start + batch_size]
            ub = torch.tensor(ub_np[:, None].reshape(ub_np.shape[0], -1), dtype=torch.float32)
            cond = torch.zeros((ub.shape[0], ub.shape[1]), dtype=torch.float32)
            x, logdet = model.forward(ub, cond)
            outs.append(x.cpu().numpy().reshape(ub_np.shape[0], ub_np.shape[1], ub_np.shape[2]))
            logdets.append(logdet.cpu().numpy())
    return np.concatenate(outs, axis=0).astype(np.float32), np.concatenate(logdets, axis=0).astype(np.float64)


def stage_forward_from_z(model, z: np.ndarray, cond: np.ndarray, lg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    import torch

    z_t = torch.tensor(z.reshape(z.shape[0], -1), dtype=torch.float32)
    cond_t = torch.tensor(cond.reshape(cond.shape[0], -1), dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        y_flat, logdet = model.forward(z_t, cond_t)
    y = y_flat.cpu().numpy().reshape(z.shape[0], z.shape[1], z.shape[2], z.shape[3]).astype(np.float32)
    x = from_model_space(y, cond, lg)
    log_base = -0.5 * np.sum(z.reshape(z.shape[0], -1).astype(np.float64) ** 2 + LOG2PI, axis=1)
    logq_y = log_base - logdet.cpu().numpy().astype(np.float64)
    return x.astype(np.float32), (logq_y - log_jacobian(cond, lg)).astype(np.float64)


def reconstruct(c: np.ndarray, d: np.ndarray) -> np.ndarray:
    psi = np.empty((c.shape[0], 2 * c.shape[1], 2 * c.shape[2]), dtype=np.float32)
    psi[:, 0::2, 0::2] = c
    psi[:, 1::2, 0::2] = d[:, 0]
    psi[:, 0::2, 1::2] = d[:, 1]
    psi[:, 1::2, 1::2] = d[:, 2]
    return psi


def sample_z(rng: np.random.Generator, n: int, l: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        rng.standard_normal((n, 1, l, l)).astype(np.float32),
        rng.standard_normal((n, 1, l, l)).astype(np.float32),
        rng.standard_normal((n, 1, l, l)).astype(np.float32),
    )


def compute_state(bundle: dict[str, Any], u: np.ndarray, z_edge: np.ndarray, z_pair: np.ndarray, z_corner: np.ndarray) -> dict[str, Any]:
    cprime, logdet = apply_refine_loaded(bundle["refine_model"], u)
    edge_model, edge_lg, _ = bundle["stages"]["edge"][:3]
    pair_model, pair_lg, _ = bundle["stages"]["pair"][:3]
    corner_model, corner_lg, _ = bundle["stages"]["corner"][:3]
    d10, l10 = stage_forward_from_z(edge_model, z_edge, cprime[:, None], edge_lg)
    d01, l01 = stage_forward_from_z(pair_model, z_pair, np.concatenate([cprime[:, None], d10], axis=1), pair_lg)
    d11, l11 = stage_forward_from_z(corner_model, z_corner, np.concatenate([cprime[:, None], d10, d01], axis=1), corner_lg)
    d = np.concatenate([d10, d01, d11], axis=1).astype(np.float32)
    psi = reconstruct(cprime, d)
    phi, inv = inverse_kernel(psi, bundle["kernel"])
    sf = action_total(phi, bundle["fine_action"])
    sc = action_total(u, bundle["coarse_action"])
    logq = l10 + l01 + l11
    logw = -sf + sc + logdet - logq
    return {
        "u": u.astype(np.float32),
        "cprime": cprime,
        "d": d,
        "psi": psi,
        "phi": phi,
        "sf": sf.astype(np.float64),
        "sc": sc.astype(np.float64),
        "logdet": logdet.astype(np.float64),
        "logq_edge": l10,
        "logq_pair": l01,
        "logq_corner": l11,
        "logq": logq,
        "logw": logw.astype(np.float64),
        "inv": inv,
    }


def reblock_consistency(fine: np.ndarray, kernel) -> dict[str, Any]:
    psi = apply_kernel(fine, kernel)
    c = psi[:, 0::2, 0::2]
    d = np.stack([psi[:, 1::2, 0::2], psi[:, 0::2, 1::2], psi[:, 1::2, 1::2]], axis=1)
    psi_re = reconstruct(c, d)
    phi_re, inv = inverse_kernel(psi_re, kernel)
    return {
        "blocked_shape": list(c.shape),
        "psi_reconstruct_rmse": float(np.sqrt(np.mean((psi_re - psi) ** 2))),
        "phi_roundtrip_rmse": float(np.sqrt(np.mean((phi_re - fine.astype(np.float32)) ** 2))),
        "inverse_info": inv,
    }


def component_compatibility(small3: dict[str, Any], five: dict[str, Any], n: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    fine = five["fine"][:n].astype(np.float32)
    u_native = five["coarse"][:n].astype(np.float32)
    psi5 = apply_kernel(fine, five["kernel"])
    u5 = psi5[:, 0::2, 0::2].astype(np.float32)
    target_d5 = np.stack([psi5[:, 1::2, 0::2], psi5[:, 0::2, 1::2], psi5[:, 1::2, 1::2]], axis=1).astype(np.float32)
    z = sample_z(rng, n, u5.shape[1])
    state5 = compute_state(five, u5, *z)
    state_small_native = compute_state(small3, u_native, *z)
    state_small_u5 = compute_state(small3, u5, *z)
    cprime5_target = u5
    cprime_rmse = state5["cprime"] - cprime5_target
    target_phi_from_model_details, _ = inverse_kernel(reconstruct(u5, state5["d"]), five["kernel"])
    target_phi_from_true_details, _ = inverse_kernel(reconstruct(u5, target_d5), five["kernel"])
    return {
        "n_samples": n,
        "coarse_distribution_native_vs_5x5_blocked": {
            "rmse": float(np.sqrt(np.mean((u_native - u5) ** 2))),
            "corr": corr(u_native, u5),
            "native_observables": ensemble_observables(u_native, five["coarse_action"]),
            "blocked5_observables": ensemble_observables(u5, five["coarse_action"]),
        },
        "coarse_refine_against_5x5_blocked_cprime": {
            "rmse": float(np.sqrt(np.mean(cprime_rmse * cprime_rmse))),
            "corr": corr(state5["cprime"], cprime5_target),
            "error": quantiles(cprime_rmse),
            "logdet": quantiles(state5["logdet"]),
        },
        "detail_against_true_5x5_psi_details_random_latent": {
            "edge_rmse": float(np.sqrt(np.mean((state5["d"][:, 0] - target_d5[:, 0]) ** 2))),
            "pair_rmse": float(np.sqrt(np.mean((state5["d"][:, 1] - target_d5[:, 1]) ** 2))),
            "corner_rmse": float(np.sqrt(np.mean((state5["d"][:, 2] - target_d5[:, 2]) ** 2))),
            "edge_corr": corr(state5["d"][:, 0], target_d5[:, 0]),
            "pair_corr": corr(state5["d"][:, 1], target_d5[:, 1]),
            "corner_corr": corr(state5["d"][:, 2], target_d5[:, 2]),
        },
        "reconstructed_phi_vs_reference": {
            "using_model_details_rmse": float(np.sqrt(np.mean((target_phi_from_model_details - fine) ** 2))),
            "using_true_5x5_details_rmse": float(np.sqrt(np.mean((target_phi_from_true_details - fine) ** 2))),
            "model_obs": local_observables(target_phi_from_model_details),
            "reference_obs": local_observables(fine),
        },
        "5x5_logweight_components": {
            "S_fine": quantiles(state5["sf"]),
            "S_coarse": quantiles(state5["sc"]),
            "logdet_refine": quantiles(state5["logdet"]),
            "logq_edge": quantiles(state5["logq_edge"]),
            "logq_pair": quantiles(state5["logq_pair"]),
            "logq_corner": quantiles(state5["logq_corner"]),
            "logq_total": quantiles(state5["logq"]),
            "logweight": quantiles(state5["logw"]),
        },
        "small3_native_vs_5x5_same_latent": {
            "phi_rmse": float(np.sqrt(np.mean((state5["phi"] - state_small_native["phi"]) ** 2))),
            "delta_S": quantiles(state5["sf"] - state_small_native["sf"]),
            "delta_logq": quantiles(state5["logq"] - state_small_native["logq"]),
            "delta_logw": quantiles(state5["logw"] - state_small_native["logw"]),
        },
        "small3_components_on_5x5_blocked_u_vs_5x5_kernel": {
            "phi_rmse": float(np.sqrt(np.mean((state5["phi"] - state_small_u5["phi"]) ** 2))),
            "delta_S": quantiles(state5["sf"] - state_small_u5["sf"]),
            "delta_logq": quantiles(state5["logq"] - state_small_u5["logq"]),
            "delta_logw": quantiles(state5["logw"] - state_small_u5["logw"]),
        },
        "dominance_note": "Direct compatibility is diagnostic only; random latent details are not teacher-forced to the paired fine reference.",
    }


def one_step_ar_smoke(bundle: dict[str, Any], seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    u = bundle["coarse"][:1].astype(np.float32)
    z = sample_z(rng, 1, u.shape[1])
    state = compute_state(bundle, u, *z)
    schedule = random_origin_patch_schedule(u.shape[1], 4, rng, "random")
    x0, y0, tile = schedule[0]
    sites = patch_sites(u.shape[1], x0, y0, 4)
    u_prop, inner_acc = inner_patch_metropolis(u[0].copy(), sites, rng)
    prop = compute_state(bundle, u_prop[None], *z)
    coarse_delta = float(prop["logw"][0] - state["logw"][0])
    rho = 0.5
    noise = math.sqrt(1.0 - rho * rho)
    z_prop = [a.copy() for a in z]
    for arr in z_prop:
        for i, j in sites:
            arr[0, 0, i, j] = rho * arr[0, 0, i, j] + noise * float(rng.standard_normal())
    latent_prop = compute_state(bundle, u, *z_prop)
    latent_delta = float(latent_prop["logw"][0] - state["logw"][0])
    return {
        "initial_logweight_finite": bool(np.isfinite(state["logw"]).all()),
        "coarse_patch": {
            "patch_x": int(x0),
            "patch_y": int(y0),
            "tile": tile,
            "inner_acceptance": float(inner_acc),
            "delta_logw": coarse_delta,
            "accept_prob": float(min(1.0, math.exp(min(0.0, coarse_delta)))),
        },
        "latent_pcn_every_sweep": {
            "rho": rho,
            "delta_logw": latent_delta,
            "accept_prob": float(min(1.0, math.exp(min(0.0, latent_delta)))),
        },
    }


def write_report(out: Path, summary: dict[str, Any]) -> None:
    compat = summary["direct_compatibility"]
    pre = summary["preflight"]
    lines = [
        "# Fixed-Eta 5x5 Transfer Preflight And Compatibility Diagnostic",
        "",
        "## Scope",
        "",
        "This diagnostic swaps only the blocking kernel to `fixedeta_5x5_lam0p022.json` while keeping the current small3-trained sampler components. It is not a retrained 5x5 sampler.",
        "",
        "## 5x5 Preflight",
        "",
        f"- kernel: `{pre['kernel']['name']}`",
        f"- eta: `{pre['kernel']['eta']}`",
        f"- normalization: `{pre['kernel']['normalization']}`",
        f"- reblocking psi RMSE: `{pre['reblock_consistency']['psi_reconstruct_rmse']:.6g}`",
        f"- kernel roundtrip phi RMSE: `{pre['reblock_consistency']['phi_roundtrip_rmse']:.6g}`",
        f"- logweight finite in one-step smoke: `{pre['one_step_ar_smoke']['initial_logweight_finite']}`",
        f"- one coarse-patch delta logw: `{pre['one_step_ar_smoke']['coarse_patch']['delta_logw']:.6g}`",
        f"- one latent pCN delta logw: `{pre['one_step_ar_smoke']['latent_pcn_every_sweep']['delta_logw']:.6g}`",
        f"- scheduler N_patch/sweep: `{pre['scheduler_preflight']['patches_per_sweep']}`",
        f"- dummy L16->L32 instantiation: `{pre['dummy_l16_l32_instantiation']}`",
        "",
        "## Direct Compatibility Diagnostic",
        "",
        f"- native L8 vs 5x5-blocked direct coarse RMSE: `{compat['coarse_distribution_native_vs_5x5_blocked']['rmse']:.6g}`",
        f"- native L8 vs 5x5-blocked direct coarse corr: `{compat['coarse_distribution_native_vs_5x5_blocked']['corr']:.6g}`",
        f"- coarse-refine cprime RMSE against 5x5 blocked cprime: `{compat['coarse_refine_against_5x5_blocked_cprime']['rmse']:.6g}`",
        f"- coarse-refine cprime corr against 5x5 blocked cprime: `{compat['coarse_refine_against_5x5_blocked_cprime']['corr']:.6g}`",
        f"- random-latent edge RMSE vs true 5x5 psi detail: `{compat['detail_against_true_5x5_psi_details_random_latent']['edge_rmse']:.6g}`",
        f"- random-latent pair RMSE vs true 5x5 psi detail: `{compat['detail_against_true_5x5_psi_details_random_latent']['pair_rmse']:.6g}`",
        f"- random-latent corner RMSE vs true 5x5 psi detail: `{compat['detail_against_true_5x5_psi_details_random_latent']['corner_rmse']:.6g}`",
        f"- reconstructed phi RMSE with model details vs held-out fine: `{compat['reconstructed_phi_vs_reference']['using_model_details_rmse']:.6g}`",
        f"- reconstructed phi RMSE with true 5x5 details vs held-out fine: `{compat['reconstructed_phi_vs_reference']['using_true_5x5_details_rmse']:.6g}`",
        f"- 5x5 logweight std: `{compat['5x5_logweight_components']['logweight']['std']:.6g}`",
        f"- 5x5-vs-small3 same native-coarse delta S std: `{compat['small3_native_vs_5x5_same_latent']['delta_S']['std']:.6g}`",
        f"- 5x5-vs-small3 same native-coarse delta logw std: `{compat['small3_native_vs_5x5_same_latent']['delta_logw']['std']:.6g}`",
        "",
        "## Interpretation",
        "",
    ]
    c_rmse = compat["coarse_refine_against_5x5_blocked_cprime"]["rmse"]
    lw_std = compat["5x5_logweight_components"]["logweight"]["std"]
    if c_rmse > 0.1:
        lines.append("- The current small3 coarse-refine is not directly compatible with 5x5-blocked coarse coordinates; retraining the coarse-refine is the first required step.")
    else:
        lines.append("- The current coarse-refine is not the dominant direct-compatibility issue by RMSE; inspect detail stages next.")
    if lw_std > 5.0:
        lines.append("- Raw 5x5 direct logweight spread is large, so do not run long 5x5 sampler validation before component retraining/audits.")
    lines.append("- Pair/corner remain small3 old-weight procedural ports; they are not accepted 5x5 components from this diagnostic.")
    lines.append("- No L16->L32 run was launched.")
    (out / "fixedeta5x5_preflight_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--small3-config", type=Path, default=PKG / "outputs" / "procedural_corner_diagnostics" / "old_pair_corner_procedural_masks.yaml")
    ap.add_argument("--five-config", type=Path, default=PKG / "outputs" / "fixedeta5x5_transfer" / "fixedeta5x5_transfer_config.yaml")
    ap.add_argument("--output-dir", type=Path, default=PKG / "outputs" / "fixedeta5x5_transfer")
    ap.add_argument("--n-samples", type=int, default=128)
    ap.add_argument("--seed", type=int, default=20260720)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    small3 = load_bundle(args.small3_config)
    five = load_bundle(args.five_config)
    reblock = reblock_consistency(five["fine"][: args.n_samples].astype(np.float32), five["kernel"])
    compat = component_compatibility(small3, five, args.n_samples, args.seed)
    sched = schedule_preflight(8, 4, "random", n_sweeps=256, seed=args.seed)
    smoke = one_step_ar_smoke(five, args.seed + 11)
    summary = {
        "status": "completed",
        "config": {
            "small3": str(args.small3_config),
            "fixedeta5x5": str(args.five_config),
            "n_samples": args.n_samples,
            "seed": args.seed,
        },
        "preflight": {
            "kernel": five["kernel_json"],
            "reblock_consistency": reblock,
            "scheduler_preflight": sched,
            "one_step_ar_smoke": smoke,
            "dummy_l16_l32_instantiation": "passed_by_shape_parametric_bundle_preflight_elsewhere; components load shape-parametrically",
        },
        "direct_compatibility": compat,
    }
    write_json(args.output_dir / "fixedeta5x5_preflight_summary.json", summary)
    write_report(args.output_dir, summary)
    print(json.dumps({"status": "completed", "report": str(args.output_dir / "fixedeta5x5_preflight_report.md")}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
