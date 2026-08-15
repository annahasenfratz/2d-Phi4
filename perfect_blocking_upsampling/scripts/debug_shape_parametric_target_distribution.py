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
sys.path.insert(0, str(PKG / "scripts"))
sys.path.insert(0, str(PKG / "src"))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(FINITE / "scripts"))
sys.path.insert(0, str(PKG / "src"))
sys.path.insert(0, str(PKG / "scripts"))

from _common import load_config, load_ensembles, load_frozen_models, load_kernel_spec  # noqa: E402
from perfect_blocking_upsampling.actions import action_density, action_total  # noqa: E402
from perfect_blocking_upsampling.kernels import apply_kernel, inverse_kernel  # noqa: E402
from perfect_blocking_upsampling.observables import observables as ensemble_observables  # noqa: E402
from train_finite_footprint_flow import local_observables, write_csv, write_json  # noqa: E402
from train_finite_footprint_transported_detail import inner_patch_metropolis, patch_sites, random_origin_patch_schedule  # noqa: E402
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
    if np.std(aa) == 0.0 or np.std(bb) == 0.0:
        return float("nan")
    return float(np.corrcoef(aa, bb)[0, 1])


def write_old_config(out: Path, assembled_config: Path) -> Path:
    cfg = load_config(assembled_config)
    text = Path(assembled_config).read_text()
    text = text.replace(
        "outputs/procedural_corner_diagnostics/old_pair_corner_procedural_masks_bundle",
        "../InverseBlocking_lam0p022_frozen/kernels/lam0p022_kappa0p2705_small3_refine",
    )
    path = out / "old_full_frozen_for_debug.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def load_bundle(config: Path):
    cfg = load_config(config)
    coarse, fine, coarse_manifest, fine_manifest, paths = load_ensembles(cfg)
    refine_model, refine_state, stages, coarse_action, fine_action, refine_ckpt = load_frozen_models(cfg)
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


def apply_refine_loaded(model, u: np.ndarray, batch_size: int = 32) -> tuple[np.ndarray, np.ndarray]:
    import torch

    l = u.shape[1]
    outs = []
    logdets = []
    model.eval()
    with torch.no_grad():
        for start in range(0, u.shape[0], batch_size):
            ub_np = u[start : start + batch_size]
            ub = torch.tensor(ub_np[:, None].reshape(ub_np.shape[0], -1), dtype=torch.float32)
            cond = torch.zeros((ub.shape[0], ub.shape[1]), dtype=torch.float32)
            x, logdet = model.forward(ub, cond)
            outs.append(x.cpu().numpy().reshape(ub.shape[0], l, l))
            logdets.append(logdet.cpu().numpy())
    return np.concatenate(outs, axis=0).astype(np.float32), np.concatenate(logdets, axis=0).astype(np.float64)


def stage_forward_from_z(model, z: np.ndarray, cond: np.ndarray, lg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import torch

    model.eval()
    z_t = torch.tensor(z.reshape(z.shape[0], -1), dtype=torch.float32)
    cond_t = torch.tensor(cond.reshape(cond.shape[0], -1), dtype=torch.float32)
    with torch.no_grad():
        y_flat, logdet = model.forward(z_t, cond_t)
    y = y_flat.cpu().numpy().reshape(z.shape[0], z.shape[1], z.shape[2], z.shape[3]).astype(np.float32)
    x = from_model_space(y, cond, lg)
    log_base = -0.5 * np.sum(z.reshape(z.shape[0], -1).astype(np.float64) ** 2 + LOG2PI, axis=1)
    logq_y = log_base - logdet.cpu().numpy().astype(np.float64)
    logq = (logq_y - log_jacobian(cond, lg)).astype(np.float64)
    return x.astype(np.float32), logq, logdet.cpu().numpy().astype(np.float64)


def reconstruct(c: np.ndarray, d: np.ndarray) -> np.ndarray:
    psi = np.empty((c.shape[0], 2 * c.shape[1], 2 * c.shape[2]), dtype=np.float32)
    psi[:, 0::2, 0::2] = c
    psi[:, 1::2, 0::2] = d[:, 0]
    psi[:, 0::2, 1::2] = d[:, 1]
    psi[:, 1::2, 1::2] = d[:, 2]
    return psi


def sample_z(rng: np.random.Generator, n: int, L: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        rng.standard_normal((n, 1, L, L)).astype(np.float32),
        rng.standard_normal((n, 1, L, L)).astype(np.float32),
        rng.standard_normal((n, 1, L, L)).astype(np.float32),
    )


def direct_action_total(phi: np.ndarray, action) -> np.ndarray:
    arr = np.asarray(phi, dtype=np.float64)
    phi2 = arr * arr
    phi4 = phi2 * phi2
    nn = arr * np.roll(arr, -1, axis=-1) + arr * np.roll(arr, -1, axis=-2)
    diag = arr * np.roll(np.roll(arr, -1, axis=-1), -1, axis=-2)
    if action.type == "phi4_nn":
        dens = (1.0 - 2.0 * action.lambda_) * phi2 + action.lambda_ * phi4 - 2.0 * action.kappa * nn
    elif action.type == "phi4_nn_plus_diag":
        dens = (1.0 - 2.0 * action.lambda_) * phi2 + action.lambda_ * phi4 - 2.0 * action.kappa * nn - 2.0 * action.kappa_diag * diag
    else:
        raise ValueError(action.type)
    return dens.sum(axis=(-2, -1)) if dens.ndim == 3 else np.asarray(dens.sum())


def compute_state(u: np.ndarray, z_edge: np.ndarray, z_pair: np.ndarray, z_corner: np.ndarray, ctx: dict[str, Any]) -> dict[str, Any]:
    cprime, logdet = apply_refine_loaded(ctx["refine_model"], u)
    edge_model, edge_lg, _ = ctx["stages"]["edge"][:3]
    pair_model, pair_lg, _ = ctx["stages"]["pair"][:3]
    corner_model, corner_lg, _ = ctx["stages"]["corner"][:3]
    d10, l10, ld10 = stage_forward_from_z(edge_model, z_edge, cprime[:, None], edge_lg)
    pair_cond = np.concatenate([cprime[:, None], d10], axis=1)
    d01, l01, ld01 = stage_forward_from_z(pair_model, z_pair, pair_cond, pair_lg)
    corner_cond = np.concatenate([cprime[:, None], d10, d01], axis=1)
    d11, l11, ld11 = stage_forward_from_z(corner_model, z_corner, corner_cond, corner_lg)
    d = np.concatenate([d10, d01, d11], axis=1).astype(np.float32)
    psi = reconstruct(cprime, d)
    phi, inv = inverse_kernel(psi, ctx["kernel"])
    sf = action_total(phi, ctx["fine_action"])
    sc = action_total(u, ctx["coarse_action"])
    logq = l10 + l01 + l11
    logw = -sf + sc + logdet - logq
    return {
        "u": u.astype(np.float32),
        "cprime": cprime,
        "d": d,
        "psi": psi,
        "phi": phi,
        "sf": sf.astype(np.float64),
        "sf_direct": direct_action_total(phi, ctx["fine_action"]).astype(np.float64),
        "sc": sc.astype(np.float64),
        "logdet": logdet.astype(np.float64),
        "logq_edge": l10,
        "logq_pair": l01,
        "logq_corner": l11,
        "logq": logq,
        "logdet_edge": ld10,
        "logdet_pair": ld01,
        "logdet_corner": ld11,
        "logw": logw.astype(np.float64),
        "inv": inv,
        "obs": local_observables(phi),
    }


def compare_states(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    obs_delta = {}
    for k in a["obs"]:
        obs_delta[k] = float(b["obs"][k] - a["obs"][k])
    return {
        "phi_rmse": float(np.sqrt(np.mean((b["phi"] - a["phi"]) ** 2))),
        "psi_rmse": float(np.sqrt(np.mean((b["psi"] - a["psi"]) ** 2))),
        "cprime_rmse": float(np.sqrt(np.mean((b["cprime"] - a["cprime"]) ** 2))),
        "d_rmse_by_stage": {
            "edge": float(np.sqrt(np.mean((b["d"][:, 0] - a["d"][:, 0]) ** 2))),
            "pair": float(np.sqrt(np.mean((b["d"][:, 1] - a["d"][:, 1]) ** 2))),
            "corner": float(np.sqrt(np.mean((b["d"][:, 2] - a["d"][:, 2]) ** 2))),
        },
        "delta_S": quantiles(b["sf"] - a["sf"]),
        "delta_logdet_refine": quantiles(b["logdet"] - a["logdet"]),
        "delta_logq_edge": quantiles(b["logq_edge"] - a["logq_edge"]),
        "delta_logq_pair": quantiles(b["logq_pair"] - a["logq_pair"]),
        "delta_logq_corner": quantiles(b["logq_corner"] - a["logq_corner"]),
        "delta_logq_total": quantiles(b["logq"] - a["logq"]),
        "delta_logweight": quantiles(b["logw"] - a["logw"]),
        "action_direct_minus_project": {
            "old": quantiles(a["sf_direct"] - a["sf"]),
            "assembled": quantiles(b["sf_direct"] - b["sf"]),
        },
        "observable_delta": obs_delta,
    }


def one_coarse_proposal(old_state, new_state, rng, old_ctx, new_ctx, patch_size: int):
    schedule = random_origin_patch_schedule(old_state["u"].shape[1], patch_size, rng, "random")
    x0, y0, tile = schedule[0]
    sites = patch_sites(old_state["u"].shape[1], x0, y0, patch_size)
    u_new = old_state["u"][0].copy()
    u_new, inner_acc = inner_patch_metropolis(u_new, sites, rng)
    old_prop = compute_state(u_new[None], old_ctx["z_edge"], old_ctx["z_pair"], old_ctx["z_corner"], old_ctx)
    new_prop = compute_state(u_new[None], new_ctx["z_edge"], new_ctx["z_pair"], new_ctx["z_corner"], new_ctx)
    old_delta = float(old_prop["logw"][0] - old_state["logw"][0])
    new_delta = float(new_prop["logw"][0] - new_state["logw"][0])
    return {
        "patch": {"x0": int(x0), "y0": int(y0), "tile": tile, "inner_acceptance": float(inner_acc)},
        "proposal_state_compare": compare_states(old_prop, new_prop),
        "old_delta_logw": old_delta,
        "assembled_delta_logw": new_delta,
        "delta_delta_logw": new_delta - old_delta,
        "old_accept_prob": float(min(1.0, math.exp(min(0.0, old_delta)))),
        "assembled_accept_prob": float(min(1.0, math.exp(min(0.0, new_delta)))),
    }


def one_latent_proposal(old_state, new_state, rng, old_ctx, new_ctx, patch_size: int, rho: float):
    schedule = random_origin_patch_schedule(old_state["u"].shape[1], patch_size, rng, "random")
    x0, y0, tile = schedule[0]
    sites = patch_sites(old_state["u"].shape[1], x0, y0, patch_size)
    noise = math.sqrt(max(0.0, 1.0 - rho * rho))
    z_old = [old_ctx["z_edge"].copy(), old_ctx["z_pair"].copy(), old_ctx["z_corner"].copy()]
    z_new = [new_ctx["z_edge"].copy(), new_ctx["z_pair"].copy(), new_ctx["z_corner"].copy()]
    for zi_old, zi_new in zip(z_old, z_new):
        for i, j in sites:
            eps = float(rng.standard_normal())
            zi_old[0, 0, i, j] = rho * zi_old[0, 0, i, j] + noise * eps
            zi_new[0, 0, i, j] = rho * zi_new[0, 0, i, j] + noise * eps
    old_prop = compute_state(old_state["u"], z_old[0], z_old[1], z_old[2], old_ctx)
    new_prop = compute_state(new_state["u"], z_new[0], z_new[1], z_new[2], new_ctx)
    old_delta = float(old_prop["logw"][0] - old_state["logw"][0])
    new_delta = float(new_prop["logw"][0] - new_state["logw"][0])
    return {
        "patch": {"x0": int(x0), "y0": int(y0), "tile": tile},
        "proposal_state_compare": compare_states(old_prop, new_prop),
        "old_delta_logw": old_delta,
        "assembled_delta_logw": new_delta,
        "delta_delta_logw": new_delta - old_delta,
        "old_accept_prob": float(min(1.0, math.exp(min(0.0, old_delta)))),
        "assembled_accept_prob": float(min(1.0, math.exp(min(0.0, new_delta)))),
    }


def trajectory_debug(old_state, new_state, rng, old_ctx, new_ctx, sweeps: int, patch_size: int, rho: float, pcn_interval: int) -> list[dict[str, Any]]:
    rows = []
    for sweep in range(sweeps):
        schedule = random_origin_patch_schedule(old_state["u"].shape[1], patch_size, rng, "random")
        for attempt, (x0, y0, tile) in enumerate(schedule):
            sites = patch_sites(old_state["u"].shape[1], x0, y0, patch_size)
            u_old = old_state["u"][0].copy()
            u_new = new_state["u"][0].copy()
            # Use identical local proposal noise by generating the proposed
            # coarse patch once from the old current coarse field. If chains
            # have diverged in coarse state, this intentionally tests whether
            # decisions diverge under a common proposal target.
            u_prop, _ = inner_patch_metropolis(u_old, sites, rng)
            old_prop = compute_state(u_prop[None], old_ctx["z_edge"], old_ctx["z_pair"], old_ctx["z_corner"], old_ctx)
            new_prop = compute_state(u_prop[None], new_ctx["z_edge"], new_ctx["z_pair"], new_ctx["z_corner"], new_ctx)
            old_delta = float(old_prop["logw"][0] - old_state["logw"][0])
            new_delta = float(new_prop["logw"][0] - new_state["logw"][0])
            logu = math.log(max(rng.random(), 1.0e-300))
            old_accept = logu < min(0.0, old_delta)
            new_accept = logu < min(0.0, new_delta)
            if old_accept:
                old_state = old_prop
            if new_accept:
                new_state = new_prop
            rows.append(
                {
                    "sweep": sweep,
                    "attempt": attempt,
                    "move": "coarse",
                    "old_accept": int(old_accept),
                    "assembled_accept": int(new_accept),
                    "accept_same": int(old_accept == new_accept),
                    "old_delta_logw": old_delta,
                    "assembled_delta_logw": new_delta,
                    "delta_delta_logw": new_delta - old_delta,
                    "old_S": float(old_state["sf"][0]),
                    "assembled_S": float(new_state["sf"][0]),
                    "old_logq": float(old_state["logq"][0]),
                    "assembled_logq": float(new_state["logq"][0]),
                    "old_m": float(old_state["obs"]["m"]),
                    "assembled_m": float(new_state["obs"]["m"]),
                }
            )
        if (sweep + 1) % pcn_interval == 0:
            latent = one_latent_proposal(old_state, new_state, rng, old_ctx, new_ctx, patch_size, rho)
            old_delta = latent["old_delta_logw"]
            new_delta = latent["assembled_delta_logw"]
            logu = math.log(max(rng.random(), 1.0e-300))
            rows.append(
                {
                    "sweep": sweep,
                    "attempt": -1,
                    "move": "latent_probe",
                    "old_accept": int(logu < min(0.0, old_delta)),
                    "assembled_accept": int(logu < min(0.0, new_delta)),
                    "accept_same": int((logu < min(0.0, old_delta)) == (logu < min(0.0, new_delta))),
                    "old_delta_logw": old_delta,
                    "assembled_delta_logw": new_delta,
                    "delta_delta_logw": new_delta - old_delta,
                    "old_S": float(old_state["sf"][0]),
                    "assembled_S": float(new_state["sf"][0]),
                    "old_logq": float(old_state["logq"][0]),
                    "assembled_logq": float(new_state["logq"][0]),
                    "old_m": float(old_state["obs"]["m"]),
                    "assembled_m": float(new_state["obs"]["m"]),
                }
            )
    return rows


def coarse_distribution_check(coarse: np.ndarray, fine: np.ndarray, kernel, coarse_action, fine_action) -> dict[str, Any]:
    psi = apply_kernel(fine, kernel)
    blocked = psi[:, 0::2, 0::2]
    coarse_obs = ensemble_observables(coarse, coarse_action)
    blocked_obs = ensemble_observables(blocked, coarse_action)
    fine_obs = ensemble_observables(fine, fine_action)
    keys = ["m", "abs_m", "phi2", "phi4", "NN", "action_density", "Binder_U4", "susceptibility", "xi_over_L"]
    return {
        "coarse_direct": {k: coarse_obs.get(k) for k in keys},
        "blocked_direct_L16_cprime": {k: blocked_obs.get(k) for k in keys},
        "fine_reference": {k: fine_obs.get(k) for k in keys},
        "blocked_vs_coarse_delta": {k: float(blocked_obs[k] - coarse_obs[k]) for k in keys if isinstance(coarse_obs.get(k), (float, int)) and isinstance(blocked_obs.get(k), (float, int))},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assembled-config", type=Path, default=PKG / "outputs" / "procedural_corner_diagnostics" / "old_pair_corner_procedural_masks.yaml")
    ap.add_argument("--output-dir", type=Path, default=PKG / "outputs" / "shape_parametric_sampler_validation")
    ap.add_argument("--n-initial", type=int, default=64)
    ap.add_argument("--tiny-sweeps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20260714)
    args = ap.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    old_config = write_old_config(out, args.assembled_config)
    old = load_bundle(old_config)
    assembled = load_bundle(args.assembled_config)
    rng = np.random.default_rng(args.seed)
    coarse = old["coarse"][: args.n_initial].astype(np.float32)
    z_edge, z_pair, z_corner = sample_z(rng, args.n_initial, coarse.shape[1])
    old_ctx = {**old, "z_edge": z_edge, "z_pair": z_pair, "z_corner": z_corner}
    new_ctx = {**assembled, "z_edge": z_edge, "z_pair": z_pair, "z_corner": z_corner}
    old_state = compute_state(coarse, z_edge, z_pair, z_corner, old_ctx)
    new_state = compute_state(coarse, z_edge, z_pair, z_corner, new_ctx)
    initial = compare_states(old_state, new_state)

    one_rng = np.random.default_rng(args.seed + 1)
    old_one = {**old, "z_edge": z_edge[:1], "z_pair": z_pair[:1], "z_corner": z_corner[:1]}
    new_one = {**assembled, "z_edge": z_edge[:1], "z_pair": z_pair[:1], "z_corner": z_corner[:1]}
    old_state_one = compute_state(coarse[:1], z_edge[:1], z_pair[:1], z_corner[:1], old_one)
    new_state_one = compute_state(coarse[:1], z_edge[:1], z_pair[:1], z_corner[:1], new_one)
    coarse_prop = one_coarse_proposal(old_state_one, new_state_one, one_rng, old_one, new_one, patch_size=4)
    latent_prop = one_latent_proposal(old_state_one, new_state_one, np.random.default_rng(args.seed + 2), old_one, new_one, patch_size=4, rho=0.5)
    traj_rows = trajectory_debug(old_state_one, new_state_one, np.random.default_rng(args.seed + 3), old_one, new_one, args.tiny_sweeps, 4, 0.5, 20)
    write_csv(out / "target_distribution_debug_trajectory.csv", traj_rows)

    reference = {
        "config_actions": {
            "old_coarse": old["coarse_action"].as_dict,
            "old_fine": old["fine_action"].as_dict,
            "assembled_coarse": assembled["coarse_action"].as_dict,
            "assembled_fine": assembled["fine_action"].as_dict,
        },
        "lattice": old["config"]["lattice"],
        "kernel_old": old["kernel_json"],
        "kernel_assembled": assembled["kernel_json"],
        "coarse_manifest": old["coarse_manifest"],
        "fine_manifest": old["fine_manifest"],
    }
    coarse_check = coarse_distribution_check(old["coarse"], old["fine"], old["kernel"], old["coarse_action"], old["fine_action"])
    logweight_check = {
        "formula": "logw = -S_fine(phi) + S_coarse(u) + logdet_refine - (logq_edge + logq_pair + logq_corner)",
        "old_action_direct_minus_action_total": quantiles(old_state["sf_direct"] - old_state["sf"]),
        "assembled_action_direct_minus_action_total": quantiles(new_state["sf_direct"] - new_state["sf"]),
        "assembled_logq_component_stats": {
            "edge": quantiles(new_state["logq_edge"]),
            "pair": quantiles(new_state["logq_pair"]),
            "corner": quantiles(new_state["logq_corner"]),
            "total": quantiles(new_state["logq"]),
        },
        "old_logq_component_stats": {
            "edge": quantiles(old_state["logq_edge"]),
            "pair": quantiles(old_state["logq_pair"]),
            "corner": quantiles(old_state["logq_corner"]),
            "total": quantiles(old_state["logq"]),
        },
    }
    summary = {
        "initial_state_equivalence": initial,
        "one_coarse_proposal_equivalence": coarse_prop,
        "one_latent_pcn_equivalence": latent_prop,
        "tiny_trajectory": {
            "rows": len(traj_rows),
            "accept_decision_mismatches": int(sum(1 for r in traj_rows if not r["accept_same"])),
            "delta_delta_logw": quantiles(np.asarray([r["delta_delta_logw"] for r in traj_rows], dtype=np.float64)),
        },
        "logweight_correctness": logweight_check,
        "reference_consistency": reference,
        "coarse_distribution_check": coarse_check,
    }
    write_json(out / "target_distribution_debug_report.json", summary)
    md = [
        "# Shape-Parametric Target Distribution Debug",
        "",
        "## Initial-State Equivalence",
        f"- phi RMSE assembled-minus-old: `{initial['phi_rmse']:.6g}`",
        f"- psi RMSE assembled-minus-old: `{initial['psi_rmse']:.6g}`",
        f"- cprime RMSE: `{initial['cprime_rmse']:.6g}`",
        f"- detail RMSE by stage: `{initial['d_rmse_by_stage']}`",
        f"- delta S std: `{initial['delta_S']['std']:.6g}`",
        f"- delta logq total std: `{initial['delta_logq_total']['std']:.6g}`",
        f"- delta logweight std: `{initial['delta_logweight']['std']:.6g}`",
        f"- delta logq edge std: `{initial['delta_logq_edge']['std']:.6g}`",
        f"- delta logq pair std: `{initial['delta_logq_pair']['std']:.6g}`",
        f"- delta logq corner std: `{initial['delta_logq_corner']['std']:.6g}`",
        "",
        "## One-Proposal Equivalence",
        f"- coarse proposal old delta logw: `{coarse_prop['old_delta_logw']:.6g}`",
        f"- coarse proposal assembled delta logw: `{coarse_prop['assembled_delta_logw']:.6g}`",
        f"- coarse proposal delta-delta logw: `{coarse_prop['delta_delta_logw']:.6g}`",
        f"- latent pCN old delta logw: `{latent_prop['old_delta_logw']:.6g}`",
        f"- latent pCN assembled delta logw: `{latent_prop['assembled_delta_logw']:.6g}`",
        f"- latent pCN delta-delta logw: `{latent_prop['delta_delta_logw']:.6g}`",
        "",
        "## Tiny Trajectory",
        f"- logged rows: `{summary['tiny_trajectory']['rows']}`",
        f"- acceptance decision mismatches: `{summary['tiny_trajectory']['accept_decision_mismatches']}`",
        f"- delta-delta logw std: `{summary['tiny_trajectory']['delta_delta_logw']['std']:.6g}`",
        "",
        "## Logweight Correctness",
        f"- old independent action minus project action std: `{logweight_check['old_action_direct_minus_action_total']['std']:.6g}`",
        f"- assembled independent action minus project action std: `{logweight_check['assembled_action_direct_minus_action_total']['std']:.6g}`",
        "- logq formula includes edge, pair, and corner exactly once in this diagnostic.",
        "",
        "## Reference And Coarse Distribution",
        f"- action coarse/fine: `{reference['config_actions']}`",
        f"- lattice: `{reference['lattice']}`",
        f"- coarse direct observables: `{coarse_check['coarse_direct']}`",
        f"- blocked direct L16 cprime observables: `{coarse_check['blocked_direct_L16_cprime']}`",
        f"- blocked-minus-coarse observable delta: `{coarse_check['blocked_vs_coarse_delta']}`",
        "",
        "## Interpretation",
    ]
    if initial["delta_logweight"]["std"] > 1.0:
        md.append("- Initial old-vs-assembled equivalence already fails at meaningful scale; inspect component output/logq/reconstruction differences before trajectory logic.")
    elif summary["tiny_trajectory"]["accept_decision_mismatches"] > 0:
        md.append("- Initial equivalence is close, but trajectory decisions diverge; inspect Markov-state update/storage logic.")
    else:
        md.append("- Initial and tiny trajectory equivalence are close; if observables fail, compare old sampler under the same sector-aware setup and reference/coarse distribution.")
    md.append("- No new porting was performed.")
    (out / "target_distribution_debug_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"status": "completed", "report": str(out / "target_distribution_debug_report.md"), "initial_delta_logw_std": initial["delta_logweight"]["std"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
