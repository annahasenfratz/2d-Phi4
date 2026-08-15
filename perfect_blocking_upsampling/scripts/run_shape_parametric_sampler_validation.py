#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, asdict
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
# Keep the local package ahead of similarly named frozen-package paths.
sys.path.insert(0, str(PKG / "src"))
sys.path.insert(0, str(PKG / "scripts"))

from _common import load_config, load_ensembles, load_frozen_models, load_kernel_spec  # noqa: E402
from perfect_blocking_upsampling.actions import action_total  # noqa: E402
from perfect_blocking_upsampling.conv_pair import build_procedural_conv_flow  # noqa: E402
from perfect_blocking_upsampling.gathered_edge import build_gathered_edge_flow  # noqa: E402
from perfect_blocking_upsampling.kernels import apply_kernel, inverse_kernel  # noqa: E402
from perfect_blocking_upsampling.observables import observables as ensemble_observables  # noqa: E402
from train_finite_footprint_flow import local_observables, write_csv, write_json  # noqa: E402
from train_finite_footprint_transported_detail import (  # noqa: E402
    inner_patch_metropolis,
    max_abs_z,
    patch_sites,
    patches_per_sweep,
    random_origin_patch_schedule,
    schedule_preflight,
    summarize_moves,
)
from ML_sampling_clean.experiments.decimated_conditional_fillin.run_staged_decimated_conditional_fillin import (  # noqa: E402
    from_model_space,
    log_jacobian,
)

LOG2PI = math.log(2.0 * math.pi)
PRIMARY_OBSERVABLES = ["phi2", "phi4", "NN", "action_density", "susceptibility", "Binder_U4", "xi_over_L"]
SECTOR_DIAGNOSTICS = ["m", "abs_m"]


def parse_bool(text: str) -> bool:
    lowered = text.strip().lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"expected boolean true/false, got {text!r}")


@dataclass
class ValidationConfig:
    patch_size: int = 4
    origin_mode: str = "random"
    smoke_sweeps: int = 200
    validation_chains: int = 2
    pcn_rho: float = 0.5
    pcn_interval_sweeps: int = 1
    seed: int = 20260713
    ar_mode: str = "full_global_logweight"
    sector_balanced_init: bool = False
    progress_every_sweeps: int = 25
    measurement_mode: str = "end_of_sweep"
    coarse_start_mode: str = "thermalized_coarse"
    detail_warmup_sweeps: int = 0
    detail_warmup_fixed_coarse: bool = True
    detail_warmup_pcn_rho: float = 0.5
    measure_during_detail_warmup: bool = False
    measure_after_detail_warmup: bool = True


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


def stage_forward_from_z(model, z: np.ndarray, cond: np.ndarray, lg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
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
    return x.astype(np.float32), (logq_y - log_jacobian(cond, lg)).astype(np.float64)


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


def compute_state(u: np.ndarray, z_edge: np.ndarray, z_pair: np.ndarray, z_corner: np.ndarray, ctx: dict[str, Any]) -> dict[str, Any]:
    cprime, logdet = apply_refine_loaded(ctx["refine_model"], u)
    edge_model, edge_lg, _ = ctx["stages"]["edge"][:3]
    pair_model, pair_lg, _ = ctx["stages"]["pair"][:3]
    corner_model, corner_lg, _ = ctx["stages"]["corner"][:3]
    d10, l10 = stage_forward_from_z(edge_model, z_edge, cprime[:, None], edge_lg)
    d01, l01 = stage_forward_from_z(pair_model, z_pair, np.concatenate([cprime[:, None], d10], axis=1), pair_lg)
    d11, l11 = stage_forward_from_z(corner_model, z_corner, np.concatenate([cprime[:, None], d10, d01], axis=1), corner_lg)
    d = np.concatenate([d10, d01, d11], axis=1).astype(np.float32)
    psi = reconstruct(cprime, d)
    phi, inv = inverse_kernel(psi, ctx["kernel"])
    sf = action_total(phi, ctx["fine_action"])
    sc = action_total(u, ctx["coarse_action"])
    logq = l10 + l01 + l11
    logw = -sf + sc + logdet - logq
    return {
        "u": u.astype(np.float32),
        "z_edge": z_edge.astype(np.float32),
        "z_pair": z_pair.astype(np.float32),
        "z_corner": z_corner.astype(np.float32),
        "phi": phi.astype(np.float32),
        "sf": sf.astype(np.float64),
        "sc": sc.astype(np.float64),
        "logdet": logdet.astype(np.float64),
        "logq": logq.astype(np.float64),
        "logw": logw.astype(np.float64),
        "inv": inv,
    }


def propose_patch(state, x0, y0, tile, rng, ctx, cfg: ValidationConfig):
    sites = patch_sites(state["u"].shape[1], x0, y0, cfg.patch_size)
    u_new = state["u"][0].copy()
    u_new, inner_acc = inner_patch_metropolis(u_new, sites, rng)
    proposal = compute_state(u_new[None], state["z_edge"], state["z_pair"], state["z_corner"], ctx)
    delta_phi = proposal["phi"][0] - state["phi"][0]
    return proposal, {
        "patch_x": x0,
        "patch_y": y0,
        "tile": tile,
        "inner_acceptance": inner_acc,
        "delta_logw": float(proposal["logw"][0] - state["logw"][0]),
        "delta_Sf": float(proposal["sf"][0] - state["sf"][0]),
        "delta_Sc": float(proposal["sc"][0] - state["sc"][0]),
        "delta_logdet_refine": float(proposal["logdet"][0] - state["logdet"][0]),
        "delta_logq_missing": float(proposal["logq"][0] - state["logq"][0]),
        "changed_fine_sites_gt_1e-3": int(np.sum(np.abs(delta_phi) > 1.0e-3)),
    }


def propose_latent(state, x0, y0, tile, rng, ctx, cfg: ValidationConfig, rho: float | None = None):
    sites = patch_sites(state["u"].shape[1], x0, y0, cfg.patch_size)
    rho = cfg.pcn_rho if rho is None else rho
    noise = math.sqrt(max(0.0, 1.0 - rho * rho))
    z_edge = state["z_edge"].copy()
    z_pair = state["z_pair"].copy()
    z_corner = state["z_corner"].copy()
    for i, j in sites:
        z_edge[0, 0, i, j] = rho * z_edge[0, 0, i, j] + noise * float(rng.standard_normal())
        z_pair[0, 0, i, j] = rho * z_pair[0, 0, i, j] + noise * float(rng.standard_normal())
        z_corner[0, 0, i, j] = rho * z_corner[0, 0, i, j] + noise * float(rng.standard_normal())
    proposal = compute_state(state["u"], z_edge, z_pair, z_corner, ctx)
    delta_phi = proposal["phi"][0] - state["phi"][0]
    return proposal, {
        "patch_x": x0,
        "patch_y": y0,
        "tile": tile,
        "delta_logw": float(proposal["logw"][0] - state["logw"][0]),
        "delta_Sf": float(proposal["sf"][0] - state["sf"][0]),
        "delta_Sc": 0.0,
        "delta_logdet_refine": float(proposal["logdet"][0] - state["logdet"][0]),
        "delta_logq_missing": float(proposal["logq"][0] - state["logq"][0]),
        "changed_fine_sites_gt_1e-3": int(np.sum(np.abs(delta_phi) > 1.0e-3)),
    }


def reblocking_error(state: dict[str, Any], ctx: dict[str, Any]) -> float:
    blocked = apply_kernel(state["phi"], ctx["kernel"])
    cprime, _ = apply_refine_loaded(ctx["refine_model"], state["u"])
    return float(np.max(np.abs(blocked[:, 0::2, 0::2] - cprime)))


def run_fixed_coarse_detail_warmup(
    state: dict[str, Any],
    *,
    chain: int,
    rng: np.random.Generator,
    ctx: dict[str, Any],
    cfg: ValidationConfig,
    warmup_rows: list[dict[str, Any]],
    warmup_obs_rows: list[dict[str, Any]],
    warmup_check_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if cfg.detail_warmup_sweeps <= 0:
        return state
    if not cfg.detail_warmup_fixed_coarse:
        raise ValueError("detail warmup currently only supports fixed coarse warmup")
    initial_u = state["u"].copy()
    previous_state = state
    lc = state["u"].shape[1]
    for warm_sweep in range(cfg.detail_warmup_sweeps):
        schedule = random_origin_patch_schedule(lc, cfg.patch_size, rng, cfg.origin_mode)
        x0, y0, tile = schedule[-1]
        proposal, delta = propose_latent(state, x0, y0, tile, rng, ctx, cfg, rho=cfg.detail_warmup_pcn_rho)
        delta["delta_Sc"] = 0.0
        delta["delta_logdet_refine"] = 0.0
        delta["delta_logw"] = float(-delta["delta_Sf"] - delta["delta_logq_missing"])
        proposed_u_delta = float(np.max(np.abs(proposal["u"] - state["u"])))
        if proposed_u_delta != 0.0:
            raise RuntimeError(f"fixed-coarse warmup proposal changed u by {proposed_u_delta}")
        state, accept = apply_ar_update(state, proposal, delta["delta_logw"], math.log(max(rng.random(), 1.0e-300)))
        u_delta_from_initial = float(np.max(np.abs(state["u"] - initial_u)))
        if u_delta_from_initial != 0.0:
            raise RuntimeError(f"fixed-coarse warmup changed u by {u_delta_from_initial}")
        rejected_repeated_previous = (not accept) and (state is previous_state)
        row = {
            "move_type": "detail_warmup_latent",
            "chain_id": chain,
            "warmup_sweep": warm_sweep,
            "attempt_in_warmup_sweep": 0,
            "accepted": int(accept),
            "pcn_rho": cfg.detail_warmup_pcn_rho,
            "fixed_coarse": int(cfg.detail_warmup_fixed_coarse),
            "u_max_abs_delta_from_initial": u_delta_from_initial,
            "proposed_u_max_abs_delta": proposed_u_delta,
            "rejected_repeats_previous_state": int(rejected_repeated_previous),
            "conditional_logweight_note": "fixed_u_latent_pcn_delta_logw_full_fine_weight_same_coarse_no_coarse_proposal_terms",
            **delta,
        }
        warmup_rows.append(row)
        warmup_check_rows.append(
            {
                "chain_id": chain,
                "warmup_sweep": warm_sweep,
                "accepted": int(accept),
                "u_max_abs_delta_from_initial": u_delta_from_initial,
                "proposed_u_max_abs_delta": proposed_u_delta,
                "delta_Sc": float(delta["delta_Sc"]),
                "delta_logdet_refine": float(delta["delta_logdet_refine"]),
                "inverse_max_imag": float(state["inv"]["max_inverse_ifft_imag"]),
                "reblocking_max_abs_error": reblocking_error(state, ctx),
                "rejected_repeats_previous_state": int(rejected_repeated_previous),
            }
        )
        if cfg.measure_during_detail_warmup:
            warmup_obs_rows.append(
                measured_observable_row(
                    state,
                    chain=chain,
                    sweep=warm_sweep,
                    move_type="detail_warmup",
                    attempt_in_sweep=0,
                    update_index=warm_sweep,
                    accepted=accept,
                )
            )
        previous_state = state
    return state


def accept_logweight_delta(delta_logw: float, log_uniform: float) -> bool:
    return log_uniform < min(0.0, float(delta_logw))


def apply_ar_update(state: dict[str, Any], proposal: dict[str, Any], delta_logw: float, log_uniform: float) -> tuple[dict[str, Any], bool]:
    accepted = accept_logweight_delta(delta_logw, log_uniform)
    return (proposal if accepted else state), accepted


def measured_observable_row(
    state: dict[str, Any],
    *,
    chain: int,
    sweep: int,
    move_type: str,
    attempt_in_sweep: int,
    update_index: int,
    accepted: bool | None,
) -> dict[str, Any]:
    row = {
        "chain_id": chain,
        "sweep": sweep,
        "move_type": move_type,
        "attempt_in_sweep": attempt_in_sweep,
        "update_index": update_index,
        "accepted": "" if accepted is None else int(accepted),
        "measurement_semantics": "post_ar_markov_state",
    }
    row.update(local_observables(state["phi"]))
    return row


def expected_observable_measurements(cfg: ValidationConfig, n_patch_per_sweep: int) -> int:
    if cfg.measurement_mode == "end_of_sweep":
        return cfg.validation_chains * cfg.smoke_sweeps
    if cfg.measurement_mode == "every_attempt":
        return cfg.validation_chains * (
            cfg.smoke_sweeps * n_patch_per_sweep + (cfg.smoke_sweeps // cfg.pcn_interval_sweeps)
        )
    raise ValueError(f"unknown measurement_mode={cfg.measurement_mode!r}")


def expected_warmup_observable_measurements(cfg: ValidationConfig) -> int:
    if not cfg.measure_during_detail_warmup:
        return 0
    return cfg.validation_chains * cfg.detail_warmup_sweeps


def choose_initial_index(coarse: np.ndarray, chain: int, cfg: ValidationConfig, rng: np.random.Generator) -> tuple[int, str]:
    if cfg.coarse_start_mode != "thermalized_coarse":
        raise ValueError("choose_initial_index is only used for coarse_start_mode='thermalized_coarse'")
    if not cfg.sector_balanced_init:
        idx = int(rng.integers(0, len(coarse)))
        sign = "positive" if float(np.mean(coarse[idx])) >= 0.0 else "negative"
        return idx, f"random_{sign}"
    m = coarse.reshape(coarse.shape[0], -1).mean(axis=1)
    want_positive = chain % 2 == 0
    mask = (m >= 0.0) if want_positive else (m < 0.0)
    pool = np.flatnonzero(mask)
    if pool.size == 0:
        idx = int(rng.integers(0, len(coarse)))
        sign = "positive" if float(np.mean(coarse[idx])) >= 0.0 else "negative"
        return idx, f"fallback_random_{sign}"
    idx = int(pool[int(rng.integers(0, pool.size))])
    return idx, "positive" if want_positive else "negative"


def sign_label(x: float) -> str:
    if x > 0.0:
        return "positive"
    if x < 0.0:
        return "negative"
    return "zero"


def summarize_sector(obs_rows: list[dict[str, Any]], initial_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_chain: dict[int, list[dict[str, Any]]] = {}
    for row in obs_rows:
        by_chain.setdefault(int(row["chain_id"]), []).append(row)
    chain_rows = []
    total_pos = 0
    total_neg = 0
    total_zero = 0
    for chain, rows in sorted(by_chain.items()):
        signs = [sign_label(float(r["m"])) for r in rows]
        pos = sum(1 for s in signs if s == "positive")
        neg = sum(1 for s in signs if s == "negative")
        zero = sum(1 for s in signs if s == "zero")
        flips = sum(1 for a, b in zip(signs, signs[1:]) if a != b and a != "zero" and b != "zero")
        total_pos += pos
        total_neg += neg
        total_zero += zero
        chain_rows.append(
            {
                "chain_id": chain,
                "initial_coarse_m": next((r["initial_coarse_m"] for r in initial_rows if r["chain_id"] == chain), float("nan")),
                "initial_phi_m": next((r["initial_phi_m"] for r in initial_rows if r["chain_id"] == chain), float("nan")),
                "target_initial_sector": next((r["target_initial_sector"] for r in initial_rows if r["chain_id"] == chain), ""),
                "fraction_positive": pos / max(len(rows), 1),
                "fraction_negative": neg / max(len(rows), 1),
                "fraction_zero": zero / max(len(rows), 1),
                "sign_flips": flips,
                "mean_m": float(np.mean([float(r["m"]) for r in rows])),
                "mean_abs_m": float(np.mean([float(r["abs_m"]) for r in rows])),
            }
        )
    denom = max(total_pos + total_neg + total_zero, 1)
    return {
        "fraction_positive": total_pos / denom,
        "fraction_negative": total_neg / denom,
        "fraction_zero": total_zero / denom,
        "total_sign_flips": int(sum(r["sign_flips"] for r in chain_rows)),
        "per_chain": chain_rows,
    }


def observable_diagnostics(obs_rows: list[dict[str, Any]], fine_ref: np.ndarray, fine_action) -> dict[str, Any]:
    keys = ["m", "abs_m", "phi2", "phi4", "NN", "action_density", "Binder_U4", "susceptibility"]
    ref = ensemble_observables(fine_ref, fine_action)
    sample_arrays = {k: np.asarray([float(r[k]) for r in obs_rows], dtype=np.float64) for k in keys if k in obs_rows[0]}
    rows = {}
    for k, arr in sample_arrays.items():
        if k == "Binder_U4":
            # The per-row Binder_U4 is single-configuration degenerate. Use
            # ensemble Binder from the sampled magnetization moments below.
            continue
        ref_val = float(ref[k])
        ref_series = None
        if k == "m":
            ref_series = fine_ref.mean(axis=(1, 2))
        elif k == "abs_m":
            ref_series = np.abs(fine_ref.mean(axis=(1, 2)))
        elif k == "phi2":
            ref_series = np.mean(fine_ref.astype(np.float64) ** 2, axis=(1, 2))
        elif k == "phi4":
            ref_series = np.mean(fine_ref.astype(np.float64) ** 4, axis=(1, 2))
        elif k == "NN":
            f = fine_ref.astype(np.float64)
            ref_series = 0.5 * (np.mean(f * np.roll(f, -1, axis=1), axis=(1, 2)) + np.mean(f * np.roll(f, -1, axis=2), axis=(1, 2)))
        elif k == "susceptibility":
            m = fine_ref.mean(axis=(1, 2))
            ref_series = fine_ref.shape[1] * fine_ref.shape[2] * m * m
        elif k == "action_density":
            ref_series = action_total(fine_ref, fine_action) / (fine_ref.shape[1] * fine_ref.shape[2])
        if ref_series is None:
            ref_se = float("nan")
        else:
            ref_se = float(np.std(ref_series, ddof=1) / math.sqrt(len(ref_series)))
        mean = float(np.mean(arr))
        rows[k] = {"mean": mean, "reference": ref_val, "reference_se": ref_se, "z": float((mean - ref_val) / max(ref_se, 1.0e-300))}
    m = np.asarray([float(r["m"]) for r in obs_rows], dtype=np.float64)
    m2 = m * m
    m4 = m2 * m2
    phi2 = float(np.mean(sample_arrays["phi2"]))
    susceptibility = float(fine_ref.shape[1] * fine_ref.shape[2] * np.mean(m2))
    xi_over_l = float(np.sqrt(max(susceptibility, 0.0) / max(phi2, 1.0e-300)) / fine_ref.shape[1])
    binder = float(1.0 - np.mean(m4) / (3.0 * max(np.mean(m2) ** 2, 1.0e-300)))
    rows["Binder_U4"] = {"mean": binder, "reference": float(ref["Binder_U4"]), "reference_se": float("nan"), "z": float("nan")}
    rows["xi_over_L"] = {"mean": xi_over_l, "reference": float(ref["xi_over_L"]), "reference_se": float("nan"), "z": float("nan")}
    finite_primary_z = [
        abs(v["z"])
        for k, v in rows.items()
        if k in PRIMARY_OBSERVABLES and math.isfinite(v.get("z", float("nan")))
    ]
    finite_sector_z = [
        abs(v["z"])
        for k, v in rows.items()
        if k in SECTOR_DIAGNOSTICS and math.isfinite(v.get("z", float("nan")))
    ]
    return {
        "observables": rows,
        "primary_observables": PRIMARY_OBSERVABLES,
        "sector_diagnostics": SECTOR_DIAGNOSTICS,
        "max_abs_z_primary_observables": float(max(finite_primary_z)) if finite_primary_z else float("nan"),
        "max_abs_z_sector_diagnostics": float(max(finite_sector_z)) if finite_sector_z else float("nan"),
        # Legacy labels retained for older analysis readers. They should not be
        # used as pass/fail summaries.
        "max_abs_z_including_signed_m": float(max(finite_primary_z + finite_sector_z)) if finite_primary_z or finite_sector_z else float("nan"),
        "max_abs_z_excluding_signed_m": float(max(finite_primary_z)) if finite_primary_z else float("nan"),
    }


def dummy_full_bundle_preflight(bundle_dir: Path, coarse_l: int = 16) -> dict[str, Any]:
    import torch

    edge_ckpt = torch.load(bundle_dir / "edge.pt", map_location="cpu")
    pair_ckpt = torch.load(bundle_dir / "pair.pt", map_location="cpu")
    corner_ckpt = torch.load(bundle_dir / "corner.pt", map_location="cpu")
    edge_cfg = edge_ckpt["config"]
    edge = build_gathered_edge_flow(
        cond_channels=1,
        lattice_size=coarse_l,
        radius=int(edge_cfg.get("gather_radius", 3)),
        stencil=str(edge_cfg.get("gather_stencil", "square")),
        hidden_width=int(edge_cfg.get("gather_hidden_width", 96)),
        hidden_layers=int(edge_cfg.get("gather_hidden_layers", 2)),
        log_scale_bound=float(edge_cfg.get("log_scale_bound", 0.75)),
    )
    pair = build_procedural_conv_flow(
        cond_channels=2,
        lattice_size=coarse_l,
        n_coupling_layers=int(pair_ckpt["config"]["n_coupling_layers"]),
        conv_hidden_channels=int(pair_ckpt["config"]["conv_hidden_channels"]),
        log_scale_bound=float(pair_ckpt["config"]["log_scale_bound"]),
    )
    corner = build_procedural_conv_flow(
        cond_channels=3,
        lattice_size=coarse_l,
        n_coupling_layers=int(corner_ckpt["config"]["n_coupling_layers"]),
        conv_hidden_channels=int(corner_ckpt["config"]["conv_hidden_channels"]),
        log_scale_bound=float(corner_ckpt["config"]["log_scale_bound"]),
    )
    return {
        "dummy_L16_to_L32_instantiation": "passed",
        "edge": edge.dependency_report(),
        "pair": pair.dependency_report(),
        "corner": corner.dependency_report(),
    }


def run_validation(coarse: np.ndarray, fine_ref: np.ndarray, ctx: dict[str, Any], cfg: ValidationConfig, out: Path) -> dict[str, Any]:
    coarse_rows: list[dict[str, Any]] = []
    latent_rows: list[dict[str, Any]] = []
    warmup_rows: list[dict[str, Any]] = []
    warmup_obs_rows: list[dict[str, Any]] = []
    warmup_check_rows: list[dict[str, Any]] = []
    obs_rows: list[dict[str, Any]] = []
    initial_rows: list[dict[str, Any]] = []
    chain_summaries = []
    t0 = time.perf_counter()
    for chain in range(cfg.validation_chains):
        rng = np.random.default_rng(cfg.seed + 10000 * chain + 777)
        lc = coarse.shape[1]
        if cfg.coarse_start_mode == "thermalized_coarse":
            init_idx, target_sector = choose_initial_index(coarse, chain, cfg, rng)
            u = coarse[init_idx][None]
            initial_source = "thermalized_native_L8_coarse_ensemble"
        elif cfg.coarse_start_mode == "random_debug":
            init_idx = -1
            target_sector = "debug_random"
            u = rng.standard_normal((1, lc, lc)).astype(np.float32)
            initial_source = "random_debug_nonthermal_coarse_field"
        else:
            raise ValueError(f"unknown coarse_start_mode={cfg.coarse_start_mode!r}")
        z_edge, z_pair, z_corner = sample_z(rng, 1, lc)
        state = compute_state(u, z_edge, z_pair, z_corner, ctx)
        initial_rows.append(
            {
                "chain_id": chain,
                "coarse_index": init_idx,
                "initial_source": initial_source,
                "target_initial_sector": target_sector,
                "initial_coarse_m": float(np.mean(u)),
                "initial_phi_m": float(np.mean(state["phi"])),
                "initial_abs_phi_m": float(abs(np.mean(state["phi"]))),
            }
        )
        print(
            f"chain {chain} init: target={target_sector} coarse_m={np.mean(u):.6g} phi_m={np.mean(state['phi']):.6g}",
            flush=True,
        )
        state = run_fixed_coarse_detail_warmup(
            state,
            chain=chain,
            rng=rng,
            ctx=ctx,
            cfg=cfg,
            warmup_rows=warmup_rows,
            warmup_obs_rows=warmup_obs_rows,
            warmup_check_rows=warmup_check_rows,
        )
        if cfg.detail_warmup_sweeps > 0 and cfg.measure_after_detail_warmup:
            warmup_obs_rows.append(
                measured_observable_row(
                    state,
                    chain=chain,
                    sweep=cfg.detail_warmup_sweeps,
                    move_type="after_detail_warmup",
                    attempt_in_sweep=-1,
                    update_index=cfg.detail_warmup_sweeps,
                    accepted=None,
                )
            )
        chain_coarse: list[dict[str, Any]] = []
        chain_latent: list[dict[str, Any]] = []
        patch_counter = 0
        update_index = 0
        for sweep in range(cfg.smoke_sweeps):
            schedule = random_origin_patch_schedule(lc, cfg.patch_size, rng, cfg.origin_mode)
            schedule_len = len(schedule)
            for attempt, (x0, y0, tile) in enumerate(schedule):
                proposal, delta = propose_patch(state, x0, y0, tile, rng, ctx, cfg)
                state, accept = apply_ar_update(state, proposal, delta["delta_logw"], math.log(max(rng.random(), 1.0e-300)))
                row = {"move_type": "coarse", "chain_id": chain, "sweep": sweep, "attempt_in_sweep": attempt, "accepted": int(accept), **delta}
                coarse_rows.append(row)
                chain_coarse.append(row)
                if cfg.measurement_mode == "every_attempt":
                    obs_rows.append(
                        measured_observable_row(
                            state,
                            chain=chain,
                            sweep=sweep,
                            move_type="coarse",
                            attempt_in_sweep=attempt,
                            update_index=update_index,
                            accepted=accept,
                        )
                    )
                    update_index += 1
                patch_counter += 1
                if patch_counter % schedule_len == 0 and (sweep + 1) % cfg.pcn_interval_sweeps == 0:
                    proposal_l, delta_l = propose_latent(state, x0, y0, tile, rng, ctx, cfg)
                    state, accept_l = apply_ar_update(state, proposal_l, delta_l["delta_logw"], math.log(max(rng.random(), 1.0e-300)))
                    lrow = {"move_type": "latent", "chain_id": chain, "sweep": sweep, "attempt_in_sweep": attempt, "accepted": int(accept_l), **delta_l}
                    latent_rows.append(lrow)
                    chain_latent.append(lrow)
                    if cfg.measurement_mode == "every_attempt":
                        obs_rows.append(
                            measured_observable_row(
                                state,
                                chain=chain,
                                sweep=sweep,
                                move_type="latent",
                                attempt_in_sweep=attempt,
                                update_index=update_index,
                                accepted=accept_l,
                            )
                        )
                        update_index += 1
            if cfg.measurement_mode == "end_of_sweep":
                obs_rows.append(
                    measured_observable_row(
                        state,
                        chain=chain,
                        sweep=sweep,
                        move_type="end_of_sweep",
                        attempt_in_sweep=-1,
                        update_index=update_index,
                        accepted=None,
                    )
                )
                update_index += 1
            if cfg.progress_every_sweeps > 0 and ((sweep + 1) % cfg.progress_every_sweeps == 0 or sweep + 1 == cfg.smoke_sweeps):
                progress = {
                    "chain_id": chain,
                    "completed_sweeps": sweep + 1,
                    "total_sweeps": cfg.smoke_sweeps,
                    "coarse_attempts_so_far": len(chain_coarse),
                    "latent_attempts_so_far": len(chain_latent),
                    "current_m": float(np.mean(state["phi"])),
                    "current_abs_m": float(abs(np.mean(state["phi"]))),
                }
                write_json(out / "progress.json", progress)
                print(
                    f"chain {chain} sweep {sweep + 1}/{cfg.smoke_sweeps}: "
                    f"coarse_attempts={len(chain_coarse)} latent_attempts={len(chain_latent)} "
                    f"m={progress['current_m']:.6g}",
                    flush=True,
                )
        chain_summary = {
            "chain_id": chain,
            **summarize_moves([r for r in warmup_rows if int(r["chain_id"]) == chain], "detail_warmup"),
            **summarize_moves(chain_coarse, "coarse"),
            **summarize_moves(chain_latent, "latent"),
        }
        chain_summaries.append(chain_summary)
        print(
            f"chain {chain}: coarse_acc={chain_summary['coarse_acceptance']:.6g} "
            f"coarse_std={chain_summary['coarse_std_delta_logw']:.6g} "
            f"latent_acc={chain_summary['latent_acceptance']:.6g} "
            f"latent_std={chain_summary['latent_std_delta_logw']:.6g}",
            flush=True,
        )
    wall = time.perf_counter() - t0
    write_csv(out / "detail_warmup_deltas.csv", warmup_rows)
    write_csv(out / "detail_warmup_observable_timeseries.csv", warmup_obs_rows)
    write_csv(out / "detail_warmup_preflight_checks.csv", warmup_check_rows)
    write_csv(out / "coarse_deltas.csv", coarse_rows)
    write_csv(out / "latent_deltas.csv", latent_rows)
    write_csv(out / "observable_timeseries.csv", obs_rows)
    write_csv(out / "chain_summaries.csv", chain_summaries)
    write_csv(out / "initial_chain_states.csv", initial_rows)
    sector = summarize_sector(obs_rows, initial_rows)
    obs_diag = observable_diagnostics(obs_rows, fine_ref, ctx["fine_action"])
    write_json(out / "sector_occupancy.json", sector)
    write_json(out / "observable_diagnostics.json", obs_diag)
    summary = {
        "wall_time_sec": wall,
        "observable_measurement_semantics": {
            "measurement_mode": cfg.measurement_mode,
            "cadence": "after_every_attempted_update" if cfg.measurement_mode == "every_attempt" else "end_of_sweep",
            "state": "post_ar_markov_state",
            "rejected_updates": "repeat_previous_state",
            "end_of_sweep_note": "records the current chain state after all accepted/rejected updates in the sweep",
            "accepted_only_observable_table": None,
            "measured_states_total": len(obs_rows),
            "measured_states_per_chain": {
                str(chain): sum(1 for row in obs_rows if int(row["chain_id"]) == chain)
                for chain in range(cfg.validation_chains)
            },
        },
        "detail_warmup": {
            "enabled": cfg.detail_warmup_sweeps > 0,
            "sweeps": cfg.detail_warmup_sweeps,
            "fixed_coarse": cfg.detail_warmup_fixed_coarse,
            "pcn_rho": cfg.detail_warmup_pcn_rho,
            "warmup_sweep_convention": "one same-patch latent pCN attempt per warmup sweep; random-origin schedule is generated only to choose the final same-patch location",
            "measure_during_detail_warmup": cfg.measure_during_detail_warmup,
            "measure_after_detail_warmup": cfg.measure_after_detail_warmup,
            "warmup_attempts_total": len(warmup_rows),
            "warmup_observable_rows": len(warmup_obs_rows),
            "warmup_acceptance": float(np.mean([int(r["accepted"]) for r in warmup_rows])) if warmup_rows else float("nan"),
            "warmup_std_delta_logw": float(np.std([float(r["delta_logw"]) for r in warmup_rows], ddof=1)) if len(warmup_rows) > 1 else float("nan"),
            "max_u_abs_delta": float(max([float(r["u_max_abs_delta_from_initial"]) for r in warmup_rows], default=0.0)),
            "max_reblocking_error": float(max([float(r["reblocking_max_abs_error"]) for r in warmup_check_rows], default=0.0)),
            "max_inverse_ifft_imag": float(max([float(r["inverse_max_imag"]) for r in warmup_check_rows], default=0.0)),
        },
        "initialization_policy": {
            "validation_mode": "native_L8_deployment_full_coarse_update",
            "coarse_start_mode": cfg.coarse_start_mode,
            "coarse_patch_updates_enabled": True,
            "conditional_blocked_coarse_fixed_u": False,
            "thermalized_coarse_definition": "sample from configured data.coarse_ensemble",
            "random_debug_note": "random_debug starts are nonthermal and must not be used for pass/fail upscaler validation",
            "burn_in_discarded_sweeps": 0,
            "detail_warmup_discarded_from_production_observables": True,
            "mode_distinction": "Do not use this full coarse-update driver as the fixed blocked-coarse conditional validation; that test should keep u fixed and report no coarse patch acceptance.",
        },
        "max_abs_observable_z": obs_diag["max_abs_z_primary_observables"],
        "max_abs_primary_observable_z": obs_diag["max_abs_z_primary_observables"],
        "max_abs_sector_diagnostic_z": obs_diag["max_abs_z_sector_diagnostics"],
        "sector_occupancy": sector,
        "observable_diagnostics": obs_diag,
        **summarize_moves(warmup_rows, "detail_warmup"),
        **summarize_moves(coarse_rows, "coarse"),
        **summarize_moves(latent_rows, "latent"),
        "chain_summaries": chain_summaries,
    }
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=PKG / "outputs" / "procedural_corner_diagnostics" / "old_pair_corner_procedural_masks.yaml")
    ap.add_argument("--output-dir", type=Path, default=PKG / "outputs" / "shape_parametric_sampler_validation" / "smoke_2x200")
    ap.add_argument("--coarse-L", type=int, default=8)
    ap.add_argument("--patch-size", type=int, default=4)
    ap.add_argument("--origin-mode", choices=["random", "deterministic_mixing"], default="random")
    ap.add_argument("--smoke-sweeps", type=int, default=200)
    ap.add_argument("--validation-chains", type=int, default=2)
    ap.add_argument("--pcn-rho", type=float, default=0.5)
    ap.add_argument("--pcn-interval-sweeps", type=int, default=1)
    ap.add_argument("--seed", type=int, default=20260713)
    ap.add_argument("--sector-balanced-init", action="store_true")
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--progress-every-sweeps", type=int, default=25)
    ap.add_argument("--measurement-mode", choices=["end_of_sweep", "every_attempt"], default="end_of_sweep")
    ap.add_argument("--coarse-start-mode", choices=["thermalized_coarse", "random_debug"], default="thermalized_coarse")
    ap.add_argument("--detail-warmup-sweeps", type=int, default=0)
    ap.add_argument("--detail-warmup-fixed-coarse", type=parse_bool, default=True)
    ap.add_argument("--detail-warmup-pcn-rho", type=float, default=0.5)
    ap.add_argument("--measure-during-detail-warmup", type=parse_bool, default=False)
    ap.add_argument("--measure-after-detail-warmup", type=parse_bool, default=True)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg = ValidationConfig(
        patch_size=args.patch_size,
        origin_mode=args.origin_mode,
        smoke_sweeps=args.smoke_sweeps,
        validation_chains=args.validation_chains,
        pcn_rho=args.pcn_rho,
        pcn_interval_sweeps=args.pcn_interval_sweeps,
        seed=args.seed,
        sector_balanced_init=args.sector_balanced_init,
        progress_every_sweeps=args.progress_every_sweeps,
        measurement_mode=args.measurement_mode,
        coarse_start_mode=args.coarse_start_mode,
        detail_warmup_sweeps=args.detail_warmup_sweeps,
        detail_warmup_fixed_coarse=args.detail_warmup_fixed_coarse,
        detail_warmup_pcn_rho=args.detail_warmup_pcn_rho,
        measure_during_detail_warmup=args.measure_during_detail_warmup,
        measure_after_detail_warmup=args.measure_after_detail_warmup,
    )
    if cfg.detail_warmup_sweeps < 0:
        raise ValueError("--detail-warmup-sweeps must be nonnegative")
    if cfg.detail_warmup_sweeps > 0 and not cfg.detail_warmup_fixed_coarse:
        raise ValueError("detail warmup is implemented only for --detail-warmup-fixed-coarse true")
    if cfg.detail_warmup_sweeps > 0 and args.coarse_L != 8:
        raise ValueError("detail warmup is currently enabled only for the reviewed L8->L16 baseline")
    if not (0.0 <= cfg.detail_warmup_pcn_rho <= 1.0):
        raise ValueError("--detail-warmup-pcn-rho must be in [0, 1]")
    loaded_cfg = load_config(args.config)
    coarse, fine_ref, _, _, paths = load_ensembles(loaded_cfg)
    if coarse.shape[1] != args.coarse_L:
        raise ValueError(f"loaded coarse L={coarse.shape[1]} but --coarse-L={args.coarse_L}")
    refine_model, refine_state, stages, coarse_action, fine_action, _ = load_frozen_models(loaded_cfg)
    refine_model.load_state_dict(refine_state, strict=False)
    refine_model.eval()
    for model, _, state, *_ in stages.values():
        model.load_state_dict(state)
        model.eval()
    kernel, kernel_json = load_kernel_spec(loaded_cfg)
    bundle_preflight = dummy_full_bundle_preflight(paths["frozen_dir"], coarse_l=16)
    n_patch = patches_per_sweep(args.coarse_L, cfg.patch_size)
    scheduler_preflight = {
        "L_c": args.coarse_L,
        "patch_size": cfg.patch_size,
        "N_patch_per_sweep": n_patch,
        "origin_mode": cfg.origin_mode,
        "protocol_label": "random_first_origin" if cfg.origin_mode == "random" else "deterministic_mixing",
        "ar_mode": cfg.ar_mode,
        "pcn_rho": cfg.pcn_rho,
        "pcn_interval_sweeps": cfg.pcn_interval_sweeps,
        "detail_warmup": {
            "sweeps": cfg.detail_warmup_sweeps,
            "fixed_coarse": cfg.detail_warmup_fixed_coarse,
            "pcn_rho": cfg.detail_warmup_pcn_rho,
            "measure_during_detail_warmup": cfg.measure_during_detail_warmup,
            "measure_after_detail_warmup": cfg.measure_after_detail_warmup,
            "warmup_sweep_convention": "one same-patch latent pCN attempt per warmup sweep; random-origin schedule is generated only to choose the final same-patch location",
            "expected_warmup_attempts": cfg.validation_chains * cfg.detail_warmup_sweeps,
            "expected_warmup_observable_measurements": expected_warmup_observable_measurements(cfg)
            + (cfg.validation_chains if cfg.detail_warmup_sweeps > 0 and cfg.measure_after_detail_warmup else 0),
        },
        "expected_coarse_attempts": cfg.validation_chains * cfg.smoke_sweeps * n_patch,
        "expected_latent_pcn_attempts": cfg.validation_chains * (cfg.smoke_sweeps // cfg.pcn_interval_sweeps),
        "expected_observable_measurements": expected_observable_measurements(cfg, n_patch),
        "observable_measurement_semantics": {
            "measurement_mode": cfg.measurement_mode,
            "cadence": "after_every_attempted_update" if cfg.measurement_mode == "every_attempt" else "end_of_sweep",
            "state": "post_ar_markov_state",
            "rejected_updates": "repeat_previous_state",
            "end_of_sweep_note": "records the current chain state after all accepted/rejected updates in the sweep",
            "expected_rows_end_of_sweep": cfg.validation_chains * cfg.smoke_sweeps,
            "expected_rows_every_attempt": cfg.validation_chains
            * (cfg.smoke_sweeps * n_patch + (cfg.smoke_sweeps // cfg.pcn_interval_sweeps)),
            "accepted_only_observable_table": None,
        },
        "initialization_policy": {
            "validation_mode": "native_L8_deployment_full_coarse_update",
            "coarse_start_mode": cfg.coarse_start_mode,
            "coarse_patch_updates_enabled": True,
            "conditional_blocked_coarse_fixed_u": False,
            "thermalized_coarse_definition": "sample from configured data.coarse_ensemble",
            "random_debug_note": "random_debug starts are nonthermal and must not be used for pass/fail upscaler validation",
            "burn_in_discarded_sweeps": 0,
            "detail_warmup_sweeps": cfg.detail_warmup_sweeps,
            "detail_warmup_fixed_coarse": cfg.detail_warmup_fixed_coarse,
            "detail_warmup_excluded_from_production_observables": True,
            "mode_distinction": "Do not use this full coarse-update driver as the fixed blocked-coarse conditional validation; that test should keep u fixed and report no coarse patch acceptance.",
        },
        "coverage_preflight": schedule_preflight(args.coarse_L, cfg.patch_size, cfg.origin_mode, n_sweeps=256, seed=cfg.seed),
    }
    run_config = {
        "description": "shape-parametric sampler validation",
        "bundle_config": str(args.config),
        "output_dir": str(args.output_dir),
        "coarse_L": args.coarse_L,
        "fine_L": 2 * args.coarse_L,
        "patch_size": cfg.patch_size,
        "origin_mode": cfg.origin_mode,
        "protocol_label": scheduler_preflight["protocol_label"],
        "validation_chains": cfg.validation_chains,
        "sweeps": cfg.smoke_sweeps,
        "pcn_rho": cfg.pcn_rho,
        "pcn_interval_sweeps": cfg.pcn_interval_sweeps,
        "detail_warmup_sweeps": cfg.detail_warmup_sweeps,
        "detail_warmup_fixed_coarse": cfg.detail_warmup_fixed_coarse,
        "detail_warmup_pcn_rho": cfg.detail_warmup_pcn_rho,
        "measure_during_detail_warmup": cfg.measure_during_detail_warmup,
        "measure_after_detail_warmup": cfg.measure_after_detail_warmup,
        "ar_mode": cfg.ar_mode,
        "sector_balanced_init": cfg.sector_balanced_init,
        "seed": cfg.seed,
        "validation_mode": "native_L8_deployment_full_coarse_update",
        "measurement_mode": cfg.measurement_mode,
        "coarse_start_mode": cfg.coarse_start_mode,
        "expected": {
            "N_patch_per_sweep": n_patch,
            "coarse_attempts": scheduler_preflight["expected_coarse_attempts"],
            "latent_pcn_attempts": scheduler_preflight["expected_latent_pcn_attempts"],
            "detail_warmup_attempts": scheduler_preflight["detail_warmup"]["expected_warmup_attempts"],
            "detail_warmup_observable_rows": scheduler_preflight["detail_warmup"]["expected_warmup_observable_measurements"],
            "observable_rows": scheduler_preflight["expected_observable_measurements"],
        },
    }
    write_json(args.output_dir / "run_config.json", run_config)
    write_json(args.output_dir / "scheduler_preflight.json", scheduler_preflight)
    write_json(args.output_dir / "dummy_l16_l32_preflight.json", bundle_preflight)
    print(json.dumps({"scheduler_preflight": scheduler_preflight, "bundle_preflight": bundle_preflight}, indent=2), flush=True)
    if args.preflight_only:
        payload = {
            "status": "preflight_completed",
            "config": asdict(cfg),
            "bundle_config": str(args.config),
            "kernel": kernel_json,
            "scheduler_preflight": scheduler_preflight,
            "dummy_l16_l32_preflight": bundle_preflight,
            "component_distinction": {
                "coarse_refine": "portable distilled coarse-refine",
                "edge": "accepted gathered strict finite-radius r_c=3, r_f=6",
                "pair": "old weights with procedural shape-parametric masks; not strict finite-radius",
                "corner": "old weights with procedural shape-parametric masks; not strict finite-radius",
            },
        }
        write_json(args.output_dir / "preflight_summary.json", payload)
        return 0
    ctx = {
        "refine_model": refine_model,
        "stages": stages,
        "coarse_action": coarse_action,
        "fine_action": fine_action,
        "kernel": kernel,
    }
    result = run_validation(coarse, fine_ref, ctx, cfg, args.output_dir)
    result["actual_coarse_attempts"] = result["coarse_attempts"]
    result["actual_latent_pcn_attempts"] = result["latent_attempts"]
    summary = {
        "status": "completed",
        "config": asdict(cfg),
        "bundle_config": str(args.config),
        "kernel": kernel_json,
        "scheduler_preflight": scheduler_preflight,
        "dummy_l16_l32_preflight": bundle_preflight,
        "result": result,
        "component_distinction": {
            "coarse_refine": "portable distilled coarse-refine",
            "edge": "accepted gathered strict finite-radius r_c=3, r_f=6",
            "pair": "old weights with procedural shape-parametric masks; not strict finite-radius",
            "corner": "old weights with procedural shape-parametric masks; not strict finite-radius",
        },
    }
    write_json(args.output_dir / "summary.json", summary)
    lines = [
        "# Shape-Parametric Transported-Detail Sampler Validation",
        "",
        f"- protocol: `{scheduler_preflight['protocol_label']}`",
        f"- ar_mode: `{cfg.ar_mode}`",
        f"- patch_size: `{cfg.patch_size}`",
        f"- coarse_L: `{args.coarse_L}`",
        f"- N_patch/sweep: `{n_patch}`",
        f"- pCN rho / interval: `{cfg.pcn_rho}` / `{cfg.pcn_interval_sweeps}` sweeps",
        f"- detail warmup sweeps: `{cfg.detail_warmup_sweeps}`",
        f"- detail warmup fixed coarse: `{cfg.detail_warmup_fixed_coarse}`",
        f"- detail warmup pCN rho: `{cfg.detail_warmup_pcn_rho}`",
        "- detail warmup sweep convention: one same-patch latent pCN attempt per warmup sweep.",
        f"- expected/actual coarse attempts: `{scheduler_preflight['expected_coarse_attempts']}` / `{result['coarse_attempts']}`",
        f"- expected/actual detail warmup attempts: `{scheduler_preflight['detail_warmup']['expected_warmup_attempts']}` / `{result['detail_warmup_attempts']}`",
        f"- expected/actual latent attempts: `{scheduler_preflight['expected_latent_pcn_attempts']}` / `{result['latent_attempts']}`",
        f"- sector-balanced init: `{cfg.sector_balanced_init}`",
        f"- validation_mode: `native_L8_deployment_full_coarse_update`",
        f"- coarse_start_mode: `{cfg.coarse_start_mode}`",
        f"- coarse patch updates enabled: `True`",
        f"- burn-in discarded sweeps: `0`",
        "",
        "## A/R",
        f"- detail warmup acceptance: `{result['detail_warmup_acceptance']:.6g}`",
        f"- detail warmup Delta logw std: `{result['detail_warmup_std_delta_logw']:.6g}`",
        f"- coarse acceptance: `{result['coarse_acceptance']:.6g}`",
        f"- coarse Delta logw std: `{result['coarse_std_delta_logw']:.6g}`",
        f"- latent pCN acceptance: `{result['latent_acceptance']:.6g}`",
        f"- latent Delta logw std: `{result['latent_std_delta_logw']:.6g}`",
        f"- max abs primary-observable z-score: `{result['max_abs_primary_observable_z']:.6g}`",
        f"- max abs sector-diagnostic z-score (not pass/fail): `{result['max_abs_sector_diagnostic_z']:.6g}`",
        f"- wall time seconds: `{result['wall_time_sec']:.6g}`",
        "",
        "## Sector Occupancy",
        f"- fraction positive: `{result['sector_occupancy']['fraction_positive']:.6g}`",
        f"- fraction negative: `{result['sector_occupancy']['fraction_negative']:.6g}`",
        f"- total sign flips: `{result['sector_occupancy']['total_sign_flips']}`",
        "",
        "## Primary Observables",
        f"- phi2: `{result['observable_diagnostics']['observables']['phi2']['mean']:.6g}`",
        f"- phi4: `{result['observable_diagnostics']['observables']['phi4']['mean']:.6g}`",
        f"- NN: `{result['observable_diagnostics']['observables']['NN']['mean']:.6g}`",
        f"- action density: `{result['observable_diagnostics']['observables']['action_density']['mean']:.6g}`",
        f"- Binder_U4: `{result['observable_diagnostics']['observables']['Binder_U4']['mean']:.6g}`",
        f"- susceptibility: `{result['observable_diagnostics']['observables']['susceptibility']['mean']:.6g}`",
        f"- xi/L: `{result['observable_diagnostics']['observables']['xi_over_L']['mean']:.6g}`",
        "",
        "## Sector Diagnostics (Not Pass/Fail)",
        f"- signed m: `{result['observable_diagnostics']['observables']['m']['mean']:.6g}`",
        f"- |m|: `{result['observable_diagnostics']['observables']['abs_m']['mean']:.6g}`",
        "",
        "## Portability Distinction",
        "- edge: strict finite-radius gathered component (`r_c=3`, `r_f=6`).",
        "- pair/corner: old circular-conv architecture with procedural shape-parametric masks; not strict finite-radius unless separately proven.",
    ]
    (args.output_dir / "shape_parametric_sampler_validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "completed", "output": str(args.output_dir), "result": result}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
