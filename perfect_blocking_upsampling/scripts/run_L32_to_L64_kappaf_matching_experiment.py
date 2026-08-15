#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PKG = PROJECT_ROOT / "perfect_blocking_upsampling"
sys.path.insert(0, str(PKG / "scripts"))
sys.path.insert(0, str(PKG / "src"))

import run_shape_parametric_sampler_validation as sampler  # noqa: E402
from _common import format_float_tag, load_config, load_frozen_models, load_kernel_spec, override_validation_config, resolve_run_paths  # noqa: E402
from perfect_blocking_upsampling.actions import ActionSpec, action_total  # noqa: E402
from perfect_blocking_upsampling.observables import observables as ensemble_observables  # noqa: E402
from train_finite_footprint_transported_detail import inner_patch_metropolis, patch_sites  # noqa: E402


DEFAULT_CONFIG = PKG / "outputs" / "shape_parametric_sampler_validation" / "L32_to_L64_smoke" / "L32_to_L64_smoke_config.yaml"
DEFAULT_OUT = None
KAPPA_FS = [0.27050, 0.27075, 0.27100, 0.27125]
SAVE_SWEEPS = [0, 1, 2, 5, 10] + list(range(20, 1001, 20))
LOCAL_KEYS = ["m2", "m4", "chi", "Binder_U4", "xi_over_L", "NN", "diag", "2nn", "action_density"]


def validate_patch_size_for_scan(lc: int, patch_size: int) -> None:
    if patch_size % 2 != 0:
        raise ValueError(f"patch size must be even, got {patch_size}")
    if patch_size < 2 or patch_size > lc:
        raise ValueError(f"patch size must satisfy 2 <= P <= Lc, got P={patch_size}, Lc={lc}")


def patches_per_sweep(lc: int, patch_size: int) -> int:
    validate_patch_size_for_scan(lc, patch_size)
    return int(math.ceil(2.0 * lc * lc / float(patch_size * patch_size)))


def random_origin_patch_schedule(lc: int, patch_size: int, rng: np.random.Generator, origin_mode: str = "random") -> list[tuple[int, int, str]]:
    validate_patch_size_for_scan(lc, patch_size)
    n_patches = patches_per_sweep(lc, patch_size)
    if origin_mode not in {"random", "deterministic_mixing"}:
        raise ValueError(f"unknown origin_mode={origin_mode!r}")
    start_x = int(rng.integers(0, lc))
    start_y = int(rng.integers(0, lc))
    origins = [(start_x, start_y)]
    if origin_mode == "random":
        for _ in range(1, n_patches):
            origins.append((int(rng.integers(0, lc)), int(rng.integers(0, lc))))
    else:
        sx = max(1, patch_size // 2)
        sy = max(1, patch_size // 2 + 1)
        for k in range(1, n_patches):
            origins.append(((start_x + k * sx) % lc, (start_y + k * sy) % lc))
    return [(int(x), int(y), f"patch_{idx}_origin_{int(x)}_{int(y)}") for idx, (x, y) in enumerate(origins)]


def rho_from_latent_beta_scale(beta_scale: float) -> float:
    base_beta = math.sqrt(1.0 - 0.5 * 0.5)
    beta = float(beta_scale) * base_beta
    if not (0.0 <= beta <= 1.0):
        raise ValueError(f"latent beta scale gives invalid beta={beta}")
    return math.sqrt(max(0.0, 1.0 - beta * beta))


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


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=float) + "\n", encoding="utf-8")


def qstats(vals: list[float] | np.ndarray) -> dict[str, float]:
    arr = np.asarray(vals, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"mean": float("nan"), "std": float("nan"), "q05": float("nan"), "q50": float("nan"), "q95": float("nan"), "n": 0}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "q05": float(np.quantile(arr, 0.05)),
        "q50": float(np.quantile(arr, 0.50)),
        "q95": float(np.quantile(arr, 0.95)),
        "n": int(len(arr)),
    }


def load_coarse(cfg: dict[str, Any]) -> tuple[np.ndarray, Path, dict[str, Any]]:
    path = resolve_run_paths(cfg)["coarse_ensemble"]
    arr = np.load(path)["phi"].astype(np.float32)
    manifest_path = path.with_name("manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    return arr, path, manifest


def load_ctx(cfg: dict[str, Any]) -> tuple[dict[str, Any], ActionSpec, ActionSpec]:
    refine_model, refine_state, stages, coarse_action, fine_action, _ = load_frozen_models(cfg)
    refine_model.load_state_dict(refine_state, strict=False)
    refine_model.eval()
    for model, _, state, *_ in stages.values():
        model.load_state_dict(state)
        model.eval()
    kernel, _ = load_kernel_spec(cfg)
    ctx = {"refine_model": refine_model, "stages": stages, "coarse_action": coarse_action, "fine_action": fine_action, "kernel": kernel}
    return ctx, coarse_action, fine_action


def build_run_config(base_cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    run_name = args.run_name or (
        f"lam{format_float_tag(args.lambda_)}_kappa{format_float_tag(args.kappa_c)}_"
        f"L{args.Lc}_to_L{args.Lf}_kappaf_matching"
    )
    return override_validation_config(
        base_cfg,
        coarse_L=args.Lc,
        fine_L=args.Lf,
        lambda_c=args.lambda_,
        lambda_f=args.lambda_,
        kappa_c=args.kappa_c,
        coarse_ensemble=args.coarse_ensemble,
        fine_reference=args.fine_reference,
        run_name=run_name,
        output_dir=args.out_dir,
    )


def ctx_for_kappaf(ctx_base: dict[str, Any], kappa_f: float) -> dict[str, Any]:
    ctx = dict(ctx_base)
    fa = ctx_base["fine_action"]
    ctx["fine_action"] = replace(fa, kappa=float(kappa_f))
    return ctx


def local_observable_row(phi: np.ndarray, action: ActionSpec) -> dict[str, float]:
    arr = np.asarray(phi, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr[None]
    obs = ensemble_observables(arr, action)
    m = arr.mean(axis=(1, 2))
    m2 = float(np.mean(m * m))
    m4 = float(np.mean(m**4))
    phi2 = float(np.mean(arr * arr))
    two_nn = 0.5 * (
        np.mean(arr * np.roll(arr, -2, axis=1), axis=(1, 2))
        + np.mean(arr * np.roll(arr, -2, axis=2), axis=(1, 2))
    )
    action_density = action_total(arr, action) / (arr.shape[1] * arr.shape[2])
    return {
        "m2": m2,
        "m4": m4,
        "chi": float(arr.shape[1] * arr.shape[2] * m2),
        "Binder_U4": float(1.0 - m4 / (3.0 * m2 * m2)) if m2 > 0 else float("nan"),
        "xi_over_L": float(math.sqrt(max(m2, 0.0) / max(phi2, 1e-300))),
        "NN": float(obs["NN"]),
        "diag": float(obs["diag"]),
        "2nn": float(np.mean(two_nn)),
        "action_density": float(np.mean(action_density)),
        "phi2": float(obs["phi2"]),
        "phi4": float(obs["phi4"]),
        "m": float(obs["m"]),
        "abs_m": float(obs["abs_m"]),
    }


def state_decomposition_row(state: dict[str, Any], kappa_f: float, chain: int, sweep: int, state_type: str, coarse_L: int, fine_L: int) -> dict[str, Any]:
    return {
        "kappa_f": kappa_f,
        "chain_id": chain,
        "sweep": sweep,
        "state_type": state_type,
        "Sf": float(state["sf"][0]),
        "minus_Sf": float(-state["sf"][0]),
        "Sc": float(state["sc"][0]),
        "logdet_refine": float(state["logdet"][0]),
        "logq_detail": float(state["logq"][0]),
        "minus_logq_detail": float(-state["logq"][0]),
        "logw": float(state["logw"][0]),
        "Sf_per_site": float(state["sf"][0] / (fine_L * fine_L)),
        "Sc_per_site": float(state["sc"][0] / (coarse_L * coarse_L)),
        "logw_per_site": float(state["logw"][0] / (fine_L * fine_L)),
    }


def split_batch_state(batch: dict[str, Any], i: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, val in batch.items():
        if isinstance(val, np.ndarray) and val.shape[:1] == (batch["u"].shape[0],):
            out[key] = val[i : i + 1].copy()
        else:
            out[key] = val
    return out


def stack_state_field(states: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.concatenate([s[key] for s in states], axis=0)


def propose_patch_scaled(state: dict[str, Any], x0: int, y0: int, tile: str, rng: np.random.Generator, ctx: dict[str, Any], cfg: sampler.ValidationConfig, coarse_proposal_scale: float) -> tuple[dict[str, Any], dict[str, Any]]:
    sites = patch_sites(state["u"].shape[1], x0, y0, cfg.patch_size)
    u_new = state["u"][0].copy()
    u_new, inner_acc = inner_patch_metropolis(u_new, sites, rng, sigma=0.1 * float(coarse_proposal_scale))
    proposal = sampler.compute_state(u_new[None], state["z_edge"], state["z_pair"], state["z_corner"], ctx)
    delta_phi = proposal["phi"][0] - state["phi"][0]
    return proposal, {
        "patch_x": x0,
        "patch_y": y0,
        "tile": tile,
        "inner_acceptance": float(inner_acc),
        "inner_sigma": 0.1 * float(coarse_proposal_scale),
        "delta_logw": float(proposal["logw"][0] - state["logw"][0]),
        "delta_Sf": float(proposal["sf"][0] - state["sf"][0]),
        "delta_Sc": float(proposal["sc"][0] - state["sc"][0]),
        "delta_logdet_refine": float(proposal["logdet"][0] - state["logdet"][0]),
        "delta_logq_missing": float(proposal["logq"][0] - state["logq"][0]),
        "changed_fine_sites_gt_1e-3": int(np.sum(np.abs(delta_phi) > 1.0e-3)),
    }


def batched_propose_patch(states: list[dict[str, Any]], schedules_at_attempt: list[tuple[int, int, str]], rngs: list[np.random.Generator], ctx: dict[str, Any], cfg: sampler.ValidationConfig, coarse_proposal_scale: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    u_new_rows = []
    inner_accs = []
    for state, (x0, y0, _), rng in zip(states, schedules_at_attempt, rngs):
        sites = patch_sites(state["u"].shape[1], x0, y0, cfg.patch_size)
        u_new = state["u"][0].copy()
        u_new, inner_acc = inner_patch_metropolis(u_new, sites, rng, sigma=0.1 * float(coarse_proposal_scale))
        u_new_rows.append(u_new[None])
        inner_accs.append(float(inner_acc))
    batch = sampler.compute_state(
        np.concatenate(u_new_rows, axis=0),
        stack_state_field(states, "z_edge"),
        stack_state_field(states, "z_pair"),
        stack_state_field(states, "z_corner"),
        ctx,
    )
    proposals = [split_batch_state(batch, i) for i in range(len(states))]
    deltas = []
    for i, (state, proposal, (x0, y0, tile)) in enumerate(zip(states, proposals, schedules_at_attempt)):
        delta_phi = proposal["phi"][0] - state["phi"][0]
        deltas.append(
            {
                "patch_x": x0,
                "patch_y": y0,
                "tile": tile,
                "inner_acceptance": inner_accs[i],
                "inner_sigma": 0.1 * float(coarse_proposal_scale),
                "delta_logw": float(proposal["logw"][0] - state["logw"][0]),
                "delta_Sf": float(proposal["sf"][0] - state["sf"][0]),
                "delta_Sc": float(proposal["sc"][0] - state["sc"][0]),
                "delta_logdet_refine": float(proposal["logdet"][0] - state["logdet"][0]),
                "delta_logq_missing": float(proposal["logq"][0] - state["logq"][0]),
                "changed_fine_sites_gt_1e-3": int(np.sum(np.abs(delta_phi) > 1.0e-3)),
            }
        )
    return proposals, deltas


def batched_propose_latent(states: list[dict[str, Any]], schedules_at_attempt: list[tuple[int, int, str]], rngs: list[np.random.Generator], ctx: dict[str, Any], cfg: sampler.ValidationConfig) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rho = cfg.pcn_rho
    noise = math.sqrt(max(0.0, 1.0 - rho * rho))
    z_edges = []
    z_pairs = []
    z_corners = []
    for state, (x0, y0, _), rng in zip(states, schedules_at_attempt, rngs):
        sites = patch_sites(state["u"].shape[1], x0, y0, cfg.patch_size)
        z_edge = state["z_edge"].copy()
        z_pair = state["z_pair"].copy()
        z_corner = state["z_corner"].copy()
        for i, j in sites:
            z_edge[0, 0, i, j] = rho * z_edge[0, 0, i, j] + noise * float(rng.standard_normal())
            z_pair[0, 0, i, j] = rho * z_pair[0, 0, i, j] + noise * float(rng.standard_normal())
            z_corner[0, 0, i, j] = rho * z_corner[0, 0, i, j] + noise * float(rng.standard_normal())
        z_edges.append(z_edge)
        z_pairs.append(z_pair)
        z_corners.append(z_corner)
    batch = sampler.compute_state(
        stack_state_field(states, "u"),
        np.concatenate(z_edges, axis=0),
        np.concatenate(z_pairs, axis=0),
        np.concatenate(z_corners, axis=0),
        ctx,
    )
    proposals = [split_batch_state(batch, i) for i in range(len(states))]
    deltas = []
    for state, proposal, (x0, y0, tile) in zip(states, proposals, schedules_at_attempt):
        delta_phi = proposal["phi"][0] - state["phi"][0]
        deltas.append(
            {
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
        )
    return proposals, deltas


def initial_diagnostics(states_by_kappa: dict[float, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows = []
    base_k = min(states_by_kappa)
    base = states_by_kappa[base_k]
    for k, states in sorted(states_by_kappa.items()):
        logw = np.asarray([float(s["logw"][0]) for s in states], dtype=np.float64)
        shifted = logw - np.max(logw)
        w = np.exp(shifted)
        ess = float(np.sum(w) ** 2 / np.sum(w * w) / len(w))
        acc = []
        for i in range(1, len(logw)):
            acc.append(min(1.0, math.exp(min(0.0, logw[i] - logw[i - 1]))))
        row = {"kappa_f": k, "diagnostic": "initial_logw", "ESS_over_N": ess, "adjacent_order_predicted_acceptance": float(np.mean(acc)) if acc else float("nan"), **qstats(logw)}
        if k != base_k:
            dlogw = np.asarray([float(s["logw"][0] - b["logw"][0]) for s, b in zip(states, base)], dtype=np.float64)
            row.update({f"delta_logw_vs_kappa_{base_k:g}_{kk}": vv for kk, vv in qstats(dlogw).items()})
        rows.append(row)
    return rows


def native_reference_rows(kappas: list[float], lambda_: float, fine_L: int) -> list[dict[str, Any]]:
    candidates = {
        0.27100: PROJECT_ROOT / f"phi4_phase-diagram/ensembles/lam{format_float_tag(lambda_)}_kappa{format_float_tag(0.27100)}_L{fine_L}_embedded_wolff_sign_cluster_plus_radial_heatbath_N500/configs.npz",
        0.27050: PROJECT_ROOT / f"phi4_phase-diagram/ensembles/lam{format_float_tag(lambda_)}_kappa{format_float_tag(0.27050)}_L{fine_L}_embedded_wolff_sign_cluster_plus_radial_heatbath_N500/configs.npz",
    }
    rows = []
    for k in kappas:
        path = candidates.get(round(float(k), 5))
        if path is None or not path.exists():
            continue
        action = ActionSpec("phi4_nn", float(lambda_), float(k), 0.0)
        phi = np.load(path)["phi"].astype(np.float32)
        rows.append({"kappa_f": k, "reference": "native_exact", "path": str(path), "n": int(phi.shape[0]), **local_observable_row(phi, action)})
    # Always include kappa=0.271 as primary anchor if available.
    if 0.271 not in kappas and candidates[0.27100].exists():
        action = ActionSpec("phi4_nn", float(lambda_), 0.271, 0.0)
        phi = np.load(candidates[0.27100])["phi"].astype(np.float32)
        rows.append({"kappa_f": 0.271, "reference": "native_primary_anchor", "path": str(candidates[0.27100]), "n": int(phi.shape[0]), **local_observable_row(phi, action)})
    return rows


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    cfg = build_run_config(load_config(args.config), args)
    coarse, coarse_path, coarse_manifest = load_coarse(cfg)
    ctx_base, coarse_action, fine_action = load_ctx(cfg)
    kappas = [float(x) for x in args.kappa_f]
    coarse_patch_size = int(args.coarse_patch_size or args.patch_size)
    latent_patch_size = int(args.latent_patch_size or coarse_patch_size)
    pcn_rho = rho_from_latent_beta_scale(args.latent_beta_scale) if args.latent_beta_scale is not None else float(args.pcn_rho)
    coarse_proposal_scale = float(args.coarse_proposal_scale if args.coarse_proposal_scale is not None else 1.0)
    cfg_run = sampler.ValidationConfig(
        patch_size=coarse_patch_size,
        origin_mode="random",
        smoke_sweeps=args.sweeps,
        validation_chains=args.chains,
        pcn_rho=pcn_rho,
        pcn_interval_sweeps=1,
        seed=args.seed,
        sector_balanced_init=False,
        measurement_mode="selected_sweeps",
        coarse_start_mode="thermalized_coarse",
        detail_warmup_sweeps=0,
    )
    cfg_latent = replace(cfg_run, patch_size=latent_patch_size)
    n_patch = patches_per_sweep(args.Lc, coarse_patch_size)
    save_sweeps = sorted(set(int(x) for x in args.save_sweeps))
    raw_rows: list[dict[str, Any]] = []
    obs_rows: list[dict[str, Any]] = []
    ar_rows: list[dict[str, Any]] = []
    decomp_rows: list[dict[str, Any]] = []
    initial_states_by_kappa: dict[float, list[dict[str, Any]]] = {k: [] for k in kappas}
    t0 = time.perf_counter()
    for kappa_f in kappas:
        ctx = ctx_for_kappaf(ctx_base, kappa_f)
        for chain in range(args.chains):
            rng = np.random.default_rng(args.seed + 100000 * int(round(kappa_f * 100000)) + 10000 * chain + 777)
            init_idx = int(rng.integers(0, len(coarse)))
            u = coarse[init_idx][None]
            z_edge, z_pair, z_corner = sampler.sample_z(rng, 1, args.Lc)
            state = sampler.compute_state(u, z_edge, z_pair, z_corner, ctx)
            initial_states_by_kappa[kappa_f].append(state)
            local0 = local_observable_row(state["phi"], ctx["fine_action"])
            row0 = {"kappa_f": kappa_f, "chain_id": chain, "sweep": 0, "state_type": "initial_pre_ar", "coarse_index": init_idx, **local0}
            raw_rows.append(row0)
            obs_rows.append(row0)
            decomp_rows.append(state_decomposition_row(state, kappa_f, chain, 0, "initial_pre_ar", args.Lc, args.Lf))
            for sweep in range(1, args.sweeps + 1):
                schedule = random_origin_patch_schedule(args.Lc, coarse_patch_size, rng, "random")
                coarse_accepts = []
                coarse_dlogw = []
                for attempt, (x0, y0, tile) in enumerate(schedule):
                    proposal, delta = propose_patch_scaled(state, x0, y0, tile, rng, ctx, cfg_run, coarse_proposal_scale)
                    state, accept = sampler.apply_ar_update(state, proposal, delta["delta_logw"], math.log(max(rng.random(), 1e-300)))
                    coarse_accepts.append(int(accept))
                    coarse_dlogw.append(float(delta["delta_logw"]))
                    if sweep in save_sweeps:
                        ar_rows.append({"kappa_f": kappa_f, "chain_id": chain, "sweep": sweep, "move_type": "coarse_patch", "attempt_in_sweep": attempt, "accepted": int(accept), **delta})
                for latent_attempt in range(args.latent_updates_per_coarse):
                    latent_schedule = random_origin_patch_schedule(args.Lc, latent_patch_size, rng, "random")
                    proposal_l, delta_l = sampler.propose_latent(state, *latent_schedule[-1], rng, ctx, cfg_latent)
                    state, accept_l = sampler.apply_ar_update(state, proposal_l, delta_l["delta_logw"], math.log(max(rng.random(), 1e-300)))
                    if sweep in save_sweeps:
                        ar_rows.append({"kappa_f": kappa_f, "chain_id": chain, "sweep": sweep, "move_type": "latent_pcn", "attempt_in_sweep": n_patch + latent_attempt, "latent_attempt_in_sweep": latent_attempt, "accepted": int(accept_l), **delta_l})
                if sweep in save_sweeps:
                    local = local_observable_row(state["phi"], ctx["fine_action"])
                    obs_rows.append({"kappa_f": kappa_f, "chain_id": chain, "sweep": sweep, "state_type": "end_of_sweep", **local})
                    decomp_rows.append(state_decomposition_row(state, kappa_f, chain, sweep, "end_of_sweep", args.Lc, args.Lf))
                if args.progress_every_sweeps and sweep % args.progress_every_sweeps == 0 and chain == 0:
                    elapsed = time.perf_counter() - t0
                    print(f"[progress] kappa_f={kappa_f:.5f} chain={chain + 1}/{args.chains} sweep={sweep}/{args.sweeps} elapsed_sec={elapsed:.1f}", flush=True)
    init_diag = initial_diagnostics(initial_states_by_kappa)
    ref_rows = native_reference_rows(kappas, args.lambda_, args.Lf)
    write_csv(out / "raw_upscaled_observables_by_kappaf.csv", raw_rows)
    write_csv(out / "patch_chain_observables_by_kappaf_chain_sweep.csv", obs_rows)
    write_csv(out / "deltaS_AR_diagnostics_by_kappaf.csv", ar_rows)
    write_csv(out / "state_logweight_decomposition_by_kappaf.csv", decomp_rows)
    write_csv(out / "initial_logweight_summary_by_kappaf.csv", init_diag)
    write_csv(out / "native_L64_reference_observables_available.csv", ref_rows)
    summary = {
        "status": "completed",
        "elapsed_sec": time.perf_counter() - t0,
        "coarse_path": str(coarse_path),
        "coarse_manifest": coarse_manifest,
        "kappa_f": kappas,
        "chains": args.chains,
        "sweeps": args.sweeps,
        "save_sweeps": save_sweeps,
        "patch_size": coarse_patch_size,
        "coarse_patch_size": coarse_patch_size,
        "latent_patch_size": latent_patch_size,
        "coarse_proposal_scale": coarse_proposal_scale,
        "pcn_rho": pcn_rho,
        "latent_beta_scale": args.latent_beta_scale,
        "latent_updates_per_coarse": args.latent_updates_per_coarse,
        "latent_attempts_per_sweep": args.latent_updates_per_coarse,
        "N_patch_per_sweep": n_patch,
        "rows": {"raw": len(raw_rows), "obs": len(obs_rows), "ar": len(ar_rows), "decomposition": len(decomp_rows)},
    }
    write_json(out / "summary.json", summary)
    write_report(out, kappas, obs_rows, ar_rows, init_diag, ref_rows, summary)
    make_plots(out, obs_rows, ref_rows)
    return summary


def run_experiment_batched_across_chains(args: argparse.Namespace) -> dict[str, Any]:
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    cfg = build_run_config(load_config(args.config), args)
    coarse, coarse_path, coarse_manifest = load_coarse(cfg)
    ctx_base, coarse_action, fine_action = load_ctx(cfg)
    kappas = [float(x) for x in args.kappa_f]
    coarse_patch_size = int(args.coarse_patch_size or args.patch_size)
    latent_patch_size = int(args.latent_patch_size or coarse_patch_size)
    pcn_rho = rho_from_latent_beta_scale(args.latent_beta_scale) if args.latent_beta_scale is not None else float(args.pcn_rho)
    coarse_proposal_scale = float(args.coarse_proposal_scale if args.coarse_proposal_scale is not None else 1.0)
    cfg_run = sampler.ValidationConfig(
        patch_size=coarse_patch_size,
        origin_mode="random",
        smoke_sweeps=args.sweeps,
        validation_chains=args.chains,
        pcn_rho=pcn_rho,
        pcn_interval_sweeps=1,
        seed=args.seed,
        sector_balanced_init=False,
        measurement_mode="selected_sweeps",
        coarse_start_mode="thermalized_coarse",
        detail_warmup_sweeps=0,
    )
    cfg_latent = replace(cfg_run, patch_size=latent_patch_size)
    n_patch = patches_per_sweep(args.Lc, coarse_patch_size)
    save_sweeps = sorted(set(int(x) for x in args.save_sweeps))
    raw_rows: list[dict[str, Any]] = []
    obs_rows: list[dict[str, Any]] = []
    ar_rows: list[dict[str, Any]] = []
    decomp_rows: list[dict[str, Any]] = []
    initial_states_by_kappa: dict[float, list[dict[str, Any]]] = {k: [] for k in kappas}
    t0 = time.perf_counter()
    for kappa_f in kappas:
        ctx = ctx_for_kappaf(ctx_base, kappa_f)
        rngs = [np.random.default_rng(args.seed + 100000 * int(round(kappa_f * 100000)) + 10000 * chain + 777) for chain in range(args.chains)]
        init_indices = [int(rng.integers(0, len(coarse))) for rng in rngs]
        u0 = np.stack([coarse[idx] for idx in init_indices], axis=0).astype(np.float32)
        z_edges, z_pairs, z_corners = [], [], []
        for rng in rngs:
            ze, zp, zc = sampler.sample_z(rng, 1, args.Lc)
            z_edges.append(ze)
            z_pairs.append(zp)
            z_corners.append(zc)
        batch0 = sampler.compute_state(u0, np.concatenate(z_edges, axis=0), np.concatenate(z_pairs, axis=0), np.concatenate(z_corners, axis=0), ctx)
        states = [split_batch_state(batch0, i) for i in range(args.chains)]
        for chain, state in enumerate(states):
            initial_states_by_kappa[kappa_f].append(state)
            local0 = local_observable_row(state["phi"], ctx["fine_action"])
            row0 = {"kappa_f": kappa_f, "chain_id": chain, "sweep": 0, "state_type": "initial_pre_ar", "coarse_index": init_indices[chain], **local0}
            raw_rows.append(row0)
            obs_rows.append(row0)
            decomp_rows.append(state_decomposition_row(state, kappa_f, chain, 0, "initial_pre_ar", args.Lc, args.Lf))
        for sweep in range(1, args.sweeps + 1):
            schedules = [random_origin_patch_schedule(args.Lc, coarse_patch_size, rng, "random") for rng in rngs]
            for attempt in range(n_patch):
                proposals, deltas = batched_propose_patch(states, [schedule[attempt] for schedule in schedules], rngs, ctx, cfg_run, coarse_proposal_scale)
                for chain in range(args.chains):
                    states[chain], accept = sampler.apply_ar_update(states[chain], proposals[chain], deltas[chain]["delta_logw"], math.log(max(rngs[chain].random(), 1e-300)))
                    if sweep in save_sweeps:
                        ar_rows.append({"kappa_f": kappa_f, "chain_id": chain, "sweep": sweep, "move_type": "coarse_patch", "attempt_in_sweep": attempt, "accepted": int(accept), **deltas[chain]})
            for latent_attempt in range(args.latent_updates_per_coarse):
                latent_schedules = [random_origin_patch_schedule(args.Lc, latent_patch_size, rng, "random") for rng in rngs]
                proposals_l, deltas_l = batched_propose_latent(states, [schedule[-1] for schedule in latent_schedules], rngs, ctx, cfg_latent)
                for chain in range(args.chains):
                    states[chain], accept_l = sampler.apply_ar_update(states[chain], proposals_l[chain], deltas_l[chain]["delta_logw"], math.log(max(rngs[chain].random(), 1e-300)))
                    if sweep in save_sweeps:
                        ar_rows.append({"kappa_f": kappa_f, "chain_id": chain, "sweep": sweep, "move_type": "latent_pcn", "attempt_in_sweep": n_patch + latent_attempt, "latent_attempt_in_sweep": latent_attempt, "accepted": int(accept_l), **deltas_l[chain]})
            if sweep in save_sweeps:
                for chain in range(args.chains):
                    local = local_observable_row(states[chain]["phi"], ctx["fine_action"])
                    obs_rows.append({"kappa_f": kappa_f, "chain_id": chain, "sweep": sweep, "state_type": "end_of_sweep", **local})
                    decomp_rows.append(state_decomposition_row(states[chain], kappa_f, chain, sweep, "end_of_sweep", args.Lc, args.Lf))
            if args.progress_every_sweeps and sweep % args.progress_every_sweeps == 0:
                elapsed = time.perf_counter() - t0
                print(f"[progress] kappa_f={kappa_f:.5f} sweep={sweep}/{args.sweeps} elapsed_sec={elapsed:.1f}", flush=True)
    init_diag = initial_diagnostics(initial_states_by_kappa)
    ref_rows = native_reference_rows(kappas, args.lambda_, args.Lf)
    write_csv(out / "raw_upscaled_observables_by_kappaf.csv", raw_rows)
    write_csv(out / "patch_chain_observables_by_kappaf_chain_sweep.csv", obs_rows)
    write_csv(out / "deltaS_AR_diagnostics_by_kappaf.csv", ar_rows)
    write_csv(out / "state_logweight_decomposition_by_kappaf.csv", decomp_rows)
    write_csv(out / "initial_logweight_summary_by_kappaf.csv", init_diag)
    write_csv(out / "native_L64_reference_observables_available.csv", ref_rows)
    summary = {
        "status": "completed",
        "mode": "batch_logq_across_chains",
        "elapsed_sec": time.perf_counter() - t0,
        "coarse_path": str(coarse_path),
        "coarse_manifest": coarse_manifest,
        "kappa_f": kappas,
        "chains": args.chains,
        "sweeps": args.sweeps,
        "save_sweeps": save_sweeps,
        "patch_size": coarse_patch_size,
        "coarse_patch_size": coarse_patch_size,
        "latent_patch_size": latent_patch_size,
        "coarse_proposal_scale": coarse_proposal_scale,
        "pcn_rho": pcn_rho,
        "latent_beta_scale": args.latent_beta_scale,
        "latent_updates_per_coarse": args.latent_updates_per_coarse,
        "latent_attempts_per_sweep": args.latent_updates_per_coarse,
        "N_patch_per_sweep": n_patch,
        "rows": {"raw": len(raw_rows), "obs": len(obs_rows), "ar": len(ar_rows), "decomposition": len(decomp_rows)},
    }
    write_json(out / "summary.json", summary)
    write_report(out, kappas, obs_rows, ar_rows, init_diag, ref_rows, summary)
    make_plots(out, obs_rows, ref_rows)
    return summary


def aggregate_obs(obs_rows: list[dict[str, Any]]) -> dict[tuple[float, int], dict[str, float]]:
    out: dict[tuple[float, int], dict[str, float]] = {}
    for k in sorted({float(r["kappa_f"]) for r in obs_rows}):
        for s in sorted({int(r["sweep"]) for r in obs_rows}):
            sub = [r for r in obs_rows if float(r["kappa_f"]) == k and int(r["sweep"]) == s]
            if not sub:
                continue
            row: dict[str, float] = {"n": float(len(sub))}
            for key in LOCAL_KEYS:
                vals = [float(r[key]) for r in sub]
                row[key] = float(np.mean(vals))
                row[key + "_se"] = float(np.std(vals, ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else 0.0
            out[(k, s)] = row
    return out


def ar_summary(ar_rows: list[dict[str, Any]]) -> dict[float, dict[str, float]]:
    out = {}
    for k in sorted({float(r["kappa_f"]) for r in ar_rows}):
        sub = [r for r in ar_rows if float(r["kappa_f"]) == k and r["move_type"] == "coarse_patch"]
        lat = [r for r in ar_rows if float(r["kappa_f"]) == k and r["move_type"] == "latent_pcn"]
        out[k] = {
            "coarse_acceptance_saved_sweeps": float(np.mean([int(r["accepted"]) for r in sub])) if sub else float("nan"),
            "coarse_delta_logw_std_saved_sweeps": float(np.std([float(r["delta_logw"]) for r in sub], ddof=1)) if len(sub) > 1 else float("nan"),
            "latent_acceptance_saved_sweeps": float(np.mean([int(r["accepted"]) for r in lat])) if lat else float("nan"),
            "latent_delta_logw_std_saved_sweeps": float(np.std([float(r["delta_logw"]) for r in lat], ddof=1)) if len(lat) > 1 else float("nan"),
        }
    return out


def write_report(out: Path, kappas: list[float], obs_rows: list[dict[str, Any]], ar_rows: list[dict[str, Any]], init_diag: list[dict[str, Any]], refs: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    agg = aggregate_obs(obs_rows)
    ar = ar_summary(ar_rows)
    ref271 = next((r for r in refs if abs(float(r["kappa_f"]) - 0.271) < 1e-12), None)
    lines = [
        "# L32->L64 kappaf matching diagnostic",
        "",
        "This is a transported-coordinate diagnostic. No broad native L64 kappa scan is launched by this script.",
        "",
        f"- chains x sweeps: `{summary['chains']} x {summary['sweeps']}`",
        f"- saved sweeps: `{summary['save_sweeps']}`",
        f"- N_patch/sweep: `{summary['N_patch_per_sweep']}`",
        "",
        "## Initial logweight diagnostics",
        "",
        "| kappaf | logw std | ESS/N | predicted adjacent acceptance |",
        "|---:|---:|---:|---:|",
    ]
    for r in init_diag:
        lines.append(f"| {float(r['kappa_f']):.5f} | {float(r['std']):.6g} | {float(r['ESS_over_N']):.6g} | {float(r['adjacent_order_predicted_acceptance']):.6g} |")
    lines += ["", "## Saved-sweep A/R diagnostics", "", "| kappaf | coarse acc | coarse std dlogw | latent acc | latent std dlogw |", "|---:|---:|---:|---:|---:|"]
    for k in kappas:
        a = ar.get(k, {})
        lines.append(f"| {k:.5f} | {a.get('coarse_acceptance_saved_sweeps', float('nan')):.6g} | {a.get('coarse_delta_logw_std_saved_sweeps', float('nan')):.6g} | {a.get('latent_acceptance_saved_sweeps', float('nan')):.6g} | {a.get('latent_delta_logw_std_saved_sweeps', float('nan')):.6g} |")
    lines += ["", "## Late observables", "", "| kappaf | sweep | Binder | xi/L | chi | NN | action_density |"]
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for k in kappas:
        sweeps = [s for kk, s in agg if kk == k]
        if not sweeps:
            continue
        s = max(sweeps)
        row = agg[(k, s)]
        lines.append(f"| {k:.5f} | {s} | {row['Binder_U4']:.6g} | {row['xi_over_L']:.6g} | {row['chi']:.6g} | {row['NN']:.6g} | {row['action_density']:.6g} |")
    if ref271:
        lines += [
            "",
            "## Native reference availability",
            "",
            f"Primary native L64 reference found for kappa=0.271: `{ref271['path']}`.",
            f"Native kappa=0.271 Binder `{float(ref271['Binder_U4']):.6g}`, xi/L `{float(ref271['xi_over_L']):.6g}`, chi `{float(ref271['chi']):.6g}`.",
        ]
    missing = [k for k in kappas if not any(abs(float(r["kappa_f"]) - k) < 1e-12 for r in refs)]
    if missing:
        lines.append(f"Missing exact native L64 references for: `{missing}`. Comparisons for those kappas should use kappa=0.271 only as a qualitative anchor.")
    lines += [
        "",
        "## Ranking guidance",
        "",
        "Rank candidates by the generated CSVs rather than only this summary: smallest initial logw std, largest ESS/N and A/R, smallest movement from sweep 0 to late saved sweeps, and closeness to available native L64 anchors.",
    ]
    (out / "KAPPAF_MATCHING_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_plots(out: Path, obs_rows: list[dict[str, Any]], refs: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    agg = aggregate_obs(obs_rows)
    keys = ["Binder_U4", "xi_over_L", "chi", "NN", "diag", "2nn", "action_density"]
    colors = {0.27050: "tab:purple", 0.27075: "tab:blue", 0.27100: "tab:orange", 0.27125: "tab:green"}
    with PdfPages(out / "observable_flow_vs_patch_sweep_by_kappaf.pdf") as pdf:
        for key in keys:
            fig, ax = plt.subplots(figsize=(7.2, 4.6))
            for k in sorted({kk for kk, _ in agg}):
                sweeps = sorted(s for kk, s in agg if kk == k)
                y = [agg[(k, s)][key] for s in sweeps]
                e = [agg[(k, s)].get(key + "_se", 0.0) for s in sweeps]
                ax.errorbar(sweeps, y, yerr=e, marker="o", lw=1.2, capsize=2, color=colors.get(round(k, 5)), label=f"kf={k:.5f}")
            for r in refs:
                if key in r:
                    ax.axhline(float(r[key]), color="black", alpha=0.18, lw=0.9)
            ax.set_xlabel("patch sweep")
            ax.set_ylabel(key)
            ax.legend(fontsize=8)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--run-name", type=str, default=None)
    ap.add_argument("--lambda", dest="lambda_", type=float, default=0.022)
    ap.add_argument("--kappa-c", type=float, default=0.2705)
    ap.add_argument("--coarse-ensemble", type=Path, default=None)
    ap.add_argument("--fine-reference", type=Path, default=None)
    ap.add_argument("--kappa-f", type=float, nargs="+", default=KAPPA_FS)
    ap.add_argument("--Lc", type=int, default=32)
    ap.add_argument("--Lf", type=int, default=64)
    ap.add_argument("--chains", type=int, default=8)
    ap.add_argument("--sweeps", type=int, default=300)
    ap.add_argument("--save-sweeps", type=int, nargs="+", default=SAVE_SWEEPS)
    ap.add_argument("--patch-size", type=int, default=4)
    ap.add_argument("--coarse-patch-size", type=int, default=None)
    ap.add_argument("--latent-patch-size", type=int, default=None)
    ap.add_argument("--coarse-proposal-scale", type=float, default=None)
    ap.add_argument("--pcn-rho", type=float, default=0.5)
    ap.add_argument("--latent-beta-scale", type=float, default=None)
    ap.add_argument("--latent-updates-per-coarse", type=int, default=1, help="Number of latent/detail pCN attempts per coarse sweep cycle. Default 1 preserves the historical one-latent-update-per-sweep behavior.")
    ap.add_argument("--progress-every-sweeps", type=int, default=0, help="Print flushed progress every N sweeps. Default 0 disables progress logging.")
    ap.add_argument("--seed", type=int, default=20260721)
    ap.add_argument("--batch-logq-across-chains", action="store_true", help="Batch full-volume proposal/logq evaluations across independent chains at the same sweep and patch index.")
    args = ap.parse_args()
    if args.out_dir is None:
        args.out_dir = PKG / "outputs" / "shape_parametric_sampler_validation" / (
            f"L{args.Lc}_to_L{args.Lf}_kappaf_matching_"
            f"lam{format_float_tag(args.lambda_)}_kc{format_float_tag(args.kappa_c)}"
        )
    summary = run_experiment_batched_across_chains(args) if args.batch_logq_across_chains else run_experiment(args)
    print(json.dumps({"out": str(args.out_dir), **summary}, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
