#!/usr/bin/env python3
"""Four-substep global MH diagnostic for wrapped factor-two inverse blocking.

This test intentionally recomputes full states after every substep.  The
acceptance path always uses the complete trusted density expression; latent
factorisations are checked only as diagnostics.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PKG = PROJECT_ROOT / "perfect_blocking_upsampling"
sys.path[:0] = [str(PKG / "src"), str(PKG / "scripts")]

from perfect_blocking_upsampling.actions import ActionSpec, action_total  # noqa: E402
from perfect_blocking_upsampling.blocking import apply_kernel, load_kernel_matrix, load_phi  # noqa: E402
from perfect_blocking_upsampling.wrapped_flow_coordinates import infer_rqspline_latents_and_logj, reconstruct_rqspline_from_latents  # noqa: E402
from run_lam1p0_l16to32_rqspline_zeroshot import build_model_from_checkpoint, stationary_stats  # noqa: E402
from run_lam1p0_mit_style_inverse_blocking_L8to16 import (  # noqa: E402
    ACCEPT_COLUMNS, DEFAULT_CHECKPOINT, DEFAULT_KERNEL, G_COLUMNS, MAIN_COLUMNS,
    State, aggregate, make_state_from_detail, measurement_rows, reblock_error,
    write_csv, write_json,
)

DEFAULT_COARSE = PROJECT_ROOT / "data/configs_phi4_2d/lam1p0_kappac0p340301_L8/configs.npz"
DEFAULT_NATIVE = PROJECT_ROOT / "data/configs_phi4_2d/lam1p0_kappac0p340301_L16/configs.npz"
DEFAULT_OUT = PKG / "outputs/controlled_patch_lam1p0/mit_style_four_substep_L8to16"
# At L32 coarse / L64 fine, independently accumulated three-sector float32
# log-J terms can differ by about 1.1e-3 solely from summation order.  This is
# a diagnostic check; the MH decision always uses the full-density expression.
DECOMP_TOL = 2.0e-3
REB_TOL = 5.0e-6

SUBSTEP_COLUMNS = [
    "sweep", "substep", "subset_y", "subset_x", "attempts", "accepts", "acceptance",
    "attempts_cumulative", "accepts_cumulative", "acceptance_cumulative",
    "coarse_kernel_proposals", "coarse_kernel_accepts", "coarse_kernel_acceptance",
]


def save_checkpoint(path: Path, current: "LatentState") -> None:
    """Atomically persist the complete latent-coordinate chain state."""
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp,
        phi=current.state.phi,
        psi=current.state.psi,
        coarse=current.state.coarse,
        detail=current.state.detail,
        z=current.z,
        action=current.state.action,
        logq=current.state.logq,
        source_config_index=current.state.source_index,
        source_native_index=current.state.source_native_index,
    )
    os.replace(tmp, path)


@dataclass
class LatentState:
    state: State
    z: np.ndarray
    prior_by_stage: np.ndarray
    flow_logj_by_stage: np.ndarray


def prior_by_stage(z: np.ndarray) -> np.ndarray:
    return -0.5 * np.sum(z.astype(np.float64) ** 2 + math.log(2.0 * math.pi), axis=(2, 3))


def reconstruct_stage_logj(model: Any, coarse: np.ndarray, z: np.ndarray, stats: dict[str, Any], *, batch_size: int, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Trusted forward flow reconstruction, with per-stage physical log-J."""
    detail, total_std = reconstruct_rqspline_from_latents(model, coarse, z, stats, batch_size=batch_size, device=device)
    # Infer each stage's forward log-J by changing only that latent sector.  The
    # total forward determinant is additive in this autoregressive factorisation.
    n, lc, _ = coarse.shape
    stage = np.zeros((n, 3), dtype=np.float64)
    with torch.no_grad():
        cstd = torch.from_numpy(((coarse - stats["coarse_mean"]) / stats["coarse_std"]).astype(np.float32)).to(device)
        zb = torch.from_numpy(z.astype(np.float32)).to(device)
        dstd = torch.zeros_like(zb)
        for s in range(3):
            cond_affine = model.affine_base.cond(cstd, dstd, s)
            x_affine, affine_ld = model.affine_base.flows[s].forward(zb[:, s].flatten(1), cond_affine)
            cond_spline = model.cond(cstd, dstd, s)
            x, spline_ld = model.spline.flows[s].forward(x_affine, cond_spline)
            dstd[:, s] = x.reshape(n, lc, lc)
            stage[:, s] = (affine_ld + spline_ld).detach().cpu().numpy().astype(np.float64)
    physical_scale = lc * lc * np.log(np.asarray(stats["detail_std"], dtype=np.float64))
    stage += physical_scale[None, :]
    # This detects any accidental autoregressive reconstruction inconsistency.
    discrepancy = np.abs(stage.sum(axis=1) - physical_scale.sum() - total_std)
    if not np.all(discrepancy <= DECOMP_TOL):
        raise RuntimeError(
            "per-stage forward log-J does not reproduce total log-J: "
            f"max discrepancy {float(np.max(discrepancy)):.6g} exceeds tolerance {DECOMP_TOL:.6g}"
        )
    return detail, stage


def build_latent_state(coarse: np.ndarray, z: np.ndarray, source: np.ndarray, source_native: np.ndarray, *, model: Any, stats: dict[str, Any], kernel: np.ndarray, action_c: ActionSpec, action_f: ActionSpec, batch_size: int, device: torch.device) -> LatentState:
    detail, flow_stage = reconstruct_stage_logj(model, coarse, z, stats, batch_size=batch_size, device=device)
    state = make_state_from_detail(coarse, detail, source, source_native, model=model, stats=stats, kernel=kernel, action_c=action_c, action_f=action_f, batch_size=batch_size, device=device)
    return LatentState(state=state, z=z.astype(np.float32), prior_by_stage=prior_by_stage(z), flow_logj_by_stage=flow_stage)


def select_state(old: LatentState, new: LatentState, accept: np.ndarray) -> LatentState:
    def choose(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.where(accept.reshape((-1,) + (1,) * (a.ndim - 1)), b, a)
    st = old.state
    ns = new.state
    chosen = State(
        phi=choose(st.phi, ns.phi), psi=choose(st.psi, ns.psi), coarse=choose(st.coarse, ns.coarse), detail=choose(st.detail, ns.detail),
        action=choose(st.action, ns.action), logq=choose(st.logq, ns.logq), logq_coarse=choose(st.logq_coarse, ns.logq_coarse),
        logq_detail=choose(st.logq_detail, ns.logq_detail), logprior_latent=choose(st.logprior_latent, ns.logprior_latent),
        flow_logabsdet=choose(st.flow_logabsdet, ns.flow_logabsdet), inverse_kernel_logabsdet=st.inverse_kernel_logabsdet,
        source_index=choose(st.source_index, ns.source_index), source_native_index=choose(st.source_native_index, ns.source_native_index),
    )
    return LatentState(chosen, choose(old.z, new.z), choose(old.prior_by_stage, new.prior_by_stage), choose(old.flow_logj_by_stage, new.flow_logj_by_stage))


def decomp_logA(old: LatentState, new: LatentState) -> np.ndarray:
    # -Delta S_f + Delta S_c + log r(old)-log r(new)+Delta log|J_flow|.
    delta_sc = -new.state.logq_coarse + old.state.logq_coarse
    return -new.state.action + old.state.action + delta_sc + old.prior_by_stage.sum(axis=1) - new.prior_by_stage.sum(axis=1) + new.flow_logj_by_stage.sum(axis=1) - old.flow_logj_by_stage.sum(axis=1)


def coarse_kernel_transition(coarse: np.ndarray, active_sites: np.ndarray, *, action: ActionSpec, sigma: float, n_steps: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Apply random-permutation, per-site Metropolis sweeps for q_c ∝ exp(-S_c).

    Every touched site has its own Gaussian proposal and A/R decision. A
    uniformly sampled permutation is paired with its reversed permutation in
    the proposal law, so one full local sweep is reversible with respect to
    q_c. Repeating independently sampled sweeps preserves that reversibility.
    """
    current = np.asarray(coarse, dtype=np.float32).copy()
    accepted = np.zeros(len(current), dtype=np.int64)
    lam = float(action.lambda_)
    kappa = float(action.kappa)
    quadratic = 1.0 - 2.0 * lam
    lc = current.shape[1]
    active_flat = np.flatnonzero(active_sites.reshape(-1))
    if len(active_flat) == 0:
        raise RuntimeError("coarse subset contains no active sites")
    for _ in range(n_steps):
        for flat in rng.permutation(active_flat):
            y, x = divmod(int(flat), lc)
            old = current[:, y, x].astype(np.float64)
            proposal = old + sigma * rng.standard_normal(len(current))
            neighbors = (
                current[:, (y - 1) % lc, x] + current[:, (y + 1) % lc, x] +
                current[:, y, (x - 1) % lc] + current[:, y, (x + 1) % lc]
            ).astype(np.float64)
            delta = quadratic * (proposal * proposal - old * old)
            delta += lam * (proposal**4 - old**4)
            delta -= 2.0 * kappa * (proposal - old) * neighbors
            take = np.log(rng.random(len(current))) < np.minimum(0.0, -delta)
            current[take, y, x] = proposal[take].astype(np.float32)
            accepted += take.astype(np.int64)
    return current, accepted


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--coarse-proposal-source", type=Path, default=DEFAULT_COARSE)
    ap.add_argument("--native-reference-source", type=Path, default=DEFAULT_NATIVE)
    ap.add_argument("--flow-checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--kernel-path", type=Path, default=DEFAULT_KERNEL)
    ap.add_argument("--from-L", type=int, default=8, help="Coarse lattice size; supported: 8, 16, or 32.")
    ap.add_argument("--to-L", type=int, default=16, help="Fine lattice size; must equal 2*--from-L.")
    ap.add_argument("--n-chains", type=int, default=16)
    ap.add_argument("--n-sweeps", type=int, default=20)
    ap.add_argument("--save-sweeps", default="0,1,2,5,10,20")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=2026072301)
    ap.add_argument("--initial-start-index", type=int, default=0)
    ap.add_argument(
        "--initialization",
        choices=("blocked_native", "direct_coarse_flow"),
        default="blocked_native",
        help="blocked_native is a stationarity control; direct_coarse_flow stores an uncorrected direct-native coarse plus flow sample at sweep 0.",
    )
    ap.add_argument(
        "--coarse-update-mode",
        choices=("coarse_mh_kernel", "direct_independence"),
        default="coarse_mh_kernel",
        help="coarse_mh_kernel applies a reversible S_c Metropolis kernel before the full-density outer MH step; direct_independence preserves iid native-coarse proposals.",
    )
    ap.add_argument("--coarse-sigma", type=float, default=0.04, help="Per-site Gaussian scale for the reversible S_c coarse kernel.")
    ap.add_argument("--coarse-updates-per-sweep", type=int, default=1, help="Number of random-permutation, per-site S_c Metropolis sweeps used to form one correlated coarse proposal before z01/z10/z11 refreshes.")
    ap.add_argument("--divide", type=int, choices=(1, 2), default=1, help="Partition each coarse and latent field into divide^2 residue classes; all subset cycles are recorded under one sweep.")
    ap.add_argument("--coarse-source-start-index", type=int, default=0)
    ap.add_argument("--coarse-source-count", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.to_L != 2 * args.from_L or args.from_L not in {8, 16, 32}:
        raise RuntimeError("four-substep diagnostic supports factor-two L8->L16, L16->L32, or L32->L64 only")
    if args.coarse_sigma <= 0.0 or args.coarse_updates_per_sweep <= 0:
        raise RuntimeError("coarse sigma and coarse updates per sweep must be positive")
    if args.smoke:
        args.n_chains, args.n_sweeps, args.save_sweeps = min(args.n_chains, 4), min(args.n_sweeps, 3), "0,1,2,3"
    saves = {int(x) for x in args.save_sweeps.split(",") if x}; saves.add(0)
    run = args.run_dir.resolve()
    for name in ["logs", "observables", "plots", "summaries", "debug", "checkpoints"]: (run / name).mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    kernel, meta = load_kernel_matrix(args.kernel_path)
    if not bool(meta.get("kernel_coefficients_include_eta_scale", False)): raise RuntimeError("kernel must include eta scale")
    coarse_all, native_all = load_phi(args.coarse_proposal_source), load_phi(args.native_reference_source)
    lc, lf = args.from_L, args.to_L
    if lc % args.divide:
        raise RuntimeError("coarse lattice size must be divisible by --divide")
    if coarse_all.shape[1:] != (lc, lc) or native_all.shape[1:] != (lf, lf):
        raise RuntimeError(f"source lattice shapes must be L{lc} and L{lf}")
    if args.initial_start_index + args.n_chains > len(native_all): raise RuntimeError("insufficient native reference states")
    if args.initialization == "direct_coarse_flow" and args.initial_start_index + args.n_chains > len(coarse_all):
        raise RuntimeError("insufficient direct coarse initial states")
    n_pool = args.coarse_source_count or (len(coarse_all) - args.coarse_source_start_index)
    pool = np.arange(args.coarse_source_start_index, args.coarse_source_start_index + n_pool, dtype=np.int64)
    if len(pool) == 0 or pool[-1] >= len(coarse_all): raise RuntimeError("invalid coarse source pool")
    ckpt = torch.load(args.flow_checkpoint, map_location=device, weights_only=False)
    model, load_report = build_model_from_checkpoint(ckpt, lc, device)
    stats = stationary_stats(ckpt["state"]["stats"], lc)
    action_c = action_f = ActionSpec("phi4_nn", 1.0, 0.340301)
    rng = np.random.default_rng(args.seed)
    native_idx = np.arange(args.initial_start_index, args.initial_start_index + args.n_chains, dtype=np.int64)
    # Keep a native reference subset available in either initialization mode.
    native_psi = apply_kernel(native_all[native_idx], kernel)
    native_c = native_psi[:, 0::2, 0::2].astype(np.float32)
    native_d = np.stack([native_psi[:, 0::2, 1::2], native_psi[:, 1::2, 0::2], native_psi[:, 1::2, 1::2]], axis=1).astype(np.float32)
    if args.initialization == "blocked_native":
        c0, d0 = native_c, native_d
        z0, _ = infer_rqspline_latents_and_logj(model, c0, d0, stats, batch_size=args.batch_size, device=device)
        initial_source = np.full(args.n_chains, -1, dtype=np.int64)
        initial_native = native_idx
        initial_role = "blocked_native_initial"
    else:
        initial_source = np.arange(args.initial_start_index, args.initial_start_index + args.n_chains, dtype=np.int64)
        c0 = coarse_all[initial_source].astype(np.float32)
        z0 = rng.standard_normal((args.n_chains, 3, lc, lc)).astype(np.float32)
        initial_native = np.full(args.n_chains, -1, dtype=np.int64)
        initial_role = "direct_native_coarse_flow_initial"
    current = build_latent_state(c0, z0, initial_source, initial_native, model=model, stats=stats, kernel=kernel, action_c=action_c, action_f=action_f, batch_size=args.batch_size, device=device)
    reb, _ = reblock_error(current.state.phi, current.state.coarse, kernel)
    if float(np.max(reb)) > REB_TOL: raise RuntimeError("initial reblocking failure")

    main_rows, g_rows = measurement_rows(current.state, 0)
    sub_rows: list[dict[str, Any]] = []
    acc_rows: list[dict[str, Any]] = []
    sub_acc_rows: list[dict[str, Any]] = []
    coarse_kernel_rows: list[dict[str, Any]] = []
    source_draw_rows: list[dict[str, Any]] = []
    all_sub_accepts = {"coarse": [], "z01": [], "z10": [], "z11": []}
    all_sub_counts = {"coarse": 0, "z01": 0, "z10": 0, "z11": 0}
    all_sub_attempts = {"coarse": 0, "z01": 0, "z10": 0, "z11": 0}
    coarse_kernel_accepts_total = 0
    coarse_kernel_attempts_total = 0
    max_decomp_error = 0.0
    inventory = ([{"role": initial_role, "chain_id": i, "source_index": int(v)} for i, v in enumerate(native_idx if args.initialization == "blocked_native" else initial_source)] + [{"role": "native_reference", "chain_id": i, "source_index": int(v)} for i, v in enumerate(native_idx)] + [{"role": "coarse_proposal_pool", "chain_id": -1, "source_index": int(v)} for v in pool])
    write_csv(run / "source_index_inventory.csv", inventory, ["role", "chain_id", "source_index"])
    coarse_acceptance = "-S_f(new)-logq_full(new)+S_f(old)+logq_full(old)"
    config = vars(args) | {"lambda": 1.0, "kappa_c": 0.340301, "kappa_f": 0.340301, "L_c": lc, "L_f": lf, "algorithm": "four_substep_global_mh", "initialization_semantics": "uncorrected direct-native coarse plus wrapped-flow latent sample at sweep 0" if args.initialization == "direct_coarse_flow" else "blocked native fine state at sweep 0 for stationarity", "coarse_acceptance": coarse_acceptance, "latent_acceptance": "-S_f(new)-logq_full(new)+S_f(old)+logq_full(old)", "kernel_metadata": meta, "flow_load_report": load_report}
    write_json(run / "run_config.json", config); (run / "run_config.yaml").write_text(json.dumps(config, indent=2, default=str) + "\n")
    (run / "submit_manifest.txt").write_text(
        "driver_command: " + " ".join(sys.argv) + "\n" + json.dumps(config, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    started = time.perf_counter()
    write_json(run / "status.json", {"status": "running", "current_sweep": 0, "run_dir": str(run)})
    save_checkpoint(run / "checkpoints" / "checkpoint_sweep_0000.npz", current)
    save_checkpoint(run / "checkpoints" / "checkpoint_latest.npz", current)
    # Sweep zero must be visible before any proposal, just like direct MIT.
    write_csv(run / "observables" / "main_per_sweep_measurements.csv", main_rows, MAIN_COLUMNS)
    write_csv(run / "observables" / "per_sweep_observables.csv", main_rows, MAIN_COLUMNS)
    write_csv(run / "observables" / "Gk_per_sweep_measurements.csv", g_rows, G_COLUMNS)
    write_csv(run / "observables" / "acceptance_history.csv", acc_rows, ACCEPT_COLUMNS)
    write_csv(run / "debug" / "substep_acceptance_history.csv", sub_acc_rows, SUBSTEP_COLUMNS)
    write_csv(run / "debug" / "coarse_kernel_acceptance_history.csv", coarse_kernel_rows)
    write_csv(run / "observables" / "ensemble_average_history.csv", aggregate(main_rows))

    yy, xx = np.indices((lc, lc))
    subsets = [(oy, ox, (yy % args.divide == oy) & (xx % args.divide == ox)) for oy in range(args.divide) for ox in range(args.divide)]
    for sweep in range(1, args.n_sweeps + 1):
        sweep_counts = {"coarse": 0, "z01": 0, "z10": 0, "z11": 0}
        sweep_attempts = {"coarse": 0, "z01": 0, "z10": 0, "z11": 0}
        for subset_y, subset_x, active_sites in subsets:
          substeps = [("coarse", None, 1), ("z01", 0, 1), ("z10", 1, 1), ("z11", 2, 1)]
          for name, sector, substep_attempt in substeps:
            old = current
            if sector is None and args.coarse_update_mode == "direct_independence":
                source = pool[rng.integers(0, len(pool), size=args.n_chains)]
                coarse_kernel_accepts = None
                candidate = build_latent_state(coarse_all[source], old.z.copy(), source, np.full(args.n_chains, -1, dtype=np.int64), model=model, stats=stats, kernel=kernel, action_c=action_c, action_f=action_f, batch_size=args.batch_size, device=device)
                full = -candidate.state.action - candidate.state.logq + old.state.action + old.state.logq
                decomposed = decomp_logA(old, candidate)
                proposal_mode = "direct_independence"
            elif sector is None:
                source = np.full(args.n_chains, -1, dtype=np.int64)
                coarse_new, coarse_kernel_accepts = coarse_kernel_transition(old.state.coarse, active_sites, action=action_c, sigma=args.coarse_sigma, n_steps=args.coarse_updates_per_sweep, rng=rng)
                candidate = build_latent_state(coarse_new, old.z.copy(), old.state.source_index, old.state.source_native_index, model=model, stats=stats, kernel=kernel, action_c=action_c, action_f=action_f, batch_size=args.batch_size, device=device)
                full = -candidate.state.action - candidate.state.logq + old.state.action + old.state.logq
                decomposed = decomp_logA(old, candidate)
                proposal_mode = "coarse_mh_kernel"
            else:
                source = old.state.source_index.copy()
                coarse_kernel_accepts = None
                znew = old.z.copy()
                proposed_subset = rng.standard_normal(znew[:, sector].shape).astype(np.float32)
                znew[:, sector] = np.where(active_sites[None, :, :], proposed_subset, znew[:, sector])
                candidate = build_latent_state(old.state.coarse, znew, old.state.source_index, old.state.source_native_index, model=model, stats=stats, kernel=kernel, action_c=action_c, action_f=action_f, batch_size=args.batch_size, device=device)
                full = -candidate.state.action - candidate.state.logq + old.state.action + old.state.logq
                decomposed = decomp_logA(old, candidate)
                proposal_mode = "latent_independence"
            err = float(np.max(np.abs(full - decomposed))); max_decomp_error = max(max_decomp_error, err)
            if err > DECOMP_TOL: raise RuntimeError(f"{name} logA decomposition mismatch {err:.6g}")
            reverse = -old.state.action - old.state.logq + candidate.state.action + candidate.state.logq
            if float(np.max(np.abs(full + reverse))) > 1.0e-10:
                raise RuntimeError(f"{name} logA swap antisymmetry failure")
            logu = np.log(rng.random(args.n_chains)); accepted = logu < np.minimum(0.0, full)
            for i in range(args.n_chains):
                row = {"sweep": sweep, "chain_id": i, "substep": name, "substep_attempt": substep_attempt, "subset_y": subset_y, "subset_x": subset_x, "active_sites": int(active_sites.sum()), "coarse_proposal_mode": proposal_mode, "coarse_kernel_steps": args.coarse_updates_per_sweep if coarse_kernel_accepts is not None else 0, "coarse_kernel_accepts": int(coarse_kernel_accepts[i]) if coarse_kernel_accepts is not None else 0, "accepted": int(accepted[i]), "old_S_f": float(old.state.action[i]), "new_S_f": float(candidate.state.action[i]), "old_S_c": float(-old.state.logq_coarse[i]), "new_S_c": float(-candidate.state.logq_coarse[i]), "old_logq_flow": float(old.state.logq_detail[i]), "new_logq_flow": float(candidate.state.logq_detail[i]), "old_logq_full": float(old.state.logq[i]), "new_logq_full": float(candidate.state.logq[i]), "old_logJ_flow": float(old.flow_logj_by_stage[i].sum()), "new_logJ_flow": float(candidate.flow_logj_by_stage[i].sum()), "inverse_kernel_logabsdet": float(old.state.inverse_kernel_logabsdet), "coordinate_permutation_logabsdet": 0.0, "logA_full": float(full[i]), "logA_decomposed": float(decomposed[i]), "logA_difference": float(full[i]-decomposed[i]), "log_uniform": float(logu[i]), "proposal_coarse_source_index": int(source[i])}
                for s, label in enumerate(["01", "10", "11"]): row[f"old_logprior_z{label}"], row[f"new_logprior_z{label}"], row[f"old_logJ_z{label}"], row[f"new_logJ_z{label}"] = float(old.prior_by_stage[i,s]), float(candidate.prior_by_stage[i,s]), float(old.flow_logj_by_stage[i,s]), float(candidate.flow_logj_by_stage[i,s])
                sub_rows.append(row)
            accepts = int(np.sum(accepted))
            all_sub_accepts[name].append(float(np.mean(accepted)))
            all_sub_counts[name] += accepts
            all_sub_attempts[name] += args.n_chains
            sweep_counts[name] += accepts
            sweep_attempts[name] += args.n_chains
            kernel_attempts = 0
            kernel_accepts = 0
            if name == "coarse" and proposal_mode == "coarse_mh_kernel":
                kernel_attempts = args.n_chains * args.coarse_updates_per_sweep * int(active_sites.sum())
                kernel_accepts = int(coarse_kernel_accepts.sum())
                coarse_kernel_attempts_total += kernel_attempts
                coarse_kernel_accepts_total += kernel_accepts
                coarse_kernel_rows.append({"sweep": sweep, "subset_y": subset_y, "subset_x": subset_x, "active_sites_per_chain": int(active_sites.sum()), "site_sweeps_per_refresh": args.coarse_updates_per_sweep, "site_attempts_per_chain": args.coarse_updates_per_sweep * int(active_sites.sum()), "attempts": kernel_attempts, "accepts": kernel_accepts, "acceptance": kernel_accepts / kernel_attempts, "attempts_cumulative": coarse_kernel_attempts_total, "accepts_cumulative": coarse_kernel_accepts_total, "acceptance_cumulative": coarse_kernel_accepts_total / coarse_kernel_attempts_total})
            sub_acc_rows.append({"sweep": sweep, "substep": name, "subset_y": subset_y, "subset_x": subset_x, "attempts": args.n_chains, "accepts": accepts, "acceptance": accepts / args.n_chains, "attempts_cumulative": all_sub_attempts[name], "accepts_cumulative": all_sub_counts[name], "acceptance_cumulative": all_sub_counts[name] / all_sub_attempts[name], "coarse_kernel_proposals": kernel_attempts, "coarse_kernel_accepts": kernel_accepts, "coarse_kernel_acceptance": kernel_accepts / kernel_attempts if kernel_attempts else float("nan")})
            if name == "coarse" and proposal_mode == "direct_independence":
                source_draw_rows.extend({"sweep": sweep, "chain_id": int(i), "coarse_source_index": int(source[i]), "accepted": int(accepted[i])} for i in range(args.n_chains))
            current = select_state(old, candidate, accepted)
        if sweep in saves:
            r, gr = measurement_rows(current.state, sweep); main_rows.extend(r); g_rows.extend(gr)
        acc_rows.append({"sweep": sweep, "update_mode": f"four_substep_{args.coarse_update_mode}_divide{args.divide}", "coarse_acceptance": sweep_counts["coarse"] / sweep_attempts["coarse"], "coarse_proposals": sweep_attempts["coarse"], "coarse_accepts": sweep_counts["coarse"], "coarse_acceptance_cumulative": all_sub_counts["coarse"] / all_sub_attempts["coarse"], "coarse_proposals_cumulative": all_sub_attempts["coarse"], "coarse_accepts_cumulative": all_sub_counts["coarse"], "detail_acceptance": sum(sweep_counts[x] for x in ["z01","z10","z11"]) / sum(sweep_attempts[x] for x in ["z01","z10","z11"]), "detail_proposals": sum(sweep_attempts[x] for x in ["z01","z10","z11"]), "detail_accepts": sum(sweep_counts[x] for x in ["z01","z10","z11"]), "detail_acceptance_cumulative": sum(all_sub_counts[x] for x in ["z01","z10","z11"]) / sum(all_sub_attempts[x] for x in ["z01","z10","z11"]), "detail_proposals_cumulative": sum(all_sub_attempts[x] for x in ["z01","z10","z11"]), "detail_accepts_cumulative": sum(all_sub_counts[x] for x in ["z01","z10","z11"]), "conditional_flow_refreshes": 4 * args.divide * args.divide * args.n_chains})
        write_csv(run / "debug" / "four_substep_diagnostics.csv", sub_rows)
        write_csv(run / "debug" / "substep_acceptance_history.csv", sub_acc_rows, SUBSTEP_COLUMNS)
        write_csv(run / "debug" / "coarse_kernel_acceptance_history.csv", coarse_kernel_rows)
        write_csv(run / "observables" / "acceptance_history.csv", acc_rows, ACCEPT_COLUMNS)
        write_csv(run / "coarse_proposal_source_draws.csv", source_draw_rows, ["sweep", "chain_id", "coarse_source_index", "accepted"])
        if sweep in saves:
            write_csv(run / "observables" / "main_per_sweep_measurements.csv", main_rows, MAIN_COLUMNS); write_csv(run / "observables" / "per_sweep_observables.csv", main_rows, MAIN_COLUMNS); write_csv(run / "observables" / "Gk_per_sweep_measurements.csv", g_rows, G_COLUMNS); write_csv(run / "observables" / "ensemble_average_history.csv", aggregate(main_rows))
            save_checkpoint(run / "checkpoints" / f"checkpoint_sweep_{sweep:04d}.npz", current)
        save_checkpoint(run / "checkpoints" / "checkpoint_latest.npz", current)
        write_json(run / "status.json", {"status": "running", "current_sweep": sweep, "substep_acceptance": {key: all_sub_counts[key] / all_sub_attempts[key] for key in all_sub_attempts}, "coarse_kernel_acceptance": coarse_kernel_accepts_total / coarse_kernel_attempts_total if coarse_kernel_attempts_total else float("nan"), "run_dir": str(run)})
        print(json.dumps({"sweep": sweep, **{f"accept_{k}": sweep_counts[k] / sweep_attempts[k] for k in sweep_counts}, "divide": args.divide}), flush=True)
    native_reference = make_state_from_detail(native_c, native_d, np.full(args.n_chains, -1, dtype=np.int64), native_idx, model=model, stats=stats, kernel=kernel, action_c=action_c, action_f=action_f, batch_size=args.batch_size, device=device)
    native_rows, _ = measurement_rows(native_reference, 0)
    write_csv(run / "observables" / f"native_L{lf}_reference_summary.csv", aggregate(native_rows))
    comparison_rows: list[dict[str, Any]] = []
    native_values = {key: np.asarray([float(row[key]) for row in native_rows]) for key in ["action_density", "phi2", "phi4", "NN", "diag", "2nn", "m2", "m4"]}
    for sweep in sorted({int(row["sweep"]) for row in main_rows}):
        observed = [row for row in main_rows if int(row["sweep"]) == sweep]
        for key, baseline in native_values.items():
            value = np.asarray([float(row[key]) for row in observed])
            comparison_rows.append({"sweep": sweep, "observable": key, "mean": float(value.mean()), "native_mean": float(baseline.mean()), "mean_shift_native_sigma": float((value.mean() - baseline.mean()) / max(baseline.std(ddof=1), 1e-300)), "std_ratio": float(value.std(ddof=1) / max(baseline.std(ddof=1), 1e-300)) if len(value) > 1 else float("nan")})
    write_csv(run / "observables" / "sweep_native_comparison.csv", comparison_rows)
    final_reb = float(np.max(reblock_error(current.state.phi, current.state.coarse, kernel)[0]))
    summary = {"status": "completed", "substep_acceptance": {k: (all_sub_counts[k] / all_sub_attempts[k]) if all_sub_attempts[k] else float("nan") for k in all_sub_attempts}, "coarse_kernel_acceptance": coarse_kernel_accepts_total / coarse_kernel_attempts_total if coarse_kernel_attempts_total else float("nan"), "coarse_kernel_steps_per_refresh": args.coarse_updates_per_sweep, "max_logA_decomposition_error": max_decomp_error, "decomposition_tolerance": DECOMP_TOL, "final_reblocking_max": final_reb, "runtime_sec": time.perf_counter() - started, "coarse_acceptance_ratio": coarse_acceptance, "latent_acceptance_ratio": "-S_f(new)-logq_full(new)+S_f(old)+logq_full(old)", "validation": {"identity_logA": 0.0, "swap_antisymmetry": "passed per substep", "full_vs_decomposed_logA": "passed per substep"}}
    write_json(run / "summaries" / "run_summary.json", summary)
    (run / "summaries" / "acceptance_ratio_derivation.md").write_text("# Four-substep MH ratios\n\nThe `z01`, `z10`, and `z11` independent standard-normal refreshes use `logA = -S_f(new) - logq_full(new) + S_f(old) + logq_full(old)`, with `logq_full = -S_c(c) + sum_s log N(z_s) - log|J_flow| - log|J_inverse_kernel|`.\n\nFor `coarse_mh_kernel`, the proposal is produced by the requested number of random-permutation per-site Metropolis sweeps targeting `q_c(c) proportional to exp[-S_c(c)]`. Each site gets its own Gaussian proposal and A/R decision. Uniform permutation averaging includes each reversed visit order, making the coarse sweep reversible with respect to exactly the same coarse proposal distribution as an iid direct-native draw. Therefore the outer coarse transition uses the same full-density ratio `logA = -S_f(new)-logq_full(new)+S_f(old)+logq_full(old)`. The inner S_c acceptance is recorded separately in `debug/coarse_kernel_acceptance_history.csv`.\n", encoding="utf-8")
    (run / "summaries" / "run_summary.md").write_text("# Four-substep MIT diagnostic\n\n- L%d to L%d wrapped final-5x5 flow\n- four sequential full-density MH substeps per sweep: coarse, z01, z10, z11\n- sweeps: `%d`\n- chains: `%d`\n- maximum full/decomposed logA discrepancy: `%.6g`\n" % (lc, lf, args.n_sweeps, args.n_chains, max_decomp_error), encoding="utf-8")
    write_json(run / "status.json", {"status": "completed", "current_sweep": args.n_sweeps, "latest_checkpoint": str(run / "checkpoints" / "checkpoint_latest.npz"), **summary})
    print(json.dumps({"run_dir": str(run), "substep_acceptance": summary["substep_acceptance"], "sweeps": args.n_sweeps}, indent=2), flush=True)
    return 0

def write_failure_status(exc: Exception) -> None:
    """Prevent a failed asynchronous run from retaining a stale status."""
    try:
        i = sys.argv.index("--run-dir")
        run = Path(sys.argv[i + 1]).resolve()
        if not run.exists():
            return
        payload = {
            "status": "failed",
            "run_dir": str(run),
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback": traceback.format_exc(),
        }
        tmp = run / "status.failed.tmp.json"
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, run / "status.json")
    except Exception:
        # Do not mask the original exception if the status write itself fails.
        pass


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        write_failure_status(exc)
        raise
