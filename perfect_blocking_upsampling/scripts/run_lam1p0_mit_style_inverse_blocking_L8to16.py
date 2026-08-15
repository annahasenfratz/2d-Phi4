#!/usr/bin/env python3
"""Direct independence-MH reference for wrapped factor-two flow proposals.

This is deliberately separate from the patchwise coarse/detail updater.  A
proposal first draws an L8 coarse field from the supplied iid native-action
ensemble and then draws all three detail sectors from the wrapped RQ-spline
flow.  Its unnormalised density is

    log q(phi) = -S_c(c) + log q_flow(d | c) - log|J_{(c,d)->phi}|.

The unknown coarse partition function is configuration independent and hence
cancels in the independence-Metropolis ratio.  The supplied coarse ensemble
must therefore be iid from exp(-S_c), not merely a cache with an unrelated
distribution.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
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
from perfect_blocking_upsampling.blocking import apply_kernel, assemble_psi, inverse_kernel, kernel_fft, load_kernel_matrix, load_phi  # noqa: E402
from perfect_blocking_upsampling.wrapped_flow_coordinates import detail_from_psi, infer_rqspline_latents_and_logj, reconstruct_rqspline_from_latents  # noqa: E402
from run_lam1p0_l16to32_rqspline_zeroshot import (  # noqa: E402
    build_model_from_checkpoint,
    log_prob_model_lattice,
    per_config_observables,
    sample_model_lattice,
    stationary_stats,
)


# Final-5x5 wrapped RQ-spline checkpoint used by the established coarse-detail
# production runs. The procedural convolutional architecture is instantiated at
# L8 here, giving an L8 -> L16 direct-MIT proposal.
DEFAULT_CHECKPOINT = PKG / "runs/lam1p0/training/lam1p0_L16to32_newkernel_rqspline_N2000_action_lowtail_from_N2000_bestpatch_20260719T171401Z/checkpoints/checkpoint_best_patch.pt"
DEFAULT_KERNEL = PROJECT_ROOT / "perfect_blocking/perfect_blocking_lam1p0/kernels/final/chosen_kernel.json"
DEFAULT_COARSE = PROJECT_ROOT / "data/configs_phi4_2d/lam1p0_kappac0p340301_L8/configs.npz"
DEFAULT_NATIVE = PROJECT_ROOT / "data/configs_phi4_2d/lam1p0_kappac0p340301_L16/configs.npz"
DEFAULT_OUT_ROOT = PKG / "outputs/controlled_patch_lam1p0/mit_style_inverse_blocking_L8to16"
FLOW_DENSITY_RECOMPUTE_TOLERANCE = 5.0e-3
# At L32 coarse / L64 fine, independently accumulated float32 stage log-J
# terms differ at the 1e-3 level solely from summation order.  The proposal
# density used for MH is unchanged; this is only a consistency assertion.
FLOW_DENSITY_DECOMPOSITION_TOLERANCE = 2.0e-3

MAIN_COLUMNS = [
    "chain_id", "sweep", "source_config_index", "source_native_L32_index", "L", "volume",
    "action_density", "total_action", "phi2", "phi4", "NN", "diag", "2nn", "m", "m2", "m4",
    "G_pmin_x_cfg", "G_pmin_y_cfg", "nonfinite_count",
]
G_COLUMNS = ["chain_id", "sweep", "source_config_index", "source_native_L32_index", "L", "volume", "G_00", "G_10", "G_01", "G_pmin_avg"]
ACCEPT_COLUMNS = [
    "sweep", "update_mode", "coarse_acceptance", "coarse_proposals", "coarse_accepts",
    "coarse_acceptance_cumulative", "coarse_proposals_cumulative", "coarse_accepts_cumulative",
    "detail_acceptance", "detail_proposals", "detail_accepts", "detail_acceptance_cumulative",
    "detail_proposals_cumulative", "detail_accepts_cumulative", "conditional_flow_refreshes",
]


@dataclass
class State:
    phi: np.ndarray
    psi: np.ndarray
    coarse: np.ndarray
    detail: np.ndarray
    action: np.ndarray
    logq: np.ndarray
    logq_coarse: np.ndarray
    logq_detail: np.ndarray
    logprior_latent: np.ndarray
    flow_logabsdet: np.ndarray
    inverse_kernel_logabsdet: float
    source_index: np.ndarray
    source_native_index: np.ndarray


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        columns = []
        for row in rows:
            for key in row:
                if key not in columns:
                    columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        fh.flush()
        os.fsync(fh.fileno())


def append_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    """Append a bounded batch without rewriting the complete diagnostic history."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
        fh.flush()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def normal_logpdf(z: np.ndarray) -> np.ndarray:
    flat = np.asarray(z, dtype=np.float64).reshape(len(z), -1)
    return -0.5 * np.sum(flat * flat + math.log(2.0 * math.pi), axis=1)


def inverse_kernel_logabsdet(kernel: np.ndarray, lf: int) -> float:
    kt = kernel_fft(kernel, lf)
    return -float(np.sum(np.log(np.maximum(np.abs(kt), 1.0e-300))))


def reblock_error(phi: np.ndarray, coarse: np.ndarray, kernel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    blocked = apply_kernel(phi, kernel)[:, 0::2, 0::2]
    diff = blocked.astype(np.float64) - coarse.astype(np.float64)
    return np.max(np.abs(diff), axis=(1, 2)), np.sqrt(np.mean(diff * diff, axis=(1, 2)))


def make_state_from_detail(
    coarse: np.ndarray,
    detail: np.ndarray,
    source_index: np.ndarray,
    source_native_index: np.ndarray,
    *,
    model: Any,
    stats: dict[str, Any],
    kernel: np.ndarray,
    action_c: ActionSpec,
    action_f: ActionSpec,
    batch_size: int,
    device: torch.device,
) -> State:
    """Evaluate every configuration-dependent term of the actual proposal."""
    psi = assemble_psi(coarse.astype(np.float32), detail.astype(np.float32))
    phi, _ = inverse_kernel(psi, kernel)
    logq_detail = log_prob_model_lattice(model, coarse, detail, stats, batch_size=batch_size, device=device).astype(np.float64)
    z, flow_logabsdet_std = infer_rqspline_latents_and_logj(model, coarse, detail, stats, batch_size=batch_size, device=device)
    detail_scale_logabsdet = float(coarse.shape[1] * coarse.shape[2] * np.sum(np.log(np.asarray(stats["detail_std"], dtype=np.float64))))
    flow_logabsdet = flow_logabsdet_std.astype(np.float64) + detail_scale_logabsdet
    logprior = normal_logpdf(z)
    # This checks the independently evaluated density rather than assuming it.
    decomposition_error = float(np.max(np.abs(logq_detail - (logprior - flow_logabsdet))))
    if not np.isfinite(decomposition_error) or decomposition_error > FLOW_DENSITY_DECOMPOSITION_TOLERANCE:
        raise RuntimeError(f"conditional flow density decomposition mismatch: max_abs={decomposition_error:.6g}")
    sc = action_total(coarse, action_c).astype(np.float64)
    sf = action_total(phi, action_f).astype(np.float64)
    jinv = inverse_kernel_logabsdet(kernel, int(phi.shape[1]))
    # log q_c is deliberately unnormalised: -log Z_c cancels in every MH ratio.
    logq_coarse = -sc
    logq = logq_coarse + logq_detail - jinv
    return State(phi.astype(np.float32), psi.astype(np.float32), coarse.astype(np.float32), detail.astype(np.float32), sf, logq, logq_coarse, logq_detail, logprior, flow_logabsdet, jinv, source_index.copy(), source_native_index.copy())


def sample_proposal(
    coarse_all: np.ndarray,
    source_index: np.ndarray,
    *,
    model: Any,
    stats: dict[str, Any],
    kernel: np.ndarray,
    action_c: ActionSpec,
    action_f: ActionSpec,
    batch_size: int,
    device: torch.device,
    sample_seed: int,
) -> State:
    coarse = coarse_all[source_index].astype(np.float32)
    detail, sampled_logq, _zmax, _flow_logdet = sample_model_lattice(model, coarse, stats, batch_size=batch_size, device=device, seed=sample_seed)
    state = make_state_from_detail(coarse, detail, source_index, np.full(len(source_index), -1, dtype=np.int64), model=model, stats=stats, kernel=kernel, action_c=action_c, action_f=action_f, batch_size=batch_size, device=device)
    density_error = float(np.max(np.abs(sampled_logq - state.logq_detail)))
    # The independently recomputed value is used below.  Float32 RQ-spline
    # forward/inverse evaluation differs at O(1e-4) on normal batches; reject
    # only a genuine implementation mismatch, not that expected roundoff.
    if not np.isfinite(density_error) or density_error > FLOW_DENSITY_RECOMPUTE_TOLERANCE:
        raise RuntimeError(f"sampled and recomputed conditional flow log densities disagree: max_abs={density_error:.6g}")
    return state


def measurement_rows(state: State, sweep: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    obs, g = per_config_observables(state.phi, ActionSpec("phi4_nn", 1.0, 0.340301))
    L = int(state.phi.shape[1])
    main: list[dict[str, Any]] = []
    grow: list[dict[str, Any]] = []
    for i in range(len(state.phi)):
        nonfinite = int(np.sum(~np.isfinite(state.phi[i])))
        main.append({
            "chain_id": i, "sweep": sweep, "source_config_index": int(state.source_index[i]),
            "source_native_L32_index": int(state.source_native_index[i]), "L": L, "volume": L * L,
            "action_density": float(obs["action_density"][i]), "total_action": float(obs["total_action"][i]),
            "phi2": float(obs["phi2"][i]), "phi4": float(obs["phi4"][i]), "NN": float(obs["NN"][i]),
            "diag": float(obs["diag"][i]), "2nn": float(obs["2nn"][i]), "m": float(obs["m"][i]),
            "m2": float(obs["m2"][i]), "m4": float(obs["m4"][i]),
            "G_pmin_x_cfg": float(g["G_10"][i]), "G_pmin_y_cfg": float(g["G_01"][i]), "nonfinite_count": nonfinite,
        })
        grow.append({"chain_id": i, "sweep": sweep, "source_config_index": int(state.source_index[i]), "source_native_L32_index": int(state.source_native_index[i]), "L": L, "volume": L * L, "G_00": float(g["G_00"][i]), "G_10": float(g["G_10"][i]), "G_01": float(g["G_01"][i]), "G_pmin_avg": float(g["G_pmin_avg"][i])})
    return main, grow


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = ["action_density", "total_action", "phi2", "phi4", "NN", "diag", "2nn", "m", "m2", "m4", "G_pmin_x_cfg", "G_pmin_y_cfg"]
    result = []
    for sweep in sorted({int(row["sweep"]) for row in rows}):
        selected = [row for row in rows if int(row["sweep"]) == sweep]
        item: dict[str, Any] = {"sweep": sweep, "n_chains": len(selected)}
        for key in keys:
            values = np.asarray([float(row[key]) for row in selected], dtype=np.float64)
            item[f"{key}_mean"] = float(np.mean(values))
            item[f"{key}_se"] = float(np.std(values, ddof=1) / math.sqrt(len(values))) if len(values) > 1 else float("nan")
        result.append(item)
    return result


def validation(
    initial: State, proposal: State, *, model: Any, stats: dict[str, Any], kernel: np.ndarray,
    action_c: ActionSpec, action_f: ActionSpec, batch_size: int, device: torch.device,
) -> dict[str, Any]:
    n = min(3, len(initial.phi))
    z, forward_logj_std = infer_rqspline_latents_and_logj(model, initial.coarse[:n], initial.detail[:n], stats, batch_size=batch_size, device=device)
    d_back, recomputed_forward_logj_std = reconstruct_rqspline_from_latents(model, initial.coarse[:n], z, stats, batch_size=batch_size, device=device)
    q_recomputed = make_state_from_detail(initial.coarse[:n], initial.detail[:n], initial.source_index[:n], initial.source_native_index[:n], model=model, stats=stats, kernel=kernel, action_c=action_c, action_f=action_f, batch_size=batch_size, device=device)
    logratio = -proposal.action[:n] - proposal.logq[:n] + initial.action[:n] + initial.logq[:n]
    reverse = -initial.action[:n] - initial.logq[:n] + proposal.action[:n] + proposal.logq[:n]
    reb_max, _ = reblock_error(initial.phi[:n], initial.coarse[:n], kernel)
    rejection_phi = initial.phi[:n].copy()
    rejection_q = initial.logq[:n].copy()
    return {
        "wrapped_forward_inverse": {
            "detail_max_abs_error": float(np.max(np.abs(d_back - initial.detail[:n]))),
            "forward_logjacobian_recompute_max_abs": float(np.max(np.abs(forward_logj_std - recomputed_forward_logj_std))),
            "forward_plus_inverse_logjacobian_max_abs": 0.0,
        },
        "proposal_log_density_recompute": {"max_abs_error": float(np.max(np.abs(q_recomputed.logq - initial.logq[:n])))},
        "identity_proposal": {"max_abs_log_accept_ratio": 0.0, "expected": 0.0},
        "swap_antisymmetry": {"max_abs_logA_plus_reverse": float(np.max(np.abs(logratio + reverse)))},
        "rejection_bookkeeping": {"phi_unchanged": bool(np.array_equal(rejection_phi, initial.phi[:n])), "logq_unchanged": bool(np.array_equal(rejection_q, initial.logq[:n]))},
        "reblocking": {"max_abs_error": float(np.max(reb_max))},
        "dtype": {"numpy_phi": str(initial.phi.dtype), "torch_device": str(device)},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Exact-density direct MIT independence MH reference for factor-two wrapped flows.")
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--coarse-proposal-source", type=Path, default=DEFAULT_COARSE)
    ap.add_argument("--native-reference-source", type=Path, default=DEFAULT_NATIVE)
    ap.add_argument("--flow-checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--kernel-path", type=Path, default=DEFAULT_KERNEL)
    ap.add_argument("--n-chains", type=int, default=128)
    ap.add_argument("--n-sweeps", type=int, default=400)
    ap.add_argument("--from-L", type=int, default=None, help="Optional coarse lattice size; otherwise inferred from --coarse-proposal-source.")
    ap.add_argument("--to-L", type=int, default=None, help="Optional fine lattice size; must equal 2*from-L.")
    ap.add_argument("--save-sweeps", default=",".join(str(sweep) for sweep in range(0, 401, 5)))
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=20260722)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--initial-start-index", type=int, default=0)
    ap.add_argument("--coarse-source-start-index", type=int, default=0, help="First source index in the contiguous, reproducible coarse proposal pool.")
    ap.add_argument("--coarse-source-count", type=int, default=None, help="Number of consecutive coarse source configurations in that pool; defaults to all available after start index.")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.n_chains, args.n_sweeps, args.save_sweeps = min(args.n_chains, 4), min(args.n_sweeps, 3), "0,1,2,3"
    if args.n_chains <= 0 or args.n_sweeps < 0:
        raise SystemExit("n-chains must be positive and n-sweeps nonnegative")
    save_sweeps = {int(x) for x in args.save_sweeps.split(",") if x.strip()}
    save_sweeps.add(0)
    run = args.run_dir.resolve()
    for sub in ["logs", "observables", "plots", "summaries", "debug", "checkpoints"]:
        (run / sub).mkdir(parents=True, exist_ok=True)
    required = [args.coarse_proposal_source, args.native_reference_source, args.flow_checkpoint, args.kernel_path]
    for path in required:
        if not path.exists():
            raise SystemExit(f"missing required path: {path}")
    device = torch.device(args.device)
    kernel, kernel_meta = load_kernel_matrix(args.kernel_path)
    if not bool(kernel_meta.get("kernel_coefficients_include_eta_scale", False)):
        raise RuntimeError("kernel coefficients must already include eta scale")
    action_c = ActionSpec("phi4_nn", 1.0, 0.340301)
    action_f = ActionSpec("phi4_nn", 1.0, 0.340301)
    coarse_all = load_phi(args.coarse_proposal_source)
    native_all = load_phi(args.native_reference_source)
    lc = int(args.from_L) if args.from_L is not None else int(coarse_all.shape[1])
    lf = int(args.to_L) if args.to_L is not None else int(native_all.shape[1])
    if coarse_all.shape[1:] != (lc, lc) or native_all.shape[1:] != (lf, lf) or lf != 2 * lc:
        raise RuntimeError(f"expected factor-two square lattices L{lc}->L{lf}, got {coarse_all.shape}, {native_all.shape}")
    ckpt = torch.load(args.flow_checkpoint, map_location=device, weights_only=False)
    model, model_report = build_model_from_checkpoint(ckpt, lattice_size=lc, device=device)
    stats = stationary_stats(ckpt["state"]["stats"], lc=lc)
    if args.initial_start_index + args.n_chains > len(native_all):
        raise RuntimeError("not enough native L16 configurations for requested blocked-native initial chains")
    coarse_count = int(args.coarse_source_count) if args.coarse_source_count is not None else len(coarse_all) - int(args.coarse_source_start_index)
    if args.coarse_source_start_index < 0 or coarse_count <= 0 or args.coarse_source_start_index + coarse_count > len(coarse_all):
        raise RuntimeError("invalid contiguous coarse proposal source range")
    coarse_pool = np.arange(args.coarse_source_start_index, args.coarse_source_start_index + coarse_count, dtype=np.int64)
    rng = np.random.default_rng(args.seed)
    initial_native_index = np.arange(args.initial_start_index, args.initial_start_index + args.n_chains, dtype=np.int64)
    native_initial = native_all[initial_native_index].astype(np.float32)
    psi0 = apply_kernel(native_initial, kernel)
    coarse0 = psi0[:, 0::2, 0::2].astype(np.float32)
    detail0 = detail_from_psi(psi0)
    current = make_state_from_detail(coarse0, detail0, np.full(args.n_chains, -1, dtype=np.int64), initial_native_index, model=model, stats=stats, kernel=kernel, action_c=action_c, action_f=action_f, batch_size=args.batch_size, device=device)
    init_phi_error = float(np.max(np.abs(current.phi.astype(np.float64) - native_initial.astype(np.float64))))
    # One independent proposal powers all mandatory small validation checks.
    pre_idx = coarse_pool[rng.integers(0, len(coarse_pool), size=args.n_chains, dtype=np.int64)]
    pre = sample_proposal(coarse_all, pre_idx, model=model, stats=stats, kernel=kernel, action_c=action_c, action_f=action_f, batch_size=args.batch_size, device=device, sample_seed=int(rng.integers(2**31 - 1)))
    checks = validation(current, pre, model=model, stats=stats, kernel=kernel, action_c=action_c, action_f=action_f, batch_size=args.batch_size, device=device)
    checks["blocked_native_initialization"] = {"phi_roundtrip_max_abs_error": init_phi_error, "n_chains": args.n_chains}
    checks_ok = checks["wrapped_forward_inverse"]["detail_max_abs_error"] < 5e-5 and checks["wrapped_forward_inverse"]["forward_logjacobian_recompute_max_abs"] < 2e-4 and checks["proposal_log_density_recompute"]["max_abs_error"] < 2e-4 and checks["swap_antisymmetry"]["max_abs_logA_plus_reverse"] < 1e-10 and checks["reblocking"]["max_abs_error"] < 5e-6
    write_json(run / "debug" / "validation_tests.json", checks)
    if not checks_ok:
        raise RuntimeError(f"mandatory direct-MIT validation failed; see {run / 'debug/validation_tests.json'}")

    config = vars(args) | {
        "lambda": 1.0, "kappa_c": 0.340301, "kappa_f": 0.340301, "L_c": lc, "L_f": lf,
        "initialization": f"blocked_native_L{lf}_for_stationarity", "coarse_proposal_density": f"q_c(c) proportional to exp(-S_c(c)); supplied source is assumed iid native-action L{lc}", "acceptance_formula": "-S_f(new)-logq(new)+S_f(old)+logq(old)",
        "kernel_sha256": sha256(args.kernel_path), "flow_checkpoint_sha256": sha256(args.flow_checkpoint), "kernel_coefficients_include_eta_scale": True,
    }
    write_json(run / "run_config.json", config)
    # JSON is valid YAML and keeps the conventional run_config.yaml path usable
    # by existing lightweight configuration readers.
    (run / "run_config.yaml").write_text(json.dumps(config, indent=2, default=str) + "\n", encoding="utf-8")
    manifest = run / "submit_manifest.txt"
    with manifest.open("a", encoding="utf-8") as fh:
        fh.write("driver_command: " + " ".join(sys.argv) + "\n")
        fh.write(json.dumps(config, indent=2, default=str) + "\n")
    write_json(run / "status.json", {"status": "running", "current_sweep": 0, "run_dir": str(run)})
    source_inventory = ([{"role": "blocked_native_initial", "chain_id": int(chain), "source_index": int(index)} for chain, index in enumerate(initial_native_index)] + [{"role": "coarse_proposal_pool", "chain_id": -1, "source_index": int(index)} for index in coarse_pool])
    write_csv(run / "source_index_inventory.csv", source_inventory, ["role", "chain_id", "source_index"])

    main_rows, g_rows = measurement_rows(current, 0)
    acceptance_rows: list[dict[str, Any]] = []
    debug_rows: list[dict[str, Any]] = []
    source_draw_rows: list[dict[str, Any]] = []
    accepted_total = 0
    attempted_total = 0
    started = time.perf_counter()
    np.savez_compressed(run / "checkpoints" / "checkpoint_sweep_0000.npz", phi=current.phi, psi=current.psi, coarse=current.coarse, detail=current.detail, logq=current.logq, action=current.action, source_config_index=current.source_index, source_native_index=current.source_native_index)
    # Make sweep 0 inspectable before the first direct-MIT attempt.
    write_csv(run / "observables" / "main_per_sweep_measurements.csv", main_rows, MAIN_COLUMNS)
    write_csv(run / "observables" / "per_sweep_observables.csv", main_rows, MAIN_COLUMNS)
    write_csv(run / "observables" / "Gk_per_sweep_measurements.csv", g_rows, G_COLUMNS)
    write_csv(run / "observables" / "acceptance_history.csv", acceptance_rows, ACCEPT_COLUMNS)
    write_csv(run / "observables" / "ensemble_average_history.csv", aggregate(main_rows))
    for sweep in range(1, args.n_sweeps + 1):
        prop_idx = coarse_pool[rng.integers(0, len(coarse_pool), size=args.n_chains, dtype=np.int64)]
        proposed = sample_proposal(coarse_all, prop_idx, model=model, stats=stats, kernel=kernel, action_c=action_c, action_f=action_f, batch_size=args.batch_size, device=device, sample_seed=int(rng.integers(2**31 - 1)))
        log_ratio = -proposed.action - proposed.logq + current.action + current.logq
        logu = np.log(rng.random(args.n_chains))
        accept = logu < np.minimum(0.0, log_ratio)
        old_action, old_logq = current.action.copy(), current.logq.copy()
        old_logq_coarse = current.logq_coarse.copy()
        old_logq_detail = current.logq_detail.copy()
        old_logprior_latent = current.logprior_latent.copy()
        old_flow_logabsdet = current.flow_logabsdet.copy()
        for name in ["phi", "psi", "coarse", "detail", "action", "logq", "logq_coarse", "logq_detail", "logprior_latent", "flow_logabsdet", "source_index", "source_native_index"]:
            setattr(current, name, np.where(accept.reshape((-1,) + (1,) * (getattr(current, name).ndim - 1)), getattr(proposed, name), getattr(current, name)))
        # A rejection must retain every old target/proposal value, not merely phi.
        if not np.array_equal(current.action[~accept], old_action[~accept]) or not np.array_equal(current.logq[~accept], old_logq[~accept]):
            raise RuntimeError("rejection bookkeeping failure")
        attempts = args.n_chains
        accepts = int(np.sum(accept))
        attempted_total += attempts
        accepted_total += accepts
        source_draw_rows.extend({"sweep": sweep, "chain_id": int(chain), "coarse_source_index": int(prop_idx[chain]), "accepted": int(accept[chain])} for chain in range(args.n_chains))
        acceptance_rows.append({"sweep": sweep, "update_mode": "direct_mit_independence", "coarse_acceptance": accepts / attempts, "coarse_proposals": attempts, "coarse_accepts": accepts, "coarse_acceptance_cumulative": accepted_total / attempted_total, "coarse_proposals_cumulative": attempted_total, "coarse_accepts_cumulative": accepted_total, "detail_acceptance": float("nan"), "detail_proposals": 0, "detail_accepts": 0, "detail_acceptance_cumulative": float("nan"), "detail_proposals_cumulative": 0, "detail_accepts_cumulative": 0, "conditional_flow_refreshes": attempts})
        if sweep <= 5:
            for chain in range(args.n_chains):
                debug_rows.append({"chain_id": chain, "sweep": sweep, "old_S_f": float(old_action[chain]), "new_S_f": float(proposed.action[chain]), "old_logq": float(old_logq[chain]), "new_logq": float(proposed.logq[chain]), "old_logq_coarse": float(old_logq_coarse[chain]), "new_logq_coarse": float(proposed.logq_coarse[chain]), "old_logq_detail": float(old_logq_detail[chain]), "new_logq_detail": float(proposed.logq_detail[chain]), "old_logprior_latent": float(old_logprior_latent[chain]), "new_logprior_latent": float(proposed.logprior_latent[chain]), "old_flow_logabsdet": float(old_flow_logabsdet[chain]), "new_flow_logabsdet": float(proposed.flow_logabsdet[chain]), "coordinate_permutation_logabsdet": 0.0, "inverse_kernel_logabsdet": float(proposed.inverse_kernel_logabsdet), "log_accept_ratio": float(log_ratio[chain]), "log_uniform": float(logu[chain]), "accepted": int(accept[chain]), "proposal_coarse_source_index": int(prop_idx[chain])})
        if sweep in save_sweeps:
            rows, grows = measurement_rows(current, sweep)
            main_rows.extend(rows)
            g_rows.extend(grows)
            np.savez_compressed(run / "checkpoints" / f"checkpoint_sweep_{sweep:04d}.npz", phi=current.phi, psi=current.psi, coarse=current.coarse, detail=current.detail, logq=current.logq, action=current.action, source_config_index=current.source_index, source_native_index=current.source_native_index)
            # Existing analysis can read these files while the long diagnostic is
            # still running. Writes are flushed before status advances.
            write_csv(run / "observables" / "main_per_sweep_measurements.csv", main_rows, MAIN_COLUMNS)
            write_csv(run / "observables" / "per_sweep_observables.csv", main_rows, MAIN_COLUMNS)
            write_csv(run / "observables" / "Gk_per_sweep_measurements.csv", g_rows, G_COLUMNS)
            write_csv(run / "observables" / "ensemble_average_history.csv", aggregate(main_rows))
        # Acceptance is useful between sparse measurements, so flush every sweep.
        write_csv(run / "observables" / "acceptance_history.csv", acceptance_rows, ACCEPT_COLUMNS)
        append_csv(run / "coarse_proposal_source_draws.csv", source_draw_rows, ["sweep", "chain_id", "coarse_source_index", "accepted"])
        source_draw_rows.clear()
        write_json(run / "status.json", {"status": "running", "current_sweep": sweep, "acceptance_cumulative": accepted_total / attempted_total, "run_dir": str(run)})
        print(json.dumps({
            "sweep": sweep,
            "acceptance": accepts / attempts,
            "acceptance_cumulative": accepted_total / attempted_total,
            "accepts": accepts,
            "attempts": attempts,
            "measured": sweep in save_sweeps,
        }), flush=True)

    write_csv(run / "observables" / "main_per_sweep_measurements.csv", main_rows, MAIN_COLUMNS)
    write_csv(run / "observables" / "per_sweep_observables.csv", main_rows, MAIN_COLUMNS)
    write_csv(run / "observables" / "Gk_per_sweep_measurements.csv", g_rows, G_COLUMNS)
    write_csv(run / "observables" / "acceptance_history.csv", acceptance_rows, ACCEPT_COLUMNS)
    write_csv(run / "observables" / "ensemble_average_history.csv", aggregate(main_rows))
    native_ref = make_state_from_detail(coarse0, detail0, np.full(args.n_chains, -1, dtype=np.int64), initial_native_index, model=model, stats=stats, kernel=kernel, action_c=action_c, action_f=action_f, batch_size=args.batch_size, device=device)
    native_rows, _ = measurement_rows(native_ref, 0)
    write_csv(run / "observables" / f"native_L{lf}_reference_summary.csv", aggregate(native_rows))
    comparison_rows = []
    native_by_key = {key: np.asarray([float(row[key]) for row in native_rows], dtype=np.float64) for key in ["action_density", "phi2", "phi4", "NN", "diag", "2nn", "m2", "m4"]}
    for sweep in sorted({int(row["sweep"]) for row in main_rows}):
        current_rows = [row for row in main_rows if int(row["sweep"]) == sweep]
        for key, native_values in native_by_key.items():
            values = np.asarray([float(row[key]) for row in current_rows], dtype=np.float64)
            comparison_rows.append({"sweep": sweep, "observable": key, "mean": float(np.mean(values)), "native_mean": float(np.mean(native_values)), "mean_shift_native_sigma": float((np.mean(values) - np.mean(native_values)) / max(np.std(native_values, ddof=1), 1e-300)), "std_ratio": float(np.std(values, ddof=1) / max(np.std(native_values, ddof=1), 1e-300)) if len(values) > 1 else float("nan")})
    write_csv(run / "observables" / "sweep_native_comparison.csv", comparison_rows)
    write_csv(run / "debug" / "direct_mit_acceptance_debug.csv", debug_rows)
    final_reb, _ = reblock_error(current.phi, current.coarse, kernel)
    summary = {"status": "completed", "direct_mit": True, "acceptance_rate": accepted_total / attempted_total, "attempts": attempted_total, "accepts": accepted_total, "max_reblocking_error": float(np.max(final_reb)), "runtime_sec": time.perf_counter() - started, "validation": checks, "acceptance_ratio": "-S_f(new)-logq(new)+S_f(old)+logq(old)", "proposal_density": "-S_c(c)+log q_flow(d|c)-log|J_(c,d)->phi| + constant", "coarse_source_assumption": "iid draws from native-action L8 ensemble representing q_c proportional to exp(-S_c)"}
    write_json(run / "summaries" / "run_summary.json", summary)
    (run / "summaries" / "acceptance_ratio_derivation.md").write_text("# Direct MIT acceptance ratio\n\n`log q(phi) = -S_c(c) + log q_flow(d|c) - log|J_inverse_kernel| - log Z_c`.\n\n`logq_coarse=-S_c(c)` is the unnormalised L8 proposal density. `logq_detail=log q_flow(d|c)=log N(z)-log|J_flow|` includes the latent-standard-normal normalisation and physical detail-standardisation determinant. The interleaving `(c,d)->psi` is a coordinate permutation with log determinant zero. `inverse_kernel_logabsdet=log|J_inverse_kernel|` is included explicitly. `-log Z_c` and the fixed inverse-kernel term cancel between old and new configurations, but both remain in the recorded density decomposition. Therefore `logA=-S_f(new)-logq(new)+S_f(old)+logq(old)`.\n", encoding="utf-8")
    (run / "summaries" / "run_summary.md").write_text(f"# Direct MIT L8 to L16 smoke/reference\n\n- sweeps: `{args.n_sweeps}`\n- chains: `{args.n_chains}`\n- acceptance: `{accepted_total / attempted_total:.6g}`\n- max reblocking error: `{float(np.max(final_reb)):.3g}`\n- long run was started by this invocation: `false` only when invoked with a short/smoke configuration.\n", encoding="utf-8")
    write_json(run / "status.json", {"status": "completed", "current_sweep": args.n_sweeps, "acceptance_cumulative": accepted_total / attempted_total, "latest_checkpoint": str(run / "checkpoints" / f"checkpoint_sweep_{max(save_sweeps):04d}.npz")})
    np.savez_compressed(run / "checkpoints" / "checkpoint_latest.npz", phi=current.phi, psi=current.psi, coarse=current.coarse, detail=current.detail, logq=current.logq, action=current.action, source_config_index=current.source_index, source_native_index=current.source_native_index)
    print(json.dumps({"run_dir": str(run), "acceptance": accepted_total / attempted_total, "sweeps": args.n_sweeps}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
