#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import platform
import socket
import subprocess
import sys
import time
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
from perfect_blocking_upsampling.kernels import apply_kernel, inverse_kernel, load_kernel  # noqa: E402
from run_lam0p2_flow_detail_rethermalization import (  # noqa: E402
    DEFAULT_AR,
    DEFAULT_BASELINE,
    DEFAULT_KERNEL,
    DEFAULT_NATIVE_L32,
    EXPECTED_KERNEL_SHA256,
    MAIN_MEASUREMENT_FIELDS,
    aggregate_history,
    git_commit,
    main_measurement_rows,
    rows_for_sweep,
    sha256,
    write_main_measurement_rows,
    write_per_chain_rows,
)
from run_lam0p2_rand5x5_0084_detail_only_correction_diagnostic import write_csv, write_json  # noqa: E402
from run_lam0p2_residual_flow_patch_chain import StreamingCsv, load_initializer, patch_correct, read_phi, sample_initializer  # noqa: E402

LAM_DEFAULT = 0.2
KAPPA_DEFAULT = 0.323124

RUN_HISTORY_FIELDS = [
    "chain_id",
    "update_mode",
    "coarse_source",
    "source_config_index",
    "source_native_L32_index",
    "sweep",
    "coarse_acceptance",
    "coarse_proposals",
    "coarse_accepts",
    "detail_acceptance",
    "detail_proposals",
    "detail_accepts",
    "acceptance_detail",
    "proposed_detail_updates",
    "accepted_detail_updates",
    "conditional_flow_refreshes",
    "reblocking_max_error",
    "reblocking_rms_error",
    "coarse_reblocking_max_error",
    "coarse_reblocking_rms_error",
    "nonfinite_count",
]

PATCH_HISTORY_FIELDS = [
    "sweep",
    "phase",
    "patch_size",
    "pass",
    "patch_index",
    "patch_x",
    "patch_y",
    "attempts",
    "accepted",
    "acceptance",
    "A_over_R",
    "deltaS_mean",
    "deltaS_std",
    "deltaS_min",
    "deltaS_max",
    "delta_logw_mean",
    "delta_logw_std",
    "log_accept_mean",
    "log_accept_std",
    "patch_l2_mean",
    "local_rms",
    "elapsed_sec",
]

FINE_MCMC_HISTORY_FIELDS = [
    "sweep",
    "phase",
    "pass",
    "update_order",
    "parity",
    "sites_touched",
    "attempts",
    "accepted",
    "acceptance",
    "DeltaSf_mean",
    "DeltaSf_std",
    "DeltaSf_min",
    "DeltaSf_max",
    "log_accept_mean",
    "log_accept_std",
    "patch_l2_mean",
    "local_rms",
    "elapsed_sec",
    "acceptance_formula",
]


def configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)
        except Exception:
            pass


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Detmold-style full fine-lattice rethermalization after inverse-blocking flow initialization.")
    ap.add_argument("--coarse-ensemble", type=Path, required=True)
    ap.add_argument("--coarse-L", "--coarse-l", dest="coarse_L", type=int, required=True)
    ap.add_argument("--fine-L", "--fine-l", dest="fine_L", type=int, required=True)
    ap.add_argument("--lambda", dest="lam", type=float, default=LAM_DEFAULT)
    ap.add_argument("--kappa-coarse", type=float, default=KAPPA_DEFAULT)
    ap.add_argument("--kappa-fine", type=float, default=KAPPA_DEFAULT)
    ap.add_argument("--kernel-config", "--kernel-path", dest="kernel_config", type=Path, default=DEFAULT_KERNEL)
    ap.add_argument("--flow-checkpoint", "--ar-checkpoint", dest="flow_checkpoint", type=Path, default=DEFAULT_AR)
    ap.add_argument("--baseline-checkpoint", type=Path, default=DEFAULT_BASELINE)
    ap.add_argument("--output-dir", "--out-dir", dest="output_dir", type=Path, required=True)
    ap.add_argument("--n-chains", "--chains", dest="n_chains", type=int, default=16)
    ap.add_argument("--source-start-index", "--start-index", dest="source_start_index", type=int, default=0)
    ap.add_argument("--seed", type=int, default=2026071001)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--source-kind", choices=["direct_native_L16", "blocked_native_L32", "generic_coarse"], default="generic_coarse")
    ap.add_argument("--native-reference", type=Path, default=DEFAULT_NATIVE_L32)

    ap.add_argument("--detail-precondition-sweeps", type=int, default=0)
    ap.add_argument("--detail-patch-size", type=int, default=8)
    ap.add_argument("--detail-passes", type=int, default=1)
    ap.add_argument("--detail-step-size", type=float, default=0.08)
    ap.add_argument("--detail-beta-z", type=float, default=None, help="Alias metadata for detail proposal scale; the current detail updater uses --detail-step-size.")

    ap.add_argument("--fine-mcmc-sweeps", type=int, default=10)
    ap.add_argument("--fine-mcmc-algorithm", choices=["single_site_metropolis"], default="single_site_metropolis")
    ap.add_argument("--fine-mcmc-update-order", choices=["checkerboard", "sequential"], default="checkerboard")
    ap.add_argument("--fine-mcmc-passes", "--fine-passes", dest="fine_mcmc_passes", type=int, default=1)
    ap.add_argument("--fine-mcmc-step-size", "--fine-step-size", dest="fine_mcmc_step_size", type=float, default=0.08)
    ap.add_argument("--fine-mcmc-target-acceptance", type=float, default=None)
    ap.add_argument("--fine-mcmc-tune-step-size", action="store_true")
    ap.add_argument("--deltaS-validation-proposals", type=int, default=32)
    ap.add_argument("--measure-every", type=int, default=1)
    ap.add_argument("--checkpoint-every", type=int, default=10)
    ap.add_argument("--resume", action="store_true")

    ap.add_argument("--initializer-kind", choices=["ar"], default="ar")
    ap.add_argument("--n-coupling-layers", type=int, default=8)
    ap.add_argument("--hidden-channels", type=int, default=32)
    ap.add_argument("--conv-kernel-size", type=int, default=5)
    ap.add_argument("--log-scale-bound", type=float, default=0.75)
    return ap.parse_args()


def load_coarse(path: Path, coarse_l: int, n: int, start: int) -> tuple[np.ndarray, np.ndarray]:
    phi = read_phi(path, coarse_l)
    stop = start + n
    if stop > len(phi):
        raise SystemExit(f"requested source range [{start}, {stop}) but {path} contains only {len(phi)} configs")
    idx = np.arange(start, stop, dtype=np.int64)
    return phi[idx].astype(np.float32), idx


def local_single_site_delta_s(phi: np.ndarray, x: int, y: int, delta: np.ndarray, action: ActionSpec) -> np.ndarray:
    if action.type != "phi4_nn":
        raise ValueError(f"single-site local delta currently supports phi4_nn, got {action.type}")
    old = phi[:, x, y].astype(np.float64)
    new = old + delta.astype(np.float64)
    nn_sum = (
        phi[:, (x + 1) % phi.shape[1], y]
        + phi[:, (x - 1) % phi.shape[1], y]
        + phi[:, x, (y + 1) % phi.shape[2]]
        + phi[:, x, (y - 1) % phi.shape[2]]
    ).astype(np.float64)
    a = 1.0 - 2.0 * float(action.lambda_)
    onsite = a * (new * new - old * old) + float(action.lambda_) * (new**4 - old**4)
    bonds = -2.0 * float(action.kappa) * (new - old) * nn_sum
    return onsite + bonds


def validate_local_delta_s(
    phi: np.ndarray,
    action: ActionSpec,
    rng: np.random.Generator,
    n_checks: int,
) -> dict[str, Any]:
    errs: list[float] = []
    rels: list[float] = []
    for _ in range(max(0, int(n_checks))):
        chain = int(rng.integers(0, len(phi)))
        x = int(rng.integers(0, phi.shape[1]))
        y = int(rng.integers(0, phi.shape[2]))
        delta = np.asarray([rng.uniform(-0.1, 0.1)], dtype=np.float64)
        local = float(local_single_site_delta_s(phi[chain : chain + 1], x, y, delta, action)[0])
        prop = phi[chain : chain + 1].copy()
        old_s = float(action_total(prop, action)[0])
        prop[0, x, y] += np.float32(delta[0])
        full = float(action_total(prop, action)[0] - old_s)
        err = abs(local - full)
        errs.append(err)
        rels.append(err / max(abs(full), 1.0e-12))
    return {
        "n_checks": int(len(errs)),
        "max_abs_error": float(np.max(errs)) if errs else float("nan"),
        "rms_abs_error": float(np.sqrt(np.mean(np.asarray(errs) ** 2))) if errs else float("nan"),
        "max_rel_error": float(np.max(rels)) if rels else float("nan"),
        "rms_rel_error": float(np.sqrt(np.mean(np.asarray(rels) ** 2))) if rels else float("nan"),
    }


def full_fine_single_site_metropolis(
    phi0: np.ndarray,
    action: ActionSpec,
    sweep: int,
    passes: int,
    step_size: float,
    update_order: str,
    rng: np.random.Generator,
    writer: StreamingCsv,
) -> tuple[np.ndarray, dict[str, Any]]:
    phi = phi0.copy().astype(np.float32)
    attempts = 0
    accepts = 0
    block_accs: list[float] = []
    log_accept_values: list[float] = []
    start = time.perf_counter()
    L = int(phi.shape[1])
    for p in range(int(passes)):
        groups: list[tuple[str, list[tuple[int, int]]]]
        if update_order == "sequential":
            groups = [("all", [(x, y) for x in range(L) for y in range(L)])]
        else:
            groups = [
                ("even", [(x, y) for x in range(L) for y in range(L) if (x + y) % 2 == 0]),
                ("odd", [(x, y) for x in range(L) for y in range(L) if (x + y) % 2 == 1]),
            ]
        for parity, sites in groups:
            group_attempts = 0
            group_accepts = 0
            group_delta_s: list[float] = []
            group_log_accept: list[float] = []
            for x, y in sites:
                delta = rng.uniform(-step_size, step_size, size=len(phi)).astype(np.float64)
                delta_s = local_single_site_delta_s(phi, x, y, delta, action)
                log_accept = np.minimum(0.0, -delta_s)
                accept = np.log(rng.random(len(phi))) < log_accept
                if np.any(accept):
                    phi[accept, x, y] += delta[accept].astype(np.float32)
                n_accept = int(np.sum(accept))
                attempts += int(len(phi))
                accepts += n_accept
                group_attempts += int(len(phi))
                group_accepts += n_accept
                group_delta_s.extend([float(v) for v in delta_s])
                group_log_accept.extend([float(v) for v in log_accept])
            acc = float(group_accepts / group_attempts) if group_attempts else float("nan")
            block_accs.append(acc)
            log_accept_values.extend(group_log_accept)
            writer.write(
                {
                    "sweep": int(sweep),
                    "phase": "fine_rethermalization",
                    "pass": int(p),
                    "update_order": update_order,
                    "parity": parity,
                    "sites_touched": int(len(sites)),
                    "attempts": int(group_attempts),
                    "accepted": int(group_accepts),
                    "acceptance": acc,
                    "DeltaSf_mean": float(np.mean(group_delta_s)) if group_delta_s else float("nan"),
                    "DeltaSf_std": float(np.std(group_delta_s, ddof=1)) if len(group_delta_s) > 1 else 0.0,
                    "DeltaSf_min": float(np.min(group_delta_s)) if group_delta_s else float("nan"),
                    "DeltaSf_max": float(np.max(group_delta_s)) if group_delta_s else float("nan"),
                    "log_accept_mean": float(np.mean(group_log_accept)) if group_log_accept else float("nan"),
                    "log_accept_std": float(np.std(group_log_accept, ddof=1)) if len(group_log_accept) > 1 else 0.0,
                    "patch_l2_mean": float("nan"),
                    "local_rms": float(step_size / math.sqrt(3.0)),
                    "elapsed_sec": float(time.perf_counter() - start),
                    "acceptance_formula": "single-site uniform random walk; log_accept=min(0,-local_delta_S_f); no coarse action, flow density, latent prior, or Jacobian terms",
                }
            )
    return phi.astype(np.float32), {
        "fine_acceptance": float(accepts / attempts) if attempts else float("nan"),
        "fine_proposals": int(attempts),
        "fine_accepts": int(accepts),
        "fine_single_site_blocks": int(len(block_accs)),
        "fine_block_acceptance": float(np.mean(block_accs)) if block_accs else float("nan"),
        "fine_log_accept_mean": float(np.mean(log_accept_values)) if log_accept_values else float("nan"),
        "fine_sites_attempted_per_pass": int(L * L),
    }


def make_run_rows(
    phi: np.ndarray,
    psi: np.ndarray,
    kernel: Any,
    action: ActionSpec,
    source_idx: np.ndarray,
    sweep: int,
    phase: str,
    source_kind: str,
    detail_meta: dict[str, Any] | None = None,
    fine_meta: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    meta = {
        "update_mode": "detmold_fine_rethermalization",
        "detail_update_acceptance": float("nan"),
        "detail_update_config_attempts": 0,
        "detail_update_accepts": 0,
        "coarse_acceptance": float("nan"),
        "coarse_proposals": 0,
        "coarse_accepts": 0,
        "conditional_flow_refreshes": 0,
    }
    if detail_meta:
        meta.update(detail_meta)
    rows = rows_for_sweep(phi, psi, kernel, action, source_idx, sweep, meta, source_kind if source_kind == "blocked_native_L32" else "generic")
    for row in rows:
        row["phase"] = phase
        if fine_meta is not None:
            row["fine_acceptance"] = float(fine_meta.get("fine_acceptance", float("nan")))
            row["fine_proposals"] = int(fine_meta.get("fine_proposals", 0))
            row["fine_accepts"] = int(fine_meta.get("fine_accepts", 0))
    return rows


def save_checkpoint(out: Path, sweep: int, phase: str, phi: np.ndarray, psi: np.ndarray, source_idx: np.ndarray, rng: np.random.Generator, run_config: dict[str, Any]) -> None:
    ckpt_dir = out / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ckpt_dir / f"state_sweep{sweep:06d}.npz", phi=phi.astype(np.float32), psi=psi.astype(np.float32), source_config_index=source_idx, sweep=np.asarray([sweep]), phase=np.asarray([phase]))
    state = {
        "sweep": int(sweep),
        "phase": phase,
        "rng_state": rng.bit_generator.state,
        "run_config": run_config,
    }
    write_json(ckpt_dir / f"state_sweep{sweep:06d}.json", state)
    write_json(ckpt_dir / "latest_checkpoint.json", {"npz": f"state_sweep{sweep:06d}.npz", "json": f"state_sweep{sweep:06d}.json", "sweep": int(sweep), "phase": phase})


def load_latest_checkpoint(out: Path, rng: np.random.Generator) -> tuple[int, str, np.ndarray, np.ndarray, np.ndarray, np.random.Generator]:
    latest = out / "checkpoints" / "latest_checkpoint.json"
    if not latest.exists():
        raise SystemExit(f"--resume requested but no checkpoint found at {latest}")
    meta = json.loads(latest.read_text(encoding="utf-8"))
    npz_path = latest.parent / meta["npz"]
    json_path = latest.parent / meta["json"]
    with np.load(npz_path) as z:
        phi = z["phi"].astype(np.float32)
        psi = z["psi"].astype(np.float32)
        source_idx = z["source_config_index"].astype(np.int64)
        sweep = int(z["sweep"][0])
        phase = str(z["phase"][0])
    state = json.loads(json_path.read_text(encoding="utf-8"))
    rng.bit_generator.state = state["rng_state"]
    return sweep, phase, phi, psi, source_idx, rng


def measured_sweeps(total: int, every: int) -> set[int]:
    every = max(1, int(every))
    return set(range(0, total + 1, every)) | {int(total)}


def detmold_measure_sweeps(detail_sweeps: int, total: int, every: int) -> set[int]:
    base = measured_sweeps(total, every)
    # The initial detail-preconditioning stage is short and diagnostic-critical:
    # always record every detail sweep even when the fine-MC output interval is coarse.
    base.update(range(0, max(0, int(detail_sweeps)) + 1))
    return base


def coarse_drift_rows(phi: np.ndarray, source_coarse: np.ndarray, kernel: Any, source_idx: np.ndarray, sweep: int, phase: str) -> list[dict[str, Any]]:
    blocked = apply_kernel(phi.astype(np.float32), kernel).astype(np.float32)[:, 0::2, 0::2]
    diff = blocked - source_coarse.astype(np.float32)
    rows: list[dict[str, Any]] = []
    for i in range(len(phi)):
        rows.append(
            {
                "chain_id": int(i),
                "sweep": int(sweep),
                "phase": phase,
                "source_config_index": int(source_idx[i]),
                "blocked_minus_source_max_abs": float(np.max(np.abs(diff[i]))),
                "blocked_minus_source_rms": float(np.sqrt(np.mean(diff[i].astype(np.float64) ** 2))),
            }
        )
    return rows


def main() -> int:
    configure_stdio()
    args = parse_args()
    t0 = time.perf_counter()
    out = args.output_dir
    for sub in ["configs", "logs", "scripts", "manifests", "observables", "checkpoints", "final_configurations", "plots"]:
        (out / sub).mkdir(parents=True, exist_ok=True)
    for path in [args.coarse_ensemble, args.kernel_config, args.flow_checkpoint, args.baseline_checkpoint]:
        if not path.exists():
            raise SystemExit(f"missing required input: {path}")
    kernel_sha = sha256(args.kernel_config)
    if args.kernel_config == DEFAULT_KERNEL and kernel_sha != EXPECTED_KERNEL_SHA256:
        raise SystemExit(f"kernel checksum mismatch for canonical rand5x5_0084: {kernel_sha}")
    kernel, kernel_json = load_kernel(args.kernel_config)
    fine_action = ActionSpec("phi4_nn", float(args.lam), float(args.kappa_fine))
    run_config = {
        "command": " ".join(sys.argv),
        "git_commit": git_commit(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch_version": torch.__version__,
        "date_unix": time.time(),
        "lambda": float(args.lam),
        "kappa_coarse": float(args.kappa_coarse),
        "kappa_fine": float(args.kappa_fine),
        "coarse_L": int(args.coarse_L),
        "fine_L": int(args.fine_L),
        "coarse_ensemble": str(args.coarse_ensemble.resolve()),
        "kernel_config": str(args.kernel_config.resolve()),
        "kernel_sha256": kernel_sha,
        "kernel": kernel_json,
        "flow_checkpoint": str(args.flow_checkpoint.resolve()),
        "baseline_checkpoint": str(args.baseline_checkpoint.resolve()),
        "output_dir": str(out.resolve()),
        "n_chains": int(args.n_chains),
        "source_start_index": int(args.source_start_index),
        "seed": int(args.seed),
        "detail_precondition_sweeps": int(args.detail_precondition_sweeps),
        "detail_patch_size": int(args.detail_patch_size),
        "detail_passes": int(args.detail_passes),
        "detail_step_size": float(args.detail_step_size),
        "fine_mcmc_sweeps": int(args.fine_mcmc_sweeps),
        "fine_mcmc_algorithm": args.fine_mcmc_algorithm,
        "fine_mcmc_update_order": args.fine_mcmc_update_order,
        "fine_mcmc_passes": int(args.fine_mcmc_passes),
        "fine_mcmc_step_size": float(args.fine_mcmc_step_size),
        "fine_mcmc_target_acceptance": args.fine_mcmc_target_acceptance,
        "fine_mcmc_tune_step_size": bool(args.fine_mcmc_tune_step_size),
        "measure_every": int(args.measure_every),
        "checkpoint_every": int(args.checkpoint_every),
        "fine_mcmc_acceptance_formula": "single-site uniform random walk; log_accept=min(0,-local_delta_S_f); no S_c, latent prior, flow density, or flow Jacobian",
    }
    write_json(out / "run_config.json", run_config)
    write_json(out / "run_manifest.json", {**run_config, "status": "running"})
    write_json(out / "manifests" / "run_manifest.json", {**run_config, "status": "running"})

    rng = np.random.default_rng(args.seed)
    source_idx: np.ndarray
    source_coarse: np.ndarray
    if args.resume:
        start_sweep, start_phase, phi, psi, source_idx, rng = load_latest_checkpoint(out, rng)
        source_coarse, _ = load_coarse(args.coarse_ensemble, int(args.coarse_L), len(source_idx), int(source_idx[0]))
        print(json.dumps({"resume": True, "sweep": start_sweep, "phase": start_phase}), flush=True)
    else:
        coarse, source_idx = load_coarse(args.coarse_ensemble, int(args.coarse_L), int(args.n_chains), int(args.source_start_index))
        source_coarse = coarse.copy()
        init_args = argparse.Namespace(**vars(args))
        init_args.best_checkpoint = args.flow_checkpoint
        init_args.ar_checkpoint = args.flow_checkpoint
        init_args.kernel_path = args.kernel_config
        init_args.from_L = int(args.coarse_L)
        init_args.to_L = int(args.fine_L)
        init_args.batch_size = int(args.batch_size)
        init_args.device = args.device
        model, predictor, stats = load_initializer(init_args)
        psi, _detail, _y = sample_initializer(model, predictor, stats, coarse, init_args)
        phi, _ = inverse_kernel(psi, kernel)
        if phi.shape[1:] != (args.fine_L, args.fine_L):
            raise SystemExit(f"flow initialization produced {phi.shape}, expected fine L={args.fine_L}")
        delta_s_check = validate_local_delta_s(phi, fine_action, rng, int(args.deltaS_validation_proposals))
        run_config["local_deltaS_validation"] = delta_s_check
        write_json(out / "run_config.json", run_config)
        write_json(out / "run_manifest.json", {**run_config, "status": "running"})
        write_json(out / "manifests" / "run_manifest.json", {**run_config, "status": "running"})
        np.savez_compressed(out / "configs" / "coarse_sources.npz", phi=coarse, source_config_index=source_idx)
        start_sweep = 0
        start_phase = "initialization"
        save_checkpoint(out, 0, start_phase, phi, psi, source_idx, rng, run_config)

    per_path = out / "observables" / "per_sweep_observables.csv"
    main_path = out / "observables" / "main_per_sweep_measurements.csv"
    acceptance_path = out / "observables" / "acceptance_history.csv"
    drift_path = out / "observables" / "coarse_coordinate_drift.csv"
    main_rows_all: list[dict[str, Any]] = []
    run_rows_all: list[dict[str, Any]] = []
    acceptance_rows: list[dict[str, Any]] = []
    drift_rows_all: list[dict[str, Any]] = []

    append = bool(args.resume)
    if not append:
        run_rows = make_run_rows(phi, psi, kernel, fine_action, source_idx, 0, "initialization", args.source_kind)
        meas_rows = main_measurement_rows(phi, fine_action, source_idx, 0, args.source_kind)
        write_per_chain_rows(per_path, run_rows, append=False)
        write_csv(out / "per_sweep_observables.csv", run_rows)
        write_main_measurement_rows(main_path, meas_rows, append=False)
        run_rows_all.extend(run_rows)
        main_rows_all.extend(meas_rows)
        acceptance_rows.append({"sweep": 0, "phase": "initialization", "detail_acceptance": float("nan"), "fine_acceptance": float("nan"), "detail_proposals": 0, "fine_proposals": 0})
        write_csv(acceptance_path, acceptance_rows)
        drift_rows = coarse_drift_rows(phi, source_coarse, kernel, source_idx, 0, "initialization")
        drift_rows_all.extend(drift_rows)
        write_csv(drift_path, drift_rows_all)

    total_global = int(args.detail_precondition_sweeps) + int(args.fine_mcmc_sweeps)
    measure_set = detmold_measure_sweeps(int(args.detail_precondition_sweeps), total_global, int(args.measure_every))
    current = int(start_sweep)
    while current < total_global:
        current += 1
        if current <= int(args.detail_precondition_sweeps):
            phase = "detail_preconditioning"
            detail_patch_path = out / "logs" / "detail_preconditioning_patch_history.csv"
            patch_writer = StreamingCsv(detail_patch_path, PATCH_HISTORY_FIELDS, append=detail_patch_path.exists())
            pc_args = argparse.Namespace(
                disable_coarse_updates=True,
                detail_passes=int(args.detail_passes),
                fine_proposal_sigma=float(args.detail_step_size),
                fine_patch_size=int(args.detail_patch_size),
                passes=0,
                proposal_sigma=0.12,
                coarse_patch_size=int(args.detail_patch_size),
                global_sweep=current,
                verbose_patch_log=False,
            )
            phi, psi, dmeta = patch_correct(psi, kernel, fine_action, pc_args, patch_writer, rng)
            patch_writer.close()
            fine_meta = None
            detail_meta = dmeta
        else:
            phase = "fine_rethermalization"
            fine_history_path = out / "logs" / "fine_mcmc_site_history.csv"
            writer = StreamingCsv(fine_history_path, FINE_MCMC_HISTORY_FIELDS, append=fine_history_path.exists())
            phi, fine_meta = full_fine_single_site_metropolis(
                phi,
                fine_action,
                current,
                int(args.fine_mcmc_passes),
                float(args.fine_mcmc_step_size),
                args.fine_mcmc_update_order,
                rng,
                writer,
            )
            writer.close()
            psi = apply_kernel(phi, kernel).astype(np.float32)
            detail_meta = None

        if current in measure_set:
            rr = make_run_rows(phi, psi, kernel, fine_action, source_idx, current, phase, args.source_kind, detail_meta=detail_meta, fine_meta=fine_meta)
            mr = main_measurement_rows(phi, fine_action, source_idx, current, args.source_kind)
            write_per_chain_rows(per_path, rr, append=True)
            write_main_measurement_rows(main_path, mr, append=True)
            run_rows_all.extend(rr)
            main_rows_all.extend(mr)
            acceptance_rows.append(
                {
                    "sweep": int(current),
                    "phase": phase,
                    "detail_acceptance": float(detail_meta.get("detail_update_acceptance", float("nan"))) if detail_meta else float("nan"),
                    "fine_acceptance": float(fine_meta.get("fine_acceptance", float("nan"))) if fine_meta else float("nan"),
                    "detail_proposals": int(detail_meta.get("detail_update_config_attempts", 0)) if detail_meta else 0,
                    "fine_proposals": int(fine_meta.get("fine_proposals", 0)) if fine_meta else 0,
                }
            )
            write_csv(acceptance_path, acceptance_rows)
            drift_rows = coarse_drift_rows(phi, source_coarse, kernel, source_idx, current, phase)
            drift_rows_all.extend(drift_rows)
            write_csv(drift_path, drift_rows_all)
            write_csv(out / "observables" / "ensemble_average_history.csv", aggregate_history(main_rows_all, run_rows_all))
            print(json.dumps({"sweep": current, "phase": phase, "fine_acceptance": None if fine_meta is None else fine_meta.get("fine_acceptance"), "detail_acceptance": None if detail_meta is None else detail_meta.get("detail_update_acceptance")}), flush=True)
        checkpoint_every = int(args.checkpoint_every)
        should_checkpoint = (
            current == total_global
            if checkpoint_every <= 0
            else (current % checkpoint_every == 0 or current == total_global)
        )
        if should_checkpoint:
            save_checkpoint(out, current, phase, phi, psi, source_idx, rng, run_config)
            np.savez_compressed(out / "final_configurations" / f"configs_sweep{current:06d}.npz", phi=phi.astype(np.float32), psi=psi.astype(np.float32), source_config_index=source_idx)

    if main_rows_all and run_rows_all:
        hist = aggregate_history(main_rows_all, run_rows_all)
        write_csv(out / "ensemble_average_history.csv", hist)
        write_csv(out / "per_sweep_observables.csv", run_rows_all)
    summary = {
        **run_config,
        "status": "completed",
        "runtime_sec": float(time.perf_counter() - t0),
        "final_sweep": int(total_global),
        "final_phase": "fine_rethermalization" if args.fine_mcmc_sweeps > 0 else "detail_preconditioning",
        "output_files": {
            "main_per_sweep_measurements": str(main_path),
            "per_sweep_observables": str(per_path),
            "acceptance_history": str(acceptance_path),
            "ensemble_average_history": str(out / "observables" / "ensemble_average_history.csv"),
        },
    }
    write_json(out / "summary.json", summary)
    write_json(out / "run_manifest.json", summary)
    print(json.dumps({"status": "completed", "output_dir": str(out), "runtime_sec": summary["runtime_sec"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
