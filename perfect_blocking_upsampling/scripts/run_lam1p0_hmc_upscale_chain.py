#!/usr/bin/env python3
"""Modular factor-two flow-upscaling chain with direct fine-field HMC at each level."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "perfect_blocking_upsampling"
sys.path[:0] = [str(PKG / "src"), str(PKG / "scripts")]

from perfect_blocking_upsampling.blocking import assemble_psi, inverse_kernel, load_kernel_matrix, load_phi
from perfect_blocking_upsampling.io import ActionSpec
from run_lam1p0_fine_hmc import hmc_trajectory, measure_phi, write_measurements, ACCEPT_COLUMNS
from run_lam1p0_mit_style_inverse_blocking_L8to16 import aggregate, write_csv, write_json
from run_lam1p0_l16to32_rqspline_zeroshot import build_model_from_checkpoint, sample_model_lattice, stationary_stats


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def run_level(phi_coarse: np.ndarray, level: dict, *, root: Path, common: dict, action: ActionSpec, source_indices: np.ndarray, level_seed: int) -> np.ndarray:
    lc = int(phi_coarse.shape[1]); lf = int(level["to_L"])
    if lf != 2 * lc:
        raise ValueError(f"level L{lc}->L{lf} must be factor two")
    divide = int(level["divide"])
    if lf % divide:
        raise ValueError(f"L{lf} must be divisible by divide={divide}")
    run = root / f"L{lc}toL{lf}"
    for name in ("observables", "checkpoints", "summaries", "logs"):
        (run / name).mkdir(parents=True, exist_ok=True)
    flow_path, kernel_path = resolve(level["flow_checkpoint"]), resolve(level["kernel_path"])
    kernel, kernel_meta = load_kernel_matrix(kernel_path)
    if not bool(kernel_meta.get("kernel_coefficients_include_eta_scale", False)):
        raise ValueError(f"kernel lacks eta-scale convention: {kernel_path}")
    ck = torch.load(flow_path, map_location="cpu", weights_only=False)
    model, report = build_model_from_checkpoint(ck, lc, torch.device("cpu"))
    stats = stationary_stats(ck["state"]["stats"], lc)
    rng = np.random.default_rng(level_seed)
    # Keep the flow/inverse map bounded at L256 and above.  When requested,
    # store the initialized fine ensemble as a memmap while batches are
    # produced, rather than retaining all L_f fields in RAM.
    flow_batch = int(common["batch_size"])
    stream_initialization = bool(level.get("stream_initialization", False))
    init_store = run / "checkpoints" / "initialization_phi.npy"
    if stream_initialization:
        phi = np.lib.format.open_memmap(init_store, mode="w+", dtype=np.float32, shape=(len(phi_coarse), lf, lf))
    else:
        phi = np.empty((len(phi_coarse), lf, lf), dtype=np.float32)
    for lo in range(0, len(phi_coarse), flow_batch):
        hi = min(lo + flow_batch, len(phi_coarse))
        detail, _, _, _ = sample_model_lattice(
            model, phi_coarse[lo:hi].astype(np.float32), stats, batch_size=flow_batch,
            device=torch.device("cpu"), seed=int(rng.integers(2**31 - 1)),
        )
        phi[lo:hi], _ = inverse_kernel(assemble_psi(phi_coarse[lo:hi].astype(np.float32), detail), kernel)
        if stream_initialization:
            phi.flush()
            write_json(run / "checkpoints" / "initialization_progress.json", {
                "status": "running", "completed_chains": hi, "total_chains": len(phi_coarse),
                "working_fields": str(init_store),
            })
        print(json.dumps({"level": f"L{lc}toL{lf}", "initialization_chains": hi,
                          "initialization_total": len(phi_coarse)}), flush=True)
    if stream_initialization:
        phi.flush()
        write_json(run / "checkpoints" / "initialization_progress.json", {
            "status": "completed", "completed_chains": len(phi_coarse), "total_chains": len(phi_coarse),
            "working_fields": str(init_store),
        })
    yy, xx = np.indices((lf, lf))
    sublattices = [((oy, ox), (yy % divide == oy) & (xx % divide == ox)) for oy in range(divide) for ox in range(divide)]
    sweeps, eps, nlf = int(level["thermalization_sweeps"]), float(level["step_size"]), int(level["leapfrog_steps"])
    measure_every = int(level.get("measure_every", 1))
    checkpoint_every = int(level.get("checkpoint_every", 1))
    hmc_batch = int(level.get("hmc_batch_size", len(phi)))
    if measure_every <= 0 or checkpoint_every <= 0 or hmc_batch <= 0:
        raise ValueError("measurement, checkpoint, and HMC batch sizes must be positive")
    nearest_native_L = lf
    native_reference = ROOT / f"data/configs_phi4_2d/lam1p0_kappac0p340301_L{nearest_native_L}/configs.npz"
    # Walk down the available direct-native volumes for a notebook-compatible
    # proxy while a true reference at L_f does not yet exist.
    while not native_reference.exists() and nearest_native_L > 8:
        nearest_native_L //= 2
        native_reference = ROOT / f"data/configs_phi4_2d/lam1p0_kappac0p340301_L{nearest_native_L}/configs.npz"
    if not native_reference.exists():
        raise FileNotFoundError("no direct native lambda=1 reference volume found")
    config = common | {"from_L": lc, "to_L": lf, "thermalization_sweeps": sweeps, "divide": divide, "step_size": eps, "leapfrog_steps": nlf, "measure_every": measure_every, "checkpoint_every": checkpoint_every, "hmc_batch_size": hmc_batch, "stream_initialization": stream_initialization, "flow_checkpoint": str(flow_path), "kernel_path": str(kernel_path), "kernel_metadata": kernel_meta, "flow_load_report": report, "flow_used": "once_before_level_HMC", "target": "exp(-S_f(phi))", "native_reference_source": str(native_reference), "native_reference_L": nearest_native_L, "native_reference_is_nearest_volume_proxy": nearest_native_L != lf, "sweep_semantics": f"{divide**2} fixed residue-class HMC subtrajectories; every L{lf} site touched exactly once per sweep"}
    write_json(run / "run_config.json", config)
    config_text = json.dumps(config, indent=2, default=str) + "\n"
    # Both names are retained: older analysis notebooks expect config.yaml,
    # whereas early chained runs used run_config.yaml.
    (run / "run_config.yaml").write_text(config_text)
    (run / "config.yaml").write_text(config_text)
    main, grows = measure_phi(phi, 0, source_indices); write_measurements(run, main, grows)
    np.savez_compressed(run / "checkpoints" / "checkpoint_sweep_0000.npz", phi=phi)
    acc_rows, accepted_total = [], 0
    for sweep in range(1, sweeps + 1):
        accepts, dh_all = 0, []
        for _, active in sublattices:
            for lo in range(0, len(phi), hmc_batch):
                hi = min(lo + hmc_batch, len(phi))
                phi[lo:hi], accepted, dh = hmc_trajectory(phi[lo:hi], active, action=action, step_size=eps, n_steps=nlf, rng=rng)
                accepts += int(accepted.sum()); dh_all.append(dh)
        accepted_total += accepts; dh = np.concatenate(dh_all); attempted = sweep * len(sublattices) * len(phi)
        acc_rows.append({"sweep": sweep, "subtrajectories": len(sublattices), "active_sites_per_subtrajectory": int(sublattices[0][1].sum()), "accepted": accepts, "attempted": len(sublattices) * len(phi), "acceptance": accepts / (len(sublattices) * len(phi)), "acceptance_cumulative": accepted_total / attempted, "mean_delta_H": float(dh.mean()), "std_delta_H": float(dh.std(ddof=1))})
        if sweep % measure_every == 0 or sweep == sweeps:
            rows, gs = measure_phi(phi, sweep, source_indices); main.extend(rows); grows.extend(gs); write_measurements(run, main, grows)
        if sweep % checkpoint_every == 0 or sweep == sweeps:
            np.savez_compressed(run / "checkpoints" / f"checkpoint_sweep_{sweep:04d}.npz", phi=phi)
        write_csv(run / "observables" / "acceptance_history.csv", acc_rows, ACCEPT_COLUMNS)
        print(json.dumps({"level": f"L{lc}toL{lf}", "sweep": sweep, "acceptance_cumulative": accepted_total / attempted}), flush=True)
    np.savez_compressed(run / "final_phi.npz", phi=phi, source_indices=source_indices)
    write_json(run / "summaries" / "run_summary.json", {"status": "completed", "acceptance": accepted_total / (sweeps * len(sublattices) * len(phi)), "final_field": str(run / "final_phi.npz")})
    return phi


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--config", type=Path, required=True); ap.add_argument("--run-dir", type=Path, required=True); ap.add_argument("--n-chains", type=int); ap.add_argument("--start-index", type=int); args = ap.parse_args()
    cfg = json.loads(args.config.read_text()); run = args.run_dir.resolve()
    # The launcher creates run/logs before spawning us.  Treat a chain config
    # as the irreversible marker that a real run already began.
    if (run / "chain_config.json").exists():
        raise FileExistsError(f"chain already initialized: {run}")
    run.mkdir(parents=True, exist_ok=True)
    if args.n_chains is not None: cfg["n_chains"] = args.n_chains
    if args.start_index is not None: cfg["start_index"] = args.start_index
    n, start = int(cfg["n_chains"]), int(cfg["start_index"])
    phi_all = load_phi(resolve(cfg["initial_source"])); phi = phi_all[start:start+n].astype(np.float32)
    if len(phi) != n: raise ValueError("initial source does not contain requested chains")
    common = {k: cfg[k] for k in ("lambda", "kappa", "n_chains", "start_index", "batch_size", "seed")}
    action = ActionSpec("phi4_nn", float(cfg["lambda"]), float(cfg["kappa"])); source_indices = np.arange(start, start+n, dtype=np.int64)
    write_json(run / "chain_config.json", cfg | {"initial_source_resolved": str(resolve(cfg["initial_source"]))})
    t0 = time.perf_counter()
    for i, level in enumerate(cfg["levels"]): phi = run_level(phi, level, root=run / "levels", common=common, action=action, source_indices=source_indices, level_seed=int(cfg["seed"]) + i)
    np.savez_compressed(run / "final_phi.npz", phi=phi, source_indices=source_indices)
    write_json(run / "status.json", {"status": "completed", "final_L": int(phi.shape[1]), "runtime_sec": time.perf_counter() - t0, "final_field": str(run / "final_phi.npz")})
    return 0

if __name__ == "__main__": raise SystemExit(main())
