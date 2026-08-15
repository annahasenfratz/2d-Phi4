#!/usr/bin/env python3
"""Fine-field HMC thermalization for lambda=1, kappa=0.340301.

The Markov state is the physical fine lattice field phi, not a blocked
coordinate state.  Every trajectory therefore targets exp[-S_f(phi)] directly
and never applies K^{-1} after the sweep-zero flow initialization.
"""
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

from perfect_blocking_upsampling.actions import action_total
from perfect_blocking_upsampling.io import ActionSpec
from run_lam1p0_mit_coordinate_mh_L8to16 import (  # shared, established measurement convention
    aggregate, load_kernel_matrix, load_phi, per_config_observables,
    write_csv, write_json,
)
from run_lam1p0_l16to32_rqspline_zeroshot import (
    build_model_from_checkpoint, sample_model_lattice, stationary_stats,
)
from perfect_blocking_upsampling.blocking import assemble_psi, inverse_kernel


MAIN_COLUMNS = ["chain_id", "sweep", "source_config_index", "source_native_L32_index", "L", "volume", "action_density", "total_action", "phi2", "phi4", "NN", "diag", "2nn", "m", "m2", "m4", "G_pmin_x_cfg", "G_pmin_y_cfg", "nonfinite_count"]
G_COLUMNS = ["chain_id", "sweep", "source_config_index", "source_native_L32_index", "L", "volume", "G_00", "G_10", "G_01", "G_pmin_avg"]
ACCEPT_COLUMNS = ["sweep", "subtrajectories", "active_sites_per_subtrajectory", "accepted", "attempted", "acceptance", "acceptance_cumulative", "mean_delta_H", "std_delta_H"]


def force(phi: np.ndarray, action: ActionSpec) -> np.ndarray:
    """dS/dphi for the exact action used by action_total."""
    x = np.asarray(phi, dtype=np.float64)
    neighbours = (
        np.roll(x, 1, axis=1) + np.roll(x, -1, axis=1)
        + np.roll(x, 1, axis=2) + np.roll(x, -1, axis=2)
    )
    return (
        2.0 * (1.0 - 2.0 * action.lambda_) * x
        + 4.0 * action.lambda_ * x**3
        - 2.0 * action.kappa * neighbours
    )


def hmc_trajectory(phi: np.ndarray, active: np.ndarray, *, action: ActionSpec, step_size: float, n_steps: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """HMC trajectory for one fixed, uniformly spaced sublattice.

    Inactive phi coordinates and their momenta remain exactly zero/fixed.  This
    is ordinary HMC for the conditional target with all other sites held fixed.
    """
    old = np.asarray(phi, dtype=np.float64)
    mask = np.asarray(active, dtype=bool)[None, :, :]
    p0 = rng.standard_normal(old.shape) * mask
    q = old.copy()
    p = p0.copy()
    p -= 0.5 * step_size * force(q, action) * mask
    for step in range(n_steps):
        q += step_size * p
        q = np.where(mask, q, old)
        if step + 1 != n_steps:
            p -= step_size * force(q, action) * mask
    p -= 0.5 * step_size * force(q, action) * mask
    p = -p  # conventional reversible end-of-trajectory momentum flip
    old_h = action_total(old, action) + 0.5 * np.sum(p0 * p0, axis=(1, 2))
    new_h = action_total(q, action) + 0.5 * np.sum(p * p, axis=(1, 2))
    delta_h = new_h - old_h
    accepted = np.log(rng.random(len(old))) < np.minimum(0.0, -delta_h)
    return np.where(accepted[:, None, None], q, old).astype(np.float32), accepted, delta_h


def write_measurements(run: Path, main: list[dict], grows: list[dict]) -> None:
    write_csv(run / "observables" / "main_per_sweep_measurements.csv", main, MAIN_COLUMNS)
    write_csv(run / "observables" / "per_sweep_observables.csv", main, MAIN_COLUMNS)
    write_csv(run / "observables" / "Gk_per_sweep_measurements.csv", grows, G_COLUMNS)
    write_csv(run / "observables" / "ensemble_average_history.csv", aggregate(main))


def measure_phi(
    phi: np.ndarray, sweep: int, source_indices: np.ndarray, *, batch_size: int | None = None,
) -> tuple[list[dict], list[dict]]:
    """Measurement rows with the same schema as the coordinate-MH outputs."""
    arr = np.asarray(phi, dtype=np.float32)
    L = int(arr.shape[1]); main, grows = [], []
    chunk = batch_size or len(arr)
    for begin in range(0, len(arr), chunk):
        end = min(begin + chunk, len(arr))
        obs, g = per_config_observables(
            arr[begin:end], ActionSpec("phi4_nn", 1.0, 0.340301),
        )
        for local_chain, chain in enumerate(range(begin, end)):
            common = {"chain_id": chain, "sweep": sweep, "source_config_index": int(source_indices[chain]), "source_native_L32_index": int(source_indices[chain]), "L": L, "volume": L * L}
            main.append(common | {"action_density": float(obs["action_density"][local_chain]), "total_action": float(obs["total_action"][local_chain]), "phi2": float(obs["phi2"][local_chain]), "phi4": float(obs["phi4"][local_chain]), "NN": float(obs["NN"][local_chain]), "diag": float(obs["diag"][local_chain]), "2nn": float(obs["2nn"][local_chain]), "m": float(obs["m"][local_chain]), "m2": float(obs["m2"][local_chain]), "m4": float(obs["m4"][local_chain]), "G_pmin_x_cfg": float(g["G_10"][local_chain]), "G_pmin_y_cfg": float(g["G_01"][local_chain]), "nonfinite_count": int(np.sum(~np.isfinite(arr[chain])))} )
            grows.append(common | {"G_00": float(g["G_00"][local_chain]), "G_10": float(g["G_10"][local_chain]), "G_01": float(g["G_01"][local_chain]), "G_pmin_avg": float(g["G_pmin_avg"][local_chain])})
    return main, grows


def main() -> int:
    p = argparse.ArgumentParser(description="Direct fine-field HMC after one optional L16->L32 flow initialization.")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--native-source", type=Path, help="Direct fine reference, required unless --initialization input is used.")
    p.add_argument("--coarse-source", type=Path)
    p.add_argument("--flow-checkpoint", type=Path)
    p.add_argument("--kernel-path", type=Path)
    p.add_argument("--initialization", choices=("direct_coarse_flow", "native", "input"), default="direct_coarse_flow")
    p.add_argument("--input-source", type=Path, help="Existing fine phi checkpoint used when --initialization input.")
    p.add_argument("--n-chains", type=int, default=500)
    p.add_argument("--n-sweeps", type=int, default=400, help="One sweep evolves every residue class once.")
    p.add_argument("--save-every", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=50)
    p.add_argument(
        "--hmc-batch-size", type=int,
        help="Chains evolved together within each HMC subtrajectory; defaults to --batch-size.",
    )
    p.add_argument(
        "--measurement-batch-size", type=int,
        help="Chains processed together for observables; defaults to --hmc-batch-size.",
    )
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--step-size", type=float, default=0.08)
    p.add_argument("--leapfrog-steps", type=int, default=10)
    p.add_argument("--divide", type=int, default=2, help="Uniform residue grid: divide=2 evolves 16^2 sites per L32 subtrajectory.")
    p.add_argument("--seed", type=int, default=2026081301)
    p.add_argument("--sweep-offset", type=int, default=0, help="Label sweep zero as this absolute sweep number (useful for continuations).")
    p.add_argument("--level-name", default="L16toL32", help="Name exposed under run-dir/levels for notebook compatibility.")
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    if a.n_chains <= 0 or a.n_sweeps <= 0 or a.save_every <= 0 or a.step_size <= 0 or a.leapfrog_steps <= 0 or a.divide <= 0:
        raise ValueError("chain count, sweeps, save interval, step size, leapfrog steps, and divide must be positive")
    if a.smoke:
        a.n_chains, a.n_sweeps, a.save_every = min(a.n_chains, 4), min(a.n_sweeps, 3), 1
    if a.hmc_batch_size is None:
        a.hmc_batch_size = a.batch_size
    if a.hmc_batch_size <= 0:
        raise ValueError("--hmc-batch-size must be positive")
    if a.measurement_batch_size is None:
        a.measurement_batch_size = a.hmc_batch_size
    if a.measurement_batch_size <= 0:
        raise ValueError("--measurement-batch-size must be positive")
    if a.initialization == "direct_coarse_flow" and any(x is None for x in (a.coarse_source, a.flow_checkpoint, a.kernel_path)):
        raise ValueError("direct_coarse_flow requires --coarse-source, --flow-checkpoint, and --kernel-path")
    if a.initialization == "input" and a.input_source is None:
        raise ValueError("input initialization requires --input-source")
    if a.initialization != "input" and a.native_source is None:
        raise ValueError("native/direct_coarse_flow initialization requires --native-source")

    run = a.run_dir.resolve()
    for name in ("observables", "checkpoints", "logs", "summaries"):
        (run / name).mkdir(parents=True, exist_ok=True)
    # Present a standalone run through the same hierarchy as a multi-level
    # chain: CHAIN_DIR / "levels" / "L16toL32".  The top-level layout remains
    # available for backward compatibility.
    level_name = str(a.level_name)
    levels = run / "levels"
    levels.mkdir(exist_ok=True)
    level_link = levels / level_name
    if not level_link.exists() and not level_link.is_symlink():
        level_link.symlink_to("..", target_is_directory=True)
    if a.initialization == "input":
        input_all = load_phi(a.input_source)
        if a.start_index + a.n_chains > len(input_all):
            raise ValueError("not enough input fine configurations for requested start index/count")
        native = input_all[a.start_index:a.start_index + a.n_chains].astype(np.float32)
    else:
        native_all = load_phi(a.native_source)
        if a.start_index + a.n_chains > len(native_all):
            raise ValueError("not enough native fine configurations for requested start index/count")
        native = native_all[a.start_index:a.start_index + a.n_chains].astype(np.float32)
    source_indices = np.arange(a.start_index, a.start_index + a.n_chains, dtype=np.int64)
    action = ActionSpec("phi4_nn", 1.0, 0.340301)
    L = int(native.shape[1])
    if L % a.divide:
        raise ValueError(f"fine L={L} must be divisible by --divide={a.divide}")
    yy, xx = np.indices((L, L))
    sublattices = [((oy, ox), (yy % a.divide == oy) & (xx % a.divide == ox)) for oy in range(a.divide) for ox in range(a.divide)]
    rng = np.random.default_rng(a.seed)

    config = vars(a).copy() | {
        "lambda": 1.0,
        "kappa": 0.340301,
        # Compatibility aliases for the established coordinate-MH notebooks.
        "kappa_f": 0.340301,
        "kappa_c": 0.340301,
        "native_reference_source": str(a.native_source) if a.native_source else None,
        "target": "exp(-S_f(phi))",
        "integrator": "second-order leapfrog",
        "hamiltonian": "S_f(phi)+1/2 sum_active_x p_x^2",
        "sweep_semantics": f"one sweep is {a.divide**2} HMC subtrajectories, one for each residue (y mod {a.divide}, x mod {a.divide}); each has {L * L // a.divide**2} uniformly spaced active sites, so every fine site is touched exactly once",
    }
    if a.initialization in ("native", "input"):
        phi = native.copy()
        config["flow_used_after_sweep_zero"] = False
        config["flow_used_at_sweep_zero"] = False
        if a.initialization == "input":
            config["input_source"] = str(a.input_source)
    else:
        coarse_all = load_phi(a.coarse_source)
        if a.start_index + a.n_chains > len(coarse_all):
            raise ValueError("not enough direct coarse configurations")
        kernel, kernel_meta = load_kernel_matrix(a.kernel_path)
        if not bool(kernel_meta.get("kernel_coefficients_include_eta_scale", False)):
            raise ValueError("kernel must include eta scale")
        ck = torch.load(a.flow_checkpoint, map_location="cpu", weights_only=False)
        coarse_L = L // 2
        model, report = build_model_from_checkpoint(ck, coarse_L, torch.device("cpu"))
        stats = stationary_stats(ck["state"]["stats"], coarse_L)
        coarse = coarse_all[a.start_index:a.start_index + a.n_chains].astype(np.float32)
        detail, _, _, _ = sample_model_lattice(model, coarse, stats, batch_size=a.batch_size, device=torch.device("cpu"), seed=int(rng.integers(2**31 - 1)))
        phi, _ = inverse_kernel(assemble_psi(coarse, detail), kernel)
        phi = np.asarray(phi, dtype=np.float32)
        config |= {"kernel_metadata": kernel_meta, "flow_load_report": report, "flow_used_at_sweep_zero": True, "flow_used_after_sweep_zero": False}
    write_json(run / "run_config.json", config)
    (run / "run_config.yaml").write_text(json.dumps(config, indent=2, default=str) + "\n")

    sweep0 = int(a.sweep_offset)
    main_rows, g_rows = measure_phi(phi, sweep0, source_indices, batch_size=a.measurement_batch_size)
    native_rows, _ = measure_phi(native, 0, source_indices, batch_size=a.measurement_batch_size)
    write_csv(run / "observables" / f"native_L{L}_reference_summary.csv", aggregate(native_rows))
    write_measurements(run, main_rows, g_rows)
    np.savez_compressed(run / "checkpoints" / f"checkpoint_sweep_{sweep0:04d}.npz", phi=phi)
    acc_rows: list[dict] = []
    accepted_total = 0
    t0 = time.perf_counter()
    for sweep in range(1, a.n_sweeps + 1):
        accepted_this, delta_h_this = 0, []
        for _residue, active in sublattices:
            for begin in range(0, a.n_chains, a.hmc_batch_size):
                end = min(begin + a.hmc_batch_size, a.n_chains)
                updated, accepted, delta_h = hmc_trajectory(
                    phi[begin:end], active, action=action,
                    step_size=a.step_size, n_steps=a.leapfrog_steps, rng=rng,
                )
                phi[begin:end] = updated
                accepted_this += int(accepted.sum())
                delta_h_this.append(delta_h)
        accepted_total += accepted_this
        all_delta_h = np.concatenate(delta_h_this)
        attempted_total = sweep * len(sublattices) * a.n_chains
        acc_rows.append({"sweep": sweep, "subtrajectories": len(sublattices), "active_sites_per_subtrajectory": int(sublattices[0][1].sum()), "accepted": accepted_this, "attempted": len(sublattices) * a.n_chains, "acceptance": accepted_this / (len(sublattices) * a.n_chains), "acceptance_cumulative": accepted_total / attempted_total, "mean_delta_H": float(all_delta_h.mean()), "std_delta_H": float(all_delta_h.std(ddof=1))})
        absolute_sweep = sweep0 + sweep
        if sweep % a.save_every == 0 or sweep == a.n_sweeps:
            rows, grows = measure_phi(phi, absolute_sweep, source_indices, batch_size=a.measurement_batch_size)
            main_rows.extend(rows); g_rows.extend(grows)
            write_measurements(run, main_rows, g_rows)
            np.savez_compressed(run / "checkpoints" / f"checkpoint_sweep_{absolute_sweep:04d}.npz", phi=phi)
        np.savez_compressed(run / "checkpoints" / "checkpoint_latest.npz", phi=phi)
        write_csv(run / "observables" / "acceptance_history.csv", acc_rows, ACCEPT_COLUMNS)
        print(json.dumps({"sweep": absolute_sweep, "subtrajectories": len(sublattices), "acceptance": accepted_this / (len(sublattices) * a.n_chains), "acceptance_cumulative": accepted_total / attempted_total}), flush=True)
    summary = {"status": "completed", "runtime_sec": time.perf_counter() - t0, "acceptance": accepted_total / (a.n_sweeps * len(sublattices) * a.n_chains), "fine_action_only_after_initialization": True}
    write_json(run / "summaries" / "run_summary.json", summary)
    write_json(run / "status.json", summary | {"current_sweep": a.n_sweeps})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
