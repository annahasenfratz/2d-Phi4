#!/usr/bin/env python3
"""Restartable embedded-Wolff + radial-heat-bath rethermalization."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "perfect_blocking_upsampling" / "src"), str(ROOT / "perfect_blocking_upsampling" / "scripts")]

from perfect_blocking_upsampling.blocking import load_phi
from perfect_blocking_upsampling.io import ActionSpec
from run_lam1p0_fine_hmc import G_COLUMNS, MAIN_COLUMNS, measure_phi, write_measurements
from run_lam1p0_mit_style_inverse_blocking_L8to16 import write_csv


def load_wolff_module():
    path = ROOT / "phi4_phase-diagram" / "src" / "generate_phi4_embedded_wolff_radial_heatbath.py"
    spec = importlib.util.spec_from_file_location("phi4_embedded_wolff", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Wolff implementation: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


WOLFF = load_wolff_module()


def write_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    tmp.replace(path)


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path,
                    help="Fine-field source checkpoint. Required unless --random-initialization is used.")
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--L", type=int, required=True)
    ap.add_argument("--start-index", type=int, required=True)
    ap.add_argument("--n-chains", type=int, required=True)
    ap.add_argument("--target-sweeps", type=int, default=100,
                    help="Total sweep count; use a larger value with --resume to continue.")
    ap.add_argument("--checkpoint-every", type=int, default=25)
    ap.add_argument("--measurement-batch-size", type=int, default=1)
    ap.add_argument("--seed", type=int, default=2026082003)
    ap.add_argument("--clusters-per-sweep", type=int, default=1,
                    help="Fixed number of Wolff clusters after each radial heat-bath sweep.")
    ap.add_argument("--kappa", type=float, default=0.340301,
                    help="Target hopping parameter for the radial and Wolff updates.")
    ap.add_argument("--random-initialization", action="store_true",
                    help="Initialize independent N(0, --random-std^2) fine fields instead of loading --source.")
    ap.add_argument("--random-std", type=float, default=0.7,
                    help="Standard deviation for --random-initialization; matches the canonical generator default.")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    if args.n_chains <= 0 or args.L <= 0 or args.target_sweeps < 0 or args.clusters_per_sweep <= 0 or args.random_std <= 0.0:
        raise ValueError("L, n-chains and clusters-per-sweep must be positive; target-sweeps non-negative")
    if not args.resume and not args.random_initialization and args.source is None:
        raise ValueError("--source is required unless --random-initialization is used")
    if args.random_initialization and args.source is not None:
        raise ValueError("choose exactly one initialization: --source or --random-initialization")
    run = args.run_dir.resolve()
    state_path = run / "checkpoints" / "state_current.npy"
    status_path = run / "status.json"
    run.mkdir(parents=True, exist_ok=True)
    for name in ("observables", "checkpoints", "summaries", "logs"):
        (run / name).mkdir(exist_ok=True)

    config = {
        "algorithm": "embedded Wolff sign cluster + exact radial heat bath",
        "lambda": 1.0, "kappa": float(args.kappa), "L": args.L,
        "source": str(args.source.resolve()) if args.source is not None else None,
        "initialization": "random_normal" if args.random_initialization else "source_checkpoint",
        "random_std": args.random_std if args.random_initialization else None,
        "start_index": args.start_index,
        "n_chains": args.n_chains, "seed": args.seed,
        "clusters_per_sweep": args.clusters_per_sweep,
        "checkpoint_every": args.checkpoint_every,
        "restart_instruction": "rerun with --resume --target-sweeps NEW_TOTAL",
    }
    config_text = json.dumps(config, indent=2) + "\n"

    if args.resume:
        if not state_path.exists() or not status_path.exists():
            raise FileNotFoundError("--resume requires checkpoints/state_current.npy and status.json")
        old = json.loads(status_path.read_text())
        if int(old["completed_sweeps"]) > args.target_sweeps:
            raise ValueError("target-sweeps precedes current state")
        if int(old["n_chains"]) != args.n_chains or int(old["start_index"]) != args.start_index:
            raise ValueError("resume n-chains/start-index does not match the saved state")
        state = np.load(state_path, mmap_mode="r+")
        completed = int(old["completed_sweeps"])
        rng = np.random.default_rng()
        rng.bit_generator.state = old["rng_state"]
        main_rows = read_rows(run / "observables" / "main_per_sweep_measurements.csv")
        g_rows = read_rows(run / "observables" / "Gk_per_sweep_measurements.csv")
        cluster_rows = read_rows(run / "observables" / "cluster_history.csv")
    else:
        if state_path.exists() or status_path.exists():
            raise FileExistsError(f"run already initialized: {run}; use --resume")
        rng = np.random.default_rng(args.seed + args.start_index)
        if args.random_initialization:
            fields = rng.normal(0.0, args.random_std, size=(args.n_chains, args.L, args.L)).astype(np.float32)
        else:
            source = load_phi(args.source)
            fields = source[args.start_index:args.start_index + args.n_chains]
            if fields.shape != (args.n_chains, args.L, args.L):
                raise ValueError(f"source slice has shape {fields.shape}, expected {(args.n_chains, args.L, args.L)}")
        state = np.lib.format.open_memmap(state_path, mode="w+", dtype=np.float32, shape=fields.shape)
        state[:] = fields
        state.flush()
        if not args.random_initialization:
            del source
        del fields
        completed = 0
        main_rows, g_rows, cluster_rows = [], [], []
        (run / "run_config.yaml").write_text(config_text)
        (run / "config.yaml").write_text(config_text)
        write_json(run / "run_config.json", config)
        source_indices = np.arange(args.start_index, args.start_index + args.n_chains, dtype=np.int64)
        action = ActionSpec("phi4_nn", 1.0, float(args.kappa))
        rows, grows = measure_phi(state, 0, source_indices, action=action, batch_size=args.measurement_batch_size)
        main_rows.extend(rows); g_rows.extend(grows); write_measurements(run, main_rows, g_rows)
        np.savez_compressed(run / "checkpoints" / "checkpoint_sweep_0000.npz", phi=state, source_indices=source_indices)
        # Persist a resumable sweep-zero state before the first potentially
        # long radial/cluster update begins.
        write_json(status_path, {
            "status": "completed" if args.target_sweeps == 0 else "running",
            "completed_sweeps": 0, "target_sweeps": args.target_sweeps,
            "n_chains": args.n_chains, "start_index": args.start_index,
            "rng_state": rng.bit_generator.state,
            "state_file": str(state_path), "last_checkpoint_sweep": 0,
        })

    action = ActionSpec("phi4_nn", 1.0, float(args.kappa))
    params = WOLFF.Params(lam=1.0, kappa=float(args.kappa), L=args.L, clusters_per_sweep=1)
    h_grid, r_grid, cdfs = WOLFF.build_heatbath_table(params)
    source_indices = np.arange(args.start_index, args.start_index + args.n_chains, dtype=np.int64)
    for sweep in range(completed + 1, args.target_sweeps + 1):
        all_sizes: list[int] = []
        per_chain_cluster_counts = np.empty(args.n_chains, dtype=np.int32)
        volume = args.L * args.L
        for chain in range(args.n_chains):
            WOLFF.radial_heatbath_sweep(state[chain], params, h_grid, r_grid, cdfs, rng)
            for _ in range(args.clusters_per_sweep):
                size = WOLFF.wolff_sign_cluster(state[chain], params, rng)
                all_sizes.append(size)
            per_chain_cluster_counts[chain] = args.clusters_per_sweep
        state.flush()
        sizes = np.asarray(all_sizes, dtype=np.int32)
        fractions = sizes.astype(np.float64) / float(volume)
        cluster_rows.append({
            "sweep": sweep, "clusters": int(sizes.size),
            "mean_cluster_fraction": float(fractions.mean()),
            "median_cluster_fraction": float(np.median(fractions)),
            "p90_cluster_fraction": float(np.quantile(fractions, 0.9)),
            "max_cluster_fraction": float(fractions.max()),
            "mean_clusters_per_configuration": float(per_chain_cluster_counts.mean()),
            "sum_cluster_fraction_per_configuration": float(sizes.sum() / (args.n_chains * volume)),
        })
        rows, grows = measure_phi(state, sweep, source_indices, action=action, batch_size=args.measurement_batch_size)
        main_rows.extend(rows); g_rows.extend(grows); write_measurements(run, main_rows, g_rows)
        write_csv(run / "observables" / "cluster_history.csv", cluster_rows)
        if sweep % args.checkpoint_every == 0 or sweep == args.target_sweeps:
            np.savez_compressed(run / "checkpoints" / f"checkpoint_sweep_{sweep:04d}.npz", phi=state, source_indices=source_indices)
        write_json(status_path, {
            "status": "running" if sweep < args.target_sweeps else "completed",
            "completed_sweeps": sweep, "target_sweeps": args.target_sweeps,
            "n_chains": args.n_chains, "start_index": args.start_index,
            "rng_state": rng.bit_generator.state,
            "state_file": str(state_path), "last_checkpoint_sweep": sweep if sweep % args.checkpoint_every == 0 or sweep == args.target_sweeps else None,
        })
        print(json.dumps({"sweep": sweep, "mean_cluster_fraction": float(fractions.mean()),
                          "median_cluster_fraction": float(np.median(fractions))}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
