#!/usr/bin/env python3
"""Long, streamed direct L32 local-Metropolis control run for lambda=1.

The proposal is a simultaneous checkerboard single-site random walk.  Sites in
one parity class have no nearest-neighbour bonds, so their local Metropolis
tests can be evaluated and applied together without changing the transition
kernel.  This vectorisation is essential for million-sweep diagnostic runs.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PKG = PROJECT_ROOT / "perfect_blocking_upsampling"
sys.path.insert(0, str(PKG / "src"))

from perfect_blocking_upsampling.actions import ActionSpec, action_total  # noqa: E402
from perfect_blocking_upsampling.observables import second_moment_components  # noqa: E402


DEFAULT_INPUT = PROJECT_ROOT / "data/configs_phi4_2d/lam1p0_kappac0p340301_L32/configs.npz"
MEASUREMENT_FIELDS = [
    "chain_id", "sweep", "source_config_index", "L", "volume",
    "action_density", "total_action", "phi2", "phi4", "NN", "diag", "2nn",
    "m", "m2", "m4", "G_pmin_x_cfg", "G_pmin_y_cfg", "nonfinite_count",
]
ACCEPTANCE_FIELDS = [
    "sweep", "recent_sweeps", "recent_acceptance", "cumulative_acceptance",
    "recent_proposals", "recent_accepted", "cumulative_proposals", "cumulative_accepted",
]


class StreamingCsv:
    def __init__(self, path: Path, fields: list[str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("w", newline="", encoding="utf-8", buffering=1)
        self._writer = csv.DictWriter(self._file, fieldnames=fields)
        self._writer.writeheader()
        self._file.flush()

    def write_many(self, rows: list[dict[str, Any]]) -> None:
        self._writer.writerows(rows)
        self._file.flush()

    def write(self, row: dict[str, Any]) -> None:
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_phi(path: Path, lattice_size: int) -> np.ndarray:
    with np.load(path) as loaded:
        key = "phi" if "phi" in loaded.files else loaded.files[0]
        phi = loaded[key].astype(np.float32)
    if phi.ndim != 3 or phi.shape[1:] != (lattice_size, lattice_size):
        raise ValueError(f"expected (N,{lattice_size},{lattice_size}) in {path}, got {phi.shape}")
    return phi


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def measurement_rows(phi: np.ndarray, action: ActionSpec, source_indices: np.ndarray, sweep: int) -> list[dict[str, Any]]:
    values = phi.astype(np.float64)
    lattice_size = int(phi.shape[1])
    volume = lattice_size * lattice_size
    total = action_total(phi, action).astype(np.float64)
    phi2 = np.mean(values**2, axis=(1, 2))
    phi4 = np.mean(values**4, axis=(1, 2))
    nn = np.mean(values * np.roll(values, -1, 1) + values * np.roll(values, -1, 2), axis=(1, 2))
    diag = np.mean(values * np.roll(np.roll(values, -1, 1), -1, 2), axis=(1, 2))
    two_nn = np.mean(values * np.roll(values, -2, 1) + values * np.roll(values, -2, 2), axis=(1, 2))
    magnetization = np.mean(values, axis=(1, 2))
    second_moment = second_moment_components(phi)
    return [
        {
            "chain_id": i, "sweep": sweep, "source_config_index": int(source_indices[i]),
            "L": lattice_size, "volume": volume,
            "action_density": float(total[i] / volume), "total_action": float(total[i]),
            "phi2": float(phi2[i]), "phi4": float(phi4[i]), "NN": float(nn[i]),
            "diag": float(diag[i]), "2nn": float(two_nn[i]),
            "m": float(magnetization[i]), "m2": float(magnetization[i] ** 2), "m4": float(magnetization[i] ** 4),
            "G_pmin_x_cfg": float(second_moment["G_pmin_x_cfg"][i]),
            "G_pmin_y_cfg": float(second_moment["G_pmin_y_cfg"][i]),
            "nonfinite_count": int(np.count_nonzero(~np.isfinite(phi[i]))),
        }
        for i in range(len(phi))
    ]


def checkerboard_sweep(phi: np.ndarray, action: ActionSpec, step_size: float, rng: np.random.Generator,
                       parity_sites: list[tuple[np.ndarray, np.ndarray]]) -> tuple[int, int]:
    """One exact full sweep: one proposed local displacement per lattice site."""
    attempts = accepts = 0
    lam, kappa = float(action.lambda_), float(action.kappa)
    onsite_quadratic = 1.0 - 2.0 * lam
    for xs, ys in parity_sites:
        old = phi[:, xs, ys].astype(np.float64)
        delta = rng.uniform(-step_size, step_size, size=old.shape)
        new = old + delta
        neighbour_sum = (
            phi[:, (xs + 1) % phi.shape[1], ys] + phi[:, (xs - 1) % phi.shape[1], ys]
            + phi[:, xs, (ys + 1) % phi.shape[2]] + phi[:, xs, (ys - 1) % phi.shape[2]]
        ).astype(np.float64)
        delta_s = (
            onsite_quadratic * (new**2 - old**2) + lam * (new**4 - old**4)
            - 2.0 * kappa * delta * neighbour_sum
        )
        accept = np.log(rng.random(old.shape)) < np.minimum(0.0, -delta_s)
        phi[:, xs, ys] = np.where(accept, new, old).astype(np.float32)
        attempts += int(accept.size)
        accepts += int(np.count_nonzero(accept))
    return attempts, accepts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-configs", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-chains", type=int, default=1)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--sweeps", type=int, default=1_000_000)
    parser.add_argument("--measure-every", type=int, default=20)
    parser.add_argument("--step-size", type=float, default=0.94,
                        help="uniform proposal half-width; pilot-calibrated to about 60%% acceptance")
    parser.add_argument("--seed", type=int, default=2026080601)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.n_chains <= 0 or args.sweeps <= 0 or args.measure_every <= 0 or args.step_size <= 0:
        raise SystemExit("n-chains, sweeps, measure-every, and step-size must be positive")
    output_dir = args.output_dir
    # The submitter creates ``logs/`` first so it can redirect stdout there.
    # Reject any real previous run, but accept that minimal pre-launch layout.
    permitted_prelaunch = {"logs", "submit_manifest.txt", "submit_pid.txt"}
    if output_dir.exists() and any(entry.name not in permitted_prelaunch for entry in output_dir.iterdir()):
        raise SystemExit(f"output directory already contains run output: {output_dir}")
    (output_dir / "observables").mkdir(parents=True)
    all_phi = read_phi(args.input_configs, 32)
    stop = args.start_index + args.n_chains
    if stop > len(all_phi):
        raise SystemExit(f"requested native indices [{args.start_index}, {stop}), but source contains {len(all_phi)}")
    source_indices = np.arange(args.start_index, stop, dtype=np.int64)
    phi = all_phi[source_indices].copy()
    action = ActionSpec("phi4_nn", 1.0, 0.340301)
    rng = np.random.default_rng(args.seed)
    grid_x, grid_y = np.indices((32, 32))
    parity_sites = [
        (grid_x[(grid_x + grid_y) % 2 == parity], grid_y[(grid_x + grid_y) % 2 == parity])
        for parity in (0, 1)
    ]
    started = time.time()
    manifest: dict[str, Any] = {
        "status": "running", "command": " ".join(sys.argv), "git_commit": git_commit(),
        "hostname": socket.gethostname(), "platform": platform.platform(), "python": sys.version,
        "started_unix": started, "input_configs": str(args.input_configs.resolve()),
        "L": 32, "lambda": 1.0, "kappa": 0.340301, "n_chains": args.n_chains,
        "source_config_indices": source_indices.tolist(), "sweeps": args.sweeps,
        "measure_every": args.measure_every, "step_size": args.step_size, "seed": args.seed,
        "algorithm": "exact simultaneous checkerboard single-site uniform random-walk Metropolis",
        "sweep_definition": "one proposal at every L32 lattice site (even parity then odd parity)",
        "storage": "measurements are flushed every measure-every sweeps; only final_config.npz is stored",
    }
    write_json(output_dir / "run_manifest.json", manifest)
    measurements = StreamingCsv(output_dir / "observables" / "main_per_sweep_measurements.csv", MEASUREMENT_FIELDS)
    acceptance = StreamingCsv(output_dir / "observables" / "acceptance_history.csv", ACCEPTANCE_FIELDS)
    measurements.write_many(measurement_rows(phi, action, source_indices, 0))
    acceptance.write({"sweep": 0, "recent_sweeps": 0, "recent_acceptance": float("nan"), "cumulative_acceptance": float("nan"), "recent_proposals": 0, "recent_accepted": 0, "cumulative_proposals": 0, "cumulative_accepted": 0})
    cumulative_attempts = cumulative_accepts = recent_attempts = recent_accepts = 0
    wall_start = time.perf_counter()
    for sweep in range(1, args.sweeps + 1):
        attempted, accepted = checkerboard_sweep(phi, action, args.step_size, rng, parity_sites)
        cumulative_attempts += attempted
        cumulative_accepts += accepted
        recent_attempts += attempted
        recent_accepts += accepted
        if sweep % args.measure_every == 0 or sweep == args.sweeps:
            measurements.write_many(measurement_rows(phi, action, source_indices, sweep))
            acceptance.write({
                "sweep": sweep, "recent_sweeps": min(args.measure_every, sweep),
                "recent_acceptance": recent_accepts / recent_attempts,
                "cumulative_acceptance": cumulative_accepts / cumulative_attempts,
                "recent_proposals": recent_attempts, "recent_accepted": recent_accepts,
                "cumulative_proposals": cumulative_attempts, "cumulative_accepted": cumulative_accepts,
            })
            status = {
                "status": "running", "sweep": sweep, "sweeps": args.sweeps,
                "recent_acceptance": recent_accepts / recent_attempts,
                "cumulative_acceptance": cumulative_accepts / cumulative_attempts,
                "elapsed_sec": time.perf_counter() - wall_start,
            }
            write_json(output_dir / "status.json", status)
            print(json.dumps(status), flush=True)
            recent_attempts = recent_accepts = 0
    np.savez_compressed(output_dir / "final_config.npz", phi=phi, source_config_index=source_indices, final_sweep=args.sweeps)
    measurements.close()
    acceptance.close()
    manifest["status"] = "completed"
    manifest["completed_unix"] = time.time()
    manifest["runtime_sec"] = time.perf_counter() - wall_start
    manifest["final_config"] = "final_config.npz"
    write_json(output_dir / "run_manifest.json", manifest)
    write_json(output_dir / "summary.json", manifest)
    write_json(output_dir / "status.json", {"status": "completed", "sweep": args.sweeps, "runtime_sec": manifest["runtime_sec"]})
    print(json.dumps({"status": "completed", "output_dir": str(output_dir), "runtime_sec": manifest["runtime_sec"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
