#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PKG = PROJECT_ROOT / "perfect_blocking_upsampling"
SCRIPT_DIR = PKG / "scripts"
sys.path.insert(0, str(PKG / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

from perfect_blocking_upsampling.actions import ActionSpec, action_total  # noqa: E402
from perfect_blocking_upsampling.observables import second_moment_components  # noqa: E402
from run_lam0p2_rand5x5_0084_detail_only_correction_diagnostic import write_csv, write_json  # noqa: E402

DEFAULT_NATIVE = PKG / "outputs" / "lam0p2_kappa0p323124" / "native" / "L32" / "configs.npz"

MAIN_MEASUREMENT_FIELDS = [
    "chain_id",
    "sweep",
    "source_config_index",
    "source_native_L32_index",
    "L",
    "volume",
    "action_density",
    "total_action",
    "phi2",
    "phi4",
    "NN",
    "diag",
    "2nn",
    "m",
    "m2",
    "m4",
    "G_pmin_x_cfg",
    "G_pmin_y_cfg",
    "nonfinite_count",
]

SITE_HISTORY_FIELDS = [
    "sweep",
    "pass",
    "update_order",
    "parity",
    "sites_touched",
    "attempts",
    "accepted",
    "acceptance",
    "DeltaS_mean",
    "DeltaS_std",
    "DeltaS_min",
    "DeltaS_max",
    "log_accept_mean",
    "log_accept_std",
    "elapsed_sec",
]


class StreamingCsv:
    def __init__(self, path: Path, fieldnames: list[str], append: bool = False):
        path.parent.mkdir(parents=True, exist_ok=True)
        exists = path.exists() and path.stat().st_size > 0
        self.file = path.open("a" if append else "w", newline="", encoding="utf-8", buffering=1)
        self.writer = csv.DictWriter(self.file, fieldnames=fieldnames, extrasaction="ignore")
        if not append or not exists:
            self.writer.writeheader()
            self.file.flush()

    def write(self, row: dict[str, Any]) -> None:
        self.writer.writerow(row)
        self.file.flush()

    def close(self) -> None:
        self.file.flush()
        self.file.close()


def read_phi(path: Path, expected_l: int) -> np.ndarray:
    with np.load(path) as z:
        key = "phi" if "phi" in z.files else z.files[0]
        phi = z[key].astype(np.float32)
    if phi.ndim != 3 or phi.shape[1:] != (expected_l, expected_l):
        raise ValueError(f"expected {path} to contain (N,{expected_l},{expected_l}) phi, got {phi.shape}")
    return phi


def write_main_measurement_rows(path: Path, rows: list[dict[str, Any]], append: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append and path.exists() else "w"
    with path.open(mode, newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MAIN_MEASUREMENT_FIELDS, extrasaction="ignore")
        if mode == "w":
            writer.writeheader()
        writer.writerows(rows)
        fh.flush()


def main_measurement_rows(phi: np.ndarray, action: ActionSpec, source_idx: np.ndarray, sweep: int) -> list[dict[str, Any]]:
    arr = phi.astype(np.float64)
    L = int(phi.shape[1])
    volume = int(L * L)
    total = action_total(phi.astype(np.float32), action).astype(np.float64)
    phi2 = np.mean(arr**2, axis=(1, 2))
    phi4 = np.mean(arr**4, axis=(1, 2))
    nn = np.mean(arr * np.roll(arr, -1, axis=1) + arr * np.roll(arr, -1, axis=2), axis=(1, 2))
    diag = np.mean(arr * np.roll(np.roll(arr, -1, axis=1), -1, axis=2), axis=(1, 2))
    two_nn = np.mean(arr * np.roll(arr, -2, axis=1) + arr * np.roll(arr, -2, axis=2), axis=(1, 2))
    m = np.mean(arr, axis=(1, 2))
    sm = second_moment_components(phi.astype(np.float32))
    rows: list[dict[str, Any]] = []
    for i in range(len(phi)):
        rows.append(
            {
                "chain_id": int(i),
                "sweep": int(sweep),
                "source_config_index": int(source_idx[i]),
                "source_native_L32_index": int(source_idx[i]),
                "L": L,
                "volume": volume,
                "action_density": float(total[i] / volume),
                "total_action": float(total[i]),
                "phi2": float(phi2[i]),
                "phi4": float(phi4[i]),
                "NN": float(nn[i]),
                "diag": float(diag[i]),
                "2nn": float(two_nn[i]),
                "m": float(m[i]),
                "m2": float(m[i] * m[i]),
                "m4": float(m[i] ** 4),
                "G_pmin_x_cfg": float(sm["G_pmin_x_cfg"][i]),
                "G_pmin_y_cfg": float(sm["G_pmin_y_cfg"][i]),
                "nonfinite_count": int((~np.isfinite(phi[i])).sum()),
            }
        )
    return rows


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)
        except Exception:
            pass


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


def validate_local_delta_s(phi: np.ndarray, action: ActionSpec, rng: np.random.Generator, n_checks: int) -> dict[str, Any]:
    errs: list[float] = []
    rels: list[float] = []
    for _ in range(max(0, int(n_checks))):
        chain = int(rng.integers(0, len(phi)))
        x = int(rng.integers(0, phi.shape[1]))
        y = int(rng.integers(0, phi.shape[2]))
        delta = np.asarray([rng.uniform(-0.5, 0.5)], dtype=np.float64)
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


def metropolis_sweep(
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
    L = int(phi.shape[1])
    block_acc: list[float] = []
    start = time.perf_counter()
    for p in range(int(passes)):
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
            delta_vals: list[float] = []
            log_vals: list[float] = []
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
                delta_vals.extend([float(v) for v in delta_s])
                log_vals.extend([float(v) for v in log_accept])
            acc = float(group_accepts / group_attempts) if group_attempts else float("nan")
            block_acc.append(acc)
            writer.write(
                {
                    "sweep": int(sweep),
                    "pass": int(p),
                    "update_order": update_order,
                    "parity": parity,
                    "sites_touched": int(len(sites)),
                    "attempts": int(group_attempts),
                    "accepted": int(group_accepts),
                    "acceptance": acc,
                    "DeltaS_mean": float(np.mean(delta_vals)) if delta_vals else float("nan"),
                    "DeltaS_std": float(np.std(delta_vals, ddof=1)) if len(delta_vals) > 1 else 0.0,
                    "DeltaS_min": float(np.min(delta_vals)) if delta_vals else float("nan"),
                    "DeltaS_max": float(np.max(delta_vals)) if delta_vals else float("nan"),
                    "log_accept_mean": float(np.mean(log_vals)) if log_vals else float("nan"),
                    "log_accept_std": float(np.std(log_vals, ddof=1)) if len(log_vals) > 1 else 0.0,
                    "elapsed_sec": float(time.perf_counter() - start),
                }
            )
    return phi.astype(np.float32), {
        "acceptance": float(accepts / attempts) if attempts else float("nan"),
        "proposals": int(attempts),
        "accepts": int(accepts),
        "block_acceptance_mean": float(np.mean(block_acc)) if block_acc else float("nan"),
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-configs", type=Path, default=DEFAULT_NATIVE)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--L", dest="lattice_size", type=int, default=32)
    ap.add_argument("--lambda", dest="lam", type=float, default=0.2)
    ap.add_argument("--kappa", type=float, default=0.323124)
    ap.add_argument("--n-configs", type=int, default=1000)
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--sweeps", type=int, default=1500)
    ap.add_argument("--measure-every", type=int, default=50)
    ap.add_argument("--checkpoint-every", type=int, default=0, help="0 means final checkpoint only")
    ap.add_argument("--write-final-configs", action="store_true", help="Also write output-dir/configs.npz at the final sweep.")
    ap.add_argument("--mcmc-step-size", type=float, default=0.5)
    ap.add_argument("--passes", type=int, default=1)
    ap.add_argument("--update-order", choices=["checkerboard", "sequential"], default="checkerboard")
    ap.add_argument("--seed", type=int, default=2026071015)
    ap.add_argument("--deltaS-validation-proposals", type=int, default=64)
    return ap.parse_args()


def main() -> int:
    configure_stdio()
    args = parse_args()
    t0 = time.perf_counter()
    out = args.output_dir
    for sub in ["logs", "observables", "checkpoints", "final_configurations", "manifests"]:
        (out / sub).mkdir(parents=True, exist_ok=True)
    phi_all = read_phi(args.input_configs, int(args.lattice_size))
    stop = int(args.start_index) + int(args.n_configs)
    if stop > len(phi_all):
        raise SystemExit(f"requested [{args.start_index}, {stop}) but only {len(phi_all)} configs in {args.input_configs}")
    source_idx = np.arange(int(args.start_index), stop, dtype=np.int64)
    phi = phi_all[source_idx].astype(np.float32)
    action = ActionSpec("phi4_nn", float(args.lam), float(args.kappa))
    rng = np.random.default_rng(int(args.seed))
    validation = validate_local_delta_s(phi, action, rng, int(args.deltaS_validation_proposals))
    manifest = {
        "command": " ".join(sys.argv),
        "git_commit": git_commit(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "date_unix": time.time(),
        "input_configs": str(args.input_configs.resolve()),
        "output_dir": str(out.resolve()),
        "L": int(args.lattice_size),
        "lambda": float(args.lam),
        "kappa": float(args.kappa),
        "n_configs": int(args.n_configs),
        "start_index": int(args.start_index),
        "source_config_indices": source_idx.tolist(),
        "sweeps": int(args.sweeps),
        "measure_every": int(args.measure_every),
        "checkpoint_every": int(args.checkpoint_every),
        "write_final_configs": bool(args.write_final_configs),
        "mcmc_step_size": float(args.mcmc_step_size),
        "passes": int(args.passes),
        "update_order": args.update_order,
        "acceptance_formula": "single-site uniform random walk; log_accept=min(0,-local_delta_S); target fine action only",
        "local_deltaS_validation": validation,
        "status": "running",
    }
    write_json(out / "run_manifest.json", manifest)
    write_json(out / "manifests" / "run_manifest.json", manifest)

    main_path = out / "observables" / "main_per_sweep_measurements.csv"
    acceptance_path = out / "observables" / "acceptance_history.csv"
    site_history = StreamingCsv(out / "logs" / "single_site_metropolis_history.csv", SITE_HISTORY_FIELDS)
    acceptance_rows: list[dict[str, Any]] = []
    measurement_rows = main_measurement_rows(phi, action, source_idx, 0)
    write_main_measurement_rows(main_path, measurement_rows, append=False)
    acceptance_rows.append({"sweep": 0, "acceptance": float("nan"), "proposals": 0, "accepts": 0})
    write_csv(acceptance_path, acceptance_rows)
    np.savez_compressed(out / "checkpoints" / "state_sweep000000.npz", phi=phi.astype(np.float32), source_config_index=source_idx)

    measure_every = max(1, int(args.measure_every))
    checkpoint_every = int(args.checkpoint_every)
    for sweep in range(1, int(args.sweeps) + 1):
        phi, meta = metropolis_sweep(phi, action, sweep, int(args.passes), float(args.mcmc_step_size), args.update_order, rng, site_history)
        if sweep % measure_every == 0 or sweep == int(args.sweeps):
            rows = main_measurement_rows(phi, action, source_idx, sweep)
            write_main_measurement_rows(main_path, rows, append=True)
            acceptance_rows.append({"sweep": int(sweep), "acceptance": meta["acceptance"], "proposals": meta["proposals"], "accepts": meta["accepts"]})
            write_csv(acceptance_path, acceptance_rows)
            print(json.dumps({"sweep": sweep, "acceptance": meta["acceptance"], "proposals": meta["proposals"]}), flush=True)
        should_checkpoint = (
            sweep == int(args.sweeps)
            if checkpoint_every <= 0
            else (sweep % checkpoint_every == 0 or sweep == int(args.sweeps))
        )
        if should_checkpoint:
            np.savez_compressed(out / "checkpoints" / f"state_sweep{sweep:06d}.npz", phi=phi.astype(np.float32), source_config_index=source_idx)
            np.savez_compressed(out / "final_configurations" / f"configs_sweep{sweep:06d}.npz", phi=phi.astype(np.float32), source_config_index=source_idx)
            if bool(args.write_final_configs) and sweep == int(args.sweeps):
                np.savez_compressed(out / "configs.npz", phi=phi.astype(np.float32), source_config_index=source_idx)
    site_history.close()
    manifest["status"] = "completed"
    manifest["runtime_sec"] = float(time.perf_counter() - t0)
    write_json(out / "run_manifest.json", manifest)
    write_json(out / "summary.json", manifest)
    print(json.dumps({"status": "completed", "output_dir": str(out), "runtime_sec": manifest["runtime_sec"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
