#!/usr/bin/env python3
"""Generate finite-lambda 2D phi4 configs with embedded Wolff + radial heat bath.

Action convention:

    S = sum_x [(1 - 2 lambda) phi_x^2 + lambda phi_x^4]
        - 2 kappa sum_{x,mu} phi_x phi_{x+mu}

The additive constant relative to lambda*(phi^2 - 1)^2 is omitted.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
GENERATOR = "embedded_wolff_sign_cluster_plus_radial_heatbath"
ACTION = "S=sum_x[(1-2*lambda)*phi_x^2 + lambda*phi_x^4] - 2*kappa*sum_x,mu phi_x phi_{x+mu}; constant lambda omitted"


@dataclass
class Params:
    lam: float = 1.0
    kappa: float = 0.30
    L: int = 16
    n_configs: int = 1000
    seed: int = 20263030
    thermal_sweeps: int = 1500
    skip_sweeps: int = 8
    clusters_per_sweep: int = 1
    r_max: float = 5.0
    r_grid_size: int = 1024
    h_abs_max: float = 8.0
    h_grid_size: int = 2001


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")


def atomic_write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    write_json(tmp, obj)
    tmp.replace(path)


def append_generation_row(path: Path, row: dict[str, Any]) -> None:
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        f.flush()


def read_generation_log_tail(path: Path) -> tuple[int, int | None]:
    if not path.exists() or path.stat().st_size == 0:
        return 0, None
    last: dict[str, str] | None = None
    count = 0
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            last = row
            count += 1
    last_sweep = int(float(last["sweep"])) if last and last.get("sweep") else None
    return count, last_sweep


def copy_existing_configs(source: np.ndarray, dest: np.ndarray, chunk_size: int = 256) -> None:
    for start in range(0, len(source), chunk_size):
        stop = min(start + chunk_size, len(source))
        dest[start:stop] = source[start:stop]
        dest.flush()


def build_heatbath_table(params: Params) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h_grid = np.linspace(-params.h_abs_max, params.h_abs_max, params.h_grid_size)
    r_grid = np.linspace(0.0, params.r_max, params.r_grid_size)
    dr = r_grid[1] - r_grid[0]
    cdfs = np.empty((h_grid.size, r_grid.size), dtype=np.float64)
    base = -params.lam * r_grid**4 - (1.0 - 2.0 * params.lam) * r_grid**2
    for i, h in enumerate(h_grid):
        logw = base + h * r_grid
        logw -= np.max(logw)
        w = np.exp(logw)
        # Trapezoid cumulative integral on r >= 0.
        increments = 0.5 * (w[:-1] + w[1:]) * dr
        cdf = np.concatenate([[0.0], np.cumsum(increments)])
        cdf /= cdf[-1]
        cdfs[i] = cdf
    return h_grid, r_grid, cdfs


def sample_radius(h: float, h_grid: np.ndarray, r_grid: np.ndarray, cdfs: np.ndarray, rng: np.random.Generator) -> float:
    idx = int(np.clip(np.searchsorted(h_grid, h), 1, h_grid.size - 1))
    if abs(h_grid[idx] - h) > abs(h_grid[idx - 1] - h):
        idx -= 1
    u = rng.random()
    j = int(np.searchsorted(cdfs[idx], u, side="right"))
    if j <= 0:
        return float(r_grid[0])
    if j >= r_grid.size:
        return float(r_grid[-1])
    c0, c1 = cdfs[idx, j - 1], cdfs[idx, j]
    if c1 <= c0:
        return float(r_grid[j])
    t = (u - c0) / (c1 - c0)
    return float((1.0 - t) * r_grid[j - 1] + t * r_grid[j])


def radial_heatbath_sweep(phi: np.ndarray, params: Params, h_grid: np.ndarray, r_grid: np.ndarray, cdfs: np.ndarray, rng: np.random.Generator) -> None:
    L = params.L
    for parity in (0, 1):
        for x in range(L):
            for y in range(L):
                if (x + y) & 1 != parity:
                    continue
                sign = 1.0 if phi[x, y] >= 0.0 else -1.0
                nn_sum = phi[(x + 1) % L, y] + phi[(x - 1) % L, y] + phi[x, (y + 1) % L] + phi[x, (y - 1) % L]
                h = 2.0 * params.kappa * sign * nn_sum
                r = sample_radius(h, h_grid, r_grid, cdfs, rng)
                phi[x, y] = sign * r


def wolff_sign_cluster(phi: np.ndarray, params: Params, rng: np.random.Generator) -> int:
    L = params.L
    seed_x = int(rng.integers(0, L))
    seed_y = int(rng.integers(0, L))
    seed_sign = 1 if phi[seed_x, seed_y] >= 0.0 else -1
    in_cluster = np.zeros((L, L), dtype=bool)
    in_cluster[seed_x, seed_y] = True
    q: deque[tuple[int, int]] = deque([(seed_x, seed_y)])
    size = 0
    while q:
        x, y = q.popleft()
        size += 1
        for nx, ny in [((x + 1) % L, y), ((x - 1) % L, y), (x, (y + 1) % L), (x, (y - 1) % L)]:
            if in_cluster[nx, ny]:
                continue
            if (1 if phi[nx, ny] >= 0.0 else -1) != seed_sign:
                continue
            p_add = 1.0 - math.exp(-4.0 * params.kappa * abs(float(phi[x, y] * phi[nx, ny])))
            if rng.random() < p_add:
                in_cluster[nx, ny] = True
                q.append((nx, ny))
    phi[in_cluster] *= -1.0
    return size


def sweep(phi: np.ndarray, params: Params, h_grid: np.ndarray, r_grid: np.ndarray, cdfs: np.ndarray, rng: np.random.Generator) -> list[int]:
    radial_heatbath_sweep(phi, params, h_grid, r_grid, cdfs, rng)
    return [wolff_sign_cluster(phi, params, rng) for _ in range(params.clusters_per_sweep)]


def observables(phi: np.ndarray, params: Params) -> dict[str, float]:
    m = float(phi.mean())
    phi2 = float(np.mean(phi**2))
    phi4 = float(np.mean(phi**4))
    nn = 0.5 * float(np.mean(phi * np.roll(phi, -1, axis=0)) + np.mean(phi * np.roll(phi, -1, axis=1)))
    action_density = (1.0 - 2.0 * params.lam) * phi2 + params.lam * phi4 - 4.0 * params.kappa * nn
    return {"m": m, "abs_m": abs(m), "phi2": phi2, "phi4": phi4, "NN": nn, "action_density": action_density}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--L", type=int, default=16)
    p.add_argument("--n-configs", type=int, default=1000)
    p.add_argument("--thermal-sweeps", type=int, default=1500)
    p.add_argument("--skip-sweeps", type=int, default=8)
    p.add_argument("--clusters-per-sweep", type=int, default=1)
    p.add_argument("--seed", type=int, default=20263030)
    p.add_argument("--lambda", dest="lam", type=float, default=1.0)
    p.add_argument("--kappa", type=float, default=0.30)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--append-existing", action="store_true", help="Append --n-configs new configs to an existing output-dir, continuing from the last saved configuration.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    params = Params(
        lam=args.lam,
        kappa=args.kappa,
        L=args.L,
        n_configs=args.n_configs,
        seed=args.seed,
        thermal_sweeps=args.thermal_sweeps,
        skip_sweeps=args.skip_sweeps,
        clusters_per_sweep=args.clusters_per_sweep,
    )
    out = args.output_dir or (
        ROOT
        / "phi4_phase-diagram"
        / "ensembles"
        / f"lam{params.lam:0.3f}_kappa{params.kappa:0.3f}_L{params.L}_{GENERATOR}".replace(".", "p")
    )
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(params.seed)
    h_grid, r_grid, cdfs = build_heatbath_table(params)
    phi = rng.normal(0.0, 0.7, size=(params.L, params.L)).astype(np.float64)
    streaming_path = out / "configs_streaming.npy"
    status_path = out / "streaming_status.json"
    existing_count = 0
    existing_last_sweep: int | None = None
    existing_seed: int | None = None
    existing_configs_path = out / "configs.npz"
    append_mode = bool(args.append_existing)
    if append_mode:
        if not existing_configs_path.exists():
            raise SystemExit(f"--append-existing requires existing final configs: {existing_configs_path}")
        with np.load(existing_configs_path) as z:
            old_phi = z["phi"]
            existing_count = int(old_phi.shape[0])
            if old_phi.shape[1:] != (params.L, params.L):
                raise SystemExit(f"existing configs have shape {old_phi.shape}; requested L={params.L}")
            if "lambda" in z.files and not np.isclose(float(z["lambda"]), params.lam):
                raise SystemExit(f"existing lambda={float(z['lambda'])} does not match requested lambda={params.lam}")
            if "kappa" in z.files and not np.isclose(float(z["kappa"]), params.kappa):
                raise SystemExit(f"existing kappa={float(z['kappa'])} does not match requested kappa={params.kappa}")
            if "seed" in z.files:
                existing_seed = int(z["seed"])
            total_configs = existing_count + params.n_configs
            stream_out = streaming_path
            if streaming_path.exists():
                stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                stream_out = out / f"configs_streaming_append_from{existing_count}_{stamp}.npy"
            configs = np.lib.format.open_memmap(stream_out, mode="w+", dtype=np.float32, shape=(total_configs, params.L, params.L))
            copy_existing_configs(old_phi, configs)
            phi = old_phi[-1].astype(np.float64)
        log_count, existing_last_sweep = read_generation_log_tail(out / "generation_log.csv")
        if log_count and log_count != existing_count:
            raise SystemExit(f"generation_log.csv has {log_count} rows but configs.npz has {existing_count} configs; refusing ambiguous append")
        backup_stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        shutil.copy2(existing_configs_path, out / f"configs.before_append_{backup_stamp}.npz")
        if (out / "manifest.json").exists():
            shutil.copy2(out / "manifest.json", out / f"manifest.before_append_{backup_stamp}.json")
        if (out / "provenance.json").exists():
            shutil.copy2(out / "provenance.json", out / f"provenance.before_append_{backup_stamp}.json")
        streaming_path = stream_out
    elif streaming_path.exists() or (out / "generation_log.csv").exists():
        raise SystemExit(
            "refusing to overwrite existing streaming output; use a fresh output directory "
            f"or remove {streaming_path} and generation_log.csv"
        )
    else:
        configs = np.lib.format.open_memmap(
            streaming_path,
            mode="w+",
            dtype=np.float32,
            shape=(params.n_configs, params.L, params.L),
        )
    cluster_sizes: list[int] = []
    start = time.time()
    atomic_write_json(
        status_path,
        {
            "status": "running",
            "mode": "append" if append_mode else "new",
            "existing_configs": existing_count,
            "new_completed_configs": 0,
            "completed_configs": existing_count,
            "target_configs": existing_count + params.n_configs,
            "streaming_configs_file": streaming_path.name,
            "final_configs_file": "configs.npz",
            "started_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    for _ in range(params.thermal_sweeps):
        cluster_sizes.extend(sweep(phi, params, h_grid, r_grid, cdfs, rng))
    for j in range(params.n_configs):
        for _ in range(params.skip_sweeps):
            cluster_sizes.extend(sweep(phi, params, h_grid, r_grid, cdfs, rng))
        i = existing_count + j
        configs[i] = phi.astype(np.float32)
        configs.flush()
        obs = observables(phi, params)
        base_sweep = existing_last_sweep if append_mode and existing_last_sweep is not None else params.thermal_sweeps
        row = {"config_index": i, "sweep": base_sweep + (j + 1) * params.skip_sweeps, **obs, "last_cluster_size": cluster_sizes[-1]}
        append_generation_row(out / "generation_log.csv", row)
        atomic_write_json(
            status_path,
            {
                "status": "running",
                "mode": "append" if append_mode else "new",
                "existing_configs": existing_count,
                "new_completed_configs": j + 1,
                "completed_configs": i + 1,
                "target_configs": existing_count + params.n_configs,
                "last_config_index": i,
                "last_sweep": row["sweep"],
                "streaming_configs_file": streaming_path.name,
                "final_configs_file": "configs.npz",
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": time.time() - start,
            },
        )
    np.savez_compressed(
        out / "configs.npz",
        phi=configs,
        **{
            "lambda": np.array(params.lam),
            "kappa": np.array(params.kappa),
            "L": np.array(params.L),
            "n_configs": np.array(existing_count + params.n_configs),
            "added_configs": np.array(params.n_configs),
            "generator": np.array(GENERATOR),
            "seed": np.array(params.seed),
            "previous_seed": np.array(-1 if existing_seed is None else existing_seed),
            "action_convention": np.array(ACTION),
        },
    )
    manifest = {
        "lambda": params.lam,
        "kappa": params.kappa,
        "M2": None,
        "action_convention": ACTION,
        "L": params.L,
        "n_configs": existing_count + params.n_configs,
        "added_configs": params.n_configs,
        "append_existing": append_mode,
        "previous_n_configs": existing_count,
        "shape": list(configs.shape),
        "dtype": str(configs.dtype),
        "generator": GENERATOR,
        "canonical": True,
        "is_canonical": True,
        "production_use": True,
        "is_superseded": False,
        "local_metropolis_used": False,
        "sign_update": "embedded_wolff_cluster",
        "amplitude_update": "radial_heatbath",
        "seed": params.seed,
        "previous_seed": existing_seed,
        "source_path": str(out / "configs.npz"),
        "date_copied": None,
        "date_generated": datetime.now(timezone.utc).isoformat(),
        "parameter_status": "metadata",
        "configs_file": "configs.npz",
        "notes": "Canonical finite-lambda generator: embedded Wolff sign clusters plus radial heat-bath amplitude updates.",
    }
    provenance = {
        "script": str(Path(__file__).resolve()),
        "archived_script": "generate_phi4_embedded_wolff_radial_heatbath.py",
        "parameters": asdict(params),
        "append_existing": append_mode,
        "previous_n_configs": existing_count,
        "added_configs": params.n_configs,
        "generator": GENERATOR,
        "action_convention": ACTION,
        "elapsed_seconds": time.time() - start,
        "mean_cluster_size": float(np.mean(cluster_sizes)) if cluster_sizes else None,
        "max_cluster_size": int(np.max(cluster_sizes)) if cluster_sizes else None,
        "heatbath_table": {
            "r_max": params.r_max,
            "r_grid_size": params.r_grid_size,
            "h_abs_max": params.h_abs_max,
            "h_grid_size": params.h_grid_size,
        },
    }
    write_json(out / "manifest.json", manifest)
    write_json(out / "provenance.json", provenance)
    atomic_write_json(
        status_path,
        {
            "status": "completed",
            "mode": "append" if append_mode else "new",
            "existing_configs": existing_count,
            "new_completed_configs": params.n_configs,
            "completed_configs": existing_count + params.n_configs,
            "target_configs": existing_count + params.n_configs,
            "streaming_configs_file": streaming_path.name,
            "final_configs_file": "configs.npz",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.time() - start,
        },
    )
    shutil.copy2(Path(__file__), out / "generate_phi4_embedded_wolff_radial_heatbath.py")
    report = f"""# Lambda={params.lam:g} Kappa={params.kappa:g} L{params.L} Generation Report

Generated `{params.n_configs}` new finite-lambda phi4 configurations with
`{GENERATOR}`.

- append existing: `{append_mode}`
- previous configs: `{existing_count}`
- total configs: `{existing_count + params.n_configs}`
- local Metropolis used: false
- sign update: embedded Wolff cluster
- amplitude update: radial heat bath
- action convention: `{ACTION}`
- mean cluster size: `{provenance['mean_cluster_size']:.6g}`
- elapsed seconds: `{provenance['elapsed_seconds']:.3f}`

This ensemble is canonical for finite-lambda phi4 production and diagnostics.
"""
    (out / "generation_report.md").write_text(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
