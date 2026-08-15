#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PKG = PROJECT_ROOT / "perfect_blocking_upsampling"
sys.path.insert(0, str(PKG / "scripts"))
sys.path.insert(0, str(PKG / "src"))

import run_shape_parametric_sampler_validation as sampler  # noqa: E402
from _common import load_config, resolve_run_paths  # noqa: E402
from profile_L32_to_L64_patch_runtime import Timers, load_ctx, measure_observables, timed_compute_state  # noqa: E402
from run_L32_to_L64_kappaf_matching_experiment import split_batch_state, stack_state_field  # noqa: E402
from train_finite_footprint_transported_detail import inner_patch_metropolis, patch_sites, patches_per_sweep, random_origin_patch_schedule  # noqa: E402

DEFAULT_CONFIG = PKG / "outputs" / "shape_parametric_sampler_validation" / "L32_to_L64_smoke" / "L32_to_L64_smoke_config.yaml"
DEFAULT_OUT = PKG / "outputs" / "shape_parametric_sampler_validation" / "L32_to_L64_batched_logq_across_chains"
LOCAL_KEYS = ["phi2", "phi4", "NN", "2nn", "diag", "action_density"]


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


def summarize_timers(mode: str, timers: Timers, chains: int, sweeps: int, n_patch: int, accepts: list[int], latent_accepts: list[int], dlogw: list[float], latent_dlogw: list[float]) -> dict[str, Any]:
    rows = {r["section"]: r for r in timers.rows()}
    wall = float(rows.get("wall_total", {}).get("total_sec", 0.0))
    pair_corner = 0.0
    for key, row in rows.items():
        if "pair_flow_logq" in key or "corner_flow_logq" in key:
            pair_corner += float(row["total_sec"])
    coarse_eval = sum(float(row["total_sec"]) for key, row in rows.items() if "coarse_patch" in key and "compute_state_total" in key)
    return {
        "mode": mode,
        "chains": chains,
        "sweeps": sweeps,
        "coarse_attempts": chains * sweeps * n_patch,
        "latent_attempts": chains * sweeps,
        "wall_sec": wall,
        "wall_sec_per_chain_sweep": wall / max(chains * sweeps, 1),
        "wall_sec_per_coarse_attempt": wall / max(chains * sweeps * n_patch, 1),
        "coarse_compute_state_total_sec": coarse_eval,
        "pair_corner_flow_logq_sec": pair_corner,
        "pair_corner_flow_logq_fraction_of_wall": pair_corner / wall if wall > 0 else float("nan"),
        "coarse_acceptance": float(np.mean(accepts)) if accepts else float("nan"),
        "latent_acceptance": float(np.mean(latent_accepts)) if latent_accepts else float("nan"),
        "coarse_delta_logw_std": float(np.std(dlogw, ddof=1)) if len(dlogw) > 1 else float("nan"),
        "latent_delta_logw_std": float(np.std(latent_dlogw, ddof=1)) if len(latent_dlogw) > 1 else float("nan"),
    }


def init_states(coarse: np.ndarray, ctx: dict[str, Any], chains: int, seed: int, timers: Timers, mode: str) -> tuple[list[dict[str, Any]], list[np.random.Generator]]:
    rngs = [np.random.default_rng(seed + 10000 * c + 777) for c in range(chains)]
    states = []
    for chain, rng in enumerate(rngs):
        idx = int(rng.integers(0, len(coarse)))
        u = coarse[idx][None]
        z_edge, z_pair, z_corner = sampler.sample_z(rng, 1, 32)
        states.append(timed_compute_state(u, z_edge, z_pair, z_corner, ctx, timers, f"{mode}_initial"))
    return states, rngs


def run_scalar(coarse: np.ndarray, ctx: dict[str, Any], chains: int, sweeps: int, patch_size: int, seed: int) -> tuple[dict[str, Any], list[dict[str, Any]], Timers]:
    timers = Timers()
    wall0 = time.perf_counter()
    states, rngs = init_states(coarse, ctx, chains, seed, timers, "scalar")
    n_patch = patches_per_sweep(32, patch_size)
    accepts: list[int] = []
    latent_accepts: list[int] = []
    dlogw: list[float] = []
    latent_dlogw: list[float] = []
    obs_rows = []
    for chain, state in enumerate(states):
        obs_rows.append({"mode": "scalar", **measure_observables(state["phi"], ctx["fine_action"], timers, chain, 0)})
    for chain in range(chains):
        rng = rngs[chain]
        state = states[chain]
        for sweep in range(1, sweeps + 1):
            schedule = random_origin_patch_schedule(32, patch_size, rng, "random")
            for attempt, (x0, y0, _) in enumerate(schedule):
                sites = patch_sites(state["u"].shape[1], x0, y0, patch_size)
                t0 = time.perf_counter()
                u_new = state["u"][0].copy()
                u_new, _ = inner_patch_metropolis(u_new, sites, rng)
                timers.add("scalar_coarse_patch_proposal_construction", time.perf_counter() - t0)
                proposal = timed_compute_state(u_new[None], state["z_edge"], state["z_pair"], state["z_corner"], ctx, timers, "scalar_coarse_patch")
                delta = float(proposal["logw"][0] - state["logw"][0])
                accept = math.log(max(float(rng.random()), 1e-300)) < min(0.0, delta)
                if accept:
                    state = proposal
                accepts.append(int(accept))
                dlogw.append(delta)
            x0, y0, _ = schedule[-1]
            sites = patch_sites(state["u"].shape[1], x0, y0, patch_size)
            rho = 0.5
            noise = math.sqrt(1.0 - rho * rho)
            z_edge = state["z_edge"].copy()
            z_pair = state["z_pair"].copy()
            z_corner = state["z_corner"].copy()
            for i, j in sites:
                z_edge[0, 0, i, j] = rho * z_edge[0, 0, i, j] + noise * float(rng.standard_normal())
                z_pair[0, 0, i, j] = rho * z_pair[0, 0, i, j] + noise * float(rng.standard_normal())
                z_corner[0, 0, i, j] = rho * z_corner[0, 0, i, j] + noise * float(rng.standard_normal())
            proposal = timed_compute_state(state["u"], z_edge, z_pair, z_corner, ctx, timers, "scalar_latent_pcn")
            delta = float(proposal["logw"][0] - state["logw"][0])
            accept = math.log(max(float(rng.random()), 1e-300)) < min(0.0, delta)
            if accept:
                state = proposal
            latent_accepts.append(int(accept))
            latent_dlogw.append(delta)
        states[chain] = state
        obs_rows.append({"mode": "scalar", **measure_observables(state["phi"], ctx["fine_action"], timers, chain, sweeps)})
    timers.add("wall_total", time.perf_counter() - wall0)
    return summarize_timers("scalar", timers, chains, sweeps, n_patch, accepts, latent_accepts, dlogw, latent_dlogw), obs_rows, timers


def run_batched(coarse: np.ndarray, ctx: dict[str, Any], chains: int, sweeps: int, patch_size: int, seed: int, replay_checks: int) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], Timers]:
    timers = Timers()
    wall0 = time.perf_counter()
    states, rngs = init_states(coarse, ctx, chains, seed, timers, "batched")
    n_patch = patches_per_sweep(32, patch_size)
    accepts: list[int] = []
    latent_accepts: list[int] = []
    dlogw: list[float] = []
    latent_dlogw: list[float] = []
    checks: list[dict[str, Any]] = []
    obs_rows = []
    for chain, state in enumerate(states):
        obs_rows.append({"mode": "batched", **measure_observables(state["phi"], ctx["fine_action"], timers, chain, 0)})
    for sweep in range(1, sweeps + 1):
        schedules = [random_origin_patch_schedule(32, patch_size, rng, "random") for rng in rngs]
        for attempt in range(n_patch):
            u_news = []
            for state, rng, sched in zip(states, rngs, schedules):
                x0, y0, _ = sched[attempt]
                sites = patch_sites(state["u"].shape[1], x0, y0, patch_size)
                u_new = state["u"][0].copy()
                u_new, _ = inner_patch_metropolis(u_new, sites, rng)
                u_news.append(u_new[None])
            batch = timed_compute_state(
                np.concatenate(u_news, axis=0),
                stack_state_field(states, "z_edge"),
                stack_state_field(states, "z_pair"),
                stack_state_field(states, "z_corner"),
                ctx,
                timers,
                "batched_coarse_patch",
            )
            proposals = [split_batch_state(batch, i) for i in range(chains)]
            for chain, (state, proposal) in enumerate(zip(states, proposals)):
                delta = float(proposal["logw"][0] - state["logw"][0])
                if len(checks) < replay_checks:
                    scalar = timed_compute_state(u_news[chain], state["z_edge"], state["z_pair"], state["z_corner"], ctx, timers, "replay_scalar_check")
                    scalar_delta = float(scalar["logw"][0] - state["logw"][0])
                    checks.append({"sweep": sweep, "attempt": attempt, "chain": chain, "batched_delta_logw": delta, "scalar_delta_logw": scalar_delta, "abs_diff": abs(delta - scalar_delta)})
                accept = math.log(max(float(rngs[chain].random()), 1e-300)) < min(0.0, delta)
                if accept:
                    states[chain] = proposal
                accepts.append(int(accept))
                dlogw.append(delta)
        z_edges, z_pairs, z_corners = [], [], []
        for state, rng, sched in zip(states, rngs, schedules):
            x0, y0, _ = sched[-1]
            sites = patch_sites(state["u"].shape[1], x0, y0, patch_size)
            rho = 0.5
            noise = math.sqrt(1.0 - rho * rho)
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
        batch = timed_compute_state(stack_state_field(states, "u"), np.concatenate(z_edges, axis=0), np.concatenate(z_pairs, axis=0), np.concatenate(z_corners, axis=0), ctx, timers, "batched_latent_pcn")
        proposals = [split_batch_state(batch, i) for i in range(chains)]
        for chain, (state, proposal) in enumerate(zip(states, proposals)):
            delta = float(proposal["logw"][0] - state["logw"][0])
            accept = math.log(max(float(rngs[chain].random()), 1e-300)) < min(0.0, delta)
            if accept:
                states[chain] = proposal
            latent_accepts.append(int(accept))
            latent_dlogw.append(delta)
    for chain, state in enumerate(states):
        obs_rows.append({"mode": "batched", **measure_observables(state["phi"], ctx["fine_action"], timers, chain, sweeps)})
    timers.add("wall_total", time.perf_counter() - wall0)
    return summarize_timers("batched", timers, chains, sweeps, n_patch, accepts, latent_accepts, dlogw, latent_dlogw), obs_rows, checks, timers


def summarize_observables(obs_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for mode in sorted({r["mode"] for r in obs_rows}):
        for sweep in sorted({int(r["sweep"]) for r in obs_rows if r["mode"] == mode}):
            sub = [r for r in obs_rows if r["mode"] == mode and int(r["sweep"]) == sweep]
            row: dict[str, Any] = {"mode": mode, "sweep": sweep, "n_chains": len(sub)}
            for key in LOCAL_KEYS:
                vals = np.asarray([float(r[key]) for r in sub], dtype=np.float64)
                row[key] = float(np.mean(vals))
                row[key + "_se"] = float(np.std(vals, ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else 0.0
            rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--kappa-f", type=float, default=0.2705)
    ap.add_argument("--chains", type=int, default=8)
    ap.add_argument("--sweeps", type=int, default=10)
    ap.add_argument("--patch-size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260722)
    ap.add_argument("--replay-checks", type=int, default=32)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    coarse = np.load(resolve_run_paths(cfg)["coarse_ensemble"])["phi"].astype(np.float32)
    timers_for_load = Timers()
    ctx = load_ctx(cfg, args.kappa_f, timers_for_load)
    scalar_summary, scalar_obs, _ = run_scalar(coarse, ctx, args.chains, args.sweeps, args.patch_size, args.seed)
    batched_summary, batched_obs, checks, _ = run_batched(coarse, ctx, args.chains, args.sweeps, args.patch_size, args.seed, args.replay_checks)
    runtime_rows = [scalar_summary, batched_summary]
    write_csv(args.out_dir / "runtime_comparison_scalar_vs_batched.csv", runtime_rows)
    write_csv(args.out_dir / "acceptance_comparison_scalar_vs_batched.csv", [
        {"mode": r["mode"], "coarse_acceptance": r["coarse_acceptance"], "latent_acceptance": r["latent_acceptance"], "coarse_delta_logw_std": r["coarse_delta_logw_std"], "latent_delta_logw_std": r["latent_delta_logw_std"]}
        for r in runtime_rows
    ])
    write_csv(args.out_dir / "log_acceptance_replay_check.csv", checks)
    obs_summary = summarize_observables(scalar_obs + batched_obs)
    write_csv(args.out_dir / "observable_comparison_scalar_vs_batched.csv", obs_summary)
    write_json(args.out_dir / "summary.json", {"runtime": runtime_rows, "max_replay_abs_diff": max([c["abs_diff"] for c in checks], default=float("nan"))})
    speedup = scalar_summary["wall_sec"] / batched_summary["wall_sec"]
    maxdiff = max([c["abs_diff"] for c in checks], default=float("nan"))
    bps = batched_summary["wall_sec_per_chain_sweep"]
    lines = [
        "# Batched logq across chains benchmark",
        "",
        f"- kappa_f: `{args.kappa_f}`",
        f"- chains x sweeps: `{args.chains} x {args.sweeps}`",
        f"- patch_size: `{args.patch_size}`",
        "",
        "## Runtime",
        "",
        "| mode | wall_sec | sec/chain-sweep | sec/coarse-attempt | pair+corner sec | coarse acc | latent acc |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in runtime_rows:
        lines.append(f"| {r['mode']} | {r['wall_sec']:.6g} | {r['wall_sec_per_chain_sweep']:.6g} | {r['wall_sec_per_coarse_attempt']:.6g} | {r['pair_corner_flow_logq_sec']:.6g} | {r['coarse_acceptance']:.6g} | {r['latent_acceptance']:.6g} |")
    lines += [
        "",
        f"Observed wall-time speedup, scalar/batched: `{speedup:.6g}`.",
        "",
        "## Replay Check",
        "",
        f"Max absolute difference between batched and scalar per-proposal `Delta logw` over `{len(checks)}` replay checks: `{maxdiff:.6g}`.",
        "",
        "## Projected Batched Cost",
        "",
        "| run | projected_hours |",
        "|---|---:|",
        f"| 8x300 | {bps * 8 * 300 / 3600:.6g} |",
        f"| 8x1000 | {bps * 8 * 1000 / 3600:.6g} |",
        f"| 16x500 | {bps * 16 * 500 / 3600:.6g} |",
        "",
        "## Answers",
        "",
        "1. Batching across chains is implemented safely because each chain retains its own proposal construction, RNG stream, and Metropolis decision. Only the deterministic full-volume state/logq evaluation is stacked across independent chains.",
        "2. The measured 8-chain speedup is shown above.",
        "3. Deterministic replay checks compare scalar and batched `Delta logw`; see `log_acceptance_replay_check.csv`.",
        "4. Local/action observables are in `observable_comparison_scalar_vs_batched.csv`.",
        "5. If this speedup is small, the next candidate is batching non-overlapping patches, but that requires a separate proof that the update ordering is equivalent or a checkerboard/Gibbs-style kernel with a correct MH construction.",
        "",
        "Future checkpoints should consider bounded-radius local detail flows: fewer coupling layers, no circular wraparound in conditioners, or explicit cropped/local conditioners, so dirty-region incremental logq becomes possible.",
    ]
    (args.out_dir / "BATCHED_LOGQ_ACROSS_CHAINS_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"out_dir": str(args.out_dir), "speedup": speedup, "max_replay_abs_diff": maxdiff}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
