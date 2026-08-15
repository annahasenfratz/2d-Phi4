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
FINITE = PROJECT_ROOT / "InverseBlocking_lam0p022_finite_footprint_flow"
FROZEN = PROJECT_ROOT / "InverseBlocking_lam0p022_frozen"
for p in [PKG / "scripts", PKG / "src", PROJECT_ROOT, FINITE / "scripts", FROZEN / "scripts"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import run_shape_parametric_sampler_validation as sampler  # noqa: E402
from _common import load_config, load_frozen_models, load_kernel_spec, resolve_run_paths  # noqa: E402
from perfect_blocking_upsampling.actions import action_total  # noqa: E402
from train_finite_footprint_transported_detail import inner_patch_metropolis, patch_sites  # noqa: E402

DEFAULT_CONFIG = PKG / "outputs/shape_parametric_sampler_validation/L32_to_L64_smoke/L32_to_L64_smoke_config.yaml"
DEFAULT_OUT = PKG / "outputs/shape_parametric_sampler_validation/L32_to_L64_batched_larger_patch_thermalization_test"
LOCAL_KEYS = ["phi2", "phi4", "NN", "2NN", "diag", "action_density"]


def validate_patch_size_for_scan(lc: int, patch_size: int) -> None:
    if patch_size % 2 != 0:
        raise ValueError(f"patch size must be even, got {patch_size}")
    if patch_size < 2 or patch_size > lc:
        raise ValueError(f"patch size must satisfy 2 <= P <= Lc, got P={patch_size}, Lc={lc}")


def patches_per_sweep(lc: int, patch_size: int) -> int:
    validate_patch_size_for_scan(lc, patch_size)
    return int(math.ceil(2.0 * lc * lc / float(patch_size * patch_size)))


def random_origin_patch_schedule(lc: int, patch_size: int, rng: np.random.Generator, origin_mode: str = "random") -> list[tuple[int, int, str]]:
    validate_patch_size_for_scan(lc, patch_size)
    n_patches = patches_per_sweep(lc, patch_size)
    if origin_mode not in {"random", "deterministic_mixing"}:
        raise ValueError(f"unknown origin_mode={origin_mode!r}")
    start_x = int(rng.integers(0, lc))
    start_y = int(rng.integers(0, lc))
    origins = [(start_x, start_y)]
    if origin_mode == "random":
        for _ in range(1, n_patches):
            origins.append((int(rng.integers(0, lc)), int(rng.integers(0, lc))))
    else:
        sx = max(1, patch_size // 2)
        sy = max(1, patch_size // 2 + 1)
        for k in range(1, n_patches):
            origins.append(((start_x + k * sx) % lc, (start_y + k * sy) % lc))
    return [(int(x), int(y), f"patch_{idx}_origin_{int(x)}_{int(y)}") for idx, (x, y) in enumerate(origins)]


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=float) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys: list[str] = []
    for row in rows:
        for k in row:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def load_phi(path: Path) -> np.ndarray:
    return np.load(path)["phi"].astype(np.float32)


def read_manifest(path: Path) -> dict[str, Any]:
    m = path.with_name("manifest.json")
    return json.loads(m.read_text()) if m.exists() else {}


def local_series(phi: np.ndarray, action: Any) -> dict[str, np.ndarray]:
    arr = np.asarray(phi, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr[None]
    nn = 0.5 * (
        np.mean(arr * np.roll(arr, -1, axis=1), axis=(1, 2))
        + np.mean(arr * np.roll(arr, -1, axis=2), axis=(1, 2))
    )
    two_nn = 0.5 * (
        np.mean(arr * np.roll(arr, -2, axis=1), axis=(1, 2))
        + np.mean(arr * np.roll(arr, -2, axis=2), axis=(1, 2))
    )
    diag = 0.5 * (
        np.mean(arr * np.roll(np.roll(arr, -1, axis=1), -1, axis=2), axis=(1, 2))
        + np.mean(arr * np.roll(np.roll(arr, -1, axis=1), 1, axis=2), axis=(1, 2))
    )
    return {
        "phi2": np.mean(arr**2, axis=(1, 2)),
        "phi4": np.mean(arr**4, axis=(1, 2)),
        "NN": nn,
        "2NN": two_nn,
        "diag": diag,
        "action_density": action_total(arr, action) / (arr.shape[1] * arr.shape[2]),
    }


def local_row(phi: np.ndarray, action: Any) -> dict[str, float]:
    return {k: float(v[0]) for k, v in local_series(phi, action).items()}


def ref_summary(phi: np.ndarray, action: Any) -> dict[str, float]:
    ser = local_series(phi, action)
    return {k: float(np.mean(v)) for k, v in ser.items()}


def mean(vals: list[float]) -> float:
    return float(np.mean(np.asarray(vals, dtype=np.float64))) if vals else float("nan")


def std(vals: list[float]) -> float:
    arr = np.asarray(vals, dtype=np.float64)
    return float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0


def rms_relative(row: dict[str, float], ref: dict[str, float]) -> float:
    terms = []
    for k in LOCAL_KEYS:
        denom = max(abs(float(ref[k])), 1.0e-12)
        terms.append(((float(row[k]) - float(ref[k])) / denom) ** 2)
    return float(math.sqrt(sum(terms) / len(terms)))


def load_ctx(config: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any], Any, Any, Path, Path, dict[str, Any]]:
    cfg = load_config(config)
    paths = resolve_run_paths(cfg)
    coarse_path = paths["coarse_ensemble"]
    fine_path = paths["fine_reference"]
    coarse = load_phi(coarse_path)
    fine_ref = load_phi(fine_path)
    refine_model, refine_state, stages, coarse_action, fine_action, _ = load_frozen_models(cfg)
    refine_model.load_state_dict(refine_state, strict=False)
    refine_model.eval()
    for model, _, state, *_ in stages.values():
        model.load_state_dict(state)
        model.eval()
    kernel, _ = load_kernel_spec(cfg)
    ctx = {"refine_model": refine_model, "stages": stages, "coarse_action": coarse_action, "fine_action": fine_action, "kernel": kernel}
    manifest = read_manifest(coarse_path)
    if coarse.shape[1:] != (32, 32):
        raise ValueError(f"expected native L32 starts, got {coarse.shape}")
    if fine_ref.shape[1:] != (64, 64):
        raise ValueError(f"expected L64 reference, got {fine_ref.shape}")
    if manifest.get("kappa") is not None and abs(float(manifest["kappa"]) - 0.2705) > 1.0e-12:
        raise ValueError(f"native L32 manifest kappa is not 0.2705: {manifest.get('kappa')}")
    return coarse, fine_ref, ctx, coarse_action, fine_action, coarse_path, fine_path, manifest


def propose_patch_scaled(state: dict[str, Any], x0: int, y0: int, tile: int, rng: np.random.Generator, ctx: dict[str, Any], patch_size: int, scale_mult: float) -> tuple[dict[str, Any], dict[str, Any]]:
    sites = patch_sites(state["u"].shape[1], x0, y0, patch_size)
    u_new = state["u"][0].copy()
    u_new, inner_acc = inner_patch_metropolis(u_new, sites, rng, sigma=0.1 * scale_mult)
    prop = sampler.compute_state(u_new[None], state["z_edge"], state["z_pair"], state["z_corner"], ctx)
    return prop, {
        "patch_x": x0,
        "patch_y": y0,
        "tile": tile,
        "inner_sigma": 0.1 * scale_mult,
        "inner_acceptance": inner_acc,
        "delta_logw": float(prop["logw"][0] - state["logw"][0]),
        "delta_Sf": float(prop["sf"][0] - state["sf"][0]),
        "delta_Sc": float(prop["sc"][0] - state["sc"][0]),
        "delta_logdet_refine": float(prop["logdet"][0] - state["logdet"][0]),
        "delta_logq_missing": float(prop["logq"][0] - state["logq"][0]),
    }


def split_batch_state(batch: dict[str, Any], i: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    n = int(batch["u"].shape[0])
    for key, val in batch.items():
        if isinstance(val, np.ndarray) and val.shape[:1] == (n,):
            out[key] = val[i : i + 1].copy()
        else:
            out[key] = val
    return out


def stack_state_field(states: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.concatenate([s[key] for s in states], axis=0)


def batched_propose_patch_scaled(states: list[dict[str, Any]], schedules_at_attempt: list[tuple[int, int, int]], rngs: list[np.random.Generator], ctx: dict[str, Any], patch_size: int, scale_mult: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    u_new_rows = []
    inner_accs = []
    for state, (x0, y0, _tile), rng in zip(states, schedules_at_attempt, rngs):
        sites = patch_sites(state["u"].shape[1], x0, y0, patch_size)
        u_new = state["u"][0].copy()
        u_new, inner_acc = inner_patch_metropolis(u_new, sites, rng, sigma=0.1 * scale_mult)
        u_new_rows.append(u_new[None])
        inner_accs.append(float(inner_acc))
    batch = sampler.compute_state(
        np.concatenate(u_new_rows, axis=0),
        stack_state_field(states, "z_edge"),
        stack_state_field(states, "z_pair"),
        stack_state_field(states, "z_corner"),
        ctx,
    )
    proposals = [split_batch_state(batch, i) for i in range(len(states))]
    deltas = []
    for i, (state, proposal, (x0, y0, tile)) in enumerate(zip(states, proposals, schedules_at_attempt)):
        deltas.append(
            {
                "patch_x": x0,
                "patch_y": y0,
                "tile": tile,
                "inner_sigma": 0.1 * scale_mult,
                "inner_acceptance": inner_accs[i],
                "delta_logw": float(proposal["logw"][0] - state["logw"][0]),
                "delta_Sf": float(proposal["sf"][0] - state["sf"][0]),
                "delta_Sc": float(proposal["sc"][0] - state["sc"][0]),
                "delta_logdet_refine": float(proposal["logdet"][0] - state["logdet"][0]),
                "delta_logq_missing": float(proposal["logq"][0] - state["logq"][0]),
            }
        )
    return proposals, deltas


def batched_propose_latent(states: list[dict[str, Any]], schedules_at_attempt: list[tuple[int, int, int]], rngs: list[np.random.Generator], ctx: dict[str, Any], cfg_tmp: sampler.ValidationConfig) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rho = cfg_tmp.pcn_rho
    noise = math.sqrt(max(0.0, 1.0 - rho * rho))
    z_edges, z_pairs, z_corners = [], [], []
    for state, (x0, y0, _tile), rng in zip(states, schedules_at_attempt, rngs):
        sites = patch_sites(state["u"].shape[1], x0, y0, cfg_tmp.patch_size)
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
    batch = sampler.compute_state(
        stack_state_field(states, "u"),
        np.concatenate(z_edges, axis=0),
        np.concatenate(z_pairs, axis=0),
        np.concatenate(z_corners, axis=0),
        ctx,
    )
    proposals = [split_batch_state(batch, i) for i in range(len(states))]
    deltas = []
    for state, proposal, (x0, y0, tile) in zip(states, proposals, schedules_at_attempt):
        deltas.append(
            {
                "patch_x": x0,
                "patch_y": y0,
                "tile": tile,
                "delta_logw": float(proposal["logw"][0] - state["logw"][0]),
                "delta_Sf": float(proposal["sf"][0] - state["sf"][0]),
                "delta_Sc": 0.0,
                "delta_logdet_refine": float(proposal["logdet"][0] - state["logdet"][0]),
                "delta_logq_missing": float(proposal["logq"][0] - state["logq"][0]),
            }
        )
    return proposals, deltas


def accept(delta_logw: float, rng: np.random.Generator) -> bool:
    return math.log(max(rng.random(), 1.0e-300)) < min(0.0, float(delta_logw))


def run_case(case: dict[str, Any], coarse: np.ndarray, ctx: dict[str, Any], fine_action: Any, direct_ref: dict[str, float], chains: int, sweeps: int, save_sweeps: set[int], seed: int, pcn_rho: float, out: Path, batch_logq_across_chains: bool, latent_patch_size: int | None = None) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    patch_size = int(case["patch_size"])
    latent_patch_size = patch_size if latent_patch_size is None else int(latent_patch_size)
    scale_mult = float(case["scale_mult"])
    n_patch = patches_per_sweep(32, patch_size)
    obs_rows: list[dict[str, Any]] = []
    coarse_rows: list[dict[str, Any]] = []
    latent_rows: list[dict[str, Any]] = []
    sweep_time_rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    if batch_logq_across_chains:
        rngs = [np.random.default_rng(seed + 10000 * chain + 1000 * patch_size + int(100 * round(scale_mult, 4))) for chain in range(chains)]
        cfg_tmp = sampler.ValidationConfig(patch_size=patch_size, origin_mode="random", pcn_rho=pcn_rho)
        cfg_latent = sampler.ValidationConfig(patch_size=latent_patch_size, origin_mode="random", pcn_rho=pcn_rho)
        init_indices = [sampler.choose_initial_index(coarse, chain, cfg_tmp, rngs[chain])[0] for chain in range(chains)]
        u0 = np.stack([coarse[idx] for idx in init_indices], axis=0).astype(np.float32)
        z_edges, z_pairs, z_corners = [], [], []
        for rng in rngs:
            ze, zp, zc = sampler.sample_z(rng, 1, 32)
            z_edges.append(ze)
            z_pairs.append(zp)
            z_corners.append(zc)
        batch0 = sampler.compute_state(u0, np.concatenate(z_edges, axis=0), np.concatenate(z_pairs, axis=0), np.concatenate(z_corners, axis=0), ctx)
        states = [split_batch_state(batch0, i) for i in range(chains)]
        if 0 in save_sweeps:
            for chain, state in enumerate(states):
                row = {"case": case["label"], "chain_id": chain, "sweep": 0, "wall_sec": 0.0, "coarse_index": int(init_indices[chain]), **local_row(state["phi"], fine_action)}
                row["rms_relative_error_vs_native_l64"] = rms_relative(row, direct_ref)
                obs_rows.append(row)
        for sweep in range(1, sweeps + 1):
            t_sweep = time.perf_counter()
            schedules = [random_origin_patch_schedule(32, patch_size, rng, "random") for rng in rngs]
            for attempt in range(n_patch):
                proposals, deltas = batched_propose_patch_scaled(states, [schedule[attempt] for schedule in schedules], rngs, ctx, patch_size, scale_mult)
                for chain in range(chains):
                    a = accept(deltas[chain]["delta_logw"], rngs[chain])
                    if a:
                        states[chain] = proposals[chain]
                    coarse_rows.append({"case": case["label"], "chain_id": chain, "sweep": sweep, "attempt_in_sweep": attempt, "accepted": int(a), **deltas[chain]})
            latent_schedules = [random_origin_patch_schedule(32, latent_patch_size, rng, "random") for rng in rngs]
            proposals_l, deltas_l = batched_propose_latent(states, [schedule[-1] for schedule in latent_schedules], rngs, ctx, cfg_latent)
            for chain in range(chains):
                a_l = accept(deltas_l[chain]["delta_logw"], rngs[chain])
                if a_l:
                    states[chain] = proposals_l[chain]
                latent_rows.append({"case": case["label"], "chain_id": chain, "sweep": sweep, "attempt_in_sweep": n_patch - 1, "accepted": int(a_l), **deltas_l[chain]})
            elapsed = time.perf_counter() - start
            for chain, state in enumerate(states):
                sweep_time_rows.append({"case": case["label"], "chain_id": chain, "sweep": sweep, "sweep_sec": time.perf_counter() - t_sweep, "wall_sec_case": elapsed})
                if sweep in save_sweeps:
                    row = {"case": case["label"], "chain_id": chain, "sweep": sweep, "wall_sec": elapsed, "coarse_index": int(init_indices[chain]), **local_row(state["phi"], fine_action)}
                    row["rms_relative_error_vs_native_l64"] = rms_relative(row, direct_ref)
                    obs_rows.append(row)
            if sweep == 1 or sweep % 25 == 0 or sweep == sweeps:
                print(f"{case['label']} batched sweep {sweep}/{sweeps}: coarse_acc={mean([r['accepted'] for r in coarse_rows]):.4f}", flush=True)
    else:
      for chain in range(chains):
        rng = np.random.default_rng(seed + 10000 * chain + 1000 * patch_size + int(100 * round(scale_mult, 4)))
        cfg_tmp = sampler.ValidationConfig(patch_size=patch_size, origin_mode="random", pcn_rho=pcn_rho)
        cfg_latent = sampler.ValidationConfig(patch_size=latent_patch_size, origin_mode="random", pcn_rho=pcn_rho)
        init_idx, _sector = sampler.choose_initial_index(coarse, chain, cfg_tmp, rng)
        u = coarse[init_idx][None]
        z_edge, z_pair, z_corner = sampler.sample_z(rng, 1, 32)
        state = sampler.compute_state(u, z_edge, z_pair, z_corner, ctx)
        if 0 in save_sweeps:
            row = {"case": case["label"], "chain_id": chain, "sweep": 0, "wall_sec": 0.0, "coarse_index": int(init_idx), **local_row(state["phi"], fine_action)}
            row["rms_relative_error_vs_native_l64"] = rms_relative(row, direct_ref)
            obs_rows.append(row)
        for sweep in range(1, sweeps + 1):
            t_sweep = time.perf_counter()
            schedule = random_origin_patch_schedule(32, patch_size, rng, "random")
            for attempt, (x0, y0, tile) in enumerate(schedule):
                prop, delta = propose_patch_scaled(state, x0, y0, tile, rng, ctx, patch_size, scale_mult)
                a = accept(delta["delta_logw"], rng)
                if a:
                    state = prop
                coarse_rows.append({"case": case["label"], "chain_id": chain, "sweep": sweep, "attempt_in_sweep": attempt, "accepted": int(a), **delta})
            latent_schedule = random_origin_patch_schedule(32, latent_patch_size, rng, "random")
            prop_l, delta_l = sampler.propose_latent(state, *latent_schedule[-1], rng, ctx, cfg_latent)
            a_l = accept(delta_l["delta_logw"], rng)
            if a_l:
                state = prop_l
            latent_rows.append({"case": case["label"], "chain_id": chain, "sweep": sweep, "attempt_in_sweep": n_patch - 1, "accepted": int(a_l), **delta_l})
            elapsed = time.perf_counter() - start
            sweep_time_rows.append({"case": case["label"], "chain_id": chain, "sweep": sweep, "sweep_sec": time.perf_counter() - t_sweep, "wall_sec_case": elapsed})
            if sweep in save_sweeps:
                row = {"case": case["label"], "chain_id": chain, "sweep": sweep, "wall_sec": elapsed, "coarse_index": int(init_idx), **local_row(state["phi"], fine_action)}
                row["rms_relative_error_vs_native_l64"] = rms_relative(row, direct_ref)
                obs_rows.append(row)
        print(f"{case['label']} chain {chain} complete: coarse_acc={mean([r['accepted'] for r in coarse_rows if r['chain_id']==chain]):.4f}", flush=True)
    wall = time.perf_counter() - start
    cacc = [float(r["accepted"]) for r in coarse_rows]
    lacc = [float(r["accepted"]) for r in latent_rows]
    summary = {
        "case": case["label"],
        "patch_size": patch_size,
        "latent_patch_size": latent_patch_size,
        "scale_mult": scale_mult,
        "pcn_rho": pcn_rho,
        "inner_sigma": 0.1 * scale_mult,
        "chains": chains,
        "sweeps": sweeps,
        "N_patch_per_sweep": n_patch,
        "coarse_attempts": len(coarse_rows),
        "latent_attempts": len(latent_rows),
        "coarse_acceptance": mean(cacc),
        "coarse_delta_logw_std": std([float(r["delta_logw"]) for r in coarse_rows]),
        "latent_acceptance": mean(lacc),
        "latent_delta_logw_std": std([float(r["delta_logw"]) for r in latent_rows]),
        "wall_sec": wall,
        "wall_sec_per_chain_sweep": wall / max(chains * sweeps, 1),
        "attempts_per_sweep": n_patch,
        "batch_logq_across_chains": bool(batch_logq_across_chains),
    }
    write_csv(out / f"{case['label']}_observable_flow.csv", obs_rows)
    write_csv(out / f"{case['label']}_coarse_deltas.csv", coarse_rows)
    write_csv(out / f"{case['label']}_latent_deltas.csv", latent_rows)
    write_csv(out / f"{case['label']}_sweep_times.csv", sweep_time_rows)
    return summary, obs_rows, coarse_rows, latent_rows


def aggregate_obs(obs_rows: list[dict[str, Any]], summaries: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    flow_rows = []
    rms_rows = []
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in obs_rows:
        grouped.setdefault((str(row["case"]), int(row["sweep"])), []).append(row)
    for (case, sweep), rows in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1])):
        out = {"case": case, "patch_size": summaries[case]["patch_size"], "scale_mult": summaries[case]["scale_mult"], "sweep": sweep, "n_chains": len(rows)}
        for k in LOCAL_KEYS:
            out[k] = mean([float(r[k]) for r in rows])
        out["rms_relative_error_vs_native_l64"] = mean([float(r["rms_relative_error_vs_native_l64"]) for r in rows])
        out["wall_sec_mean"] = mean([float(r["wall_sec"]) for r in rows])
        flow_rows.append(out)
        rms_rows.append({
            "case": case,
            "patch_size": summaries[case]["patch_size"],
            "scale_mult": summaries[case]["scale_mult"],
            "sweep": sweep,
            "wall_sec_mean": out["wall_sec_mean"],
            "rms_relative_error_vs_native_l64": out["rms_relative_error_vs_native_l64"],
        })
    return flow_rows, rms_rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--sweeps", type=int, default=100)
    ap.add_argument("--save-sweeps", default="0,1,2,5,10,20,40,60,80,100,120,140,160,180,200")
    ap.add_argument("--seed", type=int, default=20260722)
    ap.add_argument("--pcn-rho", type=float, default=0.5)
    ap.add_argument("--latent-patch-size", type=int, default=None, help="Latent/detail pCN patch size. Defaults to the coarse patch size.")
    ap.add_argument("--batch-logq-across-chains", action="store_true")
    ap.add_argument("--case", action="append", choices=["P4_scale1", "P6_scale4over6", "P8_scale4over8", "P12_scale4over12", "P16_scale4over16", "P24_scale4over24", "P32_scale4over32"], help="Run only selected case(s). May be repeated.")
    args = ap.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    save_sweeps = {int(x) for x in args.save_sweeps.replace(" ", ",").split(",") if x}
    coarse, fine_ref, ctx, _coarse_action, fine_action, coarse_path, fine_path, manifest = load_ctx(args.config)
    direct = ref_summary(fine_ref, fine_action)
    cases = [
        {"label": "P4_scale1", "patch_size": 4, "scale_mult": 1.0},
        {"label": "P6_scale4over6", "patch_size": 6, "scale_mult": 4.0 / 6.0},
        {"label": "P8_scale4over8", "patch_size": 8, "scale_mult": 4.0 / 8.0},
        {"label": "P12_scale4over12", "patch_size": 12, "scale_mult": 4.0 / 12.0},
        {"label": "P16_scale4over16", "patch_size": 16, "scale_mult": 4.0 / 16.0},
        {"label": "P24_scale4over24", "patch_size": 24, "scale_mult": 4.0 / 24.0},
        {"label": "P32_scale4over32", "patch_size": 32, "scale_mult": 4.0 / 32.0},
    ]
    if args.case:
        wanted = set(args.case)
        cases = [case for case in cases if case["label"] in wanted]
    preflight = {
        "coarse_path": str(coarse_path),
        "coarse_shape": list(coarse.shape),
        "coarse_manifest": manifest,
        "fine_reference_path": str(fine_path),
        "fine_reference_shape": list(fine_ref.shape),
        "chains": args.chains,
        "sweeps": args.sweeps,
        "save_sweeps": sorted(save_sweeps),
        "cases": [
            {
                **c,
                "latent_patch_size": args.latent_patch_size if args.latent_patch_size is not None else int(c["patch_size"]),
                "N_patch_per_sweep": patches_per_sweep(32, int(c["patch_size"])),
                "expected_coarse_attempts": args.chains * args.sweeps * patches_per_sweep(32, int(c["patch_size"])),
                "expected_latent_attempts": args.chains * args.sweeps,
            }
            for c in cases
        ],
        "native_l64_reference_local": direct,
    }
    write_json(out / "preflight.json", preflight)
    summaries: list[dict[str, Any]] = []
    obs_all: list[dict[str, Any]] = []
    for case in cases:
        print(f"running {case['label']} P={case['patch_size']} scale={case['scale_mult']}", flush=True)
        summary, obs_rows, _coarse_rows, _latent_rows = run_case(case, coarse, ctx, fine_action, direct, args.chains, args.sweeps, save_sweeps, args.seed, args.pcn_rho, out, args.batch_logq_across_chains, latent_patch_size=args.latent_patch_size)
        summaries.append(summary)
        obs_all.extend(obs_rows)
        write_csv(out / "patch_size_runtime_acceptance_summary.csv", summaries)
    summary_by_case = {s["case"]: s for s in summaries}
    flow_rows, rms_rows = aggregate_obs(obs_all, summary_by_case)
    write_csv(out / "local_operator_flow_by_patch_size.csv", flow_rows)
    write_csv(out / "rms_local_error_vs_sweep_and_time.csv", rms_rows)
    def last_rms(case: str) -> float:
        rows = [r for r in rms_rows if r["case"] == case and int(r["sweep"]) == args.sweeps]
        return float(rows[0]["rms_relative_error_vs_native_l64"]) if rows else float("nan")
    best_by_wall = min((r for r in rms_rows if int(r["sweep"]) == args.sweeps), key=lambda r: float(r["rms_relative_error_vs_native_l64"]))
    lines = [
        "# L32->L64 batched larger-patch thermalization test",
        "",
        "Short native-coarse L32->L64 thermalization comparison. No long validation was launched.",
        "",
        f"- native coarse starts: `{coarse_path}`",
        f"- native coarse manifest kappa: `{manifest.get('kappa')}`",
        f"- fine reference: `{fine_path}`",
        f"- chains x sweeps: `{args.chains} x {args.sweeps}`",
        f"- batch logq across chains: `{args.batch_logq_across_chains}`",
        f"- save sweeps: `{sorted(save_sweeps)}`",
        "",
        "## Runtime and A/R",
        "",
        "| case | P | scale | attempts/sweep | coarse attempts | coarse acc | coarse std dlogw | latent acc | wall sec | sec/chain-sweep |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for s in summaries:
        lines.append(f"| `{s['case']}` | {s['patch_size']} | {s['scale_mult']:.4g} | {s['attempts_per_sweep']} | {s['coarse_attempts']} | {s['coarse_acceptance']:.6g} | {s['coarse_delta_logw_std']:.6g} | {s['latent_acceptance']:.6g} | {s['wall_sec']:.3f} | {s['wall_sec_per_chain_sweep']:.6g} |")
    lines += [
        "",
        "## Local RMS error at final sweep",
        "",
        "| case | final RMS relative local error |",
        "| --- | ---: |",
    ]
    for s in summaries:
        lines.append(f"| `{s['case']}` | {last_rms(str(s['case'])):.6g} |")
    lines += [
        "",
        "## Answers",
        "",
        f"1. Faster local thermalization by sweep/time: best final-sweep RMS row is `{best_by_wall['case']}` with RMS `{float(best_by_wall['rms_relative_error_vs_native_l64']):.6g}`.",
        "2. Larger patches reduce attempts per sweep substantially: P4=128, P6=57, P8=32.",
        "3. P8 stability is assessed by its NaN-free completion plus A/R table above.",
        "4. Recommendation should be based on the final RMS table together with acceptance/runtime; this is still a short diagnostic, not production validation.",
        "",
        "Optional `kappa_f=0.27075` was not run in this pass; this test used same-kappa `kappa_f=0.2705` only.",
    ]
    (out / "BATCHED_LARGER_PATCH_THERMALIZATION_TEST.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"out": str(out), "summaries": summaries, "best_final_rms": best_by_wall}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
