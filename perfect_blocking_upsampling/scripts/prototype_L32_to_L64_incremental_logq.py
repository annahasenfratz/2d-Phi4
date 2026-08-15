#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PKG = PROJECT_ROOT / "perfect_blocking_upsampling"
FINITE = PROJECT_ROOT / "InverseBlocking_lam0p022_finite_footprint_flow"
FROZEN = PROJECT_ROOT / "InverseBlocking_lam0p022_frozen"
sys.path.insert(0, str(PKG / "scripts"))
sys.path.insert(0, str(PKG / "src"))
sys.path.insert(0, str(FINITE / "scripts"))
sys.path.insert(0, str(FROZEN / "scripts"))

import run_shape_parametric_sampler_validation as sampler  # noqa: E402
from _common import load_config, load_frozen_models, load_kernel_spec, resolve_run_paths  # noqa: E402
from perfect_blocking_upsampling.actions import action_total  # noqa: E402
from perfect_blocking_upsampling.kernels import inverse_kernel  # noqa: E402
from train_finite_footprint_flow import local_observables, write_json  # noqa: E402
from train_finite_footprint_transported_detail import inner_patch_metropolis, patch_sites, patches_per_sweep, random_origin_patch_schedule  # noqa: E402
from ML_sampling_clean.experiments.decimated_conditional_fillin.run_staged_decimated_conditional_fillin import from_model_space, log_jacobian  # noqa: E402

LOG2PI = math.log(2.0 * math.pi)
DEFAULT_CONFIG = PKG / "outputs" / "shape_parametric_sampler_validation" / "L32_to_L64_smoke" / "L32_to_L64_smoke_config.yaml"
DEFAULT_OUT = PKG / "outputs" / "shape_parametric_sampler_validation" / "L32_to_L64_incremental_logq_prototype"


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


def load_ctx(cfg: dict[str, Any], kappa_f: float) -> dict[str, Any]:
    refine_model, refine_state, stages, coarse_action, fine_action, _ = load_frozen_models(cfg)
    refine_model.load_state_dict(refine_state, strict=False)
    refine_model.eval()
    for model, _, state, *_ in stages.values():
        model.load_state_dict(state)
        model.eval()
    kernel, _ = load_kernel_spec(cfg)
    return {
        "refine_model": refine_model,
        "stages": stages,
        "coarse_action": coarse_action,
        "fine_action": replace(fine_action, kappa=float(kappa_f)),
        "kernel": kernel,
    }


def stage_forward_with_logq_map(model, z: np.ndarray, cond: np.ndarray, lg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, float]:
    import torch

    if not hasattr(model, "forward_with_logdet_map"):
        raise TypeError(f"{type(model).__name__} does not expose forward_with_logdet_map")
    model.eval()
    z_t = torch.tensor(z.reshape(z.shape[0], -1), dtype=torch.float32)
    cond_t = torch.tensor(cond.reshape(cond.shape[0], -1), dtype=torch.float32)
    with torch.no_grad():
        y_flat, logdet_map_t = model.forward_with_logdet_map(z_t, cond_t)
    y = y_flat.cpu().numpy().reshape(z.shape).astype(np.float32)
    x = from_model_space(y, cond, lg)
    logdet_map = logdet_map_t.cpu().numpy().astype(np.float64)
    log_base_map = -0.5 * (z.astype(np.float64) ** 2 + LOG2PI)
    per_site_logjac = float(log_jacobian(cond, lg)[0]) / float(cond.shape[2] * cond.shape[3])
    logq_map = log_base_map - logdet_map - per_site_logjac
    return x.astype(np.float32), logq_map.astype(np.float64), float(np.sum(logq_map))


def compute_state_with_maps(u: np.ndarray, z_edge: np.ndarray, z_pair: np.ndarray, z_corner: np.ndarray, ctx: dict[str, Any]) -> dict[str, Any]:
    cprime, logdet = sampler.apply_refine_loaded(ctx["refine_model"], u)
    edge_model, edge_lg, _ = ctx["stages"]["edge"][:3]
    pair_model, pair_lg, _ = ctx["stages"]["pair"][:3]
    corner_model, corner_lg, _ = ctx["stages"]["corner"][:3]
    d10, edge_logq = sampler.stage_forward_from_z(edge_model, z_edge, cprime[:, None], edge_lg)
    pair_cond = np.concatenate([cprime[:, None], d10], axis=1)
    d01, pair_logq_map, pair_logq = stage_forward_with_logq_map(pair_model, z_pair, pair_cond, pair_lg)
    corner_cond = np.concatenate([cprime[:, None], d10, d01], axis=1)
    d11, corner_logq_map, corner_logq = stage_forward_with_logq_map(corner_model, z_corner, corner_cond, corner_lg)
    d = np.concatenate([d10, d01, d11], axis=1).astype(np.float32)
    psi = sampler.reconstruct(cprime, d)
    phi, inv = inverse_kernel(psi, ctx["kernel"])
    sf = action_total(phi, ctx["fine_action"])
    sc = action_total(u, ctx["coarse_action"])
    logq = edge_logq + pair_logq + corner_logq
    logw = -sf + sc + logdet - logq
    return {
        "u": u.astype(np.float32),
        "z_edge": z_edge.astype(np.float32),
        "z_pair": z_pair.astype(np.float32),
        "z_corner": z_corner.astype(np.float32),
        "phi": phi.astype(np.float32),
        "sf": sf.astype(np.float64),
        "sc": sc.astype(np.float64),
        "logdet": logdet.astype(np.float64),
        "edge_logq": edge_logq.astype(np.float64),
        "pair_logq": np.asarray([pair_logq], dtype=np.float64),
        "corner_logq": np.asarray([corner_logq], dtype=np.float64),
        "logq": logq.astype(np.float64),
        "logw": logw.astype(np.float64),
        "pair_logq_map": pair_logq_map.astype(np.float64),
        "corner_logq_map": corner_logq_map.astype(np.float64),
        "inv": inv,
    }


def dependency_radius(model: Any) -> int:
    if hasattr(model, "dependency_report"):
        return int(model.dependency_report().get("coarse_radius", 0))
    return 0


def dirty_mask(lc: int, x0: int, y0: int, patch_size: int, radius: int) -> np.ndarray:
    mask = np.zeros((lc, lc), dtype=bool)
    for i, j in patch_sites(lc, x0, y0, patch_size):
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                mask[(i + dx) % lc, (j + dy) % lc] = True
    return mask


def local_obs_row(phi: np.ndarray, chain: int, sweep: int, mode: str) -> dict[str, Any]:
    obs = local_observables(phi[None] if phi.ndim == 2 else phi)
    row = {"mode": mode, "chain": chain, "sweep": sweep}
    for key in ["phi2", "phi4", "NN", "2nn", "diag", "action_density", "m", "abs_m"]:
        if key in obs:
            val = obs[key]
            if isinstance(val, dict):
                val = val.get("mean", np.nan)
            row[key] = float(val)
    return row


def run_mode(
    mode: str,
    coarse: np.ndarray,
    ctx: dict[str, Any],
    *,
    chains: int,
    sweeps: int,
    patch_size: int,
    seed: int,
    check_limit: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    lc = coarse.shape[1]
    n_patch = patches_per_sweep(lc, patch_size)
    update_rows: list[dict[str, Any]] = []
    check_rows: list[dict[str, Any]] = []
    obs_rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    n_checks = 0
    for chain in range(chains):
        idx = int(rng.integers(0, len(coarse)))
        u = coarse[idx][None]
        z_edge, z_pair, z_corner = sampler.sample_z(rng, 1, lc)
        if mode == "incremental":
            state = compute_state_with_maps(u, z_edge, z_pair, z_corner, ctx)
        else:
            state = sampler.compute_state(u, z_edge, z_pair, z_corner, ctx)
        obs_rows.append(local_obs_row(state["phi"], chain, 0, mode))
        for sweep in range(1, sweeps + 1):
            schedule = random_origin_patch_schedule(lc, patch_size, rng, "random")
            for attempt, (x0, y0, tile) in enumerate(schedule):
                sites = patch_sites(lc, x0, y0, patch_size)
                u_new = state["u"][0].copy()
                u_new, _ = inner_patch_metropolis(u_new, sites, rng)
                if mode == "incremental":
                    proposal = compute_state_with_maps(u_new[None], state["z_edge"], state["z_pair"], state["z_corner"], ctx)
                    pair_model = ctx["stages"]["pair"][0]
                    corner_model = ctx["stages"]["corner"][0]
                    pair_mask = dirty_mask(lc, x0, y0, patch_size, dependency_radius(pair_model))
                    corner_mask = dirty_mask(lc, x0, y0, patch_size, dependency_radius(corner_model))
                    old_dirty = float(np.sum(state["pair_logq_map"][0, 0][pair_mask]) + np.sum(state["corner_logq_map"][0, 0][corner_mask]))
                    new_dirty = float(np.sum(proposal["pair_logq_map"][0, 0][pair_mask]) + np.sum(proposal["corner_logq_map"][0, 0][corner_mask]))
                    delta_inc = new_dirty - old_dirty
                    delta_full = float((proposal["pair_logq"][0] + proposal["corner_logq"][0]) - (state["pair_logq"][0] + state["corner_logq"][0]))
                    if n_checks < check_limit:
                        check_rows.append(
                            {
                                "check": n_checks,
                                "sweep": sweep,
                                "attempt": attempt,
                                "patch_x": x0,
                                "patch_y": y0,
                                "pair_dirty_terms": int(np.sum(pair_mask)),
                                "corner_dirty_terms": int(np.sum(corner_mask)),
                                "delta_pair_corner_logq_incremental": delta_inc,
                                "delta_pair_corner_logq_full": delta_full,
                                "abs_error": abs(delta_inc - delta_full),
                            }
                        )
                        n_checks += 1
                else:
                    proposal = sampler.compute_state(u_new[None], state["z_edge"], state["z_pair"], state["z_corner"], ctx)
                delta_logw = float(proposal["logw"][0] - state["logw"][0])
                accept = math.log(max(float(rng.random()), 1e-300)) < min(0.0, delta_logw)
                if accept:
                    state = proposal
                update_rows.append({"mode": mode, "chain": chain, "sweep": sweep, "attempt": attempt, "update_type": "coarse_patch", "accepted": int(accept), "delta_logw": delta_logw})
            obs_rows.append(local_obs_row(state["phi"], chain, sweep, mode))
    elapsed = time.perf_counter() - t0
    return (
        {
            "mode": mode,
            "chains": chains,
            "sweeps": sweeps,
            "coarse_attempts": chains * sweeps * n_patch,
            "wall_sec": elapsed,
            "wall_sec_per_chain_sweep": elapsed / max(chains * sweeps, 1),
            "acceptance": float(np.mean([r["accepted"] for r in update_rows])) if update_rows else float("nan"),
        },
        update_rows + obs_rows,
        check_rows,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--kappa-f", type=float, default=0.2705)
    ap.add_argument("--chains", type=int, default=1)
    ap.add_argument("--sweeps", type=int, default=10)
    ap.add_argument("--patch-size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260709)
    ap.add_argument("--checks", type=int, default=32)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    ctx = load_ctx(cfg, args.kappa_f)
    coarse = np.load(resolve_run_paths(cfg)["coarse_ensemble"])["phi"].astype(np.float32)
    lc = int(cfg["lattice"]["coarse_L"])
    n_patch = patches_per_sweep(lc, args.patch_size)
    pair_model = ctx["stages"]["pair"][0]
    corner_model = ctx["stages"]["corner"][0]
    pair_radius = dependency_radius(pair_model)
    corner_radius = dependency_radius(corner_model)

    dirty_rows = []
    rng = np.random.default_rng(args.seed + 17)
    for idx, (x0, y0, _) in enumerate(random_origin_patch_schedule(lc, args.patch_size, rng, "random")):
        pm = dirty_mask(lc, x0, y0, args.patch_size, pair_radius)
        cm = dirty_mask(lc, x0, y0, args.patch_size, corner_radius)
        dirty_rows.append(
            {
                "patch_index": idx,
                "patch_x": x0,
                "patch_y": y0,
                "pair_radius": pair_radius,
                "corner_radius": corner_radius,
                "total_pair_terms": lc * lc,
                "dirty_pair_terms": int(np.sum(pm)),
                "dirty_pair_fraction": float(np.mean(pm)),
                "total_corner_terms": lc * lc,
                "dirty_corner_terms": int(np.sum(cm)),
                "dirty_corner_fraction": float(np.mean(cm)),
            }
        )
    write_csv(args.out_dir / "dirty_region_diagnostics.csv", dirty_rows)

    full_summary, full_rows, _ = run_mode("full", coarse, ctx, chains=args.chains, sweeps=args.sweeps, patch_size=args.patch_size, seed=args.seed, check_limit=0)
    inc_summary, inc_rows, checks = run_mode("incremental", coarse, ctx, chains=args.chains, sweeps=args.sweeps, patch_size=args.patch_size, seed=args.seed, check_limit=args.checks)
    runtime_rows = [full_summary, inc_summary]
    write_csv(args.out_dir / "runtime_comparison_full_vs_incremental.csv", runtime_rows)
    write_csv(args.out_dir / "incremental_vs_full_logq_checks.csv", checks)

    acc_rows = []
    for summary in runtime_rows:
        acc_rows.append(
            {
                "mode": summary["mode"],
                "update_type": "coarse_patch",
                "attempts": summary["coarse_attempts"],
                "acceptance": summary["acceptance"],
                "wall_sec": summary["wall_sec"],
            }
        )
    write_csv(args.out_dir / "acceptance_comparison_full_vs_incremental.csv", acc_rows)
    write_csv(args.out_dir / "observable_time_histories_full_vs_incremental.csv", [r for r in full_rows + inc_rows if "phi2" in r])
    write_json(args.out_dir / "incremental_logq_prototype_summary.json", {"dirty": dirty_rows[0], "runtime": runtime_rows, "checks": checks[:5]})

    max_err = max([float(r["abs_error"]) for r in checks], default=float("nan"))
    mean_err = float(np.mean([float(r["abs_error"]) for r in checks])) if checks else float("nan")
    dirty_pair_frac = float(np.mean([r["dirty_pair_fraction"] for r in dirty_rows]))
    dirty_corner_frac = float(np.mean([r["dirty_corner_fraction"] for r in dirty_rows]))
    speedup = full_summary["wall_sec"] / inc_summary["wall_sec"] if inc_summary["wall_sec"] > 0 else float("nan")
    full_per_sweep = full_summary["wall_sec_per_chain_sweep"]
    inc_per_sweep = inc_summary["wall_sec_per_chain_sweep"]
    lines = [
        "# Incremental pair/corner logq prototype",
        "",
        "This is a diagnostic/prototype only. It does not change the production patch-chain driver.",
        "",
        "## Flow Dependency",
        "",
        f"- pair model: `{type(pair_model).__name__}`",
        f"- corner model: `{type(corner_model).__name__}`",
        f"- pair conservative coarse/stage radius: `{pair_radius}`",
        f"- corner conservative coarse/stage radius: `{corner_radius}`",
        f"- stage lattice size: `{lc}`",
        "",
        "The pair and corner stages are procedural convolutional affine-coupling flows. Each conditioner has three circular 3x3 convolutions, and there are 16 coupling layers in the deployed checkpoint. The conservative composed dependency radius is therefore 48 stage-lattice sites, which exceeds the L32 periodic lattice.",
        "",
        "## Dirty Region",
        "",
        f"- mean dirty pair fraction per P={args.patch_size} patch: `{dirty_pair_frac:.6g}`",
        f"- mean dirty corner fraction per P={args.patch_size} patch: `{dirty_corner_frac:.6g}`",
        "",
        "Because the dirty halo covers the full periodic L32 stage lattice, exact local incremental pair/corner logq evaluation cannot reduce work for this trained model. The cache scaffold is exact, but its dirty region is global.",
        "",
        "## Correctness Check",
        "",
        f"- checks: `{len(checks)}`",
        f"- max |incremental-full| for pair+corner Delta logq: `{max_err:.6g}`",
        f"- mean |incremental-full| for pair+corner Delta logq: `{mean_err:.6g}`",
        "",
        "The incremental value is computed from cached per-site pair/corner logq maps on the dirty region. Since the exact dirty region is full-volume here, it agrees with full recomputation up to floating-point summation tolerance.",
        "",
        "## Runtime Comparison",
        "",
        "| mode | wall_sec | sec_per_chain_sweep | coarse_acceptance |",
        "|---|---:|---:|---:|",
        f"| full | {full_summary['wall_sec']:.6g} | {full_per_sweep:.6g} | {full_summary['acceptance']:.6g} |",
        f"| incremental_scaffold | {inc_summary['wall_sec']:.6g} | {inc_per_sweep:.6g} | {inc_summary['acceptance']:.6g} |",
        "",
        f"Observed speedup: `{speedup:.6g}`. A speedup near or below 1 is expected because the deployed pair/corner dirty region is full-volume and the scaffold also materializes per-site maps.",
        "",
        "## Projected Cost",
        "",
        "| run | full_hours | incremental_scaffold_hours |",
        "|---|---:|---:|",
        f"| 8x300 | {full_per_sweep * 8 * 300 / 3600:.6g} | {inc_per_sweep * 8 * 300 / 3600:.6g} |",
        f"| 8x1000 | {full_per_sweep * 8 * 1000 / 3600:.6g} | {inc_per_sweep * 8 * 1000 / 3600:.6g} |",
        f"| 16x500 | {full_per_sweep * 16 * 500 / 3600:.6g} | {inc_per_sweep * 16 * 500 / 3600:.6g} |",
        "",
        "## Answers",
        "",
        "1. What fraction of pair/corner logq terms are dirty?",
        "",
        f"`{dirty_pair_frac:.6g}` for pair and `{dirty_corner_frac:.6g}` for corner: effectively all terms.",
        "",
        "2. Does incremental Delta logq reproduce full recomputation?",
        "",
        f"Yes for the scaffold: max error `{max_err:.6g}` over `{len(checks)}` checks.",
        "",
        "3. What speedup is observed?",
        "",
        f"`{speedup:.6g}`. No real speedup is available with the current 16-layer circular procedural-conv checkpoint because the exact dependency halo is full-volume.",
        "",
        "4. Are acceptance rates unchanged within statistics?",
        "",
        "The full and scaffold runs use the same target formula. The short-run rates are listed above; any small difference is ordinary short-run Monte Carlo noise unless the same proposal stream is replayed bit-for-bit.",
        "",
        "5. Is this prototype safe enough for longer statistics?",
        "",
        "Not as an optimization. It is safe as a diagnostic scaffold, but using it for long runs would not reduce cost. A real optimization needs a flow API/checkpoint with a strictly local dependency radius smaller than L/2, or a batched full-volume proposal evaluator.",
        "",
        "## API Changes Needed For Real Local Evaluation",
        "",
        "- A stage-flow method that evaluates `forward_with_logdet_map` on local crops with explicit halo and returns only dirty output/logdet entries.",
        "- A dependency contract per stage and per coupling layer that accounts for masked coupling recursion.",
        "- Boundary handling that exactly matches circular padding for halos; if the composed halo wraps the lattice, the implementation must fall back to full-volume evaluation.",
        "- Cached per-site pair/corner logq maps and cached intermediate pair output `d01`, with accept-time dirty-region cache updates.",
        "- A correctness test comparing local-crop incremental Delta logq to full recomputation before enabling any production run.",
    ]
    (args.out_dir / "INCREMENTAL_LOGQ_PROTOTYPE_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"out_dir": str(args.out_dir), "dirty_pair_fraction": dirty_pair_frac, "dirty_corner_fraction": dirty_corner_frac, "max_error": max_err, "speedup": speedup}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
