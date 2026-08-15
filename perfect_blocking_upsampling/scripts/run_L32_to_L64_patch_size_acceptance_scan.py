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
from perfect_blocking_upsampling.kernels import apply_kernel, inverse_kernel  # noqa: E402
from train_finite_footprint_transported_detail import (  # noqa: E402
    inner_patch_metropolis,
    patch_sites,
    patches_per_sweep,
    random_origin_patch_schedule,
)

DEFAULT_CONFIG = PKG / "outputs/shape_parametric_sampler_validation/L32_to_L64_smoke/L32_to_L64_smoke_config.yaml"
DEFAULT_OUT = PKG / "outputs/shape_parametric_sampler_validation/L32_to_L64_patch_size_acceptance_scan"


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


def stats(prefix: str, values: list[float]) -> dict[str, Any]:
    if not values:
        return {f"{prefix}_{k}": float("nan") for k in ["mean", "std", "q05", "q50", "q95", "min", "max"]}
    arr = np.asarray(values, dtype=np.float64)
    qs = np.quantile(arr, [0.05, 0.5, 0.95])
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        f"{prefix}_q05": float(qs[0]),
        f"{prefix}_q50": float(qs[1]),
        f"{prefix}_q95": float(qs[2]),
        f"{prefix}_min": float(np.min(arr)),
        f"{prefix}_max": float(np.max(arr)),
    }


def load_ctx(config: Path) -> tuple[dict[str, Any], np.ndarray, dict[str, Any]]:
    cfg = load_config(config)
    refine_model, refine_state, stages, coarse_action, fine_action, _ = load_frozen_models(cfg)
    refine_model.load_state_dict(refine_state, strict=False)
    refine_model.eval()
    for model, _, state, *_ in stages.values():
        model.load_state_dict(state)
        model.eval()
    kernel, _kernel_json = load_kernel_spec(cfg)
    ctx = {
        "refine_model": refine_model,
        "stages": stages,
        "coarse_action": coarse_action,
        "fine_action": fine_action,
        "kernel": kernel,
    }
    paths = resolve_run_paths(cfg)
    fine_path = paths["fine_reference"]
    if not fine_path.exists():
        raise FileNotFoundError(f"same-kappa blocked starts require direct L64 reference at {fine_path}")
    fine = np.load(fine_path)["phi"].astype(np.float32)
    if fine.shape[1:] != (64, 64):
        raise ValueError(f"expected L64 fine reference, got {fine.shape}")
    blocked = apply_kernel(fine, kernel)[:, 0::2, 0::2].astype(np.float32)
    return cfg, blocked, ctx


def compute_state_decomposed(u: np.ndarray, z_edge: np.ndarray, z_pair: np.ndarray, z_corner: np.ndarray, ctx: dict[str, Any]) -> dict[str, Any]:
    cprime, logdet = sampler.apply_refine_loaded(ctx["refine_model"], u)
    edge_model, edge_lg, _ = ctx["stages"]["edge"][:3]
    pair_model, pair_lg, _ = ctx["stages"]["pair"][:3]
    corner_model, corner_lg, _ = ctx["stages"]["corner"][:3]
    d10, l10 = sampler.stage_forward_from_z(edge_model, z_edge, cprime[:, None], edge_lg)
    d01, l01 = sampler.stage_forward_from_z(pair_model, z_pair, np.concatenate([cprime[:, None], d10], axis=1), pair_lg)
    d11, l11 = sampler.stage_forward_from_z(corner_model, z_corner, np.concatenate([cprime[:, None], d10, d01], axis=1), corner_lg)
    d = np.concatenate([d10, d01, d11], axis=1).astype(np.float32)
    psi = sampler.reconstruct(cprime, d)
    phi, inv = inverse_kernel(psi, ctx["kernel"])
    sf = sampler.action_total(phi, ctx["fine_action"])
    sc = sampler.action_total(u, ctx["coarse_action"])
    logq = l10 + l01 + l11
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
        "edge_logq": l10.astype(np.float64),
        "pair_logq": l01.astype(np.float64),
        "corner_logq": l11.astype(np.float64),
        "logq": logq.astype(np.float64),
        "logw": logw.astype(np.float64),
        "inv": inv,
    }


def propose_patch_decomposed(state: dict[str, Any], x0: int, y0: int, tile: int, rng: np.random.Generator, ctx: dict[str, Any], patch_size: int, scale_mult: float) -> tuple[dict[str, Any], dict[str, Any]]:
    sites = patch_sites(state["u"].shape[1], x0, y0, patch_size)
    u_new = state["u"][0].copy()
    t_inner = time.perf_counter()
    u_new, inner_acc = inner_patch_metropolis(u_new, sites, rng, sigma=0.1 * scale_mult)
    inner_sec = time.perf_counter() - t_inner
    t_state = time.perf_counter()
    prop = compute_state_decomposed(u_new[None], state["z_edge"], state["z_pair"], state["z_corner"], ctx)
    state_sec = time.perf_counter() - t_state
    d = {
        "patch_x": x0,
        "patch_y": y0,
        "tile": tile,
        "inner_acceptance": inner_acc,
        "inner_sigma": 0.1 * scale_mult,
        "delta_logw": float(prop["logw"][0] - state["logw"][0]),
        "delta_Sf": float(prop["sf"][0] - state["sf"][0]),
        "delta_Sc": float(prop["sc"][0] - state["sc"][0]),
        "delta_logdet_refine": float(prop["logdet"][0] - state["logdet"][0]),
        "delta_logq_edge": float(prop["edge_logq"][0] - state["edge_logq"][0]),
        "delta_logq_pair": float(prop["pair_logq"][0] - state["pair_logq"][0]),
        "delta_logq_corner": float(prop["corner_logq"][0] - state["corner_logq"][0]),
        "delta_logq_missing": float(prop["logq"][0] - state["logq"][0]),
        "inner_sec": inner_sec,
        "compute_state_sec": state_sec,
        "proposal_sec": inner_sec + state_sec,
    }
    d["component_balance_residual"] = float(d["delta_logw"] - (-d["delta_Sf"] + d["delta_Sc"] + d["delta_logdet_refine"] - d["delta_logq_missing"]))
    return prop, d


def propose_latent_decomposed(state: dict[str, Any], x0: int, y0: int, tile: int, rng: np.random.Generator, ctx: dict[str, Any], patch_size: int, rho: float) -> tuple[dict[str, Any], dict[str, Any]]:
    sites = patch_sites(state["u"].shape[1], x0, y0, patch_size)
    noise = math.sqrt(max(0.0, 1.0 - rho * rho))
    z_edge = state["z_edge"].copy()
    z_pair = state["z_pair"].copy()
    z_corner = state["z_corner"].copy()
    for i, j in sites:
        z_edge[0, 0, i, j] = rho * z_edge[0, 0, i, j] + noise * float(rng.standard_normal())
        z_pair[0, 0, i, j] = rho * z_pair[0, 0, i, j] + noise * float(rng.standard_normal())
        z_corner[0, 0, i, j] = rho * z_corner[0, 0, i, j] + noise * float(rng.standard_normal())
    t0 = time.perf_counter()
    prop = compute_state_decomposed(state["u"], z_edge, z_pair, z_corner, ctx)
    sec = time.perf_counter() - t0
    d = {
        "patch_x": x0,
        "patch_y": y0,
        "tile": tile,
        "delta_logw": float(prop["logw"][0] - state["logw"][0]),
        "delta_Sf": float(prop["sf"][0] - state["sf"][0]),
        "delta_Sc": 0.0,
        "delta_logdet_refine": 0.0,
        "delta_logq_edge": float(prop["edge_logq"][0] - state["edge_logq"][0]),
        "delta_logq_pair": float(prop["pair_logq"][0] - state["pair_logq"][0]),
        "delta_logq_corner": float(prop["corner_logq"][0] - state["corner_logq"][0]),
        "delta_logq_missing": float(prop["logq"][0] - state["logq"][0]),
        "proposal_sec": sec,
    }
    d["component_balance_residual"] = float(d["delta_logw"] - (-d["delta_Sf"] - d["delta_logq_missing"]))
    return prop, d


def accept(delta_logw: float, rng: np.random.Generator) -> bool:
    return math.log(max(rng.random(), 1.0e-300)) < min(0.0, float(delta_logw))


def run_case(case: dict[str, Any], coarse: np.ndarray, ctx: dict[str, Any], chains: int, sweeps: int, seed: int, pcn_rho: float) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    patch_size = int(case["patch_size"])
    scale_mult = float(case["scale_mult"])
    n_patch = patches_per_sweep(32, patch_size)
    coarse_rows: list[dict[str, Any]] = []
    latent_rows: list[dict[str, Any]] = []
    t_case = time.perf_counter()
    for chain in range(chains):
        rng = np.random.default_rng(seed + 10000 * chain + 97 * patch_size + int(round(1000 * scale_mult)))
        u0 = coarse[int(rng.integers(0, len(coarse)))][None]
        z_edge, z_pair, z_corner = sampler.sample_z(rng, 1, 32)
        state = compute_state_decomposed(u0, z_edge, z_pair, z_corner, ctx)
        for sweep in range(sweeps):
            schedule = random_origin_patch_schedule(32, patch_size, rng, "random")
            for attempt, (x0, y0, tile) in enumerate(schedule):
                prop, delta = propose_patch_decomposed(state, x0, y0, tile, rng, ctx, patch_size, scale_mult)
                a = accept(delta["delta_logw"], rng)
                if a:
                    state = prop
                coarse_rows.append({"case": case["label"], "move_type": "coarse", "chain_id": chain, "sweep": sweep, "attempt_in_sweep": attempt, "accepted": int(a), **delta})
            x0, y0, tile = schedule[-1]
            prop_l, delta_l = propose_latent_decomposed(state, x0, y0, tile, rng, ctx, patch_size, pcn_rho)
            a_l = accept(delta_l["delta_logw"], rng)
            if a_l:
                state = prop_l
            latent_rows.append({"case": case["label"], "move_type": "latent", "chain_id": chain, "sweep": sweep, "attempt_in_sweep": n_patch - 1, "accepted": int(a_l), **delta_l})
    wall = time.perf_counter() - t_case
    dlogw = [float(r["delta_logw"]) for r in coarse_rows]
    acc = [int(r["accepted"]) for r in coarse_rows]
    lat_acc = [int(r["accepted"]) for r in latent_rows]
    summary = {
        "case": case["label"],
        "patch_size": patch_size,
        "scale_mult": scale_mult,
        "inner_sigma": 0.1 * scale_mult,
        "chains": chains,
        "sweeps": sweeps,
        "N_patch_per_sweep": n_patch,
        "coarse_attempts": len(coarse_rows),
        "coarse_accepts": int(sum(acc)),
        "coarse_acceptance": float(np.mean(acc)) if acc else float("nan"),
        "latent_attempts": len(latent_rows),
        "latent_acceptance": float(np.mean(lat_acc)) if lat_acc else float("nan"),
        "wall_sec": wall,
        "sec_per_coarse_proposal": float(np.mean([r["proposal_sec"] for r in coarse_rows])) if coarse_rows else float("nan"),
        "sec_per_latent_proposal": float(np.mean([r["proposal_sec"] for r in latent_rows])) if latent_rows else float("nan"),
        **stats("delta_logw", dlogw),
        **stats("delta_Sf", [float(r["delta_Sf"]) for r in coarse_rows]),
        **stats("delta_Sc", [float(r["delta_Sc"]) for r in coarse_rows]),
        **stats("delta_logdet_refine", [float(r["delta_logdet_refine"]) for r in coarse_rows]),
        **stats("delta_logq_edge", [float(r["delta_logq_edge"]) for r in coarse_rows]),
        **stats("delta_logq_pair", [float(r["delta_logq_pair"]) for r in coarse_rows]),
        **stats("delta_logq_corner", [float(r["delta_logq_corner"]) for r in coarse_rows]),
        **stats("delta_logq_missing", [float(r["delta_logq_missing"]) for r in coarse_rows]),
    }
    return summary, coarse_rows, latent_rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--chains", type=int, default=1)
    ap.add_argument("--sweeps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260721)
    ap.add_argument("--pcn-rho", type=float, default=0.5)
    args = ap.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    cfg, coarse, ctx = load_ctx(args.config)
    cases = [
        {"label": "P4_scale1", "patch_size": 4, "scale_mult": 1.0},
        {"label": "P6_scale1", "patch_size": 6, "scale_mult": 1.0},
        {"label": "P6_scale4over6", "patch_size": 6, "scale_mult": 4.0 / 6.0},
        {"label": "P8_scale4over8", "patch_size": 8, "scale_mult": 4.0 / 8.0},
    ]
    preflight = {
        "config": str(args.config),
        "coarse_start_source": "direct_L64_kappa0p2705_blocked_to_L32_small3_retained_sites",
        "coarse_shape": list(coarse.shape),
        "coarse_kappa": 0.2705,
        "fine_kappa": 0.2705,
        "chains": args.chains,
        "sweeps": args.sweeps,
        "cases": [
            {
                **c,
                "N_patch_per_sweep": patches_per_sweep(32, int(c["patch_size"])),
                "expected_coarse_attempts": args.chains * args.sweeps * patches_per_sweep(32, int(c["patch_size"])),
                "expected_latent_attempts": args.chains * args.sweeps,
            }
            for c in cases
        ],
    }
    write_json(out / "scheduler_preflight.json", preflight)
    summaries: list[dict[str, Any]] = []
    all_coarse: list[dict[str, Any]] = []
    all_latent: list[dict[str, Any]] = []
    for case in cases:
        print(f"running {case['label']} P={case['patch_size']} scale={case['scale_mult']}", flush=True)
        summary, coarse_rows, latent_rows = run_case(case, coarse, ctx, args.chains, args.sweeps, args.seed, args.pcn_rho)
        summaries.append(summary)
        all_coarse.extend(coarse_rows)
        all_latent.extend(latent_rows)
        write_csv(out / "patch_size_acceptance_summary.csv", summaries)
        write_csv(out / "patch_size_delta_logweight_components.csv", all_coarse + all_latent)
    runtime_rows = [
        {
            "case": s["case"],
            "patch_size": s["patch_size"],
            "scale_mult": s["scale_mult"],
            "wall_sec": s["wall_sec"],
            "coarse_attempts": s["coarse_attempts"],
            "sec_per_coarse_proposal": s["sec_per_coarse_proposal"],
            "sec_per_latent_proposal": s["sec_per_latent_proposal"],
        }
        for s in summaries
    ]
    write_csv(out / "patch_size_runtime_summary.csv", runtime_rows)
    p4 = next(s for s in summaries if s["case"] == "P4_scale1")
    p6 = next(s for s in summaries if s["case"] == "P6_scale1")
    naive = float(p4["coarse_acceptance"] ** (36.0 / 16.0)) if p4["coarse_acceptance"] >= 0 else float("nan")
    def dominant(s: dict[str, Any]) -> str:
        comps = {
            "delta_Sf": abs(float(s["delta_Sf_std"])),
            "delta_Sc": abs(float(s["delta_Sc_std"])),
            "delta_logdet_refine": abs(float(s["delta_logdet_refine_std"])),
            "delta_logq_edge": abs(float(s["delta_logq_edge_std"])),
            "delta_logq_pair": abs(float(s["delta_logq_pair_std"])),
            "delta_logq_corner": abs(float(s["delta_logq_corner_std"])),
        }
        return max(comps, key=comps.get)
    lines = [
        "# L32->L64 patch-size acceptance scan",
        "",
        "Short same-kappa diagnostic using `kappa_c=kappa_f=0.2705`. No long validation was launched.",
        "",
        "No native L32 `kappa=0.2705` ensemble was found, so coarse starts are direct L64 `kappa=0.2705` configurations blocked to L32 through the small3 kernel. This avoids silently using the existing native L32 `kappa=0.271` ensemble.",
        "",
        f"- chains: `{args.chains}`",
        f"- sweeps: `{args.sweeps}`",
        f"- P=4 acceptance: `{p4['coarse_acceptance']:.6g}`",
        f"- naive P=6 area-scaled expectation from P=4: `{naive:.6g}`",
        f"- observed P=6 original-scale acceptance: `{p6['coarse_acceptance']:.6g}`",
        f"- dominant P=6 original-scale component by std: `{dominant(p6)}`",
        "",
        "## Summary",
        "",
        "| case | P | scale | attempts | acceptance | std dlogw | std dSf | std dSc | std dlogdet | std dq edge | std dq pair | std dq corner | latent acc | sec/proposal |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for s in summaries:
        lines.append(
            f"| `{s['case']}` | {s['patch_size']} | {s['scale_mult']:.4g} | {s['coarse_attempts']} | {s['coarse_acceptance']:.6g} | {s['delta_logw_std']:.6g} | {s['delta_Sf_std']:.6g} | {s['delta_Sc_std']:.6g} | {s['delta_logdet_refine_std']:.6g} | {s['delta_logq_edge_std']:.6g} | {s['delta_logq_pair_std']:.6g} | {s['delta_logq_corner_std']:.6g} | {s['latent_acceptance']:.6g} | {s['sec_per_coarse_proposal']:.6g} |"
        )
    p6_scaled = next(s for s in summaries if s["case"] == "P6_scale4over6")
    lines += [
        "",
        "## Questions",
        "",
        f"1. Does P=6 follow naive area scaling? Observed `{p6['coarse_acceptance']:.6g}` versus naive `{naive:.6g}`.",
        f"2. Dominant P=6 original-scale component by standard deviation: `{dominant(p6)}`.",
        f"3. Scale retuning P=6 by 4/6 gives acceptance `{p6_scaled['coarse_acceptance']:.6g}`.",
        "4. Pair/corner sensitivity should be judged from the component CSV and table above; this run records them separately.",
        "5. P=6 viability is a short-run diagnostic conclusion only; use a longer manual run if this table shows acceptable A/R.",
        "6. A better bounded-radius pair/corner flow would matter if pair/corner logq dominates the P-scaling; otherwise the bottleneck is coarse/fine action scale.",
    ]
    (out / "PATCH_SIZE_ACCEPTANCE_SCAN.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"out": str(out), "summaries": summaries}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
