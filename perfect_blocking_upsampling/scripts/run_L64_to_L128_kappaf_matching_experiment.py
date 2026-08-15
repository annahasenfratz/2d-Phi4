#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import replace
from datetime import datetime
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
from _common import format_float_tag, load_config, load_frozen_models, load_kernel_spec, override_validation_config, resolve_run_paths  # noqa: E402
from perfect_blocking_upsampling.actions import ActionSpec, action_total  # noqa: E402
from train_finite_footprint_transported_detail import inner_patch_metropolis, patch_sites, patches_per_sweep, random_origin_patch_schedule  # noqa: E402
from run_L32_to_L64_kappaf_matching_experiment import split_batch_state, stack_state_field  # noqa: E402

DEFAULT_CONFIG = PKG / "outputs" / "shape_parametric_sampler_validation" / "L32_to_L64_smoke" / "L32_to_L64_smoke_config.yaml"
DEFAULT_OUT = None
LOCAL_KEYS = ["phi2", "phi4", "NN", "2nn", "diag", "action_density"]


def log_progress(message: str) -> None:
    stamp = datetime.now().isoformat(timespec="seconds")
    print(f"[{stamp}] {message}", flush=True)


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


def make_l64_config(base_cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    run_name = args.run_name or (
        f"lam{format_float_tag(args.lambda_)}_kappa{format_float_tag(args.kappa_c)}_"
        f"L{args.Lc}_to_L{args.Lf}_acceptance_smoke"
    )
    return override_validation_config(
        base_cfg,
        coarse_L=args.Lc,
        fine_L=args.Lf,
        lambda_c=args.lambda_,
        lambda_f=args.lambda_,
        kappa_c=args.kappa_c,
        kappa_f=args.kappa_f,
        coarse_ensemble=args.coarse_ensemble,
        fine_reference=args.fine_reference,
        run_name=run_name,
        output_dir=args.out_dir,
    )


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


def local_observable_row(phi: np.ndarray, action: ActionSpec) -> dict[str, float]:
    arr = np.asarray(phi, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr[None]
    phi2 = np.mean(arr * arr, axis=(1, 2))
    phi4 = np.mean(arr**4, axis=(1, 2))
    nn = 0.5 * (
        np.mean(arr * np.roll(arr, -1, axis=1), axis=(1, 2))
        + np.mean(arr * np.roll(arr, -1, axis=2), axis=(1, 2))
    )
    two_nn = 0.5 * (
        np.mean(arr * np.roll(arr, -2, axis=1), axis=(1, 2))
        + np.mean(arr * np.roll(arr, -2, axis=2), axis=(1, 2))
    )
    diag = np.mean(arr * np.roll(np.roll(arr, -1, axis=1), -1, axis=2), axis=(1, 2))
    action_density = action_total(arr, action) / (arr.shape[1] * arr.shape[2])
    return {
        "phi2": float(np.mean(phi2)),
        "phi4": float(np.mean(phi4)),
        "NN": float(np.mean(nn)),
        "2nn": float(np.mean(two_nn)),
        "diag": float(np.mean(diag)),
        "action_density": float(np.mean(action_density)),
    }


def quantile_row(prefix: str, vals: list[float]) -> dict[str, float]:
    arr = np.asarray(vals, dtype=np.float64)
    if arr.size == 0:
        return {f"{prefix}_mean": float("nan"), f"{prefix}_std": float("nan"), f"{prefix}_q05": float("nan"), f"{prefix}_q50": float("nan"), f"{prefix}_q95": float("nan")}
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        f"{prefix}_q05": float(np.quantile(arr, 0.05)),
        f"{prefix}_q50": float(np.quantile(arr, 0.50)),
        f"{prefix}_q95": float(np.quantile(arr, 0.95)),
    }


def preflight(cfg: dict[str, Any], ctx: dict[str, Any], coarse: np.ndarray, out_dir: Path) -> dict[str, Any]:
    report: dict[str, Any] = {"status": "started"}
    paths = resolve_run_paths(cfg)
    report["coarse_path"] = str(paths["coarse_ensemble"])
    report["fine_reference_path"] = str(paths["fine_reference"])
    report["coarse_exists"] = paths["coarse_ensemble"].exists()
    report["fine_reference_exists"] = paths["fine_reference"].exists()
    report["coarse_shape"] = list(coarse.shape)
    report["stage_dependency_reports"] = {}
    for name, bundle in ctx["stages"].items():
        model = bundle[0]
        report["stage_dependency_reports"][name] = model.dependency_report() if hasattr(model, "dependency_report") else {"model": type(model).__name__, "dependency_report": None}
    rng = np.random.default_rng(12345)
    u = coarse[int(rng.integers(0, len(coarse)))][None]
    z_edge, z_pair, z_corner = sampler.sample_z(rng, 1, int(cfg["lattice"]["coarse_L"]))
    state = sampler.compute_state(u, z_edge, z_pair, z_corner, ctx)
    report["initial_phi_shape"] = list(state["phi"].shape)
    report["initial_logw_finite"] = bool(np.isfinite(state["logw"]).all())
    report["initial_fine_action_finite"] = bool(np.isfinite(state["sf"]).all())
    report["status"] = "passed"
    write_json(out_dir / "preflight_summary.json", report)
    return report


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    log_progress("starting L64->L128 acceptance smoke")
    base_cfg = load_config(args.config)
    cfg = make_l64_config(base_cfg, args)
    ctx = load_ctx(cfg, args.kappa_f)
    coarse_path = resolve_run_paths(cfg)["coarse_ensemble"
    ]
    coarse = np.load(coarse_path)["phi"].astype(np.float32)
    preflight_report = preflight(cfg, ctx, coarse, out)
    if preflight_report["status"] != "passed":
        raise RuntimeError(f"preflight failed: {preflight_report}")
    log_progress(
        "preflight passed: "
        f"coarse_shape={preflight_report['coarse_shape']} "
        f"initial_phi_shape={preflight_report['initial_phi_shape']}"
    )

    rngs = [np.random.default_rng(args.seed + 10000 * chain + 777) for chain in range(args.chains)]
    init_indices = [int(rng.integers(0, len(coarse))) for rng in rngs]
    z_edges, z_pairs, z_corners = [], [], []
    for rng in rngs:
        ze, zp, zc = sampler.sample_z(rng, 1, args.Lc)
        z_edges.append(ze)
        z_pairs.append(zp)
        z_corners.append(zc)
    states_batch = sampler.compute_state(
        np.stack([coarse[i] for i in init_indices], axis=0).astype(np.float32),
        np.concatenate(z_edges, axis=0),
        np.concatenate(z_pairs, axis=0),
        np.concatenate(z_corners, axis=0),
        ctx,
    )
    states = [split_batch_state(states_batch, i) for i in range(args.chains)]
    save_sweeps = sorted(set(int(x) for x in args.save_sweeps))
    n_patch = patches_per_sweep(args.Lc, args.coarse_patch_size)
    beta = float(args.latent_beta_scale)
    rho = math.sqrt(max(0.0, 1.0 - beta * beta))

    obs_rows: list[dict[str, Any]] = []
    ar_rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for chain, state in enumerate(states):
        obs_rows.append({"chain_id": chain, "sweep": 0, "state_type": "initial_pre_ar", **local_observable_row(state["phi"], ctx["fine_action"])})

    for sweep in range(1, args.sweeps + 1):
        sweep_t0 = time.perf_counter()
        schedules = [random_origin_patch_schedule(args.Lc, args.coarse_patch_size, rng, "random") for rng in rngs]
        for attempt in range(n_patch):
            u_news = []
            inner_accs = []
            for state, rng, sched in zip(states, rngs, schedules):
                x0, y0, _ = sched[attempt]
                sites = patch_sites(args.Lc, x0, y0, args.coarse_patch_size)
                u_new = state["u"][0].copy()
                u_new, inner_acc = inner_patch_metropolis(u_new, sites, rng, sigma=args.coarse_proposal_scale)
                u_news.append(u_new[None])
                inner_accs.append(inner_acc)
            proposal_batch = sampler.compute_state(
                np.concatenate(u_news, axis=0),
                stack_state_field(states, "z_edge"),
                stack_state_field(states, "z_pair"),
                stack_state_field(states, "z_corner"),
                ctx,
            )
            proposals = [split_batch_state(proposal_batch, i) for i in range(args.chains)]
            for chain, (state, proposal, sched) in enumerate(zip(states, proposals, schedules)):
                x0, y0, tile = sched[attempt]
                delta = {
                    "delta_logw": float(proposal["logw"][0] - state["logw"][0]),
                    "delta_Sf": float(proposal["sf"][0] - state["sf"][0]),
                    "delta_Sc": float(proposal["sc"][0] - state["sc"][0]),
                    "delta_logdet_refine": float(proposal["logdet"][0] - state["logdet"][0]),
                    "delta_logq_missing": float(proposal["logq"][0] - state["logq"][0]),
                }
                accept = math.log(max(float(rngs[chain].random()), 1e-300)) < min(0.0, delta["delta_logw"])
                if accept:
                    states[chain] = proposal
                if sweep in save_sweeps:
                    ar_rows.append({"move_type": "coarse_patch", "chain_id": chain, "sweep": sweep, "attempt_in_sweep": attempt, "substep": 0, "patch_x": x0, "patch_y": y0, "tile": tile, "inner_acceptance": inner_accs[chain], "accepted": int(accept), **delta})

            for substep in range(1, args.latent_updates_per_coarse + 1):
                z_edges, z_pairs, z_corners = [], [], []
                latent_origins = []
                for state, rng, sched in zip(states, rngs, schedules):
                    cx, cy, _ = sched[attempt]
                    max_origin = args.Lc
                    # Center the larger latent patch on the same coarse-patch origin up to periodic wrap.
                    x0 = (cx - (args.latent_patch_size - args.coarse_patch_size) // 2) % max_origin
                    y0 = (cy - (args.latent_patch_size - args.coarse_patch_size) // 2) % max_origin
                    latent_origins.append((x0, y0, f"latent_{substep}_around_{cx}_{cy}"))
                    sites = patch_sites(args.Lc, x0, y0, args.latent_patch_size)
                    z_edge = state["z_edge"].copy()
                    z_pair = state["z_pair"].copy()
                    z_corner = state["z_corner"].copy()
                    for i, j in sites:
                        z_edge[0, 0, i, j] = rho * z_edge[0, 0, i, j] + beta * float(rng.standard_normal())
                        z_pair[0, 0, i, j] = rho * z_pair[0, 0, i, j] + beta * float(rng.standard_normal())
                        z_corner[0, 0, i, j] = rho * z_corner[0, 0, i, j] + beta * float(rng.standard_normal())
                    z_edges.append(z_edge)
                    z_pairs.append(z_pair)
                    z_corners.append(z_corner)
                proposal_batch = sampler.compute_state(
                    stack_state_field(states, "u"),
                    np.concatenate(z_edges, axis=0),
                    np.concatenate(z_pairs, axis=0),
                    np.concatenate(z_corners, axis=0),
                    ctx,
                )
                proposals = [split_batch_state(proposal_batch, i) for i in range(args.chains)]
                for chain, (state, proposal, origin) in enumerate(zip(states, proposals, latent_origins)):
                    x0, y0, tile = origin
                    delta = {
                        "delta_logw": float(proposal["logw"][0] - state["logw"][0]),
                        "delta_Sf": float(proposal["sf"][0] - state["sf"][0]),
                        "delta_Sc": 0.0,
                        "delta_logdet_refine": float(proposal["logdet"][0] - state["logdet"][0]),
                        "delta_logq_missing": float(proposal["logq"][0] - state["logq"][0]),
                    }
                    accept = math.log(max(float(rngs[chain].random()), 1e-300)) < min(0.0, delta["delta_logw"])
                    if accept:
                        states[chain] = proposal
                    if sweep in save_sweeps:
                        ar_rows.append({"move_type": "latent_pCN", "chain_id": chain, "sweep": sweep, "attempt_in_sweep": attempt, "substep": substep, "patch_x": x0, "patch_y": y0, "tile": tile, "latent_beta_scale": beta, "latent_rho": rho, "accepted": int(accept), **delta})
        if sweep in save_sweeps:
            for chain, state in enumerate(states):
                obs_rows.append({"chain_id": chain, "sweep": sweep, "state_type": "end_of_sweep", **local_observable_row(state["phi"], ctx["fine_action"])})
        if sweep in save_sweeps or sweep == args.sweeps:
            log_progress(f"completed sweep {sweep}/{args.sweeps} in {time.perf_counter() - sweep_t0:.3f} sec")

    elapsed = time.perf_counter() - t0
    write_csv(out / "patch_update_AR_rows.csv", ar_rows)
    write_csv(out / "local_action_observables_by_chain_sweep.csv", obs_rows)
    summary_rows = []
    for move_type in ["coarse_patch", "latent_pCN"]:
        sub = [r for r in ar_rows if r["move_type"] == move_type]
        if sub:
            row = {
                "move_type": move_type,
                "attempts_saved_sweeps": len(sub),
                "accepts_saved_sweeps": int(sum(int(r["accepted"]) for r in sub)),
                "acceptance_saved_sweeps": float(np.mean([int(r["accepted"]) for r in sub])),
                **quantile_row("delta_logw", [float(r["delta_logw"]) for r in sub]),
            }
            summary_rows.append(row)
    for substep in range(1, args.latent_updates_per_coarse + 1):
        sub = [r for r in ar_rows if r["move_type"] == "latent_pCN" and int(r["substep"]) == substep]
        if sub:
            summary_rows.append({"move_type": "latent_pCN_by_substep", "substep": substep, "attempts_saved_sweeps": len(sub), "accepts_saved_sweeps": int(sum(int(r["accepted"]) for r in sub)), "acceptance_saved_sweeps": float(np.mean([int(r["accepted"]) for r in sub])), **quantile_row("delta_logw", [float(r["delta_logw"]) for r in sub])})
    write_csv(out / "patch_update_acceptance_summary.csv", summary_rows)

    obs_summary = []
    for sweep in save_sweeps:
        sub = [r for r in obs_rows if int(r["sweep"]) == sweep]
        if sub:
            row: dict[str, Any] = {"sweep": sweep, "n_chains": len(sub)}
            for key in LOCAL_KEYS:
                vals = np.asarray([float(r[key]) for r in sub], dtype=np.float64)
                row[key] = float(np.mean(vals))
                row[key + "_se"] = float(np.std(vals, ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else 0.0
            obs_summary.append(row)
    write_csv(out / "local_action_observables_summary.csv", obs_summary)
    summary = {
        "status": "completed",
        "elapsed_sec": elapsed,
        "sec_per_chain_sweep": elapsed / max(args.chains * args.sweeps, 1),
        "chains": args.chains,
        "sweeps": args.sweeps,
        "save_sweeps": save_sweeps,
        "Lc": args.Lc,
        "Lf": args.Lf,
        "lambda": args.lambda_,
        "kappa_c": args.kappa_c,
        "kappa_f": args.kappa_f,
        "coarse_patch_size": args.coarse_patch_size,
        "latent_patch_size": args.latent_patch_size,
        "coarse_proposal_scale": args.coarse_proposal_scale,
        "latent_beta_scale": args.latent_beta_scale,
        "latent_rho": rho,
        "latent_updates_per_coarse": args.latent_updates_per_coarse,
        "N_coarse_patch_per_sweep": n_patch,
        "actual_total_coarse_attempts": args.chains * args.sweeps * n_patch,
        "actual_total_latent_attempts": args.chains * args.sweeps * n_patch * args.latent_updates_per_coarse,
        "preflight": preflight_report,
    }
    write_json(out / "summary.json", summary)
    report_name = f"L{summary['Lc']}_TO_L{summary['Lf']}_ACCEPTANCE_SMOKE_REPORT.md"
    write_report(out, summary, summary_rows, obs_summary, report_name)
    log_progress(f"completed L64->L128 acceptance smoke elapsed_sec={elapsed:.6g}")
    return summary


def write_report(out: Path, summary: dict[str, Any], ar_summary: list[dict[str, Any]], obs_summary: list[dict[str, Any]], report_name: str) -> None:
    lines = [
        f"# L{summary['Lc']}->L{summary['Lf']} acceptance smoke report",
        "",
        "Short acceptance smoke only; this is not a full validation.",
        "",
        "## Setup",
        "",
        f"- lambda: `{summary['lambda']}`",
        f"- Lc -> Lf: `{summary['Lc']} -> {summary['Lf']}`",
        f"- kappa_c, kappa_f: `{summary['kappa_c']}`, `{summary['kappa_f']}`",
        f"- chains x sweeps: `{summary['chains']} x {summary['sweeps']}`",
        f"- coarse patch size: `{summary['coarse_patch_size']}`",
        f"- latent patch size: `{summary['latent_patch_size']}`",
        f"- coarse proposal scale: `{summary['coarse_proposal_scale']}`",
        f"- latent beta scale/rho: `{summary['latent_beta_scale']}` / `{summary['latent_rho']:.6g}`",
        f"- latent updates per coarse: `{summary['latent_updates_per_coarse']}`",
        f"- N coarse patches per sweep: `{summary['N_coarse_patch_per_sweep']}`",
        f"- elapsed sec: `{summary['elapsed_sec']:.6g}`",
        f"- sec/chain-sweep: `{summary['sec_per_chain_sweep']:.6g}`",
        "",
        "## Preflight",
        "",
        f"- status: `{summary['preflight']['status']}`",
        f"- coarse source: `{summary['preflight']['coarse_path']}`",
        f"- coarse shape: `{summary['preflight']['coarse_shape']}`",
        f"- initial phi shape: `{summary['preflight']['initial_phi_shape']}`",
        f"- finite initial logweight: `{summary['preflight']['initial_logw_finite']}`",
        "",
        "## A/R Summary",
        "",
        "| move_type | substep | attempts(saved) | acceptance | dlogw mean | dlogw std | q05 | q50 | q95 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ar_summary:
        lines.append(
            f"| {row.get('move_type', '')} | {row.get('substep', '')} | {row.get('attempts_saved_sweeps', '')} | "
            f"{float(row.get('acceptance_saved_sweeps', float('nan'))):.6g} | {float(row.get('delta_logw_mean', float('nan'))):.6g} | "
            f"{float(row.get('delta_logw_std', float('nan'))):.6g} | {float(row.get('delta_logw_q05', float('nan'))):.6g} | "
            f"{float(row.get('delta_logw_q50', float('nan'))):.6g} | {float(row.get('delta_logw_q95', float('nan'))):.6g} |"
        )
    lines += ["", "## Local/Action Observables", "", "| sweep | phi2 | phi4 | NN | 2NN | diag | action_density |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for row in obs_summary:
        lines.append(f"| {row['sweep']} | {row['phi2']:.6g} | {row['phi4']:.6g} | {row['NN']:.6g} | {row['2nn']:.6g} | {row['diag']:.6g} | {row['action_density']:.6g} |")
    lines += [
        "",
        "## Decision",
        "",
        "Mechanical viability should be judged by finite preflight, nonzero/stable patch acceptance, no NaNs or shape errors, and tolerable sec/chain-sweep. This short smoke does not establish target correctness or equilibrium.",
    ]
    (out / report_name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--run-name", type=str, default=None)
    ap.add_argument("--lambda", dest="lambda_", type=float, default=0.022)
    ap.add_argument("--kappa-c", type=float, default=0.2705)
    ap.add_argument("--coarse-ensemble", type=Path, default=None)
    ap.add_argument("--fine-reference", type=Path, default=None)
    ap.add_argument("--kappa-f", type=float, default=0.2705)
    ap.add_argument("--Lc", type=int, default=64)
    ap.add_argument("--Lf", type=int, default=128)
    ap.add_argument("--chains", type=int, default=2)
    ap.add_argument("--sweeps", type=int, default=20)
    ap.add_argument("--save-sweeps", type=int, nargs="+", default=[0, 1, 2, 5, 10] + list(range(20, 1001, 20)))
    ap.add_argument("--coarse-patch-size", type=int, default=32)
    ap.add_argument("--latent-patch-size", type=int, default=12)
    ap.add_argument("--coarse-proposal-scale", type=float, default=0.125)
    ap.add_argument("--latent-beta-scale", type=float, default=0.4)
    ap.add_argument("--latent-updates-per-coarse", type=int, default=3)
    ap.add_argument("--batch-logq-across-chains", action="store_true")
    ap.add_argument("--seed", type=int, default=20260702)
    args = ap.parse_args()
    if args.out_dir is None:
        args.out_dir = PKG / "outputs" / "shape_parametric_sampler_validation" / (
            f"L{args.Lc}_to_L{args.Lf}_P{args.coarse_patch_size}_latentP{args.latent_patch_size}_"
            f"beta{int(round(args.latent_beta_scale * 100)):02d}_acceptance_smoke_"
            f"kc{format_float_tag(args.kappa_c)}_kf{format_float_tag(args.kappa_f)}"
        )
    if not args.batch_logq_across_chains:
        print("warning: this smoke driver always batches across chains internally; flag accepted for compatibility", file=sys.stderr)
    summary = run_smoke(args)
    print(json.dumps({"out": str(args.out_dir), "status": summary["status"], "elapsed_sec": summary["elapsed_sec"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
