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
sys.path.insert(0, str(PKG / "scripts"))
sys.path.insert(0, str(PKG / "src"))
sys.path.insert(0, str(FINITE / "scripts"))
sys.path.insert(0, str(FROZEN / "scripts"))

import run_shape_parametric_sampler_validation as sampler  # noqa: E402
from _common import load_config, load_frozen_models, load_kernel_spec, resolve_run_paths  # noqa: E402
from perfect_blocking_upsampling.actions import ActionSpec, action_total  # noqa: E402
from train_finite_footprint_transported_detail import patch_sites  # noqa: E402

DEFAULT_CONFIG = PKG / "outputs" / "shape_parametric_sampler_validation" / "L16_to_L32_smoke" / "L16_to_L32_smoke_config.yaml"
DEFAULT_OUT = PKG / "outputs" / "shape_parametric_sampler_validation" / "controlled_coarse_patch_restart_16to32_pass_scan"
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


def load_ctx(cfg: dict[str, Any]) -> dict[str, Any]:
    refine_model, refine_state, stages, coarse_action, fine_action, _ = load_frozen_models(cfg)
    refine_model.load_state_dict(refine_state, strict=False)
    refine_model.eval()
    for model, _, state, *_ in stages.values():
        model.load_state_dict(state)
        model.eval()
    kernel, _ = load_kernel_spec(cfg)
    return {"refine_model": refine_model, "stages": stages, "coarse_action": coarse_action, "fine_action": fine_action, "kernel": kernel}


def local_delta_action_site(field: np.ndarray, i: int, j: int, new_x: float, action: ActionSpec) -> float:
    old_x = float(field[i, j])
    lam = float(action.lambda_)
    kap = float(action.kappa)
    onsite_old = (1.0 - 2.0 * lam) * old_x * old_x + lam * old_x**4
    onsite_new = (1.0 - 2.0 * lam) * new_x * new_x + lam * new_x**4
    nn = float(field[(i + 1) % field.shape[0], j] + field[(i - 1) % field.shape[0], j] + field[i, (j + 1) % field.shape[1]] + field[i, (j - 1) % field.shape[1]])
    return (onsite_new - onsite_old) - 2.0 * kap * (new_x - old_x) * nn


def controlled_patch_metropolis(
    u: np.ndarray,
    sites: list[tuple[int, int]],
    rng: np.random.Generator,
    action: ActionSpec,
    step_size: float,
    sweeps: int = 1,
) -> tuple[np.ndarray, dict[str, Any]]:
    current = u.copy()
    before = current.copy()
    sc_before = float(action_total(current, action))
    rows = []
    attempted = 0
    accepted = 0
    for _ in range(sweeps):
        order = list(sites)
        rng.shuffle(order)
        for site_index, (i, j) in enumerate(order):
            old = float(current[i, j])
            new = old + float(step_size * rng.standard_normal())
            dsc = local_delta_action_site(current, i, j, new, action)
            acc = math.log(max(float(rng.random()), 1.0e-300)) < min(0.0, -dsc)
            attempted += 1
            if acc:
                current[i, j] = new
                accepted += 1
            rows.append({"site_index": site_index, "site_i": i, "site_j": j, "old_phi": old, "new_phi_proposed": new, "delta_Sc_site": dsc, "accepted_site": int(acc)})
    sc_after = float(action_total(current, action))
    patch_delta = current - before
    stats = {
        "attempted_site_updates": attempted,
        "accepted_site_updates": accepted,
        "coarse_site_acceptance": accepted / max(attempted, 1),
        "delta_Sc_patch_local_sum": float(sum(float(r["delta_Sc_site"]) for r in rows if r["accepted_site"])),
        "delta_Sc_patch_exact": sc_after - sc_before,
        "patch_l2_change": float(np.sqrt(np.sum(patch_delta * patch_delta))),
        "patch_norm2_change": float(np.sum(patch_delta * patch_delta)),
        "patch_linf_change": float(np.max(np.abs(patch_delta))) if patch_delta.size else 0.0,
        "site_rows": rows,
    }
    return current.astype(np.float32), stats


def local_observables(phi: np.ndarray, action: ActionSpec) -> dict[str, float]:
    arr = np.asarray(phi, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr[None]
    phi2 = np.mean(arr * arr, axis=(1, 2))
    phi4 = np.mean(arr**4, axis=(1, 2))
    nn = 0.5 * (np.mean(arr * np.roll(arr, -1, axis=1), axis=(1, 2)) + np.mean(arr * np.roll(arr, -1, axis=2), axis=(1, 2)))
    twonn = 0.5 * (np.mean(arr * np.roll(arr, -2, axis=1), axis=(1, 2)) + np.mean(arr * np.roll(arr, -2, axis=2), axis=(1, 2)))
    diag = np.mean(arr * np.roll(np.roll(arr, -1, axis=1), -1, axis=2), axis=(1, 2))
    ad = action_total(arr, action) / (arr.shape[1] * arr.shape[2])
    return {"phi2": float(np.mean(phi2)), "phi4": float(np.mean(phi4)), "NN": float(np.mean(nn)), "2nn": float(np.mean(twonn)), "diag": float(np.mean(diag)), "action_density": float(np.mean(ad))}


def qstats(vals: list[float] | np.ndarray) -> dict[str, float]:
    arr = np.asarray(vals, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "q05": float("nan"), "q50": float("nan"), "q95": float("nan")}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "q05": float(np.quantile(arr, 0.05)),
        "q50": float(np.quantile(arr, 0.50)),
        "q95": float(np.quantile(arr, 0.95)),
    }


def tune_step(coarse: np.ndarray, action: ActionSpec, patch_size: int, rng: np.random.Generator, n_trials: int = 16) -> dict[str, float]:
    candidates = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6]
    rows = []
    lc = coarse.shape[1]
    for step in candidates:
        accs = []
        for _ in range(n_trials):
            cfg = coarse[int(rng.integers(0, len(coarse)))].copy()
            x0 = int(rng.integers(0, lc))
            y0 = int(rng.integers(0, lc))
            _, stats = controlled_patch_metropolis(cfg, patch_sites(lc, x0, y0, patch_size), rng, action, step)
            accs.append(stats["coarse_site_acceptance"])
        rows.append({"step_size": step, "mean_acceptance": float(np.mean(accs)), "distance_to_0p5": abs(float(np.mean(accs)) - 0.5)})
    best = min(rows, key=lambda r: r["distance_to_0p5"])
    return {"step_size": float(best["step_size"]), "pilot_acceptance": float(best["mean_acceptance"]), "pilot_rows": rows}


def run(args: argparse.Namespace) -> dict[str, Any]:
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    ctx = load_ctx(cfg)
    paths = resolve_run_paths(cfg)
    coarse = np.load(paths["coarse_ensemble"])["phi"].astype(np.float32)
    coarse_manifest_path = paths["coarse_ensemble"].with_name("manifest.json")
    coarse_manifest = json.loads(coarse_manifest_path.read_text()) if coarse_manifest_path.exists() else {}
    rng = np.random.default_rng(args.seed)
    lc = int(cfg["lattice"]["coarse_L"])
    lf = int(cfg["lattice"]["fine_L"])
    patch_rows: list[dict[str, Any]] = []
    fine_rows: list[dict[str, Any]] = []
    local_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for patch_size in args.patch_sizes:
        for n_passes in args.coarse_passes:
            step = float(args.site_step_size)
            for cfg_idx in range(args.n_configs):
                source_idx = int(rng.integers(0, len(coarse)))
                u_current = coarse[source_idx : source_idx + 1].copy()
                z_edge, z_pair, z_corner = sampler.sample_z(rng, 1, lc)
                state = sampler.compute_state(u_current, z_edge, z_pair, z_corner, ctx)
                for proposal_idx in range(args.patches_per_config):
                    x0 = int(rng.integers(0, lc))
                    y0 = int(rng.integers(0, lc))
                    sites = patch_sites(lc, x0, y0, patch_size)
                    u_prop_2d, cstats = controlled_patch_metropolis(state["u"][0], sites, rng, ctx["coarse_action"], step, sweeps=n_passes)
                    proposal = sampler.compute_state(u_prop_2d[None], state["z_edge"], state["z_pair"], state["z_corner"], ctx)
                    delta_logw = float(proposal["logw"][0] - state["logw"][0])
                    accepted_fine = math.log(max(float(rng.random()), 1.0e-300)) < min(0.0, delta_logw)
                    before_obs = local_observables(state["phi"], ctx["fine_action"])
                    after_obs = local_observables(proposal["phi"], ctx["fine_action"])
                    row_base = {
                        "L_c": lc,
                        "L_f": lf,
                        "kappa_c": float(ctx["coarse_action"].kappa),
                        "kappa_f": float(ctx["fine_action"].kappa),
                        "patch_size": patch_size,
                        "n_coarse_passes": n_passes,
                        "coarse_proposal_mode": "metropolis",
                        "coarse_site_step_size": step,
                        "config_id": cfg_idx,
                        "source_coarse_index": source_idx,
                        "patch_id": ((patch_size * 1000000) + (n_passes * 100000) + cfg_idx * args.patches_per_config + proposal_idx),
                        "proposal_in_config": proposal_idx,
                        "patch_x": x0,
                        "patch_y": y0,
                    }
                    move_row = {
                        **row_base,
                        "sites_per_patch": patch_size * patch_size,
                        "attempted_site_updates": cstats["attempted_site_updates"],
                        "accepted_site_updates": cstats["accepted_site_updates"],
                        "coarse_site_acceptance": cstats["coarse_site_acceptance"],
                        "accepted_site_updates_per_site": cstats["accepted_site_updates"] / max(patch_size * patch_size, 1),
                        "delta_Sc_patch": cstats["delta_Sc_patch_exact"],
                        "delta_Sc_patch_local_sum": cstats["delta_Sc_patch_local_sum"],
                        "patch_l2_change": cstats["patch_l2_change"],
                        "patch_norm2_change": cstats["patch_norm2_change"],
                        "patch_linf_change": cstats["patch_linf_change"],
                    }
                    patch_rows.append(move_row)
                    fine_row = {
                        **move_row,
                        "fine_log_accept": delta_logw,
                        "fine_AR_accept": int(accepted_fine),
                        "delta_Sf": float(proposal["sf"][0] - state["sf"][0]),
                        "delta_logdet_refine": float(proposal["logdet"][0] - state["logdet"][0]),
                        "delta_logq_detail": float(proposal["logq"][0] - state["logq"][0]),
                        "phi2_before": before_obs["phi2"],
                        "phi2_after": after_obs["phi2"],
                        "phi4_before": before_obs["phi4"],
                        "phi4_after": after_obs["phi4"],
                        "NN_before": before_obs["NN"],
                        "NN_after": after_obs["NN"],
                        "2nn_before": before_obs["2nn"],
                        "2nn_after": after_obs["2nn"],
                        "diag_before": before_obs["diag"],
                        "diag_after": after_obs["diag"],
                    }
                    fine_rows.append(fine_row)
                    lrow = {**row_base, "fine_AR_accept": int(accepted_fine)}
                    for key in LOCAL_KEYS:
                        lrow[f"{key}_before"] = before_obs[key]
                        lrow[f"{key}_after"] = after_obs[key]
                        lrow[f"delta_{key}"] = after_obs[key] - before_obs[key]
                    local_rows.append(lrow)
                    if accepted_fine:
                        state = proposal
            sub = [r for r in fine_rows if int(r["patch_size"]) == patch_size and int(r["n_coarse_passes"]) == n_passes]
            site_attempts = int(sum(int(r["attempted_site_updates"]) for r in sub))
            site_accepts = int(sum(int(r["accepted_site_updates"]) for r in sub))
            fine_acc = [r["fine_AR_accept"] for r in sub]
            dsc = [r["delta_Sc_patch"] for r in sub]
            dlogw = [r["fine_log_accept"] for r in sub]
            l2 = [r["patch_l2_change"] for r in sub]
            norm2 = [r["patch_norm2_change"] for r in sub]
            acc_site = [r["accepted_site_updates"] for r in sub]
            summary_rows.append(
                {
                    "L_c": lc,
                    "L_f": lf,
                    "kappa_c": float(ctx["coarse_action"].kappa),
                    "kappa_f": float(ctx["fine_action"].kappa),
                    "patch_size": patch_size,
                    "sites_per_patch": patch_size * patch_size,
                    "n_coarse_passes": n_passes,
                    "coarse_proposal_mode": "metropolis",
                    "coarse_site_step_size": step,
                    "n_source_configs": args.n_configs,
                    "n_patch_proposals": len(sub),
                    "site_attempts_total": site_attempts,
                    "site_accepts_total": site_accepts,
                    "site_acceptance": site_accepts / max(site_attempts, 1),
                    "accepted_site_updates_per_patch_mean": float(np.mean(acc_site)),
                    "accepted_site_updates_per_patch_std": float(np.std(acc_site, ddof=1)),
                    "accepted_site_updates_per_site_mean": float(np.mean(acc_site) / max(patch_size * patch_size, 1)),
                    "patch_delta_Sc_mean": qstats(dsc)["mean"],
                    "patch_delta_Sc_std": qstats(dsc)["std"],
                    "patch_delta_Sc_q05": qstats(dsc)["q05"],
                    "patch_delta_Sc_q50": qstats(dsc)["q50"],
                    "patch_delta_Sc_q95": qstats(dsc)["q95"],
                    "patch_l2_change_mean": float(np.mean(l2)),
                    "patch_l2_change_std": float(np.std(l2, ddof=1)),
                    "patch_l2_change_q05": float(np.quantile(l2, 0.05)),
                    "patch_l2_change_q50": float(np.quantile(l2, 0.50)),
                    "patch_l2_change_q95": float(np.quantile(l2, 0.95)),
                    "patch_norm2_change_mean": float(np.mean(norm2)),
                    "patch_norm2_change_std": float(np.std(norm2, ddof=1)),
                    "fine_AR_attempts": len(sub),
                    "fine_AR_accepts": int(sum(fine_acc)),
                    "fine_AR_acceptance": float(np.mean(fine_acc)),
                    "fine_log_accept_mean": qstats(dlogw)["mean"],
                    "fine_log_accept_std": qstats(dlogw)["std"],
                    "fine_log_accept_q05": qstats(dlogw)["q05"],
                    "fine_log_accept_q50": qstats(dlogw)["q50"],
                    "fine_log_accept_q95": qstats(dlogw)["q95"],
                }
            )
    write_csv(out / "controlled_coarse_pass_scan_patch_moves.csv", patch_rows)
    write_csv(out / "fine_AR_after_controlled_pass_scan.csv", fine_rows)
    write_csv(out / "local_operator_changes_after_controlled_pass_scan.csv", local_rows)
    write_csv(out / "controlled_coarse_pass_scan_summary.csv", summary_rows)
    summary = {
        "status": "completed",
        "elapsed_sec": time.perf_counter() - t0,
        "config": str(args.config),
        "coarse_ensemble": str(paths["coarse_ensemble"]),
        "coarse_manifest": coarse_manifest,
        "n_configs": args.n_configs,
        "patches_per_config": args.patches_per_config,
        "patch_sizes": args.patch_sizes,
        "coarse_passes": args.coarse_passes,
        "site_step_size": args.site_step_size,
        "summary_rows": summary_rows,
    }
    write_json(out / "summary.json", summary)
    write_report(out, summary, summary_rows, local_rows)
    return summary


def write_report(out: Path, summary: dict[str, Any], summary_rows: list[dict[str, Any]], local_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Controlled coarse pass scan 16->32",
        "",
        "This diagnostic scans repeated controlled coarse Metropolis passes over the same selected patch. The site proposal step is fixed at `0.6`; every site proposal has its own Metropolis A/R targeting `S_c`; the fine/transformed A/R then uses the existing project logweight convention with fixed latent/detail field `z`.",
        "",
        "## Setup",
        "",
        f"- config: `{summary['config']}`",
        f"- coarse ensemble: `{summary['coarse_ensemble']}`",
        f"- n source configs: `{summary['n_configs']}`",
        f"- patches per config: `{summary['patches_per_config']}`",
        f"- coarse passes scanned: `{summary['coarse_passes']}`",
        f"- fixed site proposal step size: `{summary['site_step_size']}`",
        f"- elapsed sec: `{summary['elapsed_sec']:.6g}`",
        "",
        "## Main Table",
        "",
        "| P | passes | site attempts | site acc | accepted updates/patch | accepted updates/site | patch L2 mean | fine attempts | fine A/R acc | mean fine logacc | std fine logacc |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary_rows:
        lines.append(
            f"| {r['patch_size']} | {r['n_coarse_passes']} | {r['site_attempts_total']} | {r['site_acceptance']:.6g} | "
            f"{r['accepted_site_updates_per_patch_mean']:.6g} | {r['accepted_site_updates_per_site_mean']:.6g} | {r['patch_l2_change_mean']:.6g} | "
            f"{r['fine_AR_attempts']} | {r['fine_AR_acceptance']:.6g} | {r['fine_log_accept_mean']:.6g} | {r['fine_log_accept_std']:.6g} |"
        )
    lines += [
        "",
        "## Fine Log-Accept Quantiles",
        "",
        "| P | passes | q05 | q50 | q95 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for r in summary_rows:
        lines.append(f"| {r['patch_size']} | {r['n_coarse_passes']} | {r['fine_log_accept_q05']:.6g} | {r['fine_log_accept_q50']:.6g} | {r['fine_log_accept_q95']:.6g} |")
    lines += [
        "",
        "## Patch Displacement and Delta Sc",
        "",
        "| P | passes | dSc mean | dSc std | dSc q05 | dSc q50 | dSc q95 | patch L2 mean | patch L2 q50 | patch L2 q95 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary_rows:
        lines.append(f"| {r['patch_size']} | {r['n_coarse_passes']} | {r['patch_delta_Sc_mean']:.6g} | {r['patch_delta_Sc_std']:.6g} | {r['patch_delta_Sc_q05']:.6g} | {r['patch_delta_Sc_q50']:.6g} | {r['patch_delta_Sc_q95']:.6g} | {r['patch_l2_change_mean']:.6g} | {r['patch_l2_change_q50']:.6g} | {r['patch_l2_change_q95']:.6g} |")
    lines += [
        "",
        "## Local/Action-Sector Changes",
        "",
        "| P | passes | operator | mean delta | std delta |",
        "|---:|---:|---|---:|---:|",
    ]
    for patch_size in sorted({int(r["patch_size"]) for r in local_rows}):
        for n_passes in sorted({int(r["n_coarse_passes"]) for r in local_rows if int(r["patch_size"]) == patch_size}):
            sub = [r for r in local_rows if int(r["patch_size"]) == patch_size and int(r["n_coarse_passes"]) == n_passes]
            for key in LOCAL_KEYS:
                vals = np.asarray([float(r[f"delta_{key}"]) for r in sub], dtype=np.float64)
                lines.append(f"| {patch_size} | {n_passes} | {key} | {float(np.mean(vals)):.6g} | {float(np.std(vals, ddof=1)):.6g} |")
    lines += ["", "## Answers", ""]
    by_key = {(int(r["patch_size"]), int(r["n_coarse_passes"])): r for r in summary_rows}
    for patch_size in sorted({int(r["patch_size"]) for r in summary_rows}):
        base = by_key[(patch_size, min(summary["coarse_passes"]))]
        last = by_key[(patch_size, max(summary["coarse_passes"]))]
        disp_ratio = last["patch_l2_change_mean"] / max(base["patch_l2_change_mean"], 1e-300)
        ar_drop = base["fine_AR_acceptance"] - last["fine_AR_acceptance"]
        lines.append(f"- P={patch_size}: increasing passes from {min(summary['coarse_passes'])} to {max(summary['coarse_passes'])} changes mean patch L2 displacement from `{base['patch_l2_change_mean']:.6g}` to `{last['patch_l2_change_mean']:.6g}` (ratio `{disp_ratio:.6g}`), and fine A/R from `{base['fine_AR_acceptance']:.6g}` to `{last['fine_AR_acceptance']:.6g}` (drop `{ar_drop:.6g}`).")
    lines.append("- This scan keeps the site proposal step fixed at `0.6`; it tests multiple high-acceptance passes rather than forcing one pass to 50% site acceptance.")
    lines.append("- Heatbath was not implemented in this pass scan; the diagnostic uses site Metropolis only.")
    lines.append("")
    lines.append("Previous uncontrolled coarse-patch results should be treated only as diagnostics. This controlled coarse proposal is the relevant restart path.")
    (out / "CONTROLLED_COARSE_PASS_SCAN_16TO32_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--n-configs", type=int, default=16)
    ap.add_argument("--patches-per-config", type=int, default=8)
    ap.add_argument("--patch-sizes", type=int, nargs="+", default=[4, 6])
    ap.add_argument("--coarse-passes", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--site-step-size", type=float, default=0.6)
    ap.add_argument("--tune-trials", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20260702)
    args = ap.parse_args()
    summary = run(args)
    print(json.dumps({"out": str(args.out_dir), "status": summary["status"], "elapsed_sec": summary["elapsed_sec"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
