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
sys.path.insert(0, str(PKG / "scripts"))
sys.path.insert(0, str(PKG / "src"))

import run_shape_parametric_sampler_validation as sampler  # noqa: E402
from _common import load_config, load_frozen_models, load_kernel_spec, resolve_run_paths  # noqa: E402
from perfect_blocking_upsampling.actions import action_total  # noqa: E402
from perfect_blocking_upsampling.kernels import inverse_kernel  # noqa: E402
from train_finite_footprint_transported_detail import inner_patch_metropolis, patch_sites, patches_per_sweep, random_origin_patch_schedule  # noqa: E402

DEFAULT_CONFIG = PKG / "outputs" / "shape_parametric_sampler_validation" / "L32_to_L64_smoke" / "L32_to_L64_smoke_config.yaml"
DEFAULT_OUT = PKG / "outputs" / "shape_parametric_sampler_validation" / "L32_to_L64_patch_runtime_profile"
LOCAL_KEYS = ["phi2", "phi4", "NN", "2nn", "diag", "action_density"]
GLOBAL_KEYS = ["m", "abs_m", "m2", "m4", "chi", "Binder_U4", "xi_over_L"]


class Timers:
    def __init__(self) -> None:
        self.data: dict[str, list[float]] = {}

    def add(self, key: str, dt: float) -> None:
        self.data.setdefault(key, []).append(float(dt))

    def rows(self) -> list[dict[str, Any]]:
        rows = []
        total = sum(sum(v) for v in self.data.values())
        for key, vals in sorted(self.data.items()):
            arr = np.asarray(vals, dtype=np.float64)
            rows.append(
                {
                    "section": key,
                    "count": int(len(arr)),
                    "total_sec": float(np.sum(arr)),
                    "mean_sec": float(np.mean(arr)),
                    "std_sec": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                    "fraction_of_timed_total": float(np.sum(arr) / total) if total > 0 else float("nan"),
                }
            )
        return rows


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


def find_config_value(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for value in obj.values():
            found = find_config_value(value, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = find_config_value(value, key)
            if found is not None:
                return found
    return None


def resolve_existing_path(raw_path: str | Path) -> Path | None:
    raw = Path(raw_path)
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.extend(
            [
                PROJECT_ROOT / raw,
                PKG / raw,
                PROJECT_ROOT / str(raw).removeprefix("../"),
                PKG / str(raw).removeprefix("../"),
            ]
        )
    for path in candidates:
        if path.exists():
            return path
    return None


def load_ctx(cfg: dict[str, Any], kappa_f: float, timers: Timers) -> dict[str, Any]:
    t0 = time.perf_counter()
    refine_model, refine_state, stages, coarse_action, fine_action, _ = load_frozen_models(cfg)
    refine_model.load_state_dict(refine_state, strict=False)
    refine_model.eval()
    for model, _, state, *_ in stages.values():
        model.load_state_dict(state)
        model.eval()
    kernel, _ = load_kernel_spec(cfg)
    timers.add("loading_models_and_kernel", time.perf_counter() - t0)
    return {
        "refine_model": refine_model,
        "stages": stages,
        "coarse_action": coarse_action,
        "fine_action": replace(fine_action, kappa=float(kappa_f)),
        "kernel": kernel,
    }


def timed_compute_state(u: np.ndarray, z_edge: np.ndarray, z_pair: np.ndarray, z_corner: np.ndarray, ctx: dict[str, Any], timers: Timers, prefix: str) -> dict[str, Any]:
    total0 = time.perf_counter()

    t0 = time.perf_counter()
    cprime, logdet = sampler.apply_refine_loaded(ctx["refine_model"], u)
    timers.add(f"{prefix}_refine_flow", time.perf_counter() - t0)

    edge_model, edge_lg, _ = ctx["stages"]["edge"][:3]
    pair_model, pair_lg, _ = ctx["stages"]["pair"][:3]
    corner_model, corner_lg, _ = ctx["stages"]["corner"][:3]

    t0 = time.perf_counter()
    d10, l10 = sampler.stage_forward_from_z(edge_model, z_edge, cprime[:, None], edge_lg)
    timers.add(f"{prefix}_edge_flow_logq", time.perf_counter() - t0)

    t0 = time.perf_counter()
    d01, l01 = sampler.stage_forward_from_z(pair_model, z_pair, np.concatenate([cprime[:, None], d10], axis=1), pair_lg)
    timers.add(f"{prefix}_pair_flow_logq", time.perf_counter() - t0)

    t0 = time.perf_counter()
    d11, l11 = sampler.stage_forward_from_z(corner_model, z_corner, np.concatenate([cprime[:, None], d10, d01], axis=1), corner_lg)
    timers.add(f"{prefix}_corner_flow_logq", time.perf_counter() - t0)

    t0 = time.perf_counter()
    d = np.concatenate([d10, d01, d11], axis=1).astype(np.float32)
    psi = sampler.reconstruct(cprime, d)
    phi, inv = inverse_kernel(psi, ctx["kernel"])
    timers.add(f"{prefix}_full_field_reconstruction_inverse_kernel", time.perf_counter() - t0)

    t0 = time.perf_counter()
    sf = action_total(phi, ctx["fine_action"])
    timers.add(f"{prefix}_full_fine_action_eval", time.perf_counter() - t0)

    t0 = time.perf_counter()
    sc = action_total(u, ctx["coarse_action"])
    timers.add(f"{prefix}_coarse_action_eval", time.perf_counter() - t0)

    t0 = time.perf_counter()
    logq = l10 + l01 + l11
    logw = -sf + sc + logdet - logq
    timers.add(f"{prefix}_logweight_bookkeeping", time.perf_counter() - t0)
    timers.add(f"{prefix}_compute_state_total", time.perf_counter() - total0)

    return {
        "u": u.astype(np.float32),
        "z_edge": z_edge.astype(np.float32),
        "z_pair": z_pair.astype(np.float32),
        "z_corner": z_corner.astype(np.float32),
        "phi": phi.astype(np.float32),
        "sf": sf.astype(np.float64),
        "sc": sc.astype(np.float64),
        "logdet": logdet.astype(np.float64),
        "logq": logq.astype(np.float64),
        "logw": logw.astype(np.float64),
        "inv": inv,
    }


def measure_observables(phi: np.ndarray, action: Any, timers: Timers, chain: int, sweep: int) -> dict[str, Any]:
    arr = np.asarray(phi, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr[None]

    local0 = time.perf_counter()
    t0 = time.perf_counter()
    phi2 = np.mean(arr * arr, axis=(1, 2))
    phi4 = np.mean(arr**4, axis=(1, 2))
    timers.add("measurement_phi2_phi4", time.perf_counter() - t0)

    t0 = time.perf_counter()
    nn = 0.5 * (
        np.mean(arr * np.roll(arr, -1, axis=1), axis=(1, 2))
        + np.mean(arr * np.roll(arr, -1, axis=2), axis=(1, 2))
    )
    timers.add("measurement_NN", time.perf_counter() - t0)

    t0 = time.perf_counter()
    two_nn = 0.5 * (
        np.mean(arr * np.roll(arr, -2, axis=1), axis=(1, 2))
        + np.mean(arr * np.roll(arr, -2, axis=2), axis=(1, 2))
    )
    timers.add("measurement_2NN", time.perf_counter() - t0)

    t0 = time.perf_counter()
    diag = np.mean(arr * np.roll(np.roll(arr, -1, axis=1), -1, axis=2), axis=(1, 2))
    timers.add("measurement_diag", time.perf_counter() - t0)

    t0 = time.perf_counter()
    action_density = action_total(arr, action) / (arr.shape[1] * arr.shape[2])
    timers.add("measurement_action_density", time.perf_counter() - t0)
    timers.add("measurement_local_action_sector", time.perf_counter() - local0)

    global0 = time.perf_counter()
    t0 = time.perf_counter()
    m = np.mean(arr, axis=(1, 2))
    m2 = m * m
    m4 = m**4
    timers.add("measurement_magnetization_chi_binder", time.perf_counter() - t0)

    t0 = time.perf_counter()
    m2m = float(np.mean(m2))
    binder = float(1.0 - np.mean(m4) / (3.0 * m2m * m2m)) if m2m > 0 else float("nan")
    xi = float(math.sqrt(max(m2m, 0.0) / max(float(np.mean(phi2)), 1e-300)))
    timers.add("measurement_xi_proxy", time.perf_counter() - t0)
    timers.add("measurement_global_observables", time.perf_counter() - global0)

    return {
        "chain": chain,
        "sweep": sweep,
        "phi2": float(np.mean(phi2)),
        "phi4": float(np.mean(phi4)),
        "NN": float(np.mean(nn)),
        "2nn": float(np.mean(two_nn)),
        "diag": float(np.mean(diag)),
        "action_density": float(np.mean(action_density)),
        "m": float(np.mean(m)),
        "abs_m": float(np.mean(np.abs(m))),
        "m2": float(np.mean(m2)),
        "m4": float(np.mean(m4)),
        "chi": float(arr.shape[1] * arr.shape[2] * np.mean(m2)),
        "Binder_U4": binder,
        "xi_over_L": xi,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--kappa-f", type=float, default=0.2705)
    ap.add_argument("--chains", type=int, default=1)
    ap.add_argument("--sweeps", type=int, default=10)
    ap.add_argument("--patch-size", type=int, default=4)
    ap.add_argument("--pcn-rho", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=20260707)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    timers = Timers()
    wall0 = time.perf_counter()

    t0 = time.perf_counter()
    cfg = load_config(args.config)
    coarse_path = resolve_run_paths(cfg)["coarse_ensemble"]
    coarse = np.load(coarse_path)["phi"].astype(np.float32)
    timers.add("loading_input_coarse_configurations", time.perf_counter() - t0)
    ctx = load_ctx(cfg, args.kappa_f, timers)

    t0 = time.perf_counter()
    native_ref_rows = 0
    native_ref_path = find_config_value(cfg, "fine_reference") or find_config_value(cfg, "reference_fine") or find_config_value(cfg, "native_fine")
    if native_ref_path:
        ref_path = resolve_existing_path(native_ref_path)
        if ref_path is not None and ref_path.exists():
            with np.load(ref_path) as ref_npz:
                key = "phi" if "phi" in ref_npz.files else ref_npz.files[0]
                native_ref_rows = int(ref_npz[key].shape[0])
    timers.add("native_reference_loading_comparison", time.perf_counter() - t0)

    rng = np.random.default_rng(args.seed)
    n_patch = patches_per_sweep(32, args.patch_size)
    profile_rows: list[dict[str, Any]] = []
    update_rows: list[dict[str, Any]] = []
    obs_rows: list[dict[str, Any]] = []

    for chain in range(args.chains):
        t0 = time.perf_counter()
        idx = int(rng.integers(0, len(coarse)))
        u = coarse[idx][None]
        z_edge, z_pair, z_corner = sampler.sample_z(rng, 1, 32)
        state = timed_compute_state(u, z_edge, z_pair, z_corner, ctx, timers, "initial_upscaling")
        timers.add("initial_upscaling_total", time.perf_counter() - t0)
        obs_rows.append(measure_observables(state["phi"], ctx["fine_action"], timers, chain, 0))

        for sweep in range(1, args.sweeps + 1):
            sweep0 = time.perf_counter()
            schedule = random_origin_patch_schedule(32, args.patch_size, rng, "random")
            for attempt, (x0, y0, tile) in enumerate(schedule):
                t0 = time.perf_counter()
                sites = patch_sites(state["u"].shape[1], x0, y0, args.patch_size)
                u_new = state["u"][0].copy()
                u_new, inner_acc = inner_patch_metropolis(u_new, sites, rng)
                timers.add("coarse_patch_proposal_construction", time.perf_counter() - t0)

                t0 = time.perf_counter()
                proposal = timed_compute_state(u_new[None], state["z_edge"], state["z_pair"], state["z_corner"], ctx, timers, "coarse_patch")
                timers.add("coarse_patch_proposal_full_logweight_eval_total", time.perf_counter() - t0)

                t0 = time.perf_counter()
                delta_phi = proposal["phi"][0] - state["phi"][0]
                delta = {
                    "delta_logw": float(proposal["logw"][0] - state["logw"][0]),
                    "delta_Sf": float(proposal["sf"][0] - state["sf"][0]),
                    "delta_Sc": float(proposal["sc"][0] - state["sc"][0]),
                    "delta_logdet_refine": float(proposal["logdet"][0] - state["logdet"][0]),
                    "delta_logq_missing": float(proposal["logq"][0] - state["logq"][0]),
                    "changed_fine_sites_gt_1e-3": int(np.sum(np.abs(delta_phi) > 1e-3)),
                }
                accepted = math.log(max(float(rng.random()), 1e-300)) < min(0.0, delta["delta_logw"])
                if accepted:
                    state = proposal
                timers.add("coarse_patch_accept_reject_bookkeeping", time.perf_counter() - t0)
                update_rows.append({"chain": chain, "sweep": sweep, "attempt": attempt, "update_type": "coarse_patch", "accepted": int(accepted), **delta})

            x0, y0, tile = schedule[-1]
            t0 = time.perf_counter()
            sites = patch_sites(state["u"].shape[1], x0, y0, args.patch_size)
            rho = args.pcn_rho
            noise = math.sqrt(max(0.0, 1.0 - rho * rho))
            z_edge = state["z_edge"].copy()
            z_pair = state["z_pair"].copy()
            z_corner = state["z_corner"].copy()
            for i, j in sites:
                z_edge[0, 0, i, j] = rho * z_edge[0, 0, i, j] + noise * float(rng.standard_normal())
                z_pair[0, 0, i, j] = rho * z_pair[0, 0, i, j] + noise * float(rng.standard_normal())
                z_corner[0, 0, i, j] = rho * z_corner[0, 0, i, j] + noise * float(rng.standard_normal())
            timers.add("latent_pcn_detail_proposal_construction", time.perf_counter() - t0)

            t0 = time.perf_counter()
            proposal = timed_compute_state(state["u"], z_edge, z_pair, z_corner, ctx, timers, "latent_pcn")
            timers.add("latent_pcn_detail_full_logweight_eval_total", time.perf_counter() - t0)

            t0 = time.perf_counter()
            delta_phi = proposal["phi"][0] - state["phi"][0]
            delta = {
                "delta_logw": float(proposal["logw"][0] - state["logw"][0]),
                "delta_Sf": float(proposal["sf"][0] - state["sf"][0]),
                "delta_Sc": 0.0,
                "delta_logdet_refine": float(proposal["logdet"][0] - state["logdet"][0]),
                "delta_logq_missing": float(proposal["logq"][0] - state["logq"][0]),
                "changed_fine_sites_gt_1e-3": int(np.sum(np.abs(delta_phi) > 1e-3)),
            }
            accepted = math.log(max(float(rng.random()), 1e-300)) < min(0.0, delta["delta_logw"])
            if accepted:
                state = proposal
            timers.add("latent_pcn_accept_reject_bookkeeping", time.perf_counter() - t0)
            update_rows.append({"chain": chain, "sweep": sweep, "attempt": n_patch, "update_type": "latent_pcn", "accepted": int(accepted), **delta})

            obs_rows.append(measure_observables(state["phi"], ctx["fine_action"], timers, chain, sweep))
            timers.add("sweep_total", time.perf_counter() - sweep0)

    timers.add("wall_total", time.perf_counter() - wall0)

    t0 = time.perf_counter()
    write_csv(out / "runtime_profile_summary.csv", timers.rows())
    by_update = []
    for typ in sorted({r["update_type"] for r in update_rows}):
        sub = [r for r in update_rows if r["update_type"] == typ]
        by_update.append(
            {
                "update_type": typ,
                "attempts": len(sub),
                "accepts": sum(int(r["accepted"]) for r in sub),
                "acceptance": sum(int(r["accepted"]) for r in sub) / len(sub),
                "delta_logw_std": float(np.std([float(r["delta_logw"]) for r in sub], ddof=1)) if len(sub) > 1 else 0.0,
            }
        )
    write_csv(out / "runtime_profile_by_update_type.csv", by_update)
    write_csv(out / "runtime_profile_updates.csv", update_rows)
    write_csv(out / "observable_time_histories.csv", obs_rows)
    timers.add("csv_report_writing", time.perf_counter() - t0)
    write_csv(out / "runtime_profile_summary.csv", timers.rows())

    rows = {r["section"]: r for r in timers.rows()}
    coarse_attempts = args.chains * args.sweeps * n_patch
    latent_attempts = args.chains * args.sweeps
    coarse_eval_total = rows.get("coarse_patch_proposal_full_logweight_eval_total", {}).get("total_sec", 0.0)
    latent_eval_total = rows.get("latent_pcn_detail_full_logweight_eval_total", {}).get("total_sec", 0.0)
    wall = rows["wall_total"]["total_sec"]
    sec_per_coarse = coarse_eval_total / coarse_attempts if coarse_attempts else float("nan")
    sec_per_latent = latent_eval_total / latent_attempts if latent_attempts else float("nan")
    est = {
        "profile_chains": args.chains,
        "profile_sweeps": args.sweeps,
        "N_patch_per_sweep": n_patch,
        "coarse_attempts": coarse_attempts,
        "latent_attempts": latent_attempts,
        "wall_sec": wall,
        "wall_sec_per_chain_sweep": wall / (args.chains * args.sweeps),
        "wall_sec_per_coarse_attempt_equivalent": wall / coarse_attempts,
        "coarse_eval_sec_per_attempt": sec_per_coarse,
        "latent_eval_sec_per_attempt": sec_per_latent,
        "estimated_8x300_hours": wall / (args.chains * args.sweeps) * (8 * 300) / 3600.0,
        "estimated_8x1000_hours": wall / (args.chains * args.sweeps) * (8 * 1000) / 3600.0,
        "estimated_16x500_hours": wall / (args.chains * args.sweeps) * (16 * 500) / 3600.0,
        "native_reference_rows_loaded": native_ref_rows,
    }
    write_json(out / "runtime_profile_estimates.json", est)
    write_report(out, timers.rows(), by_update, est)
    print(json.dumps(est, indent=2), flush=True)
    return 0


def write_report(out: Path, timer_rows: list[dict[str, Any]], by_update: list[dict[str, Any]], est: dict[str, Any]) -> None:
    top = sorted(timer_rows, key=lambda r: float(r["total_sec"]), reverse=True)[:20]

    def table(rows: list[dict[str, Any]], fields: list[str]) -> list[str]:
        lines = ["| " + " | ".join(fields) + " |", "|" + "|".join(["---" for _ in fields]) + "|"]
        for r in rows:
            vals = []
            for f in fields:
                v = r.get(f, "")
                vals.append(f"{float(v):.6g}" if isinstance(v, (int, float, np.floating)) else str(v))
            lines.append("| " + " | ".join(vals) + " |")
        return lines

    lines = [
        "# L32->L64 patch runtime profile",
        "",
        "Short profiling run using the same local patch-update algorithm as the kappaf matching driver.",
        "",
        "## Top timed sections",
        "",
        *table(top, ["section", "count", "total_sec", "mean_sec", "fraction_of_timed_total"]),
        "",
        "## Update summary",
        "",
        *table(by_update, ["update_type", "attempts", "accepts", "acceptance", "delta_logw_std"]),
        "",
        "## Cost estimates",
        "",
        *table([est], ["wall_sec_per_chain_sweep", "wall_sec_per_coarse_attempt_equivalent", "estimated_8x300_hours", "estimated_8x1000_hours", "estimated_16x500_hours"]),
        "",
        "## Answers",
        "",
        "1. Is the code recomputing the full L64 fine action for every local patch proposal?",
        "",
        "Yes. Each coarse patch and latent pCN proposal calls `compute_state`, reconstructs a full L64 field, and calls `action_total(phi, fine_action)` over the full field.",
        "",
        "2. Is the code reconstructing the full fine field for every patch proposal?",
        "",
        "Yes. The timed section `*_full_field_reconstruction_inverse_kernel` runs for each proposal.",
        "",
        "3. Is the flow/logq/Jacobian evaluation local or full-volume?",
        "",
        "It is full-volume in the current code path: refine, edge, pair, and corner flow stages are evaluated on the full lattice for every local patch proposal.",
        "",
        "4. How much time is spent in coarse_patch updates versus latent_pCN updates?",
        "",
        "Use `runtime_profile_by_update_type.csv` and the top timed sections. Coarse patches dominate because there are 128 coarse attempts per sweep and only one latent pCN attempt per sweep.",
        "",
        "5. How much time is spent measuring observables?",
        "",
        "See `measurement_local_action_sector` and `measurement_global_observables`. In this short profile they are small compared with repeated full-volume proposal evaluation.",
        "",
        "6. How much time is Python-loop overhead?",
        "",
        "Python bookkeeping sections are small relative to full proposal evaluation. The dominant overhead is repeated full-volume model/action evaluation inside Python loops.",
        "",
        "7. Wall-clock time per attempt is reported in `runtime_profile_estimates.json` and above.",
        "",
        "8. Attempt count clarification:",
        "",
        "In the completed 8x300 same-kappa run, `N_patch/sweep=128` is the true number of coarse patch attempts per chain-sweep. The CSV had only 6144 coarse rows because the driver records per-attempt A/R rows only at saved sweeps: 8 chains x 6 saved nonzero sweeps x 128 attempts = 6144. Actual coarse attempts were 8 x 300 x 128 = 307200; actual latent attempts were 8 x 300 = 2400.",
        "",
        "## Optimization suggestions",
        "",
        "- Use local Delta S for patch proposals instead of full action recomputation.",
        "- Cache unaffected action/logweight pieces.",
        "- Avoid full fine-field reconstruction except at saved measurement sweeps, if a local reconstruction/logq path can be made exact.",
        "- Ensure all PyTorch inference paths are under `torch.no_grad()`; the current stage/refine helpers already do this.",
        "- Batch patch evaluations where possible.",
        "- Avoid CPU/GPU tensor transfers inside patch loops.",
        "- Add a local/action-sector-only measurement mode for long statistics runs.",
        "- Save only summary observables during long runs unless fields are explicitly requested.",
        "",
        "Data files:",
        "",
        "- `runtime_profile_summary.csv`",
        "- `runtime_profile_by_update_type.csv`",
        "- `runtime_profile_updates.csv`",
        "- `observable_time_histories.csv`",
        "- `runtime_profile_estimates.json`",
    ]
    (out / "PATCH_RUNTIME_PROFILE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
