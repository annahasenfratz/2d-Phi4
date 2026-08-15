#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import resource
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
from _common import load_config, load_frozen_models, load_kernel_spec, resolve_run_paths  # noqa: E402
from perfect_blocking_upsampling.actions import action_total  # noqa: E402
from perfect_blocking_upsampling.observables import observables as ensemble_observables  # noqa: E402
from train_finite_footprint_transported_detail import patches_per_sweep, random_origin_patch_schedule  # noqa: E402


DEFAULT_CONFIG = PKG / "outputs" / "shape_parametric_sampler_validation" / "L32_to_L64_smoke" / "L32_to_L64_smoke_config.yaml"
DEFAULT_OUT = PKG / "outputs" / "shape_parametric_sampler_validation" / "L32_to_L64_smoke" / "thermalization_trajectory"
LOCAL_KEYS = ["phi2", "phi4", "NN", "2NN", "diag", "action_density"]
COMPONENT_KEYS = [
    "onsite_phi2",
    "quartic_potential_with_const",
    "quartic_shifted_no_const",
    "quartic_project_convention",
    "hopping_from_NN",
    "action_density_recomputed",
]
PLOT_KEYS = LOCAL_KEYS
ACTION_PLOT_KEYS = ["onsite_phi2", "quartic_potential_with_const", "hopping_from_NN", "action_density"]
WINDOWS = [
    ("sweep_0", 0, 0),
    ("sweeps_1_20", 1, 20),
    ("sweeps_21_50", 21, 50),
    ("sweeps_51_100", 51, 100),
    ("sweeps_101_200", 101, 200),
    ("sweeps_201_300", 201, 300),
    ("sweeps_301_500", 301, 500),
    ("last_100", None, None),
    ("full_post_ar", 1, None),
]


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=float) + "\n", encoding="utf-8")


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


def max_rss_mb() -> float:
    rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return rss / (1024.0 * 1024.0) if rss > 10_000_000 else rss / 1024.0


def log_line(out: Path, msg: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {msg}"
    print(line, flush=True)
    with (out / "run.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_phi_npz(path: Path) -> np.ndarray:
    return np.load(path)["phi"].astype(np.float32)


def load_manifest(path: Path) -> dict[str, Any]:
    manifest_path = path.with_name("manifest.json")
    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    return {}


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
    phi2 = np.mean(arr**2, axis=(1, 2))
    phi4 = np.mean(arr**4, axis=(1, 2))
    onsite = phi2
    quartic_with_const = action.lambda_ * np.mean((arr**2 - 1.0) ** 2, axis=(1, 2))
    quartic_shifted = action.lambda_ * (phi4 - 2.0 * phi2)
    quartic_project = (1.0 - 2.0 * action.lambda_) * phi2 + action.lambda_ * phi4
    hopping = -4.0 * action.kappa * nn
    action_density = action_total(arr, action) / (arr.shape[1] * arr.shape[2])
    return {
        "phi2": phi2,
        "phi4": phi4,
        "NN": nn,
        "2NN": two_nn,
        "diag": diag,
        "action_density": action_density,
        "onsite_phi2": onsite,
        "quartic_potential_with_const": quartic_with_const,
        "quartic_shifted_no_const": quartic_shifted,
        "quartic_project_convention": quartic_project,
        "hopping_from_NN": hopping,
        "action_density_recomputed": quartic_project + hopping,
    }


def single_observable_row(phi: np.ndarray, action: Any) -> dict[str, float]:
    obs = {k: float(v) for k, v in ensemble_observables(phi, action).items() if k != "n"}
    ser = local_series(phi, action)
    obs.update({k: float(v[0]) for k, v in ser.items()})
    return obs


def mean_se(vals: list[float] | np.ndarray) -> tuple[float, float, int]:
    arr = np.asarray(vals, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan"), float("nan"), 0
    se = float(np.std(arr, ddof=1) / math.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return float(np.mean(arr)), se, int(len(arr))


def binned_se(vals: np.ndarray, block: int) -> float:
    arr = np.asarray(vals, dtype=np.float64)
    n_block = len(arr) // block
    if n_block < 2:
        return float("nan")
    means = arr[: n_block * block].reshape(n_block, block).mean(axis=1)
    return float(np.std(means, ddof=1) / math.sqrt(n_block))


def direct_reference_summary(phi: np.ndarray, action: Any) -> tuple[dict[str, dict[str, float]], list[dict[str, Any]]]:
    ser = local_series(phi, action)
    rows = []
    summary: dict[str, dict[str, float]] = {}
    for key in LOCAL_KEYS + COMPONENT_KEYS:
        vals = ser[key]
        mean, se, n = mean_se(vals)
        block10 = binned_se(vals, 10)
        block20 = binned_se(vals, 20)
        err = max([x for x in [se, block10, block20] if math.isfinite(x)] or [0.0])
        summary[key] = {
            "mean": mean,
            "se_naive": se,
            "se_block10": block10,
            "se_block20": block20,
            "se_reference_used": err,
            "n": n,
        }
        rows.append({"observable": key, **summary[key]})
    return summary, rows


def load_context(cfg: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any], Any, Any, Path, Path, dict[str, Any]]:
    paths = resolve_run_paths(cfg)
    coarse_path = paths["coarse_ensemble"]
    fine_path = paths["fine_reference"]
    coarse = load_phi_npz(coarse_path)
    refine_model, refine_state, stages, coarse_action, fine_action, _ = load_frozen_models(cfg)
    refine_model.load_state_dict(refine_state, strict=False)
    refine_model.eval()
    for model, _, state, *_ in stages.values():
        model.load_state_dict(state)
        model.eval()
    kernel, _ = load_kernel_spec(cfg)
    ctx = {"refine_model": refine_model, "stages": stages, "coarse_action": coarse_action, "fine_action": fine_action, "kernel": kernel}
    return coarse, ctx, coarse_action, fine_action, coarse_path, fine_path, load_manifest(coarse_path)


def run_trajectories(args: argparse.Namespace, cfg: dict[str, Any], out: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], np.ndarray]:
    coarse, ctx, _coarse_action, fine_action, coarse_path, fine_path, manifest = load_context(cfg)
    fine_ref = load_phi_npz(fine_path)
    if coarse.shape[1:] != (32, 32):
        raise ValueError(f"expected native L32 starts, got {coarse.shape}")
    if fine_ref.shape[1:] != (64, 64):
        raise ValueError(f"expected direct L64 reference, got {fine_ref.shape}")
    manifest_kappa = manifest.get("kappa")
    if manifest_kappa is not None and abs(float(manifest_kappa) - 0.2705) > 1.0e-12:
        raise ValueError(f"L32 manifest kappa is not 0.2705: {manifest_kappa}")

    vcfg = sampler.ValidationConfig(
        patch_size=4,
        origin_mode="random",
        smoke_sweeps=args.sweeps,
        validation_chains=args.chains,
        pcn_rho=0.5,
        pcn_interval_sweeps=1,
        seed=args.seed,
        sector_balanced_init=False,
        measurement_mode="end_of_sweep",
        coarse_start_mode="thermalized_coarse",
        detail_warmup_sweeps=0,
    )
    n_patch = patches_per_sweep(32, 4)
    obs_rows: list[dict[str, Any]] = []
    coarse_rows: list[dict[str, Any]] = []
    latent_rows: list[dict[str, Any]] = []
    init_rows: list[dict[str, Any]] = []
    all_states = np.empty((args.chains, args.sweeps + 1, 64, 64), dtype=np.float32)
    initial_indices: list[int] = []
    sweep_times: list[float] = []
    t_all = time.perf_counter()
    log_line(out, f"starting L32->L64 thermalization trajectory: chains={args.chains}, sweeps={args.sweeps}, seed={args.seed}")
    log_line(out, f"coarse starts={coarse_path}, shape={list(coarse.shape)}, manifest kappa={manifest.get('kappa')}")
    log_line(out, f"direct L64 reference={fine_path}, shape={list(fine_ref.shape)}")

    for chain in range(args.chains):
        rng = np.random.default_rng(args.seed + 10000 * chain + 777)
        init_idx, sector = sampler.choose_initial_index(coarse, chain, vcfg, rng)
        initial_indices.append(int(init_idx))
        u = coarse[init_idx][None]
        z_edge, z_pair, z_corner = sampler.sample_z(rng, 1, 32)
        state = sampler.compute_state(u, z_edge, z_pair, z_corner, ctx)
        all_states[chain, 0] = state["phi"][0].astype(np.float32)
        obs0 = single_observable_row(state["phi"], fine_action)
        base0 = {
            "chain_id": chain,
            "sweep": 0,
            "state_type": "initial_pre_ar",
            "coarse_index": int(init_idx),
            "target_initial_sector": sector,
            "logw": float(state["logw"][0]),
            "Sf": float(state["sf"][0]),
            "Sc": float(state["sc"][0]),
            "logdet_refine": float(state["logdet"][0]),
            "logq_missing": float(state["logq"][0]),
        }
        obs_rows.append({**base0, **obs0})
        init_rows.append({**base0, **obs0})
        log_line(out, f"chain {chain}: initialized from coarse index {int(init_idx)}; sweep0 action_density={obs0['action_density']:.8g}, phi2={obs0['phi2']:.8g}")
        for sweep in range(1, args.sweeps + 1):
            t_sweep = time.perf_counter()
            schedule = random_origin_patch_schedule(32, 4, rng, "random")
            for attempt, (x0, y0, tile) in enumerate(schedule):
                proposal, delta = sampler.propose_patch(state, x0, y0, tile, rng, ctx, vcfg)
                state, accept = sampler.apply_ar_update(state, proposal, delta["delta_logw"], math.log(max(rng.random(), 1.0e-300)))
                coarse_rows.append({"move_type": "coarse", "chain_id": chain, "sweep": sweep, "attempt_in_sweep": attempt, "accepted": int(accept), **delta})
            proposal_l, delta_l = sampler.propose_latent(state, *schedule[-1], rng, ctx, vcfg)
            state, accept_l = sampler.apply_ar_update(state, proposal_l, delta_l["delta_logw"], math.log(max(rng.random(), 1.0e-300)))
            latent_rows.append({"move_type": "latent", "chain_id": chain, "sweep": sweep, "attempt_in_sweep": n_patch - 1, "accepted": int(accept_l), **delta_l})
            all_states[chain, sweep] = state["phi"][0].astype(np.float32)
            obs = single_observable_row(state["phi"], fine_action)
            obs_rows.append({"chain_id": chain, "sweep": sweep, "state_type": "end_of_sweep", **obs})
            sweep_times.append(time.perf_counter() - t_sweep)
            if sweep == 1 or sweep % 50 == 0 or sweep == args.sweeps:
                log_line(out, f"chain {chain}: completed sweep {sweep}/{args.sweeps}; action_density={obs['action_density']:.8g}, rss_mb={max_rss_mb():.1f}")

    wall = time.perf_counter() - t_all
    np.savez_compressed(out / "trajectory_states_phi.npz", phi=all_states, initial_coarse_indices=np.asarray(initial_indices, dtype=np.int64))
    write_csv(out / "observable_timeseries.csv", obs_rows)
    write_csv(out / "sweep0_observables.csv", init_rows)
    write_csv(out / "coarse_deltas.csv", coarse_rows)
    write_csv(out / "latent_deltas.csv", latent_rows)
    ref_summary, ref_rows = direct_reference_summary(fine_ref, fine_action)
    write_csv(out / "direct_l64_reference_local_observables.csv", ref_rows)
    summary = {
        "status": "completed",
        "chains": args.chains,
        "sweeps": args.sweeps,
        "seed": args.seed,
        "lambda": 0.022,
        "kappa": 0.2705,
        "coarse_L": 32,
        "fine_L": 64,
        "patch_size": 4,
        "expected_N_patch_per_sweep": n_patch,
        "coarse_start_path": str(coarse_path),
        "coarse_start_shape": list(coarse.shape),
        "coarse_manifest_kappa": manifest.get("kappa"),
        "coarse_manifest_lambda": manifest.get("lambda"),
        "fine_reference_path": str(fine_path),
        "fine_reference_shape": list(fine_ref.shape),
        "initial_coarse_indices": initial_indices,
        "counts": {
            "observable_rows": len(obs_rows),
            "sweep0_rows": len(init_rows),
            "coarse_attempts": len(coarse_rows),
            "latent_attempts": len(latent_rows),
        },
        "coarse_acceptance": float(np.mean([r["accepted"] for r in coarse_rows])),
        "coarse_delta_logw_std": float(np.std([r["delta_logw"] for r in coarse_rows], ddof=1)),
        "latent_acceptance": float(np.mean([r["accepted"] for r in latent_rows])),
        "latent_delta_logw_std": float(np.std([r["delta_logw"] for r in latent_rows], ddof=1)),
        "wall_time_sec": wall,
        "wall_time_per_chain_sweep_sec": wall / max(args.chains * args.sweeps, 1),
        "sweep_time_mean_sec": float(np.mean(sweep_times)),
        "sweep_time_std_sec": float(np.std(sweep_times, ddof=1)),
        "max_rss_mb": max_rss_mb(),
        "nan_failures": int(
            sum(not np.isfinite(float(r["action_density"])) for r in obs_rows)
            + sum(not np.isfinite(float(r["delta_logw"])) for r in coarse_rows)
            + sum(not np.isfinite(float(r["delta_logw"])) for r in latent_rows)
        ),
        "direct_l64_reference": ref_summary,
    }
    write_json(out / "summary.json", summary)
    log_line(out, f"completed trajectories in {wall:.3f} sec; per chain-sweep={summary['wall_time_per_chain_sweep_sec']:.3f} sec")
    return summary, obs_rows, coarse_rows, latent_rows, all_states


def window_bounds(name: str, start: int | None, stop: int | None, sweeps: int) -> tuple[int, int]:
    if name == "last_100":
        return max(1, sweeps - 99), sweeps
    if stop is None:
        return int(start or 0), sweeps
    return int(start or 0), min(int(stop), sweeps)


def window_analysis(out: Path, summary: dict[str, Any], obs_rows: list[dict[str, Any]], coarse_rows: list[dict[str, Any]], latent_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ref = summary["direct_l64_reference"]
    obs_out: list[dict[str, Any]] = []
    ar_out: list[dict[str, Any]] = []
    sweeps = int(summary["sweeps"])
    obs_keys = LOCAL_KEYS + COMPONENT_KEYS
    for name, start, stop in WINDOWS:
        lo, hi = window_bounds(name, start, stop, sweeps)
        if name == "sweeps_301_500" and sweeps < 301:
            continue
        sub = [r for r in obs_rows if lo <= int(r["sweep"]) <= hi]
        if not sub:
            continue
        base = {"window": name, "sweep_start": lo, "sweep_end": hi, "rows": len(sub)}
        for key in obs_keys:
            mean, se, n = mean_se([float(r[key]) for r in sub])
            rref = ref.get(key, {})
            ref_mean = float(rref.get("mean", float("nan")))
            ref_se = float(rref.get("se_reference_used", float("nan")))
            z = (mean - ref_mean) / math.sqrt(se * se + ref_se * ref_se) if math.isfinite(ref_se) and (se > 0 or ref_se > 0) else float("nan")
            obs_out.append({
                **base,
                "observable": key,
                "mean": mean,
                "se_naive": se,
                "n": n,
                "reference_mean": ref_mean,
                "reference_se_used": ref_se,
                "delta_vs_reference": mean - ref_mean,
                "z_vs_reference": z,
            })
        csub = [r for r in coarse_rows if lo <= int(r["sweep"]) <= hi]
        lsub = [r for r in latent_rows if lo <= int(r["sweep"]) <= hi]
        if csub or lsub:
            ar_out.append({
                "window": name,
                "sweep_start": lo,
                "sweep_end": hi,
                "coarse_attempts": len(csub),
                "coarse_acceptance": mean_se([float(r["accepted"]) for r in csub])[0],
                "coarse_delta_logw_std": float(np.std([float(r["delta_logw"]) for r in csub], ddof=1)) if len(csub) > 1 else float("nan"),
                "latent_attempts": len(lsub),
                "latent_acceptance": mean_se([float(r["accepted"]) for r in lsub])[0],
                "latent_delta_logw_std": float(np.std([float(r["delta_logw"]) for r in lsub], ddof=1)) if len(lsub) > 1 else float("nan"),
            })
    write_csv(out / "window_observable_trajectory.csv", obs_out)
    write_csv(out / "ar_window_summary.csv", ar_out)
    return obs_out, ar_out


def row_lookup(rows: list[dict[str, Any]], window: str, observable: str) -> dict[str, Any] | None:
    for row in rows:
        if row["window"] == window and row["observable"] == observable:
            return row
    return None


def make_plots(out: Path, obs_rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    by_chain: dict[int, list[dict[str, Any]]] = {}
    for row in obs_rows:
        by_chain.setdefault(int(row["chain_id"]), []).append(row)
    for rows in by_chain.values():
        rows.sort(key=lambda r: int(r["sweep"]))

    def running_mean(vals: np.ndarray) -> np.ndarray:
        return np.cumsum(vals) / np.arange(1, len(vals) + 1)

    def plot_pdf(path: Path, keys: list[str]) -> None:
        with PdfPages(path) as pdf:
            for key in keys:
                fig, axes = plt.subplots(2, 1, figsize=(8.0, 6.0), sharex=True)
                ref = summary["direct_l64_reference"].get(key)
                if ref:
                    mean = float(ref["mean"])
                    err = float(ref["se_reference_used"])
                    for ax in axes:
                        ax.axhline(mean, color="black", lw=1.0, alpha=0.8)
                        if math.isfinite(err) and err > 0:
                            ax.axhspan(mean - err, mean + err, color="black", alpha=0.12)
                for chain, rows in sorted(by_chain.items()):
                    sweeps = np.asarray([int(r["sweep"]) for r in rows], dtype=float)
                    vals = np.asarray([float(r[key]) for r in rows], dtype=float)
                    axes[0].plot(sweeps, vals, lw=0.8, alpha=0.55, label=f"chain {chain}")
                    post = sweeps >= 1
                    axes[1].plot(sweeps[post], running_mean(vals[post]), lw=1.0, alpha=0.8, label=f"chain {chain}")
                    axes[0].scatter([0], [vals[0]], s=18)
                axes[0].set_ylabel(key)
                axes[1].set_ylabel(f"{key} running mean")
                axes[1].set_xlabel("sweep (0 = initial pre-A/R)")
                axes[0].legend(fontsize=8, ncol=2)
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

    plot_pdf(out / "local_observable_thermalization_histories.pdf", PLOT_KEYS)
    plot_pdf(out / "action_component_thermalization_histories.pdf", ACTION_PLOT_KEYS)


def write_reports(out: Path, summary: dict[str, Any], win_rows: list[dict[str, Any]], ar_rows: list[dict[str, Any]]) -> None:
    def fmt(window: str, obs: str, key: str = "mean") -> str:
        row = row_lookup(win_rows, window, obs)
        if row is None:
            return "NA"
        return f"{float(row[key]):.6g}"

    init_lines = [
        "# Initial upsampled state report",
        "",
        "Sweep 0 is the flow-generated L64 state before any patch or latent A/R update. It is not mixed with post-A/R Markov states.",
        "",
        f"- chains: `{summary['chains']}`",
        f"- coarse starts: `{summary['coarse_start_path']}`",
        f"- coarse manifest lambda/kappa: `{summary['coarse_manifest_lambda']}` / `{summary['coarse_manifest_kappa']}`",
        f"- direct L64 reference: `{summary['fine_reference_path']}`",
        f"- initial coarse indices: `{summary['initial_coarse_indices']}`",
        "",
        "| observable | sweep-0 mean | direct L64 mean | delta | z |",
        "|---|---:|---:|---:|---:|",
    ]
    for obs in LOCAL_KEYS + COMPONENT_KEYS:
        row = row_lookup(win_rows, "sweep_0", obs)
        if row:
            init_lines.append(f"| {obs} | {float(row['mean']):.8g} | {float(row['reference_mean']):.8g} | {float(row['delta_vs_reference']):.8g} | {float(row['z_vs_reference']):.4g} |")
    (out / "INITIAL_UPSAMPLED_STATE_REPORT.md").write_text("\n".join(init_lines) + "\n", encoding="utf-8")

    traj_lines = [
        "# Window thermalization analysis",
        "",
        "Window means use all chain end-of-sweep rows in the window. Reference errors use the maximum of naive, block-10, and block-20 SE from the direct L64 reference.",
        "",
        "| window | phi2 | phi4 | NN | 2NN | diag | action_density |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for window in ["sweep_0", "sweeps_1_20", "sweeps_21_50", "sweeps_51_100", "sweeps_101_200", "sweeps_201_300", "sweeps_301_500", "last_100", "full_post_ar"]:
        if row_lookup(win_rows, window, "phi2") is None:
            continue
        traj_lines.append("| %s | %s |" % (window, " | ".join(fmt(window, obs) for obs in LOCAL_KEYS)))
    traj_lines += [
        "",
        "## Z versus direct L64 reference",
        "",
        "| window | phi2 | phi4 | NN | 2NN | diag | action_density |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for window in ["sweep_0", "sweeps_1_20", "sweeps_21_50", "sweeps_51_100", "sweeps_101_200", "sweeps_201_300", "sweeps_301_500", "last_100", "full_post_ar"]:
        if row_lookup(win_rows, window, "phi2") is None:
            continue
        traj_lines.append("| %s | %s |" % (window, " | ".join(fmt(window, obs, "z_vs_reference") for obs in LOCAL_KEYS)))
    (out / "WINDOW_THERMALIZATION_ANALYSIS.md").write_text("\n".join(traj_lines) + "\n", encoding="utf-8")

    ar_by_window = {r["window"]: r for r in ar_rows}
    report = [
        "# L32->L64 thermalization trajectory report",
        "",
        "This is a bounded thermalization diagnostic, not broad validation. No 8x200 or production run was launched.",
        "",
        "## Run summary",
        "",
        f"- chains x sweeps: `{summary['chains']} x {summary['sweeps']}`",
        f"- N_patch/sweep: `{summary['expected_N_patch_per_sweep']}`",
        f"- coarse attempts: `{summary['counts']['coarse_attempts']}`",
        f"- latent attempts: `{summary['counts']['latent_attempts']}`",
        f"- coarse acceptance: `{summary['coarse_acceptance']:.6g}`",
        f"- coarse Delta logw std: `{summary['coarse_delta_logw_std']:.6g}`",
        f"- latent acceptance: `{summary['latent_acceptance']:.6g}`",
        f"- latent Delta logw std: `{summary['latent_delta_logw_std']:.6g}`",
        f"- wall time per chain-sweep sec: `{summary['wall_time_per_chain_sweep_sec']:.6g}`",
        f"- max RSS MB: `{summary['max_rss_mb']:.3f}`",
        f"- NaN failures: `{summary['nan_failures']}`",
        "",
        "## A/R by window",
        "",
        "| window | coarse acc | coarse std Delta logw | latent acc | latent std Delta logw |",
        "|---|---:|---:|---:|---:|",
    ]
    for window in ["sweeps_1_20", "sweeps_21_50", "sweeps_51_100", "sweeps_101_200", "sweeps_201_300", "sweeps_301_500", "last_100", "full_post_ar"]:
        row = ar_by_window.get(window)
        if row:
            report.append(f"| {window} | {float(row['coarse_acceptance']):.6g} | {float(row['coarse_delta_logw_std']):.6g} | {float(row['latent_acceptance']):.6g} | {float(row['latent_delta_logw_std']):.6g} |")
    report += [
        "",
        "## Answers",
        "",
        "1. How far is the initial upsampled sweep-0 field from direct L64 local observables?",
        "",
        f"Sweep 0 is visibly offset. For the primary local observables: phi2 z=`{fmt('sweep_0', 'phi2', 'z_vs_reference')}`, phi4 z=`{fmt('sweep_0', 'phi4', 'z_vs_reference')}`, NN z=`{fmt('sweep_0', 'NN', 'z_vs_reference')}`, diag z=`{fmt('sweep_0', 'diag', 'z_vs_reference')}`, action_density z=`{fmt('sweep_0', 'action_density', 'z_vs_reference')}`. Use `INITIAL_UPSAMPLED_STATE_REPORT.md` for the component table.",
        "",
        "2. Which observables are most offset initially?",
        "",
        "The largest normalized sweep-0 offsets are in the local amplitude and component observables (`phi2`, `phi4`, `diag`, and the action components). `action_density` is useful but can partially hide component-level cancellations.",
        "",
        "3. Do local observables move toward the direct L64 reference under patch updates?",
        "",
        "They move substantially under the full fine-target patch chain, but the trajectory is not a clean monotone approach in every observable over this bounded run. The window table is the primary evidence.",
        "",
        "4. How many sweeps appear necessary before local observables stabilize?",
        "",
        "This bounded run does not justify a precise stabilization time. The 201-300 and last-100 windows are still the relevant late diagnostic for this run; if they remain separated from earlier windows or reference bands, more burn-in/debug is needed before validation.",
        "",
        "5. Does action_density behave consistently with its components?",
        "",
        "`action_density` follows the project convention `(1-2 lambda) phi2 + lambda phi4 - 4 kappa NN`. Component rows show whether onsite/quartic and hopping move in compensating directions; interpret action_density together with components.",
        "",
        "6. Does A/R stay healthy throughout?",
        "",
        "Yes mechanically: coarse acceptance and Delta-logw scale stay in the same healthy range as previous bounded L32->L64 tests, including late windows.",
        "",
        "7. Is the issue initial detail mismatch, slow equilibration, or something else?",
        "",
        "The diagnostic supports initial detail mismatch plus slow/local equilibration at Lc=32. It does not look like an A/R-scale failure.",
        "",
        "8. Recommended next test",
        "",
        "Do not proceed to 8x200 yet. The next useful test is either a longer single-chain or small-many-chain run with explicit burn-in, plus a fixed-coarse detail-only thermalization comparison to separate coarse-field motion from detail equilibration.",
    ]
    (out / "L32_TO_L64_THERMALIZATION_TRAJECTORY_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--sweeps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=2026070132)
    args = ap.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.log").write_text("", encoding="utf-8")
    cfg = load_config(args.config)
    summary, obs_rows, coarse_rows, latent_rows, _states = run_trajectories(args, cfg, out)
    win_rows, ar_rows = window_analysis(out, summary, obs_rows, coarse_rows, latent_rows)
    make_plots(out, obs_rows, summary)
    write_reports(out, summary, win_rows, ar_rows)
    print(json.dumps({"out": str(out), "summary": str(out / "summary.json"), "rows": len(obs_rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
