#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import resource
import time
from pathlib import Path
from typing import Any

import numpy as np

import run_shape_parametric_sampler_validation as sampler
from _common import load_config, load_frozen_models, load_kernel_spec, resolve_run_paths
from perfect_blocking_upsampling.actions import action_total
from perfect_blocking_upsampling.observables import observables as ensemble_observables
from train_finite_footprint_transported_detail import patches_per_sweep, random_origin_patch_schedule


PKG = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PKG / "outputs" / "shape_parametric_sampler_validation" / "L32_to_L64_smoke" / "L32_to_L64_smoke_config.yaml"
DEFAULT_OUT = PKG / "outputs" / "shape_parametric_sampler_validation" / "L32_to_L64_smoke"


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=float) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def max_rss_mb() -> float:
    rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # macOS reports bytes; Linux reports KiB.
    return rss / (1024.0 * 1024.0) if rss > 10_000_000 else rss / 1024.0


def load_coarse(cfg: dict[str, Any]) -> tuple[np.ndarray, Path]:
    paths = resolve_run_paths(cfg)
    coarse_path = paths["coarse_ensemble"]
    coarse = np.load(coarse_path)["phi"].astype(np.float32)
    return coarse, coarse_path


def load_optional_fine_reference(cfg: dict[str, Any]) -> tuple[np.ndarray | None, Path | None]:
    paths = resolve_run_paths(cfg)
    fine_path = paths["fine_reference"]
    if not fine_path.exists():
        return None, fine_path
    fine = np.load(fine_path)["phi"].astype(np.float32)
    return fine, fine_path


def local_observable_series(phi: np.ndarray, action) -> dict[str, np.ndarray]:
    arr = np.asarray(phi, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr[None, :, :]
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


def local_observable_summary(phi: np.ndarray, action) -> dict[str, dict[str, float]]:
    series = local_observable_series(phi, action)
    return {
        key: {
            "mean": float(np.mean(val)),
            "std": float(np.std(val, ddof=1)) if len(val) > 1 else 0.0,
            "se_naive": float(np.std(val, ddof=1) / math.sqrt(len(val))) if len(val) > 1 else 0.0,
            "n": int(len(val)),
        }
        for key, val in series.items()
    }


def finite_delta(delta: dict[str, Any]) -> bool:
    vals = [
        delta.get("delta_logw"),
        delta.get("delta_Sf"),
        delta.get("delta_Sc"),
        delta.get("delta_logdet_refine"),
        delta.get("delta_logq_missing"),
    ]
    return all(np.isfinite(float(v)) for v in vals)


def summarize_moves(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    if not rows:
        return {
            f"{prefix}_attempts": 0,
            f"{prefix}_acceptance": float("nan"),
            f"{prefix}_std_delta_logw": float("nan"),
            f"{prefix}_mean_abs_delta_logw": float("nan"),
        }
    accepted = np.asarray([int(r["accepted"]) for r in rows], dtype=np.float64)
    dlogw = np.asarray([float(r["delta_logw"]) for r in rows], dtype=np.float64)
    return {
        f"{prefix}_attempts": int(len(rows)),
        f"{prefix}_acceptance": float(np.mean(accepted)),
        f"{prefix}_std_delta_logw": float(np.std(dlogw, ddof=1)) if len(dlogw) > 1 else 0.0,
        f"{prefix}_mean_abs_delta_logw": float(np.mean(np.abs(dlogw))),
        f"{prefix}_nan_delta_rows": int(np.sum(~np.isfinite(dlogw))),
    }


def run_preflight(cfg: dict[str, Any], out: Path, seed: int) -> tuple[dict[str, Any], dict[str, Any], np.ndarray]:
    coarse, coarse_path = load_coarse(cfg)
    fine_ref, fine_ref_path = load_optional_fine_reference(cfg)
    if coarse.shape[1:] != (32, 32):
        raise ValueError(f"expected L32 coarse starts, got {coarse.shape}")
    refine_model, refine_state, stages, coarse_action, fine_action, _ = load_frozen_models(cfg)
    refine_model.load_state_dict(refine_state, strict=False)
    refine_model.eval()
    for model, _, state, *_ in stages.values():
        model.load_state_dict(state)
        model.eval()
    kernel, kernel_json = load_kernel_spec(cfg)
    paths = resolve_run_paths(cfg)
    n_patch = patches_per_sweep(32, 4)

    rng = np.random.default_rng(seed)
    u0 = coarse[int(rng.integers(0, len(coarse)))][None]
    z_edge, z_pair, z_corner = sampler.sample_z(rng, 1, 32)
    ctx = {
        "refine_model": refine_model,
        "stages": stages,
        "coarse_action": coarse_action,
        "fine_action": fine_action,
        "kernel": kernel,
    }
    t0 = time.perf_counter()
    state = sampler.compute_state(u0, z_edge, z_pair, z_corner, ctx)
    compute_state_sec = time.perf_counter() - t0
    schedule = random_origin_patch_schedule(32, 4, rng, "random")

    t1 = time.perf_counter()
    patch_prop, patch_delta = sampler.propose_patch(state, *schedule[0], rng, ctx, sampler.ValidationConfig(patch_size=4))
    patch_sec = time.perf_counter() - t1

    t2 = time.perf_counter()
    latent_prop, latent_delta = sampler.propose_latent(state, *schedule[0], rng, ctx, sampler.ValidationConfig(patch_size=4, pcn_rho=0.5))
    latent_sec = time.perf_counter() - t2

    manifest_path = coarse_path.with_name("manifest.json")
    provenance_path = coarse_path.with_name("provenance.json")
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
    coarse_obs = ensemble_observables(coarse, coarse_action)
    coarse_local = local_observable_summary(coarse, coarse_action)
    coarse_obs["2NN"] = coarse_local["2NN"]["mean"]
    coarse_obs["diag"] = coarse_local["diag"]["mean"]
    coarse_obs["action_density"] = coarse_local["action_density"]["mean"]
    fine_ref_summary = None
    if fine_ref is not None and fine_ref.shape[1:] == (64, 64):
        fine_ref_summary = {
            "path": str(fine_ref_path),
            "shape": list(fine_ref.shape),
            "local_observables": local_observable_summary(fine_ref, fine_action),
            "diagnostics": {
                key: float(ensemble_observables(fine_ref, fine_action)[key])
                for key in ["m", "abs_m", "susceptibility", "Binder_U4", "xi_over_L"]
            },
        }

    preflight = {
        "status": "passed",
        "coarse_start_path": str(coarse_path),
        "coarse_shape": list(coarse.shape),
        "coarse_dtype": str(coarse.dtype),
        "coarse_all_finite": bool(np.isfinite(coarse).all()),
        "coarse_manifest": manifest,
        "coarse_provenance": provenance,
        "coarse_observable_summary": coarse_obs,
        "fine_reference_found": fine_ref_summary is not None,
        "fine_reference_path": str(fine_ref_path) if fine_ref_path is not None else None,
        "fine_reference_note": "Native L64 reference is used for local-observable diagnostics only." if fine_ref_summary else "No usable native L64 reference was found.",
        "fine_reference_summary": fine_ref_summary,
        "coarse_L": 32,
        "fine_L": 64,
        "patch_size": 4,
        "expected_N_patch_per_sweep": n_patch,
        "expected_tiny_smoke": {
            "chains": 1,
            "sweeps": 20,
            "coarse_attempts": 1 * 20 * n_patch,
            "latent_pcn_attempts": 20,
            "measured_rows": 20,
        },
        "kernel": kernel_json,
        "bundle_preflight": sampler.dummy_full_bundle_preflight(paths["frozen_dir"], coarse_l=32),
        "state_checks": {
            "phi_shape": list(state["phi"].shape),
            "logw_finite": bool(np.isfinite(state["logw"]).all()),
            "sf_finite": bool(np.isfinite(state["sf"]).all()),
            "sc_finite": bool(np.isfinite(state["sc"]).all()),
            "logdet_finite": bool(np.isfinite(state["logdet"]).all()),
            "logq_finite": bool(np.isfinite(state["logq"]).all()),
            "reblocking_max_abs_error": sampler.reblocking_error(state, ctx),
            "inverse_max_imag": float(state["inv"].get("max_imag", float("nan"))),
        },
        "proposal_checks": {
            "coarse_patch_delta_finite": finite_delta(patch_delta),
            "coarse_patch_delta": patch_delta,
            "coarse_patch_logw_finite": bool(np.isfinite(patch_prop["logw"]).all()),
            "latent_pcn_delta_finite": finite_delta(latent_delta),
            "latent_pcn_delta": latent_delta,
            "latent_pcn_logw_finite": bool(np.isfinite(latent_prop["logw"]).all()),
        },
        "timing_sec": {
            "compute_state": compute_state_sec,
            "one_coarse_patch_proposal": patch_sec,
            "one_latent_pcn_proposal": latent_sec,
        },
        "max_rss_mb_after_preflight": max_rss_mb(),
    }
    checks = [
        preflight["coarse_all_finite"],
        preflight["state_checks"]["phi_shape"] == [1, 64, 64],
        preflight["state_checks"]["logw_finite"],
        preflight["state_checks"]["sf_finite"],
        preflight["state_checks"]["sc_finite"],
        preflight["state_checks"]["logdet_finite"],
        preflight["state_checks"]["logq_finite"],
        preflight["proposal_checks"]["coarse_patch_delta_finite"],
        preflight["proposal_checks"]["coarse_patch_logw_finite"],
        preflight["proposal_checks"]["latent_pcn_delta_finite"],
        preflight["proposal_checks"]["latent_pcn_logw_finite"],
        np.isfinite(preflight["state_checks"]["reblocking_max_abs_error"]),
    ]
    if not all(checks):
        preflight["status"] = "failed"
    write_json(out / "preflight_summary.json", preflight)
    return preflight, ctx, coarse


def run_tiny_smoke(coarse: np.ndarray, ctx: dict[str, Any], out: Path, seed: int, *, sweeps: int = 20, chains: int = 1) -> dict[str, Any]:
    cfg = sampler.ValidationConfig(
        patch_size=4,
        origin_mode="random",
        smoke_sweeps=sweeps,
        validation_chains=chains,
        pcn_rho=0.5,
        pcn_interval_sweeps=1,
        seed=seed,
        sector_balanced_init=False,
        measurement_mode="end_of_sweep",
        coarse_start_mode="thermalized_coarse",
        detail_warmup_sweeps=0,
    )
    coarse_rows: list[dict[str, Any]] = []
    latent_rows: list[dict[str, Any]] = []
    obs_rows: list[dict[str, Any]] = []
    sweep_times = []
    n_patch = patches_per_sweep(32, 4)
    t_all = time.perf_counter()
    initial_indices = []
    for chain in range(chains):
        rng = np.random.default_rng(seed + 10000 * chain + 777)
        init_idx = int(rng.integers(0, len(coarse)))
        initial_indices.append(init_idx)
        u = coarse[init_idx][None]
        z_edge, z_pair, z_corner = sampler.sample_z(rng, 1, 32)
        state = sampler.compute_state(u, z_edge, z_pair, z_corner, ctx)
        for sweep in range(sweeps):
            t_sweep = time.perf_counter()
            schedule = random_origin_patch_schedule(32, 4, rng, "random")
            for attempt, (x0, y0, tile) in enumerate(schedule):
                proposal, delta = sampler.propose_patch(state, x0, y0, tile, rng, ctx, cfg)
                state, accept = sampler.apply_ar_update(state, proposal, delta["delta_logw"], math.log(max(rng.random(), 1.0e-300)))
                coarse_rows.append({"move_type": "coarse", "chain_id": chain, "sweep": sweep, "attempt_in_sweep": attempt, "accepted": int(accept), **delta})
            proposal_l, delta_l = sampler.propose_latent(state, *schedule[-1], rng, ctx, cfg)
            state, accept_l = sampler.apply_ar_update(state, proposal_l, delta_l["delta_logw"], math.log(max(rng.random(), 1.0e-300)))
            latent_rows.append({"move_type": "latent", "chain_id": chain, "sweep": sweep, "attempt_in_sweep": n_patch - 1, "accepted": int(accept_l), **delta_l})
            obs = ensemble_observables(state["phi"], ctx["fine_action"])
            local = {k: float(v[0]) for k, v in local_observable_series(state["phi"], ctx["fine_action"]).items()}
            obs_rows.append({"chain_id": chain, "sweep": sweep, **{k: float(v) for k, v in obs.items()}, **local})
            sweep_times.append(time.perf_counter() - t_sweep)
    wall = time.perf_counter() - t_all
    write_csv(out / "tiny_smoke_coarse_deltas.csv", coarse_rows)
    write_csv(out / "tiny_smoke_latent_deltas.csv", latent_rows)
    write_csv(out / "tiny_smoke_observable_timeseries.csv", obs_rows)
    summary = {
        "status": "completed",
        "initial_coarse_indices": initial_indices,
        "expected_counts": {
            "coarse_attempts": chains * sweeps * n_patch,
            "latent_pcn_attempts": chains * sweeps,
            "measured_rows": chains * sweeps,
        },
        "actual_counts": {
            "coarse_attempts": len(coarse_rows),
            "latent_pcn_attempts": len(latent_rows),
            "measured_rows": len(obs_rows),
        },
        **summarize_moves(coarse_rows, "coarse"),
        **summarize_moves(latent_rows, "latent"),
        "wall_time_sec": wall,
        "wall_time_per_sweep_sec": float(wall / max(chains * sweeps, 1)),
        "sweep_time_mean_sec": float(np.mean(sweep_times)),
        "sweep_time_std_sec": float(np.std(sweep_times, ddof=1)) if len(sweep_times) > 1 else 0.0,
        "max_rss_mb": max_rss_mb(),
        "nan_failures": int(
            sum(not finite_delta(r) for r in coarse_rows)
            + sum(not finite_delta(r) for r in latent_rows)
            + sum(not np.isfinite(float(r["action_density"])) for r in obs_rows)
        ),
        "rough_observables_mean": {
            k: float(np.mean([float(r[k]) for r in obs_rows]))
            for k in ["m", "abs_m", "phi2", "phi4", "NN", "2NN", "diag", "action_density", "susceptibility", "Binder_U4", "xi_over_L"]
        },
        "rough_observables_note": "Generated L64 tiny-smoke diagnostics; use local observables only for reference comparison.",
    }
    write_json(out / "tiny_smoke_summary.json", summary)
    return summary


def write_reports(out: Path, preflight: dict[str, Any], smoke: dict[str, Any] | None) -> None:
    pf_lines = [
        "# L32->L64 preflight report",
        "",
        f"Status: `{preflight['status']}`",
        "",
        f"- coarse starts: `{preflight['coarse_start_path']}`",
        f"- coarse shape: `{preflight['coarse_shape']}`",
        f"- generator: `{preflight['coarse_manifest'].get('generator', 'unknown')}`",
        f"- lambda/kappa metadata: `{preflight['coarse_manifest'].get('lambda')}` / `{preflight['coarse_manifest'].get('kappa')}`",
        f"- N_patch/sweep: `{preflight['expected_N_patch_per_sweep']}`",
        f"- component instantiation: `{preflight['bundle_preflight'].get('dummy_L16_to_L32_instantiation')}` at coarse_L=32",
        f"- generated phi shape: `{preflight['state_checks']['phi_shape']}`",
        f"- reblocking max abs error: `{preflight['state_checks']['reblocking_max_abs_error']:.6g}`",
        f"- one coarse patch delta_logw: `{preflight['proposal_checks']['coarse_patch_delta']['delta_logw']:.6g}`",
        f"- one latent pCN delta_logw: `{preflight['proposal_checks']['latent_pcn_delta']['delta_logw']:.6g}`",
        f"- max RSS after preflight MB: `{preflight['max_rss_mb_after_preflight']:.3f}`",
        "",
        "Coarse observable quick summary:",
        "",
        "```json",
        json.dumps(preflight["coarse_observable_summary"], indent=2, sort_keys=True, default=float),
        "```",
    ]
    (out / "L32_to_L64_preflight_report.md").write_text("\n".join(pf_lines) + "\n")

    if smoke is not None:
        sm_lines = [
            "# L32->L64 tiny smoke report",
            "",
            "This is a 1 chain x 20 sweep mechanics smoke, not production validation.",
            "",
            f"- coarse attempts: `{smoke['actual_counts']['coarse_attempts']}`",
            f"- latent pCN attempts: `{smoke['actual_counts']['latent_pcn_attempts']}`",
            f"- measured rows: `{smoke['actual_counts']['measured_rows']}`",
            f"- coarse acceptance: `{smoke['coarse_acceptance']:.6g}`",
            f"- coarse delta-logw std: `{smoke['coarse_std_delta_logw']:.6g}`",
            f"- latent pCN acceptance: `{smoke['latent_acceptance']:.6g}`",
            f"- latent delta-logw std: `{smoke['latent_std_delta_logw']:.6g}`",
            f"- wall time per sweep sec: `{smoke['wall_time_per_sweep_sec']:.6g}`",
            f"- max RSS MB: `{smoke['max_rss_mb']:.3f}`",
            f"- NaN/failure count: `{smoke['nan_failures']}`",
            "",
            "Rough generated L64 observables:",
            "",
            "```json",
            json.dumps(smoke["rough_observables_mean"], indent=2, sort_keys=True, default=float),
            "```",
        ]
        (out / "L32_to_L64_tiny_smoke_report.md").write_text("\n".join(sm_lines) + "\n")

    status_lines = [
        "# L32->L64 smoke status",
        "",
        f"- Were L32 starts found? `yes`: `{preflight['coarse_start_path']}`",
        f"- Did preflight pass? `{preflight['status'] == 'passed'}`",
    ]
    if smoke is None:
        status_lines.extend([
            "- What was A/R scale? `not run; preflight failed`",
            "- Is L32->L64 technically viable? `no conclusion`",
            "- Should we consider a short follow-up? `no, fix preflight first`",
        ])
    else:
        viable = preflight["status"] == "passed" and smoke["nan_failures"] == 0
        promising = viable and smoke["coarse_acceptance"] > 0.05 and math.isfinite(smoke["coarse_std_delta_logw"])
        status_lines.extend([
            f"- What was A/R scale? coarse acceptance `{smoke['coarse_acceptance']:.6g}`, coarse std Delta logw `{smoke['coarse_std_delta_logw']:.6g}`, latent acceptance `{smoke['latent_acceptance']:.6g}`, latent std Delta logw `{smoke['latent_std_delta_logw']:.6g}`",
            f"- Is L32->L64 technically viable? `{viable}` for mechanics; no production claim.",
            f"- Should we consider a short follow-up? `{promising}`. Manual-only suggestion: 1x100 first, then 2x100 if A/R and wall time remain acceptable.",
            "",
            "Manual follow-up command, not run:",
            "",
            "```bash",
            "../.venv/bin/python -B perfect_blocking_upsampling/scripts/run_shape_parametric_sampler_validation.py \\",
            "  --config perfect_blocking_upsampling/outputs/shape_parametric_sampler_validation/L32_to_L64_smoke/L32_to_L64_smoke_config.yaml \\",
            "  --output-dir perfect_blocking_upsampling/outputs/shape_parametric_sampler_validation/L32_to_L64_smoke/manual_1x100 \\",
            "  --coarse-L 32 --patch-size 4 --origin-mode random \\",
            "  --smoke-sweeps 100 --validation-chains 1 \\",
            "  --pcn-rho 0.5 --pcn-interval-sweeps 1 \\",
            "  --measurement-mode end_of_sweep --coarse-start-mode thermalized_coarse \\",
            "  --detail-warmup-sweeps 0",
            "```",
            "",
            "Note: the standard validation driver expects a fine reference for reference-z diagnostics; for L32->L64, use a custom no-reference follow-up until a native L64 reference exists.",
        ])
    (out / "STATUS.md").write_text("\n".join(status_lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--smoke-sweeps", type=int, default=20)
    ap.add_argument("--validation-chains", type=int, default=1)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    seed = int(cfg.get("random_seed", 20260701))
    preflight, ctx, coarse = run_preflight(cfg, args.out_dir, seed)
    smoke = None
    if preflight["status"] == "passed" and not args.preflight_only:
        smoke = run_tiny_smoke(coarse, ctx, args.out_dir, seed, sweeps=args.smoke_sweeps, chains=args.validation_chains)
    write_reports(args.out_dir, preflight, smoke)
    print(json.dumps({"preflight": preflight["status"], "smoke": None if smoke is None else smoke["status"]}, indent=2))
    return 0 if preflight["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
