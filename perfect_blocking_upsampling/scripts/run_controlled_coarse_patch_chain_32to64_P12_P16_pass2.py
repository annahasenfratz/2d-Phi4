#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
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
import run_l16to32_controlled_chain_footprint as fp_sampler  # noqa: E402
import train_l16to32_footprint_candidate as fp_train  # noqa: E402
from _common import format_float_tag, load_config, load_frozen_models, load_kernel_spec, override_validation_config, resolve_run_paths  # noqa: E402
from perfect_blocking_upsampling.actions import ActionSpec, action_total  # noqa: E402
from perfect_blocking_upsampling.observables import observables as ensemble_observables  # noqa: E402
from train_finite_footprint_transported_detail import patch_sites  # noqa: E402

DEFAULT_CONFIG = PKG / "outputs" / "shape_parametric_sampler_validation" / "L32_to_L64_smoke" / "L32_to_L64_smoke_config.yaml"
OBS_KEYS = ["phi2", "phi4", "NN", "2nn", "diag", "action_density", "xi_over_L", "abs_m", "m2", "m4", "chi"]


def default_save_sweeps(limit: int) -> list[int]:
    return [0] + list(range(5, int(limit) + 1, 5))


def normalize_save_sweeps(
    save_sweeps: list[int] | None,
    *,
    resumed_from_checkpoint: bool,
    added_sweeps: int,
    completed_sweeps: int = 0,
) -> list[int]:
    if save_sweeps is None:
        limit = completed_sweeps + added_sweeps if resumed_from_checkpoint else added_sweeps
        return default_save_sweeps(limit)
    sweeps = sorted(set(int(x) for x in save_sweeps))
    if resumed_from_checkpoint and sweeps == default_save_sweeps(added_sweeps):
        return default_save_sweeps(completed_sweeps + added_sweeps)
    return sweeps


def configure_stdio() -> None:
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    try:
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass


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


def merge_rows(existing: list[dict[str, Any]], incoming: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[tuple[tuple[str, str], ...]] = set()

    def sig(row: dict[str, Any]) -> tuple[tuple[str, str], ...]:
        return tuple(sorted((str(k), json.dumps(v, sort_keys=True, default=float)) for k, v in row.items()))

    for row in existing + incoming:
        key = sig(row)
        if key in seen:
            continue
        seen.add(key)
        merged.append(row)
    return merged


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=float) + "\n", encoding="utf-8")


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def flush_history_tables(
    out_dir: Path,
    patch_rows: list[dict[str, Any]],
    accept_history: list[dict[str, Any]],
    obs_history: list[dict[str, Any]],
    *,
    ref_by_key: dict[str, dict[str, Any]] | None = None,
    coarse_L: int | None = None,
    fine_L: int | None = None,
    detail_patch_size: int | None = None,
    detail_updates_per_sweep: int | None = None,
) -> None:
    write_csv(out_dir / "controlled_patch_chain_patch_move_summary.csv", patch_rows)
    write_csv(out_dir / "controlled_patch_chain_acceptance_history.csv", accept_history)
    write_csv(out_dir / "controlled_patch_chain_observable_history.csv", obs_history)
    if ref_by_key is not None:
        obs_summary = summarize_observable_history(obs_history, ref_by_key)
        write_csv(out_dir / "controlled_patch_chain_local_operator_summary.csv", obs_summary)
    if coarse_L is not None and fine_L is not None and detail_patch_size is not None and detail_updates_per_sweep is not None:
        export_raw_data_tables(
            out_dir,
            obs_history,
            coarse_L=coarse_L,
            fine_L=fine_L,
            detail_patch_size=detail_patch_size,
            detail_updates_per_sweep=detail_updates_per_sweep,
        )


def export_raw_data_tables(
    out_dir: Path,
    obs_history: list[dict[str, Any]],
    *,
    coarse_L: int,
    fine_L: int,
    detail_patch_size: int,
    detail_updates_per_sweep: int,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in obs_history:
        grouped.setdefault((int(row["patch_size"]), int(row["n_coarse_passes"])), []).append(row)

    manifest_rows: list[dict[str, Any]] = []
    for (patch_size, n_passes), rows in sorted(grouped.items()):
        filename = f"data_{coarse_L}to{fine_L}_P{patch_size}_pass{n_passes}_detail{detail_updates_per_sweep}.csv"
        path = out_dir / filename
        rows_to_write = merge_rows(read_csv_rows(path), rows) if path.exists() else rows
        write_csv(path, rows_to_write)
        manifest_rows.append(
            {
                "patch_size": patch_size,
                "n_coarse_passes": n_passes,
                "detail_patch_size": detail_patch_size,
                "detail_updates_per_sweep": detail_updates_per_sweep,
                "rows": len(rows_to_write),
                "csv": filename,
            }
        )
    if manifest_rows:
        manifest_path = out_dir / f"data_{coarse_L}to{fine_L}_manifest.csv"
        manifest_rows = merge_rows(read_csv_rows(manifest_path), manifest_rows) if manifest_path.exists() else manifest_rows
        write_csv(manifest_path, manifest_rows)
    if len(grouped) > 1:
        all_path = out_dir / f"data_{coarse_L}to{fine_L}_all_settings.csv"
        all_rows = merge_rows(read_csv_rows(all_path), obs_history) if all_path.exists() else obs_history
        write_csv(all_path, all_rows)
    return manifest_rows


def serialize_rng_state(rng: np.random.Generator) -> np.ndarray:
    payload = pickle.dumps(rng.bit_generator.state)
    return np.frombuffer(payload, dtype=np.uint8).copy()


def deserialize_rng_state(payload: np.ndarray) -> np.random.Generator:
    rng = np.random.default_rng()
    rng.bit_generator.state = pickle.loads(np.asarray(payload, dtype=np.uint8).tobytes())
    return rng


def save_checkpoint(
    path: Path,
    *,
    completed_sweeps: int,
    chain_states: list[dict[str, Any]],
    chain_rngs: list[np.random.Generator],
    args: argparse.Namespace,
    settings: list[tuple[int, int]],
    sweep_label: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "completed_sweeps": int(completed_sweeps),
        "sweep_label": int(sweep_label),
        "n_chains": int(len(chain_states)),
        "coarse_L": int(args.coarse_L),
        "fine_L": int(args.fine_L),
        "lambda": float(args.lambda_),
        "kappa_c": float(args.kappa_c),
        "kappa_f": float(args.kappa_f),
        "site_step_size": float(args.site_step_size),
        "detail_patch_size": int(args.detail_patch_size),
        "latent_beta": float(args.latent_beta),
        "latent_updates_per_sweep": int(args.latent_updates_per_sweep),
        "settings": [[int(p), int(n)] for p, n in settings],
    }
    arrays: dict[str, np.ndarray] = {}
    for idx, (state, rng) in enumerate(zip(chain_states, chain_rngs)):
        arrays[f"u_{idx}"] = np.asarray(state["u"], dtype=np.float32)
        arrays[f"z_edge_{idx}"] = np.asarray(state["z_edge"], dtype=np.float32)
        arrays[f"z_pair_{idx}"] = np.asarray(state["z_pair"], dtype=np.float32)
        arrays[f"z_corner_{idx}"] = np.asarray(state["z_corner"], dtype=np.float32)
        arrays[f"phi_{idx}"] = np.asarray(state["phi"], dtype=np.float32)
        arrays[f"sf_{idx}"] = np.asarray(state["sf"], dtype=np.float64)
        arrays[f"sc_{idx}"] = np.asarray(state["sc"], dtype=np.float64)
        arrays[f"logdet_{idx}"] = np.asarray(state["logdet"], dtype=np.float64)
        arrays[f"logq_{idx}"] = np.asarray(state["logq"], dtype=np.float64)
        arrays[f"logw_{idx}"] = np.asarray(state["logw"], dtype=np.float64)
        arrays[f"source_coarse_index_{idx}"] = np.asarray([int(state.get("source_coarse_index", -1))], dtype=np.int64)
        arrays[f"rng_state_{idx}"] = serialize_rng_state(rng)
    np.savez_compressed(path, meta=np.array(json.dumps(payload), dtype=np.str_), **arrays)


def load_checkpoint(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[np.random.Generator]]:
    data = np.load(path, allow_pickle=True)
    meta_raw = data["meta"]
    if isinstance(meta_raw, np.ndarray):
        meta_text = str(meta_raw.item())
    else:
        meta_text = str(meta_raw)
    meta = json.loads(meta_text)
    n_chains = int(meta["n_chains"])
    chain_states: list[dict[str, Any]] = []
    chain_rngs: list[np.random.Generator] = []
    for idx in range(n_chains):
        state = {
            "u": np.asarray(data[f"u_{idx}"], dtype=np.float32),
            "z_edge": np.asarray(data[f"z_edge_{idx}"], dtype=np.float32),
            "z_pair": np.asarray(data[f"z_pair_{idx}"], dtype=np.float32),
            "z_corner": np.asarray(data[f"z_corner_{idx}"], dtype=np.float32),
            "phi": np.asarray(data[f"phi_{idx}"], dtype=np.float32),
            "sf": np.asarray(data[f"sf_{idx}"], dtype=np.float64),
            "sc": np.asarray(data[f"sc_{idx}"], dtype=np.float64),
            "logdet": np.asarray(data[f"logdet_{idx}"], dtype=np.float64),
            "logq": np.asarray(data[f"logq_{idx}"], dtype=np.float64),
            "logw": np.asarray(data[f"logw_{idx}"], dtype=np.float64),
            "source_coarse_index": int(np.asarray(data[f"source_coarse_index_{idx}"], dtype=np.int64).reshape(-1)[0]),
        }
        chain_states.append(state)
        chain_rngs.append(deserialize_rng_state(np.asarray(data[f"rng_state_{idx}"], dtype=np.uint8)))
    return meta, chain_states, chain_rngs


def load_ctx(cfg: dict[str, Any], args: argparse.Namespace | None = None) -> dict[str, Any]:
    if args is not None and args.l16to32_footprint_checkpoint_root is not None:
        if int(args.coarse_L) != 16 or int(args.fine_L) != 32:
            raise RuntimeError("--l16to32-footprint-checkpoint-root is only valid for L16->L32")
        checkpoint_root = args.l16to32_footprint_checkpoint_root.resolve()
        cand_cfg = fp_train.CandidateConfig(
            candidate="old_controlled_chain_with_l16to32_footprint_kernel",
            footprint=int(args.l16to32_footprint),
            max_train=800,
            max_val=200,
            output_dir=str(args.out_dir),
        )
        models = {
            "edge_x": fp_sampler.load_stage_model("edge_x", checkpoint_root / "checkpoints/edge_x/checkpoint_best.pt", cand_cfg),
            "edge_y": fp_sampler.load_stage_model("edge_y", checkpoint_root / "checkpoints/edge_y/checkpoint_best.pt", cand_cfg),
            "body": fp_sampler.load_stage_model("body", checkpoint_root / "checkpoints/body/checkpoint_best.pt", cand_cfg),
        }
        kernel, _ = fp_train.load_kernel(fp_train.KERNEL)
        coarse_action = ActionSpec("phi4_nn", float(args.lambda_), float(args.kappa_c))
        fine_action = ActionSpec("phi4_nn", float(args.lambda_), float(args.kappa_f))
        return {
            "sampler_kind": "l16to32_footprint",
            "models": models,
            "candidate_config": cand_cfg,
            "kernel": kernel,
            "coarse_action": coarse_action,
            "fine_action": fine_action,
            "checkpoint_root": str(checkpoint_root),
        }
    refine_model, refine_state, stages, coarse_action, fine_action, _ = load_frozen_models(cfg)
    refine_model.load_state_dict(refine_state, strict=False)
    refine_model.eval()
    for model, _, state, *_ in stages.values():
        model.load_state_dict(state)
        model.eval()
    kernel, _ = load_kernel_spec(cfg)
    return {"sampler_kind": "shape_parametric", "refine_model": refine_model, "stages": stages, "coarse_action": coarse_action, "fine_action": fine_action, "kernel": kernel}


def sample_chain_z(rng: np.random.Generator, n: int, coarse_L: int, ctx: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if ctx.get("sampler_kind") == "l16to32_footprint":
        return (
            rng.standard_normal((n, coarse_L, coarse_L)).astype(np.float32),
            rng.standard_normal((n, coarse_L, coarse_L)).astype(np.float32),
            rng.standard_normal((n, coarse_L, coarse_L)).astype(np.float32),
        )
    return sampler.sample_z(rng, n, coarse_L)


def compute_chain_state(
    u: np.ndarray,
    z_edge: np.ndarray,
    z_pair: np.ndarray,
    z_corner: np.ndarray,
    ctx: dict[str, Any],
) -> dict[str, Any]:
    if ctx.get("sampler_kind") == "l16to32_footprint":
        state = fp_sampler.compute_state(
            u,
            z_edge,
            z_pair,
            z_corner,
            ctx["models"],
            ctx["kernel"],
            ctx["candidate_config"],
        )
        state["z_corner"] = state.pop("z_body")
        state["logdet"] = np.zeros_like(state["logw"], dtype=np.float64)
        return state
    return sampler.compute_state(u, z_edge, z_pair, z_corner, ctx)


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


def local_delta_action_site(field: np.ndarray, i: int, j: int, new_x: float, action: ActionSpec) -> float:
    old_x = float(field[i, j])
    lam = float(action.lambda_)
    kap = float(action.kappa)
    onsite_old = (1.0 - 2.0 * lam) * old_x * old_x + lam * old_x**4
    onsite_new = (1.0 - 2.0 * lam) * new_x * new_x + lam * new_x**4
    nn = float(
        field[(i + 1) % field.shape[0], j]
        + field[(i - 1) % field.shape[0], j]
        + field[i, (j + 1) % field.shape[1]]
        + field[i, (j - 1) % field.shape[1]]
    )
    return (onsite_new - onsite_old) - 2.0 * kap * (new_x - old_x) * nn


def controlled_patch_metropolis(
    u: np.ndarray,
    sites: list[tuple[int, int]],
    rng: np.random.Generator,
    action: ActionSpec,
    step_size: float,
    sweeps: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    current = u.copy()
    before = current.copy()
    sc_before = float(action_total(current, action))
    attempted = 0
    accepted = 0
    for _ in range(sweeps):
        order = list(sites)
        rng.shuffle(order)
        for i, j in order:
            old = float(current[i, j])
            new = old + float(step_size * rng.standard_normal())
            dsc = local_delta_action_site(current, i, j, new, action)
            acc = math.log(max(float(rng.random()), 1.0e-300)) < min(0.0, -dsc)
            attempted += 1
            if acc:
                current[i, j] = new
                accepted += 1
    sc_after = float(action_total(current, action))
    patch_delta = current - before
    return current.astype(np.float32), {
        "attempted_site_updates": attempted,
        "accepted_site_updates": accepted,
        "coarse_site_acceptance": accepted / max(attempted, 1),
        "delta_Sc_patch": sc_after - sc_before,
        "patch_l2_change": float(np.sqrt(np.sum(patch_delta * patch_delta))),
        "patch_norm2_change": float(np.sum(patch_delta * patch_delta)),
        "patch_linf_change": float(np.max(np.abs(patch_delta))) if patch_delta.size else 0.0,
    }


def propose_latent_pcn_patch(
    state: dict[str, Any],
    x0: int,
    y0: int,
    patch_size: int,
    rng: np.random.Generator,
    ctx: dict[str, Any],
    beta: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    rho = math.sqrt(max(0.0, 1.0 - beta * beta))
    sites = patch_sites(state["u"].shape[1], x0, y0, patch_size)
    z_edge = state["z_edge"].copy()
    z_pair = state["z_pair"].copy()
    z_corner = state["z_corner"].copy()
    for i, j in sites:
        if z_edge.ndim == 4:
            z_edge[0, 0, i, j] = rho * z_edge[0, 0, i, j] + beta * float(rng.standard_normal())
            z_pair[0, 0, i, j] = rho * z_pair[0, 0, i, j] + beta * float(rng.standard_normal())
            z_corner[0, 0, i, j] = rho * z_corner[0, 0, i, j] + beta * float(rng.standard_normal())
        else:
            z_edge[0, i, j] = rho * z_edge[0, i, j] + beta * float(rng.standard_normal())
            z_pair[0, i, j] = rho * z_pair[0, i, j] + beta * float(rng.standard_normal())
            z_corner[0, i, j] = rho * z_corner[0, i, j] + beta * float(rng.standard_normal())
    proposal = compute_chain_state(state["u"], z_edge, z_pair, z_corner, ctx)
    delta_phi = proposal["phi"][0] - state["phi"][0]
    return proposal, {
        "patch_x": x0,
        "patch_y": y0,
        "sites_per_patch": patch_size * patch_size,
        "latent_beta": beta,
        "latent_rho": rho,
        "fine_log_accept": float(proposal["logw"][0] - state["logw"][0]),
        "delta_Sf": float(proposal["sf"][0] - state["sf"][0]),
        "delta_Sc_patch": 0.0,
        "delta_logdet_refine": float(proposal["logdet"][0] - state["logdet"][0]),
        "delta_logq_detail": float(proposal["logq"][0] - state["logq"][0]),
        "patch_l2_change": float(np.sqrt(np.sum(delta_phi * delta_phi))),
        "patch_norm2_change": float(np.sum(delta_phi * delta_phi)),
        "patch_linf_change": float(np.max(np.abs(delta_phi))) if delta_phi.size else 0.0,
    }


def patches_per_sweep(lc: int, patch_size: int) -> int:
    return int(math.ceil(2.0 * lc * lc / float(patch_size * patch_size)))


def local_observable_row(phi: np.ndarray, action: ActionSpec) -> dict[str, float]:
    arr = np.asarray(phi, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr[None]
    obs = ensemble_observables(arr, action)
    m = np.mean(arr, axis=(1, 2))
    m2_series = m * m
    m4_series = m**4
    phi2 = float(np.mean(arr * arr))
    twonn = 0.5 * (
        np.mean(arr * np.roll(arr, -2, axis=1), axis=(1, 2))
        + np.mean(arr * np.roll(arr, -2, axis=2), axis=(1, 2))
    )
    action_density = action_total(arr, action) / (arr.shape[1] * arr.shape[2])
    m2 = float(np.mean(m2_series))
    m4 = float(np.mean(m4_series))
    chi = float(arr.shape[1] * arr.shape[2] * m2)
    xi_over_l = float(math.sqrt(max(chi, 0.0) / max(phi2, 1.0e-300)) / arr.shape[1])
    return {
        "phi2": float(obs["phi2"]),
        "phi4": float(obs["phi4"]),
        "NN": float(obs["NN"]),
        "2nn": float(np.mean(twonn)),
        "diag": float(obs["diag"]),
        "action_density": float(np.mean(action_density)),
        "xi_over_L": xi_over_l,
        "m": float(obs["m"]),
        "abs_m": float(obs["abs_m"]),
        "m2": m2,
        "m4": m4,
        "chi": chi,
    }


def reference_rows(fine_ref: np.ndarray, action: ActionSpec) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if fine_ref.size == 0:
        return rows
    obs_rows = [local_observable_row(fine_ref[i : i + 1], action) for i in range(fine_ref.shape[0])]
    for key in OBS_KEYS:
        vals = np.asarray([r[key] for r in obs_rows], dtype=np.float64)
        rows.append({
            "observable": key,
            "native_value": float(np.mean(vals)),
            "native_SE": float(np.std(vals, ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else float("nan"),
            "native_N": int(len(vals)),
        })
    return rows


def summarize_observable_history(rows: list[dict[str, Any]], ref_by_key: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    groups = sorted({(int(r["patch_size"]), int(r["n_coarse_passes"]), int(r["sweep"])) for r in rows})
    for patch_size, n_passes, sweep in groups:
        sub = [r for r in rows if int(r["patch_size"]) == patch_size and int(r["n_coarse_passes"]) == n_passes and int(r["sweep"]) == sweep]
        for key in OBS_KEYS:
            vals = np.asarray([float(r[key]) for r in sub], dtype=np.float64)
            ref = ref_by_key.get(key, {})
            mean = float(np.mean(vals))
            se = float(np.std(vals, ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else float("nan")
            native = float(ref.get("native_value", float("nan")))
            native_se = float(ref.get("native_SE", float("nan")))
            denom = math.sqrt(se * se + native_se * native_se) if np.isfinite(native_se) else float("nan")
            out.append({
                "patch_size": patch_size,
                "n_coarse_passes": n_passes,
                "sweep": sweep,
                "observable": key,
                "mean": mean,
                "SE_over_chains": se,
                "n_chains": len(vals),
                "native_value": native,
                "native_SE": native_se,
                "diff_vs_native": mean - native if np.isfinite(native) else float("nan"),
                "z_vs_native": (mean - native) / denom if denom and np.isfinite(denom) and denom > 0 else float("nan"),
            })
    return out


def summarize_moves(
    rows: list[dict[str, Any]],
    settings: list[tuple[int, int]],
    chains: int,
    sweeps: int,
    runtime_by_setting: dict[tuple[int, int], float],
    lc: int,
) -> list[dict[str, Any]]:
    out = []
    for patch_size, n_passes in settings:
        sub = [r for r in rows if int(r["patch_size"]) == patch_size and int(r["n_coarse_passes"]) == n_passes and r["move_type"] == "controlled_coarse_patch"]
        if not sub:
            continue
        site_attempts = int(sum(int(r["attempted_site_updates"]) for r in sub))
        site_accepts = int(sum(int(r["accepted_site_updates"]) for r in sub))
        fine_accepts = int(sum(int(r["fine_AR_accept"]) for r in sub))
        dlog = [float(r["fine_log_accept"]) for r in sub]
        dsc = [float(r["delta_Sc_patch"]) for r in sub]
        l2 = [float(r["patch_l2_change"]) for r in sub]
        accepted_per_patch = [float(r["accepted_site_updates"]) for r in sub]
        qs_dlog = qstats(dlog)
        qs_dsc = qstats(dsc)
        qs_l2 = qstats(l2)
        wall = float(runtime_by_setting.get((patch_size, n_passes), float("nan")))
        out.append({
            "move_type": "controlled_coarse_patch",
            "patch_size": patch_size,
            "n_coarse_passes": n_passes,
            "chains": chains,
            "sweeps": sweeps,
            "patches_per_sweep": patches_per_sweep(lc, patch_size),
            "total_patch_proposals": len(sub),
            "site_attempts_total": site_attempts,
            "site_accepts_total": site_accepts,
            "coarse_site_acceptance": site_accepts / max(site_attempts, 1),
            "accepted_site_updates_per_patch_mean": float(np.mean(accepted_per_patch)),
            "accepted_site_updates_per_patch_std": float(np.std(accepted_per_patch, ddof=1)),
            "patch_l2_mean": qs_l2["mean"],
            "patch_l2_std": qs_l2["std"],
            "patch_l2_q05": qs_l2["q05"],
            "patch_l2_q50": qs_l2["q50"],
            "patch_l2_q95": qs_l2["q95"],
            "patch_delta_Sc_mean": qs_dsc["mean"],
            "patch_delta_Sc_std": qs_dsc["std"],
            "patch_delta_Sc_q05": qs_dsc["q05"],
            "patch_delta_Sc_q50": qs_dsc["q50"],
            "patch_delta_Sc_q95": qs_dsc["q95"],
            "fine_AR_attempts": len(sub),
            "fine_AR_accepts": fine_accepts,
            "fine_AR_acceptance": fine_accepts / max(len(sub), 1),
            "fine_log_accept_mean": qs_dlog["mean"],
            "fine_log_accept_std": qs_dlog["std"],
            "fine_log_accept_q05": qs_dlog["q05"],
            "fine_log_accept_q50": qs_dlog["q50"],
            "fine_log_accept_q95": qs_dlog["q95"],
            "runtime_sec": wall,
            "sec_per_chain_sweep": wall / max(chains * sweeps, 1),
        })
        lat = [r for r in rows if int(r["patch_size"]) == patch_size and int(r["n_coarse_passes"]) == n_passes and r["move_type"] == "latent_detail_pCN"]
        if lat:
            dlog_lat = [float(r["fine_log_accept"]) for r in lat]
            qs_lat = qstats(dlog_lat)
            out.append({
                "move_type": "latent_detail_pCN",
                "patch_size": int(lat[0].get("detail_patch_size", lat[0]["patch_size"])),
                "n_coarse_passes": n_passes,
                "chains": chains,
                "sweeps": sweeps,
                "patches_per_sweep": float("nan"),
                "total_patch_proposals": len(lat),
                "site_attempts_total": 0,
                "site_accepts_total": 0,
                "coarse_site_acceptance": float("nan"),
                "accepted_site_updates_per_patch_mean": float("nan"),
                "accepted_site_updates_per_patch_std": float("nan"),
                "patch_l2_mean": float(np.mean([float(r["patch_l2_change"]) for r in lat])),
                "patch_l2_std": float(np.std([float(r["patch_l2_change"]) for r in lat], ddof=1)) if len(lat) > 1 else 0.0,
                "patch_l2_q05": float(np.quantile([float(r["patch_l2_change"]) for r in lat], 0.05)),
                "patch_l2_q50": float(np.quantile([float(r["patch_l2_change"]) for r in lat], 0.50)),
                "patch_l2_q95": float(np.quantile([float(r["patch_l2_change"]) for r in lat], 0.95)),
                "patch_delta_Sc_mean": 0.0,
                "patch_delta_Sc_std": 0.0,
                "patch_delta_Sc_q05": 0.0,
                "patch_delta_Sc_q50": 0.0,
                "patch_delta_Sc_q95": 0.0,
                "fine_AR_attempts": len(lat),
                "fine_AR_accepts": int(sum(int(r["fine_AR_accept"]) for r in lat)),
                "fine_AR_acceptance": float(np.mean([int(r["fine_AR_accept"]) for r in lat])),
                "fine_log_accept_mean": qs_lat["mean"],
                "fine_log_accept_std": qs_lat["std"],
                "fine_log_accept_q05": qs_lat["q05"],
                "fine_log_accept_q50": qs_lat["q50"],
                "fine_log_accept_q95": qs_lat["q95"],
                "runtime_sec": wall,
                "sec_per_chain_sweep": wall / max(chains * sweeps, 1),
            })
    return out


def setting_acceptance_snapshot(
    rows: list[dict[str, Any]],
    *,
    patch_size: int,
    n_passes: int,
) -> dict[str, Any]:
    coarse = [
        r
        for r in rows
        if int(r["patch_size"]) == patch_size
        and int(r["n_coarse_passes"]) == n_passes
        and r["move_type"] == "controlled_coarse_patch"
    ]
    latent = [
        r
        for r in rows
        if int(r["patch_size"]) == patch_size
        and int(r["n_coarse_passes"]) == n_passes
        and r["move_type"] == "latent_detail_pCN"
    ]
    coarse_attempts = len(coarse)
    latent_attempts = len(latent)
    return {
        "patch_size": patch_size,
        "n_coarse_passes": n_passes,
        "coarse_attempts": coarse_attempts,
        "coarse_acceptance": float(np.mean([int(r["fine_AR_accept"]) for r in coarse])) if coarse_attempts else float("nan"),
        "coarse_site_acceptance": float(
            sum(int(r["accepted_site_updates"]) for r in coarse) / max(sum(int(r["attempted_site_updates"]) for r in coarse), 1)
        )
        if coarse_attempts
        else float("nan"),
        "latent_attempts": latent_attempts,
        "latent_acceptance": float(np.mean([int(r["fine_AR_accept"]) for r in latent])) if latent_attempts else float("nan"),
        "latent_log_accept_mean": float(np.mean([float(r["fine_log_accept"]) for r in latent])) if latent_attempts else float("nan"),
        "latent_log_accept_std": float(np.std([float(r["fine_log_accept"]) for r in latent], ddof=1)) if latent_attempts > 1 else float("nan"),
    }


def local_rms_relative(row: dict[str, float], ref_by_key: dict[str, dict[str, Any]]) -> float:
    terms = []
    for key in ["phi2", "phi4", "NN", "2nn", "diag"]:
        ref = ref_by_key.get(key)
        if ref is None:
            return float("nan")
        denom = max(abs(float(ref["native_value"])), 1.0e-12)
        terms.append(((float(row[key]) - float(ref["native_value"])) / denom) ** 2)
    return float(math.sqrt(sum(terms) / len(terms)))


def chain_rms_snapshot(chain_states: list[dict[str, Any]], action: ActionSpec, ref_by_key: dict[str, dict[str, Any]]) -> float:
    vals = []
    for state in chain_states:
        row = local_observable_row(state["phi"], action)
        vals.append(local_rms_relative(row, ref_by_key))
    finite = [v for v in vals if np.isfinite(v)]
    return float(np.mean(finite)) if finite else float("nan")


def write_report(
    out: Path,
    summary_rows: list[dict[str, Any]],
    obs_summary: list[dict[str, Any]],
    ref_rows: list[dict[str, Any]],
    cfg_summary: dict[str, Any],
    report_name: str,
) -> None:
    lines = [
        f"# Controlled coarse patch chain {cfg_summary['coarse_L']}->{cfg_summary['fine_L']}, P12/P16 pass2",
        "",
        "This is a real controlled patch-chain diagnostic, not a one-shot proposal A/R scan. The coarse patch proposal is valid because each selected patch is updated by site-by-site Metropolis targeting `S_c`; the final fine/transformed A/R is then applied with the latent/detail field fixed during each proposal.",
        "",
        "Previous uncontrolled coarse-patch runs are diagnostic-only and are not used as validation here.",
        "",
        "## Setup",
        "",
        f"- config: `{cfg_summary['config']}`",
        f"- coarse ensemble: `{cfg_summary['coarse_ensemble']}`",
        f"- fine reference: `{cfg_summary.get('fine_reference', '')}`",
        f"- chains: `{cfg_summary['chains']}`",
        f"- sweeps: `{cfg_summary['sweeps']}`",
        f"- saved sweeps: `{cfg_summary['save_sweeps']}`",
        f"- site step: `{cfg_summary['site_step_size']}`",
        "",
        "## Acceptance Summary",
        "",
        "| P | passes | patches/sweep | proposals | site acc | accepted sites/patch | patch L2 | dSc std | fine A/R | logacc mean | logacc std | sec/chain-sweep |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary_rows:
        lines.append(
            f"| {r['patch_size']} | {r['n_coarse_passes']} | {r['patches_per_sweep']} | {r['total_patch_proposals']} | "
            f"{r['coarse_site_acceptance']:.6g} | {r['accepted_site_updates_per_patch_mean']:.6g} | {r['patch_l2_mean']:.6g} | "
            f"{r['patch_delta_Sc_std']:.6g} | {r['fine_AR_acceptance']:.6g} | {r['fine_log_accept_mean']:.6g} | "
            f"{r['fine_log_accept_std']:.6g} | {r['sec_per_chain_sweep']:.6g} |"
        )
    lines += [
        "",
        "## Native Reference",
        "",
        "| observable | native value | native SE | N |",
        "|---|---:|---:|---:|",
    ]
    for r in ref_rows:
        lines.append(f"| {r['observable']} | {r['native_value']:.6g} | {r['native_SE']:.6g} | {r['native_N']} |")
    lines += [
        "",
        "## Saved-Sweep Local/Action Observables",
        "",
        "Rows below are chain means with standard errors over chains. Full table is in `controlled_patch_chain_local_operator_summary.csv`.",
        "",
        "| P | passes | sweep | observable | mean | SE | native | z |",
        "|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    primary = {"phi2", "phi4", "NN", "2nn", "diag", "action_density"}
    for r in obs_summary:
        if r["observable"] in primary and int(r["sweep"]) in {0, 20, 100, 200}:
            lines.append(
                f"| {r['patch_size']} | {r['n_coarse_passes']} | {r['sweep']} | {r['observable']} | "
                f"{r['mean']:.6g} | {r['SE_over_chains']:.6g} | {r['native_value']:.6g} | {r['z_vs_native']:.6g} |"
            )
    lines += [
        "",
        "## Initial Interpretation",
        "",
        f"- This report is written by the run script after completion. Interpret the {cfg_summary['chains']}x{cfg_summary['sweeps']} results as a modest controlled diagnostic, not production validation.",
        "- The useful aggressive range is judged by patch-chain A/R, observable flow, and absence of obvious drift/overshoot in local/action-sector operators.",
        "- Magnetization-sector quantities are included as diagnostics but should not dominate the local/action-sector decision.",
    ]
    (out / report_name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_settings(items: list[str]) -> list[tuple[int, int]]:
    out = []
    for item in items:
        p_str, passes_str = item.split(":", 1)
        out.append((int(p_str), int(passes_str)))
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    cfg_base = load_config(args.config)
    cfg = override_validation_config(
        cfg_base,
        coarse_L=args.coarse_L,
        fine_L=args.fine_L,
        lambda_c=args.lambda_,
        lambda_f=args.lambda_,
        kappa_c=args.kappa_c,
        kappa_f=args.kappa_f,
        coarse_ensemble=args.coarse_ensemble,
        fine_reference=args.fine_reference,
        run_name=args.run_name,
        output_dir=out,
    )
    ctx = load_ctx(cfg, args)
    paths = resolve_run_paths(cfg)
    coarse = np.load(paths["coarse_ensemble"])["phi"].astype(np.float32)
    if coarse.shape[1:] != (args.coarse_L, args.coarse_L):
        raise RuntimeError(f"expected native L{args.coarse_L} coarse starts, got {coarse.shape}")
    fine_ref_path = paths.get("fine_reference")
    fine_ref = np.load(fine_ref_path)["phi"].astype(np.float32) if fine_ref_path and Path(fine_ref_path).exists() else np.empty((0, args.fine_L, args.fine_L), dtype=np.float32)
    ref = reference_rows(fine_ref, ctx["fine_action"])
    ref_by_key = {r["observable"]: r for r in ref}
    rng_master = np.random.default_rng(args.seed)
    settings = parse_settings(args.settings)
    checkpoint_sweeps = sorted(set(int(x) for x in args.save_state_sweeps))
    patch_rows: list[dict[str, Any]] = []
    accept_history: list[dict[str, Any]] = []
    obs_history: list[dict[str, Any]] = []
    runtime_by_setting: dict[tuple[int, int], float] = {}
    t_all = time.perf_counter()
    resume_history_dir: Path | None = None
    if args.resume_state_file is not None:
        resume_history_dir = args.resume_state_file.resolve().parent.parent
    for patch_size, n_passes in settings:
        t_setting = time.perf_counter()
        pps = patches_per_sweep(args.coarse_L, patch_size)
        chain_states: list[dict[str, Any]] = []
        chain_rngs: list[np.random.Generator] = []
        if args.resume_state_file is not None:
            resume_meta, chain_states, chain_rngs = load_checkpoint(args.resume_state_file)
            if int(resume_meta["n_chains"]) != args.chains:
                raise RuntimeError(f"resume checkpoint has n_chains={resume_meta['n_chains']} but args.chains={args.chains}")
            if int(resume_meta["coarse_L"]) != args.coarse_L or int(resume_meta["fine_L"]) != args.fine_L:
                raise RuntimeError("resume checkpoint lattice size mismatch")
            if float(resume_meta["lambda"]) != float(args.lambda_) or float(resume_meta["kappa_c"]) != float(args.kappa_c) or float(resume_meta["kappa_f"]) != float(args.kappa_f):
                raise RuntimeError("resume checkpoint lambda/kappa mismatch")
            if int(resume_meta["detail_patch_size"]) != int(args.detail_patch_size) or int(resume_meta["latent_updates_per_sweep"]) != int(args.latent_updates_per_sweep):
                raise RuntimeError("resume checkpoint detail settings mismatch")
            if float(resume_meta["site_step_size"]) != float(args.site_step_size):
                raise RuntimeError("resume checkpoint site_step_size mismatch")
            completed_sweeps = int(resume_meta["completed_sweeps"])
            if resume_history_dir is not None:
                patch_rows.extend(read_csv_rows(resume_history_dir / "controlled_patch_chain_patch_move_summary.csv"))
                accept_history.extend(read_csv_rows(resume_history_dir / "controlled_patch_chain_acceptance_history.csv"))
                obs_history.extend(read_csv_rows(resume_history_dir / "controlled_patch_chain_observable_history.csv"))
            start_sweep = completed_sweeps + 1
            end_sweep = completed_sweeps + args.sweeps
        else:
            for chain in range(args.chains):
                rng = np.random.default_rng(int(rng_master.integers(0, 2**63 - 1)))
                chain_rngs.append(rng)
                if args.source_coarse_index is None:
                    source_idx = int(rng.integers(0, coarse.shape[0]))
                else:
                    source_idx = int(args.source_coarse_index)
                z_edge, z_pair, z_corner = sample_chain_z(rng, 1, args.coarse_L, ctx)
                state = compute_chain_state(coarse[source_idx : source_idx + 1].copy(), z_edge, z_pair, z_corner, ctx)
                state["source_coarse_index"] = source_idx
                chain_states.append(state)
                row = {
                    "patch_size": patch_size,
                    "n_coarse_passes": n_passes,
                    "chain": chain,
                    "sweep": 0,
                    "source_coarse_index": source_idx,
                }
                row.update(local_observable_row(state["phi"], ctx["fine_action"]))
                obs_history.append(row)
            start_sweep = 1
            end_sweep = args.sweeps
        save_sweeps = normalize_save_sweeps(
            args.save_sweeps,
            resumed_from_checkpoint=args.resume_state_file is not None,
            added_sweeps=args.sweeps,
            completed_sweeps=completed_sweeps if args.resume_state_file is not None else 0,
        )
        for sweep in range(start_sweep, end_sweep + 1):
            sweep_start = len(patch_rows)
            for chain, (state, rng) in enumerate(zip(chain_states, chain_rngs)):
                for patch_idx in range(pps):
                    x0 = int(rng.integers(0, args.coarse_L))
                    y0 = int(rng.integers(0, args.coarse_L))
                    u_prop_2d, cstats = controlled_patch_metropolis(
                        state["u"][0],
                        patch_sites(args.coarse_L, x0, y0, patch_size),
                        rng,
                        ctx["coarse_action"],
                        args.site_step_size,
                        sweeps=n_passes,
                    )
                    proposal = compute_chain_state(u_prop_2d[None], state["z_edge"], state["z_pair"], state["z_corner"], ctx)
                    dlog = float(proposal["logw"][0] - state["logw"][0])
                    accept = math.log(max(float(rng.random()), 1.0e-300)) < min(0.0, dlog)
                    patch_rows.append({
                        "move_type": "controlled_coarse_patch",
                        "patch_size": patch_size,
                        "detail_patch_size": float("nan"),
                        "n_coarse_passes": n_passes,
                        "chain": chain,
                        "sweep": sweep,
                        "patch_in_sweep": patch_idx,
                        "patch_x": x0,
                        "patch_y": y0,
                        "sites_per_patch": patch_size * patch_size,
                        "attempted_site_updates": cstats["attempted_site_updates"],
                        "accepted_site_updates": cstats["accepted_site_updates"],
                        "coarse_site_acceptance": cstats["coarse_site_acceptance"],
                        "delta_Sc_patch": cstats["delta_Sc_patch"],
                        "patch_l2_change": cstats["patch_l2_change"],
                        "patch_norm2_change": cstats["patch_norm2_change"],
                        "patch_linf_change": cstats["patch_linf_change"],
                        "fine_log_accept": dlog,
                        "fine_AR_accept": int(accept),
                        "latent_beta": float("nan"),
                        "latent_rho": float("nan"),
                        "delta_Sf": float(proposal["sf"][0] - state["sf"][0]),
                        "delta_logdet_refine": float(proposal["logdet"][0] - state["logdet"][0]),
                        "delta_logq_detail": float(proposal["logq"][0] - state["logq"][0]),
                    })
                    if accept:
                        proposal["source_coarse_index"] = state.get("source_coarse_index", -1)
                        state = proposal
                for latent_idx in range(args.latent_updates_per_sweep):
                    x0 = int(rng.integers(0, args.coarse_L))
                    y0 = int(rng.integers(0, args.coarse_L))
                    proposal_l, lstats = propose_latent_pcn_patch(
                        state,
                        x0,
                        y0,
                        args.detail_patch_size,
                        rng,
                        ctx,
                        args.latent_beta,
                    )
                    dlog_l = float(lstats["fine_log_accept"])
                    accept_l = math.log(max(float(rng.random()), 1.0e-300)) < min(0.0, dlog_l)
                    patch_rows.append({
                        "move_type": "latent_detail_pCN",
                        "patch_size": patch_size,
                        "detail_patch_size": args.detail_patch_size,
                        "n_coarse_passes": n_passes,
                        "chain": chain,
                        "sweep": sweep,
                        "patch_in_sweep": pps + latent_idx,
                        "patch_x": lstats["patch_x"],
                        "patch_y": lstats["patch_y"],
                        "sites_per_patch": lstats["sites_per_patch"],
                        "attempted_site_updates": 0,
                        "accepted_site_updates": 0,
                        "coarse_site_acceptance": float("nan"),
                        "delta_Sc_patch": 0.0,
                        "patch_l2_change": lstats["patch_l2_change"],
                        "patch_norm2_change": lstats["patch_norm2_change"],
                        "patch_linf_change": lstats["patch_linf_change"],
                        "fine_log_accept": dlog_l,
                        "fine_AR_accept": int(accept_l),
                        "latent_beta": lstats["latent_beta"],
                        "latent_rho": lstats["latent_rho"],
                        "delta_Sf": lstats["delta_Sf"],
                        "delta_logdet_refine": lstats["delta_logdet_refine"],
                        "delta_logq_detail": lstats["delta_logq_detail"],
                    })
                    if accept_l:
                        proposal_l["source_coarse_index"] = state.get("source_coarse_index", -1)
                        state = proposal_l
                chain_states[chain] = state
            sweep_rows = patch_rows[sweep_start:]
            coarse_rows = [r for r in sweep_rows if r["move_type"] == "controlled_coarse_patch"]
            latent_rows = [r for r in sweep_rows if r["move_type"] == "latent_detail_pCN"]
            accept_history.append({
                "patch_size": patch_size,
                "detail_patch_size": args.detail_patch_size,
                "n_coarse_passes": n_passes,
                "sweep": sweep,
                "coarse_attempts": len(coarse_rows),
                "coarse_fine_accepts": int(sum(int(r["fine_AR_accept"]) for r in coarse_rows)),
                "coarse_fine_AR_acceptance": float(np.mean([int(r["fine_AR_accept"]) for r in coarse_rows])) if coarse_rows else float("nan"),
                "coarse_site_acceptance": float(sum(int(r["accepted_site_updates"]) for r in coarse_rows) / max(sum(int(r["attempted_site_updates"]) for r in coarse_rows), 1)),
                "coarse_log_accept_mean": float(np.mean([float(r["fine_log_accept"]) for r in coarse_rows])) if coarse_rows else float("nan"),
                "coarse_log_accept_std": float(np.std([float(r["fine_log_accept"]) for r in coarse_rows], ddof=1)) if len(coarse_rows) > 1 else float("nan"),
                "latent_attempts": len(latent_rows),
                "latent_accepts": int(sum(int(r["fine_AR_accept"]) for r in latent_rows)),
                "latent_acceptance": float(np.mean([int(r["fine_AR_accept"]) for r in latent_rows])) if latent_rows else float("nan"),
                "latent_log_accept_mean": float(np.mean([float(r["fine_log_accept"]) for r in latent_rows])) if latent_rows else float("nan"),
                "latent_log_accept_std": float(np.std([float(r["fine_log_accept"]) for r in latent_rows], ddof=1)) if len(latent_rows) > 1 else float("nan"),
            })
            if sweep in save_sweeps:
                for chain, state in enumerate(chain_states):
                    row = {
                        "patch_size": patch_size,
                        "n_coarse_passes": n_passes,
                        "chain": chain,
                        "sweep": sweep,
                        "source_coarse_index": int(state.get("source_coarse_index", -1)),
                    }
                    row.update(local_observable_row(state["phi"], ctx["fine_action"]))
                    obs_history.append(row)
                export_raw_data_tables(
                    out,
                    obs_history,
                    coarse_L=args.coarse_L,
                    fine_L=args.fine_L,
                    detail_patch_size=args.detail_patch_size,
                    detail_updates_per_sweep=args.latent_updates_per_sweep,
                )
            if args.checkpoint_dir is not None and ((checkpoint_sweeps and sweep in checkpoint_sweeps) or (not checkpoint_sweeps and sweep == end_sweep)):
                cp_name = f"checkpoint_p{patch_size}_passes{n_passes}_sweep{sweep:06d}.npz"
                save_checkpoint(
                    args.checkpoint_dir / cp_name,
                    completed_sweeps=sweep,
                    chain_states=chain_states,
                    chain_rngs=chain_rngs,
                    args=args,
                    settings=settings,
                    sweep_label=sweep,
                )
                flush_history_tables(
                    out,
                    patch_rows,
                    accept_history,
                    obs_history,
                    coarse_L=args.coarse_L,
                    fine_L=args.fine_L,
                    detail_patch_size=args.detail_patch_size,
                    detail_updates_per_sweep=args.latent_updates_per_sweep,
                )
            if sweep % max(1, args.progress_interval) == 0:
                sweep_rows = patch_rows[sweep_start:]
                coarse_rows_now = [r for r in sweep_rows if r["move_type"] == "controlled_coarse_patch"]
                latent_rows_now = [r for r in sweep_rows if r["move_type"] == "latent_detail_pCN"]
                snap = setting_acceptance_snapshot(sweep_rows, patch_size=patch_size, n_passes=n_passes)
                snap.update({
                    "sweep": sweep,
                    "elapsed_sec": time.perf_counter() - t_setting,
                    "coarse_patch_l2_mean": float(np.mean([float(r["patch_l2_change"]) for r in coarse_rows_now])) if coarse_rows_now else float("nan"),
                    "latent_patch_l2_mean": float(np.mean([float(r["patch_l2_change"]) for r in latent_rows_now])) if latent_rows_now else float("nan"),
                    "local_rms_mean": chain_rms_snapshot(chain_states, ctx["fine_action"], ref_by_key),
                })
                print(json.dumps(snap), flush=True)
        runtime_by_setting[(patch_size, n_passes)] = time.perf_counter() - t_setting
    obs_summary = summarize_observable_history(obs_history, ref_by_key)
    summary_rows = summarize_moves(patch_rows, settings, args.chains, args.sweeps, runtime_by_setting, args.coarse_L)
    flush_history_tables(
        out,
        patch_rows,
        accept_history,
        obs_history,
        ref_by_key=ref_by_key,
        coarse_L=args.coarse_L,
        fine_L=args.fine_L,
        detail_patch_size=args.detail_patch_size,
        detail_updates_per_sweep=args.latent_updates_per_sweep,
    )
    raw_data_manifest = export_raw_data_tables(
        out,
        obs_history,
        coarse_L=args.coarse_L,
        fine_L=args.fine_L,
        detail_patch_size=args.detail_patch_size,
        detail_updates_per_sweep=args.latent_updates_per_sweep,
    )
    write_csv(out / "controlled_patch_chain_summary.csv", summary_rows)
    write_csv(out / "native_L64_reference_local_observables.csv", ref)
    cfg_summary = {
        "status": "completed",
        "config": str(args.config),
        "coarse_ensemble": str(paths["coarse_ensemble"]),
        "fine_reference": str(fine_ref_path) if fine_ref_path else "",
        "chains": args.chains,
        "sweeps": args.sweeps,
        "save_sweeps": save_sweeps,
        "coarse_L": args.coarse_L,
        "fine_L": args.fine_L,
        "lambda": args.lambda_,
        "kappa_c": args.kappa_c,
        "kappa_f": args.kappa_f,
        "site_step_size": args.site_step_size,
        "detail_patch_size": args.detail_patch_size,
        "latent_beta": args.latent_beta,
        "latent_rho": math.sqrt(max(0.0, 1.0 - args.latent_beta * args.latent_beta)),
        "latent_updates_per_sweep": args.latent_updates_per_sweep,
        "resume_state_file": str(args.resume_state_file) if args.resume_state_file else None,
        "resume_history_dir": str(resume_history_dir) if resume_history_dir else None,
        "checkpoint_dir": str(args.checkpoint_dir) if args.checkpoint_dir else None,
        "save_state_sweeps": checkpoint_sweeps,
        "elapsed_sec": time.perf_counter() - t_all,
        "settings": settings,
        "summary_rows": summary_rows,
        "raw_data_manifest": raw_data_manifest,
    }
    write_json(out / "summary.json", cfg_summary)
    report_name = f"CONTROLLED_COARSE_PATCH_CHAIN_{args.coarse_L}TO{args.fine_L}_P12_P16_PASS2_REPORT.md"
    write_report(out, summary_rows, obs_summary, ref, cfg_summary, report_name)
    return cfg_summary


def main() -> int:
    configure_stdio()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--run-name", type=str, default=None)
    ap.add_argument("--coarse-L", type=int, default=32)
    ap.add_argument("--fine-L", type=int, default=64)
    ap.add_argument("--lambda", dest="lambda_", type=float, default=0.022)
    ap.add_argument("--kappa-c", type=float, default=0.2705)
    ap.add_argument("--kappa-f", type=float, default=0.2705)
    ap.add_argument("--coarse-ensemble", type=Path, default=None)
    ap.add_argument("--fine-reference", type=Path, default=None)
    ap.add_argument("--chains", type=int, default=8)
    ap.add_argument("--sweeps", type=int, default=200)
    ap.add_argument("--settings", type=str, nargs="+", default=["12:2", "16:2"], help="Patch/pass settings as P:passes.")
    ap.add_argument("--site-step-size", type=float, default=0.6)
    ap.add_argument("--detail-patch-size", type=int, default=12)
    ap.add_argument("--latent-beta", type=float, default=0.0)
    ap.add_argument("--latent-updates-per-sweep", type=int, default=0)
    ap.add_argument("--save-sweeps", type=int, nargs="+", default=None)
    ap.add_argument("--source-coarse-index", type=int, default=None)
    ap.add_argument("--resume-state-file", type=Path, default=None)
    ap.add_argument("--checkpoint-dir", type=Path, default=None)
    ap.add_argument("--save-state-sweeps", type=int, nargs="*", default=[])
    ap.add_argument("--l16to32-footprint-checkpoint-root", type=Path, default=None)
    ap.add_argument("--l16to32-footprint", type=int, default=11)
    ap.add_argument("--seed", type=int, default=20260703)
    ap.add_argument("--progress-interval", type=int, default=10)
    args = ap.parse_args()
    if args.save_sweeps is None:
        args.save_sweeps = default_save_sweeps(args.sweeps)
    if args.out_dir is None:
        args.out_dir = PKG / "outputs" / "shape_parametric_sampler_validation" / (
            f"controlled_coarse_patch_chain_{args.coarse_L}to{args.fine_L}_"
            f"lam{format_float_tag(args.lambda_)}_kc{format_float_tag(args.kappa_c)}_kf{format_float_tag(args.kappa_f)}"
        )
    summary = run(args)
    print(
        json.dumps(
            {
                "out": str(args.out_dir),
                "status": summary["status"],
                "elapsed_sec": summary["elapsed_sec"],
                "acceptance_summary": [
                    {
                        "patch_size": r["patch_size"],
                        "n_coarse_passes": r["n_coarse_passes"],
                        "coarse_site_acceptance": r["coarse_site_acceptance"],
                        "fine_AR_acceptance": r["fine_AR_acceptance"],
                        "latent_acceptance": r["latent_acceptance"] if "latent_acceptance" in r else None,
                        "fine_log_accept_mean": r["fine_log_accept_mean"],
                        "fine_log_accept_std": r["fine_log_accept_std"],
                    }
                    for r in summary["summary_rows"]
                ],
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
