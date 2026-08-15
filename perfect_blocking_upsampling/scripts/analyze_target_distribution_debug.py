#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PKG = PROJECT_ROOT / "perfect_blocking_upsampling"
FINITE = PROJECT_ROOT / "InverseBlocking_lam0p022_finite_footprint_flow"
FROZEN = PROJECT_ROOT / "InverseBlocking_lam0p022_frozen"
sys.path.insert(0, str(PKG / "src"))
sys.path.insert(0, str(PKG / "scripts"))
sys.path.insert(0, str(FINITE / "scripts"))

from _common import load_config, load_ensembles, load_kernel_spec  # noqa: E402
from perfect_blocking_upsampling.actions import action_total  # noqa: E402
from perfect_blocking_upsampling.kernels import apply_kernel  # noqa: E402
from perfect_blocking_upsampling.observables import observables as ensemble_observables  # noqa: E402
from train_finite_footprint_flow import write_csv, write_json  # noqa: E402
from train_finite_footprint_transported_detail import inner_patch_metropolis, patch_sites, random_origin_patch_schedule  # noqa: E402
from run_shape_parametric_sampler_validation import ValidationConfig, choose_initial_index, sample_z  # noqa: E402


OBS_KEYS = ["action_density", "phi2", "phi4", "NN", "m", "abs_m", "Binder_U4", "susceptibility"]
COARSE_KEYS = ["m", "abs_m", "phi2", "phi4", "NN", "Binder_U4", "susceptibility", "action_density"]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def as_float_array(rows: list[dict[str, str]], key: str) -> np.ndarray:
    return np.asarray([float(r[key]) for r in rows if r.get(key, "") != ""], dtype=np.float64)


def mean_se(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return {"mean": float("nan"), "se": float("nan"), "std": float("nan"), "n": 0}
    return {
        "mean": float(np.mean(x)),
        "se": float(np.std(x, ddof=1) / math.sqrt(x.size)) if x.size > 1 else float("nan"),
        "std": float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
        "n": int(x.size),
    }


def load_old_transport_timeseries(old_run: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for i, path in enumerate(sorted(old_run.glob("chain*/samepatch_latent_observable_timeseries.csv")), start=1):
        for row in read_csv(path):
            row = dict(row)
            row.setdefault("chain_id", str(i - 1))
            rows.append(row)
    return rows


def reference_stats(fine_ref: np.ndarray, fine_action) -> dict[str, dict[str, float]]:
    arr = np.asarray(fine_ref, dtype=np.float64)
    per_config = {
        "m": arr.mean(axis=(1, 2)),
        "abs_m": np.abs(arr.mean(axis=(1, 2))),
        "phi2": np.mean(arr**2, axis=(1, 2)),
        "phi4": np.mean(arr**4, axis=(1, 2)),
        "NN": 0.5
        * (
            np.mean(arr * np.roll(arr, -1, axis=1), axis=(1, 2))
            + np.mean(arr * np.roll(arr, -1, axis=2), axis=(1, 2))
        ),
        "action_density": action_total(arr, fine_action) / (arr.shape[1] * arr.shape[2]),
        "susceptibility": arr.shape[1] * arr.shape[2] * arr.mean(axis=(1, 2)) ** 2,
    }
    stats = {k: mean_se(v) for k, v in per_config.items()}
    ens = ensemble_observables(fine_ref, fine_action)
    stats["Binder_U4"] = {"mean": float(ens["Binder_U4"]), "se": float("nan"), "std": float("nan"), "n": int(fine_ref.shape[0])}
    stats["xi_over_L"] = {"mean": float(ens["xi_over_L"]), "se": float("nan"), "std": float("nan"), "n": int(fine_ref.shape[0])}
    return stats


def window_rows(rows: list[dict[str, str]], start: int, stop: int | None) -> list[dict[str, str]]:
    out = []
    for row in rows:
        sweep = int(row["sweep"])
        if sweep >= start and (stop is None or sweep < stop):
            out.append(row)
    return out


def summarize_windows(rows: list[dict[str, str]], ref: dict[str, dict[str, float]], old_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    max_sweep = max(int(r["sweep"]) for r in rows)
    windows = [
        ("sweep_0", 0, 1),
        ("early_0_50", 0, min(50, max_sweep + 1)),
        ("middle_200_300", min(200, max_sweep + 1), min(300, max_sweep + 1)),
        ("late_400_500", min(400, max_sweep + 1), max_sweep + 1),
        ("all", 0, max_sweep + 1),
    ]
    out = []
    for name, start, stop in windows:
        wr = window_rows(rows, start, stop)
        old_wr = window_rows(old_rows, start, stop) if old_rows else []
        for key in OBS_KEYS:
            if not wr or key not in wr[0]:
                continue
            vals = as_float_array(wr, key)
            old_vals = as_float_array(old_wr, key) if old_wr and key in old_wr[0] else np.asarray([], dtype=np.float64)
            st = mean_se(vals)
            ref_mean = ref.get(key, {}).get("mean", float("nan"))
            ref_se = ref.get(key, {}).get("se", float("nan"))
            out.append(
                {
                    "window": name,
                    "sweep_start": start,
                    "sweep_stop": stop,
                    "observable": key,
                    "assembled_mean": st["mean"],
                    "assembled_std": st["std"],
                    "assembled_n": st["n"],
                    "direct_reference_mean": ref_mean,
                    "direct_reference_se": ref_se,
                    "z_vs_direct": (st["mean"] - ref_mean) / max(ref_se, 1.0e-300) if math.isfinite(ref_se) else float("nan"),
                    "old_transport_mean": float(np.mean(old_vals)) if old_vals.size else float("nan"),
                    "assembled_minus_old_transport": st["mean"] - float(np.mean(old_vals)) if old_vals.size else float("nan"),
                }
            )
    return out


def per_chain_summary(rows: list[dict[str, str]], ref: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    chains = sorted({int(r["chain_id"]) for r in rows})
    out = []
    for chain in chains:
        cr = [r for r in rows if int(r["chain_id"]) == chain]
        first = [r for r in cr if int(r["sweep"]) == 0]
        late = [r for r in cr if int(r["sweep"]) >= max(0, max(int(x["sweep"]) for x in cr) - 99)]
        for key in OBS_KEYS:
            vals = as_float_array(cr, key)
            late_vals = as_float_array(late, key)
            out.append(
                {
                    "chain_id": chain,
                    "observable": key,
                    "sweep0": float(first[0][key]) if first else float("nan"),
                    "mean_all": float(np.mean(vals)),
                    "mean_late100": float(np.mean(late_vals)),
                    "direct_reference_mean": ref.get(key, {}).get("mean", float("nan")),
                }
            )
    return out


def plot_histories(rows: list[dict[str, str]], ref: dict[str, dict[str, float]], old_rows: list[dict[str, str]], out_pdf: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    chains = sorted({int(r["chain_id"]) for r in rows})
    fig, axes = plt.subplots(4, 2, figsize=(11.0, 13.5), sharex=True)
    axes = axes.reshape(-1)
    for ax, key in zip(axes, OBS_KEYS):
        for chain in chains:
            cr = [r for r in rows if int(r["chain_id"]) == chain]
            x = np.asarray([int(r["sweep"]) for r in cr])
            y = as_float_array(cr, key)
            ax.plot(x, y, lw=0.8, alpha=0.65, label=f"c{chain}" if key == OBS_KEYS[0] else None)
        if old_rows and key in old_rows[0]:
            old_by_sweep: dict[int, list[float]] = {}
            for r in old_rows:
                old_by_sweep.setdefault(int(r["sweep"]), []).append(float(r[key]))
            xs = np.asarray(sorted(old_by_sweep))
            ys = np.asarray([np.mean(old_by_sweep[i]) for i in xs])
            ax.plot(xs, ys, color="black", lw=1.2, alpha=0.9, label="old transport mean" if key == OBS_KEYS[0] else None)
        if key in ref:
            ax.axhline(ref[key]["mean"], color="tab:red", ls="--", lw=1.0, label="direct ref" if key == OBS_KEYS[0] else None)
        ax.set_title(key)
        ax.grid(alpha=0.25)
    axes[0].legend(ncol=3, fontsize=7)
    axes[-1].set_xlabel("sweep")
    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)


def replay_coarse_histories(run_dir: Path, coarse: np.ndarray, coarse_action, cfg: ValidationConfig) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    initial_rows = read_csv(run_dir / "initial_chain_states.csv")
    coarse_deltas = read_csv(run_dir / "coarse_deltas.csv")
    latent_deltas = read_csv(run_dir / "latent_deltas.csv")
    by_coarse: dict[tuple[int, int, int], dict[str, str]] = {
        (int(r["chain_id"]), int(r["sweep"]), int(r["attempt_in_sweep"])): r for r in coarse_deltas
    }
    latent_keys = {(int(r["chain_id"]), int(r["sweep"]), int(r["attempt_in_sweep"])) for r in latent_deltas}
    rows = []
    mismatches = []
    for init in initial_rows:
        chain = int(init["chain_id"])
        rng = np.random.default_rng(cfg.seed + 10000 * chain + 777)
        idx, _ = choose_initial_index(coarse, chain, cfg, rng)
        if idx != int(init["coarse_index"]):
            mismatches.append({"chain_id": chain, "expected_index": int(init["coarse_index"]), "replayed_index": idx})
        sample_z(rng, 1, coarse.shape[1])
        u = coarse[idx].copy()
        patch_counter = 0
        for sweep in range(cfg.smoke_sweeps):
            schedule = random_origin_patch_schedule(coarse.shape[1], cfg.patch_size, rng, cfg.origin_mode)
            schedule_len = len(schedule)
            for attempt, (x0, y0, _tile) in enumerate(schedule):
                sites = patch_sites(coarse.shape[1], x0, y0, cfg.patch_size)
                u_prop, _ = inner_patch_metropolis(u.copy(), sites, rng)
                accept = int(by_coarse[(chain, sweep, attempt)]["accepted"])
                if accept:
                    u = u_prop
                rng.random()
                patch_counter += 1
                if patch_counter % schedule_len == 0 and (sweep + 1) % cfg.pcn_interval_sweeps == 0:
                    if (chain, sweep, attempt) not in latent_keys:
                        mismatches.append({"chain_id": chain, "sweep": sweep, "attempt": attempt, "missing_latent": True})
                    for _ in range(3 * cfg.patch_size * cfg.patch_size):
                        rng.standard_normal()
                    rng.random()
            obs = ensemble_observables(u[None], coarse_action)
            rows.append({"chain_id": chain, "sweep": sweep, **{k: obs[k] for k in COARSE_KEYS if k in obs}})
    return rows, mismatches


def coarse_reference(coarse: np.ndarray, fine: np.ndarray, kernel, coarse_action) -> dict[str, Any]:
    psi = apply_kernel(fine, kernel)
    blocked = psi[:, 0::2, 0::2]
    return {
        "initial_coarse_ensemble": {k: ensemble_observables(coarse, coarse_action)[k] for k in COARSE_KEYS},
        "blocked_direct_L16": {k: ensemble_observables(blocked, coarse_action)[k] for k in COARSE_KEYS},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=PKG / "outputs" / "shape_parametric_sampler_validation" / "sector_aware_8x500")
    ap.add_argument("--config", type=Path, default=PKG / "outputs" / "procedural_corner_diagnostics" / "old_pair_corner_procedural_masks.yaml")
    ap.add_argument("--old-run", type=Path, default=FROZEN / "validation" / "rho0p5_every20_validation_8x20k_20260629_083442")
    args = ap.parse_args()

    out = args.run_dir
    cfg_json = json.loads((out / "run_config.json").read_text())
    cfg = ValidationConfig(
        patch_size=int(cfg_json["patch_size"]),
        origin_mode=str(cfg_json["origin_mode"]),
        smoke_sweeps=int(cfg_json["sweeps"]),
        validation_chains=int(cfg_json["validation_chains"]),
        pcn_rho=float(cfg_json["pcn_rho"]),
        pcn_interval_sweeps=int(cfg_json["pcn_interval_sweeps"]),
        seed=int(cfg_json["seed"]),
        sector_balanced_init=bool(cfg_json.get("sector_balanced_init", False)),
    )
    loaded_cfg = load_config(args.config)
    coarse, fine_ref, _, _, _ = load_ensembles(loaded_cfg)
    kernel, _ = load_kernel_spec(loaded_cfg)
    coarse_action = load_config(args.config)["action"]["coarse"]
    fine_action = load_config(args.config)["action"]["fine"]
    from perfect_blocking_upsampling.io import ActionSpec

    coarse_action_spec = ActionSpec(
        type=str(coarse_action["type"]),
        lambda_=float(coarse_action["lambda"]),
        kappa=float(coarse_action["kappa"]),
        kappa_diag=float(coarse_action.get("kappa_diag", 0.0)),
    )
    fine_action_spec = ActionSpec(
        type=str(fine_action["type"]),
        lambda_=float(fine_action["lambda"]),
        kappa=float(fine_action["kappa"]),
        kappa_diag=float(fine_action.get("kappa_diag", 0.0)),
    )

    obs_rows = read_csv(out / "observable_timeseries.csv")
    old_rows = load_old_transport_timeseries(args.old_run)
    ref = reference_stats(fine_ref, fine_action_spec)
    window_summary = summarize_windows(obs_rows, ref, old_rows)
    chain_summary = per_chain_summary(obs_rows, ref)
    plot_histories(obs_rows, ref, old_rows, out / "target_distribution_observable_histories.pdf")
    coarse_rows, replay_mismatches = replay_coarse_histories(out, coarse, coarse_action_spec, cfg)
    write_csv(out / "replayed_coarse_observable_timeseries.csv", coarse_rows)
    coarse_refs = coarse_reference(coarse, fine_ref, kernel, coarse_action_spec)
    coarse_summary = summarize_windows(
        [{k: str(v) for k, v in row.items()} for row in coarse_rows],
        {k: {"mean": float(v), "se": float("nan")} for k, v in coarse_refs["blocked_direct_L16"].items()},
        [],
    )
    payload = {
        "run_dir": str(out),
        "measurement_semantics": "end_of_sweep post-A/R Markov states",
        "observable_rows": len(obs_rows),
        "expected_observable_rows": cfg.validation_chains * cfg.smoke_sweeps,
        "old_transport_rows_loaded": len(old_rows),
        "reference_stats": ref,
        "window_summary": window_summary,
        "per_chain_summary": chain_summary,
        "coarse_reference": coarse_refs,
        "coarse_window_summary_vs_blocked_direct": coarse_summary,
        "coarse_replay_mismatches": replay_mismatches,
    }
    write_json(out / "target_distribution_history_debug.json", payload)
    write_csv(out / "target_distribution_window_summary.csv", window_summary)
    write_csv(out / "target_distribution_per_chain_summary.csv", chain_summary)
    write_csv(out / "target_distribution_coarse_window_summary.csv", coarse_summary)
    print(json.dumps({"status": "completed", "history_json": str(out / "target_distribution_history_debug.json"), "plot": str(out / "target_distribution_observable_histories.pdf")}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
