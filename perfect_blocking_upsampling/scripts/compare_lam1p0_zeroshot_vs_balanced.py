#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PKG = PROJECT_ROOT / "perfect_blocking_upsampling"
SCRIPT_DIR = PKG / "scripts"
sys.path.insert(0, str(PKG / "src"))
sys.path.insert(0, str(SCRIPT_DIR))
os.environ.setdefault("MPLCONFIGDIR", str((PKG / "logs" / "mplconfig").resolve()))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from perfect_blocking_upsampling.actions import ActionSpec, action_total  # noqa: E402
from run_lam1p0_l16to32_rqspline_zeroshot import (  # noqa: E402
    build_model_from_checkpoint,
    log_prob_model_lattice,
    stationary_stats,
)
from train_lam1p0_flow_detail_pilot import (  # noqa: E402
    apply_kernel,
    assemble_psi,
    inverse_kernel,
    load_kernel_matrix,
    load_phi,
    per_config_rows,
    split_pairs,
)


OBS_KEYS = [
    "action_density",
    "total_action",
    "phi2",
    "phi4",
    "local_kurtosis_ratio",
    "NN",
    "2nn",
    "diag",
    "m",
    "m2",
    "m4",
    "G_pmin_x",
    "G_pmin_y",
    "G_pmin_avg",
]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols: list[str] = []
    for row in rows:
        for key in row:
            if key not in cols:
                cols.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def model_hash(state: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for key in sorted(state):
        h.update(key.encode())
        h.update(np.ascontiguousarray(state[key].detach().cpu().numpy()).view(np.uint8))
    return h.hexdigest()


def finite(x: np.ndarray) -> np.ndarray:
    y = np.asarray(x, dtype=np.float64)
    return y[np.isfinite(y)]


def ks_stat(a: np.ndarray, b: np.ndarray) -> float:
    x = np.sort(finite(a))
    y = np.sort(finite(b))
    grid = np.sort(np.concatenate([x, y]))
    return float(np.max(np.abs(np.searchsorted(x, grid, side="right") / len(x) - np.searchsorted(y, grid, side="right") / len(y))))


def wasserstein_1(a: np.ndarray, b: np.ndarray) -> float:
    x = np.sort(finite(a))
    y = np.sort(finite(b))
    n = min(len(x), len(y))
    q = (np.arange(n) + 0.5) / n
    return float(np.mean(np.abs(np.quantile(x, q) - np.quantile(y, q))))


def union_edges(samples: list[np.ndarray], bins_min: int = 80, bins_max: int = 120) -> np.ndarray:
    x = np.concatenate([finite(s) for s in samples])
    q25, q75 = np.quantile(x, [0.25, 0.75])
    iqr = q75 - q25
    bw = 2.0 * iqr / (len(x) ** (1.0 / 3.0)) if iqr > 0 else 0.0
    nb = 100 if bw <= 0 else int(np.ceil((x.max() - x.min()) / bw))
    nb = max(bins_min, min(bins_max, nb))
    return np.linspace(float(x.min()), float(x.max()), nb + 1)


def hist_overlap(a: np.ndarray, b: np.ndarray, edges: np.ndarray) -> float:
    ha, _ = np.histogram(finite(a), bins=edges)
    hb, _ = np.histogram(finite(b), bins=edges)
    pa = ha / max(float(ha.sum()), 1.0)
    pb = hb / max(float(hb.sum()), 1.0)
    return float(np.minimum(pa, pb).sum())


def obs_arrays(phi: np.ndarray, label: str, action: ActionSpec) -> dict[str, np.ndarray]:
    rows, grows = per_config_rows(phi.astype(np.float32), action, label)
    out: dict[str, list[float]] = {key: [] for key in OBS_KEYS}
    for r, g in zip(rows, grows):
        out["action_density"].append(float(r["action_density"]))
        out["total_action"].append(float(r["action_density"]) * float(phi.shape[1] * phi.shape[2]))
        out["phi2"].append(float(r["phi2"]))
        out["phi4"].append(float(r["phi4"]))
        out["local_kurtosis_ratio"].append(float(r["local_kurtosis_ratio"]))
        out["NN"].append(float(r["NN"]))
        out["2nn"].append(float(r["2nn"]))
        out["diag"].append(float(r["diag"]))
        out["m"].append(float(r["m"]))
        out["m2"].append(float(r["m2"]))
        out["m4"].append(float(r["m4"]))
        out["G_pmin_x"].append(float(g["G_10"]))
        out["G_pmin_y"].append(float(g["G_01"]))
        out["G_pmin_avg"].append(float(g["G_pmin_avg"]))
    return {key: np.asarray(vals, dtype=np.float64) for key, vals in out.items()}


def sample_from_latents(
    model: Any,
    coarse_phys: np.ndarray,
    stats: dict[str, Any],
    z: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    coarse_std = ((coarse_phys - float(stats["coarse_mean"])) / float(stats["coarse_std"])).astype(np.float32)
    mean = np.asarray(stats["detail_mean"], dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.asarray(stats["detail_std"], dtype=np.float32).reshape(1, 3, 1, 1)
    log_jac_const = -float(coarse_phys.shape[1] * coarse_phys.shape[2] * np.sum(np.log(std.reshape(3))))
    details: list[np.ndarray] = []
    logqs: list[np.ndarray] = []
    zmaxs: list[np.ndarray] = []
    logdets: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(coarse_std), batch_size):
            stop = min(start + batch_size, len(coarse_std))
            cb = torch.from_numpy(coarse_std[start:stop]).to(device)
            zb = torch.from_numpy(z[start:stop]).to(device)
            n = stop - start
            db = torch.zeros((n, 3, cb.shape[1], cb.shape[2]), dtype=cb.dtype, device=device)
            logq = torch.zeros(n, dtype=cb.dtype, device=device)
            logdet_total = torch.zeros(n, dtype=cb.dtype, device=device)
            zmax = torch.zeros(n, dtype=cb.dtype, device=device)
            dim = cb.shape[1] * cb.shape[2]
            for stage in range(3):
                z_stage = zb[:, stage].reshape(n, dim)
                log_base = -0.5 * (z_stage * z_stage + math.log(2.0 * math.pi)).sum(dim=1)
                cond_affine = model.affine_base.cond(cb, db, stage)
                x_affine, affine_logdet = model.affine_base.flows[stage].forward(z_stage, cond_affine)
                cond_spline = model.cond(cb, db, stage)
                x, spline_logdet = model.spline.flows[stage].forward(x_affine, cond_spline)
                db[:, stage] = x.reshape(n, cb.shape[1], cb.shape[2])
                total_logdet = affine_logdet + spline_logdet
                logq = logq + log_base - total_logdet
                logdet_total = logdet_total + total_logdet
                zmax = torch.maximum(zmax, torch.amax(torch.abs(z_stage), dim=1))
            d_phys = db.detach().cpu().numpy().astype(np.float32) * std + mean
            details.append(d_phys.astype(np.float32))
            logqs.append((logq.detach().cpu().numpy() + log_jac_const).astype(np.float64))
            zmaxs.append(zmax.detach().cpu().numpy().astype(np.float32))
            logdets.append((logdet_total.detach().cpu().numpy() - log_jac_const).astype(np.float64))
    return np.concatenate(details), np.concatenate(logqs), np.concatenate(zmaxs), np.concatenate(logdets)


def metric_rows(native_obs: dict[str, np.ndarray], sample_obs: dict[str, np.ndarray], label: str, edges: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    qlist = [0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99]
    for obs in OBS_KEYS:
        a = native_obs[obs]
        b = sample_obs[obs]
        ns = float(np.std(a, ddof=1))
        row = {
            "checkpoint": label,
            "observable": obs,
            "n": len(b),
            "native_mean": float(np.mean(a)),
            "native_std": ns,
            "sample_mean": float(np.mean(b)),
            "sample_std": float(np.std(b, ddof=1)),
            "mean_shift_native_sigma": float((np.mean(b) - np.mean(a)) / max(ns, 1.0e-300)),
            "std_ratio": float(np.std(b, ddof=1) / max(ns, 1.0e-300)),
            "ks_statistic": ks_stat(a, b),
            "wasserstein_1": wasserstein_1(a, b),
            "histogram_overlap_coefficient": hist_overlap(a, b, edges[obs]),
            "sample_min": float(np.min(b)),
            "sample_max": float(np.max(b)),
        }
        for q in qlist:
            tag = f"q{int(round(100 * q)):02d}"
            nq = float(np.quantile(a, q))
            sq = float(np.quantile(b, q))
            row[f"native_{tag}"] = nq
            row[f"sample_{tag}"] = sq
            row[f"{tag}_difference"] = sq - nq
            row[f"{tag}_ratio"] = sq / nq if nq != 0 else float("nan")
        for q in [0.01, 0.05, 0.10]:
            row[f"frac_below_native_q{int(round(100*q)):02d}"] = float(np.mean(b < np.quantile(a, q)))
        for q in [0.90, 0.95, 0.99]:
            row[f"frac_above_native_q{int(round(100*q)):02d}"] = float(np.mean(b > np.quantile(a, q)))
        rows.append(row)
    return rows


def tail_summary(delta_s: np.ndarray, loga: np.ndarray, accepted_by_sweep: list[np.ndarray]) -> dict[str, Any]:
    acc = np.stack(accepted_by_sweep)
    longest = 0
    for c in range(acc.shape[1]):
        cur = 0
        for ok in acc[:, c]:
            cur = 0 if ok else cur + 1
            longest = max(longest, cur)
    return {
        "DeltaS_mean": float(np.mean(delta_s)),
        "DeltaS_std": float(np.std(delta_s, ddof=1)),
        "DeltaS_median": float(np.quantile(delta_s, 0.50)),
        "DeltaS_p90": float(np.quantile(delta_s, 0.90)),
        "DeltaS_p95": float(np.quantile(delta_s, 0.95)),
        "DeltaS_p99": float(np.quantile(delta_s, 0.99)),
        "logA_mean": float(np.mean(loga)),
        "logA_std": float(np.std(loga, ddof=1)),
        "logA_p01": float(np.quantile(loga, 0.01)),
        "logA_p05": float(np.quantile(loga, 0.05)),
        "logA_p10": float(np.quantile(loga, 0.10)),
        "frac_logA_lt_minus5": float(np.mean(loga < -5.0)),
        "frac_logA_lt_minus10": float(np.mean(loga < -10.0)),
        "frac_logA_lt_minus20": float(np.mean(loga < -20.0)),
        "longest_rejection_streak": int(longest),
    }


def global_diag(
    label: str,
    model: Any,
    stats: dict[str, Any],
    coarse: np.ndarray,
    native_phi: np.ndarray,
    native_detail: np.ndarray,
    kernel: np.ndarray,
    z_stream: np.ndarray,
    accept_uniforms: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    n, sweeps = z_stream.shape[:2]
    current_phi = native_phi[:n].copy()
    current_s = action_total(current_phi, ActionSpec("phi4_nn", 1.0, 0.340301)).astype(np.float64)
    current_logq = log_prob_model_lattice(model, coarse[:n], native_detail[:n], stats, batch_size=batch_size, device=device)
    accepted_total = 0
    ds_vals: list[np.ndarray] = []
    la_vals: list[np.ndarray] = []
    accepted_by_sweep: list[np.ndarray] = []
    acceptance_by_sweep = []
    for sweep in range(sweeps):
        prop_detail, prop_logq, _zmax, _ld = sample_from_latents(model, coarse[:n], stats, z_stream[:, sweep], batch_size=batch_size, device=device)
        prop_phi, _ = inverse_kernel(assemble_psi(coarse[:n], prop_detail), kernel)
        prop_s = action_total(prop_phi, ActionSpec("phi4_nn", 1.0, 0.340301)).astype(np.float64)
        delta_s = prop_s - current_s
        loga = -delta_s + current_logq - prop_logq
        accept = np.log(accept_uniforms[:, sweep]) < np.minimum(loga, 0.0)
        if np.any(accept):
            current_phi[accept] = prop_phi[accept]
            current_s[accept] = prop_s[accept]
            current_logq[accept] = prop_logq[accept]
        accepted_total += int(np.sum(accept))
        ds_vals.append(delta_s)
        la_vals.append(loga)
        accepted_by_sweep.append(accept)
        acceptance_by_sweep.append({"checkpoint": label, "sweep": sweep + 1, "accepted": int(np.sum(accept)), "attempts": n, "acceptance": float(np.mean(accept))})
    row = {
        "checkpoint": label,
        "diagnostic_type": "global_fixed_coarse_independence",
        "number_of_chains": n,
        "number_of_sweeps": sweeps,
        "attempts": int(n * sweeps),
        "accepted": int(accepted_total),
        "acceptance": float(accepted_total / max(n * sweeps, 1)),
        "nonfinite_count": int(np.sum(~np.isfinite(current_phi))),
        "reblocking_max_error": float(np.max(np.abs(apply_kernel(current_phi, kernel)[:, 0::2, 0::2] - coarse[:n]))),
    }
    row.update(tail_summary(np.concatenate(ds_vals), np.concatenate(la_vals), accepted_by_sweep))
    return row, np.asarray(ds_vals), np.asarray(la_vals)


def main() -> int:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = PROJECT_ROOT / "perfect_blocking_upsampling/runs/lam1p0/diagnostics" / f"L8to16_zeroshot_vs_balanced_L16to32_same_dataset_{stamp}"
    (out / "plots").mkdir(parents=True, exist_ok=True)
    (out / "debug").mkdir(parents=True, exist_ok=True)
    batch_size = 128
    seed = 202607181730
    rng = np.random.default_rng(seed)
    device = torch.device("cpu")

    ckpt_paths = {
        "L8to16_zeroshot": PROJECT_ROOT / "perfect_blocking_upsampling/runs/lam1p0/lam1p0_L8to16_kf0p340301_kc0p340301_7x7_phi2_nn_guarded_autoregressive_detail_8layer48_rqspline_localreg_from_affine_ep137_20260717T125835Z/checkpoints/checkpoint_best.pt",
        "balanced_L16to32": PROJECT_ROOT / "perfect_blocking_upsampling/runs/lam1p0/training/lam1p0_L16to32_kf0p340301_kc0p340301_7x7_rqspline_balanced_phi2_phi4_action_support_from_recovered_ep5_20260718T135039Z/checkpoints/checkpoint_best_nll.pt",
    }
    ckpts = {label: torch.load(path, map_location=device, weights_only=False) for label, path in ckpt_paths.items()}
    models = {}
    stats = {}
    inventory = []
    for label, ckpt in ckpts.items():
        model, report = build_model_from_checkpoint(ckpt, lattice_size=16, device=device)
        models[label] = model
        stats[label] = stationary_stats(ckpt["state"]["stats"], lc=16)
        inventory.append(
            {
                "checkpoint": label,
                "path": str(ckpt_paths[label].relative_to(PROJECT_ROOT)),
                "epoch": ckpt.get("epoch"),
                "absolute_epoch": ckpt.get("absolute_epoch"),
                "model_state_hash": model_hash(ckpt["model_state"]),
                "architecture": json.dumps(ckpt.get("architecture", {}), default=str),
                "spline_settings": json.dumps(ckpt.get("spline_settings", {}), default=str),
                "stats": json.dumps({k: (np.asarray(v).tolist() if k in {"detail_mean", "detail_std"} else v) for k, v in stats[label].items()}, default=str),
                "load_report": json.dumps(report, default=str),
            }
        )
    write_csv(out / "checkpoint_inventory.csv", inventory)

    kernel, kernel_json = load_kernel_matrix(PROJECT_ROOT / "perfect_blocking/perfect_blocking_lam1p0/kernels/selected_for_upscaling/best_7x7_phi2_nn_guarded_eta_included.json")
    phi32 = load_phi(PROJECT_ROOT / "data/configs_phi4_2d/lam1p0_kappac0p340301_L32/configs.npz")
    pairs = split_pairs(phi32, kernel)
    n = len(phi32)
    source_idx = np.arange(n, dtype=np.int64)
    action = ActionSpec("phi4_nn", 1.0, 0.340301)
    native_obs = obs_arrays(phi32, "native_L32", action)
    latents = rng.standard_normal((n, 3, 16, 16), dtype=np.float32)
    seed_rows = [{"source_config_index": int(i), "latent_seed": seed, "latent_row": int(i)} for i in source_idx]
    write_csv(out / "paired_seed_map.csv", seed_rows)
    sample_obs = {}
    diag_rows = []
    raw_per_cfg_rows = []
    for label in ["L8to16_zeroshot", "balanced_L16to32"]:
        detail, logq, zmax, logdet = sample_from_latents(models[label], pairs["coarse"], stats[label], latents, batch_size=batch_size, device=device)
        phi, _ = inverse_kernel(assemble_psi(pairs["coarse"], detail), kernel)
        sample_obs[label] = obs_arrays(phi, label, action)
        diag_rows.append(
            {
                "checkpoint": label,
                "nonfinite_count": int(np.sum(~np.isfinite(phi)) + np.sum(~np.isfinite(detail))),
                "max_abs_z": float(np.max(np.abs(latents))),
                "logq_mean": float(np.mean(logq)),
                "logq_std": float(np.std(logq, ddof=1)),
                "logdet_mean": float(np.mean(logdet)),
                "logdet_std": float(np.std(logdet, ddof=1)),
                "reblocking_max_error": float(np.max(np.abs(apply_kernel(phi, kernel)[:, 0::2, 0::2] - pairs["coarse"]))),
            }
        )
        for i in range(n):
            row = {"checkpoint": label, "source_config_index": int(i)}
            for obs in OBS_KEYS:
                row[obs] = float(sample_obs[label][obs][i])
            raw_per_cfg_rows.append(row)
    write_csv(out / "debug/raw_generation_diagnostics.csv", diag_rows)
    write_csv(out / "per_configuration_raw_observables.csv", raw_per_cfg_rows)

    edges = {obs: union_edges([native_obs[obs], sample_obs["L8to16_zeroshot"][obs], sample_obs["balanced_L16to32"][obs]]) for obs in OBS_KEYS}
    metric_rows_all = []
    quantile_rows = []
    for label in ["L8to16_zeroshot", "balanced_L16to32"]:
        metric_rows_all.extend(metric_rows(native_obs, sample_obs[label], label, edges))
        for obs in OBS_KEYS:
            qrow = {"checkpoint": label, "observable": obs}
            for q in [0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99]:
                tag = f"q{int(round(100*q)):02d}"
                nq = float(np.quantile(native_obs[obs], q))
                sq = float(np.quantile(sample_obs[label][obs], q))
                qrow[f"native_{tag}"] = nq
                qrow[f"sample_{tag}"] = sq
                qrow[f"{tag}_ratio"] = sq / nq if nq != 0 else float("nan")
            qrow["frac_below_native_q05"] = float(np.mean(sample_obs[label][obs] < np.quantile(native_obs[obs], 0.05)))
            qrow["frac_above_native_q95"] = float(np.mean(sample_obs[label][obs] > np.quantile(native_obs[obs], 0.95)))
            qrow["frac_above_native_q99"] = float(np.mean(sample_obs[label][obs] > np.quantile(native_obs[obs], 0.99)))
            quantile_rows.append(qrow)
    write_csv(out / "raw_observable_metrics.csv", metric_rows_all)
    write_csv(out / "raw_quantile_metrics.csv", quantile_rows)

    # Paired global independence diagnostic.
    g_n, g_sweeps = 64, 100
    global_latents = rng.standard_normal((g_n, g_sweeps, 3, 16, 16), dtype=np.float32)
    global_uniforms = rng.random((g_n, g_sweeps))
    global_rows = []
    for label in ["L8to16_zeroshot", "balanced_L16to32"]:
        row, ds, la = global_diag(
            label,
            models[label],
            stats[label],
            pairs["coarse"],
            phi32,
            pairs["detail"],
            kernel,
            global_latents,
            global_uniforms,
            batch_size=batch_size,
            device=device,
        )
        global_rows.append(row)
        np.savez_compressed(out / "debug" / f"{label}_global_diagnostics.npz", DeltaS=ds, logA=la)
    write_csv(out / "global_independence_comparison.csv", global_rows)

    # Bootstrap paired raw differences.
    boot_rows = []
    boot_rng = np.random.default_rng(seed + 1)
    for obs in ["action_density", "phi2", "phi4", "local_kurtosis_ratio", "NN", "G_pmin_avg"]:
        native = native_obs[obs]
        zero = sample_obs["L8to16_zeroshot"][obs]
        bal = sample_obs["balanced_L16to32"][obs]
        for metric_name, fn in {
            "mean_shift_native_sigma": lambda a, b: (np.mean(b) - np.mean(a)) / max(np.std(a, ddof=1), 1.0e-300),
            "ks_statistic": ks_stat,
            "wasserstein_1": wasserstein_1,
            "histogram_overlap_coefficient": lambda a, b: hist_overlap(a, b, edges[obs]),
        }.items():
            vals = []
            for _ in range(500):
                ii = boot_rng.integers(0, n, n)
                vals.append(float(fn(native[ii], bal[ii]) - fn(native[ii], zero[ii])))
            arr = np.asarray(vals)
            boot_rows.append(
                {
                    "observable": obs,
                    "metric": metric_name,
                    "balanced_minus_zeroshot": float(fn(native, bal) - fn(native, zero)),
                    "bootstrap_mean": float(np.mean(arr)),
                    "ci_low": float(np.quantile(arr, 0.025)),
                    "ci_high": float(np.quantile(arr, 0.975)),
                    "bootstrap_resamples": 500,
                }
            )
        vals = bal - zero
        boot_rows.append(
            {
                "observable": obs,
                "metric": "paired_value_difference",
                "balanced_minus_zeroshot": float(np.mean(vals)),
                "bootstrap_mean": float(np.mean(vals)),
                "ci_low": float(np.quantile([np.mean(vals[boot_rng.integers(0, n, n)]) for _ in range(500)], 0.025)),
                "ci_high": float(np.quantile([np.mean(vals[boot_rng.integers(0, n, n)]) for _ in range(500)], 0.975)),
                "bootstrap_resamples": 500,
            }
        )
    write_csv(out / "paired_bootstrap_metrics.csv", boot_rows)

    # Plots.
    colors = {"native_L32": "black", "L8to16_zeroshot": "#666666", "balanced_L16to32": "#d62728"}
    for obs in ["action_density", "phi2", "phi4", "local_kurtosis_ratio", "NN", "G_pmin_avg"]:
        samples = {"native_L32": native_obs[obs], "L8to16_zeroshot": sample_obs["L8to16_zeroshot"][obs], "balanced_L16to32": sample_obs["balanced_L16to32"][obs]}
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        for label, vals in samples.items():
            ax.hist(vals, bins=edges[obs], density=True, histtype="step", lw=2.2 if label == "native_L32" else 1.6, color=colors[label], label=label)
        ax.set_xlabel(obs)
        ax.set_ylabel("density")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8, frameon=False)
        fig.tight_layout()
        safe = "Gpmin" if obs == "G_pmin_avg" else ("local_kurtosis" if obs == "local_kurtosis_ratio" else obs)
        for ext in ["pdf", "png"]:
            fig.savefig(out / "plots" / f"{safe}_raw_overlay.{ext}", dpi=180)
        plt.close(fig)
    for obs in ["action_density", "phi2", "phi4"]:
        samples = {"native_L32": native_obs[obs], "L8to16_zeroshot": sample_obs["L8to16_zeroshot"][obs], "balanced_L16to32": sample_obs["balanced_L16to32"][obs]}
        for kind in ["linear_density", "semilog_density", "ecdf", "lower_tail_cdf_zoom", "upper_tail_survival", "quantile_ratio"]:
            fig, ax = plt.subplots(figsize=(6.4, 4.0))
            if kind == "quantile_ratio":
                qgrid = np.linspace(0.01, 0.99, 99)
                nq = np.quantile(native_obs[obs], qgrid)
                ax.axhline(1.0, color="black", lw=1)
                for label in ["L8to16_zeroshot", "balanced_L16to32"]:
                    ax.plot(qgrid, np.quantile(sample_obs[label][obs], qgrid) / nq, label=label, color=colors[label])
                ax.set_xlabel("native quantile probability")
                ax.set_ylabel("generated/native quantile ratio")
            else:
                for label, vals in samples.items():
                    x = np.sort(vals)
                    if kind in {"linear_density", "semilog_density"}:
                        h, _ = np.histogram(x, bins=edges[obs], density=True)
                        mid = 0.5 * (edges[obs][:-1] + edges[obs][1:])
                        ax.step(mid, h, where="mid", lw=2.2 if label == "native_L32" else 1.6, color=colors[label], label=label)
                        if kind == "semilog_density":
                            ax.set_yscale("log")
                        ax.set_ylabel("density")
                    elif kind in {"ecdf", "lower_tail_cdf_zoom"}:
                        y = np.arange(1, len(x) + 1) / len(x)
                        ax.plot(x, y, lw=2.2 if label == "native_L32" else 1.6, color=colors[label], label=label)
                        ax.set_ylabel("CDF")
                        if kind == "lower_tail_cdf_zoom":
                            ax.set_xlim(np.quantile(native_obs[obs], 0.0), np.quantile(native_obs[obs], 0.20))
                            ax.set_ylim(0, 0.22)
                    elif kind == "upper_tail_survival":
                        y = 1.0 - np.arange(1, len(x) + 1) / len(x)
                        ax.plot(x, y, lw=2.2 if label == "native_L32" else 1.6, color=colors[label], label=label)
                        ax.set_yscale("log")
                        ax.set_ylabel("survival")
                ax.set_xlabel(obs)
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8, frameon=False)
            fig.tight_layout()
            for ext in ["pdf", "png"]:
                fig.savefig(out / "plots" / f"{obs}_{kind}.{ext}", dpi=180)
            plt.close(fig)
    for arrname, filename, xlabel in [("DeltaS", "DeltaS_global_overlay", "DeltaS"), ("logA", "logA_global_overlay", "logA")]:
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        vals = []
        data = {}
        for label in ["L8to16_zeroshot", "balanced_L16to32"]:
            z = np.load(out / "debug" / f"{label}_global_diagnostics.npz")
            data[label] = z[arrname].reshape(-1)
            vals.append(data[label])
        ed = union_edges(vals)
        for label, x in data.items():
            ax.hist(x, bins=ed, density=True, histtype="step", lw=1.8, label=label)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("density")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
        fig.tight_layout()
        for ext in ["pdf", "png"]:
            fig.savefig(out / "plots" / f"{filename}.{ext}", dpi=180)
        plt.close(fig)

    raw = {r["checkpoint"] + ":" + r["observable"]: r for r in metric_rows_all}
    global_by = {r["checkpoint"]: r for r in global_rows}
    lines = [
        "# L8->L16 Zero-Shot vs Balanced L16->L32 Same-Dataset Diagnostic",
        "",
        f"- output: `{out.relative_to(PROJECT_ROOT)}`",
        f"- native L32 count: `{n}`",
        f"- paired raw latent seed: `{seed}`",
        f"- kernel sum: `{float(kernel.sum())}`",
        f"- kernel coefficients include eta: `{bool(kernel_json.get('kernel_coefficients_include_eta_scale'))}`",
        "- no training was run.",
        "",
        "## Raw Same-Volume Metrics",
        "",
        "| observable | zero-shot KS | balanced KS | zero-shot shift | balanced shift | zero-shot OVL | balanced OVL |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for obs in ["action_density", "phi2", "phi4", "local_kurtosis_ratio", "NN", "G_pmin_avg"]:
        z = raw[f"L8to16_zeroshot:{obs}"]
        b = raw[f"balanced_L16to32:{obs}"]
        lines.append(
            f"| {obs} | {z['ks_statistic']:.5f} | {b['ks_statistic']:.5f} | {z['mean_shift_native_sigma']:.5f} | {b['mean_shift_native_sigma']:.5f} | {z['histogram_overlap_coefficient']:.5f} | {b['histogram_overlap_coefficient']:.5f} |"
        )
    lines += [
        "",
        "## Global Full-Detail Independence",
        "",
        "| checkpoint | acceptance | DeltaS std | DeltaS p95 | DeltaS p99 | logA mean | frac logA<-10 | frac logA<-20 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label in ["L8to16_zeroshot", "balanced_L16to32"]:
        g = global_by[label]
        lines.append(
            f"| {label} | {g['acceptance']:.5f} | {g['DeltaS_std']:.5f} | {g['DeltaS_p95']:.5f} | {g['DeltaS_p99']:.5f} | {g['logA_mean']:.5f} | {g['frac_logA_lt_minus10']:.5f} | {g['frac_logA_lt_minus20']:.5f} |"
        )
    lines += [
        "",
        "Patchwise detail-only and coarse+detail comparisons are generated separately by the production runner and should be appended after those jobs complete.",
    ]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {
        "created_utc": stamp,
        "output_dir": str(out.relative_to(PROJECT_ROOT)),
        "raw_count": n,
        "global_chains": g_n,
        "global_sweeps": g_sweeps,
        "paired_raw_latent_seed": seed,
        "metric_code": "compare_lam1p0_zeroshot_vs_balanced.py",
        "checkpoints": {k: str(v.relative_to(PROJECT_ROOT)) for k, v in ckpt_paths.items()},
    }
    (out / "run_manifest.txt").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
