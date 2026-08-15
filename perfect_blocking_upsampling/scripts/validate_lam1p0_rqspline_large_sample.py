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
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PKG = PROJECT_ROOT / "perfect_blocking_upsampling"
SCRIPT_DIR = PKG / "scripts"
sys.path.insert(0, str(PKG / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

import os

os.environ.setdefault("MPLCONFIGDIR", str((PKG / "logs" / "mplconfig").resolve()))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from perfect_blocking_upsampling.io import ActionSpec
from perfect_blocking_upsampling.observables import observables as ensemble_observables
from perfect_blocking_upsampling.observables import second_moment_components
from train_lam1p0_autoregressive_detail_flow import ARDetailFlow, action_total, log_prob_model, sample_model
from train_lam1p0_flow_detail_localreg import ETA_SCALE
from train_lam1p0_flow_detail_pilot import assemble_psi, inverse_kernel, load_kernel_matrix, load_phi
from train_lam1p0_rqspline_detail_flow import RQSplineARDetailFlow, ResidualSplineARDetailFlow


OBS_KEYS = [
    "action_density",
    "phi2",
    "phi4",
    "local_kurtosis_ratio",
    "NN",
    "diag",
    "2nn",
    "m",
    "m2",
    "m4",
    "Binder_U4",
    "susceptibility",
    "xi_over_L",
    "G_pmin_avg",
]

HIST_KEYS = ["action_density", "phi2", "phi4", "local_kurtosis_ratio", "NN", "diag", "2nn", "m2", "m4"]
DIAG_KEYS = ["DeltaS", "log_q_forward", "log_q_reverse", "log_q_ratio", "log_acceptance"]


class ArgsShim:
    def __init__(self, batch_size: int, device: str):
        self.batch_size = batch_size
        self.device = device


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))


def load_affine_model(checkpoint_path: Path, device: torch.device) -> tuple[ARDetailFlow, dict[str, Any]]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = ARDetailFlow(
        layers=int(cfg["layers"]),
        hidden=int(cfg["hidden_channels"]),
        kernel_size=int(cfg["conv_kernel_size"]),
        log_scale_bound=float(cfg["log_scale_bound"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def load_residual_spline_model(checkpoint_path: Path, device: torch.device) -> tuple[ResidualSplineARDetailFlow, dict[str, Any]]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    source_path = PROJECT_ROOT / cfg["resume_checkpoint"]
    source_ckpt = torch.load(source_path, map_location=device, weights_only=False)
    source_cfg = source_ckpt["config"]
    affine = ARDetailFlow(
        layers=int(source_cfg["layers"]),
        hidden=int(source_cfg["hidden_channels"]),
        kernel_size=int(source_cfg["conv_kernel_size"]),
        log_scale_bound=float(source_cfg["log_scale_bound"]),
    ).to(device)
    affine.load_state_dict(source_ckpt["model_state"])
    spline = RQSplineARDetailFlow(
        layers=int(cfg["layers"]),
        hidden=int(cfg["hidden_channels"]),
        kernel_size=int(cfg["conv_kernel_size"]),
        num_bins=int(cfg["num_bins"]),
        tail_bound=float(cfg["tail_bound"]),
        min_bin_width=float(cfg["min_bin_width"]),
        min_bin_height=float(cfg["min_bin_height"]),
        min_derivative=float(cfg["min_derivative"]),
    ).to(device)
    model = ResidualSplineARDetailFlow(affine, spline).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def per_config_observables(phi: np.ndarray, action: ActionSpec) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    arr = np.asarray(phi, dtype=np.float64)
    L = arr.shape[1]
    V = L * L
    m = arr.mean(axis=(1, 2))
    phi2 = np.mean(arr * arr, axis=(1, 2))
    phi4 = np.mean(arr**4, axis=(1, 2))
    nn = 0.5 * (
        np.mean(arr * np.roll(arr, -1, axis=1), axis=(1, 2))
        + np.mean(arr * np.roll(arr, -1, axis=2), axis=(1, 2))
    )
    two = 0.5 * (
        np.mean(arr * np.roll(arr, -2, axis=1), axis=(1, 2))
        + np.mean(arr * np.roll(arr, -2, axis=2), axis=(1, 2))
    )
    diag = np.mean(arr * np.roll(np.roll(arr, -1, axis=1), -1, axis=2), axis=(1, 2))
    sc = second_moment_components(arr)
    gavg = 0.5 * (np.asarray(sc["G_pmin_x_cfg"]) + np.asarray(sc["G_pmin_y_cfg"]))
    out = {
        "action_density": action_total(arr, action) / V,
        "phi2": phi2,
        "phi4": phi4,
        "local_kurtosis_ratio": phi4 / np.maximum(phi2 * phi2, 1.0e-300),
        "NN": nn,
        "diag": diag,
        "2nn": two,
        "m": m,
        "m2": m * m,
        "m4": m**4,
        "G_pmin_avg": gavg,
    }
    ens = ensemble_observables(arr, action)
    ens["G_pmin_avg"] = float(np.mean(gavg))
    return out, {k: float(v) for k, v in ens.items() if isinstance(v, (int, float, np.floating))}


def histogram_metrics(native: np.ndarray, sample: np.ndarray, bins: int, range_: tuple[float, float]) -> dict[str, float | int]:
    a = np.asarray(native, dtype=np.float64)
    b = np.asarray(sample, dtype=np.float64)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    lo, hi = range_
    if not np.isfinite(lo) or not np.isfinite(hi) or math.isclose(lo, hi):
        lo = float(min(a.min(), b.min())) - 0.5
        hi = float(max(a.max(), b.max())) + 0.5
    ha, edges = np.histogram(a, bins=bins, range=(lo, hi))
    hb, _ = np.histogram(b, bins=edges)
    pa = ha / max(float(ha.sum()), 1.0)
    pb = hb / max(float(hb.sum()), 1.0)
    mask_a = pa > 0.0
    mask_b = pb > 0.0
    m = 0.5 * (pa + pb)
    js = 0.5 * float(np.sum(pa[mask_a] * np.log(pa[mask_a] / np.maximum(m[mask_a], 1.0e-300))))
    js += 0.5 * float(np.sum(pb[mask_b] * np.log(pb[mask_b] / np.maximum(m[mask_b], 1.0e-300))))
    ks = stats.ks_2samp(a, b)
    native_std = float(np.std(a, ddof=1))
    sample_std = float(np.std(b, ddof=1))
    native_se = native_std / math.sqrt(max(len(a), 1))
    return {
        "n_native": int(len(a)),
        "n_sample": int(len(b)),
        "native_mean": float(np.mean(a)),
        "sample_mean": float(np.mean(b)),
        "native_std": native_std,
        "sample_std": sample_std,
        "mean_shift_native_se": float((np.mean(b) - np.mean(a)) / max(native_se, 1.0e-300)),
        "standardized_mean_shift": float((np.mean(b) - np.mean(a)) / max(native_std, 1.0e-300)),
        "std_ratio": float(sample_std / max(native_std, 1.0e-300)),
        "total_variation": 0.5 * float(np.sum(np.abs(pa - pb))),
        "jensen_shannon": js,
        "wasserstein": float(stats.wasserstein_distance(a, b)),
        "ks_statistic": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
    }


def plot_overlay(path: Path, key: str, ensembles: dict[str, dict[str, np.ndarray]], bins: int) -> None:
    vals = [np.asarray(e[key], dtype=np.float64) for e in ensembles.values()]
    finite = np.concatenate([v[np.isfinite(v)] for v in vals])
    lo, hi = float(np.min(finite)), float(np.max(finite))
    if math.isclose(lo, hi):
        lo -= 0.5
        hi += 0.5
    colors = {
        "native_L16": "black",
        "affine_raw": "#d95f02",
        "affine_patch": "#e6ab02",
        "spline_raw": "#1b9e77",
        "spline_patch": "#7570b3",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 4.0), constrained_layout=True)
    for label, data in ensembles.items():
        histtype = "stepfilled" if label.endswith("patch") else "step"
        alpha = 0.20 if histtype == "stepfilled" else 1.0
        ax.hist(
            data[key],
            bins=bins,
            range=(lo, hi),
            density=True,
            histtype=histtype,
            linewidth=1.5,
            alpha=alpha,
            color=colors.get(label),
            label=label,
        )
    ax.set_xlabel(key)
    ax.set_ylabel("density")
    ax.legend(frameon=False, fontsize=8)
    ax.tick_params(direction="in", top=True, right=True)
    fig.savefig(path)
    plt.close(fig)


def sample_phi(model, coarse: np.ndarray, stats_: dict[str, Any], kernel: np.ndarray, args: ArgsShim, seed: int) -> dict[str, np.ndarray]:
    detail, logq, zmax, logdet = sample_model(model, coarse, stats_, args, seed)
    phi, _ = inverse_kernel(assemble_psi(coarse, detail), kernel)
    return {"detail": detail, "phi": phi.astype(np.float32), "logq": logq, "zmax": zmax, "logdet": logdet}


def patch_chain(
    model,
    coarse: np.ndarray,
    stats_: dict[str, Any],
    kernel: np.ndarray,
    action: ActionSpec,
    args: ArgsShim,
    *,
    seed_base: int,
    sweeps: int,
    initial: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    n = len(coarse)
    rng = np.random.default_rng(seed_base + 100)
    if initial is None:
        current = sample_phi(model, coarse, stats_, kernel, args, seed_base)
    else:
        current = {k: np.array(v, copy=True) for k, v in initial.items()}
    current_s = np.asarray(action_total(current["phi"], action), dtype=np.float64)
    current_logq = np.asarray(current["logq"], dtype=np.float64)
    diag_rows = []
    sweep_rows = []
    accepted_by_chain = np.zeros(n, dtype=np.int64)
    rejection_streak = np.zeros(n, dtype=np.int64)
    longest_by_chain = np.zeros(n, dtype=np.int64)
    mean_history: dict[str, list[float]] = {k: [] for k in ["action_density", "phi2", "phi4", "local_kurtosis_ratio", "m2", "m4", "G_pmin_avg"]}
    for sweep in range(1, sweeps + 1):
        prop = sample_phi(model, coarse, stats_, kernel, args, seed_base + 1000 + sweep)
        prop_s = np.asarray(action_total(prop["phi"], action), dtype=np.float64)
        prop_logq = np.asarray(prop["logq"], dtype=np.float64)
        delta_s = prop_s - current_s
        log_q_ratio = current_logq - prop_logq
        log_acc = -delta_s + log_q_ratio
        acc_prob = np.exp(np.minimum(log_acc, 0.0))
        accepted = np.log(rng.random(n)) < np.minimum(log_acc, 0.0)
        diag_rows.extend(
            {
                "sweep": sweep,
                "config_index": i,
                "DeltaS": float(delta_s[i]),
                "log_q_forward": float(prop_logq[i]),
                "log_q_reverse": float(current_logq[i]),
                "log_q_ratio": float(log_q_ratio[i]),
                "log_acceptance": float(log_acc[i]),
                "acceptance_probability": float(acc_prob[i]),
                "accepted": int(accepted[i]),
            }
            for i in range(n)
        )
        if np.any(accepted):
            for key in ["detail", "phi", "logq", "zmax", "logdet"]:
                current[key][accepted] = prop[key][accepted]
            current_s[accepted] = prop_s[accepted]
            current_logq[accepted] = prop_logq[accepted]
        accepted_by_chain += accepted.astype(np.int64)
        rejection_streak = np.where(accepted, 0, rejection_streak + 1)
        longest_by_chain = np.maximum(longest_by_chain, rejection_streak)
        obs_cfg, _ens = per_config_observables(current["phi"], action)
        for key in mean_history:
            mean_history[key].append(float(np.mean(obs_cfg[key])))
        sweep_rows.append(
            {
                "sweep": sweep,
                "accepted": int(np.sum(accepted)),
                "attempts": int(n),
                "acceptance": float(np.mean(accepted)),
                "cumulative_acceptance": float(np.sum(accepted_by_chain) / (n * sweep)),
                "longest_rejection_streak": int(np.max(longest_by_chain)),
            }
        )
    return {
        "final": current,
        "proposal_diagnostics": diag_rows,
        "sweep_rows": sweep_rows,
        "accepted_by_chain": accepted_by_chain,
        "longest_by_chain": longest_by_chain,
        "mean_history": mean_history,
    }


def summarize_diag(rows: list[dict[str, Any]], label: str) -> list[dict[str, Any]]:
    out = []
    for subset_name, subset in [
        ("all", rows),
        ("accepted", [r for r in rows if int(r["accepted"]) == 1]),
        ("rejected", [r for r in rows if int(r["accepted"]) == 0]),
    ]:
        if not subset:
            continue
        for key in DIAG_KEYS:
            x = np.asarray([float(r[key]) for r in subset], dtype=np.float64)
            out.append(
                {
                    "ensemble": label,
                    "subset": subset_name,
                    "quantity": key,
                    "n": int(len(x)),
                    "mean": float(np.mean(x)),
                    "std": float(np.std(x, ddof=1)) if len(x) > 1 else float("nan"),
                    "median": float(np.median(x)),
                    "p05": float(np.quantile(x, 0.05)),
                    "p50": float(np.quantile(x, 0.50)),
                    "p95": float(np.quantile(x, 0.95)),
                    "p99": float(np.quantile(x, 0.99)),
                    "min": float(np.min(x)),
                    "max": float(np.max(x)),
                }
            )
    return out


def autocorr_1d(x: np.ndarray, max_lag: int | None = None) -> tuple[float, float]:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 4 or np.std(x) == 0.0:
        return float("nan"), float("nan")
    if max_lag is None:
        max_lag = min(n // 2, 100)
    y = x - np.mean(x)
    var = float(np.dot(y, y) / n)
    tau = 1.0
    for lag in range(1, max_lag + 1):
        rho = float(np.dot(y[:-lag], y[lag:]) / ((n - lag) * var))
        if rho <= 0.0:
            break
        tau += 2.0 * rho
    ess = n / max(tau, 1.0e-300)
    return tau, ess


def chain_diagnostics(chain: dict[str, Any], action: ActionSpec, label: str, burn_in: int) -> list[dict[str, Any]]:
    sweep_rows = chain["sweep_rows"]
    post = [r for r in sweep_rows if int(r["sweep"]) > burn_in]
    total_accepted = sum(int(r["accepted"]) for r in post)
    total_attempts = sum(int(r["attempts"]) for r in post)
    rows = [
        {
            "ensemble": label,
            "quantity": "acceptance_after_burn_in",
            "value": float(total_accepted / max(total_attempts, 1)),
            "n": int(total_attempts),
        },
        {
            "ensemble": label,
            "quantity": "acceptance_by_chain_mean",
            "value": float(np.mean(chain["accepted_by_chain"] / max(len(sweep_rows), 1))),
            "n": int(len(chain["accepted_by_chain"])),
        },
        {
            "ensemble": label,
            "quantity": "longest_rejection_streak",
            "value": float(np.max(chain["longest_by_chain"])),
            "n": int(len(chain["longest_by_chain"])),
        },
        {
            "ensemble": label,
            "quantity": "mean_longest_rejection_streak_by_chain",
            "value": float(np.mean(chain["longest_by_chain"])),
            "n": int(len(chain["longest_by_chain"])),
        },
    ]
    for key, values in chain["mean_history"].items():
        tau, ess = autocorr_1d(np.asarray(values[burn_in:], dtype=np.float64))
        rows.append({"ensemble": label, "quantity": f"{key}_iat_sweep_mean", "value": tau, "n": int(len(values[burn_in:]))})
        rows.append({"ensemble": label, "quantity": f"{key}_ess_sweep_mean", "value": ess, "n": int(len(values[burn_in:]))})
    phi = chain["final"]["phi"]
    cfg, ens = per_config_observables(phi, action)
    for key in ["action_density", "phi2", "phi4", "local_kurtosis_ratio", "m2", "m4", "Binder_U4", "xi_over_L"]:
        value = ens[key] if key in ens else float(np.mean(cfg[key]))
        rows.append({"ensemble": label, "quantity": f"stationary_{key}", "value": float(value), "n": int(len(phi))})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--affine-checkpoint", type=Path, required=True)
    ap.add_argument("--spline-checkpoint", type=Path, required=True)
    ap.add_argument("--kernel-path", type=Path, default=Path("perfect_blocking/perfect_blocking_lam1p0/kernels/selected_for_upscaling/best_7x7_phi2_nn_guarded_eta_included.json"))
    ap.add_argument("--coarse-config-source", type=Path, default=Path("data/configs_phi4_2d/lam1p0_kappac0p340301_L8/configs.npz"))
    ap.add_argument("--fine-config-source", type=Path, default=Path("data/configs_phi4_2d/lam1p0_kappac0p340301_L16/configs.npz"))
    ap.add_argument("--n-samples", type=int, default=5000)
    ap.add_argument("--patch-sweeps", type=int, default=100)
    ap.add_argument("--chain-sweeps", type=int, default=100)
    ap.add_argument("--burn-in", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=2026071705)
    ap.add_argument("--bins", type=int, default=60)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    out = args.run_dir / "validation_large_sample"
    for sub in ["figures", "configs", "diagnostics", "logs"]:
        (out / sub).mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    shim = ArgsShim(args.batch_size, args.device)
    action = ActionSpec("phi4_nn", 1.0, 0.340301)

    kernel, kernel_raw = load_kernel_matrix(args.kernel_path)
    if not bool(kernel_raw.get("kernel_coefficients_include_eta_scale")):
        raise RuntimeError("kernel does not declare eta-included coefficients")
    if abs(float(kernel.sum()) - ETA_SCALE) > 1.0e-10:
        raise RuntimeError(f"kernel sum {kernel.sum()} does not match eta_scale {ETA_SCALE}")
    phi8 = load_phi(args.coarse_config_source)[: args.n_samples].astype(np.float32)
    phi16 = load_phi(args.fine_config_source)[: args.n_samples].astype(np.float32)
    if len(phi8) < args.n_samples or len(phi16) < args.n_samples:
        raise RuntimeError("not enough native configs for requested n-samples")

    affine, affine_ckpt = load_affine_model(args.affine_checkpoint, device)
    spline, spline_ckpt = load_residual_spline_model(args.spline_checkpoint, device)
    affine_stats = affine_ckpt["state"]["stats"]
    spline_stats = spline_ckpt["state"]["stats"]

    np.save(out / "configs" / "native_L16_reference.npy", phi16)
    np.save(out / "configs" / "source_coarse_L8.npy", phi8)
    seeds = {
        "base_seed": args.seed,
        "affine_raw_seed": args.seed + 11,
        "spline_raw_seed": args.seed + 11,
        "affine_patch_seed_base": args.seed + 10000,
        "spline_patch_seed_base": args.seed + 10000,
        "matched_seed_note": "raw proposal seed bases and patch RNG/proposal seed offsets are matched across affine and spline models",
    }
    write_json(out / "diagnostics" / "seeds.json", seeds)

    affine_raw = sample_phi(affine, phi8, affine_stats, kernel, shim, seeds["affine_raw_seed"])
    spline_raw = sample_phi(spline, phi8, spline_stats, kernel, shim, seeds["spline_raw_seed"])
    np.save(out / "configs" / "affine_epoch137_raw.npy", affine_raw["phi"])
    np.save(out / "configs" / "spline_epoch161_raw.npy", spline_raw["phi"])
    np.save(out / "configs" / "affine_epoch137_raw_detail.npy", affine_raw["detail"])
    np.save(out / "configs" / "spline_epoch161_raw_detail.npy", spline_raw["detail"])

    affine_chain = patch_chain(
        affine,
        phi8,
        affine_stats,
        kernel,
        action,
        shim,
        seed_base=seeds["affine_patch_seed_base"],
        sweeps=args.patch_sweeps,
        initial=affine_raw,
    )
    spline_chain = patch_chain(
        spline,
        phi8,
        spline_stats,
        kernel,
        action,
        shim,
        seed_base=seeds["spline_patch_seed_base"],
        sweeps=args.patch_sweeps,
        initial=spline_raw,
    )
    np.save(out / "configs" / "affine_epoch137_after_patch.npy", affine_chain["final"]["phi"])
    np.save(out / "configs" / "spline_epoch161_after_patch.npy", spline_chain["final"]["phi"])
    write_csv(out / "diagnostics" / "affine_patch_sweep_history.csv", affine_chain["sweep_rows"])
    write_csv(out / "diagnostics" / "spline_patch_sweep_history.csv", spline_chain["sweep_rows"])
    write_csv(out / "diagnostics" / "affine_proposal_diagnostics_per_proposal.csv", affine_chain["proposal_diagnostics"])
    write_csv(out / "diagnostics" / "spline_proposal_diagnostics_per_proposal.csv", spline_chain["proposal_diagnostics"])

    ensembles_phi = {
        "native_L16": phi16,
        "affine_raw": affine_raw["phi"],
        "affine_patch": affine_chain["final"]["phi"],
        "spline_raw": spline_raw["phi"],
        "spline_patch": spline_chain["final"]["phi"],
    }
    ensembles_cfg: dict[str, dict[str, np.ndarray]] = {}
    ensembles_summary: dict[str, dict[str, float]] = {}
    for label, phi in ensembles_phi.items():
        cfg, ens = per_config_observables(phi, action)
        ensembles_cfg[label] = cfg
        ensembles_summary[label] = ens

    obs_rows = []
    for label, ens in ensembles_summary.items():
        for key in OBS_KEYS:
            if key in ens:
                obs_rows.append({"ensemble": label, "observable": key, "value": ens[key], "n": int(len(ensembles_phi[label]))})
            elif key in ensembles_cfg[label]:
                x = ensembles_cfg[label][key]
                obs_rows.append({"ensemble": label, "observable": key, "value": float(np.mean(x)), "std": float(np.std(x, ddof=1)), "n": int(len(x))})
    write_csv(out / "observable_summary.csv", obs_rows)

    hist_rows = []
    for key in sorted(set(HIST_KEYS + ["G_pmin_avg"])):
        all_vals = np.concatenate([ensembles_cfg[label][key][np.isfinite(ensembles_cfg[label][key])] for label in ensembles_cfg])
        range_ = (float(np.min(all_vals)), float(np.max(all_vals)))
        for label in ["affine_raw", "affine_patch", "spline_raw", "spline_patch"]:
            row = {
                "observable": key,
                "proposal_ensemble": label,
                **histogram_metrics(ensembles_cfg["native_L16"][key], ensembles_cfg[label][key], args.bins, range_),
            }
            hist_rows.append(row)
    write_csv(out / "histogram_scores.csv", hist_rows)

    for key in HIST_KEYS:
        plot_overlay(out / "figures" / f"hist_{key}.pdf", key, ensembles_cfg, args.bins)
    for label, rows in [("affine", affine_chain["proposal_diagnostics"]), ("spline", spline_chain["proposal_diagnostics"])]:
        for key in ["DeltaS", "log_q_ratio", "log_acceptance"]:
            fig, ax = plt.subplots(figsize=(5.4, 3.5), constrained_layout=True)
            x = np.asarray([float(r[key]) for r in rows], dtype=np.float64)
            ax.hist(x[np.isfinite(x)], bins=args.bins, histtype="stepfilled", alpha=0.35, density=True)
            ax.set_xlabel(key)
            ax.set_ylabel("density")
            ax.tick_params(direction="in", top=True, right=True)
            fig.savefig(out / "figures" / f"{label}_{key}_diagnostic.pdf")
            plt.close(fig)

    prop_summary = summarize_diag(affine_chain["proposal_diagnostics"], "affine_epoch137") + summarize_diag(
        spline_chain["proposal_diagnostics"], "spline_epoch161"
    )
    write_csv(out / "proposal_diagnostics.csv", prop_summary)
    chain_rows = chain_diagnostics(affine_chain, action, "affine_epoch137", args.burn_in) + chain_diagnostics(
        spline_chain, action, "spline_epoch161", args.burn_in
    )
    write_csv(out / "chain_diagnostics.csv", chain_rows)
    write_csv(out / "diagnostics" / "affine_observable_time_history.csv", [{"sweep": i + 1, **{k: v[i] for k, v in affine_chain["mean_history"].items()}} for i in range(args.patch_sweeps)])
    write_csv(out / "diagnostics" / "spline_observable_time_history.csv", [{"sweep": i + 1, **{k: v[i] for k, v in spline_chain["mean_history"].items()}} for i in range(args.patch_sweeps)])

    # Compact summary table.
    def score(label: str, key: str) -> dict[str, Any]:
        return next(r for r in hist_rows if r["proposal_ensemble"] == label and r["observable"] == key)

    summary_rows = []
    for label in ["affine_raw", "affine_patch", "spline_raw", "spline_patch"]:
        for key in ["action_density", "phi2", "phi4", "local_kurtosis_ratio", "NN", "m2", "m4", "G_pmin_avg"]:
            r = score(label, key)
            summary_rows.append(
                {
                    "ensemble": label,
                    "observable": key,
                    "ks": r["ks_statistic"],
                    "wasserstein": r["wasserstein"],
                    "jensen_shannon": r["jensen_shannon"],
                    "mean_shift_native_se": r["mean_shift_native_se"],
                    "std_ratio": r["std_ratio"],
                }
            )
    write_csv(out / "summary.csv", summary_rows)

    affine_acc = next(r for r in chain_rows if r["ensemble"] == "affine_epoch137" and r["quantity"] == "acceptance_after_burn_in")["value"]
    spline_acc = next(r for r in chain_rows if r["ensemble"] == "spline_epoch161" and r["quantity"] == "acceptance_after_burn_in")["value"]
    raw_improves = score("spline_raw", "local_kurtosis_ratio")["ks_statistic"] < score("affine_raw", "local_kurtosis_ratio")["ks_statistic"]
    patch_improves = score("spline_patch", "local_kurtosis_ratio")["ks_statistic"] < score("affine_patch", "local_kurtosis_ratio")["ks_statistic"]
    lines = [
        "# Lambda=1.0 L8->L16 Large RQ-Spline Validation",
        "",
        f"- samples per ensemble: `{args.n_samples}`",
        f"- patch/chain sweeps: `{args.patch_sweeps}`",
        f"- kernel path: `{args.kernel_path}`",
        f"- kernel sum: `{float(kernel.sum()):.15g}`",
        f"- kernel_coefficients_include_eta_scale: `{kernel_raw.get('kernel_coefficients_include_eta_scale')}`",
        f"- affine acceptance after burn-in: `{affine_acc:.6g}`",
        f"- spline acceptance after burn-in: `{spline_acc:.6g}`",
        f"- spline raw local-kurtosis KS improves over affine raw: `{raw_improves}`",
        f"- spline patch local-kurtosis KS improves over affine patch: `{patch_improves}`",
        "",
        "## Key Histogram Scores",
        "",
        "| ensemble | observable | KS | Wasserstein | JS | mean shift/native SE | std ratio |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        if row["observable"] in ["action_density", "phi4", "local_kurtosis_ratio", "NN", "G_pmin_avg"]:
            lines.append(
                f"| {row['ensemble']} | {row['observable']} | {float(row['ks']):.6g} | {float(row['wasserstein']):.6g} | {float(row['jensen_shannon']):.6g} | {float(row['mean_shift_native_se']):.6g} | {float(row['std_ratio']):.6g} |"
            )
    lines += [
        "",
        "## Interpretation",
        "",
        "This validation distinguishes proposal-pair diagnostics from the actual independence chain acceptance after burn-in. The after-patch ensembles are the accepted states after identical fixed-coarse independence Metropolis sweeps for affine and spline proposals.",
        "",
        "Do not use this report as authorization for L16->L32 by itself; it is the evidence base for deciding whether that pilot is justified.",
    ]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    write_json(
        out / "diagnostics" / "validation_manifest.json",
        {
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "run_dir": str(args.run_dir),
            "affine_checkpoint": str(args.affine_checkpoint),
            "spline_checkpoint": str(args.spline_checkpoint),
            "n_samples": args.n_samples,
            "patch_sweeps": args.patch_sweeps,
            "chain_sweeps": args.chain_sweeps,
            "kernel_sum": float(kernel.sum()),
            "eta_scale": ETA_SCALE,
            "raw_configs_saved_under_validation_large_sample": True,
        },
    )
    print(json.dumps({"status": "completed", "out": str(out), "spline_acceptance_after_burn_in": spline_acc}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
