#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
from compare_lam1p0_zeroshot_vs_balanced import (  # noqa: E402
    hist_overlap,
    ks_stat,
    obs_arrays,
    sample_from_latents,
    union_edges,
    wasserstein_1,
)
from run_lam1p0_l16to32_rqspline_zeroshot import build_model_from_checkpoint, stationary_stats  # noqa: E402
from train_lam1p0_flow_detail_pilot import assemble_psi, inverse_kernel, load_kernel_matrix, load_phi, split_pairs  # noqa: E402


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
PLOT_KEYS = ["action_density", "phi2", "phi4", "local_kurtosis_ratio", "NN", "m2", "m4", "G_pmin_avg"]


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


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str), encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def model_hash(state: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for key in sorted(state):
        h.update(key.encode())
        arr = np.ascontiguousarray(state[key].detach().cpu().numpy())
        h.update(arr.view(np.uint8))
    return h.hexdigest()


def metric_rows(native_obs: dict[str, np.ndarray], sample_obs: dict[str, np.ndarray], edges: dict[str, np.ndarray], label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    qlist = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    for obs in OBS_KEYS:
        native = np.asarray(native_obs[obs], dtype=np.float64)
        sample = np.asarray(sample_obs[obs], dtype=np.float64)
        native_std = float(np.std(native, ddof=1))
        row: dict[str, Any] = {
            "label": label,
            "observable": obs,
            "n_native": int(len(native)),
            "n_sample": int(len(sample)),
            "native_mean": float(np.mean(native)),
            "native_std": native_std,
            "sample_mean": float(np.mean(sample)),
            "sample_std": float(np.std(sample, ddof=1)),
            "mean_shift_native_sigma": float((np.mean(sample) - np.mean(native)) / max(native_std, 1.0e-300)),
            "std_ratio": float(np.std(sample, ddof=1) / max(native_std, 1.0e-300)),
            "ks_statistic": ks_stat(native, sample),
            "wasserstein_1": wasserstein_1(native, sample),
            "histogram_overlap_coefficient": hist_overlap(native, sample, edges[obs]),
            "sample_min": float(np.min(sample)),
            "sample_max": float(np.max(sample)),
        }
        for q in qlist:
            tag = f"q{int(round(100 * q)):02d}"
            nq = float(np.quantile(native, q))
            sq = float(np.quantile(sample, q))
            row[f"native_{tag}"] = nq
            row[f"sample_{tag}"] = sq
            row[f"{tag}_difference"] = sq - nq
            row[f"{tag}_ratio"] = sq / nq if nq != 0.0 else float("nan")
        for q in [0.01, 0.05, 0.10]:
            row[f"frac_below_native_q{int(round(100 * q)):02d}"] = float(np.mean(sample < np.quantile(native, q)))
        for q in [0.90, 0.95, 0.99]:
            row[f"frac_above_native_q{int(round(100 * q)):02d}"] = float(np.mean(sample > np.quantile(native, q)))
        rows.append(row)
    return rows


def save_histograms(
    out: Path,
    native_obs: dict[str, np.ndarray],
    sample_obs: dict[str, np.ndarray],
    edges: dict[str, np.ndarray],
    label: str,
    native_label: str,
) -> None:
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for obs in PLOT_KEYS:
        for semilog in [False, True]:
            fig, ax = plt.subplots(figsize=(6.5, 4.1), constrained_layout=True)
            ax.hist(native_obs[obs], bins=edges[obs], density=True, histtype="step", lw=2.2, color="black", label=native_label)
            ax.hist(sample_obs[obs], bins=edges[obs], density=True, histtype="step", lw=1.8, color="#d62728", label=label)
            if semilog:
                ax.set_yscale("log")
            ax.set_xlabel(obs)
            ax.set_ylabel("density")
            ax.grid(alpha=0.2)
            ax.legend(frameon=False, fontsize=8)
            suffix = "semilog" if semilog else "linear"
            safe = "local_kurtosis" if obs == "local_kurtosis_ratio" else ("Gpmin" if obs == "G_pmin_avg" else obs)
            fig.savefig(fig_dir / f"{safe}_{suffix}.pdf")
            fig.savefig(fig_dir / f"{safe}_{suffix}.png", dpi=180)
            plt.close(fig)
    for obs in ["action_density", "phi2", "phi4", "local_kurtosis_ratio"]:
        fig, ax = plt.subplots(figsize=(6.5, 4.1), constrained_layout=True)
        for name, vals, color in [(native_label, native_obs[obs], "black"), (label, sample_obs[obs], "#d62728")]:
            x = np.sort(np.asarray(vals, dtype=np.float64))
            y = np.arange(1, len(x) + 1, dtype=np.float64) / len(x)
            ax.plot(x, y, lw=2.1 if name == "native L32" else 1.7, color=color, label=name)
        ax.set_xlabel(obs)
        ax.set_ylabel("CDF")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, fontsize=8)
        safe = "local_kurtosis" if obs == "local_kurtosis_ratio" else obs
        fig.savefig(fig_dir / f"{safe}_cdf.pdf")
        fig.savefig(fig_dir / f"{safe}_cdf.png", dpi=180)
        plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--kernel", default=Path("perfect_blocking/perfect_blocking_lam1p0/kernels/final/chosen_kernel.json"), type=Path)
    p.add_argument("--native-l32", default=Path("data/configs_phi4_2d/lam1p0_kappac0p340301_L32/configs.npz"), type=Path)
    p.add_argument("--native-source", type=Path, default=None, help="Generic native fine ensemble path. Overrides --native-l32.")
    p.add_argument("--coarse-source", type=Path, default=None, help="Optional independent coarse ensemble. When omitted, the coarse fields are obtained by blocking the native fine configurations.")
    p.add_argument("--coarse-lattice", type=int, default=16)
    p.add_argument("--fine-lattice", type=int, default=None)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--count", type=int, default=5000)
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--seed", type=int, default=202607191610)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--latent-mixture-weight", type=float, default=0.0, help="Mixture probability for a hotter latent Gaussian proposal; zero uses the ordinary flow.")
    p.add_argument("--latent-temperature", type=float, default=1.0, help="Latent standard-deviation multiplier for the mixture component (below one is a cold component).")
    p.add_argument("--label", default="N2000_best_patch_raw_upscaled")
    p.add_argument(
        "--upscaled-config-output",
        type=Path,
        default=None,
        help="Optional .npz path for the raw generated fine fields (key: configs).",
    )
    args = p.parse_args()
    if not 0.0 <= args.latent_mixture_weight < 1.0:
        raise ValueError("--latent-mixture-weight must lie in [0, 1)")
    if args.latent_temperature <= 0.0:
        raise ValueError("--latent-temperature must be positive")

    out = PROJECT_ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    ckpt_path = PROJECT_ROOT / args.checkpoint
    kernel_path = PROJECT_ROOT / args.kernel
    native_arg = args.native_l32 if args.native_source is None else args.native_source
    native_path = PROJECT_ROOT / native_arg
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    lc = int(args.coarse_lattice)
    lf = int(args.fine_lattice) if args.fine_lattice is not None else 2 * lc
    model, load_report = build_model_from_checkpoint(ckpt, lattice_size=lc, device=device)
    stats = stationary_stats(ckpt["state"]["stats"], lc=lc)
    kernel, kernel_json = load_kernel_matrix(kernel_path)
    phi32_all = load_phi(native_path)
    coarse_all = None if args.coarse_source is None else load_phi(PROJECT_ROOT / args.coarse_source)
    available = len(phi32_all) if coarse_all is None else min(len(phi32_all), len(coarse_all))
    stop = min(int(args.start_index) + int(args.count), available)
    idx = np.arange(int(args.start_index), stop, dtype=np.int64)
    phi32 = phi32_all[idx]
    if coarse_all is None:
        coarse = split_pairs(phi32, kernel)["coarse"]
        coarse_source_label = "blocked native fine configurations"
    else:
        coarse = coarse_all[idx]
        if coarse.shape[1:] != (lc, lc):
            raise ValueError(f"coarse source has shape {coarse.shape}; expected lattice {lc}")
        coarse_source_label = str(args.coarse_source)

    rng = np.random.default_rng(int(args.seed))
    if phi32.shape[1] != lf or phi32.shape[2] != lf:
        raise ValueError(f"native source has shape {phi32.shape}; expected fine lattice {lf}")
    latents = rng.standard_normal((len(phi32), 3, lc, lc), dtype=np.float32)
    hot_mask = rng.random(len(phi32)) < float(args.latent_mixture_weight)
    if np.any(hot_mask):
        latents[hot_mask] *= float(args.latent_temperature)
    detail, logq, zmax, logdet = sample_from_latents(model, coarse, stats, latents, batch_size=int(args.batch_size), device=device)
    if np.any(hot_mask):
        flat_z = latents.reshape(len(latents), -1).astype(np.float64)
        dim = flat_z.shape[1]
        logp1 = -0.5 * np.sum(flat_z * flat_z + math.log(2.0 * math.pi), axis=1)
        temp = float(args.latent_temperature)
        logpt = -0.5 * np.sum((flat_z / temp) ** 2 + math.log(2.0 * math.pi) + 2.0 * math.log(temp), axis=1)
        mix = float(args.latent_mixture_weight)
        logbase_mix = np.logaddexp(math.log1p(-mix) + logp1, math.log(mix) + logpt)
        logq = logq + logbase_mix - logp1
    print(json.dumps({"stage": "sampled_detail", "count": len(phi32)}), flush=True)
    phi_up, _ = inverse_kernel(assemble_psi(coarse, detail), kernel)
    if args.upscaled_config_output is not None:
        upscaled_path = PROJECT_ROOT / args.upscaled_config_output
        upscaled_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(upscaled_path, configs=phi_up.astype(np.float32, copy=False))
        print(json.dumps({"stage": "saved_fields", "path": str(args.upscaled_config_output)}), flush=True)

    action = ActionSpec("phi4_nn", 1.0, 0.340301)
    native_label = f"native L{lf}"
    print(json.dumps({"stage": "measuring_native", "count": len(phi32)}), flush=True)
    native_obs = obs_arrays(phi32, native_label, action)
    print(json.dumps({"stage": "measuring_upscaled", "count": len(phi_up)}), flush=True)
    sample_obs = obs_arrays(phi_up, args.label, action)
    print(json.dumps({"stage": "writing_comparison"}), flush=True)
    edges = {obs: union_edges([native_obs[obs], sample_obs[obs]]) for obs in OBS_KEYS}
    rows = metric_rows(native_obs, sample_obs, edges, args.label)
    write_csv(out / "raw_observable_metrics.csv", rows)

    per_rows: list[dict[str, Any]] = []
    for j, source_idx in enumerate(idx):
        row = {"source_config_index": int(source_idx), "label": args.label}
        for obs in OBS_KEYS:
            row[f"native_{obs}"] = float(native_obs[obs][j])
            row[f"upscaled_{obs}"] = float(sample_obs[obs][j])
        per_rows.append(row)
    write_csv(out / "per_configuration_observables.csv", per_rows)
    save_histograms(out, native_obs, sample_obs, edges, args.label, native_label)

    manifest = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_hash": sha256_file(ckpt_path),
        "model_state_hash": model_hash(ckpt["model_state"]),
        "kernel": str(args.kernel),
        "kernel_hash": sha256_file(kernel_path),
        "kernel_sum": float(np.sum(kernel)),
        "kernel_coefficients_include_eta_scale": kernel_json.get("kernel_coefficients_include_eta_scale"),
        "native_source": str(native_arg),
        "coarse_source": coarse_source_label,
        "coarse_lattice": lc,
        "fine_lattice": lf,
        "start_index": int(args.start_index),
        "count": int(len(phi32)),
        "seed": int(args.seed),
        "batch_size": int(args.batch_size),
        "latent_mixture": {
            "weight": float(args.latent_mixture_weight),
            "temperature": float(args.latent_temperature),
            "hot_component_count": int(np.count_nonzero(hot_mask)),
            "logq_is_exact_mixture_density": True,
        },
        "upscaled_config_output": str(args.upscaled_config_output) if args.upscaled_config_output is not None else None,
        "load_report": load_report,
        "nonfinite_count": int(np.sum(~np.isfinite(phi_up))),
        "max_abs_z": float(np.max(np.abs(latents))),
        "logq_mean": float(np.mean(logq)),
        "logq_std": float(np.std(logq, ddof=1)),
        "logdet_mean": float(np.mean(logdet)),
        "logdet_std": float(np.std(logdet, ddof=1)),
        "reblocking_max_error": float(np.max(np.abs(coarse - split_pairs(phi_up, kernel)["coarse"]))),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_json(out / "run_manifest.json", manifest)
    lines = [
        f"# Raw Upscaled Vs Native L{lf}",
        "",
        f"- checkpoint: `{args.checkpoint}`",
        f"- checkpoint epoch: `{ckpt.get('epoch')}`",
        f"- kernel: `{args.kernel}`",
        f"- volume: L{lc}->L{lf}",
        f"- coarse source: `{coarse_source_label}`",
        f"- configs: {int(idx[0])}..{int(idx[-1])} ({len(idx)})",
        f"- seed: {int(args.seed)}",
        f"- reblocking max error: {manifest['reblocking_max_error']:.6g}",
        f"- nonfinite count: {manifest['nonfinite_count']}",
        "",
        "## Key Metrics",
        "",
        "| observable | shift | KS | OVL | std ratio | native mean | upscaled mean |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    by_obs = {row["observable"]: row for row in rows}
    for obs in ["action_density", "phi2", "phi4", "local_kurtosis_ratio", "NN", "G_pmin_avg", "m2", "m4"]:
        row = by_obs[obs]
        lines.append(
            f"| {obs} | {row['mean_shift_native_sigma']:.4f} | {row['ks_statistic']:.4f} | "
            f"{row['histogram_overlap_coefficient']:.4f} | {row['std_ratio']:.4f} | "
            f"{row['native_mean']:.8g} | {row['sample_mean']:.8g} |"
        )
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "count": len(idx), "checkpoint_epoch": ckpt.get("epoch"), "reblocking_max_error": manifest["reblocking_max_error"], "nonfinite_count": manifest["nonfinite_count"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
