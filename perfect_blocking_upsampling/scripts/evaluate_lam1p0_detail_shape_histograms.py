#!/usr/bin/env python3
"""Held-out native-L32 histogram evaluation for an exact detail-shape map."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PKG = PROJECT_ROOT / "perfect_blocking_upsampling"
sys.path[:0] = [str(PKG / "src"), str(PKG / "scripts")]
os.environ.setdefault("MPLCONFIGDIR", str((PKG / "logs" / "mplconfig").resolve()))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from perfect_blocking_upsampling.actions import action_total  # noqa: E402
from perfect_blocking_upsampling.blocking import apply_kernel, assemble_psi, inverse_kernel, load_kernel_matrix, load_phi  # noqa: E402
from perfect_blocking_upsampling.io import ActionSpec  # noqa: E402
from run_lam1p0_l16to32_rqspline_zeroshot import build_model_from_checkpoint, sample_model_lattice, stationary_stats  # noqa: E402


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def write_json(path: Path, data: dict[str, object]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def per_config(phi: np.ndarray, action: ActionSpec) -> dict[str, np.ndarray]:
    arr = np.asarray(phi, dtype=np.float64)
    phi2 = np.mean(arr * arr, axis=(1, 2))
    phi4 = np.mean(arr**4, axis=(1, 2))
    nn = 0.5 * (np.mean(arr * np.roll(arr, -1, axis=1), axis=(1, 2)) + np.mean(arr * np.roll(arr, -1, axis=2), axis=(1, 2)))
    diag = np.mean(arr * np.roll(np.roll(arr, -1, axis=1), -1, axis=2), axis=(1, 2))
    twonn = 0.5 * (np.mean(arr * np.roll(arr, -2, axis=1), axis=(1, 2)) + np.mean(arr * np.roll(arr, -2, axis=2), axis=(1, 2)))
    return {
        "action_density": action_total(arr, action) / float(arr.shape[1] * arr.shape[2]),
        "phi2": phi2, "phi4": phi4, "kurtosis": phi4 / (phi2 * phi2), "NN": nn, "diag": diag, "2nn": twonn,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--coarse-source", type=Path, required=True)
    ap.add_argument("--native-fine-source", type=Path, required=True)
    ap.add_argument("--flow-checkpoint", type=Path, required=True)
    ap.add_argument("--kernel-path", type=Path, required=True)
    ap.add_argument("--from-L", type=int, default=16)
    ap.add_argument("--to-L", type=int, default=32)
    ap.add_argument("--n-configs", type=int, default=256)
    ap.add_argument("--start-index", type=int, default=512)
    ap.add_argument("--scale", type=float, required=True)
    ap.add_argument("--tail-strength", type=float, required=True)
    ap.add_argument("--tail-width", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=2026072403)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    run = args.run_dir.resolve()
    (run / "plots").mkdir(parents=True, exist_ok=True)
    (run / "observables").mkdir(parents=True, exist_ok=True)
    write_json(run / "status.json", {"status": "running", "stage": "startup", "run_dir": str(run)})
    print(json.dumps({"run_dir": str(run), "stage": "startup"}), flush=True)

    coarse_all, native_all = load_phi(args.coarse_source), load_phi(args.native_fine_source)
    stop = args.start_index + args.n_configs
    if coarse_all.shape[1:] != (args.from_L, args.from_L) or native_all.shape[1:] != (args.to_L, args.to_L) or stop > len(coarse_all) or stop > len(native_all):
        raise RuntimeError("invalid source lattice or requested held-out index range")
    coarse, native = coarse_all[args.start_index:stop].astype(np.float32), native_all[args.start_index:stop].astype(np.float32)
    kernel, meta = load_kernel_matrix(args.kernel_path)
    if not bool(meta.get("kernel_coefficients_include_eta_scale", False)):
        raise RuntimeError("kernel eta scale must already be included")
    ckpt = torch.load(args.flow_checkpoint, map_location=args.device, weights_only=False)
    model, report = build_model_from_checkpoint(ckpt, args.from_L, torch.device(args.device))
    detail, _, _, _ = sample_model_lattice(model, coarse, stationary_stats(ckpt["state"]["stats"], args.from_L), batch_size=args.batch_size, device=torch.device(args.device), seed=args.seed)
    rms = np.sqrt(np.mean(detail.astype(np.float64) ** 2, axis=(0, 2, 3)))
    u = detail / rms.reshape(1, 3, 1, 1)
    t = np.tanh(u / args.tail_width)
    transformed = (args.scale * (u - args.tail_strength * t**3) * rms.reshape(1, 3, 1, 1)).astype(np.float32)
    deriv = args.scale * (1.0 - (3.0 * args.tail_strength / args.tail_width) * t * t * (1.0 - t * t))
    if np.any(deriv <= 0.0):
        raise RuntimeError("detail map is not invertible")
    raw, _ = inverse_kernel(assemble_psi(coarse, detail), kernel)
    corrected, _ = inverse_kernel(assemble_psi(coarse, transformed), kernel)
    reblocking = float(np.max(np.abs(apply_kernel(corrected, kernel)[:, 0::2, 0::2] - coarse)))
    action = ActionSpec("phi4_nn", 1.0, 0.340301)
    ensembles = {"native": per_config(native, action), "raw_flow": per_config(raw, action), "tail_corrected": per_config(corrected, action)}
    keys = ("action_density", "phi2", "phi4", "kurtosis", "NN", "diag", "2nn")
    for label, values in ensembles.items():
        write_csv(run / "observables" / f"{label}_per_config.csv", [{"config_index": args.start_index + i, **{key: float(values[key][i]) for key in keys}} for i in range(args.n_configs)])
    comparison: list[dict[str, object]] = []
    for key in keys:
        ref = ensembles["native"][key]
        for label in ("raw_flow", "tail_corrected"):
            vals = ensembles[label][key]
            comparison.append({"observable": key, "ensemble": label, "native_mean": float(ref.mean()), "mean": float(vals.mean()),
                               "mean_shift": float(vals.mean() - ref.mean()), "native_std": float(ref.std(ddof=1)),
                               "std": float(vals.std(ddof=1)), "width_ratio": float(vals.std(ddof=1) / ref.std(ddof=1))})
    write_csv(run / "observables" / "central_value_width_comparison.csv", comparison)
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.5))
    for axis, key in zip(axes.flat, keys):
        all_values = np.concatenate([ensembles[label][key] for label in ensembles])
        bins = np.linspace(np.quantile(all_values, 0.002), np.quantile(all_values, 0.998), 45)
        for label, color in (("native", "black"), ("raw_flow", "tab:orange"), ("tail_corrected", "tab:blue")):
            axis.hist(ensembles[label][key], bins=bins, density=True, histtype="step", linewidth=1.6, color=color, label=label)
        axis.set_title(key)
        axis.legend(fontsize=7)
    axes.flat[-1].axis("off")
    fig.tight_layout()
    fig.savefig(run / "plots" / "native_raw_tail_corrected_histograms.pdf")
    fig.savefig(run / "plots" / "native_raw_tail_corrected_histograms.png", dpi=180)
    plt.close(fig)
    write_json(run / "run_config.json", vars(args) | {"sector_rms": rms.tolist(), "kernel_metadata": meta, "flow_load_report": report, "reblocking_max_error": reblocking})
    write_json(run / "status.json", {"status": "completed", "run_dir": str(run), "reblocking_max_error": reblocking})
    print(json.dumps({"run_dir": str(run), "status": "completed", "reblocking_max_error": reblocking, "comparison": comparison}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
