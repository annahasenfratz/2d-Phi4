#!/usr/bin/env python3
"""Evaluate a local-coarse-context exact detail-tail correction.

At each coarse site x, the map is
  u_s(x) -> a(x) [u_s(x) - delta tanh(u_s(x)/b)^3],
where a(x)=a0 exp(slope * standardized(c(x)^2)).  The context depends only
on unchanged coarse coordinates, so the map remains local and its Jacobian is
an exact product of per-site derivatives.
"""
from __future__ import annotations

import argparse
import csv
import gc
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


KEYS = ("action_density", "phi2", "phi4", "kurtosis", "NN")
MAIN_COLUMNS = (
    "chain_id", "sweep", "source_config_index", "source_native_L32_index", "L", "volume",
    "action_density", "total_action", "phi2", "phi4", "local_kurtosis_ratio", "NN", "diag", "2nn", "m", "m2", "m4",
    "G_pmin_x_cfg", "G_pmin_y_cfg", "nonfinite_count",
)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def write_json(path: Path, value: dict[str, object]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def parse_values(text: str) -> list[float]:
    return [float(value) for value in text.split(",") if value.strip()]


def obs(phi: np.ndarray, action: ActionSpec) -> dict[str, np.ndarray]:
    x = np.asarray(phi, dtype=np.float64)
    n, lattice_size, _ = x.shape
    phi2 = np.mean(x * x, axis=(1, 2))
    phi4 = np.mean(x**4, axis=(1, 2))
    nn = 0.5 * (np.mean(x * np.roll(x, -1, axis=1), axis=(1, 2)) + np.mean(x * np.roll(x, -1, axis=2), axis=(1, 2)))
    twonn = 0.5 * (np.mean(x * np.roll(x, -2, axis=1), axis=(1, 2)) + np.mean(x * np.roll(x, -2, axis=2), axis=(1, 2)))
    diag = np.mean(x * np.roll(np.roll(x, -1, axis=1), -1, axis=2), axis=(1, 2))
    m = np.mean(x, axis=(1, 2))
    phase = np.exp(2j * np.pi * np.arange(lattice_size) / lattice_size)
    phi_x = np.tensordot(x, phase, axes=([1], [0])).sum(axis=1)
    phi_y = np.tensordot(x, phase, axes=([2], [0])).sum(axis=1)
    return {
        "action_density": action_total(x, action) / float(lattice_size * lattice_size),
        "phi2": phi2, "phi4": phi4, "kurtosis": phi4 / phi2**2, "NN": nn,
        "diag": diag, "2nn": twonn, "m": m, "m2": m * m, "m4": m**4,
        "G_pmin_x_cfg": np.abs(phi_x) ** 2 / float(lattice_size * lattice_size),
        "G_pmin_y_cfg": np.abs(phi_y) ** 2 / float(lattice_size * lattice_size),
    }


def main_measurement_rows(values: dict[str, np.ndarray], source_indices: np.ndarray, lattice_size: int) -> list[dict[str, object]]:
    volume = lattice_size * lattice_size
    rows: list[dict[str, object]] = []
    for chain_id, source_index in enumerate(source_indices):
        row: dict[str, object] = {
            "chain_id": chain_id, "sweep": 0, "source_config_index": int(source_index),
            "source_native_L32_index": int(source_index), "L": lattice_size, "volume": volume,
            "total_action": float(values["action_density"][chain_id] * volume),
            "nonfinite_count": int(not all(np.isfinite(values[key][chain_id]) for key in values)),
        }
        for key in ("action_density", "phi2", "phi4", "NN", "diag", "2nn", "m", "m2", "m4", "G_pmin_x_cfg", "G_pmin_y_cfg"):
            row[key] = float(values[key][chain_id])
        row["local_kurtosis_ratio"] = float(values["kurtosis"][chain_id])
        rows.append({key: row[key] for key in MAIN_COLUMNS})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--coarse-source", type=Path, required=True)
    ap.add_argument("--native-fine-source", type=Path, required=True)
    ap.add_argument("--flow-checkpoint", type=Path, required=True)
    ap.add_argument("--kernel-path", type=Path, required=True)
    ap.add_argument("--coarse-lattice-size", type=int, default=None, help="Infer from --coarse-source when omitted.")
    ap.add_argument("--n-configs", type=int, default=256)
    ap.add_argument("--start-index", type=int, default=768)
    ap.add_argument("--context-bank-count", type=int, default=1000)
    ap.add_argument("--base-scale", type=float, default=1.02)
    ap.add_argument("--base-scales", default=None, help="Optional comma-separated scale scan; overrides --base-scale.")
    ap.add_argument("--tail-strength", type=float, default=0.15)
    ap.add_argument("--tail-strengths", default=None, help="Optional comma-separated tail-strength scan; overrides --tail-strength.")
    ap.add_argument("--tail-width", type=float, default=2.0)
    ap.add_argument("--slopes", default="-0.10,-0.05,0.0,0.05,0.10")
    ap.add_argument("--seed", type=int, default=2026072404)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    slopes = parse_values(args.slopes)
    base_scales = parse_values(args.base_scales) if args.base_scales else [args.base_scale]
    tail_strengths = parse_values(args.tail_strengths) if args.tail_strengths else [args.tail_strength]
    run = args.run_dir.resolve()
    (run / "observables").mkdir(parents=True, exist_ok=True)
    (run / "plots").mkdir(parents=True, exist_ok=True)
    write_json(run / "status.json", {"status": "running", "stage": "startup", "run_dir": str(run)})
    print(json.dumps({"run_dir": str(run), "stage": "startup"}), flush=True)

    coarse_all, native_all = load_phi(args.coarse_source), load_phi(args.native_fine_source)
    stop = args.start_index + args.n_configs
    coarse_lattice_size = int(args.coarse_lattice_size or coarse_all.shape[1])
    fine_lattice_size = 2 * coarse_lattice_size
    if (coarse_all.shape[1:] != (coarse_lattice_size, coarse_lattice_size)
            or native_all.shape[1:] != (fine_lattice_size, fine_lattice_size)
            or stop > len(coarse_all) or stop > len(native_all)):
        raise RuntimeError("invalid coarse/fine source shape or index range")
    coarse = coarse_all[args.start_index:stop].astype(np.float32)
    native = native_all[args.start_index:stop].astype(np.float32)
    bank = coarse_all[:args.context_bank_count].astype(np.float64)
    context_mean, context_std = float(np.mean(bank * bank)), float(np.std(bank * bank))
    if context_std <= 0.0:
        raise RuntimeError("invalid coarse-context normalization")
    h = (coarse.astype(np.float64) ** 2 - context_mean) / context_std
    kernel, meta = load_kernel_matrix(args.kernel_path)
    checkpoint = torch.load(args.flow_checkpoint, map_location=args.device, weights_only=False)
    device = torch.device(args.device)
    model, report = build_model_from_checkpoint(checkpoint, coarse_lattice_size, device)
    detail, base_logq, _, _ = sample_model_lattice(
        model, coarse, stationary_stats(checkpoint["state"]["stats"], coarse_lattice_size),
        batch_size=args.batch_size, device=device, seed=args.seed,
    )
    rms = np.sqrt(np.mean(detail.astype(np.float64) ** 2, axis=(0, 2, 3)))
    u = detail / rms.reshape(1, 3, 1, 1)
    action = ActionSpec("phi4_nn", 1.0, 0.340301)
    native_obs, raw_obs = obs(native, action), obs(inverse_kernel(assemble_psi(coarse, detail), kernel)[0], action)
    rows: list[dict[str, object]] = []
    best_current: dict[str, np.ndarray] | None = None
    target_scale = {"action_density": 0.002, "phi2": 0.0015, "phi4": 0.003, "kurtosis": 0.006, "NN": 0.002}
    for base_scale in base_scales:
        for tail_strength in tail_strengths:
            t = np.tanh(u / args.tail_width)
            core = u - tail_strength * t**3
            core_derivative = 1.0 - (3.0 * tail_strength / args.tail_width) * t * t * (1.0 - t * t)
            for slope in slopes:
                candidate_key = f"a{base_scale:.8g}_d{tail_strength:.8g}_s{slope:.8g}"
                alpha = base_scale * np.exp(slope * h)
                transformed = (alpha[:, None] * core * rms.reshape(1, 3, 1, 1)).astype(np.float32)
                derivative = alpha[:, None] * core_derivative
                if np.any(derivative <= 0.0):
                    raise RuntimeError("nonpositive map derivative")
                fine, _ = inverse_kernel(assemble_psi(coarse, transformed), kernel)
                current = obs(fine, action)
                reblock = float(np.max(np.abs(apply_kernel(fine, kernel)[:, 0::2, 0::2] - coarse)))
                score = sum(((float(current[key].mean()) - float(native_obs[key].mean())) / target_scale[key]) ** 2 for key in KEYS)
                row = {"candidate_key": candidate_key, "base_scale": base_scale, "tail_strength": tail_strength, "slope": slope,
                       "context_mean": context_mean, "context_std": context_std, "target_match_score": score,
                       "reverse_kl_surrogate_mean": float(np.mean(action_total(fine, action) - np.log(derivative).sum(axis=(1, 2, 3)) + base_logq)),
                       "reblocking_max_error": reblock}
                for key in KEYS:
                    row[f"{key}_mean"] = float(current[key].mean())
                    row[f"{key}_width_ratio_to_native"] = float(current[key].std(ddof=1) / native_obs[key].std(ddof=1))
                    row[f"{key}_mean_shift_to_native"] = float(current[key].mean() - native_obs[key].mean())
                rows.append(row)
                write_csv(run / "observables" / "local_context_slope_scan.csv", rows)
                best = min(rows, key=lambda x: float(x["target_match_score"]))
                if best["candidate_key"] == candidate_key:
                    # Retain small per-configuration observable arrays only.
                    # Full L64 fields are not needed after their observables
                    # and reblocking diagnostic have been computed.
                    best_current = current
                write_json(run / "progress.json", {"stage": "map_scan", "completed": len(rows), "total": len(base_scales) * len(tail_strengths) * len(slopes), "best": best})
                print(json.dumps({"stage": "candidate_complete", **row, "best_candidate": best["candidate_key"]}), flush=True)
                del transformed, derivative, fine, current
                gc.collect()
            del core, core_derivative
            gc.collect()
    best = min(rows, key=lambda x: float(x["target_match_score"]))
    best_key = str(best["candidate_key"])
    if best_current is None:
        raise RuntimeError("map scan completed without a retained best candidate")
    selected = {"native": native_obs, "raw_flow": raw_obs, "local_context_tail": best_current}
    source_indices = np.arange(args.start_index, stop, dtype=np.int64)
    write_csv(run / "observables" / f"main_per_sweep_measurements_direct_L{fine_lattice_size}.csv", main_measurement_rows(native_obs, source_indices, fine_lattice_size))
    write_csv(run / "observables" / "main_per_sweep_measurements_original_wrapped_flow.csv", main_measurement_rows(raw_obs, source_indices, fine_lattice_size))
    write_csv(run / "observables" / "main_per_sweep_measurements_tail_corrected_flow.csv", main_measurement_rows(best_current, source_indices, fine_lattice_size))
    # Canonical run-style aliases let existing readers consume each ensemble
    # without knowing the diagnostic-specific suffixed filenames.
    for label, values in (
        (f"direct_L{fine_lattice_size}", native_obs),
        ("original_wrapped_flow", raw_obs),
        ("tail_corrected_flow", best_current),
    ):
        write_csv(run / label / "observables" / "main_per_sweep_measurements.csv", main_measurement_rows(values, source_indices, fine_lattice_size))
    # The final files contain per-configuration observables only. Release the
    # large lattice tensors before allocating matplotlib figure buffers.
    del coarse_all, native_all, coarse, native, bank, h, detail, u, t
    gc.collect()
    summary_rows = []
    for key in KEYS:
        for label in ("raw_flow", "local_context_tail"):
            summary_rows.append({"observable": key, "ensemble": label, "native_mean": float(native_obs[key].mean()), "mean": float(selected[label][key].mean()), "mean_shift": float(selected[label][key].mean() - native_obs[key].mean()), "native_std": float(native_obs[key].std(ddof=1)), "std": float(selected[label][key].std(ddof=1)), "width_ratio": float(selected[label][key].std(ddof=1) / native_obs[key].std(ddof=1))})
    write_csv(run / "observables" / "central_value_width_comparison.csv", summary_rows)
    fig, axes = plt.subplots(1, len(KEYS), figsize=(17, 3.5))
    for ax, key in zip(axes, KEYS):
        values = np.concatenate([selected[label][key] for label in selected])
        bins = np.linspace(np.quantile(values, .002), np.quantile(values, .998), 45)
        for label, color, display in (
            ("native", "black", "direct L32"),
            ("raw_flow", "tab:orange", "original wrapped flow"),
            ("local_context_tail", "tab:blue", "tail-corrected flow"),
        ):
            ax.hist(selected[label][key], bins=bins, density=True, histtype="step", linewidth=1.5, color=color, label=display)
        ax.set_title(key)
        ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(run / "plots" / "direct_original_corrected_flow_histograms.pdf")
    fig.savefig(run / "plots" / "direct_original_corrected_flow_histograms.png", dpi=180)
    plt.close(fig)
    write_json(run / "run_config.json", vars(args) | {"context_normalization": {"mean": context_mean, "std": context_std, "bank_count": args.context_bank_count}, "best": best, "kernel_metadata": meta, "flow_load_report": report})
    write_json(run / "status.json", {"status": "completed", "run_dir": str(run), "best": best})
    print(json.dumps({"run_dir": str(run), "status": "completed", "best": best}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
