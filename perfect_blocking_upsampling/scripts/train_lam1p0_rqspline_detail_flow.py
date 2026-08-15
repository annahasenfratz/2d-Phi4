#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import platform
import random
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PKG = PROJECT_ROOT / "perfect_blocking_upsampling"
SCRIPT_DIR = PKG / "scripts"
sys.path.insert(0, str(PKG / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

from perfect_blocking_upsampling.conv_spline_pair import build_procedural_conv_spline_flow  # noqa: E402
from perfect_blocking_upsampling.io import ActionSpec  # noqa: E402
from train_lam1p0_autoregressive_detail_flow import (  # noqa: E402
    ARDetailFlow as AffineARDetailFlow,
    DEFAULT_WEIGHTS,
    ETA_SCALE,
    action_total,
    evaluate_generated,
    parse_weights,
    sample_model,
    standardize,
    torch_inverse_kernel,
    torch_kernel_fft,
    torch_observables,
)
from train_lam1p0_flow_detail_localreg import native_targets  # noqa: E402
from train_lam1p0_flow_detail_pilot import (  # noqa: E402
    apply_kernel,
    load_kernel_matrix,
    load_phi,
    per_config_rows,
    split_pairs,
    summarize_comparison,
    write_csv,
    write_json,
)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    except Exception:
        return "unknown"


class RQSplineARDetailFlow(torch.nn.Module):
    def __init__(
        self,
        *,
        layers: int,
        hidden: int,
        kernel_size: int,
        num_bins: int,
        tail_bound: float,
        min_bin_width: float,
        min_bin_height: float,
        min_derivative: float,
    ):
        super().__init__()
        kwargs = {
            "target_channels": 1,
            "lattice_size": 8,
            "n_coupling_layers": int(layers),
            "conv_hidden_channels": int(hidden),
            "conv_kernel_size": int(kernel_size),
            "num_bins": int(num_bins),
            "tail_bound": float(tail_bound),
            "min_bin_width": float(min_bin_width),
            "min_bin_height": float(min_bin_height),
            "min_derivative": float(min_derivative),
        }
        self.flows = torch.nn.ModuleList(
            [
                build_procedural_conv_spline_flow(cond_channels=1, **kwargs),
                build_procedural_conv_spline_flow(cond_channels=2, **kwargs),
                build_procedural_conv_spline_flow(cond_channels=3, **kwargs),
            ]
        )

    @staticmethod
    def cond(coarse: torch.Tensor, d: torch.Tensor, stage: int) -> torch.Tensor:
        if stage == 0:
            return coarse[:, None].flatten(1)
        if stage == 1:
            return torch.cat([coarse[:, None], d[:, 0:1]], dim=1).flatten(1)
        return torch.cat([coarse[:, None], d[:, 0:1], d[:, 1:2]], dim=1).flatten(1)

    def log_prob(self, coarse: torch.Tensor, detail: torch.Tensor) -> torch.Tensor:
        lp = coarse.new_zeros(coarse.shape[0])
        for stage, flow in enumerate(self.flows):
            lp = lp + flow.log_prob(detail[:, stage].flatten(1), self.cond(coarse, detail, stage))
        return lp

    def sample(self, coarse: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        n = coarse.shape[0]
        d = coarse.new_zeros((n, 3, 8, 8))
        logq = coarse.new_zeros(n)
        zmax = coarse.new_zeros(n)
        logdet_total = coarse.new_zeros(n)
        for stage, flow in enumerate(self.flows):
            z = torch.randn((n, 64), device=coarse.device, dtype=coarse.dtype)
            log_base = -0.5 * (z * z + math.log(2.0 * math.pi)).sum(dim=1)
            x, logdet = flow.forward(z, self.cond(coarse, d, stage))
            d[:, stage] = x.reshape(n, 8, 8)
            logq = logq + log_base - logdet
            logdet_total = logdet_total + logdet
            zmax = torch.maximum(zmax, torch.amax(torch.abs(z), dim=1))
        return d, logq, zmax, logdet_total


class ResidualSplineARDetailFlow(torch.nn.Module):
    def __init__(self, affine_base: AffineARDetailFlow, spline: RQSplineARDetailFlow):
        super().__init__()
        self.affine_base = affine_base
        self.spline = spline
        self.flows = self.spline.flows

    @staticmethod
    def cond(coarse: torch.Tensor, d: torch.Tensor, stage: int) -> torch.Tensor:
        return RQSplineARDetailFlow.cond(coarse, d, stage)

    def log_prob(self, coarse: torch.Tensor, detail: torch.Tensor) -> torch.Tensor:
        lp = coarse.new_zeros(coarse.shape[0])
        reconstructed = coarse.new_zeros(detail.shape)
        for stage in range(3):
            cond_spline = self.cond(coarse, detail, stage)
            pre_spline, spline_logdet = self.spline.flows[stage].inverse(detail[:, stage].flatten(1), cond_spline)
            cond_affine = self.affine_base.cond(coarse, reconstructed, stage)
            z, affine_logdet = self.affine_base.flows[stage].inverse(pre_spline, cond_affine)
            log_base = -0.5 * (z * z + math.log(2.0 * math.pi)).sum(dim=1)
            lp = lp + log_base + affine_logdet + spline_logdet
            reconstructed[:, stage] = detail[:, stage]
        return lp

    def sample(self, coarse: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        n = coarse.shape[0]
        d = coarse.new_zeros((n, 3, 8, 8))
        logq = coarse.new_zeros(n)
        zmax = coarse.new_zeros(n)
        logdet_total = coarse.new_zeros(n)
        for stage in range(3):
            z = torch.randn((n, 64), device=coarse.device, dtype=coarse.dtype)
            log_base = -0.5 * (z * z + math.log(2.0 * math.pi)).sum(dim=1)
            cond_affine = self.affine_base.cond(coarse, d, stage)
            x_affine, affine_logdet = self.affine_base.flows[stage].forward(z, cond_affine)
            cond_spline = self.cond(coarse, d, stage)
            x, spline_logdet = self.spline.flows[stage].forward(x_affine, cond_spline)
            d[:, stage] = x.reshape(n, 8, 8)
            total_logdet = affine_logdet + spline_logdet
            logq = logq + log_base - total_logdet
            logdet_total = logdet_total + total_logdet
            zmax = torch.maximum(zmax, torch.amax(torch.abs(z), dim=1))
        return d, logq, zmax, logdet_total


def transfer_affine_hidden(source: AffineARDetailFlow, target: RQSplineARDetailFlow) -> dict[str, Any]:
    copied = 0
    skipped = 0
    for stage in range(3):
        for layer in range(len(source.flows[stage].nets)):
            src_net = source.flows[stage].nets[layer].net
            tgt_net = target.flows[stage].nets[layer].net
            for idx in [0, 2, 4]:
                tgt_net[idx].load_state_dict(src_net[idx].state_dict())
                copied += 1
            skipped += 1
    return {
        "method": "copy_affine_conditioner_hidden_convs",
        "copied_conv_modules": copied,
        "new_spline_output_heads": skipped,
        "output_heads_initialized_to_identity_spline": True,
    }


def rng_state() -> dict[str, Any]:
    state = {"python_random": random.getstate(), "numpy_random": np.random.get_state(), "torch_cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(ckpt: dict[str, Any]) -> bool:
    state = ckpt.get("rng_state")
    if not state:
        return False
    try:
        random.setstate(state["python_random"])
        np.random.set_state(state["numpy_random"])
        torch.set_rng_state(state["torch_cpu"])
        if torch.cuda.is_available() and "torch_cuda" in state:
            torch.cuda.set_rng_state_all(state["torch_cuda"])
        return True
    except Exception:
        return False


def save_full_checkpoint(
    path: Path,
    model: RQSplineARDetailFlow,
    optimizer: torch.optim.Optimizer,
    state: dict[str, Any],
    history: list[dict[str, Any]],
    args: argparse.Namespace,
    *,
    absolute_epoch: int,
    continuation_epoch: int,
    best_validation_nll: float,
    patience_counter: int,
    observable_weights: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_class": "RQSplineARDetailFlow",
            "optimizer_state": optimizer.state_dict(),
            "state": state,
            "history": history,
            "config": vars(args),
            "absolute_epoch": int(absolute_epoch),
            "continuation_epoch": int(continuation_epoch),
            "best_validation_nll": float(best_validation_nll),
            "patience_counter": int(patience_counter),
            "observable_weights": observable_weights,
            "spline_settings": {
                "num_bins": args.num_bins,
                "tail_bound": args.tail_bound,
                "min_bin_width": args.min_bin_width,
                "min_bin_height": args.min_bin_height,
                "min_derivative": args.min_derivative,
                "tails": "linear",
            },
            "rng_state": rng_state(),
        },
        path,
    )


def source_absolute_epoch(ckpt: dict[str, Any]) -> int:
    if ckpt.get("absolute_epoch") is not None:
        return int(ckpt["absolute_epoch"])
    vals = [int(r["absolute_epoch"]) for r in ckpt.get("history", []) if isinstance(r, dict) and r.get("absolute_epoch") not in (None, "")]
    return max(vals) if vals else 0


def tail_summary(delta_s: np.ndarray, log_acc: np.ndarray, accepted: np.ndarray) -> dict[str, float | int]:
    flat_acc = accepted.reshape(-1)
    longest = 0
    current = 0
    for ok in flat_acc:
        if ok:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return {
        "DeltaS_p50": float(np.quantile(delta_s, 0.50)),
        "DeltaS_p90": float(np.quantile(delta_s, 0.90)),
        "DeltaS_p95": float(np.quantile(delta_s, 0.95)),
        "DeltaS_p99": float(np.quantile(delta_s, 0.99)),
        "log_accept_p01": float(np.quantile(log_acc, 0.01)),
        "log_accept_p05": float(np.quantile(log_acc, 0.05)),
        "log_accept_p10": float(np.quantile(log_acc, 0.10)),
        "log_accept_frac_lt_minus10": float(np.mean(log_acc < -10.0)),
        "log_accept_frac_lt_minus20": float(np.mean(log_acc < -20.0)),
        "longest_rejection_streak_flat": int(longest),
    }


def exact_detail_test_tail(model, coarse: np.ndarray, kernel: np.ndarray, stats: dict[str, Any], action: ActionSpec, args: argparse.Namespace) -> dict[str, Any]:
    n = min(args.patch_test_chains, len(coarse))
    coarse = coarse[:n]
    detail, logq, _zmax, _ld = sample_model(model, coarse, stats, args, args.random_seed + 7000)
    phi, _ = inverse_kernel(assemble_psi(coarse, detail), kernel)
    current_s = np.asarray(action_total(phi, action), dtype=np.float64)
    current_logq = logq
    rng = np.random.default_rng(args.random_seed + 7100)
    rows = []
    accepted_total = 0
    attempts_total = 0
    delta_s_vals = []
    log_acc_vals = []
    acc_vals = []
    for sweep in range(1, args.patch_test_sweeps + 1):
        prop_detail, prop_logq, _z, _ld = sample_model(model, coarse, stats, args, args.random_seed + 7200 + sweep)
        prop_phi, _ = inverse_kernel(assemble_psi(coarse, prop_detail), kernel)
        prop_s = np.asarray(action_total(prop_phi, action), dtype=np.float64)
        delta_s = prop_s - current_s
        log_acc = -prop_s + current_s + current_logq - prop_logq
        acc = np.log(rng.random(n)) < np.minimum(log_acc, 0.0)
        if np.any(acc):
            detail[acc] = prop_detail[acc]
            phi[acc] = prop_phi[acc]
            current_s[acc] = prop_s[acc]
            current_logq[acc] = prop_logq[acc]
        accepted_total += int(np.sum(acc))
        attempts_total += n
        delta_s_vals.append(delta_s)
        log_acc_vals.append(log_acc)
        acc_vals.append(acc)
        rows.append({"sweep": sweep, "attempts": n, "accepted": int(np.sum(acc)), "acceptance": float(np.mean(acc))})
    ds = np.concatenate(delta_s_vals)
    la = np.concatenate(log_acc_vals)
    acc_all = np.stack(acc_vals)
    summary = {
        "attempts": attempts_total,
        "accepted": accepted_total,
        "acceptance": float(accepted_total / max(attempts_total, 1)),
        "DeltaS_mean": float(np.mean(ds)),
        "DeltaS_std": float(np.std(ds)),
        "log_accept_mean": float(np.mean(la)),
        "log_accept_std": float(np.std(la)),
    }
    summary.update(tail_summary(ds, la, acc_all))
    return {"phi": phi.astype(np.float32), "rows": rows, "summary": summary}


# Local imports late to keep provenance obvious.
from train_lam1p0_autoregressive_detail_flow import assemble_psi, inverse_kernel  # noqa: E402


def metric_lookup(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(r["observable"]): r for r in rows}


LOCAL_OBS_KEYS = ["action_density", "phi2", "phi4", "local_kurtosis_ratio", "NN", "2nn", "diag"]


def volume_independent_torch_observables(phi: torch.Tensor) -> dict[str, torch.Tensor]:
    """Observable terms that do not use global momentum or magnetization modes."""
    phi2 = (phi * phi).mean(dim=(1, 2))
    phi4 = (phi**4).mean(dim=(1, 2))
    nn = 0.5 * (
        (phi * torch.roll(phi, shifts=-1, dims=1)).mean(dim=(1, 2))
        + (phi * torch.roll(phi, shifts=-1, dims=2)).mean(dim=(1, 2))
    )
    two = 0.5 * (
        (phi * torch.roll(phi, shifts=-2, dims=1)).mean(dim=(1, 2))
        + (phi * torch.roll(phi, shifts=-2, dims=2)).mean(dim=(1, 2))
    )
    diag = (phi * torch.roll(torch.roll(phi, shifts=-1, dims=1), shifts=-1, dims=2)).mean(dim=(1, 2))
    action_density = -phi2 + phi4 - 4.0 * 0.340301 * nn
    return {
        "action_density": action_density,
        "phi2": phi2,
        "phi4": phi4,
        "local_kurtosis_ratio": phi4 / torch.clamp(phi2 * phi2, min=1.0e-12),
        "NN": nn,
        "2nn": two,
        "diag": diag,
    }


def local_numpy_observables(phi: np.ndarray) -> dict[str, np.ndarray]:
    arr = np.asarray(phi, dtype=np.float64)
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
    return {
        "action_density": (1.0 - 2.0 * phi2) + phi4 - 4.0 * 0.340301 * nn,
        "phi2": phi2,
        "phi4": phi4,
        "local_kurtosis_ratio": phi4 / np.maximum(phi2 * phi2, 1.0e-300),
        "NN": nn,
        "2nn": two,
        "diag": diag,
    }


def local_observable_comparison(native: np.ndarray, generated: np.ndarray) -> list[dict[str, Any]]:
    a = local_numpy_observables(native)
    b = local_numpy_observables(generated)
    rows = []
    for key in LOCAL_OBS_KEYS:
        rows.append(
            {
                "observable": key,
                "native_mean": float(np.mean(a[key])),
                "generated_mean": float(np.mean(b[key])),
                "standardized_mean_shift": float((np.mean(b[key]) - np.mean(a[key])) / max(np.std(a[key], ddof=1), 1.0e-300)),
                "native_std": float(np.std(a[key], ddof=1)),
                "generated_std": float(np.std(b[key], ddof=1)),
                "std_ratio": float(np.std(b[key], ddof=1) / max(np.std(a[key], ddof=1), 1.0e-300)),
            }
        )
    return rows


def run_checkpoint_eval(model, phi16, phi8, kernel, state, args, run: Path, label: str, continuation_epoch: int, validation_nll: float, observable_penalty: float) -> dict[str, Any]:
    action = ActionSpec("phi4_nn", 1.0, 0.340301)
    patch = exact_detail_test_tail(model, phi8[: args.patch_test_chains], kernel, state["stats"], action, args)
    if getattr(args, "local_only", False):
        local = metric_lookup(local_observable_comparison(phi16[: len(patch["phi"])], patch["phi"]))
        row: dict[str, Any] = {
            "label": label,
            "continuation_epoch": continuation_epoch,
            "validation_nll": validation_nll,
            "observable_penalty": observable_penalty,
            **patch["summary"],
            "nonfinite_count": int(np.sum(~np.isfinite(patch["phi"]))),
        }
        for obs in LOCAL_OBS_KEYS:
            row[f"patch_{obs}_shift"] = float(local[obs]["standardized_mean_shift"])
            row[f"patch_{obs}_std_ratio"] = float(local[obs]["std_ratio"])
        return row
    generated = evaluate_generated(model, phi16, phi8, kernel, state["stats"], args, run, f"eval_{label}")
    patch_rows, patch_g = per_config_rows(patch["phi"], action, f"patch_eval_{label}")
    write_csv(run / "observables" / f"patch_eval_{label}_observables_per_config.csv", patch_rows)
    write_csv(run / "observables" / f"patch_eval_{label}_Gk_per_config.csv", patch_g)
    patch_comp = summarize_comparison(phi16[: len(patch["phi"])], patch["phi"], action)
    write_csv(run / "observables" / f"patch_eval_{label}_observable_comparison.csv", patch_comp)
    raw = metric_lookup(generated["comparison"])
    patched = metric_lookup(patch_comp)
    row: dict[str, Any] = {"label": label, "continuation_epoch": continuation_epoch, "validation_nll": validation_nll, "observable_penalty": observable_penalty, **patch["summary"], "nonfinite_count": generated["nonfinite_count"], "max_abs_z": generated["max_abs_z"], "reblocking_max_abs_error": generated["reblocking_max_abs_error"], "logdet_mean": generated["logdet_mean"], "logdet_std": generated["logdet_std"], "logdet_min": generated["logdet_min"], "logdet_max": generated["logdet_max"]}
    for obs in ["action_density", "phi2", "phi4", "local_kurtosis_ratio", "NN", "2nn", "diag", "m2", "m4", "G_pmin_avg"]:
        row[f"raw_{obs}_shift"] = float(raw[obs]["standardized_mean_shift"])
        row[f"raw_{obs}_std_ratio"] = float(raw[obs]["std_ratio"])
        row[f"patch_{obs}_shift"] = float(patched[obs]["standardized_mean_shift"])
        row[f"patch_{obs}_std_ratio"] = float(patched[obs]["std_ratio"])
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--source-run", type=Path)
    ap.add_argument("--resume-checkpoint", type=Path)
    ap.add_argument(
        "--resume-rqspline-checkpoint",
        type=Path,
        help="Warm-start the complete residual affine+spline flow from a prior RQSpline checkpoint.",
    )
    ap.add_argument("--from-scratch", action="store_true", help="Initialize both affine and spline flows without a source checkpoint.")
    ap.add_argument("--fine-config-source", type=Path, default=Path("data/configs_phi4_2d/lam1p0_kappac0p340301_L16/configs.npz"))
    ap.add_argument("--coarse-config-source", type=Path, default=Path("data/configs_phi4_2d/lam1p0_kappac0p340301_L8/configs.npz"))
    ap.add_argument("--kernel-path", type=Path, default=Path("perfect_blocking/perfect_blocking_lam1p0/kernels/final/chosen_kernel.json"))
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1.0e-4)
    ap.add_argument("--weight-decay", type=float, default=1.0e-5)
    ap.add_argument("--grad-clip", type=float, default=10.0)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--hidden-channels", type=int, default=48)
    ap.add_argument("--conv-kernel-size", type=int, default=3)
    ap.add_argument("--num-bins", type=int, default=8)
    ap.add_argument("--tail-bound", type=float, default=6.0)
    ap.add_argument("--min-bin-width", type=float, default=1.0e-3)
    ap.add_argument("--min-bin-height", type=float, default=1.0e-3)
    ap.add_argument("--min-derivative", type=float, default=1.0e-3)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--patch-test-chains", type=int, default=64)
    ap.add_argument("--patch-test-sweeps", type=int, default=100)
    ap.add_argument("--generated-count", type=int, default=512)
    ap.add_argument("--checkpoint-every-epochs", type=int, default=5)
    ap.add_argument("--train-count", type=int, help="Number of configurations assigned to training.")
    ap.add_argument("--validation-count", type=int, help="Number of configurations assigned to validation.")
    ap.add_argument("--split-seed", type=int, default=2026072101)
    ap.add_argument("--log-scale-bound", type=float, default=0.75)
    ap.add_argument("--obs-weights", default="")
    ap.add_argument("--local-only", action="store_true", help="Exclude global momentum/magnetization observables and diagnostics.")
    ap.add_argument("--random-seed", type=int, default=2026071703)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    run = args.run_dir
    for sub in ["logs", "checkpoints", "observables", "plots", "summaries", "debug"]:
        (run / sub).mkdir(parents=True, exist_ok=True)
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    device = torch.device(args.device)
    weights = parse_weights(args.obs_weights) if args.obs_weights else dict(DEFAULT_WEIGHTS)
    if args.local_only:
        weights = {key: float(weights[key]) for key in LOCAL_OBS_KEYS if key in weights and float(weights[key]) != 0.0}

    if args.from_scratch:
        if args.source_run is not None or args.resume_checkpoint is not None or args.resume_rqspline_checkpoint is not None:
            raise SystemExit("--from-scratch cannot be combined with a resume/source option")
        source_abs = 0
        rng_restored = False
        affine = AffineARDetailFlow(
            layers=args.layers,
            hidden=args.hidden_channels,
            kernel_size=args.conv_kernel_size,
            log_scale_bound=args.log_scale_bound,
        ).to(device)
        transfer = {"initialization": "from_scratch", "residual_affine_base": True}
    elif args.resume_rqspline_checkpoint is None:
        if args.resume_checkpoint is None:
            raise SystemExit("a resume checkpoint is required unless --from-scratch is set")
        source_ckpt = torch.load(args.resume_checkpoint, map_location=device, weights_only=False)
        source_abs = source_absolute_epoch(source_ckpt)
        rng_restored = restore_rng_state(source_ckpt)
        src_cfg = source_ckpt["config"]
        affine = AffineARDetailFlow(layers=int(src_cfg["layers"]), hidden=int(src_cfg["hidden_channels"]), kernel_size=int(src_cfg["conv_kernel_size"]), log_scale_bound=float(src_cfg["log_scale_bound"])).to(device)
        affine.load_state_dict(source_ckpt["model_state"])
    if args.resume_rqspline_checkpoint is not None:
        source_ckpt = torch.load(args.resume_rqspline_checkpoint, map_location=device, weights_only=False)
        source_abs = source_absolute_epoch(source_ckpt)
        rng_restored = restore_rng_state(source_ckpt)
        source_cfg = source_ckpt["config"]
        affine = AffineARDetailFlow(
            layers=int(source_cfg["layers"]),
            hidden=int(source_cfg["hidden_channels"]),
            kernel_size=int(source_cfg["conv_kernel_size"]),
            log_scale_bound=float(source_cfg["log_scale_bound"]),
        ).to(device)
        spline = RQSplineARDetailFlow(
            layers=args.layers, hidden=args.hidden_channels, kernel_size=args.conv_kernel_size,
            num_bins=args.num_bins, tail_bound=args.tail_bound,
            min_bin_width=args.min_bin_width, min_bin_height=args.min_bin_height,
            min_derivative=args.min_derivative,
        ).to(device)
        model = ResidualSplineARDetailFlow(affine, spline).to(device)
        model.load_state_dict(source_ckpt["model_state"])
        transfer = {
            "initialization": "complete_rqspline_warm_start",
            "rqspline_loaded_from": str(args.resume_rqspline_checkpoint),
        }
    else:
        spline = RQSplineARDetailFlow(layers=args.layers, hidden=args.hidden_channels, kernel_size=args.conv_kernel_size, num_bins=args.num_bins, tail_bound=args.tail_bound, min_bin_width=args.min_bin_width, min_bin_height=args.min_bin_height, min_derivative=args.min_derivative).to(device)
        model = ResidualSplineARDetailFlow(affine, spline).to(device)
    if not args.from_scratch and args.resume_rqspline_checkpoint is None:
        transfer = transfer_affine_hidden(affine, spline)
        transfer["residual_affine_base"] = True
        transfer["affine_base_loaded_from"] = str(args.resume_checkpoint)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    kernel, kernel_raw = load_kernel_matrix(args.kernel_path)
    phi16 = load_phi(args.fine_config_source)
    phi8 = load_phi(args.coarse_config_source)
    pairs = split_pairs(phi16, kernel)
    if args.from_scratch:
        if args.train_count is None or args.validation_count is None:
            raise SystemExit("--from-scratch requires --train-count and --validation-count")
        if args.train_count <= 0 or args.validation_count <= 0 or args.train_count + args.validation_count > len(phi16):
            raise SystemExit("invalid train/validation counts for the available fine ensemble")
        split_rng = np.random.default_rng(args.split_seed)
        split = split_rng.permutation(len(phi16))
        train_idx = np.sort(split[: args.train_count])
        val_idx = np.sort(split[args.train_count : args.train_count + args.validation_count])
    else:
        train_idx = np.asarray(source_ckpt["state"]["train_idx"], dtype=np.int64)
        val_idx = np.asarray(source_ckpt["state"]["val_idx"], dtype=np.int64)
    c_flat = pairs["coarse"].reshape(len(phi16), -1)
    d_flat = pairs["detail"].reshape(len(phi16), -1)
    c_std, c_stats = standardize(c_flat[train_idx], c_flat)
    d_std, d_stats = standardize(d_flat[train_idx], d_flat)
    state = {"stats": {"coarse": c_stats, "detail": d_stats}, "train_idx": train_idx, "val_idx": val_idx, "kernel_path": str(args.kernel_path)}
    train_ds = TensorDataset(torch.from_numpy(c_std[train_idx].reshape(-1, 8, 8)), torch.from_numpy(d_std[train_idx].reshape(-1, 3, 8, 8)), torch.from_numpy(pairs["coarse"][train_idx]))
    val_ds = TensorDataset(torch.from_numpy(c_std[val_idx].reshape(-1, 8, 8)), torch.from_numpy(d_std[val_idx].reshape(-1, 3, 8, 8)))
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    kt = torch_kernel_fft(kernel, 16, device)
    targets = native_targets(phi16, val_idx)
    d_mean = torch.tensor(d_stats["mean"].reshape(1, 3, 8, 8), dtype=torch.float32, device=device)
    d_std_t = torch.tensor(d_stats["std"].reshape(1, 3, 8, 8), dtype=torch.float32, device=device)

    def validation_nll() -> float:
        model.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for cb, db in val_loader:
                lp = model.log_prob(cb.to(device), db.to(device))
                total += float((-lp).sum().detach().cpu())
                count += int(len(cb))
        return total / max(count, 1)

    def observable_penalty_once() -> tuple[float, dict[str, float]]:
        model.eval()
        cb = torch.from_numpy(c_std[val_idx[: min(args.batch_size, len(val_idx))]].reshape(-1, 8, 8)).to(device)
        coarse_phys = torch.from_numpy(pairs["coarse"][val_idx[: len(cb)]]).to(device)
        with torch.no_grad():
            d_samp, _logq, _z, _ld = model.sample(cb)
            detail_phys = d_samp * d_std_t + d_mean
            psi = torch.empty((detail_phys.shape[0], 16, 16), dtype=detail_phys.dtype, device=device)
            psi[:, 0::2, 0::2] = coarse_phys
            psi[:, 0::2, 1::2] = detail_phys[:, 0]
            psi[:, 1::2, 0::2] = detail_phys[:, 1]
            psi[:, 1::2, 1::2] = detail_phys[:, 2]
            obs = volume_independent_torch_observables(torch_inverse_kernel(psi, kt)) if args.local_only else torch_observables(torch_inverse_kernel(psi, kt))
            penalty = 0.0
            zvals = {}
            for key, weight in weights.items():
                z = (obs[key] - float(targets[key]["mean"])) / float(targets[key]["std"])
                penalty += float(weight) * float((z * z).mean().detach().cpu())
                zvals[f"z_{key}"] = float(z.mean().detach().cpu())
        return penalty, zvals

    # Initialization tests.
    init_val = validation_nll()
    init_penalty, init_z = observable_penalty_once()
    init_eval = run_checkpoint_eval(model, phi16, phi8, kernel, state, args, run, "init", 0, init_val, init_penalty)
    write_json(run / "debug" / "weight_transfer.json", transfer)

    history: list[dict[str, Any]] = []
    patch_rows = [init_eval]
    best_val = init_val
    best_epoch = 0
    best_abs = source_abs
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_opt = copy.deepcopy(opt.state_dict())
    bad = 0
    print(json.dumps({"continuation_epoch": 0, "validation_nll": init_val, "observable_penalty": init_penalty, **init_z, "patch_acceptance": init_eval["acceptance"], "DeltaS_std": init_eval["DeltaS_std"]}), flush=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = {"loss": 0.0, "nll": 0.0, "obs": 0.0, "count": 0}
        last_z = {}
        for cb, db, coarse_phys in loader:
            cb = cb.to(device)
            db = db.to(device)
            coarse_phys = coarse_phys.to(device)
            opt.zero_grad(set_to_none=True)
            nll = -model.log_prob(cb, db).mean()
            d_samp, _logq, _z, _ld = model.sample(cb)
            detail_phys = d_samp * d_std_t + d_mean
            psi = torch.empty((detail_phys.shape[0], 16, 16), dtype=detail_phys.dtype, device=device)
            psi[:, 0::2, 0::2] = coarse_phys
            psi[:, 0::2, 1::2] = detail_phys[:, 0]
            psi[:, 1::2, 0::2] = detail_phys[:, 1]
            psi[:, 1::2, 1::2] = detail_phys[:, 2]
            obs = volume_independent_torch_observables(torch_inverse_kernel(psi, kt)) if args.local_only else torch_observables(torch_inverse_kernel(psi, kt))
            obs_loss = cb.new_tensor(0.0)
            for key, weight in weights.items():
                z = (obs[key] - float(targets[key]["mean"])) / float(targets[key]["std"])
                obs_loss = obs_loss + float(weight) * (z * z).mean()
                last_z[f"z_{key}"] = float(z.mean().detach().cpu())
            loss = nll + obs_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            bs = int(len(cb))
            totals["loss"] += float(loss.detach().cpu()) * bs
            totals["nll"] += float(nll.detach().cpu()) * bs
            totals["obs"] += float(obs_loss.detach().cpu()) * bs
            totals["count"] += bs
        val = validation_nll()
        row = {"continuation_epoch": epoch, "absolute_epoch": source_abs + epoch, "train_loss": totals["loss"] / totals["count"], "train_nll": totals["nll"] / totals["count"], "train_observable_penalty": totals["obs"] / totals["count"], "validation_nll": val, **last_z}
        history.append(row)
        print(json.dumps(row), flush=True)
        if val < best_val - 1.0e-8:
            best_val = val
            best_epoch = epoch
            best_abs = source_abs + epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_opt = copy.deepcopy(opt.state_dict())
            bad = 0
        else:
            bad += 1
        if epoch % args.checkpoint_every_epochs == 0:
            save_full_checkpoint(run / "checkpoints" / f"checkpoint_epoch_{epoch:04d}.pt", model, opt, state, history, args, absolute_epoch=source_abs + epoch, continuation_epoch=epoch, best_validation_nll=best_val, patience_counter=bad, observable_weights=weights)
        if epoch % args.eval_every == 0:
            ev = run_checkpoint_eval(model, phi16, phi8, kernel, state, args, run, f"cont{epoch:03d}", epoch, val, row["train_observable_penalty"])
            patch_rows.append(ev)
            write_csv(run / "observables" / "intermediate_patch_evaluations.csv", patch_rows)
        if bad >= args.patience:
            break

    model.load_state_dict(best_state)
    opt.load_state_dict(best_opt)
    save_full_checkpoint(run / "checkpoints" / "checkpoint_best.pt", model, opt, state, history, args, absolute_epoch=best_abs, continuation_epoch=best_epoch, best_validation_nll=best_val, patience_counter=bad, observable_weights=weights)
    save_full_checkpoint(run / "checkpoints" / "checkpoint_latest.pt", model, opt, state, history, args, absolute_epoch=source_abs + len(history), continuation_epoch=len(history), best_validation_nll=best_val, patience_counter=bad, observable_weights=weights)
    write_csv(run / "observables" / "training_history.csv", history)

    action = ActionSpec("phi4_nn", 1.0, 0.340301)
    final_patch = exact_detail_test_tail(model, phi8[: args.patch_test_chains], kernel, state["stats"], action, args)
    write_csv(run / "observables" / "acceptance_history.csv", final_patch["rows"])
    if args.local_only:
        final_patch_comp = local_observable_comparison(phi16[: len(final_patch["phi"])], final_patch["phi"])
        write_csv(run / "observables" / "patch_test_local_observable_comparison.csv", final_patch_comp)
        write_csv(run / "observables" / "patch_tail_diagnostics.csv", [{k: v for k, v in final_patch["summary"].items() if isinstance(v, (int, float))}])
        patch_rows.append(run_checkpoint_eval(model, phi16, phi8, kernel, state, args, run, "best", best_epoch, best_val, float("nan")))
        write_csv(run / "observables" / "intermediate_patch_evaluations.csv", patch_rows)
        raw_lookup = {}
        patch_lookup = metric_lookup(final_patch_comp)
        final_generated = {"nonfinite_count": int(np.sum(~np.isfinite(final_patch["phi"]))) }
    else:
        final_generated = evaluate_generated(model, phi16, phi8, kernel, state["stats"], args, run, "best_rqspline")
        final_rows, final_g = per_config_rows(final_patch["phi"], action, "patch_test_best_rqspline")
        write_csv(run / "observables" / "patch_test_observables_per_config.csv", final_rows)
        write_csv(run / "observables" / "patch_test_Gk_per_config.csv", final_g)
        final_patch_comp = summarize_comparison(phi16[: len(final_patch["phi"])], final_patch["phi"], action)
        write_csv(run / "observables" / "patch_test_observable_comparison.csv", final_patch_comp)
        patch_rows.append(run_checkpoint_eval(model, phi16, phi8, kernel, state, args, run, "best", best_epoch, best_val, float("nan")))
        write_csv(run / "observables" / "intermediate_patch_evaluations.csv", patch_rows)
        write_csv(run / "observables" / "patch_tail_diagnostics.csv", [{k: v for k, v in final_patch["summary"].items() if isinstance(v, (int, float))}])

    if not args.local_only:
        raw_lookup = metric_lookup(final_generated["comparison"])
        patch_lookup = metric_lookup(final_patch_comp)
    write_json(run / "debug" / "stability_diagnostics.json", {"generated": {k: v for k, v in final_generated.items() if k not in {"phi", "comparison"}}, "patch_test": final_patch["summary"], "kernel": {"path": str(args.kernel_path), "sum": float(kernel.sum()), "eta_scale": ETA_SCALE, "kernel_coefficients_include_eta_scale": True}, "weight_transfer": transfer, "best_continuation_epoch": best_epoch, "best_absolute_epoch": best_abs, "best_validation_nll": best_val})
    import yaml

    initialization = "from_scratch" if args.from_scratch else (
        "complete_rqspline_warm_start" if args.resume_rqspline_checkpoint is not None else "checkpoint_warm_start"
    )
    cfg = {"lambda": 1.0, "kappa_f": 0.340301, "kappa_c": 0.340301, "eta": 0.25, "eta_scale_numeric": ETA_SCALE, "block_factor": 2, "L_c": 8, "L_f": 16, "kernel_path": str(args.kernel_path), "kernel_coefficients_include_eta_scale": True, "kernel_sum": float(kernel.sum()), "mode": "train_residual_rqspline_autoregressive_detail_flow", "initialization": initialization, "source_run": None if args.from_scratch else str(args.source_run), "resume_checkpoint": None if args.from_scratch else str(args.resume_checkpoint), "resume_rqspline_checkpoint": None if args.from_scratch else str(args.resume_rqspline_checkpoint), "split": {"train_count": int(len(train_idx)), "validation_count": int(len(val_idx)), "seed": args.split_seed if args.from_scratch else None}, "architecture": {"factorization": "q(d01|coarse) q(d10|coarse,d01) q(d11|coarse,d01,d10)", "stage_flow": "random affine base plus procedural circular-conv checkerboard rational-quadratic spline residual coupling" if args.from_scratch else "trained affine base plus procedural circular-conv checkerboard rational-quadratic spline residual coupling", "layers_per_stage": args.layers, "hidden_channels": args.hidden_channels, "conv_kernel_size": args.conv_kernel_size, "parameter_count": sum(p.numel() for p in model.parameters())}, "spline": {"num_bins": args.num_bins, "tail_bound": args.tail_bound, "tails": "linear", "min_bin_width": args.min_bin_width, "min_bin_height": args.min_bin_height, "min_derivative": args.min_derivative}, "objective": {"NLL": 1.0, "observable_weights": weights}, "weight_transfer": transfer, "random_seed": args.random_seed, "rng_restored": rng_restored}
    (run / "run_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    write_json(run / "submit_manifest.txt", {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "command": " ".join(sys.argv), "git_commit": git_commit(), "host": socket.gethostname(), "platform": platform.platform(), "kernel_path": str(args.kernel_path), "kernel_sum": float(kernel.sum()), "eta_scale_numeric": ETA_SCALE})

    lines = ["# Lambda 1.0 L8->L16 Residual RQ-Spline AR Detail Flow", "", f"- best continuation epoch: `{best_epoch}`", f"- best absolute epoch: `{best_abs}`", f"- best validation NLL: `{best_val:.6g}`", f"- parameter count: `{sum(p.numel() for p in model.parameters())}`", f"- patch acceptance: `{final_patch['summary']['acceptance']:.6g}`", f"- DeltaS mean/std: `{final_patch['summary']['DeltaS_mean']:.6g}` / `{final_patch['summary']['DeltaS_std']:.6g}`", f"- DeltaS p95/p99: `{final_patch['summary']['DeltaS_p95']:.6g}` / `{final_patch['summary']['DeltaS_p99']:.6g}`", f"- logacc mean/std: `{final_patch['summary']['log_accept_mean']:.6g}` / `{final_patch['summary']['log_accept_std']:.6g}`", "", "| observable | raw shift | raw std ratio | patch shift | patch std ratio |", "|---|---:|---:|---:|---:|"]
    if args.local_only:
        lines = ["# Lambda 1.0 L8->L16 Volume-Independent Residual RQ-Spline AR Detail Flow", "", "Global momentum and magnetization observables were excluded from training and diagnostics.", "", "| observable | patch shift | patch std ratio |", "|---|---:|---:|"]
        for obs in LOCAL_OBS_KEYS:
            pp = patch_lookup[obs]
            lines.append(f"| {obs} | {float(pp['standardized_mean_shift']):.6g} | {float(pp['std_ratio']):.6g} |")
    else:
        for obs in ["action_density", "phi2", "phi4", "local_kurtosis_ratio", "NN", "2nn", "diag", "m2", "m4", "G_pmin_avg"]:
            rr = raw_lookup[obs]
            pp = patch_lookup[obs]
            lines.append(f"| {obs} | {float(rr['standardized_mean_shift']):.6g} | {float(rr['std_ratio']):.6g} | {float(pp['standardized_mean_shift']):.6g} | {float(pp['std_ratio']):.6g} |")
    (run / "summaries" / "run_summary.md").write_text("\n".join(lines) + "\n")
    write_json(run / "status.json", {"status": "completed", "best_continuation_epoch": best_epoch, "best_absolute_epoch": best_abs, "best_checkpoint": str(run / "checkpoints" / "checkpoint_best.pt"), "latest_checkpoint": str(run / "checkpoints" / "checkpoint_latest.pt"), "validation_plateau": bad >= args.patience, "epochs_completed": len(history), "summary": str(run / "summaries" / "run_summary.md")})
    print(json.dumps({"status": "completed", "run_dir": str(run), "best_epoch": best_epoch, "best_validation_nll": best_val, "acceptance": final_patch["summary"]["acceptance"], "DeltaS_std": final_patch["summary"]["DeltaS_std"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
