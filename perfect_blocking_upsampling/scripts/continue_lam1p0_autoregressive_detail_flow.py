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

from perfect_blocking_upsampling.io import ActionSpec  # noqa: E402
from train_lam1p0_autoregressive_detail_flow import (  # noqa: E402
    ARDetailFlow,
    DEFAULT_WEIGHTS,
    ETA_SCALE,
    evaluate_generated,
    exact_detail_test,
    parse_weights,
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


def metric_lookup(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(r["observable"]): r for r in rows}


def finite_float(x: Any) -> float:
    return float(x) if x is not None else float("nan")


def source_absolute_epoch(ckpt: dict[str, Any]) -> int:
    for key in ("absolute_epoch", "source_absolute_epoch"):
        if key in ckpt and ckpt[key] is not None:
            return int(ckpt[key])
    candidates: list[int] = []
    for row in ckpt.get("history", []):
        if isinstance(row, dict) and row.get("absolute_epoch") not in (None, ""):
            candidates.append(int(row["absolute_epoch"]))
    cfg = ckpt.get("config", {})
    if cfg.get("epochs") is not None:
        candidates.append(int(cfg["epochs"]))
    return max(candidates) if candidates else 0


def rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python_random": random.getstate(),
        "numpy_random": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
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
    except Exception:
        return False
    return True


def full_checkpoint(
    *,
    path: Path,
    model: ARDetailFlow,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    state: dict[str, Any],
    history: list[dict[str, Any]],
    args: argparse.Namespace,
    absolute_epoch: int,
    continuation_epoch: int,
    best_validation_nll: float,
    patience_counter: int,
    observable_weights: dict[str, float],
    optimizer_restored_from_source: bool,
    scheduler_restored_from_source: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_class": "ARDetailFlow",
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "state": state,
            "history": history,
            "config": vars(args),
            "absolute_epoch": absolute_epoch,
            "continuation_epoch": continuation_epoch,
            "best_validation_nll": best_validation_nll,
            "patience_counter": patience_counter,
            "observable_weights": observable_weights,
            "rng_state": rng_state(),
            "optimizer_restored_from_source": optimizer_restored_from_source,
            "scheduler_restored_from_source": scheduler_restored_from_source,
        },
        path,
    )


def transfer_prefix_layers(source: ARDetailFlow, target: ARDetailFlow, source_layers: int, target_layers: int) -> dict[str, Any]:
    copied = 0
    for stage in range(len(source.flows)):
        n_copy = min(source_layers, target_layers, len(source.flows[stage].nets), len(target.flows[stage].nets))
        for layer in range(n_copy):
            target.flows[stage].nets[layer].load_state_dict(source.flows[stage].nets[layer].state_dict())
            copied += 1
    return {
        "method": "prefix_coupling_layer_transfer",
        "source_layers": int(source_layers),
        "target_layers": int(target_layers),
        "stages": len(source.flows),
        "copied_stage_layers": int(min(source_layers, target_layers)),
        "copied_net_modules": int(copied),
        "new_layers_per_stage": int(max(0, target_layers - source_layers)),
        "new_layers_initialized_near_identity": True,
    }


def initialization_diagnostics(model: ARDetailFlow, source_model: ARDetailFlow | None, state: dict[str, Any], args: argparse.Namespace, run: Path, kernel: np.ndarray) -> dict[str, Any]:
    device = torch.device(args.device)
    stats = state["stats"]
    coarse_mean = torch.tensor(stats["coarse"]["mean"].reshape(1, 8, 8), dtype=torch.float32, device=device)
    coarse_std = torch.tensor(stats["coarse"]["std"].reshape(1, 8, 8), dtype=torch.float32, device=device)
    # Standardized nonzero coarse input so conditioner paths are exercised.
    coarse = torch.full((4, 8, 8), 0.1, dtype=torch.float32, device=device)
    model.eval()
    result: dict[str, Any] = {}
    with torch.no_grad():
        torch.manual_seed(args.random_seed + 171)
        detail, logq, zmax, logdet = model.sample(coarse)
        inverse_max_errors = []
        logdet_roundtrip_errors = []
        for stage, flow in enumerate(model.flows):
            cond = model.cond(coarse, detail, stage)
            x = detail[:, stage].flatten(1)
            z, inv_logdet = flow.inverse(x, cond)
            x2, fwd_logdet = flow.forward(z, cond)
            inverse_max_errors.append(float(torch.max(torch.abs(x2 - x)).detach().cpu()))
            logdet_roundtrip_errors.append(float(torch.max(torch.abs(fwd_logdet + inv_logdet)).detach().cpu()))
        d_mean = torch.tensor(stats["detail"]["mean"].reshape(1, 3, 8, 8), dtype=torch.float32, device=device)
        d_std = torch.tensor(stats["detail"]["std"].reshape(1, 3, 8, 8), dtype=torch.float32, device=device)
        detail_phys = detail * d_std + d_mean
        psi = torch.empty((detail.shape[0], 16, 16), dtype=detail.dtype, device=device)
        coarse_phys = coarse * coarse_std + coarse_mean
        psi[:, 0::2, 0::2] = coarse_phys
        psi[:, 0::2, 1::2] = detail_phys[:, 0]
        psi[:, 1::2, 0::2] = detail_phys[:, 1]
        psi[:, 1::2, 1::2] = detail_phys[:, 2]
        kt = torch_kernel_fft(kernel, 16, device)
        phi = torch_inverse_kernel(psi, kt)
        reb = torch.fft.ifft2(torch.fft.fft2(phi) * kt).real[:, 0::2, 0::2] - coarse_phys
        result.update(
            {
                "inverse_consistency_max_error": float(max(inverse_max_errors)),
                "logdet_roundtrip_max_error": float(max(logdet_roundtrip_errors)),
                "reblocking_max_abs_error": float(torch.max(torch.abs(reb)).detach().cpu()),
                "nonfinite_count": int((~torch.isfinite(phi)).sum().detach().cpu() + (~torch.isfinite(detail)).sum().detach().cpu()),
                "max_abs_z": float(torch.max(zmax).detach().cpu()),
                "logdet_mean": float(logdet.mean().detach().cpu()),
                "logdet_std": float(logdet.std(unbiased=False).detach().cpu()),
                "logdet_min": float(logdet.min().detach().cpu()),
                "logdet_max": float(logdet.max().detach().cpu()),
            }
        )
        if source_model is not None:
            source_model.eval()
            torch.manual_seed(args.random_seed + 171)
            source_detail, source_logq, _source_zmax, source_logdet = source_model.sample(coarse)
            result.update(
                {
                    "source_distribution_reproduction_detail_max_abs": float(torch.max(torch.abs(detail - source_detail)).detach().cpu()),
                    "source_distribution_reproduction_logq_max_abs": float(torch.max(torch.abs(logq - source_logq)).detach().cpu()),
                    "source_distribution_reproduction_logdet_max_abs": float(torch.max(torch.abs(logdet - source_logdet)).detach().cpu()),
                }
            )
    write_json(run / "debug" / "initialization_diagnostics.json", result)
    return result


def run_checkpoint_eval(
    *,
    model: ARDetailFlow,
    phi16: np.ndarray,
    phi8: np.ndarray,
    kernel: np.ndarray,
    state: dict[str, Any],
    args: argparse.Namespace,
    run: Path,
    label: str,
    continuation_epoch: int,
    validation_nll: float,
    observable_penalty: float,
) -> dict[str, Any]:
    action = ActionSpec("phi4_nn", 1.0, 0.340301)
    generated = evaluate_generated(model, phi16, phi8, kernel, state["stats"], args, run, f"eval_{label}")
    patch = exact_detail_test(model, phi8[: args.patch_test_chains], kernel, state["stats"], action, args)
    patch_rows, patch_g = per_config_rows(patch["phi"], action, f"patch_eval_{label}")
    write_csv(run / "observables" / f"patch_eval_{label}_observables_per_config.csv", patch_rows)
    write_csv(run / "observables" / f"patch_eval_{label}_Gk_per_config.csv", patch_g)
    patch_comp = summarize_comparison(phi16[: len(patch["phi"])], patch["phi"], action)
    write_csv(run / "observables" / f"patch_eval_{label}_observable_comparison.csv", patch_comp)
    raw = metric_lookup(generated["comparison"])
    patched = metric_lookup(patch_comp)
    row: dict[str, Any] = {
        "label": label,
        "continuation_epoch": continuation_epoch,
        "validation_nll": validation_nll,
        "observable_penalty": observable_penalty,
        "acceptance": patch["summary"]["acceptance"],
        "DeltaS_mean": patch["summary"]["DeltaS_mean"],
        "DeltaS_std": patch["summary"]["DeltaS_std"],
        "log_accept_mean": patch["summary"]["log_accept_mean"],
        "log_accept_std": patch["summary"]["log_accept_std"],
        "nonfinite_count": generated["nonfinite_count"],
        "max_abs_z": generated["max_abs_z"],
        "reblocking_max_abs_error": generated["reblocking_max_abs_error"],
        "logdet_mean": generated["logdet_mean"],
        "logdet_std": generated["logdet_std"],
        "logdet_min": generated["logdet_min"],
        "logdet_max": generated["logdet_max"],
    }
    for obs in ["action_density", "phi2", "phi4", "local_kurtosis_ratio", "NN", "2nn", "diag", "m2", "m4", "G_pmin_avg"]:
        row[f"raw_{obs}_shift"] = finite_float(raw[obs]["standardized_mean_shift"])
        row[f"raw_{obs}_std_ratio"] = finite_float(raw[obs]["std_ratio"])
        row[f"patch_{obs}_shift"] = finite_float(patched[obs]["standardized_mean_shift"])
        row[f"patch_{obs}_std_ratio"] = finite_float(patched[obs]["std_ratio"])
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--source-run", type=Path, required=True)
    ap.add_argument("--resume-checkpoint", type=Path, required=True)
    ap.add_argument("--fine-config-source", type=Path, default=Path("data/configs_phi4_2d/lam1p0_kappac0p340301_L16/configs.npz"))
    ap.add_argument("--coarse-config-source", type=Path, default=Path("data/configs_phi4_2d/lam1p0_kappac0p340301_L8/configs.npz"))
    ap.add_argument("--kernel-path", type=Path, default=Path("perfect_blocking/perfect_blocking_lam1p0/kernels/selected_for_upscaling/best_7x7_phi2_nn_guarded_eta_included.json"))
    ap.add_argument("--additional-epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2.0e-4)
    ap.add_argument("--weight-decay", type=float, default=1.0e-5)
    ap.add_argument("--grad-clip", type=float, default=10.0)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--patch-test-chains", type=int, default=64)
    ap.add_argument("--patch-test-sweeps", type=int, default=100)
    ap.add_argument("--generated-count", type=int, default=512)
    ap.add_argument("--checkpoint-every-epochs", type=int, default=5)
    ap.add_argument("--obs-weights", default="")
    ap.add_argument("--random-seed", type=int, default=2026071701)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--target-layers", type=int, default=0, help="If set above source layers, create a deeper AR flow and copy source layers as a prefix.")
    args = ap.parse_args()

    run = args.run_dir
    for sub in ["logs", "checkpoints", "observables", "plots", "summaries", "debug"]:
        (run / sub).mkdir(parents=True, exist_ok=True)
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    device = torch.device(args.device)
    weights = parse_weights(args.obs_weights)
    if not args.obs_weights:
        weights = dict(DEFAULT_WEIGHTS)

    ckpt = torch.load(args.resume_checkpoint, map_location=device, weights_only=False)
    ckpt_cfg = ckpt["config"]
    source_layers = int(ckpt_cfg.get("layers", 8))
    args.layers = int(args.target_layers) if int(args.target_layers) > 0 else source_layers
    args.hidden_channels = int(ckpt_cfg.get("hidden_channels", 48))
    args.conv_kernel_size = int(ckpt_cfg.get("conv_kernel_size", 3))
    args.log_scale_bound = float(ckpt_cfg.get("log_scale_bound", 0.75))
    source_abs_epoch = source_absolute_epoch(ckpt)
    rng_restored = restore_rng_state(ckpt)
    if args.layers == source_layers and ("optimizer_state" in ckpt or "scheduler_state" in ckpt):
        optimizer_restored = "optimizer_state" in ckpt
        scheduler_restored = "scheduler_state" in ckpt
    else:
        optimizer_restored = False
        scheduler_restored = False

    kernel, kernel_raw = load_kernel_matrix(args.kernel_path)
    phi16 = load_phi(args.fine_config_source)
    phi8 = load_phi(args.coarse_config_source)
    pairs = split_pairs(phi16, kernel)
    original_state = ckpt["state"]
    original_history = list(ckpt.get("history", []))
    train_idx = np.asarray(original_state["train_idx"], dtype=np.int64)
    val_idx = np.asarray(original_state["val_idx"], dtype=np.int64)

    c_flat = pairs["coarse"].reshape(len(phi16), -1)
    d_flat = pairs["detail"].reshape(len(phi16), -1)
    c_std, c_stats = standardize(c_flat[train_idx], c_flat)
    d_std, d_stats = standardize(d_flat[train_idx], d_flat)
    state = {"stats": {"coarse": c_stats, "detail": d_stats}, "train_idx": train_idx, "val_idx": val_idx, "kernel_path": str(args.kernel_path)}

    model = ARDetailFlow(layers=args.layers, hidden=args.hidden_channels, kernel_size=args.conv_kernel_size, log_scale_bound=args.log_scale_bound).to(device)
    transfer_info: dict[str, Any]
    source_model: ARDetailFlow | None = None
    if args.layers == source_layers:
        model.load_state_dict(ckpt["model_state"])
        transfer_info = {"method": "same_architecture_load_state_dict", "source_layers": source_layers, "target_layers": args.layers}
    elif args.layers > source_layers:
        source_model = ARDetailFlow(layers=source_layers, hidden=args.hidden_channels, kernel_size=args.conv_kernel_size, log_scale_bound=args.log_scale_bound).to(device)
        source_model.load_state_dict(ckpt["model_state"])
        transfer_info = transfer_prefix_layers(source_model, model, source_layers, args.layers)
    else:
        raise ValueError(f"target layers {args.layers} is smaller than source layers {source_layers}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if optimizer_restored:
        try:
            opt.load_state_dict(ckpt["optimizer_state"])
            for group in opt.param_groups:
                group["lr"] = args.lr
        except Exception:
            optimizer_restored = False
    scheduler = None
    scheduler_restored = False

    train_ds = TensorDataset(
        torch.from_numpy(c_std[train_idx].reshape(-1, 8, 8)),
        torch.from_numpy(d_std[train_idx].reshape(-1, 3, 8, 8)),
        torch.from_numpy(pairs["coarse"][train_idx]),
    )
    val_ds = TensorDataset(torch.from_numpy(c_std[val_idx].reshape(-1, 8, 8)), torch.from_numpy(d_std[val_idx].reshape(-1, 3, 8, 8)))
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    kt = torch_kernel_fft(kernel, 16, device)
    targets = native_targets(phi16, val_idx)
    d_mean = torch.tensor(d_stats["mean"].reshape(1, 3, 8, 8), dtype=torch.float32, device=device)
    d_std_t = torch.tensor(d_stats["std"].reshape(1, 3, 8, 8), dtype=torch.float32, device=device)
    init_diag = initialization_diagnostics(model, source_model, state, args, run, kernel)
    write_json(run / "debug" / "weight_transfer.json", transfer_info)

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
            obs = torch_observables(torch_inverse_kernel(psi, kt))
            penalty = 0.0
            zvals = {}
            for key, weight in weights.items():
                z = (obs[key] - float(targets[key]["mean"])) / float(targets[key]["std"])
                penalty += float(weight) * float((z * z).detach().cpu())
                zvals[f"z_{key}"] = float(z.detach().cpu())
        return penalty, zvals

    full_history: list[dict[str, Any]] = []
    continuation_history: list[dict[str, Any]] = []
    patch_rows: list[dict[str, Any]] = []
    best_val = validation_nll()
    best_epoch = 0
    best_abs_epoch = source_abs_epoch
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_opt_state = copy.deepcopy(opt.state_dict())
    latest_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    latest_opt_state = copy.deepcopy(opt.state_dict())
    latest_epoch = 0
    latest_abs_epoch = source_abs_epoch
    best_patch_row: dict[str, Any] | None = None
    bad = 0
    initial_penalty, initial_z = observable_penalty_once()
    initial_eval = run_checkpoint_eval(
        model=model,
        phi16=phi16,
        phi8=phi8,
        kernel=kernel,
        state=state,
        args=args,
        run=run,
        label="cont000",
        continuation_epoch=0,
        validation_nll=best_val,
        observable_penalty=initial_penalty,
    )
    patch_rows.append(initial_eval)
    best_patch_row = initial_eval
    print(json.dumps({"continuation_epoch": 0, "validation_nll": best_val, "observable_penalty": initial_penalty, **initial_z, "patch_acceptance": initial_eval["acceptance"], "DeltaS_std": initial_eval["DeltaS_std"]}), flush=True)

    for epoch in range(1, args.additional_epochs + 1):
        model.train()
        totals = {"loss": 0.0, "nll": 0.0, "obs": 0.0, "count": 0}
        last_z: dict[str, float] = {}
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
            obs = torch_observables(torch_inverse_kernel(psi, kt))
            obs_loss = cb.new_tensor(0.0)
            for key, weight in weights.items():
                z = (obs[key] - float(targets[key]["mean"])) / float(targets[key]["std"])
                obs_loss = obs_loss + float(weight) * z * z
                last_z[f"z_{key}"] = float(z.detach().cpu())
            loss = nll + obs_loss
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            bs = int(len(cb))
            totals["loss"] += float(loss.detach().cpu()) * bs
            totals["nll"] += float(nll.detach().cpu()) * bs
            totals["obs"] += float(obs_loss.detach().cpu()) * bs
            totals["count"] += bs
        val = validation_nll()
        row = {
            "continuation_epoch": epoch,
            "absolute_epoch": source_abs_epoch + epoch,
            "train_loss": totals["loss"] / totals["count"],
            "train_nll": totals["nll"] / totals["count"],
            "train_observable_penalty": totals["obs"] / totals["count"],
            "validation_nll": val,
            **last_z,
        }
        continuation_history.append(row)
        full_history.append(row)
        latest_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        latest_opt_state = copy.deepcopy(opt.state_dict())
        latest_epoch = epoch
        latest_abs_epoch = source_abs_epoch + epoch
        print(json.dumps(row), flush=True)
        if val < best_val - 1.0e-8:
            best_val = val
            best_epoch = epoch
            best_abs_epoch = source_abs_epoch + epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_opt_state = copy.deepcopy(opt.state_dict())
            bad = 0
        else:
            bad += 1
        if epoch % args.checkpoint_every_epochs == 0:
            full_checkpoint(
                path=run / "checkpoints" / f"checkpoint_cont_epoch_{epoch:04d}.pt",
                model=model,
                optimizer=opt,
                scheduler=scheduler,
                state=state,
                history=original_history + continuation_history,
                args=args,
                absolute_epoch=source_abs_epoch + epoch,
                continuation_epoch=epoch,
                best_validation_nll=best_val,
                patience_counter=bad,
                observable_weights=weights,
                optimizer_restored_from_source=optimizer_restored,
                scheduler_restored_from_source=scheduler_restored,
            )
        if epoch % args.eval_every == 0:
            eval_row = run_checkpoint_eval(
                model=model,
                phi16=phi16,
                phi8=phi8,
                kernel=kernel,
                state=state,
                args=args,
                run=run,
                label=f"cont{epoch:03d}",
                continuation_epoch=epoch,
                validation_nll=val,
                observable_penalty=row["train_observable_penalty"],
            )
            patch_rows.append(eval_row)
            if epoch == best_epoch:
                best_patch_row = eval_row
            write_csv(run / "observables" / "intermediate_patch_evaluations.csv", patch_rows)
        if bad >= args.patience:
            break

    model.load_state_dict(best_state)
    opt.load_state_dict(best_opt_state)
    full_checkpoint(
        path=run / "checkpoints" / "checkpoint_best.pt",
        model=model,
        optimizer=opt,
        scheduler=scheduler,
        state=state,
        history=original_history + continuation_history,
        args=args,
        absolute_epoch=best_abs_epoch,
        continuation_epoch=best_epoch,
        best_validation_nll=best_val,
        patience_counter=bad,
        observable_weights=weights,
        optimizer_restored_from_source=optimizer_restored,
        scheduler_restored_from_source=scheduler_restored,
    )
    model.load_state_dict(latest_state)
    opt.load_state_dict(latest_opt_state)
    full_checkpoint(
        path=run / "checkpoints" / "checkpoint_latest.pt",
        model=model,
        optimizer=opt,
        scheduler=scheduler,
        state=state,
        history=original_history + continuation_history,
        args=args,
        absolute_epoch=latest_abs_epoch,
        continuation_epoch=latest_epoch,
        best_validation_nll=best_val,
        patience_counter=bad,
        observable_weights=weights,
        optimizer_restored_from_source=optimizer_restored,
        scheduler_restored_from_source=scheduler_restored,
    )
    model.load_state_dict(best_state)
    opt.load_state_dict(best_opt_state)
    write_csv(run / "observables" / "continuation_training_history.csv", continuation_history)
    write_json(
        run / "debug" / "original_history_link.json",
        {
            "source_run": str(args.source_run),
            "source_checkpoint": str(args.resume_checkpoint),
            "source_absolute_epoch": source_abs_epoch,
            "original_history_len": len(original_history),
            "source_rng_restored": rng_restored,
        },
    )

    final_generated = evaluate_generated(model, phi16, load_phi(args.coarse_config_source), kernel, state["stats"], args, run, "best_continuation")
    action = ActionSpec("phi4_nn", 1.0, 0.340301)
    final_patch = exact_detail_test(model, phi8[: args.patch_test_chains], kernel, state["stats"], action, args)
    write_csv(run / "observables" / "acceptance_history.csv", final_patch["rows"])
    final_rows, final_g = per_config_rows(final_patch["phi"], action, "patch_test_best_continuation")
    write_csv(run / "observables" / "patch_test_observables_per_config.csv", final_rows)
    write_csv(run / "observables" / "patch_test_Gk_per_config.csv", final_g)
    final_patch_comp = summarize_comparison(phi16[: len(final_patch["phi"])], final_patch["phi"], action)
    write_csv(run / "observables" / "patch_test_observable_comparison.csv", final_patch_comp)
    write_csv(run / "observables" / "intermediate_patch_evaluations.csv", patch_rows)

    import yaml

    cfg = {
        "lambda": 1.0,
        "kappa_f": 0.340301,
        "kappa_c": 0.340301,
        "eta": 0.25,
        "eta_scale_numeric": ETA_SCALE,
        "block_factor": 2,
        "L_c": 8,
        "L_f": 16,
        "fine_config_source": str(args.fine_config_source),
        "coarse_config_source": str(args.coarse_config_source),
        "kernel_path": str(args.kernel_path),
        "kernel_name": kernel_raw.get("name"),
        "kernel_coefficients_include_eta_scale": True,
        "kernel_sum": float(kernel.sum()),
        "mode": "continue_autoregressive_coupling_detail_flow",
        "source_run": str(args.source_run),
        "resume_checkpoint": str(args.resume_checkpoint),
        "architecture_unchanged": True,
        "architecture": {
            "factorization": "q(d01|coarse) q(d10|coarse,d01) q(d11|coarse,d01,d10)",
            "stage_flow": "procedural circular-conv checkerboard affine coupling",
            "layers_per_stage": args.layers,
            "hidden_channels": args.hidden_channels,
            "conv_kernel_size": args.conv_kernel_size,
            "log_scale_bound": args.log_scale_bound,
        },
        "optimizer_restored": optimizer_restored,
        "scheduler_restored": scheduler_restored,
        "rng_restored": rng_restored,
        "source_layers": source_layers,
        "target_layers": args.layers,
        "weight_transfer": transfer_info,
        "initialization_diagnostics": init_diag,
        "learning_rate": args.lr,
        "additional_epochs_requested": args.additional_epochs,
        "patience": args.patience,
        "objective": {"NLL": 1.0, "observable_weights": weights},
        "random_seed": args.random_seed,
        "checkpoint_every": args.checkpoint_every_epochs,
        "resume": {"enabled": True, "checkpoint_path": str(args.resume_checkpoint)},
    }
    (run / "run_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    write_json(
        run / "submit_manifest.txt",
        {
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "command": " ".join(sys.argv),
            "git_commit": git_commit(),
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "kernel_path": str(args.kernel_path),
            "kernel_sum": float(kernel.sum()),
            "eta_scale_numeric": ETA_SCALE,
            "source_checkpoint": str(args.resume_checkpoint),
            "optimizer_restored": optimizer_restored,
            "scheduler_restored": scheduler_restored,
            "rng_restored": rng_restored,
            "source_absolute_epoch": source_abs_epoch,
            "source_layers": source_layers,
            "target_layers": args.layers,
            "weight_transfer": transfer_info,
        },
    )
    raw_lookup = {r["observable"]: r for r in final_generated["comparison"]}
    patch_lookup = {r["observable"]: r for r in final_patch_comp}
    write_json(
        run / "debug" / "stability_diagnostics.json",
        {
            "generated": {k: v for k, v in final_generated.items() if k not in {"phi", "comparison"}},
            "patch_test": final_patch["summary"],
            "kernel": {"path": str(args.kernel_path), "sum": float(kernel.sum()), "eta_scale": ETA_SCALE, "kernel_coefficients_include_eta_scale": True},
            "initialization_diagnostics": init_diag,
            "weight_transfer": transfer_info,
            "best_continuation_epoch": best_epoch,
            "best_absolute_epoch": best_abs_epoch,
            "best_validation_nll": best_val,
        },
    )
    lines = [
        "# Lambda 1.0 L8->L16 AR Detail Flow Continuation",
        "",
        f"- source run: `{args.source_run}`",
        f"- resumed checkpoint: `{args.resume_checkpoint}`",
        f"- optimizer restored: `{optimizer_restored}`",
        f"- scheduler restored: `{scheduler_restored}`",
        f"- source RNG restored: `{rng_restored}`",
        f"- source layers: `{source_layers}`",
        f"- target layers: `{args.layers}`",
        f"- weight transfer: `{transfer_info['method']}`",
        f"- kernel path: `{args.kernel_path}`",
        f"- kernel sum: `{float(kernel.sum()):.17g}`",
        f"- eta scale: `{ETA_SCALE:.17g}`",
        f"- best continuation epoch: `{best_epoch}`",
        f"- best absolute epoch: `{best_abs_epoch}`",
        f"- best validation NLL: `{best_val:.6g}`",
        f"- checkpoint best: `{run / 'checkpoints' / 'checkpoint_best.pt'}`",
        f"- final patch acceptance: `{final_patch['summary']['acceptance']:.6g}`",
        f"- final DeltaS mean/std: `{final_patch['summary']['DeltaS_mean']:.6g}` / `{final_patch['summary']['DeltaS_std']:.6g}`",
        f"- final logacc mean/std: `{final_patch['summary']['log_accept_mean']:.6g}` / `{final_patch['summary']['log_accept_std']:.6g}`",
        "",
        "## Final Shifts",
        "",
        "| observable | raw shift | raw std ratio | patch shift | patch std ratio |",
        "|---|---:|---:|---:|---:|",
    ]
    for obs in ["action_density", "phi2", "phi4", "local_kurtosis_ratio", "NN", "2nn", "diag", "m2", "m4", "G_pmin_avg"]:
        rr = raw_lookup[obs]
        pp = patch_lookup[obs]
        lines.append(f"| {obs} | {float(rr['standardized_mean_shift']):.6g} | {float(rr['std_ratio']):.6g} | {float(pp['standardized_mean_shift']):.6g} | {float(pp['std_ratio']):.6g} |")
    lines += [
        "",
        "## Intermediate Patch Evaluations",
        "",
        "| epoch | val NLL | acceptance | DeltaS std | logacc mean | patch kurtosis shift | patch action shift |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in patch_rows:
        lines.append(f"| {r['continuation_epoch']} | {float(r['validation_nll']):.6g} | {float(r['acceptance']):.6g} | {float(r['DeltaS_std']):.6g} | {float(r['log_accept_mean']):.6g} | {float(r['patch_local_kurtosis_ratio_shift']):.6g} | {float(r['patch_action_density_shift']):.6g} |")
    (run / "summaries" / "run_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(
        run / "status.json",
        {
            "status": "completed",
            "best_continuation_epoch": best_epoch,
            "best_absolute_epoch": best_abs_epoch,
            "best_checkpoint": str(run / "checkpoints" / "checkpoint_best.pt"),
            "latest_checkpoint": str(run / "checkpoints" / "checkpoint_latest.pt"),
            "summary": str(run / "summaries" / "run_summary.md"),
            "validation_plateau": bad >= args.patience,
            "epochs_completed": len(continuation_history),
        },
    )
    print(json.dumps({"status": "completed", "run_dir": str(run), "best_epoch": best_epoch, "best_validation_nll": best_val, "acceptance": final_patch["summary"]["acceptance"], "DeltaS_std": final_patch["summary"]["DeltaS_std"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
