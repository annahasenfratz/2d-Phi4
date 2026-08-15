#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import platform
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

from perfect_blocking_upsampling.conv_pair import build_procedural_conv_flow  # noqa: E402
from perfect_blocking_upsampling.io import ActionSpec  # noqa: E402
from train_lam1p0_flow_detail_localreg import (  # noqa: E402
    ETA_SCALE,
    git_commit,
    native_targets,
    torch_inverse_kernel,
    torch_kernel_fft,
    torch_observables,
)
from train_lam1p0_flow_detail_pilot import (  # noqa: E402
    apply_kernel,
    assemble_psi,
    inverse_kernel,
    load_kernel_matrix,
    load_phi,
    per_config_rows,
    split_pairs,
    summarize_comparison,
    write_csv,
    write_json,
)


DEFAULT_WEIGHTS = {
    "action_density": 0.06,
    "phi4": 0.08,
    "local_kurtosis_ratio": 0.08,
    "NN": 0.04,
    "phi2": 0.015,
    "2nn": 0.01,
    "diag": 0.01,
    "G_pmin_avg": 0.002,
}


def parse_weights(text: str) -> dict[str, float]:
    weights = dict(DEFAULT_WEIGHTS)
    if not text:
        return weights
    for item in text.split(","):
        if not item.strip():
            continue
        key, val = item.split("=")
        weights[key.strip()] = float(val)
    return weights


def standardize(train: np.ndarray, full: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    mean = train.mean(axis=0, keepdims=True)
    std = np.maximum(train.std(axis=0, keepdims=True), 1.0e-6)
    return ((full - mean) / std).astype(np.float32), {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}


class ARDetailFlow(torch.nn.Module):
    def __init__(self, *, layers: int, hidden: int, kernel_size: int, log_scale_bound: float):
        super().__init__()
        self.flows = torch.nn.ModuleList(
            [
                build_procedural_conv_flow(
                    cond_channels=1,
                    target_channels=1,
                    lattice_size=8,
                    n_coupling_layers=layers,
                    conv_hidden_channels=hidden,
                    conv_kernel_size=kernel_size,
                    log_scale_bound=log_scale_bound,
                ),
                build_procedural_conv_flow(
                    cond_channels=2,
                    target_channels=1,
                    lattice_size=8,
                    n_coupling_layers=layers,
                    conv_hidden_channels=hidden,
                    conv_kernel_size=kernel_size,
                    log_scale_bound=log_scale_bound,
                ),
                build_procedural_conv_flow(
                    cond_channels=3,
                    target_channels=1,
                    lattice_size=8,
                    n_coupling_layers=layers,
                    conv_hidden_channels=hidden,
                    conv_kernel_size=kernel_size,
                    log_scale_bound=log_scale_bound,
                ),
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


def save_checkpoint(path: Path, model: ARDetailFlow, state: dict[str, Any], history: list[dict[str, Any]], args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_class": "ARDetailFlow",
            "state": state,
            "history": history,
            "config": vars(args),
        },
        path,
    )


def unstandardize_detail(d: torch.Tensor, stats: dict[str, Any], device: torch.device) -> torch.Tensor:
    mean = torch.tensor(stats["detail"]["mean"].reshape(1, 3, 8, 8), dtype=d.dtype, device=device)
    std = torch.tensor(stats["detail"]["std"].reshape(1, 3, 8, 8), dtype=d.dtype, device=device)
    return d * std + mean


def sample_model(model: ARDetailFlow, coarse_phys: np.ndarray, stats: dict[str, Any], args: argparse.Namespace, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    device = torch.device(args.device)
    torch.manual_seed(seed)
    c = ((coarse_phys.reshape(len(coarse_phys), -1) - stats["coarse"]["mean"]) / stats["coarse"]["std"]).reshape(len(coarse_phys), 8, 8).astype(np.float32)
    detail_chunks = []
    logq_chunks = []
    z_chunks = []
    logdet_chunks = []
    model.eval()
    log_jac_const = -float(np.sum(np.log(stats["detail"]["std"])))
    with torch.no_grad():
        for start in range(0, len(c), args.batch_size):
            cb = torch.from_numpy(c[start : start + args.batch_size]).to(device)
            d_std, logq_std, zmax, logdet = model.sample(cb)
            d_phys = unstandardize_detail(d_std, stats, device)
            detail_chunks.append(d_phys.detach().cpu().numpy().astype(np.float32))
            logq_chunks.append((logq_std.detach().cpu().numpy() + log_jac_const).astype(np.float64))
            z_chunks.append(zmax.detach().cpu().numpy().astype(np.float32))
            logdet_chunks.append((logdet.detach().cpu().numpy() + float(np.sum(np.log(stats["detail"]["std"])))).astype(np.float64))
    return np.concatenate(detail_chunks), np.concatenate(logq_chunks), np.concatenate(z_chunks), np.concatenate(logdet_chunks)


def log_prob_model(model: ARDetailFlow, coarse_phys: np.ndarray, detail_phys: np.ndarray, stats: dict[str, Any], args: argparse.Namespace) -> np.ndarray:
    device = torch.device(args.device)
    c = ((coarse_phys.reshape(len(coarse_phys), -1) - stats["coarse"]["mean"]) / stats["coarse"]["std"]).reshape(len(coarse_phys), 8, 8).astype(np.float32)
    d = ((detail_phys.reshape(len(detail_phys), -1) - stats["detail"]["mean"]) / stats["detail"]["std"]).reshape(len(detail_phys), 3, 8, 8).astype(np.float32)
    vals = []
    log_jac_const = -float(np.sum(np.log(stats["detail"]["std"])))
    model.eval()
    with torch.no_grad():
        for start in range(0, len(c), args.batch_size):
            cb = torch.from_numpy(c[start : start + args.batch_size]).to(device)
            db = torch.from_numpy(d[start : start + args.batch_size]).to(device)
            vals.append((model.log_prob(cb, db).detach().cpu().numpy() + log_jac_const).astype(np.float64))
    return np.concatenate(vals)


def evaluate_generated(model: ARDetailFlow, phi16: np.ndarray, phi8: np.ndarray, kernel: np.ndarray, stats: dict[str, Any], args: argparse.Namespace, run: Path, tag: str) -> dict[str, Any]:
    action = ActionSpec("phi4_nn", 1.0, 0.340301)
    n_gen = min(args.generated_count, len(phi8), len(phi16))
    detail, logq, zmax, logdet = sample_model(model, phi8[:n_gen], stats, args, args.random_seed + 5000)
    psi = assemble_psi(phi8[:n_gen], detail)
    phi, inv = inverse_kernel(psi, kernel)
    reb = apply_kernel(phi, kernel)[:, 0::2, 0::2] - phi8[:n_gen]
    native_rows, native_g = per_config_rows(phi16[:n_gen], action, "native_L16")
    gen_rows, gen_g = per_config_rows(phi, action, f"generated_{tag}")
    write_csv(run / "observables" / "all_observables_per_config.csv", native_rows + gen_rows)
    write_csv(run / "observables" / "Gk_per_config.csv", native_g + gen_g)
    comp = summarize_comparison(phi16[:n_gen], phi, action)
    write_csv(run / "observables" / "first_observable_comparison.csv", comp)
    return {
        "phi": phi,
        "comparison": comp,
        "nonfinite_count": int(np.sum(~np.isfinite(phi)) + np.sum(~np.isfinite(detail))),
        "max_abs_z": float(np.max(zmax)),
        "logq_mean": float(np.mean(logq)),
        "logq_std": float(np.std(logq)),
        "logdet_mean": float(np.mean(logdet)),
        "logdet_std": float(np.std(logdet)),
        "logdet_min": float(np.min(logdet)),
        "logdet_max": float(np.max(logdet)),
        "inverse_kernel": inv,
        "reblocking_max_abs_error": float(np.max(np.abs(reb))),
        "reblocking_rms_error": float(np.sqrt(np.mean(reb.astype(np.float64) ** 2))),
    }


def exact_detail_test(model: ARDetailFlow, coarse: np.ndarray, kernel: np.ndarray, stats: dict[str, Any], action: ActionSpec, args: argparse.Namespace) -> dict[str, Any]:
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
        rows.append(
            {
                "sweep": sweep,
                "attempts": n,
                "accepted": int(np.sum(acc)),
                "acceptance": float(np.mean(acc)),
                "DeltaS_mean": float(np.mean(delta_s)),
                "DeltaS_std": float(np.std(delta_s)),
                "log_accept_mean": float(np.mean(log_acc)),
                "log_accept_std": float(np.std(log_acc)),
            }
        )
    ds = np.concatenate(delta_s_vals)
    la = np.concatenate(log_acc_vals)
    return {
        "phi": phi.astype(np.float32),
        "rows": rows,
        "summary": {
            "attempts": attempts_total,
            "accepted": accepted_total,
            "acceptance": float(accepted_total / max(attempts_total, 1)),
            "DeltaS_mean": float(np.mean(ds)),
            "DeltaS_std": float(np.std(ds)),
            "DeltaS_p01": float(np.quantile(ds, 0.01)),
            "DeltaS_p05": float(np.quantile(ds, 0.05)),
            "DeltaS_p50": float(np.quantile(ds, 0.50)),
            "DeltaS_p95": float(np.quantile(ds, 0.95)),
            "DeltaS_p99": float(np.quantile(ds, 0.99)),
            "log_accept_mean": float(np.mean(la)),
            "log_accept_std": float(np.std(la)),
            "log_accept_p01": float(np.quantile(la, 0.01)),
            "log_accept_p05": float(np.quantile(la, 0.05)),
            "log_accept_p50": float(np.quantile(la, 0.50)),
            "log_accept_p95": float(np.quantile(la, 0.95)),
            "log_accept_p99": float(np.quantile(la, 0.99)),
        },
    }


def action_total(phi: np.ndarray, action: ActionSpec) -> np.ndarray:
    arr = np.asarray(phi, dtype=np.float64)
    phi2 = arr * arr
    phi4 = phi2 * phi2
    nn = arr * np.roll(arr, -1, axis=-1) + arr * np.roll(arr, -1, axis=-2)
    dens = (1.0 - 2.0 * action.lambda_) * phi2 + action.lambda_ * phi4 - 2.0 * action.kappa * nn
    return dens.sum(axis=(1, 2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--fine-config-source", type=Path, default=Path("data/configs_phi4_2d/lam1p0_kappac0p340301_L16/configs.npz"))
    ap.add_argument("--coarse-config-source", type=Path, default=Path("data/configs_phi4_2d/lam1p0_kappac0p340301_L8/configs.npz"))
    ap.add_argument("--kernel-path", type=Path, default=Path("perfect_blocking/perfect_blocking_lam1p0/kernels/selected_for_upscaling/best_7x7_phi2_nn_guarded_eta_included.json"))
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5.0e-4)
    ap.add_argument("--weight-decay", type=float, default=1.0e-5)
    ap.add_argument("--grad-clip", type=float, default=10.0)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--hidden-channels", type=int, default=48)
    ap.add_argument("--conv-kernel-size", type=int, default=3)
    ap.add_argument("--log-scale-bound", type=float, default=0.75)
    ap.add_argument("--obs-weights", default="")
    ap.add_argument("--train-count", type=int, default=4000)
    ap.add_argument("--val-count", type=int, default=1000)
    ap.add_argument("--generated-count", type=int, default=512)
    ap.add_argument("--patch-test-chains", type=int, default=64)
    ap.add_argument("--patch-test-sweeps", type=int, default=100)
    ap.add_argument("--checkpoint-every-epochs", type=int, default=4)
    ap.add_argument("--random-seed", type=int, default=2026071603)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    run = args.run_dir
    for sub in ["logs", "checkpoints", "observables", "plots", "summaries", "debug"]:
        (run / sub).mkdir(parents=True, exist_ok=True)
    weights = parse_weights(args.obs_weights)
    device = torch.device(args.device)
    kernel, kernel_raw = load_kernel_matrix(args.kernel_path)
    phi16 = load_phi(args.fine_config_source)
    phi8 = load_phi(args.coarse_config_source)
    pairs = split_pairs(phi16, kernel)
    rng = np.random.default_rng(args.random_seed)
    idx = rng.permutation(len(phi16))
    train_idx = idx[: args.train_count]
    val_idx = idx[args.train_count : args.train_count + args.val_count]
    c_flat = pairs["coarse"].reshape(len(phi16), -1)
    d_flat = pairs["detail"].reshape(len(phi16), -1)
    c_std, c_stats = standardize(c_flat[train_idx], c_flat)
    d_std, d_stats = standardize(d_flat[train_idx], d_flat)
    stats = {"coarse": c_stats, "detail": d_stats}
    model = ARDetailFlow(layers=args.layers, hidden=args.hidden_channels, kernel_size=args.conv_kernel_size, log_scale_bound=args.log_scale_bound).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_ds = TensorDataset(torch.from_numpy(c_std[train_idx].reshape(-1, 8, 8)), torch.from_numpy(d_std[train_idx].reshape(-1, 3, 8, 8)), torch.from_numpy(pairs["coarse"][train_idx]))
    val_ds = TensorDataset(torch.from_numpy(c_std[val_idx].reshape(-1, 8, 8)), torch.from_numpy(d_std[val_idx].reshape(-1, 3, 8, 8)))
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    kt = torch_kernel_fft(kernel, 16, device)
    targets = native_targets(phi16, val_idx)
    d_mean = torch.tensor(d_stats["mean"].reshape(1, 3, 8, 8), dtype=torch.float32, device=device)
    d_std_t = torch.tensor(d_stats["std"].reshape(1, 3, 8, 8), dtype=torch.float32, device=device)

    def val_nll() -> float:
        model.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for cb, db in val_loader:
                lp = model.log_prob(cb.to(device), db.to(device))
                total += float((-lp).sum().detach().cpu())
                count += int(len(cb))
        return total / max(count, 1)

    history: list[dict[str, Any]] = []
    best_score = float("inf")
    best_state = None
    bad = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = {"loss": 0.0, "nll": 0.0, "obs": 0.0, "count": 0}
        last_z: dict[str, float] = {}
        for cb, db, coarse_phys in loader:
            cb = cb.to(device)
            db = db.to(device)
            coarse_phys = coarse_phys.to(device)
            opt.zero_grad(set_to_none=True)
            nll = -model.log_prob(cb, db).mean()
            d_samp, _logq, _zmax, _logdet = model.sample(cb)
            detail_phys = d_samp * d_std_t + d_mean
            psi = torch.empty((detail_phys.shape[0], 16, 16), dtype=detail_phys.dtype, device=device)
            psi[:, 0::2, 0::2] = coarse_phys
            psi[:, 0::2, 1::2] = detail_phys[:, 0]
            psi[:, 1::2, 0::2] = detail_phys[:, 1]
            psi[:, 1::2, 1::2] = detail_phys[:, 2]
            phi = torch_inverse_kernel(psi, kt)
            obs = torch_observables(phi)
            obs_loss = phi.new_tensor(0.0)
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
        val = val_nll()
        row = {
            "epoch": epoch,
            "train_loss": totals["loss"] / totals["count"],
            "train_nll": totals["nll"] / totals["count"],
            "train_observable_penalty": totals["obs"] / totals["count"],
            "validation_nll": val,
            **last_z,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        score = val + row["train_observable_penalty"]
        if score < best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if epoch % args.checkpoint_every_epochs == 0:
            save_checkpoint(run / "checkpoints" / f"checkpoint_epoch_{epoch:04d}.pt", model, {"stats": stats, "train_idx": train_idx, "val_idx": val_idx, "kernel_path": str(args.kernel_path)}, history, args)
        if bad >= args.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    state = {"stats": stats, "train_idx": train_idx, "val_idx": val_idx, "kernel_path": str(args.kernel_path)}
    save_checkpoint(run / "checkpoints" / "checkpoint_best.pt", model, state, history, args)
    save_checkpoint(run / "checkpoints" / "checkpoint_latest.pt", model, state, history, args)
    write_csv(run / "observables" / "training_history.csv", history)

    action = ActionSpec("phi4_nn", 1.0, 0.340301)
    generated = evaluate_generated(model, phi16, phi8, kernel, stats, args, run, "ar_detail")
    patch = exact_detail_test(model, phi8[: args.patch_test_chains], kernel, stats, action, args)
    write_csv(run / "observables" / "acceptance_history.csv", patch["rows"])
    patch_rows, patch_g = per_config_rows(patch["phi"], action, "patch_test_L16")
    write_csv(run / "observables" / "patch_test_observables_per_config.csv", patch_rows)
    write_csv(run / "observables" / "patch_test_Gk_per_config.csv", patch_g)
    patch_comp = summarize_comparison(phi16[: len(patch["phi"])], patch["phi"], action)
    write_csv(run / "observables" / "patch_test_observable_comparison.csv", patch_comp)

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
        "mode": "train_autoregressive_coupling_detail_flow",
        "architecture": {
            "factorization": "q(d01|coarse) q(d10|coarse,d01) q(d11|coarse,d01,d10)",
            "stage_flow": "procedural circular-conv checkerboard affine coupling",
            "layers_per_stage": args.layers,
            "hidden_channels": args.hidden_channels,
            "conv_kernel_size": args.conv_kernel_size,
            "log_scale_bound": args.log_scale_bound,
        },
        "objective": {"NLL": 1.0, "observable_weights": weights},
        "random_seed": args.random_seed,
        "checkpoint_every": args.checkpoint_every_epochs,
        "resume": {"enabled": False, "checkpoint_path": None},
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
            "raw_config_policy": "input configs remain under data/configs_phi4_2d; run directory stores checkpoints and observables only",
        },
    )
    write_json(
        run / "debug" / "stability_diagnostics.json",
        {
            "generated": {k: v for k, v in generated.items() if k != "phi" and k != "comparison"},
            "patch_test": patch["summary"],
            "kernel": {"path": str(args.kernel_path), "sum": float(kernel.sum()), "eta_scale": ETA_SCALE, "kernel_coefficients_include_eta_scale": True},
        },
    )
    raw = {r["observable"]: r for r in generated["comparison"]}
    pcomp = {r["observable"]: r for r in patch_comp}
    lines = [
        "# Lambda 1.0 L8->L16 Autoregressive Coupling Detail Flow",
        "",
        f"- architecture: `q(d01|c) q(d10|c,d01) q(d11|c,d01,d10)` with conv affine coupling stages",
        f"- kernel path: `{args.kernel_path}`",
        f"- kernel sum: `{float(kernel.sum()):.17g}`",
        f"- eta scale: `{ETA_SCALE:.17g}`",
        f"- epochs run: `{len(history)}`",
        f"- best score: `{best_score:.6g}`",
        f"- generated nonfinite count: `{generated['nonfinite_count']}`",
        f"- generated max |z|: `{generated['max_abs_z']:.6g}`",
        f"- log-Jacobian mean/std/min/max: `{generated['logdet_mean']:.6g}` / `{generated['logdet_std']:.6g}` / `{generated['logdet_min']:.6g}` / `{generated['logdet_max']:.6g}`",
        f"- reblocking max error: `{generated['reblocking_max_abs_error']:.6g}`",
        f"- patch acceptance: `{patch['summary']['acceptance']:.6g}`",
        f"- DeltaS mean/std: `{patch['summary']['DeltaS_mean']:.6g}` / `{patch['summary']['DeltaS_std']:.6g}`",
        f"- log-accept mean/std: `{patch['summary']['log_accept_mean']:.6g}` / `{patch['summary']['log_accept_std']:.6g}`",
        "",
        "## Generated and Patch-Test Shifts",
        "",
        "| observable | raw shift | raw std ratio | patch shift | patch std ratio |",
        "|---|---:|---:|---:|---:|",
    ]
    for obs in ["action_density", "phi2", "phi4", "local_kurtosis_ratio", "NN", "m2", "m4", "G_pmin_avg"]:
        rr = raw[obs]
        pp = pcomp[obs]
        lines.append(f"| {obs} | {float(rr['standardized_mean_shift']):.6g} | {float(rr['std_ratio']):.6g} | {float(pp['standardized_mean_shift']):.6g} | {float(pp['std_ratio']):.6g} |")
    (run / "summaries" / "run_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(
        run / "status.json",
        {
            "status": "completed",
            "epochs_completed": len(history),
            "latest_checkpoint": str(run / "checkpoints" / "checkpoint_latest.pt"),
            "best_checkpoint": str(run / "checkpoints" / "checkpoint_best.pt"),
            "summary": str(run / "summaries" / "run_summary.md"),
        },
    )
    print(json.dumps({"status": "completed", "run_dir": str(run), "acceptance": patch["summary"]["acceptance"], "DeltaS_std": patch["summary"]["DeltaS_std"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
