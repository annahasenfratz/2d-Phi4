#!/usr/bin/env python3
from __future__ import annotations

import argparse
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

from perfect_blocking_upsampling.io import ActionSpec  # noqa: E402
from train_lam1p0_flow_detail_pilot import (  # noqa: E402
    ETA_SCALE,
    DetailGaussian,
    apply_kernel,
    assemble_psi,
    exact_detail_test,
    inverse_kernel,
    load_kernel_matrix,
    load_phi,
    per_config_rows,
    sample_details,
    split_pairs,
    summarize_comparison,
    write_csv,
    write_json,
)


OBS_WEIGHTS_DEFAULT = {
    "action_density": 0.06,
    "phi4": 0.08,
    "local_kurtosis_ratio": 0.08,
    "NN": 0.04,
    "phi2": 0.015,
    "2nn": 0.01,
    "diag": 0.01,
    "G_pmin_avg": 0.002,
}


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def torch_kernel_fft(kernel: np.ndarray, L: int, device: torch.device) -> torch.Tensor:
    w = np.zeros((L, L), dtype=np.float64)
    r = kernel.shape[0] // 2
    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            w[(i - r) % L, (j - r) % L] += float(kernel[i, j])
    return torch.tensor(np.fft.fft2(w), dtype=torch.complex64, device=device)


def torch_inverse_kernel(psi: torch.Tensor, kt: torch.Tensor) -> torch.Tensor:
    return torch.fft.ifft2(torch.fft.fft2(psi.to(torch.complex64), dim=(-2, -1)) / kt[None], dim=(-2, -1)).real


def torch_observables(phi: torch.Tensor, lam: float = 1.0, kappa: float = 0.340301) -> dict[str, torch.Tensor]:
    phi2_cfg = (phi * phi).mean(dim=(1, 2))
    phi4_cfg = (phi**4).mean(dim=(1, 2))
    nn_cfg = 0.5 * (
        (phi * torch.roll(phi, shifts=-1, dims=1)).mean(dim=(1, 2))
        + (phi * torch.roll(phi, shifts=-1, dims=2)).mean(dim=(1, 2))
    )
    two_cfg = 0.5 * (
        (phi * torch.roll(phi, shifts=-2, dims=1)).mean(dim=(1, 2))
        + (phi * torch.roll(phi, shifts=-2, dims=2)).mean(dim=(1, 2))
    )
    diag_cfg = (phi * torch.roll(torch.roll(phi, shifts=-1, dims=1), shifts=-1, dims=2)).mean(dim=(1, 2))
    m_cfg = phi.mean(dim=(1, 2))
    fft = torch.fft.fft2(phi.to(torch.complex64), dim=(1, 2))
    V = float(phi.shape[1] * phi.shape[2])
    gpmin = 0.5 * ((torch.abs(fft[:, 1, 0]) ** 2) / V + (torch.abs(fft[:, 0, 1]) ** 2) / V)
    action = (1.0 - 2.0 * lam) * phi2_cfg + lam * phi4_cfg - 4.0 * kappa * nn_cfg
    return {
        "action_density": action.mean(),
        "phi2": phi2_cfg.mean(),
        "phi4": phi4_cfg.mean(),
        "local_kurtosis_ratio": (phi4_cfg / torch.clamp(phi2_cfg * phi2_cfg, min=1.0e-12)).mean(),
        "NN": nn_cfg.mean(),
        "2nn": two_cfg.mean(),
        "diag": diag_cfg.mean(),
        "m2": (m_cfg * m_cfg).mean(),
        "m4": (m_cfg**4).mean(),
        "G_pmin_avg": gpmin.real.mean(),
    }


def native_targets(phi: np.ndarray, val_idx: np.ndarray) -> dict[str, dict[str, float]]:
    action = ActionSpec("phi4_nn", 1.0, 0.340301)
    rows, grows = per_config_rows(phi[val_idx], action, "native_val")
    vals: dict[str, np.ndarray] = {}
    for key in ["action_density", "phi2", "phi4", "local_kurtosis_ratio", "NN", "2nn", "diag", "m2", "m4"]:
        vals[key] = np.asarray([float(r[key]) for r in rows], dtype=np.float64)
    vals["G_pmin_avg"] = np.asarray([float(r["G_pmin_avg"]) for r in grows], dtype=np.float64)
    return {
        key: {
            "mean": float(np.mean(v)),
            "std": float(max(np.std(v, ddof=1), 1.0e-6)),
        }
        for key, v in vals.items()
    }


def parse_weights(text: str) -> dict[str, float]:
    weights = dict(OBS_WEIGHTS_DEFAULT)
    if not text:
        return weights
    for part in text.split(","):
        if not part.strip():
            continue
        key, val = part.split("=")
        weights[key.strip()] = float(val)
    return weights


def load_baseline_model(checkpoint: Path, device: torch.device) -> tuple[DetailGaussian, dict[str, Any]]:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    coarse_dim = int(np.prod(ckpt["coarse_stats"]["mean"].shape[1:]))
    detail_dim = int(np.prod(ckpt["detail_stats"]["mean"].shape[1:]))
    # Stats were saved as shape (1, D), so prod(shape[1:]) is D.
    model = DetailGaussian(
        coarse_dim=coarse_dim,
        detail_dim=detail_dim,
        hidden=int(cfg.get("hidden_dim", 384)),
        log_std_min=float(cfg.get("log_std_min", -4.0)),
        log_std_max=float(cfg.get("log_std_max", 1.5)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    return model, ckpt


def save_checkpoint(path: Path, model: DetailGaussian, baseline: dict[str, Any], history: list[dict[str, Any]], args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_class": "DetailGaussianLocalReg",
            "coarse_stats": baseline["coarse_stats"],
            "detail_stats": baseline["detail_stats"],
            "train_idx": baseline["train_idx"],
            "val_idx": baseline["val_idx"],
            "history": history,
            "config": vars(args),
            "started_from_checkpoint": str(args.start_checkpoint),
        },
        path,
    )


def evaluate_generated(model: DetailGaussian, phi16: np.ndarray, phi8: np.ndarray, kernel: np.ndarray, baseline: dict[str, Any], args: argparse.Namespace, run: Path, tag: str) -> dict[str, Any]:
    action = ActionSpec("phi4_nn", 1.0, 0.340301)
    n_gen = min(args.generated_count, len(phi8), len(phi16))
    stats = {"coarse_stats": baseline["coarse_stats"], "detail_stats": baseline["detail_stats"]}
    detail, logq, z = sample_details(model, phi8[:n_gen], stats, args, args.random_seed + 17000)
    psi = assemble_psi(phi8[:n_gen], detail)
    phi_gen, inv_info = inverse_kernel(psi, kernel)
    reb = apply_kernel(phi_gen, kernel)[:, 0::2, 0::2] - phi8[:n_gen]
    native_rows, native_g = per_config_rows(phi16[:n_gen], action, "native_L16")
    gen_rows, gen_g = per_config_rows(phi_gen, action, f"generated_{tag}")
    write_csv(run / "observables" / f"{tag}_all_observables_per_config.csv", native_rows + gen_rows)
    write_csv(run / "observables" / f"{tag}_Gk_per_config.csv", native_g + gen_g)
    comp = summarize_comparison(phi16[:n_gen], phi_gen, action)
    write_csv(run / "observables" / f"{tag}_observable_comparison.csv", comp)
    return {
        "phi": phi_gen,
        "comparison": comp,
        "nonfinite_count": int(np.sum(~np.isfinite(phi_gen)) + np.sum(~np.isfinite(detail))),
        "max_abs_z": float(np.max(np.abs(z))),
        "logq_mean": float(np.mean(logq)),
        "logq_std": float(np.std(logq)),
        "inverse_kernel": inv_info,
        "reblocking_max_abs_error": float(np.max(np.abs(reb))),
        "reblocking_rms_error": float(np.sqrt(np.mean(reb.astype(np.float64) ** 2))),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--baseline-run", type=Path, required=True)
    ap.add_argument("--start-checkpoint", type=Path, required=True)
    ap.add_argument("--fine-config-source", type=Path, default=Path("data/configs_phi4_2d/lam1p0_kappac0p340301_L16/configs.npz"))
    ap.add_argument("--coarse-config-source", type=Path, default=Path("data/configs_phi4_2d/lam1p0_kappac0p340301_L8/configs.npz"))
    ap.add_argument("--kernel-path", type=Path, default=Path("perfect_blocking/perfect_blocking_lam1p0/kernels/selected_for_upscaling/best_7x7_phi2_nn_guarded_eta_included.json"))
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3.0e-4)
    ap.add_argument("--weight-decay", type=float, default=1.0e-5)
    ap.add_argument("--grad-clip", type=float, default=10.0)
    ap.add_argument("--obs-weights", default="")
    ap.add_argument("--train-count", type=int, default=4000)
    ap.add_argument("--val-count", type=int, default=1000)
    ap.add_argument("--generated-count", type=int, default=512)
    ap.add_argument("--patch-test-chains", type=int, default=64)
    ap.add_argument("--patch-test-sweeps", type=int, default=100)
    ap.add_argument("--random-seed", type=int, default=2026071602)
    ap.add_argument("--checkpoint-every-epochs", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    run = args.run_dir
    for sub in ["logs", "checkpoints", "observables", "plots", "summaries", "debug"]:
        (run / sub).mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    weights = parse_weights(args.obs_weights)
    kernel, kernel_raw = load_kernel_matrix(args.kernel_path)
    phi16 = load_phi(args.fine_config_source)
    phi8 = load_phi(args.coarse_config_source)
    pairs = split_pairs(phi16, kernel)
    model, baseline = load_baseline_model(args.start_checkpoint, device)
    train_idx = np.asarray(baseline["train_idx"], dtype=np.int64)[: args.train_count]
    val_idx = np.asarray(baseline["val_idx"], dtype=np.int64)[: args.val_count]
    c_stats = baseline["coarse_stats"]
    d_stats = baseline["detail_stats"]
    c_all = ((pairs["coarse"].reshape(len(pairs["coarse"]), -1) - c_stats["mean"]) / c_stats["std"]).astype(np.float32)
    d_all = ((pairs["detail"].reshape(len(pairs["detail"]), -1) - d_stats["mean"]) / d_stats["std"]).astype(np.float32)
    train_ds = TensorDataset(torch.from_numpy(c_all[train_idx]), torch.from_numpy(d_all[train_idx]), torch.from_numpy(pairs["coarse"][train_idx]))
    val_ds = TensorDataset(torch.from_numpy(c_all[val_idx]), torch.from_numpy(d_all[val_idx]))
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    kt = torch_kernel_fft(kernel, 16, device)
    target = native_targets(phi16, val_idx)
    detail_mean = torch.tensor(d_stats["mean"], dtype=torch.float32, device=device)
    detail_std = torch.tensor(d_stats["std"], dtype=torch.float32, device=device)
    history: list[dict[str, Any]] = []
    best_score = float("inf")
    best_state = None
    bad_epochs = 0

    def validation_nll() -> float:
        model.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for cb, db in val_loader:
                vals = model.nll(cb.to(device), db.to(device))
                total += float(vals.sum().detach().cpu())
                count += int(len(cb))
        return total / max(count, 1)

    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = {"loss": 0.0, "nll": 0.0, "obs": 0.0, "count": 0}
        for cb, db, coarse_img in loader:
            cb = cb.to(device)
            db = db.to(device)
            coarse_img = coarse_img.to(device)
            opt.zero_grad(set_to_none=True)
            nll = model.nll(cb, db).mean()
            mean, log_std = model.params(cb)
            eps = torch.randn_like(mean)
            d_std = mean + torch.exp(log_std) * eps
            d_phys = d_std * detail_std + detail_mean
            detail = d_phys.reshape(-1, 3, 8, 8)
            psi = torch.empty((detail.shape[0], 16, 16), dtype=detail.dtype, device=device)
            psi[:, 0::2, 0::2] = coarse_img
            psi[:, 0::2, 1::2] = detail[:, 0]
            psi[:, 1::2, 0::2] = detail[:, 1]
            psi[:, 1::2, 1::2] = detail[:, 2]
            phi = torch_inverse_kernel(psi, kt)
            obs = torch_observables(phi)
            obs_loss = phi.new_tensor(0.0)
            obs_z: dict[str, float] = {}
            for key, weight in weights.items():
                z = (obs[key] - float(target[key]["mean"])) / float(target[key]["std"])
                obs_loss = obs_loss + float(weight) * z * z
                obs_z[f"z_{key}"] = float(z.detach().cpu())
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
            "epoch": epoch,
            "train_loss": totals["loss"] / totals["count"],
            "train_nll": totals["nll"] / totals["count"],
            "train_observable_penalty": totals["obs"] / totals["count"],
            "validation_nll": val,
            **obs_z,
        }
        history.append(row)
        score = val + row["train_observable_penalty"]
        if score < best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        print(json.dumps(row), flush=True)
        if epoch % args.checkpoint_every_epochs == 0:
            save_checkpoint(run / "checkpoints" / f"checkpoint_epoch_{epoch:04d}.pt", model, baseline, history, args)
        if bad_epochs >= args.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    save_checkpoint(run / "checkpoints" / "checkpoint_best.pt", model, baseline, history, args)
    save_checkpoint(run / "checkpoints" / "checkpoint_latest.pt", model, baseline, history, args)
    write_csv(run / "observables" / "training_history.csv", history)

    import yaml

    run_cfg = {
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
        "mode": "train_flow_detail_local_observable_regularized",
        "started_from_checkpoint": str(args.start_checkpoint),
        "baseline_run": str(args.baseline_run),
        "objective": {"NLL": 1.0, "observable_weights": weights},
        "random_seed": args.random_seed,
        "checkpoint_every": args.checkpoint_every_epochs,
        "resume": {"enabled": False, "checkpoint_path": None},
    }
    (run / "run_config.yaml").write_text(yaml.safe_dump(run_cfg, sort_keys=False), encoding="utf-8")
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
            "started_from_checkpoint": str(args.start_checkpoint),
        },
    )

    generated = evaluate_generated(model, phi16, phi8, kernel, baseline, args, run, "localreg_generated")
    stats = {"coarse_stats": baseline["coarse_stats"], "detail_stats": baseline["detail_stats"]}
    action = ActionSpec("phi4_nn", 1.0, 0.340301)
    patch = exact_detail_test(model, phi8[: args.patch_test_chains], kernel, stats, action, args)
    write_csv(run / "observables" / "acceptance_history.csv", patch["rows"])
    patch_rows, patch_g = per_config_rows(patch["phi"], action, "patch_test_L16")
    write_csv(run / "observables" / "patch_test_observables_per_config.csv", patch_rows)
    write_csv(run / "observables" / "patch_test_Gk_per_config.csv", patch_g)
    patch_comp = summarize_comparison(phi16[: len(patch["phi"])], patch["phi"], action)
    write_csv(run / "observables" / "patch_test_observable_comparison.csv", patch_comp)
    write_json(
        run / "debug" / "stability_diagnostics.json",
        {
            "generated": {k: v for k, v in generated.items() if k != "phi" and k != "comparison"},
            "patch_test": patch["summary"],
            "kernel": {
                "path": str(args.kernel_path),
                "sum": float(kernel.sum()),
                "eta_scale": ETA_SCALE,
                "kernel_coefficients_include_eta_scale": True,
            },
        },
    )
    key_rows = {row["observable"]: row for row in generated["comparison"]}
    patch_rows_lookup = {row["observable"]: row for row in patch_comp}
    lines = [
        "# Lambda 1.0 L8->L16 Local-Regularized Detail Flow",
        "",
        f"- started from: `{args.start_checkpoint}`",
        f"- kernel path: `{args.kernel_path}`",
        f"- kernel sum: `{float(kernel.sum()):.17g}`",
        f"- eta scale: `{ETA_SCALE:.17g}`",
        f"- epochs run: `{len(history)}`",
        f"- final validation NLL: `{history[-1]['validation_nll']:.6g}`",
        f"- best score: `{best_score:.6g}`",
        f"- generated nonfinite count: `{generated['nonfinite_count']}`",
        f"- generated max |z|: `{generated['max_abs_z']:.6g}`",
        f"- patch acceptance: `{patch['summary']['acceptance']:.6g}`",
        f"- DeltaS mean/std: `{patch['summary']['DeltaS_mean']:.6g}` / `{patch['summary']['DeltaS_std']:.6g}`",
        f"- log-accept mean/std: `{patch['summary']['log_accept_mean']:.6g}` / `{patch['summary']['log_accept_std']:.6g}`",
        "",
        "## Generated vs Native",
        "",
        "| observable | raw shift | raw std ratio | patch shift | patch std ratio |",
        "|---|---:|---:|---:|---:|",
    ]
    for obs in ["action_density", "phi2", "phi4", "local_kurtosis_ratio", "NN", "m2", "m4", "G_pmin_avg"]:
        r = key_rows[obs]
        p = patch_rows_lookup[obs]
        lines.append(f"| {obs} | {float(r['standardized_mean_shift']):.6g} | {float(r['std_ratio']):.6g} | {float(p['standardized_mean_shift']):.6g} | {float(p['std_ratio']):.6g} |")
    lines += [
        "",
        "## Recommendation",
        "",
        "Compare this against the diagonal-Gaussian baseline before deciding whether to continue local-regularized training or move to a genuinely autoregressive/coupling detail flow.",
    ]
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
