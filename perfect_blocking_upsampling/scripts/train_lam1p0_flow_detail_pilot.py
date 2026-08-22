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
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PKG = PROJECT_ROOT / "perfect_blocking_upsampling"
SCRIPT_DIR = PKG / "scripts"
sys.path.insert(0, str(PKG / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

from perfect_blocking_upsampling.actions import action_total  # noqa: E402
from perfect_blocking_upsampling.io import ActionSpec  # noqa: E402
from perfect_blocking_upsampling.observables import observables, second_moment_components  # noqa: E402


ETA_SCALE = 2.0**0.125
OBS_KEYS = [
    "action_density",
    "phi2",
    "phi4",
    "local_kurtosis_ratio",
    "NN",
    "2nn",
    "diag",
    "m2",
    "m4",
    "Binder_U4",
    "xi_over_L",
    "G_pmin_avg",
]


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")


def json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    raise TypeError(type(obj).__name__)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_phi(path: Path) -> np.ndarray:
    with np.load(path) as z:
        arr = z["phi"] if "phi" in z.files else z[z.files[0]]
    return np.asarray(arr, dtype=np.float32)


def load_kernel_matrix(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not bool(data.get("kernel_coefficients_include_eta_scale", False)):
        raise ValueError(f"kernel is not marked eta-included: {path}")
    eta_scale = float(data.get("eta_scale_numeric", ETA_SCALE))
    if "matrix" in data:
        mat = np.asarray(data["matrix"], dtype=np.float64)
    elif "base_matrix_before_eta_scale" in data:
        mat = eta_scale * np.asarray(data["base_matrix_before_eta_scale"], dtype=np.float64)
    else:
        raise ValueError(f"kernel JSON has no matrix field: {path}")
    total = float(mat.sum())
    # Nine-decimal tabulated kernels naturally carry O(1e-8) normalization
    # roundoff; retain a strict absolute check while accepting that precision.
    if not np.isclose(total, eta_scale, atol=1.0e-8, rtol=0.0):
        raise ValueError(f"kernel sum {total:.17g} != eta_scale {eta_scale:.17g}: {path}")
    return mat, data


def apply_kernel(phi: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    arr = np.asarray(phi, dtype=np.float64)
    out = np.zeros_like(arr)
    r = kernel.shape[0] // 2
    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            w = float(kernel[i, j])
            if w == 0.0:
                continue
            out += w * np.roll(np.roll(arr, i - r, axis=-2), j - r, axis=-1)
    return out.astype(np.float32)


def kernel_fft(kernel: np.ndarray, L: int) -> np.ndarray:
    w = np.zeros((L, L), dtype=np.float64)
    r = kernel.shape[0] // 2
    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            w[(i - r) % L, (j - r) % L] += kernel[i, j]
    return np.fft.fft2(w)


def inverse_kernel(psi: np.ndarray, kernel: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    kt = kernel_fft(kernel, int(psi.shape[-1]))
    abs_kt = np.abs(kt)
    phi = np.fft.ifft2(np.fft.fft2(psi.astype(np.float64), axes=(-2, -1)) / kt[None], axes=(-2, -1))
    return phi.real.astype(np.float32), {
        "min_abs_K": float(abs_kt.min()),
        "max_abs_K": float(abs_kt.max()),
        "max_invK": float(np.max(1.0 / np.maximum(abs_kt, 1.0e-300))),
        "condition_number": float(abs_kt.max() / max(abs_kt.min(), 1.0e-300)),
        "max_inverse_imag": float(np.max(np.abs(phi.imag))),
    }


def split_pairs(phi_fine: np.ndarray, kernel: np.ndarray) -> dict[str, np.ndarray]:
    psi = apply_kernel(phi_fine, kernel)
    coarse = psi[:, 0::2, 0::2].astype(np.float32)
    detail = np.stack([psi[:, 0::2, 1::2], psi[:, 1::2, 0::2], psi[:, 1::2, 1::2]], axis=1).astype(np.float32)
    return {"psi": psi, "coarse": coarse, "detail": detail}


def assemble_psi(coarse: np.ndarray, detail: np.ndarray) -> np.ndarray:
    n, lc, _ = coarse.shape
    psi = np.empty((n, 2 * lc, 2 * lc), dtype=np.float32)
    psi[:, 0::2, 0::2] = coarse
    psi[:, 0::2, 1::2] = detail[:, 0]
    psi[:, 1::2, 0::2] = detail[:, 1]
    psi[:, 1::2, 1::2] = detail[:, 2]
    return psi


class DetailGaussian(nn.Module):
    def __init__(self, coarse_dim: int, detail_dim: int, hidden: int, log_std_min: float, log_std_max: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(coarse_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2 * detail_dim),
        )
        self.detail_dim = detail_dim
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

    def params(self, coarse: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.net(coarse)
        mean, raw_log_std = raw[:, : self.detail_dim], raw[:, self.detail_dim :]
        log_std = self.log_std_min + (self.log_std_max - self.log_std_min) * torch.sigmoid(raw_log_std)
        return mean, log_std

    def nll(self, coarse: torch.Tensor, detail: torch.Tensor) -> torch.Tensor:
        mean, log_std = self.params(coarse)
        z = (detail - mean) * torch.exp(-log_std)
        return 0.5 * torch.sum(z * z + 2.0 * log_std + math.log(2.0 * math.pi), dim=1)

    def sample(self, coarse: torch.Tensor, generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self.params(coarse)
        eps = torch.randn(mean.shape, dtype=mean.dtype, device=mean.device, generator=generator)
        detail = mean + torch.exp(log_std) * eps
        logq = -0.5 * torch.sum(eps * eps + 2.0 * log_std + math.log(2.0 * math.pi), dim=1)
        return detail, logq, eps

    def log_prob(self, coarse: torch.Tensor, detail: torch.Tensor) -> torch.Tensor:
        return -self.nll(coarse, detail)


def standardize(train: np.ndarray, val: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std = np.maximum(std, 1.0e-6)
    return (
        ((train - mean) / std).astype(np.float32),
        ((val - mean) / std).astype(np.float32),
        ((test - mean) / std).astype(np.float32),
        {"mean": mean.astype(np.float32), "std": std.astype(np.float32)},
    )


def unstandardize(x: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    return (x * stats["std"] + stats["mean"]).astype(np.float32)


def train_model(args: argparse.Namespace, run: Path, coarse: np.ndarray, detail: np.ndarray) -> tuple[DetailGaussian, dict[str, Any]]:
    n = len(coarse)
    rng = np.random.default_rng(args.random_seed)
    idx = rng.permutation(n)
    n_train = min(args.train_count, int(0.8 * n))
    n_val = min(args.val_count, n - n_train)
    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    c_flat = coarse.reshape(n, -1)
    d_flat = detail.reshape(n, -1)
    c_train, c_val, _unused, c_stats = standardize(c_flat[train_idx], c_flat[val_idx], c_flat[val_idx])
    d_train, d_val, _unused2, d_stats = standardize(d_flat[train_idx], d_flat[val_idx], d_flat[val_idx])
    train_ds = TensorDataset(torch.from_numpy(c_train), torch.from_numpy(d_train))
    val_ds = TensorDataset(torch.from_numpy(c_val), torch.from_numpy(d_val))
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    device = torch.device(args.device)
    model = DetailGaussian(c_train.shape[1], d_train.shape[1], args.hidden_dim, args.log_std_min, args.log_std_max).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history = []
    best_val = float("inf")
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for cb, db in loader:
            cb = cb.to(device)
            db = db.to(device)
            opt.zero_grad(set_to_none=True)
            loss = model.nll(cb, db).mean()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            total += float(loss.detach().cpu()) * int(len(cb))
            count += int(len(cb))
        model.eval()
        vtotal = 0.0
        vcount = 0
        with torch.no_grad():
            for cb, db in val_loader:
                loss = model.nll(cb.to(device), db.to(device))
                vtotal += float(loss.sum().detach().cpu())
                vcount += int(len(cb))
        row = {"epoch": epoch, "train_nll": total / max(count, 1), "validation_nll": vtotal / max(vcount, 1)}
        history.append(row)
        if row["validation_nll"] < best_val:
            best_val = row["validation_nll"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(json.dumps(row), flush=True)
        if epoch % args.checkpoint_every_epochs == 0:
            save_checkpoint(run / "checkpoints" / f"checkpoint_epoch_{epoch:04d}.pt", model, args, c_stats, d_stats, train_idx, val_idx, history)
    if best_state is not None:
        model.load_state_dict(best_state)
    save_checkpoint(run / "checkpoints" / "checkpoint_latest.pt", model, args, c_stats, d_stats, train_idx, val_idx, history)
    save_checkpoint(run / "checkpoints" / "checkpoint_best.pt", model, args, c_stats, d_stats, train_idx, val_idx, history)
    write_csv(run / "observables" / "training_history.csv", history)
    return model, {"coarse_stats": c_stats, "detail_stats": d_stats, "train_idx": train_idx, "val_idx": val_idx, "history": history, "best_val_nll": best_val}


def save_checkpoint(path: Path, model: DetailGaussian, args: argparse.Namespace, c_stats: dict[str, Any], d_stats: dict[str, Any], train_idx: np.ndarray, val_idx: np.ndarray, history: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_class": "DetailGaussian",
            "coarse_stats": c_stats,
            "detail_stats": d_stats,
            "train_idx": train_idx,
            "val_idx": val_idx,
            "history": history,
            "config": vars(args),
            "rng_note": "training DataLoader shuffles from the explicit random_seed used to create train/validation split",
        },
        path,
    )


def sample_details(model: DetailGaussian, coarse: np.ndarray, stats: dict[str, Any], args: argparse.Namespace, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    device = torch.device(args.device)
    c = ((coarse.reshape(len(coarse), -1) - stats["coarse_stats"]["mean"]) / stats["coarse_stats"]["std"]).astype(np.float32)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    outs = []
    logqs = []
    zs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(c), args.batch_size):
            cb = torch.from_numpy(c[start : start + args.batch_size]).to(device)
            d, logq, z = model.sample(cb, gen)
            outs.append(d.detach().cpu().numpy())
            logqs.append(logq.detach().cpu().numpy())
            zs.append(z.detach().cpu().numpy())
    d_std = np.concatenate(outs, axis=0)
    detail = unstandardize(d_std, stats["detail_stats"]).reshape(len(coarse), 3, coarse.shape[1], coarse.shape[2])
    return detail.astype(np.float32), np.concatenate(logqs).astype(np.float64), np.concatenate(zs).astype(np.float32)


def log_prob_details(model: DetailGaussian, coarse: np.ndarray, detail: np.ndarray, stats: dict[str, Any], args: argparse.Namespace) -> np.ndarray:
    device = torch.device(args.device)
    c = ((coarse.reshape(len(coarse), -1) - stats["coarse_stats"]["mean"]) / stats["coarse_stats"]["std"]).astype(np.float32)
    d = ((detail.reshape(len(detail), -1) - stats["detail_stats"]["mean"]) / stats["detail_stats"]["std"]).astype(np.float32)
    vals = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(c), args.batch_size):
            cb = torch.from_numpy(c[start : start + args.batch_size]).to(device)
            db = torch.from_numpy(d[start : start + args.batch_size]).to(device)
            vals.append(model.log_prob(cb, db).detach().cpu().numpy())
    # Include constant Jacobian from detail standardization.
    log_jac = -float(np.sum(np.log(stats["detail_stats"]["std"])))
    return np.concatenate(vals).astype(np.float64) + log_jac


def per_config_rows(phi: np.ndarray, action: ActionSpec, label: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    arr = np.asarray(phi, dtype=np.float64)
    sc = second_moment_components(arr)
    action_density = action_total(arr, action) / (arr.shape[1] * arr.shape[2])
    m = arr.mean(axis=(1, 2))
    phi2 = np.mean(arr * arr, axis=(1, 2))
    phi4 = np.mean(arr**4, axis=(1, 2))
    nn = 0.5 * (np.mean(arr * np.roll(arr, -1, axis=1), axis=(1, 2)) + np.mean(arr * np.roll(arr, -1, axis=2), axis=(1, 2)))
    two = 0.5 * (np.mean(arr * np.roll(arr, -2, axis=1), axis=(1, 2)) + np.mean(arr * np.roll(arr, -2, axis=2), axis=(1, 2)))
    diag = np.mean(arr * np.roll(np.roll(arr, -1, axis=1), -1, axis=2), axis=(1, 2))
    rows = []
    grows = []
    for i in range(len(arr)):
        rows.append(
            {
                "config_index": i,
                "ensemble": label,
                "action_density": float(action_density[i]),
                "phi2": float(phi2[i]),
                "phi4": float(phi4[i]),
                "local_kurtosis_ratio": float(phi4[i] / max(phi2[i] * phi2[i], 1.0e-300)),
                "NN": float(nn[i]),
                "2nn": float(two[i]),
                "diag": float(diag[i]),
                "m": float(m[i]),
                "m2": float(m[i] * m[i]),
                "m4": float(m[i] ** 4),
                "Binder_U4_from_averages": float(1.0 - np.mean(m**4) / max(3.0 * np.mean(m * m) ** 2, 1.0e-300)),
                "xi_over_L": float(sc["xi_over_L"]),
            }
        )
        grows.append(
            {
                "config_index": i,
                "ensemble": label,
                "G_00": float(arr.shape[1] * arr.shape[2] * m[i] * m[i]),
                "G_10": float(sc["G_pmin_x_cfg"][i]),
                "G_01": float(sc["G_pmin_y_cfg"][i]),
                "G_pmin_avg": float(0.5 * (sc["G_pmin_x_cfg"][i] + sc["G_pmin_y_cfg"][i])),
            }
        )
    return rows, grows


def summarize_comparison(native: np.ndarray, generated: np.ndarray, action: ActionSpec) -> list[dict[str, Any]]:
    nrows, ng = per_config_rows(native, action, "native_L16")
    grows, gg = per_config_rows(generated, action, "generated_L16")
    lookup_native = {key: np.asarray([r[key] for r in nrows], dtype=float) for key in nrows[0] if key not in {"config_index", "ensemble"}}
    lookup_gen = {key: np.asarray([r[key] for r in grows], dtype=float) for key in grows[0] if key not in {"config_index", "ensemble"}}
    gnat = {key: np.asarray([r[key] for r in ng], dtype=float) for key in ng[0] if key not in {"config_index", "ensemble"}}
    ggen = {key: np.asarray([r[key] for r in gg], dtype=float) for key in gg[0] if key not in {"config_index", "ensemble"}}
    rows = []
    for key in ["action_density", "phi2", "phi4", "local_kurtosis_ratio", "NN", "2nn", "diag", "m2", "m4", "xi_over_L"]:
        rows.append(metric_row(key, lookup_native[key], lookup_gen[key]))
    for key in ["G_00", "G_10", "G_01", "G_pmin_avg"]:
        rows.append(metric_row(key, gnat[key], ggen[key]))
    return rows


def metric_row(name: str, native: np.ndarray, generated: np.ndarray) -> dict[str, Any]:
    native = native[np.isfinite(native)]
    generated = generated[np.isfinite(generated)]
    return {
        "observable": name,
        "native_mean": float(np.mean(native)) if native.size else float("nan"),
        "generated_mean": float(np.mean(generated)) if generated.size else float("nan"),
        "mean_difference_generated_minus_native": float(np.mean(generated) - np.mean(native)) if native.size and generated.size else float("nan"),
        "standardized_mean_shift": float((np.mean(generated) - np.mean(native)) / max(np.std(native, ddof=1), 1.0e-300)) if native.size > 1 and generated.size else float("nan"),
        "native_std": float(np.std(native, ddof=1)) if native.size > 1 else float("nan"),
        "generated_std": float(np.std(generated, ddof=1)) if generated.size > 1 else float("nan"),
        "std_ratio": float(np.std(generated, ddof=1) / max(np.std(native, ddof=1), 1.0e-300)) if native.size > 1 and generated.size > 1 else float("nan"),
    }


def exact_detail_test(model: DetailGaussian, coarse: np.ndarray, kernel: np.ndarray, stats: dict[str, Any], action: ActionSpec, args: argparse.Namespace) -> dict[str, Any]:
    n = min(args.patch_test_chains, len(coarse))
    coarse = coarse[:n]
    detail, logq, z = sample_details(model, coarse, stats, args, args.random_seed + 9000)
    psi = assemble_psi(coarse, detail)
    phi, _ = inverse_kernel(psi, kernel)
    current_s = action_total(phi, action)
    current_logq = logq
    rng = np.random.default_rng(args.random_seed + 9100)
    rows = []
    accepted_total = 0
    attempts_total = 0
    delta_s_vals = []
    log_acc_vals = []
    for sweep in range(1, args.patch_test_sweeps + 1):
        prop_detail, prop_logq, _ = sample_details(model, coarse, stats, args, args.random_seed + 9200 + sweep)
        prop_phi, _ = inverse_kernel(assemble_psi(coarse, prop_detail), kernel)
        prop_s = action_total(prop_phi, action)
        log_acc = -prop_s + current_s + current_logq - prop_logq
        u = np.log(rng.random(n))
        acc = u < np.minimum(log_acc, 0.0)
        if np.any(acc):
            detail[acc] = prop_detail[acc]
            phi[acc] = prop_phi[acc]
            current_s[acc] = prop_s[acc]
            current_logq[acc] = prop_logq[acc]
        accepted_total += int(np.sum(acc))
        attempts_total += n
        delta_s = prop_s - current_s
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
    return {
        "phi": phi.astype(np.float32),
        "rows": rows,
        "summary": {
            "attempts": attempts_total,
            "accepted": accepted_total,
            "acceptance": float(accepted_total / max(attempts_total, 1)),
            "DeltaS_mean": float(np.mean(np.concatenate(delta_s_vals))),
            "DeltaS_std": float(np.std(np.concatenate(delta_s_vals))),
            "DeltaS_p05": float(np.quantile(np.concatenate(delta_s_vals), 0.05)),
            "DeltaS_p95": float(np.quantile(np.concatenate(delta_s_vals), 0.95)),
            "log_accept_mean": float(np.mean(np.concatenate(log_acc_vals))),
            "log_accept_std": float(np.std(np.concatenate(log_acc_vals))),
        },
    }


def write_run_config(run: Path, args: argparse.Namespace, kernel_raw: dict[str, Any], kernel_sum: float) -> None:
    cfg = {
        "lambda": 1.0,
        "kappa_f": 0.340301,
        "kappa_c": 0.340301,
        "eta": 0.25,
        "eta_scale": "2^0.125",
        "eta_scale_numeric": ETA_SCALE,
        "block_factor": 2,
        "L_c": 8,
        "L_f": 16,
        "fine_config_source": str(args.fine_config_source),
        "coarse_config_source": str(args.coarse_config_source),
        "kernel_path": str(args.kernel_path),
        "kernel_name": kernel_raw.get("name"),
        "kernel_coefficients_include_eta_scale": True,
        "kernel_sum": kernel_sum,
        "mode": "train_flow_detail",
        "flow_model": "conditional_diagonal_gaussian_detail_affine",
        "random_seed": args.random_seed,
        "n_train": args.train_count,
        "n_validation": args.val_count,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "checkpoint_every": args.checkpoint_every_epochs,
        "measure_every": args.measure_every,
        "save_every": args.save_every,
        "resume": {"enabled": False, "checkpoint_path": None},
        "patch_test": {"chains": args.patch_test_chains, "sweeps": args.patch_test_sweeps, "proposal": "independence detail redraw from trained q(detail|coarse)"},
    }
    import yaml

    (run / "run_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--fine-config-source", type=Path, default=Path("data/configs_phi4_2d/lam1p0_kappac0p340301_L16/configs.npz"))
    ap.add_argument("--coarse-config-source", type=Path, default=Path("data/configs_phi4_2d/lam1p0_kappac0p340301_L8/configs.npz"))
    ap.add_argument("--kernel-path", type=Path, default=Path("perfect_blocking/perfect_blocking_lam1p0/kernels/selected_for_upscaling/best_7x7_phi2_nn_guarded_eta_included.json"))
    ap.add_argument("--random-seed", type=int, default=2026071601)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--hidden-dim", type=int, default=384)
    ap.add_argument("--lr", type=float, default=2.0e-3)
    ap.add_argument("--weight-decay", type=float, default=1.0e-5)
    ap.add_argument("--grad-clip", type=float, default=10.0)
    ap.add_argument("--log-std-min", type=float, default=-4.0)
    ap.add_argument("--log-std-max", type=float, default=1.5)
    ap.add_argument("--train-count", type=int, default=4000)
    ap.add_argument("--val-count", type=int, default=1000)
    ap.add_argument("--generated-count", type=int, default=512)
    ap.add_argument("--patch-test-chains", type=int, default=64)
    ap.add_argument("--patch-test-sweeps", type=int, default=100)
    ap.add_argument("--checkpoint-every-epochs", type=int, default=10)
    ap.add_argument("--measure-every", type=int, default=1)
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--log-every", type=int, default=5)
    args = ap.parse_args()

    run = args.run_dir
    for sub in ["logs", "checkpoints", "observables", "plots", "summaries", "debug"]:
        (run / sub).mkdir(parents=True, exist_ok=True)
    action = ActionSpec("phi4_nn", 1.0, 0.340301)
    kernel, kernel_raw = load_kernel_matrix(args.kernel_path)
    kernel_sum = float(kernel.sum())
    write_run_config(run, args, kernel_raw, kernel_sum)
    write_json(
        run / "submit_manifest.txt",
        {
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "command": " ".join(sys.argv),
            "git_commit": git_commit(),
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "kernel_path": str(args.kernel_path),
            "kernel_name": kernel_raw.get("name"),
            "kernel_coefficients_include_eta_scale": True,
            "kernel_sum": kernel_sum,
            "eta_scale_numeric": ETA_SCALE,
            "raw_config_policy": "input configs remain under data/configs_phi4_2d; run directory stores checkpoints and observables only",
        },
    )
    write_json(run / "status.json", {"status": "running", "current_epoch": 0, "latest_checkpoint": None})

    phi16 = load_phi(args.fine_config_source)
    phi8 = load_phi(args.coarse_config_source)
    pairs = split_pairs(phi16, kernel)
    write_json(
        run / "debug" / "training_pair_summary.json",
        {
            "fine_shape": list(phi16.shape),
            "coarse_source_shape": list(phi8.shape),
            "blocked_coarse_shape": list(pairs["coarse"].shape),
            "detail_shape": list(pairs["detail"].shape),
            "coarse_mean": float(np.mean(pairs["coarse"])),
            "detail_mean": float(np.mean(pairs["detail"])),
            "detail_std": float(np.std(pairs["detail"])),
        },
    )

    model, train_info = train_model(args, run, pairs["coarse"], pairs["detail"])
    stats = {"coarse_stats": train_info["coarse_stats"], "detail_stats": train_info["detail_stats"]}
    n_gen = min(args.generated_count, len(phi8), len(phi16))
    gen_detail, gen_logq, z = sample_details(model, phi8[:n_gen], stats, args, args.random_seed + 7000)
    psi_gen = assemble_psi(phi8[:n_gen], gen_detail)
    phi_gen, inv_info = inverse_kernel(psi_gen, kernel)
    reb = apply_kernel(phi_gen, kernel)[:, 0::2, 0::2] - phi8[:n_gen]
    native_rows, native_g = per_config_rows(phi16[:n_gen], action, "native_L16")
    gen_rows, gen_g = per_config_rows(phi_gen, action, "generated_L16_from_native_L8")
    write_csv(run / "observables" / "all_observables_per_config.csv", native_rows + gen_rows)
    write_csv(run / "observables" / "Gk_per_config.csv", native_g + gen_g)
    comparison = summarize_comparison(phi16[:n_gen], phi_gen, action)
    write_csv(run / "observables" / "first_observable_comparison.csv", comparison)

    patch = exact_detail_test(model, phi8[: args.patch_test_chains], kernel, stats, action, args)
    write_csv(run / "observables" / "acceptance_history.csv", patch["rows"])
    patch_rows, patch_g = per_config_rows(patch["phi"], action, "patch_test_L16")
    write_csv(run / "observables" / "patch_test_observables_per_config.csv", patch_rows)
    write_csv(run / "observables" / "patch_test_Gk_per_config.csv", patch_g)
    patch_comp = summarize_comparison(phi16[: len(patch["phi"])], patch["phi"], action)
    write_csv(run / "observables" / "patch_test_observable_comparison.csv", patch_comp)

    stability = {
        "nonfinite_count_generated_phi": int(np.sum(~np.isfinite(phi_gen))),
        "nonfinite_count_generated_detail": int(np.sum(~np.isfinite(gen_detail))),
        "max_abs_z": float(np.max(np.abs(z))),
        "generated_logq_mean": float(np.mean(gen_logq)),
        "generated_logq_std": float(np.std(gen_logq)),
        "inverse_kernel": inv_info,
        "reblocking_max_abs_error": float(np.max(np.abs(reb))),
        "reblocking_rms_error": float(np.sqrt(np.mean(reb.astype(np.float64) ** 2))),
        "patch_test": patch["summary"],
    }
    write_json(run / "debug" / "stability_diagnostics.json", stability)
    key_rows = {row["observable"]: row for row in comparison}
    acc = patch["summary"]
    report_lines = [
        "# Lambda 1.0 L8->L16 Flow Detail Pilot",
        "",
        f"- kernel path: `{args.kernel_path}`",
        f"- kernel name: `{kernel_raw.get('name')}`",
        f"- kernel sum: `{kernel_sum:.17g}`",
        f"- eta scale: `{ETA_SCALE:.17g}`",
        f"- checkpoint latest: `{run / 'checkpoints' / 'checkpoint_latest.pt'}`",
        f"- best validation NLL: `{train_info['best_val_nll']:.6g}`",
        f"- nonfinite generated phi count: `{stability['nonfinite_count_generated_phi']}`",
        f"- max |z|: `{stability['max_abs_z']:.6g}`",
        f"- reblocking max abs error: `{stability['reblocking_max_abs_error']:.6g}`",
        f"- exact detail-test acceptance: `{acc['acceptance']:.6g}`",
        "",
        "## First Generated-vs-Native Comparison",
        "",
        "| observable | native mean | generated mean | standardized shift | std ratio |",
        "|---|---:|---:|---:|---:|",
    ]
    for obs in ["action_density", "phi2", "phi4", "local_kurtosis_ratio", "NN", "m2", "m4", "G_pmin_avg"]:
        row = key_rows[obs]
        report_lines.append(f"| {obs} | {row['native_mean']:.6g} | {row['generated_mean']:.6g} | {row['standardized_mean_shift']:.6g} | {row['std_ratio']:.6g} |")
    report_lines += [
        "",
        "## Recommendation",
        "",
        "This is a conservative diagonal-Gaussian detail-flow pilot. Continue training or move to a richer autoregressive/coupling flow only if the exact detail test has non-pathological acceptance and local observables move toward native L16 without damaging the low-momentum sector.",
    ]
    (run / "summaries" / "run_summary.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    write_json(
        run / "status.json",
        {
            "status": "completed",
            "current_epoch": args.epochs,
            "latest_checkpoint": str(run / "checkpoints" / "checkpoint_latest.pt"),
            "best_checkpoint": str(run / "checkpoints" / "checkpoint_best.pt"),
            "summary": str(run / "summaries" / "run_summary.md"),
        },
    )
    print(json.dumps({"status": "completed", "run_dir": str(run), "best_val_nll": train_info["best_val_nll"], "acceptance": acc["acceptance"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
