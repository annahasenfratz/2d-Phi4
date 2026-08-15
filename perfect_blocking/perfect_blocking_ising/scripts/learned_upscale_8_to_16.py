#!/usr/bin/env python3
"""Learn a conditional 8x8 -> 16x16 Ising upscaler with a blocking-consistency penalty."""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "perfect_blocking_ising" / "outputs_learned_upscale_8_to_16"
SUMMARY_JSON = OUT_DIR / "learned_upscale_summary.json"
REPORT_MD = OUT_DIR / "learned_upscale_report.md"
OBS_CSV = OUT_DIR / "learned_upscale_observables.csv"
HIST_CSV = OUT_DIR / "learned_upscale_training_history.csv"
PLOTS_PDF = OUT_DIR / "learned_upscale_plots.pdf"

DATA_DIR = ROOT / "perfect_blocking_ising" / "outputs"
TRUE8_REF = DATA_DIR / "critical_ising_L8.npy"
TRUE16_REF = DATA_DIR / "critical_ising_L16.npy"

SEED = 20240616
N_TOTAL = 500
N_TRAIN = 400
N_VAL = 100
N_BLOCK_REPLICA = 3
N_BOOT = 300
N_GENERATE_VAL = 100
TRAIN_EPOCHS = 35
BATCH_SIZE = 32
LR = 2e-3
CHANNELS = 64
N_BLOCK_RES = 4

KERNEL = {
    "alpha": 1.75068663213513,
    "w00": 0.2282658690109113,
    "w01": 0.19003036092905146,
    "w11": 0.0029031718182207533,
}

BETA = 0.5 * math.log(1.0 + math.sqrt(2.0))
LAMBDA_BLOCK_GRID = [0.0, 0.1, 0.3, 1.0, 3.0]


def load_opt_module():
    path = ROOT / "perfect_blocking_ising" / "scripts" / "optimize_perfect_blocking.py"
    spec = importlib.util.spec_from_file_location("perfect_blocking_optimize", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import helper module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def compact_stats(stats: dict) -> dict:
    return {k: v for k, v in stats.items() if k != "per_config"}


def summarize(mod, spins: np.ndarray) -> dict:
    return compact_stats(mod.summarize_observables(spins, bootstrap=True, seed=SEED, n_boot=N_BOOT))


def load_pairs(mod) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not TRUE8_REF.exists() or not TRUE16_REF.exists():
        raise FileNotFoundError("critical_ising_L8.npy or critical_ising_L16.npy missing")
    true8 = np.load(TRUE8_REF).astype(np.float32)[:N_TOTAL]
    true16 = np.load(TRUE16_REF).astype(np.float32)[:N_TOTAL]
    rng = np.random.default_rng(SEED + 11)
    uniforms = rng.random((1, len(true16), 8, 8), dtype=np.float32)[0]
    coarse8 = mod.block_centered_3x3(true16, KERNEL["alpha"], KERNEL["w00"], KERNEL["w01"], KERNEL["w11"], uniforms)
    coarse8 = np.asarray(coarse8, dtype=np.float32)
    return true8, true16, coarse8, uniforms


def parity_channel(L: int, device=None) -> torch.Tensor:
    ii, jj = torch.meshgrid(torch.arange(L, device=device), torch.arange(L, device=device), indexing="ij")
    parity = ((ii + jj) % 2).float() * 2.0 - 1.0
    return parity[None, None, ...]


def repeat_coarse_to_fine(coarse: torch.Tensor) -> torch.Tensor:
    return coarse.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)


def block_expectation_from_mean_spin(m: torch.Tensor, alpha: float, kernel: dict) -> torch.Tensor:
    center = m[:, :, 0::2, 0::2]
    up = torch.roll(m, shifts=1, dims=2)[:, :, 0::2, 0::2]
    down = torch.roll(m, shifts=-1, dims=2)[:, :, 0::2, 0::2]
    left = torch.roll(m, shifts=1, dims=3)[:, :, 0::2, 0::2]
    right = torch.roll(m, shifts=-1, dims=3)[:, :, 0::2, 0::2]
    ul = torch.roll(torch.roll(m, shifts=1, dims=2), shifts=1, dims=3)[:, :, 0::2, 0::2]
    ur = torch.roll(torch.roll(m, shifts=1, dims=2), shifts=-1, dims=3)[:, :, 0::2, 0::2]
    dl = torch.roll(torch.roll(m, shifts=-1, dims=2), shifts=1, dims=3)[:, :, 0::2, 0::2]
    dr = torch.roll(torch.roll(m, shifts=-1, dims=2), shifts=-1, dims=3)[:, :, 0::2, 0::2]
    p = kernel["w00"] * center + kernel["w01"] * (up + down + left + right) + kernel["w11"] * (ul + ur + dl + dr)
    return torch.tanh(2.0 * alpha * p)


def generate_noisy_replication(coarse: np.ndarray, eps: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    fine = np.repeat(np.repeat(coarse, 2, axis=1), 2, axis=2).astype(np.float32)
    flips = rng.random(fine.shape, dtype=np.float32) < eps
    return np.where(flips, -fine, fine).astype(np.float32)


def compute_p_field_np(fine: np.ndarray, kernel: dict) -> np.ndarray:
    s = np.asarray(fine, dtype=np.float32)
    squeeze = False
    if s.ndim == 2:
        s = s[None, ...]
        squeeze = True
    center = s[:, 0::2, 0::2]
    up = np.roll(s, 1, axis=1)[:, 0::2, 0::2]
    down = np.roll(s, -1, axis=1)[:, 0::2, 0::2]
    left = np.roll(s, 1, axis=2)[:, 0::2, 0::2]
    right = np.roll(s, -1, axis=2)[:, 0::2, 0::2]
    ul = np.roll(np.roll(s, 1, axis=1), 1, axis=2)[:, 0::2, 0::2]
    ur = np.roll(np.roll(s, 1, axis=1), -1, axis=2)[:, 0::2, 0::2]
    dl = np.roll(np.roll(s, -1, axis=1), 1, axis=2)[:, 0::2, 0::2]
    dr = np.roll(np.roll(s, -1, axis=1), -1, axis=2)[:, 0::2, 0::2]
    p = kernel["w00"] * center + kernel["w01"] * (up + down + left + right) + kernel["w11"] * (ul + ur + dl + dr)
    p = p.astype(np.float32)
    if squeeze:
        return p[0]
    return p


def block_centered_3x3_np(fine16: np.ndarray, alpha: float, kernel: dict, uniforms: np.ndarray) -> np.ndarray:
    p = compute_p_field_np(fine16, kernel)
    logits = np.clip(4.0 * alpha * p, -50.0, 50.0)
    prob_plus = 1.0 / (1.0 + np.exp(-logits))
    if uniforms.ndim == 3:
        u = uniforms
    elif uniforms.ndim == 4:
        u = uniforms[0]
    else:
        raise ValueError(uniforms.shape)
    t = np.where(u < prob_plus, 1.0, -1.0)
    return t.astype(np.float32)


def block_many_np(fine16: np.ndarray, alpha: float, kernel: dict, seed: int, replicas: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    uniforms = rng.random((replicas, len(fine16), 8, 8), dtype=np.float32)
    blocks = []
    for rep in range(replicas):
        blocks.append(block_centered_3x3_np(fine16, alpha, kernel, uniforms[rep]))
    return np.asarray(blocks, dtype=np.float32).reshape((-1, 8, 8))


def block_16_to_8_np(fine16: np.ndarray, seed: int, replicas: int = 1) -> np.ndarray:
    return block_many_np(fine16, KERNEL["alpha"], KERNEL, seed, replicas=replicas)


def pairwise_metrics(input8: np.ndarray, reblocked8: np.ndarray, generated16: np.ndarray) -> dict:
    input8 = np.asarray(input8, dtype=np.float32)
    reblocked8 = np.asarray(reblocked8, dtype=np.float32)
    generated16 = np.asarray(generated16, dtype=np.float32)
    if reblocked8.shape[0] != input8.shape[0]:
        rep = reblocked8.shape[0] // input8.shape[0]
        input_rep = np.repeat(input8, rep, axis=0)
    else:
        input_rep = input8
    overlap = np.mean(input_rep * reblocked8, axis=(1, 2))
    agreement = np.mean(input_rep == reblocked8, axis=(1, 2))
    in_sign = np.sign(input8.mean(axis=(1, 2)))
    in_sign[in_sign == 0] = 1.0
    gen_sign = np.sign(generated16.mean(axis=(1, 2)))
    gen_sign[gen_sign == 0] = 1.0
    sign_agree = np.mean(in_sign == gen_sign)
    corr = np.corrcoef(input_rep.reshape(len(input_rep), -1).mean(axis=1), reblocked8.reshape(len(reblocked8), -1).mean(axis=1))[0, 1]
    return {
        "overlap_mean": float(np.mean(overlap)),
        "overlap_err": float(np.std(overlap, ddof=1) / math.sqrt(len(overlap))) if len(overlap) > 1 else 0.0,
        "agreement_mean": float(np.mean(agreement)),
        "agreement_err": float(np.std(agreement, ddof=1) / math.sqrt(len(agreement))) if len(agreement) > 1 else 0.0,
        "sign_agreement": float(sign_agree),
        "coarse_field_corr": float(corr) if np.isfinite(corr) else 0.0,
    }


def compare_stats(mod, target: np.ndarray, proposal: np.ndarray) -> tuple[list[dict], dict]:
    keys = ["nn", "diag", "2nn", "nn2", "diag2", "2nn2", "abs_m", "m2"]
    true_stats = mod.summarize_observables(target, bootstrap=True, seed=SEED, n_boot=N_BOOT)
    prop_stats = mod.summarize_observables(proposal, bootstrap=True, seed=SEED + 1, n_boot=N_BOOT)
    rows = []
    abs_z = []
    for key in keys:
        if key in true_stats["means"]:
            t_mean = float(true_stats["means"][key])
            t_err = float(true_stats["errs"][key])
            p_mean = float(prop_stats["means"][key])
            p_err = float(prop_stats["errs"][key])
        else:
            t_mean = float(true_stats["extra"][key]["mean"])
            t_err = float(true_stats["extra"][key]["err"])
            p_mean = float(prop_stats["extra"][key]["mean"])
            p_err = float(prop_stats["extra"][key]["err"])
        sigma = math.sqrt(t_err**2 + p_err**2) if (t_err or p_err) else 1.0
        z = (p_mean - t_mean) / sigma if sigma else 0.0
        rows.append(
            {
                "observable": key,
                "true_mean": t_mean,
                "true_err": t_err,
                "proposal_mean": p_mean,
                "proposal_err": p_err,
                "delta": p_mean - t_mean,
                "rel_delta": (p_mean - t_mean) / t_mean if t_mean else 0.0,
                "z": z,
            }
        )
        abs_z.append(abs(z))
    stats = {
        "mean_abs_z": float(np.mean(abs_z)),
        "max_abs_z": float(np.max(abs_z)),
        "true": compact_stats(true_stats),
        "proposal": compact_stats(prop_stats),
        "rows": rows,
    }
    return rows, stats


class UpscaleCNN(nn.Module):
    def __init__(self, in_ch: int = 3, channels: int = 64, n_blocks: int = 4):
        super().__init__()
        self.register_buffer("skip_scale", torch.tensor(2.5, dtype=torch.float32))
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        blocks = []
        for _ in range(n_blocks):
            blocks.append(ResidualBlock(channels))
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Conv2d(channels, 1, 1)

    def forward(self, x):
        h = self.stem(x)
        h = self.blocks(h)
        return self.head(h) + self.skip_scale * x[:, :1]


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        h = self.act(self.conv1(x))
        h = self.conv2(h)
        return self.act(x + h)


@dataclass
class TrainRun:
    lambda_block: float
    history: list[dict]
    model_state: dict
    val_score: float
    val_metrics: dict
    eval_summary: dict
    proposal_name: str
    best_epoch: int


def make_features(coarse8: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
    coarse16 = repeat_coarse_to_fine(coarse8[:, None, ...])
    parity = parity_channel(16, device=coarse8.device).repeat(coarse8.shape[0], 1, 1, 1)
    if noise is None:
        noise = torch.randn((coarse8.shape[0], 1, 16, 16), device=coarse8.device)
    return torch.cat([coarse16, parity, noise], dim=1)


def train_one_lambda(mod, train8, train16, val8, val16, lambda_block: float, device: torch.device) -> TrainRun:
    model = UpscaleCNN(in_ch=3, channels=CHANNELS, n_blocks=N_BLOCK_RES).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    history = []
    coarse_train = torch.from_numpy(train8).to(device)
    fine_train = torch.from_numpy(train16).to(device)
    coarse_val = torch.from_numpy(val8).to(device)
    fine_val = torch.from_numpy(val16).to(device)
    parity_16 = parity_channel(16, device=device)

    best_state = None
    best_val = float("inf")
    best_eval_summary = None
    best_epoch = 0

    for epoch in range(1, TRAIN_EPOCHS + 1):
        model.train()
        perm = torch.randperm(len(coarse_train), device=device)
        batch_rows = []
        for start in range(0, len(perm), BATCH_SIZE):
            idx = perm[start : start + BATCH_SIZE]
            c = coarse_train[idx]
            f = fine_train[idx]
            noise = torch.randn((len(idx), 1, 16, 16), device=device)
            x = torch.cat([repeat_coarse_to_fine(c[:, None, ...]), parity_16.repeat(len(idx), 1, 1, 1), noise], dim=1)
            logits = model(x)
            target = ((f[:, None, ...] + 1.0) * 0.5).clamp(0.0, 1.0)
            bce = F.binary_cross_entropy_with_logits(logits, target)
            mf = torch.tanh(logits / 2.0)
            block_expect = block_expectation_from_mean_spin(mf, KERNEL["alpha"], KERNEL)
            block_loss = torch.mean((block_expect - c[:, None, :, :]) ** 2)
            total = bce + lambda_block * block_loss
            opt.zero_grad(set_to_none=True)
            total.backward()
            opt.step()
            batch_rows.append(
                {
                    "lambda_block": lambda_block,
                    "epoch": epoch,
                    "split": "train",
                    "bce": float(bce.detach().cpu()),
                    "block_loss": float(block_loss.detach().cpu()),
                    "total_loss": float(total.detach().cpu()),
                }
            )

        model.eval()
        with torch.no_grad():
            noise = torch.randn((len(coarse_val), 1, 16, 16), device=device)
            x = torch.cat([repeat_coarse_to_fine(coarse_val[:, None, ...]), parity_16.repeat(len(coarse_val), 1, 1, 1), noise], dim=1)
            logits = model(x)
            target = ((fine_val[:, None, ...] + 1.0) * 0.5).clamp(0.0, 1.0)
            bce = F.binary_cross_entropy_with_logits(logits, target)
            mf = torch.tanh(logits / 2.0)
            block_expect = block_expectation_from_mean_spin(mf, KERNEL["alpha"], KERNEL)
            block_loss = torch.mean((block_expect - coarse_val[:, None, :, :]) ** 2)
            total = bce + lambda_block * block_loss
            val_loss = float(total.detach().cpu())

            # quick sample-based metrics on the validation set
            sampled = torch.bernoulli(torch.sigmoid(logits)).mul(2.0).sub(1.0).cpu().numpy().astype(np.float32)[:, 0]
            reblocked = block_16_to_8_np(sampled, seed=SEED + int(lambda_block * 1000) + epoch, replicas=N_BLOCK_REPLICA)
            _, gen_stats = compare_stats(mod, val16, sampled)
            _, reb_stats = compare_stats(mod, val8, reblocked)
            pair = pairwise_metrics(val8, reblocked, sampled)
            eval_summary = {
                "generated16": gen_stats,
                "reblocked8_vs_true8": reb_stats,
                "pairwise": pair,
                "score": gen_stats["mean_abs_z"] + reb_stats["mean_abs_z"] + (1.0 - pair["overlap_mean"]) + (1.0 - pair["agreement_mean"]),
            }
            history.extend(batch_rows)
            history.append(
                {
                    "lambda_block": lambda_block,
                    "epoch": epoch,
                    "split": "val",
                    "bce": float(bce.detach().cpu()),
                    "block_loss": float(block_loss.detach().cpu()),
                    "total_loss": val_loss,
                    "gen_mean_abs_z": gen_stats["mean_abs_z"],
                    "reb_mean_abs_z": reb_stats["mean_abs_z"],
                    "overlap_mean": pair["overlap_mean"],
                    "agreement_mean": pair["agreement_mean"],
                    "sign_agreement": pair["sign_agreement"],
                    "score": eval_summary["score"],
                }
            )
            if eval_summary["score"] < best_val:
                best_val = eval_summary["score"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_eval_summary = eval_summary
                best_epoch = epoch

    if best_state is not None:
        model.load_state_dict(best_state)

    return TrainRun(
        lambda_block=lambda_block,
        history=history,
        model_state=best_state if best_state is not None else {k: v.cpu() for k, v in model.state_dict().items()},
        val_score=best_val,
        val_metrics=eval_summary,
        eval_summary=best_eval_summary if best_eval_summary is not None else eval_summary,
        proposal_name=f"conditional_cnn_lambda{lambda_block:g}",
        best_epoch=best_epoch,
    )


def generate_from_model(model: nn.Module, coarse8: np.ndarray, device: torch.device, seed: int) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        torch.manual_seed(seed)
        coarse = torch.from_numpy(coarse8).to(device)
        noise = torch.randn((len(coarse), 1, 16, 16), device=device)
        x = make_features(coarse, noise)
        logits = model(x)
        sample = torch.bernoulli(torch.sigmoid(logits)).mul(2.0).sub(1.0)
        return sample.cpu().numpy().astype(np.float32)[:, 0]


def evaluate_proposal(mod, proposal_name: str, generated16: np.ndarray, input8: np.ndarray, true16: np.ndarray, true8: np.ndarray) -> dict:
    gen_rows, gen_stats = compare_stats(mod, true16, generated16)
    reblocked = block_16_to_8_np(generated16, seed=SEED + 333, replicas=N_BLOCK_REPLICA)
    reb_rows, reb_stats = compare_stats(mod, true8, reblocked)
    pair = pairwise_metrics(input8, reblocked, generated16)
    return {
        "proposal": proposal_name,
        "generated16": {"rows": gen_rows, **gen_stats},
        "reblocked8_vs_true8": {"rows": reb_rows, **reb_stats},
        "pairwise": pair,
        "score": gen_stats["mean_abs_z"] + reb_stats["mean_abs_z"] + (1.0 - pair["overlap_mean"]) + (1.0 - pair["agreement_mean"]),
        "example_generated16": generated16[0].tolist(),
        "example_reblocked8": reblocked[0].tolist(),
    }


def write_csv(rows: list[dict], path: Path) -> None:
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def make_plots(true8: np.ndarray, true16: np.ndarray, input8: np.ndarray, best_method: dict, baseline_methods: list[dict], history_rows: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(3, 2)

    ax = fig.add_subplot(gs[0, 0])
    labels = [m["proposal"] for m in baseline_methods] + [best_method["proposal"]]
    scores = [m["score"] for m in baseline_methods] + [best_method["score"]]
    ax.bar(np.arange(len(labels)), scores, color="#4c78a8")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("score")
    ax.set_title("Proposal comparison")

    ax = fig.add_subplot(gs[0, 1])
    if history_rows:
        val_rows = [r for r in history_rows if r["split"] == "val"]
        ax.plot([r["epoch"] + 0.01 * r["lambda_block"] for r in val_rows], [r["score"] for r in val_rows], lw=1.0, marker="o", ms=2)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation score")
    ax.set_title("Training trace")

    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(input8[0], cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_title("Input coarse L=8")
    ax.axis("off")

    ax = fig.add_subplot(gs[1, 1])
    ax.imshow(np.asarray(best_method["example_generated16"], dtype=np.float32), cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_title("Best generated L=16")
    ax.axis("off")

    ax = fig.add_subplot(gs[2, 0])
    ax.imshow(np.asarray(best_method["example_reblocked8"], dtype=np.float32), cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_title("Best reblocked L=8")
    ax.axis("off")

    ax = fig.add_subplot(gs[2, 1])
    ax.imshow(true16[0], cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_title("True L=16 reference")
    ax.axis("off")

    fig.tight_layout()
    with PdfPages(PLOTS_PDF) as pdf:
        pdf.savefig(fig)
        plt.close(fig)


def main() -> int:
    os.environ.setdefault("MPLCONFIGDIR", str(OUT_DIR / "_mplcache"))
    (OUT_DIR / "_mplcache").mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    mod = load_opt_module()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

    try:
        true8, true16, coarse8, _ = load_pairs(mod)
        idx = np.arange(N_TOTAL)
        rng = np.random.default_rng(SEED)
        rng.shuffle(idx)
        train_idx = idx[:N_TRAIN]
        val_idx = idx[N_TRAIN : N_TRAIN + N_VAL]

        train8 = coarse8[train_idx]
        train16 = true16[train_idx]
        val8 = coarse8[val_idx]
        val16 = true16[val_idx]

        # Baselines on the same validation split.
        noisy_val = generate_noisy_replication(val8, 0.05, SEED + 5)
        noisy_eval = evaluate_proposal(mod, "noisy_replication_eps0.05", noisy_val, val8, val16, true8[val_idx])
        relax_val = noisy_val.copy()
        # Use the best proposal from the prior test directly on the validation split.
        # This is proposal-quality only and intentionally not exact sampling.
        relax_val = []
        for coarse in val8:
            init = generate_noisy_replication(coarse[None, ...], 0.05, SEED + 7)[0]
            relaxed, _ = local_relaxation(mod, coarse, init, lambda_block=3.0, n_sweeps=10, seed=SEED + 9)
            relax_val.append(relaxed)
        relax_val = np.asarray(relax_val, dtype=np.float32)
        relax_eval = evaluate_proposal(mod, "relaxation_eps0.05_lam3.0_sw10", relax_val, val8, val16, true8[val_idx])
        baseline_methods = [noisy_eval, relax_eval]

        device = torch.device("cpu")
        train_runs = []
        history_rows = []
        for lam in LAMBDA_BLOCK_GRID:
            run = train_one_lambda(mod, train8, train16, val8, val16, lam, device)
            train_runs.append(run)
            history_rows.extend(run.history)

        # Select the best run by the validation proposal score captured at training time.
        best_run = min(train_runs, key=lambda r: r.val_score)
        model = UpscaleCNN(in_ch=3, channels=CHANNELS, n_blocks=N_BLOCK_RES).to(device)
        model.load_state_dict(best_run.model_state)

        # Final evaluation on the validation set with the selected learned model.
        learned_val = generate_from_model(model, val8, device=device, seed=SEED + 123)
        learned_eval = evaluate_proposal(mod, best_run.proposal_name, learned_val, val8, val16, true8[val_idx])

        # Also compute a few condition-preserving diagnostics on the learned model with repeated samples.
        coarse_rep = np.repeat(val8, 1, axis=0)
        reblocked = block_16_to_8_np(learned_val, seed=SEED + 333, replicas=N_BLOCK_REPLICA)
        pair = pairwise_metrics(coarse_rep, reblocked, learned_val)

        # Aggregate observables tables and training history.
        obs_rows = []
        for method in baseline_methods + [learned_eval]:
            for row in method["generated16"]["rows"]:
                obs_rows.append(
                    {
                        "proposal": method["proposal"],
                        "comparison": "generated16_vs_true16",
                        **row,
                    }
                )
            for row in method["reblocked8_vs_true8"]["rows"]:
                obs_rows.append(
                    {
                        "proposal": method["proposal"],
                        "comparison": "reblocked8_vs_true8",
                        **row,
                    }
                )

        write_csv(obs_rows, OBS_CSV)
        write_csv(history_rows, HIST_CSV)
        make_plots(true8[val_idx], val16, val8, learned_eval, baseline_methods, history_rows)

        summary = {
            "status": "ok",
            "beta": BETA,
            "kernel": KERNEL,
            "references": {
                "true8": {"path": str(TRUE8_REF), "shape": list(true8.shape)},
                "true16": {"path": str(TRUE16_REF), "shape": list(true16.shape)},
            },
            "pair_generation": {
                "n_total": N_TOTAL,
                "train_n": N_TRAIN,
                "val_n": N_VAL,
                "blocking_kernel": KERNEL,
                "pairing_rule": "t_c = stochastic blocking of true s_f using the provisional kernel",
            },
            "baselines": baseline_methods,
            "learned": {
                "proposal": learned_eval["proposal"],
                "score": learned_eval["score"],
                "generated16": learned_eval["generated16"],
                "reblocked8_vs_true8": learned_eval["reblocked8_vs_true8"],
                "pairwise": learned_eval["pairwise"],
                "example_generated16": learned_eval["example_generated16"],
                "example_reblocked8": learned_eval["example_reblocked8"],
                "train_choice": {
                    "lambda_block": best_run.lambda_block,
                    "val_score": best_run.val_score,
                },
            },
            "train_runs": [
                {
                    "lambda_block": r.lambda_block,
                    "val_score": r.val_score,
                    "proposal_name": r.proposal_name,
                    "history_rows": len(r.history),
                }
                for r in train_runs
            ],
            "training": {
                "epochs": TRAIN_EPOCHS,
                "batch_size": BATCH_SIZE,
                "learning_rate": LR,
                "channels": CHANNELS,
                "n_blocks": N_BLOCK_RES,
            },
            "decision": {
                "exact_sampling": False,
                "accept_reject": False,
                "proposal_quality_only": True,
                "notes": [
                    "This is a proposal-quality and conditionality test, not exact sampling.",
                    "No A/R correction is used yet.",
                    "The learned model has an explicit factorized q(s_f|t_c), but the acceptance step is intentionally deferred.",
                ],
            },
        }

        SUMMARY_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

        report = [
            "# Learned conditional 8 -> 16 Ising upscaler",
            "",
            f"beta_c = {BETA:.15f}",
            "",
            "This is not exact sampling.",
            "No A/R correction is used yet.",
            "This only tests whether the learned conditional model preserves the coarse input under reblocking while matching the fine ensemble better than the hand-built baselines.",
            "",
            "## Selection",
            f"- chosen lambda_block = {best_run.lambda_block}",
            f"- validation score = {best_run.val_score:.4f}",
            f"- learned generated16 mean abs z = {learned_eval['generated16']['mean_abs_z']:.4f}",
            f"- learned reblocked8 mean abs z = {learned_eval['reblocked8_vs_true8']['mean_abs_z']:.4f}",
            f"- learned overlap mean = {learned_eval['pairwise']['overlap_mean']:.4f}",
            f"- learned exact site agreement = {learned_eval['pairwise']['agreement_mean']:.4f}",
            f"- learned sign agreement = {learned_eval['pairwise']['sign_agreement']:.4f}",
            "",
            "## Baselines",
        ]
        for b in baseline_methods:
            report.append(
                f"- {b['proposal']}: score={b['score']:.4f}, gen16 mean abs z={b['generated16']['mean_abs_z']:.4f}, "
                f"reblocked8 mean abs z={b['reblocked8_vs_true8']['mean_abs_z']:.4f}, overlap={b['pairwise']['overlap_mean']:.4f}, "
                f"agreement={b['pairwise']['agreement_mean']:.4f}, sign agreement={b['pairwise']['sign_agreement']:.4f}"
            )
        report.append("")
        report.append("## Learned model")
        report.append(
            "The model is a conditional CNN with replicated coarse input, parity channel, and stochastic Bernoulli output spins. "
            "It is trained with BCE plus a blocking-consistency penalty on the expected fine spins."
        )
        report.append("")
        report.append("## Decision")
        report.append(
            "The learned model should beat noisy replication in both generated16 mean abs z and input/reblocked overlap. "
            "If it does not, the next step is to revisit the architecture rather than add accept/reject."
        )
        REPORT_MD.write_text("\n".join(report) + "\n")

        print(json.dumps({"written": str(SUMMARY_JSON), "best_lambda": best_run.lambda_block}, indent=2))
        return 0

    except Exception as exc:
        err = {
            "status": "error",
            "exception": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        }
        SUMMARY_JSON.write_text(json.dumps(err, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"written": str(SUMMARY_JSON), "error": str(exc)}, indent=2))
        return 1


def local_relaxation(mod, coarse: np.ndarray, init: np.ndarray, lambda_block: float, n_sweeps: int, seed: int):
    """Local Metropolis relaxation on S_Ising + lambda_block * alpha * (t - p)^2."""
    rng = np.random.default_rng(seed)
    s = init.astype(np.float32).copy()
    t = coarse.astype(np.float32)
    p = compute_p_field_np(s, KERNEL)
    Lf = s.shape[0]
    n_accept = 0
    n_total = 0
    for _ in range(n_sweeps):
        for idx in rng.permutation(Lf * Lf):
            i = int(idx // Lf)
            j = int(idx % Lf)
            spin = s[i, j]
            nn = s[(i + 1) % Lf, j] + s[(i - 1) % Lf, j] + s[i, (j + 1) % Lf] + s[i, (j - 1) % Lf]
            dS_ising = 2.0 * BETA * spin * nn
            dS_block = 0.0
            xs = [i] if i % 2 == 0 else [(i - 1) % Lf, (i + 1) % Lf]
            ys = [j] if j % 2 == 0 else [(j - 1) % Lf, (j + 1) % Lf]
            for cx in xs:
                for cy in ys:
                    ci = (cx // 2) % (Lf // 2)
                    cj = (cy // 2) % (Lf // 2)
                    dx = abs(((i - cx + Lf // 2) % Lf) - Lf // 2)
                    dy = abs(((j - cy + Lf // 2) % Lf) - Lf // 2)
                    if dx == 0 and dy == 0:
                        coeff = KERNEL["w00"]
                    elif dx == 0 or dy == 0:
                        coeff = KERNEL["w01"]
                    else:
                        coeff = KERNEL["w11"]
                    p_old = float(p[ci, cj])
                    p_new = p_old - 2.0 * spin * coeff
                    dS_block += lambda_block * KERNEL["alpha"] * ((t[ci, cj] - p_new) ** 2 - (t[ci, cj] - p_old) ** 2)
            dS = dS_ising + dS_block
            n_total += 1
            if dS <= 0.0 or rng.random() < math.exp(-float(dS)):
                s[i, j] = -spin
                p = compute_p_field_np(s, KERNEL)
                n_accept += 1
    return s.astype(np.float32), {"acceptance": float(n_accept / max(1, n_total))}


if __name__ == "__main__":
    raise SystemExit(main())
