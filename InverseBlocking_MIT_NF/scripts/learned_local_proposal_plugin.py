#!/usr/bin/env python3
"""Train a learned local proposal and benchmark it through the proposal driver.

This is intentionally only a benchmark plug-in.  It emits projected-Haar null
coordinates for ``scripts/inverse_blocking_proposal_benchmark.py`` via
``--initial-v-npy`` and does not alter the exact constrained correction loop.
"""

from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
if str(PROJECT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT / "scripts"))

import inverse_blocking_proposal_benchmark as bench  # type: ignore


DATA = PROJECT / "outputs" / "paired_data_lam1_kappaf0p320"
OUT = PROJECT / "outputs" / "learned_local_proposal_benchmark_plugin"
MODEL_DIR = OUT / "model"
BENCH_LEARNED = OUT / "benchmark_learned"
BENCH_BASELINE = OUT / "benchmark_builtin_local_chunk"

SEED = 20260624
GROUP_SIZE = 6
N_CHUNKS = 192 // GROUP_SIZE
N_BENCH = 256
EPOCHS = 300
PATIENCE = 40
BATCH_CHUNKS = 2048
LR = 1.0e-3


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def obs_features(field: np.ndarray) -> np.ndarray:
    m = field.mean(axis=(-2, -1))
    nn = 0.5 * (
        (field * np.roll(field, -1, axis=-2)).mean(axis=(-2, -1))
        + (field * np.roll(field, -1, axis=-1)).mean(axis=(-2, -1))
    )
    nn2 = 0.5 * (
        ((field * np.roll(field, -1, axis=-2)) ** 2).mean(axis=(-2, -1))
        + ((field * np.roll(field, -1, axis=-1)) ** 2).mean(axis=(-2, -1))
    )
    return np.stack(
        [
            m,
            np.abs(m),
            (field**2).mean(axis=(-2, -1)),
            (field**4).mean(axis=(-2, -1)),
            nn,
            nn2,
        ],
        axis=1,
    )


def patch_stats(arr: np.ndarray, block_ids: list[int]) -> np.ndarray:
    """Local 3x3 coarse/block patch features for each sample and chunk."""

    # arr shape: N, 8, 8
    n = len(arr)
    out = np.zeros((n, len(block_ids), 8), dtype=np.float64)
    for j, block in enumerate(block_ids):
        by, bx = divmod(block, 8)
        patch = np.stack(
            [
                np.roll(np.roll(arr, -dy, axis=1), -dx, axis=2)[:, by, bx]
                for dy in (-1, 0, 1)
                for dx in (-1, 0, 1)
            ],
            axis=1,
        )
        center = arr[:, by, bx]
        out[:, j, :] = np.stack(
            [
                center,
                patch.mean(axis=1),
                patch.std(axis=1),
                (patch**2).mean(axis=1),
                patch.max(axis=1),
                patch.min(axis=1),
                np.mean(np.abs(patch), axis=1),
                np.mean(patch[:, [1, 3, 4, 5, 7]], axis=1),
            ],
            axis=1,
        )
    return out


def build_chunk_features(coarse: np.ndarray, backbone: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Return features with shape (N, N_CHUNKS, F)."""

    n = len(coarse)
    chunk_blocks = []
    for chunk in range(N_CHUNKS):
        # H columns are ordered 3 per 2x2 block; G=6 groups two neighboring block-detail triples.
        first_block = (chunk * 2) % 64
        chunk_blocks.append(first_block)

    coarse_stats = patch_stats(coarse, chunk_blocks)
    # Use backbone block averages as 8x8 local backbone summaries.
    back_block = 0.25 * (
        backbone[:, 0::2, 0::2]
        + backbone[:, 1::2, 0::2]
        + backbone[:, 0::2, 1::2]
        + backbone[:, 1::2, 1::2]
    )
    back_stats = patch_stats(back_block, chunk_blocks)
    global_feat = np.concatenate([obs_features(coarse), obs_features(backbone)], axis=1)
    global_feat = np.repeat(global_feat[:, None, :], N_CHUNKS, axis=1)

    pos = np.zeros((N_CHUNKS, 4), dtype=np.float64)
    for chunk, block in enumerate(chunk_blocks):
        by, bx = divmod(block, 8)
        pos[chunk] = [
            math.sin(2.0 * math.pi * by / 8.0),
            math.cos(2.0 * math.pi * by / 8.0),
            math.sin(2.0 * math.pi * bx / 8.0),
            math.cos(2.0 * math.pi * bx / 8.0),
        ]
    pos_feat = np.repeat(pos[None, :, :], n, axis=0)
    features = np.concatenate([coarse_stats, back_stats, global_feat, pos_feat], axis=2)
    return features.astype(np.float32), chunk_blocks


class ChunkGaussian(nn.Module):
    def __init__(self, n_features: int, group_size: int = GROUP_SIZE):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 96),
            nn.SiLU(),
            nn.Linear(96, 96),
            nn.SiLU(),
            nn.Linear(96, 2 * group_size),
        )
        self.group_size = group_size

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y = self.net(x)
        mu, raw = y[:, : self.group_size], y[:, self.group_size :]
        logvar = torch.clamp(raw, -7.0, 3.0)
        return mu, logvar


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std


def make_chunk_dataset(features: np.ndarray, u_true: np.ndarray, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = features[idx].reshape(-1, features.shape[-1])
    y = u_true[idx].reshape(-1, GROUP_SIZE)
    return x.astype(np.float32), y.astype(np.float32)


def gaussian_nll(mu: torch.Tensor, logvar: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return 0.5 * (((y - mu) ** 2) * torch.exp(-logvar) + logvar + math.log(2.0 * math.pi)).sum(dim=1).mean()


def evaluate_nll(model: nn.Module, x: np.ndarray, y: np.ndarray, batch: int = 4096) -> float:
    vals = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), batch):
            xt = torch.tensor(x[start : start + batch], dtype=torch.float32)
            yt = torch.tensor(y[start : start + batch], dtype=torch.float32)
            mu, logvar = model(xt)
            vals.append(float(gaussian_nll(mu, logvar, yt)))
    return float(np.mean(vals))


def sample_u(model: nn.Module, features_std: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = features_std.shape[0]
    x = torch.tensor(features_std.reshape(-1, features_std.shape[-1]), dtype=torch.float32)
    mus = []
    logvars = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), 4096):
            mu, logvar = model(x[start : start + 4096])
            mus.append(mu.cpu().numpy())
            logvars.append(logvar.cpu().numpy())
    mu = np.concatenate(mus, axis=0).reshape(n, N_CHUNKS, GROUP_SIZE)
    logvar = np.concatenate(logvars, axis=0).reshape(n, N_CHUNKS, GROUP_SIZE)
    eps = rng.normal(size=mu.shape)
    u = mu + np.exp(0.5 * logvar) * eps
    return u.reshape(n, 192), mu.reshape(n, 192), logvar.reshape(n, 192)


def run_benchmark(output_dir: Path, mode: str, initial_v: Path | None = None) -> None:
    cmd = [
        str(ROOT / "../.venv/bin/python"),
        "-B",
        str(PROJECT / "scripts" / "inverse_blocking_proposal_benchmark.py"),
        "--output-dir",
        str(output_dir),
        "--n-conditions",
        str(N_BENCH),
        "--group-size",
        str(GROUP_SIZE),
        "--step-size",
        "0.1",
        "--sweeps",
        "0,5,10,25,50",
        "--proposal-mode",
        mode,
    ]
    if initial_v is not None:
        cmd.extend(["--initial-v-npy", str(initial_v)])
    subprocess.run(cmd, cwd=ROOT, check=True)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(f"Refusing to overwrite existing outputs in {OUT}")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    data = bench.load_paired_data()
    w = bench.load_weights()
    q_basis, basis_diag = bench.build_local_q_basis(w)
    dense_basis = np.load(bench.N_DENSE_BASIS).astype(np.float64)
    u_true = bench.local_coords_from_dense(data["v_true"], dense_basis, q_basis)

    features, chunk_blocks = build_chunk_features(data["coarse"], data["backbone"])
    train_idx, val_idx, test_idx = data["train_idx"], data["val_idx"], data["test_idx"]
    x_train_raw, y_train = make_chunk_dataset(features, u_true, train_idx)
    x_val_raw, y_val = make_chunk_dataset(features, u_true, val_idx)
    x_test_raw, y_test = make_chunk_dataset(features, u_true, test_idx)

    std = Standardizer(mean=x_train_raw.mean(axis=0), std=x_train_raw.std(axis=0) + 1.0e-6)
    x_train = std.transform(x_train_raw).astype(np.float32)
    x_val = std.transform(x_val_raw).astype(np.float32)
    x_test = std.transform(x_test_raw).astype(np.float32)

    model = ChunkGaussian(x_train.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1.0e-4)
    best_state: dict[str, torch.Tensor] | None = None
    best_val = float("inf")
    best_epoch = 0
    history: list[dict[str, Any]] = []
    rng = np.random.default_rng(SEED)
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = rng.permutation(len(x_train))
        losses = []
        for start in range(0, len(order), BATCH_CHUNKS):
            ids = order[start : start + BATCH_CHUNKS]
            xt = torch.tensor(x_train[ids], dtype=torch.float32)
            yt = torch.tensor(y_train[ids], dtype=torch.float32)
            mu, logvar = model(xt)
            loss = gaussian_nll(mu, logvar, yt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            losses.append(float(loss.detach()))
        train_nll = float(np.mean(losses))
        val_nll = evaluate_nll(model, x_val, y_val)
        test_nll = evaluate_nll(model, x_test, y_test)
        improved = val_nll < best_val - 1.0e-4
        if improved:
            best_val = val_nll
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        history.append(
            {
                "epoch": epoch,
                "train_nll": train_nll,
                "val_nll": val_nll,
                "test_nll": test_nll,
                "best_val_nll": best_val,
                "best_epoch": best_epoch,
                "elapsed_sec": time.time() - t0,
            }
        )
        if epoch - best_epoch >= PATIENCE:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_mean": std.mean,
            "feature_std": std.std,
            "chunk_blocks": chunk_blocks,
            "basis_diag": basis_diag,
            "best_epoch": best_epoch,
            "best_val_nll": best_val,
            "group_size": GROUP_SIZE,
        },
        MODEL_DIR / "chunk_gaussian_model.pt",
    )
    write_csv(MODEL_DIR / "training_history.csv", history)

    # Benchmark selection mirrors inverse_blocking_proposal_benchmark.py.
    sel_idx = np.concatenate([test_idx, val_idx, train_idx])[:N_BENCH]
    features_sel = std.transform(features[sel_idx].reshape(-1, features.shape[-1])).reshape(len(sel_idx), N_CHUNKS, -1)
    learned_u, learned_mu, learned_logvar = sample_u(model, features_sel.astype(np.float32), SEED + 17)
    np.save(OUT / "learned_initial_v.npy", learned_u.astype(np.float32))
    np.save(OUT / "learned_initial_v_mean.npy", learned_mu.astype(np.float32))
    np.save(OUT / "learned_initial_v_logvar.npy", learned_logvar.astype(np.float32))
    np.save(OUT / "benchmark_selected_indices.npy", sel_idx.astype(np.int64))

    learned_phi = bench.phi_from_local_coords(learned_u, data["backbone"][sel_idx], q_basis)
    np.save(OUT / "learned_initial_phi.npy", learned_phi.astype(np.float32))
    initial_metrics = bench.averaged_metrics(learned_phi, data["coarse"][sel_idx], w)

    run_benchmark(BENCH_LEARNED, "external_v", OUT / "learned_initial_v.npy")
    run_benchmark(BENCH_BASELINE, "builtin_local_chunk", None)

    learned_rows = read_rows(BENCH_LEARNED / "proposal_observables.csv")
    baseline_rows = read_rows(BENCH_BASELINE / "proposal_observables.csv")
    compare_rows = []
    for source, rows in [("learned", learned_rows), ("builtin_local_chunk", baseline_rows)]:
        for row in rows:
            if row["proposal_mode"] in {"external_v", "builtin_local_chunk"}:
                compare_rows.append(
                    {
                        "source": source,
                        "ensemble": row["ensemble"],
                        "sweeps": row["sweeps"],
                        "phi2": row["phi2"],
                        "phi4": row["phi4"],
                        "NN": row["NN"],
                        "nn2": row["nn2"],
                        "diag": row["diag"],
                        "2nn": row["2nn"],
                        "Binder_U4": row["Binder_U4"],
                        "xi/L": row["xi/L"],
                        "action_density": row["action_density"],
                        "block_RMS": row["block_RMS"],
                        "block_max": row["block_max"],
                        "acceptance_rate": row["acceptance_rate"],
                        "cost_units_per_sample": row["cost_units_per_sample"],
                        "moment_score": row["moment_score"],
                    }
                )
    write_csv(OUT / "benchmark_comparison.csv", compare_rows)

    summary = {
        "purpose": "learned local proposal benchmark plug-in only",
        "canonical_data": str(DATA.resolve()),
        "proposal_coordinate_system": "projected-Haar local null coordinates expected by inverse_blocking_proposal_benchmark --initial-v-npy",
        "model": "shared local chunk diagonal Gaussian MLP",
        "group_size": GROUP_SIZE,
        "n_chunks": N_CHUNKS,
        "n_benchmark_conditions": N_BENCH,
        "best_epoch": best_epoch,
        "best_val_nll": best_val,
        "final_epoch": history[-1]["epoch"],
        "initial_metrics": initial_metrics,
        "basis_diagnostics": basis_diag,
        "outputs": {
            "learned_initial_v": str((OUT / "learned_initial_v.npy").resolve()),
            "learned_benchmark": str(BENCH_LEARNED.resolve()),
            "builtin_baseline_benchmark": str(BENCH_BASELINE.resolve()),
        },
        "standing_generation_note": "No new phi4 ensemble was generated by this script.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    def best_by(rows: list[dict[str, str]]) -> dict[str, str]:
        proposal_rows = [r for r in rows if r["proposal_mode"] in {"external_v", "builtin_local_chunk"}]
        return min(proposal_rows, key=lambda r: float(r["moment_score"]))

    learned_best = best_by(learned_rows)
    baseline_best = best_by(baseline_rows)
    target = next(r for r in learned_rows if r["ensemble"] == "paired_fine")

    def table_for(rows: list[dict[str, str]], source: str) -> list[str]:
        out = []
        for r in rows:
            if r["proposal_mode"] in {"external_v", "builtin_local_chunk"}:
                out.append(
                    f"| {source} | {r['sweeps']} | {float(r['phi2']):.6g} | {float(r['phi4']):.6g} | "
                    f"{float(r['nn2']):.6g} | {float(r['NN']):.6g} | {float(r['diag']):.6g} | "
                    f"{float(r['2nn']):.6g} | {float(r['block_RMS']):.3g} | {float(r['acceptance_rate']) if r['acceptance_rate'] else math.nan:.6g} | "
                    f"{float(r['cost_units_per_sample']):.0f} | {float(r['moment_score']):.6g} |"
                )
        return out

    report_lines = [
        "# Learned Local Proposal Benchmark Plug-in",
        "",
        "This run trains a learned initial proposal only so it can be evaluated by the existing benchmark driver. It does not generate new phi4 ensembles and does not change the exact constrained correction loop.",
        "",
        "## Training",
        "",
        f"- canonical data: `{DATA.resolve()}`",
        f"- target coordinates: projected-Haar local null coordinates, `G={GROUP_SIZE}` chunks",
        "- model: shared MLP predicting diagonal Gaussian mean/log-variance per local chunk",
        f"- best validation epoch: `{best_epoch}`",
        f"- best validation NLL: `{best_val:.6g}`",
        "",
        "## Benchmark Table",
        "",
        f"Paired-fine target on the benchmark subset: phi2 `{float(target['phi2']):.6g}`, phi4 `{float(target['phi4']):.6g}`, nn2 `{float(target['nn2']):.6g}`.",
        "",
        "| source | sweeps | phi2 | phi4 | nn2 | NN | diag | 2nn | block RMS | acceptance | cost units/sample | moment score |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        *table_for(learned_rows, "learned"),
        *table_for(baseline_rows, "builtin"),
        "",
        "## Answers",
        "",
        "1. Observables before correction are reported at sweep 0 in the table and in each benchmark `proposal_observables.csv`.",
        "",
        "2. Observables after 5/10/25/50 correction sweeps are reported with the same benchmark code for learned and built-in proposals.",
        "",
        f"3. Best learned sweep by phi2/phi4/nn2 score: `{learned_best['sweeps']}` with score `{float(learned_best['moment_score']):.6g}`. Best built-in local-chunk sweep: `{baseline_best['sweeps']}` with score `{float(baseline_best['moment_score']):.6g}`.",
        "",
        "4. Block residual remains the benchmark's exact-null residual; any nonzero values are numerical roundoff/projection error.",
        "",
        "5. Cost is reported as action-evaluation units per sample. With `G=6`, each correction sweep uses 32 chunk action batches, so 5 sweeps costs 161 units/sample, 10 costs 321, 25 costs 801, and 50 costs 1601.",
        "",
        "6. A learned proposal is useful only if it reaches the current 50-sweep built-in baseline with fewer sweeps. Use the scores above as the decision criterion.",
    ]
    (OUT / "report.md").write_text("\n".join(report_lines) + "\n")


if __name__ == "__main__":
    main()
