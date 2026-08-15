#!/usr/bin/env python3
"""Minimal 2604-style multilevel factorization smoke/pilot.

This experiment is intentionally separate from the current inverse-blocking
baseline. It tests only the smallest finite-lambda version of

    q(phi_f, phi_i, phi_c) = q_c(phi_c) q_i(phi_i | phi_c) q_f(phi_f | phi_c, phi_i)

with our kappa/lambda normalization. The coarse factor is empirical replay
from the paired blocked-fine dataset, so this is not a native-coarse sampler.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn


EXP = Path(__file__).resolve().parents[0]
PROJECT = EXP.parents[1]
ROOT = PROJECT.parent
DATA = PROJECT / "outputs" / "paired_data_lam1_kappaf0p320"
OUT = EXP / "outputs" / "tiny_run"
CKPT = OUT / "checkpoints"
LOGS = ROOT / "testing_mlneuralsampler_multilevel" / "docs" / "reproduction_log.md"

if str(PROJECT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT / "scripts"))
import nullspace_conditional_nf_pilot as pilot  # type: ignore


SEED = 20240624
L_FINE = 16
L_COARSE = 8
LAMBDA = 1.0
KAPPA_F = 0.320
BATCH_SIZE = 16
EPOCHS = 20
LR = 1.0e-3
EVAL_N = 128
HIDDEN = 96
LAYERS = 4
INTERMEDIATE_SCALE = 0.35
FINE_DELTA_SCALE = 0.35


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def load_data() -> dict[str, np.ndarray]:
    splits_npz = np.load(DATA / "split_indices.npz")
    return {
        "fine": np.load(DATA / "fine_configs.npy").astype(np.float32),
        "coarse": np.load(DATA / "coarse_blocked_configs.npy").astype(np.float32),
        "backbone": np.load(DATA / "backbone_configs.npy").astype(np.float32),
        "train": splits_npz["train"].astype(int),
        "val": splits_npz["val"].astype(int),
        "test": splits_npz["test"].astype(int),
    }


def coarse_broadcast(coarse: torch.Tensor) -> torch.Tensor:
    return coarse.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)


def make_condition(coarse: torch.Tensor, backbone: torch.Tensor, phi_i: torch.Tensor | None = None) -> torch.Tensor:
    parts = [coarse.reshape(len(coarse), -1), backbone.reshape(len(backbone), -1)]
    if phi_i is not None:
        parts.append(phi_i.reshape(len(phi_i), -1))
    return torch.cat(parts, dim=1)


class Coupling(nn.Module):
    def __init__(self, dim: int, cond_dim: int, mask: torch.Tensor):
        super().__init__()
        self.register_buffer("mask", mask)
        self.net = nn.Sequential(
            nn.Linear(dim + cond_dim, HIDDEN),
            nn.SiLU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.SiLU(),
            nn.Linear(HIDDEN, 2 * dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def st(self, x: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.cat([x * self.mask, cond], dim=1)
        s, t = self.net(h).chunk(2, dim=1)
        active = 1.0 - self.mask
        return 0.35 * torch.tanh(s) * active, t * active

    def forward(self, z: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        s, t = self.st(z, cond)
        x = z * self.mask + (1.0 - self.mask) * (z * torch.exp(s) + t)
        return x, s.sum(dim=1)

    def inverse(self, x: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        s, t = self.st(x, cond)
        z = x * self.mask + (1.0 - self.mask) * ((x - t) * torch.exp(-s))
        return z, -s.sum(dim=1)


class Flow(nn.Module):
    def __init__(self, dim: int, cond_dim: int, layers: int = LAYERS):
        super().__init__()
        base = (torch.arange(dim) % 2).float()
        self.layers = nn.ModuleList(
            [Coupling(dim, cond_dim, base if i % 2 == 0 else 1.0 - base) for i in range(layers)]
        )

    def forward(self, z: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = z
        ld = torch.zeros(z.shape[0], dtype=z.dtype, device=z.device)
        for layer in self.layers:
            x, d = layer(x, cond)
            ld = ld + d
        return x, ld

    def inverse(self, x: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = x
        ld = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
        for layer in reversed(self.layers):
            z, d = layer.inverse(z, cond)
            ld = ld + d
        return z, ld


def gaussian_logp(z: torch.Tensor) -> torch.Tensor:
    return -0.5 * (z.square() + math.log(2.0 * math.pi)).sum(dim=1)


def fine_action(phi: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    nn = 0.5 * (
        (phi * torch.roll(phi, -1, dims=-2)).mean(dim=(-2, -1))
        + (phi * torch.roll(phi, -1, dims=-1)).mean(dim=(-2, -1))
    )
    phi2 = phi.square().mean(dim=(-2, -1))
    phi4 = phi.pow(4).mean(dim=(-2, -1))
    density = -4.0 * KAPPA_F * nn + (1.0 - 2.0 * LAMBDA) * phi2 + LAMBDA * phi4
    return density * (L_FINE * L_FINE), {
        "action_density": density,
        "action_hopping_density": -4.0 * KAPPA_F * nn,
        "action_phi2_density": (1.0 - 2.0 * LAMBDA) * phi2,
        "action_phi4_density": LAMBDA * phi4,
    }


def obs_np(phi: np.ndarray) -> dict[str, float]:
    phi = phi.astype(np.float64)
    m = phi.mean(axis=(-2, -1))
    nn = 0.5 * (
        (phi * np.roll(phi, -1, axis=-2)).mean(axis=(-2, -1))
        + (phi * np.roll(phi, -1, axis=-1)).mean(axis=(-2, -1))
    )
    nn2 = 0.5 * (
        ((phi * np.roll(phi, -1, axis=-2)) ** 2).mean(axis=(-2, -1))
        + ((phi * np.roll(phi, -1, axis=-1)) ** 2).mean(axis=(-2, -1))
    )
    diag = (phi * np.roll(np.roll(phi, -1, axis=-2), -1, axis=-1)).mean(axis=(-2, -1))
    twonn = 0.5 * (
        (phi * np.roll(phi, -2, axis=-2)).mean(axis=(-2, -1))
        + (phi * np.roll(phi, -2, axis=-1)).mean(axis=(-2, -1))
    )
    m2 = float(np.mean(m**2))
    m4 = float(np.mean(m**4))
    b4 = m4 / (m2 * m2) if m2 > 0 else math.nan
    u4 = 1.0 - b4 / 3.0 if math.isfinite(b4) else math.nan
    hop = -4.0 * KAPPA_F * float(np.mean(nn))
    p2 = -float(np.mean(phi**2))
    p4 = float(np.mean(phi**4))
    return {
        "phi2": float(np.mean(phi**2)),
        "phi4": float(np.mean(phi**4)),
        "NN": float(np.mean(nn)),
        "nn2": float(np.mean(nn2)),
        "diag": float(np.mean(diag)),
        "2nn": float(np.mean(twonn)),
        "Binder_U4": float(u4),
        "action_hopping_density": hop,
        "action_phi2_density": p2,
        "action_phi4_density": p4,
        "action_density": hop + p2 + p4,
    }


def sample_hierarchy(
    q_i: Flow,
    q_f: Flow,
    coarse: torch.Tensor,
    backbone: torch.Tensor,
    rng_shape: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n = len(coarse) if rng_shape is None else rng_shape
    cond_i = make_condition(coarse, backbone)
    z_i = torch.randn(n, L_FINE * L_FINE)
    y_i, ld_i = q_i(z_i, cond_i)
    phi_i = backbone + INTERMEDIATE_SCALE * y_i.reshape(n, L_FINE, L_FINE)
    logq_i = gaussian_logp(z_i) - ld_i - (L_FINE * L_FINE) * math.log(INTERMEDIATE_SCALE)

    cond_f = make_condition(coarse, backbone, phi_i)
    z_f = torch.randn(n, L_FINE * L_FINE)
    y_f, ld_f = q_f(z_f, cond_f)
    phi_f = phi_i + FINE_DELTA_SCALE * y_f.reshape(n, L_FINE, L_FINE)
    logq_f = gaussian_logp(z_f) - ld_f - (L_FINE * L_FINE) * math.log(FINE_DELTA_SCALE)
    return phi_i, phi_f, logq_i, logq_f


def evaluate(q_i: Flow, q_f: Flow, coarse: np.ndarray, backbone: np.ndarray, fine: np.ndarray, label: str) -> dict[str, float]:
    n = min(EVAL_N, len(coarse))
    c = torch.tensor(coarse[:n])
    b = torch.tensor(backbone[:n])
    with torch.no_grad():
        phi_i, phi_f, logq_i, logq_f = sample_hierarchy(q_i, q_f, c, b)
        S, comps = fine_action(phi_f)
        block_res = pilot.block_sym_np(phi_f.numpy().astype(np.float64), load_weights()) - coarse[:n]
    row = {
        "label": label,
        "S_fine": float(S.mean()),
        "logq_i": float(logq_i.mean()),
        "logq_f": float(logq_f.mean()),
        "loss": float((S + logq_i + logq_f).mean()),
        "block_RMS": float(np.sqrt(np.mean(block_res**2))),
        "block_max": float(np.max(np.abs(block_res))),
        **obs_np(phi_f.numpy()),
    }
    fine_obs = obs_np(fine[:n])
    row.update({f"target_{k}": v for k, v in fine_obs.items() if k in ("phi2", "phi4", "NN", "nn2", "diag", "2nn", "action_density")})
    return row


def load_weights() -> dict[str, float]:
    meta = json.loads(pilot.KERNEL.read_text())
    return {k: float(meta["weights"][k]) for k in ["w00", "w10", "w11", "w20", "w21", "w22"]}


def preflight(q_i: Flow, q_f: Flow, data: dict[str, np.ndarray]) -> dict[str, object]:
    idx = data["train"][:8]
    coarse = torch.tensor(data["coarse"][idx])
    backbone = torch.tensor(data["backbone"][idx])
    phi_i, phi_f, logq_i, logq_f = sample_hierarchy(q_i, q_f, coarse, backbone)
    cond_i = make_condition(coarse, backbone)
    z = torch.randn(8, L_FINE * L_FINE)
    x, ld = q_i(z, cond_i)
    z_back, inv_ld = q_i.inverse(x, cond_i)
    S, _ = fine_action(phi_f)
    block_res = pilot.block_sym_np(phi_f.detach().numpy().astype(np.float64), load_weights()) - data["coarse"][idx]
    return {
        "q_factorization": "q_c empirical replay * q_i(phi_i|phi_c) * q_f(phi_f|phi_c,phi_i)",
        "normalization": {"lambda": LAMBDA, "kappa_f": KAPPA_F, "action": "our finite-lambda phi4 normalization"},
        "coarse_factor": "empirical replay from paired blocked-fine train split; log q_c treated as constant in this pilot",
        "shapes": {
            "coarse": list(coarse.shape),
            "backbone": list(backbone.shape),
            "phi_i": list(phi_i.shape),
            "phi_f": list(phi_f.shape),
            "logq_i": list(logq_i.shape),
            "logq_f": list(logq_f.shape),
        },
        "q_i_invertibility_max_abs_z_error": float(torch.max(torch.abs(z_back - z)).detach()),
        "q_i_logJ_sign_max_abs_sum": float(torch.max(torch.abs(ld + inv_ld)).detach()),
        "finite_action": bool(torch.isfinite(S).all()),
        "finite_logq_i": bool(torch.isfinite(logq_i).all()),
        "finite_logq_f": bool(torch.isfinite(logq_f).all()),
        "finite_loss": bool(torch.isfinite(S + logq_i + logq_f).all()),
        "initial_block_RMS": float(np.sqrt(np.mean(block_res**2))),
        "initial_block_max": float(np.max(np.abs(block_res))),
    }


def reproduction_context() -> str:
    if not LOGS.exists():
        return "Earlier reproduction log was not found."
    text = LOGS.read_text()
    keys = [
        "API mismatch",
        "IMH",
        "normalization",
        "conditional diagnostic",
        "global coarse distribution",
        "UV/detail failure",
        "not yet a clear improvement",
    ]
    lines = []
    for line in text.splitlines():
        if any(k.lower() in line.lower() for k in keys):
            lines.append(line)
    return "\n".join(lines[:80])


def main() -> None:
    start_time = time.time()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(f"Refusing to overwrite existing outputs in {OUT}")
    CKPT.mkdir(parents=True, exist_ok=True)

    data = load_data()
    cond_i_dim = L_COARSE * L_COARSE + L_FINE * L_FINE
    cond_f_dim = L_COARSE * L_COARSE + 2 * L_FINE * L_FINE
    q_i = Flow(L_FINE * L_FINE, cond_i_dim)
    q_f = Flow(L_FINE * L_FINE, cond_f_dim)
    opt = torch.optim.Adam(list(q_i.parameters()) + list(q_f.parameters()), lr=LR)

    pre = preflight(q_i, q_f, data)
    (OUT / "preflight_summary.json").write_text(json.dumps(pre, indent=2) + "\n")
    if not all(pre[k] for k in ["finite_action", "finite_logq_i", "finite_logq_f", "finite_loss"]):
        raise RuntimeError("Preflight failed finite checks")

    train_idx = data["train"]
    val_idx = data["val"]
    fine = data["fine"]
    coarse = data["coarse"]
    backbone = data["backbone"]

    rows: list[dict[str, object]] = []
    sample_rows: list[dict[str, object]] = []
    sample_rows.append({"label": "untrained", **evaluate(q_i, q_f, coarse[val_idx], backbone[val_idx], fine[val_idx], "untrained")})
    sample_rows.append({"label": "paired_fine", **obs_np(fine[val_idx[:EVAL_N]])})
    sample_rows.append({"label": "smooth_backbone", **obs_np(backbone[val_idx[:EVAL_N]])})

    for epoch in range(1, EPOCHS + 1):
        perm = np.random.permutation(train_idx)
        losses = []
        actions = []
        logqis = []
        logqfs = []
        block_rms = []
        for start in range(0, len(perm), BATCH_SIZE):
            ids = perm[start : start + BATCH_SIZE]
            c = torch.tensor(coarse[ids])
            b = torch.tensor(backbone[ids])
            phi_i, phi_f, logq_i, logq_f = sample_hierarchy(q_i, q_f, c, b)
            S, _ = fine_action(phi_f)
            loss = (S + logq_i + logq_f).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(q_i.parameters()) + list(q_f.parameters()), 10.0)
            opt.step()
            losses.append(float(loss.detach()))
            actions.append(float(S.mean().detach()))
            logqis.append(float(logq_i.mean().detach()))
            logqfs.append(float(logq_f.mean().detach()))
            br = pilot.block_sym_np(phi_f.detach().numpy().astype(np.float64), load_weights()) - coarse[ids]
            block_rms.append(float(np.sqrt(np.mean(br**2))))
        eval_row = evaluate(q_i, q_f, coarse[val_idx], backbone[val_idx], fine[val_idx], f"epoch_{epoch}")
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "train_S_fine": float(np.mean(actions)),
            "train_logq_i": float(np.mean(logqis)),
            "train_logq_f": float(np.mean(logqfs)),
            "train_block_RMS": float(np.mean(block_rms)),
            **{f"eval_{k}": v for k, v in eval_row.items() if k != "label"},
        }
        rows.append(row)
        if epoch % 5 == 0:
            torch.save(
                {"epoch": epoch, "q_i": q_i.state_dict(), "q_f": q_f.state_dict(), "history": rows},
                CKPT / f"epoch_{epoch:03d}.pt",
            )

    write_csv(OUT / "history.csv", rows)
    sample_rows.append({"label": "trained_final", **evaluate(q_i, q_f, coarse[val_idx], backbone[val_idx], fine[val_idx], "trained_final")})
    write_csv(OUT / "sample_observables.csv", sample_rows)

    summary = {
        "experiment": "minimal_2604_multilevel_revisit",
        "factorization": "q_c empirical replay * q_i(phi_i|phi_c) * q_f(phi_f|phi_c,phi_i)",
        "seed": SEED,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "intermediate_scale": INTERMEDIATE_SCALE,
        "fine_delta_scale": FINE_DELTA_SCALE,
        "normalization": {"lambda": LAMBDA, "kappa_f": KAPPA_F, "action": "our finite-lambda phi4 normalization"},
        "preflight": pre,
        "final": rows[-1],
        "elapsed_seconds": time.time() - start_time,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    ctx = reproduction_context()
    final = sample_rows[-1]
    target = next(r for r in sample_rows if r["label"] == "paired_fine")
    report = f"""# Minimal 2604-Style Multilevel Revisit

This is a separate experiment. It does not replace the current inverse-blocking baseline.

## Factorization Tested

`q = q_c(phi_c) * q_i(phi_i | phi_c) * q_f(phi_f | phi_c, phi_i)`

- `q_c`: empirical replay from paired blocked-fine coarse fields.
- `q_i`: tiny conditional affine flow over a 16x16 intermediate field, conditioned on `phi_c` and `phi_back`.
- `q_f`: tiny conditional affine flow over a 16x16 fine correction, conditioned on `phi_c`, `phi_back`, and `phi_i`.
- action: our finite-lambda phi4 normalization, `lambda={LAMBDA}`, `kappa_f={KAPPA_F}`.

This is a tiny plumbing/training test, not a valid full sampler, because `q_c` is not a native learned coarse model.

## Preflight

- q_i invertibility max error: `{pre['q_i_invertibility_max_abs_z_error']:.3g}`
- q_i logJ sign check: `{pre['q_i_logJ_sign_max_abs_sum']:.3g}`
- finite action/logq/loss: `{pre['finite_action']}/{pre['finite_logq_i'] and pre['finite_logq_f']}/{pre['finite_loss']}`
- initial block RMS: `{pre['initial_block_RMS']:.6g}`

## Tiny Training Result

Validation paired fine:

- phi2={target['phi2']:.6g}
- phi4={target['phi4']:.6g}
- NN={target['NN']:.6g}
- nn2={target['nn2']:.6g}
- action density={target['action_density']:.6g}

Final generated:

- phi2={final['phi2']:.6g}
- phi4={final['phi4']:.6g}
- NN={final['NN']:.6g}
- nn2={final['nn2']:.6g}
- action density={final['action_density']:.6g}
- block RMS={final['block_RMS']:.6g}

History is saved in `history.csv`; generated-vs-reference observables are saved in `sample_observables.csv`.

## Comparison to Earlier Failed Reproduction Logs

Relevant excerpts from the old logs:

```text
{ctx}
```

## Failure Classification

The previous failure is not explained by kappa/lambda normalization alone. This pilot uses our normalization consistently and passes shape, invertibility, finite-loss, and action checks.

The previous failure was partly architectural/API-related in the public reproduction path: the old logs show an upstream `sample()`/`sample_n_OBS()` mismatch and IMH reverse-path incompatibility.

For the finite-lambda inverse problem, the remaining failure is most likely architecture/training-resource related and still partly unresolved. This minimal `q_c*q_i*q_f` implementation runs and trains without numerical failure, but the tiny reverse-KL run does not establish a good sampler. The architecture is only a minimal factorization check; it lacks the full paper-style capacity, staging, and validation machinery.

## Bottom Line

We now understand enough of the method to implement the factorization cleanly in our normalization. The original reproduction failure was not just a normalization bug. It was a mix of public-code API mismatch, inadequate architecture translation, and insufficient training/validation resources for the finite-lambda problem.

Do not promote this branch yet. The next useful test would be a staged, capacity-controlled full-field conditional model with this factorization and explicit q_c handling, compared through the existing inverse-blocking benchmark diagnostics.
"""
    (OUT / "report.md").write_text(report)

    shutil.copy2(Path(__file__), OUT / "train_minimal_2604.py")


if __name__ == "__main__":
    main()
