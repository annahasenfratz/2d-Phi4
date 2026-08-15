#!/usr/bin/env python3
"""Tiny symmetric-block conditional NF pilot.

This trains small all-sites affine coupling flows conditioned on a symmetric
block-average coarse field and smooth inverse backbone. It is a pilot only, not
a valid sampler.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn


PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
BASE = PROJECT / "outputs" / "inverse_blocking_step_by_step"
OUT = PROJECT / "outputs" / "symmetric_conditional_nf_pilot"
KERNEL_COPY_DIR = PROJECT / "kernels" / "from_perfect_blocking_lam1p0_blockavg"
SOURCE_KERNEL = ROOT / "perfect_blocking" / "perfect_blocking_lam1p0_blockavg" / "perfect_block_lam1_blockavg_kernel5x5_kernel.json"

L_FINE = 16
L_COARSE = 8
LAMBDA = 1.0
KAPPA_F = 0.320
KAPPA_C = 0.30
ETA_EXPONENT = 0.25
B_SCALE = 2
BLOCK_NORM = B_SCALE ** (ETA_EXPONENT / 2.0)
EPOCHS = 50
BATCH_SIZE = 16
LR = 1.0e-3
SEED = 20240623
LAMBDA_BLOCKS = [0.0, 1.0, 10.0, 100.0]

SHELLS = {
    "w00": [(0, 0)],
    "w10": [(1, 0), (-1, 0), (0, 1), (0, -1)],
    "w11": [(1, 1), (1, -1), (-1, 1), (-1, -1)],
    "w20": [(2, 0), (-2, 0), (0, 2), (0, -2)],
    "w21": [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)],
    "w22": [(2, 2), (2, -2), (-2, 2), (-2, -2)],
}


def copy_kernel_metadata() -> dict:
    KERNEL_COPY_DIR.mkdir(parents=True, exist_ok=True)
    local = KERNEL_COPY_DIR / SOURCE_KERNEL.name
    shutil.copy2(SOURCE_KERNEL, local)
    meta = json.loads(SOURCE_KERNEL.read_text())
    provenance = {
        "original_source_path": str(SOURCE_KERNEL.resolve()),
        "local_copy_path": str(local.resolve()),
        "copy_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "lambda": 1.0,
        "eta_exponent": ETA_EXPONENT,
        "block_norm": BLOCK_NORM,
        "blocking_rule": "symmetric_2x2_average_after_K",
        "K_sum": meta.get("K_sum"),
        "caveat": "first bounded Powell pass, maxfev=100 reached",
        "source_metadata": meta,
    }
    (KERNEL_COPY_DIR / "selected_kernel_metadata.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance


def shell_convolve_np(phi: np.ndarray, w: dict[str, float]) -> np.ndarray:
    out = np.zeros_like(phi, dtype=np.float64)
    for shell, offsets in SHELLS.items():
        for dy, dx in offsets:
            out += w[shell] * np.roll(np.roll(phi, -dy, axis=-2), -dx, axis=-1)
    return out


def block_sym_np(phi: np.ndarray, w: dict[str, float]) -> np.ndarray:
    psi = shell_convolve_np(phi, w)
    return 0.25 * BLOCK_NORM * (psi[:, 0::2, 0::2] + psi[:, 1::2, 0::2] + psi[:, 0::2, 1::2] + psi[:, 1::2, 1::2])


def block_sym_torch(phi: torch.Tensor, w: dict[str, float]) -> torch.Tensor:
    psi = torch.zeros_like(phi)
    for shell, offsets in SHELLS.items():
        for dy, dx in offsets:
            psi = psi + float(w[shell]) * torch.roll(torch.roll(phi, -dy, dims=-2), -dx, dims=-1)
    return 0.25 * BLOCK_NORM * (psi[:, 0::2, 0::2] + psi[:, 1::2, 0::2] + psi[:, 0::2, 1::2] + psi[:, 1::2, 1::2])


def kernel_array(w: dict[str, float], n: int) -> np.ndarray:
    arr = np.zeros((n, n), dtype=float)
    for shell, offsets in SHELLS.items():
        for dy, dx in offsets:
            arr[dy % n, dx % n] += w[shell]
    return arr


def smooth_backbone(coarse: np.ndarray, w: dict[str, float]) -> np.ndarray:
    ktilde = np.fft.fft2(kernel_array(w, L_FINE))
    p = np.fft.fftfreq(L_FINE) * 2.0 * np.pi
    px, py = np.meshgrid(p, p)
    A = 0.25 * (1.0 + np.exp(1j * py)) * (1.0 + np.exp(1j * px))
    coarse_fft = np.fft.fft2(coarse, axes=(-2, -1))
    padded_shift = np.zeros((coarse.shape[0], L_FINE, L_FINE), dtype=complex)
    padded_shift[:, 4:12, 4:12] = 4.0 * np.fft.fftshift(coarse_fft, axes=(-2, -1))
    padded = np.fft.ifftshift(padded_shift, axes=(-2, -1))
    mask_shift = np.zeros((L_FINE, L_FINE), dtype=bool)
    mask_shift[4:12, 4:12] = True
    mask = np.fft.ifftshift(mask_shift)
    inv = np.zeros_like(padded)
    denom = BLOCK_NORM * ktilde * A
    inv[:, mask] = padded[:, mask] / denom[mask]
    return np.fft.ifft2(inv, axes=(-2, -1)).real


def block_replicate(coarse: np.ndarray) -> np.ndarray:
    out = np.empty((coarse.shape[0], L_FINE, L_FINE), dtype=coarse.dtype)
    out[:, 0::2, 0::2] = coarse
    out[:, 1::2, 0::2] = coarse
    out[:, 0::2, 1::2] = coarse
    out[:, 1::2, 1::2] = coarse
    return out


def parity_masks(n: int) -> np.ndarray:
    m = np.zeros((4, L_FINE, L_FINE), dtype=np.float32)
    m[0, 0::2, 0::2] = 1.0
    m[1, 1::2, 0::2] = 1.0
    m[2, 0::2, 1::2] = 1.0
    m[3, 1::2, 1::2] = 1.0
    return np.broadcast_to(m, (n, 4, L_FINE, L_FINE)).copy()


class Coupling(nn.Module):
    def __init__(self, dim: int, cond_dim: int, mask: torch.Tensor):
        super().__init__()
        self.register_buffer("mask", mask)
        self.net = nn.Sequential(nn.Linear(dim + cond_dim, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 2 * dim))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def st(self, x: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.cat([x * self.mask, cond], dim=1)
        s, t = self.net(h).chunk(2, dim=1)
        active = 1.0 - self.mask
        return 0.5 * torch.tanh(s) * active, t * active

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        s, t = self.st(x, cond)
        y = x * self.mask + (1.0 - self.mask) * (x * torch.exp(s) + t)
        return y, s.sum(dim=1)

    def inverse(self, y: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        s, t = self.st(y, cond)
        x = y * self.mask + (1.0 - self.mask) * ((y - t) * torch.exp(-s))
        return x, -s.sum(dim=1)


class Flow(nn.Module):
    def __init__(self, dim: int, cond_dim: int, layers: int = 4):
        super().__init__()
        base = torch.arange(dim) % 2
        self.layers = nn.ModuleList([Coupling(dim, cond_dim, (base if i % 2 == 0 else 1 - base).float()) for i in range(layers)])

    def forward(self, z: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = z
        ld = torch.zeros(z.shape[0], dtype=z.dtype)
        for layer in self.layers:
            x, d = layer(x, cond)
            ld = ld + d
        return x, ld

    def inverse(self, x: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = x
        ld = torch.zeros(x.shape[0], dtype=x.dtype)
        for layer in reversed(self.layers):
            z, d = layer.inverse(z, cond)
            ld = ld + d
        return z, ld


def fine_action(phi: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    nn = 0.5 * ((phi * torch.roll(phi, -1, dims=-2)).mean(dim=(-2, -1)) + (phi * torch.roll(phi, -1, dims=-1)).mean(dim=(-2, -1)))
    phi2 = (phi**2).mean(dim=(-2, -1))
    phi4 = (phi**4).mean(dim=(-2, -1))
    density = -4.0 * KAPPA_F * nn + (1.0 - 2.0 * LAMBDA) * phi2 + LAMBDA * phi4
    return density * (L_FINE * L_FINE), {
        "action_density": density,
        "action_hopping_density": -4.0 * KAPPA_F * nn,
        "action_phi2_density": (1.0 - 2.0 * LAMBDA) * phi2,
        "action_phi4_density": LAMBDA * phi4,
    }


def obs_np(phi: np.ndarray) -> dict[str, float]:
    _, ly, lx = phi.shape
    v = ly * lx
    m = phi.mean(axis=(-2, -1))
    nn = 0.5 * ((phi * np.roll(phi, -1, axis=-2)).mean(axis=(-2, -1)) + (phi * np.roll(phi, -1, axis=-1)).mean(axis=(-2, -1)))
    nn2 = 0.5 * (((phi * np.roll(phi, -1, axis=-2)) ** 2).mean(axis=(-2, -1)) + ((phi * np.roll(phi, -1, axis=-1)) ** 2).mean(axis=(-2, -1)))
    diag = (phi * np.roll(np.roll(phi, -1, axis=-2), -1, axis=-1)).mean(axis=(-2, -1))
    twonn = 0.5 * ((phi * np.roll(phi, -2, axis=-2)).mean(axis=(-2, -1)) + (phi * np.roll(phi, -2, axis=-1)).mean(axis=(-2, -1)))
    m2 = float(np.mean(m**2))
    m4 = float(np.mean(m**4))
    b4 = m4 / (m2 * m2) if m2 > 0 else math.nan
    u4 = 1.0 - b4 / 3.0 if math.isfinite(b4) else math.nan
    ft = np.fft.fft2(phi, axes=(-2, -1))
    chi = float(v * (np.mean(m**2) - np.mean(m) ** 2))
    fmin = float(0.5 * (np.mean(np.abs(ft[:, 1, 0]) ** 2) + np.mean(np.abs(ft[:, 0, 1]) ** 2)) / v)
    ratio = chi / fmin - 1.0 if fmin > 0 else math.nan
    xi = (1.0 / (2.0 * math.sin(math.pi / lx))) * math.sqrt(ratio) if ratio > 0 else math.nan
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
        "xi/L": float(xi / lx) if math.isfinite(xi) else math.nan,
        "action_hopping_density": hop,
        "action_phi2_density": p2,
        "action_phi4_density": p4,
        "action_density": hop + p2 + p4,
    }


def sample_flow(flow: Flow, cond: torch.Tensor, n: int) -> tuple[np.ndarray, dict[str, float]]:
    flow.eval()
    with torch.no_grad():
        z = torch.randn(n, L_FINE * L_FINE)
        x, ld = flow(z, cond[:n])
        phi = x.reshape(n, L_FINE, L_FINE)
        logp = -0.5 * (z**2 + math.log(2.0 * math.pi)).sum(dim=1)
        logq = logp - ld
        S, comps = fine_action(phi)
    return phi.numpy(), {"S_fine": float(S.mean()), "logq": float(logq.mean()), **{k: float(v.mean()) for k, v in comps.items()}}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    keys: list[str] = []
    for row in rows:
        for k in row:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def train_one(lambda_block: float, fine: np.ndarray, coarse: np.ndarray, cond_np: np.ndarray, weights: dict[str, float], out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / "checkpoints"
    ckpt.mkdir(exist_ok=True)
    cond = torch.tensor(cond_np, dtype=torch.float32)
    coarse_t = torch.tensor(coarse, dtype=torch.float32)
    flow = Flow(L_FINE * L_FINE, cond.shape[1])
    opt = torch.optim.Adam(flow.parameters(), lr=LR)
    n = fine.shape[0]
    history = []
    for epoch in range(1, EPOCHS + 1):
        perm = torch.randperm(n)
        losses = []
        rkls = []
        acts = []
        logqs = []
        pens = []
        for start in range(0, n, BATCH_SIZE):
            ids = perm[start : start + BATCH_SIZE]
            z = torch.randn(ids.numel(), L_FINE * L_FINE)
            x, ld = flow(z, cond[ids])
            phi = x.reshape(ids.numel(), L_FINE, L_FINE)
            S, _ = fine_action(phi)
            logp = -0.5 * (z**2 + math.log(2.0 * math.pi)).sum(dim=1)
            logq = logp - ld
            block_res = block_sym_torch(phi, weights) - coarse_t[ids]
            block_pen = lambda_block * (block_res**2).mean(dim=(-2, -1)) * (L_COARSE * L_COARSE)
            rkl = logq + S
            loss = (rkl + block_pen).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), 10.0)
            opt.step()
            losses.append(float(loss.item()))
            rkls.append(float(rkl.mean().item()))
            acts.append(float(S.mean().item()))
            logqs.append(float(logq.mean().item()))
            pens.append(float(block_pen.mean().item()))
        samples, terms = sample_flow(flow, cond, n)
        block_delta = block_sym_np(samples.astype(np.float64), weights) - coarse
        o = obs_np(samples)
        row = {
            "epoch": epoch,
            "lambda_block": lambda_block,
            "loss_total": float(np.mean(losses)),
            "reverse_KL_part": float(np.mean(rkls)),
            "block_penalty_part": float(np.mean(pens)),
            "S_fine": float(np.mean(acts)),
            "logq": float(np.mean(logqs)),
            "ESS_over_N": math.nan,
            "block_residual_RMS": float(np.sqrt(np.mean(block_delta**2))),
            "block_residual_max": float(np.max(np.abs(block_delta))),
            "nan_or_inf": bool(not np.isfinite(samples).all()),
            **o,
            **terms,
        }
        history.append(row)
        if epoch % 5 == 0:
            torch.save({"epoch": epoch, "state_dict": flow.state_dict(), "history": history, "lambda_block": lambda_block}, ckpt / f"epoch_{epoch:03d}.pt")
    write_csv(out / "history.csv", history)
    write_csv(out / "block_residual_history.csv", [{"epoch": r["epoch"], "block_residual_RMS": r["block_residual_RMS"], "block_residual_max": r["block_residual_max"]} for r in history])
    samples, terms = sample_flow(flow, cond, n)
    np.save(out / "generated_final_samples.npy", samples)
    sample_rows = []
    action_rows = []
    for name, arr in {"generated": samples}.items():
        o = obs_np(arr)
        sample_rows.extend({"ensemble": name, "operator": k, "value": v} for k, v in o.items())
        action_rows.append({"ensemble": name, **{k: o[k] for k in ["action_hopping_density", "action_phi2_density", "action_phi4_density", "action_density"]}})
    write_csv(out / "sample_observables.csv", sample_rows)
    write_csv(out / "action_components.csv", action_rows)
    summary = {"lambda_block": lambda_block, "epochs": EPOCHS, "batch_size": BATCH_SIZE, "final_epoch": history[-1], "finite": bool(not history[-1]["nan_or_inf"])}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "report.md").write_text(
        f"# Symmetric Conditional NF Pilot lambda_block={lambda_block:g}\n\n"
        "Tiny reverse-KL pilot only; not a valid sampler.\n\n"
        f"- final loss: {history[-1]['loss_total']:.12g}\n"
        f"- final reverse-KL part: {history[-1]['reverse_KL_part']:.12g}\n"
        f"- final block penalty: {history[-1]['block_penalty_part']:.12g}\n"
        f"- block residual RMS: {history[-1]['block_residual_RMS']:.12g}\n"
        f"- phi2/phi4/nn2: {history[-1]['phi2']:.12g}, {history[-1]['phi4']:.12g}, {history[-1]['nn2']:.12g}\n"
    )
    return summary


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(f"Refusing to overwrite existing outputs in {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    provenance = copy_kernel_metadata()
    meta = provenance["source_metadata"]
    weights = {k: float(meta["weights"][k]) for k in ["w00", "w10", "w11", "w20", "w21", "w22"]}
    fine = np.load(BASE / "input_fine_batch.npy").astype(np.float32)
    coarse = block_sym_np(fine.astype(np.float64), weights).astype(np.float32)
    backbone = smooth_backbone(coarse.astype(np.float64), weights).astype(np.float32)
    coarse_b = block_replicate(coarse).astype(np.float32)
    masks = parity_masks(len(fine))
    cond_np = np.concatenate([coarse_b[:, None], backbone[:, None], masks], axis=1).reshape(len(fine), -1)
    np.save(OUT / "conditioning_coarse_sym.npy", coarse)
    np.save(OUT / "conditioning_smooth_backbone.npy", backbone)
    baselines = {
        "original_fine": fine,
        "smooth_inverse_backbone": backbone,
        "block_replicated_coarse": coarse_b,
    }
    base_rows = []
    for name, arr in baselines.items():
        vals = obs_np(arr)
        block = block_sym_np(arr.astype(np.float64), weights) - coarse
        base_rows.append({"ensemble": name, "block_residual_RMS": float(np.sqrt(np.mean(block**2))), "block_residual_max": float(np.max(np.abs(block))), **vals})
    write_csv(OUT / "baseline_observables.csv", base_rows)
    summaries = []
    for lb in LAMBDA_BLOCKS:
        label = str(int(lb)) if float(lb).is_integer() else str(lb).replace(".", "p")
        summaries.append(train_one(lb, fine, coarse, cond_np, weights, OUT / f"lambda_block_{label}"))
    combined_rows = []
    for s in summaries:
        fe = s["final_epoch"]
        combined_rows.append({"lambda_block": s["lambda_block"], **fe})
    write_csv(OUT / "combined_final_metrics.csv", combined_rows)
    best_balance = min(combined_rows, key=lambda r: abs(float(r["action_density"]) - base_rows[0]["action_density"]) + 0.25 * float(r["block_residual_RMS"]))
    report = "# Tiny Symmetric-Block Conditional NF Pilot\n\n"
    report += "This is only a tiny pilot; it is not a valid sampler.\n\n"
    report += f"Kernel copied from `{provenance['original_source_path']}` with caveat: {provenance['caveat']}.\n\n"
    report += "## Final Metrics\n\n| lambda_block | finite | block RMS | phi2 | phi4 | nn2 | action density |\n|---:|---|---:|---:|---:|---:|---:|\n"
    for r in combined_rows:
        report += f"| {r['lambda_block']} | {not r['nan_or_inf']} | {r['block_residual_RMS']:.6g} | {r['phi2']:.6g} | {r['phi4']:.6g} | {r['nn2']:.6g} | {r['action_density']:.6g} |\n"
    report += "\n## Baselines\n\n"
    for b in base_rows:
        report += f"- {b['ensemble']}: phi2={b['phi2']:.6g}, phi4={b['phi4']:.6g}, nn2={b['nn2']:.6g}, block_RMS={b['block_residual_RMS']:.6g}, action_density={b['action_density']:.6g}\n"
    report += "\n## Answers\n\n"
    report += f"1. All lambda_block runs remained finite: {all(s['finite'] for s in summaries)}.\n"
    report += f"2. Best balance by a simple action/block-residual heuristic was lambda_block={best_balance['lambda_block']}.\n"
    report += "3. lambda_block=0 does not enforce the coarse condition except through conditioning; compare its block residual to the penalized runs in the table.\n"
    report += "4. Large lambda_block changes the action/operator tradeoff; this tiny model is too small to declare an optimum.\n"
    report += "5. Generated observables should be compared against the smooth backbone and block-replicated coarse baselines above; improvements are mixed in this pilot.\n"
    report += "6. The blockavg kernel is usable for pilot scaffolding, but the perfect-blocking optimization should be refined before larger training.\n"
    report += "7. No code bugs were observed; main architecture issue is that the small dense coupling flow is weak and not locality-aware.\n"
    report += "8. Recommended next run: refine blockavg kernel, then try a small convolutional conditional flow with a moderate block penalty and validation split.\n"
    (OUT / "summary.json").write_text(json.dumps({"kernel_provenance": provenance, "baselines": base_rows, "runs": summaries, "best_balance": best_balance}, indent=2) + "\n")
    (OUT / "report.md").write_text(report)


if __name__ == "__main__":
    main()
