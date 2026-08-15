#!/usr/bin/env python3
"""Tiny conditional NF pilot for inverse-blocking missing sites.

This is a pilot only: reverse-KL training on missing sites with fixed even-even
backbone sites. It is not a production sampler.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn


PROJECT = Path(__file__).resolve().parents[1]
BASE = PROJECT / "outputs" / "inverse_blocking_step_by_step"
OUT = PROJECT / "outputs" / "conditional_nf_pilot"
PREFLIGHT = OUT / "preflight"
RUN = OUT / "tiny_run"
CKPT = RUN / "checkpoints"
KERNEL_META = PROJECT / "kernels" / "from_perfect_blocking_lam1p0" / "selected_kernel_metadata.json"

L_FINE = 16
L_COARSE = 8
LAMBDA = 1.0
KAPPA_F = 0.320
KAPPA_C = 0.30
ETA_EXPONENT = 0.25
B = 2
BLOCK_NORM = B ** (ETA_EXPONENT / 2.0)
EPOCHS = 50
BATCH_SIZE = 16
LR = 1.0e-3
SEED = 20240623

SHELLS = {
    "w00": [(0, 0)],
    "w10": [(1, 0), (-1, 0), (0, 1), (0, -1)],
    "w11": [(1, 1), (1, -1), (-1, 1), (-1, -1)],
    "w20": [(2, 0), (-2, 0), (0, 2), (0, -2)],
    "w21": [
        (2, 1),
        (2, -1),
        (-2, 1),
        (-2, -1),
        (1, 2),
        (1, -2),
        (-1, 2),
        (-1, -2),
    ],
    "w22": [(2, 2), (2, -2), (-2, 2), (-2, -2)],
}
SECTORS = {
    "P00": (0, 0),
    "P10": (L_COARSE, 0),
    "P01": (0, L_COARSE),
    "P11": (L_COARSE, L_COARSE),
}


def kernel_array(weights: dict[str, float]) -> np.ndarray:
    arr = np.zeros((L_FINE, L_FINE), dtype=np.float64)
    for shell, offsets in SHELLS.items():
        for dy, dx in offsets:
            arr[dy % L_FINE, dx % L_FINE] += weights[shell]
    return arr


def apply_K_np(configs: np.ndarray, weights: dict[str, float]) -> np.ndarray:
    out = np.zeros_like(configs, dtype=np.float64)
    for shell, offsets in SHELLS.items():
        for dy, dx in offsets:
            out += weights[shell] * np.roll(np.roll(configs, -dy, axis=-2), -dx, axis=-1)
    return out


def forward_block_np(configs: np.ndarray, weights: dict[str, float]) -> np.ndarray:
    return BLOCK_NORM * apply_K_np(configs, weights)[:, 0::2, 0::2]


def fine_low_index(k: int) -> int:
    return k if k < L_COARSE // 2 else k + L_COARSE


def inverse_alias_np(coarse: np.ndarray, ktilde: np.ndarray) -> np.ndarray:
    coarse_fft = np.fft.fft2(coarse, axes=(-2, -1))
    alias_fft = np.zeros((coarse.shape[0], L_FINE, L_FINE), dtype=complex)
    for ky in range(L_COARSE):
        fy = fine_low_index(ky)
        for kx in range(L_COARSE):
            fx = fine_low_index(kx)
            corr = coarse_fft[:, ky, kx] / (BLOCK_NORM * ktilde[fy, fx])
            for ay in (0, L_COARSE):
                for ax in (0, L_COARSE):
                    alias_fft[:, (fy + ay) % L_FINE, (fx + ax) % L_FINE] = corr
    return np.fft.ifft2(alias_fft, axes=(-2, -1)).real


def block_replicate_np(a: np.ndarray) -> np.ndarray:
    out = np.empty((a.shape[0], L_FINE, L_FINE), dtype=a.dtype)
    out[:, 0::2, 0::2] = a
    out[:, 1::2, 0::2] = a
    out[:, 0::2, 1::2] = a
    out[:, 1::2, 1::2] = a
    return out


def parity_masks_np(n: int) -> np.ndarray:
    masks = np.zeros((4, L_FINE, L_FINE), dtype=np.float32)
    masks[0, 0::2, 0::2] = 1.0
    masks[1, 1::2, 0::2] = 1.0
    masks[2, 0::2, 1::2] = 1.0
    masks[3, 1::2, 1::2] = 1.0
    return np.broadcast_to(masks, (n, 4, L_FINE, L_FINE)).copy()


def missing_indices() -> tuple[np.ndarray, np.ndarray]:
    mask = np.ones((L_FINE, L_FINE), dtype=bool)
    mask[0::2, 0::2] = False
    return np.where(mask)


class Coupling(nn.Module):
    def __init__(self, dim: int, cond_dim: int, mask: torch.Tensor):
        super().__init__()
        self.register_buffer("mask", mask)
        self.net = nn.Sequential(
            nn.Linear(dim + cond_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 2 * dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def st(self, x: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.cat([x * self.mask, cond], dim=1)
        s, t = self.net(h).chunk(2, dim=1)
        active = 1.0 - self.mask
        s = 0.5 * torch.tanh(s) * active
        t = t * active
        return s, t

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        s, t = self.st(x, cond)
        y = x * self.mask + (1.0 - self.mask) * (x * torch.exp(s) + t)
        return y, s.sum(dim=1)

    def inverse(self, y: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        s, t = self.st(y, cond)
        x = y * self.mask + (1.0 - self.mask) * ((y - t) * torch.exp(-s))
        return x, -s.sum(dim=1)


class Flow(nn.Module):
    def __init__(self, dim: int, cond_dim: int, layers: int = 6):
        super().__init__()
        mods = []
        base = torch.arange(dim) % 2
        for i in range(layers):
            mask = (base if i % 2 == 0 else 1 - base).float()
            mods.append(Coupling(dim, cond_dim, mask))
        self.layers = nn.ModuleList(mods)

    def forward(self, z: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = z
        logdet = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
        for layer in self.layers:
            x, ld = layer(x, cond)
            logdet = logdet + ld
        return x, logdet

    def inverse(self, x: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = x
        logdet = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        for layer in reversed(self.layers):
            z, ld = layer.inverse(z, cond)
            logdet = logdet + ld
        return z, logdet


def fine_action_torch(phi: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    nn_term = 0.5 * (
        (phi * torch.roll(phi, -1, dims=-2)).mean(dim=(-2, -1))
        + (phi * torch.roll(phi, -1, dims=-1)).mean(dim=(-2, -1))
    )
    phi2 = (phi**2).mean(dim=(-2, -1))
    phi4 = (phi**4).mean(dim=(-2, -1))
    density = -4.0 * KAPPA_F * nn_term + (1.0 - 2.0 * LAMBDA) * phi2 + LAMBDA * phi4
    return density * (L_FINE * L_FINE), {
        "action_density": density,
        "action_hopping_density": -4.0 * KAPPA_F * nn_term,
        "action_phi2_density": (1.0 - 2.0 * LAMBDA) * phi2,
        "action_phi4_density": LAMBDA * phi4,
    }


def obs_np(configs: np.ndarray) -> dict[str, float]:
    _, ly, lx = configs.shape
    v = ly * lx
    m = configs.mean(axis=(-2, -1))
    nn = 0.5 * (
        (configs * np.roll(configs, -1, axis=-2)).mean(axis=(-2, -1))
        + (configs * np.roll(configs, -1, axis=-1)).mean(axis=(-2, -1))
    )
    nn2 = 0.5 * (
        ((configs * np.roll(configs, -1, axis=-2)) ** 2).mean(axis=(-2, -1))
        + ((configs * np.roll(configs, -1, axis=-1)) ** 2).mean(axis=(-2, -1))
    )
    diag = (configs * np.roll(np.roll(configs, -1, axis=-2), -1, axis=-1)).mean(axis=(-2, -1))
    twonn = 0.5 * (
        (configs * np.roll(configs, -2, axis=-2)).mean(axis=(-2, -1))
        + (configs * np.roll(configs, -2, axis=-1)).mean(axis=(-2, -1))
    )
    m2 = float(np.mean(m**2))
    m4 = float(np.mean(m**4))
    b4 = m4 / (m2 * m2) if m2 > 0 else math.nan
    u4 = 1.0 - b4 / 3.0 if math.isfinite(b4) else math.nan
    ft = np.fft.fft2(configs, axes=(-2, -1))
    chi = float(v * (np.mean(m**2) - np.mean(m) ** 2))
    fmin = float(0.5 * (np.mean(np.abs(ft[:, 1, 0]) ** 2) + np.mean(np.abs(ft[:, 0, 1]) ** 2)) / v)
    ratio = chi / fmin - 1.0 if fmin > 0 else math.nan
    xi = (1.0 / (2.0 * math.sin(math.pi / lx))) * math.sqrt(ratio) if ratio > 0 else math.nan
    action_hop = -4.0 * KAPPA_F * float(np.mean(nn))
    action_phi2 = (1.0 - 2.0 * LAMBDA) * float(np.mean(configs**2))
    action_phi4 = LAMBDA * float(np.mean(configs**4))
    return {
        "phi2": float(np.mean(configs**2)),
        "phi4": float(np.mean(configs**4)),
        "NN": float(np.mean(nn)),
        "nn2": float(np.mean(nn2)),
        "diag": float(np.mean(diag)),
        "2nn": float(np.mean(twonn)),
        "Binder_U4": float(u4),
        "xi/L": float(xi / lx) if math.isfinite(xi) else math.nan,
        "action_hopping_density": action_hop,
        "action_phi2_density": action_phi2,
        "action_phi4_density": action_phi4,
        "action_density": action_hop + action_phi2 + action_phi4,
    }


def alias_residual_metrics(configs: np.ndarray, weights: dict[str, float]) -> dict[str, float]:
    ktilde = np.fft.fft2(kernel_array(weights))
    psi_tilde = BLOCK_NORM * ktilde[None] * np.fft.fft2(configs, axes=(-2, -1))
    ky = np.array([fine_low_index(k) for k in range(L_COARSE)])
    kx = np.array([fine_low_index(k) for k in range(L_COARSE)])
    yy, xx = np.meshgrid(ky, kx, indexing="ij")
    vals = []
    powers = {}
    for name, (dy, dx) in SECTORS.items():
        arr = psi_tilde[:, (yy + dy) % L_FINE, (xx + dx) % L_FINE]
        vals.append(arr)
        powers[name] = float(np.mean(np.abs(arr) ** 2))
    s = 0.25 * sum(vals)
    residuals = [v - s for v in vals]
    total = sum(powers.values())
    return {
        "alias_residual_sum_zero_error": float(np.max(np.abs(sum(residuals)))),
        "alias_power_frac_P00": powers["P00"] / total,
        "alias_power_frac_P10": powers["P10"] / total,
        "alias_power_frac_P01": powers["P01"] / total,
        "alias_power_frac_P11": powers["P11"] / total,
        "alias_residual_power": float(sum(np.mean(np.abs(r) ** 2) for r in residuals)),
    }


def make_full(missing: torch.Tensor, chi: torch.Tensor, miss_y: np.ndarray, miss_x: np.ndarray) -> torch.Tensor:
    full = chi.clone()
    full[:, miss_y, miss_x] = missing
    return full


def generated_samples(
    flow: Flow,
    cond: torch.Tensor,
    init_missing: torch.Tensor,
    chi: torch.Tensor,
    miss_y: np.ndarray,
    miss_x: np.ndarray,
    n_samples: int | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    flow.eval()
    with torch.no_grad():
        if n_samples is None:
            n_samples = cond.shape[0]
        cond_s = cond[:n_samples]
        init_s = init_missing[:n_samples]
        chi_s = chi[:n_samples]
        z = torch.randn(n_samples, init_missing.shape[1], dtype=cond.dtype)
        residual, logdet = flow(z, cond_s)
        missing = init_s + residual
        full = make_full(missing, chi_s, miss_y, miss_x)
        logp = -0.5 * (z**2 + math.log(2.0 * math.pi)).sum(dim=1)
        logq = logp - logdet
        action, comps = fine_action_torch(full)
    return full.cpu().numpy(), {
        "S_fine": float(action.mean().item()),
        "logq": float(logq.mean().item()),
        "loss": float((logq + action).mean().item()),
        **{k: float(v.mean().item()) for k, v in comps.items()},
    }


def write_metrics_row(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(f"Refusing to overwrite existing outputs in {OUT}")
    PREFLIGHT.mkdir(parents=True, exist_ok=True)
    RUN.mkdir(parents=True, exist_ok=True)
    CKPT.mkdir(parents=True, exist_ok=True)

    meta = json.loads(KERNEL_META.read_text())
    weights = {k: float(meta["weights"][k]) for k in ["w00", "w10", "w11", "w20", "w21", "w22"]}
    ktilde = np.fft.fft2(kernel_array(weights))
    fine = np.load(BASE / "input_fine_batch.npy").astype(np.float32)
    coarse = forward_block_np(fine.astype(np.float64), weights).astype(np.float32)
    chi_alias = inverse_alias_np(coarse.astype(np.float64), ktilde).astype(np.float32)
    phi_init = block_replicate_np(chi_alias[:, 0::2, 0::2]).astype(np.float32)
    coarse_broadcast = block_replicate_np(coarse).astype(np.float32)
    masks = parity_masks_np(fine.shape[0])
    cond_np = np.concatenate(
        [
            chi_alias[:, None],
            phi_init[:, None],
            coarse_broadcast[:, None],
            masks,
        ],
        axis=1,
    ).reshape(fine.shape[0], -1)
    miss_y, miss_x = missing_indices()
    init_missing_np = phi_init[:, miss_y, miss_x]
    chi_t = torch.tensor(chi_alias, dtype=torch.float32)
    cond_t = torch.tensor(cond_np, dtype=torch.float32)
    init_missing_t = torch.tensor(init_missing_np, dtype=torch.float32)
    dim = init_missing_t.shape[1]
    flow = Flow(dim=dim, cond_dim=cond_t.shape[1])

    # Preflight.
    z = torch.randn(8, dim)
    cond8 = cond_t[:8]
    residual, ld = flow(z, cond8)
    z_back, inv_ld = flow.inverse(residual, cond8)
    missing = init_missing_t[:8] + residual
    full = make_full(missing, chi_t[:8], miss_y, miss_x)
    action, _ = fine_action_torch(full)
    logp = -0.5 * (z**2 + math.log(2.0 * math.pi)).sum(dim=1)
    logq = logp - ld
    loss = logq + action
    gen_np = full.detach().cpu().numpy()
    block_res = forward_block_np(gen_np.astype(np.float64), weights) - coarse[:8]
    alias_metrics = alias_residual_metrics(gen_np, weights)
    preflight = {
        "shape_device_dtype": {
            "fine_shape": list(fine.shape),
            "coarse_shape": list(coarse.shape),
            "condition_shape": list(cond_np.shape),
            "missing_dim": int(dim),
            "device": "cpu",
            "dtype": "float32",
        },
        "invertibility_max_abs_z_error": float(torch.max(torch.abs(z_back - z)).item()),
        "logJ_sign_max_abs_sum": float(torch.max(torch.abs(ld + inv_ld)).item()),
        "fixed_site_violation": float(torch.max(torch.abs(full[:, 0::2, 0::2] - chi_t[:8, 0::2, 0::2])).item()),
        "finite_checks": {
            "S_fine_finite": bool(torch.isfinite(action).all().item()),
            "logq_finite": bool(torch.isfinite(logq).all().item()),
            "loss_finite": bool(torch.isfinite(loss).all().item()),
            "S_fine_mean": float(action.mean().item()),
            "logq_mean": float(logq.mean().item()),
            "loss_mean": float(loss.mean().item()),
        },
        "forward_block_residual_rms": float(np.sqrt(np.mean(block_res**2))),
        "forward_block_residual_max_abs": float(np.max(np.abs(block_res))),
        **alias_metrics,
    }
    preflight_pass = (
        preflight["invertibility_max_abs_z_error"] < 1.0e-5
        and preflight["logJ_sign_max_abs_sum"] < 1.0e-5
        and preflight["fixed_site_violation"] < 1.0e-7
        and all(preflight["finite_checks"][k] for k in ["S_fine_finite", "logq_finite", "loss_finite"])
        and preflight["alias_residual_sum_zero_error"] < 1.0e-5
    )
    preflight["preflight_pass"] = bool(preflight_pass)
    (PREFLIGHT / "preflight_summary.json").write_text(json.dumps(preflight, indent=2) + "\n")
    (PREFLIGHT / "preflight_report.md").write_text(
        "# Conditional NF Pilot Preflight\n\n"
        f"- preflight pass: {preflight_pass}\n"
        f"- missing dimension: {dim}\n"
        f"- invertibility max |z_back-z|: {preflight['invertibility_max_abs_z_error']:.12g}\n"
        f"- logJ sign max |ld+inv_ld|: {preflight['logJ_sign_max_abs_sum']:.12g}\n"
        f"- fixed-site violation: {preflight['fixed_site_violation']:.12g}\n"
        f"- finite S/logq/loss: {preflight['finite_checks']}\n"
        f"- forward-block residual RMS: {preflight['forward_block_residual_rms']:.12g}\n"
        f"- alias residual sum-zero error: {preflight['alias_residual_sum_zero_error']:.12g}\n"
    )
    if not preflight_pass:
        (RUN / "report.md").write_text("Preflight failed; training was not run.\n")
        return

    opt = torch.optim.Adam(flow.parameters(), lr=LR)
    history = []
    n = fine.shape[0]
    for epoch in range(1, EPOCHS + 1):
        perm = torch.randperm(n)
        epoch_losses = []
        epoch_actions = []
        epoch_logqs = []
        for start in range(0, n, BATCH_SIZE):
            ids = perm[start : start + BATCH_SIZE]
            bsz = ids.numel()
            z = torch.randn(bsz, dim)
            residual, logdet = flow(z, cond_t[ids])
            missing = init_missing_t[ids] + residual
            full = make_full(missing, chi_t[ids], miss_y, miss_x)
            action, _ = fine_action_torch(full)
            logp = -0.5 * (z**2 + math.log(2.0 * math.pi)).sum(dim=1)
            logq = logp - logdet
            loss = (logq + action).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), 10.0)
            opt.step()
            epoch_losses.append(float(loss.item()))
            epoch_actions.append(float(action.mean().item()))
            epoch_logqs.append(float(logq.mean().item()))
        samples, sample_terms = generated_samples(flow, cond_t, init_missing_t, chi_t, miss_y, miss_x)
        o = obs_np(samples)
        alias = alias_residual_metrics(samples, weights)
        block_res_all = forward_block_np(samples.astype(np.float64), weights) - coarse
        fixed = float(np.max(np.abs(samples[:, 0::2, 0::2] - chi_alias[:, 0::2, 0::2])))
        row = {
            "epoch": epoch,
            "loss": float(np.mean(epoch_losses)),
            "S_fine": float(np.mean(epoch_actions)),
            "logq": float(np.mean(epoch_logqs)),
            "ESS_over_N": math.nan,
            "fixed_site_violation": fixed,
            "forward_block_residual_rms": float(np.sqrt(np.mean(block_res_all**2))),
            **o,
            **alias,
            **sample_terms,
        }
        history.append(row)
        if epoch % 5 == 0:
            torch.save({"epoch": epoch, "state_dict": flow.state_dict(), "history": history}, CKPT / f"epoch_{epoch:03d}.pt")
    write_metrics_row(RUN / "history.csv", history)

    final_samples, final_terms = generated_samples(flow, cond_t, init_missing_t, chi_t, miss_y, miss_x)
    np.save(RUN / "generated_final_samples.npy", final_samples)
    baselines = {
        "original_fine": fine,
        "generated_tiny_nf": final_samples,
        "block_replicated_baseline": np.load(BASE / "block_replicate_fill" / "block_replicated_phi_block.npy"),
        "empirical_same_q_residual_baseline": np.load(BASE / "alias_residual_baseline" / "sampled_same_q_cross_config_reconstruction.npy"),
        "old_neighbor_filled": np.load(BASE / "neighbor_filled_init.npy"),
    }
    comparison_rows = []
    base_obs = obs_np(baselines["original_fine"])
    for name, arr in baselines.items():
        vals = obs_np(arr)
        alias = alias_residual_metrics(arr, weights)
        for k, v in {**vals, **alias}.items():
            comparison_rows.append(
                {
                    "ensemble": name,
                    "metric": k,
                    "value": v,
                    "original_fine_value": base_obs.get(k, math.nan),
                    "difference_vs_original_fine": v - base_obs[k] if k in base_obs else math.nan,
                }
            )
    write_metrics_row(RUN / "validation_comparison.csv", comparison_rows)

    def metric(name: str, key: str) -> float:
        return next(r["value"] for r in comparison_rows if r["ensemble"] == name and r["metric"] == key)

    final_summary = {
        "preflight_pass": preflight_pass,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "final_epoch": history[-1],
        "validation_key_metrics": {
            name: {k: metric(name, k) for k in ["phi2", "phi4", "nn2", "action_density", "alias_residual_power"]}
            for name in baselines
        },
    }
    (RUN / "summary.json").write_text(json.dumps(final_summary, indent=2) + "\n")
    (RUN / "report.md").write_text(
        "# Tiny Conditional NF Pilot\n\n"
        "This is only a tiny reverse-KL conditional NF pilot, not a valid sampler.\n\n"
        f"1. Did preflight pass? {preflight_pass}.\n"
        f"2. Did fixed-site conditioning hold exactly? Max final fixed-site violation was {history[-1]['fixed_site_violation']:.12g}.\n"
        f"3. Did invertibility/logJ checks pass? Invertibility error {preflight['invertibility_max_abs_z_error']:.12g}; logJ sign error {preflight['logJ_sign_max_abs_sum']:.12g}.\n"
        f"4. Did reverse-KL training stay finite? Final loss {history[-1]['loss']:.12g}, S {history[-1]['S_fine']:.12g}, logq {history[-1]['logq']:.12g}.\n"
        "5. Did generated phi2/phi4/nn2 improve relative to block replication?\n\n"
        f"- generated: phi2 {metric('generated_tiny_nf','phi2'):.12g}, phi4 {metric('generated_tiny_nf','phi4'):.12g}, nn2 {metric('generated_tiny_nf','nn2'):.12g}\n"
        f"- block replication: phi2 {metric('block_replicated_baseline','phi2'):.12g}, phi4 {metric('block_replicated_baseline','phi4'):.12g}, nn2 {metric('block_replicated_baseline','nn2'):.12g}\n"
        f"- original fine: phi2 {metric('original_fine','phi2'):.12g}, phi4 {metric('original_fine','phi4'):.12g}, nn2 {metric('original_fine','nn2'):.12g}\n\n"
        f"6. Did alias residual power become physically nontrivial? Generated alias residual power is {metric('generated_tiny_nf','alias_residual_power'):.12g}; block replication is {metric('block_replicated_baseline','alias_residual_power'):.12g}.\n"
        f"7. Compared with empirical same-q residual baseline: empirical phi4 {metric('empirical_same_q_residual_baseline','phi4'):.12g}, nn2 {metric('empirical_same_q_residual_baseline','nn2'):.12g}, alias residual power {metric('empirical_same_q_residual_baseline','alias_residual_power'):.12g}.\n"
        "8. Remaining distortions: this reverse-KL pilot is tiny and unconstrained beyond fixed even-even sites; forward-block residuals and action/operator mismatches remain diagnostic quantities, not acceptance criteria.\n"
        "9. Code added: `scripts/conditional_nf_pilot.py`, preflight outputs, tiny-run history/checkpoints/validation files.\n"
        "10. Next: add explicit conditioning on alias residual statistics or a better constrained architecture before scaling training.\n"
    )


if __name__ == "__main__":
    main()
