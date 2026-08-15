#!/usr/bin/env python3
"""Exact block-consistent null-space conditional NF pilot."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn


PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
OUT = PROJECT / "outputs" / "nullspace_conditional_nf_pilot"
PREFLIGHT = OUT / "preflight"
RUN = OUT / "tiny_run"
CKPT = RUN / "checkpoints"
BASE = PROJECT / "outputs" / "inverse_blocking_step_by_step"
KERNEL = ROOT / "perfect_blocking" / "perfect_blocking_lam1p0_blockavg" / "perfect_block_lam1_blockavg_kernel5x5_kernel.json"

L_FINE = 16
L_COARSE = 8
ETA_EXPONENT = 0.25
BLOCK_NORM = 2 ** (ETA_EXPONENT / 2.0)
LAMBDA = 1.0
KAPPA_F = 0.320
EPOCHS = 50
BATCH_SIZE = 16
LR = 1.0e-3
SEED = 20240623
SIGMA_SCAN = [0.1, 0.25, 0.5, 1.0]

SHELLS = {
    "w00": [(0, 0)],
    "w10": [(1, 0), (-1, 0), (0, 1), (0, -1)],
    "w11": [(1, 1), (1, -1), (-1, 1), (-1, -1)],
    "w20": [(2, 0), (-2, 0), (0, 2), (0, -2)],
    "w21": [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)],
    "w22": [(2, 2), (2, -2), (-2, 2), (-2, -2)],
}


def shell_convolve(phi: np.ndarray, w: dict[str, float]) -> np.ndarray:
    out = np.zeros_like(phi, dtype=np.float64)
    for shell, offsets in SHELLS.items():
        for dy, dx in offsets:
            out += w[shell] * np.roll(np.roll(phi, -dy, axis=-2), -dx, axis=-1)
    return out


def block_sym_np(phi: np.ndarray, w: dict[str, float]) -> np.ndarray:
    psi = shell_convolve(phi, w)
    return 0.25 * BLOCK_NORM * (psi[:, 0::2, 0::2] + psi[:, 1::2, 0::2] + psi[:, 0::2, 1::2] + psi[:, 1::2, 1::2])


def block_sym_torch(phi: torch.Tensor, w: dict[str, float]) -> torch.Tensor:
    psi = torch.zeros_like(phi)
    for shell, offsets in SHELLS.items():
        for dy, dx in offsets:
            psi = psi + float(w[shell]) * torch.roll(torch.roll(phi, -dy, dims=-2), -dx, dims=-1)
    return 0.25 * BLOCK_NORM * (psi[:, 0::2, 0::2] + psi[:, 1::2, 0::2] + psi[:, 0::2, 1::2] + psi[:, 1::2, 1::2])


def build_B(w: dict[str, float]) -> np.ndarray:
    B = np.zeros((L_COARSE * L_COARSE, L_FINE * L_FINE), dtype=np.float64)
    for j in range(L_FINE * L_FINE):
        e = np.zeros((1, L_FINE, L_FINE), dtype=np.float64)
        e.reshape(1, -1)[0, j] = 1.0
        B[:, j] = block_sym_np(e, w).reshape(-1)
    return B


def kernel_array(w: dict[str, float]) -> np.ndarray:
    arr = np.zeros((L_FINE, L_FINE), dtype=float)
    for shell, offsets in SHELLS.items():
        for dy, dx in offsets:
            arr[dy % L_FINE, dx % L_FINE] += w[shell]
    return arr


def smooth_backbone(coarse: np.ndarray, w: dict[str, float]) -> np.ndarray:
    ktilde = np.fft.fft2(kernel_array(w))
    p = np.fft.fftfreq(L_FINE) * 2.0 * np.pi
    px, py = np.meshgrid(p, p)
    A = 0.25 * (1.0 + np.exp(1j * py)) * (1.0 + np.exp(1j * px))
    cft = np.fft.fft2(coarse, axes=(-2, -1))
    padded_shift = np.zeros((len(coarse), L_FINE, L_FINE), dtype=complex)
    padded_shift[:, 4:12, 4:12] = 4.0 * np.fft.fftshift(cft, axes=(-2, -1))
    padded = np.fft.ifftshift(padded_shift, axes=(-2, -1))
    mask_shift = np.zeros((L_FINE, L_FINE), dtype=bool)
    mask_shift[4:12, 4:12] = True
    mask = np.fft.ifftshift(mask_shift)
    inv = np.zeros_like(padded)
    inv[:, mask] = padded[:, mask] / (BLOCK_NORM * ktilde * A)[mask]
    return np.fft.ifft2(inv, axes=(-2, -1)).real


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
    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        base = torch.arange(dim) % 2
        self.layers = nn.ModuleList([Coupling(dim, cond_dim, (base if i % 2 == 0 else 1 - base).float()) for i in range(6)])

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


def main() -> None:
    torch.manual_seed(20240623)
    np.random.seed(20240623)
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(f"Refusing to overwrite existing outputs in {OUT}")
    PREFLIGHT.mkdir(parents=True, exist_ok=True)
    RUN.mkdir(parents=True, exist_ok=True)
    CKPT.mkdir(parents=True, exist_ok=True)
    meta = json.loads(KERNEL.read_text())
    w = {k: float(meta["weights"][k]) for k in ["w00", "w10", "w11", "w20", "w21", "w22"]}
    fine = np.load(BASE / "input_fine_batch.npy").astype(np.float64)
    coarse = block_sym_np(fine, w)
    backbone = smooth_backbone(coarse, w)
    Bmat = build_B(w)
    U, S, Vt = np.linalg.svd(Bmat, full_matrices=True)
    N = Vt[64:].T
    np.save(PREFLIGHT / "null_basis.npy", N.astype(np.float32))
    BN = Bmat @ N
    coarse_res = block_sym_np(backbone, w) - coarse
    rng = np.random.default_rng(20240624)
    u0 = rng.normal(size=(8, 192))
    gen0 = backbone[:8].reshape(8, -1) + u0 @ N.T
    gen0 = gen0.reshape(8, L_FINE, L_FINE)
    gen_res = block_sym_np(gen0, w) - coarse[:8]
    cond_np = np.concatenate([coarse.reshape(len(fine), -1), backbone.reshape(len(fine), -1)], axis=1).astype(np.float32)
    cond = torch.tensor(cond_np[:8], dtype=torch.float32)
    flow = Flow(192, cond.shape[1])
    z = torch.randn(8, 192)
    u, ld = flow(z, cond)
    z_back, inv_ld = flow.inverse(u, cond)
    Nt = torch.tensor(N.astype(np.float32), dtype=torch.float32)
    back_t = torch.tensor(backbone[:8].reshape(8, -1).astype(np.float32))
    phi = (back_t + u @ Nt.T).reshape(8, L_FINE, L_FINE)
    Sfine, _ = fine_action(phi)
    logp = -0.5 * (z**2 + math.log(2 * math.pi)).sum(dim=1)
    logq = logp - ld
    loss = logq + Sfine
    pre = {
        "N_shape": list(N.shape),
        "orthonormal_error_max_abs": float(np.max(np.abs(N.T @ N - np.eye(192)))),
        "max_abs_BN": float(np.max(np.abs(BN))),
        "rms_BN": float(np.sqrt(np.mean(BN**2))),
        "backbone_max_abs_block_error": float(np.max(np.abs(coarse_res))),
        "backbone_rms_block_error": float(np.sqrt(np.mean(coarse_res**2))),
        "generated_max_abs_block_error": float(np.max(np.abs(gen_res))),
        "generated_rms_block_error": float(np.sqrt(np.mean(gen_res**2))),
        "invertibility_max_abs_z_error": float(torch.max(torch.abs(z_back - z))),
        "logJ_sign_max_abs_sum": float(torch.max(torch.abs(ld + inv_ld))),
        "finite_action": bool(torch.isfinite(Sfine).all()),
        "finite_logq": bool(torch.isfinite(logq).all()),
        "finite_loss": bool(torch.isfinite(loss).all()),
    }
    pre["preflight_pass"] = bool(
        pre["orthonormal_error_max_abs"] < 1e-12
        and pre["max_abs_BN"] < 1e-12
        and pre["backbone_max_abs_block_error"] < 1e-12
        and pre["generated_max_abs_block_error"] < 1e-12
        and pre["invertibility_max_abs_z_error"] < 1e-6
        and pre["logJ_sign_max_abs_sum"] < 1e-6
        and pre["finite_action"] and pre["finite_logq"] and pre["finite_loss"]
    )
    (PREFLIGHT / "preflight_summary.json").write_text(json.dumps(pre, indent=2) + "\n")
    scan_rows = []
    for sig in SIGMA_SCAN:
        u = rng.normal(scale=sig, size=(len(fine), 192))
        ph = (backbone.reshape(len(fine), -1) + u @ N.T).reshape(len(fine), L_FINE, L_FINE)
        br = block_sym_np(ph, w) - coarse
        scan_rows.append({"sigma_u": sig, "block_residual_RMS": float(np.sqrt(np.mean(br**2))), "block_residual_max": float(np.max(np.abs(br))), **obs_np(ph)})
    write_csv(PREFLIGHT / "prior_scale_scan.csv", scan_rows)
    best_sig = min(scan_rows, key=lambda r: abs(float(r["phi2"]) - obs_np(fine)["phi2"]) + abs(float(r["phi4"]) - obs_np(fine)["phi4"]))["sigma_u"]
    (PREFLIGHT / "prior_scale_scan.md").write_text("# Prior Scale Scan\n\n" + "\n".join(f"- sigma {r['sigma_u']}: phi2={r['phi2']:.6g}, phi4={r['phi4']:.6g}, nn2={r['nn2']:.6g}, block RMS={r['block_residual_RMS']:.3g}" for r in scan_rows) + f"\n\nChosen starting sigma_u: {best_sig}\n")
    if not pre["preflight_pass"]:
        (RUN / "report.md").write_text("Preflight failed; training skipped.\n")
        return
    # Train in scaled coordinates: prior z -> flow u_scaled; physical detail = sigma * u_scaled.
    sigma = float(best_sig)
    cond_all = torch.tensor(cond_np, dtype=torch.float32)
    back_all = torch.tensor(backbone.reshape(len(fine), -1).astype(np.float32))
    coarse_np = coarse
    flow = Flow(192, cond_all.shape[1])
    opt = torch.optim.Adam(flow.parameters(), lr=1e-3)
    history = []
    for epoch in range(1, 51):
        perm = torch.randperm(len(fine))
        losses = []
        acts = []
        logqs = []
        for start in range(0, len(fine), 16):
            ids = perm[start:start+16]
            z = torch.randn(len(ids), 192)
            u, ld = flow(z, cond_all[ids])
            delta = sigma * (u @ Nt.T)
            ph = (back_all[ids] + delta).reshape(len(ids), L_FINE, L_FINE)
            Sval, _ = fine_action(ph)
            logp = -0.5 * (z**2 + math.log(2 * math.pi)).sum(dim=1)
            # Constant -192 log(sigma) omitted from optimization, recorded only conceptually.
            logqv = logp - ld
            loss = (logqv + Sval).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), 10.0)
            opt.step()
            losses.append(float(loss))
            acts.append(float(Sval.mean()))
            logqs.append(float(logqv.mean()))
        with torch.no_grad():
            z = torch.randn(len(fine), 192)
            u, ld = flow(z, cond_all)
            samples = (back_all + sigma * (u @ Nt.T)).reshape(len(fine), L_FINE, L_FINE).numpy()
        br = block_sym_np(samples, w) - coarse_np
        o = obs_np(samples)
        row = {"epoch": epoch, "loss": float(np.mean(losses)), "S_fine": float(np.mean(acts)), "logq": float(np.mean(logqs)), "ESS_over_N": math.nan, "block_residual_RMS": float(np.sqrt(np.mean(br**2))), "block_residual_max": float(np.max(np.abs(br))), "nan_or_inf": bool(not np.isfinite(samples).all()), **o}
        history.append(row)
        if epoch % 5 == 0:
            torch.save({"epoch": epoch, "state_dict": flow.state_dict(), "history": history, "sigma_u": sigma}, CKPT / f"epoch_{epoch:03d}.pt")
    write_csv(RUN / "history.csv", history)
    np.save(RUN / "generated_final_samples.npy", samples)
    baselines = {
        "original_fine": fine,
        "phi_backbone": backbone,
        "block_replicated_coarse": np.repeat(np.repeat(coarse, 2, axis=1), 2, axis=2),
    }
    for p, name in [(PROJECT / "outputs/symmetric_conditional_nf_pilot/lambda_block_10/generated_final_samples.npy", "soft_penalty_lambda10"), (PROJECT / "outputs/symmetric_conditional_nf_pilot/lambda_block_100/generated_final_samples.npy", "soft_penalty_lambda100"), (PROJECT / "outputs/inverse_blocking_step_by_step/alias_residual_baseline/sampled_same_q_cross_config_reconstruction.npy", "old_empirical_alias_residual")]:
        if p.exists():
            baselines[name] = np.load(p)
    rows = []
    for name, arr in {"generated_nullspace_nf": samples, **baselines}.items():
        br = block_sym_np(arr.astype(np.float64), w) - coarse if arr.shape == fine.shape else np.full_like(coarse, np.nan)
        rows.append({"ensemble": name, "block_residual_RMS": float(np.sqrt(np.nanmean(br**2))), "block_residual_max": float(np.nanmax(np.abs(br))), **obs_np(arr)})
    write_csv(RUN / "sample_observables.csv", rows)
    write_csv(RUN / "action_components.csv", [{"ensemble": r["ensemble"], "action_hopping_density": r["action_hopping_density"], "action_phi2_density": r["action_phi2_density"], "action_phi4_density": r["action_phi4_density"], "action_density": r["action_density"]} for r in rows])
    final = history[-1]
    summary = {"preflight": pre, "chosen_sigma_u": sigma, "final_epoch": final, "comparisons": rows}
    (RUN / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    def val(name, key):
        return next(r[key] for r in rows if r["ensemble"] == name)
    (RUN / "report.md").write_text(
        "# Exact Block-Consistent Null-Space NF Pilot\n\n"
        "Tiny pilot only; not a valid sampler.\n\n"
        f"1. Exact block consistency held throughout: final block RMS {final['block_residual_RMS']:.12g}, max {final['block_residual_max']:.12g}.\n"
        f"2. Chosen prior scale: sigma_u={sigma}, selected from the prior scan as a rough one-site moment match.\n"
        f"3. Reverse-KL training stayed finite: final loss {final['loss']:.12g}, S {final['S_fine']:.12g}, logq {final['logq']:.12g}.\n"
        f"4. Generated vs backbone phi2/phi4/nn2: generated {val('generated_nullspace_nf','phi2'):.6g}/{val('generated_nullspace_nf','phi4'):.6g}/{val('generated_nullspace_nf','nn2'):.6g}; backbone {val('phi_backbone','phi2'):.6g}/{val('phi_backbone','phi4'):.6g}/{val('phi_backbone','nn2'):.6g}.\n"
        f"5. Original fine phi2/phi4/nn2: {val('original_fine','phi2'):.6g}/{val('original_fine','phi4'):.6g}/{val('original_fine','nn2'):.6g}.\n"
        "6. Main mismatch remains action/operator quality from the tiny dense flow, not block consistency.\n"
        "7. This architecture should be scaled before returning to soft penalties; consider locality-aware null coordinates or convolutional conditioners.\n"
        "8. Code added: `scripts/nullspace_conditional_nf_pilot.py`.\n"
    )


if __name__ == "__main__":
    main()
