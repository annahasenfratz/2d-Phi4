#!/usr/bin/env python3
"""Heidelberg-style native kappa=0.295 -> L16 adapter/preflight.

This script is intentionally experiment-local.  It reuses the Heidelberg
Torch CNF implementation, but loads external native L8 coarse fields instead
of generating coarse Langevin fields inside the old reproduction scripts.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
BRANCH = Path(__file__).resolve().parent
HEIDELBERG = ROOT / "heidelberg-phi4-reproduction"
HEIDELBERG_SCRIPTS = HEIDELBERG / "scripts"
sys.path.insert(0, str(HEIDELBERG))
sys.path.insert(0, str(HEIDELBERG_SCRIPTS))

from heidelberg_phi4.cnf_architecture import block_average, constrained_block_noise, naive_upsample
from train_ir_matching_l8_torch_cnf import (  # noqa: E402
    PaperCNF,
    ess_fraction,
    make_unit_noise,
    naive_upsample_torch,
    noise_logpdf,
    torch_action,
)


DEFAULT_COARSE = (
    ROOT
    / "InverseBlocking_MIT_NF/outputs/coarse_distribution_calibration/generated_native_wolff/"
    / "native_coarse_lam1_kappa0p295_L8_wolff.npy"
)
DEFAULT_FINE = ROOT / "InverseBlocking_MIT_NF/outputs/paired_data_lam1_kappaf0p320/fine_configs.npy"


def current_action_np(phi: np.ndarray, kappa: float, lam: float) -> np.ndarray:
    arr = np.asarray(phi, dtype=np.float64)
    onsite = np.sum(arr**2 + lam * (arr**2 - 1.0) ** 2, axis=(-2, -1))
    hop = np.zeros(arr.shape[0], dtype=np.float64)
    for axis in (-2, -1):
        hop += np.sum(arr * np.roll(arr, -1, axis=axis), axis=(-2, -1))
    return onsite - 2.0 * kappa * hop


def local_ops(configs: np.ndarray, kappa: float, lam: float) -> dict[str, float]:
    arr = np.asarray(configs, dtype=np.float64)
    n, lx, ly = arr.shape
    vol = lx * ly
    m_cfg = np.mean(arr, axis=(-2, -1))
    m2 = np.mean(m_cfg**2)
    m4 = np.mean(m_cfg**4)
    binder_ratio = m4 / (m2 * m2) if m2 > 0 else np.nan
    binder_u4 = 1.0 - binder_ratio / 3.0 if np.isfinite(binder_ratio) else np.nan

    nn_terms = []
    nn2_terms = []
    two_terms = []
    diag_terms = []
    for axis in (-2, -1):
        prod = arr * np.roll(arr, -1, axis=axis)
        nn_terms.append(np.mean(prod, axis=(-2, -1)))
        nn2_terms.append(np.mean(prod * prod, axis=(-2, -1)))
        two_terms.append(np.mean(arr * np.roll(arr, -2, axis=axis), axis=(-2, -1)))
    diag_terms.append(np.mean(arr * np.roll(np.roll(arr, -1, axis=-2), -1, axis=-1), axis=(-2, -1)))
    diag_terms.append(np.mean(arr * np.roll(np.roll(arr, -1, axis=-2), 1, axis=-1), axis=(-2, -1)))

    fft = np.fft.fftn(arr, axes=(-2, -1))
    chi = vol * np.mean(m_cfg**2)
    fpx = np.mean(np.abs(fft[:, 1, 0]) ** 2) / vol
    fpy = np.mean(np.abs(fft[:, 0, 1]) ** 2) / vol
    fmin = 0.5 * (fpx + fpy)
    xi = np.nan
    if fmin > 0 and chi / fmin > 1.0:
        xi = float(0.5 / np.sin(np.pi / lx) * np.sqrt(chi / fmin - 1.0))

    action = current_action_np(arr, kappa=kappa, lam=lam)
    return {
        "n_cfg": int(n),
        "L": int(lx),
        "m": float(np.mean(m_cfg)),
        "abs_m": float(np.mean(np.abs(m_cfg))),
        "phi2": float(np.mean(arr**2)),
        "phi4": float(np.mean(arr**4)),
        "NN": float(np.mean(np.stack(nn_terms, axis=0))),
        "nn2": float(np.mean(np.stack(nn2_terms, axis=0))),
        "diag": float(np.mean(np.stack(diag_terms, axis=0))),
        "2nn": float(np.mean(np.stack(two_terms, axis=0))),
        "Binder_U4": float(binder_u4),
        "Binder_B4": float(binder_ratio),
        "xi": float(xi),
        "xi_over_L": float(xi / lx) if np.isfinite(xi) else np.nan,
        "action_density_current": float(np.mean(action) / vol),
        "action_density_paper": float(np.mean(action - lam * vol) / vol),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def make_init_ensemble(coarse: np.ndarray, sigma: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    out = []
    noises = []
    for cfg in coarse:
        up = naive_upsample(cfg, block=2)
        noise = constrained_block_noise(up.shape, sigma=sigma, block=2, rng=rng)
        out.append(up + noise)
        noises.append(noise)
    return np.asarray(out), np.asarray(noises)


def block_average_errors(init: np.ndarray, coarse: np.ndarray) -> dict[str, float]:
    recovered = np.stack([block_average(cfg, block=2) for cfg in init])
    diff = recovered - coarse
    return {
        "max_abs_block_average_error": float(np.max(np.abs(diff))),
        "rms_block_average_error": float(np.sqrt(np.mean(diff * diff))),
    }


def finite_torch_preflight(
    coarse: np.ndarray,
    args: argparse.Namespace,
    out_dir: Path,
) -> dict:
    device = torch.device("cpu")
    dtype = torch.float64 if args.float64 else torch.float32
    torch.manual_seed(args.seed + 99)
    model = PaperCNF(
        kernel_radius=args.kernel_radius,
        n_field_features=args.field_features,
        n_time_features=args.time_features,
        field_bond_dim=args.field_bond_dim,
        time_bond_dim=args.time_bond_dim,
        init_sigma=args.init_sigma,
        init_weight_scale=args.init_weight_scale,
        init_scale_flow=args.init_scale_flow,
        block_period=2,
    ).to(device=device, dtype=dtype)
    batch_np = coarse[: min(args.batch_size, len(coarse))]
    batch = torch.tensor(batch_np, device=device, dtype=dtype)
    coarse_action = torch_action(batch, args.kappa_c, args.lam).detach()
    sigma = torch.exp(torch.clamp(model.log_sigma, -4.0, 3.0))
    unit_noise = make_unit_noise(batch.shape[0], args.target_L, device, dtype)
    base = naive_upsample_torch(batch) + sigma * unit_noise
    phi, logdet = model(base, n_steps=args.cnf_steps)
    target_action = torch_action(phi, args.kappa_f, args.lam)
    logdet_u_term = -0.5 * (args.coarse_L * args.coarse_L) * np.log(4.0)
    logq = -coarse_action + logdet_u_term + noise_logpdf(unit_noise, sigma) - logdet
    loss = torch.mean(target_action + logq)
    logw = (-target_action - logq).detach().cpu().numpy()
    ckpt = out_dir / "preflight_model_initial.pt"
    torch.save(model.state_dict(), ckpt)
    return {
        "model_initialized": True,
        "reverse_available": False,
        "reverse_note": "PaperCNF implementation exposes forward Euler map and analytic divergence; no inverse method is implemented.",
        "dtype": str(dtype).replace("torch.", ""),
        "batch_size": int(batch.shape[0]),
        "sigma_initial": float(sigma.detach().cpu()),
        "fine_action_finite": bool(torch.isfinite(target_action).all().item()),
        "logdet_finite": bool(torch.isfinite(logdet).all().item()),
        "logq_finite": bool(torch.isfinite(logq).all().item()),
        "loss_finite": bool(torch.isfinite(loss).item()),
        "loss": float(loss.detach().cpu()),
        "fine_action_mean_paper": float(torch.mean(target_action).detach().cpu()),
        "logq_mean": float(torch.mean(logq).detach().cpu()),
        "logdet_mean": float(torch.mean(logdet).detach().cpu()),
        "ess_over_n_initial": float(ess_fraction(logw)),
        "checkpoint": str(ckpt),
    }


def evaluate_model_samples(
    model: PaperCNF,
    coarse: torch.Tensor,
    coarse_action: torch.Tensor,
    args: argparse.Namespace,
    step: int,
    dtype: torch.dtype,
) -> dict:
    with torch.no_grad():
        n = min(args.eval_samples, coarse.shape[0])
        batch = coarse[:n]
        sigma = torch.exp(torch.clamp(model.log_sigma, -4.0, 3.0))
        torch.manual_seed(args.seed + 100000 + step)
        unit_noise = make_unit_noise(n, args.target_L, coarse.device, dtype)
        base = naive_upsample_torch(batch) + sigma * unit_noise
        phi, logdet = model(base, n_steps=args.cnf_steps)
        target_action = torch_action(phi, args.kappa_f, args.lam)
        logdet_u_term = -0.5 * (args.coarse_L * args.coarse_L) * np.log(4.0)
        logq = -coarse_action[:n] + logdet_u_term + noise_logpdf(unit_noise, sigma) - logdet
        logw = (-target_action - logq).detach().cpu().numpy()
        samples = phi.detach().cpu().numpy()
        recovered = samples.reshape(n, args.coarse_L, 2, args.coarse_L, 2).mean(axis=(2, 4))
        diff = recovered - batch.detach().cpu().numpy()
        obs = local_ops(samples, kappa=args.kappa_f, lam=args.lam)
        obs.update(
            {
                "step": int(step),
                "loss": float(torch.mean(target_action + logq).detach().cpu()),
                "S_fine_paper": float(torch.mean(target_action).detach().cpu()),
                "S_fine_current_density": obs["action_density_current"],
                "logq": float(torch.mean(logq).detach().cpu()),
                "logdet": float(torch.mean(logdet).detach().cpu()),
                "sigma": float(sigma.detach().cpu()),
                "ess_over_n": float(ess_fraction(logw)),
                "logw_std": float(np.std(logw)),
                "block_average_rms": float(np.sqrt(np.mean(diff * diff))),
                "block_average_max": float(np.max(np.abs(diff))),
            }
        )
        return obs


def tiny_pilot(coarse_np: np.ndarray, args: argparse.Namespace, pilot_dir: Path) -> dict:
    pilot_dir.mkdir(parents=True, exist_ok=True)
    (pilot_dir / "checkpoints").mkdir(exist_ok=True)
    device = torch.device("cpu")
    dtype = torch.float64 if args.float64 else torch.float32
    torch.manual_seed(args.seed + 202)
    coarse = torch.tensor(coarse_np, device=device, dtype=dtype)
    coarse_action = torch_action(coarse, args.kappa_c, args.lam).detach()
    model = PaperCNF(
        kernel_radius=args.kernel_radius,
        n_field_features=args.field_features,
        n_time_features=args.time_features,
        field_bond_dim=args.field_bond_dim,
        time_bond_dim=args.time_bond_dim,
        init_sigma=args.init_sigma,
        init_weight_scale=args.init_weight_scale,
        init_scale_flow=args.init_scale_flow,
        block_period=2,
    ).to(device=device, dtype=dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.8, 0.9))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
    rng = np.random.default_rng(args.seed + 303)
    history: list[dict] = []
    sample_rows: list[dict] = []
    start = time.time()

    initial = evaluate_model_samples(model, coarse, coarse_action, args, -1, dtype)
    sample_rows.append({"label": "initial_model", **initial})

    for step in range(args.train_steps):
        idx_np = rng.choice(len(coarse_np), size=min(args.batch_size, len(coarse_np)), replace=False)
        idx = torch.tensor(idx_np, device=device)
        batch = coarse.index_select(0, idx)
        batch_coarse_action = coarse_action.index_select(0, idx)
        sigma = torch.exp(torch.clamp(model.log_sigma, -4.0, 3.0))
        unit_noise = make_unit_noise(len(idx_np), args.target_L, device, dtype)
        base = naive_upsample_torch(batch) + sigma * unit_noise
        phi, logdet = model(base, n_steps=args.cnf_steps)
        target_action = torch_action(phi, args.kappa_f, args.lam)
        logdet_u_term = -0.5 * (args.coarse_L * args.coarse_L) * np.log(4.0)
        logq = -batch_coarse_action + logdet_u_term + noise_logpdf(unit_noise, sigma) - logdet
        loss = torch.mean(target_action + logq)
        optimizer.zero_grad()
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip).detach().cpu())
        optimizer.step()
        scheduler.step()
        with torch.no_grad():
            model.log_sigma.clamp_(np.log(args.min_sigma), np.log(args.max_sigma))
            model.wtilde.clamp_(-args.wtilde_clip, args.wtilde_clip)
            model.wk.clamp_(-args.factor_clip, args.factor_clip)
            model.wh.clamp_(-args.factor_clip, args.factor_clip)
            model.log_omega.clamp_(-2.0, 2.0)
        row = {
            "step": int(step),
            "loss": float(loss.detach().cpu()),
            "S_fine_paper": float(torch.mean(target_action).detach().cpu()),
            "logq": float(torch.mean(logq).detach().cpu()),
            "logdet": float(torch.mean(logdet).detach().cpu()),
            "sigma": float(torch.exp(torch.clamp(model.log_sigma, -4.0, 3.0)).detach().cpu()),
            "grad_norm": grad_norm,
            "finite": bool(torch.isfinite(loss).item()),
        }
        history.append(row)
        if step % args.checkpoint_every == 0 or step == args.train_steps - 1:
            torch.save(model.state_dict(), pilot_dir / "checkpoints" / f"model_step_{step:04d}.pt")
            eval_row = evaluate_model_samples(model, coarse, coarse_action, args, step, dtype)
            sample_rows.append({"label": f"step_{step}", **eval_row})

    write_csv(pilot_dir / "history.csv", history)
    write_csv(pilot_dir / "sample_observables.csv", sample_rows)
    final = sample_rows[-1]
    summary = {
        "status": "complete",
        "wall_time_sec": time.time() - start,
        "train_steps": args.train_steps,
        "batch_size": args.batch_size,
        "eval_samples": args.eval_samples,
        "lambda": args.lam,
        "kappa_c": args.kappa_c,
        "kappa_f": args.kappa_f,
        "coarse_L": args.coarse_L,
        "target_L": args.target_L,
        "initial_sigma": args.init_sigma,
        "final": final,
    }
    (pilot_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def markdown_table(rows: list[dict], cols: list[str]) -> str:
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in rows:
        vals = []
        for col in cols:
            val = row.get(col, "")
            if isinstance(val, float):
                vals.append(f"{val:.6g}")
            else:
                vals.append(str(val))
        out.append("| " + " | ".join(vals) + " |")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coarse-npy", type=Path, default=DEFAULT_COARSE)
    parser.add_argument("--fine-target-npy", type=Path, default=DEFAULT_FINE)
    parser.add_argument("--subset", type=int, default=256)
    parser.add_argument("--eval-samples", type=int, default=128)
    parser.add_argument("--lambda", dest="lam", type=float, default=1.0)
    parser.add_argument("--kappa-c", type=float, default=0.295)
    parser.add_argument("--kappa-f", type=float, default=0.320)
    parser.add_argument("--coarse-L", type=int, default=8)
    parser.add_argument("--target-L", type=int, default=16)
    parser.add_argument("--sigma-grid", default="0.10,0.15,0.20")
    parser.add_argument("--init-sigma", type=float, default=0.15)
    parser.add_argument("--min-sigma", type=float, default=0.05)
    parser.add_argument("--max-sigma", type=float, default=0.50)
    parser.add_argument("--train-steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--cnf-steps", type=int, default=4)
    parser.add_argument("--kernel-radius", type=int, default=1)
    parser.add_argument("--field-features", type=int, default=5)
    parser.add_argument("--time-features", type=int, default=5)
    parser.add_argument("--field-bond-dim", type=int, default=6)
    parser.add_argument("--time-bond-dim", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=5.0e-4)
    parser.add_argument("--lr-gamma", type=float, default=0.997)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--wtilde-clip", type=float, default=0.4)
    parser.add_argument("--factor-clip", type=float, default=1.2)
    parser.add_argument("--init-weight-scale", type=float, default=1.0e-3)
    parser.add_argument("--init-scale-flow", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260624)
    parser.add_argument("--float64", action="store_true")
    parser.add_argument("--skip-pilot", action="store_true")
    args = parser.parse_args()

    if args.target_L != 2 * args.coarse_L:
        raise SystemExit("target-L must be 2*coarse-L")

    preflight_dir = BRANCH / "outputs/preflight"
    pilot_dir = BRANCH / "outputs/tiny_pilot_kappaf0p320_sigma0p15"
    preflight_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__), preflight_dir / Path(__file__).name)

    coarse_all = np.load(args.coarse_npy)
    fine_all = np.load(args.fine_target_npy)
    if coarse_all.ndim != 3 or coarse_all.shape[1:] != (args.coarse_L, args.coarse_L):
        raise ValueError(f"expected coarse shape (N,{args.coarse_L},{args.coarse_L}), got {coarse_all.shape}")
    if fine_all.ndim != 3 or fine_all.shape[1:] != (args.target_L, args.target_L):
        raise ValueError(f"expected fine shape (N,{args.target_L},{args.target_L}), got {fine_all.shape}")

    n = min(args.subset, len(coarse_all))
    coarse = np.asarray(coarse_all[:n], dtype=np.float64)
    fine_target = np.asarray(fine_all[: min(n, len(fine_all))], dtype=np.float64)
    sigmas = [float(x) for x in args.sigma_grid.split(",") if x.strip()]

    block_rows = []
    obs_rows = []
    action_rows = []
    for sigma in sigmas:
        init, _noise = make_init_ensemble(coarse, sigma=sigma, seed=args.seed + int(round(1000 * sigma)))
        err = block_average_errors(init, coarse)
        block_rows.append({"ensemble": "native_coarse_zero_sum_init", "sigma": sigma, **err})
        obs = local_ops(init, kappa=args.kappa_f, lam=args.lam)
        obs_rows.append({"ensemble": "native_coarse_zero_sum_init", "sigma": sigma, **obs})
        action_rows.append(
            {
                "ensemble": "native_coarse_zero_sum_init",
                "sigma": sigma,
                "action_density_current": obs["action_density_current"],
                "action_density_paper": obs["action_density_paper"],
            }
        )

    fine_obs = local_ops(fine_target, kappa=args.kappa_f, lam=args.lam)
    obs_rows.insert(0, {"ensemble": "canonical_fine_target", "sigma": "", **fine_obs})
    action_rows.insert(
        0,
        {
            "ensemble": "canonical_fine_target",
            "sigma": "",
            "action_density_current": fine_obs["action_density_current"],
            "action_density_paper": fine_obs["action_density_paper"],
        },
    )

    torch_preflight = finite_torch_preflight(coarse, args, preflight_dir)
    all_preflight_ok = (
        all(r["max_abs_block_average_error"] < 1.0e-12 for r in block_rows)
        and torch_preflight["fine_action_finite"]
        and torch_preflight["logq_finite"]
        and torch_preflight["logdet_finite"]
        and torch_preflight["loss_finite"]
    )

    write_csv(preflight_dir / "init_observables.csv", obs_rows)
    write_csv(preflight_dir / "block_average_check.csv", block_rows)
    write_csv(preflight_dir / "action_check.csv", action_rows)

    summary = {
        "status": "preflight_passed" if all_preflight_ok else "preflight_failed",
        "branch_directory": str(BRANCH),
        "coarse_npy": str(args.coarse_npy),
        "fine_target_npy": str(args.fine_target_npy),
        "metadata_caveat": "Existing kappa=0.295 native coarse ensemble used embedded Wolff sign-cluster plus local Metropolis amplitude updates; it is not Wolff-only.",
        "coarse_shape_full": list(coarse_all.shape),
        "fine_shape_full": list(fine_all.shape),
        "subset_used": int(n),
        "coarse_shape_used": list(coarse.shape),
        "fine_target_shape_used": list(fine_target.shape),
        "lambda": args.lam,
        "kappa_c": args.kappa_c,
        "kappa_f": args.kappa_f,
        "coarse_L": args.coarse_L,
        "target_L": args.target_L,
        "sigma_grid": sigmas,
        "action_convention_note": "Training uses Heidelberg paper action S_paper. Reported current action density adds lambda as S_current/V = S_paper/V + lambda; this constant does not affect gradients or log weights at fixed volume.",
        "block_average_checks": block_rows,
        "torch_cnf_preflight": torch_preflight,
    }

    pilot_summary = None
    if all_preflight_ok and not args.skip_pilot:
        pilot_summary = tiny_pilot(coarse, args, pilot_dir)
        summary["tiny_pilot"] = pilot_summary
        shutil.copy2(Path(__file__), pilot_dir / Path(__file__).name)

    (preflight_dir / "preflight_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))

    report_lines = [
        "# Heidelberg Native kappa=0.295 -> 16 Preflight",
        "",
        f"Status: **{summary['status']}**",
        "",
        "## Inputs",
        "",
        f"- Coarse file: `{args.coarse_npy}`",
        f"- Fine target file: `{args.fine_target_npy}`",
        f"- Coarse full shape: `{tuple(coarse_all.shape)}`",
        f"- Fine full shape: `{tuple(fine_all.shape)}`",
        f"- Subset used: `{n}`",
        "- Metadata caveat: existing native kappa=0.295 coarse ensemble used embedded Wolff sign-cluster plus local Metropolis amplitude updates, not Wolff-only.",
        "",
        "## Convention",
        "",
        "The Heidelberg trainer uses the paper action `S_paper = sum[(1-2 lambda) phi^2 + lambda phi^4 - 2 kappa hopping]`.",
        "Our current action differs by the constant `lambda * V`, so fixed-volume training weights and gradients are unchanged. Tables report both paper and current action densities where relevant.",
        "",
        "## Block-Average Preservation",
        "",
        markdown_table(block_rows, ["ensemble", "sigma", "max_abs_block_average_error", "rms_block_average_error"]),
        "",
        "## Initial Observable Scan",
        "",
        markdown_table(
            obs_rows,
            ["ensemble", "sigma", "phi2", "phi4", "NN", "nn2", "diag", "2nn", "Binder_U4", "xi_over_L", "action_density_current"],
        ),
        "",
        "## CNF / Density Preflight",
        "",
        markdown_table(
            [torch_preflight],
            ["model_initialized", "reverse_available", "sigma_initial", "fine_action_finite", "logdet_finite", "logq_finite", "loss_finite", "loss", "ess_over_n_initial"],
        ),
        "",
    ]
    if pilot_summary is not None:
        final = pilot_summary["final"]
        report_lines.extend(
            [
                "## Tiny Pilot",
                "",
                f"Ran `{args.train_steps}` tiny training steps with initial `sigma={args.init_sigma}`.",
                "",
                markdown_table(
                    [final],
                    ["step", "loss", "S_fine_paper", "logq", "logdet", "sigma", "ess_over_n", "phi2", "phi4", "NN", "nn2", "Binder_U4", "xi_over_L", "block_average_rms"],
                ),
                "",
            ]
        )
    else:
        report_lines.extend(["## Tiny Pilot", "", "Not run because preflight failed or `--skip-pilot` was set.", ""])

    report_lines.extend(
        [
            "## Answers",
            "",
            f"1. External `coarse_np` use: {'clean in this adapter; it loads the native `.npy` and feeds it to the Heidelberg Torch code path.' if all_preflight_ok else 'adapter load path exists, but preflight failed.'}",
            "2. Zero-sum noise preserves the simple 2x2 block average to roundoff, as shown in `block_average_check.csv`.",
            "3. Initial-field closeness should be judged from `init_observables.csv`; this preflight does not tune sigma beyond the requested small scan.",
            f"4. CNF finite action/logq/loss check: {'passed.' if all_preflight_ok else 'failed; inspect preflight_summary.json.'}",
            f"5. Tiny pilot: {'completed; inspect sample_observables.csv for whether observables improved.' if pilot_summary is not None else 'not completed.'}",
            "6. This branch is ready for a small kappa_f/sigma scan only if the tiny pilot metrics are acceptable; no full scan was run here.",
            "",
        ]
    )
    (preflight_dir / "report.md").write_text("\n".join(report_lines))

    if pilot_summary is not None:
        (pilot_dir / "report.md").write_text("\n".join(report_lines))

    print(json.dumps({"preflight_dir": str(preflight_dir), "pilot_dir": str(pilot_dir) if pilot_summary else None, "status": summary["status"]}, indent=2))


if __name__ == "__main__":
    main()

