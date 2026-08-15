#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PKG = PROJECT_ROOT / "perfect_blocking_upsampling"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PKG / "src"))
sys.path.insert(0, str(PKG / "scripts"))

from _common import load_config, load_ensembles, load_frozen_models, load_kernel_spec, resolve_run_paths  # noqa: E402
from perfect_blocking_upsampling.actions import action_density, action_total  # noqa: E402
from perfect_blocking_upsampling.coarse_refine import apply_refine  # noqa: E402
from perfect_blocking_upsampling.gathered_edge import build_gathered_edge_from_checkpoint, corrcoef_flat  # noqa: E402
from perfect_blocking_upsampling.kernels import inverse_kernel, kernel_fft, kernel_stencil_from_spec, normalize_kernel  # noqa: E402
from train_gathered_pair_distillation import (  # noqa: E402
    logq_from_z_logdet,
    student_forward,
    teacher_forward,
    write_checksums,
    write_yaml_config,
)
from ML_sampling_clean.experiments.decimated_conditional_fillin.run_staged_decimated_conditional_fillin import (  # noqa: E402
    from_model_space,
    torch_from_model_space,
)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=float) + "\n", encoding="utf-8")


def quantiles(x: np.ndarray) -> dict[str, float]:
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    return {
        "mean": float(np.mean(a)),
        "std": float(np.std(a, ddof=1)) if a.size > 1 else 0.0,
        "rmse": float(np.sqrt(np.mean(a * a))),
        "q05": float(np.quantile(a, 0.05)),
        "q50": float(np.quantile(a, 0.50)),
        "q95": float(np.quantile(a, 0.95)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
    }


def assemble_phi_np(cprime: np.ndarray, d10: np.ndarray, d01: np.ndarray, d11: np.ndarray, kernel) -> np.ndarray:
    psi = np.empty((cprime.shape[0], 2 * cprime.shape[1], 2 * cprime.shape[2]), dtype=np.float32)
    psi[:, 0::2, 0::2] = cprime
    psi[:, 1::2, 0::2] = d10[:, 0]
    psi[:, 0::2, 1::2] = d01[:, 0]
    psi[:, 1::2, 1::2] = d11[:, 0]
    phi, _ = inverse_kernel(psi, kernel)
    return phi


def torch_kernel_fft(spec, fine_l: int, device):
    import torch

    stencil = normalize_kernel(kernel_stencil_from_spec(spec))
    kt = kernel_fft(stencil, fine_l, spec.eta)
    return torch.tensor(kt, dtype=torch.complex64, device=device)


def torch_inverse_kernel(psi, kt):
    import torch

    return torch.fft.ifft2(torch.fft.fft2(psi, dim=(-2, -1)) / kt[None, :, :], dim=(-2, -1)).real


def torch_action_density(phi, action):
    import torch

    phi2 = phi * phi
    phi4 = phi2 * phi2
    nn = phi * torch.roll(phi, shifts=-1, dims=-1) + phi * torch.roll(phi, shifts=-1, dims=-2)
    out = (1.0 - 2.0 * action.lambda_) * phi2 + action.lambda_ * phi4 - 2.0 * action.kappa * nn
    if getattr(action, "type", "") == "phi4_nn_plus_diag":
        diag = phi * torch.roll(torch.roll(phi, shifts=-1, dims=-1), shifts=-1, dims=-2)
        out = out - 2.0 * action.kappa_diag * diag
    return out


def assemble_phi_torch(cprime, d10, d01, d11, kt):
    import torch

    b, _, h, w = d01.shape
    psi = torch.empty((b, 2 * h, 2 * w), dtype=d01.dtype, device=d01.device)
    psi[:, 0::2, 0::2] = cprime[:, 0]
    psi[:, 1::2, 0::2] = d10[:, 0]
    psi[:, 0::2, 1::2] = d01[:, 0]
    psi[:, 1::2, 1::2] = d11[:, 0]
    return torch_inverse_kernel(psi, kt)


def copy_bundle(source_dir: Path, out: Path, pair_ckpt: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name in ["coarse_refine.pt", "edge.pt", "corner.pt"]:
        dst = out / name
        if dst.exists():
            dst.unlink()
        shutil.copy2(source_dir / name, dst)
    pair_dst = out / "pair.pt"
    if pair_dst.exists():
        pair_dst.unlink()
    shutil.copy2(pair_ckpt, pair_dst)
    for stage in ["edge", "pair", "corner"]:
        dst_dir = out / stage
        dst_dir.mkdir(exist_ok=True)
        dst = dst_dir / "local_gaussian_coefficients.npz"
        if dst.exists():
            dst.unlink()
        shutil.copy2(source_dir / stage / "local_gaussian_coefficients.npz", dst)
    write_checksums(out)


def build_dataset(args, cfg_pair, cfg_old, stages_pair, stages_old, refine_model, refine_state, kernel, fine_action):
    coarse, _, _, _, _ = load_ensembles(cfg_old)
    coarse = coarse[: args.max_configs].astype(np.float32)
    cprime, refine_logdet = apply_refine(refine_model, refine_state, coarse, batch_size=32)
    rng = np.random.default_rng(args.seed)
    n = min(args.train_conditions + args.val_conditions, cprime.shape[0])
    idx = rng.integers(0, cprime.shape[0], size=n)
    coarse = coarse[idx]
    cprime = cprime[idx]
    refine_logdet = refine_logdet[idx]
    h = cprime.shape[1]

    edge_model, edge_lg, edge_state = stages_pair["edge"][:3]
    pair_old_model, pair_lg, pair_old_state = stages_old["pair"][:3]
    corner_model, corner_lg, corner_state = stages_pair["corner"][:3]
    z_edge = rng.standard_normal((n, 1, h, h)).astype(np.float32)
    z_pair = rng.standard_normal((n, 1, h, h)).astype(np.float32)
    z_corner = rng.standard_normal((n, 1, h, h)).astype(np.float32)

    y_edge, ld_edge = teacher_forward(edge_model, edge_state, z_edge, cprime[:, None].astype(np.float32))
    d10 = from_model_space(y_edge, cprime[:, None].astype(np.float32), edge_lg)
    edge_logq = logq_from_z_logdet(z_edge, ld_edge, cprime[:, None].astype(np.float32), edge_lg)
    pair_cond = np.concatenate([cprime[:, None], d10], axis=1).astype(np.float32)
    y_pair_old, ld_pair_old = teacher_forward(pair_old_model, pair_old_state, z_pair, pair_cond)
    d01_old = from_model_space(y_pair_old, pair_cond, pair_lg)
    pair_logq_old = logq_from_z_logdet(z_pair, ld_pair_old, pair_cond, pair_lg)
    corner_cond = np.concatenate([cprime[:, None], d10, d01_old], axis=1).astype(np.float32)
    y_corner_old, ld_corner_old = teacher_forward(corner_model, corner_state, z_corner, corner_cond)
    d11_old = from_model_space(y_corner_old, corner_cond, corner_lg)
    corner_logq_old = logq_from_z_logdet(z_corner, ld_corner_old, corner_cond, corner_lg)
    phi_old = assemble_phi_np(cprime, d10, d01_old, d11_old, kernel)
    density_old = action_density(phi_old, fine_action).astype(np.float32)
    s_old = action_total(phi_old, fine_action)
    return {
        "coarse": coarse,
        "cprime": cprime,
        "refine_logdet": refine_logdet,
        "d10": d10,
        "edge_logq": edge_logq,
        "pair_cond": pair_cond,
        "z_pair": z_pair,
        "y_pair_old": y_pair_old,
        "ld_pair_old": ld_pair_old,
        "d01_old": d01_old,
        "pair_logq_old": pair_logq_old,
        "z_corner": z_corner,
        "d11_old": d11_old,
        "corner_logq_old": corner_logq_old,
        "phi_old": phi_old.astype(np.float32),
        "density_old": density_old,
        "s_old": s_old.astype(np.float64),
    }


def split_dataset(data: dict[str, np.ndarray], n_train: int) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    return ({k: v[:n_train] for k, v in data.items()}, {k: v[n_train:] for k, v in data.items()})


def subset(data: dict[str, np.ndarray], idx: np.ndarray) -> dict[str, np.ndarray]:
    return {k: v[idx] for k, v in data.items()}


def evaluate_model(model, data, pair_lg, kernel, coarse_action, fine_action, kt=None) -> dict[str, float]:
    y_new, ld_new = student_forward(model, data["z_pair"], data["pair_cond"])
    d01_new = from_model_space(y_new, data["pair_cond"], pair_lg)
    pair_logq_new = logq_from_z_logdet(data["z_pair"], ld_new, data["pair_cond"], pair_lg)
    phi_new = assemble_phi_np(data["cprime"], data["d10"], d01_new, data["d11_old"], kernel)
    s_new = action_total(phi_new, fine_action)
    delta_s = s_new - data["s_old"]
    delta_logq = pair_logq_new - data["pair_logq_old"]
    delta_logw = -delta_s - delta_logq
    return {
        "pair_rmse": float(np.sqrt(np.mean((d01_new - data["d01_old"]) ** 2))),
        "pair_corr": corrcoef_flat(d01_new, data["d01_old"]),
        "phi_rmse": float(np.sqrt(np.mean((phi_new - data["phi_old"]) ** 2))),
        "delta_S_std": float(np.std(delta_s, ddof=1)),
        "delta_logq_std": float(np.std(delta_logq, ddof=1)),
        "delta_logw_std": float(np.std(delta_logw, ddof=1)),
        "delta_logw_mean": float(np.mean(delta_logw)),
        "logq_rmse": float(np.sqrt(np.mean(delta_logq**2))),
    }


def train_variant(args, name: str, weights: dict[str, float], init_ckpt: dict[str, Any], train, val, pair_lg, kernel, fine_action) -> tuple[Any, dict[str, Any]]:
    import torch

    model = build_gathered_edge_from_checkpoint(init_ckpt, cond_channels=2, lattice_size=train["cprime"].shape[1])
    model.load_state_dict(init_ckpt["model_state"])
    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    kt = torch_kernel_fft(kernel, 2 * train["cprime"].shape[1], torch.device("cpu"))
    rng = np.random.default_rng(args.seed + abs(hash(name)) % 100000)
    rows = []
    best = {"score": float("inf"), "state": None, "row": None, "epoch": 0}
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        steps = max(1, math.ceil(train["cprime"].shape[0] / args.batch_size))
        for _ in range(steps):
            idx = rng.integers(0, train["cprime"].shape[0], size=args.batch_size)
            batch = subset(train, idx)
            b = idx.shape[0]
            z_t = torch.tensor(batch["z_pair"].reshape(b, -1), dtype=torch.float32)
            c_t = torch.tensor(batch["pair_cond"].reshape(b, -1), dtype=torch.float32)
            y_t = torch.tensor(batch["y_pair_old"].reshape(b, -1), dtype=torch.float32)
            ld_t = torch.tensor(batch["ld_pair_old"], dtype=torch.float32)
            y_s, ld_s = model.forward(z_t, c_t)
            loss_pair = torch.mean((y_s - y_t) ** 2)
            loss = loss_pair + weights.get("logdet", 0.0) * torch.mean((ld_s - ld_t) ** 2)
            if weights.get("phi", 0.0) or weights.get("local", 0.0) or weights.get("global_s", 0.0):
                h = batch["cprime"].shape[1]
                d01_s = torch_from_model_space(y_s, c_t, (2, h, h), pair_lg).reshape(b, 1, h, h)
                cprime_t = torch.tensor(batch["cprime"][:, None], dtype=torch.float32)
                d10_t = torch.tensor(batch["d10"], dtype=torch.float32)
                d11_t = torch.tensor(batch["d11_old"], dtype=torch.float32)
                phi_s = assemble_phi_torch(cprime_t, d10_t, d01_s, d11_t, kt)
                if weights.get("phi", 0.0):
                    phi_old_t = torch.tensor(batch["phi_old"], dtype=torch.float32)
                    loss = loss + weights["phi"] * torch.mean((phi_s - phi_old_t) ** 2)
                if weights.get("local", 0.0) or weights.get("global_s", 0.0):
                    density_s = torch_action_density(phi_s, fine_action)
                    if weights.get("local", 0.0):
                        density_old_t = torch.tensor(batch["density_old"], dtype=torch.float32)
                        loss = loss + weights["local"] * torch.mean((density_s - density_old_t) ** 2)
                    if weights.get("global_s", 0.0):
                        s_old_t = torch.tensor(batch["s_old"], dtype=torch.float32)
                        s_s = density_s.flatten(1).sum(dim=1)
                        loss = loss + weights["global_s"] * torch.mean((s_s - s_old_t) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        metrics = evaluate_model(model, val, pair_lg, kernel, None, fine_action)
        score = metrics["delta_logw_std"]
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), **metrics}
        rows.append(row)
        if score < best["score"]:
            best = {"score": score, "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, "row": row, "epoch": epoch}
        if epoch % args.report_every == 0 or epoch == args.epochs:
            print(f"{name} epoch {epoch}/{args.epochs}: pair_rmse={metrics['pair_rmse']:.6g} dS_std={metrics['delta_S_std']:.6g} dlogw_std={metrics['delta_logw_std']:.6g}", flush=True)
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    ckpt = {
        "model_state": model.state_dict(),
        "config": {**init_ckpt["config"], "action_aware_variant": name, "action_aware_weights": weights},
        "stage": "pair",
        "selection": "best_val_delta_logw_std",
        "epoch": int(best["epoch"]),
        "val_loss": float(best["score"]),
        "best_row": best["row"],
        "dependency_report": model.dependency_report(),
    }
    return model, ckpt


def parse_variants() -> dict[str, dict[str, float]]:
    return {
        "baseline_resume": {"logdet": 0.01},
        "phi0p5": {"logdet": 0.01, "phi": 0.5},
        "local0p5": {"logdet": 0.01, "local": 0.5},
        "globalS0p002": {"logdet": 0.01, "global_s": 0.002},
        "phi0p25_local0p25": {"logdet": 0.01, "phi": 0.25, "local": 0.25},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-config", type=Path, default=PKG / "outputs" / "gathered_edge_distillation_square_r2_r3_full" / "smoke_square_r3.yaml")
    ap.add_argument("--init-pair-config", type=Path, default=PKG / "outputs" / "gathered_pair_distillation_square_r3_logdet0p01" / "pair_square_r3.yaml")
    ap.add_argument("--output-dir", type=Path, default=PKG / "outputs" / "gathered_pair_action_aware")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--learning-rate", type=float, default=1.0e-4)
    ap.add_argument("--train-conditions", type=int, default=512)
    ap.add_argument("--val-conditions", type=int, default=256)
    ap.add_argument("--max-configs", type=int, default=1024)
    ap.add_argument("--report-every", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20260707)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import torch

    cfg_old = load_config(args.old_config)
    cfg_pair = load_config(args.init_pair_config)
    refine_model, refine_state, stages_pair, coarse_action, fine_action, _ = load_frozen_models(cfg_pair)
    _, _, stages_old, _, _, _ = load_frozen_models(cfg_old)
    refine_model.load_state_dict(refine_state)
    refine_model.eval()
    for model, _, state, *_ in stages_pair.values():
        model.load_state_dict(state)
        model.eval()
    for model, _, state, *_ in stages_old.values():
        model.load_state_dict(state)
        model.eval()
    kernel, _ = load_kernel_spec(cfg_pair)
    data = build_dataset(args, cfg_pair, cfg_old, stages_pair, stages_old, refine_model, refine_state, kernel, fine_action)
    train, val = split_dataset(data, args.train_conditions)
    init_pair_path = resolve_run_paths(cfg_pair)["frozen_dir"] / "pair.pt"
    init_ckpt = torch.load(init_pair_path, map_location="cpu")
    pair_lg = stages_pair["pair"][1]
    source_bundle = resolve_run_paths(cfg_pair)["frozen_dir"]

    results = []
    best = None
    for name, weights in parse_variants().items():
        variant_dir = args.output_dir / name
        variant_dir.mkdir(parents=True, exist_ok=True)
        model, ckpt = train_variant(args, name, weights, init_ckpt, train, val, pair_lg, kernel, fine_action)
        ckpt_path = variant_dir / "pair.pt"
        torch.save(ckpt, ckpt_path)
        bundle_dir = variant_dir / "bundle"
        copy_bundle(source_bundle, bundle_dir, ckpt_path)
        cfg_path = variant_dir / "config.yaml"
        write_yaml_config(cfg_path, cfg_pair, bundle_dir)
        metrics = evaluate_model(model, val, pair_lg, kernel, coarse_action, fine_action)
        row = {"name": name, "weights": weights, "metrics": metrics, "bundle": str(bundle_dir), "config": str(cfg_path), "dependency_report": ckpt["dependency_report"]}
        write_json(variant_dir / "summary.json", row)
        results.append(row)
        if best is None or metrics["delta_logw_std"] < best["metrics"]["delta_logw_std"]:
            best = row
    summary = {"results": results, "best": best}
    write_json(args.output_dir / "pair_action_aware_summary.json", summary)
    lines = ["# Pair Action-Aware Training", "", "| variant | pair RMSE | phi RMSE | delta S std | delta logq std | delta logw std | decision |", "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for row in results:
        m = row["metrics"]
        decision = "promising" if m["delta_logw_std"] < 1.0 else ("improved" if m["delta_logw_std"] < 1.5 else "reject")
        lines.append(f"| {row['name']} | {m['pair_rmse']:.6g} | {m['phi_rmse']:.6g} | {m['delta_S_std']:.6g} | {m['delta_logq_std']:.6g} | {m['delta_logw_std']:.6g} | {decision} |")
    if best is not None:
        lines.extend(["", "## Best", f"- variant: `{best['name']}`", f"- config: `{best['config']}`", f"- dependency r_c/r_f: `{best['dependency_report']['coarse_radius']}` / `{best['dependency_report']['fine_radius']}`"])
    (args.output_dir / "pair_action_aware_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, default=float), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
