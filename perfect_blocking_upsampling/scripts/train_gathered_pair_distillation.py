#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
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

from _common import load_actions, load_config, load_ensembles, load_frozen_models, load_kernel_spec, resolve_run_paths  # noqa: E402
from perfect_blocking_upsampling.actions import action_total  # noqa: E402
from perfect_blocking_upsampling.coarse_refine import apply_refine  # noqa: E402
from perfect_blocking_upsampling.gathered_edge import build_gathered_edge_flow, corrcoef_flat  # noqa: E402
from perfect_blocking_upsampling.kernels import inverse_kernel  # noqa: E402
from ML_sampling_clean.experiments.decimated_conditional_fillin.run_staged_decimated_conditional_fillin import (  # noqa: E402
    from_model_space,
    log_jacobian,
    torch_from_model_space,
)

LOG2PI = math.log(2.0 * math.pi)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=float) + "\n", encoding="utf-8")


def write_yaml_config(path: Path, source_cfg: dict[str, Any], bundle_dir: Path) -> None:
    rel_bundle = bundle_dir.resolve().relative_to(PKG.resolve())
    text = f"""run_name: {source_cfg.get('run_name', 'lam0p022_kappa0p2705_L8_to_L16_small3_refine')}
random_seed: {int(source_cfg.get('random_seed', 20260627))}
output_dir: outputs/sample_reproduction
checkpoints:
  frozen_dir: {rel_bundle}
action:
  coarse:
    type: {source_cfg['action']['coarse']['type']}
    lambda: {source_cfg['action']['coarse']['lambda']}
    kappa: {source_cfg['action']['coarse']['kappa']}
    kappa_diag: {source_cfg['action']['coarse'].get('kappa_diag', 0.0)}
  fine:
    type: {source_cfg['action']['fine']['type']}
    lambda: {source_cfg['action']['fine']['lambda']}
    kappa: {source_cfg['action']['fine']['kappa']}
    kappa_diag: {source_cfg['action']['fine'].get('kappa_diag', 0.0)}
lattice:
  coarse_L: {int(source_cfg['lattice']['coarse_L'])}
  fine_L: {int(source_cfg['lattice']['fine_L'])}
  scale_factor: {int(source_cfg['lattice'].get('scale_factor', 2))}
kernel:
  path: {source_cfg['kernel']['path']}
  eta: {source_cfg['kernel'].get('eta', 0.25)}
  scale_factor: {int(source_cfg['kernel'].get('scale_factor', 2))}
  normalize: {str(bool(source_cfg['kernel'].get('normalize', True))).lower()}
data:
  coarse_ensemble: {source_cfg['data']['coarse_ensemble']}
  fine_reference: {source_cfg['data']['fine_reference']}
model:
  coarse_refine: true
  missing_flow_type: affine
  stage_factorization: edge_pair_corner
evaluation:
  n_proposals: {int(source_cfg.get('evaluation', {}).get('n_proposals', 512))}
  ar_chains: {int(source_cfg.get('evaluation', {}).get('ar_chains', 4))}
  ar_proposals_per_chain: {int(source_cfg.get('evaluation', {}).get('ar_proposals_per_chain', 1000))}
"""
    path.write_text(text, encoding="utf-8")


def teacher_forward(model, state: dict[str, Any], z: np.ndarray, cond: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import torch

    model.load_state_dict(state)
    model.eval()
    z_t = torch.tensor(z.reshape(z.shape[0], -1), dtype=torch.float32)
    c_t = torch.tensor(cond.reshape(cond.shape[0], -1), dtype=torch.float32)
    with torch.no_grad():
        y, logdet = model.forward(z_t, c_t)
    return (
        y.cpu().numpy().reshape(z.shape[0], z.shape[1], z.shape[2], z.shape[3]).astype(np.float32),
        logdet.cpu().numpy().astype(np.float64),
    )


def student_forward(model, z: np.ndarray, cond: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import torch

    model.eval()
    z_t = torch.tensor(z.reshape(z.shape[0], -1), dtype=torch.float32)
    c_t = torch.tensor(cond.reshape(cond.shape[0], -1), dtype=torch.float32)
    with torch.no_grad():
        y, logdet = model.forward(z_t, c_t)
    return (
        y.cpu().numpy().reshape(z.shape[0], z.shape[1], z.shape[2], z.shape[3]).astype(np.float32),
        logdet.cpu().numpy().astype(np.float64),
    )


def logq_from_z_logdet(z: np.ndarray, logdet: np.ndarray, cond: np.ndarray, lg: dict[str, Any]) -> np.ndarray:
    log_base = -0.5 * np.sum(z.reshape(z.shape[0], -1).astype(np.float64) ** 2 + LOG2PI, axis=1)
    return (log_base - logdet - log_jacobian(cond, lg)).astype(np.float64)


def make_pair_conditions(
    cprime: np.ndarray,
    edge_model,
    edge_lg: dict[str, Any],
    edge_state: dict[str, Any],
    rng: np.random.Generator,
    batch_size: int,
) -> np.ndarray:
    idx = rng.integers(0, cprime.shape[0], size=batch_size)
    c = cprime[idx, None].astype(np.float32)
    z_edge = rng.standard_normal((batch_size, 1, c.shape[2], c.shape[3])).astype(np.float32)
    edge_u, _ = teacher_forward(edge_model, edge_state, z_edge, c)
    d10 = from_model_space(edge_u, c, edge_lg)
    return np.concatenate([c, d10], axis=1).astype(np.float32)


def build_pair_condition_bank(
    cprime: np.ndarray,
    edge_model,
    edge_lg: dict[str, Any],
    edge_state: dict[str, Any],
    seed: int,
    n_conditions: int,
    batch_size: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    chunks = []
    remaining = int(n_conditions)
    while remaining > 0:
        n = min(batch_size, remaining)
        chunks.append(make_pair_conditions(cprime, edge_model, edge_lg, edge_state, rng, n))
        remaining -= n
    return np.concatenate(chunks, axis=0)


def evaluate_student(
    model,
    cond: np.ndarray,
    z_eval: np.ndarray,
    y_teacher_eval: np.ndarray,
    ld_teacher_eval: np.ndarray,
    lg: dict[str, Any],
    n_batches: int,
    batch_size: int,
) -> dict[str, float]:
    rng = np.random.default_rng(12345 + cond.shape[0] + batch_size)
    ys = []
    yt = []
    lqs = []
    lqt = []
    xs = []
    xt = []
    for _ in range(n_batches):
        idx = rng.integers(0, cond.shape[0], size=batch_size)
        cb = cond[idx]
        z = z_eval[idx]
        y_teacher = y_teacher_eval[idx]
        ld_teacher = ld_teacher_eval[idx]
        y_student, ld_student = student_forward(model, z, cb)
        ys.append(y_student)
        yt.append(y_teacher)
        lqs.append(logq_from_z_logdet(z, ld_student, cb, lg))
        lqt.append(logq_from_z_logdet(z, ld_teacher, cb, lg))
        xs.append(from_model_space(y_student, cb, lg))
        xt.append(from_model_space(y_teacher, cb, lg))
    y_s = np.concatenate(ys, axis=0)
    y_t = np.concatenate(yt, axis=0)
    x_s = np.concatenate(xs, axis=0)
    x_t = np.concatenate(xt, axis=0)
    logq_s = np.concatenate(lqs, axis=0)
    logq_t = np.concatenate(lqt, axis=0)
    return {
        "pair_model_space_rmse": float(np.sqrt(np.mean((y_s - y_t) ** 2))),
        "pair_output_rmse": float(np.sqrt(np.mean((x_s - x_t) ** 2))),
        "pair_output_corr": corrcoef_flat(x_s, x_t),
        "pair_logq_rmse": float(np.sqrt(np.mean((logq_s - logq_t) ** 2))),
        "pair_logq_corr": corrcoef_flat(logq_s, logq_t),
    }


def precompute_teacher_targets(
    teacher,
    teacher_state: dict[str, Any],
    cond: np.ndarray,
    seed: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    zs = []
    ys = []
    lds = []
    for start in range(0, cond.shape[0], batch_size):
        cb = cond[start : start + batch_size]
        z = rng.standard_normal((cb.shape[0], 1, cb.shape[2], cb.shape[3])).astype(np.float32)
        y, ld = teacher_forward(teacher, teacher_state, z, cb)
        zs.append(z)
        ys.append(y)
        lds.append(ld)
    return np.concatenate(zs, axis=0), np.concatenate(ys, axis=0), np.concatenate(lds, axis=0)


def assemble_phi(cprime: np.ndarray, d10: np.ndarray, d01: np.ndarray, d11: np.ndarray, kernel) -> np.ndarray:
    psi = np.empty((cprime.shape[0], 2 * cprime.shape[1], 2 * cprime.shape[2]), dtype=np.float32)
    psi[:, 0::2, 0::2] = cprime
    psi[:, 1::2, 0::2] = d10[:, 0]
    psi[:, 0::2, 1::2] = d01[:, 0]
    psi[:, 1::2, 1::2] = d11[:, 0]
    phi, _ = inverse_kernel(psi, kernel)
    return phi


def evaluate_hybrid_swap(
    model,
    cprime: np.ndarray,
    coarse: np.ndarray,
    stages: dict[str, tuple[Any, dict[str, Any], dict[str, Any]]],
    kernel,
    coarse_action,
    fine_action,
    refine_logdet: np.ndarray,
    pair_lg: dict[str, Any],
    seed: int,
    n_batches: int,
    batch_size: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    edge_model, edge_lg, edge_state = stages["edge"][:3]
    pair_model, _, pair_state = stages["pair"][:3]
    corner_model, corner_lg, corner_state = stages["corner"][:3]
    phi_rmse = []
    delta_logw = []
    for _ in range(n_batches):
        idx = rng.integers(0, cprime.shape[0], size=batch_size)
        c = cprime[idx, None].astype(np.float32)
        coarse_b = coarse[idx].astype(np.float32)
        z_edge = rng.standard_normal((batch_size, 1, c.shape[2], c.shape[3])).astype(np.float32)
        z_pair = rng.standard_normal((batch_size, 1, c.shape[2], c.shape[3])).astype(np.float32)
        z_corner = rng.standard_normal((batch_size, 1, c.shape[2], c.shape[3])).astype(np.float32)

        edge_u, edge_ld = teacher_forward(edge_model, edge_state, z_edge, c)
        d10 = from_model_space(edge_u, c, edge_lg)
        edge_logq = logq_from_z_logdet(z_edge, edge_ld, c, edge_lg)

        pair_cond = np.concatenate([c, d10], axis=1).astype(np.float32)
        pair_u_old, pair_ld_old = teacher_forward(pair_model, pair_state, z_pair, pair_cond)
        pair_u_new, pair_ld_new = student_forward(model, z_pair, pair_cond)
        d01_old = from_model_space(pair_u_old, pair_cond, pair_lg)
        d01_new = from_model_space(pair_u_new, pair_cond, pair_lg)
        pair_logq_old = logq_from_z_logdet(z_pair, pair_ld_old, pair_cond, pair_lg)
        pair_logq_new = logq_from_z_logdet(z_pair, pair_ld_new, pair_cond, pair_lg)

        corner_cond_old = np.concatenate([c, d10, d01_old], axis=1).astype(np.float32)
        corner_cond_new = np.concatenate([c, d10, d01_new], axis=1).astype(np.float32)
        corner_u_old, corner_ld_old = teacher_forward(corner_model, corner_state, z_corner, corner_cond_old)
        corner_u_new, corner_ld_new = teacher_forward(corner_model, corner_state, z_corner, corner_cond_new)
        d11_old = from_model_space(corner_u_old, corner_cond_old, corner_lg)
        d11_new = from_model_space(corner_u_new, corner_cond_new, corner_lg)
        corner_logq_old = logq_from_z_logdet(z_corner, corner_ld_old, corner_cond_old, corner_lg)
        corner_logq_new = logq_from_z_logdet(z_corner, corner_ld_new, corner_cond_new, corner_lg)

        phi_old = assemble_phi(c[:, 0], d10, d01_old, d11_old, kernel)
        phi_new = assemble_phi(c[:, 0], d10, d01_new, d11_new, kernel)
        s_c = action_total(coarse_b, coarse_action)
        s_old = action_total(phi_old, fine_action)
        s_new = action_total(phi_new, fine_action)
        logw_old = -s_old + s_c + refine_logdet[idx] - (edge_logq + pair_logq_old + corner_logq_old)
        logw_new = -s_new + s_c + refine_logdet[idx] - (edge_logq + pair_logq_new + corner_logq_new)
        phi_rmse.append(np.mean((phi_new - phi_old) ** 2))
        delta_logw.append(logw_new - logw_old)
    deltas = np.concatenate(delta_logw, axis=0)
    return {
        "reconstructed_phi_rmse_vs_old_hybrid": float(np.sqrt(np.mean(phi_rmse))),
        "logweight_delta_portable_minus_old_std": float(np.std(deltas, ddof=1)),
        "logweight_delta_portable_minus_old_mean": float(np.mean(deltas)),
    }


def train_one(
    args,
    cond_train: np.ndarray,
    z_train: np.ndarray,
    y_train: np.ndarray,
    ld_train: np.ndarray,
    cond_val: np.ndarray,
    z_val: np.ndarray,
    y_val: np.ndarray,
    ld_val: np.ndarray,
    lg: dict[str, Any],
    corner_model,
    corner_state: dict[str, Any],
    corner_lg: dict[str, Any],
    out: Path,
) -> tuple[Any, dict[str, Any]]:
    import torch

    model = build_gathered_edge_flow(
        cond_channels=2,
        lattice_size=cond_train.shape[2],
        radius=args.radius,
        stencil=args.stencil,
        hidden_width=args.hidden_width,
        hidden_layers=args.hidden_layers,
        log_scale_bound=args.log_scale_bound,
    )
    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    if args.corner_consistency_weight > 0.0:
        corner_model.load_state_dict(corner_state)
        corner_model.eval()
        for param in corner_model.parameters():
            param.requires_grad_(False)
    rng = np.random.default_rng(args.seed + 31 * args.radius)
    rows = []
    best = {"val_loss": float("inf"), "state": None, "epoch": 0}
    for epoch in range(1, args.epochs + 1):
        losses = []
        model.train()
        steps = max(1, math.ceil(cond_train.shape[0] / args.batch_size))
        for _ in range(steps):
            idx = rng.integers(0, cond_train.shape[0], size=args.batch_size)
            cb = cond_train[idx]
            z = z_train[idx]
            y_teacher = y_train[idx]
            ld_teacher = ld_train[idx]
            z_t = torch.tensor(z.reshape(args.batch_size, -1), dtype=torch.float32)
            c_t = torch.tensor(cb.reshape(args.batch_size, -1), dtype=torch.float32)
            y_t = torch.tensor(y_teacher.reshape(args.batch_size, -1), dtype=torch.float32)
            ld_t = torch.tensor(ld_teacher, dtype=torch.float32)
            y_s, ld_s = model.forward(z_t, c_t)
            loss_y = torch.mean((y_s - y_t) ** 2)
            loss_ld = torch.mean((ld_s - ld_t) ** 2)
            loss = loss_y + args.logdet_loss_weight * loss_ld
            if args.corner_consistency_weight > 0.0:
                h = cb.shape[2]
                zc_np = rng.standard_normal((args.batch_size, 1, h, h)).astype(np.float32)
                zc_t = torch.tensor(zc_np.reshape(args.batch_size, -1), dtype=torch.float32)
                d01_teacher_flat = torch_from_model_space(y_t, c_t, (2, h, h), lg)
                d01_student_flat = torch_from_model_space(y_s, c_t, (2, h, h), lg)
                corner_cond_teacher = torch.cat([c_t, d01_teacher_flat], dim=1)
                corner_cond_student = torch.cat([c_t, d01_student_flat], dim=1)
                with torch.no_grad():
                    corner_teacher, _ = corner_model.forward(zc_t, corner_cond_teacher)
                corner_student, _ = corner_model.forward(zc_t, corner_cond_student)
                loss_corner = torch.mean((corner_student - corner_teacher) ** 2)
                loss = loss + args.corner_consistency_weight * loss_corner
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        metrics = evaluate_student(model, cond_val, z_val, y_val, ld_val, lg, args.eval_batches, args.batch_size)
        if args.selection_metric == "output":
            selection_score = metrics["pair_model_space_rmse"]
        elif args.selection_metric == "logq":
            selection_score = metrics["pair_logq_rmse"]
        else:
            selection_score = metrics["pair_model_space_rmse"] + args.selection_logq_weight * metrics["pair_logq_rmse"]
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_loss": float(selection_score),
            "selection_score": float(selection_score),
            **metrics,
        }
        rows.append(row)
        if row["val_loss"] < best["val_loss"]:
            best = {
                "val_loss": row["val_loss"],
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "epoch": epoch,
                "row": row,
            }
        if epoch % args.report_every == 0 or epoch == args.epochs:
            print(
                f"radius {args.radius} epoch {epoch}/{args.epochs}: "
                f"pair_rmse={row['pair_output_rmse']:.6g} logq_rmse={row['pair_logq_rmse']:.6g}",
                flush=True,
            )
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    report = model.dependency_report()
    dummy = build_gathered_edge_flow(
        cond_channels=2,
        lattice_size=2 * int(cond_train.shape[2]),
        radius=args.radius,
        stencil=args.stencil,
        hidden_width=args.hidden_width,
        hidden_layers=args.hidden_layers,
        log_scale_bound=args.log_scale_bound,
    ).dependency_report()
    ckpt = {
        "model_state": model.state_dict(),
        "config": {
            "flow_arch": "gathered_local",
            "gather_radius": int(args.radius),
            "gather_stencil": args.stencil,
            "gather_hidden_width": int(args.hidden_width),
            "gather_hidden_layers": int(args.hidden_layers),
            "log_scale_bound": float(args.log_scale_bound),
            "logdet_loss_weight": float(args.logdet_loss_weight),
            "corner_consistency_weight": float(args.corner_consistency_weight),
            "lattice_size": int(cond_train.shape[2]),
            "stage": "pair",
            "teacher": str(args.teacher_pair),
        },
        "stage": "pair",
        "selection": f"best_{args.selection_metric}",
        "epoch": int(best["epoch"]),
        "val_loss": float(best["val_loss"]),
        "dependency_report": report,
        "dummy_larger_volume_dependency_report": dummy,
        "best_row": best.get("row", rows[-1]),
    }
    torch.save(ckpt, out / "pair.pt")
    write_json(out / "train_history.json", {"rows": rows, "best": ckpt["best_row"], "dependency_report": report, "dummy_larger_volume_dependency_report": dummy})
    return model, ckpt


def copy_bundle_inputs(source_dir: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name in ["coarse_refine.pt", "edge.pt", "corner.pt"]:
        dst = out / name
        if dst.exists():
            dst.unlink()
        shutil.copy2(source_dir / name, dst)
    for stage in ["edge", "pair", "corner"]:
        dst = out / stage
        dst.mkdir(exist_ok=True)
        coeff_dst = dst / "local_gaussian_coefficients.npz"
        if coeff_dst.exists():
            coeff_dst.unlink()
        shutil.copy2(source_dir / stage / "local_gaussian_coefficients.npz", coeff_dst)


def write_checksums(root: Path) -> None:
    rows = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name != "sha256_checksums.txt"):
        h = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(f"{h}  {path.relative_to(root)}")
    (root / "sha256_checksums.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> int:
    default_config = PKG / "outputs" / "gathered_edge_distillation_square_r2_r3_full" / "smoke_square_r3.yaml"
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=default_config)
    ap.add_argument("--output-dir", type=Path, default=PKG / "outputs" / "gathered_pair_distillation_square_r3")
    ap.add_argument("--radius", type=int, default=3)
    ap.add_argument("--stencil", choices=["square", "manhattan"], default="square")
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--learning-rate", type=float, default=3.0e-4)
    ap.add_argument("--hidden-width", type=int, default=96)
    ap.add_argument("--hidden-layers", type=int, default=2)
    ap.add_argument("--log-scale-bound", type=float, default=0.75)
    ap.add_argument("--logdet-loss-weight", type=float, default=1.0e-3)
    ap.add_argument("--corner-consistency-weight", type=float, default=0.0)
    ap.add_argument("--selection-metric", choices=["output", "logq", "combined"], default="output")
    ap.add_argument("--selection-logq-weight", type=float, default=0.1)
    ap.add_argument("--max-configs", type=int, default=512)
    ap.add_argument("--train-conditions", type=int, default=2048)
    ap.add_argument("--val-conditions", type=int, default=512)
    ap.add_argument("--eval-batches", type=int, default=4)
    ap.add_argument("--hybrid-eval-batches", type=int, default=4)
    ap.add_argument("--report-every", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260701)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    coarse, _, _, _, _ = load_ensembles(cfg)
    if args.max_configs > 0:
        coarse = coarse[: args.max_configs]
    refine_model, refine_state, stages, coarse_action, fine_action, _ = load_frozen_models(cfg)
    cprime, refine_logdet = apply_refine(refine_model, refine_state, coarse, batch_size=32)

    edge_model, edge_lg, edge_state = stages["edge"][:3]
    pair_teacher, pair_lg, pair_state = stages["pair"][:3]
    corner_model, corner_lg, corner_state = stages["corner"][:3]
    print("building pair condition bank", flush=True)
    cond_train = build_pair_condition_bank(cprime, edge_model, edge_lg, edge_state, args.seed + 101, args.train_conditions, args.batch_size)
    cond_val = build_pair_condition_bank(cprime, edge_model, edge_lg, edge_state, args.seed + 202, args.val_conditions, args.batch_size)
    print("precomputing old pair teacher targets", flush=True)
    z_train, y_train, ld_train = precompute_teacher_targets(pair_teacher, pair_state, cond_train, args.seed + 203, args.batch_size)
    z_val, y_val, ld_val = precompute_teacher_targets(pair_teacher, pair_state, cond_val, args.seed + 204, args.batch_size)

    source_dir = resolve_run_paths(cfg)["frozen_dir"]
    bundle_dir = args.output_dir / f"{args.stencil}_r{args.radius}"
    copy_bundle_inputs(source_dir, bundle_dir)
    args.teacher_pair = source_dir / "pair.pt"
    model, ckpt = train_one(
        args,
        cond_train,
        z_train,
        y_train,
        ld_train,
        cond_val,
        z_val,
        y_val,
        ld_val,
        pair_lg,
        corner_model,
        corner_state,
        corner_lg,
        bundle_dir,
    )

    kernel, _ = load_kernel_spec(cfg)
    hybrid = evaluate_hybrid_swap(
        model,
        cprime,
        coarse,
        stages,
        kernel,
        coarse_action,
        fine_action,
        refine_logdet,
        pair_lg,
        args.seed + 303,
        args.hybrid_eval_batches,
        args.batch_size,
    )
    summary = {
        "status": "complete",
        "config": str(args.config),
        "bundle_dir": str(bundle_dir),
        "bundle_config": str(args.output_dir / f"pair_{args.stencil}_r{args.radius}.yaml"),
        "coarse_refine": "portable frozen distilled coarse-refine",
        "edge": "accepted gathered square r_c=3",
        "corner": "old frozen component retained",
        "eta": 0.25,
        "best": ckpt["best_row"],
        "hybrid_swap": hybrid,
        "dependency_report": ckpt["dependency_report"],
        "dummy_larger_volume_dependency_report": ckpt["dummy_larger_volume_dependency_report"],
    }
    write_json(args.output_dir / "summary.json", summary)
    write_json(bundle_dir / "pair_distillation_summary.json", summary)
    write_yaml_config(args.output_dir / f"pair_{args.stencil}_r{args.radius}.yaml", cfg, bundle_dir)
    write_checksums(bundle_dir)
    lines = [
        "# Gathered Pair Distillation",
        "",
        f"- pair stencil: `{args.stencil}`",
        f"- r_c: `{ckpt['dependency_report']['coarse_radius']}`",
        f"- r_f: `{ckpt['dependency_report']['fine_radius']}`",
        "- periodic shortest-displacement validation: `passed`",
        "- dummy larger-volume instantiation: `passed`",
        f"- pair output RMSE: `{ckpt['best_row']['pair_output_rmse']:.6g}`",
        f"- pair output correlation: `{ckpt['best_row']['pair_output_corr']:.6g}`",
        f"- pair logq RMSE: `{ckpt['best_row']['pair_logq_rmse']:.6g}`",
        f"- reconstructed phi RMSE vs old hybrid: `{hybrid['reconstructed_phi_rmse_vs_old_hybrid']:.6g}`",
        f"- logweight delta portable-minus-old std: `{hybrid['logweight_delta_portable_minus_old_std']:.6g}`",
        f"- bundle config: `{summary['bundle_config']}`",
        "",
    ]
    (args.output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=float), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
