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

from _common import load_config, load_ensembles, load_frozen_models, load_kernel_spec, resolve_run_paths  # noqa: E402
from perfect_blocking_upsampling.actions import action_total  # noqa: E402
from perfect_blocking_upsampling.coarse_refine import apply_refine  # noqa: E402
from perfect_blocking_upsampling.gathered_edge import build_gathered_edge_flow, corrcoef_flat  # noqa: E402
from perfect_blocking_upsampling.kernels import inverse_kernel  # noqa: E402
from train_gathered_pair_distillation import (  # noqa: E402
    assemble_phi,
    build_pair_condition_bank,
    from_model_space,
    logq_from_z_logdet,
    student_forward,
    teacher_forward,
    write_yaml_config,
    write_checksums,
)
from ML_sampling_clean.experiments.decimated_conditional_fillin.run_staged_decimated_conditional_fillin import (  # noqa: E402
    fit_generic_local_gaussian,
    log_jacobian,
    to_model_space,
)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=float) + "\n", encoding="utf-8")


def quantiles(x: np.ndarray) -> dict[str, float]:
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    return {
        "mean": float(np.mean(a)),
        "std": float(np.std(a, ddof=1)) if a.size > 1 else 0.0,
        "min": float(np.min(a)),
        "q01": float(np.quantile(a, 0.01)),
        "q05": float(np.quantile(a, 0.05)),
        "q50": float(np.quantile(a, 0.50)),
        "q95": float(np.quantile(a, 0.95)),
        "q99": float(np.quantile(a, 0.99)),
        "max": float(np.max(a)),
        "rmse": float(np.sqrt(np.mean(a * a))),
    }


def stage_model_space_teacher(
    model,
    state: dict[str, Any],
    z: np.ndarray,
    cond: np.ndarray,
    lg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    y, ld = teacher_forward(model, state, z, cond)
    return y, logq_from_z_logdet(z, ld, cond, lg)


def fit_corner_student(
    cond_train: np.ndarray,
    target_train: np.ndarray,
    target_logdet_train: np.ndarray,
    cond_val: np.ndarray,
    target_val: np.ndarray,
    target_logdet_val: np.ndarray,
    radius: int,
    stencil: str,
    hidden_width: int,
    hidden_layers: int,
    log_scale_bound: float,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    logdet_loss_weight: float,
) -> tuple[Any, dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    import torch

    lg = fit_generic_local_gaussian(cond_train, target_train, sigma_floor=1.0e-4)
    u_train = to_model_space(target_train, cond_train, lg)
    u_val = to_model_space(target_val, cond_val, lg)
    model = build_gathered_edge_flow(
        cond_channels=cond_train.shape[1],
        lattice_size=cond_train.shape[2],
        radius=radius,
        stencil=stencil,
        hidden_width=hidden_width,
        hidden_layers=hidden_layers,
        log_scale_bound=log_scale_bound,
    )
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    rng = np.random.default_rng(seed)
    rows = []
    best = {"score": float("inf"), "state": None, "epoch": 0, "row": None}
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        steps = max(1, math.ceil(cond_train.shape[0] / batch_size))
        for _ in range(steps):
            idx = rng.integers(0, cond_train.shape[0], size=batch_size)
            cb = cond_train[idx]
            ub = u_train[idx]
            ldb = target_logdet_train[idx]
            z = rng.standard_normal((cb.shape[0], 1, cb.shape[2], cb.shape[3])).astype(np.float32)
            y_t = torch.tensor(ub.reshape(cb.shape[0], -1), dtype=torch.float32)
            c_t = torch.tensor(cb.reshape(cb.shape[0], -1), dtype=torch.float32)
            z_t = torch.tensor(z.reshape(cb.shape[0], -1), dtype=torch.float32)
            ld_t = torch.tensor(ldb, dtype=torch.float32)
            y_s, ld_s = model.forward(z_t, c_t)
            loss_y = torch.mean((y_s - y_t) ** 2)
            loss_ld = torch.mean((ld_s - ld_t) ** 2)
            loss = loss_y + logdet_loss_weight * loss_ld
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        # validation against teacher outputs in model-space
        idx = rng.integers(0, cond_val.shape[0], size=min(batch_size, cond_val.shape[0]))
        cb = cond_val[idx]
        ub = u_val[idx]
        ldb = target_logdet_val[idx]
        z = rng.standard_normal((cb.shape[0], 1, cb.shape[2], cb.shape[3])).astype(np.float32)
        y_s, ld_s = student_forward(model, z, cb)
        val_rmse = float(np.sqrt(np.mean((y_s - ub) ** 2)) + 0.1 * np.sqrt(np.mean((ld_s - ldb) ** 2)))
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_model_space_rmse": val_rmse,
        }
        rows.append(row)
        if val_rmse < best["score"]:
            best = {
                "score": val_rmse,
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "epoch": epoch,
                "row": row,
            }
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    ckpt = {
        "model_state": model.state_dict(),
        "config": {
            "flow_arch": "gathered_local",
            "gather_radius": int(radius),
            "gather_stencil": stencil,
            "gather_hidden_width": int(hidden_width),
            "gather_hidden_layers": int(hidden_layers),
            "log_scale_bound": float(log_scale_bound),
            "lattice_size": int(cond_train.shape[2]),
            "stage": "corner",
        },
        "stage": "corner",
        "selection": "best_val_model_space_rmse",
        "epoch": int(best["epoch"]),
        "val_loss": float(best["score"]),
        "dependency_report": model.dependency_report(),
        "best_row": best["row"],
    }
    return model, lg, ckpt, rows


def full_logweight_delta(
    coarse: np.ndarray,
    refine_logdet: np.ndarray,
    coarse_action,
    fine_action,
    kernel,
    edge_logq: np.ndarray,
    d10: np.ndarray,
    z_pair: np.ndarray,
    pair_y_old: np.ndarray,
    pair_ld_old: np.ndarray,
    pair_y_new: np.ndarray,
    pair_ld_new: np.ndarray,
    pair_lg: dict[str, Any],
    corner_model,
    corner_state: dict[str, Any],
    corner_lg: dict[str, Any],
    z_corner: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    c = coarse[:, None].astype(np.float32)
    d01_old = from_model_space(pair_y_old, np.concatenate([c, d10], axis=1), pair_lg)
    d01_new = from_model_space(pair_y_new, np.concatenate([c, d10], axis=1), pair_lg)
    cond_old = np.concatenate([c, d10, d01_old], axis=1).astype(np.float32)
    cond_new = np.concatenate([c, d10, d01_new], axis=1).astype(np.float32)
    y11_old, ld11_old = teacher_forward(corner_model, corner_state, z_corner, cond_old)
    y11_new, ld11_new = teacher_forward(corner_model, corner_state, z_corner, cond_new)
    d11_old = from_model_space(y11_old, cond_old, corner_lg)
    d11_new = from_model_space(y11_new, cond_new, corner_lg)
    phi_old = assemble_phi(c[:, 0], d10, d01_old, d11_old, kernel)
    phi_new = assemble_phi(c[:, 0], d10, d01_new, d11_new, kernel)
    s_c = action_total(coarse, coarse_action)
    s_old = action_total(phi_old, fine_action)
    s_new = action_total(phi_new, fine_action)
    pair_logq_old = logq_from_z_logdet(z_pair, pair_ld_old, np.concatenate([c, d10], axis=1), pair_lg)
    pair_logq_new = logq_from_z_logdet(z_pair, pair_ld_new, np.concatenate([c, d10], axis=1), pair_lg)
    corner_logq_old = logq_from_z_logdet(z_corner, ld11_old, cond_old, corner_lg)
    corner_logq_new = logq_from_z_logdet(z_corner, ld11_new, cond_new, corner_lg)
    logw_old = -s_old + s_c + refine_logdet - (edge_logq + pair_logq_old + corner_logq_old)
    logw_new = -s_new + s_c + refine_logdet - (edge_logq + pair_logq_new + corner_logq_new)
    return logw_old, logw_new


def build_bundle(root: Path, source_pair_dir: Path, corner_ckpt_path: Path, corner_lg: dict[str, Any]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name in ["coarse_refine.pt", "edge.pt", "pair.pt"]:
        dst = root / name
        if dst.exists():
            dst.unlink()
        shutil.copy2(source_pair_dir / name, dst)
    for stage in ["edge", "pair"]:
        dst = root / stage
        dst.mkdir(exist_ok=True)
        coeff = dst / "local_gaussian_coefficients.npz"
        if coeff.exists():
            coeff.unlink()
        shutil.copy2(source_pair_dir / stage / "local_gaussian_coefficients.npz", coeff)
    shutil.copy2(corner_ckpt_path, root / "corner.pt")
    corner_dir = root / "corner"
    corner_dir.mkdir(exist_ok=True)
    np.savez_compressed(corner_dir / "local_gaussian_coefficients.npz", coeffs=corner_lg["coeffs"], sigma=corner_lg["sigma"], ridge=np.asarray(corner_lg["ridge"]))
    write_checksums(root)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-config", type=Path, default=PKG / "outputs" / "gathered_edge_distillation_square_r2_r3_full" / "smoke_square_r3.yaml")
    ap.add_argument("--pair-config", type=Path, default=PKG / "outputs" / "gathered_pair_distillation_square_r3_logdet0p01" / "pair_square_r3.yaml")
    ap.add_argument("--output-dir", type=Path, default=PKG / "outputs" / "gathered_pair_corner_interface_diag")
    ap.add_argument("--radius", type=int, default=3)
    ap.add_argument("--stencil", choices=["square", "manhattan"], default="square")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--learning-rate", type=float, default=3.0e-4)
    ap.add_argument("--hidden-width", type=int, default=96)
    ap.add_argument("--hidden-layers", type=int, default=2)
    ap.add_argument("--log-scale-bound", type=float, default=0.75)
    ap.add_argument("--logdet-loss-weight", type=float, default=1.0e-2)
    ap.add_argument("--n-samples", type=int, default=512)
    ap.add_argument("--seed", type=int, default=20260705)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cfg_old = load_config(args.old_config)
    cfg_pair = load_config(args.pair_config)
    coarse, _, _, _, _ = load_ensembles(cfg_old)
    refine_model, refine_state, stages_old, coarse_action, fine_action, _ = load_frozen_models(cfg_old)
    _, _, stages_pair, _, _, _ = load_frozen_models(cfg_pair)
    kernel, _ = load_kernel_spec(cfg_old)
    refine_model.load_state_dict(refine_state)
    refine_model.eval()
    for model, _, state, *_ in stages_old.values():
        model.load_state_dict(state)
        model.eval()
    for model, _, state, *_ in stages_pair.values():
        model.load_state_dict(state)
        model.eval()

    coarse = coarse[: args.n_samples]
    cprime, refine_logdet = apply_refine(refine_model, refine_state, coarse, batch_size=32)
    edge_model, edge_lg, edge_state = stages_old["edge"][:3]
    pair_old_model, pair_lg, pair_old_state = stages_old["pair"][:3]
    pair_new_model, _, pair_new_state = stages_pair["pair"][:3]
    corner_old_model, corner_old_lg, corner_old_state = stages_old["corner"][:3]

    rng = np.random.default_rng(args.seed)
    cond_pair = build_pair_condition_bank(cprime, edge_model, edge_lg, edge_state, args.seed + 1, args.n_samples, args.batch_size)
    z_edge = rng.standard_normal((args.n_samples, 1, cprime.shape[1], cprime.shape[2])).astype(np.float32)
    y_edge, ld_edge = teacher_forward(edge_model, edge_state, z_edge, cprime[:, None].astype(np.float32))
    edge_logq = logq_from_z_logdet(z_edge, ld_edge, cprime[:, None].astype(np.float32), edge_lg)
    z_pair = rng.standard_normal((args.n_samples, 1, cprime.shape[1], cprime.shape[2])).astype(np.float32)
    y_pair_old, ld_pair_old = teacher_forward(pair_old_model, pair_old_state, z_pair, cond_pair)
    y_pair_new, ld_pair_new = student_forward(pair_new_model, z_pair, cond_pair)
    d10 = from_model_space(y_edge, cprime[:, None].astype(np.float32), edge_lg)
    cond_corner_old = np.concatenate([cprime[:, None].astype(np.float32), d10, from_model_space(y_pair_old, cond_pair, pair_lg)], axis=1).astype(np.float32)
    cond_corner_new = np.concatenate([cprime[:, None].astype(np.float32), d10, from_model_space(y_pair_new, cond_pair, pair_lg)], axis=1).astype(np.float32)
    z_corner = rng.standard_normal((args.n_samples, 1, cprime.shape[1], cprime.shape[2])).astype(np.float32)
    y_corner_old, ld_corner_old = teacher_forward(corner_old_model, corner_old_state, z_corner, cond_corner_old)
    y_corner_new, ld_corner_new = teacher_forward(corner_old_model, corner_old_state, z_corner, cond_corner_new)

    input_diff = cond_corner_new - cond_corner_old
    corner_input_stats = {
        "rmse": float(np.sqrt(np.mean(input_diff**2))),
        "corr": corrcoef_flat(cond_corner_new, cond_corner_old),
        "per_channel_rmse": [float(np.sqrt(np.mean(input_diff[:, ch] ** 2))) for ch in range(input_diff.shape[1])],
        "per_channel_corr": [corrcoef_flat(cond_corner_new[:, ch], cond_corner_old[:, ch]) for ch in range(input_diff.shape[1])],
        "relative_to_old_std": [float(np.sqrt(np.mean(input_diff[:, ch] ** 2)) / max(np.std(cond_corner_old[:, ch]), 1e-12)) for ch in range(input_diff.shape[1])],
    }
    corner_output_stats = {
        "output_rmse": float(np.sqrt(np.mean((y_corner_new - y_corner_old) ** 2))),
        "output_corr": corrcoef_flat(y_corner_new, y_corner_old),
        "logq_rmse": float(np.sqrt(np.mean((logq_from_z_logdet(z_corner, ld_corner_new, cond_corner_new, corner_old_lg) - logq_from_z_logdet(z_corner, ld_corner_old, cond_corner_old, corner_old_lg)) ** 2))),
    }

    model_c, lg_c, ckpt_c, _ = fit_corner_student(
        cond_corner_new[: max(64, args.n_samples // 2)],
        y_corner_new[: max(64, args.n_samples // 2)],
        ld_corner_new[: max(64, args.n_samples // 2)],
        cond_corner_new[max(64, args.n_samples // 2) :],
        y_corner_new[max(64, args.n_samples // 2) :],
        ld_corner_new[max(64, args.n_samples // 2) :],
        radius=args.radius,
        stencil=args.stencil,
        hidden_width=args.hidden_width,
        hidden_layers=args.hidden_layers,
        log_scale_bound=args.log_scale_bound,
        seed=args.seed + 17,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        logdet_loss_weight=args.logdet_loss_weight,
    )
    y_corner_temp, ld_corner_temp = student_forward(model_c, z_corner, cond_corner_new)
    corner_student_stats = {
        "model_space_rmse": float(np.sqrt(np.mean((y_corner_temp - y_corner_new) ** 2))),
        "output_rmse": float(np.sqrt(np.mean((from_model_space(y_corner_temp, cond_corner_new, lg_c) - from_model_space(y_corner_new, cond_corner_new, corner_old_lg)) ** 2))),
        "output_corr": corrcoef_flat(from_model_space(y_corner_temp, cond_corner_new, lg_c), from_model_space(y_corner_new, cond_corner_new, corner_old_lg)),
        "logq_rmse": float(np.sqrt(np.mean((logq_from_z_logdet(z_corner, ld_corner_temp, cond_corner_new, lg_c) - logq_from_z_logdet(z_corner, ld_corner_new, cond_corner_new, corner_old_lg)) ** 2))),
    }

    logw_old, logw_new = full_logweight_delta(
        coarse[: args.n_samples],
        refine_logdet[: args.n_samples],
        coarse_action,
        fine_action,
        kernel,
        edge_logq,
        d10,
        z_pair,
        y_pair_old,
        ld_pair_old,
        y_pair_new,
        ld_pair_new,
        pair_lg,
        corner_old_model,
        corner_old_state,
        corner_old_lg,
        z_corner,
    )
    logw_new_joint = full_logweight_delta(
        coarse[: args.n_samples],
        refine_logdet[: args.n_samples],
        coarse_action,
        fine_action,
        kernel,
        edge_logq,
        d10,
        z_pair,
        y_pair_old,
        ld_pair_old,
        y_pair_new,
        ld_pair_new,
        pair_lg,
        model_c,
        model_c.state_dict(),
        lg_c,
        z_corner,
    )

    temp_corner_ckpt_path = args.output_dir / "corner_student.pt"
    import torch

    torch.save(ckpt_c, temp_corner_ckpt_path)
    source_pair_dir = resolve_run_paths(cfg_pair)["frozen_dir"]
    temp_bundle = args.output_dir / "joint_temp_bundle"
    build_bundle(temp_bundle, source_pair_dir, temp_corner_ckpt_path, lg_c)
    temp_cfg_path = args.output_dir / "joint_temp_bundle.yaml"
    write_yaml_config(temp_cfg_path, cfg_pair, temp_bundle)
    summary = {
        "pair_config": str(args.pair_config),
        "old_config": str(args.old_config),
        "dependency_report": ckpt_c["dependency_report"],
        "corner_input_sensitivity": corner_input_stats,
        "old_corner_response_to_portable_pair": corner_output_stats,
        "temporary_corner_student": corner_student_stats,
        "old_pair_old_corner_logw_std": float(np.std(logw_old, ddof=1)),
        "portable_pair_old_corner_logw_std": float(np.std(logw_new, ddof=1)),
        "portable_pair_temp_corner_logw_std": float(np.std(logw_new_joint, ddof=1)),
        "portable_pair_old_corner_logw_delta_std": float(np.std(logw_new - logw_old, ddof=1)),
        "portable_pair_temp_corner_logw_delta_std": float(np.std(logw_new_joint - logw_old, ddof=1)),
        "portable_pair_old_corner_logw_delta_mean": float(np.mean(logw_new - logw_old)),
        "portable_pair_temp_corner_logw_delta_mean": float(np.mean(logw_new_joint - logw_old)),
        "joint_temp_bundle": str(temp_bundle),
        "joint_temp_config": str(temp_cfg_path),
    }
    write_json(args.output_dir / "pair_corner_interface_report.json", summary)
    lines = [
        "# Pair-Corner Interface Diagnostic",
        "",
        "## Corner Input Sensitivity",
        f"- input RMSE: `{corner_input_stats['rmse']:.6g}`",
        f"- input correlation: `{corner_input_stats['corr']:.6g}`",
        f"- per-channel RMSE: `{corner_input_stats['per_channel_rmse']}`",
        f"- per-channel correlation: `{corner_input_stats['per_channel_corr']}`",
        f"- relative-to-old-std: `{corner_input_stats['relative_to_old_std']}`",
        "",
        "## Old Corner Response to Portable Pair",
        f"- corner output RMSE: `{corner_output_stats['output_rmse']:.6g}`",
        f"- corner output correlation: `{corner_output_stats['output_corr']:.6g}`",
        f"- corner logq RMSE: `{corner_output_stats['logq_rmse']:.6g}`",
        "",
        "## Temporary Portable Corner Student",
        f"- model-space RMSE vs old corner on portable pair inputs: `{corner_student_stats['model_space_rmse']:.6g}`",
        f"- output RMSE: `{corner_student_stats['output_rmse']:.6g}`",
        f"- output correlation: `{corner_student_stats['output_corr']:.6g}`",
        f"- logq RMSE: `{corner_student_stats['logq_rmse']:.6g}`",
        "",
        "## Full Logweight Swap",
        f"- old pair + old corner std: `{summary['old_pair_old_corner_logw_std']:.6g}`",
        f"- portable pair + old corner std: `{summary['portable_pair_old_corner_logw_std']:.6g}`",
        f"- portable pair + temp corner std: `{summary['portable_pair_temp_corner_logw_std']:.6g}`",
        f"- portable pair + old corner delta std: `{summary['portable_pair_old_corner_logw_delta_std']:.6g}`",
        f"- portable pair + temp corner delta std: `{summary['portable_pair_temp_corner_logw_delta_std']:.6g}`",
        "",
        "## Interpretation",
    ]
    if summary["portable_pair_temp_corner_logw_delta_std"] < summary["portable_pair_old_corner_logw_delta_std"]:
        lines.append("- replacing the corner with a portable-pair-conditioned diagnostic student reduces the swap spread; the mismatch is at least partly an inter-stage compatibility problem.")
    else:
        lines.append("- the diagnostic corner student does not reduce the swap spread; the problem is not fixed by a corner-only reaction to portable pair inputs.")
    lines.append(f"- temporary joint bundle: `{temp_bundle}`")
    lines.append(f"- temporary bundle config: `{temp_cfg_path}`")
    (args.output_dir / "pair_corner_interface_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True, default=float), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
