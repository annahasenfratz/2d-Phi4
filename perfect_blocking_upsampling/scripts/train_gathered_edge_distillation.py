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

from _common import load_config, load_ensembles, load_frozen_models, resolve_run_paths  # noqa: E402
from perfect_blocking_upsampling.coarse_refine import apply_refine  # noqa: E402
from perfect_blocking_upsampling.gathered_edge import build_gathered_edge_flow, corrcoef_flat  # noqa: E402
from ML_sampling_clean.experiments.decimated_conditional_fillin.run_staged_decimated_conditional_fillin import (  # noqa: E402
    from_model_space,
    log_jacobian,
    to_model_space,
)

LOG2PI = math.log(2.0 * math.pi)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=float) + "\n")


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


def evaluate_student(model, teacher, teacher_state, cond: np.ndarray, lg: dict[str, Any], seed: int, n_batches: int, batch_size: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    ys = []
    yt = []
    lqs = []
    lqt = []
    xs = []
    xt = []
    for _ in range(n_batches):
        idx = rng.integers(0, cond.shape[0], size=batch_size)
        cb = cond[idx]
        z = rng.standard_normal((batch_size, 1, cb.shape[2], cb.shape[3])).astype(np.float32)
        y_teacher, ld_teacher = teacher_forward(teacher, teacher_state, z, cb)
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
        "edge_model_space_rmse": float(np.sqrt(np.mean((y_s - y_t) ** 2))),
        "edge_output_rmse": float(np.sqrt(np.mean((x_s - x_t) ** 2))),
        "edge_output_corr": corrcoef_flat(x_s, x_t),
        "logq_rmse": float(np.sqrt(np.mean((logq_s - logq_t) ** 2))),
        "logq_corr": corrcoef_flat(logq_s, logq_t),
    }


def train_one(args, cond_train: np.ndarray, cond_val: np.ndarray, teacher, teacher_state, lg: dict[str, Any], out: Path) -> dict[str, Any]:
    import torch

    model = build_gathered_edge_flow(
        cond_channels=1,
        lattice_size=cond_train.shape[2],
        radius=args.radius,
        stencil=args.stencil,
        hidden_width=args.hidden_width,
        hidden_layers=args.hidden_layers,
        log_scale_bound=args.log_scale_bound,
    )
    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    rng = np.random.default_rng(args.seed + 17 * args.radius)
    rows = []
    best = {"val_loss": float("inf"), "state": None, "epoch": 0}
    for epoch in range(1, args.epochs + 1):
        losses = []
        model.train()
        steps = max(1, math.ceil(cond_train.shape[0] / args.batch_size))
        for _ in range(steps):
            idx = rng.integers(0, cond_train.shape[0], size=args.batch_size)
            cb = cond_train[idx]
            z = rng.standard_normal((args.batch_size, 1, cb.shape[2], cb.shape[3])).astype(np.float32)
            y_teacher, ld_teacher = teacher_forward(teacher, teacher_state, z, cb)
            z_t = torch.tensor(z.reshape(args.batch_size, -1), dtype=torch.float32)
            c_t = torch.tensor(cb.reshape(args.batch_size, -1), dtype=torch.float32)
            y_t = torch.tensor(y_teacher.reshape(args.batch_size, -1), dtype=torch.float32)
            ld_t = torch.tensor(ld_teacher, dtype=torch.float32)
            y_s, ld_s = model.forward(z_t, c_t)
            loss_y = torch.mean((y_s - y_t) ** 2)
            loss_ld = torch.mean((ld_s - ld_t) ** 2)
            loss = loss_y + args.logdet_loss_weight * loss_ld
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        metrics = evaluate_student(model, teacher, teacher_state, cond_val, lg, args.seed + epoch, args.eval_batches, args.batch_size)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": metrics["edge_model_space_rmse"], **metrics}
        rows.append(row)
        if row["val_loss"] < best["val_loss"]:
            best = {
                "val_loss": row["val_loss"],
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "epoch": epoch,
                "row": row,
            }
        if epoch % args.report_every == 0 or epoch == args.epochs:
            print(f"radius {args.radius} epoch {epoch}/{args.epochs}: rmse={row['edge_output_rmse']:.6g} logq_rmse={row['logq_rmse']:.6g}", flush=True)
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    report = model.dependency_report()
    ckpt = {
        "model_state": model.state_dict(),
        "config": {
            "flow_arch": "gathered_edge",
            "gather_radius": int(args.radius),
            "gather_stencil": args.stencil,
            "gather_hidden_width": int(args.hidden_width),
            "gather_hidden_layers": int(args.hidden_layers),
            "log_scale_bound": float(args.log_scale_bound),
            "lattice_size": int(cond_train.shape[2]),
            "stage": "edge",
            "teacher": str(args.teacher_edge),
        },
        "stage": "edge",
        "selection": "best_val_teacher_rmse",
        "epoch": int(best["epoch"]),
        "val_loss": float(best["val_loss"]),
        "dependency_report": report,
        "best_row": best.get("row", rows[-1]),
    }
    torch.save(ckpt, out / "edge.pt")
    write_json(out / "train_history.json", {"rows": rows, "best": ckpt["best_row"], "dependency_report": report})
    return {"checkpoint": str(out / "edge.pt"), "best": ckpt["best_row"], "dependency_report": report}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=PKG / "configs" / "default_lam0p022_k02705.yaml")
    ap.add_argument("--output-dir", type=Path, default=PKG / "outputs" / "gathered_edge_distillation")
    ap.add_argument("--radii", type=str, default="2,3")
    ap.add_argument("--stencil", choices=["square", "manhattan"], default="square")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--learning-rate", type=float, default=3.0e-4)
    ap.add_argument("--hidden-width", type=int, default=96)
    ap.add_argument("--hidden-layers", type=int, default=2)
    ap.add_argument("--log-scale-bound", type=float, default=0.75)
    ap.add_argument("--logdet-loss-weight", type=float, default=1.0e-3)
    ap.add_argument("--max-configs", type=int, default=512)
    ap.add_argument("--eval-batches", type=int, default=4)
    ap.add_argument("--report-every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260630)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    coarse, _, _, _, _ = load_ensembles(cfg)
    if args.max_configs > 0:
        coarse = coarse[: args.max_configs]
    refine_model, refine_state, stages, _, _, _ = load_frozen_models(cfg)
    cprime, _ = apply_refine(refine_model, refine_state, coarse, batch_size=32)
    cond = cprime[:, None].astype(np.float32)
    split = max(1, int(0.8 * cond.shape[0]))
    cond_train = cond[:split]
    cond_val = cond[split:] if split < cond.shape[0] else cond[: min(64, cond.shape[0])]
    teacher, lg, teacher_state = stages["edge"][:3]
    args.teacher_edge = resolve_run_paths(cfg)["frozen_dir"] / "edge.pt"
    summary = {
        "status": "complete",
        "config": str(args.config),
        "coarse_refine": "portable frozen distilled coarse-refine",
        "pair_corner": "old frozen components retained; edge only distilled here",
        "eta": 0.25,
        "radii": [],
    }
    frozen_dir = resolve_run_paths(cfg)["frozen_dir"]
    for radius in [int(x) for x in args.radii.split(",") if x.strip()]:
        radius_out = args.output_dir / f"{args.stencil}_r{radius}"
        radius_out.mkdir(parents=True, exist_ok=True)
        shutil.copy2(frozen_dir / "pair.pt", radius_out / "pair.pt")
        shutil.copy2(frozen_dir / "corner.pt", radius_out / "corner.pt")
        shutil.copy2(frozen_dir / "coarse_refine.pt", radius_out / "coarse_refine.pt")
        for stage in ["edge", "pair", "corner"]:
            src = frozen_dir / stage / "local_gaussian_coefficients.npz"
            dst_dir = radius_out / stage
            dst_dir.mkdir(exist_ok=True)
            if stage == "edge":
                shutil.copy2(src, dst_dir / "local_gaussian_coefficients.npz")
            else:
                shutil.copy2(src, dst_dir / "local_gaussian_coefficients.npz")
        args.radius = radius
        result = train_one(args, cond_train, cond_val, teacher, teacher_state, lg, radius_out)
        summary["radii"].append(result)
    write_json(args.output_dir / "summary.json", summary)
    lines = ["# Gathered Edge Distillation", "", f"- eta: `{summary['eta']}`", "- old pair/corner retained", ""]
    for row in summary["radii"]:
        dep = row["dependency_report"]
        best = row["best"]
        lines.extend(
            [
                f"## {dep['metric']} r_c={dep['coarse_radius']}",
                f"- r_f: `{dep['fine_radius']}`",
                f"- edge output RMSE: `{best['edge_output_rmse']:.6g}`",
                f"- edge output correlation: `{best['edge_output_corr']:.6g}`",
                f"- logq RMSE: `{best['logq_rmse']:.6g}`",
                f"- checkpoint: `{row['checkpoint']}`",
                "",
            ]
        )
    (args.output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
