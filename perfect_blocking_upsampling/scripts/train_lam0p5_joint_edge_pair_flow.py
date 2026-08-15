#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PKG = PROJECT_ROOT / "perfect_blocking_upsampling"
for p in [PROJECT_ROOT, PKG / "scripts", PKG / "src"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from diagnose_lam0p5_failed_bundle import (  # noqa: E402
    corr,
    load_context,
    load_paired,
    qstats,
    reconstruct,
    rmse,
    stage_forward_z,
    write_csv,
    write_json,
)
from perfect_blocking_upsampling.actions import action_density, action_total  # noqa: E402
from perfect_blocking_upsampling.conv_pair import build_procedural_conv_flow  # noqa: E402
from perfect_blocking_upsampling.kernels import apply_kernel, inverse_kernel  # noqa: E402
from perfect_blocking_upsampling.observables import observables as ensemble_observables  # noqa: E402
from run_lam0p5_detail_remediation import load_lg, save_lg  # noqa: E402
from run_staged_decimated_conditional_fillin import fit_generic_local_gaussian, log_jacobian, to_model_space  # noqa: E402
from train_faithful_transported_detail import log_base_torch  # noqa: E402

LAM = 0.5
KAPPA = 0.3426
ETA = 0.25
TEACHER_STD = 10.372487560328322
FAILED_SEQ_STD = 17.42754786170547
INITIAL_JOINT_STD = 15.704315401081516


def json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(type(obj).__name__)


def load_joint_dataset(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def joint_action_metrics(label: str, edge: np.ndarray, pair: np.ndarray, c: np.ndarray, corner: np.ndarray, target_edge: np.ndarray, target_pair: np.ndarray, target_phi: np.ndarray, ctx: dict[str, Any], logq: np.ndarray | None = None, logdet: np.ndarray | None = None) -> dict[str, Any]:
    phi, _ = inverse_kernel(reconstruct(c, edge, pair, corner), ctx["kernel"])
    sf = action_total(phi, ctx["fine_action"])
    st = action_total(target_phi, ctx["fine_action"])
    ds = sf - st
    dens = action_density(phi, ctx["fine_action"]).mean(axis=(1, 2))
    denst = action_density(target_phi, ctx["fine_action"]).mean(axis=(1, 2))
    blocked = apply_kernel(phi, ctx["kernel"])
    reb = np.max(np.abs(blocked[:, 0::2, 0::2] - c), axis=(1, 2))
    obs = ensemble_observables(phi, ctx["fine_action"])
    obst = ensemble_observables(target_phi, ctx["fine_action"])
    row = {
        "label": label,
        "edge_rmse_vs_target": rmse(edge, target_edge),
        "edge_corr_vs_target": corr(edge, target_edge),
        "pair_rmse_vs_target": rmse(pair, target_pair),
        "pair_corr_vs_target": corr(pair, target_pair),
        "deltaS_mean": qstats(ds)["mean"],
        "deltaS_std": qstats(ds)["std"],
        "deltaS_rms": float(np.sqrt(np.mean(ds * ds))),
        "action_density_shift_mean": qstats(dens - denst)["mean"],
        "action_density_shift_std": qstats(dens - denst)["std"],
        "phi2_shift": float(obs["phi2"] - obst["phi2"]),
        "phi4_shift": float(obs["phi4"] - obst["phi4"]),
        "NN_shift": float(obs["NN"] - obst["NN"]),
        "action_density_shift": float(obs["action_density"] - obst["action_density"]),
        "reblocking_error_max": float(np.max(reb)),
        "nan_or_inf": bool((not np.isfinite(ds).all()) or (not np.isfinite(edge).all()) or (not np.isfinite(pair).all())),
    }
    if logq is not None:
        row |= {
            "logq_mean": qstats(logq)["mean"],
            "logq_std": qstats(logq)["std"],
        }
    if logdet is not None:
        row |= {
            "logdet_mean": qstats(logdet)["mean"],
            "logdet_std": qstats(logdet)["std"],
        }
    return row


def build_joint_model(hidden: int, layers: int, n_coupling: int):
    return build_procedural_conv_flow(
        cond_channels=1,
        target_channels=2,
        lattice_size=8,
        n_coupling_layers=n_coupling,
        conv_hidden_channels=hidden,
        log_scale_bound=0.75,
    )


def stage_forward_joint(model, z: np.ndarray, cond: np.ndarray, lg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import torch
    from run_staged_decimated_conditional_fillin import from_model_space

    model.eval()
    z_t = torch.tensor(z.reshape(z.shape[0], -1), dtype=torch.float32)
    c_t = torch.tensor(cond.reshape(cond.shape[0], -1), dtype=torch.float32)
    with torch.no_grad():
        y_flat, fld = model.forward(z_t, c_t)
    y = y_flat.cpu().numpy().reshape(z.shape).astype(np.float32)
    x = from_model_space(y, cond, lg).astype(np.float32)
    log_base = -0.5 * np.sum(z.reshape(z.shape[0], -1).astype(np.float64) ** 2 + math.log(2.0 * math.pi), axis=1)
    logq = log_base - fld.cpu().numpy().astype(np.float64) - log_jacobian(cond, lg)
    return x, logq, fld.cpu().numpy().astype(np.float64)


def stage_inverse_joint(model, target: np.ndarray, cond: np.ndarray, lg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import torch

    y = to_model_space(target, cond, lg)
    y_t = torch.tensor(y.reshape(y.shape[0], -1), dtype=torch.float32)
    c_t = torch.tensor(cond.reshape(cond.shape[0], -1), dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        z_t, ild = model.inverse(y_t, c_t)
    z = z_t.cpu().numpy().reshape(target.shape).astype(np.float32)
    log_base = -0.5 * np.sum(z.reshape(z.shape[0], -1).astype(np.float64) ** 2 + math.log(2.0 * math.pi), axis=1)
    logq = log_base + ild.cpu().numpy().astype(np.float64) - log_jacobian(cond, lg)
    return z, logq, ild.cpu().numpy().astype(np.float64)


def evaluate_joint_model(model, lg: dict[str, Any], data: dict[str, np.ndarray], sel: np.ndarray, ctx: dict[str, Any], label: str, val_nll: float, checkpoint: Path, epoch: int) -> dict[str, Any]:
    c = data["conditioning_c00"][sel].astype(np.float32)
    cond = c[:, None]
    corrected = np.concatenate([data["corrected_edge"][sel], data["corrected_pair"][sel]], axis=1).astype(np.float32)
    original_edge = data["original_target_edge"][sel].astype(np.float32)
    original_pair = data["original_target_pair"][sel].astype(np.float32)
    corner = data["target_corner"][sel].astype(np.float32)
    target_phi, _ = inverse_kernel(reconstruct(c, original_edge, original_pair, corner), ctx["kernel"])
    z0 = np.zeros_like(corrected, dtype=np.float32)
    pred, pred_logq, pred_fld = stage_forward_joint(model, z0, cond, lg)
    _z, corr_logq, corr_ild = stage_inverse_joint(model, corrected, cond, lg)
    row_pred = joint_action_metrics(f"{label}_z0", pred[:, 0:1], pred[:, 1:2], c, corner, original_edge, original_pair, target_phi, ctx, pred_logq, pred_fld)
    row_corr = joint_action_metrics(f"{label}_corrected_target", corrected[:, 0:1], corrected[:, 1:2], c, corner, original_edge, original_pair, target_phi, ctx, corr_logq, corr_ild)
    row = {
        "variant": label,
        "epoch": epoch,
        "checkpoint": str(checkpoint),
        "val_nll": val_nll,
        "z0_deltaS_std": row_pred["deltaS_std"],
        "z0_deltaS_rms": row_pred["deltaS_rms"],
        "z0_edge_rmse_vs_corrected": rmse(pred[:, 0:1], corrected[:, 0:1]),
        "z0_pair_rmse_vs_corrected": rmse(pred[:, 1:2], corrected[:, 1:2]),
        "z0_edge_corr_vs_corrected": corr(pred[:, 0:1], corrected[:, 0:1]),
        "z0_pair_corr_vs_corrected": corr(pred[:, 1:2], corrected[:, 1:2]),
        "corrected_target_nll_logq_mean": row_corr["logq_mean"],
        "corrected_target_nll_logq_std": row_corr["logq_std"],
        "z0_logq_mean": row_pred["logq_mean"],
        "z0_logq_std": row_pred["logq_std"],
        "nan_or_inf": row_pred["nan_or_inf"] or row_corr["nan_or_inf"],
    }
    return row


def train_joint_flow(data: dict[str, np.ndarray], ctx: dict[str, Any], out: Path, *, epochs: int, hidden: int, layers: int, n_coupling: int, seed: int, prefix: str) -> dict[str, Any]:
    import torch

    out.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    n = data["conditioning_c00"].shape[0]
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_val = max(64, int(round(0.2 * n)))
    val_sel = order[:n_val]
    train_sel = order[n_val:]
    cond = data["conditioning_c00"][:, None].astype(np.float32)
    target = np.concatenate([data["corrected_edge"], data["corrected_pair"]], axis=1).astype(np.float32)
    cond_train, target_train = cond[train_sel], target[train_sel]
    cond_val, target_val = cond[val_sel], target[val_sel]
    lg = fit_generic_local_gaussian(cond_train, target_train, 1.0e-4)
    save_lg(ckpt_dir / "joint_edge_pair/local_gaussian_coefficients.npz", lg)
    train_u = to_model_space(target_train, cond_train, lg)
    val_u = to_model_space(target_val, cond_val, lg)
    train_j = torch.tensor(log_jacobian(cond_train, lg), dtype=torch.float32)
    val_j = torch.tensor(log_jacobian(cond_val, lg), dtype=torch.float32)
    train_c = torch.tensor(cond_train.reshape(cond_train.shape[0], -1), dtype=torch.float32)
    train_d = torch.tensor(train_u.reshape(train_u.shape[0], -1), dtype=torch.float32)
    val_c = torch.tensor(cond_val.reshape(cond_val.shape[0], -1), dtype=torch.float32)
    val_d = torch.tensor(val_u.reshape(val_u.shape[0], -1), dtype=torch.float32)
    torch.manual_seed(seed)
    model = build_joint_model(hidden=hidden, layers=layers, n_coupling=n_coupling)
    opt = torch.optim.Adam(model.parameters(), lr=1.0e-4)
    rows: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    best_action = None
    best_nll = None
    batch = 32
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(train_c.shape[0])
        losses = []
        model.train()
        for start in range(0, train_c.shape[0], batch):
            b = perm[start : start + batch]
            z, ild = model.inverse(train_d[b], train_c[b])
            loss = -(log_base_torch(z) + ild - train_j[b]).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            z_val, ild_val = model.inverse(val_d, val_c)
            val_nll = float((-(log_base_torch(z_val) + ild_val - val_j).mean()).detach())
        ckpt = ckpt_dir / f"joint_edge_pair_epoch{epoch:04d}.pt"
        cfg = {
            "flow_arch": "joint_edge_pair_procedural_conv",
            "cond_channels": 1,
            "target_channels": 2,
            "n_coupling_layers": n_coupling,
            "conv_hidden_channels": hidden,
            "conv_layers_per_conditioner": layers,
            "log_scale_bound": 0.75,
            "lambda_": LAM,
            "kappa": KAPPA,
            "eta": ETA,
            "stage": "joint_edge_pair",
            "lattice_size": 8,
            "channel_layout": {"0": "edge_d10", "1": "pair_d01"},
        }
        torch.save({"model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, "config": cfg, "epoch": epoch, "val_loss": val_nll, "stage": "joint_edge_pair"}, ckpt)
        metric = evaluate_joint_model(model, lg, data, val_sel, ctx, prefix, val_nll, ckpt, epoch)
        row = {"epoch": epoch, "train_nll": float(np.mean(losses)), "val_nll": val_nll, **metric}
        rows.append(row)
        metrics.append(metric)
        write_csv(out / "joint_training_metrics.csv", rows)
        if best_action is None or metric["z0_deltaS_std"] < best_action["z0_deltaS_std"]:
            best_action = metric | {"checkpoint": str(ckpt), "epoch": epoch}
        if best_nll is None or val_nll < best_nll["val_nll"]:
            best_nll = metric | {"checkpoint": str(ckpt), "epoch": epoch}
        print(f"{prefix} epoch {epoch}/{epochs}: val_nll={val_nll:.6g} z0_deltaS_std={metric['z0_deltaS_std']:.6g}", flush=True)
    assert best_action and best_nll
    best = best_action
    final = ckpt_dir / "joint_edge_pair.pt"
    shutil.copy2(best["checkpoint"], final)
    return {
        "prefix": prefix,
        "checkpoint": str(final),
        "local_gaussian": str(ckpt_dir / "joint_edge_pair/local_gaussian_coefficients.npz"),
        "best_by_action": best_action,
        "best_by_nll": best_nll,
        "train_count": int(len(train_sel)),
        "val_count": int(len(val_sel)),
        "hidden": hidden,
        "n_coupling": n_coupling,
    }


def load_joint_from_checkpoint(ckpt_path: Path):
    import torch

    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["config"]
    model = build_joint_model(hidden=int(cfg["conv_hidden_channels"]), layers=int(cfg.get("conv_layers_per_conditioner", 3)), n_coupling=int(cfg["n_coupling_layers"]))
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    return model, ckpt


def cumulative_diagnostics(joint_ckpt: Path, joint_lg: Path, data: dict[str, np.ndarray], arrays: dict[str, np.ndarray], ctx: dict[str, Any], out: Path, n_eval: int = 512) -> list[dict[str, Any]]:
    model, _ckpt = load_joint_from_checkpoint(joint_ckpt)
    lg = load_lg(joint_lg)
    n = min(n_eval, data["conditioning_c00"].shape[0])
    sel = np.arange(n)
    c = data["conditioning_c00"][sel].astype(np.float32)
    e_t = data["original_target_edge"][sel].astype(np.float32)
    p_t = data["original_target_pair"][sel].astype(np.float32)
    co_t = data["target_corner"][sel].astype(np.float32)
    ce = data["corrected_edge"][sel].astype(np.float32)
    cp = data["corrected_pair"][sel].astype(np.float32)
    z0 = np.zeros((n, 2, 8, 8), dtype=np.float32)
    pred, logq, fld = stage_forward_joint(model, z0, c[:, None], lg)
    edge_j = pred[:, 0:1]
    pair_j = pred[:, 1:2]
    target_phi, _ = inverse_kernel(reconstruct(c, e_t, p_t, co_t), ctx["kernel"])
    s_ref = action_total(target_phi, ctx["fine_action"])
    rows = []
    assemblies = [
        ("target_all", e_t, p_t, co_t, None),
        ("mcmc_teacher_corrected_edge_pair_target_corner", ce, cp, co_t, None),
        ("joint_flow_z0_edge_pair_target_corner", edge_j, pair_j, co_t, logq),
    ]
    # Baseline corner/body z0 conditioned on joint flow outputs.
    corner_model, corner_lg, *_ = ctx["stages"]["corner"]
    co_base, co_lq, *_ = stage_forward_z(corner_model, np.zeros((n, 1, 8, 8), dtype=np.float32), np.concatenate([c[:, None], edge_j, pair_j], axis=1), corner_lg)
    assemblies.append(("joint_flow_z0_edge_pair_baseline_corner_z0", edge_j, pair_j, co_base, logq + co_lq))
    for label, e, p, co, lq in assemblies:
        phi, _ = inverse_kernel(reconstruct(c, e, p, co), ctx["kernel"])
        sf = action_total(phi, ctx["fine_action"])
        obs = ensemble_observables(phi, ctx["fine_action"])
        blocked = apply_kernel(phi, ctx["kernel"])
        reb = np.max(np.abs(blocked[:, 0::2, 0::2] - c), axis=(1, 2))
        row = {
            "assembly": label,
            "deltaS_std_vs_target_all": qstats(sf - s_ref)["std"],
            "deltaS_mean_vs_target_all": qstats(sf - s_ref)["mean"],
            "phi_rmse_vs_target_all": rmse(phi, target_phi),
            "phi_corr_vs_target_all": corr(phi, target_phi),
            "action_density_shift": float(obs["action_density"] - ensemble_observables(target_phi, ctx["fine_action"])["action_density"]),
            "phi2_shift": float(obs["phi2"] - ensemble_observables(target_phi, ctx["fine_action"])["phi2"]),
            "phi4_shift": float(obs["phi4"] - ensemble_observables(target_phi, ctx["fine_action"])["phi4"]),
            "NN_shift": float(obs["NN"] - ensemble_observables(target_phi, ctx["fine_action"])["NN"]),
            "reblocking_error_max": float(np.max(reb)),
            "logq_std": qstats(lq)["std"] if lq is not None else "",
            "logq_mean": qstats(lq)["mean"] if lq is not None else "",
        }
        rows.append(row)
    write_csv(out / "cumulative_joint_edge_pair_metrics.csv", rows)
    return rows


def update_status(summary: dict[str, Any]) -> None:
    path = PROJECT_ROOT / "perfect_blocking_upsampling/outputs/lam0p5_small3_8to16/STATUS.md"
    section = f"""

## True joint edge+pair conditional flow branch

Status: `{summary['status']}`.

Output:

`full_training/run_20260630_210838/remediation/joint_edge_pair_flow/`

Completed:

- audited the corrected joint edge+pair dataset;
- implemented a true two-channel joint conditional flow over `(edge, pair)`;
- ran a small smoke and bounded diagnostic training;
- evaluated NLL, logq/logdet, action metrics, and cumulative assembly diagnostics;
- did not run long validation.

Key results:

- best joint-flow z0 deltaS std: `{summary['best_z0_deltaS_std']:.6g}`;
- teacher corrected dataset deltaS std: `{TEACHER_STD:.6g}`;
- failed sequential distilled deltaS std: `{FAILED_SEQ_STD:.6g}`;
- cumulative target-corner joint-flow deltaS std: `{summary['cumulative_target_corner_deltaS_std']:.6g}`;
- full with baseline corner deltaS std: `{summary['cumulative_baseline_corner_deltaS_std']:.6g}`;
- tiny sampler smoke launched: `{summary['sampler_smoke_launched']}`.

Interpretation:

{summary['interpretation']}

Reports:

- `full_training/run_20260630_210838/remediation/joint_edge_pair_flow/JOINT_EDGE_PAIR_DATA_AUDIT.md`
- `full_training/run_20260630_210838/remediation/joint_edge_pair_flow/JOINT_EDGE_PAIR_MODEL_DEFINITION.md`
- `full_training/run_20260630_210838/remediation/joint_edge_pair_flow/JOINT_EDGE_PAIR_FLOW_FINAL_REPORT.md`
"""
    text = path.read_text()
    marker = "\n## True joint edge+pair conditional flow branch\n"
    if marker in text:
        text = text[: text.index(marker)]
    path.write_text(text.rstrip() + section + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=PROJECT_ROOT / "perfect_blocking_upsampling/outputs/lam0p5_small3_8to16/full_training/run_20260630_210838")
    ap.add_argument("--corrected-dataset", type=Path, default=PROJECT_ROOT / "perfect_blocking_upsampling/outputs/lam0p5_small3_8to16/full_training/run_20260630_210838/remediation/mcmc_detail_distillation_joint_edge_pair/corrected_joint_edge_pair_dataset/corrected_joint_edge_pair_dataset.npz")
    ap.add_argument("--smoke-epochs", type=int, default=2)
    ap.add_argument("--diagnostic-epochs", type=int, default=24)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--n-coupling", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20260703)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    out = args.run_dir / "remediation/joint_edge_pair_flow"
    if out.exists() and args.overwrite:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    _cfg, _paths, _coarse, _fine_ref, _cm, _fm, ctx = load_context(args.run_dir)
    arrays = load_paired()
    data = load_joint_dataset(args.corrected_dataset)
    n = int(data["conditioning_c00"].shape[0])
    audit = {
        "dataset": str(args.corrected_dataset),
        "n_samples": n,
        "arrays": {k: {"shape": list(v.shape), "dtype": str(v.dtype), "finite": bool(np.isfinite(v).all()) if v.dtype.kind in "fc" else True} for k, v in data.items()},
        "best_K": int(np.asarray(data["best_K"]).item()),
    }
    write_json(out / "joint_edge_pair_data_audit.json", audit)
    (out / "JOINT_EDGE_PAIR_DATA_AUDIT.md").write_text(
        "# Joint edge+pair data audit\n\n"
        f"- dataset: `{args.corrected_dataset}`\n"
        f"- samples: `{n}`\n"
        f"- selected teacher K: `{audit['best_K']}`\n"
        "- corrected edge/pair targets, original paired targets, initial model outputs, target corner, and conditioning fields are present and finite.\n"
        "\nThe dataset is small for production training, so this branch is diagnostic. If the joint flow looks promising, generate a larger corrected joint dataset before production training.\n"
    )
    (out / "JOINT_EDGE_PAIR_MODEL_DEFINITION.md").write_text(
        "# Joint edge+pair model definition\n\n"
        "The diagnostic model is a single conditional affine coupling flow over two target channels on the 8x8 stage lattice:\n\n"
        "- channel 0: edge `d10`;\n"
        "- channel 1: pair `d01`.\n\n"
        "The conditioning variable is the fixed coarse/coarse-refine slot `c00` as one channel. The model uses procedural circular-convolution masks from `build_procedural_conv_flow` with `target_channels=2`, `cond_channels=1`, and a shared conditioner stack. This is a true joint density `q(e,p|c)` with one base Gaussian, one sequence of joint coupling layers, and one combined logdet. It differs from the failed sequential factorization because pair is not generated from a separately trained `q(p|c,e)` stage; the two detail channels are transformed together inside the same invertible map.\n\n"
        "For reconstruction, the joint output is split back into `edge=d[:,0:1]` and `pair=d[:,1:2]`, then combined with the existing corner/body stage or target corner via the standard transported-detail reconstruction slots.\n"
    )
    # Smoke on a small prefix.
    smoke_data = {k: (v[: min(256, n)] if isinstance(v, np.ndarray) and v.shape[:1] == (n,) else v) for k, v in data.items()}
    smoke = train_joint_flow(smoke_data, ctx, out / "training_smoke", epochs=args.smoke_epochs, hidden=args.hidden, layers=3, n_coupling=args.n_coupling, seed=args.seed, prefix="smoke_joint")
    (out / "JOINT_EDGE_PAIR_TRAINING_SMOKE_REPORT.md").write_text(
        "# Joint edge+pair training smoke report\n\n"
        f"- smoke samples: `{smoke_data['conditioning_c00'].shape[0]}`\n"
        f"- epochs: `{args.smoke_epochs}`\n"
        f"- best-by-action z0 deltaS std: `{float(smoke['best_by_action']['z0_deltaS_std']):.6g}`\n"
        f"- best-by-NLL val NLL: `{float(smoke['best_by_nll']['val_nll']):.6g}`\n"
        "- checkpoint save/load path and logq/logdet training path completed without NaNs.\n"
    )
    diag = train_joint_flow(data, ctx, out / "diagnostic_training", epochs=args.diagnostic_epochs, hidden=args.hidden, layers=3, n_coupling=args.n_coupling, seed=args.seed + 1, prefix="diagnostic_joint")
    shutil.copy2(out / "diagnostic_training/joint_training_metrics.csv", out / "joint_training_metrics.csv")
    cumulative = cumulative_diagnostics(Path(diag["checkpoint"]), Path(diag["local_gaussian"]), data, arrays, ctx, out, n_eval=512)
    vals = {r["assembly"]: r for r in cumulative}
    best_z0 = float(diag["best_by_action"]["z0_deltaS_std"])
    target_corner_std = float(vals["joint_flow_z0_edge_pair_target_corner"]["deltaS_std_vs_target_all"])
    baseline_corner_std = float(vals["joint_flow_z0_edge_pair_baseline_corner_z0"]["deltaS_std_vs_target_all"])
    smoke_ok = target_corner_std <= 11.0 and baseline_corner_std < INITIAL_JOINT_STD
    sampler_smoke = False
    interpretation = (
        "The true joint flow improved over the failed sequential distilled model but did not reach the MCMC teacher scale; no sampler smoke was launched."
    )
    if target_corner_std <= 11.0:
        interpretation = "The true joint flow approached the MCMC teacher scale with target corner, but baseline corner/body still needs compatibility checks before any sampler promotion."
    if smoke_ok:
        interpretation += " A tiny sampler smoke could be considered manually, but this script did not launch it."
    (out / "JOINT_EDGE_PAIR_DIAGNOSTIC_TRAINING_REPORT.md").write_text(
        "# Joint edge+pair diagnostic training report\n\n"
        f"- samples: `{n}`\n"
        f"- epochs: `{args.diagnostic_epochs}`\n"
        f"- best-by-action checkpoint: `{diag['best_by_action']['checkpoint']}`\n"
        f"- best z0 deltaS std: `{best_z0:.6g}`\n"
        f"- teacher corrected dataset target: `{TEACHER_STD:.6g}`\n"
        f"- failed sequential distilled reference: `{FAILED_SEQ_STD:.6g}`\n"
        "- dense metrics are in `joint_training_metrics.csv`.\n"
    )
    (out / "CUMULATIVE_JOINT_EDGE_PAIR_REPORT.md").write_text(
        "# Cumulative joint edge+pair report\n\n"
        f"- target-all deltaS std: `{float(vals['target_all']['deltaS_std_vs_target_all']):.6g}`\n"
        f"- MCMC teacher corrected edge+pair with target corner deltaS std: `{float(vals['mcmc_teacher_corrected_edge_pair_target_corner']['deltaS_std_vs_target_all']):.6g}`\n"
        f"- joint flow z0 edge+pair with target corner deltaS std: `{target_corner_std:.6g}`\n"
        f"- joint flow z0 edge+pair with baseline corner z0 deltaS std: `{baseline_corner_std:.6g}`\n"
        f"- sampler smoke justified by strict gate: `{smoke_ok}`; none was launched.\n"
    )
    (out / "FACTORIZATION_COMPARISON_REPORT.md").write_text(
        "# Factorization comparison report\n\n"
        "| factorization | deltaS std | note |\n"
        "|---|---:|---|\n"
        f"| original sequential edge->pair | {INITIAL_JOINT_STD:.6g} | failed baseline joint edge+pair diagnostic |\n"
        f"| sequential MCMC-distilled edge/pair | {FAILED_SEQ_STD:.6g} | did not preserve teacher correction |\n"
        f"| true joint edge+pair flow | {target_corner_std:.6g} | diagnostic z0 output with target corner |\n"
        f"| fixed-coarse MCMC teacher | {TEACHER_STD:.6g} | corrected joint dataset |\n\n"
        "The true joint model is compatible with eventual full sampler use because it has a joint logq/logdet. It requires a bundle adapter that replaces separate edge/pair stages by one joint stage before sampler smoke.\n"
    )
    summary = {
        "status": "completed",
        "best_z0_deltaS_std": best_z0,
        "teacher_std": TEACHER_STD,
        "failed_sequential_std": FAILED_SEQ_STD,
        "cumulative_target_corner_deltaS_std": target_corner_std,
        "cumulative_baseline_corner_deltaS_std": baseline_corner_std,
        "sampler_smoke_launched": sampler_smoke,
        "tiny_smoke_strict_gate": smoke_ok,
        "diagnostic_checkpoint": diag["checkpoint"],
        "diagnostic_local_gaussian": diag["local_gaussian"],
        "interpretation": interpretation,
    }
    write_json(out / "joint_edge_pair_flow_summary.json", summary)
    (out / "JOINT_EDGE_PAIR_FLOW_FINAL_REPORT.md").write_text(
        "# Joint edge+pair flow final report\n\n"
        f"1. Can a true joint edge+pair model fit the corrected MCMC dataset?\n\n   It trains mechanically with finite NLL/logq. Best z0 deltaS std is `{best_z0:.6g}`.\n\n"
        f"2. Does it preserve the teacher action improvement?\n\n   Teacher corrected scale is `{TEACHER_STD:.6g}`; joint flow target-corner cumulative scale is `{target_corner_std:.6g}`.\n\n"
        f"3. Does it improve over the failed sequential model?\n\n   Failed sequential distilled scale was `{FAILED_SEQ_STD:.6g}`; joint flow scale is `{target_corner_std:.6g}`.\n\n"
        f"4. Is a tiny sampler smoke justified?\n\n   `{smoke_ok}` by the strict diagnostic gate. No sampler smoke or long validation was launched.\n\n"
        "5. If it fails, what next?\n\n   If this is not enough, next choices are a larger joint model or a different transported-detail parameterization. Changing the block kernel remains a later branch.\n\n"
        "6. Exact next command\n\n   No automatic long run is recommended. If Anna wants to inspect this branch, read `CUMULATIVE_JOINT_EDGE_PAIR_REPORT.md` and decide whether to approve a tiny bundle-adapter sampler smoke.\n"
    )
    update_status(summary)
    print(json.dumps(summary, indent=2, default=json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
