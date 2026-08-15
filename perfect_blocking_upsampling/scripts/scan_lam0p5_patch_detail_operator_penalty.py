#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PKG = PROJECT_ROOT / "perfect_blocking_upsampling"
for p in [PROJECT_ROOT, PKG / "scripts", PKG / "src"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from diagnose_lam0p5_failed_bundle import load_context, qstats, write_csv, write_json  # noqa: E402
from perfect_blocking_upsampling.kernels import kernel_fft, kernel_stencil_from_spec, normalize_kernel  # noqa: E402
from prototype_lam0p5_patch_detail_parameterization import (  # noqa: E402
    ETA,
    KAPPA,
    LAM,
    build_patch_model,
    load_paired,
    reconstruct_patch_detail,
    stack_detail,
)
from run_lam0p5_detail_remediation import save_lg  # noqa: E402
from run_staged_decimated_conditional_fillin import fit_generic_local_gaussian, from_model_space, log_jacobian, to_model_space, torch_from_model_space  # noqa: E402
from train_faithful_transported_detail import log_base_torch  # noqa: E402

V_FINE = 16 * 16
BASELINE_DELTA_S_STD = 10.653777337967485


def append_error(path: Path, where: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(f"\n## {where}\n\n```text\n{traceback.format_exc()}\n```\n")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    a = a[m]
    b = b[m]
    if len(a) < 2 or float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def quantile_summary(x: np.ndarray, prefix: str) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    return {
        f"{prefix}_mean": float(np.mean(x)),
        f"{prefix}_std": float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
        f"{prefix}_rms": float(np.sqrt(np.mean(x * x))),
        f"{prefix}_q01": float(np.quantile(x, 0.01)),
        f"{prefix}_q05": float(np.quantile(x, 0.05)),
        f"{prefix}_q10": float(np.quantile(x, 0.10)),
        f"{prefix}_q50": float(np.quantile(x, 0.50)),
        f"{prefix}_q90": float(np.quantile(x, 0.90)),
        f"{prefix}_q95": float(np.quantile(x, 0.95)),
        f"{prefix}_q99": float(np.quantile(x, 0.99)),
        f"{prefix}_min": float(np.min(x)),
        f"{prefix}_max": float(np.max(x)),
    }


def np_operator_residuals(phi_pred: np.ndarray, phi_ref: np.ndarray) -> dict[str, np.ndarray]:
    pred = np.asarray(phi_pred, dtype=np.float64)
    ref = np.asarray(phi_ref, dtype=np.float64)

    def ops(arr: np.ndarray) -> dict[str, np.ndarray]:
        phi2 = np.mean(arr * arr, axis=(1, 2))
        phi4 = np.mean(arr**4, axis=(1, 2))
        nn = 0.5 * (
            np.mean(arr * np.roll(arr, -1, axis=1), axis=(1, 2))
            + np.mean(arr * np.roll(arr, -1, axis=2), axis=(1, 2))
        )
        twonn = 0.5 * (
            np.mean(arr * np.roll(arr, -2, axis=1), axis=(1, 2))
            + np.mean(arr * np.roll(arr, -2, axis=2), axis=(1, 2))
        )
        diag = np.mean(arr * np.roll(np.roll(arr, -1, axis=1), -1, axis=2), axis=(1, 2))
        shifted_potential = np.mean(LAM * (arr * arr - 1.0) ** 2, axis=(1, 2))
        return {"phi2": phi2, "phi4": phi4, "NN": nn, "2nn": twonn, "diag": diag, "shifted_potential": shifted_potential}

    p = ops(pred)
    r = ops(ref)
    out = {f"delta_{k}": p[k] - r[k] for k in p}
    out["deltaS_hopping"] = V_FINE * (-4.0 * KAPPA * out["delta_NN"])
    out["deltaS_potential_shifted"] = V_FINE * out["delta_shifted_potential"]
    out["deltaS_phi2_raw"] = V_FINE * out["delta_phi2"]
    out["deltaS_project_quartic"] = V_FINE * (LAM * out["delta_phi4"])
    out["deltaS_total"] = out["deltaS_project_quartic"] + out["deltaS_hopping"]
    out["deltaS_shifted_total"] = out["deltaS_phi2_raw"] + out["deltaS_potential_shifted"] + out["deltaS_hopping"]
    return out


def torch_inverse_kernel(psi, kt):
    import torch

    return torch.fft.ifft2(torch.fft.fft2(psi) / kt[None, :, :]).real


def torch_reconstruct_phi(c_flat, detail_u, lg, kt):
    import torch

    n = c_flat.shape[0]
    c = c_flat.reshape(n, 1, 8, 8)
    x_flat = torch_from_model_space(detail_u.flatten(1), c_flat, (1, 8, 8), lg)
    d = x_flat.reshape(n, 3, 8, 8)
    psi = torch.zeros((n, 16, 16), dtype=c.dtype, device=c.device)
    psi[:, 0::2, 0::2] = c[:, 0]
    psi[:, 1::2, 0::2] = d[:, 0]
    psi[:, 0::2, 1::2] = d[:, 1]
    psi[:, 1::2, 1::2] = d[:, 2]
    return torch_inverse_kernel(psi, kt)


def torch_operator_residuals(phi_pred, phi_ref):
    nn_pred = 0.5 * (
        torch_mean2(phi_pred * phi_pred.roll(shifts=-1, dims=1))
        + torch_mean2(phi_pred * phi_pred.roll(shifts=-1, dims=2))
    )
    nn_ref = 0.5 * (
        torch_mean2(phi_ref * phi_ref.roll(shifts=-1, dims=1))
        + torch_mean2(phi_ref * phi_ref.roll(shifts=-1, dims=2))
    )
    phi2_pred = torch_mean2(phi_pred * phi_pred)
    phi2_ref = torch_mean2(phi_ref * phi_ref)
    phi4_pred = torch_mean2(phi_pred**4)
    phi4_ref = torch_mean2(phi_ref**4)
    pot_pred = torch_mean2(LAM * (phi_pred * phi_pred - 1.0) ** 2)
    pot_ref = torch_mean2(LAM * (phi_ref * phi_ref - 1.0) ** 2)
    delta_phi2 = phi2_pred - phi2_ref
    delta_phi4 = phi4_pred - phi4_ref
    delta_nn = nn_pred - nn_ref
    delta_pot = pot_pred - pot_ref
    delta_s_hop = V_FINE * (-4.0 * KAPPA * delta_nn)
    delta_s_pot = V_FINE * delta_pot
    delta_s_phi2 = V_FINE * delta_phi2
    delta_s_quartic = V_FINE * LAM * delta_phi4
    delta_s_total = delta_s_quartic + delta_s_hop
    return {
        "delta_phi2": delta_phi2,
        "delta_phi4": delta_phi4,
        "delta_NN": delta_nn,
        "deltaS_potential_shifted": delta_s_pot,
        "deltaS_hopping": delta_s_hop,
        "deltaS_phi2_raw": delta_s_phi2,
        "deltaS_project_quartic": delta_s_quartic,
        "deltaS_total": delta_s_total,
    }


def torch_mean2(x):
    return x.mean(dim=(-2, -1))


def centered_var_loss(x, sigma: float):
    scale = max(float(sigma), 1.0e-8)
    z = (x - x.mean()) / scale
    return torch_mean2_flat(z * z)


def bias_loss(x, sigma: float):
    scale = max(float(sigma), 1.0e-8)
    return (x.mean() / scale) ** 2


def torch_mean2_flat(x):
    return x.mean()


def compute_baseline_sigmas(path: Path | None, fallback: dict[str, float]) -> dict[str, float]:
    if path is None or not path.exists():
        return fallback
    rows = read_csv(path)
    cols = [
        "deltaS_total",
        "deltaS_hopping",
        "deltaS_potential_shifted",
        "delta_phi2",
        "delta_phi4",
        "delta_NN",
        "deltaS_project_quartic",
    ]
    sigmas: dict[str, float] = {}
    for col in cols:
        vals = np.asarray([float(r[col]) for r in rows if r.get(col, "") not in {"", "nan", "NaN"}], dtype=np.float64)
        sigmas[col] = float(np.std(vals, ddof=1)) if vals.size > 1 else fallback.get(col, 1.0)
    return {**fallback, **sigmas}


def extra_operator_loss(res: dict[str, Any], variant: str, alpha: float, sigmas: dict[str, float]):
    import torch

    if variant == "control":
        return res["deltaS_total"].new_tensor(0.0)
    if variant == "potential_bias":
        return alpha * bias_loss(res["deltaS_potential_shifted"], sigmas["deltaS_potential_shifted"])
    if variant == "potential_centered":
        return alpha * centered_var_loss(res["deltaS_potential_shifted"], sigmas["deltaS_potential_shifted"])
    if variant == "hopping_centered":
        return alpha * centered_var_loss(res["deltaS_hopping"], sigmas["deltaS_hopping"])
    if variant == "combined_potential_hopping":
        return alpha * (
            centered_var_loss(res["deltaS_potential_shifted"], sigmas["deltaS_potential_shifted"])
            + centered_var_loss(res["deltaS_hopping"], sigmas["deltaS_hopping"])
            + 0.25 * bias_loss(res["deltaS_potential_shifted"], sigmas["deltaS_potential_shifted"])
        )
    if variant == "observable_phi2_phi4_nn":
        return alpha * (
            centered_var_loss(res["delta_phi2"], sigmas["delta_phi2"])
            + centered_var_loss(res["delta_phi4"], sigmas["delta_phi4"])
            + centered_var_loss(res["delta_NN"], sigmas["delta_NN"])
        )
    raise ValueError(f"unknown variant {variant}")


def load_patch_checkpoint(ckpt: Path):
    import torch

    obj = torch.load(ckpt, map_location="cpu")
    cfg = obj["config"]
    model = build_patch_model(int(cfg["conv_hidden_channels"]), int(cfg["n_coupling_layers"]))
    model.load_state_dict(obj["model_state"])
    lg_npz = ckpt.parent / "patch_detail/local_gaussian_coefficients.npz"
    with np.load(lg_npz) as z:
        lg = {"coeffs": z["coeffs"], "sigma": z["sigma"], "ridge": float(z["ridge"])}
    return model, lg, cfg


def evaluate_model(model, lg, c_val, detail_val, fine_val, ctx, ckpt: Path, run: str, epoch: int, variant: str, alpha: float):
    import torch

    model.eval()
    cond_val = c_val[:, None].astype(np.float32)
    z0 = np.zeros((len(c_val), 3, 8, 8), dtype=np.float32)
    with torch.no_grad():
        y_flat, fld = model.forward(
            torch.tensor(z0.reshape(len(c_val), -1), dtype=torch.float32),
            torch.tensor(cond_val.reshape(len(c_val), -1), dtype=torch.float32),
        )
    y = y_flat.cpu().numpy().reshape(z0.shape).astype(np.float32)
    pred = from_model_space(y, cond_val, lg).astype(np.float32)
    phi = reconstruct_patch_detail(c_val, pred, ctx)
    res = np_operator_residuals(phi, fine_val)
    ds = res["deltaS_total"]
    centered = np.abs(ds - np.mean(ds))
    blocked = ctx["apply_kernel"](phi)
    reb = np.max(np.abs(blocked[:, 0::2, 0::2] - c_val))
    row: dict[str, Any] = {
        "run": run,
        "variant": variant,
        "alpha": alpha,
        "epoch": epoch,
        "checkpoint": str(ckpt),
        "nan_or_inf": bool((not np.isfinite(phi).all()) or (not np.isfinite(ds).all())),
        "reblocking_error_max": float(reb),
        "roundtrip_error_max": float(np.max(np.abs(reconstruct_patch_detail(c_val, detail_val, ctx) - fine_val))),
    }
    row.update(quantile_summary(ds, "deltaS"))
    row["abs_centered_deltaS_max"] = float(np.max(centered))
    for pct in [1, 5, 10]:
        n = max(1, int(math.ceil(len(ds) * pct / 100.0)))
        idx = np.argsort(centered)[::-1][:n]
        row[f"worst{pct}_abs_centered_deltaS_mean"] = float(np.mean(centered[idx]))
        row[f"worst{pct}_abs_deltaS_mean"] = float(np.mean(np.abs(ds[idx])))
        row[f"worst{pct}_deltaS_std"] = float(np.std(ds[idx], ddof=1)) if len(idx) > 1 else 0.0
    for key in ["deltaS_hopping", "deltaS_potential_shifted", "delta_phi2", "delta_phi4", "delta_NN", "delta_2nn", "delta_diag"]:
        row.update(quantile_summary(res[key], key))
        row[f"corr_{key}_deltaS"] = corr(res[key], ds)
        row[f"corr_{key}_abs_centered_deltaS"] = corr(res[key], centered)
    row["hopping_potential_mean_balance"] = float(np.mean(res["deltaS_hopping"]) + np.mean(res["deltaS_potential_shifted"]))
    row["hopping_over_potential_abs_mean"] = float(np.mean(np.abs(res["deltaS_hopping"])) / max(np.mean(np.abs(res["deltaS_potential_shifted"])), 1.0e-12))
    return row


def train_finetune_run(
    name: str,
    variant: str,
    alpha: float,
    out: Path,
    init_checkpoint: Path,
    c: np.ndarray,
    detail: np.ndarray,
    fine: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    ctx: dict[str, Any],
    sigmas: dict[str, float],
    *,
    seed: int,
    steps: int,
    eval_every: int,
    action_weight: float,
    batch: int,
    lr: float,
) -> dict[str, Any]:
    import torch

    run_dir = out / name
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model, lg, cfg0 = load_patch_checkpoint(init_checkpoint)
    save_lg(ckpt_dir / "patch_detail/local_gaussian_coefficients.npz", lg)
    cond = c[:, None].astype(np.float32)
    target = detail.astype(np.float32)
    train_u = to_model_space(target[train_idx], cond[train_idx], lg)
    val_u = to_model_space(target[val_idx], cond[val_idx], lg)
    train_j = torch.tensor(log_jacobian(cond[train_idx], lg), dtype=torch.float32)
    val_j = torch.tensor(log_jacobian(cond[val_idx], lg), dtype=torch.float32)
    train_c = torch.tensor(cond[train_idx].reshape(len(train_idx), -1), dtype=torch.float32)
    train_d = torch.tensor(train_u.reshape(len(train_idx), -1), dtype=torch.float32)
    train_fine = torch.tensor(fine[train_idx], dtype=torch.float32)
    val_c = torch.tensor(cond[val_idx].reshape(len(val_idx), -1), dtype=torch.float32)
    val_d = torch.tensor(val_u.reshape(len(val_idx), -1), dtype=torch.float32)
    stencil = normalize_kernel(kernel_stencil_from_spec(ctx["kernel"]))
    kt = torch.tensor(kernel_fft(stencil, 16, ctx["kernel"].eta), dtype=torch.complex64)
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None

    def save_and_eval(step: int, train_loss=float("nan"), train_nll=float("nan"), train_action=float("nan"), train_op=float("nan")):
        model.eval()
        with torch.no_grad():
            z_val, ild_val = model.inverse(val_d, val_c)
            val_nll = float((-(log_base_torch(z_val) + ild_val - val_j).mean()).detach())
        ckpt = ckpt_dir / f"patch_detail_step{step:04d}.pt"
        cfg = dict(cfg0)
        cfg.update(
            {
                "operator_variant": variant,
                "operator_alpha": alpha,
                "base_action_weight": action_weight,
                "finetune_from": str(init_checkpoint),
                "finetune_step": step,
                "lr": lr,
            }
        )
        torch.save({"model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, "config": cfg, "epoch": int(cfg0.get("epoch", 24)), "finetune_step": step, "val_loss": val_nll, "stage": "patch_detail"}, ckpt)
        row = evaluate_model(model, lg, c[val_idx], detail[val_idx], fine[val_idx], ctx, ckpt, name, step, variant, alpha)
        row.update(
            {
                "seed": seed,
                "hidden": int(cfg0["conv_hidden_channels"]),
                "n_coupling": int(cfg0["n_coupling_layers"]),
                "train_loss": train_loss,
                "train_nll": train_nll,
                "train_action_loss": train_action,
                "train_operator_loss": train_op,
                "val_nll": val_nll,
            }
        )
        rows.append(row)
        return row

    row0 = save_and_eval(0)
    best = dict(row0)
    write_csv(run_dir / "metrics.csv", rows)
    accum: list[tuple[float, float, float, float]] = []
    model.train()
    for step in range(1, steps + 1):
        b_np = rng.choice(len(train_idx), size=min(batch, len(train_idx)), replace=False)
        b = torch.tensor(b_np, dtype=torch.long)
        z, ild = model.inverse(train_d[b], train_c[b])
        nll = -(log_base_torch(z) + ild - train_j[b]).mean()
        z0 = torch.zeros((len(b_np), 3, 8, 8), dtype=torch.float32)
        y_flat, _ = model.forward(z0.flatten(1), train_c[b])
        phi = torch_reconstruct_phi(train_c[b], y_flat.reshape(len(b_np), 3, 8, 8), lg, kt)
        res = torch_operator_residuals(phi, train_fine[b])
        action_loss = centered_var_loss(res["deltaS_total"], sigmas["deltaS_total"])
        op_loss = extra_operator_loss(res, variant, alpha, sigmas)
        loss = nll + float(action_weight) * action_loss + op_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        accum.append((float(loss.detach()), float(nll.detach()), float(action_loss.detach()), float(op_loss.detach())))
        if step % eval_every == 0 or step == steps:
            avg = np.asarray(accum, dtype=np.float64)
            row = save_and_eval(step, float(avg[:, 0].mean()), float(avg[:, 1].mean()), float(avg[:, 2].mean()), float(avg[:, 3].mean()))
            accum.clear()
            if (row["deltaS_std"], row["abs_centered_deltaS_max"], row["val_nll"]) < (
                best["deltaS_std"],
                best["abs_centered_deltaS_max"],
                best["val_nll"],
            ):
                best = dict(row)
            write_csv(run_dir / "metrics.csv", rows)
            print(
                f"{name} step {step}/{steps}: val_nll={row['val_nll']:.6g} dSstd={row['deltaS_std']:.6g} "
                f"tailmax={row['abs_centered_deltaS_max']:.6g} op={row['train_operator_loss']:.6g}",
                flush=True,
            )
            model.train()
    assert best is not None
    shutil.copy2(best["checkpoint"], ckpt_dir / "patch_detail_best_operator.pt")
    return {"best": best, "rows": rows}


def train_run(
    name: str,
    variant: str,
    alpha: float,
    out: Path,
    c: np.ndarray,
    detail: np.ndarray,
    fine: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    ctx: dict[str, Any],
    sigmas: dict[str, float],
    *,
    seed: int,
    epochs: int,
    hidden: int,
    n_coupling: int,
    action_weight: float,
    batch: int,
) -> dict[str, Any]:
    import torch

    run_dir = out / name
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    cond = c[:, None].astype(np.float32)
    target = detail.astype(np.float32)
    cond_train, target_train = cond[train_idx], target[train_idx]
    cond_val, target_val = cond[val_idx], target[val_idx]
    lg = fit_generic_local_gaussian(cond_train, target_train, 1.0e-4)
    save_lg(ckpt_dir / "patch_detail/local_gaussian_coefficients.npz", lg)
    train_u = to_model_space(target_train, cond_train, lg)
    val_u = to_model_space(target_val, cond_val, lg)
    train_j = torch.tensor(log_jacobian(cond_train, lg), dtype=torch.float32)
    val_j = torch.tensor(log_jacobian(cond_val, lg), dtype=torch.float32)
    train_c = torch.tensor(cond_train.reshape(cond_train.shape[0], -1), dtype=torch.float32)
    train_d = torch.tensor(train_u.reshape(train_u.shape[0], -1), dtype=torch.float32)
    train_fine = torch.tensor(fine[train_idx], dtype=torch.float32)
    val_c = torch.tensor(cond_val.reshape(cond_val.shape[0], -1), dtype=torch.float32)
    val_d = torch.tensor(val_u.reshape(val_u.shape[0], -1), dtype=torch.float32)

    stencil = normalize_kernel(kernel_stencil_from_spec(ctx["kernel"]))
    kt = torch.tensor(kernel_fft(stencil, 16, ctx["kernel"].eta), dtype=torch.complex64)
    torch.manual_seed(seed)
    model = build_patch_model(hidden, n_coupling)
    opt = torch.optim.Adam(model.parameters(), lr=1.0e-4)
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(train_c.shape[0])
        losses: list[float] = []
        nlls: list[float] = []
        action_losses: list[float] = []
        op_losses: list[float] = []
        model.train()
        for start in range(0, train_c.shape[0], batch):
            b = perm[start : start + batch]
            z, ild = model.inverse(train_d[b], train_c[b])
            nll = -(log_base_torch(z) + ild - train_j[b]).mean()
            z0 = torch.zeros((len(b), 3, 8, 8), dtype=torch.float32)
            y_flat, _ = model.forward(z0.flatten(1), train_c[b])
            y = y_flat.reshape(len(b), 3, 8, 8)
            phi = torch_reconstruct_phi(train_c[b], y, lg, kt)
            res = torch_operator_residuals(phi, train_fine[b])
            action_loss = centered_var_loss(res["deltaS_total"], sigmas["deltaS_total"])
            op_loss = extra_operator_loss(res, variant, alpha, sigmas)
            loss = nll + float(action_weight) * action_loss + op_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
            nlls.append(float(nll.detach()))
            action_losses.append(float(action_loss.detach()))
            op_losses.append(float(op_loss.detach()))
        model.eval()
        with torch.no_grad():
            z_val, ild_val = model.inverse(val_d, val_c)
            val_nll = float((-(log_base_torch(z_val) + ild_val - val_j).mean()).detach())
        ckpt = ckpt_dir / f"patch_detail_epoch{epoch:04d}.pt"
        cfg = {
            "flow_arch": "joint_patch_detail_procedural_conv",
            "cond_channels": 1,
            "target_channels": 3,
            "n_coupling_layers": n_coupling,
            "conv_hidden_channels": hidden,
            "log_scale_bound": 0.75,
            "lambda_": LAM,
            "kappa": KAPPA,
            "eta": ETA,
            "stage": "patch_detail",
            "lattice_size": 8,
            "channel_layout": {"0": "d10", "1": "d01", "2": "d11"},
            "base_action_weight": action_weight,
            "operator_variant": variant,
            "operator_alpha": alpha,
        }
        torch.save({"model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, "config": cfg, "epoch": epoch, "val_loss": val_nll, "stage": "patch_detail"}, ckpt)
        row = evaluate_model(model, lg, c[val_idx], detail[val_idx], fine[val_idx], ctx, ckpt, name, epoch, variant, alpha)
        row.update(
            {
                "seed": seed,
                "hidden": hidden,
                "n_coupling": n_coupling,
                "train_loss": float(np.mean(losses)),
                "train_nll": float(np.mean(nlls)),
                "train_action_loss": float(np.mean(action_losses)),
                "train_operator_loss": float(np.mean(op_losses)),
                "val_nll": val_nll,
            }
        )
        rows.append(row)
        write_csv(run_dir / "metrics.csv", rows)
        if best is None or (row["deltaS_std"], row["abs_centered_deltaS_max"], row["val_nll"]) < (
            best["deltaS_std"],
            best["abs_centered_deltaS_max"],
            best["val_nll"],
        ):
            best = dict(row)
        print(
            f"{name} epoch {epoch}/{epochs}: val_nll={val_nll:.6g} dSstd={row['deltaS_std']:.6g} "
            f"tailmax={row['abs_centered_deltaS_max']:.6g} op={np.mean(op_losses):.6g}",
            flush=True,
        )
    assert best is not None
    shutil.copy2(best["checkpoint"], ckpt_dir / "patch_detail_best_operator.pt")
    return {"best": best, "rows": rows}


def make_report(out: Path, all_rows: list[dict[str, Any]], best_rows: list[dict[str, Any]], sigmas: dict[str, float]) -> None:
    write_csv(out / "operator_penalty_scan_all_epochs.csv", all_rows)
    write_csv(out / "operator_penalty_scan_best_by_run.csv", best_rows)
    best = min(best_rows, key=lambda r: (r["deltaS_std"], r["abs_centered_deltaS_max"], r["val_nll"]))
    material = best["deltaS_std"] < BASELINE_DELTA_S_STD - 0.25
    tail_improved = best["abs_centered_deltaS_max"] < min(r["abs_centered_deltaS_max"] for r in best_rows if r["variant"] == "control")
    smoke_ok = bool(material or (best["deltaS_std"] <= BASELINE_DELTA_S_STD and tail_improved))
    lines = [
        "# Operator-penalty scan report",
        "",
        "## Scope",
        "",
        "Bounded diagnostic scan for the lambda=0.5 patch-detail candidate. No sampler smoke or long validation was launched.",
        "",
        "## Baseline",
        "",
        f"- reference checkpoint: `patch_detail_epoch0024.pt`",
        f"- reference validation DeltaS std: `{BASELINE_DELTA_S_STD:.6g}`",
        "",
        "## Penalty normalization",
        "",
        "Penalties were normalized by baseline validation-set standard deviations:",
        "",
        "| operator | sigma |",
        "|---|---:|",
    ]
    for key in ["deltaS_total", "deltaS_hopping", "deltaS_potential_shifted", "delta_phi2", "delta_phi4", "delta_NN"]:
        lines.append(f"| `{key}` | {sigmas[key]:.6g} |")
    lines += [
        "",
        "## Best result by run",
        "",
        "| run | variant | alpha | epoch | DeltaS mean | DeltaS std | DeltaS RMS | max |DeltaS-mean| | worst 5% mean | hopping mean/std | potential mean/std | val NLL | reblock |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(best_rows, key=lambda r: (str(r["variant"]), float(r["alpha"]), str(r["run"]))):
        lines.append(
            f"| `{row['run']}` | `{row['variant']}` | {row['alpha']:.3g} | {row['epoch']} | "
            f"{row['deltaS_mean']:.6g} | {row['deltaS_std']:.6g} | {row['deltaS_rms']:.6g} | "
            f"{row['abs_centered_deltaS_max']:.6g} | {row['worst5_abs_centered_deltaS_mean']:.6g} | "
            f"{row['deltaS_hopping_mean']:.6g}/{row['deltaS_hopping_std']:.6g} | "
            f"{row['deltaS_potential_shifted_mean']:.6g}/{row['deltaS_potential_shifted_std']:.6g} | "
            f"{row['val_nll']:.6g} | {row['reblocking_error_max']:.3g} |"
        )
    lines += [
        "",
        "## Decision",
        "",
        f"- best run: `{best['run']}`",
        f"- best variant/alpha/epoch: `{best['variant']}` / `{best['alpha']}` / `{best['epoch']}`",
        f"- best DeltaS std: `{best['deltaS_std']:.6g}` versus baseline `{BASELINE_DELTA_S_STD:.6g}`",
        f"- best max |DeltaS-mean|: `{best['abs_centered_deltaS_max']:.6g}`",
        f"- material std improvement: `{material}`",
        f"- acceptable tail-only improvement: `{tail_improved}`",
        f"- candidate should be promoted to sampler integration: `{smoke_ok}`",
        "",
    ]
    if smoke_ok:
        lines.append("The scan produced a diagnostic candidate worth comparing against `loss_scan_variance` before sampler integration. Do not overwrite the previous best checkpoint; use the saved candidate checkpoint path in the table.")
    else:
        lines.append("No operator-penalty candidate clearly beats the previous `loss_scan_variance` checkpoint. Keep `patch_detail_epoch0024.pt` as the best learned candidate.")
    lines += [
        "",
        "## Output files",
        "",
        "- `operator_penalty_scan_all_epochs.csv`",
        "- `operator_penalty_scan_best_by_run.csv`",
        "- per-run `metrics.csv` files and checkpoints under each run directory",
    ]
    (out / "OPERATOR_PENALTY_SCAN_REPORT.md").write_text("\n".join(lines) + "\n")
    write_json(out / "operator_penalty_scan_summary.json", {"best": best, "material_std_improvement": material, "tail_improved": tail_improved, "promote_candidate": smoke_ok, "sigmas": sigmas})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paired-data", type=Path, default=PKG / "outputs/lam0p5_small3_8to16/paired_data/paired_lam0p5_small3_L16_to_L8.npz")
    ap.add_argument("--run-dir", type=Path, default=PKG / "outputs/lam0p5_small3_8to16/full_training/run_20260630_210838")
    ap.add_argument("--residual-csv", type=Path, default=PKG / "outputs/lam0p5_small3_8to16/remediation/patch_detail_parameterization/followup_remediation/operator_residuals_per_validation_sample.csv")
    ap.add_argument("--output-root", type=Path, default=PKG / "outputs/lam0p5_small3_8to16/remediation/patch_detail_parameterization/followup_remediation/operator_penalty_scan")
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--finetune-from", type=Path, default=PKG / "outputs/lam0p5_small3_8to16/remediation/patch_detail_parameterization/followup_remediation/loss_scan/loss_scan_variance/checkpoints/patch_detail_epoch0024.pt")
    ap.add_argument("--finetune-steps", type=int, default=60)
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--n-coupling", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20260725)
    ap.add_argument("--action-weight", type=float, default=5.0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3.0e-5)
    ap.add_argument("--from-scratch", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    out = args.output_root
    if out.exists() and args.overwrite:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    error_log = out / "ERROR_LOG.md"
    try:
        _cfg, _paths, _coarse, _fine_ref, _cm, _fm, ctx = load_context(args.run_dir)
        from perfect_blocking_upsampling.kernels import apply_kernel

        ctx = dict(ctx)
        ctx["apply_kernel"] = lambda phi: apply_kernel(phi, ctx["kernel"])
        arrays = load_paired(args.paired_data)
        c, detail, fine = stack_detail(arrays)
        train_idx = arrays["train_idx"].astype(np.int64)
        val_idx = arrays["val_idx"].astype(np.int64)
        fallback = {
            "deltaS_total": BASELINE_DELTA_S_STD,
            "deltaS_hopping": 17.503842478135848,
            "deltaS_potential_shifted": 9.830771182533034,
            "delta_phi2": 0.04078191074020347,
            "delta_phi4": 0.11808250197982893,
            "delta_NN": 0.04989374246951119,
            "deltaS_project_quartic": 15.114560253418103,
        }
        sigmas = compute_baseline_sigmas(args.residual_csv, fallback)
        write_json(out / "operator_penalty_scan_config.json", vars(args) | {"sigmas": sigmas})
        variants: list[tuple[str, str, float]] = [("control_alpha0", "control", 0.0)]
        for alpha in [0.01, 0.03, 0.1]:
            variants += [
                (f"potential_bias_alpha{str(alpha).replace('.', 'p')}", "potential_bias", alpha),
                (f"hopping_centered_alpha{str(alpha).replace('.', 'p')}", "hopping_centered", alpha),
                (f"combined_pot_hop_alpha{str(alpha).replace('.', 'p')}", "combined_potential_hopping", alpha),
            ]
        variants.append(("observable_phi2_phi4_nn_alpha0p03", "observable_phi2_phi4_nn", 0.03))
        all_rows: list[dict[str, Any]] = []
        best_rows: list[dict[str, Any]] = []
        for run_name, variant, alpha in variants:
            if args.from_scratch:
                result = train_run(
                    run_name,
                    variant,
                    alpha,
                    out,
                    c,
                    detail,
                    fine,
                    train_idx,
                    val_idx,
                    ctx,
                    sigmas,
                    seed=args.seed,
                    epochs=args.epochs,
                    hidden=args.hidden,
                    n_coupling=args.n_coupling,
                    action_weight=args.action_weight,
                    batch=args.batch,
                )
            else:
                result = train_finetune_run(
                    run_name,
                    variant,
                    alpha,
                    out,
                    args.finetune_from,
                    c,
                    detail,
                    fine,
                    train_idx,
                    val_idx,
                    ctx,
                    sigmas,
                    seed=args.seed,
                    steps=args.finetune_steps,
                    eval_every=args.eval_every,
                    action_weight=args.action_weight,
                    batch=args.batch,
                    lr=args.lr,
                )
            all_rows.extend(result["rows"])
            best_rows.append(result["best"])
            write_csv(out / "operator_penalty_scan_all_epochs.partial.csv", all_rows)
            write_csv(out / "operator_penalty_scan_best_by_run.partial.csv", best_rows)
        make_report(out, all_rows, best_rows, sigmas)
        print(json.dumps({"status": "completed", "out": str(out), "report": str(out / "OPERATOR_PENALTY_SCAN_REPORT.md")}, indent=2), flush=True)
        return 0
    except Exception:
        append_error(error_log, "operator penalty scan")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
