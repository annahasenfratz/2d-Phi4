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
from perfect_blocking_upsampling.actions import action_total  # noqa: E402
from perfect_blocking_upsampling.kernels import apply_kernel, kernel_fft, kernel_stencil_from_spec, normalize_kernel  # noqa: E402
from prototype_lam0p5_patch_detail_parameterization import (  # noqa: E402
    ETA,
    KAPPA,
    LAM,
    build_patch_model,
    load_paired,
    patch_metrics,
    reconstruct_patch_detail,
    stack_detail,
)
from run_lam0p5_detail_remediation import save_lg  # noqa: E402
from run_staged_decimated_conditional_fillin import fit_generic_local_gaussian, log_jacobian, to_model_space, torch_from_model_space  # noqa: E402
from train_faithful_transported_detail import log_base_torch  # noqa: E402


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def append_error(log: Path, where: str, exc: BaseException) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as f:
        f.write(f"\n## {where}\n\n```text\n{traceback.format_exc()}\n```\n")


def delta_quantiles(ds: np.ndarray) -> dict[str, float]:
    qs = np.quantile(ds, [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0])
    return {
        "deltaS_min": float(qs[0]),
        "deltaS_q01": float(qs[1]),
        "deltaS_q05": float(qs[2]),
        "deltaS_q25": float(qs[3]),
        "deltaS_median": float(qs[4]),
        "deltaS_q75": float(qs[5]),
        "deltaS_q95": float(qs[6]),
        "deltaS_q99": float(qs[7]),
        "deltaS_max": float(qs[8]),
        "abs_centered_deltaS_max": float(np.max(np.abs(ds - ds.mean()))),
    }


def torch_inverse_kernel(psi, kt):
    return torch.fft.ifft2(torch.fft.fft2(psi) / kt[None, :, :]).real


def torch_reconstruct_phi(c, detail_u, lg, kt):
    x_flat = torch_from_model_space(detail_u.flatten(1), c.flatten(1), (1, c.shape[-2], c.shape[-1]), lg)
    d = x_flat.reshape(c.shape[0], 3, c.shape[-2], c.shape[-1])
    psi = torch.zeros((c.shape[0], 16, 16), dtype=c.dtype, device=c.device)
    psi[:, 0::2, 0::2] = c[:, 0]
    psi[:, 1::2, 0::2] = d[:, 0]
    psi[:, 0::2, 1::2] = d[:, 1]
    psi[:, 1::2, 1::2] = d[:, 2]
    return torch_inverse_kernel(psi, kt)


def torch_action(phi):
    nn = phi * torch.roll(phi, shifts=-1, dims=-2) + phi * torch.roll(phi, shifts=-1, dims=-1)
    return (-2.0 * KAPPA * nn + phi * phi + LAM * (phi * phi - 1.0) ** 2).sum(dim=(-2, -1))


def action_aux_loss(ds, mode: str, sigma: float):
    x = (ds - ds.mean()) / max(float(sigma), 1.0e-6)
    if mode == "none":
        return ds.new_tensor(0.0)
    if mode == "variance":
        return torch.mean(x * x)
    if mode == "huber":
        return torch.nn.functional.smooth_l1_loss(x, torch.zeros_like(x), beta=1.0)
    if mode == "tail":
        threshold = 1.0
        tail = torch.relu(torch.abs(x) - threshold)
        return torch.mean(x * x) + 2.0 * torch.mean(tail * tail)
    raise ValueError(f"unknown action loss mode {mode}")


def evaluate_checkpoint(model, lg, c_val, d_val, fine_val, ctx, label: str):
    import torch

    model.eval()
    cond_val = c_val[:, None].astype(np.float32)
    z0 = np.zeros((len(c_val), 3, 8, 8), dtype=np.float32)
    z_t = torch.tensor(z0.reshape(z0.shape[0], -1), dtype=torch.float32)
    c_t = torch.tensor(cond_val.reshape(cond_val.shape[0], -1), dtype=torch.float32)
    with torch.no_grad():
        y_flat, fld = model.forward(z_t, c_t)
    y = y_flat.cpu().numpy().reshape(z0.shape).astype(np.float32)
    from run_staged_decimated_conditional_fillin import from_model_space

    pred = from_model_space(y, cond_val, lg).astype(np.float32)
    log_base = -0.5 * np.sum(z0.reshape(z0.shape[0], -1).astype(np.float64) ** 2 + math.log(2 * math.pi), axis=1)
    logq = log_base - fld.cpu().numpy().astype(np.float64) - log_jacobian(cond_val, lg)
    metric = patch_metrics(label, c_val, pred, d_val, fine_val, ctx, logq, fld.cpu().numpy().astype(np.float64))
    phi = reconstruct_patch_detail(c_val, pred, ctx)
    ds = action_total(phi, ctx["fine_action"]) - action_total(fine_val, ctx["fine_action"])
    metric.update(delta_quantiles(ds))
    return metric, ds, pred


def train_variant(
    name: str,
    out: Path,
    c: np.ndarray,
    detail: np.ndarray,
    fine: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    ctx: dict[str, Any],
    *,
    seed: int,
    epochs: int,
    hidden: int,
    n_coupling: int,
    action_mode: str,
    action_weight: float,
    sigma_ref: float,
    batch: int = 32,
) -> dict[str, Any]:
    import torch

    run_dir = out / name
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "name": name,
                "seed": seed,
                "epochs": epochs,
                "hidden": hidden,
                "n_coupling": n_coupling,
                "action_mode": action_mode,
                "action_weight": action_weight,
                "sigma_ref": sigma_ref,
            },
            indent=2,
        )
    )
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
    kt_np = kernel_fft(stencil, 16, ctx["kernel"].eta)
    kt = torch.tensor(kt_np, dtype=torch.complex64)
    torch.manual_seed(seed)
    model = build_patch_model(hidden, n_coupling)
    opt = torch.optim.Adam(model.parameters(), lr=1.0e-4)
    rows: list[dict[str, Any]] = []
    best_action: dict[str, Any] | None = None
    best_nll: dict[str, Any] | None = None
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(train_c.shape[0])
        losses = []
        nlls = []
        auxs = []
        model.train()
        for start in range(0, train_c.shape[0], batch):
            b = perm[start : start + batch]
            z, ild = model.inverse(train_d[b], train_c[b])
            nll = -(log_base_torch(z) + ild - train_j[b]).mean()
            aux = nll.new_tensor(0.0)
            if action_mode != "none" and action_weight > 0.0:
                z0 = torch.zeros((len(b), 3, 8, 8), dtype=torch.float32)
                y_flat, _ = model.forward(z0.flatten(1), train_c[b])
                y = y_flat.reshape(len(b), 3, 8, 8)
                phi = torch_reconstruct_phi(train_c[b].reshape(len(b), 1, 8, 8), y, lg, kt)
                ds = torch_action(phi) - torch_action(train_fine[b])
                aux = action_aux_loss(ds, action_mode, sigma_ref)
            loss = nll + float(action_weight) * aux
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
            nlls.append(float(nll.detach()))
            auxs.append(float(aux.detach()))
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
            "action_mode": action_mode,
            "action_weight": action_weight,
        }
        torch.save({"model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, "config": cfg, "epoch": epoch, "val_loss": val_nll, "stage": "patch_detail"}, ckpt)
        metric, _ds, _pred = evaluate_checkpoint(model, lg, c[val_idx], detail[val_idx], fine[val_idx], ctx, name)
        row = {
            "run": name,
            "epoch": epoch,
            "seed": seed,
            "hidden": hidden,
            "n_coupling": n_coupling,
            "action_mode": action_mode,
            "action_weight": action_weight,
            "train_loss": float(np.mean(losses)),
            "train_nll": float(np.mean(nlls)),
            "train_aux": float(np.mean(auxs)),
            "val_nll": val_nll,
            "checkpoint": str(ckpt),
            **metric,
        }
        rows.append(row)
        write_csv(run_dir / "metrics.csv", rows)
        if best_action is None or row["deltaS_std"] < best_action["deltaS_std"]:
            best_action = dict(row)
        if best_nll is None or val_nll < best_nll["val_nll"]:
            best_nll = dict(row)
        print(f"{name} epoch {epoch}/{epochs}: val_nll={val_nll:.6g} deltaS_std={row['deltaS_std']:.6g} max_centered={row['abs_centered_deltaS_max']:.6g}", flush=True)
    assert best_action is not None and best_nll is not None
    shutil.copy2(best_action["checkpoint"], ckpt_dir / "patch_detail_best_action.pt")
    return {"name": name, "best_action": best_action, "best_nll": best_nll, "metrics": rows}


def tail_diagnostics(best: dict[str, Any], out: Path, c: np.ndarray, detail: np.ndarray, fine: np.ndarray, val_idx: np.ndarray, ctx: dict[str, Any]) -> dict[str, Any]:
    import torch

    ckpt = Path(best["checkpoint"])
    run_dir = ckpt.parents[1]
    lg_npz = ckpt.parent / "patch_detail/local_gaussian_coefficients.npz"
    with np.load(lg_npz) as z:
        lg = {"coeffs": z["coeffs"], "sigma": z["sigma"], "ridge": float(z["ridge"])}
    cfg = torch.load(ckpt, map_location="cpu")["config"]
    model = build_patch_model(int(cfg["conv_hidden_channels"]), int(cfg["n_coupling_layers"]))
    model.load_state_dict(torch.load(ckpt, map_location="cpu")["model_state"])
    metric, ds, pred = evaluate_checkpoint(model, lg, c[val_idx], detail[val_idx], fine[val_idx], ctx, "tail_best")
    centered = np.abs(ds - ds.mean())
    order = np.argsort(centered)[::-1]
    worst = order[:50]
    typical = order[len(order) // 2 : len(order) // 2 + 50]
    phi_pred = reconstruct_patch_detail(c[val_idx], pred, ctx)
    ad_pred = action_total(phi_pred, ctx["fine_action"]) / (16 * 16)
    ad_ref = action_total(fine[val_idx], ctx["fine_action"]) / (16 * 16)
    rows = []
    for group, idxs in [("worst50", worst), ("typical50", typical)]:
        amp = np.sqrt(np.mean(pred[idxs] ** 2, axis=(1, 2, 3)))
        target_amp = np.sqrt(np.mean(detail[val_idx][idxs] ** 2, axis=(1, 2, 3)))
        coarse_m = np.mean(c[val_idx][idxs], axis=(1, 2))
        rows.append(
            {
                "group": group,
                "n": len(idxs),
                "abs_centered_deltaS_mean": float(np.mean(centered[idxs])),
                "abs_centered_deltaS_max": float(np.max(centered[idxs])),
                "pred_detail_rms_mean": float(np.mean(amp)),
                "target_detail_rms_mean": float(np.mean(target_amp)),
                "detail_rms_ratio_mean": float(np.mean(amp / np.maximum(target_amp, 1.0e-8))),
                "coarse_m_abs_mean": float(np.mean(np.abs(coarse_m))),
                "action_density_error_mean": float(np.mean(ad_pred[idxs] - ad_ref[idxs])),
                "action_density_error_std": float(np.std(ad_pred[idxs] - ad_ref[idxs])),
            }
        )
    write_csv(out / "tail_diagnostics_summary.csv", rows)
    worst_rows = []
    for rank, j in enumerate(worst[:20], start=1):
        actual = int(val_idx[j])
        worst_rows.append(
            {
                "rank": rank,
                "val_position": int(j),
                "sample_index": actual,
                "deltaS": float(ds[j]),
                "centered_abs_deltaS": float(centered[j]),
                "pred_detail_rms": float(np.sqrt(np.mean(pred[j] ** 2))),
                "target_detail_rms": float(np.sqrt(np.mean(detail[actual] ** 2))),
                "coarse_m": float(np.mean(c[actual])),
                "action_density_error": float(ad_pred[j] - ad_ref[j]),
            }
        )
    write_csv(out / "tail_worst_configurations.csv", worst_rows)
    return {"metric": metric, "summary": rows, "worst": worst_rows}


def write_report(out: Path, all_rows: list[dict[str, Any]], best: dict[str, Any], preflight: dict[str, Any], tail: dict[str, Any], smoke_launched: bool) -> None:
    write_csv(out / "patch_detail_remediation_all_runs.csv", all_rows)
    lines = [
        "# Patch-detail remediation follow-up report",
        "",
        "## Scope",
        "",
        "Bounded follow-up on the fixed non-overlapping 3-channel patch-detail coordinate. No long validation was launched.",
        "",
        "## Preflight",
        "",
        f"- coordinate roundtrip max error: `{preflight['roundtrip_max_error']:.6g}`",
        f"- small3 reblocking max error: `{preflight['reblocking_max_error']:.6g}`",
        "",
        "## Best checkpoint",
        "",
        f"- run: `{best['run']}`",
        f"- epoch: `{best['epoch']}`",
        f"- checkpoint: `{best['checkpoint']}`",
        f"- deltaS mean/std/RMS: `{best['deltaS_mean']:.6g}` / `{best['deltaS_std']:.6g}` / `{best['deltaS_rms']:.6g}`",
        f"- quantiles: q01 `{best['deltaS_q01']:.6g}`, q05 `{best['deltaS_q05']:.6g}`, median `{best['deltaS_median']:.6g}`, q95 `{best['deltaS_q95']:.6g}`, q99 `{best['deltaS_q99']:.6g}`",
        f"- deltaS min/max: `{best['deltaS_min']:.6g}` / `{best['deltaS_max']:.6g}`",
        f"- max |deltaS - mean|: `{best['abs_centered_deltaS_max']:.6g}`",
        "",
        "## Run table",
        "",
        "| run | best epoch | loss mode | hidden | couplings | best deltaS std | best q99 | best max centered |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    by_run: dict[str, dict[str, Any]] = {}
    for row in all_rows:
        if row["run"] not in by_run or row["deltaS_std"] < by_run[row["run"]]["deltaS_std"]:
            by_run[row["run"]] = row
    for row in by_run.values():
        lines.append(
            f"| `{row['run']}` | {row['epoch']} | `{row['action_mode']}` | {row['hidden']} | {row['n_coupling']} | {row['deltaS_std']:.6g} | {row['deltaS_q99']:.6g} | {row['abs_centered_deltaS_max']:.6g} |"
        )
    lines += [
        "",
        "## Tail diagnostics",
        "",
        "Worst configurations have been written to `tail_worst_configurations.csv`; group summaries are in `tail_diagnostics_summary.csv`.",
    ]
    for row in tail["summary"]:
        lines.append(
            f"- `{row['group']}`: mean |centered deltaS| `{row['abs_centered_deltaS_mean']:.6g}`, detail RMS ratio `{row['detail_rms_ratio_mean']:.6g}`, |coarse m| `{row['coarse_m_abs_mean']:.6g}`, action-density error mean `{row['action_density_error_mean']:.6g}`."
        )
    recommendation = "no smoke"
    if best["deltaS_std"] <= 12.0:
        recommendation = "smoke gate met; run only short sampler smoke after patch-detail sampler integration is available"
    elif best["deltaS_std"] < 12.2557:
        recommendation = "improved but still above smoke gate; continue with one targeted change"
    else:
        recommendation = "saturated under bounded scans"
    lines += [
        "",
        "## Final recommendation",
        "",
        f"- sampler smoke launched: `{smoke_launched}`",
        f"- recommendation: `{recommendation}`",
    ]
    (out / "PATCH_DETAIL_REMEDIATION_FOLLOWUP_REPORT.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paired-data", type=Path, default=PROJECT_ROOT / "perfect_blocking_upsampling/outputs/lam0p5_small3_8to16/paired_data/paired_lam0p5_small3_L16_to_L8.npz")
    ap.add_argument("--run-dir", type=Path, default=PROJECT_ROOT / "perfect_blocking_upsampling/outputs/lam0p5_small3_8to16/full_training/run_20260630_210838")
    ap.add_argument("--branch-root", type=Path, default=PROJECT_ROOT / "perfect_blocking_upsampling/outputs/lam0p5_small3_8to16/remediation/patch_detail_parameterization")
    ap.add_argument("--epochs-seed-scan", type=int, default=32)
    ap.add_argument("--epochs-loss-scan", type=int, default=24)
    ap.add_argument("--epochs-capacity", type=int, default=24)
    ap.add_argument("--overwrite-followup", action="store_true")
    args = ap.parse_args()
    out = args.branch_root / "followup_remediation"
    if out.exists() and args.overwrite_followup:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    errlog = out / "ERROR_FIX_LOG.md"
    errlog.write_text("# Patch-detail remediation error/fix log\n\n")
    try:
        _cfg, _paths, _coarse, _fine_ref, _cm, _fm, ctx = load_context(args.run_dir)
        arrays = load_paired(args.paired_data)
        c, detail, fine = stack_detail(arrays)
        train_idx = arrays["train_idx"].astype(np.int64)
        val_idx = arrays["val_idx"].astype(np.int64)
        phi_rt = reconstruct_patch_detail(c[:128], detail[:128], ctx)
        preflight = {
            "paired_data": str(args.paired_data),
            "run_dir": str(args.run_dir),
            "roundtrip_max_error": float(np.max(np.abs(phi_rt - fine[:128]))),
            "reblocking_max_error": float(np.max(np.abs(apply_kernel(phi_rt, ctx["kernel"])[:, 0::2, 0::2] - c[:128]))),
            "train_count": int(len(train_idx)),
            "val_count": int(len(val_idx)),
        }
        write_json(out / "preflight_io_checks.json", preflight)
        (out / "PREFLIGHT_IO_CHECKS.md").write_text(
            "# Preflight I/O checks\n\n"
            f"- paired data: `{args.paired_data}`\n"
            f"- train/val: `{len(train_idx)}` / `{len(val_idx)}`\n"
            f"- roundtrip max error: `{preflight['roundtrip_max_error']:.6g}`\n"
            f"- small3 reblocking max error: `{preflight['reblocking_max_error']:.6g}`\n"
        )
        all_rows: list[dict[str, Any]] = []
        runs = []
        for seed in [20260705, 20260706, 20260707]:
            runs.append(("seed_scan", f"seed_scan_nll_seed{seed}", seed, 64, 16, "none", 0.0, args.epochs_seed_scan))
        for mode in ["variance", "huber", "tail"]:
            runs.append(("loss_scan", f"loss_scan_{mode}", 20260725, 64, 16, mode, 5.0, args.epochs_loss_scan))
        results = []
        best: dict[str, Any] | None = None
        for family, name, seed, hidden, nc, mode, weight, epochs in runs:
            result = train_variant(name, out / family, c, detail, fine, train_idx, val_idx, ctx, seed=seed, epochs=epochs, hidden=hidden, n_coupling=nc, action_mode=mode, action_weight=weight, sigma_ref=12.2557)
            results.append(result)
            rows = read_csv((out / family / name / "metrics.csv"))
            typed = []
            for r in rows:
                tr = dict(r)
                for k, v in list(tr.items()):
                    try:
                        tr[k] = float(v)
                    except (ValueError, TypeError):
                        pass
                typed.append(tr)
            all_rows.extend(typed)
            cand = result["best_action"]
            if best is None or cand["deltaS_std"] < best["deltaS_std"]:
                best = cand
        if best is None or best["deltaS_std"] > 12.0:
            cap = train_variant("capacity_hidden96_coupling17", out / "capacity_scan", c, detail, fine, train_idx, val_idx, ctx, seed=20260735, epochs=args.epochs_capacity, hidden=96, n_coupling=17, action_mode="none", action_weight=0.0, sigma_ref=12.2557)
            results.append(cap)
            rows = read_csv(out / "capacity_scan/capacity_hidden96_coupling17/metrics.csv")
            typed = []
            for r in rows:
                tr = dict(r)
                for k, v in list(tr.items()):
                    try:
                        tr[k] = float(v)
                    except (ValueError, TypeError):
                        pass
                typed.append(tr)
            all_rows.extend(typed)
            if cap["best_action"]["deltaS_std"] < best["deltaS_std"]:
                best = cap["best_action"]
        assert best is not None
        tail = tail_diagnostics(best, out, c, detail, fine, val_idx, ctx)
        smoke_launched = False
        write_report(out, all_rows, best, preflight, tail, smoke_launched)
        write_json(out / "followup_summary.json", {"best": best, "preflight": preflight, "smoke_launched": smoke_launched})
        print(json.dumps({"best_run": best["run"], "best_epoch": best["epoch"], "best_deltaS_std": best["deltaS_std"], "smoke_launched": smoke_launched}, indent=2), flush=True)
        return 0
    except Exception as exc:
        append_error(errlog, "fatal", exc)
        raise


if __name__ == "__main__":
    import torch

    raise SystemExit(main())
