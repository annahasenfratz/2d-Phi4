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
    stage_inverse_target,
    write_csv,
    write_json,
)
from perfect_blocking_upsampling.actions import action_density, action_total  # noqa: E402
from perfect_blocking_upsampling.kernels import apply_kernel, inverse_kernel  # noqa: E402
from perfect_blocking_upsampling.observables import observables as ensemble_observables  # noqa: E402
from run_lam0p5_detail_remediation import build_candidate_ctx, evaluate_candidate, load_stage_config, save_lg  # noqa: E402
from run_staged_decimated_conditional_fillin import fit_generic_local_gaussian, log_jacobian, to_model_space  # noqa: E402
from train_faithful_transported_detail import build_detail_model, log_base_torch  # noqa: E402

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

LAM = 0.5
KAPPA = 0.3426
ETA = 0.25
PREVIOUS_BEST_EDGE_STD = 8.2555


def action_summary(edge: np.ndarray, c: np.ndarray, pair: np.ndarray, corner: np.ndarray, target_phi: np.ndarray, ctx: dict[str, Any]) -> dict[str, Any]:
    phi, _ = inverse_kernel(reconstruct(c, edge, pair, corner), ctx["kernel"])
    s = action_total(phi, ctx["fine_action"])
    st = action_total(target_phi, ctx["fine_action"])
    ds = s - st
    dens = action_density(phi, ctx["fine_action"]).mean(axis=(1, 2))
    denst = action_density(target_phi, ctx["fine_action"]).mean(axis=(1, 2))
    obs = ensemble_observables(phi, ctx["fine_action"])
    obst = ensemble_observables(target_phi, ctx["fine_action"])
    blocked = apply_kernel(phi, ctx["kernel"])
    reblock_err = np.max(np.abs(blocked[:, 0::2, 0::2] - c), axis=(1, 2))
    return {
        "deltaS": ds,
        "action_density_shift": dens - denst,
        "phi": phi,
        "reblock_err": reblock_err,
        "obs_shift_phi2": float(obs["phi2"] - obst["phi2"]),
        "obs_shift_phi4": float(obs["phi4"] - obst["phi4"]),
        "obs_shift_NN": float(obs["NN"] - obst["NN"]),
        "obs_shift_action_density": float(obs["action_density"] - obst["action_density"]),
    }


def edge_metric_row(label: str, edge: np.ndarray, c: np.ndarray, pair: np.ndarray, corner: np.ndarray, edge_target: np.ndarray, target_phi: np.ndarray, ctx: dict[str, Any], accept: float | None = None, k: int | None = None) -> dict[str, Any]:
    summ = action_summary(edge, c, pair, corner, target_phi, ctx)
    ds = summ["deltaS"]
    ad = summ["action_density_shift"]
    return {
        "label": label,
        "K": k if k is not None else "",
        "acceptance": accept if accept is not None else "",
        "deltaS_mean": qstats(ds)["mean"],
        "deltaS_std": qstats(ds)["std"],
        "deltaS_rms": float(np.sqrt(np.mean(ds * ds))),
        "action_density_shift_mean": qstats(ad)["mean"],
        "action_density_shift_std": qstats(ad)["std"],
        "edge_rmse_vs_target": rmse(edge, edge_target),
        "edge_corr_vs_target": corr(edge, edge_target),
        "reblocking_error_max": float(np.max(summ["reblock_err"])),
        "nan_or_inf": bool((not np.isfinite(ds).all()) or (not np.isfinite(edge).all())),
        "obs_shift_phi2": summ["obs_shift_phi2"],
        "obs_shift_phi4": summ["obs_shift_phi4"],
        "obs_shift_NN": summ["obs_shift_NN"],
        "obs_shift_action_density": summ["obs_shift_action_density"],
    }


def fixed_edge_mcmc(
    *,
    initial_edge: np.ndarray,
    c: np.ndarray,
    pair: np.ndarray,
    corner: np.ndarray,
    target_phi: np.ndarray,
    ctx: dict[str, Any],
    k_values: list[int],
    proposal_sigma: float,
    seed: int,
) -> tuple[dict[int, np.ndarray], list[dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    edge = initial_edge.astype(np.float32).copy()
    n, _ch, l, _ = edge.shape
    current_phi, _ = inverse_kernel(reconstruct(c, edge, pair, corner), ctx["kernel"])
    current_s = action_total(current_phi, ctx["fine_action"])
    samples: dict[int, np.ndarray] = {}
    trajectory: list[dict[str, Any]] = []
    kset = set(k_values)
    total_prop = 0
    total_acc = 0
    if 0 in kset:
        samples[0] = edge.copy()
        ds0 = current_s - action_total(target_phi, ctx["fine_action"])
        trajectory.append({"K": 0, "sweep_acceptance": "", "cumulative_acceptance": "", "deltaS_std": qstats(ds0)["std"], "deltaS_mean": qstats(ds0)["mean"]})
    for sweep in range(1, max(k_values) + 1):
        sweep_prop = 0
        sweep_acc = 0
        for i in range(l):
            for j in range(l):
                prop = edge.copy()
                prop[:, 0, i, j] += proposal_sigma * rng.standard_normal(n).astype(np.float32)
                prop_phi, _ = inverse_kernel(reconstruct(c, prop, pair, corner), ctx["kernel"])
                prop_s = action_total(prop_phi, ctx["fine_action"])
                dlog = -(prop_s - current_s)
                accept = np.log(rng.random(n)) < np.minimum(0.0, dlog)
                if np.any(accept):
                    edge[accept, 0, i, j] = prop[accept, 0, i, j]
                    current_s[accept] = prop_s[accept]
                na = int(np.sum(accept))
                sweep_acc += na
                total_acc += na
                sweep_prop += n
                total_prop += n
        ds = current_s - action_total(target_phi, ctx["fine_action"])
        trajectory.append({
            "K": sweep,
            "sweep_acceptance": sweep_acc / sweep_prop if sweep_prop else float("nan"),
            "cumulative_acceptance": total_acc / total_prop if total_prop else float("nan"),
            "deltaS_std": qstats(ds)["std"],
            "deltaS_mean": qstats(ds)["mean"],
        })
        if sweep in kset:
            samples[sweep] = edge.copy()
    return samples, trajectory


def train_corrected_edge_flow(
    *,
    arrays: dict[str, np.ndarray],
    corrected_edge: np.ndarray,
    corrected_indices: np.ndarray,
    out: Path,
    epochs: int,
    seed: int,
) -> dict[str, Any]:
    import torch

    edge_cfg = load_stage_config("edge")
    tr = edge_cfg["training"]
    batch_size = int(tr["batch_size"])
    rng = np.random.default_rng(seed)
    n = len(corrected_indices)
    order = rng.permutation(n)
    n_val = max(64, int(round(0.2 * n)))
    val_sel = order[:n_val]
    train_sel = order[n_val:]
    cond_all = arrays["c00"][corrected_indices, None].astype(np.float32)
    target_all = corrected_edge.astype(np.float32)
    cond_train, target_train = cond_all[train_sel], target_all[train_sel]
    cond_val, target_val = cond_all[val_sel], target_all[val_sel]
    lg = fit_generic_local_gaussian(cond_train, target_train, float(tr.get("local_gaussian_sigma_floor", 1.0e-4)))
    ckpt_dir = out / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_lg(ckpt_dir / "edge/local_gaussian_coefficients.npz", lg)
    train_u = to_model_space(target_train, cond_train, lg)
    val_u = to_model_space(target_val, cond_val, lg)
    train_j = torch.tensor(log_jacobian(cond_train, lg), dtype=torch.float32)
    val_j = torch.tensor(log_jacobian(cond_val, lg), dtype=torch.float32)
    train_c = torch.tensor(cond_train.reshape(cond_train.shape[0], -1), dtype=torch.float32)
    train_d = torch.tensor(train_u.reshape(train_u.shape[0], -1), dtype=torch.float32)
    val_c = torch.tensor(cond_val.reshape(cond_val.shape[0], -1), dtype=torch.float32)
    val_d = torch.tensor(val_u.reshape(val_u.shape[0], -1), dtype=torch.float32)
    torch.manual_seed(seed)
    model = build_detail_model("edge", edge_cfg)
    opt = torch.optim.Adam(model.parameters(), lr=float(tr["learning_rate"]))
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] = {"val_loss": float("inf"), "epoch": 0, "state": None}
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(train_c.shape[0])
        losses = []
        model.train()
        for start in range(0, train_c.shape[0], batch_size):
            b = perm[start : start + batch_size]
            z, inv_logdet = model.inverse(train_d[b], train_c[b])
            loss = -(log_base_torch(z) + inv_logdet - train_j[b]).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            z_val, inv_logdet_val = model.inverse(val_d, val_c)
            val_loss = float((-(log_base_torch(z_val) + inv_logdet_val - val_j).mean()).detach())
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": val_loss}
        rows.append(row)
        write_csv(out / "distilled_edge_train_log.csv", rows)
        ckpt = ckpt_dir / f"edge_epoch{epoch:04d}.pt"
        torch.save(
            {
                "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "config": edge_cfg["architecture"] | {"lambda_": LAM, "kappa": KAPPA, "eta": ETA, "stage": "edge", "lattice_size": 8},
                "stage": "edge",
                "epoch": epoch,
                "val_loss": val_loss,
                "selection": "corrected_edge_distillation_epoch",
            },
            ckpt,
        )
        if val_loss < best["val_loss"]:
            best = {"val_loss": val_loss, "epoch": epoch, "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, "checkpoint": ckpt}
        print(f"distilled_edge epoch {epoch}/{epochs}: val_loss={val_loss:.6g}", flush=True)
    final = ckpt_dir / "edge.pt"
    torch.save(
        {
            "model_state": best["state"],
            "config": edge_cfg["architecture"] | {"lambda_": LAM, "kappa": KAPPA, "eta": ETA, "stage": "edge", "lattice_size": 8},
            "stage": "edge",
            "epoch": best["epoch"],
            "val_loss": best["val_loss"],
            "selection": "best_val_nll_on_corrected_edge_targets",
        },
        final,
    )
    return {
        "checkpoint": str(final),
        "local_gaussian": str(ckpt_dir / "edge/local_gaussian_coefficients.npz"),
        "best_epoch": int(best["epoch"]),
        "best_val_loss": float(best["val_loss"]),
        "train_count": int(len(train_sel)),
        "val_count": int(len(val_sel)),
    }


def update_status(status_path: Path, branch_summary: dict[str, Any]) -> None:
    section = f"""

## Fixed-coarse MCMC detail distillation branch

Status: `{branch_summary['status']}`.

Output:

`full_training/run_20260630_210838/remediation/mcmc_detail_distillation/`

Completed:

- defined an edge-only fixed-coarse conditional MCMC teacher;
- held true coarse/blocked field and target downstream pair/corner details fixed;
- initialized edge from the failed lambda=0.5 edge model at `z=0`;
- ran short fixed-coarse edge random-walk Metropolis corrections for `K={branch_summary['k_values']}`;
- did not run pair/corner training;
- did not run long validation.

Key results:

- initial model edge deltaS std: `{branch_summary['initial_deltaS_std']:.6g}`;
- best teacher K: `{branch_summary['best_K']}`;
- best teacher edge deltaS std: `{branch_summary['best_teacher_deltaS_std']:.6g}`;
- teacher improved over initial edge: `{branch_summary['teacher_improved']}`;
- corrected-edge dataset written: `{branch_summary['dataset_written']}`;
- distilled edge trained: `{branch_summary['distilled_edge_trained']}`;
- tiny sampler smoke launched: `{branch_summary['sampler_smoke_launched']}`.

Interpretation:

{branch_summary['interpretation']}

Reports:

- `full_training/run_20260630_210838/remediation/mcmc_detail_distillation/DISTILLATION_TARGET_DEFINITION.md`
- `full_training/run_20260630_210838/remediation/mcmc_detail_distillation/EDGE_MCMC_TEACHER_REPORT.md`
- `full_training/run_20260630_210838/remediation/mcmc_detail_distillation/MCMC_DETAIL_DISTILLATION_REPORT.md`
"""
    text = status_path.read_text()
    marker = "\n## Fixed-coarse MCMC detail distillation branch\n"
    if marker in text:
        text = text[: text.index(marker)]
    status_path.write_text(text.rstrip() + section + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=PROJECT_ROOT / "perfect_blocking_upsampling/outputs/lam0p5_small3_8to16/full_training/run_20260630_210838")
    ap.add_argument("--teacher-samples", type=int, default=128)
    ap.add_argument("--dataset-samples", type=int, default=1024)
    ap.add_argument("--mcmc-steps", default="0,5,10,25,50,100")
    ap.add_argument("--proposal-sigma", type=float, default=0.08)
    ap.add_argument("--distill-epochs", type=int, default=12)
    ap.add_argument("--seed", type=int, default=20260701)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out = args.run_dir / "remediation/mcmc_detail_distillation"
    if out.exists() and args.overwrite:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    k_values = sorted({int(x) for x in args.mcmc_steps.split(",") if x.strip()})
    if 0 not in k_values:
        k_values = [0] + k_values

    _cfg, _paths, _coarse, _fine_ref, _cm, _fm, ctx = load_context(args.run_dir)
    arrays = load_paired()
    edge_model, edge_lg, *_ = ctx["stages"]["edge"]

    (out / "DISTILLATION_TARGET_DEFINITION.md").write_text(
        "# Fixed-coarse detail distillation target\n\n"
        "## Option 1: edge-only distillation\n\n"
        "The diagnostic branch fixes the paired blocked/coarse field `u=c00`, fixes downstream pair/corner transported details to their paired target values, and updates only the edge transported-detail variables `d10`.\n\n"
        "For each paired sample the target density sampled by the teacher is proportional to\n\n"
        "`exp(-S_f(phi(c00, d10, d01_target, d11_target)))`,\n\n"
        "with the optimized lambda=0.5 small3 inverse kernel and periodic fine action. The coarse proposal/action terms are not included because coarse is fixed. The MCMC teacher uses a symmetric random-walk Metropolis proposal in edge detail coordinates, so the accept rule only uses `-Delta S_f`.\n\n"
        "The supervised distillation target, if the teacher improves the action metric, is the corrected edge detail after a selected number of fixed-coarse MCMC sweeps. This differs from ordinary NLL training on paired details: the paired edge is one observed blocked-fine detail, while the corrected edge is a short-run conditional action-corrected sample at the same fixed coarse and downstream details.\n\n"
        "## Option 2: joint edge+pair distillation\n\n"
        "Not run in this branch. If edge-only updates fail or do not improve cumulative diagnostics, the next design is joint edge+pair fixed-coarse MCMC with corrected joint targets.\n"
    )

    vi = arrays["val_idx"][: min(args.teacher_samples, len(arrays["val_idx"]))]
    c = arrays["c00"][vi].astype(np.float32)
    edge_target = arrays["edge_x"][vi, None].astype(np.float32)
    pair_target = arrays["edge_y"][vi, None].astype(np.float32)
    corner_target = arrays["corner"][vi, None].astype(np.float32)
    target_phi, _ = inverse_kernel(reconstruct(c, edge_target, pair_target, corner_target), ctx["kernel"])
    z0 = np.zeros_like(edge_target, dtype=np.float32)
    model_edge, model_logq, *_ = stage_forward_z(edge_model, z0, c[:, None], edge_lg)

    samples, trajectory = fixed_edge_mcmc(
        initial_edge=model_edge,
        c=c,
        pair=pair_target,
        corner=corner_target,
        target_phi=target_phi,
        ctx=ctx,
        k_values=k_values,
        proposal_sigma=args.proposal_sigma,
        seed=args.seed,
    )
    rows = []
    for k in k_values:
        acc = next((r["cumulative_acceptance"] for r in trajectory if r["K"] == k), "")
        rows.append(edge_metric_row(f"model_start_plus_mcmc_K{k}", samples[k], c, pair_target, corner_target, edge_target, target_phi, ctx, acc if acc != "" else None, k))
    rows.append(edge_metric_row("true_paired_edge_reference", edge_target, c, pair_target, corner_target, edge_target, target_phi, ctx, None, 0))
    write_csv(out / "edge_mcmc_teacher_metrics.csv", rows)
    write_csv(out / "edge_mcmc_trajectory_summary.csv", trajectory)

    initial = [r for r in rows if r["K"] == 0 and r["label"].startswith("model_start")][0]
    candidates = [r for r in rows if isinstance(r["K"], int) and r["K"] > 0 and not r["nan_or_inf"]]
    best = min(candidates, key=lambda r: float(r["deltaS_std"])) if candidates else initial
    teacher_improved = float(best["deltaS_std"]) < float(initial["deltaS_std"])
    teacher_material = float(best["deltaS_std"]) < 0.85 * float(initial["deltaS_std"])
    teacher_report = [
        "# Edge MCMC teacher report",
        "",
        f"- teacher samples: `{len(vi)}`",
        f"- proposal sigma: `{args.proposal_sigma}`",
        f"- K values: `{k_values}`",
        f"- initial model edge deltaS std: `{float(initial['deltaS_std']):.6g}`",
        f"- best K: `{best['K']}`",
        f"- best teacher deltaS std: `{float(best['deltaS_std']):.6g}`",
        f"- teacher improved over initial: `{teacher_improved}`",
        f"- material improvement: `{teacher_material}`",
        "",
        "The teacher is edge-only and fixed-coarse: only edge variables are updated, while paired downstream pair/corner details remain fixed.",
    ]
    (out / "EDGE_MCMC_TEACHER_REPORT.md").write_text("\n".join(teacher_report) + "\n")

    dataset_written = False
    distilled_edge_trained = False
    cumulative_rows: list[dict[str, Any]] = []
    distill_summary: dict[str, Any] | None = None
    if teacher_improved:
        # Build a corrected-edge dataset with the selected K.
        train_n = min(args.dataset_samples, len(arrays["train_idx"]))
        di = arrays["train_idx"][:train_n]
        dc = arrays["c00"][di].astype(np.float32)
        de_t = arrays["edge_x"][di, None].astype(np.float32)
        dp_t = arrays["edge_y"][di, None].astype(np.float32)
        dco_t = arrays["corner"][di, None].astype(np.float32)
        dphi_t, _ = inverse_kernel(reconstruct(dc, de_t, dp_t, dco_t), ctx["kernel"])
        dz0 = np.zeros_like(de_t, dtype=np.float32)
        de0, *_ = stage_forward_z(edge_model, dz0, dc[:, None], edge_lg)
        dsamples, dtraj = fixed_edge_mcmc(
            initial_edge=de0,
            c=dc,
            pair=dp_t,
            corner=dco_t,
            target_phi=dphi_t,
            ctx=ctx,
            k_values=[0, int(best["K"])],
            proposal_sigma=args.proposal_sigma,
            seed=args.seed + 99,
        )
        corrected = dsamples[int(best["K"])]
        ds_dir = out / "corrected_edge_dataset"
        ds_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            ds_dir / "corrected_edge_dataset.npz",
            paired_indices=di,
            conditioning_c00=dc,
            corrected_edge=corrected,
            original_target_edge=de_t,
            initial_model_edge=de0,
            target_pair=dp_t,
            target_corner=dco_t,
            best_K=np.asarray(int(best["K"])),
            proposal_sigma=np.asarray(args.proposal_sigma),
        )
        write_csv(ds_dir / "dataset_mcmc_trajectory_summary.csv", dtraj)
        ds_metric_initial = edge_metric_row("dataset_initial_model_edge", de0, dc, dp_t, dco_t, de_t, dphi_t, ctx, None, 0)
        ds_metric_corr = edge_metric_row("dataset_corrected_edge", corrected, dc, dp_t, dco_t, de_t, dphi_t, ctx, None, int(best["K"]))
        write_csv(ds_dir / "corrected_edge_dataset_metrics.csv", [ds_metric_initial, ds_metric_corr])
        (out / "CORRECTED_EDGE_DATASET_REPORT.md").write_text(
            "# Corrected edge dataset report\n\n"
            f"- dataset samples: `{train_n}`\n"
            f"- selected K: `{best['K']}`\n"
            f"- initial dataset deltaS std: `{float(ds_metric_initial['deltaS_std']):.6g}`\n"
            f"- corrected dataset deltaS std: `{float(ds_metric_corr['deltaS_std']):.6g}`\n"
            "- saved file: `corrected_edge_dataset/corrected_edge_dataset.npz`\n"
        )
        dataset_written = True

        if float(ds_metric_corr["deltaS_std"]) < float(ds_metric_initial["deltaS_std"]):
            distill_dir = out / "distilled_edge_training"
            distill_summary = train_corrected_edge_flow(
                arrays=arrays,
                corrected_edge=corrected,
                corrected_indices=di,
                out=distill_dir,
                epochs=args.distill_epochs,
                seed=args.seed + 199,
            )
            distilled_edge_trained = True
            cctx = build_candidate_ctx(ctx, {"edge": (Path(distill_summary["checkpoint"]), Path(distill_summary["local_gaussian"]))})
            cumulative = evaluate_candidate("distilled_edge_baseline_pair_corner", cctx, arrays, out, args.teacher_samples)
            cumulative_rows = cumulative["assembly"]
            write_csv(out / "distilled_edge_cumulative_metrics.csv", cumulative_rows)
            vals = {r["assembly"]: r["delta_action_std_vs_fine16"] for r in cumulative_rows}
            (out / "DISTILLED_EDGE_CUMULATIVE_REPORT.md").write_text(
                "# Distilled edge cumulative report\n\n"
                f"- distilled edge checkpoint: `{distill_summary['checkpoint']}`\n"
                f"- model coarse only deltaS std: `{vals.get('model_coarse_only'):.6g}`\n"
                f"- model coarse + distilled edge deltaS std: `{vals.get('model_coarse_edge'):.6g}`\n"
                f"- model coarse + distilled edge + baseline pair deltaS std: `{vals.get('model_coarse_edge_pair'):.6g}`\n"
                f"- full model with baseline pair/corner deltaS std: `{vals.get('model_all_z0'):.6g}`\n"
                "- no sampler smoke was launched automatically.\n"
            )
            (distill_dir / "DISTILLED_EDGE_TRAINING_REPORT.md").write_text(
                "# Distilled edge training report\n\n"
                f"- train/val counts: `{distill_summary['train_count']}` / `{distill_summary['val_count']}`\n"
                f"- best epoch: `{distill_summary['best_epoch']}`\n"
                f"- best validation NLL: `{distill_summary['best_val_loss']:.6g}`\n"
                "- trained same gathered-edge architecture on corrected edge targets.\n"
            )

    if not cumulative_rows:
        write_csv(out / "distilled_edge_cumulative_metrics.csv", [])
        (out / "DISTILLED_EDGE_CUMULATIVE_REPORT.md").write_text(
            "# Distilled edge cumulative report\n\n"
            "No distilled edge cumulative diagnostics were run because the edge-only teacher did not produce a corrected dataset that justified distillation.\n"
        )

    (out / "JOINT_EDGE_PAIR_DISTILLATION_PLAN.md").write_text(
        "# Joint edge+pair distillation plan\n\n"
        "This branch was not run. If edge-only correction is insufficient, the next teacher should update edge and pair jointly at fixed coarse field, with corner/body held at paired target initially. The teacher target would be proportional to `exp(-S_f(phi(c00, d10, d01, d11_target)))`, using symmetric random-walk or block proposals over `(d10,d01)`. Diagnostics should compare corrected joint targets to paired details, then train either a joint flow or coordinated edge/pair stages before any native-L8 sampler smoke.\n"
    )

    sampler_smoke = False
    interpretation = "The fixed-coarse edge teacher improved the edge action metric, so a corrected-edge dataset/distillation branch was attempted." if teacher_improved else "The fixed-coarse edge-only teacher did not improve the edge action metric. Edge-only conditional correction is not sufficient in this setup; the next branch should be joint edge+pair distillation or a different transported-detail parameterization."
    if distilled_edge_trained and cumulative_rows:
        vals = {r["assembly"]: r["delta_action_std_vs_fine16"] for r in cumulative_rows}
        if float(vals.get("model_coarse_edge", float("inf"))) < 0.85 * float(initial["deltaS_std"]):
            interpretation += " Distilled edge cumulative diagnostics improved enough to consider a manually approved tiny sampler smoke, but none was launched by this script."
        else:
            interpretation += " Distilled edge did not materially improve cumulative action diagnostics, so no sampler smoke was launched."

    final = {
        "status": "completed",
        "k_values": k_values,
        "initial_deltaS_std": float(initial["deltaS_std"]),
        "best_K": int(best["K"]),
        "best_teacher_deltaS_std": float(best["deltaS_std"]),
        "teacher_improved": bool(teacher_improved),
        "teacher_material_improvement": bool(teacher_material),
        "dataset_written": bool(dataset_written),
        "distilled_edge_trained": bool(distilled_edge_trained),
        "distill_summary": distill_summary,
        "sampler_smoke_launched": sampler_smoke,
        "interpretation": interpretation,
    }
    write_json(out / "mcmc_detail_distillation_summary.json", final)
    (out / "MCMC_DETAIL_DISTILLATION_REPORT.md").write_text(
        "# MCMC detail distillation report\n\n"
        f"1. Does fixed-coarse edge MCMC improve action compatibility?\n\n   `{teacher_improved}`. Initial deltaS std `{float(initial['deltaS_std']):.6g}`, best teacher std `{float(best['deltaS_std']):.6g}` at K=`{best['K']}`.\n\n"
        f"2. How many MCMC correction steps are needed?\n\n   Best observed K was `{best['K']}` over `{k_values}`.\n\n"
        f"3. Can corrected edge samples be distilled into the same architecture?\n\n   Distillation run: `{distilled_edge_trained}`.\n\n"
        "4. Does distilled edge improve cumulative full-model diagnostics?\n\n   See `DISTILLED_EDGE_CUMULATIVE_REPORT.md` and `distilled_edge_cumulative_metrics.csv` if distillation ran.\n\n"
        f"5. Is a tiny sampler smoke justified?\n\n   `{sampler_smoke}`. No long validation was launched.\n\n"
        f"6. Recommendation\n\n   {interpretation}\n"
    )
    update_status(PROJECT_ROOT / "perfect_blocking_upsampling/outputs/lam0p5_small3_8to16/STATUS.md", final)
    print(json.dumps({"status": "completed", "out": str(out), "teacher_improved": teacher_improved, "best_deltaS_std": float(best["deltaS_std"]), "distilled_edge_trained": distilled_edge_trained}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
