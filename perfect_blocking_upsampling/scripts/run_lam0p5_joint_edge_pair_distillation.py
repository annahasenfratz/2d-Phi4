#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
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
from perfect_blocking_upsampling.kernels import apply_kernel, inverse_kernel  # noqa: E402
from perfect_blocking_upsampling.observables import observables as ensemble_observables  # noqa: E402
from run_lam0p5_detail_remediation import build_candidate_ctx, evaluate_candidate, load_stage_config, save_lg  # noqa: E402
from run_staged_decimated_conditional_fillin import fit_generic_local_gaussian, log_jacobian, to_model_space  # noqa: E402
from train_faithful_transported_detail import build_detail_model, log_base_torch  # noqa: E402

LAM = 0.5
KAPPA = 0.3426
ETA = 0.25


def metric_row(label: str, k: int, edge: np.ndarray, pair: np.ndarray, c: np.ndarray, corner: np.ndarray, edge_target: np.ndarray, pair_target: np.ndarray, target_phi: np.ndarray, ctx: dict[str, Any], acceptance: float | str = "") -> dict[str, Any]:
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
    return {
        "label": label,
        "K": k,
        "acceptance": acceptance,
        "deltaS_mean": qstats(ds)["mean"],
        "deltaS_std": qstats(ds)["std"],
        "deltaS_rms": float(np.sqrt(np.mean(ds * ds))),
        "local_action_density_shift_mean": qstats(dens - denst)["mean"],
        "local_action_density_shift_std": qstats(dens - denst)["std"],
        "edge_rmse_vs_target": rmse(edge, edge_target),
        "edge_corr_vs_target": corr(edge, edge_target),
        "pair_rmse_vs_target": rmse(pair, pair_target),
        "pair_corr_vs_target": corr(pair, pair_target),
        "reblocking_error_max": float(np.max(reb)),
        "phi2_shift": float(obs["phi2"] - obst["phi2"]),
        "phi4_shift": float(obs["phi4"] - obst["phi4"]),
        "NN_shift": float(obs["NN"] - obst["NN"]),
        "action_density_shift": float(obs["action_density"] - obst["action_density"]),
        "nan_or_inf": bool((not np.isfinite(ds).all()) or (not np.isfinite(edge).all()) or (not np.isfinite(pair).all())),
    }


def fixed_joint_edge_pair_mcmc(
    *,
    initial_edge: np.ndarray,
    initial_pair: np.ndarray,
    c: np.ndarray,
    corner: np.ndarray,
    target_phi: np.ndarray,
    ctx: dict[str, Any],
    k_values: list[int],
    proposal_sigma_edge: float,
    proposal_sigma_pair: float,
    seed: int,
) -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], list[dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    edge = initial_edge.astype(np.float32).copy()
    pair = initial_pair.astype(np.float32).copy()
    n, _ch, l, _ = edge.shape
    phi, _ = inverse_kernel(reconstruct(c, edge, pair, corner), ctx["kernel"])
    current_s = action_total(phi, ctx["fine_action"])
    target_s = action_total(target_phi, ctx["fine_action"])
    kset = set(k_values)
    samples: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    traj: list[dict[str, Any]] = []
    total_prop = 0
    total_acc = 0
    if 0 in kset:
        samples[0] = (edge.copy(), pair.copy())
        ds = current_s - target_s
        traj.append({"K": 0, "sweep_acceptance": "", "cumulative_acceptance": "", "deltaS_std": qstats(ds)["std"], "deltaS_mean": qstats(ds)["mean"]})
    for sweep in range(1, max(k_values) + 1):
        sweep_prop = 0
        sweep_acc = 0
        for i in range(l):
            for j in range(l):
                prop_e = edge.copy()
                prop_p = pair.copy()
                prop_e[:, 0, i, j] += proposal_sigma_edge * rng.standard_normal(n).astype(np.float32)
                prop_p[:, 0, i, j] += proposal_sigma_pair * rng.standard_normal(n).astype(np.float32)
                prop_phi, _ = inverse_kernel(reconstruct(c, prop_e, prop_p, corner), ctx["kernel"])
                prop_s = action_total(prop_phi, ctx["fine_action"])
                dlog = -(prop_s - current_s)
                accept = np.log(rng.random(n)) < np.minimum(0.0, dlog)
                if np.any(accept):
                    edge[accept, 0, i, j] = prop_e[accept, 0, i, j]
                    pair[accept, 0, i, j] = prop_p[accept, 0, i, j]
                    current_s[accept] = prop_s[accept]
                na = int(np.sum(accept))
                sweep_acc += na
                total_acc += na
                sweep_prop += n
                total_prop += n
        ds = current_s - target_s
        traj.append({
            "K": sweep,
            "sweep_acceptance": sweep_acc / sweep_prop if sweep_prop else float("nan"),
            "cumulative_acceptance": total_acc / total_prop if total_prop else float("nan"),
            "deltaS_std": qstats(ds)["std"],
            "deltaS_mean": qstats(ds)["mean"],
        })
        if sweep in kset:
            samples[sweep] = (edge.copy(), pair.copy())
    return samples, traj


def split_indices(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_val = max(64, int(round(0.2 * n)))
    return order[n_val:], order[:n_val]


def train_stage_on_corrected(stage: str, cond_all: np.ndarray, target_all: np.ndarray, out: Path, epochs: int, seed: int) -> dict[str, Any]:
    import torch

    cfg = load_stage_config(stage)
    tr = cfg["training"]
    batch = int(tr["batch_size"])
    train_sel, val_sel = split_indices(cond_all.shape[0], seed)
    cond_train, target_train = cond_all[train_sel], target_all[train_sel]
    cond_val, target_val = cond_all[val_sel], target_all[val_sel]
    lg = fit_generic_local_gaussian(cond_train, target_train, float(tr.get("local_gaussian_sigma_floor", 1.0e-4)))
    stage_name = "corner" if stage == "corner_body" else stage
    ckpt_dir = out / "checkpoints"
    save_lg(ckpt_dir / stage_name / "local_gaussian_coefficients.npz", lg)
    train_u = to_model_space(target_train, cond_train, lg)
    val_u = to_model_space(target_val, cond_val, lg)
    train_j = torch.tensor(log_jacobian(cond_train, lg), dtype=torch.float32)
    val_j = torch.tensor(log_jacobian(cond_val, lg), dtype=torch.float32)
    train_c = torch.tensor(cond_train.reshape(cond_train.shape[0], -1), dtype=torch.float32)
    train_d = torch.tensor(train_u.reshape(train_u.shape[0], -1), dtype=torch.float32)
    val_c = torch.tensor(cond_val.reshape(cond_val.shape[0], -1), dtype=torch.float32)
    val_d = torch.tensor(val_u.reshape(val_u.shape[0], -1), dtype=torch.float32)
    torch.manual_seed(seed)
    model = build_detail_model(stage, cfg)
    opt = torch.optim.Adam(model.parameters(), lr=float(tr["learning_rate"]))
    rows: list[dict[str, Any]] = []
    best = {"val_loss": float("inf"), "epoch": 0, "state": None}
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(train_c.shape[0])
        losses = []
        model.train()
        for start in range(0, train_c.shape[0], batch):
            b = perm[start : start + batch]
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
        row = {"stage": stage_name, "epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": val_loss}
        rows.append(row)
        write_csv(out / f"{stage_name}_train_log.csv", rows)
        if val_loss < best["val_loss"]:
            best = {"val_loss": val_loss, "epoch": epoch, "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
        torch.save(
            {
                "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "config": cfg["architecture"] | {"lambda_": LAM, "kappa": KAPPA, "eta": ETA, "stage": stage_name, "lattice_size": 8},
                "stage": stage_name,
                "epoch": epoch,
                "val_loss": val_loss,
                "selection": "joint_edge_pair_corrected_epoch",
            },
            ckpt_dir / f"{stage_name}_epoch{epoch:04d}.pt",
        )
        print(f"joint_distill_{stage_name} epoch {epoch}/{epochs}: val_loss={val_loss:.6g}", flush=True)
    final = ckpt_dir / f"{stage_name}.pt"
    torch.save(
        {
            "model_state": best["state"],
            "config": cfg["architecture"] | {"lambda_": LAM, "kappa": KAPPA, "eta": ETA, "stage": stage_name, "lattice_size": 8},
            "stage": stage_name,
            "epoch": best["epoch"],
            "val_loss": best["val_loss"],
            "selection": "best_val_nll_on_joint_corrected_targets",
        },
        final,
    )
    return {
        "stage": stage_name,
        "checkpoint": str(final),
        "local_gaussian": str(ckpt_dir / stage_name / "local_gaussian_coefficients.npz"),
        "best_epoch": int(best["epoch"]),
        "best_val_loss": float(best["val_loss"]),
        "train_count": int(len(train_sel)),
        "val_count": int(len(val_sel)),
    }


def write_status(summary: dict[str, Any]) -> None:
    path = PROJECT_ROOT / "perfect_blocking_upsampling/outputs/lam0p5_small3_8to16/STATUS.md"
    section = f"""

## Joint edge+pair fixed-coarse MCMC distillation branch

Status: `{summary['status']}`.

Output:

`full_training/run_20260630_210838/remediation/mcmc_detail_distillation_joint_edge_pair/`

Completed:

- held coarse field and target corner/body fixed;
- updated edge and pair jointly with fixed-coarse random-walk Metropolis;
- scanned correction lengths `K={summary['k_values']}`;
- only built corrected joint data and small sequential distillation if the teacher cleared the material-improvement gate;
- did not run long validation.

Key results:

- initial joint edge+pair deltaS std: `{summary['initial_deltaS_std']:.6g}`;
- best K: `{summary['best_K']}`;
- best teacher deltaS std: `{summary['best_teacher_deltaS_std']:.6g}`;
- material teacher improvement: `{summary['teacher_material_improvement']}`;
- corrected joint dataset written: `{summary['dataset_written']}`;
- sequential distillation run: `{summary['sequential_distillation_run']}`;
- tiny sampler smoke launched: `{summary['sampler_smoke_launched']}`.

Interpretation:

{summary['interpretation']}

Reports:

- `full_training/run_20260630_210838/remediation/mcmc_detail_distillation_joint_edge_pair/JOINT_EDGE_PAIR_TARGET_DEFINITION.md`
- `full_training/run_20260630_210838/remediation/mcmc_detail_distillation_joint_edge_pair/JOINT_EDGE_PAIR_MCMC_TEACHER_REPORT.md`
- `full_training/run_20260630_210838/remediation/mcmc_detail_distillation_joint_edge_pair/JOINT_EDGE_PAIR_DISTILLATION_REPORT.md`
"""
    text = path.read_text()
    marker = "\n## Joint edge+pair fixed-coarse MCMC distillation branch\n"
    if marker in text:
        text = text[: text.index(marker)]
    path.write_text(text.rstrip() + section + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=PROJECT_ROOT / "perfect_blocking_upsampling/outputs/lam0p5_small3_8to16/full_training/run_20260630_210838")
    ap.add_argument("--teacher-samples", type=int, default=128)
    ap.add_argument("--dataset-samples", type=int, default=1024)
    ap.add_argument("--mcmc-steps", default="0,5,10,25,50,100")
    ap.add_argument("--proposal-sigma-edge", type=float, default=0.08)
    ap.add_argument("--proposal-sigma-pair", type=float, default=0.08)
    ap.add_argument("--distill-epochs", type=int, default=12)
    ap.add_argument("--seed", type=int, default=20260702)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out = args.run_dir / "remediation/mcmc_detail_distillation_joint_edge_pair"
    if out.exists() and args.overwrite:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    k_values = sorted({int(x) for x in args.mcmc_steps.split(",") if x.strip()})
    if 0 not in k_values:
        k_values = [0] + k_values

    _cfg, _paths, _coarse, _fine_ref, _cm, _fm, ctx = load_context(args.run_dir)
    arrays = load_paired()
    edge_model, edge_lg, *_ = ctx["stages"]["edge"]
    pair_model, pair_lg, *_ = ctx["stages"]["pair"]

    (out / "JOINT_EDGE_PAIR_TARGET_DEFINITION.md").write_text(
        "# Joint edge+pair fixed-coarse MCMC target\n\n"
        "At fixed paired coarse field `u=c00`, this diagnostic holds the coarse-refine/coarse slot fixed and keeps the corner/body transported detail at its paired target value. It jointly updates edge `d10` and pair `d01` transported-detail variables.\n\n"
        "The teacher target is proportional to\n\n"
        "`exp(-S_f(phi(c00, d10, d01, d11_target)))`.\n\n"
        "The proposal is symmetric in edge/pair detail coordinates, so the Metropolis accept rule uses only `-Delta S_f`. No coarse action, coarse proposal, or full native-L8 patch terms are included because this is not a coarse Markov update. The supervised target, if the teacher materially improves action compatibility, is the corrected `(edge,pair)` sample after the selected fixed-coarse correction length.\n"
    )

    vi = arrays["val_idx"][: min(args.teacher_samples, len(arrays["val_idx"]))]
    c = arrays["c00"][vi].astype(np.float32)
    e_t = arrays["edge_x"][vi, None].astype(np.float32)
    p_t = arrays["edge_y"][vi, None].astype(np.float32)
    co_t = arrays["corner"][vi, None].astype(np.float32)
    target_phi, _ = inverse_kernel(reconstruct(c, e_t, p_t, co_t), ctx["kernel"])
    z0 = np.zeros_like(e_t, dtype=np.float32)
    e0, *_ = stage_forward_z(edge_model, z0, c[:, None], edge_lg)
    p0, *_ = stage_forward_z(pair_model, z0, np.concatenate([c[:, None], e0], axis=1), pair_lg)
    samples, traj = fixed_joint_edge_pair_mcmc(
        initial_edge=e0,
        initial_pair=p0,
        c=c,
        corner=co_t,
        target_phi=target_phi,
        ctx=ctx,
        k_values=k_values,
        proposal_sigma_edge=args.proposal_sigma_edge,
        proposal_sigma_pair=args.proposal_sigma_pair,
        seed=args.seed,
    )
    rows = []
    for k in k_values:
        ee, pp = samples[k]
        acc = next((r["cumulative_acceptance"] for r in traj if r["K"] == k), "")
        rows.append(metric_row(f"model_edge_pair_plus_mcmc_K{k}", k, ee, pp, c, co_t, e_t, p_t, target_phi, ctx, acc))
    rows.append(metric_row("target_all_reference", 0, e_t, p_t, c, co_t, e_t, p_t, target_phi, ctx, ""))
    write_csv(out / "joint_edge_pair_teacher_metrics.csv", rows)
    write_csv(out / "joint_edge_pair_trajectory_summary.csv", traj)
    initial = [r for r in rows if r["K"] == 0 and r["label"].startswith("model")][0]
    candidates = [r for r in rows if int(r["K"]) > 0 and not r["nan_or_inf"]]
    best = min(candidates, key=lambda r: float(r["deltaS_std"])) if candidates else initial
    teacher_improved = float(best["deltaS_std"]) < float(initial["deltaS_std"])
    material = float(best["deltaS_std"]) < 0.85 * float(initial["deltaS_std"])
    (out / "JOINT_EDGE_PAIR_MCMC_TEACHER_REPORT.md").write_text(
        "# Joint edge+pair MCMC teacher report\n\n"
        f"- teacher samples: `{len(vi)}`\n"
        f"- proposal sigma edge/pair: `{args.proposal_sigma_edge}` / `{args.proposal_sigma_pair}`\n"
        f"- K values: `{k_values}`\n"
        f"- initial joint edge+pair deltaS std: `{float(initial['deltaS_std']):.6g}`\n"
        f"- best K: `{best['K']}`\n"
        f"- best teacher deltaS std: `{float(best['deltaS_std']):.6g}`\n"
        f"- teacher improved: `{teacher_improved}`\n"
        f"- material improvement: `{material}`\n"
        "- corner/body is held at paired target values throughout this teacher.\n"
    )

    dataset_written = False
    sequential_run = False
    distill_results: list[dict[str, Any]] = []
    cumulative_rows: list[dict[str, Any]] = []
    if material:
        n = min(args.dataset_samples, len(arrays["train_idx"]))
        di = arrays["train_idx"][:n]
        dc = arrays["c00"][di].astype(np.float32)
        de_t = arrays["edge_x"][di, None].astype(np.float32)
        dp_t = arrays["edge_y"][di, None].astype(np.float32)
        dco_t = arrays["corner"][di, None].astype(np.float32)
        dphi_t, _ = inverse_kernel(reconstruct(dc, de_t, dp_t, dco_t), ctx["kernel"])
        dz0 = np.zeros_like(de_t, dtype=np.float32)
        de0, *_ = stage_forward_z(edge_model, dz0, dc[:, None], edge_lg)
        dp0, *_ = stage_forward_z(pair_model, dz0, np.concatenate([dc[:, None], de0], axis=1), pair_lg)
        dsamples, dtraj = fixed_joint_edge_pair_mcmc(
            initial_edge=de0,
            initial_pair=dp0,
            c=dc,
            corner=dco_t,
            target_phi=dphi_t,
            ctx=ctx,
            k_values=[0, int(best["K"])],
            proposal_sigma_edge=args.proposal_sigma_edge,
            proposal_sigma_pair=args.proposal_sigma_pair,
            seed=args.seed + 101,
        )
        ce, cp = dsamples[int(best["K"])]
        ds_dir = out / "corrected_joint_edge_pair_dataset"
        ds_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            ds_dir / "corrected_joint_edge_pair_dataset.npz",
            paired_indices=di,
            conditioning_c00=dc,
            corrected_edge=ce,
            corrected_pair=cp,
            original_target_edge=de_t,
            original_target_pair=dp_t,
            initial_model_edge=de0,
            initial_model_pair=dp0,
            target_corner=dco_t,
            best_K=np.asarray(int(best["K"])),
        )
        write_csv(ds_dir / "dataset_mcmc_trajectory_summary.csv", dtraj)
        ds_rows = [
            metric_row("dataset_initial_model_edge_pair", 0, de0, dp0, dc, dco_t, de_t, dp_t, dphi_t, ctx, ""),
            metric_row("dataset_corrected_joint_edge_pair", int(best["K"]), ce, cp, dc, dco_t, de_t, dp_t, dphi_t, ctx, ""),
        ]
        write_csv(ds_dir / "corrected_joint_edge_pair_dataset_metrics.csv", ds_rows)
        (out / "CORRECTED_JOINT_EDGE_PAIR_DATASET_REPORT.md").write_text(
            "# Corrected joint edge+pair dataset report\n\n"
            f"- samples: `{n}`\n"
            f"- selected K: `{best['K']}`\n"
            f"- initial deltaS std: `{float(ds_rows[0]['deltaS_std']):.6g}`\n"
            f"- corrected deltaS std: `{float(ds_rows[1]['deltaS_std']):.6g}`\n"
            "- saved file: `corrected_joint_edge_pair_dataset/corrected_joint_edge_pair_dataset.npz`\n"
        )
        dataset_written = True

        distill_dir = out / "sequential_distillation_smoke"
        edge_res = train_stage_on_corrected("edge", dc[:, None], ce, distill_dir / "edge", args.distill_epochs, args.seed + 201)
        pair_cond = np.concatenate([dc[:, None], ce], axis=1)
        pair_res = train_stage_on_corrected("pair", pair_cond, cp, distill_dir / "pair", args.distill_epochs, args.seed + 202)
        distill_results = [edge_res, pair_res]
        sequential_run = True
        cctx = build_candidate_ctx(ctx, {
            "edge": (Path(edge_res["checkpoint"]), Path(edge_res["local_gaussian"])),
            "pair": (Path(pair_res["checkpoint"]), Path(pair_res["local_gaussian"])),
        })
        cumulative = evaluate_candidate("sequential_joint_distilled_edge_pair", cctx, arrays, out, args.teacher_samples)
        cumulative_rows = cumulative["assembly"]
        write_csv(out / "joint_distilled_cumulative_metrics.csv", cumulative_rows)
        vals = {r["assembly"]: r["delta_action_std_vs_fine16"] for r in cumulative_rows}
        (out / "DISTILLED_JOINT_EDGE_PAIR_CUMULATIVE_REPORT.md").write_text(
            "# Distilled joint edge+pair cumulative report\n\n"
            f"- model coarse only deltaS std: `{vals.get('model_coarse_only'):.6g}`\n"
            f"- model coarse + distilled edge deltaS std: `{vals.get('model_coarse_edge'):.6g}`\n"
            f"- model coarse + distilled edge+pair deltaS std: `{vals.get('model_coarse_edge_pair'):.6g}`\n"
            f"- full model with baseline corner/body deltaS std: `{vals.get('model_all_z0'):.6g}`\n"
            "- no sampler smoke was launched automatically.\n"
        )
        write_json(distill_dir / "sequential_distillation_summary.json", {"edge": edge_res, "pair": pair_res})
    else:
        (out / "CORRECTED_JOINT_EDGE_PAIR_DATASET_REPORT.md").write_text(
            "# Corrected joint edge+pair dataset report\n\n"
            "No corrected joint dataset was generated because the teacher did not clear the material-improvement gate.\n"
        )
        write_csv(out / "joint_distilled_cumulative_metrics.csv", [])
        (out / "DISTILLED_JOINT_EDGE_PAIR_CUMULATIVE_REPORT.md").write_text(
            "# Distilled joint edge+pair cumulative report\n\n"
            "No distillation was run because the joint teacher did not clear the material-improvement gate.\n"
        )

    (out / "DISTILLATION_DESIGN_OPTIONS.md").write_text(
        "# Joint edge+pair distillation design options\n\n"
        "## Sequential distillation\n\n"
        "Train edge on corrected joint edge samples, then train pair on corrected pair samples conditioned on corrected edge. This option is implemented as a small smoke if the teacher clears the improvement gate. It preserves existing stage APIs but still uses corrected joint targets.\n\n"
        "## Joint two-output model\n\n"
        "Use one shared conditioner/flow producing both edge and pair variables. This was prepared conceptually but not implemented in this diagnostic run. It is the more direct replacement if sequential distillation cannot preserve the teacher improvement.\n"
    )

    sampler_smoke = False
    interpretation = (
        "Joint fixed-coarse MCMC materially improved the edge+pair action mismatch, so a corrected joint dataset and small sequential distillation were attempted."
        if material
        else "Joint fixed-coarse MCMC did not materially improve the edge+pair action mismatch. The issue is not resolved by short edge+pair conditional correction in the current coordinates."
    )
    if cumulative_rows:
        vals = {r["assembly"]: r["delta_action_std_vs_fine16"] for r in cumulative_rows}
        if float(vals.get("model_coarse_edge_pair", float("inf"))) < 0.85 * float(initial["deltaS_std"]):
            interpretation += " Sequential distillation improved cumulative diagnostics enough to consider a manually approved tiny sampler smoke, but none was launched."
        else:
            interpretation += " Sequential distillation did not preserve the teacher improvement in cumulative diagnostics, so no sampler smoke was launched."

    summary = {
        "status": "completed",
        "k_values": k_values,
        "initial_deltaS_std": float(initial["deltaS_std"]),
        "best_K": int(best["K"]),
        "best_teacher_deltaS_std": float(best["deltaS_std"]),
        "teacher_improved": bool(teacher_improved),
        "teacher_material_improvement": bool(material),
        "dataset_written": bool(dataset_written),
        "sequential_distillation_run": bool(sequential_run),
        "distill_results": distill_results,
        "sampler_smoke_launched": sampler_smoke,
        "interpretation": interpretation,
    }
    write_json(out / "joint_edge_pair_distillation_summary.json", summary)
    (out / "JOINT_EDGE_PAIR_DISTILLATION_REPORT.md").write_text(
        "# Joint edge+pair distillation report\n\n"
        f"1. Does joint edge+pair fixed-coarse MCMC improve the action mismatch?\n\n   Improved: `{teacher_improved}`. Material improvement: `{material}`. Initial std `{float(initial['deltaS_std']):.6g}`, best teacher std `{float(best['deltaS_std']):.6g}`.\n\n"
        f"2. How many MCMC steps are needed?\n\n   Best observed K was `{best['K']}` over `{k_values}`.\n\n"
        f"3. Is the improvement distillable?\n\n   Sequential distillation run: `{sequential_run}`. See `DISTILLED_JOINT_EDGE_PAIR_CUMULATIVE_REPORT.md` if run.\n\n"
        "4. Does this suggest replacing the edge/pair factorization?\n\n   If the teacher improves but sequential distillation does not, yes: this points toward a joint two-output edge+pair model or a different transported-detail coordinate.\n\n"
        f"5. Recommendation\n\n   {interpretation}\n"
    )
    write_status(summary)
    print(json.dumps({"status": "completed", "out": str(out), "best_deltaS_std": float(best["deltaS_std"]), "material": material, "sequential_distillation_run": sequential_run}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
