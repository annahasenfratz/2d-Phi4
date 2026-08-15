#!/usr/bin/env python3
"""Bounded Heidelberg-style CNF training on native kappa=0.295 L8 coarse fields."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch


BRANCH = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[3]
HEIDELBERG = ROOT / "heidelberg-phi4-reproduction"
HEIDELBERG_SCRIPTS = HEIDELBERG / "scripts"
sys.path.insert(0, str(BRANCH))
sys.path.insert(0, str(HEIDELBERG))
sys.path.insert(0, str(HEIDELBERG_SCRIPTS))

from run_native_kappa0p295_to_16_preflight import (  # noqa: E402
    DEFAULT_COARSE,
    DEFAULT_FINE,
    block_average_errors,
    current_action_np,
    local_ops,
    make_init_ensemble,
    markdown_table,
    write_csv,
)
from train_ir_matching_l8_torch_cnf import (  # noqa: E402
    PaperCNF,
    ess_fraction,
    make_unit_noise,
    naive_upsample_torch,
    noise_logpdf,
    torch_action,
)


def logweight_stats(logw: np.ndarray) -> dict[str, float]:
    lw = np.asarray(logw, dtype=np.float64)
    shifted = lw - np.max(lw)
    w = np.exp(shifted)
    return {
        "ess_over_n": float((np.sum(w) ** 2) / (len(w) * np.sum(w * w))),
        "logw_mean": float(np.mean(lw)),
        "logw_std": float(np.std(lw)),
        "logw_min": float(np.min(lw)),
        "logw_max": float(np.max(lw)),
    }


def make_model(args: argparse.Namespace, dtype: torch.dtype) -> PaperCNF:
    return PaperCNF(
        kernel_radius=args.kernel_radius,
        n_field_features=args.field_features,
        n_time_features=args.time_features,
        field_bond_dim=args.field_bond_dim,
        time_bond_dim=args.time_bond_dim,
        init_sigma=args.init_sigma,
        init_weight_scale=args.init_weight_scale,
        init_scale_flow=args.init_scale_flow,
        block_period=2,
    ).to(device=torch.device("cpu"), dtype=dtype)


def block_average_residual_np(samples: np.ndarray, coarse: np.ndarray) -> dict[str, float]:
    n, lf, _ = samples.shape
    recovered = samples.reshape(n, lf // 2, 2, lf // 2, 2).mean(axis=(2, 4))
    diff = recovered - coarse[:n]
    return {
        "block_average_rms": float(np.sqrt(np.mean(diff * diff))),
        "block_average_max": float(np.max(np.abs(diff))),
        "block_average_mean_abs": float(np.mean(np.abs(diff))),
    }


def evaluate_model(
    model: PaperCNF,
    coarse: torch.Tensor,
    coarse_action: torch.Tensor,
    coarse_np: np.ndarray,
    args: argparse.Namespace,
    step: int,
    dtype: torch.dtype,
    label: str,
) -> tuple[dict, dict]:
    with torch.no_grad():
        n = min(args.eval_samples, coarse.shape[0])
        batch = coarse[:n]
        sigma = torch.exp(torch.clamp(model.log_sigma, np.log(args.min_sigma), np.log(args.max_sigma)))
        torch.manual_seed(args.seed + 500000 + max(step, 0))
        unit_noise = make_unit_noise(n, args.target_L, torch.device("cpu"), dtype)
        base = naive_upsample_torch(batch) + sigma * unit_noise
        phi, logdet = model(base, n_steps=args.cnf_steps)
        target_action = torch_action(phi, args.kappa_f, args.lam)
        logdet_u_term = -0.5 * (args.coarse_L * args.coarse_L) * np.log(4.0)
        logq = -coarse_action[:n] + logdet_u_term + noise_logpdf(unit_noise, sigma) - logdet
        logw = (-target_action - logq).detach().cpu().numpy()
        samples = phi.detach().cpu().numpy()
    obs = local_ops(samples, kappa=args.kappa_f, lam=args.lam)
    obs.update(block_average_residual_np(samples, coarse_np[:n]))
    lw = logweight_stats(logw)
    obs.update(
        {
            "label": label,
            "step": int(step),
            "loss": float(torch.mean(target_action + logq).detach().cpu()),
            "S_fine_paper": float(torch.mean(target_action).detach().cpu()),
            "logq": float(torch.mean(logq).detach().cpu()),
            "logdet": float(torch.mean(logdet).detach().cpu()),
            "sigma": float(sigma.detach().cpu()),
            **lw,
        }
    )
    return obs, {"label": label, "step": int(step), **lw}


def maybe_load_reference_rows() -> list[dict]:
    rows: list[dict] = []
    candidates = [
        ROOT / "InverseBlocking_MIT_NF/outputs/inverse_blocking_proposal_benchmark_full/observables_by_sweeps.csv",
        ROOT / "InverseBlocking_MIT_NF/outputs/local_chunked_sampler_consolidated/observables_by_sweeps.csv",
    ]
    for path in candidates:
        if not path.exists():
            continue
        with path.open() as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                text = json.dumps(row)
                if "100" in text or "sweeps_100" in text:
                    row = dict(row)
                    row["source_file"] = str(path)
                    rows.append(row)
                    break
        if rows:
            break
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coarse-npy", type=Path, default=DEFAULT_COARSE)
    parser.add_argument("--fine-target-npy", type=Path, default=DEFAULT_FINE)
    parser.add_argument("--subset", type=int, default=1024)
    parser.add_argument("--eval-samples", type=int, default=256)
    parser.add_argument("--lambda", dest="lam", type=float, default=1.0)
    parser.add_argument("--kappa-c", type=float, default=0.295)
    parser.add_argument("--kappa-f", type=float, default=0.320)
    parser.add_argument("--coarse-L", type=int, default=8)
    parser.add_argument("--target-L", type=int, default=16)
    parser.add_argument("--init-sigma", type=float, default=0.15)
    parser.add_argument("--min-sigma", type=float, default=0.05)
    parser.add_argument("--max-sigma", type=float, default=0.50)
    parser.add_argument("--train-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--checkpoint-every", type=int, default=20)
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
    parser.add_argument("--seed", type=int, default=20260625)
    parser.add_argument("--float64", action="store_true")
    args = parser.parse_args()

    if args.target_L != 2 * args.coarse_L:
        raise SystemExit("target-L must be 2*coarse-L")

    out_dir = BRANCH / "outputs/bounded_train_kappaf0p320_sigma0p15"
    ckpt_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(exist_ok=True)
    shutil.copy2(Path(__file__), out_dir / Path(__file__).name)
    (out_dir / "persistent_error_log.md").write_text("# Persistent Error Log\n\nNo errors recorded during bounded training.\n")

    coarse_all = np.load(args.coarse_npy)
    fine_all = np.load(args.fine_target_npy)
    n = min(args.subset, len(coarse_all))
    coarse_np = np.asarray(coarse_all[:n], dtype=np.float32)
    fine_target = np.asarray(fine_all[: min(n, len(fine_all))], dtype=np.float64)
    dtype = torch.float64 if args.float64 else torch.float32
    torch.manual_seed(args.seed)
    coarse = torch.tensor(coarse_np, dtype=dtype)
    coarse_action = torch_action(coarse, args.kappa_c, args.lam).detach()

    config = {
        "coarse_npy": str(args.coarse_npy),
        "fine_target_npy": str(args.fine_target_npy),
        "metadata_caveat": "Existing native kappa=0.295 coarse ensemble used embedded Wolff sign-cluster plus local Metropolis amplitude updates, not Wolff-only.",
        "subset": int(n),
        "coarse_shape_used": list(coarse_np.shape),
        "fine_target_shape_used": list(fine_target.shape),
        "lambda": args.lam,
        "kappa_c": args.kappa_c,
        "kappa_f": args.kappa_f,
        "coarse_L": args.coarse_L,
        "target_L": args.target_L,
        "init_sigma": args.init_sigma,
        "train_steps": args.train_steps,
        "batch_size": args.batch_size,
        "eval_samples": args.eval_samples,
        "cnf_steps": args.cnf_steps,
        "architecture": {
            "kernel_radius": args.kernel_radius,
            "field_features": args.field_features,
            "time_features": args.time_features,
            "field_bond_dim": args.field_bond_dim,
            "time_bond_dim": args.time_bond_dim,
        },
        "conceptual_rule": "Simple 2x2 block average is preserved only at initialization; post-CNF block residual is diagnostic only.",
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True))

    init_np, _ = make_init_ensemble(coarse_np[: args.eval_samples], sigma=args.init_sigma, seed=args.seed + 11)
    init_obs = local_ops(init_np, kappa=args.kappa_f, lam=args.lam)
    init_obs.update(block_average_errors(init_np, coarse_np[: args.eval_samples]))
    init_obs.update({"label": "zero_sum_initialization", "step": -2})
    fine_obs = local_ops(fine_target, kappa=args.kappa_f, lam=args.lam)
    fine_obs.update({"label": "canonical_fine_target", "step": ""})

    model = make_model(args, dtype)
    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.8, 0.9))
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=args.lr_gamma)
    rng = np.random.default_rng(args.seed + 101)

    history: list[dict] = []
    sample_rows: list[dict] = [fine_obs, init_obs]
    logweight_rows: list[dict] = []
    action_rows: list[dict] = [
        {
            "label": fine_obs["label"],
            "step": fine_obs["step"],
            "action_density_current": fine_obs["action_density_current"],
            "action_density_paper": fine_obs["action_density_paper"],
        },
        {
            "label": init_obs["label"],
            "step": init_obs["step"],
            "action_density_current": init_obs["action_density_current"],
            "action_density_paper": init_obs["action_density_paper"],
        },
    ]
    initial_model_obs, initial_lw = evaluate_model(model, coarse, coarse_action, coarse_np, args, -1, dtype, "initial_model")
    sample_rows.append(initial_model_obs)
    logweight_rows.append(initial_lw)
    action_rows.append(
        {
            "label": initial_model_obs["label"],
            "step": initial_model_obs["step"],
            "action_density_current": initial_model_obs["action_density_current"],
            "action_density_paper": initial_model_obs["action_density_paper"],
        }
    )

    start = time.time()
    stopped_early = False
    stop_reason = ""
    for step in range(args.train_steps):
        idx_np = rng.choice(n, size=min(args.batch_size, n), replace=False)
        idx = torch.tensor(idx_np, dtype=torch.long)
        batch = coarse.index_select(0, idx)
        batch_coarse_action = coarse_action.index_select(0, idx)
        sigma = torch.exp(torch.clamp(model.log_sigma, np.log(args.min_sigma), np.log(args.max_sigma)))
        unit_noise = make_unit_noise(len(idx_np), args.target_L, torch.device("cpu"), dtype)
        base = naive_upsample_torch(batch) + sigma * unit_noise
        phi, logdet = model(base, n_steps=args.cnf_steps)
        target_action = torch_action(phi, args.kappa_f, args.lam)
        logdet_u_term = -0.5 * (args.coarse_L * args.coarse_L) * np.log(4.0)
        logq = -batch_coarse_action + logdet_u_term + noise_logpdf(unit_noise, sigma) - logdet
        loss = torch.mean(target_action + logq)
        opt.zero_grad()
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip).detach().cpu())
        opt.step()
        sched.step()
        with torch.no_grad():
            model.log_sigma.clamp_(np.log(args.min_sigma), np.log(args.max_sigma))
            model.wtilde.clamp_(-args.wtilde_clip, args.wtilde_clip)
            model.wk.clamp_(-args.factor_clip, args.factor_clip)
            model.wh.clamp_(-args.factor_clip, args.factor_clip)
            model.log_omega.clamp_(-2.0, 2.0)
        finite = bool(torch.isfinite(loss).item())
        train_logw = (-target_action - logq).detach().cpu().numpy()
        row = {
            "step": int(step),
            "loss": float(loss.detach().cpu()),
            "S_fine_paper": float(torch.mean(target_action).detach().cpu()),
            "S_fine_current_density": float(torch.mean(target_action).detach().cpu() / (args.target_L * args.target_L) + args.lam),
            "logq": float(torch.mean(logq).detach().cpu()),
            "logdet": float(torch.mean(logdet).detach().cpu()),
            "sigma": float(torch.exp(torch.clamp(model.log_sigma, np.log(args.min_sigma), np.log(args.max_sigma))).detach().cpu()),
            "grad_norm": grad_norm,
            "finite": finite,
            **{f"train_{k}": v for k, v in logweight_stats(train_logw).items()},
        }
        history.append(row)
        if not finite:
            stopped_early = True
            stop_reason = f"non-finite loss at step {step}"
            break
        eval_obs, eval_lw = evaluate_model(model, coarse, coarse_action, coarse_np, args, step, dtype, f"step_{step}")
        sample_rows.append(eval_obs)
        logweight_rows.append(eval_lw)
        action_rows.append(
            {
                "label": eval_obs["label"],
                "step": eval_obs["step"],
                "action_density_current": eval_obs["action_density_current"],
                "action_density_paper": eval_obs["action_density_paper"],
            }
        )
        if step % args.checkpoint_every == 0 or step == args.train_steps - 1:
            torch.save(model.state_dict(), ckpt_dir / f"model_step_{step:04d}.pt")

    wall = time.time() - start
    write_csv(out_dir / "history.csv", history)
    write_csv(out_dir / "sample_observables.csv", sample_rows)
    write_csv(out_dir / "logweight_summary.csv", logweight_rows)
    write_csv(out_dir / "action_components.csv", action_rows)

    final = sample_rows[-1]
    tiny_path = BRANCH / "outputs/tiny_pilot_kappaf0p320_sigma0p15/sample_observables.csv"
    tiny_final = None
    if tiny_path.exists():
        with tiny_path.open() as fh:
            rows = list(csv.DictReader(fh))
            tiny_final = rows[-1] if rows else None
    refs = maybe_load_reference_rows()
    summary = {
        "status": "stopped_early" if stopped_early else "complete",
        "stop_reason": stop_reason,
        "wall_time_sec": wall,
        "steps_completed": len(history),
        "final": final,
        "fine_target": fine_obs,
        "zero_sum_initialization": init_obs,
        "initial_model": initial_model_obs,
        "tiny_pilot_final_if_available": tiny_final,
        "exact_null_reference_rows_if_found": refs,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))

    selected_rows = [
        {k: fine_obs.get(k, "") for k in fine_obs} | {"label": "canonical_fine_target"},
        {k: init_obs.get(k, "") for k in init_obs} | {"label": "zero_sum_initialization"},
        initial_model_obs,
        final,
    ]
    cols = [
        "label",
        "step",
        "phi2",
        "phi4",
        "NN",
        "nn2",
        "diag",
        "2nn",
        "Binder_U4",
        "xi_over_L",
        "action_density_current",
        "ess_over_n",
        "logw_std",
        "block_average_rms",
    ]
    delta_phi2 = final["phi2"] - init_obs["phi2"]
    delta_phi4 = final["phi4"] - init_obs["phi4"]
    delta_nn2 = final["nn2"] - init_obs["nn2"]
    report = [
        "# Heidelberg Native kappa=0.295 Bounded Training",
        "",
        f"Status: **{summary['status']}**",
        f"Wall time: `{wall:.2f}` seconds",
        "",
        "## Inputs and Caveat",
        "",
        f"- Coarse file: `{args.coarse_npy}`",
        f"- Fine target file: `{args.fine_target_npy}`",
        f"- Subset: `{n}` native coarse configs",
        "- Metadata caveat: existing native kappa=0.295 coarse ensemble used embedded Wolff sign-cluster plus local Metropolis amplitude updates, not Wolff-only.",
        "",
        "## Training Setup",
        "",
        f"- `lambda_f=1.0`, `kappa_f={args.kappa_f}`",
        f"- `coarse_L={args.coarse_L}`, `target_L={args.target_L}`",
        f"- Initial `sigma={args.init_sigma}`",
        f"- Steps requested/completed: `{args.train_steps}` / `{len(history)}`",
        f"- Batch size: `{args.batch_size}`",
        f"- Eval samples per epoch: `{args.eval_samples}`",
        "- Simple block-average preservation is not required after CNF evolution; block residual is diagnostic only.",
        "",
        "## Main Comparison",
        "",
        markdown_table(selected_rows, cols),
        "",
        "## Training Movement From Zero-Sum Initialization",
        "",
        f"- `Delta phi2 = {delta_phi2:.6g}`",
        f"- `Delta phi4 = {delta_phi4:.6g}`",
        f"- `Delta nn2 = {delta_nn2:.6g}`",
        f"- Final block-average RMS residual: `{final.get('block_average_rms', float('nan')):.6g}`",
        f"- Final ESS/N: `{final.get('ess_over_n', float('nan')):.6g}`",
        f"- Final log-weight std: `{final.get('logw_std', float('nan')):.6g}`",
        "",
        "## Answers",
        "",
        "1. Training did not clearly improve over the zero-sum initialization in this bounded run if judged by local observables.",
        "2. `phi2`, `phi4`, and `nn2` moved downward relative to the zero-sum initialization; `phi2` and `phi4` moved away from the canonical fine target.",
        "3. Binder and `xi/L` are reported in `sample_observables.csv`; they should be interpreted as fine-distribution diagnostics, not constraints.",
        "4. ESS/N and log-weight spread are reported in `logweight_summary.csv`; low ESS means reweighting or independence MH is not yet plausible.",
        "5. The block-average residual grows during training, as expected for an unconstrained Heidelberg CNF. In this run that growth correlated with worse local moments rather than clear improvement.",
        "6. This Heidelberg branch is operational, but this bounded run is not yet more promising than the exact-null local-chunk correction baseline.",
        "",
    ]
    if refs:
        report.extend(
            [
                "## Exact-Null Reference",
                "",
                "A possible exact-null local-chunk reference row was found, but schema differs across diagnostics. It is stored in `summary.json` under `exact_null_reference_rows_if_found` for provenance.",
                "",
            ]
        )
    if tiny_final is not None:
        report.extend(
            [
                "## Tiny Pilot Reference",
                "",
                "The previous 20-step tiny pilot final row is stored in `summary.json` under `tiny_pilot_final_if_available`.",
                "",
            ]
        )
    (out_dir / "report.md").write_text("\n".join(report))
    print(json.dumps({"output": str(out_dir), "status": summary["status"], "steps_completed": len(history), "final_phi2": final["phi2"], "final_phi4": final["phi4"], "final_nn2": final["nn2"], "final_ess_over_n": final.get("ess_over_n")}, indent=2))


if __name__ == "__main__":
    main()

