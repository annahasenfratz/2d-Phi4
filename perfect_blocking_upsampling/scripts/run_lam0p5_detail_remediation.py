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
    compute_state_components,
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
from perfect_blocking_upsampling.actions import action_total  # noqa: E402
from perfect_blocking_upsampling.kernels import inverse_kernel  # noqa: E402
from perfect_blocking_upsampling.observables import observables as ensemble_observables  # noqa: E402
from train_faithful_transported_detail import build_detail_model, detail_arrays, log_base_torch  # noqa: E402
from ML_sampling_clean.experiments.decimated_conditional_fillin.run_staged_decimated_conditional_fillin import (  # noqa: E402
    fit_generic_local_gaussian,
    log_jacobian,
    to_model_space,
)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

LAM = 0.5
KAPPA = 0.3426
ETA = 0.25


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_stage_config(stage: str) -> dict[str, Any]:
    name = {"corner": "corner_body"}.get(stage, stage)
    return read_json(PROJECT_ROOT / f"perfect_blocking_upsampling/outputs/lam0p5_small3_8to16/full_training/configs/faithful_transported_detail/{name}.json")


def select_indices(arrays: dict[str, np.ndarray], max_train: int) -> dict[str, np.ndarray]:
    out = dict(arrays)
    if max_train <= 0:
        return out
    train = arrays["train_idx"][: min(max_train, len(arrays["train_idx"]))]
    val = arrays["val_idx"]
    out["train_idx"] = train
    out["val_idx"] = val
    return out


def load_lg(path: Path) -> dict[str, Any]:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def save_lg(path: Path, lg: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, coeffs=lg["coeffs"], sigma=lg["sigma"], ridge=np.asarray(lg.get("ridge", 0.0)))


def train_stage_variant(
    *,
    stage: str,
    variant: str,
    cfg: dict[str, Any],
    arrays: dict[str, np.ndarray],
    baseline_ckpt: Path | None,
    baseline_lg: Path | None,
    out: Path,
    epochs: int,
    lr: float | None = None,
    seed_offset: int = 0,
) -> dict[str, Any]:
    import torch

    stage_dir = out / variant
    ckpt_dir = stage_dir / "checkpoints"
    log_dir = stage_dir / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    tr = cfg["training"]
    batch_size = int(tr["batch_size"])
    torch.manual_seed(int(tr["seed"]) + seed_offset)
    cond, target = detail_arrays("corner_body" if stage == "corner" else stage, arrays)
    train_idx = arrays["train_idx"].astype(np.int64)
    val_idx = arrays["val_idx"].astype(np.int64)
    cond_train, target_train = cond[train_idx], target[train_idx]
    cond_val, target_val = cond[val_idx], target[val_idx]
    if baseline_lg is not None and baseline_lg.exists():
        lg = load_lg(baseline_lg)
    else:
        lg = fit_generic_local_gaussian(cond_train, target_train, float(tr.get("local_gaussian_sigma_floor", 1.0e-4)))
    save_lg(ckpt_dir / stage / "local_gaussian_coefficients.npz", lg)
    target_train_u = to_model_space(target_train, cond_train, lg)
    target_val_u = to_model_space(target_val, cond_val, lg)
    train_jac = torch.tensor(log_jacobian(cond_train, lg), dtype=torch.float32)
    val_jac = torch.tensor(log_jacobian(cond_val, lg), dtype=torch.float32)
    train_c = torch.tensor(cond_train.reshape(cond_train.shape[0], -1), dtype=torch.float32)
    train_d = torch.tensor(target_train_u.reshape(target_train_u.shape[0], -1), dtype=torch.float32)
    val_c = torch.tensor(cond_val.reshape(cond_val.shape[0], -1), dtype=torch.float32)
    val_d = torch.tensor(target_val_u.reshape(target_val_u.shape[0], -1), dtype=torch.float32)
    model = build_detail_model("corner_body" if stage == "corner" else stage, cfg)
    if baseline_ckpt is not None and baseline_ckpt.exists():
        state = torch.load(baseline_ckpt, map_location="cpu")
        model.load_state_dict(state["model_state"], strict=True)
    opt = torch.optim.Adam(model.parameters(), lr=float(lr or tr["learning_rate"]))
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] = {"val_loss": float("inf"), "epoch": 0, "state": None}
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(train_c.shape[0])
        losses = []
        model.train()
        for start in range(0, train_c.shape[0], batch_size):
            b = perm[start : start + batch_size]
            z, inv_logdet = model.inverse(train_d[b], train_c[b])
            logp = log_base_torch(z) + inv_logdet - train_jac[b]
            loss = -logp.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            z_val, inv_logdet_val = model.inverse(val_d, val_c)
            val_loss = float((-(log_base_torch(z_val) + inv_logdet_val - val_jac).mean()).detach())
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": val_loss}
        rows.append(row)
        if val_loss < best["val_loss"]:
            best = {"val_loss": val_loss, "epoch": epoch, "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
        torch.save(
            {
                "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "config": cfg["architecture"] | {"lambda_": LAM, "kappa": KAPPA, "eta": ETA, "lattice_size": int(cfg["coarse_L"]), "stage": stage},
                "stage": stage,
                "epoch": epoch,
                "val_loss": val_loss,
                "selection": "remediation_epoch",
            },
            ckpt_dir / f"{stage}_epoch{epoch:04d}.pt",
        )
        print(f"{variant} epoch {epoch}/{epochs}: val_loss={val_loss:.6g}", flush=True)
    assert best["state"] is not None
    final_ckpt = ckpt_dir / f"{stage}.pt"
    torch.save(
        {
            "model_state": best["state"],
            "config": cfg["architecture"] | {"lambda_": LAM, "kappa": KAPPA, "eta": ETA, "lattice_size": int(cfg["coarse_L"]), "stage": stage},
            "stage": stage,
            "epoch": best["epoch"],
            "val_loss": best["val_loss"],
            "selection": "remediation_best_val_nll",
        },
        final_ckpt,
    )
    write_csv(log_dir / f"{stage}_train_log.csv", rows)
    return {
        "variant": variant,
        "stage": stage,
        "checkpoint": str(final_ckpt),
        "local_gaussian": str(ckpt_dir / stage / "local_gaussian_coefficients.npz"),
        "best_epoch": best["epoch"],
        "best_val_loss": best["val_loss"],
        "last_val_loss": rows[-1]["val_loss"],
        "epochs": epochs,
    }


def build_candidate_ctx(base_ctx: dict[str, Any], replacements: dict[str, tuple[Path, Path]]) -> dict[str, Any]:
    import torch

    ctx = dict(base_ctx)
    stages = dict(base_ctx["stages"])
    for stage, (ckpt_path, lg_path) in replacements.items():
        model0, _lg0, _state0, _ckpt0 = stages[stage]
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model = type(model0)(*()) if False else None
        # Rebuild by using checkpoint config through the faithful config.
        cfg = load_stage_config(stage)
        model = build_detail_model("corner_body" if stage == "corner" else stage, cfg)
        model.load_state_dict(ckpt["model_state"], strict=True)
        model.eval()
        stages[stage] = (model, load_lg(lg_path), ckpt["model_state"], ckpt)
    ctx["stages"] = stages
    return ctx


def evaluate_candidate(name: str, ctx: dict[str, Any], arrays: dict[str, np.ndarray], out: Path, n_eval: int = 512) -> dict[str, Any]:
    vi = arrays["val_idx"][: min(n_eval, len(arrays["val_idx"]))]
    c = arrays["c00"][vi].astype(np.float32)
    e_t = arrays["edge_x"][vi, None].astype(np.float32)
    p_t = arrays["edge_y"][vi, None].astype(np.float32)
    co_t = arrays["corner"][vi, None].astype(np.float32)
    fine = arrays["fine16"][vi].astype(np.float32)
    cprime, _ = __import__("run_shape_parametric_sampler_validation").apply_refine_loaded(ctx["refine_model"], c)
    metrics = []
    z0 = np.zeros_like(e_t)
    edge_model, edge_lg, *_ = ctx["stages"]["edge"]
    e_m, edge_lq, *_ = stage_forward_z(edge_model, z0, cprime[:, None], edge_lg)
    pair_model, pair_lg, *_ = ctx["stages"]["pair"]
    p_m, pair_lq, *_ = stage_forward_z(pair_model, z0, np.concatenate([cprime[:, None], e_m], axis=1), pair_lg)
    corner_model, corner_lg, *_ = ctx["stages"]["corner"]
    co_m, corner_lq, *_ = stage_forward_z(corner_model, z0, np.concatenate([cprime[:, None], e_m, p_m], axis=1), corner_lg)
    for stage, pred, target, lq in [("edge", e_m, e_t, edge_lq), ("pair", p_m, p_t, pair_lq), ("corner", co_m, co_t, corner_lq)]:
        metrics.append({"candidate": name, "stage": stage, "rmse": rmse(pred, target), "correlation": corr(pred, target), "generated_z0_logq_std": qstats(lq)["std"]})
    s_ref = action_total(fine, ctx["fine_action"])
    assembly_rows = []
    assemblies = [
        ("target_all", c, e_t, p_t, co_t),
        ("model_coarse_only", cprime, e_t, p_t, co_t),
        ("model_coarse_edge", cprime, e_m, p_t, co_t),
        ("model_coarse_edge_pair", cprime, e_m, p_m, co_t),
        ("model_all_z0", cprime, e_m, p_m, co_m),
    ]
    for label, cc, ee, pp, coco in assemblies:
        phi, _ = inverse_kernel(reconstruct(cc, ee, pp, coco), ctx["kernel"])
        sf = action_total(phi, ctx["fine_action"])
        obs = ensemble_observables(phi, ctx["fine_action"])
        assembly_rows.append({
            "candidate": name,
            "assembly": label,
            "phi_rmse_vs_fine16": rmse(phi, fine),
            "phi_corr_vs_fine16": corr(phi, fine),
            "delta_action_std_vs_fine16": qstats(sf - s_ref)["std"],
            "delta_action_mean_vs_fine16": qstats(sf - s_ref)["mean"],
            "action_density_shift_vs_fine16": obs["action_density"] - ensemble_observables(fine, ctx["fine_action"])["action_density"],
            "phi2": obs["phi2"],
            "phi4": obs["phi4"],
            "NN": obs["NN"],
        })
    write_csv(out / f"{name}_stage_metrics.csv", metrics)
    write_csv(out / f"{name}_cumulative.csv", assembly_rows)
    return {"candidate": name, "stage_metrics": metrics, "assembly": assembly_rows}


def plot_losses(paths: list[Path], out: Path) -> None:
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for path in paths:
        data = np.genfromtxt(path, delimiter=",", names=True)
        ax.plot(data["epoch"], data["val_loss"], label=path.parent.parent.name)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation NLL")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=PROJECT_ROOT / "perfect_blocking_upsampling/outputs/lam0p5_small3_8to16/full_training/run_20260630_210838")
    ap.add_argument("--max-train-configs", type=int, default=2048)
    ap.add_argument("--eval-configs", type=int, default=512)
    ap.add_argument("--edge-epochs", type=int, default=12)
    ap.add_argument("--pair-epochs", type=int, default=24)
    ap.add_argument("--corner-epochs", type=int, default=24)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    run_dir = args.run_dir
    out = run_dir / "remediation"
    if out.exists() and args.overwrite:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    cfg, paths, coarse, fine_ref, _cm, _fm, base_ctx = load_context(run_dir)
    arrays0 = load_paired()
    arrays = select_indices(arrays0, args.max_train_configs)
    shutil.copy2(run_dir / "training_summary.json", out / "baseline_training_summary.json")
    if (run_dir / "diagnostics/LAM0P5_FAILURE_DIAGNOSTIC_REPORT.md").exists():
        shutil.copy2(run_dir / "diagnostics/LAM0P5_FAILURE_DIAGNOSTIC_REPORT.md", out / "baseline_LAM0P5_FAILURE_DIAGNOSTIC_REPORT.md")
    baseline = evaluate_candidate("baseline_failed_bundle", base_ctx, arrays0, out, args.eval_configs)
    smoke = json.loads((run_dir / "validation_smoke/native_L8_pcn1_2x100/summary.json").read_text())["result"]
    train_summary = json.loads((run_dir / "training_summary.json").read_text())
    baseline_rows = []
    for st in train_summary["stage_results"]:
        baseline_rows.append({"kind": "checkpoint", "stage": st["stage"], "path": st["checkpoint"], "selected_epoch": st["best_epoch"], "val_loss": st["best_val_loss"]})
    baseline_rows.extend([
        {"kind": "smoke", "stage": "coarse", "metric": "acceptance", "value": smoke["coarse_acceptance"]},
        {"kind": "smoke", "stage": "coarse", "metric": "delta_logw_std", "value": smoke["coarse_std_delta_logw"]},
        {"kind": "smoke", "stage": "latent", "metric": "acceptance", "value": smoke["latent_acceptance"]},
        {"kind": "smoke", "stage": "latent", "metric": "delta_logw_std", "value": smoke["latent_std_delta_logw"]},
    ])
    write_csv(out / "baseline_failure_metrics.csv", baseline_rows)
    (out / "BASELINE_FAILURE_SUMMARY.md").write_text(
        "# Baseline failure summary\n\n"
        f"- failed run: `{run_dir}`\n"
        f"- coarse acceptance: `{smoke['coarse_acceptance']}`\n"
        f"- coarse delta logw std: `{smoke['coarse_std_delta_logw']}`\n"
        f"- latent acceptance: `{smoke['latent_acceptance']}`\n"
        f"- latent delta logw std: `{smoke['latent_std_delta_logw']}`\n"
        "- baseline diagnostic tables were copied/recomputed into this remediation directory.\n"
    )

    edge_diag_dir = out / "edge_action_diagnostics"
    edge_diag_dir.mkdir(exist_ok=True)
    edge_base_rows = [r for r in baseline["stage_metrics"] if r["stage"] == "edge"]
    edge_cum_rows = [r for r in baseline["assembly"] if r["assembly"] in {"model_coarse_only", "model_coarse_edge"}]
    write_csv(edge_diag_dir / "edge_action_metrics.csv", edge_cum_rows)
    write_csv(edge_diag_dir / "edge_residual_metrics.csv", edge_base_rows)
    write_csv(edge_diag_dir / "edge_logq_metrics.csv", edge_base_rows)
    (edge_diag_dir / "EDGE_ACTION_DIAGNOSTIC_REPORT.md").write_text(
        "# Edge action diagnostic report\n\n"
        f"- baseline edge z0 RMSE: `{edge_base_rows[0]['rmse']:.6g}`\n"
        f"- baseline edge z0 correlation: `{edge_base_rows[0]['correlation']:.6g}`\n"
        f"- cumulative delta-action std after model coarse only: `{edge_cum_rows[0]['delta_action_std_vs_fine16']:.6g}`\n"
        f"- cumulative delta-action std after model edge: `{edge_cum_rows[1]['delta_action_std_vs_fine16']:.6g}`\n"
        "\nThe first action-sensitive degradation appears at edge, so edge variants are evaluated before sampler validation.\n"
    )

    ckpt_root = run_dir / "checkpoints"
    variants: list[dict[str, Any]] = []
    edge_cfg = load_stage_config("edge")
    variants.append(train_stage_variant(stage="edge", variant="edge_longer_same_objective_continue", cfg=edge_cfg, arrays=arrays, baseline_ckpt=ckpt_root / "edge.pt", baseline_lg=ckpt_root / "edge/local_gaussian_coefficients.npz", out=out / "edge_retraining_variants", epochs=args.edge_epochs, lr=1.0e-4))
    variants.append(train_stage_variant(stage="edge", variant="edge_restart_same_objective", cfg=edge_cfg, arrays=arrays, baseline_ckpt=None, baseline_lg=None, out=out / "edge_retraining_variants", epochs=args.edge_epochs, lr=3.0e-4, seed_offset=1000))
    candidate_results = [baseline]
    for v in variants:
        cctx = build_candidate_ctx(base_ctx, {"edge": (Path(v["checkpoint"]), Path(v["local_gaussian"]))})
        candidate_results.append(evaluate_candidate(v["variant"], cctx, arrays0, out / "edge_retraining_variants", args.eval_configs))
    # Pick best edge by model_coarse_edge delta-action std.
    def edge_score(res: dict[str, Any]) -> float:
        for r in res["assembly"]:
            if r["assembly"] == "model_coarse_edge":
                return float(r["delta_action_std_vs_fine16"])
        return float("inf")
    best_edge = min(candidate_results, key=edge_score)
    best_edge_name = best_edge["candidate"]
    best_edge_repl: dict[str, tuple[Path, Path]] = {}
    for v in variants:
        if v["variant"] == best_edge_name:
            best_edge_repl["edge"] = (Path(v["checkpoint"]), Path(v["local_gaussian"]))
    edge_ctx = build_candidate_ctx(base_ctx, best_edge_repl) if best_edge_repl else base_ctx

    pc_dir = out / "pair_corner_longer_training"
    pair_cfg = load_stage_config("pair")
    corner_cfg = load_stage_config("corner")
    pair_v = train_stage_variant(stage="pair", variant="pair_longer_continue", cfg=pair_cfg, arrays=arrays, baseline_ckpt=ckpt_root / "pair.pt", baseline_lg=ckpt_root / "pair/local_gaussian_coefficients.npz", out=pc_dir, epochs=args.pair_epochs, lr=1.0e-4)
    pair_ctx = build_candidate_ctx(edge_ctx, {"pair": (Path(pair_v["checkpoint"]), Path(pair_v["local_gaussian"]))})
    pair_res = evaluate_candidate("best_edge_plus_pair_longer", pair_ctx, arrays0, pc_dir, args.eval_configs)
    corner_v = train_stage_variant(stage="corner", variant="corner_longer_continue", cfg=corner_cfg, arrays=arrays, baseline_ckpt=ckpt_root / "corner.pt", baseline_lg=ckpt_root / "corner/local_gaussian_coefficients.npz", out=pc_dir, epochs=args.corner_epochs, lr=1.0e-4)
    full_ctx = build_candidate_ctx(pair_ctx, {"corner": (Path(corner_v["checkpoint"]), Path(corner_v["local_gaussian"]))})
    full_res = evaluate_candidate("best_edge_plus_pair_corner_longer", full_ctx, arrays0, pc_dir, args.eval_configs)
    candidate_results.extend([pair_res, full_res])

    rows = []
    for res in candidate_results:
        for row in res["assembly"]:
            rows.append(row)
    write_csv(out / "candidate_bundle_comparison.csv", rows)
    write_csv(out / "remediation_variant_training_summary.csv", variants + [pair_v, corner_v])
    plot_losses(list((out / "edge_retraining_variants").glob("*/logs/*_train_log.csv")) + list(pc_dir.glob("*/logs/*_train_log.csv")), out / "remediation_loss_curves.pdf")
    lines = ["# Candidate bundle comparison", "", f"- best edge by cumulative edge action std: `{best_edge_name}`", ""]
    for res in candidate_results:
        vals = {r["assembly"]: r["delta_action_std_vs_fine16"] for r in res["assembly"]}
        lines.append(f"- {res['candidate']}: edge `{vals.get('model_coarse_edge'):.6g}`, pair `{vals.get('model_coarse_edge_pair'):.6g}`, full `{vals.get('model_all_z0'):.6g}`")
    (out / "CANDIDATE_BUNDLE_COMPARISON_REPORT.md").write_text("\n".join(lines) + "\n")
    (out / "LAM0P5_DETAIL_STAGE_REMEDIATION_REPORT.md").write_text(
        "# Lambda=0.5 detail-stage remediation report\n\n"
        f"- remediation output: `{out}`\n"
        f"- bounded training subset: `{args.max_train_configs}` train configs; evaluation uses `{args.eval_configs}` held-out configs.\n"
        f"- best edge candidate by action metric: `{best_edge_name}`\n\n"
        "## Findings\n\n"
        + "\n".join(lines[3:])
        + "\n\n## Recommendation\n\n"
        "Do not launch long validation yet. Promote a candidate to tiny sampler smoke only if the cumulative action std improves materially versus baseline "
        "(`8.424` after edge, `15.004` after pair). If these short variants do not improve the action metric, the next intervention should change the edge/detail objective or selection criterion rather than simply extending the same NLL training.\n"
    )
    write_json(out / "remediation_summary.json", {"status": "completed", "best_edge": best_edge_name, "variants": variants + [pair_v, corner_v]})
    print(json.dumps({"status": "completed", "out": str(out), "best_edge": best_edge_name}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
