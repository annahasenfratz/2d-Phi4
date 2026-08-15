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
    stage_inverse_target,
    write_csv,
    write_json,
)
from perfect_blocking_upsampling.actions import action_total  # noqa: E402
from perfect_blocking_upsampling.kernels import inverse_kernel  # noqa: E402
from run_lam0p5_detail_remediation import build_candidate_ctx, evaluate_candidate, load_lg, load_stage_config, train_stage_variant  # noqa: E402
from train_faithful_transported_detail import build_detail_model  # noqa: E402

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


def load_edge_model(ckpt_path: Path):
    import torch

    cfg = load_stage_config("edge")
    model = build_detail_model("edge", cfg)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    return model, ckpt


def edge_lg_for_checkpoint(path: Path) -> Path:
    candidates = [
        path.parent / "edge" / "local_gaussian_coefficients.npz",
        path.parent.parent / "edge" / "local_gaussian_coefficients.npz",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"could not locate edge local Gaussian for {path}")


def source_label(path: Path, run_dir: Path) -> str:
    s = str(path.relative_to(run_dir))
    if s.startswith("checkpoints/"):
        return "original_full_training"
    if "edge_longer_same_objective_continue" in s:
        return "remediation_edge_continue"
    if "edge_restart_same_objective" in s:
        return "remediation_edge_restart"
    if "edge_dense_checkpoint_training" in s:
        return "dense_edge_continue"
    return "other"


def enumerate_edge_checkpoints(run_dir: Path, include_dense: bool) -> list[Path]:
    roots = [
        run_dir / "checkpoints",
        run_dir / "remediation/edge_retraining_variants/edge_longer_same_objective_continue/checkpoints",
        run_dir / "remediation/edge_retraining_variants/edge_restart_same_objective/checkpoints",
    ]
    if include_dense:
        roots.append(run_dir / "remediation/action_aware_edge_selection/edge_dense_checkpoint_training/dense_continue/checkpoints")
    paths: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        paths.extend(sorted(root.glob("edge_epoch*.pt")))
        final = root / "edge.pt"
        if final.exists():
            paths.append(final)
    # Preserve order but remove duplicates.
    seen = set()
    out = []
    for p in paths:
        rp = p.resolve()
        if rp not in seen:
            out.append(p)
            seen.add(rp)
    return out


def evaluate_edge_checkpoint(path: Path, lg_path: Path, arrays: dict[str, np.ndarray], ctx: dict[str, Any], n_eval: int, run_dir: Path) -> dict[str, Any]:
    model, ckpt = load_edge_model(path)
    lg = load_lg(lg_path)
    vi = arrays["val_idx"][: min(n_eval, len(arrays["val_idx"]))]
    c = arrays["c00"][vi].astype(np.float32)
    edge_target = arrays["edge_x"][vi, None].astype(np.float32)
    pair_target = arrays["edge_y"][vi, None].astype(np.float32)
    corner_target = arrays["corner"][vi, None].astype(np.float32)
    cond = c[:, None]
    z0 = np.zeros_like(edge_target, dtype=np.float32)
    edge_pred, generated_logq, log_base, forward_logdet = stage_forward_z(model, z0, cond, lg)
    _z_target, target_logq, _target_log_base, target_inv_logdet = stage_inverse_target(model, edge_target, cond, lg)
    phi_target, _ = inverse_kernel(reconstruct(c, edge_target, pair_target, corner_target), ctx["kernel"])
    phi_edge, _ = inverse_kernel(reconstruct(c, edge_pred, pair_target, corner_target), ctx["kernel"])
    s_target = action_total(phi_target, ctx["fine_action"])
    s_edge = action_total(phi_edge, ctx["fine_action"])
    delta_s = s_edge - s_target
    action_density_shift = delta_s / (phi_target.shape[1] * phi_target.shape[2])
    rms = float(math.sqrt(np.mean(delta_s * delta_s)))
    return {
        "source": source_label(path, run_dir),
        "checkpoint": str(path),
        "local_gaussian": str(lg_path),
        "epoch": ckpt.get("epoch", ""),
        "checkpoint_val_loss": ckpt.get("val_loss", ""),
        "selection": ckpt.get("selection", ""),
        "val_nll_from_target_inverse": float(-np.mean(target_logq)),
        "edge_rmse_z0": rmse(edge_pred, edge_target),
        "edge_corr_z0": corr(edge_pred, edge_target),
        "residual_mean": float(np.mean(edge_pred - edge_target)),
        "residual_std": float(np.std(edge_pred - edge_target, ddof=1)),
        "deltaS_mean": float(np.mean(delta_s)),
        "deltaS_std": float(np.std(delta_s, ddof=1)),
        "deltaS_rms": rms,
        "action_density_shift_mean": float(np.mean(action_density_shift)),
        "action_density_shift_std": float(np.std(action_density_shift, ddof=1)),
        "outlier_frac_abs_deltaS_gt_10": float(np.mean(np.abs(delta_s) > 10.0)),
        "outlier_frac_abs_deltaS_gt_20": float(np.mean(np.abs(delta_s) > 20.0)),
        "target_logq_mean": qstats(target_logq)["mean"],
        "target_logq_std": qstats(target_logq)["std"],
        "generated_z0_logq_mean": qstats(generated_logq)["mean"],
        "generated_z0_logq_std": qstats(generated_logq)["std"],
        "z0_forward_logdet_mean": qstats(forward_logdet)["mean"],
        "z0_forward_logdet_std": qstats(forward_logdet)["std"],
    }


def select_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    finite = [r for r in rows if np.isfinite(float(r["deltaS_std"])) and np.isfinite(float(r["val_nll_from_target_inverse"]))]
    best_action = min(finite, key=lambda r: float(r["deltaS_std"]))
    best_nll = min(finite, key=lambda r: float(r["val_nll_from_target_inverse"]))
    action_threshold = float(best_action["deltaS_std"]) * 1.03
    near_action = [r for r in finite if float(r["deltaS_std"]) <= action_threshold]
    best_compromise = min(near_action, key=lambda r: float(r["val_nll_from_target_inverse"]))
    return {"best_by_action": best_action, "best_by_nll": best_nll, "best_compromise": best_compromise}


def plot_selection(rows: list[dict[str, Any]], out: Path) -> None:
    if plt is None or not rows:
        return
    x = np.asarray([float(r["val_nll_from_target_inverse"]) for r in rows])
    y = np.asarray([float(r["deltaS_std"]) for r in rows])
    labels = [r["source"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 5))
    for label in sorted(set(labels)):
        m = np.asarray([z == label for z in labels])
        ax.scatter(x[m], y[m], s=18, alpha=0.75, label=label)
    ax.set_xlabel("validation NLL from target inverse")
    ax.set_ylabel("std(delta S edge)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "nll_vs_edge_deltaS_std.pdf")
    plt.close(fig)


def maybe_dense_train(run_dir: Path, out: Path, arrays: dict[str, np.ndarray], epochs: int, max_train: int) -> dict[str, Any]:
    cfg = load_stage_config("edge")
    arrays2 = dict(arrays)
    arrays2["train_idx"] = arrays["train_idx"][: min(max_train, len(arrays["train_idx"]))]
    return train_stage_variant(
        stage="edge",
        variant="dense_continue",
        cfg=cfg,
        arrays=arrays2,
        baseline_ckpt=run_dir / "checkpoints/edge.pt",
        baseline_lg=run_dir / "checkpoints/edge/local_gaussian_coefficients.npz",
        out=out / "edge_dense_checkpoint_training",
        epochs=epochs,
        lr=5.0e-5,
    )


def write_metric_definition(out: Path) -> None:
    path = out / "ACTION_AWARE_EDGE_SELECTION_METRIC.md"
    path.write_text(
        "# Action-aware edge checkpoint selection metric\n\n"
        "For each edge checkpoint, the model edge output is generated from the same gathered edge architecture at `z=0` on a held-out paired validation batch. "
        "The coarse field and downstream pair/corner details are kept at their target transported-detail values. The target reconstruction is\n\n"
        "`phi_target = inverse_kernel(c_target, edge_target, pair_target, corner_target)`\n\n"
        "and the edge-swapped reconstruction is\n\n"
        "`phi_model_edge = inverse_kernel(c_target, edge_model_z0, pair_target, corner_target)`.\n\n"
        "The primary action-aware score is `std(deltaS_edge)`, where\n\n"
        "`deltaS_edge = S(phi_model_edge) - S(phi_target)`.\n\n"
        "The report also records mean and RMS `deltaS_edge`, action-density shift, outlier fractions, validation NLL from the target inverse, RMSE/correlation, and logq/logdet summaries. "
        "This is a checkpoint-selection diagnostic only; it does not change the architecture or training loss.\n"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=PROJECT_ROOT / "perfect_blocking_upsampling/outputs/lam0p5_small3_8to16/full_training/run_20260630_210838")
    ap.add_argument("--n-eval", type=int, default=512)
    ap.add_argument("--dense-epochs", type=int, default=24)
    ap.add_argument("--max-train-configs", type=int, default=2048)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    run_dir = args.run_dir
    out = run_dir / "remediation/action_aware_edge_selection"
    if out.exists() and args.overwrite:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    write_metric_definition(out)
    _cfg, _paths, _coarse, _fine_ref, _cm, _fm, ctx = load_context(run_dir)
    arrays = load_paired()

    existing_paths = enumerate_edge_checkpoints(run_dir, include_dense=False)
    existing_rows = [evaluate_edge_checkpoint(p, edge_lg_for_checkpoint(p), arrays, ctx, args.n_eval, run_dir) for p in existing_paths]
    write_csv(out / "existing_edge_checkpoint_action_metrics.csv", existing_rows)
    existing_sel = select_rows(existing_rows)
    baseline = [r for r in existing_rows if r["source"] == "original_full_training" and Path(r["checkpoint"]).name == "edge.pt"][0]
    material_existing = float(existing_sel["best_by_action"]["deltaS_std"]) < 0.85 * float(baseline["deltaS_std"])
    dense_result: dict[str, Any] | None = None
    if not material_existing:
        dense_result = maybe_dense_train(run_dir, out, arrays, args.dense_epochs, args.max_train_configs)
    all_paths = enumerate_edge_checkpoints(run_dir, include_dense=True)
    all_rows = [evaluate_edge_checkpoint(p, edge_lg_for_checkpoint(p), arrays, ctx, args.n_eval, run_dir) for p in all_paths]
    write_csv(out / "all_edge_checkpoint_action_metrics.csv", all_rows)
    selection = select_rows(all_rows)
    selected_rows = []
    for name, row in selection.items():
        selected_rows.append({"selection_label": name, **row})
    write_csv(out / "selected_edge_checkpoints.csv", selected_rows)
    plot_selection(all_rows, out)

    # Cumulative assembly check with baseline pair/corner and with available pair/corner continuation.
    cumulative_rows = []
    candidate_results = []
    replacement_sets: dict[str, dict[str, tuple[Path, Path]]] = {}
    for sel_name, row in selection.items():
        replacement_sets[f"{sel_name}_baseline_pair_corner"] = {
            "edge": (Path(row["checkpoint"]), Path(row["local_gaussian"])),
        }
        pair = run_dir / "remediation/pair_corner_longer_training/pair_longer_continue/checkpoints/pair.pt"
        pair_lg = run_dir / "remediation/pair_corner_longer_training/pair_longer_continue/checkpoints/pair/local_gaussian_coefficients.npz"
        corner = run_dir / "remediation/pair_corner_longer_training/corner_longer_continue/checkpoints/corner.pt"
        corner_lg = run_dir / "remediation/pair_corner_longer_training/corner_longer_continue/checkpoints/corner/local_gaussian_coefficients.npz"
        if pair.exists() and pair_lg.exists():
            replacement_sets[f"{sel_name}_pair_continue"] = {
                "edge": (Path(row["checkpoint"]), Path(row["local_gaussian"])),
                "pair": (pair, pair_lg),
            }
        if pair.exists() and pair_lg.exists() and corner.exists() and corner_lg.exists():
            replacement_sets[f"{sel_name}_pair_corner_continue"] = {
                "edge": (Path(row["checkpoint"]), Path(row["local_gaussian"])),
                "pair": (pair, pair_lg),
                "corner": (corner, corner_lg),
            }
    for name, repl in replacement_sets.items():
        cctx = build_candidate_ctx(ctx, repl)
        res = evaluate_candidate(name, cctx, arrays, out, args.n_eval)
        candidate_results.append(res)
        cumulative_rows.extend(res["assembly"])
    write_csv(out / "cumulative_selected_edge_metrics.csv", cumulative_rows)

    best_action = selection["best_by_action"]
    improved_materially = float(best_action["deltaS_std"]) < 0.85 * float(baseline["deltaS_std"])
    report_lines = [
        "# Edge checkpoint selection report",
        "",
        f"- existing checkpoints evaluated: `{len(existing_rows)}`",
        f"- total checkpoints evaluated after dense run: `{len(all_rows)}`",
        f"- baseline edge.pt deltaS std: `{float(baseline['deltaS_std']):.6g}`",
        f"- best-by-action deltaS std: `{float(best_action['deltaS_std']):.6g}`",
        f"- best-by-action checkpoint: `{best_action['checkpoint']}`",
        f"- best-by-NLL deltaS std: `{float(selection['best_by_nll']['deltaS_std']):.6g}`",
        f"- best-by-NLL checkpoint: `{selection['best_by_nll']['checkpoint']}`",
        f"- dense training run: `{dense_result is not None}`",
        f"- material improvement threshold met: `{improved_materially}`",
        "",
        "Action-aware selection used `std(deltaS_edge)` as the primary metric. Dense edge training used the same NLL objective and architecture, with checkpoint-dense epoch saves.",
    ]
    (out / "EDGE_CHECKPOINT_SELECTION_REPORT.md").write_text("\n".join(report_lines) + "\n")
    cum_lines = ["# Cumulative selected edge report", ""]
    for res in candidate_results:
        vals = {r["assembly"]: r["delta_action_std_vs_fine16"] for r in res["assembly"]}
        cum_lines.append(f"- {res['candidate']}: edge `{vals.get('model_coarse_edge'):.6g}`, pair `{vals.get('model_coarse_edge_pair'):.6g}`, full `{vals.get('model_all_z0'):.6g}`")
    (out / "CUMULATIVE_SELECTED_EDGE_REPORT.md").write_text("\n".join(cum_lines) + "\n")
    final = [
        "# Action-aware edge selection final report",
        "",
        "1. Does action-aware checkpoint selection find a better edge?",
        "",
        f"   {'Yes' if improved_materially else 'No'}. Baseline deltaS std was `{float(baseline['deltaS_std']):.6g}`; best action-selected checkpoint is `{float(best_action['deltaS_std']):.6g}`.",
        "",
        "2. Is NLL checkpoint selection misaligned with action compatibility?",
        "",
        f"   Best-by-NLL deltaS std is `{float(selection['best_by_nll']['deltaS_std']):.6g}`; best-by-action NLL is `{float(best_action['val_nll_from_target_inverse']):.6g}`. See `nll_vs_edge_deltaS_std.pdf` and CSV tables.",
        "",
        "3. Does selected edge improve cumulative pair/full diagnostics?",
        "",
        "   See `cumulative_selected_edge_metrics.csv`; no sampler smoke is run unless the edge metric materially improves.",
        "",
        "4. Is this enough to try a tiny sampler smoke?",
        "",
        f"   {'Yes, a tiny smoke can be prepared manually.' if improved_materially else 'No. The action metric did not improve enough to justify even a tiny sampler smoke.'}",
        "",
        "5. If not, should the next intervention be an explicit local action penalty in training?",
        "",
        "   Yes if checkpoint selection does not materially improve the edge action metric. The architecture and NLL objective alone are not selecting action-compatible edge outputs.",
    ]
    (out / "ACTION_AWARE_EDGE_SELECTION_FINAL_REPORT.md").write_text("\n".join(final) + "\n")
    write_json(out / "action_aware_edge_selection_summary.json", {
        "status": "completed",
        "baseline_deltaS_std": float(baseline["deltaS_std"]),
        "best_by_action": best_action,
        "best_by_nll": selection["best_by_nll"],
        "best_compromise": selection["best_compromise"],
        "dense_training": dense_result,
        "material_improvement": improved_materially,
        "sampler_smoke_launched": False,
    })
    print(json.dumps({"status": "completed", "out": str(out), "material_improvement": improved_materially, "best_deltaS_std": float(best_action["deltaS_std"])}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
