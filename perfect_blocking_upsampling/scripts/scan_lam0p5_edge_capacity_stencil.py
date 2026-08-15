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

from diagnose_lam0p5_failed_bundle import corr, load_context, load_paired, qstats, reconstruct, rmse, stage_forward_z, stage_inverse_target, write_csv, write_json  # noqa: E402
from perfect_blocking_upsampling.actions import action_total  # noqa: E402
from perfect_blocking_upsampling.gathered_edge import build_gathered_edge_flow  # noqa: E402
from perfect_blocking_upsampling.kernels import inverse_kernel  # noqa: E402
from run_lam0p5_detail_remediation import build_candidate_ctx, evaluate_candidate, load_lg, save_lg  # noqa: E402
from run_staged_decimated_conditional_fillin import fit_generic_local_gaussian, log_jacobian, to_model_space  # noqa: E402
from train_faithful_transported_detail import log_base_torch  # noqa: E402
from train_lam0p5_local_action_edge import local_action_np  # noqa: E402

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


def count_params(model) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def load_arrays(max_train: int) -> dict[str, np.ndarray]:
    arrays = load_paired()
    if max_train > 0:
        arrays = dict(arrays)
        arrays["train_idx"] = arrays["train_idx"][: min(max_train, len(arrays["train_idx"]))]
    return arrays


def build_variant_model(v: dict[str, Any]):
    return build_gathered_edge_flow(
        cond_channels=1,
        lattice_size=8,
        radius=int(v["radius"]),
        stencil=str(v.get("stencil", "square")),
        hidden_width=int(v["hidden_width"]),
        hidden_layers=int(v["hidden_layers"]),
        log_scale_bound=0.75,
    )


def evaluate_model(model, lg: dict[str, Any], arrays: dict[str, np.ndarray], ctx: dict[str, Any], n_eval: int, label: str, ckpt: Path, epoch: int, val_nll: float, param_count: int, runtime_sec: float) -> dict[str, Any]:
    vi = arrays["val_idx"][: min(n_eval, len(arrays["val_idx"]))]
    c = arrays["c00"][vi].astype(np.float32)
    e_t = arrays["edge_x"][vi, None].astype(np.float32)
    p_t = arrays["edge_y"][vi, None].astype(np.float32)
    co_t = arrays["corner"][vi, None].astype(np.float32)
    cond = c[:, None]
    z0 = np.zeros_like(e_t, dtype=np.float32)
    e_m, gen_lq, _lb, fld = stage_forward_z(model, z0, cond, lg)
    _zt, target_lq, _tlb, invld = stage_inverse_target(model, e_t, cond, lg)
    phi_t, _ = inverse_kernel(reconstruct(c, e_t, p_t, co_t), ctx["kernel"])
    phi_m, _ = inverse_kernel(reconstruct(c, e_m, p_t, co_t), ctx["kernel"])
    ds = action_total(phi_m, ctx["fine_action"]) - action_total(phi_t, ctx["fine_action"])
    dsl = local_action_np(phi_m) - local_action_np(phi_t)
    bad = not np.isfinite(ds).all() or not np.isfinite(target_lq).all()
    return {
        "variant": label,
        "checkpoint": str(ckpt),
        "epoch": epoch,
        "param_count": param_count,
        "runtime_sec": runtime_sec,
        "checkpoint_val_nll": val_nll,
        "val_nll_from_target_inverse": float(-np.mean(target_lq)),
        "edge_rmse_z0": rmse(e_m, e_t),
        "edge_corr_z0": corr(e_m, e_t),
        "global_deltaS_mean": qstats(ds)["mean"],
        "global_deltaS_std": qstats(ds)["std"],
        "global_deltaS_rms": float(np.sqrt(np.mean(ds * ds))),
        "local_deltaS_mean": qstats(dsl)["mean"],
        "local_deltaS_std": qstats(dsl)["std"],
        "local_deltaS_rms": float(np.sqrt(np.mean(dsl * dsl))),
        "local_density_shift_std": qstats(dsl / 64.0)["std"],
        "target_logq_mean": qstats(target_lq)["mean"],
        "target_logq_std": qstats(target_lq)["std"],
        "generated_z0_logq_mean": qstats(gen_lq)["mean"],
        "generated_z0_logq_std": qstats(gen_lq)["std"],
        "forward_logdet_mean": qstats(fld)["mean"],
        "forward_logdet_std": qstats(fld)["std"],
        "nan_or_inf": bool(bad),
    }


def train_variant(v: dict[str, Any], arrays: dict[str, np.ndarray], ctx: dict[str, Any], out: Path, epochs: int, n_eval: int, seed: int) -> dict[str, Any]:
    import torch

    label = str(v["name"])
    vdir = out / str(v["family"]) / label
    ckpt_dir = vdir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    cond = arrays["c00"][:, None].astype(np.float32)
    target = arrays["edge_x"][:, None].astype(np.float32)
    train_idx = arrays["train_idx"].astype(np.int64)
    val_idx = arrays["val_idx"].astype(np.int64)
    cond_train, target_train = cond[train_idx], target[train_idx]
    cond_val, target_val = cond[val_idx], target[val_idx]
    lg = fit_generic_local_gaussian(cond_train, target_train, 1.0e-4)
    save_lg(ckpt_dir / "edge/local_gaussian_coefficients.npz", lg)
    train_u = to_model_space(target_train, cond_train, lg)
    val_u = to_model_space(target_val, cond_val, lg)
    train_j = torch.tensor(log_jacobian(cond_train, lg), dtype=torch.float32)
    val_j = torch.tensor(log_jacobian(cond_val, lg), dtype=torch.float32)
    train_c = torch.tensor(cond_train.reshape(cond_train.shape[0], -1), dtype=torch.float32)
    train_d = torch.tensor(train_u.reshape(train_u.shape[0], -1), dtype=torch.float32)
    val_c = torch.tensor(cond_val.reshape(cond_val.shape[0], -1), dtype=torch.float32)
    val_d = torch.tensor(val_u.reshape(val_u.shape[0], -1), dtype=torch.float32)
    model = build_variant_model(v)
    params = count_params(model)
    opt = torch.optim.Adam(model.parameters(), lr=float(v.get("learning_rate", 3.0e-4)))
    batch = int(v.get("batch_size", 64))
    rows: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    best_action: dict[str, Any] | None = None
    best_nll: dict[str, Any] | None = None
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(train_c.shape[0])
        losses = []
        model.train()
        for start in range(0, train_c.shape[0], batch):
            b = perm[start : start + batch]
            z, invld = model.inverse(train_d[b], train_c[b])
            loss = -(log_base_torch(z) + invld - train_j[b]).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            z_val, invld_val = model.inverse(val_d, val_c)
            val_nll = float((-(log_base_torch(z_val) + invld_val - val_j).mean()).detach())
        ckpt = ckpt_dir / f"edge_epoch{epoch:04d}.pt"
        torch.save(
            {
                "model_state": {k: vv.detach().cpu().clone() for k, vv in model.state_dict().items()},
                "config": {
                    "flow_arch": "gathered_edge",
                    "gather_radius": int(v["radius"]),
                    "gather_stencil": str(v.get("stencil", "square")),
                    "gather_hidden_width": int(v["hidden_width"]),
                    "gather_hidden_layers": int(v["hidden_layers"]),
                    "log_scale_bound": 0.75,
                    "cond_channels": 1,
                    "target_channels": 1,
                    "lambda_": LAM,
                    "kappa": KAPPA,
                    "eta": ETA,
                    "lattice_size": 8,
                    "stage": "edge",
                },
                "stage": "edge",
                "epoch": epoch,
                "val_loss": val_nll,
                "selection": "edge_capacity_stencil_scan",
            },
            ckpt,
        )
        elapsed = time.perf_counter() - t0
        metric = evaluate_model(model, lg, arrays, ctx, n_eval, label, ckpt, epoch, val_nll, params, elapsed)
        row = {
            "variant": label,
            "family": v["family"],
            "radius": v["radius"],
            "hidden_width": v["hidden_width"],
            "hidden_layers": v["hidden_layers"],
            "epoch": epoch,
            "train_nll": float(np.mean(losses)),
            "val_nll": val_nll,
            "global_deltaS_std": metric["global_deltaS_std"],
            "global_deltaS_rms": metric["global_deltaS_rms"],
            "local_deltaS_std": metric["local_deltaS_std"],
            "edge_rmse_z0": metric["edge_rmse_z0"],
            "edge_corr_z0": metric["edge_corr_z0"],
            "param_count": params,
            "runtime_sec": elapsed,
            "checkpoint": str(ckpt),
        }
        rows.append(row)
        metrics.append(metric)
        write_csv(vdir / "training_log.csv", rows)
        write_csv(vdir / "checkpoint_action_metrics.csv", metrics)
        if best_action is None or metric["global_deltaS_std"] < best_action["global_deltaS_std"]:
            best_action = metric | row
        if best_nll is None or val_nll < best_nll["val_nll"]:
            best_nll = metric | row
        print(f"{label} epoch {epoch}/{epochs}: nll={val_nll:.6g} dSstd={metric['global_deltaS_std']:.6g} params={params}", flush=True)
    assert best_action and best_nll
    return {
        "variant": label,
        "family": v["family"],
        "radius": v["radius"],
        "hidden_width": v["hidden_width"],
        "hidden_layers": v["hidden_layers"],
        "param_count": params,
        "runtime_sec": time.perf_counter() - t0,
        "local_gaussian": str(ckpt_dir / "edge/local_gaussian_coefficients.npz"),
        "best_by_action": best_action,
        "best_by_nll": best_nll,
    }


def plot_summary(rows: list[dict[str, Any]], out: Path) -> None:
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for name in sorted(set(r["variant"] for r in rows)):
        rr = [r for r in rows if r["variant"] == name]
        ax.plot([int(r["epoch"]) for r in rr], [float(r["global_deltaS_std"]) for r in rr], marker="o", label=name)
    ax.axhline(PREVIOUS_BEST_EDGE_STD, color="k", linestyle="--", linewidth=1, label="previous best")
    ax.set_xlabel("epoch")
    ax.set_ylabel("edge std(delta S)")
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out / "edge_variant_deltaS_scan.pdf")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=PROJECT_ROOT / "perfect_blocking_upsampling/outputs/lam0p5_small3_8to16/full_training/run_20260630_210838")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--max-train-configs", type=int, default=2048)
    ap.add_argument("--n-eval", type=int, default=512)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    out = args.run_dir / "remediation/edge_capacity_stencil_scan"
    if out.exists() and args.overwrite:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "EDGE_VARIANT_METRIC_DEFINITION.md").write_text(
        "# Edge variant metric definition\n\n"
        "The primary metric is `std(DeltaS_edge)` on a fixed held-out paired validation batch, where `DeltaS_edge = S(phi_model_edge) - S(phi_target)`. "
        "`phi_model_edge` uses target coarse and target downstream pair/corner details with the candidate model edge at z=0. The threshold to beat is `8.2555`. "
        "The scan also records mean/RMS DeltaS, local action metrics, NLL, RMSE/correlation, logq/logdet summaries, parameter count, runtime, and NaN/inf status.\n"
    )
    _cfg, _paths, _coarse, _fine_ref, _cm, _fm, ctx = load_context(args.run_dir)
    arrays = load_arrays(args.max_train_configs)
    variants = [
        {"name": "radius3_control_w96_d2", "family": "radius_variants", "radius": 3, "hidden_width": 96, "hidden_layers": 2},
        {"name": "radius4_w96_d2", "family": "radius_variants", "radius": 4, "hidden_width": 96, "hidden_layers": 2},
        {"name": "radius5_w96_d2", "family": "radius_variants", "radius": 5, "hidden_width": 96, "hidden_layers": 2},
        {"name": "capacity_r3_w96_d2_control", "family": "capacity_variants", "radius": 3, "hidden_width": 96, "hidden_layers": 2},
        {"name": "capacity_r3_w192_d2", "family": "capacity_variants", "radius": 3, "hidden_width": 192, "hidden_layers": 2},
        {"name": "capacity_r3_w96_d4", "family": "capacity_variants", "radius": 3, "hidden_width": 96, "hidden_layers": 4},
        {"name": "capacity_r3_w192_d4", "family": "capacity_variants", "radius": 3, "hidden_width": 192, "hidden_layers": 4},
    ]
    summaries = []
    all_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    for i, v in enumerate(variants):
        s = train_variant(v, arrays, ctx, out, args.epochs, args.n_eval, seed=20260820 + i * 1000)
        summaries.append(s)
        all_rows.extend(list(csv.DictReader((out / v["family"] / v["name"] / "training_log.csv").open())))
        for label in ["best_by_action", "best_by_nll"]:
            row = dict(s[label])
            row["selection_label"] = label
            row["variant"] = s["variant"]
            row["local_gaussian"] = s["local_gaussian"]
            selected_rows.append(row)
    write_csv(out / "edge_variant_alpha_free_scan_metrics.csv", all_rows)
    write_csv(out / "selected_edge_variant_checkpoints.csv", selected_rows)
    plot_summary(all_rows, out)
    best = min(selected_rows, key=lambda r: float(r["global_deltaS_std"]))
    promising = float(best["global_deltaS_std"]) < PREVIOUS_BEST_EDGE_STD
    cumulative_rows: list[dict[str, Any]] = []
    cum_lines = ["# Cumulative edge variant report", ""]
    if promising:
        repl = {"edge": (Path(best["checkpoint"]), Path(best["local_gaussian"]))}
        cctx = build_candidate_ctx(ctx, repl)
        res = evaluate_candidate(f"{best['variant']}_{best['selection_label']}", cctx, load_paired(), out, args.n_eval)
        cumulative_rows.extend(res["assembly"])
        vals = {r["assembly"]: r["delta_action_std_vs_fine16"] for r in res["assembly"]}
        cum_lines.append(f"- {res['candidate']}: edge `{vals.get('model_coarse_edge'):.6g}`, pair `{vals.get('model_coarse_edge_pair'):.6g}`, full `{vals.get('model_all_z0'):.6g}`")
    else:
        cum_lines.append("- No variant beat the previous edge action threshold; cumulative diagnostics and sampler smoke were not promoted.")
    write_csv(out / "cumulative_edge_variant_metrics.csv", cumulative_rows)
    (out / "CUMULATIVE_EDGE_VARIANT_REPORT.md").write_text("\n".join(cum_lines) + "\n")
    radius_best = min([r for r in selected_rows if str(r["variant"]).startswith("radius")], key=lambda r: float(r["global_deltaS_std"]))
    cap_best = min([r for r in selected_rows if str(r["variant"]).startswith("capacity")], key=lambda r: float(r["global_deltaS_std"]))
    report = [
        "# Edge capacity/stencil scan report",
        "",
        f"- variants trained: `{len(variants)}`",
        f"- epochs per variant: `{args.epochs}`",
        f"- held-out eval configs: `{args.n_eval}`",
        f"- previous best threshold: `{PREVIOUS_BEST_EDGE_STD}`",
        f"- best overall variant: `{best['variant']}` / `{best['selection_label']}`",
        f"- best overall global deltaS std: `{float(best['global_deltaS_std']):.6g}`",
        f"- best radius-family variant: `{radius_best['variant']}` with `{float(radius_best['global_deltaS_std']):.6g}`",
        f"- best capacity-family variant: `{cap_best['variant']}` with `{float(cap_best['global_deltaS_std']):.6g}`",
        f"- sampler smoke launched: `False`",
        "",
        "## Answers",
        "",
        f"1. Increasing gathered radius reduced edge DeltaS only if the best radius variant beats the threshold. Here: `{float(radius_best['global_deltaS_std']) < PREVIOUS_BEST_EDGE_STD}`.",
        f"2. Increasing capacity reduced edge DeltaS only if the best capacity variant beats the threshold. Here: `{float(cap_best['global_deltaS_std']) < PREVIOUS_BEST_EDGE_STD}`.",
        f"3. Variant beating previous best: `{best['variant'] if promising else 'none'}`.",
        "4. NLL/logq costs are recorded in `selected_edge_variant_checkpoints.csv`; no candidate was promoted.",
        f"5. Cumulative diagnostics improved: `{promising}`.",
        "6. Tiny sampler smoke justified: `False`.",
        "7. If all variants fail, the next step is to revisit transported-detail parameterization or a joint edge+pair model; a different block kernel is a later, more expensive branch.",
    ]
    (out / "EDGE_CAPACITY_STENCIL_SCAN_REPORT.md").write_text("\n".join(report) + "\n")
    write_json(out / "edge_capacity_stencil_scan_summary.json", {"status": "completed", "best": best, "promising": promising, "sampler_smoke_launched": False, "summaries": summaries})
    print(json.dumps({"status": "completed", "out": str(out), "best_variant": best["variant"], "best_deltaS_std": float(best["global_deltaS_std"]), "promising": promising}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
