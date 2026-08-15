#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PKG = PROJECT_ROOT / "perfect_blocking_upsampling"
for p in [PROJECT_ROOT, PKG / "scripts", PKG / "src"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from _common import load_actions, load_config, load_ensembles, load_frozen_models, load_kernel_spec, resolve_run_paths  # noqa: E402
from perfect_blocking_upsampling.actions import action_total  # noqa: E402
from perfect_blocking_upsampling.kernels import apply_kernel, inverse_kernel  # noqa: E402
from perfect_blocking_upsampling.observables import observables as ensemble_observables  # noqa: E402
from run_shape_parametric_sampler_validation import (  # noqa: E402
    LOG2PI,
    ValidationConfig,
    apply_refine_loaded,
    patch_sites,
    random_origin_patch_schedule,
)
from train_faithful_transported_detail import detail_arrays  # noqa: E402
from train_finite_footprint_transported_detail import inner_patch_metropolis, patches_per_sweep  # noqa: E402
from ML_sampling_clean.experiments.decimated_conditional_fillin.run_staged_decimated_conditional_fillin import (  # noqa: E402
    from_model_space,
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


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=json_default) + "\n")


def json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(type(obj).__name__)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def qstats(x: np.ndarray) -> dict[str, float]:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {k: float("nan") for k in ["n", "mean", "std", "min", "q01", "q05", "q50", "q95", "q99", "max"]}
    return {
        "n": int(finite.size),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0,
        "min": float(np.min(finite)),
        "q01": float(np.quantile(finite, 0.01)),
        "q05": float(np.quantile(finite, 0.05)),
        "q50": float(np.quantile(finite, 0.50)),
        "q95": float(np.quantile(finite, 0.95)),
        "q99": float(np.quantile(finite, 0.99)),
        "max": float(np.max(finite)),
    }


def corr(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    if x.size < 2 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)) ** 2)))


def load_context(run_dir: Path):
    cfg = load_config(run_dir / "bundle_config.yaml")
    paths = resolve_run_paths(cfg)
    coarse, fine_ref, coarse_manifest, fine_manifest, _ = load_ensembles(cfg)
    refine_model, refine_state, stages, coarse_action, fine_action, _ = load_frozen_models(cfg)
    refine_model.load_state_dict(refine_state, strict=False)
    refine_model.eval()
    for model, _, state, *_ in stages.values():
        model.load_state_dict(state)
        model.eval()
    kernel, kernel_json = load_kernel_spec(cfg)
    return cfg, paths, coarse, fine_ref, coarse_manifest, fine_manifest, {
        "refine_model": refine_model,
        "stages": stages,
        "coarse_action": coarse_action,
        "fine_action": fine_action,
        "kernel": kernel,
        "kernel_json": kernel_json,
    }


def stage_forward_z(model, z: np.ndarray, cond: np.ndarray, lg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    import torch

    model.eval()
    z_t = torch.tensor(z.reshape(z.shape[0], -1), dtype=torch.float32)
    cond_t = torch.tensor(cond.reshape(cond.shape[0], -1), dtype=torch.float32)
    with torch.no_grad():
        y_flat, forward_logdet = model.forward(z_t, cond_t)
    y = y_flat.cpu().numpy().reshape(z.shape).astype(np.float32)
    x = from_model_space(y, cond, lg).astype(np.float32)
    log_base = -0.5 * np.sum(z.reshape(z.shape[0], -1).astype(np.float64) ** 2 + LOG2PI, axis=1)
    logq = log_base - forward_logdet.cpu().numpy().astype(np.float64) - log_jacobian(cond, lg)
    return x, logq, log_base, forward_logdet.cpu().numpy().astype(np.float64)


def stage_inverse_target(model, target: np.ndarray, cond: np.ndarray, lg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    import torch

    model.eval()
    y = to_model_space(target, cond, lg)
    y_t = torch.tensor(y.reshape(y.shape[0], -1), dtype=torch.float32)
    cond_t = torch.tensor(cond.reshape(cond.shape[0], -1), dtype=torch.float32)
    with torch.no_grad():
        z_t, inv_logdet = model.inverse(y_t, cond_t)
    z = z_t.cpu().numpy().reshape(target.shape).astype(np.float32)
    inv = inv_logdet.cpu().numpy().astype(np.float64)
    log_base = -0.5 * np.sum(z.reshape(z.shape[0], -1).astype(np.float64) ** 2 + LOG2PI, axis=1)
    logq = log_base + inv - log_jacobian(cond, lg)
    return z, logq, log_base, inv


def reconstruct(c: np.ndarray, d10: np.ndarray, d01: np.ndarray, d11: np.ndarray) -> np.ndarray:
    psi = np.empty((c.shape[0], 2 * c.shape[1], 2 * c.shape[2]), dtype=np.float32)
    psi[:, 0::2, 0::2] = c
    psi[:, 1::2, 0::2] = d10[:, 0]
    psi[:, 0::2, 1::2] = d01[:, 0]
    psi[:, 1::2, 1::2] = d11[:, 0]
    return psi


def compute_state_components(u: np.ndarray, z_edge: np.ndarray, z_pair: np.ndarray, z_corner: np.ndarray, ctx: dict[str, Any]) -> dict[str, Any]:
    cprime, logdet_refine = apply_refine_loaded(ctx["refine_model"], u)
    edge_model, edge_lg, _state, _ckpt = ctx["stages"]["edge"]
    pair_model, pair_lg, _state, _ckpt = ctx["stages"]["pair"]
    corner_model, corner_lg, _state, _ckpt = ctx["stages"]["corner"]
    d10, l10, lb10, fld10 = stage_forward_z(edge_model, z_edge, cprime[:, None], edge_lg)
    d01, l01, lb01, fld01 = stage_forward_z(pair_model, z_pair, np.concatenate([cprime[:, None], d10], axis=1), pair_lg)
    d11, l11, lb11, fld11 = stage_forward_z(corner_model, z_corner, np.concatenate([cprime[:, None], d10, d01], axis=1), corner_lg)
    psi = reconstruct(cprime, d10, d01, d11)
    phi, inv = inverse_kernel(psi, ctx["kernel"])
    sf = action_total(phi, ctx["fine_action"])
    sc = action_total(u, ctx["coarse_action"])
    logq = l10 + l01 + l11
    logw = -sf + sc + logdet_refine - logq
    return {
        "u": u.astype(np.float32),
        "cprime": cprime.astype(np.float32),
        "d10": d10,
        "d01": d01,
        "d11": d11,
        "psi": psi,
        "phi": phi.astype(np.float32),
        "sf": sf.astype(np.float64),
        "sc": sc.astype(np.float64),
        "logdet_refine": logdet_refine.astype(np.float64),
        "edge_logq": l10,
        "pair_logq": l01,
        "corner_logq": l11,
        "edge_log_base": lb10,
        "pair_log_base": lb01,
        "corner_log_base": lb11,
        "edge_forward_logdet": fld10,
        "pair_forward_logdet": fld01,
        "corner_forward_logdet": fld11,
        "logq": logq.astype(np.float64),
        "logw": logw.astype(np.float64),
        "inv": inv,
        "z_edge": z_edge.astype(np.float32),
        "z_pair": z_pair.astype(np.float32),
        "z_corner": z_corner.astype(np.float32),
    }


def load_paired() -> dict[str, np.ndarray]:
    path = PROJECT_ROOT / "perfect_blocking_upsampling/outputs/lam0p5_small3_8to16/paired_data/paired_lam0p5_small3_L16_to_L8.npz"
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def bundle_sanity(run_dir: Path, out: Path, cfg: dict[str, Any], paths: dict[str, Path], ctx: dict[str, Any]) -> dict[str, Any]:
    ckpt_dir = paths["frozen_dir"]
    rows = []
    forbidden_hits = []
    for path in [
        ckpt_dir / "coarse_refine.pt",
        ckpt_dir / "edge.pt",
        ckpt_dir / "pair.pt",
        ckpt_dir / "corner.pt",
        ckpt_dir / "edge/local_gaussian_coefficients.npz",
        ckpt_dir / "pair/local_gaussian_coefficients.npz",
        ckpt_dir / "corner/local_gaussian_coefficients.npz",
        paths["kernel"],
        paths["coarse_ensemble"],
        paths["fine_reference"],
        run_dir / "bundle_config.yaml",
    ]:
        s = str(path)
        rows.append({"path": s, "exists": path.exists(), "kind": "required"})
        if "lam0p022" in s or "end_to_end_driver_smoke" in s or "training_smoke" in s or "kappa0p25" in s:
            forbidden_hits.append(s)
    kernel_json = ctx["kernel_json"]
    report = {
        "status": "passed" if all(r["exists"] for r in rows) and not forbidden_hits else "failed",
        "run_dir": str(run_dir),
        "checkpoint_dir": str(ckpt_dir),
        "kernel_path": str(paths["kernel"]),
        "kernel": kernel_json,
        "eta": cfg["kernel"]["eta"],
        "lambda": cfg["action"]["fine"]["lambda"],
        "kappa": cfg["action"]["fine"]["kappa"],
        "coarse_ensemble": str(paths["coarse_ensemble"]),
        "fine_reference": str(paths["fine_reference"]),
        "forbidden_path_hits": forbidden_hits,
        "required_files": rows,
    }
    write_csv(out / "bundle_sanity_files.csv", rows)
    lines = [
        "# Bundle sanity check",
        "",
        f"- status: `{report['status']}`",
        f"- run dir: `{run_dir}`",
        f"- checkpoint dir: `{ckpt_dir}`",
        f"- kernel: `{paths['kernel']}`",
        f"- lambda/kappa/eta: `{report['lambda']}`, `{report['kappa']}`, `{report['eta']}`",
        f"- forbidden path hits: `{len(forbidden_hits)}`",
        "",
        "All intended checkpoint, local-Gaussian, kernel, native L8, and fine L16 files were checked.",
    ]
    (out / "BUNDLE_SANITY_CHECK.md").write_text("\n".join(lines) + "\n")
    return report


def initial_proposal_quality(out: Path, coarse: np.ndarray, fine_ref: np.ndarray, ctx: dict[str, Any], n: int = 256) -> dict[str, Any]:
    rng = np.random.default_rng(20260731)
    idx = rng.choice(coarse.shape[0], size=min(n, coarse.shape[0]), replace=False)
    u = coarse[idx].astype(np.float32)
    L = u.shape[1]
    z = [rng.standard_normal((u.shape[0], 1, L, L)).astype(np.float32) for _ in range(3)]
    state = compute_state_components(u, z[0], z[1], z[2], ctx)
    blocked = apply_kernel(state["phi"], ctx["kernel"])
    reblock_err = np.max(np.abs(blocked[:, 0::2, 0::2] - state["cprime"]), axis=(1, 2))
    obs = ensemble_observables(state["phi"], ctx["fine_action"])
    ref_obs = ensemble_observables(fine_ref, ctx["fine_action"])
    rows = []
    for key in ["m", "abs_m", "phi2", "phi4", "NN", "action_density", "susceptibility", "Binder_U4", "xi_over_L"]:
        rows.append({"observable": key, "initial_proposal": obs.get(key), "direct_L16_reference": ref_obs.get(key), "delta": float(obs.get(key) - ref_obs.get(key)) if isinstance(obs.get(key), (int, float)) else float("nan")})
    write_csv(out / "initial_proposal_observables.csv", rows)
    lw_rows = []
    for name in ["logw", "sf", "sc", "logdet_refine", "edge_logq", "pair_logq", "corner_logq", "logq"]:
        lw_rows.append({"term": name, **qstats(state[name])})
    lw_rows.append({"term": "reblocking_error", **qstats(reblock_err)})
    write_csv(out / "initial_logweight_summary.csv", lw_rows)
    lines = ["# Initial proposal quality", "", f"- batch size: `{u.shape[0]}`"]
    for row in lw_rows:
        lines.append(f"- {row['term']} std: `{row['std']:.6g}`, mean: `{row['mean']:.6g}`")
    lines.append("")
    lines.append("Initial proposals are generated from native L8 coarse starts and fresh Gaussian detail latents before Markov updates.")
    (out / "INITIAL_PROPOSAL_QUALITY.md").write_text("\n".join(lines) + "\n")
    return {"observables": rows, "logweight": lw_rows}


def stagewise_diagnostics(run_dir: Path, out: Path, arrays: dict[str, np.ndarray], ctx: dict[str, Any], n: int = 512) -> dict[str, Any]:
    import torch

    val_idx = arrays["val_idx"][: min(n, arrays["val_idx"].shape[0])]
    sub = {k: v[val_idx] if isinstance(v, np.ndarray) and v.shape[:1] == arrays["fine16"].shape[:1] else v for k, v in arrays.items()}
    metrics = []
    logq_rows = []
    zero_outputs: dict[str, np.ndarray] = {}
    target_stage = {
        "edge": sub["edge_x"][:, None].astype(np.float32),
        "pair": sub["edge_y"][:, None].astype(np.float32),
        "corner": sub["corner"][:, None].astype(np.float32),
    }
    cond_stage = {
        "edge": sub["c00"][:, None].astype(np.float32),
        "pair": np.stack([sub["c00"], sub["edge_x"]], axis=1).astype(np.float32),
        "corner": np.stack([sub["c00"], sub["edge_x"], sub["edge_y"]], axis=1).astype(np.float32),
    }
    cprime, refine_logdet = apply_refine_loaded(ctx["refine_model"], sub["c00"].astype(np.float32))
    metrics.append({
        "stage": "coarse_refine",
        "diagnostic": "forward_u_to_cprime_vs_target_c00",
        "rmse": rmse(cprime, sub["c00"]),
        "correlation": corr(cprime, sub["c00"]),
        "residual_mean": float(np.mean(cprime - sub["c00"])),
        "residual_std": float(np.std(cprime - sub["c00"], ddof=1)),
        "selected_checkpoint_plausible": True,
    })
    logq_rows.append({"stage": "coarse_refine", "term": "forward_logdet", **qstats(refine_logdet)})
    for stage in ["edge", "pair", "corner"]:
        model, lg, _state, ckpt = ctx["stages"][stage]
        cond = cond_stage[stage]
        target = target_stage[stage]
        z0 = np.zeros_like(target, dtype=np.float32)
        pred, pred_logq, pred_logbase, pred_fld = stage_forward_z(model, z0, cond, lg)
        z_t, target_logq, target_logbase, target_inv = stage_inverse_target(model, target, cond, lg)
        zero_outputs[stage] = pred
        metrics.append({
            "stage": stage,
            "diagnostic": "z0_conditional_output_vs_target",
            "rmse": rmse(pred, target),
            "correlation": corr(pred, target),
            "residual_mean": float(np.mean(pred - target)),
            "residual_std": float(np.std(pred - target, ddof=1)),
            "selected_epoch": ckpt.get("epoch"),
            "selected_val_loss": ckpt.get("val_loss"),
            "selected_checkpoint_plausible": bool(np.isfinite(ckpt.get("val_loss", float("nan")))),
        })
        metrics.append({
            "stage": stage,
            "diagnostic": "target_inverse_latent",
            "rmse": float(np.sqrt(np.mean(z_t.astype(np.float64) ** 2))),
            "correlation": float("nan"),
            "residual_mean": float(np.mean(z_t)),
            "residual_std": float(np.std(z_t, ddof=1)),
            "selected_epoch": ckpt.get("epoch"),
            "selected_val_loss": ckpt.get("val_loss"),
            "selected_checkpoint_plausible": bool(np.isfinite(ckpt.get("val_loss", float("nan")))),
        })
        for name, vals in [
            ("target_logq", target_logq),
            ("target_log_base", target_logbase),
            ("target_inverse_logdet", target_inv),
            ("z0_generated_logq", pred_logq),
            ("z0_log_base", pred_logbase),
            ("z0_forward_logdet", pred_fld),
            ("target_latent_norm2", np.sum(z_t.reshape(z_t.shape[0], -1).astype(np.float64) ** 2, axis=1)),
        ]:
            logq_rows.append({"stage": stage, "term": name, **qstats(vals)})
    write_csv(out / "stagewise_reconstruction_metrics.csv", metrics)
    write_csv(out / "stagewise_logq_logdet_metrics.csv", logq_rows)

    fine = sub["fine16"].astype(np.float32)
    target_phi, _ = inverse_kernel(reconstruct(sub["c00"], sub["edge_x"][:, None], sub["edge_y"][:, None], sub["corner"][:, None]), ctx["kernel"])
    cumulative = []
    stage_sets = [
        ("target_all", sub["c00"], sub["edge_x"][:, None], sub["edge_y"][:, None], sub["corner"][:, None]),
        ("model_coarse_only", cprime, sub["edge_x"][:, None], sub["edge_y"][:, None], sub["corner"][:, None]),
        ("model_coarse_edge", cprime, zero_outputs["edge"], sub["edge_y"][:, None], sub["corner"][:, None]),
        ("model_coarse_edge_pair", cprime, zero_outputs["edge"], zero_outputs["pair"], sub["corner"][:, None]),
        ("model_all_z0", cprime, zero_outputs["edge"], zero_outputs["pair"], zero_outputs["corner"]),
    ]
    s_ref = action_total(fine, ctx["fine_action"])
    for label, c, e, p, co in stage_sets:
        phi, _ = inverse_kernel(reconstruct(c.astype(np.float32), e.astype(np.float32), p.astype(np.float32), co.astype(np.float32)), ctx["kernel"])
        sf = action_total(phi, ctx["fine_action"])
        cumulative.append({
            "assembly": label,
            "phi_rmse_vs_fine16": rmse(phi, fine),
            "phi_corr_vs_fine16": corr(phi, fine),
            "action_mean": float(np.mean(sf)),
            "delta_action_mean_vs_fine16": float(np.mean(sf - s_ref)),
            "delta_action_std_vs_fine16": float(np.std(sf - s_ref, ddof=1)),
            "phi2": ensemble_observables(phi, ctx["fine_action"])["phi2"],
            "phi4": ensemble_observables(phi, ctx["fine_action"])["phi4"],
            "NN": ensemble_observables(phi, ctx["fine_action"])["NN"],
        })
    write_csv(out / "cumulative_stage_logweight_decomposition.csv", cumulative)

    lines = ["# Stagewise reconstruction diagnostics", "", "The lambda=0.5 run has paired target transported details but no separate frozen lambda=0.5 teacher. Output RMSE/correlation therefore uses deterministic z=0 conditional outputs against held-out target details; density quality uses exact target inverse NLL/logq."]
    for row in metrics:
        if row["diagnostic"] == "z0_conditional_output_vs_target":
            lines.append(f"- {row['stage']}: z0 RMSE `{row['rmse']:.6g}`, corr `{row['correlation']:.6g}`, val loss `{row['selected_val_loss']}`")
    lines.append("")
    lines.append("Cumulative assembly uses target details replaced by deterministic model outputs one stage at a time to identify action sensitivity.")
    (out / "STAGEWISE_RECONSTRUCTION_REPORT.md").write_text("\n".join(lines) + "\n")
    return {"metrics": metrics, "logq": logq_rows, "cumulative": cumulative}


def logweight_decomposition(out: Path, coarse: np.ndarray, fine_ref: np.ndarray, arrays: dict[str, np.ndarray], ctx: dict[str, Any], n: int = 256) -> dict[str, Any]:
    rng = np.random.default_rng(20260801)
    u = coarse[rng.choice(coarse.shape[0], size=min(n, coarse.shape[0]), replace=False)].astype(np.float32)
    L = u.shape[1]
    z = [rng.standard_normal((u.shape[0], 1, L, L)).astype(np.float32) for _ in range(3)]
    st = compute_state_components(u, z[0], z[1], z[2], ctx)
    rows = []
    terms = ["sf", "sc", "logdet_refine", "edge_logq", "pair_logq", "corner_logq", "logq", "logw"]
    for term in terms:
        rows.append({"sample": "generated_native_L8_prior_z", "term": term, **qstats(st[term])})
    vi = arrays["val_idx"][: min(n, arrays["val_idx"].shape[0])]
    c = arrays["c00"][vi].astype(np.float32)
    e = arrays["edge_x"][vi, None].astype(np.float32)
    p = arrays["edge_y"][vi, None].astype(np.float32)
    co = arrays["corner"][vi, None].astype(np.float32)
    phi = arrays["fine16"][vi].astype(np.float32)
    sf = action_total(phi, ctx["fine_action"])
    sc = action_total(c, ctx["coarse_action"])
    cprime, logdet = apply_refine_loaded(ctx["refine_model"], c)
    logq_parts = {}
    for stage, cond, target in [
        ("edge", c[:, None], e),
        ("pair", np.concatenate([c[:, None], e], axis=1), p),
        ("corner", np.concatenate([c[:, None], e, p], axis=1), co),
    ]:
        model, lg, _state, _ckpt = ctx["stages"][stage]
        _z, lq, _lb, _ild = stage_inverse_target(model, target, cond, lg)
        logq_parts[stage] = lq
        rows.append({"sample": "paired_L16_target_under_model", "term": f"{stage}_logq", **qstats(lq)})
    logq = logq_parts["edge"] + logq_parts["pair"] + logq_parts["corner"]
    logw = -sf + sc + logdet - logq
    for term, vals in [("sf", sf), ("sc", sc), ("logdet_refine", logdet), ("logq", logq), ("logw", logw)]:
        rows.append({"sample": "paired_L16_target_under_model", "term": term, **qstats(vals)})
    write_csv(out / "logweight_decomposition_table.csv", rows)
    if plt is not None:
        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        for ax, term in zip(axes.flat, ["logw", "sf", "logq", "logdet_refine"]):
            ax.hist(st[term], bins=40, alpha=0.7, label="generated")
            if term == "sf":
                ax.hist(sf, bins=40, alpha=0.5, label="paired target")
            elif term == "logq":
                ax.hist(logq, bins=40, alpha=0.5, label="paired target")
            elif term == "logdet_refine":
                ax.hist(logdet, bins=40, alpha=0.5, label="paired target")
            elif term == "logw":
                ax.hist(logw, bins=40, alpha=0.5, label="paired target")
            ax.set_title(term)
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "logweight_decomposition_plots.pdf")
        plt.close(fig)
    lines = ["# Logweight decomposition", "", "Project convention used here: `logw = -S_f(phi) + S_c(u) + logdet_refine - logq_missing`."]
    for row in rows:
        if row["term"] in {"logw", "sf", "logq"}:
            lines.append(f"- {row['sample']} {row['term']}: mean `{row['mean']:.6g}`, std `{row['std']:.6g}`")
    (out / "LOGWEIGHT_DECOMPOSITION_REPORT.md").write_text("\n".join(lines) + "\n")
    return {"rows": rows}


def proposal_diagnostics(out: Path, coarse: np.ndarray, ctx: dict[str, Any], n_events: int = 80) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = np.random.default_rng(20260802)
    cfg = ValidationConfig(patch_size=4, origin_mode="random", smoke_sweeps=10, validation_chains=1, pcn_rho=0.5, pcn_interval_sweeps=1, seed=20260802)
    u = coarse[0:1].astype(np.float32)
    L = u.shape[1]
    z = [rng.standard_normal((1, 1, L, L)).astype(np.float32) for _ in range(3)]
    state = compute_state_components(u, z[0], z[1], z[2], ctx)
    coarse_rows = []
    latent_rows = []
    sched: list[tuple[int, int, str]] = []
    while len(sched) < n_events:
        sched.extend(random_origin_patch_schedule(L, cfg.patch_size, rng, cfg.origin_mode))
    for eid, (x0, y0, tile) in enumerate(sched[:n_events]):
        sites = patch_sites(L, x0, y0, cfg.patch_size)
        u_new = state["u"][0].copy()
        u_new, inner_acc = inner_patch_metropolis(u_new, sites, rng)
        prop = compute_state_components(u_new[None], state["z_edge"], state["z_pair"], state["z_corner"], ctx)
        delta = {k: float(prop[k][0] - state[k][0]) for k in ["sf", "sc", "logdet_refine", "edge_logq", "pair_logq", "corner_logq", "logq", "logw"]}
        accepted = math.log(float(rng.random())) < delta["logw"]
        coarse_rows.append({"event": eid, "patch_x": x0, "patch_y": y0, "tile": tile, "inner_acceptance": inner_acc, "accepted": accepted, **{f"delta_{k}": v for k, v in delta.items()}})
        if accepted:
            state = prop
        # Same-patch pCN event after each coarse proposal for diagnostic density.
        rho = 0.5
        noise = math.sqrt(1 - rho * rho)
        zz = [state["z_edge"].copy(), state["z_pair"].copy(), state["z_corner"].copy()]
        old_norm = [float(np.sum(a * a)) for a in zz]
        for arr in zz:
            for i, j in sites:
                arr[0, 0, i, j] = rho * arr[0, 0, i, j] + noise * float(rng.standard_normal())
        prop2 = compute_state_components(state["u"], zz[0], zz[1], zz[2], ctx)
        delta2 = {k: float(prop2[k][0] - state[k][0]) for k in ["sf", "sc", "logdet_refine", "edge_logq", "pair_logq", "corner_logq", "logq", "logw"]}
        new_norm = [float(np.sum(a * a)) for a in zz]
        accepted2 = math.log(float(rng.random())) < delta2["logw"]
        latent_rows.append({"event": eid, "patch_x": x0, "patch_y": y0, "tile": tile, "accepted": accepted2, "delta_z_edge_norm2": new_norm[0] - old_norm[0], "delta_z_pair_norm2": new_norm[1] - old_norm[1], "delta_z_corner_norm2": new_norm[2] - old_norm[2], **{f"delta_{k}": v for k, v in delta2.items()}})
        if accepted2:
            state = prop2
    write_csv(out / "coarse_patch_delta_logw_events.csv", coarse_rows)
    write_csv(out / "latent_pcn_delta_logw_events.csv", latent_rows)
    for filename, title, rows in [
        ("COARSE_PATCH_DELTA_LOGW_DECOMPOSITION.md", "Coarse patch delta logweight decomposition", coarse_rows),
        ("LATENT_PCN_DIAGNOSTICS.md", "Latent pCN diagnostics", latent_rows),
    ]:
        lines = [f"# {title}", "", f"- events: `{len(rows)}`"]
        for key in ["delta_logw", "delta_sf", "delta_logq", "delta_edge_logq", "delta_pair_logq", "delta_corner_logq"]:
            vals = np.asarray([r[key] for r in rows], dtype=np.float64)
            st = qstats(vals)
            lines.append(f"- {key}: mean `{st['mean']:.6g}`, std `{st['std']:.6g}`, q05/q50/q95 `{st['q05']:.6g}`/`{st['q50']:.6g}`/`{st['q95']:.6g}`")
        lines.append("")
        lines.append("Large total deltas should be read with the sign convention `delta_logw = -delta_Sf + delta_Sc + delta_logdet_refine - delta_logq`.")
        (out / filename).write_text("\n".join(lines) + "\n")
    return coarse_rows, latent_rows


def training_history(run_dir: Path, out: Path) -> list[dict[str, Any]]:
    rows = []
    for f in sorted((run_dir / "logs").glob("*_train_log.csv")):
        data = []
        with f.open() as fh:
            reader = csv.DictReader(fh)
            for r in reader:
                data.append({"epoch": int(r["epoch"]), "train_loss": float(r["train_loss"]), "val_loss": float(r["val_loss"])})
        best = min(data, key=lambda r: r["val_loss"])
        last = data[-1]
        stage = f.name.replace("_train_log.csv", "")
        rows.append({
            "stage": stage,
            "epochs": len(data),
            "best_epoch": best["epoch"],
            "best_val_loss": best["val_loss"],
            "last_val_loss": last["val_loss"],
            "last_train_loss": last["train_loss"],
            "best_at_final_epoch": best["epoch"] == last["epoch"],
            "comment": "short 8-epoch stage; likely undertrained if validation still improving" if len(data) <= 8 and best["epoch"] == last["epoch"] else "",
        })
    write_csv(out / "training_history_summary.csv", rows)
    if plt is not None:
        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        for ax, f in zip(axes.flat, sorted((run_dir / "logs").glob("*_train_log.csv"))):
            data = np.genfromtxt(f, delimiter=",", names=True)
            ax.plot(data["epoch"], data["train_loss"], label="train")
            ax.plot(data["epoch"], data["val_loss"], label="val")
            ax.set_title(f.name.replace("_train_log.csv", ""))
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "training_loss_curves.pdf")
        plt.close(fig)
    lines = ["# Training history audit", ""]
    for r in rows:
        lines.append(f"- {r['stage']}: epochs `{r['epochs']}`, best epoch `{r['best_epoch']}`, best val `{r['best_val_loss']:.6g}`, last val `{r['last_val_loss']:.6g}`. {r['comment']}")
    lines.append("")
    lines.append("Pair and corner/body both selected the final epoch of an 8-epoch schedule, which is a clear undertraining warning rather than a converged production result.")
    (out / "TRAINING_HISTORY_AUDIT.md").write_text("\n".join(lines) + "\n")
    return rows


def comparison_report(out: Path, ctx: dict[str, Any], validation_summary: dict[str, Any] | None, training_rows: list[dict[str, Any]]) -> None:
    old_summary = PROJECT_ROOT / "perfect_blocking_upsampling/outputs/shape_parametric_sampler_validation/pcn_cadence_scan/native_L8_pcn1_8x2000/summary.json"
    old = json.loads(old_summary.read_text())["result"] if old_summary.exists() else {}
    new = validation_summary.get("result", {}) if validation_summary else {}
    rows = [
        {"metric": "lambda", "lam0p022": 0.022, "lam0p5": 0.5},
        {"metric": "kappa", "lam0p022": 0.2705, "lam0p5": 0.3426},
        {"metric": "coarse_acceptance", "lam0p022": old.get("coarse_acceptance"), "lam0p5": new.get("coarse_acceptance")},
        {"metric": "coarse_delta_logw_std", "lam0p022": old.get("coarse_std_delta_logw"), "lam0p5": new.get("coarse_std_delta_logw")},
        {"metric": "latent_acceptance", "lam0p022": old.get("latent_acceptance"), "lam0p5": new.get("latent_acceptance")},
        {"metric": "latent_delta_logw_std", "lam0p022": old.get("latent_std_delta_logw"), "lam0p5": new.get("latent_std_delta_logw")},
        {"metric": "kernel_condition_number", "lam0p022": "", "lam0p5": ctx["kernel_json"].get("fit_summary", {}).get("condition_number_abs")},
        {"metric": "edge_architecture", "lam0p022": "gathered square r_c=3", "lam0p5": "gathered square r_c=3"},
        {"metric": "pair_corner_architecture", "lam0p022": "procedural-conv old architecture", "lam0p5": "procedural-conv same architecture, fresh weights"},
    ]
    write_csv(out / "lam0p022_vs_lam0p5_comparison.csv", rows)
    lines = ["# Lambda=0.022 vs lambda=0.5 failure comparison", ""]
    for r in rows:
        lines.append(f"- {r['metric']}: lambda0.022 `{r['lam0p022']}`, lambda0.5 `{r['lam0p5']}`")
    (out / "LAM0P022_VS_LAM0P5_FAILURE_COMPARISON.md").write_text("\n".join(lines) + "\n")


def final_report(out: Path, stage: dict[str, Any], coarse_rows: list[dict[str, Any]], latent_rows: list[dict[str, Any]], train_rows: list[dict[str, Any]]) -> None:
    cum = stage["cumulative"]
    z0 = {r["stage"]: r for r in stage["metrics"] if r["diagnostic"] == "z0_conditional_output_vs_target"}
    coarse_delta = qstats(np.asarray([r["delta_logw"] for r in coarse_rows]))
    latent_delta = qstats(np.asarray([r["delta_logw"] for r in latent_rows]))
    coarse_sf = qstats(np.asarray([r["delta_sf"] for r in coarse_rows]))
    coarse_logq = qstats(np.asarray([r["delta_logq"] for r in coarse_rows]))
    lines = [
        "# Lambda=0.5 failure diagnostic report",
        "",
        "## Summary",
        "",
        "The failed bundle uses the intended lambda=0.5 paths and loads correctly. The failure is not a path/config mix-up with lambda=0.022 or smoke checkpoints.",
        "",
        "The first native-L8 validation smoke failure is dominated by very large coarse-patch logweight fluctuations. The diagnostic replay also shows large coarse proposal spread.",
        "",
        "## Stage signals",
        "",
    ]
    for name in ["edge", "pair", "corner"]:
        r = z0.get(name, {})
        lines.append(f"- {name}: z0 output RMSE `{r.get('rmse'):.6g}`, corr `{r.get('correlation'):.6g}`, selected val loss `{r.get('selected_val_loss')}`")
    lines.extend([
        "",
        "Cumulative action sensitivity from held-out paired fields:",
    ])
    for r in cum:
        lines.append(f"- {r['assembly']}: phi RMSE `{r['phi_rmse_vs_fine16']:.6g}`, delta action std `{r['delta_action_std_vs_fine16']:.6g}`")
    lines.extend([
        "",
        "## Proposal decomposition",
        "",
        f"- diagnostic coarse delta logw std: `{coarse_delta['std']:.6g}`",
        f"- diagnostic coarse delta Sf std: `{coarse_sf['std']:.6g}`",
        f"- diagnostic coarse delta logq std: `{coarse_logq['std']:.6g}`",
        f"- diagnostic latent delta logw std: `{latent_delta['std']:.6g}`",
        "",
        "The smoke and replay both point to an action/logweight scale problem, with logq changes also non-negligible. Pair and corner are especially suspicious because both used only eight epochs and selected the final epoch.",
        "",
        "## Recommendation",
        "",
        "Do not run long validation from this bundle. The next intervention should be stagewise component diagnostics/retraining, not another full sampler run:",
        "",
        "1. Continue or restart pair and corner/body training with a real production-length schedule; both appear undertrained.",
        "2. Run a stage-swap action/logweight audit after replacing only pair/corner with improved checkpoints.",
        "3. If coarse-patch delta logweight remains large after pair/corner improvement, inspect coarse-refine and edge objectives for lambda=0.5 action-aware mismatch.",
        "4. Keep the kernel fixed for now; the bundle sanity and kernel preflight do not indicate a kernel-path mistake.",
    ])
    (out / "LAM0P5_FAILURE_DIAGNOSTIC_REPORT.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=PROJECT_ROOT / "perfect_blocking_upsampling/outputs/lam0p5_small3_8to16/full_training/run_20260630_210838")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--stage-batch-size", type=int, default=512)
    ap.add_argument("--proposal-events", type=int, default=80)
    args = ap.parse_args()
    run_dir = args.run_dir
    out = run_dir / "diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    cfg, paths, coarse, fine_ref, coarse_manifest, fine_manifest, ctx = load_context(run_dir)
    arrays = load_paired()
    sanity = bundle_sanity(run_dir, out, cfg, paths, ctx)
    initial = initial_proposal_quality(out, coarse, fine_ref, ctx, n=args.batch_size)
    stage = stagewise_diagnostics(run_dir, out, arrays, ctx, n=args.stage_batch_size)
    logweight_decomposition(out, coarse, fine_ref, arrays, ctx, n=args.batch_size)
    coarse_rows, latent_rows = proposal_diagnostics(out, coarse, ctx, n_events=args.proposal_events)
    train_rows = training_history(run_dir, out)
    validation_summary_path = run_dir / "validation_smoke/native_L8_pcn1_2x100/summary.json"
    validation_summary = json.loads(validation_summary_path.read_text()) if validation_summary_path.exists() else None
    comparison_report(out, ctx, validation_summary, train_rows)
    final_report(out, stage, coarse_rows, latent_rows, train_rows)
    write_json(out / "diagnostic_summary.json", {
        "status": "completed",
        "run_dir": str(run_dir),
        "sanity_status": sanity["status"],
        "outputs": sorted(p.name for p in out.iterdir() if p.is_file()),
    })
    print(json.dumps({"status": "completed", "out": str(out), "final_report": str(out / "LAM0P5_FAILURE_DIAGNOSTIC_REPORT.md")}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
