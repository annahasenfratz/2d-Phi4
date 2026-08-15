#!/usr/bin/env python3
"""Train one local finite-footprint L16->L32 conditional detail candidate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FINITE = PROJECT_ROOT / "InverseBlocking_lam0p022_finite_footprint_flow"
FROZEN = PROJECT_ROOT / "InverseBlocking_lam0p022_frozen"
sys.path.insert(0, str(FINITE / "scripts"))
sys.path.insert(0, str(FROZEN / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "InverseBlocking_lam0p022_restricted_local_small3" / "scripts"))

from perfect_blocking_upsampling.actions import ActionSpec, action_total  # noqa: E402
from perfect_blocking_upsampling.kernels import apply_kernel, inverse_kernel, load_kernel  # noqa: E402
from train_finite_footprint_flow import PatchAffineNF, condition_grid, gather_site_features_np, logweight_diagnostics, write_csv, write_json  # noqa: E402
from train_restricted_local_small3 import load_wolff, local_observables, reconstruct, split_psi, stable_ess  # noqa: E402

LAMBDA = 0.022
KAPPA = 0.2705
COARSE_L = 16
FINE_L = 32
FINE = PROJECT_ROOT / "phi4_phase-diagram/ensembles/lam0p022_kappa0p2705_L32_embedded_wolff_sign_cluster_plus_radial_heatbath/configs.npz"
PRIOR = PROJECT_ROOT / "phi4_phase-diagram/ensembles/lam0p022_kappa0p2705_L16_embedded_wolff_sign_cluster_plus_radial_heatbath_N5000/configs.npz"
KERNEL = PROJECT_ROOT / "InverseBlocking_lam0p022_rg_blocking_validation/configs/kernels/fixedeta_5x5_lam0p022.json"
OUT_ROOT = PROJECT_ROOT / "perfect_blocking_upsampling/outputs/lam0p022_L16to32_flow_footprint_scan"
LOG2PI = math.log(2.0 * math.pi)


@dataclass
class CandidateConfig:
    candidate: str
    footprint: int
    epochs: int = 10
    max_train: int = 1024
    max_val: int = 256
    n_proposals: int = 512
    batch_size_sites: int = 8192
    hidden_channels: int = 64
    conditioner_layers: int = 3
    learning_rate: float = 2.0e-3
    seed: int = 20260704
    base_init: bool = True
    output_dir: str = ""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def footprint_offsets(size: int) -> list[tuple[int, int]]:
    if size < 1:
        raise ValueError(f"footprint must be positive, got {size}")
    if size % 2 == 1:
        r = size // 2
        return [(di, dj) for di in range(-r, r + 1) for dj in range(-r, r + 1)]
    lo = -(size // 2 - 1)
    hi = size // 2
    return [(di, dj) for di in range(lo, hi + 1) for dj in range(lo, hi + 1)]


def max_radius(size: int) -> int:
    return max(max(abs(i), abs(j)) for i, j in footprint_offsets(size))


def detail_site_coords(stage: str, l_coarse: int) -> tuple[np.ndarray, np.ndarray]:
    ii, jj = np.meshgrid(np.arange(l_coarse), np.arange(l_coarse), indexing="ij")
    if stage == "edge_x":
        return 2 * ii + 1, 2 * jj
    if stage == "edge_y":
        return 2 * ii, 2 * jj + 1
    if stage == "body":
        return 2 * ii + 1, 2 * jj + 1
    raise ValueError(stage)


def flatten_targets(d: np.ndarray, stage: str) -> np.ndarray:
    idx = {"edge_x": 0, "edge_y": 1, "body": 2}[stage]
    return d[:, idx].reshape(-1, 1).astype(np.float32)


def gather_features(cond: np.ndarray, stage: str, footprint: int) -> np.ndarray:
    # Reuse the reviewed finite-window feature layout after swapping in the
    # generalized offset set in this module.
    import train_finite_footprint_flow as old

    old_fp = old.footprint_offsets
    old_coords = old.detail_site_coords
    try:
        old.footprint_offsets = footprint_offsets
        old.detail_site_coords = detail_site_coords
        x, _ = gather_site_features_np(cond, stage, footprint)
        return x
    finally:
        old.footprint_offsets = old_fp
        old.detail_site_coords = old_coords


def yaml_text(cfg: CandidateConfig) -> str:
    lines = [
        f"candidate: {cfg.candidate}",
        "lambda: 0.022",
        "kappa_c: 0.2705",
        "kappa_f: 0.2705",
        "coarse_L: 16",
        "fine_L: 32",
    ]
    for key, value in asdict(cfg).items():
        lines.append(f"{key}: {value}")
    lines += [
        f"fine_reference: {FINE}",
        f"coarse_prior: {PRIOR}",
        f"kernel: {KERNEL}",
        "objective: conditional affine flow maximum likelihood plus DeltaS/logweight validation",
    ]
    return "\n".join(lines) + "\n"


def footprint_report(cfg: CandidateConfig, out: Path) -> dict[str, Any]:
    offsets = footprint_offsets(cfg.footprint)
    radius = max_radius(cfg.footprint)
    report = {
        "candidate": cfg.candidate,
        "footprint_size": cfg.footprint,
        "offset_count": len(offsets),
        "max_radius_fine_lattice_sites": radius,
        "fine_L": FINE_L,
        "conservative_no_wrap_rule": "max_radius < fine_L/2",
        "non_wrapping": bool(radius < FINE_L / 2),
        "offset_min": [min(i for i, _ in offsets), min(j for _, j in offsets)],
        "offset_max": [max(i for i, _ in offsets), max(j for _, j in offsets)],
        "padding": "explicit zero outside finite lattice during training/scoring; no circular feature gathering",
        "volume_independence": "same shared site-local MLP is applied at every detail site and can be instantiated at larger L",
    }
    write_json(out / "footprint_report.json", report)
    md = [
        f"# Footprint Report: {cfg.candidate}",
        "",
        f"- footprint size: `{cfg.footprint}`",
        f"- max fine-lattice radius: `{radius}`",
        f"- no-wrap rule: `radius < {FINE_L / 2:g}`",
        f"- non-wrapping on L32: `{report['non_wrapping']}`",
        "- feature access: explicit finite-window gather with zero outside-lattice context; no periodic feature wrap",
    ]
    (out / "footprint_report.md").write_text("\n".join(md) + "\n")
    print(json.dumps({"candidate": cfg.candidate, "footprint": cfg.footprint, "max_radius": radius, "non_wrapping_L32": report["non_wrapping"]}), flush=True)
    if not report["non_wrapping"]:
        raise RuntimeError(f"candidate {cfg.candidate} would wrap on L32: radius={radius}")
    return report


def load_data() -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, Any]]:
    fine, fine_meta = load_wolff(FINE, FINE_L, KAPPA)
    coarse, coarse_meta = load_wolff(PRIOR, COARSE_L, KAPPA)
    return fine, coarse, fine_meta, coarse_meta


def train_stage(stage: str, cond_train: np.ndarray, target_train: np.ndarray, cond_val: np.ndarray, target_val: np.ndarray, cfg: CandidateConfig, out: Path):
    import torch

    x_train = gather_features(cond_train, stage, cfg.footprint)
    y_train = flatten_targets(target_train, stage)
    x_val = gather_features(cond_val, stage, cfg.footprint)
    y_val = flatten_targets(target_val, stage)
    init_mean = float(np.mean(y_train)) if cfg.base_init else 0.0
    init_logstd = float(np.log(max(np.std(y_train, ddof=1), 1.0e-6))) if cfg.base_init else 0.0
    model = PatchAffineNF(x_train.shape[1], cfg.hidden_channels, cfg.conditioner_layers, init_mean, init_logstd)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    xt = torch.tensor(x_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.float32)
    xv = torch.tensor(x_val, dtype=torch.float32)
    yv = torch.tensor(y_val, dtype=torch.float32)
    rows = []
    best = float("inf")
    stage_dir = out / "checkpoints" / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, cfg.epochs + 1):
        perm = torch.randperm(xt.shape[0])
        losses = []
        model.train()
        for start in range(0, xt.shape[0], cfg.batch_size_sites):
            idx = perm[start : start + cfg.batch_size_sites]
            loss = -model.log_prob(yt[idx], xt[idx]).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            val = float((-model.log_prob(yv, xv).mean()).detach())
        row = {"stage": stage, "epoch": epoch, "train_nll_site": float(np.mean(losses)), "val_nll_site": val, "val_nll_total": val * COARSE_L * COARSE_L}
        rows.append(row)
        payload = {"stage": stage, "model_state": model.state_dict(), "config": asdict(cfg), "epoch": epoch, "val_nll_site": val, "target_init_mean": init_mean, "target_init_logstd": init_logstd}
        torch.save(payload, stage_dir / f"epoch{epoch:04d}.pt")
        if val < best:
            best = val
            torch.save(payload, stage_dir / "checkpoint_best.pt")
    best_payload = torch.load(stage_dir / "checkpoint_best.pt", map_location="cpu")
    model.load_state_dict(best_payload["model_state"])
    write_csv(out / f"{stage}_train_log.csv", rows)
    return model, rows


def sample_stage(model: PatchAffineNF, cond: np.ndarray, stage: str, footprint: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    import torch

    x = gather_features(cond, stage, footprint)
    with torch.no_grad():
        y, logq = model.sample(torch.tensor(x, dtype=torch.float32), seed)
    n = cond.shape[0]
    lc = cond.shape[-1] // 2
    return y.cpu().numpy().reshape(n, lc, lc).astype(np.float32), logq.cpu().numpy().reshape(n, lc, lc).sum(axis=(1, 2)).astype(np.float64)


def logq_stage(model: PatchAffineNF, cond: np.ndarray, d: np.ndarray, stage: str, footprint: int) -> np.ndarray:
    import torch

    x = gather_features(cond, stage, footprint)
    y = flatten_targets(d, stage)
    with torch.no_grad():
        lq = model.log_prob(torch.tensor(y, dtype=torch.float32), torch.tensor(x, dtype=torch.float32))
    n = cond.shape[0]
    lc = cond.shape[-1] // 2
    return lq.cpu().numpy().reshape(n, lc, lc).sum(axis=(1, 2)).astype(np.float64)


def sample_all(models: dict[str, PatchAffineNF], c: np.ndarray, cfg: CandidateConfig, seed: int) -> tuple[np.ndarray, np.ndarray]:
    d0, l0 = sample_stage(models["edge_x"], condition_grid(c, None, "edge_x"), "edge_x", cfg.footprint, seed)
    d1, l1 = sample_stage(models["edge_y"], condition_grid(c, None, "edge_y"), "edge_y", cfg.footprint, seed + 1)
    dprev = np.stack([d0, d1], axis=1)
    d2, l2 = sample_stage(models["body"], condition_grid(c, dprev, "body"), "body", cfg.footprint, seed + 2)
    return np.stack([d0, d1, d2], axis=1).astype(np.float32), l0 + l1 + l2


def logq_all(models: dict[str, PatchAffineNF], c: np.ndarray, d: np.ndarray, cfg: CandidateConfig) -> np.ndarray:
    return (
        logq_stage(models["edge_x"], condition_grid(c, None, "edge_x"), d, "edge_x", cfg.footprint)
        + logq_stage(models["edge_y"], condition_grid(c, None, "edge_y"), d, "edge_y", cfg.footprint)
        + logq_stage(models["body"], condition_grid(c, d[:, 0:2], "body"), d, "body", cfg.footprint)
    )


def xi_over_l(phi: np.ndarray) -> float:
    arr = phi.astype(np.float64)
    n, l, _ = arr.shape
    f0 = np.sum(arr, axis=(1, 2))
    fx = np.sum(arr * np.exp(2j * np.pi * np.arange(l)[None, :, None] / l), axis=(1, 2))
    fy = np.sum(arr * np.exp(2j * np.pi * np.arange(l)[None, None, :] / l), axis=(1, 2))
    s0 = float(np.mean(np.abs(f0) ** 2) / (l * l))
    sk = float(0.5 * np.mean(np.abs(fx) ** 2 + np.abs(fy) ** 2) / (l * l))
    if sk <= 0.0 or s0 <= sk:
        return float("nan")
    xi = math.sqrt(max(s0 / sk - 1.0, 0.0)) / (2.0 * math.sin(math.pi / l))
    return float(xi / l)


def validate_liftability(cfg: CandidateConfig) -> dict[str, Any]:
    import torch

    out = {}
    for lc in [32, 64]:
        lf = 2 * lc
        c = np.zeros((1, lc, lc), dtype=np.float32)
        d = np.zeros((1, 3, lc, lc), dtype=np.float32)
        rows = {}
        for stage, cond in {
            "edge_x": condition_grid(c, None, "edge_x"),
            "edge_y": condition_grid(c, None, "edge_y"),
            "body": condition_grid(c, d[:, 0:2], "body"),
        }.items():
            x = gather_features(cond, stage, cfg.footprint)
            model = PatchAffineNF(x.shape[1], cfg.hidden_channels, cfg.conditioner_layers)
            with torch.no_grad():
                shift, logscale = model.shift_logscale(torch.tensor(x[: min(len(x), 256)], dtype=torch.float32))
            rows[stage] = {"feature_shape": list(x.shape), "finite": bool(torch.isfinite(shift).all() and torch.isfinite(logscale).all())}
        out[f"L{lc}_to_L{lf}"] = {"passed": all(v["finite"] for v in rows.values()), "stages": rows}
    return out


def write_checksums(out: Path) -> None:
    rows = []
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.name != "sha256_checksums.txt":
            rows.append(f"{sha256(path)}  {path.relative_to(out)}")
    (out / "sha256_checksums.txt").write_text("\n".join(rows) + "\n")


def run(cfg: CandidateConfig, preflight_only: bool = False) -> dict[str, Any]:
    out = Path(cfg.output_dir) if cfg.output_dir else OUT_ROOT / cfg.candidate
    cfg.output_dir = str(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "plots").mkdir(exist_ok=True)
    (out / "config.yaml").write_text(yaml_text(cfg))
    fp = footprint_report(cfg, out)
    fine, coarse, fine_meta, coarse_meta = load_data()
    required = cfg.max_train + cfg.max_val
    split_report = {
        "available_reference_samples": int(len(fine)),
        "requested_max_train": int(cfg.max_train),
        "requested_max_val": int(cfg.max_val),
        "total_requested": int(required),
        "train_val_split_valid": bool(required <= len(fine)),
    }
    print(json.dumps(split_report), flush=True)
    if required > len(fine):
        raise RuntimeError(
            f"requested max_train + max_val = {required} exceeds available L32 fine/reference "
            f"configurations = {len(fine)}; reduce --max-train/--max-val or generate more L32 data"
        )
    kernel, kernel_json = load_kernel(KERNEL)
    lift = validate_liftability(cfg)
    (out / "liftability_preflight.md").write_text(
        f"# Liftability Preflight\n\n- L32->L64: `{lift['L32_to_L64']['passed']}`\n- L64->L128: `{lift['L64_to_L128']['passed']}`\n"
    )
    write_json(out / "liftability_preflight.json", lift)
    if preflight_only:
        return {"candidate": cfg.candidate, "footprint": fp, "liftability": lift, "split": split_report}

    psi = apply_kernel(fine, kernel)
    c_all, d_all = split_psi(psi)
    rng = np.random.default_rng(cfg.seed)
    idx = rng.permutation(len(fine))
    train_idx = idx[: cfg.max_train]
    val_idx = idx[cfg.max_train : cfg.max_train + cfg.max_val]
    c_train, d_train = c_all[train_idx], d_all[train_idx]
    c_val, d_val, phi_val = c_all[val_idx], d_all[val_idx], fine[val_idx]
    models = {}
    train_rows = []
    for stage, arrays in {
        "edge_x": (condition_grid(c_train, None, "edge_x"), d_train, condition_grid(c_val, None, "edge_x"), d_val),
        "edge_y": (condition_grid(c_train, None, "edge_y"), d_train, condition_grid(c_val, None, "edge_y"), d_val),
        "body": (condition_grid(c_train, d_train[:, 0:2], "body"), d_train, condition_grid(c_val, d_val[:, 0:2], "body"), d_val),
    }.items():
        model, rows = train_stage(stage, *arrays, cfg, out)
        models[stage] = model
        train_rows.extend(rows)
    write_csv(out / "training_curves.csv", train_rows)

    val_logq = logq_all(models, c_val, d_val, cfg)
    d_gen, logq_gen = sample_all(models, c_val, cfg, cfg.seed + 100)
    logq_regen = logq_all(models, c_val, d_gen, cfg)
    phi_gen, inv_gen = inverse_kernel(reconstruct(c_val, d_gen), kernel)
    c_reblocked, _ = split_psi(apply_kernel(phi_gen, kernel))
    draw = rng.integers(0, len(coarse), size=cfg.n_proposals)
    c_prop = coarse[draw]
    d_prop, logq_prop = sample_all(models, c_prop, cfg, cfg.seed + 200)
    phi_prop, inv_prop = inverse_kernel(reconstruct(c_prop, d_prop), kernel)
    c_reblocked_prop, _ = split_psi(apply_kernel(phi_prop, kernel))
    sf = action_total(phi_prop, ActionSpec("phi4_nn", LAMBDA, KAPPA))
    sc = action_total(c_prop, ActionSpec("phi4_nn", LAMBDA, KAPPA))
    delta_s = sf - sc
    logw = -delta_s - logq_prop
    validation = [
        {"case": "direct_L32_validation", **local_observables(phi_val), "xi_over_L": xi_over_l(phi_val)},
        {"case": "paired_generated_L32", **local_observables(phi_gen), "xi_over_L": xi_over_l(phi_gen)},
        {"case": "native_L16_prior_proposal_L32", **local_observables(phi_prop), "xi_over_L": xi_over_l(phi_prop)},
    ]
    write_csv(out / "validation_observables.csv", validation)
    log_rows = [
        {"quantity": "validation_true_logq", "mean": float(np.mean(val_logq)), "std": float(np.std(val_logq, ddof=1))},
        {"quantity": "validation_generated_logq", "mean": float(np.mean(logq_gen)), "std": float(np.std(logq_gen, ddof=1))},
        {"quantity": "proposal_deltaS", "mean": float(np.mean(delta_s)), "std": float(np.std(delta_s, ddof=1))},
        {"quantity": "proposal_logw", "mean": float(np.mean(logw)), "std": float(np.std(logw, ddof=1)), "ess_over_n": stable_ess(logw) / len(logw)},
    ]
    write_csv(out / "logweight_diagnostics.csv", log_rows)
    params = sum(int(np.prod(p.shape)) for m in models.values() for p in m.parameters())
    nan_inf = {
        "phi_generated_finite": bool(np.all(np.isfinite(phi_gen))),
        "phi_proposal_finite": bool(np.all(np.isfinite(phi_prop))),
        "logq_generated_finite": bool(np.all(np.isfinite(logq_gen))),
        "logw_finite": bool(np.all(np.isfinite(logw))),
    }
    summary = {
        "status": "completed",
        "candidate": cfg.candidate,
        "config": asdict(cfg),
        "lambda": LAMBDA,
        "kappa_c": KAPPA,
        "kappa_f": KAPPA,
        "coarse_L": COARSE_L,
        "fine_L": FINE_L,
        "kernel": kernel_json,
        "data": {"fine": fine_meta, "coarse_prior": coarse_meta},
        "footprint": fp,
        "trainable_parameters": params,
        "validation": validation,
        "validation_deltaS": {"mean": log_rows[2]["mean"], "std": log_rows[2]["std"]},
        "logweight": log_rows,
        "logweight_diagnostics_extended": logweight_diagnostics(logw, cfg.seed + 350),
        "roundtrip_max_error": float(np.max(np.abs(reconstruct(c_val, d_val) - psi[val_idx]))),
        "reblocking_max_error_paired_generated": float(np.max(np.abs(c_reblocked - c_val))),
        "reblocking_max_error_prior_proposal": float(np.max(np.abs(c_reblocked_prop - c_prop))),
        "nan_inf_check": nan_inf,
        "logq_transformed_density_consistency": {"generated_sample_vs_rescore_max_abs": float(np.max(np.abs(logq_gen - logq_regen)))},
        "shape_mask_compatibility": {"L16_to_L32": True, "liftability": lift},
        "inverse_kernel_validation": inv_gen,
        "inverse_kernel_proposal": inv_prop,
    }
    write_json(out / "validation_summary.json", summary)
    write_json(out / "summary.json", summary)
    report = [
        f"# Validation Report: {cfg.candidate}",
        "",
        f"- DeltaS mean/std: `{log_rows[2]['mean']:.6g}` / `{log_rows[2]['std']:.6g}`",
        f"- proposal logw std: `{log_rows[3]['std']:.6g}`",
        f"- action-density direct/generated/proposal: `{validation[0]['action_density']:.6g}` / `{validation[1]['action_density']:.6g}` / `{validation[2]['action_density']:.6g}`",
        f"- reblocking max error: `{summary['reblocking_max_error_prior_proposal']:.6g}`",
        f"- logq sample/rescore max error: `{summary['logq_transformed_density_consistency']['generated_sample_vs_rescore_max_abs']:.6g}`",
        f"- parameters: `{params}`",
    ]
    (out / "validation_report.md").write_text("\n".join(report) + "\n")
    if shutil.which("shasum"):
        write_checksums(out)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--footprint", type=int, required=True)
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--max-train", type=int, default=1024)
    ap.add_argument("--max-val", type=int, default=256)
    ap.add_argument("--n-proposals", type=int, default=512)
    ap.add_argument("--batch-size-sites", type=int, default=8192)
    ap.add_argument("--hidden-channels", type=int, default=64)
    ap.add_argument("--conditioner-layers", type=int, default=3)
    ap.add_argument("--learning-rate", type=float, default=2.0e-3)
    ap.add_argument("--seed", type=int, default=20260704)
    ap.add_argument("--no-base-init", action="store_true")
    ap.add_argument("--preflight-only", action="store_true")
    args = ap.parse_args()
    cfg = CandidateConfig(
        candidate=args.candidate,
        footprint=args.footprint,
        epochs=args.epochs,
        max_train=args.max_train,
        max_val=args.max_val,
        n_proposals=args.n_proposals,
        batch_size_sites=args.batch_size_sites,
        hidden_channels=args.hidden_channels,
        conditioner_layers=args.conditioner_layers,
        learning_rate=args.learning_rate,
        seed=args.seed,
        base_init=not args.no_base_init,
        output_dir=str(args.output_dir or OUT_ROOT / args.candidate),
    )
    result = run(cfg, preflight_only=args.preflight_only)
    print(json.dumps({"status": "ok", "candidate": args.candidate, "output_dir": cfg.output_dir, "result_keys": sorted(result)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
