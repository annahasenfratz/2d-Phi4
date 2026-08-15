#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PKG = PROJECT_ROOT / "perfect_blocking_upsampling"
SRC = PKG / "src"
EXP = PROJECT_ROOT / "ML_sampling_clean" / "experiments" / "decimated_conditional_fillin"
for p in [PROJECT_ROOT, SRC, EXP]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from perfect_blocking_upsampling.conv_pair import build_procedural_conv_flow  # noqa: E402
from perfect_blocking_upsampling.gathered_edge import build_gathered_edge_flow  # noqa: E402
from perfect_blocking_upsampling.kernels import apply_kernel, load_kernel  # noqa: E402
from run_decimated_conditional_fillin import Config as BaseConfig, build_model  # noqa: E402
from run_staged_decimated_conditional_fillin import (  # noqa: E402
    fit_generic_local_gaussian,
    log_jacobian,
    to_model_space,
)

LOG2PI = math.log(2.0 * math.pi)
LAM = 0.5
KAPPA = 0.3426
ETA = 0.25


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")


def json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(type(obj).__name__)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def torch_action_total(x, lam: float, kappa: float):
    phi2 = x * x
    local = ((1.0 - 2.0 * lam) * phi2 + lam * phi2 * phi2).sum(dim=(1, 2, 3))
    hop = (x * __import__("torch").roll(x, shifts=-1, dims=2)).sum(dim=(1, 2, 3))
    hop = hop + (x * __import__("torch").roll(x, shifts=-1, dims=3)).sum(dim=(1, 2, 3))
    return local - 2.0 * kappa * hop


def guard_stage_config(cfg: dict[str, Any], stage: str) -> None:
    if abs(float(cfg["lambda"]) - LAM) > 1.0e-12:
        raise RuntimeError(f"{stage}: lambda must be {LAM}, got {cfg['lambda']}")
    if abs(float(cfg["kappa"]) - KAPPA) > 1.0e-12:
        raise RuntimeError(f"{stage}: kappa must be {KAPPA}, got {cfg['kappa']}")
    if abs(float(cfg["eta"]) - ETA) > 1.0e-12:
        raise RuntimeError(f"{stage}: eta must be {ETA}, got {cfg['eta']}")
    paired = str(cfg["paired_data"])
    kernel = str(cfg["kernel"])
    bad = ["kappa0p25", "unknown", "lam0p022"]
    for b in bad:
        if b in paired or b in kernel:
            raise RuntimeError(f"{stage}: forbidden path fragment {b!r}")
    if "lam0p5_small3_8to16" not in paired:
        raise RuntimeError(f"{stage}: paired data must come from the lambda=0.5 workflow path")
    if "lam0p5" not in kernel or "kappa0p3426" not in kernel:
        raise RuntimeError(f"{stage}: kernel must be the lambda=0.5 kappa=0.3426 artifact")
    arch = cfg["architecture"]
    if arch.get("implementation_for_this_launcher") == "fresh_lambda0p5_supervised_stage_circular_cnn_gaussian_baseline":
        raise RuntimeError(f"{stage}: simplified smoke architecture selected")


def load_configs(config_dir: Path) -> dict[str, dict[str, Any]]:
    configs = {}
    for name in ["coarse_refine", "edge", "pair", "corner_body", "validation_smoke"]:
        path = config_dir / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(path)
        configs[name] = read_json(path)
        if name != "validation_smoke":
            guard_stage_config(configs[name], name)
    return configs


def load_data(configs: dict[str, dict[str, Any]], max_configs: int | None) -> dict[str, np.ndarray]:
    paired_path = resolve(configs["edge"]["paired_data"])
    with np.load(paired_path) as z:
        arrays = {k: z[k] for k in z.files}
    required = {
        "fine16": (5000, 16, 16),
        "blocked_channels": (5000, 4, 8, 8),
        "c00": (5000, 8, 8),
        "edge_x": (5000, 8, 8),
        "edge_y": (5000, 8, 8),
        "corner": (5000, 8, 8),
        "native_l8": (5000, 8, 8),
    }
    for key, shape in required.items():
        if key not in arrays or tuple(arrays[key].shape) != shape:
            raise RuntimeError(f"bad paired data array {key}: {arrays.get(key, np.empty(())).shape}")
        if not np.isfinite(arrays[key]).all():
            raise RuntimeError(f"nonfinite values in {key}")
    if max_configs is not None and max_configs > 0:
        keep = np.arange(min(max_configs, arrays["c00"].shape[0]))
        for key in required:
            arrays[key] = arrays[key][keep]
        n = len(keep)
        idx = np.arange(n)
        n_val = max(1, int(round(0.2 * n)))
        arrays["val_idx"] = idx[:n_val].astype(np.int64)
        arrays["train_idx"] = idx[n_val:].astype(np.int64)
    return arrays


def check_kernel(configs: dict[str, dict[str, Any]], arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    kernel_path = resolve(configs["edge"]["kernel"])
    if kernel_path.suffix == ".yaml":
        kernel_path = kernel_path.with_suffix(".json")
    if "small3_lam0p5_kappa0p3426_eta0p25" not in kernel_path.name:
        raise RuntimeError(f"wrong kernel artifact: {kernel_path}")
    spec, data = load_kernel(kernel_path)
    if abs(float(spec.eta) - ETA) > 1.0e-12:
        raise RuntimeError(f"kernel eta must be {ETA}, got {spec.eta}")
    psi = apply_kernel(arrays["fine16"][:2], spec)
    if psi.shape != (2, 16, 16) or not np.isfinite(psi).all():
        raise RuntimeError("kernel apply failed")
    return {"path": str(kernel_path), "eta": spec.eta, "orbits": data.get("orbits"), "sha256": sha256(kernel_path)}


def make_refine_model(cfg: dict[str, Any]):
    arch = cfg["architecture"]
    base_cfg = BaseConfig(
        lambda_=LAM,
        kappa=KAPPA,
        seed=int(cfg["training"]["seed"]),
        n_coupling_layers=int(arch["n_coupling_layers"]),
        conv_hidden_channels=int(arch["conv_hidden_channels"]),
        log_scale_bound=float(arch["log_scale_bound"]),
        flow_arch=str(arch["flow_arch"]),
    )
    l = int(cfg["coarse_L"])
    return build_model(l * l, l * l, (1, l, l), (1, l, l), base_cfg), base_cfg


def zero_cond(n: int, l: int):
    import torch

    return torch.zeros((n, l * l), dtype=torch.float32)


def train_coarse_refine(cfg: dict[str, Any], arrays: dict[str, np.ndarray], out: Path, resume: bool, epochs_override: int | None) -> dict[str, Any]:
    import torch

    stage_out = out / "coarse_refine_work"
    stage_out.mkdir(parents=True, exist_ok=True)
    ckpt_out = out / "checkpoints" / "coarse_refine.pt"
    if resume and ckpt_out.exists():
        return {"stage": "coarse_refine", "status": "skipped_resume", "checkpoint": str(ckpt_out)}
    tr = cfg["training"]
    epochs = int(epochs_override if epochs_override is not None else tr["pilot_epochs"])
    batch_size = int(tr["batch_size"])
    residual_penalty = float(tr.get("residual_penalty", 0.05))
    torch.manual_seed(int(tr["seed"]))
    model, model_cfg = make_refine_model(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=float(tr["learning_rate"]))
    c = arrays["c00"].astype(np.float32)
    train_idx = arrays["train_idx"].astype(np.int64)
    val_idx = arrays["val_idx"].astype(np.int64)
    train_x = torch.tensor(c[train_idx, None].reshape(len(train_idx), -1), dtype=torch.float32)
    val_x = torch.tensor(c[val_idx, None].reshape(len(val_idx), -1), dtype=torch.float32)
    rows = []
    best = {"val_loss": float("inf"), "state": None, "epoch": 0}
    l = c.shape[1]
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(train_x.shape[0])
        losses = []
        model.train()
        for start in range(0, train_x.shape[0], batch_size):
            b = perm[start : start + batch_size]
            target = train_x[b]
            z, logdet_inv = model.inverse(target, zero_cond(target.shape[0], l))
            u_img = z.reshape(z.shape[0], 1, l, l)
            nll = torch_action_total(u_img, LAM, KAPPA) - logdet_inv
            penalty = ((target - z) ** 2).mean(dim=1) * (l * l)
            loss = (nll + residual_penalty * penalty).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            z_val, logdet_inv_val = model.inverse(val_x, zero_cond(val_x.shape[0], l))
            val_nll = torch_action_total(z_val.reshape(z_val.shape[0], 1, l, l), LAM, KAPPA) - logdet_inv_val
            val_penalty = ((val_x - z_val) ** 2).mean(dim=1) * (l * l)
            val_loss = float((val_nll + residual_penalty * val_penalty).mean().detach())
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": val_loss}
        rows.append(row)
        write_csv(out / "logs" / "coarse_refine_train_log.csv", rows)
        torch.save({"model_state": model.state_dict(), "optimizer_state": opt.state_dict(), "config": asdict(model_cfg), "epoch": epoch, "val_loss": val_loss}, stage_out / f"epoch{epoch:04d}.pt")
        if val_loss < best["val_loss"]:
            best = {"val_loss": val_loss, "epoch": epoch, "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
        print(f"coarse_refine epoch {epoch}/{epochs}: val_loss={val_loss:.6g}", flush=True)
    assert best["state"] is not None
    ckpt = {
        "model_state": best["state"],
        "config": asdict(model_cfg) | {
            "lambda_": LAM,
            "kappa": KAPPA,
            "eta": ETA,
            "stage": "coarse_refine",
            "source": "faithful_lambda0p5_transported_detail_driver",
        },
        "epoch": best["epoch"],
        "val_loss": best["val_loss"],
        "selection": "best_val_density_plus_residual_penalty",
    }
    ckpt_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, ckpt_out)
    return {"stage": "coarse_refine", "status": "trained", "checkpoint": str(ckpt_out), "best_epoch": best["epoch"], "best_val_loss": best["val_loss"], "rows": rows}


def build_detail_model(stage: str, cfg: dict[str, Any]):
    arch = cfg["architecture"]
    if arch["flow_arch"] == "gathered_edge":
        return build_gathered_edge_flow(
            cond_channels=int(arch["cond_channels"]),
            lattice_size=int(cfg["coarse_L"]),
            radius=int(arch["gather_radius"]),
            stencil=str(arch["gather_stencil"]),
            hidden_width=int(arch["gather_hidden_width"]),
            hidden_layers=int(arch["gather_hidden_layers"]),
            log_scale_bound=float(arch["log_scale_bound"]),
        )
    if arch["flow_arch"] == "procedural_conv":
        return build_procedural_conv_flow(
            cond_channels=int(arch["cond_channels"]),
            target_channels=int(arch["target_channels"]),
            lattice_size=int(cfg["coarse_L"]),
            n_coupling_layers=int(arch["n_coupling_layers"]),
            conv_hidden_channels=int(arch["conv_hidden_channels"]),
            log_scale_bound=float(arch["log_scale_bound"]),
        )
    raise RuntimeError(f"unsupported detail architecture {arch['flow_arch']}")


def log_base_torch(z):
    return -0.5 * (z * z + LOG2PI).sum(dim=1)


def detail_arrays(stage: str, arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    c = arrays["c00"].astype(np.float32)
    ex = arrays["edge_x"].astype(np.float32)
    ey = arrays["edge_y"].astype(np.float32)
    co = arrays["corner"].astype(np.float32)
    if stage == "edge":
        return c[:, None], ex[:, None]
    if stage == "pair":
        return np.stack([c, ex], axis=1), ey[:, None]
    if stage == "corner_body":
        return np.stack([c, ex, ey], axis=1), co[:, None]
    raise RuntimeError(stage)


def train_detail_stage(stage: str, cfg: dict[str, Any], arrays: dict[str, np.ndarray], out: Path, resume: bool, epochs_override: int | None) -> dict[str, Any]:
    import torch

    ckpt_name = "corner" if stage == "corner_body" else stage
    ckpt_out = out / "checkpoints" / f"{ckpt_name}.pt"
    coeff_out = out / "checkpoints" / ckpt_name / "local_gaussian_coefficients.npz"
    if resume and ckpt_out.exists() and coeff_out.exists():
        return {"stage": stage, "status": "skipped_resume", "checkpoint": str(ckpt_out)}
    tr = cfg["training"]
    epochs = int(epochs_override if epochs_override is not None else tr.get("epochs", tr.get("pilot_epochs", 8)))
    batch_size = int(tr["batch_size"])
    torch.manual_seed(int(tr["seed"]))
    cond, target = detail_arrays(stage, arrays)
    train_idx = arrays["train_idx"].astype(np.int64)
    val_idx = arrays["val_idx"].astype(np.int64)
    cond_train, target_train = cond[train_idx], target[train_idx]
    cond_val, target_val = cond[val_idx], target[val_idx]
    lg = fit_generic_local_gaussian(cond_train, target_train, float(tr.get("local_gaussian_sigma_floor", 1.0e-4)))
    target_train_u = to_model_space(target_train, cond_train, lg)
    target_val_u = to_model_space(target_val, cond_val, lg)
    train_jac = torch.tensor(log_jacobian(cond_train, lg), dtype=torch.float32)
    val_jac = torch.tensor(log_jacobian(cond_val, lg), dtype=torch.float32)
    train_c = torch.tensor(cond_train.reshape(cond_train.shape[0], -1), dtype=torch.float32)
    train_d = torch.tensor(target_train_u.reshape(target_train_u.shape[0], -1), dtype=torch.float32)
    val_c = torch.tensor(cond_val.reshape(cond_val.shape[0], -1), dtype=torch.float32)
    val_d = torch.tensor(target_val_u.reshape(target_val_u.shape[0], -1), dtype=torch.float32)
    model = build_detail_model(stage, cfg)
    opt = torch.optim.Adam(model.parameters(), lr=float(tr["learning_rate"]))
    rows = []
    best = {"val_loss": float("inf"), "state": None, "epoch": 0}
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
        write_csv(out / "logs" / f"{stage}_train_log.csv", rows)
        torch.save(
            {"model_state": model.state_dict(), "optimizer_state": opt.state_dict(), "config": cfg["architecture"] | {"lambda_": LAM, "kappa": KAPPA, "eta": ETA}, "stage": ckpt_name, "epoch": epoch, "val_loss": val_loss},
            out / "checkpoints" / f"{ckpt_name}_epoch{epoch:04d}.pt",
        )
        if val_loss < best["val_loss"]:
            best = {"val_loss": val_loss, "epoch": epoch, "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
        print(f"{stage} epoch {epoch}/{epochs}: val_loss={val_loss:.6g}", flush=True)
    assert best["state"] is not None
    coeff_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(coeff_out, coeffs=lg["coeffs"], sigma=lg["sigma"], ridge=np.asarray(lg["ridge"]))
    dep = model.dependency_report() if hasattr(model, "dependency_report") else {}
    dummy = None
    if cfg["architecture"]["flow_arch"] == "procedural_conv":
        dummy_model = build_procedural_conv_flow(
            cond_channels=int(cfg["architecture"]["cond_channels"]),
            target_channels=int(cfg["architecture"]["target_channels"]),
            lattice_size=16,
            n_coupling_layers=int(cfg["architecture"]["n_coupling_layers"]),
            conv_hidden_channels=int(cfg["architecture"]["conv_hidden_channels"]),
            log_scale_bound=float(cfg["architecture"]["log_scale_bound"]),
        )
        dummy = dummy_model.dependency_report()
    ckpt = {
        "model_state": best["state"],
        "config": cfg["architecture"] | {
            "lambda_": LAM,
            "kappa": KAPPA,
            "eta": ETA,
            "lattice_size": int(cfg["coarse_L"]),
            "stage": ckpt_name,
            "source": "faithful_lambda0p5_transported_detail_driver",
        },
        "stage": ckpt_name,
        "selection": "best_val_nll",
        "epoch": best["epoch"],
        "val_loss": best["val_loss"],
        "dependency_report": dep,
    }
    if dummy is not None:
        ckpt["dummy_larger_volume_dependency_report"] = dummy
    torch.save(ckpt, ckpt_out)
    return {"stage": stage, "status": "trained", "checkpoint": str(ckpt_out), "local_gaussian": str(coeff_out), "best_epoch": best["epoch"], "best_val_loss": best["val_loss"], "dependency_report": dep}


def write_bundle_config(out: Path, configs: dict[str, dict[str, Any]]) -> Path:
    cfg_path = out / "bundle_config.yaml"
    ckpt_dir = (out / "checkpoints").resolve()
    kernel = resolve(configs["edge"]["kernel"])
    if kernel.suffix == ".yaml":
        kernel = kernel.with_suffix(".json")
    coarse = PROJECT_ROOT / "phi4_phase-diagram/ensembles/lam0p5_kappa0p3426_L8_embedded_wolff_sign_cluster_plus_radial_heatbath_N5000/configs.npz"
    fine = PROJECT_ROOT / "phi4_phase-diagram/ensembles/lam0p5_kappa0p3426_L16_embedded_wolff_sign_cluster_plus_radial_heatbath_N5000/configs.npz"
    text = f"""run_name: lam0p5_kappa0p3426_L8_to_L16_small3_faithful_transported_detail
random_seed: 20260721
output_dir: outputs/lam0p5_small3_8to16/full_training/validation_placeholder
checkpoints:
  frozen_dir: {ckpt_dir}
action:
  coarse:
    type: phi4_nn
    lambda: 0.5
    kappa: 0.3426
    kappa_diag: 0.0
  fine:
    type: phi4_nn
    lambda: 0.5
    kappa: 0.3426
    kappa_diag: 0.0
lattice:
  coarse_L: 8
  fine_L: 16
  scale_factor: 2
kernel:
  path: {kernel}
  eta: 0.25
  scale_factor: 2
  normalize: true
data:
  coarse_ensemble: {coarse}
  fine_reference: {fine}
model:
  coarse_refine: true
  missing_flow_type: affine
  stage_factorization: edge_pair_corner
evaluation:
  validation_mode: native_L8_deployment_full_coarse_update
  measurement_mode: end_of_sweep
  n_proposals: 512
  ar_chains: 2
  ar_proposals_per_chain: 100
"""
    cfg_path.write_text(text, encoding="utf-8")
    return cfg_path


def write_manifest(out: Path, configs: dict[str, dict[str, Any]], kernel_info: dict[str, Any], summary: dict[str, Any]) -> None:
    manifest = {
        "status": "complete",
        "lambda": LAM,
        "kappa": KAPPA,
        "eta": ETA,
        "note": "Faithful transported-detail lambda=0.5 training driver output. Smoke runs are not sampler-quality.",
        "kernel": kernel_info,
        "configs": configs,
        "summary": summary,
    }
    write_json(out / "training_manifest.json", manifest)
    with (out / "sha256_checksums.txt").open("w", encoding="utf-8") as f:
        for path in sorted((out / "checkpoints").rglob("*")):
            if path.is_file():
                f.write(f"{sha256(path)}  {path.relative_to(out)}\n")
        for name in ["bundle_config.yaml", "training_manifest.json", "training_summary.json"]:
            path = out / name
            if path.exists():
                f.write(f"{sha256(path)}  {name}\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-configs", type=int, default=0)
    ap.add_argument("--epochs-override", type=int, default=0)
    args = ap.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not (args.overwrite or args.resume):
        raise RuntimeError(f"output directory exists and is nonempty: {args.output_dir}")
    if args.overwrite and args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for sub in ["checkpoints", "logs", "reports"]:
        (args.output_dir / sub).mkdir(exist_ok=True)
    configs = load_configs(args.config_dir)
    arrays = load_data(configs, args.max_configs if args.max_configs > 0 else None)
    kernel_info = check_kernel(configs, arrays)
    config_snapshot = {k: v for k, v in configs.items()}
    write_json(args.output_dir / "config_snapshot.json", config_snapshot)
    t0 = time.time()
    epochs_override = args.epochs_override if args.epochs_override > 0 else None
    stage_results = []
    stage_results.append(train_coarse_refine(configs["coarse_refine"], arrays, args.output_dir, args.resume, epochs_override))
    stage_results.append(train_detail_stage("edge", configs["edge"], arrays, args.output_dir, args.resume, epochs_override))
    stage_results.append(train_detail_stage("pair", configs["pair"], arrays, args.output_dir, args.resume, epochs_override))
    stage_results.append(train_detail_stage("corner_body", configs["corner_body"], arrays, args.output_dir, args.resume, epochs_override))
    bundle_cfg = write_bundle_config(args.output_dir, configs)
    summary = {
        "status": "completed",
        "scope": "full_driver" if args.max_configs <= 0 and epochs_override is None else "tiny_or_limited_run",
        "wall_time_sec": time.time() - t0,
        "max_configs": args.max_configs,
        "epochs_override": args.epochs_override,
        "stage_results": stage_results,
        "bundle_config": str(bundle_cfg),
        "full_training_launched_by_this_run": args.max_configs <= 0 and epochs_override is None,
    }
    write_json(args.output_dir / "training_summary.json", summary)
    write_manifest(args.output_dir, configs, kernel_info, summary)
    print(json.dumps(summary, indent=2, default=json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
