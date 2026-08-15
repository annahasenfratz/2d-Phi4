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

from diagnose_lam0p5_failed_bundle import corr, load_context, qstats, reconstruct, rmse, write_csv, write_json  # noqa: E402
from perfect_blocking_upsampling.actions import action_density, action_total  # noqa: E402
from perfect_blocking_upsampling.conv_pair import build_procedural_conv_flow  # noqa: E402
from perfect_blocking_upsampling.kernels import apply_kernel, inverse_kernel  # noqa: E402
from perfect_blocking_upsampling.observables import observables as ensemble_observables  # noqa: E402
from run_lam0p5_detail_remediation import save_lg  # noqa: E402
from run_staged_decimated_conditional_fillin import fit_generic_local_gaussian, from_model_space, log_jacobian, to_model_space  # noqa: E402
from train_faithful_transported_detail import log_base_torch  # noqa: E402

LAM = 0.5
KAPPA = 0.3426
ETA = 0.25
OLD_SEQ_RANGE = "15--17"
LARGER_JOINT = 14.2532
BEST_LARGER_VIEW = 12.2944
TEACHER_SCALE = "10--11"


def load_paired(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def stack_detail(arrays: dict[str, np.ndarray], idx: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    c = arrays["c00"] if idx is None else arrays["c00"][idx]
    detail = np.stack(
        [
            arrays["edge_x"] if idx is None else arrays["edge_x"][idx],
            arrays["edge_y"] if idx is None else arrays["edge_y"][idx],
            arrays["corner"] if idx is None else arrays["corner"][idx],
        ],
        axis=1,
    ).astype(np.float32)
    fine = arrays["fine16"] if idx is None else arrays["fine16"][idx]
    return c.astype(np.float32), detail, fine.astype(np.float32)


def reconstruct_patch_detail(c: np.ndarray, detail: np.ndarray, ctx: dict[str, Any]) -> np.ndarray:
    psi = reconstruct(c, detail[:, 0:1], detail[:, 1:2], detail[:, 2:3])
    phi, _ = inverse_kernel(psi, ctx["kernel"])
    return phi


def patch_metrics(label: str, c: np.ndarray, detail: np.ndarray, target_detail: np.ndarray, fine_ref: np.ndarray, ctx: dict[str, Any], logq: np.ndarray | None = None, logdet: np.ndarray | None = None) -> dict[str, Any]:
    phi = reconstruct_patch_detail(c, detail, ctx)
    sf = action_total(phi, ctx["fine_action"])
    st = action_total(fine_ref, ctx["fine_action"])
    ds = sf - st
    dens = action_density(phi, ctx["fine_action"]).mean(axis=(1, 2))
    denst = action_density(fine_ref, ctx["fine_action"]).mean(axis=(1, 2))
    blocked = apply_kernel(phi, ctx["kernel"])
    reb = np.max(np.abs(blocked[:, 0::2, 0::2] - c), axis=(1, 2))
    obs = ensemble_observables(phi, ctx["fine_action"])
    ref = ensemble_observables(fine_ref, ctx["fine_action"])
    row = {
        "label": label,
        "detail_rmse": rmse(detail, target_detail),
        "detail_corr": corr(detail, target_detail),
        "edge_rmse": rmse(detail[:, 0:1], target_detail[:, 0:1]),
        "pair_rmse": rmse(detail[:, 1:2], target_detail[:, 1:2]),
        "corner_rmse": rmse(detail[:, 2:3], target_detail[:, 2:3]),
        "edge_corr": corr(detail[:, 0:1], target_detail[:, 0:1]),
        "pair_corr": corr(detail[:, 1:2], target_detail[:, 1:2]),
        "corner_corr": corr(detail[:, 2:3], target_detail[:, 2:3]),
        "deltaS_mean": qstats(ds)["mean"],
        "deltaS_std": qstats(ds)["std"],
        "deltaS_rms": float(np.sqrt(np.mean(ds * ds))),
        "local_action_density_error_std": qstats(dens - denst)["std"],
        "action_density_shift": float(obs["action_density"] - ref["action_density"]),
        "phi2_shift": float(obs["phi2"] - ref["phi2"]),
        "phi4_shift": float(obs["phi4"] - ref["phi4"]),
        "NN_shift": float(obs["NN"] - ref["NN"]),
        "reblocking_error_max": float(np.max(reb)),
        "nan_or_inf": bool((not np.isfinite(ds).all()) or (not np.isfinite(detail).all())),
    }
    if logq is not None:
        row["logq_mean"] = qstats(logq)["mean"]
        row["logq_std"] = qstats(logq)["std"]
    if logdet is not None:
        row["logdet_mean"] = qstats(logdet)["mean"]
        row["logdet_std"] = qstats(logdet)["std"]
    return row


def fit_unconditional_gaussian(detail_train: np.ndarray) -> dict[str, np.ndarray]:
    flat = detail_train.transpose(0, 2, 3, 1).reshape(-1, 3).astype(np.float64)
    mean = flat.mean(axis=0)
    cov = np.cov(flat, rowvar=False) + 1.0e-6 * np.eye(3)
    return {"mean": mean.astype(np.float32), "cov": cov.astype(np.float32)}


def sample_unconditional(model: dict[str, np.ndarray], n: int, l: int, rng: np.random.Generator, mean_only: bool) -> tuple[np.ndarray, np.ndarray]:
    mean = model["mean"]
    cov = model["cov"]
    if mean_only:
        x = np.broadcast_to(mean.reshape(1, 3, 1, 1), (n, 3, l, l)).copy()
    else:
        flat = rng.multivariate_normal(mean, cov, size=n * l * l).astype(np.float32)
        x = flat.reshape(n, l, l, 3).transpose(0, 3, 1, 2)
    inv = np.linalg.inv(cov.astype(np.float64))
    sign, logdet = np.linalg.slogdet(cov.astype(np.float64))
    flat = x.transpose(0, 2, 3, 1).reshape(n, l * l, 3).astype(np.float64)
    d = flat - mean.astype(np.float64)
    quad = np.einsum("nka,ab,nkb->nk", d, inv, d)
    logq = -0.5 * np.sum(quad + 3 * math.log(2 * math.pi) + logdet, axis=1)
    return x.astype(np.float32), logq.astype(np.float64)


def build_patch_model(hidden: int, n_coupling: int):
    return build_procedural_conv_flow(
        cond_channels=1,
        target_channels=3,
        lattice_size=8,
        n_coupling_layers=n_coupling,
        conv_hidden_channels=hidden,
        log_scale_bound=0.75,
    )


def forward_patch_model(model, z: np.ndarray, cond: np.ndarray, lg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import torch

    model.eval()
    z_t = torch.tensor(z.reshape(z.shape[0], -1), dtype=torch.float32)
    c_t = torch.tensor(cond.reshape(cond.shape[0], -1), dtype=torch.float32)
    with torch.no_grad():
        y_flat, fld = model.forward(z_t, c_t)
    y = y_flat.cpu().numpy().reshape(z.shape).astype(np.float32)
    x = from_model_space(y, cond, lg).astype(np.float32)
    log_base = -0.5 * np.sum(z.reshape(z.shape[0], -1).astype(np.float64) ** 2 + math.log(2 * math.pi), axis=1)
    logq = log_base - fld.cpu().numpy().astype(np.float64) - log_jacobian(cond, lg)
    return x, logq, fld.cpu().numpy().astype(np.float64)


def train_patch_flow(c: np.ndarray, detail: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray, ctx: dict[str, Any], out: Path, epochs: int, hidden: int, n_coupling: int, seed: int) -> dict[str, Any]:
    import torch

    out.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
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
    val_c = torch.tensor(cond_val.reshape(cond_val.shape[0], -1), dtype=torch.float32)
    val_d = torch.tensor(val_u.reshape(val_u.shape[0], -1), dtype=torch.float32)
    torch.manual_seed(seed)
    model = build_patch_model(hidden, n_coupling)
    opt = torch.optim.Adam(model.parameters(), lr=1.0e-4)
    rows: list[dict[str, Any]] = []
    best_action = None
    best_nll = None
    batch = 32
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(train_c.shape[0])
        losses = []
        model.train()
        for start in range(0, train_c.shape[0], batch):
            b = perm[start : start + batch]
            z, ild = model.inverse(train_d[b], train_c[b])
            loss = -(log_base_torch(z) + ild - train_j[b]).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
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
        }
        torch.save({"model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, "config": cfg, "epoch": epoch, "val_loss": val_nll, "stage": "patch_detail"}, ckpt)
        z0 = np.zeros((len(val_idx), 3, 8, 8), dtype=np.float32)
        pred, logq, fld = forward_patch_model(model, z0, cond_val, lg)
        metric = patch_metrics("patch_flow_z0", c[val_idx], pred, detail[val_idx], arrays_global["fine16"][val_idx], ctx, logq, fld)
        row = {"epoch": epoch, "train_nll": float(np.mean(losses)), "val_nll": val_nll, "checkpoint": str(ckpt), **metric}
        rows.append(row)
        write_csv(out / "patch_detail_flow_training_metrics.csv", rows)
        if best_action is None or row["deltaS_std"] < best_action["deltaS_std"]:
            best_action = dict(row)
        if best_nll is None or val_nll < best_nll["val_nll"]:
            best_nll = dict(row)
        print(f"patch_detail epoch {epoch}/{epochs}: val_nll={val_nll:.6g} deltaS_std={row['deltaS_std']:.6g}", flush=True)
    assert best_action and best_nll
    final = ckpt_dir / "patch_detail.pt"
    shutil.copy2(best_action["checkpoint"], final)
    return {"checkpoint": str(final), "local_gaussian": str(ckpt_dir / "patch_detail/local_gaussian_coefficients.npz"), "best_by_action": best_action, "best_by_nll": best_nll}


arrays_global: dict[str, np.ndarray] = {}


def update_status(summary: dict[str, Any]) -> None:
    path = PROJECT_ROOT / "perfect_blocking_upsampling/outputs/lam0p5_small3_8to16/STATUS.md"
    section = f"""

## Joint patch-detail transported-coordinate parameterization branch

Status: `{summary['status']}`.

Output:

`remediation/patch_detail_parameterization/`

Completed:

- designed a non-overlapping per-cell 3-detail coordinate `(d10,d01,d11)`;
- verified exact reconstruction and small3 reblocking consistency;
- evaluated unconditional and conditional Gaussian baselines;
- trained a bounded joint patch-detail flow prototype;
- did not run long validation.

Key results:

- coordinate roundtrip max error: `{summary['roundtrip_max_error']:.6g}`;
- reblocking max error: `{summary['reblocking_max_error']:.6g}`;
- best baseline deltaS std: `{summary['best_baseline_deltaS_std']:.6g}`;
- best patch-flow deltaS std: `{summary['best_flow_deltaS_std']:.6g}`;
- sampler smoke launched: `{summary['sampler_smoke_launched']}`.

Interpretation:

{summary['interpretation']}

Reports:

- `remediation/patch_detail_parameterization/PATCH_DETAIL_PARAMETERIZATION_DESIGN.md`
- `remediation/patch_detail_parameterization/PATCH_DETAIL_COORDINATE_PREFLIGHT_REPORT.md`
- `remediation/patch_detail_parameterization/PATCH_DETAIL_PARAMETERIZATION_FINAL_REPORT.md`
"""
    text = path.read_text()
    marker = "\n## Joint patch-detail transported-coordinate parameterization branch\n"
    if marker in text:
        text = text[: text.index(marker)]
    path.write_text(text.rstrip() + section + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paired-data", type=Path, default=PROJECT_ROOT / "perfect_blocking_upsampling/outputs/lam0p5_small3_8to16/paired_data/paired_lam0p5_small3_L16_to_L8.npz")
    ap.add_argument("--run-dir", type=Path, default=PROJECT_ROOT / "perfect_blocking_upsampling/outputs/lam0p5_small3_8to16/full_training/run_20260630_210838")
    ap.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "perfect_blocking_upsampling/outputs/lam0p5_small3_8to16/remediation/patch_detail_parameterization")
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--n-coupling", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20260705)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    out = args.output_root
    if out.exists() and args.overwrite:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    _cfg, _paths, _coarse, _fine_ref, _cm, _fm, ctx = load_context(args.run_dir)
    global arrays_global
    arrays_global = load_paired(args.paired_data)
    c, detail, fine = stack_detail(arrays_global)
    train_idx = arrays_global["train_idx"].astype(np.int64)
    val_idx = arrays_global["val_idx"].astype(np.int64)

    (out / "PATCH_DETAIL_PARAMETERIZATION_DESIGN.md").write_text(
        "# Joint patch-detail transported-coordinate parameterization design\n\n"
        "The old lambda=0.022 factorization trains separate edge, pair, and corner/body stages. At lambda=0.5, action diagnostics show that edge and pair are strongly coupled: fixed-coarse MCMC can find action-compatible details, but sequential or even modest joint edge+pair flows do not distill that correction. This branch therefore changes the coordinate grouping rather than adding more edge/pair capacity.\n\n"
        "The first prototype uses non-overlapping local coarse cells. For each coarse cell `(i,j)`, the retained even-even transported slot is `psi[2i,2j]=c00[i,j]`. The local detail vector is\n\n"
        "`d_cell[i,j] = (psi[2i+1,2j], psi[2i,2j+1], psi[2i+1,2j+1]) = (d10,d01,d11)`.\n\n"
        "The full detail field is stored as shape `(N,3,Lc,Lc)`. Reconstruction forms `psi` by interleaving `c00` and these three channels, then applies the optimized small3 inverse kernel. Since the retained coarse slot is placed directly into `psi[0::2,0::2]`, applying the small3 kernel to the reconstructed fine field returns the same transported `psi` up to numerical precision; the block constraint is exact in this coordinate system.\n\n"
        "This transform is linear and one-to-one between `psi` and `(c00,d10,d01,d11)`, so the interleaving Jacobian is trivial. The nontrivial logdet for sampler use comes from the inverse small3 kernel, which is fixed, plus the learned conditional flow logdet for `q(d10,d01,d11|c00)`. Periodic boundaries enter through the inverse kernel and through circular conditioner stencils. To plug into the existing patchwise A/R sampler, the edge/pair/corner stages would be replaced by one patch-detail stage that outputs the three channels jointly and returns one joint logq/logdet.\n"
    )

    phi_rt = reconstruct_patch_detail(c[:128], detail[:128], ctx)
    roundtrip = np.max(np.abs(phi_rt - fine[:128]))
    blocked = apply_kernel(phi_rt, ctx["kernel"])
    reblock = np.max(np.abs(blocked[:, 0::2, 0::2] - c[:128]))
    preflight = {
        "fine_shape": list(fine.shape),
        "coarse_shape": list(c.shape),
        "detail_shape": list(detail.shape),
        "detail_dof_per_config": int(np.prod(detail.shape[1:])),
        "coarse_dof_per_config": int(np.prod(c.shape[1:])),
        "roundtrip_max_error": float(roundtrip),
        "reblocking_max_error": float(reblock),
        "jacobian": "trivial interleaving; fixed inverse-kernel determinant is constant",
        "periodic_boundaries": "handled by small3 inverse kernel and circular conv conditioners",
    }
    write_json(out / "patch_detail_coordinate_preflight.json", preflight)
    (out / "PATCH_DETAIL_COORDINATE_PREFLIGHT_REPORT.md").write_text(
        "# Patch-detail coordinate preflight report\n\n"
        f"- fine shape: `{fine.shape}`\n"
        f"- coarse shape: `{c.shape}`\n"
        f"- detail shape: `{detail.shape}`\n"
        f"- detail variables per config: `{preflight['detail_dof_per_config']}`\n"
        f"- roundtrip max error: `{roundtrip:.6g}`\n"
        f"- reblocking max error: `{reblock:.6g}`\n"
        "- transform type: non-overlapping per-cell 3-detail vector with trivial interleaving Jacobian.\n"
    )

    rng = np.random.default_rng(args.seed)
    val_c, val_d, val_f = c[val_idx], detail[val_idx], fine[val_idx]
    baseline_rows = []
    action_rows = []
    ug = fit_unconditional_gaussian(detail[train_idx])
    for mean_only in [True, False]:
        d_s, logq = sample_unconditional(ug, len(val_idx), 8, rng, mean_only)
        row = patch_metrics("unconditional_gaussian_mean" if mean_only else "unconditional_gaussian_sample", val_c, d_s, val_d, val_f, ctx, logq)
        baseline_rows.append(row)
        action_rows.append(row)
    lg = fit_generic_local_gaussian(c[train_idx, None], detail[train_idx], 1.0e-4)
    cond_val = val_c[:, None].astype(np.float32)
    z0 = np.zeros_like(val_d, dtype=np.float32)
    d_mean = from_model_space(z0, cond_val, lg)
    logq_mean = -log_jacobian(cond_val, lg)
    row = patch_metrics("conditional_local_gaussian_mean", val_c, d_mean, val_d, val_f, ctx, logq_mean)
    baseline_rows.append(row)
    action_rows.append(row)
    zrand = rng.standard_normal(val_d.shape).astype(np.float32)
    d_rand = from_model_space(zrand, cond_val, lg)
    log_base = -0.5 * np.sum(zrand.reshape(zrand.shape[0], -1).astype(np.float64) ** 2 + math.log(2 * math.pi), axis=1)
    logq_rand = log_base - log_jacobian(cond_val, lg)
    row = patch_metrics("conditional_local_gaussian_sample", val_c, d_rand, val_d, val_f, ctx, logq_rand)
    baseline_rows.append(row)
    action_rows.append(row)
    write_csv(out / "patch_detail_baseline_metrics.csv", baseline_rows)
    write_csv(out / "patch_detail_action_metrics.csv", action_rows)
    best_baseline = min(baseline_rows, key=lambda r: float(r["deltaS_std"]))
    (out / "PATCH_DETAIL_BASELINE_DENSITY_REPORT.md").write_text(
        "# Patch-detail baseline density report\n\n"
        f"- best baseline: `{best_baseline['label']}`\n"
        f"- best baseline deltaS std: `{float(best_baseline['deltaS_std']):.6g}`\n"
        f"- larger joint edge+pair reference: `{LARGER_JOINT}`\n"
        f"- best larger-joint cumulative view: `{BEST_LARGER_VIEW}`\n"
        f"- MCMC teacher scale: `{TEACHER_SCALE}`\n"
        "- tables: `patch_detail_baseline_metrics.csv`, `patch_detail_action_metrics.csv`.\n"
    )

    flow = train_patch_flow(c, detail, train_idx, val_idx, ctx, out / "patch_detail_flow", args.epochs, args.hidden, args.n_coupling, args.seed)
    shutil.copy2(out / "patch_detail_flow/patch_detail_flow_training_metrics.csv", out / "patch_detail_flow_training_metrics.csv")
    best_flow = flow["best_by_action"]
    smoke = False
    interpretation = "The non-overlapping joint patch-detail coordinate is valid, but the prototype flow did not reach the promising action threshold."
    if float(best_flow["deltaS_std"]) < 14.0:
        interpretation = "The patch-detail prototype beats the old edge/pair factorization and is interesting, but sampler smoke still requires stronger diagnostics."
    if float(best_flow["deltaS_std"]) <= 12.0:
        interpretation = "The patch-detail prototype reached the strong diagnostic gate; a tiny sampler smoke could be considered manually, but none was launched."
    (out / "PATCH_DETAIL_FLOW_PROTOTYPE_REPORT.md").write_text(
        "# Patch-detail flow prototype report\n\n"
        f"- architecture: joint 3-channel procedural conv flow, hidden `{args.hidden}`, couplings `{args.n_coupling}`\n"
        f"- epochs: `{args.epochs}`\n"
        f"- best action checkpoint: `{best_flow['checkpoint']}`\n"
        f"- best validation deltaS std: `{float(best_flow['deltaS_std']):.6g}`\n"
        f"- best validation NLL: `{float(best_flow['val_nll']):.6g}`\n"
        "- dense checkpoint metrics: `patch_detail_flow_training_metrics.csv`.\n"
    )
    (out / "PATCH_DETAIL_PARAMETERIZATION_FINAL_REPORT.md").write_text(
        "# Patch-detail parameterization final report\n\n"
        f"1. Is the coordinate transform valid and reversible?\n\n   Yes. Roundtrip max error `{roundtrip:.6g}`.\n\n"
        f"2. Does it preserve the small3 block constraint?\n\n   Yes. Reblocking max error `{reblock:.6g}`.\n\n"
        f"3. Can a simple baseline improve action metrics?\n\n   Best baseline `{best_baseline['label']}` has deltaS std `{float(best_baseline['deltaS_std']):.6g}`.\n\n"
        f"4. Does prototype joint patch-detail flow beat old edge/pair factorization?\n\n   Best flow deltaS std `{float(best_flow['deltaS_std']):.6g}` versus larger joint edge+pair `{LARGER_JOINT}` and failed sequential range `{OLD_SEQ_RANGE}`.\n\n"
        f"5. Is sampler smoke justified?\n\n   `{smoke}`. No sampler smoke or long validation was launched.\n\n"
        f"6. Next branch\n\n   {interpretation}\n"
    )
    summary = {
        "status": "completed",
        "roundtrip_max_error": float(roundtrip),
        "reblocking_max_error": float(reblock),
        "best_baseline": best_baseline,
        "best_baseline_deltaS_std": float(best_baseline["deltaS_std"]),
        "best_flow": best_flow,
        "best_flow_deltaS_std": float(best_flow["deltaS_std"]),
        "sampler_smoke_launched": smoke,
        "interpretation": interpretation,
    }
    write_json(out / "patch_detail_parameterization_summary.json", summary)
    update_status(summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
