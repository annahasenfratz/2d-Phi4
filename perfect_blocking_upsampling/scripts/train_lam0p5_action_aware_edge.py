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
EXP = PROJECT_ROOT / "ML_sampling_clean" / "experiments" / "decimated_conditional_fillin"
for p in [PROJECT_ROOT, PKG / "scripts", PKG / "src", EXP]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from diagnose_lam0p5_failed_bundle import corr, load_context, load_paired, qstats, reconstruct, rmse, stage_forward_z, stage_inverse_target, write_csv, write_json  # noqa: E402
from perfect_blocking_upsampling.actions import action_total  # noqa: E402
from perfect_blocking_upsampling.kernels import ORBIT_OFFSETS, inverse_kernel, kernel_stencil_from_spec, normalize_kernel  # noqa: E402
from perfect_blocking_upsampling.observables import observables as ensemble_observables  # noqa: E402
from run_lam0p5_detail_remediation import build_candidate_ctx, evaluate_candidate, load_stage_config, save_lg  # noqa: E402
from train_faithful_transported_detail import build_detail_model, log_base_torch  # noqa: E402
from run_staged_decimated_conditional_fillin import fit_generic_local_gaussian, log_jacobian, to_model_space, torch_from_model_space  # noqa: E402

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

LAM = 0.5
KAPPA = 0.3426
ETA = 0.25
SIGMA_EDGE_BASELINE = 8.32341
LOG2PI = math.log(2.0 * math.pi)


def torch_phi4_action(phi):
    import torch

    phi2 = phi * phi
    local = ((1.0 - 2.0 * LAM) * phi2 + LAM * phi2 * phi2).sum(dim=(1, 2))
    hop = (phi * torch.roll(phi, shifts=-1, dims=1)).sum(dim=(1, 2))
    hop = hop + (phi * torch.roll(phi, shifts=-1, dims=2)).sum(dim=(1, 2))
    return local - 2.0 * KAPPA * hop


def torch_inverse_kernel(psi, kernel_spec):
    import torch

    L = psi.shape[-1]
    stencil_np = normalize_kernel(kernel_stencil_from_spec(kernel_spec))
    w = np.zeros((L, L), dtype=np.float64)
    k = stencil_np.shape[0] // 2
    for i in range(stencil_np.shape[0]):
        for j in range(stencil_np.shape[1]):
            w[(i - k) % L, (j - k) % L] = stencil_np[i, j]
    kt_np = (2.0 ** (kernel_spec.eta / 2.0)) * np.fft.fft2(w)
    kt = torch.tensor(kt_np, dtype=torch.complex64, device=psi.device)
    phi = torch.fft.ifft2(torch.fft.fft2(psi.to(torch.complex64), dim=(-2, -1)) / kt[None], dim=(-2, -1)).real
    return phi


def torch_reconstruct(c, edge, pair, corner):
    import torch

    b, _, L, _ = edge.shape
    psi = torch.empty((b, 2 * L, 2 * L), dtype=edge.dtype, device=edge.device)
    psi[:, 0::2, 0::2] = c[:, 0]
    psi[:, 1::2, 0::2] = edge[:, 0]
    psi[:, 0::2, 1::2] = pair[:, 0]
    psi[:, 1::2, 1::2] = corner[:, 0]
    return psi


def load_arrays(max_train: int) -> dict[str, np.ndarray]:
    arrays = load_paired()
    if max_train > 0:
        arrays = dict(arrays)
        arrays["train_idx"] = arrays["train_idx"][: min(max_train, len(arrays["train_idx"]))]
    return arrays


def evaluate_edge_model(model, lg: dict[str, Any], arrays: dict[str, np.ndarray], ctx: dict[str, Any], n_eval: int, alpha: float, variant: str, ckpt_path: Path) -> dict[str, Any]:
    vi = arrays["val_idx"][: min(n_eval, len(arrays["val_idx"]))]
    c = arrays["c00"][vi].astype(np.float32)
    edge_target = arrays["edge_x"][vi, None].astype(np.float32)
    pair_target = arrays["edge_y"][vi, None].astype(np.float32)
    corner_target = arrays["corner"][vi, None].astype(np.float32)
    cond = c[:, None]
    z0 = np.zeros_like(edge_target, dtype=np.float32)
    edge_pred, generated_logq, _lb, fwd_logdet = stage_forward_z(model, z0, cond, lg)
    _zt, target_logq, _tlb, inv_logdet = stage_inverse_target(model, edge_target, cond, lg)
    phi_target, _ = inverse_kernel(reconstruct(c, edge_target, pair_target, corner_target), ctx["kernel"])
    phi_edge, _ = inverse_kernel(reconstruct(c, edge_pred, pair_target, corner_target), ctx["kernel"])
    s_target = action_total(phi_target, ctx["fine_action"])
    s_edge = action_total(phi_edge, ctx["fine_action"])
    delta_s = s_edge - s_target
    return {
        "variant": variant,
        "alpha": alpha,
        "checkpoint": str(ckpt_path),
        "val_nll_from_target_inverse": float(-np.mean(target_logq)),
        "edge_rmse_z0": rmse(edge_pred, edge_target),
        "edge_corr_z0": corr(edge_pred, edge_target),
        "deltaS_mean": float(np.mean(delta_s)),
        "deltaS_std": float(np.std(delta_s, ddof=1)),
        "deltaS_rms": float(np.sqrt(np.mean(delta_s * delta_s))),
        "action_density_shift_mean": float(np.mean(delta_s) / (16 * 16)),
        "outlier_frac_abs_deltaS_gt_10": float(np.mean(np.abs(delta_s) > 10.0)),
        "outlier_frac_abs_deltaS_gt_20": float(np.mean(np.abs(delta_s) > 20.0)),
        "target_logq_mean": qstats(target_logq)["mean"],
        "target_logq_std": qstats(target_logq)["std"],
        "generated_z0_logq_mean": qstats(generated_logq)["mean"],
        "generated_z0_logq_std": qstats(generated_logq)["std"],
        "forward_logdet_mean": qstats(fwd_logdet)["mean"],
        "forward_logdet_std": qstats(fwd_logdet)["std"],
    }


def train_alpha(alpha: float, arrays: dict[str, np.ndarray], ctx: dict[str, Any], out: Path, epochs: int, n_eval: int, seed_offset: int) -> dict[str, Any]:
    import torch

    cfg = load_stage_config("edge")
    tr = cfg["training"]
    batch_size = int(tr["batch_size"])
    torch.manual_seed(int(tr["seed"]) + seed_offset)
    cond = arrays["c00"][:, None].astype(np.float32)
    edge_target = arrays["edge_x"][:, None].astype(np.float32)
    pair_target = arrays["edge_y"][:, None].astype(np.float32)
    corner_target = arrays["corner"][:, None].astype(np.float32)
    train_idx = arrays["train_idx"].astype(np.int64)
    val_idx = arrays["val_idx"].astype(np.int64)
    cond_train, target_train = cond[train_idx], edge_target[train_idx]
    cond_val, target_val = cond[val_idx], edge_target[val_idx]
    lg = fit_generic_local_gaussian(cond_train, target_train, float(tr.get("local_gaussian_sigma_floor", 1.0e-4)))
    target_train_u = to_model_space(target_train, cond_train, lg)
    target_val_u = to_model_space(target_val, cond_val, lg)
    train_jac = torch.tensor(log_jacobian(cond_train, lg), dtype=torch.float32)
    val_jac = torch.tensor(log_jacobian(cond_val, lg), dtype=torch.float32)
    train_c = torch.tensor(cond_train.reshape(cond_train.shape[0], -1), dtype=torch.float32)
    train_d = torch.tensor(target_train_u.reshape(target_train_u.shape[0], -1), dtype=torch.float32)
    val_c = torch.tensor(cond_val.reshape(cond_val.shape[0], -1), dtype=torch.float32)
    val_d = torch.tensor(target_val_u.reshape(target_val_u.shape[0], -1), dtype=torch.float32)
    # Tensors for differentiable action penalty.
    train_c_img = torch.tensor(cond_train, dtype=torch.float32)
    train_pair = torch.tensor(pair_target[train_idx], dtype=torch.float32)
    train_corner = torch.tensor(corner_target[train_idx], dtype=torch.float32)
    train_edge_target = torch.tensor(target_train, dtype=torch.float32)
    model = build_detail_model("edge", cfg)
    opt = torch.optim.Adam(model.parameters(), lr=float(tr["learning_rate"]))
    variant = f"alpha_{str(alpha).replace('.', 'p').replace('-', 'm')}"
    vdir = out / variant
    ckpt_dir = vdir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_lg(ckpt_dir / "edge/local_gaussian_coefficients.npz", lg)
    rows: list[dict[str, Any]] = []
    metrics_rows: list[dict[str, Any]] = []
    best_action: dict[str, Any] | None = None
    best_total: dict[str, Any] | None = None
    best_nll: dict[str, Any] | None = None
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(train_c.shape[0])
        accum = {"loss": [], "nll": [], "action": [], "action_centered": []}
        model.train()
        for start in range(0, train_c.shape[0], batch_size):
            b = perm[start : start + batch_size]
            z, inv_logdet = model.inverse(train_d[b], train_c[b])
            logp = log_base_torch(z) + inv_logdet - train_jac[b]
            nll = -logp.mean()
            z0 = torch.zeros_like(train_d[b])
            edge_u, _fwd = model.forward(z0, train_c[b])
            edge_x = torch_from_model_space(edge_u, train_c[b], (1, 8, 8), lg).reshape(-1, 1, 8, 8)
            psi_target = torch_reconstruct(train_c_img[b], train_edge_target[b], train_pair[b], train_corner[b])
            psi_model = torch_reconstruct(train_c_img[b], edge_x, train_pair[b], train_corner[b])
            ds = torch_phi4_action(torch_inverse_kernel(psi_model, ctx["kernel"])) - torch_phi4_action(torch_inverse_kernel(psi_target, ctx["kernel"]))
            action_loss = torch.mean((ds / SIGMA_EDGE_BASELINE) ** 2)
            action_centered = torch.var(ds / SIGMA_EDGE_BASELINE, unbiased=False)
            loss = nll + float(alpha) * action_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            accum["loss"].append(float(loss.detach()))
            accum["nll"].append(float(nll.detach()))
            accum["action"].append(float(action_loss.detach()))
            accum["action_centered"].append(float(action_centered.detach()))
        model.eval()
        with torch.no_grad():
            z_val, inv_logdet_val = model.inverse(val_d, val_c)
            val_nll = float((-(log_base_torch(z_val) + inv_logdet_val - val_jac).mean()).detach())
        ckpt_path = ckpt_dir / f"edge_epoch{epoch:04d}.pt"
        torch.save(
            {
                "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "config": cfg["architecture"] | {"lambda_": LAM, "kappa": KAPPA, "eta": ETA, "lattice_size": 8, "stage": "edge"},
                "stage": "edge",
                "epoch": epoch,
                "val_loss": val_nll,
                "selection": f"action_aware_alpha_{alpha}",
            },
            ckpt_path,
        )
        metric = evaluate_edge_model(model, lg, arrays, ctx, n_eval, alpha, variant, ckpt_path)
        total_val = val_nll + float(alpha) * (metric["deltaS_rms"] / SIGMA_EDGE_BASELINE) ** 2
        row = {
            "variant": variant,
            "alpha": alpha,
            "epoch": epoch,
            "train_total_loss": float(np.mean(accum["loss"])),
            "train_nll_loss": float(np.mean(accum["nll"])),
            "train_action_loss": float(np.mean(accum["action"])),
            "train_action_centered_loss": float(np.mean(accum["action_centered"])),
            "val_nll": val_nll,
            "val_total_proxy": total_val,
            "val_deltaS_std": metric["deltaS_std"],
            "val_deltaS_rms": metric["deltaS_rms"],
            "val_edge_rmse_z0": metric["edge_rmse_z0"],
            "val_edge_corr_z0": metric["edge_corr_z0"],
            "checkpoint": str(ckpt_path),
        }
        rows.append(row)
        metrics_rows.append(metric | {"epoch": epoch, "val_total_proxy": total_val})
        write_csv(vdir / "training_log.csv", rows)
        write_csv(vdir / "checkpoint_action_metrics.csv", metrics_rows)
        if best_action is None or row["val_deltaS_std"] < best_action["val_deltaS_std"]:
            best_action = row
        if best_total is None or row["val_total_proxy"] < best_total["val_total_proxy"]:
            best_total = row
        if best_nll is None or row["val_nll"] < best_nll["val_nll"]:
            best_nll = row
        print(f"{variant} epoch {epoch}/{epochs}: val_nll={val_nll:.6g} dSstd={metric['deltaS_std']:.6g} total={total_val:.6g}", flush=True)
    assert best_action and best_total and best_nll
    for label, row in [("best_by_action", best_action), ("best_by_total", best_total), ("best_by_nll", best_nll)]:
        shutil.copy2(row["checkpoint"], ckpt_dir / f"edge_{label}.pt")
    return {
        "variant": variant,
        "alpha": alpha,
        "best_by_action": best_action,
        "best_by_total": best_total,
        "best_by_nll": best_nll,
        "local_gaussian": str(ckpt_dir / "edge/local_gaussian_coefficients.npz"),
    }


def plot_scan(rows: list[dict[str, Any]], out: Path) -> None:
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    for alpha in sorted(set(float(r["alpha"]) for r in rows)):
        rr = [r for r in rows if float(r["alpha"]) == alpha]
        ax.plot([int(r["epoch"]) for r in rr], [float(r["val_deltaS_std"]) for r in rr], marker="o", label=f"alpha={alpha:g}")
    ax.axhline(SIGMA_EDGE_BASELINE, color="k", linestyle="--", linewidth=1, label="baseline")
    ax.set_xlabel("epoch")
    ax.set_ylabel("edge std(delta S)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "edge_alpha_scan_deltaS.pdf")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=PROJECT_ROOT / "perfect_blocking_upsampling/outputs/lam0p5_small3_8to16/full_training/run_20260630_210838")
    ap.add_argument("--alphas", default="0,1e-3,1e-2,1e-1")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--max-train-configs", type=int, default=2048)
    ap.add_argument("--n-eval", type=int, default=512)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    out = args.run_dir / "remediation/action_aware_edge_loss"
    if out.exists() and args.overwrite:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "ACTION_AWARE_EDGE_LOSS_DEFINITION.md").write_text(
        "# Action-aware edge loss definition\n\n"
        "The gathered edge architecture is unchanged (`r_c=3`, `r_f=6`). The training loss is\n\n"
        "`L_total = L_NLL + alpha * mean((DeltaS_edge / sigma_edge_baseline)^2)`\n\n"
        f"with `sigma_edge_baseline = {SIGMA_EDGE_BASELINE}`. `DeltaS_edge = S(phi_model_edge) - S(phi_target)`, where `phi_model_edge` uses target coarse and target downstream pair/corner details but the model edge generated at `z=0`. "
        "A centered action loss is logged as a diagnostic but is not used in the optimizer. Signed magnetization and Binder are not used.\n"
    )
    _cfg, _paths, _coarse, _fine_ref, _cm, _fm, ctx = load_context(args.run_dir)
    arrays = load_arrays(args.max_train_configs)
    alphas = [float(x) for x in args.alphas.split(",")]
    summaries = []
    all_rows: list[dict[str, Any]] = []
    for i, alpha in enumerate(alphas):
        summaries.append(train_alpha(alpha, arrays, ctx, out, args.epochs, args.n_eval, seed_offset=10000 * i))
        log_rows = list(csv.DictReader((out / summaries[-1]["variant"] / "training_log.csv").open()))
        all_rows.extend(log_rows)
    write_csv(out / "action_aware_edge_alpha_scan_metrics.csv", all_rows)
    plot_scan(all_rows, out)
    selected = []
    for s in summaries:
        for label in ["best_by_action", "best_by_total", "best_by_nll"]:
            row = dict(s[label])
            row["selection_label"] = label
            row["variant"] = s["variant"]
            row["local_gaussian"] = s["local_gaussian"]
            selected.append(row)
    write_csv(out / "selected_action_aware_edge_checkpoints.csv", selected)
    best = min(selected, key=lambda r: float(r["val_deltaS_std"]))
    # Cumulative diagnostics for selected candidates.
    cumulative_rows = []
    candidate_lines = []
    pair = args.run_dir / "remediation/pair_corner_longer_training/pair_longer_continue/checkpoints/pair.pt"
    pair_lg = args.run_dir / "remediation/pair_corner_longer_training/pair_longer_continue/checkpoints/pair/local_gaussian_coefficients.npz"
    corner = args.run_dir / "remediation/pair_corner_longer_training/corner_longer_continue/checkpoints/corner.pt"
    corner_lg = args.run_dir / "remediation/pair_corner_longer_training/corner_longer_continue/checkpoints/corner/local_gaussian_coefficients.npz"
    for row in selected:
        if row["selection_label"] != "best_by_action":
            continue
        repl = {"edge": (Path(row["checkpoint"]), Path(row["local_gaussian"]))}
        cctx = build_candidate_ctx(ctx, repl)
        res = evaluate_candidate(f"{row['variant']}_edge_only", cctx, load_paired(), out, args.n_eval)
        cumulative_rows.extend(res["assembly"])
        vals = {r["assembly"]: r["delta_action_std_vs_fine16"] for r in res["assembly"]}
        candidate_lines.append(f"- {res['candidate']}: edge `{vals.get('model_coarse_edge'):.6g}`, pair `{vals.get('model_coarse_edge_pair'):.6g}`, full `{vals.get('model_all_z0'):.6g}`")
        if pair.exists() and corner.exists():
            repl2 = repl | {"pair": (pair, pair_lg), "corner": (corner, corner_lg)}
            cctx2 = build_candidate_ctx(ctx, repl2)
            res2 = evaluate_candidate(f"{row['variant']}_plus_pair_corner_continue", cctx2, load_paired(), out, args.n_eval)
            cumulative_rows.extend(res2["assembly"])
            vals2 = {r["assembly"]: r["delta_action_std_vs_fine16"] for r in res2["assembly"]}
            candidate_lines.append(f"- {res2['candidate']}: edge `{vals2.get('model_coarse_edge'):.6g}`, pair `{vals2.get('model_coarse_edge_pair'):.6g}`, full `{vals2.get('model_all_z0'):.6g}`")
    write_csv(out / "cumulative_action_aware_edge_metrics.csv", cumulative_rows)
    material = float(best["val_deltaS_std"]) < 0.75 * SIGMA_EDGE_BASELINE
    (out / "ACTION_AWARE_EDGE_ALPHA_SCAN_REPORT.md").write_text(
        "# Action-aware edge alpha scan report\n\n"
        f"- alphas: `{alphas}`\n"
        f"- best edge deltaS std: `{float(best['val_deltaS_std']):.6g}`\n"
        f"- best row: variant `{best['variant']}`, epoch `{best['epoch']}`, selection `{best['selection_label']}`\n"
        f"- material improvement threshold met: `{material}`\n\n"
        "See `action_aware_edge_alpha_scan_metrics.csv`, `selected_action_aware_edge_checkpoints.csv`, and `edge_alpha_scan_deltaS.pdf`.\n"
    )
    (out / "CUMULATIVE_ACTION_AWARE_EDGE_REPORT.md").write_text("# Cumulative action-aware edge report\n\n" + "\n".join(candidate_lines) + "\n")
    (out / "ACTION_AWARE_EDGE_LOSS_FINAL_REPORT.md").write_text(
        "# Action-aware edge loss final report\n\n"
        f"1. Does explicit action-aware loss reduce edge DeltaS std?\n\n   Best observed std is `{float(best['val_deltaS_std']):.6g}` versus baseline `{SIGMA_EDGE_BASELINE}`.\n\n"
        f"2. Which alpha works best?\n\n   Best row uses `{best['variant']}` at epoch `{best['epoch']}`.\n\n"
        "3. Does reducing edge DeltaS break NLL/logq?\n\n   Compare selected rows in `selected_action_aware_edge_checkpoints.csv`; NLL/logq were tracked for every checkpoint.\n\n"
        "4. Does cumulative pair/full behavior improve?\n\n   See `cumulative_action_aware_edge_metrics.csv` and `CUMULATIVE_ACTION_AWARE_EDGE_REPORT.md`.\n\n"
        f"5. Is a tiny sampler smoke justified?\n\n   {'Yes, by the configured threshold.' if material else 'No. The edge action metric did not improve by the required 25% threshold.'}\n\n"
        "6. If not, should the next step be local action loss, larger edge architecture, or joint edge+pair training?\n\n   If this explicit global edge action penalty is insufficient, the next step is a local action-density edge loss or an edge capacity/stencil variant before joint edge+pair training.\n"
    )
    write_json(out / "action_aware_edge_loss_summary.json", {"status": "completed", "best": best, "material_improvement": material, "sampler_smoke_launched": False})
    print(json.dumps({"status": "completed", "out": str(out), "best_deltaS_std": float(best["val_deltaS_std"]), "material_improvement": material}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
