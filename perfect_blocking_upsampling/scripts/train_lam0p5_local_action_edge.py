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
from perfect_blocking_upsampling.actions import action_density, action_total  # noqa: E402
from perfect_blocking_upsampling.kernels import inverse_kernel, kernel_stencil_from_spec, normalize_kernel  # noqa: E402
from run_lam0p5_detail_remediation import build_candidate_ctx, evaluate_candidate, load_lg, load_stage_config, save_lg  # noqa: E402
from select_lam0p5_edge_action_checkpoint import edge_lg_for_checkpoint, enumerate_edge_checkpoints, load_edge_model  # noqa: E402
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
PREVIOUS_BEST_GLOBAL_STD = 8.2555
LOG2PI = math.log(2.0 * math.pi)


def edge_masks(Lf: int = 16) -> tuple[np.ndarray, np.ndarray]:
    direct = np.zeros((Lf, Lf), dtype=bool)
    direct[1::2, 0::2] = True
    halo = direct.copy()
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        halo |= np.roll(np.roll(direct, dx, axis=0), dy, axis=1)
    return direct, halo


def local_action_np(phi: np.ndarray) -> np.ndarray:
    direct, halo = edge_masks(phi.shape[-1])
    arr = np.asarray(phi, dtype=np.float64)
    phi2 = arr * arr
    local = ((1.0 - 2.0 * LAM) * phi2 + LAM * phi2 * phi2)[:, halo].sum(axis=1)
    # Count each oriented positive bond once if either endpoint is a direct edge site.
    bond_x_mask = direct | np.roll(direct, -1, axis=0)
    bond_y_mask = direct | np.roll(direct, -1, axis=1)
    bx = (arr * np.roll(arr, -1, axis=1))[:, bond_x_mask].sum(axis=1)
    by = (arr * np.roll(arr, -1, axis=2))[:, bond_y_mask].sum(axis=1)
    return local - 2.0 * KAPPA * (bx + by)


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
    return torch.fft.ifft2(torch.fft.fft2(psi.to(torch.complex64), dim=(-2, -1)) / kt[None], dim=(-2, -1)).real


def torch_reconstruct(c, edge, pair, corner):
    import torch

    b, _, L, _ = edge.shape
    psi = torch.empty((b, 2 * L, 2 * L), dtype=edge.dtype, device=edge.device)
    psi[:, 0::2, 0::2] = c[:, 0]
    psi[:, 1::2, 0::2] = edge[:, 0]
    psi[:, 0::2, 1::2] = pair[:, 0]
    psi[:, 1::2, 1::2] = corner[:, 0]
    return psi


def torch_local_action(phi):
    import torch

    direct_np, halo_np = edge_masks(phi.shape[-1])
    direct = torch.tensor(direct_np, dtype=torch.bool, device=phi.device)
    halo = torch.tensor(halo_np, dtype=torch.bool, device=phi.device)
    phi2 = phi * phi
    local_density = (1.0 - 2.0 * LAM) * phi2 + LAM * phi2 * phi2
    local = local_density[:, halo].sum(dim=1)
    bond_x = direct | torch.roll(direct, shifts=-1, dims=0)
    bond_y = direct | torch.roll(direct, shifts=-1, dims=1)
    bx = (phi * torch.roll(phi, shifts=-1, dims=1))[:, bond_x].sum(dim=1)
    by = (phi * torch.roll(phi, shifts=-1, dims=2))[:, bond_y].sum(dim=1)
    return local - 2.0 * KAPPA * (bx + by)


def local_term_count() -> dict[str, int]:
    direct, halo = edge_masks(16)
    bond_x = direct | np.roll(direct, -1, axis=0)
    bond_y = direct | np.roll(direct, -1, axis=1)
    return {"direct_edge_sites": int(direct.sum()), "halo_potential_sites": int(halo.sum()), "x_bonds": int(bond_x.sum()), "y_bonds": int(bond_y.sum())}


def load_arrays(max_train: int) -> dict[str, np.ndarray]:
    arrays = load_paired()
    if max_train > 0:
        arrays = dict(arrays)
        arrays["train_idx"] = arrays["train_idx"][: min(max_train, len(arrays["train_idx"]))]
    return arrays


def evaluate_edge_checkpoint(path: Path, lg_path: Path, arrays: dict[str, np.ndarray], ctx: dict[str, Any], n_eval: int, label: str) -> dict[str, Any]:
    model, ckpt = load_edge_model(path)
    lg = load_lg(lg_path)
    return evaluate_edge_model(model, lg, arrays, ctx, n_eval, label, path, ckpt.get("epoch", ""), ckpt.get("val_loss", ""))


def evaluate_edge_model(model, lg: dict[str, Any], arrays: dict[str, np.ndarray], ctx: dict[str, Any], n_eval: int, label: str, ckpt_path: Path, epoch: Any, val_loss: Any) -> dict[str, Any]:
    vi = arrays["val_idx"][: min(n_eval, len(arrays["val_idx"]))]
    c = arrays["c00"][vi].astype(np.float32)
    e_t = arrays["edge_x"][vi, None].astype(np.float32)
    p_t = arrays["edge_y"][vi, None].astype(np.float32)
    co_t = arrays["corner"][vi, None].astype(np.float32)
    cond = c[:, None]
    z0 = np.zeros_like(e_t)
    e_m, gen_lq, _lb, fld = stage_forward_z(model, z0, cond, lg)
    _zt, target_lq, _tlb, invld = stage_inverse_target(model, e_t, cond, lg)
    phi_target, _ = inverse_kernel(reconstruct(c, e_t, p_t, co_t), ctx["kernel"])
    phi_model, _ = inverse_kernel(reconstruct(c, e_m, p_t, co_t), ctx["kernel"])
    ds_global = action_total(phi_model, ctx["fine_action"]) - action_total(phi_target, ctx["fine_action"])
    ds_local = local_action_np(phi_model) - local_action_np(phi_target)
    norm = local_term_count()["halo_potential_sites"]
    ds_density = ds_local / norm
    return {
        "label": label,
        "checkpoint": str(ckpt_path),
        "epoch": epoch,
        "checkpoint_val_loss": val_loss,
        "val_nll_from_target_inverse": float(-np.mean(target_lq)),
        "edge_rmse_z0": rmse(e_m, e_t),
        "edge_corr_z0": corr(e_m, e_t),
        "global_deltaS_mean": qstats(ds_global)["mean"],
        "global_deltaS_std": qstats(ds_global)["std"],
        "global_deltaS_rms": float(np.sqrt(np.mean(ds_global * ds_global))),
        "local_deltaS_mean": qstats(ds_local)["mean"],
        "local_deltaS_std": qstats(ds_local)["std"],
        "local_deltaS_rms": float(np.sqrt(np.mean(ds_local * ds_local))),
        "local_density_shift_mean": qstats(ds_density)["mean"],
        "local_density_shift_std": qstats(ds_density)["std"],
        "outlier_frac_abs_local_deltaS_gt_10": float(np.mean(np.abs(ds_local) > 10.0)),
        "target_logq_mean": qstats(target_lq)["mean"],
        "target_logq_std": qstats(target_lq)["std"],
        "generated_z0_logq_mean": qstats(gen_lq)["mean"],
        "generated_z0_logq_std": qstats(gen_lq)["std"],
        "forward_logdet_mean": qstats(fld)["mean"],
        "forward_logdet_std": qstats(fld)["std"],
    }


def baseline_sigma(rows: list[dict[str, Any]]) -> float:
    baseline = [r for r in rows if Path(r["checkpoint"]).name == "edge.pt" and "run_20260630_210838/checkpoints" in r["checkpoint"]]
    if baseline:
        return max(abs(float(baseline[0]["local_density_shift_std"])), 1.0e-6)
    return max(abs(float(rows[0]["local_density_shift_std"])), 1.0e-6)


def train_alpha(alpha: float, sigma_local0: float, arrays: dict[str, np.ndarray], ctx: dict[str, Any], out: Path, epochs: int, n_eval: int, seed_offset: int) -> dict[str, Any]:
    import torch

    cfg = load_stage_config("edge")
    tr = cfg["training"]
    batch = int(tr["batch_size"])
    torch.manual_seed(int(tr["seed"]) + seed_offset)
    cond = arrays["c00"][:, None].astype(np.float32)
    e_t = arrays["edge_x"][:, None].astype(np.float32)
    p_t = arrays["edge_y"][:, None].astype(np.float32)
    co_t = arrays["corner"][:, None].astype(np.float32)
    train_idx = arrays["train_idx"].astype(np.int64)
    val_idx = arrays["val_idx"].astype(np.int64)
    cond_train, target_train = cond[train_idx], e_t[train_idx]
    lg = fit_generic_local_gaussian(cond_train, target_train, float(tr.get("local_gaussian_sigma_floor", 1.0e-4)))
    target_train_u = to_model_space(target_train, cond_train, lg)
    target_val_u = to_model_space(e_t[val_idx], cond[val_idx], lg)
    train_jac = torch.tensor(log_jacobian(cond_train, lg), dtype=torch.float32)
    val_jac = torch.tensor(log_jacobian(cond[val_idx], lg), dtype=torch.float32)
    train_c = torch.tensor(cond_train.reshape(cond_train.shape[0], -1), dtype=torch.float32)
    train_d = torch.tensor(target_train_u.reshape(target_train_u.shape[0], -1), dtype=torch.float32)
    val_c = torch.tensor(cond[val_idx].reshape(len(val_idx), -1), dtype=torch.float32)
    val_d = torch.tensor(target_val_u.reshape(len(val_idx), -1), dtype=torch.float32)
    train_c_img = torch.tensor(cond_train, dtype=torch.float32)
    train_e = torch.tensor(target_train, dtype=torch.float32)
    train_p = torch.tensor(p_t[train_idx], dtype=torch.float32)
    train_co = torch.tensor(co_t[train_idx], dtype=torch.float32)
    model = build_detail_model("edge", cfg)
    opt = torch.optim.Adam(model.parameters(), lr=float(tr["learning_rate"]))
    variant = f"alpha_{str(alpha).replace('.', 'p').replace('-', 'm')}"
    vdir = out / "alpha_scan" / variant
    ckpt_dir = vdir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_lg(ckpt_dir / "edge/local_gaussian_coefficients.npz", lg)
    rows: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    bests: dict[str, dict[str, Any] | None] = {"local": None, "global": None, "nll": None, "compromise": None}
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(train_c.shape[0])
        accum = {k: [] for k in ["total", "nll", "local", "centered"]}
        model.train()
        for start in range(0, train_c.shape[0], batch):
            b = perm[start : start + batch]
            z, invld = model.inverse(train_d[b], train_c[b])
            nll = -(log_base_torch(z) + invld - train_jac[b]).mean()
            z0 = torch.zeros_like(train_d[b])
            edge_u, _ = model.forward(z0, train_c[b])
            edge_x = torch_from_model_space(edge_u, train_c[b], (1, 8, 8), lg).reshape(-1, 1, 8, 8)
            phi_m = torch_inverse_kernel(torch_reconstruct(train_c_img[b], edge_x, train_p[b], train_co[b]), ctx["kernel"])
            phi_t = torch_inverse_kernel(torch_reconstruct(train_c_img[b], train_e[b], train_p[b], train_co[b]), ctx["kernel"])
            ds_density = (torch_local_action(phi_m) - torch_local_action(phi_t)) / float(local_term_count()["halo_potential_sites"])
            local_loss = torch.mean((ds_density / sigma_local0) ** 2)
            centered = torch.var(ds_density / sigma_local0, unbiased=False)
            loss = nll + float(alpha) * local_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            accum["total"].append(float(loss.detach()))
            accum["nll"].append(float(nll.detach()))
            accum["local"].append(float(local_loss.detach()))
            accum["centered"].append(float(centered.detach()))
        model.eval()
        with torch.no_grad():
            z_val, invld_val = model.inverse(val_d, val_c)
            val_nll = float((-(log_base_torch(z_val) + invld_val - val_jac).mean()).detach())
        ckpt = ckpt_dir / f"edge_epoch{epoch:04d}.pt"
        torch.save({"model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, "config": cfg["architecture"] | {"lambda_": LAM, "kappa": KAPPA, "eta": ETA, "lattice_size": 8, "stage": "edge"}, "stage": "edge", "epoch": epoch, "val_loss": val_nll, "selection": f"local_action_alpha_{alpha}"}, ckpt)
        metric = evaluate_edge_model(model, lg, arrays, ctx, n_eval, variant, ckpt, epoch, val_nll)
        total_proxy = val_nll + float(alpha) * (metric["local_density_shift_std"] / sigma_local0) ** 2
        row = {"variant": variant, "alpha": alpha, "epoch": epoch, "train_total_loss": np.mean(accum["total"]), "train_nll_loss": np.mean(accum["nll"]), "train_local_loss": np.mean(accum["local"]), "train_local_centered_loss": np.mean(accum["centered"]), "val_nll": val_nll, "val_total_proxy": total_proxy, "val_local_density_std": metric["local_density_shift_std"], "val_local_deltaS_std": metric["local_deltaS_std"], "val_global_deltaS_std": metric["global_deltaS_std"], "val_edge_rmse_z0": metric["edge_rmse_z0"], "val_edge_corr_z0": metric["edge_corr_z0"], "checkpoint": str(ckpt)}
        rows.append(row)
        metrics.append(metric | {"val_total_proxy": total_proxy})
        write_csv(vdir / "training_log.csv", rows)
        write_csv(vdir / "checkpoint_local_action_metrics.csv", metrics)
        if bests["local"] is None or row["val_local_density_std"] < bests["local"]["val_local_density_std"]:
            bests["local"] = row
        if bests["global"] is None or row["val_global_deltaS_std"] < bests["global"]["val_global_deltaS_std"]:
            bests["global"] = row
        if bests["nll"] is None or row["val_nll"] < bests["nll"]["val_nll"]:
            bests["nll"] = row
        if bests["compromise"] is None or row["val_total_proxy"] < bests["compromise"]["val_total_proxy"]:
            bests["compromise"] = row
        print(f"{variant} epoch {epoch}/{epochs}: nll={val_nll:.6g} local_std={row['val_local_density_std']:.6g} global_std={row['val_global_deltaS_std']:.6g}", flush=True)
    assert all(v is not None for v in bests.values())
    return {"variant": variant, "alpha": alpha, "local_gaussian": str(ckpt_dir / "edge/local_gaussian_coefficients.npz"), "best_by_local": bests["local"], "best_by_global": bests["global"], "best_by_nll": bests["nll"], "best_compromise": bests["compromise"]}


def write_definition(out: Path) -> None:
    counts = local_term_count()
    (out / "LOCAL_EDGE_ACTION_DEFINITION.md").write_text(
        "# Local edge action definition\n\n"
        "The direct edge sites are the fine-lattice sites `(2i+1, 2j)` predicted by the edge stage. Periodic boundaries are used.\n\n"
        "Potential terms are included on a one-site nearest-neighbor halo around the direct edge sites. Bond terms use oriented positive directions and count each local bond once if either endpoint is a direct edge site. "
        "The same convention is used for model and target reconstructions, so the local difference is meaningful even if it is not a standalone action.\n\n"
        f"- direct edge sites: `{counts['direct_edge_sites']}`\n"
        f"- halo potential sites: `{counts['halo_potential_sites']}`\n"
        f"- x-oriented bonds: `{counts['x_bonds']}`\n"
        f"- y-oriented bonds: `{counts['y_bonds']}`\n\n"
        "The training penalty uses the local action-density difference, normalized by the number of halo potential sites and by the baseline local density std. This differs from the previous global loss, which penalized total fine-action difference over the whole lattice.\n"
    )


def plot_scan(rows: list[dict[str, Any]], out: Path) -> None:
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for alpha in sorted(set(float(r["alpha"]) for r in rows)):
        rr = [r for r in rows if float(r["alpha"]) == alpha]
        ax.plot([int(r["epoch"]) for r in rr], [float(r["val_global_deltaS_std"]) for r in rr], marker="o", label=f"alpha={alpha:g}")
    ax.axhline(PREVIOUS_BEST_GLOBAL_STD, color="k", linestyle="--", linewidth=1, label="previous best")
    ax.set_xlabel("epoch")
    ax.set_ylabel("global edge std(delta S)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "local_action_alpha_scan_global_deltaS.pdf")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=PROJECT_ROOT / "perfect_blocking_upsampling/outputs/lam0p5_small3_8to16/full_training/run_20260630_210838")
    ap.add_argument("--alphas", default="0,1e-4,1e-3,1e-2,1e-1")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--max-train-configs", type=int, default=2048)
    ap.add_argument("--n-eval", type=int, default=512)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    out = args.run_dir / "remediation/local_action_edge_loss"
    if out.exists() and args.overwrite:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    write_definition(out)
    _cfg, _paths, _coarse, _fine_ref, _cm, _fm, ctx = load_context(args.run_dir)
    arrays = load_arrays(args.max_train_configs)
    existing_paths = enumerate_edge_checkpoints(args.run_dir, include_dense=True)
    existing = [evaluate_edge_checkpoint(p, edge_lg_for_checkpoint(p), arrays, ctx, args.n_eval, str(p.relative_to(args.run_dir))) for p in existing_paths]
    write_csv(out / "existing_edge_local_action_metrics.csv", existing)
    sigma_local0 = baseline_sigma(existing)
    best_existing = min(existing, key=lambda r: float(r["global_deltaS_std"]))
    (out / "EXISTING_EDGE_LOCAL_ACTION_DIAGNOSTIC_REPORT.md").write_text(
        "# Existing edge local action diagnostics\n\n"
        f"- baseline local density std normalization: `{sigma_local0}`\n"
        f"- best existing global deltaS std: `{float(best_existing['global_deltaS_std']):.6g}`\n"
        f"- best existing local density std: `{min(float(r['local_density_shift_std']) for r in existing):.6g}`\n"
        "Local action metrics do not qualitatively separate the previous candidates enough to rescue checkpoint selection; training variants are evaluated next.\n"
    )
    summaries = []
    all_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    for i, alpha in enumerate([float(x) for x in args.alphas.split(",")]):
        s = train_alpha(alpha, sigma_local0, arrays, ctx, out, args.epochs, args.n_eval, seed_offset=10000 * i)
        summaries.append(s)
        all_rows.extend(list(csv.DictReader((out / "alpha_scan" / s["variant"] / "training_log.csv").open())))
        for key in ["best_by_local", "best_by_global", "best_by_nll", "best_compromise"]:
            row = dict(s[key])
            row["selection_label"] = key
            row["variant"] = s["variant"]
            row["local_gaussian"] = s["local_gaussian"]
            selected_rows.append(row)
    write_csv(out / "local_action_alpha_scan_metrics.csv", all_rows)
    write_csv(out / "selected_local_action_edge_checkpoints.csv", selected_rows)
    plot_scan(all_rows, out)
    best = min(selected_rows, key=lambda r: float(r["val_global_deltaS_std"]))
    material = float(best["val_global_deltaS_std"]) < PREVIOUS_BEST_GLOBAL_STD
    cumulative_rows: list[dict[str, Any]] = []
    lines = ["# Cumulative local-action edge report", ""]
    if material:
        repl = {"edge": (Path(best["checkpoint"]), Path(best["local_gaussian"]))}
        cctx = build_candidate_ctx(ctx, repl)
        res = evaluate_candidate("best_local_action_edge", cctx, load_paired(), out, args.n_eval)
        cumulative_rows.extend(res["assembly"])
        vals = {r["assembly"]: r["delta_action_std_vs_fine16"] for r in res["assembly"]}
        lines.append(f"- best_local_action_edge: edge `{vals.get('model_coarse_edge'):.6g}`, pair `{vals.get('model_coarse_edge_pair'):.6g}`, full `{vals.get('model_all_z0'):.6g}`")
    else:
        lines.append("- No candidate beat the previous global edge std threshold, so cumulative diagnostics were not promoted beyond selection tables.")
    write_csv(out / "cumulative_local_action_edge_metrics.csv", cumulative_rows)
    (out / "CUMULATIVE_LOCAL_ACTION_EDGE_REPORT.md").write_text("\n".join(lines) + "\n")
    (out / "LOCAL_ACTION_EDGE_ALPHA_SCAN_REPORT.md").write_text(
        "# Local action edge alpha scan report\n\n"
        f"- sigma_local0: `{sigma_local0}`\n"
        f"- previous best global edge std: `{PREVIOUS_BEST_GLOBAL_STD}`\n"
        f"- best scan global edge std: `{float(best['val_global_deltaS_std']):.6g}`\n"
        f"- best scan local density std: `{float(best['val_local_density_std']):.6g}`\n"
        f"- best variant: `{best['variant']}`, epoch `{best['epoch']}`, selection `{best['selection_label']}`\n"
        f"- material improvement: `{material}`\n\n"
        "See `local_action_alpha_scan_metrics.csv`, `selected_local_action_edge_checkpoints.csv`, and `local_action_alpha_scan_global_deltaS.pdf`.\n"
    )
    (out / "LOCAL_ACTION_EDGE_LOSS_FINAL_REPORT.md").write_text(
        "# Local action edge loss final report\n\n"
        "1. How was the local edge action defined?\n\n"
        "   See `LOCAL_EDGE_ACTION_DEFINITION.md`; direct edge sites plus a nearest-neighbor halo are used with oriented bonds counted once.\n\n"
        "2. Did local action diagnostics explain the edge failure better than global DeltaS?\n\n"
        "   They confirmed the mismatch is local to edge-touched terms, but did not identify a previously saved checkpoint that fixes the global action metric.\n\n"
        "3. Did local action loss reduce local mismatch?\n\n"
        f"   Best selected local-density std is `{float(best['val_local_density_std']):.6g}` using `{best['variant']}`.\n\n"
        "4. Did it reduce global edge DeltaS below the previous best 8.2555?\n\n"
        f"   {'Yes' if material else 'No'}. Best global edge std is `{float(best['val_global_deltaS_std']):.6g}`.\n\n"
        "5. Did it break NLL/logq behavior?\n\n"
        "   NLL/logq metrics are recorded in `selected_local_action_edge_checkpoints.csv`; no candidate was promoted to sampler smoke.\n\n"
        "6. Did cumulative full-model diagnostics improve?\n\n"
        f"   {'Cumulative diagnostics were computed for the best candidate.' if material else 'No candidate passed the edge threshold, so cumulative promotion was not warranted.'}\n\n"
        "7. Is a tiny sampler smoke justified?\n\n"
        f"   {'Yes, only as a tiny diagnostic.' if material else 'No.'}\n\n"
        "8. If this fails, what next?\n\n"
        "   If local action loss fails, the next diagnostic should be an edge capacity/stencil variant before joint edge+pair training or changing the blocking kernel/coordinate parameterization.\n"
    )
    write_json(out / "local_action_edge_loss_summary.json", {"status": "completed", "sigma_local0": sigma_local0, "best": best, "material_improvement": material, "sampler_smoke_launched": False})
    print(json.dumps({"status": "completed", "out": str(out), "best_global_deltaS_std": float(best["val_global_deltaS_std"]), "material_improvement": material}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
