"""Supervised linear/Gaussian diagnostics for detail variables."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/matplotlib")

from inverse_blocking_flow.data import load_or_generate_fine_configs
from inverse_blocking_flow.detail_correlation_diagnostics import channel_diagnostics
from inverse_blocking_flow.flow import ConditionalDetailFlow, make_conditioning, n_conditioning_channels
from inverse_blocking_flow.haar import average_block, reconstruct_from_average_block
from inverse_blocking_flow.logw_kappa_scan_blocked_coarse import (
    aggregate_abs_rel,
    ensemble_summary,
    kappa_grid,
    stabilized_logw_stats,
)
from inverse_blocking_flow.phi4 import Phi4Params, phi4_action


CHANNELS = ["HL", "LH", "HH"]
OBS_KEYS = ["S_mean", "S_std", "phi2", "NN_corr", "high_p_power"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fine-size", type=int, default=32)
    parser.add_argument("--kappa", type=float, default=0.31)
    parser.add_argument("--lambda-f", dest="lam", type=float, default=1.0)
    parser.add_argument("--n-configs", type=int, default=512)
    parser.add_argument("--n-eval", type=int, default=256)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--data-path", type=Path, default=Path("inverse_blocking_flow/outputs/fine_configs.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("inverse_blocking_flow/outputs"))
    parser.add_argument("--baseline-flow", type=Path, default=Path("inverse_blocking_flow/outputs/conditional_detail_flow_reverse_kl.pt"))
    parser.add_argument("--physics-flow", type=Path, default=Path("inverse_blocking_flow/outputs/conditional_detail_flow_physics_reverse_kl.pt"))
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--hidden-channels", type=int, default=48)
    parser.add_argument("--cnn-depth", type=int, default=3)
    parser.add_argument("--kappa-min", type=float, default=0.20)
    parser.add_argument("--kappa-max", type=float, default=0.38)
    parser.add_argument("--kappa-step", type=float, default=0.01)
    parser.add_argument("--burn-in", type=int, default=200)
    parser.add_argument("--sample-interval", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--proposal-width", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=818181)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    return parser


def local_features(phi_c: torch.Tensor) -> torch.Tensor:
    grad_x = 0.5 * (torch.roll(phi_c, -1, dims=-1) - torch.roll(phi_c, 1, dims=-1))
    grad_y = 0.5 * (torch.roll(phi_c, -1, dims=-2) - torch.roll(phi_c, 1, dims=-2))
    grad_sq = grad_x.square() + grad_y.square()
    lap = (
        torch.roll(phi_c, 1, dims=-1)
        + torch.roll(phi_c, -1, dims=-1)
        + torch.roll(phi_c, 1, dims=-2)
        + torch.roll(phi_c, -1, dims=-2)
        - 4.0 * phi_c
    )
    ones = torch.ones_like(phi_c)
    return torch.stack((ones, phi_c, phi_c.square(), grad_x, grad_y, grad_sq, lap), dim=-1)


def conv_features(phi_c: torch.Tensor, radius: int) -> torch.Tensor:
    feats = [torch.ones_like(phi_c)]
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            feats.append(torch.roll(torch.roll(phi_c, dy, dims=-2), dx, dims=-1))
    return torch.stack(feats, dim=-1)


def flatten_xy(features: torch.Tensor, d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x = features.reshape(-1, features.shape[-1]).double()
    y = d.permute(0, 2, 3, 1).reshape(-1, 3).double()
    return x, y


def fit_linear(features: torch.Tensor, d: torch.Tensor, ridge: float = 1e-8) -> torch.Tensor:
    x, y = flatten_xy(features, d)
    xtx = x.T @ x
    xtx = xtx + ridge * torch.eye(xtx.shape[0], dtype=xtx.dtype)
    return torch.linalg.solve(xtx, x.T @ y).float()


def predict(features: torch.Tensor, coef: torch.Tensor) -> torch.Tensor:
    y = features.float().reshape(-1, features.shape[-1]) @ coef.float()
    return y.reshape(*features.shape[:-1], 3).permute(0, 3, 1, 2).contiguous()


def tensor_moments(x: torch.Tensor) -> dict[str, float]:
    x = x.float().flatten()
    mean = x.mean()
    centered = x - mean
    std = x.std(unbiased=False).clamp_min(1e-12)
    return {
        "mean": float(mean.item()),
        "std": float(std.item()),
        "skewness": float((centered.pow(3).mean() / std.pow(3)).item()),
        "kurtosis": float((centered.pow(4).mean() / std.pow(4)).item()),
    }


def correlation(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.float().flatten()
    y = y.float().flatten()
    x = x - x.mean()
    y = y - y.mean()
    denom = x.square().mean().sqrt() * y.square().mean().sqrt()
    if float(denom.item()) < 1e-12:
        return float("nan")
    return float((x * y).mean().div(denom).item())


def residual_stats(residual: torch.Tensor, d_true: torch.Tensor, phi_c: torch.Tensor) -> dict[str, object]:
    out = {"channels": {}}
    var_true = d_true.float().var(dim=(0, 2, 3), unbiased=False)
    var_res = residual.float().var(dim=(0, 2, 3), unbiased=False)
    ll2 = phi_c.square()
    for i, name in enumerate(CHANNELS):
        ch = residual[:, i]
        moments = tensor_moments(ch)
        moments["variance_explained"] = float((1.0 - var_res[i] / var_true[i].clamp_min(1e-12)).item())
        moments["R2"] = moments["variance_explained"]
        moments["nn_x"] = correlation(ch, torch.roll(ch, -1, dims=-1))
        moments["nn_y"] = correlation(ch, torch.roll(ch, -1, dims=-2))
        moments["corr_LL2_r2"] = correlation(ll2, ch.square())
        moments["gaussian_skewness_error"] = abs(moments["skewness"])
        moments["gaussian_excess_kurtosis"] = moments["kurtosis"] - 3.0
        out["channels"][name] = moments
    flat = residual.permute(1, 0, 2, 3).reshape(3, -1).float()
    flat = flat - flat.mean(dim=1, keepdim=True)
    cov = flat @ flat.T / flat.shape[1]
    std = cov.diag().clamp_min(1e-12).sqrt()
    out["cross_channel_covariance"] = cov.tolist()
    out["cross_channel_correlation"] = (cov / (std[:, None] * std[None, :])).tolist()
    out["power_spectrum"] = channel_diagnostics(residual, phi_c)["power_spectrum"]
    return out


def residual_covariances(residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    flat = residual.permute(0, 2, 3, 1).reshape(-1, 3).float()
    mean = flat.mean(dim=0, keepdim=True)
    centered = flat - mean
    cov = centered.T @ centered / centered.shape[0]
    diag_var = cov.diag().clamp_min(1e-8)
    cov = cov + 1e-6 * torch.eye(3)
    return diag_var, cov


def sample_diag(pred: torch.Tensor, diag_var: torch.Tensor, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    std = diag_var.sqrt().view(1, 3, 1, 1)
    eps = torch.randn(pred.shape, generator=generator)
    sample = pred + std * eps
    logq = -0.5 * ((eps.square() + math.log(2.0 * math.pi)).sum(dim=(1, 2, 3)) + pred.shape[-2] * pred.shape[-1] * torch.log(diag_var).sum())
    return sample, logq


def sample_full(pred: torch.Tensor, cov: torch.Tensor, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    chol = torch.linalg.cholesky(cov)
    eps = torch.randn((*pred.shape[:1], pred.shape[-2], pred.shape[-1], 3), generator=generator)
    noise = eps @ chol.T
    sample = pred + noise.permute(0, 3, 1, 2)
    logdet = torch.logdet(cov)
    qform = eps.square().sum(dim=(1, 2, 3))
    volume = pred.shape[-2] * pred.shape[-1]
    logq = -0.5 * (qform + volume * (3.0 * math.log(2.0 * math.pi) + logdet))
    return sample, logq


def logw_stats_for(phi: torch.Tensor, logq: torch.Tensor, params: Phi4Params) -> dict[str, float]:
    return stabilized_logw_stats(-phi4_action(phi, params) - logq)


def kappa_scan(phi: torch.Tensor, logq: torch.Tensor, true_summary: dict[str, float], args: argparse.Namespace) -> dict[str, object]:
    rows = []
    for kappa in kappa_grid(args.kappa_min, args.kappa_max, args.kappa_step):
        params = Phi4Params(kappa=kappa, lam=args.lam)
        obs = ensemble_summary(phi, params)
        rows.append(
            {
                "kappa_f": kappa,
                "logw": stabilized_logw_stats(-phi4_action(phi, params) - logq),
                "observables": obs,
                "aggregate_abs_rel_error_vs_true": aggregate_abs_rel(obs, true_summary),
            }
        )
    best_width = min(rows, key=lambda r: r["logw"]["std_logw_centered"])
    best_obs = min(rows, key=lambda r: r["aggregate_abs_rel_error_vs_true"])
    return {"scan": rows, "best_by_logw_width": best_width, "best_by_observable_error": best_obs}


def load_flow(path: Path, args: argparse.Namespace, device: torch.device) -> tuple[ConditionalDetailFlow, str]:
    state = torch.load(path, map_location=device, weights_only=False)
    meta = state.get("args", {})
    mode = state.get("conditioning_mode") or meta.get("conditioning_mode") or "basic"
    n_cond = int(state.get("n_conditioning_channels") or n_conditioning_channels(mode))
    layers = int(meta.get("layers", args.layers))
    hidden = int(meta.get("hidden_channels", args.hidden_channels))
    depth = int(meta.get("cnn_depth", args.cnn_depth))
    flow = ConditionalDetailFlow(layers, hidden, depth, n_cond).to(device)
    flow.load_state_dict(state["model"])
    flow.eval()
    return flow, mode


@torch.no_grad()
def flow_sample_summary(path: Path, name: str, phi_c: torch.Tensor, true_summary: dict[str, float], args: argparse.Namespace) -> dict[str, object] | None:
    if not path.exists():
        return None
    device = torch.device(args.device)
    flow, mode = load_flow(path, args, device)
    cond = make_conditioning(phi_c.to(device), mode)
    generator = torch.Generator(device=device).manual_seed(args.seed + 55)
    d, logq = flow.sample(cond, generator=generator)
    phi = reconstruct_from_average_block(phi_c.to(device), d)
    params = Phi4Params(kappa=args.kappa, lam=args.lam)
    return {
        "name": name,
        "observables": ensemble_summary(phi.cpu(), params),
        "aggregate_abs_rel_error": aggregate_abs_rel(ensemble_summary(phi.cpu(), params), true_summary),
        "logw": logw_stats_for(phi.cpu(), logq.cpu(), params),
        "detail_diagnostics": channel_diagnostics(d.cpu(), phi_c.cpu()),
    }


def write_report(path: Path, summary: dict[str, object]) -> None:
    lines = [
        "# Detail Gaussian Preconditioner",
        "",
        "This supervised diagnostic fits simple predictors for true 16->32 detail variables from blocked true `LL=phi16` fields. Fits are evaluated on held-out configurations.",
        "",
        "## Predictive Fits",
        "",
        "| model | channel | R^2 | residual std | residual nn_x | residual nn_y | corr(LL^2,r^2) | skew | kurtosis |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model_name, model in summary["models"].items():
        for ch in CHANNELS:
            row = model["residual_stats"]["channels"][ch]
            lines.append(
                f"| {model_name} | {ch} | {row['R2']:.6g} | {row['std']:.6g} | {row['nn_x']:.6g} | "
                f"{row['nn_y']:.6g} | {row['corr_LL2_r2']:.6g} | {row['skewness']:.6g} | {row['kurtosis']:.6g} |"
            )
    lines.extend(
        [
            "",
            "## Sampling Reconstructions",
            "",
            "| ensemble | S mean | S std | phi2 | NN corr | high-p | agg rel err | logw std | ESS/N |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, row in summary["sampling"].items():
        obs = row["observables"]
        logw = row.get("logw", {})
        lines.append(
            f"| {name} | {obs['S_mean']:.6g} | {obs['S_std']:.6g} | {obs['phi2']:.6g} | {obs['NN_corr']:.6g} | "
            f"{obs['high_p_power']:.6g} | {row['aggregate_abs_rel_error']:.6g} | "
            f"{logw.get('std_logw_centered', float('nan')):.6g} | {logw.get('ess_over_n', float('nan')):.6g} |"
        )
    ans = summary["answers"]
    lines.extend(
        [
            "",
            "## Main Answers",
            "",
            f"1. How much is linearly predictable? {ans['predictability']}",
            f"2. Are HL and LH symmetric? {ans['hl_lh_symmetry']}",
            f"3. Which correlations remain? {ans['remaining_residual_correlations']}",
            f"4. Does Gaussian preconditioning improve reconstruction/logw versus flow? {ans['gaussian_vs_flow']}",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def plot_outputs(path: Path, summary: dict[str, object]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    models = list(summary["models"])
    with PdfPages(path) as pdf:
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
        x = torch.arange(len(models)).numpy()
        for i, ch in enumerate(CHANNELS):
            axes[0].bar(x + (i - 1) * 0.24, [summary["models"][m]["residual_stats"]["channels"][ch]["R2"] for m in models], width=0.24, label=ch)
            axes[1].bar(x + (i - 1) * 0.24, [summary["models"][m]["residual_stats"]["channels"][ch]["nn_x"] for m in models], width=0.24, label=ch)
            axes[2].bar(x + (i - 1) * 0.24, [summary["models"][m]["residual_stats"]["channels"][ch]["corr_LL2_r2"] for m in models], width=0.24, label=ch)
        for ax, title in zip(axes, ["R2", "residual nn_x", "corr(LL^2,r^2)"]):
            ax.set_xticks(x, models, rotation=30, ha="right")
            ax.set_title(title)
            ax.legend(fontsize=7)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        sampling = list(summary["sampling"])
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].bar(range(len(sampling)), [summary["sampling"][s]["aggregate_abs_rel_error"] for s in sampling])
        axes[0].set_xticks(range(len(sampling)), sampling, rotation=35, ha="right")
        axes[0].set_ylabel("aggregate observable error")
        axes[1].bar(range(len(sampling)), [summary["sampling"][s].get("logw", {}).get("std_logw_centered", float("nan")) for s in sampling])
        axes[1].set_xticks(range(len(sampling)), sampling, rotation=35, ha="right")
        axes[1].set_ylabel("std centered logw")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    params = Phi4Params(kappa=args.kappa, lam=args.lam)
    phi32 = load_or_generate_fine_configs(
        args.data_path,
        n_configs=args.n_configs,
        fine_size=args.fine_size,
        params=params,
        burn_in=args.burn_in,
        interval=args.sample_interval,
        batch_size=args.batch_size,
        proposal_width=args.proposal_width,
        seed=args.seed,
        device=args.device,
    ).float()
    phi_c, d_true = average_block(phi32)
    n_train = max(1, int(args.train_fraction * len(phi32)))
    n_eval = min(args.n_eval, len(phi32) - n_train)
    phi_c_train, d_train = phi_c[:n_train], d_true[:n_train]
    phi_c_eval, d_eval, phi_eval = phi_c[n_train:n_train + n_eval], d_true[n_train:n_train + n_eval], phi32[n_train:n_train + n_eval]
    true_summary = ensemble_summary(phi_eval, params)

    feature_sets = {
        "zero_mean": (None, torch.zeros_like(d_eval)),
        "local_features": (local_features(phi_c_train), None),
        "conv_R1": (conv_features(phi_c_train, 1), None),
        "conv_R2": (conv_features(phi_c_train, 2), None),
        "conv_R3": (conv_features(phi_c_train, 3), None),
    }
    models: dict[str, object] = {}
    predictions: dict[str, torch.Tensor] = {"zero_mean": torch.zeros_like(d_eval)}
    for name, (features_train, fixed_pred) in feature_sets.items():
        if fixed_pred is not None:
            pred = fixed_pred
            coef_shape = None
        else:
            if name == "local_features":
                features_eval = local_features(phi_c_eval)
            else:
                radius = int(name.replace("conv_R", ""))
                features_eval = conv_features(phi_c_eval, radius)
            coef = fit_linear(features_train, d_train)
            pred = predict(features_eval, coef)
            coef_shape = list(coef.shape)
        residual = d_eval - pred
        diag_var, cov = residual_covariances(residual)
        stats = residual_stats(residual, d_eval, phi_c_eval)
        models[name] = {
            "coef_shape": coef_shape,
            "residual_diag_variance": diag_var.tolist(),
            "residual_full_covariance": cov.tolist(),
            "residual_stats": stats,
        }
        predictions[name] = pred

    sampling: dict[str, object] = {
        "true": {
            "observables": true_summary,
            "aggregate_abs_rel_error": 0.0,
        }
    }
    generator = torch.Generator().manual_seed(args.seed + 9)
    for name, pred in predictions.items():
        residual = d_eval - pred
        diag_var, cov = residual_covariances(residual)
        for cov_name, sampler_arg in (("diag", diag_var), ("full3", cov)):
            if cov_name == "diag":
                d_sample, logq = sample_diag(pred, sampler_arg, generator)
            else:
                d_sample, logq = sample_full(pred, sampler_arg, generator)
            phi_sample = reconstruct_from_average_block(phi_c_eval, d_sample)
            obs = ensemble_summary(phi_sample, params)
            sampling[f"{name}_{cov_name}"] = {
                "observables": obs,
                "aggregate_abs_rel_error": aggregate_abs_rel(obs, true_summary),
                "logw": logw_stats_for(phi_sample, logq, params),
                "detail_diagnostics": channel_diagnostics(d_sample, phi_c_eval),
                "kappa_scan": kappa_scan(phi_sample, logq, true_summary, args),
            }

    for path, name in ((args.baseline_flow, "baseline_reverse_kl_flow"), (args.physics_flow, "physics_conditioned_flow")):
        flow_summary = flow_sample_summary(path, name, phi_c_eval, true_summary, args)
        if flow_summary is not None:
            sampling[name] = flow_summary

    best_model = max(models, key=lambda m: sum(models[m]["residual_stats"]["channels"][ch]["R2"] for ch in CHANNELS) / 3.0)
    flow_candidates = [k for k in sampling if k.endswith("_flow")]
    gauss_candidates = [k for k in sampling if not k.endswith("_flow") and k != "true"]
    best_gauss_name = min(gauss_candidates, key=lambda k: sampling[k]["aggregate_abs_rel_error"])
    best_flow_err = min((sampling[k]["aggregate_abs_rel_error"] for k in flow_candidates), default=float("inf"))
    best_flow_name = min(flow_candidates, key=lambda k: sampling[k]["aggregate_abs_rel_error"]) if flow_candidates else "missing"
    best_gauss_err = sampling[best_gauss_name]["aggregate_abs_rel_error"]
    hl_r2 = models[best_model]["residual_stats"]["channels"]["HL"]["R2"]
    lh_r2 = models[best_model]["residual_stats"]["channels"]["LH"]["R2"]
    max_res_nn = max(
        abs(models[best_model]["residual_stats"]["channels"][ch]["nn_x"]) + abs(models[best_model]["residual_stats"]["channels"][ch]["nn_y"])
        for ch in CHANNELS
    ) / 2.0
    summary = {
        "setup": {
            "fine_size": args.fine_size,
            "kappa": args.kappa,
            "lambda": args.lam,
            "n_train": n_train,
            "n_eval": n_eval,
        },
        "true_observables": true_summary,
        "models": models,
        "sampling": sampling,
        "answers": {
            "predictability": f"Best mean R2 model is {best_model}; R2 values are HL={hl_r2:.6g}, LH={lh_r2:.6g}, HH={models[best_model]['residual_stats']['channels']['HH']['R2']:.6g}.",
            "hl_lh_symmetry": f"For {best_model}, HL R2={hl_r2:.6g} and LH R2={lh_r2:.6g}; difference={abs(hl_r2-lh_r2):.6g}.",
            "remaining_residual_correlations": f"For {best_model}, largest average absolute residual NN correlation is about {max_res_nn:.6g}; see residual covariance/power spectra in JSON.",
            "gaussian_vs_flow": f"Best Gaussian sampler is {best_gauss_name} with aggregate error {best_gauss_err:.6g}; best flow is {best_flow_name} with aggregate error {best_flow_err:.6g}.",
        },
    }
    (args.output_dir / "detail_gaussian_preconditioner_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_report(args.output_dir / "detail_gaussian_preconditioner_report.md", summary)
    plot_outputs(args.output_dir / "detail_gaussian_preconditioner_plots.pdf", summary)
    print(f"wrote {args.output_dir / 'detail_gaussian_preconditioner_summary.json'}")
    print(f"wrote {args.output_dir / 'detail_gaussian_preconditioner_report.md'}")
    print(f"wrote {args.output_dir / 'detail_gaussian_preconditioner_plots.pdf'}")
    print("best_predictor", best_model)
    print("best_gaussian_sampling", best_gauss_name, "aggregate_error", f"{best_gauss_err:.6g}")
    print("best_flow_sampling", best_flow_name, "aggregate_error", f"{best_flow_err:.6g}")


if __name__ == "__main__":
    main()
