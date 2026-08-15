"""Detail-field diagnostics for 16->32 conditional flows."""

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
from inverse_blocking_flow.flow import ConditionalDetailFlow, make_conditioning, n_conditioning_channels
from inverse_blocking_flow.haar import average_block
from inverse_blocking_flow.phi4 import Phi4Params


CHANNELS = ["HL", "LH", "HH"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fine-size", type=int, default=32)
    parser.add_argument("--kappa", type=float, default=0.31)
    parser.add_argument("--lambda-f", dest="lam", type=float, default=1.0)
    parser.add_argument("--n-configs", type=int, default=512)
    parser.add_argument("--n-eval", type=int, default=512)
    parser.add_argument("--data-path", type=Path, default=Path("inverse_blocking_flow/outputs/fine_configs.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("inverse_blocking_flow/outputs"))
    parser.add_argument("--baseline-checkpoint", type=Path, default=Path("inverse_blocking_flow/outputs/conditional_detail_flow_reverse_kl.pt"))
    parser.add_argument("--logwvar-checkpoint", type=Path, default=Path("inverse_blocking_flow/outputs/logw_var_penalty/conditional_detail_flow_logwvar_alpha_0p01.pt"))
    parser.add_argument("--physics-checkpoint", type=Path, default=Path("inverse_blocking_flow/outputs/conditional_detail_flow_physics_reverse_kl.pt"))
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--hidden-channels", type=int, default=48)
    parser.add_argument("--cnn-depth", type=int, default=3)
    parser.add_argument("--burn-in", type=int, default=200)
    parser.add_argument("--sample-interval", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--proposal-width", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=717171)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    return parser


def tensor_stats(x: torch.Tensor) -> dict[str, float]:
    x = x.detach().float().flatten().cpu()
    mean = x.mean()
    centered = x - mean
    std = x.std(unbiased=False).clamp_min(1e-12)
    skew = (centered.pow(3).mean() / std.pow(3)).item()
    kurt = (centered.pow(4).mean() / std.pow(4)).item()
    return {
        "mean": float(mean.item()),
        "std": float(std.item()),
        "skewness": float(skew),
        "kurtosis": float(kurt),
    }


def correlation(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.detach().float().flatten()
    y = y.detach().float().flatten()
    x = x - x.mean()
    y = y - y.mean()
    denom = x.square().mean().sqrt() * y.square().mean().sqrt()
    if float(denom.item()) < 1e-12:
        return float("nan")
    return float((x * y).mean().div(denom).item())


def spatial_correlations(channel: torch.Tensor) -> dict[str, float]:
    return {
        "nn_x": correlation(channel, torch.roll(channel, -1, dims=-1)),
        "nn_y": correlation(channel, torch.roll(channel, -1, dims=-2)),
        "next_nn_x2": correlation(channel, torch.roll(channel, -2, dims=-1)),
        "next_nn_y2": correlation(channel, torch.roll(channel, -2, dims=-2)),
        "diag_1_1": correlation(channel, torch.roll(torch.roll(channel, -1, dims=-1), -1, dims=-2)),
    }


def covariance_and_correlation(d: torch.Tensor) -> tuple[list[list[float]], list[list[float]]]:
    flat = d.permute(1, 0, 2, 3).reshape(3, -1).float()
    flat = flat - flat.mean(dim=1, keepdim=True)
    cov = flat @ flat.T / flat.shape[1]
    std = cov.diag().clamp_min(1e-12).sqrt()
    corr = cov / (std[:, None] * std[None, :])
    return cov.cpu().tolist(), corr.cpu().tolist()


def radial_power_spectrum_detail(d: torch.Tensor) -> dict[str, list[float]]:
    spectra = {}
    for i, name in enumerate(CHANNELS):
        x = d[:, i] - d[:, i].mean(dim=(-2, -1), keepdim=True)
        fft = torch.fft.fftn(x, dim=(-2, -1))
        power = (fft.real.square() + fft.imag.square()).mean(dim=0) / (x.shape[-2] * x.shape[-1])
        ly, lx = power.shape
        ky = torch.fft.fftfreq(ly, d=1.0, device=power.device) * ly
        kx = torch.fft.fftfreq(lx, d=1.0, device=power.device) * lx
        yy, xx = torch.meshgrid(ky, kx, indexing="ij")
        shell = torch.round(torch.sqrt(xx.square() + yy.square())).long()
        values = []
        for s in range(int(shell.max().item()) + 1):
            mask = shell == s
            if mask.any():
                values.append(float(power[mask].mean().item()))
        spectra[name] = values
    return spectra


def coarse_features(phi_c: torch.Tensor) -> dict[str, torch.Tensor]:
    grad_x = torch.roll(phi_c, -1, dims=-1) - phi_c
    grad_y = torch.roll(phi_c, -1, dims=-2) - phi_c
    return {
        "phi_c": phi_c,
        "phi_c2": phi_c.square(),
        "grad_phi_c2": grad_x.square() + grad_y.square(),
    }


def channel_diagnostics(d: torch.Tensor, phi_c: torch.Tensor) -> dict[str, object]:
    features = coarse_features(phi_c)
    per_channel = {}
    for i, name in enumerate(CHANNELS):
        ch = d[:, i]
        amp = ch.square()
        row = tensor_stats(ch)
        row.update(spatial_correlations(ch))
        row["corr_phi_c_detail_amp"] = correlation(features["phi_c"], amp)
        row["corr_phi_c2_detail_amp"] = correlation(features["phi_c2"], amp)
        row["corr_grad_phi_c2_detail_amp"] = correlation(features["grad_phi_c2"], amp)
        per_channel[name] = row
    cov, corr = covariance_and_correlation(d)
    return {
        "channels": per_channel,
        "cross_channel_covariance": cov,
        "cross_channel_correlation": corr,
        "within_block_detail_covariance": cov,
        "within_block_detail_correlation": corr,
        "power_spectrum": radial_power_spectrum_detail(d),
    }


def relative_error(value: float, target: float) -> float:
    return float("nan") if abs(target) < 1e-12 else (value - target) / target


def compare_to_true(diag: dict[str, object], true_diag: dict[str, object]) -> dict[str, object]:
    out = {"channels": {}, "cross_channel_covariance_abs_error": [], "cross_channel_correlation_abs_error": []}
    for name in CHANNELS:
        out["channels"][name] = {}
        for key, value in diag["channels"][name].items():
            target = true_diag["channels"][name][key]
            out["channels"][name][f"{key}_rel_error"] = relative_error(float(value), float(target))
            out["channels"][name][f"{key}_abs_error"] = float(value) - float(target)
    for key in ("cross_channel_covariance", "cross_channel_correlation"):
        errors = []
        for row, true_row in zip(diag[key], true_diag[key]):
            errors.append([float(abs(a - b)) for a, b in zip(row, true_row)])
        out[f"{key}_abs_error"] = errors
    return out


def checkpoint_conditioning_metadata(state: dict[str, object]) -> tuple[str, int]:
    args_meta = state.get("args", {}) if isinstance(state, dict) else {}
    mode = state.get("conditioning_mode") or args_meta.get("conditioning_mode") or "basic"
    n_cond = int(state.get("n_conditioning_channels") or n_conditioning_channels(str(mode)))
    return str(mode), n_cond


def load_flow(path: Path, args: argparse.Namespace, device: torch.device) -> tuple[ConditionalDetailFlow, str, int]:
    if not path.exists():
        raise FileNotFoundError(f"missing checkpoint: {path}")
    state = torch.load(path, map_location=device, weights_only=False)
    args_meta = state.get("args", {}) if isinstance(state, dict) else {}
    mode, n_cond = checkpoint_conditioning_metadata(state)
    layers = int(args_meta.get("layers", args.layers))
    hidden = int(args_meta.get("hidden_channels", args.hidden_channels))
    depth = int(args_meta.get("cnn_depth", args.cnn_depth))
    flow = ConditionalDetailFlow(layers, hidden, depth, n_cond).to(device)
    flow.load_state_dict(state["model"])
    flow.eval()
    return flow, mode, n_cond


def max_abs_channel_error(diag: dict[str, object], true_diag: dict[str, object], key: str) -> float:
    return max(abs(float(diag["channels"][name][key]) - float(true_diag["channels"][name][key])) for name in CHANNELS)


def mean_abs_channel_error(diag: dict[str, object], true_diag: dict[str, object], key: str) -> float:
    return sum(abs(float(diag["channels"][name][key]) - float(true_diag["channels"][name][key])) for name in CHANNELS) / len(CHANNELS)


def write_report(path: Path, summary: dict[str, object]) -> None:
    true = summary["ensembles"]["true_details"]
    flows = {k: v for k, v in summary["ensembles"].items() if k != "true_details"}
    lines = [
        "# Detail Correlation Diagnostics",
        "",
        "Coarse fields are blocked from true 32x32 configurations at `kappa=0.31`, `lambda=1.0`. The same `phi_c` is used for true extracted details and flow-sampled details.",
        "",
        "Detail channels are reported as `HL`, `LH`, `HH` in the order used by the residual detail parameterization.",
        "",
        "## Channel Moments and Spatial Correlations",
        "",
        "| ensemble | channel | mean | std | skew | kurtosis | nn_x | nn_y | r2_x | r2_y | diag |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for ens_name, diag in summary["ensembles"].items():
        for ch in CHANNELS:
            row = diag["channels"][ch]
            lines.append(
                f"| {ens_name} | {ch} | {row['mean']:.6g} | {row['std']:.6g} | {row['skewness']:.6g} | "
                f"{row['kurtosis']:.6g} | {row['nn_x']:.6g} | {row['nn_y']:.6g} | "
                f"{row['next_nn_x2']:.6g} | {row['next_nn_y2']:.6g} | {row['diag_1_1']:.6g} |"
            )
    lines.extend(
        [
            "",
            "## Conditioning Correlations",
            "",
            "| ensemble | channel | corr(phi_c, d^2) | corr(phi_c^2, d^2) | corr(|grad phi_c|^2, d^2) |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for ens_name, diag in summary["ensembles"].items():
        for ch in CHANNELS:
            row = diag["channels"][ch]
            lines.append(
                f"| {ens_name} | {ch} | {row['corr_phi_c_detail_amp']:.6g} | "
                f"{row['corr_phi_c2_detail_amp']:.6g} | {row['corr_grad_phi_c2_detail_amp']:.6g} |"
            )
    lines.extend(["", "## Cross-Channel Correlation Matrices", ""])
    for ens_name, diag in summary["ensembles"].items():
        lines.extend([f"### {ens_name}", "", "| | HL | LH | HH |", "|---|---:|---:|---:|"])
        for ch, row in zip(CHANNELS, diag["cross_channel_correlation"]):
            lines.append(f"| {ch} | {row[0]:.6g} | {row[1]:.6g} | {row[2]:.6g} |")
        lines.append("")
    lines.extend(
        [
            "## Summary Errors vs True Details",
            "",
            "| flow | mean abs std error | mean abs nn error | mean abs amp-phi_c2 corr error | max cross-corr abs error |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, diag in flows.items():
        cross = torch.tensor(summary["comparisons"][name]["cross_channel_correlation_abs_error"])
        lines.append(
            f"| {name} | {mean_abs_channel_error(diag, true, 'std'):.6g} | "
            f"{0.5 * (mean_abs_channel_error(diag, true, 'nn_x') + mean_abs_channel_error(diag, true, 'nn_y')):.6g} | "
            f"{mean_abs_channel_error(diag, true, 'corr_phi_c2_detail_amp'):.6g} | {float(cross.max().item()):.6g} |"
        )
    answers = summary["answers"]
    lines.extend(
        [
            "",
            "## Main Answers",
            "",
            f"1. Are flow details too decorrelated spatially? {answers['spatial_decorrelation']}",
            f"2. Are detail variances wrong? {answers['variance_mismatch']}",
            f"3. Are cross-channel correlations missing? {answers['cross_channel_mismatch']}",
            f"4. Does the flow fail to condition amplitudes on coarse features? {answers['conditioning_mismatch']}",
            f"5. Does logwvar alpha=0.01 improve these correlations? {answers['logwvar_improvement']}",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def plot_outputs(path: Path, details: dict[str, torch.Tensor], summary: dict[str, object]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(path) as pdf:
        fig, axes = plt.subplots(3, 3, figsize=(11, 9))
        for row, ch in enumerate(CHANNELS):
            for ens_name, d in details.items():
                axes[row, 0].hist(d[:, row].flatten().cpu().numpy(), bins=50, density=True, alpha=0.35, label=ens_name)
            axes[row, 0].set_title(f"{ch} values")
            axes[row, 0].legend(fontsize=7)
            for ens_name, d in details.items():
                ps = summary["ensembles"][ens_name]["power_spectrum"][ch]
                axes[row, 1].plot(range(len(ps)), ps, marker="o", ms=2.5, label=ens_name)
            axes[row, 1].set_yscale("log")
            axes[row, 1].set_title(f"{ch} power")
            axes[row, 1].legend(fontsize=7)
            vals = [summary["ensembles"][name]["channels"][ch]["corr_phi_c2_detail_amp"] for name in details]
            axes[row, 2].bar(range(len(vals)), vals)
            axes[row, 2].set_xticks(range(len(vals)), list(details), rotation=25, ha="right")
            axes[row, 2].set_title(f"{ch}: corr(phi_c^2,d^2)")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, axes = plt.subplots(1, len(details), figsize=(4 * len(details), 3.6))
        if len(details) == 1:
            axes = [axes]
        for ax, (ens_name, diag) in zip(axes, summary["ensembles"].items()):
            im = ax.imshow(diag["cross_channel_correlation"], vmin=-1, vmax=1, cmap="coolwarm")
            ax.set_title(ens_name)
            ax.set_xticks(range(3), CHANNELS)
            ax.set_yticks(range(3), CHANNELS)
        fig.colorbar(im, ax=axes, shrink=0.8)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


@torch.no_grad()
def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
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
    n_eval = min(args.n_eval, len(phi32))
    phi32 = phi32[:n_eval].to(device)
    phi_c, d_true = average_block(phi32)
    details = {"true_details": d_true.cpu()}
    ensembles = {"true_details": channel_diagnostics(d_true.cpu(), phi_c.cpu())}
    checkpoints = {
        "baseline_reverse_kl": args.baseline_checkpoint,
        "logwvar_alpha_0p01": args.logwvar_checkpoint,
    }
    if args.physics_checkpoint.exists():
        checkpoints["physics_conditioned"] = args.physics_checkpoint
    checkpoint_metadata = {}
    for i, (name, path) in enumerate(checkpoints.items()):
        flow, mode, n_cond = load_flow(path, args, device)
        checkpoint_metadata[name] = {
            "checkpoint": str(path),
            "conditioning_mode": mode,
            "n_conditioning_channels": n_cond,
        }
        generator = torch.Generator(device=device).manual_seed(args.seed + 100 + i)
        cond = make_conditioning(phi_c, mode)
        d_flow, _ = flow.sample(cond, generator=generator)
        details[name] = d_flow.cpu()
        ensembles[name] = channel_diagnostics(d_flow.cpu(), phi_c.cpu())

    comparisons = {
        name: compare_to_true(diag, ensembles["true_details"])
        for name, diag in ensembles.items()
        if name != "true_details"
    }
    base = ensembles["baseline_reverse_kl"]
    lwv = ensembles["logwvar_alpha_0p01"]
    true = ensembles["true_details"]
    base_nn_err = 0.5 * (mean_abs_channel_error(base, true, "nn_x") + mean_abs_channel_error(base, true, "nn_y"))
    lwv_nn_err = 0.5 * (mean_abs_channel_error(lwv, true, "nn_x") + mean_abs_channel_error(lwv, true, "nn_y"))
    base_std_err = mean_abs_channel_error(base, true, "std")
    lwv_std_err = mean_abs_channel_error(lwv, true, "std")
    base_cond_err = mean_abs_channel_error(base, true, "corr_phi_c2_detail_amp")
    lwv_cond_err = mean_abs_channel_error(lwv, true, "corr_phi_c2_detail_amp")
    base_cross_err = float(torch.tensor(comparisons["baseline_reverse_kl"]["cross_channel_correlation_abs_error"]).max().item())
    lwv_cross_err = float(torch.tensor(comparisons["logwvar_alpha_0p01"]["cross_channel_correlation_abs_error"]).max().item())
    answers = {
        "spatial_decorrelation": (
            f"Baseline mean NN-correlation error is {base_nn_err:.6g}; logwvar is {lwv_nn_err:.6g}. "
            + ("The flow spatial correlations are substantially shifted from true." if base_nn_err > 0.03 else "No large spatial decorrelation is visible by this metric.")
        ),
        "variance_mismatch": (
            f"Baseline mean absolute std error is {base_std_err:.6g}; logwvar is {lwv_std_err:.6g}."
        ),
        "cross_channel_mismatch": (
            f"Max cross-channel correlation error is {base_cross_err:.6g} for baseline and {lwv_cross_err:.6g} for logwvar."
        ),
        "conditioning_mismatch": (
            f"Mean abs corr(phi_c^2,d^2) error is {base_cond_err:.6g} for baseline and {lwv_cond_err:.6g} for logwvar."
        ),
        "logwvar_improvement": (
            f"logwvar improves NN={lwv_nn_err < base_nn_err}, std={lwv_std_err < base_std_err}, "
            f"cross={lwv_cross_err < base_cross_err}, conditioning={lwv_cond_err < base_cond_err}."
        ),
    }
    summary = {
        "setup": {
            "fine_size": args.fine_size,
            "coarse_size": args.fine_size // 2,
            "kappa": args.kappa,
            "lambda": args.lam,
            "n_eval": n_eval,
            "baseline_checkpoint": str(args.baseline_checkpoint),
            "logwvar_checkpoint": str(args.logwvar_checkpoint),
            "physics_checkpoint": str(args.physics_checkpoint) if args.physics_checkpoint.exists() else None,
            "checkpoint_metadata": checkpoint_metadata,
        },
        "ensembles": ensembles,
        "comparisons": comparisons,
        "answers": answers,
    }
    (args.output_dir / "detail_correlation_diagnostics_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_report(args.output_dir / "detail_correlation_diagnostics_report.md", summary)
    plot_outputs(args.output_dir / "detail_correlation_diagnostics_plots.pdf", details, summary)
    print(f"wrote {args.output_dir / 'detail_correlation_diagnostics_summary.json'}")
    print(f"wrote {args.output_dir / 'detail_correlation_diagnostics_report.md'}")
    print(f"wrote {args.output_dir / 'detail_correlation_diagnostics_plots.pdf'}")
    print("baseline_nn_error", f"{base_nn_err:.6g}", "logwvar_nn_error", f"{lwv_nn_err:.6g}")
    print("baseline_std_error", f"{base_std_err:.6g}", "logwvar_std_error", f"{lwv_std_err:.6g}")


if __name__ == "__main__":
    main()
