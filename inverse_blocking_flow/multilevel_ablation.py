"""Ablation diagnostics for the 8->16->32 multilevel inverse-blocking run."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/matplotlib")

from inverse_blocking_flow.data import load_or_generate_fine_configs
from inverse_blocking_flow.flow import ConditionalDetailFlow
from inverse_blocking_flow.haar import average_block, block_average, reconstruct_from_average_block
from inverse_blocking_flow.multilevel_inverse_blocking import (
    aggregate_abs_rel,
    ensemble_summary,
    logw_stats,
    radial_power_spectrum,
    rel_errors,
)
from inverse_blocking_flow.phi4 import Phi4Params, binder_cumulant, mean_phi2, nearest_neighbor_correlator, phi4_action


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fine-size", type=int, default=32)
    parser.add_argument("--lambda", dest="lam", type=float, default=1.0)
    parser.add_argument("--kappa-fine", type=float, default=0.31)
    parser.add_argument("--n-configs", type=int, default=512)
    parser.add_argument("--n-eval", type=int, default=256)
    parser.add_argument("--data-path", type=Path, default=Path("inverse_blocking_flow/outputs/fine_configs.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("inverse_blocking_flow/outputs"))
    parser.add_argument("--checkpoint-8-to-16", type=Path, default=Path("inverse_blocking_flow/outputs/multilevel_flow_8_to_16.pt"))
    parser.add_argument("--checkpoint-16-to-32", type=Path, default=Path("inverse_blocking_flow/outputs/multilevel_flow_16_to_32.pt"))
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--cnn-depth", type=int, default=3)
    parser.add_argument("--burn-in", type=int, default=200)
    parser.add_argument("--sample-interval", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--proposal-width", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=979797)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    return parser


def load_flow(path: Path, args: argparse.Namespace, device: torch.device) -> ConditionalDetailFlow:
    if not path.exists():
        raise FileNotFoundError(f"missing checkpoint: {path}")
    flow = ConditionalDetailFlow(args.layers, args.hidden_channels, args.cnn_depth).to(device)
    state = torch.load(path, map_location=device, weights_only=False)
    flow.load_state_dict(state["model"])
    flow.eval()
    return flow


def low_high_power(phi: torch.Tensor) -> dict[str, float]:
    centered = phi - phi.mean(dim=(-2, -1), keepdim=True)
    fft = torch.fft.fftn(centered, dim=(-2, -1))
    power = (fft.real.square() + fft.imag.square()) / (phi.shape[-2] * phi.shape[-1])
    ly, lx = phi.shape[-2:]
    ky = torch.fft.fftfreq(ly, d=1.0, device=phi.device) * ly
    kx = torch.fft.fftfreq(lx, d=1.0, device=phi.device) * lx
    yy, xx = torch.meshgrid(ky, kx, indexing="ij")
    radius = torch.sqrt(xx.square() + yy.square())
    low_mask = (radius > 0) & (radius <= 2)
    high_mask = radius >= 0.5 * float(radius.max().item())
    return {
        "low_p_power": float(power[:, low_mask].mean().item()),
        "high_p_power": float(power[:, high_mask].mean().item()),
    }


def susceptibility(phi: torch.Tensor) -> torch.Tensor:
    volume = phi.shape[-2] * phi.shape[-1]
    return volume * phi.mean(dim=(-2, -1)).square()


def summary_16(phi: torch.Tensor, params: Phi4Params) -> dict[str, float]:
    action = phi4_action(phi, params)
    out = {
        "S_mean": float(action.mean().item()),
        "S_std": float(action.std(unbiased=False).item()),
        "phi2": float(mean_phi2(phi).mean().item()),
        "binder": float(binder_cumulant(phi).item()),
        "NN_corr": float(nearest_neighbor_correlator(phi).mean().item()),
        "susceptibility": float(susceptibility(phi).mean().item()),
    }
    out.update(low_high_power(phi))
    return out


def rel_summary(summary: dict[str, float], target: dict[str, float]) -> dict[str, float]:
    return {
        key: float("nan") if abs(target[key]) < 1e-14 else (summary[key] - target[key]) / target[key]
        for key in summary
        if key in target
    }


def agg_rel(summary: dict[str, float], target: dict[str, float]) -> float:
    keys = ["S_mean", "S_std", "phi2", "NN_corr", "high_p_power"]
    rel = rel_summary(summary, target)
    return sum(abs(rel[key]) for key in keys) / len(keys)


def conditioning_shift(phi16_true: torch.Tensor, phi16_rec: torch.Tensor) -> dict[str, float | dict[str, float]]:
    diff = phi16_rec - phi16_true
    true_norm = phi16_true.flatten(1).norm(dim=1).clamp_min(1e-12)
    diff_norm = diff.flatten(1).norm(dim=1)
    mean_abs = diff.abs().mean(dim=(-2, -1))
    max_abs = diff.abs().amax(dim=(-2, -1))
    return {
        "rmse": float(diff.square().mean().sqrt().item()),
        "relative_l2_mean": float((diff_norm / true_norm).mean().item()),
        "relative_l2_std": float((diff_norm / true_norm).std(unbiased=False).item()),
        "mean_abs_difference": float(mean_abs.mean().item()),
        "max_abs_difference_mean": float(max_abs.mean().item()),
    }


def histogram_values(phi: torch.Tensor) -> list[float]:
    hist = torch.histc(phi.detach().cpu(), bins=50, min=-3.0, max=3.0)
    hist = hist / hist.sum().clamp_min(1.0)
    return [float(x) for x in hist]


def plot_outputs(output_dir: Path, ensembles32: dict[str, torch.Tensor], phi16_true: torch.Tensor, phi16_rec: torch.Tensor, params: Phi4Params) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5))
    ax = axes[0, 0]
    for name, phi in ensembles32.items():
        ax.hist(phi4_action(phi, params).detach().cpu().numpy(), bins=40, density=True, alpha=0.4, label=name)
    ax.set_xlabel("S_32")
    ax.set_ylabel("density")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    for name, phi in ensembles32.items():
        k, p = radial_power_spectrum(phi)
        ax.plot(k.numpy(), p.numpy(), marker="o", ms=3, label=name)
    ax.set_yscale("log")
    ax.set_xlabel("|p| shell")
    ax.set_ylabel("<|phi(p)|^2>")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.hist(phi16_true.detach().cpu().flatten().numpy(), bins=50, density=True, alpha=0.5, label="true phi16")
    ax.hist(phi16_rec.detach().cpu().flatten().numpy(), bins=50, density=True, alpha=0.5, label="rec phi16")
    ax.set_xlabel("phi16 value")
    ax.set_ylabel("density")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    labels = list(ensembles32)
    x = torch.arange(len(labels)).numpy()
    ax.bar(x - 0.18, [float(mean_phi2(ensembles32[n]).mean().item()) for n in labels], width=0.36, label="phi2")
    ax.bar(x + 0.18, [low_high_power(ensembles32[n])["high_p_power"] for n in labels], width=0.36, label="high-p")
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "multilevel_ablation_plots.pdf")
    plt.close(fig)


def write_report(path: Path, summary: dict[str, object]) -> None:
    lines = [
        "# Multilevel Ablation",
        "",
        "This ablation identifies whether two-step degradation comes from the first-level 8->16 lift or from applying the 16->32 flow off its true conditioning distribution.",
        "",
        "## 32x32 Cases",
        "",
        "| case | S mean | S std | phi2 | Binder | NN corr | susceptibility | low-p | high-p | agg rel err | logw std | ESS/N |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, case in summary["cases32"].items():
        density = summary["density_diagnostics"].get(name, {})
        logw = density.get("logw", {})
        lines.append(
            f"| {name} | {case['S_mean']:.6g} | {case['S_std']:.6g} | {case['phi2']:.6g} | {case['binder']:.6g} | "
            f"{case['NN_corr']:.6g} | {case['susceptibility']:.6g} | {case['low_p_power']:.6g} | {case['high_p_power']:.6g} | "
            f"{case['aggregate_abs_rel_error']:.6g} | {logw.get('std', float('nan')):.6g} | {logw.get('ess_over_n', float('nan')):.6g} |"
        )
    first = summary["first_level_only"]
    shift = summary["conditioning_shift"]
    lines.extend(
        [
            "",
            "## First-Level 8->16 Quality",
            "",
            "| ensemble | S mean | S std | phi2 | Binder | NN corr | susceptibility | low-p | high-p | agg rel err |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, item, agg in [
        ("true_phi16", first["true_phi16"], 0.0),
        ("reconstructed_phi16", first["reconstructed_phi16"], first["aggregate_abs_rel_error"]),
    ]:
        lines.append(
            f"| {name} | {item['S_mean']:.6g} | {item['S_std']:.6g} | {item['phi2']:.6g} | {item['binder']:.6g} | "
            f"{item['NN_corr']:.6g} | {item['susceptibility']:.6g} | {item['low_p_power']:.6g} | {item['high_p_power']:.6g} | {agg:.6g} |"
        )
    lines.extend(
        [
            "",
            "Relative errors for reconstructed_phi16 versus true_phi16:",
            "",
            "| observable | relative error |",
            "|---|---:|",
        ]
    )
    for key in ["S_mean", "S_std", "phi2", "binder", "NN_corr", "susceptibility", "low_p_power", "high_p_power"]:
        lines.append(f"| {key} | {first['relative_errors'][key]:.6g} |")
    lines.extend(
        [
            "",
            f"- aggregate relative error at 16x16: `{first['aggregate_abs_rel_error']:.6g}`",
            f"- phi2 rel error: `{first['relative_errors']['phi2']:.6g}`",
            f"- NN rel error: `{first['relative_errors']['NN_corr']:.6g}`",
            f"- high-p rel error: `{first['relative_errors']['high_p_power']:.6g}`",
            f"- block-to-8 consistency max error: `{first['block_to_8_consistency_max_error']:.6g}`",
            f"- phi16 value histogram L1 distance: `{first['phi16_value_histogram_l1']:.6g}`",
            "",
            "## Conditioning Shift phi16_true -> phi16_rec",
            "",
            f"- RMSE: `{shift['rmse']:.6g}`",
            f"- relative L2 mean/std: `{shift['relative_l2_mean']:.6g}` / `{shift['relative_l2_std']:.6g}`",
            f"- mean abs difference: `{shift['mean_abs_difference']:.6g}`",
            f"- mean max abs difference per configuration: `{shift['max_abs_difference_mean']:.6g}`",
            "",
            "## Answers",
            "",
            f"- Main diagnosis: {summary['answers']['main_diagnosis']}",
            f"- Does true-first-level then second match one-step? {summary['answers']['true_first_matches_one_step']}",
            f"- Mixed-detail test: {summary['mixed_detail_test']['status']} ({summary['mixed_detail_test']['explanation']})",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


@torch.no_grad()
def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    params = Phi4Params(kappa=args.kappa_fine, lam=args.lam)
    flow16 = load_flow(args.checkpoint_8_to_16, args, device)
    flow32 = load_flow(args.checkpoint_16_to_32, args, device)

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
    phi16, d32_true = average_block(phi32)
    phi8, d16_true = average_block(phi16)
    n_eval = min(args.n_eval, len(phi32))
    phi32 = phi32[:n_eval].to(device)
    phi16 = phi16[:n_eval].to(device)
    phi8 = phi8[:n_eval].to(device)
    d32_true = d32_true[:n_eval].to(device)
    d16_true = d16_true[:n_eval].to(device)
    gen = torch.Generator(device=device).manual_seed(args.seed + 17)

    # A. One-step with true phi16 conditioning.
    d32_a, _, _, logq32_a = flow32.sample_with_decomposition(phi16.unsqueeze(1), generator=gen)
    phi32_a = reconstruct_from_average_block(phi16, d32_a)
    logw_a = -phi4_action(phi32_a, params) - logq32_a

    # B/C. Reconstruct phi16 from q16, then feed q32 with generated phi16.
    d16_b, _, _, logq16_b = flow16.sample_with_decomposition(phi8.unsqueeze(1), generator=gen)
    phi16_rec = reconstruct_from_average_block(phi8, d16_b)
    d32_b, _, _, logq32_b = flow32.sample_with_decomposition(phi16_rec.unsqueeze(1), generator=gen)
    phi32_b = reconstruct_from_average_block(phi16_rec, d32_b)
    logw_b = -phi4_action(phi32_b, params) - (logq16_b + logq32_b)

    # D. True first level exactly reconstructs true phi16, then use q32.
    phi16_d = reconstruct_from_average_block(phi8, d16_true)
    d32_d, _, _, logq32_d = flow32.sample_with_decomposition(phi16_d.unsqueeze(1), generator=gen)
    phi32_d = reconstruct_from_average_block(phi16_d, d32_d)
    logw_d = -phi4_action(phi32_d, params) - logq32_d

    true32_summary = ensemble_summary(phi32.cpu(), params)
    cases32_tensors = {
        "true_32": phi32.cpu(),
        "A_one_step_true_phi16": phi32_a.cpu(),
        "B_second_level_on_reconstructed_phi16": phi32_b.cpu(),
        "D_true_first_level_then_second": phi32_d.cpu(),
    }
    cases32 = {}
    for name, phi in cases32_tensors.items():
        s = ensemble_summary(phi, params)
        s["relative_errors"] = rel_errors(s, true32_summary)
        s["aggregate_abs_rel_error"] = aggregate_abs_rel(s, true32_summary)
        cases32[name] = s

    true16_summary = summary_16(phi16.cpu(), params)
    rec16_summary = summary_16(phi16_rec.cpu(), params)
    first_level = {
        "true_phi16": true16_summary,
        "reconstructed_phi16": rec16_summary,
        "relative_errors": rel_summary(rec16_summary, true16_summary),
        "aggregate_abs_rel_error": agg_rel(rec16_summary, true16_summary),
        "block_to_8_consistency_max_error": float((block_average(phi16_rec) - phi8).abs().max().item()),
        "phi16_value_histogram_true": histogram_values(phi16),
        "phi16_value_histogram_reconstructed": histogram_values(phi16_rec),
    }
    first_level["phi16_value_histogram_l1"] = float(
        sum(
            abs(a - b)
            for a, b in zip(
                first_level["phi16_value_histogram_true"],
                first_level["phi16_value_histogram_reconstructed"],
            )
        )
    )

    density = {
        "A_one_step_true_phi16": {"logw": logw_stats(logw_a)},
        "B_second_level_on_reconstructed_phi16": {"logw": logw_stats(logw_b)},
        "D_true_first_level_then_second": {"logw": logw_stats(logw_d)},
    }
    one_err = cases32["A_one_step_true_phi16"]["aggregate_abs_rel_error"]
    two_err = cases32["B_second_level_on_reconstructed_phi16"]["aggregate_abs_rel_error"]
    d_err = cases32["D_true_first_level_then_second"]["aggregate_abs_rel_error"]
    true_first_matches = abs(d_err - one_err) / max(one_err, 1e-12) < 0.25
    if first_level["aggregate_abs_rel_error"] > 0.1:
        diagnosis = "q16 produces a visibly shifted phi16_rec; first-level error is a major contributor."
    elif two_err > one_err * 1.25:
        diagnosis = "q16 marginal errors are moderate, but q32 is sensitive to generated phi16_rec conditioning shift."
    else:
        diagnosis = "two-step degradation is not clearly isolated by this ablation."

    summary = {
        "fine_size": args.fine_size,
        "middle_size": 16,
        "coarse_size": 8,
        "kappa_fine": args.kappa_fine,
        "lambda": args.lam,
        "n_eval": n_eval,
        "cases32": cases32,
        "first_level_only": first_level,
        "conditioning_shift": conditioning_shift(phi16, phi16_rec),
        "density_diagnostics": density,
        "mixed_detail_test": {
            "status": "skipped",
            "explanation": "true d32 is defined relative to true phi16. Combining it with generated phi16_rec would reconstruct phi16_rec + chi_true32 and no longer represent the original fiber or a meaningful conditional sample.",
        },
        "answers": {
            "main_diagnosis": diagnosis,
            "true_first_matches_one_step": true_first_matches,
            "one_step_aggregate_error": one_err,
            "two_step_aggregate_error": two_err,
            "true_first_then_second_aggregate_error": d_err,
        },
    }
    (args.output_dir / "multilevel_ablation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_report(args.output_dir / "multilevel_ablation_report.md", summary)
    plot_outputs(args.output_dir, cases32_tensors, phi16.cpu(), phi16_rec.cpu(), params)
    print(f"wrote {args.output_dir / 'multilevel_ablation_summary.json'}")
    print(f"wrote {args.output_dir / 'multilevel_ablation_report.md'}")
    print(f"wrote {args.output_dir / 'multilevel_ablation_plots.pdf'}")
    print("first_level_agg_error", f"{first_level['aggregate_abs_rel_error']:.6g}")
    print("one_step_agg_error", f"{one_err:.6g}")
    print("two_step_agg_error", f"{two_err:.6g}")
    print("true_first_then_second_agg_error", f"{d_err:.6g}")
    print("diagnosis", diagnosis)


if __name__ == "__main__":
    main()
