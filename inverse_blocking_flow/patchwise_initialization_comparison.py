"""Compare patchwise A/R equilibration from different detail initializations."""

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

from inverse_blocking_flow.data import load_or_generate_fine_configs, make_paired_dataset
from inverse_blocking_flow.flow import ConditionalDetailFlow
from inverse_blocking_flow.haar import reconstruct_from_average_block
from inverse_blocking_flow.patchwise_ar import low_high_power
from inverse_blocking_flow.phi4 import Phi4Params, mean_phi2, nearest_neighbor_correlator, phi4_action


OBS = ["S_mean", "phi2", "NN_corr", "high_p_power"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fine-size", type=int, default=16)
    parser.add_argument("--lambda", dest="lam", type=float, default=1.0)
    parser.add_argument("--kappa-fine", type=float, default=0.31)
    parser.add_argument("--n-configs", type=int, default=512)
    parser.add_argument("--n-chains", type=int, default=64)
    parser.add_argument("--patch-size", type=int, default=2)
    parser.add_argument("--step-size", type=float, default=0.1)
    parser.add_argument("--n-sweeps", type=int, default=300)
    parser.add_argument("--data-path", type=Path, default=Path("inverse_blocking_flow/outputs_fine16/fine_configs.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("inverse_blocking_flow/outputs_fine16"))
    parser.add_argument("--checkpoint", type=Path, default=Path("inverse_blocking_flow/outputs_fine16/conditional_detail_flow_reverse_kl.pt"))
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--cnn-depth", type=int, default=3)
    parser.add_argument("--burn-in", type=int, default=100)
    parser.add_argument("--sample-interval", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--proposal-width", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=515151)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    return parser


def load_flow(args: argparse.Namespace, device: torch.device) -> ConditionalDetailFlow:
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"missing checkpoint: {args.checkpoint}")
    flow = ConditionalDetailFlow(args.layers, args.hidden_channels, args.cnn_depth).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    flow.load_state_dict(state["model"])
    flow.eval()
    return flow


def observables(phi: torch.Tensor, params: Phi4Params) -> dict[str, float]:
    action = phi4_action(phi, params)
    return {
        "S_mean": float(action.mean().item()),
        "S_std": float(action.std(unbiased=False).item()),
        "phi2": float(mean_phi2(phi).mean().item()),
        "NN_corr": float(nearest_neighbor_correlator(phi).mean().item()),
        "high_p_power": low_high_power(phi)["high_momentum_power"],
    }


def true_ranges(phi: torch.Tensor, params: Phi4Params) -> dict[str, dict[str, float]]:
    action = phi4_action(phi, params)
    per_config = {
        "S_mean": action,
        "phi2": mean_phi2(phi),
        "NN_corr": nearest_neighbor_correlator(phi),
    }
    # high-p power is an ensemble statistic; use a 5% relative band around it.
    high = low_high_power(phi)["high_momentum_power"]
    out = {}
    for key, values in per_config.items():
        mean = float(values.mean().item())
        std_mean = float(values.std(unbiased=False).item() / (values.numel() ** 0.5))
        out[key] = {"mean": mean, "lo": mean - 2.0 * std_mean, "hi": mean + 2.0 * std_mean}
    out["high_p_power"] = {"mean": high, "lo": high * 0.95, "hi": high * 1.05}
    return out


def distance_to_true(obs: dict[str, float], true: dict[str, dict[str, float]]) -> float:
    vals = []
    for key in OBS:
        denom = abs(true[key]["mean"])
        if denom > 1e-14:
            vals.append(abs(obs[key] - true[key]["mean"]) / denom)
    return float(sum(vals) / len(vals))


def first_sweep_in_range(history: list[dict[str, float]], true: dict[str, dict[str, float]]) -> int | None:
    for i, row in enumerate(history):
        if all(true[key]["lo"] <= row[key] <= true[key]["hi"] for key in OBS):
            return i
    return None


@torch.no_grad()
def run_one(
    name: str,
    d_init: torch.Tensor,
    phi_c: torch.Tensor,
    params: Phi4Params,
    args: argparse.Namespace,
    seed: int,
    true: dict[str, dict[str, float]],
) -> dict[str, object]:
    generator = torch.Generator(device=phi_c.device).manual_seed(seed)
    d = d_init.clone()
    coarse_y, coarse_x = phi_c.shape[-2:]
    phi = reconstruct_from_average_block(phi_c[:, 0], d)
    action = phi4_action(phi, params)
    history = []
    accept_history = []

    row = observables(phi.cpu(), params)
    row["distance_to_true"] = distance_to_true(row, true)
    row["patch_acceptance_rate"] = 0.0
    history.append(row)

    for _ in range(args.n_sweeps):
        accepts = 0
        proposals = 0
        for y0 in range(0, coarse_y, args.patch_size):
            for x0 in range(0, coarse_x, args.patch_size):
                y1 = min(y0 + args.patch_size, coarse_y)
                x1 = min(x0 + args.patch_size, coarse_x)
                d_new = d.clone()
                noise = torch.randn(d[:, :, y0:y1, x0:x1].shape, generator=generator, device=d.device)
                d_new[:, :, y0:y1, x0:x1] = d[:, :, y0:y1, x0:x1] + args.step_size * noise
                phi_new = reconstruct_from_average_block(phi_c[:, 0], d_new)
                action_new = phi4_action(phi_new, params)
                log_accept = -action_new + action
                log_u = torch.log(torch.rand(log_accept.shape, generator=generator, device=d.device))
                accept = log_u < log_accept
                if accept.any():
                    d[accept] = d_new[accept]
                    action[accept] = action_new[accept]
                accepts += int(accept.sum().item())
                proposals += int(accept.numel())
        phi = reconstruct_from_average_block(phi_c[:, 0], d)
        row = observables(phi.cpu(), params)
        acc = accepts / float(proposals)
        row["patch_acceptance_rate"] = acc
        row["distance_to_true"] = distance_to_true(row, true)
        history.append(row)
        accept_history.append(acc)

    return {
        "name": name,
        "initial": history[0],
        "final": history[-1],
        "mean_patch_acceptance": float(sum(accept_history) / len(accept_history)),
        "first_sweep_in_true_range": first_sweep_in_range(history, true),
        "history": history,
    }


def write_md(path: Path, results: dict[str, object]) -> None:
    true = results["true_ranges"]
    runs = results["runs"]
    lines = [
        "# Patchwise Initialization Comparison",
        "",
        f"Settings: `patch_size={results['patch_size']}`, `step_size={results['step_size']}`, `n_sweeps={results['n_sweeps']}`.",
        "",
        "True ranges use full fine-field observables. For `S_mean`, `phi2`, and `NN_corr`, the band is mean +/- 2 standard errors over true configurations. For high-p power, the band is +/- 5%.",
        "",
        "## True Target Bands",
        "",
        "| observable | mean | lo | hi |",
        "|---|---:|---:|---:|",
    ]
    for key in OBS:
        band = true[key]
        lines.append(f"| {key} | {band['mean']:.6g} | {band['lo']:.6g} | {band['hi']:.6g} |")
    lines.extend([
        "",
        "## Initial and Final Values",
        "",
        "| init | mean accept | first sweep in true range | initial distance | final distance | initial S | final S | initial phi2 | final phi2 | initial NN | final NN | initial high-p | final high-p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for run in runs:
        first = run["first_sweep_in_true_range"]
        first_text = "not reached" if first is None else str(first)
        ini, fin = run["initial"], run["final"]
        lines.append(
            f"| {run['name']} | {run['mean_patch_acceptance']:.6g} | {first_text} | "
            f"{ini['distance_to_true']:.6g} | {fin['distance_to_true']:.6g} | "
            f"{ini['S_mean']:.6g} | {fin['S_mean']:.6g} | "
            f"{ini['phi2']:.6g} | {fin['phi2']:.6g} | "
            f"{ini['NN_corr']:.6g} | {fin['NN_corr']:.6g} | "
            f"{ini['high_p_power']:.6g} | {fin['high_p_power']:.6g} |"
        )
    best_initial = min(runs, key=lambda r: r["initial"]["distance_to_true"])
    best_final = min(runs, key=lambda r: r["final"]["distance_to_true"])
    lines.extend([
        "",
        "## Main Answer",
        "",
        f"The closest initialization at sweep 0 is `{best_initial['name']}`. The closest final chain after patchwise A/R is `{best_final['name']}`.",
    ])
    if best_initial["name"] == "reverse_kl":
        lines.append("The learned reverse-KL inverse map reduces initial equilibration distance relative to Gaussian or zero-detail initialization for these observables.")
    else:
        lines.append("The learned reverse-KL inverse map does not reduce the initial equilibration distance in this run; patchwise A/R must compensate.")
    path.write_text("\n".join(lines) + "\n")


def plot(path: Path, results: dict[str, object]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = results["runs"]
    fig, axes = plt.subplots(3, 2, figsize=(11, 10))
    panels = ["S_mean", "phi2", "NN_corr", "high_p_power", "patch_acceptance_rate", "distance_to_true"]
    true = results["true_ranges"]
    for ax, key in zip(axes.ravel(), panels):
        for run in runs:
            vals = [row[key] for row in run["history"]]
            ax.plot(vals, label=run["name"])
        if key in true:
            ax.axhline(true[key]["mean"], color="black", lw=0.8)
            ax.axhspan(true[key]["lo"], true[key]["hi"], color="black", alpha=0.08)
        ax.set_title(key)
        ax.set_xlabel("sweep")
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


@torch.no_grad()
def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    params = Phi4Params(kappa=args.kappa_fine, lam=args.lam)
    phi_f = load_or_generate_fine_configs(
        args.data_path,
        n_configs=max(args.n_configs, args.n_chains),
        fine_size=args.fine_size,
        params=params,
        burn_in=args.burn_in,
        interval=args.sample_interval,
        batch_size=args.batch_size,
        proposal_width=args.proposal_width,
        seed=args.seed,
        device=args.device,
    )
    dataset = make_paired_dataset(phi_f)
    phi_c_all, _, true_phi_all = dataset.tensors
    idx = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(args.seed))[: args.n_chains]
    phi_c = phi_c_all[idx].to(device)
    true_phi = true_phi_all[idx].cpu()
    true = true_ranges(true_phi, params)

    flow = load_flow(args, device)
    flow_gen = torch.Generator(device=device).manual_seed(args.seed + 17)
    d_reverse, _, _, _ = flow.sample_with_decomposition(phi_c, generator=flow_gen)
    d_gaussian = torch.randn(d_reverse.shape, generator=torch.Generator(device=device).manual_seed(args.seed + 18), device=device)
    d_zero = torch.zeros_like(d_reverse)

    init = {
        "reverse_kl": d_reverse,
        "gaussian": d_gaussian,
        "zero_detail": d_zero,
    }
    runs = [
        run_one(name, d, phi_c, params, args, args.seed + 100, true)
        for name, d in init.items()
    ]
    result = {
        "fine_size": args.fine_size,
        "coarse_size": args.fine_size // 2,
        "kappa_fine": args.kappa_fine,
        "lambda": args.lam,
        "patch_size": args.patch_size,
        "step_size": args.step_size,
        "n_sweeps": args.n_sweeps,
        "n_chains": args.n_chains,
        "true_ranges": true,
        "runs": runs,
    }
    json_path = args.output_dir / "patchwise_initialization_comparison.json"
    md_path = args.output_dir / "patchwise_initialization_comparison.md"
    pdf_path = args.output_dir / "patchwise_initialization_comparison.pdf"
    json_path.write_text(json.dumps(result, indent=2) + "\n")
    write_md(md_path, result)
    plot(pdf_path, result)
    for run in runs:
        first = run["first_sweep_in_true_range"]
        print(run["name"], "initial_dist", f"{run['initial']['distance_to_true']:.6g}", "final_dist", f"{run['final']['distance_to_true']:.6g}", "mean_accept", f"{run['mean_patch_acceptance']:.6g}", "first_in_range", first)
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    print(f"wrote {pdf_path}")


if __name__ == "__main__":
    main()
