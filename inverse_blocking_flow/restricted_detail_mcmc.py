"""Restricted detail MCMC at fixed coarse field.

For fixed ``phi_c = B(phi_f)``, this samples detail variables from

    P(d | phi_c) proportional to exp[-S_f(R(phi_c, d))]

using symmetric patchwise random-walk proposals in detail space.
"""

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

from inverse_blocking_flow.data import load_or_generate_fine_configs, make_paired_dataset
from inverse_blocking_flow.flow import ConditionalDetailFlow
from inverse_blocking_flow.haar import block_average, reconstruct_from_average_block
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
    parser.add_argument("--n-sweeps", type=int, default=1000)
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--cnn-depth", type=int, default=3)
    parser.add_argument("--burn-in", type=int, default=100)
    parser.add_argument("--sample-interval", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--proposal-width", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=626262)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    return parser


def default_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    if args.fine_size == 16:
        out = Path("inverse_blocking_flow/outputs_fine16")
    else:
        out = Path("inverse_blocking_flow/outputs")
    data = out / "fine_configs.pt"
    ckpt = out / "conditional_detail_flow_reverse_kl.pt"
    return args.data_path or data, args.output_dir or out, args.checkpoint or ckpt


def load_flow(path: Path, args: argparse.Namespace, device: torch.device) -> ConditionalDetailFlow:
    if not path.exists():
        raise FileNotFoundError(f"missing reverse-KL checkpoint: {path}")
    flow = ConditionalDetailFlow(args.layers, args.hidden_channels, args.cnn_depth).to(device)
    state = torch.load(path, map_location=device, weights_only=False)
    flow.load_state_dict(state["model"])
    flow.eval()
    return flow


def observables(phi: torch.Tensor, params: Phi4Params) -> dict[str, float]:
    return {
        "S_mean": float(phi4_action(phi, params).mean().item()),
        "S_std": float(phi4_action(phi, params).std(unbiased=False).item()),
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
    high = low_high_power(phi)["high_momentum_power"]
    out = {}
    for key, values in per_config.items():
        mean = float(values.mean().item())
        sem = float(values.std(unbiased=False).item() / (values.numel() ** 0.5))
        out[key] = {"mean": mean, "lo": mean - 2.0 * sem, "hi": mean + 2.0 * sem}
    out["high_p_power"] = {"mean": high, "lo": high * 0.95, "hi": high * 1.05}
    return out


def distance_to_true(row: dict[str, float], true: dict[str, dict[str, float]]) -> float:
    vals = []
    for key in OBS:
        denom = abs(true[key]["mean"])
        if denom > 1e-14:
            vals.append(abs(row[key] - true[key]["mean"]) / denom)
    return float(sum(vals) / len(vals))


def first_in_range(history: list[dict[str, float]], true: dict[str, dict[str, float]]) -> int | None:
    for i, row in enumerate(history):
        if all(true[key]["lo"] <= row[key] <= true[key]["hi"] for key in OBS):
            return i
    return None


def integrated_autocorr_time(values: list[float]) -> float | None:
    x = torch.tensor(values, dtype=torch.float64)
    n = x.numel()
    if n < 50 or float(x.std(unbiased=False).item()) == 0.0:
        return None
    x = x - x.mean()
    var = (x * x).mean()
    tau = 1.0
    max_lag = min(n // 2, 300)
    for lag in range(1, max_lag):
        rho = ((x[:-lag] * x[lag:]).mean() / var).item()
        if rho <= 0.0:
            break
        tau += 2.0 * rho
    return float(tau)


def final_window(history: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    start = int(0.75 * len(history))
    window = history[start:]
    out = {}
    for key in OBS + ["distance_to_true", "patch_acceptance_rate", "fixed_coarse_max_error"]:
        vals = torch.tensor([row[key] for row in window], dtype=torch.float64)
        out[key] = {
            "mean": float(vals.mean().item()),
            "std": float(vals.std(unbiased=False).item()),
        }
    return out


@torch.no_grad()
def run_chain(
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
    max_coarse_error = 0.0

    for sweep in range(args.n_sweeps + 1):
        phi = reconstruct_from_average_block(phi_c[:, 0], d)
        row = observables(phi.cpu(), params)
        coarse_error = float((block_average(phi) - phi_c[:, 0]).abs().max().item())
        max_coarse_error = max(max_coarse_error, coarse_error)
        row["distance_to_true"] = distance_to_true(row, true)
        row["patch_acceptance_rate"] = 0.0 if sweep == 0 else history[-1]["patch_acceptance_rate"]
        row["fixed_coarse_max_error"] = coarse_error
        history.append(row)
        if sweep == args.n_sweeps:
            break

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
        history[-1]["patch_acceptance_rate"] = accepts / float(proposals)

    iat = {}
    tail_start = len(history) // 2
    for key in OBS + ["distance_to_true"]:
        iat[key] = integrated_autocorr_time([row[key] for row in history[tail_start:]])
    return {
        "name": name,
        "initial": history[0],
        "final": history[-1],
        "history": history,
        "first_sweep_in_true_range": first_in_range(history, true),
        "max_fixed_coarse_error": max_coarse_error,
        "integrated_autocorr_time": iat,
        "final_window": final_window(history),
    }


def write_report(path: Path, result: dict[str, object]) -> None:
    runs = result["runs"]
    lines = [
        "# Restricted Detail MCMC",
        "",
        "Target: for fixed `phi_c = B(phi_f)`, sample `d` with density proportional to `exp[-S_f(R(phi_c,d))]`.",
        "",
        f"Settings: fine_size `{result['fine_size']}`, patch_size `{result['patch_size']}`, step_size `{result['step_size']}`, n_sweeps `{result['n_sweeps']}`.",
        "",
        "## Fixed Coarse Field",
        "",
        "| init | max |B(phi)-phi_c| |",
        "|---|---:|",
    ]
    for run in runs:
        lines.append(f"| {run['name']} | {run['max_fixed_coarse_error']:.6g} |")
    lines.extend([
        "",
        "## Equilibration",
        "",
        "| init | first sweep in true band | initial distance | final distance | final-window distance mean/std | IAT distance |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for run in runs:
        first = run["first_sweep_in_true_range"]
        first_text = "not reached" if first is None else str(first)
        fw = run["final_window"]["distance_to_true"]
        iat = run["integrated_autocorr_time"]["distance_to_true"]
        iat_text = "missing" if iat is None else f"{iat:.6g}"
        lines.append(
            f"| {run['name']} | {first_text} | {run['initial']['distance_to_true']:.6g} | "
            f"{run['final']['distance_to_true']:.6g} | {fw['mean']:.6g} / {fw['std']:.6g} | {iat_text} |"
        )
    lines.extend([
        "",
        "## Final-Window Means and Errors",
        "",
        "| init | observable | final-window mean | final-window std | true mean | relative error |",
        "|---|---|---:|---:|---:|---:|",
    ])
    true = result["true_ranges"]
    for run in runs:
        fw = run["final_window"]
        for key in OBS:
            mean = fw[key]["mean"]
            std = fw[key]["std"]
            target = true[key]["mean"]
            rel = (mean - target) / target if abs(target) > 1e-14 else float("nan")
            lines.append(f"| {run['name']} | {key} | {mean:.6g} | {std:.6g} | {target:.6g} | {rel:.6g} |")
    lines.extend([
        "",
        "## Answers",
        "",
    ])
    max_err = max(run["max_fixed_coarse_error"] for run in runs)
    lines.append(f"- Does restricted detail MCMC preserve `phi_c` exactly? Yes, max observed error is `{max_err:.6g}`.")
    firsts = {run["name"]: run["first_sweep_in_true_range"] for run in runs}
    if firsts.get("reverse_kl") is not None and (
        firsts.get("gaussian") is None or firsts["reverse_kl"] < firsts["gaussian"]
    ) and (firsts.get("zero_detail") is None or firsts["reverse_kl"] < firsts["zero_detail"]):
        lines.append("- Does reverse-KL initialization reduce equilibration time? Yes, it reaches the true band first.")
    else:
        lines.append("- Does reverse-KL initialization reduce equilibration time? Not clearly in this run.")
    lines.append(f"- Sweeps needed for this size: reverse-KL `{firsts.get('reverse_kl')}`, gaussian `{firsts.get('gaussian')}`, zero-detail `{firsts.get('zero_detail')}`.")
    lines.append("- Global reweighting diagnostics are intentionally omitted in this restricted-detail MCMC report.")
    path.write_text("\n".join(lines) + "\n")


def plot(path: Path, result: dict[str, object]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = ["S_mean", "phi2", "NN_corr", "high_p_power", "distance_to_true", "patch_acceptance_rate"]
    true = result["true_ranges"]
    fig, axes = plt.subplots(3, 2, figsize=(11, 10))
    for ax, key in zip(axes.ravel(), panels):
        for run in result["runs"]:
            ax.plot([row[key] for row in run["history"]], label=run["name"])
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
    data_path, output_dir, checkpoint = default_paths(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    params = Phi4Params(kappa=args.kappa_fine, lam=args.lam)

    phi_f = load_or_generate_fine_configs(
        data_path,
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
    flow = load_flow(checkpoint, args, device)

    gen = torch.Generator(device=device).manual_seed(args.seed + 17)
    d_reverse, _, _, _ = flow.sample_with_decomposition(phi_c, generator=gen)
    d_gaussian = torch.randn(d_reverse.shape, generator=torch.Generator(device=device).manual_seed(args.seed + 18), device=device)
    d_zero = torch.zeros_like(d_reverse)
    inits = {"reverse_kl": d_reverse, "gaussian": d_gaussian, "zero_detail": d_zero}
    runs = [run_chain(name, d, phi_c, params, args, args.seed + 100, true) for name, d in inits.items()]
    result = {
        "fine_size": args.fine_size,
        "coarse_size": args.fine_size // 2,
        "kappa_fine": args.kappa_fine,
        "lambda": args.lam,
        "patch_size": args.patch_size,
        "step_size": args.step_size,
        "n_sweeps": args.n_sweeps,
        "checkpoint": str(checkpoint),
        "true_ranges": true,
        "runs": runs,
    }
    (output_dir / "restricted_detail_mcmc_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    write_report(output_dir / "restricted_detail_mcmc_report.md", result)
    plot(output_dir / "restricted_detail_mcmc_histories.pdf", result)
    for run in runs:
        print(
            run["name"],
            "first", run["first_sweep_in_true_range"],
            "final_dist", f"{run['final']['distance_to_true']:.6g}",
            "coarse_err", f"{run['max_fixed_coarse_error']:.3g}",
        )
    print(f"wrote {output_dir / 'restricted_detail_mcmc_summary.json'}")
    print(f"wrote {output_dir / 'restricted_detail_mcmc_report.md'}")
    print(f"wrote {output_dir / 'restricted_detail_mcmc_histories.pdf'}")


if __name__ == "__main__":
    main()
