"""Patchwise random-walk A/R correction for detail variables at fixed phi_c."""

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
from inverse_blocking_flow.haar import reconstruct_from_average_block
from inverse_blocking_flow.phi4 import (
    Phi4Params,
    binder_cumulant,
    mean_phi2,
    nearest_neighbor_correlator,
    phi4_action,
    summarize_observables,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fine-size", type=int, default=16)
    parser.add_argument("--lambda", dest="lam", type=float, default=1.0)
    parser.add_argument("--kappa-fine", type=float, default=0.31)
    parser.add_argument("--n-configs", type=int, default=512)
    parser.add_argument("--n-chains", type=int, default=64)
    parser.add_argument("--n-sweeps", type=int, default=100)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--step-size", type=float, default=0.2)
    parser.add_argument("--data-path", type=Path, default=Path("inverse_blocking_flow/outputs_fine16/fine_configs.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("inverse_blocking_flow/outputs_fine16"))
    parser.add_argument("--checkpoint", type=Path, default=Path("inverse_blocking_flow/outputs_fine16/conditional_detail_flow_reverse_kl.pt"))
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--hidden-channels", type=int, default=48)
    parser.add_argument("--cnn-depth", type=int, default=3)
    parser.add_argument("--burn-in", type=int, default=200)
    parser.add_argument("--sample-interval", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--proposal-width", type=float, default=1.0)
    parser.add_argument("--ar-diagnostics", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=13579)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    return parser


def tensor_stats(x: torch.Tensor) -> dict[str, float]:
    x = x.detach().float().cpu()
    return {
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
    }


def susceptibility(phi: torch.Tensor) -> torch.Tensor:
    volume = phi.shape[-2] * phi.shape[-1]
    return volume * phi.mean(dim=(-2, -1)).square()


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
        "low_momentum_power": float(power[:, low_mask].mean().item()),
        "high_momentum_power": float(power[:, high_mask].mean().item()),
    }


def ensemble_summary(phi: torch.Tensor, params: Phi4Params) -> dict[str, object]:
    action = phi4_action(phi, params)
    return {
        "S_f": tensor_stats(action),
        "mean_phi2": float(mean_phi2(phi).mean().item()),
        "binder": float(binder_cumulant(phi).item()),
        "nearest_neighbor_correlator": float(nearest_neighbor_correlator(phi).mean().item()),
        "susceptibility": float(susceptibility(phi).mean().item()),
        **low_high_power(phi),
    }


def load_reverse_flow(args: argparse.Namespace, device: torch.device) -> ConditionalDetailFlow:
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"missing reverse-KL checkpoint: {args.checkpoint}")
    flow = ConditionalDetailFlow(args.layers, args.hidden_channels, args.cnn_depth).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    flow.load_state_dict(state["model"])
    flow.eval()
    return flow


@torch.no_grad()
def global_ar_reference(
    path: Path | None,
    output_dir: Path,
    params: Phi4Params,
) -> dict[str, object] | None:
    candidates = []
    if path is not None:
        candidates.append(path)
    candidates.extend(
        [
            output_dir / "ar_diagnostics_reverse_kl.json",
            output_dir / "ar_diagnostics_mixed.json",
            output_dir / "ar_diagnostics.json",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            data = json.loads(candidate.read_text())
            if "flow_after_ar" in data:
                return {
                    "source": str(candidate),
                    "acceptance_rate": data.get("acceptance_rate"),
                    "kept_window_acceptance_rate": data.get("kept_window_acceptance_rate"),
                    "after_ar_summary": data["flow_after_ar"],
                }
            if "global_ar" in data and "reverse_kl" in data["global_ar"]:
                return {"source": str(candidate), **data["global_ar"]["reverse_kl"]}
    return None


@torch.no_grad()
def run_patchwise_ar(
    phi_c: torch.Tensor,
    d: torch.Tensor,
    params: Phi4Params,
    *,
    patch_size: int,
    step_size: float,
    n_sweeps: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, dict[str, float], list[float]]:
    if patch_size <= 0:
        raise ValueError("--patch-size must be positive")
    if step_size <= 0:
        raise ValueError("--step-size must be positive")
    coarse_y, coarse_x = phi_c.shape[-2:]
    phi = reconstruct_from_average_block(phi_c[:, 0], d)
    action = phi4_action(phi, params)
    accepts = 0
    proposals = 0
    action_trace = [float(action.mean().cpu().item())]

    for _ in range(n_sweeps):
        for y0 in range(0, coarse_y, patch_size):
            for x0 in range(0, coarse_x, patch_size):
                y1 = min(y0 + patch_size, coarse_y)
                x1 = min(x0 + patch_size, coarse_x)
                d_new = d.clone()
                noise = torch.randn(d[:, :, y0:y1, x0:x1].shape, generator=generator, device=d.device)
                d_new[:, :, y0:y1, x0:x1] = d[:, :, y0:y1, x0:x1] + step_size * noise
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
        action_trace.append(float(action.mean().cpu().item()))

    return d, {"patch_acceptance_rate": accepts / float(proposals), "patch_proposals": float(proposals)}, action_trace


def plot_diagnostics(
    path: Path,
    ensembles: dict[str, torch.Tensor],
    action_trace: list[float],
    params: Phi4Params,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.0))
    ax = axes[0, 0]
    for name, phi in ensembles.items():
        ax.hist(phi4_action(phi, params).detach().cpu().numpy(), bins=35, density=True, alpha=0.4, label=name)
    ax.set_xlabel("S_f")
    ax.set_ylabel("density")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(action_trace, marker="o", markersize=2)
    ax.set_xlabel("patch sweep")
    ax.set_ylabel("mean S_f")

    ax = axes[1, 0]
    names = list(ensembles)
    phi2 = [float(mean_phi2(ensembles[name]).mean().item()) for name in names]
    binder = [float(binder_cumulant(ensembles[name]).item()) for name in names]
    x = torch.arange(len(names)).numpy()
    ax.bar(x - 0.18, phi2, width=0.36, label="phi^2")
    ax.bar(x + 0.18, binder, width=0.36, label="Binder")
    ax.set_xticks(x, names, rotation=25, ha="right")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    low = [low_high_power(ensembles[name])["low_momentum_power"] for name in names]
    high = [low_high_power(ensembles[name])["high_momentum_power"] for name in names]
    ax.bar(x - 0.18, low, width=0.36, label="low-p")
    ax.bar(x + 0.18, high, width=0.36, label="high-p")
    ax.set_xticks(x, names, rotation=25, ha="right")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    if args.fine_size % 2 != 0:
        raise ValueError("--fine-size must be even")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    params = Phi4Params(kappa=args.kappa_fine, lam=args.lam)
    generator = torch.Generator(device=device).manual_seed(args.seed)

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
    if args.n_chains > len(dataset):
        raise ValueError("--n-chains cannot exceed available configurations")
    idx = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(args.seed))[: args.n_chains]
    phi_c = phi_c_all[idx].to(device)
    true_phi = true_phi_all[idx].cpu()

    flow = load_reverse_flow(args, device)
    d0, _, _, logq0 = flow.sample_with_decomposition(phi_c, generator=generator)
    raw_phi = reconstruct_from_average_block(phi_c[:, 0], d0).cpu()
    d_after, patch_stats, action_trace = run_patchwise_ar(
        phi_c,
        d0.clone(),
        params,
        patch_size=args.patch_size,
        step_size=args.step_size,
        n_sweeps=args.n_sweeps,
        generator=generator,
    )
    patch_phi = reconstruct_from_average_block(phi_c[:, 0], d_after).cpu()
    ensembles = {
        "true": true_phi,
        "raw_reverse_kl": raw_phi,
        "patchwise_ar": patch_phi,
    }
    global_ar = global_ar_reference(args.ar_diagnostics, args.output_dir, params)
    summary = {
        "fine_size": args.fine_size,
        "coarse_size": args.fine_size // 2,
        "detail_dimension": 3 * (args.fine_size // 2) ** 2,
        "kappa_fine": args.kappa_fine,
        "lambda": args.lam,
        "checkpoint": str(args.checkpoint),
        "n_chains": args.n_chains,
        "n_sweeps": args.n_sweeps,
        "patch_size": args.patch_size,
        "step_size": args.step_size,
        "initial_logq": tensor_stats(logq0),
        **patch_stats,
        "ensembles": {name: ensemble_summary(phi, params) for name, phi in ensembles.items()},
        "global_ar_reverse_kl": global_ar if global_ar is not None else "missing",
        "action_trace": action_trace,
    }
    summary_path = args.output_dir / "patchwise_ar_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    plot_path = args.output_dir / "patchwise_ar_diagnostics.pdf"
    plot_diagnostics(plot_path, ensembles, action_trace, params)

    after = summary["ensembles"]["patchwise_ar"]
    print(f"patch_acceptance_rate {summary['patch_acceptance_rate']:.6g}")
    print(f"S_f_mean_after {after['S_f']['mean']:.6g}")
    print(f"S_f_std_after {after['S_f']['std']:.6g}")
    print(f"mean_phi2_after {after['mean_phi2']:.6g}")
    print(f"binder_after {after['binder']:.6g}")
    print(f"nn_corr_after {after['nearest_neighbor_correlator']:.6g}")
    print(f"susceptibility_after {after['susceptibility']:.6g}")
    print(f"low_momentum_power_after {after['low_momentum_power']:.6g}")
    print(f"high_momentum_power_after {after['high_momentum_power']:.6g}")
    print(f"wrote {summary_path}")
    print(f"wrote {plot_path}")


if __name__ == "__main__":
    main()
