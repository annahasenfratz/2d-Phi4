"""Patchwise Metropolis updates of residual detail variables at fixed phi_c."""

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
from inverse_blocking_flow.phi4 import Phi4Params, phi4_action, summarize_observables


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fine-size", type=int, default=16)
    parser.add_argument("--lambda", dest="lam", type=float, default=1.0)
    parser.add_argument("--kappa-fine", type=float, default=0.31)
    parser.add_argument("--n-configs", type=int, default=512)
    parser.add_argument("--n-chains", type=int, default=64)
    parser.add_argument("--n-sweeps", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--data-path", type=Path, default=Path("inverse_blocking_flow/outputs_fine16/fine_configs.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("inverse_blocking_flow/outputs_fine16"))
    parser.add_argument("--init", choices=("gaussian", "zeros", "flow", "true"), default="gaussian")
    parser.add_argument("--checkpoint", type=Path, default=Path("inverse_blocking_flow/outputs_fine16/conditional_detail_flow_reverse_kl.pt"))
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--hidden-channels", type=int, default=48)
    parser.add_argument("--cnn-depth", type=int, default=3)
    parser.add_argument("--burn-in", type=int, default=200)
    parser.add_argument("--sample-interval", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--proposal-width", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=54321)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--tag", type=str, default="patchwise")
    return parser


def gaussian_patch_logq(patch: torch.Tensor) -> torch.Tensor:
    return -0.5 * (patch.square() + math.log(2.0 * math.pi)).sum(dim=(1, 2, 3))


@torch.no_grad()
def initial_details(
    args: argparse.Namespace,
    phi_c: torch.Tensor,
    d_true: torch.Tensor,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    if args.init == "zeros":
        return torch.zeros((phi_c.shape[0], 3, phi_c.shape[-2], phi_c.shape[-1]), device=device)
    if args.init == "true":
        return d_true.to(device).clone()
    if args.init == "flow":
        if not args.checkpoint.exists():
            raise FileNotFoundError(f"missing checkpoint for --init flow: {args.checkpoint}")
        flow = ConditionalDetailFlow(args.layers, args.hidden_channels, args.cnn_depth).to(device)
        state = torch.load(args.checkpoint, map_location=device, weights_only=False)
        flow.load_state_dict(state["model"])
        flow.eval()
        d, _ = flow.sample(phi_c, generator=generator)
        return d
    return torch.randn(
        (phi_c.shape[0], 3, phi_c.shape[-2], phi_c.shape[-1]),
        generator=generator,
        device=device,
    )


@torch.no_grad()
def patchwise_sweeps(
    phi_c: torch.Tensor,
    d: torch.Tensor,
    params: Phi4Params,
    *,
    n_sweeps: int,
    patch_size: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, dict[str, float]]:
    coarse_size = phi_c.shape[-1]
    if patch_size <= 0:
        raise ValueError("--patch-size must be positive")
    phi = reconstruct_from_average_block(phi_c[:, 0], d)
    action = phi4_action(phi, params)
    accepts = 0
    proposals = 0

    for _ in range(n_sweeps):
        for y0 in range(0, coarse_size, patch_size):
            for x0 in range(0, coarse_size, patch_size):
                y1 = min(y0 + patch_size, coarse_size)
                x1 = min(x0 + patch_size, coarse_size)
                old_patch = d[:, :, y0:y1, x0:x1]
                new_patch = torch.randn(old_patch.shape, generator=generator, device=d.device)
                d_new = d.clone()
                d_new[:, :, y0:y1, x0:x1] = new_patch
                phi_new = reconstruct_from_average_block(phi_c[:, 0], d_new)
                action_new = phi4_action(phi_new, params)
                logq_old_patch = gaussian_patch_logq(old_patch)
                logq_new_patch = gaussian_patch_logq(new_patch)
                log_accept = -action_new + action + logq_old_patch - logq_new_patch
                log_u = torch.log(torch.rand(log_accept.shape, generator=generator, device=d.device))
                accept = log_u < log_accept
                if accept.any():
                    d[accept] = d_new[accept]
                    action[accept] = action_new[accept]
                accepts += int(accept.sum().item())
                proposals += int(accept.numel())

    return d, {
        "patch_acceptance_rate": accepts / float(proposals),
        "patch_proposals": float(proposals),
    }


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
    phi_c_all, d_true_all, true_phi_all = dataset.tensors
    if args.n_chains > len(dataset):
        raise ValueError("--n-chains cannot exceed available configurations")
    idx = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(args.seed))[: args.n_chains]
    phi_c = phi_c_all[idx].to(device)
    d_true = d_true_all[idx].to(device)
    d = initial_details(args, phi_c, d_true, generator, device)
    d, patch_stats = patchwise_sweeps(
        phi_c,
        d,
        params,
        n_sweeps=args.n_sweeps,
        patch_size=args.patch_size,
        generator=generator,
    )
    phi_after = reconstruct_from_average_block(phi_c[:, 0], d).cpu()
    diagnostics = {
        "fine_size": args.fine_size,
        "coarse_size": args.fine_size // 2,
        "detail_dimension": 3 * (args.fine_size // 2) ** 2,
        "kappa_fine": args.kappa_fine,
        "lambda": args.lam,
        "n_chains": args.n_chains,
        "n_sweeps": args.n_sweeps,
        "patch_size": args.patch_size,
        "init": args.init,
        "proposal": "independent Gaussian patch details",
        **patch_stats,
        "after_patch_sweeps": summarize_observables(phi_after, params),
        "reference_true": summarize_observables(true_phi_all[idx], params),
    }
    out_path = args.output_dir / f"patchwise_detail_diagnostics_{args.tag}.json"
    out_path.write_text(json.dumps(diagnostics, indent=2) + "\n")
    after = diagnostics["after_patch_sweeps"]
    print(f"patch_acceptance_rate {diagnostics['patch_acceptance_rate']:.6g}")
    print(f"S_mean_after_patch_sweeps {after['S_mean']:.6g}")
    print(f"binder_after_patch_sweeps {after['binder']:.6g}")
    print(f"mean_phi2_after_patch_sweeps {after['mean_phi2']:.6g}")
    print(f"nn_corr_after_patch_sweeps {after['nn_corr']:.6g}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
