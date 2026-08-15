"""Sample inverse-blocked proposals and apply independence Metropolis A/R."""

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


def log_weight_diagnostics(log_weights: torch.Tensor) -> dict[str, float]:
    log_weights = log_weights.detach().float()
    log_norm = torch.logsumexp(log_weights, dim=0)
    log_ess = 2.0 * log_norm - torch.logsumexp(2.0 * log_weights, dim=0)
    ess = torch.exp(log_ess)
    return {
        "logw_mean": float(log_weights.mean().item()),
        "logw_std": float(log_weights.std(unbiased=False).item()),
        "logw_min": float(log_weights.min().item()),
        "logw_max": float(log_weights.max().item()),
        "ess": float(ess.item()),
        "ess_over_n": float((ess / log_weights.numel()).item()),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fine-size", type=int, default=16)
    parser.add_argument("--lambda", dest="lam", type=float, default=1.0)
    parser.add_argument("--kappa-fine", type=float, default=0.31)
    parser.add_argument("--n-configs", type=int, default=512)
    parser.add_argument("--n-steps", type=int, default=512, help="number of post-burn-in samples to keep")
    parser.add_argument("--ar-burn-in", type=int, default=128, help="discard this many A/R transitions")
    parser.add_argument("--thin", type=int, default=1, help="keep one sample every THIN A/R transitions")
    parser.add_argument("--data-path", type=Path, default=Path("inverse_blocking_flow/outputs_fine16/fine_configs.pt"))
    parser.add_argument("--checkpoint", type=Path, default=Path("inverse_blocking_flow/outputs_fine16/conditional_detail_flow_reverse_kl.pt"))
    parser.add_argument("--checkpoint-list", type=Path, nargs="*", default=None)
    parser.add_argument(
        "--compare-defaults",
        action="store_true",
        help="run MLE, reverse-KL, and mixed checkpoints if present",
    )
    parser.add_argument(
        "--ar-with-true-details",
        action="store_true",
        help="sanity mode: propose full fine fields from held-out true configs and accept all",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("inverse_blocking_flow/outputs_fine16"))
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--hidden-channels", type=int, default=48)
    parser.add_argument("--cnn-depth", type=int, default=3)
    parser.add_argument("--burn-in", type=int, default=200)
    parser.add_argument("--sample-interval", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--proposal-width", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=4321)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--tag", type=str, default=None, help="output filename tag; defaults to checkpoint stem")
    return parser


@torch.no_grad()
def propose(
    flow: ConditionalDetailFlow,
    coarse_pool: torch.Tensor,
    params: Phi4Params,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    idx = torch.randint(coarse_pool.shape[0], (1,), generator=generator)
    phi_c = coarse_pool[idx].to(device)
    d, log_q = flow.sample(phi_c, generator=generator)
    phi = reconstruct_from_average_block(phi_c[:, 0], d)
    action = phi4_action(phi, params)
    return phi.cpu(), phi_c.cpu(), d.cpu(), (log_q.cpu(), action.cpu())


def checkpoint_tag(path: Path) -> str:
    return path.stem.replace("conditional_detail_flow_", "")


def load_flow_from_checkpoint(
    path: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> ConditionalDetailFlow:
    flow = ConditionalDetailFlow(args.layers, args.hidden_channels, args.cnn_depth).to(device)
    if not path.exists():
        raise FileNotFoundError(f"missing checkpoint: {path}")
    state = torch.load(path, map_location=device, weights_only=False)
    flow.load_state_dict(state["model"])
    flow.eval()
    return flow


def comparison_row(diagnostics: dict[str, object]) -> dict[str, float | str]:
    proposal_logw = diagnostics["proposal_log_weight_diagnostics"]
    before = diagnostics["flow_before_ar"]
    after = diagnostics["flow_after_ar"]
    return {
        "tag": str(diagnostics["tag"]),
        "checkpoint": str(diagnostics["checkpoint"]),
        "logw_std": float(proposal_logw["logw_std"]),
        "ess_over_n": float(proposal_logw["ess_over_n"]),
        "acceptance_rate": float(diagnostics["acceptance_rate"]),
        "kept_window_acceptance_rate": float(diagnostics["kept_window_acceptance_rate"]),
        "S_mean_before_ar": float(before["S_mean"]),
        "S_mean_after_ar": float(after["S_mean"]),
        "binder_before_ar": float(before["binder"]),
        "binder_after_ar": float(after["binder"]),
        "mean_phi2_before_ar": float(before["mean_phi2"]),
        "mean_phi2_after_ar": float(after["mean_phi2"]),
    }


def plot_ar_histograms(
    path: Path,
    true_phi: torch.Tensor,
    gaussian_phi: torch.Tensor,
    flow_phi: torch.Tensor,
    accepted_phi: torch.Tensor,
    params: Phi4Params,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = {
        "true": phi4_action(true_phi, params),
        "gaussian_details": phi4_action(gaussian_phi, params),
        "flow_before_ar": phi4_action(flow_phi, params),
        "flow_after_ar": phi4_action(accepted_phi, params),
    }
    plt.figure(figsize=(6.4, 4.2))
    for label, values in series.items():
        plt.hist(values.numpy(), bins=40, density=True, alpha=0.42, label=label)
    plt.xlabel("S_fine")
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


@torch.no_grad()
def run_ar_with_true_details(
    dataset: torch.utils.data.TensorDataset,
    params: Phi4Params,
    args: argparse.Namespace,
    generator: torch.Generator,
) -> dict[str, object]:
    _, _, true_phi = dataset.tensors
    split = max(1, len(true_phi) // 2)
    reference_phi = true_phi[:split]
    proposal_pool = true_phi[split:] if split < len(true_phi) else true_phi
    if proposal_pool.numel() == 0:
        raise ValueError("need at least one true configuration for --ar-with-true-details")

    current_idx = torch.randint(proposal_pool.shape[0], (1,), generator=generator)
    current_phi = proposal_pool[current_idx][0].clone()
    accepted = []
    raw_flow = []
    raw_log_weights = []
    kept_log_weights = []
    accepts_total = 0
    accepts_kept_window = 0
    total_transitions = args.ar_burn_in + args.n_steps * args.thin

    # Sanity mode: proposals are full configurations drawn from the empirical
    # fine ensemble. We treat the empirical target/proposal factors as
    # cancelling, so there is no flow logq term and the MH correction is the
    # identity. This checks the A/R storage and histogram plumbing only.
    for step in range(total_transitions):
        new_idx = torch.randint(proposal_pool.shape[0], (1,), generator=generator)
        new_phi = proposal_pool[new_idx][0].clone()
        new_action = phi4_action(new_phi.unsqueeze(0), params)
        current_phi = new_phi
        current_action = new_action
        accepts_total += 1
        accepted_this_step = True
        if step >= args.ar_burn_in and (step - args.ar_burn_in) % args.thin == 0:
            raw_flow.append(new_phi)
            raw_log_weights.append((-new_action).reshape(()).cpu())
            accepted.append(current_phi.clone())
            kept_log_weights.append((-current_action).reshape(()).cpu())
            accepts_kept_window += int(accepted_this_step)

    accepted_phi = torch.stack(accepted)
    raw_flow_phi = torch.stack(raw_flow)
    raw_log_weights_tensor = torch.stack(raw_log_weights)
    kept_log_weights_tensor = torch.stack(kept_log_weights)
    n_eval = min(args.n_steps, len(reference_phi))
    phi_c_eval = dataset.tensors[0]
    gaussian_d = torch.randn(
        (n_eval, 3, args.fine_size // 2, args.fine_size // 2),
        generator=torch.Generator().manual_seed(args.seed + 1),
    )
    gaussian_phi = reconstruct_from_average_block(phi_c_eval[:n_eval, 0], gaussian_d)
    acceptance_rate = accepts_total / float(total_transitions)
    kept_window_acceptance_rate = accepts_kept_window / float(args.n_steps)
    tag = args.tag or "true_details"

    diagnostics = {
        "tag": tag,
        "mode": "ar_with_true_details",
        "fine_size": args.fine_size,
        "coarse_size": args.fine_size // 2,
        "kappa_fine": args.kappa_fine,
        "lambda": args.lam,
        "checkpoint": None,
        "ar_burn_in": args.ar_burn_in,
        "thin": args.thin,
        "n_steps_kept": args.n_steps,
        "proposal_pool_size": int(proposal_pool.shape[0]),
        "reference_pool_size": int(reference_phi.shape[0]),
        "acceptance_rate": acceptance_rate,
        "kept_window_acceptance_rate": kept_window_acceptance_rate,
        "proposal_log_weight_diagnostics": log_weight_diagnostics(raw_log_weights_tensor),
        "chain_log_weight_diagnostics": log_weight_diagnostics(kept_log_weights_tensor),
        "true": summarize_observables(reference_phi[:n_eval], params),
        "gaussian_details": summarize_observables(gaussian_phi, params),
        "flow_before_ar": summarize_observables(raw_flow_phi, params),
        "flow_after_ar": summarize_observables(accepted_phi, params),
    }
    diag_path = args.output_dir / f"ar_diagnostics_{tag}.json"
    diag_path.write_text(json.dumps(diagnostics, indent=2) + "\n")
    plot_path = args.output_dir / f"action_histograms_ar_{tag}.pdf"
    plot_ar_histograms(plot_path, reference_phi[:n_eval], gaussian_phi, raw_flow_phi, accepted_phi, params)
    print("running A/R sanity mode with held-out true details")
    print(f"acceptance_rate {acceptance_rate:.6g}")
    print(f"kept_window_acceptance_rate {kept_window_acceptance_rate:.6g}")
    print(f"wrote {diag_path}")
    print(f"wrote {plot_path}")
    return diagnostics


@torch.no_grad()
def run_ar_chain(
    flow: ConditionalDetailFlow,
    coarse_pool: torch.Tensor,
    dataset: torch.utils.data.TensorDataset,
    params: Phi4Params,
    args: argparse.Namespace,
    device: torch.device,
    generator: torch.Generator,
    checkpoint: Path,
    tag: str,
) -> dict[str, object]:
    current_phi, current_c, current_d, packed = propose(flow, coarse_pool, params, generator, device)
    current_logq, current_action = packed
    accepted = []
    raw_flow = []
    raw_log_weights = []
    kept_log_weights = []
    accepts_total = 0
    accepts_kept_window = 0
    total_transitions = args.ar_burn_in + args.n_steps * args.thin

    # The empirical phi_c proposal factor cancels because both old and new
    # states draw phi_c uniformly from the same blocked fine ensemble. If phi_c
    # later comes from an independent coarse action S_coarse, include its
    # proposal/target contribution in this ratio.
    for step in range(total_transitions):
        new_phi, new_c, new_d, new_packed = propose(flow, coarse_pool, params, generator, device)
        new_logq, new_action = new_packed
        log_a = (-new_action - new_logq) - (-current_action - current_logq)
        accepted_this_step = False
        if math.log(torch.rand((), generator=generator).item()) < float(log_a.item()):
            current_phi, current_c, current_d = new_phi, new_c, new_d
            current_logq, current_action = new_logq, new_action
            accepts_total += 1
            accepted_this_step = True
        if step >= args.ar_burn_in and (step - args.ar_burn_in) % args.thin == 0:
            raw_flow.append(new_phi[0])
            raw_log_weights.append((-new_action - new_logq).reshape(()).cpu())
            accepted.append(current_phi[0].clone())
            kept_log_weights.append((-current_action - current_logq).reshape(()).cpu())
            accepts_kept_window += int(accepted_this_step)

    accepted_phi = torch.stack(accepted)
    raw_flow_phi = torch.stack(raw_flow)
    raw_log_weights_tensor = torch.stack(raw_log_weights)
    kept_log_weights_tensor = torch.stack(kept_log_weights)
    phi_c_eval, _, true_phi = dataset.tensors
    n_eval = min(args.n_steps, len(true_phi))
    gaussian_d = torch.randn(
        (n_eval, 3, args.fine_size // 2, args.fine_size // 2),
        generator=torch.Generator().manual_seed(args.seed + 1),
    )
    gaussian_phi = reconstruct_from_average_block(phi_c_eval[:n_eval, 0], gaussian_d)
    acceptance_rate = accepts_total / float(total_transitions)
    kept_window_acceptance_rate = accepts_kept_window / float(args.n_steps)

    diagnostics = {
        "tag": tag,
        "fine_size": args.fine_size,
        "coarse_size": args.fine_size // 2,
        "kappa_fine": args.kappa_fine,
        "lambda": args.lam,
        "checkpoint": str(checkpoint),
        "ar_burn_in": args.ar_burn_in,
        "thin": args.thin,
        "n_steps_kept": args.n_steps,
        "acceptance_rate": acceptance_rate,
        "kept_window_acceptance_rate": kept_window_acceptance_rate,
        "proposal_log_weight_diagnostics": log_weight_diagnostics(raw_log_weights_tensor),
        "chain_log_weight_diagnostics": log_weight_diagnostics(kept_log_weights_tensor),
        "true": summarize_observables(true_phi[:n_eval], params),
        "gaussian_details": summarize_observables(gaussian_phi, params),
        "flow_before_ar": summarize_observables(raw_flow_phi, params),
        "flow_after_ar": summarize_observables(accepted_phi, params),
    }
    diag_path = args.output_dir / f"ar_diagnostics_{tag}.json"
    diag_path.write_text(json.dumps(diagnostics, indent=2) + "\n")
    plot_path = args.output_dir / f"action_histograms_ar_{tag}.pdf"
    plot_ar_histograms(plot_path, true_phi[:n_eval], gaussian_phi, raw_flow_phi, accepted_phi, params)
    print(f"acceptance_rate {acceptance_rate:.6g}")
    print(f"kept_window_acceptance_rate {kept_window_acceptance_rate:.6g}")
    print(f"wrote {diag_path}")
    print(f"wrote {plot_path}")
    return diagnostics


def checkpoints_to_run(args: argparse.Namespace) -> list[Path]:
    if args.compare_defaults:
        paths = [
            args.output_dir / "conditional_detail_flow_mle.pt",
            args.output_dir / "conditional_detail_flow_reverse_kl.pt",
            args.output_dir / "conditional_detail_flow_mixed.pt",
        ]
        return [path for path in paths if path.exists()]
    if args.checkpoint_list:
        return args.checkpoint_list
    return [args.checkpoint]


def main() -> None:
    args = build_parser().parse_args()
    if args.n_steps <= 0:
        raise ValueError("--n-steps must be positive")
    if args.ar_burn_in < 0:
        raise ValueError("--ar-burn-in must be non-negative")
    if args.thin <= 0:
        raise ValueError("--thin must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    params = Phi4Params(kappa=args.kappa_fine, lam=args.lam)

    phi_f = load_or_generate_fine_configs(
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
    )
    dataset = make_paired_dataset(phi_f)
    coarse_pool = dataset.tensors[0]

    if args.ar_with_true_details:
        generator = torch.Generator(device=device).manual_seed(args.seed)
        run_ar_with_true_details(dataset, params, args, generator)
        return

    diagnostics_by_tag = []
    for i, checkpoint in enumerate(checkpoints_to_run(args)):
        tag = args.tag if args.tag and not (args.compare_defaults or args.checkpoint_list) else checkpoint_tag(checkpoint)
        print(f"running A/R for {tag}: {checkpoint}")
        generator = torch.Generator(device=device).manual_seed(args.seed + 1009 * i)
        flow = load_flow_from_checkpoint(checkpoint, args, device)
        diagnostics = run_ar_chain(
            flow,
            coarse_pool,
            dataset,
            params,
            args,
            device,
            generator,
            checkpoint,
            tag,
        )
        diagnostics_by_tag.append(diagnostics)

    if len(diagnostics_by_tag) > 1:
        summary = {
            "fine_size": args.fine_size,
            "coarse_size": args.fine_size // 2,
            "kappa_fine": args.kappa_fine,
            "lambda": args.lam,
            "rows": [comparison_row(diagnostics) for diagnostics in diagnostics_by_tag],
        }
        summary_path = args.output_dir / "ar_comparison_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
