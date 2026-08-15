"""Train a small conditional inverse-blocking flow."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/matplotlib")

from torch.utils.data import TensorDataset

from inverse_blocking_flow.data import load_or_generate_fine_configs, make_paired_dataset
from inverse_blocking_flow.flow import ConditionalDetailFlow, make_conditioning, n_conditioning_channels, sanity_check_inverse
from inverse_blocking_flow.haar import (
    average_block,
    eta_scaling_factor,
    reconstruct_from_average_block,
    reconstruct_from_weighted_block,
    soft_block,
    soft_kernel_term,
    soft_reconstruct,
    soft_weighted_block,
    soft_weighted_kernel_term,
    soft_weighted_reconstruct,
    weighted_block,
    weighted_kernel_normalization,
)
from inverse_blocking_flow.phi4 import Phi4Params, phi4_action, summarize_observables


def tensor_summary(x: torch.Tensor) -> dict[str, float]:
    x = x.detach().float().cpu()
    return {
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
    }


def log_weight_diagnostics(log_weights: torch.Tensor) -> dict[str, float]:
    log_weights = log_weights.detach().float().cpu()
    log_norm = torch.logsumexp(log_weights, dim=0)
    log_ess = 2.0 * log_norm - torch.logsumexp(2.0 * log_weights, dim=0)
    ess = torch.exp(log_ess)
    out = tensor_summary(log_weights)
    out.update(
        {
            "ess": float(ess.item()),
            "ess_over_n": float((ess / log_weights.numel()).item()),
        }
    )
    return out


def jsonable_args(args: argparse.Namespace) -> dict[str, object]:
    out = {}
    for key, value in vars(args).items():
        out[key] = str(value) if isinstance(value, Path) else value
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("mle", "reverse_kl", "mixed", "tempered_reverse_kl"),
        default="mle",
    )
    parser.add_argument("--fine-size", type=int, default=16)
    parser.add_argument("--lambda", dest="lam", type=float, default=1.0)
    parser.add_argument("--kappa-fine", type=float, default=0.31)
    parser.add_argument("--n-configs", type=int, default=512)
    parser.add_argument("--data-path", type=Path, default=Path("inverse_blocking_flow/outputs_fine16/fine_configs.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("inverse_blocking_flow/outputs_fine16"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--mle-pretrain-epochs", type=int, default=5)
    parser.add_argument("--mle-mix-alpha", type=float, default=0.5)
    parser.add_argument(
        "--logw-var-alpha",
        type=float,
        default=0.0,
        help="add alpha * mean((logw - mean(logw))^2) to the reverse-KL component",
    )
    parser.add_argument(
        "--checkpoint-tag",
        type=str,
        default=None,
        help="optional checkpoint/diagnostic filename tag; defaults to mode",
    )
    parser.add_argument("--beta-start", type=float, default=0.1)
    parser.add_argument("--beta-end", type=float, default=1.0)
    parser.add_argument(
        "--diagnostic-betas",
        type=str,
        default="0.1,0.25,0.5,0.75,1.0",
        help="comma-separated beta values for tempered log-weight diagnostics",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--hidden-channels", type=int, default=48)
    parser.add_argument("--cnn-depth", type=int, default=3)
    parser.add_argument("--conditioning-mode", choices=("basic", "physics"), default="basic")
    parser.add_argument("--blocking-mode", choices=("hard", "soft", "weighted", "soft_weighted"), default="hard")
    parser.add_argument("--soft-alpha", type=float, default=1.0)
    parser.add_argument("--weighted-a", type=float, default=0.18)
    parser.add_argument("--weighted-b", type=float, default=0.04)
    parser.add_argument("--train-weighted-kernel", type=parse_bool, default=False)
    parser.add_argument("--weighted-kernel-reg", type=float, default=0.0)
    parser.add_argument("--weighted-kernel-prior-reg", type=float, default=None)
    parser.add_argument("--weighted-highp-reg", type=float, default=0.0)
    parser.add_argument("--weighted-min-a", type=float)
    parser.add_argument("--weighted-min-b", type=float)
    parser.add_argument("--eta", type=float, default=0.25)
    parser.add_argument("--use-eta-scaling", type=parse_bool, default=True)
    parser.add_argument("--burn-in", type=int, default=200)
    parser.add_argument("--sample-interval", type=int, default=10)
    parser.add_argument("--proposal-width", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    return parser


def parse_bool(value: str) -> bool:
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def beta_for_epoch(epoch: int, epochs: int, beta_start: float, beta_end: float) -> float:
    if epochs <= 1:
        return beta_end
    frac = (epoch - 1) / float(epochs - 1)
    return beta_start + frac * (beta_end - beta_start)


def reconstruct_for_mode(
    phi_c: torch.Tensor,
    detail: torch.Tensor,
    args: argparse.Namespace,
    weighted_a: float | torch.Tensor | None = None,
    weighted_b: float | torch.Tensor | None = None,
) -> torch.Tensor:
    blocking_mode = args.blocking_mode
    if blocking_mode == "hard":
        return reconstruct_from_average_block(phi_c[:, 0], detail)
    if blocking_mode == "soft":
        return soft_reconstruct(phi_c[:, 0], detail)
    if blocking_mode == "weighted":
        return reconstruct_from_weighted_block(
            phi_c[:, 0],
            detail,
            args.weighted_a if weighted_a is None else weighted_a,
            args.weighted_b if weighted_b is None else weighted_b,
            eta=args.eta,
            use_eta_scaling=args.use_eta_scaling,
        )
    if blocking_mode == "soft_weighted":
        return soft_weighted_reconstruct(
            phi_c[:, 0],
            detail,
            args.weighted_a if weighted_a is None else weighted_a,
            args.weighted_b if weighted_b is None else weighted_b,
            eta=args.eta,
            use_eta_scaling=args.use_eta_scaling,
        )
    raise ValueError(f"unknown blocking mode: {blocking_mode}")


def conditional_action_for_mode(
    phi_c: torch.Tensor,
    detail: torch.Tensor,
    params: Phi4Params,
    args: argparse.Namespace,
    weighted_a: float | torch.Tensor | None = None,
    weighted_b: float | torch.Tensor | None = None,
) -> torch.Tensor:
    phi = reconstruct_for_mode(phi_c, detail, args, weighted_a, weighted_b)
    action = phi4_action(phi, params)
    if args.blocking_mode == "soft":
        action = action + soft_kernel_term(detail, args.soft_alpha)
    if args.blocking_mode == "soft_weighted":
        action = action + soft_weighted_kernel_term(detail, args.soft_alpha)
    return action


def weighted_metadata(a: float | torch.Tensor, b: float | torch.Tensor, args: argparse.Namespace) -> dict[str, object]:
    a_value = float(torch.as_tensor(a).detach().cpu().item())
    b_value = float(torch.as_tensor(b).detach().cpu().item())
    lambda_prior = args.weighted_kernel_reg if args.weighted_kernel_prior_reg is None else args.weighted_kernel_prior_reg
    return {
        "weighted_a": a_value,
        "weighted_b": b_value,
        "weighted_N": float(weighted_kernel_normalization(a_value, b_value)),
        "train_weighted_kernel": args.train_weighted_kernel,
        "weighted_kernel_reg": args.weighted_kernel_reg,
        "weighted_kernel_prior_reg": lambda_prior,
        "weighted_highp_reg": args.weighted_highp_reg,
        "weighted_min_a": args.weighted_min_a,
        "weighted_min_b": args.weighted_min_b,
        "weighted_kernel_initial_a": args.weighted_a,
        "weighted_kernel_initial_b": args.weighted_b,
        "eta": args.eta,
        "Delta_phi": 0.5 * args.eta,
        "Z_eta": float(eta_scaling_factor(args.eta, use_eta_scaling=args.use_eta_scaling)),
        "use_eta_scaling": args.use_eta_scaling,
    }


def weighted_fourier_response(
    a: float | torch.Tensor,
    b: float | torch.Tensor,
    kx: float | torch.Tensor,
    ky: float | torch.Tensor,
) -> torch.Tensor:
    a_tensor = torch.as_tensor(a)
    b_tensor = torch.as_tensor(b, dtype=a_tensor.dtype, device=a_tensor.device)
    kx_tensor = torch.as_tensor(kx, dtype=a_tensor.dtype, device=a_tensor.device)
    ky_tensor = torch.as_tensor(ky, dtype=a_tensor.dtype, device=a_tensor.device)
    n = weighted_kernel_normalization(a_tensor, b_tensor)
    return n * (
        1.0
        + 2.0 * a_tensor * (torch.cos(kx_tensor) + torch.cos(ky_tensor))
        + 4.0 * b_tensor * torch.cos(kx_tensor) * torch.cos(ky_tensor)
    )


def weighted_highp_penalty(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    pi = torch.as_tensor(torch.pi, dtype=a.dtype, device=a.device)
    momenta = [
        (0.5 * pi, 0.0),
        (0.0, 0.5 * pi),
        (0.5 * pi, 0.5 * pi),
        (pi, 0.0),
        (0.0, pi),
        (pi, pi),
        (0.5 * pi, pi),
        (pi, 0.5 * pi),
    ]
    values = torch.stack([weighted_fourier_response(a, b, kx, ky).square() for kx, ky in momenta])
    return values.mean()


def plot_action_histograms(
    path: Path,
    true_phi: torch.Tensor,
    gaussian_phi: torch.Tensor,
    flow_phi: torch.Tensor,
    params: Phi4Params,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    series = {
        "true": phi4_action(true_phi, params).detach().cpu(),
        "gaussian_details": phi4_action(gaussian_phi, params).detach().cpu(),
        "flow": phi4_action(flow_phi, params).detach().cpu(),
    }
    plt.figure(figsize=(6.0, 4.0))
    for label, values in series.items():
        plt.hist(values.numpy(), bins=40, density=True, alpha=0.45, label=label)
    plt.xlabel("S_fine")
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_full_density_diagnostics(
    output_dir: Path,
    tag: str,
    true_action: torch.Tensor,
    true_logq: torch.Tensor,
    true_logw: torch.Tensor,
    prop_action: torch.Tensor,
    prop_logq: torch.Tensor,
    prop_logw: torch.Tensor,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    true_action = true_action.detach().cpu()
    true_logq = true_logq.detach().cpu()
    true_logw = true_logw.detach().cpu()
    prop_action = prop_action.detach().cpu()
    prop_logq = prop_logq.detach().cpu()
    prop_logw = prop_logw.detach().cpu()

    plt.figure(figsize=(6.0, 4.0))
    plt.hist(true_logq.numpy(), bins=40, density=True, alpha=0.45, label="true details")
    plt.hist(prop_logq.numpy(), bins=40, density=True, alpha=0.45, label="generated details")
    plt.xlabel("log q(d | phi_c)")
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"logq_histograms_{tag}.pdf")
    plt.close()

    plt.figure(figsize=(6.0, 4.0))
    plt.hist(true_logw.numpy(), bins=40, density=True, alpha=0.45, label="true details")
    plt.hist(prop_logw.numpy(), bins=40, density=True, alpha=0.45, label="generated details")
    plt.xlabel("-S_f - log q")
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"logw_histograms_{tag}.pdf")
    plt.close()

    plt.figure(figsize=(6.0, 4.4))
    plt.scatter(true_action.numpy(), true_logq.numpy(), s=10, alpha=0.55, label="true details")
    plt.scatter(prop_action.numpy(), prop_logq.numpy(), s=10, alpha=0.55, label="generated details")
    plt.xlabel("S_f")
    plt.ylabel("log q(d | phi_c)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"scatter_action_logq_{tag}.pdf")
    plt.close()

    plt.figure(figsize=(6.0, 4.4))
    plt.scatter(prop_action.numpy(), prop_logw.numpy(), s=10, alpha=0.65, label="generated details")
    plt.xlabel("S_f")
    plt.ylabel("-S_f - log q")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"scatter_action_logw_proposals_{tag}.pdf")
    plt.close()


def main() -> None:
    args = build_parser().parse_args()
    if args.fine_size % 2 != 0:
        raise ValueError("--fine-size must be even")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not 0.0 <= args.mle_mix_alpha <= 1.0:
        raise ValueError("--mle-mix-alpha must be in [0, 1]")
    if args.logw_var_alpha < 0.0:
        raise ValueError("--logw-var-alpha must be nonnegative")
    if args.weighted_kernel_reg < 0.0:
        raise ValueError("--weighted-kernel-reg must be nonnegative")
    if args.weighted_kernel_prior_reg is not None and args.weighted_kernel_prior_reg < 0.0:
        raise ValueError("--weighted-kernel-prior-reg must be nonnegative")
    if args.weighted_highp_reg < 0.0:
        raise ValueError("--weighted-highp-reg must be nonnegative")
    if args.beta_start <= 0.0 or args.beta_end <= 0.0:
        raise ValueError("--beta-start and --beta-end must be positive")
    if args.soft_alpha <= 0.0:
        raise ValueError("--soft-alpha must be positive")
    if abs(1.0 + 4.0 * args.weighted_a + 4.0 * args.weighted_b) <= 1e-12:
        raise ValueError("weighted kernel normalization denominator is too close to zero")
    diagnostic_betas = [float(x) for x in args.diagnostic_betas.split(",") if x.strip()]
    torch.manual_seed(args.seed)

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
    if args.blocking_mode == "hard":
        dataset = make_paired_dataset(phi_f)
        n_detail = 3
    elif args.blocking_mode == "soft":
        psi, u = soft_block(phi_f, args.soft_alpha, torch.Generator().manual_seed(args.seed + 991))
        dataset = TensorDataset(psi.unsqueeze(1).float(), u.float(), phi_f.float())
        n_detail = 4
    elif args.blocking_mode == "soft_weighted":
        psi, u = soft_weighted_block(
            phi_f,
            args.soft_alpha,
            args.weighted_a,
            args.weighted_b,
            eta=args.eta,
            use_eta_scaling=args.use_eta_scaling,
            generator=torch.Generator().manual_seed(args.seed + 991),
        )
        dataset = TensorDataset(psi.unsqueeze(1).float(), u.float(), phi_f.float())
        n_detail = 4
    else:
        psi = weighted_block(
            phi_f,
            args.weighted_a,
            args.weighted_b,
            eta=args.eta,
            use_eta_scaling=args.use_eta_scaling,
        )
        _ll, d = average_block(phi_f)
        dataset = TensorDataset(psi.unsqueeze(1).float(), d.float(), phi_f.float())
        n_detail = 3
    train_len = max(1, int(0.9 * len(dataset)))
    val_len = len(dataset) - train_len
    train_data, val_data = random_split(
        dataset,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(args.seed),
    )
    loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    n_cond = n_conditioning_channels(args.conditioning_mode)
    flow = ConditionalDetailFlow(args.layers, args.hidden_channels, args.cnn_depth, n_cond, n_detail).to(device)
    weighted_a = torch.tensor(args.weighted_a, dtype=torch.float32, device=device)
    weighted_b = torch.tensor(args.weighted_b, dtype=torch.float32, device=device)
    if args.blocking_mode == "weighted" and args.train_weighted_kernel:
        weighted_a.requires_grad_(True)
        weighted_b.requires_grad_(True)
    if args.checkpoint is not None and args.checkpoint.exists():
        state = torch.load(args.checkpoint, map_location=device, weights_only=False)
        flow.load_state_dict(state["model"])
        if args.blocking_mode == "weighted":
            weighted_a.data.fill_(float(state.get("weighted_a", state.get("args", {}).get("weighted_a", args.weighted_a))))
            weighted_b.data.fill_(float(state.get("weighted_b", state.get("args", {}).get("weighted_b", args.weighted_b))))
    optim_params = list(flow.parameters())
    if args.blocking_mode == "weighted" and args.train_weighted_kernel:
        optim_params.extend([weighted_a, weighted_b])
    optimizer = torch.optim.Adam(optim_params, lr=args.lr)

    history = []
    total_epochs = args.epochs + (args.mle_pretrain_epochs if args.mode == "mixed" else 0)
    for epoch in range(1, total_epochs + 1):
        flow.train()
        losses = []
        mle_losses = []
        reverse_kl_losses = []
        logw_var_penalties = []
        phase = args.mode
        beta = 1.0
        if args.mode == "mixed" and epoch <= args.mle_pretrain_epochs:
            phase = "mle_pretrain"
        elif args.mode == "mixed":
            phase = "mixed"
        elif args.mode == "tempered_reverse_kl":
            beta = beta_for_epoch(epoch, total_epochs, args.beta_start, args.beta_end)
        for phi_c, d_true, phi_true_batch in loader:
            phi_c = phi_c.to(device)
            d_true = d_true.to(device)
            phi_true_batch = phi_true_batch.to(device)
            if args.blocking_mode == "weighted":
                phi_c = weighted_block(
                    phi_true_batch,
                    weighted_a,
                    weighted_b,
                    eta=args.eta,
                    use_eta_scaling=args.use_eta_scaling,
                ).unsqueeze(1)
            cond = make_conditioning(phi_c, args.conditioning_mode)
            optimizer.zero_grad(set_to_none=True)
            _, true_batch_logq = flow.inverse_logq(d_true, cond)
            mle_loss = -true_batch_logq.mean()
            d, log_q = flow.sample(cond)
            action_rec = conditional_action_for_mode(phi_c, d, params, args, weighted_a, weighted_b)
            reverse_kl_loss = (action_rec + log_q).mean()
            logw = -action_rec - log_q
            logw_centered = logw - logw.mean()
            logw_var_penalty = logw_centered.square().mean()
            penalized_reverse_kl_loss = reverse_kl_loss + args.logw_var_alpha * logw_var_penalty
            kernel_penalty = torch.zeros((), dtype=action_rec.dtype, device=device)
            if args.blocking_mode == "weighted" and args.train_weighted_kernel:
                lambda_prior = args.weighted_kernel_reg if args.weighted_kernel_prior_reg is None else args.weighted_kernel_prior_reg
                if lambda_prior > 0.0:
                    kernel_penalty = kernel_penalty + lambda_prior * (
                        (weighted_a - args.weighted_a).square()
                        + (weighted_b - args.weighted_b).square()
                    )
                if args.weighted_highp_reg > 0.0:
                    kernel_penalty = kernel_penalty + args.weighted_highp_reg * weighted_highp_penalty(weighted_a, weighted_b)
                penalized_reverse_kl_loss = penalized_reverse_kl_loss + kernel_penalty
            tempered_reverse_kl_loss = (beta * action_rec + log_q).mean()
            if phase in ("mle", "mle_pretrain"):
                loss = mle_loss + kernel_penalty
            elif phase == "mixed":
                loss = args.mle_mix_alpha * mle_loss + (1.0 - args.mle_mix_alpha) * penalized_reverse_kl_loss
            elif phase == "tempered_reverse_kl":
                loss = tempered_reverse_kl_loss
            else:
                loss = penalized_reverse_kl_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(optim_params, 10.0)
            optimizer.step()
            if args.blocking_mode == "weighted" and args.train_weighted_kernel:
                with torch.no_grad():
                    if args.weighted_min_a is not None:
                        weighted_a.clamp_(min=args.weighted_min_a)
                    if args.weighted_min_b is not None:
                        weighted_b.clamp_(min=args.weighted_min_b)
            losses.append(float(loss.detach().cpu().item()))
            mle_losses.append(float(mle_loss.detach().cpu().item()))
            reverse_kl_losses.append(float(reverse_kl_loss.detach().cpu().item()))
            logw_var_penalties.append(float(logw_var_penalty.detach().cpu().item()))

        mean_loss = sum(losses) / max(1, len(losses))
        row = {
            "epoch": epoch,
            "phase": phase,
            "beta": beta,
            "loss": mean_loss,
            "mle_loss": sum(mle_losses) / max(1, len(mle_losses)),
            "reverse_kl_loss": sum(reverse_kl_losses) / max(1, len(reverse_kl_losses)),
            "logw_var_alpha": args.logw_var_alpha,
            "logw_var_penalty": sum(logw_var_penalties) / max(1, len(logw_var_penalties)),
        }
        history.append(row)
        beta_text = f" beta {beta:.4g}" if args.mode == "tempered_reverse_kl" else ""
        print(f"epoch {epoch:04d} {phase}{beta_text} loss {mean_loss:.6g}")

    tag = args.checkpoint_tag or args.mode
    checkpoint = args.output_dir / f"conditional_detail_flow_{tag}.pt"
    learned_weighted_metadata = weighted_metadata(weighted_a, weighted_b, args)
    torch.save(
        {
            "model": flow.state_dict(),
            "args": jsonable_args(args),
            "params": {"kappa": params.kappa, "lambda": params.lam},
            "conditioning_mode": args.conditioning_mode,
            "n_conditioning_channels": n_cond,
            "blocking_mode": args.blocking_mode,
            "soft_alpha": args.soft_alpha,
            **learned_weighted_metadata,
            "n_detail_channels": n_detail,
            "history": history,
        },
        checkpoint,
    )

    flow.eval()
    with torch.no_grad():
        phi_c_all, d_all, phi_true = dataset.tensors
        phi_c_eval = phi_c_all[: min(256, len(dataset))].to(device)
        phi_true_eval = phi_true[: phi_c_eval.shape[0]].to(device)
        if args.blocking_mode == "weighted":
            phi_c_eval = weighted_block(
                phi_true_eval,
                weighted_a,
                weighted_b,
                eta=args.eta,
                use_eta_scaling=args.use_eta_scaling,
            ).unsqueeze(1)
        cond_eval = make_conditioning(phi_c_eval, args.conditioning_mode)
        gaussian_d = torch.randn(
            (phi_c_eval.shape[0], n_detail, phi_c_eval.shape[-2], phi_c_eval.shape[-1]),
            device=device,
        )
        gaussian_phi = reconstruct_for_mode(phi_c_eval, gaussian_d, args, weighted_a, weighted_b)
        flow_d, prop_logq = flow.sample(cond_eval)
        flow_phi = reconstruct_for_mode(phi_c_eval, flow_d, args, weighted_a, weighted_b)
        true_action = phi4_action(phi_true_eval, params)
        _, true_logq = flow.inverse_logq(d_all[: phi_c_eval.shape[0]].to(device), cond_eval)
        true_logw = -true_action - true_logq
        if args.blocking_mode == "soft":
            true_logw = true_logw - soft_kernel_term(d_all[: phi_c_eval.shape[0]].to(device), args.soft_alpha)
        if args.blocking_mode == "soft_weighted":
            true_logw = true_logw - soft_weighted_kernel_term(d_all[: phi_c_eval.shape[0]].to(device), args.soft_alpha)
        prop_action = phi4_action(flow_phi, params)
        prop_logw = -prop_action - prop_logq
        if args.blocking_mode == "soft":
            prop_logw = prop_logw - soft_kernel_term(flow_d, args.soft_alpha)
        if args.blocking_mode == "soft_weighted":
            prop_logw = prop_logw - soft_weighted_kernel_term(flow_d, args.soft_alpha)
        recon = reconstruct_for_mode(phi_c_eval, d_all[: phi_c_eval.shape[0]].to(device), args, weighted_a, weighted_b)
        recon_error = float((recon - phi_true_eval).abs().max().cpu().item())
        inverse_error = sanity_check_inverse(flow.cpu(), size=args.fine_size // 2)
        flow.to(device)

        diagnostics = {
            "mode": args.mode,
            "tag": tag,
            "fine_size": args.fine_size,
            "coarse_size": args.fine_size // 2,
            "kappa_fine": args.kappa_fine,
            "lambda": args.lam,
            "checkpoint": str(checkpoint),
            "conditioning_mode": args.conditioning_mode,
            "n_conditioning_channels": n_cond,
            "blocking_mode": args.blocking_mode,
            "soft_alpha": args.soft_alpha,
            **learned_weighted_metadata,
            "n_detail_channels": n_detail,
            "average_block_reconstruction_max_abs_error": recon_error,
            "flow_inverse_max_abs_error": inverse_error,
            "true": summarize_observables(phi_true_eval.cpu(), params),
            "gaussian_details": summarize_observables(gaussian_phi.cpu(), params),
            "flow": summarize_observables(flow_phi.cpu(), params),
            "full_density": {
                "true_logq": tensor_summary(true_logq),
                "generated_logq": tensor_summary(prop_logq),
                "true_logw": log_weight_diagnostics(true_logw),
                "generated_logw": log_weight_diagnostics(prop_logw),
            },
            "tempered_beta_diagnostics": {
                str(beta_value): {
                    "true_logw_beta": log_weight_diagnostics(-beta_value * true_action - true_logq),
                    "generated_logw_beta": log_weight_diagnostics(-beta_value * prop_action - prop_logq),
                }
                for beta_value in diagnostic_betas
            },
            "history": history,
        }
        (args.output_dir / f"diagnostics_{tag}.json").write_text(
            json.dumps(diagnostics, indent=2) + "\n"
        )
        plot_action_histograms(
            args.output_dir / f"action_histograms_{tag}.pdf",
            phi_true_eval.cpu(),
            gaussian_phi.cpu(),
            flow_phi.cpu(),
            params,
        )
        plot_full_density_diagnostics(
            args.output_dir,
            tag,
            true_action,
            true_logq,
            true_logw,
            prop_action,
            prop_logq,
            prop_logw,
        )

    print(f"wrote {checkpoint}")
    print(f"wrote {args.output_dir / f'diagnostics_{tag}.json'}")
    print(f"wrote {args.output_dir / f'action_histograms_{tag}.pdf'}")
    print(f"wrote {args.output_dir / f'logq_histograms_{tag}.pdf'}")
    print(f"wrote {args.output_dir / f'logw_histograms_{tag}.pdf'}")


if __name__ == "__main__":
    main()
