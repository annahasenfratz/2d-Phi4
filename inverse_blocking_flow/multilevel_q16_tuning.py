"""Focused tuning study for the 8x8 -> 16x16 conditional detail flow."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

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
from inverse_blocking_flow.phi4 import Phi4Params, phi4_action


OBS_KEYS = ["S_mean", "S_std", "phi2", "binder", "NN_corr", "susceptibility", "low_p_power", "high_p_power"]


@dataclass(frozen=True)
class ModelConfig:
    name: str
    layers: int
    hidden_channels: int
    cnn_depth: int
    reverse_kl_epochs: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fine-size", type=int, default=32)
    parser.add_argument("--lambda", dest="lam", type=float, default=1.0)
    parser.add_argument("--kappa-fine", type=float, default=0.31)
    parser.add_argument("--n-configs", type=int, default=512)
    parser.add_argument("--n-eval", type=int, default=256)
    parser.add_argument("--data-path", type=Path, default=Path("inverse_blocking_flow/outputs/fine_configs.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("inverse_blocking_flow/outputs"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--mle-epochs", type=int, default=10)
    parser.add_argument("--reverse-kl-epochs", type=str, default="25,50")
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--burn-in", type=int, default=200)
    parser.add_argument("--sample-interval", type=int, default=10)
    parser.add_argument("--proposal-width", type=float, default=1.0)
    parser.add_argument("--reuse-checkpoints", action="store_true")
    parser.add_argument("--patch-size", type=int, default=2)
    parser.add_argument("--step-size", type=float, default=0.1)
    parser.add_argument("--n-sweeps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=424242)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    return parser


def model_grid(reverse_kl_epochs: list[int]) -> list[ModelConfig]:
    architectures = [
        ("A_small", 4, 32, 3),
        ("B_larger", 6, 48, 3),
        ("C_larger_deeper", 8, 64, 4),
    ]
    return [
        ModelConfig(f"{name}_rk{epochs}", layers, hidden, depth, epochs)
        for name, layers, hidden, depth in architectures
        for epochs in reverse_kl_epochs
    ]


def histogram_values(phi: torch.Tensor) -> list[float]:
    hist = torch.histc(phi.detach().cpu(), bins=60, min=-3.0, max=3.0)
    hist = hist / hist.sum().clamp_min(1.0)
    return [float(x) for x in hist]


def reconstruction_distance(phi_rec: torch.Tensor, phi_true: torch.Tensor) -> dict[str, float]:
    diff = phi_rec - phi_true
    true_norm = phi_true.flatten(1).norm(dim=1).clamp_min(1e-12)
    rel = diff.flatten(1).norm(dim=1) / true_norm
    return {
        "rmse": float(diff.square().mean().sqrt().item()),
        "relative_l2_mean": float(rel.mean().item()),
        "relative_l2_std": float(rel.std(unbiased=False).item()),
        "mean_abs_difference": float(diff.abs().mean().item()),
    }


def histogram_l1(a: list[float], b: list[float]) -> float:
    return float(sum(abs(x - y) for x, y in zip(a, b)))


def checkpoint_path(output_dir: Path, config: ModelConfig) -> Path:
    return output_dir / f"q16_{config.name}.pt"


def train_q16(
    config: ModelConfig,
    phi8: torch.Tensor,
    d16_true: torch.Tensor,
    params: Phi4Params,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[ConditionalDetailFlow, list[dict[str, float | str | int]]]:
    ckpt = checkpoint_path(args.output_dir, config)
    flow = ConditionalDetailFlow(config.layers, config.hidden_channels, config.cnn_depth).to(device)
    if args.reuse_checkpoints and ckpt.exists():
        state = torch.load(ckpt, map_location=device, weights_only=False)
        flow.load_state_dict(state["model"])
        flow.eval()
        print(f"loaded {config.name} from {ckpt}")
        return flow, state.get("history", [])

    loader = DataLoader(TensorDataset(phi8.unsqueeze(1).float(), d16_true.float()), batch_size=args.batch_size, shuffle=True)
    opt = torch.optim.Adam(flow.parameters(), lr=args.lr)
    history: list[dict[str, float | str | int]] = []
    for phase, epochs in (("mle", args.mle_epochs), ("reverse_kl", config.reverse_kl_epochs)):
        for epoch in range(1, epochs + 1):
            flow.train()
            losses = []
            for phi_c, d_true in loader:
                phi_c = phi_c.to(device)
                d_true = d_true.to(device)
                opt.zero_grad(set_to_none=True)
                if phase == "mle":
                    _, logq_true = flow.inverse_logq(d_true, phi_c)
                    loss = -logq_true.mean()
                else:
                    d, logq = flow.sample(phi_c)
                    phi_rec = reconstruct_from_average_block(phi_c[:, 0], d)
                    loss = (phi4_action(phi_rec, params) + logq).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(flow.parameters(), 10.0)
                opt.step()
                losses.append(float(loss.detach().cpu().item()))
            mean_loss = sum(losses) / max(len(losses), 1)
            history.append({"phase": phase, "epoch": epoch, "loss": mean_loss})
            print(f"{config.name} {phase} epoch {epoch:03d}/{epochs:03d} loss {mean_loss:.6g}")

    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": flow.state_dict(),
            "history": history,
            "config": {
                "name": config.name,
                "layers": config.layers,
                "hidden_channels": config.hidden_channels,
                "cnn_depth": config.cnn_depth,
                "mle_epochs": args.mle_epochs,
                "reverse_kl_epochs": config.reverse_kl_epochs,
            },
        },
        ckpt,
    )
    flow.eval()
    return flow, history


@torch.no_grad()
def restricted_detail_mcmc(
    phi8: torch.Tensor,
    d_init: torch.Tensor,
    params: Phi4Params,
    args: argparse.Namespace,
    generator: torch.Generator,
) -> tuple[torch.Tensor, dict[str, float]]:
    d = d_init.clone()
    phi = reconstruct_from_average_block(phi8, d)
    action = phi4_action(phi, params)
    accepts = 0
    proposals = 0
    coarse_y, coarse_x = phi8.shape[-2:]
    for _ in range(args.n_sweeps):
        for y0 in range(0, coarse_y, args.patch_size):
            for x0 in range(0, coarse_x, args.patch_size):
                y1 = min(y0 + args.patch_size, coarse_y)
                x1 = min(x0 + args.patch_size, coarse_x)
                d_new = d.clone()
                noise = torch.randn(d[:, :, y0:y1, x0:x1].shape, generator=generator, device=d.device)
                d_new[:, :, y0:y1, x0:x1] = d[:, :, y0:y1, x0:x1] + args.step_size * noise
                phi_new = reconstruct_from_average_block(phi8, d_new)
                action_new = phi4_action(phi_new, params)
                log_accept = -action_new + action
                log_u = torch.log(torch.rand(log_accept.shape, generator=generator, device=d.device))
                accept = log_u < log_accept
                if accept.any():
                    d[accept] = d_new[accept]
                    action[accept] = action_new[accept]
                accepts += int(accept.sum().item())
                proposals += int(accept.numel())
    return d, {"patch_acceptance_rate": accepts / float(proposals)}


def evaluate_phi16(
    phi: torch.Tensor,
    true_phi16: torch.Tensor,
    true_summary: dict[str, object],
    params: Phi4Params,
) -> dict[str, object]:
    summary = ensemble_summary(phi.cpu(), params)
    summary["relative_errors"] = rel_errors(summary, true_summary)
    summary["aggregate_abs_rel_error"] = aggregate_abs_rel(summary, true_summary)
    true_hist = histogram_values(true_phi16)
    rec_hist = histogram_values(phi)
    summary["phi16_value_histogram"] = rec_hist
    summary["phi16_value_histogram_l1_vs_true"] = histogram_l1(rec_hist, true_hist)
    summary["reconstruction_distance"] = reconstruction_distance(phi.cpu(), true_phi16.cpu())
    return summary


def logw_for(flow: ConditionalDetailFlow, phi8: torch.Tensor, phi16_rec: torch.Tensor, d: torch.Tensor, params: Phi4Params) -> dict[str, float]:
    _, logq = flow.inverse_logq(d, phi8.unsqueeze(1))
    logw = -phi4_action(phi16_rec, params) - logq
    return logw_stats(logw)


def best_name(results: dict[str, dict[str, object]], key: str) -> str:
    return min(results, key=lambda name: float(results[name][key]["aggregate_abs_rel_error"]))


def stable_name_seed(name: str) -> int:
    return sum((i + 1) * ord(char) for i, char in enumerate(name)) % 100000


def write_report(path: Path, summary: dict[str, object]) -> None:
    results = summary["results"]
    true16 = summary["true_phi16"]
    lines = [
        "# Multilevel q16 Tuning",
        "",
        "Focused study of the first-level `8x8 -> 16x16` conditional detail flow `q16(d16 | phi8)`.",
        "",
        f"Data: `{summary['n_eval']}` evaluation configurations, lambda `{summary['lambda']}`, kappa `{summary['kappa_fine']}`.",
        "",
        "## True 16x16 Reference",
        "",
        "| S mean | S std | phi2 | Binder | NN corr | susceptibility | low-p | high-p |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| {true16['S_mean']:.6g} | {true16['S_std']:.6g} | {true16['phi2']:.6g} | {true16['binder']:.6g} | "
        f"{true16['NN_corr']:.6g} | {true16['susceptibility']:.6g} | {true16['low_p_power']:.6g} | {true16['high_p_power']:.6g} |",
        "",
        "## Raw q16 Reconstructions",
        "",
        "| run | layers | hidden | depth | RK epochs | S mean | S std | phi2 | Binder | NN corr | susc | low-p | high-p | hist L1 | agg err | RMSE | rel L2 | logw std | ESS/N |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, result in results.items():
        raw = result["raw"]
        cfg = result["config"]
        dist = raw["reconstruction_distance"]
        logw = raw["logw"]
        lines.append(
            f"| {name} | {cfg['layers']} | {cfg['hidden_channels']} | {cfg['cnn_depth']} | {cfg['reverse_kl_epochs']} | "
            f"{raw['S_mean']:.6g} | {raw['S_std']:.6g} | {raw['phi2']:.6g} | {raw['binder']:.6g} | "
            f"{raw['NN_corr']:.6g} | {raw['susceptibility']:.6g} | {raw['low_p_power']:.6g} | "
            f"{raw['high_p_power']:.6g} | {raw['phi16_value_histogram_l1_vs_true']:.6g} | "
            f"{raw['aggregate_abs_rel_error']:.6g} | {dist['rmse']:.6g} | "
            f"{dist['relative_l2_mean']:.6g} | {logw['std']:.6g} | {logw['ess_over_n']:.6g} |"
        )
    lines.extend(
        [
        "",
        "## Diagnostic-Only 8->16 Restricted MCMC",
        "",
        "Important: this is not a valid correction step for the multilevel sampler. At the intermediate 16x16 level, the correct target is the exact blocked effective action induced from the 32x32 theory, `S16^blocked`. That action is unknown and is not the bare phi4 action used here. These numbers are retained only as a diagnostic of what the bare-action local update would do, and they are not evidence that q16 can or cannot be repaired.",
        "",
        f"Settings: patch_size `{summary['restricted_mcmc']['patch_size']}`, step_size `{summary['restricted_mcmc']['step_size']}`, n_sweeps `{summary['restricted_mcmc']['n_sweeps']}`.",
            "",
            "| run | acceptance | S mean | S std | phi2 | Binder | NN corr | susc | low-p | high-p | hist L1 | agg err | RMSE | rel L2 | block-to-8 max error |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, result in results.items():
        mcmc = result["restricted_mcmc"]
        dist = mcmc["reconstruction_distance"]
        lines.append(
            f"| {name} | {mcmc['patch_acceptance_rate']:.6g} | {mcmc['S_mean']:.6g} | {mcmc['S_std']:.6g} | "
            f"{mcmc['phi2']:.6g} | {mcmc['binder']:.6g} | {mcmc['NN_corr']:.6g} | "
            f"{mcmc['susceptibility']:.6g} | {mcmc['low_p_power']:.6g} | {mcmc['high_p_power']:.6g} | "
            f"{mcmc['phi16_value_histogram_l1_vs_true']:.6g} | {mcmc['aggregate_abs_rel_error']:.6g} | "
            f"{dist['rmse']:.6g} | {dist['relative_l2_mean']:.6g} | "
            f"{mcmc['block_to_8_consistency_max_error']:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Main Answers",
            "",
            f"1. Can q16 be trained well enough? {summary['answers']['can_q16_match_true_phi16']}",
            f"2. Can the 8->16 restricted-MCMC numbers be used as a correction test? {summary['answers']['intermediate_restricted_mcmc_validity']}",
            f"3. Recommended raw q16 checkpoint: `{summary['answers']['recommended_raw_checkpoint']}`",
            f"4. Recommended next multilevel test: {summary['answers']['recommended_next_multilevel_test']}",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def plot_outputs(path: Path, true_phi16: torch.Tensor, samples: dict[str, torch.Tensor], mcmc_samples: dict[str, torch.Tensor], params: Phi4Params) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0))
    ax = axes[0, 0]
    ax.hist(phi4_action(true_phi16, params).detach().cpu().numpy(), bins=40, density=True, alpha=0.45, label="true_phi16")
    for name, phi in samples.items():
        ax.hist(phi4_action(phi, params).detach().cpu().numpy(), bins=40, density=True, histtype="step", lw=1.2, label=name)
    ax.set_xlabel("S16")
    ax.set_ylabel("density")
    ax.legend(fontsize=7)

    ax = axes[0, 1]
    for name, phi in {"true_phi16": true_phi16, **samples}.items():
        k, p = radial_power_spectrum(phi.cpu())
        ax.plot(k.numpy(), p.numpy(), marker="o", ms=2.5, label=name)
    ax.set_yscale("log")
    ax.set_xlabel("|p| shell")
    ax.set_ylabel("<|phi(p)|^2>")
    ax.legend(fontsize=7)

    ax = axes[1, 0]
    ax.hist(true_phi16.detach().cpu().flatten().numpy(), bins=50, density=True, alpha=0.45, label="true_phi16")
    for name, phi in samples.items():
        ax.hist(phi.detach().cpu().flatten().numpy(), bins=50, density=True, histtype="step", lw=1.2, label=name)
    ax.set_xlabel("phi16 value")
    ax.set_ylabel("density")
    ax.legend(fontsize=7)

    ax = axes[1, 1]
    names = list(samples)
    raw = [float(samples[name].new_tensor(0.0).item()) for name in names]
    mcmc = [float(mcmc_samples[name].new_tensor(0.0).item()) for name in names]
    for i, name in enumerate(names):
        raw[i] = float(phi4_action(samples[name], params).mean().item())
        mcmc[i] = float(phi4_action(mcmc_samples[name], params).mean().item())
    x = torch.arange(len(names)).numpy()
    ax.bar(x - 0.18, raw, width=0.36, label="raw")
    ax.bar(x + 0.18, mcmc, width=0.36, label="8->16 bare-action MCMC diagnostic")
    ax.axhline(float(phi4_action(true_phi16, params).mean().item()), color="k", lw=1.0, ls="--", label="true")
    ax.set_xticks(x, names, rotation=35, ha="right")
    ax.set_ylabel("mean S16")
    ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


@torch.no_grad()
def evaluate_config(
    config: ModelConfig,
    flow: ConditionalDetailFlow,
    phi8_eval: torch.Tensor,
    phi16_eval: torch.Tensor,
    true_summary: dict[str, object],
    params: Phi4Params,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, object], torch.Tensor, torch.Tensor]:
    name_seed = stable_name_seed(config.name)
    generator = torch.Generator(device=device).manual_seed(args.seed + 1000 + name_seed)
    d_raw, _, _, logq = flow.sample_with_decomposition(phi8_eval.unsqueeze(1), generator=generator)
    phi_raw = reconstruct_from_average_block(phi8_eval, d_raw)
    raw = evaluate_phi16(phi_raw, phi16_eval.cpu(), true_summary, params)
    raw["logw"] = logw_stats(-phi4_action(phi_raw, params) - logq)
    raw["block_to_8_consistency_max_error"] = float((block_average(phi_raw) - phi8_eval).abs().max().item())

    d_mcmc, mcmc_info = restricted_detail_mcmc(
        phi8_eval,
        d_raw,
        params,
        args,
        torch.Generator(device=device).manual_seed(args.seed + 2000 + name_seed),
    )
    phi_mcmc = reconstruct_from_average_block(phi8_eval, d_mcmc)
    mcmc = evaluate_phi16(phi_mcmc, phi16_eval.cpu(), true_summary, params)
    mcmc.update(mcmc_info)
    mcmc["block_to_8_consistency_max_error"] = float((block_average(phi_mcmc) - phi8_eval).abs().max().item())

    result = {
        "config": {
            "layers": config.layers,
            "hidden_channels": config.hidden_channels,
            "cnn_depth": config.cnn_depth,
            "mle_epochs": args.mle_epochs,
            "reverse_kl_epochs": config.reverse_kl_epochs,
            "checkpoint": str(checkpoint_path(args.output_dir, config)),
        },
        "raw": raw,
        "restricted_mcmc": mcmc,
    }
    return result, phi_raw.cpu(), phi_mcmc.cpu()


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    params = Phi4Params(kappa=args.kappa_fine, lam=args.lam)
    reverse_kl_epochs = [int(x) for x in args.reverse_kl_epochs.split(",") if x.strip()]

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
    phi16, _ = average_block(phi32)
    phi8, d16_true = average_block(phi16)
    n_eval = min(args.n_eval, len(phi16))
    phi16_eval = phi16[:n_eval].to(device)
    phi8_eval = phi8[:n_eval].to(device)
    true_summary = ensemble_summary(phi16_eval.cpu(), params)
    true_hist = histogram_values(phi16_eval)

    results: dict[str, dict[str, object]] = {}
    raw_samples: dict[str, torch.Tensor] = {}
    mcmc_samples: dict[str, torch.Tensor] = {}
    for config in model_grid(reverse_kl_epochs):
        flow, history = train_q16(config, phi8, d16_true, params, args, device)
        result, phi_raw, phi_mcmc = evaluate_config(config, flow, phi8_eval, phi16_eval, true_summary, params, args, device)
        result["training_history"] = history
        result["raw"]["phi16_value_histogram_l1_vs_true"] = histogram_l1(result["raw"]["phi16_value_histogram"], true_hist)
        result["restricted_mcmc"]["phi16_value_histogram_l1_vs_true"] = histogram_l1(
            result["restricted_mcmc"]["phi16_value_histogram"],
            true_hist,
        )
        results[config.name] = result
        raw_samples[config.name] = phi_raw
        mcmc_samples[config.name] = phi_mcmc
        print(
            config.name,
            "raw_agg",
            f"{result['raw']['aggregate_abs_rel_error']:.6g}",
            "mcmc_agg",
            f"{result['restricted_mcmc']['aggregate_abs_rel_error']:.6g}",
        )

    raw_best = best_name(results, "raw")
    mcmc_best = best_name(results, "restricted_mcmc")
    raw_best_err = float(results[raw_best]["raw"]["aggregate_abs_rel_error"])
    mcmc_best_err = float(results[mcmc_best]["restricted_mcmc"]["aggregate_abs_rel_error"])
    summary = {
        "fine_size": args.fine_size,
        "middle_size": 16,
        "coarse_size": 8,
        "lambda": args.lam,
        "kappa_fine": args.kappa_fine,
        "n_eval": n_eval,
        "true_phi16": true_summary,
        "true_phi16_value_histogram": true_hist,
        "restricted_mcmc": {
            "patch_size": args.patch_size,
            "step_size": args.step_size,
            "n_sweeps": args.n_sweeps,
        },
        "results": results,
        "answers": {
            "can_q16_match_true_phi16": (
                f"Best raw aggregate error is {raw_best_err:.6g} from {raw_best}; "
                + ("this is a clear improvement but still not a close match." if raw_best_err > 0.1 else "this is a close observable-level match.")
            ),
            "intermediate_restricted_mcmc_validity": (
                "Invalid as a correction or repair test. Restricted detail MCMC is exact only at the final level where the known fine action S32 is used. "
                "At 8->16 it would require the exact blocked action S16^blocked induced by the 32x32 theory, which is unknown; the bare phi4 S16 numbers here are diagnostic-only."
            ),
            "recommended_raw_checkpoint": str(checkpoint_path(args.output_dir, next(c for c in model_grid(reverse_kl_epochs) if c.name == raw_best))),
            "diagnostic_only_best_intermediate_mcmc": {
                "checkpoint": str(checkpoint_path(args.output_dir, next(c for c in model_grid(reverse_kl_epochs) if c.name == mcmc_best))),
                "aggregate_abs_rel_error": mcmc_best_err,
                "use_in_recommendations": False,
            },
            "recommended_next_multilevel_test": (
                f"Use the best raw learned q16 checkpoint ({checkpoint_path(args.output_dir, next(c for c in model_grid(reverse_kl_epochs) if c.name == raw_best))}) "
                "for the next 8->16->32 lift, and apply correction only at the final 16->32 detail level using S32. "
                "Alternatively, improve q16 as a learned conditional from blocked data before rerunning the two-step test."
            ),
        },
    }

    (args.output_dir / "multilevel_q16_tuning_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_report(args.output_dir / "multilevel_q16_tuning_report.md", summary)
    plot_outputs(args.output_dir / "multilevel_q16_tuning_plots.pdf", phi16_eval.cpu(), raw_samples, mcmc_samples, params)
    print(f"wrote {args.output_dir / 'multilevel_q16_tuning_summary.json'}")
    print(f"wrote {args.output_dir / 'multilevel_q16_tuning_report.md'}")
    print(f"wrote {args.output_dir / 'multilevel_q16_tuning_plots.pdf'}")
    print("best_raw", raw_best, raw_best_err)
    print("best_restricted_mcmc", mcmc_best, mcmc_best_err)


if __name__ == "__main__":
    main()
