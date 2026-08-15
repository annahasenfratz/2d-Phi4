"""Train a fine16 upscaling flow from independently generated kappa_c=0.30 coarse fields."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/matplotlib")

from inverse_blocking_flow.data import load_or_generate_fine_configs
from inverse_blocking_flow.flow import ConditionalDetailFlow, make_conditioning
from inverse_blocking_flow.haar import prolong_constant, soft_kernel_term, soft_reconstruct
from inverse_blocking_flow.logw_kappa_scan_blocked_coarse import aggregate_abs_rel, ensemble_summary
from inverse_blocking_flow.phi4 import (
    Phi4Params,
    checkerboard_metropolis_sweep,
    generate_phi4_configs,
    phi4_action,
)


REF_KAPPAS = [0.315, 0.320, 0.325]
OBS_KEYS = ["S_mean", "S_std", "phi2", "binder", "NN_corr", "susceptibility", "low_p_power", "high_p_power"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coarse-size", type=int, default=8)
    parser.add_argument("--fine-size", type=int, default=16)
    parser.add_argument("--kappa-c", type=float, default=0.30)
    parser.add_argument("--kappa-f-initial", type=float, default=0.32)
    parser.add_argument("--kappa-f-min", type=float, default=0.30)
    parser.add_argument("--kappa-f-max", type=float, default=0.335)
    parser.add_argument("--lambda-f", dest="lam", type=float, default=1.0)
    parser.add_argument("--soft-alpha", type=float, default=2.0)
    parser.add_argument("--n-configs", type=int, default=512)
    parser.add_argument("--n-eval", type=int, default=512)
    parser.add_argument("--coarse-data-path", type=Path, default=Path("inverse_blocking_flow/outputs_fine16/coarse_kappac030_configs.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("inverse_blocking_flow/outputs_fine16"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--kappa-lr-mult", type=float, default=0.2)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--hidden-channels", type=int, default=48)
    parser.add_argument("--cnn-depth", type=int, default=4)
    parser.add_argument("--conditioning-mode", choices=("physics",), default="physics")
    parser.add_argument("--burn-in", type=int, default=200)
    parser.add_argument("--sample-interval", type=int, default=10)
    parser.add_argument("--proposal-width", type=float, default=1.0)
    parser.add_argument("--reference-burn-in", type=int, default=200)
    parser.add_argument("--correction-sweeps", type=str, default="0,1,2,5,10,20")
    parser.add_argument("--correction-n", type=int, default=128)
    parser.add_argument("--skip-correction", action="store_true")
    parser.add_argument("--seed", type=int, default=969696)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    return parser


def raw_from_kappa(kappa: float, lo: float, hi: float) -> float:
    frac = (kappa - lo) / (hi - lo)
    frac = min(max(frac, 1e-6), 1.0 - 1e-6)
    return math.log(frac / (1.0 - frac))


def bounded_kappa(raw: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return lo + (hi - lo) * torch.sigmoid(raw)


def tensor_stats(x: torch.Tensor) -> dict[str, float]:
    x = x.detach().float().cpu()
    return {
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
    }


def stabilized_logw_stats(logw: torch.Tensor) -> dict[str, float]:
    logw = logw.detach().float().cpu()
    centered = logw - logw.mean()
    log_norm = torch.logsumexp(centered, dim=0)
    log_ess = 2.0 * log_norm - torch.logsumexp(2.0 * centered, dim=0)
    diffs = centered.unsqueeze(0) - centered.unsqueeze(1)
    accept = torch.minimum(torch.ones_like(diffs), torch.exp(diffs.clamp(max=80.0)))
    target_weights = torch.softmax(centered, dim=0)
    return {
        "std_logw_centered": float(centered.std(unbiased=False).item()),
        "ess_over_n": float((torch.exp(log_ess) / centered.numel()).item()),
        "independence_acceptance_proxy": float((accept * target_weights.view(-1, 1)).sum(dim=0).mean().item()),
        "proposal_pair_acceptance_proxy": float(accept.mean().item()),
        "min_logw_centered": float(centered.min().item()),
        "max_logw_centered": float(centered.max().item()),
    }


def aggregate_for_keys(obs: dict[str, float], ref: dict[str, float]) -> float:
    return float(sum(abs((obs[k] - ref[k]) / ref[k]) for k in OBS_KEYS if abs(ref[k]) > 1e-14) / len(OBS_KEYS))


def load_or_generate_coarse(args: argparse.Namespace) -> torch.Tensor:
    if args.coarse_data_path.exists():
        data = torch.load(args.coarse_data_path, map_location="cpu")
        if isinstance(data, dict):
            data = data["phi"]
        if data.shape[-2:] != (args.coarse_size, args.coarse_size):
            raise ValueError(f"coarse data shape {data.shape[-2:]} does not match {(args.coarse_size, args.coarse_size)}")
        return data[: args.n_configs].float()
    phi = generate_phi4_configs(
        args.n_configs,
        args.coarse_size,
        Phi4Params(kappa=args.kappa_c, lam=args.lam),
        burn_in=args.burn_in,
        interval=args.sample_interval,
        batch_size=args.batch_size,
        proposal_width=args.proposal_width,
        seed=args.seed,
        device=args.device,
    ).float()
    args.coarse_data_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"phi": phi, "kappa": args.kappa_c, "lambda": args.lam}, args.coarse_data_path)
    return phi


def train_upscaler(
    flow: ConditionalDetailFlow,
    raw_kappa: torch.Tensor,
    psi: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, float]]:
    loader = DataLoader(TensorDataset(psi), batch_size=args.batch_size, shuffle=True)
    opt = torch.optim.Adam(
        [
            {"params": flow.parameters(), "lr": args.lr},
            {"params": [raw_kappa], "lr": args.lr * args.kappa_lr_mult},
        ]
    )
    history = []
    for epoch in range(1, args.epochs + 1):
        losses = []
        kappas = []
        for (psi_b,) in loader:
            psi_b = psi_b.to(device)
            cond = make_conditioning(psi_b, args.conditioning_mode)
            opt.zero_grad(set_to_none=True)
            u, logq = flow.sample(cond)
            phi = soft_reconstruct(psi_b[:, 0], u)
            kappa = bounded_kappa(raw_kappa, args.kappa_f_min, args.kappa_f_max)
            action = phi4_action(phi, Phi4Params(kappa=float(kappa.detach().cpu().item()), lam=args.lam))
            # Keep the action differentiable with respect to kappa.
            neighbor_sum = torch.roll(phi, -1, dims=-2) + torch.roll(phi, -1, dims=-1)
            local = phi.square() + args.lam * (phi.square() - 1.0).square()
            action = (local - 2.0 * kappa * phi * neighbor_sum).sum(dim=(-2, -1))
            loss = (action + soft_kernel_term(u, args.soft_alpha) + logq).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(flow.parameters()) + [raw_kappa], 10.0)
            opt.step()
            losses.append(float(loss.detach().cpu().item()))
            kappas.append(float(kappa.detach().cpu().item()))
        row = {
            "epoch": epoch,
            "loss": sum(losses) / len(losses),
            "kappa_f": sum(kappas) / len(kappas),
        }
        history.append(row)
        print(f"epoch {epoch:04d} reverse_kl loss {row['loss']:.6g} kappa_f {row['kappa_f']:.6g}", flush=True)
    return history


@torch.no_grad()
def sample_upscaled(
    flow: ConditionalDetailFlow,
    psi: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    seed_offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    psi = psi[: args.n_eval].to(device)
    cond = make_conditioning(psi, args.conditioning_mode)
    gen = torch.Generator(device=device).manual_seed(args.seed + 1000 + seed_offset)
    u, logq = flow.sample(cond, generator=gen)
    phi = soft_reconstruct(psi[:, 0], u)
    return phi.cpu(), u.cpu(), logq.cpu()


def generate_reference(args: argparse.Namespace, kappa: float) -> torch.Tensor:
    path = args.output_dir / f"fine_reference_kappa_{str(kappa).replace('.', 'p')}.pt"
    return load_or_generate_fine_configs(
        path,
        n_configs=args.n_eval,
        fine_size=args.fine_size,
        params=Phi4Params(kappa=kappa, lam=args.lam),
        burn_in=args.reference_burn_in,
        interval=args.sample_interval,
        batch_size=args.batch_size,
        proposal_width=args.proposal_width,
        seed=args.seed + int(round(10000 * kappa)),
        device=args.device,
    ).float()


def naive_upscale(psi: torch.Tensor) -> torch.Tensor:
    return prolong_constant(psi[:, 0])


@torch.no_grad()
def correction_test(
    generated_phi: torch.Tensor,
    refs: dict[str, dict[str, float]],
    args: argparse.Namespace,
    target_kappa: float = 0.32,
) -> dict[str, object]:
    sweeps = [int(x) for x in args.correction_sweeps.split(",") if x.strip()]
    n = min(args.correction_n, generated_phi.shape[0])
    target_ref = refs[f"{target_kappa:.3f}"]
    params = Phi4Params(kappa=target_kappa, lam=args.lam)
    starts = {
        "upscaled": generated_phi[:n].clone(),
        "hot": 0.5 * torch.randn((n, args.fine_size, args.fine_size), generator=torch.Generator().manual_seed(args.seed + 77)),
        "cold": torch.zeros((n, args.fine_size, args.fine_size)),
    }
    out: dict[str, list[dict[str, object]]] = {}
    for name, phi0 in starts.items():
        phi = phi0.clone()
        rows = []
        gen = torch.Generator().manual_seed(args.seed + 9100)
        current = 0
        for target in sweeps:
            for _ in range(target - current):
                checkerboard_metropolis_sweep(phi, params, args.proposal_width, gen)
            current = target
            obs = ensemble_summary(phi, params)
            rows.append(
                {
                    "sweeps": target,
                    "observables": obs,
                    "aggregate_error_vs_ref_0p320": aggregate_for_keys(obs, target_ref),
                }
            )
        out[name] = rows
    return out


def write_report(path: Path, summary: dict[str, object]) -> None:
    gen = summary["generated"]
    learned = summary["learned_kappa_f"]
    refs = summary["reference_comparison"]
    best = min(refs.items(), key=lambda item: item[1]["aggregate_error_generated"])[0]
    naive_best = min(refs.items(), key=lambda item: item[1]["aggregate_error_naive"])[0]
    logw = summary["logw_diagnostics"]
    lines = [
        "# kappa_c=0.30 Trainable kappa_f Upscale",
        "",
        f"Learned kappa_f moved from `{learned['initial']:.6g}` to `{learned['final']:.6g}` with bounds `{learned['min']}` to `{learned['max']}`.",
        "",
        "## Main Answers",
        "",
        f"1. Best reference match for generated fields is kappa_f `{best}`.",
        f"2. kappa_f {'stayed near' if abs(learned['final'] - learned['initial']) < 0.005 else 'drifted from'} 0.32; final `{learned['final']:.6g}`.",
        f"3. Generated beats naive upscaling for best reference `{best}` if `{refs[best]['aggregate_error_generated']:.6g} < {refs[best]['aggregate_error_naive']:.6g}`.",
        "4. Correction-MCMC results are in the correction table below.",
        "",
        "## Generated Fine Observables",
        "",
        "| action kappa | S mean | S std | phi2 | Binder | susceptibility | NN corr | low-p | high-p |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, obs in gen["observables"].items():
        lines.append(
            f"| {key} | {obs['S_mean']:.6g} | {obs['S_std']:.6g} | {obs['phi2']:.6g} | {obs['binder']:.6g} | "
            f"{obs['susceptibility']:.6g} | {obs['NN_corr']:.6g} | {obs['low_p_power']:.6g} | {obs['high_p_power']:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Reference Match",
            "",
            "| reference kappa | generated agg err | naive agg err |",
            "|---:|---:|---:|",
        ]
    )
    for kappa, row in refs.items():
        lines.append(f"| {kappa} | {row['aggregate_error_generated']:.6g} | {row['aggregate_error_naive']:.6g} |")
    lines.extend(
        [
            "",
            "## Logweight Diagnostic",
            "",
            f"- centered logw std: `{logw['std_logw_centered']:.6g}`",
            f"- ESS/N: `{logw['ess_over_n']:.6g}`",
            f"- A/R proxy: `{logw['independence_acceptance_proxy']:.6g}`",
        ]
    )
    if summary.get("correction_mcmc"):
        lines.extend(["", "## Correction MCMC", "", "| start | sweeps | agg err vs ref kappa=0.320 |", "|---|---:|---:|"])
        for start, rows in summary["correction_mcmc"].items():
            for row in rows:
                lines.append(f"| {start} | {row['sweeps']} | {row['aggregate_error_vs_ref_0p320']:.6g} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_outputs(path: Path, summary: dict[str, object]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    history = summary["training_history"]
    refs = summary["reference_comparison"]
    with PdfPages(path) as pdf:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot([row["epoch"] for row in history], [row["loss"] for row in history])
        axes[0].set_xlabel("epoch")
        axes[0].set_ylabel("reverse-KL loss")
        axes[1].plot([row["epoch"] for row in history], [row["kappa_f"] for row in history])
        axes[1].axhline(0.32, color="k", ls="--")
        axes[1].set_xlabel("epoch")
        axes[1].set_ylabel("kappa_f")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4))
        labels = list(refs.keys())
        ax.plot(labels, [refs[k]["aggregate_error_generated"] for k in labels], marker="o", label="generated")
        ax.plot(labels, [refs[k]["aggregate_error_naive"] for k in labels], marker="s", label="naive")
        ax.set_xlabel("reference kappa_f")
        ax.set_ylabel("aggregate observable error")
        ax.legend()
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        if summary.get("correction_mcmc"):
            fig, ax = plt.subplots(figsize=(7, 4))
            for start, rows in summary["correction_mcmc"].items():
                ax.plot([r["sweeps"] for r in rows], [r["aggregate_error_vs_ref_0p320"] for r in rows], marker="o", label=start)
            ax.set_xlabel("fine MCMC sweeps")
            ax.set_ylabel("agg err vs ref kappa=0.320")
            ax.legend()
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    if args.coarse_size * 2 != args.fine_size:
        raise ValueError("fine_size must be twice coarse_size")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    coarse = load_or_generate_coarse(args)
    psi = coarse.unsqueeze(1).float()
    flow = ConditionalDetailFlow(args.layers, args.hidden_channels, args.cnn_depth, 6, 4).to(device)
    raw = torch.tensor(
        raw_from_kappa(args.kappa_f_initial, args.kappa_f_min, args.kappa_f_max),
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    history = train_upscaler(flow, raw, psi, args, device)
    learned_kappa = float(bounded_kappa(raw, args.kappa_f_min, args.kappa_f_max).detach().cpu().item())
    generated_phi, u, logq = sample_upscaled(flow, psi, args, device)
    kernel = soft_kernel_term(u, args.soft_alpha)
    learned_params = Phi4Params(kappa=learned_kappa, lam=args.lam)
    fixed_params = Phi4Params(kappa=args.kappa_f_initial, lam=args.lam)
    logw = -phi4_action(generated_phi, learned_params) - kernel - logq
    naive_phi = naive_upscale(psi[: args.n_eval])

    generated_obs = {
        "learned": ensemble_summary(generated_phi, learned_params),
        "fixed_0p320": ensemble_summary(generated_phi, fixed_params),
    }
    naive_obs = {
        "learned": ensemble_summary(naive_phi, learned_params),
        "fixed_0p320": ensemble_summary(naive_phi, fixed_params),
    }
    refs = {}
    ref_obs = {}
    for kappa in REF_KAPPAS:
        ref_phi = generate_reference(args, kappa)
        ref_key = f"{kappa:.3f}"
        obs = ensemble_summary(ref_phi, Phi4Params(kappa=kappa, lam=args.lam))
        ref_obs[ref_key] = obs
        gen_obs_at_ref = ensemble_summary(generated_phi, Phi4Params(kappa=kappa, lam=args.lam))
        naive_obs_at_ref = ensemble_summary(naive_phi, Phi4Params(kappa=kappa, lam=args.lam))
        refs[ref_key] = {
            "reference_observables": obs,
            "generated_observables_at_reference_kappa": gen_obs_at_ref,
            "naive_observables_at_reference_kappa": naive_obs_at_ref,
            "aggregate_error_generated": aggregate_for_keys(gen_obs_at_ref, obs),
            "aggregate_error_naive": aggregate_for_keys(naive_obs_at_ref, obs),
        }
    correction = None
    if not args.skip_correction:
        correction = correction_test(generated_phi, ref_obs, args, target_kappa=args.kappa_f_initial)

    summary = {
        "setup": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "learned_kappa_f": {
            "initial": args.kappa_f_initial,
            "final": learned_kappa,
            "min": args.kappa_f_min,
            "max": args.kappa_f_max,
        },
        "training_history": history,
        "generated": {
            "observables": generated_obs,
            "naive_observables": naive_obs,
        },
        "reference_comparison": refs,
        "logw_diagnostics": stabilized_logw_stats(logw),
        "logq": tensor_stats(logq),
        "kernel_term": tensor_stats(kernel),
        "correction_mcmc": correction,
    }
    summary_path = args.output_dir / "kappac030_trainable_kappaf_upscale_summary.json"
    report_path = args.output_dir / "kappac030_trainable_kappaf_upscale_report.md"
    plots_path = args.output_dir / "kappac030_trainable_kappaf_upscale_plots.pdf"
    model_path = args.output_dir / "kappac030_trainable_kappaf_upscale_model.pt"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_report(report_path, summary)
    plot_outputs(plots_path, summary)
    torch.save(
        {
            "model": flow.state_dict(),
            "raw_kappa": raw.detach().cpu(),
            "learned_kappa_f": learned_kappa,
            "args": summary["setup"],
            "history": history,
        },
        model_path,
    )
    print(f"wrote {summary_path}")
    print(f"wrote {report_path}")
    print(f"wrote {plots_path}")
    print(f"wrote {model_path}")


if __name__ == "__main__":
    main()
