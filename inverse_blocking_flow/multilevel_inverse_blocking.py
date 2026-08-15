"""Two-level inverse-blocking diagnostic: 8x8 -> 16x16 -> 32x32."""

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
from inverse_blocking_flow.flow import ConditionalDetailFlow
from inverse_blocking_flow.haar import average_block, block_average, reconstruct_from_average_block
from inverse_blocking_flow.phi4 import (
    Phi4Params,
    binder_cumulant,
    mean_phi2,
    nearest_neighbor_correlator,
    phi4_action,
)


OBS_KEYS = ["S_mean", "S_std", "phi2", "binder", "NN_corr", "susceptibility", "low_p_power", "high_p_power"]
AGG_KEYS = ["S_mean", "S_std", "phi2", "NN_corr", "high_p_power"]


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
    parser.add_argument("--mle-epochs", type=int, default=3)
    parser.add_argument("--reverse-kl-epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--cnn-depth", type=int, default=3)
    parser.add_argument("--burn-in", type=int, default=200)
    parser.add_argument("--sample-interval", type=int, default=10)
    parser.add_argument("--proposal-width", type=float, default=1.0)
    parser.add_argument("--reuse-checkpoints", action="store_true")
    parser.add_argument("--restricted-mcmc", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--patch-size", type=int, default=2)
    parser.add_argument("--step-size", type=float, default=0.1)
    parser.add_argument("--n-sweeps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=868686)
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
        "low_p_power": float(power[:, low_mask].mean().item()),
        "high_p_power": float(power[:, high_mask].mean().item()),
    }


def radial_power_spectrum(phi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    centered = phi - phi.mean(dim=(-2, -1), keepdim=True)
    fft = torch.fft.fftn(centered, dim=(-2, -1))
    power = (fft.real.square() + fft.imag.square()).mean(dim=0) / (phi.shape[-2] * phi.shape[-1])
    ly, lx = power.shape
    ky = torch.fft.fftfreq(ly, d=1.0, device=power.device) * ly
    kx = torch.fft.fftfreq(lx, d=1.0, device=power.device) * lx
    yy, xx = torch.meshgrid(ky, kx, indexing="ij")
    shell = torch.round(torch.sqrt(xx.square() + yy.square())).long()
    out_k = []
    out_p = []
    for s in range(int(shell.max().item()) + 1):
        mask = shell == s
        if mask.any():
            out_k.append(torch.sqrt(xx.square() + yy.square())[mask].mean())
            out_p.append(power[mask].mean())
    return torch.stack(out_k).cpu(), torch.stack(out_p).cpu()


def ensemble_summary(phi: torch.Tensor, params: Phi4Params) -> dict[str, float | dict[str, float]]:
    action = phi4_action(phi, params)
    summary: dict[str, float | dict[str, float]] = {
        "S_mean": float(action.mean().item()),
        "S_std": float(action.std(unbiased=False).item()),
        "phi2": float(mean_phi2(phi).mean().item()),
        "binder": float(binder_cumulant(phi).item()),
        "NN_corr": float(nearest_neighbor_correlator(phi).mean().item()),
        "susceptibility": float(susceptibility(phi).mean().item()),
    }
    summary.update(low_high_power(phi))
    summary["action_stats"] = tensor_stats(action)
    return summary


def logw_stats(logw: torch.Tensor) -> dict[str, float]:
    logw = logw.detach().float().cpu()
    log_norm = torch.logsumexp(logw, dim=0)
    log_ess = 2.0 * log_norm - torch.logsumexp(2.0 * logw, dim=0)
    ess = torch.exp(log_ess)
    out = tensor_stats(logw)
    out["ess_over_n"] = float((ess / logw.numel()).item())
    return out


def rel_errors(summary: dict[str, object], true_summary: dict[str, object]) -> dict[str, float]:
    out = {}
    for key in OBS_KEYS:
        target = float(true_summary[key])
        value = float(summary[key])
        out[key] = float("nan") if abs(target) < 1e-14 else (value - target) / target
    return out


def aggregate_abs_rel(summary: dict[str, object], true_summary: dict[str, object]) -> float:
    rel = rel_errors(summary, true_summary)
    return sum(abs(rel[key]) for key in AGG_KEYS) / len(AGG_KEYS)


def train_flow(
    name: str,
    cond: torch.Tensor,
    detail: torch.Tensor,
    params: Phi4Params,
    args: argparse.Namespace,
    checkpoint: Path,
    device: torch.device,
) -> ConditionalDetailFlow:
    flow = ConditionalDetailFlow(args.layers, args.hidden_channels, args.cnn_depth).to(device)
    if args.reuse_checkpoints and checkpoint.exists():
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        flow.load_state_dict(state["model"])
        flow.eval()
        print(f"loaded {name} from {checkpoint}")
        return flow

    dataset = TensorDataset(cond.unsqueeze(1).float(), detail.float())
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    opt = torch.optim.Adam(flow.parameters(), lr=args.lr)
    history = []
    for phase, epochs in (("mle", args.mle_epochs), ("reverse_kl", args.reverse_kl_epochs)):
        for epoch in range(1, epochs + 1):
            losses = []
            for phi_c, d_true in loader:
                phi_c = phi_c.to(device)
                d_true = d_true.to(device)
                opt.zero_grad(set_to_none=True)
                if phase == "mle":
                    _, logq = flow.inverse_logq(d_true, phi_c)
                    loss = -logq.mean()
                else:
                    d, logq = flow.sample(phi_c)
                    phi = reconstruct_from_average_block(phi_c[:, 0], d)
                    loss = (phi4_action(phi, params) + logq).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(flow.parameters(), 10.0)
                opt.step()
                losses.append(float(loss.detach().cpu().item()))
            mean_loss = sum(losses) / len(losses)
            history.append({"phase": phase, "epoch": epoch, "loss": mean_loss})
            print(f"{name} {phase} epoch {epoch:04d} loss {mean_loss:.6g}")

    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": flow.state_dict(),
            "history": history,
            "args": {
                "layers": args.layers,
                "hidden_channels": args.hidden_channels,
                "cnn_depth": args.cnn_depth,
            },
        },
        checkpoint,
    )
    flow.eval()
    return flow


@torch.no_grad()
def restricted_detail_mcmc(
    phi_c: torch.Tensor,
    d_init: torch.Tensor,
    params: Phi4Params,
    args: argparse.Namespace,
    generator: torch.Generator,
) -> tuple[torch.Tensor, dict[str, float]]:
    d = d_init.clone()
    phi = reconstruct_from_average_block(phi_c[:, 0], d)
    action = phi4_action(phi, params)
    accepts = 0
    proposals = 0
    coarse_y, coarse_x = phi_c.shape[-2:]
    for _ in range(args.n_sweeps):
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
    return d, {"patch_acceptance_rate": accepts / float(proposals)}


def plot_outputs(output_dir: Path, ensembles: dict[str, torch.Tensor], params: Phi4Params) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5))
    ax = axes[0, 0]
    for name, phi in ensembles.items():
        ax.hist(phi4_action(phi, params).detach().cpu().numpy(), bins=40, density=True, alpha=0.4, label=name)
    ax.set_xlabel("S_f")
    ax.set_ylabel("density")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    for name, phi in ensembles.items():
        k, p = radial_power_spectrum(phi)
        ax.plot(k.detach().numpy(), p.detach().numpy(), marker="o", ms=3, label=name)
    ax.set_yscale("log")
    ax.set_xlabel("|p| shell")
    ax.set_ylabel("<|phi(p)|^2>")
    ax.legend(fontsize=8)

    names = list(ensembles)
    x = torch.arange(len(names)).numpy()
    ax = axes[1, 0]
    ax.bar(x - 0.18, [float(mean_phi2(ensembles[n]).mean().item()) for n in names], width=0.36, label="phi2")
    ax.bar(x + 0.18, [float(binder_cumulant(ensembles[n]).item()) for n in names], width=0.36, label="Binder")
    ax.set_xticks(x, names, rotation=25, ha="right")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.bar(x - 0.18, [low_high_power(ensembles[n])["low_p_power"] for n in names], width=0.36, label="low-p")
    ax.bar(x + 0.18, [low_high_power(ensembles[n])["high_p_power"] for n in names], width=0.36, label="high-p")
    ax.set_xticks(x, names, rotation=25, ha="right")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / "multilevel_inverse_blocking_plots.pdf")
    plt.close(fig)


def write_report(path: Path, summary: dict[str, object]) -> None:
    lines = [
        "# Multilevel Inverse Blocking",
        "",
        "Diagnostic setup: true 32x32 configurations are blocked to 16x16 and 8x8. This is still supervised/conditional; phi_8 and phi_16 are obtained from true fine fields.",
        "",
        "## Algebraic Checks",
        "",
    ]
    alg = summary["algebraic_reconstruction"]
    for key, value in alg.items():
        lines.append(f"- {key}: `{value:.6g}`")
    lines.extend(
        [
            "",
            "## Ensemble Summary",
            "",
            "| ensemble | S mean | S std | phi2 | Binder | NN corr | susceptibility | low-p | high-p | agg abs rel err |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, ens in summary["ensembles"].items():
        lines.append(
            f"| {name} | {ens['S_mean']:.6g} | {ens['S_std']:.6g} | {ens['phi2']:.6g} | "
            f"{ens['binder']:.6g} | {ens['NN_corr']:.6g} | {ens['susceptibility']:.6g} | "
            f"{ens['low_p_power']:.6g} | {ens['high_p_power']:.6g} | {ens['aggregate_abs_rel_error']:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Density Diagnostics",
            "",
            "| proposal | logw std | ESS/N |",
            "|---|---:|---:|",
        ]
    )
    for name, diag in summary["density_diagnostics"].items():
        lines.append(f"| {name} | {diag['logw']['std']:.6g} | {diag['logw']['ess_over_n']:.6g} |")

    answers = summary["answers"]
    lines.extend(
        [
            "",
            "## Answers",
            "",
            f"- Does two-step lift improve observables compared with one-step? {answers['two_step_improves_observables']}.",
            f"- Does two-step reduce logw variance / improve ESS/N? {answers['two_step_improves_density_overlap']}.",
            f"- Does final restricted detail MCMC help? {answers['restricted_mcmc_helps']}.",
            f"- Is degradation from reconstructed phi_16 significant? {answers['reconstructed_phi16_degradation']}.",
            "",
            "Restricted detail MCMC is used here only at the final 16->32 level, where the known fine action `S32` defines the exact conditional target on each fixed-phi_16 fiber. An analogous 8->16 restricted-MCMC correction would require the exact blocked effective action `S16^blocked` induced from the 32x32 theory. That action is unknown and is not the bare phi4 action, so intermediate restricted MCMC is not a valid correction in this experiment.",
            "",
            "## Conditional Scope",
            "",
            "The current experiment uses phi_8 and phi_16 obtained by blocking true fine configurations. It tests the multilevel conditional reconstruction, not a full sampler with independently generated coarse fields.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = build_parser().parse_args()
    if args.fine_size != 32:
        raise ValueError("this diagnostic currently expects --fine-size 32")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    params = Phi4Params(kappa=args.kappa_fine, lam=args.lam)
    torch.manual_seed(args.seed)

    phi32 = load_or_generate_fine_configs(
        args.data_path,
        n_configs=args.n_configs,
        fine_size=32,
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
    rec16_true = reconstruct_from_average_block(phi8, d16_true)
    rec32_true_direct = reconstruct_from_average_block(phi16, d32_true)
    rec32_true_two_step = reconstruct_from_average_block(rec16_true, d32_true)
    algebra = {
        "phi16_from_phi8_true_detail_max_abs_error": float((rec16_true - phi16).abs().max().item()),
        "phi32_from_phi16_true_detail_max_abs_error": float((rec32_true_direct - phi32).abs().max().item()),
        "phi32_two_step_true_detail_max_abs_error": float((rec32_true_two_step - phi32).abs().max().item()),
        "block32_to_16_consistency_error": float((block_average(phi32) - phi16).abs().max().item()),
        "block16_to_8_consistency_error": float((block_average(phi16) - phi8).abs().max().item()),
    }

    ckpt16 = args.output_dir / "multilevel_flow_8_to_16.pt"
    ckpt32 = args.output_dir / "multilevel_flow_16_to_32.pt"
    flow16 = train_flow("8_to_16", phi8, d16_true, params, args, ckpt16, device)
    flow32 = train_flow("16_to_32", phi16, d32_true, params, args, ckpt32, device)

    n_eval = min(args.n_eval, len(phi32))
    phi32_eval = phi32[:n_eval].to(device)
    phi16_eval = phi16[:n_eval].to(device)
    phi8_eval = phi8[:n_eval].to(device)
    gen = torch.Generator(device=device).manual_seed(args.seed + 100)

    d32_one, _, _, logq32_one = flow32.sample_with_decomposition(phi16_eval.unsqueeze(1), generator=gen)
    phi32_one = reconstruct_from_average_block(phi16_eval, d32_one)
    action_one = phi4_action(phi32_one, params)
    logw_one = -action_one - logq32_one

    d16_two, _, _, logq16_two = flow16.sample_with_decomposition(phi8_eval.unsqueeze(1), generator=gen)
    phi16_two = reconstruct_from_average_block(phi8_eval, d16_two)
    d32_two, _, _, logq32_two = flow32.sample_with_decomposition(phi16_two.unsqueeze(1), generator=gen)
    phi32_two = reconstruct_from_average_block(phi16_two, d32_two)
    action_two = phi4_action(phi32_two, params)
    logq_two = logq16_two + logq32_two
    logw_two = -action_two - logq_two

    ensembles = {
        "true_32": phi32_eval.cpu(),
        "one_step_true_phi16": phi32_one.cpu(),
        "two_step_reconstructed_phi16": phi32_two.cpu(),
    }
    restricted_stats = {}
    if args.restricted_mcmc:
        d32_corr, restricted_stats = restricted_detail_mcmc(
            phi16_two.unsqueeze(1),
            d32_two.clone(),
            params,
            args,
            torch.Generator(device=device).manual_seed(args.seed + 200),
        )
        phi32_two_corr = reconstruct_from_average_block(phi16_two, d32_corr)
        ensembles["two_step_final_restricted_mcmc"] = phi32_two_corr.cpu()

    true_summary = ensemble_summary(ensembles["true_32"], params)
    summaries = {}
    for name, phi in ensembles.items():
        s = ensemble_summary(phi, params)
        s["relative_errors"] = rel_errors(s, true_summary)
        s["aggregate_abs_rel_error"] = aggregate_abs_rel(s, true_summary)
        summaries[name] = s

    density = {
        "one_step": {"logw": logw_stats(logw_one), "logq": tensor_stats(logq32_one)},
        "two_step": {"logw": logw_stats(logw_two), "logq": tensor_stats(logq_two)},
    }
    one_err = summaries["one_step_true_phi16"]["aggregate_abs_rel_error"]
    two_err = summaries["two_step_reconstructed_phi16"]["aggregate_abs_rel_error"]
    corr_err = summaries.get("two_step_final_restricted_mcmc", {}).get("aggregate_abs_rel_error")
    answers = {
        "two_step_improves_observables": bool(two_err < one_err),
        "two_step_improves_density_overlap": bool(
            density["two_step"]["logw"]["std"] < density["one_step"]["logw"]["std"]
            or density["two_step"]["logw"]["ess_over_n"] > density["one_step"]["logw"]["ess_over_n"]
        ),
        "restricted_mcmc_helps": bool(corr_err is not None and corr_err < two_err),
        "reconstructed_phi16_degradation": (
            f"two-step aggregate error {two_err:.6g} vs one-step {one_err:.6g}; "
            f"relative change {(two_err - one_err) / max(one_err, 1e-12):.6g}"
        ),
    }

    summary = {
        "fine_size": 32,
        "middle_size": 16,
        "coarse_size": 8,
        "kappa_fine": args.kappa_fine,
        "lambda": args.lam,
        "n_eval": n_eval,
        "checkpoints": {"8_to_16": str(ckpt16), "16_to_32": str(ckpt32)},
        "algebraic_reconstruction": algebra,
        "ensembles": summaries,
        "density_diagnostics": density,
        "restricted_detail_mcmc": restricted_stats,
        "answers": answers,
    }
    (args.output_dir / "multilevel_inverse_blocking_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_report(args.output_dir / "multilevel_inverse_blocking_report.md", summary)
    plot_outputs(args.output_dir, ensembles, params)
    print(f"wrote {args.output_dir / 'multilevel_inverse_blocking_summary.json'}")
    print(f"wrote {args.output_dir / 'multilevel_inverse_blocking_report.md'}")
    print(f"wrote {args.output_dir / 'multilevel_inverse_blocking_plots.pdf'}")
    print("one_step_agg_error", f"{one_err:.6g}")
    print("two_step_agg_error", f"{two_err:.6g}")
    if corr_err is not None:
        print("two_step_restricted_mcmc_agg_error", f"{float(corr_err):.6g}")
    print("one_step_logw_std", f"{density['one_step']['logw']['std']:.6g}", "ess/n", f"{density['one_step']['logw']['ess_over_n']:.6g}")
    print("two_step_logw_std", f"{density['two_step']['logw']['std']:.6g}", "ess/n", f"{density['two_step']['logw']['ess_over_n']:.6g}")


if __name__ == "__main__":
    main()
