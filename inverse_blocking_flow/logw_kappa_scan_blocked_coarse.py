"""Scan fine kappa using log weights with empirically blocked coarse fields.

This diagnostic deliberately does not use an approximate coarse action. The
coarse fields are blocked from true fine configurations, so the coarse marginal
is represented empirically and only the conditional proposal quality is tested.
"""

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
from inverse_blocking_flow.flow import ConditionalDetailFlow, make_conditioning, n_conditioning_channels
from inverse_blocking_flow.haar import block_average, reconstruct_from_average_block, reconstruct_from_weighted_block, weighted_block
from inverse_blocking_flow.phi4 import (
    Phi4Params,
    binder_cumulant,
    mean_phi2,
    nearest_neighbor_correlator,
    phi4_action,
    susceptibility,
)


OBS_KEYS = ["S_mean", "S_std", "phi2", "binder", "NN_corr", "susceptibility", "low_p_power", "high_p_power"]
AGG_KEYS = ["S_mean", "S_std", "phi2", "NN_corr", "high_p_power"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fine-size", type=int, default=32)
    parser.add_argument("--kappa-true", type=float, default=0.31)
    parser.add_argument("--lambda-f", dest="lam", type=float, default=1.0)
    parser.add_argument("--kappa-min", type=float, default=0.20)
    parser.add_argument("--kappa-max", type=float, default=0.38)
    parser.add_argument("--kappa-step", type=float, default=0.01)
    parser.add_argument("--fine-window", type=float, default=0.025)
    parser.add_argument("--fine-step", type=float, default=0.0025)
    parser.add_argument("--no-fine-scan", action="store_true")
    parser.add_argument("--n-configs", type=int, default=512)
    parser.add_argument("--n-eval", type=int, default=512)
    parser.add_argument("--data-path", type=Path, default=Path("inverse_blocking_flow/outputs/fine_configs.pt"))
    parser.add_argument("--checkpoint", type=Path, default=Path("inverse_blocking_flow/outputs/conditional_detail_flow_reverse_kl.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("inverse_blocking_flow/outputs"))
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--hidden-channels", type=int, default=48)
    parser.add_argument("--cnn-depth", type=int, default=3)
    parser.add_argument("--conditioning-mode", choices=("basic", "physics"), default=None)
    parser.add_argument("--burn-in", type=int, default=200)
    parser.add_argument("--sample-interval", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--proposal-width", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=313131)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    return parser


def kappa_grid(start: float, stop: float, step: float) -> list[float]:
    values = []
    n = int(round((stop - start) / step))
    for i in range(n + 1):
        value = start + i * step
        if value <= stop + 1e-12:
            values.append(round(value, 10))
    return values


def tensor_stats(x: torch.Tensor) -> dict[str, float]:
    x = x.detach().float().cpu()
    return {
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
    }


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
    ks = []
    ps = []
    for s in range(int(shell.max().item()) + 1):
        mask = shell == s
        if mask.any():
            ks.append(torch.sqrt(xx.square() + yy.square())[mask].mean())
            ps.append(power[mask].mean())
    return torch.stack(ks).cpu(), torch.stack(ps).cpu()


def ensemble_summary(phi: torch.Tensor, params: Phi4Params) -> dict[str, float]:
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


def rel_errors(summary: dict[str, float], true_summary: dict[str, float]) -> dict[str, float]:
    out = {}
    for key in OBS_KEYS:
        denom = true_summary[key]
        out[key] = float("nan") if abs(denom) < 1e-14 else (summary[key] - denom) / denom
    return out


def aggregate_abs_rel(summary: dict[str, float], true_summary: dict[str, float]) -> float:
    rel = rel_errors(summary, true_summary)
    return float(sum(abs(rel[key]) for key in AGG_KEYS) / len(AGG_KEYS))


def stabilized_logw_stats(logw_raw: torch.Tensor) -> dict[str, float]:
    logw_raw = logw_raw.detach().float().cpu()
    centered = logw_raw - logw_raw.mean()
    log_norm = torch.logsumexp(centered, dim=0)
    log_ess = 2.0 * log_norm - torch.logsumexp(2.0 * centered, dim=0)
    ess = torch.exp(log_ess)
    # Independence-MH proxy with the current state drawn from the
    # self-normalized target represented by the proposal cloud and the new
    # proposal drawn uniformly from the same cloud.
    diffs = centered.unsqueeze(0) - centered.unsqueeze(1)  # rows=old, cols=new: w_new - w_old
    accept = torch.minimum(torch.ones_like(diffs), torch.exp(diffs.clamp(max=80.0)))
    target_weights = torch.softmax(centered, dim=0)
    acceptance_proxy = (accept * target_weights.view(-1, 1)).sum(dim=0).mean()
    proposal_pair_acceptance = accept.mean()
    return {
        "mean_logw_raw": float(logw_raw.mean().item()),
        "std_logw_centered": float(centered.std(unbiased=False).item()),
        "min_logw_centered": float(centered.min().item()),
        "max_logw_centered": float(centered.max().item()),
        "ess": float(ess.item()),
        "ess_over_n": float((ess / centered.numel()).item()),
        "independence_acceptance_proxy": float(acceptance_proxy.item()),
        "proposal_pair_acceptance_proxy": float(proposal_pair_acceptance.item()),
    }


def checkpoint_conditioning_metadata(state: dict[str, object], requested_mode: str | None) -> tuple[str, int]:
    args_meta = state.get("args", {}) if isinstance(state, dict) else {}
    mode = requested_mode or state.get("conditioning_mode") or args_meta.get("conditioning_mode") or "basic"
    n_cond = int(state.get("n_conditioning_channels") or n_conditioning_channels(str(mode)))
    return str(mode), n_cond


def load_flow(args: argparse.Namespace, device: torch.device) -> tuple[ConditionalDetailFlow, str, int, dict[str, object]]:
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"missing checkpoint: {args.checkpoint}")
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    args_meta = state.get("args", {}) if isinstance(state, dict) else {}
    mode, n_cond = checkpoint_conditioning_metadata(state, args.conditioning_mode)
    layers = int(args_meta.get("layers", args.layers))
    hidden = int(args_meta.get("hidden_channels", args.hidden_channels))
    depth = int(args_meta.get("cnn_depth", args.cnn_depth))
    flow = ConditionalDetailFlow(layers, hidden, depth, n_cond).to(device)
    flow.load_state_dict(state["model"])
    flow.eval()
    return flow, mode, n_cond, state


def checkpoint_blocking_metadata(state: dict[str, object]) -> dict[str, object]:
    args_meta = state.get("args", {}) if isinstance(state, dict) else {}
    mode = str(state.get("blocking_mode") or args_meta.get("blocking_mode") or "hard")
    return {
        "blocking_mode": mode,
        "weighted_a": float(state.get("weighted_a", args_meta.get("weighted_a", 0.18))),
        "weighted_b": float(state.get("weighted_b", args_meta.get("weighted_b", 0.04))),
        "eta": float(state.get("eta", args_meta.get("eta", 0.25))),
        "use_eta_scaling": bool(state.get("use_eta_scaling", args_meta.get("use_eta_scaling", True))),
    }


def plot_outputs(
    output_dir: Path,
    rows: list[dict[str, object]],
    phi_true: torch.Tensor,
    phi_rec: torch.Tensor,
    kappa_true: float,
    kappa_best_width: float,
    params_true: Phi4Params,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kappa = [float(row["kappa_f"]) for row in rows]
    logw_std = [float(row["logw"]["std_logw_centered"]) for row in rows]
    ess = [float(row["logw"]["ess_over_n"]) for row in rows]
    accept = [float(row["logw"]["independence_acceptance_proxy"]) for row in rows]
    agg = [float(row["aggregate_abs_rel_error_vs_true_kappa_true"]) for row in rows]

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.8))
    ax = axes[0, 0]
    ax.plot(kappa, logw_std, marker="o")
    ax.axvline(kappa_true, color="k", ls="--", lw=1.0, label="kappa true")
    ax.axvline(kappa_best_width, color="tab:red", ls=":", lw=1.3, label="min width")
    ax.set_xlabel("kappa_f")
    ax.set_ylabel("std centered logw")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(kappa, ess, marker="o", label="ESS/N")
    ax.plot(kappa, accept, marker="s", label="A/R proxy")
    ax.set_xlabel("kappa_f")
    ax.set_ylabel("fraction")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.plot(kappa, agg, marker="o")
    ax.axvline(kappa_true, color="k", ls="--", lw=1.0)
    ax.set_xlabel("kappa_f")
    ax.set_ylabel("aggregate observable rel error")

    ax = axes[1, 1]
    ax.hist(phi4_action(phi_true, params_true).detach().cpu().numpy(), bins=40, density=True, alpha=0.45, label="true")
    ax.hist(phi4_action(phi_rec, params_true).detach().cpu().numpy(), bins=40, density=True, alpha=0.45, label="flow rec")
    ax.set_xlabel("S_f at kappa_true")
    ax.set_ylabel("density")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "logw_kappa_scan_blocked_plots.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for name, phi in {"true": phi_true, "flow_rec": phi_rec}.items():
        k, p = radial_power_spectrum(phi.cpu())
        ax.plot(k.numpy(), p.numpy(), marker="o", ms=3, label=name)
    ax.set_yscale("log")
    ax.set_xlabel("|p| shell")
    ax.set_ylabel("<|phi(p)|^2>")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "logw_kappa_scan_blocked_power_spectrum.pdf")
    plt.close(fig)


def write_report(path: Path, summary: dict[str, object]) -> None:
    rows = summary["scan"]
    answers = summary["answers"]
    lines = [
        "# Blocked-Coarse Logw Kappa Scan",
        "",
        "This scan uses the supervised blocked-coarse setup only. Coarse fields are `phi_c = B(phi_f_true)` from true fine configurations generated at `kappa_true=0.31`, `lambda=1.0`. No approximate coarse action `S_c` is used.",
        "",
        "Details are sampled from the reverse-KL conditional flow, reconstructed as `phi_rec = R(phi_c, d)`, and rescored under candidate fine actions.",
        "",
        "## Answers",
        "",
        f"1. Is std(logw) minimized near kappa_true? {answers['is_min_near_kappa_true']}",
        f"2. Preferred kappa_f by logw width: `{answers['preferred_kappa_by_logw_width']:.6g}`",
        f"3. Preferred kappa_f by observable agreement: `{answers['preferred_kappa_by_observables']:.6g}`",
        f"4. Is global A/R viable at the logw-width minimum? {answers['global_ar_viability']}",
        f"5. Does this indicate bias in q(d|phi_c)? {answers['conditional_flow_bias']}",
        "",
        "## True Fine Reference",
        "",
        "| S mean | S std | phi2 | Binder | NN corr | susceptibility | low-p | high-p |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    true = summary["true_observables_kappa_true"]
    lines.append(
        f"| {true['S_mean']:.6g} | {true['S_std']:.6g} | {true['phi2']:.6g} | {true['binder']:.6g} | "
        f"{true['NN_corr']:.6g} | {true['susceptibility']:.6g} | {true['low_p_power']:.6g} | {true['high_p_power']:.6g} |"
    )
    lines.extend(
        [
            "",
            "## Kappa Scan",
            "",
            "| kappa_f | mean logw raw | std centered logw | min centered | max centered | ESS/N | A/R proxy | S mean | S std | phi2 | Binder | NN corr | susceptibility | low-p | high-p | agg rel err |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        logw = row["logw"]
        obs = row["observables"]
        lines.append(
            f"| {row['kappa_f']:.6g} | {logw['mean_logw_raw']:.6g} | {logw['std_logw_centered']:.6g} | "
            f"{logw['min_logw_centered']:.6g} | {logw['max_logw_centered']:.6g} | {logw['ess_over_n']:.6g} | "
            f"{logw['independence_acceptance_proxy']:.6g} | {obs['S_mean']:.6g} | {obs['S_std']:.6g} | "
            f"{obs['phi2']:.6g} | {obs['binder']:.6g} | {obs['NN_corr']:.6g} | {obs['susceptibility']:.6g} | "
            f"{obs['low_p_power']:.6g} | {obs['high_p_power']:.6g} | {row['aggregate_abs_rel_error_vs_true_kappa_true']:.6g} |"
        )
    lines.extend(
        [
            "",
            "Classification rule used here: global A/R is treated as viable only if `ESS/N > 0.05` and the self-normalized independence acceptance proxy is greater than `0.2`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


@torch.no_grad()
def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.fine_size != 32:
        raise ValueError("this diagnostic is configured for --fine-size 32")
    device = torch.device(args.device)
    params_true = Phi4Params(kappa=args.kappa_true, lam=args.lam)
    phi_true = load_or_generate_fine_configs(
        args.data_path,
        n_configs=args.n_configs,
        fine_size=args.fine_size,
        params=params_true,
        burn_in=args.burn_in,
        interval=args.sample_interval,
        batch_size=args.batch_size,
        proposal_width=args.proposal_width,
        seed=args.seed,
        device=args.device,
    ).float()
    n_eval = min(args.n_eval, len(phi_true))
    phi_true = phi_true[:n_eval].to(device)
    flow, conditioning_mode, n_cond, state = load_flow(args, device)
    blocking = checkpoint_blocking_metadata(state)
    if blocking["blocking_mode"] == "weighted":
        phi_c = weighted_block(
            phi_true,
            blocking["weighted_a"],
            blocking["weighted_b"],
            eta=blocking["eta"],
            use_eta_scaling=blocking["use_eta_scaling"],
        )
    else:
        phi_c = block_average(phi_true)

    generator = torch.Generator(device=device).manual_seed(args.seed + 17)
    cond = make_conditioning(phi_c, conditioning_mode)
    d, logq = flow.sample(cond, generator=generator)
    if blocking["blocking_mode"] == "weighted":
        phi_rec = reconstruct_from_weighted_block(
            phi_c,
            d,
            blocking["weighted_a"],
            blocking["weighted_b"],
            eta=blocking["eta"],
            use_eta_scaling=blocking["use_eta_scaling"],
        )
    else:
        phi_rec = reconstruct_from_average_block(phi_c, d)
    true_summary = ensemble_summary(phi_true.cpu(), params_true)

    coarse = kappa_grid(args.kappa_min, args.kappa_max, args.kappa_step)
    initial_rows = []
    for kappa in coarse:
        params = Phi4Params(kappa=kappa, lam=args.lam)
        action = phi4_action(phi_rec, params)
        logw_raw = -action - logq
        obs = ensemble_summary(phi_rec.cpu(), params)
        initial_rows.append(
            {
                "kappa_f": kappa,
                "logw": stabilized_logw_stats(logw_raw),
                "observables": obs,
                "relative_errors_vs_true_kappa_true": rel_errors(obs, true_summary),
                "aggregate_abs_rel_error_vs_true_kappa_true": aggregate_abs_rel(obs, true_summary),
            }
        )

    best_initial = min(initial_rows, key=lambda row: row["logw"]["std_logw_centered"])
    rows = initial_rows
    if not args.no_fine_scan:
        start = max(args.kappa_min, best_initial["kappa_f"] - args.fine_window)
        stop = min(args.kappa_max, best_initial["kappa_f"] + args.fine_window)
        seen = {row["kappa_f"] for row in rows}
        for kappa in kappa_grid(start, stop, args.fine_step):
            if kappa in seen:
                continue
            params = Phi4Params(kappa=kappa, lam=args.lam)
            action = phi4_action(phi_rec, params)
            logw_raw = -action - logq
            obs = ensemble_summary(phi_rec.cpu(), params)
            rows.append(
                {
                    "kappa_f": kappa,
                    "logw": stabilized_logw_stats(logw_raw),
                    "observables": obs,
                    "relative_errors_vs_true_kappa_true": rel_errors(obs, true_summary),
                    "aggregate_abs_rel_error_vs_true_kappa_true": aggregate_abs_rel(obs, true_summary),
                }
            )
        rows = sorted(rows, key=lambda row: row["kappa_f"])

    best_width = min(rows, key=lambda row: row["logw"]["std_logw_centered"])
    best_obs = min(rows, key=lambda row: row["aggregate_abs_rel_error_vs_true_kappa_true"])
    near_true = abs(best_width["kappa_f"] - args.kappa_true) <= max(args.kappa_step, args.fine_step) + 1e-12
    viable = (
        best_width["logw"]["ess_over_n"] > 0.05
        and best_width["logw"]["independence_acceptance_proxy"] > 0.2
    )
    shifted = abs(best_width["kappa_f"] - args.kappa_true)
    bias_text = (
        "Yes. The logw-width minimum is shifted from kappa_true and the minimum width remains large, consistent with conditional proposal bias."
        if shifted > 0.015 or best_width["logw"]["std_logw_centered"] > 5.0
        else "No strong shift is visible in this scan, though density overlap should still be judged by ESS and acceptance proxy."
    )
    summary = {
        "setup": {
            "fine_size": args.fine_size,
            "coarse_size": args.fine_size // 2,
            "kappa_true": args.kappa_true,
            "lambda_f": args.lam,
            "n_eval": n_eval,
            "checkpoint": str(args.checkpoint),
            "conditioning_mode": conditioning_mode,
            "n_conditioning_channels": n_cond,
            **blocking,
            "uses_approximate_coarse_action": False,
            "coarse_marginal": "empirical blocked phi_c = B(phi_f_true)",
        },
        "true_observables_kappa_true": true_summary,
        "proposal_observables_at_kappa_true": ensemble_summary(phi_rec.cpu(), params_true),
        "proposal_logq": tensor_stats(logq),
        "scan": rows,
        "best_by_logw_width": best_width,
        "best_by_observable_error": best_obs,
        "answers": {
            "is_min_near_kappa_true": (
                f"{near_true}; min at {best_width['kappa_f']:.6g}, true is {args.kappa_true:.6g}."
            ),
            "preferred_kappa_by_logw_width": float(best_width["kappa_f"]),
            "preferred_kappa_by_observables": float(best_obs["kappa_f"]),
            "global_ar_viability": (
                f"{'viable' if viable else 'not viable'}; ESS/N={best_width['logw']['ess_over_n']:.6g}, "
                f"A/R proxy={best_width['logw']['independence_acceptance_proxy']:.6g}, "
                f"logw std={best_width['logw']['std_logw_centered']:.6g}."
            ),
            "conditional_flow_bias": bias_text,
        },
    }
    (args.output_dir / "logw_kappa_scan_blocked_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_report(args.output_dir / "logw_kappa_scan_blocked_report.md", summary)
    plot_outputs(
        args.output_dir,
        rows,
        phi_true.cpu(),
        phi_rec.cpu(),
        args.kappa_true,
        float(best_width["kappa_f"]),
        params_true,
    )
    print(f"wrote {args.output_dir / 'logw_kappa_scan_blocked_summary.json'}")
    print(f"wrote {args.output_dir / 'logw_kappa_scan_blocked_report.md'}")
    print(f"wrote {args.output_dir / 'logw_kappa_scan_blocked_plots.pdf'}")
    print("best_logw_width_kappa", f"{best_width['kappa_f']:.6g}", "std", f"{best_width['logw']['std_logw_centered']:.6g}")
    print("best_observable_kappa", f"{best_obs['kappa_f']:.6g}", "agg_err", f"{best_obs['aggregate_abs_rel_error_vs_true_kappa_true']:.6g}")


if __name__ == "__main__":
    main()
