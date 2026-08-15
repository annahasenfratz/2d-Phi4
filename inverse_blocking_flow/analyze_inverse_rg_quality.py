"""Diagnostic study of inverse-RG reconstruction quality.

This script does not train a model. It compares true fine fields, deterministic
blocked/prolongated fields, Gaussian-detail reconstructions, flow
reconstructions, and a short global A/R chain when checkpoints are available.
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
from inverse_blocking_flow.haar import block_average, prolong_constant, reconstruct_from_average_block
from inverse_blocking_flow.phi4 import Phi4Params, binder_cumulant, mean_phi2, nearest_neighbor_correlator, phi4_action


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fine-size", type=int, default=16)
    parser.add_argument("--lambda", dest="lam", type=float, default=1.0)
    parser.add_argument("--kappa-fine", type=float, default=0.31)
    parser.add_argument("--n-configs", type=int, default=512)
    parser.add_argument("--n-eval", type=int, default=256)
    parser.add_argument("--data-path", type=Path, default=Path("inverse_blocking_flow/outputs_fine16/fine_configs.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("inverse_blocking_flow/outputs_fine16"))
    parser.add_argument("--mle-checkpoint", type=Path, default=Path("inverse_blocking_flow/outputs_fine16/conditional_detail_flow_mle.pt"))
    parser.add_argument("--reverse-kl-checkpoint", type=Path, default=Path("inverse_blocking_flow/outputs_fine16/conditional_detail_flow_reverse_kl.pt"))
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--hidden-channels", type=int, default=48)
    parser.add_argument("--cnn-depth", type=int, default=3)
    parser.add_argument("--burn-in", type=int, default=200)
    parser.add_argument("--sample-interval", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--proposal-width", type=float, default=1.0)
    parser.add_argument("--ar-steps", type=int, default=512)
    parser.add_argument("--ar-burn-in", type=int, default=128)
    parser.add_argument("--seed", type=int, default=24680)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    return parser


def load_flow(path: Path, args: argparse.Namespace, device: torch.device) -> ConditionalDetailFlow | None:
    if not path.exists():
        return None
    flow = ConditionalDetailFlow(args.layers, args.hidden_channels, args.cnn_depth).to(device)
    state = torch.load(path, map_location=device, weights_only=False)
    flow.load_state_dict(state["model"])
    flow.eval()
    return flow


def tensor_stats(x: torch.Tensor) -> dict[str, float]:
    x = x.detach().float().cpu()
    return {
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
    }


def log_weight_stats(logw: torch.Tensor) -> dict[str, float]:
    logw = logw.detach().float().cpu()
    log_norm = torch.logsumexp(logw, dim=0)
    log_ess = 2.0 * log_norm - torch.logsumexp(2.0 * logw, dim=0)
    ess = torch.exp(log_ess)
    out = tensor_stats(logw)
    out["ess"] = float(ess.item())
    out["ess_over_n"] = float((ess / logw.numel()).item())
    return out


def correlation(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.detach().float().cpu()
    y = y.detach().float().cpu()
    xc = x - x.mean()
    yc = y - y.mean()
    denom = torch.sqrt((xc.square().mean() * yc.square().mean()).clamp_min(1e-30))
    return float((xc * yc).mean().div(denom).item())


def susceptibility(phi: torch.Tensor) -> torch.Tensor:
    volume = phi.shape[-2] * phi.shape[-1]
    return volume * phi.mean(dim=(-2, -1)).square()


def mean_phi4(phi: torch.Tensor) -> torch.Tensor:
    return phi.pow(4).mean(dim=(-2, -1))


def correlator_x(phi: torch.Tensor) -> torch.Tensor:
    centered = phi - phi.mean(dim=(-2, -1), keepdim=True)
    values = []
    for r in range(phi.shape[-1] // 2 + 1):
        values.append((centered * torch.roll(centered, shifts=-r, dims=-1)).mean(dim=(-2, -1)))
    return torch.stack(values, dim=-1).mean(dim=0)


def radial_power_spectrum(phi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    centered = phi - phi.mean(dim=(-2, -1), keepdim=True)
    fft = torch.fft.fftn(centered, dim=(-2, -1))
    power = (fft.real.square() + fft.imag.square()).mean(dim=0) / (phi.shape[-2] * phi.shape[-1])
    ly, lx = power.shape
    ky = torch.fft.fftfreq(ly, d=1.0, device=power.device) * ly
    kx = torch.fft.fftfreq(lx, d=1.0, device=power.device) * lx
    yy, xx = torch.meshgrid(ky, kx, indexing="ij")
    radius = torch.sqrt(xx.square() + yy.square())
    shell = torch.round(radius).long()
    max_shell = int(shell.max().item())
    shell_power = []
    shell_k = []
    for s in range(max_shell + 1):
        mask = shell == s
        if mask.any():
            shell_power.append(power[mask].mean())
            shell_k.append(radius[mask].mean())
    return torch.stack(shell_k).cpu(), torch.stack(shell_power).cpu()


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
    blocked = block_average(phi)
    coarse_params = Phi4Params(kappa=params.kappa, lam=params.lam)
    return {
        "n": int(phi.shape[0]),
        "action": tensor_stats(action),
        "mean_phi": tensor_stats(phi.mean(dim=(-2, -1))),
        "mean_phi2": tensor_stats(mean_phi2(phi)),
        "mean_phi4": tensor_stats(mean_phi4(phi)),
        "binder": float(binder_cumulant(phi).item()),
        "nearest_neighbor_correlator": tensor_stats(nearest_neighbor_correlator(phi)),
        "susceptibility": tensor_stats(susceptibility(phi)),
        "blocked": {
            "mean_phi": tensor_stats(blocked.mean(dim=(-2, -1))),
            "mean_phi2": tensor_stats(mean_phi2(blocked)),
            "mean_phi4": tensor_stats(mean_phi4(blocked)),
            "binder": float(binder_cumulant(blocked).item()),
            "nearest_neighbor_correlator": tensor_stats(nearest_neighbor_correlator(blocked)),
            "susceptibility": tensor_stats(susceptibility(blocked)),
            "action_like": tensor_stats(phi4_action(blocked, coarse_params)),
        },
        "fourier": low_high_power(phi),
        "correlator_x": [float(x) for x in correlator_x(phi)],
    }


@torch.no_grad()
def make_flow_ensemble(
    name: str,
    flow: ConditionalDetailFlow,
    phi_c: torch.Tensor,
    params: Phi4Params,
    generator: torch.Generator,
) -> tuple[torch.Tensor, dict[str, object]]:
    d, log_prior, logdet, logq = flow.sample_with_decomposition(phi_c, generator=generator)
    phi = reconstruct_from_average_block(phi_c[:, 0], d)
    action = phi4_action(phi, params)
    logw = -action - logq
    return phi.cpu(), {
        "name": name,
        "S_f": tensor_stats(action),
        "log_prior_eta": tensor_stats(log_prior),
        "logdetJ": tensor_stats(logdet),
        "logq": tensor_stats(logq),
        "logw": log_weight_stats(logw),
        "corr_S_logq": correlation(action, logq),
        "corr_S_logw": correlation(action, logw),
        "corr_logq_logw": correlation(logq, logw),
    }


@torch.no_grad()
def run_global_ar(
    flow: ConditionalDetailFlow,
    phi_c_pool: torch.Tensor,
    params: Phi4Params,
    args: argparse.Namespace,
    generator: torch.Generator,
) -> tuple[torch.Tensor, dict[str, object]]:
    def propose() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = torch.randint(phi_c_pool.shape[0], (1,), generator=generator, device=phi_c_pool.device)
        phi_c = phi_c_pool[idx]
        d, _, _, logq = flow.sample_with_decomposition(phi_c, generator=generator)
        phi = reconstruct_from_average_block(phi_c[:, 0], d)
        action = phi4_action(phi, params)
        return phi, action, logq, -action - logq

    current_phi, current_action, current_logq, current_logw = propose()
    kept = []
    accepts = 0
    kept_accepts = 0
    total = args.ar_burn_in + args.ar_steps
    kept_logw = []
    for step in range(total):
        new_phi, new_action, new_logq, new_logw = propose()
        log_a = new_logw - current_logw
        accepted = math.log(torch.rand((), generator=generator).item()) < float(log_a.item())
        if accepted:
            current_phi, current_action, current_logq, current_logw = new_phi, new_action, new_logq, new_logw
            accepts += 1
        if step >= args.ar_burn_in:
            kept.append(current_phi[0].cpu())
            kept_logw.append(current_logw.reshape(()).cpu())
            kept_accepts += int(accepted)
    return torch.stack(kept), {
        "acceptance_rate": accepts / float(total),
        "kept_window_acceptance_rate": kept_accepts / float(args.ar_steps),
        "chain_logw": log_weight_stats(torch.stack(kept_logw)),
    }


def plot_outputs(
    output_dir: Path,
    ensembles: dict[str, torch.Tensor],
    density: dict[str, dict[str, object]],
    params: Phi4Params,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7.0, 4.4))
    for name, phi in ensembles.items():
        values = phi4_action(phi, params).detach().cpu().numpy()
        plt.hist(values, bins=45, density=True, alpha=0.35, label=name)
    plt.xlabel("S_f")
    plt.ylabel("density")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "action_comparison.pdf")
    plt.close()

    names = list(ensembles)
    phi2 = [float(mean_phi2(ensembles[name]).mean().item()) for name in names]
    binder = [float(binder_cumulant(ensembles[name]).item()) for name in names]
    x = torch.arange(len(names)).numpy()
    plt.figure(figsize=(8.0, 4.4))
    plt.bar(x - 0.18, phi2, width=0.36, label="mean phi^2")
    plt.bar(x + 0.18, binder, width=0.36, label="Binder")
    plt.xticks(x, names, rotation=30, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "binder_phi2_summary.pdf")
    plt.close()

    plt.figure(figsize=(7.0, 4.4))
    for name, phi in ensembles.items():
        c = correlator_x(phi).detach().cpu().numpy()
        plt.plot(range(len(c)), c, marker="o", markersize=3, label=name)
    plt.xlabel("r along x")
    plt.ylabel("C(r)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "correlator_comparison.pdf")
    plt.close()

    plt.figure(figsize=(7.0, 4.4))
    for name, phi in ensembles.items():
        k, p = radial_power_spectrum(phi)
        plt.plot(k.numpy(), p.numpy(), marker="o", markersize=3, label=name)
    plt.xlabel("|p| shell")
    plt.ylabel("<|phi(p)|^2>")
    plt.yscale("log")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "power_spectrum_comparison.pdf")
    plt.close()

    plt.figure(figsize=(7.0, 4.4))
    plotted = False
    for name, fields in density.items():
        if "proposal_logw_values" not in fields:
            continue
        plt.hist(fields["proposal_logw_values"], bins=45, density=True, alpha=0.4, label=name)
        plotted = True
    plt.xlabel("-S_f - log q")
    plt.ylabel("density")
    if plotted:
        plt.legend(fontsize=8)
    else:
        plt.text(0.5, 0.5, "No flow checkpoint available", ha="center", va="center", transform=plt.gca().transAxes)
    plt.tight_layout()
    plt.savefig(output_dir / "logw_diagnostics.pdf")
    plt.close()


def write_report(path: Path, summary: dict[str, object]) -> None:
    def fmt(value: object, digits: int = 6) -> str:
        if value is None:
            return "missing"
        if isinstance(value, str):
            return value
        try:
            x = float(value)
        except (TypeError, ValueError):
            return "missing"
        if not math.isfinite(x):
            return "missing"
        return f"{x:.{digits}g}"

    def ens(name: str) -> dict[str, object] | None:
        ensembles = summary.get("ensembles", {})
        return ensembles.get(name) if isinstance(ensembles, dict) else None

    def obs(name: str, key: str) -> float | None:
        data = ens(name)
        if data is None:
            return None
        if key == "S_mean":
            return data["action"]["mean"]
        if key == "S_std":
            return data["action"]["std"]
        if key == "phi2":
            return data["mean_phi2"]["mean"]
        if key == "binder":
            return data["binder"]
        if key == "nn":
            return data["nearest_neighbor_correlator"]["mean"]
        if key == "susceptibility":
            return data["susceptibility"]["mean"]
        if key == "low_power":
            return data["fourier"]["low_momentum_power"]
        if key == "high_power":
            return data["fourier"]["high_momentum_power"]
        if key == "blocked_phi2":
            return data["blocked"]["mean_phi2"]["mean"]
        return None

    def rel(name: str, key: str) -> float | None:
        value = obs(name, key)
        true_value = obs("true", key)
        if value is None or true_value is None or abs(float(true_value)) < 1e-14:
            return None
        return (float(value) - float(true_value)) / float(true_value)

    def abs_rel(name: str, key: str) -> float:
        value = rel(name, key)
        return float("inf") if value is None else abs(value)

    def best_by(keys: list[str], candidates: list[str]) -> str:
        available = [name for name in candidates if ens(name) is not None]
        if not available:
            return "missing"
        return min(available, key=lambda name: sum(abs_rel(name, key) for key in keys))

    def exact_classification(acceptance: float | None, ess_over_n: float | None) -> str:
        if acceptance is None or ess_over_n is None:
            return "missing"
        if acceptance > 0.2 and ess_over_n > 0.05:
            return "usable"
        if acceptance > 0.05 and ess_over_n > 0.01:
            return "marginal"
        return "failed"

    def ar_diag(tag: str) -> dict[str, object]:
        ar = summary.get("global_ar", {})
        if isinstance(ar, dict) and tag in ar and isinstance(ar[tag], dict):
            return ar[tag]
        return {}

    ensemble_order = ["true", "gaussian_details", "mle_flow", "reverse_kl_flow", "ar_mle", "ar_reverse_kl"]
    observable_keys = [
        ("S mean", "S_mean"),
        ("S std", "S_std"),
        ("phi2", "phi2"),
        ("Binder", "binder"),
        ("NN corr", "nn"),
        ("susceptibility", "susceptibility"),
        ("low-p power", "low_power"),
        ("high-p power", "high_power"),
    ]
    model_candidates = [name for name in ensemble_order if name != "true"]
    best_action = best_by(["S_mean", "S_std"], model_candidates)
    best_phi2_binder = best_by(["phi2", "binder"], model_candidates)
    best_high_power = best_by(["high_power"], model_candidates)

    def aggregate_error(name: str, keys: list[str]) -> float | None:
        if ens(name) is None:
            return None
        vals = [abs_rel(name, key) for key in keys]
        vals = [v for v in vals if math.isfinite(v)]
        return sum(vals) / len(vals) if vals else None

    reverse_error = aggregate_error("reverse_kl_flow", ["S_mean", "phi2", "binder", "nn", "high_power"])
    ar_reverse_error = aggregate_error("ar_reverse_kl", ["S_mean", "phi2", "binder", "nn", "high_power"])
    mle_error = aggregate_error("mle_flow", ["S_mean", "phi2", "binder", "nn", "high_power"])
    ar_mle_error = aggregate_error("ar_mle", ["S_mean", "phi2", "binder", "nn", "high_power"])
    if ar_mle_error is None or mle_error is None:
        mle_ar_sentence = "MLE global A/R is missing for this run"
    elif ar_mle_error < mle_error:
        mle_ar_sentence = "MLE global A/R improves the finite-chain aggregate observable error"
    else:
        mle_ar_sentence = "MLE global A/R worsens or does not improve the finite-chain aggregate observable error"

    reverse_exact = ar_diag("reverse_kl")
    reverse_density = summary.get("full_density", {}).get("reverse_kl_flow", {})
    reverse_acceptance = reverse_exact.get("acceptance_rate")
    reverse_kept = reverse_exact.get("kept_window_acceptance_rate")
    reverse_logw = reverse_density.get("logw", {})
    reverse_logw_std = reverse_logw.get("std")
    reverse_ess = reverse_logw.get("ess_over_n")
    reverse_class = exact_classification(reverse_acceptance, reverse_ess)

    blocked_rel_errors = {
        name: rel(name, "blocked_phi2")
        for name in ["blocked_prolongated", "gaussian_details", "mle_flow", "reverse_kl_flow"]
        if ens(name) is not None
    }
    blocked_auto = all(abs(v) < 1e-5 for v in blocked_rel_errors.values() if v is not None)
    uv_error = rel(best_high_power, "high_power") if best_high_power != "missing" else None

    lines = [
        "# Inverse-RG Quality Diagnostics",
        "",
        f"Run metadata: fine size `{summary.get('fine_size', 'missing')}`, coarse size `{summary.get('coarse_size', 'missing')}`, `n_eval={summary.get('n_eval', 'missing')}`.",
        "",
        "## Main Conclusions",
        "",
        f"- Best action-distribution match by mean/std `S_f`: `{best_action}`.",
        f"- Best joint `phi^2`/Binder match: `{best_phi2_binder}`.",
        f"- Blocked/IR `B(phi)` observables are {'automatically matched for fixed-phi_c reconstructions' if blocked_auto else 'not automatically matched in this run'}; blocked `phi^2` relative errors are "
        + ", ".join(f"`{name}` {fmt(value)}" for name, value in blocked_rel_errors.items())
        + ".",
        f"- UV/detail behavior is best by high-momentum power for `{best_high_power}` with relative error `{fmt(uv_error)}`; see the high-p power table for remaining mismatch.",
        f"- Reverse-KL global A/R {'improves' if ar_reverse_error is not None and reverse_error is not None and ar_reverse_error < reverse_error else 'worsens or does not improve'} the finite-chain aggregate observable error: raw `{fmt(reverse_error)}`, A/R `{fmt(ar_reverse_error)}`.",
        f"- {mle_ar_sentence}: raw `{fmt(mle_error)}`, A/R `{fmt(ar_mle_error)}`.",
        f"- Global exact sampling is classified as `{reverse_class}` for reverse-KL using acceptance `{fmt(reverse_acceptance)}`, kept-window acceptance `{fmt(reverse_kept)}`, ESS/N `{fmt(reverse_ess)}`, and logw std `{fmt(reverse_logw_std)}`.",
        "",
        "## Ensemble Observable Table",
        "",
        "| ensemble | mean S_f | std S_f | mean phi^2 | Binder | NN corr | susceptibility | low-p power | high-p power |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ensemble_order:
        row = [name]
        row.extend(fmt(obs(name, key)) for _, key in observable_keys)
        lines.append("| " + " | ".join(row) + " |")

    lines.extend(
        [
            "",
            "## Relative Errors Versus True",
            "",
            "`rel_error = (observable_model - observable_true) / observable_true`.",
            "",
            "| ensemble | mean S_f | std S_f | mean phi^2 | Binder | NN corr | susceptibility | low-p power | high-p power |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name in ensemble_order:
        row = [name]
        row.extend("0" if name == "true" else fmt(rel(name, key)) for _, key in observable_keys)
        lines.append("| " + " | ".join(row) + " |")

    lines.extend(
        [
            "",
            "## Exact-Sampling Diagnostic",
            "",
            "| proposal | acceptance | kept-window acceptance | logw std | ESS/N | classification |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    exact_rows = [("mle", "mle_flow"), ("reverse_kl", "reverse_kl_flow")]
    for tag, density_name in exact_rows:
        diag = ar_diag(tag)
        density = summary.get("full_density", {}).get(density_name, {})
        logw = density.get("logw", {}) if isinstance(density, dict) else {}
        acc = diag.get("acceptance_rate")
        kept = diag.get("kept_window_acceptance_rate")
        std = logw.get("std")
        ess = logw.get("ess_over_n")
        lines.append(
            f"| {tag} | {fmt(acc)} | {fmt(kept)} | {fmt(std)} | {fmt(ess)} | {exact_classification(acc, ess)} |"
        )

    lines.extend(
        [
            "",
            "Classification rule: usable if acceptance > 0.2 and ESS/N > 0.05; marginal if acceptance > 0.05 and ESS/N > 0.01; failed otherwise.",
            "",
            "## Full-Density Decomposition",
            "",
            "| proposal | mean S_f | std S_f | mean log_prior_eta | std log_prior_eta | mean logdetJ | std logdetJ | mean logq | std logq | mean logw | std logw | ESS/N | corr(S, logq) | corr(S, logw) | corr(logq, logw) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name in ["mle_flow", "reverse_kl_flow"]:
        density = summary.get("full_density", {}).get(name, {})
        row = [
            name,
            fmt(density.get("S_f", {}).get("mean")),
            fmt(density.get("S_f", {}).get("std")),
            fmt(density.get("log_prior_eta", {}).get("mean")),
            fmt(density.get("log_prior_eta", {}).get("std")),
            fmt(density.get("logdetJ", {}).get("mean")),
            fmt(density.get("logdetJ", {}).get("std")),
            fmt(density.get("logq", {}).get("mean")),
            fmt(density.get("logq", {}).get("std")),
            fmt(density.get("logw", {}).get("mean")),
            fmt(density.get("logw", {}).get("std")),
            fmt(density.get("logw", {}).get("ess_over_n")),
            fmt(density.get("corr_S_logq")),
            fmt(density.get("corr_S_logw")),
            fmt(density.get("corr_logq_logw")),
        ]
        lines.append("| " + " | ".join(row) + " |")

    lines.extend(
        [
            "",
            "## Observable-Level vs Exact-Sampling Distinction",
            "",
            f"The best observable-level reconstruction by action mean/std is `{best_action}`, while the reverse-KL exact-sampling diagnostic is `{reverse_class}`. "
            f"For reverse-KL, action mean relative error is `{fmt(rel('reverse_kl_flow', 'S_mean'))}`, phi2 relative error is `{fmt(rel('reverse_kl_flow', 'phi2'))}`, "
            f"and high-p power relative error is `{fmt(rel('reverse_kl_flow', 'high_power'))}`. "
            f"However, exact sampling has acceptance `{fmt(reverse_acceptance)}`, kept-window acceptance `{fmt(reverse_kept)}`, ESS/N `{fmt(reverse_ess)}`, and logw std `{fmt(reverse_logw_std)}`. "
            "Thus the map can be useful as an observable-level inverse RG diagnostic if the relevant observables are close, even when it is not usable as a global exact sampler.",
            "",
            "## Next Recommended Experiment",
            "",
        ]
    )
    if reverse_class == "failed":
        lines.append("Global A/R failed by the stated thresholds, so the next experiment should be patchwise A/R at fixed `phi_c`.")
        if reverse_logw_std is not None and float(reverse_logw_std) > 10.0:
            lines.append(f"Because reverse-KL logw std is `{fmt(reverse_logw_std)}`, also debug `logq`/`logdetJ` decomposition before attempting longer training.")
    elif reverse_logw_std is not None and float(reverse_logw_std) > 10.0:
        lines.append(f"Reverse-KL logw std is `{fmt(reverse_logw_std)}`, so prioritize `logq`/`logdetJ` debugging.")
    elif reverse_error is not None and reverse_error > 0.1:
        lines.append("Exact-sampling diagnostics are not the main blocker; try stronger or longer MLE/reverse-KL training because observables are still poor.")
    else:
        lines.append("The current diagnostics are relatively sane; increase statistics and compare patchwise A/R against global A/R.")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    params = Phi4Params(kappa=args.kappa_fine, lam=args.lam)
    generator = torch.Generator(device=device).manual_seed(args.seed)

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
    phi_c_all, _, true_phi_all = dataset.tensors
    n_eval = min(args.n_eval, len(dataset))
    phi_c = phi_c_all[:n_eval].to(device)
    true_phi = true_phi_all[:n_eval].to(device)

    ensembles: dict[str, torch.Tensor] = {"true": true_phi.cpu()}
    full_density: dict[str, dict[str, object]] = {}
    global_ar: dict[str, dict[str, object]] = {}

    phi0 = prolong_constant(phi_c[:, 0])
    ensembles["blocked_prolongated"] = phi0.cpu()

    gaussian_d = torch.randn(
        (n_eval, 3, args.fine_size // 2, args.fine_size // 2),
        generator=generator,
        device=device,
    )
    gaussian_phi = reconstruct_from_average_block(phi_c[:, 0], gaussian_d)
    ensembles["gaussian_details"] = gaussian_phi.cpu()

    mle_flow = load_flow(args.mle_checkpoint, args, device)
    if mle_flow is not None:
        mle_phi, mle_density = make_flow_ensemble("mle_flow", mle_flow, phi_c, params, generator)
        ensembles["mle_flow"] = mle_phi
        # Store compact stats in JSON and raw values only for plotting.
        d, lp, ld, lq = mle_flow.sample_with_decomposition(phi_c, generator=generator)
        action = phi4_action(reconstruct_from_average_block(phi_c[:, 0], d), params)
        mle_density["proposal_logw_values"] = (-action - lq).detach().cpu().numpy().tolist()
        full_density["mle_flow"] = mle_density
        ar_mle_phi, ar_mle_diag = run_global_ar(mle_flow, phi_c, params, args, generator)
        ensembles["ar_mle"] = ar_mle_phi
        global_ar["mle"] = ar_mle_diag

    reverse_flow = load_flow(args.reverse_kl_checkpoint, args, device)
    if reverse_flow is not None:
        reverse_phi, reverse_density = make_flow_ensemble("reverse_kl_flow", reverse_flow, phi_c, params, generator)
        ensembles["reverse_kl_flow"] = reverse_phi
        d, lp, ld, lq = reverse_flow.sample_with_decomposition(phi_c, generator=generator)
        action = phi4_action(reconstruct_from_average_block(phi_c[:, 0], d), params)
        reverse_density["proposal_logw_values"] = (-action - lq).detach().cpu().numpy().tolist()
        full_density["reverse_kl_flow"] = reverse_density
        ar_phi, ar_diag = run_global_ar(reverse_flow, phi_c, params, args, generator)
        ensembles["ar_reverse_kl"] = ar_phi
        global_ar["reverse_kl"] = ar_diag

    summary = {
        "fine_size": args.fine_size,
        "coarse_size": args.fine_size // 2,
        "kappa_fine": args.kappa_fine,
        "lambda": args.lam,
        "n_eval": n_eval,
        "ensembles": {name: ensemble_summary(phi, params) for name, phi in ensembles.items()},
        "full_density": {
            name: {k: v for k, v in values.items() if k != "proposal_logw_values"}
            for name, values in full_density.items()
        },
        "global_ar": global_ar,
    }
    summary_path = args.output_dir / "inverse_rg_quality_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    plot_outputs(args.output_dir, ensembles, full_density, params)
    report_path = args.output_dir / "inverse_rg_quality_report.md"
    write_report(report_path, summary)
    print(f"wrote {summary_path}")
    print(f"wrote {args.output_dir / 'action_comparison.pdf'}")
    print(f"wrote {args.output_dir / 'binder_phi2_summary.pdf'}")
    print(f"wrote {args.output_dir / 'correlator_comparison.pdf'}")
    print(f"wrote {args.output_dir / 'power_spectrum_comparison.pdf'}")
    print(f"wrote {args.output_dir / 'logw_diagnostics.pdf'}")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
