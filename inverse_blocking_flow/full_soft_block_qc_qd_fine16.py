"""Full learned q_c * q_d soft-blocking test on fine_size=16."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/matplotlib")

from inverse_blocking_flow.data import load_or_generate_fine_configs
from inverse_blocking_flow.flow import ConditionalDetailFlow, make_conditioning
from inverse_blocking_flow.haar import soft_block, soft_kernel_term, soft_reconstruct
from inverse_blocking_flow.logw_kappa_scan_blocked_coarse import (
    aggregate_abs_rel,
    ensemble_summary,
    kappa_grid,
    low_high_power,
    stabilized_logw_stats,
)
from inverse_blocking_flow.phi4 import (
    Phi4Params,
    binder_cumulant,
    mean_phi2,
    nearest_neighbor_correlator,
    phi4_action,
    susceptibility,
)


class SpatialAffineCoupling(nn.Module):
    def __init__(self, mask: torch.Tensor, hidden_channels: int, depth: int, scale_clip: float = 3.0):
        super().__init__()
        self.register_buffer("mask", mask.view(1, 1, *mask.shape).float())
        self.scale_clip = scale_clip
        layers: list[nn.Module] = [
            nn.Conv2d(2, hidden_channels, kernel_size=3, padding=1, padding_mode="circular"),
            nn.SiLU(),
        ]
        for _ in range(depth - 2):
            layers.extend(
                [
                    nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, padding_mode="circular"),
                    nn.SiLU(),
                ]
            )
        layers.append(nn.Conv2d(hidden_channels, 2, kernel_size=3, padding=1, padding_mode="circular"))
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(layers[-1].weight)
        nn.init.zeros_(layers[-1].bias)

    def _st(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mask = self.mask.to(dtype=x.dtype, device=x.device)
        active = 1.0 - mask
        h = self.net(torch.cat((x * mask, mask.expand_as(x)), dim=1))
        shift, log_scale = h[:, :1], h[:, 1:]
        log_scale = active * self.scale_clip * torch.tanh(log_scale / self.scale_clip)
        shift = active * shift
        return shift, log_scale, mask

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shift, log_scale, mask = self._st(x)
        y = mask * x + (1.0 - mask) * (x * torch.exp(log_scale) + shift)
        return y, log_scale.sum(dim=(1, 2, 3))

    def inverse(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shift, log_scale, mask = self._st(y)
        x = mask * y + (1.0 - mask) * ((y - shift) * torch.exp(-log_scale))
        return x, -log_scale.sum(dim=(1, 2, 3))


class CoarseFieldFlow(nn.Module):
    def __init__(self, size: int = 8, n_layers: int = 8, hidden_channels: int = 48, depth: int = 3):
        super().__init__()
        yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
        masks = [((yy + xx + i) % 2 == 0).float() for i in range(2)]
        self.size = size
        self.layers = nn.ModuleList(
            [SpatialAffineCoupling(masks[i % 2], hidden_channels, depth) for i in range(n_layers)]
        )

    @staticmethod
    def standard_normal_logprob(x: torch.Tensor) -> torch.Tensor:
        return -0.5 * (x.square() + math.log(2.0 * math.pi)).sum(dim=(1, 2, 3))

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = z
        logdet = torch.zeros(z.shape[0], dtype=z.dtype, device=z.device)
        for layer in self.layers:
            x, ld = layer(x)
            logdet = logdet + ld
        return x, self.standard_normal_logprob(z) - logdet

    def inverse_logq(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = x
        logdet_inv = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
        for layer in reversed(self.layers):
            z, ld = layer.inverse(z)
            logdet_inv = logdet_inv + ld
        return z, self.standard_normal_logprob(z) + logdet_inv

    def sample(self, n: int, device: torch.device, generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        z = torch.randn((n, 1, self.size, self.size), device=device, generator=generator)
        return self.forward(z)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fine-size", type=int, default=16)
    parser.add_argument("--kappa", type=float, default=0.31)
    parser.add_argument("--lambda-f", dest="lam", type=float, default=1.0)
    parser.add_argument("--soft-alpha", type=float, default=2.0)
    parser.add_argument("--n-configs", type=int, default=512)
    parser.add_argument("--n-eval", type=int, default=512)
    parser.add_argument("--data-path", type=Path, default=Path("inverse_blocking_flow/outputs_fine16/fine_configs.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("inverse_blocking_flow/outputs_fine16"))
    parser.add_argument("--qc-epochs", type=int, default=30)
    parser.add_argument("--qd-mle-epochs", type=int, default=20)
    parser.add_argument("--qd-rkl-epochs", type=int, default=50)
    parser.add_argument("--joint-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--layers-qc", type=int, default=8)
    parser.add_argument("--layers-qd", type=int, default=6)
    parser.add_argument("--hidden-channels", type=int, default=48)
    parser.add_argument("--cnn-depth", type=int, default=3)
    parser.add_argument("--conditioning-mode", choices=("physics",), default="physics")
    parser.add_argument("--conditional-samples-per-coarse", type=int, default=16)
    parser.add_argument("--burn-in", type=int, default=200)
    parser.add_argument("--sample-interval", type=int, default=10)
    parser.add_argument("--proposal-width", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=959595)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    return parser


def field_observables(x: torch.Tensor) -> dict[str, float]:
    x = x.detach().cpu()
    out = {
        "psi2": float(mean_phi2(x).mean().item()),
        "binder": float(binder_cumulant(x).item()),
        "NN_corr": float(nearest_neighbor_correlator(x).mean().item()),
        "susceptibility": float(susceptibility(x).mean().item()),
    }
    out.update(low_high_power(x))
    return out


def rel_error_dict(obs: dict[str, float], true: dict[str, float]) -> dict[str, float]:
    out = {}
    for key, value in obs.items():
        denom = true.get(key, float("nan"))
        out[key] = float("nan") if abs(denom) < 1e-14 else (value - denom) / denom
    return out


def aggregate_fine_error(obs: dict[str, float], true: dict[str, float]) -> float:
    return aggregate_abs_rel(obs, true)


def train_qc(flow: CoarseFieldFlow, train_loader: DataLoader, args: argparse.Namespace, device: torch.device) -> list[dict[str, float]]:
    opt = torch.optim.Adam(flow.parameters(), lr=args.lr)
    history = []
    for epoch in range(1, args.qc_epochs + 1):
        losses = []
        for (psi,) in train_loader:
            psi = psi.to(device)
            opt.zero_grad(set_to_none=True)
            _z, logq = flow.inverse_logq(psi)
            loss = -logq.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), 10.0)
            opt.step()
            losses.append(float(loss.detach().cpu().item()))
        history.append({"epoch": epoch, "loss": sum(losses) / len(losses)})
        print(f"q_c epoch {epoch:04d} mle loss {history[-1]['loss']:.6g}", flush=True)
    return history


def train_qd(
    flow: ConditionalDetailFlow,
    train_loader: DataLoader,
    args: argparse.Namespace,
    params: Phi4Params,
    device: torch.device,
) -> list[dict[str, float | str]]:
    opt = torch.optim.Adam(flow.parameters(), lr=args.lr)
    history = []
    phases = [("mle", args.qd_mle_epochs), ("reverse_kl", args.qd_rkl_epochs)]
    for phase, epochs in phases:
        for epoch in range(1, epochs + 1):
            losses = []
            for psi, u in train_loader:
                psi = psi.to(device)
                u = u.to(device)
                cond = make_conditioning(psi, args.conditioning_mode)
                opt.zero_grad(set_to_none=True)
                if phase == "mle":
                    _eta, logq = flow.inverse_logq(u, cond)
                    loss = -logq.mean()
                else:
                    sampled_u, logq = flow.sample(cond)
                    phi = soft_reconstruct(psi[:, 0], sampled_u)
                    loss = (phi4_action(phi, params) + soft_kernel_term(sampled_u, args.soft_alpha) + logq).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(flow.parameters(), 10.0)
                opt.step()
                losses.append(float(loss.detach().cpu().item()))
            row = {"phase": phase, "epoch": epoch, "loss": sum(losses) / len(losses)}
            history.append(row)
            print(f"q_d epoch {epoch:04d} {phase} loss {row['loss']:.6g}", flush=True)
    return history


def joint_train(
    qc: CoarseFieldFlow,
    qd: ConditionalDetailFlow,
    args: argparse.Namespace,
    params: Phi4Params,
    device: torch.device,
) -> list[dict[str, float]]:
    opt = torch.optim.Adam(list(qc.parameters()) + list(qd.parameters()), lr=args.lr)
    history = []
    for epoch in range(1, args.joint_epochs + 1):
        losses = []
        n_batches = max(1, math.ceil(args.n_configs / args.batch_size))
        for step in range(n_batches):
            opt.zero_grad(set_to_none=True)
            gen = torch.Generator(device=device).manual_seed(args.seed + 100000 + 1000 * epoch + step)
            psi, logq_c = qc.sample(args.batch_size, device, gen)
            cond = make_conditioning(psi, args.conditioning_mode)
            u, logq_d = qd.sample(cond, generator=gen)
            phi = soft_reconstruct(psi[:, 0], u)
            loss = (phi4_action(phi, params) + soft_kernel_term(u, args.soft_alpha) + logq_c + logq_d).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(qc.parameters()) + list(qd.parameters()), 10.0)
            opt.step()
            losses.append(float(loss.detach().cpu().item()))
        row = {"epoch": epoch, "loss": sum(losses) / len(losses)}
        history.append(row)
        print(f"joint epoch {epoch:04d} reverse_kl loss {row['loss']:.6g}", flush=True)
    return history


@torch.no_grad()
def qc_diagnostics(qc: CoarseFieldFlow, psi_true: torch.Tensor, args: argparse.Namespace, device: torch.device) -> dict[str, object]:
    psi_eval = psi_true[: args.n_eval].to(device)
    _z, heldout_logq = qc.inverse_logq(psi_eval)
    gen = torch.Generator(device=device).manual_seed(args.seed + 211)
    psi_gen, logq_gen = qc.sample(args.n_eval, device, gen)
    true_obs = field_observables(psi_eval[:, 0].cpu())
    gen_obs = field_observables(psi_gen[:, 0].cpu())
    return {
        "true_psi_observables": true_obs,
        "generated_psi_observables": gen_obs,
        "generated_vs_true_rel_error": rel_error_dict(gen_obs, true_obs),
        "heldout_true_logq": tensor_stats(heldout_logq),
        "generated_logq": tensor_stats(logq_gen),
    }


@torch.no_grad()
def conditional_decomposition(
    qd: ConditionalDetailFlow,
    psi_true: torch.Tensor,
    args: argparse.Namespace,
    params: Phi4Params,
    device: torch.device,
) -> dict[str, float | dict[str, float]]:
    n = min(args.n_eval, psi_true.shape[0])
    psi = psi_true[:n].to(device)
    rows = []
    for start in range(0, n, args.batch_size):
        psi_b = psi[start : start + args.batch_size]
        repeated = psi_b.repeat_interleave(args.conditional_samples_per_coarse, dim=0)
        cond = make_conditioning(repeated, args.conditioning_mode)
        gen = torch.Generator(device=device).manual_seed(args.seed + 307 + start)
        u, logq = qd.sample(cond, generator=gen)
        phi = soft_reconstruct(repeated[:, 0], u)
        ell = -phi4_action(phi, params) - soft_kernel_term(u, args.soft_alpha) - logq
        rows.append(ell.reshape(psi_b.shape[0], args.conditional_samples_per_coarse).cpu())
    ell_ij = torch.cat(rows, dim=0).float()
    per_mean = ell_ij.mean(dim=1)
    per_std = ell_ij.std(dim=1, unbiased=False)
    total_var = ell_ij.flatten().var(unbiased=False)
    between_var = per_mean.var(unbiased=False)
    within_var = per_std.square().mean()
    ell_cond = ell_ij - per_mean[:, None]
    return {
        "total_std": float(total_var.sqrt().item()),
        "between_coarse_std": float(between_var.sqrt().item()),
        "within_coarse_std": float(within_var.sqrt().item()),
        "between_fraction": float((between_var / total_var.clamp_min(1e-12)).item()),
        "within_fraction": float((within_var / total_var.clamp_min(1e-12)).item()),
        "variance_closure_error": float((total_var - between_var - within_var).item()),
        "centered_conditional_std": float(ell_cond.flatten().std(unbiased=False).item()),
        "centered_conditional_ess_over_n": log_ess_over_n(ell_cond.flatten()),
    }


@torch.no_grad()
def full_proposal_diagnostics(
    name: str,
    qc: CoarseFieldFlow,
    qd: ConditionalDetailFlow,
    true_phi: torch.Tensor,
    args: argparse.Namespace,
    params: Phi4Params,
    device: torch.device,
) -> dict[str, object]:
    gen = torch.Generator(device=device).manual_seed(args.seed + (401 if name == "separate" else 503))
    psi, logq_c = qc.sample(args.n_eval, device, gen)
    cond = make_conditioning(psi, args.conditioning_mode)
    u, logq_d = qd.sample(cond, generator=gen)
    phi = soft_reconstruct(psi[:, 0], u)
    kernel = soft_kernel_term(u, args.soft_alpha)
    logq_full = logq_c + logq_d
    logw = -phi4_action(phi, params) - kernel - logq_full
    obs = ensemble_summary(phi.cpu(), params)
    true_summary = ensemble_summary(true_phi[: args.n_eval].cpu(), params)
    scan_rows = []
    for kappa in kappa_grid(0.20, 0.38, 0.01):
        p = Phi4Params(kappa=kappa, lam=args.lam)
        obs_k = ensemble_summary(phi.cpu(), p)
        logw_k = -phi4_action(phi, p) - kernel.cpu() - logq_full.cpu()
        scan_rows.append(
            {
                "kappa_f": kappa,
                "logw": stabilized_logw_stats(logw_k),
                "observables": obs_k,
                "aggregate_abs_rel_error_vs_true": aggregate_abs_rel(obs_k, true_summary),
            }
        )
    return {
        "name": name,
        "observables": obs,
        "true_observables": true_summary,
        "aggregate_observable_error": aggregate_fine_error(obs, true_summary),
        "logq_c": tensor_stats(logq_c),
        "logq_d": tensor_stats(logq_d),
        "logq_full": tensor_stats(logq_full),
        "kernel_term": tensor_stats(kernel),
        "logw": stabilized_logw_stats(logw.cpu()),
        "kappa_scan": {
            "scan": scan_rows,
            "best_by_logw_width": min(scan_rows, key=lambda row: row["logw"]["std_logw_centered"]),
            "best_by_observable_error": min(scan_rows, key=lambda row: row["aggregate_abs_rel_error_vs_true"]),
        },
    }


def tensor_stats(x: torch.Tensor) -> dict[str, float]:
    x = x.detach().float().cpu()
    return {
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
    }


def log_ess_over_n(logw: torch.Tensor) -> float:
    centered = logw.detach().float().flatten() - logw.detach().float().flatten().mean()
    log_norm = torch.logsumexp(centered, dim=0)
    log_ess = 2.0 * log_norm - torch.logsumexp(2.0 * centered, dim=0)
    return float((torch.exp(log_ess) / centered.numel()).item())


def write_report(path: Path, summary: dict[str, object]) -> None:
    sep = summary["full_proposal"]["separate"]
    joint = summary["full_proposal"].get("joint")
    cond = summary["conditional_decomposition"]
    lines = [
        "# Full Soft Blocking q_c q_d Fine16",
        "",
        "Soft Haar blocking with `alpha=2`, `fine_size=16`, `coarse_size=8`, and physics conditioning for `q_d`.",
        "",
        "## Main Answers",
        "",
        (
            f"1. Adding `q_c` reduces the total full-proposal width to `{sep['logw']['std_logw_centered']:.6g}`. "
            f"The fixed-true-psi conditional width is `{cond['within_coarse_std']:.6g}`."
        ),
        (
            f"2. The full width is {'close to' if sep['logw']['std_logw_centered'] <= 1.5 * cond['within_coarse_std'] else 'not close to'} "
            "the fixed-coarse conditional width under the separately trained models."
        ),
        (
            f"3. Global A/R proxy is `{sep['logw']['independence_acceptance_proxy']:.6g}` with ESS/N "
            f"`{sep['logw']['ess_over_n']:.6g}` for separate training."
        ),
    ]
    if joint is None:
        lines.append("4. Joint fine-tuning was not run.")
    else:
        lines.append(
            f"4. Joint fine-tuning changed logw std from `{sep['logw']['std_logw_centered']:.6g}` to "
            f"`{joint['logw']['std_logw_centered']:.6g}` and ESS/N from `{sep['logw']['ess_over_n']:.6g}` "
            f"to `{joint['logw']['ess_over_n']:.6g}`."
        )
    lines.extend(
        [
            "",
            "## q_c Diagnostics",
            "",
            "| observable | true psi | generated psi | rel error |",
            "|---|---:|---:|---:|",
        ]
    )
    qc = summary["qc_diagnostics"]
    for key, true_value in qc["true_psi_observables"].items():
        lines.append(
            f"| {key} | {true_value:.6g} | {qc['generated_psi_observables'][key]:.6g} | "
            f"{qc['generated_vs_true_rel_error'][key]:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Conditional q_d",
            "",
            "| total std | between std | within std | between frac | within frac | centered cond ESS/N |",
            "|---:|---:|---:|---:|---:|---:|",
            (
                f"| {cond['total_std']:.6g} | {cond['between_coarse_std']:.6g} | {cond['within_coarse_std']:.6g} | "
                f"{cond['between_fraction']:.6g} | {cond['within_fraction']:.6g} | "
                f"{cond['centered_conditional_ess_over_n']:.6g} |"
            ),
            "",
            "## Full Proposal",
            "",
            "| model | logw std | ESS/N | A/R proxy | kappa_min | obs kappa_min | agg obs err | S mean | S std | phi2 | Binder | NN corr | susceptibility | low-p | high-p |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in [sep] + ([joint] if joint is not None else []):
        obs = row["observables"]
        lines.append(
            f"| {row['name']} | {row['logw']['std_logw_centered']:.6g} | {row['logw']['ess_over_n']:.6g} | "
            f"{row['logw']['independence_acceptance_proxy']:.6g} | "
            f"{row['kappa_scan']['best_by_logw_width']['kappa_f']:.6g} | "
            f"{row['kappa_scan']['best_by_observable_error']['kappa_f']:.6g} | "
            f"{row['aggregate_observable_error']:.6g} | {obs['S_mean']:.6g} | {obs['S_std']:.6g} | "
            f"{obs['phi2']:.6g} | {obs['binder']:.6g} | {obs['NN_corr']:.6g} | "
            f"{obs['susceptibility']:.6g} | {obs['low_p_power']:.6g} | {obs['high_p_power']:.6g} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_outputs(path: Path, summary: dict[str, object]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    full = [summary["full_proposal"]["separate"]]
    if summary["full_proposal"].get("joint") is not None:
        full.append(summary["full_proposal"]["joint"])
    names = [row["name"] for row in full]
    with PdfPages(path) as pdf:
        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        axes[0, 0].bar(names, [row["logw"]["std_logw_centered"] for row in full])
        axes[0, 0].axhline(summary["conditional_decomposition"]["within_coarse_std"], color="k", ls="--", label="fixed-psi within")
        axes[0, 0].set_ylabel("logw std")
        axes[0, 0].legend(fontsize=8)
        axes[0, 1].bar(names, [row["logw"]["ess_over_n"] for row in full])
        axes[0, 1].set_ylabel("ESS/N")
        axes[1, 0].bar(names, [row["logw"]["independence_acceptance_proxy"] for row in full])
        axes[1, 0].set_ylabel("A/R proxy")
        axes[1, 1].bar(names, [row["aggregate_observable_error"] for row in full])
        axes[1, 1].set_ylabel("aggregate obs error")
        for ax in axes.ravel():
            ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        qc = summary["qc_diagnostics"]
        keys = list(qc["true_psi_observables"].keys())
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(keys, [qc["generated_vs_true_rel_error"][key] for key in keys])
        ax.tick_params(axis="x", rotation=30)
        ax.set_ylabel("q_c generated psi relative error")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    if args.fine_size != 16:
        raise ValueError("this script is intended for fine_size=16 first")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    params = Phi4Params(kappa=args.kappa, lam=args.lam)
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
    ).float()
    psi, u = soft_block(phi_f, args.soft_alpha, torch.Generator().manual_seed(args.seed + 991))
    psi = psi.unsqueeze(1).float()
    u = u.float()
    qc_dataset = TensorDataset(psi)
    qd_dataset = TensorDataset(psi, u)
    train_len = max(1, int(0.9 * len(qc_dataset)))
    val_len = len(qc_dataset) - train_len
    qc_train, _qc_val = random_split(qc_dataset, [train_len, val_len], generator=torch.Generator().manual_seed(args.seed))
    qd_train, _qd_val = random_split(qd_dataset, [train_len, val_len], generator=torch.Generator().manual_seed(args.seed))
    qc_loader = DataLoader(qc_train, batch_size=args.batch_size, shuffle=True)
    qd_loader = DataLoader(qd_train, batch_size=args.batch_size, shuffle=True)

    qc = CoarseFieldFlow(args.fine_size // 2, args.layers_qc, args.hidden_channels, args.cnn_depth).to(device)
    qd = ConditionalDetailFlow(args.layers_qd, args.hidden_channels, args.cnn_depth, 6, 4).to(device)
    qc_history = train_qc(qc, qc_loader, args, device)
    qd_history = train_qd(qd, qd_loader, args, params, device)
    qc_diag = qc_diagnostics(qc, psi, args, device)
    cond_diag = conditional_decomposition(qd, psi, args, params, device)
    separate_full = full_proposal_diagnostics("separate", qc, qd, phi_f, args, params, device)

    joint_history = []
    joint_full = None
    if args.joint_epochs > 0:
        joint_history = joint_train(qc, qd, args, params, device)
        joint_full = full_proposal_diagnostics("joint", qc, qd, phi_f, args, params, device)

    torch.save(
        {
            "qc_model": qc.state_dict(),
            "qd_model": qd.state_dict(),
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "qc_history": qc_history,
            "qd_history": qd_history,
            "joint_history": joint_history,
        },
        args.output_dir / "full_soft_block_qc_qd_models.pt",
    )
    setup = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    setup["coarse_size"] = args.fine_size // 2
    summary = {
        "setup": setup,
        "qc_history": qc_history,
        "qd_history": qd_history,
        "joint_history": joint_history,
        "qc_diagnostics": qc_diag,
        "conditional_decomposition": cond_diag,
        "full_proposal": {
            "separate": separate_full,
            "joint": joint_full,
        },
        "comparisons": {
            "conditional_only_soft_haar_alpha_2_fine16": {
                "note": "fixed-psi conditional width is reported in conditional_decomposition; full q_c is not used there"
            },
            "direct_full_field_flow_fine16": {
                "available": False,
                "note": "no matching direct full-field flow checkpoint was found in inverse_blocking_flow/outputs_fine16",
            },
            "ordinary_equilibrium_fine_ensemble": ensemble_summary(phi_f[: args.n_eval].cpu(), params),
        },
    }
    summary_path = args.output_dir / "full_soft_block_qc_qd_summary.json"
    report_path = args.output_dir / "full_soft_block_qc_qd_report.md"
    plots_path = args.output_dir / "full_soft_block_qc_qd_plots.pdf"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_report(report_path, summary)
    plot_outputs(plots_path, summary)
    print(f"wrote {summary_path}")
    print(f"wrote {report_path}")
    print(f"wrote {plots_path}")


if __name__ == "__main__":
    main()
