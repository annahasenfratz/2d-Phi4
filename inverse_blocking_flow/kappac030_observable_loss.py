"""Observable-loss fine tuning for kappa_c=0.30 independent-coarse upscaling."""

from __future__ import annotations

import argparse
import copy
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

from inverse_blocking_flow.correlation_length_bootstrap import bootstrap as bootstrap_corr
from inverse_blocking_flow.data import load_or_generate_fine_configs
from inverse_blocking_flow.flow import ConditionalDetailFlow, make_conditioning
from inverse_blocking_flow.haar import soft_kernel_term, soft_reconstruct
from inverse_blocking_flow.kappac030_trainable_kappaf_upscale import (
    aggregate_for_keys,
    bounded_kappa,
    load_or_generate_coarse,
    raw_from_kappa,
    stabilized_logw_stats,
    tensor_stats,
)
from inverse_blocking_flow.logw_kappa_scan_blocked_coarse import ensemble_summary
from inverse_blocking_flow.phi4 import Phi4Params, checkerboard_metropolis_sweep, phi4_action


REF_KAPPAS = [0.320, 0.325, 0.330]
LOCAL_KEYS = ["S_density", "phi2", "phi4", "NN", "Q2"]
IR_KEYS = ["chi", "binder", "P10", "P01", "P11"]
OBS_AGG_KEYS = ["S_mean", "S_std", "phi2", "binder", "NN_corr", "susceptibility", "low_p_power", "high_p_power"]


@dataclass(frozen=True)
class Variant:
    name: str
    epochs: int
    lambda_local: float
    lambda_ir: float
    anneal: bool = False
    freeze_kappa: bool = False
    fixed_kappa: float | None = None


VARIANTS = [
    Variant("A_baseline_B", 0, 0.0, 0.0),
    Variant("B_local_only", 30, 0.1, 0.0),
    Variant("C_local_very_mild_IR", 30, 0.1, 0.02),
    Variant("D_local_mild_IR", 30, 0.1, 0.05),
    Variant("E_annealed", 50, 0.1, 0.05, anneal=True),
    Variant("F_freeze_kappa_0p320", 30, 0.1, 0.05, freeze_kappa=True, fixed_kappa=0.320),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fine-size", type=int, default=16)
    parser.add_argument("--coarse-size", type=int, default=8)
    parser.add_argument("--kappa-c", type=float, default=0.30)
    parser.add_argument("--kappa-f-initial", type=float, default=0.32)
    parser.add_argument("--kappa-f-min", type=float, default=0.30)
    parser.add_argument("--kappa-f-max", type=float, default=0.335)
    parser.add_argument("--lambda-f", dest="lam", type=float, default=1.0)
    parser.add_argument("--soft-alpha", type=float, default=2.0)
    parser.add_argument("--n-configs", type=int, default=512)
    parser.add_argument("--n-eval", type=int, default=512)
    parser.add_argument("--n-bootstrap", type=int, default=120)
    parser.add_argument("--coarse-data-path", type=Path, default=Path("inverse_blocking_flow/outputs_fine16/coarse_kappac030_configs.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("inverse_blocking_flow/outputs_fine16"))
    parser.add_argument("--baseline-epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--kappa-lr-mult", type=float, default=0.2)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--hidden-channels", type=int, default=48)
    parser.add_argument("--cnn-depth", type=int, default=4)
    parser.add_argument("--conditioning-mode", choices=("physics",), default="physics")
    parser.add_argument("--reference-burn-in", type=int, default=400)
    parser.add_argument("--sample-interval", type=int, default=10)
    parser.add_argument("--batch-size-ref", type=int, default=64)
    parser.add_argument("--proposal-width", type=float, default=1.0)
    parser.add_argument("--correction-n", type=int, default=128)
    parser.add_argument("--correction-sweeps", type=str, default="0,10,20,50")
    parser.add_argument("--seed", type=int, default=989898)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--smoke", action="store_true")
    return parser


def differentiable_action(phi: torch.Tensor, kappa: torch.Tensor, lam: float) -> torch.Tensor:
    neighbor_sum = torch.roll(phi, -1, dims=-2) + torch.roll(phi, -1, dims=-1)
    local = phi.square() + lam * (phi.square() - 1.0).square()
    return (local - 2.0 * kappa * phi * neighbor_sum).sum(dim=(-2, -1))


def batch_observables(phi: torch.Tensor, kappa: torch.Tensor, lam: float) -> dict[str, torch.Tensor]:
    volume = phi.shape[-2] * phi.shape[-1]
    action = differentiable_action(phi, kappa, lam)
    phi2_cfg = phi.square().mean(dim=(-2, -1))
    phi4_cfg = phi.pow(4).mean(dim=(-2, -1))
    nn_cfg = 0.5 * (
        (phi * torch.roll(phi, -1, dims=-2)).mean(dim=(-2, -1))
        + (phi * torch.roll(phi, -1, dims=-1)).mean(dim=(-2, -1))
    )
    sum_phi2 = phi.square().sum(dim=(-2, -1))
    q2_cfg = sum_phi2.square() / float(volume * volume)
    mag = phi.mean(dim=(-2, -1))
    chi = volume * (mag.square().mean() - mag.mean().square())
    m2 = mag.square().mean()
    binder = 1.0 - mag.pow(4).mean() / (3.0 * m2.square().clamp_min(1e-12))
    fft = torch.fft.fftn(phi, dim=(-2, -1))
    power = (fft.real.square() + fft.imag.square()).mean(dim=0) / volume
    return {
        "S_density": action.mean() / float(volume),
        "phi2": phi2_cfg.mean(),
        "phi4": phi4_cfg.mean(),
        "NN": nn_cfg.mean(),
        "Q2": q2_cfg.mean(),
        "chi": chi,
        "binder": binder,
        "P10": power[1, 0],
        "P01": power[0, 1],
        "P11": power[1, 1],
    }


def scalar_observables(phi: torch.Tensor, kappa: float, lam: float) -> dict[str, float]:
    with torch.no_grad():
        obs = batch_observables(phi.float(), torch.tensor(kappa, dtype=phi.dtype), lam)
        return {key: float(value.detach().cpu().item()) for key, value in obs.items()}


def bootstrap_targets(phi: torch.Tensor, kappa: float, lam: float, n_bootstrap: int, seed: int) -> dict[str, dict[str, float]]:
    base = scalar_observables(phi, kappa, lam)
    gen = torch.Generator().manual_seed(seed)
    values = {key: [] for key in LOCAL_KEYS + IR_KEYS}
    n = phi.shape[0]
    for _ in range(n_bootstrap):
        idx = torch.randint(0, n, (n,), generator=gen)
        row = scalar_observables(phi[idx], kappa, lam)
        for key in values:
            values[key].append(row[key])
    out = {}
    for key, samples in values.items():
        tensor = torch.tensor(samples, dtype=torch.float64)
        stderr = float(tensor.std(unbiased=True).item())
        mean = base[key]
        scale = max(stderr, 0.05 * abs(mean), 1e-6)
        out[key] = {"mean": mean, "stderr": stderr, "scale": scale}
    return out


def load_reference(args: argparse.Namespace, kappa: float) -> torch.Tensor:
    path = args.output_dir / f"fine_reference_bootstrap_kappa_{str(kappa).replace('.', 'p')}.pt"
    return load_or_generate_fine_configs(
        path,
        n_configs=2048,
        fine_size=args.fine_size,
        params=Phi4Params(kappa=kappa, lam=args.lam),
        burn_in=args.reference_burn_in,
        interval=args.sample_interval,
        batch_size=args.batch_size_ref,
        proposal_width=args.proposal_width,
        seed=args.seed + int(round(10000 * kappa)),
        device=args.device,
    ).float()


def train_reverse_kl(
    flow: ConditionalDetailFlow,
    raw_kappa: torch.Tensor,
    psi: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    epochs: int,
) -> list[dict[str, float]]:
    loader = DataLoader(TensorDataset(psi), batch_size=args.batch_size, shuffle=True)
    opt = torch.optim.Adam(
        [{"params": flow.parameters(), "lr": args.lr}, {"params": [raw_kappa], "lr": args.lr * args.kappa_lr_mult}]
    )
    history = []
    for epoch in range(1, epochs + 1):
        losses = []
        kappas = []
        for (psi_b,) in loader:
            psi_b = psi_b.to(device)
            opt.zero_grad(set_to_none=True)
            cond = make_conditioning(psi_b, args.conditioning_mode)
            u, logq = flow.sample(cond)
            phi = soft_reconstruct(psi_b[:, 0], u)
            kappa = bounded_kappa(raw_kappa, args.kappa_f_min, args.kappa_f_max)
            loss = (differentiable_action(phi, kappa, args.lam) + soft_kernel_term(u, args.soft_alpha) + logq).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(flow.parameters()) + [raw_kappa], 10.0)
            opt.step()
            losses.append(float(loss.detach().cpu().item()))
            kappas.append(float(kappa.detach().cpu().item()))
        row = {"epoch": epoch, "loss": sum(losses) / len(losses), "kappa_f": sum(kappas) / len(kappas)}
        history.append(row)
        print(f"baseline epoch {epoch:04d} loss {row['loss']:.6g} kappa_f {row['kappa_f']:.6g}", flush=True)
    return history


def fine_tune(
    flow: ConditionalDetailFlow,
    raw_kappa: torch.Tensor,
    psi: torch.Tensor,
    targets: dict[str, dict[str, float]],
    variant: Variant,
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, float]]:
    params = list(flow.parameters())
    opt_params = [{"params": params, "lr": args.lr}]
    if not variant.freeze_kappa:
        opt_params.append({"params": [raw_kappa], "lr": args.lr * args.kappa_lr_mult})
    opt = torch.optim.Adam(opt_params)
    loader = DataLoader(TensorDataset(psi), batch_size=args.batch_size, shuffle=True)
    history = []
    for epoch in range(1, variant.epochs + 1):
        if variant.anneal:
            frac = epoch / max(1, variant.epochs)
            lambda_local = variant.lambda_local * frac
            lambda_ir = variant.lambda_ir * frac
        else:
            lambda_local = variant.lambda_local
            lambda_ir = variant.lambda_ir
        losses = []
        obs_losses = []
        kappas = []
        for (psi_b,) in loader:
            psi_b = psi_b.to(device)
            opt.zero_grad(set_to_none=True)
            cond = make_conditioning(psi_b, args.conditioning_mode)
            u, logq = flow.sample(cond)
            phi = soft_reconstruct(psi_b[:, 0], u)
            if variant.fixed_kappa is not None:
                kappa = torch.tensor(variant.fixed_kappa, dtype=phi.dtype, device=phi.device)
            else:
                kappa = bounded_kappa(raw_kappa, args.kappa_f_min, args.kappa_f_max)
            base = (differentiable_action(phi, kappa, args.lam) + soft_kernel_term(u, args.soft_alpha) + logq).mean()
            obs = batch_observables(phi, kappa, args.lam)
            local_loss = sum(((obs[key] - targets[key]["mean"]) / targets[key]["scale"]).square() for key in LOCAL_KEYS)
            ir_loss = sum(((obs[key] - targets[key]["mean"]) / targets[key]["scale"]).square() for key in IR_KEYS)
            obs_loss = lambda_local * local_loss + lambda_ir * ir_loss
            loss = base + obs_loss
            loss.backward()
            grad_params = params if variant.freeze_kappa else params + [raw_kappa]
            torch.nn.utils.clip_grad_norm_(grad_params, 10.0)
            opt.step()
            losses.append(float(loss.detach().cpu().item()))
            obs_losses.append(float(obs_loss.detach().cpu().item()))
            kappas.append(float(kappa.detach().cpu().item()))
        row = {
            "epoch": epoch,
            "loss": sum(losses) / len(losses),
            "obs_loss": sum(obs_losses) / len(obs_losses),
            "lambda_local": lambda_local,
            "lambda_IR": lambda_ir,
            "kappa_f": sum(kappas) / len(kappas),
        }
        history.append(row)
        print(f"{variant.name} epoch {epoch:04d} loss {row['loss']:.6g} obs {row['obs_loss']:.6g} kappa {row['kappa_f']:.6g}", flush=True)
    return history


@torch.no_grad()
def sample_model(
    flow: ConditionalDetailFlow,
    raw_kappa: torch.Tensor,
    psi: torch.Tensor,
    variant: Variant,
    args: argparse.Namespace,
    device: torch.device,
    seed_offset: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    psi_eval = psi[: args.n_eval].to(device)
    cond = make_conditioning(psi_eval, args.conditioning_mode)
    gen = torch.Generator(device=device).manual_seed(args.seed + seed_offset)
    u, logq = flow.sample(cond, generator=gen)
    phi = soft_reconstruct(psi_eval[:, 0], u).cpu()
    kappa = variant.fixed_kappa if variant.fixed_kappa is not None else float(bounded_kappa(raw_kappa, args.kappa_f_min, args.kappa_f_max).cpu().item())
    return phi, u.cpu(), logq.cpu(), float(kappa)


def aggregate_local_ir(phi: torch.Tensor, targets: dict[str, dict[str, float]], kappa: float, args: argparse.Namespace) -> tuple[float, float]:
    obs = scalar_observables(phi, kappa, args.lam)
    local = sum(abs((obs[key] - targets[key]["mean"]) / targets[key]["mean"]) for key in LOCAL_KEYS) / len(LOCAL_KEYS)
    ir = sum(abs((obs[key] - targets[key]["mean"]) / targets[key]["mean"]) for key in IR_KEYS) / len(IR_KEYS)
    return float(local), float(ir)


def aggregate_ref_error(phi: torch.Tensor, ref_phi: torch.Tensor, kappa: float, args: argparse.Namespace) -> float:
    obs = ensemble_summary(phi, Phi4Params(kappa=kappa, lam=args.lam))
    ref = ensemble_summary(ref_phi, Phi4Params(kappa=kappa, lam=args.lam))
    return float(sum(abs((obs[key] - ref[key]) / ref[key]) for key in OBS_AGG_KEYS) / len(OBS_AGG_KEYS))


def correction_rows(phi: torch.Tensor, ref_phi: torch.Tensor, args: argparse.Namespace) -> dict[str, list[dict[str, float]]]:
    sweeps = [int(x) for x in args.correction_sweeps.split(",") if x.strip()]
    n = min(args.correction_n, phi.shape[0])
    ref_obs = ensemble_summary(ref_phi[:n], Phi4Params(kappa=0.320, lam=args.lam))
    starts = {
        "upscaled": phi[:n].clone(),
        "hot": 0.5 * torch.randn((n, args.fine_size, args.fine_size), generator=torch.Generator().manual_seed(args.seed + 77)),
        "cold": torch.zeros((n, args.fine_size, args.fine_size)),
    }
    out = {}
    for name, start in starts.items():
        current = 0
        field = start.clone()
        rows = []
        gen = torch.Generator().manual_seed(args.seed + 9100)
        for target in sweeps:
            for _ in range(target - current):
                checkerboard_metropolis_sweep(field, Phi4Params(kappa=0.320, lam=args.lam), args.proposal_width, gen)
            current = target
            obs = ensemble_summary(field, Phi4Params(kappa=0.320, lam=args.lam))
            err = float(sum(abs((obs[key] - ref_obs[key]) / ref_obs[key]) for key in OBS_AGG_KEYS) / len(OBS_AGG_KEYS))
            rows.append({"sweeps": target, "aggregate_error": err})
        out[name] = rows
    return out


def evaluate(
    name: str,
    variant: Variant,
    flow: ConditionalDetailFlow,
    raw_kappa: torch.Tensor,
    history: list[dict[str, float]],
    psi: torch.Tensor,
    refs: dict[str, torch.Tensor],
    targets_032: dict[str, dict[str, float]],
    args: argparse.Namespace,
    device: torch.device,
    seed_offset: int,
) -> dict[str, object]:
    phi, u, logq, kappa = sample_model(flow, raw_kappa, psi, variant, args, device, seed_offset)
    kernel = soft_kernel_term(u, args.soft_alpha)
    logw = -phi4_action(phi, Phi4Params(kappa=kappa, lam=args.lam)) - kernel - logq
    corr = bootstrap_corr(phi, args.n_bootstrap, args.seed + seed_offset + 10000)
    ref_errors = {
        key: aggregate_ref_error(phi, ref_phi[: args.n_eval], float(key), args)
        for key, ref_phi in refs.items()
    }
    local_err, ir_err = aggregate_local_ir(phi, targets_032, 0.320, args)
    obs = ensemble_summary(phi, Phi4Params(kappa=0.320, lam=args.lam))
    return {
        "name": name,
        "variant": variant.__dict__,
        "history": history,
        "kappa_f_final": kappa,
        "observables_at_0p320": obs,
        "aggregate_error_vs_refs": ref_errors,
        "local_error_vs_0p320": local_err,
        "IR_error_vs_0p320": ir_err,
        "correlation_bootstrap": corr,
        "logw": stabilized_logw_stats(logw),
        "logq": tensor_stats(logq),
        "kernel_term": tensor_stats(kernel),
        "correction_mcmc_0p320": correction_rows(phi, refs["0.320"], args),
    }


def write_report(path: Path, summary: dict[str, object]) -> None:
    lines = [
        "# kappa_c=0.30 Observable Loss Fine Tune",
        "",
        "Primary reference is high-stat kappa_f=0.320. Errors use generated samples from the saved independent coarse ensemble.",
        "",
        "## Summary",
        "",
        "| variant | kappa_f | err 0.320 | err 0.325 | err 0.330 | local err | IR err | xi/L | chi | Binder | logw std | A/R proxy | corr50 upscaled |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["variants"]:
        corr = row["correlation_bootstrap"]["estimate"]
        boot = row["correlation_bootstrap"]["bootstrap"]
        last_corr = max(row["correction_mcmc_0p320"]["upscaled"], key=lambda x: x["sweeps"])
        lines.append(
            f"| {row['name']} | {row['kappa_f_final']:.6g} | {row['aggregate_error_vs_refs']['0.320']:.6g} | "
            f"{row['aggregate_error_vs_refs']['0.325']:.6g} | {row['aggregate_error_vs_refs']['0.330']:.6g} | "
            f"{row['local_error_vs_0p320']:.6g} | {row['IR_error_vs_0p320']:.6g} | "
            f"{corr['xi_2nd_over_L']:.6g} +/- {boot['xi_2nd']['stderr'] / 16.0:.3g} | "
            f"{corr['chi']:.6g} +/- {boot['chi']['stderr']:.3g} | {corr['binder']:.6g} +/- {boot['binder']['stderr']:.3g} | "
            f"{row['logw']['std_logw_centered']:.6g} | {row['logw']['independence_acceptance_proxy']:.6g} | {last_corr['aggregate_error']:.6g} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_outputs(path: Path, summary: dict[str, object]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    rows = summary["variants"]
    labels = [row["name"].split("_", 1)[0] for row in rows]
    with PdfPages(path) as pdf:
        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        axes[0, 0].bar(labels, [row["aggregate_error_vs_refs"]["0.320"] for row in rows])
        axes[0, 0].set_ylabel("agg err vs 0.320")
        axes[0, 1].bar(labels, [row["local_error_vs_0p320"] for row in rows], label="local")
        axes[0, 1].bar(labels, [row["IR_error_vs_0p320"] for row in rows], alpha=0.55, label="IR")
        axes[0, 1].legend()
        axes[1, 0].bar(labels, [row["correlation_bootstrap"]["estimate"]["xi_2nd_over_L"] for row in rows])
        axes[1, 0].axhline(summary["targets"]["0.320"]["xi_2nd_over_L"], color="k", ls="--")
        axes[1, 0].set_ylabel("xi_2nd/L")
        axes[1, 1].bar(labels, [row["kappa_f_final"] for row in rows])
        axes[1, 1].axhline(0.320, color="k", ls="--")
        axes[1, 1].set_ylabel("kappa_f")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if not hasattr(args, "n_configs"):
        args.n_configs = args.n_configs if hasattr(args, "n_configs") else 512
    coarse = load_or_generate_coarse(args)
    psi = coarse.unsqueeze(1).float()
    refs = {f"{kappa:.3f}": load_reference(args, kappa) for kappa in REF_KAPPAS}
    targets_032 = bootstrap_targets(refs["0.320"], 0.320, args.lam, args.n_bootstrap, args.seed + 123)
    targets_meta = {
        "0.320": {
            "xi_2nd_over_L": bootstrap_corr(refs["0.320"], args.n_bootstrap, args.seed + 321)["estimate"]["xi_2nd_over_L"],
            "targets": targets_032,
        }
    }

    base_flow = ConditionalDetailFlow(args.layers, args.hidden_channels, args.cnn_depth, 6, 4).to(device)
    base_raw = torch.tensor(
        raw_from_kappa(args.kappa_f_initial, args.kappa_f_min, args.kappa_f_max),
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    baseline_history = train_reverse_kl(base_flow, base_raw, psi, args, device, args.baseline_epochs)
    base_state = copy.deepcopy(base_flow.state_dict())
    base_raw_value = base_raw.detach().clone()

    results = []
    models = {
        "baseline_state": base_state,
        "baseline_raw_kappa": base_raw_value.cpu(),
    }
    variant_plan = [Variant("smoke_local_IR", 1, 0.1, 0.05)] if args.smoke else VARIANTS
    for i, variant in enumerate(variant_plan):
        flow = ConditionalDetailFlow(args.layers, args.hidden_channels, args.cnn_depth, 6, 4).to(device)
        flow.load_state_dict(base_state)
        raw = base_raw_value.detach().clone().to(device).requires_grad_(not variant.freeze_kappa)
        history = baseline_history if variant.name == "A_baseline_B" else fine_tune(flow, raw, psi, targets_032, variant, args, device)
        result = evaluate(variant.name, variant, flow, raw, history, psi, refs, targets_032, args, device, 1000 + i * 100)
        results.append(result)
        models[variant.name] = {
            "model": flow.state_dict(),
            "raw_kappa": raw.detach().cpu(),
            "result": result,
        }

    summary = {
        "setup": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "targets": targets_meta,
        "variants": results,
    }
    summary_path = args.output_dir / "kappac030_observable_loss_summary.json"
    report_path = args.output_dir / "kappac030_observable_loss_report.md"
    plots_path = args.output_dir / "kappac030_observable_loss_plots.pdf"
    model_path = args.output_dir / "kappac030_observable_loss_models.pt"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_report(report_path, summary)
    plot_outputs(plots_path, summary)
    torch.save(models, model_path)
    print(f"wrote {summary_path}")
    print(f"wrote {report_path}")
    print(f"wrote {plots_path}")
    print(f"wrote {model_path}")


if __name__ == "__main__":
    main()
