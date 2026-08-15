"""Train and validate the production fine16 upscaler for kappa_c=0.30."""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
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
from inverse_blocking_flow.kappac030_observable_loss import (
    IR_KEYS,
    LOCAL_KEYS,
    OBS_AGG_KEYS,
    aggregate_ref_error,
    batch_observables,
    differentiable_action,
    load_reference,
    scalar_observables,
)
from inverse_blocking_flow.kappac030_trainable_kappaf_upscale import (
    bounded_kappa,
    raw_from_kappa,
    stabilized_logw_stats,
    tensor_stats,
)
from inverse_blocking_flow.logw_kappa_scan_blocked_coarse import ensemble_summary
from inverse_blocking_flow.phi4 import Phi4Params, checkerboard_metropolis_sweep, generate_phi4_configs, phi4_action


REF_KAPPAS = [0.320, 0.325, 0.330]


@dataclass(frozen=True)
class Candidate:
    name: str
    trainable_kappa: bool
    fixed_kappa: float | None
    lambda_local: float
    lambda_ir: float
    epochs: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fine-size", type=int, default=16)
    parser.add_argument("--coarse-size", type=int, default=8)
    parser.add_argument("--kappa-c", type=float, default=0.300)
    parser.add_argument("--kappa-f-initial", type=float, default=0.320)
    parser.add_argument("--kappa-f-min", type=float, default=0.30)
    parser.add_argument("--kappa-f-max", type=float, default=0.335)
    parser.add_argument("--lambda-f", dest="lam", type=float, default=1.0)
    parser.add_argument("--soft-alpha", type=float, default=2.0)
    parser.add_argument("--n-configs", type=int, default=1024)
    parser.add_argument("--n-eval", type=int, default=1024)
    parser.add_argument("--n-bootstrap", type=int, default=160)
    parser.add_argument("--baseline-epochs", type=int, default=150)
    parser.add_argument("--finetune-epochs", type=int, default=50)
    parser.add_argument("--lambda-local", type=float, default=0.1)
    parser.add_argument("--lambda-ir", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--kappa-lr-mult", type=float, default=0.2)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--hidden-channels", type=int, default=48)
    parser.add_argument("--cnn-depth", type=int, default=4)
    parser.add_argument("--conditioning-mode", choices=("physics",), default="physics")
    parser.add_argument("--coarse-data-path", type=Path, default=Path("inverse_blocking_flow/outputs_fine16/production_coarse_kappac030_configs.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("inverse_blocking_flow/outputs_fine16"))
    parser.add_argument("--reference-burn-in", type=int, default=400)
    parser.add_argument("--burn-in", type=int, default=400)
    parser.add_argument("--sample-interval", type=int, default=10)
    parser.add_argument("--batch-size-ref", type=int, default=64)
    parser.add_argument("--proposal-width", type=float, default=1.0)
    parser.add_argument("--correction-n", type=int, default=256)
    parser.add_argument("--correction-sweeps", type=str, default="0,10,20,50")
    parser.add_argument("--seed", type=int, default=4242001)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--smoke", action="store_true")
    return parser


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def ensure_coarse_configs(args: argparse.Namespace) -> torch.Tensor:
    if args.coarse_data_path.exists():
        data = torch.load(args.coarse_data_path, map_location="cpu")
        phi = data["phi"] if isinstance(data, dict) else data
        if phi.shape[-2:] != (args.coarse_size, args.coarse_size):
            raise ValueError(f"coarse data shape {phi.shape[-2:]} does not match {(args.coarse_size, args.coarse_size)}")
        if phi.shape[0] >= args.n_configs:
            return phi[: args.n_configs].float()
    phi = generate_phi4_configs(
        args.n_configs,
        args.coarse_size,
        Phi4Params(kappa=args.kappa_c, lam=args.lam),
        burn_in=args.burn_in,
        interval=args.sample_interval,
        batch_size=args.batch_size,
        proposal_width=args.proposal_width,
        seed=args.seed + 11,
        device=args.device,
    ).float()
    args.coarse_data_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"phi": phi, "kappa_c": args.kappa_c, "lambda": args.lam, "seed": args.seed + 11}, args.coarse_data_path)
    return phi


def bootstrap_targets_broad(phi: torch.Tensor, kappa: float, lam: float, n_bootstrap: int, seed: int) -> dict[str, dict[str, float]]:
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
        frac = 0.10 if key in IR_KEYS else 0.05
        out[key] = {"mean": mean, "stderr": stderr, "scale": max(stderr, frac * abs(mean), 1e-6)}
    return out


def target_diagnostics(refs: dict[str, torch.Tensor], targets: dict[str, dict[str, float]], args: argparse.Namespace) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, phi in refs.items():
        corr = bootstrap_corr(phi, args.n_bootstrap, args.seed + int(float(key) * 100000))
        out[key] = {
            "observables": ensemble_summary(phi[: args.n_eval], Phi4Params(kappa=float(key), lam=args.lam)),
            "correlation_bootstrap": corr,
        }
    out["0.320_targets"] = targets
    out["coarse_xi_over_L_target"] = 0.3655
    return out


def train_reverse_kl(
    flow: ConditionalDetailFlow,
    raw_kappa: torch.Tensor,
    psi: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, float]]:
    loader = DataLoader(TensorDataset(psi), batch_size=args.batch_size, shuffle=True)
    opt = torch.optim.Adam(
        [{"params": flow.parameters(), "lr": args.lr}, {"params": [raw_kappa], "lr": args.lr * args.kappa_lr_mult}]
    )
    history = []
    for epoch in range(1, args.baseline_epochs + 1):
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
            losses.append(float(loss.detach().cpu()))
            kappas.append(float(kappa.detach().cpu()))
        row = {"phase": "reverse_kl", "epoch": epoch, "loss": sum(losses) / len(losses), "kappa_f": sum(kappas) / len(kappas)}
        history.append(row)
        print(f"reverse_kl epoch {epoch:04d} loss {row['loss']:.6g} kappa_f {row['kappa_f']:.6g}", flush=True)
    return history


def fine_tune_candidate(
    flow: ConditionalDetailFlow,
    raw_kappa: torch.Tensor,
    psi: torch.Tensor,
    targets: dict[str, dict[str, float]],
    candidate: Candidate,
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, float]]:
    opt_params: list[dict[str, object]] = [{"params": flow.parameters(), "lr": args.lr}]
    if candidate.trainable_kappa:
        opt_params.append({"params": [raw_kappa], "lr": args.lr * args.kappa_lr_mult})
    opt = torch.optim.Adam(opt_params)
    loader = DataLoader(TensorDataset(psi), batch_size=args.batch_size, shuffle=True)
    history = []
    for epoch in range(1, candidate.epochs + 1):
        frac = epoch / max(1, candidate.epochs)
        lambda_local = candidate.lambda_local * frac
        lambda_ir = candidate.lambda_ir * frac
        losses = []
        obs_losses = []
        kappas = []
        for (psi_b,) in loader:
            psi_b = psi_b.to(device)
            opt.zero_grad(set_to_none=True)
            cond = make_conditioning(psi_b, args.conditioning_mode)
            u, logq = flow.sample(cond)
            phi = soft_reconstruct(psi_b[:, 0], u)
            if candidate.fixed_kappa is None:
                kappa = bounded_kappa(raw_kappa, args.kappa_f_min, args.kappa_f_max)
            else:
                kappa = torch.tensor(candidate.fixed_kappa, dtype=phi.dtype, device=phi.device)
            base = (differentiable_action(phi, kappa, args.lam) + soft_kernel_term(u, args.soft_alpha) + logq).mean()
            obs = batch_observables(phi, kappa, args.lam)
            local_loss = sum(((obs[key] - targets[key]["mean"]) / targets[key]["scale"]).square() for key in LOCAL_KEYS)
            ir_loss = sum(((obs[key] - targets[key]["mean"]) / targets[key]["scale"]).square() for key in IR_KEYS)
            obs_loss = lambda_local * local_loss + lambda_ir * ir_loss
            loss = base + obs_loss
            loss.backward()
            grad_params = list(flow.parameters()) + ([raw_kappa] if candidate.trainable_kappa else [])
            torch.nn.utils.clip_grad_norm_(grad_params, 10.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
            obs_losses.append(float(obs_loss.detach().cpu()))
            kappas.append(float(kappa.detach().cpu()))
        row = {
            "phase": candidate.name,
            "epoch": epoch,
            "loss": sum(losses) / len(losses),
            "obs_loss": sum(obs_losses) / len(obs_losses),
            "lambda_local": lambda_local,
            "lambda_IR": lambda_ir,
            "kappa_f": sum(kappas) / len(kappas),
        }
        history.append(row)
        print(f"{candidate.name} epoch {epoch:04d} loss {row['loss']:.6g} obs {row['obs_loss']:.6g} kappa {row['kappa_f']:.6g}", flush=True)
    return history


@torch.no_grad()
def sample_upscaled(
    flow: ConditionalDetailFlow,
    raw_kappa: torch.Tensor,
    candidate: Candidate,
    psi: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    seed_offset: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    psi_eval = psi[: args.n_eval].to(device)
    cond = make_conditioning(psi_eval, args.conditioning_mode)
    gen = torch.Generator(device=device).manual_seed(args.seed + seed_offset)
    u, logq = flow.sample(cond, generator=gen)
    phi = soft_reconstruct(psi_eval[:, 0], u).cpu()
    if candidate.fixed_kappa is None:
        kappa = float(bounded_kappa(raw_kappa, args.kappa_f_min, args.kappa_f_max).detach().cpu())
    else:
        kappa = candidate.fixed_kappa
    return phi, u.cpu(), logq.cpu(), float(kappa)


def local_ir_errors(phi: torch.Tensor, targets: dict[str, dict[str, float]], kappa: float, args: argparse.Namespace) -> tuple[float, float]:
    obs = scalar_observables(phi, kappa, args.lam)
    local = sum(abs((obs[key] - targets[key]["mean"]) / targets[key]["mean"]) for key in LOCAL_KEYS) / len(LOCAL_KEYS)
    ir = sum(abs((obs[key] - targets[key]["mean"]) / targets[key]["mean"]) for key in IR_KEYS) / len(IR_KEYS)
    return float(local), float(ir)


def generated_observables(phi: torch.Tensor, kappa: float, args: argparse.Namespace) -> dict[str, object]:
    obs = scalar_observables(phi, kappa, args.lam)
    corr = bootstrap_corr(phi, args.n_bootstrap, args.seed + 6161)
    return {
        **obs,
        "xi_2nd_over_L": corr["estimate"]["xi_2nd_over_L"],
        "xi_2nd_over_L_stderr": None if corr["bootstrap"]["xi_2nd"]["stderr"] is None else corr["bootstrap"]["xi_2nd"]["stderr"] / args.fine_size,
        "correlation_bootstrap": corr,
    }


def correction_rows(phi: torch.Tensor, args: argparse.Namespace, kappa: float) -> dict[str, list[dict[str, object]]]:
    sweeps = [int(x) for x in args.correction_sweeps.split(",") if x.strip()]
    n = min(args.correction_n, phi.shape[0])
    params = Phi4Params(kappa=kappa, lam=args.lam)
    starts = {
        "upscaled": phi[:n].clone(),
        "hot": 0.5 * torch.randn((n, args.fine_size, args.fine_size), generator=torch.Generator().manual_seed(args.seed + 77)),
        "cold": torch.zeros((n, args.fine_size, args.fine_size)),
    }
    out: dict[str, list[dict[str, object]]] = {}
    for name, start in starts.items():
        field = start.clone()
        rows = []
        current = 0
        gen = torch.Generator().manual_seed(args.seed + 9100 + int(round(kappa * 100000)))
        for target in sweeps:
            for _ in range(target - current):
                checkerboard_metropolis_sweep(field, params, args.proposal_width, gen)
            current = target
            rows.append({"sweeps": target, "observables": generated_observables(field, kappa, args)})
        out[name] = rows
    return out


def candidate_score(result: dict[str, object]) -> float:
    corr320 = result["correction_mcmc"]["0.320"]["upscaled"][-1]["observables"]
    xi = result["generated_observables"]["xi_2nd_over_L"]
    xi_penalty = abs(float(xi) - 0.3655) / 0.3655 if xi is not None else 1.0
    corr_phi2 = abs(float(corr320["phi2"]) - float(result["generated_observables"]["phi2"])) / max(abs(float(result["generated_observables"]["phi2"])), 1e-12)
    return (
        float(result["aggregate_error_vs_refs"]["0.320"])
        + 0.5 * float(result["local_error_vs_0p320"])
        + 0.5 * float(result["IR_error_vs_0p320"])
        + 0.5 * xi_penalty
        + 0.1 * corr_phi2
    )


@torch.no_grad()
def evaluate_candidate(
    name: str,
    candidate: Candidate,
    flow: ConditionalDetailFlow,
    raw_kappa: torch.Tensor,
    history: list[dict[str, float]],
    psi: torch.Tensor,
    refs: dict[str, torch.Tensor],
    targets: dict[str, dict[str, float]],
    args: argparse.Namespace,
    device: torch.device,
    seed_offset: int,
) -> dict[str, object]:
    phi, u, logq, kappa = sample_upscaled(flow, raw_kappa, candidate, psi, args, device, seed_offset)
    kernel = soft_kernel_term(u, args.soft_alpha)
    logw = -phi4_action(phi, Phi4Params(kappa=kappa, lam=args.lam)) - kernel - logq
    ref_errors = {key: aggregate_ref_error(phi, ref[: args.n_eval], float(key), args) for key, ref in refs.items()}
    local_err, ir_err = local_ir_errors(phi, targets, 0.320, args)
    obs = generated_observables(phi, 0.320, args)
    result = {
        "name": name,
        "candidate": candidate.__dict__,
        "kappa_f_final": kappa,
        "history": history,
        "kappa_f_trajectory": [row["kappa_f"] for row in history],
        "generated_observables": obs,
        "aggregate_error_vs_refs": ref_errors,
        "local_error_vs_0p320": local_err,
        "IR_error_vs_0p320": ir_err,
        "logw": stabilized_logw_stats(logw),
        "logq": tensor_stats(logq),
        "kernel_term": tensor_stats(kernel),
        "correction_mcmc": {
            f"{kappa:.3f}": correction_rows(phi, args, kappa),
            "0.320": correction_rows(phi, args, 0.320),
        },
    }
    result["selection_score"] = candidate_score(result)
    return result


def write_report(path: Path, summary: dict[str, object]) -> None:
    lines = [
        "# Production Upscaler kappa_c=0.30 soft alpha=2",
        "",
        f"Default checkpoint: `{summary['default_model']}`.",
        "",
        "## Candidate Diagnostics",
        "",
        "| candidate | kappa_f | score | err 0.320 | err 0.325 | err 0.330 | local err | IR err | xi/L | chi | Binder | logw std |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["candidates"]:
        obs = row["generated_observables"]
        xi_err = obs["xi_2nd_over_L_stderr"]
        xi_text = f"{obs['xi_2nd_over_L']:.6g}" if xi_err is None else f"{obs['xi_2nd_over_L']:.6g} +/- {xi_err:.3g}"
        lines.append(
            f"| {row['name']} | {row['kappa_f_final']:.6g} | {row['selection_score']:.6g} | "
            f"{row['aggregate_error_vs_refs']['0.320']:.6g} | {row['aggregate_error_vs_refs']['0.325']:.6g} | "
            f"{row['aggregate_error_vs_refs']['0.330']:.6g} | {row['local_error_vs_0p320']:.6g} | "
            f"{row['IR_error_vs_0p320']:.6g} | {xi_text} | {obs['chi']:.6g} | {obs['binder']:.6g} | "
            f"{row['logw']['std_logw_centered']:.6g} |"
        )
    lines.extend(["", "## Decision", "", summary["decision_note"]])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_outputs(path: Path, summary: dict[str, object]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    rows = summary["candidates"]
    names = [row["name"] for row in rows]
    with PdfPages(path) as pdf:
        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        axes[0, 0].bar(names, [row["aggregate_error_vs_refs"]["0.320"] for row in rows])
        axes[0, 0].set_ylabel("agg err vs 0.320")
        axes[0, 1].bar(names, [row["local_error_vs_0p320"] for row in rows], label="local")
        axes[0, 1].bar(names, [row["IR_error_vs_0p320"] for row in rows], alpha=0.55, label="IR")
        axes[0, 1].legend()
        axes[1, 0].bar(names, [row["generated_observables"]["xi_2nd_over_L"] for row in rows])
        axes[1, 0].axhline(0.3655, color="k", ls="--", label="coarse target")
        axes[1, 0].set_ylabel("xi_2nd/L")
        axes[1, 0].legend()
        axes[1, 1].bar(names, [row["logw"]["std_logw_centered"] for row in rows])
        axes[1, 1].set_ylabel("logw std")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4))
        for row in rows:
            ax.plot(row["kappa_f_trajectory"], label=row["name"])
        ax.axhline(0.320, color="k", ls="--", lw=1)
        ax.set_ylabel("kappa_f")
        ax.set_xlabel("training epoch")
        ax.legend()
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    if args.smoke:
        args.n_configs = 64
        args.n_eval = 64
        args.n_bootstrap = 8
        args.baseline_epochs = 1
        args.finetune_epochs = 1
        args.correction_n = 16
        args.correction_sweeps = "0,1"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    coarse = ensure_coarse_configs(args)
    if coarse.shape[0] < args.n_eval:
        raise ValueError(f"need at least {args.n_eval} coarse configs, got {coarse.shape[0]}")
    psi = coarse.unsqueeze(1).float()
    refs = {f"{kappa:.3f}": load_reference(args, kappa) for kappa in REF_KAPPAS}
    targets = bootstrap_targets_broad(refs["0.320"], 0.320, args.lam, args.n_bootstrap, args.seed + 123)
    ref_diag = target_diagnostics(refs, targets, args)

    base_flow = ConditionalDetailFlow(args.layers, args.hidden_channels, args.cnn_depth, 6, 4).to(device)
    raw_base = torch.tensor(
        raw_from_kappa(args.kappa_f_initial, args.kappa_f_min, args.kappa_f_max),
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    baseline_history = train_reverse_kl(base_flow, raw_base, psi, args, device)
    base_state = copy.deepcopy(base_flow.state_dict())
    raw_state = raw_base.detach().clone()
    candidates = [
        Candidate("trainable_annealed", True, None, args.lambda_local, args.lambda_ir, args.finetune_epochs),
        Candidate("fixed_kappa_0p320", False, 0.320, args.lambda_local, args.lambda_ir, args.finetune_epochs),
    ]
    results = []
    checkpoint_candidates = {}
    for i, candidate in enumerate(candidates):
        flow = ConditionalDetailFlow(args.layers, args.hidden_channels, args.cnn_depth, 6, 4).to(device)
        flow.load_state_dict(base_state)
        raw = raw_state.detach().clone().to(device).requires_grad_(candidate.trainable_kappa)
        fine_history = fine_tune_candidate(flow, raw, psi, targets, candidate, args, device)
        history = baseline_history + fine_history
        result = evaluate_candidate(candidate.name, candidate, flow, raw, history, psi, refs, targets, args, device, 1000 + i * 500)
        results.append(result)
        checkpoint_candidates[candidate.name] = {
            "model": copy.deepcopy(flow.state_dict()),
            "raw_kappa": raw.detach().cpu(),
            "result": result,
        }
    default = min(results, key=lambda row: row["selection_score"])
    default_name = default["name"]
    if default_name == "fixed_kappa_0p320":
        decision = "The fixed-kappa model is selected as default because its combined local/IR/correction stability score is lower."
    else:
        decision = "The trainable-kappa annealed model is selected as default because its combined local/IR/correction stability score is lower."

    architecture = {"layers": args.layers, "hidden_channels": args.hidden_channels, "cnn_depth": args.cnn_depth}
    setup = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    metadata = {
        "coarse_size": args.coarse_size,
        "fine_size": args.fine_size,
        "kappa_c": args.kappa_c,
        "kappa_f_initial": args.kappa_f_initial,
        "kappa_f_final": default["kappa_f_final"],
        "lambda": args.lam,
        "soft_alpha": args.soft_alpha,
        "blocking_mode": "soft_haar",
        "conditioning_mode": args.conditioning_mode,
        "architecture": architecture,
        "training_schedule": {
            "reverse_kl_epochs": args.baseline_epochs,
            "observable_ir_finetune_epochs": args.finetune_epochs,
            "annealed_lambda_local": [0.0, args.lambda_local],
            "annealed_lambda_IR": [0.0, args.lambda_ir],
        },
        "observable_loss_weights": {"local": LOCAL_KEYS, "IR": IR_KEYS, "lambda_local": args.lambda_local, "lambda_IR": args.lambda_ir},
        "reference_diagnostics": ref_diag,
        "random_seeds": {"main": args.seed, "coarse": args.seed + 11},
        "git_commit": git_commit(),
        "default_model": default_name,
        "decision_note": decision,
    }
    summary = {
        "setup": setup,
        "metadata": metadata,
        "default_model": default_name,
        "decision_note": decision,
        "candidates": results,
    }

    stem = "production_upscaler_kappac030_softalpha2"
    checkpoint_path = args.output_dir / f"{stem}.pt"
    metadata_path = args.output_dir / f"{stem}_metadata.json"
    summary_path = args.output_dir / f"{stem}_summary.json"
    report_path = args.output_dir / f"{stem}_report.md"
    plots_path = args.output_dir / f"{stem}_plots.pdf"
    selected = checkpoint_candidates[default_name]
    torch.save(
        {
            "default_model": default_name,
            "model": selected["model"],
            "raw_kappa": selected["raw_kappa"],
            "kappa_f_final": default["kappa_f_final"],
            "metadata": metadata,
            "candidate_models": checkpoint_candidates,
        },
        checkpoint_path,
    )
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_report(report_path, summary)
    plot_outputs(plots_path, summary)
    print(f"default_model {default_name}")
    print(f"wrote {checkpoint_path}")
    print(f"wrote {metadata_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {report_path}")
    print(f"wrote {plots_path}")


if __name__ == "__main__":
    main()
