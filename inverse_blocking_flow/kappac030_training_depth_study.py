"""Controlled training-depth study for independent kappa_c=0.30 upscaling."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/matplotlib")

from inverse_blocking_flow.flow import ConditionalDetailFlow
from inverse_blocking_flow.kappac030_trainable_kappaf_upscale import (
    OBS_KEYS,
    aggregate_for_keys,
    bounded_kappa,
    build_parser as base_parser,
    correction_test,
    generate_reference,
    load_or_generate_coarse,
    naive_upscale,
    raw_from_kappa,
    sample_upscaled,
    stabilized_logw_stats,
    tensor_stats,
    train_upscaler,
)
from inverse_blocking_flow.logw_kappa_scan_blocked_coarse import ensemble_summary
from inverse_blocking_flow.haar import soft_kernel_term
from inverse_blocking_flow.phi4 import Phi4Params, checkerboard_metropolis_sweep, phi4_action


REF_KAPPAS = [0.315, 0.320, 0.325, 0.330]
SWEEPS = [0, 10, 20, 50]


@dataclass(frozen=True)
class Variant:
    name: str
    qd_epochs: int
    joint_epochs: int
    layers: int
    hidden_channels: int
    cnn_depth: int


VARIANTS = [
    Variant("A_baseline_current", 50, 20, 6, 48, 4),
    Variant("B_longer_conditional_only", 150, 0, 6, 48, 4),
    Variant("C_longer_conditional_mild_joint", 150, 10, 6, 48, 4),
    Variant("D_longer_all", 150, 50, 6, 48, 4),
    Variant("E_larger_model", 150, 10, 8, 64, 5),
]


def build_parser() -> argparse.ArgumentParser:
    parser = base_parser()
    parser.description = __doc__
    parser.set_defaults(epochs=1, correction_sweeps="0,10,20,50")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="run a tiny one-variant validation pass")
    return parser


def train_phase(
    flow: ConditionalDetailFlow,
    raw: torch.Tensor,
    psi: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    epochs: int,
    phase: str,
) -> list[dict[str, float | str]]:
    if epochs <= 0:
        return []
    old_epochs = args.epochs
    args.epochs = epochs
    rows = train_upscaler(flow, raw, psi, args, device)
    args.epochs = old_epochs
    out = []
    for row in rows:
        out.append({"phase": phase, **row})
    return out


def generate_hot_cold_corrections(
    args: argparse.Namespace,
    reference_observables: dict[str, float],
    target_kappa: float,
) -> dict[str, list[dict[str, object]]]:
    n = args.correction_n
    params = Phi4Params(kappa=target_kappa, lam=args.lam)
    starts = {
        "hot": 0.5 * torch.randn((n, args.fine_size, args.fine_size), generator=torch.Generator().manual_seed(args.seed + 770)),
        "cold": torch.zeros((n, args.fine_size, args.fine_size)),
    }
    out = {}
    for name, phi0 in starts.items():
        phi = phi0.clone()
        current = 0
        rows = []
        gen = torch.Generator().manual_seed(args.seed + int(round(10000 * target_kappa)) + 99)
        for target in SWEEPS:
            for _ in range(target - current):
                checkerboard_metropolis_sweep(phi, params, args.proposal_width, gen)
            current = target
            obs = ensemble_summary(phi, params)
            rows.append(
                {
                    "sweeps": target,
                    "observables": obs,
                    "aggregate_error": aggregate_for_keys(obs, reference_observables),
                }
            )
        out[name] = rows
    return out


def upscaled_correction(
    phi_start: torch.Tensor,
    args: argparse.Namespace,
    reference_observables: dict[str, float],
    target_kappa: float,
) -> list[dict[str, object]]:
    n = min(args.correction_n, phi_start.shape[0])
    phi = phi_start[:n].clone()
    params = Phi4Params(kappa=target_kappa, lam=args.lam)
    current = 0
    rows = []
    gen = torch.Generator().manual_seed(args.seed + int(round(10000 * target_kappa)) + 199)
    for target in SWEEPS:
        for _ in range(target - current):
            checkerboard_metropolis_sweep(phi, params, args.proposal_width, gen)
        current = target
        obs = ensemble_summary(phi, params)
        rows.append(
            {
                "sweeps": target,
                "observables": obs,
                "aggregate_error": aggregate_for_keys(obs, reference_observables),
            }
        )
    return rows


def evaluate_variant(
    variant: Variant,
    coarse: torch.Tensor,
    references: dict[str, torch.Tensor],
    reference_observables: dict[str, dict[str, float]],
    hot_cold: dict[str, dict[str, list[dict[str, object]]]],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    print(f"=== variant {variant.name} ===", flush=True)
    psi = coarse.unsqueeze(1).float()
    flow = ConditionalDetailFlow(variant.layers, variant.hidden_channels, variant.cnn_depth, 6, 4).to(device)
    raw = torch.tensor(
        raw_from_kappa(args.kappa_f_initial, args.kappa_f_min, args.kappa_f_max),
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    history = []
    history.extend(train_phase(flow, raw, psi, args, device, variant.qd_epochs, "conditional_reverse_kl"))
    history.extend(train_phase(flow, raw, psi, args, device, variant.joint_epochs, "joint_continuation"))
    learned_kappa = float(bounded_kappa(raw, args.kappa_f_min, args.kappa_f_max).detach().cpu().item())
    phi, u, logq = sample_upscaled(flow, psi, args, device, seed_offset=hash(variant.name) % 10000)
    kernel = soft_kernel_term(u, args.soft_alpha)
    logw = -phi4_action(phi, Phi4Params(kappa=learned_kappa, lam=args.lam)) - kernel - logq
    generated_observables = {
        "learned": ensemble_summary(phi, Phi4Params(kappa=learned_kappa, lam=args.lam)),
        "fixed_0p320": ensemble_summary(phi, Phi4Params(kappa=args.kappa_f_initial, lam=args.lam)),
    }
    naive_phi = naive_upscale(psi[: args.n_eval])
    refs = {}
    for key, ref_obs in reference_observables.items():
        kappa = float(key)
        gen_obs_at_ref = ensemble_summary(phi, Phi4Params(kappa=kappa, lam=args.lam))
        naive_obs_at_ref = ensemble_summary(naive_phi, Phi4Params(kappa=kappa, lam=args.lam))
        refs[key] = {
            "reference_observables": ref_obs,
            "generated_observables_at_reference_kappa": gen_obs_at_ref,
            "naive_observables_at_reference_kappa": naive_obs_at_ref,
            "aggregate_error_generated": aggregate_for_keys(gen_obs_at_ref, ref_obs),
            "aggregate_error_naive": aggregate_for_keys(naive_obs_at_ref, ref_obs),
        }
    best_ref = min(refs.items(), key=lambda item: item[1]["aggregate_error_generated"])[0]
    correction = {}
    for target_key in ["0.320", f"{learned_kappa:.3f}"]:
        nearest = min(reference_observables, key=lambda key: abs(float(key) - float(target_key)))
        target_kappa = float(nearest)
        correction[target_key] = {
            "target_kappa_used": target_kappa,
            "upscaled": upscaled_correction(phi, args, reference_observables[nearest], target_kappa),
            "hot": hot_cold[nearest]["hot"],
            "cold": hot_cold[nearest]["cold"],
        }
    return {
        "name": variant.name,
        "qd_reverse_kl_epochs": variant.qd_epochs,
        "joint_full_reverse_kl_epochs": variant.joint_epochs,
        "layers": variant.layers,
        "hidden_channels": variant.hidden_channels,
        "cnn_depth": variant.cnn_depth,
        "kappa_f_initial": args.kappa_f_initial,
        "kappa_f_final": learned_kappa,
        "kappa_f_trajectory": history,
        "best_reference_kappa_f": best_ref,
        "observables": generated_observables,
        "reference_comparison": refs,
        "logw_diagnostic": stabilized_logw_stats(logw),
        "logq": tensor_stats(logq),
        "kernel_term": tensor_stats(kernel),
        "correction_mcmc": correction,
    }


def write_report(path: Path, summary: dict[str, object]) -> None:
    rows = summary["variants"]
    lines = [
        "# kappa_c=0.30 Training Depth Study",
        "",
        "The `joint` phase is implemented as an additional labeled continuation of the same upscaling reverse-KL objective over the independent empirical coarse ensemble; no learned `q_c` is used in this test.",
        "",
        "## Variant Summary",
        "",
        "| variant | qd epochs | joint epochs | model | kappa final | best ref | err 0.315 | err 0.320 | err 0.325 | err 0.330 | logw std | ESS/N | A/R proxy | S mean | S std | phi2 | Binder | susc | NN | low-p | high-p |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        obs = row["observables"]["learned"]
        refs = row["reference_comparison"]
        model = f"{row['layers']}x{row['hidden_channels']}d{row['cnn_depth']}"
        lines.append(
            f"| {row['name']} | {row['qd_reverse_kl_epochs']} | {row['joint_full_reverse_kl_epochs']} | {model} | "
            f"{row['kappa_f_final']:.6g} | {row['best_reference_kappa_f']} | "
            f"{refs['0.315']['aggregate_error_generated']:.6g} | {refs['0.320']['aggregate_error_generated']:.6g} | "
            f"{refs['0.325']['aggregate_error_generated']:.6g} | {refs['0.330']['aggregate_error_generated']:.6g} | "
            f"{row['logw_diagnostic']['std_logw_centered']:.6g} | {row['logw_diagnostic']['ess_over_n']:.6g} | "
            f"{row['logw_diagnostic']['independence_acceptance_proxy']:.6g} | {obs['S_mean']:.6g} | "
            f"{obs['S_std']:.6g} | {obs['phi2']:.6g} | {obs['binder']:.6g} | {obs['susceptibility']:.6g} | "
            f"{obs['NN_corr']:.6g} | {obs['low_p_power']:.6g} | {obs['high_p_power']:.6g} |"
        )
    best_032 = min(rows, key=lambda row: row["reference_comparison"]["0.320"]["aggregate_error_generated"])
    best_any = min(rows, key=lambda row: row["reference_comparison"][row["best_reference_kappa_f"]]["aggregate_error_generated"])
    lines.extend(
        [
            "",
            "## Main Answers",
            "",
            f"1. Best error versus kappa=0.320 is `{best_032['name']}` with `{best_032['reference_comparison']['0.320']['aggregate_error_generated']:.6g}`.",
            f"2. Final kappas span `{min(r['kappa_f_final'] for r in rows):.6g}` to `{max(r['kappa_f_final'] for r in rows):.6g}`.",
            f"3. Best overall reference match is `{best_any['name']}` at reference `{best_any['best_reference_kappa_f']}`.",
            "4. Capacity comparison is A-D versus E in the table above.",
            "",
            "## Correction MCMC",
            "",
            "| variant | target | start | sweeps | agg err |",
            "|---|---:|---|---:|---:|",
        ]
    )
    for row in rows:
        for target, starts in row["correction_mcmc"].items():
            for start, entries in starts.items():
                if start == "target_kappa_used":
                    continue
                for entry in entries:
                    lines.append(f"| {row['name']} | {target} | {start} | {entry['sweeps']} | {entry['aggregate_error']:.6g} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_outputs(path: Path, summary: dict[str, object]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    rows = summary["variants"]
    names = [row["name"].split("_", 1)[0] for row in rows]
    with PdfPages(path) as pdf:
        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        axes[0, 0].bar(names, [row["kappa_f_final"] for row in rows])
        axes[0, 0].axhline(0.32, color="k", ls="--")
        axes[0, 0].set_ylabel("final kappa_f")
        for ref in ["0.315", "0.320", "0.325", "0.330"]:
            axes[0, 1].plot(names, [row["reference_comparison"][ref]["aggregate_error_generated"] for row in rows], marker="o", label=ref)
        axes[0, 1].set_ylabel("aggregate obs error")
        axes[0, 1].legend(fontsize=8)
        axes[1, 0].bar(names, [row["logw_diagnostic"]["std_logw_centered"] for row in rows])
        axes[1, 0].set_ylabel("logw std")
        axes[1, 1].bar(names, [row["logw_diagnostic"]["independence_acceptance_proxy"] for row in rows])
        axes[1, 1].set_ylabel("A/R proxy")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        for row in rows:
            xs = list(range(1, len(row["kappa_f_trajectory"]) + 1))
            ax.plot(xs, [h["kappa_f"] for h in row["kappa_f_trajectory"]], label=row["name"].split("_", 1)[0])
        ax.axhline(0.32, color="k", ls="--")
        ax.set_xlabel("phase epoch")
        ax.set_ylabel("kappa_f")
        ax.legend(fontsize=8)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    coarse = load_or_generate_coarse(args)
    references = {}
    reference_observables = {}
    for kappa in REF_KAPPAS:
        key = f"{kappa:.3f}"
        references[key] = generate_reference(args, kappa)
        reference_observables[key] = ensemble_summary(references[key], Phi4Params(kappa=kappa, lam=args.lam))
    hot_cold = {
        key: generate_hot_cold_corrections(args, obs, float(key))
        for key, obs in reference_observables.items()
        if key in {"0.320", "0.323", "0.325"}
    }
    if "0.323" not in hot_cold:
        hot_cold["0.323"] = hot_cold["0.325"]
    variant_plan = [Variant("smoke", 1, 1, 2, 8, 2)] if args.smoke else VARIANTS
    variants = []
    for variant in variant_plan:
        variants.append(evaluate_variant(variant, coarse, references, reference_observables, hot_cold, args, device))
    setup = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    summary = {
        "setup": setup,
        "reference_observables": reference_observables,
        "variants": variants,
    }
    summary_path = args.output_dir / "kappac030_training_depth_summary.json"
    report_path = args.output_dir / "kappac030_training_depth_report.md"
    plots_path = args.output_dir / "kappac030_training_depth_plots.pdf"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_report(report_path, summary)
    plot_outputs(plots_path, summary)
    print(f"wrote {summary_path}")
    print(f"wrote {report_path}")
    print(f"wrote {plots_path}")


if __name__ == "__main__":
    main()
