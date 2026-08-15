"""Train and compare fixed/trainable weighted inverse-blocking kernels."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/matplotlib")

from inverse_blocking_flow.data import load_or_generate_fine_configs
from inverse_blocking_flow.detail_correlation_diagnostics import (
    channel_diagnostics,
    mean_abs_channel_error,
)
from inverse_blocking_flow.flow import ConditionalDetailFlow, make_conditioning, n_conditioning_channels
from inverse_blocking_flow.haar import (
    average_block,
    reconstruct_from_weighted_block,
    weighted_block,
    weighted_kernel_normalization,
    weighted_ll_fft_stats,
)
from inverse_blocking_flow.logw_kappa_scan_blocked_coarse import (
    aggregate_abs_rel,
    ensemble_summary,
    kappa_grid,
    stabilized_logw_stats,
)
from inverse_blocking_flow.phi4 import Phi4Params, phi4_action


VARIANTS = [
    {
        "name": "A_fixed_weights",
        "train_weighted_kernel": False,
        "weighted_kernel_reg": 0.0,
    },
    {
        "name": "B_trainable_weights",
        "train_weighted_kernel": True,
        "weighted_kernel_reg": 0.0,
    },
    {
        "name": "C_trainable_reg_1e-3",
        "train_weighted_kernel": True,
        "weighted_kernel_reg": 1e-3,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fine-size", type=int, default=32)
    parser.add_argument("--kappa-true", type=float, default=0.31)
    parser.add_argument("--lambda-f", dest="lam", type=float, default=1.0)
    parser.add_argument("--n-configs", type=int, default=512)
    parser.add_argument("--n-eval", type=int, default=512)
    parser.add_argument("--data-path", type=Path, default=Path("inverse_blocking_flow/outputs/fine_configs.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("inverse_blocking_flow/outputs"))
    parser.add_argument("--eta", type=float, default=None)
    parser.add_argument("--eta-summary", type=Path, default=Path("inverse_blocking_flow/outputs/weighted_eta_fixed_summary.json"))
    parser.add_argument("--weighted-a", type=float, default=0.25)
    parser.add_argument("--weighted-b", type=float, default=0.0625)
    parser.add_argument("--mle-epochs", type=int, default=20)
    parser.add_argument("--reverse-kl-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--hidden-channels", type=int, default=48)
    parser.add_argument("--cnn-depth", type=int, default=3)
    parser.add_argument("--conditioning-mode", choices=("basic", "physics"), default="basic")
    parser.add_argument("--kappa-min", type=float, default=0.20)
    parser.add_argument("--kappa-max", type=float, default=0.38)
    parser.add_argument("--kappa-step", type=float, default=0.01)
    parser.add_argument("--fine-window", type=float, default=0.025)
    parser.add_argument("--fine-step", type=float, default=0.0025)
    parser.add_argument("--burn-in", type=int, default=200)
    parser.add_argument("--sample-interval", type=int, default=10)
    parser.add_argument("--proposal-width", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=919191)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--skip-training", action="store_true")
    return parser.parse_args()


def choose_eta(args: argparse.Namespace) -> tuple[float, str]:
    if args.eta is not None:
        return args.eta, "command line"
    if args.eta_summary.exists():
        summary = json.loads(args.eta_summary.read_text(encoding="utf-8"))
        rows = summary.get("eta_scan", summary.get("variants", summary.get("rows", [])))
        if rows:
            best = min(
                rows,
                key=lambda row: (
                    row.get("logw_std", row.get("logw", {}).get("std_logw_centered", float("inf"))),
                    -row.get("ess_over_n", row.get("logw", {}).get("ess_over_n", 0.0)),
                ),
            )
            return float(best.get("eta", 0.25)), str(args.eta_summary)
    return 0.25, "fallback"


def run_training(args: argparse.Namespace, eta: float, variant: dict[str, object]) -> Path:
    tag_base = str(variant["name"])
    train_kernel = "true" if variant["train_weighted_kernel"] else "false"
    reg = str(variant["weighted_kernel_reg"])
    common = [
        sys.executable,
        "-B",
        "inverse_blocking_flow/train_conditional_flow.py",
        "--blocking-mode",
        "weighted",
        "--weighted-a",
        str(args.weighted_a),
        "--weighted-b",
        str(args.weighted_b),
        "--eta",
        str(eta),
        "--use-eta-scaling",
        "true",
        "--train-weighted-kernel",
        train_kernel,
        "--weighted-kernel-reg",
        reg,
        "--fine-size",
        str(args.fine_size),
        "--lambda",
        str(args.lam),
        "--kappa-fine",
        str(args.kappa_true),
        "--n-configs",
        str(args.n_configs),
        "--data-path",
        str(args.data_path),
        "--output-dir",
        str(args.output_dir),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--layers",
        str(args.layers),
        "--hidden-channels",
        str(args.hidden_channels),
        "--cnn-depth",
        str(args.cnn_depth),
        "--conditioning-mode",
        args.conditioning_mode,
        "--burn-in",
        str(args.burn_in),
        "--sample-interval",
        str(args.sample_interval),
        "--proposal-width",
        str(args.proposal_width),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
    ]
    mle_tag = f"{tag_base}_mle"
    rk_tag = f"{tag_base}_reverse_kl"
    mle_cmd = common + ["--mode", "mle", "--epochs", str(args.mle_epochs), "--checkpoint-tag", mle_tag]
    subprocess.run(mle_cmd, check=True)
    mle_checkpoint = args.output_dir / f"conditional_detail_flow_{mle_tag}.pt"
    rk_cmd = common + [
        "--mode",
        "reverse_kl",
        "--epochs",
        str(args.reverse_kl_epochs),
        "--checkpoint",
        str(mle_checkpoint),
        "--checkpoint-tag",
        rk_tag,
    ]
    subprocess.run(rk_cmd, check=True)
    return args.output_dir / f"conditional_detail_flow_{rk_tag}.pt"


def load_flow(checkpoint: Path, fallback_args: argparse.Namespace, device: torch.device) -> tuple[ConditionalDetailFlow, dict[str, object]]:
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    args_meta = state.get("args", {})
    mode = str(state.get("conditioning_mode") or args_meta.get("conditioning_mode") or fallback_args.conditioning_mode)
    n_cond = int(state.get("n_conditioning_channels") or n_conditioning_channels(mode))
    layers = int(args_meta.get("layers", fallback_args.layers))
    hidden = int(args_meta.get("hidden_channels", fallback_args.hidden_channels))
    depth = int(args_meta.get("cnn_depth", fallback_args.cnn_depth))
    flow = ConditionalDetailFlow(layers, hidden, depth, n_cond).to(device)
    flow.load_state_dict(state["model"])
    flow.eval()
    return flow, state


@torch.no_grad()
def evaluate_variant(
    args: argparse.Namespace,
    checkpoint: Path,
    variant: dict[str, object],
    phi_true: torch.Tensor,
    d_true: torch.Tensor,
    true_observables: dict[str, float],
    eta: float,
) -> dict[str, object]:
    device = phi_true.device
    flow, state = load_flow(checkpoint, args, device)
    args_meta = state.get("args", {})
    a = float(state.get("weighted_a", args_meta.get("weighted_a", args.weighted_a)))
    b = float(state.get("weighted_b", args_meta.get("weighted_b", args.weighted_b)))
    n = float(weighted_kernel_normalization(a, b))
    psi = weighted_block(phi_true, a, b, eta=eta, use_eta_scaling=True)
    cond_mode = str(state.get("conditioning_mode") or args_meta.get("conditioning_mode") or args.conditioning_mode)
    cond = make_conditioning(psi, cond_mode)
    generator = torch.Generator(device=device).manual_seed(args.seed + 17)
    d_flow, logq = flow.sample(cond, generator=generator)
    phi_rec = reconstruct_from_weighted_block(psi, d_flow, a, b, eta=eta, use_eta_scaling=True)

    params_true = Phi4Params(kappa=args.kappa_true, lam=args.lam)
    proposal_obs_true = ensemble_summary(phi_rec.cpu(), params_true)

    rows = []
    coarse_rows = []
    for kappa in kappa_grid(args.kappa_min, args.kappa_max, args.kappa_step):
        params = Phi4Params(kappa=kappa, lam=args.lam)
        logw = -phi4_action(phi_rec, params) - logq
        obs = ensemble_summary(phi_rec.cpu(), params)
        coarse_rows.append(
            {
                "kappa_f": kappa,
                "logw": stabilized_logw_stats(logw),
                "observables": obs,
                "aggregate_abs_rel_error_vs_true_kappa_true": aggregate_abs_rel(obs, true_observables),
            }
        )
    best_initial = min(coarse_rows, key=lambda row: row["logw"]["std_logw_centered"])
    rows.extend(coarse_rows)
    start = max(args.kappa_min, float(best_initial["kappa_f"]) - args.fine_window)
    stop = min(args.kappa_max, float(best_initial["kappa_f"]) + args.fine_window)
    seen = {row["kappa_f"] for row in rows}
    for kappa in kappa_grid(start, stop, args.fine_step):
        if kappa in seen:
            continue
        params = Phi4Params(kappa=kappa, lam=args.lam)
        logw = -phi4_action(phi_rec, params) - logq
        obs = ensemble_summary(phi_rec.cpu(), params)
        rows.append(
            {
                "kappa_f": kappa,
                "logw": stabilized_logw_stats(logw),
                "observables": obs,
                "aggregate_abs_rel_error_vs_true_kappa_true": aggregate_abs_rel(obs, true_observables),
            }
        )
    rows = sorted(rows, key=lambda row: row["kappa_f"])
    best_width = min(rows, key=lambda row: row["logw"]["std_logw_centered"])
    best_obs = min(rows, key=lambda row: row["aggregate_abs_rel_error_vs_true_kappa_true"])

    true_detail_diag = channel_diagnostics(d_true.cpu(), psi.cpu())
    flow_detail_diag = channel_diagnostics(d_flow.cpu(), psi.cpu())
    detail_std_error = mean_abs_channel_error(flow_detail_diag, true_detail_diag, "std")
    corr_psi2_d2_error = mean_abs_channel_error(flow_detail_diag, true_detail_diag, "corr_phi_c2_detail_amp")
    fft = weighted_ll_fft_stats(tuple(psi.shape[-2:]), a, b)

    return {
        "name": variant["name"],
        "checkpoint": str(checkpoint),
        "eta": eta,
        "Delta_phi": 0.5 * eta,
        "Z_eta": 2.0 ** (-0.5 * eta),
        "train_weighted_kernel": bool(variant["train_weighted_kernel"]),
        "weighted_kernel_reg": float(variant["weighted_kernel_reg"]),
        "learned_a": a,
        "learned_b": b,
        "learned_N": n,
        "fourier_A_w": fft,
        "logw_std": float(best_width["logw"]["std_logw_centered"]),
        "ESS_over_N": float(best_width["logw"]["ess_over_n"]),
        "kappa_min": float(best_width["kappa_f"]),
        "obs_kappa_min": float(best_obs["kappa_f"]),
        "aggregate_observable_error": float(best_obs["aggregate_abs_rel_error_vs_true_kappa_true"]),
        "aggregate_observable_error_at_kappa_true": float(aggregate_abs_rel(proposal_obs_true, true_observables)),
        "detail_std_error": float(detail_std_error),
        "corr_psi2_d2_error": float(corr_psi2_d2_error),
        "proposal_observables_at_kappa_true": proposal_obs_true,
        "best_by_logw_width": best_width,
        "best_by_observable_error": best_obs,
    }


def write_report(path: Path, summary: dict[str, object]) -> None:
    rows = summary["variants"]
    best_overlap = min(rows, key=lambda row: row["logw_std"])
    fixed = rows[0]
    improved = best_overlap["logw_std"] < fixed["logw_std"] and best_overlap["ESS_over_N"] >= fixed["ESS_over_N"]
    lines = [
        "# Weighted Kernel Trainable Scan",
        "",
        f"Eta used: `{summary['eta']:.6g}` from `{summary['eta_source']}`.",
        "",
        "## Main Answer",
        "",
        (
            f"Optimizing the blocking kernel {'improves' if improved else 'does not clearly improve'} overlap beyond eta scaling alone. "
            f"Best logw width is `{best_overlap['logw_std']:.6g}` for `{best_overlap['name']}`, "
            f"compared with fixed-kernel `{fixed['logw_std']:.6g}`."
        ),
        "",
        "## Variant Summary",
        "",
        "| variant | a | b | N | min A_w | max A_w | logw std | ESS/N | kappa_min | obs kappa_min | agg obs err | detail std err | corr(psi^2,d^2) err |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        fft = row["fourier_A_w"]
        lines.append(
            f"| {row['name']} | {row['learned_a']:.6g} | {row['learned_b']:.6g} | {row['learned_N']:.6g} | "
            f"{fft['min_real']:.6g} | {fft['max_real']:.6g} | {row['logw_std']:.6g} | {row['ESS_over_N']:.6g} | "
            f"{row['kappa_min']:.6g} | {row['obs_kappa_min']:.6g} | {row['aggregate_observable_error']:.6g} | "
            f"{row['detail_std_error']:.6g} | {row['corr_psi2_d2_error']:.6g} |"
        )
    path.write_text("\n".join(lines) + "\n")


def plot_outputs(path: Path, summary: dict[str, object]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    rows = summary["variants"]
    names = [row["name"].replace("_", "\n") for row in rows]
    with PdfPages(path) as pdf:
        fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5))
        axes[0, 0].bar(names, [row["logw_std"] for row in rows])
        axes[0, 0].set_ylabel("logw std")
        axes[0, 1].bar(names, [row["ESS_over_N"] for row in rows])
        axes[0, 1].set_ylabel("ESS/N")
        axes[1, 0].plot(names, [row["kappa_min"] for row in rows], marker="o", label="logw")
        axes[1, 0].plot(names, [row["obs_kappa_min"] for row in rows], marker="s", label="observables")
        axes[1, 0].axhline(summary["setup"]["kappa_true"], color="k", ls="--", lw=1)
        axes[1, 0].set_ylabel("kappa minimum")
        axes[1, 0].legend(fontsize=8)
        axes[1, 1].bar(names, [row["corr_psi2_d2_error"] for row in rows], label="corr")
        axes[1, 1].bar(names, [row["detail_std_error"] for row in rows], alpha=0.5, label="std")
        axes[1, 1].set_ylabel("detail diagnostic error")
        axes[1, 1].legend(fontsize=8)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        ax.scatter([row["learned_a"] for row in rows], [row["learned_b"] for row in rows])
        for row in rows:
            ax.annotate(row["name"], (row["learned_a"], row["learned_b"]), fontsize=8)
        ax.set_xlabel("a")
        ax.set_ylabel("b")
        ax.set_title("Learned weighted kernel parameters")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    eta, eta_source = choose_eta(args)
    checkpoints = {}
    if not args.skip_training:
        for variant in VARIANTS:
            checkpoints[variant["name"]] = run_training(args, eta, variant)
    else:
        for variant in VARIANTS:
            checkpoints[variant["name"]] = args.output_dir / f"conditional_detail_flow_{variant['name']}_reverse_kl.pt"

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
    _ll, d_true = average_block(phi_true)
    true_observables = ensemble_summary(phi_true.cpu(), params_true)

    rows = [
        evaluate_variant(args, checkpoints[variant["name"]], variant, phi_true, d_true, true_observables, eta)
        for variant in VARIANTS
    ]
    summary = {
        "eta": eta,
        "eta_source": eta_source,
        "setup": {
            "fine_size": args.fine_size,
            "coarse_size": args.fine_size // 2,
            "kappa_true": args.kappa_true,
            "lambda": args.lam,
            "n_eval": n_eval,
            "weighted_a_init": args.weighted_a,
            "weighted_b_init": args.weighted_b,
            "mle_epochs": args.mle_epochs,
            "reverse_kl_epochs": args.reverse_kl_epochs,
            "architecture": {
                "layers": args.layers,
                "hidden_channels": args.hidden_channels,
                "cnn_depth": args.cnn_depth,
                "conditioning_mode": args.conditioning_mode,
            },
        },
        "true_observables_kappa_true": true_observables,
        "variants": rows,
    }
    summary_path = args.output_dir / "weighted_kernel_trainable_summary.json"
    report_path = args.output_dir / "weighted_kernel_trainable_report.md"
    plots_path = args.output_dir / "weighted_kernel_trainable_plots.pdf"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_report(report_path, summary)
    plot_outputs(plots_path, summary)
    print(f"wrote {summary_path}")
    print(f"wrote {report_path}")
    print(f"wrote {plots_path}")


if __name__ == "__main__":
    main()
