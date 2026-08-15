"""Scan regularized weighted-kernel inverse-blocking models."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/matplotlib")

import torch

from inverse_blocking_flow.data import load_or_generate_fine_configs
from inverse_blocking_flow.haar import average_block, weighted_kernel_normalization
from inverse_blocking_flow.logw_kappa_scan_blocked_coarse import ensemble_summary
from inverse_blocking_flow.phi4 import Phi4Params
from inverse_blocking_flow.weighted_kernel_trainable_scan import evaluate_variant


LAMBDA_PRIOR = [0.0, 1e-4, 1e-3, 1e-2, 1e-1]
LAMBDA_HIGHP = [0.0, 1e-4, 1e-3, 1e-2]
REPRESENTATIVE_MOMENTA = [
    ("W_0_0", 0.0, 0.0),
    ("W_pi_over_2_0", 0.5 * math.pi, 0.0),
    ("W_pi_over_2_pi_over_2", 0.5 * math.pi, 0.5 * math.pi),
    ("W_pi_0", math.pi, 0.0),
    ("W_pi_pi", math.pi, math.pi),
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
    parser.add_argument("--eta", type=float, default=0.25)
    parser.add_argument("--weighted-a", type=float, default=0.25)
    parser.add_argument("--weighted-b", type=float, default=0.0625)
    parser.add_argument("--weighted-min-a", type=float, default=0.15)
    parser.add_argument("--weighted-min-b", type=float, default=0.02)
    parser.add_argument("--mle-epochs", type=int, default=20)
    parser.add_argument("--reverse-kl-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--hidden-channels", type=int, default=48)
    parser.add_argument("--cnn-depth", type=int, default=3)
    parser.add_argument("--conditioning-mode", choices=("physics",), default="physics")
    parser.add_argument("--kappa-min", type=float, default=0.20)
    parser.add_argument("--kappa-max", type=float, default=0.38)
    parser.add_argument("--kappa-step", type=float, default=0.01)
    parser.add_argument("--fine-window", type=float, default=0.025)
    parser.add_argument("--fine-step", type=float, default=0.0025)
    parser.add_argument("--burn-in", type=int, default=200)
    parser.add_argument("--sample-interval", type=int, default=10)
    parser.add_argument("--proposal-width", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=929292)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--skip-training", action="store_true")
    return parser.parse_args()


def tag_for(lambda_prior: float, lambda_highp: float) -> str:
    def fmt(x: float) -> str:
        return f"{x:g}".replace(".", "p").replace("-", "m")

    return f"weighted_reg_prior_{fmt(lambda_prior)}_highp_{fmt(lambda_highp)}"


def response_value(a: float, b: float, kx: float, ky: float) -> float:
    n = float(weighted_kernel_normalization(a, b))
    return n * (1.0 + 2.0 * a * (math.cos(kx) + math.cos(ky)) + 4.0 * b * math.cos(kx) * math.cos(ky))


def representative_responses(a: float, b: float) -> dict[str, float]:
    return {name: response_value(a, b, kx, ky) for name, kx, ky in REPRESENTATIVE_MOMENTA}


def run_training(args: argparse.Namespace, lambda_prior: float, lambda_highp: float) -> Path:
    tag = tag_for(lambda_prior, lambda_highp)
    common = [
        sys.executable,
        "-B",
        "inverse_blocking_flow/train_conditional_flow.py",
        "--blocking-mode",
        "weighted",
        "--conditioning-mode",
        "physics",
        "--weighted-a",
        str(args.weighted_a),
        "--weighted-b",
        str(args.weighted_b),
        "--train-weighted-kernel",
        "true",
        "--weighted-kernel-prior-reg",
        str(lambda_prior),
        "--weighted-highp-reg",
        str(lambda_highp),
        "--weighted-min-a",
        str(args.weighted_min_a),
        "--weighted-min-b",
        str(args.weighted_min_b),
        "--eta",
        str(args.eta),
        "--use-eta-scaling",
        "true",
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
    mle_tag = f"{tag}_mle"
    reverse_tag = f"{tag}_reverse_kl"
    subprocess.run(common + ["--mode", "mle", "--epochs", str(args.mle_epochs), "--checkpoint-tag", mle_tag], check=True)
    mle_checkpoint = args.output_dir / f"conditional_detail_flow_{mle_tag}.pt"
    subprocess.run(
        common
        + [
            "--mode",
            "reverse_kl",
            "--epochs",
            str(args.reverse_kl_epochs),
            "--checkpoint",
            str(mle_checkpoint),
            "--checkpoint-tag",
            reverse_tag,
        ],
        check=True,
    )
    return args.output_dir / f"conditional_detail_flow_{reverse_tag}.pt"


def write_report(path: Path, summary: dict[str, object]) -> None:
    rows = summary["models"]
    best_kappa = min(rows, key=lambda row: abs(row["kappa_min"] - 0.31))
    best_logw = min(rows, key=lambda row: row["logw_std"])
    lines = [
        "# Weighted Kernel Regularized Scan",
        "",
        "All models use fixed `eta=0.25`, physics conditioning, trainable `a,b`, and lower-bound clipping.",
        "",
        "## Main Answer",
        "",
        (
            f"Closest logw-width kappa minimum to 0.31 is `{best_kappa['kappa_min']:.6g}` "
            f"for `{best_kappa['name']}` with logw std `{best_kappa['logw_std']:.6g}`. "
            f"Lowest logw std is `{best_logw['logw_std']:.6g}` for `{best_logw['name']}` "
            f"with kappa_min `{best_logw['kappa_min']:.6g}`."
        ),
        "",
        "## Model Summary",
        "",
        "| model | lambda_prior | lambda_highp | a | b | N | W00 | Wpi/2,0 | Wpi/2,pi/2 | Wpi,0 | Wpi,pi | logw std | ESS/N | kappa_min | obs kappa_min | agg obs err | detail std err | corr(psi^2,d^2) err |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        w = row["representative_W"]
        lines.append(
            f"| {row['name']} | {row['lambda_prior']:.6g} | {row['lambda_highp']:.6g} | "
            f"{row['learned_a']:.6g} | {row['learned_b']:.6g} | {row['learned_N']:.6g} | "
            f"{w['W_0_0']:.6g} | {w['W_pi_over_2_0']:.6g} | {w['W_pi_over_2_pi_over_2']:.6g} | "
            f"{w['W_pi_0']:.6g} | {w['W_pi_pi']:.6g} | {row['logw_std']:.6g} | "
            f"{row['ESS_over_N']:.6g} | {row['kappa_min']:.6g} | {row['obs_kappa_min']:.6g} | "
            f"{row['aggregate_observable_error']:.6g} | {row['detail_std_error']:.6g} | "
            f"{row['corr_psi2_d2_error']:.6g} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_outputs(path: Path, summary: dict[str, object]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    rows = summary["models"]
    x = [row["lambda_prior"] for row in rows]
    y = [row["lambda_highp"] for row in rows]
    with PdfPages(path) as pdf:
        for key, title in [
            ("logw_std", "logw std"),
            ("ESS_over_N", "ESS/N"),
            ("kappa_min", "kappa_min"),
            ("aggregate_observable_error", "aggregate observable error"),
            ("corr_psi2_d2_error", "corr(psi^2,d^2) error"),
        ]:
            fig, ax = plt.subplots(figsize=(6.2, 4.8))
            sc = ax.scatter(x, y, c=[row[key] for row in rows], s=120, cmap="viridis")
            ax.set_xscale("symlog", linthresh=1e-5)
            ax.set_yscale("symlog", linthresh=1e-5)
            ax.set_xlabel("lambda_prior")
            ax.set_ylabel("lambda_highp")
            ax.set_title(title)
            fig.colorbar(sc, ax=ax)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.2, 4.8))
        sc = ax.scatter([row["learned_a"] for row in rows], [row["learned_b"] for row in rows], c=[row["kappa_min"] for row in rows], s=100)
        ax.axvline(summary["setup"]["weighted_min_a"], color="k", ls=":", lw=1)
        ax.axhline(summary["setup"]["weighted_min_b"], color="k", ls=":", lw=1)
        ax.set_xlabel("learned a")
        ax.set_ylabel("learned b")
        ax.set_title("learned kernel parameters colored by kappa_min")
        fig.colorbar(sc, ax=ax)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = {}
    for lambda_prior in LAMBDA_PRIOR:
        for lambda_highp in LAMBDA_HIGHP:
            tag = tag_for(lambda_prior, lambda_highp)
            if args.skip_training:
                checkpoints[tag] = args.output_dir / f"conditional_detail_flow_{tag}_reverse_kl.pt"
            else:
                checkpoints[tag] = run_training(args, lambda_prior, lambda_highp)

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

    rows = []
    for lambda_prior in LAMBDA_PRIOR:
        for lambda_highp in LAMBDA_HIGHP:
            tag = tag_for(lambda_prior, lambda_highp)
            variant = {
                "name": tag,
                "train_weighted_kernel": True,
                "weighted_kernel_reg": lambda_prior,
            }
            row = evaluate_variant(args, checkpoints[tag], variant, phi_true, d_true, true_observables, args.eta)
            row["name"] = tag
            row["lambda_prior"] = lambda_prior
            row["lambda_highp"] = lambda_highp
            row["representative_W"] = representative_responses(row["learned_a"], row["learned_b"])
            rows.append(row)

    summary = {
        "setup": {
            "eta": args.eta,
            "fine_size": args.fine_size,
            "coarse_size": args.fine_size // 2,
            "kappa_true": args.kappa_true,
            "lambda": args.lam,
            "n_eval": n_eval,
            "conditioning_mode": "physics",
            "weighted_a0": args.weighted_a,
            "weighted_b0": args.weighted_b,
            "weighted_min_a": args.weighted_min_a,
            "weighted_min_b": args.weighted_min_b,
            "lambda_prior_grid": LAMBDA_PRIOR,
            "lambda_highp_grid": LAMBDA_HIGHP,
            "mle_epochs": args.mle_epochs,
            "reverse_kl_epochs": args.reverse_kl_epochs,
        },
        "true_observables_kappa_true": true_observables,
        "models": rows,
    }
    summary_path = args.output_dir / "weighted_kernel_regularized_summary.json"
    report_path = args.output_dir / "weighted_kernel_regularized_report.md"
    plots_path = args.output_dir / "weighted_kernel_regularized_plots.pdf"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_report(report_path, summary)
    plot_outputs(plots_path, summary)
    print(f"wrote {summary_path}")
    print(f"wrote {report_path}")
    print(f"wrote {plots_path}")


if __name__ == "__main__":
    main()
