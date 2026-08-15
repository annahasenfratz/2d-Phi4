"""Train and evaluate soft weighted-blocking conditional flows."""

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
from inverse_blocking_flow.flow import ConditionalDetailFlow, make_conditioning, n_conditioning_channels
from inverse_blocking_flow.haar import (
    eta_scaling_factor,
    soft_weighted_block,
    soft_weighted_kernel_term,
    soft_weighted_reconstruct,
    weighted_kernel_normalization,
    weighted_ll_fft_stats,
)
from inverse_blocking_flow.logw_kappa_scan_blocked_coarse import aggregate_abs_rel, ensemble_summary, kappa_grid, stabilized_logw_stats
from inverse_blocking_flow.phi4 import Phi4Params, phi4_action
from inverse_blocking_flow.soft_blocking_scan import corr, u_channel_stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alphas", type=str, default="0.5,1.0,2.0,5.0")
    parser.add_argument("--fine-size", type=int, default=32)
    parser.add_argument("--kappa", type=float, default=0.31)
    parser.add_argument("--lambda-f", dest="lam", type=float, default=1.0)
    parser.add_argument("--n-configs", type=int, default=512)
    parser.add_argument("--n-eval", type=int, default=256)
    parser.add_argument("--data-path", type=Path, default=Path("inverse_blocking_flow/outputs/fine_configs.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("inverse_blocking_flow/outputs"))
    parser.add_argument("--mle-epochs", type=int, default=20)
    parser.add_argument("--reverse-kl-epochs", type=int, default=50)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--hidden-channels", type=int, default=48)
    parser.add_argument("--cnn-depth", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--conditioning-mode", choices=("basic", "physics"), default="physics")
    parser.add_argument("--weighted-a", type=float, default=0.25)
    parser.add_argument("--weighted-b", type=float, default=0.0625)
    parser.add_argument("--eta", type=float, default=0.25)
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--seed", type=int, default=939393)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    return parser


def alpha_tag(alpha: float) -> str:
    return f"soft_weighted_alpha_{alpha:g}".replace(".", "p")


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def load_flow(path: Path, device: torch.device) -> tuple[ConditionalDetailFlow, str]:
    state = torch.load(path, map_location=device, weights_only=False)
    meta = state.get("args", {})
    mode = state.get("conditioning_mode") or meta.get("conditioning_mode") or "physics"
    n_cond = int(state.get("n_conditioning_channels") or n_conditioning_channels(mode))
    n_detail = int(state.get("n_detail_channels") or 4)
    layers = int(meta.get("layers", 6))
    hidden = int(meta.get("hidden_channels", 48))
    depth = int(meta.get("cnn_depth", 4))
    flow = ConditionalDetailFlow(layers, hidden, depth, n_cond, n_detail).to(device)
    flow.load_state_dict(state["model"])
    flow.eval()
    return flow, str(mode)


def diagnostic_errors(generated: dict[str, object], true: dict[str, object]) -> dict[str, float]:
    gen_channels = generated["channels"]
    true_channels = true["channels"]
    detail_names = ["HL", "LH", "HH"]
    return {
        "rho_std_error": abs(gen_channels["rho"]["std"] - true_channels["rho"]["std"]),
        "rho_corr_psi2_rho2_error": abs(gen_channels["rho"]["corr_psi2_amp"] - true_channels["rho"]["corr_psi2_amp"]),
        "detail_std_error": sum(abs(gen_channels[name]["std"] - true_channels[name]["std"]) for name in detail_names) / 3.0,
        "corr_psi2_d2_error": sum(
            abs(gen_channels[name]["corr_psi2_amp"] - true_channels[name]["corr_psi2_amp"]) for name in detail_names
        )
        / 3.0,
    }


def kappa_scan(
    phi: torch.Tensor,
    u: torch.Tensor,
    logq: torch.Tensor,
    alpha: float,
    true_summary: dict[str, float],
    lam: float,
) -> dict[str, object]:
    rows = []
    kernel = soft_weighted_kernel_term(u, alpha)
    for kappa in kappa_grid(0.20, 0.38, 0.01):
        params = Phi4Params(kappa=kappa, lam=lam)
        obs = ensemble_summary(phi, params)
        logw = -phi4_action(phi, params) - kernel - logq
        rows.append(
            {
                "kappa_f": kappa,
                "logw": stabilized_logw_stats(logw),
                "observables": obs,
                "aggregate_abs_rel_error_vs_true": aggregate_abs_rel(obs, true_summary),
            }
        )
    return {
        "scan": rows,
        "best_by_logw_width": min(rows, key=lambda row: row["logw"]["std_logw_centered"]),
        "best_by_observable_error": min(rows, key=lambda row: row["aggregate_abs_rel_error_vs_true"]),
    }


@torch.no_grad()
def evaluate_alpha(alpha: float, checkpoint: Path, phi_f: torch.Tensor, args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(args.device)
    params = Phi4Params(kappa=args.kappa, lam=args.lam)
    gen = torch.Generator(device="cpu").manual_seed(args.seed + int(1000 * alpha) + 17)
    psi, u_true = soft_weighted_block(
        phi_f,
        alpha,
        args.weighted_a,
        args.weighted_b,
        eta=args.eta,
        use_eta_scaling=True,
        generator=gen,
    )
    psi = psi[: args.n_eval].to(device)
    u_true = u_true[: args.n_eval].to(device)
    phi_eval = phi_f[: args.n_eval].to(device)
    true_summary = ensemble_summary(phi_eval.cpu(), params)
    flow, mode = load_flow(checkpoint, device)
    cond = make_conditioning(psi, mode)
    sample_gen = torch.Generator(device=device).manual_seed(args.seed + int(1000 * alpha) + 29)
    u, logq = flow.sample(cond, generator=sample_gen)
    phi = soft_weighted_reconstruct(psi, u, args.weighted_a, args.weighted_b, eta=args.eta, use_eta_scaling=True)
    kernel = soft_weighted_kernel_term(u, alpha)
    logw = -phi4_action(phi, params) - kernel - logq
    obs = ensemble_summary(phi.cpu(), params)
    scan = kappa_scan(phi.cpu(), u.cpu(), logq.cpu(), alpha, true_summary, args.lam)
    u_diag = u_channel_stats(u.cpu(), psi.cpu())
    true_u_diag = u_channel_stats(u_true.cpu(), psi.cpu())
    kernel_cpu = kernel.cpu()
    return {
        "alpha": alpha,
        "checkpoint": str(checkpoint),
        "conditioning_mode": mode,
        "observables": obs,
        "relative_aggregate_error": aggregate_abs_rel(obs, true_summary),
        "kernel_term": {
            "mean": float(kernel_cpu.mean().item()),
            "std": float(kernel_cpu.std(unbiased=False).item()),
            "min": float(kernel_cpu.min().item()),
            "max": float(kernel_cpu.max().item()),
        },
        "logw": stabilized_logw_stats(logw.cpu()),
        "u_diagnostics": u_diag,
        "true_u_diagnostics": true_u_diag,
        "diagnostic_errors": diagnostic_errors(u_diag, true_u_diag),
        "rho_psi_corr": corr(u.cpu()[:, 0], psi.cpu()),
        "kappa_scan": scan,
    }


def write_report(path: Path, summary: dict[str, object]) -> None:
    best = min(summary["results"], key=lambda row: row["logw"]["std_logw_centered"])
    best_kappa = min(summary["results"], key=lambda row: abs(row["kappa_scan"]["best_by_logw_width"]["kappa_f"] - 0.31))
    lines = [
        "# Soft Weighted Blocking Scan",
        "",
        (
            "Soft weighted blocking samples `psi = Z_eta B_w phi + noise` and trains a four-channel "
            "conditional flow for `u=(rho,HL,LH,HH)`. Reconstruction solves the weighted LL system from "
            "`psi+rho` before Haar unblocking."
        ),
        "",
        "## Main Answer",
        "",
        (
            f"Best logw width is `{best['logw']['std_logw_centered']:.6g}` at alpha `{best['alpha']:.6g}`. "
            f"The closest kappa minimum to 0.31 is `{best_kappa['kappa_scan']['best_by_logw_width']['kappa_f']:.6g}` "
            f"at alpha `{best_kappa['alpha']:.6g}`."
        ),
        "",
        "## Baselines",
        "",
        "| model | logw std | kappa_min | aggregate observable error |",
        "|---|---:|---:|---:|",
    ]
    for name, ref in summary["references"].items():
        lines.append(
            f"| {name} | {ref.get('logw_std', float('nan')):.6g} | {ref.get('kappa_min', float('nan')):.6g} | "
            f"{ref.get('aggregate_observable_error', float('nan')):.6g} |"
        )
    lines.extend(
        [
            "",
            "## Alpha Summary",
            "",
            "| alpha | logw std | ESS/N | kappa_min | obs kappa_min | agg err | K mean | K std | rho std err | detail std err | corr(psi^2,d^2) err |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["results"]:
        best_width = row["kappa_scan"]["best_by_logw_width"]
        best_obs = row["kappa_scan"]["best_by_observable_error"]
        err = row["diagnostic_errors"]
        lines.append(
            f"| {row['alpha']:.6g} | {row['logw']['std_logw_centered']:.6g} | {row['logw']['ess_over_n']:.6g} | "
            f"{best_width['kappa_f']:.6g} | {best_obs['kappa_f']:.6g} | {row['relative_aggregate_error']:.6g} | "
            f"{row['kernel_term']['mean']:.6g} | {row['kernel_term']['std']:.6g} | {err['rho_std_error']:.6g} | "
            f"{err['detail_std_error']:.6g} | {err['corr_psi2_d2_error']:.6g} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_outputs(path: Path, summary: dict[str, object]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    rows = summary["results"]
    alpha = [r["alpha"] for r in rows]
    refs = summary["references"]
    with PdfPages(path) as pdf:
        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        axes[0, 0].plot(alpha, [r["logw"]["std_logw_centered"] for r in rows], marker="o")
        axes[0, 0].axhline(refs["soft_haar_alpha_2"]["logw_std"], color="tab:blue", ls="--", label="soft Haar alpha=2")
        axes[0, 0].axhline(refs["hard_physics"]["logw_std"], color="tab:orange", ls="--", label="hard physics")
        axes[0, 0].set_ylabel("std centered logw")
        axes[0, 0].legend(fontsize=8)
        axes[0, 1].plot(alpha, [r["logw"]["ess_over_n"] for r in rows], marker="o")
        axes[0, 1].set_ylabel("ESS/N")
        axes[1, 0].plot(alpha, [r["kappa_scan"]["best_by_logw_width"]["kappa_f"] for r in rows], marker="o")
        axes[1, 0].axhline(0.31, color="k", ls="--")
        axes[1, 0].set_ylabel("kappa_min")
        axes[1, 1].plot(alpha, [r["relative_aggregate_error"] for r in rows], marker="o")
        axes[1, 1].axhline(refs["soft_haar_alpha_2"]["aggregate_observable_error"], color="tab:blue", ls="--")
        axes[1, 1].set_ylabel("aggregate observable error")
        for ax in axes.ravel():
            ax.set_xscale("log")
            ax.set_xlabel("alpha")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(alpha, [r["kernel_term"]["mean"] for r in rows], marker="o")
        axes[0].set_xscale("log")
        axes[0].set_xlabel("alpha")
        axes[0].set_ylabel("K mean")
        axes[1].plot(alpha, [r["kernel_term"]["std"] for r in rows], marker="o")
        axes[1].set_xscale("log")
        axes[1].set_xlabel("alpha")
        axes[1].set_ylabel("K std")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
    phi_f = load_or_generate_fine_configs(
        args.data_path,
        n_configs=args.n_configs,
        fine_size=args.fine_size,
        params=Phi4Params(kappa=args.kappa, lam=args.lam),
        burn_in=200,
        interval=10,
        batch_size=args.batch_size,
        proposal_width=1.0,
        seed=args.seed,
        device=args.device,
    ).float()
    results = []
    for alpha in alphas:
        tag = alpha_tag(alpha)
        mle_tag = f"{tag}_mle"
        rk_tag = f"{tag}_reverse_kl"
        mle_ckpt = args.output_dir / f"conditional_detail_flow_{mle_tag}.pt"
        rk_ckpt = args.output_dir / f"conditional_detail_flow_{rk_tag}.pt"
        common = [
            sys.executable,
            "-B",
            "inverse_blocking_flow/train_conditional_flow.py",
            "--fine-size",
            str(args.fine_size),
            "--kappa-fine",
            str(args.kappa),
            "--lambda",
            str(args.lam),
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
            "--blocking-mode",
            "soft_weighted",
            "--soft-alpha",
            str(alpha),
            "--weighted-a",
            str(args.weighted_a),
            "--weighted-b",
            str(args.weighted_b),
            "--eta",
            str(args.eta),
            "--use-eta-scaling",
            "true",
            "--seed",
            str(args.seed),
            "--device",
            args.device,
        ]
        if not (args.reuse and rk_ckpt.exists()):
            if not (args.reuse and mle_ckpt.exists()):
                run(common + ["--mode", "mle", "--epochs", str(args.mle_epochs), "--checkpoint-tag", mle_tag])
            run(
                common
                + [
                    "--mode",
                    "reverse_kl",
                    "--epochs",
                    str(args.reverse_kl_epochs),
                    "--checkpoint",
                    str(mle_ckpt),
                    "--checkpoint-tag",
                    rk_tag,
                ]
            )
        result = evaluate_alpha(alpha, rk_ckpt, phi_f, args)
        print(
            "alpha",
            alpha,
            "logw_std",
            result["logw"]["std_logw_centered"],
            "kappa_min",
            result["kappa_scan"]["best_by_logw_width"]["kappa_f"],
            flush=True,
        )
        results.append(result)

    setup = vars(args).copy()
    setup["data_path"] = str(args.data_path)
    setup["output_dir"] = str(args.output_dir)
    setup["weighted_N"] = float(weighted_kernel_normalization(args.weighted_a, args.weighted_b))
    setup["Delta_phi"] = 0.5 * args.eta
    setup["Z_eta"] = float(eta_scaling_factor(args.eta, use_eta_scaling=True))
    setup["weighted_ll_fft"] = weighted_ll_fft_stats(
        (args.fine_size // 2, args.fine_size // 2),
        args.weighted_a,
        args.weighted_b,
    )
    summary = {
        "setup": setup,
        "references": {
            "soft_haar_alpha_2": {
                "logw_std": 14.65,
                "kappa_min": 0.31,
                "aggregate_observable_error": 0.0293,
            },
            "hard_physics": {"logw_std": 12.56, "kappa_min": 0.26},
            "constrained_weighted_hard": {"logw_std": 12.0, "kappa_min": 0.26},
        },
        "results": results,
    }
    (args.output_dir / "soft_weighted_blocking_scan_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_report(args.output_dir / "soft_weighted_blocking_scan_report.md", summary)
    plot_outputs(args.output_dir / "soft_weighted_blocking_scan_plots.pdf", summary)
    print(f"wrote {args.output_dir / 'soft_weighted_blocking_scan_summary.json'}")
    print(f"wrote {args.output_dir / 'soft_weighted_blocking_scan_report.md'}")
    print(f"wrote {args.output_dir / 'soft_weighted_blocking_scan_plots.pdf'}")


if __name__ == "__main__":
    main()
