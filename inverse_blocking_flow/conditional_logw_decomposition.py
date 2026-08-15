"""Decompose conditional log-weight variance at fixed blocked fields."""

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
from inverse_blocking_flow.haar import (
    average_block,
    reconstruct_from_average_block,
    soft_block,
    soft_kernel_term,
    soft_reconstruct,
    soft_weighted_block,
    soft_weighted_kernel_term,
    soft_weighted_reconstruct,
    weighted_block,
    reconstruct_from_weighted_block,
)
from inverse_blocking_flow.phi4 import Phi4Params, phi4_action


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fine-size", type=int, default=32)
    parser.add_argument("--kappa", type=float, default=0.31)
    parser.add_argument("--lambda-f", dest="lam", type=float, default=1.0)
    parser.add_argument("--n-configs", type=int, default=512)
    parser.add_argument("--n-coarse", type=int, default=256)
    parser.add_argument("--samples-per-coarse", type=int, default=16)
    parser.add_argument("--data-path", type=Path, default=Path("inverse_blocking_flow/outputs/fine_configs.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("inverse_blocking_flow/outputs"))
    parser.add_argument("--hard-checkpoint", type=Path, default=Path("inverse_blocking_flow/outputs/conditional_detail_flow_physics_reverse_kl.pt"))
    parser.add_argument("--soft-checkpoint", type=Path, default=Path("inverse_blocking_flow/outputs/conditional_detail_flow_soft_alpha_2_reverse_kl.pt"))
    parser.add_argument(
        "--soft-weighted-checkpoint",
        type=Path,
        default=Path("inverse_blocking_flow/outputs/conditional_detail_flow_soft_weighted_alpha_5_reverse_kl.pt"),
    )
    parser.add_argument("--batch-coarse", type=int, default=32)
    parser.add_argument("--burn-in", type=int, default=200)
    parser.add_argument("--sample-interval", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--proposal-width", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=949494)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    return parser


def checkpoint_metadata(path: Path) -> dict[str, object]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    meta = state.get("args", {}) if isinstance(state, dict) else {}
    mode = str(state.get("conditioning_mode") or meta.get("conditioning_mode") or "basic")
    blocking_mode = str(state.get("blocking_mode") or meta.get("blocking_mode") or "hard")
    return {
        "state": state,
        "conditioning_mode": mode,
        "blocking_mode": blocking_mode,
        "n_conditioning_channels": int(state.get("n_conditioning_channels") or n_conditioning_channels(mode)),
        "n_detail_channels": int(state.get("n_detail_channels") or (4 if blocking_mode.startswith("soft") else 3)),
        "layers": int(meta.get("layers", 6)),
        "hidden_channels": int(meta.get("hidden_channels", 48)),
        "cnn_depth": int(meta.get("cnn_depth", 3)),
        "soft_alpha": float(state.get("soft_alpha", meta.get("soft_alpha", 1.0))),
        "weighted_a": float(state.get("weighted_a", meta.get("weighted_a", 0.25))),
        "weighted_b": float(state.get("weighted_b", meta.get("weighted_b", 0.0625))),
        "eta": float(state.get("eta", meta.get("eta", 0.25))),
        "use_eta_scaling": bool(state.get("use_eta_scaling", meta.get("use_eta_scaling", True))),
    }


def load_flow(path: Path, device: torch.device) -> tuple[ConditionalDetailFlow, dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"missing checkpoint: {path}")
    meta = checkpoint_metadata(path)
    flow = ConditionalDetailFlow(
        int(meta["layers"]),
        int(meta["hidden_channels"]),
        int(meta["cnn_depth"]),
        int(meta["n_conditioning_channels"]),
        int(meta["n_detail_channels"]),
    ).to(device)
    flow.load_state_dict(meta["state"]["model"])
    flow.eval()
    return flow, meta


def exact_blocked_fields(phi_f: torch.Tensor, meta: dict[str, object], seed: int) -> torch.Tensor:
    mode = str(meta["blocking_mode"])
    if mode == "hard":
        ll, _d = average_block(phi_f)
        return ll
    if mode == "weighted":
        return weighted_block(
            phi_f,
            float(meta["weighted_a"]),
            float(meta["weighted_b"]),
            eta=float(meta["eta"]),
            use_eta_scaling=bool(meta["use_eta_scaling"]),
        )
    gen = torch.Generator(device="cpu").manual_seed(seed)
    if mode == "soft":
        psi, _u = soft_block(phi_f, float(meta["soft_alpha"]), gen)
        return psi
    if mode == "soft_weighted":
        psi, _u = soft_weighted_block(
            phi_f,
            float(meta["soft_alpha"]),
            float(meta["weighted_a"]),
            float(meta["weighted_b"]),
            eta=float(meta["eta"]),
            use_eta_scaling=bool(meta["use_eta_scaling"]),
            generator=gen,
        )
        return psi
    raise ValueError(f"unsupported blocking mode: {mode}")


def reconstruct_and_kernel(c: torch.Tensor, d: torch.Tensor, meta: dict[str, object]) -> tuple[torch.Tensor, torch.Tensor]:
    mode = str(meta["blocking_mode"])
    if mode == "hard":
        phi = reconstruct_from_average_block(c[:, 0], d)
        kernel = torch.zeros(c.shape[0], dtype=phi.dtype, device=phi.device)
        return phi, kernel
    if mode == "weighted":
        phi = reconstruct_from_weighted_block(
            c[:, 0],
            d,
            float(meta["weighted_a"]),
            float(meta["weighted_b"]),
            eta=float(meta["eta"]),
            use_eta_scaling=bool(meta["use_eta_scaling"]),
        )
        kernel = torch.zeros(c.shape[0], dtype=phi.dtype, device=phi.device)
        return phi, kernel
    if mode == "soft":
        phi = soft_reconstruct(c[:, 0], d)
        return phi, soft_kernel_term(d, float(meta["soft_alpha"]))
    if mode == "soft_weighted":
        phi = soft_weighted_reconstruct(
            c[:, 0],
            d,
            float(meta["weighted_a"]),
            float(meta["weighted_b"]),
            eta=float(meta["eta"]),
            use_eta_scaling=bool(meta["use_eta_scaling"]),
        )
        return phi, soft_weighted_kernel_term(d, float(meta["soft_alpha"]))
    raise ValueError(f"unsupported blocking mode: {mode}")


def log_ess_over_n(logw: torch.Tensor) -> float:
    centered = logw.flatten().float() - logw.flatten().float().mean()
    log_norm = torch.logsumexp(centered, dim=0)
    log_ess = 2.0 * log_norm - torch.logsumexp(2.0 * centered, dim=0)
    return float((torch.exp(log_ess) / centered.numel()).item())


def quantiles(x: torch.Tensor) -> dict[str, float]:
    labels = ["q0", "q1", "q5", "q25", "q50", "q75", "q95", "q99", "q100"]
    probs = torch.tensor([0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0], dtype=x.dtype)
    vals = torch.quantile(x.flatten().cpu(), probs)
    return {label: float(v.item()) for label, v in zip(labels, vals)}


@torch.no_grad()
def evaluate_model(
    name: str,
    checkpoint: Path,
    phi_f: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, object]:
    device = torch.device(args.device)
    flow, meta = load_flow(checkpoint, device)
    n_coarse = min(args.n_coarse, phi_f.shape[0])
    phi_f = phi_f[:n_coarse].float()
    c = exact_blocked_fields(phi_f, meta, args.seed + 1009).unsqueeze(1)
    ell_rows = []
    params = Phi4Params(kappa=args.kappa, lam=args.lam)
    sample_seed = args.seed + 7919
    for start in range(0, n_coarse, args.batch_coarse):
        c_batch = c[start : start + args.batch_coarse].to(device)
        repeated_c = c_batch.repeat_interleave(args.samples_per_coarse, dim=0)
        cond = make_conditioning(repeated_c, str(meta["conditioning_mode"]))
        generator = torch.Generator(device=device).manual_seed(sample_seed + start)
        d, logq = flow.sample(cond, generator=generator)
        phi, kernel = reconstruct_and_kernel(repeated_c, d, meta)
        ell = -phi4_action(phi, params) - kernel - logq
        ell_rows.append(ell.reshape(c_batch.shape[0], args.samples_per_coarse).cpu())
    ell_ij = torch.cat(ell_rows, dim=0).float()
    per_coarse_mean = ell_ij.mean(dim=1)
    per_coarse_std = ell_ij.std(dim=1, unbiased=False)
    total_var = ell_ij.flatten().var(unbiased=False)
    between_var = per_coarse_mean.var(unbiased=False)
    within_var = per_coarse_std.square().mean()
    explained = between_var + within_var
    ell_cond = ell_ij - per_coarse_mean[:, None]
    return {
        "name": name,
        "checkpoint": str(checkpoint),
        "blocking_mode": str(meta["blocking_mode"]),
        "conditioning_mode": str(meta["conditioning_mode"]),
        "n_coarse": int(ell_ij.shape[0]),
        "samples_per_coarse": int(ell_ij.shape[1]),
        "kernel_term_included": str(meta["blocking_mode"]).startswith("soft"),
        "soft_alpha": float(meta["soft_alpha"]),
        "weighted_a": float(meta["weighted_a"]),
        "weighted_b": float(meta["weighted_b"]),
        "eta": float(meta["eta"]),
        "total_var": float(total_var.item()),
        "between_coarse_var": float(between_var.item()),
        "within_coarse_var": float(within_var.item()),
        "variance_closure_error": float((total_var - explained).item()),
        "total_std": float(total_var.sqrt().item()),
        "between_coarse_std": float(between_var.sqrt().item()),
        "within_coarse_std": float(within_var.sqrt().item()),
        "between_fraction": float((between_var / total_var.clamp_min(1e-12)).item()),
        "within_fraction": float((within_var / total_var.clamp_min(1e-12)).item()),
        "centered_conditional": {
            "std": float(ell_cond.flatten().std(unbiased=False).item()),
            "ess_over_n": log_ess_over_n(ell_cond),
            "quantiles": quantiles(ell_cond),
        },
        "per_coarse_mean_quantiles": quantiles(per_coarse_mean),
        "per_coarse_std_quantiles": quantiles(per_coarse_std),
    }


def write_report(path: Path, summary: dict[str, object]) -> None:
    rows = summary["models"]
    dominant = max(rows, key=lambda row: row["within_fraction"])
    lines = [
        "# Conditional Logweight Decomposition",
        "",
        (
            "For soft blocking modes, the reported logweight includes the auxiliary kernel term "
            "`-alpha sum rho^2`, matching the target used during reverse-KL training."
        ),
        "",
        "## Main Answer",
        "",
        (
            f"The largest within-coarse fraction is `{dominant['within_fraction']:.6g}` for `{dominant['name']}`. "
            "Values near one mean the fluctuations are mostly fixed-coarse conditional fluctuations that a "
            "coarse marginal model cannot cancel."
        ),
        "",
        "## Summary",
        "",
        "| model | mode | total std | between std | within std | between frac | within frac | cond std | cond ESS/N | closure err |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        cond = row["centered_conditional"]
        lines.append(
            f"| {row['name']} | {row['blocking_mode']} | {row['total_std']:.6g} | {row['between_coarse_std']:.6g} | "
            f"{row['within_coarse_std']:.6g} | {row['between_fraction']:.6g} | {row['within_fraction']:.6g} | "
            f"{cond['std']:.6g} | {cond['ess_over_n']:.6g} | {row['variance_closure_error']:.6g} |"
        )
    lines.extend(["", "## Centered Conditional Quantiles", ""])
    for row in rows:
        q = row["centered_conditional"]["quantiles"]
        lines.append(
            f"- `{row['name']}`: q01={q['q1']:.6g}, q05={q['q5']:.6g}, q50={q['q50']:.6g}, "
            f"q95={q['q95']:.6g}, q99={q['q99']:.6g}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_outputs(path: Path, summary: dict[str, object]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    rows = summary["models"]
    names = [row["name"] for row in rows]
    x = torch.arange(len(rows)).numpy()
    with PdfPages(path) as pdf:
        fig, ax = plt.subplots(figsize=(8.0, 4.8))
        ax.bar(x - 0.18, [row["between_fraction"] for row in rows], width=0.36, label="between coarse")
        ax.bar(x + 0.18, [row["within_fraction"] for row in rows], width=0.36, label="within coarse")
        ax.set_xticks(x, names, rotation=20, ha="right")
        ax.set_ylabel("fraction of total variance")
        ax.legend()
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.0, 4.8))
        ax.plot(names, [row["total_std"] for row in rows], marker="o", label="total")
        ax.plot(names, [row["between_coarse_std"] for row in rows], marker="o", label="between")
        ax.plot(names, [row["within_coarse_std"] for row in rows], marker="o", label="within")
        ax.tick_params(axis="x", rotation=20)
        ax.set_ylabel("logweight std")
        ax.legend()
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.0, 4.8))
        ax.bar(names, [row["centered_conditional"]["ess_over_n"] for row in rows])
        ax.tick_params(axis="x", rotation=20)
        ax.set_ylabel("centered conditional ESS/N")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    phi_f = load_or_generate_fine_configs(
        args.data_path,
        n_configs=args.n_configs,
        fine_size=args.fine_size,
        params=Phi4Params(kappa=args.kappa, lam=args.lam),
        burn_in=args.burn_in,
        interval=args.sample_interval,
        batch_size=args.batch_size,
        proposal_width=args.proposal_width,
        seed=args.seed,
        device=args.device,
    )
    specs = [
        ("hard_physics", args.hard_checkpoint),
        ("soft_haar_alpha_2", args.soft_checkpoint),
        ("soft_weighted_best", args.soft_weighted_checkpoint),
    ]
    models = [evaluate_model(name, checkpoint, phi_f, args) for name, checkpoint in specs]
    setup = vars(args).copy()
    for key, value in list(setup.items()):
        if isinstance(value, Path):
            setup[key] = str(value)
    summary = {"setup": setup, "models": models}
    summary_path = args.output_dir / "conditional_logw_decomposition_summary.json"
    report_path = args.output_dir / "conditional_logw_decomposition_report.md"
    plots_path = args.output_dir / "conditional_logw_decomposition_plots.pdf"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_report(report_path, summary)
    plot_outputs(plots_path, summary)
    print(f"wrote {summary_path}")
    print(f"wrote {report_path}")
    print(f"wrote {plots_path}")


if __name__ == "__main__":
    main()
