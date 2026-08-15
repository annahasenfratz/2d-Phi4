"""Controlled scan of patchwise detail A/R settings."""

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

from inverse_blocking_flow.data import load_or_generate_fine_configs, make_paired_dataset
from inverse_blocking_flow.flow import ConditionalDetailFlow
from inverse_blocking_flow.haar import reconstruct_from_average_block
from inverse_blocking_flow.patchwise_ar import ensemble_summary, run_patchwise_ar
from inverse_blocking_flow.phi4 import Phi4Params


OBS_KEYS = [
    ("S_mean", ("S_f", "mean")),
    ("S_std", ("S_f", "std")),
    ("phi2", ("mean_phi2",)),
    ("Binder", ("binder",)),
    ("NN_corr", ("nearest_neighbor_correlator",)),
    ("susceptibility", ("susceptibility",)),
    ("low_p_power", ("low_momentum_power",)),
    ("high_p_power", ("high_momentum_power",)),
]

AGG_KEYS = ["S_mean", "S_std", "phi2", "NN_corr", "high_p_power"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fine-size", type=int, default=16)
    parser.add_argument("--lambda", dest="lam", type=float, default=1.0)
    parser.add_argument("--kappa-fine", type=float, default=0.31)
    parser.add_argument("--n-configs", type=int, default=512)
    parser.add_argument("--n-chains", type=int, default=64)
    parser.add_argument("--patch-sizes", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--step-sizes", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.4])
    parser.add_argument("--n-sweeps-grid", type=int, nargs="+", default=[50, 100, 300])
    parser.add_argument("--data-path", type=Path, default=Path("inverse_blocking_flow/outputs_fine16/fine_configs.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("inverse_blocking_flow/outputs_fine16"))
    parser.add_argument("--checkpoint", type=Path, default=Path("inverse_blocking_flow/outputs_fine16/conditional_detail_flow_reverse_kl.pt"))
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--cnn-depth", type=int, default=3)
    parser.add_argument("--burn-in", type=int, default=100)
    parser.add_argument("--sample-interval", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--proposal-width", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=424242)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    return parser


def get_value(summary: dict[str, object], path: tuple[str, ...]) -> float:
    value: object = summary
    for key in path:
        value = value[key]  # type: ignore[index]
    return float(value)


def rel_error(value: float, true_value: float) -> float:
    if abs(true_value) < 1e-14:
        return float("nan")
    return (value - true_value) / true_value


def load_flow(args: argparse.Namespace, device: torch.device) -> ConditionalDetailFlow:
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"missing checkpoint: {args.checkpoint}")
    flow = ConditionalDetailFlow(args.layers, args.hidden_channels, args.cnn_depth).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    flow.load_state_dict(state["model"])
    flow.eval()
    return flow


def row_to_md(row: dict[str, object]) -> str:
    fields = [
        "patch_size",
        "step_size",
        "n_sweeps",
        "patch_acceptance_rate",
        "aggregate_abs_rel_error",
        "S_mean",
        "S_std",
        "phi2",
        "Binder",
        "NN_corr",
        "susceptibility",
        "low_p_power",
        "high_p_power",
    ]
    return "| " + " | ".join(format_cell(row[field]) for field in fields) + " |"


def format_cell(value: object) -> str:
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_markdown(path: Path, true_summary: dict[str, object], rows: list[dict[str, object]], best: dict[str, object]) -> None:
    lines = [
        "# Patchwise A/R Scan",
        "",
        "All Binder cumulants and susceptibilities in this scan are computed from the full reconstructed fine field, not from blocked/coarse fields.",
        "",
        "Aggregate error is the mean absolute relative error over `S_mean`, `S_std`, `phi2`, `NN_corr`, and `high_p_power`.",
        "",
        "## True Fine Reference",
        "",
        "| observable | value |",
        "|---|---:|",
    ]
    for label, path_keys in OBS_KEYS:
        lines.append(f"| {label} | {get_value(true_summary, path_keys):.6g} |")

    lines.extend(
        [
            "",
            "## Best Run",
            "",
            f"- patch_size: `{best['patch_size']}`",
            f"- step_size: `{best['step_size']}`",
            f"- n_sweeps: `{best['n_sweeps']}`",
            f"- patch acceptance: `{best['patch_acceptance_rate']:.6g}`",
            f"- aggregate abs relative error: `{best['aggregate_abs_rel_error']:.6g}`",
            "",
            "## Scan Table",
            "",
            "| patch_size | step_size | n_sweeps | accept | agg abs rel err | S mean | S std | phi2 | Binder | NN corr | susceptibility | low-p | high-p |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    lines.extend(row_to_md(row) for row in rows)
    path.write_text("\n".join(lines) + "\n")


def plot_scan(path: Path, rows: list[dict[str, object]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = list(range(len(rows)))
    labels = [f"p{r['patch_size']}/s{r['step_size']}/n{r['n_sweeps']}" for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.5))

    axes[0, 0].plot(x, [r["aggregate_abs_rel_error"] for r in rows], marker="o", ms=3)
    axes[0, 0].set_ylabel("aggregate abs rel error")

    axes[0, 1].plot(x, [r["patch_acceptance_rate"] for r in rows], marker="o", ms=3)
    axes[0, 1].set_ylabel("patch acceptance")

    axes[1, 0].plot(x, [r["S_mean_rel_error"] for r in rows], marker="o", ms=3, label="S mean")
    axes[1, 0].plot(x, [r["phi2_rel_error"] for r in rows], marker="o", ms=3, label="phi2")
    axes[1, 0].plot(x, [r["NN_corr_rel_error"] for r in rows], marker="o", ms=3, label="NN")
    axes[1, 0].axhline(0.0, color="black", lw=0.8)
    axes[1, 0].set_ylabel("relative error")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].plot(x, [r["high_p_power_rel_error"] for r in rows], marker="o", ms=3, label="high-p")
    axes[1, 1].plot(x, [r["low_p_power_rel_error"] for r in rows], marker="o", ms=3, label="low-p")
    axes[1, 1].axhline(0.0, color="black", lw=0.8)
    axes[1, 1].set_ylabel("power relative error")
    axes[1, 1].legend(fontsize=8)

    for ax in axes.ravel():
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=6)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


@torch.no_grad()
def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    params = Phi4Params(kappa=args.kappa_fine, lam=args.lam)
    base_generator = torch.Generator(device=device).manual_seed(args.seed)

    phi_f = load_or_generate_fine_configs(
        args.data_path,
        n_configs=max(args.n_configs, args.n_chains),
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
    idx = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(args.seed))[: args.n_chains]
    phi_c = phi_c_all[idx].to(device)
    true_phi = true_phi_all[idx].cpu()
    true_summary = ensemble_summary(true_phi, params)

    flow = load_flow(args, device)
    d0, _, _, _ = flow.sample_with_decomposition(phi_c, generator=base_generator)

    true_values = {label: get_value(true_summary, path_keys) for label, path_keys in OBS_KEYS}
    rows = []
    run_index = 0
    for patch_size in args.patch_sizes:
        for step_size in args.step_sizes:
            for n_sweeps in args.n_sweeps_grid:
                generator = torch.Generator(device=device).manual_seed(args.seed + 1000 + run_index)
                d_after, patch_stats, _ = run_patchwise_ar(
                    phi_c,
                    d0.clone(),
                    params,
                    patch_size=patch_size,
                    step_size=step_size,
                    n_sweeps=n_sweeps,
                    generator=generator,
                )
                phi_after = reconstruct_from_average_block(phi_c[:, 0], d_after).cpu()
                summary = ensemble_summary(phi_after, params)
                row: dict[str, object] = {
                    "patch_size": patch_size,
                    "step_size": step_size,
                    "n_sweeps": n_sweeps,
                    "patch_acceptance_rate": patch_stats["patch_acceptance_rate"],
                }
                abs_rel = []
                for label, path_keys in OBS_KEYS:
                    value = get_value(summary, path_keys)
                    rel = rel_error(value, true_values[label])
                    row[label] = value
                    row[f"{label}_rel_error"] = rel
                    if label in AGG_KEYS:
                        abs_rel.append(abs(rel))
                row["aggregate_abs_rel_error"] = sum(abs_rel) / len(abs_rel)
                rows.append(row)
                print(
                    f"patch={patch_size} step={step_size:g} sweeps={n_sweeps} "
                    f"accept={patch_stats['patch_acceptance_rate']:.4g} "
                    f"agg={row['aggregate_abs_rel_error']:.4g}"
                )
                run_index += 1

    rows = sorted(rows, key=lambda row: float(row["aggregate_abs_rel_error"]))
    best = rows[0]
    output = {
        "fine_size": args.fine_size,
        "coarse_size": args.fine_size // 2,
        "kappa_fine": args.kappa_fine,
        "lambda": args.lam,
        "checkpoint": str(args.checkpoint),
        "n_chains": args.n_chains,
        "true_reference": true_summary,
        "aggregate_error_observables": AGG_KEYS,
        "binder_susceptibility_note": "Binder and susceptibility are computed from the full fine field, not the blocked/coarse field.",
        "best_run": best,
        "rows": rows,
    }
    summary_path = args.output_dir / "patchwise_scan_summary.json"
    table_path = args.output_dir / "patchwise_scan_table.md"
    plot_path = args.output_dir / "patchwise_scan_plots.pdf"
    summary_path.write_text(json.dumps(output, indent=2) + "\n")
    write_markdown(table_path, true_summary, rows, best)
    plot_scan(plot_path, rows)
    print(f"best patch={best['patch_size']} step={best['step_size']} sweeps={best['n_sweeps']} agg={best['aggregate_abs_rel_error']:.6g}")
    print(f"wrote {summary_path}")
    print(f"wrote {table_path}")
    print(f"wrote {plot_path}")


if __name__ == "__main__":
    main()
