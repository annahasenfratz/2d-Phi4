"""Tune coarse-field patch proposal width by accepted movement.

The proposal is a symmetric random-walk update on a coarse ``psi`` patch,

    psi'_P = psi_P + sigma_psi * Normal(0, 1),

accepted with the coarse phi4 action.  The scan reports both acceptance and
actual movement, and selects ``sigma_psi`` by maximizing ``D_eff`` or
``D_patch`` with acceptance used only as a guardrail.
"""

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

from inverse_blocking_flow.kappac030_trainable_kappaf_upscale import load_or_generate_coarse
from inverse_blocking_flow.phi4 import Phi4Params, phi4_action


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coarse-size", type=int, default=8)
    parser.add_argument("--kappa-c", type=float, default=0.30)
    parser.add_argument("--lambda", dest="lam", type=float, default=1.0)
    parser.add_argument("--n-configs", type=int, default=512)
    parser.add_argument("--n-chains", type=int, default=256)
    parser.add_argument("--patch-sizes", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument(
        "--sigma-psis",
        type=float,
        nargs="+",
        default=[0.02, 0.04, 0.06, 0.08, 0.1, 0.14, 0.18, 0.24, 0.32, 0.45, 0.6],
    )
    parser.add_argument("--n-sweeps", type=int, default=40)
    parser.add_argument("--selection-metric", choices=("D_eff", "D_patch"), default="D_eff")
    parser.add_argument("--coarse-data-path", type=Path, default=Path("inverse_blocking_flow/outputs_fine16/coarse_kappac030_configs.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("inverse_blocking_flow/outputs_fine16"))
    parser.add_argument("--burn-in", type=int, default=400)
    parser.add_argument("--sample-interval", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--proposal-width", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=24681357)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    return parser


def acceptance_guardrail(patch_size: int) -> tuple[float, float]:
    if patch_size == 1:
        return 0.50, 0.90
    if patch_size == 2:
        return 0.70, 0.95
    if patch_size == 4:
        return 0.50, 0.90
    return 0.50, 0.95


@torch.no_grad()
def measure_patch_sigma(
    psi0: torch.Tensor,
    params: Phi4Params,
    *,
    patch_size: int,
    sigma_psi: float,
    n_sweeps: int,
    generator: torch.Generator,
) -> dict[str, float]:
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    if sigma_psi <= 0.0:
        raise ValueError("sigma_psi must be positive")
    if n_sweeps <= 0:
        raise ValueError("n_sweeps must be positive")

    psi = psi0.clone()
    action = phi4_action(psi, params)
    coarse_y, coarse_x = psi.shape[-2:]
    accepts = 0
    proposals = 0
    prop_ms_sum = 0.0
    prop_patch_sum = 0.0
    acc_ms_sum = 0.0
    acc_patch_sum = 0.0

    for _ in range(n_sweeps):
        for y0 in range(0, coarse_y, patch_size):
            for x0 in range(0, coarse_x, patch_size):
                y1 = min(y0 + patch_size, coarse_y)
                x1 = min(x0 + patch_size, coarse_x)
                touched = float((y1 - y0) * (x1 - x0))
                patch = psi[:, y0:y1, x0:x1]
                delta = sigma_psi * torch.randn(patch.shape, dtype=psi.dtype, device=psi.device, generator=generator)
                move_sq_sum = delta.square().sum(dim=(-2, -1))
                move_ms = move_sq_sum / touched

                psi_new = psi.clone()
                psi_new[:, y0:y1, x0:x1] = patch + delta
                action_new = phi4_action(psi_new, params)
                log_accept = -action_new + action
                log_u = torch.log(torch.rand(log_accept.shape, dtype=psi.dtype, device=psi.device, generator=generator))
                accept = log_u < log_accept

                n_prop = int(accept.numel())
                n_acc = int(accept.sum().item())
                proposals += n_prop
                accepts += n_acc
                prop_ms_sum += float(move_ms.sum().item())
                prop_patch_sum += float(move_sq_sum.sum().item())
                if n_acc:
                    acc_ms_sum += float(move_ms[accept].sum().item())
                    acc_patch_sum += float(move_sq_sum[accept].sum().item())
                    psi[accept] = psi_new[accept]
                    action[accept] = action_new[accept]

    acceptance = accepts / float(proposals)
    ms_prop = prop_ms_sum / float(proposals)
    patch_prop = prop_patch_sum / float(proposals)
    ms_acc = acc_ms_sum / float(accepts) if accepts else 0.0
    patch_acc = acc_patch_sum / float(accepts) if accepts else 0.0
    return {
        "patch_size": float(patch_size),
        "sigma_psi": float(sigma_psi),
        "acceptance_rate": float(acceptance),
        "ms_prop": float(ms_prop),
        "ms_acc": float(ms_acc),
        "D_eff": float(acceptance * ms_acc),
        "patch_sq_prop": float(patch_prop),
        "patch_sq_acc": float(patch_acc),
        "D_patch": float(acceptance * patch_acc),
        "accepted_moves": float(accepts),
        "proposals": float(proposals),
    }


def select_by_movement(rows: list[dict[str, float]], metric: str) -> dict[str, object]:
    by_patch: dict[int, dict[str, object]] = {}
    patch_sizes = sorted({int(row["patch_size"]) for row in rows})
    for patch_size in patch_sizes:
        lo, hi = acceptance_guardrail(patch_size)
        patch_rows = [row for row in rows if int(row["patch_size"]) == patch_size]
        guarded = [row for row in patch_rows if lo <= row["acceptance_rate"] <= hi]
        candidates = guarded if guarded else patch_rows
        best = max(candidates, key=lambda row: row[metric])
        by_patch[patch_size] = {
            "acceptance_guardrail": [lo, hi],
            "selection_metric": metric,
            "used_guardrail": bool(guarded),
            "best": best,
        }
    return by_patch


def write_report(path: Path, summary: dict[str, object]) -> None:
    lines = [
        "# Coarse Patch Sigma Tuning",
        "",
        "Sigma is selected by accepted movement, not by acceptance alone.",
        "",
        "Definitions: `ms_prop` is proposed mean squared move per touched site, `ms_acc` is the same quantity averaged over accepted moves, `D_eff = A * ms_acc`, and `D_patch = A * patch_sq_acc`.",
        "",
        "## Selected Sigmas",
        "",
        "| patch | guardrail | used guardrail | sigma_psi | accept | ms_prop | ms_acc | D_eff | D_patch |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for patch_size, row in summary["selected"].items():
        best = row["best"]
        lo, hi = row["acceptance_guardrail"]
        lines.append(
            f"| {patch_size} | {lo:.2f}-{hi:.2f} | {row['used_guardrail']} | "
            f"{best['sigma_psi']:.6g} | {best['acceptance_rate']:.6g} | {best['ms_prop']:.6g} | "
            f"{best['ms_acc']:.6g} | {best['D_eff']:.6g} | {best['D_patch']:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Full Scan",
            "",
            "| patch | sigma_psi | accept | ms_prop | ms_acc | D_eff | patch_sq_prop | patch_sq_acc | D_patch |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["rows"]:
        lines.append(
            f"| {int(row['patch_size'])} | {row['sigma_psi']:.6g} | {row['acceptance_rate']:.6g} | "
            f"{row['ms_prop']:.6g} | {row['ms_acc']:.6g} | {row['D_eff']:.6g} | "
            f"{row['patch_sq_prop']:.6g} | {row['patch_sq_acc']:.6g} | {row['D_patch']:.6g} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_scan(path: Path, rows: list[dict[str, float]], selected: dict[str, object]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    patch_sizes = sorted({int(row["patch_size"]) for row in rows})
    with PdfPages(path) as pdf:
        fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5))
        panels = [
            ("acceptance_rate", "acceptance"),
            ("ms_acc", "accepted ms move/site"),
            ("D_eff", "D_eff"),
            ("D_patch", "D_patch"),
        ]
        for ax, (key, ylabel) in zip(axes.ravel(), panels):
            for patch_size in patch_sizes:
                patch_rows = [row for row in rows if int(row["patch_size"]) == patch_size]
                x = [row["sigma_psi"] for row in patch_rows]
                y = [row[key] for row in patch_rows]
                ax.plot(x, y, marker="o", ms=3, label=f"{patch_size}x{patch_size}")
                best = selected[str(patch_size)]["best"]
                ax.axvline(best["sigma_psi"], color=ax.lines[-1].get_color(), ls="--", alpha=0.35)
            ax.set_xlabel("sigma_psi")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    params = Phi4Params(kappa=args.kappa_c, lam=args.lam)
    coarse = load_or_generate_coarse(args)[: args.n_chains].to(device)
    rows = []
    run_index = 0
    for patch_size in args.patch_sizes:
        for sigma_psi in args.sigma_psis:
            generator = torch.Generator(device=device).manual_seed(args.seed + 1000 * patch_size + run_index)
            row = measure_patch_sigma(
                coarse,
                params,
                patch_size=patch_size,
                sigma_psi=sigma_psi,
                n_sweeps=args.n_sweeps,
                generator=generator,
            )
            rows.append(row)
            print(
                f"patch={patch_size} sigma_psi={sigma_psi:g} "
                f"A={row['acceptance_rate']:.4g} ms_acc={row['ms_acc']:.4g} "
                f"D_eff={row['D_eff']:.4g} D_patch={row['D_patch']:.4g}",
                flush=True,
            )
            run_index += 1

    selected = select_by_movement(rows, args.selection_metric)
    summary = {
        "setup": {
            "coarse_size": args.coarse_size,
            "kappa_c": args.kappa_c,
            "lambda": args.lam,
            "n_chains": min(args.n_chains, int(coarse.shape[0])),
            "n_sweeps": args.n_sweeps,
            "selection_metric": args.selection_metric,
            "coarse_data_path": str(args.coarse_data_path),
        },
        "rows": rows,
        "selected": {str(key): value for key, value in selected.items()},
    }
    summary_path = args.output_dir / "coarse_patch_sigma_tuning_summary.json"
    report_path = args.output_dir / "coarse_patch_sigma_tuning_report.md"
    plots_path = args.output_dir / "coarse_patch_sigma_tuning_plots.pdf"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_report(report_path, summary)
    plot_scan(plots_path, rows, summary["selected"])
    for patch_size, row in summary["selected"].items():
        best = row["best"]
        print(
            f"selected patch={patch_size} sigma_psi={best['sigma_psi']:.6g} "
            f"A={best['acceptance_rate']:.6g} D_eff={best['D_eff']:.6g} D_patch={best['D_patch']:.6g}"
        )
    print(f"wrote {summary_path}")
    print(f"wrote {report_path}")
    print(f"wrote {plots_path}")


if __name__ == "__main__":
    main()
