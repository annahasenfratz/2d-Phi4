"""Long same-eta promoted-patch benchmark with local-MCMC controls."""

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

from inverse_blocking_flow.coarse_patch_eta_fixed_ar import run_chain
from inverse_blocking_flow.kappac030_trainable_kappaf_upscale import load_or_generate_coarse
from inverse_blocking_flow.patch_promote_ar_transport_benchmark import (
    aggregate_errors,
    compare_observables,
    initial_upscaled_phi,
    cost_units,
    load_cluster_or_reference,
    references_for_setup,
    run_local_mcmc,
)
from inverse_blocking_flow.fixed_flow_patch_inner_mcmc_ar_pilot import load_flow


PROMOTED_SETUPS = [
    {"name": "same_eta_4x4_h200", "kappa_f": 0.320000, "patch_size": 4, "sigma_psi": 0.14, "n_inner_hits": 200},
    {"name": "same_eta_2x2_h200", "kappa_f": 0.320000, "patch_size": 2, "sigma_psi": 0.18, "n_inner_hits": 200},
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("inverse_blocking_flow/outputs_fine16/production_upscaler_kappac030_softalpha2.pt"))
    parser.add_argument("--metadata", type=Path, default=Path("inverse_blocking_flow/outputs_fine16/production_upscaler_kappac030_softalpha2_metadata.json"))
    parser.add_argument("--coarse-data-path", type=Path, default=Path("inverse_blocking_flow/outputs_fine16/production_coarse_kappac030_configs.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("inverse_blocking_flow/outputs_fine16"))
    parser.add_argument("--n-configs", type=int, default=64)
    parser.add_argument("--n-promoted-attempts-per-config", type=int, default=2000)
    parser.add_argument("--measure-every", type=int, default=50)
    parser.add_argument("--diagnostic-bootstrap", type=int, default=32)
    parser.add_argument("--reference-burn-in", type=int, default=400)
    parser.add_argument("--sample-interval", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--proposal-width", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=939393)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--smoke", action="store_true")
    return parser


def run_local_control(
    phi0: torch.Tensor,
    kappa: float,
    args: argparse.Namespace,
    *,
    label: str,
    ref: dict[str, object],
) -> dict[str, object]:
    result = run_local_mcmc(phi0, kappa, args)
    result["label"] = label
    result["final_observables"] = result["history"][-1]["observables"]["mean"]
    result["autocorrelation_summary"] = result["autocorrelation"]
    result["final_comparison"] = compare_observables(result["history"][-1]["observables"], ref)
    result["final_aggregate_errors"] = aggregate_errors(result["final_comparison"])
    return result


def write_report(path: Path, summary: dict[str, object]) -> None:
    promoted = summary["promoted_runs"]
    local = summary["local_mcmc"]
    ranked = sorted(
        promoted,
        key=lambda row: (
            row["accepted_D_patch_per_attempt"],
            row["accepted_fine_sites_per_attempt"],
            row["promoted_acceptance"],
        ),
        reverse=True,
    )

    lines = [
        "# Same-Eta Promoted Patch Long Benchmark",
        "",
        "State is represented as `(psi, eta)` and the proposal keeps `eta` fixed.",
        "",
        "## Movement Ranking",
        "",
        "| rank | start | setup | patch | A/R | accepted D_patch/attempt | accepted fine sites/attempt | final xi/L | local err | IR err | total err |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for i, row in enumerate(ranked, 1):
        final = row["history"][-1]
        final_mean = final["observables"]["mean"]
        comp = row["final_aggregate_errors"]
        xi = final_mean.get("xi_2nd_over_L")
        xi_str = "" if xi is None else f"{float(xi):.6g}"
        lines.append(
            f"| {i} | {row['start_mode']} | {row['name']} | {row['patch_size']} | {row['promoted_acceptance']:.6g} | "
            f"{row['accepted_D_patch_per_attempt']:.6g} | {row['accepted_fine_sites_per_attempt']:.6g} | "
            f"{xi_str} | "
            f"{comp['local']:.6g} | {comp['IR']:.6g} | {comp['total']:.6g} |"
        )

    lines.extend(
        [
            "",
            "## Promoted Runs",
            "",
            "| start | setup | kappa | patch | hits | inner A | promoted A/R | accepted coarse sites/attempt | accepted fine sites/attempt | accepted D_patch/attempt | accepted D_site/attempt | final xi/L |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in promoted:
        final = row["history"][-1]["observables"]["mean"]
        lines.append(
            f"| {row['start_mode']} | {row['name']} | {row['kappa_f']:.6g} | {row['patch_size']} | {row['n_inner_hits']} | "
            f"{row['inner_coarse_acceptance']:.6g} | {row['promoted_acceptance']:.6g} | "
            f"{row['accepted_coarse_sites_per_attempt']:.6g} | {row['accepted_fine_sites_per_attempt']:.6g} | "
            f"{row['accepted_D_patch_per_attempt']:.6g} | {row['accepted_D_site_per_attempt']:.6g} | "
            f"{final['xi_2nd_over_L']} |"
        )

    lines.extend(
        [
            "",
            "## LogA Components",
            "",
            "| start | setup | group | n | A/R | mean -dSf | mean -dK | mean +dSc | mean logq ratio | mean logA | corr logA dSf | corr logA dK | corr logA logq | corr logA Dpatch |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(promoted, key=lambda r: (r["start_mode"], r["name"])):
        split = row["logA_component_split"]
        for group in ("accepted", "rejected"):
            part = split.get(group, {})
            comp = part.get("components", {})
            corr = part.get("correlations", {})

            def fmt(value: object) -> str:
                return "" if value is None else f"{float(value):.6g}"

            def comp_mean(key: str) -> object:
                return comp.get(key, {}).get("mean")

            lines.append(
                f"| {row['start_mode']} | {row['name']} | {group} | {part.get('n', 0)} | {row['promoted_acceptance']:.6g} | "
                f"{fmt(comp_mean('minus_Delta_S_f'))} | {fmt(comp_mean('minus_Delta_K_alpha'))} | "
                f"{fmt(comp_mean('plus_Delta_S_c'))} | {fmt(comp_mean('logq_old_minus_logq_new'))} | "
                f"{fmt(comp_mean('total_logA'))} | {fmt(corr.get('corr_logA_Delta_S_f'))} | "
                f"{fmt(corr.get('corr_logA_Delta_K_alpha'))} | {fmt(corr.get('corr_logA_logq_ratio'))} | "
                f"{fmt(corr.get('corr_logA_D_patch'))} |"
            )

    lines.extend(
        [
            "",
            "## Local MCMC Control",
            "",
            "| label | kappa | final xi/L | tau_int(M) | tau_int(absM) | tau_int(M2) | tau_int(S_density) | tau_int(phi2) | tau_int(lowest_momentum_mode) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for label, result in local.items():
        final = result["final_observables"]
        ac = result["autocorrelation_summary"]
        lines.append(
            f"| {label} | {result['kappa_f']:.6g} | {final.get('xi_2nd_over_L')} | "
            f"{ac['M']['tau_int_initial_positive']} | {ac['absM']['tau_int_initial_positive']} | "
            f"{ac['M2']['tau_int_initial_positive']} | {ac['S_density']['tau_int_initial_positive']} | "
            f"{ac['phi2']['tau_int_initial_positive']} | {ac['lowest_momentum_mode']['tau_int_initial_positive']} |"
        )

    lines.extend(
        [
            "",
            "## Main Answers",
            "",
            f"The best transport by accepted movement is `{ranked[0]['start_mode']} / {ranked[0]['name']}`.",
            f"The equilibrium-start promoted chain is `{', '.join(r['name'] for r in promoted if r['start_mode'] == 'equilibrium_start')}`.",
            "The local-MCMC baseline is reported separately and shares the same target kappa.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_outputs(path: Path, summary: dict[str, object]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    promoted = summary["promoted_runs"]
    local = summary["local_mcmc"]

    with PdfPages(path) as pdf:
        labels = [f"{r['start_mode']}\n{r['name']}" for r in promoted]
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes[0, 0].bar(labels, [r["promoted_acceptance"] for r in promoted])
        axes[0, 0].set_ylabel("promoted A/R")
        axes[0, 1].bar(labels, [r["accepted_D_patch_per_attempt"] for r in promoted])
        axes[0, 1].set_ylabel("accepted D_patch/attempt")
        axes[1, 0].bar(labels, [r["accepted_fine_sites_per_attempt"] for r in promoted])
        axes[1, 0].set_ylabel("accepted fine sites/attempt")
        axes[1, 1].bar(labels, [r["history"][-1]["observables"]["mean"]["xi_2nd_over_L"] for r in promoted])
        axes[1, 1].set_ylabel("final xi/L")
        for ax in axes.ravel():
            ax.tick_params(axis="x", rotation=25)
            ax.grid(alpha=0.2, axis="y")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 5))
        for row in promoted:
            xs = [h["attempt"] for h in row["history"]]
            ys = [h["observables"]["mean"]["xi_2nd_over_L"] for h in row["history"]]
            ax.plot(xs, ys, marker="o", ms=2, label=f"{row['start_mode']} {row['name']}")
        ax.set_xlabel("promoted attempts/config")
        ax.set_ylabel("xi_2nd/L")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 5))
        for label, result in local.items():
            xs = [h.get("sweeps", h.get("attempt", 0)) for h in result["history"]]
            ys = [h["observables"]["mean"]["xi_2nd_over_L"] for h in result["history"]]
            ax.plot(xs, ys, marker="o", ms=3, label=label)
        ax.set_xlabel("sweeps")
        ax.set_ylabel("xi_2nd/L")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    if args.smoke:
        args.n_configs = 8
        args.n_promoted_attempts_per_config = 8
        args.measure_every = 2
        args.diagnostic_bootstrap = 8
    args.n_attempts_per_config = args.n_promoted_attempts_per_config
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    metadata = json.loads(args.metadata.read_text())
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    flow = load_flow(checkpoint, metadata, device)
    coarse_args = argparse.Namespace(
        coarse_data_path=args.coarse_data_path,
        coarse_size=int(metadata["coarse_size"]),
        n_configs=args.n_configs,
        kappa_c=float(metadata["kappa_c"]),
        lam=float(metadata["lambda"]),
        burn_in=400,
        sample_interval=10,
        batch_size=args.batch_size,
        proposal_width=1.0,
        seed=args.seed + 17,
        device=args.device,
    )
    psi0 = load_or_generate_coarse(coarse_args)[: args.n_configs].to(device).unsqueeze(1)
    refs = references_for_setup(args, 0.320)

    promoted_runs = []
    for mode_index, start_mode in enumerate(("upscaled_start", "equilibrium_start")):
        for setup_index, setup in enumerate(PROMOTED_SETUPS):
            gen = torch.Generator(device=device).manual_seed(args.seed + 10000 * mode_index + 1000 * setup_index)
            row = run_chain(flow, psi0, metadata, setup, args, start_mode, gen, device)
            row["start_mode"] = start_mode
            row["primary_reference"] = "ref_0.320"
            row["promoted_acceptance"] = row["acceptance"]
            row["accepted_coarse_sites_per_attempt"] = row["promoted_acceptance"] * float(setup["patch_size"] ** 2)
            row["accepted_fine_sites_per_attempt"] = row["promoted_acceptance"] * float((2 * setup["patch_size"]) ** 2)
            row["accepted_D_site_per_attempt"] = row["accepted_D_patch_per_attempt"] / float(setup["patch_size"] ** 2)
            row["final_comparison"] = compare_observables(row["history"][-1]["observables"], refs["ref_0.320"])
            row["final_aggregate_errors"] = aggregate_errors(row["final_comparison"])
            promoted_runs.append(row)
            print(
                f"{start_mode} {setup['name']} A={row['promoted_acceptance']:.4g} "
                f"D={row['accepted_D_patch_per_attempt']:.4g} xi={row['history'][-1]['observables']['mean']['xi_2nd_over_L']}",
                flush=True,
            )

    local_mcmc = {}
    phi_upscaled = initial_upscaled_phi(flow, psi0, metadata, 0.320, args, args.seed + 4242)
    local_mcmc["upscaled_start"] = run_local_control(phi_upscaled, 0.320, args, label="upscaled_start", ref=refs["ref_0.320"])
    ref_phi = load_cluster_or_reference(args, 0.320)[: args.n_configs].to(device)
    local_mcmc["equilibrium_start"] = run_local_control(ref_phi, 0.320, args, label="equilibrium_start", ref=refs["ref_0.320"])
    summary = {
        "setup": {
            "checkpoint": str(args.checkpoint),
            "metadata": str(args.metadata),
            "n_configs": args.n_configs,
            "promoted_attempts_per_config": args.n_promoted_attempts_per_config,
            "measure_every": args.measure_every,
            "cost_note": "Promoted cost units count 200 coarse hits + one flow eval approximated as 256 units + one fine action eval approximated as 256 units. One fine local MCMC sweep is 256 units.",
            "controls": ["same_eta_4x4_h200", "same_eta_2x2_h200", "local_mcmc", "equilibrium_start_continuation"],
        },
        "references": refs,
        "promoted_runs": promoted_runs,
        "local_mcmc": local_mcmc,
        "reference_summary": {
            "kappa_f": 0.320,
            "xi_2nd_over_L": refs["ref_0.320"]["mean"]["xi_2nd_over_L"],
            "chi": refs["ref_0.320"]["mean"]["chi"],
            "Binder": refs["ref_0.320"]["mean"]["Binder"],
        },
        "cost_units": cost_units(200),
    }

    summary_path = args.output_dir / "same_eta_promoted_patch_long_benchmark_summary.json"
    report_path = args.output_dir / "same_eta_promoted_patch_long_benchmark_report.md"
    plots_path = args.output_dir / "same_eta_promoted_patch_long_benchmark_plots.pdf"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_report(report_path, summary)
    plot_outputs(plots_path, summary)
    print(f"wrote {summary_path}")
    print(f"wrote {report_path}")
    print(f"wrote {plots_path}")


if __name__ == "__main__":
    main()
