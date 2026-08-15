"""Exact coarse-patch A/R diagnostic with soft variables u held fixed."""

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

from inverse_blocking_flow.fixed_flow_patch_inner_mcmc_ar_pilot import inner_patch_mcmc, load_flow, quantiles, sample_u
from inverse_blocking_flow.flow import make_conditioning
from inverse_blocking_flow.haar import soft_block, soft_kernel_term, soft_reconstruct
from inverse_blocking_flow.kappac030_trainable_kappaf_upscale import load_or_generate_coarse
from inverse_blocking_flow.patch_promote_ar_transport_benchmark import (
    autocorr_summary,
    bootstrap_observables,
    component_block_stats,
    correlation,
    cost_units,
    load_cluster_or_reference,
    split_component_summary,
)
from inverse_blocking_flow.phi4 import Phi4Params, phi4_action


SETUPS = [
    {"name": "target_4x4_h200", "kappa_f": 0.320, "patch_size": 4, "sigma_psi": 0.14, "n_inner_hits": 200},
    {"name": "target_2x2_h200", "kappa_f": 0.320, "patch_size": 2, "sigma_psi": 0.18, "n_inner_hits": 200},
    {"name": "prod_4x4_h200", "kappa_f": 0.3288467228412628, "patch_size": 4, "sigma_psi": 0.14, "n_inner_hits": 200},
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("inverse_blocking_flow/outputs_fine16/production_upscaler_kappac030_softalpha2.pt"))
    parser.add_argument("--metadata", type=Path, default=Path("inverse_blocking_flow/outputs_fine16/production_upscaler_kappac030_softalpha2_metadata.json"))
    parser.add_argument("--coarse-data-path", type=Path, default=Path("inverse_blocking_flow/outputs_fine16/production_coarse_kappac030_configs.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("inverse_blocking_flow/outputs_fine16"))
    parser.add_argument("--n-configs", type=int, default=128)
    parser.add_argument("--n-attempts-per-config", type=int, default=1000)
    parser.add_argument("--measure-every", type=int, default=20)
    parser.add_argument("--diagnostic-bootstrap", type=int, default=48)
    parser.add_argument("--include-equilibrium-start", action="store_true", default=True)
    parser.add_argument("--reference-burn-in", type=int, default=400)
    parser.add_argument("--sample-interval", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--proposal-width", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=919191)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--smoke", action="store_true")
    return parser


@torch.no_grad()
def initial_from_upscaler(flow, psi0: torch.Tensor, metadata: dict[str, object], generator: torch.Generator):
    u, _ = sample_u(flow, psi0, str(metadata["conditioning_mode"]), generator)
    phi = soft_reconstruct(psi0[:, 0], u)
    return psi0.clone(), u, phi


@torch.no_grad()
def initial_from_equilibrium(kappa_f: float, metadata: dict[str, object], args: argparse.Namespace, generator: torch.Generator, device: torch.device):
    ref = load_cluster_or_reference(args, 0.320 if abs(kappa_f - 0.320) < 1e-8 else 0.330)[: args.n_configs].to(device)
    psi_3d, u = soft_block(ref, float(metadata["soft_alpha"]), generator=generator)
    psi = psi_3d.unsqueeze(1).float()
    phi = soft_reconstruct(psi[:, 0], u)
    return psi, u, phi


def obs_keys() -> list[str]:
    return ["S_density", "phi2", "phi4", "NN", "chi", "Binder", "xi_2nd_over_L", "P10", "P01", "P11"]


@torch.no_grad()
def run_chain(flow, psi0: torch.Tensor, metadata: dict[str, object], setup: dict[str, object], args: argparse.Namespace, start_mode: str, generator: torch.Generator, device: torch.device) -> dict[str, object]:
    params_c = Phi4Params(float(metadata["kappa_c"]), float(metadata["lambda"]))
    params_f = Phi4Params(float(setup["kappa_f"]), float(metadata["lambda"]))
    alpha = float(metadata["soft_alpha"])
    if start_mode == "equilibrium_start":
        psi, u, phi = initial_from_equilibrium(float(setup["kappa_f"]), metadata, args, generator, device)
    else:
        psi, u, phi = initial_from_upscaler(flow, psi0, metadata, generator)
    sf = phi4_action(phi, params_f)
    sc = phi4_action(psi[:, 0], params_c)
    kernel = soft_kernel_term(u, alpha)
    history = [{"attempt": 0, "observables": bootstrap_observables(phi.cpu(), params_f, args.diagnostic_bootstrap, args.seed + 10)}]
    patch_size = int(setup["patch_size"])
    n_inner_hits = int(setup["n_inner_hits"])
    sigma = float(setup["sigma_psi"])
    accepts = 0
    inner_accepts = 0
    inner_props = 0
    accepted_patch_sq = []
    component_rows = []
    for attempt in range(1, args.n_attempts_per_config + 1):
        psi_prop, patch_sq, acc, prop = inner_patch_mcmc(
            psi,
            params_c,
            patch_size=patch_size,
            sigma_psi=sigma,
            n_inner_hits=n_inner_hits,
            generator=generator,
        )
        inner_accepts += acc
        inner_props += prop
        phi_prop = soft_reconstruct(psi_prop[:, 0], u)
        sf_new = phi4_action(phi_prop, params_f)
        sc_new = phi4_action(psi_prop[:, 0], params_c)
        k_new = soft_kernel_term(u, alpha)
        d_sf = sf_new - sf
        d_sc = sc_new - sc
        d_k = k_new - kernel
        loga = -d_sf - d_k + d_sc
        logu = torch.log(torch.rand(loga.shape, dtype=psi.dtype, device=psi.device, generator=generator))
        accept = logu < loga
        accepts += int(accept.sum())
        for j in range(psi.shape[0]):
            component_rows.append(
                {
                    "accepted": float(bool(accept[j])),
                    "minus_Delta_S_f": float((-d_sf[j]).cpu()),
                    "minus_Delta_K_alpha": float((-d_k[j]).cpu()),
                    "plus_Delta_S_c": float(d_sc[j].cpu()),
                    "logq_old_minus_logq_new": 0.0,
                    "total_logA": float(loga[j].cpu()),
                    "Delta_S_f": float(d_sf[j].cpu()),
                    "Delta_K_alpha": float(d_k[j].cpu()),
                    "Delta_S_c": float(d_sc[j].cpu()),
                    "D_patch": float(patch_sq[j].cpu()),
                }
            )
        if accept.any():
            accepted_patch_sq.extend(patch_sq[accept].cpu().tolist())
            psi[accept] = psi_prop[accept]
            phi[accept] = phi_prop[accept]
            sf[accept] = sf_new[accept]
            sc[accept] = sc_new[accept]
            kernel[accept] = k_new[accept]
        if attempt % args.measure_every == 0 or attempt == args.n_attempts_per_config:
            history.append({"attempt": attempt, "observables": bootstrap_observables(phi.cpu(), params_f, args.diagnostic_bootstrap, args.seed + 10 + attempt)})
    total = args.n_attempts_per_config * psi.shape[0]
    ar = accepts / float(total)
    mean_d = float(torch.tensor(accepted_patch_sq).mean()) if accepted_patch_sq else 0.0
    return {
        **setup,
        "start_mode": start_mode,
        "inner_coarse_acceptance": inner_accepts / float(inner_props),
        "acceptance": ar,
        "accepted_D_patch_per_attempt": ar * mean_d,
        "accepted_fine_sites_per_attempt": ar * float((2 * patch_size) ** 2),
        "accepted_patch_displacement": {"mean": mean_d, "quantiles": quantiles(accepted_patch_sq)},
        "logA_component_split": split_component_summary(component_rows),
        "history": history,
        "autocorrelation": {
            key: autocorr_summary([row["observables"]["mean"].get(key) for row in history])
            for key in ["S_density", "phi2", "phi4", "NN", "chi", "Binder", "xi_2nd_over_L", "P10"]
        },
    }


def write_report(path: Path, summary: dict[str, object]) -> None:
    rows = sorted(summary["runs"], key=lambda r: r["accepted_D_patch_per_attempt"], reverse=True)
    lines = [
        "# Coarse Patch U-Fixed A/R",
        "",
        "| rank | start | setup | kappa | patch | A/R | D_patch/attempt | fine sites/attempt | final xi/L | mean -dSf rejected | mean -dK rejected | mean +dSc rejected |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for i, row in enumerate(rows, 1):
        final = row["history"][-1]["observables"]["mean"]
        rej = row["logA_component_split"]["rejected"]["components"]
        lines.append(
            f"| {i} | {row['start_mode']} | {row['name']} | {row['kappa_f']:.6g} | {row['patch_size']} | "
            f"{row['acceptance']:.6g} | {row['accepted_D_patch_per_attempt']:.6g} | {row['accepted_fine_sites_per_attempt']:.6g} | "
            f"{final['xi_2nd_over_L']} | {rej['minus_Delta_S_f']['mean']:.6g} | {rej['minus_Delta_K_alpha']['mean']:.6g} | {rej['plus_Delta_S_c']['mean']:.6g} |"
        )
    lines.extend(["", "No logq ratio is present because u is held fixed."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_outputs(path: Path, summary: dict[str, object]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    rows = summary["runs"]
    labels = [f"{r['start_mode']}\n{r['name']}" for r in rows]
    with PdfPages(path) as pdf:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes[0, 0].bar(labels, [r["acceptance"] for r in rows])
        axes[0, 0].set_ylabel("A/R")
        axes[0, 1].bar(labels, [r["accepted_D_patch_per_attempt"] for r in rows])
        axes[0, 1].set_ylabel("D_patch/attempt")
        axes[1, 0].bar(labels, [r["accepted_fine_sites_per_attempt"] for r in rows])
        axes[1, 0].set_ylabel("fine sites/attempt")
        axes[1, 1].bar(labels, [r["history"][-1]["observables"]["mean"]["xi_2nd_over_L"] or 0.0 for r in rows])
        axes[1, 1].set_ylabel("final xi/L")
        for ax in axes.ravel():
            ax.tick_params(axis="x", rotation=75, labelsize=6)
            ax.grid(alpha=0.2, axis="y")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    if args.smoke:
        args.n_configs = 8
        args.n_attempts_per_config = 4
        args.measure_every = 2
        args.diagnostic_bootstrap = 4
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    metadata = json.loads(args.metadata.read_text())
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    flow = load_flow(checkpoint, metadata, device)
    coarse_args = argparse.Namespace(
        coarse_data_path=args.coarse_data_path,
        coarse_size=8,
        n_configs=args.n_configs,
        kappa_c=0.300,
        lam=1.0,
        burn_in=400,
        sample_interval=10,
        batch_size=64,
        proposal_width=1.0,
        seed=args.seed + 17,
        device=args.device,
    )
    psi0 = load_or_generate_coarse(coarse_args)[: args.n_configs].to(device).unsqueeze(1)
    start_modes = ["upscaled_start", "equilibrium_start"] if args.include_equilibrium_start else ["upscaled_start"]
    runs = []
    for mode_i, mode in enumerate(start_modes):
        for i, setup in enumerate(SETUPS):
            gen = torch.Generator(device=device).manual_seed(args.seed + 10000 * mode_i + 1000 * i)
            row = run_chain(flow, psi0, metadata, setup, args, mode, gen, device)
            print(
                f"{mode} {setup['name']} A={row['acceptance']:.4g} D={row['accepted_D_patch_per_attempt']:.4g} xi={row['history'][-1]['observables']['mean']['xi_2nd_over_L']}",
                flush=True,
            )
            runs.append(row)
    summary = {"setup": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, "runs": runs}
    summary_path = args.output_dir / "coarse_patch_u_fixed_ar_summary.json"
    report_path = args.output_dir / "coarse_patch_u_fixed_ar_report.md"
    plots_path = args.output_dir / "coarse_patch_u_fixed_ar_plots.pdf"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_report(report_path, summary)
    plot_outputs(plots_path, summary)
    print(f"wrote {summary_path}")
    print(f"wrote {report_path}")
    print(f"wrote {plots_path}")


if __name__ == "__main__":
    main()
