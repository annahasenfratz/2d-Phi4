"""Pilot coarse-patch inner-MCMC promotion with the fixed production upscaler."""

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

from inverse_blocking_flow.correlation_length_bootstrap import ensemble_core
from inverse_blocking_flow.flow import ConditionalDetailFlow, make_conditioning
from inverse_blocking_flow.haar import soft_kernel_term, soft_reconstruct
from inverse_blocking_flow.kappac030_trainable_kappaf_upscale import load_or_generate_coarse
from inverse_blocking_flow.logw_kappa_scan_blocked_coarse import ensemble_summary
from inverse_blocking_flow.phi4 import Phi4Params, phi4_action


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("inverse_blocking_flow/outputs_fine16/production_upscaler_kappac030_softalpha2.pt"))
    parser.add_argument("--metadata", type=Path, default=Path("inverse_blocking_flow/outputs_fine16/production_upscaler_kappac030_softalpha2_metadata.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("inverse_blocking_flow/outputs_fine16"))
    parser.add_argument("--coarse-data-path", type=Path, default=Path("inverse_blocking_flow/outputs_fine16/production_coarse_kappac030_configs.pt"))
    parser.add_argument("--n-configs", type=int, default=64)
    parser.add_argument("--n-promoted-attempts-per-config", type=int, default=50)
    parser.add_argument("--patch-sigmas", nargs="+", default=["2:0.18", "4:0.14"])
    parser.add_argument("--inner-hits", type=int, nargs="+", default=[10, 50, 200])
    parser.add_argument("--target-kappas", nargs="+", default=["metadata", "0.320"])
    parser.add_argument("--coarse-proposal-width", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=707070)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    return parser


def parse_patch_sigmas(values: list[str]) -> list[tuple[int, float]]:
    out = []
    for value in values:
        left, right = value.split(":", 1)
        out.append((int(left), float(right)))
    return out


def quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"q05": None, "q25": None, "q50": None, "q75": None, "q95": None}
    x = torch.tensor(values, dtype=torch.float64)
    qs = torch.quantile(x, torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95], dtype=torch.float64))
    return {name: float(val) for name, val in zip(("q05", "q25", "q50", "q75", "q95"), qs)}


def stats(values: list[float]) -> dict[str, object]:
    if not values:
        return {"mean": None, "std": None, "min": None, "max": None, "quantiles": quantiles(values)}
    x = torch.tensor(values, dtype=torch.float64)
    return {
        "mean": float(x.mean()),
        "std": float(x.std(unbiased=False)),
        "min": float(x.min()),
        "max": float(x.max()),
        "quantiles": quantiles(values),
    }


def observables(phi: torch.Tensor, params: Phi4Params) -> dict[str, float | None]:
    obs = ensemble_summary(phi.detach().cpu(), params)
    core = ensemble_core(phi.detach().cpu())
    volume = phi.shape[-2] * phi.shape[-1]
    return {
        "S_density": obs["S_mean"] / float(volume),
        "phi2": obs["phi2"],
        "NN_corr": obs["NN_corr"],
        "chi": core["chi"],
        "binder": core["binder"],
        "xi_2nd_over_L": core["xi_2nd_over_L"],
    }


def load_flow(checkpoint: dict[str, object], metadata: dict[str, object], device: torch.device) -> ConditionalDetailFlow:
    arch = metadata["architecture"]
    flow = ConditionalDetailFlow(
        int(arch["layers"]),
        int(arch["hidden_channels"]),
        int(arch["cnn_depth"]),
        6,
        4,
    ).to(device)
    flow.load_state_dict(checkpoint["model"])
    flow.eval()
    return flow


@torch.no_grad()
def sample_u(flow: ConditionalDetailFlow, psi: torch.Tensor, conditioning_mode: str, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    cond = make_conditioning(psi, conditioning_mode)
    u, _ = flow.sample(cond, generator=generator)
    _eta, logq = flow.inverse_logq(u, cond)
    return u, logq


@torch.no_grad()
def inner_patch_mcmc(
    psi: torch.Tensor,
    params_c: Phi4Params,
    *,
    patch_size: int,
    sigma_psi: float,
    n_inner_hits: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    psi_new = psi.clone()
    action = phi4_action(psi_new[:, 0], params_c)
    y0 = int(torch.randint(0, psi.shape[-2] - patch_size + 1, (1,), generator=generator, device=psi.device).item())
    x0 = int(torch.randint(0, psi.shape[-1] - patch_size + 1, (1,), generator=generator, device=psi.device).item())
    accepts = 0
    proposals = 0
    for _ in range(n_inner_hits):
        dy = int(torch.randint(0, patch_size, (1,), generator=generator, device=psi.device).item())
        dx = int(torch.randint(0, patch_size, (1,), generator=generator, device=psi.device).item())
        yy = y0 + dy
        xx = x0 + dx
        proposal = psi_new.clone()
        delta = sigma_psi * torch.randn((psi.shape[0],), dtype=psi.dtype, device=psi.device, generator=generator)
        proposal[:, 0, yy, xx] = proposal[:, 0, yy, xx] + delta
        action_prop = phi4_action(proposal[:, 0], params_c)
        loga = -(action_prop - action)
        logu = torch.log(torch.rand(loga.shape, dtype=psi.dtype, device=psi.device, generator=generator))
        accept = logu < loga
        if accept.any():
            psi_new[accept] = proposal[accept]
            action[accept] = action_prop[accept]
        accepts += int(accept.sum().item())
        proposals += int(accept.numel())
    patch_delta = psi_new[:, :, y0 : y0 + patch_size, x0 : x0 + patch_size] - psi[:, :, y0 : y0 + patch_size, x0 : x0 + patch_size]
    patch_sq = patch_delta.square().sum(dim=(1, 2, 3))
    return psi_new, patch_sq, accepts, proposals


@torch.no_grad()
def run_combo(
    flow: ConditionalDetailFlow,
    psi0: torch.Tensor,
    metadata: dict[str, object],
    *,
    kappa_f: float,
    patch_size: int,
    sigma_psi: float,
    n_inner_hits: int,
    args: argparse.Namespace,
    generator: torch.Generator,
) -> dict[str, object]:
    params_c = Phi4Params(kappa=float(metadata["kappa_c"]), lam=float(metadata["lambda"]))
    params_f = Phi4Params(kappa=kappa_f, lam=float(metadata["lambda"]))
    alpha = float(metadata["soft_alpha"])
    conditioning_mode = str(metadata["conditioning_mode"])

    psi = psi0.clone()
    u, logq = sample_u(flow, psi, conditioning_mode, generator)
    phi = soft_reconstruct(psi[:, 0], u)
    sf = phi4_action(phi, params_f)
    sc = phi4_action(psi[:, 0], params_c)
    kernel = soft_kernel_term(u, alpha)
    start_obs = observables(phi, params_f)

    inner_accepts = 0
    inner_proposals = 0
    fine_accepts = 0
    patch_sq_before: list[float] = []
    patch_sq_accepted: list[float] = []
    loga_values: list[float] = []
    dsf_values: list[float] = []
    dsc_values: list[float] = []
    dk_values: list[float] = []
    logq_ratio_values: list[float] = []

    for _ in range(args.n_promoted_attempts_per_config):
        psi_prop, patch_sq, acc, prop = inner_patch_mcmc(
            psi,
            params_c,
            patch_size=patch_size,
            sigma_psi=sigma_psi,
            n_inner_hits=n_inner_hits,
            generator=generator,
        )
        inner_accepts += acc
        inner_proposals += prop
        patch_sq_before.extend(patch_sq.detach().cpu().tolist())
        u_prop, logq_new = sample_u(flow, psi_prop, conditioning_mode, generator)
        phi_prop = soft_reconstruct(psi_prop[:, 0], u_prop)
        sf_new = phi4_action(phi_prop, params_f)
        sc_new = phi4_action(psi_prop[:, 0], params_c)
        k_new = soft_kernel_term(u_prop, alpha)
        logq_ratio = logq - logq_new
        loga = -(sf_new - sf) - (k_new - kernel) + (sc_new - sc) + logq_ratio
        logu = torch.log(torch.rand(loga.shape, dtype=psi.dtype, device=psi.device, generator=generator))
        accept = logu < loga
        loga_values.extend(loga.detach().cpu().tolist())
        dsf_values.extend((sf_new - sf).detach().cpu().tolist())
        dsc_values.extend((sc_new - sc).detach().cpu().tolist())
        dk_values.extend((k_new - kernel).detach().cpu().tolist())
        logq_ratio_values.extend(logq_ratio.detach().cpu().tolist())
        if accept.any():
            patch_sq_accepted.extend(patch_sq[accept].detach().cpu().tolist())
            psi[accept] = psi_prop[accept]
            u[accept] = u_prop[accept]
            phi[accept] = phi_prop[accept]
            sf[accept] = sf_new[accept]
            sc[accept] = sc_new[accept]
            kernel[accept] = k_new[accept]
            logq[accept] = logq_new[accept]
        fine_accepts += int(accept.sum().item())

    total_promoted = args.n_promoted_attempts_per_config * psi.shape[0]
    promoted_acceptance = fine_accepts / float(total_promoted)
    accepted_patch_mean = stats(patch_sq_accepted)["mean"]
    accepted_patch_mean_float = 0.0 if accepted_patch_mean is None else float(accepted_patch_mean)
    patch_coarse_sites = patch_size * patch_size
    patch_fine_sites = (2 * patch_size) * (2 * patch_size)
    accepted_d_site = accepted_patch_mean_float / float(patch_coarse_sites)
    return {
        "kappa_f": kappa_f,
        "patch_size": patch_size,
        "sigma_psi": sigma_psi,
        "n_inner_hits": n_inner_hits,
        "patch_coarse_sites": patch_coarse_sites,
        "patch_fine_sites": patch_fine_sites,
        "n_configs": int(psi.shape[0]),
        "promoted_attempts_per_config": args.n_promoted_attempts_per_config,
        "inner_coarse_acceptance": inner_accepts / float(inner_proposals),
        "mean_patch_displacement_before_promotion": stats(patch_sq_before),
        "promoted_fine_acceptance": promoted_acceptance,
        "accepted_patch_displacement": stats(patch_sq_accepted),
        "accepted_coarse_sites_per_attempt": promoted_acceptance * float(patch_coarse_sites),
        "accepted_fine_sites_per_attempt": promoted_acceptance * float(patch_fine_sites),
        "accepted_D_patch_per_attempt": promoted_acceptance * accepted_patch_mean_float,
        "accepted_D_site_per_attempt": promoted_acceptance * accepted_d_site,
        "logA": stats(loga_values),
        "Delta_S_f": stats(dsf_values),
        "Delta_S_c": stats(dsc_values),
        "Delta_K": stats(dk_values),
        "logq_old_minus_logq_new": stats(logq_ratio_values),
        "start_observables": start_obs,
        "end_observables": observables(phi, params_f),
    }


def write_report(path: Path, summary: dict[str, object]) -> None:
    ranked = sorted(
        summary["runs"],
        key=lambda row: (
            row["accepted_D_patch_per_attempt"],
            row["accepted_fine_sites_per_attempt"],
            row["promoted_fine_acceptance"],
        ),
        reverse=True,
    )
    lines = [
        "# Fixed Flow Patch Inner-MCMC A/R Pilot",
        "",
        f"Checkpoint: `{summary['setup']['checkpoint']}`",
        "",
        "Rows are ranked by accepted movement, not by A/R acceptance alone.",
        "",
        "## Movement Ranking",
        "",
        "| rank | kappa_f | patch | sigma | inner hits | fine A/R | accepted D/attempt | accepted D/site/attempt | accepted fine sites/attempt | disp accepted |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for i, row in enumerate(ranked, start=1):
        acc = row["accepted_patch_displacement"]["mean"]
        lines.append(
            f"| {i} | {row['kappa_f']:.6g} | {row['patch_size']} | {row['sigma_psi']:.6g} | {row['n_inner_hits']} | "
            f"{row['promoted_fine_acceptance']:.6g} | {row['accepted_D_patch_per_attempt']:.6g} | "
            f"{row['accepted_D_site_per_attempt']:.6g} | {row['accepted_fine_sites_per_attempt']:.6g} | "
            f"{acc if acc is not None else 0:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Full Table",
            "",
            "| kappa_f | patch | coarse sites | fine sites | sigma | inner hits | inner A | fine A/R | accepted coarse sites/attempt | accepted fine sites/attempt | accepted D/attempt | accepted D/site/attempt | disp before | disp accepted | logA mean | logA std |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["runs"]:
        before = row["mean_patch_displacement_before_promotion"]["mean"]
        acc = row["accepted_patch_displacement"]["mean"]
        lines.append(
            f"| {row['kappa_f']:.6g} | {row['patch_size']} | {row['patch_coarse_sites']} | {row['patch_fine_sites']} | "
            f"{row['sigma_psi']:.6g} | {row['n_inner_hits']} | {row['inner_coarse_acceptance']:.6g} | "
            f"{row['promoted_fine_acceptance']:.6g} | {row['accepted_coarse_sites_per_attempt']:.6g} | "
            f"{row['accepted_fine_sites_per_attempt']:.6g} | {row['accepted_D_patch_per_attempt']:.6g} | "
            f"{row['accepted_D_site_per_attempt']:.6g} | {before if before is not None else 0:.6g} | "
            f"{acc if acc is not None else 0:.6g} | {row['logA']['mean']:.6g} | {row['logA']['std']:.6g} |"
        )
    good = [row for row in summary["runs"] if row["promoted_fine_acceptance"] >= 0.05]
    lines.extend(["", "## Stop Condition", ""])
    if good:
        tags = ", ".join(f"k={r['kappa_f']:.3f}/p{r['patch_size']}/h{r['n_inner_hits']}" for r in good)
        lines.append(f"At least one combination exceeded 5% promoted A/R; mark for larger benchmark: {tags}.")
    else:
        lines.append("Promoted A/R was below 5% for all combinations; do not expand the scan yet.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_outputs(path: Path, summary: dict[str, object]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    runs = summary["runs"]
    labels = [f"k{r['kappa_f']:.3f} p{r['patch_size']} h{r['n_inner_hits']}" for r in runs]
    with PdfPages(path) as pdf:
        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        axes[0, 0].bar(range(len(runs)), [r["inner_coarse_acceptance"] for r in runs])
        axes[0, 0].set_ylabel("inner coarse acceptance")
        axes[0, 1].bar(range(len(runs)), [r["promoted_fine_acceptance"] for r in runs])
        axes[0, 1].axhline(0.05, color="k", ls="--", lw=1)
        axes[0, 1].set_ylabel("promoted fine A/R")
        axes[1, 0].bar(range(len(runs)), [r["accepted_D_patch_per_attempt"] for r in runs])
        axes[1, 0].set_ylabel("accepted D_patch/attempt")
        axes[1, 1].bar(range(len(runs)), [r["accepted_fine_sites_per_attempt"] for r in runs])
        axes[1, 1].set_ylabel("accepted fine sites/attempt")
        for ax in axes.ravel():
            ax.set_xticks(range(len(runs)), labels, rotation=75, ha="right", fontsize=6)
            ax.grid(alpha=0.2, axis="y")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
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
        batch_size=64,
        proposal_width=1.0,
        seed=args.seed + 17,
        device=args.device,
    )
    psi0 = load_or_generate_coarse(coarse_args)[: args.n_configs].to(device).unsqueeze(1)
    target_kappas = []
    for value in args.target_kappas:
        target_kappas.append(float(metadata["kappa_f_final"]) if value == "metadata" else float(value))
    runs = []
    combo = 0
    for kappa_f in target_kappas:
        for patch_size, sigma_psi in parse_patch_sigmas(args.patch_sigmas):
            for n_inner_hits in args.inner_hits:
                gen = torch.Generator(device=device).manual_seed(args.seed + 1000 * combo)
                row = run_combo(
                    flow,
                    psi0,
                    metadata,
                    kappa_f=kappa_f,
                    patch_size=patch_size,
                    sigma_psi=sigma_psi,
                    n_inner_hits=n_inner_hits,
                    args=args,
                    generator=gen,
                )
                runs.append(row)
                print(
                    f"kappa={kappa_f:.6g} patch={patch_size} sigma={sigma_psi:g} hits={n_inner_hits} "
                    f"innerA={row['inner_coarse_acceptance']:.4g} fineA={row['promoted_fine_acceptance']:.4g}",
                    flush=True,
                )
                combo += 1
    summary = {
        "setup": {
            "checkpoint": str(args.checkpoint),
            "metadata": str(args.metadata),
            "model": checkpoint.get("default_model", metadata.get("default_model")),
            "n_configs": args.n_configs,
            "promoted_attempts_per_config": args.n_promoted_attempts_per_config,
            "alpha": metadata["soft_alpha"],
        },
        "runs": runs,
    }
    summary_path = args.output_dir / "fixed_flow_patch_inner_mcmc_ar_pilot_summary.json"
    report_path = args.output_dir / "fixed_flow_patch_inner_mcmc_ar_pilot_report.md"
    plots_path = args.output_dir / "fixed_flow_patch_inner_mcmc_ar_pilot_plots.pdf"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_report(report_path, summary)
    plot_outputs(plots_path, summary)
    print(f"wrote {summary_path}")
    print(f"wrote {report_path}")
    print(f"wrote {plots_path}")


if __name__ == "__main__":
    main()
