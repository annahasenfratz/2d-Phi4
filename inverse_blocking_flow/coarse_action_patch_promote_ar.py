"""Coarse-action patch updates promoted through learned fine upscalers.

This is a diagnostic for the independent coarse-ensemble setup.  Coarse patch
moves are first accepted against the known coarse action.  Accepted coarse
moves are promoted by resampling a halo of soft-Haar variables from a learned
upscaling flow, followed by a fine Metropolis A/R test.

The current coupling flow exposes exact full-field ``log q(u | psi)`` but not
exact local conditional halo densities.  The halo update therefore uses a
spliced full-flow sample and the full-field log-density ratio as a documented
pseudo-MH diagnostic.  A full-u resampling mode would be exact for the proposal
density but less local.
"""

from __future__ import annotations

import argparse
import copy
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
from inverse_blocking_flow.kappac030_trainable_kappaf_upscale import load_or_generate_coarse, tensor_stats
from inverse_blocking_flow.logw_kappa_scan_blocked_coarse import ensemble_summary
from inverse_blocking_flow.phi4 import Phi4Params, checkerboard_metropolis_sweep, phi4_action


MODEL_KEYS = {
    "B": "A_baseline_B",
    "E": "E_annealed",
    "F": "F_freeze_kappa_0p320",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fine-size", type=int, default=16)
    parser.add_argument("--coarse-size", type=int, default=8)
    parser.add_argument("--kappa-c", type=float, default=0.30)
    parser.add_argument("--kappa-f", type=float, default=0.320)
    parser.add_argument("--lambda", dest="lam", type=float, default=1.0)
    parser.add_argument("--soft-alpha", type=float, default=2.0)
    parser.add_argument("--n-configs", type=int, default=512)
    parser.add_argument("--n-chains", type=int, default=64)
    parser.add_argument("--attempts", type=int, nargs="+", default=[100, 500, 1000])
    parser.add_argument("--model-names", nargs="+", default=["B", "E", "F"], choices=sorted(MODEL_KEYS))
    parser.add_argument("--patch-sigma", nargs="+", default=["1:0.6", "1:0.3", "2:0.18", "2:0.10", "4:0.14", "4:0.08"])
    parser.add_argument("--halo", type=int, default=1)
    parser.add_argument("--models-path", type=Path, default=Path("inverse_blocking_flow/outputs_fine16/kappac030_observable_loss_models.pt"))
    parser.add_argument("--coarse-data-path", type=Path, default=Path("inverse_blocking_flow/outputs_fine16/coarse_kappac030_configs.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("inverse_blocking_flow/outputs_fine16"))
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--hidden-channels", type=int, default=48)
    parser.add_argument("--cnn-depth", type=int, default=4)
    parser.add_argument("--conditioning-mode", choices=("physics",), default="physics")
    parser.add_argument("--burn-in", type=int, default=400)
    parser.add_argument("--sample-interval", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--proposal-width", type=float, default=1.0)
    parser.add_argument("--local-mcmc-sweeps", type=int, nargs="+", default=[0, 10, 20, 50])
    parser.add_argument("--seed", type=int, default=5551212)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--smoke", action="store_true")
    return parser


def parse_patch_sigma(items: list[str]) -> list[tuple[int, float]]:
    out = []
    for item in items:
        left, right = item.split(":", 1)
        out.append((int(left), float(right)))
    return out


def load_flow(state: dict[str, torch.Tensor], args: argparse.Namespace, device: torch.device) -> ConditionalDetailFlow:
    flow = ConditionalDetailFlow(args.layers, args.hidden_channels, args.cnn_depth, 6, 4).to(device)
    flow.load_state_dict(state)
    flow.eval()
    return flow


def halo_mask(size: int, y0: int, x0: int, patch_size: int, halo: int, device: torch.device) -> torch.Tensor:
    mask = torch.zeros((size, size), dtype=torch.bool, device=device)
    ys = [(y % size) for y in range(y0 - halo, y0 + patch_size + halo)]
    xs = [(x % size) for x in range(x0 - halo, x0 + patch_size + halo)]
    mask[torch.tensor(ys, device=device)[:, None], torch.tensor(xs, device=device)[None, :]] = True
    return mask.view(1, 1, size, size)


def low_k_powers(phi: torch.Tensor) -> dict[str, float]:
    volume = phi.shape[-2] * phi.shape[-1]
    fft = torch.fft.fftn(phi.detach().float().cpu(), dim=(-2, -1))
    power = (fft.real.square() + fft.imag.square()).mean(dim=0) / volume
    return {"P10": float(power[1, 0]), "P01": float(power[0, 1]), "P11": float(power[1, 1])}


def observables(phi: torch.Tensor, params: Phi4Params) -> dict[str, float]:
    obs = ensemble_summary(phi.detach().cpu(), params)
    core = ensemble_core(phi.detach().cpu())
    out = {
        "S_density": obs["S_mean"] / float(phi.shape[-2] * phi.shape[-1]),
        "phi2": obs["phi2"],
        "NN_corr": obs["NN_corr"],
        "chi": core["chi"],
        "binder": core["binder"],
        "xi_2nd_over_L": core["xi_2nd_over_L"],
    }
    out.update(low_k_powers(phi))
    return out


def stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "std": None, "min": None, "max": None}
    x = torch.tensor(values, dtype=torch.float64)
    return {
        "mean": float(x.mean()),
        "std": float(x.std(unbiased=False)),
        "min": float(x.min()),
        "max": float(x.max()),
    }


@torch.no_grad()
def initial_state(
    flow: ConditionalDetailFlow,
    psi: torch.Tensor,
    args: argparse.Namespace,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    cond = make_conditioning(psi, args.conditioning_mode)
    u, _logq_sample = flow.sample(cond, generator=generator)
    _eta, logq = flow.inverse_logq(u, cond)
    phi = soft_reconstruct(psi[:, 0], u)
    sf = phi4_action(phi, Phi4Params(kappa=args.kappa_f, lam=args.lam))
    sc = phi4_action(psi[:, 0], Phi4Params(kappa=args.kappa_c, lam=args.lam))
    kernel = soft_kernel_term(u, args.soft_alpha)
    return u, phi, sf, sc, kernel, logq


@torch.no_grad()
def run_promoted_chain(
    flow: ConditionalDetailFlow,
    psi_start: torch.Tensor,
    *,
    patch_size: int,
    sigma_psi: float,
    args: argparse.Namespace,
    generator: torch.Generator,
) -> dict[str, object]:
    attempts_requested = sorted(set(args.attempts))
    max_attempts = max(attempts_requested)
    params_c = Phi4Params(kappa=args.kappa_c, lam=args.lam)
    params_f = Phi4Params(kappa=args.kappa_f, lam=args.lam)
    psi = psi_start.clone()
    u, phi, sf, sc, kernel, logq = initial_state(flow, psi, args, generator)
    snapshots = {0: observables(phi, params_f)}
    coarse_accepts = 0
    fine_accepts = 0
    coarse_rejects = 0
    fine_rejects = 0
    accepted_patch_sq_sum = 0.0
    accepted_abs_delta_sum = 0.0
    accepted_sites = 0.0
    delta_sf_values: list[float] = []
    delta_sc_values: list[float] = []
    delta_k_values: list[float] = []
    logq_ratio_values: list[float] = []
    loga_values: list[float] = []

    for attempt in range(1, max_attempts + 1):
        y0 = int(torch.randint(0, args.coarse_size - patch_size + 1, (1,), generator=generator, device=psi.device).item())
        x0 = int(torch.randint(0, args.coarse_size - patch_size + 1, (1,), generator=generator, device=psi.device).item())
        patch = psi[:, :, y0 : y0 + patch_size, x0 : x0 + patch_size]
        delta = sigma_psi * torch.randn(patch.shape, dtype=psi.dtype, device=psi.device, generator=generator)
        psi_prop = psi.clone()
        psi_prop[:, :, y0 : y0 + patch_size, x0 : x0 + patch_size] = patch + delta
        sc_prop = phi4_action(psi_prop[:, 0], params_c)
        loga_c = -(sc_prop - sc)
        log_u_c = torch.log(torch.rand(loga_c.shape, dtype=psi.dtype, device=psi.device, generator=generator))
        coarse_accept = log_u_c < loga_c
        coarse_accepts += int(coarse_accept.sum().item())
        coarse_rejects += int((~coarse_accept).sum().item())
        if coarse_accept.any():
            cond_prop = make_conditioning(psi_prop, args.conditioning_mode)
            u_sample, _ = flow.sample(cond_prop, generator=generator)
            mask = halo_mask(args.coarse_size, y0, x0, patch_size, args.halo, psi.device)
            u_prop = torch.where(mask, u_sample, u)
            _eta_new, logq_forward = flow.inverse_logq(u_prop, cond_prop)
            phi_prop = soft_reconstruct(psi_prop[:, 0], u_prop)
            sf_prop = phi4_action(phi_prop, params_f)
            kernel_prop = soft_kernel_term(u_prop, args.soft_alpha)
            logq_reverse = logq
            logq_ratio = logq_reverse - logq_forward
            loga_f = -(sf_prop - sf) - (kernel_prop - kernel) + (sc_prop - sc) + logq_ratio
            log_u_f = torch.log(torch.rand(loga_f.shape, dtype=psi.dtype, device=psi.device, generator=generator))
            fine_accept = coarse_accept & (log_u_f < loga_f)
            if fine_accept.any():
                move_sq = delta.square().sum(dim=(1, 2, 3))
                move_abs = delta.abs().sum(dim=(1, 2, 3))
                accepted_patch_sq_sum += float(move_sq[fine_accept].sum().item())
                accepted_abs_delta_sum += float(move_abs[fine_accept].sum().item())
                accepted_sites += float(fine_accept.sum().item() * patch_size * patch_size)
                delta_sf_values.extend((sf_prop[fine_accept] - sf[fine_accept]).detach().cpu().tolist())
                delta_sc_values.extend((sc_prop[fine_accept] - sc[fine_accept]).detach().cpu().tolist())
                delta_k_values.extend((kernel_prop[fine_accept] - kernel[fine_accept]).detach().cpu().tolist())
                logq_ratio_values.extend(logq_ratio[fine_accept].detach().cpu().tolist())
                loga_values.extend(loga_f[fine_accept].detach().cpu().tolist())
                psi[fine_accept] = psi_prop[fine_accept]
                u[fine_accept] = u_prop[fine_accept]
                phi[fine_accept] = phi_prop[fine_accept]
                sf[fine_accept] = sf_prop[fine_accept]
                sc[fine_accept] = sc_prop[fine_accept]
                kernel[fine_accept] = kernel_prop[fine_accept]
                logq[fine_accept] = logq_forward[fine_accept]
            fine_accepts += int(fine_accept.sum().item())
            fine_rejects += int((coarse_accept & ~fine_accept).sum().item())
        if attempt in attempts_requested:
            snapshots[attempt] = observables(phi, params_f)

    total = float(max_attempts * psi.shape[0])
    coarse_acc = coarse_accepts / total
    fine_cond = fine_accepts / float(coarse_accepts) if coarse_accepts else 0.0
    total_acc = fine_accepts / total
    return {
        "patch_size": patch_size,
        "sigma_psi": sigma_psi,
        "attempts_per_config": max_attempts,
        "coarse_acceptance": coarse_acc,
        "fine_promotion_acceptance_given_coarse": fine_cond,
        "total_accepted_move_fraction": total_acc,
        "coarse_rejects": coarse_rejects,
        "fine_rejects": fine_rejects,
        "fine_accepts": fine_accepts,
        "D_patch_after_fine_accept": accepted_patch_sq_sum / total,
        "mean_abs_delta_psi_per_accepted_fine_move": accepted_abs_delta_sum / accepted_sites if accepted_sites else 0.0,
        "delta_S_f": stats(delta_sf_values),
        "delta_S_c": stats(delta_sc_values),
        "delta_K": stats(delta_k_values),
        "logq_ratio": stats(logq_ratio_values),
        "logA_fine": stats(loga_values),
        "snapshots": {str(key): value for key, value in snapshots.items()},
    }


@torch.no_grad()
def local_mcmc_comparison(phi0: torch.Tensor, args: argparse.Namespace) -> list[dict[str, object]]:
    params = Phi4Params(kappa=args.kappa_f, lam=args.lam)
    field = phi0.clone().cpu()
    generator = torch.Generator().manual_seed(args.seed + 9090)
    rows = []
    current = 0
    for sweeps in sorted(set(args.local_mcmc_sweeps)):
        for _ in range(sweeps - current):
            checkerboard_metropolis_sweep(field, params, args.proposal_width, generator)
        current = sweeps
        rows.append({"sweeps": sweeps, "observables": observables(field, params)})
    return rows


def write_report(path: Path, summary: dict[str, object]) -> None:
    lines = [
        "# Coarse-Action Patch Promote A/R",
        "",
        "Coarse moves use `S_c(psi; kappa_c=0.30)`, fine moves use `S_f(phi; kappa_f=0.320)` plus the soft-Haar kernel.",
        "",
        "Important implementation note: halo promotion uses spliced full-flow samples and full-field `log q(u|psi)` ratios because the current flow does not expose exact local conditional halo densities. Treat the A/R numbers as a diagnostic for this pseudo-local proposal.",
        "",
        "## Acceptance And Movement",
        "",
        "| model | patch | sigma | coarse A | fine A|coarse | total A | D_patch | mean |dpsi| |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["runs"]:
        lines.append(
            f"| {row['model']} | {row['patch_size']} | {row['sigma_psi']:.6g} | "
            f"{row['coarse_acceptance']:.6g} | {row['fine_promotion_acceptance_given_coarse']:.6g} | "
            f"{row['total_accepted_move_fraction']:.6g} | {row['D_patch_after_fine_accept']:.6g} | "
            f"{row['mean_abs_delta_psi_per_accepted_fine_move']:.6g} |"
        )
    lines.extend(["", "## Main Readout", ""])
    best = max(summary["runs"], key=lambda row: row["D_patch_after_fine_accept"])
    lines.append(
        f"Best movement by `D_patch_after_fine_accept` is `{best['model']}` patch `{best['patch_size']}` "
        f"sigma `{best['sigma_psi']}` with total acceptance `{best['total_accepted_move_fraction']:.6g}`."
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_outputs(path: Path, summary: dict[str, object]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    runs = summary["runs"]
    labels = [f"{r['model']} p{r['patch_size']} s{r['sigma_psi']}" for r in runs]
    with PdfPages(path) as pdf:
        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        for ax, key, title in [
            (axes[0, 0], "coarse_acceptance", "coarse acceptance"),
            (axes[0, 1], "fine_promotion_acceptance_given_coarse", "fine A | coarse"),
            (axes[1, 0], "total_accepted_move_fraction", "total accepted fraction"),
            (axes[1, 1], "D_patch_after_fine_accept", "D_patch after fine A/R"),
        ]:
            ax.bar(range(len(runs)), [r[key] for r in runs])
            ax.set_ylabel(title)
            ax.set_xticks(range(len(runs)), labels, rotation=75, ha="right", fontsize=6)
            ax.grid(alpha=0.2, axis="y")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 5))
        for run in runs:
            xs = sorted(int(k) for k in run["snapshots"] if int(k) > 0)
            ys = [run["snapshots"][str(x)]["xi_2nd_over_L"] for x in xs]
            ax.plot(xs, ys, marker="o", label=f"{run['model']} p{run['patch_size']} s{run['sigma_psi']}")
        ax.axhline(0.36396, color="k", ls="--", lw=1, label="ref 0.320")
        ax.set_xlabel("patch attempts/config")
        ax.set_ylabel("xi_2nd/L")
        ax.legend(fontsize=6, ncol=2)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    if args.smoke:
        args.n_chains = min(args.n_chains, 8)
        args.n_configs = max(args.n_configs, args.n_chains)
        args.attempts = [2, 4]
        args.model_names = args.model_names[:1]
        args.patch_sigma = args.patch_sigma[:2]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    coarse = load_or_generate_coarse(args)[: args.n_chains].to(device).unsqueeze(1)
    bundle = torch.load(args.models_path, map_location=device, weights_only=False)
    runs = []
    initial_phis = {}
    for model_name in args.model_names:
        key = MODEL_KEYS[model_name]
        flow = load_flow(bundle[key]["model"], args, device)
        init_gen = torch.Generator(device=device).manual_seed(args.seed + 100 * (ord(model_name[0]) if model_name else 1))
        u0, phi0, *_ = initial_state(flow, coarse, args, init_gen)
        initial_phis[model_name] = phi0.detach().cpu()
        for patch_size, sigma_psi in parse_patch_sigma(args.patch_sigma):
            gen = torch.Generator(device=device).manual_seed(args.seed + 1000 * patch_size + int(round(10000 * sigma_psi)) + 17 * len(runs))
            row = run_promoted_chain(
                flow,
                coarse,
                patch_size=patch_size,
                sigma_psi=sigma_psi,
                args=args,
                generator=gen,
            )
            row["model"] = model_name
            row["model_key"] = key
            runs.append(row)
            print(
                f"model={model_name} patch={patch_size} sigma={sigma_psi:g} "
                f"coarseA={row['coarse_acceptance']:.4g} fineA={row['fine_promotion_acceptance_given_coarse']:.4g} "
                f"totalA={row['total_accepted_move_fraction']:.4g} D_patch={row['D_patch_after_fine_accept']:.4g}",
                flush=True,
            )
    comparisons = {
        name: {
            "no_update": observables(phi, Phi4Params(kappa=args.kappa_f, lam=args.lam)),
            "ordinary_fine_local_mcmc": local_mcmc_comparison(phi, args),
        }
        for name, phi in initial_phis.items()
    }
    summary = {
        "setup": {
            "coarse_action": {"L": args.coarse_size, "kappa_c": args.kappa_c, "lambda": args.lam},
            "fine_target": {"L": args.fine_size, "kappa_f": args.kappa_f, "lambda": args.lam},
            "soft_alpha": args.soft_alpha,
            "n_chains": args.n_chains,
            "attempts": args.attempts,
            "halo": args.halo,
            "patch_sigma": args.patch_sigma,
            "proposal_density_note": "Halo resampling uses spliced full-flow samples and full-field logq ratios; exact local conditional q(u_H|psi) is not available from this flow.",
        },
        "runs": runs,
        "comparisons": comparisons,
        "autocorrelation_note": "Autocorrelation estimates are not reported in this first pass; snapshots are too sparse for stable integrated autocorrelation estimates.",
    }
    summary_path = args.output_dir / "coarse_action_patch_promote_ar_summary.json"
    report_path = args.output_dir / "coarse_action_patch_promote_ar_report.md"
    plots_path = args.output_dir / "coarse_action_patch_promote_ar_plots.pdf"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_report(report_path, summary)
    plot_outputs(plots_path, summary)
    print(f"wrote {summary_path}")
    print(f"wrote {report_path}")
    print(f"wrote {plots_path}")


if __name__ == "__main__":
    main()
