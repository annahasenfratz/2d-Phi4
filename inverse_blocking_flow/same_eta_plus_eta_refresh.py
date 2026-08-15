"""Same-eta promoted patch updates with occasional eta refresh moves."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/matplotlib")

from inverse_blocking_flow.fixed_flow_patch_inner_mcmc_ar_pilot import load_flow
from inverse_blocking_flow.flow import make_conditioning
from inverse_blocking_flow.haar import soft_block, soft_kernel_term, soft_reconstruct
from inverse_blocking_flow.kappac030_trainable_kappaf_upscale import load_or_generate_coarse
from inverse_blocking_flow.patch_promote_ar_transport_benchmark import (
    aggregate_errors,
    autocorr_summary,
    bootstrap_observables,
    compare_observables,
    cost_units,
    initial_upscaled_phi,
    load_cluster_or_reference,
    references_for_setup,
    run_local_mcmc,
    split_component_summary,
)
from inverse_blocking_flow.phi4 import Phi4Params, phi4_action


SCHEDULES = [
    {"name": "A_no_eta_refresh", "eta_refresh_every": None, "theta": None},
    {"name": "B_every10_theta005", "eta_refresh_every": 10, "theta": 0.05},
    {"name": "C_every10_theta010", "eta_refresh_every": 10, "theta": 0.10},
    {"name": "D_every10_theta020", "eta_refresh_every": 10, "theta": 0.20},
    {"name": "E_every50_theta010", "eta_refresh_every": 50, "theta": 0.10},
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("inverse_blocking_flow/outputs_fine16/production_upscaler_kappac030_softalpha2.pt"),
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("inverse_blocking_flow/outputs_fine16/production_upscaler_kappac030_softalpha2_metadata.json"),
    )
    parser.add_argument(
        "--coarse-data-path",
        type=Path,
        default=Path("inverse_blocking_flow/outputs_fine16/production_coarse_kappac030_configs.pt"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("inverse_blocking_flow/outputs_fine16"))
    parser.add_argument("--n-configs", type=int, default=64)
    parser.add_argument("--n-promoted-attempts-per-config", type=int, default=1000)
    parser.add_argument("--measure-every", type=int, default=20)
    parser.add_argument("--diagnostic-bootstrap", type=int, default=32)
    parser.add_argument("--reference-burn-in", type=int, default=400)
    parser.add_argument("--sample-interval", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--proposal-width", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=949494)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--smoke", action="store_true")
    return parser


def quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"q01": None, "q05": None, "q50": None, "q95": None, "q99": None}
    x = torch.tensor(values, dtype=torch.float64)
    qs = torch.quantile(x, torch.tensor([0.01, 0.05, 0.5, 0.95, 0.99], dtype=torch.float64))
    return {name: float(val) for name, val in zip(("q01", "q05", "q50", "q95", "q99"), qs)}


def stats(values: list[float]) -> dict[str, object]:
    if not values:
        return {"mean": None, "std": None, "quantiles": quantiles(values)}
    x = torch.tensor(values, dtype=torch.float64)
    return {"mean": float(x.mean()), "std": float(x.std(unbiased=False)), "quantiles": quantiles(values)}


def low_mode(phi: torch.Tensor) -> float:
    phi = phi.detach().float().cpu()
    volume = phi.shape[-2] * phi.shape[-1]
    fft = torch.fft.fftn(phi - phi.mean(dim=(-2, -1), keepdim=True), dim=(-2, -1))
    power = (fft.real.square() + fft.imag.square()).mean(dim=0) / volume
    return float(0.5 * (power[1, 0] + power[0, 1]))


def detailed_observables(phi: torch.Tensor, params: Phi4Params) -> dict[str, float | list[float] | None]:
    phi = phi.detach().float().cpu()
    n, ly, lx = phi.shape
    volume = ly * lx
    mag_density = phi.mean(dim=(-2, -1))
    m_sum = phi.sum(dim=(-2, -1))
    m2 = m_sum.square().mean()
    m4 = m_sum.pow(4).mean()
    phi2_cfg = phi.square().mean(dim=(-2, -1))
    phi4_cfg = phi.pow(4).mean(dim=(-2, -1))
    nn_cfg = 0.5 * (
        (phi * torch.roll(phi, -1, dims=-2)).mean(dim=(-2, -1))
        + (phi * torch.roll(phi, -1, dims=-1)).mean(dim=(-2, -1))
    )
    onsite_cfg = (phi.square() + params.lam * (phi.square() - 1.0).square()).mean(dim=(-2, -1))
    action = phi4_action(phi, params)
    kinetic = -2.0 * params.kappa * nn_cfg
    q2 = phi.square().sum(dim=(-2, -1)).square() / float(volume * volume)
    fft = torch.fft.fftn(phi, dim=(-2, -1))
    power = (fft.real.square() + fft.imag.square()).mean(dim=0) / volume
    p00 = float(power[0, 0])
    p10 = float(power[1, 0])
    p01 = float(power[0, 1])
    p11 = float(power[1, 1])
    f = 0.5 * (power[1, 0] + power[0, 1])
    chi = m2 / float(volume)
    xi = None
    if float(f) > 0.0 and float(chi) > float(f):
        xi = (1.0 / (2.0 * torch.sin(torch.tensor(torch.pi / lx)))) * torch.sqrt(chi / f - 1.0)
    from inverse_blocking_flow.correlation_length_bootstrap import projected_connected_correlator

    ct = projected_connected_correlator(phi)
    c0 = float(ct[0]) if float(ct[0]) != 0.0 else None
    return {
        "S_density": float(action.mean() / float(volume)),
        "phi_mean": float(mag_density.mean()),
        "phi2": float(phi2_cfg.mean()),
        "phi4": float(phi4_cfg.mean()),
        "onsite_potential": float(onsite_cfg.mean()),
        "NN": float(nn_cfg.mean()),
        "kinetic_or_link_term": float(kinetic.mean()),
        "Q2": float(q2.mean()),
        "M": float(m_sum.mean()),
        "absM": float(m_sum.abs().mean()),
        "M2": float(m2),
        "M4": float(m4),
        "chi": float(chi),
        "Binder": float(1.0 - m4 / (3.0 * m2.square().clamp_min(1e-12))),
        "xi_2nd_over_L": None if xi is None else float(xi / lx),
        "C_over_C0": [None if c0 is None else float(x / c0) for x in ct],
        "P00": p00,
        "P10": p10,
        "P01": p01,
        "P11": p11,
        "P10_over_P00": None if abs(p00) < 1e-12 else p10 / p00,
        "P01_over_P00": None if abs(p00) < 1e-12 else p01 / p00,
        "P11_over_P00": None if abs(p00) < 1e-12 else p11 / p00,
        "lowest_momentum_mode": low_mode(phi),
    }


LOCAL_OBS = ["S_density", "phi_mean", "phi2", "phi4", "onsite_potential", "NN", "kinetic_or_link_term", "Q2"]
IR_OBS = ["M", "absM", "M2", "M4", "chi", "Binder", "xi_2nd_over_L", "P00", "P10", "P01", "P11", "P10_over_P00", "P01_over_P00", "P11_over_P00"]


def bootstrap_observables_local(phi: torch.Tensor, params: Phi4Params, n_bootstrap: int, seed: int) -> dict[str, object]:
    estimate = detailed_observables(phi, params)
    gen = torch.Generator().manual_seed(seed)
    n = phi.shape[0]
    sample_keys = LOCAL_OBS + IR_OBS
    samples: dict[str, list[float]] = {key: [] for key in sample_keys}
    for _ in range(n_bootstrap):
        idx = torch.randint(0, n, (n,), generator=gen)
        row = detailed_observables(phi[idx], params)
        for key in sample_keys:
            value = row.get(key)
            if value is not None:
                samples[key].append(float(value))
    boot = {}
    for key, values in samples.items():
        x = torch.tensor(values, dtype=torch.float64)
        boot[key] = {"stderr": float(x.std(unbiased=True)) if x.numel() > 1 else None, "bootstrap_mean": float(x.mean()) if x.numel() else None}
    return {"mean": estimate, "bootstrap": boot}


def all_observables(phi: torch.Tensor, params: Phi4Params) -> dict[str, float]:
    return detailed_observables(phi, params)


def eta_state_from_eta(flow, eta: torch.Tensor, psi: torch.Tensor, conditioning_mode: str):
    cond = make_conditioning(psi, conditioning_mode)
    u, logq = flow.sample_and_logq(eta, cond)
    log_prior = flow.standard_normal_logprob(eta)
    logJ = log_prior - logq
    return u, logq, logJ


def eta_refresh(flow, psi: torch.Tensor, eta: torch.Tensor, theta: float, conditioning_mode: str, generator: torch.Generator):
    noise = torch.randn_like(eta, generator=generator)
    eta_prop = math.cos(theta) * eta + math.sin(theta) * noise
    cond = make_conditioning(psi, conditioning_mode)
    u_prop, logq_new = flow.sample_and_logq(eta_prop, cond)
    log_prior_new = flow.standard_normal_logprob(eta_prop)
    logJ_new = log_prior_new - logq_new
    return eta_prop, u_prop, logq_new, logJ_new


def component_stats(values: list[float]) -> dict[str, object]:
    if not values:
        return {"mean": None, "std": None, "quantiles": quantiles(values)}
    x = torch.tensor(values, dtype=torch.float64)
    return {"mean": float(x.mean()), "std": float(x.std(unbiased=False)), "quantiles": quantiles(values)}


def autocorr_proxy(history: list[dict[str, object]], key: str) -> dict[str, float | None]:
    return autocorr_summary([row["observables"]["mean"].get(key) for row in history])


def run_chain_with_refresh(
    flow,
    psi0: torch.Tensor,
    metadata: dict[str, object],
    setup: dict[str, object],
    schedule: dict[str, object],
    args: argparse.Namespace,
    start_mode: str,
    generator: torch.Generator,
    ref: dict[str, object],
    device: torch.device,
) -> dict[str, object]:
    params_c = Phi4Params(float(metadata["kappa_c"]), float(metadata["lambda"]))
    params_f = Phi4Params(float(setup["kappa_f"]), float(metadata["lambda"]))
    alpha = float(metadata["soft_alpha"])
    conditioning_mode = str(metadata["conditioning_mode"])
    patch_size = int(setup["patch_size"])
    sigma = float(setup["sigma_psi"])
    n_inner_hits = int(setup["n_inner_hits"])

    if start_mode == "equilibrium_start":
        ref_phi = load_cluster_or_reference(args, float(setup["kappa_f"]))[: args.n_configs].to(device)
        psi_3d, u = soft_block(ref_phi, alpha, generator=generator)
        psi = psi_3d.unsqueeze(1).float()
        cond = make_conditioning(psi, conditioning_mode)
        eta, logq = flow.inverse_logq(u, cond)
        log_prior = flow.standard_normal_logprob(eta)
        logJ = log_prior - logq
        phi = soft_reconstruct(psi[:, 0], u)
    else:
        psi = psi0.clone()
        eta = torch.randn((psi.shape[0], 4, psi.shape[-2], psi.shape[-1]), dtype=psi.dtype, device=psi.device, generator=generator)
        u, logq = eta_state_from_eta(flow, eta, psi, conditioning_mode)[:2]
        log_prior = flow.standard_normal_logprob(eta)
        logJ = log_prior - logq
        phi = soft_reconstruct(psi[:, 0], u)

    sf = phi4_action(phi, params_f)
    sc = phi4_action(psi[:, 0], params_c)
    kernel = soft_kernel_term(u, alpha)

    obs0 = bootstrap_observables_local(phi.cpu(), params_f, args.diagnostic_bootstrap, args.seed + 50)
    comp0 = compare_observables(obs0, ref)
    history = [
        {
            "attempt": 0,
            "observables": obs0,
            "reference_comparison": comp0,
            "aggregate_errors": aggregate_errors(comp0),
            "state_consistency": {
                "phi_reconstruction_error": 0.0,
                "kernel_consistency_error": 0.0,
            },
        }
    ]

    promoted_accepts = 0
    promoted_total = 0
    eta_accepts = 0
    eta_total = 0
    inner_accepts = 0
    inner_proposals = 0
    accepted_patch_sq: list[float] = []
    promoted_loga: list[float] = []
    promoted_dsf: list[float] = []
    promoted_dsc: list[float] = []
    promoted_dk: list[float] = []
    promoted_logq: list[float] = []
    eta_loga: list[float] = []
    eta_dsf: list[float] = []
    eta_dk: list[float] = []
    eta_dlogj: list[float] = []
    eta_accept_loga: list[float] = []
    eta_refresh_acceptance = 0
    eta_refresh_attempts = 0
    eta_every = schedule["eta_refresh_every"]
    theta = schedule["theta"]
    promoted_component_rows: list[dict[str, float]] = []
    eta_component_rows: list[dict[str, float]] = []

    for attempt in range(1, args.n_promoted_attempts_per_config + 1):
        from inverse_blocking_flow.coarse_patch_eta_fixed_ar import inner_patch_mcmc

        psi_prop, patch_sq, acc, prop = inner_patch_mcmc(
            psi,
            params_c,
            patch_size=patch_size,
            sigma_psi=sigma,
            n_inner_hits=n_inner_hits,
            generator=generator,
        )
        inner_accepts += acc
        inner_proposals += prop

        u_prop, logq_new = eta_state_from_eta(flow, eta, psi_prop, conditioning_mode)[:2]
        cond_prop = make_conditioning(psi_prop, conditioning_mode)
        log_prior_new = flow.standard_normal_logprob(eta)
        logJ_new = log_prior_new - logq_new
        phi_prop = soft_reconstruct(psi_prop[:, 0], u_prop)
        sf_new = phi4_action(phi_prop, params_f)
        sc_new = phi4_action(psi_prop[:, 0], params_c)
        k_new = soft_kernel_term(u_prop, alpha)
        loga = -(sf_new - sf) - (k_new - kernel) + (sc_new - sc) + (logq - logq_new)
        logu = torch.log(torch.rand(loga.shape, dtype=psi.dtype, device=psi.device, generator=generator))
        accept = logu < loga
        promoted_total += int(accept.numel())
        promoted_accepts += int(accept.sum())
        accepted_patch_sq.extend(patch_sq[accept].detach().cpu().tolist())
        promoted_loga.extend(loga.detach().cpu().tolist())
        promoted_dsf.extend((sf_new - sf).detach().cpu().tolist())
        promoted_dsc.extend((sc_new - sc).detach().cpu().tolist())
        promoted_dk.extend((k_new - kernel).detach().cpu().tolist())
        promoted_logq.extend((logq - logq_new).detach().cpu().tolist())
        for j in range(int(accept.numel())):
            promoted_component_rows.append(
                {
                    "accepted": float(bool(accept[j])),
                    "minus_Delta_S_f": float((-(sf_new - sf)[j]).detach().cpu()),
                    "minus_Delta_K_alpha": float((-(k_new - kernel)[j]).detach().cpu()),
                    "plus_Delta_S_c": float((sc_new - sc)[j].detach().cpu()),
                    "logq_old_minus_logq_new": float((logq - logq_new)[j].detach().cpu()),
                    "total_logA": float(loga[j].detach().cpu()),
                    "Delta_S_f": float((sf_new - sf)[j].detach().cpu()),
                    "Delta_K_alpha": float((k_new - kernel)[j].detach().cpu()),
                    "Delta_S_c": float((sc_new - sc)[j].detach().cpu()),
                    "D_patch": float(patch_sq[j].detach().cpu()),
                }
            )
        if accept.any():
            psi[accept] = psi_prop[accept]
            u[accept] = u_prop[accept]
            eta[accept] = eta[accept]
            phi[accept] = phi_prop[accept]
            sf[accept] = sf_new[accept]
            sc[accept] = sc_new[accept]
            kernel[accept] = k_new[accept]
            logq[accept] = logq_new[accept]
            logJ[accept] = logJ_new[accept]

        if eta_every is not None and attempt % int(eta_every) == 0:
            eta_refresh_attempts += int(psi.shape[0])
            eta_prop, u_eta, logq_eta_new, logJ_eta_new = eta_refresh(flow, psi, eta, float(theta), conditioning_mode, generator)
            phi_eta = soft_reconstruct(psi[:, 0], u_eta)
            sf_eta = phi4_action(phi_eta, params_f)
            k_eta = soft_kernel_term(u_eta, alpha)
            d_sf = sf_eta - sf
            d_k = k_eta - kernel
            d_logj = logJ_eta_new - logJ
            loga_eta = -d_sf - d_k + d_logj
            logu_eta = torch.log(torch.rand(loga_eta.shape, dtype=psi.dtype, device=psi.device, generator=generator))
            accept_eta = logu_eta < loga_eta
            eta_accepts += int(accept_eta.sum())
            eta_loga.extend(loga_eta.detach().cpu().tolist())
            eta_dsf.extend(d_sf.detach().cpu().tolist())
            eta_dk.extend(d_k.detach().cpu().tolist())
            eta_dlogj.extend(d_logj.detach().cpu().tolist())
            eta_accept_loga.extend(loga_eta.detach().cpu().tolist()[i] for i, ok in enumerate(accept_eta.detach().cpu().tolist()) if ok)
            for j in range(int(accept_eta.numel())):
                eta_component_rows.append(
                    {
                        "accepted": float(bool(accept_eta[j])),
                        "minus_Delta_S_f": float((-(d_sf)[j]).detach().cpu()),
                        "minus_Delta_K_alpha": float((-(d_k)[j]).detach().cpu()),
                        "plus_Delta_S_c": 0.0,
                        "logq_old_minus_logq_new": float((d_logj)[j].detach().cpu()),
                        "total_logA": float(loga_eta[j].detach().cpu()),
                        "Delta_S_f": float(d_sf[j].detach().cpu()),
                        "Delta_K_alpha": float(d_k[j].detach().cpu()),
                        "Delta_S_c": 0.0,
                        "D_patch": 0.0,
                    }
                )
            if accept_eta.any():
                eta[accept_eta] = eta_prop[accept_eta]
                u[accept_eta] = u_eta[accept_eta]
                phi[accept_eta] = phi_eta[accept_eta]
                sf[accept_eta] = sf_eta[accept_eta]
                kernel[accept_eta] = k_eta[accept_eta]
                logq[accept_eta] = logq_eta_new[accept_eta]
                logJ[accept_eta] = logJ_eta_new[accept_eta]
            eta_total += int(accept_eta.numel())

        if attempt % args.measure_every == 0 or attempt == args.n_promoted_attempts_per_config:
            obs = bootstrap_observables_local(phi.cpu(), params_f, args.diagnostic_bootstrap, args.seed + 500 + attempt)
            comp = compare_observables(obs, ref)
            history.append(
                {
                    "attempt": attempt,
                    "observables": obs,
                    "reference_comparison": comp,
                    "aggregate_errors": aggregate_errors(comp),
                    "state_consistency": {
                        "phi_reconstruction_error": float((phi - soft_reconstruct(psi[:, 0], u)).abs().max().detach().cpu()),
                        "kernel_consistency_error": float((kernel - soft_kernel_term(u, alpha)).abs().max().detach().cpu()),
                    },
                }
            )

    total = args.n_promoted_attempts_per_config * psi.shape[0]
    promoted_ar = promoted_accepts / float(total)
    eta_ar = eta_accepts / float(eta_total) if eta_total else 0.0
    accepted_patch_mean = float(torch.tensor(accepted_patch_sq).mean()) if accepted_patch_sq else 0.0
    final_obs = history[-1]["observables"]["mean"]
    final_comp = history[-1]["reference_comparison"]
    return {
        "start_mode": start_mode,
        "schedule": schedule,
        "kappa_f": float(setup["kappa_f"]),
        "patch_size": patch_size,
        "sigma_psi": sigma,
        "n_inner_hits": n_inner_hits,
        "promoted_acceptance": promoted_ar,
        "eta_refresh_acceptance": eta_ar,
        "eta_refresh_attempts": eta_refresh_attempts,
        "accepted_D_patch_per_attempt": promoted_ar * accepted_patch_mean,
        "accepted_fine_sites_per_attempt": promoted_ar * float((2 * patch_size) ** 2),
        "accepted_D_site_per_attempt": promoted_ar * (accepted_patch_mean / float(patch_size * patch_size)),
        "inner_coarse_acceptance": inner_accepts / float(inner_proposals),
        "logA": component_stats(promoted_loga),
        "Delta_S_f": component_stats(promoted_dsf),
        "Delta_S_c": component_stats(promoted_dsc),
        "Delta_K": component_stats(promoted_dk),
        "logq_old_minus_logq_new": component_stats(promoted_logq),
        "eta_logA": component_stats(eta_loga),
        "eta_Delta_S_f": component_stats(eta_dsf),
        "eta_Delta_K": component_stats(eta_dk),
        "eta_Delta_logJ": component_stats(eta_dlogj),
        "logA_component_split": split_component_summary(promoted_component_rows),
        "eta_logA_component_split": split_component_summary(eta_component_rows),
        "history": history,
        "final_comparison": final_comp,
        "final_aggregate_errors": aggregate_errors(final_comp),
        "autocorrelation": {
            "M": autocorr_proxy(history, "M"),
            "S_density": autocorr_proxy(history, "S_density"),
            "phi2": autocorr_proxy(history, "phi2"),
            "lowest_momentum_mode": autocorr_proxy(history, "lowest_momentum_mode"),
        },
        "final_observables": final_obs,
        "state_consistency": history[-1]["state_consistency"],
    }


def load_control_summaries(args: argparse.Namespace) -> dict[str, object]:
    out: dict[str, object] = {}
    same_eta_path = args.output_dir / "same_eta_promoted_patch_long_benchmark_summary.json"
    if same_eta_path.exists():
        data = json.loads(same_eta_path.read_text())
        out["same_eta_baseline"] = data.get("promoted_runs", [])
        out["local_mcmc"] = data.get("local_mcmc", {})
    restricted_path = args.output_dir / "restricted_detail_mcmc_summary.json"
    if restricted_path.exists():
        data = json.loads(restricted_path.read_text())
        runs = {run["name"]: run for run in data.get("runs", [])}
        out["restricted_detail"] = runs
    return out


def write_report(path: Path, summary: dict[str, object]) -> None:
    runs = summary["runs"]
    local = summary.get("controls", {}).get("local_mcmc", {})
    restricted = summary.get("controls", {}).get("restricted_detail", {})
    ranked = sorted(runs, key=lambda row: row["accepted_D_patch_per_attempt"], reverse=True)

    lines = [
        "# Same-Eta Plus Eta-Refresh Benchmark",
        "",
        "State is represented as `(psi, eta)`. Promoted patch moves keep `eta` fixed; eta-refresh moves update `eta` at fixed `psi`.",
        "",
        "## Promoted Ranking",
        "",
        "| rank | start | schedule | A/R | eta A/R | accepted D_patch/attempt | accepted fine sites/attempt | final xi/L | local err | IR err | total err |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for i, row in enumerate(ranked, 1):
        lines.append(
            f"| {i} | {row['start_mode']} | {row['schedule']['name']} | {row['promoted_acceptance']:.6g} | {row['eta_refresh_acceptance']:.6g} | "
            f"{row['accepted_D_patch_per_attempt']:.6g} | {row['accepted_fine_sites_per_attempt']:.6g} | "
            f"{row['final_observables']['xi_2nd_over_L']} | {row['final_aggregate_errors']['local']:.6g} | "
            f"{row['final_aggregate_errors']['IR']:.6g} | {row['final_aggregate_errors']['total']:.6g} |"
        )

    lines.extend(
        [
            "",
            "## Schedule Table",
            "",
            "| start | schedule | patch | eta every | theta | promoted A/R | eta A/R | accepted D_patch/attempt | xi/L |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in runs:
        lines.append(
            f"| {row['start_mode']} | {row['schedule']['name']} | {row['patch_size']} | {'' if row['schedule']['eta_refresh_every'] is None else row['schedule']['eta_refresh_every']} | "
            f"{'' if row['schedule']['theta'] is None else row['schedule']['theta']} | {row['promoted_acceptance']:.6g} | {row['eta_refresh_acceptance']:.6g} | "
            f"{row['accepted_D_patch_per_attempt']:.6g} | {row['final_observables']['xi_2nd_over_L']} |"
        )

    lines.extend(
        [
            "",
            "## Component Diagnostics",
            "",
            "| start | schedule | group | n | A/R | mean -dSf | mean -dK | mean +dSc | mean logq/logJ term | mean logA |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in runs:
        split = row.get("logA_component_split")
        if split is None:
            split = split_component_summary(
                [
                    {
                        "accepted": float(True),
                        "minus_Delta_S_f": 0.0,
                        "minus_Delta_K_alpha": 0.0,
                        "plus_Delta_S_c": 0.0,
                        "logq_old_minus_logq_new": 0.0,
                        "total_logA": 0.0,
                        "Delta_S_f": 0.0,
                        "Delta_K_alpha": 0.0,
                        "Delta_S_c": 0.0,
                        "D_patch": 0.0,
                    }
                ]
            )
        for group in ("accepted", "rejected"):
            part = split.get(group, {})
            comp = part.get("components", {})

            def fmt(value: object) -> str:
                return "" if value is None else f"{float(value):.6g}"

            def comp_mean(key: str) -> object:
                return comp.get(key, {}).get("mean")

            lines.append(
                f"| {row['start_mode']} | {row['schedule']['name']} | {group} | {part.get('n', 0)} | {row['promoted_acceptance']:.6g} | "
                f"{fmt(comp_mean('minus_Delta_S_f'))} | {fmt(comp_mean('minus_Delta_K_alpha'))} | {fmt(comp_mean('plus_Delta_S_c'))} | "
                f"{fmt(comp_mean('logq_old_minus_logq_new'))} | {fmt(comp_mean('total_logA'))} |"
            )

    lines.extend(
        [
            "",
            "## Eta-Refresh Components",
            "",
            "| start | schedule | group | n | eta A/R | mean -dSf | mean -dK | mean dlogJ | mean logA |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in runs:
        split = row.get("eta_logA_component_split") or split_component_summary([])
        for group in ("accepted", "rejected"):
            part = split.get(group, {})
            comp = part.get("components", {})

            def fmt(value: object) -> str:
                return "" if value is None else f"{float(value):.6g}"

            def comp_mean(key: str) -> object:
                return comp.get(key, {}).get("mean")

            lines.append(
                f"| {row['start_mode']} | {row['schedule']['name']} | {group} | {part.get('n', 0)} | {row['eta_refresh_acceptance']:.6g} | "
                f"{fmt(comp_mean('minus_Delta_S_f'))} | {fmt(comp_mean('minus_Delta_K_alpha'))} | "
                f"{fmt(comp_mean('logq_old_minus_logq_new'))} | {fmt(comp_mean('total_logA'))} |"
            )

    lines.extend(
        [
            "",
            "## Controls",
            "",
            "| control | source | final xi/L | local err | IR err | total err | tau_int(M) | tau_int(S_density) | tau_int(phi2) | tau_int(lowest_k) |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, row in local.items():
        final = row["history"][-1]["observables"]["mean"]
        comp = row.get("final_aggregate_errors", {"local": None, "IR": None, "total": None})
        ac = row["autocorrelation_summary"]
        lines.append(
            f"| {name} | local MCMC | {final.get('xi_2nd_over_L')} | {comp['local']} | {comp['IR']} | {comp['total']} | "
            f"{ac['M']['tau_int_initial_positive']} | {ac['S_density']['tau_int_initial_positive']} | {ac['phi2']['tau_int_initial_positive']} | {ac['lowest_momentum_mode']['tau_int_initial_positive']} |"
        )
    for name, row in restricted.items():
        final = row.get("final", {})
        lines.append(
            f"| {name} | restricted detail | {final.get('xi_2nd_over_L')} | {final.get('distance_to_true')} | {final.get('patch_acceptance_rate')} | {final.get('fixed_coarse_max_error')} |  |  |  |  |"
        )

    lines.extend(
        [
            "",
            "## Main Answers",
            "",
            f"Best accepted movement is `{ranked[0]['schedule']['name']}` with start `{ranked[0]['start_mode']}`.",
            "Eta refresh is judged by both its acceptance and whether it improves local observables without pushing xi/L away from the cluster reference.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_outputs(path: Path, summary: dict[str, object]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    runs = summary["runs"]
    labels = [f"{r['start_mode']}\n{r['schedule']['name']}" for r in runs]
    with PdfPages(path) as pdf:
        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        axes[0, 0].bar(labels, [r["promoted_acceptance"] for r in runs])
        axes[0, 0].set_ylabel("promoted A/R")
        axes[0, 1].bar(labels, [r["eta_refresh_acceptance"] for r in runs])
        axes[0, 1].set_ylabel("eta refresh A/R")
        axes[1, 0].bar(labels, [r["accepted_D_patch_per_attempt"] for r in runs])
        axes[1, 0].set_ylabel("accepted D_patch/attempt")
        axes[1, 1].bar(labels, [r["final_observables"]["xi_2nd_over_L"] for r in runs])
        axes[1, 1].set_ylabel("final xi/L")
        for ax in axes.ravel():
            ax.tick_params(axis="x", rotation=25)
            ax.grid(alpha=0.2, axis="y")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 5))
        for row in runs:
            xs = [h["attempt"] for h in row["history"]]
            ys = [h["observables"]["mean"]["xi_2nd_over_L"] for h in row["history"]]
            ax.plot(xs, ys, marker="o", ms=2, label=f"{row['start_mode']} {row['schedule']['name']}")
        ax.set_xlabel("promoted attempts/config")
        ax.set_ylabel("xi_2nd/L")
        ax.legend(fontsize=7)
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
    ref = references_for_setup(args, 0.320)["ref_0.320"]
    controls = load_control_summaries(args)

    runs = []
    for start_index, start_mode in enumerate(("upscaled_start", "equilibrium_start")):
        for schedule_index, schedule in enumerate(SCHEDULES):
            gen = torch.Generator(device=device).manual_seed(args.seed + 10000 * start_index + 1000 * schedule_index)
            row = run_chain_with_refresh(flow, psi0, metadata, {"kappa_f": 0.320, "patch_size": 2, "sigma_psi": 0.18, "n_inner_hits": 200}, schedule, args, start_mode, gen, ref, device)
            runs.append(row)
            print(
                f"{start_mode} {schedule['name']} A={row['promoted_acceptance']:.4g} "
                f"etaA={row['eta_refresh_acceptance']:.4g} D={row['accepted_D_patch_per_attempt']:.4g} "
                f"xi={row['final_observables']['xi_2nd_over_L']}",
                flush=True,
            )

    local_controls = {}
    phi_upscaled = initial_upscaled_phi(flow, psi0, metadata, 0.320, args, args.seed + 4242)
    local_controls["upscaled_start"] = run_local_mcmc(phi_upscaled, 0.320, args)
    local_controls["upscaled_start"]["final_observables"] = local_controls["upscaled_start"]["history"][-1]["observables"]["mean"]
    local_controls["upscaled_start"]["final_comparison"] = compare_observables(local_controls["upscaled_start"]["history"][-1]["observables"], ref)
    local_controls["upscaled_start"]["final_aggregate_errors"] = aggregate_errors(local_controls["upscaled_start"]["final_comparison"])
    ref_phi = load_cluster_or_reference(args, 0.320)[: args.n_configs].to(device)
    local_controls["equilibrium_start"] = run_local_mcmc(ref_phi, 0.320, args)
    local_controls["equilibrium_start"]["final_observables"] = local_controls["equilibrium_start"]["history"][-1]["observables"]["mean"]
    local_controls["equilibrium_start"]["final_comparison"] = compare_observables(local_controls["equilibrium_start"]["history"][-1]["observables"], ref)
    local_controls["equilibrium_start"]["final_aggregate_errors"] = aggregate_errors(local_controls["equilibrium_start"]["final_comparison"])

    summary = {
        "setup": {
            "checkpoint": str(args.checkpoint),
            "metadata": str(args.metadata),
            "n_configs": args.n_configs,
            "promoted_attempts_per_config": args.n_promoted_attempts_per_config,
            "measure_every": args.measure_every,
            "reference": {"kappa_f": 0.320, "xi_2nd_over_L": ref["mean"]["xi_2nd_over_L"], "chi": ref["mean"]["chi"], "Binder": ref["mean"]["Binder"]},
            "cost_note": "Promoted cost units count 200 coarse hits + one flow eval approximated as 256 units + one fine action eval approximated as 256 units. Eta refresh adds one extra flow eval and one fine action eval at fixed psi.",
        },
        "controls": controls,
        "runs": runs,
        "local_mcmc": local_controls,
        "cost_units": cost_units(200),
    }

    summary_path = args.output_dir / "same_eta_plus_eta_refresh_summary.json"
    report_path = args.output_dir / "same_eta_plus_eta_refresh_report.md"
    plots_path = args.output_dir / "same_eta_plus_eta_refresh_plots.pdf"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_report(report_path, summary)
    plot_outputs(plots_path, summary)
    print(f"wrote {summary_path}")
    print(f"wrote {report_path}")
    print(f"wrote {plots_path}")


if __name__ == "__main__":
    main()
