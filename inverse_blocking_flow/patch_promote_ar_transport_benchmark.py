"""Focused transport benchmark for promoted coarse-patch A/R updates."""

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

from inverse_blocking_flow.correlation_length_bootstrap import projected_connected_correlator
from inverse_blocking_flow.data import load_or_generate_fine_configs
from inverse_blocking_flow.fixed_flow_patch_inner_mcmc_ar_pilot import inner_patch_mcmc, load_flow, quantiles, sample_u
from inverse_blocking_flow.flow import make_conditioning
from inverse_blocking_flow.haar import soft_block, soft_kernel_term, soft_reconstruct
from inverse_blocking_flow.kappac030_trainable_kappaf_upscale import load_or_generate_coarse
from inverse_blocking_flow.logw_kappa_scan_blocked_coarse import ensemble_summary
from inverse_blocking_flow.phi4 import Phi4Params, checkerboard_metropolis_sweep, phi4_action


SETUPS = [
    {"name": "A_prod_kappa_4x4_h200", "kappa_f": 0.3288467228412628, "patch_size": 4, "sigma_psi": 0.14, "n_inner_hits": 200},
    {"name": "B_target_kappa_4x4_h200", "kappa_f": 0.320000, "patch_size": 4, "sigma_psi": 0.14, "n_inner_hits": 200},
    {"name": "C_target_kappa_2x2_h200", "kappa_f": 0.320000, "patch_size": 2, "sigma_psi": 0.18, "n_inner_hits": 200},
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("inverse_blocking_flow/outputs_fine16/production_upscaler_kappac030_softalpha2.pt"))
    parser.add_argument("--metadata", type=Path, default=Path("inverse_blocking_flow/outputs_fine16/production_upscaler_kappac030_softalpha2_metadata.json"))
    parser.add_argument("--coarse-data-path", type=Path, default=Path("inverse_blocking_flow/outputs_fine16/production_coarse_kappac030_configs.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("inverse_blocking_flow/outputs_fine16"))
    parser.add_argument("--n-configs", type=int, default=128)
    parser.add_argument("--n-promoted-attempts-per-config", type=int, default=1000)
    parser.add_argument("--measure-every", type=int, default=20)
    parser.add_argument("--diagnostic-bootstrap", type=int, default=80)
    parser.add_argument("--start-from-equilibrium-fine", action="store_true")
    parser.add_argument("--include-equilibrium-start", action="store_true")
    parser.add_argument("--consistency-atol", type=float, default=1e-5)
    parser.add_argument("--reference-burn-in", type=int, default=400)
    parser.add_argument("--sample-interval", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--proposal-width", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=818181)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--smoke", action="store_true")
    return parser


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
SCALAR_OBS = LOCAL_OBS + IR_OBS


def bootstrap_observables(phi: torch.Tensor, params: Phi4Params, n_bootstrap: int, seed: int) -> dict[str, object]:
    estimate = detailed_observables(phi, params)
    gen = torch.Generator().manual_seed(seed)
    n = phi.shape[0]
    samples: dict[str, list[float]] = {key: [] for key in SCALAR_OBS}
    for _ in range(n_bootstrap):
        idx = torch.randint(0, n, (n,), generator=gen)
        row = detailed_observables(phi[idx], params)
        for key in SCALAR_OBS:
            value = row.get(key)
            if value is not None:
                samples[key].append(float(value))
    bootstrap = {}
    for key, values in samples.items():
        x = torch.tensor(values, dtype=torch.float64)
        bootstrap[key] = {
            "stderr": float(x.std(unbiased=True)) if x.numel() > 1 else None,
            "bootstrap_mean": float(x.mean()) if x.numel() else None,
        }
    return {"mean": estimate, "bootstrap": bootstrap, "histograms": histograms(phi, params)}


def histograms(phi: torch.Tensor, params: Phi4Params) -> dict[str, object]:
    phi = phi.detach().float().cpu()
    volume = phi.shape[-2] * phi.shape[-1]
    values = {
        "S_density": (phi4_action(phi, params) / float(volume)).tolist(),
        "phi2": phi.square().mean(dim=(-2, -1)).tolist(),
        "M": phi.sum(dim=(-2, -1)).tolist(),
        "absM": phi.sum(dim=(-2, -1)).abs().tolist(),
    }
    out = {}
    for key, vals in values.items():
        x = torch.tensor(vals, dtype=torch.float64)
        counts, edges = torch.histogram(x, bins=30)
        out[key] = {"counts": [int(v) for v in counts], "edges": [float(v) for v in edges]}
    return out


def autocorr_summary(values: list[float | None]) -> dict[str, float | None]:
    xs = [float(v) for v in values if v is not None]
    if len(xs) < 4:
        return {"lag_1": None, "decorrelation_lag_1_over_e": None, "tau_int_initial_positive": None}
    x = torch.tensor(xs, dtype=torch.float64)
    x = x - x.mean()
    var = float((x.square().mean()).item())
    if var <= 1e-16:
        return {"lag_1": 0.0, "decorrelation_lag_1_over_e": 0.0, "tau_int_initial_positive": 0.5}
    ac = []
    max_lag = min(len(xs) - 1, 50)
    for lag in range(max_lag + 1):
        ac.append(float((x[: len(x) - lag] * x[lag:]).mean().item() / var))
    decor = None
    for lag, val in enumerate(ac):
        if val < 1.0 / torch.e:
            decor = float(lag)
            break
    tau = 0.5
    for val in ac[1:]:
        if val <= 0.0:
            break
        tau += val
    return {"lag_1": ac[1] if len(ac) > 1 else None, "decorrelation_lag_1_over_e": decor, "tau_int_initial_positive": float(tau)}


def reference_for_kappa(args: argparse.Namespace, kappa: float) -> torch.Tensor:
    tag = str(round(kappa, 6)).replace(".", "p")
    path = args.output_dir / f"fine_reference_transport_kappa_{tag}.pt"
    return load_or_generate_fine_configs(
        path,
        n_configs=max(args.n_configs, 512),
        fine_size=16,
        params=Phi4Params(kappa=kappa, lam=1.0),
        burn_in=args.reference_burn_in,
        interval=args.sample_interval,
        batch_size=args.batch_size,
        proposal_width=args.proposal_width,
        seed=args.seed + int(round(kappa * 100000)),
        device=args.device,
    ).float()


def load_cluster_or_reference(args: argparse.Namespace, kappa: float) -> torch.Tensor:
    cluster_names = {
        0.320: "cluster_refs_L16_k0320.pt",
        0.325: "cluster_refs_L16_k0325.pt",
        0.330: "cluster_refs_L16_k0330.pt",
    }
    rounded = round(kappa, 3)
    name = cluster_names.get(rounded)
    if name is not None:
        path = args.output_dir / name
        if path.exists():
            data = torch.load(path, map_location="cpu", weights_only=False)
            return (data["phi"] if isinstance(data, dict) else data).float()
    return reference_for_kappa(args, kappa)


def references_for_setup(args: argparse.Namespace, kappa: float) -> dict[str, object]:
    refs = {}
    if abs(kappa - 0.320) < 1e-8:
        phi = load_cluster_or_reference(args, 0.320)[: max(args.n_configs, 512)]
        refs["ref_0.320"] = bootstrap_observables(phi, Phi4Params(0.320, 1.0), args.diagnostic_bootstrap, args.seed + 320)
    elif abs(kappa - 0.3288467228412628) < 1e-5:
        for ref_kappa in (0.325, 0.330):
            phi = load_cluster_or_reference(args, ref_kappa)[: max(args.n_configs, 512)]
            refs[f"bracket_ref_{ref_kappa:.3f}"] = bootstrap_observables(phi, Phi4Params(ref_kappa, 1.0), args.diagnostic_bootstrap, args.seed + int(ref_kappa * 1000))
        refs["note"] = "No exact cluster reference at kappa_f=0.328847; comparisons bracket with 0.325 and 0.330."
    else:
        phi = reference_for_kappa(args, kappa)[: max(args.n_configs, 512)]
        refs[f"generated_ref_{kappa:.6f}"] = bootstrap_observables(phi, Phi4Params(kappa, 1.0), args.diagnostic_bootstrap, args.seed + int(kappa * 100000))
    return refs


def compare_observables(obs: dict[str, object], ref: dict[str, object]) -> dict[str, object]:
    out = {}
    obs_mean = obs["mean"]
    obs_boot = obs["bootstrap"]
    ref_mean = ref["mean"]
    ref_boot = ref["bootstrap"]
    for key in SCALAR_OBS:
        value = obs_mean.get(key)
        ref_value = ref_mean.get(key)
        obs_err = obs_boot.get(key, {}).get("stderr")
        ref_err = ref_boot.get(key, {}).get("stderr")
        if value is None or ref_value is None:
            out[key] = {"mean": value, "bootstrap_error": obs_err, "reference_mean": ref_value, "reference_error": ref_err, "z_score": None, "relative_difference": None}
        else:
            denom = (float(obs_err or 0.0) ** 2 + float(ref_err or 0.0) ** 2) ** 0.5
            rel = None if abs(float(ref_value)) < 1e-12 else (float(value) - float(ref_value)) / float(ref_value)
            out[key] = {
                "mean": float(value),
                "bootstrap_error": obs_err,
                "reference_mean": float(ref_value),
                "reference_error": ref_err,
                "z_score": None if denom <= 1e-12 else (float(value) - float(ref_value)) / denom,
                "relative_difference": rel,
            }
    return out


def aggregate_errors(comp: dict[str, object]) -> dict[str, float]:
    def agg(keys: list[str]) -> float:
        vals = [abs(comp[key]["relative_difference"]) for key in keys if comp.get(key, {}).get("relative_difference") is not None]
        return float(sum(vals) / len(vals)) if vals else float("nan")

    return {"local": agg(LOCAL_OBS), "IR": agg(IR_OBS), "total": agg(LOCAL_OBS + IR_OBS)}


def cost_units(n_inner_hits: int, fine_size: int = 16) -> dict[str, float]:
    fine_sites = fine_size * fine_size
    return {
        "coarse_site_hits": float(n_inner_hits),
        "flow_eval_units": float(fine_sites),
        "fine_action_eval_units": float(fine_sites),
        "rough_total_units": float(n_inner_hits + 2 * fine_sites),
        "fine_local_sweep_units": float(fine_sites),
    }


@torch.no_grad()
def initial_upscaled_phi(flow, psi0: torch.Tensor, metadata: dict[str, object], kappa_f: float, args: argparse.Namespace, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=psi0.device).manual_seed(seed)
    u, _logq = sample_u(flow, psi0, str(metadata["conditioning_mode"]), generator)
    return soft_reconstruct(psi0[:, 0], u).cpu()


@torch.no_grad()
def initial_state_from_flow(flow, psi0: torch.Tensor, metadata: dict[str, object], setup: dict[str, object], args: argparse.Namespace, generator: torch.Generator):
    params_c = Phi4Params(float(metadata["kappa_c"]), float(metadata["lambda"]))
    params_f = Phi4Params(float(setup["kappa_f"]), float(metadata["lambda"]))
    alpha = float(metadata["soft_alpha"])
    conditioning_mode = str(metadata["conditioning_mode"])
    psi = psi0.clone()
    u, logq = sample_u(flow, psi, conditioning_mode, generator)
    phi = soft_reconstruct(psi[:, 0], u)
    return psi, u, phi, phi4_action(phi, params_f), phi4_action(psi[:, 0], params_c), soft_kernel_term(u, alpha), logq


@torch.no_grad()
def initial_state_from_equilibrium(flow, metadata: dict[str, object], setup: dict[str, object], args: argparse.Namespace, generator: torch.Generator, device: torch.device):
    params_c = Phi4Params(float(metadata["kappa_c"]), float(metadata["lambda"]))
    params_f = Phi4Params(float(setup["kappa_f"]), float(metadata["lambda"]))
    alpha = float(metadata["soft_alpha"])
    conditioning_mode = str(metadata["conditioning_mode"])
    phi_ref = reference_for_kappa(args, float(setup["kappa_f"]))[: args.n_configs].to(device)
    psi_3d, u = soft_block(phi_ref, alpha, generator=generator)
    psi = psi_3d.unsqueeze(1).float()
    cond = make_conditioning(psi, conditioning_mode)
    _eta, logq = flow.inverse_logq(u, cond)
    phi = soft_reconstruct(psi[:, 0], u)
    return psi, u, phi, phi4_action(phi, params_f), phi4_action(psi[:, 0], params_c), soft_kernel_term(u, alpha), logq


@torch.no_grad()
def consistency_check(flow, psi: torch.Tensor, u: torch.Tensor, phi: torch.Tensor, sf: torch.Tensor, kernel: torch.Tensor, logq: torch.Tensor, metadata: dict[str, object], params_f: Phi4Params) -> dict[str, float]:
    alpha = float(metadata["soft_alpha"])
    cond = make_conditioning(psi, str(metadata["conditioning_mode"]))
    phi_rec = soft_reconstruct(psi[:, 0], u)
    kernel_rec = soft_kernel_term(u, alpha)
    _eta, logq_rec = flow.inverse_logq(u, cond)
    sf_rec = phi4_action(phi_rec, params_f)
    return {
        "max_phi_reconstruction_error": float((phi - phi_rec).abs().max().cpu()),
        "max_K_error": float((kernel - kernel_rec).abs().max().cpu()),
        "max_logq_error": float((logq - logq_rec).abs().max().cpu()),
        "max_S_f_error": float((sf - sf_rec).abs().max().cpu()),
    }


def component_block_stats(values: list[float]) -> dict[str, object]:
    if not values:
        return {"mean": None, "std": None, "quantiles": quantiles(values)}
    x = torch.tensor(values, dtype=torch.float64)
    qs = torch.quantile(x, torch.tensor([0.01, 0.05, 0.5, 0.95, 0.99], dtype=torch.float64))
    return {
        "mean": float(x.mean()),
        "std": float(x.std(unbiased=False)),
        "quantiles": {k: float(v) for k, v in zip(["q01", "q05", "q50", "q95", "q99"], qs)},
    }


def correlation(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(ys) < 2:
        return None
    x = torch.tensor(xs, dtype=torch.float64)
    y = torch.tensor(ys, dtype=torch.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = x.square().mean().sqrt() * y.square().mean().sqrt()
    if float(denom) <= 1e-14:
        return None
    return float((x * y).mean() / denom)


def split_component_summary(rows: list[dict[str, float]]) -> dict[str, object]:
    out: dict[str, object] = {}
    for label, subset in {
        "all": rows,
        "accepted": [row for row in rows if row["accepted"] > 0.5],
        "rejected": [row for row in rows if row["accepted"] <= 0.5],
    }.items():
        out[label] = {
            "n": len(subset),
            "components": {
                key: component_block_stats([row[key] for row in subset])
                for key in ["minus_Delta_S_f", "minus_Delta_K_alpha", "plus_Delta_S_c", "logq_old_minus_logq_new", "total_logA"]
            },
            "correlations": {
                "corr_logA_Delta_S_f": correlation([row["total_logA"] for row in subset], [row["Delta_S_f"] for row in subset]),
                "corr_logA_Delta_K_alpha": correlation([row["total_logA"] for row in subset], [row["Delta_K_alpha"] for row in subset]),
                "corr_logA_Delta_S_c": correlation([row["total_logA"] for row in subset], [row["Delta_S_c"] for row in subset]),
                "corr_logA_logq_ratio": correlation([row["total_logA"] for row in subset], [row["logq_old_minus_logq_new"] for row in subset]),
                "corr_logA_D_patch": correlation([row["total_logA"] for row in subset], [row["D_patch"] for row in subset]),
            },
        }
    return out


@torch.no_grad()
def run_promoted(
    flow,
    psi0: torch.Tensor,
    metadata: dict[str, object],
    setup: dict[str, object],
    args: argparse.Namespace,
    generator: torch.Generator,
    references: dict[str, object],
    device: torch.device,
) -> dict[str, object]:
    params_c = Phi4Params(float(metadata["kappa_c"]), float(metadata["lambda"]))
    params_f = Phi4Params(float(setup["kappa_f"]), float(metadata["lambda"]))
    alpha = float(metadata["soft_alpha"])
    conditioning_mode = str(metadata["conditioning_mode"])
    patch_size = int(setup["patch_size"])
    sigma_psi = float(setup["sigma_psi"])
    n_inner_hits = int(setup["n_inner_hits"])
    if args.start_from_equilibrium_fine:
        psi, u, phi, sf, sc, kernel, logq = initial_state_from_equilibrium(flow, metadata, setup, args, generator, device)
    else:
        psi, u, phi, sf, sc, kernel, logq = initial_state_from_flow(flow, psi0, metadata, setup, args, generator)
    obs0 = bootstrap_observables(phi.cpu(), params_f, args.diagnostic_bootstrap, args.seed + 500)
    comp0 = {name: compare_observables(obs0, ref) for name, ref in references.items()}
    history = [
        {
            "attempt": 0,
            "cost_units": 0.0,
            "observables": obs0,
            "reference_comparisons": comp0,
            "aggregate_errors": {name: aggregate_errors(comp) for name, comp in comp0.items()},
            "ar_block": None,
            "state_consistency": consistency_check(flow, psi, u, phi, sf, kernel, logq, metadata, params_f),
        }
    ]
    inner_accepts = 0
    inner_proposals = 0
    fine_accepts = 0
    accepted_patch_sq = []
    accepted_site_sq = []
    all_patch_sq = []
    loga_vals = []
    dsf_vals = []
    dsc_vals = []
    dk_vals = []
    logq_ratio_vals = []
    block_accepts = 0
    block_total = 0
    block_loga: list[float] = []
    block_dsf: list[float] = []
    block_dsc: list[float] = []
    block_dk: list[float] = []
    block_logq: list[float] = []
    component_rows: list[dict[str, float]] = []
    accepted_by_patch_bin = {key: [0, 0] for key in ["0-0.5", "0.5-1", "1-2", "2-4", "4+"]}
    units = cost_units(n_inner_hits)
    for attempt in range(1, args.n_promoted_attempts_per_config + 1):
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
        u_prop, logq_new = sample_u(flow, psi_prop, conditioning_mode, generator)
        phi_prop = soft_reconstruct(psi_prop[:, 0], u_prop)
        sf_new = phi4_action(phi_prop, params_f)
        sc_new = phi4_action(psi_prop[:, 0], params_c)
        k_new = soft_kernel_term(u_prop, alpha)
        logq_ratio = logq - logq_new
        loga = -(sf_new - sf) - (k_new - kernel) + (sc_new - sc) + logq_ratio
        logu = torch.log(torch.rand(loga.shape, dtype=psi.dtype, device=psi.device, generator=generator))
        accept = logu < loga
        fine_accepts += int(accept.sum())
        block_accepts += int(accept.sum())
        block_total += int(accept.numel())
        all_patch_sq.extend(patch_sq.cpu().tolist())
        loga_vals.extend(loga.cpu().tolist())
        dsf_vals.extend((sf_new - sf).cpu().tolist())
        dsc_vals.extend((sc_new - sc).cpu().tolist())
        dk_vals.extend((k_new - kernel).cpu().tolist())
        logq_ratio_vals.extend(logq_ratio.cpu().tolist())
        block_loga.extend(loga.cpu().tolist())
        block_dsf.extend((sf_new - sf).cpu().tolist())
        block_dsc.extend((sc_new - sc).cpu().tolist())
        block_dk.extend((k_new - kernel).cpu().tolist())
        block_logq.extend(logq_ratio.cpu().tolist())
        neg_dsf_list = (-(sf_new - sf)).cpu().tolist()
        neg_dk_list = (-(k_new - kernel)).cpu().tolist()
        dsc_list = (sc_new - sc).cpu().tolist()
        logq_list = logq_ratio.cpu().tolist()
        loga_list = loga.cpu().tolist()
        dsf_list = (sf_new - sf).cpu().tolist()
        dk_list = (k_new - kernel).cpu().tolist()
        patch_list = patch_sq.cpu().tolist()
        accept_list = accept.cpu().tolist()
        for j in range(len(loga_list)):
            component_rows.append(
                {
                    "accepted": float(bool(accept_list[j])),
                    "minus_Delta_S_f": float(neg_dsf_list[j]),
                    "minus_Delta_K_alpha": float(neg_dk_list[j]),
                    "plus_Delta_S_c": float(dsc_list[j]),
                    "logq_old_minus_logq_new": float(logq_list[j]),
                    "total_logA": float(loga_list[j]),
                    "Delta_S_f": float(dsf_list[j]),
                    "Delta_K_alpha": float(dk_list[j]),
                    "Delta_S_c": float(dsc_list[j]),
                    "D_patch": float(patch_list[j]),
                }
            )
        for val, ok in zip(patch_sq.cpu().tolist(), accept.cpu().tolist()):
            if val < 0.5:
                key = "0-0.5"
            elif val < 1.0:
                key = "0.5-1"
            elif val < 2.0:
                key = "1-2"
            elif val < 4.0:
                key = "2-4"
            else:
                key = "4+"
            accepted_by_patch_bin[key][1] += 1
            accepted_by_patch_bin[key][0] += int(bool(ok))
        if accept.any():
            accepted_patch_sq.extend(patch_sq[accept].cpu().tolist())
            accepted_site_sq.extend((patch_sq[accept] / float(patch_size * patch_size)).cpu().tolist())
            psi[accept] = psi_prop[accept]
            u[accept] = u_prop[accept]
            phi[accept] = phi_prop[accept]
            sf[accept] = sf_new[accept]
            sc[accept] = sc_new[accept]
            kernel[accept] = k_new[accept]
            logq[accept] = logq_new[accept]
        if attempt % args.measure_every == 0 or attempt == args.n_promoted_attempts_per_config:
            obs = bootstrap_observables(phi.cpu(), params_f, args.diagnostic_bootstrap, args.seed + 500 + attempt)
            comps = {name: compare_observables(obs, ref) for name, ref in references.items()}
            history.append(
                {
                    "attempt": attempt,
                    "cost_units": attempt * units["rough_total_units"],
                    "observables": obs,
                    "reference_comparisons": comps,
                    "aggregate_errors": {name: aggregate_errors(comp) for name, comp in comps.items()},
                    "ar_block": {
                        "promoted_acceptance": block_accepts / float(block_total) if block_total else 0.0,
                        "cumulative_acceptance": fine_accepts / float(attempt * psi.shape[0]),
                        "logA": component_block_stats(block_loga),
                        "Delta_S_f": component_block_stats(block_dsf),
                        "Delta_S_c": component_block_stats(block_dsc),
                        "Delta_K_alpha": component_block_stats(block_dk),
                        "logq_old_minus_logq_new": component_block_stats(block_logq),
                        "mean_total_logA_components": {
                            "-Delta_S_f": -component_block_stats(block_dsf)["mean"] if component_block_stats(block_dsf)["mean"] is not None else None,
                            "-Delta_K_alpha": -component_block_stats(block_dk)["mean"] if component_block_stats(block_dk)["mean"] is not None else None,
                            "Delta_S_c": component_block_stats(block_dsc)["mean"],
                            "logq_old_minus_logq_new": component_block_stats(block_logq)["mean"],
                        },
                    },
                    "state_consistency": consistency_check(flow, psi, u, phi, sf, kernel, logq, metadata, params_f),
                }
            )
            block_accepts = 0
            block_total = 0
            block_loga = []
            block_dsf = []
            block_dsc = []
            block_dk = []
            block_logq = []
    total = args.n_promoted_attempts_per_config * psi.shape[0]
    fine_a = fine_accepts / float(total)
    accepted_patch_mean = float(torch.tensor(accepted_patch_sq).mean()) if accepted_patch_sq else 0.0
    return {
        **setup,
        "inner_coarse_acceptance": inner_accepts / float(inner_proposals),
        "promoted_acceptance": fine_a,
        "accepted_D_patch_per_attempt": fine_a * accepted_patch_mean,
        "accepted_D_site_per_attempt": fine_a * (accepted_patch_mean / float(patch_size * patch_size)),
        "accepted_fine_sites_per_attempt": fine_a * float((2 * patch_size) ** 2),
        "accepted_patch_displacement": {"mean": accepted_patch_mean, "quantiles": quantiles(accepted_patch_sq)},
        "all_patch_displacement": {"mean": float(torch.tensor(all_patch_sq).mean()), "quantiles": quantiles(all_patch_sq)},
        "logA": {"mean": float(torch.tensor(loga_vals).mean()), "std": float(torch.tensor(loga_vals).std(unbiased=False)), "quantiles": quantiles(loga_vals)},
        "Delta_S_f": {"mean": float(torch.tensor(dsf_vals).mean()), "std": float(torch.tensor(dsf_vals).std(unbiased=False)), "quantiles": quantiles(dsf_vals)},
        "Delta_S_c": {"mean": float(torch.tensor(dsc_vals).mean()), "std": float(torch.tensor(dsc_vals).std(unbiased=False)), "quantiles": quantiles(dsc_vals)},
        "Delta_K": {"mean": float(torch.tensor(dk_vals).mean()), "std": float(torch.tensor(dk_vals).std(unbiased=False)), "quantiles": quantiles(dk_vals)},
        "logq_ratio": {"mean": float(torch.tensor(logq_ratio_vals).mean()), "std": float(torch.tensor(logq_ratio_vals).std(unbiased=False)), "quantiles": quantiles(logq_ratio_vals)},
        "acceptance_vs_D_patch": {key: (num / den if den else None) for key, (num, den) in accepted_by_patch_bin.items()},
        "logA_component_split": split_component_summary(component_rows),
        "cost_units_per_attempt": units,
        "history": history,
        "autocorrelation": {
            key: autocorr_summary([row["observables"]["mean"][key] for row in history])
            for key in ["M", "absM", "M2", "S_density", "phi2", "Binder", "chi", "xi_2nd_over_L", "lowest_momentum_mode"]
        },
        "final_phi": phi.cpu(),
    }


@torch.no_grad()
def run_local_mcmc(phi0: torch.Tensor, kappa: float, args: argparse.Namespace) -> dict[str, object]:
    params = Phi4Params(kappa, 1.0)
    phi = phi0.clone().cpu()
    generator = torch.Generator().manual_seed(args.seed + int(kappa * 100000) + 33)
    history = [{"sweeps": 0, "cost_units": 0.0, "observables": bootstrap_observables(phi, params, args.diagnostic_bootstrap, args.seed + 700)}]
    max_sweeps = args.n_promoted_attempts_per_config
    for sweep in range(1, max_sweeps + 1):
        checkerboard_metropolis_sweep(phi, params, args.proposal_width, generator)
        if sweep % args.measure_every == 0 or sweep == max_sweeps:
            history.append({"sweeps": sweep, "cost_units": sweep * 16 * 16, "observables": bootstrap_observables(phi, params, args.diagnostic_bootstrap, args.seed + 700 + sweep)})
    return {
        "kappa_f": kappa,
        "history": history,
        "autocorrelation": {
            key: autocorr_summary([row["observables"]["mean"][key] for row in history])
            for key in ["M", "absM", "M2", "S_density", "phi2", "Binder", "chi", "xi_2nd_over_L", "lowest_momentum_mode"]
        },
    }


def strip_phi(result: dict[str, object]) -> dict[str, object]:
    out = dict(result)
    out.pop("final_phi", None)
    return out


def write_report(path: Path, summary: dict[str, object]) -> None:
    ranked = sorted(summary["promoted_runs"], key=lambda r: r["accepted_D_patch_per_attempt"], reverse=True)
    lines = [
        "# Patch Promote A/R Transport Diagnostics",
        "",
        "Ranked by accepted movement, not raw A/R.",
        "",
        "| rank | start | setup | kappa | patch | hits | A/R | D_patch/attempt | fine sites/attempt | final xi/L | local err | IR err | total err | max state err |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for i, row in enumerate(ranked, 1):
        final = row["history"][-1]
        final_mean = final["observables"]["mean"]
        comparison_name = row["primary_reference"]
        final_agg = final["aggregate_errors"][comparison_name]
        state = final["state_consistency"]
        max_state = max(abs(float(v)) for v in state.values())
        lines.append(
            f"| {i} | {row.get('start_mode', 'upscaled_start')} | {row['name']} | {row['kappa_f']:.6g} | {row['patch_size']} | {row['n_inner_hits']} | "
            f"{row['promoted_acceptance']:.6g} | {row['accepted_D_patch_per_attempt']:.6g} | "
            f"{row['accepted_fine_sites_per_attempt']:.6g} | {final_mean['xi_2nd_over_L']} | "
            f"{final_agg['local']:.6g} | {final_agg['IR']:.6g} | {final_agg['total']:.6g} | {max_state:.3g} |"
        )
    lines.extend(["", "## Main Answers", ""])
    best = ranked[0]
    lines.append(f"Best raw transport by accepted movement is `{best['name']}` with `D_patch/attempt={best['accepted_D_patch_per_attempt']:.6g}`.")
    lines.append("")
    lines.extend(
        [
            "## LogA Components",
            "",
            "| start | setup | group | n | A/R | mean -dSf | mean -dK | mean +dSc | mean logq ratio | mean logA | corr logA dSf | corr logA dK | corr logA logq | corr logA Dpatch |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(summary["promoted_runs"], key=lambda r: (r.get("start_mode", ""), r["kappa_f"], r["patch_size"])):
        split = row.get("logA_component_split", {})
        for group in ("accepted", "rejected"):
            part = split.get(group, {})
            comp = part.get("components", {})
            corr = part.get("correlations", {})
            def mean_of(key: str) -> object:
                return comp.get(key, {}).get("mean")
            def fmt(value: object) -> str:
                return "" if value is None else f"{float(value):.6g}"

            lines.append(
                f"| {row.get('start_mode', 'upscaled_start')} | {row['name']} | {group} | {part.get('n', 0)} | "
                f"{row['promoted_acceptance']:.6g} | {fmt(mean_of('minus_Delta_S_f'))} | "
                f"{fmt(mean_of('minus_Delta_K_alpha'))} | {fmt(mean_of('plus_Delta_S_c'))} | "
                f"{fmt(mean_of('logq_old_minus_logq_new'))} | {fmt(mean_of('total_logA'))} | "
                f"{fmt(corr.get('corr_logA_Delta_S_f'))} | {fmt(corr.get('corr_logA_Delta_K_alpha'))} | "
                f"{fmt(corr.get('corr_logA_logq_ratio'))} | {fmt(corr.get('corr_logA_D_patch'))} |"
            )
    lines.append("")
    lines.append("The JSON contains full per-checkpoint observable tables with bootstrap errors, z-scores, relative differences, histograms, A/R component blocks, and state consistency diagnostics.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_outputs(path: Path, summary: dict[str, object]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    runs = summary["promoted_runs"]
    labels = [f"{r.get('start_mode', 'upscaled')}\n{r['name']}" for r in runs]
    with PdfPages(path) as pdf:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes[0, 0].bar(labels, [r["promoted_acceptance"] for r in runs])
        axes[0, 0].set_ylabel("promoted A/R")
        axes[0, 1].bar(labels, [r["accepted_D_patch_per_attempt"] for r in runs])
        axes[0, 1].set_ylabel("accepted D_patch/attempt")
        axes[1, 0].bar(labels, [r["accepted_fine_sites_per_attempt"] for r in runs])
        axes[1, 0].set_ylabel("accepted fine sites/attempt")
        axes[1, 1].bar(labels, [r["history"][-1]["aggregate_errors"][r["primary_reference"]]["total"] for r in runs])
        axes[1, 1].set_ylabel("final mean abs ref rel err")
        for ax in axes.ravel():
            ax.tick_params(axis="x", rotation=25)
            ax.grid(alpha=0.2, axis="y")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 5))
        for r in runs:
            xs = [h["attempt"] for h in r["history"]]
            ys = [h["observables"]["mean"]["xi_2nd_over_L"] for h in r["history"]]
            ax.plot(xs, ys, marker="o", ms=2, label=r["name"])
        ax.set_xlabel("promoted attempts/config")
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
        args.n_promoted_attempts_per_config = 4
        args.measure_every = 2
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
    promoted = []
    local = {}
    reference_cache: dict[str, object] = {}
    start_modes = [("upscaled_start", False)]
    if args.start_from_equilibrium_fine:
        start_modes = [("equilibrium_start", True)]
    elif args.include_equilibrium_start:
        start_modes.append(("equilibrium_start", True))
    for mode_index, (start_mode, use_equilibrium) in enumerate(start_modes):
        original_start_flag = args.start_from_equilibrium_fine
        args.start_from_equilibrium_fine = use_equilibrium
        for i, setup in enumerate(SETUPS):
            gen = torch.Generator(device=device).manual_seed(args.seed + 10000 * mode_index + 1000 * i)
            refs = references_for_setup(args, float(setup["kappa_f"]))
            reference_cache[f"{start_mode}:{setup['name']}"] = refs
            comparable_refs = {k: v for k, v in refs.items() if isinstance(v, dict) and "mean" in v}
            primary_reference = "ref_0.320" if "ref_0.320" in comparable_refs else sorted(comparable_refs)[0]
            result = run_promoted(flow, psi0, metadata, setup, args, gen, comparable_refs, device)
            result["start_mode"] = start_mode
            result["primary_reference"] = primary_reference
            result["reference_note"] = refs.get("note")
            print(
                f"{start_mode} {setup['name']} A={result['promoted_acceptance']:.4g} "
                f"D={result['accepted_D_patch_per_attempt']:.4g} xi={result['history'][-1]['observables']['mean']['xi_2nd_over_L']}",
                flush=True,
            )
            promoted.append(result)
        args.start_from_equilibrium_fine = original_start_flag
    initial_phi_by_kappa = {}
    for kappa in {0.320, 0.3288467228412628}:
        initial_phi_by_kappa[kappa] = initial_upscaled_phi(flow, psi0, metadata, kappa, args, args.seed + int(kappa * 100000))
    for kappa, phi0 in initial_phi_by_kappa.items():
        local[str(kappa)] = run_local_mcmc(phi0, kappa, args)
    serializable_promoted = [strip_phi(r) for r in promoted]
    summary = {
        "setup": {
            "checkpoint": str(args.checkpoint),
            "n_configs": args.n_configs,
            "promoted_attempts_per_config": args.n_promoted_attempts_per_config,
            "measure_every": args.measure_every,
            "setups": SETUPS,
            "include_equilibrium_start": args.include_equilibrium_start,
            "cost_note": "Promoted cost units count n_inner_hits coarse-site hits + one flow eval approximated as 256 units + one fine action eval approximated as 256 units. One fine local MCMC sweep is 256 units.",
        },
        "references": reference_cache,
        "promoted_runs": serializable_promoted,
        "local_mcmc": local,
        "restricted_u_space_note": "No matching restricted u-space correction benchmark is loaded in this script.",
    }
    summary_path = args.output_dir / "patch_promote_ar_transport_diagnostics_summary.json"
    report_path = args.output_dir / "patch_promote_ar_transport_diagnostics_report.md"
    plots_path = args.output_dir / "patch_promote_ar_transport_diagnostics_plots.pdf"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_report(report_path, summary)
    plot_outputs(plots_path, summary)
    print(f"wrote {summary_path}")
    print(f"wrote {report_path}")
    print(f"wrote {plots_path}")


if __name__ == "__main__":
    main()
