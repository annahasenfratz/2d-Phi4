"""Conditional equilibration study for restricted detail MCMC.

For selected fixed coarse fields, run multiple restricted-detail MCMC chains
from reverse-KL, Gaussian, and zero-detail initializations and compare their
final-window conditional observables.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/matplotlib")

from inverse_blocking_flow.data import load_or_generate_fine_configs, make_paired_dataset
from inverse_blocking_flow.flow import ConditionalDetailFlow
from inverse_blocking_flow.haar import block_average, reconstruct_from_average_block
from inverse_blocking_flow.phi4 import Phi4Params, mean_phi2, nearest_neighbor_correlator, phi4_action


OBS = ["S_f", "phi2", "NN_corr", "high_p_power"]
INITS = ["reverse_kl", "gaussian", "zero_detail"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fine-size", type=int, default=32)
    parser.add_argument("--lambda", dest="lam", type=float, default=1.0)
    parser.add_argument("--kappa-fine", type=float, default=0.31)
    parser.add_argument("--n-configs", type=int, default=512)
    parser.add_argument("--n-coarse", type=int, default=8)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--patch-size", type=int, default=2)
    parser.add_argument("--step-size", type=float, default=0.1)
    parser.add_argument("--n-sweeps", type=int, default=1500)
    parser.add_argument("--plateau-window", type=int, default=50)
    parser.add_argument("--data-path", type=Path, default=Path("inverse_blocking_flow/outputs/fine_configs.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("inverse_blocking_flow/outputs"))
    parser.add_argument("--checkpoint", type=Path, default=Path("inverse_blocking_flow/outputs/conditional_detail_flow_reverse_kl.pt"))
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--hidden-channels", type=int, default=48)
    parser.add_argument("--cnn-depth", type=int, default=3)
    parser.add_argument("--burn-in", type=int, default=200)
    parser.add_argument("--sample-interval", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--proposal-width", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=737373)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    return parser


def load_flow(args: argparse.Namespace, device: torch.device) -> ConditionalDetailFlow:
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"missing checkpoint: {args.checkpoint}")
    flow = ConditionalDetailFlow(args.layers, args.hidden_channels, args.cnn_depth).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    flow.load_state_dict(state["model"])
    flow.eval()
    return flow


def high_power_per_config(phi: torch.Tensor) -> torch.Tensor:
    centered = phi - phi.mean(dim=(-2, -1), keepdim=True)
    fft = torch.fft.fftn(centered, dim=(-2, -1))
    power = (fft.real.square() + fft.imag.square()) / (phi.shape[-2] * phi.shape[-1])
    ly, lx = phi.shape[-2:]
    ky = torch.fft.fftfreq(ly, d=1.0, device=phi.device) * ly
    kx = torch.fft.fftfreq(lx, d=1.0, device=phi.device) * lx
    yy, xx = torch.meshgrid(ky, kx, indexing="ij")
    radius = torch.sqrt(xx.square() + yy.square())
    high_mask = radius >= 0.5 * float(radius.max().item())
    return power[:, high_mask].mean(dim=-1)


def per_chain_observables(phi: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "S_f": phi4_action(phi, PARAMS),
        "phi2": mean_phi2(phi),
        "NN_corr": nearest_neighbor_correlator(phi),
        "high_p_power": high_power_per_config(phi),
    }


def true_targets(phi_true: torch.Tensor) -> dict[str, dict[str, float]]:
    obs = per_chain_observables(phi_true)
    out = {}
    for key, values in obs.items():
        mean = float(values.mean().item())
        sem = float(values.std(unbiased=False).item() / (values.numel() ** 0.5))
        # High-p power can be noisy; keep the same +/- 2 SEM rule here.
        out[key] = {"mean": mean, "lo": mean - 2.0 * sem, "hi": mean + 2.0 * sem}
    return out


def aggregate_distance(row: dict[str, float], target: dict[str, dict[str, float]]) -> float:
    vals = []
    for key in OBS:
        denom = abs(target[key]["mean"])
        if denom > 1e-14:
            vals.append(abs(row[key] - target[key]["mean"]) / denom)
    return float(sum(vals) / len(vals))


def first_in_band(history: list[dict[str, float]], target: dict[str, dict[str, float]]) -> int | None:
    for i, row in enumerate(history):
        if all(target[key]["lo"] <= row[key] <= target[key]["hi"] for key in OBS):
            return i
    return None


def median_or_none(values: list[int | None]) -> float | None:
    xs = sorted(v for v in values if v is not None)
    if not xs:
        return None
    mid = len(xs) // 2
    if len(xs) % 2:
        return float(xs[mid])
    return 0.5 * float(xs[mid - 1] + xs[mid])


def plateau_distance(row: dict[str, float], plateau: dict[str, dict[str, float]]) -> float:
    vals = []
    for key in OBS:
        scale = max(abs(plateau[key]["mean"]), plateau[key]["error"], 1e-12)
        vals.append(abs(row[key] - plateau[key]["mean"]) / scale)
    return float(sum(vals) / len(vals))


def first_plateau_sweep(
    history: list[dict[str, float]],
    plateau: dict[str, dict[str, float]],
    window_size: int,
) -> int | None:
    if len(history) < window_size:
        return None
    inside = []
    for start in range(0, len(history) - window_size + 1):
        window = history[start : start + window_size]
        ok = True
        for key in OBS:
            mean = sum(row[key] for row in window) / float(window_size)
            lo = plateau[key]["mean"] - plateau[key]["error"]
            hi = plateau[key]["mean"] + plateau[key]["error"]
            if not (lo <= mean <= hi):
                ok = False
                break
        inside.append(ok)
    suffix_ok = [False] * len(inside)
    running = True
    for i in range(len(inside) - 1, -1, -1):
        running = running and inside[i]
        suffix_ok[i] = running
    for i, ok in enumerate(suffix_ok):
        if ok:
            return i
    return None


def rhat(chains: list[list[float]]) -> float | None:
    if len(chains) < 2:
        return None
    n = min(len(chain) for chain in chains)
    if n < 2:
        return None
    x = torch.tensor([chain[-n:] for chain in chains], dtype=torch.float64)
    m = x.shape[0]
    chain_means = x.mean(dim=1)
    chain_vars = x.var(dim=1, unbiased=True)
    w = chain_vars.mean()
    if float(w.item()) <= 0.0:
        return None
    b = n * chain_means.var(unbiased=True)
    var_hat = ((n - 1) / n) * w + b / n
    return float(torch.sqrt(var_hat / w).item())


def add_plateau_diagnostics(
    chains: list[dict[str, object]],
    n_coarse: int,
    window_size: int,
) -> dict[str, object]:
    by_coarse = []
    for coarse_idx in range(n_coarse):
        subset = [c for c in chains if c["coarse_index"] == coarse_idx]
        history_len = len(subset[0]["history"])
        final_start = int(0.75 * history_len)
        plateau = {}
        for key in OBS:
            values = [
                row[key]
                for chain in subset
                for row in chain["history"][final_start:]
            ]
            st = stats(values)
            # For a running-window mean from one Markov chain, the relevant
            # plateau uncertainty is the observed final-window fluctuation
            # scale, not the SEM of the combined plateau mean.
            error = max(st["std"], 1e-6 * max(abs(st["mean"]), 1.0))
            plateau[key] = {**st, "error": error, "n_samples": len(values)}

        for chain in subset:
            for row in chain["history"]:
                row["plateau_distance"] = plateau_distance(row, plateau)
            chain["first_plateau_sweep"] = first_plateau_sweep(chain["history"], plateau, window_size)

        by_init = {}
        for init in INITS:
            init_subset = [c for c in subset if c["init"] == init]
            by_init[init] = {
                "first_plateau_sweeps": [c["first_plateau_sweep"] for c in init_subset],
                "median_first_plateau_sweep": median_or_none([c["first_plateau_sweep"] for c in init_subset]),
                "final_window_plateau_distance": stats(
                    [
                        row["plateau_distance"]
                        for c in init_subset
                        for row in c["history"][final_start:]
                    ]
                ),
            }

        rhats = {}
        rhat_start = int(0.5 * history_len)
        for key in OBS:
            rhats[key] = rhat([[row[key] for row in c["history"][rhat_start:]] for c in subset])

        by_coarse.append(
            {
                "coarse_index": coarse_idx,
                "plateau": plateau,
                "by_initialization": by_init,
                "rhat": rhats,
            }
        )

    median_by_init = {}
    for init in INITS:
        per_coarse = [
            item["by_initialization"][init]["median_first_plateau_sweep"]
            for item in by_coarse
        ]
        median_by_init[init] = median_or_none(per_coarse)
    reverse = median_by_init.get("reverse_kl")
    gaussian = median_by_init.get("gaussian")
    zero = median_by_init.get("zero_detail")
    reverse_reduces = (
        reverse is not None
        and (gaussian is None or reverse < gaussian)
        and (zero is None or reverse < zero)
    )
    return {
        "window_size": window_size,
        "by_coarse": by_coarse,
        "median_first_plateau_sweep_by_initialization": median_by_init,
        "reverse_kl_reduces_time_to_plateau": reverse_reduces,
    }


def stats(values: list[float]) -> dict[str, float]:
    x = torch.tensor(values, dtype=torch.float64)
    return {"mean": float(x.mean().item()), "std": float(x.std(unbiased=False).item())}


@torch.no_grad()
def run_batched_chains(
    phi_c_base: torch.Tensor,
    d_by_init: dict[str, torch.Tensor],
    args: argparse.Namespace,
    target: dict[str, dict[str, float]],
    device: torch.device,
) -> tuple[list[dict[str, object]], float]:
    labels = []
    phi_c_batches = []
    d_batches = []
    for init_name in INITS:
        for coarse_idx in range(phi_c_base.shape[0]):
            for seed_idx in range(args.n_seeds):
                labels.append({"init": init_name, "coarse_index": coarse_idx, "seed_index": seed_idx})
                phi_c_batches.append(phi_c_base[coarse_idx : coarse_idx + 1])
                d_batches.append(d_by_init[init_name][coarse_idx, seed_idx : seed_idx + 1])
    phi_c = torch.cat(phi_c_batches, dim=0).to(device)
    d = torch.cat(d_batches, dim=0).to(device)
    n_chains = d.shape[0]
    histories: list[list[dict[str, float]]] = [[] for _ in range(n_chains)]
    max_coarse_error = 0.0
    phi = reconstruct_from_average_block(phi_c[:, 0], d)
    action = phi4_action(phi, PARAMS)
    coarse_y, coarse_x = phi_c.shape[-2:]
    generator = torch.Generator(device=device).manual_seed(args.seed + 900)

    for sweep in range(args.n_sweeps + 1):
        phi = reconstruct_from_average_block(phi_c[:, 0], d)
        obs = per_chain_observables(phi)
        coarse_error = (block_average(phi) - phi_c[:, 0]).abs().amax(dim=(-2, -1))
        max_coarse_error = max(max_coarse_error, float(coarse_error.max().item()))
        for i in range(n_chains):
            row = {key: float(obs[key][i].item()) for key in OBS}
            row["aggregate_distance"] = aggregate_distance(row, target)
            row["patch_acceptance_rate"] = 0.0 if sweep == 0 else histories[i][-1]["patch_acceptance_rate"]
            row["fixed_coarse_error"] = float(coarse_error[i].item())
            histories[i].append(row)
        if sweep == args.n_sweeps:
            break
        accepts = torch.zeros(n_chains, dtype=torch.float32, device=device)
        proposals = 0
        for y0 in range(0, coarse_y, args.patch_size):
            for x0 in range(0, coarse_x, args.patch_size):
                y1 = min(y0 + args.patch_size, coarse_y)
                x1 = min(x0 + args.patch_size, coarse_x)
                d_new = d.clone()
                noise = torch.randn(d[:, :, y0:y1, x0:x1].shape, generator=generator, device=device)
                d_new[:, :, y0:y1, x0:x1] = d[:, :, y0:y1, x0:x1] + args.step_size * noise
                phi_new = reconstruct_from_average_block(phi_c[:, 0], d_new)
                action_new = phi4_action(phi_new, PARAMS)
                log_accept = -action_new + action
                log_u = torch.log(torch.rand(log_accept.shape, generator=generator, device=device))
                accept = log_u < log_accept
                if accept.any():
                    d[accept] = d_new[accept]
                    action[accept] = action_new[accept]
                accepts += accept.float()
                proposals += 1
        for i in range(n_chains):
            histories[i][-1]["patch_acceptance_rate"] = float((accepts[i] / proposals).item())

    chains = []
    start = int(0.75 * len(histories[0]))
    for label, history in zip(labels, histories):
        window = history[start:]
        final_window = {key: stats([row[key] for row in window]) for key in OBS + ["aggregate_distance", "patch_acceptance_rate"]}
        chains.append(
            {
                **label,
                "first_true_band_sweep": first_in_band(history, target),
                "final_window": final_window,
                "history": history,
            }
        )
    return chains, max_coarse_error


def summarize_by_coarse(chains: list[dict[str, object]], n_coarse: int) -> list[dict[str, object]]:
    out = []
    for coarse_idx in range(n_coarse):
        subset = [c for c in chains if c["coarse_index"] == coarse_idx]
        init_stats = {}
        for init in INITS:
            init_subset = [c for c in subset if c["init"] == init]
            init_stats[init] = {
                "first_true_band_sweeps": [c["first_true_band_sweep"] for c in init_subset],
                "mean_acceptance": stats([c["final_window"]["patch_acceptance_rate"]["mean"] for c in init_subset]),
                "final_window": {
                    key: stats([c["final_window"][key]["mean"] for c in init_subset])
                    for key in OBS + ["aggregate_distance"]
                },
            }
        spread = {}
        for key in OBS + ["aggregate_distance"]:
            init_means = [init_stats[init]["final_window"][key]["mean"] for init in INITS]
            all_seed_means = [c["final_window"][key]["mean"] for c in subset]
            spread[key] = {
                "between_initialization_std": stats(init_means)["std"],
                "between_seed_std": stats(all_seed_means)["std"],
                "overall_mean": stats(all_seed_means)["mean"],
                "overall_std": stats(all_seed_means)["std"],
            }
        out.append({"coarse_index": coarse_idx, "by_initialization": init_stats, "spreads": spread})
    return out


def agreement_flags(coarse_summary: list[dict[str, object]]) -> dict[str, object]:
    flags = []
    for item in coarse_summary:
        max_ratio = 0.0
        for key in OBS + ["aggregate_distance"]:
            sp = item["spreads"][key]
            denom = sp["between_seed_std"] if sp["between_seed_std"] > 1e-12 else 1e-12
            max_ratio = max(max_ratio, sp["between_initialization_std"] / denom)
        flags.append({"coarse_index": item["coarse_index"], "max_init_to_seed_spread_ratio": max_ratio, "agrees_within_errors": max_ratio <= 2.0})
    return {
        "per_coarse": flags,
        "all_agree_within_errors": all(flag["agrees_within_errors"] for flag in flags),
    }


def plateau_agreement_flags(plateau: dict[str, object], rhat_threshold: float = 1.1) -> dict[str, object]:
    flags = []
    for item in plateau["by_coarse"]:
        rhats = {key: item["rhat"][key] for key in OBS}
        finite = [value for value in rhats.values() if value is not None]
        max_rhat = max(finite) if finite else None
        flags.append(
            {
                "coarse_index": item["coarse_index"],
                "max_rhat": max_rhat,
                "agrees_within_errors": max_rhat is not None and max_rhat <= rhat_threshold,
            }
        )
    return {
        "criterion": f"all final-50%-sample Rhat values <= {rhat_threshold}",
        "per_coarse": flags,
        "all_agree_within_errors": all(flag["agrees_within_errors"] for flag in flags),
    }


def write_report(path: Path, result: dict[str, object]) -> None:
    lines = [
        "# Conditional Equilibration of Restricted Detail MCMC",
        "",
        f"Settings: fine_size `{result['fine_size']}`, N_coarse `{result['n_coarse']}`, patch_size `{result['patch_size']}`, step_size `{result['step_size']}`, n_sweeps `{result['n_sweeps']}`, seeds/init `{result['n_seeds']}`.",
        "",
        f"Max fixed-coarse violation over all saved sweeps: `{result['max_fixed_coarse_error']:.6g}`.",
        "",
        "The main burn-in criterion is convergence to a conditional plateau for each fixed `phi_c`. The plateau is estimated from the combined final 25% of sweeps over all initializations and seeds for that `phi_c`. The global true-observable band is retained only as a secondary diagnostic.",
        "",
        "## Per-Coarse Summary",
        "",
        "| coarse | init | first plateau sweeps | median plateau sweep | secondary true-band sweeps | final plateau-distance mean/std | acceptance mean/std |",
        "|---:|---|---|---:|---|---:|---:|",
    ]
    plateau_by_coarse = {item["coarse_index"]: item for item in result["plateau_diagnostics"]["by_coarse"]}
    for item in result["coarse_summary"]:
        plateau_item = plateau_by_coarse[item["coarse_index"]]
        for init in INITS:
            row = item["by_initialization"][init]
            plateau_row = plateau_item["by_initialization"][init]
            firsts = ", ".join("missing" if x is None else str(x) for x in plateau_row["first_plateau_sweeps"])
            true_firsts = ", ".join("missing" if x is None else str(x) for x in row["first_true_band_sweeps"])
            acc = row["mean_acceptance"]
            pd = plateau_row["final_window_plateau_distance"]
            lines.append(
                f"| {item['coarse_index']} | {init} | {firsts} | "
                f"{plateau_row['median_first_plateau_sweep']} | {true_firsts} | "
                f"{pd['mean']:.6g}/{pd['std']:.6g} | "
                f"{acc['mean']:.6g}/{acc['std']:.6g} |"
            )

    med = result["plateau_diagnostics"]["median_first_plateau_sweep_by_initialization"]
    def fmt_missing(x: object) -> str:
        return "missing" if x is None else f"{float(x):.6g}"
    if result["plateau_diagnostics"]["reverse_kl_reduces_time_to_plateau"]:
        burnin_answer = "reverse-KL reduces time-to-plateau by the median first-plateau-sweep criterion"
    else:
        burnin_answer = "reverse-KL does not clearly reduce time-to-plateau by the median first-plateau-sweep criterion"

    lines.extend(
        [
            "",
            "## Plateau Means and Rhat",
            "",
            "| coarse | observable | plateau mean | plateau error | Rhat final 50% |",
            "|---:|---|---:|---:|---:|",
        ]
    )
    for item in result["plateau_diagnostics"]["by_coarse"]:
        for key in OBS:
            pl = item["plateau"][key]
            rh = item["rhat"][key]
            rh_text = "missing" if rh is None else f"{rh:.6g}"
            lines.append(f"| {item['coarse_index']} | {key} | {pl['mean']:.6g} | {pl['error']:.6g} | {rh_text} |")

    lines.extend(
        [
            "",
            "## Answers",
            "",
            f"1. Same conditional distribution independent of initialization? `{result['agreement']['all_agree_within_errors']}` by `{result['agreement']['criterion']}`.",
            f"2. Reverse-KL burn-in: {burnin_answer}. Median first plateau sweep reverse-KL `{fmt_missing(med.get('reverse_kl'))}`, Gaussian `{fmt_missing(med.get('gaussian'))}`, zero-detail `{fmt_missing(med.get('zero_detail'))}`.",
            "3. Remaining discrepancies are diagnosed by Rhat values and by comparing between-initialization spread to between-seed spread. Large Rhat or spread ratios indicate insufficient sweeps or patch-size/step-size choice; ratios near one indicate ordinary observable noise.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def plot(path: Path, result: dict[str, object]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    chains = result["chains"]
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    panels = ["S_f", "phi2", "NN_corr", "high_p_power", "plateau_distance", "patch_acceptance_rate"]
    for ax, key in zip(axes.ravel(), panels):
        for init in INITS:
            selected = [c for c in chains if c["init"] == init and c["seed_index"] == 0]
            y = torch.tensor([[row[key] for row in c["history"]] for c in selected], dtype=torch.float64)
            ax.plot(y.mean(dim=0).numpy(), label=init)
        ax.set_title(key)
        ax.set_xlabel("sweep")
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


PARAMS = Phi4Params()


@torch.no_grad()
def main() -> None:
    global PARAMS
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    PARAMS = Phi4Params(kappa=args.kappa_fine, lam=args.lam)
    phi_f = load_or_generate_fine_configs(
        args.data_path,
        n_configs=max(args.n_configs, args.n_coarse),
        fine_size=args.fine_size,
        params=PARAMS,
        burn_in=args.burn_in,
        interval=args.sample_interval,
        batch_size=args.batch_size,
        proposal_width=args.proposal_width,
        seed=args.seed,
        device=args.device,
    )
    dataset = make_paired_dataset(phi_f)
    phi_c_all, _, true_phi_all = dataset.tensors
    idx = torch.linspace(0, len(dataset) - 1, steps=args.n_coarse).round().long()
    phi_c = phi_c_all[idx].to(device)
    true_target = true_targets(true_phi_all, )
    flow = load_flow(args, device)
    rev_gen = torch.Generator(device=device).manual_seed(args.seed + 11)
    d_reverse_single, _, _, _ = flow.sample_with_decomposition(phi_c, generator=rev_gen)
    detail_shape = (args.n_coarse, args.n_seeds, 3, args.fine_size // 2, args.fine_size // 2)
    d_by_init = {
        "reverse_kl": d_reverse_single[:, None].repeat(1, args.n_seeds, 1, 1, 1),
        "gaussian": torch.randn(detail_shape, generator=torch.Generator(device=device).manual_seed(args.seed + 12), device=device),
        "zero_detail": torch.zeros(detail_shape, device=device),
    }
    chains, max_err = run_batched_chains(phi_c, d_by_init, args, true_target, device)
    plateau = add_plateau_diagnostics(chains, args.n_coarse, args.plateau_window)
    coarse_summary = summarize_by_coarse(chains, args.n_coarse)
    result = {
        "fine_size": args.fine_size,
        "coarse_size": args.fine_size // 2,
        "n_coarse": args.n_coarse,
        "n_seeds": args.n_seeds,
        "patch_size": args.patch_size,
        "step_size": args.step_size,
        "n_sweeps": args.n_sweeps,
        "checkpoint": str(args.checkpoint),
        "max_fixed_coarse_error": max_err,
        "true_targets": true_target,
        "chains": chains,
        "coarse_summary": coarse_summary,
        "plateau_diagnostics": plateau,
        "agreement": plateau_agreement_flags(plateau),
    }
    summary_path = args.output_dir / "conditional_equilibration_summary.json"
    report_path = args.output_dir / "conditional_equilibration_report.md"
    plot_path = args.output_dir / "conditional_equilibration_histories.pdf"
    summary_path.write_text(json.dumps(result, indent=2) + "\n")
    write_report(report_path, result)
    plot(plot_path, result)
    print("max_fixed_coarse_error", f"{max_err:.6g}")
    print("all_agree_within_errors", result["agreement"]["all_agree_within_errors"])
    print(f"wrote {summary_path}")
    print(f"wrote {report_path}")
    print(f"wrote {plot_path}")


if __name__ == "__main__":
    main()
