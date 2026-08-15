"""Cluster-algorithm validation of phi4 reference ensembles."""

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

from inverse_blocking_flow.correlation_length_bootstrap import bootstrap as corr_bootstrap, ensemble_core
from inverse_blocking_flow.kappac030_observable_loss import batch_observables, scalar_observables
from inverse_blocking_flow.logw_kappa_scan_blocked_coarse import ensemble_summary
from inverse_blocking_flow.phi4 import Phi4Params, checkerboard_metropolis_sweep, generate_phi4_configs, phi4_action


ENSEMBLES = [
    {"name": "L8_k0300", "L": 8, "kappa": 0.300, "path": "cluster_refs_L8_k0300.pt"},
    {"name": "L16_k0320", "L": 16, "kappa": 0.320, "path": "cluster_refs_L16_k0320.pt"},
    {"name": "L16_k0325", "L": 16, "kappa": 0.325, "path": "cluster_refs_L16_k0325.pt"},
    {"name": "L16_k0330", "L": 16, "kappa": 0.330, "path": "cluster_refs_L16_k0330.pt"},
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lambda", dest="lam", type=float, default=1.0)
    parser.add_argument("--n-configs", type=int, default=2048)
    parser.add_argument("--n-therm", type=int, default=400)
    parser.add_argument("--interval", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--local-proposal-width", type=float, default=1.0)
    parser.add_argument("--cluster-flips-per-sweep", type=int, default=1)
    parser.add_argument("--n-bootstrap", type=int, default=240)
    parser.add_argument("--output-dir", type=Path, default=Path("inverse_blocking_flow/outputs_fine16"))
    parser.add_argument("--seed", type=int, default=313202)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser


def wolff_bond_probability(phi_x: torch.Tensor, phi_y: torch.Tensor, kappa: float) -> torch.Tensor:
    return 1.0 - torch.exp(-4.0 * kappa * (phi_x * phi_y).abs())


def onsite_action_density(phi: torch.Tensor, lam: float) -> torch.Tensor:
    return phi.square() + lam * (phi.square() - 1.0).square()


@torch.no_grad()
def wolff_cluster_flip_single(phi: torch.Tensor, kappa: float, generator: torch.Generator) -> int:
    size_y, size_x = phi.shape
    sy = int(torch.randint(0, size_y, (1,), generator=generator, device=phi.device).item())
    sx = int(torch.randint(0, size_x, (1,), generator=generator, device=phi.device).item())
    seed_sign = 1.0 if float(phi[sy, sx].item()) >= 0.0 else -1.0
    in_cluster = torch.zeros((size_y, size_x), dtype=torch.bool, device=phi.device)
    in_cluster[sy, sx] = True
    stack = [(sy, sx)]
    while stack:
        y, x = stack.pop()
        value = phi[y, x]
        for yy, xx in (((y + 1) % size_y, x), ((y - 1) % size_y, x), (y, (x + 1) % size_x), (y, (x - 1) % size_x)):
            if bool(in_cluster[yy, xx].item()):
                continue
            neigh_sign = 1.0 if float(phi[yy, xx].item()) >= 0.0 else -1.0
            if neigh_sign != seed_sign:
                continue
            p = wolff_bond_probability(value, phi[yy, xx], kappa)
            if bool((torch.rand((), generator=generator, device=phi.device) < p).item()):
                in_cluster[yy, xx] = True
                stack.append((yy, xx))
    phi[in_cluster] = -phi[in_cluster]
    return int(in_cluster.sum().item())


@torch.no_grad()
def cluster_sweep(phi: torch.Tensor, params: Phi4Params, proposal_width: float, cluster_flips: int, generator: torch.Generator) -> dict[str, float]:
    local_acceptance = checkerboard_metropolis_sweep(phi, params, proposal_width, generator)
    sizes = []
    for _ in range(cluster_flips):
        for i in range(phi.shape[0]):
            sizes.append(wolff_cluster_flip_single(phi[i], params.kappa, generator))
    return {"local_acceptance": local_acceptance, "cluster_size_mean": float(sum(sizes) / len(sizes)), "cluster_size_max": float(max(sizes))}


@torch.no_grad()
def generate_cluster_configs(
    n_configs: int,
    size: int,
    params: Phi4Params,
    *,
    n_therm: int,
    interval: int,
    batch_size: int,
    proposal_width: float,
    cluster_flips: int,
    seed: int,
    device: str,
) -> tuple[torch.Tensor, dict[str, object]]:
    generator = torch.Generator(device=device).manual_seed(seed)
    phi = 0.5 * torch.randn((batch_size, size, size), generator=generator, device=device)
    local_acc = []
    cluster_sizes = []
    for _ in range(n_therm):
        row = cluster_sweep(phi, params, proposal_width, cluster_flips, generator)
        local_acc.append(row["local_acceptance"])
        cluster_sizes.append(row["cluster_size_mean"])
    chunks = []
    saved = 0
    while saved < n_configs:
        for _ in range(interval):
            row = cluster_sweep(phi, params, proposal_width, cluster_flips, generator)
            local_acc.append(row["local_acceptance"])
            cluster_sizes.append(row["cluster_size_mean"])
        chunks.append(phi.detach().cpu().clone())
        saved += phi.shape[0]
    configs = torch.cat(chunks, dim=0)[:n_configs]
    diag = {
        "thermalization_sweeps": n_therm,
        "interval": interval,
        "mean_local_acceptance": float(torch.tensor(local_acc).mean().item()),
        "mean_cluster_size": float(torch.tensor(cluster_sizes).mean().item()),
        "max_recorded_cluster_size_mean": float(torch.tensor(cluster_sizes).max().item()),
    }
    return configs, diag


def autocorr_summary(values: list[float]) -> dict[str, float | None]:
    if len(values) < 4:
        return {"lag_1": None, "tau_int_initial_positive": None}
    x = torch.tensor(values, dtype=torch.float64)
    x = x - x.mean()
    var = float(x.square().mean())
    if var <= 1e-16:
        return {"lag_1": 0.0, "tau_int_initial_positive": 0.5}
    max_lag = min(100, len(values) - 1)
    ac = []
    for lag in range(max_lag + 1):
        ac.append(float((x[: len(x) - lag] * x[lag:]).mean() / var))
    tau = 0.5
    for val in ac[1:]:
        if val <= 0.0:
            break
        tau += val
    return {"lag_1": ac[1] if len(ac) > 1 else None, "tau_int_initial_positive": float(tau)}


def time_series_observables(phi: torch.Tensor, params: Phi4Params) -> dict[str, list[float]]:
    action = phi4_action(phi, params) / float(phi.shape[-2] * phi.shape[-1])
    mag = phi.mean(dim=(-2, -1))
    return {
        "M": [float(x) for x in mag],
        "M2": [float(x) for x in mag.square()],
        "S_density": [float(x) for x in action],
        "phi2": [float(x) for x in phi.square().mean(dim=(-2, -1))],
    }


def bootstrap_scalar_observables(phi: torch.Tensor, params: Phi4Params, n_bootstrap: int, seed: int) -> dict[str, object]:
    base = scalar_observables(phi, params.kappa, params.lam)
    mag = phi.mean(dim=(-2, -1))
    extras = {
        "M": float(mag.mean()),
        "absM": float(mag.abs().mean()),
        "M2": float(mag.square().mean()),
        "M4": float(mag.pow(4).mean()),
    }
    base = {**base, **extras}
    gen = torch.Generator().manual_seed(seed)
    samples = {key: [] for key in base}
    n = phi.shape[0]
    for _ in range(n_bootstrap):
        idx = torch.randint(0, n, (n,), generator=gen)
        row = scalar_observables(phi[idx], params.kappa, params.lam)
        m = phi[idx].mean(dim=(-2, -1))
        row.update({"M": float(m.mean()), "absM": float(m.abs().mean()), "M2": float(m.square().mean()), "M4": float(m.pow(4).mean())})
        for key, value in row.items():
            samples[key].append(value)
    return {
        key: {
            "mean": value,
            "stderr": float(torch.tensor(samples[key], dtype=torch.float64).std(unbiased=True).item()),
        }
        for key, value in base.items()
    }


def compare_local_cached(args: argparse.Namespace, size: int, kappa: float, cluster_obs: dict[str, object]) -> dict[str, object] | None:
    if size == 8:
        path = args.output_dir / "coarse_kappac030_configs.pt"
    else:
        tag = str(kappa).replace(".", "p")
        path = args.output_dir / f"fine_reference_bootstrap_kappa_{tag}.pt"
    if not path.exists():
        return None
    data = torch.load(path, map_location="cpu", weights_only=False)
    phi = data["phi"] if isinstance(data, dict) else data
    phi = phi.float()
    if phi.shape[-1] != size:
        return None
    params = Phi4Params(kappa=kappa, lam=args.lam)
    corr = corr_bootstrap(phi, min(args.n_bootstrap, 120), args.seed + int(10000 * kappa) + size)
    return {
        "path": str(path),
        "n": int(phi.shape[0]),
        "correlation": corr,
        "observables": bootstrap_scalar_observables(phi, params, min(args.n_bootstrap, 120), args.seed + 55),
        "xi_2nd_over_L_difference_vs_cluster": None
        if corr["estimate"]["xi_2nd_over_L"] is None or cluster_obs["correlation"]["estimate"]["xi_2nd_over_L"] is None
        else float(corr["estimate"]["xi_2nd_over_L"] - cluster_obs["correlation"]["estimate"]["xi_2nd_over_L"]),
    }


def run_one(args: argparse.Namespace, spec: dict[str, object]) -> dict[str, object]:
    params = Phi4Params(kappa=float(spec["kappa"]), lam=args.lam)
    path = args.output_dir / str(spec["path"])
    if path.exists() and not args.force:
        data = torch.load(path, map_location="cpu", weights_only=False)
        phi = data["phi"] if isinstance(data, dict) else data
        gen_diag = data.get("generation_diagnostics", {}) if isinstance(data, dict) else {}
    else:
        phi, gen_diag = generate_cluster_configs(
            args.n_configs,
            int(spec["L"]),
            params,
            n_therm=args.n_therm,
            interval=args.interval,
            batch_size=args.batch_size,
            proposal_width=args.local_proposal_width,
            cluster_flips=args.cluster_flips_per_sweep,
            seed=args.seed + int(10000 * float(spec["kappa"])) + int(spec["L"]),
            device=args.device,
        )
        torch.save({"phi": phi, "params": params.__dict__, "generation_diagnostics": gen_diag}, path)
    phi = phi[: args.n_configs].float()
    corr = corr_bootstrap(phi, args.n_bootstrap, args.seed + int(10000 * float(spec["kappa"])))
    obs = bootstrap_scalar_observables(phi, params, args.n_bootstrap, args.seed + int(100000 * float(spec["kappa"])))
    ts = time_series_observables(phi, params)
    autocorr = {key: autocorr_summary(vals) for key, vals in ts.items()}
    row: dict[str, object] = {
        "name": spec["name"],
        "L": int(spec["L"]),
        "kappa": float(spec["kappa"]),
        "lambda": args.lam,
        "path": str(path),
        "n": int(phi.shape[0]),
        "generation_diagnostics": gen_diag,
        "observables": obs,
        "correlation": corr,
        "autocorrelation": autocorr,
    }
    row["local_cached_comparison"] = compare_local_cached(args, int(spec["L"]), float(spec["kappa"]), row)
    return row


def write_report(path: Path, summary: dict[str, object]) -> None:
    rows = summary["ensembles"]
    lines = [
        "# Cluster Phi4 Reference Check",
        "",
        "Embedded Wolff sign clusters are alternated with local Metropolis sweeps.",
        "",
        "| ensemble | L | kappa | n | xi_2nd/L | chi | Binder | phi2 | S_density | tau M2 | local xi/L diff |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        corr = row["correlation"]
        boot = corr["bootstrap"]
        obs = row["observables"]
        local = row.get("local_cached_comparison")
        diff = None if local is None else local["xi_2nd_over_L_difference_vs_cluster"]
        xi_err = boot["xi_2nd"]["stderr"]
        xi = corr["estimate"]["xi_2nd_over_L"]
        xi_text = "nan" if xi is None else f"{xi:.6g} +/- {(xi_err / row['L']) if xi_err is not None else float('nan'):.3g}"
        lines.append(
            f"| {row['name']} | {row['L']} | {row['kappa']:.6g} | {row['n']} | "
            f"{xi_text} | "
            f"{corr['estimate']['chi']:.6g} | {corr['estimate']['binder']:.6g} | "
            f"{obs['phi2']['mean']:.6g} | {obs['S_density']['mean']:.6g} | "
            f"{row['autocorrelation']['M2']['tau_int_initial_positive']} | "
            f"{'' if diff is None else f'{diff:.6g}'} |"
        )
    monotonic = summary["monotonic_L16_xi"]
    lines.extend(
        [
            "",
            "## Checks",
            "",
            f"- L=8 kappa=0.300 xi/L vs previous 0.3655: `{summary['L8_xi_minus_previous']}`.",
            f"- L=16 xi/L monotonic in kappa: `{monotonic}`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_outputs(path: Path, summary: dict[str, object]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    rows = summary["ensembles"]
    labels = [row["name"] for row in rows]
    with PdfPages(path) as pdf:
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        axes[0, 0].bar(labels, [row["correlation"]["estimate"]["xi_2nd_over_L"] or 0.0 for row in rows])
        axes[0, 0].set_ylabel("xi_2nd/L")
        axes[0, 1].bar(labels, [row["correlation"]["estimate"]["chi"] for row in rows])
        axes[0, 1].set_ylabel("chi")
        axes[1, 0].bar(labels, [row["correlation"]["estimate"]["binder"] for row in rows])
        axes[1, 0].set_ylabel("Binder")
        axes[1, 1].bar(labels, [row["generation_diagnostics"].get("mean_cluster_size", 0.0) for row in rows])
        axes[1, 1].set_ylabel("mean cluster size")
        for ax in axes.ravel():
            ax.tick_params(axis="x", rotation=25)
            ax.grid(alpha=0.2, axis="y")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4.5))
        l16 = [row for row in rows if row["L"] == 16]
        ax.errorbar(
            [row["kappa"] for row in l16],
            [row["correlation"]["estimate"]["xi_2nd_over_L"] or 0.0 for row in l16],
            yerr=[
                (row["correlation"]["bootstrap"]["xi_2nd"]["stderr"] or 0.0) / row["L"]
                for row in l16
            ],
            marker="o",
        )
        ax.set_xlabel("kappa")
        ax.set_ylabel("L=16 xi_2nd/L")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    if args.smoke:
        args.n_configs = 32
        args.n_therm = 4
        args.interval = 1
        args.batch_size = 8
        args.n_bootstrap = 8
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for spec in ENSEMBLES:
        row = run_one(args, spec)
        rows.append(row)
        xi = row["correlation"]["estimate"]["xi_2nd_over_L"]
        xi_text = "nan" if xi is None else f"{xi:.6g}"
        print(
            f"{row['name']} xi/L={xi_text} "
            f"chi={row['correlation']['estimate']['chi']:.6g} binder={row['correlation']['estimate']['binder']:.6g}",
            flush=True,
        )
    l8 = next(row for row in rows if row["L"] == 8)
    l16 = sorted([row for row in rows if row["L"] == 16], key=lambda row: row["kappa"])
    xis = [row["correlation"]["estimate"]["xi_2nd_over_L"] for row in l16]
    summary = {
        "setup": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "ensembles": rows,
        "L8_xi_minus_previous": None
        if l8["correlation"]["estimate"]["xi_2nd_over_L"] is None
        else float(l8["correlation"]["estimate"]["xi_2nd_over_L"] - 0.3655),
        "monotonic_L16_xi": bool(all(xis[i] is not None and xis[i + 1] is not None and xis[i] < xis[i + 1] for i in range(len(xis) - 1))),
    }
    summary_path = args.output_dir / "cluster_reference_check_summary.json"
    report_path = args.output_dir / "cluster_reference_check_report.md"
    plots_path = args.output_dir / "cluster_reference_check_plots.pdf"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_report(report_path, summary)
    plot_outputs(plots_path, summary)
    print(f"wrote {summary_path}")
    print(f"wrote {report_path}")
    print(f"wrote {plots_path}")


if __name__ == "__main__":
    main()
