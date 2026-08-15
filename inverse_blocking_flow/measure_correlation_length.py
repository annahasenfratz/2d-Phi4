"""MIT-I-style connected correlator and correlation-length diagnostics."""

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

from inverse_blocking_flow.haar import prolong_constant
from inverse_blocking_flow.kappac030_trainable_kappaf_upscale import (
    build_parser as upscale_parser,
    generate_reference,
    load_or_generate_coarse,
)
from inverse_blocking_flow.kappac030_training_depth_study import VARIANTS
from inverse_blocking_flow.phi4 import Phi4Params, binder_cumulant, checkerboard_metropolis_sweep


REF_KAPPAS = [0.320, 0.325, 0.330]
CORRECTION_SWEEPS = [10, 20, 50]


def build_parser() -> argparse.ArgumentParser:
    parser = upscale_parser()
    parser.description = __doc__
    parser.add_argument("--t-min", type=int, default=1)
    parser.add_argument("--t-max", type=int, default=None)
    parser.add_argument("--retrain-missing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include-corrections", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eight-by-eight-only", action="store_true", help="measure only the saved 8x8 coarse configs")
    parser.add_argument("--smoke", action="store_true")
    parser.set_defaults(correction_sweeps="0,10,20,50")
    return parser


def projected_connected_correlator(phi: torch.Tensor) -> torch.Tensor:
    phi = phi.detach().float().cpu()
    batch, ly, lx = phi.shape
    max_t = ly // 2
    curves = []
    for oriented in (phi, phi.transpose(-1, -2)):
        m_t = oriented.mean(dim=-1)
        mean_mt = m_t.mean()
        vals = []
        for dt in range(max_t + 1):
            vals.append((m_t * torch.roll(m_t, shifts=-dt, dims=-1)).mean() - mean_mt.square())
        curves.append(torch.stack(vals))
    return 0.5 * (curves[0] + curves[1])


def effective_mass(c: torch.Tensor) -> list[float | None]:
    out: list[float | None] = []
    for t in range(c.numel() - 1):
        if float(c[t + 1].item()) <= 0.0 or float(c[t].item()) <= 0.0:
            out.append(None)
        else:
            out.append(float(torch.log(c[t] / c[t + 1]).item()))
    return out


def fit_cosh_mass(c: torch.Tensor, t_min: int, t_max: int, lattice_size: int) -> dict[str, float | None]:
    ts = torch.arange(t_min, t_max + 1, dtype=torch.float64)
    ys = c[t_min : t_max + 1].double()
    mask = ys > 0
    ts = ts[mask]
    ys = ys[mask]
    if ts.numel() < 2:
        return {"m_p": None, "amplitude": None, "loss": None}
    best = None
    for m in torch.linspace(0.03, 3.0, 2000, dtype=torch.float64):
        basis = torch.exp(-m * ts) + torch.exp(-m * (lattice_size - ts))
        amp = (ys * basis).sum() / basis.square().sum().clamp_min(1e-30)
        pred = amp * basis
        loss = (torch.log(ys) - torch.log(pred.clamp_min(1e-30))).square().mean()
        if best is None or float(loss.item()) < best["loss"]:
            best = {"m_p": float(m.item()), "amplitude": float(amp.item()), "loss": float(loss.item())}
    return best if best is not None else {"m_p": None, "amplitude": None, "loss": None}


def second_moment_xi(phi: torch.Tensor) -> dict[str, float | None]:
    phi = phi.detach().float().cpu()
    volume = phi.shape[-2] * phi.shape[-1]
    magnetization = phi.mean(dim=(-2, -1))
    chi = volume * (magnetization.square().mean() - magnetization.mean().square())
    fft = torch.fft.fftn(phi, dim=(-2, -1))
    power = (fft.real.square() + fft.imag.square()).mean(dim=0) / (phi.shape[-2] * phi.shape[-1])
    f = 0.5 * (power[1, 0] + power[0, 1])
    if float(f.item()) <= 0.0 or float(chi.item()) <= float(f.item()):
        return {"chi": float(chi.item()), "F": float(f.item()), "xi_2nd": None}
    l = phi.shape[-1]
    xi = (1.0 / (2.0 * torch.sin(torch.tensor(torch.pi / l)))) * torch.sqrt(chi / f - 1.0)
    return {"chi": float(chi.item()), "F": float(f.item()), "xi_2nd": float(xi.item())}


def measure(phi: torch.Tensor, label: str, kappa_label: str, t_min: int, t_max: int | None) -> dict[str, object]:
    l = phi.shape[-1]
    fit_t_max = t_max if t_max is not None else l // 2 - 1
    c = projected_connected_correlator(phi)
    fit = fit_cosh_mass(c, t_min, fit_t_max, l)
    m_p = fit["m_p"]
    xi_pole = None if m_p is None or m_p <= 0 else 1.0 / float(m_p)
    second = second_moment_xi(phi)
    xi_for_ratio = xi_pole if xi_pole is not None else second["xi_2nd"]
    return {
        "label": label,
        "kappa_label": kappa_label,
        "n": int(phi.shape[0]),
        "L": int(l),
        "C_t": [float(x.item()) for x in c],
        "m_eff": effective_mass(c),
        "fit_range": [t_min, fit_t_max],
        "m_p": m_p,
        "xi_pole": xi_pole,
        "xi_2nd": second["xi_2nd"],
        "xi_over_L": None if xi_for_ratio is None else float(xi_for_ratio / l),
        "chi": second["chi"],
        "F": second["F"],
        "binder": float(binder_cumulant(phi).item()),
    }


def cached_variant_path(output_dir: Path, variant_name: str) -> Path:
    return output_dir / f"correlation_input_{variant_name}.pt"


def get_variant_phi(variant, coarse: torch.Tensor, args: argparse.Namespace, device: torch.device) -> tuple[torch.Tensor, float] | None:
    path = cached_variant_path(args.output_dir, variant.name)
    if path.exists():
        data = torch.load(path, map_location="cpu", weights_only=False)
        return data["phi"].float(), float(data["kappa_f"])
    if not args.retrain_missing:
        print(f"skipping missing cached variant ensemble: {path}", flush=True)
        return None
    raise RuntimeError("retraining support is disabled in this measurement-only script")


def corrected(phi: torch.Tensor, kappa: float, sweeps: int, args: argparse.Namespace) -> torch.Tensor:
    out = phi[: args.correction_n].clone()
    gen = torch.Generator().manual_seed(args.seed + int(round(10000 * kappa)) + 1234 + sweeps)
    params = Phi4Params(kappa=kappa, lam=args.lam)
    for _ in range(sweeps):
        checkerboard_metropolis_sweep(out, params, args.proposal_width, gen)
    return out


def write_report(path: Path, summary: dict[str, object]) -> None:
    rows = summary["ensembles"]
    refs = [row for row in rows if row["label"].startswith("reference")]
    variants = [row for row in rows if row["label"].startswith("variant")]
    lines = [
        "# Correlation Length Diagnostics",
        "",
        "Connected projected correlator uses both lattice orientations and periodic boundary conditions. Pole masses are from a one-parameter cosh fit after profiling the amplitude.",
        "",
        "## Summary",
        "",
        "| ensemble | kappa label | m_p | xi_pole | xi_2nd | xi/L | chi | Binder |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['label']} | {row['kappa_label']} | {fmt(row['m_p'])} | {fmt(row['xi_pole'])} | "
            f"{fmt(row['xi_2nd'])} | {fmt(row['xi_over_L'])} | {fmt(row['chi'])} | {fmt(row['binder'])} |"
        )
    if refs and variants:
        ref_by_label = {row["kappa_label"]: row for row in refs}
        lines.extend(["", "## Closest Reference By xi_2nd", "", "| variant | xi_2nd | closest ref | abs diff |", "|---|---:|---:|---:|"])
        for row in variants:
            if row["xi_2nd"] is None:
                continue
            best = min(ref_by_label.values(), key=lambda ref: abs(float(ref["xi_2nd"]) - float(row["xi_2nd"])))
            lines.append(
                f"| {row['label']} | {row['xi_2nd']:.6g} | {best['kappa_label']} | "
                f"{abs(float(best['xi_2nd']) - float(row['xi_2nd'])):.6g} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(x) -> str:
    return "nan" if x is None else f"{float(x):.6g}"


def plot_outputs(path: Path, summary: dict[str, object]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    rows = summary["ensembles"]
    def show_line(row: dict[str, object]) -> bool:
        if len(rows) <= 3:
            return True
        label = str(row["label"])
        return label.startswith("reference") or label in {"naive"} or label.startswith("variant_D")

    with PdfPages(path) as pdf:
        fig, ax = plt.subplots(figsize=(8, 5))
        for row in rows:
            if show_line(row):
                ax.plot(range(len(row["C_t"])), row["C_t"], marker="o", label=row["label"])
        ax.set_yscale("symlog", linthresh=1e-4)
        ax.set_xlabel("t")
        ax.set_ylabel("C(t)")
        ax.legend(fontsize=7)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        for row in rows:
            if show_line(row):
                xs = [i for i, v in enumerate(row["m_eff"]) if v is not None]
                ys = [v for v in row["m_eff"] if v is not None]
                ax.plot(xs, ys, marker="o", label=row["label"])
        ax.set_xlabel("t")
        ax.set_ylabel("m_eff(t)")
        ax.legend(fontsize=7)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 5))
        labels = [row["label"] for row in rows]
        xi = [float("nan") if row["xi_2nd"] is None else row["xi_2nd"] for row in rows]
        ax.bar(labels, xi)
        ax.tick_params(axis="x", rotation=45)
        ax.set_ylabel("xi_2nd")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 5))
        for row in rows:
            if show_line(row) or row["label"].startswith("reference") or row["label"].startswith("variant"):
                try:
                    x = float(row["kappa_label"])
                except ValueError:
                    continue
                y = row["xi_over_L"]
                if y is not None:
                    ax.scatter([x], [y], label=row["label"])
        ax.set_xlabel("kappa/reference label")
        ax.set_ylabel("xi/L")
        ax.legend(fontsize=6)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if args.eight_by_eight_only:
        coarse = load_or_generate_coarse(args)
        rows = [measure(coarse[: args.n_eval], "coarse_8x8_kappa_0.300", f"{args.kappa_c:.3f}", args.t_min, args.t_max)]
        summary = {
            "setup": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "ensembles": rows,
        }
        summary_path = args.output_dir / "correlation_length_8x8_summary.json"
        report_path = args.output_dir / "correlation_length_8x8_report.md"
        plots_path = args.output_dir / "correlation_length_8x8_plots.pdf"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        write_report(report_path, summary)
        plot_outputs(plots_path, summary)
        print(f"wrote {summary_path}")
        print(f"wrote {report_path}")
        print(f"wrote {plots_path}")
        return
    variants = [VARIANTS[0]] if args.smoke else VARIANTS
    coarse = load_or_generate_coarse(args)
    ensembles: list[tuple[str, str, torch.Tensor]] = []
    for kappa in REF_KAPPAS:
        phi = generate_reference(args, kappa)
        ensembles.append((f"reference_{kappa:.3f}", f"{kappa:.3f}", phi[: args.n_eval]))
    naive = prolong_constant(coarse[: args.n_eval])
    ensembles.append(("naive", "coarse_0.300", naive))
    variant_phis = {}
    for variant in variants:
        loaded = get_variant_phi(variant, coarse, args, device)
        if loaded is None:
            continue
        phi, kappa = loaded
        label = f"variant_{variant.name.split('_', 1)[0]}"
        ensembles.append((label, f"{kappa:.3f}", phi[: args.n_eval]))
        variant_phis[label] = (phi, kappa)
    if args.include_corrections:
        for label, (phi, kappa) in variant_phis.items():
            if label not in {"variant_B", "variant_D"}:
                continue
            for sweeps in CORRECTION_SWEEPS:
                ensembles.append((f"{label}_corr{sweeps}", f"{kappa:.3f}", corrected(phi, kappa, sweeps, args)))
    rows = [measure(phi, label, kappa_label, args.t_min, args.t_max) for label, kappa_label, phi in ensembles]
    summary = {
        "setup": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "ensembles": rows,
    }
    summary_path = args.output_dir / "correlation_length_summary.json"
    report_path = args.output_dir / "correlation_length_report.md"
    plots_path = args.output_dir / "correlation_length_plots.pdf"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_report(report_path, summary)
    plot_outputs(plots_path, summary)
    print(f"wrote {summary_path}")
    print(f"wrote {report_path}")
    print(f"wrote {plots_path}")


if __name__ == "__main__":
    main()
