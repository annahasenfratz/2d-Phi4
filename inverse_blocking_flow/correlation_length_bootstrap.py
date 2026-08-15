"""Bootstrap correlation-length diagnostics for fine16 reference and generated ensembles."""

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

from inverse_blocking_flow.data import load_or_generate_fine_configs
from inverse_blocking_flow.haar import prolong_constant
from inverse_blocking_flow.kappac030_trainable_kappaf_upscale import load_or_generate_coarse
from inverse_blocking_flow.kappac030_training_depth_study import VARIANTS
from inverse_blocking_flow.measure_correlation_length import cached_variant_path, corrected, fit_cosh_mass
from inverse_blocking_flow.phi4 import Phi4Params, binder_cumulant


REF_KAPPAS = [0.320, 0.325, 0.330]
BOOT_KEYS = ["xi_2nd", "chi", "binder", "P10", "P01", "P11"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fine-size", type=int, default=16)
    parser.add_argument("--coarse-size", type=int, default=8)
    parser.add_argument("--lambda-f", dest="lam", type=float, default=1.0)
    parser.add_argument("--kappa-c", type=float, default=0.30)
    parser.add_argument("--n-ref", type=int, default=2048)
    parser.add_argument("--n-generated", type=int, default=512)
    parser.add_argument("--n-bootstrap", type=int, default=300)
    parser.add_argument("--reference-burn-in", type=int, default=400)
    parser.add_argument("--sample-interval", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--proposal-width", type=float, default=1.0)
    parser.add_argument("--coarse-data-path", type=Path, default=Path("inverse_blocking_flow/outputs_fine16/coarse_kappac030_configs.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("inverse_blocking_flow/outputs_fine16"))
    parser.add_argument("--correction-n", type=int, default=128)
    parser.add_argument("--t-min", type=int, default=1)
    parser.add_argument("--t-max", type=int, default=None)
    parser.add_argument("--seed", type=int, default=979797)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    return parser


def load_reference(args: argparse.Namespace, kappa: float) -> torch.Tensor:
    tag = str(kappa).replace(".", "p")
    path = args.output_dir / f"fine_reference_bootstrap_kappa_{tag}.pt"
    return load_or_generate_fine_configs(
        path,
        n_configs=args.n_ref,
        fine_size=args.fine_size,
        params=Phi4Params(kappa=kappa, lam=args.lam),
        burn_in=args.reference_burn_in,
        interval=args.sample_interval,
        batch_size=args.batch_size,
        proposal_width=args.proposal_width,
        seed=args.seed + int(round(10000 * kappa)),
        device=args.device,
    ).float()


def ensemble_core(phi: torch.Tensor) -> dict[str, float | list[float] | None]:
    phi = phi.detach().float().cpu()
    n, ly, lx = phi.shape
    volume = ly * lx
    magnetization = phi.mean(dim=(-2, -1))
    chi = volume * (magnetization.square().mean() - magnetization.mean().square())
    fft = torch.fft.fftn(phi, dim=(-2, -1))
    power = (fft.real.square() + fft.imag.square()).mean(dim=0) / volume
    p10 = float(power[1, 0].item())
    p01 = float(power[0, 1].item())
    p11 = float(power[1, 1].item())
    f = 0.5 * (power[1, 0] + power[0, 1])
    xi_2nd = None
    if float(f.item()) > 0.0 and float(chi.item()) > float(f.item()):
        xi = (1.0 / (2.0 * torch.sin(torch.tensor(torch.pi / lx)))) * torch.sqrt(chi / f - 1.0)
        xi_2nd = float(xi.item())
    c_t = projected_connected_correlator(phi)
    m_eff = effective_mass(c_t)
    t_max = lx // 2 - 1
    fit = fit_cosh_mass(c_t, 1, t_max, lx)
    m_p = fit["m_p"]
    xi_pole = None if m_p is None or m_p <= 0 else 1.0 / float(m_p)
    return {
        "n": int(n),
        "xi_2nd": xi_2nd,
        "xi_2nd_over_L": None if xi_2nd is None else xi_2nd / lx,
        "chi": float(chi.item()),
        "binder": float(binder_cumulant(phi).item()),
        "P10": p10,
        "P01": p01,
        "P11": p11,
        "C_t": [float(x.item()) for x in c_t],
        "m_eff": m_eff,
        "m_p": m_p,
        "xi_pole": xi_pole,
    }


def projected_connected_correlator(phi: torch.Tensor) -> torch.Tensor:
    phi = phi.detach().float().cpu()
    max_t = phi.shape[-1] // 2
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
    vals: list[float | None] = []
    for t in range(c.numel() - 1):
        if float(c[t].item()) <= 0.0 or float(c[t + 1].item()) <= 0.0:
            vals.append(None)
        else:
            vals.append(float(torch.log(c[t] / c[t + 1]).item()))
    return vals


def bootstrap(phi: torch.Tensor, n_bootstrap: int, seed: int) -> dict[str, object]:
    base = ensemble_core(phi)
    gen = torch.Generator().manual_seed(seed)
    samples: dict[str, list[float]] = {key: [] for key in BOOT_KEYS}
    c_samples = []
    m_samples = []
    n = phi.shape[0]
    for _ in range(n_bootstrap):
        idx = torch.randint(0, n, (n,), generator=gen)
        row = ensemble_core(phi[idx])
        for key in BOOT_KEYS:
            value = row[key]
            if value is not None:
                samples[key].append(float(value))
        c_samples.append(row["C_t"])
        m_samples.append(row["m_eff"])
    out = {"estimate": base, "bootstrap": {}}
    for key, values in samples.items():
        tensor = torch.tensor(values, dtype=torch.float64)
        out["bootstrap"][key] = {
            "mean": float(tensor.mean().item()) if tensor.numel() else None,
            "stderr": float(tensor.std(unbiased=True).item()) if tensor.numel() > 1 else None,
        }
    c_tensor = torch.tensor(c_samples, dtype=torch.float64)
    out["bootstrap"]["C_t"] = {
        "mean": [float(x) for x in c_tensor.mean(dim=0)],
        "stderr": [float(x) for x in c_tensor.std(dim=0, unbiased=True)],
    }
    max_len = max(len(x) for x in m_samples)
    means = []
    errors = []
    for i in range(max_len):
        vals = [row[i] for row in m_samples if i < len(row) and row[i] is not None]
        if vals:
            tensor = torch.tensor(vals, dtype=torch.float64)
            means.append(float(tensor.mean().item()))
            errors.append(float(tensor.std(unbiased=True).item()) if tensor.numel() > 1 else None)
        else:
            means.append(None)
            errors.append(None)
    out["bootstrap"]["m_eff"] = {"mean": means, "stderr": errors}
    return out


def load_cached_variant(output_dir: Path, variant_name: str) -> tuple[torch.Tensor, float] | None:
    path = cached_variant_path(output_dir, variant_name)
    if not path.exists():
        return None
    data = torch.load(path, map_location="cpu", weights_only=False)
    return data["phi"].float(), float(data["kappa_f"])


def build_ensembles(args: argparse.Namespace) -> list[tuple[str, str, torch.Tensor]]:
    ensembles = []
    for kappa in REF_KAPPAS:
        ensembles.append((f"reference_{kappa:.3f}", f"{kappa:.3f}", load_reference(args, kappa)))
    if not hasattr(args, "n_configs"):
        args.n_configs = args.n_generated
    coarse = load_or_generate_coarse(args)
    ensembles.append(("naive", "coarse_0.300", prolong_constant(coarse[: args.n_generated].unsqueeze(1)[:, 0])))
    cached: dict[str, tuple[torch.Tensor, float]] = {}
    for variant in VARIANTS:
        loaded = load_cached_variant(args.output_dir, variant.name)
        if loaded is None:
            continue
        short = variant.name.split("_", 1)[0]
        phi, kappa = loaded
        cached[short] = (phi[: args.n_generated], kappa)
        ensembles.append((f"variant_{short}", f"{kappa:.3f}", phi[: args.n_generated]))
    if "B" in cached:
        phi_b, kappa_b = cached["B"]
        for sweeps in (10, 20, 50):
            ensembles.append((f"variant_B_corr{sweeps}", f"{kappa_b:.3f}", corrected(phi_b, kappa_b, sweeps, args)))
    return ensembles


def write_report(path: Path, summary: dict[str, object]) -> None:
    rows = summary["ensembles"]
    lines = [
        "# Correlation Length Bootstrap",
        "",
        "Bootstrap errors are over configurations. `xi_2nd/L`, `chi`, Binder, and low-k powers are the primary targets.",
        "",
        "## Summary",
        "",
        "| ensemble | n | kappa | xi_2nd/L | chi | Binder | P10 | P01 | P11 | xi_pole |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        est = row["diagnostics"]["estimate"]
        boot = row["diagnostics"]["bootstrap"]
        lines.append(
            f"| {row['label']} | {est['n']} | {row['kappa_label']} | {fmt_pm(est['xi_2nd_over_L'], boot['xi_2nd']['stderr'], scale=1.0 / 16.0)} | "
            f"{fmt_pm(est['chi'], boot['chi']['stderr'])} | {fmt_pm(est['binder'], boot['binder']['stderr'])} | "
            f"{fmt_pm(est['P10'], boot['P10']['stderr'])} | {fmt_pm(est['P01'], boot['P01']['stderr'])} | "
            f"{fmt_pm(est['P11'], boot['P11']['stderr'])} | {fmt(est['xi_pole'])} |"
        )
    refs = [row for row in rows if row["label"].startswith("reference")]
    ref_xi = [row["diagnostics"]["estimate"]["xi_2nd"] for row in refs]
    mono = None if any(x is None for x in ref_xi) else all(ref_xi[i] <= ref_xi[i + 1] for i in range(len(ref_xi) - 1))
    lines.extend(["", "## Monotonicity", "", f"Reference xi_2nd is monotonic increasing over 0.320, 0.325, 0.330: `{mono}`."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(x) -> str:
    return "nan" if x is None else f"{float(x):.6g}"


def fmt_pm(x, err, scale: float = 1.0) -> str:
    if x is None:
        return "nan"
    if err is None:
        return f"{float(x):.6g}"
    return f"{float(x):.6g} +/- {float(err) * scale:.3g}"


def plot_outputs(path: Path, summary: dict[str, object]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    rows = summary["ensembles"]
    with PdfPages(path) as pdf:
        fig, ax = plt.subplots(figsize=(8, 5))
        for row in rows:
            if row["label"].startswith("reference") or row["label"] in {"naive", "variant_A", "variant_B"}:
                est = row["diagnostics"]["estimate"]
                ax.errorbar(range(len(est["C_t"])), est["C_t"], yerr=row["diagnostics"]["bootstrap"]["C_t"]["stderr"], marker="o", label=row["label"])
        ax.set_yscale("symlog", linthresh=1e-4)
        ax.set_xlabel("t")
        ax.set_ylabel("C(t)")
        ax.legend(fontsize=7)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        for row in rows:
            if row["label"].startswith("reference") or row["label"] in {"naive", "variant_A", "variant_B"}:
                mean = row["diagnostics"]["bootstrap"]["m_eff"]["mean"]
                err = row["diagnostics"]["bootstrap"]["m_eff"]["stderr"]
                xs = [i for i, v in enumerate(mean) if v is not None]
                ys = [v for v in mean if v is not None]
                es = [float("nan") if err[i] is None else err[i] for i in xs]
                ax.errorbar(xs, ys, yerr=es, marker="o", label=row["label"])
        ax.set_xlabel("t")
        ax.set_ylabel("m_eff(t)")
        ax.legend(fontsize=7)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 5))
        labels = [row["label"] for row in rows]
        y = [
            float("nan") if row["diagnostics"]["estimate"]["xi_2nd_over_L"] is None else row["diagnostics"]["estimate"]["xi_2nd_over_L"]
            for row in rows
        ]
        e = [
            float("nan") if row["diagnostics"]["bootstrap"]["xi_2nd"]["stderr"] is None else row["diagnostics"]["bootstrap"]["xi_2nd"]["stderr"] / 16.0
            for row in rows
        ]
        ax.bar(labels, y, yerr=e)
        ax.tick_params(axis="x", rotation=45)
        ax.set_ylabel("xi_2nd/L")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        refs = [row for row in rows if row["label"].startswith("reference")]
        ax.errorbar(
            [float(row["kappa_label"]) for row in refs],
            [
                float("nan") if row["diagnostics"]["estimate"]["xi_2nd_over_L"] is None else row["diagnostics"]["estimate"]["xi_2nd_over_L"]
                for row in refs
            ],
            yerr=[
                float("nan") if row["diagnostics"]["bootstrap"]["xi_2nd"]["stderr"] is None else row["diagnostics"]["bootstrap"]["xi_2nd"]["stderr"] / 16.0
                for row in refs
            ],
            marker="o",
            label="reference",
        )
        ax.set_xlabel("kappa_f")
        ax.set_ylabel("xi_2nd/L")
        ax.legend()
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def corrected(phi: torch.Tensor, kappa: float, sweeps: int, args: argparse.Namespace) -> torch.Tensor:
    from inverse_blocking_flow.phi4 import checkerboard_metropolis_sweep

    out = phi[: args.correction_n].clone()
    gen = torch.Generator().manual_seed(args.seed + int(round(10000 * kappa)) + 1234 + sweeps)
    params = Phi4Params(kappa=kappa, lam=args.lam)
    for _ in range(sweeps):
        checkerboard_metropolis_sweep(out, params, args.proposal_width, gen)
    return out


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ensembles = build_ensembles(args)
    rows = []
    for idx, (label, kappa_label, phi) in enumerate(ensembles):
        print(f"measuring {label} n={phi.shape[0]}", flush=True)
        rows.append(
            {
                "label": label,
                "kappa_label": kappa_label,
                "diagnostics": bootstrap(phi, args.n_bootstrap, args.seed + 1009 * idx),
            }
        )
    summary = {
        "setup": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "ensembles": rows,
    }
    summary_path = args.output_dir / "correlation_length_bootstrap_summary.json"
    report_path = args.output_dir / "correlation_length_bootstrap_report.md"
    plots_path = args.output_dir / "correlation_length_bootstrap_plots.pdf"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_report(report_path, summary)
    plot_outputs(plots_path, summary)
    print(f"wrote {summary_path}")
    print(f"wrote {report_path}")
    print(f"wrote {plots_path}")


if __name__ == "__main__":
    main()
