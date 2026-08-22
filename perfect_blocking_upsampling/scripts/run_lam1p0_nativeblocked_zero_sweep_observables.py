#!/usr/bin/env python3
"""Generate only sweep-0 observables for native L32 blocked to L16 flow samples."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import ks_2samp

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PKG = PROJECT_ROOT / "perfect_blocking_upsampling"
sys.path.insert(0, str(PKG / "src"))
sys.path.insert(0, str(PKG / "scripts"))

from perfect_blocking_upsampling.actions import ActionSpec  # noqa: E402
from perfect_blocking_upsampling.kernels import apply_kernel, load_kernel  # noqa: E402
from run_lam1p0_l16to32_rqspline_zeroshot import (  # noqa: E402
    build_model_from_checkpoint,
    sample_model_lattice,
    stationary_stats,
)
from train_lam1p0_flow_detail_pilot import assemble_psi, load_phi, per_config_rows  # noqa: E402


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def comparison_rows(native: np.ndarray, sample: np.ndarray, action: ActionSpec) -> list[dict]:
    """Per-observable matched-coarse reconstruction test summary."""
    native_main, native_g = per_config_rows(native, action, "native_L32")
    sample_main, sample_g = per_config_rows(sample, action, "flow_sample_from_blocked_L32")
    observables = [
        ("action_density", native_main, sample_main), ("phi2", native_main, sample_main),
        ("phi4", native_main, sample_main), ("local_kurtosis_ratio", native_main, sample_main),
        ("NN", native_main, sample_main), ("2nn", native_main, sample_main),
        ("diag", native_main, sample_main), ("m2", native_main, sample_main),
        ("m4", native_main, sample_main), ("G_pmin_avg", native_g, sample_g),
    ]
    out = []
    for name, ref_rows, flow_rows in observables:
        ref = np.asarray([row[name] for row in ref_rows], dtype=np.float64)
        flow = np.asarray([row[name] for row in flow_rows], dtype=np.float64)
        out.append({
            "observable": name, "n": int(len(ref)),
            "native_mean": float(ref.mean()), "flow_mean": float(flow.mean()),
            "native_std": float(ref.std(ddof=1)), "flow_std": float(flow.std(ddof=1)),
            "std_ratio": float(flow.std(ddof=1) / max(ref.std(ddof=1), 1.0e-300)),
            "mean_shift_native_sigma": float((flow.mean() - ref.mean()) / max(ref.std(ddof=1), 1.0e-300)),
            "ks_statistic": float(ks_2samp(ref, flow).statistic),
        })
    return out


def save_histogram_overlays(
    native: np.ndarray,
    sample: np.ndarray,
    action: ActionSpec,
    out_dir: Path,
    *,
    coarse_mode: str,
) -> None:
    """Write transparent native-vs-flow marginal overlays for the matched-coarse test."""
    native_main, native_g = per_config_rows(native, action, "native_L32")
    sample_main, sample_g = per_config_rows(sample, action, "flow_sample_from_blocked_L32")
    items = [
        ("action_density", "action density", native_main, sample_main),
        ("phi2", r"$\phi^2$", native_main, sample_main),
        ("phi4", r"$\phi^4$", native_main, sample_main),
        ("local_kurtosis_ratio", r"$\langle\phi^4\rangle/\langle\phi^2\rangle^2$", native_main, sample_main),
        ("NN", "NN", native_main, sample_main),
        ("2nn", "2nn", native_main, sample_main),
        ("diag", "diag", native_main, sample_main),
        ("m2", r"$m^2$", native_main, sample_main),
        ("m4", r"$m^4$", native_main, sample_main),
        ("G_pmin_avg", r"$G(p_{\min})$", native_g, sample_g),
    ]
    if coarse_mode == "direct_native":
        flow_label = "flow from direct native L16"
        title_prefix = "Native L32 vs. flow reconstruction from direct native L16"
        file_prefix = "L16_direct_to_L32_flow"
    else:
        flow_label = "flow from blocked L32 → L16"
        title_prefix = "Native L32 vs. flow reconstruction from blocked L32 → L16"
        file_prefix = "L32_blocked_to_L16_flow"
    plot_dir = out_dir / "histograms"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for key, xlabel, native_rows, sample_rows in items:
        ref = np.asarray([row[key] for row in native_rows], dtype=np.float64)
        flow = np.asarray([row[key] for row in sample_rows], dtype=np.float64)
        lo, hi = float(min(ref.min(), flow.min())), float(max(ref.max(), flow.max()))
        pad = 0.15 * max(hi - lo, 1.0e-12)
        bins = np.linspace(lo - pad, hi + pad, 61)
        ks = float(ks_2samp(ref, flow).statistic)
        ratio = float(flow.std(ddof=1) / max(ref.std(ddof=1), 1.0e-300))
        fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
        ax.hist(ref, bins=bins, density=True, histtype="stepfilled", color="0.60", alpha=0.35, label="native L32")
        ax.hist(flow, bins=bins, density=True, histtype="step", color="tab:red", linewidth=1.8, label=flow_label)
        ax.axvline(ref.mean(), color="0.25", linestyle="--", linewidth=1.25)
        ax.axvline(flow.mean(), color="tab:red", linestyle="--", linewidth=1.25)
        ax.set_xlim(bins[0], bins[-1])
        ax.set_xlabel(xlabel, fontsize=13)
        ax.set_ylabel("density")
        ax.set_title(f"{title_prefix}: {xlabel}", fontsize=14)
        ax.text(0.02, 0.95, f"KS = {ks:.4f}\n$\\sigma_{{\\rm flow}}/\\sigma_{{\\rm native}}$ = {ratio:.3f}", transform=ax.transAxes, va="top", ha="left", bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"})
        ax.legend(loc="upper right")
        fig.savefig(plot_dir / f"{file_prefix}_{key}.png", dpi=180)
        fig.savefig(plot_dir / f"{file_prefix}_{key}.pdf")
        plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--native-l32", type=Path, default=Path("data/configs_phi4_2d/lam1p0_kappac0p340301_L32/configs.npz"))
    ap.add_argument("--direct-l16", type=Path, default=Path("data/configs_phi4_2d/lam1p0_kappac0p340301_L16/configs.npz"))
    ap.add_argument("--coarse-mode", choices=("blocked_native", "direct_native"), default="blocked_native")
    ap.add_argument("--checkpoint", type=Path, default=Path("perfect_blocking_upsampling/runs/lam1p0/training/lam1p0_L16to32_newkernel_rqspline_N2000_action_lowtail_from_N2000_bestpatch_20260719T171401Z/checkpoints/checkpoint_best_patch.pt"))
    ap.add_argument("--kernel", type=Path, default=Path("perfect_blocking/perfect_blocking_lam1p0/kernels/final/chosen_kernel.json"))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--save-flow-phi",
        action="store_true",
        help="Save the exact sweep-zero fine fields as sweep0_flow_phi.npz for HMC restart.",
    )
    ap.add_argument("--n-chains", type=int, default=None, help="Use the first N native L32 sources.")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=2026072100)
    args = ap.parse_args()

    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    native_l32 = load_phi(PROJECT_ROOT / args.native_l32).astype(np.float32)
    if args.n_chains is not None:
        if not 0 < args.n_chains <= len(native_l32):
            raise ValueError(f"n-chains must be in [1, {len(native_l32)}]")
        native_l32 = native_l32[: args.n_chains]
    kernel, kernel_meta = load_kernel(PROJECT_ROOT / args.kernel)
    if args.coarse_mode == "blocked_native":
        blocked_psi = apply_kernel(native_l32, kernel).astype(np.float32)
        coarse_l16 = blocked_psi[:, 0::2, 0::2].astype(np.float32)
    else:
        direct_l16 = load_phi(PROJECT_ROOT / args.direct_l16).astype(np.float32)
        if args.n_chains is not None:
            direct_l16 = direct_l16[: args.n_chains]
        if len(direct_l16) < len(native_l32):
            raise ValueError("direct L16 source has fewer configurations than requested native L32 reference")
        coarse_l16 = direct_l16[: len(native_l32)]
    source_idx = np.arange(len(native_l32), dtype=np.int64)

    device = torch.device("cpu")
    ckpt = torch.load(PROJECT_ROOT / args.checkpoint, map_location=device, weights_only=False)
    model, load_report = build_model_from_checkpoint(ckpt, lattice_size=16, device=device)
    stats = stationary_stats(ckpt["state"]["stats"], lc=16)
    detail, logq, zmax, logdet = sample_model_lattice(
        model,
        coarse_l16,
        stats,
        batch_size=args.batch_size,
        device=device,
        seed=args.seed,
    )
    psi = assemble_psi(coarse_l16, detail).astype(np.float32)
    phi_upscaled, inverse_meta = __import__(
        "perfect_blocking_upsampling.kernels", fromlist=["inverse_kernel"]
    ).inverse_kernel(psi, kernel)
    phi_upscaled = phi_upscaled.astype(np.float32)

    action = ActionSpec("phi4_nn", 1.0, 0.340301)
    source_label = "blocked_native_L32" if args.coarse_mode == "blocked_native" else "direct_native_L16"
    main_rows, g_rows = per_config_rows(phi_upscaled, action, source_label)
    ref_main, ref_g = per_config_rows(native_l32, action, "native_L32_reference")
    metric_rows = comparison_rows(native_l32, phi_upscaled, action)

    write_csv(args.out_dir / "flow_sample_observables_per_config.csv", main_rows, list(main_rows[0]))
    write_csv(args.out_dir / "flow_sample_G_per_config.csv", g_rows, list(g_rows[0]))
    write_csv(args.out_dir / "native_L32_observables_per_config.csv", ref_main, list(ref_main[0]))
    write_csv(args.out_dir / "native_L32_G_per_config.csv", ref_g, list(ref_g[0]))
    write_csv(args.out_dir / "comparison_metrics.csv", metric_rows, list(metric_rows[0]))
    if args.save_flow_phi:
        np.savez_compressed(
            args.out_dir / "sweep0_flow_phi.npz",
            phi=phi_upscaled,
            source_indices=source_idx,
        )
    save_histogram_overlays(native_l32, phi_upscaled, action, args.out_dir, coarse_mode=args.coarse_mode)
    manifest = {
        "status": "completed",
        "sweep": 0,
        "observable_only": True,
        "native_l32_source": str(args.native_l32),
        "native_l32_count": int(len(native_l32)),
        "coarse_mode": args.coarse_mode,
        "coarse_construction": "native L32 -> supplied kernel -> retained L16 sites" if args.coarse_mode == "blocked_native" else "direct native L16 source",
        "flow_checkpoint": str(args.checkpoint),
        "kernel": str(args.kernel),
        "kernel_metadata": kernel_meta,
        "load_report": load_report,
        "sample_seed": args.seed,
        "max_abs_z": float(np.max(zmax)),
        "logq_mean": float(np.mean(logq)),
        "logdet_mean": float(np.mean(logdet)),
        "inverse_kernel_metadata": inverse_meta,
        "nonfinite_count": int(np.sum(~np.isfinite(phi_upscaled))),
        "files": [
            "flow_sample_observables_per_config.csv",
            "flow_sample_G_per_config.csv",
            "native_L32_observables_per_config.csv",
            "native_L32_G_per_config.csv",
            "comparison_metrics.csv",
            "histograms/*.png",
            "histograms/*.pdf",
            *( ["sweep0_flow_phi.npz"] if args.save_flow_phi else [] ),
        ],
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"out_dir": str(args.out_dir), "n": len(native_l32), "sweep": 0}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
