#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PKG = PROJECT_ROOT / "perfect_blocking_upsampling"
SCRIPT_DIR = PKG / "scripts"
sys.path.insert(0, str(PKG / "src"))
sys.path.insert(0, str(SCRIPT_DIR))
os.environ.setdefault("MPLCONFIGDIR", str((PKG / "logs" / "mplconfig").resolve()))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats

from perfect_blocking_upsampling.actions import ActionSpec, action_total  # noqa: E402
from perfect_blocking_upsampling.kernels import apply_kernel, inverse_kernel, kernel_stencil_from_spec, load_kernel  # noqa: E402
from run_lam1p0_l16to32_rqspline_zeroshot import build_model_from_checkpoint, sample_model_lattice, stationary_stats, per_config_observables  # noqa: E402
from train_lam1p0_flow_detail_pilot import assemble_psi, load_phi  # noqa: E402


DEFAULT_FLOW = PROJECT_ROOT / "perfect_blocking_upsampling/runs/lam1p0/lam1p0_L8to16_kf0p340301_kc0p340301_7x7_phi2_nn_guarded_autoregressive_detail_8layer48_rqspline_localreg_from_affine_ep137_20260717T125835Z/checkpoints/checkpoint_best.pt"
DEFAULT_KERNEL = PROJECT_ROOT / "perfect_blocking/perfect_blocking_lam1p0/kernels/final/chosen_kernel.json"
DEFAULT_NATIVE_L8 = PROJECT_ROOT / "data/configs_phi4_2d/lam1p0_kappac0p340301_L8/configs.npz"
DEFAULT_NATIVE_L16 = PROJECT_ROOT / "data/configs_phi4_2d/lam1p0_kappac0p340301_L16/configs.npz"
DEFAULT_OUT = PROJECT_ROOT / "perfect_blocking/perfect_blocking_lam1p0/tests/final/small_mit_nf_L8to16_flow_proposal_check_20260720"
OBS = ["action_density", "phi2", "phi4", "local_kurtosis_ratio", "NN"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        columns = []
        for row in rows:
            for key in row:
                if key not in columns:
                    columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        fh.flush()
        os.fsync(fh.fileno())


def histogram_edges(samples: list[np.ndarray]) -> np.ndarray:
    x = np.concatenate([np.asarray(s)[np.isfinite(s)] for s in samples])
    if float(np.min(x)) == float(np.max(x)):
        return np.linspace(float(np.min(x)) - 0.5, float(np.max(x)) + 0.5, 61)
    return np.linspace(float(np.min(x)), float(np.max(x)), 81)


def make_histograms(native_obs: dict[str, np.ndarray], proposal_obs: dict[str, np.ndarray], out_dir: Path) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for key in OBS:
        bins = histogram_edges([native_obs[key], proposal_obs[key]])
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        ax.hist(native_obs[key], bins=bins, density=True, histtype="step", lw=2.2, color="black", label="native L16")
        ax.hist(proposal_obs[key], bins=bins, density=True, histtype="step", lw=1.8, color="tab:blue", label="raw flow L8->L16")
        ax.set_xlabel(key)
        ax.set_ylabel("density")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(fig_dir / f"{key}_native_vs_raw_flow.pdf")
        plt.close(fig)
        print(f"wrote {fig_dir / f'{key}_native_vs_raw_flow.pdf'}", flush=True)


def metric_rows(native_obs: dict[str, np.ndarray], proposal_obs: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    rows = []
    for key in OBS:
        a = proposal_obs[key]
        b = native_obs[key]
        bins = histogram_edges([a, b])
        ca, _ = np.histogram(a, bins=bins)
        cb, _ = np.histogram(b, bins=bins)
        pa = ca.astype(float) / max(1, ca.sum())
        pb = cb.astype(float) / max(1, cb.sum())
        tv = 0.5 * float(np.sum(np.abs(pa - pb)))
        ovl = float(np.sum(np.minimum(pa, pb)))
        m = 0.5 * (pa + pb)
        js = 0.5 * float(np.sum(np.where(pa > 0, pa * np.log2(pa / np.maximum(m, 1e-300)), 0.0)))
        js += 0.5 * float(np.sum(np.where(pb > 0, pb * np.log2(pb / np.maximum(m, 1e-300)), 0.0)))
        rows.append(
            {
                "observable": key,
                "native_mean": float(np.mean(b)),
                "proposal_mean": float(np.mean(a)),
                "native_std": float(np.std(b, ddof=1)),
                "proposal_std": float(np.std(a, ddof=1)),
                "shift_native_sigma": float((np.mean(a) - np.mean(b)) / np.std(b, ddof=1)),
                "std_ratio": float(np.std(a, ddof=1) / np.std(b, ddof=1)),
                "KS": float(scipy_stats.ks_2samp(a, b).statistic),
                "TV": tv,
                "JS": js,
                "OVL": ovl,
            }
        )
    return rows


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator, nboot: int = 2000) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    means = np.empty(nboot, dtype=np.float64)
    for b in range(nboot):
        pick = rng.integers(0, len(values), size=len(values))
        means[b] = float(np.mean(values[pick]))
    return float(np.mean(values)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--flow-checkpoint", type=Path, default=DEFAULT_FLOW)
    ap.add_argument("--kernel", type=Path, default=DEFAULT_KERNEL)
    ap.add_argument("--native-l8", type=Path, default=DEFAULT_NATIVE_L8)
    ap.add_argument("--native-l16", type=Path, default=DEFAULT_NATIVE_L16)
    ap.add_argument("--n", type=int, default=None, help="Legacy total proposal count. Prefer --n-coarse and --proposals-per-coarse.")
    ap.add_argument("--n-coarse", type=int, default=8)
    ap.add_argument("--proposals-per-coarse", type=int, default=16)
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20260720)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--skip-histograms", action="store_true")
    args = ap.parse_args()
    if args.n is not None:
        if args.n % args.n_coarse != 0:
            raise ValueError("--n must be divisible by --n-coarse")
        args.proposals_per_coarse = args.n // args.n_coarse
    if args.proposals_per_coarse < 2:
        raise ValueError("--proposals-per-coarse must be at least 2")
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"writing to {out_dir}", flush=True)
    action = ActionSpec("phi4_nn", 1.0, 0.340301)
    kernel, _kernel_json = load_kernel(args.kernel)
    ckpt = torch.load(args.flow_checkpoint, map_location="cpu", weights_only=False)
    model, load_report = build_model_from_checkpoint(ckpt, lattice_size=8, device=torch.device("cpu"))
    stats = stationary_stats(ckpt["state"]["stats"], lc=8)
    l8 = load_phi(args.native_l8)
    l16 = load_phi(args.native_l16)
    idx = np.arange(args.start_index, args.start_index + args.n_coarse, dtype=np.int64)
    coarse_unique = l8[idx].astype(np.float32)
    coarse = np.repeat(coarse_unique, args.proposals_per_coarse, axis=0)
    native16 = l16[idx].astype(np.float32)
    n_total = int(len(coarse))
    print("loaded data and checkpoint", flush=True)

    detail, logq, zmax, logj = sample_model_lattice(model, coarse, stats, batch_size=args.batch_size, device=torch.device("cpu"), seed=args.seed)
    psi = assemble_psi(coarse, detail).astype(np.float32)
    phi, _ = inverse_kernel(psi, kernel)
    blocked = apply_kernel(phi, kernel)[:, 0::2, 0::2]
    reblock_max = float(np.max(np.abs(blocked.astype(np.float64) - coarse.astype(np.float64))))
    sf = action_total(phi, action).astype(np.float64)
    log_gaussian = logq + logj
    nonfinite = np.sum(~np.isfinite(phi).reshape(len(phi), -1), axis=1)
    print(f"generated {len(phi)} proposals from {args.n_coarse} fixed coarse fields; reblocking max {reblock_max:.6g}", flush=True)

    proposal_rows = []
    for i in range(n_total):
        coarse_i = i // args.proposals_per_coarse
        proposal_i = i % args.proposals_per_coarse
        proposal_rows.append(
            {
                "proposal_index": i,
                "coarse_slot": int(coarse_i),
                "proposal_within_coarse": int(proposal_i),
                "coarse_config_index": int(idx[coarse_i]),
                "S_f": float(sf[i]),
                "action_density": float(sf[i] / (16 * 16)),
                "log_gaussian_density": float(log_gaussian[i]),
                "log_forward_jacobian": float(logj[i]),
                "log_proposal_density": float(logq[i]),
                "max_abs_z": float(zmax[i]),
                "nonfinite_count": int(nonfinite[i]),
            }
        )
    write_csv(out_dir / "proposal_diagnostics.csv", proposal_rows)
    print("wrote proposal_diagnostics.csv", flush=True)

    pair_rows = []
    accepted = 0
    coarse_rows = []
    coarse_obs = per_config_observables(coarse_unique, action)[0]
    for coarse_i in range(args.n_coarse):
        base = coarse_i * args.proposals_per_coarse
        coarse_pair_acc = []
        coarse_pair_logr = []
        for local_pair in range(args.proposals_per_coarse // 2):
            old = base + 2 * local_pair
            new = old + 1
            logR = -sf[new] + sf[old] + logq[old] - logq[new]
            acc_prob = min(1.0, math.exp(min(0.0, float(logR))))
            accept_est = float(logR >= 0.0)
            accepted += int(accept_est)
            coarse_pair_acc.append(acc_prob)
            coarse_pair_logr.append(float(logR))
            pair_rows.append(
                {
                    "pair": len(pair_rows),
                    "coarse_slot": coarse_i,
                    "coarse_config_index": int(idx[coarse_i]),
                    "pair_within_coarse": local_pair,
                    "old_proposal_index": old,
                    "new_proposal_index": new,
                    "S_f_old": float(sf[old]),
                    "S_f_new": float(sf[new]),
                    "log_q_old": float(logq[old]),
                    "log_q_new": float(logq[new]),
                    "logR": float(logR),
                    "acceptance_probability": acc_prob,
                    "accepted_by_threshold_logR_ge_0": accept_est,
                    "coarse_action_density": float(action_total(coarse_unique[coarse_i : coarse_i + 1], action)[0] / (8 * 8)),
                    "coarse_phi2": float(coarse_obs["phi2"][coarse_i]),
                    "note": "conditional fixed-coarse proposal pair; both proposals share the same coarse field",
                }
            )
        pbar = float(np.mean(coarse_pair_acc))
        coarse_rows.append(
            {
                "coarse_slot": coarse_i,
                "coarse_config_index": int(idx[coarse_i]),
                "n_pairs": len(coarse_pair_acc),
                "acceptance_probability_mean": pbar,
                "acceptance_binomial_se": float(math.sqrt(max(0.0, pbar * (1.0 - pbar)) / max(1, len(coarse_pair_acc)))),
                "logR_mean": float(np.mean(coarse_pair_logr)),
                "logR_std": float(np.std(coarse_pair_logr, ddof=1)) if len(coarse_pair_logr) > 1 else 0.0,
                "coarse_action_density": float(action_total(coarse_unique[coarse_i : coarse_i + 1], action)[0] / (8 * 8)),
                "coarse_phi2": float(coarse_obs["phi2"][coarse_i]),
            }
        )
    write_csv(out_dir / "conditional_fixed_coarse_metropolis_logratios.csv", pair_rows)
    write_csv(out_dir / "acceptance_by_coarse_config.csv", coarse_rows)
    acc_values = np.asarray([r["acceptance_probability"] for r in pair_rows], dtype=np.float64)
    logr_values = np.asarray([r["logR"] for r in pair_rows], dtype=np.float64)
    mean_acc_prob, boot_lo, boot_hi = bootstrap_mean_ci(acc_values, np.random.default_rng(args.seed + 1))
    binom_se = float(math.sqrt(max(0.0, mean_acc_prob * (1.0 - mean_acc_prob)) / max(1, len(acc_values))))
    threshold_accept = float(accepted / len(pair_rows))
    min_coarse_acc = float(min(r["acceptance_probability_mean"] for r in coarse_rows))
    max_coarse_acc = float(max(r["acceptance_probability_mean"] for r in coarse_rows))
    corr_action = float(np.corrcoef([r["acceptance_probability_mean"] for r in coarse_rows], [r["coarse_action_density"] for r in coarse_rows])[0, 1])
    corr_phi2 = float(np.corrcoef([r["acceptance_probability_mean"] for r in coarse_rows], [r["coarse_phi2"] for r in coarse_rows])[0, 1])
    print("wrote conditional fixed-coarse acceptance CSVs", flush=True)

    native_obs, _native_g = per_config_observables(l16[: max(args.n_coarse, n_total)].astype(np.float32), action)
    proposal_obs, _proposal_g = per_config_observables(phi.astype(np.float32), action)
    metrics = metric_rows(native_obs, proposal_obs)
    write_csv(out_dir / "raw_flow_vs_native_metrics.csv", metrics)
    if not args.skip_histograms:
        make_histograms(native_obs, proposal_obs, out_dir)
    else:
        print("skipped histogram generation by request", flush=True)

    manifest = {
        "command": " ".join(sys.argv),
        "flow_checkpoint": str(args.flow_checkpoint),
        "flow_sha256": sha256(args.flow_checkpoint),
        "flow_absolute_epoch": ckpt.get("absolute_epoch"),
        "kernel": str(args.kernel),
        "kernel_sha256": sha256(args.kernel),
        "kernel_sum": float(kernel_stencil_from_spec(kernel).sum()),
        "kernel_coefficients_include_eta_scale": bool(kernel.kernel_coefficients_include_eta_scale),
        "n_coarse": args.n_coarse,
        "proposals_per_coarse": args.proposals_per_coarse,
        "n_proposals": n_total,
        "n_within_coarse_pairs": len(pair_rows),
        "start_index": args.start_index,
        "source_indices": idx.tolist(),
        "seed": args.seed,
        "lambda": 1.0,
        "kappa_f": 0.340301,
        "reblocking_max_error": reblock_max,
        "model_load_report": {k: str(v) for k, v in load_report.items()},
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("wrote run_manifest.json", flush=True)
    summary = [
        "# Small L8->L16 Flow Proposal Check",
        "",
        "This is the conditional fixed-coarse L8->L16 MIT independence-Metropolis acceptance-rate diagnostic.",
        "",
        f"- fixed coarse fields: `{args.n_coarse}`",
        f"- proposals per coarse field: `{args.proposals_per_coarse}`",
        f"- total proposals: `{n_total}`",
        f"- within-coarse old/new log-ratio pairs: `{len(pair_rows)}`",
        f"- mean Metropolis acceptance probability: `{mean_acc_prob:.6g}`",
        f"- binomial standard error estimate: `{binom_se:.6g}`",
        f"- bootstrap 95% CI for mean acceptance probability: `[{boot_lo:.6g}, {boot_hi:.6g}]`",
        f"- min/max coarse-field mean acceptance: `{min_coarse_acc:.6g}` / `{max_coarse_acc:.6g}`",
        f"- fraction with logR >= 0: `{threshold_accept:.6g}`",
        f"- logR mean/std: `{float(np.mean(logr_values)):.6g}` / `{float(np.std(logr_values, ddof=1)):.6g}`",
        f"- correlation of per-coarse acceptance with coarse action density: `{corr_action:.6g}`",
        f"- correlation of per-coarse acceptance with coarse phi2: `{corr_phi2:.6g}`",
        f"- reblocking max error: `{reblock_max:.6g}`",
        f"- nonfinite proposals: `{int(np.sum(nonfinite > 0))}`",
        f"- flow checkpoint: `{args.flow_checkpoint}`",
        f"- kernel: `{args.kernel}`",
        "",
        "Each Metropolis ratio pairs two proposals conditioned on the same fixed coarse field. This is not a Markov chain and does not test fixed-coarse stationarity.",
        "",
        "## Acceptance by Coarse Field",
        "",
        "| coarse index | coarse action density | coarse phi2 | mean acceptance | logR mean | logR std |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for r in coarse_rows:
        summary.append(f"| {r['coarse_config_index']} | {r['coarse_action_density']:.6g} | {r['coarse_phi2']:.6g} | {r['acceptance_probability_mean']:.6g} | {r['logR_mean']:.6g} | {r['logR_std']:.6g} |")
    summary.extend([
        "",
        "## Raw Flow vs Native L16 Metrics",
        "",
        "| observable | shift native sigma | std ratio | KS | TV | JS | OVL |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for r in metrics:
        summary.append(f"| {r['observable']} | {r['shift_native_sigma']:.6g} | {r['std_ratio']:.6g} | {r['KS']:.6g} | {r['TV']:.6g} | {r['JS']:.6g} | {r['OVL']:.6g} |")
    (out_dir / "summary.md").write_text("\n".join(summary) + "\n")
    print(out_dir, flush=True)
    print("\n".join(summary[:20]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
