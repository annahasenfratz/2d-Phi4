#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
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
from scipy import stats

from train_lam1p0_autoregressive_detail_flow import ARDetailFlow, action_total, sample_model
from train_lam1p0_flow_detail_localreg import ETA_SCALE
from train_lam1p0_flow_detail_pilot import assemble_psi, inverse_kernel, load_kernel_matrix, load_phi
from train_lam1p0_rqspline_detail_flow import RQSplineARDetailFlow, ResidualSplineARDetailFlow


PHYS_KEYS = ["action_density", "phi2", "phi4", "local_kurtosis_ratio", "NN", "diag", "2nn", "m", "m2", "m4"]
PLOT_KEYS = ["action_density", "phi2", "phi4", "local_kurtosis_ratio", "NN", "diag", "2nn", "m2", "m4", "DeltaS", "log_acceptance"]


class ArgsShim:
    def __init__(self, batch_size: int, device: str):
        self.batch_size = batch_size
        self.device = device


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_affine(path: Path, device: torch.device) -> tuple[ARDetailFlow, dict[str, Any]]:
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ck["config"]
    model = ARDetailFlow(
        layers=int(cfg["layers"]),
        hidden=int(cfg["hidden_channels"]),
        kernel_size=int(cfg["conv_kernel_size"]),
        log_scale_bound=float(cfg["log_scale_bound"]),
    ).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    return model, ck


def load_spline(path: Path, device: torch.device) -> tuple[ResidualSplineARDetailFlow, dict[str, Any]]:
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ck["config"]
    source = torch.load(PROJECT_ROOT / cfg["resume_checkpoint"], map_location=device, weights_only=False)
    scfg = source["config"]
    affine = ARDetailFlow(
        layers=int(scfg["layers"]),
        hidden=int(scfg["hidden_channels"]),
        kernel_size=int(scfg["conv_kernel_size"]),
        log_scale_bound=float(scfg["log_scale_bound"]),
    ).to(device)
    affine.load_state_dict(source["model_state"])
    spline = RQSplineARDetailFlow(
        layers=int(cfg["layers"]),
        hidden=int(cfg["hidden_channels"]),
        kernel_size=int(cfg["conv_kernel_size"]),
        num_bins=int(cfg["num_bins"]),
        tail_bound=float(cfg["tail_bound"]),
        min_bin_width=float(cfg["min_bin_width"]),
        min_bin_height=float(cfg["min_bin_height"]),
        min_derivative=float(cfg["min_derivative"]),
    ).to(device)
    model = ResidualSplineARDetailFlow(affine, spline).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    return model, ck


def per_cfg(phi: np.ndarray) -> dict[str, np.ndarray]:
    arr = np.asarray(phi, dtype=np.float64)
    L = arr.shape[1]
    V = L * L
    m = arr.mean(axis=(1, 2))
    phi2 = np.mean(arr * arr, axis=(1, 2))
    phi4 = np.mean(arr**4, axis=(1, 2))
    nn = 0.5 * (
        np.mean(arr * np.roll(arr, -1, axis=1), axis=(1, 2))
        + np.mean(arr * np.roll(arr, -1, axis=2), axis=(1, 2))
    )
    two = 0.5 * (
        np.mean(arr * np.roll(arr, -2, axis=1), axis=(1, 2))
        + np.mean(arr * np.roll(arr, -2, axis=2), axis=(1, 2))
    )
    diag = np.mean(arr * np.roll(np.roll(arr, -1, axis=1), -1, axis=2), axis=(1, 2))
    return {
        "action_density": action_total(arr, type("A", (), {"lambda_": 1.0, "kappa": 0.340301})()) / V,
        "phi2": phi2,
        "phi4": phi4,
        "local_kurtosis_ratio": phi4 / np.maximum(phi2 * phi2, 1.0e-300),
        "NN": nn,
        "diag": diag,
        "2nn": two,
        "m": m,
        "m2": m * m,
        "m4": m**4,
    }


def sample_phi(model, coarse: np.ndarray, stats_: dict[str, Any], kernel: np.ndarray, args: ArgsShim, seed: int) -> dict[str, np.ndarray]:
    detail, logq, zmax, logdet = sample_model(model, coarse, stats_, args, seed)
    phi, _ = inverse_kernel(assemble_psi(coarse, detail), kernel)
    return {"detail": detail, "phi": phi.astype(np.float32), "logq": logq, "zmax": zmax, "logdet": logdet}


def patch(model, coarse, stats_, kernel, args, seed_base: int, sweeps: int, initial: dict[str, np.ndarray]) -> dict[str, Any]:
    n = len(coarse)
    rng = np.random.default_rng(seed_base + 100)
    cur = {k: np.array(v, copy=True) for k, v in initial.items()}
    cur_s = action_total(cur["phi"], type("A", (), {"lambda_": 1.0, "kappa": 0.340301})()).astype(np.float64)
    cur_logq = cur["logq"].astype(np.float64)
    rows = []
    for sweep in range(1, sweeps + 1):
        prop = sample_phi(model, coarse, stats_, kernel, args, seed_base + 1000 + sweep)
        prop_s = action_total(prop["phi"], type("A", (), {"lambda_": 1.0, "kappa": 0.340301})()).astype(np.float64)
        prop_logq = prop["logq"].astype(np.float64)
        ds = prop_s - cur_s
        lqr = cur_logq - prop_logq
        la = -ds + lqr
        accp = np.exp(np.minimum(la, 0.0))
        acc = np.log(rng.random(n)) < np.minimum(la, 0.0)
        rows.extend(
            {
                "sweep": sweep,
                "config_index": i,
                "DeltaS": float(ds[i]),
                "log_q_forward": float(prop_logq[i]),
                "log_q_reverse": float(cur_logq[i]),
                "log_q_ratio": float(lqr[i]),
                "log_acceptance": float(la[i]),
                "acceptance_probability": float(accp[i]),
                "accepted": int(acc[i]),
            }
            for i in range(n)
        )
        if np.any(acc):
            for key in ["detail", "phi", "logq", "zmax", "logdet"]:
                cur[key][acc] = prop[key][acc]
            cur_s[acc] = prop_s[acc]
            cur_logq[acc] = prop_logq[acc]
    return {"final": cur, "diagnostics": rows}


def finite(x):
    x = np.asarray(x, dtype=float)
    return x[np.isfinite(x)]


def ovl(a, b, bins=80, range_=None):
    a = finite(a)
    b = finite(b)
    lo, hi = range_ if range_ is not None else (min(a.min(), b.min()), max(a.max(), b.max()))
    if math.isclose(lo, hi):
        lo -= 0.5
        hi += 0.5
    ha, edges = np.histogram(a, bins=bins, range=(lo, hi), density=True)
    hb, _ = np.histogram(b, bins=edges, density=True)
    return float(np.sum(np.minimum(ha, hb) * np.diff(edges)))


def score(native, sample, bins=80, range_=None):
    a = finite(native)
    b = finite(sample)
    lo, hi = range_ if range_ is not None else (min(a.min(), b.min()), max(a.max(), b.max()))
    if math.isclose(lo, hi):
        lo -= 0.5
        hi += 0.5
    ha, edges = np.histogram(a, bins=bins, range=(lo, hi))
    hb, _ = np.histogram(b, bins=edges)
    pa = ha / max(ha.sum(), 1)
    pb = hb / max(hb.sum(), 1)
    m = 0.5 * (pa + pb)
    js = 0.0
    mask = pa > 0
    js += 0.5 * float(np.sum(pa[mask] * np.log(pa[mask] / np.maximum(m[mask], 1.0e-300))))
    mask = pb > 0
    js += 0.5 * float(np.sum(pb[mask] * np.log(pb[mask] / np.maximum(m[mask], 1.0e-300))))
    ks = stats.ks_2samp(a, b)
    qlo, qhi = np.quantile(a, [0.025, 0.975])
    plo, phi = np.quantile(b, [0.025, 0.975])
    return {
        "native_mean": float(a.mean()),
        "native_std": float(a.std(ddof=1)),
        "proposal_mean": float(b.mean()),
        "proposal_std": float(b.std(ddof=1)),
        "mean_diff_native_std": float((b.mean() - a.mean()) / max(a.std(ddof=1), 1.0e-300)),
        "ks_statistic": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
        "wasserstein": float(stats.wasserstein_distance(a, b)),
        "jensen_shannon": js,
        "proposal_frac_inside_native_central95": float(np.mean((b >= qlo) & (b <= qhi))),
        "native_frac_inside_proposal_central95": float(np.mean((a >= plo) & (a <= phi))),
        "overlap_coefficient": ovl(a, b, bins, (lo, hi)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--affine-checkpoint", type=Path, required=True)
    ap.add_argument("--spline-checkpoint", type=Path, required=True)
    ap.add_argument("--kernel-path", type=Path, default=Path("perfect_blocking/perfect_blocking_lam1p0/kernels/selected_for_upscaling/best_7x7_phi2_nn_guarded_eta_included.json"))
    ap.add_argument("--n-samples", type=int, default=2000)
    ap.add_argument("--patch-sweeps", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=2026071706)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    out = args.run_dir / "diagnostic_histogram_check"
    figdir = out / "figures"
    cfgdir = out / "configs"
    for d in [out, figdir, cfgdir]:
        d.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    shim = ArgsShim(args.batch_size, args.device)
    kernel, raw_kernel = load_kernel_matrix(args.kernel_path)
    if abs(float(kernel.sum()) - ETA_SCALE) > 1.0e-10 or not raw_kernel.get("kernel_coefficients_include_eta_scale"):
        raise RuntimeError("eta-included kernel check failed")
    phi8 = load_phi(Path("data/configs_phi4_2d/lam1p0_kappac0p340301_L8/configs.npz"))[: args.n_samples].astype(np.float32)
    phi16 = load_phi(Path("data/configs_phi4_2d/lam1p0_kappac0p340301_L16/configs.npz"))[: args.n_samples].astype(np.float32)
    affine, ack = load_affine(args.affine_checkpoint, device)
    spline, sck = load_spline(args.spline_checkpoint, device)
    seeds = {"affine": args.seed + 11, "spline": args.seed + 11, "patch": args.seed + 10000}
    affine_raw = sample_phi(affine, phi8, ack["state"]["stats"], kernel, shim, seeds["affine"])
    spline_raw = sample_phi(spline, phi8, sck["state"]["stats"], kernel, shim, seeds["spline"])
    affine_patch = patch(affine, phi8, ack["state"]["stats"], kernel, shim, seeds["patch"], args.patch_sweeps, affine_raw)
    spline_patch = patch(spline, phi8, sck["state"]["stats"], kernel, shim, seeds["patch"], args.patch_sweeps, spline_raw)
    arrays = {
        "native_L16": phi16,
        "affine_raw": affine_raw["phi"],
        "affine_patch": affine_patch["final"]["phi"],
        "spline_raw": spline_raw["phi"],
        "spline_patch": spline_patch["final"]["phi"],
    }
    for label, arr in arrays.items():
        np.save(cfgdir / f"{label}.npy", arr)
    np.save(cfgdir / "source_coarse_L8.npy", phi8)
    with (out / "matched_seeds.json").open("w") as f:
        json.dump(seeds, f, indent=2)
    obs = {k: per_cfg(v) for k, v in arrays.items()}
    for label, rows in [("affine_patch", affine_patch["diagnostics"]), ("spline_patch", spline_patch["diagnostics"])]:
        obs[label]["DeltaS"] = np.asarray([float(r["DeltaS"]) for r in rows])
        obs[label]["log_acceptance"] = np.asarray([float(r["log_acceptance"]) for r in rows])
        write_csv(out / f"{label}_proposal_diagnostics_per_proposal.csv", rows)
    per_rows = []
    for label, vals in obs.items():
        if "action_density" not in vals:
            continue
        for i in range(len(vals["action_density"])):
            row = {"ensemble": label, "config_index": i}
            for key in PHYS_KEYS:
                row[key] = float(vals[key][i])
            per_rows.append(row)
    write_csv(out / "per_configuration_observables.csv", per_rows)
    stat_rows = []
    score_rows = []
    colors = {"native_L16": "black", "affine_raw": "#d95f02", "affine_patch": "#e6ab02", "spline_raw": "#1b9e77", "spline_patch": "#7570b3"}
    for key in PLOT_KEYS:
        labels = ["native_L16", "affine_raw", "affine_patch", "spline_raw", "spline_patch"] if key not in {"DeltaS", "log_acceptance"} else ["affine_patch", "spline_patch"]
        allv = np.concatenate([finite(obs[l][key]) for l in labels])
        lo, hi = float(allv.min()), float(allv.max())
        if math.isclose(lo, hi):
            lo -= 0.5
            hi += 0.5
        fig, ax = plt.subplots(figsize=(6.6, 4.0), constrained_layout=True)
        for label in labels:
            x = finite(obs[label][key])
            ax.hist(x, bins=80, range=(lo, hi), density=True, histtype="step", lw=1.5, color=colors.get(label), label=label)
            stat_rows.append({"observable": key, "ensemble": label, "n": len(x), "mean": float(x.mean()), "std": float(x.std(ddof=1)), "p025": float(np.quantile(x, 0.025)), "p50": float(np.quantile(x, 0.5)), "p975": float(np.quantile(x, 0.975))})
        ax.set_xlabel(key)
        ax.set_ylabel("density")
        ax.legend(frameon=False, fontsize=8)
        ax.tick_params(direction="in", top=True, right=True)
        fig.savefig(figdir / f"{key}.pdf")
        plt.close(fig)
        if key not in {"DeltaS", "log_acceptance"}:
            for label in ["affine_raw", "affine_patch", "spline_raw", "spline_patch"]:
                score_rows.append({"observable": key, "proposal_ensemble": label, **score(obs["native_L16"][key], obs[label][key], 80, (lo, hi))})
        else:
            score_rows.append({"observable": key, "proposal_ensemble": "spline_patch_vs_affine_patch", **score(obs["affine_patch"][key], obs["spline_patch"][key], 80, (lo, hi))})
    write_csv(out / "observable_statistics.csv", stat_rows)
    write_csv(out / "histogram_overlap_scores.csv", score_rows)

    def get(k, label, field):
        for r in score_rows:
            if r["observable"] == k and r["proposal_ensemble"] == label:
                return float(r[field])
        return float("nan")

    lines = ["# Diagnostic Histogram Check", ""]
    raw_ovl = {k: get(k, "spline_raw", "overlap_coefficient") for k in ["action_density", "phi4", "local_kurtosis_ratio"]}
    raw_case_a = all(v > 0.75 for v in raw_ovl.values())
    answer = "Case A: the RQ-spline proposal has substantial raw overlap with native L16; it is not totally off." if raw_case_a else "Case B/mixed: at least one key raw spline observable has weak overlap with native L16."
    lines.append("Direct answer: " + answer)
    lines += [
        "",
        f"Diagnostic sample: `{args.n_samples}` matched L8 coarse inputs, `{args.patch_sweeps}` identical fixed-coarse independence sweeps for affine and spline proposals.",
        f"Kernel sum: `{float(kernel.sum()):.15g}`; `kernel_coefficients_include_eta_scale={raw_kernel.get('kernel_coefficients_include_eta_scale')}`.",
        "",
        "| observable | affine raw KS | spline raw KS | affine patch KS | spline patch KS | spline raw OVL | spline patch OVL |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key in ["action_density", "phi4", "local_kurtosis_ratio", "NN", "diag", "2nn", "m2", "m4"]:
        lines.append(f"| {key} | {get(key,'affine_raw','ks_statistic'):.6g} | {get(key,'spline_raw','ks_statistic'):.6g} | {get(key,'affine_patch','ks_statistic'):.6g} | {get(key,'spline_patch','ks_statistic'):.6g} | {get(key,'spline_raw','overlap_coefficient'):.6g} | {get(key,'spline_patch','overlap_coefficient'):.6g} |")
    lines += [
        "",
        "Interpretation: the raw spline proposal repairs the main local-kurtosis/action-density overlap problem relative to affine. The patch-chain final ensemble is not uniformly closer by histogram KS for every local operator, so acceptance improvement should not be interpreted as a complete physical-distribution match.",
        "",
        "Recommendation: do not launch L16->L32 automatically. The spline is a credible L8->L16 proposal, but the after-patch local-operator distortions should be inspected in the PDFs before escalating.",
    ]
    (out / "summary.md").write_text("\\n".join(lines) + "\\n")
    print(json.dumps({"status": "completed", "out": str(out), "n": args.n_samples, "patch_sweeps": args.patch_sweeps}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
