#!/usr/bin/env python3
"""Reweight the direct native L16 ensemble in coarse kappa and lambda."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "perfect_blocking" / "perfect_blocking_lam1p0" / "coarse_action_reweight_kappa_lambda_20260720"
DIRECT = ROOT / "perfect_blocking" / "perfect_blocking_lam1p0" / "observables/native/direct_L16_all_observables_per_config.csv"
BLOCKED = ROOT / "perfect_blocking" / "perfect_blocking_lam1p0" / "observables/blocked/native_L32_blocked_to_L16_observables_per_config.csv"
KAPPA0 = 0.340301
LAMBDA0 = 1.0
OBS = ["action_density", "phi2", "phi4", "local_kurtosis_ratio", "NN", "diag", "2nn", "G_pmin_x_cfg", "G_pmin_y_cfg"]


def load_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return {k: np.asarray([float(r[k]) for r in rows], dtype=float) for k in rows[0] if k not in {"sample", "direct_L16_config_index", "native_L32_blocked_to_L16_config_index", "L", "volume", "lambda", "kappa", "nonfinite_count"}}


def weighted_ks(x: np.ndarray, wx: np.ndarray, y: np.ndarray) -> float:
    order_x = np.argsort(x)
    order_y = np.argsort(y)
    xs, wxs = x[order_x], wx[order_x]
    ys = y[order_y]
    allv = np.sort(np.unique(np.concatenate([xs, ys])))
    ix = np.searchsorted(xs, allv, side="right")
    iy = np.searchsorted(ys, allv, side="right")
    cdfx = np.cumsum(wxs)[np.maximum(ix - 1, 0)] / wx.sum()
    cdfx[ix == 0] = 0.0
    cdfy = iy / len(ys)
    return float(np.max(np.abs(cdfx - cdfy)))


def hist_metrics(x: np.ndarray, wx: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    edges = np.histogram_bin_edges(np.concatenate([x, y]), bins=50)
    hx = np.histogram(x, bins=edges, weights=wx)[0]
    hy = np.histogram(y, bins=edges)[0].astype(float)
    hx /= max(hx.sum(), 1e-300)
    hy /= max(hy.sum(), 1e-300)
    tv = 0.5 * np.abs(hx - hy).sum()
    js = 0.5 * np.sum(np.where(hx > 0, hx * np.log(hx / np.maximum((hx + hy) / 2, 1e-300)), 0)) + 0.5 * np.sum(np.where(hy > 0, hy * np.log(hy / np.maximum((hx + hy) / 2, 1e-300)), 0))
    overlap = float(np.minimum(hx, hy).sum())
    return float(tv), float(js), overlap


def trial_observables(d: dict[str, np.ndarray], kappa: float, lam: float) -> dict[str, np.ndarray]:
    out = {k: np.asarray(v) for k, v in d.items()}
    # `NN` is the average over the two forward lattice directions, while the
    # action contains their sum: -2*kappa*(NN_x + NN_y) = -4*kappa*NN.
    out["action_density"] = (1.0 - 2.0 * lam) * d["phi2"] + lam * d["phi4"] - 4.0 * kappa * d["NN"]
    out["local_kurtosis_ratio"] = d["phi4"] / np.maximum(d["phi2"] ** 2, 1e-300)
    out["G_pmin_avg"] = 0.5 * (d["G_pmin_x_cfg"] + d["G_pmin_y_cfg"])
    out["PC1"] = 0.4525 * d["phi2"] + 0.8918 * d["phi4"]
    out["PC2"] = 0.8918 * d["phi2"] - 0.4525 * d["phi4"]
    return out


def evaluate(d: dict[str, np.ndarray], b: dict[str, np.ndarray], kappa: float, lam: float, baseline: dict[str, np.ndarray]) -> dict[str, float]:
    trial = trial_observables(d, kappa, lam)
    target = trial_observables(b, KAPPA0, LAMBDA0)
    logw = -(trial["action_density"] - baseline["action_density"]) * 256.0
    logw -= np.max(logw)
    w = np.exp(logw)
    w /= w.sum()
    ess = 1.0 / np.sum(w * w)
    out: dict[str, float] = {"kappa": kappa, "lambda": lam, "delta_kappa": kappa - KAPPA0, "delta_lambda": lam - LAMBDA0, "ESS": ess, "ESS_fraction": ess / len(w), "max_log_weight_shift": float(np.max(logw) - np.min(logw))}
    for key in ["PC1", "PC2", *OBS]:
        x, y = trial[key], target.get(key, None)
        if y is None:
            continue
        mu = float(np.sum(w * x))
        sd = float(np.sqrt(np.sum(w * (x - mu) ** 2)))
        bm, bsd = float(np.mean(y)), float(np.std(y))
        out[f"{key}_mean"] = mu
        out[f"{key}_std"] = sd
        out[f"{key}_mean_shift_blocked_sigma"] = (mu - bm) / max(bsd, 1e-300)
        out[f"{key}_std_ratio"] = sd / max(bsd, 1e-300)
        out[f"{key}_KS"] = weighted_ks(x, w, y)
        tv, js, overlap = hist_metrics(x, w, y)
        out[f"{key}_TV"], out[f"{key}_JS"], out[f"{key}_overlap"] = tv, js, overlap
        for q in [0.05, 0.50, 0.95, 0.99]:
            out[f"{key}_q{int(q*100):02d}"] = float(np.quantile(np.repeat(x, 1), q)) if False else float(weighted_quantile(x, w, q))
    return out


def weighted_quantile(x: np.ndarray, w: np.ndarray, q: float) -> float:
    order = np.argsort(x)
    xx, ww = x[order], w[order]
    return float(np.interp(q, np.cumsum(ww), xx))


def write_rows(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0])
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def plot_stage(rows: list[dict[str, float]], stage: str) -> None:
    x = [r["delta_kappa"] if stage == "kappa" else r["delta_lambda"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(x, [r["PC1_mean_shift_blocked_sigma"] for r in rows], "o-", label="PC1")
    ax[0].plot(x, [r["PC2_mean_shift_blocked_sigma"] for r in rows], "o-", label="PC2")
    ax[0].axhline(0, color="black", lw=0.8)
    ax[0].set(xlabel=f"delta {stage}", ylabel="mean shift / blocked sigma")
    ax[0].legend()
    ax[1].plot(x, [r["ESS_fraction"] for r in rows], "o-")
    ax[1].axhline(0.5, color="tab:red", ls="--", lw=0.8)
    ax[1].set(xlabel=f"delta {stage}", ylabel="ESS fraction")
    fig.tight_layout()
    fig.savefig(OUT / f"stage_{1 if stage == 'kappa' else 2}_{stage}_scan.pdf")
    plt.close(fig)


def baseline_kurtosis_ks(d: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> float:
    x = trial_observables(d, KAPPA0, LAMBDA0)
    y = trial_observables(b, KAPPA0, LAMBDA0)
    return weighted_ks(x["local_kurtosis_ratio"], np.ones(len(x["phi2"])), y["local_kurtosis_ratio"])


def baseline_nn_ks(d: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> float:
    x = trial_observables(d, KAPPA0, LAMBDA0)
    y = trial_observables(b, KAPPA0, LAMBDA0)
    return weighted_ks(x["NN"], np.ones(len(x["phi2"])), y["NN"])


def stage3_summary(rows: list[dict[str, float]]) -> str:
    lines = [
        "# Stage 3: kappa-lambda reweighting grid",
        "",
        "The 3x3 grid used `delta_kappa={+0.0003,+0.0005,+0.0007}` and `delta_lambda={+0.001,+0.002,+0.003}`. Candidates are ranked by the transparent PC1/PC2 loss: mean-shift squares, width-ratio deviations, and PC1/PC2 KS squares. No configurations were regenerated.",
        "",
        "| rank | delta kappa | delta lambda | loss | ESS fraction | PC1 shift | PC1 width | PC1 KS | PC2 shift | PC2 width | PC2 KS | action shift | action width | kurtosis KS | NN KS | guardrails |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for i, r in enumerate(rows, 1):
        lines.append("| %d | %+.4f | %+.3f | %.6f | %.6f | %+.5f | %.5f | %.4f | %+.5f | %.5f | %.4f | %+.4f | %.5f | %.4f | %.4f | %s |" % (i, r["delta_kappa"], r["delta_lambda"], r["loss"], r["ESS_fraction"], r["PC1_mean_shift_blocked_sigma"], r["PC1_std_ratio"], r["PC1_KS"], r["PC2_mean_shift_blocked_sigma"], r["PC2_std_ratio"], r["PC2_KS"], r["action_density_mean_shift_blocked_sigma"], r["action_density_std_ratio"], r["local_kurtosis_ratio_KS"], r["NN_KS"], "PASS" if r["passes_all_guardrails"] else "FAIL"))
    best = rows[0]
    passing = [r for r in rows if r["passes_all_guardrails"]]
    lines += [
        "",
        "## Stage 3 conclusion",
        "",
        f"The loss-ranked point is `delta_kappa={best['delta_kappa']:+.4f}`, `delta_lambda={best['delta_lambda']:+.3f}`. It has PC1 shift `{best['PC1_mean_shift_blocked_sigma']:+.4f}σ`, PC2 shift `{best['PC2_mean_shift_blocked_sigma']:+.4f}σ`, ESS fraction `{best['ESS_fraction']:.6f}`, action shift `{best['action_density_mean_shift_blocked_sigma']:+.4f}σ`, and action width ratio `{best['action_density_std_ratio']:.4f}`.",
        "",
        f"Guardrail result: `{len(passing)}` of `{len(rows)}` grid points pass the implemented ESS, action mean/width, local-kurtosis, and NN checks.",
        "",
        "The grid is a reweighting diagnostic only. It does not justify changing the production coarse action or running a new coarse ensemble without the full distribution and histogram checks.",
        "",
        "The CSV contains all requested per-observable means, widths, KS, TV, JS, overlap, and weighted quantiles.",
    ]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["1", "2", "3"], required=True)
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    d, b = load_csv(DIRECT), load_csv(BLOCKED)
    base = trial_observables(d, KAPPA0, LAMBDA0)
    if args.stage == "1":
        points = [0, -0.0001, 0.0001, -0.0002, 0.0002, -0.0003, 0.0003, -0.0005, 0.0005]
        rows = [evaluate(d, b, KAPPA0 + dk, LAMBDA0, base) for dk in points]
        write_rows(OUT / "stage1_kappa_scan.csv", rows)
        plot_stage(rows, "kappa")
    elif args.stage == "2":
        points = [0, -0.002, 0.002, -0.005, 0.005, -0.01, 0.01, -0.02, 0.02]
        rows = [evaluate(d, b, KAPPA0, LAMBDA0 + dl, base) for dl in points]
        write_rows(OUT / "stage2_lambda_scan.csv", rows)
        plot_stage(rows, "lambda")
    else:
        kappas = [KAPPA0 + x for x in (0.0003, 0.0005, 0.0007)]
        lams = [LAMBDA0 + x for x in (0.001, 0.002, 0.003)]
        rows = []
        for kappa in kappas:
            for lam in lams:
                row = evaluate(d, b, kappa, lam, base)
                loss = sum(row[f"{key}_mean_shift_blocked_sigma"] ** 2 for key in ["PC1", "PC2"])
                loss += sum((row[f"{key}_std_ratio"] - 1.0) ** 2 for key in ["PC1", "PC2"])
                loss += row["PC1_KS"] ** 2 + row["PC2_KS"] ** 2
                row["loss"] = loss
                row["passes_ESS"] = row["ESS_fraction"] >= 0.25
                row["passes_action_mean"] = abs(row["action_density_mean_shift_blocked_sigma"]) < 0.05
                row["passes_action_width"] = abs(row["action_density_std_ratio"] - 1.0) < 0.03
                row["passes_kurtosis"] = row["local_kurtosis_ratio_KS"] <= baseline_kurtosis_ks(base, b) + 0.01
                row["passes_NN"] = row["NN_KS"] <= baseline_nn_ks(base, b) + 0.01
                row["passes_all_guardrails"] = all(row[k] for k in ["passes_ESS", "passes_action_mean", "passes_action_width", "passes_kurtosis", "passes_NN"])
                rows.append(row)
        rows.sort(key=lambda r: r["loss"])
        write_rows(OUT / "stage3_kappa_lambda_grid.csv", rows)
        (OUT / "stage3_kappa_lambda_summary.md").write_text(stage3_summary(rows) + "\n")
    (OUT / "reweighting_diagnostics.json").write_text(json.dumps({"direct_csv": str(DIRECT), "blocked_csv": str(BLOCKED), "kappa0": KAPPA0, "lambda0": LAMBDA0, "N_direct": len(d["phi2"]), "N_blocked": len(b["phi2"])}, indent=2) + "\n")
    print(f"stage {args.stage} complete: {len(rows)} points, N_direct={len(d['phi2'])}, flush=complete", flush=True)


if __name__ == "__main__":
    main()
