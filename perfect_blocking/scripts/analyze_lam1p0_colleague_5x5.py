#!/usr/bin/env python3
"""All-observable direct-L16 versus direct-L32-blocked validation of a supplied 5x5 kernel."""
from __future__ import annotations

import csv
import argparse
import json
import sys
from itertools import combinations_with_replacement
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "perfect_blocking/scripts"))
from run_lam1p0_7x7_kernel_search import block, momentum_extrema, observable_arrays  # noqa: E402

DIRECT = ROOT / "data/configs_phi4_2d/lam1p0_kappac0p340301_L16/configs.npz"
FINE = ROOT / "data/configs_phi4_2d/lam1p0_kappac0p340301_L32/configs.npz"
DEFAULT_KERNEL = ROOT / "perfect_blocking/perfect_blocking_lam1p0/kernels/candidates/colleague_paper_objective_5x5_eta_included.json"
DEFAULT_OUT = ROOT / "perfect_blocking/perfect_blocking_lam1p0/tests/colleague_paper_objective_5x5_all_direct"
OBS = ["action_density", "phi2", "phi4", "local_kurtosis_ratio", "NN", "diag", "2nn", "m2", "m4", "G_pmin_avg"]
BOOTSTRAPS, SEED = 500, 2026080709


def load_phi(path: Path) -> np.ndarray:
    with np.load(path) as z:
        return np.asarray(z["phi"], dtype=np.float64)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    keys: list[str] = []
    for row in rows:
        keys.extend(k for k in row if k not in keys)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


def bootstrap_means(values: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    n = len(values)
    weights = rng.multinomial(n, np.full(n, 1.0 / n), size=BOOTSTRAPS) / n
    boot = weights @ values
    return values.mean(axis=0), boot.std(axis=0, ddof=1)


def hist_metrics(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    lo, hi = np.quantile(np.concatenate([x, y]), [0.001, 0.999])
    edges = np.linspace(lo, hi, 51) if hi > lo else np.linspace(lo - .5, hi + .5, 51)
    px = np.histogram(x, edges)[0].astype(float); py = np.histogram(y, edges)[0].astype(float)
    px /= px.sum(); py /= py.sum(); mid = .5 * (px + py)
    kl = lambda a, b: np.sum(a[a > 0] * np.log(a[a > 0] / b[a > 0]))
    return {"KS": float(stats.ks_2samp(x, y).statistic), "KS_pvalue": float(stats.ks_2samp(x, y).pvalue), "std_ratio_blocked_over_direct": float(np.std(y, ddof=1) / np.std(x, ddof=1)), "TV": float(.5 * np.abs(px - py).sum()), "JS": float(.5 * kl(px, mid) + .5 * kl(py, mid))}


def pair_correlations(values: np.ndarray) -> np.ndarray:
    return np.corrcoef(values, rowvar=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", type=Path, default=DEFAULT_KERNEL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--direct", type=Path, default=DIRECT,
                        help="direct L16 config archive; defaults to the kappa=0.340301 reference")
    parser.add_argument("--fine", type=Path, default=FINE,
                        help="direct L32 config archive to be blocked")
    args = parser.parse_args()
    kernel = args.kernel if args.kernel.is_absolute() else ROOT / args.kernel
    out = args.out if args.out.is_absolute() else ROOT / args.out
    direct_path = args.direct if args.direct.is_absolute() else ROOT / args.direct
    fine_path = args.fine if args.fine.is_absolute() else ROOT / args.fine
    out.mkdir(parents=True, exist_ok=True)
    matrix = np.asarray(json.loads(kernel.read_text())["matrix"], dtype=np.float64)
    direct_phi, fine_phi = load_phi(direct_path), load_phi(fine_path)
    # These are independent ensembles, so no configuration-by-configuration
    # pairing is required.  Use all 5,000 direct L16 fields and all 10,000
    # available direct L32 fields after blocking.
    blocked_phi = block(fine_phi, matrix)
    direct, blocked = observable_arrays(direct_phi), observable_arrays(blocked_phi)
    x = np.column_stack([direct[k] for k in OBS]); y = np.column_stack([blocked[k] for k in OBS])
    rng = np.random.default_rng(SEED)
    xm, xse = bootstrap_means(x, rng); ym, yse = bootstrap_means(y, rng)
    rows: list[dict[str, object]] = []
    for i, key in enumerate(OBS):
        h = hist_metrics(x[:, i], y[:, i]); delta = ym[i] - xm[i]
        rows.append({"kind": "operator", "operator": key, "direct_mean": xm[i], "direct_bootstrap_se": xse[i], "blocked_mean": ym[i], "blocked_bootstrap_se": yse[i], "difference_blocked_minus_direct": delta, "difference_bootstrap_z": delta / max(np.hypot(xse[i], yse[i]), 1e-300), "direct_std": float(np.std(x[:, i], ddof=1)), "blocked_std": float(np.std(y[:, i], ddof=1)), **h})
    # All products including O_i^2: 55 pair-product expectation values.
    combo_names = [f"{OBS[i]}*{OBS[j]}" for i, j in combinations_with_replacement(range(len(OBS)), 2)]
    xp = np.column_stack([x[:, i] * x[:, j] for i, j in combinations_with_replacement(range(len(OBS)), 2)])
    yp = np.column_stack([y[:, i] * y[:, j] for i, j in combinations_with_replacement(range(len(OBS)), 2)])
    xpm, xpse = bootstrap_means(xp, rng); ypm, ypse = bootstrap_means(yp, rng)
    for i, name in enumerate(combo_names):
        delta = ypm[i] - xpm[i]
        rows.append({"kind": "pair_product", "operator": name, "direct_mean": xpm[i], "direct_bootstrap_se": xpse[i], "blocked_mean": ypm[i], "blocked_bootstrap_se": ypse[i], "difference_blocked_minus_direct": delta, "difference_bootstrap_z": delta / max(np.hypot(xpse[i], ypse[i]), 1e-300)})
    write_csv(out / "all_operator_and_pair_product_comparison.csv", rows)
    rd, rb = pair_correlations(x), pair_correlations(y)
    cd, cb = np.cov(x, rowvar=False, ddof=1), np.cov(y, rowvar=False, ddof=1)
    corr_rows = [{"operator_a": OBS[i], "operator_b": OBS[j],
                  "cov_direct": float(cd[i, j]), "cov_blocked": float(cb[i, j]),
                  "cov_blocked_minus_direct": float(cb[i, j] - cd[i, j]),
                  "rho_direct": float(rd[i, j]), "rho_blocked": float(rb[i, j]),
                  "rho_blocked_minus_direct": float(rb[i, j] - rd[i, j])}
                 for i in range(len(OBS)) for j in range(i + 1, len(OBS))]
    write_csv(out / "all_operator_correlations.csv", corr_rows)
    fig, axes = plt.subplots(2, 5, figsize=(15, 5.8), constrained_layout=True)
    for ax, i in zip(axes.flat, range(len(OBS))):
        lo, hi = np.quantile(np.concatenate([x[:, i], y[:, i]]), [.001, .999])
        ax.hist(x[:, i], bins=45, range=(lo, hi), density=True, histtype="stepfilled", alpha=.26, color="black", label="direct L16")
        ax.hist(y[:, i], bins=45, range=(lo, hi), density=True, histtype="step", lw=1.55, color="tab:red", label="L32 blocked to L16")
        ax.set_title(OBS[i]); ax.tick_params(direction="in", top=True, right=True)
    axes.flat[0].legend(frameon=False, fontsize=8)
    fig.savefig(out / "all_operator_histograms_direct_vs_blocked.pdf", bbox_inches="tight"); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    vmax = 1.0; dv = float(np.max(np.abs(rb - rd)))
    for ax, mat, title, cmap, lim in [(axes[0], rd, "direct L16", "coolwarm", (-vmax, vmax)), (axes[1], rb, "blocked L32→L16", "coolwarm", (-vmax, vmax)), (axes[2], rb-rd, "blocked − direct", "PiYG", (-dv, dv))]:
        im=ax.imshow(mat, cmap=cmap, vmin=lim[0], vmax=lim[1]); ax.set(title=title, xticks=range(len(OBS)), yticks=range(len(OBS)), xticklabels=OBS, yticklabels=OBS)
        plt.colorbar(im, ax=ax, shrink=.8)
        ax.tick_params(axis="x", labelrotation=55, labelsize=8); ax.tick_params(axis="y", labelsize=8)
    fig.savefig(out / "all_operator_correlation_matrices.pdf", bbox_inches="tight"); plt.close(fig)
    summary = {"direct_configs": str(direct_path), "fine_configs": str(fine_path), "n_direct": len(direct_phi), "n_blocked": len(blocked_phi), "kernel": str(kernel), "kernel_sum": float(matrix.sum()), "momentum_stability": momentum_extrema(matrix, grid=1024), "bootstrap_resamples": BOOTSTRAPS, "fit_operators": ["phi2", "phi4", "NN", "2nn", "diag", "m2"], "held_out_operators": ["m4", "G_pmin_avg"], "all_observables": OBS, "pair_products": len(combo_names)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"out_dir": str(out), **summary}, indent=2))


if __name__ == "__main__": main()
