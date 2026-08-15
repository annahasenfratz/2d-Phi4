#!/usr/bin/env python3
"""Test whether small (kappa, lambda) reweighting aligns direct L16 with blocked L32."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "perfect_blocking/scripts"))
from run_lam1p0_7x7_kernel_search import block, observable_arrays  # noqa: E402
from run_coarse_action_reweight_kappa_lambda import hist_metrics, weighted_ks  # noqa: E402

KAPPA0, LAMBDA0, VOLUME = 0.340301, 1.0, 16 * 16
DIRECT = ROOT / "data/configs_phi4_2d/lam1p0_kappac0p340301_L16/configs.npz"
FINE = ROOT / "data/configs_phi4_2d/lam1p0_kappac0p340301_L32/configs.npz"
KERNELS = {
    "colleague5": ROOT / "perfect_blocking/perfect_blocking_lam1p0/kernels/candidates/colleague_paper_objective_5x5_eta_included.json",
    "highcorr5": ROOT / "perfect_blocking/perfect_blocking_lam1p0/kernels/candidates/allL16_chi2_R2_corrW5000_highcorr_5x5_eta_included.json",
    "highcorr7": ROOT / "perfect_blocking/perfect_blocking_lam1p0/kernels/candidates/allL16_chi2_R3_corrW5000_highcorr_7x7_eta_included.json",
}
OUT = ROOT / "perfect_blocking/perfect_blocking_lam1p0/tests/reweight_highcorr_kernels"
OBS = ["action_density", "phi2", "phi4", "local_kurtosis_ratio", "NN", "diag", "2nn", "m2", "m4", "G_pmin_avg"]
CORR_PAIRS = [("phi2", "NN"), ("phi2", "phi4"), ("phi2", "local_kurtosis_ratio"), ("NN", "diag"), ("action_density", "local_kurtosis_ratio")]


def load_phi(path: Path) -> np.ndarray:
    with np.load(path) as z:
        return np.asarray(z["phi"], dtype=np.float64)


def action_density(o: dict[str, np.ndarray], kappa: float, lam: float) -> np.ndarray:
    # `NN` is 1/2 times the sum of the x- and y-forward bond densities.
    # The action contains -2*kappa times that full two-direction sum.
    return (1.0 - 2.0 * lam) * o["phi2"] + lam * o["phi4"] - 4.0 * kappa * o["NN"]


def with_action(base: dict[str, np.ndarray], kappa: float, lam: float) -> dict[str, np.ndarray]:
    # All non-action observables are coupling-independent.  Reusing the
    # precomputed arrays is important: the scan otherwise repeats expensive
    # FFT observables for every (kappa, lambda) point.
    o = {key: np.asarray(value) for key, value in base.items()}
    o["action_density"] = action_density(o, kappa, lam)
    return o


def weights(direct_base: dict[str, np.ndarray], kappa: float, lam: float) -> tuple[np.ndarray, float, float]:
    delta_s = VOLUME * ((lam - LAMBDA0) * (direct_base["phi4"] - 2.0 * direct_base["phi2"]) - 4.0 * (kappa - KAPPA0) * direct_base["NN"])
    logw = -delta_s
    logw -= np.max(logw)
    w = np.exp(logw)
    w /= w.sum()
    ess = 1.0 / np.sum(w * w)
    return w, ess, float(np.ptp(logw))


def wcorr(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    mx, my = np.sum(w * x), np.sum(w * y)
    vx, vy = np.sum(w * (x - mx) ** 2), np.sum(w * (y - my) ** 2)
    return float(np.sum(w * (x - mx) * (y - my)) / np.sqrt(max(vx * vy, 1e-300)))


def evaluate(direct_base: dict[str, np.ndarray], blocked_base: dict[str, np.ndarray], kappa: float, lam: float, kernel_name: str) -> dict[str, float]:
    w, ess, span = weights(direct_base, kappa, lam)
    direct, target = with_action(direct_base, kappa, lam), with_action(blocked_base, kappa, lam)
    row: dict[str, float] = {"kernel": kernel_name, "kappa": kappa, "lambda": lam, "delta_kappa": kappa - KAPPA0, "delta_lambda": lam - LAMBDA0, "ESS": ess, "ESS_fraction": ess / len(w), "log_weight_span": span}
    mean_loss = width_loss = ks_loss = 0.0
    for key in OBS:
        x, y = direct[key], target[key]
        mu = float(np.sum(w * x)); sd = float(np.sqrt(np.sum(w * (x - mu) ** 2)))
        tm, ts = float(np.mean(y)), float(np.std(y, ddof=1))
        shift, ratio = (mu - tm) / max(ts, 1e-300), sd / max(ts, 1e-300)
        ks = weighted_ks(x, w, y)
        tv, js, overlap = hist_metrics(x, w, y)
        row.update({f"{key}_mean": mu, f"{key}_target_mean": tm, f"{key}_mean_shift_sigma": shift, f"{key}_std_ratio": ratio, f"{key}_KS": ks, f"{key}_TV": tv, f"{key}_JS": js, f"{key}_overlap": overlap})
        mean_loss += shift * shift
        width_loss += (ratio - 1.0) ** 2
        ks_loss += ks * ks
    corr_loss = 0.0
    for a, b in CORR_PAIRS:
        r, rt = wcorr(direct[a], direct[b], w), float(np.corrcoef(target[a], target[b])[0, 1])
        row[f"rho_{a}__{b}"] = r; row[f"rho_target_{a}__{b}"] = rt; row[f"rho_delta_{a}__{b}"] = r - rt
        corr_loss += (r - rt) ** 2
    # Diagnostic ranking only: all normalized mean, width, KS, and selected-rho mismatches.
    row.update({"mean_loss": mean_loss, "width_loss": width_loss, "ks_loss": ks_loss, "corr_loss": corr_loss, "score": mean_loss + width_loss + 10.0 * ks_loss + 10.0 * corr_loss})
    return row


def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def plot_best(direct_base: dict[str, np.ndarray], blocked_base: dict[str, np.ndarray], row: dict[str, float], out: Path) -> None:
    w, _, _ = weights(direct_base, row["kappa"], row["lambda"])
    direct, target = with_action(direct_base, row["kappa"], row["lambda"]), with_action(blocked_base, row["kappa"], row["lambda"])
    fig, axes = plt.subplots(2, 4, figsize=(13, 6), constrained_layout=True)
    for ax, key in zip(axes.flat, OBS[:8]):
        lo, hi = np.quantile(np.concatenate([direct[key], target[key]]), [0.002, 0.998])
        ax.hist(target[key], bins=45, range=(lo, hi), density=True, histtype="stepfilled", alpha=.28, color="black", label="blocked L32")
        ax.hist(direct[key], bins=45, range=(lo, hi), weights=w, density=True, histtype="step", lw=1.6, color="tab:red", label="reweighted direct L16")
        ax.set_title(key); ax.tick_params(direction="in", top=True, right=True)
    axes.flat[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"{row['kernel']}: best reweighting  Δκ={row['delta_kappa']:+.6f}, Δλ={row['delta_lambda']:+.4f}, ESS/N={row['ESS_fraction']:.3f}")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    direct_phi, fine_phi = load_phi(DIRECT), load_phi(FINE)
    direct_base = observable_arrays(direct_phi)
    all_rows: list[dict[str, float]] = []
    kappas = KAPPA0 + np.linspace(-0.0010, 0.0010, 17)
    lambdas = LAMBDA0 + np.linspace(-0.010, 0.010, 17)
    for name, path in KERNELS.items():
        matrix = np.asarray(json.loads(path.read_text())["matrix"], dtype=np.float64)
        blocked = block(fine_phi, matrix)
        blocked_base = observable_arrays(blocked)
        rows = [evaluate(direct_base, blocked_base, float(k), float(l), name) for k in kappas for l in lambdas]
        valid = [r for r in rows if r["ESS_fraction"] >= 0.20]
        valid.sort(key=lambda r: r["score"])
        write_csv(OUT / f"{name}_reweight_grid.csv", rows)
        write_csv(OUT / f"{name}_reweight_ranked_ESSge0p20.csv", valid)
        plot_best(direct_base, blocked_base, valid[0], OUT / f"{name}_best_reweighted_vs_blocked.pdf")
        all_rows.extend(valid[:10])
    write_csv(OUT / "top10_per_kernel.csv", all_rows)
    print(json.dumps({"out_dir": str(OUT), "grid_points_per_kernel": len(kappas) * len(lambdas), "ESS_cut": 0.20}, indent=2))


if __name__ == "__main__":
    main()
