#!/usr/bin/env python3
"""Held-out distributional validation of a fitted L32->L16 blocked action."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[2]
FIT_SCRIPT = ROOT / "perfect_blocking/scripts/fit_lam1p0_blocked_action_relative_entropy.py"
DEFAULT_OUT = ROOT / "perfect_blocking/perfect_blocking_lam1p0/tests/softcond7_blocked_action_relative_entropy_greedy_diag_phi2_diag_bondsq_3nn"


def load_module():
    spec = importlib.util.spec_from_file_location("blocked_action_fit", FIT_SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def weighted_corr(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    mx, my = w @ x, w @ y
    return float((w @ ((x - mx) * (y - my))) / np.sqrt((w @ (x - mx)**2) * (w @ (y - my)**2)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--n-bootstrap", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20260810)
    args = ap.parse_args()
    out = args.out if args.out.is_absolute() else ROOT / args.out
    summary = json.loads((out / "summary.json").read_text())
    coeff = pd.read_csv(out / "blocked_action_coefficients.csv")
    m = load_module(); names = list(summary["action_basis"])
    alpha_map = dict(zip(coeff.operator, coeff.standardized_log_ratio_coefficient))
    alpha = np.asarray([alpha_map[name] for name in names], float)
    rng = np.random.default_rng(20260809)
    n = int(summary["n_train"] + summary["n_validation"] + summary["n_test"])
    direct_phi = m.load_configs(Path(summary["direct"])); fine_phi = m.load_configs(Path(summary["fine"]))
    direct_phi = direct_phi[rng.permutation(len(direct_phi))[:n]]
    fine_phi = fine_phi[rng.permutation(len(fine_phi))[:n]]
    blocked_phi = m.block_configs(fine_phi, m.load_kernel(Path(summary["kernel"])))
    direct, blocked = m.local_action_features(direct_phi), m.local_action_features(blocked_phi)
    x_direct = np.column_stack([direct[name] for name in names]); x_blocked = np.column_stack([blocked[name] for name in names])
    fit = np.arange(int(summary["n_train"] + summary["n_validation"])); test = np.arange(len(fit), n)
    joined = np.r_[x_direct[fit], x_blocked[fit]]
    center, scale = joined.mean(0), np.maximum(joined.std(0, ddof=1), 1e-12)
    w, logw = m.normalized_weights((x_direct[test] - center) / scale, alpha)
    rng = np.random.default_rng(args.seed)
    boot_direct = rng.choice(len(test), size=(args.n_bootstrap, len(test)), replace=True, p=w)
    boot_blocked = rng.integers(len(test), size=(args.n_bootstrap, len(test)))
    report = list(dict.fromkeys(summary["report_basis"] + ["m"]))
    rows = []
    for name in report:
        xd, xb = direct[name][test], blocked[name][test]
        rw_boot = xd[boot_direct].mean(axis=1); b_boot = xb[boot_blocked].mean(axis=1)
        rows.append({"operator": name, "blocked_mean": float(xb.mean()), "blocked_bootstrap_se": float(b_boot.std(ddof=1)),
                     "reweighted_mean": float(w @ xd), "reweighted_bootstrap_se": float(rw_boot.std(ddof=1)),
                     "difference_reweighted_minus_blocked": float(w @ xd - xb.mean()),
                     "difference_bootstrap_se": float((rw_boot-b_boot).std(ddof=1)),
                     "direct_mean": float(xd.mean())})
    pd.DataFrame(rows).to_csv(out / "heldout_weighted_bootstrap_operator_comparison.csv", index=False)

    # These are ensemble observables: compute them from a bootstrap ensemble,
    # never configuration-by-configuration.
    L = direct_phi.shape[1]
    def derived(frame: dict[str, np.ndarray], take: np.ndarray) -> np.ndarray:
        mm = frame["m"][test][take].mean(axis=1)
        m2 = frame["m2"][test][take].mean(axis=1)
        m4 = frame["m4"][test][take].mean(axis=1)
        gp = frame["G_pmin_avg"][test][take].mean(axis=1)
        binder = 1.0 - m4 / (3.0 * m2**2)
        chi = L**2 * (m2 - mm**2)
        xi_arg = chi / gp - 1.0
        xi = np.where(xi_arg > 0, np.sqrt(xi_arg) / (2.0 * L * np.sin(np.pi / L)), np.nan)
        return np.column_stack([binder, xi, chi])
    rw_derived = derived(direct, boot_direct)
    blocked_derived = derived(blocked, boot_blocked)
    # Point estimates use normalized reweighting rather than a bootstrap draw.
    mm, m2, m4, gp = (w @ direct[key][test] for key in ("m", "m2", "m4", "G_pmin_avg"))
    rw_point = np.array([1.0-m4/(3*m2*m2), np.sqrt(max((L**2*(m2-mm*mm))/gp-1.0, 0.0))/(2*L*np.sin(np.pi/L)), L**2*(m2-mm*mm)])
    mm, m2, m4, gp = (blocked[key][test].mean() for key in ("m", "m2", "m4", "G_pmin_avg"))
    blocked_point = np.array([1.0-m4/(3*m2*m2), np.sqrt(max((L**2*(m2-mm*mm))/gp-1.0, 0.0))/(2*L*np.sin(np.pi/L)), L**2*(m2-mm*mm)])
    derived_rows = []
    for i, name in enumerate(("Binder_U4", "xi_over_L", "chi")):
        derived_rows.append({"observable": name, "blocked_mean": float(blocked_point[i]), "blocked_bootstrap_se": float(np.nanstd(blocked_derived[:,i], ddof=1)),
                             "reweighted_mean": float(rw_point[i]), "reweighted_bootstrap_se": float(np.nanstd(rw_derived[:,i], ddof=1)),
                             "difference": float(rw_point[i]-blocked_point[i]), "difference_bootstrap_se": float(np.nanstd(rw_derived[:,i]-blocked_derived[:,i], ddof=1))})
    pd.DataFrame(derived_rows).to_csv(out / "heldout_weighted_bootstrap_derived.csv", index=False)

    corr_rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            xa, ya, xb, yb = direct[a][test], direct[b][test], blocked[a][test], blocked[b][test]
            rw_boot = np.array([np.corrcoef(xa[take], ya[take])[0, 1] for take in boot_direct])
            b_boot = np.array([np.corrcoef(xb[take], yb[take])[0, 1] for take in boot_blocked])
            corr_rows.append({"pair": f"{a} x {b}", "blocked_rho": float(np.corrcoef(xb, yb)[0, 1]),
                              "blocked_bootstrap_se": float(b_boot.std(ddof=1)), "reweighted_rho": weighted_corr(xa, ya, w),
                              "reweighted_bootstrap_se": float(rw_boot.std(ddof=1)),
                              "difference": float(weighted_corr(xa, ya, w)-np.corrcoef(xb, yb)[0, 1]),
                              "difference_bootstrap_se": float((rw_boot-b_boot).std(ddof=1))})
    pd.DataFrame(corr_rows).to_csv(out / "heldout_weighted_action_correlations.csv", index=False)

    # Multi-page PDF: reweighted empirical density versus the independent
    # blocked target density.  Weights are applied directly, not by a noisy
    # resampled surrogate.
    plotted = [name for name in report if name != "m"]
    with PdfPages(out / "heldout_reweighted_vs_blocked_histograms.pdf") as pdf:
        for start in range(0, len(plotted), 12):
            page = plotted[start:start + 12]
            fig, axes = plt.subplots(3, 4, figsize=(11.0, 7.8), constrained_layout=True)
            for ax, name in zip(axes.flat, page):
                xd, xb = direct[name][test], blocked[name][test]
                lo, hi = min(xd.min(), xb.min()), max(xd.max(), xb.max())
                ax.hist(xb, bins=45, range=(lo, hi), density=True, histtype="stepfilled", alpha=.30, color="C0", label="blocked L32→L16")
                ax.hist(xd, bins=45, range=(lo, hi), density=True, weights=w, histtype="step", lw=1.4, color="C3", label="reweighted direct L16")
                ax.axvline(xb.mean(), color="C0", ls="--", lw=.9)
                ax.axvline(w @ xd, color="C3", ls="--", lw=.9)
                ax.set_title(name); ax.tick_params(direction="in", top=True, right=True)
            for ax in axes.flat[len(page):]: ax.remove()
            axes.flat[0].legend(frameon=False, fontsize=7)
            fig.suptitle("Held-out blocked action: reweighted direct versus blocked target")
            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
    report_summary = {"n_test": len(test), "n_bootstrap": args.n_bootstrap,
                      "heldout_ess": m.effective_sample_size(w), "heldout_ess_fraction": m.effective_sample_size(w)/len(test),
                      "log_weight_span": float(np.ptp(logw)), "action_basis": names}
    (out / "heldout_weighted_validation_summary.json").write_text(json.dumps(report_summary, indent=2) + "\n")
    print(json.dumps(report_summary, indent=2))


if __name__ == "__main__":
    main()
