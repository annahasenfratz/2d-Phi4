#!/usr/bin/env python3
"""Side-by-side direct-L16 validation of the colleague and extended 5x5 kernels."""
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

BASE = ROOT / "perfect_blocking/perfect_blocking_lam1p0/tests"
COLLEAGUE = BASE / "colleague_paper_objective_5x5_all_direct"
EXTENDED = BASE / "extended_local2_5x5_all_direct"
OUT = BASE / "colleague_vs_extended_local2_5x5"
DIRECT = ROOT / "data/configs_phi4_2d/lam1p0_kappac0p340301_L16/configs.npz"
FINE = ROOT / "data/configs_phi4_2d/lam1p0_kappac0p340301_L32/configs.npz"
COL_KERNEL = ROOT / "perfect_blocking/perfect_blocking_lam1p0/kernels/candidates/colleague_paper_objective_5x5_eta_included.json"
EXT_KERNEL = ROOT / "perfect_blocking/perfect_blocking_lam1p0/tests/intermediate/allL16_chi2_R2_corrW5000_extraLocal2_frozenBlockCov_invCap1p65_train3000_val1000_test1000/result.json"
OBS = ["action_density", "phi2", "phi4", "local_kurtosis_ratio", "NN", "diag", "2nn", "m2", "m4", "G_pmin_avg"]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    c = {r["operator"]: r for r in csv.DictReader((COLLEAGUE / "all_operator_and_pair_product_comparison.csv").open()) if r["kind"] == "operator"}
    e = {r["operator"]: r for r in csv.DictReader((EXTENDED / "all_operator_and_pair_product_comparison.csv").open()) if r["kind"] == "operator"}
    rows = [{"operator": o, "direct_mean": c[o]["direct_mean"], "direct_bootstrap_se": c[o]["direct_bootstrap_se"],
             "colleague_mean": c[o]["blocked_mean"], "colleague_z": c[o]["difference_bootstrap_z"], "colleague_KS": c[o]["KS"],
             "extended_mean": e[o]["blocked_mean"], "extended_z": e[o]["difference_bootstrap_z"], "extended_KS": e[o]["KS"]} for o in OBS]
    write_csv(OUT / "operator_comparison.csv", rows)
    cc = {(r["operator_a"], r["operator_b"]): r for r in csv.DictReader((COLLEAGUE / "all_operator_correlations.csv").open())}
    ce = {(r["operator_a"], r["operator_b"]): r for r in csv.DictReader((EXTENDED / "all_operator_correlations.csv").open())}
    corr_rows = [{"operator_a": a, "operator_b": b, "rho_direct": cc[a,b]["rho_direct"], "rho_colleague": cc[a,b]["rho_blocked"], "rho_extended": ce[a,b]["rho_blocked"], "delta_colleague": cc[a,b]["rho_blocked_minus_direct"], "delta_extended": ce[a,b]["rho_blocked_minus_direct"]} for a,b in cc]
    write_csv(OUT / "correlation_comparison.csv", corr_rows)

    with np.load(DIRECT) as z: direct_phi = np.asarray(z["phi"], float)
    with np.load(FINE) as z: fine_phi = np.asarray(z["phi"], float)
    mc = np.asarray(json.loads(COL_KERNEL.read_text())["matrix"], float)
    me = np.asarray(json.loads(EXT_KERNEL.read_text())["matrix"], float)
    direct, col, ext = observable_arrays(direct_phi), observable_arrays(block(fine_phi, mc)), observable_arrays(block(fine_phi, me))
    fig, axes = plt.subplots(2, 5, figsize=(15, 5.8), constrained_layout=True)
    for ax, key in zip(axes.flat, OBS):
        lo, hi = np.quantile(np.r_[direct[key], col[key], ext[key]], [.001, .999])
        ax.hist(direct[key], bins=45, range=(lo,hi), density=True, histtype="stepfilled", alpha=.20, color="black", label="direct L16")
        ax.hist(col[key], bins=45, range=(lo,hi), density=True, histtype="step", lw=1.4, color="tab:red", label="colleague 5x5")
        ax.hist(ext[key], bins=45, range=(lo,hi), density=True, histtype="step", lw=1.4, color="tab:green", label="extended 5x5")
        ax.set_title(key); ax.tick_params(direction="in", top=True, right=True)
    axes.flat[0].legend(fontsize=7.5, frameon=False)
    fig.savefig(OUT / "operator_histograms_direct_colleague_extended.pdf", bbox_inches="tight"); plt.close(fig)

    cd = np.asarray([[float(r["rho_direct"]),float(r["rho_colleague"]),float(r["rho_extended"])] for r in corr_rows])
    fig, axes = plt.subplots(1, 2, figsize=(10.3, 4.4), constrained_layout=True)
    axes[0].scatter(cd[:,0], cd[:,1], s=26, color="tab:red", label="colleague")
    axes[0].scatter(cd[:,0], cd[:,2], s=26, color="tab:green", marker="s", label="extended")
    axes[0].plot([-1,1],[-1,1], color="black", lw=.9); axes[0].set(xlim=(-1,1),ylim=(-1,1),xlabel="direct $\\rho$",ylabel="blocked $\\rho$")
    axes[0].legend(frameon=False); axes[0].tick_params(direction="in", top=True, right=True)
    delta_c, delta_e = cd[:,1]-cd[:,0], cd[:,2]-cd[:,0]
    axes[1].scatter(delta_c, delta_e, s=28, color="tab:purple")
    lim=1.08*max(np.max(np.abs(delta_c)),np.max(np.abs(delta_e)))
    axes[1].plot([-lim,lim],[-lim,lim],color="black",lw=.9); axes[1].axhline(0,color="0.7",lw=.7); axes[1].axvline(0,color="0.7",lw=.7)
    axes[1].set(xlim=(-lim,lim),ylim=(-lim,lim),xlabel=r"$\Delta\rho$ colleague",ylabel=r"$\Delta\rho$ extended",title="Below diagonal: extended is closer")
    axes[1].tick_params(direction="in", top=True, right=True)
    fig.savefig(OUT / "correlation_scatter_comparison.pdf", bbox_inches="tight"); plt.close(fig)
    print(json.dumps({"out_dir": str(OUT), "n_operators": len(rows), "n_correlations": len(corr_rows)}, indent=2))

if __name__ == "__main__": main()
