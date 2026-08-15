#!/usr/bin/env python3
"""Stage B: direct L16 coarse starts, frozen empirical initialization, exact detail-only A/R."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.stats import ks_2samp, wasserstein_distance

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "perfect_blocking_upsampling"
sys.path.insert(0, str(PKG / "src")); sys.path.insert(0, str(PKG / "scripts"))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/inverse_rg_mpl")

import run_lam1p0_empirical_joint_2x2_mixture as empirical
from perfect_blocking_upsampling.actions import ActionSpec
from perfect_blocking_upsampling.kernels import apply_kernel, inverse_kernel, load_kernel
from run_calibrated_empirical_blocked_native_detail_only import OBS, comparison, ensemble, obs, write_csv
from run_lam0p2_residual_flow_patch_chain import StreamingCsv, patch_correct
from train_lam1p0_autoregressive_detail_flow import torch_kernel_fft, torch_inverse_kernel
from train_lam1p0_flow_detail_pilot import load_phi, split_pairs
from train_lam1p0_local_multistage_rqspline import assemble_psi

OUT = PKG / "runs/lam1p0/calibrated_empirical_patchwise_rethermalization_20260721"
FRESH = PKG / "runs/lam1p0/fresh_disjoint_calibrated_empirical_validation_20260721"
L16_PATH = ROOT / "data/configs_phi4_2d/lam1p0_kappac0p340301_L16/configs.npz"
L32_PATH = ROOT / "data/configs_phi4_2d/lam1p0_kappac0p340301_L32/configs.npz"
KERNEL_PATH = ROOT / "perfect_blocking/perfect_blocking_lam1p0/kernels/final/chosen_kernel.json"
SAVES = (0, 1, 2, 5, 10, 20, 50, 100, 200)


def empirical_initial(c: np.ndarray, kernel_matrix: np.ndarray, donor_h: np.ndarray, donor_d: np.ndarray, hmean: np.ndarray, hstd: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    meta = empirical.meta(len(c), c.shape[1])
    context = (empirical.features(c, meta) - hmean) / hstd
    distances, indices = cKDTree(donor_h).query(context, k=8)
    tau = float(np.quantile(distances[:, 0], .25))
    rng = np.random.default_rng(seed)
    sigma = .01 * (donor_d.std(0) + 1.e-6)
    blocks = np.empty((len(meta), 12), np.float32)
    chosen = np.empty(len(meta), np.int64)
    for row in range(len(meta)):
        weights = np.exp(-(distances[row]**2 - distances[row, 0]**2) / (2. * tau * tau)); weights /= weights.sum()
        pick = int(rng.choice(8, p=weights)); chosen[row] = indices[row, pick]
        blocks[row] = donor_d[chosen[row]] + rng.normal(size=12).astype(np.float32) * sigma
    raw, details = empirical.reconstruct(c, meta, blocks, kernel_matrix, 2 * c.shape[1])
    details = details.copy(); z = rng.normal(size=len(c)).astype(np.float32)
    gamma = (.97 * np.exp((.32 / c.shape[1]) * z)).astype(np.float32)
    details[:, :2] *= gamma[:, None, None, None]
    psi = assemble_psi(torch.from_numpy(c), *[torch.from_numpy(details[:, sector]) for sector in range(3)])
    calibrated = torch_inverse_kernel(psi, torch_kernel_fft(kernel_matrix, 2 * c.shape[1], torch.device("cpu"))).detach().numpy().astype(np.float32)
    return calibrated, z, gamma, chosen


def make_plots(reference: dict[str, np.ndarray], states: dict[int, np.ndarray], action: ActionSpec) -> None:
    directory = OUT / "plots/direct_coarse_detail_only"; directory.mkdir(parents=True, exist_ok=True)
    for name in ("action_density", "phi2", "phi4", "NN", "m2", "G_pmin_avg"):
        values = np.concatenate([reference[name]] + [obs(states[s], action)[name] for s in SAVES])
        lo, hi = np.quantile(values, [.001, .999]); bins = np.linspace(lo - .05 * (hi - lo), hi + .05 * (hi - lo), 61)
        fig, axes = plt.subplots(3, 3, figsize=(10.5, 8), constrained_layout=True)
        for axis, sweep in zip(axes.flat, SAVES):
            axis.hist(reference[name], bins=bins, density=True, histtype="step", lw=1.5, label="native L32")
            axis.hist(obs(states[sweep], action)[name], bins=bins, density=True, histtype="step", lw=1.3, label=f"direct L16, s={sweep}")
            axis.set_title(f"sweep {sweep}"); axis.set_xlabel(name)
            if sweep == 0: axis.legend(frameon=False, fontsize=7)
        fig.savefig(directory / f"direct_coarse_{name}.pdf"); fig.savefig(directory / f"direct_coarse_{name}.png", dpi=180); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--chains", type=int, default=128); parser.add_argument("--sweeps", type=int, default=200); args = parser.parse_args()
    if args.sweeps != 200: raise ValueError("Stage B is fixed to the requested sweep-200 diagnostic")
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    kernel, _ = load_kernel(KERNEL_PATH); kernel_matrix, _ = empirical.load_kernel_matrix(KERNEL_PATH)
    direct_all = load_phi(L16_PATH); native_all = load_phi(L32_PATH)
    if len(direct_all) < args.chains or len(native_all) < 2128: raise RuntimeError("insufficient independent direct-L16 or native-L32 configurations")
    coarse = direct_all[:args.chains].astype(np.float32).copy()
    reference_field = native_all[2000:2000 + args.chains].astype(np.float32).copy()
    donor_pairs = split_pairs(native_all[:1000], kernel_matrix)
    donor_meta = empirical.meta(1000, 16)
    donor_raw_h = empirical.features(donor_pairs["coarse"], donor_meta)
    hmean, hstd = donor_raw_h.mean(0), donor_raw_h.std(0) + 1.e-6
    donor_h = (donor_raw_h - hmean) / hstd
    donor_d = empirical.vectors(donor_pairs["detail"], donor_meta)
    initial, z, gamma, chosen = empirical_initial(coarse, kernel_matrix, donor_h, donor_d, hmean, hstd, 2026072116)
    psi = apply_kernel(initial, kernel).astype(np.float32)
    reblock = float(np.max(np.abs(psi[:, 0::2, 0::2] - coarse)))
    if reblock > 2.e-6: raise RuntimeError(f"initial empirical reconstruction does not preserve direct coarse: {reblock}")
    action = ActionSpec("phi4_nn", 1., .340301); reference = obs(reference_field, action); initial_obs = obs(initial, action)
    rows: list[dict] = []; acceptance: list[dict] = []; states: dict[int, np.ndarray] = {}
    rng = np.random.default_rng(2026072117)
    for sweep in range(args.sweeps + 1):
        if sweep in SAVES:
            current = obs(initial, action); rows.extend(comparison(reference, current, "direct_L16_calibrated_empirical", sweep, initial_obs))
            rows.extend({"sweep": sweep, "ensemble": "direct_L16_calibrated_empirical", "observable": name, "value_mean": value} for name, value in ensemble(initial).items())
            states[sweep] = initial.copy()
        if sweep == args.sweeps: break
        writer = StreamingCsv(OUT / "logs" / f"direct_L16_detail_patches_sweep{sweep+1:03d}.csv", ["sweep", "phase", "patch_size", "pass", "patch_index", "patch_x", "patch_y", "attempts", "accepted", "acceptance", "A_over_R", "deltaS_mean", "deltaS_std", "deltaS_min", "deltaS_max", "delta_logw_mean", "delta_logw_std", "log_accept_mean", "log_accept_std", "patch_l2_mean", "local_rms", "elapsed_sec"])
        detail_args = argparse.Namespace(disable_coarse_updates=True, detail_passes=10, fine_proposal_sigma=.04, fine_patch_size=16, passes=0, proposal_sigma=.0, coarse_patch_size=16, global_sweep=sweep+1, verbose_patch_log=False)
        initial, psi, meta = patch_correct(psi, kernel, action, detail_args, writer, rng); writer.close()
        acceptance.append({"ensemble": "direct_L16_calibrated_empirical", "sweep": sweep+1, **meta})
    final_reblock = float(np.max(np.abs(apply_kernel(initial, kernel)[:, 0::2, 0::2] - coarse)))
    if final_reblock > 2.e-6: raise RuntimeError(f"final direct fixed-coarse reblocking failure: {final_reblock}")
    write_csv(OUT / "direct_coarse_detail_only_metrics.csv", rows); write_csv(OUT / "direct_coarse_detail_acceptance.csv", acceptance)
    blocked_rows = list(csv.DictReader((OUT / "blocked_native_detail_only_metrics.csv").open()))
    comparison_rows = []
    for sweep in SAVES:
        for name in OBS:
            direct = next(r for r in rows if r.get("ensemble") == "direct_L16_calibrated_empirical" and r.get("observable") == name and r.get("sweep") == sweep)
            blocked = next((r for r in blocked_rows if r["ensemble"] == "calibrated_empirical" and r["observable"] == name and int(r["sweep"]) == sweep), None)
            comparison_rows.append({"sweep": sweep, "observable": name, "direct_shift_native_sigma": direct.get("shift_native_sigma"), "direct_width_ratio": direct.get("width_ratio"), "direct_KS": direct.get("KS"), "blocked_native_shift_native_sigma": None if blocked is None else blocked.get("shift_native_sigma"), "blocked_native_width_ratio": None if blocked is None else blocked.get("width_ratio"), "blocked_native_KS": None if blocked is None else blocked.get("KS"), "blocked_native_reference_available": blocked is not None})
    write_csv(OUT / "blocked_native_vs_direct_coarse_comparison.csv", comparison_rows)
    write_csv(OUT / "direct_L16_source_inventory.csv", [{"role":"direct_native_L16","start":0,"stop":args.chains-1,"count":args.chains,"blocked_from_L32":False}, {"role":"empirical_L32_donors","start":0,"stop":999,"count":1000}, {"role":"native_L32_reference","start":2000,"stop":2000+args.chains-1,"count":args.chains}, {"initial_max_reblocking_error":reblock,"final_max_reblocking_error":final_reblock}])
    np.savez_compressed(OUT / "direct_L16_calibrated_initialization.npz", coarse=coarse, initial=states[0], z=z, gamma=gamma, selected_donor_blocks=chosen)
    make_plots(reference, states, action)
    mean_acceptance = float(np.mean([float(r["detail_update_acceptance"]) for r in acceptance]))
    summary = ["# Stage B: Direct L16 Coarse, Detail-Only", "", "The coarse inputs are independently generated native-action L16 fields, not blocked L32 configurations. The empirical proposal appears only at sweep 0. All subsequent updates use the unchanged exact fixed-coarse detail A/R; no empirical proposal density enters acceptance.", "", f"- Chains: `{args.chains}`; sweeps: `0,1,2,5,10,20,50,100,200`.", "- Detail updater: `P=16`, 10 passes/sweep, sigma=0.04.", f"- Mean detail acceptance: `{mean_acceptance:.6f}`.", f"- Max reblocking error: initial `{reblock:.3e}`, final `{final_reblock:.3e}`.", "", "Part II coarse-plus-detail has not been run."]
    (OUT / "direct_coarse_detail_only_summary.md").write_text("\n".join(summary) + "\n")


if __name__ == "__main__": main()
