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

from perfect_blocking_upsampling.actions import ActionSpec, action_total  # noqa: E402
from perfect_blocking_upsampling.kernels import apply_kernel, inverse_kernel, kernel_stencil_from_spec, load_kernel  # noqa: E402
from run_lam1p0_l16to32_rqspline_zeroshot import build_model_from_checkpoint, per_config_observables, stationary_stats  # noqa: E402
from train_lam1p0_flow_detail_pilot import assemble_psi, load_phi  # noqa: E402


DEFAULT_FLOW = PROJECT_ROOT / "perfect_blocking_upsampling/runs/lam1p0/training/lam1p0_L16to32_newkernel_rqspline_N2000_action_lowtail_from_N2000_bestpatch_20260719T171401Z/checkpoints/checkpoint_best_patch.pt"
DEFAULT_KERNEL = PROJECT_ROOT / "perfect_blocking/perfect_blocking_lam1p0/kernels/final/chosen_kernel.json"
DEFAULT_NATIVE_L16 = PROJECT_ROOT / "data/configs_phi4_2d/lam1p0_kappac0p340301_L16/configs.npz"
DEFAULT_NATIVE_L32 = PROJECT_ROOT / "data/configs_phi4_2d/lam1p0_kappac0p340301_L32/configs.npz"
DEFAULT_OUT = PROJECT_ROOT / "perfect_blocking/perfect_blocking_lam1p0/tests/final/small_mit_nf_L16to32_alternating_detail_coarse_20260720"
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


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())


def coarse_standardized(coarse_phys: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    return ((coarse_phys - stats["coarse_mean"]) / stats["coarse_std"]).astype(np.float32)


def draw_latents(n: int, lc: int, rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal((n, 3, lc * lc)).astype(np.float32)


def flow_from_latents(
    model: Any,
    coarse_phys: np.ndarray,
    z_np: np.ndarray,
    stats: dict[str, Any],
    kernel: Any,
    action: ActionSpec,
    *,
    device: torch.device,
) -> dict[str, np.ndarray]:
    coarse_std_np = coarse_standardized(coarse_phys, stats)
    n, lc, _ = coarse_std_np.shape
    mean = np.asarray(stats["detail_mean"], dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.asarray(stats["detail_std"], dtype=np.float32).reshape(1, 3, 1, 1)
    log_jac_const = -float(lc * lc * np.sum(np.log(std.reshape(3))))
    with torch.no_grad():
        coarse = torch.from_numpy(coarse_std_np).to(device)
        z_all = torch.from_numpy(z_np.astype(np.float32)).to(device)
        d = coarse.new_zeros((n, 3, lc, lc))
        log_base = coarse.new_zeros(n)
        logdet_total = coarse.new_zeros(n)
        zmax = torch.amax(torch.abs(z_all.reshape(n, -1)), dim=1)
        for stage in range(3):
            z = z_all[:, stage]
            log_base = log_base + (-0.5 * (z * z + math.log(2.0 * math.pi)).sum(dim=1))
            cond_affine = model.affine_base.cond(coarse, d, stage)
            x_affine, affine_logdet = model.affine_base.flows[stage].forward(z, cond_affine)
            cond_spline = model.cond(coarse, d, stage)
            x, spline_logdet = model.spline.flows[stage].forward(x_affine, cond_spline)
            d[:, stage] = x.reshape(n, lc, lc)
            logdet_total = logdet_total + affine_logdet + spline_logdet
    detail = d.detach().cpu().numpy().astype(np.float32) * std + mean
    log_gaussian = log_base.detach().cpu().numpy().astype(np.float64)
    logj = (logdet_total.detach().cpu().numpy() - log_jac_const).astype(np.float64)
    logq = (log_gaussian - logj).astype(np.float64)
    psi = assemble_psi(coarse_phys, detail.astype(np.float32)).astype(np.float32)
    phi, _ = inverse_kernel(psi, kernel)
    blocked = apply_kernel(phi, kernel)[:, 0::2, 0::2]
    reblock = np.max(np.abs(blocked.astype(np.float64) - coarse_phys.astype(np.float64)), axis=(1, 2))
    sf = action_total(phi, action).astype(np.float64)
    return {
        "detail": detail.astype(np.float32),
        "phi": phi.astype(np.float32),
        "S_f": sf,
        "log_gaussian": log_gaussian,
        "logJ": logj,
        "logq": logq,
        "zmax": zmax.detach().cpu().numpy().astype(np.float64),
        "reblock_error": reblock.astype(np.float64),
        "nonfinite": np.sum(~np.isfinite(phi).reshape(n, -1), axis=1),
    }


def observable_rows(phi: np.ndarray) -> dict[str, np.ndarray]:
    obs, _ = per_config_observables(phi.astype(np.float32), ActionSpec("phi4_nn", 1.0, 0.340301))
    return obs


def state_rows_for(
    *,
    chain_indices: range,
    cycle: int,
    substep_type: str,
    substep_index: int,
    attempt_global: int,
    accepted: np.ndarray,
    coarse_source_indices: np.ndarray,
    coarse_action: np.ndarray,
    current: dict[str, np.ndarray],
    category: str,
) -> list[dict[str, Any]]:
    obs = observable_rows(current["phi"])
    rows = []
    for chain in chain_indices:
        row = {
            "chain_id": chain,
            "cycle": cycle,
            "substep_type": substep_type,
            "substep_index": substep_index,
            "attempt_global": attempt_global,
            "accepted": int(accepted[chain]) if len(accepted) else 0,
            "state_category": category,
            "current_coarse_source_index": int(coarse_source_indices[chain]),
            "current_coarse_action": float(coarse_action[chain]),
            "current_fine_action": float(current["S_f"][chain]),
            "current_fine_action_density": float(current["S_f"][chain] / (32 * 32)),
            "current_log_r_z": float(current["log_gaussian"][chain]),
            "current_logJ": float(current["logJ"][chain]),
            "current_log_q_f": float(current["logq"][chain]),
            "reblocking_max_error": float(current["reblock_error"][chain]),
            "nonfinite_count": int(current["nonfinite"][chain]),
        }
        for key in OBS:
            row[key] = float(obs[key][chain])
        row["m2"] = float(obs["m2"][chain])
        row["m4"] = float(obs["m4"][chain])
        rows.append(row)
    return rows


def state_rows_for_per_chain_category(
    *,
    cycle: int,
    substep_type: str,
    substep_index: int,
    attempt_global: int,
    accepted: np.ndarray,
    coarse_source_indices: np.ndarray,
    coarse_action: np.ndarray,
    current: dict[str, np.ndarray],
    categories: np.ndarray,
) -> list[dict[str, Any]]:
    obs = observable_rows(current["phi"])
    rows = []
    for chain in range(len(current["phi"])):
        row = {
            "chain_id": chain,
            "cycle": cycle,
            "substep_type": substep_type,
            "substep_index": substep_index,
            "attempt_global": attempt_global,
            "accepted": int(accepted[chain]),
            "state_category": str(categories[chain]),
            "current_coarse_source_index": int(coarse_source_indices[chain]),
            "current_coarse_action": float(coarse_action[chain]),
            "current_fine_action": float(current["S_f"][chain]),
            "current_fine_action_density": float(current["S_f"][chain] / (32 * 32)),
            "current_log_r_z": float(current["log_gaussian"][chain]),
            "current_logJ": float(current["logJ"][chain]),
            "current_log_q_f": float(current["logq"][chain]),
            "reblocking_max_error": float(current["reblock_error"][chain]),
            "nonfinite_count": int(current["nonfinite"][chain]),
        }
        for key in OBS:
            row[key] = float(obs[key][chain])
        row["m2"] = float(obs["m2"][chain])
        row["m4"] = float(obs["m4"][chain])
        rows.append(row)
    return rows


def append_attempt_rows(
    *,
    target: list[dict[str, Any]],
    update_type: str,
    cycle: int,
    substep_index: int,
    proposal_indices: np.ndarray,
    old_coarse_source_indices: np.ndarray,
    new_coarse_source_indices: np.ndarray,
    old_coarse_action: np.ndarray,
    new_coarse_action: np.ndarray,
    old: dict[str, np.ndarray],
    proposal: dict[str, np.ndarray],
    logR: np.ndarray,
    accepted: np.ndarray,
    restore_error: np.ndarray,
    invariant_error: np.ndarray,
    z_unchanged_error: np.ndarray,
    c_unchanged_error: np.ndarray,
) -> None:
    for chain in range(len(logR)):
        dS = float(proposal["S_f"][chain] - old["S_f"][chain])
        dlogj = float(proposal["logJ"][chain] - old["logJ"][chain])
        dsc = float(new_coarse_action[chain] - old_coarse_action[chain])
        row = {
            "chain_id": chain,
            "cycle": cycle,
            "substep_type": update_type,
            "substep_index": substep_index,
            "proposed_coarse_source_index": int(proposal_indices[chain]),
            "old_coarse_source_index": int(old_coarse_source_indices[chain]),
            "new_coarse_source_index": int(new_coarse_source_indices[chain]),
            "accepted": int(accepted[chain]),
            "old_coarse_action": float(old_coarse_action[chain]),
            "proposed_coarse_action": float(new_coarse_action[chain]),
            "old_fine_action": float(old["S_f"][chain]),
            "proposed_fine_action": float(proposal["S_f"][chain]),
            "old_log_r_z": float(old["log_gaussian"][chain]),
            "proposed_log_r_z": float(proposal["log_gaussian"][chain]),
            "old_logJ": float(old["logJ"][chain]),
            "proposed_logJ": float(proposal["logJ"][chain]),
            "old_log_q_f": float(old["logq"][chain]),
            "proposed_log_q_f": float(proposal["logq"][chain]),
            "minus_delta_fine_action": -dS,
            "delta_logJ": dlogj,
            "delta_coarse_action": dsc,
            "logq_old_minus_new": float(old["logq"][chain] - proposal["logq"][chain]),
            "logR": float(logR[chain]),
            "acceptance_probability": float(min(1.0, math.exp(min(0.0, float(logR[chain]))))),
            "proposal_reblocking_max_error": float(proposal["reblock_error"][chain]),
            "proposal_nonfinite_count": int(proposal["nonfinite"][chain]),
            "restore_error_if_rejected": float(restore_error[chain]),
            "detail_c_unchanged_error": float(c_unchanged_error[chain]),
            "coarse_z_unchanged_error": float(z_unchanged_error[chain]),
            "accepted_state_invariant_error": float(invariant_error[chain]),
        }
        target.append(row)


def copy_state(state: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: val.copy() if isinstance(val, np.ndarray) else val for key, val in state.items()}


def set_accepted(current: dict[str, np.ndarray], proposal: dict[str, np.ndarray], accepted: np.ndarray) -> None:
    for key in current:
        current[key][accepted] = proposal[key][accepted]


def rejection_streaks(rows: list[dict[str, Any]], n_chains: int, update_type: str) -> tuple[int, float]:
    streaks = []
    for chain in range(n_chains):
        run = 0
        selected = [r for r in rows if int(r["chain_id"]) == chain and r["substep_type"] == update_type]
        for row in selected:
            if int(row["accepted"]) == 0:
                run += 1
            elif run:
                streaks.append(run)
                run = 0
        if run:
            streaks.append(run)
    return (max(streaks) if streaks else 0), (float(np.mean(streaks)) if streaks else 0.0)


def histogram_edges(samples: list[np.ndarray]) -> np.ndarray:
    vals = np.concatenate([np.asarray(s, dtype=np.float64)[np.isfinite(s)] for s in samples])
    lo = float(np.min(vals))
    hi = float(np.max(vals))
    if lo == hi:
        return np.linspace(lo - 0.5, hi + 0.5, 61)
    return np.linspace(lo, hi, 81)


def make_plots(out_dir: Path, state_rows: list[dict[str, Any]], native32: np.ndarray) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    native_obs, _ = per_config_observables(native32.astype(np.float32), ActionSpec("phi4_nn", 1.0, 0.340301))
    chains = sorted({int(r["chain_id"]) for r in state_rows})
    x = np.arange(len([r for r in state_rows if int(r["chain_id"]) == chains[0]]))
    for obs in OBS:
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        for chain in chains:
            vals = [float(r[obs]) for r in state_rows if int(r["chain_id"]) == chain]
            ax.plot(x, vals, lw=0.9, alpha=0.75)
        ax.set_xlabel("recorded attempt index")
        ax.set_ylabel(obs)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(fig_dir / f"{obs}_evolution_by_chain.pdf")
        plt.close(fig)
        print(f"wrote {fig_dir / f'{obs}_evolution_by_chain.pdf'}", flush=True)

        fig, ax = plt.subplots(figsize=(6.8, 4.4))
        categories = {
            "initial": [float(r[obs]) for r in state_rows if r["state_category"] == "initial"],
            "after_detail": [float(r[obs]) for r in state_rows if r["state_category"] == "after_detail"],
            "after_accepted_coarse": [float(r[obs]) for r in state_rows if r["state_category"] == "after_accepted_coarse"],
            "late_cycle": [float(r[obs]) for r in state_rows if r["state_category"] == "late_cycle"],
        }
        bins = histogram_edges([native_obs[obs], *[np.asarray(v) for v in categories.values() if v]])
        ax.hist(native_obs[obs], bins=bins, density=True, histtype="step", lw=2.2, color="black", label="native L32")
        styles = {
            "initial": ("tab:gray", "--"),
            "after_detail": ("tab:blue", "-"),
            "after_accepted_coarse": ("tab:orange", "-"),
            "late_cycle": ("tab:green", "-"),
        }
        for label, vals in categories.items():
            if vals:
                color, linestyle = styles[label]
                ax.hist(vals, bins=bins, density=True, histtype="step", lw=1.5, linestyle=linestyle, color=color, label=label)
        ax.set_xlabel(obs)
        ax.set_ylabel("density")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / f"{obs}_hist_by_state_category_vs_native_L32.pdf")
        plt.close(fig)
        print(f"wrote {fig_dir / f'{obs}_hist_by_state_category_vs_native_L32.pdf'}", flush=True)


def summarize_attempts(rows: list[dict[str, Any]], update_type: str) -> dict[str, Any]:
    sel = [r for r in rows if r["substep_type"] == update_type]
    accepted = np.asarray([int(r["accepted"]) for r in sel], dtype=np.float64)
    logR = np.asarray([float(r["logR"]) for r in sel], dtype=np.float64)
    longest, mean_streak = rejection_streaks(rows, len({int(r["chain_id"]) for r in rows}), update_type)
    return {
        "attempts": len(sel),
        "accepted": int(np.sum(accepted)),
        "acceptance": float(np.mean(accepted)) if len(accepted) else float("nan"),
        "binomial_se": float(math.sqrt(max(0.0, float(np.mean(accepted)) * (1.0 - float(np.mean(accepted)))) / len(accepted))) if len(accepted) else float("nan"),
        "logR_mean": float(np.mean(logR)) if len(logR) else float("nan"),
        "logR_std": float(np.std(logR, ddof=1)) if len(logR) > 1 else 0.0,
        "logR_min": float(np.min(logR)) if len(logR) else float("nan"),
        "logR_max": float(np.max(logR)) if len(logR) else float("nan"),
        "frac_logR_ge_0": float(np.mean(logR >= 0.0)) if len(logR) else float("nan"),
        "longest_rejection_streak": longest,
        "mean_rejection_streak": mean_streak,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--flow-checkpoint", type=Path, default=DEFAULT_FLOW)
    ap.add_argument("--kernel", type=Path, default=DEFAULT_KERNEL)
    ap.add_argument("--native-l16", type=Path, default=DEFAULT_NATIVE_L16)
    ap.add_argument("--native-l32", type=Path, default=DEFAULT_NATIVE_L32)
    ap.add_argument("--n-chains", type=int, default=10)
    ap.add_argument("--cycles", type=int, default=20)
    ap.add_argument("--detail-per-cycle", type=int, default=5)
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20260720)
    args = ap.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"writing to {out_dir}", flush=True)

    device = torch.device("cpu")
    action_f = ActionSpec("phi4_nn", 1.0, 0.340301)
    action_c = ActionSpec("phi4_nn", 1.0, 0.340301)
    kernel, _kernel_json = load_kernel(args.kernel)
    ckpt = torch.load(args.flow_checkpoint, map_location=device, weights_only=False)
    model, load_report = build_model_from_checkpoint(ckpt, lattice_size=16, device=device)
    stats = stationary_stats(ckpt["state"]["stats"], lc=16)
    l16 = load_phi(args.native_l16)
    l32 = load_phi(args.native_l32)
    rng = np.random.default_rng(args.seed)
    source_idx = np.arange(args.start_index, args.start_index + args.n_chains, dtype=np.int64)
    current_coarse = l16[source_idx].astype(np.float32)
    current_coarse_source = source_idx.copy()
    current_sc = action_total(current_coarse, action_c).astype(np.float64)
    current_z = draw_latents(args.n_chains, 16, rng)
    current = flow_from_latents(model, current_coarse, current_z, stats, kernel, action_f, device=device)
    print("initialized alternating chains", flush=True)

    all_attempt_rows: list[dict[str, Any]] = []
    detail_attempt_rows: list[dict[str, Any]] = []
    coarse_attempt_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    state_rows.extend(
        state_rows_for(
            chain_indices=range(args.n_chains),
            cycle=0,
            substep_type="initial",
            substep_index=0,
            attempt_global=0,
            accepted=np.zeros(args.n_chains, dtype=bool),
            coarse_source_indices=current_coarse_source,
            coarse_action=current_sc,
            current=current,
            category="initial",
        )
    )

    max_reblock = float(np.max(current["reblock_error"]))
    total_nonfinite = int(np.sum(current["nonfinite"] > 0))
    attempt_global = 0
    restore_errors: list[float] = []
    detail_c_errors: list[float] = []
    coarse_z_errors: list[float] = []

    for cycle in range(1, args.cycles + 1):
        for dstep in range(1, args.detail_per_cycle + 1):
            attempt_global += 1
            old = copy_state(current)
            old_coarse = current_coarse.copy()
            old_z = current_z.copy()
            proposal_z = draw_latents(args.n_chains, 16, rng)
            proposal = flow_from_latents(model, current_coarse, proposal_z, stats, kernel, action_f, device=device)
            logR = -proposal["S_f"] + current["S_f"] + current["logq"] - proposal["logq"]
            accepted = np.log(rng.random(args.n_chains)) < np.minimum(0.0, logR)
            restore_error = np.zeros(args.n_chains, dtype=np.float64)
            if np.any(~accepted):
                restore_error[~accepted] = np.max(np.abs(current["phi"][~accepted] - old["phi"][~accepted]), axis=(1, 2))
            set_accepted(current, proposal, accepted)
            current_z[accepted] = proposal_z[accepted]
            c_error = np.max(np.abs(current_coarse - old_coarse), axis=(1, 2))
            detail_c_errors.extend(c_error.tolist())
            invariant_error = np.zeros(args.n_chains, dtype=np.float64)
            z_same = np.zeros(args.n_chains, dtype=np.float64)
            append_attempt_rows(
                target=detail_attempt_rows,
                update_type="detail",
                cycle=cycle,
                substep_index=dstep,
                proposal_indices=current_coarse_source,
                old_coarse_source_indices=current_coarse_source,
                new_coarse_source_indices=current_coarse_source,
                old_coarse_action=current_sc,
                new_coarse_action=current_sc,
                old=old,
                proposal=proposal,
                logR=logR,
                accepted=accepted,
                restore_error=restore_error,
                invariant_error=invariant_error,
                z_unchanged_error=z_same,
                c_unchanged_error=c_error,
            )
            all_attempt_rows.extend(detail_attempt_rows[-args.n_chains :])
            state_rows.extend(
                state_rows_for(
                    chain_indices=range(args.n_chains),
                    cycle=cycle,
                    substep_type="detail",
                    substep_index=dstep,
                    attempt_global=attempt_global,
                    accepted=accepted,
                    coarse_source_indices=current_coarse_source,
                    coarse_action=current_sc,
                    current=current,
                    category="late_cycle" if cycle >= max(1, args.cycles - 4) else "after_detail",
                )
            )
            max_reblock = max(max_reblock, float(np.max(proposal["reblock_error"])))
            total_nonfinite += int(np.sum(proposal["nonfinite"] > 0))
            restore_errors.extend(restore_error.tolist())

        attempt_global += 1
        old = copy_state(current)
        old_coarse = current_coarse.copy()
        old_coarse_source = current_coarse_source.copy()
        old_sc = current_sc.copy()
        old_z = current_z.copy()
        proposal_source = rng.integers(0, len(l16), size=args.n_chains, dtype=np.int64)
        proposal_coarse = l16[proposal_source].astype(np.float32)
        proposal_sc = action_total(proposal_coarse, action_c).astype(np.float64)
        proposal = flow_from_latents(model, proposal_coarse, current_z, stats, kernel, action_f, device=device)
        logR = -proposal["S_f"] + current["S_f"] + (proposal["logJ"] - current["logJ"]) + (proposal_sc - current_sc)
        accepted = np.log(rng.random(args.n_chains)) < np.minimum(0.0, logR)
        restore_error = np.zeros(args.n_chains, dtype=np.float64)
        if np.any(~accepted):
            restore_error[~accepted] = np.max(np.abs(current["phi"][~accepted] - old["phi"][~accepted]), axis=(1, 2))
        set_accepted(current, proposal, accepted)
        current_coarse[accepted] = proposal_coarse[accepted]
        current_coarse_source[accepted] = proposal_source[accepted]
        current_sc[accepted] = proposal_sc[accepted]
        z_error = np.max(np.abs(current_z - old_z), axis=(1, 2))
        coarse_z_errors.extend(z_error.tolist())
        invariant = np.zeros(args.n_chains, dtype=np.float64)
        if np.any(accepted):
            check = flow_from_latents(model, current_coarse[accepted], current_z[accepted], stats, kernel, action_f, device=device)
            invariant[accepted] = np.max(np.abs(check["phi"] - current["phi"][accepted]), axis=(1, 2))
        append_attempt_rows(
            target=coarse_attempt_rows,
            update_type="coarse",
            cycle=cycle,
            substep_index=1,
            proposal_indices=proposal_source,
            old_coarse_source_indices=old_coarse_source,
            new_coarse_source_indices=np.where(accepted, proposal_source, current_coarse_source),
            old_coarse_action=old_sc,
            new_coarse_action=proposal_sc,
            old=old,
            proposal=proposal,
            logR=logR,
            accepted=accepted,
            restore_error=restore_error,
            invariant_error=invariant,
            z_unchanged_error=z_error,
            c_unchanged_error=np.zeros(args.n_chains, dtype=np.float64),
        )
        all_attempt_rows.extend(coarse_attempt_rows[-args.n_chains :])
        category = np.array(["after_accepted_coarse" if ok else ("late_cycle" if cycle >= max(1, args.cycles - 4) else "after_detail") for ok in accepted])
        state_rows.extend(
            state_rows_for_per_chain_category(
                cycle=cycle,
                substep_type="coarse",
                substep_index=1,
                attempt_global=attempt_global,
                accepted=accepted,
                coarse_source_indices=current_coarse_source,
                coarse_action=current_sc,
                current=current,
                categories=category,
            )
        )
        max_reblock = max(max_reblock, float(np.max(proposal["reblock_error"])))
        total_nonfinite += int(np.sum(proposal["nonfinite"] > 0))
        restore_errors.extend(restore_error.tolist())
        print(
            f"cycle {cycle}: detail accepted {sum(int(r['accepted']) for r in detail_attempt_rows if int(r['cycle']) == cycle)}/{args.detail_per_cycle * args.n_chains}; "
            f"coarse accepted {int(np.sum(accepted))}/{args.n_chains}",
            flush=True,
        )

    write_csv(out_dir / "all_attempts.csv", all_attempt_rows)
    write_csv(out_dir / "detail_attempts.csv", detail_attempt_rows)
    write_csv(out_dir / "coarse_attempts.csv", coarse_attempt_rows)
    write_csv(out_dir / "chain_state_observables.csv", state_rows)
    print("wrote attempt and state CSVs", flush=True)

    by_chain = []
    for chain in range(args.n_chains):
        detail_sel = [r for r in detail_attempt_rows if int(r["chain_id"]) == chain]
        coarse_sel = [r for r in coarse_attempt_rows if int(r["chain_id"]) == chain]
        by_chain.append(
            {
                "chain_id": chain,
                "initial_coarse_source_index": int(source_idx[chain]),
                "final_coarse_source_index": int(current_coarse_source[chain]),
                "detail_attempts": len(detail_sel),
                "detail_accepted": sum(int(r["accepted"]) for r in detail_sel),
                "detail_acceptance": float(np.mean([int(r["accepted"]) for r in detail_sel])),
                "coarse_attempts": len(coarse_sel),
                "coarse_accepted": sum(int(r["accepted"]) for r in coarse_sel),
                "coarse_acceptance": float(np.mean([int(r["accepted"]) for r in coarse_sel])),
            }
        )
    write_csv(out_dir / "acceptance_by_chain.csv", by_chain)
    make_plots(out_dir, state_rows, l32.astype(np.float32))

    detail_summary = summarize_attempts(all_attempt_rows, "detail")
    coarse_summary = summarize_attempts(all_attempt_rows, "coarse")
    manifest = {
        "command": " ".join(sys.argv),
        "flow_checkpoint": str(args.flow_checkpoint),
        "flow_sha256": sha256(args.flow_checkpoint),
        "flow_epoch": ckpt.get("epoch", ckpt.get("absolute_epoch")),
        "kernel": str(args.kernel),
        "kernel_sha256": sha256(args.kernel),
        "kernel_sum": float(kernel_stencil_from_spec(kernel).sum()),
        "kernel_coefficients_include_eta_scale": bool(kernel.kernel_coefficients_include_eta_scale),
        "native_l16": str(args.native_l16),
        "native_l32": str(args.native_l32),
        "n_chains": args.n_chains,
        "cycles": args.cycles,
        "schedule": f"D^{args.detail_per_cycle},C",
        "start_index": args.start_index,
        "source_indices": source_idx.tolist(),
        "seed": args.seed,
        "lambda": 1.0,
        "kappa_c": 0.340301,
        "kappa_f": 0.340301,
        "model_load_report": {k: str(v) for k, v in load_report.items()},
    }
    write_text(out_dir / "run_manifest.json", json.dumps(manifest, indent=2) + "\n")

    summary = [
        "# Small L16->L32 Alternating MIT Detail/Coarse Chain",
        "",
        "This is a short alternating MIT-style diagnostic at lambda=1.0. It uses fixed-latent coarse replacement moves with the direct coarse-action proposal-density ratio and fixed-coarse conditional detail independence moves. It is not a production chain.",
        "",
        f"- chains: `{args.n_chains}`",
        f"- cycles: `{args.cycles}`",
        f"- schedule: `D^{args.detail_per_cycle}, C`",
        f"- detail attempts/accepted/acceptance: `{detail_summary['attempts']}` / `{detail_summary['accepted']}` / `{detail_summary['acceptance']:.6g}`",
        f"- coarse attempts/accepted/acceptance: `{coarse_summary['attempts']}` / `{coarse_summary['accepted']}` / `{coarse_summary['acceptance']:.6g}`",
        f"- detail logR mean/std/min/max: `{detail_summary['logR_mean']:.6g}` / `{detail_summary['logR_std']:.6g}` / `{detail_summary['logR_min']:.6g}` / `{detail_summary['logR_max']:.6g}`",
        f"- coarse logR mean/std/min/max: `{coarse_summary['logR_mean']:.6g}` / `{coarse_summary['logR_std']:.6g}` / `{coarse_summary['logR_min']:.6g}` / `{coarse_summary['logR_max']:.6g}`",
        f"- coarse fraction logR >= 0: `{coarse_summary['frac_logR_ge_0']:.6g}`",
        f"- detail longest/mean rejection streak: `{detail_summary['longest_rejection_streak']}` / `{detail_summary['mean_rejection_streak']:.6g}`",
        f"- coarse longest/mean rejection streak: `{coarse_summary['longest_rejection_streak']}` / `{coarse_summary['mean_rejection_streak']:.6g}`",
        f"- maximum reblocking error: `{max_reblock:.6g}`",
        f"- nonfinite proposals: `{total_nonfinite}`",
        f"- max rejected-state restore error: `{float(np.max(restore_errors)) if restore_errors else 0.0:.6g}`",
        f"- max detail-update coarse-change error: `{float(np.max(detail_c_errors)) if detail_c_errors else 0.0:.6g}`",
        f"- max coarse-update latent-change error: `{float(np.max(coarse_z_errors)) if coarse_z_errors else 0.0:.6g}`",
        f"- flow checkpoint: `{args.flow_checkpoint}`",
        f"- kernel: `{args.kernel}`",
        "",
        "## Per-chain Acceptance",
        "",
        "| chain | initial coarse | final coarse | detail acc | coarse acc |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in by_chain:
        summary.append(f"| {row['chain_id']} | {row['initial_coarse_source_index']} | {row['final_coarse_source_index']} | {row['detail_accepted']}/{row['detail_attempts']} ({row['detail_acceptance']:.3f}) | {row['coarse_accepted']}/{row['coarse_attempts']} ({row['coarse_acceptance']:.3f}) |")
    write_text(out_dir / "summary.md", "\n".join(summary) + "\n")
    print(out_dir, flush=True)
    print("\n".join(summary[:22]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
