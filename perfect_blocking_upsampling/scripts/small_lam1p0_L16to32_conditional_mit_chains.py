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
from run_lam1p0_l16to32_rqspline_zeroshot import build_model_from_checkpoint, per_config_observables, sample_model_lattice, stationary_stats  # noqa: E402
from train_lam1p0_flow_detail_pilot import assemble_psi, load_phi  # noqa: E402


DEFAULT_FLOW = PROJECT_ROOT / "perfect_blocking_upsampling/runs/lam1p0/training/lam1p0_L16to32_newkernel_rqspline_N2000_action_lowtail_from_N2000_bestpatch_20260719T171401Z/checkpoints/checkpoint_best_patch.pt"
DEFAULT_KERNEL = PROJECT_ROOT / "perfect_blocking/perfect_blocking_lam1p0/kernels/final/chosen_kernel.json"
DEFAULT_NATIVE_L16 = PROJECT_ROOT / "data/configs_phi4_2d/lam1p0_kappac0p340301_L16/configs.npz"
DEFAULT_NATIVE_L32 = PROJECT_ROOT / "data/configs_phi4_2d/lam1p0_kappac0p340301_L32/configs.npz"
DEFAULT_OUT = PROJECT_ROOT / "perfect_blocking/perfect_blocking_lam1p0/tests/final/small_mit_nf_L16to32_conditional_chains_20260720"
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


def finite_noncount(phi: np.ndarray) -> np.ndarray:
    return np.sum(~np.isfinite(phi).reshape(len(phi), -1), axis=1)


def generate_proposals(
    *,
    model: Any,
    coarse: np.ndarray,
    stats: dict[str, Any],
    kernel: Any,
    action: ActionSpec,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    detail, logq, zmax, logj = sample_model_lattice(model, coarse, stats, batch_size=batch_size, device=device, seed=seed)
    psi = assemble_psi(coarse, detail).astype(np.float32)
    phi, _ = inverse_kernel(psi, kernel)
    blocked = apply_kernel(phi, kernel)[:, 0::2, 0::2]
    sf = action_total(phi, action).astype(np.float64)
    log_gaussian = logq + logj
    return {
        "detail": detail,
        "phi": phi.astype(np.float32),
        "S_f": sf,
        "logq": logq.astype(np.float64),
        "logj": logj.astype(np.float64),
        "log_gaussian": log_gaussian.astype(np.float64),
        "zmax": zmax.astype(np.float64),
        "reblock_error": np.max(np.abs(blocked.astype(np.float64) - coarse.astype(np.float64)), axis=(1, 2)),
        "nonfinite": finite_noncount(phi),
    }


def rows_for_states(
    *,
    step: int,
    source_indices: np.ndarray,
    phi: np.ndarray,
    sf: np.ndarray,
    logq: np.ndarray,
    logj: np.ndarray,
    log_gaussian: np.ndarray,
    reblock_error: np.ndarray,
) -> list[dict[str, Any]]:
    obs, _g = per_config_observables(phi, ActionSpec("phi4_nn", 1.0, 0.340301))
    rows = []
    for chain in range(len(phi)):
        row = {
            "chain_id": chain,
            "step": step,
            "coarse_config_index": int(source_indices[chain]),
            "S_f": float(sf[chain]),
            "log_gaussian_density": float(log_gaussian[chain]),
            "log_forward_jacobian": float(logj[chain]),
            "log_proposal_density": float(logq[chain]),
            "reblocking_max_error": float(reblock_error[chain]),
            "nonfinite_count": int(finite_noncount(phi[chain : chain + 1])[0]),
        }
        for key in OBS:
            row[key] = float(obs[key][chain])
        rows.append(row)
    return rows


def histogram_edges(samples: list[np.ndarray]) -> np.ndarray:
    vals = np.concatenate([np.asarray(s, dtype=np.float64)[np.isfinite(s)] for s in samples])
    lo = float(np.min(vals))
    hi = float(np.max(vals))
    if lo == hi:
        return np.linspace(lo - 0.5, hi + 0.5, 61)
    return np.linspace(lo, hi, 81)


def make_plots(out_dir: Path, state_rows: list[dict[str, Any]], attempt_rows: list[dict[str, Any]], native32: np.ndarray, action: ActionSpec) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    native_obs, _ = per_config_observables(native32.astype(np.float32), action)
    chains = sorted({int(r["chain_id"]) for r in state_rows})
    steps = sorted({int(r["step"]) for r in state_rows})

    for obs in OBS:
        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        for chain in chains:
            vals = [r[obs] for r in state_rows if int(r["chain_id"]) == chain]
            ax.plot(steps, vals, lw=1.0, alpha=0.75)
        ax.set_xlabel("Metropolis step")
        ax.set_ylabel(obs)
        ax.set_title(f"{obs} by fixed L16 coarse configuration")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(fig_dir / f"{obs}_vs_step_by_chain.pdf")
        plt.close(fig)
        print(f"wrote {fig_dir / f'{obs}_vs_step_by_chain.pdf'}", flush=True)

        chain_vals = np.asarray([float(r[obs]) for r in state_rows], dtype=np.float64)
        bins = histogram_edges([native_obs[obs], chain_vals])
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        ax.hist(native_obs[obs], bins=bins, density=True, histtype="step", lw=2.2, color="black", label="native L32")
        ax.hist(chain_vals, bins=bins, density=True, histtype="step", lw=1.8, color="tab:blue", label="conditional MIT states")
        ax.set_xlabel(obs)
        ax.set_ylabel("density")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(fig_dir / f"{obs}_hist_chain_states_vs_native_L32.pdf")
        plt.close(fig)
        print(f"wrote {fig_dir / f'{obs}_hist_chain_states_vs_native_L32.pdf'}", flush=True)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for chain in chains:
        xs = [int(r["step"]) for r in attempt_rows if int(r["chain_id"]) == chain]
        ys = [int(r["accepted"]) for r in attempt_rows if int(r["chain_id"]) == chain]
        ax.step(xs, ys, where="mid", lw=1.0, alpha=0.8, label=f"chain {chain}")
    ax.set_xlabel("proposal step")
    ax.set_ylabel("accepted")
    ax.set_yticks([0, 1])
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "acceptance_history_by_chain.pdf")
    plt.close(fig)
    print(f"wrote {fig_dir / 'acceptance_history_by_chain.pdf'}", flush=True)


def rejection_streaks(attempt_rows: list[dict[str, Any]], n_chains: int) -> tuple[list[int], int]:
    streaks: list[int] = []
    longest = 0
    for chain in range(n_chains):
        run = 0
        for row in sorted((r for r in attempt_rows if int(r["chain_id"]) == chain), key=lambda r: int(r["step"])):
            if int(row["accepted"]) == 0:
                run += 1
            elif run:
                streaks.append(run)
                longest = max(longest, run)
                run = 0
        if run:
            streaks.append(run)
            longest = max(longest, run)
    return streaks, longest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--flow-checkpoint", type=Path, default=DEFAULT_FLOW)
    ap.add_argument("--kernel", type=Path, default=DEFAULT_KERNEL)
    ap.add_argument("--native-l16", type=Path, default=DEFAULT_NATIVE_L16)
    ap.add_argument("--native-l32", type=Path, default=DEFAULT_NATIVE_L32)
    ap.add_argument("--n-chains", type=int, default=10)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20260720)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"writing to {out_dir}", flush=True)

    device = torch.device("cpu")
    action = ActionSpec("phi4_nn", 1.0, 0.340301)
    kernel, _kernel_json = load_kernel(args.kernel)
    ckpt = torch.load(args.flow_checkpoint, map_location=device, weights_only=False)
    model, load_report = build_model_from_checkpoint(ckpt, lattice_size=16, device=device)
    stats = stationary_stats(ckpt["state"]["stats"], lc=16)
    l16 = load_phi(args.native_l16)
    l32 = load_phi(args.native_l32)
    source_indices = np.arange(args.start_index, args.start_index + args.n_chains, dtype=np.int64)
    coarse = l16[source_indices].astype(np.float32)
    native32 = l32.astype(np.float32)
    print("loaded data, kernel, and checkpoint", flush=True)

    current = generate_proposals(model=model, coarse=coarse, stats=stats, kernel=kernel, action=action, batch_size=args.batch_size, seed=args.seed, device=device)
    current_phi = current["phi"].copy()
    current_sf = current["S_f"].copy()
    current_logq = current["logq"].copy()
    current_logj = current["logj"].copy()
    current_log_gaussian = current["log_gaussian"].copy()
    current_reblock = current["reblock_error"].copy()
    state_rows = rows_for_states(
        step=0,
        source_indices=source_indices,
        phi=current_phi,
        sf=current_sf,
        logq=current_logq,
        logj=current_logj,
        log_gaussian=current_log_gaussian,
        reblock_error=current_reblock,
    )
    attempt_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(args.seed + 1000003)
    total_nonfinite_proposals = int(np.sum(current["nonfinite"] > 0))
    max_reblock = float(np.max(current_reblock))
    print("initialized chains", flush=True)

    for step in range(1, args.steps + 1):
        proposal = generate_proposals(
            model=model,
            coarse=coarse,
            stats=stats,
            kernel=kernel,
            action=action,
            batch_size=args.batch_size,
            seed=args.seed + step,
            device=device,
        )
        logR = -proposal["S_f"] + current_sf + current_logq - proposal["logq"]
        logu = np.log(rng.random(args.n_chains))
        accepted = logu < np.minimum(0.0, logR)
        total_nonfinite_proposals += int(np.sum(proposal["nonfinite"] > 0))
        max_reblock = max(max_reblock, float(np.max(proposal["reblock_error"])))
        for chain in range(args.n_chains):
            attempt_rows.append(
                {
                    "chain_id": chain,
                    "step": step,
                    "coarse_config_index": int(source_indices[chain]),
                    "S_f_old": float(current_sf[chain]),
                    "S_f_new": float(proposal["S_f"][chain]),
                    "log_q_old": float(current_logq[chain]),
                    "log_q_new": float(proposal["logq"][chain]),
                    "log_gaussian_old": float(current_log_gaussian[chain]),
                    "log_gaussian_new": float(proposal["log_gaussian"][chain]),
                    "logJ_old": float(current_logj[chain]),
                    "logJ_new": float(proposal["logj"][chain]),
                    "logR": float(logR[chain]),
                    "acceptance_probability": float(min(1.0, math.exp(min(0.0, float(logR[chain]))))),
                    "uniform": float(math.exp(logu[chain])),
                    "accepted": int(accepted[chain]),
                    "proposal_reblocking_max_error": float(proposal["reblock_error"][chain]),
                    "proposal_nonfinite_count": int(proposal["nonfinite"][chain]),
                    "proposal_max_abs_z": float(proposal["zmax"][chain]),
                }
            )
        if np.any(accepted):
            current_phi[accepted] = proposal["phi"][accepted]
            current_sf[accepted] = proposal["S_f"][accepted]
            current_logq[accepted] = proposal["logq"][accepted]
            current_logj[accepted] = proposal["logj"][accepted]
            current_log_gaussian[accepted] = proposal["log_gaussian"][accepted]
            current_reblock[accepted] = proposal["reblock_error"][accepted]
        state_rows.extend(
            rows_for_states(
                step=step,
                source_indices=source_indices,
                phi=current_phi,
                sf=current_sf,
                logq=current_logq,
                logj=current_logj,
                log_gaussian=current_log_gaussian,
                reblock_error=current_reblock,
            )
        )
        print(f"step {step}: accepted {int(np.sum(accepted))}/{args.n_chains}", flush=True)

    write_csv(out_dir / "proposal_attempts.csv", attempt_rows)
    write_csv(out_dir / "chain_state_observables.csv", state_rows)
    print("wrote proposal_attempts.csv and chain_state_observables.csv", flush=True)

    chain_rows = []
    for chain in range(args.n_chains):
        rows = [r for r in attempt_rows if int(r["chain_id"]) == chain]
        acc = np.asarray([int(r["accepted"]) for r in rows], dtype=np.float64)
        logrs = np.asarray([float(r["logR"]) for r in rows], dtype=np.float64)
        chain_rows.append(
            {
                "chain_id": chain,
                "coarse_config_index": int(source_indices[chain]),
                "attempts": len(rows),
                "accepted": int(np.sum(acc)),
                "acceptance_rate": float(np.mean(acc)),
                "logR_mean": float(np.mean(logrs)),
                "logR_std": float(np.std(logrs, ddof=1)) if len(logrs) > 1 else 0.0,
                "logR_min": float(np.min(logrs)),
                "logR_max": float(np.max(logrs)),
                "frac_logR_ge_0": float(np.mean(logrs >= 0.0)),
            }
        )
    write_csv(out_dir / "acceptance_by_chain.csv", chain_rows)
    print("wrote acceptance_by_chain.csv", flush=True)

    make_plots(out_dir, state_rows, attempt_rows, native32, action)

    acc = np.asarray([int(r["accepted"]) for r in attempt_rows], dtype=np.float64)
    logrs = np.asarray([float(r["logR"]) for r in attempt_rows], dtype=np.float64)
    streaks, longest_streak = rejection_streaks(attempt_rows, args.n_chains)
    p = float(np.mean(acc))
    binom_se = float(math.sqrt(max(0.0, p * (1.0 - p)) / max(1, len(acc))))

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
        "steps": args.steps,
        "start_index": args.start_index,
        "source_indices": source_indices.tolist(),
        "seed": args.seed,
        "lambda": 1.0,
        "kappa_f": 0.340301,
        "model_load_report": {k: str(v) for k, v in load_report.items()},
    }
    write_text(out_dir / "run_manifest.json", json.dumps(manifest, indent=2) + "\n")
    print("wrote run_manifest.json", flush=True)

    summary = [
        "# Small L16->L32 Conditional MIT Independence Chains",
        "",
        "These are short conditional chains at fixed direct-generated L16 coarse fields. They test the MIT detail proposal and Metropolis accept/reject correction, not the full sampler in which the coarse field is also redrawn.",
        "",
        f"- fixed coarse fields / chains: `{args.n_chains}`",
        f"- proposal steps per chain: `{args.steps}`",
        f"- total attempts: `{len(attempt_rows)}`",
        f"- accepted proposals: `{int(np.sum(acc))}`",
        f"- overall acceptance rate: `{p:.6g}`",
        f"- binomial standard error: `{binom_se:.6g}`",
        f"- min/max chain acceptance: `{min(r['acceptance_rate'] for r in chain_rows):.6g}` / `{max(r['acceptance_rate'] for r in chain_rows):.6g}`",
        f"- rejection streak count: `{len(streaks)}`",
        f"- longest rejection streak: `{longest_streak}`",
        f"- mean rejection streak length: `{float(np.mean(streaks)) if streaks else 0.0:.6g}`",
        f"- logR mean/std: `{float(np.mean(logrs)):.6g}` / `{float(np.std(logrs, ddof=1)):.6g}`",
        f"- logR min/max: `{float(np.min(logrs)):.6g}` / `{float(np.max(logrs)):.6g}`",
        f"- fraction logR >= 0: `{float(np.mean(logrs >= 0.0)):.6g}`",
        f"- maximum reblocking error: `{max_reblock:.6g}`",
        f"- nonfinite proposals: `{total_nonfinite_proposals}`",
        f"- flow checkpoint: `{args.flow_checkpoint}`",
        f"- kernel: `{args.kernel}`",
        "",
        "## Acceptance by Chain",
        "",
        "| chain | coarse config | accepted / attempts | acceptance | logR mean | logR std | logR min | logR max | frac logR >= 0 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in chain_rows:
        summary.append(
            f"| {row['chain_id']} | {row['coarse_config_index']} | {row['accepted']} / {row['attempts']} | {row['acceptance_rate']:.6g} | {row['logR_mean']:.6g} | {row['logR_std']:.6g} | {row['logR_min']:.6g} | {row['logR_max']:.6g} | {row['frac_logR_ge_0']:.6g} |"
        )
    write_text(out_dir / "summary.md", "\n".join(summary) + "\n")
    print(out_dir, flush=True)
    print("\n".join(summary[:24]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
