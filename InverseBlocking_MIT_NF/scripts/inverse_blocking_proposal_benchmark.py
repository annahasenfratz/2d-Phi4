#!/usr/bin/env python3
"""Benchmark driver for inverse-blocking proposals.

Baseline:
    local projected-Haar null proposal + constrained correction

The driver benchmarks proposal-quality and correction-cost at a fixed set of
constrained MCMC sweep counts, using the paired fine/coarse data already
available in ``outputs/paired_data_lam1_kappaf0p320``.

Future learned models can plug into the same benchmark by providing either:
    - ``--initial-v-npy``: null coordinates in the projected-Haar basis, or
    - ``--initial-phi-npy``: proposal fields on the 16x16 lattice.

The correction loop, observables, and cost accounting stay unchanged.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
if str(PROJECT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT / "scripts"))
import local_nullspace_pilot as local_pilot  # type: ignore
import nullspace_conditional_nf_pilot as pilot  # type: ignore


DATA = PROJECT / "outputs" / "paired_data_lam1_kappaf0p320"
DEFAULT_OUT = PROJECT / "outputs" / "inverse_blocking_proposal_benchmark"
DEFAULT_SWEEPS = [0, 5, 10, 25, 50]
DEFAULT_GROUP_SIZE = 6
DEFAULT_STEP_SIZE = 0.1
DEFAULT_N_CONDITIONS = 128
SEED = 20240624

KERNEL = ROOT / "perfect_blocking" / "perfect_blocking_lam1p0_blockavg" / "perfect_block_lam1_blockavg_kernel5x5_kernel.json"
N_DENSE_BASIS = PROJECT / "outputs" / "nullspace_conditional_nf_pilot" / "preflight" / "null_basis.npy"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def parse_sweeps(text: str) -> list[int]:
    sweeps = sorted({int(x) for x in text.split(",") if x.strip()})
    if not sweeps:
        raise ValueError("sweeps list is empty")
    if sweeps[0] != 0:
        sweeps = [0] + sweeps
    return sweeps


def load_weights() -> dict[str, float]:
    meta = json.loads(KERNEL.read_text())
    return {k: float(meta["weights"][k]) for k in ["w00", "w10", "w11", "w20", "w21", "w22"]}


def load_paired_data() -> dict[str, np.ndarray]:
    fine = np.load(DATA / "fine_configs.npy").astype(np.float64)
    coarse = np.load(DATA / "coarse_blocked_configs.npy").astype(np.float64)
    back = np.load(DATA / "backbone_configs.npy").astype(np.float64)
    v_true = np.load(DATA / "residual_v_true.npy").astype(np.float64)
    splits_npz = np.load(DATA / "split_indices.npz")
    splits = {k: splits_npz[k].astype(int) for k in ["train", "val", "test"]}
    return {
        "fine": fine,
        "coarse": coarse,
        "backbone": back,
        "v_true": v_true,
        "train_idx": splits["train"],
        "val_idx": splits["val"],
        "test_idx": splits["test"],
    }


def build_local_q_basis(w: dict[str, float]) -> tuple[np.ndarray, dict[str, float]]:
    b = pilot.build_B(w)
    p_null = np.eye(256) - b.T @ np.linalg.inv(b @ b.T) @ b
    h = local_pilot.haar_detail_basis()
    m = p_null @ h
    q, _ = np.linalg.qr(m, mode="reduced")
    diag = {
        "rank_M": int(np.linalg.matrix_rank(m, tol=1.0e-10)),
        "cond_MtM": float(np.linalg.cond(m.T @ m)),
        "max_abs_BQ": float(np.max(np.abs(b @ q))),
        "rms_BQ": float(np.sqrt(np.mean((b @ q) ** 2))),
        "orthonormal_error": float(np.max(np.abs(q.T @ q - np.eye(q.shape[1])))),
        "local_basis_spread": locality_spread(q),
    }
    return q, diag


def locality_spread(a: np.ndarray) -> float:
    cols = a.reshape(16, 16, a.shape[1])
    coords_y, coords_x = np.meshgrid(np.arange(16), np.arange(16), indexing="ij")
    vals = []
    for j in range(a.shape[1]):
        p = cols[:, :, j] ** 2
        p = p / p.sum()
        my = float((p * coords_y).sum())
        mx = float((p * coords_x).sum())
        vals.append(float((p * ((coords_y - my) ** 2 + (coords_x - mx) ** 2)).sum()))
    return float(np.mean(vals))


def obs_with_m(phi: np.ndarray) -> dict[str, float]:
    obs = pilot.obs_np(phi.astype(np.float64))
    m = phi.mean(axis=(-2, -1))
    obs["m"] = float(np.mean(m))
    obs["abs_m"] = float(np.mean(np.abs(m)))
    return obs


def ensemble_metrics(phi: np.ndarray, coarse: np.ndarray | None, w: dict[str, float] | None) -> dict[str, float]:
    obs = obs_with_m(phi)
    if coarse is not None and w is not None and phi.shape[-2:] == (16, 16):
        br = pilot.block_sym_np(phi.astype(np.float64), w) - coarse
        obs["block_RMS"] = float(np.sqrt(np.mean(br**2)))
        obs["block_max"] = float(np.max(np.abs(br)))
    else:
        obs["block_RMS"] = math.nan
        obs["block_max"] = math.nan
    return obs


def pca_fit(v_train: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = v_train.mean(axis=0)
    cov = np.cov(v_train - mean, rowvar=False)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    return mean, np.maximum(evals[order], 0.0), evecs[:, order]


def sample_fixed_pca_v(
    rng: np.random.Generator,
    mean_v: np.ndarray,
    evals: np.ndarray,
    n: int,
    k: int = 32,
    scale: float = 0.9,
    eps_tail: float = 0.1,
) -> np.ndarray:
    sigmas = np.empty(192, dtype=np.float64)
    sigmas[:k] = scale * np.sqrt(np.maximum(evals[:k], 1.0e-12))
    tail_floor = np.sqrt(max(float(np.mean(evals[k:])), float(evals[k - 1]), 1.0e-12))
    sigmas[k:] = eps_tail * max(tail_floor, 1.0e-12)
    return mean_v + rng.normal(size=(n, 192)) * sigmas


def local_coords_from_dense(v_dense: np.ndarray, dense_basis: np.ndarray, q_basis: np.ndarray) -> np.ndarray:
    """Map dense null coordinates to projected-Haar coordinates."""
    return v_dense @ (dense_basis.T @ q_basis)


def fit_chunk_gaussians(u_train: np.ndarray, group_size: int) -> list[dict[str, np.ndarray]]:
    if u_train.shape[1] % group_size != 0:
        raise ValueError("group_size must divide the local-coordinate dimension")
    ridge = 1.0e-6
    models: list[dict[str, np.ndarray]] = []
    for start in range(0, u_train.shape[1], group_size):
        idx = slice(start, start + group_size)
        x = u_train[:, idx]
        mu = x.mean(axis=0)
        cov = np.cov(x - mu, rowvar=False)
        cov = np.asarray(cov, dtype=np.float64) + ridge * np.eye(group_size)
        models.append({"mu": mu, "cov": cov, "chol": np.linalg.cholesky(cov)})
    return models


def sample_chunk_gaussian(
    rng: np.random.Generator,
    models: list[dict[str, np.ndarray]],
    n: int,
    group_size: int,
) -> np.ndarray:
    u = np.zeros((n, group_size * len(models)), dtype=np.float64)
    for g, model in enumerate(models):
        z = rng.normal(size=(n, group_size))
        u[:, g * group_size : (g + 1) * group_size] = model["mu"][None, :] + z @ model["chol"].T
    return u


def action_totals(phi: np.ndarray) -> np.ndarray:
    vals = []
    with torch.no_grad():
        for start in range(0, len(phi), 64):
            batch = torch.tensor(phi[start : start + 64].astype(np.float32))
            s, _ = pilot.fine_action(batch)
            vals.append(s.cpu().numpy())
    return np.concatenate(vals)


def phi_from_local_coords(u: np.ndarray, back: np.ndarray, q_basis: np.ndarray) -> np.ndarray:
    detail = u @ q_basis.T
    return back + detail.reshape(len(u), 16, 16)


def acceptance_rate(accepted: int, attempts: int) -> float:
    return float(accepted / attempts) if attempts > 0 else math.nan


@dataclass
class ProposalSource:
    name: str
    u0: np.ndarray
    metadata: dict[str, object]


def build_builtin_baseline(
    rng: np.random.Generator,
    q_basis: np.ndarray,
    dense_basis: np.ndarray,
    v_true: np.ndarray,
    train_idx: np.ndarray,
    group_size: int,
    n_conditions: int,
) -> ProposalSource:
    u_train = local_coords_from_dense(v_true[train_idx], dense_basis, q_basis)
    models = fit_chunk_gaussians(u_train, group_size)
    u0 = sample_chunk_gaussian(rng, models, n_conditions, group_size)
    return ProposalSource(
        name=f"builtin_local_chunk_G{group_size}",
        u0=u0,
        metadata={
            "fit_basis": "projected_haar_Q",
            "fit_source": "paired_data_lam1_kappaf0p320/train",
            "local_coordinate_source": "dense_null_to_projected_haar",
            "group_size": group_size,
            "n_chunks": len(models),
        },
    )


def build_external_source(
    proposal_mode: str,
    args: argparse.Namespace,
    back_sel: np.ndarray,
    q_basis: np.ndarray,
    dense_basis: np.ndarray,
) -> ProposalSource:
    if proposal_mode == "external_v":
        if args.initial_v_npy is None:
            raise ValueError("--initial-v-npy is required for external_v")
        u0 = np.load(args.initial_v_npy).astype(np.float64)
        if u0.ndim != 2 or u0.shape[1] != 192:
            raise ValueError(f"Expected initial-v array with shape (N, 192), got {u0.shape}")
        if len(u0) != len(back_sel):
            raise ValueError("Initial-v array length must match the selected conditions")
        return ProposalSource(
            name=f"external_v:{Path(args.initial_v_npy).name}",
            u0=u0,
            metadata={"source_file": str(Path(args.initial_v_npy).resolve())},
        )
    if proposal_mode == "external_phi":
        if args.initial_phi_npy is None:
            raise ValueError("--initial-phi-npy is required for external_phi")
        phi0 = np.load(args.initial_phi_npy).astype(np.float64)
        if phi0.shape != back_sel.shape:
            raise ValueError(f"Expected initial-phi array with shape {back_sel.shape}, got {phi0.shape}")
        # Project the external field onto the exact projected-Haar null coordinates.
        u0 = (phi0.reshape(len(phi0), -1) - back_sel.reshape(len(back_sel), -1)) @ q_basis
        return ProposalSource(
            name=f"external_phi:{Path(args.initial_phi_npy).name}",
            u0=u0,
            metadata={"source_file": str(Path(args.initial_phi_npy).resolve())},
        )
    raise ValueError(f"Unsupported proposal mode {proposal_mode}")


def proposal_row_name(source: ProposalSource, sweep: int) -> str:
    return f"{source.name}_sweeps_{sweep}"


def averaged_metrics(arr: np.ndarray, coarse: np.ndarray, w: dict[str, float]) -> dict[str, float]:
    obs = ensemble_metrics(arr, coarse, w)
    obs["S_total_mean"] = float(np.mean(action_totals(arr)))
    return obs


def score_against_target(row: dict[str, float], target: dict[str, float]) -> float:
    keys = ["phi2", "phi4", "nn2"]
    return float(sum(abs(float(row[k]) - float(target[k])) for k in keys))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n-conditions", type=int, default=DEFAULT_N_CONDITIONS)
    parser.add_argument("--group-size", type=int, default=DEFAULT_GROUP_SIZE)
    parser.add_argument("--step-size", type=float, default=DEFAULT_STEP_SIZE)
    parser.add_argument("--sweeps", type=str, default=",".join(map(str, DEFAULT_SWEEPS)))
    parser.add_argument(
        "--proposal-mode",
        choices=["builtin_local_chunk", "external_v", "external_phi"],
        default="builtin_local_chunk",
        help="Initial proposal source. The benchmark correction loop is the same for all modes.",
    )
    parser.add_argument("--initial-v-npy", type=Path, default=None, help="External null-coordinate proposal, shape (N, 192).")
    parser.add_argument("--initial-phi-npy", type=Path, default=None, help="External 16x16 proposal field batch.")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    sweeps = parse_sweeps(args.sweeps)
    out = args.output_dir
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"Refusing to overwrite existing outputs in {out}")
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    w = load_weights()
    data = load_paired_data()
    fine = data["fine"]
    coarse = data["coarse"]
    back = data["backbone"]
    v_true = data["v_true"]
    train_idx = data["train_idx"]
    val_idx = data["val_idx"]
    test_idx = data["test_idx"]

    # The benchmark uses a deterministic condition selection over test, val, train.
    sel_idx = np.concatenate([test_idx, val_idx, train_idx])[: args.n_conditions]
    fine_sel = fine[sel_idx]
    coarse_sel = coarse[sel_idx]
    back_sel = back[sel_idx]

    q_basis, basis_diag = build_local_q_basis(w)
    dense_basis = np.load(N_DENSE_BASIS).astype(np.float64)
    dense_to_local = dense_basis.T @ q_basis

    # Fit the built-in baseline proposal on the projected-Haar local coordinates.
    builtin_source = build_builtin_baseline(
        rng=rng,
        q_basis=q_basis,
        dense_basis=dense_basis,
        v_true=v_true,
        train_idx=train_idx,
        group_size=args.group_size,
        n_conditions=len(sel_idx),
    )

    if args.proposal_mode == "builtin_local_chunk":
        source = builtin_source
    else:
        source = build_external_source(args.proposal_mode, args, back_sel, q_basis, dense_basis)

    # The correction loop operates in the projected-Haar null coordinates.
    u = source.u0.copy()
    phi = phi_from_local_coords(u, back_sel, q_basis)
    s = action_totals(phi)
    groups = [np.arange(i, min(i + args.group_size, 192)) for i in range(0, 192, args.group_size)]
    if any(len(g) != args.group_size for g in groups):
        raise ValueError("group-size must divide 192 for the correction loop")

    baseline_rows: list[dict[str, object]] = []
    correction_rows: list[dict[str, object]] = []
    accept_rows: list[dict[str, object]] = []
    block_rows: list[dict[str, object]] = []
    per_condition_rows: list[dict[str, object]] = []
    snapshots: dict[int, np.ndarray] = {0: phi.copy()}
    accepted = 0
    attempts = 0

    target_avg = averaged_metrics(fine_sel, coarse_sel, w)
    target_score = {k: target_avg[k] for k in ["phi2", "phi4", "nn2"]}

    # Proposal-only row for the selected batch.
    baseline_rows.append(
        {
            "ensemble": source.name,
            "sweeps": 0,
            "proposal_mode": args.proposal_mode,
            "group_size": args.group_size,
            "step_size": args.step_size,
            "acceptance_rate": math.nan,
            "action_evals_per_sample": 1,
            "proposal_cost_units": 1,
            "cost_units_per_sample": 1,
            "moment_score": score_against_target(averaged_metrics(phi, coarse_sel, w), target_score),
            **averaged_metrics(phi, coarse_sel, w),
        }
    )

    # Correction sweeps.
    max_sweep = max(sweeps)
    for sweep in range(1, max_sweep + 1):
        for gidx, sl in enumerate(groups):
            noise = np.zeros_like(u)
            noise[:, sl] = rng.normal(scale=args.step_size, size=(len(u), args.group_size))
            u_prop = u + noise
            phi_prop = phi_from_local_coords(u_prop, back_sel, q_basis)
            s_prop = action_totals(phi_prop)
            log_alpha = -(s_prop - s)
            accept = np.log(rng.random(len(u))) < np.minimum(0.0, log_alpha)
            accepted += int(np.sum(accept))
            attempts += len(u)
            u[accept] = u_prop[accept]
            phi[accept] = phi_prop[accept]
            s[accept] = s_prop[accept]

        if sweep in sweeps:
            snapshots[sweep] = phi.copy()
            obs_batch = averaged_metrics(phi, coarse_sel, w)
            correction_rows.append(
                {
                    "ensemble": proposal_row_name(source, sweep),
                    "sweeps": sweep,
                    "proposal_mode": args.proposal_mode,
                    "group_size": args.group_size,
                    "step_size": args.step_size,
                    "acceptance_rate": acceptance_rate(accepted, attempts),
                    "action_evals_per_sample": 1 + sweep * len(groups),
                    "proposal_cost_units": 1 + sweep * len(groups),
                    "cost_units_per_sample": 1 + sweep * len(groups),
                    "moment_score": score_against_target(obs_batch, target_score),
                    **obs_batch,
                }
            )
            accept_rows.append(
                {
                    "sweeps": sweep,
                    "proposal_mode": args.proposal_mode,
                    "group_size": args.group_size,
                    "step_size": args.step_size,
                    "accepted": accepted,
                    "attempts": attempts,
                    "acceptance_rate": acceptance_rate(accepted, attempts),
                    "action_evals_per_sample": 1 + sweep * len(groups),
                }
            )
            # Per-condition rows vs paired fine.
            for i in range(len(sel_idx)):
                sample_obs = ensemble_metrics(phi[i : i + 1], coarse_sel[i : i + 1], w)
                target_obs = ensemble_metrics(fine_sel[i : i + 1], coarse_sel[i : i + 1], w)
                row = {
                    "condition": int(i),
                    "sweep": sweep,
                    "proposal_mode": args.proposal_mode,
                    "group_size": args.group_size,
                    "step_size": args.step_size,
                    "accepted": accepted,
                    "attempts": attempts,
                    "acceptance_rate_running": acceptance_rate(accepted, attempts),
                    "action_evals_per_sample": 1 + sweep * len(groups),
                    "cost_units_per_sample": 1 + sweep * len(groups),
                    "sample_phi2": sample_obs["phi2"],
                    "sample_phi4": sample_obs["phi4"],
                    "sample_NN": sample_obs["NN"],
                    "sample_nn2": sample_obs["nn2"],
                    "sample_diag": sample_obs["diag"],
                    "sample_2nn": sample_obs["2nn"],
                    "sample_Binder_U4": sample_obs["Binder_U4"],
                    "sample_xi_over_L": sample_obs["xi/L"],
                    "sample_action_density": sample_obs["action_density"],
                    "sample_action_hopping_density": sample_obs["action_hopping_density"],
                    "sample_action_phi2_density": sample_obs["action_phi2_density"],
                    "sample_action_phi4_density": sample_obs["action_phi4_density"],
                    "sample_block_RMS": sample_obs["block_RMS"],
                    "sample_block_max": sample_obs["block_max"],
                    "target_phi2": target_obs["phi2"],
                    "target_phi4": target_obs["phi4"],
                    "target_NN": target_obs["NN"],
                    "target_nn2": target_obs["nn2"],
                    "target_diag": target_obs["diag"],
                    "target_2nn": target_obs["2nn"],
                    "target_Binder_U4": target_obs["Binder_U4"],
                    "target_xi_over_L": target_obs["xi/L"],
                    "target_action_density": target_obs["action_density"],
                    "target_action_hopping_density": target_obs["action_hopping_density"],
                    "target_action_phi2_density": target_obs["action_phi2_density"],
                    "target_action_phi4_density": target_obs["action_phi4_density"],
                    "target_block_RMS": target_obs["block_RMS"],
                    "target_block_max": target_obs["block_max"],
                    "delta_phi2": sample_obs["phi2"] - target_obs["phi2"],
                    "delta_phi4": sample_obs["phi4"] - target_obs["phi4"],
                    "delta_nn2": sample_obs["nn2"] - target_obs["nn2"],
                    "delta_action_density": sample_obs["action_density"] - target_obs["action_density"],
                    "delta_block_RMS": sample_obs["block_RMS"] - target_obs["block_RMS"],
                }
                per_condition_rows.append(row)
                block_rows.append(
                    {
                        "condition": int(i),
                        "sweep": sweep,
                        "proposal_mode": args.proposal_mode,
                        "group_size": args.group_size,
                        "step_size": args.step_size,
                        "block_RMS": sample_obs["block_RMS"],
                        "block_max": sample_obs["block_max"],
                    }
                )

    # Save snapshots at requested sweeps.
    for sweep in sweeps:
        np.save(out / f"samples_sweeps_{sweep}.npy", snapshots[sweep].astype(np.float32))

    # Reference rows for comparison.
    fixed_mean_v, fixed_evals, _ = pca_fit(v_true[train_idx])
    fixed_v = sample_fixed_pca_v(rng, fixed_mean_v, fixed_evals, len(sel_idx), k=32, scale=0.9, eps_tail=0.1)
    fixed_phi = back_sel + (fixed_v @ dense_basis.T).reshape(len(sel_idx), 16, 16)

    ref_rows = [
        {
            "ensemble": "paired_fine",
            "sweeps": 0,
            "proposal_mode": "reference",
            "group_size": math.nan,
            "step_size": math.nan,
            "acceptance_rate": math.nan,
            "action_evals_per_sample": math.nan,
            "proposal_cost_units": math.nan,
            "cost_units_per_sample": math.nan,
            "moment_score": 0.0,
            **target_avg,
        },
        {
            "ensemble": "smooth_backbone",
            "sweeps": 0,
            "proposal_mode": "reference",
            "group_size": math.nan,
            "step_size": math.nan,
            "acceptance_rate": math.nan,
            "action_evals_per_sample": math.nan,
            "proposal_cost_units": math.nan,
            "cost_units_per_sample": math.nan,
            "moment_score": score_against_target(averaged_metrics(back_sel, coarse_sel, w), target_score),
            **averaged_metrics(back_sel, coarse_sel, w),
        },
        {
            "ensemble": "fixed_PCA_K32_s0.9",
            "sweeps": 0,
            "proposal_mode": "reference",
            "group_size": math.nan,
            "step_size": math.nan,
            "acceptance_rate": math.nan,
            "action_evals_per_sample": math.nan,
            "proposal_cost_units": math.nan,
            "cost_units_per_sample": math.nan,
            "moment_score": score_against_target(averaged_metrics(fixed_phi, coarse_sel, w), target_score),
            **averaged_metrics(fixed_phi, coarse_sel, w),
        },
    ]

    proposal_rows = baseline_rows + correction_rows + ref_rows
    write_csv(out / "proposal_observables.csv", proposal_rows)
    write_csv(out / "observables_by_sweeps.csv", per_condition_rows)
    write_csv(out / "acceptance_by_sweeps.csv", accept_rows)
    write_csv(out / "block_residuals.csv", block_rows)

    summary = {
        "seed": args.seed,
        "proposal_mode": args.proposal_mode,
        "proposal_source": source.name,
        "proposal_metadata": source.metadata,
        "n_conditions": len(sel_idx),
        "selected_indices": sel_idx.tolist(),
        "group_size": args.group_size,
        "step_size": args.step_size,
        "sweeps": sweeps,
        "basis_diagnostics": basis_diag,
        "dense_basis_source": str(N_DENSE_BASIS.resolve()),
        "benchmark_definition": {
            "baseline": "local projected-Haar null proposal + constrained correction",
            "correction_coordinate_system": "projected_haar_Q",
            "future_plugin_interface": ["--initial-v-npy", "--initial-phi-npy"],
            "cost_units": "one action evaluation on a batch of selected conditions",
        },
        "reference_means": target_avg,
        "best_sweep_by_moment_score": min(correction_rows, key=lambda r: float(r["moment_score"])) if correction_rows else None,
        "acceptance_summary": accept_rows,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    best = min(correction_rows, key=lambda r: float(r["moment_score"])) if correction_rows else None
    baseline_avg = averaged_metrics(snapshots[0], coarse_sel, w)
    report = [
        "# Inverse-Blocking Proposal Benchmark",
        "",
        "Baseline:",
        f"- local projected-Haar null proposal + constrained correction",
        f"- group size: `G={args.group_size}`",
        f"- step size: `{args.step_size}`",
        f"- sweep checkpoints: {', '.join(map(str, sweeps))}",
        "",
        "Reference means on the selected paired-fine subset:",
        f"- phi2={target_avg['phi2']:.6g}, phi4={target_avg['phi4']:.6g}, nn2={target_avg['nn2']:.6g}",
        f"- NN={target_avg['NN']:.6g}, diag={target_avg['diag']:.6g}, 2nn={target_avg['2nn']:.6g}",
        "",
        "Proposal / correction summary:",
        f"- proposal source: `{source.name}`",
        f"- sweep 0 phi2/phi4/nn2: {baseline_avg['phi2']:.6g}, {baseline_avg['phi4']:.6g}, {baseline_avg['nn2']:.6g}",
    ]
    if best is not None:
        report.extend(
            [
                f"- best sweep by `phi2/phi4/nn2` score: {int(best['sweeps'])}",
                f"- best sweep observables: phi2={best['phi2']:.6g}, phi4={best['phi4']:.6g}, nn2={best['nn2']:.6g}",
                f"- best sweep acceptance: {float(best['acceptance_rate']):.6g}",
                f"- best sweep cost units/sample: {int(best['cost_units_per_sample'])}",
            ]
        )
    report.extend(
        [
            "",
            "Interpretation:",
            "This benchmark keeps the correction loop and metrics fixed so future learned proposals can be compared directly. ",
            "A future model should emit either projected-Haar null coordinates (`--initial-v-npy`) or proposal fields (`--initial-phi-npy`) and be evaluated through the same sweep ladder and observable tables.",
        ]
    )
    (out / "report.md").write_text("\n".join(report) + "\n")


if __name__ == "__main__":
    main()
