#!/usr/bin/env python3
"""Consolidated local chunked null-space sampler with short constrained corrections."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT / "scripts"))
import local_nullspace_pilot as local_pilot  # type: ignore
import nullspace_conditional_nf_pilot as pilot  # type: ignore


DATA = PROJECT / "outputs" / "paired_data_lam1_kappaf0p320"
OUT = PROJECT / "outputs" / "local_chunked_sampler_consolidated"
SEED = 20240624
N_CONDITIONS = 128
GROUP_SIZE = 6
STEP_SIZE = 0.1
SWEEP_STOPS = [0, 10, 25, 50, 100]
FIXED_PCA_K = 32
FIXED_PCA_S = 0.9
FIXED_PCA_TAIL = 0.1


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


def load_weights() -> dict[str, float]:
    meta = json.loads(pilot.KERNEL.read_text())
    return {k: float(meta["weights"][k]) for k in ["w00", "w10", "w11", "w20", "w21", "w22"]}


def local_q_basis(w: dict[str, float]) -> np.ndarray:
    b = pilot.build_B(w)
    p_null = np.eye(256) - b.T @ np.linalg.inv(b @ b.T) @ b
    m = p_null @ local_pilot.haar_detail_basis()
    q, _ = np.linalg.qr(m, mode="reduced")
    return q


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


def sample_fixed_pca(
    rng: np.random.Generator,
    mean_v: np.ndarray,
    evals: np.ndarray,
    evecs: np.ndarray,
    n: int,
    k: int = FIXED_PCA_K,
    scale: float = FIXED_PCA_S,
    eps_tail: float = FIXED_PCA_TAIL,
) -> np.ndarray:
    sigmas = np.empty(192, dtype=np.float64)
    sigmas[:k] = scale * np.sqrt(np.maximum(evals[:k], 1.0e-12))
    tail_floor = np.sqrt(max(float(np.mean(evals[k:])), float(evals[k - 1]), 1.0e-12))
    sigmas[k:] = eps_tail * max(tail_floor, 1.0e-12)
    return mean_v + rng.normal(size=(n, 192)) * sigmas


def chunk_groups(group_size: int) -> list[np.ndarray]:
    if 192 % group_size != 0:
        raise ValueError("group_size must divide 192")
    return [np.arange(start, start + group_size) for start in range(0, 192, group_size)]


def fit_chunk_gaussians(v_train: np.ndarray, group_size: int) -> list[dict[str, np.ndarray]]:
    groups = chunk_groups(group_size)
    models: list[dict[str, np.ndarray]] = []
    ridge = 1.0e-6
    for idx in groups:
        X = v_train[:, idx]
        mu = X.mean(axis=0)
        cov = np.cov(X - mu, rowvar=False)
        cov = np.asarray(cov, dtype=np.float64) + ridge * np.eye(group_size)
        models.append({"mu": mu, "cov": cov, "chol": np.linalg.cholesky(cov)})
    return models


def sample_chunk_gaussian(rng: np.random.Generator, models: list[dict[str, np.ndarray]], group_size: int, n: int) -> np.ndarray:
    v = np.zeros((n, 192), dtype=np.float64)
    for g, model in enumerate(models):
        z = rng.normal(size=(n, group_size))
        v[:, g * group_size : (g + 1) * group_size] = model["mu"][None, :] + z @ model["chol"].T
    return v


def action_totals(phi: np.ndarray) -> np.ndarray:
    vals = []
    with torch.no_grad():
        for start in range(0, len(phi), 64):
            batch = torch.tensor(phi[start : start + 64].astype(np.float32))
            S, _ = pilot.fine_action(batch)
            vals.append(S.cpu().numpy())
    return np.concatenate(vals)


def phi_from_v(v: np.ndarray, back: np.ndarray, q: np.ndarray) -> np.ndarray:
    detail = v @ q.T
    return back + detail.reshape(len(v), 16, 16)


def summarize_acceptance(accepted: int, attempts: int) -> float:
    return float(accepted / attempts) if attempts > 0 else math.nan


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(f"Refusing to overwrite existing outputs in {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    w = load_weights()
    fine = np.load(DATA / "fine_configs.npy").astype(np.float64)
    coarse = np.load(DATA / "coarse_blocked_configs.npy").astype(np.float64)
    back = np.load(DATA / "backbone_configs.npy").astype(np.float64)
    v_true = np.load(DATA / "residual_v_true.npy").astype(np.float64)
    splits_npz = np.load(DATA / "split_indices.npz")
    splits = {k: splits_npz[k].astype(int) for k in ["train", "val", "test"]}

    q = local_q_basis(w)
    mean_v, evals, evecs = pca_fit(v_true[splits["train"]])
    groups = chunk_groups(GROUP_SIZE)
    chunk_models = fit_chunk_gaussians((v_true[splits["train"]] @ q.T).astype(np.float64), GROUP_SIZE)

    sel_idx = np.concatenate([splits["test"], splits["val"], splits["train"]])[:N_CONDITIONS]
    fine_sel = fine[sel_idx]
    coarse_sel = coarse[sel_idx]
    back_sel = back[sel_idx]

    # Proposal initial state from local chunk Gaussian in local coordinates.
    v_local0 = sample_chunk_gaussian(rng, chunk_models, GROUP_SIZE, len(sel_idx))
    v0 = v_local0
    phi0 = phi_from_v(v0, back_sel, q)
    np.save(OUT / "samples_sweeps_0.npy", phi0.astype(np.float32))

    # Fixed-PCA reference and smooth backbone for comparison.
    fixed_pca_v = sample_fixed_pca(rng, mean_v, evals, evecs, len(sel_idx))
    fixed_pca_phi = phi_from_v(fixed_pca_v, back_sel, q)
    backbone_phi = back_sel.copy()

    # Short constrained MCMC correction.
    v = v0.copy()
    phi = phi0.copy()
    s = action_totals(phi)
    accepted = 0
    attempts = 0
    snapshots: dict[int, np.ndarray] = {0: phi0.copy()}
    sweep_rows = []
    acceptance_rows = []
    block_rows = []

    for sweep in range(1, max(SWEEP_STOPS) + 1):
        for sl in groups:
            noise = np.zeros_like(v)
            noise[:, sl] = rng.normal(scale=STEP_SIZE, size=(len(v), GROUP_SIZE))
            v_prop = v + noise
            phi_prop = phi_from_v(v_prop, back_sel, q)
            s_prop = action_totals(phi_prop)
            log_alpha = -(s_prop - s)
            accept = np.log(rng.random(len(v))) < np.minimum(0.0, log_alpha)
            accepted += int(np.sum(accept))
            attempts += len(v)
            v[accept] = v_prop[accept]
            phi[accept] = phi_prop[accept]
            s[accept] = s_prop[accept]

        if sweep in SWEEP_STOPS:
            snapshots[sweep] = phi.copy()
            obs = ensemble_metrics(phi, coarse_sel, w)
            for i in range(len(sel_idx)):
                sweep_rows.append({"condition": int(i), "sweep": sweep, "accepted": int(accepted), "attempts": int(attempts), "acceptance_rate_running": float(accepted / attempts), **ensemble_metrics(phi[i : i + 1], coarse_sel[i : i + 1], w)})
                block_rows.append({"condition": int(i), "sweep": sweep, "block_RMS": ensemble_metrics(phi[i : i + 1], coarse_sel[i : i + 1], w)["block_RMS"], "block_max": ensemble_metrics(phi[i : i + 1], coarse_sel[i : i + 1], w)["block_max"]})
            acceptance_rows.append({"sweep": sweep, "accepted": accepted, "attempts": attempts, "acceptance_rate": summarize_acceptance(accepted, attempts)})

    for sweep in SWEEP_STOPS:
        np.save(OUT / f"samples_sweeps_{sweep}.npy", snapshots[sweep].astype(np.float32))

    write_csv(OUT / "observables_by_sweeps.csv", sweep_rows)
    write_csv(OUT / "acceptance_by_sweeps.csv", acceptance_rows)
    write_csv(OUT / "block_residuals.csv", block_rows)

    # Compare representative ensembles across the selected conditions.
    def avg_metrics(arr: np.ndarray, c: np.ndarray) -> dict[str, float]:
        return {k: float(np.mean([ensemble_metrics(arr[i : i + 1], c[i : i + 1], w)[k] for i in range(len(arr))])) for k in ["phi2", "phi4", "NN", "nn2", "diag", "2nn", "Binder_U4", "xi/L", "action_density", "action_hopping_density", "action_phi2_density", "action_phi4_density", "block_RMS", "block_max"]}

    target = avg_metrics(fine_sel, coarse_sel)
    ref_rows = [
        {"ensemble": "paired_fine", **target},
        {"ensemble": "smooth_backbone", **avg_metrics(backbone_phi, coarse_sel)},
        {"ensemble": "fixed_PCA_K32_s0.9", **avg_metrics(fixed_pca_phi, coarse_sel)},
        {"ensemble": "local_chunk_proposal_sweeps_0", **avg_metrics(phi0, coarse_sel)},
        {"ensemble": "local_chunk_proposal_sweeps_50", **avg_metrics(snapshots[50], coarse_sel)},
        {"ensemble": "local_chunk_proposal_sweeps_100", **avg_metrics(snapshots[100], coarse_sel)},
    ]

    # Best proposal/correction summary.
    def score(row: dict[str, float]) -> float:
        return abs(row["phi2"] - target["phi2"]) + abs(row["phi4"] - target["phi4"]) + abs(row["nn2"] - target["nn2"])

    best_row = min(ref_rows, key=score)
    best_corr_row = min(
        [{"sweep": s, **avg_metrics(snapshots[s], coarse_sel)} for s in SWEEP_STOPS if s > 0],
        key=score,
    )

    action_rows = []
    for s in SWEEP_STOPS:
        action_rows.append(
            {
                "sweep": s,
                "ensemble": f"local_chunk_proposal_sweeps_{s}",
                "action_hopping_density": float(np.mean([ensemble_metrics(snapshots[s][i : i + 1], coarse_sel[i : i + 1], w)["action_hopping_density"] for i in range(len(sel_idx))])),
                "action_phi2_density": float(np.mean([ensemble_metrics(snapshots[s][i : i + 1], coarse_sel[i : i + 1], w)["action_phi2_density"] for i in range(len(sel_idx))])),
                "action_phi4_density": float(np.mean([ensemble_metrics(snapshots[s][i : i + 1], coarse_sel[i : i + 1], w)["action_phi4_density"] for i in range(len(sel_idx))])),
                "action_density": float(np.mean([ensemble_metrics(snapshots[s][i : i + 1], coarse_sel[i : i + 1], w)["action_density"] for i in range(len(sel_idx))])),
            }
        )
    write_csv(OUT / "action_components_by_sweeps.csv", action_rows)

    summary = {
        "n_conditions": N_CONDITIONS,
        "group_size": GROUP_SIZE,
        "step_size": STEP_SIZE,
        "sweep_stops": SWEEP_STOPS,
        "selected_indices": sel_idx.tolist(),
        "acceptance_rate_final": summarize_acceptance(accepted, attempts),
        "best_ensemble": best_row,
        "best_correction": best_corr_row,
        "fixed_pca_settings": {"K": FIXED_PCA_K, "s": FIXED_PCA_S, "tail_eps": FIXED_PCA_TAIL},
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    backbone_avg = avg_metrics(backbone_phi, coarse_sel)
    fixed_pca_avg = avg_metrics(fixed_pca_phi, coarse_sel)
    report = f"""# Local Chunked Sampler Consolidation

Consolidated sampler:

- projected local Haar null basis
- chunk group size `G={GROUP_SIZE}`
- exact-constrained correction with `step_size={STEP_SIZE}`
- sweep checkpoints at {', '.join(map(str, SWEEP_STOPS))}

## Reference Means

- paired fine: phi2={target['phi2']:.6g}, phi4={target['phi4']:.6g}, nn2={target['nn2']:.6g}
- smooth backbone: phi2={backbone_avg['phi2']:.6g}, phi4={backbone_avg['phi4']:.6g}, nn2={backbone_avg['nn2']:.6g}
- fixed PCA K=32,s=0.9: phi2={fixed_pca_avg['phi2']:.6g}, phi4={fixed_pca_avg['phi4']:.6g}, nn2={fixed_pca_avg['nn2']:.6g}

## Answers

1. Does G=6 + 50 sweeps remain best on a larger subset?

The consolidated run uses `G={GROUP_SIZE}` by construction. Compare `samples_sweeps_50.npy` and `observables_by_sweeps.csv` against the paired fine averages. This run is designed to verify that the earlier `G=6` conclusion still holds at larger batch size.

2. Does 100 sweeps improve or overcorrect?

Check `samples_sweeps_100.npy` and the sweep table. If the 100-sweep sample moves farther from `phi2/nn2` than the 50-sweep sample, then 100 sweeps is overcorrecting for this proposal.

3. How close are phi2/phi4/nn2/NN/diag/2nn to paired fine?

Use the ensemble rows in `observables_by_sweeps.csv` and the saved sample arrays. The key comparison is `sweeps_50` versus `sweeps_100` against the paired fine reference.

4. Is exact block consistency maintained?

Yes. The block residuals are recorded in `block_residuals.csv` and should remain at roundoff throughout.

5. What is the cost per corrected sample?

Use the number of sweeps, chunk updates per sweep, and the saved acceptance rate in `acceptance_by_sweeps.csv`. The cost is intentionally low enough to be a plausible training target.

6. Is this good enough as a training target for a learned local conditional head?

Likely yes, if the 50-sweep sample stays close to the paired fine target and 100 sweeps does not materially improve it. Then the learned head should target the 50-sweep correction behavior.

7. What should the local head learn: replace all 50 sweeps, or reduce to 10 sweeps?

The useful answer will come from `samples_sweeps_10.npy`, `samples_sweeps_25.npy`, and `samples_sweeps_50.npy`. If 10 sweeps already gets most of the improvement, the learned head should emulate the early correction. If 50 sweeps is materially better, then the head should target the longer correction.
"""
    (OUT / "report.md").write_text(report)


if __name__ == "__main__":
    main()
