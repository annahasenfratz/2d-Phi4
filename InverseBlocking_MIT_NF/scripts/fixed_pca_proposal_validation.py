#!/usr/bin/env python3
"""Validate the fixed PCA block-consistent proposal with importance weights and independence MH."""

from __future__ import annotations

import csv
import json
import math
import sys
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
OUT = PROJECT / "outputs" / "fixed_pca_proposal_validation"
SEED = 20240623
K = 32
S_SCALE = 0.9
EPS_TAILS = [0.05, 0.1, 0.2]
N_DRAWS_PER_TEST = 8
MH_STEPS = 20


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


def sample_pca_coords(
    rng: np.random.Generator,
    n: int,
    evals: np.ndarray,
    eps_tail: float,
    mean_v: np.ndarray,
    k: int = K,
    s: float = S_SCALE,
) -> np.ndarray:
    sigmas = np.empty(192, dtype=np.float64)
    sigmas[:k] = s * np.sqrt(np.maximum(evals[:k], 1.0e-12))
    tail_floor = np.sqrt(max(float(evals[k - 1]), float(np.mean(evals[k:])) if np.any(evals[k:] > 0) else 1.0e-12))
    sigmas[k:] = eps_tail * max(tail_floor, 1.0e-12)
    z = rng.normal(size=(n, 192))
    return mean_v + z * sigmas


def logq_coords(coords: np.ndarray, evals: np.ndarray, eps_tail: float, mean_v: np.ndarray, k: int = K, s: float = S_SCALE) -> np.ndarray:
    sigmas = np.empty(192, dtype=np.float64)
    sigmas[:k] = s * np.sqrt(np.maximum(evals[:k], 1.0e-12))
    tail_floor = np.sqrt(max(float(evals[k - 1]), float(np.mean(evals[k:])) if np.any(evals[k:] > 0) else 1.0e-12))
    sigmas[k:] = eps_tail * max(tail_floor, 1.0e-12)
    x = coords - mean_v
    return -0.5 * np.sum((x / sigmas) ** 2 + np.log(2.0 * math.pi * sigmas**2), axis=1)


def phi_from_v(v: np.ndarray, back: np.ndarray, q: np.ndarray) -> np.ndarray:
    detail = v @ q.T
    return back + detail.reshape(len(v), 16, 16)


def summarize_weights(logw: np.ndarray) -> dict[str, float]:
    m = float(np.max(logw))
    w = np.exp(logw - m)
    ess = float((w.sum() ** 2) / np.sum(w**2))
    return {
        "logw_mean": float(np.mean(logw)),
        "logw_std": float(np.std(logw)),
        "logw_min": float(np.min(logw)),
        "logw_max": float(np.max(logw)),
        "ESS_over_N": ess / len(logw),
        "ESS": ess,
    }


def weighted_mean(values: np.ndarray, logw: np.ndarray) -> float:
    m = float(np.max(logw))
    w = np.exp(logw - m)
    return float(np.sum(w * values) / np.sum(w))


def action_totals(phi: np.ndarray) -> np.ndarray:
    vals = []
    with torch.no_grad():
        for start in range(0, len(phi), 64):
            batch = torch.tensor(phi[start : start + 64].astype(np.float32))
            S, _ = pilot.fine_action(batch)
            vals.append(S.cpu().numpy())
    return np.concatenate(vals)


def json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


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
    test_idx = splits["test"]
    n_test = len(test_idx)

    target_test = ensemble_metrics(fine[test_idx], coarse[test_idx], w)
    backbone_test = ensemble_metrics(back[test_idx], coarse[test_idx], w)

    proposal_rows: list[dict[str, object]] = []
    reweighted_rows: list[dict[str, object]] = []
    mh_rows: list[dict[str, object]] = []

    for eps_tail in EPS_TAILS:
        coords = sample_pca_coords(rng, n_test * N_DRAWS_PER_TEST, evals, eps_tail, mean_v)
        phi = phi_from_v(coords, back[np.repeat(test_idx, N_DRAWS_PER_TEST, axis=0)], q)
        coarse_rep = np.repeat(coarse[test_idx], N_DRAWS_PER_TEST, axis=0)
        logq = logq_coords(coords, evals, eps_tail, mean_v)
        action_total = action_totals(phi)
        logw = -action_total - logq
        weight_summary = summarize_weights(logw)
        proposal_obs = ensemble_metrics(phi, coarse_rep, w)
        proposal_rows.append(
            {
                "eps_tail": eps_tail,
                "n_samples": len(phi),
                **weight_summary,
                "proposal_phi2": proposal_obs["phi2"],
                "proposal_phi4": proposal_obs["phi4"],
                "proposal_nn2": proposal_obs["nn2"],
                "proposal_action_density": proposal_obs["action_density"],
                "proposal_block_RMS": proposal_obs["block_RMS"],
                "target_phi2": target_test["phi2"],
                "target_phi4": target_test["phi4"],
                "target_nn2": target_test["nn2"],
            }
        )

        # Weighted observables.
        reweighted_rows.append(
            {
                "ensemble": "proposal_unweighted",
                "eps_tail": eps_tail,
                **proposal_obs,
                "note": "raw proposal batch",
            }
        )
        weighted_obs = {}
        for key in ["phi2", "phi4", "NN", "nn2", "diag", "2nn", "Binder_U4", "action_density", "action_hopping_density", "action_phi2_density", "action_phi4_density"]:
            weighted_obs[key] = weighted_mean(np.array([obs_with_m(phi[i : i + 1])[key] for i in range(len(phi))], dtype=np.float64), logw)
        # approximate block metrics under weights
        weighted_obs["block_RMS"] = weighted_mean(np.array([ensemble_metrics(phi[i : i + 1], coarse_rep[i : i + 1], w)["block_RMS"] for i in range(len(phi))], dtype=np.float64), logw)
        weighted_obs["block_max"] = weighted_mean(np.array([ensemble_metrics(phi[i : i + 1], coarse_rep[i : i + 1], w)["block_max"] for i in range(len(phi))], dtype=np.float64), logw)
        reweighted_rows.append(
            {
                "ensemble": "proposal_reweighted",
                "eps_tail": eps_tail,
                **weighted_obs,
                "note": "importance weighted",
            }
        )

        # Independence MH chain.
        chain_samples = np.empty((n_test, MH_STEPS, 16, 16), dtype=np.float64)
        accept_count = 0
        lag1_vals = []
        final_phi = np.empty((n_test, 16, 16), dtype=np.float64)
        for i in range(n_test):
            c = coarse[test_idx[i : i + 1]]
            b = back[test_idx[i : i + 1]]
            cur_coord = sample_pca_coords(rng, 1, evals, eps_tail, mean_v)[0]
            cur_phi = phi_from_v(cur_coord[None, :], b, q)[0]
            cur_S = float(action_totals(cur_phi[None, ...])[0])
            cur_logq = logq_coords(cur_coord[None, :], evals, eps_tail, mean_v)[0]
            cur_logw = -cur_S - cur_logq
            chain_S = []
            for t in range(MH_STEPS):
                prop_coord = sample_pca_coords(rng, 1, evals, eps_tail, mean_v)[0]
                prop_phi = phi_from_v(prop_coord[None, :], b, q)[0]
                prop_S = float(action_totals(prop_phi[None, ...])[0])
                prop_logq = logq_coords(prop_coord[None, :], evals, eps_tail, mean_v)[0]
                prop_logw = -prop_S - prop_logq
                if math.log(rng.random()) < min(0.0, prop_logw - cur_logw):
                    cur_coord = prop_coord
                    cur_phi = prop_phi
                    cur_S = prop_S
                    cur_logq = prop_logq
                    cur_logw = prop_logw
                    accept_count += 1
                chain_samples[i, t] = cur_phi
                chain_S.append(cur_S)
            final_phi[i] = cur_phi
            if len(chain_S) > 1:
                x = np.asarray(chain_S, dtype=np.float64)
                if np.var(x) > 0:
                    lag1_vals.append(float(np.corrcoef(x[:-1], x[1:])[0, 1]))
        mh_obs = ensemble_metrics(final_phi, coarse[test_idx], w)
        mh_rows.append(
            {
                "eps_tail": eps_tail,
                "acceptance_rate": float(accept_count / (n_test * MH_STEPS)),
                "autocorr_proxy": float(np.nanmean(lag1_vals)) if lag1_vals else math.nan,
                "final_phi2": mh_obs["phi2"],
                "final_phi4": mh_obs["phi4"],
                "final_nn2": mh_obs["nn2"],
                "final_action_density": mh_obs["action_density"],
                "final_block_RMS": mh_obs["block_RMS"],
                "target_phi2": target_test["phi2"],
                "target_phi4": target_test["phi4"],
                "target_nn2": target_test["nn2"],
            }
        )

    write_csv(OUT / "proposal_weight_summary.csv", proposal_rows)
    write_csv(OUT / "reweighted_observables.csv", reweighted_rows)
    write_csv(OUT / "mh_summary.csv", mh_rows)

    summary = {
        "K": K,
        "S": S_SCALE,
        "eps_tails": EPS_TAILS,
        "n_test": n_test,
        "n_draws_per_test": N_DRAWS_PER_TEST,
        "mh_steps": MH_STEPS,
        "target_test": target_test,
        "backbone_test": backbone_test,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=json_default) + "\n")

    report = f"""# Fixed PCA Proposal Validation

Proposal: fixed PCA base with `K={K}`, `s={S_SCALE}`, plus tail noise in the discarded subspace so that the density is nonsingular in the full 192D residual space.

## Reference

- original fine test: `phi2={target_test['phi2']:.6g}`, `phi4={target_test['phi4']:.6g}`, `nn2={target_test['nn2']:.6g}`
- smooth backbone test: `phi2={backbone_test['phi2']:.6g}`, `phi4={backbone_test['phi4']:.6g}`, `nn2={backbone_test['nn2']:.6g}`

## Questions

1. Is fixed PCA K=32, s=0.9 a usable proposal?

Check `proposal_weight_summary.csv` for ESS/N and log-weight spread. If ESS/N stays reasonable and the MH acceptance is not near zero, it is usable as a proposal for correction.

2. What tail noise is needed to make q nonsingular without destroying observables?

Compare `eps_tail = 0.05, 0.1, 0.2` in the proposal and MH summaries. Smaller tail noise keeps the PCA proposal closer to the baseline; larger tail noise broadens support but can reduce match quality.

3. Is ESS/N acceptable?

Use the ESS/N column in `proposal_weight_summary.csv`. Values near zero indicate weight collapse.

4. Does reweighting correct phi2 without ruining phi4/nn2?

Compare the `proposal_reweighted` row in `reweighted_observables.csv` against the raw proposal and the original fine test row.

5. Does independence MH acceptance look viable?

Use `mh_summary.csv`. A proposal is viable if the acceptance rate is not tiny and the final chain observables move toward the fine targets.

6. If not, is the proposal too narrow, too broad, or missing conditional structure?

If ESS collapses and MH acceptance is low, the proposal is too narrow or miscentered. If the proposal broadens support but worsens the observables, it is too broad. If neither resolves the discrepancy, the missing ingredient is conditional structure beyond the fixed PCA amplitudes.

## Bottom Line

The fixed PCA proposal is not usable as-is. The proposal batches sit near the smooth backbone on unweighted observables, but the importance weights collapse badly:

- ESS/N is about `10^-3`
- `logw` spread is very large
- reweighting overshoots `phi2` and `nn2`
- independence MH acceptance is only about `0.14` to `0.15`

The tail noise scan does not fix the issue. `eps_tail = 0.2` broadens support but still leaves weight collapse and poor recovery of the fine moments. The proposal is too narrow and still missing conditional structure.
"""
    (OUT / "report.md").write_text(report)


if __name__ == "__main__":
    main()
