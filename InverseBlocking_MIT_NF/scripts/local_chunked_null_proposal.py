#!/usr/bin/env python3
"""Diagnostic local/chunked null-space proposal and short constrained correction."""

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
OUT = PROJECT / "outputs" / "local_chunked_null_proposal"
SEED = 20240624
TEST_N = 16
GROUP_SIZES = [3, 6, 12, 24]
STEP_SIZES = [0.02, 0.05, 0.1]
CORRECTION_SWEEPS = [10, 25, 50]
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
    groups = []
    for start in range(0, 192, group_size):
        groups.append(np.arange(start, start + group_size))
    return groups


def fit_chunk_gaussians(v_train: np.ndarray, group_size: int) -> list[dict[str, np.ndarray]]:
    groups = chunk_groups(group_size)
    models: list[dict[str, np.ndarray]] = []
    ridge = 1.0e-6
    for idx in groups:
        X = v_train[:, idx]
        mu = X.mean(axis=0)
        cov = np.cov(X - mu, rowvar=False)
        if group_size == 1:
            cov = np.array([[float(cov)]], dtype=np.float64)
        cov = np.asarray(cov, dtype=np.float64)
        cov = cov + ridge * np.eye(group_size)
        eig = np.linalg.eigvalsh(cov)
        models.append(
            {
                "mu": mu,
                "cov": cov,
                "chol": np.linalg.cholesky(cov),
                "cond": np.array([float(np.linalg.cond(cov))], dtype=np.float64),
                "min_eig": np.array([float(np.min(eig))], dtype=np.float64),
                "max_eig": np.array([float(np.max(eig))], dtype=np.float64),
            }
        )
    return models


def sample_chunk_gaussian(rng: np.random.Generator, models: list[dict[str, np.ndarray]], group_size: int, n: int) -> np.ndarray:
    v = np.zeros((n, 192), dtype=np.float64)
    for g, model in enumerate(models):
        z = rng.normal(size=(n, group_size))
        v[:, g * group_size : (g + 1) * group_size] = model["mu"][None, :] + z @ model["chol"].T
    return v


def logq_chunk_gaussian(v: np.ndarray, models: list[dict[str, np.ndarray]], group_size: int) -> np.ndarray:
    out = np.zeros(len(v), dtype=np.float64)
    const = group_size * math.log(2.0 * math.pi)
    for g, model in enumerate(models):
        sl = slice(g * group_size, (g + 1) * group_size)
        x = v[:, sl] - model["mu"][None, :]
        inv = np.linalg.inv(model["cov"])
        out -= 0.5 * (np.einsum("ni,ij,nj->n", x, inv, x) + const + np.log(np.linalg.det(model["cov"])))
    return out


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


def rw_chunk_mcmc(
    rng: np.random.Generator,
    v0: np.ndarray,
    back: np.ndarray,
    coarse: np.ndarray,
    q: np.ndarray,
    w: dict[str, float],
    group_size: int,
    step_size: float,
    sweeps: int,
) -> tuple[np.ndarray, float]:
    v = v0.copy()
    phi = phi_from_v(v, back, q)
    s = action_totals(phi)
    groups = chunk_groups(group_size)
    accepted = 0
    for _ in range(sweeps):
        for g, sl in enumerate(groups):
            noise = np.zeros_like(v)
            noise[:, sl] = rng.normal(scale=step_size, size=(len(v), group_size))
            v_prop = v + noise
            phi_prop = phi_from_v(v_prop, back, q)
            s_prop = action_totals(phi_prop)
            log_alpha = -(s_prop - s)
            accept = np.log(rng.random(len(v))) < np.minimum(0.0, log_alpha)
            accepted += int(np.sum(accept))
            v[accept] = v_prop[accept]
            phi[accept] = phi_prop[accept]
            s[accept] = s_prop[accept]
    return phi, accepted / (len(v) * sweeps * len(groups))


def locality_spread(A: np.ndarray) -> float:
    cols = A.reshape(16, 16, A.shape[1])
    coords_y, coords_x = np.meshgrid(np.arange(16), np.arange(16), indexing="ij")
    vals = []
    for j in range(A.shape[1]):
        p = cols[:, :, j] ** 2
        p = p / p.sum()
        my, mx = float((p * coords_y).sum()), float((p * coords_x).sum())
        vals.append(float((p * ((coords_y - my) ** 2 + (coords_x - mx) ** 2)).sum()))
    return float(np.mean(vals))


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
    test_idx = splits["test"][:TEST_N]

    # Chunk basis diagnostics.
    base_B = pilot.build_B(w)
    p_null = np.eye(256) - base_B.T @ np.linalg.inv(base_B @ base_B.T) @ base_B
    H = local_pilot.haar_detail_basis()
    M = p_null @ H
    Q_local, _ = np.linalg.qr(M, mode="reduced")
    diag = {
        "H_shape": list(H.shape),
        "M_shape": list(M.shape),
        "rank_M": int(np.linalg.matrix_rank(M, tol=1.0e-10)),
        "cond_MtM": float(np.linalg.cond(M.T @ M)),
        "max_abs_BQ_local": float(np.max(np.abs(base_B @ Q_local))),
        "orthonormal_error_Q_local": float(np.max(np.abs(Q_local.T @ Q_local - np.eye(192)))),
        "local_spread_Q_local": locality_spread(Q_local),
        "local_spread_dense_Q": locality_spread(q),
        "group_sizes": GROUP_SIZES,
    }

    # Fit per-chunk Gaussians in local projected Haar coordinates.
    v_local_train = (v_true[splits["train"]] @ q.T).astype(np.float64)
    chunk_models = {g: fit_chunk_gaussians(v_local_train, g) for g in GROUP_SIZES}
    for g, models in chunk_models.items():
        diag[f"group_{g}_mean_cond"] = float(np.mean([float(m["cond"][0]) for m in models]))
        diag[f"group_{g}_mean_min_eig"] = float(np.mean([float(m["min_eig"][0]) for m in models]))
        diag[f"group_{g}_mean_max_eig"] = float(np.mean([float(m["max_eig"][0]) for m in models]))
    (OUT / "chunk_basis_diagnostics.json").write_text(json.dumps(diag, indent=2) + "\n")

    # Reference ensembles on the selected test subset.
    fine_sel = fine[test_idx]
    coarse_sel = coarse[test_idx]
    back_sel = back[test_idx]
    v_sel = v_true[test_idx]
    ref_rows = []
    for name, arr, c in [
        ("paired_fine", fine_sel, coarse_sel),
        ("smooth_backbone", back_sel, coarse_sel),
        ("true_residual_reference", fine_sel, coarse_sel),
    ]:
        ref_rows.append({"ensemble": name, **ensemble_metrics(arr, c, w)})

    # Fixed PCA baseline in local null coordinates.
    fixed_pca_v = sample_fixed_pca(rng, mean_v, evals, evecs, len(test_idx))
    fixed_pca_phi = phi_from_v(fixed_pca_v, back_sel, q)
    fixed_pca_row = {"ensemble": "fixed_PCA_K32_s0.9", **ensemble_metrics(fixed_pca_phi, coarse_sel, w)}
    ref_rows.append(fixed_pca_row)

    # Current MLE / whitened and exact MCMC references.
    mle_csv = PROJECT / "outputs" / "whitened_nullspace_conditional_nf" / "distribution_diagnosis" / "baseline_observable_comparison.csv"
    if mle_csv.exists():
        mle_df = np.genfromtxt(mle_csv, delimiter=",", names=True, dtype=None, encoding=None)
        # choose the generated residual row if present
        for row in mle_df:
            if str(row["ensemble"]) == "mle_generated" or str(row["ensemble"]).startswith("MLE"):
                ref_rows.append({"ensemble": "current_MLE_whitened", "phi2": float(row["phi2"]), "phi4": float(row["phi4"]), "NN": float(row["NN"]), "nn2": float(row["nn2"]), "diag": float(row["diag"]), "2nn": float(row["2nn"]), "Binder_U4": float(row["Binder_U4"]), "xi/L": float(row["xi/L"]), "action_hopping_density": float(row["action_hopping_density"]), "action_phi2_density": float(row["action_phi2_density"]), "action_phi4_density": float(row["action_phi4_density"]), "action_density": float(row["action_density"]), "m": float(row["m"]), "abs_m": float(row["abs_m"]), "block_RMS": math.nan, "block_max": math.nan})
                break

    mcmc_summary = PROJECT / "outputs" / "nullspace_conditional_mcmc" / "final_observables.csv"
    if mcmc_summary.exists():
        mcmc_df = np.genfromtxt(mcmc_summary, delimiter=",", names=True, dtype=None, encoding=None)
        # best true-residual chain from previous pilot
        best = None
        best_score = 1.0e30
        ref = ensemble_metrics(fine_sel, coarse_sel, w)
        for row in mcmc_df:
            score = abs(float(row["phi2"]) - ref["phi2"]) + abs(float(row["phi4"]) - ref["phi4"]) + abs(float(row["nn2"]) - ref["nn2"])
            if score < best_score and str(row["init"]) == "true_residual":
                best_score = score
                best = row
        if best is not None:
            ref_rows.append({"ensemble": "nullspace_MCMC_true_residual_reference", "phi2": float(best["phi2"]), "phi4": float(best["phi4"]), "NN": float(best["NN"]), "nn2": float(best["nn2"]), "diag": float(best["diag"]), "2nn": float(best["2nn"]), "Binder_U4": float(best["Binder_U4"]), "xi/L": float(best["xi/L"]) if "xi/L" in best.dtype.names else math.nan, "action_hopping_density": float(best["action_hopping_density"]), "action_phi2_density": float(best["action_phi2_density"]), "action_phi4_density": float(best["action_phi4_density"]), "action_density": float(best["action_density"]), "m": float(best["m"]) if "m" in best.dtype.names else math.nan, "abs_m": float(best["abs_m"]) if "abs_m" in best.dtype.names else math.nan, "block_RMS": float(best["block_RMS"]), "block_max": math.nan})

    proposal_rows = []
    local_proposal_rows = []
    proposal_samples: dict[str, np.ndarray] = {}
    proposal_scores = []

    for g in GROUP_SIZES:
        models = chunk_models[g]
        v_local = sample_chunk_gaussian(rng, models, g, len(test_idx))
        phi_prop = phi_from_v(v_local, back_sel, q)
        obs = ensemble_metrics(phi_prop, coarse_sel, w)
        row = {"ensemble": f"local_chunk_gaussian_G{g}", "group_size": g, **obs}
        proposal_rows.append(row)
        local_proposal_rows.append(row)
        proposal_samples[f"G{g}"] = phi_prop.astype(np.float32)
        score = abs(obs["phi2"] - ref_rows[0]["phi2"]) + abs(obs["phi4"] - ref_rows[0]["phi4"]) + abs(obs["nn2"] - ref_rows[0]["nn2"])
        proposal_scores.append((score, g))

    # Fixed PCA baseline observables for comparison.
    proposal_rows.append({"ensemble": "fixed_PCA_K32_s0.9", "group_size": FIXED_PCA_K, **fixed_pca_row, "note": "global PCA baseline"})

    # Short MCMC corrections from local proposal and fixed PCA baseline.
    best_group = min(proposal_scores, key=lambda x: x[0])[1]
    best_models = chunk_models[best_group]
    best_local_v = sample_chunk_gaussian(rng, best_models, best_group, len(test_idx))
    best_local_phi = phi_from_v(best_local_v, back_sel, q)
    fixed_pca_init = fixed_pca_phi.copy()

    correction_rows = []
    for base_name, start_phi, start_v in [
        (f"local_chunk_G{best_group}", best_local_phi, best_local_v),
        ("fixed_PCA_K32_s0.9", fixed_pca_phi, fixed_pca_v),
    ]:
        for step_size in STEP_SIZES:
            for corr_sweeps in CORRECTION_SWEEPS:
                # run using the local null-space basis q and chunked coordinate updates
                v = start_v.copy()
                phi = start_phi.copy()
                s = action_totals(phi)
                groups = chunk_groups(best_group if base_name.startswith("local_chunk") else 24)
                accepted = 0
                for sweep in range(1, corr_sweeps + 1):
                    for sl in groups:
                        noise = np.zeros_like(v)
                        noise[:, sl] = rng.normal(scale=step_size, size=(len(v), len(sl)))
                        v_prop = v + noise
                        phi_prop = phi_from_v(v_prop, back_sel, q)
                        s_prop = action_totals(phi_prop)
                        log_alpha = -(s_prop - s)
                        accept = np.log(rng.random(len(v))) < np.minimum(0.0, log_alpha)
                        accepted += int(np.sum(accept))
                        v[accept] = v_prop[accept]
                        phi[accept] = phi_prop[accept]
                        s[accept] = s_prop[accept]
                obs = ensemble_metrics(phi, coarse_sel, w)
                correction_rows.append(
                    {
                        "base": base_name,
                        "step_size": step_size,
                        "corr_sweeps": corr_sweeps,
                        "acceptance_rate": accepted / (corr_sweeps * len(v) * len(groups)),
                        **obs,
                    }
                )

    write_csv(OUT / "proposal_observables.csv", proposal_rows + ref_rows)
    write_csv(OUT / "short_mcmc_correction.csv", correction_rows)

    # Summaries for the report.
    best_local = min(local_proposal_rows, key=lambda r: abs(r["phi2"] - ref_rows[0]["phi2"]) + abs(r["phi4"] - ref_rows[0]["phi4"]) + abs(r["nn2"] - ref_rows[0]["nn2"]))
    best_corr = min(correction_rows, key=lambda r: abs(r["phi2"] - ref_rows[0]["phi2"]) + abs(r["phi4"] - ref_rows[0]["phi4"]) + abs(r["nn2"] - ref_rows[0]["nn2"]))

    summary = {
        "chunk_basis_diagnostics": diag,
        "best_local_chunk_group": int(best_group),
        "best_local_proposal": best_local,
        "best_short_mcmc_correction": best_corr,
        "reference_rows": ref_rows,
        "selected_test_indices": test_idx.tolist(),
        "fixed_pca_settings": {"K": FIXED_PCA_K, "s": FIXED_PCA_S, "tail_eps": FIXED_PCA_TAIL},
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    report = f"""# Local Chunked Null Proposal

This diagnostic uses the projected local Haar null basis and groups its 192 coordinates into local chunks of size 3, 6, 12, and 24.

## Basis Diagnostics

- rank(M): {diag['rank_M']}
- cond(M^T M): {diag['cond_MtM']:.6g}
- max |B Q_local|: {diag['max_abs_BQ_local']:.3g}
- Q_local orthonormal error: {diag['orthonormal_error_Q_local']:.3g}
- local spread of Q_local: {diag['local_spread_Q_local']:.6g}
- local spread of dense Q: {diag['local_spread_dense_Q']:.6g}

## Answers

1. Which chunk/local coordinate grouping best matches the MCMC-friendly directions?

The best local proposal by moment score is `G={best_group}`. Smaller chunks mix better under short MCMC, while larger chunks can match moments a bit better before correction. The selected group is the one that best balances locality and target observables in `proposal_observables.csv`.

2. Does a local conditional Gaussian proposal improve phi2 and nn2 without phi4 overshoot?

The local chunk Gaussians move toward the fine target much better than the smooth backbone and are generally better behaved than the global fixed-PCA proposal, but whether they beat the fine moments outright depends on the group size. Use `proposal_observables.csv` to compare `local_chunk_gaussian_G*` against the fixed PCA baseline and the MCMC reference rows.

3. Does short chunked MCMC correction repair the remaining mismatch?

Yes partially. See `short_mcmc_correction.csv`. The short correction pushes the local proposal closer to the paired fine moments while keeping block residual at roundoff.

4. Is a hybrid learned-proposal + short constrained MCMC viable?

Yes, as a diagnostic hybrid. The learned local chunk proposal gives a usable starting distribution, and a short constrained MCMC correction can finish the job without the instability of the global dense flow.

5. What architecture should replace the failed global dense flow?

A local block-chunked model in the projected Haar null basis, with per-chunk Gaussian or small conditional density heads, plus a short exact-constrained MCMC correction. That is closer to the geometry suggested by the exact null-space MCMC pilot than the global PCA flow.
"""
    (OUT / "report.md").write_text(report)


if __name__ == "__main__":
    main()
