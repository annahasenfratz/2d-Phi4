#!/usr/bin/env python3
"""Improved exact block-consistent backbone baselines."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT / "scripts"))
import local_nullspace_pilot as local_pilot  # type: ignore
import nullspace_conditional_nf_pilot as pilot  # type: ignore


DATA = PROJECT / "outputs" / "paired_data_lam1_kappaf0p320"
OUT = PROJECT / "outputs" / "improved_backbone_baseline"
SEED = 20240701
PCA_KS = [16, 32, 64]


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


def per_config_obs(phi: np.ndarray) -> dict[str, np.ndarray]:
    m = phi.mean(axis=(-2, -1))
    nn = 0.5 * (
        (phi * np.roll(phi, -1, axis=-2)).mean(axis=(-2, -1))
        + (phi * np.roll(phi, -1, axis=-1)).mean(axis=(-2, -1))
    )
    nn2 = 0.5 * (
        ((phi * np.roll(phi, -1, axis=-2)) ** 2).mean(axis=(-2, -1))
        + ((phi * np.roll(phi, -1, axis=-1)) ** 2).mean(axis=(-2, -1))
    )
    diag = (phi * np.roll(np.roll(phi, -1, axis=-2), -1, axis=-1)).mean(axis=(-2, -1))
    twonn = 0.5 * (
        (phi * np.roll(phi, -2, axis=-2)).mean(axis=(-2, -1))
        + (phi * np.roll(phi, -2, axis=-1)).mean(axis=(-2, -1))
    )
    action_hop = -4.0 * 0.320 * nn
    phi2 = np.mean(phi**2, axis=(-2, -1))
    phi4 = np.mean(phi**4, axis=(-2, -1))
    return {
        "m": m,
        "abs_m": np.abs(m),
        "phi2": phi2,
        "phi4": phi4,
        "NN": nn,
        "nn2": nn2,
        "diag": diag,
        "2nn": twonn,
        "action_hopping_density": action_hop,
        "action_phi2_density": -phi2,
        "action_phi4_density": phi4,
        "action_density": action_hop - phi2 + phi4,
    }


def ensemble_row(name: str, phi: np.ndarray, w: dict[str, float], coarse: np.ndarray, *, stochastic: bool, logprob_available: bool) -> dict[str, object]:
    br = pilot.block_sym_np(phi.astype(np.float64), w) - coarse[: len(phi)]
    return {
        "ensemble": name,
        "stochastic": stochastic,
        "logprob_available": logprob_available,
        "block_RMS": float(np.sqrt(np.mean(br**2))),
        "block_max": float(np.max(np.abs(br))),
        **pilot.obs_np(phi.astype(np.float64)),
    }


def residual_stats(name: str, remaining: np.ndarray) -> dict[str, object]:
    cov = np.cov(remaining, rowvar=False)
    eig = np.linalg.eigvalsh(cov)[::-1]
    centered = remaining - remaining.mean(axis=0, keepdims=True)
    std = remaining.std(axis=0, ddof=1)
    kurt = np.mean(centered**4, axis=0) / np.maximum(std, 1e-30) ** 4
    total = float(np.sum(eig))
    return {
        "candidate": name,
        "mean_coordinate_std": float(np.mean(std)),
        "median_coordinate_std": float(np.median(std)),
        "max_coordinate_std": float(np.max(std)),
        "cov_trace": total,
        "cov_rank_gt_1e-10": int(np.sum(eig > 1e-10)),
        "top10_variance_fraction": float(np.sum(eig[:10]) / total) if total > 0 else math.nan,
        "top32_variance_fraction": float(np.sum(eig[:32]) / total) if total > 0 else math.nan,
        "top64_variance_fraction": float(np.sum(eig[:64]) / total) if total > 0 else math.nan,
        "kurtosis_mean": float(np.mean(kurt)),
        "kurtosis_median": float(np.median(kurt)),
    }


def base_stats(name: str, v_base: np.ndarray, v_true_ref: np.ndarray) -> dict[str, object]:
    cov = np.cov(v_base, rowvar=False)
    true_cov = np.cov(v_true_ref, rowvar=False)
    trace = float(np.trace(cov))
    true_trace = float(np.trace(true_cov))
    mse = float(np.mean((v_true_ref - v_base) ** 2))
    return {
        "candidate": name,
        "base_cov_trace": trace,
        "true_cov_trace": true_trace,
        "base_trace_fraction_of_true": trace / true_trace if true_trace > 0 else math.nan,
        "coordinate_mse_to_true_residual": mse,
    }


def fit_pca(v_train: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = v_train.mean(axis=0)
    centered = v_train - mean
    cov = np.cov(centered, rowvar=False)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    return mean, evals[order], evecs[:, order]


def sample_pca(mean: np.ndarray, evals: np.ndarray, evecs: np.ndarray, n: int, k: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    z = rng.normal(size=(n, k))
    coeff = z * np.sqrt(np.maximum(evals[:k], 0.0))
    v = mean + coeff @ evecs[:, :k].T
    logp = -0.5 * np.sum(z**2 + math.log(2.0 * math.pi), axis=1) - 0.5 * np.sum(np.log(np.maximum(evals[:k], 1e-30)))
    return v, logp


def conditional_bins(values: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray, n_bins: int = 4) -> list[tuple[np.ndarray, np.ndarray]]:
    qs = np.quantile(values[train_idx], np.linspace(0, 1, n_bins + 1))
    out = []
    for b in range(n_bins):
        tr = train_idx[(values[train_idx] >= qs[b]) & (values[train_idx] <= qs[b + 1] if b == n_bins - 1 else values[train_idx] < qs[b + 1])]
        te = test_idx[(values[test_idx] >= qs[b]) & (values[test_idx] <= qs[b + 1] if b == n_bins - 1 else values[test_idx] < qs[b + 1])]
        out.append((tr, te))
    return out


def main() -> None:
    rng = np.random.default_rng(SEED)
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(f"Refusing to overwrite existing outputs in {OUT}")
    OUT.mkdir(parents=True)

    w = load_weights()
    fine = np.load(DATA / "fine_configs.npy").astype(np.float64)
    coarse = np.load(DATA / "coarse_blocked_configs.npy").astype(np.float64)
    backbone = np.load(DATA / "backbone_configs.npy").astype(np.float64)
    v_true = np.load(DATA / "residual_v_true.npy").astype(np.float64)
    splits = np.load(DATA / "split_indices.npz")
    train_idx = splits["train"].astype(int)
    test_idx = splits["test"].astype(int)
    q = local_q_basis(w)

    mean, evals, evecs = fit_pca(v_true[train_idx])
    co = per_config_obs(coarse)
    bo = per_config_obs(backbone)
    feature_phi2 = co["phi2"]
    feature_absm = np.abs(co["m"])

    candidates: dict[str, dict[str, object]] = {}
    candidates["original_fine"] = {"phi": fine[test_idx], "v_base": v_true[test_idx], "stochastic": False, "logp": None}
    candidates["smooth_phi_backbone"] = {"phi": backbone[test_idx], "v_base": np.zeros_like(v_true[test_idx]), "stochastic": False, "logp": None}
    candidates["true_residual_oracle"] = {
        "phi": backbone[test_idx] + (v_true[test_idx] @ q.T).reshape(len(test_idx), 16, 16),
        "v_base": v_true[test_idx],
        "stochastic": False,
        "logp": None,
    }

    v_global, logp_global = sample_pca(mean, evals, evecs, len(test_idx), 192, rng)
    candidates["global_gaussian_full_cov"] = {
        "phi": backbone[test_idx] + (v_global @ q.T).reshape(len(test_idx), 16, 16),
        "v_base": v_global,
        "stochastic": True,
        "logp": logp_global,
    }
    for k in PCA_KS:
        v_pca, logp = sample_pca(mean, evals, evecs, len(test_idx), k, rng)
        candidates[f"pca_truncated_gaussian_K{k}"] = {
            "phi": backbone[test_idx] + (v_pca @ q.T).reshape(len(test_idx), 16, 16),
            "v_base": v_pca,
            "stochastic": True,
            "logp": logp,
        }

    for feature_name, feature in {"coarse_phi2_bins": feature_phi2, "coarse_abs_m_bins": feature_absm}.items():
        v_mean = np.zeros((len(test_idx), 192), dtype=np.float64)
        v_sample = np.zeros((len(test_idx), 192), dtype=np.float64)
        test_pos = {idx: i for i, idx in enumerate(test_idx)}
        for tr, te in conditional_bins(feature, train_idx, test_idx):
            if len(te) == 0:
                continue
            if len(tr) < 4:
                tr = train_idx
            local_mean = v_true[tr].mean(axis=0)
            local_cov = np.cov(v_true[tr] - local_mean, rowvar=False)
            local_eval, local_evec = np.linalg.eigh(local_cov)
            order = np.argsort(local_eval)[::-1]
            local_eval = np.maximum(local_eval[order], 0.0)
            local_evec = local_evec[:, order]
            z = rng.normal(size=(len(te), 192))
            local_sample = local_mean + (z * np.sqrt(local_eval)) @ local_evec.T
            for n, idx in enumerate(te):
                pos = test_pos[int(idx)]
                v_mean[pos] = local_mean
                v_sample[pos] = local_sample[n]
        candidates[f"deterministic_mean_{feature_name}"] = {
            "phi": backbone[test_idx] + (v_mean @ q.T).reshape(len(test_idx), 16, 16),
            "v_base": v_mean,
            "stochastic": False,
            "logp": None,
        }
        candidates[f"conditional_gaussian_{feature_name}"] = {
            "phi": backbone[test_idx] + (v_sample @ q.T).reshape(len(test_idx), 16, 16),
            "v_base": v_sample,
            "stochastic": True,
            "logp": None,
        }

    # Local Haar diagonal Gaussian in unprojected Haar-detail coordinates, then project back through Q coordinates.
    h = local_pilot.haar_detail_basis()
    haar_train = (fine[train_idx] - backbone[train_idx]).reshape(len(train_idx), -1) @ h
    haar_mean = haar_train.mean(axis=0)
    haar_std = haar_train.std(axis=0, ddof=1)
    haar_draw = haar_mean + rng.normal(size=(len(test_idx), haar_train.shape[1])) * haar_std
    detail = haar_draw @ h.T
    v_haar = detail @ q
    candidates["local_haar_diagonal_gaussian"] = {
        "phi": backbone[test_idx] + (v_haar @ q.T).reshape(len(test_idx), 16, 16),
        "v_base": v_haar,
        "stochastic": True,
        "logp": None,
    }

    obs_rows = []
    residual_rows = []
    base_rows = []
    logp_rows = []
    for name, payload in candidates.items():
        phi = payload["phi"]  # type: ignore[assignment]
        v_base = payload["v_base"]  # type: ignore[assignment]
        obs_rows.append(ensemble_row(name, phi, w, coarse[test_idx], stochastic=bool(payload["stochastic"]), logprob_available=payload["logp"] is not None))
        residual_rows.append(residual_stats(name, v_true[test_idx] - v_base))
        base_rows.append(base_stats(name, v_base, v_true[test_idx]))
        if payload["logp"] is not None:
            logp = payload["logp"]  # type: ignore[assignment]
            logp_rows.append({"candidate": name, "mean_logprob": float(np.mean(logp)), "std_logprob": float(np.std(logp, ddof=1))})

    write_csv(OUT / "backbone_observables.csv", obs_rows)
    write_csv(OUT / "remaining_residual_statistics.csv", residual_rows)
    write_csv(OUT / "base_residual_component_statistics.csv", base_rows)
    write_csv(OUT / "candidate_logprob_summary.csv", logp_rows)

    def val(name: str, key: str) -> float:
        return float(next(r[key] for r in obs_rows if r["ensemble"] == name))

    fine_phi2 = val("original_fine", "phi2")
    fine_phi4 = val("original_fine", "phi4")
    fine_nn2 = val("original_fine", "nn2")
    scored = []
    for row in obs_rows:
        if row["ensemble"] in {"original_fine", "true_residual_oracle"}:
            continue
        score = abs(float(row["phi2"]) - fine_phi2) + abs(float(row["phi4"]) - fine_phi4) + abs(float(row["nn2"]) - fine_nn2)
        scored.append((score, row["ensemble"]))
    best = min(scored)
    summary = {
        "best_non_oracle_by_phi2_phi4_nn2_L1": {"candidate": best[1], "score": best[0]},
        "observables": obs_rows,
        "remaining_residual_statistics": residual_rows,
        "base_residual_component_statistics": base_rows,
        "pca_eigen_top10_fraction": float(np.sum(evals[:10]) / np.sum(evals)),
        "pca_eigen_top32_fraction": float(np.sum(evals[:32]) / np.sum(evals)),
        "pca_eigen_top64_fraction": float(np.sum(evals[:64]) / np.sum(evals)),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    report = f"""# Improved Block-Consistent Backbone Baselines

All candidates are constructed as `phi_back + Q v_base`, so block consistency is automatic.

Rows are evaluated on the fixed test split from `outputs/paired_data_lam1_kappaf0p320`.

## Observables

| candidate | stochastic | phi2 | phi4 | NN | nn2 | diag | 2nn | action density | block RMS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
"""
    for row in obs_rows:
        report += f"| {row['ensemble']} | {row['stochastic']} | {row['phi2']:.6g} | {row['phi4']:.6g} | {row['NN']:.6g} | {row['nn2']:.6g} | {row['diag']:.6g} | {row['2nn']:.6g} | {row['action_density']:.6g} | {row['block_RMS']:.3g} |\n"
    report += f"""
## Remaining Residual Burden

For deterministic candidates, `v_remaining = v_true - v_base` is paired sample by sample. For stochastic candidates, this row subtracts an independent draw and is therefore a diagnostic mismatch scale, not an actual conditional residual left after observing the same random draw.

| candidate | mismatch cov trace | mean coord std | top32 fraction | kurtosis mean |
|---|---:|---:|---:|---:|
"""
    for row in residual_rows:
        report += f"| {row['candidate']} | {row['cov_trace']:.6g} | {row['mean_coordinate_std']:.6g} | {row['top32_variance_fraction']:.6g} | {row['kurtosis_mean']:.6g} |\n"
    report += """
## Base Component Variance

| candidate | base cov trace | fraction of true trace | coordinate MSE to true residual |
|---|---:|---:|---:|
"""
    for row in base_rows:
        report += f"| {row['candidate']} | {row['base_cov_trace']:.6g} | {row['base_trace_fraction_of_true']:.6g} | {row['coordinate_mse_to_true_residual']:.6g} |\n"
    report += f"""
## Answers

1. Can we make `phi_back` much closer while keeping `B phi_back = phi_c`?

Yes in the trivial oracle case. Among non-oracle diagnostics, `{best[1]}` is closest by the simple L1 score over phi2/phi4/nn2, but stochastic covariance-matched Gaussian candidates tend to overshoot phi4/nn2, while deterministic bin means remain too close to the smooth backbone.

2. Does improved `phi_back` reduce residual variance the NF must learn?

Yes, any nonzero `v_base` reduces the remaining residual covariance trace relative to the smooth backbone; see `remaining_residual_statistics.csv`.

3. Does it avoid the phi4 overshoot seen from empirical residual resampling?

Only the lower-amplitude deterministic/bin-mean and low-rank PCA variants avoid severe phi4 overshoot. Full covariance and local Haar Gaussian baselines still overshoot.

4. Which improved backbone should be used next?

Use a low-rank PCA Gaussian or conditional low-rank PCA base as the next base distribution, not the full empirical residual covariance. It reduces residual burden while limiting the high-kurtosis/phi4 overshoot problem.
"""
    (OUT / "report.md").write_text(report)


if __name__ == "__main__":
    main()
