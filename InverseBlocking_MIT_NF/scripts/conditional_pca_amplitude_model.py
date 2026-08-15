#!/usr/bin/env python3
"""Conditional low-rank PCA amplitude model for exact block-consistent inverse blocking."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
if str(PROJECT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT / "scripts"))
import local_nullspace_pilot as local_pilot  # type: ignore
import nullspace_conditional_nf_pilot as pilot  # type: ignore


DATA = PROJECT / "outputs" / "paired_data_lam1_kappaf0p320"
OUT = PROJECT / "outputs" / "conditional_pca_amplitude_model"
SEED = 20240623
K_VALUES = [16, 24, 32, 48]
S_GLOBALS = [0.8, 0.9, 1.0, 1.1]
N_DRAWS = 4
RIDGE_ALPHA = 1.0e-3
LOGVAR_ALPHA = 1.0e-3
LOGVAR_EPS = 1.0e-6
LOGVAR_CLIP = 2.0


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


def feature_names() -> list[str]:
    return [
        "coarse_m",
        "coarse_abs_m",
        "coarse_phi2",
        "coarse_phi4",
        "coarse_NN",
        "coarse_nn2",
        "coarse_diag",
        "coarse_2nn",
        "coarse_Binder_U4",
        "coarse_xi_over_L",
        "coarse_action_density",
        "coarse_action_hopping_density",
        "coarse_action_phi2_density",
        "coarse_action_phi4_density",
        "back_m",
        "back_abs_m",
        "back_phi2",
        "back_phi4",
        "back_NN",
        "back_nn2",
        "back_diag",
        "back_2nn",
        "back_Binder_U4",
        "back_xi_over_L",
        "back_action_density",
        "back_action_hopping_density",
        "back_action_phi2_density",
        "back_action_phi4_density",
        "delta_phi2",
        "delta_phi4",
        "delta_NN",
        "delta_nn2",
        "delta_diag",
        "delta_2nn",
        "delta_action_density",
        "delta_abs_m",
    ]


def make_features(fine: np.ndarray, coarse: np.ndarray, back: np.ndarray, splits: dict[str, np.ndarray]) -> tuple[list[str], np.ndarray, list[dict[str, object]]]:
    names = feature_names()
    split_labels = np.empty(len(fine), dtype=object)
    for split_name, idx in splits.items():
        split_labels[idx] = split_name
    X = np.empty((len(fine), len(names)), dtype=np.float64)
    rows: list[dict[str, object]] = []
    for i in range(len(fine)):
        c = obs_with_m(coarse[i : i + 1])
        b = obs_with_m(back[i : i + 1])
        feat = np.array([
            c["m"],
            c["abs_m"],
            c["phi2"],
            c["phi4"],
            c["NN"],
            c["nn2"],
            c["diag"],
            c["2nn"],
            c["Binder_U4"],
            c["xi/L"],
            c["action_density"],
            c["action_hopping_density"],
            c["action_phi2_density"],
            c["action_phi4_density"],
            b["m"],
            b["abs_m"],
            b["phi2"],
            b["phi4"],
            b["NN"],
            b["nn2"],
            b["diag"],
            b["2nn"],
            b["Binder_U4"],
            b["xi/L"],
            b["action_density"],
            b["action_hopping_density"],
            b["action_phi2_density"],
            b["action_phi4_density"],
            b["phi2"] - c["phi2"],
            b["phi4"] - c["phi4"],
            b["NN"] - c["NN"],
            b["nn2"] - c["nn2"],
            b["diag"] - c["diag"],
            b["2nn"] - c["2nn"],
            b["action_density"] - c["action_density"],
            b["abs_m"] - c["abs_m"],
        ], dtype=np.float64)
        feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
        X[i] = feat
        row = {"sample": i, "split": str(split_labels[i])}
        for j, name in enumerate(names):
            row[name] = float(feat[j])
        rows.append(row)
    return names, X, rows


def standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    std = X.std(axis=0, ddof=1)
    std = np.where(std > 0, std, 1.0)
    return mean, std


def ridge_fit(X: np.ndarray, Y: np.ndarray, alpha: float) -> np.ndarray:
    X1 = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
    reg = np.eye(X1.shape[1], dtype=np.float64)
    reg[0, 0] = 0.0
    return np.linalg.solve(X1.T @ X1 + alpha * reg, X1.T @ Y)


def ridge_predict(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    X1 = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
    return X1 @ beta


def pca_fit(v_train: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = v_train.mean(axis=0)
    cov = np.cov(v_train - mean, rowvar=False)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    return mean, np.maximum(evals[order], 0.0), evecs[:, order]


def project_coords(v: np.ndarray, mean_v: np.ndarray, evecs: np.ndarray, k: int) -> np.ndarray:
    return (v - mean_v) @ evecs[:, :k]


def reconstruct_v(coords: np.ndarray, mean_v: np.ndarray, evecs: np.ndarray, k: int) -> np.ndarray:
    return mean_v + coords @ evecs[:, :k].T


def fit_model(X_train: np.ndarray, coords_train: np.ndarray, evals: np.ndarray, variant: str, k: int) -> dict[str, np.ndarray]:
    feat_mean, feat_scale = standardize(X_train)
    Xs = (X_train - feat_mean) / feat_scale
    out: dict[str, np.ndarray] = {"feature_mean": feat_mean, "feature_scale": feat_scale}
    if variant in {"conditional_mean", "conditional_mean_var"}:
        out[f"mu_beta_K{k}"] = ridge_fit(Xs, coords_train, RIDGE_ALPHA)
    if variant in {"conditional_var", "conditional_mean_var"}:
        if variant == "conditional_mean_var":
            mu = ridge_predict(Xs, out[f"mu_beta_K{k}"])
            resid = coords_train - mu
        else:
            resid = coords_train
        base = np.sqrt(evals[:k])[None, :]
        target = np.log((resid / np.maximum(base, 1e-30)) ** 2 + LOGVAR_EPS)
        out[f"logvar_beta_K{k}"] = ridge_fit(Xs, target, LOGVAR_ALPHA)
    return out


def sample_coords(
    rng: np.random.Generator,
    X: np.ndarray,
    evals: np.ndarray,
    variant: str,
    k: int,
    s_global: float,
    model: dict[str, np.ndarray],
    n_draws: int,
) -> np.ndarray:
    Xs = (X - model["feature_mean"]) / model["feature_scale"]
    base = np.sqrt(evals[:k])[None, :] * s_global
    mu = np.zeros((len(X), k), dtype=np.float64)
    sigma = np.broadcast_to(base, (len(X), k)).copy()
    if variant in {"conditional_mean", "conditional_mean_var"}:
        mu = ridge_predict(Xs, model[f"mu_beta_K{k}"])
    if variant in {"conditional_var", "conditional_mean_var"}:
        logvar = ridge_predict(Xs, model[f"logvar_beta_K{k}"])
        logvar = np.clip(logvar, -LOGVAR_CLIP, LOGVAR_CLIP)
        sigma = sigma * np.exp(0.5 * logvar)
    out = []
    for _ in range(n_draws):
        z = rng.normal(size=(len(X), k))
        out.append(mu + sigma * z)
    return np.stack(out, axis=0)


def sample_baseline(
    rng: np.random.Generator,
    X: np.ndarray,
    evals: np.ndarray,
    k: int,
    s_global: float,
    n_draws: int,
) -> np.ndarray:
    base = np.sqrt(evals[:k])[None, :] * s_global
    out = []
    for _ in range(n_draws):
        z = rng.normal(size=(len(X), k))
        out.append(base * z)
    return np.stack(out, axis=0)


def phi_from_v(v: np.ndarray, back: np.ndarray, q: np.ndarray) -> np.ndarray:
    detail = v @ q.T
    return back + detail.reshape(len(v), 16, 16)


def row_score(row: dict[str, float], target: dict[str, float]) -> float:
    return abs(row["phi2"] - target["phi2"]) + abs(row["phi4"] - target["phi4"]) + abs(row["nn2"] - target["nn2"])


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(f"Refusing to overwrite existing outputs in {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)

    w = load_weights()
    fine = np.load(DATA / "fine_configs.npy").astype(np.float64)
    coarse = np.load(DATA / "coarse_blocked_configs.npy").astype(np.float64)
    back = np.load(DATA / "backbone_configs.npy").astype(np.float64)
    v_true = np.load(DATA / "residual_v_true.npy").astype(np.float64)
    splits_npz = np.load(DATA / "split_indices.npz")
    splits = {k: splits_npz[k].astype(int) for k in ["train", "val", "test"]}

    q = local_q_basis(w)
    mean_v, evals, evecs = pca_fit(v_true[splits["train"]])
    names, X, feature_rows = make_features(fine, coarse, back, splits)
    write_csv(OUT / "feature_table.csv", feature_rows)

    target_val = ensemble_metrics(fine[splits["val"]], coarse[splits["val"]], w)
    target_test = ensemble_metrics(fine[splits["test"]], coarse[splits["test"]], w)

    model_store: dict[str, np.ndarray] = {
        "feature_names": np.array(names, dtype=object),
        "feature_mean": X[splits["train"]].mean(axis=0),
        "feature_scale": np.where(X[splits["train"]].std(axis=0, ddof=1) > 0, X[splits["train"]].std(axis=0, ddof=1), 1.0),
        "pca_mean_v": mean_v,
        "pca_evals": evals,
        "pca_evecs": evecs,
        "Q_basis": q,
        "train_indices": splits["train"],
        "val_indices": splits["val"],
        "test_indices": splits["test"],
    }

    model_cache: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    scan_rows: list[dict[str, object]] = []
    for variant in ["conditional_mean", "conditional_var", "conditional_mean_var"]:
        for k in K_VALUES:
            model = fit_model(X[splits["train"]], project_coords(v_true[splits["train"]], mean_v, evecs, k), evals, variant, k)
            model_cache[(variant, k)] = model
            for name, arr in model.items():
                if name in {"feature_mean", "feature_scale"} or name.startswith(("mu_beta", "logvar_beta")):
                    model_store[f"{variant}_K{k}__{name}"] = arr

            for s_global in S_GLOBALS:
                rng = np.random.default_rng(SEED + 1000 * k + int(round(100 * s_global)) + {"conditional_mean": 0, "conditional_var": 1, "conditional_mean_var": 2}[variant])
                coords_draws = sample_coords(rng, X[splits["val"]], evals, variant, k, s_global, model, N_DRAWS)
                phis = np.concatenate([phi_from_v(coords_draws[d] @ evecs[:, :k].T + mean_v, back[splits["val"]], q) for d in range(N_DRAWS)], axis=0)
                coarse_rep = np.tile(coarse[splits["val"]], (N_DRAWS, 1, 1))
                row = {"split": "val", "variant": variant, "K": k, "s_global": s_global, **ensemble_metrics(phis, coarse_rep, w)}
                row["score_phi2_phi4_nn2"] = row_score(row, target_val)
                row["action_abs_error"] = abs(row["action_density"] - target_val["action_density"])
                scan_rows.append(row)

    for k in K_VALUES:
        for s_global in S_GLOBALS:
            rng = np.random.default_rng(SEED + 5000 + 100 * k + int(round(100 * s_global)))
            coords_draws = sample_baseline(rng, X[splits["val"]], evals, k, s_global, N_DRAWS)
            phis = np.concatenate([phi_from_v(coords_draws[d] @ evecs[:, :k].T + mean_v, back[splits["val"]], q) for d in range(N_DRAWS)], axis=0)
            coarse_rep = np.tile(coarse[splits["val"]], (N_DRAWS, 1, 1))
            row = {"split": "val", "variant": "unconditional", "K": k, "s_global": s_global, **ensemble_metrics(phis, coarse_rep, w)}
            row["score_phi2_phi4_nn2"] = row_score(row, target_val)
            row["action_abs_error"] = abs(row["action_density"] - target_val["action_density"])
            scan_rows.append(row)

    write_csv(OUT / "validation_scan.csv", scan_rows)

    def select_best(rows: list[dict[str, object]]) -> dict[str, object]:
        return min(rows, key=lambda r: float(r["score_phi2_phi4_nn2"]))

    selected = {
        variant: select_best([r for r in scan_rows if r["variant"] == variant]) for variant in ["conditional_mean", "conditional_var", "conditional_mean_var"]
    }
    baseline = select_best([r for r in scan_rows if r["variant"] == "unconditional" and int(r["K"]) == 32 and abs(float(r["s_global"]) - 0.9) < 1e-12])

    test_rows: list[dict[str, object]] = []
    test_rows.append({"ensemble": "original_fine", **ensemble_metrics(fine[splits["test"]], coarse[splits["test"]], w), "note": "reference"})
    test_rows.append({"ensemble": "blocked_coarse", **ensemble_metrics(coarse[splits["test"]], None, None), "note": "condition"})
    test_rows.append({"ensemble": "phi_backbone", **ensemble_metrics(back[splits["test"]], coarse[splits["test"]], w), "note": "smooth backbone"})
    test_rows.append({"ensemble": "true_reconstruction_phi_back_plus_Q_v_true", **ensemble_metrics(fine[splits["test"]], coarse[splits["test"]], w), "note": "oracle"})

    # Gaussian residual baseline = fixed PCA baseline K=32, s=0.9.
    base_rng = np.random.default_rng(SEED + 31415)
    base_coords = sample_baseline(base_rng, X[splits["test"]], evals, 32, 0.9, N_DRAWS)
    base_phi = np.concatenate([phi_from_v(base_coords[d] @ evecs[:, :32].T + mean_v, back[splits["test"]], q) for d in range(N_DRAWS)], axis=0)
    test_rows.append(
        {
            "ensemble": "Gaussian_residual_baseline_K32_s0p9",
            **ensemble_metrics(base_phi, np.tile(coarse[splits["test"]], (N_DRAWS, 1, 1)), w),
            "note": "fixed PCA baseline",
        }
    )

    mle_test = PROJECT / "outputs" / "paired_data_lam1_kappaf0p320" / "whitened_mle_diagnostic" / "test_samples.npy"
    if mle_test.exists():
        mle_arr = np.load(mle_test).astype(np.float64)
        test_rows.append({"ensemble": "MLE_residual_samples", **ensemble_metrics(mle_arr, coarse[splits["test"]], w), "note": "previous whitening MLE"})
    rk_test = PROJECT / "outputs" / "paired_data_lam1_kappaf0p320" / "whitened_reverse_kl_admixture" / "test_samples.npy"
    if rk_test.exists():
        rk_arr = np.load(rk_test).astype(np.float64)
        test_rows.append({"ensemble": "reverse_KL_samples", **ensemble_metrics(rk_arr, coarse[splits["test"]], w), "note": "previous reverse-KL"})

    for variant in ["conditional_mean", "conditional_var", "conditional_mean_var"]:
        row = selected[variant]
        k = int(row["K"])
        s_global = float(row["s_global"])
        model = model_cache[(variant, k)]
        rng = np.random.default_rng(SEED + 9000 + 100 * k + int(round(100 * s_global)) + {"conditional_mean": 0, "conditional_var": 1, "conditional_mean_var": 2}[variant])
        coords_draws = sample_coords(rng, X[splits["test"]], evals, variant, k, s_global, model, N_DRAWS)
        phis = np.concatenate([phi_from_v(coords_draws[d] @ evecs[:, :k].T + mean_v, back[splits["test"]], q) for d in range(N_DRAWS)], axis=0)
        test_rows.append(
            {
                "ensemble": f"{variant}_best_K{k}_s{s_global}",
                **ensemble_metrics(phis, np.tile(coarse[splits["test"]], (N_DRAWS, 1, 1)), w),
                "note": f"selected validation score={float(row['score_phi2_phi4_nn2']):.6g}",
            }
        )

    write_csv(OUT / "test_observables.csv", test_rows)

    model_store["selected_conditional_mean"] = np.array(json.dumps(selected["conditional_mean"], default=float))
    model_store["selected_conditional_var"] = np.array(json.dumps(selected["conditional_var"], default=float))
    model_store["selected_conditional_mean_var"] = np.array(json.dumps(selected["conditional_mean_var"], default=float))
    model_store["selected_unconditional"] = np.array(json.dumps(baseline, default=float))
    np.savez_compressed(OUT / "model_coefficients.npz", **model_store)

    summary = {
        "K_values": K_VALUES,
        "s_globals": S_GLOBALS,
        "ridge_alpha": RIDGE_ALPHA,
        "logvar_alpha": LOGVAR_ALPHA,
        "n_draws": N_DRAWS,
        "selected": {k: {kk: (float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else v) for kk, v in row.items()} for k, row in selected.items()},
        "baseline": {kk: (float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else v) for kk, v in baseline.items()},
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    def fmt(row: dict[str, object]) -> str:
        return f"K={int(row['K'])}, s={float(row['s_global']):.1f}, phi2/phi4/nn2={float(row['phi2']):.6g}/{float(row['phi4']):.6g}/{float(row['nn2']):.6g}, score={float(row['score_phi2_phi4_nn2']):.6g}"

    report = f"""# Conditional PCA Amplitude Model

Exact block consistency is preserved by reconstructing fields as `phi = phi_back + Q v_sample`.

## Validation Selection

- conditional mean: {fmt(selected['conditional_mean'])}
- conditional variance: {fmt(selected['conditional_var'])}
- conditional mean + variance: {fmt(selected['conditional_mean_var'])}
- fixed PCA baseline: K=32, s=0.9

## Comparison Table

| ensemble | m | abs(m) | phi2 | phi4 | NN | nn2 | diag | 2nn | Binder U4 | xi/L | action density | block RMS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
"""
    for row in test_rows:
        report += (
            f"| {row['ensemble']} | {row['m']:.6g} | {row['abs_m']:.6g} | {row['phi2']:.6g} | {row['phi4']:.6g} | {row['NN']:.6g} | "
            f"{row['nn2']:.6g} | {row['diag']:.6g} | {row['2nn']:.6g} | {row['Binder_U4']:.6g} | {row['xi/L']:.6g} | "
            f"{row['action_density']:.6g} | {row['block_RMS']:.3g} |\n"
        )

    report += f"""
## Answers

1. Does conditional PCA amplitude modeling raise phi2 without overshooting phi4/nn2?

Compare the selected conditional rows against `Gaussian_residual_baseline_K32_s0p9`. The validation scan in `validation_scan.csv` is the selection criterion.

2. Are conditional means important, or mainly conditional variances?

Use the selected conditional-mean and conditional-variance rows. If the mean-only row is closest on phi2 while the variance-only row is not, the dominant effect is in the mean. If the variance-only row is the better correction, the amplitude width is the main signal.

3. Which features matter most?

The model uses coarse and backbone observables plus coarse-backbone deltas. The ridge coefficients in `model_coefficients.npz` identify which channels are active.

4. Which K and amplitude setting is best on validation and test?

The scan is in `validation_scan.csv`. The selected rows above are the best validation points for each variant.

5. Does this outperform the fixed PCA K=32, s=0.9 baseline?

Compare each selected row to `Gaussian_residual_baseline_K32_s0p9`. A useful model should raise phi2 toward the fine target without pushing phi4 or nn2 past it.

6. Should this conditional PCA model become the base distribution for a later small flow correction?

Only if it improves on the fixed PCA baseline and on the previous MLE / reverse-KL residual samples. Otherwise the next change should be in the features or the coordinate system, not another residual flow stacked on top.
"""
    (OUT / "report.md").write_text(report)


if __name__ == "__main__":
    main()
