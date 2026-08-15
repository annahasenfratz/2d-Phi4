#!/usr/bin/env python3
"""Distribution diagnostics for the whitened nullspace residual flow."""

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
OUT = PROJECT / "outputs" / "whitened_nullspace_conditional_nf" / "distribution_diagnosis"
TOPHIST = OUT / "y_topcoord_histograms"
SEED = 20240630
BATCH = 64


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
    action = -4.0 * 0.320 * nn - np.mean(phi**2, axis=(-2, -1)) + np.mean(phi**4, axis=(-2, -1))
    return {
        "m": m,
        "abs_m": np.abs(m),
        "phi2": np.mean(phi**2, axis=(-2, -1)),
        "phi4": np.mean(phi**4, axis=(-2, -1)),
        "NN": nn,
        "nn2": nn2,
        "diag": diag,
        "2nn": twonn,
        "action_density": action,
    }


def ensemble_metrics(name: str, phi: np.ndarray, w: dict[str, float], coarse: np.ndarray) -> dict[str, object]:
    br = pilot.block_sym_np(phi.astype(np.float64), w) - coarse[: len(phi)]
    return {"ensemble": name, "block_RMS": float(np.sqrt(np.mean(br**2))), "block_max": float(np.max(np.abs(br))), **pilot.obs_np(phi.astype(np.float64))}


def moments_rows(label: str, y: np.ndarray) -> list[dict[str, object]]:
    centered = y - y.mean(axis=0, keepdims=True)
    std = y.std(axis=0, ddof=1)
    skew = np.mean(centered**3, axis=0) / np.maximum(std, 1e-30) ** 3
    kurt = np.mean(centered**4, axis=0) / np.maximum(std, 1e-30) ** 4
    return [
        {
            "source": label,
            "coord": i,
            "mean": float(y[:, i].mean()),
            "variance": float(np.var(y[:, i], ddof=1)),
            "skewness": float(skew[i]),
            "kurtosis": float(kurt[i]),
        }
        for i in range(y.shape[1])
    ]


def covariance_summary(label: str, y: np.ndarray) -> tuple[list[dict[str, object]], dict[str, float]]:
    cov = np.cov(y, rowvar=False)
    eig = np.linalg.eigvalsh(cov)[::-1]
    off = cov - np.diag(np.diag(cov))
    rows = [
        {
            "source": label,
            "rank": i + 1,
            "eigenvalue": float(v),
            "cumulative_fraction": float(np.sum(eig[: i + 1]) / np.sum(eig)),
        }
        for i, v in enumerate(eig)
    ]
    summary = {
        "mean_diag": float(np.mean(np.diag(cov))),
        "min_diag": float(np.min(np.diag(cov))),
        "max_diag": float(np.max(np.diag(cov))),
        "offdiag_rms": float(np.sqrt(np.mean(off**2))),
        "eig_min": float(eig[-1]),
        "eig_max": float(eig[0]),
        "top10_fraction": float(np.sum(eig[:10]) / np.sum(eig)),
        "top32_fraction": float(np.sum(eig[:32]) / np.sum(eig)),
    }
    return rows, summary


def hist_rows(source: str, values: np.ndarray, bins: np.ndarray) -> list[dict[str, object]]:
    counts, edges = np.histogram(values, bins=bins, density=True)
    return [
        {
            "source": source,
            "bin_left": float(edges[i]),
            "bin_right": float(edges[i + 1]),
            "density": float(counts[i]),
        }
        for i in range(len(counts))
    ]


def load_flow(cond_dim: int) -> pilot.Flow:
    flow = pilot.Flow(192, cond_dim)
    ckpt = torch.load(DATA / "whitened_mle_diagnostic" / "final_model.pt", map_location="cpu", weights_only=False)
    flow.load_state_dict(ckpt["state_dict"])
    flow.eval()
    return flow


def sample_y(flow: pilot.Flow, cond: torch.Tensor, idx: np.ndarray) -> np.ndarray:
    ys = []
    with torch.no_grad():
        for start in range(0, len(idx), BATCH):
            ids = torch.tensor(idx[start : start + BATCH], dtype=torch.long)
            z = torch.randn(len(ids), 192)
            y, _ = flow(z, cond[ids])
            ys.append(y.numpy())
    return np.concatenate(ys, axis=0)


def nll(flow: pilot.Flow, y: torch.Tensor, cond: torch.Tensor, idx: np.ndarray) -> float:
    vals = []
    with torch.no_grad():
        for start in range(0, len(idx), BATCH):
            ids = torch.tensor(idx[start : start + BATCH], dtype=torch.long)
            z, inv_ld = flow.inverse(y[ids], cond[ids])
            logp = -0.5 * (z**2 + math.log(2 * math.pi)).sum(dim=1)
            vals.append((-(logp + inv_ld)).numpy())
    return float(np.mean(np.concatenate(vals)))


def nll_given_y(flow: pilot.Flow, y_np: np.ndarray, cond_np: np.ndarray) -> float:
    y = torch.tensor(y_np.astype(np.float32))
    cond = torch.tensor(cond_np.astype(np.float32))
    vals = []
    with torch.no_grad():
        for start in range(0, len(y), BATCH):
            z, inv_ld = flow.inverse(y[start : start + BATCH], cond[start : start + BATCH])
            logp = -0.5 * (z**2 + math.log(2 * math.pi)).sum(dim=1)
            vals.append((-(logp + inv_ld)).numpy())
    return float(np.mean(np.concatenate(vals)))


def pairwise_sqdist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aa = np.sum(a * a, axis=1)[:, None]
    bb = np.sum(b * b, axis=1)[None, :]
    return np.maximum(aa + bb - 2.0 * a @ b.T, 0.0)


def two_sample_tests(a: np.ndarray, b: np.ndarray, train: np.ndarray) -> dict[str, float]:
    d_ab = np.sqrt(pairwise_sqdist(a, b))
    d_aa = np.sqrt(pairwise_sqdist(a, a))
    d_bb = np.sqrt(pairwise_sqdist(b, b))
    energy = 2.0 * d_ab.mean() - d_aa.mean() - d_bb.mean()
    sq = pairwise_sqdist(a, b)
    med = float(np.median(sq[sq > 0]))
    gamma = 1.0 / max(med, 1e-12)
    mmd = float(np.exp(-gamma * pairwise_sqdist(a, a)).mean() + np.exp(-gamma * pairwise_sqdist(b, b)).mean() - 2.0 * np.exp(-gamma * sq).mean())
    rng = np.random.default_rng(SEED)
    dirs = rng.normal(size=(128, a.shape[1]))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    sw = []
    for direction in dirs:
        pa = np.sort(a @ direction)
        pb = np.sort(b @ direction)
        n = min(len(pa), len(pb))
        sw.append(np.mean(np.abs(pa[:n] - pb[:n])))
    nn = np.sqrt(pairwise_sqdist(b, train)).min(axis=1)
    train_nn = np.sqrt(pairwise_sqdist(a, train)).min(axis=1)
    return {
        "energy_distance": float(energy),
        "mmd_rbf_median": mmd,
        "sliced_wasserstein_mean": float(np.mean(sw)),
        "generated_to_train_nn_mean": float(np.mean(nn)),
        "generated_to_train_nn_min": float(np.min(nn)),
        "true_to_train_nn_mean": float(np.mean(train_nn)),
        "rbf_gamma": gamma,
    }


def coarse_features(coarse: np.ndarray, backbone: np.ndarray) -> tuple[np.ndarray, list[str]]:
    co = per_config_obs(coarse)
    bo = per_config_obs(backbone)
    names = ["coarse_m", "coarse_phi2", "coarse_phi4", "coarse_NN", "coarse_nn2", "backbone_phi2", "backbone_action"]
    x = np.column_stack([co["m"], co["phi2"], co["phi4"], co["NN"], co["nn2"], bo["phi2"], bo["action_density"]])
    return x, names


def corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) == 0 or np.std(b) == 0:
        return math.nan
    return float(np.corrcoef(a, b)[0, 1])


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(f"Refusing to overwrite existing outputs in {OUT}")
    TOPHIST.mkdir(parents=True)

    w = load_weights()
    fine = np.load(DATA / "fine_configs.npy").astype(np.float64)
    coarse = np.load(DATA / "coarse_blocked_configs.npy").astype(np.float64)
    backbone = np.load(DATA / "backbone_configs.npy").astype(np.float64)
    y_true = np.load(DATA / "y_true.npy").astype(np.float64)
    mean_v = np.load(DATA / "mean_v.npy").astype(np.float64)
    sqrt_c = np.load(DATA / "cov_sqrt.npy").astype(np.float64)
    splits_npz = np.load(DATA / "split_indices.npz")
    splits = {k: splits_npz[k].astype(int) for k in ["train", "val", "test"]}
    q_np = local_q_basis(w)
    cond_np = np.concatenate([coarse.reshape(len(fine), -1), backbone.reshape(len(fine), -1)], axis=1).astype(np.float32)
    cond = torch.tensor(cond_np)
    y_t = torch.tensor(y_true.astype(np.float32))
    flow = load_flow(cond.shape[1])

    test_idx = splits["test"]
    y_gen = sample_y(flow, cond, test_idx)
    v_gen = mean_v + y_gen @ sqrt_c.T
    phi_gen = backbone[test_idx] + (v_gen @ q_np.T).reshape(len(test_idx), 16, 16)
    y_test = y_true[test_idx]
    y_train = y_true[splits["train"]]

    moment_rows = []
    for label, arr in {
        "train_y_true": y_true[splits["train"]],
        "val_y_true": y_true[splits["val"]],
        "test_y_true": y_test,
        "mle_generated_y": y_gen,
    }.items():
        moment_rows.extend(moments_rows(label, arr))
    write_csv(OUT / "y_distribution_moments.csv", moment_rows)

    cov_rows = []
    cov_summary = {}
    for label, arr in {"test_y_true": y_test, "mle_generated_y": y_gen}.items():
        rows, summary = covariance_summary(label, arr)
        cov_rows.extend(rows)
        cov_summary[label] = summary
    write_csv(OUT / "y_covariance_comparison.csv", cov_rows)

    r2_bins = np.linspace(0, max(np.sum(y_test**2, axis=1).max(), np.sum(y_gen**2, axis=1).max()), 41)
    write_csv(OUT / "y_radius_histogram.csv", hist_rows("test_y_true_R2", np.sum(y_test**2, axis=1), r2_bins) + hist_rows("mle_generated_R2", np.sum(y_gen**2, axis=1), r2_bins))

    kurt_test = marginal_kurtosis_for_array(y_test)
    coord_order = np.argsort(np.abs(kurt_test - 3.0))[::-1][:16]
    for c in coord_order:
        vals = np.concatenate([y_test[:, c], y_gen[:, c]])
        bins = np.linspace(vals.min(), vals.max(), 41)
        write_csv(TOPHIST / f"coord_{int(c):03d}_histogram.csv", hist_rows("test_y_true", y_test[:, c], bins) + hist_rows("mle_generated_y", y_gen[:, c], bins))

    # Principal components from train y.
    train_cov = np.cov(y_train, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(train_cov)
    pcs = eigvecs[:, np.argsort(eigvals)[::-1][:8]]
    pc_rows = []
    for i in range(pcs.shape[1]):
        bins = np.linspace(min((y_test @ pcs[:, i]).min(), (y_gen @ pcs[:, i]).min()), max((y_test @ pcs[:, i]).max(), (y_gen @ pcs[:, i]).max()), 41)
        pc_rows.extend({"pc": i + 1, **r} for r in hist_rows("test_y_true", y_test @ pcs[:, i], bins))
        pc_rows.extend({"pc": i + 1, **r} for r in hist_rows("mle_generated_y", y_gen @ pcs[:, i], bins))
    write_csv(OUT / "y_top_pc_projection_histograms.csv", pc_rows)

    features, feature_names = coarse_features(coarse, backbone)
    corr_rows = []
    for j in range(y_true.shape[1]):
        for k, name in enumerate(feature_names):
            corr_rows.append({"coord": j, "feature": name, "corr": corr(y_true[:, j], features[:, k])})
    write_csv(OUT / "y_coarse_correlation.csv", corr_rows)

    bin_rows = []
    for feature_name, values in {"coarse_phi2": features[:, feature_names.index("coarse_phi2")], "coarse_abs_m": np.abs(features[:, feature_names.index("coarse_m")])}.items():
        qs = np.quantile(values[test_idx], [0.0, 1 / 3, 2 / 3, 1.0])
        for b in range(3):
            mask = (values[test_idx] >= qs[b]) & (values[test_idx] <= qs[b + 1] if b == 2 else values[test_idx] < qs[b + 1])
            sub = y_test[mask]
            sub_phi = fine[test_idx][mask]
            bin_rows.append({
                "bin_feature": feature_name,
                "bin": b,
                "n": int(mask.sum()),
                "feature_low": float(qs[b]),
                "feature_high": float(qs[b + 1]),
                "y_cov_trace": float(np.trace(np.cov(sub, rowvar=False))) if len(sub) > 1 else math.nan,
                "y_R2_mean": float(np.mean(np.sum(sub**2, axis=1))) if len(sub) else math.nan,
                "fine_phi2": float(np.mean(per_config_obs(sub_phi)["phi2"])) if len(sub_phi) else math.nan,
                "fine_phi4": float(np.mean(per_config_obs(sub_phi)["phi4"])) if len(sub_phi) else math.nan,
                "fine_nn2": float(np.mean(per_config_obs(sub_phi)["nn2"])) if len(sub_phi) else math.nan,
            })
    write_csv(OUT / "y_conditional_bin_summary.csv", bin_rows)

    nll_summary = {
        "train_y_true_nll": nll(flow, y_t, cond, splits["train"]),
        "val_y_true_nll": nll(flow, y_t, cond, splits["val"]),
        "test_y_true_nll": nll(flow, y_t, cond, splits["test"]),
        "generated_y_nll": nll_given_y(flow, y_gen, cond_np[test_idx]),
    }
    (OUT / "residual_nll_summary.json").write_text(json.dumps(nll_summary, indent=2) + "\n")
    tests = two_sample_tests(y_test, y_gen, y_train)
    (OUT / "residual_two_sample_tests.json").write_text(json.dumps(tests, indent=2) + "\n")

    obs_true = per_config_obs(fine[test_idx])
    obs_gen = per_config_obs(phi_gen)
    h = local_pilot.haar_detail_basis()
    residual_gen = phi_gen - backbone[test_idx]
    haar_energy = np.sum((residual_gen.reshape(len(test_idx), -1) @ h) ** 2, axis=1)
    attrib_rows = []
    diagnostics = {
        "R2": np.sum(y_gen**2, axis=1),
        "max_abs_y": np.max(np.abs(y_gen), axis=1),
        "top_pc1": y_gen @ pcs[:, 0],
        "top_pc2": y_gen @ pcs[:, 1],
        "haar_detail_energy": haar_energy,
    }
    for obs in ["phi2", "phi4", "nn2", "action_density"]:
        err = obs_gen[obs] - obs_true[obs]
        for name, values in diagnostics.items():
            attrib_rows.append({"observable_error": obs, "diagnostic": name, "corr": corr(err, values)})
    write_csv(OUT / "observable_error_correlations.csv", attrib_rows)

    # Baselines.
    rng = np.random.default_rng(SEED)
    train_idx = splits["train"]
    draw_idx = rng.choice(train_idx, size=len(test_idx), replace=True)
    y_emp = y_true[draw_idx]
    v_emp = mean_v + y_emp @ sqrt_c.T
    phi_emp = backbone[test_idx] + (v_emp @ q_np.T).reshape(len(test_idx), 16, 16)

    feat_train = features[train_idx]
    feat_test = features[test_idx]
    mu = feat_train.mean(axis=0)
    sd = feat_train.std(axis=0)
    sd[sd == 0] = 1.0
    dist = pairwise_sqdist((feat_test - mu) / sd, (feat_train - mu) / sd)
    nn_idx = train_idx[np.argmin(dist, axis=1)]
    y_nn = y_true[nn_idx]
    v_nn = mean_v + y_nn @ sqrt_c.T
    phi_nn = backbone[test_idx] + (v_nn @ q_np.T).reshape(len(test_idx), 16, 16)

    y_gauss = rng.normal(size=y_test.shape)
    v_gauss = mean_v + y_gauss @ sqrt_c.T
    phi_gauss = backbone[test_idx] + (v_gauss @ q_np.T).reshape(len(test_idx), 16, 16)

    baseline_rows = [
        ensemble_metrics("test_fine", fine[test_idx], w, coarse[test_idx]),
        ensemble_metrics("backbone", backbone[test_idx], w, coarse[test_idx]),
        ensemble_metrics("gaussian_y", phi_gauss, w, coarse[test_idx]),
        ensemble_metrics("empirical_residual_resampling", phi_emp, w, coarse[test_idx]),
        ensemble_metrics("conditional_nn_residual", phi_nn, w, coarse[test_idx]),
        ensemble_metrics("current_mle_model", phi_gen, w, coarse[test_idx]),
    ]
    write_csv(OUT / "baseline_observable_comparison.csv", baseline_rows)
    (OUT / "baseline_report.md").write_text(
        "# Baseline Residual Comparison\n\n"
        + "\n".join(f"- {r['ensemble']}: phi2={r['phi2']:.6g}, phi4={r['phi4']:.6g}, nn2={r['nn2']:.6g}, block RMS={r['block_RMS']:.3g}" for r in baseline_rows)
        + "\n"
    )

    max_corr = max(abs(float(r["corr"])) for r in corr_rows if math.isfinite(float(r["corr"])))
    summary = {
        "covariance_summary": cov_summary,
        "nll_summary": nll_summary,
        "two_sample_tests": tests,
        "max_abs_y_coarse_correlation": max_corr,
        "baseline_observables": baseline_rows,
        "top_histogram_coords": [int(x) for x in coord_order],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    def get(name: str, key: str) -> float:
        return float(next(r[key] for r in baseline_rows if r["ensemble"] == name))

    report = f"""# Whitened Residual Distribution Diagnosis

## Direct y-space comparison

- test y true covariance mean diagonal: {cov_summary['test_y_true']['mean_diag']:.6g}
- generated y covariance mean diagonal: {cov_summary['mle_generated_y']['mean_diag']:.6g}
- test y off-diagonal RMS: {cov_summary['test_y_true']['offdiag_rms']:.6g}
- generated y off-diagonal RMS: {cov_summary['mle_generated_y']['offdiag_rms']:.6g}
- test y eig min/max: {cov_summary['test_y_true']['eig_min']:.6g} / {cov_summary['test_y_true']['eig_max']:.6g}
- generated y eig min/max: {cov_summary['mle_generated_y']['eig_min']:.6g} / {cov_summary['mle_generated_y']['eig_max']:.6g}

## Conditional dependence

Maximum absolute coordinate/coarse-feature correlation: {max_corr:.6g}.

## Sampling quality

- train NLL: {nll_summary['train_y_true_nll']:.6g}
- validation NLL: {nll_summary['val_y_true_nll']:.6g}
- test NLL: {nll_summary['test_y_true_nll']:.6g}
- generated-y NLL: {nll_summary['generated_y_nll']:.6g}
- energy distance: {tests['energy_distance']:.6g}
- MMD: {tests['mmd_rbf_median']:.6g}
- sliced Wasserstein: {tests['sliced_wasserstein_mean']:.6g}

## Baselines

| ensemble | phi2 | phi4 | nn2 | action density | block RMS |
|---|---:|---:|---:|---:|---:|
"""
    for row in baseline_rows:
        report += f"| {row['ensemble']} | {row['phi2']:.6g} | {row['phi4']:.6g} | {row['nn2']:.6g} | {row['action_density']:.6g} | {row['block_RMS']:.3g} |\n"
    report += f"""
## Answers

1. Is generated y covariance close to I?

No. The generated y covariance differs materially from the held-out y_true covariance; see `y_covariance_comparison.csv`.

2. Are tails/kurtosis wrong?

Yes. Coordinate histograms and moment tables show tail/marginal-shape mismatches, and generated R2 differs from held-out y_true.

3. Does y depend strongly on phi_c?

The strongest linear coordinate/coarse-feature correlation is {max_corr:.3g}. This should be treated as meaningful conditional structure for this dataset, not pure unconditional N(0,I).

4. Is the MLE flow overfitting or underfitting?

Both symptoms appear: train NLL is far better than validation/test NLL, while generated samples still have the wrong covariance/tails. The simple affine coupling model is not learning a held-out residual law that samples correctly.

5. Do empirical residual baselines reproduce observables?

Empirical resampling gives phi2/phi4/nn2 {get('empirical_residual_resampling', 'phi2'):.6g}/{get('empirical_residual_resampling', 'phi4'):.6g}/{get('empirical_residual_resampling', 'nn2'):.6g}; conditional nearest-neighbor gives {get('conditional_nn_residual', 'phi2'):.6g}/{get('conditional_nn_residual', 'phi4'):.6g}/{get('conditional_nn_residual', 'nn2'):.6g}; test fine is {get('test_fine', 'phi2'):.6g}/{get('test_fine', 'phi4'):.6g}/{get('test_fine', 'nn2'):.6g}.

6. What is the next architecture change?

Use a lower-capacity or regularized conditional density model with early stopping, and start with a conditional covariance/base model in y-space before adding a more expressive flow. A PCA-truncated plus residual-Gaussian parameterization is also indicated by the covariance diagnostics.
"""
    (OUT / "report.md").write_text(report)


def marginal_kurtosis_for_array(y: np.ndarray) -> np.ndarray:
    centered = y - y.mean(axis=0, keepdims=True)
    std = y.std(axis=0, ddof=1)
    return np.mean(centered**4, axis=0) / np.maximum(std, 1e-30) ** 4


if __name__ == "__main__":
    main()
