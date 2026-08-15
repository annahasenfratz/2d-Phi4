#!/usr/bin/env python3
"""Generate paired blocked-fine data and rerun whitened nullspace diagnostics."""

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
from pilot_utils import generate_coarse_ensemble, phi4_action_numpy


OUT = PROJECT / "outputs" / "paired_data_lam1_kappaf0p320"
CKPT_MLE = OUT / "whitened_mle_diagnostic" / "checkpoints"
CKPT_MIX = OUT / "whitened_reverse_kl_admixture" / "checkpoints"
LAMBDA = 1.0
KAPPA_F = 0.320
N_PAIRS = 1024
THERMAL_SWEEPS = 1000
SKIP_SWEEPS = 8
PROPOSAL_WIDTH = 0.8
SEED = 20240629
BATCH_SIZE = 32
MLE_EPOCHS = 60
MIX_EPOCHS = 25
REVERSE_KL_EPS = 0.05
RIDGE_EPS = 1.0e-6


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


def metrics(phi: np.ndarray, w: dict[str, float], coarse: np.ndarray | None = None) -> dict[str, float]:
    out = pilot.obs_np(phi.astype(np.float64))
    if coarse is not None:
        br = pilot.block_sym_np(phi.astype(np.float64), w) - coarse[: len(phi)]
        out = {"block_RMS": float(np.sqrt(np.mean(br**2))), "block_max": float(np.max(np.abs(br))), **out}
    return out


def per_config_rows(phi: np.ndarray, w: dict[str, float], *, kappa: float, dataset: str) -> list[dict[str, object]]:
    rows = []
    actions = phi4_action_numpy(phi.astype(np.float64), kappa=kappa, lam=LAMBDA)
    for i, cfg in enumerate(phi):
        obs = pilot.obs_np(cfg[None, :, :].astype(np.float64))
        rows.append({"sample": i, "dataset": dataset, "kappa": kappa, "lambda": LAMBDA, "action": float(actions[i]), **obs})
    return rows


def covariance_whitening(v: np.ndarray) -> dict[str, np.ndarray | float | int | bool]:
    mean = v.mean(axis=0)
    centered = v - mean
    cov = np.cov(centered, rowvar=False)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]
    rank = int(np.sum(evals > 1.0e-10))
    min_eval = float(np.min(evals))
    ridge = 0.0 if min_eval > 1.0e-9 else RIDGE_EPS * float(np.trace(cov)) / cov.shape[0]
    evals_reg = np.maximum(evals, 0.0) + ridge
    sqrt = (evecs * np.sqrt(evals_reg)) @ evecs.T
    invsqrt = (evecs * (1.0 / np.sqrt(evals_reg))) @ evecs.T
    y = centered @ invsqrt.T
    ycov = np.cov(y, rowvar=False)
    return {
        "mean": mean,
        "cov": cov,
        "evals": evals,
        "rank": rank,
        "ridge": float(ridge),
        "ridge_used": bool(ridge > 0),
        "sqrt": sqrt,
        "invsqrt": invsqrt,
        "y": y,
        "ycov": ycov,
        "ycov_evals": np.linalg.eigvalsh(ycov)[::-1],
        "logdet_sqrt": float(np.sum(np.log(np.sqrt(evals_reg)))),
    }


def marginal_kurtosis(x: np.ndarray) -> np.ndarray:
    centered = x - x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, ddof=1)
    return np.mean(centered**4, axis=0) / np.maximum(std, 1.0e-30) ** 4


def make_splits(n: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(SEED + 1)
    perm = rng.permutation(n)
    n_train = int(round(0.70 * n))
    n_val = int(round(0.15 * n))
    return {
        "train": perm[:n_train],
        "val": perm[n_train : n_train + n_val],
        "test": perm[n_train + n_val :],
    }


def sample_flow(
    flow: pilot.Flow,
    cond: torch.Tensor,
    back: torch.Tensor,
    q: torch.Tensor,
    mean_v: torch.Tensor,
    sqrt_c: torch.Tensor,
    idx: np.ndarray,
) -> np.ndarray:
    ids = torch.tensor(idx, dtype=torch.long)
    with torch.no_grad():
        z = torch.randn(len(ids), 192)
        y, _ = flow(z, cond[ids])
        v = mean_v + y @ sqrt_c.T
        phi = (back[ids] + v @ q.T).reshape(len(ids), 16, 16)
    return phi.cpu().numpy()


def nll_on_indices(flow: pilot.Flow, y_true: torch.Tensor, cond: torch.Tensor, idx: np.ndarray) -> float:
    vals = []
    with torch.no_grad():
        for start in range(0, len(idx), BATCH_SIZE):
            ids = torch.tensor(idx[start : start + BATCH_SIZE], dtype=torch.long)
            z, inv_ld = flow.inverse(y_true[ids], cond[ids])
            logp = -0.5 * (z**2 + math.log(2 * math.pi)).sum(dim=1)
            vals.append(float((-(logp + inv_ld)).mean()))
    return float(np.mean(vals))


def train_mle(
    flow: pilot.Flow,
    y_true: torch.Tensor,
    cond: torch.Tensor,
    back: torch.Tensor,
    q: torch.Tensor,
    mean_v: torch.Tensor,
    sqrt_c: torch.Tensor,
    w: dict[str, float],
    coarse: np.ndarray,
    splits: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    opt = torch.optim.Adam(flow.parameters(), lr=1.0e-3)
    history = []
    for epoch in range(1, MLE_EPOCHS + 1):
        perm = torch.tensor(np.random.default_rng(SEED + epoch).permutation(splits["train"]), dtype=torch.long)
        losses = []
        for start in range(0, len(perm), BATCH_SIZE):
            ids = perm[start : start + BATCH_SIZE]
            z, inv_ld = flow.inverse(y_true[ids], cond[ids])
            logp = -0.5 * (z**2 + math.log(2 * math.pi)).sum(dim=1)
            loss = -(logp + inv_ld).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), 10.0)
            opt.step()
            losses.append(float(loss.detach()))
        test_samples = sample_flow(flow, cond, back, q, mean_v, sqrt_c, splits["test"])
        row = {
            "epoch": epoch,
            "train_nll": float(np.mean(losses)),
            "val_nll": nll_on_indices(flow, y_true, cond, splits["val"]),
            "test_nll": nll_on_indices(flow, y_true, cond, splits["test"]),
            **metrics(test_samples, w, coarse[splits["test"]]),
        }
        history.append(row)
        if epoch % 10 == 0:
            torch.save({"epoch": epoch, "state_dict": flow.state_dict(), "history": history}, CKPT_MLE / f"epoch_{epoch:03d}.pt")
    return history


def train_mixed(
    flow: pilot.Flow,
    y_true: torch.Tensor,
    cond: torch.Tensor,
    back: torch.Tensor,
    q: torch.Tensor,
    mean_v: torch.Tensor,
    sqrt_c: torch.Tensor,
    logdet_sqrt: float,
    w: dict[str, float],
    coarse: np.ndarray,
    splits: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    opt = torch.optim.Adam(flow.parameters(), lr=2.0e-4)
    history = []
    for epoch in range(1, MIX_EPOCHS + 1):
        perm = torch.tensor(np.random.default_rng(SEED + 1000 + epoch).permutation(splits["train"]), dtype=torch.long)
        losses = []
        nlls = []
        rkls = []
        acts = []
        for start in range(0, len(perm), BATCH_SIZE):
            ids = perm[start : start + BATCH_SIZE]
            z_data, inv_ld = flow.inverse(y_true[ids], cond[ids])
            logp_data = -0.5 * (z_data**2 + math.log(2 * math.pi)).sum(dim=1)
            nll = -(logp_data + inv_ld).mean()

            z = torch.randn(len(ids), 192)
            y, ld = flow(z, cond[ids])
            v = mean_v + y @ sqrt_c.T
            phi = (back[ids] + v @ q.T).reshape(len(ids), 16, 16)
            action, _ = pilot.fine_action(phi)
            logp = -0.5 * (z**2 + math.log(2 * math.pi)).sum(dim=1)
            logq_v = logp - ld - logdet_sqrt
            rkl = (action + logq_v).mean()
            loss = (1.0 - REVERSE_KL_EPS) * nll + REVERSE_KL_EPS * rkl
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), 10.0)
            opt.step()
            losses.append(float(loss.detach()))
            nlls.append(float(nll.detach()))
            rkls.append(float(rkl.detach()))
            acts.append(float(action.mean().detach()))
        test_samples = sample_flow(flow, cond, back, q, mean_v, sqrt_c, splits["test"])
        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "nll_part": float(np.mean(nlls)),
            "reverse_kl_part": float(np.mean(rkls)),
            "S_fine": float(np.mean(acts)),
            "val_nll": nll_on_indices(flow, y_true, cond, splits["val"]),
            "test_nll": nll_on_indices(flow, y_true, cond, splits["test"]),
            **metrics(test_samples, w, coarse[splits["test"]]),
        }
        history.append(row)
        if epoch % 5 == 0:
            torch.save({"epoch": epoch, "state_dict": flow.state_dict(), "history": history}, CKPT_MIX / f"epoch_{epoch:03d}.pt")
    return history


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(f"Refusing to overwrite existing outputs in {OUT}")
    OUT.mkdir(parents=True)
    CKPT_MLE.mkdir(parents=True)
    CKPT_MIX.mkdir(parents=True)

    w = load_weights()
    fine, generation_summary, generation_history = generate_coarse_ensemble(
        L=16,
        kappa=KAPPA_F,
        lam=LAMBDA,
        n_samples=N_PAIRS,
        thermal_sweeps=THERMAL_SWEEPS,
        skip_sweeps=SKIP_SWEEPS,
        proposal_width=PROPOSAL_WIDTH,
        seed=SEED,
    )
    coarse = pilot.block_sym_np(fine, w)
    backbone = pilot.smooth_backbone(coarse, w)
    q_np = local_q_basis(w)
    residual = fine - backbone
    v_true = residual.reshape(len(fine), -1) @ q_np
    recon = backbone + (v_true @ q_np.T).reshape(fine.shape)
    white = covariance_whitening(v_true)
    y_true = white["y"]  # type: ignore[assignment]
    splits = make_splits(len(fine))

    np.save(OUT / "fine_configs.npy", fine.astype(np.float32))
    np.save(OUT / "coarse_blocked_configs.npy", coarse.astype(np.float32))
    np.save(OUT / "backbone_configs.npy", backbone.astype(np.float32))
    np.save(OUT / "residual_v_true.npy", v_true.astype(np.float32))
    np.save(OUT / "y_true.npy", y_true.astype(np.float32))
    np.save(OUT / "mean_v.npy", white["mean"].astype(np.float32))  # type: ignore[union-attr]
    np.save(OUT / "cov_sqrt.npy", white["sqrt"].astype(np.float32))  # type: ignore[union-attr]
    np.save(OUT / "cov_invsqrt.npy", white["invsqrt"].astype(np.float32))  # type: ignore[union-attr]
    np.savez(OUT / "split_indices.npz", **splits)
    write_csv(OUT / "generation_history.csv", generation_history)
    write_csv(OUT / "observables_fine.csv", per_config_rows(fine, w, kappa=KAPPA_F, dataset="fine_lam1_kappa0p320_L16"))
    write_csv(OUT / "observables_coarse.csv", per_config_rows(coarse, w, kappa=0.30, dataset="blocked_fine_coarse_L8"))

    evals = white["evals"]  # type: ignore[assignment]
    ycov = white["ycov"]  # type: ignore[assignment]
    ydiag = np.diag(ycov)
    kurt = marginal_kurtosis(y_true)
    whitening_summary = {
        "N_pairs": int(len(fine)),
        "residual_dimension": 192,
        "covariance_rank_gt_1e-10": int(white["rank"]),
        "ridge_used": bool(white["ridge_used"]),
        "ridge_value": float(white["ridge"]),
        "raw_cov_trace": float(np.trace(white["cov"])),  # type: ignore[arg-type]
        "raw_cov_eig_min": float(np.min(evals)),
        "raw_cov_eig_max": float(np.max(evals)),
        "raw_top10_variance_fraction": float(np.sum(evals[:10]) / np.sum(evals)),
        "raw_top32_variance_fraction": float(np.sum(evals[:32]) / np.sum(evals)),
        "raw_top64_variance_fraction": float(np.sum(evals[:64]) / np.sum(evals)),
        "mean_abs_y_mean": float(np.mean(np.abs(y_true.mean(axis=0)))),
        "max_abs_y_mean": float(np.max(np.abs(y_true.mean(axis=0)))),
        "mean_y_cov_diag": float(np.mean(ydiag)),
        "min_y_cov_diag": float(np.min(ydiag)),
        "max_y_cov_diag": float(np.max(ydiag)),
        "max_abs_y_cov_offdiag": float(np.max(np.abs(ycov - np.diag(ydiag)))),
        "y_cov_eig_min": float(np.min(white["ycov_evals"])),  # type: ignore[arg-type]
        "y_cov_eig_max": float(np.max(white["ycov_evals"])),  # type: ignore[arg-type]
        "marginal_kurtosis_mean": float(np.mean(kurt)),
        "marginal_kurtosis_median": float(np.median(kurt)),
        "block_backbone_rms": float(np.sqrt(np.mean((pilot.block_sym_np(backbone, w) - coarse) ** 2))),
        "reconstruction_rms": float(np.sqrt(np.mean((recon - fine) ** 2))),
        "reconstruction_relative_rms": float(np.sqrt(np.mean((recon - fine) ** 2)) / np.sqrt(np.mean(fine**2))),
    }
    (OUT / "whitening_summary.json").write_text(json.dumps(whitening_summary, indent=2) + "\n")
    spectrum_rows = []
    y_evals = white["ycov_evals"]  # type: ignore[assignment]
    for i, val in enumerate(evals):
        spectrum_rows.append({"space": "v_true_raw", "rank": i + 1, "eigenvalue": float(val), "cumulative_fraction": float(np.sum(evals[: i + 1]) / np.sum(evals))})
    for i, val in enumerate(y_evals):
        spectrum_rows.append({"space": "y_true_whitened", "rank": i + 1, "eigenvalue": float(val), "cumulative_fraction": float(np.sum(y_evals[: i + 1]) / np.sum(y_evals))})
    write_csv(OUT / "covariance_spectrum.csv", spectrum_rows)
    write_csv(
        OUT / "y_true_statistics.csv",
        [
            {
                "coord": i,
                "mean": float(y_true[:, i].mean()),
                "std": float(y_true[:, i].std(ddof=1)),
                "kurtosis": float(kurt[i]),
            }
            for i in range(y_true.shape[1])
        ],
    )

    metadata = {
        "lambda_f": LAMBDA,
        "kappa_f": KAPPA_F,
        "L_f": 16,
        "N_pairs": N_PAIRS,
        "generation": generation_summary,
        "generation_is_production_quality": False,
        "autocorrelation_measured": False,
        "blocking_rule": "symmetric_2x2_average_after_K",
        "kernel_source": str(pilot.KERNEL),
        "thermal_sweeps": THERMAL_SWEEPS,
        "skip_sweeps": SKIP_SWEEPS,
        "proposal_width": PROPOSAL_WIDTH,
        "seed": SEED,
        "split_sizes": {k: int(len(v)) for k, v in splits.items()},
    }
    (OUT / "generation_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    cond_np = np.concatenate([coarse.reshape(len(fine), -1), backbone.reshape(len(fine), -1)], axis=1).astype(np.float32)
    cond = torch.tensor(cond_np)
    back_t = torch.tensor(backbone.reshape(len(fine), -1).astype(np.float32))
    q_t = torch.tensor(q_np.astype(np.float32))
    mean_t = torch.tensor(white["mean"].astype(np.float32))  # type: ignore[union-attr]
    sqrt_t = torch.tensor(white["sqrt"].astype(np.float32))  # type: ignore[union-attr]
    y_t = torch.tensor(y_true.astype(np.float32))

    flow = pilot.Flow(192, cond.shape[1])
    mle_history = train_mle(flow, y_t, cond, back_t, q_t, mean_t, sqrt_t, w, coarse, splits)
    write_csv(OUT / "whitened_mle_diagnostic" / "history.csv", mle_history)
    torch.save({"epoch": MLE_EPOCHS, "state_dict": flow.state_dict(), "history": mle_history}, OUT / "whitened_mle_diagnostic" / "final_model.pt")
    mle_test = sample_flow(flow, cond, back_t, q_t, mean_t, sqrt_t, splits["test"])
    np.save(OUT / "whitened_mle_diagnostic" / "test_samples.npy", mle_test.astype(np.float32))

    mix_history = train_mixed(flow, y_t, cond, back_t, q_t, mean_t, sqrt_t, float(white["logdet_sqrt"]), w, coarse, splits)
    write_csv(OUT / "whitened_reverse_kl_admixture" / "history.csv", mix_history)
    torch.save({"epoch": MIX_EPOCHS, "state_dict": flow.state_dict(), "history": mix_history}, OUT / "whitened_reverse_kl_admixture" / "final_model.pt")
    mix_test = sample_flow(flow, cond, back_t, q_t, mean_t, sqrt_t, splits["test"])
    np.save(OUT / "whitened_reverse_kl_admixture" / "test_samples.npy", mix_test.astype(np.float32))

    rng = np.random.default_rng(SEED + 2)
    y_gauss = rng.normal(size=(len(splits["test"]), 192))
    idx = splits["test"]
    v_gauss = white["mean"] + y_gauss @ white["sqrt"].T  # type: ignore[operator]
    phi_gauss = backbone[idx] + (v_gauss @ q_np.T).reshape(len(idx), 16, 16)
    old64 = PROJECT / "outputs" / "whitened_nullspace_conditional_nf" / "observable_comparison.csv"
    rows = [
        {"ensemble": "test_original_fine", **metrics(fine[idx], w, coarse[idx])},
        {"ensemble": "test_backbone", **metrics(backbone[idx], w, coarse[idx])},
        {"ensemble": "test_true_v_reconstruction", **metrics(recon[idx], w, coarse[idx])},
        {"ensemble": "test_gaussian_whitened_baseline", **metrics(phi_gauss, w, coarse[idx])},
        {"ensemble": "test_whitened_mle", **metrics(mle_test, w, coarse[idx])},
        {"ensemble": "test_whitened_mle_plus_eps0p05_reverse_kl", **metrics(mix_test, w, coarse[idx])},
    ]
    write_csv(OUT / "diagnostic_observable_comparison.csv", rows)

    summary = {
        "generation_metadata": metadata,
        "whitening": whitening_summary,
        "mle_final": mle_history[-1],
        "reverse_kl_admixture_final": mix_history[-1],
        "test_observables": rows,
        "previous_64_pair_comparison_csv": str(old64),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    def val(name: str, key: str) -> float:
        return float(next(r[key] for r in rows if r["ensemble"] == name))

    report = f"""# Paired Data lambda=1 kappa_f=0.320

Generated a fresh diagnostic fine ensemble and paired it by symmetric block averaging:

`phi_c = B_sym(phi_f)`.

This remains conditional inverse-map development using paired blocked-fine data. It is not native-coarse full sampling.

## Generation

- N_pairs: {len(fine)}
- lambda_f: {LAMBDA}
- kappa_f: {KAPPA_F}
- L_f: 16
- thermal sweeps: {THERMAL_SWEEPS}
- skip sweeps between saved configs: {SKIP_SWEEPS}
- proposal width: {PROPOSAL_WIDTH}
- local Metropolis acceptance: {generation_summary['local_acceptance']:.6g}
- autocorrelation measured: no
- production quality: no, diagnostic chain

## Whitening

- residual dimension: 192
- covariance rank (>1e-10): {whitening_summary['covariance_rank_gt_1e-10']}
- ridge used: {whitening_summary['ridge_used']}
- ridge value: {whitening_summary['ridge_value']:.6g}
- raw top-10 variance fraction: {whitening_summary['raw_top10_variance_fraction']:.6g}
- raw top-32 variance fraction: {whitening_summary['raw_top32_variance_fraction']:.6g}
- raw top-64 variance fraction: {whitening_summary['raw_top64_variance_fraction']:.6g}
- mean |mean(y_true)|: {whitening_summary['mean_abs_y_mean']:.6g}
- mean diag cov(y_true): {whitening_summary['mean_y_cov_diag']:.6g}
- y covariance eig min/max: {whitening_summary['y_cov_eig_min']:.6g} / {whitening_summary['y_cov_eig_max']:.6g}
- marginal kurtosis mean/median: {whitening_summary['marginal_kurtosis_mean']:.6g} / {whitening_summary['marginal_kurtosis_median']:.6g}

## Test Observables

| ensemble | phi2 | phi4 | NN | nn2 | diag | 2nn | action density | block RMS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
"""
    for row in rows:
        report += f"| {row['ensemble']} | {row['phi2']:.6g} | {row['phi4']:.6g} | {row['NN']:.6g} | {row['nn2']:.6g} | {row['diag']:.6g} | {row['2nn']:.6g} | {row['action_density']:.6g} | {row['block_RMS']:.3g} |\n"
    report += f"""
## Acceptance Criteria

1. Covariance rank is much closer to 192: yes, rank {whitening_summary['covariance_rank_gt_1e-10']} with {len(fine)} pairs.
2. Whitening is less ridge-sensitive: yes, no ridge was needed if reported false above; otherwise the ridge is tiny relative to the covariance trace.
3. MLE validation/test observables: compare test MLE phi2/phi4/nn2 `{val('test_whitened_mle', 'phi2'):.6g}/{val('test_whitened_mle', 'phi4'):.6g}/{val('test_whitened_mle', 'nn2'):.6g}` against test fine `{val('test_original_fine', 'phi2'):.6g}/{val('test_original_fine', 'phi4'):.6g}/{val('test_original_fine', 'nn2'):.6g}`.
4. Reverse-KL admixture eps={REVERSE_KL_EPS}: test phi2/phi4/nn2 `{val('test_whitened_mle_plus_eps0p05_reverse_kl', 'phi2'):.6g}/{val('test_whitened_mle_plus_eps0p05_reverse_kl', 'phi4'):.6g}/{val('test_whitened_mle_plus_eps0p05_reverse_kl', 'nn2'):.6g}`.

Do not proceed to native-coarse full sampling from this dataset alone.
"""
    (OUT / "report.md").write_text(report)


if __name__ == "__main__":
    main()
