#!/usr/bin/env python3
"""Whitened exact-nullspace conditional NF diagnostic on paired blocked-fine data."""

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
import nullspace_conditional_nf_pilot as pilot  # type: ignore
import local_nullspace_pilot as local_pilot  # type: ignore


OUT = PROJECT / "outputs" / "whitened_nullspace_conditional_nf"
PREFLIGHT = OUT / "preflight"
MLE = OUT / "mle_pretrain"
RK = OUT / "reverse_kl_finetune"
SEED = 20240628
BATCH_SIZE = 16
MLE_EPOCHS = 100
RK_EPOCHS = 50
RIDGE_EPS = 1.0e-3


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


def metrics(phi: np.ndarray, w: dict[str, float], coarse: np.ndarray) -> dict[str, float]:
    br = pilot.block_sym_np(phi.astype(np.float64), w) - coarse[: len(phi)]
    return {
        "block_RMS": float(np.sqrt(np.mean(br**2))),
        "block_max": float(np.max(np.abs(br))),
        **pilot.obs_np(phi.astype(np.float64)),
    }


def covariance_whitening(v: np.ndarray) -> dict[str, np.ndarray | float | int]:
    mean = v.mean(axis=0)
    centered = v - mean
    cov = np.cov(centered, rowvar=False)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]
    rank = int(np.sum(evals > 1.0e-10))
    ridge = RIDGE_EPS * float(np.trace(cov)) / cov.shape[0]
    evals_reg = np.maximum(evals, 0.0) + ridge
    sqrt = (evecs * np.sqrt(evals_reg)) @ evecs.T
    invsqrt = (evecs * (1.0 / np.sqrt(evals_reg))) @ evecs.T
    y = centered @ invsqrt.T
    ycov = np.cov(y, rowvar=False)
    yc_evals = np.linalg.eigvalsh(ycov)
    return {
        "mean": mean,
        "cov": cov,
        "evals": evals,
        "rank": rank,
        "ridge": ridge,
        "sqrt": sqrt,
        "invsqrt": invsqrt,
        "y": y,
        "ycov": ycov,
        "yc_evals": yc_evals,
        "logdet_sqrt": float(np.sum(np.log(np.sqrt(evals_reg)))),
    }


def marginal_kurtosis(x: np.ndarray) -> np.ndarray:
    centered = x - x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, ddof=1)
    return np.mean(centered**4, axis=0) / np.maximum(std, 1.0e-30) ** 4


def sample_flow(flow: pilot.Flow, cond: torch.Tensor, back: torch.Tensor, q: torch.Tensor, mean_v: torch.Tensor, sqrt_c: torch.Tensor, n: int) -> tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        z = torch.randn(n, 192)
        y, _ = flow(z, cond[:n])
        v = mean_v + y @ sqrt_c.T
        phi = (back[:n] + v @ q.T).reshape(n, 16, 16)
    return phi.cpu().numpy(), y.cpu().numpy()


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
) -> list[dict[str, object]]:
    opt = torch.optim.Adam(flow.parameters(), lr=1.0e-3)
    history: list[dict[str, object]] = []
    ckpt = MLE / "checkpoints"
    ckpt.mkdir(parents=True)
    for epoch in range(1, MLE_EPOCHS + 1):
        perm = torch.randperm(len(y_true))
        losses = []
        for start in range(0, len(y_true), BATCH_SIZE):
            ids = perm[start : start + BATCH_SIZE]
            z, inv_ld = flow.inverse(y_true[ids], cond[ids])
            logp = -0.5 * (z**2 + math.log(2 * math.pi)).sum(dim=1)
            logq = logp + inv_ld
            loss = -logq.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), 10.0)
            opt.step()
            losses.append(float(loss.detach()))
        samples, _ = sample_flow(flow, cond, back, q, mean_v, sqrt_c, len(y_true))
        row = {"epoch": epoch, "nll": float(np.mean(losses)), **metrics(samples, w, coarse)}
        history.append(row)
        if epoch % 10 == 0:
            torch.save({"epoch": epoch, "state_dict": flow.state_dict(), "history": history}, ckpt / f"epoch_{epoch:03d}.pt")
    return history


def train_reverse_kl(
    flow: pilot.Flow,
    cond: torch.Tensor,
    back: torch.Tensor,
    q: torch.Tensor,
    mean_v: torch.Tensor,
    sqrt_c: torch.Tensor,
    logdet_sqrt: float,
    w: dict[str, float],
    coarse: np.ndarray,
) -> list[dict[str, object]]:
    opt = torch.optim.Adam(flow.parameters(), lr=5.0e-4)
    history: list[dict[str, object]] = []
    ckpt = RK / "checkpoints"
    ckpt.mkdir(parents=True)
    for epoch in range(1, RK_EPOCHS + 1):
        perm = torch.randperm(len(cond))
        losses = []
        acts = []
        logqs = []
        for start in range(0, len(cond), BATCH_SIZE):
            ids = perm[start : start + BATCH_SIZE]
            z = torch.randn(len(ids), 192)
            y, ld = flow(z, cond[ids])
            v = mean_v + y @ sqrt_c.T
            phi = (back[ids] + v @ q.T).reshape(len(ids), 16, 16)
            action, _ = pilot.fine_action(phi)
            logp = -0.5 * (z**2 + math.log(2 * math.pi)).sum(dim=1)
            logq_y = logp - ld
            logq_v = logq_y - logdet_sqrt
            loss = (action + logq_v).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), 10.0)
            opt.step()
            losses.append(float(loss.detach()))
            acts.append(float(action.mean().detach()))
            logqs.append(float(logq_v.mean().detach()))
        samples, _ = sample_flow(flow, cond, back, q, mean_v, sqrt_c, len(cond))
        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "S_fine": float(np.mean(acts)),
            "logq": float(np.mean(logqs)),
            **metrics(samples, w, coarse),
        }
        history.append(row)
        if epoch % 10 == 0:
            torch.save({"epoch": epoch, "state_dict": flow.state_dict(), "history": history}, ckpt / f"epoch_{epoch:03d}.pt")
    return history


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(f"Refusing to overwrite existing outputs in {OUT}")
    PREFLIGHT.mkdir(parents=True)
    MLE.mkdir()
    RK.mkdir()

    w = load_weights()
    fine = np.load(pilot.BASE / "input_fine_batch.npy").astype(np.float64)
    coarse = pilot.block_sym_np(fine, w)
    backbone = pilot.smooth_backbone(coarse, w)
    q_np = local_q_basis(w)
    residual = fine - backbone
    v_true = residual.reshape(len(fine), -1) @ q_np
    white = covariance_whitening(v_true)
    y_true = white["y"]  # type: ignore[assignment]
    mean_v = white["mean"]  # type: ignore[assignment]
    sqrt_c = white["sqrt"]  # type: ignore[assignment]
    invsqrt = white["invsqrt"]  # type: ignore[assignment]

    ycov = white["ycov"]  # type: ignore[assignment]
    y_diag = np.diag(ycov)
    pre = {
        "n_configs": int(len(fine)),
        "dimension": 192,
        "raw_cov_rank_gt_1e-10": int(white["rank"]),
        "raw_cov_trace": float(np.trace(white["cov"])),  # type: ignore[arg-type]
        "raw_cov_ridge": float(white["ridge"]),
        "raw_cov_eig_max": float(np.max(white["evals"])),  # type: ignore[arg-type]
        "raw_cov_eig_min": float(np.min(white["evals"])),  # type: ignore[arg-type]
        "raw_cov_top10_fraction": float(np.sum(white["evals"][:10]) / np.sum(white["evals"])),  # type: ignore[index]
        "raw_cov_top32_fraction": float(np.sum(white["evals"][:32]) / np.sum(white["evals"])),  # type: ignore[index]
        "mean_abs_y": float(np.mean(np.abs(y_true.mean(axis=0)))),
        "max_abs_y_mean": float(np.max(np.abs(y_true.mean(axis=0)))),
        "mean_y_cov_diag": float(np.mean(y_diag)),
        "min_y_cov_diag": float(np.min(y_diag)),
        "max_y_cov_diag": float(np.max(y_diag)),
        "max_abs_y_cov_offdiag": float(np.max(np.abs(ycov - np.diag(y_diag)))),
        "y_cov_eig_min": float(np.min(white["yc_evals"])),  # type: ignore[arg-type]
        "y_cov_eig_max": float(np.max(white["yc_evals"])),  # type: ignore[arg-type]
        "y_marginal_kurtosis_mean": float(np.mean(marginal_kurtosis(y_true))),
        "y_marginal_kurtosis_median": float(np.median(marginal_kurtosis(y_true))),
        "logdet_Csqrt_regularized": float(white["logdet_sqrt"]),
        "block_residual_backbone_rms": float(np.sqrt(np.mean((pilot.block_sym_np(backbone, w) - coarse) ** 2))),
        "block_residual_true_reconstruction_rms": float(np.sqrt(np.mean((pilot.block_sym_np(backbone + (v_true @ q_np.T).reshape(fine.shape), w) - coarse) ** 2))),
    }
    (PREFLIGHT / "whitening_summary.json").write_text(json.dumps(pre, indent=2) + "\n")
    np.save(PREFLIGHT / "mean_v.npy", mean_v.astype(np.float32))
    np.save(PREFLIGHT / "cov_sqrt_regularized.npy", sqrt_c.astype(np.float32))
    np.save(PREFLIGHT / "cov_invsqrt_regularized.npy", invsqrt.astype(np.float32))
    np.save(PREFLIGHT / "y_true.npy", y_true.astype(np.float32))

    spectrum_rows = []
    raw_evals = white["evals"]  # type: ignore[assignment]
    y_evals = np.linalg.eigvalsh(ycov)[::-1]
    for i, val in enumerate(raw_evals):
        spectrum_rows.append({"space": "v_true_raw", "rank": i + 1, "eigenvalue": float(val), "cumulative_fraction": float(np.sum(raw_evals[: i + 1]) / np.sum(raw_evals))})
    for i, val in enumerate(y_evals):
        spectrum_rows.append({"space": "y_true_whitened_regularized", "rank": i + 1, "eigenvalue": float(val), "cumulative_fraction": float(np.sum(y_evals[: i + 1]) / np.sum(y_evals))})
    write_csv(PREFLIGHT / "covariance_spectrum.csv", spectrum_rows)

    cond_np = np.concatenate([coarse.reshape(len(fine), -1), backbone.reshape(len(fine), -1)], axis=1).astype(np.float32)
    cond = torch.tensor(cond_np)
    back_t = torch.tensor(backbone.reshape(len(fine), -1).astype(np.float32))
    q_t = torch.tensor(q_np.astype(np.float32))
    mean_t = torch.tensor(mean_v.astype(np.float32))
    sqrt_t = torch.tensor(sqrt_c.astype(np.float32))
    y_t = torch.tensor(y_true.astype(np.float32))

    rng = np.random.default_rng(SEED)
    y_gauss = rng.normal(size=y_true.shape)
    v_gauss = mean_v + y_gauss @ sqrt_c.T
    phi_gauss = backbone + (v_gauss @ q_np.T).reshape(fine.shape)

    flow = pilot.Flow(192, cond.shape[1])
    mle_history = train_mle(flow, y_t, cond, back_t, q_t, mean_t, sqrt_t, w, coarse)
    write_csv(MLE / "history.csv", mle_history)
    mle_samples, mle_y = sample_flow(flow, cond, back_t, q_t, mean_t, sqrt_t, len(fine))
    np.save(MLE / "generated_final_samples.npy", mle_samples)
    np.save(MLE / "generated_final_y.npy", mle_y)
    torch.save({"epoch": MLE_EPOCHS, "state_dict": flow.state_dict(), "history": mle_history}, MLE / "final_model.pt")

    rk_history = train_reverse_kl(flow, cond, back_t, q_t, mean_t, sqrt_t, float(white["logdet_sqrt"]), w, coarse)
    write_csv(RK / "history.csv", rk_history)
    rk_samples, rk_y = sample_flow(flow, cond, back_t, q_t, mean_t, sqrt_t, len(fine))
    np.save(RK / "generated_final_samples.npy", rk_samples)
    np.save(RK / "generated_final_y.npy", rk_y)
    torch.save({"epoch": RK_EPOCHS, "state_dict": flow.state_dict(), "history": rk_history}, RK / "final_model.pt")

    comparisons: dict[str, np.ndarray] = {
        "original_fine": fine,
        "backbone": backbone,
        "true_v_reconstruction": backbone + (v_true @ q_np.T).reshape(fine.shape),
        "gaussian_whitened_baseline": phi_gauss,
        "whitened_mle_pretrain": mle_samples,
        "whitened_reverse_kl_finetune": rk_samples,
        "local_Q_reverse_KL_epoch50": np.load(PROJECT / "outputs" / "local_nullspace_pilot" / "tiny_run" / "generated_final_samples.npy"),
        "dense_null_reverse_KL_epoch50": np.load(PROJECT / "outputs" / "nullspace_conditional_nf_pilot" / "tiny_run" / "generated_final_samples.npy"),
        "unwhitened_MLE_pretrain": np.load(PROJECT / "outputs" / "nullspace_true_residual_diagnosis" / "mle_pretrain" / "generated_final_samples.npy"),
    }
    obs_rows = [{"ensemble": name, **metrics(arr.astype(np.float64), w, coarse)} for name, arr in comparisons.items()]
    write_csv(OUT / "observable_comparison.csv", obs_rows)
    write_csv(MLE / "sample_observables.csv", [{"ensemble": "whitened_mle_pretrain", **metrics(mle_samples, w, coarse)}])
    write_csv(RK / "sample_observables.csv", [{"ensemble": "whitened_reverse_kl_finetune", **metrics(rk_samples, w, coarse)}])

    summary = {
        "whitening": pre,
        "mle_final": mle_history[-1],
        "reverse_kl_final": rk_history[-1],
        "comparisons": obs_rows,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    def val(name: str, key: str) -> float:
        return float(next(r[key] for r in obs_rows if r["ensemble"] == name))

    report = f"""# Whitened Nullspace Conditional NF

This run uses paired blocked-fine data only:

`phi_c = B_sym(phi_f)`, `phi_back` with `B_sym(phi_back)=phi_c`, and `v_true = Q^T(phi_f - phi_back)`.

The native coarse-action distribution is not used.

## Whitening

- configurations: {len(fine)}
- coordinate dimension: 192
- empirical covariance rank (>1e-10): {pre['raw_cov_rank_gt_1e-10']}
- ridge: {pre['raw_cov_ridge']:.6g}
- raw top-10 variance fraction: {pre['raw_cov_top10_fraction']:.6g}
- raw top-32 variance fraction: {pre['raw_cov_top32_fraction']:.6g}
- mean |mean(y)|: {pre['mean_abs_y']:.6g}
- mean diag cov(y): {pre['mean_y_cov_diag']:.6g}
- min/max diag cov(y): {pre['min_y_cov_diag']:.6g} / {pre['max_y_cov_diag']:.6g}
- y marginal kurtosis mean/median: {pre['y_marginal_kurtosis_mean']:.6g} / {pre['y_marginal_kurtosis_median']:.6g}

Because only 64 paired configurations are available for 192 dimensions, whitening uses a trace-scaled ridge and cannot make all 192 empirical covariance eigenvalues exactly one.

## Observables

| ensemble | phi2 | phi4 | NN | nn2 | diag | 2nn | action density | block RMS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
"""
    for row in obs_rows:
        report += f"| {row['ensemble']} | {row['phi2']:.6g} | {row['phi4']:.6g} | {row['NN']:.6g} | {row['nn2']:.6g} | {row['diag']:.6g} | {row['2nn']:.6g} | {row['action_density']:.6g} | {row['block_RMS']:.3g} |\n"
    report += f"""
## Interpretation

1. The whitened Gaussian baseline is the direct test of whether a covariance-aware base distribution fixes the missing residual amplitude.
2. MLE pretraining in whitened y-space tests architecture expressivity against paired residual coordinates.
3. The reverse-KL fine-tune starts from the MLE-pretrained model and keeps exact block consistency automatically.
4. This is still conditional inverse-map development, not a full sampler, because the standalone coarse law `p_c(phi_c)` is not calibrated.

Final whitened reverse-KL phi2/phi4/nn2: {val('whitened_reverse_kl_finetune', 'phi2'):.6g}, {val('whitened_reverse_kl_finetune', 'phi4'):.6g}, {val('whitened_reverse_kl_finetune', 'nn2'):.6g}.
Original fine phi2/phi4/nn2: {val('original_fine', 'phi2'):.6g}, {val('original_fine', 'phi4'):.6g}, {val('original_fine', 'nn2'):.6g}.
"""
    (OUT / "report.md").write_text(report)


if __name__ == "__main__":
    main()
