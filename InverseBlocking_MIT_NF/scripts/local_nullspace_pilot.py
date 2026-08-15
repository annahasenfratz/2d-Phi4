#!/usr/bin/env python3
"""Local Haar-style exact-nullspace pilot."""

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


OUT = PROJECT / "outputs" / "local_nullspace_pilot"
RUN = OUT / "tiny_run"
CKPT = RUN / "checkpoints"
BASE_NULL = PROJECT / "outputs" / "nullspace_conditional_nf_pilot"
SEED = 20240625
SIGMA = 0.25


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def haar_detail_basis() -> np.ndarray:
    # Per 2x2 block: h=[1,-1,0,0]/sqrt2, v=[1,0,-1,0]/sqrt2, d=[1,0,0,-1]/sqrt2
    H = np.zeros((256, 192), dtype=np.float64)
    col = 0
    modes = [
        np.array([1.0, -1.0, 0.0, 0.0]) / math.sqrt(2.0),
        np.array([1.0, 0.0, -1.0, 0.0]) / math.sqrt(2.0),
        np.array([1.0, 0.0, 0.0, -1.0]) / math.sqrt(2.0),
    ]
    for iy in range(8):
        for ix in range(8):
            sites = [(2 * iy, 2 * ix), (2 * iy + 1, 2 * ix), (2 * iy, 2 * ix + 1), (2 * iy + 1, 2 * ix + 1)]
            inds = [y * 16 + x for y, x in sites]
            for m in modes:
                H[inds, col] = m
                col += 1
    return H


def metrics(phi: np.ndarray, w: dict[str, float], coarse: np.ndarray) -> dict[str, float]:
    idx = np.arange(len(phi)) % len(coarse)
    br = pilot.block_sym_np(phi.astype(np.float64), w) - coarse[idx]
    return {"block_RMS": float(np.sqrt(np.mean(br**2))), "block_max": float(np.max(np.abs(br))), **pilot.obs_np(phi)}


def generate_with_basis(backbone: np.ndarray, basis: np.ndarray, sigma: float, n: int, rng: np.random.Generator) -> np.ndarray:
    idx = np.arange(n) % len(backbone)
    z = rng.normal(scale=sigma, size=(n, basis.shape[1]))
    return (backbone[idx].reshape(n, -1) + z @ basis.T).reshape(n, 16, 16)


def main() -> None:
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(f"Refusing to overwrite existing outputs in {OUT}")
    OUT.mkdir(parents=True)
    RUN.mkdir()
    CKPT.mkdir()

    meta = json.loads(pilot.KERNEL.read_text())
    w = {k: float(meta["weights"][k]) for k in ["w00", "w10", "w11", "w20", "w21", "w22"]}
    fine = np.load(pilot.BASE / "input_fine_batch.npy").astype(np.float64)
    coarse = pilot.block_sym_np(fine, w)
    backbone = pilot.smooth_backbone(coarse, w)
    B = pilot.build_B(w)
    BBt_inv = np.linalg.inv(B @ B.T)
    P = np.eye(256) - B.T @ BBt_inv @ B
    H = haar_detail_basis()
    M = P @ H
    rank_M = int(np.linalg.matrix_rank(M, tol=1e-10))
    gram = M.T @ M
    eig = np.linalg.eigvalsh(gram)
    Q, R = np.linalg.qr(M, mode="reduced")
    N_dense = np.load(BASE_NULL / "preflight" / "null_basis.npy").astype(np.float64)
    # Align signs irrelevant; compare locality spread as participation ratio.
    def spread(A: np.ndarray) -> float:
        cols = A.reshape(16, 16, A.shape[1])
        coords_y, coords_x = np.meshgrid(np.arange(16), np.arange(16), indexing="ij")
        vals = []
        for j in range(A.shape[1]):
            p = cols[:, :, j] ** 2
            p = p / p.sum()
            my, mx = float((p * coords_y).sum()), float((p * coords_x).sum())
            vals.append(float((p * ((coords_y - my) ** 2 + (coords_x - mx) ** 2)).sum()))
        return float(np.mean(vals))
    pre = {
        "H_shape": list(H.shape),
        "M_shape": list(M.shape),
        "rank_M": rank_M,
        "cond_MtM": float(np.linalg.cond(gram)),
        "min_eig_MtM": float(eig[0]),
        "max_eig_MtM": float(eig[-1]),
        "max_abs_BM": float(np.max(np.abs(B @ M))),
        "rms_BM": float(np.sqrt(np.mean((B @ M) ** 2))),
        "max_abs_BQ": float(np.max(np.abs(B @ Q))),
        "orthonormal_Q_error": float(np.max(np.abs(Q.T @ Q - np.eye(192)))),
        "local_projected_spread": spread(Q),
        "dense_svd_spread": spread(N_dense),
        "full_rank_stable": bool(rank_M == 192 and np.linalg.cond(gram) < 1e8),
    }
    (OUT / "basis_preflight.json").write_text(json.dumps(pre, indent=2) + "\n")
    (OUT / "basis_preflight.md").write_text(
        "# Local Nullspace Basis Preflight\n\n"
        f"- rank(M): {rank_M}\n- cond(M^T M): {pre['cond_MtM']:.6g}\n- max |B M|: {pre['max_abs_BM']:.3g}\n- max |B Q|: {pre['max_abs_BQ']:.3g}\n- Q orthonormal error: {pre['orthonormal_Q_error']:.3g}\n- local projected spread: {pre['local_projected_spread']:.6g}\n- dense SVD spread: {pre['dense_svd_spread']:.6g}\n"
    )
    np.save(OUT / "local_projected_Q_basis.npy", Q.astype(np.float32))
    rows = []
    for name, basis in {"dense_svd_N": N_dense, "local_projected_Q": Q}.items():
        ph = generate_with_basis(backbone, basis, SIGMA, len(fine), rng)
        rows.append({"basis": name, "sigma": SIGMA, **metrics(ph, w, coarse)})
    rows.append({"basis": "original_fine", "sigma": math.nan, **metrics(fine, w, coarse)})
    rows.append({"basis": "backbone", "sigma": math.nan, **metrics(backbone, w, coarse)})
    write_csv(OUT / "gaussian_basis_comparison.csv", rows)
    if not pre["full_rank_stable"]:
        (RUN / "report.md").write_text("Basis preflight failed; training skipped.\n")
        return

    cond_np = np.concatenate([coarse.reshape(len(fine), -1), backbone.reshape(len(fine), -1)], axis=1).astype(np.float32)
    cond = torch.tensor(cond_np, dtype=torch.float32)
    back = torch.tensor(backbone.reshape(len(fine), -1).astype(np.float32))
    Qt = torch.tensor(Q.astype(np.float32))
    flow = pilot.Flow(192, cond.shape[1])
    opt = torch.optim.Adam(flow.parameters(), lr=1e-3)
    history = []
    for epoch in range(1, 51):
        perm = torch.randperm(len(fine))
        losses = []; acts = []; logqs = []
        for start in range(0, len(fine), 16):
            ids = perm[start:start+16]
            z = torch.randn(len(ids), 192)
            v, ld = flow(z, cond[ids])
            ph = (back[ids] + SIGMA * (v @ Qt.T)).reshape(len(ids), 16, 16)
            S, _ = pilot.fine_action(ph)
            logp = -0.5 * (z**2 + math.log(2 * math.pi)).sum(dim=1)
            logq = logp - ld
            loss = (S + logq).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), 10.0)
            opt.step()
            losses.append(float(loss.detach())); acts.append(float(S.mean().detach())); logqs.append(float(logq.mean().detach()))
        with torch.no_grad():
            z = torch.randn(len(fine), 192)
            v, ld = flow(z, cond)
            samples = (back + SIGMA * (v @ Qt.T)).reshape(len(fine), 16, 16).numpy()
        row = {"epoch": epoch, "loss": float(np.mean(losses)), "S_fine": float(np.mean(acts)), "logq": float(np.mean(logqs)), "ESS_over_N": math.nan, "nan_or_inf": bool(not np.isfinite(samples).all()), **metrics(samples, w, coarse)}
        history.append(row)
        if epoch % 5 == 0:
            torch.save({"epoch": epoch, "state_dict": flow.state_dict(), "history": history, "basis": "local_projected_Q", "sigma": SIGMA}, CKPT / f"epoch_{epoch:03d}.pt")
    write_csv(RUN / "history.csv", history)
    np.save(RUN / "generated_final_samples.npy", samples)
    comp = []
    for name, arr in {
        "local_nullspace_nf": samples,
        "original_fine": fine,
        "backbone": backbone,
        "dense_epoch50": np.load(BASE_NULL / "tiny_run" / "generated_final_samples.npy"),
        "dense_continued": np.load(BASE_NULL / "continued_run" / "generated_final_samples.npy"),
    }.items():
        comp.append({"ensemble": name, **metrics(arr.astype(np.float64), w, coarse)})
    write_csv(RUN / "sample_observables.csv", comp)
    final = history[-1]
    (RUN / "summary.json").write_text(json.dumps({"basis_preflight": pre, "final_epoch": final, "comparisons": comp}, indent=2) + "\n")
    def val(name, key):
        return next(r[key] for r in comp if r["ensemble"] == name)
    (RUN / "report.md").write_text(
        "# Local Nullspace Pilot\n\n"
        f"Basis rank {rank_M}, cond(M^T M) {pre['cond_MtM']:.6g}; max |B Q| {pre['max_abs_BQ']:.3g}.\n\n"
        f"Final local NF phi2/phi4/nn2: {val('local_nullspace_nf','phi2'):.6g}, {val('local_nullspace_nf','phi4'):.6g}, {val('local_nullspace_nf','nn2'):.6g}; block RMS {val('local_nullspace_nf','block_RMS'):.3g}.\n\n"
        f"Original fine phi2/phi4/nn2: {val('original_fine','phi2'):.6g}, {val('original_fine','phi4'):.6g}, {val('original_fine','nn2'):.6g}.\n\n"
        f"Dense epoch50 phi2/phi4/nn2: {val('dense_epoch50','phi2'):.6g}, {val('dense_epoch50','phi4'):.6g}, {val('dense_epoch50','nn2'):.6g}.\n\n"
        "This tests locality-aware projected Haar coordinates with exact block consistency. Do not treat it as a valid sampler.\n"
    )


if __name__ == "__main__":
    main()
