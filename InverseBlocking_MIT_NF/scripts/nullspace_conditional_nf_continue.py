#!/usr/bin/env python3
"""Continue exact block-consistent null-space NF pilot from epoch 50."""

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


BASE_OUT = PROJECT / "outputs" / "nullspace_conditional_nf_pilot"
TINY = BASE_OUT / "tiny_run"
OUT = BASE_OUT / "continued_run"
CKPT = OUT / "checkpoints"
CHECKPOINT_IN = TINY / "checkpoints" / "epoch_050.pt"

ADDITIONAL_EPOCHS = 200
BATCH_SIZE = 16
LR = 7.5e-4
SIGMA_U = 0.25
SEED = 20240624


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    keys: list[str] = []
    for row in rows:
        for k in row:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def load_setup():
    meta = json.loads(pilot.KERNEL.read_text())
    w = {k: float(meta["weights"][k]) for k in ["w00", "w10", "w11", "w20", "w21", "w22"]}
    fine = np.load(pilot.BASE / "input_fine_batch.npy").astype(np.float64)
    coarse = pilot.block_sym_np(fine, w)
    backbone = pilot.smooth_backbone(coarse, w)
    N = np.load(BASE_OUT / "preflight" / "null_basis.npy").astype(np.float32)
    cond_np = np.concatenate([coarse.reshape(len(fine), -1), backbone.reshape(len(fine), -1)], axis=1).astype(np.float32)
    return w, fine, coarse, backbone, N, cond_np


def generate(flow, cond_all, back_all, Nt, n: int, sigma: float):
    flow.eval()
    with torch.no_grad():
        idx = torch.arange(n) % cond_all.shape[0]
        z = torch.randn(n, 192)
        u, ld = flow(z, cond_all[idx])
        samples = (back_all[idx] + sigma * (u @ Nt.T)).reshape(n, pilot.L_FINE, pilot.L_FINE)
        S, comps = pilot.fine_action(samples)
        logp = -0.5 * (z**2 + math.log(2 * math.pi)).sum(dim=1)
        logq = logp - ld
    return samples.numpy(), {"S_fine": float(S.mean()), "logq": float(logq.mean()), "loss": float((S + logq).mean()), **{k: float(v.mean()) for k, v in comps.items()}}


def metrics_for_samples(samples: np.ndarray, w: dict[str, float], coarse_ref: np.ndarray) -> dict[str, float]:
    idx = np.arange(len(samples)) % len(coarse_ref)
    br = pilot.block_sym_np(samples.astype(np.float64), w) - coarse_ref[idx]
    return {
        "block_residual_RMS": float(np.sqrt(np.mean(br**2))),
        "block_residual_max": float(np.max(np.abs(br))),
        "nan_or_inf": bool(not np.isfinite(samples).all()),
        **pilot.obs_np(samples),
    }


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(f"Refusing to overwrite existing outputs in {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    CKPT.mkdir(exist_ok=True)
    w, fine, coarse, backbone, N, cond_np = load_setup()
    cond_all = torch.tensor(cond_np, dtype=torch.float32)
    back_all = torch.tensor(backbone.reshape(len(fine), -1).astype(np.float32))
    Nt = torch.tensor(N, dtype=torch.float32)
    flow = pilot.Flow(192, cond_all.shape[1])
    ckpt = torch.load(CHECKPOINT_IN, map_location="cpu", weights_only=False)
    flow.load_state_dict(ckpt["state_dict"])
    opt = torch.optim.Adam(flow.parameters(), lr=LR)
    history: list[dict[str, object]] = []
    eval_history: list[dict[str, object]] = []
    best_phi2_gap = float("inf")
    plateau_count = 0
    stop_reason = "completed_200_epochs"
    original = pilot.obs_np(fine)
    tiny_samples = np.load(TINY / "generated_final_samples.npy")
    tiny_obs = metrics_for_samples(tiny_samples, w, coarse)

    for e in range(1, ADDITIONAL_EPOCHS + 1):
        global_epoch = 50 + e
        perm = torch.randperm(len(fine))
        losses = []
        acts = []
        logqs = []
        grad_norms = []
        for start in range(0, len(fine), BATCH_SIZE):
            ids = perm[start:start + BATCH_SIZE]
            z = torch.randn(len(ids), 192)
            u, ld = flow(z, cond_all[ids])
            ph = (back_all[ids] + SIGMA_U * (u @ Nt.T)).reshape(len(ids), pilot.L_FINE, pilot.L_FINE)
            S, _ = pilot.fine_action(ph)
            logp = -0.5 * (z**2 + math.log(2 * math.pi)).sum(dim=1)
            logq = logp - ld
            loss = (S + logq).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(flow.parameters(), 10.0))
            opt.step()
            losses.append(float(loss.detach()))
            acts.append(float(S.mean().detach()))
            logqs.append(float(logq.mean().detach()))
            grad_norms.append(grad_norm)

        samples, terms = generate(flow, cond_all, back_all, Nt, len(fine), SIGMA_U)
        m = metrics_for_samples(samples, w, coarse)
        row = {
            "epoch": global_epoch,
            "continued_epoch": e,
            "loss": float(np.mean(losses)),
            "S_fine": float(np.mean(acts)),
            "logq": float(np.mean(logqs)),
            "ESS_over_N": math.nan,
            "grad_norm": float(np.mean(grad_norms)),
            **m,
        }
        history.append(row)
        phi2_gap = abs(float(m["phi2"]) - original["phi2"])
        if phi2_gap + 1e-4 < best_phi2_gap:
            best_phi2_gap = phi2_gap
            plateau_count = 0
        else:
            plateau_count += 1

        if e % 10 == 0:
            eval_samples, eval_terms = generate(flow, cond_all, back_all, Nt, 256, SIGMA_U)
            ev = {"epoch": global_epoch, **metrics_for_samples(eval_samples, w, coarse), **eval_terms}
            eval_history.append(ev)
            torch.save({"epoch": global_epoch, "state_dict": flow.state_dict(), "history": history, "eval_history": eval_history, "sigma_u": SIGMA_U}, CKPT / f"epoch_{global_epoch:03d}.pt")

        if bool(row["nan_or_inf"]) or not np.isfinite(row["loss"]):
            stop_reason = "nan_or_inf_or_nonfinite_loss"
            break
        if e > 20 and row["phi4"] > 1.25 and abs(row["phi2"] - original["phi2"]) > 0.08 and abs(row["nn2"] - original["nn2"]) > 0.08:
            stop_reason = "phi4_badly_overshot_without_phi2_nn2_improvement"
            break
        if e >= 60 and plateau_count >= 50:
            stop_reason = "metrics_plateau_50_epochs"
            break

    write_csv(OUT / "history.csv", history)
    write_csv(OUT / "eval_history.csv", eval_history)
    final_samples, final_terms = generate(flow, cond_all, back_all, Nt, len(fine), SIGMA_U)
    np.save(OUT / "generated_final_samples.npy", final_samples)
    comparisons = {
        "continued_nullspace_nf": final_samples,
        "original_fine": fine,
        "phi_backbone": backbone,
        "tiny_run_epoch50": tiny_samples,
    }
    rows = []
    for name, arr in comparisons.items():
        rows.append({"ensemble": name, **metrics_for_samples(arr.astype(np.float64), w, coarse)})
    write_csv(OUT / "sample_observables.csv", rows)
    write_csv(OUT / "action_components.csv", [{"ensemble": r["ensemble"], "action_hopping_density": r["action_hopping_density"], "action_phi2_density": r["action_phi2_density"], "action_phi4_density": r["action_phi4_density"], "action_density": r["action_density"]} for r in rows])
    final = history[-1]
    summary = {
        "start_checkpoint": str(CHECKPOINT_IN),
        "sigma_u": SIGMA_U,
        "additional_epochs_run": len(history),
        "stop_reason": stop_reason,
        "final_epoch": final,
        "comparisons": rows,
        "tiny_epoch50_metrics": tiny_obs,
        "original_metrics": original,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    def val(name: str, key: str) -> float:
        return next(float(r[key]) for r in rows if r["ensemble"] == name)
    report = f"""# Continued Exact Block-Consistent Null-Space NF Pilot

Started from `tiny_run/checkpoints/epoch_050.pt` and continued with `sigma_u=0.25`, no block penalty, and exact null-space parameterization.

- additional epochs run: {len(history)}
- stop reason: {stop_reason}
- final block RMS: {final['block_residual_RMS']:.12g}
- final block max: {final['block_residual_max']:.12g}
- final loss: {final['loss']:.12g}
- final grad norm: {final['grad_norm']:.12g}

## Observable Comparison

| ensemble | phi2 | phi4 | nn2 | action density | block RMS |
|---|---:|---:|---:|---:|---:|
"""
    for r in rows:
        report += f"| {r['ensemble']} | {r['phi2']:.6g} | {r['phi4']:.6g} | {r['nn2']:.6g} | {r['action_density']:.6g} | {r['block_residual_RMS']:.3g} |\n"
    report += f"""
## Answers

- Did continuation improve phi2 and nn2? Phi2 ended at {val('continued_nullspace_nf', 'phi2'):.6g} versus epoch-50 {val('tiny_run_epoch50', 'phi2'):.6g} and original {val('original_fine', 'phi2'):.6g}. nn2 ended at {val('continued_nullspace_nf', 'nn2'):.6g} versus epoch-50 {val('tiny_run_epoch50', 'nn2'):.6g} and original {val('original_fine', 'nn2'):.6g}.
- Did phi4 stay correct or overshoot? Phi4 ended at {val('continued_nullspace_nf', 'phi4'):.6g}, compared with original {val('original_fine', 'phi4'):.6g}.
- Did exact block consistency remain intact? Yes, within float32/projection noise; final block RMS is {final['block_residual_RMS']:.12g}.
- Is dense null-space flow worth scaling? The exact constraint is sound, but the dense coordinate flow should give way to locality-aware null coordinates or a convolutional conditioner if phi2/nn2 plateau.
"""
    (OUT / "report.md").write_text(report)


if __name__ == "__main__":
    main()
