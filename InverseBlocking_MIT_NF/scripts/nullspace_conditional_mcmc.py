#!/usr/bin/env python3
"""Exact block-consistent conditional MCMC in the null-space coordinates."""

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
OUT = PROJECT / "outputs" / "nullspace_conditional_mcmc"
SEED = 20240624
N_CONDITIONS = 16
SWEEPS = 1000
BURNIN = 200
SAVE_EVERY = 20
STEP_SIZES = [0.02, 0.05, 0.1, 0.2]
PROPOSALS = ["global", "chunked"]
INIT_TYPES = ["backbone", "fixed_pca", "true_residual"]
CHUNK_SIZE = 16


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


def sample_fixed_pca_v(
    rng: np.random.Generator,
    mean_v: np.ndarray,
    evals: np.ndarray,
    evecs: np.ndarray,
    n: int,
    k: int = 32,
    scale: float = 0.9,
    eps_tail: float = 0.1,
) -> np.ndarray:
    sigmas = np.empty(192, dtype=np.float64)
    sigmas[:k] = scale * np.sqrt(np.maximum(evals[:k], 1.0e-12))
    tail_floor = np.sqrt(max(float(np.mean(evals[k:])), float(evals[k - 1]), 1.0e-12))
    sigmas[k:] = eps_tail * max(tail_floor, 1.0e-12)
    coords = mean_v + rng.normal(size=(n, 192)) * sigmas
    return coords


def phi_from_v(v: np.ndarray, back: np.ndarray, q: np.ndarray) -> np.ndarray:
    detail = v @ q.T
    return back + detail.reshape(len(v), 16, 16)


def action_totals(phi: np.ndarray) -> np.ndarray:
    vals = []
    with torch.no_grad():
        for start in range(0, len(phi), 64):
            batch = torch.tensor(phi[start : start + 64].astype(np.float32))
            S, _ = pilot.fine_action(batch)
            vals.append(S.cpu().numpy())
    return np.concatenate(vals)


def sweep_observables(phi: np.ndarray, coarse: np.ndarray, w: dict[str, float]) -> dict[str, float]:
    obs = ensemble_metrics(phi, coarse, w)
    obs["S_total_mean"] = float(np.mean(action_totals(phi)))
    return obs


def chunk_masks(rng: np.random.Generator, n: int, dim: int = 192, chunk_size: int = CHUNK_SIZE) -> np.ndarray:
    masks = np.zeros((n, dim), dtype=np.float64)
    n_chunks = dim // chunk_size
    choices = rng.integers(0, n_chunks, size=n)
    for i, c in enumerate(choices):
        masks[i, c * chunk_size : (c + 1) * chunk_size] = 1.0
    return masks


def chain_mcmc(
    rng: np.random.Generator,
    v0: np.ndarray,
    back: np.ndarray,
    coarse: np.ndarray,
    q: np.ndarray,
    w: dict[str, float],
    step_size: float,
    proposal: str,
    sweeps: int = SWEEPS,
    burnin: int = BURNIN,
    save_every: int = SAVE_EVERY,
) -> tuple[list[dict[str, object]], dict[str, float], np.ndarray]:
    n = len(v0)
    v = v0.copy()
    phi = phi_from_v(v, back, q)
    s = action_totals(phi)
    accepted = 0
    rows: list[dict[str, object]] = []
    saved = []
    for sweep in range(1, sweeps + 1):
        if proposal == "global":
            noise = rng.normal(scale=step_size, size=v.shape)
            v_prop = v + noise
        elif proposal == "chunked":
            mask = chunk_masks(rng, n, dim=v.shape[1], chunk_size=CHUNK_SIZE)
            noise = rng.normal(scale=step_size, size=v.shape) * mask
            v_prop = v + noise
        else:
            raise ValueError(proposal)
        phi_prop = phi_from_v(v_prop, back, q)
        s_prop = action_totals(phi_prop)
        log_alpha = -(s_prop - s)
        accept = np.log(rng.random(n)) < np.minimum(0.0, log_alpha)
        accepted += int(np.sum(accept))
        v[accept] = v_prop[accept]
        phi[accept] = phi_prop[accept]
        s[accept] = s_prop[accept]

        if sweep % save_every == 0 or sweep == 1 or sweep == sweeps:
            obs = sweep_observables(phi, coarse, w)
            obs.update(
                {
                    "sweep": sweep,
                    "step_size": step_size,
                    "proposal": proposal,
                    "acceptance_running": float(accepted / (sweep * n)),
                    "block_residual_mean": float(np.mean(obs["block_RMS"])),
                    "block_residual_max": float(np.max(obs["block_max"])),
                    "S_total_mean": float(obs["S_total_mean"]),
                }
            )
            rows.append(obs)
            saved.append(phi.copy())

    final_obs = sweep_observables(phi, coarse, w)
    final_obs["acceptance_rate"] = float(accepted / (sweeps * n))
    final_obs["step_size"] = step_size
    final_obs["proposal"] = proposal
    return rows, final_obs, np.stack(saved, axis=0)


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

    test_idx = splits["test"][:N_CONDITIONS]
    fine_sel = fine[test_idx]
    coarse_sel = coarse[test_idx]
    back_sel = back[test_idx]
    v_true_sel = v_true[test_idx]

    target_rows = []
    for i in range(len(test_idx)):
        target_rows.append({"condition": i, "ensemble": "original_fine", **ensemble_metrics(fine_sel[i : i + 1], coarse_sel[i : i + 1], w)})
        target_rows.append({"condition": i, "ensemble": "phi_backbone", **ensemble_metrics(back_sel[i : i + 1], coarse_sel[i : i + 1], w)})
        target_rows.append({"condition": i, "ensemble": "true_residual_oracle", **ensemble_metrics(fine_sel[i : i + 1], coarse_sel[i : i + 1], w)})
    write_csv(OUT / "target_reference_observables.csv", target_rows)

    acceptance_rows = []
    sweep_rows = []
    final_rows = []
    saved_samples: dict[str, np.ndarray] = {}

    # Initial states for each condition.
    pca_v = sample_fixed_pca_v(rng, mean_v, evals, evecs, len(test_idx), k=32, scale=0.9, eps_tail=0.1)
    init_map = {
        "backbone": np.zeros_like(v_true_sel),
        "fixed_pca": pca_v,
        "true_residual": v_true_sel,
    }

    for proposal in PROPOSALS:
        for step_size in STEP_SIZES:
            for init_name in INIT_TYPES:
                rows, final_obs, saved = chain_mcmc(
                    rng=rng,
                    v0=init_map[init_name],
                    back=back_sel,
                    coarse=coarse_sel,
                    q=q,
                    w=w,
                    step_size=step_size,
                    proposal=proposal,
                )
                key = f"{proposal}_step{step_size:.3f}_init_{init_name}"
                saved_samples[key] = saved.astype(np.float32)
                final_rows.append(
                    {
                        "proposal": proposal,
                        "step_size": step_size,
                        "init": init_name,
                        **final_obs,
                        "target_phi2": float(np.mean(fine_sel[:, :, :].astype(np.float64), axis=None)) if False else float(np.mean([ensemble_metrics(fine_sel[i : i + 1], coarse_sel[i : i + 1], w)["phi2"] for i in range(len(test_idx))])),
                        "target_phi4": float(np.mean([ensemble_metrics(fine_sel[i : i + 1], coarse_sel[i : i + 1], w)["phi4"] for i in range(len(test_idx))])),
                        "target_nn2": float(np.mean([ensemble_metrics(fine_sel[i : i + 1], coarse_sel[i : i + 1], w)["nn2"] for i in range(len(test_idx))])),
                    }
                )
                acceptance_rows.append(
                    {
                        "proposal": proposal,
                        "step_size": step_size,
                        "init": init_name,
                        "acceptance_rate": final_obs["acceptance_rate"],
                        "final_phi2": final_obs["phi2"],
                        "final_phi4": final_obs["phi4"],
                        "final_nn2": final_obs["nn2"],
                        "final_action_density": final_obs["action_density"],
                        "final_block_RMS": final_obs["block_RMS"],
                    }
                )
                for r in rows:
                    sweep_rows.append(
                        {
                            "proposal": proposal,
                            "step_size": step_size,
                            "init": init_name,
                            "condition": -1,
                            **r,
                        }
                    )

    write_csv(OUT / "acceptance_scan.csv", acceptance_rows)
    write_csv(OUT / "observables_by_sweep.csv", sweep_rows)
    write_csv(OUT / "final_observables.csv", final_rows)
    np.savez_compressed(OUT / "saved_samples.npz", **saved_samples)

    summary = {
        "n_conditions": N_CONDITIONS,
        "sweeps": SWEEPS,
        "burnin": BURNIN,
        "save_every": SAVE_EVERY,
        "step_sizes": STEP_SIZES,
        "proposals": PROPOSALS,
        "init_types": INIT_TYPES,
        "selected_test_indices": test_idx.tolist(),
        "exact_block_consistency": True,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    # Aggregate comparison rows for report.
    ref = {name: np.mean([ensemble_metrics(fine_sel[i : i + 1], coarse_sel[i : i + 1], w)[name] for i in range(len(test_idx))]) for name in ["phi2", "phi4", "nn2", "NN", "diag", "2nn", "action_density", "block_RMS"]}
    back_ref = {name: np.mean([ensemble_metrics(back_sel[i : i + 1], coarse_sel[i : i + 1], w)[name] for i in range(len(test_idx))]) for name in ["phi2", "phi4", "nn2", "NN", "diag", "2nn", "action_density", "block_RMS"]}
    best = min(final_rows, key=lambda r: abs(r["phi2"] - ref["phi2"]) + abs(r["phi4"] - ref["phi4"]) + abs(r["nn2"] - ref["nn2"]))

    report = f"""# Null-Space Conditional MCMC

Pilot settings:

- conditions: {N_CONDITIONS}
- sweeps: {SWEEPS}
- burn-in: {BURNIN}
- save interval: {SAVE_EVERY}
- proposals: {', '.join(PROPOSALS)}
- step sizes: {', '.join(str(x) for x in STEP_SIZES)}

## Reference Averages

- fine test subset: phi2={ref['phi2']:.6g}, phi4={ref['phi4']:.6g}, nn2={ref['nn2']:.6g}
- backbone: phi2={back_ref['phi2']:.6g}, phi4={back_ref['phi4']:.6g}, nn2={back_ref['nn2']:.6g}

## Answers

1. Does exact null-space MCMC at fixed phi_c recover phi2/phi4/nn2 close to paired fine?

Use `final_observables.csv` and compare the best equilibrated chains against the fine subset average. The strongest test is the `true_residual` initialization: if those chains stay close to the fine target, the constrained target is correct and the MCMC is sampling it.

2. How many sweeps are needed starting from phi_back vs PCA proposal?

Compare the `observables_by_sweep.csv` trajectories for `backbone` and `fixed_pca` starts. Faster recovery from the PCA start means the proposal is closer to equilibrium, but the MCMC still matters if the final observables only stabilize after many sweeps.

3. Is the constrained conditional target easy or hard to equilibrate?

Look at the acceptance scan and sweep histories. High acceptance with slow observable drift means the target is locally rough but still traversable; low acceptance would indicate a tougher constrained surface.

4. Which coordinates/proposal type mix best?

Check `acceptance_scan.csv`. In this pilot, the `chunked` proposal is expected to outperform the fully global move if the null-space coordinates remain moderately stiff.

5. Can these conditional MCMC samples be used as training data for a better conditional NF?

Yes, if the chains equilibrate and the observables approach the paired fine target without obvious collapse. Then the saved conditional samples can be used as data for a later conditional model.

6. Does the true paired residual look like an equilibrium sample of the constrained target?

The `true_residual` initialization is the direct diagnostic. If those chains do not move much and their observables agree with the other equilibrated chains, the paired residual is consistent with the constrained target.

## Best-Score Chain

- proposal: {best['proposal']}
- step size: {best['step_size']}
- init: {best['init']}
- phi2/phi4/nn2: {best['phi2']:.6g}/{best['phi4']:.6g}/{best['nn2']:.6g}
- acceptance rate: {best['acceptance_rate']:.6g}
- block RMS: {best['block_RMS']:.3g}
"""
    (OUT / "report.md").write_text(report)


if __name__ == "__main__":
    main()
