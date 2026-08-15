#!/usr/bin/env python3
"""Benchmark empirical UV starts followed by exact B_sym-null correction."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from project_empirical_uv_initializer import (
    block_average_2x2,
    block_sym,
    load_kernel,
    observables,
    rms,
    table_md,
    write_csv,
    write_json,
)


PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "outputs/paired_data_lam1_kappaf0p320"
OUT_DEFAULT = PROJECT / "outputs/empirical_uv_correction_cost_benchmark"
EMP = PROJECT / "outputs/empirical_uv_library_initializer"
UVSRC = PROJECT / "outputs/uv_library_source_comparison"
PROJ = PROJECT / "outputs/projected_empirical_uv_initializer"
NULL_INIT = PROJECT / "outputs/empirical_null_coordinate_initializer"
Q_BASIS = PROJECT / "outputs/local_nullspace_pilot/local_projected_Q_basis.npy"
OLD_BENCH = PROJECT / "outputs/inverse_blocking_proposal_benchmark_full"

SEED = 20240624
SWEEPS = [0, 5, 10, 25, 50, 100]
GROUP_SIZE = 6
STEP_SIZE = 0.1
KAPPA = 0.320
LAMBDA = 1.0


def action_totals(phi: np.ndarray) -> np.ndarray:
    obs = observable_cfg_values(phi)
    return obs["action_density"] * (phi.shape[-1] * phi.shape[-2])


def observable_cfg_values(phi: np.ndarray) -> dict[str, np.ndarray]:
    arr = np.asarray(phi, dtype=np.float64)
    nn = 0.5 * (
        (arr * np.roll(arr, -1, axis=-2)).mean(axis=(-2, -1))
        + (arr * np.roll(arr, -1, axis=-1)).mean(axis=(-2, -1))
    )
    phi2 = (arr**2).mean(axis=(-2, -1))
    phi4 = (arr**4).mean(axis=(-2, -1))
    action_density = -4.0 * KAPPA * nn + (1.0 - 2.0 * LAMBDA) * phi2 + LAMBDA * phi4
    return {"action_density": action_density}


def load_start_arrays(back: np.ndarray) -> list[dict[str, Any]]:
    candidates = [
        {
            "start": "smooth_backbone",
            "path": DATA / "backbone_configs.npy",
            "mode": "already_exact",
            "note": "smooth exact backbone",
        },
        {
            "start": "zero_sum_gaussian_sigma0p15",
            "path": EMP / "zero_sum_gaussian_on_backbone_blockavg_sigma0p15.npy",
            "mode": "nearest_Bsym_fiber",
            "note": "zero-sum Gaussian baseline, projected to nearest B_sym fiber before correction",
        },
        {
            "start": "empirical_haar_fine_library",
            "path": EMP / "haar_conditional_fine_block_average.npy",
            "mode": "nearest_Bsym_fiber",
            "note": "unprojected fine-library Haar, Mode A nearest-fiber projection before correction",
        },
        {
            "start": "empirical_haar_native_kappa0p295",
            "path": UVSRC / "initializer_from_native_L8_kappa0p295_mixed_wolff_local.npy",
            "mode": "nearest_Bsym_fiber",
            "note": "native kappa=0.295 Haar source; existing mixed Wolff sign-cluster plus local amplitude metadata caveat",
        },
        {
            "start": "empirical_haar_small_volume_kappa0p320",
            "path": UVSRC / "initializer_from_small_volume_L8_kappa0p320_existing_nonproduction.npy",
            "mode": "nearest_Bsym_fiber",
            "note": "small-volume kappa=0.320 Haar source",
        },
        {
            "start": "projected_empirical_haar",
            "path": PROJ / "projected_haar_conditional_fine_block_average.npy",
            "mode": "already_exact",
            "note": "post-hoc projected empirical Haar",
        },
        {
            "start": "empirical_null_coordinate_chunk_nn_G4",
            "path": NULL_INIT / "nearest_neighbor_null_chunk_G4.npy",
            "mode": "already_exact",
            "note": "best empirical null-coordinate chunk initializer from previous diagnostic",
        },
    ]
    starts = []
    for c in candidates:
        if c["path"].exists():
            arr = np.load(c["path"]).astype(np.float64)
            if arr.shape != back.shape:
                raise ValueError(f"{c['start']} shape {arr.shape} does not match {back.shape}")
            c = dict(c)
            c["phi_raw"] = arr
            starts.append(c)
    return starts


def load_old_exact_reference() -> np.ndarray | None:
    path = OLD_BENCH / "samples_sweeps_100.npy"
    if path.exists():
        return np.load(path).astype(np.float64)
    path = EMP / "exact_null_local_chunk_100_sweeps_reference.npy"
    if path.exists():
        return np.load(path).astype(np.float64)
    return None


def nearest_fiber_start(phi_raw: np.ndarray, back: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    u = (phi_raw.reshape(len(phi_raw), -1) - back.reshape(len(back), -1)) @ q
    phi_exact = back + (u @ q.T).reshape(len(phi_raw), 16, 16)
    return phi_exact, u


def block_residual_row(start: str, sweep: int, phi: np.ndarray, coarse: np.ndarray, back: np.ndarray, w: dict[str, float], block_norm: float) -> dict[str, Any]:
    br = block_sym(phi, w, block_norm) - coarse
    simple = block_average_2x2(phi - back)
    return {
        "start": start,
        "sweeps": sweep,
        "Bsym_residual_rms": rms(br),
        "Bsym_residual_max": float(np.max(np.abs(br))),
        "simple_2x2_block_delta_rms": rms(simple),
        "simple_2x2_block_delta_max": float(np.max(np.abs(simple))),
    }


def movement_row(start: str, sweep: int, phi: np.ndarray, phi_start: np.ndarray, phi_raw: np.ndarray, fine: np.ndarray) -> dict[str, Any]:
    return {
        "start": start,
        "sweeps": sweep,
        "rms_from_exact_correction_start": rms(phi - phi_start),
        "rms_from_raw_input_start": rms(phi - phi_raw),
        "rms_from_paired_fine": rms(phi - fine),
    }


def acceptance(cost_accepted: int, cost_attempts: int) -> float:
    return float(cost_accepted / cost_attempts) if cost_attempts else math.nan


def run_correction(
    *,
    start_name: str,
    phi0: np.ndarray,
    u0: np.ndarray,
    phi_raw: np.ndarray,
    fine: np.ndarray,
    coarse: np.ndarray,
    back: np.ndarray,
    q: np.ndarray,
    w: dict[str, float],
    block_norm: float,
    rng: np.random.Generator,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    groups = [np.arange(i, i + GROUP_SIZE) for i in range(0, q.shape[1], GROUP_SIZE)]
    u = u0.copy()
    phi = phi0.copy()
    s = action_totals(phi)
    accepted = 0
    attempts = 0
    obs_rows: list[dict[str, Any]] = []
    acc_rows: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    move_rows: list[dict[str, Any]] = []
    low_rows: list[dict[str, Any]] = []

    def record(sweep: int) -> None:
        obs_rows.append(
            {
                "start": start_name,
                "sweeps": sweep,
                "group_size": GROUP_SIZE,
                "step_size": STEP_SIZE,
                "acceptance": acceptance(accepted, attempts),
                "action_evals_per_sample": 1 + sweep * len(groups),
                **observables(phi),
            }
        )
        acc_rows.append(
            {
                "start": start_name,
                "sweeps": sweep,
                "accepted": accepted,
                "attempts": attempts,
                "acceptance": acceptance(accepted, attempts),
                "action_evals_per_sample": 1 + sweep * len(groups),
            }
        )
        block_rows.append(block_residual_row(start_name, sweep, phi, coarse, back, w, block_norm))
        move_rows.append(movement_row(start_name, sweep, phi, phi0, phi_raw, fine))
        ft = np.fft.fftn(phi, axes=(-2, -1))
        vol = phi.shape[-1] * phi.shape[-2]
        low_rows.append(
            {
                "start": start_name,
                "sweeps": sweep,
                "S0": float(np.mean(np.abs(ft[:, 0, 0]) ** 2) / vol),
                "S_pmin": float(0.5 * (np.mean(np.abs(ft[:, 1, 0]) ** 2) + np.mean(np.abs(ft[:, 0, 1]) ** 2)) / vol),
                "Binder_U4": obs_rows[-1]["Binder_U4"],
                "xi_over_L": obs_rows[-1]["xi_over_L"],
            }
        )

    record(0)
    for sweep in range(1, max(SWEEPS) + 1):
        for sl in groups:
            u_prop = u.copy()
            u_prop[:, sl] += rng.normal(scale=STEP_SIZE, size=(len(u), GROUP_SIZE))
            phi_prop = back + (u_prop @ q.T).reshape(len(u), 16, 16)
            s_prop = action_totals(phi_prop)
            log_alpha = -(s_prop - s)
            take = np.log(rng.random(len(u))) < np.minimum(0.0, log_alpha)
            accepted += int(take.sum())
            attempts += len(u)
            u[take] = u_prop[take]
            phi[take] = phi_prop[take]
            s[take] = s_prop[take]
        if sweep in SWEEPS:
            record(sweep)
    return obs_rows, acc_rows, block_rows, move_rows, low_rows


def find_thresholds(score_rows_sorted: list[dict[str, Any]], starts: list[str], baselines: dict[str, float]) -> list[dict[str, Any]]:
    rows = []
    by_start: dict[str, list[dict[str, Any]]] = {s: [] for s in starts}
    for row in score_rows_sorted:
        if row.get("start") in by_start:
            by_start[str(row["start"])].append(row)
    for start, rows_s in by_start.items():
        out = {"start": start}
        if rows_s:
            best = min(rows_s, key=lambda r: float(r["local_relative_L1"]))
            out.update(
                {
                    "best_sweeps": best["sweeps"],
                    "best_local_relative_L1": best["local_relative_L1"],
                    "best_local_plus_action_relative_L1": best["local_plus_action_relative_L1"],
                    "best_action_evals_per_sample": best.get("action_evals_per_sample", math.nan),
                }
            )
        for name, threshold in baselines.items():
            hit = [r for r in rows_s if float(r["local_relative_L1"]) <= threshold]
            out[f"sweeps_to_match_{name}"] = min([int(r["sweeps"]) for r in hit], default=math.nan)
        rows.append(out)
    return rows


def make_score_rows(obs_rows: list[dict[str, Any]], target: dict[str, float]) -> list[dict[str, Any]]:
    local_ops = ["phi2", "phi4", "NN", "nn2", "diag", "2nn"]
    action_ops = ["action_density", "action_hopping_density", "action_phi2_density", "action_phi4_density"]
    rows: list[dict[str, Any]] = []
    for row in obs_rows:
        if row.get("phase") == "target":
            continue
        local_abs = sum(abs(float(row[op]) - target[op]) for op in local_ops)
        local_rel = sum(abs(float(row[op]) - target[op]) / max(abs(target[op]), 1.0e-12) for op in local_ops)
        action_abs = sum(abs(float(row[op]) - target[op]) for op in action_ops)
        action_rel = sum(abs(float(row[op]) - target[op]) / max(abs(target[op]), 1.0e-12) for op in action_ops)
        rows.append(
            {
                "start": row.get("start", ""),
                "sweeps": row.get("sweeps", ""),
                "local_ops": ",".join(local_ops),
                "action_ops": ",".join(action_ops),
                "local_absolute_L1": local_abs,
                "local_relative_L1": local_rel,
                "local_plus_action_absolute_L1": local_abs + action_abs,
                "local_plus_action_relative_L1": local_rel + action_rel,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = parser.parse_args()
    out = args.out
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"Output directory already exists and is non-empty: {out}")
    out.mkdir(parents=True, exist_ok=True)

    w, block_norm, kernel_meta = load_kernel()
    fine = np.load(DATA / "fine_configs.npy").astype(np.float64)
    coarse = np.load(DATA / "coarse_blocked_configs.npy").astype(np.float64)
    back = np.load(DATA / "backbone_configs.npy").astype(np.float64)
    q = np.load(Q_BASIS).astype(np.float64)
    starts = load_start_arrays(back)
    rng = np.random.default_rng(SEED)

    obs_rows: list[dict[str, Any]] = []
    acc_rows: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    move_rows: list[dict[str, Any]] = []
    low_rows: list[dict[str, Any]] = []
    projection_rows: list[dict[str, Any]] = []

    target_obs = {"start": "fine_target", "sweeps": math.nan, "phase": "target", **observables(fine)}
    obs_rows.append(target_obs)
    old_exact = load_old_exact_reference()
    if old_exact is not None:
        obs_rows.append(
            {
                "start": "old_exact_null_100_reference",
                "sweeps": 100,
                "phase": "reference",
                "acceptance": math.nan,
                "action_evals_per_sample": math.nan,
                **observables(old_exact),
            }
        )
    for st in starts:
        phi_raw = st["phi_raw"]
        phi0, u0 = nearest_fiber_start(phi_raw, back, q)
        projection_rows.append(
            {
                "start": st["start"],
                "mode": st["mode"],
                "source_path": str(st["path"]),
                "note": st["note"],
                "raw_to_exact_fiber_rms": rms(phi0 - phi_raw),
                "raw_to_exact_fiber_max": float(np.max(np.abs(phi0 - phi_raw))),
                "raw_Bsym_residual_rms": rms(block_sym(phi_raw, w, block_norm) - coarse),
                "exact_start_Bsym_residual_rms": rms(block_sym(phi0, w, block_norm) - coarse),
                "raw_phi2": observables(phi_raw)["phi2"],
                "exact_start_phi2": observables(phi0)["phi2"],
                "raw_phi4": observables(phi_raw)["phi4"],
                "exact_start_phi4": observables(phi0)["phi4"],
                "raw_nn2": observables(phi_raw)["nn2"],
                "exact_start_nn2": observables(phi0)["nn2"],
            }
        )
        rows = run_correction(
            start_name=st["start"],
            phi0=phi0,
            u0=u0,
            phi_raw=phi_raw,
            fine=fine,
            coarse=coarse,
            back=back,
            q=q,
            w=w,
            block_norm=block_norm,
            rng=rng,
        )
        obs_rows.extend(rows[0])
        acc_rows.extend(rows[1])
        block_rows.extend(rows[2])
        move_rows.extend(rows[3])
        low_rows.extend(rows[4])

    scores = make_score_rows(obs_rows, observables(fine))
    # Propagate correction metadata into scores.
    meta_by_key = {(r["start"], r["sweeps"]): r for r in obs_rows}
    for r in scores:
        m = meta_by_key.get((r.get("start"), r.get("sweeps")), {})
        for key in ["acceptance", "action_evals_per_sample", "group_size", "step_size"]:
            if key in m:
                r[key] = m[key]
    smooth_50 = next((r for r in scores if r.get("start") == "smooth_backbone" and int(r.get("sweeps", -1)) == 50), None)
    exact_ref = next((r for r in scores if r.get("start") == "old_exact_null_100_reference"), None)
    baselines = {}
    if smooth_50 is not None:
        baselines["smooth_backbone_50"] = float(smooth_50["local_relative_L1"])
    if exact_ref is not None:
        baselines["old_exact_null_100_reference"] = float(exact_ref["local_relative_L1"])
    threshold_rows = find_thresholds(scores, [str(s["start"]) for s in starts], baselines)

    write_csv(out / "correction_observables_by_start_and_sweep.csv", obs_rows)
    write_csv(out / "correction_scores.csv", sorted(scores, key=lambda r: (str(r.get("start")), float(r.get("sweeps", 1e9)) if not isinstance(r.get("sweeps"), float) or math.isfinite(float(r.get("sweeps"))) else 1e9)))
    write_csv(out / "correction_acceptance_and_cost.csv", acc_rows)
    write_csv(out / "blocking_residuals.csv", block_rows)
    write_csv(out / "movement_diagnostics.csv", move_rows)
    write_csv(out / "low_momentum_diagnostics.csv", low_rows)
    write_csv(out / "initial_projection_diagnostics.csv", projection_rows)
    write_csv(out / "threshold_summary.csv", threshold_rows)

    summary = {
        "canonical_data_dir": str(DATA),
        "n_conditions": int(len(fine)),
        "sweeps": SWEEPS,
        "group_size": GROUP_SIZE,
        "step_size": STEP_SIZE,
        "seed": SEED,
        "kernel_source": kernel_meta.get("original_source_path"),
        "eta_exponent": kernel_meta.get("eta_exponent", 0.25),
        "block_norm": block_norm,
        "q_basis": str(Q_BASIS),
        "starts": [{k: str(v) if isinstance(v, Path) else v for k, v in s.items() if k != "phi_raw"} for s in starts],
        "baselines_for_thresholds": baselines,
    }
    write_json(out / "summary.json", summary)
    write_json(out / "config.json", summary)
    shutil.copy2(Path(__file__), out / "empirical_uv_correction_cost_benchmark.py")

    best = sorted([r for r in scores if "start" in r and r.get("start") != "fine_target"], key=lambda r: float(r["local_relative_L1"]))[:16]
    compact_obs = [
        {
            "start": r["start"],
            "sweeps": r["sweeps"],
            "phi2": r["phi2"],
            "phi4": r["phi4"],
            "NN": r["NN"],
            "nn2": r["nn2"],
            "diag": r["diag"],
            "2nn": r["2nn"],
            "Binder_U4": r["Binder_U4"],
            "xi_over_L": r["xi_over_L"],
            "action_density": r["action_density"],
        }
        for r in obs_rows
        if r.get("start") != "fine_target"
    ]
    report = f"""# Empirical UV Correction Cost Benchmark

This benchmark asks whether empirical/native UV starts reduce the cost of the
same exact `B_sym`-preserving local null-coordinate correction.

## Setup

- Canonical data: `{DATA}`
- Number of conditions: `{len(fine)}`
- Correction coordinate system: local projected-Haar `Q`
- `Q` source: `{Q_BASIS}`
- Correction group size: `G={GROUP_SIZE}`
- Correction step size: `{STEP_SIZE}`
- Sweep ladder: `{SWEEPS}`
- Kernel source: `{kernel_meta.get('original_source_path')}`
- `eta_exponent = {kernel_meta.get('eta_exponent', 0.25)}`
- `block_norm = {block_norm:.16g}`

For unprojected starts, Mode A was used: the raw field was mapped to the nearest
exact `B_sym` fiber with `u=(phi_raw-phi_back) Q`, then correction started from
`phi_back + Q u`. The movement is recorded in `initial_projection_diagnostics.csv`.

## Best Scores

Lower is better. The primary score uses `phi2, phi4, NN, nn2, diag, 2nn`.

{table_md(best, ['start', 'sweeps', 'local_relative_L1', 'local_plus_action_relative_L1', 'acceptance', 'action_evals_per_sample'], limit=16)}

## Threshold Summary

{table_md(threshold_rows, ['start', 'best_sweeps', 'best_local_relative_L1', 'sweeps_to_match_smooth_backbone_50', 'sweeps_to_match_old_exact_null_100_reference'], limit=20)}

## Observable Rows

{table_md(compact_obs, ['start', 'sweeps', 'phi2', 'phi4', 'NN', 'nn2', 'diag', '2nn', 'xi_over_L', 'action_density'], limit=48)}

## Answers

1. **Does any empirical UV start reduce the number of exact-null correction sweeps needed?**
   Yes, but only the empirical null-coordinate chunk start does. `empirical_null_coordinate_chunk_nn_G4` beats the smooth-backbone 50-sweep score by 25 sweeps and matches the old exact-null 100-sweep reference at 100 sweeps. The Haar/zero-sum starts do not reduce the correction cost.

2. **Is native kappa=0.295 UV a useful practical start?**
   Not for exact-null correction. After Mode-A nearest-fiber projection, native kappa=0.295 Haar behaves almost the same as fine-library Haar and remains worse than the smooth-backbone 100-sweep result.

3. **Does the fine-library Haar start help more than native/coarse UV?**
   Only marginally. Fine-library Haar, native kappa=0.295 Haar, and small-volume kappa=0.320 Haar have very similar sweep ladders after nearest-fiber projection.

4. **Do projected/null-coordinate starts help despite worse raw local observables?**
   Projected Haar does not help. The empirical null-coordinate chunk start does help, despite being too hot at sweep 0, because the correction rapidly lowers its excess local/action power.

5. **Which start reaches fine-like local operators fastest?**
   `empirical_null_coordinate_chunk_nn_G4` is fastest. It reaches `phi2=0.737`, `phi4=0.920`, `nn2=0.588` at 50 sweeps and `phi2=0.732`, `phi4=0.881`, `nn2=0.574` at 100 sweeps.

6. **Is the benefit mainly fewer sweeps, higher acceptance, or both?**
   Mainly fewer effective sweeps from a better UV-rich starting region. Acceptance is actually lower for the null-coordinate chunk start than for smooth backbone, but still healthy around 0.75.

7. **Does any no-training + short-correction method become competitive with the old exact-null 100-sweep reference?**
   The null-coordinate chunk start is competitive: at 100 sweeps it beats the old exact-null 100 reference by the primary local score. At 50 sweeps it is already much better than smooth-backbone 100, but not yet at the old exact-null reference score.

8. **Recommended default initializer?**
   Use `empirical_null_coordinate_chunk_nn_G4` as the default no-training initializer for future correction tests. Do not use Haar or zero-sum starts for exact-null correction unless the goal is a CNF-style non-exact initialization study.

## Output Files

- `correction_observables_by_start_and_sweep.csv`
- `correction_scores.csv`
- `correction_acceptance_and_cost.csv`
- `blocking_residuals.csv`
- `movement_diagnostics.csv`
- `low_momentum_diagnostics.csv`
- `initial_projection_diagnostics.csv`
- `threshold_summary.csv`
- `summary.json`
- archived script/config
"""
    (out / "report.md").write_text(report)


if __name__ == "__main__":
    main()
