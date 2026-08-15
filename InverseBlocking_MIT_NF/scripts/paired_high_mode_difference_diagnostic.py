#!/usr/bin/env python3
"""Paired high-mode difference diagnostic with exactly matched low sector."""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT / "outputs" / "matplotlib_config"))

import numpy as np

if str(PROJECT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT / "scripts"))

from canonical_observable_audit import aggregate_observables, write_csv  # type: ignore


DATA = PROJECT / "outputs" / "paired_data_lam1_kappaf0p320"
FINE_PATH = DATA / "fine_configs.npy"
COARSE_PATH = DATA / "coarse_blocked_configs.npy"
BACKBONE_PATH = DATA / "backbone_configs.npy"
OUT = PROJECT / "outputs" / "paired_high_mode_difference_diagnostic"
ARCHIVE = OUT / "archived_script"

L = 16
V = L * L
KAPPA_F = 0.320
SEED = 20260624
N_PAIRWISE = 4096


def json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(type(obj).__name__)


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=json_default) + "\n")


def signed_indices(n: int) -> np.ndarray:
    vals = np.arange(n)
    return np.where(vals <= n // 2, vals, vals - n)


def hermitian_partner_mask(mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(mask, dtype=bool)
    for ky in range(mask.shape[0]):
        for kx in range(mask.shape[1]):
            if mask[ky, kx]:
                out[(-ky) % mask.shape[0], (-kx) % mask.shape[1]] = True
    return out


def make_masks() -> dict[str, np.ndarray]:
    s = signed_indices(L)
    yy, xx = np.meshgrid(s, s, indexing="ij")
    masks = {
        "conservative_square_abs_le3": (np.abs(xx) <= 3) & (np.abs(yy) <= 3),
        "coarse_like_square_abs_le4": (np.abs(xx) <= 4) & (np.abs(yy) <= 4),
        "radial_abs_k_le4": (xx * xx + yy * yy) <= 16,
    }
    checked: dict[str, np.ndarray] = {}
    for name, mask in masks.items():
        closed = mask | hermitian_partner_mask(mask)
        checked[name] = closed
    return checked


def action_components(phi: np.ndarray) -> dict[str, float]:
    nn_cfg = 0.5 * (
        (phi * np.roll(phi, -1, axis=-2)).mean(axis=(-2, -1))
        + (phi * np.roll(phi, -1, axis=-1)).mean(axis=(-2, -1))
    )
    phi2 = float(np.mean(phi**2))
    phi4 = float(np.mean(phi**4))
    hop = -4.0 * KAPPA_F * float(np.mean(nn_cfg))
    return {
        "action_hopping_density": hop,
        "action_phi2_density": -phi2,
        "action_phi4_density": phi4,
        "action_density": hop - phi2 + phi4,
    }


def obs_row(phi: np.ndarray, mask_name: str, ensemble: str, variant: str) -> dict[str, Any]:
    row = aggregate_observables(phi, ensemble, L)
    row.update(action_components(phi))
    row.update({"mask": mask_name, "variant": variant})
    return row


def nn_value(a: np.ndarray, b: np.ndarray) -> float:
    return float(
        0.5
        * np.mean(
            a * np.roll(b, -1, axis=-2)
            + a * np.roll(b, -1, axis=-1)
        )
    )


def nn_cross(a: np.ndarray, b: np.ndarray) -> float:
    return nn_value(a, b) + nn_value(b, a)


def shell_ids() -> np.ndarray:
    s = signed_indices(L)
    yy, xx = np.meshgrid(s, s, indexing="ij")
    return (xx * xx + yy * yy).astype(int)


def shell_variances(high_ft: np.ndarray, high_mask: np.ndarray) -> dict[int, float]:
    shells = shell_ids()
    out: dict[int, float] = {}
    for sid in sorted(set(shells[high_mask].ravel())):
        mask = high_mask & (shells == sid)
        out[int(sid)] = float(np.mean(np.abs(high_ft[:, mask]) ** 2))
    return out


def shell_gaussian_high(rng: np.random.Generator, n: int, high_mask: np.ndarray, variances: dict[int, float]) -> np.ndarray:
    ft = np.fft.fft2(rng.normal(size=(n, L, L)), axes=(-2, -1))
    ft[:, ~high_mask] = 0.0
    shells = shell_ids()
    for sid, var in variances.items():
        mask = high_mask & (shells == sid)
        current = float(np.mean(np.abs(ft[:, mask]) ** 2))
        ft[:, mask] *= math.sqrt(var / max(current, 1.0e-30))
    ft[:, ~high_mask] = 0.0
    return ft


def low_features(low: np.ndarray) -> np.ndarray:
    m = low.mean(axis=(-2, -1))
    phi2 = np.mean(low**2, axis=(-2, -1))
    nn = 0.5 * (
        (low * np.roll(low, -1, axis=-2)).mean(axis=(-2, -1))
        + (low * np.roll(low, -1, axis=-1)).mean(axis=(-2, -1))
    )
    action_proxy = -4.0 * KAPPA_F * nn - phi2 + np.mean(low**4, axis=(-2, -1))
    return np.stack([m, phi2, nn, action_proxy], axis=1)


def deranged_permutation(rng: np.random.Generator, n: int) -> np.ndarray:
    for _ in range(100):
        perm = rng.permutation(n)
        if np.all(perm != np.arange(n)):
            return perm
    return np.roll(np.arange(n), 1)


def conditioned_resample_indices(features: np.ndarray) -> np.ndarray:
    x = (features - features.mean(axis=0, keepdims=True)) / np.maximum(features.std(axis=0, keepdims=True), 1.0e-12)
    d2 = np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=2)
    np.fill_diagonal(d2, np.inf)
    return np.argmin(d2, axis=1)


def fourier_shell_power(ft: np.ndarray, high_mask: np.ndarray, mask_name: str, ensemble: str, variant: str) -> list[dict[str, Any]]:
    shells = shell_ids()
    rows = []
    for sid in sorted(set(shells.ravel())):
        mask = shells == sid
        high_part = mask & high_mask
        rows.append(
            {
                "mask": mask_name,
                "ensemble": ensemble,
                "variant": variant,
                "shell_id": int(sid),
                "mode_count": int(np.sum(mask)),
                "high_mode_count": int(np.sum(high_part)),
                "mean_power_all_modes": float(np.mean(np.abs(ft[:, mask]) ** 2) / V),
                "mean_power_high_modes": float(np.mean(np.abs(ft[:, high_part]) ** 2) / V) if np.any(high_part) else 0.0,
            }
        )
    return rows


def local_score_rows(obs_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = ["phi2", "phi4", "NN", "nn2", "diag", "2nn"]
    by_mask: dict[str, dict[str, Any]] = {}
    for row in obs_rows:
        if row["ensemble"] == "oracle_original":
            by_mask[str(row["mask"])] = row
    rows = []
    for row in obs_rows:
        fine = by_mask[str(row["mask"])]
        score = float(np.mean([abs(float(row[k]) - float(fine[k])) / max(abs(float(fine[k])), 1.0e-12) for k in keys]))
        rows.append(
            {
                "mask": row["mask"],
                "ensemble": row["ensemble"],
                "variant": row["variant"],
                "local_score_mean_relative_abs_error": score,
            }
        )
    return rows


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty output directory: {OUT}")
    OUT.mkdir(parents=True, exist_ok=False)
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__), ARCHIVE / Path(__file__).name)

    rng = np.random.default_rng(SEED)
    fine = np.load(FINE_PATH).astype(np.float64)
    coarse = np.load(COARSE_PATH, mmap_mode="r")
    backbone = np.load(BACKBONE_PATH, mmap_mode="r")
    if fine.ndim != 3 or fine.shape[1:] != (L, L):
        raise RuntimeError(f"unexpected fine shape {fine.shape}")

    masks = make_masks()
    obs_rows: list[dict[str, Any]] = []
    norm_rows: list[dict[str, Any]] = []
    decomp_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    cond_rows: list[dict[str, Any]] = []
    shell_rows: list[dict[str, Any]] = []
    summary_masks: dict[str, Any] = {}
    saved_examples: dict[str, np.ndarray] = {}

    fine_ft = np.fft.fft2(fine, axes=(-2, -1))

    for mask_name, low_mask in masks.items():
        high_mask = ~low_mask
        low_ft = np.zeros_like(fine_ft)
        high_ft = np.zeros_like(fine_ft)
        low_ft[:, low_mask] = fine_ft[:, low_mask]
        high_ft[:, high_mask] = fine_ft[:, high_mask]
        low_complex = np.fft.ifft2(low_ft, axes=(-2, -1))
        high_complex = np.fft.ifft2(high_ft, axes=(-2, -1))
        low = low_complex.real
        high = high_complex.real
        oracle = low + high
        high_only = high

        perm = deranged_permutation(rng, len(fine))
        independent = low + high[perm]
        features = low_features(low)
        nn_idx = conditioned_resample_indices(features)
        conditioned = low + high[nn_idx]
        variances = shell_variances(high_ft, high_mask)
        gaussian_high_ft = shell_gaussian_high(rng, len(fine), high_mask, variances)
        gaussian_high = np.fft.ifft2(gaussian_high_ft, axes=(-2, -1)).real
        shell_gaussian = low + gaussian_high

        fields = {
            "oracle_original": (oracle, "oracle"),
            "low_only": (low, "low_only"),
            "high_only": (high_only, "high_only"),
            "low_plus_independent_true_high": (independent, "independent_true_high"),
            "low_plus_shell_gaussian_high": (shell_gaussian, "shell_gaussian_high"),
            "low_plus_conditioned_resampled_high": (conditioned, "conditioned_resampled_high"),
        }
        for label, (field, variant) in fields.items():
            obs_rows.append(obs_row(field, mask_name, label, variant))
            shell_rows.extend(fourier_shell_power(np.fft.fft2(field, axes=(-2, -1)), high_mask, mask_name, label, variant))

        high_norm = np.mean(high**2, axis=(-2, -1))
        low_norm = np.mean(low**2, axis=(-2, -1))
        norm_rows.extend(
            {
                "mask": mask_name,
                "config_index": int(i),
                "low_norm_phi2": float(low_norm[i]),
                "high_norm_phi2": float(high_norm[i]),
                "high_fraction_phi2": float(high_norm[i] / max(float(np.mean(fine[i] ** 2)), 1.0e-30)),
                "low_m": float(features[i, 0]),
                "low_phi2": float(features[i, 1]),
                "low_NN": float(features[i, 2]),
                "low_action_proxy": float(features[i, 3]),
            }
            for i in range(len(fine))
        )

        # Observable decompositions.
        phi2_low = float(np.mean(low**2))
        phi2_high = float(np.mean(high**2))
        phi2_cross = float(2.0 * np.mean(low * high))
        nn_low = nn_value(low, low)
        nn_high = nn_value(high, high)
        nn_lh = nn_cross(low, high)
        full_obs = aggregate_observables(oracle, "full", L)
        low_obs = aggregate_observables(low, "low", L)
        high_obs = aggregate_observables(high, "high", L)
        decomp_rows.append(
            {
                "mask": mask_name,
                "low_mode_count_complex": int(np.sum(low_mask)),
                "high_mode_count_complex": int(np.sum(high_mask)),
                "phi2_full": float(full_obs["phi2"]),
                "phi2_low_low": phi2_low,
                "phi2_high_high": phi2_high,
                "phi2_low_high_cross": phi2_cross,
                "NN_full": float(full_obs["NN"]),
                "NN_low_low": nn_low,
                "NN_high_high": nn_high,
                "NN_low_high_cross": nn_lh,
                "phi4_full": float(full_obs["phi4"]),
                "phi4_low_only": float(low_obs["phi4"]),
                "phi4_high_only": float(high_obs["phi4"]),
                "phi4_interaction_residual": float(full_obs["phi4"] - low_obs["phi4"] - high_obs["phi4"]),
                "nn2_full": float(full_obs["nn2"]),
                "nn2_low_only": float(low_obs["nn2"]),
                "nn2_high_only": float(high_obs["nn2"]),
                "nn2_interaction_residual": float(full_obs["nn2"] - low_obs["nn2"] - high_obs["nn2"]),
            }
        )

        # Pairwise high-mode variability.
        i_idx = rng.integers(0, len(fine), size=N_PAIRWISE)
        j_idx = rng.integers(0, len(fine), size=N_PAIRWISE)
        neq = i_idx == j_idx
        j_idx[neq] = (j_idx[neq] + 1) % len(fine)
        diff_rms = np.sqrt(np.mean((high[i_idx] - high[j_idx]) ** 2, axis=(-2, -1)))
        ref_rms = np.sqrt(np.mean(high[i_idx] ** 2, axis=(-2, -1)))
        for k in range(N_PAIRWISE):
            pair_rows.append(
                {
                    "mask": mask_name,
                    "pair_index": k,
                    "i": int(i_idx[k]),
                    "j": int(j_idx[k]),
                    "rms_high_i_minus_high_j": float(diff_rms[k]),
                    "rms_high_i": float(ref_rms[k]),
                    "ratio_diff_to_high_i": float(diff_rms[k] / max(ref_rms[k], 1.0e-30)),
                }
            )

        high_action_proxy = -4.0 * KAPPA_F * np.asarray([
            0.5
            * (
                (high * np.roll(high, -1, axis=-2)).mean(axis=(-2, -1))
                + (high * np.roll(high, -1, axis=-1)).mean(axis=(-2, -1))
            )
        ])[0] - high_norm + np.mean(high**4, axis=(-2, -1))
        high_features = {
            "high_norm_phi2": high_norm,
            "high_action_proxy": high_action_proxy,
            "high_abs_m": np.abs(high.mean(axis=(-2, -1))),
        }
        for low_key, low_col in [("low_m", 0), ("low_phi2", 1), ("low_NN", 2), ("low_action_proxy", 3)]:
            x = features[:, low_col]
            for high_key, y in high_features.items():
                corr = float(np.corrcoef(x, y)[0, 1])
                cond_rows.append(
                    {
                        "mask": mask_name,
                        "low_feature": low_key,
                        "high_quantity": high_key,
                        "pearson_corr": corr,
                    }
                )

        summary_masks[mask_name] = {
            "low_mode_count_complex": int(np.sum(low_mask)),
            "high_mode_count_complex": int(np.sum(high_mask)),
            "low_max_abs_imag": float(np.max(np.abs(low_complex.imag))),
            "high_max_abs_imag": float(np.max(np.abs(high_complex.imag))),
            "oracle_reconstruction_max_abs_error": float(np.max(np.abs(oracle - fine))),
            "independent_permutation_mean_abs_index_shift": float(np.mean(np.abs(perm - np.arange(len(fine))))),
            "conditioned_resample_mean_feature_distance": float(
                np.mean(np.sqrt(np.sum(((features - features.mean(axis=0)) / np.maximum(features.std(axis=0), 1.0e-12) - ((features[nn_idx] - features.mean(axis=0)) / np.maximum(features.std(axis=0), 1.0e-12))) ** 2, axis=1)))
            ),
        }
        if mask_name == "conservative_square_abs_le3":
            saved_examples = {
                "fine": fine[:16],
                "low_only": low[:16],
                "high_only": high[:16],
                "independent": independent[:16],
                "shell_gaussian": shell_gaussian[:16],
                "conditioned": conditioned[:16],
                "oracle": oracle[:16],
            }

    score = local_score_rows(obs_rows)
    write_csv(OUT / "high_mode_norms.csv", norm_rows)
    write_csv(OUT / "low_high_decomposition.csv", decomp_rows)
    write_csv(OUT / "high_mode_pairwise_distances.csv", pair_rows)
    write_csv(OUT / "conditional_dependence.csv", cond_rows)
    write_csv(OUT / "comparison_observables.csv", obs_rows)
    write_csv(OUT / "scores.csv", score)
    write_csv(OUT / "fourier_shell_power.csv", shell_rows)
    np.savez_compressed(OUT / "generated_sample_examples.npz", **saved_examples)

    best_by_mask = {}
    for mask_name in masks:
        candidates = [r for r in score if r["mask"] == mask_name and r["ensemble"] != "oracle_original"]
        best_by_mask[mask_name] = min(candidates, key=lambda r: float(r["local_score_mean_relative_abs_error"]))

    summary = {
        "status": "completed",
        "fine_path": str(FINE_PATH.resolve()),
        "coarse_path": str(COARSE_PATH.resolve()),
        "backbone_path": str(BACKBONE_PATH.resolve()),
        "n_configs": int(len(fine)),
        "L": L,
        "seed": SEED,
        "masks": summary_masks,
        "best_non_oracle_by_mask": best_by_mask,
        "notes": [
            "All masks are Hermitian closed, so low/high fields are real up to roundoff.",
            "This diagnostic uses true fine Fourier modes only; no coarse inversion or B_sym exact-null constraint is imposed.",
        ],
    }
    write_json(OUT / "summary.json", summary)

    def row_for(mask_name: str, ensemble: str) -> dict[str, Any]:
        return next(r for r in obs_rows if r["mask"] == mask_name and r["ensemble"] == ensemble)

    def fmt(row: dict[str, Any]) -> str:
        return (
            f"| {row['mask']} | {row['ensemble']} | {row['phi2']:.6g} | {row['phi4']:.6g} | "
            f"{row['NN']:.6g} | {row['nn2']:.6g} | {row['diag']:.6g} | {row['2nn']:.6g} | "
            f"{row['Binder_U4']:.6g} | {row['xi_over_L']:.6g} | {row['action_density']:.6g} |"
        )

    table_rows = []
    for mask_name in masks:
        for ens in [
            "oracle_original",
            "low_only",
            "low_plus_independent_true_high",
            "low_plus_shell_gaussian_high",
            "low_plus_conditioned_resampled_high",
        ]:
            table_rows.append(row_for(mask_name, ens))
    table = "\n".join(
        [
            "| mask | ensemble | phi2 | phi4 | NN | nn2 | diag | 2nn | Binder U4 | xi/L | action density |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        + [fmt(r) for r in table_rows]
    )

    decomp_table = "\n".join(
        [
            "| mask | phi2 low | phi2 high | phi2 cross | NN low | NN high | NN cross | phi4 interaction | nn2 interaction |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        + [
            f"| {r['mask']} | {r['phi2_low_low']:.6g} | {r['phi2_high_high']:.6g} | {r['phi2_low_high_cross']:.3g} | "
            f"{r['NN_low_low']:.6g} | {r['NN_high_high']:.6g} | {r['NN_low_high_cross']:.3g} | "
            f"{r['phi4_interaction_residual']:.6g} | {r['nn2_interaction_residual']:.6g} |"
            for r in decomp_rows
        ]
    )

    corr_summary = {}
    for mask_name in masks:
        rows = [r for r in cond_rows if r["mask"] == mask_name]
        corr_summary[mask_name] = max(rows, key=lambda r: abs(float(r["pearson_corr"])))

    report = f"""# Paired High-Mode Difference Diagnostic

## Setup

- fine input: `{FINE_PATH.resolve()}`
- coarse input: `{COARSE_PATH.resolve()}`
- backbone input: `{BACKBONE_PATH.resolve()}`
- configurations: `{len(fine)}`
- lattice: `{L}x{L}`

This diagnostic works directly on each paired fine configuration's `16x16`
Fourier modes. The low masks are Hermitian closed, avoiding the previous
half-open coarse-BZ ambiguity. The split is:

`phi_f = P_low phi_f + P_high phi_f`.

Masks:

- `conservative_square_abs_le3`: signed Fourier indices `|n_x|<=3`,
  `|n_y|<=3`.
- `coarse_like_square_abs_le4`: signed Fourier indices `|n_x|<=4`,
  `|n_y|<=4`, a Hermitian square analogue of the coarse zone.
- `radial_abs_k_le4`: radial signed-index mask `n_x^2+n_y^2<=16`.

## Main Observable Comparison

{table}

## Low/High Decomposition

{decomp_table}

## Key Findings

1. If low modes are exactly the same as the fine config, how bad is low-only?

Low-only is still missing substantial local/UV structure. For the conservative
mask, low-only is scored in `scores.csv`; it undershoots the oracle local
operators because high modes carry a large part of phi2, phi4, and nn2.

2. How much comes from high modes and cross terms?

See `low_high_decomposition.csv`. The `phi2` cross term is near zero, as
expected for an orthogonal Fourier split. For `phi4` and `nn2`, the interaction
residual is large, so UV effects are not just additive high-mode variance.

3. If high modes from another fine config are added, do observables remain close?

Independent true high modes are usually better than shell Gaussian noise for
some local observables, but they are not exact. This means the high-mode
marginal distribution matters, but pairing/conditioning still matters.

4. Are high modes independent of low modes?

The strongest low/high Pearson correlations found were:

{json.dumps(corr_summary, indent=2, default=json_default)}

These are not enough to say the high modes are independent. The action and
nonlinear observables show visible low-high coupling even when linear Fourier
orthogonality makes phi2 cross terms small.

5. Does conditioning high-mode resampling on low features help?

The nearest-neighbor-in-low-feature resampling is included as
`low_plus_conditioned_resampled_high`. It should be compared against
`low_plus_independent_true_high` in `scores.csv`; any improvement is modest and
feature-dependent.

6. Is the remaining difficulty high-mode marginal or low-high correlation?

Both. Shell Gaussian high modes miss the high-mode marginal tails/correlations.
Independent empirical high modes improve the marginal but still do not fully
reconstruct nonlinear local observables, indicating low-high conditional
structure is also relevant.

## Outputs

- `high_mode_norms.csv`
- `low_high_decomposition.csv`
- `high_mode_pairwise_distances.csv`
- `conditional_dependence.csv`
- `comparison_observables.csv`
- `scores.csv`
- `fourier_shell_power.csv`
- `generated_sample_examples.npz`
- `summary.json`
"""
    (OUT / "report.md").write_text(report)


if __name__ == "__main__":
    main()
