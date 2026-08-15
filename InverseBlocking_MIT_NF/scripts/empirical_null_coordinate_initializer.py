#!/usr/bin/env python3
"""Empirical B_sym-null coordinate UV initializers.

This diagnostic samples empirical residual chunks directly in the
projected-Haar B_sym-null coordinate system used by the local-chunk correction
branch. It intentionally avoids "sample arbitrary UV then project" as the
primary construction.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT / "scripts"))

from project_empirical_uv_initializer import (  # noqa: E402
    block_average_2x2,
    block_sym,
    load_kernel,
    low_momentum_rows,
    observables,
    pmin_cross_terms,
    rms,
    score_rows,
    table_md,
    write_csv,
    write_json,
)


DATA = PROJECT / "outputs/paired_data_lam1_kappaf0p320"
OUT_DEFAULT = PROJECT / "outputs/empirical_null_coordinate_initializer"
Q_BASIS = PROJECT / "outputs/local_nullspace_pilot/local_projected_Q_basis.npy"
PROJECTED_HAAR = PROJECT / "outputs/projected_empirical_uv_initializer"
EMP_HAAR = PROJECT / "outputs/empirical_uv_library_initializer"
UV_SOURCE = PROJECT / "outputs/uv_library_source_comparison"
BENCH = PROJECT / "outputs/inverse_blocking_proposal_benchmark_full"

SEED = 20240624
GROUP_SIZES = [4, 6, 8, 12]
N_BINS = 12


def chunk_block_ids(start: int, stop: int) -> np.ndarray:
    # Projected-Haar coordinates inherit the original Haar detail ordering:
    # three local detail coordinates per 2x2 block before projection/QR.
    return np.unique(np.arange(start, stop, dtype=int) // 3)


def block_feature_tables(back: np.ndarray, coarse: np.ndarray) -> dict[str, np.ndarray]:
    back_avg = block_average_2x2(back)
    back_phi2 = block_average_2x2(back * back)
    gy = np.roll(back_avg, -1, axis=1) - back_avg
    gx = np.roll(back_avg, -1, axis=2) - back_avg
    grad = np.sqrt(0.5 * (gx * gx + gy * gy))
    return {
        "back_mean": back_avg.reshape(len(back), -1),
        "back_phi2": back_phi2.reshape(len(back), -1),
        "coarse_mean": coarse.reshape(len(coarse), -1),
        "grad": grad.reshape(len(back), -1),
    }


def chunk_features(back_feats: dict[str, np.ndarray], start: int, stop: int) -> np.ndarray:
    ids = chunk_block_ids(start, stop)
    return np.column_stack(
        [
            back_feats["back_mean"][:, ids].mean(axis=1),
            back_feats["back_phi2"][:, ids].mean(axis=1),
            back_feats["coarse_mean"][:, ids].mean(axis=1),
            back_feats["grad"][:, ids].mean(axis=1),
        ]
    )


def bin_edges(x: np.ndarray, n_bins: int) -> np.ndarray:
    edges = np.quantile(x, np.linspace(0.0, 1.0, n_bins + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = np.nextafter(edges[i - 1], np.inf)
    return edges


def sample_unconditional(
    rng: np.random.Generator,
    library: np.ndarray,
    n: int,
) -> np.ndarray:
    idx = rng.integers(0, len(library), size=n)
    return library[idx]


def sample_conditional_bin(
    rng: np.random.Generator,
    library: np.ndarray,
    train_feature: np.ndarray,
    target_feature: np.ndarray,
    n_bins: int,
) -> np.ndarray:
    edges = bin_edges(train_feature, n_bins)
    train_bins = np.clip(np.searchsorted(edges, train_feature, side="right") - 1, 0, n_bins - 1)
    target_bins = np.clip(np.searchsorted(edges, target_feature, side="right") - 1, 0, n_bins - 1)
    pools = [np.flatnonzero(train_bins == b) for b in range(n_bins)]
    out = np.empty((len(target_feature), library.shape[1]), dtype=np.float64)
    all_idx = np.arange(len(library))
    for b in range(n_bins):
        mask = target_bins == b
        if not np.any(mask):
            continue
        pool = pools[b] if len(pools[b]) else all_idx
        out[mask] = library[rng.choice(pool, size=int(mask.sum()), replace=True)]
    return out


def sample_nearest_neighbor(
    rng: np.random.Generator,
    library: np.ndarray,
    train_features: np.ndarray,
    target_features: np.ndarray,
    k: int = 16,
) -> np.ndarray:
    try:
        from scipy.spatial import cKDTree
    except Exception:
        k = min(k, len(train_features))
        out = np.empty((len(target_features), library.shape[1]), dtype=np.float64)
        for i, feat in enumerate(target_features):
            d = np.sum((train_features - feat[None, :]) ** 2, axis=1)
            nn = np.argpartition(d, k - 1)[:k]
            out[i] = library[rng.choice(nn)]
        return out
    mean = train_features.mean(axis=0)
    std = np.maximum(train_features.std(axis=0), 1.0e-12)
    tree = cKDTree((train_features - mean) / std)
    _, inds = tree.query((target_features - mean) / std, k=min(k, len(train_features)))
    if inds.ndim == 1:
        inds = inds[:, None]
    choose = rng.integers(0, inds.shape[1], size=len(target_features))
    return library[inds[np.arange(len(target_features)), choose]]


def make_initializer(
    *,
    variant: str,
    group_size: int,
    u_true: np.ndarray,
    train_idx: np.ndarray,
    back_feats: dict[str, np.ndarray],
    back: np.ndarray,
    q_basis: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    if u_true.shape[1] % group_size:
        raise ValueError(f"group_size={group_size} does not divide {u_true.shape[1]}")
    n, dim = u_true.shape
    u = np.empty_like(u_true)
    rows: list[dict[str, Any]] = []
    for start in range(0, dim, group_size):
        stop = start + group_size
        chunk_train = u_true[train_idx, start:stop]
        feats_all = chunk_features(back_feats, start, stop)
        feats_train = feats_all[train_idx]
        if variant == "unconditional_null_chunk":
            sampled = sample_unconditional(rng, chunk_train, n)
        elif variant == "conditional_null_chunk_phi_back_mean":
            sampled = sample_conditional_bin(rng, chunk_train, feats_train[:, 0], feats_all[:, 0], N_BINS)
        elif variant == "conditional_null_chunk_phi_back_phi2":
            sampled = sample_conditional_bin(rng, chunk_train, feats_train[:, 1], feats_all[:, 1], N_BINS)
        elif variant == "nearest_neighbor_null_chunk":
            sampled = sample_nearest_neighbor(rng, chunk_train, feats_train, feats_all)
        else:
            raise ValueError(variant)
        u[:, start:stop] = sampled
        rows.append(
            {
                "variant": variant,
                "group_size": group_size,
                "chunk_start": start,
                "chunk_stop": stop,
                "chunk_blocks": ",".join(str(int(x)) for x in chunk_block_ids(start, stop)),
                "train_chunk_mean_norm": float(np.linalg.norm(chunk_train.mean(axis=0))),
                "train_chunk_rms": rms(chunk_train),
                "sample_chunk_rms": rms(sampled),
                "feature_back_mean_std": float(feats_train[:, 0].std()),
                "feature_back_phi2_std": float(feats_train[:, 1].std()),
            }
        )
    phi = back + (u @ q_basis.T).reshape(n, 16, 16)
    return phi, u, rows


def coord_statistics(label: str, phase: str, u: np.ndarray) -> dict[str, Any]:
    centered = u - u.mean(axis=0, keepdims=True)
    var = centered.var(axis=0)
    fourth = np.mean(centered**4, axis=0)
    kurt = fourth / np.maximum(var**2, 1.0e-30)
    cov = np.cov(centered, rowvar=False)
    evals = np.linalg.eigvalsh(cov)
    off = cov - np.diag(np.diag(cov))
    return {
        "ensemble": label,
        "phase": phase,
        "coord_mean_rms": rms(u.mean(axis=0)),
        "coord_std_mean": float(np.mean(np.sqrt(np.maximum(var, 0.0)))),
        "coord_std_min": float(np.min(np.sqrt(np.maximum(var, 0.0)))),
        "coord_std_max": float(np.max(np.sqrt(np.maximum(var, 0.0)))),
        "mean_marginal_kurtosis": float(np.mean(kurt)),
        "median_marginal_kurtosis": float(np.median(kurt)),
        "cov_eig_min": float(np.min(evals)),
        "cov_eig_max": float(np.max(evals)),
        "cov_offdiag_rms": rms(off),
    }


def simple_block_residual(phi: np.ndarray, back: np.ndarray) -> tuple[float, float]:
    d = block_average_2x2(phi - back)
    return rms(d), float(np.max(np.abs(d)))


def low_diag(label: str, phase: str, phi: np.ndarray, back: np.ndarray) -> dict[str, Any]:
    delta = phi - back
    obs = observables(phi)
    terms = pmin_cross_terms(back, delta)
    ft = np.fft.fftn(phi, axes=(-2, -1))
    vol = phi.shape[-1] * phi.shape[-2]
    return {
        "ensemble": label,
        "phase": phase,
        "S0": float(np.mean(np.abs(ft[:, 0, 0]) ** 2) / vol),
        "S_pmin": float(0.5 * (np.mean(np.abs(ft[:, 1, 0]) ** 2) + np.mean(np.abs(ft[:, 0, 1]) ** 2)) / vol),
        **terms,
        "Binder_U4": obs["Binder_U4"],
        "xi_over_L": obs["xi_over_L"],
    }


def load_exact_reference(back: np.ndarray, coarse: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    path = EMP_HAAR / "exact_null_local_chunk_100_sweeps_reference.npy"
    if not path.exists():
        path = BENCH / "samples_sweeps_100.npy"
    if not path.exists():
        return None
    phi = np.load(path).astype(np.float64)
    summary = BENCH / "summary.json"
    if summary.exists():
        idx = np.asarray(json.loads(summary.read_text()).get("selected_indices", []), dtype=int)
        if len(idx) >= len(phi):
            return phi, back[idx[: len(phi)]], coarse[idx[: len(phi)]]
    return phi, back[: len(phi)], coarse[: len(phi)]


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
    splits = np.load(DATA / "split_indices.npz")
    train_idx = splits["train"].astype(int)
    residual = fine - back
    u_true = residual.reshape(len(fine), -1) @ q
    recon = back + (u_true @ q.T).reshape(len(fine), 16, 16)
    b_res = block_sym(residual, w, block_norm)
    back_feats = block_feature_tables(back, coarse)

    obs_rows: list[dict[str, Any]] = []
    score_base_rows: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    coord_rows: list[dict[str, Any]] = []
    low_rows: list[dict[str, Any]] = []
    spectrum_rows: list[dict[str, Any]] = []
    library_rows: list[dict[str, Any]] = []
    generated: dict[str, str] = {}

    references: list[tuple[str, str, np.ndarray, np.ndarray, np.ndarray | None]] = [
        ("fine_target", "target", fine, back, coarse),
        ("smooth_backbone", "reference", back, back, coarse),
    ]
    for opt_path, label in [
        (EMP_HAAR / "haar_conditional_fine_block_average.npy", "unprojected_empirical_haar_fine"),
        (PROJECTED_HAAR / "projected_haar_conditional_fine_block_average.npy", "projected_empirical_haar_fine"),
        (UV_SOURCE / "initializer_from_native_L8_kappa0p295_mixed_wolff_local.npy", "native_kappa0p295_haar_unprojected"),
    ]:
        if opt_path.exists():
            references.append((label, "reference", np.load(opt_path).astype(np.float64), back, coarse))
    exact = load_exact_reference(back, coarse)
    if exact is not None:
        references.append(("exact_null_100_sweep_reference", "reference", exact[0], exact[1], exact[2]))

    for label, phase, phi, ref_back, ref_coarse in references:
        obs_rows.append({"ensemble": label, "phase": phase, **observables(phi)})
        score_base_rows.append(obs_rows[-1])
        low_rows.append(low_diag(label, phase, phi, ref_back))
        spectrum_rows.extend(low_momentum_rows(label, phase, phi))
        if ref_coarse is not None and phi.shape[-2:] == (16, 16):
            br = block_sym(phi, w, block_norm) - ref_coarse
            sd_rms, sd_max = simple_block_residual(phi, ref_back)
            block_rows.append(
                {
                    "ensemble": label,
                    "phase": phase,
                    "Bsym_residual_rms_vs_phi_c": rms(br),
                    "Bsym_residual_max_vs_phi_c": float(np.max(np.abs(br))),
                    "simple_block_avg_delta_rms": sd_rms,
                    "simple_block_avg_delta_max": sd_max,
                }
            )

    variants = [
        "unconditional_null_chunk",
        "conditional_null_chunk_phi_back_mean",
        "conditional_null_chunk_phi_back_phi2",
        "nearest_neighbor_null_chunk",
    ]
    for group_size in GROUP_SIZES:
        for variant in variants:
            rng = np.random.default_rng(SEED + 1000 * group_size + 17 * variants.index(variant))
            phi, u, lib_rows = make_initializer(
                variant=variant,
                group_size=group_size,
                u_true=u_true,
                train_idx=train_idx,
                back_feats=back_feats,
                back=back,
                q_basis=q,
                rng=rng,
            )
            label = f"{variant}_G{group_size}"
            path = out / f"{label}.npy"
            np.save(path, phi.astype(np.float32))
            generated[label] = str(path)
            obs = {"ensemble": label, "phase": "generated", "variant": variant, "group_size": group_size, **observables(phi)}
            obs_rows.append(obs)
            score_base_rows.append(obs)
            br = block_sym(phi, w, block_norm) - coarse
            sd_rms, sd_max = simple_block_residual(phi, back)
            block_rows.append(
                {
                    "ensemble": label,
                    "phase": "generated",
                    "variant": variant,
                    "group_size": group_size,
                    "Bsym_residual_rms_vs_phi_c": rms(br),
                    "Bsym_residual_max_vs_phi_c": float(np.max(np.abs(br))),
                    "simple_block_avg_delta_rms": sd_rms,
                    "simple_block_avg_delta_max": sd_max,
                    "delta_UV_rms": rms(phi - back),
                }
            )
            coord_rows.append(coord_statistics(label, "generated", u))
            low_rows.append(low_diag(label, "generated", phi, back))
            spectrum_rows.extend(low_momentum_rows(label, "generated", phi))
            library_rows.extend(lib_rows)

    coord_rows.append(coord_statistics("true_residual", "target", u_true))
    scores = sorted(score_rows(score_base_rows, observables(fine)), key=lambda r: float(r["local_relative_L1"]))

    write_csv(out / "initializer_observables.csv", obs_rows)
    write_csv(out / "scores.csv", scores)
    write_csv(out / "blocking_residuals.csv", block_rows)
    write_csv(out / "null_coordinate_statistics.csv", coord_rows)
    write_csv(out / "low_momentum_diagnostics.csv", low_rows)
    write_csv(out / "low_momentum_spectrum.csv", spectrum_rows)
    write_csv(out / "chunk_library_summary.csv", library_rows)

    summary = {
        "canonical_data_dir": str(DATA),
        "kernel_metadata_path": str(PROJECT / "kernels/from_perfect_blocking_lam1p0_blockavg/selected_kernel_metadata.json"),
        "kernel_source": kernel_meta.get("original_source_path"),
        "eta_exponent": kernel_meta.get("eta_exponent", 0.25),
        "block_norm": block_norm,
        "q_basis": str(Q_BASIS),
        "q_shape": list(q.shape),
        "max_abs_Bsym_true_residual": float(np.max(np.abs(b_res))),
        "rms_Bsym_true_residual": rms(b_res),
        "true_reconstruction_max_abs": float(np.max(np.abs(recon - fine))),
        "true_reconstruction_rms": rms(recon - fine),
        "train_size": int(len(train_idx)),
        "group_sizes": GROUP_SIZES,
        "variants": variants,
        "generated_ensembles": generated,
    }
    write_json(out / "library_summary.json", summary)
    write_json(out / "summary.json", summary)
    write_json(out / "config.json", summary)
    shutil.copy2(Path(__file__), out / "empirical_null_coordinate_initializer.py")

    compact_obs = [
        {
            "ensemble": r["ensemble"],
            "phase": r["phase"],
            "phi2": r["phi2"],
            "phi4": r["phi4"],
            "NN": r["NN"],
            "nn2": r["nn2"],
            "Binder_U4": r["Binder_U4"],
            "xi_over_L": r["xi_over_L"],
            "action_density": r["action_density"],
        }
        for r in obs_rows
    ]
    compact_block = [
        {
            "ensemble": r["ensemble"],
            "phase": r["phase"],
            "Bsym_residual_rms_vs_phi_c": r["Bsym_residual_rms_vs_phi_c"],
            "simple_block_avg_delta_rms": r["simple_block_avg_delta_rms"],
        }
        for r in block_rows
    ]
    best_generated = next((r for r in scores if str(r["phase"]) == "generated"), None)
    report = f"""# Empirical Local B_sym-Null UV Initializer

This diagnostic samples empirical UV residual chunks directly in the local
projected-Haar `B_sym`-null coordinate system.

## Provenance

- Canonical paired data: `{DATA}`
- Kernel source: `{kernel_meta.get('original_source_path')}`
- `eta_exponent = {kernel_meta.get('eta_exponent', 0.25)}`
- `block_norm = {block_norm:.16g}`
- Local projected-Haar basis: `{Q_BASIS}`
- `Q` shape: `{q.shape[0]} x {q.shape[1]}`
- Train split size for chunk libraries: `{len(train_idx)}`
- True residual check: RMS `B_sym(r) = {rms(b_res):.3g}`, max `{float(np.max(np.abs(b_res))):.3g}`
- Reconstruction from `phi_back + Q Q^T r`: RMS `{rms(recon - fine):.3g}`, max `{float(np.max(np.abs(recon - fine))):.3g}`

## Best Scores

Lower is better. Primary score uses `phi2, phi4, NN, nn2, diag, 2nn`; `xi/L` is diagnostic only.

{table_md(scores, ['ensemble', 'phase', 'local_relative_L1', 'local_plus_action_relative_L1'], limit=16)}

## Observable Summary

{table_md(compact_obs, ['ensemble', 'phase', 'phi2', 'phi4', 'NN', 'nn2', 'Binder_U4', 'xi_over_L', 'action_density'], limit=28)}

## Blocking Residuals

{table_md(compact_block, ['ensemble', 'phase', 'Bsym_residual_rms_vs_phi_c', 'simple_block_avg_delta_rms'], limit=28)}

## Answers

1. **Can empirical null-coordinate chunks reproduce local observables without B_sym leakage?**
   They preserve `B_sym` to roundoff because all generated details are assembled in the projected-Haar null basis. They do **not** reproduce the local observables well: the best generated row is `{best_generated['ensemble'] if best_generated else 'none'}`, but it still overshoots `phi4`, `nn2`, and action density relative to the fine target.

2. **Which chunk size works best?**
   `G=4` with nearest-neighbor feature matching is best by the primary local score in this scan. Larger chunks did not improve the no-training initializer.

3. **Does conditioning on local phi_back improve over unconditional chunk sampling?**
   Yes, but not enough. Nearest-neighbor conditioning improves substantially over unconditional chunk sampling, and simple feature bins help somewhat. All generated variants remain worse than projected Haar and far worse than exact-null correction.

4. **Does this beat projected Haar?**
   No. It beats the post-hoc projection only in exact block consistency, not in local-observable quality. The best generated null-coordinate row has a worse local score than projected Haar.

5. **Does this approach the exact-null 100-sweep correction without MCMC?**
   No. The exact-null 100-sweep reference remains much closer to the fine target in both local and action scores.

6. **Is the remaining failure mostly inter-chunk correlation, action density, or tails?**
   The leading failure is lost inter-chunk/cross-coordinate structure. Marginal coordinate standard deviations and kurtoses are similar to the true residuals, but independent chunk assembly changes covariance structure and produces too-large `phi4`, `nn2`, and action density. A usable empirical null-coordinate initializer would need to preserve broader correlations, not just local marginal chunks.

## Output Files

- `initializer_observables.csv`
- `scores.csv`
- `blocking_residuals.csv`
- `null_coordinate_statistics.csv`
- `low_momentum_diagnostics.csv`
- `low_momentum_spectrum.csv`
- `chunk_library_summary.csv`
- `library_summary.json`
- `summary.json`
- generated `*.npy` initializer ensembles
- archived script: `empirical_null_coordinate_initializer.py`
"""
    (out / "report.md").write_text(report)


if __name__ == "__main__":
    main()
