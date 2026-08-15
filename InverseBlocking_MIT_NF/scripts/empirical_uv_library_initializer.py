#!/usr/bin/env python3
"""Empirical local UV residual library initializers.

No training is performed. Libraries are harvested from the canonical paired
16x16 fine/backbone data and used to build UV/detail initial fields around
the smooth IR backbone.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
DATA = PROJECT / "outputs/paired_data_lam1_kappaf0p320"
OUT = PROJECT / "outputs/empirical_uv_library_initializer"
KERNEL = PROJECT / "kernels/from_perfect_blocking_lam1p0_blockavg/perfect_block_lam1_blockavg_kernel5x5_kernel.json"
LOCAL_CHUNK_100 = PROJECT / "outputs/inverse_blocking_proposal_benchmark_full/samples_sweeps_100.npy"
LOCAL_CHUNK_SUMMARY = PROJECT / "outputs/inverse_blocking_proposal_benchmark_full/summary.json"
HEID_EXTENDED = (
    PROJECT
    / "experiments/heidelberg_native_kappa0p295_to_16/outputs/amplitude_rescaling_diagnostic/extended_operator_comparison/extended_operator_comparison.csv"
)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def block_average_2x2(phi: np.ndarray) -> np.ndarray:
    n, lf, _ = phi.shape
    return phi.reshape(n, lf // 2, 2, lf // 2, 2).mean(axis=(2, 4))


def haar_decompose_2x2(phi: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x00 = phi[:, 0::2, 0::2]
    x10 = phi[:, 1::2, 0::2]
    x01 = phi[:, 0::2, 1::2]
    x11 = phi[:, 1::2, 1::2]
    a = 0.25 * (x00 + x10 + x01 + x11)
    h = 0.25 * (x00 - x10 + x01 - x11)
    v = 0.25 * (x00 + x10 - x01 - x11)
    d = 0.25 * (x00 - x10 - x01 + x11)
    return a, h, v, d


def haar_reconstruct_2x2(a: np.ndarray, h: np.ndarray, v: np.ndarray, d: np.ndarray) -> np.ndarray:
    n, lc, _ = a.shape
    out = np.empty((n, 2 * lc, 2 * lc), dtype=np.float64)
    out[:, 0::2, 0::2] = a + h + v + d
    out[:, 1::2, 0::2] = a - h + v - d
    out[:, 0::2, 1::2] = a + h - v - d
    out[:, 1::2, 1::2] = a - h - v + d
    return out


def make_zero_sum_noise(coarse_or_avg: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n, lc, _ = coarse_or_avg.shape
    noise = rng.normal(0.0, sigma, size=(n, lc, lc, 2, 2))
    noise -= noise.mean(axis=(3, 4), keepdims=True)
    out = np.repeat(np.repeat(coarse_or_avg, 2, axis=1), 2, axis=2)
    out = out.reshape(n, lc, 2, lc, 2).transpose(0, 1, 3, 2, 4)
    out = out + noise
    return out.transpose(0, 1, 3, 2, 4).reshape(n, 2 * lc, 2 * lc)


def sample_by_quantile_bins(
    target_feature: np.ndarray,
    library_feature: np.ndarray,
    library_details: np.ndarray,
    *,
    n_bins: int,
    rng: np.random.Generator,
) -> np.ndarray:
    flat_target = target_feature.reshape(-1)
    edges = np.quantile(library_feature, np.linspace(0.0, 1.0, n_bins + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf
    lib_bins = np.clip(np.searchsorted(edges, library_feature, side="right") - 1, 0, n_bins - 1)
    target_bins = np.clip(np.searchsorted(edges, flat_target, side="right") - 1, 0, n_bins - 1)
    by_bin = [np.flatnonzero(lib_bins == b) for b in range(n_bins)]
    all_idx = np.arange(len(library_feature))
    sampled = np.empty((len(flat_target), library_details.shape[1]), dtype=np.float64)
    for b in range(n_bins):
        mask = target_bins == b
        if not np.any(mask):
            continue
        pool = by_bin[b] if len(by_bin[b]) else all_idx
        idx = rng.choice(pool, size=int(np.sum(mask)), replace=True)
        sampled[mask] = library_details[idx]
    return sampled.reshape(*target_feature.shape, library_details.shape[1])


def kernel_weights() -> tuple[dict[str, float], float]:
    meta = json.loads(KERNEL.read_text())
    return {k: float(meta["weights"][k]) for k in ["w00", "w10", "w11", "w20", "w21", "w22"]}, float(meta["block_norm"])


def apply_k(phi: np.ndarray, w: dict[str, float]) -> np.ndarray:
    y = w["w00"] * phi
    shells = {
        "w10": [(1, 0), (-1, 0), (0, 1), (0, -1)],
        "w11": [(1, 1), (1, -1), (-1, 1), (-1, -1)],
        "w20": [(2, 0), (-2, 0), (0, 2), (0, -2)],
        "w21": [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)],
        "w22": [(2, 2), (2, -2), (-2, 2), (-2, -2)],
    }
    for key, shifts in shells.items():
        acc = np.zeros_like(phi)
        for dy, dx in shifts:
            acc += np.roll(np.roll(phi, dy, axis=-2), dx, axis=-1)
        y += w[key] * acc
    return y


def block_sym(phi: np.ndarray, w: dict[str, float], block_norm: float) -> np.ndarray:
    psi = apply_k(phi, w)
    return block_norm * 0.25 * (
        psi[:, 0::2, 0::2] + psi[:, 1::2, 0::2] + psi[:, 0::2, 1::2] + psi[:, 1::2, 1::2]
    )


def observables(phi: np.ndarray, *, kappa: float, lam: float) -> dict[str, float]:
    arr = np.asarray(phi, dtype=np.float64)
    n, ly, lx = arr.shape
    vol = ly * lx
    m_cfg = arr.mean(axis=(-2, -1))
    m2 = float(np.mean(m_cfg**2))
    m4 = float(np.mean(m_cfg**4))
    b4 = m4 / (m2 * m2) if m2 > 0 else math.nan
    u4 = 1.0 - b4 / 3.0 if math.isfinite(b4) else math.nan
    nn = 0.5 * (
        np.mean(arr * np.roll(arr, -1, axis=-2), axis=(-2, -1))
        + np.mean(arr * np.roll(arr, -1, axis=-1), axis=(-2, -1))
    )
    nn2 = 0.5 * (
        np.mean((arr * np.roll(arr, -1, axis=-2)) ** 2, axis=(-2, -1))
        + np.mean((arr * np.roll(arr, -1, axis=-1)) ** 2, axis=(-2, -1))
    )
    diag = 0.5 * (
        np.mean(arr * np.roll(np.roll(arr, -1, axis=-2), -1, axis=-1), axis=(-2, -1))
        + np.mean(arr * np.roll(np.roll(arr, -1, axis=-2), 1, axis=-1), axis=(-2, -1))
    )
    twonn = 0.5 * (
        np.mean(arr * np.roll(arr, -2, axis=-2), axis=(-2, -1))
        + np.mean(arr * np.roll(arr, -2, axis=-1), axis=(-2, -1))
    )
    ft = np.fft.fftn(arr, axes=(-2, -1))
    chi = vol * np.mean(m_cfg**2)
    fmin = 0.5 * (np.mean(np.abs(ft[:, 1, 0]) ** 2) + np.mean(np.abs(ft[:, 0, 1]) ** 2)) / vol
    xi = math.nan
    if fmin > 0 and chi / fmin > 1.0:
        xi = float(0.5 / np.sin(np.pi / lx) * np.sqrt(chi / fmin - 1.0))
    phi2 = float(np.mean(arr**2))
    phi4 = float(np.mean(arr**4))
    hopping = float(-4.0 * kappa * np.mean(nn))
    action_density = phi2 + lam * (phi4 - 2.0 * phi2 + 1.0) + hopping
    return {
        "N": int(n),
        "L": int(lx),
        "m": float(np.mean(m_cfg)),
        "abs_m": float(np.mean(np.abs(m_cfg))),
        "phi2": phi2,
        "phi4": phi4,
        "NN": float(np.mean(nn)),
        "nn2": float(np.mean(nn2)),
        "diag": float(np.mean(diag)),
        "2nn": float(np.mean(twonn)),
        "Binder_U4": float(u4),
        "Binder_ratio_B4": float(b4),
        "xi": float(xi),
        "xi_over_L": float(xi / lx) if math.isfinite(xi) else math.nan,
        "action_hopping_density": hopping,
        "action_phi2_density": float((1.0 - 2.0 * lam) * phi2),
        "action_phi4_density": float(lam * phi4),
        "action_density": float(action_density),
    }


def low_momentum_rows(label: str, phi: np.ndarray) -> list[dict[str, Any]]:
    arr = np.asarray(phi, dtype=np.float64)
    n, ly, lx = arr.shape
    ft = np.fft.fftn(arr, axes=(-2, -1))
    volume = ly * lx
    modes = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (0, 2)]
    rows = []
    for ky, kx in modes:
        rows.append(
            {
                "ensemble": label,
                "L": int(lx),
                "mode_index_y": ky,
                "mode_index_x": kx,
                "p_y_units_2pi_over_L": ky,
                "p_x_units_2pi_over_L": kx,
                "S_p": float(np.mean(np.abs(ft[:, ky % ly, kx % lx]) ** 2) / volume),
            }
        )
    return rows


def extract_patch(arr: np.ndarray, y: int, x: int, size: int = 4) -> np.ndarray:
    ys = (np.arange(y, y + size) % arr.shape[-2]).astype(int)
    xs = (np.arange(x, x + size) % arr.shape[-1]).astype(int)
    return arr[np.ix_(ys, xs)]


def patch_library_initializer(fine: np.ndarray, back: np.ndarray, *, seed: int, n_query: int = 16) -> tuple[np.ndarray, dict[str, Any]]:
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(seed)
    residual = fine - back
    n, lf, _ = fine.shape
    lc = lf // 2
    features = []
    cfg_ids = []
    coords = []
    for c in range(n):
        for i in range(lc):
            for j in range(lc):
                y, x = 2 * i, 2 * j
                p = extract_patch(back[c], y, x)
                features.append([p.mean(), p.std(), np.mean(p * p), back[c, y, x], back[c, (y + 1) % lf, (x + 1) % lf]])
                cfg_ids.append(c)
                coords.append((c, y, x))
    feat = np.asarray(features, dtype=np.float64)
    cfg_ids_arr = np.asarray(cfg_ids, dtype=int)
    mean = feat.mean(axis=0)
    std = feat.std(axis=0)
    z = (feat - mean) / np.maximum(std, 1.0e-12)
    tree = cKDTree(z)
    dists, inds = tree.query(z, k=min(n_query, len(z)))
    if inds.ndim == 1:
        inds = inds[:, None]
    out_res = np.zeros_like(residual)
    counts = np.zeros_like(residual)
    chosen = np.empty(len(z), dtype=int)
    for row, cand in enumerate(inds):
        allowed = [int(idx) for idx in cand if cfg_ids_arr[int(idx)] != cfg_ids_arr[row]]
        if not allowed:
            allowed = [int(cand[0])]
        pick = int(rng.choice(allowed))
        chosen[row] = pick
        target_c, target_y, target_x = coords[row]
        src_c, src_y, src_x = coords[pick]
        patch = extract_patch(residual[src_c], src_y, src_x)
        for dy in range(4):
            for dx in range(4):
                yy = (target_y + dy) % lf
                xx = (target_x + dx) % lf
                out_res[target_c, yy, xx] += patch[dy, dx]
                counts[target_c, yy, xx] += 1.0
    out = back + out_res / np.maximum(counts, 1.0)
    diag = {
        "library_size": int(len(z)),
        "feature_dim": int(feat.shape[1]),
        "mean_query_distance_first_neighbor": float(np.mean(dists[:, 0] if np.ndim(dists) == 2 else dists)),
        "fraction_same_config_after_filter": float(np.mean(cfg_ids_arr[chosen] == cfg_ids_arr[np.arange(len(chosen))])),
    }
    np.save(OUT / "residual_patch_library_features.npy", feat)
    np.save(OUT / "residual_patch_sampled_indices.npy", chosen)
    return out, diag


def external_heidelberg_rows() -> list[dict[str, Any]]:
    if not HEID_EXTENDED.exists():
        return []
    keep = {"zero_sum_initialization_sigma0p15", "heidelberg_checkpoint_step_0000", "heidelberg_checkpoint_step_0199"}
    rows = []
    with HEID_EXTENDED.open() as fh:
        for row in csv.DictReader(fh):
            if row.get("ensemble") in keep and row.get("scale") == "1.0":
                rows.append({"ensemble": "external_" + row["ensemble"], "source": str(HEID_EXTENDED), **row})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260624)
    ap.add_argument("--kappa", type=float, default=0.320)
    ap.add_argument("--lambda", dest="lam", type=float, default=1.0)
    ap.add_argument("--n-bins", type=int, default=32)
    ap.add_argument("--zero-sum-sigma", type=float, default=0.15)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    fine = np.load(DATA / "fine_configs.npy").astype(np.float64)
    coarse = np.load(DATA / "coarse_blocked_configs.npy").astype(np.float64)
    back = np.load(DATA / "backbone_configs.npy").astype(np.float64)
    metadata = json.loads((DATA / "generation_metadata.json").read_text())
    w, block_norm = kernel_weights()
    rng = np.random.default_rng(args.seed)

    a_f, h_f, v_f, d_f = haar_decompose_2x2(fine)
    a_b, _, _, _ = haar_decompose_2x2(back)
    detail_lib = np.column_stack([h_f.reshape(-1), v_f.reshape(-1), d_f.reshape(-1)])
    fine_a_lib = a_f.reshape(-1)
    back_a_lib = a_b.reshape(-1)
    back_amp_lib = block_average_2x2(back * back).reshape(-1)
    np.savez_compressed(
        OUT / "haar_detail_library.npz",
        fine_block_average=fine_a_lib,
        backbone_block_average=back_a_lib,
        backbone_block_phi2=back_amp_lib,
        details_hvd=detail_lib,
    )

    target_a = a_b
    samples: dict[str, dict[str, Any]] = {
        "fine_target": {"phi": fine, "coarse_for_block": coarse, "back_for_simple": back, "block_residual_note": "canonical_order"},
        "smooth_backbone": {"phi": back, "coarse_for_block": coarse, "back_for_simple": back, "block_residual_note": "canonical_order"},
        "zero_sum_gaussian_on_backbone_blockavg_sigma0p15": {
            "phi": make_zero_sum_noise(target_a, args.zero_sum_sigma, args.seed + 1),
            "coarse_for_block": coarse,
            "back_for_simple": back,
            "block_residual_note": "canonical_order",
        },
    }
    idx = rng.integers(0, len(detail_lib), size=(*target_a.shape,))
    details = detail_lib[idx]
    samples["haar_unconditional"] = {
        "phi": haar_reconstruct_2x2(target_a, details[..., 0], details[..., 1], details[..., 2]),
        "coarse_for_block": coarse,
        "back_for_simple": back,
        "block_residual_note": "canonical_order",
    }

    details = sample_by_quantile_bins(target_a, fine_a_lib, detail_lib, n_bins=args.n_bins, rng=rng)
    samples["haar_conditional_fine_block_average"] = {
        "phi": haar_reconstruct_2x2(target_a, details[..., 0], details[..., 1], details[..., 2]),
        "coarse_for_block": coarse,
        "back_for_simple": back,
        "block_residual_note": "canonical_order",
    }

    target_back_amp = block_average_2x2(back * back)
    details = sample_by_quantile_bins(target_back_amp, back_amp_lib, detail_lib, n_bins=args.n_bins, rng=rng)
    samples["haar_conditional_backbone_phi2"] = {
        "phi": haar_reconstruct_2x2(target_a, details[..., 0], details[..., 1], details[..., 2]),
        "coarse_for_block": coarse,
        "back_for_simple": back,
        "block_residual_note": "canonical_order",
    }

    patch_samples, patch_diag = patch_library_initializer(fine, back, seed=args.seed + 3)
    samples["residual_patch_nearest_neighbor_4x4"] = {
        "phi": patch_samples,
        "coarse_for_block": coarse,
        "back_for_simple": back,
        "block_residual_note": "canonical_order",
    }

    if LOCAL_CHUNK_100.exists():
        local_chunk = np.load(LOCAL_CHUNK_100).astype(np.float64)
        local_coarse = None
        note = "external_reference_no_condition_alignment"
        if LOCAL_CHUNK_SUMMARY.exists():
            selected = np.asarray(json.loads(LOCAL_CHUNK_SUMMARY.read_text()).get("selected_indices", []), dtype=int)
            if len(selected) == len(local_chunk):
                local_coarse = coarse[selected]
                local_back = back[selected]
                note = "aligned_to_benchmark_selected_indices"
            else:
                local_back = None
        else:
            local_back = None
        samples["exact_null_local_chunk_100_sweeps_reference"] = {
            "phi": local_chunk,
            "coarse_for_block": local_coarse,
            "back_for_simple": local_back,
            "block_residual_note": note,
        }

    obs_rows = []
    block_rows = []
    low_rows = []
    for label, item in samples.items():
        arr = np.asarray(item["phi"], dtype=np.float64)
        np.save(OUT / f"{label}.npy", arr)
        row = {"ensemble": label, **observables(arr, kappa=args.kappa, lam=args.lam)}
        obs_rows.append(row)
        coarse_for_block = item.get("coarse_for_block")
        if coarse_for_block is not None and len(arr) == len(coarse_for_block):
            bsym = block_sym(arr, w, block_norm)
            diff = bsym - np.asarray(coarse_for_block, dtype=np.float64)
            back_for_simple = item.get("back_for_simple")
            simple_diff = (
                block_average_2x2(arr) - block_average_2x2(np.asarray(back_for_simple, dtype=np.float64))
                if back_for_simple is not None
                else np.full_like(block_average_2x2(arr), np.nan)
            )
            block_rows.append(
                {
                    "ensemble": label,
                    "block_residual_note": item["block_residual_note"],
                    "Bsym_rms": float(np.sqrt(np.mean(diff * diff))),
                    "Bsym_max": float(np.max(np.abs(diff))),
                    "Bsym_relative_rms": float(np.sqrt(np.mean(diff * diff)) / max(np.sqrt(np.mean(np.asarray(coarse_for_block) ** 2)), 1.0e-30)),
                    "simple_blockavg_vs_backbone_blockavg_rms": float(np.sqrt(np.mean(simple_diff * simple_diff))),
                    "simple_blockavg_vs_backbone_blockavg_max": float(np.max(np.abs(simple_diff))),
                }
            )
        low_rows.extend(low_momentum_rows(label, arr))

    write_csv(OUT / "initializer_observables.csv", obs_rows)
    write_csv(OUT / "block_residuals.csv", block_rows)
    write_csv(OUT / "low_momentum_spectrum.csv", low_rows)
    write_csv(OUT / "external_reference_rows.csv", external_heidelberg_rows())

    fine_row = next(r for r in obs_rows if r["ensemble"] == "fine_target")
    candidate_rows = [r for r in obs_rows if r["ensemble"] not in {"fine_target"}]
    for r in candidate_rows:
        r["local_moment_score_phi2_phi4_nn2"] = (
            abs(float(r["phi2"]) - float(fine_row["phi2"]))
            + abs(float(r["phi4"]) - float(fine_row["phi4"]))
            + abs(float(r["nn2"]) - float(fine_row["nn2"]))
        )
    best = min(candidate_rows, key=lambda r: float(r["local_moment_score_phi2_phi4_nn2"]))
    library_initializer_names = {
        "haar_unconditional",
        "haar_conditional_fine_block_average",
        "haar_conditional_backbone_phi2",
        "residual_patch_nearest_neighbor_4x4",
    }
    initializer_rows = [r for r in candidate_rows if r["ensemble"] in library_initializer_names]
    best_initializer = min(initializer_rows, key=lambda r: float(r["local_moment_score_phi2_phi4_nn2"]))
    write_csv(OUT / "initializer_scores.csv", candidate_rows)

    summary = {
        "canonical_data": str(DATA),
        "n_configs": int(len(fine)),
        "lattice": [16, 16],
        "generation_metadata": metadata,
        "standing_rule_note": "No new ensembles were generated. The canonical paired data predates the Wolff-only standing rule and preserves its original metadata.",
        "kernel": str(KERNEL),
        "block_norm": block_norm,
        "haar_library_size": int(len(detail_lib)),
        "haar_features": ["fine_block_average", "backbone_block_average", "backbone_block_phi2", "h", "v", "d"],
        "patch_library": patch_diag,
        "best_candidate_by_phi2_phi4_nn2_including_references": best,
        "best_empirical_library_initializer_by_phi2_phi4_nn2": best_initializer,
    }
    write_json(OUT / "summary.json", summary)
    write_json(
        OUT / "library_summary.json",
        {
            "haar_detail_library_npz": str(OUT / "haar_detail_library.npz"),
            "haar_library_size": int(len(detail_lib)),
            "detail_means_hvd": np.mean(detail_lib, axis=0).tolist(),
            "detail_stds_hvd": np.std(detail_lib, axis=0).tolist(),
            "fine_block_average_mean": float(np.mean(fine_a_lib)),
            "fine_block_average_std": float(np.std(fine_a_lib)),
            "backbone_block_phi2_mean": float(np.mean(back_amp_lib)),
            "backbone_block_phi2_std": float(np.std(back_amp_lib)),
            "residual_patch_features_npy": str(OUT / "residual_patch_library_features.npy"),
            "residual_patch_sampled_indices_npy": str(OUT / "residual_patch_sampled_indices.npy"),
            **patch_diag,
        },
    )

    def md_table(rows: list[dict[str, Any]], cols: list[str]) -> str:
        lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
        for row in rows:
            vals = []
            for col in cols:
                val = row.get(col, "")
                if isinstance(val, float):
                    vals.append(f"{val:.6g}")
                else:
                    vals.append(str(val))
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    report_rows = [fine_row] + sorted(candidate_rows, key=lambda r: float(r["local_moment_score_phi2_phi4_nn2"]))
    report = f"""# Empirical Local UV Residual Library Initializer

No training was run. No new ensembles were generated.

The libraries were harvested from the canonical paired data:

`{DATA}`

Metadata caveat: this canonical paired fine ensemble predates the Wolff-only standing rule. Its saved metadata reports local Metropolis-style generation parameters, and this diagnostic preserves that provenance rather than relabeling it.

## Libraries

- Haar blocks: `{len(detail_lib)}` 2x2 blocks with `(a,h,v,d)` and backbone/local-amplitude features.
- Residual patches: 4x4 residual patches from `r = phi_f - phi_back`, matched by nearest neighbor in simple backbone-patch feature space while avoiding same-source configurations when possible.
- Patch same-config fraction after filtering: `{patch_diag['fraction_same_config_after_filter']:.6g}`.

## Observable Comparison

{md_table(report_rows, ["ensemble", "phi2", "phi4", "NN", "nn2", "diag", "2nn", "Binder_U4", "xi_over_L", "action_density", "local_moment_score_phi2_phi4_nn2"])}

## Block Residuals

{md_table(block_rows, ["ensemble", "Bsym_rms", "Bsym_max", "Bsym_relative_rms", "simple_blockavg_vs_backbone_blockavg_rms"])}

## Interpretation

1. The empirical Haar initializers inject local UV fluctuations around the smooth IR backbone without training.
2. The best empirical library initializer by the simple `phi2/phi4/nn2` score is `{best_initializer['ensemble']}`. The exact-null 100-sweep row is still better, but it is a constrained-correction reference, not a no-training library initializer.
3. The Haar variants preserve the simple 2x2 block average of `phi_back` by construction, but they do not enforce the exact symmetric `B_sym` map.
4. The patch-residual nearest-neighbor variant uses structured residual patches and can change simple block averages; its block residual is measured rather than constrained.
5. Binder and `xi/L` should be read as IR diagnostics. Any candidate that substantially damages them is not a good IR-preserving initializer even if local moments improve.
6. This is an initializer/library diagnostic, not a sampler and not a trained CNF.

## Output Files

- `haar_detail_library.npz`
- `residual_patch_library_features.npy`
- `residual_patch_sampled_indices.npy`
- `initializer_observables.csv`
- `initializer_scores.csv`
- `block_residuals.csv`
- `low_momentum_spectrum.csv`
- generated sample arrays `*.npy`
"""
    (OUT / "report.md").write_text(report)
    print(json.dumps({"output": str(OUT), "best": best["ensemble"], "best_score": best["local_moment_score_phi2_phi4_nn2"]}, indent=2))


if __name__ == "__main__":
    main()
