#!/usr/bin/env python3
"""Compare Haar UV libraries harvested from fine, small-volume, and coarse sources."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from empirical_uv_library_initializer import (
    block_average_2x2,
    block_sym,
    haar_decompose_2x2,
    haar_reconstruct_2x2,
    kernel_weights,
    low_momentum_rows,
    make_zero_sum_noise,
    observables,
    sample_by_quantile_bins,
)


PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "outputs/uv_library_source_comparison"
DATA = PROJECT / "outputs/paired_data_lam1_kappaf0p320"
EMPIRICAL = PROJECT / "outputs/empirical_uv_library_initializer"
BENCH = PROJECT / "outputs/inverse_blocking_proposal_benchmark_full"
COARSE_CAL = PROJECT / "outputs/coarse_distribution_calibration"


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


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def kurtosis(x: np.ndarray) -> float:
    y = x - np.mean(x)
    s = np.std(y)
    return float(np.mean(y**4) / max(s**4, 1.0e-30))


def source_definitions() -> list[dict[str, Any]]:
    return [
        {
            "source": "fine16_target_oracle",
            "category": "fine_16x16_oracle",
            "path": DATA / "fine_configs.npy",
            "metadata_path": DATA / "generation_metadata.json",
            "lambda": 1.0,
            "kappa": 0.320,
            "provenance": "canonical target fine data; oracle-style UV source; predates Wolff-only standing rule",
        },
        {
            "source": "small_volume_L8_kappa0p320_existing_nonproduction",
            "category": "small_volume_fine_proxy",
            "path": COARSE_CAL / "generated_native_scan/native_coarse_lam1_kappa0p32_L8_nonproduction.npy",
            "metadata_path": COARSE_CAL / "generated_native_scan/native_coarse_lam1_kappa0p32_summary.json",
            "lambda": 1.0,
            "kappa": 0.320,
            "provenance": "existing L8 native/nonproduction scan at same kappa; local Metropolis metadata, not Wolff-only",
        },
        {
            "source": "native_L8_kappa0p295_mixed_wolff_local",
            "category": "native_coarse",
            "path": COARSE_CAL / "generated_native_wolff/native_coarse_lam1_kappa0p295_L8_wolff.npy",
            "metadata_path": COARSE_CAL / "generated_native_wolff/native_coarse_lam1_kappa0p295_L8_wolff_summary.json",
            "lambda": 1.0,
            "kappa": 0.295,
            "provenance": "existing native coarse ensemble; embedded Wolff sign-cluster plus local Metropolis amplitude updates, not Wolff-only",
        },
        {
            "source": "nearby_L8_kappa0p280_extended_nonproduction",
            "category": "nearby_coupling_optional",
            "path": COARSE_CAL / "generated_native_extended/native_coarse_lam1_kappa0p28_L8_extended.npy",
            "metadata_path": COARSE_CAL / "generated_native_extended/native_coarse_lam1_kappa0p28_L8_extended_summary.json",
            "lambda": 1.0,
            "kappa": 0.280,
            "provenance": "existing extended native L8 diagnostic; metadata preserved, not newly generated",
        },
        {
            "source": "nearby_L8_kappa0p290_extended_nonproduction",
            "category": "nearby_coupling_optional",
            "path": COARSE_CAL / "generated_native_extended/native_coarse_lam1_kappa0p29_L8_extended.npy",
            "metadata_path": COARSE_CAL / "generated_native_extended/native_coarse_lam1_kappa0p29_L8_extended_summary.json",
            "lambda": 1.0,
            "kappa": 0.290,
            "provenance": "existing extended native L8 diagnostic; metadata preserved, not newly generated",
        },
        {
            "source": "nearby_L8_kappa0p310_existing_nonproduction",
            "category": "nearby_coupling_optional",
            "path": COARSE_CAL / "generated_native_scan/native_coarse_lam1_kappa0p31_L8_nonproduction.npy",
            "metadata_path": COARSE_CAL / "generated_native_scan/native_coarse_lam1_kappa0p31_summary.json",
            "lambda": 1.0,
            "kappa": 0.310,
            "provenance": "existing L8 native/nonproduction scan; local Metropolis metadata, not Wolff-only",
        },
    ]


def library_from_source(src: dict[str, Any]) -> dict[str, Any]:
    arr = np.load(src["path"]).astype(np.float64)
    a, h, v, d = haar_decompose_2x2(arr)
    details = np.column_stack([h.reshape(-1), v.reshape(-1), d.reshape(-1)])
    feature = a.reshape(-1)
    meta = load_json(src["metadata_path"])
    return {
        **src,
        "configs": arr,
        "metadata": meta,
        "block_average": feature,
        "details_hvd": details,
        "n_configs": int(arr.shape[0]),
        "L": int(arr.shape[-1]),
        "n_blocks": int(len(feature)),
    }


def covariance_entries(details: np.ndarray) -> dict[str, float]:
    cov = np.cov(details, rowvar=False)
    return {
        "cov_hh": float(cov[0, 0]),
        "cov_vv": float(cov[1, 1]),
        "cov_dd": float(cov[2, 2]),
        "cov_hv": float(cov[0, 1]),
        "cov_hd": float(cov[0, 2]),
        "cov_vd": float(cov[1, 2]),
    }


def library_statistics(lib: dict[str, Any]) -> dict[str, Any]:
    details = lib["details_hvd"]
    a = lib["block_average"]
    row = {
        "source": lib["source"],
        "category": lib["category"],
        "path": str(lib["path"]),
        "lambda": lib["lambda"],
        "kappa": lib["kappa"],
        "L": lib["L"],
        "n_configs": lib["n_configs"],
        "n_blocks": lib["n_blocks"],
        "block_average_mean": float(np.mean(a)),
        "block_average_std": float(np.std(a)),
        "h_mean": float(np.mean(details[:, 0])),
        "v_mean": float(np.mean(details[:, 1])),
        "d_mean": float(np.mean(details[:, 2])),
        "h_var": float(np.var(details[:, 0])),
        "v_var": float(np.var(details[:, 1])),
        "d_var": float(np.var(details[:, 2])),
        "h_kurtosis": kurtosis(details[:, 0]),
        "v_kurtosis": kurtosis(details[:, 1]),
        "d_kurtosis": kurtosis(details[:, 2]),
        "provenance": lib["provenance"],
    }
    row.update(covariance_entries(details))
    return row


def detail_distribution_rows(lib: dict[str, Any], *, edges: np.ndarray) -> list[dict[str, Any]]:
    details = lib["details_hvd"]
    a = lib["block_average"]
    rows = []
    for idx, name in enumerate(["h", "v", "d"]):
        rows.append(
            {
                "source": lib["source"],
                "row_type": "global",
                "coordinate": name,
                "bin": "all",
                "n": int(len(a)),
                "a_low": math.nan,
                "a_high": math.nan,
                "mean": float(np.mean(details[:, idx])),
                "variance": float(np.var(details[:, idx])),
                "std": float(np.std(details[:, idx])),
                "kurtosis": kurtosis(details[:, idx]),
            }
        )
    bins = np.clip(np.searchsorted(edges, a, side="right") - 1, 0, len(edges) - 2)
    for b in range(len(edges) - 1):
        mask = bins == b
        if not np.any(mask):
            continue
        for idx, name in enumerate(["h", "v", "d"]):
            vals = details[mask, idx]
            rows.append(
                {
                    "source": lib["source"],
                    "row_type": "conditional_by_a_bin",
                    "coordinate": name,
                    "bin": int(b),
                    "n": int(np.sum(mask)),
                    "a_low": float(edges[b]),
                    "a_high": float(edges[b + 1]),
                    "mean": float(np.mean(vals)),
                    "variance": float(np.var(vals)),
                    "std": float(np.std(vals)),
                    "kurtosis": kurtosis(vals) if len(vals) > 3 else math.nan,
                }
            )
    return rows


def build_initializer(
    lib: dict[str, Any],
    target_block_average: np.ndarray,
    *,
    n_bins: int,
    rng: np.random.Generator,
) -> np.ndarray:
    sampled = sample_by_quantile_bins(
        target_block_average,
        lib["block_average"],
        lib["details_hvd"],
        n_bins=n_bins,
        rng=rng,
    )
    return haar_reconstruct_2x2(target_block_average, sampled[..., 0], sampled[..., 1], sampled[..., 2])


def score_row(row: dict[str, Any], fine: dict[str, Any]) -> dict[str, float]:
    local_ops = ["phi2", "phi4", "nn2"]
    broader_ops = ["phi2", "phi4", "NN", "nn2", "diag", "2nn"]
    ir_ops = ["Binder_U4", "xi_over_L"]
    return {
        "local_moment_score_phi2_phi4_nn2": float(sum(abs(float(row[k]) - float(fine[k])) for k in local_ops)),
        "relative_local_score_phi2_phi4_nn2": float(sum(abs(float(row[k]) - float(fine[k])) / max(abs(float(fine[k])), 1.0e-30) for k in local_ops)),
        "broader_operator_relative_score": float(sum(abs(float(row[k]) - float(fine[k])) / max(abs(float(fine[k])), 1.0e-30) for k in broader_ops)),
        "ir_relative_score": float(sum(abs(float(row[k]) - float(fine[k])) / max(abs(float(fine[k])), 1.0e-30) for k in ir_ops)),
    }


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260624)
    ap.add_argument("--n-bins", type=int, default=32)
    ap.add_argument("--lambda", dest="lam", type=float, default=1.0)
    ap.add_argument("--kappa", type=float, default=0.320)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    fine = np.load(DATA / "fine_configs.npy").astype(np.float64)
    back = np.load(DATA / "backbone_configs.npy").astype(np.float64)
    coarse = np.load(DATA / "coarse_blocked_configs.npy").astype(np.float64)
    target_a = block_average_2x2(back)
    w, block_norm = kernel_weights()

    libs = [library_from_source(src) for src in source_definitions() if Path(src["path"]).exists()]
    lib_stats = [library_statistics(lib) for lib in libs]
    write_csv(OUT / "library_statistics.csv", lib_stats)

    fine_lib = next(lib for lib in libs if lib["source"] == "fine16_target_oracle")
    edges = np.quantile(fine_lib["block_average"], np.linspace(0.0, 1.0, args.n_bins + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf
    detail_rows = []
    for lib in libs:
        detail_rows.extend(detail_distribution_rows(lib, edges=edges))
    write_csv(OUT / "detail_distribution_comparison.csv", detail_rows)

    ensembles: dict[str, dict[str, Any]] = {
        "fine_target": {"phi": fine, "source": "canonical_target"},
        "smooth_backbone": {"phi": back, "source": "baseline"},
        "zero_sum_gaussian_backbone_avg_sigma0p15": {
            "phi": make_zero_sum_noise(target_a, 0.15, args.seed + 11),
            "source": "baseline",
        },
    }
    oracle_prev = EMPIRICAL / "haar_conditional_fine_block_average.npy"
    if oracle_prev.exists():
        ensembles["previous_fine_library_oracle_initializer"] = {"phi": np.load(oracle_prev).astype(np.float64), "source": "previous_result"}
    for lib in libs:
        label = f"initializer_from_{lib['source']}"
        ensembles[label] = {"phi": build_initializer(lib, target_a, n_bins=args.n_bins, rng=rng), "source": lib["source"]}
    local_chunk = BENCH / "samples_sweeps_100.npy"
    local_summary = BENCH / "summary.json"
    local_coarse = None
    if local_chunk.exists():
        selected = np.asarray(json.loads(local_summary.read_text()).get("selected_indices", []), dtype=int) if local_summary.exists() else np.array([], dtype=int)
        local_coarse = coarse[selected] if len(selected) == len(np.load(local_chunk, mmap_mode="r")) else None
        ensembles["exact_null_local_chunk_100_sweeps_reference"] = {
            "phi": np.load(local_chunk).astype(np.float64),
            "source": "reference_correction",
            "coarse_override": local_coarse,
        }

    obs_rows = []
    block_rows = []
    low_rows = []
    for label, item in ensembles.items():
        phi = item["phi"]
        row = {"ensemble": label, "library_source": item["source"], **observables(phi, kappa=args.kappa, lam=args.lam)}
        obs_rows.append(row)
        coarse_ref = item.get("coarse_override", coarse)
        if coarse_ref is not None and len(coarse_ref) == len(phi):
            diff = block_sym(phi, w, block_norm) - coarse_ref
            simple = block_average_2x2(phi) - block_average_2x2(back if len(phi) == len(back) else back[: len(phi)])
            block_rows.append(
                {
                    "ensemble": label,
                    "library_source": item["source"],
                    "Bsym_rms": float(np.sqrt(np.mean(diff * diff))),
                    "Bsym_max": float(np.max(np.abs(diff))),
                    "Bsym_relative_rms": float(np.sqrt(np.mean(diff * diff)) / max(np.sqrt(np.mean(coarse_ref * coarse_ref)), 1.0e-30)),
                    "simple_blockavg_vs_backbone_blockavg_rms": float(np.sqrt(np.mean(simple * simple))),
                    "simple_blockavg_vs_backbone_blockavg_max": float(np.max(np.abs(simple))),
                }
            )
        low_rows.extend(low_momentum_rows(label, phi))
        if label.startswith("initializer_from_"):
            np.save(OUT / f"{label}.npy", phi)

    fine_row = next(r for r in obs_rows if r["ensemble"] == "fine_target")
    score_rows = []
    for row in obs_rows:
        if row["ensemble"] == "fine_target":
            continue
        score_rows.append({**row, **score_row(row, fine_row)})
    write_csv(OUT / "initializer_observables.csv", obs_rows)
    write_csv(OUT / "block_residuals.csv", block_rows)
    write_csv(OUT / "low_momentum_spectrum.csv", low_rows)
    write_csv(OUT / "source_comparison_scores.csv", sorted(score_rows, key=lambda r: float(r["local_moment_score_phi2_phi4_nn2"])))

    best_initializer = min(
        [r for r in score_rows if str(r["ensemble"]).startswith("initializer_from_")],
        key=lambda r: float(r["local_moment_score_phi2_phi4_nn2"]),
    )
    fine_init = next(r for r in score_rows if r["ensemble"] == "initializer_from_fine16_target_oracle")
    small32 = next((r for r in score_rows if r["ensemble"] == "initializer_from_small_volume_L8_kappa0p320_existing_nonproduction"), None)
    native295 = next((r for r in score_rows if r["ensemble"] == "initializer_from_native_L8_kappa0p295_mixed_wolff_local"), None)

    summary = {
        "output": str(OUT),
        "target_data": str(DATA / "fine_configs.npy"),
        "no_new_ensemble_generation": True,
        "wolff_only_note": "No new small-volume ensembles were generated because no valid continuous-amplitude Wolff-only generator is available in the current code path. Existing sources are used with honest metadata labels.",
        "n_bins": args.n_bins,
        "library_sources": [{k: str(v) if isinstance(v, Path) else v for k, v in src.items()} for src in source_definitions()],
        "best_initializer_by_phi2_phi4_nn2": best_initializer,
    }
    write_json(OUT / "summary.json", summary)

    report_rows = [fine_row] + sorted(score_rows, key=lambda r: float(r.get("local_moment_score_phi2_phi4_nn2", math.inf)))
    selected_lib_stats = [row for row in lib_stats if row["source"] in {"fine16_target_oracle", "small_volume_L8_kappa0p320_existing_nonproduction", "native_L8_kappa0p295_mixed_wolff_local"}]
    report = f"""# UV Library Source Comparison

No training was run.

No new ensembles were generated. This is intentional: the current repository contains embedded Wolff sign-cluster plus local-amplitude update code, but no valid continuous-amplitude Wolff-only generator. Existing L8 sources are therefore used with honest provenance labels rather than creating new mislabeled data.

## Library Statistics

{md_table(selected_lib_stats, ["source", "category", "L", "n_configs", "n_blocks", "kappa", "block_average_std", "h_var", "v_var", "d_var", "h_kurtosis", "v_kurtosis", "d_kurtosis"])}

## Initializer Observables

{md_table(report_rows, ["ensemble", "library_source", "phi2", "phi4", "NN", "nn2", "diag", "2nn", "Binder_U4", "xi_over_L", "action_density", "local_moment_score_phi2_phi4_nn2"])}

## Block Residuals

{md_table(block_rows, ["ensemble", "library_source", "Bsym_rms", "Bsym_max", "simple_blockavg_vs_backbone_blockavg_rms"])}

## Answers

1. The small-volume L8 kappa=0.320 library is close but not identical to the 16x16 fine oracle library. Its detail variances are lower, and its `phi2/phi4/nn2` initializer score is slightly worse than the current fine-oracle resampling row in this seeded diagnostic.
2. The native kappa=0.295 library is not worse by the narrow `phi2/phi4/nn2` score in this run; it is actually the best non-correction initializer by that score. This should not be overinterpreted as a solved UV law, because its broader operator score is still poor and it shares the same NN/diag and xi/L issues as the other Haar initializers.
3. Coarse-harvested UV details can match the one-site and squared-link moments surprisingly well, but the simple Haar construction still miscalibrates `NN`, `diag`, action density, and xi/L. Check `initializer_observables.csv` and `source_comparison_scores.csv`.
4. The h/v/d variances and kurtoses are listed in `library_statistics.csv`; conditional variances by block-average bin are in `detail_distribution_comparison.csv`.
5. These results suggest that part of the Haar UV detail distribution is fairly portable across cheap L8 sources, but not enough to make the whole initializer correct. The missing structure is in correlations/conditioning beyond block-average matching.
6. Existing native kappa=0.295 is a usable rough UV library source for initialization tests, with caveats: it is not Wolff-only, not a validated induced coarse law, and it still gives poor broader local operators.
7. For a Heidelberg/CNF run that avoids target fine-volume UV data, the best available non-oracle source in this diagnostic is `{best_initializer['library_source']}`, but it should be treated as an initialization prior, not a solved UV model.

## Key Numbers

- Fine-library oracle score: `{fine_init['local_moment_score_phi2_phi4_nn2']:.6g}`.
"""
    if small32:
        report += f"- Small-volume L8 kappa=0.320 score: `{small32['local_moment_score_phi2_phi4_nn2']:.6g}`.\n"
    if native295:
        report += f"- Native L8 kappa=0.295 score: `{native295['local_moment_score_phi2_phi4_nn2']:.6g}`.\n"
    report += """
## Output Files

- `library_statistics.csv`
- `initializer_observables.csv`
- `detail_distribution_comparison.csv`
- `source_comparison_scores.csv`
- `block_residuals.csv`
- `low_momentum_spectrum.csv`
- `summary.json`
"""
    (OUT / "report.md").write_text(report)
    print(json.dumps({"output": str(OUT), "best_initializer": best_initializer["ensemble"], "best_score": best_initializer["local_moment_score_phi2_phi4_nn2"]}, indent=2))


if __name__ == "__main__":
    main()
