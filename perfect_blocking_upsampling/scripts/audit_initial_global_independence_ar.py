#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from _common import load_actions, load_config, load_ensembles, load_frozen_models, load_kernel_spec
from perfect_blocking_upsampling.ar import independence_ar, stable_ess
from perfect_blocking_upsampling.sampling import generate_proposals


DEFAULT_OUT = (
    Path(__file__).resolve().parents[1]
    / "outputs"
    / "shape_parametric_sampler_validation"
    / "coarse_ar_audit"
    / "initial_global_independence_ar"
)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=float) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def top_weight_stats(logw: np.ndarray) -> dict[str, float]:
    shifted = logw - np.max(logw)
    weights = np.exp(shifted)
    weights /= max(float(np.sum(weights)), 1.0e-300)
    desc = np.sort(weights)[::-1]
    n = len(desc)
    return {
        "max_normalized_weight": float(desc[0]),
        "top_5_weight_sum": float(np.sum(desc[: min(5, n)])),
        "top_10_weight_sum": float(np.sum(desc[: min(10, n)])),
        "top_1pct_weight_sum": float(np.sum(desc[: max(1, int(np.ceil(0.01 * n)))])),
        "top_5pct_weight_sum": float(np.sum(desc[: max(1, int(np.ceil(0.05 * n)))])),
    }


def paired_acceptance_estimate(logw: np.ndarray, seed: int, n_pairs: int) -> float:
    rng = np.random.default_rng(seed)
    n = len(logw)
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    delta = logw[j] - logw[i]
    return float(np.mean(np.minimum(1.0, np.exp(np.minimum(delta, 0.0)))))


def run_case(label: str, config_path: Path, n: int, chains: int, seed: int, out_dir: Path) -> dict[str, Any]:
    cfg = load_config(config_path)
    coarse_action, fine_action = load_actions(cfg)
    coarse, _, _, _, _ = load_ensembles(cfg)
    kernel_spec, _ = load_kernel_spec(cfg)
    refine_model, refine_state, stage_bundles, _, _, _ = load_frozen_models(cfg)

    rows = []
    chain_summaries = []
    all_logw = []
    for chain in range(chains):
        rng = np.random.default_rng(seed + 1009 * chain)
        coarse_batch = coarse[rng.integers(0, len(coarse), size=n)]
        proposals = generate_proposals(
            coarse_batch,
            refine_model,
            refine_state,
            stage_bundles,
            kernel_spec,
            fine_action,
            coarse_action,
            seed + 17017 + chain,
        )
        logw = proposals.logw.astype(np.float64)
        all_logw.append(logw)
        ar = independence_ar(proposals, np.random.default_rng(seed + 99009 + chain), fine_action)
        summary = {
            "case": label,
            "chain_id": chain,
            "n_proposals": int(n),
            "logweight_convention": "-S_f(phi) + S_c(u) + logdet_refine - logq_missing",
            "logw_mean": float(np.mean(logw)),
            "logw_std": float(np.std(logw, ddof=1)) if n > 1 else 0.0,
            "logw_min": float(np.min(logw)),
            "logw_q05": float(np.quantile(logw, 0.05)),
            "logw_q50": float(np.quantile(logw, 0.50)),
            "logw_q95": float(np.quantile(logw, 0.95)),
            "logw_max": float(np.max(logw)),
            "ess_over_n": float(stable_ess(logw) / max(n, 1)),
            "paired_independence_acceptance_estimate": paired_acceptance_estimate(logw, seed + 88001 + chain, max(10000, 20 * n)),
            "simulated_independence_mh_acceptance": float(ar["acceptance_rate"]),
            "max_rejection_streak": int(ar["max_rejection_streak"]),
            **top_weight_stats(logw),
        }
        chain_summaries.append(summary)
        rows.extend({"case": label, "chain_id": chain, "proposal_id": i, "logw": float(w)} for i, w in enumerate(logw))

    pooled = np.concatenate(all_logw)
    pooled_summary = {
        "case": label,
        "config": str(config_path),
        "coarse_L": int(coarse.shape[1]),
        "fine_L": int(2 * coarse.shape[1]),
        "chains": int(chains),
        "n_proposals_per_chain": int(n),
        "n_proposals_total": int(len(pooled)),
        "coarse_action": coarse_action.as_dict,
        "fine_action": fine_action.as_dict,
        "logweight_convention": "-S_f(phi) + S_c(u) + logdet_refine - logq_missing",
        "logw_mean": float(np.mean(pooled)),
        "logw_std": float(np.std(pooled, ddof=1)) if len(pooled) > 1 else 0.0,
        "logw_min": float(np.min(pooled)),
        "logw_q05": float(np.quantile(pooled, 0.05)),
        "logw_q50": float(np.quantile(pooled, 0.50)),
        "logw_q95": float(np.quantile(pooled, 0.95)),
        "logw_max": float(np.max(pooled)),
        "ess_over_n": float(stable_ess(pooled) / max(len(pooled), 1)),
        "paired_independence_acceptance_estimate": paired_acceptance_estimate(pooled, seed + 7777, max(20000, 20 * len(pooled))),
        **top_weight_stats(pooled),
        "chain_summaries": chain_summaries,
    }

    case_dir = out_dir / label
    write_json(case_dir / "summary.json", pooled_summary)
    write_csv(case_dir / "chain_summaries.csv", chain_summaries)
    write_csv(case_dir / "proposal_logweights.csv", rows)
    return pooled_summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--n-proposals", type=int, default=1024)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260701)
    ap.add_argument(
        "--case",
        action="append",
        nargs=2,
        metavar=("LABEL", "CONFIG"),
        help="Run one named case. Can be repeated. Defaults to the accepted L8->L16 and L16->L32 configs.",
    )
    args = ap.parse_args()
    if args.n_proposals <= 0:
        raise ValueError("--n-proposals must be positive")
    if args.chains <= 0:
        raise ValueError("--chains must be positive")

    cases = args.case or [
        (
            "L8_to_L16_lam0p022",
            "perfect_blocking_upsampling/outputs/procedural_corner_diagnostics/old_pair_corner_procedural_masks.yaml",
        ),
        (
            "L16_to_L32_lam0p022",
            "perfect_blocking_upsampling/outputs/shape_parametric_sampler_validation/L16_to_L32_smoke/L16_to_L32_smoke_config.yaml",
        ),
    ]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summaries = [run_case(label, Path(config), args.n_proposals, args.chains, args.seed + 100000 * idx, args.out_dir) for idx, (label, config) in enumerate(cases)]
    write_json(args.out_dir / "combined_summary.json", summaries)

    lines = [
        "# Initial/global independence A/R diagnostic",
        "",
        "Logweight convention: `logw = -S_f(phi) + S_c(u) + logdet_refine - logq_missing`.",
        "",
        "| case | proposals | logw mean | logw std | ESS/N | paired A/R est. | max weight | top 1% | top 5% |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        lines.append(
            f"| {s['case']} | {s['n_proposals_total']} | {s['logw_mean']:.6g} | {s['logw_std']:.6g} | "
            f"{s['ess_over_n']:.6g} | {s['paired_independence_acceptance_estimate']:.6g} | "
            f"{s['max_normalized_weight']:.6g} | {s['top_1pct_weight_sum']:.6g} | {s['top_5pct_weight_sum']:.6g} |"
        )
    (args.out_dir / "INITIAL_GLOBAL_INDEPENDENCE_AR_SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summaries, indent=2, sort_keys=True, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
