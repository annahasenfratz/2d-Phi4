#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from scipy import stats as scipy_stats  # type: ignore
except Exception:  # pragma: no cover
    scipy_stats = None

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PKG = PROJECT_ROOT / "perfect_blocking_upsampling"
DEFAULT_OUT = PKG / "outputs" / "controlled_patch_lam0p2" / "rand5x5_0084_repro_workflow" / "upscaled_distribution_diagnostic" / "quantified_comparison"
DIRECT_L32 = PKG / "outputs" / "controlled_patch_lam0p2" / "coarse_source_overlap_diagnostic" / "native_L32_all_observables_per_config.csv"
BLOCKED_NATIVE = PKG / "outputs" / "controlled_patch_lam0p2" / "coarse_source_overlap_diagnostic" / "blocked_native_1000_rand5x5_0084_all_observables_per_config.csv"
UPSCALED = PKG / "outputs" / "controlled_patch_lam0p2" / "rand5x5_0084_repro_workflow" / "upscaled_distribution_diagnostic" / "upscaled_flow_1000_rand5x5_0084_width0p003_tail0p003_all_observables_per_config.csv"

OBSERVABLES = ["phi2", "phi4", "local_kurtosis_ratio", "NN", "2nn", "diag", "action_density"]
Q_LEVELS = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=True, default=_json_default) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    pd.DataFrame(rows, columns=fields).to_csv(path, index=False)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(type(obj).__name__)


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def as_float_series(df: pd.DataFrame, column: str) -> np.ndarray:
    x = df[column].to_numpy(dtype=np.float64)
    return x[np.isfinite(x)]


def quantiles(x: np.ndarray, levels: list[float]) -> dict[float, float]:
    return {q: float(np.quantile(x, q)) for q in levels}


def safe_std(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return float(np.std(x, ddof=1)) if x.size > 1 else 0.0


def safe_mean(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if x.size else float("nan")


def ks_wasserstein(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    if scipy_stats is None:
        return float("nan"), float("nan")
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan"), float("nan")
    return float(scipy_stats.ks_2samp(a, b).statistic), float(scipy_stats.wasserstein_distance(a, b))


def tail_occupancy(x: np.ndarray, q05: float, q95: float) -> dict[str, float]:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"low_tail_ratio": float("nan"), "high_tail_ratio": float("nan"), "tail_mass_ratio": float("nan")}
    low = float(np.mean(x <= q05))
    high = float(np.mean(x >= q95))
    return {
        "low_tail_ratio": low / 0.05 if 0.05 > 0 else float("nan"),
        "high_tail_ratio": high / 0.05 if 0.05 > 0 else float("nan"),
        "tail_mass_ratio": (low + high) / 0.10,
    }


def row_aligned_corr(blocked: pd.DataFrame, upscaled: pd.DataFrame, column: str) -> dict[str, float]:
    x = blocked[column].to_numpy(dtype=np.float64)
    y = upscaled[column].to_numpy(dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    out = {"pearson": float("nan"), "spearman": float("nan"), "n": int(mask.sum())}
    if x.size < 2:
        return out
    if scipy_stats is not None:
        out["pearson"] = float(scipy_stats.pearsonr(x, y).statistic)
        out["spearman"] = float(scipy_stats.spearmanr(x, y).statistic)
    else:
        out["pearson"] = float(np.corrcoef(x, y)[0, 1])
        rx = pd.Series(x).rank(method="average").to_numpy(dtype=np.float64)
        ry = pd.Series(y).rank(method="average").to_numpy(dtype=np.float64)
        out["spearman"] = float(np.corrcoef(rx, ry)[0, 1])
    return out


def plot_histogram(direct: np.ndarray, blocked: np.ndarray, upscaled: np.ndarray, observable: str, out_path: Path) -> None:
    values = np.concatenate([direct, blocked, upscaled])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return
    lo, hi = np.quantile(values, [0.005, 0.995])
    if lo == hi:
        lo -= 1.0
        hi += 1.0
    bins = np.linspace(lo, hi, 70)
    fig, ax = plt.subplots(figsize=(7.4, 4.7))
    ax.hist(direct, bins=bins, density=True, alpha=0.45, label=f"direct/native L32 (N={len(direct)})", color="#1f77b4")
    ax.hist(blocked, bins=bins, density=True, alpha=0.45, label=f"blocked native (N={len(blocked)})", color="#ff7f0e")
    ax.hist(upscaled, bins=bins, density=True, alpha=0.45, label=f"upscaled flow (N={len(upscaled)})", color="#2ca02c")
    ax.set_xlabel(observable)
    ax.set_ylabel("density")
    ax.set_title(f"{observable}: direct/native L32 vs blocked native vs upscaled flow")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.2, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--direct-l32", type=Path, default=DIRECT_L32)
    ap.add_argument("--blocked-native", type=Path, default=BLOCKED_NATIVE)
    ap.add_argument("--upscaled-flow", type=Path, default=UPSCALED)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    direct = load_csv(args.direct_l32)
    blocked = load_csv(args.blocked_native)
    upscaled = load_csv(args.upscaled_flow)

    summary_rows: list[dict[str, Any]] = []
    quantile_rows: list[dict[str, Any]] = []
    tail_rows: list[dict[str, Any]] = []
    corr_rows: list[dict[str, Any]] = []

    blocked_sample = blocked.sort_values(["sample"]) if "sample" in blocked.columns else blocked.copy()
    upscaled_sample = upscaled.sort_values(["sample"]) if "sample" in upscaled.columns else upscaled.copy()

    for obs in OBSERVABLES:
        direct_x = as_float_series(direct, obs)
        blocked_x = as_float_series(blocked, obs)
        upscaled_x = as_float_series(upscaled, obs)

        direct_q = quantiles(direct_x, Q_LEVELS)
        blocked_q = quantiles(blocked_x, Q_LEVELS)
        upscaled_q = quantiles(upscaled_x, Q_LEVELS)

        ks_u, wass_u = ks_wasserstein(direct_x, upscaled_x)
        ks_b, wass_b = ks_wasserstein(direct_x, blocked_x)

        q_abs = [abs(upscaled_q[q] - direct_q[q]) for q in Q_LEVELS]
        q_rms = float(np.sqrt(np.mean(np.square([upscaled_q[q] - direct_q[q] for q in Q_LEVELS]))))
        q_abs_blocked = [abs(blocked_q[q] - direct_q[q]) for q in Q_LEVELS]
        q_rms_blocked = float(np.sqrt(np.mean(np.square([blocked_q[q] - direct_q[q] for q in Q_LEVELS]))))

        summary_rows.append(
            {
                "observable": obs,
                "n_direct": int(direct_x.size),
                "n_blocked": int(blocked_x.size),
                "n_upscaled": int(upscaled_x.size),
                "mean_direct": safe_mean(direct_x),
                "std_direct": safe_std(direct_x),
                "mean_blocked": safe_mean(blocked_x),
                "std_blocked": safe_std(blocked_x),
                "mean_upscaled": safe_mean(upscaled_x),
                "std_upscaled": safe_std(upscaled_x),
                "mean_diff_upscaled_minus_direct": safe_mean(upscaled_x) - safe_mean(direct_x),
                "mean_diff_blocked_minus_direct": safe_mean(blocked_x) - safe_mean(direct_x),
                "standardized_mean_diff_direct_std": (safe_mean(upscaled_x) - safe_mean(direct_x)) / safe_std(direct_x) if safe_std(direct_x) > 0 else float("nan"),
                "width_ratio_upscaled_over_direct": safe_std(upscaled_x) / safe_std(direct_x) if safe_std(direct_x) > 0 else float("nan"),
                "width_ratio_blocked_over_direct": safe_std(blocked_x) / safe_std(direct_x) if safe_std(direct_x) > 0 else float("nan"),
                "ks_stat_upscaled_vs_direct": ks_u,
                "wasserstein_upscaled_vs_direct": wass_u,
                "ks_stat_blocked_vs_direct": ks_b,
                "wasserstein_blocked_vs_direct": wass_b,
                "mean_abs_quantile_mismatch_upscaled": float(np.mean(q_abs)),
                "rms_quantile_mismatch_upscaled": q_rms,
                "mean_abs_quantile_mismatch_blocked": float(np.mean(q_abs_blocked)),
                "rms_quantile_mismatch_blocked": q_rms_blocked,
            }
        )

        for q in Q_LEVELS:
            quantile_rows.append(
                {
                    "observable": obs,
                    "quantile": q,
                    "direct": direct_q[q],
                    "blocked": blocked_q[q],
                    "upscaled": upscaled_q[q],
                    "blocked_minus_direct": blocked_q[q] - direct_q[q],
                    "upscaled_minus_direct": upscaled_q[q] - direct_q[q],
                    "upscaled_minus_blocked": upscaled_q[q] - blocked_q[q],
                }
            )

        direct_q05 = direct_q[0.05]
        direct_q95 = direct_q[0.95]
        tail_u = tail_occupancy(upscaled_x, direct_q05, direct_q95)
        tail_b = tail_occupancy(blocked_x, direct_q05, direct_q95)
        tail_rows.append(
            {
                "observable": obs,
                "direct_q05": direct_q05,
                "direct_q95": direct_q95,
                "direct_low_tail_mass": float(np.mean(direct_x <= direct_q05)),
                "direct_high_tail_mass": float(np.mean(direct_x >= direct_q95)),
                "blocked_low_tail_mass": float(np.mean(blocked_x <= direct_q05)),
                "blocked_high_tail_mass": float(np.mean(blocked_x >= direct_q95)),
                "upscaled_low_tail_mass": float(np.mean(upscaled_x <= direct_q05)),
                "upscaled_high_tail_mass": float(np.mean(upscaled_x >= direct_q95)),
                "upscaled_low_tail_ratio_vs_direct": tail_u["low_tail_ratio"],
                "upscaled_high_tail_ratio_vs_direct": tail_u["high_tail_ratio"],
                "upscaled_tail_mass_ratio_vs_direct": tail_u["tail_mass_ratio"],
                "blocked_low_tail_ratio_vs_direct": tail_b["low_tail_ratio"],
                "blocked_high_tail_ratio_vs_direct": tail_b["high_tail_ratio"],
                "blocked_tail_mass_ratio_vs_direct": tail_b["tail_mass_ratio"],
            }
        )

        corr = row_aligned_corr(blocked_sample, upscaled_sample, obs)
        corr_rows.append(
            {
                "observable": obs,
                "n_row_aligned": corr["n"],
                "pearson_r_blocked_vs_upscaled": corr["pearson"],
                "spearman_r_blocked_vs_upscaled": corr["spearman"],
            }
        )

        plot_histogram(direct_x, blocked_x, upscaled_x, obs, args.out_dir / f"hist_{obs}_direct_L32_vs_upscaled_flow.pdf")

    write_csv(args.out_dir / "comparison_summary.csv", summary_rows)
    write_csv(args.out_dir / "quantile_mismatch.csv", quantile_rows)
    write_csv(args.out_dir / "tail_occupancy_ratios.csv", tail_rows)
    write_csv(args.out_dir / "row_aligned_correlations.csv", corr_rows)

    payload = {
        "status": "completed",
        "direct_l32": str(args.direct_l32),
        "blocked_native": str(args.blocked_native),
        "upscaled_flow": str(args.upscaled_flow),
        "scipy_available": scipy_stats is not None,
        "n_direct": int(len(direct)),
        "n_blocked": int(len(blocked)),
        "n_upscaled": int(len(upscaled)),
        "comparison_summary_csv": str(args.out_dir / "comparison_summary.csv"),
        "quantile_mismatch_csv": str(args.out_dir / "quantile_mismatch.csv"),
        "tail_occupancy_csv": str(args.out_dir / "tail_occupancy_ratios.csv"),
        "row_aligned_correlations_csv": str(args.out_dir / "row_aligned_correlations.csv"),
        "plots": [str(args.out_dir / f"hist_{obs}_direct_L32_vs_upscaled_flow.pdf") for obs in OBSERVABLES],
    }
    write_json(args.out_dir / "comparison_summary.json", payload)

    # Short report.
    s = pd.DataFrame(summary_rows).set_index("observable")
    report_lines = [
        "# rand5x5_0084 UpScaled vs Direct L32 Distribution Comparison",
        "",
        f"- direct/native L32 file: `{args.direct_l32}`",
        f"- blocked-native file: `{args.blocked_native}`",
        f"- upscaled-flow file: `{args.upscaled_flow}`",
        f"- scipy available: `{scipy_stats is not None}`",
        "",
        "## Key Means and Widths",
        "",
        "| observable | mean direct | mean blocked | mean upscaled | std direct | std upscaled | mean diff upscaled-direct | width ratio | KS | Wasserstein |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for obs in OBSERVABLES:
        row = s.loc[obs]
        report_lines.append(
            f"| {obs} | {row['mean_direct']:.6g} | {row['mean_blocked']:.6g} | {row['mean_upscaled']:.6g} | {row['std_direct']:.6g} | {row['std_upscaled']:.6g} | {row['mean_diff_upscaled_minus_direct']:.6g} | {row['width_ratio_upscaled_over_direct']:.6g} | {row['ks_stat_upscaled_vs_direct']:.6g} | {row['wasserstein_upscaled_vs_direct']:.6g} |"
        )

    report_lines += [
        "",
        "## Interpretation",
        "",
    ]
    ad = s.loc["action_density"]
    mean_shift = abs(float(ad["mean_diff_upscaled_minus_direct"]))
    width_mismatch = abs(float(ad["width_ratio_upscaled_over_direct"]) - 1.0)
    report_lines.append(
        f"- action_density mismatch is dominated by {'mean shift' if mean_shift >= width_mismatch else 'width mismatch'} at the direct-L32 scale."
    )
    fk = s.loc["phi4"]
    lk = s.loc["local_kurtosis_ratio"]
    report_lines.append(
        f"- phi4 mean diff `{float(fk['mean_diff_upscaled_minus_direct']):.6g}`, width ratio `{float(fk['width_ratio_upscaled_over_direct']):.6g}`."
    )
    report_lines.append(
        f"- local_kurtosis_ratio mean diff `{float(lk['mean_diff_upscaled_minus_direct']):.6g}`, width ratio `{float(lk['width_ratio_upscaled_over_direct']):.6g}`."
    )
    report_lines.append("")
    report_lines.append("Row-aligned correlations compare the blocked-native and upscaled-flow rows in sample order; they probe whether the flow preserves source ordering beyond the marginal.")
    (args.out_dir / "upscaled_vs_direct_distribution_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
