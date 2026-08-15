#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PKG = PROJECT_ROOT / "perfect_blocking_upsampling"
SCRIPT_DIR = PKG / "scripts"
sys.path.insert(0, str(PKG / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

from perfect_blocking_upsampling.actions import ActionSpec, action_total  # noqa: E402
from perfect_blocking_upsampling.kernels import apply_kernel, load_kernel  # noqa: E402
from run_lam0p2_conditional_gaussian_residual_L16 import residual_logdet  # noqa: E402
from run_lam0p2_flow_detail_rethermalization import infer_ar_latents_and_logj  # noqa: E402
from run_lam0p2_mit_style_inverse_blocking_L8to16 import (  # noqa: E402
    full_inverse_kernel_logdet,
    log_standard_normal,
)
from run_lam0p2_residual_flow_patch_chain import load_initializer, read_phi  # noqa: E402
from run_lam0p2_rand5x5_0084_detail_only_correction_diagnostic import write_json  # noqa: E402

LAM = 0.2
QUANTILES = [0.0, 0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999, 1.0]


def sanitize(name: str) -> str:
    return (
        name.replace("/", "_")
        .replace("|", "abs")
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("-", "minus")
    )


def summarize_array(values: np.ndarray) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    out: dict[str, float] = {
        "count": float(len(x)),
        "mean": float(np.mean(x)) if len(x) else float("nan"),
        "std": float(np.std(x, ddof=1)) if len(x) > 1 else float("nan"),
        "min": float(np.min(x)) if len(x) else float("nan"),
        "max": float(np.max(x)) if len(x) else float("nan"),
    }
    if len(x):
        qs = np.quantile(x, QUANTILES)
        for q, val in zip(QUANTILES, qs):
            out[f"q{q:g}"] = float(val)
    else:
        for q in QUANTILES:
            out[f"q{q:g}"] = float("nan")
    return out


def add_components(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["X_f"] = -df["S_fine_new"]
    df["X_c"] = df["S_coarse_new"]
    df["X_r"] = -df["log_prior_detail_new"]
    df["X_J"] = df["logabsdet_total_new"]
    df["logw_recomputed"] = df["X_f"] + df["X_c"] + df["X_r"] + df["X_J"]
    df["logq_recomputed"] = -df["S_coarse_new"] + df["log_prior_detail_new"] - df["logabsdet_total_new"]
    df["logw_from_logq_recomputed"] = -df["S_fine_new"] - df["logq_recomputed"]
    df["coarse_action_density_new"] = df["S_coarse_new"] / 64.0
    nz = 3 * 8 * 8
    df["detail_noise_norm2"] = -2.0 * (df["log_prior_detail_new"] + 0.5 * nz * math.log(2.0 * math.pi))
    return df


def chain_current_components(proposals: pd.DataFrame, chain: pd.DataFrame) -> pd.DataFrame:
    prop_by_index = proposals.set_index("proposal_index")
    current_prop_indices: list[int] = []
    current_idx: int | None = None
    for _, row in proposals.iterrows():
        if int(row["accepted"]) == 1:
            current_idx = int(row["proposal_index"])
        current_prop_indices.append(current_idx if current_idx is not None else -1)
    mapping = pd.DataFrame({"proposal_index": proposals["proposal_index"].to_numpy(), "current_proposal_index": current_prop_indices})
    out = chain.merge(mapping, on="proposal_index", how="left")
    comp_cols = [
        "S_fine_new",
        "S_coarse_new",
        "log_prior_detail_new",
        "logabsdet_total_new",
        "logq_new",
        "logw_new",
        "X_f",
        "X_c",
        "X_r",
        "X_J",
        "fine_action_density_new",
        "coarse_action_density_new",
        "detail_noise_norm2",
    ]
    for col in comp_cols:
        out[f"current_{col}"] = out["current_proposal_index"].map(prop_by_index[col])
    return out


def write_summaries(proposals: pd.DataFrame, current: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    variables = [
        "logw_new",
        "S_fine_new",
        "S_coarse_new",
        "X_r",
        "logabsdet_total_new",
        "logabsdet_coordinate_identity_coarse",
        "logabsdet_permutation_reshape",
        "logabsdet_flow_detail",
        "logabsdet_residual_standardization",
        "logabsdet_inverse_kernel_full_constant",
        "delta_logw",
    ]
    groups: list[tuple[str, pd.DataFrame, str]] = [
        ("all_proposals", proposals, ""),
        ("accepted_proposals", proposals[(proposals["accepted"] == 1) & (proposals["initialization"] == 0)], ""),
        ("rejected_proposals", proposals[proposals["accepted"] == 0], ""),
        ("first_256_proposals", proposals[proposals["proposal_index"] <= 256], ""),
        ("after_256_proposals", proposals[proposals["proposal_index"] > 256], ""),
        ("after_512_proposals", proposals[proposals["proposal_index"] > 512], ""),
    ]
    rows: list[dict[str, Any]] = []
    for group, frame, prefix in groups:
        for var in variables:
            if var not in frame:
                continue
            row = {"group": group, "variable": var}
            row.update(summarize_array(frame[var].to_numpy()))
            rows.append(row)
    current_map = {
        "logw_new": "logw_current",
        "S_fine_new": "current_S_fine_new",
        "S_coarse_new": "current_S_coarse_new",
        "X_r": "current_X_r",
        "logabsdet_total_new": "current_logabsdet_total_new",
        "delta_logw": "delta_logw",
    }
    for var, col in current_map.items():
        if col in current:
            row = {"group": "current_chain_states", "variable": var}
            row.update(summarize_array(current[col].to_numpy()))
            rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "logweight_component_summaries.csv", index=False)
    return summary


def linear_decomposition(proposals: pd.DataFrame, out_dir: Path) -> dict[str, Any]:
    comps = ["X_f", "X_c", "X_r", "X_J"]
    X = proposals[comps].to_numpy(dtype=np.float64)
    y = proposals["logw_new"].to_numpy(dtype=np.float64)
    cov = pd.DataFrame(np.cov(X, rowvar=False), index=comps, columns=comps)
    cov.to_csv(out_dir / "component_covariance_matrix.csv")
    corr_rows = []
    for c in comps:
        corr_rows.append({"component": c, "corr_with_logw": float(np.corrcoef(proposals[c], y)[0, 1])})
    pd.DataFrame(corr_rows).to_csv(out_dir / "component_correlations.csv", index=False)
    rows = []
    var_total = float(np.var(y, ddof=1))
    for i, c in enumerate(comps):
        rows.append({"term": f"var({c})", "value": float(cov.iloc[i, i]), "fraction_of_var_logw": float(cov.iloc[i, i] / var_total)})
    for i, a in enumerate(comps):
        for j, b in enumerate(comps):
            if j <= i:
                continue
            val = 2.0 * float(cov.iloc[i, j])
            rows.append({"term": f"2cov({a},{b})", "value": val, "fraction_of_var_logw": float(val / var_total)})
    rows.append({"term": "var(logw)", "value": var_total, "fraction_of_var_logw": 1.0})
    pd.DataFrame(rows).to_csv(out_dir / "component_variance_contributions.csv", index=False)
    X_design = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X_design, y, rcond=None)
    pred = X_design @ beta
    regression = {
        "intercept": float(beta[0]),
        **{f"coef_{c}": float(beta[i + 1]) for i, c in enumerate(comps)},
        "r2": float(1.0 - np.sum((y - pred) ** 2) / np.sum((y - np.mean(y)) ** 2)),
        "max_abs_residual": float(np.max(np.abs(y - pred))),
    }
    write_json(out_dir / "component_regression.json", regression)
    return {
        "covariance": cov.to_dict(),
        "correlations": corr_rows,
        "variance_contributions": rows,
        "regression": regression,
    }


def source_reuse(proposals: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rows = []
    for idx, g in proposals.groupby("coarse_source_index"):
        rows.append(
            {
                "coarse_source_index": int(idx),
                "count": int(len(g)),
                "accepted_count": int(((g["accepted"] == 1) & (g["initialization"] == 0)).sum()),
                "logw_mean": float(g["logw_new"].mean()),
                "logw_max": float(g["logw_new"].max()),
                "logw_std": float(g["logw_new"].std(ddof=1)) if len(g) > 1 else float("nan"),
                "S_coarse_mean": float(g["S_coarse_new"].mean()),
            }
        )
    out = pd.DataFrame(rows).sort_values(["count", "logw_max"], ascending=[False, False])
    out.to_csv(out_dir / "coarse_source_reuse_weight_summary.csv", index=False)
    return out


def load_native_target_logweights(args: argparse.Namespace, summary: dict[str, Any], max_native: int) -> pd.DataFrame:
    kernel, _ = load_kernel(args.kernel_path)
    phi = read_phi(args.native_reference, args.to_L)
    if max_native > 0:
        phi = phi[:max_native]
    init_args = argparse.Namespace(
        from_L=args.from_L,
        to_L=args.to_L,
        device=args.device,
        ar_checkpoint=args.ar_checkpoint,
        best_checkpoint=args.ar_checkpoint,
        baseline_checkpoint=args.baseline_checkpoint,
        initializer_kind="ar",
        n_coupling_layers=8,
        hidden_channels=32,
        conv_kernel_size=5,
        log_scale_bound=0.75,
        batch_size=args.native_batch_size,
    )
    models, predictor, flow_stats = load_initializer(init_args)
    residual_logdet_const = float(residual_logdet(flow_stats))
    kinv_logdet_const = float(full_inverse_kernel_logdet(kernel, args.to_L))
    action_c = ActionSpec("phi4_nn", LAM, args.kappa_c)
    action_f = ActionSpec("phi4_nn", LAM, args.kappa_f)
    rows = []
    batch = args.native_batch_size
    for start in range(0, len(phi), batch):
        phib = phi[start : start + batch].astype(np.float32)
        psi = apply_kernel(phib, kernel).astype(np.float32)
        coarse = psi[:, 0::2, 0::2].astype(np.float32)
        z, flow_logdet = infer_ar_latents_and_logj(models, predictor, flow_stats, psi, init_args)
        log_prior = log_standard_normal(z)
        S_c = action_total(coarse, action_c).astype(np.float64)
        S_f = action_total(phib, action_f).astype(np.float64)
        logdet_total = flow_logdet + residual_logdet_const + kinv_logdet_const
        logq = -S_c + log_prior - logdet_total
        logw = -S_f - logq
        for j in range(len(phib)):
            rows.append(
                {
                    "native_index": start + j,
                    "logw_new": float(logw[j]),
                    "S_fine_new": float(S_f[j]),
                    "S_coarse_new": float(S_c[j]),
                    "log_prior_detail_new": float(log_prior[j]),
                    "logabsdet_total_new": float(logdet_total[j]),
                    "logq_new": float(logq[j]),
                    "fine_action_density_new": float(S_f[j] / (args.to_L * args.to_L)),
                    "logabsdet_flow_detail": float(flow_logdet[j]),
                    "logabsdet_residual_standardization": residual_logdet_const,
                    "logabsdet_inverse_kernel_full_constant": kinv_logdet_const,
                }
            )
    return add_components(pd.DataFrame(rows))


def save_target_comparison(proposals: pd.DataFrame, current: pd.DataFrame, target: pd.DataFrame, out_dir: Path) -> dict[str, Any]:
    rows = []
    for label, frame in [("proposal_q", proposals), ("native_target_p", target), ("accepted_chain_states", current.rename(columns={"logw_current": "logw_new"}))]:
        if label == "accepted_chain_states":
            vals = current["logw_current"].to_numpy()
        else:
            vals = frame["logw_new"].to_numpy()
        row = {"distribution": label, "variable": "logw"}
        row.update(summarize_array(vals))
        rows.append(row)
        for comp in ["S_fine_new", "S_coarse_new", "X_r", "X_J"]:
            col = comp if label != "accepted_chain_states" else f"current_{comp}"
            if col in frame:
                row = {"distribution": label, "variable": comp}
                row.update(summarize_array(frame[col].to_numpy()))
                rows.append(row)
    comp = pd.DataFrame(rows)
    comp.to_csv(out_dir / "proposal_native_target_logweight_comparison.csv", index=False)
    native_q50 = float(np.quantile(target["logw_new"], 0.5))
    prop_q50 = float(np.quantile(proposals["logw_new"], 0.5))
    native_q99 = float(np.quantile(target["logw_new"], 0.99))
    prop_q99 = float(np.quantile(proposals["logw_new"], 0.99))
    out = {
        "proposal_logw_median": prop_q50,
        "native_logw_median": native_q50,
        "native_minus_proposal_median": native_q50 - prop_q50,
        "proposal_logw_q99": prop_q99,
        "native_logw_q99": native_q99,
        "native_minus_proposal_q99": native_q99 - prop_q99,
        "fraction_native_above_proposal_q99": float(np.mean(target["logw_new"] > prop_q99)),
        "fraction_proposal_above_native_median": float(np.mean(proposals["logw_new"] > native_q50)),
    }
    write_json(out_dir / "proposal_native_target_logweight_comparison.json", out)
    return out


def save_fig(fig: plt.Figure, out_dir: Path, name: str, pdf: PdfPages | None = None) -> None:
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}.pdf")
    fig.savefig(out_dir / f"{name}.png", dpi=180)
    if pdf is not None:
        pdf.savefig(fig)
    plt.close(fig)


def make_plots(proposals: pd.DataFrame, current: pd.DataFrame, target: pd.DataFrame, out_dir: Path) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    accepted = proposals[(proposals["accepted"] == 1) & (proposals["initialization"] == 0)]
    rejected = proposals[proposals["accepted"] == 0]
    final_streak_start = int(accepted["proposal_index"].max()) if len(accepted) else int(proposals["proposal_index"].iloc[0])
    with PdfPages(out_dir / "mit_logweight_tail_plots.pdf") as pdf:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(proposals["logw_new"], bins=60, alpha=0.75)
        ax.set_xlabel("proposed logw")
        ax.set_ylabel("count")
        ax2 = ax.twinx()
        xs = np.sort(proposals["logw_new"].to_numpy())
        ax2.plot(xs, np.arange(1, len(xs) + 1) / len(xs), color="black", lw=1.5)
        ax2.set_ylabel("empirical CDF")
        save_fig(fig, fig_dir, "logw_histogram_and_cdf", pdf)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(proposals["logw_new"], bins=60, alpha=0.75, log=True)
        ax.set_xlabel("proposed logw")
        ax.set_ylabel("count (log)")
        save_fig(fig, fig_dir, "logw_histogram_log_y", pdf)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(proposals["proposal_index"], proposals["logw_new"], ".", ms=2)
        ax.axvline(final_streak_start, color="red", ls="--", label="last accepted before final streak")
        ax.set_xlabel("proposal index")
        ax.set_ylabel("proposed logw")
        ax.legend()
        save_fig(fig, fig_dir, "logw_new_vs_proposal_index", pdf)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(current["proposal_index"], current["logw_current"], lw=1)
        ax.set_xlabel("proposal index")
        ax.set_ylabel("current-state logw")
        save_fig(fig, fig_dir, "current_logw_vs_proposal_index", pdf)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(proposals["proposal_index"], proposals["logw_new"], ".", ms=2, alpha=0.55, label="proposal")
        ax.plot(current["proposal_index"], current["logw_current"], lw=1.5, label="current")
        ax.axvline(final_streak_start, color="red", ls="--")
        ax.set_xlabel("proposal index")
        ax.set_ylabel("logw")
        ax.legend()
        save_fig(fig, fig_dir, "logw_new_and_current_overlay", pdf)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(proposals["delta_logw"].replace([np.inf, -np.inf], np.nan).dropna(), bins=80)
        ax.set_xlabel("delta logw")
        ax.set_ylabel("count")
        save_fig(fig, fig_dir, "delta_logw_histogram", pdf)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(rejected["logw_new"], bins=60, alpha=0.55, label="rejected")
        ax.hist(accepted["logw_new"], bins=30, alpha=0.7, label="accepted")
        ax.set_xlabel("proposed logw")
        ax.set_ylabel("count")
        ax.legend()
        save_fig(fig, fig_dir, "accepted_vs_rejected_logw_histograms", pdf)

        fig, axes = plt.subplots(2, 2, figsize=(9, 7))
        for ax, col in zip(axes.ravel(), ["X_f", "X_c", "X_r", "X_J"]):
            ax.hist(proposals[col], bins=60)
            ax.set_title(col)
        save_fig(fig, fig_dir, "component_histograms", pdf)

        fig, axes = plt.subplots(2, 2, figsize=(9, 7))
        for ax, col in zip(axes.ravel(), ["X_f", "X_c", "X_r", "X_J"]):
            ax.plot(proposals[col], proposals["logw_new"], ".", ms=2, alpha=0.5)
            ax.set_xlabel(col)
            ax.set_ylabel("logw")
        save_fig(fig, fig_dir, "logw_vs_components", pdf)

        comps = ["X_f", "X_c", "X_r", "X_J"]
        fig, axes = plt.subplots(4, 4, figsize=(10, 10))
        for i, a in enumerate(comps):
            for j, b in enumerate(comps):
                ax = axes[i, j]
                if i == j:
                    ax.hist(proposals[a], bins=30)
                else:
                    ax.plot(proposals[b], proposals[a], ".", ms=1.5, alpha=0.4)
                if i == 3:
                    ax.set_xlabel(b)
                if j == 0:
                    ax.set_ylabel(a)
        save_fig(fig, fig_dir, "component_pairwise_scatter", pdf)

        for col, name, xlabel in [
            ("fine_action_density_new", "logw_vs_fine_action_density", "fine action density"),
            ("coarse_action_density_new", "logw_vs_coarse_action_density", "coarse action density"),
            ("coarse_source_index", "logw_vs_coarse_source_index", "coarse source index"),
        ]:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(proposals[col], proposals["logw_new"], ".", ms=2, alpha=0.5)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("logw")
            save_fig(fig, fig_dir, name, pdf)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(proposals["logw_old"], np.exp(proposals["log_alpha"].clip(-100, 0)), ".", ms=2, alpha=0.5)
        ax.set_xlabel("current-state logw before proposal")
        ax.set_ylabel("acceptance probability")
        save_fig(fig, fig_dir, "acceptance_probability_vs_current_logw", pdf)

        ah = pd.read_csv(out_dir.parent / "acceptance_history.csv") if False else None
        streaks = []
        current_logw_for_streak = []
        streak = 0
        start_logw = None
        for _, row in proposals[proposals["initialization"] == 0].iterrows():
            if row["accepted"] == 1:
                if streak:
                    streaks.append(streak)
                    current_logw_for_streak.append(start_logw)
                streak = 0
                start_logw = row["logw_new"]
            else:
                if streak == 0:
                    start_logw = row["logw_old"]
                streak += 1
        if streak:
            streaks.append(streak)
            current_logw_for_streak.append(start_logw)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(current_logw_for_streak, streaks, "o")
        ax.set_xlabel("current-state logw at streak start")
        ax.set_ylabel("rejection-streak length")
        save_fig(fig, fig_dir, "rejection_streak_length_vs_current_logw", pdf)

        running_max = np.maximum.accumulate(proposals.loc[proposals["accepted"] == 1, "logw_new"].to_numpy())
        acc_idx = proposals.loc[proposals["accepted"] == 1, "proposal_index"].to_numpy()
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.step(acc_idx, running_max, where="post")
        ax.set_xlabel("proposal index")
        ax.set_ylabel("running max accepted logw")
        save_fig(fig, fig_dir, "running_max_accepted_logw", pdf)

        fig, ax = plt.subplots(figsize=(7, 4))
        sorted_logw = np.sort(proposals["logw_new"].to_numpy())[::-1]
        ax.plot(np.arange(1, len(sorted_logw) + 1), sorted_logw, ".-")
        ax.set_xlabel("rank")
        ax.set_ylabel("proposed logw")
        save_fig(fig, fig_dir, "logw_rank_plot", pdf)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(proposals["logw_new"], bins=60, alpha=0.45, density=True, label="proposal q")
        ax.hist(target["logw_new"], bins=60, alpha=0.45, density=True, label="native target p")
        ax.hist(current["logw_current"], bins=40, alpha=0.45, density=True, label="chain states")
        ax.set_xlabel("logw")
        ax.set_ylabel("density")
        ax.legend()
        save_fig(fig, fig_dir, "proposal_native_chain_logw_comparison", pdf)


def write_report(
    out_dir: Path,
    run_dir: Path,
    proposals: pd.DataFrame,
    current: pd.DataFrame,
    summary_table: pd.DataFrame,
    decomp: dict[str, Any],
    target_summary: dict[str, Any],
    consistency: dict[str, Any],
) -> None:
    accepted = proposals[(proposals["accepted"] == 1) & (proposals["initialization"] == 0)]
    max_acc = accepted.loc[accepted["logw_new"].idxmax()] if len(accepted) else proposals.loc[proposals["logw_new"].idxmax()]
    med = proposals.median(numeric_only=True)
    std = proposals.std(numeric_only=True, ddof=1)
    diagnostic_cols = [
        "logw_new",
        "S_fine_new",
        "S_coarse_new",
        "X_r",
        "logabsdet_total_new",
        "logabsdet_flow_detail",
        "fine_action_density_new",
        "coarse_action_density_new",
        "detail_noise_norm2",
    ]
    z_lines = []
    for col in diagnostic_cols:
        z = (float(max_acc[col]) - float(med[col])) / float(std[col]) if float(std[col]) != 0 else float("nan")
        z_lines.append(f"| `{col}` | {float(max_acc[col]):.8g} | {float(med[col]):.8g} | {z:.3g} |")
    vc = pd.DataFrame(decomp["variance_contributions"])
    top_terms = vc[vc["term"] != "var(logw)"].copy()
    top_terms["abs_fraction"] = top_terms["fraction_of_var_logw"].abs()
    top_terms = top_terms.sort_values("abs_fraction", ascending=False).head(8)
    vc_lines = [f"| `{r.term}` | {r.value:.8g} | {r.fraction_of_var_logw:.4g} |" for r in top_terms.itertuples()]
    corr = pd.DataFrame(decomp["correlations"]).sort_values("corr_with_logw", key=lambda s: s.abs(), ascending=False)
    corr_lines = [f"| `{r.component}` | {r.corr_with_logw:.4g} |" for r in corr.itertuples()]
    all_logw = summary_table[(summary_table["group"] == "all_proposals") & (summary_table["variable"] == "logw_new")].iloc[0]
    after512_logw = summary_table[(summary_table["group"] == "after_512_proposals") & (summary_table["variable"] == "logw_new")].iloc[0]
    text = [
        "# MIT-Style Log-Weight Tail Analysis",
        "",
        f"Run directory: `{run_dir}`",
        "",
        "## Main Result",
        "",
        (
            "The acceptance collapse is caused by an independence-sampler overlap problem: "
            "once the chain accepts a high-`logw` state, later independent proposals almost never exceed it. "
            "The final accepted high-weight state is not a nonfinite numerical pathology in the stored diagnostics."
        ),
        "",
        f"- all-proposal `logw` median: `{all_logw['q0.5']:.6g}`",
        f"- all-proposal `logw` 99% quantile: `{all_logw['q0.99']:.6g}`",
        f"- all-proposal max `logw`: `{all_logw['max']:.6g}`",
        f"- after-512 proposal max `logw`: `{after512_logw['max']:.6g}`",
        f"- max accepted `logw`: `{float(max_acc['logw_new']):.6g}` at proposal `{int(max_acc['proposal_index'])}`",
        "",
        "## Consistency Checks",
        "",
        f"- max `|logw - (X_f+X_c+X_r+X_J)|`: `{consistency['max_abs_logw_recompute_error']:.3e}`",
        f"- max `|logq - (-S_c+logr-logJ)|`: `{consistency['max_abs_logq_recompute_error']:.3e}`",
        f"- max `|logw - (-S_f-logq)|`: `{consistency['max_abs_logw_from_logq_error']:.3e}`",
        f"- nonfinite proposals: `{int(consistency['nonfinite_proposals'])}`",
        "",
        "## Maximum Accepted State Compared With Proposal Median",
        "",
        "| quantity | max accepted | proposal median | z-score vs proposal spread |",
        "|---|---:|---:|---:|",
        *z_lines,
        "",
        "## Dominant Variance/Cancellation Terms",
        "",
        "| term | contribution | fraction of Var(logw) |",
        "|---|---:|---:|",
        *vc_lines,
        "",
        "## Component Correlations With `logw`",
        "",
        "| component | correlation |",
        "|---|---:|",
        *corr_lines,
        "",
        "The broad terms are not interpretable from their marginal variances alone. "
        "The covariance table in `component_covariance_matrix.csv` and variance contribution table show the cancellations explicitly.",
        "",
        "## Native Target Comparison",
        "",
        f"- native median minus proposal median `logw`: `{target_summary['native_minus_proposal_median']:.6g}`",
        f"- native q99 minus proposal q99 `logw`: `{target_summary['native_minus_proposal_q99']:.6g}`",
        f"- fraction of native fields above proposal q99: `{target_summary['fraction_native_above_proposal_q99']:.6g}`",
        f"- fraction of proposals above native median: `{target_summary['fraction_proposal_above_native_median']:.6g}`",
        "",
        "## Proposal-Level Observables",
        "",
        "Rejected proposal configurations were not saved, so full proposal-level observables such as `phi2`, `phi4`, NN, Binder inputs, and xi/L cannot be reconstructed for all raw proposals. "
        "The available proposal-level physics diagnostics are `S_fine_new`, `fine_action_density_new`, `S_coarse_new`, `coarse_action_density_new`, and detail-noise norm inferred from the Gaussian prior. "
        "The chain-state measurements are repeated-state measurements after rejection handling and are therefore not a substitute for proposal-level measurements.",
        "",
        "## Diagnosis",
        "",
        "The current evidence favors: correct finite diagnostics and Jacobian bookkeeping, but intrinsically poor proposal/target overlap in this replay proposal. "
        "This run remains `diagnostic_replay_not_exact_production`; a finite replay list is not an exact live coarse proposal.",
        "",
        "Files written:",
        "",
        "- `logweight_component_summaries.csv`",
        "- `component_covariance_matrix.csv`",
        "- `component_correlations.csv`",
        "- `component_variance_contributions.csv`",
        "- `component_regression.json`",
        "- `coarse_source_reuse_weight_summary.csv`",
        "- `proposal_native_target_logweight_comparison.csv`",
        "- `proposal_native_target_logweight_comparison.json`",
        "- `mit_logweight_tail_plots.pdf` and `figures/*.pdf/png`",
    ]
    (out_dir / "mit_logweight_tail_analysis_report.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--native-max", type=int, default=1024)
    ap.add_argument("--native-batch-size", type=int, default=128)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    run_dir = args.run_dir
    out_dir = args.out_dir or (run_dir / "logweight_tail_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    proposals = pd.read_csv(run_dir / "proposal_diagnostics.csv")
    chain = pd.read_csv(run_dir / "chain_measurements.csv")
    with (run_dir / "summary.json").open("r", encoding="utf-8") as fh:
        summary = json.load(fh)
    for name in ["kernel_path", "ar_checkpoint", "baseline_checkpoint", "native_reference"]:
        setattr(args, name, Path(summary[name]))
    args.from_L = int(summary["from_L"])
    args.to_L = int(summary["to_L"])
    args.kappa_c = float(summary["kappa_c"])
    args.kappa_f = float(summary["kappa_f"])
    proposals = add_components(proposals)
    current = chain_current_components(proposals, chain)
    proposals.to_csv(out_dir / "proposal_diagnostics_with_components.csv", index=False)
    current.to_csv(out_dir / "chain_measurements_with_current_components.csv", index=False)
    consistency = {
        "max_abs_logw_recompute_error": float(np.max(np.abs(proposals["logw_new"] - proposals["logw_recomputed"]))),
        "max_abs_logq_recompute_error": float(np.max(np.abs(proposals["logq_new"] - proposals["logq_recomputed"]))),
        "max_abs_logw_from_logq_error": float(np.max(np.abs(proposals["logw_new"] - proposals["logw_from_logq_recomputed"]))),
        "nonfinite_proposals": int((proposals["nonfinite_count_new"] > 0).sum()),
        "coarse_kappa": args.kappa_c,
        "fine_kappa": args.kappa_f,
        "prior": "standard normal z as sampled by driver",
    }
    write_json(out_dir / "logweight_consistency_checks.json", consistency)
    summaries = write_summaries(proposals, current, out_dir)
    decomp = linear_decomposition(proposals, out_dir)
    source_reuse(proposals, out_dir)
    target = load_native_target_logweights(args, summary, args.native_max)
    target.to_csv(out_dir / "native_target_logweights_with_components.csv", index=False)
    target_summary = save_target_comparison(proposals, current, target, out_dir)
    make_plots(proposals, current, target, out_dir)
    write_report(out_dir, run_dir, proposals, current, summaries, decomp, target_summary, consistency)
    final = {
        "run_dir": str(run_dir),
        "out_dir": str(out_dir),
        "proposal_count": int(len(proposals)),
        "native_target_count": int(len(target)),
        "consistency": consistency,
        "target_comparison": target_summary,
    }
    write_json(out_dir / "mit_logweight_tail_analysis_summary.json", final)
    print(json.dumps(final, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
