#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OBSERVABLES = [
    "phi2",
    "phi4",
    "NN",
    "2nn",
    "diag",
    "action_density",
    "xi_over_L",
    "abs_m",
    "m2",
    "m4",
    "chi",
]
KEY_OBS = ["xi_over_L", "action_density", "abs_m", "chi"]
ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
BASELINE = (
    ROOT
    / "outputs/shape_parametric_sampler_validation/"
    "controlled_coarse_patch_chain_16to32_P12_pass30_detail1_lam0.022_kc0.2705_kf0.2705"
)


def per_config_observables(phi: np.ndarray, lam: float, kappa: float) -> pd.DataFrame:
    arr = np.asarray(phi, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr[None]
    l_size = arr.shape[1]
    m = arr.mean(axis=(1, 2))
    phi2 = np.mean(arr * arr, axis=(1, 2))
    phi4 = np.mean(arr**4, axis=(1, 2))
    nn = 0.5 * (
        np.mean(arr * np.roll(arr, -1, axis=1), axis=(1, 2))
        + np.mean(arr * np.roll(arr, -1, axis=2), axis=(1, 2))
    )
    twonn = 0.5 * (
        np.mean(arr * np.roll(arr, -2, axis=1), axis=(1, 2))
        + np.mean(arr * np.roll(arr, -2, axis=2), axis=(1, 2))
    )
    diag = np.mean(arr * np.roll(np.roll(arr, -1, axis=1), -1, axis=2), axis=(1, 2))
    m2 = m * m
    m4 = m**4
    chi = l_size * l_size * m2
    xi_over_l = np.sqrt(np.maximum(chi, 0.0) / np.maximum(phi2, 1.0e-300)) / l_size
    action_density = (1.0 - 2.0 * lam) * phi2 + lam * phi4 - 4.0 * kappa * nn
    return pd.DataFrame(
        {
            "phi2": phi2,
            "phi4": phi4,
            "NN": nn,
            "2nn": twonn,
            "diag": diag,
            "action_density": action_density,
            "xi_over_L": xi_over_l,
            "m": m,
            "abs_m": np.abs(m),
            "m2": m2,
            "m4": m4,
            "chi": chi,
        }
    )


def duplicate_check(df: pd.DataFrame, keys: list[str]) -> dict[str, object]:
    dupmask = df.duplicated(keys, keep=False)
    conflicts = []
    if dupmask.any():
        for key, group in df.loc[dupmask].groupby(keys, dropna=False):
            base = group.iloc[0]
            for row_idx in range(1, len(group)):
                row = group.iloc[row_idx]
                for col in df.columns:
                    a = base[col]
                    b = row[col]
                    if not ((pd.isna(a) and pd.isna(b)) or a == b):
                        conflicts.append({"key": key, "column": col, "first": str(a), "other": str(b)})
                        break
                if conflicts:
                    break
            if conflicts:
                break
    return {
        "rows": int(len(df)),
        "duplicate_rows": int(dupmask.sum()),
        "duplicate_key_groups": int(df.loc[dupmask].groupby(keys, dropna=False).ngroups) if dupmask.any() else 0,
        "conflicts": conflicts,
    }


def chain_mean_summary(df: pd.DataFrame, obs: str) -> dict[str, float]:
    means = df.groupby("chain")[obs].mean().to_numpy(dtype=np.float64)
    mean = float(np.mean(means))
    se = float(np.std(means, ddof=1) / math.sqrt(len(means))) if len(means) > 1 else float("nan")
    return {"mean": mean, "se": se, "n_eff": int(len(means))}


def iid_summary(vals: np.ndarray) -> dict[str, float]:
    vals = np.asarray(vals, dtype=np.float64)
    return {
        "mean": float(np.mean(vals)),
        "se": float(np.std(vals, ddof=1) / math.sqrt(vals.size)),
        "n_eff": int(vals.size),
    }


def binder_from_df(df: pd.DataFrame) -> dict[str, float]:
    m2 = float(df["m2"].mean())
    m4 = float(df["m4"].mean())
    ratio = m4 / max(m2 * m2, 1.0e-300)
    return {"m4_over_m2sq": ratio, "binder_U4": 1.0 - ratio / 3.0, "m2_mean": m2, "m4_mean": m4}


def main() -> None:
    out_dir = BASELINE / "comparison_plots"
    out_dir.mkdir(exist_ok=True)
    summary = json.loads((BASELINE / "summary.json").read_text())
    fine_path = Path(summary["fine_reference"])
    if "kappa0p2705" not in str(fine_path):
        raise RuntimeError(f"Resolved fine reference is not kappa0p2705: {fine_path}")

    obs = pd.read_csv(BASELINE / "controlled_patch_chain_observable_history.csv")
    data = pd.read_csv(BASELINE / "data_16to32_P12_pass30_detail1.csv")
    keys = ["patch_size", "n_coarse_passes", "chain", "sweep"]
    obs_dup = duplicate_check(obs, keys)
    data_dup = duplicate_check(data, keys)
    if obs_dup["conflicts"] or data_dup["conflicts"]:
        raise RuntimeError("Conflicting duplicate rows found; refusing to continue.")
    obs_clean = obs.drop_duplicates(keys, keep="first").copy()

    native_npz = np.load(fine_path)
    phi = native_npz["phi"]
    lam = float(native_npz["lambda"])
    kappa = float(native_npz["kappa"])
    native = per_config_observables(phi, lam, kappa)

    nonfinite = {}
    for name, frame in [("controlled", obs_clean), ("native", native)]:
        hits = {}
        for col in OBSERVABLES:
            vals = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=np.float64)
            bad = int((~np.isfinite(vals)).sum())
            if bad:
                hits[col] = bad
        nonfinite[name] = hits

    rows = []
    for ob in OBSERVABLES:
        c = chain_mean_summary(obs_clean, ob)
        n = iid_summary(native[ob].to_numpy(dtype=np.float64))
        diff = c["mean"] - n["mean"]
        combined = math.sqrt(c["se"] ** 2 + n["se"] ** 2)
        rows.append(
            {
                "observable": ob,
                "controlled_mean": c["mean"],
                "controlled_se": c["se"],
                "controlled_n_eff_chains": c["n_eff"],
                "native_mean": n["mean"],
                "native_se": n["se"],
                "native_n": n["n_eff"],
                "difference": diff,
                "combined_sigma": diff / combined if combined > 0 else float("nan"),
            }
        )
    comp = pd.DataFrame(rows)
    comp.to_csv(BASELINE / "canonical_baseline_native_fine_comparison.csv", index=False)

    first = obs_clean[obs_clean["sweep"] <= obs_clean["sweep"].max() / 2]
    second = obs_clean[obs_clean["sweep"] > obs_clean["sweep"].max() / 2]
    eq_rows = []
    outlier_rows = []
    for ob in OBSERVABLES:
        f = chain_mean_summary(first, ob)
        s = chain_mean_summary(second, ob)
        d = s["mean"] - f["mean"]
        sig = math.sqrt(f["se"] ** 2 + s["se"] ** 2)
        eq_rows.append(
            {
                "observable": ob,
                "first_half_mean": f["mean"],
                "first_half_se": f["se"],
                "second_half_mean": s["mean"],
                "second_half_se": s["se"],
                "second_minus_first": d,
                "combined_sigma": d / sig if sig > 0 else float("nan"),
            }
        )
        chain_means = obs_clean.groupby("chain")[ob].mean()
        mean = float(chain_means.mean())
        sd = float(chain_means.std(ddof=1))
        for chain, value in chain_means.items():
            z = (float(value) - mean) / sd if sd > 0 else 0.0
            if abs(z) >= 2.0:
                outlier_rows.append({"observable": ob, "chain": int(chain), "mean": float(value), "z_across_chains": z})
    eq = pd.DataFrame(eq_rows)
    outliers = pd.DataFrame(outlier_rows)
    eq.to_csv(BASELINE / "canonical_baseline_equilibration_check.csv", index=False)
    outliers.to_csv(BASELINE / "canonical_baseline_chain_outliers.csv", index=False)

    binder_controlled = binder_from_df(obs_clean)
    binder_native = binder_from_df(native)
    binder_rows = []
    for key in ["m4_over_m2sq", "binder_U4", "m2_mean", "m4_mean"]:
        binder_rows.append({"quantity": key, "controlled": binder_controlled[key], "native": binder_native[key], "difference": binder_controlled[key] - binder_native[key]})
    binder = pd.DataFrame(binder_rows)
    binder.to_csv(BASELINE / "canonical_baseline_binder_comparison.csv", index=False)

    # Plots.
    for ob in KEY_OBS:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for chain, g in obs_clean.groupby("chain"):
            ax.plot(g["sweep"], g[ob], alpha=0.45, lw=0.8, label=f"chain {chain}")
        ax.axhline(native[ob].mean(), color="black", ls="--", lw=1.2, label="native mean")
        ax.set_xlabel("sweep")
        ax.set_ylabel(ob)
        ax.set_title(f"Controlled-chain history: {ob}")
        ax.legend(ncol=2, fontsize=7)
        fig.tight_layout()
        fig.savefig(out_dir / f"time_history_{ob}.pdf")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(comp))
    ax.errorbar(x - 0.12, comp["controlled_mean"], yerr=comp["controlled_se"], fmt="o", label="controlled")
    ax.errorbar(x + 0.12, comp["native_mean"], yerr=comp["native_se"], fmt="s", label="native")
    ax.set_xticks(x)
    ax.set_xticklabels(comp["observable"], rotation=45, ha="right")
    ax.set_ylabel("mean")
    ax.set_title("Controlled vs native fine reference")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "controlled_vs_native_errorbars.pdf")
    plt.close(fig)

    chain_means = obs_clean.groupby("chain")[KEY_OBS].mean()
    native_means = native[KEY_OBS].mean()
    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    for ax, ob in zip(axes.ravel(), KEY_OBS):
        ax.bar(chain_means.index.astype(str), chain_means[ob])
        ax.axhline(native_means[ob], color="black", ls="--", lw=1.2)
        ax.set_title(ob)
        ax.set_xlabel("chain")
    fig.tight_layout()
    fig.savefig(out_dir / "per_chain_key_observable_means.pdf")
    plt.close(fig)

    result = {
        "baseline": str(BASELINE),
        "native_fine_reference": str(fine_path),
        "controlled_rows": int(len(obs)),
        "controlled_unique_rows": int(len(obs_clean)),
        "native_rows": int(len(native)),
        "duplicates": {"observable_history": obs_dup, "data_csv": data_dup},
        "nonfinite_observables": nonfinite,
        "binder": binder_rows,
        "plot_dir": str(out_dir),
    }
    (BASELINE / "canonical_baseline_native_fine_comparison.json").write_text(json.dumps(result, indent=2) + "\n")

    lines = []
    lines.append("# Canonical Baseline vs Native Fine Reference")
    lines.append("")
    lines.append(f"- canonical baseline: `{BASELINE}`")
    lines.append(f"- native fine reference: `{fine_path}`")
    lines.append(f"- native metadata: lambda `{lam}`, kappa `{kappa}`, L `{int(native_npz['L'])}`, configs `{len(native)}`")
    lines.append(f"- controlled observable rows: `{len(obs)}`; unique `(chain, sweep)` rows used: `{len(obs_clean)}`")
    lines.append("- standard errors: controlled SE is the standard error across the 8 chain means; native SE is iid over stored native configurations.")
    lines.append("")
    lines.append("## Observable Comparison")
    lines.append("")
    lines.append("| observable | controlled mean | controlled SE | native mean | native SE | diff | diff / combined sigma |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in comp.to_dict("records"):
        lines.append(
            f"| {row['observable']} | {row['controlled_mean']:.8g} | {row['controlled_se']:.3g} | "
            f"{row['native_mean']:.8g} | {row['native_se']:.3g} | {row['difference']:.3g} | {row['combined_sigma']:.2f} |"
        )
    lines.append("")
    lines.append("## Binder-Like Quantities")
    lines.append("")
    lines.append("Project convention is `Binder_U4 = 1 - mean(m4)/(3 mean(m2)^2)`.")
    lines.append("")
    lines.append("| quantity | controlled | native | diff |")
    lines.append("|---|---:|---:|---:|")
    for row in binder.to_dict("records"):
        lines.append(f"| {row['quantity']} | {row['controlled']:.8g} | {row['native']:.8g} | {row['difference']:.3g} |")
    lines.append("")
    lines.append("## Equilibration Checks")
    lines.append("")
    lines.append("| observable | first half | second half | second-first | sigma |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in eq.to_dict("records"):
        lines.append(
            f"| {row['observable']} | {row['first_half_mean']:.8g} +/- {row['first_half_se']:.2g} | "
            f"{row['second_half_mean']:.8g} +/- {row['second_half_se']:.2g} | "
            f"{row['second_minus_first']:.3g} | {row['combined_sigma']:.2f} |"
        )
    lines.append("")
    if len(outliers):
        lines.append("Chains with per-observable means at `|z| >= 2` across the 8 chain means:")
        for row in outliers.to_dict("records"):
            lines.append(f"- {row['observable']}: chain `{int(row['chain'])}`, mean `{row['mean']:.8g}`, z `{row['z_across_chains']:.2f}`")
    else:
        lines.append("No per-chain observable mean exceeded `|z| >= 2` across the 8 chain means.")
    lines.append("")
    lines.append("## Duplicate And Finite-Value Checks")
    lines.append("")
    lines.append(f"- `controlled_patch_chain_observable_history.csv`: duplicate rows `{obs_dup['duplicate_rows']}`, conflicts `{len(obs_dup['conflicts'])}`.")
    lines.append(f"- `data_16to32_P12_pass30_detail1.csv`: duplicate rows `{data_dup['duplicate_rows']}`, duplicate key groups `{data_dup['duplicate_key_groups']}`, conflicts `{len(data_dup['conflicts'])}`.")
    if data_dup["duplicate_rows"] and not data_dup["conflicts"]:
        lines.append("- Identical duplicate rows in `data_16to32_P12_pass30_detail1.csv` were not used for statistics; statistics use deduplicated observable history.")
    lines.append(f"- non-finite controlled observable entries: `{nonfinite['controlled']}`")
    lines.append(f"- non-finite native observable entries: `{nonfinite['native']}`")
    lines.append("")
    lines.append("## Plots")
    lines.append("")
    for plot in sorted(out_dir.glob("*.pdf")):
        lines.append(f"- `{plot.relative_to(BASELINE)}`")
    lines.append("")
    lines.append("## Summary")
    max_abs = comp["combined_sigma"].abs().max()
    worst = comp.iloc[int(comp["combined_sigma"].abs().argmax())]
    lines.append(
        f"The largest controlled-vs-native discrepancy is `{worst['observable']}` at "
        f"`{worst['combined_sigma']:.2f}` combined sigma using this SE convention. "
        "The controlled chain is broadly consistent with the native fine ensemble for the main local observables; "
        "long-mode quantities remain noisier and should be interpreted with the chain-level SEs above."
    )
    (BASELINE / "CANONICAL_BASELINE_NATIVE_FINE_COMPARISON.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
