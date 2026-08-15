#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FROZEN = PROJECT_ROOT / "InverseBlocking_lam0p022_frozen"

PRIMARY_OBSERVABLES = ["phi2", "phi4", "NN", "action_density", "susceptibility", "Binder_U4", "xi_over_L"]
SECTOR_DIAGNOSTICS = ["m", "abs_m"]
OBSERVABLES = PRIMARY_OBSERVABLES + SECTOR_DIAGNOSTICS
SIMPLE_COLUMNS = {"m", "abs_m", "phi2", "phi4", "NN", "action_density"}
AUTOCORR_KEYS = ["action_density", "phi2", "phi4", "NN", "susceptibility", "Binder_U4"]
BLOCK_SIZES = [1, 2, 5, 10, 20, 50]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def as_float(row: dict[str, str], key: str) -> float:
    if key == "chi":
        key = "susceptibility"
    return float(row[key])


def chain_ids(rows: list[dict[str, str]]) -> list[int]:
    return sorted({int(r["chain_id"]) for r in rows})


def select_rows(rows: list[dict[str, str]], chain: int | None = None, start: int | None = None, stop: int | None = None) -> list[dict[str, str]]:
    out = []
    for row in rows:
        if chain is not None and int(row["chain_id"]) != chain:
            continue
        sweep = int(row["sweep"])
        if start is not None and sweep < start:
            continue
        if stop is not None and sweep >= stop:
            continue
        out.append(row)
    return out


def bin_observable(rows: list[dict[str, str]], observable: str, lattice_l: int = 16) -> float:
    if not rows:
        return float("nan")
    if observable in SIMPLE_COLUMNS:
        return float(np.mean([as_float(r, observable) for r in rows]))
    m2 = np.asarray([as_float(r, "m2") for r in rows], dtype=np.float64)
    m4 = np.asarray([as_float(r, "m4") for r in rows], dtype=np.float64)
    if observable == "Binder_U4":
        return float(1.0 - np.mean(m4) / (3.0 * max(np.mean(m2) ** 2, 1.0e-300)))
    if observable == "susceptibility":
        return float(lattice_l * lattice_l * np.mean(m2))
    if observable == "xi_over_L":
        phi2 = float(np.mean([as_float(r, "phi2") for r in rows]))
        chi = lattice_l * lattice_l * float(np.mean(m2))
        return float(math.sqrt(max(chi, 0.0) / max(phi2, 1.0e-300)) / lattice_l)
    raise KeyError(observable)


def row_series(rows: list[dict[str, str]], observable: str, lattice_l: int = 16) -> np.ndarray:
    if observable in SIMPLE_COLUMNS:
        return np.asarray([as_float(r, observable) for r in rows], dtype=np.float64)
    if observable == "susceptibility":
        return lattice_l * lattice_l * np.asarray([as_float(r, "m2") for r in rows], dtype=np.float64)
    if observable == "xi_over_L":
        m2 = np.asarray([as_float(r, "m2") for r in rows], dtype=np.float64)
        phi2 = np.asarray([as_float(r, "phi2") for r in rows], dtype=np.float64)
        return np.sqrt(np.maximum(lattice_l * lattice_l * m2, 0.0) / np.maximum(phi2, 1.0e-300)) / lattice_l
    if observable == "Binder_U4":
        m2 = np.asarray([as_float(r, "m2") for r in rows], dtype=np.float64)
        m4 = np.asarray([as_float(r, "m4") for r in rows], dtype=np.float64)
        return 1.0 - m4 / (3.0 * np.maximum(m2 * m2, 1.0e-300))
    raise KeyError(observable)


def mean_se(values: list[float] | np.ndarray) -> tuple[float, float, float]:
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    if x.size == 1:
        return float(x[0]), float("nan"), 0.0
    std = float(np.std(x, ddof=1))
    return float(np.mean(x)), std / math.sqrt(x.size), std


def load_direct_reference(run_dir: Path) -> dict[str, dict[str, float]]:
    p = run_dir / "target_distribution_history_debug.json"
    if p.exists():
        data = json.loads(p.read_text())
        ref = data.get("reference_stats", {})
        out = {}
        for key, row in ref.items():
            out[key] = {"mean": float(row["mean"]), "se": float(row.get("se", float("nan")))}
        if "susceptibility" in out:
            out["chi"] = out["susceptibility"]
        return out
    raise FileNotFoundError(f"reference stats not found: {p}")


def load_old_transport_rows(old_run: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for idx, path in enumerate(sorted(old_run.glob("chain*/samepatch_latent_observable_timeseries.csv"))):
        for row in read_csv(path):
            r = dict(row)
            r["chain_id"] = str(idx)
            rows.append(r)
    return rows


def old_reference_from_bins(old_rows: list[dict[str, str]], observable: str) -> dict[str, float]:
    vals = []
    for chain in chain_ids(old_rows):
        cr = select_rows(old_rows, chain=chain)
        vals.append(bin_observable(cr, observable))
    mean, se, std = mean_se(vals)
    return {"mean": mean, "se": se, "std": std, "n_bins": len(vals)}


def z_score(mean: float, ref: dict[str, float], se: float) -> float:
    denom2 = 0.0
    if math.isfinite(se):
        denom2 += se * se
    ref_se = ref.get("se", float("nan"))
    if math.isfinite(ref_se):
        denom2 += ref_se * ref_se
    denom = math.sqrt(denom2) if denom2 > 0.0 else float("nan")
    return (mean - ref["mean"]) / denom if math.isfinite(denom) and denom > 0.0 else float("nan")


def split_chain_summary(rows: list[dict[str, str]], old_rows: list[dict[str, str]], direct_ref: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    out = []
    n_sweeps = max(int(r["sweep"]) for r in rows) + 1
    levels = [("full_chain", 1), ("half_chain", 2), ("quarter_chain", 4)]
    old_refs = {obs: old_reference_from_bins(old_rows, obs) for obs in OBSERVABLES} if old_rows else {}
    for label, parts in levels:
        width = n_sweeps // parts
        for obs in OBSERVABLES:
            vals = []
            for chain in chain_ids(rows):
                for part in range(parts):
                    start = part * width
                    stop = (part + 1) * width if part < parts - 1 else n_sweeps
                    vals.append(bin_observable(select_rows(rows, chain=chain, start=start, stop=stop), obs))
            mean, se, std = mean_se(vals)
            dref = direct_ref[obs]
            oref = old_refs.get(obs, {"mean": float("nan"), "se": float("nan")})
            out.append(
                {
                    "binning": label,
                    "category": "primary" if obs in PRIMARY_OBSERVABLES else "sector_diagnostic_not_pass_fail",
                    "observable": obs,
                    "n_bins": len(vals),
                    "bin_width_sweeps": width,
                    "mean": mean,
                    "std_across_bins": std,
                    "standard_error": se,
                    "direct_reference_mean": dref["mean"],
                    "direct_reference_se": dref.get("se", float("nan")),
                    "z_vs_direct_reference": z_score(mean, dref, se),
                    "old_transport_mean": oref.get("mean", float("nan")),
                    "old_transport_se": oref.get("se", float("nan")),
                    "z_vs_old_transport": z_score(mean, oref, se) if math.isfinite(oref.get("mean", float("nan"))) else float("nan"),
                }
            )
    return out


def window_summary(rows: list[dict[str, str]], old_rows: list[dict[str, str]], direct_ref: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    n_sweeps = max(int(r["sweep"]) for r in rows) + 1
    windows = [
        ("first_half", 0, n_sweeps // 2),
        ("second_half", n_sweeps // 2, n_sweeps),
        ("quarter_1", 0, n_sweeps // 4),
        ("quarter_2", n_sweeps // 4, n_sweeps // 2),
        ("quarter_3", n_sweeps // 2, 3 * n_sweeps // 4),
        ("quarter_4", 3 * n_sweeps // 4, n_sweeps),
        ("last_25pct", 3 * n_sweeps // 4, n_sweeps),
        ("last_50pct", n_sweeps // 2, n_sweeps),
    ]
    out = []
    for name, start, stop in windows:
        for obs in OBSERVABLES:
            vals = [bin_observable(select_rows(rows, chain=chain, start=start, stop=stop), obs) for chain in chain_ids(rows)]
            mean, se, std = mean_se(vals)
            old_vals = [bin_observable(select_rows(old_rows, chain=chain, start=start, stop=stop), obs) for chain in chain_ids(old_rows)] if old_rows else []
            old_mean, old_se, _ = mean_se(old_vals)
            dref = direct_ref[obs]
            out.append(
                {
                    "window": name,
                    "category": "primary" if obs in PRIMARY_OBSERVABLES else "sector_diagnostic_not_pass_fail",
                    "sweep_start": start,
                    "sweep_stop": stop,
                    "observable": obs,
                    "n_chain_bins": len(vals),
                    "mean": mean,
                    "standard_error": se,
                    "std_across_chains": std,
                    "direct_reference_mean": dref["mean"],
                    "z_vs_direct_reference": z_score(mean, dref, se),
                    "old_transport_mean_same_window": old_mean,
                    "old_transport_se_same_window": old_se,
                    "z_vs_old_transport_same_window": z_score(mean, {"mean": old_mean, "se": old_se}, se) if math.isfinite(old_mean) else float("nan"),
                }
            )
    return out


def blocking_analysis(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out = []
    for chain in chain_ids(rows):
        cr = select_rows(rows, chain=chain)
        for obs in OBSERVABLES:
            series = row_series(cr, obs)
            for block_size in BLOCK_SIZES:
                n_blocks = len(series) // block_size
                if n_blocks < 2:
                    continue
                blocks = series[: n_blocks * block_size].reshape(n_blocks, block_size).mean(axis=1)
                mean, se, std = mean_se(blocks)
                out.append(
                    {
                        "chain_id": chain,
                        "category": "primary" if obs in PRIMARY_OBSERVABLES else "sector_diagnostic_not_pass_fail",
                        "observable": obs,
                        "block_size_sweeps": block_size,
                        "n_blocks": n_blocks,
                        "mean": mean,
                        "standard_error": se,
                        "std_block_means": std,
                    }
                )
    return out


def autocorr_1d(x: np.ndarray, max_lag: int) -> np.ndarray:
    y = np.asarray(x, dtype=np.float64)
    y = y - np.mean(y)
    var = np.dot(y, y) / len(y)
    if var <= 0.0:
        return np.full(max_lag + 1, np.nan)
    out = np.empty(max_lag + 1, dtype=np.float64)
    out[0] = 1.0
    for lag in range(1, max_lag + 1):
        out[lag] = np.dot(y[:-lag], y[lag:]) / ((len(y) - lag) * var)
    return out


def tau_int_initial_positive(ac: np.ndarray) -> tuple[float, int]:
    tau = 0.5
    cutoff = 0
    for lag in range(1, len(ac)):
        if not math.isfinite(float(ac[lag])) or ac[lag] <= 0.0:
            break
        tau += float(ac[lag])
        cutoff = lag
    return tau, cutoff


def autocorrelation_summary(rows: list[dict[str, str]], out_dir: Path) -> list[dict[str, Any]]:
    max_lag = min(150, max(int(r["sweep"]) for r in rows) // 2)
    summary = []
    ac_plot: dict[str, dict[int, np.ndarray]] = {}
    for obs in AUTOCORR_KEYS:
        ac_plot[obs] = {}
        taus = []
        for chain in chain_ids(rows):
            cr = select_rows(rows, chain=chain)
            ac = autocorr_1d(row_series(cr, obs), max_lag=max_lag)
            tau, cutoff = tau_int_initial_positive(ac)
            ess = len(cr) / max(2.0 * tau, 1.0e-300) if math.isfinite(tau) else float("nan")
            summary.append(
                {
                    "chain_id": chain,
                    "category": "primary" if obs in PRIMARY_OBSERVABLES else "sector_diagnostic_not_pass_fail",
                    "observable": obs,
                    "n": len(cr),
                    "max_lag": max_lag,
                    "tau_int_initial_positive": tau,
                    "positive_cutoff_lag": cutoff,
                    "effective_sample_size": ess,
                    "caveat": "Binder_U4 uses a per-configuration proxy; ensemble Binder is nonlinear." if obs == "Binder_U4" else "short 500-sweep chains; tau estimate is noisy.",
                }
            )
            ac_plot[obs][chain] = ac
    plot_autocorr(ac_plot, out_dir / "sector_aware_autocorrelation_plots.pdf")
    return summary


def plot_autocorr(ac_plot: dict[str, dict[int, np.ndarray]], out_pdf: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(ac_plot), 1, figsize=(8.5, 10.5), sharex=True)
    if len(ac_plot) == 1:
        axes = [axes]
    for ax, (obs, by_chain) in zip(axes, ac_plot.items()):
        for chain, ac in by_chain.items():
            ax.plot(np.arange(len(ac)), ac, lw=0.8, alpha=0.6, label=f"c{chain}")
        ax.axhline(0.0, color="black", lw=0.8)
        ax.set_ylabel(obs)
        ax.grid(alpha=0.25)
    axes[0].legend(ncol=4, fontsize=7)
    axes[-1].set_xlabel("lag (sweeps)")
    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)


def plot_improved_histories(
    rows: list[dict[str, str]],
    old_rows: list[dict[str, str]],
    direct_ref: dict[str, dict[str, float]],
    out_pdf: Path,
    keys: list[str],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ncols = 2
    nrows = int(math.ceil(len(keys) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11.0, 3.3 * nrows), sharex=True)
    axes = axes.reshape(-1)
    chains = chain_ids(rows)
    for ax, obs in zip(axes, keys):
        pooled_by_sweep: dict[int, list[float]] = {}
        for chain in chains:
            cr = select_rows(rows, chain=chain)
            x = np.asarray([int(r["sweep"]) for r in cr])
            y = row_series(cr, obs)
            running = np.cumsum(y) / np.arange(1, len(y) + 1)
            ax.plot(x, running, lw=0.75, alpha=0.55, label=f"chain {chain}" if obs == keys[0] else None)
            for sweep, val in zip(x, y):
                pooled_by_sweep.setdefault(int(sweep), []).append(float(val))
        xs = np.asarray(sorted(pooled_by_sweep))
        pooled = np.asarray([np.mean(pooled_by_sweep[i]) for i in xs])
        pooled_running = np.cumsum(pooled) / np.arange(1, len(pooled) + 1)
        ax.plot(xs, pooled_running, color="black", lw=1.5, label="pooled running" if obs == keys[0] else None)
        late = [r for r in rows if int(r["sweep"]) >= int(0.75 * (max(xs) + 1))]
        ax.axhline(bin_observable(late, obs), color="tab:green", ls="-.", lw=1.0, label="late 25%" if obs == keys[0] else None)
        if obs in direct_ref:
            ref = direct_ref[obs]
            ax.axhline(ref["mean"], color="tab:red", ls="--", lw=1.1, label="direct ref" if obs == keys[0] else None)
            if math.isfinite(ref.get("se", float("nan"))):
                ax.axhspan(ref["mean"] - ref["se"], ref["mean"] + ref["se"], color="tab:red", alpha=0.12)
        if old_rows:
            old_ref = old_reference_from_bins(old_rows, obs)
            ax.axhline(old_ref["mean"], color="tab:purple", ls=":", lw=1.1, label="old transported" if obs == keys[0] else None)
            if math.isfinite(old_ref["se"]):
                ax.axhspan(old_ref["mean"] - old_ref["se"], old_ref["mean"] + old_ref["se"], color="tab:purple", alpha=0.10)
        ax.set_title(obs)
        ax.grid(alpha=0.25)
    for ax in axes[len(keys) :]:
        ax.axis("off")
    axes[0].legend(ncol=2, fontsize=7)
    axes[-1].set_xlabel("sweep")
    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)


def write_report(out_dir: Path, split_rows: list[dict[str, Any]], window_rows: list[dict[str, Any]], auto_rows: list[dict[str, Any]]) -> None:
    def find(rows: list[dict[str, Any]], **kw):
        for r in rows:
            if all(r.get(k) == v for k, v in kw.items()):
                return r
        raise KeyError(kw)

    primary_full = [r for r in split_rows if r["binning"] == "full_chain" and r["category"] == "primary"]
    sector_full = [r for r in split_rows if r["binning"] == "full_chain" and r["category"] != "primary"]
    primary_max_z = max(abs(float(r["z_vs_direct_reference"])) for r in primary_full if math.isfinite(float(r["z_vs_direct_reference"])))
    sector_max_z = max(abs(float(r["z_vs_direct_reference"])) for r in sector_full if math.isfinite(float(r["z_vs_direct_reference"])))
    lines = [
        "# Sector-Aware Statistical Diagnostics",
        "",
        "Input measurements are end-of-sweep post-A/R Markov-chain states.",
        "",
        f"- max |z| primary observables only: `{primary_max_z:.6g}`",
        f"- max |z| sector diagnostics, not pass/fail: `{sector_max_z:.6g}`",
        "",
        "## Split-Chain Binning",
        "",
        "| binning | observable | mean | SE | z vs direct | z vs old transported |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for binning in ["full_chain", "half_chain", "quarter_chain"]:
        for obs in ["action_density", "phi2", "phi4", "NN", "susceptibility", "Binder_U4", "xi_over_L"]:
            r = find(split_rows, binning=binning, observable=obs)
            lines.append(
                f"| {binning} | `{obs}` | {r['mean']:.6g} | {r['standard_error']:.6g} | "
                f"{r['z_vs_direct_reference']:.6g} | {r['z_vs_old_transport']:.6g} |"
            )
    lines += [
        "",
        "## Sector Diagnostics (Not Pass/Fail)",
        "",
        "| binning | diagnostic | mean | SE | z vs direct | z vs old transported |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for binning in ["full_chain", "half_chain", "quarter_chain"]:
        for obs in SECTOR_DIAGNOSTICS:
            r = find(split_rows, binning=binning, observable=obs)
            lines.append(
                f"| {binning} | `{obs}` | {r['mean']:.6g} | {r['standard_error']:.6g} | "
                f"{r['z_vs_direct_reference']:.6g} | {r['z_vs_old_transport']:.6g} |"
            )
    lines += [
        "",
        "## Window Stability",
        "",
        "| observable | first half | second half | quarter 1 | quarter 2 | quarter 3 | quarter 4 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for obs in ["action_density", "phi2", "phi4", "NN", "susceptibility", "Binder_U4", "xi_over_L"]:
        vals = {r["window"]: r for r in window_rows if r["observable"] == obs}
        lines.append(
            f"| `{obs}` | {vals['first_half']['mean']:.6g} | {vals['second_half']['mean']:.6g} | "
            f"{vals['quarter_1']['mean']:.6g} | {vals['quarter_2']['mean']:.6g} | "
            f"{vals['quarter_3']['mean']:.6g} | {vals['quarter_4']['mean']:.6g} |"
        )
    lines += [
        "",
        "Window finding: compare the quarter and half-chain rows above; substantial quarter-to-quarter motion indicates residual autocorrelation or slow sector drift.",
        "",
        "## Autocorrelation",
        "",
        "| observable | mean tau_int | min tau_int | max tau_int | mean ESS per chain |",
        "|---|---:|---:|---:|---:|",
    ]
    for obs in AUTOCORR_KEYS:
        vals = [r for r in auto_rows if r["observable"] == obs]
        tau = np.asarray([float(r["tau_int_initial_positive"]) for r in vals], dtype=np.float64)
        ess = np.asarray([float(r["effective_sample_size"]) for r in vals], dtype=np.float64)
        lines.append(f"| `{obs}` | {np.nanmean(tau):.6g} | {np.nanmin(tau):.6g} | {np.nanmax(tau):.6g} | {np.nanmean(ess):.6g} |")
    lines += [
        "",
        "Caveats:",
        "",
        "- The chains are short for reliable autocorrelation estimates, so tau and ESS should be treated as rough diagnostics.",
        "- `Binder_U4` is nonlinear and should be interpreted from bin/window ensembles; the autocorrelation table uses a per-configuration proxy only as a rough stability diagnostic.",
        "- Signed `m`, `abs_m`, sector fractions, and sign flips are sector/mixing diagnostics only, not pass/fail thermodynamic observables.",
        "- Split-chain errors are more conservative than treating all 4000 rows as independent, and the apparent failures remain sensitive to binning/window choice.",
        "",
        "Plots:",
        "",
        "- `sector_aware_primary_running_mean_diagnostics.pdf`",
        "- `sector_aware_sector_running_mean_diagnostics.pdf`",
        "- `sector_aware_autocorrelation_plots.pdf`",
    ]
    (out_dir / "sector_aware_statistical_diagnostics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=PROJECT_ROOT / "perfect_blocking_upsampling" / "outputs" / "shape_parametric_sampler_validation" / "sector_aware_8x500")
    ap.add_argument("--old-run", type=Path, default=FROZEN / "validation" / "rho0p5_every20_validation_8x20k_20260629_083442")
    args = ap.parse_args()

    rows = read_csv(args.run_dir / "observable_timeseries.csv")
    old_rows = load_old_transport_rows(args.old_run)
    direct_ref = load_direct_reference(args.run_dir)
    for alias in [("chi", "susceptibility")]:
        if alias[1] in direct_ref:
            direct_ref[alias[0]] = direct_ref[alias[1]]

    split_rows = split_chain_summary(rows, old_rows, direct_ref)
    win_rows = window_summary(rows, old_rows, direct_ref)
    block_rows = blocking_analysis(rows)
    auto_rows = autocorrelation_summary(rows, args.run_dir)
    plot_improved_histories(
        rows,
        old_rows,
        direct_ref,
        args.run_dir / "sector_aware_primary_running_mean_diagnostics.pdf",
        ["action_density", "phi2", "phi4", "NN", "susceptibility", "Binder_U4", "xi_over_L"],
    )
    plot_improved_histories(
        rows,
        old_rows,
        direct_ref,
        args.run_dir / "sector_aware_sector_running_mean_diagnostics.pdf",
        ["m", "abs_m"],
    )

    write_csv(args.run_dir / "split_chain_binning_summary.csv", split_rows)
    write_csv(args.run_dir / "window_stability_summary.csv", win_rows)
    write_csv(args.run_dir / "blocking_analysis_summary.csv", block_rows)
    write_csv(args.run_dir / "autocorrelation_summary.csv", auto_rows)
    write_report(args.run_dir, split_rows, win_rows, auto_rows)
    print(
        json.dumps(
            {
                "status": "completed",
                "split_chain": str(args.run_dir / "split_chain_binning_summary.csv"),
                "window": str(args.run_dir / "window_stability_summary.csv"),
                "blocking": str(args.run_dir / "blocking_analysis_summary.csv"),
                "autocorrelation": str(args.run_dir / "autocorrelation_summary.csv"),
                "report": str(args.run_dir / "sector_aware_statistical_diagnostics.md"),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
