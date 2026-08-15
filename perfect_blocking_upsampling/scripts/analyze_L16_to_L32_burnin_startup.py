#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

RUN_DIR = Path(
    "perfect_blocking_upsampling/outputs/shape_parametric_sampler_validation/"
    "L16_to_L32_smoke/native_L16_pcn1_8x2000"
)
OUT_DIR = RUN_DIR / "burnin_startup_analysis"
PRIMARY = ["phi2", "phi4", "NN", "action_density", "susceptibility", "Binder_U4", "xi_over_L"]
PRIMARY_FOR_SPLIT_MAX = PRIMARY
AUTOCORR = ["action_density", "phi2", "NN", "Binder_U4"]
SIMPLE = {"phi2", "phi4", "NN", "action_density", "m", "abs_m"}
BURNINS = [0, 50, 100, 200, 300, 500]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0])
    for row in rows[1:]:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def chain_ids(rows: list[dict[str, str]]) -> list[int]:
    return sorted({int(r["chain_id"]) for r in rows})


def select(rows: list[dict[str, str]], chain: int | None = None, start: int | None = None, stop: int | None = None) -> list[dict[str, str]]:
    out = []
    for r in rows:
        sweep = int(r["sweep"])
        if chain is not None and int(r["chain_id"]) != chain:
            continue
        if start is not None and sweep < start:
            continue
        if stop is not None and sweep >= stop:
            continue
        out.append(r)
    return out


def mean_se(vals: list[float] | np.ndarray) -> tuple[float, float, float, int]:
    x = np.asarray(vals, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan"), float("nan"), 0
    if x.size == 1:
        return float(x[0]), float("nan"), 0.0, 1
    std = float(np.std(x, ddof=1))
    return float(np.mean(x)), std / math.sqrt(x.size), std, int(x.size)


def obs_value(rows: list[dict[str, str]], obs: str, L: int = 32) -> float:
    if not rows:
        return float("nan")
    if obs in SIMPLE:
        return float(np.mean([f(r, obs) for r in rows]))
    m2 = np.asarray([f(r, "m2") for r in rows])
    m4 = np.asarray([f(r, "m4") for r in rows])
    if obs == "susceptibility":
        return float(L * L * np.mean(m2))
    if obs == "Binder_U4":
        return float(1.0 - np.mean(m4) / max(3.0 * np.mean(m2) ** 2, 1e-300))
    if obs == "xi_over_L":
        phi2 = np.mean([f(r, "phi2") for r in rows])
        chi = L * L * np.mean(m2)
        return float(math.sqrt(max(chi, 0.0) / max(phi2, 1e-300)) / L)
    raise KeyError(obs)


def series(rows: list[dict[str, str]], obs: str, L: int = 32) -> np.ndarray:
    if obs in SIMPLE:
        return np.asarray([f(r, obs) for r in rows], dtype=float)
    m2 = np.asarray([f(r, "m2") for r in rows], dtype=float)
    if obs == "susceptibility":
        return L * L * m2
    if obs == "Binder_U4":
        m4 = np.asarray([f(r, "m4") for r in rows], dtype=float)
        return 1.0 - m4 / np.maximum(3.0 * m2 * m2, 1e-300)
    if obs == "xi_over_L":
        phi2 = np.asarray([f(r, "phi2") for r in rows], dtype=float)
        return np.sqrt(np.maximum(L * L * m2, 0.0) / np.maximum(phi2, 1e-300)) / L
    raise KeyError(obs)


def autocorr(x: np.ndarray, max_lag: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return np.full(max_lag + 1, np.nan)
    y = x - np.mean(x)
    var = float(np.dot(y, y) / y.size)
    if var <= 0:
        out = np.zeros(max_lag + 1)
        out[0] = 1.0
        return out
    out = np.empty(max_lag + 1)
    out[0] = 1.0
    for lag in range(1, max_lag + 1):
        out[lag] = float(np.dot(y[:-lag], y[lag:]) / ((y.size - lag) * var))
    return out


def tau_initial_positive(x: np.ndarray) -> tuple[float, int]:
    if x.size < 3:
        return float("nan"), 0
    ac = autocorr(x, min(150, max(1, x.size // 3)))
    tau = 0.5
    cutoff = 0
    for lag in range(1, len(ac)):
        if not math.isfinite(float(ac[lag])) or ac[lag] <= 0:
            break
        tau += float(ac[lag])
        cutoff = lag
    return tau, cutoff


def sign_stats(rows: list[dict[str, str]]) -> dict[str, float | int]:
    vals = np.asarray([f(r, "m") for r in rows], dtype=float)
    if vals.size == 0:
        return {"fraction_positive": float("nan"), "fraction_negative": float("nan"), "fraction_zero": float("nan"), "sign_flips": 0}
    signs = np.sign(vals)
    nz = signs[signs != 0]
    flips = int(np.sum(nz[1:] != nz[:-1])) if nz.size > 1 else 0
    return {
        "fraction_positive": float(np.mean(vals > 0)),
        "fraction_negative": float(np.mean(vals < 0)),
        "fraction_zero": float(np.mean(vals == 0)),
        "sign_flips": flips,
    }


def total_sign_flips_by_chain(rows: list[dict[str, str]]) -> int:
    return int(sum(int(sign_stats(select(rows, chain=c))["sign_flips"]) for c in chain_ids(rows)))


def refs_from_existing() -> dict[str, dict[str, float]]:
    refs: dict[str, dict[str, float]] = {}
    for row in read_csv(RUN_DIR / "split_chain_binning_summary.csv"):
        if row["binning"] == "full_chain":
            refs[row["observable"]] = {
                "mean": float(row["direct_reference_mean"]),
                "se": float(row["direct_reference_se"]) if row["direct_reference_se"] not in ("", "nan", "NaN") else float("nan"),
            }
    return refs


def z(mean: float, se: float, ref: dict[str, float]) -> float:
    denom2 = 0.0
    if math.isfinite(se):
        denom2 += se * se
    if math.isfinite(ref.get("se", float("nan"))):
        denom2 += ref["se"] * ref["se"]
    return (mean - ref["mean"]) / math.sqrt(denom2) if denom2 > 0 else float("nan")


def split_rows_after_burn(rows: list[dict[str, str]], burn: int, refs: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    kept = [r for r in rows if int(r["sweep"]) >= burn]
    n_remaining = max(int(r["sweep"]) for r in kept) - burn + 1
    out = []
    for label, parts in [("full_chain", 1), ("half_chain", 2), ("quarter_chain", 4)]:
        width = n_remaining // parts
        if width < 50:
            continue
        for obs in PRIMARY:
            vals = []
            for c in chain_ids(kept):
                for p in range(parts):
                    start = burn + p * width
                    stop = burn + (p + 1) * width if p < parts - 1 else burn + n_remaining
                    vals.append(obs_value(select(rows, c, start, stop), obs))
            mean, se, std, n = mean_se(vals)
            out.append(
                {
                    "burn_in_sweeps": burn,
                    "binning": label,
                    "observable": obs,
                    "n_bins": n,
                    "bin_width_sweeps": width,
                    "mean": mean,
                    "std_across_bins": std,
                    "standard_error": se,
                    "direct_reference_mean": refs.get(obs, {}).get("mean", float("nan")),
                    "direct_reference_se": refs.get(obs, {}).get("se", float("nan")),
                    "z_vs_direct_reference": z(mean, se, refs[obs]) if obs in refs else float("nan"),
                }
            )
    return out


def startup_windows(rows: list[dict[str, str]], refs: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    n_sweeps = max(int(r["sweep"]) for r in rows) + 1
    windows = [
        ("1-50", 0, 50),
        ("1-100", 0, 100),
        ("1-200", 0, 200),
        ("1-500", 0, 500),
        ("100-500", 99, 500),
        ("200-500", 199, 500),
        ("500-1000", 499, 1000),
        ("1000-2000", 999, 2000),
        ("last_50pct", n_sweeps // 2, n_sweeps),
        ("last_25pct", 3 * n_sweeps // 4, n_sweeps),
    ]
    out = []
    for name, start, stop in windows:
        wr = select(rows, start=start, stop=stop)
        for scope in ["pooled", "chain_bins"]:
            for obs in PRIMARY + ["m", "abs_m"]:
                if scope == "pooled":
                    mean = obs_value(wr, obs)
                    n = len(wr)
                    if obs in SIMPLE:
                        _, se, std, _ = mean_se(series(wr, obs))
                    else:
                        se = float("nan")
                        std = float("nan")
                else:
                    vals = [obs_value(select(wr, chain=c), obs) for c in chain_ids(wr)]
                    mean, se, std, n = mean_se(vals)
                ref = refs.get(obs)
                out.append(
                    {
                        "window": name,
                        "scope": scope,
                        "csv_sweep_start_inclusive": start,
                        "csv_sweep_stop_exclusive": stop,
                        "observable": obs,
                        "category": "primary_noisy" if obs in {"Binder_U4", "susceptibility"} else ("primary" if obs in PRIMARY else "sector"),
                        "n": n,
                        "mean": mean,
                        "standard_error": se,
                        "std": std,
                        "direct_reference_mean": ref.get("mean", float("nan")) if ref else float("nan"),
                        "z_vs_direct_reference": z(mean, se, ref) if ref else float("nan"),
                    }
                )
        stats = sign_stats(wr)
        out.append(
            {
                "window": name,
                "scope": "pooled",
                "csv_sweep_start_inclusive": start,
                "csv_sweep_stop_exclusive": stop,
                "observable": "sector_occupancy_and_flips",
                "category": "sector",
                "n": len(wr),
                "mean": stats["fraction_positive"],
                "standard_error": float("nan"),
                "std": float("nan"),
                "direct_reference_mean": float("nan"),
                "z_vs_direct_reference": float("nan"),
                "fraction_negative": stats["fraction_negative"],
                "fraction_zero": stats["fraction_zero"],
                "sign_flips": total_sign_flips_by_chain(wr),
            }
        )
        for c in chain_ids(wr):
            cr = select(wr, chain=c)
            stats = sign_stats(cr)
            out.append(
                {
                    "window": name,
                    "scope": f"chain_{c}",
                    "csv_sweep_start_inclusive": start,
                    "csv_sweep_stop_exclusive": stop,
                    "observable": "sector_occupancy_and_flips",
                    "category": "sector",
                    "n": len(cr),
                    "mean": stats["fraction_positive"],
                    "standard_error": float("nan"),
                    "std": float("nan"),
                    "direct_reference_mean": float("nan"),
                    "z_vs_direct_reference": float("nan"),
                    "fraction_negative": stats["fraction_negative"],
                    "fraction_zero": stats["fraction_zero"],
                    "sign_flips": stats["sign_flips"],
                }
            )
    return out


def burnin_sensitivity(rows: list[dict[str, str]], refs: dict[str, dict[str, float]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows = []
    obs_rows = []
    for burn in BURNINS:
        kept = [r for r in rows if int(r["sweep"]) >= burn]
        splits = split_rows_after_burn(rows, burn, refs)
        max_by = {}
        for label in ["full_chain", "half_chain", "quarter_chain"]:
            vals = [abs(float(r["z_vs_direct_reference"])) for r in splits if r["binning"] == label and math.isfinite(float(r["z_vs_direct_reference"]))]
            max_by[label] = max(vals) if vals else float("nan")
        taus = {}
        ess = {}
        for obs in AUTOCORR:
            tau_vals = []
            ess_vals = []
            for c in chain_ids(kept):
                x = series(select(kept, chain=c), obs)
                tau, _ = tau_initial_positive(x)
                tau_vals.append(tau)
                ess_vals.append(x.size / max(2 * tau, 1e-300) if math.isfinite(tau) else float("nan"))
            taus[obs] = float(np.nanmean(tau_vals))
            ess[obs] = float(np.nanmean(ess_vals))
        st = sign_stats(kept)
        summary_rows.append(
            {
                "burn_in_sweeps": burn,
                "remaining_states_per_chain": min(len(select(kept, chain=c)) for c in chain_ids(kept)),
                "total_retained_states": len(kept),
                "full_chain_split_max_abs_primary_z": max_by["full_chain"],
                "half_chain_split_max_abs_primary_z": max_by["half_chain"],
                "quarter_chain_split_max_abs_primary_z": max_by["quarter_chain"],
                "tau_action_density_mean": taus["action_density"],
                "tau_phi2_mean": taus["phi2"],
                "tau_NN_mean": taus["NN"],
                "tau_Binder_U4_mean": taus["Binder_U4"],
                "ESS_per_chain_action_density_mean": ess["action_density"],
                "ESS_per_chain_phi2_mean": ess["phi2"],
                "ESS_per_chain_NN_mean": ess["NN"],
                "ESS_per_chain_Binder_U4_mean": ess["Binder_U4"],
                "fraction_positive": st["fraction_positive"],
                "fraction_negative": st["fraction_negative"],
                "fraction_zero": st["fraction_zero"],
                "sign_flips_total": total_sign_flips_by_chain(kept),
            }
        )
        for obs in PRIMARY:
            vals = [obs_value(select(kept, chain=c), obs) for c in chain_ids(kept)]
            mean, se, std, n = mean_se(vals)
            obs_rows.append(
                {
                    "burn_in_sweeps": burn,
                    "observable": obs,
                    "n_chain_bins": n,
                    "mean": mean,
                    "standard_error": se,
                    "std_across_chains": std,
                    "direct_reference_mean": refs.get(obs, {}).get("mean", float("nan")),
                    "direct_reference_se": refs.get(obs, {}).get("se", float("nan")),
                    "z_vs_direct_reference": z(mean, se, refs[obs]) if obs in refs else float("nan"),
                    "noisy_caveat": "yes" if obs in {"Binder_U4", "susceptibility"} else "no",
                }
            )
    return summary_rows, obs_rows


def plot_outputs(rows: list[dict[str, str]], refs: dict[str, dict[str, float]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    burns = [50, 100, 200, 500]

    def running(x: np.ndarray) -> np.ndarray:
        return np.cumsum(x) / np.arange(1, len(x) + 1)

    with PdfPages(OUT_DIR / "burnin_running_means.pdf") as pdf:
        for obs in ["phi2", "phi4", "NN", "action_density", "susceptibility"]:
            fig, ax = plt.subplots(figsize=(8.5, 4.8))
            pooled_num = None
            pooled_den = None
            for c in chain_ids(rows):
                cr = select(rows, chain=c)
                x = series(cr, obs)
                sw = np.asarray([int(r["sweep"]) + 1 for r in cr])
                ax.plot(sw, running(x), lw=0.8, alpha=0.65, label=f"c{c}")
                if pooled_num is None:
                    pooled_num = np.zeros_like(x, dtype=float)
                    pooled_den = np.zeros_like(x, dtype=float)
                pooled_num += np.cumsum(x)
                pooled_den += np.arange(1, len(x) + 1)
            ax.plot(np.arange(1, len(pooled_num) + 1), pooled_num / pooled_den, color="black", lw=2.0, label="pooled running")
            ref = refs.get(obs)
            if ref:
                ax.axhline(ref["mean"], color="black", ls="--", lw=1.0, label="direct ref")
                if math.isfinite(ref.get("se", float("nan"))):
                    ax.axhspan(ref["mean"] - ref["se"], ref["mean"] + ref["se"], color="black", alpha=0.10)
            for b in burns:
                ax.axvline(b, color="tab:red", alpha=0.35, lw=0.9)
            ax.set_title(f"{obs} running means")
            ax.set_xlabel("completed sweep (1-based)")
            ax.set_ylabel(obs)
            ax.grid(alpha=0.2)
            ax.legend(ncol=5, fontsize=7)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    with PdfPages(OUT_DIR / "burnin_primary_observable_histories.pdf") as pdf:
        for obs in PRIMARY:
            fig, ax = plt.subplots(figsize=(8.5, 4.8))
            for c in chain_ids(rows):
                cr = select(rows, chain=c)
                ax.plot([int(r["sweep"]) + 1 for r in cr], series(cr, obs), lw=0.55, alpha=0.45, label=f"c{c}")
            ref = refs.get(obs)
            if ref:
                ax.axhline(ref["mean"], color="black", ls="--", lw=1.0)
                if math.isfinite(ref.get("se", float("nan"))):
                    ax.axhspan(ref["mean"] - ref["se"], ref["mean"] + ref["se"], color="black", alpha=0.10)
            for b in burns:
                ax.axvline(b, color="tab:red", alpha=0.35, lw=0.9)
            ax.set_title(f"{obs} histories")
            ax.set_xlabel("completed sweep (1-based)")
            ax.set_ylabel(obs)
            ax.grid(alpha=0.2)
            ax.legend(ncol=8, fontsize=7)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    with PdfPages(OUT_DIR / "burnin_sector_histories.pdf") as pdf:
        for obs in ["m", "abs_m"]:
            fig, ax = plt.subplots(figsize=(8.5, 4.8))
            for c in chain_ids(rows):
                cr = select(rows, chain=c)
                ax.plot([int(r["sweep"]) + 1 for r in cr], series(cr, obs), lw=0.75, alpha=0.75, label=f"c{c}")
            for b in burns:
                ax.axvline(b, color="tab:red", alpha=0.35, lw=0.9)
            ax.axhline(0.0, color="black", lw=0.8)
            ax.set_title(f"{obs} sector history")
            ax.set_xlabel("completed sweep (1-based)")
            ax.set_ylabel(obs)
            ax.grid(alpha=0.2)
            ax.legend(ncol=8, fontsize=7)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def md_table(rows: list[dict[str, Any]], cols: list[str], max_rows: int | None = None) -> str:
    use = rows if max_rows is None else rows[:max_rows]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for r in use:
        vals = []
        for c in cols:
            v = r.get(c, "")
            if isinstance(v, float):
                vals.append(f"{v:.6g}" if math.isfinite(v) else "nan")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_reports(rows: list[dict[str, str]], refs: dict[str, dict[str, float]], startup: list[dict[str, Any]], burnsum: list[dict[str, Any]], burnobs: list[dict[str, Any]]) -> None:
    run_config = json.loads((RUN_DIR / "run_config.json").read_text())
    scheduler = json.loads((RUN_DIR / "scheduler_preflight.json").read_text())
    summary = json.loads((RUN_DIR / "summary.json").read_text())
    wall = float(summary["result"]["wall_time_sec"])
    sec_per_chain_sweep = wall / (run_config["validation_chains"] * run_config["sweeps"])

    semantics = [
        "# Burn-in and Startup Measurement Semantics",
        "",
        f"- `burn_in_discarded_sweeps`: `{summary['result']['initialization_policy']['burn_in_discarded_sweeps']}`",
        f"- `measurement_mode`: `{run_config['measurement_mode']}`",
        f"- observable cadence: `{scheduler['observable_measurement_semantics']['cadence']}`",
        f"- recorded state: `{scheduler['observable_measurement_semantics']['state']}`",
        f"- rejected updates: `{scheduler['observable_measurement_semantics']['rejected_updates']}`",
        f"- expected/actual observable rows: `{run_config['expected']['observable_rows']}` / `{len(rows)}`",
        "",
        "The driver constructs an initial fine/detail state for each thermalized native L16 coarse start, but does not append that pre-sweep state to `observable_timeseries.csv`. In `end_of_sweep` mode, rows are appended after all coarse patch attempts and the per-sweep latent pCN attempt have been accepted or rejected.",
        "",
        "CSV sweep indices are zero-based. Row `sweep=0` is the first completed sweep, so the requested window `sweeps 1-50` is analyzed as CSV sweep indices `0..49`.",
        "",
        "Rejected coarse or latent proposals leave the Markov state unchanged; if the end-of-sweep state has only rejected changes relative to the previous measurement, the observable row repeats the previous state values. The observable table is not accepted-only.",
    ]
    (OUT_DIR / "BURNIN_SEMANTICS.md").write_text("\n".join(semantics) + "\n", encoding="utf-8")

    pooled_primary = [
        r for r in startup
        if r.get("scope") == "chain_bins" and r.get("observable") in PRIMARY and r["window"] in ["1-50", "1-100", "1-200", "1-500", "100-500", "200-500", "500-1000", "1000-2000", "last_50pct", "last_25pct"]
    ]
    lines = [
        "# Startup Window Summary",
        "",
        "Errors in the main table are across-chain standard errors of each window mean. Binder and susceptibility are marked noisy because sector persistence and nonlinear ratios make short-window interpretation fragile.",
        "",
        md_table(pooled_primary, ["window", "observable", "category", "n", "mean", "standard_error", "z_vs_direct_reference"]),
        "",
        "Sector occupancy remains globally balanced by construction, but individual chains mostly stay in their initialized sign sector. Chain 7 starts near zero and then settles negative. The first 50-100 sweeps show a modest startup relaxation in action density and nearest-neighbor observables, but the 1-500 and later windows are not separated by a large monotone bias at the across-chain error scale.",
    ]
    (OUT_DIR / "startup_window_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    lines = [
        "# Burn-in Sensitivity Report",
        "",
        "Discarding 50-500 completed sweeps leaves the primary observable means stable within the present split-chain errors. The first 50-100 sweeps have visible short-window relaxation, especially in action density, but discarding 100 or 200 sweeps does not move the production-scale primary means by more than the current split-chain uncertainty.",
        "",
        md_table(burnsum, ["burn_in_sweeps", "remaining_states_per_chain", "full_chain_split_max_abs_primary_z", "half_chain_split_max_abs_primary_z", "quarter_chain_split_max_abs_primary_z", "tau_action_density_mean", "tau_phi2_mean", "tau_NN_mean", "tau_Binder_U4_mean", "fraction_positive", "sign_flips_total"]),
        "",
        "Primary means by burn-in:",
        "",
        md_table(burnobs, ["burn_in_sweeps", "observable", "mean", "standard_error", "z_vs_direct_reference", "noisy_caveat"]),
    ]
    (OUT_DIR / "burnin_sensitivity_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    taus = {r["burn_in_sweeps"]: r for r in burnsum}
    tau0 = taus[0]
    strategies = [
        ("existing_8x2000_burn0", 8, 2000, 0),
        ("existing_8x2000_burn100", 8, 2000, 100),
        ("existing_8x2000_burn200", 8, 2000, 200),
        ("medium_32x500_burn100", 32, 500, 100),
        ("medium_32x500_burn150", 32, 500, 150),
        ("short_64x300_burn100", 64, 300, 100),
        ("conservative_32x1000_burn200", 32, 1000, 200),
    ]
    strat_rows = []
    for name, chains, sweeps, burn in strategies:
        retained_per_chain = max(0, sweeps - burn)
        retained = chains * retained_per_chain
        row = {
            "strategy": name,
            "chains": chains,
            "sweeps_per_chain": sweeps,
            "burn_in_analysis_sweeps": burn,
            "total_chain_sweeps": chains * sweeps,
            "retained_sweeps": retained,
            "rough_ESS_action_density": retained / (2 * tau0["tau_action_density_mean"]),
            "rough_ESS_phi2": retained / (2 * tau0["tau_phi2_mean"]),
            "rough_ESS_NN": retained / (2 * tau0["tau_NN_mean"]),
            "expected_wall_time_hours": chains * sweeps * sec_per_chain_sweep / 3600,
            "sector_balance_consideration": "use sector-balanced starts; many chains reduce dependence on rare sign flips",
        }
        strat_rows.append(row)
    write_csv(OUT_DIR / "many_short_chain_strategy.csv", strat_rows)

    rec = [
        "# Many-Short-Chain Recommendation",
        "",
        "The evidence favors many independent thermalized native-L16 starts for production-style primary observables. The current 8 long chains show no strong startup bias, but sign sectors are sticky within a chain; more sector-balanced starts improve ensemble balance more directly than waiting for many sign flips.",
        "",
        md_table(strat_rows, ["strategy", "total_chain_sweeps", "retained_sweeps", "rough_ESS_action_density", "rough_ESS_phi2", "rough_ESS_NN", "expected_wall_time_hours"]),
        "",
        "Recommended next production-like run: `32 x 500` with analysis burn-in `100` sweeps. It keeps the same total chain-sweeps as the existing 8x2000 validation, retains 12,800 post-burn states, and should improve sector/start averaging. Use `32 x 1000` with burn-in `200` if Binder or susceptibility are central, because those remain slow/noisy.",
        "",
        "Do not overinterpret Binder from the short-chain options; treat it as a stability diagnostic unless the conservative run agrees.",
    ]
    (OUT_DIR / "MANY_SHORT_CHAIN_RECOMMENDATION.md").write_text("\n".join(rec) + "\n", encoding="utf-8")

    launch = [
        "# Next Many-Short-Chain Manual Instructions",
        "",
        "These commands are prepared for manual launch only. They do not discard burn-in during simulation; burn-in is recorded for analysis/reporting.",
        "",
        "## Recommended: 32 x 500, analyze burn-in 100",
        "",
        "```bash",
        "cd /Users/anna/Work/Research/Normalizing-flow/Inverse_RG",
        "OUT=\"perfect_blocking_upsampling/outputs/shape_parametric_sampler_validation/L16_to_L32_smoke/native_L16_pcn1_32x500_burnin100_candidate\"",
        "test ! -e \"$OUT\" || { echo \"Refusing to overwrite $OUT\"; exit 1; }",
        "mkdir -p \"$OUT\"",
        "echo \"Expected rows: 16000 raw, 12800 retained after analysis burn-in 100\"",
        f"echo \"Rough wall time: {32 * 500 * sec_per_chain_sweep / 3600:.2f} hours from observed {sec_per_chain_sweep:.4f} sec/chain-sweep\"",
        "nohup ../.venv/bin/python -B perfect_blocking_upsampling/scripts/run_shape_parametric_sampler_validation.py \\",
        "  --config perfect_blocking_upsampling/outputs/shape_parametric_sampler_validation/L16_to_L32_smoke/L16_to_L32_smoke_config.yaml \\",
        "  --output-dir \"$OUT\" --coarse-L 16 --patch-size 4 --origin-mode random \\",
        "  --pcn-rho 0.5 --pcn-interval-sweeps 1 --validation-chains 32 --smoke-sweeps 500 \\",
        "  --measurement-mode end_of_sweep --coarse-start-mode thermalized_coarse \\",
        "  --sector-balanced-init --progress-every-sweeps 10 --seed 20260835 \\",
        "  > \"$OUT/run.log\" 2>&1 &",
        "```",
        "",
        "After completion:",
        "",
        "```bash",
        "../.venv/bin/python -B perfect_blocking_upsampling/scripts/analyze_L16_to_L32_validation.py --run-dir \"$OUT\"",
        "echo \"analysis_burn_in_sweeps=100\" > \"$OUT/analysis_burnin_choice.txt\"",
        "```",
        "",
        "## Alternative: 64 x 300, analyze burn-in 100",
        "",
        "```bash",
        "cd /Users/anna/Work/Research/Normalizing-flow/Inverse_RG",
        "OUT=\"perfect_blocking_upsampling/outputs/shape_parametric_sampler_validation/L16_to_L32_smoke/native_L16_pcn1_64x300_burnin100_candidate\"",
        "test ! -e \"$OUT\" || { echo \"Refusing to overwrite $OUT\"; exit 1; }",
        "mkdir -p \"$OUT\"",
        "echo \"Expected rows: 19200 raw, 12800 retained after analysis burn-in 100\"",
        f"echo \"Rough wall time: {64 * 300 * sec_per_chain_sweep / 3600:.2f} hours from observed {sec_per_chain_sweep:.4f} sec/chain-sweep\"",
        "nohup ../.venv/bin/python -B perfect_blocking_upsampling/scripts/run_shape_parametric_sampler_validation.py \\",
        "  --config perfect_blocking_upsampling/outputs/shape_parametric_sampler_validation/L16_to_L32_smoke/L16_to_L32_smoke_config.yaml \\",
        "  --output-dir \"$OUT\" --coarse-L 16 --patch-size 4 --origin-mode random \\",
        "  --pcn-rho 0.5 --pcn-interval-sweeps 1 --validation-chains 64 --smoke-sweeps 300 \\",
        "  --measurement-mode end_of_sweep --coarse-start-mode thermalized_coarse \\",
        "  --sector-balanced-init --progress-every-sweeps 10 --seed 20260836 \\",
        "  > \"$OUT/run.log\" 2>&1 &",
        "```",
        "",
        "After completion:",
        "",
        "```bash",
        "../.venv/bin/python -B perfect_blocking_upsampling/scripts/analyze_L16_to_L32_validation.py --run-dir \"$OUT\"",
        "echo \"analysis_burn_in_sweeps=100\" > \"$OUT/analysis_burnin_choice.txt\"",
        "```",
    ]
    (OUT_DIR / "run_next_many_short_chain_instructions.md").write_text("\n".join(launch) + "\n", encoding="utf-8")

    final = [
        "# L16 to L32 Burn-in and Many-Chain Report",
        "",
        "## Answers",
        "",
        "1. Evidence for startup bias: a modest first-50-to-100-sweep relaxation is visible, especially in action density and NN, but no large production-scale monotone startup bias remains once windows of a few hundred sweeps are used. Signed sector diagnostics show expected dependence on sector starts.",
        "2. Zero burn-in for exploratory validation: acceptable. The existing no-burn validation is a fair exploratory diagnostic because starts are thermalized native L16 coarse configurations and observables begin after the first full fine/detail sweep.",
        "3. Production-style burn-in: use an analysis burn-in of 100 sweeps for many 300-500 sweep chains; use 200 sweeps for conservative 1000-sweep chains or Binder/susceptibility-focused checks.",
        "4. Effect of discarding 100 or 200 sweeps: it does not materially change primary means at the current precision. Split-chain max |z| remains around the same scale, while retained statistics decrease modestly.",
        "5. Prefer many independent starts: yes for primary observables and sector balance. Many starts reduce sensitivity to sticky sign sectors better than a small number of long chains at fixed total chain-sweeps.",
        "6. Recommended next manual production-like run: 32 chains x 500 sweeps, P=4, pCN every sweep, native thermalized L16 starts, sector-balanced initialization, end-of-sweep measurement, burn-in 100 applied only in analysis.",
        "7. Exact command Anna should run next: use the first command block in `run_next_many_short_chain_instructions.md`.",
        "",
        "## Key Numbers",
        "",
        md_table(burnsum, ["burn_in_sweeps", "remaining_states_per_chain", "full_chain_split_max_abs_primary_z", "tau_action_density_mean", "tau_phi2_mean", "tau_NN_mean", "fraction_positive", "sign_flips_total"]),
        "",
        "Binder and susceptibility remain slow/noisy. They should not drive the short-chain recommendation without a conservative follow-up.",
    ]
    (OUT_DIR / "L16_TO_L32_BURNIN_AND_MANY_CHAIN_REPORT.md").write_text("\n".join(final) + "\n", encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = read_csv(RUN_DIR / "observable_timeseries.csv")
    refs = refs_from_existing()
    startup = startup_windows(rows, refs)
    burnsum, burnobs = burnin_sensitivity(rows, refs)
    write_csv(OUT_DIR / "startup_window_summary.csv", startup)
    write_csv(OUT_DIR / "burnin_sensitivity_summary.csv", burnsum)
    write_csv(OUT_DIR / "burnin_sensitivity_observables.csv", burnobs)
    plot_outputs(rows, refs)
    write_reports(rows, refs, startup, burnsum, burnobs)
    print(json.dumps({"status": "done", "out_dir": str(OUT_DIR)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
