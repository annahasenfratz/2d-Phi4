#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASELINE = PROJECT_ROOT / "perfect_blocking_upsampling/outputs/shape_parametric_sampler_validation/L16_to_L32_smoke/native_L16_pcn1_8x2000"
OUT_ROOT = PROJECT_ROOT / "perfect_blocking_upsampling/outputs/shape_parametric_sampler_validation/L16_to_L32_many_short"
RUN_DIR = OUT_ROOT / "native_L16_pcn1_P4_32x500"
PRIMARY = ["phi2", "phi4", "NN", "action_density", "susceptibility", "Binder_U4", "xi_over_L"]
AUTOCORR = ["action_density", "phi2", "NN", "Binder_U4"]
BURNINS = [0, 50, 100, 200]
WINDOWS = [
    ("sweeps_1_50", 0, 50),
    ("sweeps_51_100", 50, 100),
    ("sweeps_101_200", 100, 200),
    ("sweeps_201_500", 200, 500),
    ("sweeps_1_500", 0, 500),
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def fval(row: dict[str, str], key: str, default: float = float("nan")) -> float:
    try:
        return float(row[key])
    except Exception:
        return default


def chain_ids(rows: list[dict[str, str]]) -> list[int]:
    return sorted({int(r["chain_id"]) for r in rows})


def select_rows(
    rows: list[dict[str, str]], chain: int | None = None, start: int | None = None, stop: int | None = None
) -> list[dict[str, str]]:
    out = []
    for row in rows:
        sweep = int(row["sweep"])
        if chain is not None and int(row["chain_id"]) != chain:
            continue
        if start is not None and sweep < start:
            continue
        if stop is not None and sweep >= stop:
            continue
        out.append(row)
    return out


def mean_se(values: list[float] | np.ndarray) -> tuple[float, float, float, int]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan"), float("nan"), 0
    if x.size == 1:
        return float(x[0]), float("nan"), 0.0, 1
    std = float(np.std(x, ddof=1))
    return float(np.mean(x)), std / math.sqrt(x.size), std, int(x.size)


def reference_stats() -> dict[str, dict[str, float]]:
    refs = {}
    for row in read_csv(BASELINE / "split_chain_binning_summary.csv"):
        if row["binning"] == "full_chain":
            refs[row["observable"]] = {
                "mean": fval(row, "direct_reference_mean"),
                "se": fval(row, "direct_reference_se"),
            }
    return refs


def z_score(mean: float, se: float, ref: dict[str, float]) -> float:
    denom2 = 0.0
    if math.isfinite(se):
        denom2 += se * se
    if math.isfinite(ref.get("se", float("nan"))):
        denom2 += ref["se"] * ref["se"]
    return (mean - ref["mean"]) / math.sqrt(denom2) if denom2 > 0 else float("nan")


def bin_observable(rows: list[dict[str, str]], obs: str, lattice_l: int = 32) -> float:
    if not rows:
        return float("nan")
    if obs in {"phi2", "phi4", "NN", "action_density", "m", "abs_m"}:
        return float(np.mean([fval(r, obs) for r in rows]))
    m2 = np.asarray([fval(r, "m2") for r in rows], dtype=np.float64)
    m4 = np.asarray([fval(r, "m4") for r in rows], dtype=np.float64)
    if obs == "susceptibility":
        return float(lattice_l * lattice_l * np.mean(m2))
    if obs == "Binder_U4":
        return float(1.0 - np.mean(m4) / max(3.0 * np.mean(m2) ** 2, 1.0e-300))
    if obs == "xi_over_L":
        phi2 = float(np.mean([fval(r, "phi2") for r in rows]))
        chi = lattice_l * lattice_l * float(np.mean(m2))
        return float(math.sqrt(max(chi, 0.0) / max(phi2, 1.0e-300)) / lattice_l)
    raise KeyError(obs)


def row_series(rows: list[dict[str, str]], obs: str, lattice_l: int = 32) -> np.ndarray:
    if obs in {"phi2", "phi4", "NN", "action_density", "m", "abs_m"}:
        return np.asarray([fval(r, obs) for r in rows], dtype=np.float64)
    m2 = np.asarray([fval(r, "m2") for r in rows], dtype=np.float64)
    if obs == "susceptibility":
        return lattice_l * lattice_l * m2
    if obs == "Binder_U4":
        m4 = np.asarray([fval(r, "m4") for r in rows], dtype=np.float64)
        return 1.0 - m4 / np.maximum(3.0 * m2 * m2, 1.0e-300)
    if obs == "xi_over_L":
        phi2 = np.asarray([fval(r, "phi2") for r in rows], dtype=np.float64)
        return np.sqrt(np.maximum(lattice_l * lattice_l * m2, 0.0) / np.maximum(phi2, 1.0e-300)) / lattice_l
    raise KeyError(obs)


def tau_initial_positive(series: np.ndarray) -> float:
    x = np.asarray(series, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return float("nan")
    y = x - np.mean(x)
    var = float(np.dot(y, y) / y.size)
    if var <= 0.0:
        return 0.5
    tau = 0.5
    max_lag = min(150, max(1, x.size // 3))
    for lag in range(1, max_lag + 1):
        ac = float(np.dot(y[:-lag], y[lag:]) / ((y.size - lag) * var))
        if not math.isfinite(ac) or ac <= 0.0:
            break
        tau += ac
    return tau


def acceptance_windows(run_dir: Path) -> list[dict[str, Any]]:
    rows_out: list[dict[str, Any]] = []
    for move_type, filename in [("coarse", "coarse_deltas.csv"), ("latent", "latent_deltas.csv")]:
        rows = read_csv(run_dir / filename)
        chains = [None] + chain_ids(rows)
        for chain in chains:
            scope = "pooled" if chain is None else f"chain_{chain}"
            for window, start, stop in WINDOWS:
                wr = select_rows(rows, chain=chain, start=start, stop=stop)
                acc = [fval(r, "accepted") for r in wr]
                dlogw = [fval(r, "delta_logw") for r in wr]
                rows_out.append(
                    {
                        "move_type": move_type,
                        "scope": scope,
                        "window": window,
                        "csv_sweep_start_inclusive": start,
                        "csv_sweep_stop_exclusive": stop,
                        "attempts": len(wr),
                        "acceptance": float(np.mean(acc)) if acc else float("nan"),
                        "delta_logw_std": float(np.std(dlogw, ddof=1)) if len(dlogw) > 1 else float("nan"),
                        "delta_logw_mean": float(np.mean(dlogw)) if dlogw else float("nan"),
                        "delta_logw_mean_abs": float(np.mean(np.abs(dlogw))) if dlogw else float("nan"),
                    }
                )
    return rows_out


def split_summaries(rows: list[dict[str, str]], burn: int, refs: dict[str, dict[str, float]]) -> tuple[dict[str, float], list[dict[str, Any]]]:
    n_sweeps = max(int(r["sweep"]) for r in rows) + 1
    remaining = n_sweeps - burn
    details: list[dict[str, Any]] = []
    max_z: dict[str, float] = {}
    for label, parts in [("full", 1), ("half", 2), ("quarter", 4)]:
        width = remaining // parts
        zvals = []
        for obs in PRIMARY:
            vals = []
            for chain in chain_ids(rows):
                for part in range(parts):
                    start = burn + part * width
                    stop = burn + (part + 1) * width if part < parts - 1 else n_sweeps
                    vals.append(bin_observable(select_rows(rows, chain=chain, start=start, stop=stop), obs))
            mean, se, std, n = mean_se(vals)
            z = z_score(mean, se, refs[obs])
            details.append(
                {
                    "burn_in": burn,
                    "binning": label,
                    "observable": obs,
                    "n_bins": n,
                    "bin_width_sweeps": width,
                    "mean": mean,
                    "standard_error": se,
                    "std": std,
                    "z_vs_direct_reference": z,
                }
            )
            if math.isfinite(z):
                zvals.append(abs(z))
        max_z[label] = max(zvals) if zvals else float("nan")
    return max_z, details


def sign_stats(rows: list[dict[str, str]], burn: int) -> tuple[float, float, int]:
    kept = select_rows(rows, start=burn)
    pos = neg = total = flips = 0
    for chain in chain_ids(kept):
        vals = np.asarray([fval(r, "m") for r in select_rows(kept, chain=chain)], dtype=np.float64)
        pos += int(np.sum(vals > 0))
        neg += int(np.sum(vals < 0))
        total += int(vals.size)
        signs = np.sign(vals)
        nz = signs[signs != 0]
        flips += int(np.sum(nz[1:] != nz[:-1])) if nz.size > 1 else 0
    return pos / max(total, 1), neg / max(total, 1), flips


def burnin_sensitivity(run_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = read_csv(run_dir / "observable_timeseries.csv")
    refs = reference_stats()
    acc_rows = acceptance_windows(run_dir)
    out = []
    obs_detail: list[dict[str, Any]] = []
    for burn in BURNINS:
        max_z, details = split_summaries(rows, burn, refs)
        obs_detail.extend(details)
        kept = select_rows(rows, start=burn)
        taus = {}
        ess = {}
        for obs in AUTOCORR:
            tau_vals = []
            for chain in chain_ids(rows):
                tau_vals.append(tau_initial_positive(row_series(select_rows(rows, chain=chain, start=burn), obs)))
            taus[obs] = float(np.nanmean(tau_vals))
            ess[obs] = len(kept) / max(2.0 * taus[obs], 1.0e-300) if math.isfinite(taus[obs]) else float("nan")
        fp, fn, flips = sign_stats(rows, burn)
        coarse_after = [r for r in acc_rows if r["move_type"] == "coarse" and r["scope"] == "pooled" and r["csv_sweep_start_inclusive"] >= burn]
        latent_after = [r for r in acc_rows if r["move_type"] == "latent" and r["scope"] == "pooled" and r["csv_sweep_start_inclusive"] >= burn]
        coarse_rows = [r for r in read_csv(run_dir / "coarse_deltas.csv") if int(r["sweep"]) >= burn]
        latent_rows = [r for r in read_csv(run_dir / "latent_deltas.csv") if int(r["sweep"]) >= burn]
        out.append(
            {
                "burn_in": burn,
                "retained_states": len(kept),
                "split_max_z_full": max_z["full"],
                "split_max_z_half": max_z["half"],
                "split_max_z_quarter": max_z["quarter"],
                "tau_action_density": taus["action_density"],
                "tau_phi2": taus["phi2"],
                "tau_NN": taus["NN"],
                "tau_Binder_U4": taus["Binder_U4"],
                "ESS_action_density_rough": ess["action_density"],
                "ESS_phi2_rough": ess["phi2"],
                "ESS_NN_rough": ess["NN"],
                "sector_fraction_positive": fp,
                "sector_fraction_negative": fn,
                "sign_flips": flips,
                "coarse_acceptance_after_cut": float(np.mean([fval(r, "accepted") for r in coarse_rows])) if coarse_rows else float("nan"),
                "coarse_delta_logw_std_after_cut": float(np.std([fval(r, "delta_logw") for r in coarse_rows], ddof=1)) if len(coarse_rows) > 1 else float("nan"),
                "latent_acceptance_after_cut": float(np.mean([fval(r, "accepted") for r in latent_rows])) if latent_rows else float("nan"),
                "latent_delta_logw_std_after_cut": float(np.std([fval(r, "delta_logw") for r in latent_rows], ddof=1)) if len(latent_rows) > 1 else float("nan"),
            }
        )
    return out, obs_detail


def max_existing_z(binning: str) -> float:
    vals = [
        abs(fval(r, "z_vs_direct_reference"))
        for r in read_csv(BASELINE / "split_chain_binning_summary.csv")
        if r["binning"] == binning and r["category"] == "primary"
    ]
    return max(vals) if vals else float("nan")


def baseline_tau(obs: str) -> float:
    vals = [fval(r, "tau_int_initial_positive") for r in read_csv(BASELINE / "autocorrelation_summary.csv") if r["observable"] == obs]
    return float(np.nanmean(vals))


def write_report(run_dir: Path, burn_rows: list[dict[str, Any]], acc_rows: list[dict[str, Any]]) -> None:
    summary = json.loads((run_dir / "summary.json").read_text())
    base_summary = json.loads((BASELINE / "summary.json").read_text())
    result = summary["result"]
    base_result = base_summary["result"]
    run_wall = float(result["wall_time_sec"])
    base_wall = float(base_result["wall_time_sec"])
    first_windows = [
        r
        for r in acc_rows
        if r["scope"] == "pooled" and r["window"] in {"sweeps_1_50", "sweeps_51_100", "sweeps_101_200", "sweeps_201_500", "sweeps_1_500"}
    ]
    best = min(burn_rows, key=lambda r: r["split_max_z_full"])
    ess_run = burn_rows[0]["ESS_action_density_rough"] / (run_wall / 3600.0)
    ess_base = (16000.0 / (2.0 * baseline_tau("action_density"))) / (base_wall / 3600.0)

    def table(rows: list[dict[str, Any]], cols: list[str]) -> list[str]:
        lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
        for row in rows:
            vals = []
            for col in cols:
                val = row.get(col, "")
                if isinstance(val, float):
                    vals.append(f"{val:.6g}" if math.isfinite(val) else "nan")
                else:
                    vals.append(str(val))
            lines.append("| " + " | ".join(vals) + " |")
        return lines

    lines = [
        "# L16 to L32 8x2000 vs 32x500 Analysis Report",
        "",
        "## Startup Acceptance",
        "",
        "The startup windows directly test whether no-burn-in short chains begin with poor acceptance and then recover.",
        "",
    ]
    lines.extend(table(first_windows, ["move_type", "window", "attempts", "acceptance", "delta_logw_std", "delta_logw_mean_abs"]))
    lines.extend(
        [
            "",
            "## Burn-in Sensitivity",
            "",
        ]
    )
    lines.extend(
        table(
            burn_rows,
            [
                "burn_in",
                "retained_states",
                "split_max_z_full",
                "split_max_z_half",
                "split_max_z_quarter",
                "tau_action_density",
                "tau_phi2",
                "tau_NN",
                "tau_Binder_U4",
                "coarse_acceptance_after_cut",
                "latent_acceptance_after_cut",
                "sector_fraction_positive",
                "sign_flips",
            ],
        )
    )
    lines.extend(
        [
            "",
            "## Answers",
            "",
            f"1. Poor early acceptance: compare `sweeps_1_50` to later windows in `startup_acceptance_window_summary.csv`; full-run coarse/latent A/R are `{result['coarse_acceptance']:.6g}` / `{result['latent_acceptance']:.6g}`.",
            "2. Recovery after 50/100/200 sweeps: see startup windows and after-cut A/R columns above.",
            f"3. Best split-chain full max |z| among tested burn-ins: burn-in `{best['burn_in']}` with max |z| `{best['split_max_z_full']:.6g}`.",
            f"4. Baseline 8x2000 full/half/quarter max |z|: `{max_existing_z('full_chain'):.6g}` / `{max_existing_z('half_chain'):.6g}` / `{max_existing_z('quarter_chain'):.6g}`. Compare to burn-in 0 row above.",
            f"5. Rough action ESS per wall hour: baseline `{ess_base:.6g}`, 32x500 `{ess_run:.6g}`. Treat as rough because short-chain tau estimates are noisy.",
            "6. Recommendation should be based on the startup A/R window, split-chain max |z|, and ESS/wall-time comparison above; Binder remains noisy.",
        ]
    )
    (OUT_ROOT / "L16_TO_L32_8x2000_VS_32x500_ANALYSIS_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_compare_template(run_dir: Path, burn_rows: list[dict[str, Any]]) -> None:
    summary = json.loads((run_dir / "summary.json").read_text())
    result = summary["result"]
    base_summary = json.loads((BASELINE / "summary.json").read_text())
    base_result = base_summary["result"]
    fields = [
        "run label",
        "chains",
        "sweeps",
        "total states",
        "coarse attempts",
        "latent attempts",
        "wall time",
        "coarse acceptance",
        "coarse Delta logw std",
        "latent acceptance",
        "latent Delta logw std",
        "burn-in cut",
        "retained states",
        "split-chain max primary |z|",
        "tau_action",
        "tau_phi2",
        "tau_NN",
        "tau_Binder",
        "sector fraction positive",
        "sign flips",
        "comments",
    ]
    rows = [
        {
            "run label": "baseline_L16_to_L32_8x2000",
            "chains": 8,
            "sweeps": 2000,
            "total states": 16000,
            "coarse attempts": 512000,
            "latent attempts": 16000,
            "wall time": base_result["wall_time_sec"],
            "coarse acceptance": base_result["coarse_acceptance"],
            "coarse Delta logw std": base_result["coarse_std_delta_logw"],
            "latent acceptance": base_result["latent_acceptance"],
            "latent Delta logw std": base_result["latent_std_delta_logw"],
            "burn-in cut": 0,
            "retained states": 16000,
            "split-chain max primary |z|": max_existing_z("full_chain"),
            "tau_action": baseline_tau("action_density"),
            "tau_phi2": baseline_tau("phi2"),
            "tau_NN": baseline_tau("NN"),
            "tau_Binder": baseline_tau("Binder_U4"),
            "sector fraction positive": json.loads((BASELINE / "sector_occupancy.json").read_text())["fraction_positive"],
            "sign flips": json.loads((BASELINE / "sector_occupancy.json").read_text())["total_sign_flips"],
            "comments": "existing accepted 8x2000 baseline",
        }
    ]
    for row in burn_rows:
        rows.append(
            {
                "run label": f"L16_to_L32_32x500_burn{row['burn_in']}",
                "chains": 32,
                "sweeps": 500,
                "total states": 16000,
                "coarse attempts": result["coarse_attempts"],
                "latent attempts": result["latent_attempts"],
                "wall time": result["wall_time_sec"],
                "coarse acceptance": result["coarse_acceptance"],
                "coarse Delta logw std": result["coarse_std_delta_logw"],
                "latent acceptance": result["latent_acceptance"],
                "latent Delta logw std": result["latent_std_delta_logw"],
                "burn-in cut": row["burn_in"],
                "retained states": row["retained_states"],
                "split-chain max primary |z|": row["split_max_z_full"],
                "tau_action": row["tau_action_density"],
                "tau_phi2": row["tau_phi2"],
                "tau_NN": row["tau_NN"],
                "tau_Binder": row["tau_Binder_U4"],
                "sector fraction positive": row["sector_fraction_positive"],
                "sign flips": row["sign_flips"],
                "comments": "filled after 32x500 analysis; burn-in discarded only in analysis",
            }
        )
    write_csv(OUT_ROOT / "compare_8x2000_vs_32x500_filled.csv", rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=RUN_DIR)
    args = ap.parse_args()
    run_dir = args.run_dir
    if not (run_dir / "summary.json").exists():
        raise FileNotFoundError(f"run is not complete; missing {run_dir / 'summary.json'}")
    acc_rows = acceptance_windows(run_dir)
    write_csv(OUT_ROOT / "startup_acceptance_window_summary.csv", acc_rows)
    burn_rows, obs_rows = burnin_sensitivity(run_dir)
    write_csv(OUT_ROOT / "burnin_sensitivity_32x500.csv", burn_rows)
    write_csv(OUT_ROOT / "burnin_sensitivity_32x500_observables.csv", obs_rows)
    update_compare_template(run_dir, burn_rows)
    write_report(run_dir, burn_rows, acc_rows)
    print(json.dumps({"status": "completed", "out_root": str(OUT_ROOT)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
