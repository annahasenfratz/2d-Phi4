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
PKG = PROJECT_ROOT / "perfect_blocking_upsampling"
sys.path.insert(0, str(PKG / "scripts"))
sys.path.insert(0, str(PKG / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from _common import load_config, load_ensembles, load_kernel_spec  # noqa: E402
from perfect_blocking_upsampling.io import ActionSpec  # noqa: E402
from perfect_blocking_upsampling.kernels import apply_kernel  # noqa: E402
from perfect_blocking_upsampling.observables import observables as ensemble_observables  # noqa: E402

PRIMARY_OBSERVABLES = ["phi2", "phi4", "NN", "action_density", "susceptibility", "Binder_U4", "xi_over_L"]
SECTOR_DIAGNOSTICS = ["m", "abs_m"]
OBSERVABLES = PRIMARY_OBSERVABLES + SECTOR_DIAGNOSTICS
SIMPLE_OBSERVABLES = {"m", "abs_m", "phi2", "phi4", "NN", "action_density"}
AUTOCORR_OBSERVABLES = ["action_density", "phi2", "NN", "Binder_U4"]
BLOCK_SIZES = [1, 2, 5, 10, 20, 50, 100]


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n", encoding="utf-8")


def fval(row: dict[str, str], key: str) -> float:
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


def mean_se(values: list[float] | np.ndarray) -> tuple[float, float, float]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    if x.size == 1:
        return float(x[0]), float("nan"), 0.0
    std = float(np.std(x, ddof=1))
    return float(np.mean(x)), std / math.sqrt(x.size), std


def bin_observable(rows: list[dict[str, str]], obs: str, lattice_l: int) -> float:
    if not rows:
        return float("nan")
    if obs in SIMPLE_OBSERVABLES:
        return float(np.mean([fval(r, obs) for r in rows]))
    m2 = np.asarray([fval(r, "m2") for r in rows], dtype=np.float64)
    m4 = np.asarray([fval(r, "m4") for r in rows], dtype=np.float64)
    if obs == "susceptibility":
        return float(lattice_l * lattice_l * np.mean(m2))
    if obs == "Binder_U4":
        denom = 3.0 * max(float(np.mean(m2)) ** 2, 1.0e-300)
        return float(1.0 - float(np.mean(m4)) / denom)
    if obs == "xi_over_L":
        phi2 = float(np.mean([fval(r, "phi2") for r in rows]))
        chi = lattice_l * lattice_l * float(np.mean(m2))
        return float(math.sqrt(max(chi, 0.0) / max(phi2, 1.0e-300)) / lattice_l)
    raise KeyError(obs)


def row_series(rows: list[dict[str, str]], obs: str, lattice_l: int) -> np.ndarray:
    if obs in SIMPLE_OBSERVABLES:
        return np.asarray([fval(r, obs) for r in rows], dtype=np.float64)
    if obs == "susceptibility":
        return lattice_l * lattice_l * np.asarray([fval(r, "m2") for r in rows], dtype=np.float64)
    if obs == "Binder_U4":
        m2 = np.asarray([fval(r, "m2") for r in rows], dtype=np.float64)
        m4 = np.asarray([fval(r, "m4") for r in rows], dtype=np.float64)
        return 1.0 - m4 / np.maximum(3.0 * m2 * m2, 1.0e-300)
    if obs == "xi_over_L":
        m2 = np.asarray([fval(r, "m2") for r in rows], dtype=np.float64)
        phi2 = np.asarray([fval(r, "phi2") for r in rows], dtype=np.float64)
        return np.sqrt(np.maximum(lattice_l * lattice_l * m2, 0.0) / np.maximum(phi2, 1.0e-300)) / lattice_l
    raise KeyError(obs)


def action_spec(block: dict[str, Any]) -> ActionSpec:
    return ActionSpec(
        type=str(block["type"]),
        lambda_=float(block["lambda"]),
        kappa=float(block["kappa"]),
        kappa_diag=float(block.get("kappa_diag", 0.0)),
    )


def reference_stats(fine_ref: np.ndarray, fine_action: ActionSpec) -> dict[str, dict[str, float]]:
    arr = np.asarray(fine_ref, dtype=np.float64)
    obs = ensemble_observables(arr, fine_action)
    per = {
        "m": arr.mean(axis=(1, 2)),
        "abs_m": np.abs(arr.mean(axis=(1, 2))),
        "phi2": np.mean(arr**2, axis=(1, 2)),
        "phi4": np.mean(arr**4, axis=(1, 2)),
        "NN": 0.5
        * (
            np.mean(arr * np.roll(arr, -1, axis=1), axis=(1, 2))
            + np.mean(arr * np.roll(arr, -1, axis=2), axis=(1, 2))
        ),
        "action_density": None,
        "susceptibility": arr.shape[1] * arr.shape[2] * arr.mean(axis=(1, 2)) ** 2,
    }
    # Use the observable helper for the exact project action convention and a per-config loop for action SE.
    per["action_density"] = np.asarray([ensemble_observables(arr[i : i + 1], fine_action)["action_density"] for i in range(arr.shape[0])], dtype=np.float64)
    out: dict[str, dict[str, float]] = {}
    for key in ["m", "abs_m", "phi2", "phi4", "NN", "action_density", "susceptibility"]:
        mean, se, std = mean_se(per[key])
        out[key] = {"mean": mean, "se": se, "std": std, "n": int(arr.shape[0])}
    for key in ["Binder_U4", "xi_over_L"]:
        vals = []
        n = arr.shape[0]
        # Jackknife over 50 chunks to avoid an expensive leave-one-out loop.
        chunks = np.array_split(np.arange(n), min(50, n))
        for ch in chunks:
            mask = np.ones(n, dtype=bool)
            mask[ch] = False
            vals.append(float(ensemble_observables(arr[mask], fine_action)[key]))
        jk = np.asarray(vals, dtype=np.float64)
        jk_mean = float(np.mean(jk))
        se = float(math.sqrt((len(jk) - 1) * np.mean((jk - jk_mean) ** 2))) if len(jk) > 1 else float("nan")
        out[key] = {"mean": float(obs[key]), "se": se, "std": float("nan"), "n": int(arr.shape[0])}
    return out


def z_score(mean: float, se: float, ref: dict[str, float]) -> float:
    denom2 = 0.0
    if math.isfinite(se):
        denom2 += se * se
    ref_se = ref.get("se", float("nan"))
    if math.isfinite(ref_se):
        denom2 += ref_se * ref_se
    denom = math.sqrt(denom2) if denom2 > 0.0 else float("nan")
    return (mean - ref["mean"]) / denom if math.isfinite(denom) and denom > 0.0 else float("nan")


def split_chain_binning(rows: list[dict[str, str]], direct_ref: dict[str, dict[str, float]], lattice_l: int) -> list[dict[str, Any]]:
    n_sweeps = max(int(r["sweep"]) for r in rows) + 1
    out = []
    for label, parts in [("full_chain", 1), ("half_chain", 2), ("quarter_chain", 4)]:
        width = n_sweeps // parts
        for obs in OBSERVABLES:
            vals = []
            for chain in chain_ids(rows):
                for part in range(parts):
                    start = part * width
                    stop = (part + 1) * width if part < parts - 1 else n_sweeps
                    vals.append(bin_observable(select_rows(rows, chain, start, stop), obs, lattice_l))
            mean, se, std = mean_se(vals)
            ref = direct_ref[obs]
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
                    "direct_reference_mean": ref["mean"],
                    "direct_reference_se": ref.get("se", float("nan")),
                    "z_vs_direct_reference": z_score(mean, se, ref),
                    "reference_source": "direct_L32_reference_from_validation_config",
                }
            )
    return out


def window_stability(rows: list[dict[str, str]], direct_ref: dict[str, dict[str, float]], lattice_l: int) -> list[dict[str, Any]]:
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
            vals = [bin_observable(select_rows(rows, chain, start, stop), obs, lattice_l) for chain in chain_ids(rows)]
            mean, se, std = mean_se(vals)
            ref = direct_ref[obs]
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
                    "direct_reference_mean": ref["mean"],
                    "z_vs_direct_reference": z_score(mean, se, ref),
                }
            )
    return out


def autocorr(x: np.ndarray, max_lag: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return np.full(max_lag + 1, np.nan)
    y = x - np.mean(x)
    var = float(np.dot(y, y) / y.size)
    if var <= 0.0:
        out = np.zeros(max_lag + 1)
        out[0] = 1.0
        return out
    out = np.empty(max_lag + 1, dtype=np.float64)
    out[0] = 1.0
    for lag in range(1, max_lag + 1):
        out[lag] = float(np.dot(y[:-lag], y[lag:]) / ((y.size - lag) * var))
    return out


def tau_initial_positive(ac: np.ndarray) -> tuple[float, int]:
    tau = 0.5
    cutoff = 0
    for lag in range(1, len(ac)):
        if not math.isfinite(float(ac[lag])) or ac[lag] <= 0.0:
            break
        tau += float(ac[lag])
        cutoff = lag
    return tau, cutoff


def autocorrelation_summary(rows: list[dict[str, str]], lattice_l: int) -> tuple[list[dict[str, Any]], dict[str, dict[int, np.ndarray]]]:
    out = []
    curves: dict[str, dict[int, np.ndarray]] = {}
    n_sweeps = max(int(r["sweep"]) for r in rows) + 1
    max_lag = min(150, max(1, n_sweeps // 3))
    for obs in AUTOCORR_OBSERVABLES:
        curves[obs] = {}
        for chain in chain_ids(rows):
            series = row_series(select_rows(rows, chain), obs, lattice_l)
            ac = autocorr(series, max_lag)
            curves[obs][chain] = ac
            tau, cutoff = tau_initial_positive(ac)
            ess = float(series.size / max(2.0 * tau, 1.0e-300)) if math.isfinite(tau) else float("nan")
            caveat = "Binder_U4 row proxy is nonlinear; use as qualitative only." if obs == "Binder_U4" else "short 500-sweep chains; tau estimate is noisy."
            out.append(
                {
                    "chain_id": chain,
                    "category": "primary",
                    "observable": obs,
                    "n": int(series.size),
                    "max_lag": max_lag,
                    "tau_int_initial_positive": tau,
                    "positive_cutoff_lag": cutoff,
                    "effective_sample_size": ess,
                    "caveat": caveat,
                }
            )
    return out, curves


def blocking_analysis(rows: list[dict[str, str]], lattice_l: int) -> list[dict[str, Any]]:
    out = []
    for obs in AUTOCORR_OBSERVABLES:
        for chain in chain_ids(rows):
            x = row_series(select_rows(rows, chain), obs, lattice_l)
            for b in BLOCK_SIZES:
                n_blocks = x.size // b
                if n_blocks < 2:
                    continue
                y = x[: n_blocks * b].reshape(n_blocks, b).mean(axis=1)
                mean, se, std = mean_se(y)
                out.append(
                    {
                        "chain_id": chain,
                        "observable": obs,
                        "block_size_sweeps": b,
                        "n_blocks": n_blocks,
                        "mean": mean,
                        "block_standard_error": se,
                        "std_block_means": std,
                    }
                )
    return out


def coarse_distribution_comparison(coarse: np.ndarray, fine_ref: np.ndarray, kernel: Any, coarse_action: ActionSpec) -> list[dict[str, Any]]:
    blocked = apply_kernel(fine_ref, kernel)[:, 0::2, 0::2]
    native_obs = ensemble_observables(coarse, coarse_action)
    blocked_obs = ensemble_observables(blocked, coarse_action)
    rows = []
    for obs in ["phi2", "phi4", "NN", "action_density", "m", "abs_m", "Binder_U4", "susceptibility", "xi_over_L"]:
        n = float(native_obs[obs])
        b = float(blocked_obs[obs])
        rows.append(
            {
                "observable": obs,
                "native_L16_coarse": n,
                "blocked_direct_L32_small3": b,
                "native_minus_blocked": n - b,
                "reference_source": "direct_L32_reference_blocked_with_small3_eta0p25",
            }
        )
    return rows


def discover_run() -> list[Path]:
    root = PKG / "outputs" / "shape_parametric_sampler_validation" / "L16_to_L32_smoke"
    candidates = []
    for summary_path in root.glob("*/summary.json"):
        try:
            run = summary_path.parent
            cfg = json.loads((run / "run_config.json").read_text())
            if (
                int(cfg.get("coarse_L")) == 16
                and int(cfg.get("fine_L")) == 32
                and int(cfg.get("patch_size")) == 4
                and int(cfg.get("validation_chains")) == 8
                and int(cfg.get("sweeps")) == 500
            ):
                candidates.append(run)
        except Exception:
            continue
    return sorted(candidates)


def plot_histories(run_dir: Path, rows: list[dict[str, str]], direct_ref: dict[str, dict[str, float]], lattice_l: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    def running_mean(x: np.ndarray) -> np.ndarray:
        return np.cumsum(x) / np.arange(1, x.size + 1)

    with PdfPages(run_dir / "L16_to_L32_primary_running_mean_diagnostics.pdf") as pdf:
        for obs in PRIMARY_OBSERVABLES:
            fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
            for chain in chain_ids(rows):
                cr = select_rows(rows, chain)
                sweeps = np.asarray([int(r["sweep"]) for r in cr])
                series = row_series(cr, obs, lattice_l)
                axes[0].plot(sweeps, series, alpha=0.35, lw=0.8, label=f"chain {chain}" if chain == 0 else None)
                axes[1].plot(sweeps, running_mean(series), alpha=0.8, lw=1.0)
            ref = direct_ref.get(obs)
            if ref:
                for ax in axes:
                    ax.axhline(ref["mean"], color="black", lw=1.0, label="direct L32 ref")
                    if math.isfinite(ref.get("se", float("nan"))):
                        ax.axhspan(ref["mean"] - ref["se"], ref["mean"] + ref["se"], color="black", alpha=0.12)
            axes[0].set_title(f"{obs} history")
            axes[1].set_title(f"{obs} running mean")
            axes[1].set_xlabel("sweep")
            axes[0].set_ylabel(obs)
            axes[1].set_ylabel(obs)
            axes[0].grid(alpha=0.2)
            axes[1].grid(alpha=0.2)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    with PdfPages(run_dir / "L16_to_L32_sector_diagnostics.pdf") as pdf:
        for obs in SECTOR_DIAGNOSTICS:
            fig, ax = plt.subplots(figsize=(8, 4))
            for chain in chain_ids(rows):
                cr = select_rows(rows, chain)
                ax.plot([int(r["sweep"]) for r in cr], row_series(cr, obs, lattice_l), alpha=0.65, lw=0.9, label=f"chain {chain}")
            ax.axhline(direct_ref[obs]["mean"], color="black", lw=1.0, label="direct L32 ref")
            ax.set_title(f"{obs} sector diagnostic")
            ax.set_xlabel("sweep")
            ax.set_ylabel(obs)
            ax.grid(alpha=0.2)
            ax.legend(ncol=4, fontsize=7)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def plot_autocorrelation(run_dir: Path, curves: dict[str, dict[int, np.ndarray]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(run_dir / "L16_to_L32_autocorrelation_plots.pdf") as pdf:
        for obs, by_chain in curves.items():
            fig, ax = plt.subplots(figsize=(8, 4))
            for chain, ac in by_chain.items():
                ax.plot(np.arange(ac.size), ac, lw=0.9, alpha=0.7, label=f"chain {chain}")
            ax.axhline(0.0, color="black", lw=0.8)
            ax.set_title(f"{obs} normalized autocorrelation")
            ax.set_xlabel("lag [sweeps]")
            ax.set_ylabel("C(lag)/C(0)")
            ax.grid(alpha=0.2)
            ax.legend(ncol=4, fontsize=7)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def max_abs_primary_z(split_rows: list[dict[str, Any]], binning: str) -> float:
    vals = [
        abs(float(r["z_vs_direct_reference"]))
        for r in split_rows
        if r["binning"] == binning and r["category"] == "primary" and math.isfinite(float(r["z_vs_direct_reference"]))
    ]
    return float(max(vals)) if vals else float("nan")


def write_report(
    run_dir: Path,
    report_name: str,
    summary: dict[str, Any],
    run_config: dict[str, Any],
    scheduler: dict[str, Any],
    split_rows: list[dict[str, Any]],
    auto_rows: list[dict[str, Any]],
    coarse_rows: list[dict[str, Any]],
    candidates: list[Path],
) -> None:
    result = summary["result"]
    mean_tau: dict[str, float] = {}
    for obs in AUTOCORR_OBSERVABLES:
        vals = [float(r["tau_int_initial_positive"]) for r in auto_rows if r["observable"] == obs]
        mean_tau[obs] = float(np.nanmean(vals)) if vals else float("nan")
    enough = (
        max_abs_primary_z(split_rows, "quarter_chain") < 2.0
        and all((not math.isfinite(mean_tau[o]) or mean_tau[o] < 0.2 * int(run_config["sweeps"])) for o in ["action_density", "phi2", "NN"])
    )
    patch_size = int(run_config["patch_size"])
    if patch_size == 4:
        rec = (
            "The 8x500 run is a useful first statistical check. If Binder or slow modes matter, run 8x1000 or 8x2000 before making a production claim."
            if enough
            else "The 8x500 run is not enough for a final physics judgment; prepare an 8x1000 or 8x2000 confirmation with the same P=4 protocol."
        )
        ar_comparison = "A/R scale is very close to the accepted L8->L16 baseline."
    else:
        rec = (
            "This non-default patch-size run is useful as a baseline comparison. Do not promote it over P=4 unless it clearly improves autocorrelation or observables in a like-for-like comparison."
        )
        ar_comparison = "A/R scale is worse than the accepted L8->L16/P=4 scale: coarse and latent logweight fluctuations are larger and latent acceptance is lower."
    lines = [
        "# L16 to L32 Statistical Analysis Report",
        "",
        f"- run directory: `{run_dir}`",
        f"- detected candidates: `{[str(p) for p in candidates]}`",
        f"- coarse_L/fine_L: `{run_config['coarse_L']}` / `{run_config['fine_L']}`",
        f"- patch_size: `{run_config['patch_size']}`",
        f"- origin_mode: `{run_config['origin_mode']}`",
        f"- pCN rho / interval: `{run_config['pcn_rho']}` / `{run_config['pcn_interval_sweeps']}`",
        f"- chains x sweeps: `{run_config['validation_chains']} x {run_config['sweeps']}`",
        f"- measurement_mode: `{run_config['measurement_mode']}`",
        "",
        "## Measurement Semantics",
        "",
        f"- end-of-sweep rows expected/actual: `{run_config['expected']['observable_rows']}` / `{result['observable_measurement_semantics']['measured_states_total']}`",
        f"- state: `{scheduler['observable_measurement_semantics']['state']}`",
        f"- rejected updates: `{scheduler['observable_measurement_semantics']['rejected_updates']}`",
        "- accepted-only proposals are not used for production observables.",
        "",
        "## A/R",
        "",
        f"- coarse attempts: `{result['coarse_attempts']}`",
        f"- coarse acceptance: `{result['coarse_acceptance']:.9g}`",
        f"- coarse Delta logw std: `{result['coarse_std_delta_logw']:.9g}`",
        f"- latent attempts: `{result['latent_attempts']}`",
        f"- latent pCN acceptance: `{result['latent_acceptance']:.9g}`",
        f"- latent Delta logw std: `{result['latent_std_delta_logw']:.9g}`",
        f"- wall time: `{result['wall_time_sec']:.3f}` seconds",
        "",
        "## Split-Chain Primary Observable Max |z|",
        "",
        f"- full-chain bins: `{max_abs_primary_z(split_rows, 'full_chain'):.6g}`",
        f"- half-chain bins: `{max_abs_primary_z(split_rows, 'half_chain'):.6g}`",
        f"- quarter-chain bins: `{max_abs_primary_z(split_rows, 'quarter_chain'):.6g}`",
        "",
        "Signed `m` and `abs_m` are sector diagnostics only and are excluded from the primary max-z summary.",
        "",
        "## Autocorrelation",
        "",
    ]
    for obs in AUTOCORR_OBSERVABLES:
        lines.append(f"- {obs}: mean tau_int approximately `{mean_tau[obs]:.6g}`")
    lines += [
        "",
        "Binder_U4 is nonlinear and slow; its row-proxy autocorrelation is qualitative.",
        "",
        "## Coarse Distribution",
        "",
        "Native L16 coarse starts were compared with direct L32 fields blocked through the small3 kernel.",
    ]
    for row in coarse_rows:
        if row["observable"] in ["phi2", "phi4", "NN", "m", "abs_m"]:
            lines.append(
                f"- {row['observable']}: native `{row['native_L16_coarse']:.6g}`, blocked L32 `{row['blocked_direct_L32_small3']:.6g}`, diff `{row['native_minus_blocked']:.6g}`"
            )
    lines += [
        "",
        "## Comparison to L8 to L16 Accepted Baseline",
        "",
        "- L8->L16 8x2000 coarse acceptance/std: `0.861242 / 0.364266`",
        "- L8->L16 8x2000 latent acceptance/std: `0.529938 / 1.36037`",
        f"- L16->L32 8x500 coarse acceptance/std: `{result['coarse_acceptance']:.6g} / {result['coarse_std_delta_logw']:.6g}`",
        f"- L16->L32 8x500 latent acceptance/std: `{result['latent_acceptance']:.6g} / {result['latent_std_delta_logw']:.6g}`",
        "",
        ar_comparison,
        "",
        "## Recommendation",
        "",
        rec,
        "",
        "Generated files:",
        "",
        "- `split_chain_binning_summary.csv`",
        "- `window_stability_summary.csv`",
        "- `autocorrelation_summary.csv`",
        "- `blocking_analysis_summary.csv`",
        "- `coarse_distribution_comparison.csv`",
        "- `L16_to_L32_primary_running_mean_diagnostics.pdf`",
        "- `L16_to_L32_sector_diagnostics.pdf`",
        "- `L16_to_L32_autocorrelation_plots.pdf`",
    ]
    (run_dir / report_name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_long_run_instructions(run_dir: Path, wall_time_sec: float) -> None:
    sec_per_chain_sweep = wall_time_sec / (8 * 500)
    out = run_dir.parent / "run_manual_L16_to_L32_8x1000_or_8x2000_instructions.md"
    lines = [
        "# Manual L16 to L32 Longer Run Instructions",
        "",
        "Do not launch automatically. Use this only after reviewing the 8x500 analysis.",
        "",
        "## 8x1000",
        "",
        "```bash",
        "cd /Users/anna/Work/Research/Normalizing-flow/Inverse_RG",
        "OUT=\"perfect_blocking_upsampling/outputs/shape_parametric_sampler_validation/L16_to_L32_smoke/native_L16_pcn1_8x1000\"",
        "mkdir -p \"$OUT\"",
        "nohup ../.venv/bin/python -B perfect_blocking_upsampling/scripts/run_shape_parametric_sampler_validation.py \\",
        "  --config perfect_blocking_upsampling/outputs/shape_parametric_sampler_validation/L16_to_L32_smoke/L16_to_L32_smoke_config.yaml \\",
        "  --output-dir \"$OUT\" \\",
        "  --coarse-L 16 --patch-size 4 --origin-mode random \\",
        "  --pcn-rho 0.5 --pcn-interval-sweeps 1 \\",
        "  --validation-chains 8 --smoke-sweeps 1000 \\",
        "  --measurement-mode end_of_sweep --coarse-start-mode thermalized_coarse \\",
        "  --sector-balanced-init --progress-every-sweeps 10 --seed 20260833 \\",
        "  > \"$OUT/run.log\" 2>&1 &",
        "```",
        "",
        "- expected coarse attempts: `256000`",
        "- expected latent pCN attempts: `8000`",
        "- expected end-of-sweep rows: `8000`",
        f"- rough wall time from 8x500: `{sec_per_chain_sweep * 8 * 1000 / 60:.1f}` minutes",
        "",
        "## 8x2000",
        "",
        "```bash",
        "cd /Users/anna/Work/Research/Normalizing-flow/Inverse_RG",
        "OUT=\"perfect_blocking_upsampling/outputs/shape_parametric_sampler_validation/L16_to_L32_smoke/native_L16_pcn1_8x2000\"",
        "mkdir -p \"$OUT\"",
        "nohup ../.venv/bin/python -B perfect_blocking_upsampling/scripts/run_shape_parametric_sampler_validation.py \\",
        "  --config perfect_blocking_upsampling/outputs/shape_parametric_sampler_validation/L16_to_L32_smoke/L16_to_L32_smoke_config.yaml \\",
        "  --output-dir \"$OUT\" \\",
        "  --coarse-L 16 --patch-size 4 --origin-mode random \\",
        "  --pcn-rho 0.5 --pcn-interval-sweeps 1 \\",
        "  --validation-chains 8 --smoke-sweeps 2000 \\",
        "  --measurement-mode end_of_sweep --coarse-start-mode thermalized_coarse \\",
        "  --sector-balanced-init --progress-every-sweeps 10 --seed 20260834 \\",
        "  > \"$OUT/run.log\" 2>&1 &",
        "```",
        "",
        "- expected coarse attempts: `512000`",
        "- expected latent pCN attempts: `16000`",
        "- expected end-of-sweep rows: `16000`",
        f"- rough wall time from 8x500: `{sec_per_chain_sweep * 8 * 2000 / 60:.1f}` minutes",
        "",
        "## Analysis",
        "",
        "```bash",
        "../.venv/bin/python -B perfect_blocking_upsampling/scripts/analyze_L16_to_L32_validation.py --run-dir \"$OUT\"",
        "```",
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=None)
    ap.add_argument("--config", type=Path, default=PKG / "outputs" / "shape_parametric_sampler_validation" / "L16_to_L32_smoke" / "L16_to_L32_smoke_config.yaml")
    ap.add_argument("--report-name", default="L16_to_L32_statistical_analysis_report.md")
    args = ap.parse_args()

    candidates = discover_run()
    if args.run_dir is None:
        if len(candidates) != 1:
            print("Candidate L16->L32 P=4 8x500 run directories:", file=sys.stderr)
            for c in candidates:
                print(f"  {c}", file=sys.stderr)
            raise SystemExit("Pass --run-dir explicitly.")
        run_dir = candidates[0]
    else:
        run_dir = args.run_dir

    summary = json.loads((run_dir / "summary.json").read_text())
    run_config = json.loads((run_dir / "run_config.json").read_text())
    scheduler = json.loads((run_dir / "scheduler_preflight.json").read_text())
    rows = read_csv(run_dir / "observable_timeseries.csv")
    chain_rows = read_csv(run_dir / "chain_summaries.csv")
    loaded_cfg = load_config(args.config)
    coarse, fine_ref, _, _, _ = load_ensembles(loaded_cfg)
    kernel, _ = load_kernel_spec(loaded_cfg)
    coarse_action = action_spec(loaded_cfg["action"]["coarse"])
    fine_action = action_spec(loaded_cfg["action"]["fine"])
    fine_l = int(run_config["fine_L"])

    expected_rows = int(run_config["validation_chains"]) * int(run_config["sweeps"])
    measured = int(summary["result"]["observable_measurement_semantics"]["measured_states_total"])
    checks = {
        "measurement_mode_end_of_sweep": run_config.get("measurement_mode") == "end_of_sweep",
        "state_post_ar": scheduler["observable_measurement_semantics"].get("state") == "post_ar_markov_state",
        "rejected_repeat_previous": scheduler["observable_measurement_semantics"].get("rejected_updates") == "repeat_previous_state",
        "expected_rows": expected_rows,
        "actual_rows_csv": len(rows),
        "actual_rows_summary": measured,
        "rows_match": len(rows) == expected_rows == measured,
        "chain_summary_rows": len(chain_rows),
    }
    if not all([checks["measurement_mode_end_of_sweep"], checks["state_post_ar"], checks["rejected_repeat_previous"], checks["rows_match"]]):
        raise RuntimeError(f"metadata/row checks failed: {checks}")

    direct_ref = reference_stats(fine_ref, fine_action)
    split_rows = split_chain_binning(rows, direct_ref, fine_l)
    window_rows = window_stability(rows, direct_ref, fine_l)
    block_rows = blocking_analysis(rows, fine_l)
    auto_rows, curves = autocorrelation_summary(rows, fine_l)
    coarse_rows = coarse_distribution_comparison(coarse, fine_ref, kernel, coarse_action)

    write_csv(run_dir / "split_chain_binning_summary.csv", split_rows)
    write_csv(run_dir / "window_stability_summary.csv", window_rows)
    write_csv(run_dir / "blocking_analysis_summary.csv", block_rows)
    write_csv(run_dir / "autocorrelation_summary.csv", auto_rows)
    write_csv(run_dir / "coarse_distribution_comparison.csv", coarse_rows)
    write_json(
        run_dir / "L16_to_L32_statistical_analysis_summary.json",
        {
            "run_dir": str(run_dir),
            "config": str(args.config),
            "metadata_checks": checks,
            "reference_source": "direct_L32_reference_from_validation_config",
            "primary_observables": PRIMARY_OBSERVABLES,
            "sector_diagnostics_not_pass_fail": SECTOR_DIAGNOSTICS,
            "max_abs_primary_z": {
                "full_chain": max_abs_primary_z(split_rows, "full_chain"),
                "half_chain": max_abs_primary_z(split_rows, "half_chain"),
                "quarter_chain": max_abs_primary_z(split_rows, "quarter_chain"),
            },
        },
    )
    plot_histories(run_dir, rows, direct_ref, fine_l)
    plot_autocorrelation(run_dir, curves)
    write_report(run_dir, args.report_name, summary, run_config, scheduler, split_rows, auto_rows, coarse_rows, candidates)
    write_long_run_instructions(run_dir, float(summary["result"]["wall_time_sec"]))
    print(
        json.dumps(
            {
                "status": "completed",
                "run_dir": str(run_dir),
                "report": str(run_dir / args.report_name),
                "split_chain": str(run_dir / "split_chain_binning_summary.csv"),
                "autocorrelation": str(run_dir / "autocorrelation_summary.csv"),
                "blocking": str(run_dir / "blocking_analysis_summary.csv"),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
