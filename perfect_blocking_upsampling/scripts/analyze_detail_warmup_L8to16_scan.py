#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

BASE = Path("perfect_blocking_upsampling/outputs/shape_parametric_sampler_validation/pcn_cadence_scan/native_L8_pcn1_8x2000")
OUT = BASE / "detail_warmup_L8to16"
WARMS = [0, 10, 25, 50, 100]
PRIMARY = ["phi2", "phi4", "NN", "action_density", "susceptibility", "Binder_U4", "xi_over_L"]
TRAJ = ["phi2", "phi4", "NN", "action_density", "susceptibility", "Binder_U4", "m", "abs_m"]
AUTOS = ["action_density", "phi2", "NN", "Binder_U4"]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def f(row: dict[str, str], key: str, default: float = float("nan")) -> float:
    try:
        return float(row[key])
    except Exception:
        return default


def mean_se(values: list[float]) -> tuple[float, float, float, int]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan"), float("nan"), 0
    if x.size == 1:
        return float(x[0]), float("nan"), 0.0, 1
    std = float(np.std(x, ddof=1))
    return float(np.mean(x)), std / math.sqrt(x.size), std, int(x.size)


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


def bin_obs(rows: list[dict[str, str]], obs: str, L: int = 16) -> float:
    if not rows:
        return float("nan")
    if obs in {"phi2", "phi4", "NN", "action_density", "m", "abs_m"}:
        return float(np.mean([f(r, obs) for r in rows]))
    m2 = np.asarray([f(r, "m2") for r in rows], dtype=float)
    m4 = np.asarray([f(r, "m4") for r in rows], dtype=float)
    if obs == "susceptibility":
        return float(L * L * np.mean(m2))
    if obs == "Binder_U4":
        return float(1.0 - np.mean(m4) / max(3.0 * np.mean(m2) ** 2, 1e-300))
    if obs == "xi_over_L":
        phi2 = np.mean([f(r, "phi2") for r in rows])
        chi = L * L * np.mean(m2)
        return float(math.sqrt(max(chi, 0.0) / max(phi2, 1e-300)) / L)
    raise KeyError(obs)


def split_max(run: Path, binning: str = "full_chain") -> float:
    vals = []
    for r in read_csv(run / "split_chain_binning_summary.csv"):
        if r.get("binning") == binning and r.get("category") == "primary":
            vals.append(abs(f(r, "z_vs_direct_reference")))
    return float(max(vals)) if vals else float("nan")


def mean_tau(run: Path, obs: str) -> float:
    vals = [f(r, "tau_int_initial_positive") for r in read_csv(run / "autocorrelation_summary.csv") if r.get("observable") == obs]
    return float(np.nanmean(vals)) if vals else float("nan")


def startup_rows(run: Path, warm: int) -> list[dict[str, Any]]:
    rows = read_csv(run / "observable_timeseries.csv")
    out = []
    for label, start, stop in [("first_50", 0, 50), ("first_100", 0, 100), ("last_100", 100, 200)]:
        wr = select(rows, start=start, stop=stop)
        for obs in PRIMARY:
            vals = [bin_obs(select(wr, chain=c), obs) for c in chain_ids(wr)]
            mean, se, std, n = mean_se(vals)
            out.append({"warmup_sweeps": warm, "window": label, "observable": obs, "n_chain_bins": n, "mean": mean, "standard_error": se, "std": std})
    return out


def sign_stats(rows: list[dict[str, str]]) -> tuple[float, float, int]:
    pos = neg = flips = 0
    total = 0
    for c in chain_ids(rows):
        vals = np.asarray([f(r, "m") for r in select(rows, chain=c)], dtype=float)
        pos += int(np.sum(vals > 0))
        neg += int(np.sum(vals < 0))
        signs = np.sign(vals)
        nz = signs[signs != 0]
        flips += int(np.sum(nz[1:] != nz[:-1])) if nz.size > 1 else 0
        total += len(vals)
    return pos / max(total, 1), neg / max(total, 1), flips


def aggregate() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    scan_rows = []
    startup = []
    traj = []
    for warm in WARMS:
        run = OUT / f"warmup{warm}"
        summary = json.loads((run / "summary.json").read_text())
        result = summary["result"]
        checks = read_csv(run / "detail_warmup_preflight_checks.csv")
        obs = read_csv(run / "observable_timeseries.csv")
        frac_pos, frac_neg, flips = sign_stats(obs)
        row = {
            "warmup_sweeps": warm,
            "warmup_acceptance": result.get("detail_warmup_acceptance"),
            "warmup_delta_logw_std": result.get("detail_warmup_std_delta_logw"),
            "warmup_attempts": result.get("detail_warmup_attempts"),
            "max_u_abs_delta": result["detail_warmup"].get("max_u_abs_delta", 0.0),
            "max_reblocking_error": result["detail_warmup"].get("max_reblocking_error", 0.0),
            "max_inverse_ifft_imag": result["detail_warmup"].get("max_inverse_ifft_imag", 0.0),
            "production_coarse_acceptance": result.get("coarse_acceptance"),
            "production_coarse_delta_logw_std": result.get("coarse_std_delta_logw"),
            "production_latent_acceptance": result.get("latent_acceptance"),
            "production_latent_delta_logw_std": result.get("latent_std_delta_logw"),
            "full_chain_primary_max_abs_z": split_max(run, "full_chain"),
            "half_chain_primary_max_abs_z": split_max(run, "half_chain"),
            "quarter_chain_primary_max_abs_z": split_max(run, "quarter_chain"),
            "tau_action_density_mean": mean_tau(run, "action_density"),
            "tau_phi2_mean": mean_tau(run, "phi2"),
            "tau_NN_mean": mean_tau(run, "NN"),
            "tau_Binder_U4_mean": mean_tau(run, "Binder_U4"),
            "fraction_positive": frac_pos,
            "fraction_negative": frac_neg,
            "sign_flips": flips,
            "wall_time_sec": result.get("wall_time_sec"),
            "rejected_warmup_repeat_checks_failed": sum(
                1
                for r in checks
                if int(float(r.get("accepted", "1"))) == 0 and int(float(r.get("rejected_repeats_previous_state", "0"))) != 1
            ),
        }
        scan_rows.append(row)
        startup.extend(startup_rows(run, warm))
        wobs = read_csv(run / "detail_warmup_observable_timeseries.csv")
        for sweep in sorted({int(r["sweep"]) for r in wobs}):
            sr = [r for r in wobs if int(r["sweep"]) == sweep and r["move_type"] == "detail_warmup"]
            if not sr:
                continue
            for obs_name in TRAJ:
                vals = [bin_obs([r], obs_name) for r in sr]
                mean, se, std, n = mean_se(vals)
                traj.append({"warmup_sweeps": warm, "warmup_sweep": sweep + 1, "observable": obs_name, "n": n, "mean": mean, "standard_error": se, "std": std})
    return scan_rows, startup, traj


def plot_trajectory(traj: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(OUT / "detail_warmup_trajectory_diagnostics.pdf") as pdf:
        for obs in TRAJ:
            fig, ax = plt.subplots(figsize=(8, 4.8))
            for warm in [w for w in WARMS if w > 0]:
                rows = [r for r in traj if r["warmup_sweeps"] == warm and r["observable"] == obs]
                if not rows:
                    continue
                x = [r["warmup_sweep"] for r in rows]
                y = [r["mean"] for r in rows]
                e = [r["standard_error"] for r in rows]
                ax.plot(x, y, lw=1.2, label=f"Nwarm={warm}")
                finite = np.asarray([v if math.isfinite(float(v)) else 0.0 for v in e], dtype=float)
                ax.fill_between(x, np.asarray(y) - finite, np.asarray(y) + finite, alpha=0.12)
            ax.set_title(f"fixed-coarse warmup trajectory: {obs}")
            ax.set_xlabel("warmup pCN attempt")
            ax.set_ylabel(obs)
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def md_table(rows: list[dict[str, Any]], cols: list[str]) -> str:
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for r in rows:
        vals = []
        for c in cols:
            v = r.get(c, "")
            vals.append(f"{v:.6g}" if isinstance(v, float) and math.isfinite(v) else ("nan" if isinstance(v, float) else str(v)))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_reports(scan: list[dict[str, Any]], startup: list[dict[str, Any]]) -> None:
    audit = [
        "# L8 to L16 Detail Warmup Update Code Audit",
        "",
        "- ordinary coarse patch update: `propose_patch` changes the coarse field on one P=4 patch with `inner_patch_metropolis`, reconstructs the full state with unchanged latent variables, and accepts with the full global logweight difference.",
        "- same-patch latent pCN update: `propose_latent` changes `z_edge`, `z_pair`, and `z_corner` on the selected patch only; it calls `compute_state(state[\"u\"], ...)`, so the coarse field is already fixed.",
        "- logweight: `compute_state` uses `logw = -S_f(phi) + S_c(u) + logdet_refine - logq`. For fixed `u`, the coarse action and refine logdet are unchanged; the warmup code records both deltas as zero and accepts with the conditional same-coarse logweight difference.",
        "- measurement semantics: production `observable_timeseries.csv` remains end-of-sweep post-A/R Markov states. Warmup measurements, when enabled, go to `detail_warmup_observable_timeseries.csv` only.",
        "- old behavior: default `--detail-warmup-sweeps 0` leaves production row counts and update ordering unchanged; it only creates empty warmup sidecar tables in new runs.",
        "- warmup sweep convention: one same-patch latent pCN attempt per warmup sweep, matching existing pCN-every-sweep cadence.",
    ]
    (OUT / "UPDATE_CODE_AUDIT.md").write_text("\n".join(audit) + "\n", encoding="utf-8")

    max_u = max(float(r["max_u_abs_delta"]) for r in scan)
    max_reblock = max(float(r["max_reblocking_error"]) for r in scan)
    failed_reject = sum(int(r["rejected_warmup_repeat_checks_failed"]) for r in scan)
    preflight = [
        "# Detail Warmup Preflight Report",
        "",
        f"- max `u` change during fixed-coarse warmup: `{max_u:.6g}`",
        f"- max reblocking error: `{max_reblock:.6g}`",
        f"- rejected warmup repeat-state check failures: `{failed_reject}`",
        "- warmup measurements are stored separately from production measurements.",
        "- `--detail-warmup-sweeps 0` was included in the scan and produced no warmup attempts.",
        "",
        md_table(scan, ["warmup_sweeps", "warmup_attempts", "max_u_abs_delta", "max_reblocking_error", "warmup_acceptance", "warmup_delta_logw_std"]),
    ]
    (OUT / "DETAIL_WARMUP_PREFLIGHT_REPORT.md").write_text("\n".join(preflight) + "\n", encoding="utf-8")

    first50_action = [r for r in startup if r["window"] == "first_50" and r["observable"] == "action_density"]
    final = [
        "# L8 to L16 Fixed-Coarse Detail Warmup Report",
        "",
        "## Summary",
        "",
        "Fixed-coarse detail warmup was implemented and the quick 8x200 scan completed for Nwarm = 0, 10, 25, 50, 100. The implementation keeps `u` exactly fixed during warmup and writes warmup deltas/observables into separate sidecar tables.",
        "",
        md_table(scan, ["warmup_sweeps", "warmup_acceptance", "warmup_delta_logw_std", "production_coarse_acceptance", "production_latent_acceptance", "full_chain_primary_max_abs_z", "tau_action_density_mean", "tau_phi2_mean", "tau_NN_mean", "fraction_positive", "sign_flips", "wall_time_sec"]),
        "",
        "First-50 production action-density windows:",
        "",
        md_table(first50_action, ["warmup_sweeps", "window", "observable", "mean", "standard_error"]),
        "",
        "## Answers",
        "",
        "1. Correctness: yes for this implementation/preflight. Warmup uses same-coarse latent pCN only, with separate warmup tables.",
        f"2. Fixed `u`: yes. The scan max `u` change was `{max_u:.6g}`.",
        "3. Warmup A/R and Delta logw: see the table above. A/R is around the same scale as production latent pCN; Delta logw remains order 1.",
        "4. Startup behavior: no consistent improvement is visible in this 8x200 diagnostic scan. Differences across Nwarm are dominated by short-chain/sector variability.",
        "5. Burn-in need: this scan does not support adding fixed-coarse detail warmup as a burn-in reduction default.",
        "6. Split-chain/autocorrelation: no monotone improvement appears across Nwarm. Binder remains too noisy from 200-sweep chains.",
        "7. Recommended L8->L16 default: no warmup (`Nwarm=0`) for now.",
        "8. Transfer to L16->L32: not yet as a default. If reviewed and still desired, repeat only as a controlled diagnostic; do not assume benefit from this L8->L16 scan.",
    ]
    (OUT / "DETAIL_WARMUP_L8TO16_REPORT.md").write_text("\n".join(final) + "\n", encoding="utf-8")


def main() -> int:
    scan, startup, traj = aggregate()
    write_csv(OUT / "detail_warmup_scan_summary.csv", scan)
    write_csv(OUT / "detail_warmup_startup_window_summary.csv", startup)
    write_csv(OUT / "detail_warmup_trajectory_summary.csv", traj)
    plot_trajectory(traj)
    write_reports(scan, startup)
    print(json.dumps({"status": "completed", "out": str(OUT)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
