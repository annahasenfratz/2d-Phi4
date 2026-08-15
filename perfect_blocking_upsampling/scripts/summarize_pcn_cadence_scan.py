#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

PRIMARY = ["phi2", "phi4", "NN", "action_density", "susceptibility", "Binder_U4", "xi_over_L"]
AUTOCORR_PRIMARY = ["action_density", "phi2", "NN", "Binder_U4"]


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


def finite_max(vals: list[float]) -> float:
    x = [abs(float(v)) for v in vals if math.isfinite(float(v))]
    return float(max(x)) if x else float("nan")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def interval_from_dir(path: Path) -> int:
    name = path.name
    if "interval" in name:
        return int(name.split("interval")[-1])
    return int(name)


def summarize_cadence(run_dir: Path) -> dict[str, Any]:
    interval = interval_from_dir(run_dir)
    summary = load_json(run_dir / "summary.json")["result"]
    preflight = load_json(run_dir / "scheduler_preflight.json")
    sector = load_json(run_dir / "sector_occupancy.json")
    split = read_csv(run_dir / "split_chain_binning_summary.csv")
    auto = read_csv(run_dir / "autocorrelation_summary.csv")
    obs_diag = load_json(run_dir / "observable_diagnostics.json").get("observables", {})
    full_primary = [r for r in split if r["binning"] == "full_chain" and r.get("category") == "primary"]
    quarter_primary = [r for r in split if r["binning"] == "quarter_chain" and r.get("category") == "primary"]
    primary_max_z_full = finite_max([float(r["z_vs_direct_reference"]) for r in full_primary])
    primary_max_z_quarter = finite_max([float(r["z_vs_direct_reference"]) for r in quarter_primary])
    tau_rows = [r for r in auto if r["observable"] in AUTOCORR_PRIMARY]
    tau_mean = float(np.nanmean([float(r["tau_int_initial_positive"]) for r in tau_rows])) if tau_rows else float("nan")
    ess_mean = float(np.nanmean([float(r["effective_sample_size"]) for r in tau_rows])) if tau_rows else float("nan")
    action_tau = [float(r["tau_int_initial_positive"]) for r in auto if r["observable"] == "action_density"]
    phi2_tau = [float(r["tau_int_initial_positive"]) for r in auto if r["observable"] == "phi2"]
    nn_tau = [float(r["tau_int_initial_positive"]) for r in auto if r["observable"] == "NN"]
    binder_tau = [float(r["tau_int_initial_positive"]) for r in auto if r["observable"] == "Binder_U4"]
    return {
        "pcn_interval_sweeps": interval,
        "run_dir": str(run_dir),
        "validation_mode": "native_L8_deployment_full_coarse_update",
        "measurement_mode": "end_of_sweep",
        "coarse_attempts": summary["coarse_attempts"],
        "coarse_acceptance": summary["coarse_acceptance"],
        "coarse_delta_logw_std": summary["coarse_std_delta_logw"],
        "latent_attempts": summary["latent_attempts"],
        "expected_latent_attempts": preflight["expected_latent_pcn_attempts"],
        "latent_acceptance": summary["latent_acceptance"],
        "latent_delta_logw_std": summary["latent_std_delta_logw"],
        "wall_time_sec": summary["wall_time_sec"],
        "fraction_positive": sector["fraction_positive"],
        "fraction_negative": sector["fraction_negative"],
        "sign_flips": sector["total_sign_flips"],
        "signed_m_mean": obs_diag.get("m", {}).get("mean", float("nan")),
        "primary_max_abs_z_full_chain_bins": primary_max_z_full,
        "primary_max_abs_z_quarter_chain_bins": primary_max_z_quarter,
        "mean_tau_int_primary": tau_mean,
        "mean_ess_per_chain_primary": ess_mean,
        "tau_action_density_mean": float(np.nanmean(action_tau)) if action_tau else float("nan"),
        "tau_phi2_mean": float(np.nanmean(phi2_tau)) if phi2_tau else float("nan"),
        "tau_NN_mean": float(np.nanmean(nn_tau)) if nn_tau else float("nan"),
        "tau_Binder_U4_mean": float(np.nanmean(binder_tau)) if binder_tau else float("nan"),
        **{
            f"{obs}_mean_full_chain_bins": next(
                (float(r["mean"]) for r in full_primary if r["observable"] == obs),
                float("nan"),
            )
            for obs in PRIMARY
        },
    }


def choose_best(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def score(row: dict[str, Any]) -> tuple[float, float, float]:
        latent_penalty = 0.0 if row["latent_acceptance"] >= 0.5 else (0.5 - row["latent_acceptance"]) * 10.0
        return (
            -float(row["mean_ess_per_chain_primary"]) + latent_penalty,
            float(row["primary_max_abs_z_full_chain_bins"]),
            float(row["latent_delta_logw_std"]),
        )

    return sorted(rows, key=score)[0]


def plot_scan(rows: list[dict[str, Any]], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = sorted(rows, key=lambda r: r["pcn_interval_sweeps"])
    x = np.asarray([r["pcn_interval_sweeps"] for r in rows], dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5))
    panels = [
        ("latent_acceptance", "latent acceptance"),
        ("latent_delta_logw_std", "latent Delta logw std"),
        ("mean_tau_int_primary", "mean tau_int primary"),
        ("mean_ess_per_chain_primary", "mean ESS/chain primary"),
    ]
    for ax, (key, title) in zip(axes.reshape(-1), panels):
        y = np.asarray([r[key] for r in rows], dtype=float)
        ax.plot(x, y, marker="o")
        ax.set_xscale("log")
        ax.invert_xaxis()
        ax.set_xlabel("pCN interval (sweeps)")
        ax.set_title(title)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def write_report(rows: list[dict[str, Any]], best: dict[str, Any], out: Path) -> None:
    lines = [
        "# pCN Cadence Scan",
        "",
        "Validation mode: `native_L8_deployment_full_coarse_update`.",
        "",
        "- Starts: thermalized native L8 coarse configurations.",
        "- Coarse patch updates: enabled.",
        "- Measurement mode: end-of-sweep Markov states.",
        "- pCN rho: 0.5.",
        "- `m` and `|m|` are sector diagnostics only, not pass/fail observables.",
        "",
        "## Summary",
        "",
        "| interval | coarse acc | coarse std | latent attempts | latent acc | latent std | primary max |z| | mean tau primary | mean ESS/chain | sign flips | wall sec |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda r: r["pcn_interval_sweeps"]):
        lines.append(
            f"| {row['pcn_interval_sweeps']} | {row['coarse_acceptance']:.6g} | "
            f"{row['coarse_delta_logw_std']:.6g} | {row['latent_attempts']} | "
            f"{row['latent_acceptance']:.6g} | {row['latent_delta_logw_std']:.6g} | "
            f"{row['primary_max_abs_z_full_chain_bins']:.6g} | {row['mean_tau_int_primary']:.6g} | "
            f"{row['mean_ess_per_chain_primary']:.6g} | {row['sign_flips']} | {row['wall_time_sec']:.1f} |"
        )
    lines += [
        "",
        "## Decision",
        "",
        f"Recommended cadence from this moderate scan: pCN every `{best['pcn_interval_sweeps']}` sweeps.",
        "",
        "This recommendation favors the largest primary-observable ESS per chain while requiring latent acceptance to remain reasonable. Treat it as provisional because each chain is only 500 sweeps.",
        "",
        "If a longer confirmation keeps latent acceptance acceptable and autocorrelation improved, use this cadence as the native-L8 deployment default. If latent acceptance becomes marginal in the longer run, scan rho at the selected cadence.",
        "",
        "## Caveats",
        "",
        "- These are moderate 4x500 runs, so autocorrelation estimates are noisy.",
        "- The scan is native-L8 deployment only, not fixed blocked-coarse conditional validation.",
        "- No L16->L32 or larger-volume run was launched.",
        "",
        "Plots:",
        "",
        "- `pcn_cadence_scan_autocorr_blocking.pdf`",
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_long_run_instructions(scan_dir: Path, best: dict[str, Any]) -> None:
    interval = int(best["pcn_interval_sweeps"])
    out_dir = scan_dir / f"confirm_interval{interval}_8x1000"
    cmd = (
        "../.venv/bin/python -B perfect_blocking_upsampling/scripts/run_shape_parametric_sampler_validation.py "
        f"--output-dir {out_dir} --coarse-L 8 --patch-size 4 --origin-mode random "
        f"--smoke-sweeps 1000 --validation-chains 8 --pcn-rho 0.5 --pcn-interval-sweeps {interval} "
        "--seed 20260801 --sector-balanced-init --progress-every-sweeps 25 "
        "--measurement-mode end_of_sweep --coarse-start-mode thermalized_coarse"
    )
    n_patch = 8
    coarse_attempts = 8 * 1000 * n_patch
    latent_attempts = 8 * (1000 // interval)
    md = f"""# Longer Confirmation Run Instructions

Do not launch this automatically from Codex. This is for Anna to run manually.

Recommended cadence from the moderate scan: pCN every `{interval}` sweeps.

Working directory:

`/Users/anna/Work/Research/Normalizing-flow/Inverse_RG`

Expected output directory:

`{out_dir}`

Expected attempts:

- coarse attempts: `{coarse_attempts}`
- latent pCN attempts: `{latent_attempts}`

Expected wall time:

- Use the scan wall time as a rough guide. The 8x1000 confirmation is about 4x the work of one 4x500 cadence run, plus overhead.

Preflight command:

```bash
{cmd} --preflight-only
```

Run command:

```bash
mkdir -p {out_dir}
nohup {cmd} > {out_dir}/run.log 2>&1 &
echo $! > {out_dir}/run.pid
```

Analysis commands after completion:

```bash
../.venv/bin/python -B perfect_blocking_upsampling/scripts/analyze_shape_parametric_validation.py --run-dir {out_dir}
../.venv/bin/python -B perfect_blocking_upsampling/scripts/analyze_target_distribution_debug.py --run-dir {out_dir}
MPLCONFIGDIR={out_dir}/mplconfig ../.venv/bin/python -B perfect_blocking_upsampling/scripts/sector_aware_statistical_diagnostics.py --run-dir {out_dir}
```

Files to inspect:

- `{out_dir}/summary.json`
- `{out_dir}/chain_summaries.csv`
- `{out_dir}/sector_aware_analysis_report.md`
- `{out_dir}/sector_aware_statistical_diagnostics.md`
- `{out_dir}/split_chain_binning_summary.csv`
- `{out_dir}/autocorrelation_summary.csv`
- `{out_dir}/sector_aware_primary_running_mean_diagnostics.pdf`
- `{out_dir}/run.log`

Do not start L16->L32 from this instruction set.
"""
    (scan_dir / "long_run_instructions.md").write_text(md, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan-dir", type=Path, required=True)
    args = ap.parse_args()
    rows = []
    for run_dir in sorted(args.scan_dir.glob("interval*"), key=interval_from_dir):
        if (run_dir / "summary.json").exists() and (run_dir / "split_chain_binning_summary.csv").exists():
            rows.append(summarize_cadence(run_dir))
    if not rows:
        raise RuntimeError(f"no completed cadence directories found in {args.scan_dir}")
    best = choose_best(rows)
    write_csv(args.scan_dir / "pcn_cadence_scan_summary.csv", sorted(rows, key=lambda r: r["pcn_interval_sweeps"]))
    plot_scan(rows, args.scan_dir / "pcn_cadence_scan_autocorr_blocking.pdf")
    write_report(rows, best, args.scan_dir / "pcn_cadence_scan_report.md")
    write_long_run_instructions(args.scan_dir, best)
    print(json.dumps({"status": "completed", "best_interval": best["pcn_interval_sweeps"], "summary": str(args.scan_dir / "pcn_cadence_scan_summary.csv")}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
