#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def mean_float(rows: list[dict[str, Any]], key: str) -> float:
    vals = [float(r[key]) for r in rows if r.get(key, "") != ""]
    return float(np.mean(vals)) if vals else float("nan")


PRIMARY_OBSERVABLES = ["phi2", "phi4", "NN", "action_density", "Binder_U4", "susceptibility", "xi_over_L"]
SECTOR_DIAGNOSTICS = ["m", "abs_m"]


def max_abs_z_for(obs_diag: dict[str, Any], keys: list[str]) -> float:
    vals = []
    for key in keys:
        z = obs_diag.get("observables", {}).get(key, {}).get("z")
        if z is not None and np.isfinite(float(z)):
            vals.append(abs(float(z)))
    return float(max(vals)) if vals else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    args = ap.parse_args()
    run = args.run_dir
    summary_path = run / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"run summary not found: {summary_path}")
    summary = json.loads(summary_path.read_text())
    chain_rows = read_csv(run / "chain_summaries.csv")
    obs_rows = read_csv(run / "observable_timeseries.csv")
    sector = json.loads((run / "sector_occupancy.json").read_text()) if (run / "sector_occupancy.json").exists() else {}
    obs_diag = json.loads((run / "observable_diagnostics.json").read_text()) if (run / "observable_diagnostics.json").exists() else {}
    result = summary.get("result", {})
    max_primary_z = obs_diag.get("max_abs_z_primary_observables", max_abs_z_for(obs_diag, PRIMARY_OBSERVABLES))
    max_sector_z = obs_diag.get("max_abs_z_sector_diagnostics", max_abs_z_for(obs_diag, SECTOR_DIAGNOSTICS))
    payload = {
        "run_dir": str(run),
        "coarse_acceptance": result.get("coarse_acceptance"),
        "coarse_delta_logw_std": result.get("coarse_std_delta_logw"),
        "latent_pcn_acceptance": result.get("latent_acceptance"),
        "latent_delta_logw_std": result.get("latent_std_delta_logw"),
        "coarse_attempts": result.get("coarse_attempts"),
        "latent_attempts": result.get("latent_attempts"),
        "sector_occupancy": sector,
        "observable_diagnostics": obs_diag,
        "validation_observable_policy": {
            "primary_observables": PRIMARY_OBSERVABLES,
            "sector_diagnostics_not_pass_fail": SECTOR_DIAGNOSTICS,
            "max_abs_z_primary_observables": max_primary_z,
            "max_abs_z_sector_diagnostics_not_pass_fail": max_sector_z,
        },
        "per_chain_mean_m": {str(i): mean_float([r for r in obs_rows if int(r["chain_id"]) == i], "m") for i in sorted({int(r["chain_id"]) for r in obs_rows})} if obs_rows else {},
        "per_chain_mean_abs_m": {str(i): mean_float([r for r in obs_rows if int(r["chain_id"]) == i], "abs_m") for i in sorted({int(r["chain_id"]) for r in obs_rows})} if obs_rows else {},
        "chain_summaries": chain_rows,
    }
    out_json = run / "sector_aware_analysis_summary.json"
    out_json.write_text(json.dumps(payload, indent=2, default=float) + "\n", encoding="utf-8")
    obs = obs_diag.get("observables", {})
    lines = [
        "# Sector-Aware Validation Analysis",
        "",
        f"- coarse acceptance: `{payload['coarse_acceptance']}`",
        f"- coarse Delta logw std: `{payload['coarse_delta_logw_std']}`",
        f"- latent pCN acceptance: `{payload['latent_pcn_acceptance']}`",
        f"- latent Delta logw std: `{payload['latent_delta_logw_std']}`",
        f"- coarse attempts: `{payload['coarse_attempts']}`",
        f"- latent attempts: `{payload['latent_attempts']}`",
        f"- fraction positive: `{sector.get('fraction_positive')}`",
        f"- fraction negative: `{sector.get('fraction_negative')}`",
        f"- total sign flips: `{sector.get('total_sign_flips')}`",
        f"- max |z| primary observables: `{max_primary_z}`",
        f"- max |z| sector diagnostics, not pass/fail: `{max_sector_z}`",
        "",
        "## Primary Observables",
    ]
    for key in PRIMARY_OBSERVABLES:
        if key in obs:
            row = obs[key]
            lines.append(f"- `{key}`: mean `{row.get('mean')}`, reference `{row.get('reference')}`, z `{row.get('z')}`")
    lines.append("")
    lines.append("## Sector Diagnostics (Not Pass/Fail)")
    for key in SECTOR_DIAGNOSTICS:
        if key in obs:
            row = obs[key]
            lines.append(f"- `{key}`: mean `{row.get('mean')}`, reference `{row.get('reference')}`, z `{row.get('z')}`")
    (run / "sector_aware_analysis_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "completed", "summary": str(out_json)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
