#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PKG = PROJECT_ROOT / "perfect_blocking_upsampling"
sys.path.insert(0, str(PKG / "scripts"))

from run_L64_to_L128_kappaf_matching_experiment import DEFAULT_CONFIG, run_smoke  # noqa: E402

DEFAULT_OUT = PKG / "outputs" / "shape_parametric_sampler_validation" / "L64_to_L128_P32_latentP12_beta04_kappaf_scan_kappac0p2705"
SAME_KAPPA_DIR = PKG / "outputs" / "shape_parametric_sampler_validation" / "L64_to_L128_P32_latentP12_beta04_acceptance_smoke_same_kappa_0p2705"


def log(message: str) -> None:
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {message}", flush=True)


def tag(kappa: float) -> str:
    return f"kappaf_{kappa:.5f}".replace(".", "p")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def finite_count(df: pd.DataFrame, cols: list[str]) -> int:
    return int(df[cols].replace([np.inf, -np.inf], np.nan).isna().sum().sum())


def collect_run(run_dir: Path, kappa_f: float, source: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    summary = read_json(run_dir / "summary.json")
    ar_summary = pd.read_csv(run_dir / "patch_update_acceptance_summary.csv")
    obs = pd.read_csv(run_dir / "local_action_observables_summary.csv")
    ar_rows = pd.read_csv(run_dir / "patch_update_AR_rows.csv")
    obs_rows = pd.read_csv(run_dir / "local_action_observables_by_chain_sweep.csv")
    req_ar = ["accepted", "delta_logw", "delta_Sf", "delta_Sc", "delta_logdet_refine", "delta_logq_missing", "patch_x", "patch_y", "sweep", "attempt_in_sweep"]
    req_obs = ["phi2", "phi4", "NN", "2nn", "diag", "action_density"]
    nan_ar = finite_count(ar_rows, req_ar)
    nan_obs = finite_count(obs_rows, req_obs)

    acc_rows: list[dict[str, Any]] = []
    dlog_rows: list[dict[str, Any]] = []
    for _, r in ar_summary.iterrows():
        substep = "" if pd.isna(r.get("substep", np.nan)) else int(r["substep"])
        actual = summary["actual_total_coarse_attempts"] if r["move_type"] == "coarse_patch" else summary["actual_total_latent_attempts"]
        if r["move_type"] == "latent_pCN_by_substep":
            actual = summary["actual_total_latent_attempts"] // summary["latent_updates_per_coarse"]
        acc_rows.append(
            {
                "kappa_f": kappa_f,
                "source": source,
                "move_type": r["move_type"],
                "substep": substep,
                "attempts_saved_sweeps": int(r["attempts_saved_sweeps"]),
                "accepts_saved_sweeps": int(r["accepts_saved_sweeps"]),
                "acceptance_saved_sweeps": float(r["acceptance_saved_sweeps"]),
                "actual_total_attempts_estimate": int(actual),
            }
        )
        dlog_rows.append(
            {
                "kappa_f": kappa_f,
                "source": source,
                "move_type": r["move_type"],
                "substep": substep,
                "delta_logw_mean": float(r["delta_logw_mean"]),
                "delta_logw_std": float(r["delta_logw_std"]),
                "delta_logw_q05": float(r["delta_logw_q05"]),
                "delta_logw_q50": float(r["delta_logw_q50"]),
                "delta_logw_q95": float(r["delta_logw_q95"]),
            }
        )
    obs_rows_out = []
    for _, r in obs.iterrows():
        row = {"kappa_f": kappa_f, "source": source, "sweep": int(r["sweep"]), "n_chains": int(r["n_chains"])}
        for key in ["phi2", "phi4", "NN", "2nn", "diag", "action_density"]:
            row[key] = float(r[key])
            row[key + "_se"] = float(r.get(key + "_se", np.nan))
        obs_rows_out.append(row)
    meta = {
        "kappa_f": kappa_f,
        "source": source,
        "run_dir": str(run_dir),
        "coarse_source": summary["preflight"]["coarse_path"],
        "elapsed_sec": float(summary["elapsed_sec"]),
        "sec_per_chain_sweep": float(summary["sec_per_chain_sweep"]),
        "nan_required_ar": nan_ar,
        "nan_required_obs": nan_obs,
        "shape_errors": False,
        "preflight_status": summary["preflight"]["status"],
    }
    return acc_rows, dlog_rows, obs_rows_out, meta


def write_report(out: Path, acc: pd.DataFrame, dlog: pd.DataFrame, obs: pd.DataFrame, metas: list[dict[str, Any]]) -> None:
    lines: list[str] = []
    lines += ["# L64->L128 kappaf scan P32 smoke report", ""]
    lines += ["Short smoke scan only. Fine targets scanned: `0.27075`, `0.27100`; existing same-kappa `0.27050` is included as baseline.", ""]
    lines += ["## Runtime / Preflight", "", "| kappa_f | source | sec/chain-sweep | elapsed_sec | preflight | required NaNs | run_dir |", "|---:|---|---:|---:|---|---:|---|"]
    for m in metas:
        lines.append(f"| {m['kappa_f']:.5f} | {m['source']} | {m['sec_per_chain_sweep']:.6g} | {m['elapsed_sec']:.6g} | {m['preflight_status']} | {m['nan_required_ar'] + m['nan_required_obs']} | `{m['run_dir']}` |")
    lines += ["", "## Acceptance", "", "| kappa_f | move_type | substep | acceptance | attempts(saved) | actual attempts estimate |", "|---:|---|---:|---:|---:|---:|"]
    for _, r in acc.iterrows():
        lines.append(f"| {r.kappa_f:.5f} | {r.move_type} | {r.substep} | {r.acceptance_saved_sweeps:.6g} | {int(r.attempts_saved_sweeps)} | {int(r.actual_total_attempts_estimate)} |")
    lines += ["", "## Delta logw", "", "| kappa_f | move_type | substep | mean | std | q05 | q50 | q95 |", "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for _, r in dlog.iterrows():
        if r.move_type in {"coarse_patch", "latent_pCN"}:
            lines.append(f"| {r.kappa_f:.5f} | {r.move_type} | {r.substep} | {r.delta_logw_mean:.6g} | {r.delta_logw_std:.6g} | {r.delta_logw_q05:.6g} | {r.delta_logw_q50:.6g} | {r.delta_logw_q95:.6g} |")
    lines += ["", "## Final Saved Local/Action Observables", "", "| kappa_f | sweep | phi2 | phi4 | NN | 2NN | diag | action_density |", "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for kappa in sorted(obs.kappa_f.unique()):
        sub = obs[obs.kappa_f == kappa].sort_values("sweep")
        r = sub.iloc[-1]
        lines.append(f"| {kappa:.5f} | {int(r.sweep)} | {r.phi2:.6g} | {r.phi4:.6g} | {r.NN:.6g} | {r['2nn']:.6g} | {r.diag:.6g} | {r.action_density:.6g} |")

    coarse = acc[acc.move_type == "coarse_patch"].sort_values("kappa_f")
    latent = acc[acc.move_type == "latent_pCN"].sort_values("kappa_f")
    coarse_d = dlog[dlog.move_type == "coarse_patch"].sort_values("kappa_f")
    best_acc = coarse.iloc[int(np.argmax(coarse.acceptance_saved_sweeps.to_numpy()))]
    best_std = coarse_d.iloc[int(np.argmin(coarse_d.delta_logw_std.to_numpy()))]
    lines += ["", "## Answers", ""]
    lines.append(f"1. Best saved-sweep coarse acceptance is at `kappa_f={best_acc.kappa_f:.5f}` with acceptance `{best_acc.acceptance_saved_sweeps:.6g}`.")
    lines.append(f"2. Smallest coarse `Delta logw` std is at `kappa_f={best_std.kappa_f:.5f}` with std `{best_std.delta_logw_std:.6g}`.")
    lines.append("3. Latent acceptance remains healthy if it stays O(0.5); see table above.")
    lines.append("4. First longer-run candidate should be chosen after comparing both acceptance and local-operator drift; this smoke is not an equilibrium test.")
    lines.append("5. If the higher-kappa targets improve coarse acceptance/std relative to 0.27050, that supports kappa_f mismatch as a plausible contributor to the same-kappa acceptance drop.")
    (out / "L64_TO_L128_KAPPAF_SCAN_P32_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--kappa-f", type=float, nargs="+", default=[0.27075, 0.27100])
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log("starting L64->L128 P32 kappaf scan smoke")
    all_acc: list[dict[str, Any]] = []
    all_dlog: list[dict[str, Any]] = []
    all_obs: list[dict[str, Any]] = []
    metas: list[dict[str, Any]] = []
    for kappa in args.kappa_f:
        run_dir = args.out_dir / tag(kappa)
        log(f"starting kappa_f={kappa:.5f} run_dir={run_dir}")
        ns = SimpleNamespace(
            config=args.config,
            out_dir=run_dir,
            kappa_f=float(kappa),
            Lc=64,
            Lf=128,
            chains=2,
            sweeps=20,
            save_sweeps=[0, 1, 5, 10, 20],
            coarse_patch_size=32,
            latent_patch_size=12,
            coarse_proposal_scale=0.125,
            latent_beta_scale=0.4,
            latent_updates_per_coarse=3,
            batch_logq_across_chains=True,
            seed=20260702,
        )
        run_smoke(ns)
        acc, dlog, obs, meta = collect_run(run_dir, float(kappa), "scan")
        all_acc.extend(acc)
        all_dlog.extend(dlog)
        all_obs.extend(obs)
        metas.append(meta)
        log(f"completed kappa_f={kappa:.5f}")
    if SAME_KAPPA_DIR.exists() and (SAME_KAPPA_DIR / "summary.json").exists():
        acc, dlog, obs, meta = collect_run(SAME_KAPPA_DIR, 0.27050, "existing_same_kappa")
        all_acc.extend(acc)
        all_dlog.extend(dlog)
        all_obs.extend(obs)
        metas.append(meta)
    acc_df = pd.DataFrame(all_acc).sort_values(["kappa_f", "move_type", "substep"])
    dlog_df = pd.DataFrame(all_dlog).sort_values(["kappa_f", "move_type", "substep"])
    obs_df = pd.DataFrame(all_obs).sort_values(["kappa_f", "sweep"])
    acc_df.to_csv(args.out_dir / "kappaf_acceptance_summary.csv", index=False)
    dlog_df.to_csv(args.out_dir / "kappaf_delta_logw_summary.csv", index=False)
    obs_df.to_csv(args.out_dir / "kappaf_local_operator_flow.csv", index=False)
    write_report(args.out_dir, acc_df, dlog_df, obs_df, metas)
    (args.out_dir / "summary.json").write_text(json.dumps({"status": "completed", "metas": metas}, indent=2) + "\n", encoding="utf-8")
    log(f"completed L64->L128 P32 kappaf scan smoke out={args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
