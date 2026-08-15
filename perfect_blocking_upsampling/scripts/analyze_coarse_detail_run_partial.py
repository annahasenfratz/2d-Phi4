#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


OBS_KEYS = [
    "Binder_U4_from_averages",
    "Binder_U4",
    "xi_over_L",
    "chi",
    "susceptibility_connected",
    "m2",
    "m4",
    "phi2",
    "phi4",
    "NN",
    "2nn",
    "diag",
    "action_density",
    "local_kurtosis_ratio_from_averages",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def finite_float(row: dict[str, str], key: str, default: float = float("nan")) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def grouped_by_sweep(rows: list[dict[str, str]]) -> dict[int, list[dict[str, str]]]:
    out: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        try:
            sweep = int(float(row["sweep"]))
        except (KeyError, ValueError):
            continue
        out.setdefault(sweep, []).append(row)
    return out


def aggregate_measurements(rows: list[dict[str, str]], idx: np.ndarray | None = None) -> dict[str, float]:
    if idx is None:
        selected = rows
    else:
        selected = [rows[int(i)] for i in idx]
    if not selected:
        return {k: float("nan") for k in OBS_KEYS}
    L = int(round(finite_float(selected[0], "L")))
    V = float(L * L)

    def mean(key: str) -> float:
        vals = np.asarray([finite_float(r, key) for r in selected], dtype=np.float64)
        return float(np.nanmean(vals))

    m = mean("m")
    m2 = mean("m2")
    m4 = mean("m4")
    phi2 = mean("phi2")
    phi4 = mean("phi4")
    Gpx = mean("G_pmin_x_cfg")
    Gpy = mean("G_pmin_y_cfg")
    Gp = 0.5 * (Gpx + Gpy)
    chi = V * max(m2 - m * m, 0.0)
    sqrt_arg = chi / Gp - 1.0 if Gp > 0.0 else float("nan")
    xi = (
        (1.0 / (2.0 * L * math.sin(math.pi / L))) * math.sqrt(sqrt_arg)
        if chi > 0.0 and Gp > 0.0 and sqrt_arg > 0.0
        else float("nan")
    )
    binder = 1.0 - m4 / max(3.0 * m2 * m2, 1.0e-300)
    return {
        "Binder_U4_from_averages": float(binder),
        "Binder_U4": float(binder),
        "xi_over_L": float(xi),
        "chi": float(chi),
        "susceptibility_connected": float(chi),
        "m": float(m),
        "m2": float(m2),
        "m4": float(m4),
        "phi2": float(phi2),
        "phi4": float(phi4),
        "NN": mean("NN"),
        "2nn": mean("2nn"),
        "diag": mean("diag"),
        "action_density": mean("action_density"),
        "local_kurtosis_ratio_from_averages": float(phi4 / max(phi2 * phi2, 1.0e-300)),
        "G0_connected": float(chi),
        "G_pmin_x": float(Gpx),
        "G_pmin_y": float(Gpy),
        "G_pmin": float(Gp),
        "xi_2nd_sqrt_arg": float(sqrt_arg),
        "xi_2nd_rotational_asymmetry": float(abs(Gpx - Gpy) / max(Gp, 1.0e-300)),
    }


def bootstrap_errors(rows: list[dict[str, str]], nboot: int, seed: int, bin_size: int = 1) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(rows)
    if bin_size < 1:
        raise ValueError("bin_size must be positive")
    n_bins = n // bin_size
    n_used = n_bins * bin_size
    if n_bins <= 1 or nboot <= 0:
        return {k: float("nan") for k in OBS_KEYS + ["G0_connected", "G_pmin_x", "G_pmin_y", "G_pmin", "xi_2nd_sqrt_arg", "xi_2nd_rotational_asymmetry"]}
    rows = rows[:n_used]
    keys = OBS_KEYS + ["G0_connected", "G_pmin_x", "G_pmin_y", "G_pmin", "xi_2nd_sqrt_arg", "xi_2nd_rotational_asymmetry"]
    vals = {k: np.empty(nboot, dtype=np.float64) for k in keys}
    invalid_xi = 0
    for b in range(nboot):
        bins = rng.integers(0, n_bins, size=n_bins)
        idx = (bins[:, None] * bin_size + np.arange(bin_size)[None, :]).reshape(-1)
        agg = aggregate_measurements(rows, idx)
        if not np.isfinite(agg["xi_over_L"]):
            invalid_xi += 1
        for k in keys:
            vals[k][b] = agg[k]
    out = {k: float(np.nanstd(v, ddof=1)) for k, v in vals.items()}
    out["xi_over_L_invalid_bootstrap_fraction"] = invalid_xi / nboot
    return out


def summarize_acceptance(rows: list[dict[str, str]]) -> dict[str, float]:
    def sum_key(key: str) -> float:
        return float(np.nansum([finite_float(r, key, 0.0) for r in rows]))

    coarse_props = sum_key("coarse_proposals")
    coarse_accs = sum_key("coarse_accepts")
    detail_props = sum_key("detail_proposals")
    detail_accs = sum_key("detail_accepts")
    reb_max = np.asarray([finite_float(r, "reblocking_max_error") for r in rows], dtype=np.float64)
    reb_rms = np.asarray([finite_float(r, "reblocking_rms_error") for r in rows], dtype=np.float64)
    return {
        "coarse_proposals": coarse_props,
        "coarse_accepts": coarse_accs,
        "coarse_acceptance": coarse_accs / coarse_props if coarse_props > 0 else float("nan"),
        "detail_proposals": detail_props,
        "detail_accepts": detail_accs,
        "detail_acceptance": detail_accs / detail_props if detail_props > 0 else float("nan"),
        "reblocking_max_error_max": float(np.nanmax(reb_max)) if np.isfinite(reb_max).any() else float("nan"),
        "reblocking_rms_error_mean": float(np.nanmean(reb_rms)) if np.isfinite(reb_rms).any() else float("nan"),
        "nonfinite_count_sum": sum_key("nonfinite_count"),
    }


def find_file(run_dir: Path, name: str) -> Path:
    candidates = [run_dir / "observables" / name, run_dir / name]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find {name} under {run_dir} or {run_dir / 'observables'}")


def find_optional_file(run_dir: Path, name: str) -> Path | None:
    for path in (run_dir / "observables" / name, run_dir / name):
        if path.exists():
            return path
    return None


def is_acceptance_schema(rows: list[dict[str, str]]) -> bool:
    """MIT runs duplicate measurements in per_sweep_observables.csv.

    Established patch runs store per-chain acceptance fields there, whereas the
    MIT drivers keep their aggregate A/R data in acceptance_history.csv.
    """
    if not rows:
        return False
    fields = set(rows[0])
    return bool({"coarse_proposals", "detail_proposals", "proposed_detail_updates"} & fields)


def summarize_substeps(rows: list[dict[str, str]], sweep: int) -> list[dict[str, Any]]:
    selected = [r for r in rows if int(finite_float(r, "sweep", -1)) == sweep]
    groups: dict[str, list[dict[str, str]]] = {}
    for row in selected:
        groups.setdefault(row.get("substep", "unknown"), []).append(row)
    out: list[dict[str, Any]] = []
    for substep, group in sorted(groups.items()):
        attempts = float(np.nansum([finite_float(r, "attempts", 0.0) for r in group]))
        accepts = float(np.nansum([finite_float(r, "accepts", 0.0) for r in group]))
        out.append({
            "sweep": sweep,
            "substep": substep,
            "attempts": attempts,
            "accepts": accepts,
            "acceptance": accepts / attempts if attempts else float("nan"),
        })
    return out


def fmt_pair(mean: float, err: float) -> str:
    return f"({mean:.8f}, {err:.8f})"


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze a running/partial coarse+detail flow-detail run.")
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--label", default=None)
    ap.add_argument("--sweep", default="latest", help="'latest', 'all', or an integer sweep.")
    ap.add_argument("--nboot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--bin-size", type=int, default=1, help="Consecutive rows per bootstrap bin within each selected sweep.")
    args = ap.parse_args()

    run_dir = args.run_dir
    main_path = find_file(run_dir, "main_per_sweep_measurements.csv")
    per_sweep_path = find_optional_file(run_dir, "per_sweep_observables.csv")
    per_sweep_rows = read_csv(per_sweep_path) if per_sweep_path else []
    acceptance_history_path = find_optional_file(run_dir, "acceptance_history.csv")
    if is_acceptance_schema(per_sweep_rows):
        acc_path = per_sweep_path
        acc = per_sweep_rows
    elif acceptance_history_path is not None:
        # MIT-style runs: per_sweep_observables is intentionally a copy of the
        # measurement table, so use their one-row-per-sweep A/R history.
        acc_path = acceptance_history_path
        acc = read_csv(acc_path)
    else:
        # Combined measurement directories deliberately contain only observable
        # rows.  They remain useful for distribution and binned-error analysis.
        acc_path = None
        acc = []
    meas = read_csv(main_path)
    meas_by = grouped_by_sweep(meas)
    acc_by = grouped_by_sweep(acc)
    # Sweep zero has no attempted move in MIT runs and is still a useful
    # initializer measurement.  Missing acceptance data is represented by NaN.
    sweeps = sorted(meas_by)
    if not sweeps:
        raise RuntimeError("No completed sweeps found in measurement file.")

    if args.sweep == "latest":
        selected_sweeps = [sweeps[-1]]
    elif args.sweep == "all":
        selected_sweeps = sweeps
    else:
        selected_sweeps = [int(args.sweep)]
        if selected_sweeps[0] not in meas_by:
            raise RuntimeError(f"Sweep {selected_sweeps[0]} is not present. Available range: {sweeps[0]}..{sweeps[-1]}")

    out_dir = run_dir / "partial_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    label = args.label or run_dir.name

    obs_rows: list[dict[str, Any]] = []
    acc_rows: list[dict[str, Any]] = []
    substep_rows: list[dict[str, Any]] = []
    selected_payload: dict[str, Any] | None = None
    for sweep in sweeps:
        a = summarize_acceptance(acc_by.get(sweep, []))
        acc_rows.append({"sweep": sweep, **a})
    for sweep in selected_sweeps:
        rows_all = meas_by[sweep]
        if args.bin_size < 1:
            raise RuntimeError("--bin-size must be positive")
        n_bins = len(rows_all) // args.bin_size
        n_used = n_bins * args.bin_size
        if n_bins < 2:
            raise RuntimeError(f"--bin-size={args.bin_size} leaves fewer than two bootstrap bins at sweep {sweep}")
        rows = rows_all[:n_used]
        agg = aggregate_measurements(rows)
        err = bootstrap_errors(rows, args.nboot, args.seed + sweep, args.bin_size)
        acc_summary = summarize_acceptance(acc_by.get(sweep, []))
        for key in OBS_KEYS:
            obs_rows.append({"sweep": sweep, "observable": key, "mean": agg[key], "error": err.get(key, float("nan"))})
        if len(selected_sweeps) == 1:
            selected_payload = {
                "sweep": sweep, "observables": agg, "errors": err, "acceptance": acc_summary,
                "bootstrap_binning": {
                    "bin_size": args.bin_size, "raw_rows": len(rows_all), "used_rows": n_used,
                    "discarded_rows": len(rows_all) - n_used, "n_bins": n_bins,
                    "scope": "consecutive measurement rows within this sweep",
                },
            }

    substep_path = run_dir / "debug" / "substep_acceptance_history.csv"
    if substep_path.exists():
        substeps = read_csv(substep_path)
        for sweep in sweeps:
            substep_rows.extend(summarize_substeps(substeps, sweep))
    coarse_kernel_path = run_dir / "debug" / "coarse_kernel_acceptance_history.csv"
    if coarse_kernel_path.exists():
        kernel_rows = read_csv(coarse_kernel_path)
        for sweep in sweeps:
            selected = [r for r in kernel_rows if int(finite_float(r, "sweep", -1)) == sweep]
            attempts = float(np.nansum([finite_float(r, "attempts", 0.0) for r in selected]))
            accepts = float(np.nansum([finite_float(r, "accepts", 0.0) for r in selected]))
            if selected:
                substep_rows.append({
                    "sweep": sweep,
                    "substep": "coarse_kernel_inner",
                    "attempts": attempts,
                    "accepts": accepts,
                    "acceptance": accepts / attempts if attempts else float("nan"),
                })

    write_csv(out_dir / "observable_summary_by_sweep.csv", obs_rows)
    write_csv(out_dir / "acceptance_summary_by_sweep.csv", acc_rows)
    if substep_rows:
        write_csv(out_dir / "mit_substep_acceptance_by_sweep.csv", substep_rows)

    manifest_paths = [run_dir / "run_manifest.json", run_dir / "manifests" / "run_manifest.json"]
    manifest = {}
    for path in manifest_paths:
        if path.exists():
            manifest = json.loads(path.read_text())
            break

    summary = {
        "run_dir": str(run_dir),
        "main_measurements": str(main_path),
        "acceptance_file": str(acc_path),
        "acceptance_source": "per_sweep_observables" if acc_path == per_sweep_path else ("acceptance_history" if acc_path is not None else "unavailable"),
        "mit_substep_acceptance_file": str(substep_path) if substep_path.exists() else None,
        "mit_coarse_kernel_acceptance_file": str(coarse_kernel_path) if coarse_kernel_path.exists() else None,
        "available_sweeps": {"first": sweeps[0], "last": sweeps[-1], "count": len(sweeps)},
        "selected_sweeps": selected_sweeps,
        "nboot": args.nboot,
        "seed": args.seed,
        "bin_size": args.bin_size,
        "binning_scope": "consecutive measurement rows within each selected sweep",
        "manifest_subset": {
            k: manifest.get(k)
            for k in [
                "lambda",
                "kappa_coarse",
                "kappa_fine",
                "chains",
                "update_mode",
                "coarse_acceptance_mode",
                "coarse_patch_size",
                "detail_patch_size",
                "coarse_step_size",
                "fine_proposal_sigma",
            ]
            if k in manifest
        },
        "selected": selected_payload,
    }
    (out_dir / "partial_analysis_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    if selected_payload is not None:
        sweep = selected_payload["sweep"]
        agg = selected_payload["observables"]
        err = selected_payload["errors"]
        acceptance = selected_payload["acceptance"]
        lines = [f'    "{label}_sweep{sweep}": {{']
        for key in OBS_KEYS:
            lines.append(f'        "{key}": {fmt_pair(agg[key], err.get(key, float("nan")))},')
        lines += [
            f'        "coarse_acceptance": {fmt_pair(acceptance["coarse_acceptance"], float("nan"))},',
            f'        "detail_acceptance": {fmt_pair(acceptance["detail_acceptance"], float("nan"))},',
            f'        "coarse_proposals": ({acceptance["coarse_proposals"]:.0f}, 0.0),',
            f'        "detail_proposals": ({acceptance["detail_proposals"]:.0f}, 0.0),',
            f'        "reblocking_max_error_max": ({acceptance["reblocking_max_error_max"]:.8e}, 0.0),',
            "    }",
        ]
        pyfrag = "\n".join(lines) + "\n"
        (out_dir / f"{label}_sweep{sweep}_summary_with_acceptance.pyfrag").write_text(pyfrag)
        md = [
            f"# Partial coarse+detail analysis: {label}",
            "",
            f"- run dir: `{run_dir}`",
            f"- latest available sweep: `{sweeps[-1]}`",
            f"- selected sweep: `{sweep}`",
            f"- rows at selected sweep: `{len(meas_by[sweep])}`",
            f"- coarse acceptance at selected sweep: `{acceptance['coarse_acceptance']:.8g}` ({acceptance['coarse_accepts']:.0f}/{acceptance['coarse_proposals']:.0f})",
            f"- detail acceptance at selected sweep: `{acceptance['detail_acceptance']:.8g}` ({acceptance['detail_accepts']:.0f}/{acceptance['detail_proposals']:.0f})",
            f"- max reblocking error at selected sweep: `{acceptance['reblocking_max_error_max']:.8e}`",
            "",
            "```python",
            pyfrag.rstrip(),
            "```",
            "",
        ]
        (out_dir / f"{label}_sweep{sweep}_partial_report.md").write_text("\n".join(md))
        print(pyfrag, end="")
        print(f"\nWrote partial analysis to {out_dir}")
    else:
        print(f"Wrote summaries for {len(selected_sweeps)} sweeps to {out_dir}")


if __name__ == "__main__":
    main()
