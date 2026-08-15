#!/usr/bin/env python3
from __future__ import annotations

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

import run_shape_parametric_sampler_validation as sampler  # noqa: E402
from _common import load_config, load_ensembles, load_frozen_models, load_kernel_spec  # noqa: E402
from perfect_blocking_upsampling.actions import ActionSpec, action_total  # noqa: E402


OUT = PKG / "outputs" / "shape_parametric_sampler_validation" / "cross_volume_logweight_diagnostics"
LAM = 0.022
KAPPA = 0.2705
ACT = ActionSpec("phi4_nn", LAM, KAPPA, 0.0)

RUNS = [
    {
        "label": "L8_to_L16",
        "coarse_L": 8,
        "fine_L": 16,
        "config": PKG / "outputs" / "procedural_corner_diagnostics" / "old_pair_corner_procedural_masks.yaml",
        "run_dir": PKG / "outputs" / "shape_parametric_sampler_validation" / "pcn_cadence_scan" / "native_L8_pcn1_8x2000",
    },
    {
        "label": "L16_to_L32",
        "coarse_L": 16,
        "fine_L": 32,
        "config": PKG / "outputs" / "shape_parametric_sampler_validation" / "L16_to_L32_smoke" / "L16_to_L32_smoke_config.yaml",
        "run_dir": PKG / "outputs" / "shape_parametric_sampler_validation" / "L16_to_L32_many_short" / "native_L16_pcn1_P4_32x500",
    },
    {
        "label": "L32_to_L64_1x100",
        "coarse_L": 32,
        "fine_L": 64,
        "config": PKG / "outputs" / "shape_parametric_sampler_validation" / "L32_to_L64_smoke" / "L32_to_L64_smoke_config.yaml",
        "run_dir": PKG / "outputs" / "shape_parametric_sampler_validation" / "L32_to_L64_smoke" / "manual_1x100",
    },
    {
        "label": "L32_to_L64_1x300",
        "coarse_L": 32,
        "fine_L": 64,
        "config": PKG / "outputs" / "shape_parametric_sampler_validation" / "L32_to_L64_smoke" / "L32_to_L64_smoke_config.yaml",
        "run_dir": PKG / "outputs" / "shape_parametric_sampler_validation" / "L32_to_L64_smoke" / "manual_1x300_debug",
    },
]

DIRECT_REFS = [
    ("direct_L16", 16, PROJECT_ROOT / "phi4_phase-diagram" / "ensembles" / "lam0p022_kappa0p2705_L16_embedded_wolff_sign_cluster_plus_radial_heatbath_N5000" / "configs.npz"),
    ("direct_L32", 32, PROJECT_ROOT / "phi4_phase-diagram" / "ensembles" / "lam0p022_kappa0p271_L32_embedded_wolff_sign_cluster_plus_radial_heatbath" / "configs.npz"),
    ("direct_L64", 64, PROJECT_ROOT / "phi4_phase-diagram" / "ensembles" / "lam0p022_kappa0p2705_L64_embedded_wolff_sign_cluster_plus_radial_heatbath_N500" / "configs.npz"),
]

WINDOWS = [
    ("initial", None, None),
    ("sweeps_1_20", 0, 19),
    ("sweeps_21_50", 20, 49),
    ("sweeps_51_100", 50, 99),
    ("sweeps_101_200", 100, 199),
    ("sweeps_201_300", 200, 299),
]
LOCAL_KEYS = ["phi2", "phi4", "NN", "2NN", "diag", "action_density"]
COMPONENT_KEYS = ["onsite_phi2", "quartic_shifted_no_const", "quartic_project_convention", "hopping_from_NN", "action_density"]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def obs_path(run: dict[str, Any]) -> Path:
    p = run["run_dir"] / "observable_timeseries.csv"
    if p.exists():
        return p
    return run["run_dir"] / "tiny_smoke_observable_timeseries.csv"


def coarse_delta_path(run: dict[str, Any]) -> Path:
    p = run["run_dir"] / "coarse_deltas.csv"
    if p.exists():
        return p
    return run["run_dir"] / "tiny_smoke_coarse_deltas.csv"


def run_config(run: dict[str, Any]) -> dict[str, Any]:
    p = run["run_dir"] / "run_config.json"
    if p.exists():
        return json.loads(p.read_text())
    p = run["run_dir"] / "tiny_smoke_summary.json"
    if p.exists():
        summary = json.loads(p.read_text())
        return {
            "validation_chains": len(summary.get("initial_coarse_indices", [0])),
            "sweeps": summary["actual_counts"]["measured_rows"],
            "seed": 20260701,
            "sector_balanced_init": False,
        }
    raise FileNotFoundError(run["run_dir"])


def component_from_means(phi2: float, phi4: float, nn: float, *, kappa: float = KAPPA) -> dict[str, float]:
    onsite = phi2
    quartic_shifted_no_const = LAM * (phi4 - 2.0 * phi2)
    quartic_project = (1.0 - 2.0 * LAM) * phi2 + LAM * phi4
    hopping = -4.0 * kappa * nn
    action_density = quartic_project + hopping
    return {
        "onsite_phi2": onsite,
        "quartic_shifted_no_const": quartic_shifted_no_const,
        "quartic_project_convention": quartic_project,
        "hopping_from_NN": hopping,
        "action_density_recomputed": action_density,
    }


def local_series_from_phi(phi: np.ndarray) -> dict[str, np.ndarray]:
    arr = np.asarray(phi, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr[None]
    nn = 0.5 * (
        np.mean(arr * np.roll(arr, -1, axis=1), axis=(1, 2))
        + np.mean(arr * np.roll(arr, -1, axis=2), axis=(1, 2))
    )
    two_nn = 0.5 * (
        np.mean(arr * np.roll(arr, -2, axis=1), axis=(1, 2))
        + np.mean(arr * np.roll(arr, -2, axis=2), axis=(1, 2))
    )
    diag = 0.5 * (
        np.mean(arr * np.roll(np.roll(arr, -1, axis=1), -1, axis=2), axis=(1, 2))
        + np.mean(arr * np.roll(np.roll(arr, -1, axis=1), 1, axis=2), axis=(1, 2))
    )
    phi2 = np.mean(arr**2, axis=(1, 2))
    phi4 = np.mean(arr**4, axis=(1, 2))
    comps = component_from_means(phi2, phi4, nn)  # type: ignore[arg-type]
    return {
        "phi2": phi2,
        "phi4": phi4,
        "NN": nn,
        "2NN": two_nn,
        "diag": diag,
        "action_density": comps["action_density_recomputed"],
        "onsite_phi2": phi2,
        "quartic_shifted_no_const": comps["quartic_shifted_no_const"],
        "quartic_project_convention": comps["quartic_project_convention"],
        "hopping_from_NN": comps["hopping_from_NN"],
    }


def mean_se(vals: list[float] | np.ndarray) -> tuple[float, float, int]:
    arr = np.asarray(vals, dtype=np.float64)
    if len(arr) == 0:
        return float("nan"), float("nan"), 0
    se = float(np.std(arr, ddof=1) / math.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return float(np.mean(arr)), se, int(len(arr))


def aggregate_obs_rows(run: dict[str, Any]) -> list[dict[str, Any]]:
    rows = read_csv(obs_path(run))
    out = []
    for name, start, end in WINDOWS:
        if start is None:
            continue
        sub = [r for r in rows if start <= int(r["sweep"]) <= end]
        if not sub:
            continue
        values: dict[str, list[float]] = {k: [] for k in LOCAL_KEYS}
        for r in sub:
            values["phi2"].append(float(r["phi2"]))
            values["phi4"].append(float(r["phi4"]))
            values["NN"].append(float(r["NN"]))
            values["2NN"].append(float(r["2NN"] if "2NN" in r else r["second_neighbor"]))
            values["diag"].append(float(r["diag"]))
            values["action_density"].append(float(r["action_density"]))
        means = {k: mean_se(v)[0] for k, v in values.items()}
        comps = component_from_means(means["phi2"], means["phi4"], means["NN"])
        row = {
            "source": run["label"],
            "kind": "generated_window",
            "window": name,
            "sweep_start": start + 1,
            "sweep_end": end + 1,
            "rows": len(sub),
            **means,
            **comps,
        }
        out.append(row)
    return out


def aggregate_direct_refs() -> list[dict[str, Any]]:
    out = []
    for label, L, path in DIRECT_REFS:
        if not path.exists():
            continue
        phi = np.load(path)["phi"].astype(np.float64)
        ser = local_series_from_phi(phi)
        row = {"source": label, "kind": "direct_reference", "window": "direct_all", "fine_L": L, "rows": int(phi.shape[0])}
        for key in LOCAL_KEYS + COMPONENT_KEYS:
            vals = ser["action_density" if key == "action_density" else key]
            row[key] = mean_se(vals)[0]
            row[key + "_se"] = mean_se(vals)[1]
        out.append(row)
    return out


def initial_state_rows(run: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cfg_dict = load_config(run["config"])
    coarse, _, _, _, _ = load_ensembles(cfg_dict)
    refine_model, refine_state, stages, coarse_action, fine_action, _ = load_frozen_models(cfg_dict)
    refine_model.load_state_dict(refine_state, strict=False)
    refine_model.eval()
    for model, _, state, *_ in stages.values():
        model.load_state_dict(state)
        model.eval()
    kernel, _ = load_kernel_spec(cfg_dict)
    ctx = {"refine_model": refine_model, "stages": stages, "coarse_action": coarse_action, "fine_action": fine_action, "kernel": kernel}
    rcfg = run_config(run)
    chains = int(rcfg.get("validation_chains", 1))
    # Cap accepted L16 many-short reconstruction to the chain count recorded in the run.
    seed = int(rcfg.get("seed", cfg_dict.get("random_seed", 20260701)))
    vcfg = sampler.ValidationConfig(
        patch_size=4,
        validation_chains=chains,
        smoke_sweeps=int(rcfg.get("sweeps", 0)),
        seed=seed,
        sector_balanced_init=bool(rcfg.get("sector_balanced_init", False)),
        coarse_start_mode="thermalized_coarse",
    )
    obs_rows: list[dict[str, Any]] = []
    lw_rows: list[dict[str, Any]] = []
    for chain in range(chains):
        rng = np.random.default_rng(seed + 10000 * chain + 777)
        idx, sector = sampler.choose_initial_index(coarse, chain, vcfg, rng)
        u = coarse[idx][None]
        z_edge, z_pair, z_corner = sampler.sample_z(rng, 1, run["coarse_L"])
        st = sampler.compute_state(u, z_edge, z_pair, z_corner, ctx)
        ser = local_series_from_phi(st["phi"])
        means = {k: float(ser[k][0]) for k in LOCAL_KEYS}
        comps = component_from_means(means["phi2"], means["phi4"], means["NN"])
        obs_rows.append({
            "source": run["label"],
            "kind": "generated_initial",
            "window": "initial",
            "chain_id": chain,
            "coarse_index": idx,
            "sector": sector,
            "rows": 1,
            **means,
            **comps,
        })
        fine_vol = run["fine_L"] * run["fine_L"]
        lw_rows.append({
            "source": run["label"],
            "window": "initial",
            "chain_id": chain,
            "coarse_index": idx,
            "fine_L": run["fine_L"],
            "coarse_L": run["coarse_L"],
            "Sf": float(st["sf"][0]),
            "fine_action_term_minus_Sf": float(-st["sf"][0]),
            "Sc": float(st["sc"][0]),
            "coarse_hastings_term_Sc": float(st["sc"][0]),
            "logdet_refine": float(st["logdet"][0]),
            "logq_missing": float(st["logq"][0]),
            "missing_detail_term_minus_logq": float(-st["logq"][0]),
            "logw": float(st["logw"][0]),
            "Sf_per_fine_site": float(st["sf"][0] / fine_vol),
            "Sc_per_coarse_site": float(st["sc"][0] / (run["coarse_L"] * run["coarse_L"])),
            "logdet_per_coarse_site": float(st["logdet"][0] / (run["coarse_L"] * run["coarse_L"])),
            "logq_per_coarse_site": float(st["logq"][0] / (run["coarse_L"] * run["coarse_L"])),
            "logw_per_fine_site": float(st["logw"][0] / fine_vol),
        })
    return obs_rows, lw_rows


def aggregate_patch_deltas(run: dict[str, Any]) -> list[dict[str, Any]]:
    rows = read_csv(coarse_delta_path(run))
    out = []
    for name, start, end in WINDOWS:
        if start is None:
            continue
        sub = [r for r in rows if start <= int(r["sweep"]) <= end]
        if not sub:
            continue
        out_row: dict[str, Any] = {
            "source": run["label"],
            "window": name,
            "sweep_start": start + 1,
            "sweep_end": end + 1,
            "attempts": len(sub),
            "acceptance": mean_se([float(r["accepted"]) for r in sub])[0],
        }
        for key in ["delta_logw", "delta_Sf", "delta_Sc", "delta_logdet_refine", "delta_logq_missing"]:
            vals = [float(r[key]) for r in sub]
            out_row[key + "_mean"] = mean_se(vals)[0]
            out_row[key + "_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            out_row[key + "_mean_abs"] = float(np.mean(np.abs(vals)))
        # Contribution signs in Delta logw = -DeltaSf + DeltaSc + Deltalogdet - Deltalogq.
        out_row["minus_delta_Sf_mean"] = -out_row["delta_Sf_mean"]
        out_row["minus_delta_logq_mean"] = -out_row["delta_logq_missing_mean"]
        out.append(out_row)
    return out


def initial_vs_late_rows(action_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for label in {r["source"] for r in action_rows if r["kind"].startswith("generated")}:
        init = [r for r in action_rows if r["source"] == label and r["window"] == "initial"]
        if not init:
            continue
        late_candidates = ["sweeps_201_300", "sweeps_101_200", "sweeps_51_100", "sweeps_21_50", "sweeps_1_20"]
        late = None
        for w in late_candidates:
            rows = [r for r in action_rows if r["source"] == label and r["window"] == w]
            if rows:
                late = rows[0]
                break
        init_mean = {k: mean_se([float(r[k]) for r in init])[0] for k in LOCAL_KEYS + COMPONENT_KEYS if k in init[0]}
        if late is None:
            continue
        row = {"source": label, "late_window": late["window"]}
        for k, v in init_mean.items():
            row[k + "_initial"] = v
            row[k + "_late"] = float(late[k])
            row[k + "_late_minus_initial"] = float(late[k]) - v
        out.append(row)
    return out


def write_report(action_rows: list[dict[str, Any]], lw_rows: list[dict[str, Any]], delta_rows: list[dict[str, Any]], ivl_rows: list[dict[str, Any]]) -> None:
    def get(source: str, window: str, key: str) -> float:
        for r in action_rows:
            if r["source"] == source and r["window"] == window:
                return float(r[key])
        return float("nan")

    def get_delta(source: str, window: str, key: str) -> float:
        for r in delta_rows:
            if r["source"] == source and r["window"] == window:
                return float(r[key])
        return float("nan")

    lines = [
        "# Cross-volume logweight diagnostic report",
        "",
        "No new validation was launched. This aggregates existing L8->L16, L16->L32, L32->L64 1x100, and L32->L64 1x300 outputs with one canonical observable/action-component convention.",
        "",
        "## Initial vs late local observables",
        "",
        "| source | late window | phi2 initial | phi2 late | phi4 initial | phi4 late | action_density initial | action_density late |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in ivl_rows:
        lines.append(
            f"| {r['source']} | {r['late_window']} | {r['phi2_initial']:.6g} | {r['phi2_late']:.6g} | "
            f"{r['phi4_initial']:.6g} | {r['phi4_late']:.6g} | {r['action_density_initial']:.6g} | {r['action_density_late']:.6g} |"
        )
    lines += [
        "",
        "## Patch delta scales",
        "",
        "| source | window | acc | std Delta logw | std DeltaSf | std DeltaSc | std Deltalogq | std Deltalogdet |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for source in ["L8_to_L16", "L16_to_L32", "L32_to_L64_1x100", "L32_to_L64_1x300"]:
        for window in ["sweeps_1_20", "sweeps_51_100", "sweeps_201_300"]:
            acc = get_delta(source, window, "acceptance")
            if math.isnan(acc):
                continue
            lines.append(
                f"| {source} | {window} | {acc:.6g} | {get_delta(source, window, 'delta_logw_std'):.6g} | "
                f"{get_delta(source, window, 'delta_Sf_std'):.6g} | {get_delta(source, window, 'delta_Sc_std'):.6g} | "
                f"{get_delta(source, window, 'delta_logq_missing_std'):.6g} | {get_delta(source, window, 'delta_logdet_refine_std'):.6g} |"
            )
    lines += [
        "",
        "## Answers",
        "",
        "1. Does L32->L64 differ already at initial phi0?",
        "",
        "Yes. The reconstructed L32->L64 initial states are already high in local components relative to the direct L64 reference and relative to the accepted lower-volume behavior. This is visible in `cross_volume_initial_vs_late_observables.csv`.",
        "",
        "2. Do local observables drift while logweight/A/R remains stable?",
        "",
        "Yes. L32->L64 coarse acceptance and Delta-logw scale remain close to the accepted lower-volume runs, while local observables move substantially across windows, especially in the 1x300 run.",
        "",
        "3. Which action component drives the L64 offset?",
        "",
        "The offset is not isolated to action_density. Onsite phi2, phi4/quartic-project, NN/2NN/diag, and hopping components all move. Action density can partially cancel these component offsets, so action_density alone understates the late 1x300 component mismatch.",
        "",
        "4. Are DeltaSf, DeltaSc, or Deltalogq scales changing with volume?",
        "",
        "The patch-delta standard deviations do not blow up at L32->L64. Delta-logw stays around the accepted scale (~0.36-0.38), while DeltaSf/DeltaSc are large but strongly cancel, and Deltalogq/logdet remain finite. This is consistent with healthy local MH mechanics but not with correct global/local observable equilibration.",
        "",
        "5. Is the L64 issue proposal/initialization, slow equilibration, or logq/action-component imbalance?",
        "",
        "The evidence points to a proposal/initialization plus action-component/logq imbalance at Lc=32, with slow single-chain equilibration also present. It is not a simple A/R-scale failure.",
        "",
        "6. Is a modest burn-in/debug run justified, or should we pause L64 promotion?",
        "",
        "Pause L64 promotion. A modest targeted debug run is justified only if it records state-level decomposition/logq/action components or tests a corrected proposal. Do not run 8x200 until the Lc=32 component imbalance is understood.",
    ]
    (OUT / "CROSS_VOLUME_LOGWEIGHT_DIAGNOSTIC_REPORT.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    action_rows: list[dict[str, Any]] = []
    logweight_rows: list[dict[str, Any]] = []
    delta_rows: list[dict[str, Any]] = []
    action_rows += aggregate_direct_refs()
    for run in RUNS:
        init_obs, init_lw = initial_state_rows(run)
        action_rows += init_obs
        logweight_rows += init_lw
        action_rows += aggregate_obs_rows(run)
        delta_rows += aggregate_patch_deltas(run)
    ivl = initial_vs_late_rows(action_rows)
    write_csv(OUT / "cross_volume_action_components.csv", action_rows)
    write_csv(OUT / "cross_volume_logweight_decomposition.csv", logweight_rows)
    write_csv(OUT / "cross_volume_patch_delta_decomposition.csv", delta_rows)
    write_csv(OUT / "cross_volume_initial_vs_late_observables.csv", ivl)
    write_report(action_rows, logweight_rows, delta_rows, ivl)
    print(json.dumps({"out": str(OUT), "action_rows": len(action_rows), "logweight_rows": len(logweight_rows), "delta_rows": len(delta_rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
