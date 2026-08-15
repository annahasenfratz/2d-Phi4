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
sys.path.insert(0, str(PKG / "src"))

from perfect_blocking_upsampling.kernels import apply_kernel, inverse_kernel, load_kernel  # noqa: E402

LAM = 0.2
KAPPA = 0.323124
OUT = PKG / "outputs" / "controlled_patch_lam0p2" / "rand5x5_0084_residual_flow_pilot"
RAND_KERNEL = PKG / "outputs" / "controlled_patch_lam0p2" / "tail_aware_kernel_search_L16to32" / "rand5x5_0084_kernel.json"
SMALL3_KERNEL = PKG / "configs" / "kernel_small3_eta0p25.json"
FIVEX5_KERNEL = PKG / "outputs" / "kernel_conditioning_scan" / "candidate_kernels" / "lam0p2_candidate_5x5_114.json"


def json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(type(obj).__name__)


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def per_config(phi: np.ndarray) -> dict[str, np.ndarray]:
    arr = np.asarray(phi, dtype=np.float64)
    m = np.mean(arr, axis=(1, 2))
    phi2 = np.mean(arr**2, axis=(1, 2))
    phi4 = np.mean(arr**4, axis=(1, 2))
    nn = 0.5 * (
        np.mean(arr * np.roll(arr, -1, axis=1), axis=(1, 2))
        + np.mean(arr * np.roll(arr, -1, axis=2), axis=(1, 2))
    )
    twonn = 0.5 * (
        np.mean(arr * np.roll(arr, -2, axis=1), axis=(1, 2))
        + np.mean(arr * np.roll(arr, -2, axis=2), axis=(1, 2))
    )
    diag = np.mean(arr * np.roll(np.roll(arr, -1, axis=1), -1, axis=2), axis=(1, 2))
    return {
        "m": m,
        "m2": m * m,
        "m4": m**4,
        "phi2": phi2,
        "phi4": phi4,
        "NN": nn,
        "2nn": twonn,
        "diag": diag,
        "action_density": (1.0 - 2.0 * LAM) * phi2 + LAM * phi4 - 4.0 * KAPPA * nn,
    }


def ensemble(pc: dict[str, np.ndarray]) -> dict[str, float]:
    out = {k: float(np.mean(v)) for k, v in pc.items() if k != "m"}
    out["Binder_U4"] = float(1.0 - out["m4"] / max(3.0 * out["m2"] * out["m2"], 1.0e-300))
    out["xi_over_L"] = float(math.sqrt(max(out["m2"], 0.0) / max(out["phi2"], 1.0e-300)))
    return out


def smooth_occupancy(gen: dict[str, np.ndarray], ref: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    cuts_ref = {
        "low_action": ref["action_density"] <= np.quantile(ref["action_density"], 0.25),
        "high_NN": ref["NN"] >= np.quantile(ref["NN"], 0.75),
        "high_2nn": ref["2nn"] >= np.quantile(ref["2nn"], 0.75),
        "high_diag": ref["diag"] >= np.quantile(ref["diag"], 0.75),
    }
    cuts_gen = {
        "low_action": gen["action_density"] <= np.quantile(ref["action_density"], 0.25),
        "high_NN": gen["NN"] >= np.quantile(ref["NN"], 0.75),
        "high_2nn": gen["2nn"] >= np.quantile(ref["2nn"], 0.75),
        "high_diag": gen["diag"] >= np.quantile(ref["diag"], 0.75),
    }
    specs = {
        "low_action": ("low_action",),
        "low_action_and_high_NN": ("low_action", "high_NN"),
        "low_action_and_high_2nn": ("low_action", "high_2nn"),
        "low_action_and_high_diag": ("low_action", "high_diag"),
    }
    rows = []
    for sector, parts in specs.items():
        rm = np.ones(len(ref["action_density"]), dtype=bool)
        gm = np.ones(len(gen["action_density"]), dtype=bool)
        for p in parts:
            rm &= cuts_ref[p]
            gm &= cuts_gen[p]
        ro = float(np.mean(rm))
        go = float(np.mean(gm))
        rows.append({"sector": sector, "native_occupancy": ro, "generated_occupancy": go, "generated_over_native_ratio": go / max(ro, 1.0e-300)})
    return rows


def load_secondary_row(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as f:
        return next(csv.DictReader(f))


def load_run(label: str, root: Path, variant: str, kernel_path: Path, summary_path: Path, samples_path: Path, secondary_csv: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    kernel, _ = load_kernel(kernel_path)
    with np.load(samples_path) as z:
        psi_n = z["psi_native"].astype(np.float32)
        psi_g = z["psi_generated"].astype(np.float32)
    phi_n, inv_n = inverse_kernel(psi_n, kernel)
    phi_g, inv_g = inverse_kernel(psi_g, kernel)
    roundtrip_native = float(np.max(np.abs(apply_kernel(phi_n, kernel).astype(np.float64) - psi_n.astype(np.float64))))
    roundtrip_generated = float(np.max(np.abs(apply_kernel(phi_g, kernel).astype(np.float64) - psi_g.astype(np.float64))))
    pc_n = per_config(phi_n)
    pc_g = per_config(phi_g)
    ens_n = ensemble(pc_n)
    ens_g = ensemble(pc_g)
    summary = read_json(summary_path)
    sec = load_secondary_row(secondary_csv)
    rows = smooth_occupancy(pc_g, pc_n)
    for r in rows:
        r["label"] = label
        r["variant"] = variant
    obs_rows = []
    for obs in ["Binder_U4", "xi_over_L", "m2", "m4", "phi2", "phi4", "NN", "2nn", "diag", "action_density"]:
        obs_rows.append({"label": label, "variant": variant, "observable": obs, "native_mean": ens_n[obs], "generated_mean": ens_g[obs], "delta": ens_g[obs] - ens_n[obs]})
    nonfinite = int(np.sum(~np.isfinite(psi_g)) + np.sum(~np.isfinite(phi_g)))
    main = {
        "label": label,
        "variant": variant,
        "validation_nll": summary.get("validation_residual_nll", summary.get("validation_nll_detail")),
        "validation_nll_whitened": summary.get("validation_nll_whitened"),
        "psi_max_z": summary.get("psi_max_primary_z"),
        "secondary_phi_max_z": summary.get("secondary_phi_max_primary_z", sec.get("max_primary_z_phi_space")),
        "DeltaS_std": summary.get("secondary_DeltaS_std", sec.get("DeltaS_std")),
        "Binder_U4_generated": ens_g["Binder_U4"],
        "xi_over_L_generated": ens_g["xi_over_L"],
        "phi2_generated": ens_g["phi2"],
        "phi4_generated": ens_g["phi4"],
        "NN_generated": ens_g["NN"],
        "2nn_generated": ens_g["2nn"],
        "diag_generated": ens_g["diag"],
        "action_density_generated": ens_g["action_density"],
        "K_roundtrip_native_max_error": roundtrip_native,
        "K_roundtrip_generated_max_error": roundtrip_generated,
        "inverse_ifft_imag_native_max": inv_n["max_inverse_ifft_imag"],
        "inverse_ifft_imag_generated_max": inv_g["max_inverse_ifft_imag"],
        "inverse_condition": inv_g["condition_number_abs"],
        "nonfinite_count": summary.get("nonfinite_count", nonfinite),
        "top_operator_zscores": summary.get("top_operator_zscores", ""),
        "root": str(root),
    }
    return main, rows, obs_rows


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    runs = [
        (
            "current_small3_op_alpha0p01",
            PKG / "outputs" / "lam0p2_kappa0p323124/remediation/rg_preblocked_level_flow_L16/current_small3_residual_refinement/op_alpha0p01",
            "op_alpha0p01",
            SMALL3_KERNEL,
            PKG / "outputs" / "lam0p2_kappa0p323124/remediation/rg_preblocked_level_flow_L16/current_small3_residual_refinement/op_alpha0p01/summary.json",
            PKG / "outputs" / "lam0p2_kappa0p323124/remediation/rg_preblocked_level_flow_L16/current_small3_residual_refinement/op_alpha0p01/validation_samples.npz",
            PKG / "outputs" / "lam0p2_kappa0p323124/remediation/rg_preblocked_level_flow_L16/current_small3_residual_refinement/op_alpha0p01/secondary_Kinv_op_alpha0p01_operator_comparison.csv",
        ),
        (
            "old_5x5_114_residual",
            PKG / "outputs" / "lam0p2_kappa0p323124/remediation/rg_preblocked_level_flow_L16/safe_kernel_residual_pilots/lam0p2_candidate_5x5_114",
            "nll_only",
            FIVEX5_KERNEL,
            PKG / "outputs" / "lam0p2_kappa0p323124/remediation/rg_preblocked_level_flow_L16/safe_kernel_residual_pilots/lam0p2_candidate_5x5_114/residual_flow_training_summary.json",
            PKG / "outputs" / "lam0p2_kappa0p323124/remediation/rg_preblocked_level_flow_L16/safe_kernel_residual_pilots/lam0p2_candidate_5x5_114/residual_flow_validation_samples.npz",
            PKG / "outputs" / "lam0p2_kappa0p323124/remediation/rg_preblocked_level_flow_L16/safe_kernel_residual_pilots/lam0p2_candidate_5x5_114/secondary_Kinv_residual_flow_operator_comparison.csv",
        ),
        (
            "rand5x5_0084_nll_only",
            OUT,
            "nll_only",
            RAND_KERNEL,
            OUT / "residual_flow_training_summary.json",
            OUT / "residual_flow_validation_samples.npz",
            OUT / "secondary_Kinv_residual_flow_operator_comparison.csv",
        ),
        (
            "rand5x5_0084_op_alpha0p01",
            OUT / "refinement/op_alpha0p01",
            "op_alpha0p01",
            RAND_KERNEL,
            OUT / "refinement/op_alpha0p01/summary.json",
            OUT / "refinement/op_alpha0p01/validation_samples.npz",
            OUT / "refinement/op_alpha0p01/secondary_Kinv_op_alpha0p01_operator_comparison.csv",
        ),
    ]
    summary_rows = []
    occ_rows = []
    obs_rows = []
    for args in runs:
        main_row, occ, obs = load_run(*args)
        summary_rows.append(main_row)
        occ_rows.extend(occ)
        obs_rows.extend(obs)
    write_csv(OUT / "rand5x5_0084_residual_flow_pilot_summary.csv", summary_rows)
    write_csv(OUT / "rand5x5_0084_flow_init_fine_observables.csv", obs_rows)
    write_csv(OUT / "rand5x5_0084_smooth_sector_occupancy_after_flow_init.csv", occ_rows)
    detail_history = [
        {
            "kernel": "rand5x5_0084",
            "status": "not_run",
            "reason": "Skipped because flow-init phi-space/action diagnostics already fail the promotion gate; launching a new detail-only Markov diagnostic was not cheap enough to justify without explicit confirmation.",
            "requested_sweeps": "0,10,50,100,300",
        }
    ]
    write_csv(OUT / "rand5x5_0084_detail_only_correction_history.csv", detail_history)
    payload = {
        "summary": summary_rows,
        "smooth_sector_occupancy_after_flow_init": occ_rows,
        "detail_only_correction_history": detail_history,
        "decision": "do_not_promote",
        "reason": "rand5x5_0084 improves static coarse overlap but residual-flow phi-space action_density/phi4 diagnostics are substantially worse than current_small3 op_alpha0p01.",
    }
    write_json(OUT / "rand5x5_0084_residual_flow_pilot_summary.json", payload)
    by = {r["label"]: r for r in summary_rows}
    lines = [
        "# rand5x5_0084 residual-flow pilot",
        "",
        "No production chain was launched. A short conditional Gaussian residual-flow pilot and the same light refinement scan used for the current-small3 branch were run.",
        "",
        "## Summary",
        "",
        "| run | val NLL | psi max |z| | phi max |z| | DeltaS std | Binder | xi/L | phi4 | action density | nonfinite |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in ["current_small3_op_alpha0p01", "old_5x5_114_residual", "rand5x5_0084_nll_only", "rand5x5_0084_op_alpha0p01"]:
        r = by[key]
        lines.append(
            f"| {key} | {float(r['validation_nll']):.6g} | {float(r['psi_max_z']):.6g} | {float(r['secondary_phi_max_z']):.6g} | {float(r['DeltaS_std']):.6g} | {float(r['Binder_U4_generated']):.6g} | {float(r['xi_over_L_generated']):.6g} | {float(r['phi4_generated']):.6g} | {float(r['action_density_generated']):.6g} | {r['nonfinite_count']} |"
        )
    lines += [
        "",
        "## Smooth-sector occupancy after flow init",
        "",
        "| run | low_action | low+NN | low+2nn | low+diag |",
        "|---|---:|---:|---:|---:|",
    ]
    for key in ["current_small3_op_alpha0p01", "old_5x5_114_residual", "rand5x5_0084_nll_only", "rand5x5_0084_op_alpha0p01"]:
        rows = {r["sector"]: r for r in occ_rows if r["label"] == key}
        lines.append(
            f"| {key} | {rows['low_action']['generated_over_native_ratio']:.6g} | {rows['low_action_and_high_NN']['generated_over_native_ratio']:.6g} | {rows['low_action_and_high_2nn']['generated_over_native_ratio']:.6g} | {rows['low_action_and_high_diag']['generated_over_native_ratio']:.6g} |"
        )
    lines += [
        "",
        "## Decision",
        "",
        "Do not promote `rand5x5_0084` from this pilot. It improves static coarse-overlap/smooth-sector support, but after residual-flow initialization its secondary phi-space max |z| and action-density mismatch are worse than current_small3/op_alpha0p01.",
        "",
        "Detail-only correction history was not run: the flow-init diagnostics already fail the promotion gate, and a new 300-sweep Markov diagnostic would be an additional chain rather than a cheap postprocess.",
        "",
        "## Files",
        "",
        "- `rand5x5_0084_residual_flow_pilot_summary.csv`",
        "- `rand5x5_0084_residual_flow_pilot_summary.json`",
        "- `rand5x5_0084_flow_init_fine_observables.csv`",
        "- `rand5x5_0084_smooth_sector_occupancy_after_flow_init.csv`",
        "- `rand5x5_0084_detail_only_correction_history.csv`",
    ]
    (OUT / "rand5x5_0084_residual_flow_pilot_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
