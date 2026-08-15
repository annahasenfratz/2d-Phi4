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
sys.path.insert(0, str(PKG / "src"))

from perfect_blocking_upsampling.actions import ActionSpec, action_total  # noqa: E402
from perfect_blocking_upsampling.observables import bootstrap_second_moment_xi, second_moment_components  # noqa: E402

LAM = 0.2
KAPPA = 0.323124
DEFAULT_CONFIGS = PKG / "outputs" / "controlled_patch_lam0p2" / "blocked_native_L64_to_L32_coarse_rand5x5_0084" / "configs.npz"
DEFAULT_OUT = PKG / "outputs" / "controlled_patch_lam0p2" / "coarse_source_overlap_diagnostic" / "kappa_scan_per_config_observables"


def json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(type(obj).__name__)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=json_default, allow_nan=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def upsert_csv_row(path: Path, row: dict[str, Any], key: str, fields: list[str]) -> None:
    rows: list[dict[str, Any]] = []
    if path.exists() and path.stat().st_size > 0:
        with path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    rows = [r for r in rows if r.get(key) != str(row[key])]
    rows.append({field: row.get(field, "") for field in fields})
    write_csv(path, rows, fields)


def load_blocked(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    with np.load(path) as z:
        phi = np.asarray(z["phi"], dtype=np.float32)
        if "source_native_L64_index" in z.files:
            source_idx = np.asarray(z["source_native_L64_index"], dtype=np.int64)
        else:
            source_idx = np.arange(len(phi), dtype=np.int64)
        meta: dict[str, Any] = {}
        for k in z.files:
            if k in {"phi", "source_native_L64_index"}:
                continue
            arr = z[k]
            if arr.shape == ():
                meta[k] = arr.item()
            elif arr.size <= 16:
                meta[k] = arr.tolist()
            else:
                meta[k] = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
    if phi.ndim != 3 or phi.shape[1:] != (32, 32):
        raise ValueError(f"expected blocked L32 configs with shape (N,32,32), got {phi.shape}")
    if len(source_idx) != len(phi):
        raise ValueError(f"source index length {len(source_idx)} does not match configs {len(phi)}")
    return phi, source_idx, meta


def per_config(phi: np.ndarray, action: ActionSpec) -> dict[str, np.ndarray]:
    arr = np.asarray(phi, dtype=np.float64)
    L = arr.shape[1]
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
    total_action = action_total(arr, action).astype(np.float64)
    sm = second_moment_components(arr.astype(np.float32))
    return {
        "m": m,
        "m2": m * m,
        "m4": m**4,
        "phi2": phi2,
        "phi4": phi4,
        "local_kurtosis_ratio": phi4 / np.maximum(phi2 * phi2, 1.0e-300),
        "NN": nn,
        "2nn": twonn,
        "diag": diag,
        "action_density": total_action / float(L * L),
        "total_action": total_action,
        "G_pmin_x_cfg": np.asarray(sm["G_pmin_x_cfg"], dtype=np.float64),
        "G_pmin_y_cfg": np.asarray(sm["G_pmin_y_cfg"], dtype=np.float64),
    }


def second_moment_from_pc(pc: dict[str, np.ndarray], L: int) -> dict[str, float | int]:
    G0 = float(L * L * (np.mean(pc["m2"]) - np.mean(pc["m"]) ** 2))
    Gpx = float(np.mean(pc["G_pmin_x_cfg"]))
    Gpy = float(np.mean(pc["G_pmin_y_cfg"]))
    Gp = 0.5 * (Gpx + Gpy)
    ratio = G0 / Gp if Gp > 0.0 else float("nan")
    sqrt_arg = ratio - 1.0 if np.isfinite(ratio) else float("nan")
    valid = bool(np.isfinite(G0) and np.isfinite(Gp) and G0 > 0.0 and Gp > 0.0 and sqrt_arg > 0.0)
    xi_over_L = float(math.sqrt(sqrt_arg) / (2.0 * L * math.sin(math.pi / L))) if valid else float("nan")
    return {
        "G0_connected": G0,
        "G_pmin_x": Gpx,
        "G_pmin_y": Gpy,
        "G_pmin": Gp,
        "xi_2nd_sqrt_arg": sqrt_arg,
        "xi_over_L": xi_over_L,
        "xi_2nd_valid": int(valid),
        "xi_2nd_rotational_asymmetry": float((Gpx - Gpy) / Gp) if Gp > 0.0 else float("nan"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Write kappa-scan-style observables for blocked native L64->L32 configs.")
    ap.add_argument("--configs", type=Path, default=DEFAULT_CONFIGS)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    phi, source_idx, meta = load_blocked(args.configs)
    n = int(len(phi))
    label = args.label or f"blocked_native_L64_kappa0p323124_N{n}_rand5x5_0084"
    action = ActionSpec("phi4_nn", LAM, KAPPA)
    pc = per_config(phi, action)
    L = int(phi.shape[1])
    volume = L * L
    nonfinite = np.sum(~np.isfinite(phi), axis=(1, 2)).astype(np.int64)
    rows: list[dict[str, Any]] = []
    for i in range(n):
        rows.append(
            {
                "sample": i,
                "source_config_index": int(source_idx[i]),
                "source_native_L64_index": int(source_idx[i]),
                "L": L,
                "volume": volume,
                "lambda": LAM,
                "kappa": KAPPA,
                "m": float(pc["m"][i]),
                "m2": float(pc["m2"][i]),
                "m4": float(pc["m4"][i]),
                "phi2": float(pc["phi2"][i]),
                "phi4": float(pc["phi4"][i]),
                "local_kurtosis_ratio": float(pc["local_kurtosis_ratio"][i]),
                "NN": float(pc["NN"][i]),
                "2nn": float(pc["2nn"][i]),
                "diag": float(pc["diag"][i]),
                "action_density": float(pc["action_density"][i]),
                "action_density_proxy": float(pc["action_density"][i]),
                "total_action": float(pc["total_action"][i]),
                "fine_action_proxy": float(pc["total_action"][i]),
                "G_pmin_x_cfg": float(pc["G_pmin_x_cfg"][i]),
                "G_pmin_y_cfg": float(pc["G_pmin_y_cfg"][i]),
                "nonfinite_count": int(nonfinite[i]),
            }
        )

    fields = [
        "sample",
        "source_config_index",
        "source_native_L64_index",
        "L",
        "volume",
        "lambda",
        "kappa",
        "m",
        "m2",
        "m4",
        "phi2",
        "phi4",
        "local_kurtosis_ratio",
        "NN",
        "2nn",
        "diag",
        "action_density",
        "action_density_proxy",
        "total_action",
        "fine_action_proxy",
        "G_pmin_x_cfg",
        "G_pmin_y_cfg",
        "nonfinite_count",
    ]
    csv_path = args.out_dir / f"{label}_all_observables_per_config.csv"
    write_csv(csv_path, rows, fields)

    means = {k: float(np.mean(v)) for k, v in pc.items()}
    means.update(second_moment_from_pc(pc, L))
    means["Binder_U4_from_averages"] = float(1.0 - means["m4"] / max(3.0 * means["m2"] * means["m2"], 1.0e-300))
    means["local_kurtosis_ratio_from_averages"] = float(means["phi4"] / max(means["phi2"] * means["phi2"], 1.0e-300))
    means["susceptibility_connected"] = float(means["G0_connected"])
    boot = bootstrap_second_moment_xi(phi, n_bootstrap=1000, seed=12345)

    summary = {
        "label": label,
        "csv_path": csv_path,
        "configs": args.configs,
        "configs_metadata": meta,
        "n_configs": n,
        "L": L,
        "lambda": LAM,
        "kappa": KAPPA,
        "blocking_convention": meta.get("blocking", "psi=apply_kernel(phi_L64, rand5x5_0084); blocked=psi[:,0::2,0::2]"),
        "action_density_is_proxy": False,
        "means": {
            "m": means["m"],
            "m2": means["m2"],
            "m4": means["m4"],
            "phi2": means["phi2"],
            "phi4": means["phi4"],
            "NN": means["NN"],
            "2nn": means["2nn"],
            "diag": means["diag"],
            "action_density": means["action_density"],
            "action_density_proxy": means["action_density"],
            "Binder_U4_from_averages": means["Binder_U4_from_averages"],
            "local_kurtosis_ratio_from_averages": means["local_kurtosis_ratio_from_averages"],
            "susceptibility_connected": means["susceptibility_connected"],
            "G0_connected": means["G0_connected"],
            "G_pmin_x": means["G_pmin_x"],
            "G_pmin_y": means["G_pmin_y"],
            "G_pmin": means["G_pmin"],
            "xi_2nd_sqrt_arg": means["xi_2nd_sqrt_arg"],
            "xi_over_L_2nd": means["xi_over_L"],
            "xi_2nd_valid": means["xi_2nd_valid"],
            "xi_2nd_rotational_asymmetry": means["xi_2nd_rotational_asymmetry"],
        },
        "bootstrap": boot,
        "nonfinite_count": int(np.sum(nonfinite)),
    }
    write_json(args.out_dir / f"{label}_all_observables_per_config.summary.json", {"name": label, "output_csv": csv_path, "means": means, "n_configs": n, "blocked_shape": list(phi.shape), "nonfinite_count": int(np.sum(nonfinite))})
    write_json(args.out_dir / f"{label}_all_observables_per_config.second_moment_summary.json", summary)

    summary_fields = [
        "label",
        "n_configs",
        "L",
        "lambda",
        "kappa",
        "xi_over_L_2nd",
        "xi_over_L_2nd_bootstrap_se",
        "G0_connected",
        "G_pmin_x",
        "G_pmin_y",
        "G_pmin",
        "xi_2nd_valid",
        "xi_2nd_rotational_asymmetry",
        "csv_path",
    ]
    upsert_csv_row(
        args.out_dir / "second_moment_observable_summary.csv",
        {
            "label": label,
            "n_configs": n,
            "L": L,
            "lambda": LAM,
            "kappa": KAPPA,
            "xi_over_L_2nd": means["xi_over_L"],
            "xi_over_L_2nd_bootstrap_se": boot["xi_over_L_2nd_bootstrap_se"],
            "G0_connected": means["G0_connected"],
            "G_pmin_x": means["G_pmin_x"],
            "G_pmin_y": means["G_pmin_y"],
            "G_pmin": means["G_pmin"],
            "xi_2nd_valid": means["xi_2nd_valid"],
            "xi_2nd_rotational_asymmetry": means["xi_2nd_rotational_asymmetry"],
            "csv_path": str(csv_path),
        },
        "label",
        summary_fields,
    )
    print(f"Wrote {csv_path}")
    print(f"Wrote {args.out_dir / (label + '_all_observables_per_config.second_moment_summary.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
