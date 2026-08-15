#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PKG = PROJECT_ROOT / "perfect_blocking_upsampling"
sys.path.insert(0, str(PKG / "src"))

from perfect_blocking_upsampling.actions import ActionSpec, action_total  # noqa: E402


def json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(type(obj).__name__)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ensemble_dir", type=Path)
    ap.add_argument("--lambda", dest="lam", type=float, default=None)
    ap.add_argument("--kappa", type=float, default=None)
    ap.add_argument("--sweep", type=int, default=0)
    args = ap.parse_args()

    configs_path = args.ensemble_dir / "configs.npz" if args.ensemble_dir.is_dir() else args.ensemble_dir
    ensemble_dir = configs_path.parent
    with np.load(configs_path) as z:
        phi = np.asarray(z["phi"], dtype=np.float32)
        lam = float(args.lam if args.lam is not None else z["lambda"])
        kappa = float(args.kappa if args.kappa is not None else z["kappa"])
        meta = {k: z[k] for k in z.files if k != "phi"}

    n, L, _ = phi.shape
    volume = L * L
    arr = phi.astype(np.float64)
    action = ActionSpec("phi4_nn", lam, kappa)
    m = np.mean(arr, axis=(1, 2))
    phi2 = np.mean(arr**2, axis=(1, 2))
    phi4 = np.mean(arr**4, axis=(1, 2))
    nn = 0.5 * (
        np.mean(arr * np.roll(arr, -1, axis=1), axis=(1, 2))
        + np.mean(arr * np.roll(arr, -1, axis=2), axis=(1, 2))
    )
    diag = np.mean(arr * np.roll(np.roll(arr, -1, axis=1), -1, axis=2), axis=(1, 2))
    twonn = 0.5 * (
        np.mean(arr * np.roll(arr, -2, axis=1), axis=(1, 2))
        + np.mean(arr * np.roll(arr, -2, axis=2), axis=(1, 2))
    )
    phase = np.exp(2j * np.pi * np.arange(L) / L)
    phi_x = np.tensordot(arr, phase, axes=([1], [0])).sum(axis=1)
    phi_y = np.tensordot(arr, phase, axes=([2], [0])).sum(axis=1)
    gpx = np.abs(phi_x) ** 2 / float(volume)
    gpy = np.abs(phi_y) ** 2 / float(volume)
    total_action = action_total(phi, action).astype(np.float64)
    action_density = total_action / float(volume)
    nonfinite = (~np.isfinite(arr.reshape(n, -1)).all(axis=1)).astype(np.int64)

    fields = [
        "chain_id",
        "sweep",
        "source_config_index",
        "source_native_L32_index",
        "L",
        "volume",
        "action_density",
        "total_action",
        "phi2",
        "phi4",
        "NN",
        "diag",
        "2nn",
        "m",
        "m2",
        "m4",
        "G_pmin_x_cfg",
        "G_pmin_y_cfg",
        "nonfinite_count",
    ]
    out_dir = ensemble_dir / "observables"
    out_dir.mkdir(exist_ok=True)
    out_csv = out_dir / "main_per_sweep_measurements.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i in range(n):
            mi = float(m[i])
            writer.writerow(
                {
                    "chain_id": i,
                    "sweep": args.sweep,
                    "source_config_index": i,
                    "source_native_L32_index": i,
                    "L": L,
                    "volume": volume,
                    "action_density": float(action_density[i]),
                    "total_action": float(total_action[i]),
                    "phi2": float(phi2[i]),
                    "phi4": float(phi4[i]),
                    "NN": float(nn[i]),
                    "diag": float(diag[i]),
                    "2nn": float(twonn[i]),
                    "m": mi,
                    "m2": mi * mi,
                    "m4": mi**4,
                    "G_pmin_x_cfg": float(gpx[i]),
                    "G_pmin_y_cfg": float(gpy[i]),
                    "nonfinite_count": int(nonfinite[i]),
                }
            )
    root_csv = ensemble_dir / "main_per_sweep_measurements.csv"
    root_csv.write_bytes(out_csv.read_bytes())

    summary = {
        "status": "completed",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_configs": configs_path,
        "output_csv": out_csv,
        "root_mirror_csv": root_csv,
        "schema": "flow_detail main_per_sweep_measurements static native ensemble",
        "sweep": args.sweep,
        "metadata": meta,
        "n_configs": int(n),
        "L": int(L),
        "volume": int(volume),
        "lambda": lam,
        "kappa": kappa,
        "nonfinite_count": int(nonfinite.sum()),
        "means": {
            "action_density": float(np.mean(action_density)),
            "total_action": float(np.mean(total_action)),
            "phi2": float(np.mean(phi2)),
            "phi4": float(np.mean(phi4)),
            "NN": float(np.mean(nn)),
            "diag": float(np.mean(diag)),
            "2nn": float(np.mean(twonn)),
            "m": float(np.mean(m)),
            "m2": float(np.mean(m * m)),
            "m4": float(np.mean(m**4)),
            "G_pmin_x_cfg": float(np.mean(gpx)),
            "G_pmin_y_cfg": float(np.mean(gpy)),
        },
    }
    summary_path = out_dir / "main_per_sweep_measurements_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=json_default, allow_nan=True) + "\n")
    print(json.dumps({"csv": out_csv, "rows": n, "nonfinite": int(nonfinite.sum()), "means": summary["means"]}, indent=2, default=json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
