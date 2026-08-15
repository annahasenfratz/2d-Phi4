#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=float) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def local_series(phi: np.ndarray, kappa: float, lam: float) -> dict[str, np.ndarray]:
    arr = np.asarray(phi, dtype=np.float64)
    m = arr.mean(axis=(1, 2))
    phi2 = np.mean(arr**2, axis=(1, 2))
    phi4 = np.mean(arr**4, axis=(1, 2))
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
    action_density = (1.0 - 2.0 * lam) * phi2 + lam * phi4 - 4.0 * kappa * nn
    return {
        "m": m,
        "abs_m": np.abs(m),
        "phi2": phi2,
        "phi4": phi4,
        "NN": nn,
        "2NN": two_nn,
        "diag": diag,
        "action_density": action_density,
        "susceptibility_per_config_proxy": arr.shape[1] * arr.shape[2] * m * m,
    }


def stat(vals: np.ndarray) -> dict[str, float]:
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "se_naive": float(np.std(arr, ddof=1) / math.sqrt(len(arr))) if len(arr) > 1 else 0.0,
        "n": int(len(arr)),
    }


def summarize(ensemble_dir: Path) -> dict[str, Any]:
    cfg_path = ensemble_dir / "configs.npz"
    manifest_path = ensemble_dir / "manifest.json"
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    data = np.load(cfg_path)
    phi = data["phi"].astype(np.float32)
    lam = float(manifest.get("lambda", data.get("lambda", 0.022)))
    kappa = float(manifest.get("kappa", data.get("kappa", 0.271)))
    ser = local_series(phi, kappa, lam)
    m2 = ser["m"] ** 2
    m4 = ser["m"] ** 4
    binder_u4 = 1.0 - float(np.mean(m4)) / (3.0 * float(np.mean(m2)) ** 2)
    susceptibility = phi.shape[1] * phi.shape[2] * float(np.mean(m2))
    xi_over_L_proxy = math.sqrt(max(susceptibility, 0.0) / max(float(np.mean(ser["phi2"])), 1.0e-300)) / phi.shape[1]
    summary = {
        "path": str(cfg_path),
        "shape": list(phi.shape),
        "lambda": lam,
        "kappa": kappa,
        "L": int(phi.shape[1]),
        "n_configs": int(phi.shape[0]),
        "binder_U4": binder_u4,
        "susceptibility": susceptibility,
        "xi_over_L_project_proxy": xi_over_L_proxy,
        "observables": {key: stat(vals) for key, vals in ser.items()},
        "manifest": manifest,
    }
    write_json(ensemble_dir / "local_observable_summary.json", summary)
    checksum_rows = []
    for path in sorted(ensemble_dir.glob("*")):
        if path.is_file() and path.name != "sha256_checksums.txt":
            checksum_rows.append((sha256(path), path.name))
    with (ensemble_dir / "sha256_checksums.txt").open("w", encoding="utf-8") as f:
        for digest, name in checksum_rows:
            f.write(f"{digest}  {name}\n")
    with (ensemble_dir / "local_observable_summary.csv").open("w", newline="", encoding="utf-8") as f:
        rows = [{"observable": key, **value} for key, value in summary["observables"].items()]
        rows += [
            {"observable": "Binder_U4", "mean": binder_u4, "std": "", "se_naive": "", "n": int(phi.shape[0])},
            {"observable": "susceptibility", "mean": susceptibility, "std": "", "se_naive": "", "n": int(phi.shape[0])},
            {"observable": "xi_over_L_project_proxy", "mean": xi_over_L_proxy, "std": "", "se_naive": "", "n": int(phi.shape[0])},
        ]
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ensemble-dir", type=Path, required=True)
    args = ap.parse_args()
    summary = summarize(args.ensemble_dir)
    print(json.dumps({
        "ensemble_dir": str(args.ensemble_dir),
        "shape": summary["shape"],
        "binder_U4": summary["binder_U4"],
        "susceptibility": summary["susceptibility"],
        "xi_over_L_project_proxy": summary["xi_over_L_project_proxy"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
