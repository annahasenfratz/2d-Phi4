from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import numpy as np


OBSERVABLES = [
    "phi2",
    "phi4",
    "local_kurtosis_ratio",
    "NN",
    "2nn",
    "diag",
    "action_density",
    "m",
    "m2",
    "m4",
    "Binder_U4_from_averages",
    "xi_over_L",
]


def action_total(phi: np.ndarray, lam: float, kappa: float) -> np.ndarray:
    phi2 = phi * phi
    phi4 = phi2 * phi2
    nn = phi * np.roll(phi, -1, axis=1) + phi * np.roll(phi, -1, axis=2)
    density = (1.0 - 2.0 * lam) * phi2 + lam * phi4 - 2.0 * kappa * nn
    return density.sum(axis=(1, 2))


def per_config_observables(phi: np.ndarray, lam: float, kappa: float, source_prefix: str = "source") -> list[dict[str, Any]]:
    arr = np.asarray(phi, dtype=np.float64)
    n, L, _ = arr.shape
    volume = L * L
    m = arr.mean(axis=(1, 2))
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
    total_action = action_total(arr, lam, kappa)
    phase = np.exp(2j * np.pi * np.arange(L) / L)
    phi_x = np.tensordot(arr, phase, axes=([1], [0])).sum(axis=1)
    phi_y = np.tensordot(arr, phase, axes=([2], [0])).sum(axis=1)
    gpmin_x = np.abs(phi_x) ** 2 / float(volume)
    gpmin_y = np.abs(phi_y) ** 2 / float(volume)
    nonfinite = np.sum(~np.isfinite(arr), axis=(1, 2))
    rows: list[dict[str, Any]] = []
    for i in range(n):
        rows.append(
            {
                "sample": i,
                f"{source_prefix}_config_index": i,
                "L": L,
                "volume": volume,
                "lambda": lam,
                "kappa": kappa,
                "m": float(m[i]),
                "m2": float(m[i] * m[i]),
                "m4": float(m[i] ** 4),
                "phi2": float(phi2[i]),
                "phi4": float(phi4[i]),
                "local_kurtosis_ratio": float(phi4[i] / max(phi2[i] * phi2[i], 1.0e-300)),
                "NN": float(nn[i]),
                "2nn": float(twonn[i]),
                "diag": float(diag[i]),
                "action_density": float(total_action[i] / volume),
                "total_action": float(total_action[i]),
                "G_pmin_x_cfg": float(gpmin_x[i]),
                "G_pmin_y_cfg": float(gpmin_y[i]),
                "nonfinite_count": int(nonfinite[i]),
            }
        )
    return rows


def add_ensemble_observables(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    L = int(rows[0]["L"])
    volume = L * L
    m = np.asarray([float(r["m"]) for r in rows])
    m2 = np.asarray([float(r["m2"]) for r in rows])
    m4 = np.asarray([float(r["m4"]) for r in rows])
    gp = 0.5 * (
        np.mean([float(r["G_pmin_x_cfg"]) for r in rows])
        + np.mean([float(r["G_pmin_y_cfg"]) for r in rows])
    )
    mean_m = float(np.mean(m))
    mean_m2 = float(np.mean(m2))
    mean_m4 = float(np.mean(m4))
    binder = float(1.0 - mean_m4 / max(3.0 * mean_m2 * mean_m2, 1.0e-300))
    g0 = float(volume * max(mean_m2 - mean_m * mean_m, 0.0))
    sqrt_arg = g0 / gp - 1.0 if gp > 0.0 else float("nan")
    xi = (
        float((1.0 / (2.0 * L * math.sin(math.pi / L))) * math.sqrt(sqrt_arg))
        if g0 > 0.0 and gp > 0.0 and sqrt_arg > 0.0
        else float("nan")
    )
    for row in rows:
        row["Binder_U4_from_averages"] = binder
        row["xi_over_L"] = xi


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
