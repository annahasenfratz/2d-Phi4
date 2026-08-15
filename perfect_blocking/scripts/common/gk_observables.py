from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DEFAULT_MOMENTA: list[tuple[int, int]] = [(0, 0), (1, 0), (0, 1)]


def parse_momenta(raw: Iterable[Iterable[int]] | None) -> list[tuple[int, int]]:
    if raw is None:
        return list(DEFAULT_MOMENTA)
    momenta: list[tuple[int, int]] = []
    for item in raw:
        pair = list(item)
        if len(pair) != 2:
            raise ValueError(f"momentum must have two integer indices, got {item}")
        momenta.append((int(pair[0]), int(pair[1])))
    return momenta


def momentum_label(nx: int, ny: int) -> str:
    return f"G_{nx}{ny}"


def compute_Gk_per_config(
    phi: np.ndarray,
    momenta: Iterable[tuple[int, int]] | None = None,
    ensemble_label: str = "",
    source_file: str = "",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Compute per-config structure factors.

    The convention is the unconnected structure factor

        G_cfg(k) = |sum_x exp(i k.x) phi(x)|^2 / V.

    With m = V^{-1} sum_x phi(x), this gives G_cfg(0,0) = V m^2.
    """

    arr = np.asarray(phi, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[1] != arr.shape[2]:
        raise ValueError(f"expected configs with shape (N,L,L), got {arr.shape}")
    ncfg, L, _ = arr.shape
    volume = L * L
    selected = list(momenta or DEFAULT_MOMENTA)
    coords = np.arange(L, dtype=np.float64)
    long_rows: list[dict[str, Any]] = []
    wide_rows: list[dict[str, Any]] = [
        {
            "config_index": i,
            "source_file": source_file,
            "ensemble_label": ensemble_label,
            "L": L,
            "volume": volume,
        }
        for i in range(ncfg)
    ]

    values_by_label: dict[str, np.ndarray] = {}
    for nx, ny in selected:
        kx = 2.0 * math.pi * nx / L
        ky = 2.0 * math.pi * ny / L
        phase_x = np.exp(1j * kx * coords)
        phase_y = np.exp(1j * ky * coords)
        phase = phase_x[:, None] * phase_y[None, :]
        phi_tilde = np.sum(arr * phase[None, :, :], axis=(1, 2))
        gk = (np.abs(phi_tilde) ** 2) / float(volume)
        label = momentum_label(nx, ny)
        values_by_label[label] = gk
        for i in range(ncfg):
            value = float(gk[i])
            long_rows.append(
                {
                    "config_index": i,
                    "source_file": source_file,
                    "ensemble_label": ensemble_label,
                    "L": L,
                    "volume": volume,
                    "kx_index": nx,
                    "ky_index": ny,
                    "kx": kx,
                    "ky": ky,
                    "momentum_label": label,
                    "Gk": value,
                }
            )
            wide_rows[i][label] = value

    if "G_10" in values_by_label and "G_01" in values_by_label:
        avg = 0.5 * (values_by_label["G_10"] + values_by_label["G_01"])
        for i in range(ncfg):
            wide_rows[i]["G_pmin_avg"] = float(avg[i])
    return long_rows, wide_rows


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
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
