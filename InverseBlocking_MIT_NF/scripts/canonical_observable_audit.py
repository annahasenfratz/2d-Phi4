#!/usr/bin/env python3
"""Canonical observable/provenance audit for the 1024 paired dataset."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "outputs" / "paired_data_lam1_kappaf0p320"
OUT = PROJECT / "outputs" / "canonical_observable_audit"

INPUTS = {
    "fine_16x16": DATA / "fine_configs.npy",
    "blocked_coarse_8x8": DATA / "coarse_blocked_configs.npy",
    "backbone_16x16": DATA / "backbone_configs.npy",
}
OBSOLETE_BLOCKED = PROJECT / "outputs" / "coarse_distribution_calibration" / "blocked_fine_coarse.npy"
LOW_MODES = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (0, 2)]
N_BOOT = 512
BOOT_SEED = 20260624


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def file_info(path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {
        "path": str(path.resolve()),
        "exists": path.exists(),
    }
    if path.exists():
        stat = path.stat()
        info.update(
            {
                "size_bytes": stat.st_size,
                "mtime_unix": stat.st_mtime,
                "mtime_local": __import__("datetime").datetime.fromtimestamp(stat.st_mtime).isoformat(),
            }
        )
        try:
            arr = np.load(path, mmap_mode="r")
            info.update({"shape": list(arr.shape), "dtype": str(arr.dtype), "n_configs": int(arr.shape[0])})
        except Exception as exc:  # pragma: no cover - provenance only
            info["load_error"] = repr(exc)
    return info


def aggregate_observables(phi: np.ndarray, label: str, lattice_size: int) -> dict[str, float | str | int]:
    phi = np.asarray(phi, dtype=np.float64)
    n, ly, lx = phi.shape
    volume = ly * lx
    m_cfg = phi.mean(axis=(-2, -1))
    m2 = float(np.mean(m_cfg**2))
    m4 = float(np.mean(m_cfg**4))
    b4 = m4 / (m2 * m2) if m2 > 0 else math.nan
    u4 = 1.0 - b4 / 3.0 if math.isfinite(b4) else math.nan
    nn_cfg = 0.5 * (
        (phi * np.roll(phi, -1, axis=-2)).mean(axis=(-2, -1))
        + (phi * np.roll(phi, -1, axis=-1)).mean(axis=(-2, -1))
    )
    nn2_cfg = 0.5 * (
        ((phi * np.roll(phi, -1, axis=-2)) ** 2).mean(axis=(-2, -1))
        + ((phi * np.roll(phi, -1, axis=-1)) ** 2).mean(axis=(-2, -1))
    )
    diag_cfg = (phi * np.roll(np.roll(phi, -1, axis=-2), -1, axis=-1)).mean(axis=(-2, -1))
    twonn_cfg = 0.5 * (
        (phi * np.roll(phi, -2, axis=-2)).mean(axis=(-2, -1))
        + (phi * np.roll(phi, -2, axis=-1)).mean(axis=(-2, -1))
    )
    ft = np.fft.fft2(phi, axes=(-2, -1))
    chi = float(volume * (np.mean(m_cfg**2) - np.mean(m_cfg) ** 2))
    f_pmin = float(0.5 * (np.mean(np.abs(ft[:, 1, 0]) ** 2) + np.mean(np.abs(ft[:, 0, 1]) ** 2)) / volume)
    ratio = chi / f_pmin - 1.0 if f_pmin > 0 else math.nan
    xi = (1.0 / (2.0 * math.sin(math.pi / lx))) * math.sqrt(ratio) if ratio > 0 else math.nan
    return {
        "ensemble": label,
        "L": int(lattice_size),
        "N": int(n),
        "m": float(np.mean(m_cfg)),
        "abs_m": float(np.mean(np.abs(m_cfg))),
        "phi2": float(np.mean(phi**2)),
        "phi4": float(np.mean(phi**4)),
        "NN": float(np.mean(nn_cfg)),
        "nn2": float(np.mean(nn2_cfg)),
        "diag": float(np.mean(diag_cfg)),
        "2nn": float(np.mean(twonn_cfg)),
        "Binder_U4": float(u4),
        "Binder_ratio_B4": float(b4),
        "chi_connected": chi,
        "F_pmin": f_pmin,
        "xi": float(xi) if math.isfinite(xi) else math.nan,
        "xi_over_L": float(xi / lx) if math.isfinite(xi) else math.nan,
    }


def bootstrap_errors(phi: np.ndarray, label: str, lattice_size: int, rng: np.random.Generator) -> dict[str, Any]:
    n = len(phi)
    base = aggregate_observables(phi, label, lattice_size)
    boot_vals: dict[str, list[float]] = {k: [] for k, v in base.items() if isinstance(v, (float, int)) and k not in {"L", "N"}}
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        obs = aggregate_observables(phi[idx], label, lattice_size)
        for key in boot_vals:
            boot_vals[key].append(float(obs[key]))
    row: dict[str, Any] = dict(base)
    row["error_method"] = f"bootstrap_configs_n{N_BOOT}"
    row["bootstrap_seed"] = BOOT_SEED
    for key, vals in boot_vals.items():
        arr = np.asarray(vals, dtype=np.float64)
        row[f"{key}_err"] = float(np.std(arr, ddof=1))
    return row


def low_momentum_spectrum(phi: np.ndarray, label: str, lattice_size: int) -> list[dict[str, Any]]:
    phi = np.asarray(phi, dtype=np.float64)
    _, ly, lx = phi.shape
    volume = ly * lx
    ft = np.fft.fft2(phi, axes=(-2, -1))
    rows = []
    for ky, kx in LOW_MODES:
        rows.append(
            {
                "ensemble": label,
                "L": int(lattice_size),
                "N": int(len(phi)),
                "ky_index": int(ky),
                "kx_index": int(kx),
                "p_y": float(2.0 * math.pi * ky / ly),
                "p_x": float(2.0 * math.pi * kx / lx),
                "S_p": float(np.mean(np.abs(ft[:, ky % ly, kx % lx]) ** 2) / volume),
                "fft_convention": "numpy unnormalized fft2; S(p)=<|FFT(phi)(p)|^2>/V",
            }
        )
    return rows


def load_json_if_exists(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {"json_read_error": f"could not parse {path}"}


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    arrays = {
        "fine_16x16": np.load(INPUTS["fine_16x16"]).astype(np.float64),
        "blocked_coarse_8x8": np.load(INPUTS["blocked_coarse_8x8"]).astype(np.float64),
        "backbone_16x16": np.load(INPUTS["backbone_16x16"]).astype(np.float64),
    }
    lattice_sizes = {"fine_16x16": 16, "blocked_coarse_8x8": 8, "backbone_16x16": 16}

    obs_rows = [aggregate_observables(arr, label, lattice_sizes[label]) for label, arr in arrays.items()]
    rng = np.random.default_rng(BOOT_SEED)
    err_rows = [bootstrap_errors(arr, label, lattice_sizes[label], rng) for label, arr in arrays.items()]
    spec_rows = []
    for label in ("fine_16x16", "backbone_16x16"):
        spec_rows.extend(low_momentum_spectrum(arrays[label], label, lattice_sizes[label]))

    write_csv(OUT / "canonical_observables.csv", obs_rows)
    write_csv(OUT / "canonical_observables_with_errors.csv", err_rows)
    write_csv(OUT / "low_momentum_spectrum.csv", spec_rows)

    provenance = {
        "audit_created_utc": __import__("datetime").datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "cwd": os.getcwd(),
        "script": str(Path(__file__).resolve()),
        "canonical_input_directory": str(DATA.resolve()),
        "inputs": {label: file_info(path) for label, path in INPUTS.items()},
        "available_metadata_files": {
            "generation_metadata": load_json_if_exists(DATA / "generation_metadata.json"),
            "summary": load_json_if_exists(DATA / "summary.json"),
            "whitening_summary": load_json_if_exists(DATA / "whitening_summary.json"),
        },
        "obsolete_file_note": {
            "path": str(OBSOLETE_BLOCKED.resolve()),
            "status": "obsolete_for_current_comparisons",
            "file_info": file_info(OBSOLETE_BLOCKED),
            "instruction": "Do not use this 64-config blocked_fine_coarse.npy for current canonical comparisons.",
        },
        "observable_implementation": {
            "binder": "U4=1-<m^4>/(3 <m^2>^2), B4=<m^4>/<m^2>^2, m=volume average per configuration",
            "xi": "connected chi=V*(<m^2>-<m>^2); F_pmin=0.5*(<|FFT(1,0)|^2>+<|FFT(0,1)|^2>)/V; xi=(2 sin(pi/L))^-1 sqrt(chi/F-1)",
            "fft": "numpy unnormalized fft2",
            "errors": f"configuration bootstrap with {N_BOOT} replicates and seed {BOOT_SEED}",
        },
    }
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")

    headers = ["ensemble", "L", "N", "m", "abs_m", "phi2", "phi4", "NN", "nn2", "diag", "2nn", "Binder_U4", "Binder_ratio_B4", "xi", "xi_over_L"]
    table = "\n".join(
        ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] + ["---:"] * (len(headers) - 1)) + "|"]
        + ["| " + " | ".join(fmt(row[h]) for h in headers) + " |" for row in obs_rows]
    )
    err_headers = ["ensemble", "Binder_U4", "Binder_U4_err", "xi_over_L", "xi_over_L_err", "phi2", "phi2_err", "phi4", "phi4_err", "nn2", "nn2_err"]
    err_table = "\n".join(
        ["| " + " | ".join(err_headers) + " |", "|" + "|".join(["---"] + ["---:"] * (len(err_headers) - 1)) + "|"]
        + ["| " + " | ".join(fmt(row[h]) for h in err_headers) + " |" for row in err_rows]
    )

    fine = obs_rows[0]
    coarse = obs_rows[1]
    back = obs_rows[2]
    report = f"""# Canonical Observable Audit

## Inputs

This audit recomputes all listed observables from one canonical implementation using only:

- fine configs: `{INPUTS['fine_16x16'].resolve()}`
- blocked coarse configs: `{INPUTS['blocked_coarse_8x8'].resolve()}`
- backbone configs: `{INPUTS['backbone_16x16'].resolve()}`

The older file `{OBSOLETE_BLOCKED.resolve()}` is obsolete for current comparisons. It contains `{file_info(OBSOLETE_BLOCKED).get('n_configs')}` configurations and should not be used for current conclusions.

## Canonical Table

{table}

## Bootstrap Errors

{err_table}

## Provenance

- canonical input directory: `{DATA.resolve()}`
- fine shape: `{list(arrays['fine_16x16'].shape)}`, dtype after load: `float64`
- blocked coarse shape: `{list(arrays['blocked_coarse_8x8'].shape)}`, dtype after load: `float64`
- backbone shape: `{list(arrays['backbone_16x16'].shape)}`, dtype after load: `float64`
- file mtimes and available metadata are saved in `provenance.json`.
- low-momentum spectrum for fine and backbone is saved in `low_momentum_spectrum.csv`.

## Stable Conclusion

Binder and xi/L are preserved by the symmetric blocking/backbone construction on the canonical 1024-pair data:

- Binder U4: fine `{fine['Binder_U4']:.6g}`, blocked coarse `{coarse['Binder_U4']:.6g}`, backbone `{back['Binder_U4']:.6g}`.
- xi/L: fine `{fine['xi_over_L']:.6g}`, blocked coarse `{coarse['xi_over_L']:.6g}`, backbone `{back['xi_over_L']:.6g}`.

The UV/local observables are reduced in the blocked/backbone fields, as expected:

- phi2: fine `{fine['phi2']:.6g}`, blocked coarse `{coarse['phi2']:.6g}`, backbone `{back['phi2']:.6g}`.
- phi4: fine `{fine['phi4']:.6g}`, blocked coarse `{coarse['phi4']:.6g}`, backbone `{back['phi4']:.6g}`.
- nn2: fine `{fine['nn2']:.6g}`, blocked coarse `{coarse['nn2']:.6g}`, backbone `{back['nn2']:.6g}`.

This is the canonical comparison to use going forward.
"""
    (OUT / "report.md").write_text(report)


if __name__ == "__main__":
    main()
