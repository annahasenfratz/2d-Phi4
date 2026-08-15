#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np


def load_configs(path: Path) -> np.ndarray:
    if path.is_dir():
        path = path / "configs.npz"
    with np.load(path) as data:
        for key in ("phi", "configs", "arr_0"):
            if key in data.files:
                arr = data[key]
                break
        else:
            arr = data[data.files[0]]
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[1] != arr.shape[2]:
        raise ValueError(f"expected configs with shape (N,L,L), got {arr.shape}")
    return arr


def infer_metadata(config_path: Path) -> dict[str, Any]:
    directory = config_path if config_path.is_dir() else config_path.parent
    manifest = directory / "manifest.json"
    out: dict[str, Any] = {}
    if manifest.exists():
        out.update(json.loads(manifest.read_text()))
    text = " ".join(str(part) for part in directory.parts[-4:])
    if "lambda" not in out:
        m = re.search(r"lam([0-9]+)p([0-9]+)", text)
        if m:
            out["lambda"] = float(f"{m.group(1)}.{m.group(2)}")
    if "kappa" not in out:
        m = re.search(r"kappa(?:c)?([0-9]+)p([0-9]+)", text)
        if m:
            out["kappa"] = float(f"{m.group(1)}.{m.group(2)}")
    if "L" not in out:
        m = re.search(r"_L([0-9]+)(?:_|$)", directory.name)
        if m:
            out["L"] = int(m.group(1))
    return out


def binder_from_moments(m2: float, m4: float) -> float:
    return float(1.0 - m4 / max(3.0 * m2 * m2, 1.0e-300))


def make_bins(n: int, bin_size: int, drop_partial: bool) -> list[tuple[int, int]]:
    bins = [(start, min(start + bin_size, n)) for start in range(0, n, bin_size)]
    if drop_partial and bins and (bins[-1][1] - bins[-1][0]) < bin_size:
        bins.pop()
    return [(a, b) for a, b in bins if b > a]


def jackknife_error_from_bin_sums(bin_m2_sum: np.ndarray, bin_m4_sum: np.ndarray, bin_count: np.ndarray) -> tuple[float, float]:
    n_bins = len(bin_count)
    total_m2 = float(np.sum(bin_m2_sum))
    total_m4 = float(np.sum(bin_m4_sum))
    total_n = float(np.sum(bin_count))
    full = binder_from_moments(total_m2 / total_n, total_m4 / total_n)
    if n_bins <= 1:
        return full, float("nan")
    jk = np.empty(n_bins, dtype=np.float64)
    for i in range(n_bins):
        n_leave = total_n - float(bin_count[i])
        if n_leave <= 0:
            jk[i] = float("nan")
        else:
            jk[i] = binder_from_moments((total_m2 - float(bin_m2_sum[i])) / n_leave, (total_m4 - float(bin_m4_sum[i])) / n_leave)
    valid = jk[np.isfinite(jk)]
    if len(valid) <= 1:
        return full, float("nan")
    jk_mean = float(np.mean(valid))
    err = math.sqrt((len(valid) - 1) / len(valid) * float(np.sum((valid - jk_mean) ** 2)))
    return full, float(err)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute Binder cumulant with sequential bin/chunk jackknife errors.")
    ap.add_argument("configs", type=Path, help="Path to configs.npz or ensemble directory containing configs.npz.")
    ap.add_argument("--label", default=None)
    ap.add_argument("--bin", dest="bin_size", type=int, default=250, help="Sequential bin/chunk size. Default: 250.")
    ap.add_argument("--drop-partial", action="store_true", help="Drop the final partial bin instead of including it.")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    if args.bin_size <= 0:
        raise ValueError(f"--bin must be positive, got {args.bin_size}")

    phi = load_configs(args.configs)
    n, L, _ = phi.shape
    meta = infer_metadata(args.configs)
    label = args.label or (args.configs.name if args.configs.is_dir() else args.configs.parent.name)
    out_dir = args.out_dir or ((args.configs if args.configs.is_dir() else args.configs.parent) / "binder_bins")

    m = phi.mean(axis=(1, 2))
    m2_cfg = m * m
    m4_cfg = m**4
    bins = make_bins(n, args.bin_size, args.drop_partial)
    if not bins:
        raise RuntimeError(f"No bins available for n={n}, bin={args.bin_size}, drop_partial={args.drop_partial}")

    bin_rows: list[dict[str, Any]] = []
    bin_m2_sum = np.empty(len(bins), dtype=np.float64)
    bin_m4_sum = np.empty(len(bins), dtype=np.float64)
    bin_count = np.empty(len(bins), dtype=np.int64)
    for i, (start, stop) in enumerate(bins):
        m2 = float(np.mean(m2_cfg[start:stop]))
        m4 = float(np.mean(m4_cfg[start:stop]))
        binder = binder_from_moments(m2, m4)
        count = int(stop - start)
        bin_m2_sum[i] = float(np.sum(m2_cfg[start:stop]))
        bin_m4_sum[i] = float(np.sum(m4_cfg[start:stop]))
        bin_count[i] = count
        bin_rows.append(
            {
                "bin_index": i,
                "start_config": start,
                "stop_config_exclusive": stop,
                "count": count,
                "m2": m2,
                "m4": m4,
                "Binder_U4_from_bin_averages": binder,
            }
        )

    full_binder, jackknife_err = jackknife_error_from_bin_sums(bin_m2_sum, bin_m4_sum, bin_count)
    bin_binders = np.asarray([r["Binder_U4_from_bin_averages"] for r in bin_rows], dtype=np.float64)
    bin_mean = float(np.mean(bin_binders))
    bin_sem = float(np.std(bin_binders, ddof=1) / math.sqrt(len(bin_binders))) if len(bin_binders) > 1 else float("nan")
    full_m2 = float(np.mean(m2_cfg))
    full_m4 = float(np.mean(m4_cfg))

    summary = {
        "label": label,
        "configs": str(args.configs),
        "metadata": meta,
        "n_configs_total": int(n),
        "L": int(L),
        "bin_size": int(args.bin_size),
        "drop_partial": bool(args.drop_partial),
        "n_bins": int(len(bins)),
        "n_configs_used": int(np.sum(bin_count)),
        "full_m2": full_m2,
        "full_m4": full_m4,
        "Binder_U4_from_averages": full_binder,
        "Binder_U4_jackknife_error": jackknife_err,
        "mean_of_bin_Binder_U4": bin_mean,
        "mean_of_bin_Binder_U4_sem": bin_sem,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        out_dir / f"{label}_binder_bins_bin{args.bin_size}.csv",
        bin_rows,
        ["bin_index", "start_config", "stop_config_exclusive", "count", "m2", "m4", "Binder_U4_from_bin_averages"],
    )
    (out_dir / f"{label}_binder_bins_bin{args.bin_size}_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    pyfrag = (
        f'    "{label}_binder_bin{args.bin_size}": {{\n'
        f'        "Binder_U4_from_averages": ({full_binder:.8f}, {jackknife_err:.8f}),\n'
        f'        "m2": ({full_m2:.8f}, nan),\n'
        f'        "m4": ({full_m4:.8f}, nan),\n'
        f'        "bin_size": ({args.bin_size}, 0.0),\n'
        f'        "n_bins": ({len(bins)}, 0.0),\n'
        f'        "n_configs_used": ({int(np.sum(bin_count))}, 0.0),\n'
        "    }\n"
    )
    (out_dir / f"{label}_binder_bins_bin{args.bin_size}.pyfrag").write_text(pyfrag)
    print(pyfrag, end="")
    print(f"\nWrote Binder bin outputs under {out_dir}", flush=True)


if __name__ == "__main__":
    main()
