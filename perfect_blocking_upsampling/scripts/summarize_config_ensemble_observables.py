#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "perfect_blocking_upsampling" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from perfect_blocking_upsampling.actions import action_total  # noqa: E402
from perfect_blocking_upsampling.io import ActionSpec  # noqa: E402


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


def load_observables_csv(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load precomputed per-configuration observables in the native CSV schema."""
    required = ("m", "m2", "m4", "phi2", "phi4", "NN", "2nn", "diag", "action_density", "G_pmin_x_cfg", "G_pmin_y_cfg")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"observable CSV has no rows: {path}")
    missing = [key for key in required if key not in rows[0]]
    if missing:
        raise ValueError(f"observable CSV is missing required columns: {missing}")
    out = {key: np.asarray([float(row[key]) for row in rows], dtype=np.float64) for key in required}
    metadata: dict[str, Any] = {"n_rows": len(rows)}
    for key, target in (("L", "L"), ("lambda", "lambda"), ("kappa", "kappa")):
        if key in rows[0]:
            metadata[target] = float(rows[0][key]) if key != "L" else int(rows[0][key])
    return out, metadata


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


def per_config_observables(phi: np.ndarray, action: ActionSpec) -> dict[str, np.ndarray]:
    n, L, _ = phi.shape
    V = L * L
    m = phi.mean(axis=(1, 2))
    phi2 = np.mean(phi**2, axis=(1, 2))
    phi4 = np.mean(phi**4, axis=(1, 2))
    nnx = phi * np.roll(phi, -1, axis=1)
    nny = phi * np.roll(phi, -1, axis=2)
    links = np.concatenate([nnx.reshape(n, -1), nny.reshape(n, -1)], axis=1)
    nn = links.mean(axis=1)
    twonn = 0.5 * (
        np.mean(phi * np.roll(phi, -2, axis=1), axis=(1, 2))
        + np.mean(phi * np.roll(phi, -2, axis=2), axis=(1, 2))
    )
    diag = np.mean(phi * np.roll(np.roll(phi, -1, axis=1), -1, axis=2), axis=(1, 2))
    action_density = action_total(phi, action) / float(V)

    phase = np.exp(2j * np.pi * np.arange(L) / L)
    Phi_x = np.tensordot(phi, phase, axes=([1], [0])).sum(axis=1)
    Phi_y = np.tensordot(phi, phase, axes=([2], [0])).sum(axis=1)
    G_pmin_x_cfg = np.abs(Phi_x) ** 2 / float(V)
    G_pmin_y_cfg = np.abs(Phi_y) ** 2 / float(V)

    return {
        "m": m,
        "m2": m * m,
        "m4": m**4,
        "phi2": phi2,
        "phi4": phi4,
        "NN": nn,
        "2nn": twonn,
        "diag": diag,
        "action_density": action_density,
        "G_pmin_x_cfg": G_pmin_x_cfg,
        "G_pmin_y_cfg": G_pmin_y_cfg,
    }


def aggregate(pc: dict[str, np.ndarray], L: int, idx: np.ndarray | None = None) -> dict[str, float]:
    if idx is None:
        idx = np.arange(len(pc["m"]))
    mean = lambda key: float(np.mean(pc[key][idx]))
    m_mean = mean("m")
    m2 = mean("m2")
    m4 = mean("m4")
    phi2 = mean("phi2")
    phi4 = mean("phi4")
    G0 = float(L * L * max(m2 - m_mean * m_mean, 0.0))
    Gpx = mean("G_pmin_x_cfg")
    Gpy = mean("G_pmin_y_cfg")
    Gp = 0.5 * (Gpx + Gpy)
    sqrt_arg = G0 / Gp - 1.0 if Gp > 0.0 else float("nan")
    xi_over_L = (
        float((1.0 / (2.0 * L * math.sin(math.pi / L))) * math.sqrt(sqrt_arg))
        if G0 > 0.0 and Gp > 0.0 and sqrt_arg > 0.0
        else float("nan")
    )
    binder = float(1.0 - m4 / max(3.0 * m2 * m2, 1.0e-300))
    local_kurt = float(phi4 / max(phi2 * phi2, 1.0e-300))
    return {
        "Binder_U4_from_averages": binder,
        "Binder_U4": binder,
        "xi_over_L": xi_over_L,
        "chi": G0,
        "susceptibility_connected": G0,
        "m": m_mean,
        "m2": m2,
        "m4": m4,
        "phi2": phi2,
        "phi4": phi4,
        "NN": mean("NN"),
        "2nn": mean("2nn"),
        "diag": mean("diag"),
        "action_density": mean("action_density"),
        "local_kurtosis_ratio_from_averages": local_kurt,
        "G0_connected": G0,
        "G_pmin_x": Gpx,
        "G_pmin_y": Gpy,
        "G_pmin": Gp,
        "xi_2nd_sqrt_arg": sqrt_arg,
        "xi_2nd_rotational_asymmetry": float(abs(Gpx - Gpy) / max(Gp, 1.0e-300)),
    }


def bootstrap_errors(
    pc: dict[str, np.ndarray],
    L: int,
    keys: list[str],
    nboot: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(pc["m"])
    vals = {key: np.empty(nboot, dtype=np.float64) for key in keys}
    invalid_xi = 0
    for b in range(nboot):
        idx = rng.integers(0, n, size=n)
        agg = aggregate(pc, L, idx)
        if not np.isfinite(agg["xi_over_L"]):
            invalid_xi += 1
        for key in keys:
            vals[key][b] = agg[key]
    out = {key: float(np.nanstd(vals[key], ddof=1)) for key in keys}
    out["xi_over_L_invalid_bootstrap_fraction"] = invalid_xi / max(nboot, 1)
    return out


def bin_per_config(pc: dict[str, np.ndarray], bin_size: int) -> tuple[dict[str, np.ndarray], int]:
    """Average consecutive configurations into bootstrap bins.

    The input order is assumed to be the Markov-chain order.  A final partial
    bin is omitted so every bootstrap unit has the same weight.
    """
    if bin_size < 1:
        raise ValueError("bin_size must be positive")
    n_raw = len(pc["m"])
    n_bins = n_raw // bin_size
    if n_bins < 2:
        raise ValueError(f"bin_size={bin_size} leaves fewer than two bins from N={n_raw}")
    n_used = n_bins * bin_size
    return {
        key: values[:n_used].reshape(n_bins, bin_size).mean(axis=1)
        for key, values in pc.items()
    }, n_used


def fmt_key_value(key: str, value: float, err: float) -> str:
    return f'        "{key}": ({value:.8f}, {err:.8f}),'


def write_outputs(
    out_dir: Path,
    label: str,
    agg: dict[str, float],
    errs: dict[str, float],
    meta: dict[str, Any],
    keys: list[str],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [{"observable": key, "mean": agg[key], "error": errs.get(key, float("nan"))} for key in keys]
    with (out_dir / f"{label}_observable_summary_with_chi.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["observable", "mean", "error"])
        w.writeheader()
        w.writerows(rows)
    summary = {"label": label, "metadata": meta, "observables": {key: [agg[key], errs.get(key, float("nan"))] for key in keys}}
    for extra in ["G0_connected", "G_pmin_x", "G_pmin_y", "G_pmin", "xi_2nd_sqrt_arg", "xi_2nd_rotational_asymmetry"]:
        summary["observables"][extra] = [agg[extra], errs.get(extra, float("nan"))]
    (out_dir / f"{label}_observable_summary_with_chi.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    lines = [f'    "{label}": {{']
    for key in keys:
        lines.append(fmt_key_value(key, agg[key], errs.get(key, float("nan"))))
    lines.append("    }")
    (out_dir / f"{label}_observable_summary_with_chi.pyfrag").write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize phi4 configuration observables in dictionary-style format with connected chi.")
    ap.add_argument("configs", type=Path, help="Path to configs.npz, an ensemble directory, or a per-configuration observable CSV.")
    ap.add_argument("--label", default=None)
    ap.add_argument("--lambda", dest="lam", type=float, default=None)
    ap.add_argument("--kappa", type=float, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--nboot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--thermalization-cut", type=int, default=0, help="Drop this many leading configurations before analysis.")
    ap.add_argument("--bin-size", type=int, default=1, help="Consecutive configurations per bootstrap bin; use >1 for ordered MCMC output.")
    args = ap.parse_args()

    meta = infer_metadata(args.configs)
    csv_mode = args.configs.suffix.lower() == ".csv"
    pc_all: dict[str, np.ndarray] | None = None
    if csv_mode:
        pc_all, csv_meta = load_observables_csv(args.configs)
        meta = {**meta, **csv_meta}
    lam = float(args.lam if args.lam is not None else meta.get("lambda", 0.2))
    kappa = float(args.kappa if args.kappa is not None else meta["kappa"])
    if args.thermalization_cut < 0:
        raise SystemExit("--thermalization-cut must be nonnegative")
    if csv_mode:
        assert pc_all is not None
        n_raw = len(pc_all["m"])
        if args.thermalization_cut >= n_raw:
            raise SystemExit(f"--thermalization-cut={args.thermalization_cut} leaves no configurations from N={n_raw}")
        pc = {key: values[args.thermalization_cut :] for key, values in pc_all.items()}
        n = len(pc["m"])
        L = int(meta["L"])
    else:
        phi_all = load_configs(args.configs)
        if args.thermalization_cut >= len(phi_all):
            raise SystemExit(f"--thermalization-cut={args.thermalization_cut} leaves no configurations from N={len(phi_all)}")
        phi = phi_all[args.thermalization_cut :]
        n, L, _ = phi.shape
        action = ActionSpec("phi4_nn", lam, kappa)
        pc = per_config_observables(phi, action)
        n_raw = len(phi_all)
    n_post_cut_raw = len(pc["m"])
    try:
        pc_binned, n_used = bin_per_config(pc, args.bin_size)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    agg = aggregate(pc_binned, L)
    keys = [
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
    errs = bootstrap_errors(pc_binned, L, keys + ["G0_connected", "G_pmin_x", "G_pmin_y", "G_pmin", "xi_2nd_sqrt_arg", "xi_2nd_rotational_asymmetry"], args.nboot, args.seed)
    label = args.label or f"L{L}_kappac_{str(kappa).replace('.', 'p')}"
    out_dir = args.out_dir or (args.configs if args.configs.is_dir() else args.configs.parent)
    out_meta = {
        "configs": str(args.configs),
        "n_configs_raw": int(n_raw),
        "thermalization_cut": int(args.thermalization_cut),
        "n_configs": n_post_cut_raw,
        "n_configs_post_cut": n_post_cut_raw,
        "bootstrap_bin_size": int(args.bin_size),
        "n_bootstrap_bins": int(len(pc_binned["m"])),
        "n_configs_used_after_binning": int(n_used),
        "n_configs_discarded_by_binning": int(n_post_cut_raw - n_used),
        "L": L,
        "lambda": lam,
        "kappa": kappa,
        "action_convention": "phi4_nn: (1-2 lambda) phi2 + lambda phi4 - 2 kappa sum_positive_nn",
        "xi_over_L_definition": "connected second-moment Fourier estimator using G0=V(<m2>-<m>^2), averaged pmin x/y",
        "chi_definition": "connected chi = G0 = V(<m2>-<m>^2)",
        "nboot": args.nboot,
        "seed": args.seed,
        "input_mode": "per_config_observables_csv" if csv_mode else "configuration_npz",
    }
    write_outputs(out_dir, label, agg, errs, out_meta, keys)
    print((out_dir / f"{label}_observable_summary_with_chi.pyfrag").read_text(), end="")
    print(f"\nWrote outputs under {out_dir}", flush=True)


if __name__ == "__main__":
    main()
