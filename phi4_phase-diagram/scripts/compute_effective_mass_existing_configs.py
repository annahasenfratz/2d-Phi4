#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def resolve_input(path: Path) -> tuple[Path, Path | None]:
    if path.is_dir():
        cfg = path / "configs.npz"
        manifest = path / "manifest.json"
    else:
        cfg = path
        manifest = path.with_name("manifest.json")
    if not cfg.exists():
        raise FileNotFoundError(f"configuration file not found: {cfg}")
    return cfg, manifest if manifest.exists() else None


def read_manifest(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_phi(cfg_path: Path, burn: int, max_configs: int | None) -> np.ndarray:
    with np.load(cfg_path) as data:
        key = "phi" if "phi" in data else data.files[0]
        phi = np.asarray(data[key], dtype=np.float64)
    if phi.ndim != 3:
        raise ValueError(f"expected configs with shape (N,L,L), got {phi.shape}")
    if burn < 0:
        raise ValueError("--burn-configs must be non-negative")
    if burn >= phi.shape[0]:
        raise ValueError(f"--burn-configs={burn} leaves no configurations from N={phi.shape[0]}")
    phi = phi[burn:]
    if max_configs is not None:
        phi = phi[:max_configs]
    return phi


def per_config_correlators(phi: np.ndarray) -> np.ndarray:
    # Phi(x) = sum_y phi(x,y); C(t) = sum_x Phi(x) Phi(x+t).
    slices = np.sum(phi, axis=2)
    fft = np.fft.fft(slices, axis=1)
    corr = np.fft.ifft(fft * np.conj(fft), axis=1).real
    return corr


def fold_correlator(corr: np.ndarray) -> np.ndarray:
    n = len(corr)
    half = n // 2
    folded = np.empty(half + 1, dtype=np.float64)
    folded[0] = corr[0]
    for t in range(1, half):
        folded[t] = 0.5 * (corr[t] + corr[n - t])
    folded[half] = corr[half] if n % 2 == 0 else 0.5 * (corr[half] + corr[n - half])
    return folded


def effective_mass(corr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arg = (corr[2:] + corr[:-2]) / (2.0 * corr[1:-1])
    out = np.full_like(arg, np.nan, dtype=np.float64)
    ok = np.isfinite(arg) & (arg >= 1.0)
    out[ok] = np.arccosh(arg[ok])
    return out, arg


def bootstrap_errors(per_cfg_corr: np.ndarray, folded: bool, n_boot: int, seed: int) -> dict[str, np.ndarray]:
    n = per_cfg_corr.shape[0]
    rng = np.random.default_rng(seed)
    corr_samples = []
    meff_samples = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        c = np.mean(per_cfg_corr[idx], axis=0)
        if folded:
            c = fold_correlator(c)
        m, _ = effective_mass(c)
        corr_samples.append(c)
        meff_samples.append(m)
    return {
        "corr_se": np.nanstd(np.asarray(corr_samples), axis=0, ddof=1),
        "meff_se": np.nanstd(np.asarray(meff_samples), axis=0, ddof=1),
    }


def make_plots(out: Path, meff_rows: list[dict[str, Any]], label: str, L: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.asarray([float(r["t"]) for r in meff_rows])
    m = np.asarray([float(r["m_eff"]) for r in meff_rows])
    e = np.asarray([float(r["m_eff_se"]) for r in meff_rows])
    finite = np.isfinite(m)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.errorbar(t[finite], m[finite], yerr=e[finite], marker="o", ms=3, lw=1.0, capsize=2)
    ax.set_xlabel("t")
    ax.set_ylabel("m_eff(t)")
    ax.set_title(label)
    ax.grid(True, alpha=0.18)
    fig.tight_layout()
    fig.savefig(out / "effective_mass_vs_t.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.errorbar((t / L)[finite], (m * L)[finite], yerr=(e * L)[finite], marker="o", ms=3, lw=1.0, capsize=2)
    ax.set_xlabel("t / L")
    ax.set_ylabel("m_eff(t) L")
    ax.set_title(label)
    ax.grid(True, alpha=0.18)
    fig.tight_layout()
    fig.savefig(out / "effective_mass_times_L_vs_t_over_L.pdf")
    plt.close(fig)


def write_report(
    out: Path,
    cfg_path: Path,
    manifest: dict[str, Any],
    args: argparse.Namespace,
    n_raw: int,
    n_used: int,
    symmetry_max: float,
    finite_range: tuple[int | None, int | None],
    plateau: dict[str, float] | None,
    nonfinite_count: int,
    total_meff_count: int,
) -> None:
    lam = args.lam if args.lam is not None else manifest.get("lambda", "")
    kappa = args.kappa if args.kappa is not None else manifest.get("kappa", "")
    L = args.L if args.L is not None else manifest.get("L", "")
    fold_text = (
        "Folded convention: `C_fold[0]=C[0]`, `C_fold[t]=0.5*(C[t]+C[L-t])` for `1 <= t < L/2`, and `C_fold[L/2]=C[L/2]` for even L."
        if args.fold
        else "No folding was applied before the effective-mass extraction."
    )
    lines = [
        "# Effective mass first pass, lambda=0.022 L128",
        "",
        f"- input: `{cfg_path}`",
        f"- lambda: `{lam}`",
        f"- kappa: `{kappa}`",
        f"- L: `{L}`",
        f"- raw configs: `{n_raw}`",
        f"- burn configs: `{args.burn_configs}`",
        f"- used configs: `{n_used}`",
        f"- bootstrap samples: `{args.bootstrap}`",
        "",
        "Correlator definition:",
        "",
        "`Phi(x)=sum_y phi(x,y)` and `C(t)=sum_x Phi(x) Phi(x+t)` with periodic x direction, averaged over configurations.",
        "",
        fold_text,
        "",
        f"Maximum relative asymmetry `|C(t)-C(L-t)|/max(|C(t)|,|C(L-t)|)` over raw unfurled correlator: `{symmetry_max:.6g}`.",
        f"Finite effective-mass points span t=`{finite_range[0]}` to `{finite_range[1]}`, with `{nonfinite_count}` non-finite points out of `{total_meff_count}` effective-mass times.",
        "",
    ]
    if plateau:
        lines += [
            "Preliminary plateau diagnostic:",
            "",
            f"- window: `t={plateau['t_min']}` to `t={plateau['t_max']}`",
            f"- mean `m_eff L`: `{plateau['m_eff_L_mean']:.6g}`",
            f"- scatter: `{plateau['m_eff_L_std']:.6g}`",
            "",
            "Qualitative assessment: this is not a clean plateau. The quoted window is a visual first-pass range; the scatter is sizable and late folded points include arccosh failures, so this should not be treated as a fitted mass.",
            "",
        ]
    else:
        lines += [
            "Qualitative assessment: no finite plateau window was identified by the simple first-pass rule.",
            "",
        ]
    lines += [
        "Plots:",
        "",
        "- `effective_mass_vs_t.pdf`",
        "- `effective_mass_times_L_vs_t_over_L.pdf`",
        "",
        "Data:",
        "",
        "- `correlator_raw.csv`",
        "- `correlator_folded.csv` if folding was enabled",
        "- `effective_mass.csv`",
    ]
    (out / "EFFECTIVE_MASS_L128_FIRST_PASS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path, help="configs.npz file or ensemble directory")
    ap.add_argument("--L", type=int, default=None)
    ap.add_argument("--max-configs", type=int, default=None)
    ap.add_argument("--burn-configs", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--fold", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--lambda", dest="lam", type=float, default=None)
    ap.add_argument("--kappa", type=float, default=None)
    ap.add_argument("--label", type=str, default="")
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260701)
    args = ap.parse_args()

    cfg_path, manifest_path = resolve_input(args.input)
    manifest = read_manifest(manifest_path)
    raw_n = int(np.load(cfg_path)["phi"].shape[0])
    phi = load_phi(cfg_path, args.burn_configs, args.max_configs)
    L = int(args.L or phi.shape[1])
    if phi.shape[1:] != (L, L):
        raise ValueError(f"L={L} does not match config shape {phi.shape[1:]}")

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    per_cfg = per_config_correlators(phi)
    corr = np.mean(per_cfg, axis=0)
    corr_se = np.std(per_cfg, axis=0, ddof=1) / math.sqrt(per_cfg.shape[0])
    symmetry = []
    for t in range(1, L):
        denom = max(abs(corr[t]), abs(corr[L - t]), 1e-300)
        symmetry.append(abs(corr[t] - corr[L - t]) / denom)
    symmetry_max = float(np.max(symmetry)) if symmetry else 0.0

    corr_for_mass = fold_correlator(corr) if args.fold else corr
    boot = bootstrap_errors(per_cfg, args.fold, args.bootstrap, args.seed) if args.bootstrap > 0 else {
        "corr_se": np.full_like(corr_for_mass, np.nan),
        "meff_se": np.full(max(len(corr_for_mass) - 2, 0), np.nan),
    }
    meff, acosh_arg = effective_mass(corr_for_mass)

    raw_rows = [
        {"t": t, "C": float(corr[t]), "C_se_naive": float(corr_se[t]), "C_sym_partner": int((L - t) % L)}
        for t in range(L)
    ]
    write_csv(out / "correlator_raw.csv", raw_rows)

    if args.fold:
        folded_rows = [
            {"t": t, "C_fold": float(corr_for_mass[t]), "C_fold_bootstrap_se": float(boot["corr_se"][t])}
            for t in range(len(corr_for_mass))
        ]
        write_csv(out / "correlator_folded.csv", folded_rows)

    meff_rows = []
    for i, val in enumerate(meff):
        t = i + 1
        meff_rows.append(
            {
                "t": t,
                "t_over_L": t / L,
                "m_eff": float(val),
                "m_eff_se": float(boot["meff_se"][i]) if i < len(boot["meff_se"]) else float("nan"),
                "m_eff_times_L": float(val * L) if np.isfinite(val) else float("nan"),
                "m_eff_times_L_se": float(boot["meff_se"][i] * L) if i < len(boot["meff_se"]) else float("nan"),
                "arccosh_argument": float(acosh_arg[i]),
                "finite": bool(np.isfinite(val)),
            }
        )
    write_csv(out / "effective_mass.csv", meff_rows)

    bad = [r for r in meff_rows if not r["finite"]]
    if bad:
        print(f"warning: {len(bad)} effective-mass points are non-finite or have arccosh argument < 1")

    finite_ts = [int(r["t"]) for r in meff_rows if r["finite"]]
    finite_range = (min(finite_ts), max(finite_ts)) if finite_ts else (None, None)
    plateau = None
    plateau_rows = [r for r in meff_rows if r["finite"] and 0.12 <= float(r["t_over_L"]) <= 0.25]
    if len(plateau_rows) >= 3:
        vals = np.asarray([float(r["m_eff_times_L"]) for r in plateau_rows], dtype=np.float64)
        plateau = {
            "t_min": min(int(r["t"]) for r in plateau_rows),
            "t_max": max(int(r["t"]) for r in plateau_rows),
            "m_eff_L_mean": float(np.mean(vals)),
            "m_eff_L_std": float(np.std(vals, ddof=1)),
        }

    label_parts = [args.label] if args.label else []
    label_parts.append(f"L={L}")
    if args.kappa is not None:
        label_parts.append(f"kappa={args.kappa:g}")
    label = ", ".join(label_parts)
    make_plots(out, meff_rows, label, L)
    write_report(out, cfg_path, manifest, args, raw_n, int(phi.shape[0]), symmetry_max, finite_range, plateau, len(bad), len(meff_rows))

    summary = {
        "input": str(cfg_path),
        "manifest": str(manifest_path) if manifest_path else None,
        "L": L,
        "lambda": args.lam if args.lam is not None else manifest.get("lambda"),
        "kappa": args.kappa if args.kappa is not None else manifest.get("kappa"),
        "raw_configs": raw_n,
        "used_configs": int(phi.shape[0]),
        "burn_configs": args.burn_configs,
        "fold": args.fold,
        "folding_convention": "C_fold[t]=0.5*(C[t]+C[L-t]) for 1<=t<L/2; endpoints kept alone",
        "symmetry_max_relative": symmetry_max,
        "finite_t_min": finite_range[0],
        "finite_t_max": finite_range[1],
        "plateau_first_pass": plateau,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
