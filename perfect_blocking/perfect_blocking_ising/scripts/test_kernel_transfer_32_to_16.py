#!/usr/bin/env python3
"""Test the current optimized blocking kernel on critical 32x32 -> 16x16 transfer."""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import os
import sys
import traceback
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "perfect_blocking_ising" / "outputs"
SUMMARY_IN = OUT_DIR / "perfect_blocking_summary.json"
SUMMARY_JSON = OUT_DIR / "kernel_transfer_32_to_16_summary.json"
REPORT_MD = OUT_DIR / "kernel_transfer_32_to_16_report.md"
OBS_CSV = OUT_DIR / "kernel_transfer_32_to_16_observables.csv"
PLOTS_PDF = OUT_DIR / "kernel_transfer_32_to_16_plots.pdf"

L16_REF = OUT_DIR / "critical_ising_L16.npy"
L32_REF = OUT_DIR / "critical_ising_L32.npy"

N_GENERATE = 500
N_THERM = 500
N_SKIP = 6
N_BLOCK_REPLICA = 4
TRUE16_BOOT = 300
BLOCK_BOOT = 300
TRUE16_SEED = 20240616 + 16
L32_SEED = 20240616 + 32
BLOCK_SEEDS = [20240616 + 101, 20240616 + 202, 20240616 + 303]


def load_opt_module():
    path = ROOT / "perfect_blocking_ising" / "scripts" / "optimize_perfect_blocking.py"
    spec = importlib.util.spec_from_file_location("perfect_blocking_optimize", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import helper module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_summary() -> dict:
    return json.loads(SUMMARY_IN.read_text())


def load_kernel(summary: dict) -> dict:
    best = summary["optimization"]["best"]
    return {
        "alpha": float(best["alpha"]),
        "w00": float(best["w00"]),
        "w01": float(best["w01"]),
        "w11": float(best["w11"]),
        "normalization": float(best["normalization"]),
    }


def save_np(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr.astype(np.float32))


def stack_stats(mod, spins: np.ndarray, bootstrap: bool, seed: int, n_boot: int) -> dict:
    return mod.summarize_observables(spins, bootstrap=bootstrap, seed=seed, n_boot=n_boot)


def compact_stats(stats: dict) -> dict:
    return {k: v for k, v in stats.items() if k != "per_config"}


def compare_rows(true_stats: dict, blocked_stats: dict, keys: list[str]) -> list[dict]:
    rows = []
    for key in keys:
        if key in true_stats["means"]:
            t_mean = float(true_stats["means"][key])
            t_err = float(true_stats["errs"][key])
            b_mean = float(blocked_stats["means"][key])
            b_err = float(blocked_stats["errs"][key])
        else:
            t_mean = float(true_stats["extra"][key]["mean"])
            t_err = float(true_stats["extra"][key]["err"])
            b_mean = float(blocked_stats["extra"][key]["mean"])
            b_err = float(blocked_stats["extra"][key]["err"])
        sigma = math.sqrt(t_err**2 + b_err**2) if (t_err or b_err) else 1.0
        rows.append(
            {
                "observable": key,
                "true16_mean": t_mean,
                "true16_err": t_err,
                "blocked32to16_mean": b_mean,
                "blocked32to16_err": b_err,
                "delta": b_mean - t_mean,
                "sigma": sigma,
                "z_blocked_vs_true16": (b_mean - t_mean) / sigma if sigma else 0.0,
            }
        )
    return rows


def write_csv(rows: list[dict]) -> None:
    with OBS_CSV.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_plots(mod, true16: np.ndarray, blocked_by_seed: list[dict], blocked_agg: np.ndarray, rows: list[dict], kernel: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    names = [r["observable"] for r in rows]
    true_means = [r["true16_mean"] for r in rows]
    true_errs = [r["true16_err"] for r in rows]
    blk_means = [r["blocked32to16_mean"] for r in rows]
    blk_errs = [r["blocked32to16_err"] for r in rows]

    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(2, 2)

    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(len(names))
    ax.errorbar(x - 0.12, true_means, yerr=true_errs, fmt="o", label="true L=16")
    ax.errorbar(x + 0.12, blk_means, yerr=blk_errs, fmt="s", label="blocked L=32 -> 16")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30)
    ax.set_ylabel("mean")
    ax.set_title("Observable comparison")
    ax.legend(frameon=False)

    ax = fig.add_subplot(gs[0, 1])
    nn_true = mod.summarize_observables(true16)["per_config"]["nn"]
    nn_blk = mod.summarize_observables(blocked_agg)["per_config"]["nn"]
    ax.hist(nn_true, bins=40, alpha=0.6, density=True, label="true L=16")
    ax.hist(nn_blk, bins=40, alpha=0.6, density=True, label="blocked aggregate")
    ax.set_title("Nearest-neighbor histogram")
    ax.legend(frameon=False)

    ax = fig.add_subplot(gs[1, 0])
    abs_true = mod.summarize_observables(true16)["per_config"]["abs_m"]
    abs_blk = mod.summarize_observables(blocked_agg)["per_config"]["abs_m"]
    ax.hist(abs_true, bins=40, alpha=0.6, density=True, label="true L=16")
    ax.hist(abs_blk, bins=40, alpha=0.6, density=True, label="blocked aggregate")
    ax.set_title("|m| histogram")
    ax.legend(frameon=False)

    ax = fig.add_subplot(gs[1, 1])
    ax.bar([0, 1, 2], [r["z_blocked_vs_true16"] for r in rows[:3]], color=["#4c78a8", "#f58518", "#54a24b"])
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels([r["observable"] for r in rows[:3]])
    ax.set_ylabel("z-score")
    ax.set_title("First three observable z-scores")

    fig.suptitle(
        f"Kernel transfer 32->16 | alpha={kernel['alpha']:.3f}, w00={kernel['w00']:.3f}, "
        f"w01={kernel['w01']:.3f}, w11={kernel['w11']:.3e}",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    with PdfPages(PLOTS_PDF) as pdf:
        pdf.savefig(fig)
        plt.close(fig)


def main() -> int:
    os.environ.setdefault("MPLCONFIGDIR", str(OUT_DIR / "_mplcache"))
    (OUT_DIR / "_mplcache").mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    mod = load_opt_module()
    summary = load_summary()
    kernel = load_kernel(summary)

    try:
        # True critical references: reuse the existing 16x16 ensemble and generate a new 32x32 ensemble.
        if not L16_REF.exists():
            raise FileNotFoundError(L16_REF)
        true16 = np.load(L16_REF).astype(np.float32)
        fine32 = mod.generate_critical_ising(32, N_GENERATE, N_THERM, N_SKIP, L32_SEED, beta=mod.BETA_EXACT)
        fine32_samples = np.asarray(fine32["samples"], dtype=np.float32)
        save_np(L32_REF, fine32_samples)

        true16_stats = stack_stats(mod, true16, bootstrap=True, seed=TRUE16_SEED, n_boot=TRUE16_BOOT)
        fine32_stats = stack_stats(mod, fine32_samples, bootstrap=True, seed=L32_SEED, n_boot=BLOCK_BOOT)

        blocked_by_seed = []
        blocked_samples = []
        for seed in BLOCK_SEEDS:
            rng = np.random.default_rng(seed)
            uniforms = rng.random((N_BLOCK_REPLICA, len(fine32_samples), 16, 16), dtype=np.float32)
            blocked = mod.block_centered_3x3(
                fine32_samples,
                kernel["alpha"],
                kernel["w00"],
                kernel["w01"],
                kernel["w11"],
                uniforms,
            )
            blocked = np.asarray(blocked, dtype=np.float32)
            blocked_samples.append(blocked)
            stats = stack_stats(mod, blocked, bootstrap=True, seed=seed, n_boot=BLOCK_BOOT)
            blocked_by_seed.append(
                {
                    "seed": seed,
                    "shape": list(blocked.shape),
                    "stats": compact_stats(stats),
                }
            )

        blocked_agg = np.concatenate(blocked_samples, axis=0)
        blocked_stats = stack_stats(mod, blocked_agg, bootstrap=True, seed=BLOCK_SEEDS[0], n_boot=BLOCK_BOOT)

        keys = ["nn", "diag", "2nn", "nn2", "diag2", "2nn2", "abs_m", "m2"]
        rows = compare_rows(true16_stats, blocked_stats, keys)
        write_csv(rows)
        make_plots(mod, true16, blocked_by_seed, blocked_agg, rows, kernel)

        summary_out = {
            "status": "ok",
            "beta_exact": mod.BETA_EXACT,
            "kernel_source": str(SUMMARY_IN),
            "kernel": kernel,
            "reference_sources": {
                "true16": {"path": str(L16_REF), "shape": list(true16.shape), "n_configs": int(true16.shape[0])},
                "generated32": {"path": str(L32_REF), "shape": list(fine32_samples.shape), "n_configs": int(fine32_samples.shape[0]), "metadata": fine32["metadata"]},
            },
            "generation": {
                "n_configs": N_GENERATE,
                "n_therm": N_THERM,
                "n_skip": N_SKIP,
                "wolff_seed": L32_SEED,
            },
            "blocking": {
                "replica_count": N_BLOCK_REPLICA,
                "blocking_seeds": BLOCK_SEEDS,
                "combined_shape": list(blocked_agg.shape),
            },
            "observables": {
                "true16": compact_stats(true16_stats),
                "fine32": compact_stats(fine32_stats),
                "blocked32to16": compact_stats(blocked_stats),
                "blocked_by_seed": blocked_by_seed,
                "comparison_table": rows,
            },
            "notes": [
                "The 32x32 critical ensemble was generated specifically for this transfer test at beta_c.",
                "The existing 16x16 critical ensemble from the earlier critical run was used as the finite-volume target.",
                "Blocking is stochastic; several blocking RNG seeds were used to probe stability of the fixed kernel.",
                "This is a transfer/stability check for the previously optimized kernel, not a new optimization run.",
            ],
        }
        SUMMARY_JSON.write_text(json.dumps(summary_out, indent=2, sort_keys=True) + "\n")

        report = []
        report.append("# Kernel Transfer 32 -> 16")
        report.append("")
        report.append(f"beta_c = {mod.BETA_EXACT:.15f}")
        report.append(f"kernel source = {SUMMARY_IN}")
        report.append("")
        report.append("## References")
        report.append(f"- true L=16 target: {L16_REF} shape={list(true16.shape)}")
        report.append(f"- generated critical L=32: {L32_REF} shape={list(fine32_samples.shape)}")
        report.append("")
        report.append("## Optimized kernel")
        report.append(f"- alpha = {kernel['alpha']:.6f}")
        report.append(f"- w00 = {kernel['w00']:.6f}")
        report.append(f"- w01 = {kernel['w01']:.6f}")
        report.append(f"- w11 = {kernel['w11']:.6e}")
        report.append(f"- normalization = {kernel['normalization']:.6f}")
        report.append("")
        report.append("## Observable comparison")
        for row in rows:
            report.append(
                f"- {row['observable']}: true16={row['true16_mean']:.6f}±{row['true16_err']:.6f}, "
                f"blocked={row['blocked32to16_mean']:.6f}±{row['blocked32to16_err']:.6f}, "
                f"Δ={row['delta']:+.6f}, z={row['z_blocked_vs_true16']:+.3f}"
            )
        report.append("")
        report.append("## Blocking-seed stability")
        for item in blocked_by_seed:
            rep = item["stats"]
            report.append(
                f"- seed {item['seed']}: nn={rep['means']['nn']:.6f}±{rep['errs']['nn']:.6f}, "
                f"abs_m={rep['extra']['abs_m']['mean']:.6f}±{rep['extra']['abs_m']['err']:.6f}"
            )
        report.append("")
        report.append("## Interpretation")
        report.append(
            "This is a transfer test of the previously optimized critical kernel from 16->8 onto 32->16, "
            "using 500 critical 32x32 Wolff configs. The blocked 32x32 ensemble is compared to the existing "
            "critical 16x16 reference, and the dependence on stochastic-blocking RNG seed is reported separately."
        )
        REPORT_MD.write_text("\n".join(report) + "\n")

        print(json.dumps({"written": str(SUMMARY_JSON), "kernel": kernel}, indent=2))
        return 0

    except Exception as exc:
        err = {
            "status": "error",
            "exception": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        }
        SUMMARY_JSON.write_text(json.dumps(err, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"written": str(SUMMARY_JSON), "error": str(exc)}, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
