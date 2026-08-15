#!/usr/bin/env python3
"""Optimize the centered 3x3 blocking kernel on critical 32x32 -> 16x16 transfer."""

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
from scipy.optimize import differential_evolution, minimize


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "perfect_blocking_ising" / "outputs"
SUMMARY_IN = OUT_DIR / "perfect_blocking_summary.json"
SUMMARY_JSON = OUT_DIR / "kernel_opt_32_to_16_summary.json"
REPORT_MD = OUT_DIR / "kernel_opt_32_to_16_report.md"
OBS_CSV = OUT_DIR / "kernel_opt_32_to_16_observables.csv"
PLOTS_PDF = OUT_DIR / "kernel_opt_32_to_16_plots.pdf"

L16_REF = OUT_DIR / "critical_ising_L16.npy"
L32_REF = OUT_DIR / "critical_ising_L32.npy"

SEED = 20240616
N_OPT_FINE = 500
N_OPT_REPLICA = 4
N_VAL_REPLICA = 8
N_VALIDATE = 500
BLOCK_SEEDS = [SEED + 11, SEED + 22, SEED + 33]


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


def compact_stats(stats: dict) -> dict:
    return {k: v for k, v in stats.items() if k != "per_config"}


def params_to_weights(x: np.ndarray) -> tuple[float, float, float, float]:
    x = np.asarray(x, dtype=np.float64)
    log_alpha = float(x[0])
    logits = np.array(x[1:4], dtype=np.float64)
    logits = logits - logits.max()
    q = np.exp(logits)
    q = q / q.sum()
    alpha = float(np.exp(log_alpha))
    w00 = float(q[0])
    w01 = float(q[1] / 4.0)
    w11 = float(q[2] / 4.0)
    return alpha, w00, w01, w11


def obs_stats(mod, spins: np.ndarray, bootstrap: bool, seed: int, n_boot: int) -> dict:
    return mod.summarize_observables(spins, bootstrap=bootstrap, seed=seed, n_boot=n_boot)


def objective_rows(true_stats: dict, blocked_stats: dict, keys: list[str]) -> list[dict]:
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
        sigma = max(math.sqrt(t_err**2), 0.02)
        z = (b_mean - t_mean) / sigma
        rows.append(
            {
                "observable": key,
                "true16_mean": t_mean,
                "true16_err": t_err,
                "blocked32to16_mean": b_mean,
                "blocked32to16_err": b_err,
                "delta": b_mean - t_mean,
                "sigma": sigma,
                "z_over_floor": z,
            }
        )
    return rows


def write_csv(rows: list[dict]) -> None:
    with OBS_CSV.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def optimize_kernel(mod, fine32: np.ndarray, true16: np.ndarray) -> dict:
    n_opt = min(N_OPT_FINE, len(fine32), len(true16))
    fine_opt = np.asarray(fine32[:n_opt], dtype=np.float32)
    true_opt = np.asarray(true16[:n_opt], dtype=np.float32)
    rng = np.random.default_rng(SEED)
    uniforms = rng.random((N_OPT_REPLICA, n_opt, 16, 16), dtype=np.float32)
    target_mean, target_err = mod.obs_stats(true_opt)
    target_err = np.maximum(target_err, 0.02)

    cache: dict[str, dict] = {}

    def evaluate(x: np.ndarray) -> float:
        alpha, w00, w01, w11 = params_to_weights(x)
        blocked = mod.block_centered_3x3(fine_opt, alpha, w00, w01, w11, uniforms)
        mean_block, _ = mod.obs_stats(blocked)
        z = (mean_block - target_mean) / target_err
        loss = float(np.sum(z**2))
        cache["last"] = {
            "alpha": alpha,
            "w00": w00,
            "w01": w01,
            "w11": w11,
            "loss": loss,
            "blocked_mean": mean_block.tolist(),
            "z": z.tolist(),
        }
        return loss

    history = []

    def callback(xk, convergence=None):
        history.append({"x": [float(v) for v in xk], "loss": float(evaluate(xk))})
        return False

    de = differential_evolution(
        evaluate,
        bounds=[(-4.0, 4.0), (-4.0, 4.0), (-4.0, 4.0), (-4.0, 4.0)],
        strategy="best1bin",
        popsize=10,
        maxiter=20,
        tol=0.01,
        mutation=(0.5, 1.0),
        recombination=0.7,
        seed=SEED,
        polish=False,
        updating="deferred",
        workers=1,
        callback=callback,
    )
    local = minimize(evaluate, de.x, method="Powell", options={"maxiter": 80, "xtol": 1e-4, "ftol": 1e-4})

    best_x = np.asarray(local.x if local.fun <= de.fun else de.x, dtype=np.float64)
    best_loss = float(min(local.fun, de.fun))
    alpha, w00, w01, w11 = params_to_weights(best_x)

    validate_uniforms_a = np.random.default_rng(SEED + 1).random((N_VAL_REPLICA, len(fine32), 16, 16), dtype=np.float32)
    validate_uniforms_b = np.random.default_rng(SEED + 2).random((N_VAL_REPLICA, len(fine32), 16, 16), dtype=np.float32)
    val_a = mod.block_centered_3x3(fine32, alpha, w00, w01, w11, validate_uniforms_a)
    val_b = mod.block_centered_3x3(fine32, alpha, w00, w01, w11, validate_uniforms_b)
    mean_a, err_a = mod.obs_stats(val_a)
    mean_b, err_b = mod.obs_stats(val_b)
    loss_a = float(np.sum(((mean_a - target_mean) / target_err) ** 2))
    loss_b = float(np.sum(((mean_b - target_mean) / target_err) ** 2))

    return {
        "best_x": [float(v) for v in best_x],
        "best_loss": best_loss,
        "alpha": float(alpha),
        "weights": {"w00": float(w00), "w01": float(w01), "w11": float(w11), "normalization": float(w00 + 4.0 * w01 + 4.0 * w11)},
        "target_mean": target_mean.tolist(),
        "target_err": target_err.tolist(),
        "optimization_history": history,
        "de_result": {"x": [float(v) for v in de.x], "fun": float(de.fun), "nit": int(de.nit)},
        "powell_result": {"x": [float(v) for v in local.x], "fun": float(local.fun), "nit": int(local.nit), "success": bool(local.success), "message": str(local.message)},
        "validation": {
            "seed_A": {"mean": mean_a.tolist(), "err": err_a.tolist(), "loss": loss_a},
            "seed_B": {"mean": mean_b.tolist(), "err": err_b.tolist(), "loss": loss_b},
        },
    }


def compare_table(true16: np.ndarray, blocked: np.ndarray, mod) -> list[dict]:
    true_stats = mod.summarize_observables(true16, bootstrap=True, seed=SEED)
    blk_stats = mod.summarize_observables(blocked, bootstrap=True, seed=SEED + 1)
    rows = []
    keys = ["nn", "diag", "2nn", "nn2", "diag2", "2nn2", "abs_m", "m2"]
    for key in keys:
        if key in true_stats["means"]:
            t_mean = float(true_stats["means"][key])
            t_err = float(true_stats["errs"][key])
            b_mean = float(blk_stats["means"][key])
            b_err = float(blk_stats["errs"][key])
        else:
            t_mean = float(true_stats["extra"][key]["mean"])
            t_err = float(true_stats["extra"][key]["err"])
            b_mean = float(blk_stats["extra"][key]["mean"])
            b_err = float(blk_stats["extra"][key]["err"])
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


def save_outputs(mod, true16: np.ndarray, fine32: np.ndarray, opt: dict) -> None:
    alpha = opt["alpha"]
    w00 = opt["weights"]["w00"]
    w01 = opt["weights"]["w01"]
    w11 = opt["weights"]["w11"]
    rng = np.random.default_rng(SEED + 99)
    blocked = mod.block_centered_3x3(fine32, alpha, w00, w01, w11, rng.random((N_VAL_REPLICA, len(fine32), 16, 16), dtype=np.float32))
    blocked2 = mod.block_centered_3x3(fine32, alpha, w00, w01, w11, np.random.default_rng(SEED + 100).random((N_VAL_REPLICA, len(fine32), 16, 16), dtype=np.float32))

    rows = compare_table(true16, blocked, mod)
    write_csv(rows)

    figpath = PLOTS_PDF
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    true_stats = mod.summarize_observables(true16, bootstrap=True, seed=SEED)
    blk_stats = mod.summarize_observables(blocked, bootstrap=True, seed=SEED + 1)

    def mean_err(stats: dict, key: str) -> tuple[float, float]:
        if key in stats["means"]:
            return float(stats["means"][key]), float(stats["errs"][key])
        return float(stats["extra"][key]["mean"]), float(stats["extra"][key]["err"])

    fig = plt.figure(figsize=(11, 8))
    gs = fig.add_gridspec(2, 2)
    ax = fig.add_subplot(gs[0, 0])
    names = [r["observable"] for r in rows]
    x = np.arange(len(names))
    ax.errorbar(x - 0.12, [mean_err(true_stats, k)[0] for k in names],
                yerr=[mean_err(true_stats, k)[1] for k in names], fmt="o", label="true L=16")
    ax.errorbar(x + 0.12, [mean_err(blk_stats, k)[0] for k in names],
                yerr=[mean_err(blk_stats, k)[1] for k in names], fmt="s", label="blocked L=32 -> 16")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30)
    ax.legend(frameon=False)
    ax.set_ylabel("mean")
    ax.set_title("Observable comparison")

    ax = fig.add_subplot(gs[0, 1])
    ax.hist(true_stats["per_config"]["nn"], bins=40, density=True, alpha=0.6, label="true L=16")
    ax.hist(mod.summarize_observables(blocked2)["per_config"]["nn"], bins=40, density=True, alpha=0.6, label="blocked seed B")
    ax.legend(frameon=False)
    ax.set_title("NN histogram")

    ax = fig.add_subplot(gs[1, 0])
    ax.hist(true_stats["per_config"]["abs_m"], bins=40, density=True, alpha=0.6, label="true L=16")
    ax.hist(mod.summarize_observables(blocked2)["per_config"]["abs_m"], bins=40, density=True, alpha=0.6, label="blocked seed B")
    ax.legend(frameon=False)
    ax.set_title("|m| histogram")

    ax = fig.add_subplot(gs[1, 1])
    hist = opt["optimization_history"]
    ax.plot([h["loss"] for h in hist], lw=1.5)
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss")
    ax.set_title("Optimization trace")

    fig.tight_layout()
    with PdfPages(figpath) as pdf:
        pdf.savefig(fig)
        plt.close(fig)

    summary = {
        "status": "ok",
        "beta_exact": mod.BETA_EXACT,
        "kernel": opt["weights"] | {"alpha": opt["alpha"], "loss": opt["best_loss"]},
        "optimization": {
            "objective": "diagonal chi2 on [nn, diag, 2nn, nn2, diag2, 2nn2] using true L=16 errors with floor 0.02",
            "target_n": N_OPT_FINE,
            "replicas": N_OPT_REPLICA,
            "de_result": opt["de_result"],
            "powell_result": opt["powell_result"],
            "validation": opt["validation"],
            "history": opt["optimization_history"],
        },
        "references": {
            "true16": {"path": str(L16_REF), "shape": list(true16.shape)},
            "fine32": {"path": str(L32_REF), "shape": list(fine32.shape)},
        },
        "comparison_table": rows,
        "notes": [
            "This run re-optimizes the kernel on critical 32x32 -> 16x16 transfer using 500 configurations.",
            "The validation loss is evaluated on two independent blocking RNG seeds.",
            "The result should be judged by blocked32->16 vs true16 observables, not by the old 16->8 fit.",
        ],
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    report = []
    report.append("# Kernel Optimization 32 -> 16")
    report.append("")
    report.append(f"beta_c = {mod.BETA_EXACT:.15f}")
    report.append("")
    report.append("## References")
    report.append(f"- true L=16: {L16_REF} shape={list(true16.shape)}")
    report.append(f"- critical L=32: {L32_REF} shape={list(fine32.shape)}")
    report.append("")
    report.append("## Optimized kernel")
    report.append(f"- alpha = {opt['alpha']:.6f}")
    report.append(f"- w00 = {opt['weights']['w00']:.6f}")
    report.append(f"- w01 = {opt['weights']['w01']:.6f}")
    report.append(f"- w11 = {opt['weights']['w11']:.6e}")
    report.append(f"- normalization = {opt['weights']['normalization']:.6f}")
    report.append("")
    report.append("## Validation")
    report.append(f"- seed A loss = {opt['validation']['seed_A']['loss']:.4f}")
    report.append(f"- seed B loss = {opt['validation']['seed_B']['loss']:.4f}")
    report.append("")
    report.append("## Observable comparison")
    for row in rows:
        report.append(
            f"- {row['observable']}: true16={row['true16_mean']:.6f}±{row['true16_err']:.6f}, "
            f"blocked={row['blocked32to16_mean']:.6f}±{row['blocked32to16_err']:.6f}, "
            f"Δ={row['delta']:+.6f}, z={row['z_blocked_vs_true16']:+.3f}"
        )
    report.append("")
    report.append("## Interpretation")
    report.append(
        "This is a fresh re-optimization on critical 32x32 -> 16x16 transfer with 500 configurations. "
        "Use the validation and transfer observables to judge whether the new kernel is stable enough to carry forward."
    )
    REPORT_MD.write_text("\n".join(report) + "\n")


def main() -> int:
    os.environ.setdefault("MPLCONFIGDIR", str(OUT_DIR / "_mplcache"))
    (OUT_DIR / "_mplcache").mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    mod = load_opt_module()
    summary = load_summary()

    try:
        if not L16_REF.exists():
            raise FileNotFoundError(L16_REF)
        if not L32_REF.exists():
            raise FileNotFoundError(L32_REF)

        true16 = np.load(L16_REF).astype(np.float32)
        fine32 = np.load(L32_REF).astype(np.float32)
        opt = optimize_kernel(mod, fine32, true16)
        save_outputs(mod, true16, fine32, opt)
        print(json.dumps({"written": str(SUMMARY_JSON), "kernel": opt["weights"] | {"alpha": opt["alpha"]}}, indent=2))
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
