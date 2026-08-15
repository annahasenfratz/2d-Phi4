#!/usr/bin/env python3
"""Compare the provisional 32->16 blocker against 16->8 and a cross-only simplification."""

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
from scipy.optimize import minimize


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "perfect_blocking_ising" / "outputs"
SUMMARY_IN = OUT_DIR / "kernel_opt_32_to_16_summary.json"
SUMMARY_JSON = OUT_DIR / "provisional_blocker_check_summary.json"
REPORT_MD = OUT_DIR / "provisional_blocker_check_report.md"
CSV_OUT = OUT_DIR / "provisional_blocker_check_comparison.csv"
PLOTS_PDF = OUT_DIR / "provisional_blocker_check_plots.pdf"

L8_REF = OUT_DIR / "critical_ising_L8.npy"
L16_REF = OUT_DIR / "critical_ising_L16.npy"
L32_REF = OUT_DIR / "critical_ising_L32.npy"

SEED = 20240616
BLOCK_SEEDS = [SEED + 7, SEED + 17, SEED + 27]
N_BLOCK_REPLICA = 4
N_BOOT = 300
N_OPT_REPLICA = 4


def load_opt_module():
    path = ROOT / "perfect_blocking_ising" / "scripts" / "optimize_perfect_blocking.py"
    spec = importlib.util.spec_from_file_location("perfect_blocking_optimize", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import helper module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def compact_stats(stats: dict) -> dict:
    return {k: v for k, v in stats.items() if k != "per_config"}


def load_summary() -> dict:
    return json.loads(SUMMARY_IN.read_text())


def compare_table(true_stats: dict, blocked_stats: dict, keys: list[str], label: str) -> list[dict]:
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
                "setup": label,
                "observable": key,
                "true_mean": t_mean,
                "true_err": t_err,
                "blocked_mean": b_mean,
                "blocked_err": b_err,
                "delta": b_mean - t_mean,
                "z": (b_mean - t_mean) / sigma if sigma else 0.0,
            }
        )
    return rows


def block_many(mod, fine: np.ndarray, alpha: float, w00: float, w01: float, w11: float, seeds: list[int]) -> tuple[np.ndarray, list[dict]]:
    blocks = []
    details = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        uniforms = rng.random((N_BLOCK_REPLICA, len(fine), fine.shape[1] // 2, fine.shape[2] // 2), dtype=np.float32)
        blk = mod.block_centered_3x3(fine, alpha, w00, w01, w11, uniforms)
        blk = np.asarray(blk, dtype=np.float32)
        blocks.append(blk)
        details.append({"seed": seed, "shape": list(blk.shape), "stats": compact_stats(mod.summarize_observables(blk, bootstrap=True, seed=seed, n_boot=N_BOOT))})
    return np.concatenate(blocks, axis=0), details


def summarize(mod, spins: np.ndarray) -> dict:
    return compact_stats(mod.summarize_observables(spins, bootstrap=True, seed=SEED, n_boot=N_BOOT))


def fit_cross_only(mod, fine: np.ndarray, target: np.ndarray, init_alpha: float, init_w00: float) -> dict:
    n = min(len(fine), len(target), 500)
    fine_opt = np.asarray(fine[:n], dtype=np.float32)
    target_opt = np.asarray(target[:n], dtype=np.float32)
    target_mean, target_err = mod.obs_stats(target_opt)
    target_err = np.maximum(target_err, 0.02)
    uniforms = np.random.default_rng(SEED).random((N_OPT_REPLICA, n, fine.shape[1] // 2, fine.shape[2] // 2), dtype=np.float32)

    def weights(x):
        alpha = float(np.exp(x[0]))
        w00 = float(1.0 / (1.0 + np.exp(-x[1])))
        w01 = float((1.0 - w00) / 4.0)
        return alpha, w00, w01, 0.0

    def loss(x):
        alpha, w00, w01, w11 = weights(x)
        blk = mod.block_centered_3x3(fine_opt, alpha, w00, w01, w11, uniforms)
        mean_blk, _ = mod.obs_stats(blk)
        z = (mean_blk - target_mean) / target_err
        return float(np.sum(z**2))

    x0 = np.array([math.log(init_alpha), math.log(init_w00 / (1.0 - init_w00))], dtype=np.float64)
    res = minimize(loss, x0, method="Powell", options={"maxiter": 60, "xtol": 1e-4, "ftol": 1e-4})
    alpha, w00, w01, w11 = weights(res.x)
    return {"alpha": alpha, "w00": w00, "w01": w01, "w11": w11, "loss": float(res.fun), "success": bool(res.success), "message": str(res.message)}


def write_csv(rows: list[dict]) -> None:
    with CSV_OUT.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    os.environ.setdefault("MPLCONFIGDIR", str(OUT_DIR / "_mplcache"))
    (OUT_DIR / "_mplcache").mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    mod = load_opt_module()
    summary = load_summary()
    base = summary["kernel"]
    base_alpha = float(base["alpha"])
    base_w00 = float(base["w00"])
    base_w01 = float(base["w01"])
    base_w11 = float(base["w11"])

    try:
        if not (L8_REF.exists() and L16_REF.exists() and L32_REF.exists()):
            raise FileNotFoundError("missing one of critical_ising_L8/L16/L32.npy")

        true8 = np.load(L8_REF).astype(np.float32)
        true16 = np.load(L16_REF).astype(np.float32)
        fine32 = np.load(L32_REF).astype(np.float32)

        # Fixed provisional kernel on both transfer directions.
        blk16to8, blk16to8_seeds = block_many(mod, true16, base_alpha, base_w00, base_w01, base_w11, BLOCK_SEEDS)
        blk32to16, blk32to16_seeds = block_many(mod, fine32, base_alpha, base_w00, base_w01, base_w11, BLOCK_SEEDS)

        true8_stats = summarize(mod, true8)
        true16_stats = summarize(mod, true16)
        blk16to8_stats = summarize(mod, blk16to8)
        blk32to16_stats = summarize(mod, blk32to16)

        rows = []
        rows.extend(compare_table(true8_stats, blk16to8_stats, ["nn", "diag", "2nn", "nn2", "diag2", "2nn2", "abs_m", "m2"], "fixed_32to16_kernel_on_16to8"))
        rows.extend(compare_table(true16_stats, blk32to16_stats, ["nn", "diag", "2nn", "nn2", "diag2", "2nn2", "abs_m", "m2"], "fixed_32to16_kernel_on_32to16"))

        # Cross-only simplification with base alpha and w00.
        cross_w00 = base_w00
        cross_w01 = (1.0 - cross_w00) / 4.0
        cross_alpha = base_alpha
        cross16to8, cross16to8_seeds = block_many(mod, true16, cross_alpha, cross_w00, cross_w01, 0.0, BLOCK_SEEDS)
        cross32to16, cross32to16_seeds = block_many(mod, fine32, cross_alpha, cross_w00, cross_w01, 0.0, BLOCK_SEEDS)
        cross16to8_stats = summarize(mod, cross16to8)
        cross32to16_stats = summarize(mod, cross32to16)

        rows.extend(compare_table(true8_stats, cross16to8_stats, ["nn", "diag", "2nn", "nn2", "diag2", "2nn2", "abs_m", "m2"], "cross_only_fixed_on_16to8"))
        rows.extend(compare_table(true16_stats, cross32to16_stats, ["nn", "diag", "2nn", "nn2", "diag2", "2nn2", "abs_m", "m2"], "cross_only_fixed_on_32to16"))

        # Optional cross-only refit on 32->16 target, then back-test on 16->8.
        cross_fit = fit_cross_only(mod, fine32, true16, base_alpha, base_w00)
        fit16to8, fit16to8_seeds = block_many(mod, true16, cross_fit["alpha"], cross_fit["w00"], cross_fit["w01"], 0.0, BLOCK_SEEDS)
        fit32to16, fit32to16_seeds = block_many(mod, fine32, cross_fit["alpha"], cross_fit["w00"], cross_fit["w01"], 0.0, BLOCK_SEEDS)
        fit16to8_stats = summarize(mod, fit16to8)
        fit32to16_stats = summarize(mod, fit32to16)

        rows.extend(compare_table(true8_stats, fit16to8_stats, ["nn", "diag", "2nn", "nn2", "diag2", "2nn2", "abs_m", "m2"], "cross_only_refit_on_16to8"))
        rows.extend(compare_table(true16_stats, fit32to16_stats, ["nn", "diag", "2nn", "nn2", "diag2", "2nn2", "abs_m", "m2"], "cross_only_refit_on_32to16"))

        write_csv(rows)

        # tiny report
        best_rows = [
            {"label": "fixed_32to16_kernel_on_16to8", "nn_z": next(r["z"] for r in rows if r["setup"] == "fixed_32to16_kernel_on_16to8" and r["observable"] == "nn")},
            {"label": "fixed_32to16_kernel_on_32to16", "nn_z": next(r["z"] for r in rows if r["setup"] == "fixed_32to16_kernel_on_32to16" and r["observable"] == "nn")},
            {"label": "cross_only_fixed_on_16to8", "nn_z": next(r["z"] for r in rows if r["setup"] == "cross_only_fixed_on_16to8" and r["observable"] == "nn")},
            {"label": "cross_only_fixed_on_32to16", "nn_z": next(r["z"] for r in rows if r["setup"] == "cross_only_fixed_on_32to16" and r["observable"] == "nn")},
            {"label": "cross_only_refit_on_16to8", "nn_z": next(r["z"] for r in rows if r["setup"] == "cross_only_refit_on_16to8" and r["observable"] == "nn")},
            {"label": "cross_only_refit_on_32to16", "nn_z": next(r["z"] for r in rows if r["setup"] == "cross_only_refit_on_32to16" and r["observable"] == "nn")},
        ]

        summary_out = {
            "status": "ok",
            "kernel_source": str(SUMMARY_IN),
            "base_kernel": {"alpha": base_alpha, "w00": base_w00, "w01": base_w01, "w11": base_w11, "normalization": base.get("normalization")},
            "references": {"L8": {"path": str(L8_REF), "shape": list(true8.shape)}, "L16": {"path": str(L16_REF), "shape": list(true16.shape)}, "L32": {"path": str(L32_REF), "shape": list(fine32.shape)}},
            "fixed_kernel": {
                "16to8": compact_stats(mod.summarize_observables(blk16to8, bootstrap=True, seed=SEED, n_boot=N_BOOT)),
                "32to16": compact_stats(mod.summarize_observables(blk32to16, bootstrap=True, seed=SEED + 1, n_boot=N_BOOT)),
                "seeds": {"16to8": blk16to8_seeds, "32to16": blk32to16_seeds},
            },
            "cross_only_fixed": {
                "alpha": cross_alpha,
                "w00": cross_w00,
                "w01": cross_w01,
                "w11": 0.0,
                "16to8": compact_stats(mod.summarize_observables(cross16to8, bootstrap=True, seed=SEED + 2, n_boot=N_BOOT)),
                "32to16": compact_stats(mod.summarize_observables(cross32to16, bootstrap=True, seed=SEED + 3, n_boot=N_BOOT)),
            },
            "cross_only_refit": {
                "alpha": cross_fit["alpha"],
                "w00": cross_fit["w00"],
                "w01": cross_fit["w01"],
                "w11": 0.0,
                "loss": cross_fit["loss"],
                "success": cross_fit["success"],
                "message": cross_fit["message"],
                "16to8": compact_stats(mod.summarize_observables(fit16to8, bootstrap=True, seed=SEED + 4, n_boot=N_BOOT)),
                "32to16": compact_stats(mod.summarize_observables(fit32to16, bootstrap=True, seed=SEED + 5, n_boot=N_BOOT)),
            },
            "summary_rows": best_rows,
            "notes": [
                "The provisional kernel was trained on critical 32x32 -> 16x16.",
                "This check tests whether it back-transfers acceptably to 16x16 -> 8x8.",
                "Cross-only uses w11=0 and w01=(1-w00)/4.",
            ],
        }
        SUMMARY_JSON.write_text(json.dumps(summary_out, indent=2, sort_keys=True) + "\n")

        report = [
            "# Provisional blocker check",
            "",
            f"base kernel alpha={base_alpha:.6f}, w00={base_w00:.6f}, w01={base_w01:.6f}, w11={base_w11:.6e}",
            "",
            "## Fixed provisional kernel",
            f"- 16->8 nn z = {next(r['z'] for r in rows if r['setup']=='fixed_32to16_kernel_on_16to8' and r['observable']=='nn'):+.3f}",
            f"- 32->16 nn z = {next(r['z'] for r in rows if r['setup']=='fixed_32to16_kernel_on_32to16' and r['observable']=='nn'):+.3f}",
            "",
            "## Cross-only fixed kernel",
            f"- alpha = {cross_alpha:.6f}",
            f"- w00 = {cross_w00:.6f}",
            f"- w01 = {cross_w01:.6f}",
            f"- 16->8 nn z = {next(r['z'] for r in rows if r['setup']=='cross_only_fixed_on_16to8' and r['observable']=='nn'):+.3f}",
            f"- 32->16 nn z = {next(r['z'] for r in rows if r['setup']=='cross_only_fixed_on_32to16' and r['observable']=='nn'):+.3f}",
            "",
            "## Cross-only refit",
            f"- alpha = {cross_fit['alpha']:.6f}",
            f"- w00 = {cross_fit['w00']:.6f}",
            f"- w01 = {cross_fit['w01']:.6f}",
            f"- loss = {cross_fit['loss']:.4f}",
            f"- 16->8 nn z = {next(r['z'] for r in rows if r['setup']=='cross_only_refit_on_16to8' and r['observable']=='nn'):+.3f}",
            f"- 32->16 nn z = {next(r['z'] for r in rows if r['setup']=='cross_only_refit_on_32to16' and r['observable']=='nn'):+.3f}",
            "",
            "## Takeaway",
            "Judge equivalence primarily by the joint pattern of nn/diag/2nn and squares on both transfer directions.",
        ]
        REPORT_MD.write_text("\n".join(report) + "\n")

        # simple PDF
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages

        fig, ax = plt.subplots(figsize=(10, 6))
        setups = [s["label"] for s in best_rows]
        nnz = [s["nn_z"] for s in best_rows]
        ax.bar(range(len(setups)), nnz)
        ax.set_xticks(range(len(setups)))
        ax.set_xticklabels(setups, rotation=40, ha="right")
        ax.set_ylabel("nn z-score")
        ax.set_title("NN transfer summary")
        fig.tight_layout()
        with PdfPages(PLOTS_PDF) as pdf:
            pdf.savefig(fig)
            plt.close(fig)

        print(json.dumps({"written": str(SUMMARY_JSON)}, indent=2))
        return 0

    except Exception as exc:
        err = {
            "status": "error",
            "exception": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()},
        }
        SUMMARY_JSON.write_text(json.dumps(err, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"written": str(SUMMARY_JSON), "error": str(exc)}, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
