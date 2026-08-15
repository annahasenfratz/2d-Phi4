#!/usr/bin/env python3
"""Proposal-quality 8x8 -> 16x16 upscaling test using the provisional blocking kernel."""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "perfect_blocking_ising" / "outputs_upscale_8_to_16"
SUMMARY_JSON = OUT_DIR / "upscale_summary.json"
REPORT_MD = OUT_DIR / "upscale_report.md"
OBS_CSV = OUT_DIR / "upscale_observables.csv"
GRID_CSV = OUT_DIR / "upscale_grid_results.csv"
PLOTS_PDF = OUT_DIR / "upscale_plots.pdf"

DATA_DIR = ROOT / "perfect_blocking_ising" / "outputs"
L8_REF = DATA_DIR / "critical_ising_L8.npy"
L16_REF = DATA_DIR / "critical_ising_L16.npy"

SEED = 20240616
N_INPUT = 500
N_BLOCK_REPLICA = 3
N_BOOT = 300
TUNE_N = 64

KERNEL = {
    "alpha": 1.75068663213513,
    "w00": 0.2282658690109113,
    "w01": 0.19003036092905146,
    "w11": 0.0029031718182207533,
}

BETA = 0.5 * math.log(1.0 + math.sqrt(2.0))


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


def load_configs() -> tuple[np.ndarray, np.ndarray]:
    if not L8_REF.exists() or not L16_REF.exists():
        raise FileNotFoundError("critical_ising_L8.npy or critical_ising_L16.npy missing")
    return np.load(L8_REF).astype(np.float32), np.load(L16_REF).astype(np.float32)


def make_naive_replication(coarse8: np.ndarray) -> np.ndarray:
    c = np.asarray(coarse8, dtype=np.float32)
    out = np.repeat(np.repeat(c, 2, axis=1), 2, axis=2)
    return out.astype(np.float32)


def make_noisy_replication(coarse8: np.ndarray, eps: float, rng: np.random.Generator) -> np.ndarray:
    fine = make_naive_replication(coarse8)
    flips = rng.random(fine.shape, dtype=np.float32) < eps
    fine = np.where(flips, -fine, fine)
    return fine.astype(np.float32)


def site_to_affected_coarse_map(Lf: int, w00: float, w01: float, w11: float):
    Lc = Lf // 2
    mapping = []
    for x in range(Lf):
        for y in range(Lf):
            entries = []
            for ci in range(Lc):
                cx = 2 * ci
                dx = ((x - cx + Lf // 2) % Lf) - Lf // 2
                if abs(dx) > 1:
                    continue
                for cj in range(Lc):
                    cy = 2 * cj
                    dy = ((y - cy + Lf // 2) % Lf) - Lf // 2
                    if abs(dy) > 1:
                        continue
                    if dx == 0 and dy == 0:
                        coeff = w00
                    elif dx == 0 or dy == 0:
                        coeff = w01
                    else:
                        coeff = w11
                    entries.append((ci, cj, coeff))
            mapping.append(entries)
    return mapping


def compute_p_field(fine: np.ndarray, kernel: dict) -> np.ndarray:
    s = np.asarray(fine, dtype=np.float32)
    squeeze = False
    if s.ndim == 2:
        s = s[None, ...]
        squeeze = True
    center = s[:, 0::2, 0::2]
    up = np.roll(s, 1, axis=1)[:, 0::2, 0::2]
    down = np.roll(s, -1, axis=1)[:, 0::2, 0::2]
    left = np.roll(s, 1, axis=2)[:, 0::2, 0::2]
    right = np.roll(s, -1, axis=2)[:, 0::2, 0::2]
    ul = np.roll(np.roll(s, 1, axis=1), 1, axis=2)[:, 0::2, 0::2]
    ur = np.roll(np.roll(s, 1, axis=1), -1, axis=2)[:, 0::2, 0::2]
    dl = np.roll(np.roll(s, -1, axis=1), 1, axis=2)[:, 0::2, 0::2]
    dr = np.roll(np.roll(s, -1, axis=1), -1, axis=2)[:, 0::2, 0::2]
    out = (
        kernel["w00"] * center
        + kernel["w01"] * (up + down + left + right)
        + kernel["w11"] * (ul + ur + dl + dr)
    ).astype(np.float32)
    if squeeze:
        return out[0]
    return out


def logistic(x: float | np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def metropolis_relax(
    coarse_t: np.ndarray,
    fine_init: np.ndarray,
    beta: float,
    kernel: dict,
    lambda_block: float,
    n_sweeps: int,
    rng: np.random.Generator,
    mapping,
) -> tuple[np.ndarray, dict]:
    """Single-spin Metropolis on S_eff = S_Ising + lambda_block * alpha * (t - p)^2."""

    s = np.asarray(fine_init, dtype=np.float32).copy()
    t = np.asarray(coarse_t, dtype=np.float32)
    p = compute_p_field(s, kernel)
    Lf = s.shape[1]
    Lc = t.shape[1]
    n_accept = 0
    n_total = 0

    for _ in range(n_sweeps):
        for idx in rng.permutation(Lf * Lf):
            i = int(idx // Lf)
            j = int(idx % Lf)
            spin = s[i, j]
            nn = s[(i + 1) % Lf, j] + s[(i - 1) % Lf, j] + s[i, (j + 1) % Lf] + s[i, (j - 1) % Lf]
            dS_ising = 2.0 * beta * spin * nn
            dS_block = 0.0
            for ci, cj, coeff in mapping[i * Lf + j]:
                p_old = float(p[ci, cj])
                dp = -2.0 * spin * coeff
                p_new = p_old + dp
                dS_block += lambda_block * kernel["alpha"] * ((t[ci, cj] - p_new) ** 2 - (t[ci, cj] - p_old) ** 2)
            dS = dS_ising + dS_block
            n_total += 1
            if dS <= 0.0 or rng.random() < math.exp(-float(dS)):
                s[i, j] = -spin
                for ci, cj, coeff in mapping[i * Lf + j]:
                    p[ci, cj] += -2.0 * spin * coeff
                n_accept += 1

    return s.astype(np.float32), {"acceptance": float(n_accept / max(1, n_total)), "n_accept": int(n_accept), "n_total": int(n_total)}


def block_16_to_8(mod, fine16: np.ndarray, kernel: dict, rng: np.random.Generator, replicas: int = 1) -> np.ndarray:
    uniforms = rng.random((replicas, len(fine16), 8, 8), dtype=np.float32)
    blk = mod.block_centered_3x3(fine16, kernel["alpha"], kernel["w00"], kernel["w01"], kernel["w11"], uniforms)
    return np.asarray(blk, dtype=np.float32)


def sign_nozero(x: np.ndarray) -> np.ndarray:
    out = np.sign(x)
    out[out == 0] = 1.0
    return out


def compare_stats(mod, target: np.ndarray, proposal: np.ndarray, obs_keys: list[str]) -> tuple[list[dict], dict]:
    target_stats = mod.summarize_observables(target, bootstrap=True, seed=SEED, n_boot=N_BOOT)
    prop_stats = mod.summarize_observables(proposal, bootstrap=True, seed=SEED + 1, n_boot=N_BOOT)
    rows = []
    abs_z = []
    for key in obs_keys:
        if key in target_stats["means"]:
            t_mean = float(target_stats["means"][key])
            t_err = float(target_stats["errs"][key])
            p_mean = float(prop_stats["means"][key])
            p_err = float(prop_stats["errs"][key])
        else:
            t_mean = float(target_stats["extra"][key]["mean"])
            t_err = float(target_stats["extra"][key]["err"])
            p_mean = float(prop_stats["extra"][key]["mean"])
            p_err = float(prop_stats["extra"][key]["err"])
        sigma = math.sqrt(t_err**2 + p_err**2) if (t_err or p_err) else 1.0
        z = (p_mean - t_mean) / sigma if sigma else 0.0
        rows.append(
            {
                "observable": key,
                "true_mean": t_mean,
                "true_err": t_err,
                "proposal_mean": p_mean,
                "proposal_err": p_err,
                "delta": p_mean - t_mean,
                "rel_delta": (p_mean - t_mean) / t_mean if t_mean else 0.0,
                "z": z,
            }
        )
        abs_z.append(abs(z))
    stats = {
        "mean_abs_z": float(np.mean(abs_z)) if abs_z else 0.0,
        "max_abs_z": float(np.max(abs_z)) if abs_z else 0.0,
        "true": compact_stats(target_stats),
        "proposal": compact_stats(prop_stats),
        "rows": rows,
    }
    return rows, stats


def pairwise_metrics(input8: np.ndarray, reblocked8: np.ndarray, generated16: np.ndarray) -> dict:
    input8 = np.asarray(input8, dtype=np.float32)
    reblocked8 = np.asarray(reblocked8, dtype=np.float32)
    generated16 = np.asarray(generated16, dtype=np.float32)
    if reblocked8.shape[0] != input8.shape[0] and reblocked8.shape[0] % input8.shape[0] == 0:
        rep = reblocked8.shape[0] // input8.shape[0]
        input_rep = np.repeat(input8, rep, axis=0)
    else:
        input_rep = input8[: reblocked8.shape[0]]
    overlap = np.mean(input_rep * reblocked8, axis=(1, 2))
    agree = np.mean(input_rep == reblocked8, axis=(1, 2))
    in_sign = sign_nozero(input8.mean(axis=(1, 2)))
    gen_sign = sign_nozero(generated16.mean(axis=(1, 2)))
    sign_agree = np.mean(in_sign == gen_sign)
    return {
        "overlap_mean": float(np.mean(overlap)),
        "overlap_err": float(np.std(overlap, ddof=1) / math.sqrt(len(overlap))) if len(overlap) > 1 else 0.0,
        "agreement_mean": float(np.mean(agree)),
        "agreement_err": float(np.std(agree, ddof=1) / math.sqrt(len(agree))) if len(agree) > 1 else 0.0,
        "sign_agreement": float(sign_agree),
    }


def score_candidate(gen_stats: dict, reb_stats: dict, overlap: float) -> float:
    return gen_stats["mean_abs_z"] + reb_stats["mean_abs_z"] + 0.5 * abs(overlap - 0.8)


def grid_search_relaxation(mod, input8: np.ndarray, true16: np.ndarray, true8: np.ndarray, mapping) -> list[dict]:
    tune_input = input8[:TUNE_N]
    tune_true16 = true16[:TUNE_N]
    tune_true8 = true8[:TUNE_N]
    results = []
    eps_grid = [0.05, 0.10, 0.20]
    lambda_grid = [0.1, 0.3, 1.0, 3.0]
    sweeps_grid = [10, 50, 100, 200]
    for eps in eps_grid:
        for lam in lambda_grid:
            for sweeps in sweeps_grid:
                rng = np.random.default_rng(SEED + int(eps * 1000) + int(lam * 100) + sweeps)
                generated = []
                accept_rates = []
                for idx, coarse in enumerate(tune_input):
                    rel_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))
                    init = make_noisy_replication(coarse[None, ...], eps, rel_rng)[0]
                    relaxed, diag = metropolis_relax(coarse, init, BETA, KERNEL, lam, sweeps, rel_rng, mapping)
                    generated.append(relaxed)
                    accept_rates.append(diag["acceptance"])
                generated = np.asarray(generated, dtype=np.float32)
                reblocked = block_16_to_8(mod, generated, KERNEL, np.random.default_rng(SEED + 99), replicas=1)
                gen_rows, gen_stats = compare_stats(mod, tune_true16, generated, ["nn", "diag", "2nn", "nn2", "diag2", "2nn2", "abs_m", "m2"])
                reb_rows, reb_stats = compare_stats(mod, tune_true8, reblocked, ["nn", "diag", "2nn", "nn2", "diag2", "2nn2", "abs_m", "m2"])
                overlap_metrics = pairwise_metrics(tune_input, reblocked, generated)
                results.append(
                    {
                        "proposal": "relaxation",
                        "eps": eps,
                        "lambda_block": lam,
                        "n_sweeps": sweeps,
                        "grid_n": int(len(tune_input)),
                        "mean_acceptance": float(np.mean(accept_rates)),
                        "generated_mean_abs_z": gen_stats["mean_abs_z"],
                        "reblocked_mean_abs_z": reb_stats["mean_abs_z"],
                        "overlap_mean": overlap_metrics["overlap_mean"],
                        "agreement_mean": overlap_metrics["agreement_mean"],
                        "sign_agreement": overlap_metrics["sign_agreement"],
                        "score": score_candidate(gen_stats, reb_stats, overlap_metrics["overlap_mean"]),
                    }
                )
    return results


def evaluate_method(mod, input8: np.ndarray, true16: np.ndarray, true8: np.ndarray, proposal_name: str, generated16: np.ndarray) -> dict:
    rows16, gen_stats = compare_stats(mod, true16, generated16, ["nn", "diag", "2nn", "nn2", "diag2", "2nn2", "abs_m", "m2"])
    reblocked = block_16_to_8(mod, generated16, KERNEL, np.random.default_rng(SEED + 123), replicas=N_BLOCK_REPLICA)
    rows8, reb_stats = compare_stats(mod, true8, reblocked, ["nn", "diag", "2nn", "nn2", "diag2", "2nn2", "abs_m", "m2"])
    pair = pairwise_metrics(input8, reblocked, generated16)
    result = {
        "proposal": proposal_name,
        "generated16": {"rows": rows16, **gen_stats},
        "reblocked8_vs_true8": {"rows": rows8, **reb_stats},
        "pairwise": pair,
        "score": score_candidate(gen_stats, reb_stats, pair["overlap_mean"]),
        "generated16_shape": list(generated16.shape),
        "reblocked8_shape": list(reblocked.shape),
        "reblocked8": compact_stats(mod.summarize_observables(reblocked, bootstrap=True, seed=SEED + 2, n_boot=N_BOOT)),
        "example_generated16": generated16[0].tolist(),
        "example_reblocked8": reblocked[0].tolist(),
        "example_input8": input8[0].tolist(),
    }
    return result


def write_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def select_best(results: list[dict]) -> dict:
    return min(results, key=lambda r: r["score"])


def make_plots(mod, input8: np.ndarray, true16: np.ndarray, true8: np.ndarray, methods: list[dict], best_method: dict, grid_results: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(2, 2)

    ax = fig.add_subplot(gs[0, 0])
    method_labels = [m["proposal"] for m in methods]
    scores = [m["score"] for m in methods]
    ax.bar(np.arange(len(method_labels)), scores)
    ax.set_xticks(np.arange(len(method_labels)))
    ax.set_xticklabels(method_labels, rotation=25, ha="right")
    ax.set_title("Proposal scores")
    ax.set_ylabel("lower is better")

    ax = fig.add_subplot(gs[0, 1])
    if grid_results:
        by_sweep = sorted(grid_results, key=lambda r: (r["lambda_block"], r["n_sweeps"], r["eps"]))
        xs = np.arange(len(by_sweep))
        ax.plot(xs, [r["score"] for r in by_sweep], marker="o", ms=3, lw=1)
        ax.set_title("Relaxation grid score")
        ax.set_ylabel("score")
        ax.set_xlabel("grid point")

    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(input8[0], cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_title("Input L=8 example")
    ax.axis("off")

    ax = fig.add_subplot(gs[1, 1])
    if best_method:
        ax.imshow(np.asarray(best_method["example_generated16"], dtype=np.float32), cmap="coolwarm", vmin=-1, vmax=1)
    else:
        ax.imshow(true16[0], cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_title("Best generated L=16 example")
    ax.axis("off")

    fig.tight_layout()
    with PdfPages(PLOTS_PDF) as pdf:
        pdf.savefig(fig)
        plt.close(fig)


def main() -> int:
    os.environ.setdefault("MPLCONFIGDIR", str(OUT_DIR / "_mplcache"))
    (OUT_DIR / "_mplcache").mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    mod = load_opt_module()

    try:
        true8, true16 = load_configs()
        input8 = true8[:N_INPUT]
        true16 = true16[:N_INPUT]
        mapping = site_to_affected_coarse_map(16, KERNEL["w00"], KERNEL["w01"], KERNEL["w11"])

        methods = []
        observable_rows = []
        grid_rows = []

        # A. naive replication
        gen_naive = make_naive_replication(input8)
        method_naive = evaluate_method(mod, input8, true16, true8[:N_INPUT], "naive_replication", gen_naive)
        methods.append(method_naive)
        for r in method_naive["generated16"]["rows"]:
            r = dict(r)
            r["proposal"] = "naive_replication"
            r["comparison"] = "generated16_vs_true16"
            observable_rows.append(r)
        for r in method_naive["reblocked8_vs_true8"]["rows"]:
            r = dict(r)
            r["proposal"] = "naive_replication"
            r["comparison"] = "reblocked8_vs_true8"
            observable_rows.append(r)

        # B. noisy replication grid
        eps_grid = [0.02, 0.05, 0.10, 0.15, 0.20]
        noisy_candidates = []
        noisy_eval_rows = []
        for eps in eps_grid:
            rng = np.random.default_rng(SEED + int(eps * 1000))
            gen = make_noisy_replication(input8, eps, rng)
            method = evaluate_method(mod, input8, true16, true8[:N_INPUT], f"noisy_replication_eps{eps:.2f}", gen)
            noisy_candidates.append(method)
            noisy_eval_rows.append({"proposal": "noisy_replication", "eps": eps, "score": method["score"], "generated_mean_abs_z": method["generated16"]["mean_abs_z"], "reblocked_mean_abs_z": method["reblocked8_vs_true8"]["mean_abs_z"], "overlap_mean": method["pairwise"]["overlap_mean"], "agreement_mean": method["pairwise"]["agreement_mean"], "sign_agreement": method["pairwise"]["sign_agreement"]})
            for r in method["generated16"]["rows"]:
                rr = dict(r)
                rr["proposal"] = f"noisy_replication_eps{eps:.2f}"
                rr["comparison"] = "generated16_vs_true16"
                observable_rows.append(rr)
            for r in method["reblocked8_vs_true8"]["rows"]:
                rr = dict(r)
                rr["proposal"] = f"noisy_replication_eps{eps:.2f}"
                rr["comparison"] = "reblocked8_vs_true8"
                observable_rows.append(rr)
        methods.extend(noisy_candidates)

        # C. constrained relaxation grid on a subset
        grid_rows = grid_search_relaxation(mod, input8, true16, true8[:N_INPUT], mapping)
        grid_rows_sorted = sorted(grid_rows, key=lambda r: r["score"])
        top_grid = grid_rows_sorted[:3]

        relaxed_methods = []
        for idx, g in enumerate(top_grid):
            rng = np.random.default_rng(SEED + 500 + idx)
            generated = []
            accept_rates = []
            for coarse in input8:
                init = make_noisy_replication(coarse[None, ...], g["eps"], np.random.default_rng(rng.integers(0, 2**32 - 1)))[0]
                relaxed, diag = metropolis_relax(coarse, init, BETA, KERNEL, g["lambda_block"], g["n_sweeps"], np.random.default_rng(rng.integers(0, 2**32 - 1)), mapping)
                generated.append(relaxed)
                accept_rates.append(diag["acceptance"])
            generated = np.asarray(generated, dtype=np.float32)
            method = evaluate_method(mod, input8, true16, true8[:N_INPUT], f"relax_eps{g['eps']:.2f}_lam{g['lambda_block']:.1f}_sw{g['n_sweeps']}", generated)
            method["mean_relax_acceptance"] = float(np.mean(accept_rates))
            method["grid_params"] = {"eps": g["eps"], "lambda_block": g["lambda_block"], "n_sweeps": g["n_sweeps"], "grid_score": g["score"]}
            relaxed_methods.append(method)
            for r in method["generated16"]["rows"]:
                rr = dict(r)
                rr["proposal"] = method["proposal"]
                rr["comparison"] = "generated16_vs_true16"
                observable_rows.append(rr)
            for r in method["reblocked8_vs_true8"]["rows"]:
                rr = dict(r)
                rr["proposal"] = method["proposal"]
                rr["comparison"] = "reblocked8_vs_true8"
                observable_rows.append(rr)

        methods.extend(relaxed_methods)

        best_method = select_best(methods)

        # final summary and outputs
        write_csv(observable_rows, OBS_CSV)
        write_csv(grid_rows, GRID_CSV)
        make_plots(mod, input8, true16, true8[:N_INPUT], methods, best_method, grid_rows)

        summary = {
            "status": "ok",
            "beta": BETA,
            "kernel": KERNEL,
            "references": {
                "true8": {"path": str(L8_REF), "shape": list(true8.shape)},
                "true16": {"path": str(L16_REF), "shape": list(true16.shape)},
            },
            "methods": methods,
            "grid_results": grid_rows,
            "best_method": best_method,
            "decision": {
                "exact_sampling": False,
                "accept_reject": False,
                "proposal_quality_only": True,
                "notes": [
                    "This is not exact sampling.",
                    "No A/R correction is used yet.",
                    "The eventual A/R test requires q(s_f|t_c) or a detailed-balance-preserving transition.",
                ],
            },
            "proposals": {
                "A_naive_replication": "s[2i+a,2j+b] = t[i,j]",
                "B_noisy_replication": "replication plus iid spin flips with eps",
                "C_relaxation": "single-spin Metropolis on S_Ising + lambda_block * alpha * (t - p)^2",
            },
        }
        SUMMARY_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

        report = [
            "# 8 -> 16 proposal-quality upscaling test",
            "",
            f"beta_c = {BETA:.15f}",
            "",
            "This is not exact sampling.",
            "No A/R correction is used yet.",
            "This only tests whether the provisional blocker gives a useful conditional proposal.",
            "The eventual A/R test requires the proposal probability q(s_f | t_c), or an exact Markov transition satisfying detailed balance.",
            "",
            "## Best method",
            f"- {best_method['proposal']}",
            f"- score = {best_method['score']:.4f}",
            f"- generated16 mean abs z = {best_method['generated16']['mean_abs_z']:.4f}",
            f"- reblocked8 mean abs z = {best_method['reblocked8_vs_true8']['mean_abs_z']:.4f}",
            f"- overlap mean = {best_method['pairwise']['overlap_mean']:.4f}",
            f"- agreement mean = {best_method['pairwise']['agreement_mean']:.4f}",
            f"- sign agreement = {best_method['pairwise']['sign_agreement']:.4f}",
            "",
            "## Method summaries",
        ]
        for m in methods:
            report.append(
                f"- {m['proposal']}: score={m['score']:.4f}, gen16 mean abs z={m['generated16']['mean_abs_z']:.4f}, "
                f"reblocked8 mean abs z={m['reblocked8_vs_true8']['mean_abs_z']:.4f}, overlap={m['pairwise']['overlap_mean']:.4f}, "
                f"agreement={m['pairwise']['agreement_mean']:.4f}"
            )
        report.append("")
        report.append("## Grid search")
        for r in grid_rows_sorted[:10]:
            report.append(
                f"- eps={r['eps']:.2f}, lambda={r['lambda_block']:.1f}, sweeps={r['n_sweeps']}: "
                f"score={r['score']:.4f}, overlap={r['overlap_mean']:.4f}, accept={r['mean_acceptance']:.3f}"
            )
        report.append("")
        report.append("## Decision")
        report.append(
            "Pick the best non-neural proposal by generated16 observables close to true16, reblocked8 close to true8, "
            "high input/reblocked overlap, but not trivially frozen. If no hand-built proposal works, move to a learned conditional network."
        )
        REPORT_MD.write_text("\n".join(report) + "\n")

        print(json.dumps({"written": str(SUMMARY_JSON), "best": best_method["proposal"]}, indent=2))
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
