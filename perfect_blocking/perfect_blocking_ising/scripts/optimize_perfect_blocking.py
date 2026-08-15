#!/usr/bin/env python3
"""Optimize a stochastic centered 3x3 blocking kernel for critical 2D Ising."""

from __future__ import annotations

import csv
import json
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution, minimize


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "perfect_blocking_ising" / "outputs"
REPORT_MD = OUT_DIR / "perfect_blocking_report.md"
SUMMARY_JSON = OUT_DIR / "perfect_blocking_summary.json"
OBS_CSV = OUT_DIR / "perfect_blocking_observables.csv"
PLOTS_PDF = OUT_DIR / "perfect_blocking_plots.pdf"

DATA_DIR = ROOT / "external" / "mlneuralsampler_multilevel" / "data" / "config"

SEED = 20240616

BETA_EXACT = 0.5 * math.log(1.0 + math.sqrt(2.0))
DATA_BETA = BETA_EXACT
GEN_L8 = OUT_DIR / "critical_ising_L8.npy"
GEN_L16 = OUT_DIR / "critical_ising_L16.npy"
GEN_L8_VALID = OUT_DIR / "critical_ising_L8_validation.npy"

N_GENERATE = {8: 500, 16: 500}
N_THERM = {8: 400, 16: 400}
N_SKIP = {8: 4, 16: 5}

N_OPT_FINE = 500
N_OPT_REPLICA = 4
N_VAL_FINE = 500
N_VAL_REPLICA = 8


def set_cache_env() -> None:
    cache_root = Path("/private/tmp/perfect-blocking-ising-cache")
    (cache_root / "matplotlib").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))


def locate_existing_configs() -> list[dict]:
    results = []
    for L in (8, 16):
        candidates = sorted(DATA_DIR.glob(f"*nx{L}_beta0.4400000000*"))
        for p in candidates:
            results.append({"path": str(p), "exists": True, "L": L})
    return results


def load_text_ising(path: Path, L: int, n: int | None = None) -> np.ndarray:
    rows = []
    with path.open("r") as fh:
        for line in fh:
            if not line.strip():
                continue
            rows.append(np.fromstring(line, sep=" "))
            if n is not None and len(rows) >= n:
                break
    arr = np.asarray(rows, dtype=np.float32)
    if arr.shape[1] != L * L:
        raise ValueError(f"unexpected row length {arr.shape[1]} for L={L}")
    arr = arr.reshape((-1, L, L))
    uniq = set(np.unique(arr).tolist())
    if uniq <= {0.0, 1.0}:
        arr = arr * 2.0 - 1.0
    return arr.astype(np.float32)


def load_reference_configs(L: int, n: int | None = None) -> np.ndarray:
    path = DATA_DIR / f"Ising_data_nx{L}_beta0.4400000000_data1000000.dat"
    if not path.exists():
        zip_path = DATA_DIR / f"ising{L}x{L}.zip"
        if zip_path.exists():
            import zipfile

            with zipfile.ZipFile(zip_path) as zf:
                name = f"Ising_data_nx{L}_beta0.4400000000_data1000000.dat"
                with zf.open(name) as fh:
                    rows = []
                    for raw in fh:
                        line = raw.decode("utf-8", errors="ignore").strip()
                        if not line:
                            continue
                        rows.append(np.fromstring(line, sep=" "))
                        if n is not None and len(rows) >= n:
                            break
            arr = np.asarray(rows, dtype=np.float32)
            return arr.reshape((-1, L, L))
        raise FileNotFoundError(path)
    return load_text_ising(path, L, n=n)


def wolff_step(spins: np.ndarray, beta: float, rng: np.random.Generator) -> np.ndarray:
    L = spins.shape[0]
    p_add = 1.0 - np.exp(-2.0 * beta)
    seed = (int(rng.integers(L)), int(rng.integers(L)))
    spin0 = spins[seed]
    cluster = {seed}
    stack = [seed]
    while stack:
        i, j = stack.pop()
        for ni, nj in ((i + 1) % L, j), ((i - 1) % L, j), (i, (j + 1) % L), (i, (j - 1) % L):
            if (ni, nj) in cluster:
                continue
            if spins[ni, nj] != spin0:
                continue
            if rng.random() < p_add:
                cluster.add((ni, nj))
                stack.append((ni, nj))
    out = spins.copy()
    for i, j in cluster:
        out[i, j] *= -1.0
    return out


def generate_critical_ising(L: int, n_configs: int, n_therm: int, n_skip: int, seed: int, beta: float = DATA_BETA) -> dict:
    rng = np.random.default_rng(seed)
    spins = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(L, L))
    cluster_sizes = []
    t0 = time.time()
    for _ in range(n_therm):
        old = spins
        spins = wolff_step(spins, beta, rng)
        cluster_sizes.append(int(np.sum(old != spins)))
    samples = []
    for _ in range(n_configs):
        for _ in range(n_skip):
            old = spins
            spins = wolff_step(spins, beta, rng)
            cluster_sizes.append(int(np.sum(old != spins)))
        samples.append(spins.copy())
    samples = np.asarray(samples, dtype=np.float32)
    m = samples.mean(axis=(1, 2))
    m_centered = m - m.mean()
    ac1 = float(np.corrcoef(m_centered[:-1], m_centered[1:])[0, 1]) if len(m) > 2 else 0.0
    return {
        "samples": samples,
        "metadata": {
            "L": L,
            "beta": beta,
            "n_configs": n_configs,
            "n_therm": n_therm,
            "n_skip": n_skip,
            "seed": seed,
            "elapsed_sec": time.time() - t0,
            "mean_cluster_size": float(np.mean(cluster_sizes)) if cluster_sizes else 0.0,
            "median_cluster_size": float(np.median(cluster_sizes)) if cluster_sizes else 0.0,
            "magnetization_ac1": ac1,
            "autocorr_caveat": "Wolff-cluster samples are approximately decorrelated, but the chain is not exact independent sampling.",
        },
    }


def ensure_critical_refs() -> dict[int, np.ndarray]:
    """Generate and cache critical-point references with a Wolff cluster sampler."""
    refs = {}
    refs[8] = generate_critical_ising(8, N_GENERATE[8], N_THERM[8], N_SKIP[8], SEED + 8, beta=BETA_EXACT)["samples"]
    refs[16] = generate_critical_ising(16, N_GENERATE[16], N_THERM[16], N_SKIP[16], SEED + 16, beta=BETA_EXACT)["samples"]
    np.save(GEN_L8, refs[8])
    np.save(GEN_L16, refs[16])
    return refs


def generate_validation_sample_8() -> dict:
    """Generate a fresh 8x8 cluster sample for sanity-checking the bundled 8x8 reference."""
    gen = generate_critical_ising(8, N_GENERATE[8], N_THERM[8], N_SKIP[8], SEED + 108, beta=BETA_EXACT)
    np.save(GEN_L8_VALID, gen["samples"])
    return gen


def observables_per_config(spins: np.ndarray) -> dict[str, np.ndarray]:
    spins = np.asarray(spins, dtype=np.float32)
    if spins.ndim == 2:
        spins = spins[None, ...]
    m = spins.mean(axis=(1, 2))
    abs_m = np.abs(m)
    m2 = m**2
    nn_x = np.mean(spins * np.roll(spins, -1, axis=1), axis=(1, 2))
    nn_y = np.mean(spins * np.roll(spins, -1, axis=2), axis=(1, 2))
    nn = 0.5 * (nn_x + nn_y)
    diag1 = np.mean(spins * np.roll(np.roll(spins, -1, axis=1), -1, axis=2), axis=(1, 2))
    diag2 = np.mean(spins * np.roll(np.roll(spins, -1, axis=1), 1, axis=2), axis=(1, 2))
    diag = 0.5 * (diag1 + diag2)
    nn2 = nn**2
    diag2sq = diag**2
    two_nn_x = np.mean(spins * np.roll(spins, -2, axis=1), axis=(1, 2))
    two_nn_y = np.mean(spins * np.roll(spins, -2, axis=2), axis=(1, 2))
    two_nn = 0.5 * (two_nn_x + two_nn_y)
    two_nn2 = two_nn**2
    return {
        "m": m,
        "abs_m": abs_m,
        "m2": m2,
        "nn": nn,
        "diag": diag,
        "2nn": two_nn,
        "nn2": nn2,
        "diag2": diag2sq,
        "2nn2": two_nn2,
    }


def summarize_observables(spins: np.ndarray, bootstrap: bool = False, n_boot: int = 200, seed: int = 0) -> dict:
    obs = observables_per_config(spins)
    keys = ["nn", "diag", "2nn", "nn2", "diag2", "2nn2"]
    values = np.vstack([obs[k] for k in keys])
    means = {k: float(obs[k].mean()) for k in keys}
    errs = {}
    if bootstrap:
        rng = np.random.default_rng(seed)
        n = spins.shape[0]
        for i, k in enumerate(keys):
            boot = []
            arr = values[i]
            for _ in range(n_boot):
                idx = rng.integers(0, n, size=n)
                boot.append(float(arr[idx].mean()))
            errs[k] = float(np.std(boot, ddof=1))
    else:
        n = spins.shape[0]
        for i, k in enumerate(keys):
            errs[k] = float(values[i].std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
    extra = {
        "m": {"mean": float(obs["m"].mean()), "err": float(obs["m"].std(ddof=1) / math.sqrt(spins.shape[0])) if spins.shape[0] > 1 else 0.0},
        "abs_m": {"mean": float(obs["abs_m"].mean()), "err": float(obs["abs_m"].std(ddof=1) / math.sqrt(spins.shape[0])) if spins.shape[0] > 1 else 0.0},
        "m2": {"mean": float(obs["m2"].mean()), "err": float(obs["m2"].std(ddof=1) / math.sqrt(spins.shape[0])) if spins.shape[0] > 1 else 0.0},
    }
    return {"means": means, "errs": errs, "extra": extra, "per_config": obs}


def block_centered_3x3(configs16: np.ndarray, alpha: float, w00: float, w01: float, w11: float, uniforms: np.ndarray) -> np.ndarray:
    s = np.asarray(configs16, dtype=np.float32)
    if s.ndim == 2:
        s = s[None, ...]
    center = s[:, 0::2, 0::2]
    up = np.roll(s, 1, axis=1)[:, 0::2, 0::2]
    down = np.roll(s, -1, axis=1)[:, 0::2, 0::2]
    left = np.roll(s, 1, axis=2)[:, 0::2, 0::2]
    right = np.roll(s, -1, axis=2)[:, 0::2, 0::2]
    ul = np.roll(np.roll(s, 1, axis=1), 1, axis=2)[:, 0::2, 0::2]
    ur = np.roll(np.roll(s, 1, axis=1), -1, axis=2)[:, 0::2, 0::2]
    dl = np.roll(np.roll(s, -1, axis=1), 1, axis=2)[:, 0::2, 0::2]
    dr = np.roll(np.roll(s, -1, axis=1), -1, axis=2)[:, 0::2, 0::2]
    p = w00 * center + w01 * (up + down + left + right) + w11 * (ul + ur + dl + dr)
    if uniforms.ndim == 3:
        if uniforms.shape != p.shape:
            raise ValueError(f"uniforms shape mismatch: {uniforms.shape} vs {p.shape}")
        u = uniforms[None, ...]
    elif uniforms.ndim == 4:
        if uniforms.shape[1:] != p.shape:
            raise ValueError(f"uniforms shape mismatch: {uniforms.shape} vs {p.shape}")
        u = uniforms
    else:
        raise ValueError(f"uniforms must have 3 or 4 dims, got {uniforms.shape}")
    logits = np.clip(4.0 * alpha * p, -50.0, 50.0)
    prob_plus = 1.0 / (1.0 + np.exp(-logits))
    t = np.where(u < prob_plus[None, ...], 1.0, -1.0)
    return t.reshape((-1, p.shape[1], p.shape[2])).astype(np.float32)


def params_to_weights(x: np.ndarray) -> tuple[float, float, float, float]:
    x = np.asarray(x, dtype=np.float64)
    log_alpha = float(x[0])
    logits = np.array(x[1:4], dtype=np.float64)
    logits = logits - logits.max()
    q = np.exp(logits)
    q = q / q.sum()
    w00 = float(q[0])
    w01 = float(q[1] / 4.0)
    w11 = float(q[2] / 4.0)
    alpha = float(np.exp(log_alpha))
    return alpha, w00, w01, w11


def obs_vector(spins: np.ndarray) -> np.ndarray:
    obs = observables_per_config(spins)
    return np.array(
        [
            float(obs["nn"].mean()),
            float(obs["diag"].mean()),
            float(obs["2nn"].mean()),
            float(obs["nn2"].mean()),
            float(obs["diag2"].mean()),
            float(obs["2nn2"].mean()),
        ],
        dtype=np.float64,
    )


def obs_stats(spins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    obs = observables_per_config(spins)
    keys = ["nn", "diag", "2nn", "nn2", "diag2", "2nn2"]
    means = np.array([float(obs[k].mean()) for k in keys], dtype=np.float64)
    errs = np.array([float(obs[k].std(ddof=1) / math.sqrt(spins.shape[0])) if spins.shape[0] > 1 else 0.0 for k in keys], dtype=np.float64)
    return means, errs


@dataclass
class ObjectiveContext:
    fine16: np.ndarray
    true8: np.ndarray
    uniforms: np.ndarray
    target_mean: np.ndarray
    target_err: np.ndarray
    values_cache: dict[str, dict]

    def evaluate(self, x: np.ndarray) -> float:
        alpha, w00, w01, w11 = params_to_weights(x)
        blocked = block_centered_3x3(self.fine16, alpha, w00, w01, w11, self.uniforms)
        mean_block, _ = obs_stats(blocked)
        sigma = np.maximum(self.target_err, 0.02)
        z = (mean_block - self.target_mean) / sigma
        loss = float(np.sum(z**2))
        self.values_cache["last"] = {
            "alpha": alpha,
            "w00": w00,
            "w01": w01,
            "w11": w11,
            "loss": loss,
            "blocked_mean": mean_block.tolist(),
            "z": z.tolist(),
        }
        return loss


def optimize_kernel(fine16: np.ndarray, true8: np.ndarray) -> dict:
    rng = np.random.default_rng(SEED)
    n_opt = min(N_OPT_FINE, len(fine16))
    fine_opt = np.asarray(fine16[:n_opt], dtype=np.float32)
    true_opt = np.asarray(true8[:n_opt], dtype=np.float32)
    uniforms = rng.random((N_OPT_REPLICA, n_opt, 8, 8), dtype=np.float32)
    target_mean, target_err = obs_stats(true_opt)
    ctx = ObjectiveContext(fine_opt, true_opt, uniforms, target_mean, target_err, {})

    x0 = np.array([math.log(2.0), math.log(1.0 / 9.0), math.log(4.0 / 9.0), math.log(4.0 / 9.0)], dtype=np.float64)
    history = []

    def callback(xk, convergence=None):
        history.append({"x": [float(v) for v in xk], "loss": float(ctx.evaluate(xk))})
        return False

    de = differential_evolution(
        ctx.evaluate,
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

    local = minimize(ctx.evaluate, de.x, method="Powell", options={"maxiter": 80, "xtol": 1e-4, "ftol": 1e-4})

    best_x = np.asarray(local.x if local.fun <= de.fun else de.x, dtype=np.float64)
    best_loss = float(min(local.fun, de.fun))
    alpha, w00, w01, w11 = params_to_weights(best_x)
    final_blocked = block_centered_3x3(fine16, alpha, w00, w01, w11, np.random.default_rng(SEED + 1).random((N_VAL_REPLICA, len(fine16), 8, 8), dtype=np.float32))
    final_mean, final_err = obs_stats(final_blocked)
    val2_blocked = block_centered_3x3(fine16, alpha, w00, w01, w11, np.random.default_rng(SEED + 2).random((N_VAL_REPLICA, len(fine16), 8, 8), dtype=np.float32))
    val2_mean, val2_err = obs_stats(val2_blocked)

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
            "primary": {"mean": final_mean.tolist(), "err": final_err.tolist(), "loss": float(np.sum(((final_mean - target_mean) / np.maximum(target_err, 0.02)) ** 2))},
            "replica_seed2": {"mean": val2_mean.tolist(), "err": val2_err.tolist(), "loss": float(np.sum(((val2_mean - target_mean) / np.maximum(target_err, 0.02)) ** 2))},
        },
        "context": ctx,
    }


def compare_table(true8: np.ndarray, blocked: np.ndarray) -> list[dict]:
    true_stats = summarize_observables(true8, bootstrap=True, seed=SEED)
    blk_stats = summarize_observables(blocked, bootstrap=True, seed=SEED + 1)
    rows = []
    names = ["nn", "diag", "2nn", "nn2", "diag2", "2nn2"]
    for name in names:
        t_mean, t_err = true_stats["means"][name], true_stats["errs"][name]
        b_mean, b_err = blk_stats["means"][name], blk_stats["errs"][name]
        sigma = math.sqrt(t_err**2 + b_err**2) if (t_err or b_err) else 1.0
        rows.append(
            {
                "observable": name,
                "true8_mean": t_mean,
                "true8_err": t_err,
                "blocked16_mean": b_mean,
                "blocked16_err": b_err,
                "delta": b_mean - t_mean,
                "sigma": sigma,
                "delta_over_sigma": (b_mean - t_mean) / sigma if sigma else 0.0,
            }
        )
    return rows


def save_csv(rows: list[dict]) -> None:
    with OBS_CSV.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_results(true8: np.ndarray, blocked_a: np.ndarray, blocked_b: np.ndarray, opt: dict, rows: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [r["observable"] for r in rows]
    true_means = [r["true8_mean"] for r in rows]
    true_errs = [r["true8_err"] for r in rows]
    blk_means = [r["blocked16_mean"] for r in rows]
    blk_errs = [r["blocked16_err"] for r in rows]

    fig = plt.figure(figsize=(11, 8))
    gs = fig.add_gridspec(2, 2)

    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(len(names))
    ax.errorbar(x - 0.12, true_means, yerr=true_errs, fmt="o", label="true L=8")
    ax.errorbar(x + 0.12, blk_means, yerr=blk_errs, fmt="s", label="blocked L=16")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30)
    ax.set_ylabel("mean")
    ax.legend(frameon=False)
    ax.set_title("Observable comparison")

    ax = fig.add_subplot(gs[0, 1])
    ax.hist(summarize_observables(true8)["per_config"]["nn"], bins=40, alpha=0.6, label="true L=8", density=True)
    ax.hist(summarize_observables(blocked_a)["per_config"]["nn"], bins=40, alpha=0.6, label="blocked seed A", density=True)
    ax.set_title("Nearest-neighbor histogram")
    ax.legend(frameon=False)

    ax = fig.add_subplot(gs[1, 0])
    ax.hist(summarize_observables(true8)["per_config"]["abs_m"], bins=40, alpha=0.6, label="true L=8", density=True)
    ax.hist(summarize_observables(blocked_a)["per_config"]["abs_m"], bins=40, alpha=0.6, label="blocked seed A", density=True)
    ax.set_title("|m| histogram")
    ax.legend(frameon=False)

    ax = fig.add_subplot(gs[1, 1])
    history = opt["optimization_history"]
    if history:
        ax.plot([h["loss"] for h in history], lw=1.5)
    ax.set_title("Optimization trace")
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss")

    fig.tight_layout()
    fig.savefig(PLOTS_PDF)
    plt.close(fig)


def strip_for_json(obj):
    if isinstance(obj, dict):
        return {k: strip_for_json(v) for k, v in obj.items() if k != "per_config" and k != "context"}
    if isinstance(obj, list):
        return [strip_for_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [strip_for_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def main() -> int:
    set_cache_env()
    np.random.seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        refs = ensure_critical_refs()
        true8 = refs[8]
        fine16 = refs[16]
        generated8 = generate_validation_sample_8()

        opt = optimize_kernel(fine16[:N_OPT_FINE], true8[:N_OPT_FINE])
        alpha = opt["alpha"]
        w00 = opt["weights"]["w00"]
        w01 = opt["weights"]["w01"]
        w11 = opt["weights"]["w11"]

        blocked_val = block_centered_3x3(fine16, alpha, w00, w01, w11, np.random.default_rng(SEED + 3).random((N_VAL_REPLICA, len(fine16), 8, 8), dtype=np.float32))
        blocked_val2 = block_centered_3x3(fine16, alpha, w00, w01, w11, np.random.default_rng(SEED + 4).random((N_VAL_REPLICA, len(fine16), 8, 8), dtype=np.float32))

        true_summary = summarize_observables(true8, bootstrap=True, seed=SEED)
        blocked_summary = summarize_observables(blocked_val, bootstrap=True, seed=SEED + 2)
        blocked_summary2 = summarize_observables(blocked_val2, bootstrap=True, seed=SEED + 3)

        rows = compare_table(true8, blocked_val)
        save_csv(rows)
        plot_results(true8, blocked_val, blocked_val2, opt, rows)

        worst = max(rows, key=lambda r: abs(r["delta_over_sigma"]))
        out = {
            "status": "ok",
            "beta_exact": BETA_EXACT,
            "beta_data": BETA_EXACT,
            "seed": SEED,
            "existing_config_files": locate_existing_configs(),
            "critical_reference_sources": {
                "L8": {"path": str(GEN_L8), "shape": list(true8.shape), "n_configs": int(true8.shape[0]), "beta": BETA_EXACT},
                "L16": {"path": str(GEN_L16), "shape": list(fine16.shape), "n_configs": int(fine16.shape[0]), "beta": BETA_EXACT},
                "generated_validation_8": {"path": str(GEN_L8_VALID), "shape": list(generated8["samples"].shape), "metadata": generated8["metadata"], "beta": BETA_EXACT},
            },
            "blocking_formula": {
                "stencil": "centered 3x3",
                "normalization": "w00 + 4*w01 + 4*w11 = 1",
                "p_n": "w00*s[2i,2j] + w01*(s[2i+1,2j] + s[2i-1,2j] + s[2i,2j+1] + s[2i,2j-1]) + w11*(s[2i+1,2j+1] + s[2i+1,2j-1] + s[2i-1,2j+1] + s[2i-1,2j-1])",
                "P(t=+1|p)": "sigmoid(4*alpha*p)",
            },
            "optimization": {
                "objective": "diagonal chi2 on [nn, diag, 2nn, nn^2, diag^2, 2nn^2] using true L=8 errors with floor 0.02",
                "target_sample_sizes": {"opt_fine16": int(N_OPT_FINE), "val_fine16": int(N_VAL_FINE)},
                "replicas": {"opt": int(N_OPT_REPLICA), "val": int(N_VAL_REPLICA)},
                "best": opt["weights"] | {"alpha": opt["alpha"], "loss": opt["best_loss"]},
                "de_result": opt["de_result"],
                "powell_result": opt["powell_result"],
                "validation": opt["validation"],
            },
            "observables": {
                "true8": true_summary,
                "blocked16": blocked_summary,
                "blocked16_replica_seed2": blocked_summary2,
                "generated8_validation": summarize_observables(generated8["samples"], bootstrap=True, seed=SEED + 7),
                "comparison_table": rows,
                "worst_matched_observable": worst,
            },
            "stability": {
                "seed_A_loss": opt["validation"]["primary"]["loss"],
                "seed_B_loss": opt["validation"]["replica_seed2"]["loss"],
                "loss_difference": abs(opt["validation"]["primary"]["loss"] - opt["validation"]["replica_seed2"]["loss"]),
            },
            "notes": [
                "Critical references were generated locally at beta_c with a Wolff cluster sampler and cached under perfect_blocking_ising/outputs.",
                "A fresh Wolff-generated 8x8 sample at beta_c was also produced as an independent sanity check on the 500-config reference size.",
                "Blocking is stochastic and was optimized with common random numbers to make the objective deterministic enough for SciPy search.",
                "This is an approximation test for RG-like blocking, not an exact perfect-blocking claim.",
            ],
        }
        SUMMARY_JSON.write_text(json.dumps(strip_for_json(out), indent=2, sort_keys=True) + "\n")

        report = []
        report.append("# Perfect Blocking Ising")
        report.append("")
        report.append(f"beta_exact = {BETA_EXACT:.12f}")
        report.append("critical_refs_generated = 500 configs per volume")
        report.append("")
        report.append("## Data sources")
        report.append("- Critical Wolff references generated at beta_c.")
        report.append("")
        report.append("## Critical reference sources")
        report.append(f"- L=8:  {GEN_L8}  shape={list(true8.shape)}")
        report.append(f"- L=16: {GEN_L16} shape={list(fine16.shape)}")
        report.append(f"- validation 8x8 sample: {GEN_L8_VALID} shape={list(generated8['samples'].shape)}")
        report.append("")
        report.append("## Blocking formula")
        report.append("Centered 3x3 stencil with normalization `w00 + 4*w01 + 4*w11 = 1`.")
        report.append("")
        report.append("## Optimized parameters")
        report.append(f"- alpha = {opt['alpha']:.6f}")
        report.append(f"- w00 = {opt['weights']['w00']:.6f}")
        report.append(f"- w01 = {opt['weights']['w01']:.6f}")
        report.append(f"- w11 = {opt['weights']['w11']:.6f}")
        report.append(f"- normalization = {opt['weights']['normalization']:.6f}")
        report.append("")
        report.append("## Observable comparison")
        for row in rows:
            report.append(
                f"- {row['observable']}: true8={row['true8_mean']:.6f}±{row['true8_err']:.6f}, "
                f"blocked16={row['blocked16_mean']:.6f}±{row['blocked16_err']:.6f}, "
                f"Δ/σ={row['delta_over_sigma']:.3f}"
            )
        report.append("")
        report.append(f"Worst matched observable: {worst['observable']} with |Δ/σ|={abs(worst['delta_over_sigma']):.3f}.")
        report.append("")
        report.append("## Stability")
        report.append(f"- validation loss seed A = {out['stability']['seed_A_loss']:.4f}")
        report.append(f"- validation loss seed B = {out['stability']['seed_B_loss']:.4f}")
        report.append(f"- loss difference = {out['stability']['loss_difference']:.4f}")
        generated8_summary = summarize_observables(generated8["samples"], bootstrap=True, seed=SEED + 7)
        report.append("")
        report.append("## Independent 8x8 Wolff check")
        report.append(f"- generated 8x8 |m| = {generated8_summary['extra']['abs_m']['mean']:.6f} ± {generated8_summary['extra']['abs_m']['err']:.6f}")
        report.append(f"- reference 8x8 |m| = {true_summary['extra']['abs_m']['mean']:.6f} ± {true_summary['extra']['abs_m']['err']:.6f}")
        report.append(f"- generated 8x8 nn = {generated8_summary['means']['nn']:.6f} ± {generated8_summary['errs']['nn']:.6f}")
        report.append(f"- reference 8x8 nn = {true_summary['means']['nn']:.6f} ± {true_summary['errs']['nn']:.6f}")
        report.append("")
        report.append("## Interpretation")
        report.append("The blocked L=16 ensemble is compared against the true critical L=8 ensemble, not against itself. The result is an optimized stochastic restriction rule, but it is not an exact perfect-blocking map.")
        REPORT_MD.write_text("\n".join(report) + "\n")

        print(json.dumps({"written": str(SUMMARY_JSON)}, indent=2))
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
        print(json.dumps({"written": str(SUMMARY_JSON)}, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
