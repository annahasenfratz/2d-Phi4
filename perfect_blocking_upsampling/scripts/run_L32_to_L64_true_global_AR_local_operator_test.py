#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PKG = PROJECT_ROOT / "perfect_blocking_upsampling"
sys.path.insert(0, str(PKG / "scripts"))
sys.path.insert(0, str(PKG / "src"))

import run_shape_parametric_sampler_validation as sampler  # noqa: E402
from _common import load_config, load_frozen_models, load_kernel_spec, resolve_run_paths  # noqa: E402

DEFAULT_CONFIG = PKG / "outputs" / "shape_parametric_sampler_validation" / "L32_to_L64_smoke" / "L32_to_L64_smoke_config.yaml"
DEFAULT_OUT = PKG / "outputs" / "shape_parametric_sampler_validation" / "L32_to_L64_same_kappa_0p2705_true_global_AR_local_test"
LOCAL_OPS = ["phi2", "phi4", "NN", "2nn", "diag"]
BOOT = 500


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=float) + "\n", encoding="utf-8")


def local_ops(phi: np.ndarray) -> dict[str, float]:
    arr = np.asarray(phi, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr[None]
    nn = 0.5 * (
        np.mean(arr * np.roll(arr, -1, axis=1), axis=(1, 2))
        + np.mean(arr * np.roll(arr, -1, axis=2), axis=(1, 2))
    )
    two_nn = 0.5 * (
        np.mean(arr * np.roll(arr, -2, axis=1), axis=(1, 2))
        + np.mean(arr * np.roll(arr, -2, axis=2), axis=(1, 2))
    )
    diag = np.mean(arr * np.roll(np.roll(arr, -1, axis=1), -1, axis=2), axis=(1, 2))
    return {
        "phi2": float(np.mean(arr * arr)),
        "phi4": float(np.mean(arr**4)),
        "NN": float(np.mean(nn)),
        "2nn": float(np.mean(two_nn)),
        "diag": float(np.mean(diag)),
    }


def native_series(path: Path) -> dict[str, np.ndarray]:
    phi = np.load(path)["phi"].astype(np.float64)
    arr = phi
    nn = 0.5 * (
        np.mean(arr * np.roll(arr, -1, axis=1), axis=(1, 2))
        + np.mean(arr * np.roll(arr, -1, axis=2), axis=(1, 2))
    )
    two_nn = 0.5 * (
        np.mean(arr * np.roll(arr, -2, axis=1), axis=(1, 2))
        + np.mean(arr * np.roll(arr, -2, axis=2), axis=(1, 2))
    )
    diag = np.mean(arr * np.roll(np.roll(arr, -1, axis=1), -1, axis=2), axis=(1, 2))
    return {
        "phi2": np.mean(arr * arr, axis=(1, 2)),
        "phi4": np.mean(arr**4, axis=(1, 2)),
        "NN": nn,
        "2nn": two_nn,
        "diag": diag,
    }


def mean_se(vals: np.ndarray) -> tuple[float, float]:
    vals = np.asarray(vals, dtype=np.float64)
    return float(np.mean(vals)), float(np.std(vals, ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else float("nan")


def weighted_mean(vals: np.ndarray, logw: np.ndarray) -> float:
    w = np.exp(logw - float(np.max(logw)))
    return float(np.sum(w * vals) / np.sum(w))


def weighted_boot(vals: np.ndarray, logw: np.ndarray, rng: np.random.Generator) -> float:
    n = len(vals)
    outs = []
    for _ in range(BOOT):
        idx = rng.integers(0, n, size=n)
        outs.append(weighted_mean(vals[idx], logw[idx]))
    return float(np.std(outs, ddof=1))


def predicted_independence(logw: np.ndarray) -> float:
    vals = []
    for i in range(len(logw)):
        for j in range(len(logw)):
            if i != j:
                vals.append(min(1.0, math.exp(min(0.0, logw[j] - logw[i]))))
    return float(np.mean(vals)) if vals else float("nan")


def run_chain(logw: np.ndarray, rng: np.random.Generator) -> tuple[list[dict[str, Any]], float, int]:
    current = 0
    accepted = 0
    unique = {current}
    rows = [{"chain_step": 0, "proposal_index": current, "accepted": 1, "logw": float(logw[current])}]
    for step in range(1, len(logw)):
        proposal = step
        d = float(logw[proposal] - logw[current])
        acc = math.log(max(float(rng.random()), 1e-300)) < min(0.0, d)
        if acc:
            current = proposal
            accepted += 1
            unique.add(current)
        rows.append({"chain_step": step, "proposal_index": current, "candidate_index": proposal, "accepted": int(acc), "delta_logw": d, "logw": float(logw[current])})
    return rows, accepted / max(1, len(logw) - 1), len(unique)


def load_ctx(cfg: dict[str, Any], kappa_f: float) -> dict[str, Any]:
    refine_model, refine_state, stages, coarse_action, fine_action, _ = load_frozen_models(cfg)
    refine_model.load_state_dict(refine_state, strict=False)
    refine_model.eval()
    for model, _, state, *_ in stages.values():
        model.load_state_dict(state)
        model.eval()
    kernel, _ = load_kernel_spec(cfg)
    return {
        "refine_model": refine_model,
        "stages": stages,
        "coarse_action": replace(coarse_action, kappa=float(coarse_action.kappa)),
        "fine_action": replace(fine_action, kappa=float(kappa_f)),
        "kernel": kernel,
    }


def proposal_stream(args: argparse.Namespace, cfg: dict[str, Any], ctx: dict[str, Any]) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
    coarse_path = resolve_run_paths(cfg)["coarse_ensemble"]
    coarse = np.load(coarse_path)["phi"].astype(np.float32)
    rng = np.random.default_rng(args.seed)
    rows = []
    phis: list[np.ndarray] = []
    for i in range(args.n_proposals):
        idx = int(rng.integers(0, len(coarse)))
        u = coarse[idx][None]
        z_edge, z_pair, z_corner = sampler.sample_z(rng, 1, int(args.Lc))
        state = sampler.compute_state(u, z_edge, z_pair, z_corner, ctx)
        loc = local_ops(state["phi"][0])
        rows.append(
            {
                "proposal_index": i,
                "source_coarse_index": idx,
                "Sf": float(state["sf"][0]),
                "Sc": float(state["sc"][0]),
                "logdet_refine": float(state["logdet"][0]),
                "logq_detail": float(state["logq"][0]),
                "logw": float(state["logw"][0]),
                **loc,
            }
        )
        if args.save_fields:
            phis.append(state["phi"][0].astype(np.float32))
        if (i + 1) % max(1, args.progress_every) == 0:
            print(f"generated {i+1}/{args.n_proposals} proposals", flush=True)
    return rows, phis


def comparison_rows(proposals: list[dict[str, Any]], chain: list[dict[str, Any]], native_path: Path, seed: int) -> list[dict[str, Any]]:
    native = native_series(native_path)
    logw = np.asarray([float(r["logw"]) for r in proposals], dtype=np.float64)
    chain_idx = np.asarray([int(r["proposal_index"]) for r in chain], dtype=int)
    rng = np.random.default_rng(seed + 99)
    rows = []
    for op in LOCAL_OPS:
        native_mean, native_se = mean_se(native[op])
        vals = np.asarray([float(r[op]) for r in proposals], dtype=np.float64)
        raw_mean, raw_se = mean_se(vals)
        rw_mean = weighted_mean(vals, logw)
        rw_se = weighted_boot(vals, logw, rng)
        ar_mean, ar_se = mean_se(vals[chain_idx])
        def rel(x: float) -> float:
            return (x - native_mean) / native_mean if native_mean != 0 else float("nan")
        def pull(x: float, se: float) -> float:
            c = math.sqrt(native_se * native_se + se * se)
            return (x - native_mean) / c if c > 0 else float("nan")
        rows.append(
            {
                "operator": op,
                "native_value": native_mean,
                "native_SE": native_se,
                "raw_value": raw_mean,
                "raw_SE": raw_se,
                "reweighted_value": rw_mean,
                "reweighted_SE": rw_se,
                "AR_chain_value": ar_mean,
                "AR_chain_SE": ar_se,
                "raw_rel_diff": rel(raw_mean),
                "reweighted_rel_diff": rel(rw_mean),
                "AR_chain_rel_diff": rel(ar_mean),
                "raw_pull": pull(raw_mean, raw_se),
                "reweighted_pull": pull(rw_mean, rw_se),
                "AR_chain_pull": pull(ar_mean, ar_se),
            }
        )
    return rows


def diagnostics(args: argparse.Namespace, proposals: list[dict[str, Any]], actual_acc: float, unique: int) -> dict[str, Any]:
    logw = np.asarray([float(r["logw"]) for r in proposals], dtype=np.float64)
    shifted = logw - float(np.max(logw))
    w = np.exp(shifted)
    adj = [min(1.0, math.exp(min(0.0, float(logw[i] - logw[i - 1])))) for i in range(1, len(logw))]
    return {
        "lambda": args.lambda_,
        "kappa_c": args.kappa_c,
        "kappa_f": args.kappa_f,
        "Lc": args.Lc,
        "Lf": args.Lf,
        "N_prop": len(proposals),
        "N_chain": len(proposals),
        "logw_mean": float(np.mean(logw)),
        "logw_std": float(np.std(logw, ddof=1)) if len(logw) > 1 else 0.0,
        "ESS_per_N": float(np.sum(w) ** 2 / np.sum(w * w) / len(w)),
        "predicted_independence_acceptance": predicted_independence(logw),
        "predicted_adjacent_acceptance": float(np.mean(adj)) if adj else float("nan"),
        "actual_AR_acceptance": actual_acc,
        "num_unique_accepted": unique,
        "max_logw_minus_median": float(np.max(logw) - np.median(logw)),
        "notes": "Actual global independence Metropolis chain over proposal stream.",
    }


def fmt(x: Any) -> str:
    try:
        y = float(x)
    except Exception:
        return str(x)
    return f"{y:.6g}" if math.isfinite(y) else "nan"


def md_table(rows: list[dict[str, Any]], fields: list[str]) -> list[str]:
    out = ["| " + " | ".join(fields) + " |", "|" + "|".join(["---" for _ in fields]) + "|"]
    for r in rows:
        out.append("| " + " | ".join(fmt(r.get(f, "")) if f != "operator" else str(r[f]) for f in fields) + " |")
    return out


def write_report(out: Path, comp: list[dict[str, Any]], diag: dict[str, Any]) -> None:
    fields = ["operator", "native_value", "native_SE", "raw_value", "raw_SE", "reweighted_value", "reweighted_SE", "AR_chain_value", "AR_chain_SE", "raw_rel_diff", "reweighted_rel_diff", "AR_chain_rel_diff", "raw_pull", "reweighted_pull", "AR_chain_pull"]
    dfields = ["lambda", "kappa_c", "kappa_f", "Lc", "Lf", "N_prop", "N_chain", "logw_mean", "logw_std", "ESS_per_N", "predicted_independence_acceptance", "predicted_adjacent_acceptance", "actual_AR_acceptance", "num_unique_accepted", "max_logw_minus_median", "notes"]
    lines = [
        "# True global A/R local operator test",
        "",
        "This is an actual global independence Metropolis A/R test over a generated proposal stream. The main operators are local/action-sector only: `phi2`, `phi4`, `NN`, `2nn`, `diag`.",
        "",
        "## Local operator estimates",
        "",
        *md_table(comp, fields),
        "",
        "## Diagnostics",
        "",
        *md_table([diag], dfields),
        "",
        "## Interpretation checklist",
        "",
        "- Does true global A/R correction move local operators toward native? Inspect `AR_chain_rel_diff` against `raw_rel_diff` and `reweighted_rel_diff` above.",
        "- Was the N=8 reweighting diagnostic misleading? Compare this N_prop run with the earlier N=8 report; this run is the first actual chain estimate.",
        "- Is the same-kappa upscaler acceptable after exact A/R correction? This depends on whether `AR_chain_pull` is small across the five local operators and whether actual A/R has enough unique accepted states.",
        "- If A/R fails despite reasonable acceptance, check the saved logweight components for a sign/normalization issue.",
        "- If A/R works but patch chains do not, that suggests the patch kernel may not be sampling the full target efficiently.",
    ]
    (out / "TRUE_GLOBAL_AR_LOCAL_OPERATOR_TEST.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--lambda", dest="lambda_", type=float, default=0.022)
    ap.add_argument("--kappa-c", type=float, default=0.2705)
    ap.add_argument("--kappa-f", type=float, default=0.2705)
    ap.add_argument("--Lc", type=int, default=32)
    ap.add_argument("--Lf", type=int, default=64)
    ap.add_argument("--n-proposals", type=int, default=128)
    ap.add_argument("--seed", type=int, default=20260706)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--save-fields", action="store_true")
    ap.add_argument("--progress-every", type=int, default=16)
    args = ap.parse_args()

    t0 = time.perf_counter()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    ctx = load_ctx(cfg, args.kappa_f)
    proposals, phis = proposal_stream(args, cfg, ctx)
    write_csv(out / "proposal_logweights_and_local_operators.csv", proposals)
    if phis:
        np.savez_compressed(out / "proposal_fields_phi.npz", phi=np.asarray(phis, dtype=np.float32))
    logw = np.asarray([float(r["logw"]) for r in proposals], dtype=np.float64)
    chain_rows, actual_acc, unique = run_chain(logw, np.random.default_rng(args.seed + 1))
    for r in chain_rows:
        pr = proposals[int(r["proposal_index"])]
        for op in LOCAL_OPS:
            r[op] = pr[op]
    write_csv(out / "global_AR_chain_local_operator_estimates.csv", chain_rows)
    native_path = Path(cfg["data"]["fine_reference"])
    if not native_path.is_absolute():
        native_path = PKG / native_path
    comp = comparison_rows(proposals, chain_rows, native_path, args.seed)
    diag = diagnostics(args, proposals, actual_acc, unique)
    write_csv(out / "local_operator_raw_reweighted_AR_vs_native.csv", comp)
    write_csv(out / "global_AR_diagnostics_summary.csv", [diag])
    write_json(out / "summary.json", {**diag, "elapsed_sec": time.perf_counter() - t0})
    write_report(out, comp, diag)
    print(json.dumps({"status": "completed", "out": str(out), "elapsed_sec": time.perf_counter() - t0, **diag}, indent=2, default=float), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
