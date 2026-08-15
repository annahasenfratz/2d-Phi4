#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PKG = PROJECT_ROOT / "perfect_blocking_upsampling"
sys.path.insert(0, str(PKG / "src"))

from perfect_blocking_upsampling.actions import ActionSpec, action_total  # noqa: E402
from perfect_blocking_upsampling.kernels import KernelSpec, apply_kernel, kernel_stencil_from_spec, load_kernel, normalize_kernel  # noqa: E402

OUT = PKG / "outputs" / "kernel_conditioning_scan"
ETA = 0.25
LAMBDA_KAPPA = {
    "lam0p022": (0.022, 0.2705),
    "lam0p2": (0.2, 0.323124),
    "lam0p5": (0.5, 0.3426),
}
OPS = ["phi2", "phi4", "NN", "diag", "second_neighbor", "m2", "m4", "action_density"]


def json_default(x: Any) -> Any:
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(type(x).__name__)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def qgrid(n: int) -> tuple[np.ndarray, np.ndarray]:
    vals = np.linspace(-0.5 * np.pi, 0.5 * np.pi, n)
    return np.meshgrid(vals, vals, indexing="ij")


def qgrid_full(n: int) -> tuple[np.ndarray, np.ndarray]:
    vals = np.linspace(-np.pi, np.pi, n)
    return np.meshgrid(vals, vals, indexing="ij")


def symbol_from_stencil(stencil: np.ndarray, qx: np.ndarray, qy: np.ndarray, eta: float) -> np.ndarray:
    st = normalize_kernel(stencil)
    c = st.shape[0] // 2
    out = np.zeros_like(qx, dtype=np.complex128)
    for i in range(st.shape[0]):
        for j in range(st.shape[1]):
            w = float(st[i, j])
            if w == 0.0:
                continue
            dx = i - c
            dy = j - c
            out += w * np.exp(1j * (qx * dx + qy * dy))
    return (2.0 ** (eta / 2.0)) * out


def polyphase_row_singular(stencil: np.ndarray, qx: np.ndarray, qy: np.ndarray, eta: float) -> np.ndarray:
    aliases = [(0.0, 0.0), (np.pi, 0.0), (0.0, np.pi), (np.pi, np.pi)]
    acc = np.zeros_like(qx, dtype=np.float64)
    for ax, ay in aliases:
        acc += np.abs(symbol_from_stencil(stencil, qx + ax, qy + ay, eta)) ** 2
    return 0.5 * np.sqrt(acc)


def support_coeffs(stencil: np.ndarray) -> list[dict[str, float]]:
    st = normalize_kernel(stencil)
    c = st.shape[0] // 2
    rows = []
    for i in range(st.shape[0]):
        for j in range(st.shape[1]):
            w = float(st[i, j])
            if abs(w) > 1.0e-14:
                rows.append({"dx": int(i - c), "dy": int(j - c), "weight": w})
    return rows


def conditioning(spec: KernelSpec, n_grid: int) -> dict[str, Any]:
    stencil_raw = kernel_stencil_from_spec(spec)
    stencil = normalize_kernel(stencil_raw)
    qx, qy = qgrid(n_grid)
    sym = symbol_from_stencil(stencil, qx, qy, spec.eta)
    amp = np.abs(sym)
    fqx, fqy = qgrid_full(n_grid)
    full_amp = np.abs(symbol_from_stencil(stencil, fqx, fqy, spec.eta))
    idx = np.unravel_index(int(np.argmin(amp)), amp.shape)
    fidx = np.unravel_index(int(np.argmin(full_amp)), full_amp.shape)
    poly = polyphase_row_singular(stencil, qx, qy, spec.eta)
    pidx = np.unravel_index(int(np.argmin(poly)), poly.shape)
    return {
        "normalization_sum_raw": float(np.sum(stencil_raw)),
        "normalization_sum_used": float(np.sum(stencil)),
        "support_size": int(sum(abs(stencil.reshape(-1)) > 1.0e-14)),
        "support_radius": int(max(max(abs(r["dx"]), abs(r["dy"])) for r in support_coeffs(stencil))),
        "coefficients": support_coeffs(stencil),
        "min_abs_K": float(amp[idx]),
        "max_abs_K": float(np.max(amp)),
        "mean_abs_K": float(np.mean(amp)),
        "condition_ratio": float(np.max(amp) / max(float(amp[idx]), 1.0e-300)),
        "min_qx": float(qx[idx]),
        "min_qy": float(qy[idx]),
        "full_bz_min_abs_K": float(full_amp[fidx]),
        "full_bz_max_abs_K": float(np.max(full_amp)),
        "full_bz_mean_abs_K": float(np.mean(full_amp)),
        "full_bz_condition_ratio": float(np.max(full_amp) / max(float(full_amp[fidx]), 1.0e-300)),
        "full_bz_min_qx": float(fqx[fidx]),
        "full_bz_min_qy": float(fqy[fidx]),
        "polyphase_min_singular": float(poly[pidx]),
        "polyphase_max_singular": float(np.max(poly)),
        "polyphase_condition": float(np.max(poly) / max(float(poly[pidx]), 1.0e-300)),
        "polyphase_min_qx": float(qx[pidx]),
        "polyphase_min_qy": float(qy[pidx]),
        "amp_grid": amp,
    }


def spec_from_orbits(name: str, orbits: dict[str, float], eta: float = ETA) -> KernelSpec:
    return KernelSpec(name, "orbit_kernel", eta, 2, "sum_to_one", {k: float(v) for k, v in orbits.items()}, None)


def specs_from_summary(path: Path, name: str, lam_label: str, family: str) -> tuple[KernelSpec, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    w = data.get("weights_shells") or data.get("weights") or {}
    if "w00" in w:
        orbits = {
            "00": float(w.get("w00", 0.0)),
            "10": float(w.get("w10", w.get("w01", 0.0))),
            "11": float(w.get("w11", 0.0)),
            "20": float(w.get("w20", 0.0)),
            "21": float(w.get("w21", 0.0)),
            "22": float(w.get("w22", 0.0)),
            "30": float(w.get("w30", 0.0)),
            "31": float(w.get("w31", 0.0)),
        }
        orbits = {k: v for k, v in orbits.items() if abs(v) > 1.0e-14 or k in {"00", "10", "11"}}
    else:
        raise ValueError(f"cannot parse weights from {path}")
    score = data.get("score") or data.get("primary_L32_to_L16", {}).get("score") or {}
    meta = {"lambda_label": lam_label, "family": family, "source": str(path), "operator_D": score.get("D_op"), "operator_max_abs_z": score.get("max_abs_z")}
    return spec_from_orbits(name, orbits, float(data.get("eta", ETA))), meta


def collect_saved_specs() -> list[tuple[KernelSpec, dict[str, Any]]]:
    out: list[tuple[KernelSpec, dict[str, Any]]] = []
    small3, _ = load_kernel(PKG / "configs" / "kernel_small3_eta0p25.json")
    out.append((small3, {"lambda_label": "lam0p022", "family": "current_small3_transported", "source": str(PKG / "configs" / "kernel_small3_eta0p25.json")}))
    out.append((small3, {"lambda_label": "lam0p2", "family": "cloned_current_small3_L8to16", "source": "same coefficients via lam0p2 L8->L16 YAML"}))
    lam05_init, _ = load_kernel(PKG / "outputs" / "lam0p5_small3_8to16" / "configs" / "kernel_small3_lam0p5_initial_eta0p25.json")
    out.append((lam05_init, {"lambda_label": "lam0p5", "family": "initial_cloned_small3", "source": str(PKG / "outputs" / "lam0p5_small3_8to16" / "configs" / "kernel_small3_lam0p5_initial_eta0p25.json")}))
    for path, name, fam in [
        (PROJECT_ROOT / "perfect_blocking/perfect_blocking_lam0p022_kappa0p2705_fixedeta/kernel5x5_summary.json", "fixedeta_5x5_lam0p022", "optimized_5x5"),
        (PROJECT_ROOT / "perfect_blocking/perfect_blocking_lam0p022_kappa0p2705_fixedeta_kernel_large/kernel_large_summary.json", "fixedeta_large_lam0p022_known_kernel", "optimized_large"),
        (PROJECT_ROOT / "perfect_blocking/perfect_blocking_lam0p022_kappa0p2705_fixedeta_kernel_small3/kernel_small3_summary.json", "fixedeta_small3_lam0p022_summary", "optimized_small3"),
        (PROJECT_ROOT / "perfect_blocking/perfect_blocking_lam0p022_kappa0p2705_fixedeta_kernel_small2/kernel_small2_summary.json", "fixedeta_small2_lam0p022_summary", "optimized_small2"),
    ]:
        if path.exists():
            out.append(specs_from_summary(path, name, "lam0p022", fam))
    selected = PKG / "outputs" / "lam0p5_small3_8to16" / "kernel_optimization" / "fit_from_canonical_L8_L16" / "diagnostics" / "selected_kernel_summary.json"
    if selected.exists():
        data = json.loads(selected.read_text(encoding="utf-8"))
        for key, fam in [("baseline", "lam0p5_baseline_small3"), ("selected_full_summary", "lam0p5_selected_small3")]:
            w = data[key]["weights"]
            out.append((spec_from_orbits(f"{fam}_{key}", {"00": w["w00"], "10": w["w10"], "11": w["w11"]}), {"lambda_label": "lam0p5", "family": fam, "source": str(selected), "operator_D": data[key]["score"].get("D_op"), "operator_max_abs_z": data[key]["score"].get("max_abs_z")}))
    return out


def per_config_observables(phi: np.ndarray, lam: float, kappa: float) -> dict[str, np.ndarray]:
    arr = phi.astype(np.float64)
    m = arr.mean(axis=(1, 2))
    nn = 0.5 * (np.mean(arr * np.roll(arr, -1, axis=1), axis=(1, 2)) + np.mean(arr * np.roll(arr, -1, axis=2), axis=(1, 2)))
    diag = 0.5 * (
        np.mean(arr * np.roll(np.roll(arr, -1, axis=1), -1, axis=2), axis=(1, 2))
        + np.mean(arr * np.roll(np.roll(arr, -1, axis=1), 1, axis=2), axis=(1, 2))
    )
    second = 0.5 * (np.mean(arr * np.roll(arr, -2, axis=1), axis=(1, 2)) + np.mean(arr * np.roll(arr, -2, axis=2), axis=(1, 2)))
    m2 = m * m
    return {
        "phi2": np.mean(arr**2, axis=(1, 2)),
        "phi4": np.mean(arr**4, axis=(1, 2)),
        "NN": nn,
        "diag": diag,
        "second_neighbor": second,
        "m2": m2,
        "m4": m2 * m2,
        "action_density": action_total(arr, ActionSpec("phi4_nn", lam, kappa)) / (arr.shape[1] * arr.shape[2]),
    }


def load_phi(path: Path) -> np.ndarray:
    with np.load(path) as z:
        key = "phi" if "phi" in z.files else z.files[0]
        return z[key].astype(np.float32)


def lam0p2_data(max_n: int) -> tuple[np.ndarray, np.ndarray]:
    fine = load_phi(PKG / "outputs" / "lam0p2_kappa0p323124" / "native" / "L16" / "configs.npz")[:max_n]
    coarse = load_phi(PKG / "outputs" / "lam0p2_kappa0p323124" / "native" / "L8" / "configs.npz")[:max_n]
    return fine, coarse


def operator_score(spec: KernelSpec, fine: np.ndarray, direct: np.ndarray, reg: float) -> dict[str, Any]:
    lam, kappa = LAMBDA_KAPPA["lam0p2"]
    blocked = apply_kernel(fine, spec)[:, 0::2, 0::2]
    bo = per_config_observables(blocked, lam, kappa)
    do = per_config_observables(direct, lam, kappa)
    xb = np.stack([bo[k] for k in OPS], axis=1)
    xd = np.stack([do[k] for k in OPS], axis=1)
    delta = xb.mean(axis=0) - xd.mean(axis=0)
    cov = np.cov(xb, rowvar=False) / xb.shape[0] + np.cov(xd, rowvar=False) / xd.shape[0]
    diag = np.maximum(np.diag(cov), 1.0e-12)
    cov_reg = cov + np.diag(reg * diag)
    z = delta / np.sqrt(diag)
    score = float(delta @ np.linalg.solve(cov_reg, delta))
    return {"operator_D": score, "operator_max_abs_z": float(np.max(np.abs(z))), "operator_rms_z": float(np.sqrt(np.mean(z * z))), "operator_z": {k: float(v) for k, v in zip(OPS, z)}}


def orbit_sum(orbits: dict[str, float]) -> float:
    mult = {"00": 1, "10": 4, "11": 4, "20": 4, "21": 8, "22": 4, "30": 4, "31": 8}
    return float(sum(mult[k] * v for k, v in orbits.items()))


def normalized_orbits(params: np.ndarray, keys: list[str]) -> dict[str, float]:
    raw = {k: float(v) for k, v in zip(keys, params)}
    s = orbit_sum(raw)
    if abs(s) < 1.0e-12:
        raw["00"] += 1.0
        s = orbit_sum(raw)
    return {k: v / s for k, v in raw.items()}


def optimize_lam0p2(args: argparse.Namespace, fine: np.ndarray, direct: np.ndarray) -> list[dict[str, Any]]:
    rng = np.random.default_rng(args.seed)
    starts = {
        "small3": (["00", "10", "11"], np.array([0.648926023044007, 0.10213606494124894, -0.01436757070225069]), 0.12),
        "5x5": (["00", "10", "11", "20", "21", "22"], np.array([0.8413760658101683, -0.04608057817575256, -0.014530493999777745, 0.040431789165677294, 0.035554773988093584, -0.011274281418876275]), 0.08),
        "7x7": (["00", "10", "11", "20", "21", "22", "30", "31"], np.array([0.8448484838113469, -0.03243293770804412, -0.02457514089145793, 0.07403460047436479, 0.02146316055747416, -0.014058670989381367, -0.001225742713799506, -0.002940275119733455]), 0.05),
    }
    rows = []
    for family, (keys, center, scale) in starts.items():
        candidates = [center]
        for _ in range(args.random_candidates):
            candidates.append(center + scale * rng.standard_normal(len(center)))
        best: tuple[float, dict[str, Any]] | None = None
        for i, cand in enumerate(candidates):
            orbits = normalized_orbits(cand, keys)
            spec = spec_from_orbits(f"lam0p2_candidate_{family}_{i}", orbits)
            cond = conditioning(spec, args.grid)
            op = operator_score(spec, fine, direct, args.reg)
            objective = op["operator_D"] - args.gamma * math.log(max(cond["min_abs_K"], 1.0e-12))
            row = {
                "candidate": spec.name,
                "support_family": family,
                "objective": float(objective),
                **op,
                "min_abs_K": cond["min_abs_K"],
                "condition_ratio": cond["condition_ratio"],
                "mean_abs_K": cond["mean_abs_K"],
                "orbits": orbits,
            }
            if best is None or row["objective"] < best[0]:
                best = (float(row["objective"]), row)
        assert best is not None
        rows.append(best[1])
    return rows


def safe_lam0p2_candidates(args: argparse.Namespace, fine: np.ndarray, direct: np.ndarray) -> list[dict[str, Any]]:
    lam0022 = {"00": 0.648926023044007, "10": 0.10213606494124894, "11": -0.01436757070225069}
    lam05 = {"00": 0.7688950570378823, "10": 0.07407430267935117, "11": -0.016298066938821736}
    rows: list[dict[str, Any]] = []
    candidates: list[tuple[str, str, dict[str, float]]] = [
        ("small3_lam0p022_current", "small3_current", lam0022),
        ("small3_lam0p5_selected", "small3_selected_lam0p5", lam05),
    ]
    for alpha in [0.25, 0.5, 0.75, 1.0]:
        orbits = {k: (1.0 - alpha) * lam0022[k] + alpha * lam05[k] for k in lam0022}
        candidates.append((f"small3_interp_alpha{alpha:g}".replace(".", "p"), "small3_interpolation", orbits))
    candidates += [
        (
            "lam0p2_candidate_5x5_114",
            "candidate_5x5_relaxed_safe",
            {"00": 0.8672919276487524, "10": -0.015883175998064533, "11": -0.09797073745503171, "20": 0.059258610545793934, "21": 0.04863105429717138, "22": -0.009489787599228574},
        ),
        (
            "fixedeta_5x5_lam0p022",
            "archived_5x5_safe",
            {"00": 0.8413760658101683, "10": -0.04608057817575256, "11": -0.014530493999777745, "20": 0.040431789165677294, "21": 0.035554773988093584, "22": -0.011274281418876275},
        ),
    ]
    for name, family, orbits in candidates:
        spec = spec_from_orbits(name, orbits)
        cond = conditioning(spec, args.grid)
        op = operator_score(spec, fine, direct, args.reg)
        full_cond = float(cond["full_bz_condition_ratio"])
        full_min = float(cond["full_bz_min_abs_K"])
        pass_strict = full_cond < 4.0 and full_min > 0.15
        pass_relaxed = full_cond < 5.0 and full_min > 0.15
        rows.append(
            {
                "candidate": name,
                "family": family,
                "operator_D": op["operator_D"],
                "operator_max_abs_z": op["operator_max_abs_z"],
                "operator_rms_z": op["operator_rms_z"],
                "coarse_min_abs_K": cond["min_abs_K"],
                "coarse_condition_ratio": cond["condition_ratio"],
                "full_bz_min_abs_K": full_min,
                "full_bz_condition_ratio": full_cond,
                "passes_strict_gate": pass_strict,
                "passes_relaxed_gate": pass_relaxed,
                "orbits": orbits,
                "rank_key": op["operator_D"] if pass_relaxed else float("inf"),
            }
        )
    ranked = sorted(rows, key=lambda r: (not bool(r["passes_relaxed_gate"]), float(r["operator_D"])))
    for i, row in enumerate(ranked, start=1):
        row["safe_rank"] = i if row["passes_relaxed_gate"] else ""
    return ranked


def plot_heatmaps(entries: list[dict[str, Any]], out_dir: Path, n: int) -> None:
    import matplotlib.pyplot as plt

    vals = np.linspace(-0.5 * np.pi, 0.5 * np.pi, n)
    for entry in entries:
        amp = entry.pop("amp_grid")
        fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
        im = ax.imshow(amp.T, origin="lower", extent=[vals[0], vals[-1], vals[0], vals[-1]], aspect="equal")
        ax.set_title(entry["kernel"])
        ax.set_xlabel("qx")
        ax.set_ylabel("qy")
        fig.colorbar(im, ax=ax, label="|K(q)|")
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in entry["kernel"])
        fig.savefig(out_dir / f"{safe}_absK.pdf")
        plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=257)
    ap.add_argument("--max-n", type=int, default=2000)
    ap.add_argument("--reg", type=float, default=0.10)
    ap.add_argument("--gamma", type=float, default=50.0)
    ap.add_argument("--random-candidates", type=int, default=120)
    ap.add_argument("--seed", type=int, default=20260705)
    ap.add_argument("--output-dir", type=Path, default=OUT)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    plot_entries = []
    seen = {}
    for spec, meta in collect_saved_specs():
        key = (meta["lambda_label"], meta["family"], spec.name)
        if key in seen:
            continue
        seen[key] = True
        cond = conditioning(spec, args.grid)
        row = {
            "kernel": spec.name,
            "lambda_label": meta["lambda_label"],
            "family": meta["family"],
            "source": meta["source"],
            "eta": spec.eta,
            "orbits": spec.orbits,
            "operator_D_saved": meta.get("operator_D"),
            "operator_max_abs_z_saved": meta.get("operator_max_abs_z"),
            **{k: v for k, v in cond.items() if k not in {"coefficients", "amp_grid"}},
            "coefficients": json.dumps(cond["coefficients"], sort_keys=True),
        }
        summary_rows.append(row)
        if len(plot_entries) < 8:
            plot_entries.append({"kernel": f"{meta['lambda_label']}_{meta['family']}_{spec.name}", "amp_grid": cond["amp_grid"]})

    fine, direct = lam0p2_data(args.max_n)
    opt_rows = optimize_lam0p2(args, fine, direct)
    safe_rows = safe_lam0p2_candidates(args, fine, direct)
    for opt in opt_rows:
        spec = spec_from_orbits(opt["candidate"], opt["orbits"])
        cond = conditioning(spec, args.grid)
        row = {
            "kernel": spec.name,
            "lambda_label": "lam0p2",
            "family": f"conditioning_operator_candidate_{opt['support_family']}",
            "source": "random local orbit scan preserving sum normalization",
            "eta": spec.eta,
            "orbits": spec.orbits,
            "operator_D_saved": opt["operator_D"],
            "operator_max_abs_z_saved": opt["operator_max_abs_z"],
            **{k: v for k, v in cond.items() if k not in {"coefficients", "amp_grid"}},
            "coefficients": json.dumps(cond["coefficients"], sort_keys=True),
        }
        summary_rows.append(row)
        plot_entries.append({"kernel": f"lam0p2_candidate_{opt['support_family']}", "amp_grid": cond["amp_grid"]})

    write_csv(args.output_dir / "kernel_conditioning_summary.csv", summary_rows)
    write_csv(args.output_dir / "lam0p2_safe_kernel_scan.csv", safe_rows)
    payload = {"kernels": summary_rows, "lam0p2_optimization": opt_rows, "lam0p2_safe_kernel_scan": safe_rows, "settings": vars(args)}
    write_json(args.output_dir / "kernel_conditioning_summary.json", payload)
    if not args.no_plots:
        plot_heatmaps(plot_entries, args.output_dir, args.grid)
    write_report(args.output_dir, summary_rows, opt_rows, safe_rows)
    print(json.dumps({"status": "completed", "output_dir": str(args.output_dir), "n_kernels": len(summary_rows)}, indent=2))
    return 0


def write_report(out_dir: Path, rows: list[dict[str, Any]], opt_rows: list[dict[str, Any]], safe_rows: list[dict[str, Any]]) -> None:
    by_cond = sorted(rows, key=lambda r: float(r["condition_ratio"]))
    lines = [
        "# Kernel Conditioning Scan",
        "",
        "Fourier diagnostics use the scalar fixed-eta symbol over the coarse Brillouin zone `[-pi/2, pi/2]^2`. The polyphase diagnostic reports the singular value of the 1x4 alias row; its condition is therefore a row-norm variation diagnostic, not a square invertibility test.",
        "",
        "## Saved Kernels Ranked By Scalar Condition Ratio",
        "",
        "| kernel | lambda | family | support | coarse min |K| | coarse cond | full min |K| | full cond | min q coarse | saved max |z| |",
        "|---|---|---|---:|---:|---:|---:|---:|---|---:|",
    ]
    for r in by_cond:
        lines.append(
            f"| {r['kernel']} | {r['lambda_label']} | {r['family']} | {int(r['support_size'])} | {float(r['min_abs_K']):.6g} | {float(r['condition_ratio']):.6g} | {float(r['full_bz_min_abs_K']):.6g} | {float(r['full_bz_condition_ratio']):.6g} | ({float(r['min_qx']):.3g},{float(r['min_qy']):.3g}) | {r.get('operator_max_abs_z_saved', '')} |"
        )
    lines += [
        "",
        "## Lambda=0.2 Conditioning/Operator Candidate Scan",
        "",
        "| candidate | support | objective | D_op | max |z| | min |K| | cond | orbits |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in sorted(opt_rows, key=lambda x: float(x["objective"])):
        lines.append(
            f"| {r['candidate']} | {r['support_family']} | {float(r['objective']):.6g} | {float(r['operator_D']):.6g} | {float(r['operator_max_abs_z']):.6g} | {float(r['min_abs_K']):.6g} | {float(r['condition_ratio']):.6g} | `{json.dumps(r['orbits'], sort_keys=True)}` |"
        )
    lines += [
        "",
        "## Lambda=0.2 Full-BZ Safe Kernel Scan",
        "",
        "Safety gates: strict requires full-BZ condition < 4 and min |K| > 0.15; relaxed requires full-BZ condition < 5 and min |K| > 0.15. Kernels are ranked only if they pass the relaxed gate.",
        "",
        "| rank | candidate | family | D_op | max |z| | coarse min |K| | coarse cond | full min |K| | full cond | strict | relaxed |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in safe_rows:
        lines.append(
            f"| {r['safe_rank']} | {r['candidate']} | {r['family']} | {float(r['operator_D']):.6g} | {float(r['operator_max_abs_z']):.6g} | {float(r['coarse_min_abs_K']):.6g} | {float(r['coarse_condition_ratio']):.6g} | {float(r['full_bz_min_abs_K']):.6g} | {float(r['full_bz_condition_ratio']):.6g} | {r['passes_strict_gate']} | {r['passes_relaxed_gate']} |"
        )
    small = [r for r in rows if r["lambda_label"] == "lam0p2" and "small3" in r["family"]]
    if small and opt_rows:
        best = min(opt_rows, key=lambda x: float(x["objective"]))
        lines += [
            "",
            "## Interpretation",
            "",
            f"The current lambda=0.2 branch uses the cloned small3 kernel. In this scan, the best lambda=0.2 candidate is `{best['candidate']}` with min |K| `{float(best['min_abs_K']):.6g}` and condition ratio `{float(best['condition_ratio']):.6g}`.",
            "This identifies better-conditioned alternatives to test before training another flow; no flow training was launched.",
        ]
    lines.append("")
    (out_dir / "kernel_conditioning_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
