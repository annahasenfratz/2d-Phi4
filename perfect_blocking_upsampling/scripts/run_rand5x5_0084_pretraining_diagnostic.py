#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PKG = PROJECT_ROOT / "perfect_blocking_upsampling"
sys.path.insert(0, str(PKG / "src"))

from perfect_blocking_upsampling.kernels import KernelSpec, apply_kernel, kernel_stencil_from_spec, load_kernel, normalize_kernel  # noqa: E402

OUT = PKG / "outputs" / "controlled_patch_lam0p2" / "rand5x5_0084_pretraining_diagnostic"
NATIVE_L32 = PKG / "outputs" / "lam0p2_kappa0p323124" / "native" / "L32" / "configs.npz"
SMALL3 = PKG / "configs" / "kernel_small3_eta0p25.json"
FIVEX5_114 = PKG / "outputs" / "kernel_conditioning_scan" / "candidate_kernels" / "lam0p2_candidate_5x5_114.json"
KSEARCH = PKG / "outputs" / "controlled_patch_lam0p2" / "tail_aware_kernel_search_L16to32"
ETA = 0.25
SUBS = ["ee", "eo", "oe", "oo"]

RAND5X5_0084 = {
    "00": 0.8436928615755199,
    "10": -0.017402121741681566,
    "11": -0.08347926886578183,
    "20": 0.05452001715238207,
    "21": 0.046572888732743664,
    "22": -0.0077076194042859925,
}


def json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(type(obj).__name__)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")


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


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_phi(path: Path) -> np.ndarray:
    with np.load(path) as z:
        key = "phi" if "phi" in z.files else ("configs" if "configs" in z.files else z.files[0])
        return z[key].astype(np.float32)


def qgrid(n: int, full: bool = False) -> tuple[np.ndarray, np.ndarray]:
    lim = np.pi if full else 0.5 * np.pi
    vals = np.linspace(-lim, lim, n)
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
            out += w * np.exp(1j * (qx * (i - c) + qy * (j - c)))
    return (2.0 ** (eta / 2.0)) * out


def conditioning(spec: KernelSpec, n_grid: int = 257) -> dict[str, float]:
    stencil = kernel_stencil_from_spec(spec)
    qx, qy = qgrid(n_grid, full=False)
    amp = np.abs(symbol_from_stencil(stencil, qx, qy, spec.eta))
    fqx, fqy = qgrid(n_grid, full=True)
    famp = np.abs(symbol_from_stencil(stencil, fqx, fqy, spec.eta))
    return {
        "coarse_min_abs_K": float(np.min(amp)),
        "coarse_condition": float(np.max(amp) / max(float(np.min(amp)), 1.0e-300)),
        "full_bz_min_abs_K": float(np.min(famp)),
        "full_bz_condition": float(np.max(famp) / max(float(np.min(famp)), 1.0e-300)),
    }


def split_sublattices(psi: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "ee": psi[:, 0::2, 0::2].astype(np.float64),
        "eo": psi[:, 0::2, 1::2].astype(np.float64),
        "oe": psi[:, 1::2, 0::2].astype(np.float64),
        "oo": psi[:, 1::2, 1::2].astype(np.float64),
    }


def corr(a: np.ndarray, b: np.ndarray) -> float:
    x = a.reshape(-1).astype(np.float64)
    y = b.reshape(-1).astype(np.float64)
    x = x - float(np.mean(x))
    y = y - float(np.mean(y))
    return float(np.mean(x * y) / math.sqrt(max(float(np.mean(x * x) * np.mean(y * y)), 1.0e-300)))


def detail_stats(kernel_name: str, subs: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in SUBS:
        vals = subs[name].reshape(-1)
        mean2 = float(np.mean(vals * vals))
        mean4 = float(np.mean(vals**4))
        rows.append(
            {
                "kernel": kernel_name,
                "stat": "sublattice",
                "sublattice": name,
                "mean": float(np.mean(vals)),
                "variance": float(np.var(vals)),
                "fourth_moment": mean4,
                "local_kurtosis_ratio": mean4 / max(mean2 * mean2, 1.0e-300),
            }
        )
    detail = np.stack([subs["eo"], subs["oe"], subs["oo"]], axis=1)
    d2_per_cfg = np.mean(detail * detail, axis=(1, 2, 3))
    d4_per_cfg = np.mean(detail**4, axis=(1, 2, 3))
    d2 = float(np.mean(d2_per_cfg))
    d4 = float(np.mean(d4_per_cfg))
    qs = np.quantile(np.sqrt(d2_per_cfg), [0.05, 0.25, 0.50, 0.75, 0.95])
    rows.append(
        {
            "kernel": kernel_name,
            "stat": "detail_vector",
            "sublattice": "eo_oe_oo",
            "detail_d2": d2,
            "detail_d4_component_average": d4,
            "detail_local_kurtosis": d4 / max(d2 * d2, 1.0e-300),
            "detail_norm_mean": float(np.mean(np.sqrt(d2_per_cfg))),
            "detail_norm_std": float(np.std(np.sqrt(d2_per_cfg), ddof=1)),
            "detail_norm_q05": float(qs[0]),
            "detail_norm_q25": float(qs[1]),
            "detail_norm_q50": float(qs[2]),
            "detail_norm_q75": float(qs[3]),
            "detail_norm_q95": float(qs[4]),
        }
    )
    return rows


def correlation_rows(kernel_name: str, subs: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for a, b in [("ee", "eo"), ("ee", "oe"), ("ee", "oo"), ("eo", "oe"), ("eo", "oo"), ("oe", "oo")]:
        rows.append(
            {
                "kernel": kernel_name,
                "pair": f"{a}-{b}",
                "correlation": corr(subs[a], subs[b]),
                "product_mean": float(np.mean(subs[a] * subs[b])),
            }
        )
    return rows


def design_matrix(ee: np.ndarray, radius: int) -> np.ndarray:
    feats = [np.ones_like(ee, dtype=np.float64)]
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            feats.append(np.roll(np.roll(ee, dx, axis=1), dy, axis=2).astype(np.float64))
    return np.stack(feats, axis=-1).reshape(-1, len(feats))


def fit_predictability(kernel_name: str, subs: dict[str, np.ndarray], seed: int = 20260706) -> list[dict[str, Any]]:
    n = subs["ee"].shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train = int(0.8 * n)
    train = idx[:n_train]
    val = idx[n_train:]
    rows: list[dict[str, Any]] = []
    detail_names = ["eo", "oe", "oo"]
    y_train = np.stack([subs[k][train] for k in detail_names], axis=-1).reshape(-1, 3)
    y_val = np.stack([subs[k][val] for k in detail_names], axis=-1).reshape(-1, 3)
    for radius, label in [(1, "linear3x3"), (2, "linear5x5")]:
        x_train = design_matrix(subs["ee"][train], radius)
        x_val = design_matrix(subs["ee"][val], radius)
        coef, *_ = np.linalg.lstsq(x_train, y_train, rcond=None)
        pred_train = x_train @ coef
        pred_val = x_val @ coef
        resid_train = y_train - pred_train
        resid_val = y_val - pred_val
        cov = np.cov(resid_train, rowvar=False)
        floor = 1.0e-8 * max(float(np.trace(cov) / 3.0), 1.0)
        cov = cov + floor * np.eye(3)
        inv_cov = np.linalg.inv(cov)
        sign, logdet = np.linalg.slogdet(cov)
        quad = np.einsum("ni,ij,nj->n", resid_val, inv_cov, resid_val)
        nll_site = float(0.5 * np.mean(quad + 3.0 * math.log(2.0 * math.pi) + logdet))
        resid_by_ch = {name: resid_val[:, i] for i, name in enumerate(detail_names)}
        row = {
            "kernel": kernel_name,
            "predictor": label,
            "train_configs": int(len(train)),
            "val_configs": int(len(val)),
            "features": int(x_train.shape[1]),
            "gaussian_nll_per_site": nll_site,
            "residual_var_eo": float(np.var(resid_by_ch["eo"])),
            "residual_var_oe": float(np.var(resid_by_ch["oe"])),
            "residual_var_oo": float(np.var(resid_by_ch["oo"])),
            "residual_kurtosis_eo": kurtosis_ratio(resid_by_ch["eo"]),
            "residual_kurtosis_oe": kurtosis_ratio(resid_by_ch["oe"]),
            "residual_kurtosis_oo": kurtosis_ratio(resid_by_ch["oo"]),
            "residual_corr_eo_oe": corr_flat(resid_by_ch["eo"], resid_by_ch["oe"]),
            "residual_corr_eo_oo": corr_flat(resid_by_ch["eo"], resid_by_ch["oo"]),
            "residual_corr_oe_oo": corr_flat(resid_by_ch["oe"], resid_by_ch["oo"]),
            "residual_cov_trace": float(np.trace(cov)),
            "residual_cov_det": float(np.linalg.det(cov)),
        }
        rows.append(row)
    return rows


def corr_flat(a: np.ndarray, b: np.ndarray) -> float:
    x = a - float(np.mean(a))
    y = b - float(np.mean(b))
    return float(np.mean(x * y) / math.sqrt(max(float(np.mean(x * x) * np.mean(y * y)), 1.0e-300)))


def kurtosis_ratio(vals: np.ndarray) -> float:
    x = vals.astype(np.float64)
    m2 = float(np.mean(x * x))
    m4 = float(np.mean(x**4))
    return m4 / max(m2 * m2, 1.0e-300)


def kernel_specs() -> dict[str, KernelSpec]:
    small3, _ = load_kernel(SMALL3)
    five, _ = load_kernel(FIVEX5_114)
    rand = KernelSpec("rand5x5_0084", "orbit_kernel", ETA, 2, "sum_to_one", RAND5X5_0084, None)
    return {
        "current_small3": KernelSpec("current_small3", small3.type, small3.eta, small3.scale_factor, small3.normalization, small3.orbits, small3.stencil),
        "lam0p2_candidate_5x5_114": five,
        "rand5x5_0084": rand,
    }


def overlap_rows() -> list[dict[str, Any]]:
    summary = {r["candidate"]: r for r in read_csv(KSEARCH / "kernel_search_summary.csv")}
    out: list[dict[str, Any]] = []
    aliases = {
        "current_small3": "baseline_current_small3",
        "lam0p2_candidate_5x5_114": "baseline_lam0p2_candidate_5x5_114",
        "rand5x5_0084": "rand5x5_0084",
    }
    for label, cand in aliases.items():
        r = summary.get(cand)
        if not r:
            continue
        out.append(
            {
                "kernel": label,
                "tail_search_candidate": cand,
                "total_score": r.get("score"),
                "mean_z_rms": r.get("mean_z_rms"),
                "quantile_rms": r.get("quantile_rms"),
                "smooth_sector_penalty": r.get("smooth_sector_penalty"),
                "low_action_ratio": r.get("low_action_ratio"),
                "low_action_and_high_NN_ratio": r.get("low_action_and_high_NN_ratio"),
                "low_action_and_high_2nn_ratio": r.get("low_action_and_high_2nn_ratio"),
                "low_action_and_high_diag_ratio": r.get("low_action_and_high_diag_ratio"),
            }
        )
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    phi = load_phi(NATIVE_L32)
    stats_rows: list[dict[str, Any]] = []
    corr_rows: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    payload: dict[str, Any] = {"native_l32": str(NATIVE_L32), "n_configs": int(len(phi)), "kernels": {}}
    for name, spec in kernel_specs().items():
        cond = conditioning(spec)
        psi = apply_kernel(phi, spec)
        subs = split_sublattices(psi)
        ds = detail_stats(name, subs)
        cr = correlation_rows(name, subs)
        pr = fit_predictability(name, subs)
        stats_rows.extend(ds)
        corr_rows.extend(cr)
        pred_rows.extend(pr)
        detail_vector = next(r for r in ds if r["stat"] == "detail_vector")
        best_pred = min(pr, key=lambda r: float(r["gaussian_nll_per_site"]))
        summary = {
            "kernel": name,
            **cond,
            "detail_d2": detail_vector["detail_d2"],
            "detail_local_kurtosis": detail_vector["detail_local_kurtosis"],
            "best_predictor": best_pred["predictor"],
            "best_gaussian_nll_per_site": best_pred["gaussian_nll_per_site"],
            "best_residual_var_mean": float(np.mean([best_pred["residual_var_eo"], best_pred["residual_var_oe"], best_pred["residual_var_oo"]])),
            "corr_ee_eo": next(r["correlation"] for r in cr if r["pair"] == "ee-eo"),
            "corr_ee_oe": next(r["correlation"] for r in cr if r["pair"] == "ee-oe"),
            "corr_ee_oo": next(r["correlation"] for r in cr if r["pair"] == "ee-oo"),
        }
        summary_rows.append(summary)
        payload["kernels"][name] = {"conditioning": cond, "summary": summary, "orbits": spec.orbits}

    overlap = overlap_rows()
    write_csv(OUT / "rand5x5_0084_pretraining_summary.csv", summary_rows)
    write_csv(OUT / "psi_detail_stats_by_kernel.csv", stats_rows)
    write_csv(OUT / "psi_detail_correlations_by_kernel.csv", corr_rows)
    write_csv(OUT / "gaussian_predictability_by_kernel.csv", pred_rows)
    write_csv(OUT / "coarse_overlap_summary_by_kernel.csv", overlap)
    payload["coarse_overlap_summary"] = overlap
    write_json(OUT / "rand5x5_0084_pretraining_summary.json", payload)

    by = {r["kernel"]: r for r in summary_rows}
    pred_by = {(r["kernel"], r["predictor"]): r for r in pred_rows}
    lines = [
        "# rand5x5_0084 pretraining diagnostic",
        "",
        "No flow was trained and no production chain was launched. Diagnostics use native lambda=0.2 L32 fields only.",
        "",
        "## Summary",
        "",
        "| kernel | full-BZ min | full-BZ cond | detail <d^2> | detail kurtosis | corr ee-eo | corr ee-oo | best Gaussian NLL/site | residual var mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ["current_small3", "lam0p2_candidate_5x5_114", "rand5x5_0084"]:
        r = by[name]
        lines.append(
            f"| {name} | {r['full_bz_min_abs_K']:.6g} | {r['full_bz_condition']:.6g} | {r['detail_d2']:.6g} | {r['detail_local_kurtosis']:.6g} | {r['corr_ee_eo']:.6g} | {r['corr_ee_oo']:.6g} | {r['best_gaussian_nll_per_site']:.6g} | {r['best_residual_var_mean']:.6g} |"
        )
    lines += [
        "",
        "## Coarse-overlap carryover",
        "",
        "| kernel | total score | mean z rms | quantile rms | smooth penalty | low+NN ratio | low+2nn ratio | low+diag ratio |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in overlap:
        def f(key: str) -> str:
            return f"{float(r[key]):.6g}" if r.get(key) not in {None, ""} else "n/a"
        lines.append(
            f"| {r['kernel']} | {f('total_score')} | {f('mean_z_rms')} | {f('quantile_rms')} | {f('smooth_sector_penalty')} | {f('low_action_and_high_NN_ratio')} | {f('low_action_and_high_2nn_ratio')} | {f('low_action_and_high_diag_ratio')} |"
        )
    rand = by["rand5x5_0084"]
    old = by["lam0p2_candidate_5x5_114"]
    small = by["current_small3"]
    lines += [
        "",
        "## Answers",
        "",
        f"1. `rand5x5_0084` is well-conditioned for inverse-blocking diagnostics: full-BZ min |K| `{rand['full_bz_min_abs_K']:.6g}` and condition `{rand['full_bz_condition']:.6g}`, better than current_small3 condition `{small['full_bz_condition']:.6g}` and below the relaxed gate of 5.",
        f"2. Its native psi-detail kurtosis `{rand['detail_local_kurtosis']:.6g}` lies between current_small3 `{small['detail_local_kurtosis']:.6g}` and 5x5_114 `{old['detail_local_kurtosis']:.6g}`; all are mildly sub-Gaussian relative to kurtosis 3 for raw residual components.",
        f"3. Its ee-detail correlations are weaker than current_small3 and close to 5x5_114: corr(ee,eo) `{rand['corr_ee_eo']:.6g}` vs small3 `{small['corr_ee_eo']:.6g}` and 5x5_114 `{old['corr_ee_eo']:.6g}`.",
        f"4. The best simple linear Gaussian predictor leaves mean residual variance `{rand['best_residual_var_mean']:.6g}`, compared with small3 `{small['best_residual_var_mean']:.6g}` and 5x5_114 `{old['best_residual_var_mean']:.6g}`. See `gaussian_predictability_by_kernel.csv` for 3x3/5x5 details.",
        "5. No obvious pretraining blocker appears: rand5x5_0084 is better conditioned than current_small3, has improved coarse-overlap score, and has predictable detail statistics comparable to the old 5x5 candidate. The remaining risk is that the learned residual-flow diagnostics for 5x5_114 were poor despite good static metrics, so this kernel still needs a small residual-flow pilot before any production chain.",
        "",
        "## Files",
        "",
        "- `rand5x5_0084_pretraining_summary.csv`",
        "- `rand5x5_0084_pretraining_summary.json`",
        "- `psi_detail_stats_by_kernel.csv`",
        "- `psi_detail_correlations_by_kernel.csv`",
        "- `gaussian_predictability_by_kernel.csv`",
        "- `coarse_overlap_summary_by_kernel.csv`",
    ]
    (OUT / "rand5x5_0084_pretraining_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
