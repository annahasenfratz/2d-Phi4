#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ENSEMBLES = ROOT / "phi4_phase-diagram" / "ensembles"
OUT = ROOT / "phi4_phase-diagram" / "reports" / "lambda0p022_effective_mass"
LAM = 0.022
VOLUMES = [32, 64, 128]
TARGET_KAPPAS = [0.27075, 0.27100, 0.27125]
BOOTSTRAP = 200
SEED = 20260703


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


def manifest(path: Path) -> dict[str, Any] | None:
    mp = path / "manifest.json"
    cfg = path / "configs.npz"
    if not mp.exists() or not cfg.exists():
        return None
    m = json.loads(mp.read_text(encoding="utf-8"))
    try:
        L = int(m["L"])
        lam = float(m["lambda"])
        kappa = float(m["kappa"])
    except (KeyError, TypeError, ValueError):
        return None
    if L not in VOLUMES or abs(lam - LAM) > 1e-12:
        return None
    if m.get("generator") != "embedded_wolff_sign_cluster_plus_radial_heatbath":
        return None
    return m | {"path": str(path), "L": L, "kappa": round(kappa, 7)}


def discover() -> dict[tuple[int, float], list[dict[str, Any]]]:
    groups: dict[tuple[int, float], list[dict[str, Any]]] = {}
    for d in sorted(ENSEMBLES.glob("lam0p022_kappa*_L*_embedded_wolff_sign_cluster_plus_radial_heatbath*")):
        if not d.is_dir():
            continue
        m = manifest(d)
        if m is None:
            continue
        groups.setdefault((int(m["L"]), float(m["kappa"])), []).append(m)
    return groups


def per_config_quantities(phi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    slices = np.sum(phi, axis=2)
    fft = np.fft.fft(slices, axis=1)
    corr = np.fft.ifft(fft * np.conj(fft), axis=1).real
    L = phi.shape[1]
    V = L * L
    nn = 0.5 * (
        np.mean(phi * np.roll(phi, -1, axis=1), axis=(1, 2))
        + np.mean(phi * np.roll(phi, -1, axis=2), axis=(1, 2))
    )
    hopping_sum = 2.0 * V * nn
    return corr, hopping_sum


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


def load_anchor(files: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    all_corr: list[np.ndarray] = []
    all_h: list[np.ndarray] = []
    info = []
    L = int(files[0]["L"])
    burn_default = 150 if L == 128 else 0
    for m in files:
        cfg = Path(m["path"]) / "configs.npz"
        with np.load(cfg) as data:
            phi = np.asarray(data["phi"], dtype=np.float64)
        raw_n = int(phi.shape[0])
        burn = min(burn_default, max(0, raw_n - 1))
        kept = phi[burn:]
        corr, hopping_sum = per_config_quantities(kept)
        all_corr.append(corr)
        all_h.append(hopping_sum)
        info.append({"path": str(cfg), "raw_n": raw_n, "burn": burn, "kept_n": int(kept.shape[0]), "seed": m.get("seed")})
    return np.concatenate(all_corr, axis=0), np.concatenate(all_h), info


def weights_for(hopping_sum: np.ndarray, anchor_kappa: float, target_kappa: float) -> tuple[np.ndarray, float]:
    logw = 2.0 * (target_kappa - anchor_kappa) * hopping_sum
    logw = logw - float(np.max(logw))
    w = np.exp(logw)
    ess = float(np.sum(w) ** 2 / np.sum(w * w) / len(w))
    return w, ess


def weighted_corr(per_cfg_corr: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.sum(per_cfg_corr * w[:, None], axis=0) / float(np.sum(w))


def bootstrap(per_cfg_corr: np.ndarray, hopping_sum: np.ndarray, anchor_kappa: float, target_kappa: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    n = per_cfg_corr.shape[0]
    rng = np.random.default_rng(seed)
    corr_samples = []
    meff_samples = []
    for _ in range(BOOTSTRAP):
        idx = rng.integers(0, n, size=n)
        w, _ = weights_for(hopping_sum[idx], anchor_kappa, target_kappa)
        folded = fold_correlator(weighted_corr(per_cfg_corr[idx], w))
        m, _ = effective_mass(folded)
        corr_samples.append(folded)
        meff_samples.append(m)
    return np.nanstd(np.asarray(corr_samples), axis=0, ddof=1), np.nanstd(np.asarray(meff_samples), axis=0, ddof=1)


def choose_best_anchor(groups: dict[tuple[int, float], list[dict[str, Any]]], L: int, target_kappa: float) -> tuple[float, np.ndarray, np.ndarray, list[dict[str, Any]], float]:
    candidates = []
    for (LL, anchor_kappa), files in sorted(groups.items()):
        if LL != L:
            continue
        per_cfg_corr, hopping_sum, info = load_anchor(files)
        _, ess = weights_for(hopping_sum, anchor_kappa, target_kappa)
        candidates.append((ess, anchor_kappa, per_cfg_corr, hopping_sum, info))
    if not candidates:
        raise RuntimeError(f"no anchors found for L={L}")
    ess, anchor_kappa, per_cfg_corr, hopping_sum, info = max(candidates, key=lambda x: x[0])
    return anchor_kappa, per_cfg_corr, hopping_sum, info, ess


def analyze_target(groups: dict[tuple[int, float], list[dict[str, Any]]], L: int, target_kappa: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    anchor_kappa, per_cfg_corr, hopping_sum, info, ess = choose_best_anchor(groups, L, target_kappa)
    w, ess = weights_for(hopping_sum, anchor_kappa, target_kappa)
    corr = weighted_corr(per_cfg_corr, w)
    folded = fold_correlator(corr)
    folded_se, meff_se = bootstrap(per_cfg_corr, hopping_sum, anchor_kappa, target_kappa, SEED + int(L * 100000 + round(target_kappa * 1_000_000)))
    meff, arg = effective_mass(folded)

    symmetry = []
    for t in range(1, L):
        denom = max(abs(corr[t]), abs(corr[L - t]), 1e-300)
        symmetry.append(abs(corr[t] - corr[L - t]) / denom)
    symmetry_max = float(np.max(symmetry)) if symmetry else 0.0

    corr_rows = []
    for t in range(L):
        corr_rows.append(
            {
                "L": L,
                "target_kappa": target_kappa,
                "anchor_kappa": anchor_kappa,
                "t": t,
                "t_over_L": t / L,
                "C_reweighted": float(corr[t]),
                "C_fold_reweighted": float(folded[t]) if t < len(folded) else "",
                "C_fold_bootstrap_se": float(folded_se[t]) if t < len(folded_se) else "",
                "ess_fraction": ess,
            }
        )
    meff_rows = []
    for i, val in enumerate(meff):
        t = i + 1
        meff_rows.append(
            {
                "L": L,
                "target_kappa": target_kappa,
                "anchor_kappa": anchor_kappa,
                "t": t,
                "t_over_L": t / L,
                "m_eff": float(val),
                "m_eff_se": float(meff_se[i]) if i < len(meff_se) else float("nan"),
                "m_eff_times_L": float(val * L) if np.isfinite(val) else float("nan"),
                "m_eff_times_L_se": float(meff_se[i] * L) if i < len(meff_se) else float("nan"),
                "arccosh_argument": float(arg[i]),
                "finite": bool(np.isfinite(val)),
                "ess_fraction": ess,
            }
        )
    finite = [r for r in meff_rows if r["finite"]]
    window = [r for r in finite if 0.15 <= float(r["t_over_L"]) <= 0.30]
    summary = {
        "L": L,
        "target_kappa": target_kappa,
        "anchor_kappa": anchor_kappa,
        "ess_fraction": ess,
        "n_configs_used": int(per_cfg_corr.shape[0]),
        "n_files": len(info),
        "finite_t_min": min((int(r["t"]) for r in finite), default=None),
        "finite_t_max": max((int(r["t"]) for r in finite), default=None),
        "nonfinite_meff_points": len(meff_rows) - len(finite),
        "meff_L_t_over_L_0p15_0p30_mean": float(np.nanmean([float(r["m_eff_times_L"]) for r in window])) if window else float("nan"),
        "meff_L_t_over_L_0p15_0p30_std": float(np.nanstd([float(r["m_eff_times_L"]) for r in window], ddof=1)) if len(window) > 1 else float("nan"),
        "symmetry_max_relative": symmetry_max,
        "file_info_json": json.dumps(info, sort_keys=True),
    }
    return corr_rows, meff_rows, summary


def make_plots(rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    colors = {32: "tab:green", 64: "tab:red", 128: "tab:purple"}

    def emit(path: Path, zoom: bool) -> None:
        with PdfPages(path) as pdf:
            for k in TARGET_KAPPAS:
                fig, ax = plt.subplots(figsize=(7.4, 4.8))
                for L in VOLUMES:
                    sub = [r for r in rows if int(r["L"]) == L and abs(float(r["target_kappa"]) - k) < 1e-12 and r["finite"]]
                    if zoom:
                        sub = [r for r in sub if float(r["t_over_L"]) > 0.15]
                    if not sub:
                        continue
                    sub.sort(key=lambda r: float(r["t_over_L"]))
                    x = np.asarray([float(r["t_over_L"]) for r in sub])
                    y = np.asarray([float(r["m_eff_times_L"]) for r in sub])
                    e = np.asarray([float(r["m_eff_times_L_se"]) for r in sub])
                    ax.errorbar(x, y, yerr=e, marker="o", ms=3, lw=1.0, capsize=1.8, color=colors[L], label=f"L={L}")
                if zoom:
                    ax.set_xlim(0.15, 0.52)
                    ax.set_ylim(bottom=0.0)
                    ax.set_title(f"lambda=0.022, target kappa={k:.5f}, reweighted m_eff L, t/L > 0.15")
                else:
                    ax.set_title(f"lambda=0.022, target kappa={k:.5f}, reweighted m_eff L")
                ax.set_xlabel("t / L")
                ax.set_ylabel("m_eff(t) L")
                ax.grid(True, alpha=0.18)
                ax.legend()
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

    emit(OUT / "lambda0p022_effective_mass_reweighted_targets_mL_by_kappa_volumes.pdf", zoom=False)
    emit(OUT / "lambda0p022_effective_mass_reweighted_targets_mL_by_kappa_volumes_zoom_toverL_gt_0p15.pdf", zoom=True)


def write_report(summaries: list[dict[str, Any]]) -> None:
    lines = [
        "# Lambda=0.022 reweighted effective mass targets",
        "",
        "Targets: kappa=0.27075, 0.27100, 0.27125.",
        "",
        "For each `(L,target kappa)`, the script selects the available native anchor with the largest reweighting ESS/N. Reweighting uses `w=exp[2*(kappa_target-kappa_anchor)*sum_x,mu phi_x phi_{x+mu}]`.",
        "",
        "Folded convention: `C_fold[0]=C[0]`, `C_fold[t]=0.5*(C[t]+C[L-t])` for `1 <= t < L/2`, endpoint kept alone.",
        "",
        "Analysis cuts: L128 discards the first 150 saved configs per file; L32/L64 use all saved configs. Same `(L,kappa)` add-on files are combined before reweighting.",
        "",
        "| L | target kappa | anchor kappa | ESS/N | N used | files | finite t range | non-finite | m_eff L mean, 0.15<=t/L<=0.30 | scatter |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in sorted(summaries, key=lambda r: (float(r["target_kappa"]), int(r["L"]))):
        lines.append(
            f"| {s['L']} | {float(s['target_kappa']):.5f} | {float(s['anchor_kappa']):.5f} | {float(s['ess_fraction']):.4f} | "
            f"{s['n_configs_used']} | {s['n_files']} | {s['finite_t_min']}..{s['finite_t_max']} | {s['nonfinite_meff_points']} | "
            f"{float(s['meff_L_t_over_L_0p15_0p30_mean']):.6g} | {float(s['meff_L_t_over_L_0p15_0p30_std']):.6g} |"
        )
    lines += [
        "",
        "Plots:",
        "",
        "- `lambda0p022_effective_mass_reweighted_targets_mL_by_kappa_volumes.pdf`",
        "- `lambda0p022_effective_mass_reweighted_targets_mL_by_kappa_volumes_zoom_toverL_gt_0p15.pdf`",
        "",
        "Data:",
        "",
        "- `lambda0p022_effective_mass_reweighted_targets_correlators.csv`",
        "- `lambda0p022_effective_mass_reweighted_targets_by_L_kappa.csv`",
        "- `lambda0p022_effective_mass_reweighted_targets_summary.csv`",
        "",
        "These are reweighted correlator diagnostics, not new native ensembles. Large-t points remain noisy and arccosh failures should be interpreted as noise-dominated correlators.",
    ]
    (OUT / "EFFECTIVE_MASS_REWEIGHTED_TARGETS_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    groups = discover()
    all_corr: list[dict[str, Any]] = []
    all_meff: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for k in TARGET_KAPPAS:
        for L in VOLUMES:
            corr_rows, meff_rows, summary = analyze_target(groups, L, k)
            all_corr.extend(corr_rows)
            all_meff.extend(meff_rows)
            summaries.append(summary)
            print(f"done target={k:.5f} L={L} anchor={summary['anchor_kappa']:.5f} ESS/N={summary['ess_fraction']:.3f}")
    write_csv(OUT / "lambda0p022_effective_mass_reweighted_targets_correlators.csv", all_corr)
    write_csv(OUT / "lambda0p022_effective_mass_reweighted_targets_by_L_kappa.csv", all_meff)
    write_csv(OUT / "lambda0p022_effective_mass_reweighted_targets_summary.csv", summaries)
    make_plots(all_meff)
    write_report(summaries)
    print(json.dumps({"targets": TARGET_KAPPAS, "groups": len(summaries), "out": str(OUT)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
