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
BOOTSTRAP = 200
SEED = 20260702


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


def per_config_correlators(phi: np.ndarray) -> np.ndarray:
    slices = np.sum(phi, axis=2)
    fft = np.fft.fft(slices, axis=1)
    return np.fft.ifft(fft * np.conj(fft), axis=1).real


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


def load_group_corr(files: list[dict[str, Any]]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    all_corr: list[np.ndarray] = []
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
        all_corr.append(per_config_correlators(kept))
        info.append({"path": str(cfg), "raw_n": raw_n, "burn": burn, "kept_n": int(kept.shape[0]), "seed": m.get("seed")})
    return np.concatenate(all_corr, axis=0), info


def bootstrap(per_cfg_corr: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    n = per_cfg_corr.shape[0]
    rng = np.random.default_rng(seed)
    corr_samples = []
    meff_samples = []
    for _ in range(BOOTSTRAP):
        idx = rng.integers(0, n, size=n)
        folded = fold_correlator(np.mean(per_cfg_corr[idx], axis=0))
        m, _ = effective_mass(folded)
        corr_samples.append(folded)
        meff_samples.append(m)
    return np.nanstd(np.asarray(corr_samples), axis=0, ddof=1), np.nanstd(np.asarray(meff_samples), axis=0, ddof=1)


def analyze_group(L: int, kappa: float, files: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    per_cfg_corr, file_info = load_group_corr(files)
    corr = np.mean(per_cfg_corr, axis=0)
    corr_se = np.std(per_cfg_corr, axis=0, ddof=1) / math.sqrt(per_cfg_corr.shape[0])
    folded = fold_correlator(corr)
    folded_se, meff_se = bootstrap(per_cfg_corr, SEED + int(L * 100000 + round(kappa * 1_000_000)))
    meff, arg = effective_mass(folded)

    symmetry = []
    for t in range(1, L):
        denom = max(abs(corr[t]), abs(corr[L - t]), 1e-300)
        symmetry.append(abs(corr[t] - corr[L - t]) / denom)
    symmetry_max = float(np.max(symmetry)) if symmetry else 0.0

    corr_rows: list[dict[str, Any]] = []
    for t in range(L):
        corr_rows.append(
            {
                "L": L,
                "kappa": kappa,
                "t": t,
                "t_over_L": t / L,
                "C": float(corr[t]),
                "C_se_naive": float(corr_se[t]),
                "C_fold": float(folded[t]) if t < len(folded) else "",
                "C_fold_bootstrap_se": float(folded_se[t]) if t < len(folded_se) else "",
            }
        )
    meff_rows: list[dict[str, Any]] = []
    for i, val in enumerate(meff):
        t = i + 1
        meff_rows.append(
            {
                "L": L,
                "kappa": kappa,
                "t": t,
                "t_over_L": t / L,
                "m_eff": float(val),
                "m_eff_se": float(meff_se[i]) if i < len(meff_se) else float("nan"),
                "m_eff_times_L": float(val * L) if np.isfinite(val) else float("nan"),
                "m_eff_times_L_se": float(meff_se[i] * L) if i < len(meff_se) else float("nan"),
                "arccosh_argument": float(arg[i]),
                "finite": bool(np.isfinite(val)),
            }
        )
    finite = [r for r in meff_rows if r["finite"]]
    plateau_rows = [r for r in finite if 0.15 <= float(r["t_over_L"]) <= 0.30]
    summary = {
        "L": L,
        "kappa": kappa,
        "n_files": len(files),
        "n_configs_used": int(per_cfg_corr.shape[0]),
        "symmetry_max_relative": symmetry_max,
        "nonfinite_meff_points": len(meff_rows) - len(finite),
        "finite_t_min": min((int(r["t"]) for r in finite), default=None),
        "finite_t_max": max((int(r["t"]) for r in finite), default=None),
        "meff_L_t_over_L_0p15_0p30_mean": float(np.nanmean([float(r["m_eff_times_L"]) for r in plateau_rows])) if plateau_rows else float("nan"),
        "meff_L_t_over_L_0p15_0p30_std": float(np.nanstd([float(r["m_eff_times_L"]) for r in plateau_rows], ddof=1)) if len(plateau_rows) > 1 else float("nan"),
        "file_info_json": json.dumps(file_info, sort_keys=True),
    }
    return corr_rows, meff_rows, summary


def make_plots(meff_rows: list[dict[str, Any]], summaries: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    colors = {32: "tab:green", 64: "tab:red", 128: "tab:purple"}
    kappas = sorted({float(r["kappa"]) for r in meff_rows})

    def plot_one(pdf_path: Path, zoom: bool) -> None:
        with PdfPages(pdf_path) as pdf:
            for k in kappas:
                fig, ax = plt.subplots(figsize=(7.4, 4.8))
                for L in VOLUMES:
                    sub = [r for r in meff_rows if int(r["L"]) == L and abs(float(r["kappa"]) - k) < 1e-12 and r["finite"]]
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
                    ax.set_title(f"lambda=0.022, kappa={k:.5f}, folded m_eff L, t/L > 0.15")
                else:
                    ax.set_title(f"lambda=0.022, kappa={k:.5f}, folded m_eff L")
                ax.set_xlabel("t / L")
                ax.set_ylabel("m_eff(t) L")
                ax.grid(True, alpha=0.18)
                ax.legend()
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

    plot_one(OUT / "lambda0p022_effective_mass_mL_by_kappa_volumes.pdf", zoom=False)
    plot_one(OUT / "lambda0p022_effective_mass_mL_by_kappa_volumes_zoom_toverL_gt_0p15.pdf", zoom=True)


def write_report(summaries: list[dict[str, Any]]) -> None:
    lines = [
        "# Lambda=0.022 effective mass by volume and kappa",
        "",
        "Correlator convention: `Phi(x)=sum_y phi(x,y)` and `C(t)=sum_x Phi(x) Phi(x+t)` with periodic x direction.",
        "",
        "Folded convention: `C_fold[0]=C[0]`, `C_fold[t]=0.5*(C[t]+C[L-t])` for `1 <= t < L/2`, endpoint kept alone.",
        "",
        "Analysis cuts: L128 discards the first 150 saved configs per file; L32/L64 use all saved configs. Same `(L,kappa)` add-on files are combined.",
        "",
        "| L | kappa | N used | files | finite t range | non-finite points | m_eff L mean, 0.15<=t/L<=0.30 | scatter | symmetry max |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in sorted(summaries, key=lambda r: (float(r["kappa"]), int(r["L"]))):
        lines.append(
            f"| {s['L']} | {float(s['kappa']):.5f} | {s['n_configs_used']} | {s['n_files']} | "
            f"{s['finite_t_min']}..{s['finite_t_max']} | {s['nonfinite_meff_points']} | "
            f"{float(s['meff_L_t_over_L_0p15_0p30_mean']):.6g} | {float(s['meff_L_t_over_L_0p15_0p30_std']):.6g} | "
            f"{float(s['symmetry_max_relative']):.3g} |"
        )
    lines += [
        "",
        "Plots:",
        "",
        "- `lambda0p022_effective_mass_mL_by_kappa_volumes.pdf`",
        "- `lambda0p022_effective_mass_mL_by_kappa_volumes_zoom_toverL_gt_0p15.pdf`",
        "",
        "Data:",
        "",
        "- `lambda0p022_effective_mass_correlators_by_L_kappa.csv`",
        "- `lambda0p022_effective_mass_by_L_kappa.csv`",
        "- `lambda0p022_effective_mass_summary_by_L_kappa.csv`",
        "",
        "The `m_eff L` window averages are first-pass diagnostics only. The curves remain noisy at larger `t/L`, especially for L128.",
    ]
    (OUT / "EFFECTIVE_MASS_VOLUME_KAPPA_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    groups = discover()
    all_corr: list[dict[str, Any]] = []
    all_meff: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for (L, kappa), files in sorted(groups.items()):
        corr_rows, meff_rows, summary = analyze_group(L, kappa, files)
        all_corr.extend(corr_rows)
        all_meff.extend(meff_rows)
        summaries.append(summary)
        print(f"done L={L} kappa={kappa:.5f} N={summary['n_configs_used']} files={summary['n_files']}")
    write_csv(OUT / "lambda0p022_effective_mass_correlators_by_L_kappa.csv", all_corr)
    write_csv(OUT / "lambda0p022_effective_mass_by_L_kappa.csv", all_meff)
    write_csv(OUT / "lambda0p022_effective_mass_summary_by_L_kappa.csv", summaries)
    make_plots(all_meff, summaries)
    write_report(summaries)
    print(json.dumps({"groups": len(summaries), "out": str(OUT)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
