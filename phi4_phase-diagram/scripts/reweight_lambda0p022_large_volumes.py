#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PHASE = ROOT / "phi4_phase-diagram"
ENSEMBLES = PHASE / "ensembles"
OUT = PHASE / "reports" / "lambda0p022_large_volume_reweighting"
LAM = 0.022
VOLUMES = {8, 16, 32, 64, 128}


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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_manifest(path: Path) -> dict[str, Any] | None:
    manifest = path.with_name("manifest.json")
    if not manifest.exists():
        return None
    data = read_json(manifest)
    try:
        lam = float(data.get("lambda"))
        kappa = float(data.get("kappa"))
        L = int(data.get("L"))
    except (TypeError, ValueError):
        return None
    gen = str(data.get("generator", ""))
    if abs(lam - LAM) > 1.0e-12 or L not in VOLUMES or "embedded_wolff_sign_cluster_plus_radial_heatbath" not in gen:
        return None
    return data | {"_configs_path": str(path), "_dir": str(path.parent)}


def discover_anchors() -> list[dict[str, Any]]:
    candidates = []
    for path in sorted(ENSEMBLES.glob("lam0p022*/configs.npz")):
        meta = parse_manifest(path)
        if meta is not None:
            candidates.append(meta)
    # Prefer the largest canonical sample for duplicate (L, kappa), but keep the
    # independent L8 kappa=0.225 check out of the main curve to avoid overweighting.
    best: dict[tuple[int, float], dict[str, Any]] = {}
    for meta in candidates:
        key = (int(meta["L"]), round(float(meta["kappa"]), 7))
        n = int(meta.get("n_configs", meta.get("n_configs_manifest", 0)) or 0)
        old_n = int(best.get(key, {}).get("n_configs", 0) or 0)
        if key not in best or n > old_n:
            best[key] = meta
    return [best[k] for k in sorted(best)]


def per_config_stats(phi: np.ndarray) -> dict[str, np.ndarray]:
    arr = np.asarray(phi, dtype=np.float64)
    L = arr.shape[1]
    V = L * L
    m = arr.mean(axis=(1, 2))
    nn = 0.5 * (
        np.mean(arr * np.roll(arr, -1, axis=1), axis=(1, 2))
        + np.mean(arr * np.roll(arr, -1, axis=2), axis=(1, 2))
    )
    return {
        "m": m,
        "abs_m": np.abs(m),
        "m2": m * m,
        "m4": m**4,
        "NN": nn,
        "H": 2.0 * V * nn,
    }


def weighted_mean(x: np.ndarray, w: np.ndarray) -> float:
    return float(np.sum(w * x) / np.sum(w))


def reweight_anchor(stats: dict[str, np.ndarray], L: int, kappa0: float, kappa: float) -> dict[str, float]:
    V = L * L
    # S(kappa) = local - 2 kappa H, so -S(k)+S(k0)=2(k-k0)H = 4(k-k0)V*NN.
    logw = 2.0 * (kappa - kappa0) * stats["H"]
    logw -= float(np.max(logw))
    w = np.exp(logw)
    sw = float(np.sum(w))
    ess = sw * sw / float(np.sum(w * w))
    m = weighted_mean(stats["m"], w)
    abs_m = weighted_mean(stats["abs_m"], w)
    m2 = weighted_mean(stats["m2"], w)
    m4 = weighted_mean(stats["m4"], w)
    binder = 1.0 - m4 / (3.0 * m2 * m2)
    chi = V * m2
    chi_abs_centered = V * max(m2 - abs_m * abs_m, 0.0)
    return {
        "m_mean": m,
        "abs_m_mean": abs_m,
        "m2": m2,
        "m4": m4,
        "Binder_U4": binder,
        "susceptibility": chi,
        "susceptibility_abs_centered": chi_abs_centered,
        "NN": weighted_mean(stats["NN"], w),
        "ess": ess,
        "ess_fraction": ess / len(w),
    }


def summarize_anchor(path: Path, meta: dict[str, Any]) -> dict[str, Any]:
    phi = np.load(path)["phi"].astype(np.float32)
    stats = per_config_stats(phi)
    L = int(meta["L"])
    kappa = float(meta["kappa"])
    rw = reweight_anchor(stats, L, kappa, kappa)
    return {
        "L": L,
        "kappa": kappa,
        "n": int(phi.shape[0]),
        "shape": list(phi.shape),
        "seed": meta.get("seed"),
        "thermal_sweeps": meta.get("thermal_sweeps"),
        "skip_sweeps": meta.get("skip_sweeps"),
        "generator": meta.get("generator"),
        "path": str(path),
        **rw,
    }


def make_grid(anchors: list[dict[str, Any]]) -> np.ndarray:
    # Broad enough to show L8/L16 scans and the limited large-volume overlap.
    return np.round(np.arange(0.210, 0.2760001, 0.00025), 8)


def select_best_rows(all_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[tuple[int, float], dict[str, Any]] = {}
    for row in all_rows:
        key = (int(row["L"]), round(float(row["target_kappa"]), 8))
        if key not in best or float(row["ess_fraction"]) > float(best[key]["ess_fraction"]):
            best[key] = row | {"selection": "max_ess_anchor"}
    return [best[k] for k in sorted(best)]


def peak_rows(rows: list[dict[str, Any]], ess_cut: float) -> list[dict[str, Any]]:
    out = []
    for L in sorted({int(r["L"]) for r in rows}):
        sub = [r for r in rows if int(r["L"]) == L and float(r["ess_fraction"]) >= ess_cut]
        if not sub:
            continue
        for obs in ["susceptibility", "susceptibility_abs_centered"]:
            peak = max(sub, key=lambda r: float(r[obs]))
            out.append({
                "L": L,
                "observable": obs,
                "peak_kappa": peak["target_kappa"],
                "peak_value": peak[obs],
                "anchor_kappa": peak["anchor_kappa"],
                "ess_fraction": peak["ess_fraction"],
                "n": peak["n"],
                "ess_cut": ess_cut,
            })
        # Binder is monotone-ish here; report closest to Ising reference as a rough comparison.
        ising_u4 = 0.6106901
        closest = min(sub, key=lambda r: abs(float(r["Binder_U4"]) - ising_u4))
        out.append({
            "L": L,
            "observable": "Binder_U4_closest_2d_Ising",
            "peak_kappa": closest["target_kappa"],
            "peak_value": closest["Binder_U4"],
            "anchor_kappa": closest["anchor_kappa"],
            "ess_fraction": closest["ess_fraction"],
            "n": closest["n"],
            "ess_cut": ess_cut,
        })
    return out


def plot_curves(best_rows: list[dict[str, Any]], anchors_summary: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    volumes = [8, 16, 32, 64, 128]
    colors = {8: "tab:blue", 16: "tab:orange", 32: "tab:green", 64: "tab:red", 128: "tab:purple"}

    def emit(path: Path, *, zoom: bool) -> None:
        with PdfPages(path) as pdf:
            for obs, ylabel in [
                ("Binder_U4", "Binder U4"),
                ("susceptibility", "susceptibility = V <m^2>"),
                ("susceptibility_abs_centered", "V(<m^2>-<|m|>^2)"),
                ("ess_fraction", "ESS/N for selected anchor"),
            ]:
                fig, ax = plt.subplots(figsize=(7.2, 4.6))
                for L in volumes:
                    sub = [r for r in best_rows if int(r["L"]) == L and float(r["ess_fraction"]) >= 0.05]
                    if zoom:
                        sub = [r for r in sub if float(r["target_kappa"]) > 0.265]
                    if not sub:
                        continue
                    x = np.asarray([float(r["target_kappa"]) for r in sub])
                    y = np.asarray([float(r[obs]) for r in sub])
                    order = np.argsort(x)
                    ax.plot(x[order], y[order], lw=1.4, color=colors[L], label=f"L={L}")
                    native = [r for r in anchors_summary if int(r["L"]) == L]
                    if zoom:
                        native = [r for r in native if float(r["kappa"]) > 0.265]
                    ax.scatter(
                        [float(r["kappa"]) for r in native],
                        [float(r[obs]) if obs != "ess_fraction" else 1.0 for r in native],
                        s=14,
                        color=colors[L],
                        alpha=0.75,
                    )
                ax.axvline(0.2705, color="black", lw=0.8, alpha=0.35)
                if zoom:
                    ax.set_xlim(0.265, 0.276)
                    ax.set_title("lambda=0.022, critical-region zoom")
                ax.set_xlabel("kappa")
                ax.set_ylabel(ylabel)
                ax.legend()
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

    emit(OUT / "lambda0p022_binder_susceptibility_reweighted.pdf", zoom=False)
    emit(OUT / "lambda0p022_binder_susceptibility_reweighted_zoom_kappa_gt_0p265.pdf", zoom=True)


def write_report(anchors: list[dict[str, Any]], best_rows: list[dict[str, Any]], peaks: list[dict[str, Any]]) -> None:
    lines = [
        "# Lambda=0.022 Binder and susceptibility reweighting",
        "",
        "This recomputes Binder cumulant and susceptibility directly from canonical native configurations for L=8,16,32,64,128 where available. It does not generate new configurations.",
        "",
        "Reweighting convention at fixed lambda:",
        "",
        "`w(kappa) = exp[-S(kappa)+S(kappa0)] = exp[4 (kappa-kappa0) V NN]`, using the project NN normalization.",
        "",
        "## Available anchors",
        "",
        "| L | kappa | N | Binder U4 | susceptibility | chi abs-centered | path |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in anchors:
        lines.append(
            f"| {r['L']} | {float(r['kappa']):.6g} | {r['n']} | {float(r['Binder_U4']):.6g} | "
            f"{float(r['susceptibility']):.6g} | {float(r['susceptibility_abs_centered']):.6g} | `{r['path']}` |"
        )
    lines += [
        "",
        "## Peak/marker summary with ESS/N >= 0.2",
        "",
        "| L | observable | kappa | value | anchor | ESS/N |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for r in peaks:
        lines.append(
            f"| {r['L']} | {r['observable']} | {float(r['peak_kappa']):.6g} | {float(r['peak_value']):.6g} | "
            f"{float(r['anchor_kappa']):.6g} | {float(r['ess_fraction']):.3f} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- L8 and L16 have multiple kappa anchors, so reweighting genuinely extends a broad interval.",
        "- L32/L64/L128 large-volume anchors are sparse compared with L8/L16, so their reweighted curves are local diagnostics; check ESS before interpreting any point.",
        "- Use `lambda0p022_reweighted_best_ess.csv` for the max-ESS envelope and check `ess_fraction` before interpreting any point.",
        "- `susceptibility` is the project convention `V <m^2>`. The centered absolute-magnetization variant is included because it is often less dominated by sector tunneling in finite samples.",
        "- Binder values for the largest volumes near kappa=0.2705 are sensitive to sparse anchoring and finite-sample sector behavior; crossings or peak locations for large volumes should be treated as diagnostics unless bracketed by multiple independent anchors.",
    ]
    (OUT / "LAMBDA0P022_BINDER_SUSCEPTIBILITY_REWEIGHTING_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    anchors = discover_anchors()
    if not anchors:
        raise SystemExit("no lambda=0.022 anchors found")
    anchor_summaries = []
    all_rows = []
    grid = make_grid(anchors)
    for meta in anchors:
        path = Path(meta["_configs_path"])
        phi = np.load(path)["phi"].astype(np.float32)
        stats = per_config_stats(phi)
        L = int(meta["L"])
        kappa0 = float(meta["kappa"])
        anchor_summaries.append(summarize_anchor(path, meta))
        for target in grid:
            rw = reweight_anchor(stats, L, kappa0, float(target))
            all_rows.append({
                "L": L,
                "target_kappa": float(target),
                "anchor_kappa": kappa0,
                "n": int(phi.shape[0]),
                "anchor_path": str(path),
                **rw,
            })
    best = select_best_rows(all_rows)
    peaks = peak_rows(best, ess_cut=0.2)
    write_csv(OUT / "lambda0p022_native_anchor_summary.csv", anchor_summaries)
    write_csv(OUT / "lambda0p022_reweighted_all_anchors.csv", all_rows)
    write_csv(OUT / "lambda0p022_reweighted_best_ess.csv", best)
    write_csv(OUT / "lambda0p022_reweighted_peak_summary.csv", peaks)
    plot_curves(best, anchor_summaries)
    write_report(anchor_summaries, best, peaks)
    print(json.dumps({"out": str(OUT), "anchors": len(anchor_summaries), "all_rows": len(all_rows), "best_rows": len(best)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
