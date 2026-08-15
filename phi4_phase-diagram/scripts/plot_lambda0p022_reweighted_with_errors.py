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
VOLUMES = [8, 16, 32, 64, 128]


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


def manifest_for(path: Path) -> dict[str, Any] | None:
    mp = path.with_name("manifest.json")
    if not mp.exists():
        return None
    m = json.loads(mp.read_text(encoding="utf-8"))
    try:
        lam = float(m.get("lambda"))
        kappa = float(m.get("kappa"))
        L = int(m.get("L"))
    except (TypeError, ValueError):
        return None
    gen = str(m.get("generator", ""))
    if abs(lam - LAM) > 1e-12 or L not in VOLUMES or "embedded_wolff_sign_cluster_plus_radial_heatbath" not in gen:
        return None
    return m | {"path": str(path), "kappa": kappa, "L": L}


def discover() -> list[dict[str, Any]]:
    candidates = []
    for p in sorted(ENSEMBLES.glob("lam0p022*/configs.npz")):
        m = manifest_for(p)
        if m is not None:
            candidates.append(m)
    best: dict[tuple[int, float], dict[str, Any]] = {}
    for m in candidates:
        key = (int(m["L"]), round(float(m["kappa"]), 7))
        n = int(m.get("n_configs", 0) or 0)
        if key not in best or n > int(best[key].get("n_configs", 0) or 0):
            best[key] = m
    return [best[k] for k in sorted(best)]


def stats(phi: np.ndarray) -> dict[str, np.ndarray]:
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
        "m2": m * m,
        "m4": m**4,
        "abs_m": np.abs(m),
        "phi2": np.mean(arr * arr, axis=(1, 2)),
        "H": 2.0 * V * nn,
    }


def obs_from_weighted(st: dict[str, np.ndarray], w: np.ndarray, L: int) -> dict[str, float]:
    sw = float(np.sum(w))
    if sw <= 0.0 or not math.isfinite(sw):
        return {"Binder_U4": float("nan"), "susceptibility": float("nan"), "susceptibility_over_L_7over4": float("nan"), "xi_over_L": float("nan"), "phi2": float("nan")}
    m2 = float(np.sum(w * st["m2"]) / sw)
    m4 = float(np.sum(w * st["m4"]) / sw)
    abs_m = float(np.sum(w * st["abs_m"]) / sw)
    phi2 = float(np.sum(w * st["phi2"]) / sw)
    V = L * L
    chi = V * max(m2 - abs_m * abs_m, 0.0)
    return {
        "Binder_U4": 1.0 - m4 / (3.0 * m2 * m2),
        "susceptibility": chi,
        "susceptibility_over_L_7over4": chi / (L**1.75),
        "xi_over_L": math.sqrt(max(m2, 0.0) / max(phi2, 1e-300)),
        "phi2": phi2,
    }


def reweighted_with_error(st: dict[str, np.ndarray], L: int, k0: float, k: float, block: int) -> dict[str, float]:
    logw_raw = 2.0 * (k - k0) * st["H"]
    logw = logw_raw - float(np.max(logw_raw))
    w = np.exp(logw)
    point = obs_from_weighted(st, w, L)
    sw = float(np.sum(w))
    ess = sw * sw / float(np.sum(w * w))
    n = len(w)
    nb = n // block
    errors = {key: float("nan") for key in point}
    if nb >= 5:
        vals = {key: [] for key in point}
        keep_n = nb * block
        for i in range(nb):
            mask = np.ones(keep_n, dtype=bool)
            mask[i * block : (i + 1) * block] = False
            sub_st = {key: val[:keep_n][mask] for key, val in st.items()}
            sub_logw = logw_raw[:keep_n][mask]
            sub_w = np.exp(sub_logw - float(np.max(sub_logw)))
            o = obs_from_weighted(sub_st, sub_w, L)
            for key, val in o.items():
                vals[key].append(val)
        for key, arr_vals in vals.items():
            arr = np.asarray(arr_vals, dtype=float)
            errors[key] = math.sqrt((nb - 1) / nb * float(np.sum((arr - np.mean(arr)) ** 2)))
    out = dict(point)
    out.update({key + "_se": val for key, val in errors.items()})
    out.update({"ess": ess, "ess_fraction": ess / n, "block_size": block, "n_blocks": nb})
    return out


def make_rows() -> list[dict[str, Any]]:
    anchors = discover()
    grid = np.round(np.arange(0.210, 0.2760001, 0.00025), 8)
    all_rows = []
    for a in anchors:
        path = Path(a["path"])
        phi = np.load(path)["phi"].astype(np.float32)
        L = int(a["L"])
        k0 = float(a["kappa"])
        st = stats(phi)
        # Use a conservative visible error bar: N=500 gets 10 blocks; larger
        # samples get more blocks without making delete-block too expensive.
        block = 50 if phi.shape[0] <= 5000 else 100
        for k in grid:
            rw = reweighted_with_error(st, L, k0, float(k), block)
            all_rows.append({
                "L": L,
                "target_kappa": float(k),
                "anchor_kappa": k0,
                "n": int(phi.shape[0]),
                "anchor_path": str(path),
                **rw,
            })
    best: dict[tuple[int, float], dict[str, Any]] = {}
    for r in all_rows:
        key = (int(r["L"]), round(float(r["target_kappa"]), 8))
        if key not in best or float(r["ess_fraction"]) > float(best[key]["ess_fraction"]):
            best[key] = r | {"selection": "max_ess_anchor"}
    rows = [best[k] for k in sorted(best)]
    write_csv(OUT / "lambda0p022_reweighted_best_ess_with_errors.csv", rows)
    return rows


def plot(rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    colors = {8: "tab:blue", 16: "tab:orange", 32: "tab:green", 64: "tab:red", 128: "tab:purple"}

    def emit(path: Path, zoom: bool) -> None:
        with PdfPages(path) as pdf:
            for key, ylabel in [
                ("Binder_U4", "Binder U4"),
                ("susceptibility", "susceptibility = V(<m^2> - <|m|>^2)"),
                ("susceptibility_over_L_7over4", "susceptibility / L^(7/4)"),
                ("xi_over_L", "xi/L proxy"),
                ("ess_fraction", "ESS/N for selected anchor"),
            ]:
                fig, ax = plt.subplots(figsize=(7.4, 4.8))
                for L in VOLUMES:
                    sub = [r for r in rows if int(r["L"]) == L and float(r["ess_fraction"]) >= 0.05]
                    if zoom:
                        sub = [r for r in sub if float(r["target_kappa"]) > 0.265]
                    if not sub:
                        continue
                    x = np.asarray([float(r["target_kappa"]) for r in sub])
                    y = np.asarray([float(r[key]) for r in sub])
                    order = np.argsort(x)
                    x = x[order]
                    y = y[order]
                    ax.plot(x, y, color=colors[L], lw=1.3, label=f"L={L}")
                    if key != "ess_fraction":
                        e = np.asarray([float(r.get(key + "_se", float("nan"))) for r in sub])[order]
                        stride = max(1, len(x) // 20)
                        ax.errorbar(x[::stride], y[::stride], yerr=e[::stride], fmt="none", ecolor=colors[L], elinewidth=0.8, alpha=0.55, capsize=1.5)
                ax.axvline(0.2705, color="black", lw=0.8, alpha=0.35)
                if zoom:
                    ax.set_xlim(0.265, 0.276)
                    ax.set_title("lambda=0.022, reweighted critical-region zoom")
                ax.set_xlabel("kappa")
                ax.set_ylabel(ylabel)
                ax.legend(ncol=2, fontsize=8)
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

    emit(OUT / "lambda0p022_reweighted_binder_chi_xi_with_errors.pdf", zoom=False)
    emit(OUT / "lambda0p022_reweighted_binder_chi_xi_with_errors_zoom_kappa_gt_0p265.pdf", zoom=True)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = make_rows()
    plot(rows)
    print(json.dumps({
        "rows": len(rows),
        "csv": str(OUT / "lambda0p022_reweighted_best_ess_with_errors.csv"),
        "pdf": str(OUT / "lambda0p022_reweighted_binder_chi_xi_with_errors.pdf"),
        "zoom_pdf": str(OUT / "lambda0p022_reweighted_binder_chi_xi_with_errors_zoom_kappa_gt_0p265.pdf"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
