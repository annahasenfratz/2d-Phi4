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
OUT = ROOT / "phi4_phase-diagram" / "reports" / "lambda0p022_large_volume_reweighting"
LAM = 0.022
VOLUMES = [16, 32, 64, 128]
KAPPAS = [0.2700, 0.2705, 0.2710, 0.2715, 0.2720]


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
    if not mp.exists():
        return None
    m = json.loads(mp.read_text(encoding="utf-8"))
    if m.get("lambda") != LAM:
        return None
    if m.get("generator") != "embedded_wolff_sign_cluster_plus_radial_heatbath":
        return None
    L = int(m.get("L"))
    k = float(m.get("kappa"))
    if L not in VOLUMES:
        return None
    if min(abs(k - kk) for kk in KAPPAS) > 1e-9:
        return None
    return m


def discover() -> dict[tuple[int, float], list[Path]]:
    groups: dict[tuple[int, float], list[Path]] = {}
    for d in sorted(ENSEMBLES.glob("lam0p022_kappa*_L*_embedded_wolff_sign_cluster_plus_radial_heatbath*")):
        if not d.is_dir():
            continue
        m = manifest(d)
        if m is None:
            continue
        cfg = d / "configs.npz"
        if not cfg.exists():
            continue
        key = (int(m["L"]), round(float(m["kappa"]), 7))
        groups.setdefault(key, []).append(d)
    return groups


def series_from_dirs(dirs: list[Path], L: int) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    ms: list[np.ndarray] = []
    phi2s: list[np.ndarray] = []
    info: list[dict[str, Any]] = []
    burn = 150 if L == 128 else 0
    for d in dirs:
        m = manifest(d)
        phi = np.load(d / "configs.npz")["phi"].astype(np.float64)
        n_raw = int(phi.shape[0])
        b = min(burn, max(0, n_raw - 1))
        kept = phi[b:]
        ms.append(kept.mean(axis=(1, 2)))
        phi2s.append(np.mean(kept * kept, axis=(1, 2)))
        info.append({"path": str(d), "raw_n": n_raw, "burn": b, "kept_n": int(kept.shape[0]), "seed": m.get("seed") if m else None})
    return np.concatenate(ms), np.concatenate(phi2s), info


def obs(m: np.ndarray, phi2: np.ndarray, L: int) -> dict[str, float]:
    V = L * L
    m2 = m * m
    m4 = m**4
    m2m = float(np.mean(m2))
    abs_m = float(np.mean(np.abs(m)))
    chi = V * max(m2m - abs_m * abs_m, 0.0)
    return {
        "xi_over_L": math.sqrt(max(m2m, 0.0) / max(float(np.mean(phi2)), 1e-300)),
        "susceptibility": chi,
        "susceptibility_over_L_7over4": chi / (L**1.75),
        "Binder_U4": 1.0 - float(np.mean(m4)) / (3.0 * m2m * m2m),
        "abs_m": abs_m,
        "phi2": float(np.mean(phi2)),
        "positive_fraction": float(np.mean(m > 0)),
        "sign_flips": int(np.sum(np.sign(m[1:]) * np.sign(m[:-1]) < 0)) if len(m) > 1 else 0,
    }


def block_jackknife(m: np.ndarray, phi2: np.ndarray, L: int, block: int) -> dict[str, float]:
    n = len(m)
    nb = n // block
    if nb < 5:
        return {}
    n_use = nb * block
    vals = []
    for i in range(nb):
        mask = np.ones(n_use, dtype=bool)
        mask[i * block : (i + 1) * block] = False
        vals.append(obs(m[:n_use][mask], phi2[:n_use][mask], L))
    out: dict[str, float] = {}
    for key in ["xi_over_L", "susceptibility", "susceptibility_over_L_7over4", "Binder_U4", "abs_m", "phi2"]:
        arr = np.asarray([v[key] for v in vals], dtype=float)
        out[key + "_se"] = math.sqrt((nb - 1) / nb * float(np.sum((arr - np.mean(arr)) ** 2)))
    return out


def build_table() -> list[dict[str, Any]]:
    groups = discover()
    rows: list[dict[str, Any]] = []
    for (L, k), dirs in sorted(groups.items()):
        m, phi2, info = series_from_dirs(dirs, L)
        point = obs(m, phi2, L)
        se_rows = []
        for block in [10, 20, 25, 50, 100, 200, 500]:
            e = block_jackknife(m, phi2, L, block)
            if e:
                se_rows.append((block, e))
        row: dict[str, Any] = {
            "L": L,
            "kappa": k,
            "n_kept_total": int(len(m)),
            "n_files": len(dirs),
            "file_info_json": json.dumps(info, sort_keys=True),
            "analysis_burn_per_file": 150 if L == 128 else 0,
        }
        row.update(point)
        for key in ["xi_over_L", "susceptibility", "susceptibility_over_L_7over4", "Binder_U4", "abs_m", "phi2"]:
            vals = [e[key + "_se"] for _, e in se_rows if key + "_se" in e]
            row[key + "_se_max"] = max(vals) if vals else float("nan")
            for block, e in se_rows:
                if key + "_se" in e:
                    row[f"{key}_se_block{block}"] = e[key + "_se"]
        rows.append(row)
    return rows


def plot(rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    colors = {16: "tab:orange", 32: "tab:green", 64: "tab:red", 128: "tab:purple"}
    with PdfPages(OUT / "lambda0p022_combined_cut_binder_chi_xi_vs_kappa.pdf") as pdf:
        for key, ylabel in [
            ("Binder_U4", "Binder U4"),
            ("susceptibility", "susceptibility = V(<m^2> - <|m|>^2)"),
            ("susceptibility_over_L_7over4", "susceptibility / L^(7/4)"),
            ("xi_over_L", "xi/L proxy"),
        ]:
            fig, ax = plt.subplots(figsize=(7.2, 4.6))
            for L in VOLUMES:
                sub = sorted([r for r in rows if int(r["L"]) == L], key=lambda r: float(r["kappa"]))
                if not sub:
                    continue
                x = np.asarray([float(r["kappa"]) for r in sub])
                y = np.asarray([float(r[key]) for r in sub])
                e = np.asarray([float(r.get(key + "_se_max", float("nan"))) for r in sub])
                ax.errorbar(x, y, yerr=e, marker="o", lw=1.2, capsize=2.5, color=colors[L], label=f"L={L}")
            ax.set_xlabel("kappa")
            ax.set_ylabel(ylabel)
            ax.legend()
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def write_report(rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Lambda=0.022 combined cut Binder/chi/xi table",
        "",
        "This table combines all native Wolff/radial-heatbath files with matching `(L,kappa)`. Analysis cuts are applied per file: L128 discards the first 150 saved configurations; smaller volumes use all saved configs. L64 N50 add-ons are included as extra statistics.",
        "",
        "| L | kappa | N kept | files | xi/L | SE | Binder U4 | SE | chi | SE | chi/L^(7/4) | SE |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['L']} | {float(r['kappa']):.4f} | {r['n_kept_total']} | {r['n_files']} | "
            f"{float(r['xi_over_L']):.8f} | {float(r['xi_over_L_se_max']):.8f} | "
            f"{float(r['Binder_U4']):.6g} | {float(r['Binder_U4_se_max']):.6g} | "
            f"{float(r['susceptibility']):.6g} | {float(r['susceptibility_se_max']):.6g} | "
            f"{float(r['susceptibility_over_L_7over4']):.6g} | {float(r['susceptibility_over_L_7over4_se_max']):.6g} |"
        )
    lines += [
        "",
        "Outputs:",
        "",
        "- `lambda0p022_combined_cut_binder_chi_xi_table.csv`",
        "- `lambda0p022_combined_cut_binder_chi_xi_vs_kappa.pdf`",
    ]
    (OUT / "LAMBDA0P022_COMBINED_CUT_BINDER_CHI_XI_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = build_table()
    write_csv(OUT / "lambda0p022_combined_cut_binder_chi_xi_table.csv", rows)
    plot(rows)
    write_report(rows)
    print(json.dumps({"rows": len(rows), "out": str(OUT)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
