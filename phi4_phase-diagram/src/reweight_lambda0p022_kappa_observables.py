#!/usr/bin/env python3
"""Single-ensemble kappa reweighting for canonical lambda=0.022 phi4 configs."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "phi4_mplconfig"))

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
ENSEMBLES = ROOT / "phi4_phase-diagram" / "ensembles"
REPORTS = ROOT / "phi4_phase-diagram" / "reports"
PLOTS = REPORTS / "plots"
GENERATOR = "embedded_wolff_sign_cluster_plus_radial_heatbath"
LAM = 0.022


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def accepted(manifest: dict[str, Any]) -> bool:
    try:
        lam = float(manifest.get("lambda"))
    except Exception:
        return False
    return (
        abs(lam - LAM) < 1.0e-12
        and manifest.get("generator") == GENERATOR
        and manifest.get("canonical", manifest.get("is_canonical", False)) is True
        and manifest.get("production_use") is True
        and manifest.get("local_metropolis_used") is False
        and not manifest.get("is_superseded", False)
        and not manifest.get("quarantined", False)
    )


def load_anchor(configs: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    phi = np.load(configs)["phi"].astype(np.float64)
    n, l, _ = phi.shape
    m = phi.mean(axis=(1, 2))
    h_links = (
        np.sum(phi * np.roll(phi, -1, axis=1), axis=(1, 2))
        + np.sum(phi * np.roll(phi, -1, axis=2), axis=(1, 2))
    )
    return {
        "path": str(configs),
        "kappa0": float(manifest["kappa"]),
        "L": int(manifest["L"]),
        "n": int(n),
        "m": m,
        "abs_m": np.abs(m),
        "m2_cfg": m**2,
        "m4_cfg": m**4,
        "H": h_links,
    }


def weighted_mean(x: np.ndarray, w: np.ndarray) -> float:
    return float(np.sum(w * x))


def reweight(anchor: dict[str, Any], kappa: float) -> dict[str, Any]:
    dk = float(kappa - anchor["kappa0"])
    logw = 2.0 * dk * anchor["H"]
    logw = logw - float(np.max(logw))
    w_raw = np.exp(logw)
    w = w_raw / np.sum(w_raw)
    ess = float(1.0 / np.sum(w**2))
    m2 = weighted_mean(anchor["m2_cfg"], w)
    m4 = weighted_mean(anchor["m4_cfg"], w)
    b4 = m4 / (m2 * m2) if m2 > 0 else float("nan")
    u4 = 1.0 - b4 / 3.0 if np.isfinite(b4) else float("nan")
    q = (m2 * m2) / m4 if m4 > 0 else float("nan")
    v = int(anchor["L"]) ** 2
    return {
        "target_kappa": float(kappa),
        "anchor_kappa": float(anchor["kappa0"]),
        "L": int(anchor["L"]),
        "n": int(anchor["n"]),
        "ess": ess,
        "ess_fraction": ess / float(anchor["n"]),
        "m_mean": weighted_mean(anchor["m"], w),
        "abs_m_mean": weighted_mean(anchor["abs_m"], w),
        "m2": m2,
        "m4": m4,
        "binder_U4": u4,
        "binder_B4": b4,
        "binder_Q": q,
        "susceptibility": float(v * m2),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict[str, Any]], metric: str, ylabel: str, suffix: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    for L in sorted({int(r["L"]) for r in rows}):
        lr = sorted([r for r in rows if int(r["L"]) == L], key=lambda r: float(r["target_kappa"]))
        ax.plot([r["target_kappa"] for r in lr], [r[metric] for r in lr], label=f"L={L}", lw=1.8)
        anchors = sorted({float(r["anchor_kappa"]) for r in lr})
        direct = [r for r in lr if any(abs(float(r["target_kappa"]) - a) < 1.0e-12 for a in anchors)]
        ax.scatter([r["target_kappa"] for r in direct], [r[metric] for r in direct], s=18)
    ax.set_xlabel("kappa")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} reweighted vs kappa, lambda=0.022")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(PLOTS / f"{suffix}_vs_kappa_lambda0p022_reweighted.png", dpi=180)
    fig.savefig(PLOTS / f"{suffix}_vs_kappa_lambda0p022_reweighted.pdf")
    plt.close(fig)


def main() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    anchors: list[dict[str, Any]] = []
    for configs in sorted(ENSEMBLES.glob("*/configs.npz")):
        manifest_path = configs.parent / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = read_json(manifest_path)
        if accepted(manifest):
            anchors.append(load_anchor(configs, manifest))
    anchors = [a for a in anchors if int(a["L"]) in {8, 16, 32}]
    rows_all: list[dict[str, Any]] = []
    rows_best: list[dict[str, Any]] = []
    for L in sorted({int(a["L"]) for a in anchors}):
        aL = [a for a in anchors if int(a["L"]) == L]
        kmin = min(float(a["kappa0"]) for a in aL)
        kmax = max(float(a["kappa0"]) for a in aL)
        step = 0.0005
        grid = np.arange(kmin, kmax + 0.5 * step, step)
        for k in grid:
            cand = []
            for a in aL:
                row = reweight(a, float(k))
                row["anchor_path"] = a["path"]
                cand.append(row)
                rows_all.append(row)
            best = max(cand, key=lambda r: float(r["ess"]))
            best = dict(best)
            best["selection"] = "max_ess_anchor"
            rows_best.append(best)
    write_csv(REPORTS / "lambda0p022_reweighted_kappa_observables_all_anchors.csv", rows_all)
    write_csv(REPORTS / "lambda0p022_reweighted_kappa_observables.csv", rows_best)
    plot(rows_best, "binder_U4", "Binder U4", "binder_U4")
    plot(rows_best, "susceptibility", "susceptibility", "susceptibility")
    plot(rows_best, "abs_m_mean", "|m|", "abs_m_mean")
    min_ess = min(float(r["ess_fraction"]) for r in rows_best) if rows_best else float("nan")
    report = f"""# Lambda=0.022 Kappa Reweighting

Reweighting uses the action dependence

```text
S(kappa) = S_0 - 2 kappa H
H = sum_x,mu phi_x phi_(x+mu)
w(kappa_target) / w(kappa0) = exp(2 (kappa_target-kappa0) H)
```

For each target kappa and each L, all same-L anchors are evaluated and the row
with the largest effective sample size is selected for the main curve.

- all-anchor CSV: `lambda0p022_reweighted_kappa_observables_all_anchors.csv`
- selected max-ESS CSV: `lambda0p022_reweighted_kappa_observables.csv`
- minimum selected ESS fraction: `{min_ess:.6g}`
- plots: `binder_U4_vs_kappa_lambda0p022_reweighted.*`,
  `susceptibility_vs_kappa_lambda0p022_reweighted.*`, and
  `abs_m_mean_vs_kappa_lambda0p022_reweighted.*`

This is single-ensemble reweighting, not a full multi-histogram estimator.
"""
    (REPORTS / "lambda0p022_reweighted_kappa_observables.md").write_text(report)
    print(json.dumps({"status": "completed", "anchors": len(anchors), "selected_rows": len(rows_best), "min_ess_fraction": min_ess}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
