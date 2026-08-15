#!/usr/bin/env python3
"""Recompute standard Binder observables for canonical phi4 ensembles."""

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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def is_true(value: Any) -> bool:
    return bool(value) if isinstance(value, bool) else False


def is_canonical_manifest(manifest: dict[str, Any]) -> bool:
    if not is_true(manifest.get("canonical", manifest.get("is_canonical", False))):
        return False
    if not is_true(manifest.get("production_use", False)):
        return False
    if manifest.get("local_metropolis_used") is not False:
        return False
    if manifest.get("generator") != GENERATOR:
        return False
    if is_true(manifest.get("is_superseded", False)) or is_true(manifest.get("quarantined", False)):
        return False
    return True


def load_phi(configs: Path) -> np.ndarray:
    z = np.load(configs)
    return z["phi"].astype(np.float64, copy=False)


def xi_over_l(phi: np.ndarray) -> tuple[float, float]:
    n, l, _ = phi.shape
    ft = np.fft.fftn(phi, axes=(1, 2))
    m = phi.mean(axis=(1, 2))
    chi = float(l * l * np.mean(m * m))
    fpx = float(np.mean(np.abs(ft[:, 1, 0]) ** 2) / (l * l))
    fpy = float(np.mean(np.abs(ft[:, 0, 1]) ** 2) / (l * l))
    f = 0.5 * (fpx + fpy)
    if f <= 0.0 or chi / f <= 1.0:
        return float("nan"), float("nan")
    xi = (1.0 / (2.0 * math.sin(math.pi / l))) * math.sqrt(chi / f - 1.0)
    return float(xi), float(xi / l)


def observables(phi: np.ndarray, lam: float, kappa: float) -> dict[str, float]:
    n, l, _ = phi.shape
    m = phi.mean(axis=(1, 2))
    m_mean = float(np.mean(m))
    abs_m_mean = float(np.mean(np.abs(m)))
    m2 = float(np.mean(m**2))
    m4 = float(np.mean(m**4))
    binder_b4 = m4 / (m2 * m2) if m2 > 0 else float("nan")
    binder_u4 = 1.0 - binder_b4 / 3.0 if np.isfinite(binder_b4) else float("nan")
    binder_q = (m2 * m2) / m4 if m4 > 0 else float("nan")
    phi2_cfg = np.mean(phi**2, axis=(1, 2))
    phi4_cfg = np.mean(phi**4, axis=(1, 2))
    nn = 0.5 * (
        np.mean(phi * np.roll(phi, -1, axis=1), axis=(1, 2))
        + np.mean(phi * np.roll(phi, -1, axis=2), axis=(1, 2))
    )
    diag = 0.5 * (
        np.mean(phi * np.roll(np.roll(phi, -1, axis=1), -1, axis=2), axis=(1, 2))
        + np.mean(phi * np.roll(np.roll(phi, -1, axis=1), 1, axis=2), axis=(1, 2))
    )
    twonn = 0.5 * (
        np.mean(phi * np.roll(phi, -2, axis=1), axis=(1, 2))
        + np.mean(phi * np.roll(phi, -2, axis=2), axis=(1, 2))
    )
    xi, xi_l = xi_over_l(phi)
    return {
        "n_configs": int(n),
        "m_mean": m_mean,
        "abs_m_mean": abs_m_mean,
        "m2": m2,
        "m4": m4,
        "binder_U4": binder_u4,
        "binder_B4": binder_b4,
        "binder_Q": binder_q,
        "susceptibility": float(l * l * m2),
        "phi2": float(np.mean(phi2_cfg)),
        "phi4": float(np.mean(phi4_cfg)),
        "NN": float(np.mean(nn)),
        "diag": float(np.mean(diag)),
        "2nn": float(np.mean(twonn)),
        "xi": xi,
        "xi_over_L": xi_l,
        "action_density": float((1.0 - 2.0 * lam) * np.mean(phi2_cfg) + lam * np.mean(phi4_cfg) - 4.0 * kappa * np.mean(nn)),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def lambda_tag(lam: float) -> str:
    return f"lambda{lam:.3f}".replace(".", "p")


def plot_metric(rows: list[dict[str, Any]], lam: float, metric: str, ylabel: str) -> None:
    subset = [r for r in rows if abs(float(r["lambda"]) - lam) < 5.0e-10 and np.isfinite(float(r[metric]))]
    if not subset:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    for L in sorted({int(r["L"]) for r in subset}):
        lr = sorted([r for r in subset if int(r["L"]) == L], key=lambda r: float(r["kappa"]))
        ax.plot([float(r["kappa"]) for r in lr], [float(r[metric]) for r in lr], marker="o", label=f"L={L}")
    ax.set_xlabel("kappa")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs kappa, lambda={lam:g}")
    ax.grid(alpha=0.25)
    ax.legend()
    tag = lambda_tag(lam)
    png = PLOTS / f"{metric}_vs_kappa_{tag}.png"
    pdf = PLOTS / f"{metric}_vs_kappa_{tag}.pdf"
    fig.savefig(png, dpi=180)
    fig.savefig(pdf)
    plt.close(fig)


def main() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for configs in sorted(ENSEMBLES.glob("*/configs.npz")):
        manifest_path = configs.parent / "manifest.json"
        if not manifest_path.exists():
            excluded.append({"path": str(configs), "reason": "missing manifest.json"})
            continue
        manifest = read_json(manifest_path)
        if not is_canonical_manifest(manifest):
            excluded.append({"path": str(configs), "reason": "not canonical production embedded-wolff/radial-heatbath"})
            continue
        phi = load_phi(configs)
        lam = float(manifest["lambda"])
        kappa = float(manifest["kappa"])
        row = {
            "ensemble_dir": str(configs.parent),
            "configs_path": str(configs),
            "lambda": lam,
            "kappa": kappa,
            "L": int(manifest["L"]),
            "generator": manifest.get("generator"),
            "canonical": manifest.get("canonical", manifest.get("is_canonical")),
            "production_use": manifest.get("production_use"),
            "local_metropolis_used": manifest.get("local_metropolis_used"),
        }
        row.update(observables(phi, lam, kappa))
        rows.append(row)
    rows.sort(key=lambda r: (float(r["lambda"]), int(r["L"]), float(r["kappa"]), str(r["ensemble_dir"])))
    write_csv(REPORTS / "canonical_binder_observables.csv", rows)
    (REPORTS / "canonical_binder_observables.json").write_text(json.dumps({"rows": rows, "excluded": excluded}, indent=2) + "\n")

    for lam in sorted({float(r["lambda"]) for r in rows}):
        plot_metric(rows, lam, "binder_U4", "Binder U4")
        plot_metric(rows, lam, "xi_over_L", "xi/L")
        plot_metric(rows, lam, "susceptibility", "susceptibility")
        plot_metric(rows, lam, "abs_m_mean", "|m|")

    important = [r for r in rows if float(r["lambda"]) in {1.0, 0.022}]
    table = "\n".join(
        f"| {r['lambda']:.6g} | {r['kappa']:.6g} | {int(r['L'])} | {int(r['n_configs'])} | {r['binder_U4']:.6g} | {r['binder_B4']:.6g} | {r['binder_Q']:.6g} | {r['susceptibility']:.6g} | {r['xi_over_L']:.6g} |"
        for r in important
    )
    report = f"""# Binder Definition Audit

## Standard Definition

The primary Binder cumulant is now standardized as:

```text
m = (1/V) sum_x phi_x
U4 = 1 - <m^4> / (3 <m^2>^2)
```

The moments `<m^2>` and `<m^4>` are ensemble averages of the magnetization
density moments. The calculation does not use site moments `<phi^4>`, does not
use `|m|` inside `m2` or `m4`, and does not average a per-configuration Binder
quantity.

For transparency the central table also reports:

```text
B4 = <m^4> / <m^2>^2
Q  = <m^2>^2 / <m^4>
```

## Audit Result

- Canonical ensembles evaluated: `{len(rows)}`
- Excluded directories/files: `{len(excluded)}`
- CSV: `phi4_phase-diagram/reports/canonical_binder_observables.csv`
- JSON: `phi4_phase-diagram/reports/canonical_binder_observables.json`
- Plots: `phi4_phase-diagram/reports/plots/`

## Source Audit Notes

- `ML_sampling_clean/experiments/decimated_conditional_fillin/run_decimated_conditional_fillin.py`
  is the shared observable path used by the recent fill-in/proposal diagnostics.
  It now emits `Binder_U4`, `binder_U4`, `binder_B4`, `binder_Q`, and the legacy
  alias `Binder_ratio_B4`.
- The recent decimated/proposal diagnostics were annotated to point at this
  central Binder definition and report.
- The perfect-blocking optimization scripts inspected in `perfect_blocking/`
  already compute Binder from ensemble magnetization moments; their historical
  report labels are left as-is, but this central report should be used for
  canonical config comparisons.

The recent small Binder values in proposal diagnostics are not caused by the
shared `evaluate_ensemble` routine using field-site moments; that routine was
already using ensemble magnetization-density moments. The small values indicate
that those generated/proposal magnetization distributions are close to a broad
Gaussian-like shape, for which `B4` approaches 3 and `U4` approaches 0.

## Important Lambda Branches

| lambda | kappa | L | N | binder_U4 | binder_B4 | binder_Q | susceptibility | xi/L |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{table}
"""
    (REPORTS / "binder_definition_audit.md").write_text(report)
    print(json.dumps({"status": "completed", "n_canonical": len(rows), "n_excluded": len(excluded)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
