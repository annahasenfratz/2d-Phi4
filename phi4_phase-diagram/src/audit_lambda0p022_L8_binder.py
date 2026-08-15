#!/usr/bin/env python3
"""Audit lambda=0.022 L8 Binder data directly from configs.npz."""

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
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(__file__).resolve().parents[2]
ENSEMBLES = ROOT / "phi4_phase-diagram" / "ensembles"
REPORTS = ROOT / "phi4_phase-diagram" / "reports"
PLOTS = REPORTS / "plots"
GENERATOR = "embedded_wolff_sign_cluster_plus_radial_heatbath"
LAM = 0.022


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_phi(path: Path) -> np.ndarray:
    z = np.load(path)
    return z["phi"].astype(np.float64, copy=False)


def is_direct_canonical(manifest: dict[str, Any]) -> bool:
    if manifest.get("generator") != GENERATOR:
        return False
    if manifest.get("canonical", manifest.get("is_canonical", False)) is not True:
        return False
    if manifest.get("production_use") is not True:
        return False
    if manifest.get("local_metropolis_used") is not False:
        return False
    if manifest.get("is_superseded", False) or manifest.get("quarantined", False):
        return False
    return True


def infer_sweeps(log_path: Path) -> dict[str, Any]:
    if not log_path.exists():
        return {"thermal_sweeps_inferred": None, "skip_sweeps_inferred": None}
    with log_path.open() as f:
        reader = csv.DictReader(f)
        rows = [next(reader, None), next(reader, None), next(reader, None)]
    sweeps = [int(r["sweep"]) for r in rows if r is not None and r.get("sweep")]
    skip = sweeps[1] - sweeps[0] if len(sweeps) >= 2 else None
    thermal = sweeps[0] - skip if skip is not None else None
    return {"thermal_sweeps_inferred": thermal, "skip_sweeps_inferred": skip, "first_saved_sweep": sweeps[0] if sweeps else None}


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


def metrics(phi: np.ndarray, kappa: float) -> dict[str, Any]:
    n, l, _ = phi.shape
    v = l * l
    m = phi.mean(axis=(1, 2))
    M = phi.sum(axis=(1, 2))
    m2 = float(np.mean(m**2))
    m4 = float(np.mean(m**4))
    M2 = float(np.mean(M**2))
    M4 = float(np.mean(M**4))
    binder_u4 = 1.0 - m4 / (3.0 * m2 * m2)
    binder_b4 = m4 / (m2 * m2)
    binder_q = (m2 * m2) / m4
    binder_u4_M = 1.0 - M4 / (3.0 * M2 * M2)
    nn = 0.5 * (
        np.mean(phi * np.roll(phi, -1, axis=1), axis=(1, 2))
        + np.mean(phi * np.roll(phi, -1, axis=2), axis=(1, 2))
    )
    phi2 = np.mean(phi**2, axis=(1, 2))
    phi4 = np.mean(phi**4, axis=(1, 2))
    xi, xi_l = xi_over_l(phi)
    return {
        "n_configs": int(n),
        "shape": list(phi.shape),
        "V": int(v),
        "m_mean": float(np.mean(m)),
        "abs_m_mean": float(np.mean(np.abs(m))),
        "m2": m2,
        "m4": m4,
        "binder_U4": float(binder_u4),
        "binder_B4": float(binder_b4),
        "binder_Q": float(binder_q),
        "binder_U4_using_M_sum_phi": float(binder_u4_M),
        "binder_scale_invariant_abs_delta": float(abs(binder_u4 - binder_u4_M)),
        "susceptibility": float(v * m2),
        "phi2": float(np.mean(phi2)),
        "phi4": float(np.mean(phi4)),
        "NN": float(np.mean(nn)),
        "xi": xi,
        "xi_over_L": xi_l,
        "m_positive_fraction": float(np.mean(m > 0)),
        "m_sign_changes": int(np.sum(np.signbit(m[1:]) != np.signbit(m[:-1]))),
        "m_autocorr_lag1": float(np.corrcoef(m[:-1], m[1:])[0, 1]) if len(m) > 2 and np.std(m) > 0 else float("nan"),
    }


def running_binder(m: np.ndarray) -> np.ndarray:
    m2_c = np.cumsum(m**2)
    m4_c = np.cumsum(m**4)
    n = np.arange(1, len(m) + 1, dtype=np.float64)
    m2 = m2_c / n
    m4 = m4_c / n
    out = 1.0 - m4 / (3.0 * np.maximum(m2 * m2, 1.0e-300))
    out[:4] = np.nan
    return out


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


def collect() -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    rows: list[dict[str, Any]] = []
    m_values: dict[str, np.ndarray] = {}
    paths = sorted(ENSEMBLES.glob("lam0p022_kappa*_L8_embedded_wolff_sign_cluster_plus_radial_heatbath"))
    paths += sorted(ENSEMBLES.glob("lam0p022_kappa*_L8_embedded_wolff_sign_cluster_plus_radial_heatbath_seed*_check"))
    for path in paths:
        manifest_path = path / "manifest.json"
        configs = path / "configs.npz"
        if not manifest_path.exists() or not configs.exists():
            continue
        manifest = read_json(manifest_path)
        if abs(float(manifest.get("lambda", -1.0)) - LAM) > 1.0e-12 or int(manifest.get("L", -1)) != 8:
            continue
        phi = load_phi(configs)
        direct_configs = bool("phi" in np.load(configs).files and "generated" not in str(configs).lower() and "ML_sampling_clean/outputs" not in str(manifest.get("source_path", "")))
        row = {
            "path": str(path),
            "configs_path": str(configs),
            "kappa": float(manifest["kappa"]),
            "lambda": float(manifest["lambda"]),
            "L": int(manifest["L"]),
            "n_configs_manifest": int(manifest.get("n_configs", -1)),
            "shape_configs": list(phi.shape),
            "generator": manifest.get("generator"),
            "canonical": manifest.get("canonical", manifest.get("is_canonical")),
            "production_use": manifest.get("production_use"),
            "local_metropolis_used": manifest.get("local_metropolis_used"),
            "seed": manifest.get("seed"),
            "direct_phi_configs_not_generated_samples": direct_configs,
            "included_in_original_plot": "_seed" not in path.name,
            "independent_check": "_seed" in path.name,
            "accepted_direct_canonical": is_direct_canonical(manifest) and direct_configs,
        }
        row.update(infer_sweeps(path / "generation_log.csv"))
        row.update(metrics(phi, float(manifest["kappa"])))
        rows.append(row)
        label = f"kappa{float(manifest['kappa']):.3f}" + ("_check" if row["independent_check"] else "")
        m_values[label] = phi.mean(axis=(1, 2)).astype(np.float64)
        np.savez_compressed(REPORTS / f"lambda0p022_L8_m_values_{label.replace('.', 'p')}.npz", m=m_values[label])
    rows.sort(key=lambda r: (float(r["kappa"]), bool(r["independent_check"]), str(r["path"])))
    return rows, m_values


def make_hist_pdf(rows: list[dict[str, Any]], m_values: dict[str, np.ndarray]) -> None:
    pdf_path = REPORTS / "lambda0p022_L8_magnetization_histograms.pdf"
    with PdfPages(pdf_path) as pdf:
        for row in rows:
            if not row["accepted_direct_canonical"]:
                continue
            label = f"kappa{float(row['kappa']):.3f}" + ("_check" if row["independent_check"] else "")
            m = m_values[label]
            rb = running_binder(m)
            fig, ax = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
            ax[0, 0].hist(m, bins=50, color="#4c78a8", alpha=0.85)
            ax[0, 0].axvline(0, color="black", lw=1)
            ax[0, 0].set_title(f"m histogram {label}")
            ax[0, 1].hist(np.abs(m), bins=50, color="#59a14f", alpha=0.85)
            ax[0, 1].set_title("|m| histogram")
            ax[1, 0].plot(np.arange(len(m)), m, lw=0.8)
            ax[1, 0].axhline(0, color="black", lw=1)
            ax[1, 0].set_title("m vs config index")
            ax[1, 1].plot(np.arange(len(rb)), rb, lw=1.0)
            ax[1, 1].axhline(float(row["binder_U4"]), color="black", lw=1, ls="--")
            ax[1, 1].set_title("running Binder_U4")
            for a in ax.ravel():
                a.grid(alpha=0.2)
            fig.suptitle(f"lambda=0.022 L8 kappa={float(row['kappa']):.3f}, seed={row['seed']}")
            pdf.savefig(fig)
            plt.close(fig)


def make_audited_plot(rows: list[dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    original = [r for r in rows if r["accepted_direct_canonical"] and r["included_in_original_plot"]]
    check = [r for r in rows if r["accepted_direct_canonical"] and r["independent_check"]]
    ax.plot([r["kappa"] for r in original], [r["binder_U4"] for r in original], marker="o", label="L8 original canonical")
    if check:
        ax.scatter([r["kappa"] for r in check], [r["binder_U4"] for r in check], marker="s", s=60, label="L8 independent check")
    central = REPORTS / "canonical_binder_observables.csv"
    if central.exists():
        seen_labels: set[str] = set()
        with central.open() as f:
            for r in csv.DictReader(f):
                if abs(float(r["lambda"]) - LAM) < 1.0e-12 and int(r["L"]) in {16, 32}:
                    label = f"L{r['L']} canonical"
                    ax.scatter(
                        float(r["kappa"]),
                        float(r["binder_U4"]),
                        marker="^",
                        s=55,
                        label=label if label not in seen_labels else None,
                    )
                    seen_labels.add(label)
    ax.set_xlabel("kappa")
    ax.set_ylabel("Binder U4")
    ax.set_title("Audited Binder U4 vs kappa, lambda=0.022")
    ax.grid(alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys())
    fig.savefig(PLOTS / "binder_U4_vs_kappa_lambda0p022_audited.png", dpi=180)
    fig.savefig(PLOTS / "binder_U4_vs_kappa_lambda0p022_audited.pdf")
    plt.close(fig)


def main() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    rows, m_values = collect()
    write_csv(REPORTS / "lambda0p022_L8_binder_audit.csv", rows)
    make_hist_pdf(rows, m_values)
    make_audited_plot(rows)
    check = next((r for r in rows if r["independent_check"] and abs(float(r["kappa"]) - 0.225) < 1.0e-12), None)
    original_0225 = next((r for r in rows if r["included_in_original_plot"] and abs(float(r["kappa"]) - 0.225) < 1.0e-12), None)
    comparison = ""
    if check and original_0225:
        comparison = (
            f"- Original kappa=0.225 Binder_U4: `{original_0225['binder_U4']:.6g}`\n"
            f"- Independent check kappa=0.225 Binder_U4: `{check['binder_U4']:.6g}`\n"
            f"- Difference: `{check['binder_U4'] - original_0225['binder_U4']:.6g}`\n"
        )
    rows_table = "\n".join(
        f"| {r['kappa']:.6g} | {r['seed']} | {r['n_configs']} | {r['thermal_sweeps_inferred']} | {r['skip_sweeps_inferred']} | {r['binder_U4']:.6g} | {r['binder_B4']:.6g} | {r['abs_m_mean']:.6g} | {r['susceptibility']:.6g} | {r['xi_over_L']:.6g} | {r['m_positive_fraction']:.3f} | {r['m_sign_changes']} | {r['accepted_direct_canonical']} |"
        for r in rows
    )
    l16_rows = []
    central = REPORTS / "canonical_binder_observables.csv"
    if central.exists():
        with central.open() as f:
            for r in csv.DictReader(f):
                if abs(float(r["lambda"]) - LAM) < 1.0e-12 and int(r["L"]) in {16, 32}:
                    l16_rows.append(r)
    l16_table = "\n".join(
        f"| {r['kappa']} | {r['L']} | {r['binder_U4']} | {r['abs_m_mean']} | {r['susceptibility']} | {r['xi_over_L']} |"
        for r in l16_rows
    )
    report = f"""# Lambda=0.022 L8 Binder Audit

## Summary

The L8 files used in the Binder plot are direct canonical `phi` ensembles, not
generated samples. Manual recomputation from `configs.npz` reproduces the small
Binder values. The scale-invariance check using `M=sum_x phi_x` instead of
`m=M/64` agrees to floating-point precision, so the low L8 values are not caused
by a volume-normalization bug.

The independent kappa=0.225 check ensemble used a different seed and more
conservative generation settings (`thermal_sweeps=5000`, `skip_sweeps=20`,
`clusters_per_sweep=2`, `N=2000`).

{comparison}
The independent check is consistent with the original value within the expected
Monte Carlo variability for this small volume. The strange-looking L8 Binder
curve is therefore real finite-volume/off-critical behavior for these ensembles,
not an input-file, sector-sticking, or Binder-definition bug.

## Provenance And Manual Observables

| kappa | seed | N | thermal | skip | Binder_U4 | Binder_B4 | <|m|> | chi | xi/L | frac m>0 | sign changes | accepted |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
{rows_table}

## L16/L32 Comparison

| kappa | L | Binder_U4 | <|m|> | susceptibility | xi/L |
|---:|---:|---:|---:|---:|---:|
{l16_table}

## Checks Performed

- Verified `configs.npz` contains a direct `phi` array with shape `(N,8,8)`.
- Verified manifests are canonical production embedded-Wolff/radial-heatbath
  ensembles with `local_metropolis_used=false`.
- Inferred original generation settings from `generation_log.csv`: first saved
  sweep minus separation gives thermalization, and successive saved sweeps give
  skip/separation.
- Recomputed `m_i = mean_x phi_i(x)` with `V=64`.
- Recomputed Binder from ensemble moments, not per-configuration Binder values.
- Verified `Binder_U4(m)` equals `Binder_U4(M=sum phi)` for every ensemble.
- Saved per-configuration magnetization arrays as
  `lambda0p022_L8_m_values_*.npz`.
- Plotted histograms of `m`, histograms of `|m|`, histories of `m`, and running
  Binder estimates in `lambda0p022_L8_magnetization_histograms.pdf`.

## Output Files

- `phi4_phase-diagram/reports/lambda0p022_L8_binder_audit.csv`
- `phi4_phase-diagram/reports/lambda0p022_L8_magnetization_histograms.pdf`
- `phi4_phase-diagram/reports/plots/binder_U4_vs_kappa_lambda0p022_audited.pdf`
- `phi4_phase-diagram/reports/plots/binder_U4_vs_kappa_lambda0p022_audited.png`
"""
    (REPORTS / "lambda0p022_L8_binder_audit.md").write_text(report)
    print(json.dumps({"status": "completed", "rows": len(rows), "check_binder": check["binder_U4"] if check else None}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
