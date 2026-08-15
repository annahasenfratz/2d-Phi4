#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "phi4_phase-diagram" / "reports" / "lambda0p022_large_volume_reweighting"
IN_CSV = OUT / "lambda0p022_reweighted_best_ess_with_errors.csv"
KAPPA_CANDIDATES = [0.2705, 0.27075, 0.271, 0.27125]
VOLUMES = [16, 32, 64, 128]
KAPPA_MIN = 0.2695
KAPPA_MAX = 0.2725
ESS_MIN = 0.05


def read_rows() -> list[dict[str, Any]]:
    with IN_CSV.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


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


def collapse_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in rows:
        L = int(r["L"])
        kappa = float(r["target_kappa"])
        ess_fraction = float(r["ess_fraction"])
        if L not in VOLUMES or not (KAPPA_MIN <= kappa <= KAPPA_MAX) or ess_fraction < ESS_MIN:
            continue
        for kc in KAPPA_CANDIDATES:
            out.append(
                {
                    "kappa_c_trial": kc,
                    "L": L,
                    "target_kappa": kappa,
                    "collapse_x": (kappa - kc) * L,
                    "Binder_U4": float(r["Binder_U4"]),
                    "Binder_U4_se": float(r["Binder_U4_se"]),
                    "ess_fraction": ess_fraction,
                    "anchor_kappa": float(r["anchor_kappa"]),
                    "anchor_path": r["anchor_path"],
                }
            )
    return out


def plot(rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {16: "tab:orange", 32: "tab:green", 64: "tab:red", 128: "tab:purple"}
    markers = {16: "o", 32: "s", 64: "^", 128: "D"}

    fig, axes = plt.subplots(1, len(KAPPA_CANDIDATES), figsize=(14.2, 4.4), sharey=True)
    for ax, kc in zip(axes, KAPPA_CANDIDATES, strict=True):
        for L in VOLUMES:
            sub = [r for r in rows if int(r["L"]) == L and abs(float(r["kappa_c_trial"]) - kc) < 1e-12]
            if not sub:
                continue
            sub.sort(key=lambda r: float(r["collapse_x"]))
            x = np.asarray([float(r["collapse_x"]) for r in sub])
            y = np.asarray([float(r["Binder_U4"]) for r in sub])
            e = np.asarray([float(r["Binder_U4_se"]) for r in sub])
            ax.plot(x, y, lw=1.0, color=colors[L], alpha=0.75)
            stride = max(1, len(x) // 14)
            ax.errorbar(
                x[::stride],
                y[::stride],
                yerr=e[::stride],
                fmt=markers[L],
                ms=3.0,
                lw=0.0,
                elinewidth=0.7,
                capsize=1.5,
                color=colors[L],
                label=f"L={L}",
            )
        ax.axvline(0.0, color="black", lw=0.8, alpha=0.35)
        ax.set_title(rf"$\kappa_c={kc:.5f}$")
        ax.set_xlabel(r"$(\kappa-\kappa_c)L$")
        ax.grid(True, alpha=0.18)
    axes[0].set_ylabel("Binder U4")
    axes[-1].legend(fontsize=8, loc="best")
    fig.suptitle(r"lambda=0.022 Binder collapse diagnostic, $\nu=1$", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "lambda0p022_binder_collapse_kappac_trials.pdf")
    plt.close(fig)


def write_report(rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Lambda=0.022 Binder collapse diagnostic",
        "",
        r"Scaling variable: \(x=(\kappa-\kappa_c)L\), with \(\nu=1\).",
        "",
        f"Input: `{IN_CSV.name}`.",
        f"Kappa window: `{KAPPA_MIN} <= kappa <= {KAPPA_MAX}`.",
        f"ESS cut: `ESS/N >= {ESS_MIN}`.",
        "Volumes plotted: L=16,32,64,128 where available.",
        "",
        "| kappa_c trial | plotted points | x min | x max |",
        "|---:|---:|---:|---:|",
    ]
    for kc in KAPPA_CANDIDATES:
        sub = [r for r in rows if abs(float(r["kappa_c_trial"]) - kc) < 1e-12]
        xs = [float(r["collapse_x"]) for r in sub]
        lines.append(f"| {kc:.5f} | {len(sub)} | {min(xs):.6g} | {max(xs):.6g} |")
    lines += [
        "",
        "Outputs:",
        "",
        "- `lambda0p022_binder_collapse_kappac_trials.csv`",
        "- `lambda0p022_binder_collapse_kappac_trials.pdf`",
        "",
        "This is a diagnostic collapse plot only. It inherits the anchor-selection and jackknife errors from the reweighted max-ESS table.",
    ]
    (OUT / "LAMBDA0P022_BINDER_COLLAPSE_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    rows = collapse_rows(read_rows())
    write_csv(OUT / "lambda0p022_binder_collapse_kappac_trials.csv", rows)
    plot(rows)
    write_report(rows)
    print(
        {
            "rows": len(rows),
            "csv": str(OUT / "lambda0p022_binder_collapse_kappac_trials.csv"),
            "pdf": str(OUT / "lambda0p022_binder_collapse_kappac_trials.pdf"),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
