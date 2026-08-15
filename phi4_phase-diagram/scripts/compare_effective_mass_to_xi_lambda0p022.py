#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PHASE = ROOT / "phi4_phase-diagram"
MASS = PHASE / "reports" / "lambda0p022_effective_mass"
RW = PHASE / "reports" / "lambda0p022_large_volume_reweighting"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
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


def exact_xi() -> dict[tuple[int, float], dict[str, float]]:
    rows = read_csv(RW / "lambda0p022_combined_cut_binder_chi_xi_table.csv")
    out = {}
    for r in rows:
        L = int(r["L"])
        if L not in {32, 64, 128}:
            continue
        k = round(float(r["kappa"]), 7)
        out[(L, k)] = {
            "xi_over_L": float(r["xi_over_L"]),
            "xi_over_L_se": float(r["xi_over_L_se_max"]),
            "xi_source": "exact_combined_cut",
            "xi_anchor_kappa": k,
            "xi_ess_fraction": 1.0,
        }
    return out


def reweighted_xi() -> dict[tuple[int, float], dict[str, float]]:
    rows = read_csv(RW / "lambda0p022_reweighted_best_ess_with_errors.csv")
    targets = {0.27075, 0.27100, 0.27125}
    out = {}
    for r in rows:
        L = int(r["L"])
        k = round(float(r["target_kappa"]), 7)
        if L not in {32, 64, 128} or not any(abs(k - t) < 1e-12 for t in targets):
            continue
        out[(L, k)] = {
            "xi_over_L": float(r["xi_over_L"]),
            "xi_over_L_se": float(r["xi_over_L_se"]),
            "xi_source": "reweighted_max_ess",
            "xi_anchor_kappa": float(r["anchor_kappa"]),
            "xi_ess_fraction": float(r["ess_fraction"]),
        }
    return out


def comparison_rows() -> list[dict[str, Any]]:
    exact_mass = read_csv(MASS / "lambda0p022_effective_mass_summary_by_L_kappa.csv")
    target_mass = read_csv(MASS / "lambda0p022_effective_mass_reweighted_targets_summary.csv")
    xi_exact = exact_xi()
    xi_rw = reweighted_xi()
    rows: list[dict[str, Any]] = []

    def add_row(r: dict[str, str], kkey: str, mode: str, xi_map: dict[tuple[int, float], dict[str, float]]) -> None:
        L = int(r["L"])
        k = round(float(r[kkey]), 7)
        xi = xi_map.get((L, k))
        if xi is None:
            return
        mL = float(r["meff_L_t_over_L_0p15_0p30_mean"])
        mL_scatter = float(r["meff_L_t_over_L_0p15_0p30_std"])
        xiL = float(xi["xi_over_L"])
        rows.append(
            {
                "mode": mode,
                "L": L,
                "kappa": k,
                "m_eff_L_window_mean": mL,
                "m_eff_L_window_scatter": mL_scatter,
                "xi_over_L": xiL,
                "xi_over_L_se": xi["xi_over_L_se"],
                "inverse_xi_over_L": 1.0 / xiL if xiL > 0 else float("nan"),
                "m_eff_L_times_xi_over_L": mL * xiL,
                "n_configs_mass": r["n_configs_used"],
                "mass_nonfinite_points": r["nonfinite_meff_points"],
                "mass_anchor_kappa": r.get("anchor_kappa", r.get("kappa", "")),
                "mass_ess_fraction": r.get("ess_fraction", 1.0),
                "xi_source": xi["xi_source"],
                "xi_anchor_kappa": xi["xi_anchor_kappa"],
                "xi_ess_fraction": xi["xi_ess_fraction"],
            }
        )

    for r in exact_mass:
        add_row(r, "kappa", "exact_anchor", xi_exact)
    for r in target_mass:
        add_row(r, "target_kappa", "reweighted_target", xi_rw)
    return rows


def make_plots(rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    colors = {32: "tab:green", 64: "tab:red", 128: "tab:purple"}
    markers = {"exact_anchor": "o", "reweighted_target": "s"}
    target_kappas = sorted({float(r["kappa"]) for r in rows if r["mode"] == "reweighted_target"})

    with PdfPages(MASS / "lambda0p022_m_eff_L_vs_xi_over_L_comparison.pdf") as pdf:
        for mode, title in [("exact_anchor", "exact native anchors"), ("reweighted_target", "reweighted target kappas")]:
            sub_all = [r for r in rows if r["mode"] == mode]
            if not sub_all:
                continue
            fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.6))
            for L in sorted({int(r["L"]) for r in sub_all}):
                sub = sorted([r for r in sub_all if int(r["L"]) == L], key=lambda r: float(r["kappa"]))
                x = np.asarray([float(r["kappa"]) for r in sub])
                mL = np.asarray([float(r["m_eff_L_window_mean"]) for r in sub])
                mLe = np.asarray([float(r["m_eff_L_window_scatter"]) for r in sub])
                inv_xi = np.asarray([float(r["inverse_xi_over_L"]) for r in sub])
                prod = np.asarray([float(r["m_eff_L_times_xi_over_L"]) for r in sub])
                axes[0].errorbar(x, mL, yerr=mLe, marker=markers[mode], color=colors[L], lw=1.0, capsize=2, label=f"L={L} m_eff L")
                axes[0].plot(x, inv_xi, marker="x", color=colors[L], ls="--", lw=1.0, alpha=0.75, label=f"L={L} 1/(xi/L)")
                axes[1].plot(x, prod, marker=markers[mode], color=colors[L], lw=1.0, label=f"L={L}")
            axes[0].set_xlabel("kappa")
            axes[0].set_ylabel("scaled quantity")
            axes[0].set_title(f"m_eff L vs 1/(xi/L), {title}")
            axes[0].grid(True, alpha=0.18)
            axes[0].legend(fontsize=7, ncol=2)
            axes[1].axhline(1.0, color="black", lw=0.8, alpha=0.35)
            axes[1].set_xlabel("kappa")
            axes[1].set_ylabel("(m_eff L) (xi/L)")
            axes[1].set_title("product diagnostic")
            axes[1].grid(True, alpha=0.18)
            axes[1].legend(fontsize=8)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        for k in target_kappas:
            sub = sorted([r for r in rows if r["mode"] == "reweighted_target" and abs(float(r["kappa"]) - k) < 1e-12], key=lambda r: int(r["L"]))
            fig, ax = plt.subplots(figsize=(7.2, 4.6))
            Ls = np.asarray([int(r["L"]) for r in sub])
            mL = np.asarray([float(r["m_eff_L_window_mean"]) for r in sub])
            mLe = np.asarray([float(r["m_eff_L_window_scatter"]) for r in sub])
            inv_xi = np.asarray([float(r["inverse_xi_over_L"]) for r in sub])
            ax.errorbar(Ls, mL, yerr=mLe, marker="o", lw=1.0, capsize=2, label="m_eff L")
            ax.plot(Ls, inv_xi, marker="x", ls="--", lw=1.0, label="1/(xi/L)")
            ax.set_xscale("log", base=2)
            ax.set_xticks(Ls)
            ax.set_xticklabels([str(int(x)) for x in Ls])
            ax.set_xlabel("L")
            ax.set_ylabel("scaled quantity")
            ax.set_title(f"lambda=0.022, target kappa={k:.5f}")
            ax.grid(True, alpha=0.18)
            ax.legend()
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def write_report(rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Lambda=0.022 effective mass versus xi/L",
        "",
        "`m_eff L` uses the folded-correlator window average over `0.15 <= t/L <= 0.30`.",
        "`xi/L` comes from the current Binder/chi/xi tables: exact combined-cut rows for exact anchors, and max-ESS reweighted rows for target kappas.",
        "",
        "A simple massive-pole expectation would make `m_eff L` comparable to `1/(xi/L)`, so the product `(m_eff L)(xi/L)` is reported as a diagnostic. The two xi definitions are not guaranteed to match exactly because the project xi is a proxy/second-moment-style observable and the effective mass is extracted from a noisy long-distance correlator.",
        "",
        "| mode | L | kappa | m_eff L | scatter | xi/L | SE | 1/(xi/L) | product | mass ESS | xi ESS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in sorted(rows, key=lambda x: (x["mode"], float(x["kappa"]), int(x["L"]))):
        lines.append(
            f"| {r['mode']} | {r['L']} | {float(r['kappa']):.5f} | "
            f"{float(r['m_eff_L_window_mean']):.6g} | {float(r['m_eff_L_window_scatter']):.6g} | "
            f"{float(r['xi_over_L']):.6g} | {float(r['xi_over_L_se']):.3g} | "
            f"{float(r['inverse_xi_over_L']):.6g} | {float(r['m_eff_L_times_xi_over_L']):.6g} | "
            f"{float(r['mass_ess_fraction']):.3g} | {float(r['xi_ess_fraction']):.3g} |"
        )
    lines += [
        "",
        "Outputs:",
        "",
        "- `lambda0p022_m_eff_L_vs_xi_over_L_comparison.csv`",
        "- `lambda0p022_m_eff_L_vs_xi_over_L_comparison.pdf`",
    ]
    (MASS / "M_EFF_L_VS_XI_OVER_L_COMPARISON.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    rows = comparison_rows()
    write_csv(MASS / "lambda0p022_m_eff_L_vs_xi_over_L_comparison.csv", rows)
    make_plots(rows)
    write_report(rows)
    print({"rows": len(rows), "out": str(MASS)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
