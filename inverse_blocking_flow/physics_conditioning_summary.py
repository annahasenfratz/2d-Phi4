"""Summarize physics-conditioned flow diagnostics against baseline flows."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/matplotlib")


CHANNELS = ["HL", "LH", "HH"]
ROOT = Path("inverse_blocking_flow/outputs")


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def mean_abs_error(values: dict[str, float], target: dict[str, float]) -> float:
    return sum(abs(values[ch] - target[ch]) for ch in CHANNELS) / len(CHANNELS)


def channel_values(detail: dict[str, object], key: str) -> dict[str, float]:
    return {ch: float(detail["channels"][ch][key]) for ch in CHANNELS}


def kappa_row(scan: dict[str, object], kappa: float) -> dict[str, object] | None:
    for row in scan["scan"]:
        if abs(float(row["kappa_f"]) - kappa) < 1e-12:
            return row
    return None


def fmt(x: object) -> str:
    if x is None:
        return "missing"
    try:
        return f"{float(x):.6g}"
    except (TypeError, ValueError):
        return str(x)


def write_report(path: Path, summary: dict[str, object]) -> None:
    lines = [
        "# Physics Conditioning Summary",
        "",
        "Physics conditioning uses six coarse-lattice channels: `phi_c`, `phi_c^2`, centered periodic `grad_x`, centered periodic `grad_y`, `grad_sq`, and periodic laplacian. The blocking/reconstruction map and detail variables are unchanged.",
        "",
        "## Detail Diagnostics",
        "",
        "| ensemble | std HL | std LH | std HH | mean abs std error | corr(phi_c^2,d_HL^2) | corr(phi_c^2,d_LH^2) | corr(phi_c^2,d_HH^2) | mean abs conditioning error |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in summary["detail_rows"].items():
        std = row["std"]
        cond = row["corr_phi_c2_detail_amp"]
        lines.append(
            f"| {name} | {fmt(std['HL'])} | {fmt(std['LH'])} | {fmt(std['HH'])} | {fmt(row['mean_abs_std_error'])} | "
            f"{fmt(cond['HL'])} | {fmt(cond['LH'])} | {fmt(cond['HH'])} | {fmt(row['mean_abs_phi_c2_amp_corr_error'])} |"
        )
    lines.extend(
        [
            "",
            "## Kappa Logw Scan",
            "",
            "| ensemble | kappa_min width | min logw std | ESS/N at min | A/R proxy at min | kappa_min obs | best obs agg err | agg err at kappa_true |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, row in summary["kappa_rows"].items():
        lines.append(
            f"| {name} | {fmt(row['preferred_kappa_by_logw_width'])} | {fmt(row['min_logw_std'])} | "
            f"{fmt(row['min_ess_over_n'])} | {fmt(row['min_ar_proxy'])} | {fmt(row['preferred_kappa_by_observables'])} | "
            f"{fmt(row['best_observable_aggregate_error'])} | {fmt(row['aggregate_error_at_kappa_true'])} |"
        )
    answers = summary["answers"]
    lines.extend(
        [
            "",
            "## Main Answers",
            "",
            f"- Does physics conditioning increase detail variances toward true values? {answers['detail_variance']}",
            f"- Does it improve corr(phi_c^2,d^2)? {answers['phi_c2_conditioning']}",
            f"- Does kappa_min move toward 0.31? {answers['kappa_min']}",
            f"- Does logw std decrease? {answers['logw_std']}",
            f"- Does ESS/N improve? {answers['ess']}",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def write_plots(path: Path, summary: dict[str, object]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ["true_details", "baseline_reverse_kl", "logwvar_alpha_0p01", "physics_conditioned"]
    labels = ["true", "baseline", "logwvar", "physics"]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5))
    ax = axes[0, 0]
    x = torch.arange(len(CHANNELS)).numpy()
    width = 0.2
    for i, (name, label) in enumerate(zip(names, labels)):
        std = summary["detail_rows"][name]["std"]
        ax.bar(x + (i - 1.5) * width, [std[ch] for ch in CHANNELS], width=width, label=label)
    ax.set_xticks(x, CHANNELS)
    ax.set_ylabel("detail std")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    for i, (name, label) in enumerate(zip(names, labels)):
        cond = summary["detail_rows"][name]["corr_phi_c2_detail_amp"]
        ax.bar(x + (i - 1.5) * width, [cond[ch] for ch in CHANNELS], width=width, label=label)
    ax.set_xticks(x, CHANNELS)
    ax.set_ylabel("corr(phi_c^2,d^2)")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    scan_names = ["baseline_reverse_kl", "logwvar_alpha_0p01", "physics_conditioned"]
    ax.bar(range(len(scan_names)), [summary["kappa_rows"][n]["min_logw_std"] for n in scan_names])
    ax.set_xticks(range(len(scan_names)), ["baseline", "logwvar", "physics"], rotation=20, ha="right")
    ax.set_ylabel("min std(logw)")

    ax = axes[1, 1]
    ax.bar(range(len(scan_names)), [summary["kappa_rows"][n]["preferred_kappa_by_logw_width"] for n in scan_names])
    ax.axhline(0.31, color="k", ls="--", lw=1.0, label="true")
    ax.set_xticks(range(len(scan_names)), ["baseline", "logwvar", "physics"], rotation=20, ha="right")
    ax.set_ylabel("kappa_min width")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    detail = load_json(ROOT / "detail_correlation_diagnostics_summary.json")
    scans = {
        "baseline_reverse_kl": load_json(ROOT / "logw_kappa_scan_blocked_summary.json"),
        "logwvar_alpha_0p01": load_json(ROOT / "logw_var_penalty/scan_logwvar_alpha_0p01/logw_kappa_scan_blocked_summary.json"),
        "physics_conditioned": load_json(ROOT / "physics_kappa_scan/logw_kappa_scan_blocked_summary.json"),
    }
    true_detail = detail["ensembles"]["true_details"]
    true_std = channel_values(true_detail, "std")
    true_cond = channel_values(true_detail, "corr_phi_c2_detail_amp")
    detail_rows = {}
    for name, diag in detail["ensembles"].items():
        std = channel_values(diag, "std")
        cond = channel_values(diag, "corr_phi_c2_detail_amp")
        detail_rows[name] = {
            "std": std,
            "corr_phi_c2_detail_amp": cond,
            "mean_abs_std_error": mean_abs_error(std, true_std),
            "mean_abs_phi_c2_amp_corr_error": mean_abs_error(cond, true_cond),
        }
    kappa_rows = {}
    for name, scan in scans.items():
        best_width = scan["best_by_logw_width"]
        best_obs = scan["best_by_observable_error"]
        true_row = kappa_row(scan, 0.31)
        kappa_rows[name] = {
            "preferred_kappa_by_logw_width": float(best_width["kappa_f"]),
            "min_logw_std": float(best_width["logw"]["std_logw_centered"]),
            "min_ess_over_n": float(best_width["logw"]["ess_over_n"]),
            "min_ar_proxy": float(best_width["logw"]["independence_acceptance_proxy"]),
            "preferred_kappa_by_observables": float(best_obs["kappa_f"]),
            "best_observable_aggregate_error": float(best_obs["aggregate_abs_rel_error_vs_true_kappa_true"]),
            "aggregate_error_at_kappa_true": None if true_row is None else float(true_row["aggregate_abs_rel_error_vs_true_kappa_true"]),
        }
    base = kappa_rows["baseline_reverse_kl"]
    phys = kappa_rows["physics_conditioned"]
    base_detail = detail_rows["baseline_reverse_kl"]
    phys_detail = detail_rows["physics_conditioned"]
    summary = {
        "detail_rows": detail_rows,
        "kappa_rows": kappa_rows,
        "answers": {
            "detail_variance": (
                f"Yes. Mean abs std error drops from {base_detail['mean_abs_std_error']:.6g} to {phys_detail['mean_abs_std_error']:.6g}."
            ),
            "phi_c2_conditioning": (
                f"Mostly yes. Mean abs corr(phi_c^2,d^2) error drops from {base_detail['mean_abs_phi_c2_amp_corr_error']:.6g} to {phys_detail['mean_abs_phi_c2_amp_corr_error']:.6g}."
            ),
            "kappa_min": (
                f"Yes. Width-minimizing kappa moves from {base['preferred_kappa_by_logw_width']:.6g} to {phys['preferred_kappa_by_logw_width']:.6g}, closer to 0.31."
            ),
            "logw_std": (
                f"Yes. Minimum logw std drops from {base['min_logw_std']:.6g} to {phys['min_logw_std']:.6g}."
            ),
            "ess": (
                f"No. ESS/N at the width minimum changes from {base['min_ess_over_n']:.6g} to {phys['min_ess_over_n']:.6g}; it remains far below viability."
            ),
        },
    }
    (ROOT / "physics_conditioning_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_report(ROOT / "physics_conditioning_report.md", summary)
    write_plots(ROOT / "physics_conditioning_plots.pdf", summary)
    print(f"wrote {ROOT / 'physics_conditioning_summary.json'}")
    print(f"wrote {ROOT / 'physics_conditioning_report.md'}")
    print(f"wrote {ROOT / 'physics_conditioning_plots.pdf'}")


if __name__ == "__main__":
    main()
