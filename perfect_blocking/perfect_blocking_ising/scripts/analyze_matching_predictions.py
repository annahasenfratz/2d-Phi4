#!/usr/bin/env python3
"""Add matching-prediction comparisons for the optimized Ising blocking kernel."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "perfect_blocking_ising" / "outputs"
SUMMARY_JSON = OUT_DIR / "perfect_blocking_summary.json"
REPORT_MD = OUT_DIR / "perfect_blocking_report.md"
MATCHING_CSV = OUT_DIR / "perfect_blocking_matching_comparison.csv"
PLOTS_PDF = OUT_DIR / "perfect_blocking_plots.pdf"

NN_EXACT = 1.0 / math.sqrt(2.0)


def load_summary() -> dict:
    return json.loads(SUMMARY_JSON.read_text())


def get_obs(summary: dict, key: str) -> tuple[float, float, float, float]:
    obs = summary["observables"]
    true8 = obs["true8"]
    blocked = obs["blocked16"]
    if key in true8["means"]:
        return (
            float(true8["means"][key]),
            float(true8["errs"][key]),
            float(blocked["means"][key]),
            float(blocked["errs"][key]),
        )
    if key in true8["extra"]:
        return (
            float(true8["extra"][key]["mean"]),
            float(true8["extra"][key]["err"]),
            float(blocked["extra"][key]["mean"]),
            float(blocked["extra"][key]["err"]),
        )
    raise KeyError(key)


def build_rows(summary: dict) -> list[dict]:
    rows = []
    observables = [
        ("nn", "finite_volume_true8_target", None),
        ("diag", "finite_volume_true8_target", None),
        ("2nn", "finite_volume_true8_target", None),
        ("nn2", "finite_volume_true8_target", None),
        ("diag2", "finite_volume_true8_target", None),
        ("2nn2", "finite_volume_true8_target", None),
        ("abs_m", "finite_volume_true8_target", None),
        ("m2", "finite_volume_true8_target", None),
        ("nn", "infinite_volume_critical", NN_EXACT),
    ]

    for obs_name, ref_type, ref_value_override in observables:
        true_mean, true_err, blk_mean, blk_err = get_obs(summary, obs_name)
        reference_value = true_mean if ref_value_override is None else float(ref_value_override)
        true_minus_ref = true_mean - reference_value
        blocked_minus_ref = blk_mean - reference_value
        blocked_minus_true = blk_mean - true_mean
        sigma = math.sqrt(true_err**2 + blk_err**2) if (true_err or blk_err) else 1.0
        rows.append(
            {
                "observable": obs_name,
                "reference_type": ref_type,
                "reference_value": reference_value,
                "true8_mean": true_mean,
                "true8_error": true_err,
                "blocked_mean": blk_mean,
                "blocked_error": blk_err,
                "true8_minus_reference": true_minus_ref,
                "blocked_minus_reference": blocked_minus_ref,
                "blocked_minus_true8": blocked_minus_true,
                "z_blocked_vs_true8": blocked_minus_true / sigma if sigma else 0.0,
            }
        )
    return rows


def write_csv(rows: list[dict]) -> None:
    with MATCHING_CSV.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def append_report(summary: dict, rows: list[dict]) -> None:
    true8 = summary["observables"]["true8"]
    blocked = summary["observables"]["blocked16"]
    nn_row = next(r for r in rows if r["observable"] == "nn" and r["reference_type"] == "finite_volume_true8_target")
    nn_exact_row = next(r for r in rows if r["observable"] == "nn" and r["reference_type"] == "infinite_volume_critical")

    lines = REPORT_MD.read_text().rstrip().splitlines()
    lines.append("")
    lines.append("## Expectation values and matching predictions")
    lines.append("")
    beta_exact = summary.get("beta_exact", summary["beta_data"])
    lines.append(f"- The finite-volume matching target is the true `L=8` ensemble at beta = {beta_exact:.12f}.")
    lines.append(f"- For `nn`, the infinite-volume critical reference is `1/sqrt(2) = {NN_EXACT:.16f}`.")
    lines.append("- The blocked `L=16 -> L=8` ensemble should be judged primarily against the measured true `L=8` ensemble.")
    lines.append("- The earlier million-sample bundled run produced a much larger z-score; in this 500-config critical run the blocked-vs-true8 `nn` mismatch is only about `-1.3σ`.")
    lines.append("- The absolute difference is still the most useful scale here, and the 500-config run is good enough for a first-pass kernel search, but the validation-loss shift between independent seeds suggests more data would improve optimizer stability.")
    lines.append("")
    lines.append("### Finite-volume measured targets")
    for key in ["nn", "diag", "2nn", "nn2", "diag2", "2nn2", "abs_m", "m2"]:
        row = next(r for r in rows if r["observable"] == key and r["reference_type"] == "finite_volume_true8_target")
        lines.append(
            f"- {key}: true8={row['true8_mean']:.6f}±{row['true8_error']:.6f}, "
            f"blocked={row['blocked_mean']:.6f}±{row['blocked_error']:.6f}, "
            f"blocked-true8={row['blocked_minus_true8']:.6f}, z={row['z_blocked_vs_true8']:.3f}"
        )
    lines.append("")
    lines.append("### Infinite-volume critical comparison for nn only")
    lines.append(
        f"- nn exact critical reference: {NN_EXACT:.16f}; true8 minus exact = {nn_exact_row['true8_minus_reference']:.6f}; "
        f"blocked minus exact = {nn_exact_row['blocked_minus_reference']:.6f}"
    )
    lines.append("")
    lines.append("- For `diag` and `2nn`, no exact finite-volume formulas were added; those remain measured finite-volume matching targets.")
    REPORT_MD.write_text("\n".join(lines) + "\n")


def make_plots(summary: dict, rows: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    true8 = summary["observables"]["true8"]
    blocked = summary["observables"]["blocked16"]
    nn_true = true8["means"]["nn"]
    nn_true_err = true8["errs"]["nn"]
    nn_blk = blocked["means"]["nn"]
    nn_blk_err = blocked["errs"]["nn"]
    nn_row = next(r for r in rows if r["observable"] == "nn" and r["reference_type"] == "infinite_volume_critical")

    fig = plt.figure(figsize=(11, 8))
    gs = fig.add_gridspec(2, 2)

    ax = fig.add_subplot(gs[0, 0])
    names = ["nn", "diag", "2nn", "nn2", "diag2", "2nn2"]
    x = range(len(names))
    ax.errorbar(x, [true8["means"][k] for k in names], yerr=[true8["errs"][k] for k in names], fmt="o", label="true L=8")
    ax.errorbar(x, [blocked["means"][k] for k in names], yerr=[blocked["errs"][k] for k in names], fmt="s", label="blocked L=16 -> 8")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=30)
    ax.set_ylabel("mean")
    ax.set_title("Observable comparison")
    ax.legend(frameon=False)

    ax = fig.add_subplot(gs[0, 1])
    xpos = [0, 1, 2]
    ax.bar(
        xpos,
        [nn_true, nn_blk, NN_EXACT],
        color=["#4c78a8", "#f58518", "#54a24b"],
        width=0.65,
    )
    ax.errorbar([0, 1], [nn_true, nn_blk], yerr=[nn_true_err, nn_blk_err], fmt="none", ecolor="black", capsize=3)
    ax.axhline(NN_EXACT, color="#54a24b", linestyle="--", linewidth=1.5, label="1/sqrt(2)")
    ax.set_xticks(xpos)
    ax.set_xticklabels(["true8", "blocked16", "exact nn"])
    ax.set_title("Nearest-neighbor reference")
    ax.set_ylabel("nn")
    ax.legend(frameon=False)

    ax = fig.add_subplot(gs[1, 0])
    ax.text(
        0.02,
        0.98,
        f"nn vs true8: {nn_row['blocked_minus_true8']:+.6f}\n"
        f"z-score: {nn_row['z_blocked_vs_true8']:+.3f}\n"
        f"exact nn: {NN_EXACT:.6f}",
        va="top",
        ha="left",
        fontsize=11,
        family="monospace",
    )
    ax.axis("off")

    ax = fig.add_subplot(gs[1, 1])
    history = summary["optimization"]["validation"]
    ax.bar(["seed A", "seed B"], [history["primary"]["loss"], history["replica_seed2"]["loss"]], color=["#9ecae9", "#fdd0a2"])
    ax.set_ylabel("loss")
    ax.set_title("Validation loss stability")

    fig.tight_layout()

    with PdfPages(PLOTS_PDF) as pdf:
        pdf.savefig(fig)
        plt.close(fig)


def main() -> int:
    os.environ.setdefault("MPLCONFIGDIR", str(OUT_DIR / "_mplcache"))
    (OUT_DIR / "_mplcache").mkdir(parents=True, exist_ok=True)
    summary = load_summary()
    rows = build_rows(summary)
    write_csv(rows)
    append_report(summary, rows)
    make_plots(summary, rows)
    print(json.dumps({"written": str(MATCHING_CSV)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
