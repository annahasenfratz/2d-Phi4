#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "perfect_blocking_upsampling" / "outputs" / "shape_parametric_sampler_validation"
IN_OP = OUT / "same_kappa_0p2705_operator_expected_vs_predicted.csv"
IN_AR = OUT / "same_kappa_0p2705_AR_logweight_summary.csv"
OUT_OP = OUT / "same_kappa_0p2705_operator_actual_values_with_errors.csv"
REPORT = OUT / "L32_to_L64_same_kappa_0p2705_operator_AR_summary.md"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def to_float(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def fmt(v: Any) -> str:
    x = to_float(v)
    if not math.isfinite(x):
        return "NA"
    return f"{x:.6g}"


def expanded_rows() -> list[dict[str, Any]]:
    rows = []
    for r in read_csv(IN_OP):
        native = to_float(r["native_L64_expected"])
        raw = to_float(r["raw_upscaled_predicted"])
        diff = raw - native
        rel = diff / native if math.isfinite(native) and native != 0.0 else float("nan")
        cse = to_float(r["combined_SE"])
        pull = to_float(r["pull"])
        rows.append(
            {
                "operator": r["operator"],
                "native_L64_value": native,
                "native_L64_SE": to_float(r["native_SE"]),
                "raw_upscaled_value": raw,
                "raw_upscaled_SE": to_float(r["upscaled_SE"]),
                "difference": diff,
                "relative_difference": rel,
                "combined_SE": cse,
                "pull": pull,
            }
        )
    rows.sort(key=lambda r: abs(to_float(r["pull"])) if math.isfinite(to_float(r["pull"])) else -1.0, reverse=True)
    return rows


def table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| operator | native_L64_value | native_L64_SE | raw_upscaled_value | raw_upscaled_SE | difference | relative_difference | combined_SE | pull |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['operator']} | {fmt(r['native_L64_value'])} | {fmt(r['native_L64_SE'])} | "
            f"{fmt(r['raw_upscaled_value'])} | {fmt(r['raw_upscaled_SE'])} | {fmt(r['difference'])} | "
            f"{fmt(r['relative_difference'])} | {fmt(r['combined_SE'])} | {fmt(r['pull'])} |"
        )
    return lines


def write_report(rows: list[dict[str, Any]]) -> None:
    ar = read_csv(IN_AR)[0]
    top = rows[:10]
    by_name = {r["operator"]: r for r in rows}
    lines = [
        "# L32->L64 same-kappa 0.2705 raw-upscaling operator/A-R summary",
        "",
        "This report expands the same-kappa sweep-0 comparison to show actual expectation values and errors, not only pulls.",
        "",
        "Source files:",
        "",
        f"- `{IN_OP}`",
        f"- `{IN_AR}`",
        "",
        "Pull definition:",
        "",
        r"\[",
        r"\mathrm{pull} = \frac{O_{\rm upscaled}-O_{\rm native}}{\sqrt{\sigma_{\rm upscaled}^2+\sigma_{\rm native}^2}}.",
        r"\]",
        "",
        "Susceptibility follows the project validation convention in this output, `chi = V <m^2>`, not the connected phase-diagram convention.",
        "",
        "## Top discrepancies",
        "",
        *table(top),
        "",
        "## All operators",
        "",
        *table(rows),
        "",
        "## A/R and logweight summary",
        "",
        "| lambda | kappa_c | kappa_f | Lc | Lf | N_states | logw_mean | logw_std | ESS/N | predicted independence acceptance | predicted adjacent acceptance | actual acceptance |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        f"| {fmt(ar['lambda'])} | {fmt(ar['kappa_c'])} | {fmt(ar['kappa_f'])} | {ar['Lc']} | {ar['Lf']} | {ar['N_states']} | "
        f"{fmt(ar['logw_mean'])} | {fmt(ar['logw_std'])} | {fmt(ar['ESS_per_N'])} | "
        f"{fmt(ar['predicted_independence_acceptance'])} | {fmt(ar['predicted_adjacent_acceptance'])} |  |",
        "",
        "**Actual A/R was not run; only logweight/predicted acceptance diagnostics are available.**",
        "",
        "## Interpretation",
        "",
        "1. What are the actual native L64 expectation values?",
        "",
        "They are listed in the `native_L64_value` column above. Representative native values at `(L64, kappa=0.2705)` are: "
        f"`phi2={fmt(by_name['phi2']['native_L64_value'])}`, `phi4={fmt(by_name['phi4']['native_L64_value'])}`, "
        f"`NN={fmt(by_name['NN']['native_L64_value'])}`, `action_density={fmt(by_name['action_density']['native_L64_value'])}`, "
        f"`m2={fmt(by_name['m2']['native_L64_value'])}`, `Binder_U4={fmt(by_name['Binder_U4']['native_L64_value'])}`, "
        f"`xi_over_L={fmt(by_name['xi_over_L']['native_L64_value'])}`.",
        "",
        "2. What are the raw upscaled expectation values?",
        "",
        "They are listed in the `raw_upscaled_value` column above. Representative raw-upscaled values are: "
        f"`phi2={fmt(by_name['phi2']['raw_upscaled_value'])}`, `phi4={fmt(by_name['phi4']['raw_upscaled_value'])}`, "
        f"`NN={fmt(by_name['NN']['raw_upscaled_value'])}`, `action_density={fmt(by_name['action_density']['raw_upscaled_value'])}`, "
        f"`m2={fmt(by_name['m2']['raw_upscaled_value'])}`, `Binder_U4={fmt(by_name['Binder_U4']['raw_upscaled_value'])}`, "
        f"`xi_over_L={fmt(by_name['xi_over_L']['raw_upscaled_value'])}`.",
        "",
        "3. Are the discrepancies physically large, or only statistically visible?",
        "",
        "They are physically visible in the raw-upscaled ensemble, not just microscopic statistical shifts. The largest relative changes include "
        f"`m4` at `{fmt(by_name['m4']['relative_difference'])}`, `m2` at `{fmt(by_name['m2']['relative_difference'])}`, "
        f"`susceptibility` at `{fmt(by_name['susceptibility']['relative_difference'])}`, `phi4` at `{fmt(by_name['phi4']['relative_difference'])}`, "
        f"and `xi_over_L` at `{fmt(by_name['xi_over_L']['relative_difference'])}`. Local action density itself shifts only by "
        f"`{fmt(by_name['action_density']['relative_difference'])}`, because component shifts partially cancel.",
        "",
        "4. Which sector is most affected?",
        "",
        "The amplitude/magnetization sector is most affected: `m2`, `m4`, susceptibility, `abs_m`, and `xi_over_L` are high. Local amplitude operators `phi2` and especially `phi4` are also high. Derivative/action observables (`NN`, `diag`, `2nn`) are shifted upward at the 7-8% level, while total `action_density` is much less shifted because onsite and hopping terms compensate.",
        "",
        "5. Does same-kappa raw upscaling look close enough that patch rethermalization should plausibly fix it?",
        "",
        "Plausibly, but not automatically. The raw state is not catastrophically wrong in local action density or logweight spread, and the A/R diagnostics are mechanically healthy (`logw_std=7.60`, `ESS/N=0.264`, predicted adjacent acceptance `0.714`). However, the raw observables are visibly displaced, especially amplitude and long-distance sector quantities. Patch rethermalization could fix this if the Markov updates mix those modes on the run length used; the raw-upscaled state alone is not already native-like.",
        "",
        "Machine-readable outputs:",
        "",
        f"- `{OUT_OP.name}`",
        f"- `{IN_AR.name}`",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    rows = expanded_rows()
    write_csv(OUT_OP, rows)
    write_report(rows)
    print({"rows": len(rows), "csv": str(OUT_OP), "report": str(REPORT)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
