#!/usr/bin/env python3
"""Compare PC1 shifts across independent coarse-source blocks."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / "perfect_blocking_upsampling/outputs/controlled_patch_lam1p0/coarse_detail_L16to32"
PATTERN = "prod_cd_bL32_RQS_cfg*_N500_Pc16x1_Pd16x16_kf0p340301_kc0p340301"
NATIVE = ROOT / "data/configs_phi4_2d/lam1p0_kappac0p340301_L32/native_L32_all_observables_per_config.csv"
OUT = RUN_ROOT / "independent_source_block_pc1_audit.md"
ALPHA = 0.4525
BETA = 0.8918


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def pc1(r: dict[str, str]) -> float:
    return ALPHA * float(r["phi2"]) + BETA * float(r["phi4"])


def fmt(x: float) -> str:
    return f"{x:.9g}"


def main() -> None:
    native_values = np.asarray([pc1(r) for r in rows(NATIVE)], dtype=np.float64)
    native_mean = float(native_values.mean())
    run_data: list[tuple[str, int, int, dict[int, np.ndarray], str]] = []
    for run in sorted(RUN_ROOT.glob(PATTERN)):
        measurement = run / "observables/main_per_sweep_measurements.csv"
        if not measurement.exists():
            continue
        data = rows(measurement)
        by_sweep: dict[int, np.ndarray] = {}
        for sweep in sorted({int(r["sweep"]) for r in data}):
            selected = [r for r in data if int(r["sweep"]) == sweep]
            by_sweep[sweep] = np.asarray([pc1(r) for r in selected], dtype=np.float64)
        source = sorted({int(r["source_config_index"]) for r in data})
        status_path = run / "status.json"
        status = "unknown"
        if status_path.exists():
            import json

            status = str(json.loads(status_path.read_text()).get("status", "unknown"))
        run_data.append((run.name, source[0], source[-1], by_sweep, status))

    lines = [
        "# Independent coarse-source block PC1 audit",
        "",
        f"PC1 is `X = {ALPHA} phi2 + {BETA} phi4`. Native L32 PC1 mean: `{fmt(native_mean)}`.",
        "Source ranges are reported to verify that blocks do not reuse the same native coarse configurations.",
        "",
        "## Block inventory",
        "",
        "| run | source range | status | available sweeps |",
        "|---|---:|---|---|",
    ]
    for name, first, last, by_sweep, status in run_data:
        available = sorted(by_sweep)
        lines.append(f"| `{name}` | `{first}-{last}` | {status} | `{available[0]}..{available[-1]}` |")

    def block_table(sweep: int) -> list[tuple[str, float, float, float]]:
        out = []
        for name, first, last, by_sweep, status in run_data:
            if sweep not in by_sweep:
                continue
            x = by_sweep[sweep]
            shift = float(x.mean() - native_mean)
            se = float(x.std(ddof=1) / math.sqrt(len(x)))
            out.append((name, shift, se, len(x)))
        return out

    for sweep in (75, 100, 200, 250, 300):
        table = block_table(sweep)
        if not table:
            continue
        shifts = np.asarray([r[1] for r in table])
        ses = np.asarray([r[2] for r in table])
        lines += [
            "",
            f"## Sweep {sweep}",
            "",
            "| run | n | PC1 mean shift | within-block SE | shift / SE |",
            "|---|---:|---:|---:|---:|",
        ]
        for name, shift, se, n in table:
            lines.append(f"| `{name.split('_N500')[0]}` | {n} | {fmt(shift)} | {fmt(se)} | {fmt(shift / se)} |")
        if len(table) > 1:
            lines += [
                "",
                f"Block mean shift: `{fmt(float(shifts.mean()))}`; block-to-block SD: `{fmt(float(shifts.std(ddof=1)))}`; mean within-block SE: `{fmt(float(ses.mean()))}`; scatter/within-SE ratio: `{fmt(float(shifts.std(ddof=1) / ses.mean()))}`.",
            ]

    lines += [
        "",
        "## Long-chain movement in completed blocks",
        "",
        "| run | shift at 200 | shift at 250 | shift at 300 | 200->300 change |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, first, last, by_sweep, status in run_data:
        if not all(s in by_sweep for s in (200, 250, 300)):
            continue
        shifts = {s: float(by_sweep[s].mean() - native_mean) for s in (200, 250, 300)}
        lines.append(f"| `{name.split('_N500')[0]}` | {fmt(shifts[200])} | {fmt(shifts[250])} | {fmt(shifts[300])} | {fmt(shifts[300] - shifts[200])} |")

    lines += [
        "",
        "## Current reading",
        "",
        "At sweep 75, the independent blocks do not all show the same negative shift: one is positive and the block scatter is comparable to the within-block error. This is consistent with substantial statistical block-to-block variation at that time.",
        "At sweep 300, only cfg2000 and cfg2500 are complete. They agree at approximately `-0.0152` PC1 units, with negligible block-to-block scatter relative to their within-block SE. This supports a common late shift, but two completed late blocks are insufficient to separate a systematic coarse-marginal shift from a shared slow-mixing effect.",
        "The completed-block trajectories do not move PC1 toward zero from sweep 200 to 300; they become slightly more negative and then flatten. The three other source blocks are still running and should be compared at sweep 300 before making the final attribution.",
        "",
    ]
    OUT.write_text("\n".join(lines))
    print(OUT)


if __name__ == "__main__":
    main()
