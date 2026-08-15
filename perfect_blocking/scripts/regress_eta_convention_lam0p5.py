#!/usr/bin/env python3
"""Regression check for the lambda=0.5 eta-included final kernel convention."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.blocking import block_configs, load_configs  # noqa: E402
from scripts.common.kernel_io import load_kernel  # noqa: E402


DEFAULT_KERNEL = PROJECT_ROOT / "perfect_blocking/perfect_blocking_lam0p5/kernels/final/chosen_kernel.json"
DEFAULT_CONFIGS = PROJECT_ROOT / "data/configs_phi4_2d/lam0p5_kappac0p343469_L32/configs.npz"
DEFAULT_OUT_ROOT = PROJECT_ROOT / "perfect_blocking/perfect_blocking_lam0p5/debug"
ETA = 0.25
ETA_SCALE = 2.0 ** (ETA / 2.0)


def json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(type(obj).__name__)


def blocked_means(blocked: np.ndarray) -> dict[str, float]:
    arr = np.asarray(blocked, dtype=np.float64)
    nn = 0.5 * (
        np.mean(arr * np.roll(arr, -1, axis=1))
        + np.mean(arr * np.roll(arr, -1, axis=2))
    )
    diag = np.mean(arr * np.roll(np.roll(arr, -1, axis=1), -1, axis=2))
    m = np.mean(arr, axis=(1, 2))
    volume = int(arr.shape[-1] * arr.shape[-1])
    g00 = volume * m * m
    return {
        "phi2": float(np.mean(arr**2)),
        "phi4": float(np.mean(arr**4)),
        "NN": float(nn),
        "diag": float(diag),
        "m2": float(np.mean(m * m)),
        "m4": float(np.mean(m**4)),
        "G_00_mean": float(np.mean(g00)),
        "G_00_from_m2": float(volume * np.mean(m * m)),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kernel", type=Path, default=DEFAULT_KERNEL)
    p.add_argument("--configs", type=Path, default=DEFAULT_CONFIGS)
    p.add_argument("--max-configs", type=int, default=512)
    p.add_argument("--out-dir", type=Path)
    args = p.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.out_dir or (DEFAULT_OUT_ROOT / f"eta_convention_regression_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    kernel = load_kernel(args.kernel)
    metadata = kernel.metadata
    kernel_sum = float(kernel.matrix.sum())
    checks: list[dict[str, Any]] = []
    checks.append(
        {
            "check": "chosen_kernel_exists",
            "observed": args.kernel.exists(),
            "expected": True,
            "passed": args.kernel.exists(),
        }
    )
    checks.append(
        {
            "check": "kernel_coefficients_include_eta_scale",
            "observed": bool(metadata.get("kernel_coefficients_include_eta_scale")),
            "expected": True,
            "passed": bool(metadata.get("kernel_coefficients_include_eta_scale")) is True,
        }
    )
    checks.append(
        {
            "check": "kernel_sum",
            "observed": kernel_sum,
            "expected": ETA_SCALE,
            "passed": math.isclose(kernel_sum, ETA_SCALE, rel_tol=1.0e-12, abs_tol=1.0e-12),
        }
    )

    phi = load_configs(args.configs, max_configs=args.max_configs)
    blocked = block_configs(phi, kernel, block_factor=2)
    means = blocked_means(blocked)
    finite = all(np.isfinite(v) for v in means.values()) and bool(np.all(np.isfinite(blocked)))
    checks.append({"check": "blocked_observables_finite", "observed": finite, "expected": True, "passed": finite})
    g00_delta = abs(means["G_00_mean"] - means["G_00_from_m2"])
    checks.append(
        {
            "check": "G_00_equals_volume_m2",
            "observed": g00_delta,
            "expected": 0.0,
            "passed": g00_delta <= 1.0e-10,
        }
    )
    no_double_eta = kernel.kernel_coefficients_include_eta_scale and math.isclose(kernel_sum, ETA_SCALE, rel_tol=1.0e-12, abs_tol=1.0e-12)
    checks.append(
        {
            "check": "no_extra_eta_multiplier_required",
            "observed": no_double_eta,
            "expected": True,
            "passed": no_double_eta,
        }
    )

    passed = all(bool(row["passed"]) for row in checks)
    summary = {
        "passed": passed,
        "kernel": args.kernel,
        "configs": args.configs,
        "max_configs": int(args.max_configs),
        "eta": ETA,
        "eta_scale": ETA_SCALE,
        "kernel_sum": kernel_sum,
        "blocked_shape": list(blocked.shape),
        "means": means,
        "checks": checks,
        "generated_utc": stamp,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=json_default) + "\n")
    lines = [
        "# Lambda 0.5 eta convention regression",
        "",
        f"- generated UTC: `{stamp}`",
        f"- passed: `{passed}`",
        f"- kernel: `{args.kernel}`",
        f"- configs: `{args.configs}`",
        f"- max configs: `{args.max_configs}`",
        f"- eta_scale: `{ETA_SCALE:.17g}`",
        f"- kernel sum: `{kernel_sum:.17g}`",
        "",
        "| check | observed | expected | passed |",
        "|---|---:|---:|---|",
    ]
    for row in checks:
        lines.append(f"| {row['check']} | {row['observed']} | {row['expected']} | {row['passed']} |")
    lines.extend(["", "## Blocked Means", "", "| observable | value |", "|---|---:|"])
    for key, value in means.items():
        lines.append(f"| {key} | {value:.17g} |")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"passed": passed, "out_dir": str(out_dir), "kernel_sum": kernel_sum}, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
