#!/usr/bin/env python3
"""Entry point placeholder for reusable 5x5 kernel training scans.

The current reorganization records the lambda=0.2 selected kernel and exposes
testing utilities. New training scans should use this entry point rather than
hard-coding lambda-specific paths in ad hoc scripts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.scan_utils import load_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--kappa-f", type=float)
    p.add_argument("--kappa-c", type=float)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    lam = cfg.get("lambda")
    family = cfg.get("kernel_family", "5x5")
    output_dir = args.output_dir or Path(cfg.get("candidate_output_dir", "kernels/candidates"))
    kappa_f = args.kappa_f if args.kappa_f is not None else cfg.get("kappa_f")
    kappa_c = args.kappa_c if args.kappa_c is not None else cfg.get("kappa_c")
    raise SystemExit(
        "kernel training scan is not implemented in the reorganized entry point yet; "
        f"config parsed successfully for lambda={lam}, family={family}, "
        f"kappa_f={kappa_f}, kappa_c={kappa_c}, output_dir={output_dir}. "
        "Port the selected legacy objective into scripts/common/kernel_objective.py "
        "before launching new scans."
    )


if __name__ == "__main__":
    main()
