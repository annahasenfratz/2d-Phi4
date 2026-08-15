#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description="Write a compact markdown summary for an upscaling run.")
    ap.add_argument("--run-dir", type=Path, required=True)
    args = ap.parse_args()
    status_path = args.run_dir / "status.json"
    config_path = args.run_dir / "run_config.yaml"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    lines = [
        "# Upscaling Run Summary",
        "",
        f"- run directory: `{args.run_dir}`",
        f"- run config: `{config_path}`",
        f"- status: `{status.get('status', 'unknown')}`",
        f"- current sweep: `{status.get('current_sweep', 'unknown')}`",
        f"- latest checkpoint: `{status.get('latest_checkpoint', 'unknown')}`",
        "",
        "Observables are expected under `observables/`; raw field configurations should remain under `data/configs_phi4_2d/` unless they are checkpoint state required for continuation.",
    ]
    out = args.run_dir / "summaries" / "run_summary.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

