#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from common.checkpoint_io import require_latest_checkpoint  # noqa: E402
from common.config_io import load_config, read_json, write_json  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Extend an existing upscaling run from checkpoint_latest.*.")
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--target-sweeps", required=True)
    ap.add_argument("--execute", action="store_true", help="Call configured legacy_extend_runner.")
    args, passthrough = ap.parse_known_args()
    try:
        target_sweeps = int(str(args.target_sweeps))
    except ValueError:
        raise SystemExit(f"--target-sweeps must be an integer without punctuation, got {args.target_sweeps!r}") from None

    run_dir = args.run_dir
    config = load_config(run_dir / "run_config.yaml")
    checkpoint = require_latest_checkpoint(run_dir)
    status = read_json(run_dir / "status.json") if (run_dir / "status.json").exists() else {}
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    extend_log = run_dir / "logs" / f"extend_{stamp}.log"
    command = " ".join(shlex.quote(x) for x in sys.argv)
    write_json(
        run_dir / "status.json",
        {
            **status,
            "status": "extension_prepared",
            "latest_checkpoint": str(checkpoint),
            "target_sweeps": target_sweeps,
            "last_extend_command": command,
            "resume_reuses_saved_state": True,
            "initializer_type": status.get("initializer_metadata", {}).get(
                "initializer_type", config.get("initializer_type", config.get("initialization_mode"))
            ),
        },
    )

    runner = config.get("legacy_extend_runner")
    if args.execute:
        if not runner:
            raise SystemExit("run_config.yaml has no legacy_extend_runner; extension prepared but cannot execute")
        cmd = [
            sys.executable,
            "-B",
            str(PROJECT_ROOT / str(runner)),
            "--run-dir",
            str(run_dir),
            "--target-sweeps",
            str(target_sweeps),
            *passthrough,
        ]
        with extend_log.open("a", encoding="utf-8") as log:
            return subprocess.call(cmd, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT)

    print(checkpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
