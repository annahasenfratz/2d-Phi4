#!/usr/bin/env python3
"""Detached launcher for controlled L16->L32 checkpoint chain validation."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON = PROJECT_ROOT.parent / ".venv/bin/python"
SCRIPT = PROJECT_ROOT / "perfect_blocking_upsampling/scripts/run_l16to32_controlled_chain_footprint.py"
ROOT = PROJECT_ROOT / "perfect_blocking_upsampling/outputs/lam0p022_L16to32_flow_footprint_scan"

RUNS = {
    "controlled_chain_fp_medium_1_baseline": {
        "candidate": "controlled_chain_fp_medium_1_baseline",
        "footprint": 11,
        "seed": 2026070411,
        "output_dir": ROOT / "controlled_chain_fp_medium_1_baseline",
        "edge_x": ROOT / "fp_medium_1/checkpoints/edge_x/checkpoint_best.pt",
        "edge_y": ROOT / "fp_medium_1/checkpoints/edge_y/checkpoint_best.pt",
        "body": ROOT / "fp_medium_1/checkpoints/body/checkpoint_best.pt",
    },
    "controlled_chain_fp_medium_1_deep_epoch45": {
        "candidate": "controlled_chain_fp_medium_1_deep_epoch45",
        "footprint": 11,
        "seed": 2026070412,
        "output_dir": ROOT / "controlled_chain_fp_medium_1_deep_epoch45",
        "edge_x": ROOT / "fp_medium_1_deep/checkpoints/edge_x/epoch0045.pt",
        "edge_y": ROOT / "fp_medium_1_deep/checkpoints/edge_y/epoch0045.pt",
        "body": ROOT / "fp_medium_1_deep/checkpoints/body/epoch0045.pt",
    },
}


def command(run: dict, preflight_only: bool = False) -> list[str]:
    cmd = [
        str(PYTHON), "-u", "-B", str(SCRIPT),
        "--candidate", run["candidate"],
        "--output-dir", str(run["output_dir"]),
        "--footprint", str(run["footprint"]),
        "--edge-x-checkpoint", str(run["edge_x"]),
        "--edge-y-checkpoint", str(run["edge_y"]),
        "--body-checkpoint", str(run["body"]),
        "--seed", str(run["seed"]),
        "--n-chains", "8",
        "--n-sweeps", "2000",
        "--record-every", "20",
        "--p-coarse", "12",
        "--coarse-patches-per-sweep", "4",
        "--coarse-passes", "5",
        "--epsilon-c", "0.6",
        "--p-detail", "12",
        "--beta-z", "0.4",
        "--n-detail-updates-per-sweep", "2",
    ]
    if preflight_only:
        cmd.append("--preflight-only")
    return cmd


def main() -> int:
    launches = []
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["MPLCONFIGDIR"] = str(ROOT / ".mplconfig")
    (ROOT / ".mplconfig").mkdir(parents=True, exist_ok=True)
    for run in RUNS.values():
        out = Path(run["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        pre = subprocess.run(command(run, preflight_only=True), cwd=PROJECT_ROOT, text=True, capture_output=True, env=env)
        (out / "preflight_stdout.log").write_text(pre.stdout)
        (out / "preflight_stderr.log").write_text(pre.stderr)
        if pre.returncode != 0:
            raise RuntimeError(f"preflight failed for {run['candidate']}: {pre.stderr[-2000:]}")
    for run in RUNS.values():
        log_path = Path(run["output_dir"]) / "training.log"
        fh = log_path.open("wb", buffering=0)
        proc = subprocess.Popen(command(run), cwd=PROJECT_ROOT, stdout=fh, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        (Path(run["output_dir"]) / "pid.txt").write_text(f"{proc.pid}\n")
        launches.append({
            "candidate": run["candidate"],
            "pid": proc.pid,
            "command": command(run),
            "log_path": str(log_path),
            "output_dir": str(run["output_dir"]),
            "checkpoint_path": {
                "edge_x": str(run["edge_x"]),
                "edge_y": str(run["edge_y"]),
                "body": str(run["body"]),
            },
        })
        notes = [
            f"# Launch Notes: {run['candidate']}",
            "",
            f"- timestamp: `{time.strftime('%Y-%m-%d %H:%M:%S %Z')}`",
            f"- PID: `{proc.pid}`",
            f"- command: `{' '.join(command(run))}`",
            f"- log path: `{log_path}`",
            f"- output dir: `{run['output_dir']}`",
            f"- edge_x checkpoint: `{run['edge_x']}`",
            f"- edge_y checkpoint: `{run['edge_y']}`",
            f"- body checkpoint: `{run['body']}`",
            "- chain parameters: `n_chains=8, n_sweeps=2000, record_every=20, P_coarse=12, coarse_patches_per_sweep=4, coarse_passes=5, epsilon_c=0.6, P_detail=12, beta_z=0.4, n_detail_updates_per_sweep=2`",
        ]
        (Path(run["output_dir"]) / "LAUNCH_NOTES.md").write_text("\n".join(notes) + "\n")
    print(json.dumps({"launches": launches}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
