#!/usr/bin/env python3
"""Detached launcher for the first L16->L32 footprint scan candidates."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON = PROJECT_ROOT.parent / ".venv/bin/python"
SCRIPT = PROJECT_ROOT / "perfect_blocking_upsampling/scripts/train_l16to32_footprint_candidate.py"
OUT_ROOT = PROJECT_ROOT / "perfect_blocking_upsampling/outputs/lam0p022_L16to32_flow_footprint_scan"

CANDIDATES = {
    "fp_small_ref": {"footprint": 8, "seed": 2026070401},
    "fp_medium_1": {"footprint": 11, "seed": 2026070402},
    "fp_medium_2": {"footprint": 13, "seed": 2026070403},
    "fp_large_safe": {"footprint": 15, "seed": 2026070404},
    "fp_medium_1_deep": {"footprint": 11, "seed": 2026070405},
    "fp_large_safe_deep": {"footprint": 15, "seed": 2026070406},
}


def run_preflight(name: str, cfg: dict[str, int], out: Path, common: argparse.Namespace) -> dict:
    cmd = [
        str(PYTHON),
        "-u",
        "-B",
        str(SCRIPT),
        "--candidate",
        name,
        "--footprint",
        str(cfg["footprint"]),
        "--output-dir",
        str(out),
        "--epochs",
        str(common.epochs),
        "--max-train",
        str(common.max_train),
        "--max-val",
        str(common.max_val),
        "--n-proposals",
        str(common.n_proposals),
        "--batch-size-sites",
        str(common.batch_size_sites),
        "--hidden-channels",
        str(common.hidden_channels),
        "--conditioner-layers",
        str(common.conditioner_layers),
        "--seed",
        str(cfg["seed"]),
        "--preflight-only",
    ]
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, text=True, capture_output=True)
    (out / "preflight_stdout.log").write_text(proc.stdout)
    (out / "preflight_stderr.log").write_text(proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"preflight failed for {name}: {proc.stderr[-2000:]}")
    report = json.loads((out / "footprint_report.json").read_text())
    lift = json.loads((out / "liftability_preflight.json").read_text())
    if not report["non_wrapping"]:
        raise RuntimeError(f"refusing to launch wrapping candidate {name}")
    if not lift["L32_to_L64"]["passed"]:
        raise RuntimeError(f"refusing to launch non-liftable candidate {name}")
    split = {
        "available_reference_samples": 1000,
        "requested_max_train": common.max_train,
        "requested_max_val": common.max_val,
        "total_requested": common.max_train + common.max_val,
        "train_val_split_valid": common.max_train + common.max_val <= 1000,
    }
    for line in proc.stdout.splitlines():
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "available_reference_samples" in obj:
            split = obj
    if not split["train_val_split_valid"]:
        raise RuntimeError(f"refusing to launch invalid split for {name}: {split}")
    offset_count = int(report["offset_count"])
    edge_in = 2 * offset_count
    body_in = 4 * offset_count
    hidden = int(common.hidden_channels)
    layers = int(common.conditioner_layers)

    def mlp_params(in_dim: int) -> int:
        return in_dim * hidden + hidden + (layers - 1) * (hidden * hidden + hidden) + hidden * 2 + 2

    estimated_parameter_count = 2 * mlp_params(edge_in) + mlp_params(body_in)
    return {"command": cmd, "footprint_report": report, "liftability": lift, "split": split, "estimated_parameter_count": estimated_parameter_count}


def archive_previous_failed_log(out: Path) -> str | None:
    log = out / "training.log"
    if not log.exists():
        return None
    archive_dir = out / "failed_initial_split"
    archive_dir.mkdir(parents=True, exist_ok=True)
    dest = archive_dir / "training_failed_split_1280.log"
    if dest.exists():
        stamp = time.strftime("%Y%m%d_%H%M%S")
        dest = archive_dir / f"training_failed_split_1280_{stamp}.log"
    shutil.move(str(log), str(dest))
    return str(dest)


def launch(name: str, cfg: dict[str, int], out: Path, common: argparse.Namespace) -> dict:
    log = out / "training.log"
    archived = archive_previous_failed_log(out)
    cmd = [
        str(PYTHON),
        "-u",
        "-B",
        str(SCRIPT),
        "--candidate",
        name,
        "--footprint",
        str(cfg["footprint"]),
        "--output-dir",
        str(out),
        "--epochs",
        str(common.epochs),
        "--max-train",
        str(common.max_train),
        "--max-val",
        str(common.max_val),
        "--n-proposals",
        str(common.n_proposals),
        "--batch-size-sites",
        str(common.batch_size_sites),
        "--hidden-channels",
        str(common.hidden_channels),
        "--conditioner-layers",
        str(common.conditioner_layers),
        "--seed",
        str(cfg["seed"]),
    ]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    fh = log.open("wb", buffering=0)
    proc = subprocess.Popen(cmd, cwd=PROJECT_ROOT, stdout=fh, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    (out / "pid.txt").write_text(f"{proc.pid}\n")
    return {"candidate": name, "pid": proc.pid, "command": cmd, "log_path": str(log), "output_dir": str(out), "archived_previous_log": archived}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="fp_small_ref,fp_medium_1")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--max-train", type=int, default=1024)
    ap.add_argument("--max-val", type=int, default=256)
    ap.add_argument("--n-proposals", type=int, default=512)
    ap.add_argument("--batch-size-sites", type=int, default=8192)
    ap.add_argument("--hidden-channels", type=int, default=64)
    ap.add_argument("--conditioner-layers", type=int, default=3)
    args = ap.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    selected = [x.strip() for x in args.candidates.split(",") if x.strip()]
    preflights = {}
    launches = []
    for name in selected:
        if name not in CANDIDATES:
            raise SystemExit(f"unknown candidate {name}")
        out = OUT_ROOT / name
        out.mkdir(parents=True, exist_ok=True)
        preflights[name] = run_preflight(name, CANDIDATES[name], out, args)
    for name in selected:
        launches.append(launch(name, CANDIDATES[name], OUT_ROOT / name, args))
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    notes = ["# Launch Notes", "", f"- timestamp: `{timestamp}`", f"- launcher: `{Path(__file__).resolve()}`", f"- python: `{PYTHON}`", "- changed-file summary: added L16->L32 finite-footprint trainer and detached launcher; existing dirty worktree left untouched.", ""]
    for item in launches:
        report = preflights[item["candidate"]]["footprint_report"]
        notes += [
            f"## {item['candidate']}",
            "",
            f"- PID: `{item['pid']}`",
            f"- available reference samples: `{preflights[item['candidate']]['split']['available_reference_samples']}`",
            f"- requested max_train: `{preflights[item['candidate']]['split']['requested_max_train']}`",
            f"- requested max_val: `{preflights[item['candidate']]['split']['requested_max_val']}`",
            f"- total requested: `{preflights[item['candidate']]['split']['total_requested']}`",
            f"- train/val split valid: `yes`",
            f"- max radius: `{report['max_radius_fine_lattice_sites']}`",
            f"- footprint: `{report['footprint_size']}`",
            f"- non-wrapping on L_f=32: `{report['non_wrapping']}`",
            f"- estimated parameter count: `{preflights[item['candidate']]['estimated_parameter_count']}`",
            f"- log path: `{item['log_path']}`",
            f"- output dir: `{item['output_dir']}`",
            f"- archived previous log: `{item['archived_previous_log']}`",
            f"- command: `{' '.join(item['command'])}`",
            "",
        ]
    (OUT_ROOT / "LAUNCH_NOTES.md").write_text("\n".join(notes))
    print(json.dumps({"launches": launches, "preflights": {k: v["footprint_report"] for k, v in preflights.items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
