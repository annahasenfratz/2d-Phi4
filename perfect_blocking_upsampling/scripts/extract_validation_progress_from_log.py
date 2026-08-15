#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


INIT_RE = re.compile(
    r"chain (?P<chain>\d+) init: target=(?P<target>\S+) coarse_m=(?P<coarse_m>[-+0-9.eE]+) phi_m=(?P<phi_m>[-+0-9.eE]+)"
)
PROGRESS_RE = re.compile(
    r"chain (?P<chain>\d+) sweep (?P<sweep>\d+)/(?P<total>\d+): "
    r"coarse_attempts=(?P<coarse>\d+) latent_attempts=(?P<latent>\d+) m=(?P<m>[-+0-9.eE]+)"
)
SUMMARY_RE = re.compile(
    r"chain (?P<chain>\d+): coarse_acc=(?P<coarse_acc>[-+0-9.eE]+) "
    r"coarse_std=(?P<coarse_std>[-+0-9.eE]+) latent_acc=(?P<latent_acc>[-+0-9.eE]+) "
    r"latent_std=(?P<latent_std>[-+0-9.eE]+)"
)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    args = ap.parse_args()
    log = args.run_dir / "run.log"
    if not log.exists():
        raise FileNotFoundError(log)

    init_rows = []
    progress_rows = []
    summary_rows = []
    for line in log.read_text(errors="replace").splitlines():
        m = INIT_RE.search(line)
        if m:
            init_rows.append(
                {
                    "chain_id": int(m.group("chain")),
                    "target_initial_sector": m.group("target"),
                    "initial_coarse_m": float(m.group("coarse_m")),
                    "initial_phi_m": float(m.group("phi_m")),
                }
            )
            continue
        m = PROGRESS_RE.search(line)
        if m:
            progress_rows.append(
                {
                    "chain_id": int(m.group("chain")),
                    "completed_sweeps": int(m.group("sweep")),
                    "total_sweeps": int(m.group("total")),
                    "coarse_attempts_so_far": int(m.group("coarse")),
                    "latent_attempts_so_far": int(m.group("latent")),
                    "m": float(m.group("m")),
                }
            )
            continue
        m = SUMMARY_RE.search(line)
        if m:
            summary_rows.append(
                {
                    "chain_id": int(m.group("chain")),
                    "coarse_acceptance": float(m.group("coarse_acc")),
                    "coarse_std_delta_logw": float(m.group("coarse_std")),
                    "latent_acceptance": float(m.group("latent_acc")),
                    "latent_std_delta_logw": float(m.group("latent_std")),
                }
            )

    write_csv(args.run_dir / "progress_history_from_log.csv", progress_rows)
    write_csv(args.run_dir / "initial_chain_states_from_log.csv", init_rows)
    write_csv(args.run_dir / "chain_summaries_from_log.csv", summary_rows)
    payload = {
        "run_log": str(log),
        "initial_rows": len(init_rows),
        "progress_rows": len(progress_rows),
        "completed_chain_summaries": len(summary_rows),
        "latest_progress": progress_rows[-1] if progress_rows else None,
    }
    (args.run_dir / "progress_from_log_summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
