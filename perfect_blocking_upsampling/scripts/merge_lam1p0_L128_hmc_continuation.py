#!/usr/bin/env python3
"""Append a separately written L128 fine-HMC continuation to its parent run.

The continuation labels its initial field as sweep 100.  That field is the
parent's saved sweep-100 field, so it is retained once, from the parent; only
continuation measurements at sweeps 101--200 are appended.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


OBS_FILES = (
    "main_per_sweep_measurements.csv",
    "per_sweep_observables.csv",
    "Gk_per_sweep_measurements.csv",
    "ensemble_average_history.csv",
)


def backup(path: Path) -> None:
    target = path.with_name(path.stem + ".before_sweep200" + path.suffix)
    if not target.exists():
        shutil.copy2(path, target)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", type=Path, required=True, help="original L64toL128 level directory")
    ap.add_argument("--continuation", type=Path, required=True, help="separate continuation run directory")
    args = ap.parse_args()
    parent = args.parent.resolve()
    cont = args.continuation.resolve()
    pobs, cobs = parent / "observables", cont / "observables"

    # A prior interrupted merge may have written an initial subset of files.
    # Restore its automatic backups before attempting one coherent merge.
    for name in (*OBS_FILES, "acceptance_history.csv"):
        dst = pobs / name
        saved = dst.with_name(dst.stem + ".before_sweep200" + dst.suffix)
        if saved.exists():
            shutil.copy2(saved, dst)

    for name in OBS_FILES:
        dst, src = pobs / name, cobs / name
        if not dst.exists() or not src.exists():
            raise FileNotFoundError(f"missing {dst if not dst.exists() else src}")
        old, extra = pd.read_csv(dst), pd.read_csv(src)
        if list(old.columns) != list(extra.columns):
            raise ValueError(f"column mismatch in {name}")
        if int(old.sweep.max()) != 100 or int(extra.sweep.min()) != 100:
            raise ValueError(f"unexpected sweep ranges in {name}")
        merged = pd.concat([old, extra.loc[extra.sweep > 100]], ignore_index=True)
        if "chain_id" in merged and merged.duplicated(["chain_id", "sweep"]).any():
            raise ValueError(f"duplicate chain/sweep rows after merge in {name}")
        backup(dst)
        merged.to_csv(dst, index=False)

    # Continuation acceptance is locally numbered 1..100.  Shift it to
    # absolute 101..200 and recalculate its cumulative rate from the parent.
    pdst, csrc = pobs / "acceptance_history.csv", cobs / "acceptance_history.csv"
    old, extra = pd.read_csv(pdst), pd.read_csv(csrc)
    if int(old.sweep.max()) != 100 or list(old.columns) != list(extra.columns):
        raise ValueError("unexpected acceptance-history layout")
    extra = extra.copy()
    extra["sweep"] += 100
    acc0, att0 = int(old.accepted.sum()), int(old.attempted.sum())
    accepted = acc0 + extra["accepted"].cumsum()
    attempted = att0 + extra["attempted"].cumsum()
    extra["acceptance_cumulative"] = accepted / attempted
    backup(pdst)
    pd.concat([old, extra], ignore_index=True).to_csv(pdst, index=False)

    # Promote the final L128 field while keeping the sweep-100 checkpoint.
    final_source = cont / "checkpoints" / "checkpoint_latest.npz"
    if not final_source.exists():
        raise FileNotFoundError(f"missing continuation final checkpoint: {final_source}")
    for dst in (parent / "final_phi.npz", parent.parents[1] / "final_phi.npz"):
        backup(dst)
        shutil.copy2(final_source, dst)
    shutil.copy2(final_source, parent / "checkpoints" / "checkpoint_latest.npz")
    shutil.copy2(final_source, parent / "checkpoints" / "checkpoint_sweep_0200.npz")

    config_path = parent / "run_config.json"
    config = json.loads(config_path.read_text())
    config["thermalization_sweeps"] = 200
    config["merged_continuation"] = str(cont)
    config["merged_sweep_range"] = [101, 200]
    text = json.dumps(config, indent=2, default=str) + "\n"
    for name in ("run_config.json", "run_config.yaml", "config.yaml"):
        (parent / name).write_text(text)
    print(f"merged sweeps 101--200 into {parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
