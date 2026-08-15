#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from _common import load_config, load_ensembles, load_frozen_models, load_kernel_spec, resolve_run_paths, verify_frozen_hashes
from perfect_blocking_upsampling.actions import action_total
from perfect_blocking_upsampling.checks import validate_ensemble_manifest
from perfect_blocking_upsampling.kernels import inverse_kernel, apply_kernel
from perfect_blocking_upsampling.observables import observables
from perfect_blocking_upsampling.coarse_refine import apply_refine
from perfect_blocking_upsampling.staged_flow import sample_missing_fields
from perfect_blocking_upsampling.actions import validate_action_blocks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    coarse_action, fine_action = validate_action_blocks(cfg)
    coarse, fine, coarse_manifest, fine_manifest, paths = load_ensembles(cfg)
    kernel_spec, kernel_json = load_kernel_spec(cfg)
    hash_rows = verify_frozen_hashes(paths["frozen_dir"])
    if not all(r["matches"] for r in hash_rows):
        raise RuntimeError("frozen checkpoint hash check failed")
    coarse_errors = validate_ensemble_manifest(coarse_manifest, coarse_action, int(cfg["lattice"]["coarse_L"]))
    fine_errors = validate_ensemble_manifest(fine_manifest, fine_action, int(cfg["lattice"]["fine_L"]))
    print(json.dumps({
        "coarse_errors": coarse_errors,
        "fine_errors": fine_errors,
        "kernel_name": kernel_spec.name,
        "kernel_type": kernel_spec.type,
        "kernel_eta": kernel_spec.eta,
        "frozen_hashes_ok": True,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

