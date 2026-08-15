#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import numpy as np

from _common import load_actions, load_config, load_ensembles, load_frozen_models, load_kernel_spec, resolve_run_paths
from perfect_blocking_upsampling.sampling import generate_proposals, proposal_observables


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    coarse_action, fine_action = load_actions(cfg)
    coarse, fine, _, _, paths = load_ensembles(cfg)
    kernel_spec, _ = load_kernel_spec(cfg)
    refine_model, refine_state, stage_bundles, _, _, _ = load_frozen_models(cfg)
    rng = np.random.default_rng(int(cfg["random_seed"]))
    n = int(cfg["evaluation"]["n_proposals"])
    coarse_batch = coarse[rng.integers(0, len(coarse), size=n)]
    proposals = generate_proposals(coarse_batch, refine_model, refine_state, stage_bundles, kernel_spec, fine_action, coarse_action, int(cfg["random_seed"]) + 17)
    out = resolve_run_paths(cfg)["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    summary = proposal_observables(proposals, fine_action)
    (out / "proposal_only_summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

