#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import numpy as np

from _common import load_actions, load_config, load_ensembles, load_frozen_models, load_kernel_spec, resolve_run_paths
from perfect_blocking_upsampling.ar import independence_ar
from perfect_blocking_upsampling.sampling import generate_proposals


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    coarse_action, fine_action = load_actions(cfg)
    coarse, _, _, _, _ = load_ensembles(cfg)
    kernel_spec, _ = load_kernel_spec(cfg)
    refine_model, refine_state, stage_bundles, _, _, _ = load_frozen_models(cfg)
    n = int(cfg["evaluation"]["ar_proposals_per_chain"])
    chains = int(cfg["evaluation"]["ar_chains"])
    out = resolve_run_paths(cfg)["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    chain_summaries = []
    seed0 = int(cfg["random_seed"])
    for c in range(chains):
        rng = np.random.default_rng(seed0 + 1009 * c)
        coarse_batch = coarse[rng.integers(0, len(coarse), size=n)]
        proposals = generate_proposals(coarse_batch, refine_model, refine_state, stage_bundles, kernel_spec, fine_action, coarse_action, seed0 + 17 + c)
        ar = independence_ar(proposals, np.random.default_rng(seed0 + 999 + c), fine_action)
        chain_summaries.append({
            "chain_id": c,
            "acceptance_rate": ar["acceptance_rate"],
            "max_rejection_streak": ar["max_rejection_streak"],
            "ess_over_n": ar["ess_over_n"],
            "phi4": ar["phi4"],
            "NN": ar["NN"],
            "action_density": ar["action_density"],
            "Binder_U4": ar["Binder_U4"],
            "xi_over_L": ar["xi_over_L"],
        })
    (out / "ar_summary.json").write_text(json.dumps(chain_summaries, indent=2) + "\n")
    print(json.dumps(chain_summaries, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

