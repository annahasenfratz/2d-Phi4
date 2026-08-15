#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from _common import load_actions, load_config, load_ensembles, load_frozen_models, load_kernel_spec, resolve_run_paths, verify_frozen_hashes
from perfect_blocking_upsampling.ar import independence_ar
from perfect_blocking_upsampling.evaluate import evaluate_proposal_batch
from perfect_blocking_upsampling.sampling import generate_proposals, proposal_observables
from perfect_blocking_upsampling.observables import observables
from perfect_blocking_upsampling.actions import action_total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    coarse_action, fine_action = load_actions(cfg)
    coarse, fine, coarse_manifest, fine_manifest, paths = load_ensembles(cfg)
    kernel_spec, kernel_json = load_kernel_spec(cfg)
    hashes = verify_frozen_hashes(paths["frozen_dir"])
    if not all(r["matches"] for r in hashes):
        raise RuntimeError("frozen hash verification failed")
    refine_model, refine_state, stage_bundles, _, _, refine_ckpt = load_frozen_models(cfg)
    eval_n = int(cfg["evaluation"]["n_proposals"])
    ar_chains = int(cfg["evaluation"]["ar_chains"])
    ar_n = int(cfg["evaluation"]["ar_proposals_per_chain"])
    seed = int(cfg["random_seed"])
    rng = np.random.default_rng(seed)
    coarse_batch = coarse[rng.integers(0, len(coarse), size=eval_n)]
    proposals = generate_proposals(coarse_batch, refine_model, refine_state, stage_bundles, kernel_spec, fine_action, coarse_action, seed + 17)
    proposal_obs = proposal_observables(proposals, fine_action)
    target_obs = observables(fine[: min(64, len(fine))], fine_action)
    summary = {
        "config": str(args.config),
        "frozen_dir": str(paths["frozen_dir"]),
        "proposal_std_logw": proposal_obs["std(logw)"],
        "proposal_ess_over_n": proposal_obs["ESS/N"],
        "proposal_phi4": proposal_obs["phi4"],
        "proposal_NN": proposal_obs["NN"],
        "proposal_action_density": proposal_obs["action_density"],
        "proposal_Binder_U4": proposal_obs["Binder_U4"],
        "proposal_xi_over_L": proposal_obs["xi_over_L"],
        "target_phi4": target_obs["phi4"],
        "target_NN": target_obs["NN"],
        "target_action_density": target_obs["action_density"],
        "target_Binder_U4": target_obs["Binder_U4"],
        "target_xi_over_L": target_obs["xi_over_L"],
        "ar_chains": ar_chains,
        "ar_proposals_per_chain": ar_n,
        "hashes_ok": True,
    }
    out = resolve_run_paths(cfg)["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "proposal_summary.json").write_text(json.dumps(proposal_obs, indent=2, default=float) + "\n")
    # Short A/R diagnostic, using the same proposal batch for a compact regression check.
    ar_result = independence_ar(proposals, np.random.default_rng(seed + 99), fine_action)
    (out / "ar_summary.json").write_text(json.dumps({
        "acceptance_rate": ar_result["acceptance_rate"],
        "max_rejection_streak": ar_result["max_rejection_streak"],
        "ess_over_n": ar_result["ess_over_n"],
        "phi4": ar_result["phi4"],
        "NN": ar_result["NN"],
        "action_density": ar_result["action_density"],
        "Binder_U4": ar_result["Binder_U4"],
        "xi_over_L": ar_result["xi_over_L"],
    }, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

