from __future__ import annotations

import numpy as np

from perfect_blocking_upsampling.actions import validate_action_blocks
from perfect_blocking_upsampling.io import load_yaml
from perfect_blocking_upsampling.kernels import load_kernel
from perfect_blocking_upsampling.sampling import generate_proposals, proposal_observables
from perfect_blocking_upsampling.coarse_refine import build_refine_model_from_checkpoint
from perfect_blocking_upsampling.staged_flow import load_stage_model


def test_small_frozen_reproduction_batch_is_finite():
    cfg = load_yaml("configs/default_lam0p022_k02705.yaml")
    coarse_action, fine_action = validate_action_blocks(cfg)
    coarse = np.load("data/coarse_L8/configs.npz")["phi"].astype(np.float32)
    kernel_spec, _ = load_kernel(cfg["kernel"]["path"])
    import torch
    frozen = "checkpoints/frozen/lam0p022_kappa0p2705_small3_refine"
    ref_ckpt = torch.load(f"{frozen}/coarse_refine.pt", map_location="cpu")
    refine_model, _ = build_refine_model_from_checkpoint(ref_ckpt, coarse_action, int(cfg["lattice"]["coarse_L"]))
    stages = {}
    for stage, cond_channels in [("edge", 1), ("pair", 2), ("corner", 3)]:
        stage_model, lg, state, ckpt = load_stage_model(stage, f"{frozen}/{stage}.pt", cond_channels, f"{frozen}/{stage}/local_gaussian_coefficients.npz")
        stages[stage] = (stage_model, lg, state)
    proposals = generate_proposals(coarse[:32], refine_model, ref_ckpt["model_state"], stages, kernel_spec, fine_action, coarse_action, int(cfg["random_seed"]))
    summary = proposal_observables(proposals, fine_action)
    assert np.isfinite(summary["std(logw)"])
    assert np.isfinite(summary["ESS/N"])
    assert np.isfinite(summary["phi4"])
    assert np.isfinite(summary["Binder_U4"])

