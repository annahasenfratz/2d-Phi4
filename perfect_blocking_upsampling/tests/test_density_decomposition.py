from __future__ import annotations

import json
import numpy as np

from perfect_blocking_upsampling.actions import validate_action_blocks
from perfect_blocking_upsampling.kernels import load_kernel
from perfect_blocking_upsampling.sampling import generate_proposals
from perfect_blocking_upsampling.observables import observables
from perfect_blocking_upsampling.checks import verify_sha256_manifest
from perfect_blocking_upsampling.io import load_yaml
from perfect_blocking_upsampling.coarse_refine import build_refine_model_from_checkpoint
from perfect_blocking_upsampling.staged_flow import load_stage_model


def test_density_decomposition_is_finite_on_small_batch():
    cfg = load_yaml("configs/default_lam0p022_k02705.yaml")
    coarse_action, fine_action = validate_action_blocks(cfg)
    coarse = np.load("data/coarse_L8/configs.npz")["phi"].astype(np.float32)[:8]
    kernel_spec, _ = load_kernel(cfg["kernel"]["path"])
    frozen = cfg["checkpoints"]["frozen_dir"]
    hashes = verify_sha256_manifest(f"{frozen}/sha256_checksums.txt", root=frozen)
    assert all(r["matches"] for r in hashes)
    import torch
    ref_ckpt = torch.load(f"{frozen}/coarse_refine.pt", map_location="cpu")
    refine_model, _ = build_refine_model_from_checkpoint(ref_ckpt, coarse_action, int(cfg["lattice"]["coarse_L"]))
    stages = {}
    for stage, cond_channels in [("edge", 1), ("pair", 2), ("corner", 3)]:
        stage_model, lg, state, ckpt = load_stage_model(stage, f"{frozen}/{stage}.pt", cond_channels, f"{frozen}/{stage}/local_gaussian_coefficients.npz")
        stages[stage] = (stage_model, lg, state)
    result = generate_proposals(coarse, refine_model, ref_ckpt["model_state"], stages, kernel_spec, fine_action, coarse_action, int(cfg["random_seed"]))
    assert np.isfinite(result.logw).all()
    assert np.isfinite(result.s_f).all()
    assert np.isfinite(result.s_c).all()
    assert np.isfinite(result.logdet_refine).all()
    assert np.isfinite(result.logq_missing).all()

