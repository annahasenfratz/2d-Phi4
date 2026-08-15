from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .io import ActionSpec
from .actions import action_total


def load_stage_model(
    stage: str,
    checkpoint_path: str | Path,
    cond_channels: int,
    coeff_path: str | Path | None = None,
    lattice_size: int | None = None,
):
    import torch
    from ML_sampling_clean.experiments.decimated_conditional_fillin.run_lam0p022_k0265_N5000_interaction_uv_rank import stage_model

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    flow_arch = ckpt.get("config", {}).get("flow_arch")
    if flow_arch in {"gathered_edge", "gathered_local"}:
        if ckpt.get("config", {}).get("flow_arch") == "gathered_edge" and stage != "edge":
            raise ValueError("gathered_edge checkpoints are only valid for the edge stage")
        from .gathered_edge import build_gathered_edge_from_checkpoint

        model_lattice_size = int(lattice_size or ckpt["config"].get("lattice_size", ckpt["config"].get("coarse_L", 8)))
        model = build_gathered_edge_from_checkpoint(ckpt, cond_channels, model_lattice_size)
    elif flow_arch == "procedural_conv":
        from .conv_pair import build_procedural_conv_from_checkpoint

        model_lattice_size = int(lattice_size or ckpt["config"].get("lattice_size", ckpt["config"].get("coarse_L", 8)))
        model = build_procedural_conv_from_checkpoint(ckpt, cond_channels, model_lattice_size)
    else:
        model = stage_model(cond_channels, 1, ckpt)
    lg_path = Path(coeff_path) if coeff_path is not None else Path(checkpoint_path).with_name("local_gaussian_coefficients.npz")
    if not lg_path.exists():
        raise FileNotFoundError(lg_path)
    lg_npz = np.load(lg_path)
    lg = {"coeffs": lg_npz["coeffs"], "sigma": lg_npz["sigma"], "ridge": lg_npz["ridge"]}
    return model, lg, ckpt["model_state"], ckpt


def stage_sample(model, state: dict[str, Any], cond: np.ndarray, lg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    import torch
    from ML_sampling_clean.experiments.decimated_conditional_fillin.run_staged_decimated_conditional_fillin import sample_stage

    return sample_stage(model, state, cond, lg)


def sample_missing_fields(stages: dict[str, tuple[Any, dict[str, Any], dict[str, Any]]], coarse: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    import torch

    torch.manual_seed(seed)
    edge_bundle = stages["edge"]
    pair_bundle = stages["pair"]
    corner_bundle = stages["corner"]
    edge_model, edge_lg, edge_state = edge_bundle[:3]
    pair_model, pair_lg, pair_state = pair_bundle[:3]
    corner_model, corner_lg, corner_state = corner_bundle[:3]
    d10, l10 = stage_sample(edge_model, edge_state, coarse[:, None], edge_lg)
    d01, l01 = stage_sample(pair_model, pair_state, np.concatenate([coarse[:, None], d10], axis=1), pair_lg)
    d11, l11 = stage_sample(corner_model, corner_state, np.concatenate([coarse[:, None], d10, d01], axis=1), corner_lg)
    return np.concatenate([d10, d01, d11], axis=1).astype(np.float32), (l10 + l01 + l11).astype(np.float64)
