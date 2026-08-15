from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .io import ActionSpec


def adapt_refine_state_for_model(model, state: dict[str, Any]) -> dict[str, Any]:
    """Drop regenerated lattice-size buffers from an otherwise portable checkpoint."""
    model_state = model.state_dict()
    adapted = {}
    for key, value in state.items():
        target = model_state.get(key)
        if target is not None and hasattr(value, "shape") and tuple(value.shape) != tuple(target.shape):
            if "mask" in key:
                continue
        adapted[key] = value
    return adapted


def build_refine_model_from_checkpoint(checkpoint: dict[str, Any], action_coarse: ActionSpec, coarse_L: int):
    from ML_sampling_clean.experiments.decimated_conditional_fillin.run_decimated_conditional_fillin import build_model, Config

    cfg = Config(
        lambda_=action_coarse.lambda_,
        kappa=action_coarse.kappa,
        seed=int(checkpoint.get("config", {}).get("seed", 0)),
        n_coupling_layers=int(checkpoint["config"]["n_coupling_layers"]),
        conv_hidden_channels=int(checkpoint["config"]["conv_hidden_channels"]),
        log_scale_bound=float(checkpoint["config"]["log_scale_bound"]),
        flow_arch=str(checkpoint["config"].get("flow_arch", "conv")),
    )
    model = build_model(coarse_L * coarse_L, coarse_L * coarse_L, (1, coarse_L, coarse_L), (1, coarse_L, coarse_L), cfg)
    model.load_state_dict(adapt_refine_state_for_model(model, checkpoint["model_state"]), strict=False)
    model.eval()
    return model, cfg


def apply_refine(model, state: dict[str, Any], u: np.ndarray, batch_size: int = 32) -> tuple[np.ndarray, np.ndarray]:
    import torch

    l = u.shape[1]
    model.load_state_dict(adapt_refine_state_for_model(model, state), strict=False)
    model.eval()
    outs = []
    logdets = []
    with torch.no_grad():
        for start in range(0, u.shape[0], batch_size):
            ub_np = u[start : start + batch_size]
            ub = torch.tensor(ub_np[:, None].reshape(ub_np.shape[0], -1), dtype=torch.float32)
            cond = torch.zeros((ub.shape[0], ub.shape[1]), dtype=torch.float32)
            x, logdet = model.forward(ub, cond)
            outs.append(x.cpu().numpy().reshape(ub.shape[0], l, l))
            logdets.append(logdet.cpu().numpy())
    return np.concatenate(outs, axis=0).astype(np.float32), np.concatenate(logdets, axis=0).astype(np.float64)
