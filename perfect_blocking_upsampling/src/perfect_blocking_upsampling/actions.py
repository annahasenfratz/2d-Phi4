from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from .io import ActionSpec


def action_density(phi: np.ndarray, action: ActionSpec) -> np.ndarray:
    arr = np.asarray(phi, dtype=np.float64)
    phi2 = arr**2
    phi4 = phi2**2
    nn = arr * np.roll(arr, -1, axis=-1) + arr * np.roll(arr, -1, axis=-2)
    diag = arr * np.roll(np.roll(arr, -1, axis=-1), -1, axis=-2)
    if action.type == "phi4_nn":
        return (1.0 - 2.0 * action.lambda_) * phi2 + action.lambda_ * phi4 - 2.0 * action.kappa * nn
    if action.type == "phi4_nn_plus_diag":
        return (1.0 - 2.0 * action.lambda_) * phi2 + action.lambda_ * phi4 - 2.0 * action.kappa * nn - 2.0 * action.kappa_diag * diag
    raise ValueError(f"unknown action type: {action.type}")


def action_total(phi: np.ndarray, action: ActionSpec) -> np.ndarray:
    arr = np.asarray(phi, dtype=np.float64)
    density = action_density(arr, action)
    if density.ndim == 3:
        return density.sum(axis=(1, 2))
    if density.ndim == 2:
        return density.sum()
    raise ValueError(f"unexpected density rank: {density.shape}")


def validate_action_blocks(cfg: dict[str, Any]) -> tuple[ActionSpec, ActionSpec]:
    act = cfg["action"]
    coarse = act["coarse"]
    fine = act["fine"]
    return (
        ActionSpec(type=str(coarse["type"]), lambda_=float(coarse["lambda"]), kappa=float(coarse["kappa"]), kappa_diag=float(coarse.get("kappa_diag", 0.0))),
        ActionSpec(type=str(fine["type"]), lambda_=float(fine["lambda"]), kappa=float(fine["kappa"]), kappa_diag=float(fine.get("kappa_diag", 0.0))),
    )


def action_blocks_to_json(action: tuple[ActionSpec, ActionSpec]) -> dict[str, Any]:
    coarse, fine = action
    return {"coarse": asdict(coarse), "fine": asdict(fine)}
