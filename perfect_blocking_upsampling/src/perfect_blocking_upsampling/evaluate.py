from __future__ import annotations

from typing import Any

import numpy as np

from .sampling import generate_proposals, proposal_observables


def evaluate_proposal_batch(
    coarse_u: np.ndarray,
    refine_model: Any,
    refine_state: dict[str, Any],
    stages: dict[str, tuple[Any, dict[str, Any], dict[str, Any]]],
    kernel,
    fine_action,
    coarse_action,
    seed: int,
) -> dict[str, Any]:
    result = generate_proposals(coarse_u, refine_model, refine_state, stages, kernel, fine_action, coarse_action, seed)
    return proposal_observables(result, fine_action)

