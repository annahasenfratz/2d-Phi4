from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .actions import action_total
from .coarse_refine import apply_refine
from .kernels import inverse_kernel, KernelSpec
from .observables import observables
from .staged_flow import sample_missing_fields


@dataclass(frozen=True)
class ProposalResult:
    u: np.ndarray
    psi00_used: np.ndarray
    d: np.ndarray
    phi: np.ndarray
    s_f: np.ndarray
    s_c: np.ndarray
    logdet_refine: np.ndarray
    logq_missing: np.ndarray
    logw: np.ndarray
    inverse_stats: dict[str, float]


def generate_proposals(
    coarse_u: np.ndarray,
    refine_model: Any,
    refine_state: dict[str, Any],
    stages: dict[str, tuple[Any, dict[str, Any], dict[str, Any]]],
    kernel: KernelSpec,
    fine_action,
    coarse_action,
    seed: int,
) -> ProposalResult:
    psi00_used, logdet_refine = apply_refine(refine_model, refine_state, coarse_u, batch_size=32)
    d, logq_missing = sample_missing_fields(stages, psi00_used, seed)
    psi = np.empty((coarse_u.shape[0], 2 * psi00_used.shape[1], 2 * psi00_used.shape[2]), dtype=np.float32)
    psi[:, 0::2, 0::2] = psi00_used
    psi[:, 1::2, 0::2] = d[:, 0]
    psi[:, 0::2, 1::2] = d[:, 1]
    psi[:, 1::2, 1::2] = d[:, 2]
    phi, inv_stats = inverse_kernel(psi, kernel)
    s_f = action_total(phi, fine_action)
    s_c = action_total(coarse_u, coarse_action)
    logw = -s_f + s_c + logdet_refine - logq_missing
    return ProposalResult(coarse_u, psi00_used, d, phi, s_f, s_c, logdet_refine, logq_missing, logw, inv_stats)


def proposal_observables(result: ProposalResult, fine_action) -> dict[str, Any]:
    obs = observables(result.phi, fine_action)
    w = np.exp(result.logw - np.max(result.logw))
    ess = float((w.sum() ** 2) / max(np.sum(w * w), 1.0e-300))
    return {
        **obs,
        "std(logw)": float(np.std(result.logw, ddof=1)),
        "ESS/N": float(ess / max(len(result.logw), 1)),
        "logw_mean": float(np.mean(result.logw)),
        "logw_q05": float(np.quantile(result.logw, 0.05)),
        "logw_q50": float(np.quantile(result.logw, 0.50)),
        "logw_q95": float(np.quantile(result.logw, 0.95)),
    }
