from __future__ import annotations

from typing import Any

import numpy as np

from .sampling import ProposalResult
from .observables import observables


def rejection_streaks(accepted: np.ndarray) -> list[int]:
    streaks: list[int] = []
    cur = 0
    for flag in accepted[1:]:
        if flag:
            if cur:
                streaks.append(cur)
            cur = 0
        else:
            cur += 1
    if cur:
        streaks.append(cur)
    return streaks or [0]


def stable_ess(logw: np.ndarray) -> float:
    w = np.exp(logw - np.max(logw))
    s1 = float(np.sum(w))
    s2 = float(np.sum(w * w))
    return s1 * s1 / max(s2, 1.0e-300)


def independence_ar(proposals: ProposalResult, rng: np.random.Generator, fine_action) -> dict[str, Any]:
    n = proposals.logw.shape[0]
    accepted = np.zeros(n, dtype=bool)
    chain = np.empty_like(proposals.phi)
    current = proposals.phi[0].copy()
    current_logw = float(proposals.logw[0])
    accepted[0] = True
    chain[0] = current
    for i in range(1, n):
        if np.log(rng.random()) < min(0.0, float(proposals.logw[i] - current_logw)):
            current = proposals.phi[i].copy()
            current_logw = float(proposals.logw[i])
            accepted[i] = True
        chain[i] = current
    return {
        "accepted": accepted,
        "chain": chain,
        "acceptance_rate": float(np.mean(accepted[1:])),
        "max_rejection_streak": int(max(rejection_streaks(accepted))),
        "ess_over_n": float(stable_ess(proposals.logw) / n),
        **observables(chain, fine_action),
    }
