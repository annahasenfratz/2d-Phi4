from __future__ import annotations

from dataclasses import dataclass

import torch

from .actions import Phi4Action
from .conditional_flow import ConditionalPhi4Flow


@dataclass
class TrainConfig:
    L: int = 16
    batch_size: int = 64
    n_steps: int = 1000
    lr: float = 1.0e-3
    checkpoint_every: int = 100


def reverse_kl_step(
    flow: ConditionalPhi4Flow,
    action: Phi4Action,
    optimizer: torch.optim.Optimizer,
    condition_batch: torch.Tensor,
) -> dict[str, float]:
    """One inverse-KL/reverse-KL step: E_q[log q + S_fine]."""
    optimizer.zero_grad(set_to_none=True)
    x, logq = flow.sample(condition_batch.shape[0], condition_batch)
    S = action(x)
    loss = (logq + S).mean()
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        logw = -S - logq
        ess = torch.exp(2 * torch.logsumexp(logw, dim=0) - torch.logsumexp(2 * logw, dim=0)) / logw.numel()
    return {
        "loss": float(loss.detach().cpu()),
        "S": float(S.mean().detach().cpu()),
        "logq": float(logq.mean().detach().cpu()),
        "ess_frac_batch": float(ess.detach().cpu()),
    }
