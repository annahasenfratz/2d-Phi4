"""Small 2D phi4 action, sampler, and observables."""

from __future__ import annotations

from dataclasses import dataclass

import torch


Tensor = torch.Tensor


@dataclass(frozen=True)
class Phi4Params:
    """Hopping-parameter convention.

    S = -2 kappa sum_{x,mu} phi_x phi_{x+mu}
        + sum_x phi_x^2
        + lambda sum_x (phi_x^2 - 1)^2.
    """

    kappa: float = 0.31
    lam: float = 1.0


def phi4_action_density(phi: Tensor, params: Phi4Params) -> Tensor:
    neighbor_sum = torch.roll(phi, -1, dims=-2) + torch.roll(phi, -1, dims=-1)
    local = phi.square() + params.lam * (phi.square() - 1.0).square()
    hopping = -2.0 * params.kappa * phi * neighbor_sum
    return local + hopping


def phi4_action(phi: Tensor, params: Phi4Params) -> Tensor:
    """Return one action value per leading configuration."""

    return phi4_action_density(phi, params).sum(dim=(-2, -1))


def _local_action(phi: Tensor, params: Phi4Params) -> Tensor:
    phi2 = phi.square()
    return phi2 + params.lam * (phi2 - 1.0).square()


@torch.no_grad()
def checkerboard_metropolis_sweep(
    phi: Tensor,
    params: Phi4Params,
    proposal_width: float,
    generator: torch.Generator | None = None,
) -> float:
    """One batched red-black local Metropolis sweep in place."""

    if phi.ndim != 3:
        raise ValueError("phi must have shape (batch, L, L)")
    batch, size_y, size_x = phi.shape
    yy, xx = torch.meshgrid(
        torch.arange(size_y, device=phi.device),
        torch.arange(size_x, device=phi.device),
        indexing="ij",
    )
    accepted = 0
    for parity in (0, 1):
        mask = ((yy + xx) % 2 == parity).unsqueeze(0)
        old = phi
        proposal = old + proposal_width * torch.randn(
            old.shape, dtype=old.dtype, device=old.device, generator=generator
        )
        neighbor_sum = (
            torch.roll(old, 1, dims=-2)
            + torch.roll(old, -1, dims=-2)
            + torch.roll(old, 1, dims=-1)
            + torch.roll(old, -1, dims=-1)
        )
        delta = (
            _local_action(proposal, params)
            - _local_action(old, params)
            - 2.0 * params.kappa * (proposal - old) * neighbor_sum
        )
        log_u = torch.log(torch.rand(old.shape, dtype=old.dtype, device=old.device, generator=generator))
        accept = mask & ((delta <= 0.0) | (log_u < -delta))
        phi[accept] = proposal[accept]
        accepted += int(accept.sum().item())
    return accepted / float(batch * size_y * size_x)


@torch.no_grad()
def generate_phi4_configs(
    n_configs: int,
    fine_size: int,
    params: Phi4Params,
    *,
    burn_in: int = 200,
    interval: int = 10,
    batch_size: int = 64,
    proposal_width: float = 1.0,
    seed: int = 1234,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Generate a small approximate fine ensemble with local Metropolis updates."""

    generator = torch.Generator(device=device).manual_seed(seed)
    phi = 0.5 * torch.randn(
        (batch_size, fine_size, fine_size), generator=generator, device=device
    )
    for _ in range(burn_in):
        checkerboard_metropolis_sweep(phi, params, proposal_width, generator)

    chunks = []
    while sum(chunk.shape[0] for chunk in chunks) < n_configs:
        for _ in range(interval):
            checkerboard_metropolis_sweep(phi, params, proposal_width, generator)
        chunks.append(phi.detach().cpu().clone())
    return torch.cat(chunks, dim=0)[:n_configs]


def magnetization(phi: Tensor) -> Tensor:
    return phi.mean(dim=(-2, -1))


def mean_phi2(phi: Tensor) -> Tensor:
    return phi.square().mean(dim=(-2, -1))


def nearest_neighbor_correlator(phi: Tensor) -> Tensor:
    corr = phi * torch.roll(phi, -1, dims=-2) + phi * torch.roll(phi, -1, dims=-1)
    return corr.mean(dim=(-2, -1)) / 2.0


def susceptibility(phi: Tensor) -> Tensor:
    volume = phi.shape[-2] * phi.shape[-1]
    return volume * magnetization(phi).square()


def binder_cumulant(phi: Tensor, eps: float = 1e-12) -> Tensor:
    m = magnetization(phi)
    m2 = m.square().mean()
    m4 = m.pow(4).mean()
    return 1.0 - m4 / (3.0 * m2.square().clamp_min(eps))


def summarize_observables(phi: Tensor, params: Phi4Params) -> dict[str, float]:
    with torch.no_grad():
        action = phi4_action(phi, params)
        return {
            "S_mean": float(action.mean().item()),
            "S_std": float(action.std(unbiased=False).item()),
            "mean_phi": float(magnetization(phi).mean().item()),
            "mean_phi2": float(mean_phi2(phi).mean().item()),
            "binder": float(binder_cumulant(phi).item()),
            "nn_corr": float(nearest_neighbor_correlator(phi).mean().item()),
            "susceptibility": float(susceptibility(phi).mean().item()),
        }

