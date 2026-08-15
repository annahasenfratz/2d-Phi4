from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass(frozen=True)
class Phi4Params:
    """Finite-lambda phi^4 parameters in our normalization."""

    kappa: float
    lam: float

    def to_mit(self) -> tuple[float, float]:
        """Return (M2, lambda_MIT) for the MIT tutorial normalization.

        MIT action in 2D:
            S = sum_x [(M2 + 4) varphi_x^2 + lam_MIT varphi_x^4]
                - 2 sum_{x,mu} varphi_x varphi_{x+mu}

        Our action:
            S = -2 kappa sum_{x,mu} phi_x phi_{x+mu}
                + sum_x [phi_x^2 + lam (phi_x^2 - 1)^2]

        Rescaling varphi = sqrt(kappa) phi gives, up to an additive constant,
            M2 = (1 - 2 lam)/kappa - 4
            lam_MIT = lam/kappa^2.
        """
        if self.kappa <= 0:
            raise ValueError("kappa must be positive")
        return (1.0 - 2.0 * self.lam) / self.kappa - 4.0, self.lam / (self.kappa**2)

    @staticmethod
    def from_mit(M2: float, lam_mit: float) -> "Phi4Params":
        """Invert the MIT conversion.

        Solves lam = lam_mit kappa^2 and M2 + 4 = (1 - 2 lam)/kappa.
        This gives 2 lam_mit kappa^2 + (M2 + 4) kappa - 1 = 0.
        """
        if lam_mit <= 0:
            raise ValueError("lam_mit must be positive")
        b = M2 + 4.0
        disc = b * b + 8.0 * lam_mit
        kappa = (-b + disc**0.5) / (4.0 * lam_mit)
        lam = lam_mit * kappa * kappa
        return Phi4Params(kappa=kappa, lam=lam)


class Phi4Action:
    """2D scalar phi^4 action in our finite-lambda normalization."""

    def __init__(self, params: Phi4Params):
        self.params = params

    def __call__(self, cfgs: torch.Tensor) -> torch.Tensor:
        """Return action for cfgs with shape [batch, L, L] or [..., L, L]."""
        if cfgs.ndim < 2:
            raise ValueError("cfgs must have at least two lattice dimensions")
        kappa = self.params.kappa
        lam = self.params.lam
        dims = (-2, -1)
        potential = cfgs**2 + lam * (cfgs**2 - 1.0) ** 2
        nearest = torch.zeros_like(cfgs)
        for dim in dims:
            nearest = nearest + cfgs * torch.roll(cfgs, shifts=-1, dims=dim)
        density = potential - 2.0 * kappa * nearest
        return density.sum(dim=dims)


class MITPhi4Action:
    """MIT tutorial action, useful for cross-checks only."""

    def __init__(self, M2: float, lam_mit: float):
        self.M2 = M2
        self.lam_mit = lam_mit

    def __call__(self, cfgs: torch.Tensor) -> torch.Tensor:
        dims = (-2, -1)
        density = self.M2 * cfgs**2 + self.lam_mit * cfgs**4
        for dim in dims:
            density = density + 2.0 * cfgs**2
            density = density - cfgs * torch.roll(cfgs, shifts=-1, dims=dim)
            density = density - cfgs * torch.roll(cfgs, shifts=1, dims=dim)
        return density.sum(dim=dims)


def rescale_ours_to_mit(phi_ours: torch.Tensor, kappa: float) -> torch.Tensor:
    return (kappa**0.5) * phi_ours


def rescale_mit_to_ours(phi_mit: torch.Tensor, kappa: float) -> torch.Tensor:
    return phi_mit / (kappa**0.5)
