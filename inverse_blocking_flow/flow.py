"""Conditional affine-coupling flow for residual detail fields."""

from __future__ import annotations

import math

import torch
from torch import nn
Tensor = torch.Tensor


def make_conditioning(phi_c: Tensor, mode: str = "basic") -> Tensor:
    """Build conditioning channels from a coarse field.

    ``phi_c`` may have shape ``(B,L,L)`` or ``(B,1,L,L)``. The returned tensor
    always has shape ``(B,C,L,L)``.
    """

    if phi_c.ndim == 3:
        base = phi_c.unsqueeze(1)
    elif phi_c.ndim == 4 and phi_c.shape[1] == 1:
        base = phi_c
    else:
        raise ValueError("phi_c must have shape (B,L,L) or (B,1,L,L)")
    if mode == "basic":
        return base
    if mode != "physics":
        raise ValueError(f"unknown conditioning mode: {mode}")
    field = base[:, 0]
    grad_x = 0.5 * (torch.roll(field, -1, dims=-1) - torch.roll(field, 1, dims=-1))
    grad_y = 0.5 * (torch.roll(field, -1, dims=-2) - torch.roll(field, 1, dims=-2))
    grad_sq = grad_x.square() + grad_y.square()
    laplacian = (
        torch.roll(field, 1, dims=-1)
        + torch.roll(field, -1, dims=-1)
        + torch.roll(field, 1, dims=-2)
        + torch.roll(field, -1, dims=-2)
        - 4.0 * field
    )
    return torch.stack((field, field.square(), grad_x, grad_y, grad_sq, laplacian), dim=1)


def n_conditioning_channels(mode: str) -> int:
    if mode == "basic":
        return 1
    if mode == "physics":
        return 6
    raise ValueError(f"unknown conditioning mode: {mode}")


def _channel_mask(mask_values: tuple[int, ...], device: torch.device) -> Tensor:
    return torch.tensor(mask_values, dtype=torch.float32, device=device).view(1, len(mask_values), 1, 1)


def coupling_masks(n_channels: int) -> list[tuple[int, ...]]:
    if n_channels < 2:
        raise ValueError("n_channels must be at least 2")
    masks = []
    for shift in range(4):
        mask = tuple(1 if (i + shift) % 2 == 0 else 0 for i in range(n_channels))
        if sum(mask) == 0 or sum(mask) == n_channels:
            continue
        masks.append(mask)
    if n_channels == 3:
        masks.extend([(1, 1, 0), (0, 0, 1)])
    return masks


class CircularConvNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int, depth: int):
        super().__init__()
        if depth < 2:
            raise ValueError("depth must be at least 2")
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, padding_mode="circular"),
            nn.SiLU(),
        ]
        for _ in range(depth - 2):
            layers.extend(
                [
                    nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, padding_mode="circular"),
                    nn.SiLU(),
                ]
            )
        layers.append(
            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1, padding_mode="circular")
        )
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(layers[-1].weight)
        nn.init.zeros_(layers[-1].bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class AffineCoupling(nn.Module):
    def __init__(
        self,
        mask_values: tuple[int, ...],
        n_detail_channels: int = 3,
        hidden_channels: int = 48,
        depth: int = 3,
        n_conditioning_channels: int = 1,
        scale_clip: float = 3.0,
    ):
        super().__init__()
        self.mask_values = mask_values
        self.scale_clip = scale_clip
        self.n_detail_channels = n_detail_channels
        self.net = CircularConvNet(n_detail_channels + n_conditioning_channels, 2 * n_detail_channels, hidden_channels, depth)

    def _st(self, x: Tensor, cond: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        mask = _channel_mask(self.mask_values, x.device)
        frozen = x * mask
        h = self.net(torch.cat((frozen, cond), dim=1))
        shift, log_scale = h[:, : self.n_detail_channels], h[:, self.n_detail_channels :]
        active = 1.0 - mask
        log_scale = active * self.scale_clip * torch.tanh(log_scale / self.scale_clip)
        shift = active * shift
        return shift, log_scale, mask

    def forward(self, x: Tensor, cond: Tensor) -> tuple[Tensor, Tensor]:
        shift, log_scale, mask = self._st(x, cond)
        y = mask * x + (1.0 - mask) * (x * torch.exp(log_scale) + shift)
        logdet = log_scale.sum(dim=(1, 2, 3))
        return y, logdet

    def inverse(self, y: Tensor, cond: Tensor) -> tuple[Tensor, Tensor]:
        shift, log_scale, mask = self._st(y, cond)
        x = mask * y + (1.0 - mask) * ((y - shift) * torch.exp(-log_scale))
        logdet = -log_scale.sum(dim=(1, 2, 3))
        return x, logdet


class ConditionalDetailFlow(nn.Module):
    """Invertible flow in detail/noise channels for fixed coarse condition."""

    def __init__(
        self,
        n_layers: int = 6,
        hidden_channels: int = 48,
        depth: int = 3,
        n_conditioning_channels: int = 1,
        n_detail_channels: int = 3,
    ):
        super().__init__()
        self.n_conditioning_channels = n_conditioning_channels
        self.n_detail_channels = n_detail_channels
        masks = coupling_masks(n_detail_channels)
        self.layers = nn.ModuleList(
            [
                AffineCoupling(
                    masks[i % len(masks)],
                    n_detail_channels,
                    hidden_channels,
                    depth,
                    n_conditioning_channels,
                )
                for i in range(n_layers)
            ]
        )

    @staticmethod
    def standard_normal_logprob(x: Tensor) -> Tensor:
        return -0.5 * (x.square() + math.log(2.0 * math.pi)).sum(dim=(1, 2, 3))

    def forward(self, eta: Tensor, phi_c: Tensor) -> tuple[Tensor, Tensor]:
        """Map noise to details and return ``(d, log_q(d | phi_c))``.

        The returned density uses the forward change of variables
        ``d = F(eta; phi_c)``:

            log q(d | phi_c) = log p(eta) - log |det dF/deta|.
        """

        d, _, _, log_q = self.forward_decomposition(eta, phi_c)
        return d, log_q

    def forward_decomposition(self, eta: Tensor, phi_c: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Map noise to details and return ``(d, log_prior, logdet, logq)``."""

        x = eta
        logdet = torch.zeros(eta.shape[0], dtype=eta.dtype, device=eta.device)
        for layer in self.layers:
            x, ld = layer(x, phi_c)
            logdet = logdet + ld
        log_prior = self.standard_normal_logprob(eta)
        log_q = log_prior - logdet
        return x, log_prior, logdet, log_q

    def sample_and_logq(self, eta: Tensor, phi_c: Tensor) -> tuple[Tensor, Tensor]:
        """Explicit forward-direction API for caller-supplied noise."""

        return self.forward(eta, phi_c)

    def inverse(self, d: Tensor, phi_c: Tensor) -> tuple[Tensor, Tensor]:
        """Map details to noise and return ``(eta, logdet_inverse)``."""

        x = d
        logdet = torch.zeros(d.shape[0], dtype=d.dtype, device=d.device)
        for layer in reversed(self.layers):
            x, ld = layer.inverse(x, phi_c)
            logdet = logdet + ld
        return x, logdet

    def inverse_logq(self, d: Tensor, phi_c: Tensor) -> tuple[Tensor, Tensor]:
        """Map details to noise and return ``(eta, log_q(d | phi_c))``.

        The inverse log determinant is ``log |det deta/dd|``, so this returns
        ``log p(eta) + logdet_reverse``. This is the same value as the forward
        convention evaluated at the corresponding ``eta``.
        """

        eta, inverse_logdet = self.inverse(d, phi_c)
        log_q = self.standard_normal_logprob(eta) + inverse_logdet
        return eta, log_q

    def log_prob(self, d: Tensor, phi_c: Tensor) -> Tensor:
        _, log_q = self.inverse_logq(d, phi_c)
        return log_q

    def sample(
        self,
        phi_c: Tensor,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor]:
        eta = torch.randn(
            (phi_c.shape[0], self.n_detail_channels, phi_c.shape[-2], phi_c.shape[-1]),
            dtype=phi_c.dtype,
            device=phi_c.device,
            generator=generator,
        )
        return self.sample_and_logq(eta, phi_c)

    def sample_with_decomposition(
        self,
        phi_c: Tensor,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        eta = torch.randn(
            (phi_c.shape[0], self.n_detail_channels, phi_c.shape[-2], phi_c.shape[-1]),
            dtype=phi_c.dtype,
            device=phi_c.device,
            generator=generator,
        )
        return self.forward_decomposition(eta, phi_c)


def sanity_check_inverse(flow: ConditionalDetailFlow, batch: int = 4, size: int = 8) -> float:
    phi_c = torch.randn(batch, flow.n_conditioning_channels, size, size)
    eta = torch.randn(batch, flow.n_detail_channels, size, size)
    d, _ = flow(eta, phi_c)
    eta_back, _ = flow.inverse(d, phi_c)
    return float((eta - eta_back).abs().max().item())
