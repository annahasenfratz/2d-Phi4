from __future__ import annotations

import torch
from torch import nn

from .blocking import even_even_mask


class ConvNetConditioner(nn.Module):
    """Small CNN conditioner for affine coupling layers."""

    def __init__(self, channels: int = 2, hidden: int = 32, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size, padding=pad, padding_mode="circular"),
            nn.LeakyReLU(0.01),
            nn.Conv2d(hidden, hidden, kernel_size, padding=pad, padding_mode="circular"),
            nn.LeakyReLU(0.01),
            nn.Conv2d(hidden, 2, kernel_size, padding=pad, padding_mode="circular"),
        )

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inp = torch.stack([x, condition], dim=1)
        s_t = self.net(inp)
        log_s = 0.8 * torch.tanh(s_t[:, 0])
        t = s_t[:, 1]
        return log_s, t


class ConditionalAffineCoupling(nn.Module):
    """Affine coupling that never changes fixed even-even sites."""

    def __init__(self, L: int, parity: int, hidden: int = 32):
        super().__init__()
        if parity not in (0, 1):
            raise ValueError("parity must be 0 or 1")
        ii, jj = torch.meshgrid(torch.arange(L), torch.arange(L), indexing="ij")
        checker = ((ii + jj) % 2 == parity)
        fixed = even_even_mask(L)
        active = checker & (~fixed)
        self.register_buffer("active", active.float())
        self.conditioner = ConvNetConditioner(hidden=hidden)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        log_s, t = self.conditioner(x * (1.0 - self.active), condition)
        log_s = log_s * self.active
        t = t * self.active
        y = x * torch.exp(log_s) + t
        y = torch.where(self.active.bool().unsqueeze(0), y, x)
        log_det = log_s.flatten(1).sum(dim=1)
        return y, log_det

    def inverse(self, y: torch.Tensor, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        log_s, t = self.conditioner(y * (1.0 - self.active), condition)
        log_s = log_s * self.active
        t = t * self.active
        x = (y - t) * torch.exp(-log_s)
        x = torch.where(self.active.bool().unsqueeze(0), x, y)
        log_det = -log_s.flatten(1).sum(dim=1)
        return x, log_det


class ConditionalPhi4Flow(nn.Module):
    """MIT-style conditional RealNVP flow for missing sites at fixed even-even field."""

    def __init__(self, L: int, n_layers: int = 8, hidden: int = 32):
        super().__init__()
        self.L = L
        self.layers = nn.ModuleList([
            ConditionalAffineCoupling(L, parity=i % 2, hidden=hidden) for i in range(n_layers)
        ])
        self.register_buffer("fixed_mask", even_even_mask(L).float())

    def sample_base(self, batch: int, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = torch.randn((batch, self.L, self.L), dtype=condition.dtype, device=condition.device)
        z = z * (1.0 - self.fixed_mask) + condition * self.fixed_mask
        n_free = int((1.0 - self.fixed_mask).sum().item())
        logq = -0.5 * ((z * (1.0 - self.fixed_mask)) ** 2).flatten(1).sum(dim=1)
        logq = logq - 0.5 * n_free * torch.log(torch.tensor(2.0 * torch.pi, dtype=condition.dtype, device=condition.device))
        return z, logq

    def forward(self, z: torch.Tensor, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = z
        total_log_det = torch.zeros(z.shape[0], dtype=z.dtype, device=z.device)
        for layer in self.layers:
            x, log_det = layer(x, condition)
            total_log_det = total_log_det + log_det
        x = x * (1.0 - self.fixed_mask) + condition * self.fixed_mask
        return x, total_log_det

    def reverse(self, y: torch.Tensor, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = y
        total_log_det = torch.zeros(y.shape[0], dtype=y.dtype, device=y.device)
        for layer in reversed(self.layers):
            x, log_det = layer.inverse(x, condition)
            total_log_det = total_log_det + log_det
        x = x * (1.0 - self.fixed_mask) + condition * self.fixed_mask
        return x, total_log_det

    def log_prob(self, y: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        z, log_det = self.reverse(y, condition)
        n_free = int((1.0 - self.fixed_mask).sum().item())
        logq_base = -0.5 * ((z * (1.0 - self.fixed_mask)) ** 2).flatten(1).sum(dim=1)
        logq_base = logq_base - 0.5 * n_free * torch.log(
            torch.tensor(2.0 * torch.pi, dtype=y.dtype, device=y.device)
        )
        return logq_base + log_det

    def sample(self, batch: int, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z, logq_base = self.sample_base(batch, condition)
        x, log_det = self.forward(z, condition)
        logq = logq_base - log_det
        return x, logq
