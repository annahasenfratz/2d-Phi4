from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from .rational_quadratic_spline import inverse_softplus, unconstrained_rational_quadratic_spline


def build_procedural_conv_spline_flow(
    *,
    cond_channels: int,
    target_channels: int = 1,
    lattice_size: int = 8,
    n_coupling_layers: int = 8,
    conv_hidden_channels: int = 48,
    conv_kernel_size: int = 3,
    num_bins: int = 8,
    tail_bound: float = 6.0,
    min_bin_width: float = 1.0e-3,
    min_bin_height: float = 1.0e-3,
    min_derivative: float = 1.0e-3,
):
    d_channels = int(target_channels)
    c_channels = int(cond_channels)
    h_size = int(lattice_size)
    w_size = int(lattice_size)
    k_size = int(conv_kernel_size)
    bins = int(num_bins)
    if k_size < 1 or k_size % 2 != 1:
        raise ValueError("conv_kernel_size must be a positive odd integer")
    if d_channels != 1:
        raise ValueError("this spline coupling builder currently supports target_channels=1")
    pad = k_size // 2
    params_per_channel = 3 * bins - 1
    derivative_identity_bias = inverse_softplus(1.0 - float(min_derivative))

    class CircularConvSplineNet(nn.Module):
        def __init__(self):
            super().__init__()
            h = int(conv_hidden_channels)
            self.net = nn.Sequential(
                nn.Conv2d(d_channels + c_channels, h, kernel_size=k_size, padding=pad, padding_mode="circular"),
                nn.SiLU(),
                nn.Conv2d(h, h, kernel_size=k_size, padding=pad, padding_mode="circular"),
                nn.SiLU(),
                nn.Conv2d(h, h, kernel_size=k_size, padding=pad, padding_mode="circular"),
                nn.SiLU(),
                nn.Conv2d(h, params_per_channel * d_channels, kernel_size=k_size, padding=pad, padding_mode="circular"),
            )
            self.net[-1].weight.data.zero_()
            self.net[-1].bias.data.zero_()
            start = 2 * bins
            self.net[-1].bias.data[start : start + bins - 1].fill_(derivative_identity_bias)

        def forward(self, d_masked, c_img):
            return self.net(torch.cat([d_masked, c_img], dim=1))

    def make_masks() -> list[torch.Tensor]:
        masks = []
        yy, xx = torch.meshgrid(torch.arange(h_size), torch.arange(w_size), indexing="ij")
        checker0 = ((xx + yy) % 2 == 0).float()
        checker1 = 1.0 - checker0
        for i in range(int(n_coupling_layers)):
            mask = torch.zeros(d_channels, h_size, w_size)
            mask[0] = checker0 if i % 2 == 0 else checker1
            masks.append(mask)
        return masks

    class ProceduralConvSplineFlow(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("masks", torch.stack(make_masks()), persistent=False)
            self.nets = nn.ModuleList([CircularConvSplineNet() for _ in range(int(n_coupling_layers))])
            self.cond_channels = c_channels
            self.target_channels = d_channels
            self.lattice_size = h_size

        def _reshape_d(self, x):
            return x.reshape(x.shape[0], d_channels, h_size, w_size)

        def _reshape_c(self, ccond):
            return ccond.reshape(ccond.shape[0], c_channels, h_size, w_size)

        def _params(self, net_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            n = net_out.shape[0]
            p = net_out.reshape(n, d_channels, params_per_channel, h_size, w_size).permute(0, 1, 3, 4, 2)
            widths = p[..., :bins]
            heights = p[..., bins : 2 * bins]
            derivatives = p[..., 2 * bins :]
            return widths, heights, derivatives

        def _elementwise(self, x: torch.Tensor, widths: torch.Tensor, heights: torch.Tensor, derivatives: torch.Tensor, *, inverse: bool):
            y, lad = unconstrained_rational_quadratic_spline(
                x,
                widths,
                heights,
                derivatives,
                inverse=inverse,
                tail_bound=tail_bound,
                min_bin_width=min_bin_width,
                min_bin_height=min_bin_height,
                min_derivative=min_derivative,
            )
            return y, lad

        def forward(self, z, ccond):
            x = self._reshape_d(z)
            c_img = self._reshape_c(ccond)
            logdet = z.new_zeros(z.shape[0])
            for layer, net in enumerate(self.nets):
                mask = self.masks[layer]
                xm = x * mask
                widths, heights, derivatives = self._params(net(xm, c_img))
                active = 1.0 - mask
                y, lad = self._elementwise(x, widths, heights, derivatives, inverse=False)
                x = xm + active * y
                logdet = logdet + (lad * active).flatten(1).sum(dim=1)
            return x.flatten(1), logdet

        def inverse(self, d, ccond):
            z = self._reshape_d(d)
            c_img = self._reshape_c(ccond)
            logdet = d.new_zeros(d.shape[0])
            for layer, net in reversed(list(enumerate(self.nets))):
                mask = self.masks[layer]
                zm = z * mask
                widths, heights, derivatives = self._params(net(zm, c_img))
                active = 1.0 - mask
                x, lad = self._elementwise(z, widths, heights, derivatives, inverse=True)
                z = zm + active * x
                logdet = logdet + (lad * active).flatten(1).sum(dim=1)
            return z.flatten(1), logdet

        def log_prob(self, d, ccond):
            z, logdet = self.inverse(d, ccond)
            log_base = -0.5 * (z * z + math.log(2.0 * math.pi)).sum(dim=1)
            return log_base + logdet

        def dependency_report(self) -> dict[str, Any]:
            coarse_radius = 3 * pad * int(n_coupling_layers)
            return {
                "coarse_radius": coarse_radius,
                "fine_radius": 2 * coarse_radius,
                "metric": "chebyshev_composed_circular_conv_spline",
                "n_coupling_layers": int(n_coupling_layers),
                "conv_kernel": k_size,
                "conv_layers_per_conditioner": 3,
                "lattice_size": h_size,
                "cond_channels": c_channels,
                "target_channels": d_channels,
                "num_bins": bins,
                "tail_bound": float(tail_bound),
            }

    return ProceduralConvSplineFlow()
