from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any


def build_procedural_conv_flow(
    *,
    cond_channels: int,
    target_channels: int = 1,
    lattice_size: int = 8,
    n_coupling_layers: int = 16,
    conv_hidden_channels: int = 64,
    conv_kernel_size: int = 3,
    log_scale_bound: float = 0.75,
):
    import torch
    from torch import nn

    d_channels = int(target_channels)
    c_channels = int(cond_channels)
    h_size = int(lattice_size)
    w_size = int(lattice_size)
    k_size = int(conv_kernel_size)
    if k_size < 1 or k_size % 2 != 1:
        raise ValueError("conv_kernel_size must be a positive odd integer")
    pad = k_size // 2

    class CircularConvNet(nn.Module):
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
                nn.Conv2d(h, 2 * d_channels, kernel_size=k_size, padding=pad, padding_mode="circular"),
            )
            self.net[-1].weight.data.zero_()
            self.net[-1].bias.data.zero_()

        def forward(self, d_masked, c_img):
            return self.net(torch.cat([d_masked, c_img], dim=1))

    def make_masks() -> list[Any]:
        masks = []
        yy, xx = torch.meshgrid(torch.arange(h_size), torch.arange(w_size), indexing="ij")
        checker0 = ((xx + yy) % 2 == 0).float()
        checker1 = 1.0 - checker0
        for i in range(int(n_coupling_layers)):
            mask = torch.zeros(d_channels, h_size, w_size)
            if d_channels == 1:
                mask[0] = checker0 if i % 2 == 0 else checker1
            else:
                kind = i % 4
                if kind == 0:
                    mask[:] = checker0
                elif kind == 1:
                    mask[:] = checker1
                elif kind == 2:
                    for ch in range(d_channels):
                        mask[ch] = 1.0 if ch % 2 == 0 else checker0
                else:
                    for ch in range(d_channels):
                        mask[ch] = 1.0 if ch % 2 == 1 else checker1
            masks.append(mask)
        return masks

    class ProceduralConvAffineFlow(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("masks", torch.stack(make_masks()), persistent=False)
            self.nets = nn.ModuleList([CircularConvNet() for _ in range(int(n_coupling_layers))])
            self.log_scale_bound = float(log_scale_bound)
            self.cond_channels = c_channels
            self.target_channels = d_channels
            self.lattice_size = h_size

        def _reshape_d(self, x):
            return x.reshape(x.shape[0], d_channels, h_size, w_size)

        def _reshape_c(self, ccond):
            return ccond.reshape(ccond.shape[0], c_channels, h_size, w_size)

        def forward(self, z, ccond):
            x = self._reshape_d(z)
            c_img = self._reshape_c(ccond)
            logdet = z.new_zeros(z.shape[0])
            for layer, net in enumerate(self.nets):
                mask = self.masks[layer]
                xm = x * mask
                shift, log_scale = net(xm, c_img).chunk(2, dim=1)
                active = 1.0 - mask
                log_scale = self.log_scale_bound * torch.tanh(log_scale / self.log_scale_bound) * active
                shift = shift * active
                x = xm + active * (x * torch.exp(log_scale) + shift)
                logdet = logdet + log_scale.flatten(1).sum(dim=1)
            return x.flatten(1), logdet

        def forward_with_logdet_map(self, z, ccond):
            x = self._reshape_d(z)
            c_img = self._reshape_c(ccond)
            logdet_map = x.new_zeros(x.shape)
            for layer, net in enumerate(self.nets):
                mask = self.masks[layer]
                xm = x * mask
                shift, log_scale = net(xm, c_img).chunk(2, dim=1)
                active = 1.0 - mask
                log_scale = self.log_scale_bound * torch.tanh(log_scale / self.log_scale_bound) * active
                shift = shift * active
                x = xm + active * (x * torch.exp(log_scale) + shift)
                logdet_map = logdet_map + log_scale
            return x.flatten(1), logdet_map

        def inverse(self, d, ccond):
            z = self._reshape_d(d)
            c_img = self._reshape_c(ccond)
            logdet = d.new_zeros(d.shape[0])
            for layer, net in reversed(list(enumerate(self.nets))):
                mask = self.masks[layer]
                zm = z * mask
                shift, log_scale = net(zm, c_img).chunk(2, dim=1)
                active = 1.0 - mask
                log_scale = self.log_scale_bound * torch.tanh(log_scale / self.log_scale_bound) * active
                shift = shift * active
                z = zm + active * ((z - shift) * torch.exp(-log_scale))
                logdet = logdet - log_scale.flatten(1).sum(dim=1)
            return z.flatten(1), logdet

        def log_prob(self, d, ccond):
            z, logdet = self.inverse(d, ccond)
            log_base = -0.5 * (z * z + math.log(2.0 * math.pi)).sum(dim=1)
            return log_base + logdet

        def sample(self, ccond):
            z = torch.randn(
                ccond.shape[0],
                d_channels * h_size * w_size,
                device=ccond.device,
                dtype=ccond.dtype,
            )
            log_base = -0.5 * (z * z + math.log(2.0 * math.pi)).sum(dim=1)
            x, logdet = self.forward(z, ccond)
            return x, log_base - logdet

        def dependency_report(self) -> dict[str, Any]:
            # Each conditioner has three circular convolutions. Across
            # coupling layers the transformed field becomes part of later
            # conditioner input, so the conservative composed radius is
            # 3 * floor(kernel/2) per
            # layer on the coarse/stage lattice.
            coarse_radius = 3 * pad * int(n_coupling_layers)
            return {
                "coarse_radius": coarse_radius,
                "fine_radius": 2 * coarse_radius,
                "metric": "chebyshev_composed_circular_conv",
                "n_coupling_layers": int(n_coupling_layers),
                "conv_kernel": k_size,
                "conv_layers_per_conditioner": 3,
                "lattice_size": h_size,
                "cond_channels": c_channels,
                "target_channels": d_channels,
            }

    return ProceduralConvAffineFlow()


def build_procedural_conv_from_checkpoint(checkpoint: dict[str, Any], cond_channels: int, lattice_size: int):
    cfg = SimpleNamespace(**checkpoint["config"])
    return build_procedural_conv_flow(
        cond_channels=cond_channels,
        target_channels=int(getattr(cfg, "target_channels", 1)),
        lattice_size=int(lattice_size),
        n_coupling_layers=int(getattr(cfg, "n_coupling_layers", 16)),
        conv_hidden_channels=int(getattr(cfg, "conv_hidden_channels", 64)),
        conv_kernel_size=int(getattr(cfg, "conv_kernel_size", 3)),
        log_scale_bound=float(getattr(cfg, "log_scale_bound", 0.75)),
    )
