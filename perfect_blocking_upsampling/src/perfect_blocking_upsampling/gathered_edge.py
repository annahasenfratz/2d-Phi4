from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class GatheredStencil:
    radius: int
    offsets: tuple[tuple[int, int], ...]
    metric: str = "chebyshev"

    @property
    def fine_radius(self) -> int:
        return 2 * self.radius


def square_stencil(radius: int) -> GatheredStencil:
    if radius < 0:
        raise ValueError("radius must be non-negative")
    offsets = tuple((dx, dy) for dx in range(-radius, radius + 1) for dy in range(-radius, radius + 1))
    return GatheredStencil(radius=radius, offsets=offsets, metric="chebyshev")


def manhattan_stencil(radius: int) -> GatheredStencil:
    if radius < 0:
        raise ValueError("radius must be non-negative")
    offsets = tuple(
        (dx, dy)
        for dx in range(-radius, radius + 1)
        for dy in range(-radius, radius + 1)
        if abs(dx) + abs(dy) <= radius
    )
    return GatheredStencil(radius=radius, offsets=offsets, metric="manhattan")


def periodic_shortest_displacement(delta: int, length: int) -> int:
    if length <= 0:
        raise ValueError("length must be positive")
    wrapped = ((int(delta) + length // 2) % length) - length // 2
    if length % 2 == 0 and wrapped == -length // 2 and int(delta) > 0:
        return length // 2
    return wrapped


def validate_periodic_offsets(offsets: Iterable[tuple[int, int]], radius: int, lattice_size: int, metric: str = "chebyshev") -> tuple[tuple[int, int], ...]:
    out: list[tuple[int, int]] = []
    for dx, dy in offsets:
        sx = periodic_shortest_displacement(dx, lattice_size)
        sy = periodic_shortest_displacement(dy, lattice_size)
        if metric == "chebyshev":
            dist = max(abs(sx), abs(sy))
        elif metric == "manhattan":
            dist = abs(sx) + abs(sy)
        else:
            raise ValueError(f"unknown stencil metric {metric!r}")
        if dist > radius:
            raise ValueError(
                f"offset {(dx, dy)} has shortest periodic displacement {(sx, sy)} "
                f"outside declared radius {radius}"
            )
        out.append((int(dx), int(dy)))
    return tuple(out)


def build_gathered_edge_flow(
    *,
    cond_channels: int,
    lattice_size: int,
    radius: int = 3,
    stencil: str = "square",
    hidden_width: int = 96,
    hidden_layers: int = 2,
    log_scale_bound: float = 0.75,
):
    import torch
    from torch import nn

    if cond_channels <= 0:
        raise ValueError("cond_channels must be positive")
    if stencil == "square":
        spec = square_stencil(radius)
    elif stencil == "manhattan":
        spec = manhattan_stencil(radius)
    else:
        raise ValueError(f"unknown stencil {stencil!r}")
    offsets = validate_periodic_offsets(spec.offsets, spec.radius, lattice_size, spec.metric)
    in_dim = cond_channels * len(offsets)

    class GatheredEdgeAffineFlow(nn.Module):
        dependency_radius = spec.radius
        dependency_fine_radius = spec.fine_radius
        stencil_metric = spec.metric
        stencil_offsets = offsets

        def __init__(self):
            super().__init__()
            layers: list[nn.Module] = []
            width = int(hidden_width)
            last = in_dim
            for _ in range(int(hidden_layers)):
                layers.append(nn.Linear(last, width))
                layers.append(nn.SiLU())
                last = width
            layers.append(nn.Linear(last, 2))
            self.net = nn.Sequential(*layers)
            self.log_scale_bound = float(log_scale_bound)
            self.cond_channels = int(cond_channels)
            self.lattice_size = int(lattice_size)
            self.register_buffer("_dummy", torch.zeros(()), persistent=False)
            final = self.net[-1]
            assert isinstance(final, nn.Linear)
            final.weight.data.zero_()
            final.bias.data.zero_()

        def _reshape_d(self, x):
            return x.reshape(x.shape[0], 1, self.lattice_size, self.lattice_size)

        def _reshape_c(self, ccond):
            return ccond.reshape(ccond.shape[0], self.cond_channels, self.lattice_size, self.lattice_size)

        def conditioner(self, ccond):
            c_img = self._reshape_c(ccond)
            gathered = [torch.roll(c_img, shifts=(-dx, -dy), dims=(2, 3)) for dx, dy in self.stencil_offsets]
            feat = torch.cat(gathered, dim=1).permute(0, 2, 3, 1)
            st = self.net(feat.reshape(-1, feat.shape[-1]))
            st = st.reshape(c_img.shape[0], self.lattice_size, self.lattice_size, 2).permute(0, 3, 1, 2)
            shift = st[:, 0:1]
            log_scale_raw = st[:, 1:2]
            log_scale = self.log_scale_bound * torch.tanh(log_scale_raw / self.log_scale_bound)
            return shift, log_scale

        def forward(self, z, ccond):
            z_img = self._reshape_d(z)
            shift, log_scale = self.conditioner(ccond)
            x = z_img * torch.exp(log_scale) + shift
            logdet = log_scale.flatten(1).sum(dim=1)
            return x.flatten(1), logdet

        def inverse(self, d, ccond):
            d_img = self._reshape_d(d)
            shift, log_scale = self.conditioner(ccond)
            z = (d_img - shift) * torch.exp(-log_scale)
            logdet = -log_scale.flatten(1).sum(dim=1)
            return z.flatten(1), logdet

        def log_prob(self, d, ccond):
            z, logdet = self.inverse(d, ccond)
            log_base = -0.5 * (z * z + math.log(2.0 * math.pi)).sum(dim=1)
            return log_base + logdet

        def sample(self, ccond):
            z = torch.randn(ccond.shape[0], self.lattice_size * self.lattice_size, device=ccond.device, dtype=ccond.dtype)
            x, logdet = self.forward(z, ccond)
            log_base = -0.5 * (z * z + math.log(2.0 * math.pi)).sum(dim=1)
            return x, log_base - logdet

        def dependency_report(self) -> dict[str, Any]:
            return {
                "coarse_radius": int(self.dependency_radius),
                "fine_radius": int(self.dependency_fine_radius),
                "metric": self.stencil_metric,
                "n_offsets": len(self.stencil_offsets),
                "offsets": [list(x) for x in self.stencil_offsets],
                "lattice_size": int(self.lattice_size),
                "cond_channels": int(self.cond_channels),
            }

    return GatheredEdgeAffineFlow()


def build_gathered_edge_from_checkpoint(checkpoint: dict[str, Any], cond_channels: int, lattice_size: int):
    cfg = checkpoint["config"]
    model = build_gathered_edge_flow(
        cond_channels=cond_channels,
        lattice_size=lattice_size,
        radius=int(cfg.get("gather_radius", cfg.get("radius", 3))),
        stencil=str(cfg.get("gather_stencil", "square")),
        hidden_width=int(cfg.get("gather_hidden_width", 96)),
        hidden_layers=int(cfg.get("gather_hidden_layers", 2)),
        log_scale_bound=float(cfg.get("log_scale_bound", 0.75)),
    )
    return model


def corrcoef_flat(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    if aa.size == 0 or np.std(aa) == 0.0 or np.std(bb) == 0.0:
        return float("nan")
    return float(np.corrcoef(aa, bb)[0, 1])
