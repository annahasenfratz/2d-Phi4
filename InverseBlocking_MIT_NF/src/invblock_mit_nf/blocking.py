from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch


@dataclass(frozen=True)
class BlockingKernel2D:
    """Translation-invariant 2D blocking kernel.

    The default representation is a small real-space stencil mapping fine fields to
    a coarse even-even field.  For inverse blocking we use its Fourier symbol K(q)
    on the coarse Brillouin zone and divide by K(q), with a safety floor.
    """

    weights: dict[tuple[int, int], float]
    name: str = "finite_lambda_kernel"
    eps: float = 1.0e-8

    def symbol(self, Lc: int, device: torch.device | None = None, dtype: torch.dtype = torch.float64) -> torch.Tensor:
        q = 2.0 * np.pi * torch.fft.fftfreq(Lc, d=1.0, device=device).to(dtype)
        qx, qy = torch.meshgrid(q, q, indexing="ij")
        K = torch.zeros((Lc, Lc), dtype=torch.complex128, device=device)
        for (dx, dy), w in self.weights.items():
            phase = torch.exp(-1j * (qx * dx + qy * dy)).to(torch.complex128)
            K = K + float(w) * phase
        return K

    def inverse_symbol(self, Lc: int, device: torch.device | None = None) -> torch.Tensor:
        K = self.symbol(Lc, device=device)
        absK = torch.abs(K)
        safe = torch.where(absK < self.eps, self.eps * torch.exp(1j * torch.angle(K)), K)
        return 1.0 / safe


def load_kernel_json(path: str) -> BlockingKernel2D:
    import json

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    weights = {tuple(map(int, k.split(","))): float(v) for k, v in data["weights"].items()}
    return BlockingKernel2D(weights=weights, name=data.get("name", "finite_lambda_kernel"), eps=data.get("eps", 1e-8))


def momentum_inverse_upscale_to_even_even(coarse: torch.Tensor, kernel: BlockingKernel2D, Lf: int | None = None) -> torch.Tensor:
    """Upscale Lc x Lc coarse fields to Lf x Lf with support on even-even sites.

    Steps:
      1. FFT coarse field.
      2. Divide by the finite-lambda blocking kernel K(q).
      3. IFFT to obtain an inferred even-even field.
      4. Embed into an Lf=2*Lc lattice on even-even sites; missing sites are zero.

    This is not the final fine configuration; it is the fixed condition for the
    conditional flow.
    """
    if coarse.ndim != 3:
        raise ValueError("coarse must have shape [batch, Lc, Lc]")
    batch, Lc, Lc2 = coarse.shape
    if Lc != Lc2:
        raise ValueError("coarse lattice must be square")
    Lf = Lf or 2 * Lc
    if Lf != 2 * Lc:
        raise ValueError("currently only scale factor 2 is implemented")
    coarse_hat = torch.fft.fft2(coarse.to(torch.float64))
    invK = kernel.inverse_symbol(Lc, device=coarse.device)
    even_hat = coarse_hat * invK.unsqueeze(0)
    even = torch.fft.ifft2(even_hat).real.to(coarse.dtype)
    fine_condition = torch.zeros((batch, Lf, Lf), dtype=coarse.dtype, device=coarse.device)
    fine_condition[:, 0::2, 0::2] = even
    return fine_condition


def even_even_mask(L: int, device: torch.device | None = None) -> torch.Tensor:
    mask = torch.zeros((L, L), dtype=torch.bool, device=device)
    mask[0::2, 0::2] = True
    return mask
