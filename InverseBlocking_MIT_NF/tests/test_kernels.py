from pathlib import Path

import torch

from invblock_mit_nf.blocking import load_kernel_json, momentum_inverse_upscale_to_even_even


def test_preferred_kernel_sums_to_one():
    root = Path(__file__).resolve().parents[1]
    kernel = load_kernel_json(str(root / "kernels" / "finite_lambda_lam1_L32_to_L16_5x5_KL.json"))
    assert abs(sum(kernel.weights.values()) - 1.0) < 1e-12


def test_inverse_upscale_forward_block_reconstructs_coarse():
    root = Path(__file__).resolve().parents[1]
    kernel = load_kernel_json(str(root / "kernels" / "finite_lambda_lam1_L32_to_L16_5x5_KL.json"))
    torch.manual_seed(123)
    coarse = torch.randn(3, 8, 8, dtype=torch.float64)
    condition = momentum_inverse_upscale_to_even_even(coarse, kernel, Lf=16)
    even = condition[:, 0::2, 0::2]
    K = kernel.symbol(8, device=coarse.device)
    reconstructed = torch.fft.ifft2(torch.fft.fft2(even) * K.unsqueeze(0)).real
    assert torch.max(torch.abs(reconstructed - coarse)).item() < 1e-8
