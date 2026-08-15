import torch

from invblock_mit_nf.blocking import BlockingKernel2D, momentum_inverse_upscale_to_even_even
from invblock_mit_nf.conditional_flow import ConditionalPhi4Flow


def test_identity_inverse_upscale_places_coarse_on_even_even():
    coarse = torch.randn(2, 8, 8)
    cond = momentum_inverse_upscale_to_even_even(coarse, BlockingKernel2D({(0, 0): 1.0}))
    assert cond.shape == (2, 16, 16)
    assert torch.allclose(cond[:, 0::2, 0::2], coarse)
    assert torch.all(cond[:, 1::2, :] == 0)
    assert torch.all(cond[:, :, 1::2][:, 0::2, :] == 0)


def test_flow_keeps_even_even_fixed():
    coarse = torch.randn(2, 8, 8)
    cond = momentum_inverse_upscale_to_even_even(coarse, BlockingKernel2D({(0, 0): 1.0}))
    flow = ConditionalPhi4Flow(L=16, n_layers=2, hidden=8)
    y, _ = flow.sample(2, cond)
    assert torch.allclose(y[:, 0::2, 0::2], cond[:, 0::2, 0::2])
