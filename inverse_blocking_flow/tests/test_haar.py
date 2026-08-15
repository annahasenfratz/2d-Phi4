from __future__ import annotations

import pytest
import torch

from inverse_blocking_flow.flow import ConditionalDetailFlow
from inverse_blocking_flow.haar import haar_block, haar_unblock
from inverse_blocking_flow.phi4 import Phi4Params, phi4_action


pytestmark = pytest.mark.slow


def test_haar_roundtrip_single_config() -> None:
    torch.manual_seed(1234)
    phi = torch.randn(32, 32, dtype=torch.float64)
    phi_c, d = haar_block(phi)
    reconstructed = haar_unblock(phi_c, d)
    assert torch.allclose(reconstructed, phi, atol=1e-12, rtol=1e-12)


def test_haar_roundtrip_batch() -> None:
    torch.manual_seed(1234)
    phi = torch.randn(5, 16, 16)
    phi_c, d = haar_block(phi)
    reconstructed = haar_unblock(phi_c, d)
    assert torch.allclose(reconstructed, phi, atol=1e-6, rtol=1e-6)


def test_phi4_action_is_batched() -> None:
    torch.manual_seed(1234)
    params = Phi4Params(kappa=0.31, lam=1.0)
    phi = torch.randn(7, 32, 32)
    action = phi4_action(phi, params)
    assert action.shape == (7,)
    assert torch.isfinite(action).all()


def test_flow_inverse_roundtrip() -> None:
    torch.manual_seed(1234)
    flow = ConditionalDetailFlow(n_layers=4, hidden_channels=8, depth=2)
    phi_c = torch.randn(3, 1, 8, 8)
    eta = torch.randn(3, 3, 8, 8)
    d, log_q_forward = flow.sample_and_logq(eta, phi_c)
    eta_back, log_q_inverse = flow.inverse_logq(d, phi_c)
    assert torch.allclose(eta_back, eta, atol=1e-5, rtol=1e-5)
    assert torch.allclose(log_q_forward, log_q_inverse, atol=1e-5, rtol=1e-5)
