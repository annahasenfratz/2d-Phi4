from __future__ import annotations

import importlib

import pytest
import torch

from inverse_blocking_flow.flow import ConditionalDetailFlow, make_conditioning
from inverse_blocking_flow.haar import (
    average_block,
    block_average,
    detail_to_residual,
    haar_block,
    haar_unblock,
    prolong_constant,
    reconstruct_from_average_block,
    residual_to_detail,
    reconstruct_from_weighted_block,
    soft_block,
    soft_kernel_term,
    soft_reconstruct,
    soft_weighted_block,
    soft_weighted_kernel_term,
    soft_weighted_reconstruct,
    weighted_block,
    weighted_ll_fft_min_abs,
)
from inverse_blocking_flow.phi4 import Phi4Params, phi4_action
from inverse_blocking_flow.cluster_phi4_reference_check import onsite_action_density, wolff_bond_probability


pytestmark = pytest.mark.fast


def seeded_randn(*shape: int, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    gen = torch.Generator().manual_seed(1234)
    return torch.randn(*shape, dtype=dtype, generator=gen)


def test_haar_block_unblock_exact_reconstruction_float64() -> None:
    phi_f = seeded_randn(1, 16, 16)

    ll, d = haar_block(phi_f)
    phi_rec = haar_unblock(ll, d)

    max_abs_error = (phi_rec - phi_f).abs().max().item()
    assert max_abs_error < 1e-10


def test_haar_shapes() -> None:
    batch = 1
    phi_f = seeded_randn(batch, 16, 16, dtype=torch.float32)

    ll, d = haar_block(phi_f)

    assert phi_f.shape == (batch, 16, 16)
    assert ll.shape in ((batch, 8, 8), (batch, 1, 8, 8))
    assert d.shape == (batch, 3, 8, 8)


def test_average_block_reconstruction_preserves_coarse_field() -> None:
    batch = 1
    phi_c = seeded_randn(batch, 8, 8)
    d = seeded_randn(batch, 3, 8, 8)

    phi_rec = reconstruct_from_average_block(phi_c, d)

    assert torch.allclose(block_average(phi_rec), phi_c, atol=1e-12, rtol=1e-12)


def test_detail_residual_has_zero_block_average() -> None:
    batch = 1
    d = seeded_randn(batch, 3, 8, 8)

    chi = detail_to_residual(d)

    zeros = torch.zeros(batch, 8, 8, dtype=torch.float64)
    assert torch.allclose(block_average(chi), zeros, atol=1e-12, rtol=1e-12)


def test_reconstruct_from_true_residual_equals_original() -> None:
    batch = 1
    phi_f = seeded_randn(batch, 16, 16)

    phi_c = block_average(phi_f)
    phi0 = prolong_constant(phi_c)
    chi_true = phi_f - phi0
    d_true = residual_to_detail(chi_true)
    phi_rec = reconstruct_from_average_block(phi_c, d_true)
    phi_c_from_helper, d_from_helper = average_block(phi_f)

    assert torch.allclose(phi_rec, phi_f, atol=1e-10, rtol=1e-10)
    assert torch.allclose(phi_c_from_helper, phi_c, atol=1e-12, rtol=1e-12)
    assert torch.allclose(d_from_helper, d_true, atol=1e-12, rtol=1e-12)


def test_phi4_action_shape_finite_and_translation_invariant() -> None:
    batch = 1
    params = Phi4Params(kappa=0.31, lam=1.0)
    phi = seeded_randn(batch, 8, 8, dtype=torch.float32)

    action = phi4_action(phi, params)
    rolled_action = phi4_action(torch.roll(phi, shifts=1, dims=-1), params)

    assert action.shape == (batch,)
    assert torch.isfinite(action).all()
    assert torch.allclose(action, rolled_action, atol=1e-5, rtol=1e-5)


def test_wolff_bond_probability_matches_phi4_convention() -> None:
    phi_x = torch.tensor([1.0, -2.0, 0.5])
    phi_y = torch.tensor([3.0, -0.25, -4.0])
    kappa = 0.3

    expected = 1.0 - torch.exp(-4.0 * kappa * (phi_x * phi_y).abs())

    assert torch.allclose(wolff_bond_probability(phi_x, phi_y, kappa), expected)


def test_sign_flip_preserves_phi4_onsite_terms() -> None:
    phi = seeded_randn(1, 8, 8)
    mask = torch.zeros_like(phi, dtype=torch.bool)
    mask[..., ::2, 1::2] = True
    flipped = phi.clone()
    flipped[mask] = -flipped[mask]

    assert torch.allclose(onsite_action_density(phi, lam=1.0), onsite_action_density(flipped, lam=1.0))


@pytest.mark.parametrize("n_detail_channels", [3, 4])
def test_conditional_flow_detail_channel_shapes(n_detail_channels: int) -> None:
    batch = 1
    coarse_size = 8
    flow = ConditionalDetailFlow(
        n_layers=1,
        hidden_channels=8,
        depth=2,
        n_conditioning_channels=1,
        n_detail_channels=n_detail_channels,
    )
    phi_c = seeded_randn(batch, 1, coarse_size, coarse_size, dtype=torch.float32)
    eta = seeded_randn(batch, n_detail_channels, coarse_size, coarse_size, dtype=torch.float32)

    d, logq = flow(eta, phi_c)

    assert d.shape == (batch, n_detail_channels, coarse_size, coarse_size)
    assert logq.shape == (batch,)
    assert torch.isfinite(d).all()
    assert torch.isfinite(logq).all()


def test_make_conditioning_basic_shape_and_finite() -> None:
    phi_c = seeded_randn(1, 8, 8, dtype=torch.float32)

    cond = make_conditioning(phi_c, "basic")

    assert cond.shape == (1, 1, 8, 8)
    assert torch.isfinite(cond).all()
    assert torch.allclose(cond[:, 0], phi_c)


def test_make_conditioning_physics_shape_rolls_and_finite() -> None:
    phi_c = seeded_randn(1, 8, 8, dtype=torch.float32)

    cond = make_conditioning(phi_c, "physics")
    grad_x_expected = 0.5 * (torch.roll(phi_c, -1, dims=-1) - torch.roll(phi_c, 1, dims=-1))
    grad_y_expected = 0.5 * (torch.roll(phi_c, -1, dims=-2) - torch.roll(phi_c, 1, dims=-2))
    lap_expected = (
        torch.roll(phi_c, 1, dims=-1)
        + torch.roll(phi_c, -1, dims=-1)
        + torch.roll(phi_c, 1, dims=-2)
        + torch.roll(phi_c, -1, dims=-2)
        - 4.0 * phi_c
    )

    assert cond.shape == (1, 6, 8, 8)
    assert torch.roll(phi_c, -1, dims=-1).shape == phi_c.shape
    assert torch.roll(phi_c, 1, dims=-2).shape == phi_c.shape
    assert torch.isfinite(cond).all()
    assert torch.allclose(cond[:, 0], phi_c)
    assert torch.allclose(cond[:, 1], phi_c.square())
    assert torch.allclose(cond[:, 2], grad_x_expected)
    assert torch.allclose(cond[:, 3], grad_y_expected)
    assert torch.allclose(cond[:, 4], grad_x_expected.square() + grad_y_expected.square())
    assert torch.allclose(cond[:, 5], lap_expected)


def test_soft_reconstruct_matches_hard_when_rho_zero() -> None:
    phi_f = seeded_randn(1, 16, 16)
    ll, d = average_block(phi_f)
    u = torch.cat((torch.zeros_like(ll).unsqueeze(1), d), dim=1)

    phi_soft = soft_reconstruct(ll, u)
    phi_hard = reconstruct_from_average_block(ll, d)

    assert u.shape == (1, 4, 8, 8)
    assert torch.allclose(phi_soft, phi_hard, atol=1e-12, rtol=1e-12)


def test_soft_block_reconstruct_exact_with_noisy_psi() -> None:
    phi_f = seeded_randn(1, 16, 16)
    gen = torch.Generator().manual_seed(123)
    psi, u = soft_block(phi_f, alpha=2.0, generator=gen)

    phi_rec = soft_reconstruct(psi, u)

    assert psi.shape == (1, 8, 8)
    assert u.shape == (1, 4, 8, 8)
    assert torch.allclose(phi_rec, phi_f, atol=1e-10, rtol=1e-10)


def test_soft_kernel_term_alpha_rho_squared() -> None:
    u = seeded_randn(1, 4, 8, 8)
    alpha = 1.7

    expected = alpha * u[:, 0].square().sum(dim=(-2, -1))

    assert torch.allclose(soft_kernel_term(u, alpha), expected, atol=1e-12, rtol=1e-12)


def test_soft_weighted_block_reconstruct_exact_with_noisy_psi() -> None:
    phi = seeded_randn(1, 16, 16)
    gen = torch.Generator().manual_seed(321)
    a = 0.25
    b = 0.0625
    eta = 0.25

    psi, u = soft_weighted_block(phi, alpha=2.0, a=a, b=b, eta=eta, generator=gen)
    reconstructed = soft_weighted_reconstruct(psi, u, a, b, eta=eta)

    assert psi.shape == (1, 8, 8)
    assert u.shape == (1, 4, 8, 8)
    assert torch.allclose(reconstructed, phi, atol=1e-10, rtol=1e-10)


def test_soft_weighted_constant_blocks_to_eta_scaled_weighted_constant() -> None:
    value = 1.7
    phi = torch.full((1, 16, 16), value, dtype=torch.float64)
    eta = 0.25
    a = 0.25
    b = 0.0625
    expected = (2.0 ** (-eta / 2.0)) * phi[..., 0::2, 0::2]

    psi = weighted_block(phi, a, b, eta=eta)

    assert torch.allclose(psi, expected, atol=1e-12, rtol=1e-12)


def test_soft_weighted_rho_zero_matches_hard_weighted_reconstruction() -> None:
    phi = seeded_randn(1, 16, 16)
    _ll, d = average_block(phi)
    a = 0.25
    b = 0.0625
    eta = 0.25
    psi = weighted_block(phi, a, b, eta=eta)
    u = torch.cat((torch.zeros_like(psi).unsqueeze(1), d), dim=1)

    soft_rec = soft_weighted_reconstruct(psi, u, a, b, eta=eta)
    hard_rec = reconstruct_from_weighted_block(psi, d, a, b, eta=eta)

    assert u.shape == (1, 4, 8, 8)
    assert torch.allclose(soft_rec, hard_rec, atol=1e-12, rtol=1e-12)
    assert torch.allclose(soft_rec, phi, atol=1e-10, rtol=1e-10)


def test_soft_weighted_kernel_term_alpha_rho_squared() -> None:
    u = seeded_randn(1, 4, 8, 8)
    alpha = 1.7

    expected = alpha * u[:, 0].square().sum(dim=(-2, -1))

    assert torch.allclose(soft_weighted_kernel_term(u, alpha), expected, atol=1e-12, rtol=1e-12)


def test_weighted_block_without_eta_scaling_matches_previous_behavior() -> None:
    phi = seeded_randn(1, 16, 16)
    a = 0.18
    b = 0.04

    psi_eta_off = weighted_block(phi, a, b, eta=0.25, use_eta_scaling=False)
    psi_eta_zero = weighted_block(phi, a, b, eta=0.0, use_eta_scaling=True)

    assert torch.allclose(psi_eta_off, psi_eta_zero, atol=1e-12, rtol=1e-12)


def test_weighted_block_constant_gets_eta_scaling() -> None:
    value = 1.7
    phi = torch.full((1, 16, 16), value, dtype=torch.float64)
    eta = 0.25
    expected = (2.0 ** (-eta / 2.0)) * phi[..., 0::2, 0::2]

    psi = weighted_block(phi, a=0.18, b=0.04, eta=eta, use_eta_scaling=True)

    assert torch.allclose(psi, expected, atol=1e-12, rtol=1e-12)


def test_weighted_reconstruct_exact_random_field() -> None:
    phi = seeded_randn(1, 16, 16)
    _ll, d = average_block(phi)
    a = 0.18
    b = 0.04
    eta = 0.25
    psi = weighted_block(phi, a, b, eta=eta, use_eta_scaling=True)

    reconstructed = reconstruct_from_weighted_block(psi, d, a, b, eta=eta, use_eta_scaling=True)

    assert torch.allclose(reconstructed, phi, atol=1e-10, rtol=1e-10)


def test_changing_eta_changes_weighted_psi_but_same_eta_reconstructs() -> None:
    phi = seeded_randn(1, 16, 16)
    _ll, d = average_block(phi)
    a = 0.18
    b = 0.04
    psi_low = weighted_block(phi, a, b, eta=0.125, use_eta_scaling=True)
    psi_high = weighted_block(phi, a, b, eta=0.375, use_eta_scaling=True)

    rec_low = reconstruct_from_weighted_block(psi_low, d, a, b, eta=0.125, use_eta_scaling=True)
    rec_high = reconstruct_from_weighted_block(psi_high, d, a, b, eta=0.375, use_eta_scaling=True)

    assert not torch.allclose(psi_low, psi_high, atol=1e-12, rtol=1e-12)
    assert torch.allclose(rec_low, phi, atol=1e-10, rtol=1e-10)
    assert torch.allclose(rec_high, phi, atol=1e-10, rtol=1e-10)


def test_weighted_fft_inverse_stable_for_learned_kernel_like_ab() -> None:
    min_abs = weighted_ll_fft_min_abs((8, 8), a=0.18, b=0.04)

    assert min_abs.item() > 1e-3


def test_train_conditional_flow_import_smoke() -> None:
    module = importlib.import_module("inverse_blocking_flow.train_conditional_flow")

    assert hasattr(module, "build_parser")


@pytest.mark.slow
def test_flow_forward_and_inverse_logq_conventions_match() -> None:
    batch = 1
    coarse_size = 8
    flow = ConditionalDetailFlow(n_layers=2, hidden_channels=8, depth=2)
    phi_c = seeded_randn(batch, 1, coarse_size, coarse_size, dtype=torch.float32)
    eta = seeded_randn(batch, 3, coarse_size, coarse_size, dtype=torch.float32)

    d, logq_forward = flow.sample_and_logq(eta, phi_c)
    eta_back, logq_reverse = flow.inverse_logq(d, phi_c)

    assert torch.allclose(eta_back, eta, atol=1e-5, rtol=1e-5)
    assert torch.allclose(logq_forward, logq_reverse, atol=1e-5, rtol=1e-5)
