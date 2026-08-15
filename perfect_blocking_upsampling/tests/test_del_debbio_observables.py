from __future__ import annotations

import numpy as np

from perfect_blocking_upsampling.observables import (
    second_moment_components,
    second_moment_xi_over_L,
    del_debbio_default_window,
    del_debbio_plateau_from_correlator,
    del_debbio_pole_observables,
    periodic_cosh_effective_mass,
)


def test_periodic_cosh_effective_mass_recovers_synthetic_mass() -> None:
    L = 32
    m = 0.37
    t = np.arange(L, dtype=np.float64)
    C = np.exp(-m * t) + np.exp(-m * (L - t))

    m_eff, arg = periodic_cosh_effective_mass(C)

    assert m_eff.shape == (L,)
    assert arg.shape == (L,)
    np.testing.assert_allclose(m_eff[1 : L // 2], m, rtol=1.0e-12, atol=1.0e-12)
    assert np.isnan(m_eff[0])
    assert np.all(np.isnan(m_eff[L // 2 :]))


def test_del_debbio_plateau_uses_default_large_distance_window() -> None:
    L = 32
    m = 0.23
    t = np.arange(L, dtype=np.float64)
    C = 1.7 * (np.exp(-m * t) + np.exp(-m * (L - t)))

    out = del_debbio_plateau_from_correlator(C)

    assert (out["m_eff_window_t_min"], out["m_eff_window_t_max"]) == del_debbio_default_window(L)
    assert out["m_eff_window_valid_points"] == 10
    np.testing.assert_allclose(out["m_pole_DD_plateau"], m, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(out["xi_pole_DD"], 1.0 / m, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(out["xi_over_L_pole_DD"], 1.0 / (L * m), rtol=1.0e-12, atol=1.0e-12)


def test_del_debbio_plateau_ignores_nan_effective_masses() -> None:
    L = 32
    m = 0.31
    t = np.arange(L, dtype=np.float64)
    C = np.exp(-m * t) + np.exp(-m * (L - t))
    C[8] = -1.0

    out = del_debbio_plateau_from_correlator(C, t_min=5, t_max=14)

    assert out["m_eff_bad_arccosh_count"] > 0
    assert 2 <= out["m_eff_window_valid_points"] < 10
    assert np.isfinite(out["m_pole_DD_plateau"])


def test_del_debbio_random_batch_shapes_and_nan_guards() -> None:
    rng = np.random.default_rng(1234)
    phi = rng.standard_normal((7, 8, 8))

    out = del_debbio_pole_observables(phi)

    assert out["C_connected_slice_t"].shape == (8,)
    assert out["C_connected_slice_t_se"].shape == (8,)
    assert out["C_connected_slice_x_t"].shape == (8,)
    assert out["C_connected_slice_y_t"].shape == (8,)
    assert out["C_connected_t"].shape == (8,)
    assert out["C_connected_t_se"].shape == (8,)
    assert out["m_eff_cosh_t"].shape == (8,)
    assert out["m_eff_cosh_t_arccosh_arg"].shape == (8,)
    assert "m_pole_DD_plateau" in out
    assert "xi_pole_DD" in out
    assert "xi_over_L_pole_DD" in out
    assert out["m_eff_window_t_min"] == 2
    assert out["m_eff_window_t_max"] == 2

    C = np.array([0.5, 2.0, 0.5, 0.5], dtype=np.float64)
    m_eff, arg = periodic_cosh_effective_mass(C)
    assert arg[1] < 1.0
    assert np.isnan(m_eff[1])


def test_del_debbio_nonpositive_c0_returns_nan() -> None:
    phi = np.ones((3, 6, 6), dtype=np.float64)

    out = del_debbio_pole_observables(phi)

    assert out["C_connected_t0_nonpositive"] == 1
    assert np.isnan(out["m_pole_DD_plateau"])
    assert np.isnan(out["xi_pole_DD"])
    assert np.isnan(out["xi_over_L_pole_DD"])


def _gaussian_lattice_field_batch(n: int, L: int, mass: float, seed: int = 123) -> np.ndarray:
    rng = np.random.default_rng(seed)
    white = rng.standard_normal((n, L, L))
    white_k = np.fft.fft2(white, axes=(1, 2))
    q = 2.0 * np.pi * np.fft.fftfreq(L)
    qx, qy = np.meshgrid(q, q, indexing="ij")
    khat2 = 4.0 * np.sin(qx / 2.0) ** 2 + 4.0 * np.sin(qy / 2.0) ** 2
    filt = 1.0 / np.sqrt(mass * mass + khat2)
    return np.fft.ifft2(white_k * filt[None, :, :], axes=(1, 2)).real


def test_second_moment_xi_recovers_gaussian_lattice_propagator() -> None:
    L = 16
    mass = 0.6
    phi = _gaussian_lattice_field_batch(6000, L, mass)
    out = second_moment_components(phi)
    expected_xi = 1.0 / mass
    expected_xi_over_L = expected_xi / L

    assert out["xi_2nd_valid"] == 1
    np.testing.assert_allclose(out["xi_2nd"], expected_xi, rtol=0.08, atol=0.0)
    np.testing.assert_allclose(out["xi_over_L_2nd"], expected_xi_over_L, rtol=0.08, atol=0.0)


def test_second_moment_xi_fft_normalization_cancels() -> None:
    rng = np.random.default_rng(5678)
    phi = rng.standard_normal((128, 12, 12))

    xi1 = second_moment_xi_over_L(phi)
    xi2 = second_moment_xi_over_L(3.7 * phi)

    np.testing.assert_allclose(xi1, xi2, rtol=1.0e-12, atol=1.0e-12)


def test_second_moment_xi_constant_field_is_invalid() -> None:
    phi = np.ones((10, 8, 8), dtype=np.float64)

    out = second_moment_components(phi)

    assert out["xi_2nd_valid"] == 0
    assert out["xi_2nd_G0_positive"] == 0
    assert out["xi_2nd_Gpmin_positive"] == 0
    assert np.isnan(out["xi_over_L_2nd"])


def test_second_moment_xi_uncorrelated_field_is_near_zero() -> None:
    rng = np.random.default_rng(4321)
    phi = rng.standard_normal((5000, 16, 16))

    out = second_moment_components(phi)

    assert out["xi_2nd_valid"] == 1
    assert out["xi_over_L_2nd"] < 0.04


def test_second_moment_xi_axis_swap_invariant() -> None:
    rng = np.random.default_rng(2468)
    phi = rng.standard_normal((512, 10, 10))
    phi += 0.2 * np.roll(phi, 1, axis=1)

    xi = second_moment_xi_over_L(phi)
    xi_swapped = second_moment_xi_over_L(np.swapaxes(phi, 1, 2))

    np.testing.assert_allclose(xi, xi_swapped, rtol=1.0e-12, atol=1.0e-12)
