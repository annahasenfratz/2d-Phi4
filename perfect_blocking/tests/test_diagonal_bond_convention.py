import numpy as np

from perfect_blocking.scripts.run_stage4a_kappa_diag_scan import diagonal_two_orientation_density


def test_two_forward_diagonal_orientations_count_two_bonds_per_site():
    phi = np.ones((1, 4, 4))
    assert np.allclose(diagonal_two_orientation_density(phi), [1.0])


def test_diagonal_operator_is_translation_invariant():
    phi = np.arange(16.0).reshape(1, 4, 4)
    shifted = np.roll(phi, 1, axis=1)
    assert np.allclose(diagonal_two_orientation_density(phi), diagonal_two_orientation_density(shifted))
