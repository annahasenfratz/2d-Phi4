import numpy as np

from perfect_blocking.scripts.run_stage5b_g22_scan import o22_density


def test_o22_counts_two_forward_bonds_once():
    phi = np.ones((1, 4, 4))
    assert np.allclose(o22_density(phi), [2.0])


def test_o22_is_translation_invariant():
    phi = np.arange(16.0).reshape(1, 4, 4)
    assert np.allclose(o22_density(phi), o22_density(np.roll(phi, 1, axis=1)))
