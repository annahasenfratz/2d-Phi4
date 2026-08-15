from __future__ import annotations

import numpy as np

from perfect_blocking_upsampling.actions import action_total
from perfect_blocking_upsampling.io import ActionSpec


def test_coarse_and_fine_actions_are_distinct_inputs():
    rng = np.random.default_rng(1)
    phi = rng.normal(size=(2, 8, 8)).astype(np.float32)
    coarse = ActionSpec(type="phi4_nn", lambda_=0.022, kappa=0.258)
    fine = ActionSpec(type="phi4_nn", lambda_=0.022, kappa=0.265)
    s_c = action_total(phi, coarse)
    s_f = action_total(phi, fine)
    assert not np.allclose(s_c, s_f)


def test_phi4_nn_plus_diag_uses_diag_coupling():
    rng = np.random.default_rng(2)
    phi = rng.normal(size=(2, 8, 8)).astype(np.float32)
    base = ActionSpec(type="phi4_nn", lambda_=0.022, kappa=0.2705)
    diag = ActionSpec(type="phi4_nn_plus_diag", lambda_=0.022, kappa=0.2705, kappa_diag=0.005)
    assert not np.allclose(action_total(phi, base), action_total(phi, diag))

