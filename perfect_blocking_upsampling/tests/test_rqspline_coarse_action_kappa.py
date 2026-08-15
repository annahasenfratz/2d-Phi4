from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "perfect_blocking_upsampling" / "scripts" / "run_lam1p0_rqspline_patchwise.py"
sys.path.insert(0, str(ROOT / "perfect_blocking_upsampling" / "src"))
spec = importlib.util.spec_from_file_location("run_lam1p0_rqspline_patchwise", SRC)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_coarse_stage_uses_coarse_kappa_not_fine_kappa() -> None:
    current = np.asarray([[[0.7, -0.2], [0.1, 0.4]]], dtype=np.float32)
    proposed = current.copy()
    proposed[0, 0, 0] += 0.23
    coarse_action = module.ActionSpec("phi4_nn", 1.0, 0.19)
    fine_action = module.ActionSpec("phi4_nn", 1.0, 0.47)
    current_sc = module.action_total(current, coarse_action).astype(np.float64)

    log_stage1, delta_sc, proposed_sc = module.coarse_action_log_acceptance(
        current, proposed, current_sc, coarse_action
    )
    expected_delta = module.action_total(proposed, coarse_action) - module.action_total(current, coarse_action)
    fine_delta = module.action_total(proposed, fine_action) - module.action_total(current, fine_action)

    np.testing.assert_allclose(delta_sc, expected_delta)
    np.testing.assert_allclose(log_stage1, -expected_delta)
    np.testing.assert_allclose(proposed_sc, module.action_total(proposed, coarse_action))
    assert not np.allclose(expected_delta, fine_delta)
