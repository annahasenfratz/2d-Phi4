#!/usr/bin/env python3
"""Non-promoting L32->L16 kernel search with joint operator constraints.

This reuses the established Pareto search but augments every candidate with
configuration-level composite observables.  Since the individual factors are
also guarded, matching each composite constrains its connected covariance.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
BASE_PATH = HERE.with_name("run_lam1p0_redo_kernel_phi2_phi4_pareto_2000.py")
spec = importlib.util.spec_from_file_location("base_kernel_search", BASE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load {BASE_PATH}")
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)

TAG = "joint_kurtosis_correlations_5000"
base.OUT = base.LAM_ROOT / f"tests/intermediate/{TAG}"
base.CAND_DIR = base.LAM_ROOT / f"kernels/candidates/{TAG}"
base.FINAL = base.LAM_ROOT / f"tests/final/{TAG}"
base.SEARCH_N_CONFIGS = 5000
base.SUBSET_N = 2500

JOINT_OBS = (
    "phi2_x_NN",
    "phi2_x_phi4",
    "phi2_x_local_kurtosis",
    "NN_x_diag",
    "action_x_local_kurtosis",
)
base.OBS = list(base.OBS) + list(JOINT_OBS)


def with_joint_observables(values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = dict(values)
    out["phi2_x_NN"] = out["phi2"] * out["NN"]
    out["phi2_x_phi4"] = out["phi2"] * out["phi4"]
    out["phi2_x_local_kurtosis"] = out["phi2"] * out["local_kurtosis_ratio"]
    out["NN_x_diag"] = out["NN"] * out["diag"]
    out["action_x_local_kurtosis"] = out["action_density"] * out["local_kurtosis_ratio"]
    return out


original_full_metrics = base.full_metrics


def full_metrics_with_joints(direct: dict[str, np.ndarray], blocked: dict[str, np.ndarray], bins: int = 70):
    return original_full_metrics(with_joint_observables(direct), with_joint_observables(blocked), bins=bins)


base.full_metrics = full_metrics_with_joints
original_objective = base.objective


def objective_with_joints(rows, mom, classes, radius, selected):
    value = original_objective(rows, mom, classes, radius, selected)
    # Products retain their distributional information; because their factors
    # are separately constrained, this also penalizes connected-correlation
    # mismatch rather than merely a shifted product mean.
    weights = {
        "phi2_x_NN": 18.0,
        "phi2_x_phi4": 16.0,
        "phi2_x_local_kurtosis": 24.0,
        "NN_x_diag": 10.0,
        "action_x_local_kurtosis": 20.0,
    }
    return float(value + sum(weight * base.scalar_score(rows[name]) for name, weight in weights.items()))


base.objective = objective_with_joints

if __name__ == "__main__":
    raise SystemExit(base.main())
