from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_shape_parametric_sampler_validation.py"


def load_validation_module():
    spec = importlib.util.spec_from_file_location("run_shape_parametric_sampler_validation", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_rejected_update_repeats_previous_markov_state_observable():
    mod = load_validation_module()
    state = {"phi": np.zeros((1, 4, 4), dtype=np.float32)}
    proposal = {"phi": np.ones((1, 4, 4), dtype=np.float32)}

    next_state, accepted = mod.apply_ar_update(state, proposal, delta_logw=-100.0, log_uniform=0.0)
    measured = mod.measured_observable_row(
        next_state,
        chain=0,
        sweep=0,
        move_type="coarse",
        attempt_in_sweep=0,
        update_index=0,
        accepted=accepted,
    )

    assert not accepted
    assert next_state is state
    assert measured["measurement_semantics"] == "post_ar_markov_state"
    assert measured["m"] == mod.local_observables(state["phi"])["m"]


def test_accepted_update_measures_proposed_markov_state_observable():
    mod = load_validation_module()
    state = {"phi": np.zeros((1, 4, 4), dtype=np.float32)}
    proposal = {"phi": np.ones((1, 4, 4), dtype=np.float32)}

    next_state, accepted = mod.apply_ar_update(state, proposal, delta_logw=0.0, log_uniform=-1.0)
    measured = mod.measured_observable_row(
        next_state,
        chain=0,
        sweep=0,
        move_type="coarse",
        attempt_in_sweep=1,
        update_index=1,
        accepted=accepted,
    )

    assert accepted
    assert next_state is proposal
    assert measured["measurement_semantics"] == "post_ar_markov_state"
    assert measured["m"] == mod.local_observables(proposal["phi"])["m"]
    assert measured["m"] != mod.local_observables(state["phi"])["m"]


def test_expected_observable_rows_for_measurement_modes():
    mod = load_validation_module()
    cfg = mod.ValidationConfig(validation_chains=8, smoke_sweeps=500, pcn_interval_sweeps=20)
    cfg.measurement_mode = "end_of_sweep"
    assert mod.expected_observable_measurements(cfg, n_patch_per_sweep=8) == 4000

    cfg.measurement_mode = "every_attempt"
    assert mod.expected_observable_measurements(cfg, n_patch_per_sweep=8) == 32200


def test_default_detail_warmup_preserves_old_observable_count():
    mod = load_validation_module()
    cfg = mod.ValidationConfig(validation_chains=8, smoke_sweeps=2000, pcn_interval_sweeps=1)
    assert cfg.detail_warmup_sweeps == 0
    assert mod.expected_observable_measurements(cfg, n_patch_per_sweep=8) == 16000
    assert mod.expected_warmup_observable_measurements(cfg) == 0


def test_fixed_coarse_detail_warmup_keeps_u_and_uses_separate_measurements(monkeypatch):
    mod = load_validation_module()
    cfg = mod.ValidationConfig(
        validation_chains=1,
        smoke_sweeps=1,
        detail_warmup_sweeps=2,
        detail_warmup_fixed_coarse=True,
        detail_warmup_pcn_rho=0.5,
        measure_during_detail_warmup=True,
    )
    state = {
        "u": np.ones((1, 2, 2), dtype=np.float32),
        "phi": np.zeros((1, 4, 4), dtype=np.float32),
        "logw": np.asarray([0.0]),
        "sf": np.asarray([1.0]),
        "sc": np.asarray([2.0]),
        "logdet": np.asarray([3.0]),
        "logq": np.asarray([4.0]),
        "inv": {"max_inverse_ifft_imag": 0.0},
    }

    calls = {"n": 0}

    def fake_schedule(*_args, **_kwargs):
        return [(0, 0, "patch_0")]

    def fake_propose_latent(current, x0, y0, tile, rng, ctx, warm_cfg, rho=None):
        calls["n"] += 1
        proposal = dict(current)
        proposal["u"] = current["u"].copy()
        proposal["phi"] = current["phi"] + calls["n"]
        proposal["logw"] = np.asarray([float(calls["n"])])
        proposal["sf"] = current["sf"] + 1.0
        proposal["sc"] = current["sc"].copy()
        proposal["logdet"] = current["logdet"].copy()
        proposal["logq"] = current["logq"] + 1.0
        return proposal, {
            "patch_x": x0,
            "patch_y": y0,
            "tile": tile,
            "delta_logw": float(proposal["logw"][0] - current["logw"][0]),
            "delta_Sf": float(proposal["sf"][0] - current["sf"][0]),
            "delta_Sc": 0.0,
            "delta_logdet_refine": 0.0,
            "delta_logq_missing": float(proposal["logq"][0] - current["logq"][0]),
            "changed_fine_sites_gt_1e-3": 16,
        }

    monkeypatch.setattr(mod, "random_origin_patch_schedule", fake_schedule)
    monkeypatch.setattr(mod, "propose_latent", fake_propose_latent)
    monkeypatch.setattr(mod, "reblocking_error", lambda *_args, **_kwargs: 0.0)

    warmup_rows = []
    warmup_obs_rows = []
    warmup_check_rows = []
    out = mod.run_fixed_coarse_detail_warmup(
        state,
        chain=0,
        rng=np.random.default_rng(1),
        ctx={},
        cfg=cfg,
        warmup_rows=warmup_rows,
        warmup_obs_rows=warmup_obs_rows,
        warmup_check_rows=warmup_check_rows,
    )

    assert np.array_equal(out["u"], state["u"])
    assert len(warmup_rows) == 2
    assert len(warmup_obs_rows) == 2
    assert all(r["move_type"] == "detail_warmup_latent" for r in warmup_rows)
    assert all(float(r["delta_Sc"]) == 0.0 for r in warmup_rows)
    assert all(float(r["delta_logdet_refine"]) == 0.0 for r in warmup_rows)
    assert all(float(r["u_max_abs_delta_from_initial"]) == 0.0 for r in warmup_rows)
