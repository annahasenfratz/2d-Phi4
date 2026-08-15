from __future__ import annotations

import numpy as np

from perfect_blocking_upsampling.kernels import apply_kernel, inverse_kernel, kernel_stencil_from_spec, load_kernel, normalize_kernel


def test_small3_kernel_normalizes_and_roundtrips():
    spec, _ = load_kernel("configs/kernels/small3_lam0p022_kappa0p2705_eta0p25.json")
    stencil = normalize_kernel(kernel_stencil_from_spec(spec))
    assert np.isclose(stencil.sum(), 1.0)
    rng = np.random.default_rng(0)
    phi = rng.normal(size=(4, 16, 16)).astype(np.float32)
    psi = apply_kernel(phi, spec)
    roundtrip, info = inverse_kernel(psi, spec)
    assert info["condition_number_abs"] > 0
    assert np.max(np.abs(roundtrip - phi)) < 1e-6
