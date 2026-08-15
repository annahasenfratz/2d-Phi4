from __future__ import annotations

import numpy as np

from perfect_blocking_upsampling.gathered_edge import (
    build_gathered_edge_flow,
    square_stencil,
    validate_periodic_offsets,
)


def test_gathered_edge_radius_report_and_l16_dummy():
    model = build_gathered_edge_flow(cond_channels=1, lattice_size=16, radius=3, stencil="square")
    report = model.dependency_report()
    assert report["coarse_radius"] == 3
    assert report["fine_radius"] == 6
    assert report["fine_radius"] < 16
    assert report["n_offsets"] == 49


def test_periodic_offset_validation_rejects_out_of_radius():
    offsets = square_stencil(3).offsets
    assert validate_periodic_offsets(offsets, radius=3, lattice_size=8, metric="chebyshev") == offsets
    try:
        validate_periodic_offsets([(4, 0)], radius=3, lattice_size=8, metric="chebyshev")
    except ValueError as exc:
        assert "outside declared radius" in str(exc)
    else:
        raise AssertionError("expected out-of-radius periodic offset to fail")


def test_gathered_edge_forward_inverse_roundtrip():
    import torch

    torch.manual_seed(123)
    model = build_gathered_edge_flow(cond_channels=1, lattice_size=8, radius=2, stencil="square")
    z = torch.randn(5, 64)
    c = torch.randn(5, 64)
    x, logdet_f = model.forward(z, c)
    z_rt, logdet_i = model.inverse(x, c)
    np.testing.assert_allclose(z_rt.detach().numpy(), z.detach().numpy(), atol=1.0e-6)
    np.testing.assert_allclose((logdet_f + logdet_i).detach().numpy(), np.zeros(5), atol=1.0e-6)
