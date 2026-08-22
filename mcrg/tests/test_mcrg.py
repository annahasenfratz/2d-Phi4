import numpy as np
from mcrg.blocking import average_block, perfect_block, perfect_kernel, production_block_configs
from mcrg.operators import EVEN_OPERATORS, ODD_OPERATORS, measure
from mcrg.rg import connected_covariance, solve_rg


def test_periodic_operator_sums_and_parity():
    p = np.arange(16., dtype=float).reshape(1,4,4)
    assert measure(p, ["E3_nn"])[0,0] == sum(p[0,i,j]*(p[0,(i+1)%4,j]+p[0,i,(j+1)%4]) for i in range(4) for j in range(4))
    for op in EVEN_OPERATORS: assert np.allclose(op.evaluate(p), op.evaluate(-p))
    for op in ODD_OPERATORS: assert np.allclose(op.evaluate(-p), -op.evaluate(p))


def test_average_block_and_shape():
    p = np.arange(16., dtype=float).reshape(1,4,4)
    assert np.allclose(average_block(p, "literal"), [[[2.5,4.5],[10.5,12.5]]])
    assert average_block(p).shape == (1,2,2)


def test_covariance_and_known_linear_solve():
    rng = np.random.default_rng(2); x = rng.normal(size=(500,3)); a = np.array([[3.,.2,0.],[.2,2.,.1],[0.,.1,1.]])
    coarse = x @ a.T; target = np.array([[1.2,.3,0.],[-.1,.8,.2],[0.,.1,1.1]])
    # Construct fine so the empirical B is exactly A target.
    fine = coarse @ target
    r = solve_rg(fine, coarse, 1e-13)
    assert np.allclose(r.T, target, atol=1e-10)
    assert np.allclose(connected_covariance(x), connected_covariance(x,x))


def test_common_basis_rescaling_preserves_eigenvalues():
    rng = np.random.default_rng(3); c = rng.normal(size=(200,3)); f = c @ np.array([[1,.2,0],[0,.8,.1],[.1,0,1.1]])
    r = solve_rg(f, c); rs = solve_rg(f, c, scales=np.array([.2,5.,2.]))
    assert np.allclose(np.sort(r.eigenvalues), np.sort(rs.eigenvalues))


def test_perfect_adapter_agrees_with_production():
    p = np.random.default_rng(4).normal(size=(2,8,8))
    assert np.allclose(perfect_block(p), production_block_configs(p, perfect_kernel()))
