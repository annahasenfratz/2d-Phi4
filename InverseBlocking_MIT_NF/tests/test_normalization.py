import torch

from invblock_mit_nf.actions import Phi4Action, Phi4Params, MITPhi4Action, rescale_ours_to_mit


def test_mit_tutorial_point_conversion():
    params = Phi4Params.from_mit(M2=-4.0, lam_mit=8.0)
    assert abs(params.kappa - 0.25) < 1e-12
    assert abs(params.lam - 0.5) < 1e-12
    assert Phi4Params(kappa=0.25, lam=0.5).to_mit() == (-4.0, 8.0)


def test_actions_match_up_to_constant():
    params = Phi4Params(kappa=0.25, lam=0.5)
    M2, lam_mit = params.to_mit()
    x = torch.randn(3, 8, 8)
    S_ours = Phi4Action(params)(x)
    S_mit = MITPhi4Action(M2, lam_mit)(rescale_ours_to_mit(x, params.kappa))
    assert torch.allclose(S_mit, S_ours - params.lam * 8 * 8, atol=1e-5, rtol=1e-5)
