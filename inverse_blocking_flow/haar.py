"""Explicit average blocking, constant prolongation, and residual details."""

from __future__ import annotations

import torch


Tensor = torch.Tensor


def eta_scaling_factor(
    eta: float | Tensor = 0.25,
    *,
    dimension: float = 2.0,
    block_factor: float = 2.0,
    use_eta_scaling: bool = True,
) -> Tensor:
    """Return ``Z_eta = b^{-Delta_phi}`` for anomalous field scaling."""

    eta_tensor = torch.as_tensor(eta)
    if not use_eta_scaling:
        return torch.ones((), dtype=eta_tensor.dtype, device=eta_tensor.device)
    delta_phi = 0.5 * (dimension - 2.0 + eta_tensor)
    base = torch.as_tensor(block_factor, dtype=eta_tensor.dtype, device=eta_tensor.device)
    return base.pow(-delta_phi)


def block_average(phi_f: Tensor) -> Tensor:
    """Average each non-overlapping 2x2 fine block.

    ``phi_f`` has trailing shape ``(2L, 2L)`` and the returned coarse field has
    trailing shape ``(L, L)``. Leading batch dimensions are preserved.
    """

    if phi_f.shape[-2] % 2 != 0 or phi_f.shape[-1] % 2 != 0:
        raise ValueError("fine lattice dimensions must be even")
    return (
        phi_f[..., 0::2, 0::2]
        + phi_f[..., 1::2, 0::2]
        + phi_f[..., 0::2, 1::2]
        + phi_f[..., 1::2, 1::2]
    ) / 4.0


def prolong_constant(phi_c: Tensor) -> Tensor:
    """Copy each coarse value to its corresponding 2x2 fine block."""

    return phi_c.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)


def detail_to_residual(d: Tensor) -> Tensor:
    """Map three detail channels to a zero-block-average fine residual."""

    if d.shape[-3] != 3:
        raise ValueError("detail field must have three channels")
    d1, d2, d3 = d.unbind(dim=-3)
    chi00 = (d1 + d2 + d3) / 2.0
    chi10 = (-d1 + d2 - d3) / 2.0
    chi01 = (d1 - d2 - d3) / 2.0
    chi11 = (-d1 - d2 + d3) / 2.0
    chi = torch.empty(
        (*d.shape[:-3], 2 * d.shape[-2], 2 * d.shape[-1]),
        dtype=d.dtype,
        device=d.device,
    )
    chi[..., 0::2, 0::2] = chi00
    chi[..., 1::2, 0::2] = chi10
    chi[..., 0::2, 1::2] = chi01
    chi[..., 1::2, 1::2] = chi11
    return chi


def residual_to_detail(chi: Tensor) -> Tensor:
    """Invert ``detail_to_residual`` for zero-block-average residuals."""

    if chi.shape[-2] % 2 != 0 or chi.shape[-1] % 2 != 0:
        raise ValueError("residual lattice dimensions must be even")
    chi00 = chi[..., 0::2, 0::2]
    chi10 = chi[..., 1::2, 0::2]
    chi01 = chi[..., 0::2, 1::2]
    chi11 = chi[..., 1::2, 1::2]
    d1 = chi00 + chi01
    d2 = chi00 + chi10
    d3 = chi00 + chi11
    return torch.stack((d1, d2, d3), dim=-3)


def reconstruct_from_average_block(phi_c: Tensor, d: Tensor) -> Tensor:
    """Reconstruct a fine field as constant prolongation plus residual."""

    if phi_c.shape[-2:] != d.shape[-2:]:
        raise ValueError("coarse and detail lattice shapes must match")
    return prolong_constant(phi_c) + detail_to_residual(d)


def average_block(phi_f: Tensor) -> tuple[Tensor, Tensor]:
    """Return ``(phi_c, d_true)`` for a fine field using residual details."""

    phi_c = block_average(phi_f)
    chi = phi_f - prolong_constant(phi_c)
    return phi_c, residual_to_detail(chi)


def weighted_kernel_normalization(a: float | Tensor, b: float | Tensor) -> Tensor:
    """Return ``N = 1 / (1 + 4a + 4b)`` for the 3x3 weighted blocker."""

    a_tensor = torch.as_tensor(a)
    b_tensor = torch.as_tensor(b, dtype=a_tensor.dtype, device=a_tensor.device)
    return (1.0 + 4.0 * a_tensor + 4.0 * b_tensor).reciprocal()


def _weighted_sum_3x3(phi_f: Tensor, a: float | Tensor, b: float | Tensor) -> Tensor:
    if phi_f.shape[-2] % 2 != 0 or phi_f.shape[-1] % 2 != 0:
        raise ValueError("fine lattice dimensions must be even")
    a_tensor = torch.as_tensor(a, dtype=phi_f.dtype, device=phi_f.device)
    b_tensor = torch.as_tensor(b, dtype=phi_f.dtype, device=phi_f.device)
    total = phi_f
    total = total + a_tensor * (
        torch.roll(phi_f, shifts=(1, 0), dims=(-2, -1))
        + torch.roll(phi_f, shifts=(-1, 0), dims=(-2, -1))
        + torch.roll(phi_f, shifts=(0, 1), dims=(-2, -1))
        + torch.roll(phi_f, shifts=(0, -1), dims=(-2, -1))
    )
    total = total + b_tensor * (
        torch.roll(phi_f, shifts=(1, 1), dims=(-2, -1))
        + torch.roll(phi_f, shifts=(1, -1), dims=(-2, -1))
        + torch.roll(phi_f, shifts=(-1, 1), dims=(-2, -1))
        + torch.roll(phi_f, shifts=(-1, -1), dims=(-2, -1))
    )
    return total[..., 0::2, 0::2]


def weighted_block(
    phi_f: Tensor,
    a: float | Tensor,
    b: float | Tensor,
    *,
    eta: float | Tensor = 0.25,
    dimension: float = 2.0,
    use_eta_scaling: bool = True,
) -> Tensor:
    """Block by a normalized 3x3 weighted average and optional ``Z_eta``."""

    n = weighted_kernel_normalization(
        torch.as_tensor(a, dtype=phi_f.dtype, device=phi_f.device),
        torch.as_tensor(b, dtype=phi_f.dtype, device=phi_f.device),
    )
    z_eta = eta_scaling_factor(
        torch.as_tensor(eta, dtype=phi_f.dtype, device=phi_f.device),
        dimension=dimension,
        use_eta_scaling=use_eta_scaling,
    )
    return z_eta * n * _weighted_sum_3x3(phi_f, a, b)


def _weighted_ll_operator(ll: Tensor, a: float | Tensor, b: float | Tensor) -> Tensor:
    zeros = torch.zeros((*ll.shape[:-2], 3, ll.shape[-2], ll.shape[-1]), dtype=ll.dtype, device=ll.device)
    return weighted_block(
        reconstruct_from_average_block(ll, zeros),
        a,
        b,
        use_eta_scaling=False,
    )


def _weighted_detail_contribution(d: Tensor, a: float | Tensor, b: float | Tensor) -> Tensor:
    ll = torch.zeros((*d.shape[:-3], d.shape[-2], d.shape[-1]), dtype=d.dtype, device=d.device)
    return weighted_block(
        reconstruct_from_average_block(ll, d),
        a,
        b,
        use_eta_scaling=False,
    )


def _solve_weighted_ll(rhs: Tensor, a: float | Tensor, b: float | Tensor, eps: float = 1e-12) -> Tensor:
    impulse = torch.zeros(rhs.shape[-2:], dtype=rhs.dtype, device=rhs.device)
    impulse[0, 0] = 1.0
    response = _weighted_ll_operator(impulse, a, b)
    eigenvalues = torch.fft.fftn(response, dim=(-2, -1))
    if eigenvalues.abs().min().item() <= eps:
        raise ValueError("weighted LL operator is singular or ill-conditioned")
    rhs_fft = torch.fft.fftn(rhs, dim=(-2, -1))
    return torch.fft.ifftn(rhs_fft / eigenvalues, dim=(-2, -1)).real


def weighted_ll_fft_min_abs(
    coarse_shape: tuple[int, int],
    a: float | Tensor,
    b: float | Tensor,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> Tensor:
    """Return the minimum absolute FFT eigenvalue of ``A_w``."""

    impulse = torch.zeros(coarse_shape, dtype=dtype, device=device)
    impulse[0, 0] = 1.0
    response = _weighted_ll_operator(impulse, a, b)
    return torch.fft.fftn(response, dim=(-2, -1)).abs().min()


def weighted_ll_fft_stats(
    coarse_shape: tuple[int, int],
    a: float | Tensor,
    b: float | Tensor,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> dict[str, float]:
    """Return simple Fourier-spectrum diagnostics for the weighted ``A_w``."""

    impulse = torch.zeros(coarse_shape, dtype=dtype, device=device)
    impulse[0, 0] = 1.0
    response = _weighted_ll_operator(impulse, a, b)
    spectrum = torch.fft.fftn(response, dim=(-2, -1))
    real = spectrum.real
    absval = spectrum.abs()
    return {
        "min_real": float(real.min().detach().cpu().item()),
        "max_real": float(real.max().detach().cpu().item()),
        "min_abs": float(absval.min().detach().cpu().item()),
        "max_abs": float(absval.max().detach().cpu().item()),
    }


def reconstruct_from_weighted_block(
    psi: Tensor,
    d: Tensor,
    a: float | Tensor,
    b: float | Tensor,
    *,
    eta: float | Tensor = 0.25,
    dimension: float = 2.0,
    use_eta_scaling: bool = True,
) -> Tensor:
    """Invert ``weighted_block`` at fixed Haar detail channels.

    The weighted field obeys ``psi = Z_eta * (A_w LL + C_w d)``.  This routine
    solves ``LL = A_w^{-1}(psi / Z_eta - C_w d)`` by FFT on the periodic coarse
    lattice, then returns ``haar_unblock(LL, d)``.
    """

    if psi.shape[-2:] != d.shape[-2:]:
        raise ValueError("weighted coarse field and detail lattice shapes must match")
    z_eta = eta_scaling_factor(
        torch.as_tensor(eta, dtype=psi.dtype, device=psi.device),
        dimension=dimension,
        use_eta_scaling=use_eta_scaling,
    )
    rhs = psi / z_eta - _weighted_detail_contribution(d, a, b)
    ll = _solve_weighted_ll(rhs, a, b)
    return reconstruct_from_average_block(ll, d)


def soft_weighted_block(
    phi_f: Tensor,
    alpha: float,
    a: float | Tensor,
    b: float | Tensor,
    *,
    eta: float | Tensor = 0.25,
    dimension: float = 2.0,
    use_eta_scaling: bool = True,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Return ``(psi, u)`` for the soft weighted blocking kernel.

    ``u`` has four channels ``(rho, HL, LH, HH)`` with
    ``rho = m - psi`` and ``m = Z_eta * B_w(phi_f)``.
    """

    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    m = weighted_block(
        phi_f,
        a,
        b,
        eta=eta,
        dimension=dimension,
        use_eta_scaling=use_eta_scaling,
    )
    _ll, detail = average_block(phi_f)
    sigma = (1.0 / (2.0 * alpha)) ** 0.5
    noise = sigma * torch.randn(m.shape, dtype=m.dtype, device=m.device, generator=generator)
    psi = m + noise
    rho = m - psi
    u = torch.cat((rho.unsqueeze(-3), detail), dim=-3)
    return psi, u


def soft_weighted_reconstruct(
    psi: Tensor,
    u: Tensor,
    a: float | Tensor,
    b: float | Tensor,
    *,
    eta: float | Tensor = 0.25,
    dimension: float = 2.0,
    use_eta_scaling: bool = True,
) -> Tensor:
    """Reconstruct a fine field from soft weighted ``psi`` and ``u``."""

    if u.shape[-3] != 4:
        raise ValueError("soft weighted detail field u must have four channels")
    rho = u[..., 0, :, :]
    detail = u[..., 1:, :, :]
    m = psi + rho
    return reconstruct_from_weighted_block(
        m,
        detail,
        a,
        b,
        eta=eta,
        dimension=dimension,
        use_eta_scaling=use_eta_scaling,
    )


def soft_weighted_kernel_term(u: Tensor, alpha: float) -> Tensor:
    if u.shape[-3] != 4:
        raise ValueError("soft weighted detail field u must have four channels")
    rho = u[..., 0, :, :]
    return alpha * rho.square().sum(dim=(-2, -1))


def soft_block(
    phi_f: Tensor,
    alpha: float,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Return ``(psi, u)`` for the soft RG blocking kernel.

    ``u`` has four channels ``(rho, HL, LH, HH)`` with
    ``rho = LL - psi`` and ``psi = LL + N(0, 1/(2 alpha))``.
    """

    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    ll, detail = average_block(phi_f)
    sigma = (1.0 / (2.0 * alpha)) ** 0.5
    noise = sigma * torch.randn(ll.shape, dtype=ll.dtype, device=ll.device, generator=generator)
    psi = ll + noise
    rho = ll - psi
    u = torch.cat((rho.unsqueeze(-3), detail), dim=-3)
    return psi, u


def soft_reconstruct(psi: Tensor, u: Tensor) -> Tensor:
    """Reconstruct a fine field from soft blocked field and four-channel ``u``."""

    if u.shape[-3] != 4:
        raise ValueError("soft detail field u must have four channels")
    rho = u[..., 0, :, :]
    detail = u[..., 1:, :, :]
    ll_rec = psi + rho
    return reconstruct_from_average_block(ll_rec, detail)


def soft_kernel_term(u: Tensor, alpha: float) -> Tensor:
    if u.shape[-3] != 4:
        raise ValueError("soft detail field u must have four channels")
    rho = u[..., 0, :, :]
    return alpha * rho.square().sum(dim=(-2, -1))


def soft_conditional_action(psi: Tensor, u: Tensor, alpha: float, params) -> Tensor:
    """Return ``S_f(soft_reconstruct(psi,u)) + alpha * sum rho^2``."""

    from inverse_blocking_flow.phi4 import phi4_action

    return phi4_action(soft_reconstruct(psi, u), params) + soft_kernel_term(u, alpha)


def haar_block(phi_f: Tensor) -> tuple[Tensor, Tensor]:
    """Compatibility alias for ``average_block``.

    Older prototype code used the names ``haar_block`` and ``haar_unblock``.
    They now use the explicit average/prolong/residual parameterization.
    """

    return average_block(phi_f)


def haar_unblock(phi_c: Tensor, d: Tensor) -> Tensor:
    """Compatibility alias for ``reconstruct_from_average_block``."""

    return reconstruct_from_average_block(phi_c, d)
