from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def inverse_softplus(x: float) -> float:
    return math.log(math.expm1(float(x)))


def unconstrained_rational_quadratic_spline(
    inputs: torch.Tensor,
    unnormalized_widths: torch.Tensor,
    unnormalized_heights: torch.Tensor,
    unnormalized_derivatives: torch.Tensor,
    *,
    inverse: bool = False,
    tail_bound: float = 6.0,
    min_bin_width: float = 1.0e-3,
    min_bin_height: float = 1.0e-3,
    min_derivative: float = 1.0e-3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Elementwise monotonic rational-quadratic spline with linear tails.

    Widths/heights have shape ``(..., K)``. Derivatives may have shape
    ``(..., K - 1)`` and endpoint derivatives are then fixed to one, or shape
    ``(..., K + 1)`` for fully specified endpoints.
    """

    if unnormalized_widths.shape != unnormalized_heights.shape:
        raise ValueError("width and height parameters must have matching shapes")
    if unnormalized_widths.shape[:-1] != inputs.shape:
        raise ValueError("spline parameter leading shape must match inputs")
    num_bins = int(unnormalized_widths.shape[-1])
    if num_bins < 2:
        raise ValueError("at least two spline bins are required")
    if min_bin_width * num_bins >= 1.0 or min_bin_height * num_bins >= 1.0:
        raise ValueError("minimum bin size is too large for number of bins")

    inside = (inputs >= -tail_bound) & (inputs <= tail_bound)
    outputs = inputs.clone()
    logabsdet = torch.zeros_like(inputs)
    if not torch.any(inside):
        return outputs, logabsdet

    x = inputs[inside]
    widths = unnormalized_widths[inside]
    heights = unnormalized_heights[inside]
    derivs_raw = unnormalized_derivatives[inside]

    widths = F.softmax(widths, dim=-1)
    widths = min_bin_width + (1.0 - min_bin_width * num_bins) * widths
    heights = F.softmax(heights, dim=-1)
    heights = min_bin_height + (1.0 - min_bin_height * num_bins) * heights

    if derivs_raw.shape[-1] == num_bins - 1:
        internal = min_derivative + F.softplus(derivs_raw)
        ones = torch.ones_like(internal[..., :1])
        derivatives = torch.cat([ones, internal, ones], dim=-1)
    elif derivs_raw.shape[-1] == num_bins + 1:
        derivatives = min_derivative + F.softplus(derivs_raw)
    else:
        raise ValueError("derivatives must have K-1 internal or K+1 full entries")

    left = -float(tail_bound)
    interval = 2.0 * float(tail_bound)
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = F.pad(cumwidths, pad=(1, 0), mode="constant", value=0.0)
    cumwidths = left + interval * cumwidths
    cumwidths[..., 0] = left
    cumwidths[..., -1] = float(tail_bound)
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, pad=(1, 0), mode="constant", value=0.0)
    cumheights = left + interval * cumheights
    cumheights[..., 0] = left
    cumheights[..., -1] = float(tail_bound)
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    bin_locations = torch.searchsorted(cumheights if inverse else cumwidths, x[..., None]).squeeze(-1) - 1
    bin_locations = bin_locations.clamp(min=0, max=num_bins - 1)
    gather = bin_locations[..., None]

    input_cumwidths = cumwidths.gather(-1, gather).squeeze(-1)
    input_bin_widths = widths.gather(-1, gather).squeeze(-1)
    input_cumheights = cumheights.gather(-1, gather).squeeze(-1)
    input_heights = heights.gather(-1, gather).squeeze(-1)
    delta = input_heights / input_bin_widths
    deriv_left = derivatives.gather(-1, gather).squeeze(-1)
    deriv_right = derivatives.gather(-1, gather + 1).squeeze(-1)
    common = deriv_left + deriv_right - 2.0 * delta

    if inverse:
        y_minus = x - input_cumheights
        a = y_minus * common + input_heights * (delta - deriv_left)
        b = input_heights * deriv_left - y_minus * common
        c = -delta * y_minus
        discriminant = (b * b - 4.0 * a * c).clamp_min(0.0)
        root = (2.0 * c) / (-b - torch.sqrt(discriminant))
        root = torch.where(torch.abs(a) < 1.0e-12, -c / b, root)
        theta = root.clamp(0.0, 1.0)
        out = input_cumwidths + theta * input_bin_widths
    else:
        theta = (x - input_cumwidths) / input_bin_widths
        theta_one_minus = theta * (1.0 - theta)
        numerator = input_heights * (delta * theta * theta + deriv_left * theta_one_minus)
        denominator = delta + common * theta_one_minus
        out = input_cumheights + numerator / denominator

    theta_one_minus = theta * (1.0 - theta)
    derivative_numerator = delta * delta * (
        deriv_right * theta * theta + 2.0 * delta * theta_one_minus + deriv_left * (1.0 - theta) * (1.0 - theta)
    )
    derivative_denominator = (delta + common * theta_one_minus).pow(2)
    lad = torch.log(derivative_numerator.clamp_min(1.0e-30)) - torch.log(derivative_denominator.clamp_min(1.0e-30))
    if inverse:
        lad = -lad

    outputs[inside] = out
    logabsdet[inside] = lad
    return outputs, logabsdet
