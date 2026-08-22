"""Extensive, translationally invariant operator registry for 2D phi4."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class Operator:
    name: str
    parity: str
    evaluate: Callable[[Array], Array]


def _sum(phi: Array) -> Array:
    return np.asarray(phi, dtype=np.float64).sum(axis=(-2, -1))


def _nn(phi: Array) -> Array:
    return _sum(phi * np.roll(phi, -1, axis=-2) + phi * np.roll(phi, -1, axis=-1))


def _diag(phi: Array) -> Array:
    return _sum(phi * np.roll(np.roll(phi, -1, axis=-2), -1, axis=-1))


def _axial2(phi: Array) -> Array:
    return _sum(phi * np.roll(phi, -2, axis=-2) + phi * np.roll(phi, -2, axis=-1))


def _nn_mixed(phi: Array) -> Array:
    x = np.roll(phi, -1, axis=-2)
    y = np.roll(phi, -1, axis=-1)
    return _sum(phi**2 * x + phi * x**2 + phi**2 * y + phi * y**2)


def _bilinear_shell(phi: Array, offsets: tuple[tuple[int, int], ...]) -> Array:
    """D4-symmetrized scalar bilinear; offsets are chosen modulo reversal."""
    return sum((_sum(phi * np.roll(np.roll(phi, -dx, axis=-2), -dy, axis=-1)) for dx, dy in offsets), np.zeros(phi.shape[0]))


def _nn_cubic(phi: Array) -> Array:
    x, y = np.roll(phi, -1, axis=-2), np.roll(phi, -1, axis=-1)
    return _sum(phi**3*x + phi*x**3 + phi**3*y + phi*y**3)


def _phi2_gradphi2(phi: Array) -> Array:
    x, y = np.roll(phi, -1, axis=-2), np.roll(phi, -1, axis=-1)
    return _sum((phi-x)**2*(phi**2+x**2) + (phi-y)**2*(phi**2+y**2))


def _laplacian2(phi: Array) -> Array:
    lap = np.roll(phi, 1, axis=-2)+np.roll(phi, -1, axis=-2)+np.roll(phi, 1, axis=-1)+np.roll(phi, -1, axis=-1)-4*phi
    return _sum(lap**2)


EVEN_OPERATORS = (
    Operator("E1_phi2", "even", lambda p: _sum(p**2)),
    Operator("E2_phi4", "even", lambda p: _sum(p**4)),
    Operator("E3_nn", "even", _nn),
    Operator("E4_diag", "even", _diag),
    Operator("E5_axial2", "even", _axial2),
    Operator("E6_phi6", "even", lambda p: _sum(p**6)),
    Operator("E7_nn_phi2phi2", "even", lambda p: _sum(p**2 * np.roll(p, -1, axis=-2)**2 + p**2 * np.roll(p, -1, axis=-1)**2)),
    # Systematic scalar-even extensions.  They remain in the A1 (D4-invariant)
    # sector; lattice-anisotropy irreps are intentionally not mixed in here.
    Operator("E8_phi8", "even", lambda p: _sum(p**8)),
    Operator("E9_bilinear_21", "even", lambda p: _bilinear_shell(p, ((2, 1), (2, -1), (1, 2), (1, -2)))),
    Operator("E10_bilinear_22", "even", lambda p: _bilinear_shell(p, ((2, 2), (2, -2)))),
    Operator("E11_bilinear_30", "even", lambda p: _bilinear_shell(p, ((3, 0), (0, 3)))),
    Operator("E12_nn_phi3phi", "even", _nn_cubic),
    Operator("E13_phi2_gradphi2", "even", _phi2_gradphi2),
    Operator("E14_laplacian2", "even", _laplacian2),
)
ODD_OPERATORS = (
    Operator("O1_phi", "odd", _sum),
    Operator("O2_phi3", "odd", lambda p: _sum(p**3)),
    Operator("O3_phi5", "odd", lambda p: _sum(p**5)),
    Operator("O4_nn_mixed", "odd", _nn_mixed),
)
REGISTRY = {op.name: op for op in EVEN_OPERATORS + ODD_OPERATORS}


def names(parity: str) -> list[str]:
    return [op.name for op in (EVEN_OPERATORS if parity == "even" else ODD_OPERATORS)]


def measure(phi: Array, operator_names: list[str]) -> Array:
    """Return per-configuration extensive operator sums, shape ``(N, n_ops)``."""
    unknown = set(operator_names).difference(REGISTRY)
    if unknown:
        raise KeyError(f"unknown MCRG operators: {sorted(unknown)}")
    values = [REGISTRY[name].evaluate(phi) for name in operator_names]
    return np.column_stack(values)
