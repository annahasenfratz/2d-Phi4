from __future__ import annotations

import numpy as np

from .io import ActionSpec
from .actions import action_total


def second_moment_components(phi: np.ndarray) -> dict[str, np.ndarray | float | int]:
    """Return Fourier second-moment correlation-length ingredients.

    This is the standard finite-volume second-moment estimator on a periodic
    square lattice.  The Fourier amplitudes are computed configuration by
    configuration and ensemble averages are formed before the nonlinear ratio is
    taken.  It is distinct from the Del Debbio pole-mass estimator below.
    """
    arr = np.asarray(phi, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr[None, :, :]
    if arr.ndim != 3 or arr.shape[1] != arr.shape[2]:
        raise ValueError(f"expected phi with shape (N,L,L) or (L,L), got {arr.shape}")
    n, L, _ = arr.shape
    V = float(L * L)
    m = arr.mean(axis=(1, 2))
    # Numpy's fft2 convention uses exp(-2*pi*i*k*x/L); |Phi|^2 is invariant
    # under the sign convention.
    fft = np.fft.fft2(arr, axes=(1, 2))
    G0 = float(V * (np.mean(m * m) - np.mean(m) ** 2))
    Gpx_cfg = (np.abs(fft[:, 1, 0]) ** 2) / V
    Gpy_cfg = (np.abs(fft[:, 0, 1]) ** 2) / V
    Gpx = float(np.mean(Gpx_cfg))
    Gpy = float(np.mean(Gpy_cfg))
    Gp = float(0.5 * (Gpx + Gpy))
    ratio = float(G0 / Gp) if Gp > 0.0 else float("nan")
    sqrt_arg = ratio - 1.0 if np.isfinite(ratio) else float("nan")
    valid = bool(np.isfinite(G0) and np.isfinite(Gp) and G0 > 0.0 and Gp > 0.0 and sqrt_arg > 0.0)
    if valid:
        xi = float(np.sqrt(sqrt_arg) / (2.0 * np.sin(np.pi / L)))
        xi_over_L = float(xi / L)
    else:
        xi = float("nan")
        xi_over_L = float("nan")
    return {
        "m": m,
        "m2": m * m,
        "G_pmin_x_cfg": Gpx_cfg,
        "G_pmin_y_cfg": Gpy_cfg,
        "G0_connected": G0,
        "G_pmin_x": Gpx,
        "G_pmin_y": Gpy,
        "G_pmin": Gp,
        "G0_over_Gpmin": ratio,
        "xi_2nd_sqrt_arg": sqrt_arg,
        "xi_2nd": xi,
        "xi_over_L_2nd": xi_over_L,
        "xi_over_L": xi_over_L,
        "xi_2nd_valid": int(valid),
        "xi_2nd_G0_positive": int(np.isfinite(G0) and G0 > 0.0),
        "xi_2nd_Gpmin_positive": int(np.isfinite(Gp) and Gp > 0.0),
        "xi_2nd_ratio_gt_one": int(np.isfinite(ratio) and ratio > 1.0),
        "xi_2nd_rotational_asymmetry": float((Gpx - Gpy) / Gp) if Gp > 0.0 else float("nan"),
        "xi_2nd_n_configs": int(n),
    }


def second_moment_xi_over_L(phi: np.ndarray) -> float:
    """Convenience wrapper returning the Fourier second-moment xi/L."""
    return float(second_moment_components(phi)["xi_over_L_2nd"])


def bootstrap_second_moment_xi(
    phi: np.ndarray,
    *,
    n_bootstrap: int = 1000,
    seed: int = 12345,
    bin_size: int = 1,
) -> dict[str, float | int]:
    """Bootstrap the Fourier second-moment xi/L over configurations or bins."""
    arr = np.asarray(phi, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr[None, :, :]
    n = int(arr.shape[0])
    if bin_size < 1:
        raise ValueError("bin_size must be positive")
    n_bins = n // bin_size
    if n_bins < 1:
        raise ValueError(f"bin_size {bin_size} is larger than sample count {n}")
    trimmed = arr[: n_bins * bin_size]
    rng = np.random.default_rng(seed)
    vals = np.empty(n_bootstrap, dtype=np.float64)
    rejected = 0
    for b in range(n_bootstrap):
        idx = rng.integers(0, n_bins, size=n_bins)
        sample = trimmed.reshape(n_bins, bin_size, arr.shape[1], arr.shape[2])[idx].reshape(n_bins * bin_size, arr.shape[1], arr.shape[2])
        val = second_moment_xi_over_L(sample)
        vals[b] = val
        if not np.isfinite(val):
            rejected += 1
    finite = vals[np.isfinite(vals)]
    return {
        "xi_over_L_2nd_bootstrap_mean": float(np.mean(finite)) if finite.size else float("nan"),
        "xi_over_L_2nd_bootstrap_se": float(np.std(finite, ddof=1)) if finite.size > 1 else float("nan"),
        "xi_over_L_2nd_bootstrap_valid": int(finite.size),
        "xi_over_L_2nd_bootstrap_rejected": int(rejected),
        "xi_over_L_2nd_bootstrap_rejected_fraction": float(rejected / max(n_bootstrap, 1)),
        "xi_over_L_2nd_bootstrap_bin_size": int(bin_size),
        "xi_over_L_2nd_bootstrap_n_bins": int(n_bins),
        "xi_over_L_2nd_bootstrap_samples": int(n_bootstrap),
    }


def periodic_cosh_effective_mass(C: np.ndarray, *, clip_arccosh: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Return periodic-cosh effective masses from an ensemble-averaged correlator.

    Values are only reported for 1 <= t <= L/2 - 1. Outside that range, or when
    C(t) <= 0 or the arccosh argument is invalid, the effective mass is NaN unless
    clipping is explicitly enabled.
    """
    corr = np.asarray(C, dtype=np.float64)
    denom = 2.0 * corr
    with np.errstate(divide="ignore", invalid="ignore"):
        arg = (np.roll(corr, 1) + np.roll(corr, -1)) / denom
    mass = np.full_like(arg, np.nan, dtype=np.float64)
    if corr.ndim != 1:
        raise ValueError(f"expected one-dimensional correlator, got shape {corr.shape}")
    L = int(corr.shape[0])
    allowed_t = np.zeros(L, dtype=bool)
    allowed_t[1 : max(1, L // 2)] = True
    base_good = allowed_t & np.isfinite(arg) & np.isfinite(corr) & (corr > 0.0)
    if clip_arccosh:
        good = base_good
        mass[good] = np.arccosh(np.maximum(arg[good], 1.0))
    else:
        good = base_good & (arg >= 1.0)
        mass[good] = np.arccosh(arg[good])
    return mass, arg


def del_debbio_default_window(L: int) -> tuple[int, int]:
    t_min = max(2, L // 6)
    t_max = max(t_min, L // 2 - 2)
    return int(t_min), int(t_max)


def del_debbio_plateau_from_correlator(
    C: np.ndarray,
    *,
    t_min: int | None = None,
    t_max: int | None = None,
    clip_arccosh: bool = False,
) -> dict[str, np.ndarray | float | int]:
    corr = np.asarray(C, dtype=np.float64)
    if corr.ndim != 1:
        raise ValueError(f"expected one-dimensional correlator, got shape {corr.shape}")
    L = int(corr.shape[0])
    m_eff, m_eff_arg = periodic_cosh_effective_mass(corr, clip_arccosh=clip_arccosh)
    default_t_min, default_t_max = del_debbio_default_window(L)
    t_min = default_t_min if t_min is None else int(t_min)
    t_max = default_t_max if t_max is None else int(t_max)
    t_min = max(1, min(t_min, L // 2 - 1))
    t_max = max(t_min, min(t_max, L // 2 - 1))
    window = m_eff[t_min : t_max + 1]
    valid = np.isfinite(window)
    n_valid = int(np.sum(valid))
    if n_valid >= 2:
        m_pole = float(np.mean(window[valid]))
    else:
        m_pole = float("nan")
    xi = float(1.0 / m_pole) if np.isfinite(m_pole) and m_pole > 0.0 else float("nan")
    return {
        "m_eff_cosh_t": m_eff,
        "m_eff_cosh_t_arccosh_arg": m_eff_arg,
        "m_pole_DD_plateau": m_pole,
        "xi_pole_DD": xi,
        "xi_over_L_pole_DD": float(xi / L) if np.isfinite(xi) else float("nan"),
        "m_eff_window_t_min": int(t_min),
        "m_eff_window_t_max": int(t_max),
        "m_eff_window_valid_points": n_valid,
        "m_eff_bad_arccosh_count": int(np.sum(~np.isfinite(m_eff[t_min : t_max + 1]))),
    }


def plot_del_debbio_effective_mass(dd_obs: dict[str, np.ndarray | float | int], path: str) -> None:
    """Save a PDF/PNG diagnostic plot of m_eff(t) with the plateau window shaded."""
    import matplotlib.pyplot as plt

    m_eff = np.asarray(dd_obs["m_eff_cosh_t"], dtype=np.float64)
    t = np.arange(len(m_eff))
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(t, m_eff, marker="o", linestyle="-", label=r"$m_{\rm eff}(t)$")
    t_min = int(dd_obs["m_eff_window_t_min"])
    t_max = int(dd_obs["m_eff_window_t_max"])
    ax.axvspan(t_min - 0.5, t_max + 0.5, alpha=0.18, color="tab:orange", label="plateau window")
    m_pole = float(dd_obs["m_pole_DD_plateau"])
    if np.isfinite(m_pole):
        ax.axhline(m_pole, color="tab:red", linestyle="--", label=r"$m_{\rm pole}^{\rm plateau}$")
    ax.set_xlabel("t")
    ax.set_ylabel(r"$m_{\rm eff}^{\rm cosh}(t)$")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_del_debbio_correlator_and_effective_mass(dd_obs: dict[str, np.ndarray | float | int], path: str) -> None:
    """Save C(t) and m_eff(t) diagnostics with the plateau window highlighted."""
    import matplotlib.pyplot as plt

    C = np.asarray(dd_obs["C_connected_slice_t"], dtype=np.float64)
    m_eff = np.asarray(dd_obs["m_eff_cosh_t"], dtype=np.float64)
    t = np.arange(len(C))
    t_min = int(dd_obs["m_eff_window_t_min"])
    t_max = int(dd_obs["m_eff_window_t_max"])
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 7.0), sharex=True)
    pos = C > 0.0
    axes[0].plot(t[pos], C[pos], marker="o", linestyle="-")
    axes[0].axvspan(t_min - 0.5, t_max + 0.5, alpha=0.18, color="tab:orange")
    axes[0].set_yscale("log")
    axes[0].set_ylabel(r"$C_{\rm slice}(t)$")
    axes[1].plot(t, m_eff, marker="o", linestyle="-", label=r"$m_{\rm eff}(t)$")
    axes[1].axvspan(t_min - 0.5, t_max + 0.5, alpha=0.18, color="tab:orange", label="plateau window")
    m_pole = float(dd_obs["m_pole_DD_plateau"])
    if np.isfinite(m_pole):
        axes[1].axhline(m_pole, color="tab:red", linestyle="--", label=r"$m_{\rm pole}^{\rm plateau}$")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel(r"$m_{\rm eff}^{\rm cosh}(t)$")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def del_debbio_pole_observables(
    phi: np.ndarray,
    *,
    t_min: int | None = None,
    t_max: int | None = None,
    clip_arccosh: bool = False,
) -> dict[str, np.ndarray | float | int]:
    """Compute the Del Debbio et al. connected-correlator pole-mass diagnostic.

    This implements the Del Debbio et al. pole-mass correlation-length estimator
    from the large-distance periodic-cosh decay of the connected two-point
    function. It is distinct from the second-moment/Fourier correlation length.
    The pole mass should be extracted from a large-distance plateau window,
    avoiding both the UV region near t=0 and the flat/noisy midpoint region near
    L/2.
    """
    arr = np.asarray(phi, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr[None, :, :]
    if arr.ndim != 3 or arr.shape[1] != arr.shape[2]:
        raise ValueError(f"expected phi with shape (N,L,L) or (L,L), got {arr.shape}")
    n, L, _ = arr.shape
    m = arr.mean(axis=(1, 2))
    Cx_cfg = np.empty((n, L), dtype=np.float64)
    Cy_cfg = np.empty((n, L), dtype=np.float64)
    m2 = m * m
    # Timeslice-projected zero-spatial-momentum correlator. This differs from a
    # point-displacement correlator averaged over sites: first project each
    # timeslice to p_spatial=0, then correlate the slice averages.
    slice_y = arr.mean(axis=1)  # average over x, shape (N, y)
    slice_x = arr.mean(axis=2)  # average over y, shape (N, x)
    for t in range(L):
        Cx_cfg[:, t] = np.mean(slice_x * np.roll(slice_x, -t, axis=1), axis=1) - m2
        Cy_cfg[:, t] = np.mean(slice_y * np.roll(slice_y, -t, axis=1), axis=1) - m2
    C_cfg = 0.5 * (Cx_cfg + Cy_cfg)
    C = C_cfg.mean(axis=0)
    C_se = C_cfg.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros(L, dtype=np.float64)
    Cx = Cx_cfg.mean(axis=0)
    Cy = Cy_cfg.mean(axis=0)
    plateau = del_debbio_plateau_from_correlator(C, t_min=t_min, t_max=t_max, clip_arccosh=clip_arccosh)
    return {
        "C_connected_slice_t": C,
        "C_connected_slice_t_se": C_se,
        "C_connected_slice_x_t": Cx,
        "C_connected_slice_y_t": Cy,
        "C_connected_t": C,
        "C_connected_t_se": C_se,
        "C_connected_x_t": Cx,
        "C_connected_y_t": Cy,
        "C_connected_slice_t0_nonpositive": int(not np.isfinite(C[0]) or C[0] <= 0.0),
        "C_connected_t0_nonpositive": int(not np.isfinite(C[0]) or C[0] <= 0.0),
        **plateau,
    }


def observables(phi: np.ndarray, action: ActionSpec) -> dict[str, np.ndarray | float]:
    arr = np.asarray(phi, dtype=np.float64)
    n = arr.shape[0] if arr.ndim == 3 else 1
    if arr.ndim == 2:
        arr = arr[None, :, :]
    m = arr.mean(axis=(1, 2))
    phi2 = np.mean(arr**2, axis=(1, 2))
    phi4 = np.mean(arr**4, axis=(1, 2))
    nnx = arr * np.roll(arr, -1, axis=1)
    nny = arr * np.roll(arr, -1, axis=2)
    diag = arr * np.roll(np.roll(arr, -1, axis=1), -1, axis=2)
    links = np.concatenate([nnx.reshape(arr.shape[0], -1), nny.reshape(arr.shape[0], -1)], axis=1)
    nn = links.mean(axis=1)
    nn2 = (links**2).mean(axis=1)
    diag_link = diag.reshape(arr.shape[0], -1).mean(axis=1)
    m2 = m**2
    m4 = m**4
    binder_u4 = 1.0 - m4.mean() / (3.0 * (m2.mean() ** 2))
    binder_b4 = m4.mean() / (m2.mean() ** 2)
    binder_q = (m2.mean() ** 2) / m4.mean()
    susceptibility = float(arr.shape[1] * arr.shape[2] * (np.mean(m2) - np.mean(m) ** 2))
    xi_proxy = float(np.sqrt(max(np.mean(m2), 0.0) / max(float(phi2.mean()), 1.0e-300)))
    xi_2nd = second_moment_components(arr)
    xi_over_L = float(xi_2nd["xi_over_L_2nd"])
    action_density = action_total(arr, action) / (arr.shape[1] * arr.shape[2])
    out = {
        "n": int(arr.shape[0]),
        "L": int(arr.shape[1]),
        "m": float(m.mean()),
        "abs_m": float(np.abs(m).mean()),
        "m2": float(m2.mean()),
        "m4": float(m4.mean()),
        "phi2": float(phi2.mean()),
        "phi4": float(phi4.mean()),
        "NN": float(nn.mean()),
        "nn2": float(nn2.mean()),
        "diag": float(diag_link.mean()),
        "Binder_U4": float(binder_u4),
        "binder_U4": float(binder_u4),
        "binder_B4": float(binder_b4),
        "binder_Q": float(binder_q),
        "susceptibility": float(susceptibility),
        "xi_over_L": float(xi_over_L),
        "xi_over_L_2nd": float(xi_over_L),
        "xi_2nd": float(xi_2nd["xi_2nd"]),
        "G0_connected": float(xi_2nd["G0_connected"]),
        "G_pmin_x": float(xi_2nd["G_pmin_x"]),
        "G_pmin_y": float(xi_2nd["G_pmin_y"]),
        "G_pmin": float(xi_2nd["G_pmin"]),
        "G0_over_Gpmin": float(xi_2nd["G0_over_Gpmin"]),
        "xi_2nd_sqrt_arg": float(xi_2nd["xi_2nd_sqrt_arg"]),
        "xi_2nd_valid": int(xi_2nd["xi_2nd_valid"]),
        "xi_2nd_rotational_asymmetry": float(xi_2nd["xi_2nd_rotational_asymmetry"]),
        "xi_over_L_m2_phi2_proxy": float(xi_proxy),
        "action_density": float(action_density.mean()),
    }
    dd = del_debbio_pole_observables(arr)
    dd_xi_over_l = float(dd["xi_over_L_pole_DD"])
    xi_ratio = dd_xi_over_l / xi_over_L if np.isfinite(dd_xi_over_l) and np.isfinite(xi_over_L) and xi_over_L > 0.0 else float("nan")
    if np.isfinite(xi_ratio) and (xi_ratio > 2.0 or xi_ratio < 0.5):
        print(
            f"warning: xi_over_L_pole_DD={dd_xi_over_l:.6g} differs from xi_over_L={xi_over_L:.6g} by factor {xi_ratio:.6g}",
            flush=True,
        )
    for key in [
        "m_pole_DD_plateau",
        "xi_pole_DD",
        "xi_over_L_pole_DD",
        "m_eff_window_t_min",
        "m_eff_window_t_max",
        "m_eff_window_valid_points",
        "C_connected_t0_nonpositive",
        "m_eff_bad_arccosh_count",
    ]:
        out[key] = dd[key]
    out["xi_over_L_pole_DD_to_second_moment_ratio"] = float(xi_ratio)
    out["xi_over_L_pole_DD_second_moment_warning"] = int(np.isfinite(xi_ratio) and (xi_ratio > 2.0 or xi_ratio < 0.5))
    return out
