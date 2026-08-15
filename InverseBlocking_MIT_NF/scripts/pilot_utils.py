from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from invblock_mit_nf.blocking import BlockingKernel2D, momentum_inverse_upscale_to_even_even


def onsite(phi: np.ndarray | float, lam: float) -> np.ndarray | float:
    phi = np.asarray(phi)
    return (1.0 - 2.0 * lam) * phi * phi + lam * phi**4


def phi4_action_numpy(cfgs: np.ndarray, *, kappa: float, lam: float) -> np.ndarray:
    density = onsite(cfgs, lam)
    nearest = np.zeros_like(cfgs)
    nearest = nearest + cfgs * np.roll(cfgs, -1, axis=1)
    nearest = nearest + cfgs * np.roll(cfgs, -1, axis=2)
    density = density - 2.0 * kappa * nearest
    return density.sum(axis=(1, 2))


def metropolis_sweep(phi: np.ndarray, *, kappa: float, lam: float, width: float, rng: np.random.Generator) -> tuple[int, int]:
    xx, yy = np.indices(phi.shape)
    accepts = 0
    trials = 0
    for parity in (0, 1):
        mask = (xx + yy) % 2 == parity
        old = phi[mask]
        prop = old + width * rng.normal(size=old.shape)
        neigh = (
            np.roll(phi, 1, axis=0)
            + np.roll(phi, -1, axis=0)
            + np.roll(phi, 1, axis=1)
            + np.roll(phi, -1, axis=1)
        )[mask]
        delta = onsite(prop, lam) - onsite(old, lam) - 2.0 * kappa * (prop - old) * neigh
        acc = np.log(rng.random(size=old.shape)) < -delta
        updated = old.copy()
        updated[acc] = prop[acc]
        phi[mask] = updated
        accepts += int(np.sum(acc))
        trials += int(mask.sum())
    return accepts, trials


def generate_coarse_ensemble(
    *,
    L: int,
    kappa: float,
    lam: float,
    n_samples: int,
    thermal_sweeps: int,
    skip_sweeps: int,
    proposal_width: float,
    seed: int,
) -> tuple[np.ndarray, dict[str, float], list[dict[str, float]]]:
    rng = np.random.default_rng(seed)
    phi = rng.normal(size=(L, L))
    accepts = 0
    trials = 0
    history: list[dict[str, float]] = []

    def sweep() -> None:
        nonlocal accepts, trials
        a, t = metropolis_sweep(phi, kappa=kappa, lam=lam, width=proposal_width, rng=rng)
        accepts += a
        trials += t

    for _ in range(thermal_sweeps):
        sweep()

    out = np.empty((n_samples, L, L), dtype=np.float64)
    for i in range(n_samples):
        for _ in range(skip_sweeps):
            sweep()
        out[i] = phi
        S = float(phi4_action_numpy(out[i : i + 1], kappa=kappa, lam=lam)[0])
        mean_phi = float(np.mean(phi))
        abs_mean_phi = float(abs(mean_phi))
        phi2 = float(np.mean(phi**2))
        nn_x = float(np.mean(phi * np.roll(phi, -1, axis=0)))
        nn_y = float(np.mean(phi * np.roll(phi, -1, axis=1)))
        history.append(
            {
                "sample": i,
                "action": S,
                "mean_phi": mean_phi,
                "abs_mean_phi": abs_mean_phi,
                "phi2": phi2,
                "nn_bond_mean": 0.5 * (nn_x + nn_y),
                "nn_action_term": float(-2.0 * kappa * (nn_x + nn_y)),
            }
        )

    summary = {
        "L": L,
        "kappa": kappa,
        "lam": lam,
        "n_samples": n_samples,
        "thermal_sweeps": thermal_sweeps,
        "skip_sweeps": skip_sweeps,
        "proposal_width": proposal_width,
        "seed": seed,
        "local_acceptance": accepts / trials if trials else float("nan"),
        "mean_action": float(np.mean(phi4_action_numpy(out, kappa=kappa, lam=lam))),
        "mean_phi": float(np.mean(out)),
        "mean_abs_phi": float(np.mean(np.abs(np.mean(out, axis=(1, 2))))),
        "mean_phi2": float(np.mean(out**2)),
        "mean_nn_bond": float(
            0.5
            * (
                np.mean(out * np.roll(out, -1, axis=1))
                + np.mean(out * np.roll(out, -1, axis=2))
            )
        ),
        "finite": bool(np.isfinite(out).all()),
    }
    return out, summary, history


def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def torch_from_numpy_configs(configs: np.ndarray, *, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    return torch.as_tensor(configs, dtype=dtype)


def observables_numpy(cfg: np.ndarray) -> dict[str, float]:
    cfg = np.asarray(cfg, dtype=np.float64)
    mean_phi = float(np.mean(cfg))
    abs_mean_phi = float(abs(mean_phi))
    m2 = float(np.mean(cfg**2))
    m4 = float(np.mean(cfg**4))
    binder = float(1.0 - m4 / (3.0 * m2 * m2)) if m2 > 0 else float("nan")
    nn = float(0.5 * (np.mean(cfg * np.roll(cfg, -1, axis=-2)) + np.mean(cfg * np.roll(cfg, -1, axis=-1))))
    diag = float(
        0.25
        * (
            np.mean(cfg * np.roll(np.roll(cfg, -1, axis=-2), -1, axis=-1))
            + np.mean(cfg * np.roll(np.roll(cfg, -1, axis=-2), 1, axis=-1))
            + np.mean(cfg * np.roll(np.roll(cfg, 1, axis=-2), -1, axis=-1))
            + np.mean(cfg * np.roll(np.roll(cfg, 1, axis=-2), 1, axis=-1))
        )
    )
    return {
        "mean_phi": mean_phi,
        "abs_mean_phi": abs_mean_phi,
        "m2": m2,
        "m4": m4,
        "binder": binder,
        "nn": nn,
        "diag": diag,
    }


def action_components_numpy(cfgs: np.ndarray, *, kappa: float, lam: float) -> dict[str, np.ndarray]:
    cfgs = np.asarray(cfgs, dtype=np.float64)
    quadratic = np.sum(cfgs**2, axis=(-2, -1))
    quartic = np.sum(lam * (cfgs**2 - 1.0) ** 2, axis=(-2, -1))
    hopping = -2.0 * kappa * (
        np.sum(cfgs * np.roll(cfgs, -1, axis=-2), axis=(-2, -1))
        + np.sum(cfgs * np.roll(cfgs, -1, axis=-1), axis=(-2, -1))
    )
    total = quadratic + quartic + hopping
    return {
        "quadratic": quadratic,
        "quartic": quartic,
        "hopping": hopping,
        "total": total,
    }


def ensemble_observables_numpy(cfgs: np.ndarray) -> dict[str, float]:
    cfgs = np.asarray(cfgs, dtype=np.float64)
    obs = [observables_numpy(sample) for sample in cfgs]
    mean_phi = np.array([o["mean_phi"] for o in obs])
    abs_mean_phi = np.array([o["abs_mean_phi"] for o in obs])
    m2 = np.array([o["m2"] for o in obs])
    m4 = np.array([o["m4"] for o in obs])
    nn = np.array([o["nn"] for o in obs])
    diag = np.array([o["diag"] for o in obs])
    twolink = 0.5 * (
        np.mean(cfgs * np.roll(cfgs, -2, axis=-2), axis=(-2, -1))
        + np.mean(cfgs * np.roll(cfgs, -2, axis=-1), axis=(-2, -1))
    )
    V = cfgs.shape[-1] * cfgs.shape[-2]
    mbar = float(np.mean(mean_phi))
    chi = float(V * (np.mean(mean_phi**2) - mbar**2))
    binder = float(1.0 - np.mean(mean_phi**4) / (3.0 * np.mean(mean_phi**2) ** 2)) if np.mean(mean_phi**2) > 0 else float("nan")
    phif2 = float(np.mean(cfgs**2))
    phif4 = float(np.mean(cfgs**4))
    xi_over_L = float("nan")
    try:
        ft = np.fft.fftn(cfgs, axes=(-2, -1))
        g0 = np.mean(np.abs(ft[:, 0, 0]) ** 2) / V
        gp = 0.5 * (
            np.mean(np.abs(ft[:, 1, 0]) ** 2) / V
            + np.mean(np.abs(ft[:, 0, 1]) ** 2) / V
        )
        if gp > 0 and g0 > gp:
            xi = 1.0 / (2.0 * np.sin(np.pi / cfgs.shape[-1])) * np.sqrt(g0 / gp - 1.0)
            xi_over_L = float(xi / cfgs.shape[-1])
    except Exception:
        xi_over_L = float("nan")
    return {
        "mean_phi": float(np.mean(mean_phi)),
        "abs_mean_phi": float(np.mean(abs_mean_phi)),
        "phi2": phif2,
        "phi4": phif4,
        "nn": float(np.mean(nn)),
        "diag": float(np.mean(diag)),
        "two_link": float(np.mean(twolink)),
        "susceptibility": chi,
        "binder": binder,
        "xi_over_L": xi_over_L,
    }


def block_fine_to_coarse(fine: np.ndarray, kernel: BlockingKernel2D) -> np.ndarray:
    fine = np.asarray(fine, dtype=np.float64)
    if fine.ndim != 3 or fine.shape[-1] != fine.shape[-2]:
        raise ValueError("fine must have shape [batch, L, L]")
    Lf = fine.shape[-1]
    if Lf % 2 != 0:
        raise ValueError("fine lattice size must be even")
    Lc = Lf // 2
    fine_t = torch.as_tensor(fine[:, 0::2, 0::2], dtype=torch.float64)
    coarse = torch.fft.ifft2(torch.fft.fft2(fine_t) * kernel.symbol(Lc, device=fine_t.device).unsqueeze(0)).real
    return coarse.cpu().numpy()


def inverse_upscale_to_condition(coarse: np.ndarray, kernel: BlockingKernel2D) -> np.ndarray:
    coarse_t = torch_from_numpy_configs(coarse)
    cond = momentum_inverse_upscale_to_even_even(coarse_t, kernel, Lf=coarse.shape[-1] * 2)
    return cond.cpu().numpy()
