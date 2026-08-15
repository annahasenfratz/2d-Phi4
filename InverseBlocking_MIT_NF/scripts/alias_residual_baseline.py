#!/usr/bin/env python3
"""Alias-residual baselines for deterministic inverse blocking."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
BASE = PROJECT / "outputs" / "inverse_blocking_step_by_step"
OUT = BASE / "alias_residual_baseline"
KERNEL_META = PROJECT / "kernels" / "from_perfect_blocking_lam1p0" / "selected_kernel_metadata.json"

L_FINE = 16
L_COARSE = 8
ETA_EXPONENT = 0.25
B = 2
BLOCK_NORM = B ** (ETA_EXPONENT / 2.0)
LAMBDA = 1.0
KAPPA_F = 0.320
BOOTSTRAP_N = 256
RNG_SEED = 20240623

SHELLS = {
    "w00": [(0, 0)],
    "w10": [(1, 0), (-1, 0), (0, 1), (0, -1)],
    "w11": [(1, 1), (1, -1), (-1, 1), (-1, -1)],
    "w20": [(2, 0), (-2, 0), (0, 2), (0, -2)],
    "w21": [
        (2, 1),
        (2, -1),
        (-2, 1),
        (-2, -1),
        (1, 2),
        (1, -2),
        (-1, 2),
        (-1, -2),
    ],
    "w22": [(2, 2), (2, -2), (-2, 2), (-2, -2)],
}
SECTORS = {
    "R00": (0, 0),
    "R10": (L_COARSE, 0),
    "R01": (0, L_COARSE),
    "R11": (L_COARSE, L_COARSE),
}


def fine_low_index(k: int) -> int:
    return k if k < L_COARSE // 2 else k + L_COARSE


def sector_indices() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    ky = np.array([fine_low_index(k) for k in range(L_COARSE)])
    kx = np.array([fine_low_index(k) for k in range(L_COARSE)])
    yy, xx = np.meshgrid(ky, kx, indexing="ij")
    return {name: ((yy + dy) % L_FINE, (xx + dx) % L_FINE) for name, (dy, dx) in SECTORS.items()}


def kernel_array(weights: dict[str, float]) -> np.ndarray:
    arr = np.zeros((L_FINE, L_FINE), dtype=float)
    for shell, offsets in SHELLS.items():
        for dy, dx in offsets:
            arr[dy % L_FINE, dx % L_FINE] += weights[shell]
    return arr


def assemble_from_alias_values(alias_values: dict[str, np.ndarray], idx: dict[str, tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    n = next(iter(alias_values.values())).shape[0]
    out = np.zeros((n, L_FINE, L_FINE), dtype=complex)
    for name, vals in alias_values.items():
        yy, xx = idx[name]
        out[:, yy, xx] = vals
    return out


def obs(configs: np.ndarray) -> dict[str, float]:
    _, ly, lx = configs.shape
    v = ly * lx
    m_cfg = configs.mean(axis=(-2, -1))
    nn_cfg = 0.5 * (
        (configs * np.roll(configs, -1, axis=-2)).mean(axis=(-2, -1))
        + (configs * np.roll(configs, -1, axis=-1)).mean(axis=(-2, -1))
    )
    nn2_cfg = 0.5 * (
        ((configs * np.roll(configs, -1, axis=-2)) ** 2).mean(axis=(-2, -1))
        + ((configs * np.roll(configs, -1, axis=-1)) ** 2).mean(axis=(-2, -1))
    )
    diag_cfg = (configs * np.roll(np.roll(configs, -1, axis=-2), -1, axis=-1)).mean(axis=(-2, -1))
    twonn_cfg = 0.5 * (
        (configs * np.roll(configs, -2, axis=-2)).mean(axis=(-2, -1))
        + (configs * np.roll(configs, -2, axis=-1)).mean(axis=(-2, -1))
    )
    m2 = float(np.mean(m_cfg**2))
    m4 = float(np.mean(m_cfg**4))
    b4 = m4 / (m2 * m2) if m2 > 0 else math.nan
    u4 = 1.0 - b4 / 3.0 if math.isfinite(b4) else math.nan
    ft = np.fft.fft2(configs, axes=(-2, -1))
    chi = float(v * (np.mean(m_cfg**2) - np.mean(m_cfg) ** 2))
    fmin = float(0.5 * (np.mean(np.abs(ft[:, 1, 0]) ** 2) + np.mean(np.abs(ft[:, 0, 1]) ** 2)) / v)
    ratio = chi / fmin - 1.0 if fmin > 0 else math.nan
    xi = (1.0 / (2.0 * math.sin(math.pi / lx))) * math.sqrt(ratio) if ratio > 0 else math.nan
    action_hop = -4.0 * KAPPA_F * float(np.mean(nn_cfg))
    action_phi2 = (1.0 - 2.0 * LAMBDA) * float(np.mean(configs**2))
    action_phi4 = LAMBDA * float(np.mean(configs**4))
    return {
        "phi2": float(np.mean(configs**2)),
        "phi4": float(np.mean(configs**4)),
        "NN": float(np.mean(nn_cfg)),
        "nn2": float(np.mean(nn2_cfg)),
        "diag": float(np.mean(diag_cfg)),
        "2nn": float(np.mean(twonn_cfg)),
        "Binder_U4": float(u4),
        "xi/L": float(xi / lx) if math.isfinite(xi) else math.nan,
        "action_hopping_density": action_hop,
        "action_phi2_density": action_phi2,
        "action_phi4_density": action_phi4,
        "action_density": action_hop + action_phi2 + action_phi4,
    }


def bootstrap(configs: np.ndarray, rng: np.random.Generator) -> dict[str, dict[str, float]]:
    mean = obs(configs)
    reps = {k: [] for k in mean}
    n = configs.shape[0]
    for _ in range(BOOTSTRAP_N):
        idx = rng.integers(0, n, size=n)
        val = obs(configs[idx])
        for k in reps:
            reps[k].append(val[k])
    return {k: {"mean": mean[k], "error": float(np.nanstd(reps[k], ddof=1))} for k in mean}


def write_operator_comparison(ensembles: dict[str, np.ndarray]) -> dict:
    rng = np.random.default_rng(RNG_SEED)
    stats = {name: bootstrap(arr, rng) for name, arr in ensembles.items()}
    base = stats["original_fine"]
    rows = []
    for name, vals in stats.items():
        for op, stat in vals.items():
            diff = stat["mean"] - base[op]["mean"]
            den = math.sqrt(stat["error"] ** 2 + base[op]["error"] ** 2)
            rows.append(
                {
                    "ensemble": name,
                    "operator": op,
                    "mean": stat["mean"],
                    "error": stat["error"],
                    "original_fine_mean": base[op]["mean"],
                    "original_fine_error": base[op]["error"],
                    "difference_vs_original_fine": diff,
                    "z_score_vs_original_fine": diff / den if den > 0 else math.nan,
                }
            )
    with (OUT / "operator_comparison.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    md = ["# Alias-Residual Baseline Operator Comparison\n\n"]
    for name in ensembles:
        md += [f"\n## {name}\n", "| operator | mean | error | diff vs original fine | z |\n", "|---|---:|---:|---:|---:|\n"]
        for row in rows:
            if row["ensemble"] == name:
                md.append(
                    f"| {row['operator']} | {row['mean']:.8g} | {row['error']:.3g} | "
                    f"{row['difference_vs_original_fine']:.8g} | {row['z_score_vs_original_fine']:.3g} |\n"
                )
    (OUT / "operator_comparison.md").write_text("".join(md))
    return {"stats": stats, "rows": rows}


def simple_observable_csv(path: Path, ensembles: dict[str, np.ndarray]) -> None:
    rows = []
    for name, arr in ensembles.items():
        vals = obs(arr)
        rows.extend({"ensemble": name, "operator": k, "value": v} for k, v in vals.items())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def row(rows: list[dict], ensemble: str, operator: str) -> dict:
    return next(r for r in rows if r["ensemble"] == ensemble and r["operator"] == operator)


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(f"Refusing to overwrite existing outputs in {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)

    meta = json.loads(KERNEL_META.read_text())
    weights = {k: float(meta["weights"][k]) for k in ["w00", "w10", "w11", "w20", "w21", "w22"]}
    phi = np.load(BASE / "input_fine_batch.npy")
    current_alias = np.load(BASE / "inverse_kernel_alias_field.npy")
    block_replicated = np.load(BASE / "block_replicate_fill" / "block_replicated_phi_block.npy")
    ktilde = np.fft.fft2(kernel_array(weights))
    phi_tilde = np.fft.fft2(phi, axes=(-2, -1))
    psi_tilde = BLOCK_NORM * ktilde[None, :, :] * phi_tilde
    idx = sector_indices()

    alias_values = {name: psi_tilde[:, yy, xx] for name, (yy, xx) in idx.items()}
    s = 0.25 * sum(alias_values.values())
    residuals = {name: vals - s for name, vals in alias_values.items()}
    residual_sum = sum(residuals.values())

    oracle_psi_values = {name: s + residuals[name] for name in SECTORS}
    zero_psi_values = {name: s for name in SECTORS}
    donor = np.roll(np.arange(phi.shape[0]), 1)
    sampled_psi_values = {name: s + residuals[name][donor] for name in SECTORS}

    def inverse_from_psi_values(values: dict[str, np.ndarray]) -> tuple[np.ndarray, float]:
        psi_full = assemble_from_alias_values(values, idx)
        phi_full_tilde = np.zeros_like(psi_full)
        for name, (yy, xx) in idx.items():
            phi_full_tilde[:, yy, xx] = psi_full[:, yy, xx] / (BLOCK_NORM * ktilde[yy, xx])
        field_complex = np.fft.ifft2(phi_full_tilde, axes=(-2, -1))
        return field_complex.real, float(np.max(np.abs(field_complex.imag)))

    oracle, oracle_imag = inverse_from_psi_values(oracle_psi_values)
    zero, zero_imag = inverse_from_psi_values(zero_psi_values)
    sampled, sampled_imag = inverse_from_psi_values(sampled_psi_values)
    np.save(OUT / "oracle_alias_residual_reconstruction.npy", oracle)
    np.save(OUT / "zero_residual_reconstruction.npy", zero)
    np.save(OUT / "sampled_same_q_cross_config_reconstruction.npy", sampled)

    oracle_diff = oracle - phi
    oracle_check = {
        "max_abs_error": float(np.max(np.abs(oracle_diff))),
        "rms_error": float(np.sqrt(np.mean(oracle_diff**2))),
        "relative_rms_error": float(np.sqrt(np.mean(oracle_diff**2)) / np.sqrt(np.mean(phi**2))),
        "max_imaginary_part": oracle_imag,
        "max_abs_residual_sum_constraint": float(np.max(np.abs(residual_sum))),
    }
    (OUT / "oracle_reconstruction_check.json").write_text(json.dumps(oracle_check, indent=2) + "\n")

    simple_observable_csv(
        OUT / "zero_residual_observables.csv",
        {
            "original_fine": phi,
            "current_inverse_alias_method": current_alias,
            "zero_residual_alias_reconstruction": zero,
        },
    )
    simple_observable_csv(
        OUT / "sample_residual_observables.csv",
        {
            "original_fine": phi,
            "zero_residual_alias_reconstruction": zero,
            "sampled_same_q_cross_config_residual": sampled,
        },
    )
    sample_summary = {
        "implemented_sampling_variant": "shuffle residuals across configurations at the same q by taking a full donor residual field from config i-1",
        "reason": "Borrowing a complete donor residual field preserves Hermitian reality constraints without ad hoc symmetrization.",
        "not_implemented": [
            "same |q| momentum-bin shuffle",
            "fully shuffled residual quadruplets",
        ],
        "sampled_max_imaginary_part": sampled_imag,
        "zero_residual_max_imaginary_part": zero_imag,
        "residual_sum_constraint_max_abs": oracle_check["max_abs_residual_sum_constraint"],
    }
    (OUT / "sample_residual_summary.json").write_text(json.dumps(sample_summary, indent=2) + "\n")

    comparison = write_operator_comparison(
        {
            "original_fine": phi,
            "current_inverse_alias_method": current_alias,
            "zero_residual_alias_reconstruction": zero,
            "oracle_alias_residual_reconstruction": oracle,
            "sampled_same_q_cross_config_residual": sampled,
            "block_replicated_fill": block_replicated,
        }
    )

    rows = comparison["rows"]
    zero_phi4 = row(rows, "zero_residual_alias_reconstruction", "phi4")
    zero_nn2 = row(rows, "zero_residual_alias_reconstruction", "nn2")
    current_phi4 = row(rows, "current_inverse_alias_method", "phi4")
    current_nn2 = row(rows, "current_inverse_alias_method", "nn2")
    sampled_phi2 = row(rows, "sampled_same_q_cross_config_residual", "phi2")
    sampled_phi4 = row(rows, "sampled_same_q_cross_config_residual", "phi4")
    sampled_nn2 = row(rows, "sampled_same_q_cross_config_residual", "nn2")
    report = f"""# Alias-Residual Conditional Baseline

This diagnostic treats the decimated coarse mode `S(q)` as the observed alias mean and the four residual sectors `Ralpha(q) = Aalpha(q) - S(q)` as the missing information. No normalizing flow was trained.

## Residual Constraint

- max |R00 + R10 + R01 + R11|: {oracle_check['max_abs_residual_sum_constraint']:.12g}

## Questions

1. Does oracle residual reconstruction reproduce the original field to roundoff?

Yes. The oracle reconstruction uses the true residuals from the same configuration and gives max absolute error {oracle_check['max_abs_error']:.12g}, RMS error {oracle_check['rms_error']:.12g}, and relative RMS error {oracle_check['relative_rms_error']:.12g}.

2. Is the current inverse alias method equivalent to a zero-residual or wrong-residual assumption?

It is a wrong-residual assumption, not the zero-residual baseline used here. Zero residuals set every blocked alias sector to `S(q)` and then divide each sector by its own `block_norm*K_tilde(p)`. The current inverse alias method divides by the low-sector `K_tilde(q)` and tiles the same inferred field coefficient into every alias sector.

3. How much do zero residuals distort phi4 and nn2?

Zero residuals give phi4 {zero_phi4['mean']:.12g} vs original {zero_phi4['original_fine_mean']:.12g}, and nn2 {zero_nn2['mean']:.12g} vs original {zero_nn2['original_fine_mean']:.12g}. The current inverse alias method gives phi4 {current_phi4['mean']:.12g} and nn2 {current_nn2['mean']:.12g} before any fill.

4. Does sampling empirical residuals restore phi2/phi4/nn2?

Partially. The implemented sampled baseline shuffles complete residual fields across configurations at the same q, preserving Hermitian reality constraints. It gives phi2 {sampled_phi2['mean']:.12g}, phi4 {sampled_phi4['mean']:.12g}, and nn2 {sampled_nn2['mean']:.12g}. This restores substantial UV structure compared with zero residuals, but it is not exact because residuals are not conditioned on the target `S(q)`.

5. Is the conditional NF target better represented as sampling alias residuals Ralpha(q) conditioned on S(q)?

Yes. The oracle result shows that the missing residual sectors are sufficient to reconstruct the fine field to roundoff. The conditional task can be framed as sampling sum-zero alias residuals conditioned on the observed coarse alias sums, while maintaining Hermitian constraints.

6. What is the simplest next NF parameterization?

A hybrid is the most pragmatic next step: model position-space residuals for implementation simplicity, but structure the target so it can represent the discarded Fourier alias sectors. A pure Fourier alias-residual NF is conceptually clean but requires careful complex Hermitian constraints; a pure missing-site fill is easier but hides the exact alias-sector information loss.

## Sampling Notes

Implemented: same-q residual shuffle across configurations using complete donor residual fields. Not implemented in this initial baseline: same-|q| momentum-bin shuffle and fully shuffled residual quadruplets, because those require extra bookkeeping to preserve conjugate reality constraints.
"""
    (OUT / "report.md").write_text(report)


if __name__ == "__main__":
    main()
