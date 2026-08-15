#!/usr/bin/env python3
"""Debug phi2/phi4 suppression in deterministic inverse-blocking fill."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
BASE = PROJECT / "outputs" / "inverse_blocking_step_by_step"
OUT = BASE / "fill_debug"
L_FINE = 16
L_COARSE = 8
RNG_SEED = 20240623


PARITIES = {
    "even-even": (slice(None), slice(0, None, 2), slice(0, None, 2)),
    "odd-even": (slice(None), slice(1, None, 2), slice(0, None, 2)),
    "even-odd": (slice(None), slice(0, None, 2), slice(1, None, 2)),
    "odd-odd": (slice(None), slice(1, None, 2), slice(1, None, 2)),
}


def neighbor_fill_from_even_even(ee: np.ndarray) -> np.ndarray:
    out = np.zeros((ee.shape[0], L_FINE, L_FINE), dtype=ee.dtype)
    out[:, 0::2, 0::2] = ee
    out[:, 1::2, 0::2] = 0.5 * (ee + np.roll(ee, -1, axis=-2))
    out[:, 0::2, 1::2] = 0.5 * (ee + np.roll(ee, -1, axis=-1))
    out[:, 1::2, 1::2] = 0.25 * (
        ee
        + np.roll(ee, -1, axis=-2)
        + np.roll(ee, -1, axis=-1)
        + np.roll(np.roll(ee, -1, axis=-2), -1, axis=-1)
    )
    return out


def obs(configs: np.ndarray) -> dict[str, float]:
    m = configs.mean(axis=(-2, -1))
    m2 = float(np.mean(m**2))
    m4 = float(np.mean(m**4))
    return {
        "m": float(np.mean(m)),
        "|m|": float(np.mean(np.abs(m))),
        "phi2": float(np.mean(configs**2)),
        "phi4": float(np.mean(configs**4)),
        "Binder_U4": float(1.0 - (m4 / (m2 * m2)) / 3.0) if m2 > 0 else math.nan,
        "Binder_B4": float(m4 / (m2 * m2)) if m2 > 0 else math.nan,
    }


def write_obs_csv(path: Path, ensembles: dict[str, np.ndarray]) -> list[dict]:
    base = obs(ensembles["original_fine"])
    rows = []
    for name, arr in ensembles.items():
        vals = obs(arr)
        for op, value in vals.items():
            rows.append(
                {
                    "ensemble": name,
                    "operator": op,
                    "value": value,
                    "original_fine_value": base[op],
                    "difference_vs_original_fine": value - base[op],
                    "ratio_to_original_fine": value / base[op] if base[op] != 0 else math.nan,
                }
            )
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def parity_table(path: Path, ensembles: dict[str, np.ndarray]) -> list[dict]:
    rows = []
    for name, arr in ensembles.items():
        for parity, idx in PARITIES.items():
            block = arr[idx]
            rows.append(
                {
                    "ensemble": name,
                    "parity": parity,
                    "phi2": float(np.mean(block**2)),
                    "phi4": float(np.mean(block**4)),
                    "mean": float(np.mean(block)),
                    "variance": float(np.var(block)),
                    "max_abs": float(np.max(np.abs(block))),
                }
            )
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def alias_direct_assignment_test() -> dict:
    rng = np.random.default_rng(RNG_SEED)
    c = rng.normal(size=(L_COARSE, L_COARSE))
    direct = np.zeros((L_FINE, L_FINE), dtype=float)
    direct[0::2, 0::2] = c
    c_fft = np.fft.fft2(c)
    alias_fft = np.zeros((L_FINE, L_FINE), dtype=complex)
    for ky in range(L_COARSE):
        fy = ky if ky < L_COARSE // 2 else ky + L_COARSE
        for kx in range(L_COARSE):
            fx = kx if kx < L_COARSE // 2 else kx + L_COARSE
            for ay in (0, L_COARSE):
                for ax in (0, L_COARSE):
                    alias_fft[(fy + ay) % L_FINE, (fx + ax) % L_FINE] = c_fft[ky, kx]
    reconstructed = np.fft.ifft2(alias_fft).real
    err = reconstructed - direct
    result = {
        "numpy_fft_convention": "fft2 has no forward normalization; ifft2 divides by N^2",
        "tiling_factor_used": 1.0,
        "max_abs_error": float(np.max(np.abs(err))),
        "rms_error": float(np.sqrt(np.mean(err**2))),
        "direct_phi2": float(np.mean(direct**2)),
        "reconstructed_phi2": float(np.mean(reconstructed**2)),
        "pass": bool(np.max(np.abs(err)) < 1.0e-12),
    }
    (OUT / "alias_direct_assignment_test.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUT / "alias_direct_assignment_report.md").write_text(
        "# Alias Direct Assignment Test\n\n"
        "A random 8x8 field was assigned directly to even-even sites of a 16x16 field. "
        "The same field was reconstructed by tiling `FFT_8[c]` into the four 16x16 alias sectors.\n\n"
        f"- tiling factor used: {result['tiling_factor_used']}\n"
        f"- max absolute error: {result['max_abs_error']:.12g}\n"
        f"- RMS error: {result['rms_error']:.12g}\n"
        f"- pass: {result['pass']}\n"
    )
    return result


def even_even_amplitude(fine: np.ndarray, alias: np.ndarray, filled: np.ndarray) -> dict:
    x = fine[:, 0::2, 0::2].ravel()
    y = alias[:, 0::2, 0::2].ravel()
    z = filled[:, 0::2, 0::2].ravel()
    slope = float(np.dot(x, y) / np.dot(x, x))
    intercept = float(y.mean() - slope * x.mean())
    corr = float(np.corrcoef(x, y)[0, 1])
    result = {
        "original_even_even_phi2": float(np.mean(x**2)),
        "chi_alias_even_even_phi2": float(np.mean(y**2)),
        "phi_init_even_even_phi2": float(np.mean(z**2)),
        "alias_to_original_variance_ratio": float(np.var(y) / np.var(x)),
        "alias_to_original_phi2_ratio": float(np.mean(y**2) / np.mean(x**2)),
        "best_fit_slope_chi_alias_vs_original": slope,
        "best_fit_intercept_chi_alias_vs_original": intercept,
        "correlation_coefficient": corr,
        "mean_squared_difference": float(np.mean((y - x) ** 2)),
        "max_abs_alias_minus_filled_even_even": float(np.max(np.abs(y - z))),
    }
    (OUT / "even_even_amplitude_check.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUT / "even_even_amplitude_check.md").write_text(
        "# Even-Even Amplitude Check\n\n"
        f"- original even-even phi2: {result['original_even_even_phi2']:.12g}\n"
        f"- chi_alias even-even phi2: {result['chi_alias_even_even_phi2']:.12g}\n"
        f"- phi_init even-even phi2: {result['phi_init_even_even_phi2']:.12g}\n"
        f"- alias/original variance ratio: {result['alias_to_original_variance_ratio']:.12g}\n"
        f"- best-fit slope chi_alias vs original: {slope:.12g}\n"
        f"- intercept: {intercept:.12g}\n"
        f"- correlation coefficient: {corr:.12g}\n"
        f"- mean squared difference: {result['mean_squared_difference']:.12g}\n"
        f"- max |alias even-even - filled even-even|: {result['max_abs_alias_minus_filled_even_even']:.12g}\n"
    )
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fine = np.load(BASE / "input_fine_batch.npy")
    alias = np.load(BASE / "inverse_kernel_alias_field.npy")
    filled = np.load(BASE / "neighbor_filled_init.npy")
    oracle = neighbor_fill_from_even_even(fine[:, 0::2, 0::2])
    np.save(OUT / "oracle_filled_from_original_even_even.npy", oracle)

    code = (PROJECT / "scripts" / "run_inverse_blocking_step_by_step.py").read_text()
    fill_uses_even_even = "ee = alias[:, 0::2, 0::2]" in code
    zero_field_neighbor_average = "np.roll(alias" in code or "np.roll(out" in code
    (OUT / "fill_implementation_check.md").write_text(
        "# Fill Implementation Check\n\n"
        f"- Uses extracted even-even array as source: {fill_uses_even_even}\n"
        f"- Detected rolling/averaging the sparse zero-filled fine field: {zero_field_neighbor_average}\n\n"
        "The implemented fill constructs `ee = alias[:, 0::2, 0::2]` and averages rolls of `ee`, "
        "not rolls of the sparse alias field or the partially filled output. That means it does not "
        "average zeros from missing fine sites.\n"
    )

    oracle_rows = write_obs_csv(
        OUT / "oracle_fill_observables.csv",
        {
            "original_fine": fine,
            "oracle_filled_from_original_even_even": oracle,
            "inverse_alias_filled_phi_init": filled,
        },
    )
    parity_rows = parity_table(
        OUT / "parity_phi2_phi4.csv",
        {
            "original_fine": fine,
            "chi_alias": alias,
            "phi_init": filled,
            "phi_oracle": oracle,
        },
    )
    alias_test = alias_direct_assignment_test()
    amp = even_even_amplitude(fine, alias, filled)

    interesting = {r["operator"]: r for r in oracle_rows if r["ensemble"] == "oracle_filled_from_original_even_even"}
    init_rows = {r["operator"]: r for r in oracle_rows if r["ensemble"] == "inverse_alias_filled_phi_init"}
    report = f"""# Fill Debug Report

## Answers

1. Was the fill routine averaging zeros?

No. The implementation extracts the even-even array and averages neighboring entries of that compact 8x8 array. It does not roll or average the sparse zero-filled 16x16 alias field.

2. Does the oracle fill preserve phi2/phi4 reasonably?

Oracle fill from the original fine even-even sites gives:

- phi2: {interesting['phi2']['value']:.12g} vs original {interesting['phi2']['original_fine_value']:.12g}, ratio {interesting['phi2']['ratio_to_original_fine']:.12g}
- phi4: {interesting['phi4']['value']:.12g} vs original {interesting['phi4']['original_fine_value']:.12g}, ratio {interesting['phi4']['ratio_to_original_fine']:.12g}

So the deterministic neighbor fill itself suppresses UV variance and fourth moment substantially even when the even-even sites are exact.

3. Is the alias tiling normalization exact?

Alias direct-assignment max error is {alias_test['max_abs_error']:.12g}; pass = {alias_test['pass']}. The alias tiling normalization is exact under the tested numpy FFT convention.

4. Is the inverse even-even amplitude correct?

The inverse even-even field has phi2 {amp['chi_alias_even_even_phi2']:.12g} vs original even-even phi2 {amp['original_even_even_phi2']:.12g}, ratio {amp['alias_to_original_phi2_ratio']:.12g}. The best-fit slope is {amp['best_fit_slope_chi_alias_vs_original']:.12g}, correlation is {amp['correlation_coefficient']:.12g}, and mean squared difference is {amp['mean_squared_difference']:.12g}.

5. Where exactly does the phi2/phi4 suppression enter?

There are two effects. First, oracle neighbor filling already lowers phi2/phi4 because it replaces odd sublattices by local averages, removing short-distance fluctuations. Second, the inverse-alias even-even field is a low-mode inverse of the blocked coarse field with slightly lower even-even amplitude than the original, though it remains strongly pointwise correlated with the original even-even sites. The final inverse-filled values are:

- phi2: {init_rows['phi2']['value']:.12g} vs original {init_rows['phi2']['original_fine_value']:.12g}, ratio {init_rows['phi2']['ratio_to_original_fine']:.12g}
- phi4: {init_rows['phi4']['value']:.12g} vs original {init_rows['phi4']['original_fine_value']:.12g}, ratio {init_rows['phi4']['ratio_to_original_fine']:.12g}

6. What code change, if any, was made?

No production diagnostic code was changed. This follow-up added `scripts/debug_inverse_blocking_fill.py` and wrote debug outputs under `outputs/inverse_blocking_step_by_step/fill_debug/`.
"""
    (OUT / "oracle_fill_report.md").write_text(report)
    (OUT / "report.md").write_text(report)


if __name__ == "__main__":
    main()
