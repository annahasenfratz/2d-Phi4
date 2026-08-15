"""Run deterministic blocking/reconstruction algebra checks.

This script does not load or train a flow. It only verifies the exact
average-blocking, residual-detail, and reconstruction identities.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from inverse_blocking_flow.haar import (
    average_block,
    block_average,
    detail_to_residual,
    prolong_constant,
    reconstruct_from_average_block,
    residual_to_detail,
)


def run_one(fine_size: int, batch_size: int, dtype: torch.dtype, seed: int) -> dict[str, float | str | int]:
    generator = torch.Generator().manual_seed(seed + fine_size)
    phi_f = torch.randn(batch_size, fine_size, fine_size, dtype=dtype, generator=generator)

    phi_c, d_true = average_block(phi_f)
    phi_rec = reconstruct_from_average_block(phi_c, d_true)
    phi_c_rec = block_average(phi_rec)

    phi0 = prolong_constant(phi_c)
    chi_true = phi_f - phi0
    block_avg_chi = block_average(chi_true)
    chi_rec = detail_to_residual(residual_to_detail(chi_true))

    exact_reconstruction_error = (phi_rec - phi_f).abs().max().item()
    coarse_consistency_error = (phi_c_rec - phi_c).abs().max().item()
    residual_block_average_error = block_avg_chi.abs().max().item()
    detail_roundtrip_error = (chi_rec - chi_true).abs().max().item()
    tolerance = 1e-10 if dtype == torch.float64 else 1e-6
    passed = (
        exact_reconstruction_error < tolerance
        and coarse_consistency_error < tolerance
        and residual_block_average_error < tolerance
        and detail_roundtrip_error < tolerance
    )
    return {
        "fine_size": fine_size,
        "coarse_size": fine_size // 2,
        "batch_size": batch_size,
        "dtype": str(dtype).replace("torch.", ""),
        "tolerance": tolerance,
        "exact_reconstruction_max_abs_error": exact_reconstruction_error,
        "coarse_consistency_max_abs_error": coarse_consistency_error,
        "residual_block_average_max_abs_error": residual_block_average_error,
        "detail_roundtrip_max_abs_error": detail_roundtrip_error,
        "status": "PASS" if passed else "FAIL",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fine-sizes", type=int, nargs="+", default=[16, 32])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    parser.add_argument("--seed", type=int, default=20260614)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    rows = [run_one(size, args.batch_size, dtype, args.seed) for size in args.fine_sizes]

    if args.json:
        print(json.dumps(rows, indent=2))
        return

    for row in rows:
        print(f"fine_size={row['fine_size']} coarse_size={row['coarse_size']} dtype={row['dtype']}")
        print(f"  exact_reconstruction_max_abs_error     {row['exact_reconstruction_max_abs_error']:.12g}")
        print(f"  coarse_consistency_max_abs_error       {row['coarse_consistency_max_abs_error']:.12g}")
        print(f"  residual_block_average_max_abs_error   {row['residual_block_average_max_abs_error']:.12g}")
        print(f"  detail_roundtrip_max_abs_error         {row['detail_roundtrip_max_abs_error']:.12g}")
        print(f"  tolerance                              {row['tolerance']:.12g}")
        print(f"  status                                 {row['status']}")

    if any(row["status"] != "PASS" for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
