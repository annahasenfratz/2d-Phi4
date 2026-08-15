#!/usr/bin/env python3
"""Generate only sweep-0 observables for native L32 blocked to L16 flow samples."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PKG = PROJECT_ROOT / "perfect_blocking_upsampling"
sys.path.insert(0, str(PKG / "src"))
sys.path.insert(0, str(PKG / "scripts"))

from perfect_blocking_upsampling.actions import ActionSpec  # noqa: E402
from perfect_blocking_upsampling.kernels import apply_kernel, load_kernel  # noqa: E402
from run_lam0p2_flow_detail_rethermalization import (  # noqa: E402
    main_measurement_rows,
    rows_for_sweep,
)
from run_lam1p0_l16to32_rqspline_zeroshot import (  # noqa: E402
    build_model_from_checkpoint,
    sample_model_lattice,
    stationary_stats,
)
from run_lam1p0_rqspline_patchwise import g_rows_for_sweep  # noqa: E402
from train_lam1p0_flow_detail_pilot import assemble_psi, load_phi  # noqa: E402


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--native-l32", type=Path, default=Path("data/configs_phi4_2d/lam1p0_kappac0p340301_L32/configs.npz"))
    ap.add_argument("--direct-l16", type=Path, default=Path("data/configs_phi4_2d/lam1p0_kappac0p340301_L16/configs.npz"))
    ap.add_argument("--coarse-mode", choices=("blocked_native", "direct_native"), default="blocked_native")
    ap.add_argument("--checkpoint", type=Path, default=Path("perfect_blocking_upsampling/runs/lam1p0/training/lam1p0_L16to32_newkernel_rqspline_N2000_action_lowtail_from_N2000_bestpatch_20260719T171401Z/checkpoints/checkpoint_best_patch.pt"))
    ap.add_argument("--kernel", type=Path, default=Path("perfect_blocking/perfect_blocking_lam1p0/kernels/final/chosen_kernel.json"))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-chains", type=int, default=None, help="Use the first N native L32 sources.")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=2026072100)
    args = ap.parse_args()

    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    native_l32 = load_phi(PROJECT_ROOT / args.native_l32).astype(np.float32)
    if args.n_chains is not None:
        if not 0 < args.n_chains <= len(native_l32):
            raise ValueError(f"n-chains must be in [1, {len(native_l32)}]")
        native_l32 = native_l32[: args.n_chains]
    kernel, kernel_meta = load_kernel(PROJECT_ROOT / args.kernel)
    if args.coarse_mode == "blocked_native":
        blocked_psi = apply_kernel(native_l32, kernel).astype(np.float32)
        coarse_l16 = blocked_psi[:, 0::2, 0::2].astype(np.float32)
    else:
        direct_l16 = load_phi(PROJECT_ROOT / args.direct_l16).astype(np.float32)
        if args.n_chains is not None:
            direct_l16 = direct_l16[: args.n_chains]
        if len(direct_l16) < len(native_l32):
            raise ValueError("direct L16 source has fewer configurations than requested native L32 reference")
        coarse_l16 = direct_l16[: len(native_l32)]
    source_idx = np.arange(len(native_l32), dtype=np.int64)

    device = torch.device("cpu")
    ckpt = torch.load(PROJECT_ROOT / args.checkpoint, map_location=device, weights_only=False)
    model, load_report = build_model_from_checkpoint(ckpt, lattice_size=16, device=device)
    stats = stationary_stats(ckpt["state"]["stats"], lc=16)
    detail, logq, zmax, logdet = sample_model_lattice(
        model,
        coarse_l16,
        stats,
        batch_size=args.batch_size,
        device=device,
        seed=args.seed,
    )
    psi = assemble_psi(coarse_l16, detail).astype(np.float32)
    phi_upscaled, inverse_meta = __import__(
        "perfect_blocking_upsampling.kernels", fromlist=["inverse_kernel"]
    ).inverse_kernel(psi, kernel)
    phi_upscaled = phi_upscaled.astype(np.float32)

    action = ActionSpec("phi4_nn", 1.0, 0.340301)
    source_label = "blocked_native_L32" if args.coarse_mode == "blocked_native" else "direct_native_L16"
    main_rows = main_measurement_rows(phi_upscaled, action, source_idx, 0, source_label)
    g_rows = g_rows_for_sweep(phi_upscaled, source_idx, 0, source_label)
    per_rows = rows_for_sweep(
        phi_upscaled,
        psi,
        kernel,
        action,
        source_idx,
        0,
        {
            "update_mode": "zero_sweep_flow_sample",
            "detail_update_acceptance": float("nan"),
            "conditional_flow_refreshes": 1,
        },
        source_label,
    )

    write_csv(args.out_dir / "main_per_sweep_measurements.csv", main_rows, list(main_rows[0]))
    write_csv(args.out_dir / "Gk_per_sweep_measurements.csv", g_rows, list(g_rows[0]))
    write_csv(args.out_dir / "per_sweep_observables.csv", per_rows, list(per_rows[0]))
    manifest = {
        "status": "completed",
        "sweep": 0,
        "observable_only": True,
        "native_l32_source": str(args.native_l32),
        "native_l32_count": int(len(native_l32)),
        "coarse_mode": args.coarse_mode,
        "coarse_construction": "native L32 -> supplied kernel -> retained L16 sites" if args.coarse_mode == "blocked_native" else "direct native L16 source",
        "flow_checkpoint": str(args.checkpoint),
        "kernel": str(args.kernel),
        "kernel_metadata": kernel_meta,
        "load_report": load_report,
        "sample_seed": args.seed,
        "max_abs_z": float(np.max(zmax)),
        "logq_mean": float(np.mean(logq)),
        "logdet_mean": float(np.mean(logdet)),
        "inverse_kernel_metadata": inverse_meta,
        "nonfinite_count": int(np.sum(~np.isfinite(phi_upscaled))),
        "files": [
            "main_per_sweep_measurements.csv",
            "Gk_per_sweep_measurements.csv",
            "per_sweep_observables.csv",
        ],
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"out_dir": str(args.out_dir), "n": len(native_l32), "sweep": 0}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
