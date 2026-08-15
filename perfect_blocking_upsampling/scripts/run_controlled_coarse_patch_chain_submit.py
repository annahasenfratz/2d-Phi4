#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from _common import format_float_tag  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_SCRIPT = PROJECT_ROOT / "perfect_blocking_upsampling" / "scripts" / "run_controlled_coarse_patch_chain.py"
CONFIG_MAP = {
    (16, 32): PROJECT_ROOT / "perfect_blocking_upsampling" / "outputs" / "shape_parametric_sampler_validation" / "L16_to_L32_smoke" / "L16_to_L32_smoke_config.yaml",
    (32, 64): PROJECT_ROOT / "perfect_blocking_upsampling" / "outputs" / "shape_parametric_sampler_validation" / "L32_to_L64_smoke" / "L32_to_L64_smoke_config.yaml",
}


def configure_stdio() -> None:
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    try:
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass


def latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    if not checkpoint_dir.exists():
        return None
    candidates = sorted(checkpoint_dir.glob("checkpoint_p*_passes*_sweep*.npz"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def checkpoint_completed_sweeps(path: Path | None) -> int | None:
    if path is None or not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    meta_raw = data["meta"]
    if isinstance(meta_raw, np.ndarray):
        meta_text = str(meta_raw.item())
    else:
        meta_text = str(meta_raw)
    meta = json.loads(meta_text)
    return int(meta.get("completed_sweeps", 0))


def build_out_dir(args: argparse.Namespace) -> Path:
    if args.out_dir is not None:
        return args.out_dir
    return PROJECT_ROOT / "perfect_blocking_upsampling" / "outputs" / "shape_parametric_sampler_validation" / (
        f"controlled_coarse_patch_chain_{args.coarse_L}to{args.fine_L}_"
        f"P{args.patch_size}_pass{args.coarse_passes}_detail{args.detail_updates_per_sweep}_"
        f"lam{format_float_tag(args.lambda_)}_kc{format_float_tag(args.kappa_c)}_kf{format_float_tag(args.kappa_f)}"
    )


def default_config_for_pair(coarse_L: int, fine_L: int) -> Path | None:
    return CONFIG_MAP.get((coarse_L, fine_L))


def default_save_sweeps(limit: int) -> list[int]:
    return [0] + list(range(5, int(limit) + 1, 5))


def main() -> int:
    configure_stdio()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--run-name", type=str, default=None)
    ap.add_argument("--coarse-L", type=int, required=True)
    ap.add_argument("--fine-L", type=int, required=True)
    ap.add_argument("--lambda", dest="lambda_", type=float, required=True)
    ap.add_argument("--kappa-c", type=float, required=True)
    ap.add_argument("--kappa-f", type=float, required=True)
    ap.add_argument("--coarse-ensemble", type=Path, default=None)
    ap.add_argument("--fine-reference", type=Path, default=None)
    ap.add_argument("--chains", type=int, default=8)
    ap.add_argument("--sweeps", type=int, default=200)
    ap.add_argument("--patch-size", type=int, required=True)
    ap.add_argument("--coarse-passes", type=int, required=True)
    ap.add_argument("--detail-patch-size", type=int, required=True)
    ap.add_argument("--detail-updates-per-sweep", type=int, required=True)
    ap.add_argument("--site-step-size", type=float, default=0.6)
    ap.add_argument("--latent-beta", type=float, default=0.4)
    ap.add_argument("--save-sweeps", type=int, nargs="+", default=None)
    ap.add_argument("--source-coarse-index", type=int, default=None)
    ap.add_argument("--resume-state-file", type=Path, default=None)
    ap.add_argument("--checkpoint-dir", type=Path, default=None)
    ap.add_argument("--save-state-sweeps", type=int, nargs="*", default=[])
    ap.add_argument("--l16to32-footprint-checkpoint-root", type=Path, default=None)
    ap.add_argument("--l16to32-footprint", type=int, default=11)
    ap.add_argument("--seed", type=int, default=20260703)
    ap.add_argument("--progress-interval", type=int, default=10)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    requested_out_dir = build_out_dir(args)
    old_output_root = PROJECT_ROOT / "perfect_blocking_upsampling" / "outputs" / "shape_parametric_sampler_validation"
    requested_uses_old_root = requested_out_dir.resolve() == old_output_root.resolve() or old_output_root.resolve() in requested_out_dir.resolve().parents
    if args.coarse_L == 16 and args.fine_L == 32 and requested_uses_old_root and os.environ.get("ALLOW_OLD_SHAPE_PARAMETRIC_OUTPUT") != "1":
        raise SystemExit(
            "refusing default L16->L32 old shape_parametric_sampler_validation output; "
            "use launch_controlled_coarse_patch_chain_new_l16to32_kernel.sh or set "
            "ALLOW_OLD_SHAPE_PARAMETRIC_OUTPUT=1 for an intentional old-workflow run"
        )
    if args.save_sweeps is None:
        args.save_sweeps = default_save_sweeps(args.sweeps)

    args.out_dir = build_out_dir(args)
    if args.config is None:
        args.config = default_config_for_pair(args.coarse_L, args.fine_L)
        if args.config is None:
            raise SystemExit(
                f"no default config is registered for L{args.coarse_L}->L{args.fine_L}; pass --config explicitly"
            )
    if args.checkpoint_dir is None:
        args.checkpoint_dir = args.out_dir / "checkpoints"

    if args.resume_state_file is None:
        args.resume_state_file = latest_checkpoint(args.checkpoint_dir)

    if args.save_sweeps is None:
        total_sweeps = args.sweeps
        completed = checkpoint_completed_sweeps(args.resume_state_file)
        if completed is not None:
            total_sweeps += completed
        args.save_sweeps = default_save_sweeps(total_sweeps)

    cmd = [
        sys.executable,
        "-B",
        str(BASE_SCRIPT),
        "--config",
        str(args.config),
        "--coarse-L",
        str(args.coarse_L),
        "--fine-L",
        str(args.fine_L),
        "--lambda",
        str(args.lambda_),
        "--kappa-c",
        str(args.kappa_c),
        "--kappa-f",
        str(args.kappa_f),
        "--chains",
        str(args.chains),
        "--sweeps",
        str(args.sweeps),
        "--settings",
        f"{args.patch_size}:{args.coarse_passes}",
        "--site-step-size",
        str(args.site_step_size),
        "--detail-patch-size",
        str(args.detail_patch_size),
        "--latent-beta",
        str(args.latent_beta),
        "--latent-updates-per-sweep",
        str(args.detail_updates_per_sweep),
        "--save-sweeps",
        *[str(x) for x in args.save_sweeps],
        "--seed",
        str(args.seed),
        "--progress-interval",
        str(args.progress_interval),
        "--out-dir",
        str(args.out_dir),
    ]
    if args.run_name is not None:
        cmd.extend(["--run-name", args.run_name])
    if args.coarse_ensemble is not None:
        cmd.extend(["--coarse-ensemble", str(args.coarse_ensemble)])
    if args.fine_reference is not None:
        cmd.extend(["--fine-reference", str(args.fine_reference)])
    if args.source_coarse_index is not None:
        cmd.extend(["--source-coarse-index", str(args.source_coarse_index)])
    if args.resume_state_file is not None:
        cmd.extend(["--resume-state-file", str(args.resume_state_file)])
    if args.checkpoint_dir is not None:
        cmd.extend(["--checkpoint-dir", str(args.checkpoint_dir)])
    if args.save_state_sweeps:
        cmd.extend(["--save-state-sweeps", *[str(x) for x in args.save_state_sweeps]])
    if args.l16to32_footprint_checkpoint_root is not None:
        cmd.extend([
            "--l16to32-footprint-checkpoint-root",
            str(args.l16to32_footprint_checkpoint_root),
            "--l16to32-footprint",
            str(args.l16to32_footprint),
        ])

    print("launch command:", " ".join(cmd), flush=True)
    print(f"output dir: {args.out_dir}", flush=True)
    if args.resume_state_file is not None:
        print(f"resuming from checkpoint: {args.resume_state_file}", flush=True)
    if args.dry_run:
        return 0

    completed = subprocess.run(cmd, check=False)
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
