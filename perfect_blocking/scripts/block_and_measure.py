#!/usr/bin/env python3
"""Measure native configs or block fine configs and measure the blocked fields."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.blocking import block_configs, load_configs
from scripts.common.gk_observables import compute_Gk_per_config, parse_momenta, write_rows as write_gk_rows
from scripts.common.kernel_io import load_kernel
from scripts.common.observables import add_ensemble_observables, per_config_observables, write_csv
from scripts.common.scan_utils import load_config


DATA_ROOT_NAME = "data"
PERFECT_BLOCKING_ROOT_NAME = "perfect_blocking"


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value_l = value.lower()
    if value_l in {"1", "true", "yes", "y"}:
        return True
    if value_l in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"not a boolean: {value}")


def assert_config_output_policy(path: Path) -> None:
    resolved = path.resolve()
    parts = resolved.parts
    if PERFECT_BLOCKING_ROOT_NAME in parts:
        raise SystemExit(
            "refusing to write field configurations under perfect_blocking/. "
            "Use data/configs_phi4_2d/blocked/ or another data/ subdirectory."
        )
    if DATA_ROOT_NAME not in parts:
        raise SystemExit(
            "refusing to write field configurations outside data/. "
            "Use data/configs_phi4_2d/blocked/ or another data/ subdirectory."
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, help="YAML config with defaults")
    p.add_argument("--mode", choices=["native", "blocked"], default="blocked")
    p.add_argument("--fine-configs", type=Path, help="Input fine config .npz")
    p.add_argument("--configs", type=Path, help="Input native config .npz")
    p.add_argument("--kernel", type=Path, help="Kernel JSON, required for blocked mode")
    p.add_argument(
        "--apply-eta-scale",
        action="store_true",
        help="For base-normalized kernels that explicitly declare kernel_coefficients_include_eta_scale=false, multiply by eta_scale on load.",
    )
    p.add_argument("--output-csv", type=Path, required=True)
    p.add_argument("--gk-output-csv", type=Path)
    p.add_argument("--gk-summary-output-csv", type=Path)
    p.add_argument("--fine-L", type=int)
    p.add_argument("--coarse-L", type=int)
    p.add_argument("--lambda", dest="lam", type=float)
    p.add_argument("--kappa-f", type=float)
    p.add_argument("--kappa-c", type=float)
    p.add_argument("--block-factor", type=int)
    p.add_argument("--max-configs", type=int)
    p.add_argument("--save-blocked-configs", type=str_to_bool, default=False)
    p.add_argument("--blocked-config-output", type=Path)
    p.add_argument("--source-prefix", default="")
    p.add_argument("--ensemble-label", default="")
    p.add_argument("--manifest", type=Path)
    return p.parse_args()


def config_value(args: argparse.Namespace, cfg: dict, cli_name: str, cfg_name: str, default=None):
    value = getattr(args, cli_name)
    if value is not None:
        return value
    return cfg.get(cfg_name, default)


def derive_gk_path(output_csv: Path, summary: bool = False) -> Path:
    name = output_csv.name
    if name.endswith("_all_observables_per_config.csv"):
        stem = name[: -len("_all_observables_per_config.csv")]
    elif name.endswith("_observables_per_config.csv"):
        stem = name[: -len("_observables_per_config.csv")]
    else:
        stem = output_csv.stem
    suffix = "_Gk_summary_per_config.csv" if summary else "_Gk_per_config.csv"
    return output_csv.with_name(stem + suffix)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config) if args.config else {}
    lam = config_value(args, cfg, "lam", "lambda")
    if lam is None:
        raise SystemExit("missing lambda: pass --lambda or set lambda in config")

    block_factor = int(config_value(args, cfg, "block_factor", "block_factor", 2))
    fine_l = config_value(args, cfg, "fine_L", "fine_L")
    coarse_l = config_value(args, cfg, "coarse_L", "coarse_L")
    kappa_f = config_value(args, cfg, "kappa_f", "kappa_f")
    kappa_c = config_value(args, cfg, "kappa_c", "kappa_c")

    if args.mode == "native":
        input_path = args.configs or Path(cfg.get("native_config_source", ""))
        if not input_path:
            raise SystemExit("native mode needs --configs or native_config_source")
        kappa = kappa_c if args.source_prefix.startswith("direct") else (kappa_f or kappa_c)
        phi = load_configs(input_path, max_configs=args.max_configs)
        source_prefix = args.source_prefix or f"native_L{phi.shape[-1]}"
    else:
        input_path = args.fine_configs or Path(cfg.get("fine_config_source", ""))
        if not input_path:
            raise SystemExit("blocked mode needs --fine-configs or fine_config_source")
        kernel_path = args.kernel or Path(cfg.get("kernel", ""))
        if not kernel_path:
            raise SystemExit("blocked mode needs --kernel or kernel in config")
        kernel = load_kernel(kernel_path, apply_eta_scale=args.apply_eta_scale)
        fine_phi = load_configs(input_path, max_configs=args.max_configs)
        phi = block_configs(fine_phi, kernel, block_factor=block_factor)
        if args.save_blocked_configs:
            out_npz = args.blocked_config_output
            if out_npz is None:
                raise SystemExit("--save-blocked-configs true needs --blocked-config-output")
            assert_config_output_policy(out_npz)
            out_npz.parent.mkdir(parents=True, exist_ok=True)
            import numpy as np

            np.savez_compressed(out_npz, phi=phi)
        kappa = kappa_c
        source_prefix = args.source_prefix or f"blocked_L{fine_phi.shape[-1]}_to_L{phi.shape[-1]}"

    if kappa is None:
        raise SystemExit("missing measurement kappa: pass --kappa-f/--kappa-c or set them in config")

    if coarse_l is not None and int(coarse_l) != int(phi.shape[-1]) and args.mode == "blocked":
        raise SystemExit(f"blocked output has L={phi.shape[-1]}, expected coarse_L={coarse_l}")
    if fine_l is not None and args.mode == "native" and args.source_prefix.startswith("native") and int(fine_l) != int(phi.shape[-1]):
        raise SystemExit(f"native input has L={phi.shape[-1]}, expected fine_L={fine_l}")

    rows = per_config_observables(phi, lam=float(lam), kappa=float(kappa), source_prefix=source_prefix)
    add_ensemble_observables(rows)
    write_csv(args.output_csv, rows)

    gk_momenta = parse_momenta(cfg.get("gk_momenta"))
    gk_output_csv = args.gk_output_csv or derive_gk_path(args.output_csv, summary=False)
    gk_summary_output_csv = args.gk_summary_output_csv or derive_gk_path(args.output_csv, summary=True)
    ensemble_label = args.ensemble_label or source_prefix
    gk_rows, gk_summary_rows = compute_Gk_per_config(
        phi,
        momenta=gk_momenta,
        ensemble_label=ensemble_label,
        source_file=str(input_path),
    )
    write_gk_rows(gk_output_csv, gk_rows)
    write_gk_rows(gk_summary_output_csv, gk_summary_rows)

    manifest_path = args.manifest
    if manifest_path:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "mode": args.mode,
            "input_path": str(input_path),
            "output_csv": str(args.output_csv),
            "gk_output_csv": str(gk_output_csv),
            "gk_summary_output_csv": str(gk_summary_output_csv),
            "gk_momenta": gk_momenta,
            "n_configs": int(phi.shape[0]),
            "L": int(phi.shape[-1]),
            "lambda": float(lam),
            "kappa_measurement": float(kappa),
            "kappa_f": None if kappa_f is None else float(kappa_f),
            "kappa_c": None if kappa_c is None else float(kappa_c),
            "block_factor": block_factor,
            "save_blocked_configs": bool(args.save_blocked_configs),
        }
        if args.mode == "blocked":
            payload["kernel"] = str(args.kernel or cfg.get("kernel"))
        manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
