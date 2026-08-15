#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from common.config_io import load_config, require_keys, write_json  # noqa: E402
from common.kernels import validate_eta_included  # noqa: E402
from common.run_manifest import build_run_id, default_run_dir, ensure_kappa_tags, prepare_run_directory  # noqa: E402


REQUIRED = [
    "lambda",
    "kappa_f",
    "kappa_c",
    "eta",
    "L_f",
    "L_c",
    "block_factor",
    "fine_config_source",
    "coarse_config_source",
    "kernel_path",
    "kernel_coefficients_include_eta_scale",
    "mode",
    "patch",
    "random_seed",
    "n_chains",
    "n_sweeps",
    "checkpoint_every",
    "measure_every",
    "save_every",
]


def parse_scalar(value: str):
    lowered = value.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def apply_override(config: dict, override: str) -> None:
    if "=" not in override:
        raise SystemExit(f"--set expects key=value, got {override!r}")
    key, value = override.split("=", 1)
    if key == "sweeps":
        key = "n_sweeps"
    target = config
    parts = key.split(".")
    for part in parts[:-1]:
        if part not in target or not isinstance(target[part], dict):
            target[part] = {}
        target = target[part]
    target[parts[-1]] = parse_scalar(value)


def main() -> int:
    ap = argparse.ArgumentParser(description="Prepare or launch a flow/detail/coarse-detail upscaling run.")
    ap.add_argument("--config", type=Path, required=True, help="YAML run configuration.")
    ap.add_argument("--run-dir", type=Path, default=None, help="Override output run directory.")
    ap.add_argument("--run-id", default=None, help="Override generated run id.")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="Override a YAML field. Use dotted keys for nested fields, e.g. patch.detail_passes=10. Alias: sweeps=n maps to n_sweeps.")
    ap.add_argument("--execute", action="store_true", help="Call the configured legacy_runner after preparing the run directory.")
    args, passthrough = ap.parse_known_args()

    config = load_config(args.config)
    for override in args.set:
        apply_override(config, override)
    require_keys(config, REQUIRED)
    run_id = args.run_id or build_run_id(config)
    if args.run_id and str(config.get("run_id_style", "")) == "controlled_patch":
        run_id = ensure_kappa_tags(run_id, config)
    run_dir = args.run_dir or default_run_dir(config, run_id)
    command = " ".join(shlex.quote(x) for x in sys.argv)

    eta_scale = float(config.get("eta_scale_numeric", 2.0 ** (float(config["eta"]) / 2.0)))
    validate_eta_included(PROJECT_ROOT / str(config["kernel_path"]), eta_scale)
    prepare_run_directory(config, run_dir, command)

    runner = config.get("legacy_runner")
    if args.execute:
        if not runner:
            raise SystemExit("configuration has no legacy_runner; prepared run directory but cannot execute")
        cmd = [sys.executable, "-B", str(PROJECT_ROOT / str(runner)), "--run-dir", str(run_dir), *passthrough]
        write_json(run_dir / "status.json", {"status": "launching", "current_sweep": 0, "latest_checkpoint": None, "command": cmd})
        with (run_dir / "logs" / "run.log").open("a", encoding="utf-8") as log:
            return subprocess.call(cmd, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT)

    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
