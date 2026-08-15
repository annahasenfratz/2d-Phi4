from __future__ import annotations

import hashlib
import platform
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

from .config_io import write_config, write_json


PROJECT_ROOT = Path(__file__).resolve().parents[3]
UPSAMPLING_ROOT = PROJECT_ROOT / "perfect_blocking_upsampling"


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def path_sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def lambda_slug(value: Any) -> str:
    text = str(value).replace(".", "p")
    return "lam" + text if not text.startswith("lam") else text


def kappa_slug(value: Any) -> str:
    return str(value).replace(".", "p")


def controlled_patch_run_id(config: dict[str, Any], timestamp: str) -> str:
    lf = int(config["L_f"])
    n_chains = int(config["n_chains"])
    patch = config.get("patch", {})
    kf = kappa_slug(config["kappa_f"])
    kc = kappa_slug(config["kappa_c"])
    flow = str(config.get("flow_type", "flow"))
    mode = str(config.get("mode", ""))
    detail_patch = int(patch.get("detail_patch_size", 0))
    detail_passes = int(patch.get("detail_passes", 0))
    if mode == "patchwise_detail_only":
        return f"prod_do_bL{lf}_N{n_chains}_kf{kf}_kc{kc}_Pd{detail_patch}x{detail_passes}_{flow}_{timestamp}"
    if mode == "patchwise_coarse_detail":
        coarse_patch = int(patch.get("coarse_patch_size", 0))
        coarse_passes = int(patch.get("coarse_passes", 0))
        return (
            f"prod_cd_bL{lf}_N{n_chains}_kf{kf}_kc{kc}"
            f"_Pc{coarse_patch}x{coarse_passes}_Pd{detail_patch}x{detail_passes}_{flow}_{timestamp}"
        )
    return ""


def ensure_kappa_tags(run_id: str, config: dict[str, Any]) -> str:
    kf = f"kf{kappa_slug(config['kappa_f'])}"
    kc = f"kc{kappa_slug(config['kappa_c'])}"
    missing = [tag for tag in (kf, kc) if tag not in run_id]
    if not missing:
        return run_id
    return f"{run_id}_{'_'.join(missing)}"


def build_run_id(config: dict[str, Any], timestamp: str | None = None) -> str:
    stamp = timestamp or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    if str(config.get("run_id_style", "")) == "controlled_patch":
        run_id = controlled_patch_run_id(config, stamp)
        if run_id:
            return run_id
    lam = lambda_slug(config["lambda"])
    lf = int(config["L_f"])
    lc = int(config["L_c"])
    seed = int(config["random_seed"])
    mode = str(config.get("mode", "flow_detail"))
    return (
        f"{lam}_L{lc}to{lf}_kf{kappa_slug(config['kappa_f'])}"
        f"_kc{kappa_slug(config['kappa_c'])}_seed{seed}_{mode}_{stamp}"
    )


def default_run_dir(config: dict[str, Any], run_id: str) -> Path:
    if config.get("output_root"):
        return PROJECT_ROOT / str(config["output_root"]) / run_id
    lam = lambda_slug(config["lambda"])
    return UPSAMPLING_ROOT / "runs" / lam / run_id


def prepare_run_directory(config: dict[str, Any], run_dir: Path, command: str) -> dict[str, Any]:
    for sub in ["logs", "checkpoints", "observables", "plots", "summaries", "debug"]:
        (run_dir / sub).mkdir(parents=True, exist_ok=True)

    resolved = dict(config)
    resolved["run_dir"] = str(run_dir)
    resolved.setdefault("resume", {"enabled": False, "checkpoint_path": None})
    write_config(run_dir / "run_config.yaml", resolved)

    kernel_path = PROJECT_ROOT / str(config["kernel_path"])
    manifest = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": command,
        "git_commit": git_commit(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "run_dir": str(run_dir),
        "kernel_path": str(config["kernel_path"]),
        "kernel_sha256": path_sha256(kernel_path),
        "fine_config_source": str(config.get("fine_config_source", "")),
        "coarse_config_source": str(config.get("coarse_config_source", "")),
        "raw_config_policy": "raw field configurations stay under data/configs_phi4_2d; run directories store metadata, checkpoints, observables, plots, summaries, and logs only",
    }
    write_json(run_dir / "submit_manifest.txt", manifest)
    write_json(
        run_dir / "status.json",
        {
            "status": "prepared",
            "current_sweep": 0,
            "latest_checkpoint": None,
            "run_config": str(run_dir / "run_config.yaml"),
        },
    )
    return manifest
