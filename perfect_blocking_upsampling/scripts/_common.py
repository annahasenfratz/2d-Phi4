from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "perfect_blocking_upsampling"
SRC = ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np

from perfect_blocking_upsampling.actions import validate_action_blocks
from perfect_blocking_upsampling.checks import load_manifest, validate_ensemble_manifest, verify_sha256_manifest
from perfect_blocking_upsampling.coarse_refine import adapt_refine_state_for_model, build_refine_model_from_checkpoint
from perfect_blocking_upsampling.io import load_yaml, resolve_path
from perfect_blocking_upsampling.kernels import KernelSpec, load_kernel
from perfect_blocking_upsampling.staged_flow import load_stage_model


def load_config(path: str | Path) -> dict[str, Any]:
    cfg = load_yaml(path)
    cfg["_config_path"] = str(Path(path).resolve())
    return cfg


def format_float_tag(value: float) -> str:
    text = f"{float(value):g}"
    return text.replace(".", "p")


def default_ensemble_path(lambda_: float, kappa: float, lattice_L: int, *, n: int | None = None) -> str:
    suffix = f"_N{int(n)}" if n is not None else ""
    return (
        f"../phi4_phase-diagram/ensembles/"
        f"lam{format_float_tag(lambda_)}_kappa{format_float_tag(kappa)}_L{int(lattice_L)}"
        f"_embedded_wolff_sign_cluster_plus_radial_heatbath{suffix}/configs.npz"
    )


def override_validation_config(
    cfg: dict[str, Any],
    *,
    coarse_L: int | None = None,
    fine_L: int | None = None,
    lambda_c: float | None = None,
    lambda_f: float | None = None,
    kappa_c: float | None = None,
    kappa_f: float | None = None,
    coarse_ensemble: str | Path | None = None,
    fine_reference: str | Path | None = None,
    run_name: str | None = None,
    output_dir: str | Path | None = None,
    frozen_dir: str | Path | None = None,
    kernel_path: str | Path | None = None,
) -> dict[str, Any]:
    out = copy.deepcopy(cfg)
    if coarse_L is not None:
        out["lattice"]["coarse_L"] = int(coarse_L)
    if fine_L is not None:
        out["lattice"]["fine_L"] = int(fine_L)
    if lambda_c is not None:
        out["action"]["coarse"]["lambda"] = float(lambda_c)
    if lambda_f is not None:
        out["action"]["fine"]["lambda"] = float(lambda_f)
    if kappa_c is not None:
        out["action"]["coarse"]["kappa"] = float(kappa_c)
    if kappa_f is not None:
        out["action"]["fine"]["kappa"] = float(kappa_f)
    if coarse_ensemble is not None:
        out["data"]["coarse_ensemble"] = str(coarse_ensemble)
    if fine_reference is not None:
        out["data"]["fine_reference"] = str(fine_reference)
    if run_name is not None:
        out["run_name"] = str(run_name)
    if output_dir is not None:
        out["output_dir"] = str(output_dir)
    if frozen_dir is not None:
        out["checkpoints"]["frozen_dir"] = str(frozen_dir)
    if kernel_path is not None:
        out["kernel"]["path"] = str(kernel_path)
    return out


def config_root(cfg: dict[str, Any]) -> Path:
    return ROOT


def resolve_run_paths(cfg: dict[str, Any]) -> dict[str, Path]:
    base = ROOT
    out = {
        "config": Path(cfg["_config_path"]).resolve(),
        "output_dir": resolve_path(base, cfg["output_dir"]),
        "coarse_ensemble": resolve_path(base, cfg["data"]["coarse_ensemble"]),
        "fine_reference": resolve_path(base, cfg["data"]["fine_reference"]),
        "kernel": resolve_path(base, cfg["kernel"]["path"]),
        "frozen_dir": resolve_path(base, cfg["checkpoints"]["frozen_dir"]),
    }
    return out


def load_actions(cfg: dict[str, Any]):
    return validate_action_blocks(cfg)


def load_kernel_spec(cfg: dict[str, Any]):
    return load_kernel(resolve_run_paths(cfg)["kernel"])


def load_ensembles(cfg: dict[str, Any]):
    paths = resolve_run_paths(cfg)
    coarse = np.load(paths["coarse_ensemble"])["phi"].astype(np.float32)
    fine = np.load(paths["fine_reference"])["phi"].astype(np.float32)
    coarse_manifest = load_manifest(paths["coarse_ensemble"].with_name("manifest.json"))
    fine_manifest = load_manifest(paths["fine_reference"].with_name("manifest.json"))
    return coarse, fine, coarse_manifest, fine_manifest, paths


def verify_frozen_hashes(frozen_dir: Path) -> list[dict[str, Any]]:
    return verify_sha256_manifest(frozen_dir / "sha256_checksums.txt", root=frozen_dir)


def load_frozen_models(cfg: dict[str, Any]):
    paths = resolve_run_paths(cfg)
    coarse_action, fine_action = load_actions(cfg)
    frozen = paths["frozen_dir"]
    refine_ckpt = __import__("torch").load(frozen / "coarse_refine.pt", map_location="cpu")
    refine_model, _ = build_refine_model_from_checkpoint(refine_ckpt, coarse_action, int(cfg["lattice"]["coarse_L"]))
    refine_state = adapt_refine_state_for_model(refine_model, refine_ckpt["model_state"])
    stage_bundles = {}
    lattice_size = int(cfg["lattice"]["coarse_L"])
    for stage, cond_channels in [("edge", 1), ("pair", 2), ("corner", 3)]:
        stage_model, lg, state, ckpt = load_stage_model(
            stage,
            frozen / f"{stage}.pt",
            cond_channels,
            frozen / stage / "local_gaussian_coefficients.npz",
            lattice_size=lattice_size,
        )
        stage_bundles[stage] = (stage_model, lg, state, ckpt)
    return refine_model, refine_state, stage_bundles, coarse_action, fine_action, refine_ckpt
