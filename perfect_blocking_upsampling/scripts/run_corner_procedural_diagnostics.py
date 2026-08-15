#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PKG = PROJECT_ROOT / "perfect_blocking_upsampling"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PKG / "src"))
sys.path.insert(0, str(PKG / "scripts"))

from _common import load_config, resolve_run_paths  # noqa: E402
from perfect_blocking_upsampling.conv_pair import build_procedural_conv_flow  # noqa: E402
from train_gathered_pair_distillation import write_checksums, write_yaml_config  # noqa: E402


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=float) + "\n", encoding="utf-8")


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    shutil.copy2(src, dst)


def copy_coeff(src_root: Path, dst_root: Path, stage: str) -> None:
    copy_file(src_root / stage / "local_gaussian_coefficients.npz", dst_root / stage / "local_gaussian_coefficients.npz")


def export_corner_bundle(config: Path, pair_config: Path, output_dir: Path) -> dict[str, Any]:
    import torch

    cfg = load_config(config)
    pair_cfg = load_config(pair_config)
    source_dir = resolve_run_paths(cfg)["frozen_dir"]
    pair_dir = resolve_run_paths(pair_cfg)["frozen_dir"]
    out = output_dir / "old_pair_corner_procedural_masks_bundle"
    out.mkdir(parents=True, exist_ok=True)

    copy_file(source_dir / "coarse_refine.pt", out / "coarse_refine.pt")
    copy_file(source_dir / "edge.pt", out / "edge.pt")
    copy_file(pair_dir / "pair.pt", out / "pair.pt")
    for stage in ["edge", "pair"]:
        copy_coeff(pair_dir if stage == "pair" else source_dir, out, stage)
    copy_coeff(source_dir, out, "corner")

    old_ckpt = torch.load(source_dir / "corner.pt", map_location="cpu")
    old_masks = old_ckpt["model_state"]["masks"]
    conv_state = {k: v for k, v in old_ckpt["model_state"].items() if k != "masks"}
    cfg_old = old_ckpt["config"]
    model8 = build_procedural_conv_flow(
        cond_channels=3,
        target_channels=1,
        lattice_size=int(cfg["lattice"]["coarse_L"]),
        n_coupling_layers=int(cfg_old["n_coupling_layers"]),
        conv_hidden_channels=int(cfg_old["conv_hidden_channels"]),
        log_scale_bound=float(cfg_old["log_scale_bound"]),
    )
    masks_match = bool(torch.equal(old_masks.cpu(), model8.masks.cpu()))
    missing, unexpected = model8.load_state_dict(conv_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"unexpected procedural corner load keys: missing={missing}, unexpected={unexpected}")
    dummy16 = build_procedural_conv_flow(
        cond_channels=3,
        target_channels=1,
        lattice_size=2 * int(cfg["lattice"]["coarse_L"]),
        n_coupling_layers=int(cfg_old["n_coupling_layers"]),
        conv_hidden_channels=int(cfg_old["conv_hidden_channels"]),
        log_scale_bound=float(cfg_old["log_scale_bound"]),
    )
    proc_ckpt = {
        "model_state": conv_state,
        "config": {
            **cfg_old,
            "flow_arch": "procedural_conv",
            "target_channels": 1,
            "lattice_size": int(cfg["lattice"]["coarse_L"]),
            "mask_export": "procedural shape-parametric masks; old L8 serialized masks removed",
        },
        "stage": "corner",
        "selection": old_ckpt.get("selection"),
        "epoch": old_ckpt.get("epoch"),
        "val_loss": old_ckpt.get("val_loss"),
        "dependency_report": model8.dependency_report(),
        "dummy_larger_volume_dependency_report": dummy16.dependency_report(),
    }
    torch.save(proc_ckpt, out / "corner.pt")
    write_checksums(out)
    bundle_cfg = output_dir / "old_pair_corner_procedural_masks.yaml"
    write_yaml_config(bundle_cfg, cfg, out)
    report = {
        "bundle_config": str(bundle_cfg),
        "bundle_dir": str(out),
        "old_l8_corner_masks_match_procedural": masks_match,
        "corner_dependency_report": model8.dependency_report(),
        "corner_dummy_larger_volume_dependency_report": dummy16.dependency_report(),
        "pair_config": str(pair_config),
        "component_distinction": {
            "edge": "accepted gathered strict finite-radius r_c=3, r_f=6",
            "pair": "old weights with procedural shape-parametric masks; not strict finite-radius",
            "corner": "old weights with procedural shape-parametric masks; not strict finite-radius",
        },
    }
    write_json(output_dir / "corner_procedural_export.json", report)
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=PKG / "outputs" / "gathered_edge_distillation_square_r2_r3_full" / "smoke_square_r3.yaml")
    ap.add_argument("--pair-config", type=Path, default=PKG / "outputs" / "gathered_pair_structure_diagnostics" / "old_pair_procedural_masks.yaml")
    ap.add_argument("--output-dir", type=Path, default=PKG / "outputs" / "procedural_corner_diagnostics")
    ap.add_argument("--n-samples", type=int, default=512)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    export = export_corner_bundle(args.config, args.pair_config, args.output_dir)
    print(f"exported corner procedural bundle: {export['bundle_config']}", flush=True)
    cmd = [
        sys.executable,
        "-u",
        "-B",
        str(PKG / "scripts" / "audit_corner_stage_equivalence.py"),
        "--old-config",
        str(args.pair_config),
        "--candidate-config",
        str(export["bundle_config"]),
        "--output-dir",
        str(args.output_dir / "corner_audit"),
        "--n-samples",
        str(args.n_samples),
    ]
    subprocess.run(cmd, check=True)
    audit = json.loads((args.output_dir / "corner_audit" / "corner_equivalence_audit_report.json").read_text())
    summary = {"export": export, "audit": audit}
    write_json(args.output_dir / "corner_procedural_diagnostics_summary.json", summary)
    dep = export["corner_dependency_report"]
    rec = audit["reconstruction"]
    match = audit["corner_teacher_match"]
    lines = [
        "# Corner/Body Procedural-Mask Diagnostics",
        "",
        f"- old L8 corner masks match procedural masks: `{export['old_l8_corner_masks_match_procedural']}`",
        f"- output RMSE: `{match['output_rmse']:.6g}`",
        f"- output correlation: `{match['output_corr']:.6g}`",
        f"- logq RMSE: `{match['logq_rmse']:.6g}`",
        f"- reconstructed phi RMSE: `{rec['phi_rmse']:.6g}`",
        f"- delta S std: `{rec['delta_S']['std']:.6g}`",
        f"- delta logq std: `{rec['delta_logq']['std']:.6g}`",
        f"- full swap delta logw std: `{rec['full_swap_delta_logw']['std']:.6g}`",
        f"- dependency r_c/r_f: `{dep['coarse_radius']}` / `{dep['fine_radius']}`",
        f"- dummy L16->L32 corner instantiation: `passed`",
        f"- full bundle config: `{export['bundle_config']}`",
        "",
        "## Portability Distinction",
        "- edge: strict finite-radius gathered component (`r_c=3`, `r_f=6`).",
        "- pair/corner: old circular-conv architecture with procedural shape-parametric masks; not strict finite-radius unless separately proven.",
    ]
    (args.output_dir / "corner_procedural_diagnostics_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
