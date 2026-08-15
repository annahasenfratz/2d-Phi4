#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PKG = PROJECT_ROOT / "perfect_blocking_upsampling"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PKG / "src"))
sys.path.insert(0, str(PKG / "scripts"))

from _common import load_config, load_frozen_models, resolve_run_paths  # noqa: E402
from perfect_blocking_upsampling.conv_pair import build_procedural_conv_flow  # noqa: E402
from train_gathered_pair_distillation import write_checksums, write_yaml_config  # noqa: E402


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=float) + "\n", encoding="utf-8")


def copy_bundle_inputs(source_dir: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name in ["coarse_refine.pt", "edge.pt", "corner.pt"]:
        if (out / name).exists():
            (out / name).unlink()
        shutil.copy2(source_dir / name, out / name)
    for stage in ["edge", "pair", "corner"]:
        dst = out / stage
        dst.mkdir(exist_ok=True)
        coeff = dst / "local_gaussian_coefficients.npz"
        if coeff.exists():
            coeff.unlink()
        shutil.copy2(source_dir / stage / "local_gaussian_coefficients.npz", coeff)


def export_old_pair_procedural_bundle(config: Path, output_dir: Path) -> dict[str, Any]:
    import torch

    cfg = load_config(config)
    source_dir = resolve_run_paths(cfg)["frozen_dir"]
    out = output_dir / "old_pair_procedural_masks_bundle"
    copy_bundle_inputs(source_dir, out)
    old_ckpt = torch.load(source_dir / "pair.pt", map_location="cpu")
    old_masks = old_ckpt["model_state"]["masks"]
    conv_state = {k: v for k, v in old_ckpt["model_state"].items() if k != "masks"}
    cfg_old = old_ckpt["config"]
    model8 = build_procedural_conv_flow(
        cond_channels=2,
        target_channels=1,
        lattice_size=int(cfg["lattice"]["coarse_L"]),
        n_coupling_layers=int(cfg_old["n_coupling_layers"]),
        conv_hidden_channels=int(cfg_old["conv_hidden_channels"]),
        log_scale_bound=float(cfg_old["log_scale_bound"]),
    )
    masks_match = bool(torch.equal(old_masks.cpu(), model8.masks.cpu()))
    missing, unexpected = model8.load_state_dict(conv_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"unexpected procedural load keys: missing={missing}, unexpected={unexpected}")
    dummy16 = build_procedural_conv_flow(
        cond_channels=2,
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
        "stage": "pair",
        "selection": old_ckpt.get("selection"),
        "epoch": old_ckpt.get("epoch"),
        "val_loss": old_ckpt.get("val_loss"),
        "dependency_report": model8.dependency_report(),
        "dummy_larger_volume_dependency_report": dummy16.dependency_report(),
    }
    torch.save(proc_ckpt, out / "pair.pt")
    write_checksums(out)
    bundle_cfg = output_dir / "old_pair_procedural_masks.yaml"
    write_yaml_config(bundle_cfg, cfg, out)
    report = {
        "bundle_config": str(bundle_cfg),
        "bundle_dir": str(out),
        "old_l8_masks_match_procedural": masks_match,
        "dependency_report": model8.dependency_report(),
        "dummy_larger_volume_dependency_report": dummy16.dependency_report(),
    }
    write_json(output_dir / "old_pair_procedural_export.json", report)
    return report


def run_audit(pair_config: Path, output_dir: Path, n_samples: int) -> dict[str, Any]:
    print(f"running pair equivalence audit: {pair_config}", flush=True)
    cmd = [
        sys.executable,
        "-u",
        "-B",
        str(PKG / "scripts" / "audit_pair_stage_equivalence.py"),
        "--pair-config",
        str(pair_config),
        "--output-dir",
        str(output_dir),
        "--n-samples",
        str(n_samples),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    report = json.loads((output_dir / "pair_equivalence_audit_report.json").read_text())
    a = report.get("action_logq_decomposition", report.get("action_vs_logq_decomposition", {}))
    p = report["pair_teacher_match"]
    print(
        f"finished audit {pair_config.name}: rmse={p['output_rmse']:.6g} "
        f"logq_rmse={p['logq_rmse']:.6g} deltaS_std={a.get('delta_S', {}).get('std', float('nan')):.6g}",
        flush=True,
    )
    return report


def metric_row(label: str, kind: str, dep: dict[str, Any], audit: dict[str, Any], dummy: dict[str, Any] | None = None) -> dict[str, Any]:
    p = audit["pair_teacher_match"]
    a = audit.get("action_logq_decomposition", audit.get("action_vs_logq_decomposition"))
    if a is None:
        raise KeyError("audit missing action/logq decomposition")
    pcn = audit["pair_pcn_local_diagnostic"]
    return {
        "label": label,
        "kind": kind,
        "pair_output_rmse": p["output_rmse"],
        "pair_output_corr": p["output_corr"],
        "logq_rmse": p["logq_rmse"],
        "delta_S_std": a["delta_S"]["std"],
        "delta_logq_std": a["delta_total_logq"]["std"],
        "full_swap_delta_logw_std": a["delta_logw"]["std"],
        "pair_only_pcn_delta_logw_std": pcn["portable_pair_only_delta_logw"]["std"],
        "pair_only_pcn_acceptance_estimate": pcn["portable_acceptance_estimate"],
        "r_c": dep.get("coarse_radius"),
        "r_f": dep.get("fine_radius"),
        "dependency_metric": dep.get("metric"),
        "dummy_l16_to_l32_instantiation": "passed" if dummy else "not_reported",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=PKG / "outputs" / "gathered_edge_distillation_square_r2_r3_full" / "smoke_square_r3.yaml")
    ap.add_argument("--output-dir", type=Path, default=PKG / "outputs" / "gathered_pair_structure_diagnostics")
    ap.add_argument("--n-samples", type=int, default=512)
    ap.add_argument("--skip-old-procedural", action="store_true")
    ap.add_argument("--candidate", action="append", default=[], help="label=path/to/pair_config.yaml for extra gathered diagnostics")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    exports: dict[str, Any] = {}
    if not args.skip_old_procedural:
        proc = export_old_pair_procedural_bundle(args.config, args.output_dir)
        exports["old_pair_procedural_masks"] = proc
        audit = run_audit(Path(proc["bundle_config"]), args.output_dir / "old_pair_procedural_masks_audit", args.n_samples)
        rows.append(metric_row("old_pair_procedural_masks", "old_weights_procedural_masks", proc["dependency_report"], audit, proc["dummy_larger_volume_dependency_report"]))

    for spec in args.candidate:
        if "=" not in spec:
            raise ValueError("--candidate must be label=path")
        label, raw_path = spec.split("=", 1)
        pair_config = Path(raw_path)
        audit = run_audit(pair_config, args.output_dir / f"{label}_audit", args.n_samples)
        _, _, stages, _, _, _ = load_frozen_models(load_config(pair_config))
        ckpt = stages["pair"][3]
        rows.append(metric_row(label, "candidate", ckpt.get("dependency_report", {}), audit, ckpt.get("dummy_larger_volume_dependency_report")))

    summary = {"rows": rows, "exports": exports}
    write_json(args.output_dir / "pair_structure_diagnostics_summary.json", summary)
    lines = ["# Pair Structure Diagnostics", ""]
    lines.append("| diagnostic | kind | r_c | r_f | pair RMSE | corr | logq RMSE | delta S std | delta logq std | full swap std | pair pCN std | pCN acc | dummy |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for r in rows:
        lines.append(
            f"| {r['label']} | {r['kind']} | {r['r_c']} | {r['r_f']} | "
            f"{r['pair_output_rmse']:.6g} | {r['pair_output_corr']:.6g} | {r['logq_rmse']:.6g} | "
            f"{r['delta_S_std']:.6g} | {r['delta_logq_std']:.6g} | {r['full_swap_delta_logw_std']:.6g} | "
            f"{r['pair_only_pcn_delta_logw_std']:.6g} | {r['pair_only_pcn_acceptance_estimate']:.6g} | {r['dummy_l16_to_l32_instantiation']} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "- `old_pair_procedural_masks` removes serialized L8 masks and regenerates checkerboard masks from lattice shape while reusing old convolution weights.",
            "- Expanded gathered radii are diagnostics only when `r_c >= 4`; they are not strict-local L8->L16 candidates.",
            "- Corner/body was not trained.",
        ]
    )
    (args.output_dir / "pair_structure_diagnostics_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, default=float), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
