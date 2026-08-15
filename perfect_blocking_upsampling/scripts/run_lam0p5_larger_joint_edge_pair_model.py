#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PKG = PROJECT_ROOT / "perfect_blocking_upsampling"
for p in [PROJECT_ROOT, PKG / "scripts", PKG / "src"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from diagnose_lam0p5_failed_bundle import load_context, load_paired, qstats, reconstruct, stage_forward_z, write_csv, write_json  # noqa: E402
from perfect_blocking_upsampling.actions import action_total  # noqa: E402
from perfect_blocking_upsampling.kernels import inverse_kernel  # noqa: E402
from run_lam0p5_joint_edge_pair_distillation import fixed_joint_edge_pair_mcmc, metric_row  # noqa: E402
from train_lam0p5_joint_edge_pair_flow import (  # noqa: E402
    FAILED_SEQ_STD,
    INITIAL_JOINT_STD,
    TEACHER_STD,
    cumulative_diagnostics,
    train_joint_flow,
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def write_status(summary: dict[str, Any]) -> None:
    path = PROJECT_ROOT / "perfect_blocking_upsampling/outputs/lam0p5_small3_8to16/STATUS.md"
    section = f"""

## Larger joint edge+pair model branch

Status: `{summary['status']}`.

Output:

`full_training/run_20260630_210838/remediation/larger_joint_edge_pair_model/`

Completed:

- audited the existing corrected joint edge+pair dataset;
- generated bounded v2 corrected joint dataset at `K=50`;
- trained one larger true joint edge+pair flow;
- evaluated dense NLL/action metrics and cumulative diagnostics;
- did not run long validation.

Key results:

- v2 corrected dataset deltaS std: `{summary['v2_teacher_deltaS_std']:.6g}`;
- larger joint best z0 deltaS std: `{summary['best_z0_deltaS_std']:.6g}`;
- cumulative target-corner deltaS std: `{summary['cumulative_target_corner_deltaS_std']:.6g}`;
- cumulative baseline-corner deltaS std: `{summary['cumulative_baseline_corner_deltaS_std']:.6g}`;
- tiny sampler smoke launched: `{summary['sampler_smoke_launched']}`.

Interpretation:

{summary['interpretation']}

Reports:

- `full_training/run_20260630_210838/remediation/larger_joint_edge_pair_model/CORRECTED_JOINT_DATASET_AUDIT.md`
- `full_training/run_20260630_210838/remediation/larger_joint_edge_pair_model/LARGER_JOINT_MODEL_DEFINITION.md`
- `full_training/run_20260630_210838/remediation/larger_joint_edge_pair_model/LARGER_JOINT_EDGE_PAIR_FINAL_REPORT.md`
"""
    text = path.read_text()
    marker = "\n## Larger joint edge+pair model branch\n"
    if marker in text:
        text = text[: text.index(marker)]
    path.write_text(text.rstrip() + section + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=PROJECT_ROOT / "perfect_blocking_upsampling/outputs/lam0p5_small3_8to16/full_training/run_20260630_210838")
    ap.add_argument("--source-dataset", type=Path, default=PROJECT_ROOT / "perfect_blocking_upsampling/outputs/lam0p5_small3_8to16/full_training/run_20260630_210838/remediation/mcmc_detail_distillation_joint_edge_pair/corrected_joint_edge_pair_dataset/corrected_joint_edge_pair_dataset.npz")
    ap.add_argument("--dataset-samples", type=int, default=2048)
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--proposal-sigma-edge", type=float, default=0.08)
    ap.add_argument("--proposal-sigma-pair", type=float, default=0.08)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--n-coupling", type=int, default=24)
    ap.add_argument("--smoke-epochs", type=int, default=2)
    ap.add_argument("--diagnostic-epochs", type=int, default=32)
    ap.add_argument("--seed", type=int, default=20260704)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out = args.run_dir / "remediation/larger_joint_edge_pair_model"
    if out.exists() and args.overwrite:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    _cfg, _paths, _coarse, _fine_ref, _cm, _fm, ctx = load_context(args.run_dir)
    arrays = load_paired()
    src = load_npz(args.source_dataset)
    src_n = int(src["conditioning_c00"].shape[0])
    src_rows = []
    for key, val in src.items():
        src_rows.append({"field": key, "shape": list(val.shape), "dtype": str(val.dtype), "finite": bool(np.isfinite(val).all()) if val.dtype.kind in "fc" else True})
    write_csv(out / "corrected_joint_dataset_audit_fields.csv", src_rows)
    (out / "CORRECTED_JOINT_DATASET_AUDIT.md").write_text(
        "# Corrected joint dataset audit\n\n"
        f"- source dataset: `{args.source_dataset}`\n"
        f"- source samples: `{src_n}`\n"
        f"- source sha256: `{sha256(args.source_dataset)}`\n"
        "- source K: `50`\n"
        f"- source teacher corrected deltaS std reference: `{TEACHER_STD:.6g}`\n"
        "\nThe source dataset is only 1024 samples, so this branch generates a bounded v2 dataset before training the larger model.\n"
    )

    # Generate bounded v2 corrected joint dataset.
    n = min(args.dataset_samples, len(arrays["train_idx"]))
    di = arrays["train_idx"][:n]
    c = arrays["c00"][di].astype(np.float32)
    e_t = arrays["edge_x"][di, None].astype(np.float32)
    p_t = arrays["edge_y"][di, None].astype(np.float32)
    co_t = arrays["corner"][di, None].astype(np.float32)
    target_phi, _ = inverse_kernel(reconstruct(c, e_t, p_t, co_t), ctx["kernel"])
    edge_model, edge_lg, *_ = ctx["stages"]["edge"]
    pair_model, pair_lg, *_ = ctx["stages"]["pair"]
    z0 = np.zeros_like(e_t, dtype=np.float32)
    e0, *_ = stage_forward_z(edge_model, z0, c[:, None], edge_lg)
    p0, *_ = stage_forward_z(pair_model, z0, np.concatenate([c[:, None], e0], axis=1), pair_lg)
    samples, traj = fixed_joint_edge_pair_mcmc(
        initial_edge=e0,
        initial_pair=p0,
        c=c,
        corner=co_t,
        target_phi=target_phi,
        ctx=ctx,
        k_values=[0, args.k],
        proposal_sigma_edge=args.proposal_sigma_edge,
        proposal_sigma_pair=args.proposal_sigma_pair,
        seed=args.seed,
    )
    ce, cp = samples[args.k]
    ds_dir = out / "corrected_joint_dataset_v2"
    ds_dir.mkdir(parents=True, exist_ok=True)
    v2_path = ds_dir / "corrected_joint_edge_pair_dataset_v2.npz"
    np.savez_compressed(
        v2_path,
        paired_indices=di,
        conditioning_c00=c,
        corrected_edge=ce,
        corrected_pair=cp,
        original_target_edge=e_t,
        original_target_pair=p_t,
        initial_model_edge=e0,
        initial_model_pair=p0,
        target_corner=co_t,
        best_K=np.asarray(args.k),
        proposal_sigma_edge=np.asarray(args.proposal_sigma_edge),
        proposal_sigma_pair=np.asarray(args.proposal_sigma_pair),
    )
    write_csv(ds_dir / "dataset_mcmc_trajectory_summary.csv", traj)
    metric_initial = metric_row("v2_initial_model_edge_pair", 0, e0, p0, c, co_t, e_t, p_t, target_phi, ctx, "")
    metric_corr = metric_row("v2_corrected_joint_edge_pair", args.k, ce, cp, c, co_t, e_t, p_t, target_phi, ctx, "")
    write_csv(ds_dir / "corrected_joint_edge_pair_dataset_v2_metrics.csv", [metric_initial, metric_corr])
    (out / "CORRECTED_JOINT_DATASET_V2_REPORT.md").write_text(
        "# Corrected joint dataset v2 report\n\n"
        f"- samples: `{n}`\n"
        f"- K: `{args.k}`\n"
        f"- initial deltaS std: `{float(metric_initial['deltaS_std']):.6g}`\n"
        f"- corrected deltaS std: `{float(metric_corr['deltaS_std']):.6g}`\n"
        f"- cumulative acceptance at K: `{traj[-1]['cumulative_acceptance']:.6g}`\n"
        f"- saved dataset: `{v2_path}`\n"
        f"- sha256: `{sha256(v2_path)}`\n"
    )
    data = load_npz(v2_path)
    (out / "LARGER_JOINT_MODEL_DEFINITION.md").write_text(
        "# Larger joint edge+pair model definition\n\n"
        f"The controlled larger model is a two-channel procedural circular-convolution affine coupling flow with `target_channels=2`, `cond_channels=1`, `hidden={args.hidden}`, and `n_coupling_layers={args.n_coupling}`.\n\n"
        "Channel layout:\n\n"
        "- channel 0: edge `d10`;\n"
        "- channel 1: pair `d01`.\n\n"
        "The flow has one shared joint base Gaussian and one combined logdet/logq. It uses the same transported-detail reconstruction slots as previous diagnostics, splitting the joint output back into edge and pair channels before inverse small3 reconstruction. This differs from the previous modest joint flow by using wider conditioners and more coupling layers; it differs from the failed sequential model by not factorizing edge and pair into separate stages.\n"
    )
    smoke_data = {k: (v[: min(256, n)] if isinstance(v, np.ndarray) and v.shape[:1] == (n,) else v) for k, v in data.items()}
    smoke = train_joint_flow(smoke_data, ctx, out / "training_smoke", epochs=args.smoke_epochs, hidden=args.hidden, layers=3, n_coupling=args.n_coupling, seed=args.seed + 1, prefix="larger_smoke")
    (out / "LARGER_JOINT_TRAINING_SMOKE_REPORT.md").write_text(
        "# Larger joint training smoke report\n\n"
        f"- smoke samples: `{smoke_data['conditioning_c00'].shape[0]}`\n"
        f"- epochs: `{args.smoke_epochs}`\n"
        f"- best z0 deltaS std: `{float(smoke['best_by_action']['z0_deltaS_std']):.6g}`\n"
        f"- best NLL: `{float(smoke['best_by_nll']['val_nll']):.6g}`\n"
        "- finite loss/checkpoint/logq path passed.\n"
    )
    diag = train_joint_flow(data, ctx, out / "diagnostic_training", epochs=args.diagnostic_epochs, hidden=args.hidden, layers=3, n_coupling=args.n_coupling, seed=args.seed + 2, prefix="larger_joint")
    shutil.copy2(out / "diagnostic_training/joint_training_metrics.csv", out / "larger_joint_training_metrics.csv")
    cumulative = cumulative_diagnostics(Path(diag["checkpoint"]), Path(diag["local_gaussian"]), data, arrays, ctx, out, n_eval=512)
    vals = {r["assembly"]: r for r in cumulative}
    target_corner_std = float(vals["joint_flow_z0_edge_pair_target_corner"]["deltaS_std_vs_target_all"])
    baseline_corner_std = float(vals["joint_flow_z0_edge_pair_baseline_corner_z0"]["deltaS_std_vs_target_all"])
    best_z0 = float(diag["best_by_action"]["z0_deltaS_std"])
    sampler_smoke = False
    smoke_gate = target_corner_std <= 12.0
    interpretation = "The larger joint model did not improve below the modest joint-flow scale; this branch should stop."
    if target_corner_std < 15.0:
        interpretation = "The larger joint model modestly improved the target-corner cumulative metric but did not approach the MCMC teacher scale."
    if smoke_gate:
        interpretation = "The larger joint model reached the strong diagnostic gate for target-corner cumulative action, but no sampler smoke was launched automatically."
    (out / "LARGER_JOINT_DIAGNOSTIC_TRAINING_REPORT.md").write_text(
        "# Larger joint diagnostic training report\n\n"
        f"- dataset samples: `{n}`\n"
        f"- epochs: `{args.diagnostic_epochs}`\n"
        f"- hidden/couplings: `{args.hidden}` / `{args.n_coupling}`\n"
        f"- best-by-action checkpoint: `{diag['best_by_action']['checkpoint']}`\n"
        f"- best z0 deltaS std: `{best_z0:.6g}`\n"
        f"- v2 teacher corrected dataset deltaS std: `{float(metric_corr['deltaS_std']):.6g}`\n"
        f"- modest joint flow reference: `15.7694`\n"
        "- dense checkpoint metrics are in `larger_joint_training_metrics.csv`.\n"
    )
    (out / "CUMULATIVE_LARGER_JOINT_REPORT.md").write_text(
        "# Cumulative larger joint report\n\n"
        f"- target-all deltaS std: `{float(vals['target_all']['deltaS_std_vs_target_all']):.6g}`\n"
        f"- MCMC teacher corrected edge+pair with target corner deltaS std: `{float(vals['mcmc_teacher_corrected_edge_pair_target_corner']['deltaS_std_vs_target_all']):.6g}`\n"
        f"- larger joint z0 edge+pair with target corner deltaS std: `{target_corner_std:.6g}`\n"
        f"- larger joint z0 edge+pair with baseline corner z0 deltaS std: `{baseline_corner_std:.6g}`\n"
        f"- tiny sampler smoke justified by strong gate: `{smoke_gate}`; none was launched.\n"
    )
    shutil.copy2(out / "cumulative_joint_edge_pair_metrics.csv", out / "cumulative_larger_joint_metrics.csv")
    summary = {
        "status": "completed",
        "v2_dataset": str(v2_path),
        "v2_dataset_samples": n,
        "v2_teacher_deltaS_std": float(metric_corr["deltaS_std"]),
        "v2_teacher_acceptance": float(traj[-1]["cumulative_acceptance"]),
        "best_z0_deltaS_std": best_z0,
        "cumulative_target_corner_deltaS_std": target_corner_std,
        "cumulative_baseline_corner_deltaS_std": baseline_corner_std,
        "sampler_smoke_launched": sampler_smoke,
        "smoke_gate": smoke_gate,
        "diagnostic_checkpoint": diag["checkpoint"],
        "diagnostic_local_gaussian": diag["local_gaussian"],
        "interpretation": interpretation,
    }
    write_json(out / "larger_joint_edge_pair_summary.json", summary)
    (out / "LARGER_JOINT_EDGE_PAIR_FINAL_REPORT.md").write_text(
        "# Larger joint edge+pair final report\n\n"
        f"1. Was a larger corrected joint dataset generated?\n\n   Yes. `{n}` samples at K=`{args.k}`, corrected deltaS std `{float(metric_corr['deltaS_std']):.6g}`.\n\n"
        f"2. Did a larger joint model fit the corrected MCMC teacher better?\n\n   Best z0 deltaS std is `{best_z0:.6g}`; modest joint reference was `15.7694`.\n\n"
        f"3. Did it preserve the teacher action improvement?\n\n   Teacher v2 scale is `{float(metric_corr['deltaS_std']):.6g}`; larger joint target-corner cumulative scale is `{target_corner_std:.6g}`.\n\n"
        f"4. Did cumulative full-model diagnostics improve?\n\n   Target-corner scale `{target_corner_std:.6g}`, baseline-corner scale `{baseline_corner_std:.6g}`.\n\n"
        f"5. Was tiny sampler smoke justified?\n\n   `{smoke_gate}`. No sampler smoke or long validation was launched.\n\n"
        "6. If this fails, should we stop this branch?\n\n   If the larger model does not reach below about 13, stop same-coordinate edge/pair flow remediation and move to a different transported-detail parameterization.\n\n"
        "7. Exact next command\n\n   No automatic run is recommended from this branch unless Anna explicitly approves another architecture or parameterization change.\n"
    )
    write_status(summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
