#!/usr/bin/env python3
"""Scan saved L16->L32 finite-footprint checkpoints."""

from __future__ import annotations

import csv
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCAN_ROOT = PROJECT_ROOT / "perfect_blocking_upsampling/outputs/lam0p022_L16to32_flow_footprint_scan"
sys.path.insert(0, str(PROJECT_ROOT / "perfect_blocking_upsampling/scripts"))

import train_l16to32_footprint_candidate as train  # noqa: E402

STAGES = ("edge_x", "edge_y", "body")
DEEP_CANDIDATES = ("fp_medium_1_deep", "fp_large_safe_deep")
PILOT_BASELINES = ("fp_medium_1", "fp_large_safe")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def load_cfg(candidate: str) -> train.CandidateConfig:
    summary = json.loads((SCAN_ROOT / candidate / "validation_summary.json").read_text())
    cfg = summary["config"]
    return train.CandidateConfig(**cfg)


def prepare_arrays(cfg: train.CandidateConfig) -> dict[str, Any]:
    fine, coarse, fine_meta, coarse_meta = train.load_data()
    kernel, kernel_json = train.load_kernel(train.KERNEL)
    psi = train.apply_kernel(fine, kernel)
    c_all, d_all = train.split_psi(psi)
    rng = np.random.default_rng(cfg.seed)
    idx = rng.permutation(len(fine))
    train_idx = idx[: cfg.max_train]
    val_idx = idx[cfg.max_train : cfg.max_train + cfg.max_val]
    c_val, d_val, phi_val = c_all[val_idx], d_all[val_idx], fine[val_idx]
    draw = rng.integers(0, len(coarse), size=cfg.n_proposals)
    c_prop = coarse[draw]
    direct_obs = train.local_observables(phi_val)
    direct_obs["xi_over_L"] = train.xi_over_l(phi_val)
    return {
        "kernel": kernel,
        "kernel_json": kernel_json,
        "c_val": c_val,
        "d_val": d_val,
        "phi_val": phi_val,
        "c_prop": c_prop,
        "direct_obs": direct_obs,
        "fine_meta": fine_meta,
        "coarse_meta": coarse_meta,
    }


def model_for_stage(stage: str, cfg: train.CandidateConfig, state: dict[str, Any], c_val: np.ndarray, d_val: np.ndarray) -> train.PatchAffineNF:
    if stage == "edge_x":
        cond = train.condition_grid(c_val[:1], None, "edge_x")
    elif stage == "edge_y":
        cond = train.condition_grid(c_val[:1], None, "edge_y")
    else:
        cond = train.condition_grid(c_val[:1], d_val[:1, 0:2], "body")
    x = train.gather_features(cond, stage, cfg.footprint)
    model = train.PatchAffineNF(x.shape[1], cfg.hidden_channels, cfg.conditioner_layers)
    model.load_state_dict(state["model_state"])
    model.eval()
    return model


def checkpoint_paths(candidate: str, label: str) -> dict[str, Path]:
    cdir = SCAN_ROOT / candidate / "checkpoints"
    if label == "best_combo":
        return {stage: cdir / stage / "checkpoint_best.pt" for stage in STAGES}
    return {stage: cdir / stage / f"epoch{int(label):04d}.pt" for stage in STAGES}


def evaluate_checkpoint(candidate: str, label: str, cfg: train.CandidateConfig, arrays: dict[str, Any]) -> dict[str, Any]:
    import torch

    paths = checkpoint_paths(candidate, label)
    states = {stage: torch.load(path, map_location="cpu") for stage, path in paths.items()}
    models = {
        stage: model_for_stage(stage, cfg, states[stage], arrays["c_val"], arrays["d_val"])
        for stage in STAGES
    }
    c_val = arrays["c_val"]
    d_val = arrays["d_val"]
    c_prop = arrays["c_prop"]
    kernel = arrays["kernel"]
    direct = arrays["direct_obs"]

    val_logq = train.logq_all(models, c_val, d_val, cfg)
    d_gen, logq_gen = train.sample_all(models, c_val, cfg, cfg.seed + 100)
    logq_regen = train.logq_all(models, c_val, d_gen, cfg)
    phi_gen, _ = train.inverse_kernel(train.reconstruct(c_val, d_gen), kernel)
    c_reblocked, _ = train.split_psi(train.apply_kernel(phi_gen, kernel))
    d_prop, logq_prop = train.sample_all(models, c_prop, cfg, cfg.seed + 200)
    phi_prop, _ = train.inverse_kernel(train.reconstruct(c_prop, d_prop), kernel)
    c_reblocked_prop, _ = train.split_psi(train.apply_kernel(phi_prop, kernel))
    sf = train.action_total(phi_prop, train.ActionSpec("phi4_nn", train.LAMBDA, train.KAPPA))
    sc = train.action_total(c_prop, train.ActionSpec("phi4_nn", train.LAMBDA, train.KAPPA))
    delta_s = sf - sc
    logw = -delta_s - logq_prop
    proposal = train.local_observables(phi_prop)
    nan_inf_clean = bool(
        np.all(np.isfinite(phi_gen))
        and np.all(np.isfinite(phi_prop))
        and np.all(np.isfinite(logq_gen))
        and np.all(np.isfinite(logw))
    )
    roundtrip = float(np.max(np.abs(train.reconstruct(c_val, d_val) - train.apply_kernel(arrays["phi_val"], kernel))))
    reblock = float(np.max(np.abs(c_reblocked_prop - c_prop)))
    epoch = states["body"].get("epoch") if label == "best_combo" else int(label)
    return {
        "candidate": candidate,
        "checkpoint": str(label),
        "epoch": epoch,
        "edge_x_checkpoint": str(paths["edge_x"]),
        "edge_y_checkpoint": str(paths["edge_y"]),
        "body_checkpoint": str(paths["body"]),
        "footprint": cfg.footprint,
        "radius": train.max_radius(cfg.footprint),
        "deltaS_mean": float(np.mean(delta_s)),
        "deltaS_std": float(np.std(delta_s, ddof=1)),
        "logw_mean": float(np.mean(logw)),
        "logw_std": float(np.std(logw, ddof=1)),
        "ess_over_n": float(train.stable_ess(logw) / len(logw)),
        "action_density_error": float(proposal["action_density"] - direct["action_density"]),
        "phi2_error": float(proposal["phi2"] - direct["phi2"]),
        "phi4_error": float(proposal["phi4"] - direct["phi4"]),
        "NN_error": float(proposal["NN"] - direct["NN"]),
        "2nn_error": float(proposal["second_neighbor"] - direct["second_neighbor"]),
        "diag_error": float(proposal["diag"] - direct["diag"]),
        "roundtrip_max_error": roundtrip,
        "reblocking_max_error": reblock,
        "logq_rescore_max_error": float(np.max(np.abs(logq_gen - logq_regen))),
        "nan_inf_status": "clean" if nan_inf_clean else "failed",
        "val_true_logq_mean": float(np.mean(val_logq)),
        "val_true_logq_std": float(np.std(val_logq, ddof=1)),
    }


def baseline_row(candidate: str) -> dict[str, Any]:
    summary = json.loads((SCAN_ROOT / candidate / "validation_summary.json").read_text())
    validation = {row["case"]: row for row in summary["validation"]}
    direct = validation["direct_L32_validation"]
    proposal = validation["native_L16_prior_proposal_L32"]
    logw = {row["quantity"]: row for row in summary["logweight"]}["proposal_logw"]
    cfg = summary["config"]
    return {
        "candidate": candidate,
        "checkpoint": "pilot_summary_best_combo",
        "epoch": int(cfg["epochs"]),
        "edge_x_checkpoint": str(SCAN_ROOT / candidate / "checkpoints/edge_x/checkpoint_best.pt"),
        "edge_y_checkpoint": str(SCAN_ROOT / candidate / "checkpoints/edge_y/checkpoint_best.pt"),
        "body_checkpoint": str(SCAN_ROOT / candidate / "checkpoints/body/checkpoint_best.pt"),
        "footprint": summary["footprint"]["footprint_size"],
        "radius": summary["footprint"]["max_radius_fine_lattice_sites"],
        "deltaS_mean": summary["validation_deltaS"]["mean"],
        "deltaS_std": summary["validation_deltaS"]["std"],
        "logw_mean": logw["mean"],
        "logw_std": logw["std"],
        "ess_over_n": logw["ess_over_n"],
        "action_density_error": proposal["action_density"] - direct["action_density"],
        "phi2_error": proposal["phi2"] - direct["phi2"],
        "phi4_error": proposal["phi4"] - direct["phi4"],
        "NN_error": proposal["NN"] - direct["NN"],
        "2nn_error": proposal["second_neighbor"] - direct["second_neighbor"],
        "diag_error": proposal["diag"] - direct["diag"],
        "roundtrip_max_error": summary["roundtrip_max_error"],
        "reblocking_max_error": summary["reblocking_max_error_prior_proposal"],
        "logq_rescore_max_error": summary["logq_transformed_density_consistency"]["generated_sample_vs_rescore_max_abs"],
        "nan_inf_status": "clean" if all(summary["nan_inf_check"].values()) else "failed",
        "val_true_logq_mean": "not rescanned",
        "val_true_logq_std": "not rescanned",
    }


def score_local(row: dict[str, Any]) -> float:
    keys = ("action_density_error", "phi2_error", "phi4_error", "NN_error", "diag_error")
    return float(sum(float(row[k]) ** 2 for k in keys) ** 0.5)


def eligible(row: dict[str, Any]) -> bool:
    return (
        row["nan_inf_status"] == "clean"
        and float(row["roundtrip_max_error"]) <= 1.0e-10
        and float(row["reblocking_max_error"]) <= 1.0e-5
    )


def make_plots(rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    deep_rows = [r for r in rows if r["candidate"] in DEEP_CANDIDATES and r["checkpoint"] != "best_combo"]
    fig, axes = plt.subplots(3, 2, figsize=(12, 12), constrained_layout=True)
    axes = axes.ravel()
    metrics = [
        ("deltaS_std", "DeltaS std"),
        ("logw_std", "logw std"),
        ("ess_over_n", "ESS/N"),
        ("action_density_error", "action-density error"),
        ("phi2_error", "phi2 error"),
        ("phi4_error", "phi4 error"),
    ]
    for ax, (key, title) in zip(axes, metrics):
        for cand in DEEP_CANDIDATES:
            vals = [r for r in deep_rows if r["candidate"] == cand]
            ax.plot([int(r["epoch"]) for r in vals], [float(r[key]) for r in vals], marker="o", markersize=2, label=cand)
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.savefig(SCAN_ROOT / "checkpoint_scan_plots.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for cand in DEEP_CANDIDATES:
        vals = [r for r in deep_rows if r["candidate"] == cand]
        x = [int(r["epoch"]) for r in vals]
        axes[0].plot(x, [float(r["NN_error"]) for r in vals], marker="o", markersize=2, label=cand)
        axes[1].plot(x, [float(r["diag_error"]) for r in vals], marker="o", markersize=2, label=cand)
        axes[2].plot(x, [score_local(r) for r in vals], marker="o", markersize=2, label=cand)
    axes[0].set_title("NN error")
    axes[1].set_title("diag error")
    axes[2].set_title("local-observable L2 score")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.savefig(SCAN_ROOT / "checkpoint_scan_local_observables.pdf")
    plt.close(fig)


def main() -> int:
    rows: list[dict[str, Any]] = []
    for candidate in PILOT_BASELINES:
        rows.append(baseline_row(candidate))
    for candidate in DEEP_CANDIDATES:
        cfg = load_cfg(candidate)
        arrays = prepare_arrays(cfg)
        epochs = sorted(
            int(path.stem.replace("epoch", ""))
            for path in (SCAN_ROOT / candidate / "checkpoints/body").glob("epoch*.pt")
        )
        for epoch in epochs:
            rows.append(evaluate_checkpoint(candidate, str(epoch), cfg, arrays))
        rows.append(evaluate_checkpoint(candidate, "best_combo", cfg, arrays))

    for row in rows:
        row["local_observable_l2_score"] = score_local(row)
    write_csv(SCAN_ROOT / "checkpoint_scan_results.csv", rows)

    eligible_rows = [r for r in rows if eligible(r)]
    best_stability = min(
        eligible_rows,
        key=lambda r: (float(r["deltaS_std"]), float(r["logw_std"]), -float(r["ess_over_n"])),
    )
    best_local = min(eligible_rows, key=score_local)
    medium1_baseline = next(r for r in rows if r["candidate"] == "fp_medium_1")
    beats_medium1 = [
        r for r in eligible_rows
        if r["candidate"] in DEEP_CANDIDATES
        and (float(r["deltaS_std"]), float(r["logw_std"])) < (float(medium1_baseline["deltaS_std"]), float(medium1_baseline["logw_std"]))
    ]
    payload = {
        "best_stability_checkpoint": best_stability,
        "best_local_observable_checkpoint": best_local,
        "same_checkpoint": best_stability["candidate"] == best_local["candidate"] and best_stability["checkpoint"] == best_local["checkpoint"],
        "intermediate_beats_original_fp_medium_1_on_deltaS_logw": bool(beats_medium1),
        "rows": rows,
    }
    (SCAN_ROOT / "checkpoint_scan_results.json").write_text(json.dumps(payload, indent=2) + "\n")
    make_plots(rows)

    def fmt(x: Any) -> str:
        if isinstance(x, str):
            return x
        return f"{float(x):.6g}"

    lines = [
        "# Checkpoint Scan Summary",
        "",
        "Scanned synchronized epoch checkpoints for `fp_medium_1_deep` and `fp_large_safe_deep`, plus each run's stage-wise `checkpoint_best.pt` combination. The 10-epoch `fp_medium_1` and `fp_large_safe` summary checkpoints are included as baselines.",
        "",
        "## Best Selections",
        "",
        f"- best stability checkpoint: `{best_stability['candidate']}` / `{best_stability['checkpoint']}` with DeltaS std `{fmt(best_stability['deltaS_std'])}`, logw std `{fmt(best_stability['logw_std'])}`, ESS/N `{fmt(best_stability['ess_over_n'])}`",
        f"- best local-observable checkpoint: `{best_local['candidate']}` / `{best_local['checkpoint']}` with local L2 score `{fmt(best_local['local_observable_l2_score'])}`",
        f"- same checkpoint: `{payload['same_checkpoint']}`",
        f"- any deep intermediate beats original `fp_medium_1` on DeltaS/logw: `{payload['intermediate_beats_original_fp_medium_1_on_deltaS_logw']}`",
        "",
        "## Compact Table",
        "",
        "| candidate | checkpoint | epoch | DeltaS std | logw std | ESS/N | action err | phi2 err | phi4 err | NN err | 2nn err | diag err | local L2 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    key_rows = [r for r in rows if r["checkpoint"] in {"pilot_summary_best_combo", "best_combo"}]
    key_rows += [best_stability, best_local]
    seen: set[tuple[str, str]] = set()
    for row in key_rows:
        ident = (row["candidate"], row["checkpoint"])
        if ident in seen:
            continue
        seen.add(ident)
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["candidate"]),
                    str(row["checkpoint"]),
                    str(row["epoch"]),
                    fmt(row["deltaS_std"]),
                    fmt(row["logw_std"]),
                    fmt(row["ess_over_n"]),
                    fmt(row["action_density_error"]),
                    fmt(row["phi2_error"]),
                    fmt(row["phi4_error"]),
                    fmt(row["NN_error"]),
                    fmt(row["2nn_error"]),
                    fmt(row["diag_error"]),
                    fmt(row["local_observable_l2_score"]),
                ]
            )
            + " |"
        )
    lines += [
        "",
        "Full per-epoch rows are in `checkpoint_scan_results.csv` and `checkpoint_scan_results.json`.",
        "Plots are in `checkpoint_scan_plots.pdf` and `checkpoint_scan_local_observables.pdf`.",
    ]
    (SCAN_ROOT / "checkpoint_scan_summary.md").write_text("\n".join(lines) + "\n")

    summary_path = SCAN_ROOT / "FOOTPRINT_SCAN_SUMMARY.md"
    existing = summary_path.read_text() if summary_path.exists() else ""
    marker = "\n## Checkpoint History Scan\n"
    addition = (
        marker
        + "\n"
        + f"- best stability checkpoint: `{best_stability['candidate']}` / `{best_stability['checkpoint']}`\n"
        + f"- best local-observable checkpoint: `{best_local['candidate']}` / `{best_local['checkpoint']}`\n"
        + f"- same checkpoint: `{payload['same_checkpoint']}`\n"
        + f"- any deep intermediate beats original `fp_medium_1` on DeltaS/logw: `{payload['intermediate_beats_original_fp_medium_1_on_deltaS_logw']}`\n"
        + "- see `checkpoint_scan_summary.md` for details\n"
    )
    if marker in existing:
        existing = existing.split(marker)[0].rstrip() + "\n" + addition
    else:
        existing = existing.rstrip() + "\n" + addition
    summary_path.write_text(existing)
    print(json.dumps({
        "best_stability": {"candidate": best_stability["candidate"], "checkpoint": best_stability["checkpoint"], "deltaS_std": best_stability["deltaS_std"], "logw_std": best_stability["logw_std"]},
        "best_local": {"candidate": best_local["candidate"], "checkpoint": best_local["checkpoint"], "local_score": best_local["local_observable_l2_score"]},
        "same": payload["same_checkpoint"],
        "beats_fp_medium_1": payload["intermediate_beats_original_fp_medium_1_on_deltaS_logw"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
