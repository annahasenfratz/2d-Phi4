#!/usr/bin/env python3
"""Controlled L16->L32 patch-chain validation for finite-footprint checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCAN_ROOT = PROJECT_ROOT / "perfect_blocking_upsampling/outputs/lam0p022_L16to32_flow_footprint_scan"
sys.path.insert(0, str(PROJECT_ROOT / "perfect_blocking_upsampling/scripts"))

import train_l16to32_footprint_candidate as flowtrain  # noqa: E402

NATIVE_REF = PROJECT_ROOT / "phi4_phase-diagram/ensembles/lam0p022_kappa0p2705_L32_embedded_wolff_sign_cluster_plus_radial_heatbath/configs.npz"
NATIVE_OBS = {
    "action_density": {"mean": 0.390910727557, "stderr": 0.0008353},
    "phi2": {"mean": 1.48647333021, "stderr": 0.005779},
    "phi4": {"mean": 4.93200757927, "stderr": 0.02955},
    "NN": {"mean": 1.05236778454, "stderr": 0.006092},
    "2nn": {"mean": 0.874513715773, "stderr": 0.006551},
    "diag": {"mean": 0.940780431018, "stderr": 0.006298},
    "m": {"mean": 0.00690037175982, "stderr": 0.02446},
    "abs_m": {"mean": 0.733854343535, "stderr": 0.007682},
    "chi": {"mean": 611.839009921, "stderr": 10.04},
    "Binder_U4": {"mean": 0.57702067183, "stderr": None},
    "xi_over_L": {"mean": 0.766173127923, "stderr": None},
}
LOCAL_KEYS = ["action_density", "phi2", "phi4", "NN", "2nn", "diag"]
LONG_KEYS = ["abs_m", "chi", "Binder_U4", "xi_over_L"]
ALL_KEYS = LOCAL_KEYS + LONG_KEYS


@dataclass
class ChainConfig:
    candidate: str
    output_dir: str
    footprint: int
    edge_x_checkpoint: str
    edge_y_checkpoint: str
    body_checkpoint: str
    seed: int = 2026070411
    n_chains: int = 8
    n_sweeps: int = 2000
    record_every: int = 20
    p_coarse: int = 12
    coarse_patches_per_sweep: int = 4
    coarse_passes: int = 5
    epsilon_c: float = 0.6
    p_detail: int = 12
    beta_z: float = 0.4
    n_detail_updates_per_sweep: int = 2


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class StreamingCsvWriter:
    def __init__(self, path: Path, fieldnames: list[str]):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = self.path.open("w", newline="", buffering=1)
        self.writer = csv.DictWriter(self.fh, fieldnames=fieldnames, extrasaction="ignore")
        self.writer.writeheader()
        self.fh.flush()

    def write(self, row: dict[str, Any]) -> None:
        self.writer.writerow(row)
        self.fh.flush()

    def close(self) -> None:
        self.fh.flush()
        self.fh.close()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=json_default) + "\n")


def raw_data_filename(cfg: ChainConfig) -> str:
    return f"data_16to32_P{cfg.p_coarse}_pass{cfg.coarse_passes}_detail{cfg.n_detail_updates_per_sweep}.csv"


def patches_per_sweep(lc: int, patch_size: int) -> int:
    return int(math.ceil(2.0 * lc * lc / float(patch_size * patch_size)))


def json_default(x: Any) -> Any:
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(type(x).__name__)


def patch_sites(l: int, x0: int, y0: int, p: int) -> list[tuple[int, int]]:
    return [((x0 + dx) % l, (y0 + dy) % l) for dx in range(p) for dy in range(p)]


def local_delta_action_site(field: np.ndarray, i: int, j: int, new_x: float, action: flowtrain.ActionSpec) -> float:
    old_x = float(field[i, j])
    lam = float(action.lambda_)
    kap = float(action.kappa)
    onsite_old = (1.0 - 2.0 * lam) * old_x * old_x + lam * old_x**4
    onsite_new = (1.0 - 2.0 * lam) * new_x * new_x + lam * new_x**4
    nn = float(field[(i + 1) % field.shape[0], j] + field[(i - 1) % field.shape[0], j] + field[i, (j + 1) % field.shape[1]] + field[i, (j - 1) % field.shape[1]])
    return (onsite_new - onsite_old) - 2.0 * kap * (new_x - old_x) * nn


def controlled_patch_metropolis(u: np.ndarray, sites: list[tuple[int, int]], rng: np.random.Generator, action: flowtrain.ActionSpec, step_size: float, passes: int) -> tuple[np.ndarray, dict[str, Any]]:
    current = u.copy()
    before = current.copy()
    sc_before = float(flowtrain.action_total(current[None], action)[0])
    attempted = 0
    accepted = 0
    accepted_dsc_sum = 0.0
    for _ in range(passes):
        order = list(sites)
        rng.shuffle(order)
        for i, j in order:
            old = float(current[i, j])
            new = old + float(step_size * rng.standard_normal())
            dsc = local_delta_action_site(current, i, j, new, action)
            attempted += 1
            if math.log(max(float(rng.random()), 1.0e-300)) < min(0.0, -dsc):
                current[i, j] = new
                accepted += 1
                accepted_dsc_sum += float(dsc)
    sc_after = float(flowtrain.action_total(current[None], action)[0])
    delta = current - before
    return current.astype(np.float32), {
        "coarse_site_attempts": attempted,
        "coarse_site_accepts": accepted,
        "coarse_site_acceptance": accepted / max(attempted, 1),
        "delta_Sc_patch_exact": sc_after - sc_before,
        "delta_Sc_patch_local_sum": accepted_dsc_sum,
        "patch_l2_change": float(np.sqrt(np.sum(delta * delta))),
        "patch_linf_change": float(np.max(np.abs(delta))) if delta.size else 0.0,
    }


def load_stage_model(stage: str, path: Path, cfg: flowtrain.CandidateConfig) -> flowtrain.PatchAffineNF:
    import torch

    ckpt = torch.load(path, map_location="cpu")
    c_dummy = np.zeros((1, 16, 16), dtype=np.float32)
    d_dummy = np.zeros((1, 3, 16, 16), dtype=np.float32)
    cond = flowtrain.condition_grid(c_dummy, d_dummy[:, 0:2] if stage == "body" else None, stage)
    x = flowtrain.gather_features(cond, stage, cfg.footprint)
    model = flowtrain.PatchAffineNF(x.shape[1], cfg.hidden_channels, cfg.conditioner_layers)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def model_forward_from_z(model: flowtrain.PatchAffineNF, cond: np.ndarray, z: np.ndarray, stage: str, footprint: int) -> tuple[np.ndarray, np.ndarray]:
    import torch

    n = cond.shape[0]
    lc = cond.shape[-1] // 2
    x = torch.tensor(flowtrain.gather_features(cond, stage, footprint), dtype=torch.float32)
    z_t = torch.tensor(z.reshape(-1, 1), dtype=torch.float32)
    with torch.no_grad():
        shift, logscale = model.shift_logscale(x)
        y = shift + torch.exp(logscale) * z_t
        logq_site = -0.5 * (z_t * z_t + flowtrain.LOG2PI).sum(dim=1) - logscale.sum(dim=1)
    return y.cpu().numpy().reshape(n, lc, lc).astype(np.float32), logq_site.cpu().numpy().reshape(n, lc, lc).sum(axis=(1, 2)).astype(np.float64)


def compute_state(u: np.ndarray, z_edge: np.ndarray, z_pair: np.ndarray, z_body: np.ndarray, models: dict[str, flowtrain.PatchAffineNF], kernel, cfg: flowtrain.CandidateConfig) -> dict[str, Any]:
    d0, l0 = model_forward_from_z(models["edge_x"], flowtrain.condition_grid(u, None, "edge_x"), z_edge, "edge_x", cfg.footprint)
    d1, l1 = model_forward_from_z(models["edge_y"], flowtrain.condition_grid(u, None, "edge_y"), z_pair, "edge_y", cfg.footprint)
    dprev = np.stack([d0, d1], axis=1)
    d2, l2 = model_forward_from_z(models["body"], flowtrain.condition_grid(u, dprev, "body"), z_body, "body", cfg.footprint)
    d = np.stack([d0, d1, d2], axis=1)
    phi, inv = flowtrain.inverse_kernel(flowtrain.reconstruct(u, d), kernel)
    sf = flowtrain.action_total(phi, flowtrain.ActionSpec("phi4_nn", flowtrain.LAMBDA, flowtrain.KAPPA))
    sc = flowtrain.action_total(u, flowtrain.ActionSpec("phi4_nn", flowtrain.LAMBDA, flowtrain.KAPPA))
    logq = l0 + l1 + l2
    logw = -sf + sc - logq
    return {"u": u.astype(np.float32), "z_edge": z_edge.astype(np.float32), "z_pair": z_pair.astype(np.float32), "z_body": z_body.astype(np.float32), "phi": phi.astype(np.float32), "sf": sf.astype(np.float64), "sc": sc.astype(np.float64), "logq": logq.astype(np.float64), "logw": logw.astype(np.float64), "inv": inv}


def observables(phi: np.ndarray) -> dict[str, float]:
    arr = np.asarray(phi, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr[None]
    m = arr.mean(axis=(1, 2))
    m2 = m * m
    m4 = m2 * m2
    nn = 0.5 * (np.mean(arr * np.roll(arr, -1, axis=1), axis=(1, 2)) + np.mean(arr * np.roll(arr, -1, axis=2), axis=(1, 2)))
    twonn = 0.5 * (np.mean(arr * np.roll(arr, -2, axis=1), axis=(1, 2)) + np.mean(arr * np.roll(arr, -2, axis=2), axis=(1, 2)))
    diag = 0.5 * (
        np.mean(arr * np.roll(np.roll(arr, -1, axis=1), -1, axis=2), axis=(1, 2))
        + np.mean(arr * np.roll(np.roll(arr, -1, axis=1), 1, axis=2), axis=(1, 2))
    )
    l = arr.shape[1]
    return {
        "action_density": float(np.mean(flowtrain.action_total(arr, flowtrain.ActionSpec("phi4_nn", flowtrain.LAMBDA, flowtrain.KAPPA)) / (l * l))),
        "phi2": float(np.mean(arr * arr)),
        "phi4": float(np.mean(arr**4)),
        "NN": float(np.mean(nn)),
        "2nn": float(np.mean(twonn)),
        "diag": float(np.mean(diag)),
        "m": float(np.mean(m)),
        "abs_m": float(np.mean(np.abs(m))),
        "chi": float(l * l * np.mean(m2)),
        "Binder_U4": float(1.0 - np.mean(m4) / (3.0 * max(np.mean(m2) ** 2, 1.0e-300))),
        "xi_over_L": flowtrain.xi_over_l(arr.astype(np.float32)),
    }


def row_with_errors(base: dict[str, Any], obs: dict[str, float]) -> dict[str, Any]:
    row = dict(base)
    local_sq = []
    long_sq = []
    for key in ALL_KEYS:
        val = float(obs[key])
        ref = float(NATIVE_OBS[key]["mean"])
        err = val - ref
        row[key] = val
        row[f"{key}_error"] = err
        stderr = NATIVE_OBS[key]["stderr"]
        row[f"{key}_z"] = err / stderr if stderr else ""
        if key in LOCAL_KEYS:
            local_sq.append(err * err)
        else:
            long_sq.append(err * err)
    row["local_rms_error"] = float(math.sqrt(sum(local_sq) / len(local_sq)))
    row["long_rms_error"] = float(math.sqrt(sum(long_sq) / len(long_sq)))
    return row


def tau_int_1d(x: np.ndarray, max_lag: int | None = None) -> float:
    y = np.asarray(x, dtype=np.float64)
    y = y[np.isfinite(y)]
    n = len(y)
    if n < 4:
        return float("nan")
    y = y - np.mean(y)
    var = float(np.dot(y, y) / n)
    if var <= 0:
        return 0.5
    max_lag = min(max_lag or n // 2, n - 1)
    tau = 0.5
    for lag in range(1, max_lag + 1):
        ac = float(np.dot(y[:-lag], y[lag:]) / (n - lag) / var)
        if ac <= 0:
            break
        tau += ac
    return float(tau)


def make_outputs(out: Path, cfg: ChainConfig, obs_rows: list[dict[str, Any]], acc_rows: list[dict[str, Any]], move_rows: list[dict[str, Any]], runtime: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    write_csv(out / "observable_history_obs20.csv", obs_rows)
    write_csv(out / "acceptance_history.csv", acc_rows)
    write_csv(out / "move_type_summary.csv", move_rows)

    # Acceptance windows.
    win_rows = []
    for start in range(1, cfg.n_sweeps + 1, 20):
        stop = min(start + 19, cfg.n_sweeps)
        sub = [r for r in acc_rows if start <= int(r["sweep"]) <= stop]
        if not sub:
            continue
        win_rows.append({
            "sweep_start": start,
            "sweep_stop": stop,
            "coarse_fine_AR_acceptance": float(np.mean([float(r["coarse_fine_AR_acceptance"]) for r in sub])),
            "latent_acceptance": float(np.mean([float(r["latent_acceptance"]) for r in sub])),
            "coarse_site_acceptance": float(np.mean([float(r["coarse_site_acceptance"]) for r in sub])),
        })
    write_csv(out / "acceptance_windows_20sweep.csv", win_rows)

    # Window observable summary.
    window_defs = [(250, 1000), (1000, 2000), (250, 2000), (500, 2000)]
    window_rows = []
    for lo, hi in window_defs:
        sub = [r for r in obs_rows if lo <= int(r["sweep"]) <= hi]
        if not sub:
            continue
        row = {"window": f"{lo}-{hi}", "sweep_start": lo, "sweep_stop": hi, "n_records": len(sub)}
        for key in ALL_KEYS + ["local_rms_error", "long_rms_error"]:
            vals = np.asarray([float(r[key]) for r in sub], dtype=np.float64)
            row[f"{key}_mean"] = float(np.mean(vals))
            row[f"{key}_stderr"] = float(np.std(vals, ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else 0.0
        window_rows.append(row)
    write_csv(out / "window_observable_summary.csv", window_rows)

    rms_rows = [{"sweep": r["sweep"], "local_rms_error": r["local_rms_error"], "long_rms_error": r["long_rms_error"]} for r in obs_rows]
    write_csv(out / "local_rms_error_by_sweep_obs20.csv", rms_rows)

    # Autocorrelation by chain, averaged.
    ac_rows = []
    for cut in [250, 500]:
        for key in ALL_KEYS:
            taus = []
            for chain in sorted({int(r["chain"]) for r in obs_rows}):
                vals = [float(r[key]) for r in obs_rows if int(r["chain"]) == chain and int(r["sweep"]) >= cut]
                if len(vals) >= 4:
                    taus.append(tau_int_1d(np.asarray(vals), max_lag=min(50, len(vals)//2)) * cfg.record_every)
            ac_rows.append({"cut_sweeps": cut, "observable": key, "tau_int_sweeps_mean": float(np.mean(taus)) if taus else float("nan"), "tau_int_sweeps_std": float(np.std(taus, ddof=1)) if len(taus) > 1 else 0.0, "n_chains": len(taus), "note": "rough estimate; record spacing and finite chain length limit accuracy"})
    write_csv(out / "autocorrelation_summary.csv", ac_rows)

    # Plots.
    def mean_by_sweep(key: str) -> tuple[list[int], list[float]]:
        sweeps = sorted({int(r["sweep"]) for r in obs_rows})
        return sweeps, [float(np.mean([float(r[key]) for r in obs_rows if int(r["sweep"]) == s])) for s in sweeps]

    for filename, keys in [
        ("phi2_phi4_nn_2nn_time_history.pdf", ["phi2", "phi4", "NN", "2nn"]),
        ("xi_chi_binder_time_history.pdf", ["xi_over_L", "chi", "Binder_U4"]),
    ]:
        fig, axes = plt.subplots(len(keys), 1, figsize=(8, 2.4 * len(keys)), sharex=True, constrained_layout=True)
        if len(keys) == 1:
            axes = [axes]
        for ax, key in zip(axes, keys):
            x, y = mean_by_sweep(key)
            ax.plot(x, y)
            ax.axhline(NATIVE_OBS[key]["mean"], color="k", linestyle="--", linewidth=0.8)
            ax.set_ylabel(key)
            ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel("sweep")
        fig.savefig(out / filename)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    x, y = mean_by_sweep("local_rms_error")
    ax.plot(x, y, label="local RMS")
    x, y = mean_by_sweep("long_rms_error")
    ax.plot(x, y, label="long RMS")
    ax.set_xlabel("sweep")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(out / "local_rms_time_history.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    sweeps = [int(r["sweep"]) for r in acc_rows]
    ax.plot(sweeps, [float(r["coarse_fine_AR_acceptance"]) for r in acc_rows], label="coarse fine A/R")
    ax.plot(sweeps, [float(r["latent_acceptance"]) for r in acc_rows], label="latent A/R")
    ax.plot(sweeps, [float(r["coarse_site_acceptance"]) for r in acc_rows], label="coarse site")
    ax.set_xlabel("sweep")
    ax.set_ylabel("per-sweep acceptance")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(out / "acceptance_history.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    for key in ["action_density", "phi2", "phi4", "NN", "diag"]:
        sub = [r for r in ac_rows if int(r["cut_sweeps"]) == 250 and r["observable"] == key]
        if sub:
            ax.bar(key, float(sub[0]["tau_int_sweeps_mean"]))
    ax.set_ylabel("tau_int sweeps, cut 250")
    ax.tick_params(axis="x", rotation=30)
    fig.savefig(out / "autocorrelation_plots.pdf")
    plt.close(fig)

    summary = {
        "candidate": cfg.candidate,
        "config": asdict(cfg),
        "native_reference": {"path": str(NATIVE_REF), "observables": NATIVE_OBS},
        "runtime_sec": runtime,
        "move_type_summary": move_rows,
        "window_observable_summary": window_rows,
        "autocorrelation_summary": ac_rows,
    }
    write_json(out / "controlled_chain_summary.json", summary)
    report = [
        f"# Controlled Chain Report: {cfg.candidate}",
        "",
        f"- runtime_sec: `{runtime:.6g}`",
        f"- n_chains: `{cfg.n_chains}`",
        f"- n_sweeps: `{cfg.n_sweeps}`",
        f"- record_every: `{cfg.record_every}`",
        f"- coarse proposal: P={cfg.p_coarse}, patches/sweep={cfg.coarse_patches_per_sweep}, passes={cfg.coarse_passes}, epsilon={cfg.epsilon_c}",
        f"- latent proposal: P={cfg.p_detail}, beta_z={cfg.beta_z}, updates/sweep={cfg.n_detail_updates_per_sweep}",
        "",
        "## Move Summary",
        "",
        "| move | attempts | acceptance | log_accept_mean | log_accept_std |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in move_rows:
        report.append(f"| {row['move_type']} | {row['attempts']} | {row['acceptance']:.6g} | {row.get('log_accept_mean',''):.6g} | {row.get('log_accept_std',''):.6g} |")
    report += ["", "## Windows", "", "`window_observable_summary.csv` contains post-cut windows 250-1000, 1000-2000, 250-2000, and 500-2000."]
    (out / "controlled_chain_report.md").write_text("\n".join(report) + "\n")


def preflight(cfg: ChainConfig) -> dict[str, Any]:
    c = flowtrain.CandidateConfig(candidate=cfg.candidate, footprint=cfg.footprint, max_train=800, max_val=200, output_dir=cfg.output_dir)
    models = {
        "edge_x": load_stage_model("edge_x", Path(cfg.edge_x_checkpoint), c),
        "edge_y": load_stage_model("edge_y", Path(cfg.edge_y_checkpoint), c),
        "body": load_stage_model("body", Path(cfg.body_checkpoint), c),
    }
    kernel, _ = flowtrain.load_kernel(flowtrain.KERNEL)
    u = np.zeros((1, 16, 16), dtype=np.float32)
    z = np.zeros((1, 16, 16), dtype=np.float32)
    state = compute_state(u, z, z, z, models, kernel, c)
    blocked, _ = flowtrain.split_psi(flowtrain.apply_kernel(state["phi"], kernel))
    return {
        "reference_path": str(NATIVE_REF),
        "reference_shape": list(np.load(NATIVE_REF)["phi"].shape),
        "native_observables": NATIVE_OBS,
        "footprint": cfg.footprint,
        "radius": flowtrain.max_radius(cfg.footprint),
        "non_wrapping_L32": flowtrain.max_radius(cfg.footprint) < 16,
        "checkpoints": {"edge_x": cfg.edge_x_checkpoint, "edge_y": cfg.edge_y_checkpoint, "body": cfg.body_checkpoint},
        "roundtrip_dry_finite": bool(np.all(np.isfinite(state["phi"])) and np.all(np.isfinite(state["logw"]))),
        "reblocking_dry_max_error": float(np.max(np.abs(blocked - u))),
        "liftability": flowtrain.validate_liftability(c),
        "chain_parameters": asdict(cfg),
    }


def run_chain(cfg: ChainConfig) -> None:
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pf = preflight(cfg)
    write_json(out / "preflight.json", pf)
    c = flowtrain.CandidateConfig(candidate=cfg.candidate, footprint=cfg.footprint, max_train=800, max_val=200, output_dir=cfg.output_dir)
    models = {
        "edge_x": load_stage_model("edge_x", Path(cfg.edge_x_checkpoint), c),
        "edge_y": load_stage_model("edge_y", Path(cfg.edge_y_checkpoint), c),
        "body": load_stage_model("body", Path(cfg.body_checkpoint), c),
    }
    kernel, _ = flowtrain.load_kernel(flowtrain.KERNEL)
    coarse, coarse_meta = flowtrain.load_wolff(flowtrain.PRIOR, 16, flowtrain.KAPPA)
    rng = np.random.default_rng(cfg.seed)
    obs_rows: list[dict[str, Any]] = []
    sweep_acc_rows: list[dict[str, Any]] = []
    coarse_move_logs: list[float] = []
    coarse_move_acc: list[int] = []
    coarse_site_accs: list[float] = []
    latent_move_logs: list[float] = []
    latent_move_acc: list[int] = []
    obs_fields = ["chain", "sweep", "source_idx"] + [key for key in ALL_KEYS] + [f"{key}_error" for key in ALL_KEYS] + [f"{key}_z" for key in ALL_KEYS] + ["local_rms_error", "long_rms_error"]
    data_fields = ["patch_size", "n_coarse_passes", "detail_patch_size", "detail_updates_per_sweep"] + obs_fields
    acc_fields = [
        "chain",
        "sweep",
        "patch_size",
        "detail_patch_size",
        "n_coarse_passes",
        "coarse_attempts",
        "coarse_fine_accepts",
        "coarse_fine_AR_acceptance",
        "coarse_site_attempts",
        "coarse_site_accepts",
        "coarse_site_acceptance",
        "coarse_log_accept_mean",
        "coarse_log_accept_std",
        "latent_attempts",
        "latent_accepts",
        "latent_acceptance",
        "latent_log_accept_mean",
        "latent_log_accept_std",
    ]
    rms_fields = ["sweep", "local_rms_error", "long_rms_error"]
    stream_writers = [
        StreamingCsvWriter(out / "observable_history_obs20.csv", obs_fields),
        StreamingCsvWriter(out / "controlled_patch_chain_observable_history.csv", obs_fields),
        StreamingCsvWriter(out / raw_data_filename(cfg), data_fields),
        StreamingCsvWriter(out / "acceptance_history.csv", acc_fields),
        StreamingCsvWriter(out / "controlled_patch_chain_acceptance_history.csv", acc_fields),
        StreamingCsvWriter(out / "local_rms_error_by_sweep_obs20.csv", rms_fields),
    ]
    obs_streams = stream_writers[0:2]
    data_stream = stream_writers[2]
    acc_streams = stream_writers[3:5]
    rms_stream = stream_writers[5]
    t0 = time.perf_counter()
    try:
        print(json.dumps({"event": "start", "candidate": cfg.candidate, "n_chains": cfg.n_chains, "n_sweeps": cfg.n_sweeps, "record_every": cfg.record_every}), flush=True)
        for chain in range(cfg.n_chains):
            source_idx = int(rng.integers(0, len(coarse)))
            u0 = coarse[source_idx : source_idx + 1].copy()
            z_edge = rng.standard_normal((1, 16, 16)).astype(np.float32)
            z_pair = rng.standard_normal((1, 16, 16)).astype(np.float32)
            z_body = rng.standard_normal((1, 16, 16)).astype(np.float32)
            state = compute_state(u0, z_edge, z_pair, z_body, models, kernel, c)
            obs0 = row_with_errors({"chain": chain, "sweep": 0, "source_idx": source_idx}, observables(state["phi"]))
            obs_rows.append(obs0)
            for writer in obs_streams:
                writer.write(obs0)
            data_stream.write({
                "patch_size": cfg.p_coarse,
                "n_coarse_passes": cfg.coarse_passes,
                "detail_patch_size": cfg.p_detail,
                "detail_updates_per_sweep": cfg.n_detail_updates_per_sweep,
                **obs0,
            })
            rms_stream.write({"sweep": 0, "local_rms_error": obs0["local_rms_error"], "long_rms_error": obs0["long_rms_error"]})
            print(json.dumps({"event": "chain_start", "chain": chain, "source_idx": source_idx, "elapsed_sec": time.perf_counter() - t0}), flush=True)
            for sweep in range(1, cfg.n_sweeps + 1):
                sweep_coarse_acc: list[int] = []
                sweep_coarse_site: list[float] = []
                sweep_coarse_site_attempts = 0
                sweep_coarse_site_accepts = 0
                sweep_coarse_log: list[float] = []
                sweep_latent_acc: list[int] = []
                sweep_latent_log: list[float] = []
                for _ in range(patches_per_sweep(16, cfg.p_coarse)):
                    x0 = int(rng.integers(0, 16))
                    y0 = int(rng.integers(0, 16))
                    u_prop, stats = controlled_patch_metropolis(state["u"][0], patch_sites(16, x0, y0, cfg.p_coarse), rng, flowtrain.ActionSpec("phi4_nn", flowtrain.LAMBDA, flowtrain.KAPPA), cfg.epsilon_c, cfg.coarse_passes)
                    proposal = compute_state(u_prop[None], state["z_edge"], state["z_pair"], state["z_body"], models, kernel, c)
                    logacc = float(proposal["logw"][0] - state["logw"][0])
                    acc = int(math.log(max(float(rng.random()), 1.0e-300)) < min(0.0, logacc))
                    if acc:
                        state = proposal
                    sweep_coarse_acc.append(acc)
                    sweep_coarse_site.append(float(stats["coarse_site_acceptance"]))
                    sweep_coarse_site_attempts += int(stats["coarse_site_attempts"])
                    sweep_coarse_site_accepts += int(stats["coarse_site_accepts"])
                    sweep_coarse_log.append(logacc)
                    coarse_move_acc.append(acc)
                    coarse_site_accs.append(float(stats["coarse_site_acceptance"]))
                    coarse_move_logs.append(logacc)
                for _ in range(cfg.n_detail_updates_per_sweep):
                    x0 = int(rng.integers(0, 16))
                    y0 = int(rng.integers(0, 16))
                    sites = patch_sites(16, x0, y0, cfg.p_detail)
                    z_edge_p = state["z_edge"].copy()
                    z_pair_p = state["z_pair"].copy()
                    z_body_p = state["z_body"].copy()
                    rho = math.sqrt(max(0.0, 1.0 - cfg.beta_z * cfg.beta_z))
                    for i, j in sites:
                        z_edge_p[0, i, j] = rho * z_edge_p[0, i, j] + cfg.beta_z * float(rng.standard_normal())
                        z_pair_p[0, i, j] = rho * z_pair_p[0, i, j] + cfg.beta_z * float(rng.standard_normal())
                        z_body_p[0, i, j] = rho * z_body_p[0, i, j] + cfg.beta_z * float(rng.standard_normal())
                    proposal = compute_state(state["u"], z_edge_p, z_pair_p, z_body_p, models, kernel, c)
                    logacc = float(proposal["logw"][0] - state["logw"][0])
                    acc = int(math.log(max(float(rng.random()), 1.0e-300)) < min(0.0, logacc))
                    if acc:
                        state = proposal
                    sweep_latent_acc.append(acc)
                    sweep_latent_log.append(logacc)
                    latent_move_acc.append(acc)
                    latent_move_logs.append(logacc)
                acc_row = {
                    "chain": chain,
                    "sweep": sweep,
                    "patch_size": cfg.p_coarse,
                    "detail_patch_size": cfg.p_detail,
                    "n_coarse_passes": cfg.coarse_passes,
                    "coarse_attempts": len(sweep_coarse_acc),
                    "coarse_fine_accepts": int(sum(sweep_coarse_acc)),
                    "coarse_fine_AR_acceptance": float(np.mean(sweep_coarse_acc)),
                    "coarse_site_attempts": sweep_coarse_site_attempts,
                    "coarse_site_accepts": sweep_coarse_site_accepts,
                    "coarse_site_acceptance": sweep_coarse_site_accepts / max(sweep_coarse_site_attempts, 1),
                    "coarse_log_accept_mean": float(np.mean(sweep_coarse_log)),
                    "coarse_log_accept_std": float(np.std(sweep_coarse_log, ddof=1)) if len(sweep_coarse_log) > 1 else 0.0,
                    "latent_attempts": len(sweep_latent_acc),
                    "latent_accepts": int(sum(sweep_latent_acc)),
                    "latent_acceptance": float(np.mean(sweep_latent_acc)),
                    "latent_log_accept_mean": float(np.mean(sweep_latent_log)),
                    "latent_log_accept_std": float(np.std(sweep_latent_log, ddof=1)) if len(sweep_latent_log) > 1 else 0.0,
                }
                sweep_acc_rows.append(acc_row)
                for writer in acc_streams:
                    writer.write(acc_row)
                latest_obs = None
                if sweep % cfg.record_every == 0:
                    latest_obs = row_with_errors({"chain": chain, "sweep": sweep, "source_idx": source_idx}, observables(state["phi"]))
                    obs_rows.append(latest_obs)
                    for writer in obs_streams:
                        writer.write(latest_obs)
                    data_stream.write({
                        "patch_size": cfg.p_coarse,
                        "n_coarse_passes": cfg.coarse_passes,
                        "detail_patch_size": cfg.p_detail,
                        "detail_updates_per_sweep": cfg.n_detail_updates_per_sweep,
                        **latest_obs,
                    })
                    rms_stream.write({"sweep": sweep, "local_rms_error": latest_obs["local_rms_error"], "long_rms_error": latest_obs["long_rms_error"]})
                if sweep % 10 == 0 or sweep == cfg.n_sweeps:
                    window_start = max(1, sweep - 9)
                    window_rows = [
                        r
                        for r in sweep_acc_rows
                        if int(r["chain"]) == chain and window_start <= int(r["sweep"]) <= sweep
                    ]
                    chain_rows = [r for r in sweep_acc_rows if int(r["chain"]) == chain]
                    window_coarse_site_attempts = sum(int(r["coarse_site_attempts"]) for r in window_rows)
                    window_coarse_site_accepts = sum(int(r["coarse_site_accepts"]) for r in window_rows)
                    window_coarse_attempts = sum(int(r["coarse_attempts"]) for r in window_rows)
                    window_coarse_fine_accepts = sum(int(r["coarse_fine_accepts"]) for r in window_rows)
                    window_latent_attempts = sum(int(r["latent_attempts"]) for r in window_rows)
                    window_latent_accepts = sum(int(r["latent_accepts"]) for r in window_rows)
                    chain_coarse_site_attempts = sum(int(r["coarse_site_attempts"]) for r in chain_rows)
                    chain_coarse_site_accepts = sum(int(r["coarse_site_accepts"]) for r in chain_rows)
                    progress = {
                        "event": "progress",
                        "chain": chain,
                        "sweep": sweep,
                        "elapsed_sec": time.perf_counter() - t0,
                        "acceptance_window_sweeps": [window_start, sweep],
                        "coarse_acceptance": window_coarse_site_accepts / max(window_coarse_site_attempts, 1),
                        "coarse_site_acceptance": window_coarse_site_accepts / max(window_coarse_site_attempts, 1),
                        "coarse_site_attempts": window_coarse_site_attempts,
                        "coarse_site_accepts": window_coarse_site_accepts,
                        "coarse_site_acceptance_sweep": sweep_coarse_site_accepts / max(sweep_coarse_site_attempts, 1),
                        "coarse_site_acceptance_cumulative_chain": chain_coarse_site_accepts / max(chain_coarse_site_attempts, 1),
                        "coarse_fine_AR_acceptance": window_coarse_fine_accepts / max(window_coarse_attempts, 1),
                        "coarse_fine_AR_attempts": window_coarse_attempts,
                        "coarse_fine_AR_accepts": window_coarse_fine_accepts,
                        "coarse_fine_AR_acceptance_sweep": float(np.mean(sweep_coarse_acc)),
                        "coarse_fine_AR_acceptance_cumulative_chain": float(np.mean([float(r["coarse_fine_AR_acceptance"]) for r in chain_rows])),
                        "latent_acceptance": window_latent_accepts / max(window_latent_attempts, 1),
                        "latent_attempts": window_latent_attempts,
                        "latent_accepts": window_latent_accepts,
                        "latent_acceptance_sweep": float(np.mean(sweep_latent_acc)),
                        "latent_acceptance_cumulative_chain": float(np.mean([float(r["latent_acceptance"]) for r in chain_rows])),
                    }
                    if latest_obs is not None:
                        progress["local_rms_error"] = latest_obs["local_rms_error"]
                    print(json.dumps(progress), flush=True)
    finally:
        for writer in stream_writers:
            writer.close()
    runtime = time.perf_counter() - t0
    move_rows = [
        {"move_type": "coarse_site", "attempts": cfg.n_chains * cfg.n_sweeps * patches_per_sweep(16, cfg.p_coarse) * cfg.coarse_passes * cfg.p_coarse * cfg.p_coarse, "acceptance": float(np.mean(coarse_site_accs)), "log_accept_mean": 0.0, "log_accept_std": 0.0},
        {"move_type": "coarse_transported_fine_AR", "attempts": len(coarse_move_acc), "acceptance": float(np.mean(coarse_move_acc)), "log_accept_mean": float(np.mean(coarse_move_logs)), "log_accept_std": float(np.std(coarse_move_logs, ddof=1))},
        {"move_type": "latent_detail_pCN_AR", "attempts": len(latent_move_acc), "acceptance": float(np.mean(latent_move_acc)), "log_accept_mean": float(np.mean(latent_move_logs)), "log_accept_std": float(np.std(latent_move_logs, ddof=1))},
    ]
    make_outputs(out, cfg, obs_rows, sweep_acc_rows, move_rows, runtime)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--footprint", type=int, required=True)
    ap.add_argument("--edge-x-checkpoint", required=True)
    ap.add_argument("--edge-y-checkpoint", required=True)
    ap.add_argument("--body-checkpoint", required=True)
    ap.add_argument("--seed", type=int, default=2026070411)
    ap.add_argument("--n-chains", type=int, default=8)
    ap.add_argument("--n-sweeps", type=int, default=2000)
    ap.add_argument("--record-every", type=int, default=20)
    ap.add_argument("--p-coarse", type=int, default=12)
    ap.add_argument("--coarse-patches-per-sweep", type=int, default=4)
    ap.add_argument("--coarse-passes", type=int, default=5)
    ap.add_argument("--epsilon-c", type=float, default=0.6)
    ap.add_argument("--p-detail", type=int, default=12)
    ap.add_argument("--beta-z", type=float, default=0.4)
    ap.add_argument("--n-detail-updates-per-sweep", type=int, default=2)
    ap.add_argument("--preflight-only", action="store_true")
    args = ap.parse_args()
    cfg = ChainConfig(
        candidate=args.candidate,
        output_dir=args.output_dir,
        footprint=args.footprint,
        edge_x_checkpoint=args.edge_x_checkpoint,
        edge_y_checkpoint=args.edge_y_checkpoint,
        body_checkpoint=args.body_checkpoint,
        seed=args.seed,
        n_chains=args.n_chains,
        n_sweeps=args.n_sweeps,
        record_every=args.record_every,
        p_coarse=args.p_coarse,
        coarse_patches_per_sweep=args.coarse_patches_per_sweep,
        coarse_passes=args.coarse_passes,
        epsilon_c=args.epsilon_c,
        p_detail=args.p_detail,
        beta_z=args.beta_z,
        n_detail_updates_per_sweep=args.n_detail_updates_per_sweep,
    )
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pf = preflight(cfg)
    write_json(out / "preflight.json", pf)
    print(json.dumps(pf, indent=2), flush=True)
    if args.preflight_only:
        return 0
    run_chain(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
