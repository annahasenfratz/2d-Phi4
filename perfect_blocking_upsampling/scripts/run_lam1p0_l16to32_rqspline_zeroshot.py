#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PKG = PROJECT_ROOT / "perfect_blocking_upsampling"
SCRIPT_DIR = PKG / "scripts"
sys.path.insert(0, str(PKG / "src"))
sys.path.insert(0, str(SCRIPT_DIR))
os.environ.setdefault("MPLCONFIGDIR", str((PKG / "logs" / "mplconfig").resolve()))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from perfect_blocking_upsampling.conv_pair import build_procedural_conv_flow  # noqa: E402
from perfect_blocking_upsampling.conv_spline_pair import build_procedural_conv_spline_flow  # noqa: E402
from perfect_blocking_upsampling.actions import action_total  # noqa: E402
from perfect_blocking_upsampling.io import ActionSpec  # noqa: E402
from perfect_blocking_upsampling.blocking import (  # noqa: E402
    apply_kernel,
    assemble_psi,
    inverse_kernel,
    load_kernel_matrix,
    load_phi,
    write_json,
)


ETA_SCALE = 2.0**0.125
MAIN_COLUMNS = [
    "chain_id",
    "sweep",
    "source_config_index",
    "source_native_L32_index",
    "L",
    "volume",
    "action_density",
    "total_action",
    "phi2",
    "phi4",
    "NN",
    "diag",
    "2nn",
    "m",
    "m2",
    "m4",
    "G_pmin_x_cfg",
    "G_pmin_y_cfg",
    "nonfinite_count",
]
ACCEPTANCE_COLUMNS = [
    "sweep",
    "update_mode",
    "coarse_acceptance",
    "coarse_proposals",
    "coarse_accepts",
    "coarse_acceptance_cumulative",
    "coarse_proposals_cumulative",
    "coarse_accepts_cumulative",
    "detail_acceptance",
    "detail_proposals",
    "detail_accepts",
    "detail_acceptance_cumulative",
    "detail_proposals_cumulative",
    "detail_accepts_cumulative",
    "conditional_flow_refreshes",
]
PER_SWEEP_COLUMNS = [
    "chain_id",
    "update_mode",
    "coarse_source",
    "source_config_index",
    "source_native_L32_index",
    "sweep",
    "coarse_acceptance",
    "coarse_proposals",
    "coarse_accepts",
    "detail_acceptance",
    "detail_proposals",
    "detail_accepts",
    "acceptance_detail",
    "proposed_detail_updates",
    "accepted_detail_updates",
    "conditional_flow_refreshes",
    "reblocking_max_error",
    "reblocking_rms_error",
    "coarse_reblocking_max_error",
    "coarse_reblocking_rms_error",
    "nonfinite_count",
]
G_COLUMNS = [
    "chain_id",
    "sweep",
    "source_config_index",
    "source_native_L32_index",
    "L",
    "volume",
    "G_00",
    "G_10",
    "G_01",
    "G_pmin_avg",
]


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        columns = []
        for row in rows:
            for key in row:
                if key not in columns:
                    columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


class AffineARDetailFlow(torch.nn.Module):
    def __init__(self, *, lattice_size: int, layers: int, hidden: int, kernel_size: int, log_scale_bound: float):
        super().__init__()
        kwargs = {
            "target_channels": 1,
            "lattice_size": int(lattice_size),
            "n_coupling_layers": int(layers),
            "conv_hidden_channels": int(hidden),
            "conv_kernel_size": int(kernel_size),
            "log_scale_bound": float(log_scale_bound),
        }
        self.lattice_size = int(lattice_size)
        self.flows = torch.nn.ModuleList(
            [
                build_procedural_conv_flow(cond_channels=1, **kwargs),
                build_procedural_conv_flow(cond_channels=2, **kwargs),
                build_procedural_conv_flow(cond_channels=3, **kwargs),
            ]
        )

    @staticmethod
    def cond(coarse: torch.Tensor, d: torch.Tensor, stage: int) -> torch.Tensor:
        if stage == 0:
            return coarse[:, None].flatten(1)
        if stage == 1:
            return torch.cat([coarse[:, None], d[:, 0:1]], dim=1).flatten(1)
        return torch.cat([coarse[:, None], d[:, 0:1], d[:, 1:2]], dim=1).flatten(1)


class SplineARDetailFlow(torch.nn.Module):
    def __init__(
        self,
        *,
        lattice_size: int,
        layers: int,
        hidden: int,
        kernel_size: int,
        num_bins: int,
        tail_bound: float,
        min_bin_width: float,
        min_bin_height: float,
        min_derivative: float,
    ):
        super().__init__()
        kwargs = {
            "target_channels": 1,
            "lattice_size": int(lattice_size),
            "n_coupling_layers": int(layers),
            "conv_hidden_channels": int(hidden),
            "conv_kernel_size": int(kernel_size),
            "num_bins": int(num_bins),
            "tail_bound": float(tail_bound),
            "min_bin_width": float(min_bin_width),
            "min_bin_height": float(min_bin_height),
            "min_derivative": float(min_derivative),
        }
        self.lattice_size = int(lattice_size)
        self.flows = torch.nn.ModuleList(
            [
                build_procedural_conv_spline_flow(cond_channels=1, **kwargs),
                build_procedural_conv_spline_flow(cond_channels=2, **kwargs),
                build_procedural_conv_spline_flow(cond_channels=3, **kwargs),
            ]
        )

    @staticmethod
    def cond(coarse: torch.Tensor, d: torch.Tensor, stage: int) -> torch.Tensor:
        return AffineARDetailFlow.cond(coarse, d, stage)


class ResidualSplineARDetailFlow(torch.nn.Module):
    def __init__(self, affine_base: AffineARDetailFlow, spline: SplineARDetailFlow):
        super().__init__()
        self.affine_base = affine_base
        self.spline = spline
        self.lattice_size = int(spline.lattice_size)

    @staticmethod
    def cond(coarse: torch.Tensor, d: torch.Tensor, stage: int) -> torch.Tensor:
        return SplineARDetailFlow.cond(coarse, d, stage)

    def sample(self, coarse: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        n = coarse.shape[0]
        lc = self.lattice_size
        d = coarse.new_zeros((n, 3, lc, lc))
        logq = coarse.new_zeros(n)
        zmax = coarse.new_zeros(n)
        logdet_total = coarse.new_zeros(n)
        dim = lc * lc
        for stage in range(3):
            z = torch.randn((n, dim), device=coarse.device, dtype=coarse.dtype)
            log_base = -0.5 * (z * z + math.log(2.0 * math.pi)).sum(dim=1)
            cond_affine = self.affine_base.cond(coarse, d, stage)
            x_affine, affine_logdet = self.affine_base.flows[stage].forward(z, cond_affine)
            cond_spline = self.cond(coarse, d, stage)
            x, spline_logdet = self.spline.flows[stage].forward(x_affine, cond_spline)
            d[:, stage] = x.reshape(n, lc, lc)
            total_logdet = affine_logdet + spline_logdet
            logq = logq + log_base - total_logdet
            logdet_total = logdet_total + total_logdet
            zmax = torch.maximum(zmax, torch.amax(torch.abs(z), dim=1))
        return d, logq, zmax, logdet_total

    def log_prob(self, coarse: torch.Tensor, detail: torch.Tensor) -> torch.Tensor:
        lp = coarse.new_zeros(coarse.shape[0])
        reconstructed = coarse.new_zeros(detail.shape)
        for stage in range(3):
            cond_spline = self.cond(coarse, detail, stage)
            pre_spline, spline_logdet = self.spline.flows[stage].inverse(detail[:, stage].flatten(1), cond_spline)
            cond_affine = self.affine_base.cond(coarse, reconstructed, stage)
            z, affine_logdet = self.affine_base.flows[stage].inverse(pre_spline, cond_affine)
            log_base = -0.5 * (z * z + math.log(2.0 * math.pi)).sum(dim=1)
            lp = lp + log_base + affine_logdet + spline_logdet
            reconstructed[:, stage] = detail[:, stage]
        return lp


def build_model_from_checkpoint(ckpt: dict[str, Any], lattice_size: int, device: torch.device) -> tuple[ResidualSplineARDetailFlow, dict[str, Any]]:
    cfg = ckpt["config"]
    if cfg.get("initialization_mode") == "fresh" or not cfg.get("resume_checkpoint"):
        scfg = cfg
    else:
        src_affine = torch.load(PROJECT_ROOT / cfg["resume_checkpoint"], map_location=device, weights_only=False)
        scfg = src_affine["config"]
    affine = AffineARDetailFlow(
        lattice_size=lattice_size,
        layers=int(scfg["layers"]),
        hidden=int(scfg["hidden_channels"]),
        kernel_size=int(scfg["conv_kernel_size"]),
        log_scale_bound=float(scfg["log_scale_bound"]),
    )
    spline = SplineARDetailFlow(
        lattice_size=lattice_size,
        layers=int(cfg["layers"]),
        hidden=int(cfg["hidden_channels"]),
        kernel_size=int(cfg["conv_kernel_size"]),
        num_bins=int(cfg["num_bins"]),
        tail_bound=float(cfg["tail_bound"]),
        min_bin_width=float(cfg["min_bin_width"]),
        min_bin_height=float(cfg["min_bin_height"]),
        min_derivative=float(cfg["min_derivative"]),
    )
    model = ResidualSplineARDetailFlow(affine, spline).to(device)
    raw_state = ckpt["model_state"]
    alias_keys = [key for key in raw_state if key.startswith("flows.")]
    state = {key: val for key, val in raw_state.items() if not key.startswith("flows.")}
    result = model.load_state_dict(state, strict=True)
    model.eval()
    return model, {
        "load_strict": True,
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
        "source_resume_checkpoint": cfg.get("resume_checkpoint"),
        "source_spline_settings": ckpt.get("spline_settings", {}),
        "stripped_duplicate_alias_keys": len(alias_keys),
        "initialization_mode": cfg.get("initialization_mode", "transferred"),
    }


def stationary_stats(source_stats: dict[str, Any], lc: int) -> dict[str, Any]:
    if {"coarse_mean", "coarse_std", "detail_mean", "detail_std"}.issubset(source_stats):
        detail_mean = np.asarray(source_stats["detail_mean"], dtype=np.float32).reshape(3)
        detail_std = np.asarray(source_stats["detail_std"], dtype=np.float32).reshape(3)
        return {
            "coarse_mean": float(source_stats["coarse_mean"]),
            "coarse_std": float(source_stats["coarse_std"]),
            "detail_mean": detail_mean,
            "detail_std": detail_std,
            "method": str(source_stats.get("method", "pre-resolved scalar coarse and per-sector detail statistics")),
            "lattice_size": int(source_stats.get("lattice_size", lc)),
        }
    c_mean = np.asarray(source_stats["coarse"]["mean"], dtype=np.float32).reshape(8, 8)
    c_std = np.asarray(source_stats["coarse"]["std"], dtype=np.float32).reshape(8, 8)
    d_mean = np.asarray(source_stats["detail"]["mean"], dtype=np.float32).reshape(3, 8, 8)
    d_std = np.asarray(source_stats["detail"]["std"], dtype=np.float32).reshape(3, 8, 8)
    return {
        "coarse_mean": float(np.mean(c_mean)),
        "coarse_std": float(np.mean(c_std)),
        "detail_mean": np.mean(d_mean, axis=(1, 2)).astype(np.float32),
        "detail_std": np.mean(d_std, axis=(1, 2)).astype(np.float32),
        "method": "stationary spatial average of L8-to-L16 standardization statistics, applied per site/sector on L16-to-L32",
        "lattice_size": int(lc),
    }


def sample_model_lattice(
    model: ResidualSplineARDetailFlow,
    coarse_phys: np.ndarray,
    stats: dict[str, Any],
    *,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    torch.manual_seed(seed)
    coarse_std = ((coarse_phys - stats["coarse_mean"]) / stats["coarse_std"]).astype(np.float32)
    mean = np.asarray(stats["detail_mean"], dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.asarray(stats["detail_std"], dtype=np.float32).reshape(1, 3, 1, 1)
    log_jac_const = -float(coarse_phys.shape[1] * coarse_phys.shape[2] * np.sum(np.log(std.reshape(3))))
    details: list[np.ndarray] = []
    logqs: list[np.ndarray] = []
    zmaxs: list[np.ndarray] = []
    logdets: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(coarse_std), batch_size):
            cb = torch.from_numpy(coarse_std[start : start + batch_size]).to(device)
            d_std, logq_std, zmax, logdet = model.sample(cb)
            d_phys = d_std.detach().cpu().numpy().astype(np.float32) * std + mean
            details.append(d_phys.astype(np.float32))
            logqs.append((logq_std.detach().cpu().numpy() + log_jac_const).astype(np.float64))
            zmaxs.append(zmax.detach().cpu().numpy().astype(np.float32))
            logdets.append((logdet.detach().cpu().numpy() - log_jac_const).astype(np.float64))
    return np.concatenate(details), np.concatenate(logqs), np.concatenate(zmaxs), np.concatenate(logdets)


def log_prob_model_lattice(
    model: ResidualSplineARDetailFlow,
    coarse_phys: np.ndarray,
    detail_phys: np.ndarray,
    stats: dict[str, Any],
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    coarse_std = ((coarse_phys - stats["coarse_mean"]) / stats["coarse_std"]).astype(np.float32)
    mean = np.asarray(stats["detail_mean"], dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.asarray(stats["detail_std"], dtype=np.float32).reshape(1, 3, 1, 1)
    detail_std = ((detail_phys - mean) / std).astype(np.float32)
    log_jac_const = -float(coarse_phys.shape[1] * coarse_phys.shape[2] * np.sum(np.log(std.reshape(3))))
    vals: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(coarse_std), batch_size):
            cb = torch.from_numpy(coarse_std[start : start + batch_size]).to(device)
            db = torch.from_numpy(detail_std[start : start + batch_size]).to(device)
            vals.append((model.log_prob(cb, db).detach().cpu().numpy() + log_jac_const).astype(np.float64))
    return np.concatenate(vals)


def per_config_observables(phi: np.ndarray, action: ActionSpec) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    arr = np.asarray(phi, dtype=np.float64)
    n, L, _ = arr.shape
    V = L * L
    total_action = action_total(arr, action).astype(np.float64)
    m = arr.mean(axis=(1, 2))
    phi2 = np.mean(arr * arr, axis=(1, 2))
    phi4 = np.mean(arr**4, axis=(1, 2))
    nn = 0.5 * (
        np.mean(arr * np.roll(arr, -1, axis=1), axis=(1, 2))
        + np.mean(arr * np.roll(arr, -1, axis=2), axis=(1, 2))
    )
    two = 0.5 * (
        np.mean(arr * np.roll(arr, -2, axis=1), axis=(1, 2))
        + np.mean(arr * np.roll(arr, -2, axis=2), axis=(1, 2))
    )
    diag = np.mean(arr * np.roll(np.roll(arr, -1, axis=1), -1, axis=2), axis=(1, 2))
    ft = np.fft.fft2(arr, axes=(1, 2))
    g00 = np.abs(ft[:, 0, 0]) ** 2 / V
    g10 = np.abs(ft[:, 1, 0]) ** 2 / V
    g01 = np.abs(ft[:, 0, 1]) ** 2 / V
    obs = {
        "action_density": total_action / V,
        "total_action": total_action,
        "phi2": phi2,
        "phi4": phi4,
        "local_kurtosis_ratio": phi4 / np.maximum(phi2 * phi2, 1.0e-300),
        "NN": nn,
        "diag": diag,
        "2nn": two,
        "m": m,
        "m2": m * m,
        "m4": m**4,
    }
    g = {"G_00": g00, "G_10": g10, "G_01": g01, "G_pmin_avg": 0.5 * (g10 + g01)}
    return obs, g


def append_measurement_rows(
    main_rows: list[dict[str, Any]],
    g_rows: list[dict[str, Any]],
    phi: np.ndarray,
    *,
    sweep: int,
    source_indices: np.ndarray,
    action: ActionSpec,
) -> None:
    obs, g = per_config_observables(phi, action)
    L = int(phi.shape[1])
    V = L * L
    nonfinite = np.sum(~np.isfinite(phi).reshape(len(phi), -1), axis=1)
    for i in range(len(phi)):
        main_rows.append(
            {
                "chain_id": i,
                "sweep": sweep,
                "source_config_index": int(source_indices[i]),
                "source_native_L32_index": -1,
                "L": L,
                "volume": V,
                "action_density": float(obs["action_density"][i]),
                "total_action": float(obs["total_action"][i]),
                "phi2": float(obs["phi2"][i]),
                "phi4": float(obs["phi4"][i]),
                "NN": float(obs["NN"][i]),
                "diag": float(obs["diag"][i]),
                "2nn": float(obs["2nn"][i]),
                "m": float(obs["m"][i]),
                "m2": float(obs["m2"][i]),
                "m4": float(obs["m4"][i]),
                "G_pmin_x_cfg": float(g["G_10"][i]),
                "G_pmin_y_cfg": float(g["G_01"][i]),
                "nonfinite_count": int(nonfinite[i]),
            }
        )
        g_rows.append(
            {
                "chain_id": i,
                "sweep": sweep,
                "source_config_index": int(source_indices[i]),
                "source_native_L32_index": -1,
                "L": L,
                "volume": V,
                "G_00": float(g["G_00"][i]),
                "G_10": float(g["G_10"][i]),
                "G_01": float(g["G_01"][i]),
                "G_pmin_avg": float(g["G_pmin_avg"][i]),
            }
        )


def summary_stats(values: np.ndarray) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
        "median": float(np.quantile(x, 0.50)),
        "p01": float(np.quantile(x, 0.01)),
        "p05": float(np.quantile(x, 0.05)),
        "p10": float(np.quantile(x, 0.10)),
        "p90": float(np.quantile(x, 0.90)),
        "p95": float(np.quantile(x, 0.95)),
        "p99": float(np.quantile(x, 0.99)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def longest_rejection_streak(accepted_by_sweep_chain: list[np.ndarray]) -> int:
    arr = np.asarray(accepted_by_sweep_chain, dtype=bool)
    longest = 0
    for ch in range(arr.shape[1]):
        current = 0
        for ok in arr[:, ch]:
            if ok:
                current = 0
            else:
                current += 1
                longest = max(longest, current)
    return int(longest)


def run_preflight(
    model: ResidualSplineARDetailFlow,
    coarse: np.ndarray,
    kernel: np.ndarray,
    stats: dict[str, Any],
    *,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    n = min(4, len(coarse))
    detail, logq, zmax, logdet = sample_model_lattice(model, coarse[:n], stats, batch_size=batch_size, device=device, seed=seed)
    psi = assemble_psi(coarse[:n], detail)
    phi, inv = inverse_kernel(psi, kernel)
    reb = apply_kernel(phi, kernel)[:, 0::2, 0::2] - coarse[:n]
    logp = log_prob_model_lattice(model, coarse[:n], detail, stats, batch_size=batch_size, device=device)
    lc = int(coarse.shape[1])
    with torch.no_grad():
        cb = torch.from_numpy(((coarse[:n] - stats["coarse_mean"]) / stats["coarse_std"]).astype(np.float32)).to(device)
        db = torch.from_numpy(((detail - np.asarray(stats["detail_mean"]).reshape(1, 3, 1, 1)) / np.asarray(stats["detail_std"]).reshape(1, 3, 1, 1)).astype(np.float32)).to(device)
        roundtrip_errors = []
        logdet_roundtrip_errors = []
        for stage in range(3):
            cond = model.cond(cb, db, stage)
            z, ild = model.spline.flows[stage].inverse(db[:, stage].flatten(1), cond)
            back, fld = model.spline.flows[stage].forward(z, cond)
            roundtrip_errors.append(float(torch.max(torch.abs(back - db[:, stage].flatten(1))).detach().cpu()))
            logdet_roundtrip_errors.append(float(torch.max(torch.abs(fld + ild)).detach().cpu()))
    return {
        "n_preflight": n,
        "model_accepts_l16_coarse": True,
        "generated_l32_shape": list(phi.shape),
        "nonfinite_count": int(np.sum(~np.isfinite(phi)) + np.sum(~np.isfinite(detail))),
        "reblocking_max_error": float(np.max(np.abs(reb))),
        "reblocking_rms_error": float(np.sqrt(np.mean(reb.astype(np.float64) ** 2))),
        "max_abs_z": float(np.max(zmax)),
        "logq_mean": float(np.mean(logq)),
        "logprob_roundtrip_max_abs": float(np.max(np.abs(logp - logq))),
        "spline_forward_inverse_max_error": float(max(roundtrip_errors)),
        "spline_logdet_roundtrip_max_error": float(max(logdet_roundtrip_errors)),
        "inverse_kernel": inv,
        "lattice_size": lc,
        "logdet_stats": summary_stats(logdet),
    }


def plot_hist(path: Path, series: dict[str, np.ndarray], key: str) -> None:
    vals = [np.asarray(v, dtype=np.float64) for v in series.values()]
    concat = np.concatenate(vals)
    lo, hi = np.quantile(concat, [0.005, 0.995])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = float(np.min(concat)), float(np.max(concat))
    pad = 0.05 * max(hi - lo, 1.0e-12)
    bins = 70
    fig, ax = plt.subplots(figsize=(6.2, 3.9), constrained_layout=True)
    styles = {
        "native_L32": dict(color="black", lw=2.1, ls="-"),
        "sweep_0": dict(color="0.35", lw=1.5, ls="--"),
        "sweep_10": dict(color="#1b9e77", lw=1.5, ls="-"),
        "sweep_100": dict(color="#d95f02", lw=1.7, ls="-"),
    }
    for label, x in series.items():
        hist, edges = np.histogram(x, bins=bins, range=(lo - pad, hi + pad), density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax.step(centers, hist, where="mid", label=label, **styles.get(label, {}))
    ax.set_xlabel(key)
    ax.set_ylabel("density")
    ax.legend(frameon=False, fontsize=8)
    ax.tick_params(direction="in", top=True, right=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"), dpi=180)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, default=Path("perfect_blocking_upsampling/runs/lam1p0/lam1p0_L8to16_kf0p340301_kc0p340301_7x7_phi2_nn_guarded_autoregressive_detail_8layer48_rqspline_localreg_from_affine_ep137_20260717T125835Z/checkpoints/checkpoint_best.pt"))
    ap.add_argument("--kernel-path", type=Path, default=Path("perfect_blocking/perfect_blocking_lam1p0/kernels/selected_for_upscaling/best_7x7_phi2_nn_guarded_eta_included.json"))
    ap.add_argument("--coarse-config-source", type=Path, default=Path("data/configs_phi4_2d/lam1p0_kappac0p340301_L16/configs.npz"))
    ap.add_argument("--native-reference-source", type=Path, default=Path("data/configs_phi4_2d/lam1p0_kappac0p340301_L32/configs.npz"))
    ap.add_argument("--n-chains", type=int, default=64)
    ap.add_argument("--n-sweeps", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--random-seed", type=int, default=2026071716)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    run = args.run_dir
    for sub in ["logs", "observables", "plots", "summaries", "debug", "checkpoints"]:
        (run / sub).mkdir(parents=True, exist_ok=True)
    started = time.time()
    device = torch.device(args.device)
    kernel, kernel_meta = load_kernel_matrix(PROJECT_ROOT / args.kernel_path)
    eta_sum = float(kernel.sum())
    if not bool(kernel_meta.get("kernel_coefficients_include_eta_scale", False)):
        raise RuntimeError("kernel is not eta-included")
    if not np.isclose(eta_sum, ETA_SCALE, atol=1.0e-10):
        raise RuntimeError(f"kernel sum {eta_sum} != eta_scale {ETA_SCALE}")

    ck = torch.load(PROJECT_ROOT / args.checkpoint, map_location=device, weights_only=False)
    model, load_report = build_model_from_checkpoint(ck, lattice_size=16, device=device)
    stats = stationary_stats(ck["state"]["stats"], lc=16)

    coarse_all = load_phi(PROJECT_ROOT / args.coarse_config_source)
    native_all = load_phi(PROJECT_ROOT / args.native_reference_source)
    if coarse_all.shape[1:] != (16, 16):
        raise RuntimeError(f"coarse input is not L16: {coarse_all.shape}")
    if native_all.shape[1:] != (32, 32):
        raise RuntimeError(f"native reference is not L32: {native_all.shape}")

    rng = np.random.default_rng(args.random_seed)
    source_indices = rng.choice(len(coarse_all), size=min(args.n_chains, len(coarse_all)), replace=False)
    coarse = coarse_all[source_indices]
    action = ActionSpec("phi4_nn", 1.0, 0.340301)

    preflight = run_preflight(model, coarse, kernel, stats, batch_size=args.batch_size, device=device, seed=args.random_seed + 100)
    if preflight["nonfinite_count"] != 0:
        raise RuntimeError(f"preflight generated nonfinite fields: {preflight['nonfinite_count']}")
    if preflight["reblocking_max_error"] > 5.0e-6:
        raise RuntimeError(f"preflight reblocking error too large: {preflight['reblocking_max_error']}")
    write_json(run / "debug" / "preflight.json", preflight)
    write_json(run / "debug" / "checkpoint_load_report.json", load_report)
    write_json(run / "debug" / "stationary_transfer_stats.json", stats)

    main_rows: list[dict[str, Any]] = []
    g_rows: list[dict[str, Any]] = []
    per_rows: list[dict[str, Any]] = []
    acc_rows: list[dict[str, Any]] = []
    proposal_rows: list[dict[str, Any]] = []
    selected_phi: dict[int, np.ndarray] = {}
    selected_sweeps = {0, 1, 5, 10, 25, 50, 100}

    detail, logq, zmax, logdet = sample_model_lattice(
        model, coarse, stats, batch_size=args.batch_size, device=device, seed=args.random_seed + 1000
    )
    psi = assemble_psi(coarse, detail)
    phi, inv = inverse_kernel(psi, kernel)
    reb = apply_kernel(phi, kernel)[:, 0::2, 0::2] - coarse
    current_s = action_total(phi, action).astype(np.float64)
    current_logq = logq.astype(np.float64)
    cumulative_accepts = 0
    cumulative_props = 0
    append_measurement_rows(main_rows, g_rows, phi, sweep=0, source_indices=source_indices, action=action)
    selected_phi[0] = phi.copy()
    acc_rows.append({k: 0 for k in ACCEPTANCE_COLUMNS} | {"sweep": 0, "update_mode": "detail", "detail_acceptance": "", "coarse_acceptance": "", "conditional_flow_refreshes": 1})
    for i in range(len(coarse)):
        per_rows.append(
            {
                "chain_id": i,
                "update_mode": "detail",
                "coarse_source": "native_L16",
                "source_config_index": int(source_indices[i]),
                "source_native_L32_index": -1,
                "sweep": 0,
                "coarse_acceptance": "",
                "coarse_proposals": 0,
                "coarse_accepts": 0,
                "detail_acceptance": "",
                "detail_proposals": 0,
                "detail_accepts": 0,
                "acceptance_detail": "",
                "proposed_detail_updates": 0,
                "accepted_detail_updates": 0,
                "conditional_flow_refreshes": 1,
                "reblocking_max_error": float(np.max(np.abs(reb[i]))),
                "reblocking_rms_error": float(np.sqrt(np.mean(reb[i].astype(np.float64) ** 2))),
                "coarse_reblocking_max_error": float(np.max(np.abs(reb[i]))),
                "coarse_reblocking_rms_error": float(np.sqrt(np.mean(reb[i].astype(np.float64) ** 2))),
                "nonfinite_count": int(np.sum(~np.isfinite(phi[i]))),
            }
        )

    patch_rng = np.random.default_rng(args.random_seed + 2000)
    accepted_by_sweep: list[np.ndarray] = []
    for sweep in range(1, args.n_sweeps + 1):
        prop_detail, prop_logq, prop_zmax, prop_logdet = sample_model_lattice(
            model, coarse, stats, batch_size=args.batch_size, device=device, seed=args.random_seed + 3000 + sweep
        )
        prop_phi, _ = inverse_kernel(assemble_psi(coarse, prop_detail), kernel)
        prop_s = action_total(prop_phi, action).astype(np.float64)
        delta_s = prop_s - current_s
        log_accept = -prop_s + current_s + current_logq - prop_logq
        accept_prob = np.exp(np.minimum(log_accept, 0.0))
        accepted = np.log(patch_rng.random(len(coarse))) < np.minimum(log_accept, 0.0)
        if np.any(accepted):
            detail[accepted] = prop_detail[accepted]
            phi[accepted] = prop_phi[accepted]
            current_s[accepted] = prop_s[accepted]
            current_logq[accepted] = prop_logq[accepted]
        accepted_by_sweep.append(accepted.copy())
        accepts = int(np.sum(accepted))
        cumulative_accepts += accepts
        cumulative_props += len(coarse)
        append_measurement_rows(main_rows, g_rows, phi, sweep=sweep, source_indices=source_indices, action=action)
        if sweep in selected_sweeps:
            selected_phi[sweep] = phi.copy()
        acc_rows.append(
            {
                "sweep": sweep,
                "update_mode": "detail",
                "coarse_acceptance": "",
                "coarse_proposals": 0,
                "coarse_accepts": 0,
                "coarse_acceptance_cumulative": "",
                "coarse_proposals_cumulative": 0,
                "coarse_accepts_cumulative": 0,
                "detail_acceptance": float(accepts / len(coarse)),
                "detail_proposals": len(coarse),
                "detail_accepts": accepts,
                "detail_acceptance_cumulative": float(cumulative_accepts / max(cumulative_props, 1)),
                "detail_proposals_cumulative": cumulative_props,
                "detail_accepts_cumulative": cumulative_accepts,
                "conditional_flow_refreshes": 1,
            }
        )
        reb = apply_kernel(phi, kernel)[:, 0::2, 0::2] - coarse
        for i in range(len(coarse)):
            per_rows.append(
                {
                    "chain_id": i,
                    "update_mode": "detail",
                    "coarse_source": "native_L16",
                    "source_config_index": int(source_indices[i]),
                    "source_native_L32_index": -1,
                    "sweep": sweep,
                    "coarse_acceptance": "",
                    "coarse_proposals": 0,
                    "coarse_accepts": 0,
                    "detail_acceptance": float(accepts / len(coarse)),
                    "detail_proposals": len(coarse),
                    "detail_accepts": accepts,
                    "acceptance_detail": float(accepts / len(coarse)),
                    "proposed_detail_updates": 1,
                    "accepted_detail_updates": int(accepted[i]),
                    "conditional_flow_refreshes": 1,
                    "reblocking_max_error": float(np.max(np.abs(reb[i]))),
                    "reblocking_rms_error": float(np.sqrt(np.mean(reb[i].astype(np.float64) ** 2))),
                    "coarse_reblocking_max_error": float(np.max(np.abs(reb[i]))),
                    "coarse_reblocking_rms_error": float(np.sqrt(np.mean(reb[i].astype(np.float64) ** 2))),
                    "nonfinite_count": int(np.sum(~np.isfinite(phi[i]))),
                }
            )
            proposal_rows.append(
                {
                    "sweep": sweep,
                    "chain_id": i,
                    "source_config_index": int(source_indices[i]),
                    "DeltaS": float(delta_s[i]),
                    "log_q_forward": float(prop_logq[i]),
                    "log_q_reverse": float(current_logq[i]),
                    "log_q_ratio": float(current_logq[i] - prop_logq[i]),
                    "log_acceptance": float(log_accept[i]),
                    "acceptance_probability": float(accept_prob[i]),
                    "accepted": int(accepted[i]),
                    "proposal_total_action": float(prop_s[i]),
                    "current_total_action_before_accept": float((prop_s - delta_s)[i]),
                    "max_abs_z": float(prop_zmax[i]),
                    "log_jacobian": float(prop_logdet[i]),
                }
            )

    write_csv(run / "observables" / "main_per_sweep_measurements.csv", main_rows, MAIN_COLUMNS)
    write_csv(run / "observables" / "Gk_per_sweep_measurements.csv", g_rows, G_COLUMNS)
    write_csv(run / "observables" / "acceptance_history.csv", acc_rows, ACCEPTANCE_COLUMNS)
    write_csv(run / "observables" / "per_sweep_observables.csv", per_rows, PER_SWEEP_COLUMNS)
    write_csv(run / "observables" / "proposal_diagnostics.csv", proposal_rows)

    native_obs, native_g = per_config_observables(native_all, action)
    native_rows = []
    for key, vals in native_obs.items():
        native_rows.append({"observable": key, **summary_stats(vals)})
    for key, vals in native_g.items():
        native_rows.append({"observable": key, **summary_stats(vals)})
    write_csv(run / "observables" / "native_L32_reference_summary.csv", native_rows)

    diagnostics_rows = []
    if proposal_rows:
        delta_all = np.asarray([r["DeltaS"] for r in proposal_rows], dtype=np.float64)
        loga_all = np.asarray([r["log_acceptance"] for r in proposal_rows], dtype=np.float64)
        diagnostics_rows.append({"quantity": "DeltaS", **summary_stats(delta_all)})
        diagnostics_rows.append({"quantity": "log_acceptance", **summary_stats(loga_all)})
        diagnostics_rows.append(
            {
                "quantity": "tail_flags",
                "frac_logA_lt_minus5": float(np.mean(loga_all < -5.0)),
                "frac_logA_lt_minus10": float(np.mean(loga_all < -10.0)),
                "frac_logA_lt_minus20": float(np.mean(loga_all < -20.0)),
                "longest_rejection_streak": longest_rejection_streak(accepted_by_sweep),
            }
        )
    write_csv(run / "observables" / "tail_diagnostics.csv", diagnostics_rows)

    hist_keys = ["action_density", "phi4", "local_kurtosis_ratio", "NN", "m2", "m4"]
    sweep_obs = {s: per_config_observables(p, action)[0] for s, p in selected_phi.items() if s in {0, 10, 100}}
    for key in hist_keys:
        series = {"native_L32": native_obs[key]}
        for s in [0, 10, 100]:
            if s in sweep_obs:
                series[f"sweep_{s}"] = sweep_obs[s][key]
        plot_hist(run / "plots" / f"{key}_native_vs_sweeps", series, key)
    if proposal_rows:
        pr = {s: [r for r in proposal_rows if r["sweep"] == s] for s in [1, 10, 100]}
        plot_hist(run / "plots" / "DeltaS_native_vs_sweeps", {f"sweep_{s}": np.asarray([r["DeltaS"] for r in rows]) for s, rows in pr.items() if rows}, "DeltaS")
        plot_hist(run / "plots" / "log_acceptance_native_vs_sweeps", {f"sweep_{s}": np.asarray([r["log_acceptance"] for r in rows]) for s, rows in pr.items() if rows}, "log_acceptance")

    final_acceptance = float(cumulative_accepts / max(cumulative_props, 1))
    sweep_comp_rows = []
    for s in [0, 1, 5, 10, 25, 50, 100]:
        if s not in selected_phi:
            continue
        obs_s, g_s = per_config_observables(selected_phi[s], action)
        for key in ["action_density", "phi2", "phi4", "local_kurtosis_ratio", "NN", "diag", "2nn", "m2", "m4"]:
            sweep_comp_rows.append(
                {
                    "sweep": s,
                    "observable": key,
                    "generated_mean": float(np.mean(obs_s[key])),
                    "native_mean": float(np.mean(native_obs[key])),
                    "mean_shift_native_sigma": float((np.mean(obs_s[key]) - np.mean(native_obs[key])) / max(np.std(native_obs[key], ddof=1), 1.0e-300)),
                    "generated_std": float(np.std(obs_s[key], ddof=1)),
                    "native_std": float(np.std(native_obs[key], ddof=1)),
                }
            )
        sweep_comp_rows.append(
            {
                "sweep": s,
                "observable": "G_pmin_avg",
                "generated_mean": float(np.mean(g_s["G_pmin_avg"])),
                "native_mean": float(np.mean(native_g["G_pmin_avg"])),
                "mean_shift_native_sigma": float((np.mean(g_s["G_pmin_avg"]) - np.mean(native_g["G_pmin_avg"])) / max(np.std(native_g["G_pmin_avg"], ddof=1), 1.0e-300)),
                "generated_std": float(np.std(g_s["G_pmin_avg"], ddof=1)),
                "native_std": float(np.std(native_g["G_pmin_avg"], ddof=1)),
            }
        )
    write_csv(run / "observables" / "sweep_native_comparison.csv", sweep_comp_rows)

    summary = {
        "status": "completed",
        "run_dir": str(run),
        "source_checkpoint": str(args.checkpoint),
        "source_checkpoint_resolution": "20260717T125835Z selected; 20260717T125627Z was an aborted hidden-transfer diagnostic without checkpoints",
        "kernel_path": str(args.kernel_path),
        "kernel_sum": eta_sum,
        "eta_scale": ETA_SCALE,
        "kernel_coefficients_include_eta_scale": True,
        "no_extra_eta_multiplier": True,
        "native_L16_coarse_count": int(len(coarse_all)),
        "native_L32_reference_count": int(len(native_all)),
        "n_chains": int(len(coarse)),
        "n_sweeps": int(args.n_sweeps),
        "patch_geometry": "full fixed-coarse detail refresh proposal; coarse field held fixed; one proposal per chain per sweep",
        "attempts_per_sweep": int(len(coarse)),
        "acceptance_cumulative": final_acceptance,
        "reblocking_max_error": float(max(abs(float(r["reblocking_max_error"])) for r in per_rows)),
        "nonfinite_count": int(sum(int(r["nonfinite_count"]) for r in main_rows)),
        "preflight": preflight,
        "checkpoint_load_report": load_report,
        "output_schema": {
            "main_per_sweep_measurements.csv": MAIN_COLUMNS,
            "Gk_per_sweep_measurements.csv": G_COLUMNS,
            "acceptance_history.csv": ACCEPTANCE_COLUMNS,
        },
        "elapsed_seconds": float(time.time() - started),
    }
    write_json(run / "status.json", summary)

    md = [
        "# Lambda=1.0 L16->L32 zero-shot RQ-spline transfer",
        "",
        f"Source checkpoint: `{args.checkpoint}`",
        "",
        "The 20260717T125835Z checkpoint was used. The similarly named 20260717T125627Z run was an aborted hidden-transfer diagnostic and has no checkpoint to use.",
        "",
        f"Kernel sum: `{eta_sum:.16f}`; eta scale: `{ETA_SCALE:.16f}`. Coefficients include eta and no additional eta multiplier was applied.",
        "",
        f"Native L16 coarse count: {len(coarse_all)}. Native L32 reference count: {len(native_all)}.",
        f"Chains/sweeps: {len(coarse)} / {args.n_sweeps}. Detail-only fixed-coarse refreshes; no coarse updates.",
        "",
        f"Cumulative detail acceptance: `{final_acceptance:.6f}`.",
        f"Max reblocking error: `{summary['reblocking_max_error']:.3e}`.",
        f"Nonfinite count in measured states: `{summary['nonfinite_count']}`.",
        "",
        "Key files:",
        f"- `observables/main_per_sweep_measurements.csv`",
        f"- `observables/Gk_per_sweep_measurements.csv`",
        f"- `observables/acceptance_history.csv`",
        f"- `observables/proposal_diagnostics.csv`",
        f"- `observables/sweep_native_comparison.csv`",
    ]
    (run / "summaries" / "run_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
