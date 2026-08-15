#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import torch

from pilot_utils import (
    action_components_numpy,
    block_fine_to_coarse,
    ensemble_observables_numpy,
    inverse_upscale_to_condition,
    observables_numpy,
    torch_from_numpy_configs,
    write_csv,
    write_json,
)

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from invblock_mit_nf.actions import Phi4Action, Phi4Params
from invblock_mit_nf.blocking import load_kernel_json
from invblock_mit_nf.conditional_flow import ConditionalPhi4Flow


def _save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _bootstrap_stat(values: np.ndarray, fn, *, n_boot: int = 200, seed: int = 0) -> float:
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    stats = []
    n = len(values)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stats.append(fn(values[idx]))
    return float(np.std(stats, ddof=1))


def _summarize_ensemble(
    cfgs: np.ndarray, *, kappa: float, lam: float, seed: int = 0
) -> tuple[dict[str, float], list[dict[str, float]], dict[str, np.ndarray], dict[str, np.ndarray]]:
    action = Phi4Action(Phi4Params(kappa=kappa, lam=lam))
    cfgs_t = torch_from_numpy_configs(cfgs)
    S = action(cfgs_t).detach().cpu().numpy()
    comps = action_components_numpy(cfgs, kappa=kappa, lam=lam)
    obs = [observables_numpy(cfg) for cfg in cfgs]
    sample = {
        "mean_phi": np.array([o["mean_phi"] for o in obs], dtype=np.float64),
        "abs_mean_phi": np.array([o["abs_mean_phi"] for o in obs], dtype=np.float64),
        "phi2": np.array([o["m2"] for o in obs], dtype=np.float64),
        "phi4": np.array([o["m4"] for o in obs], dtype=np.float64),
        "nn": np.array([o["nn"] for o in obs], dtype=np.float64),
        "diag": np.array([o["diag"] for o in obs], dtype=np.float64),
        "two_link": np.array(
            [
                0.5
                * (
                    float(np.mean(cfg * np.roll(cfg, -2, axis=-2)))
                    + float(np.mean(cfg * np.roll(cfg, -2, axis=-1)))
                )
                for cfg in cfgs
            ],
            dtype=np.float64,
        ),
        "action": np.asarray(S, dtype=np.float64),
        "action_density": np.asarray(S, dtype=np.float64) / (cfgs.shape[-1] * cfgs.shape[-2]),
        "quadratic": np.asarray(comps["quadratic"], dtype=np.float64),
        "quartic": np.asarray(comps["quartic"], dtype=np.float64),
        "hopping": np.asarray(comps["hopping"], dtype=np.float64),
        "xi_over_L": np.array([ensemble_observables_numpy(cfgs[i : i + 1])["xi_over_L"] for i in range(len(cfgs))], dtype=np.float64),
    }
    obs_rows = []
    for idx in range(len(cfgs)):
        row = {k: float(v[idx]) for k, v in sample.items() if np.ndim(v) > 0}
        row["sample"] = idx
        row["action"] = float(sample["action"][idx])
        row["action_density"] = float(sample["action_density"][idx])
        row["quadratic"] = float(sample["quadratic"][idx])
        row["quartic"] = float(sample["quartic"][idx])
        row["hopping"] = float(sample["hopping"][idx])
        row["total_action"] = float(comps["total"][idx])
        row["m2"] = float(sample["phi2"][idx])
        row["m4"] = float(sample["phi4"][idx])
        row["binder"] = float(1.0 - sample["phi4"][idx] / (3.0 * sample["phi2"][idx] ** 2)) if sample["phi2"][idx] > 0 else float("nan")
        row["susceptibility"] = float(cfgs.shape[-1] * cfgs.shape[-2] * (sample["phi2"][idx] - sample["mean_phi"][idx] ** 2))
        obs_rows.append(row)
    summary = ensemble_observables_numpy(cfgs)
    summary.update(
        {
            "action_mean": float(np.mean(S)),
            "action_std": float(np.std(S, ddof=1)),
            "action_density_mean": float(np.mean(S) / (cfgs.shape[-1] * cfgs.shape[-2])),
            "action_density_se": float(_se(sample["action_density"])),
            "mean_phi_se": float(_se(sample["mean_phi"])),
            "abs_mean_phi_se": float(_se(sample["abs_mean_phi"])),
            "phi2_se": float(_se(sample["phi2"])),
            "phi4_se": float(_se(sample["phi4"])),
            "nn_se": float(_se(sample["nn"])),
            "diag_se": float(_se(sample["diag"])),
            "two_link_se": float(_se(sample["two_link"])),
            "xi_over_L_se": float(_se(sample["xi_over_L"][np.isfinite(sample["xi_over_L"])])) if np.isfinite(sample["xi_over_L"]).any() else float("nan"),
            "quadratic_mean": float(np.mean(comps["quadratic"])),
            "quartic_mean": float(np.mean(comps["quartic"])),
            "hopping_mean": float(np.mean(comps["hopping"])),
            "quadratic_se": float(_se(sample["quadratic"])),
            "quartic_se": float(_se(sample["quartic"])),
            "hopping_se": float(_se(sample["hopping"])),
            "susceptibility_se": float(_bootstrap_stat(sample["mean_phi"], lambda m: cfgs.shape[-1] * cfgs.shape[-2] * (np.mean(m**2) - np.mean(m) ** 2), seed=seed)),
            "binder_se": float(_bootstrap_stat(sample["mean_phi"], lambda m: 1.0 - np.mean(m**4) / (3.0 * np.mean(m**2) ** 2) if np.mean(m**2) > 0 else np.nan, seed=seed + 1)),
            "finite": bool(np.isfinite(cfgs).all() and np.isfinite(S).all()),
        }
    )
    return summary, obs_rows, comps, sample


def _se(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if len(x) <= 1:
        return float("nan")
    return float(np.std(x, ddof=1) / np.sqrt(len(x)))


def _write_rows(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _load_latest_checkpoint(path: Path) -> Path | None:
    if path.is_file():
        return path
    if not path.exists():
        return None
    cands = sorted(path.glob("epoch_*.pt"))
    return cands[-1] if cands else None


def _fill_zero(condition: np.ndarray) -> np.ndarray:
    out = np.array(condition, copy=True)
    out[:, 1::2, :] = 0.0
    out[:, :, 1::2] = 0.0
    return out


def _fill_gaussian(condition: np.ndarray, *, sigma: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.array(condition, copy=True)
    mask = np.ones_like(out, dtype=bool)
    mask[:, 0::2, 0::2] = False
    out[mask] = rng.normal(scale=sigma, size=mask.sum())
    return out


def _fill_interpolation(condition: np.ndarray) -> np.ndarray:
    out = np.array(condition, copy=True)
    even = out[:, 0::2, 0::2]
    # body centers: average the four surrounding coarse values
    out[:, 1::2, 1::2] = 0.25 * (
        even
        + np.roll(even, -1, axis=1)
        + np.roll(even, -1, axis=2)
        + np.roll(np.roll(even, -1, axis=1), -1, axis=2)
    )
    # horizontal edges: average left/right coarse values
    out[:, 0::2, 1::2] = 0.5 * (even + np.roll(even, -1, axis=2))
    # vertical edges: average up/down coarse values
    out[:, 1::2, 0::2] = 0.5 * (even + np.roll(even, -1, axis=1))
    return out


def _model_sample_from_checkpoint(checkpoint: Path, condition: torch.Tensor, *, batch_size: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    torch.manual_seed(seed)
    flow = ConditionalPhi4Flow(L=condition.shape[-1], n_layers=2, hidden=8).double()
    payload = torch.load(checkpoint, map_location="cpu")
    flow.load_state_dict(payload["model_state_dict"])
    y, logq = flow.sample(batch_size, condition)
    return y.detach().cpu().numpy(), logq.detach().cpu().numpy()


def _direct_action_baseline(cfgs: np.ndarray, *, kappa: float, lam: float) -> dict[str, np.ndarray]:
    action = Phi4Action(Phi4Params(kappa=kappa, lam=lam))
    cfgs_t = torch_from_numpy_configs(cfgs)
    S = action(cfgs_t).detach().cpu().numpy()
    return {"logp": -S, "S": S}


def _make_observable_table(
    name: str,
    cfgs: np.ndarray,
    *,
    kappa: float,
    lam: float,
    logq: np.ndarray | None = None,
    seed: int = 0,
) -> dict[str, float]:
    summary, _, comps, sample = _summarize_ensemble(cfgs, kappa=kappa, lam=lam, seed=seed)
    result = {
        "ensemble": name,
        "n_samples": int(len(cfgs)),
        "mean_phi": summary["mean_phi"],
        "mean_phi_se": summary["mean_phi_se"],
        "abs_mean_phi": summary["abs_mean_phi"],
        "abs_mean_phi_se": summary["abs_mean_phi_se"],
        "phi2": summary["phi2"],
        "phi2_se": summary["phi2_se"],
        "phi4": summary["phi4"],
        "phi4_se": summary["phi4_se"],
        "nn": summary["nn"],
        "nn_se": summary["nn_se"],
        "diag": summary["diag"],
        "diag_se": summary["diag_se"],
        "two_link": summary["two_link"],
        "two_link_se": summary["two_link_se"],
        "susceptibility": summary["susceptibility"],
        "susceptibility_se": summary["susceptibility_se"],
        "binder": summary["binder"],
        "binder_se": summary["binder_se"],
        "xi_over_L": summary["xi_over_L"],
        "xi_over_L_se": summary["xi_over_L_se"],
        "action_mean": summary["action_mean"],
        "action_mean_se": float(_se(sample["action"])),
        "action_density_mean": summary["action_density_mean"],
        "action_density_se": summary["action_density_se"],
        "quadratic_mean": summary["quadratic_mean"],
        "quadratic_se": summary["quadratic_se"],
        "quartic_mean": summary["quartic_mean"],
        "quartic_se": summary["quartic_se"],
        "hopping_mean": summary["hopping_mean"],
        "hopping_se": summary["hopping_se"],
        "finite": summary["finite"],
    }
    if logq is not None:
        logw = -sample["action"] - np.asarray(logq, dtype=np.float64)
        result.update(
            {
                "mean_logq": float(np.mean(logq)),
                "mean_logq_se": float(_se(np.asarray(logq, dtype=np.float64))),
                "logw_mean": float(np.mean(logw)),
                "logw_mean_se": float(_se(logw)),
                "logw_std": float(np.std(logw, ddof=1)),
                "logw_min": float(np.min(logw)),
                "logw_max": float(np.max(logw)),
                "ess": float(
                    np.exp(
                        2.0 * np.log(np.sum(np.exp(logw - np.max(logw))))
                        - np.log(np.sum(np.exp(2.0 * (logw - np.max(logw)))))
                    )
                ),
            }
        )
        result["ess_over_n"] = result["ess"] / len(cfgs)
    else:
        result.update(
            {
                "mean_logq": float("nan"),
                "mean_logq_se": float("nan"),
                "logw_mean": float("nan"),
                "logw_mean_se": float("nan"),
                "logw_std": float("nan"),
                "logw_min": float("nan"),
                "logw_max": float("nan"),
                "ess": float("nan"),
                "ess_over_n": float("nan"),
            }
        )
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", type=Path, default=Path("outputs/physics_diagnostics_kc030_kf032"))
    p.add_argument("--coarse-n", type=int, default=512)
    p.add_argument("--fine-n", type=int, default=512)
    p.add_argument("--fixed-batch", type=int, default=64)
    p.add_argument("--thermal-sweeps", type=int, default=600)
    p.add_argument("--skip-sweeps", type=int, default=8)
    p.add_argument("--proposal-width", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--pilot-checkpoint", type=Path, default=Path("outputs/tiny_pilot/checkpoints/epoch_0020.pt"))
    args = p.parse_args()

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    kernel = load_kernel_json(str(ROOT / "kernels" / "finite_lambda_kernel_template.json"))
    coarse_dir = outdir / "reference_ensembles" / "coarse"
    fine_dir = outdir / "reference_ensembles" / "fine"
    coarse_dir.mkdir(parents=True, exist_ok=True)
    fine_dir.mkdir(parents=True, exist_ok=True)

    from pilot_utils import generate_coarse_ensemble

    coarse_cfgs, coarse_summary, coarse_history = generate_coarse_ensemble(
        L=8,
        kappa=0.30,
        lam=1.0,
        n_samples=args.coarse_n,
        thermal_sweeps=args.thermal_sweeps,
        skip_sweeps=args.skip_sweeps,
        proposal_width=args.proposal_width,
        seed=args.seed,
    )
    fine_cfgs, fine_summary, fine_history = generate_coarse_ensemble(
        L=16,
        kappa=0.32,
        lam=1.0,
        n_samples=args.fine_n,
        thermal_sweeps=args.thermal_sweeps,
        skip_sweeps=args.skip_sweeps,
        proposal_width=args.proposal_width,
        seed=args.seed + 1,
    )
    np.save(coarse_dir / "configs.npy", coarse_cfgs)
    np.save(fine_dir / "configs.npy", fine_cfgs)
    _write_rows(coarse_dir / "history.csv", coarse_history)
    _write_rows(fine_dir / "history.csv", fine_history)
    write_json(coarse_dir / "summary.json", coarse_summary)
    write_json(fine_dir / "summary.json", fine_summary)
    _save_text(
        coarse_dir / "report.md",
        "\n".join(
            [
                "# Coarse reference ensemble",
                f"- L = 8",
                f"- kappa = 0.30",
                f"- lambda = 1.0",
                f"- n_samples = {args.coarse_n}",
                f"- finite = {coarse_summary['finite']}",
                f"- local_acceptance = {coarse_summary['local_acceptance']:.6f}",
            ]
        )
        + "\n",
    )
    _save_text(
        fine_dir / "report.md",
        "\n".join(
            [
                "# Fine reference ensemble",
                f"- L = 16",
                f"- kappa = 0.32",
                f"- lambda = 1.0",
                f"- n_samples = {args.fine_n}",
                f"- finite = {fine_summary['finite']}",
                f"- local_acceptance = {fine_summary['local_acceptance']:.6f}",
            ]
        )
        + "\n",
    )

    # Baselines from the coarse ensemble.
    condition = inverse_upscale_to_condition(coarse_cfgs, kernel)
    inv_kernel = condition.copy()
    zero_baseline = _fill_zero(condition)
    fine_var = float(np.var(fine_cfgs))
    gaussian_baseline = _fill_gaussian(condition, sigma=np.sqrt(fine_var), seed=args.seed + 11)
    interp_baseline = _fill_interpolation(condition)

    baselines = {
        "inverse_kernel": inv_kernel,
        "zero_missing": zero_baseline,
        "gaussian_missing": gaussian_baseline,
        "interpolation_missing": interp_baseline,
    }

    # Tiny pilot samples, if available.
    pilot_samples = None
    pilot_logq = None
    pilot_ckpt = _load_latest_checkpoint(ROOT / "outputs" / "tiny_pilot" / "checkpoints")
    if pilot_ckpt is None:
        pilot_status = "unavailable"
    else:
        pilot_batch = min(args.fixed_batch, len(coarse_cfgs))
        pilot_cond = torch_from_numpy_configs(condition[:pilot_batch])
        pilot_samples, pilot_logq = _model_sample_from_checkpoint(pilot_ckpt, pilot_cond, batch_size=pilot_batch, seed=args.seed + 23)
        pilot_status = str(pilot_ckpt)

    # Direct-fine comparison set and block->condition test.
    fixed_batch = min(args.fixed_batch, len(fine_cfgs))
    fine_subset = fine_cfgs[:fixed_batch]
    blocked_coarse = block_fine_to_coarse(fine_subset, kernel)
    blocked_condition = inverse_upscale_to_condition(blocked_coarse, kernel)
    direct_conditional_samples, direct_conditional_logq = _model_sample_from_checkpoint(
        pilot_ckpt if pilot_ckpt is not None else (ROOT / "outputs" / "tiny_pilot" / "checkpoints" / "epoch_0020.pt"),
        torch_from_numpy_configs(blocked_condition),
        batch_size=fixed_batch,
        seed=args.seed + 31,
    ) if pilot_ckpt is not None else (None, None)

    # Observable tables.
    ensemble_rows = []
    action_rows = []
    logweight_rows = []
    summaries = {}

    def add_ensemble(name: str, cfgs: np.ndarray, logq: np.ndarray | None = None) -> None:
        summary = _make_observable_table(name, cfgs, kappa=0.32, lam=1.0, logq=logq, seed=args.seed)
        summaries[name] = summary
        ensemble_rows.append(summary)
        action_rows.append(
            {
                "ensemble": name,
                "quadratic_mean": summary["quadratic_mean"],
                "quartic_mean": summary["quartic_mean"],
                "hopping_mean": summary["hopping_mean"],
                "total_mean": summary["action_mean"],
            }
        )
        logweight_rows.append(
            {
                "ensemble": name,
                "mean_logq": summary["mean_logq"],
                "logw_mean": summary["logw_mean"],
                "logw_std": summary["logw_std"],
                "logw_min": summary["logw_min"],
                "logw_max": summary["logw_max"],
                "ess": summary["ess"],
                "ess_over_n": summary["ess_over_n"],
            }
        )

    add_ensemble("direct_fine_reference", fine_cfgs)
    for name, cfgs in baselines.items():
        add_ensemble(name, cfgs)
    if pilot_samples is not None:
        add_ensemble("tiny_pilot", pilot_samples, logq=pilot_logq)

    # Fixed coarse test rows.
    fixed_rows = []
    fixed_logweights = []
    fixed_labels = []
    if pilot_samples is not None:
        true_summary = _make_observable_table("true_fine", fine_subset, kappa=0.32, lam=1.0)
        inv_summary = _make_observable_table("blocked_inverse", blocked_condition, kappa=0.32, lam=1.0)
        model_summary = _make_observable_table("model_sample", direct_conditional_samples, kappa=0.32, lam=1.0, logq=direct_conditional_logq)
        for summ in [true_summary, inv_summary, model_summary]:
            fixed_rows.append(summ)
            fixed_logweights.append(
                {
                    "ensemble": summ["ensemble"],
                    "mean_logq": summ["mean_logq"],
                    "logw_mean": summ["logw_mean"],
                    "logw_std": summ["logw_std"],
                    "ess_over_n": summ["ess_over_n"],
                }
            )
        fixed_base = {
            "zero_missing": _make_observable_table("zero_missing", _fill_zero(blocked_condition), kappa=0.32, lam=1.0),
            "gaussian_missing": _make_observable_table(
                "gaussian_missing", _fill_gaussian(blocked_condition, sigma=np.sqrt(fine_var), seed=args.seed + 41), kappa=0.32, lam=1.0
            ),
            "interpolation_missing": _make_observable_table("interpolation_missing", _fill_interpolation(blocked_condition), kappa=0.32, lam=1.0),
        }
        for summ in fixed_base.values():
            fixed_rows.append(summ)
    else:
        fixed_base = {}

    # Tables and comparisons.
    ref = summaries["direct_fine_reference"]
    ref_map = {
        "mean_phi": ref["mean_phi"],
        "abs_mean_phi": ref["abs_mean_phi"],
        "phi2": ref["phi2"],
        "phi4": ref["phi4"],
        "nn": ref["nn"],
        "diag": ref["diag"],
        "two_link": ref["two_link"],
        "susceptibility": ref["susceptibility"],
        "binder": ref["binder"],
        "xi_over_L": ref["xi_over_L"],
        "action_density_mean": ref["action_density_mean"],
    }
    z_rows = []
    z_metrics = [
        ("mean_phi", "mean_phi_se"),
        ("abs_mean_phi", "abs_mean_phi_se"),
        ("phi2", "phi2_se"),
        ("phi4", "phi4_se"),
        ("nn", "nn_se"),
        ("diag", "diag_se"),
        ("two_link", "two_link_se"),
        ("susceptibility", "susceptibility_se"),
        ("binder", "binder_se"),
        ("xi_over_L", "xi_over_L_se"),
        ("action_density_mean", "action_density_se"),
    ]
    for row in ensemble_rows:
        if row["ensemble"] == "direct_fine_reference":
            continue
        z = {"ensemble": row["ensemble"]}
        for key, se_key in z_metrics:
            diff = row.get(key, float("nan")) - ref_map.get(key, float("nan"))
            se = np.sqrt(row.get(se_key, float("nan")) ** 2 + ref.get(se_key, float("nan")) ** 2)
            z[f"{key}_z"] = float(diff / se) if np.isfinite(diff) and np.isfinite(se) and se > 0 else float("nan")
            z[f"{key}_diff"] = float(diff)
        z_rows.append(z)

    # Save outputs.
    _write_rows(outdir / "observable_table.csv", ensemble_rows)
    _write_rows(outdir / "zscore_table.csv", z_rows)
    _write_rows(outdir / "action_component_table.csv", action_rows)
    _write_rows(outdir / "logweight_table.csv", logweight_rows)
    if fixed_rows:
        fixed_dir = outdir / "fixed_coarse_test"
        fixed_dir.mkdir(parents=True, exist_ok=True)
        _write_rows(fixed_dir / "observable_table.csv", fixed_rows)
        _write_rows(fixed_dir / "logweight_table.csv", fixed_logweights)
        write_json(
            fixed_dir / "summary.json",
            {
                "pilot_checkpoint": pilot_status,
                "fixed_batch": fixed_batch,
                "blocked_coarse_finite": bool(np.isfinite(blocked_coarse).all()),
                "pilot_available": pilot_samples is not None,
                "direct_fine_n": fixed_batch,
            },
        )
        _save_text(
            fixed_dir / "report.md",
            "\n".join(
                [
                    "# Fixed coarse test",
                    f"- pilot_checkpoint = {pilot_status}",
                    f"- fixed_batch = {fixed_batch}",
                    f"- pilot_available = {pilot_samples is not None}",
                    f"- blocked_coarse_finite = {bool(np.isfinite(blocked_coarse).all())}",
                ]
            )
            + "\n",
        )
        np.save(fixed_dir / "direct_fine_subset.npy", fine_subset)
        np.save(fixed_dir / "blocked_coarse.npy", blocked_coarse)
        np.save(fixed_dir / "blocked_condition.npy", blocked_condition)
        if pilot_samples is not None:
            np.save(fixed_dir / "model_samples.npy", direct_conditional_samples)
        np.save(fixed_dir / "zero_baseline.npy", _fill_zero(blocked_condition))
        np.save(fixed_dir / "gaussian_baseline.npy", _fill_gaussian(blocked_condition, sigma=np.sqrt(fine_var), seed=args.seed + 41))
        np.save(fixed_dir / "interp_baseline.npy", _fill_interpolation(blocked_condition))

    # Overall summary.
    summary = {
        "kernel": str(ROOT / "kernels" / "finite_lambda_kernel_template.json"),
        "coarse_reference": coarse_summary,
        "fine_reference": fine_summary,
        "pilot_checkpoint": pilot_status,
        "fine_var_estimate": fine_var,
        "reference_direct_fine_n": len(fine_cfgs),
        "coarse_reference_n": len(coarse_cfgs),
        "fixed_batch": fixed_batch,
        "inverse_kernel_minus_direct_fine_action_density_gap": float(summaries["inverse_kernel"]["action_density_mean"] - ref["action_density_mean"]),
        "inverse_kernel_finite": summaries["inverse_kernel"]["finite"],
        "pilot_finite": summaries.get("tiny_pilot", {}).get("finite", False),
        "pilot_logw_std": summaries.get("tiny_pilot", {}).get("logw_std", float("nan")),
        "pilot_ess_over_n": summaries.get("tiny_pilot", {}).get("ess_over_n", float("nan")),
        "fixed_coarse_available": pilot_samples is not None,
    }
    write_json(outdir / "summary.json", summary)

    report_lines = [
        "# Physics diagnostics",
        "",
        f"- coarse point: kappa=0.30, lambda=1.0, L=8",
        f"- fine point: kappa=0.32, lambda=1.0, L=16",
        f"- inverse-kernel condition built from `{kernel.name}`",
        f"- coarse reference n = {len(coarse_cfgs)}",
        f"- fine reference n = {len(fine_cfgs)}",
        f"- pilot checkpoint = {pilot_status}",
        f"- inverse-kernel baseline action density = {summaries['inverse_kernel']['action_density_mean']:.6f}",
        f"- direct fine action density = {ref['action_density_mean']:.6f}",
        f"- action-density difference (inverse - direct) = {summary['inverse_kernel_minus_direct_fine_action_density_gap']:.6f}",
        f"- tiny pilot logw std = {summary['pilot_logw_std']}",
        f"- tiny pilot ESS/N = {summary['pilot_ess_over_n']}",
    ]
    if fixed_rows:
        report_lines.extend([
            "",
            "## Fixed coarse test",
            f"- fixed batch = {fixed_batch}",
            f"- model samples available = {pilot_samples is not None}",
        ])
    _save_text(outdir / "report.md", "\n".join(report_lines) + "\n")

    # Plots.
    with PdfPages(outdir / "plots.pdf") as pdf:
        metrics = ["action_density_mean", "phi2", "nn", "diag", "susceptibility", "binder"]
        fig, axes = plt.subplots(len(metrics), 1, figsize=(8, 2.2 * len(metrics)), sharex=True)
        x = np.arange(len(ensemble_rows))
        labels = [r["ensemble"] for r in ensemble_rows]
        for ax, metric in zip(axes, metrics):
            vals = [r.get(metric, np.nan) for r in ensemble_rows]
            ax.bar(x, vals)
            ax.axhline(ref[metric], color="k", linestyle="--", linewidth=1)
            ax.set_ylabel(metric)
        axes[-1].set_xticks(x, labels, rotation=30, ha="right")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        if "tiny_pilot" in summaries:
            fig, ax = plt.subplots(figsize=(8, 4))
            for row in logweight_rows:
                if row["ensemble"] == "tiny_pilot":
                    ax.bar(["mean_logq", "logw_std"], [row["mean_logq"], row["logw_std"]])
            ax.set_title("Tiny pilot logweight diagnostics")
            pdf.savefig(fig)
            plt.close(fig)

    _save_text(outdir / "report.md", "\n".join(report_lines) + "\n")
    print("\n".join(report_lines))


if __name__ == "__main__":
    main()
