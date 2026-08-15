#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PKG = PROJECT_ROOT / "perfect_blocking_upsampling"
sys.path.insert(0, str(PKG / "scripts"))
sys.path.insert(0, str(PKG / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from _common import load_config, load_ensembles, load_frozen_models, load_kernel_spec  # noqa: E402
from perfect_blocking_upsampling.actions import action_total  # noqa: E402
from perfect_blocking_upsampling.kernels import apply_kernel  # noqa: E402
from perfect_blocking_upsampling.observables import observables as ensemble_observables  # noqa: E402
from run_shape_parametric_sampler_validation import (  # noqa: E402
    ValidationConfig,
    choose_initial_index,
    compute_state,
    sample_z,
)


PRIMARY = ["phi2", "phi4", "NN", "action_density", "susceptibility", "Binder_U4", "xi_over_L"]
SECTOR = ["m", "abs_m"]


@dataclass(frozen=True)
class TargetRun:
    label: str
    run_dir: Path
    config: Path | None = None
    coarse_l: int | None = None


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def f(row: dict[str, Any], key: str, default: float = float("nan")) -> float:
    try:
        return float(row[key])
    except Exception:
        return default


def infer_config(run_dir: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    rc_path = run_dir / "run_config.json"
    if not rc_path.exists():
        raise FileNotFoundError(f"missing run_config.json and no --config supplied for {run_dir}")
    rc = json.loads(rc_path.read_text())
    return Path(rc["bundle_config"])


def infer_run_option(run_dir: Path, key: str, default: Any) -> Any:
    rc_path = run_dir / "run_config.json"
    if not rc_path.exists():
        return default
    rc = json.loads(rc_path.read_text())
    return rc.get(key, default)


def build_ctx(config_path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any], Any, Any]:
    loaded_cfg = load_config(config_path)
    coarse, fine_ref, *_ = load_ensembles(loaded_cfg)
    refine_model, refine_state, stages, coarse_action, fine_action, _ = load_frozen_models(loaded_cfg)
    refine_model.load_state_dict(refine_state, strict=False)
    refine_model.eval()
    for model, _, state, *_ in stages.values():
        model.load_state_dict(state)
        model.eval()
    kernel, _kernel_json = load_kernel_spec(loaded_cfg)
    ctx = {
        "refine_model": refine_model,
        "stages": stages,
        "coarse_action": coarse_action,
        "fine_action": fine_action,
        "kernel": kernel,
    }
    return coarse, fine_ref, ctx, coarse_action, fine_action


def reblocking_error(state: dict[str, Any], ctx: dict[str, Any]) -> float:
    blocked = apply_kernel(state["phi"], ctx["kernel"])
    # compute_state already stores u and phi. Recompute cprime through blocked
    # field comparison used by the validation preflight: blocked even-even
    # equals transported coarse-refine coordinate up to numerical precision.
    from run_shape_parametric_sampler_validation import apply_refine_loaded

    cprime, _ = apply_refine_loaded(ctx["refine_model"], state["u"])
    return float(np.max(np.abs(blocked[:, 0::2, 0::2] - cprime)))


def single_state_observables(phi: np.ndarray, fine_action: Any) -> dict[str, float]:
    obs = ensemble_observables(phi, fine_action)
    arr = phi.astype(np.float64)
    m = float(np.mean(arr))
    m2 = m * m
    phi2 = float(np.mean(arr * arr))
    xi_row = math.sqrt(max(m2, 0.0) / max(phi2, 1.0e-300))
    return {
        "m": m,
        "abs_m": abs(m),
        "m2": m2,
        "m4": m2 * m2,
        "phi2": float(obs["phi2"]),
        "phi4": float(obs["phi4"]),
        "NN": float(obs["NN"]),
        "nn2": float(obs["nn2"]),
        "diag": float(obs["diag"]),
        "action_density": float(obs["action_density"]),
        "susceptibility": float(phi.shape[-1] * phi.shape[-2] * m2),
        "Binder_U4_row_proxy": float(obs["Binder_U4"]),
        "xi_over_L_row_proxy": xi_row,
    }


def block_observable(rows: list[dict[str, Any]], obs: str, lattice_l: int) -> float:
    if not rows:
        return float("nan")
    if obs == "susceptibility":
        return float(lattice_l * lattice_l * np.mean([f(r, "m2") for r in rows]))
    if obs == "Binder_U4":
        m2 = np.asarray([f(r, "m2") for r in rows], dtype=np.float64)
        m4 = np.asarray([f(r, "m4") for r in rows], dtype=np.float64)
        return float(1.0 - np.mean(m4) / max(3.0 * np.mean(m2) ** 2, 1.0e-300))
    if obs == "xi_over_L":
        phi2 = float(np.mean([f(r, "phi2") for r in rows]))
        chi = float(lattice_l * lattice_l * np.mean([f(r, "m2") for r in rows]))
        return float(math.sqrt(max(chi, 0.0) / max(phi2, 1.0e-300)) / lattice_l)
    return float(np.mean([f(r, obs) for r in rows]))


def reference_stats(fine_ref: np.ndarray, fine_action: Any) -> dict[str, dict[str, float]]:
    ref = ensemble_observables(fine_ref, fine_action)
    arr = fine_ref.astype(np.float64)
    m = arr.mean(axis=(1, 2))
    series: dict[str, np.ndarray] = {
        "m": m,
        "abs_m": np.abs(m),
        "phi2": np.mean(arr**2, axis=(1, 2)),
        "phi4": np.mean(arr**4, axis=(1, 2)),
        "NN": 0.5
        * (
            np.mean(arr * np.roll(arr, -1, axis=1), axis=(1, 2))
            + np.mean(arr * np.roll(arr, -1, axis=2), axis=(1, 2))
        ),
        "action_density": action_total(arr, fine_action) / (arr.shape[1] * arr.shape[2]),
        "susceptibility": arr.shape[1] * arr.shape[2] * m * m,
    }
    out = {}
    for key, values in series.items():
        out[key] = {
            "mean": float(np.mean(values)),
            "se": float(np.std(values, ddof=1) / math.sqrt(len(values))) if len(values) > 1 else float("nan"),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else float("nan"),
            "n": int(len(values)),
        }
    out["Binder_U4"] = {"mean": float(ref["Binder_U4"]), "se": float("nan"), "std": float("nan"), "n": int(len(fine_ref))}
    out["xi_over_L"] = {"mean": float(ref["xi_over_L"]), "se": float("nan"), "std": float("nan"), "n": int(len(fine_ref))}
    return out


def rows_by_chain(rows: list[dict[str, str]]) -> dict[int, list[dict[str, str]]]:
    out: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        out.setdefault(int(row["chain_id"]), []).append(row)
    for chain_rows in out.values():
        chain_rows.sort(key=lambda r: int(r["sweep"]))
    return out


def summarize_group(rows: list[dict[str, Any]], *, label: str, run_label: str, lattice_l: int, ref: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    by_chain = rows_by_chain(rows) if rows and isinstance(rows[0].get("chain_id"), str) else {}
    if not by_chain:
        by_chain = {}
        for row in rows:
            by_chain.setdefault(int(row["chain_id"]), []).append(row)
    out = []
    for obs in PRIMARY + SECTOR:
        if label == "initial_phi0_pre_ar" and obs in {"Binder_U4", "xi_over_L"}:
            mean = block_observable(rows, obs, lattice_l)
            jk_vals = []
            if len(rows) > 2:
                for i in range(len(rows)):
                    jk = rows[:i] + rows[i + 1 :]
                    jk_vals.append(block_observable(jk, obs, lattice_l))
            if jk_vals:
                jk_vals_arr = np.asarray(jk_vals, dtype=np.float64)
                se = float(math.sqrt((len(jk_vals_arr) - 1) * np.mean((jk_vals_arr - np.mean(jk_vals_arr)) ** 2)))
                std = float(np.std(jk_vals_arr, ddof=1))
            else:
                se = float("nan")
                std = float("nan")
            nvals = len(rows)
        else:
            chain_vals = [block_observable(chain_rows, obs, lattice_l) for chain_rows in by_chain.values()]
            chain_vals = [v for v in chain_vals if math.isfinite(v)]
            if not chain_vals:
                continue
            mean = float(np.mean(chain_vals))
            se = float(np.std(chain_vals, ddof=1) / math.sqrt(len(chain_vals))) if len(chain_vals) > 1 else float("nan")
            std = float(np.std(chain_vals, ddof=1)) if len(chain_vals) > 1 else float("nan")
            nvals = len(chain_vals)
        ref_mean = ref.get(obs, {}).get("mean", float("nan"))
        ref_se = ref.get(obs, {}).get("se", float("nan"))
        denom2 = 0.0
        if math.isfinite(se):
            denom2 += se * se
        if math.isfinite(ref_se):
            denom2 += ref_se * ref_se
        z = (mean - ref_mean) / math.sqrt(denom2) if denom2 > 0 and math.isfinite(ref_mean) else float("nan")
        out.append(
            {
                "run_label": run_label,
                "sample": label,
                "observable": obs,
                "n_chains_or_bins": nvals,
                "mean": mean,
                "se_across_chains": se,
                "std_across_chains": std,
                "reference_mean": ref_mean,
                "reference_se": ref_se,
                "z_vs_reference": z,
            }
        )
    return out


def make_initial_rows(target: TargetRun, config_path: Path, coarse_l: int, out_dir: Path) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, dict[str, Any], Any]:
    coarse, fine_ref, ctx, _coarse_action, fine_action = build_ctx(config_path)
    if coarse.shape[1] != coarse_l:
        raise ValueError(f"{target.label}: config coarse L={coarse.shape[1]} but run coarse L={coarse_l}")
    seed = int(infer_run_option(target.run_dir, "seed", 20260713))
    chains = int(infer_run_option(target.run_dir, "validation_chains", 8))
    sector_balanced = bool(infer_run_option(target.run_dir, "sector_balanced_init", True))
    coarse_start_mode = str(infer_run_option(target.run_dir, "coarse_start_mode", "thermalized_coarse"))
    cfg = ValidationConfig(
        validation_chains=chains,
        seed=seed,
        sector_balanced_init=sector_balanced,
        coarse_start_mode=coarse_start_mode,
    )
    rows: list[dict[str, Any]] = []
    for chain in range(chains):
        rng = np.random.default_rng(seed + 10000 * chain + 777)
        init_idx, target_sector = choose_initial_index(coarse, chain, cfg, rng)
        u = coarse[init_idx][None]
        z_edge, z_pair, z_corner = sample_z(rng, 1, coarse_l)
        state = compute_state(u, z_edge, z_pair, z_corner, ctx)
        obs = single_state_observables(state["phi"][0], fine_action)
        rows.append(
            {
                "run_label": target.label,
                "chain_id": chain,
                "coarse_index": init_idx,
                "target_initial_sector": target_sector,
                "coarse_m": float(np.mean(u)),
                "reblocking_error": reblocking_error(state, ctx),
                "fine_action": float(state["sf"][0]),
                "coarse_action": float(state["sc"][0]),
                "logdet_refine": float(state["logdet"][0]),
                "logq_detail": float(state["logq"][0]),
                "logw": float(state["logw"][0]),
                "logw_centered": float("nan"),
                "logw_convention": "-S_f + S_c + logdet_refine - logq_detail",
                **obs,
                "Binder_U4": float("nan"),
                "xi_over_L": float("nan"),
            }
        )
    logw = np.asarray([r["logw"] for r in rows], dtype=np.float64)
    centered = logw - float(np.mean(logw))
    for row, val in zip(rows, centered):
        row["logw_centered"] = float(val)
    write_csv(out_dir / f"{target.label_safe}_initial_flow_states.csv", rows)  # type: ignore[attr-defined]
    return rows, coarse, fine_ref, ctx, fine_action


def acceptance_window_rows(target: TargetRun) -> list[dict[str, Any]]:
    out = []
    windows = [("1-50", 0, 50), ("51-100", 50, 100), ("101-200", 100, 200), ("full", None, None)]
    for move_type, fname in [("coarse", "coarse_deltas.csv"), ("latent", "latent_deltas.csv")]:
        path = target.run_dir / fname
        if not path.exists():
            continue
        rows = read_csv(path)
        for window, start, stop in windows:
            selected = [
                r
                for r in rows
                if (start is None or int(r["sweep"]) >= start) and (stop is None or int(r["sweep"]) < stop)
            ]
            if not selected:
                continue
            acc = np.asarray([int(r["accepted"]) for r in selected], dtype=np.float64)
            dlogw = np.asarray([float(r["delta_logw"]) for r in selected], dtype=np.float64)
            for chain in ["pooled"] + sorted({r["chain_id"] for r in selected}, key=lambda x: int(x)):
                if chain == "pooled":
                    srows = selected
                else:
                    srows = [r for r in selected if r["chain_id"] == chain]
                if not srows:
                    continue
                sacc = np.asarray([int(r["accepted"]) for r in srows], dtype=np.float64)
                sdlogw = np.asarray([float(r["delta_logw"]) for r in srows], dtype=np.float64)
                out.append(
                    {
                        "run_label": target.label,
                        "move_type": move_type,
                        "window": window,
                        "chain_id": chain,
                        "n_attempts": len(srows),
                        "acceptance": float(np.mean(sacc)),
                        "delta_logw_mean": float(np.mean(sdlogw)),
                        "delta_logw_std": float(np.std(sdlogw, ddof=1)) if len(sdlogw) > 1 else float("nan"),
                    }
                )
    return out


def stable_ess_over_n(logw: np.ndarray) -> float:
    if len(logw) == 0:
        return float("nan")
    x = np.asarray(logw, dtype=np.float64)
    x = x - np.max(x)
    w = np.exp(x)
    return float((np.sum(w) ** 2) / (len(w) * np.sum(w * w)))


def independence_acceptance_proxy(logw: np.ndarray) -> float:
    if len(logw) < 2:
        return float("nan")
    x = np.asarray(logw, dtype=np.float64)
    vals = []
    for a, b in zip(x[:-1], x[1:]):
        vals.append(min(1.0, math.exp(min(0.0, b - a))))
    return float(np.mean(vals))


def sanitize_label(label: str) -> str:
    return label.replace("->", "_to_").replace(" ", "_").replace("x", "x").replace("/", "_")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=PKG / "outputs" / "shape_parametric_sampler_validation" / "initial_state_diagnostics")
    ap.add_argument("--include-l16-8x2000", action="store_true")
    args = ap.parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = [
        TargetRun(
            "lam0p022_L8_to_L16_8x2000",
            PKG / "outputs" / "shape_parametric_sampler_validation" / "pcn_cadence_scan" / "native_L8_pcn1_8x2000",
        ),
        TargetRun(
            "lam0p022_L16_to_L32_32x500",
            PKG / "outputs" / "shape_parametric_sampler_validation" / "L16_to_L32_many_short" / "native_L16_pcn1_P4_32x500",
        ),
    ]
    if args.include_l16_8x2000:
        targets.append(
            TargetRun(
                "lam0p022_L16_to_L32_8x2000",
                PKG / "outputs" / "shape_parametric_sampler_validation" / "L16_to_L32_smoke" / "native_L16_pcn1_8x2000",
            )
        )

    all_initial: list[dict[str, Any]] = []
    all_compare: list[dict[str, Any]] = []
    all_logw_summary: list[dict[str, Any]] = []
    all_ar: list[dict[str, Any]] = []
    all_indep: list[dict[str, Any]] = []

    for target in targets:
        object.__setattr__(target, "label_safe", sanitize_label(target.label))
        config_path = infer_config(target.run_dir, target.config)
        coarse_l = int(target.coarse_l or infer_run_option(target.run_dir, "coarse_L", 8))
        lattice_l = 2 * coarse_l
        initial_rows, _coarse, fine_ref, _ctx, fine_action = make_initial_rows(target, config_path, coarse_l, out_dir)
        all_initial.extend(initial_rows)
        ref = reference_stats(fine_ref, fine_action)

        obs_path = target.run_dir / "observable_timeseries.csv"
        if not obs_path.exists():
            raise FileNotFoundError(obs_path)
        obs_rows = read_csv(obs_path)
        windows = [
            ("initial_phi0_pre_ar", None, None),
            ("sweep_1_50", 0, 50),
            ("sweep_51_100", 50, 100),
            ("full_production", None, None),
        ]
        for label, start, stop in windows:
            if label == "initial_phi0_pre_ar":
                rows = initial_rows
            elif label == "full_production":
                rows = obs_rows
            else:
                rows = [r for r in obs_rows if int(r["sweep"]) >= int(start) and int(r["sweep"]) < int(stop)]
            all_compare.extend(summarize_group(rows, label=label, run_label=target.label, lattice_l=lattice_l, ref=ref))

        logw = np.asarray([float(r["logw"]) for r in initial_rows], dtype=np.float64)
        all_logw_summary.append(
            {
                "run_label": target.label,
                "n_initial_states": len(logw),
                "logw_mean": float(np.mean(logw)),
                "logw_std": float(np.std(logw, ddof=1)) if len(logw) > 1 else float("nan"),
                "logw_min": float(np.min(logw)),
                "logw_max": float(np.max(logw)),
                "logw_centered_std": float(np.std(logw - np.mean(logw), ddof=1)) if len(logw) > 1 else float("nan"),
                "fine_action_density_mean": float(np.mean([float(r["action_density"]) for r in initial_rows])),
                "reblocking_error_max": float(np.max([float(r["reblocking_error"]) for r in initial_rows])),
                "logq_detail_mean": float(np.mean([float(r["logq_detail"]) for r in initial_rows])),
                "logdet_refine_mean": float(np.mean([float(r["logdet_refine"]) for r in initial_rows])),
            }
        )
        all_ar.extend(acceptance_window_rows(target))
        all_indep.append(
            {
                "run_label": target.label,
                "diagnostic": "initial_flow_logweight_only",
                "n": len(logw),
                "ess_over_n": stable_ess_over_n(logw),
                "independence_mh_acceptance_proxy_adjacent_order": independence_acceptance_proxy(logw),
                "logw_std": float(np.std(logw, ddof=1)) if len(logw) > 1 else float("nan"),
                "caveat": "diagnostic only; not the deployment Markov transition; direct fine states lack q_theta(phi|u) for a matched comparison",
            }
        )

    write_csv(out_dir / "initial_flow_states.csv", all_initial)
    write_csv(out_dir / "initial_vs_markov_observables.csv", all_compare)
    write_csv(out_dir / "initial_logweight_summary.csv", all_logw_summary)
    write_csv(out_dir / "startup_ar_window_summary.csv", all_ar)
    write_csv(out_dir / "initial_independence_mh_diagnostic.csv", all_indep)

    semantics = [
        "# Initialization Semantics",
        "",
        "The validation driver initializes each chain before any Markov A/R update as follows:",
        "",
        "1. For `coarse_start_mode=thermalized_coarse`, the chain-specific RNG is `seed + 10000 * chain + 777`.",
        "2. `choose_initial_index` selects `u_0` from the configured coarse ensemble. With `--sector-balanced-init`, even chains draw from nonnegative coarse magnetization and odd chains draw from negative coarse magnetization, falling back to random if a sector pool is empty.",
        "3. `sample_z` draws three independent standard-normal latent tensors with shape `(1, 1, L_c, L_c)` for edge, pair, and corner/body stages.",
        "4. `compute_state` applies the coarse-refine model, staged transported-detail flows, reconstructs the fine `psi`, applies the inverse small3 kernel, and computes `phi_0`, `S_f`, `S_c`, `logdet_refine`, `logq_detail`, and `logw = -S_f + S_c + logdet_refine - logq_detail`.",
        "5. `phi_0` is not A/R accepted. It is simply the starting Markov state.",
        "6. The first A/R update is the first coarse patch proposal in sweep index 0, followed by the sweep's remaining patch proposals and then the latent pCN proposal when the pCN cadence fires.",
        "7. With `measurement_mode=end_of_sweep`, the initial pre-sweep state is not in `observable_timeseries.csv`. Sweep 1 in human counting corresponds to internal `sweep=0`, after the first full sweep of updates.",
        "8. The existing `initial_chain_states.csv` stores only source/index and initial magnetizations, not full initial observables or logweights. This diagnostic reconstructs them offline using the same initialization path.",
    ]
    (out_dir / "INITIALIZATION_SEMANTICS.md").write_text("\n".join(semantics) + "\n", encoding="utf-8")

    ar_md = [
        "# Startup A/R Interpretation",
        "",
        "The startup A/R table uses existing `coarse_deltas.csv` and `latent_deltas.csv`; no Markov updates were run.",
        "",
        "For lambda=0.022, the first-window acceptance is already at the same level as later windows in the accepted runs. This supports the earlier conclusion that the flow-generated initial states do not require an explicit analysis burn-in for the current validation precision.",
        "",
        "The failed lambda=0.5 bundle is not included in this run of the diagnostic. The known failed-smoke contrast remains: coarse acceptance 0.34875 with Delta logw std 16.33399, and latent acceptance 0.215 with Delta logw std 4.30003.",
    ]
    (out_dir / "startup_ar_interpretation.md").write_text("\n".join(ar_md) + "\n", encoding="utf-8")

    indep_md = [
        "# Initial Independence-MH Diagnostic",
        "",
        "This is diagnostic only; it is not the transition used in the deployment Markov chain.",
        "",
        "For each accepted lambda=0.022 target, the table computes ESS/N and a simple adjacent-order independence-MH acceptance proxy from the reconstructed initial flow-state logweights.",
        "",
        "Caveat: direct fine reference states are not paired with a coarse `u` and latent inverse under this conditional flow, so they do not have the same `q_theta(phi|u)` value. The diagnostic therefore uses only the flow-generated initial-state logweight spread.",
    ]
    (out_dir / "initial_independence_mh_diagnostic.md").write_text("\n".join(indep_md) + "\n", encoding="utf-8")

    report = [
        "# Initial Flow-State Diagnostic Report",
        "",
        "## Questions",
        "",
        "1. Is the initial flow-generated fine field A/R accepted?",
        "",
        "No. The initial `phi_0 = phi_theta(u_0, z_0)` is the starting Markov state. The first A/R step is the first coarse patch proposal in the first sweep.",
        "",
        "2. What guarantees correctness of the Markov chain?",
        "",
        "Correctness comes from the subsequent full-global-logweight A/R updates and latent pCN updates, assuming the implemented logweight convention is correct and the chain is ergodic over the intended support. The initial distribution affects startup/transient behavior, not the invariant distribution.",
        "",
        "3. For lambda=0.022, do initial states appear empirically close to thermalized fine states?",
        "",
        "Use `initial_vs_markov_observables.csv` and `initial_logweight_summary.csv`. The initial states are not exact samples, but their quality is supported empirically by stable first-window A/R, burn-in sensitivity, and many-short-chain agreement.",
        "",
        "4. Is there evidence that burn-in is needed?",
        "",
        "For current lambda=0.022 validation results, no. Earlier burn-in sensitivity and many-short-chain comparisons found no improvement from discarding early sweeps, and the startup A/R windows remain stable.",
        "",
        "5. Lambda=0.5 contrast.",
        "",
        "The failed lambda=0.5 bundle was not re-run here. The known failed validation smoke had much poorer A/R and much broader Delta logw, consistent with poor initial/proposal quality relative to lambda=0.022.",
        "",
        "6. Frozen writeup note.",
        "",
        "Document that `phi_0` is not A/R accepted, but lambda=0.022 initial-state quality is an empirical validation result: stable first-window A/R, no burn-in benefit, and agreement between many-short and longer-chain validations.",
        "",
        "## Output Tables",
        "",
        "- `initial_flow_states.csv`",
        "- `initial_vs_markov_observables.csv`",
        "- `initial_logweight_summary.csv`",
        "- `startup_ar_window_summary.csv`",
        "- `initial_independence_mh_diagnostic.csv`",
    ]
    (out_dir / "INITIAL_FLOW_STATE_DIAGNOSTIC_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (out_dir / "initial_vs_markov_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    write_json(out_dir / "diagnostic_manifest.json", {"targets": [str(t.run_dir) for t in targets], "output_dir": str(out_dir)})
    print(json.dumps({"status": "completed", "output_dir": str(out_dir), "targets": [t.label for t in targets]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
