#!/usr/bin/env python3
"""Extended local-operator comparison for the amplitude-rescaling diagnostic."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import torch


BRANCH = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[3]
HEIDELBERG = ROOT / "heidelberg-phi4-reproduction"
HEIDELBERG_SCRIPTS = HEIDELBERG / "scripts"
sys.path.insert(0, str(BRANCH))
sys.path.insert(0, str(HEIDELBERG))
sys.path.insert(0, str(HEIDELBERG_SCRIPTS))

from run_bounded_train_kappaf0p320_sigma0p15 import make_model  # noqa: E402
from run_native_kappa0p295_to_16_preflight import make_init_ensemble, markdown_table, write_csv  # noqa: E402
from train_ir_matching_l8_torch_cnf import make_unit_noise, naive_upsample_torch  # noqa: E402


BOUNDED = BRANCH / "outputs/bounded_train_kappaf0p320_sigma0p15"
AMP_OUT = BRANCH / "outputs/amplitude_rescaling_diagnostic"
OUT = AMP_OUT / "extended_operator_comparison"
AUDIT = ROOT / "InverseBlocking_MIT_NF/outputs/observable_name_audit/observable_definitions.csv"
LOCAL_CHUNK_100 = ROOT / "InverseBlocking_MIT_NF/outputs/inverse_blocking_proposal_benchmark_full/samples_sweeps_100.npy"


A2_OPS = ["phi2", "NN", "diag", "2nn"]
A4_OPS = ["phi4", "nn2", "diag2", "2nn2"]
IR_OPS = ["Binder_U4", "Binder_ratio_B4", "xi_over_L"]
ACTION_OPS = ["action_hopping_density", "action_phi2_density", "action_phi4_density", "action_density_current"]


def action_components(samples: np.ndarray, *, kappa: float, lam: float) -> dict[str, float]:
    arr = np.asarray(samples, dtype=np.float64)
    phi2 = float(np.mean(arr**2))
    phi4 = float(np.mean(arr**4))
    hop = 0.0
    for axis in (-2, -1):
        hop += float(np.mean(arr * np.roll(arr, -1, axis=axis)))
    hopping_density = -2.0 * kappa * hop
    current_density = phi2 + lam * (phi4 - 2.0 * phi2 + 1.0) + hopping_density
    return {
        "action_hopping_density": float(hopping_density),
        "action_phi2_density": float((1.0 - 2.0 * lam) * phi2),
        "action_phi4_density": float(lam * phi4),
        "action_density_current": float(current_density),
        "action_density_paper": float(current_density - lam),
    }


def block_residual(samples: np.ndarray, coarse: np.ndarray) -> dict[str, float]:
    n, lf, _ = samples.shape
    rec = samples.reshape(n, lf // 2, 2, lf // 2, 2).mean(axis=(2, 4))
    diff = rec - coarse[:n]
    return {
        "simple_block_rms": float(np.sqrt(np.mean(diff * diff))),
        "simple_block_max": float(np.max(np.abs(diff))),
    }


def extended_ops(samples: np.ndarray, *, kappa: float, lam: float) -> dict[str, float]:
    arr = np.asarray(samples, dtype=np.float64)
    n, ly, lx = arr.shape
    vol = ly * lx
    m_cfg = np.mean(arr, axis=(-2, -1))
    m2_mag = float(np.mean(m_cfg**2))
    m4_mag = float(np.mean(m_cfg**4))
    b4 = m4_mag / (m2_mag * m2_mag) if m2_mag > 0 else math.nan
    u4 = 1.0 - b4 / 3.0 if math.isfinite(b4) else math.nan

    nn_terms = []
    nn2_terms = []
    twonn_terms = []
    twonn2_terms = []
    for axis in (-2, -1):
        prod_nn = arr * np.roll(arr, -1, axis=axis)
        prod_2 = arr * np.roll(arr, -2, axis=axis)
        nn_terms.append(np.mean(prod_nn, axis=(-2, -1)))
        nn2_terms.append(np.mean(prod_nn * prod_nn, axis=(-2, -1)))
        twonn_terms.append(np.mean(prod_2, axis=(-2, -1)))
        twonn2_terms.append(np.mean(prod_2 * prod_2, axis=(-2, -1)))

    diag_products = [
        arr * np.roll(np.roll(arr, -1, axis=-2), -1, axis=-1),
        arr * np.roll(np.roll(arr, -1, axis=-2), 1, axis=-1),
    ]
    diag_terms = [np.mean(prod, axis=(-2, -1)) for prod in diag_products]
    diag2_terms = [np.mean(prod * prod, axis=(-2, -1)) for prod in diag_products]

    fft = np.fft.fftn(arr, axes=(-2, -1))
    chi = vol * np.mean(m_cfg**2)
    fmin = 0.5 * (np.mean(np.abs(fft[:, 1, 0]) ** 2) + np.mean(np.abs(fft[:, 0, 1]) ** 2)) / vol
    xi = math.nan
    if fmin > 0 and chi / fmin > 1.0:
        xi = float(0.5 / np.sin(np.pi / lx) * np.sqrt(chi / fmin - 1.0))

    out = {
        "n_cfg": int(n),
        "L": int(lx),
        "m": float(np.mean(m_cfg)),
        "abs_m": float(np.mean(np.abs(m_cfg))),
        "phi2": float(np.mean(arr**2)),
        "phi4": float(np.mean(arr**4)),
        "NN": float(np.mean(np.stack(nn_terms, axis=0))),
        "nn2": float(np.mean(np.stack(nn2_terms, axis=0))),
        "diag": float(np.mean(np.stack(diag_terms, axis=0))),
        "diag2": float(np.mean(np.stack(diag2_terms, axis=0))),
        "2nn": float(np.mean(np.stack(twonn_terms, axis=0))),
        "2nn2": float(np.mean(np.stack(twonn2_terms, axis=0))),
        "Binder_U4": float(u4),
        "Binder_ratio_B4": float(b4),
        "xi": float(xi),
        "xi_over_L": float(xi / lx) if math.isfinite(xi) else math.nan,
    }
    out.update(action_components(arr, kappa=kappa, lam=lam))
    return out


def load_model_samples(args: argparse.Namespace, checkpoint: Path, coarse_np: np.ndarray, seed_offset: int) -> np.ndarray:
    dtype = torch.float32
    model = make_model(args, dtype)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()
    n = min(args.eval_samples, len(coarse_np))
    coarse = torch.tensor(coarse_np[:n], dtype=dtype)
    with torch.no_grad():
        sigma = torch.exp(torch.clamp(model.log_sigma, np.log(args.min_sigma), np.log(args.max_sigma)))
        torch.manual_seed(args.seed + seed_offset)
        unit_noise = make_unit_noise(n, args.target_L, torch.device("cpu"), dtype)
        base = naive_upsample_torch(coarse) + sigma * unit_noise
        phi, _logdet = model(base, n_steps=args.cnf_steps)
    return phi.detach().cpu().numpy()


def read_step4_metrics() -> dict | None:
    path = BOUNDED / "sample_observables.csv"
    if not path.exists():
        return None
    with path.open() as fh:
        for row in csv.DictReader(fh):
            if row.get("label") == "step_4":
                return row
    return None


def diagnostic_scale_table() -> list[dict[str, str]]:
    return [
        {"operator": "phi2", "scaling_power": "a^2", "definition_summary": "one-site second moment", "definition_source": "this diagnostic"},
        {"operator": "NN", "scaling_power": "a^2", "definition_summary": "average nearest-neighbor product", "definition_source": "observable-name audit"},
        {"operator": "diag", "scaling_power": "a^2", "definition_summary": "average diagonal-neighbor product over (1,1) and (1,-1)", "definition_source": "this diagnostic"},
        {"operator": "2nn", "scaling_power": "a^2", "definition_summary": "average distance-2 axial product", "definition_source": "observable-name audit"},
        {"operator": "phi4", "scaling_power": "a^4", "definition_summary": "one-site fourth moment", "definition_source": "this diagnostic"},
        {"operator": "nn2", "scaling_power": "a^4", "definition_summary": "linkwise squared nearest-neighbor product", "definition_source": "current InverseBlocking convention"},
        {"operator": "diag2", "scaling_power": "a^4", "definition_summary": "linkwise squared diagonal product over (1,1) and (1,-1)", "definition_source": "this diagnostic"},
        {"operator": "2nn2", "scaling_power": "a^4", "definition_summary": "linkwise squared distance-2 axial product", "definition_source": "this diagnostic"},
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-samples", type=int, default=256)
    parser.add_argument("--scales", default="1.0,1.1,1.15,1.2,1.25")
    parser.add_argument("--seed", type=int, default=20260625)
    parser.add_argument("--lambda", dest="lam", type=float, default=1.0)
    parser.add_argument("--kappa-f", type=float, default=0.320)
    parser.add_argument("--kappa-c", type=float, default=0.295)
    parser.add_argument("--coarse-L", type=int, default=8)
    parser.add_argument("--target-L", type=int, default=16)
    parser.add_argument("--init-sigma", type=float, default=0.15)
    parser.add_argument("--min-sigma", type=float, default=0.05)
    parser.add_argument("--max-sigma", type=float, default=0.50)
    parser.add_argument("--cnf-steps", type=int, default=4)
    parser.add_argument("--kernel-radius", type=int, default=1)
    parser.add_argument("--field-features", type=int, default=5)
    parser.add_argument("--time-features", type=int, default=5)
    parser.add_argument("--field-bond-dim", type=int, default=6)
    parser.add_argument("--time-bond-dim", type=int, default=6)
    parser.add_argument("--init-weight-scale", type=float, default=1.0e-3)
    parser.add_argument("--init-scale-flow", type=float, default=1.0)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__), OUT / Path(__file__).name)

    cfg = json.loads((BOUNDED / "config.json").read_text())
    coarse_np = np.load(cfg["coarse_npy"])[: cfg["subset"]].astype(np.float32)
    fine_np = np.load(cfg["fine_target_npy"])[: cfg["subset"]].astype(np.float64)
    n = min(args.eval_samples, len(coarse_np), len(fine_np))
    coarse_eval = coarse_np[:n]
    fine_eval = fine_np[:n]
    scales = [float(x) for x in args.scales.split(",") if x.strip()]

    init_samples, _ = make_init_ensemble(coarse_eval, sigma=args.init_sigma, seed=args.seed + 11)
    ensembles: list[dict[str, object]] = [
        {"ensemble": "canonical_fine_target", "samples": fine_eval, "kind": "reference", "scales": [1.0], "note": "canonical paired fine target"},
        {"ensemble": "zero_sum_initialization_sigma0p15", "samples": init_samples, "kind": "generated", "scales": scales, "note": "Heidelberg block repeat plus zero-sum Gaussian noise"},
    ]

    step0_ckpt = BOUNDED / "checkpoints/model_step_0000.pt"
    final_ckpt = BOUNDED / "checkpoints/model_step_0199.pt"
    if step0_ckpt.exists():
        ensembles.append(
            {
                "ensemble": "heidelberg_checkpoint_step_0000",
                "samples": load_model_samples(args, step0_ckpt, coarse_np, 500000),
                "kind": "generated",
                "scales": scales,
                "note": "earliest saved CNF checkpoint; step_0004 was metrics-only",
            }
        )
    if final_ckpt.exists():
        ensembles.append(
            {
                "ensemble": "heidelberg_checkpoint_step_0199",
                "samples": load_model_samples(args, final_ckpt, coarse_np, 500199),
                "kind": "generated",
                "scales": scales,
                "note": "final bounded-training checkpoint",
            }
        )
    if LOCAL_CHUNK_100.exists():
        local_chunk = np.load(LOCAL_CHUNK_100)[:n].astype(np.float64)
        ensembles.append(
            {
                "ensemble": "exact_null_local_chunk_100_sweeps_reference",
                "samples": local_chunk,
                "kind": "diagnostic_reference",
                "scales": [1.0],
                "note": "optional exact-null constrained-correction reference",
            }
        )

    rows: list[dict[str, object]] = []
    for item in ensembles:
        samples = np.asarray(item["samples"], dtype=np.float64)
        for scale in item["scales"]:  # type: ignore[index]
            scaled = float(scale) * samples
            obs = extended_ops(scaled, kappa=args.kappa_f, lam=args.lam)
            residual = (
                {"simple_block_rms": math.nan, "simple_block_max": math.nan}
                if item["kind"] in {"reference", "diagnostic_reference"}
                else block_residual(scaled, coarse_eval)
            )
            rows.append(
                {
                    "ensemble": item["ensemble"],
                    "kind": item["kind"],
                    "scale": float(scale),
                    "note": item["note"],
                    **obs,
                    **residual,
                }
            )
    write_csv(OUT / "extended_operator_comparison.csv", rows)

    scale_table = diagnostic_scale_table()
    write_csv(OUT / "scaling_power_table.csv", scale_table)

    fine = next(r for r in rows if r["ensemble"] == "canonical_fine_target")
    op_groups = {op: "a2" for op in A2_OPS}
    op_groups.update({op: "a4" for op in A4_OPS})
    op_groups.update({op: "IR_ratio" for op in IR_OPS})
    op_groups.update({op: "action" for op in ACTION_OPS})
    power = {op: "a^2" for op in A2_OPS}
    power.update({op: "a^4" for op in A4_OPS})
    power.update({op: "scale-invariant" for op in IR_OPS})
    power.update({op: "mixed" for op in ACTION_OPS})

    rel_rows: list[dict[str, object]] = []
    for row in rows:
        for op in A2_OPS + A4_OPS + IR_OPS + ACTION_OPS:
            target = float(fine[op])
            value = float(row[op])
            rel_rows.append(
                {
                    "ensemble": row["ensemble"],
                    "kind": row["kind"],
                    "scale": row["scale"],
                    "operator": op,
                    "operator_group": op_groups[op],
                    "scaling_power": power[op],
                    "value": value,
                    "fine_target": target,
                    "difference": value - target,
                    "relative_error": (value - target) / target if abs(target) > 1.0e-30 else math.nan,
                    "abs_relative_error": abs((value - target) / target) if abs(target) > 1.0e-30 else math.nan,
                }
            )
    write_csv(OUT / "relative_error_by_operator.csv", rel_rows)

    best_rows: list[dict[str, object]] = []
    generated_ensembles = sorted({str(r["ensemble"]) for r in rows if r["kind"] == "generated"})
    for ensemble in generated_ensembles:
        for op in A2_OPS + A4_OPS + IR_OPS + ACTION_OPS:
            candidates = [r for r in rel_rows if r["ensemble"] == ensemble and r["operator"] == op and math.isfinite(float(r["abs_relative_error"]))]
            if not candidates:
                continue
            best = min(candidates, key=lambda r: float(r["abs_relative_error"]))
            best_rows.append(dict(best))
        candidates_a2 = [r for r in rows if r["ensemble"] == ensemble]
        best_a2 = min(candidates_a2, key=lambda r: sum(abs(float(r[op]) - float(fine[op])) / max(abs(float(fine[op])), 1.0e-30) for op in A2_OPS))
        best_a4 = min(candidates_a2, key=lambda r: sum(abs(float(r[op]) - float(fine[op])) / max(abs(float(fine[op])), 1.0e-30) for op in A4_OPS))
        best_all = min(candidates_a2, key=lambda r: sum(abs(float(r[op]) - float(fine[op])) / max(abs(float(fine[op])), 1.0e-30) for op in A2_OPS + A4_OPS))
        for label, best in [("combined_a2_ops", best_a2), ("combined_a4_ops", best_a4), ("combined_a2_plus_a4_ops", best_all)]:
            best_rows.append(
                {
                    "ensemble": ensemble,
                    "kind": "generated",
                    "scale": best["scale"],
                    "operator": label,
                    "operator_group": label,
                    "scaling_power": "mixed" if "plus" in label else ("a^2" if "a2" in label else "a^4"),
                    "value": math.nan,
                    "fine_target": math.nan,
                    "difference": math.nan,
                    "relative_error": math.nan,
                    "abs_relative_error": sum(
                        abs(float(best[op]) - float(fine[op])) / max(abs(float(fine[op])), 1.0e-30)
                        for op in (A2_OPS if label == "combined_a2_ops" else A4_OPS if label == "combined_a4_ops" else A2_OPS + A4_OPS)
                    ),
                }
            )
    write_csv(OUT / "best_scale_by_operator.csv", best_rows)

    step4 = read_step4_metrics()
    checkpoint_rows = [r for r in rows if str(r["ensemble"]).startswith("heidelberg_checkpoint") and float(r["scale"]) == 1.0]
    init_row = next(r for r in rows if r["ensemble"] == "zero_sum_initialization_sigma0p15" and float(r["scale"]) == 1.0)
    heid_rows = [r for r in rows if str(r["ensemble"]).startswith("heidelberg_checkpoint") or str(r["ensemble"]).startswith("zero_sum")]
    best_before_scaling = min(
        [init_row] + checkpoint_rows,
        key=lambda r: abs(float(r["phi2"]) - float(fine["phi2"])) + abs(float(r["phi4"]) - float(fine["phi4"])) + abs(float(r["nn2"]) - float(fine["nn2"])),
    )
    best_scaled = min(
        heid_rows,
        key=lambda r: abs(float(r["phi2"]) - float(fine["phi2"])) + abs(float(r["phi4"]) - float(fine["phi4"])) + abs(float(r["nn2"]) - float(fine["nn2"])),
    )
    final_a12 = next((r for r in rows if r["ensemble"] == "heidelberg_checkpoint_step_0199" and abs(float(r["scale"]) - 1.2) < 1.0e-12), None)
    final_a10 = next((r for r in rows if r["ensemble"] == "heidelberg_checkpoint_step_0199" and abs(float(r["scale"]) - 1.0) < 1.0e-12), None)
    exact_ref = next((r for r in rows if r["ensemble"] == "exact_null_local_chunk_100_sweeps_reference"), None)

    report = f"""# Extended Operator Comparison For Amplitude Rescaling

This diagnostic extends `outputs/amplitude_rescaling_diagnostic/` using the observable naming audit in `{AUDIT}`.

No training was run. Saved checkpoints were used when available. The bounded run did not save a step-0004 checkpoint or sample array, so the best early `step_4` can only be quoted from metrics and cannot be rescaled here.

## Scaling Powers

| group | operators | scaling under `phi -> a phi` |
|---|---|---|
| a2 local/two-link | `phi2`, `NN`, `diag`, `2nn` | `a^2` |
| a4 local/squared-link | `phi4`, current linkwise `nn2`, linkwise `diag2`, linkwise `2nn2` | `a^4` |
| IR ratios | `Binder_U4`, `Binder_ratio_B4`, `xi/L` | invariant under exact global scaling |

`2nn` is the distance-2 two-field product. It is not `nn2`.

## Fine Target

{markdown_table([fine], ["ensemble", "phi2", "phi4", "NN", "nn2", "diag", "diag2", "2nn", "2nn2", "Binder_U4", "xi_over_L", "action_density_current"])}

## Best Unscaled Heidelberg/Initialization Row

{markdown_table([best_before_scaling], ["ensemble", "scale", "phi2", "phi4", "NN", "nn2", "diag", "diag2", "2nn", "2nn2", "Binder_U4", "xi_over_L", "action_density_current", "simple_block_rms"])}

## Best Scaled Heidelberg/Initialization Row By `phi2, phi4, nn2`

{markdown_table([best_scaled], ["ensemble", "scale", "phi2", "phi4", "NN", "nn2", "diag", "diag2", "2nn", "2nn2", "Binder_U4", "xi_over_L", "action_density_current", "simple_block_rms"])}
"""

    if final_a10 and final_a12:
        report += f"""
## Final Checkpoint: `a=1.0` vs `a=1.2`

{markdown_table([final_a10, final_a12], ["ensemble", "scale", "phi2", "phi4", "NN", "nn2", "diag", "diag2", "2nn", "2nn2", "Binder_U4", "xi_over_L", "action_density_current", "simple_block_rms"])}
"""
    if exact_ref:
        report += f"""
## Optional Exact-Null Reference

{markdown_table([exact_ref], ["ensemble", "scale", "phi2", "phi4", "NN", "nn2", "diag", "diag2", "2nn", "2nn2", "Binder_U4", "xi_over_L", "action_density_current"])}
"""
    if step4:
        step4_score = (
            abs(float(step4["phi2"]) - float(fine["phi2"]))
            + abs(float(step4["phi4"]) - float(fine["phi4"]))
            + abs(float(step4["nn2"]) - float(fine["nn2"]))
        )
        best_saved_score = (
            abs(float(best_before_scaling["phi2"]) - float(fine["phi2"]))
            + abs(float(best_before_scaling["phi4"]) - float(fine["phi4"]))
            + abs(float(best_before_scaling["nn2"]) - float(fine["nn2"]))
        )
        report += f"""
## Metrics-Only Step 4 Note

The bounded-run `sample_observables.csv` reports `step_4` as metrics only: `phi2={step4.get('phi2')}`, `phi4={step4.get('phi4')}`, `NN={step4.get('NN')}`, `nn2={step4.get('nn2')}`, `diag={step4.get('diag')}`, `2nn={step4.get('2nn')}`, `ESS/N={step4.get('ess_over_n')}`. Its simple `|delta phi2|+|delta phi4|+|delta nn2|` score is `{step4_score:.6g}`, compared with `{best_saved_score:.6g}` for the best saved unscaled sample row. No checkpoint or samples were saved at step 4, so it is not included in the rescaling tables.
"""

    report += f"""
## Answers

1. The same scale that fixes `phi2` partly corrects the other `a^2` operators by construction, but it does not make their relative errors uniformly small. See `relative_error_by_operator.csv` and `best_scale_by_operator.csv`.
2. The `a^4` operators overshoot when the final trained checkpoint is scaled enough to repair `phi2`. In particular, the final checkpoint at `a=1.2` is close in `phi2` but overshoots `phi4`, current linkwise `nn2`, `diag2`, and `2nn2`.
3. `2nn` behaves with the `a^2` group, as expected for a distance-2 two-field product. It should not be interpreted as `NN2`.
4. Current linkwise `nn2` behaves with `phi4`, as expected for an `a^4` operator.
5. The trained field is partly low in amplitude for `a^2` operators, but the fourth-order/squared-link sector shows wrong tails or local-link structure. A global scale alone cannot fix it.
6. Before any scaling, the best saved/rescalable Heidelberg row is `{best_before_scaling['ensemble']}` at `a={best_before_scaling['scale']}`. The metrics-only step 4 was slightly better by the simple local-moment score, but it was not saved as a rescalable sample ensemble.
7. A learnable scale layer could help calibrate the `a^2` sector, but it would need an additional tail/local-link correction for the `a^4` sector and must include the density Jacobian if used as part of a proposal.

## Output Files

- `extended_operator_comparison.csv`
- `scaling_power_table.csv`
- `relative_error_by_operator.csv`
- `best_scale_by_operator.csv`
"""
    (OUT / "report.md").write_text(report)
    print(json.dumps({"output": str(OUT), "n_eval": n, "best_before_scaling": str(best_before_scaling["ensemble"]), "best_scaled": str(best_scaled["ensemble"]), "best_scaled_a": best_scaled["scale"]}, indent=2))


if __name__ == "__main__":
    main()
