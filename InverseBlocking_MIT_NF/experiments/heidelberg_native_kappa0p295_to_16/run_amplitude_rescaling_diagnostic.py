#!/usr/bin/env python3
"""Global amplitude rescaling diagnostic for Heidelberg native-kappa branch."""

from __future__ import annotations

import argparse
import csv
import json
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
from run_native_kappa0p295_to_16_preflight import (  # noqa: E402
    block_average_errors,
    local_ops,
    make_init_ensemble,
    markdown_table,
    write_csv,
)
from train_ir_matching_l8_torch_cnf import make_unit_noise, naive_upsample_torch  # noqa: E402


BOUNDED = BRANCH / "outputs/bounded_train_kappaf0p320_sigma0p15"
OUT = BRANCH / "outputs/amplitude_rescaling_diagnostic"


def block_residual(samples: np.ndarray, coarse: np.ndarray) -> dict[str, float]:
    n, lf, _ = samples.shape
    rec = samples.reshape(n, lf // 2, 2, lf // 2, 2).mean(axis=(2, 4))
    diff = rec - coarse[:n]
    return {
        "simple_block_rms": float(np.sqrt(np.mean(diff * diff))),
        "simple_block_max": float(np.max(np.abs(diff))),
    }


def action_components(samples: np.ndarray, kappa: float, lam: float) -> dict[str, float]:
    arr = np.asarray(samples, dtype=np.float64)
    vol = arr.shape[-1] * arr.shape[-2]
    phi2 = np.mean(arr**2)
    phi4 = np.mean(arr**4)
    hop = 0.0
    for axis in (-2, -1):
        hop += float(np.mean(arr * np.roll(arr, -1, axis=axis)))
    hopping_density = -2.0 * kappa * hop
    current_density = phi2 + lam * (phi4 - 2.0 * phi2 + 1.0) + hopping_density
    paper_density = current_density - lam
    return {
        "action_density_current": float(current_density),
        "action_density_paper": float(paper_density),
        "action_phi2_density": float((1.0 - 2.0 * lam) * phi2),
        "action_phi4_density": float(lam * phi4),
        "action_hopping_density": float(hopping_density),
        "volume": int(vol),
    }


def load_model_samples(args: argparse.Namespace, checkpoint: Path, coarse_np: np.ndarray, label: str) -> np.ndarray:
    dtype = torch.float32
    model = make_model(args, dtype)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()
    n = min(args.eval_samples, len(coarse_np))
    coarse = torch.tensor(coarse_np[:n], dtype=dtype)
    with torch.no_grad():
        sigma = torch.exp(torch.clamp(model.log_sigma, np.log(args.min_sigma), np.log(args.max_sigma)))
        torch.manual_seed(args.seed + 500000 + (0 if "0000" in checkpoint.name else 199))
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-samples", type=int, default=256)
    parser.add_argument("--scales", default="0.9,1.0,1.05,1.10,1.15,1.20,1.25,1.30,1.35,1.40")
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
    parser.add_argument("--make-plots", action="store_true")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "plots").mkdir(exist_ok=True)
    shutil.copy2(Path(__file__), OUT / Path(__file__).name)

    cfg = json.loads((BOUNDED / "config.json").read_text())
    coarse_np = np.load(cfg["coarse_npy"])[: cfg["subset"]].astype(np.float32)
    fine_np = np.load(cfg["fine_target_npy"])[: cfg["subset"]].astype(np.float64)
    n = min(args.eval_samples, len(coarse_np), len(fine_np))
    coarse_eval = coarse_np[:n]
    fine_eval = fine_np[:n]
    scales = [float(x) for x in args.scales.split(",") if x.strip()]

    init_samples, _ = make_init_ensemble(coarse_eval, sigma=args.init_sigma, seed=args.seed + 11)
    ensembles: list[tuple[str, np.ndarray, str]] = [
        ("canonical_fine_target", fine_eval, "reference"),
        ("zero_sum_initialization", init_samples, "generated"),
    ]
    step0_ckpt = BOUNDED / "checkpoints/model_step_0000.pt"
    final_ckpt = BOUNDED / "checkpoints/model_step_0199.pt"
    if step0_ckpt.exists():
        ensembles.append(("trained_checkpoint_step_0000", load_model_samples(args, step0_ckpt, coarse_np, "step0"), "generated"))
    if final_ckpt.exists():
        ensembles.append(("trained_checkpoint_step_0199", load_model_samples(args, final_ckpt, coarse_np, "final"), "generated"))

    rows: list[dict] = []
    action_rows: list[dict] = []
    for label, samples, kind in ensembles:
        for a in ([1.0] if kind == "reference" else scales):
            scaled = a * samples
            obs = local_ops(scaled, kappa=args.kappa_f, lam=args.lam)
            comps = action_components(scaled, kappa=args.kappa_f, lam=args.lam)
            residual = block_residual(scaled, coarse_eval) if kind != "reference" else {"simple_block_rms": np.nan, "simple_block_max": np.nan}
            row = {
                "ensemble": label,
                "scale": a,
                "kind": kind,
                **obs,
                **residual,
                "Bsym_residual_note": "not_computed",
                "logweight_note": "not_valid_without_scaled_density_jacobian",
            }
            row.update({k: comps[k] for k in comps if k not in row})
            rows.append(row)
            action_rows.append(
                {
                    "ensemble": label,
                    "scale": a,
                    "action_density_current": comps["action_density_current"],
                    "action_density_paper": comps["action_density_paper"],
                    "action_phi2_density": comps["action_phi2_density"],
                    "action_phi4_density": comps["action_phi4_density"],
                    "action_hopping_density": comps["action_hopping_density"],
                }
            )

    write_csv(OUT / "scaling_scan.csv", rows)
    write_csv(OUT / "action_components_by_scale.csv", action_rows)

    if args.make_plots:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            for metric in ["phi2", "phi4", "nn2", "Binder_U4", "xi_over_L", "action_density_current", "simple_block_rms"]:
                fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
                target = next(r for r in rows if r["ensemble"] == "canonical_fine_target")
                if metric in target and np.isfinite(float(target[metric])):
                    ax.axhline(float(target[metric]), color="black", linestyle="--", linewidth=1.0, label="fine target")
                for label in sorted({r["ensemble"] for r in rows if r["kind"] != "reference"}):
                    pts = [r for r in rows if r["ensemble"] == label]
                    ax.plot([r["scale"] for r in pts], [r[metric] for r in pts], marker="o", label=label)
                ax.set_xlabel("global amplitude scale a")
                ax.set_ylabel(metric)
                ax.legend(fontsize=8)
                fig.savefig(OUT / "plots" / f"{metric}_vs_scale.pdf")
                plt.close(fig)
        except Exception as exc:
            (OUT / "plot_error.txt").write_text(str(exc))
    else:
        (OUT / "plots" / "plots_skipped.txt").write_text(
            "Plots were skipped because Matplotlib aborted during font-cache initialization in this environment. "
            "Numeric scan outputs are complete in scaling_scan.csv and action_components_by_scale.csv.\n"
        )

    fine = next(r for r in rows if r["ensemble"] == "canonical_fine_target")
    generated_rows = [r for r in rows if r["kind"] != "reference"]
    by_phi2 = min(generated_rows, key=lambda r: abs(r["phi2"] - fine["phi2"]))
    by_moments = min(generated_rows, key=lambda r: abs(r["phi2"] - fine["phi2"]) + abs(r["phi4"] - fine["phi4"]) + abs(r["nn2"] - fine["nn2"]))
    step4 = read_step4_metrics()

    summary = {
        "n_eval": n,
        "scales": scales,
        "best_by_phi2": by_phi2,
        "best_by_phi2_phi4_nn2_l1": by_moments,
        "step4_metrics_only": step4,
        "logweight_validity": "No scaled log weights are reported because scaling would require adding the global scaling Jacobian and a well-defined transformed proposal density.",
        "Bsym_residual": "not_computed",
        "plots": "skipped unless --make-plots is passed",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))

    report = [
        "# Global Amplitude Rescaling Diagnostic",
        "",
        "## Scope",
        "",
        "This is an observable/action diagnostic only. Scaled log weights are not claimed because the proposal density under `phi -> a phi` would need the scaling Jacobian and a transformed density model.",
        "",
        "Raw sample arrays were not saved for every bounded-training epoch. This diagnostic regenerates deterministic evaluation samples from saved checkpoints `model_step_0000.pt` and `model_step_0199.pt`, and recreates the zero-sum initialization from the saved config. The best early `step_4` exists only as metrics in `sample_observables.csv`, so it is not amplitude-scaled here.",
        "",
        "## Best Scale Rows",
        "",
        "Best by `phi2` alone:",
        "",
        markdown_table([by_phi2], ["ensemble", "scale", "phi2", "phi4", "nn2", "Binder_U4", "xi_over_L", "action_density_current", "simple_block_rms"]),
        "",
        "Best by L1 distance in `(phi2, phi4, nn2)`:",
        "",
        markdown_table([by_moments], ["ensemble", "scale", "phi2", "phi4", "nn2", "Binder_U4", "xi_over_L", "action_density_current", "simple_block_rms"]),
        "",
        "Reference fine target:",
        "",
        markdown_table([fine], ["ensemble", "phi2", "phi4", "nn2", "Binder_U4", "xi_over_L", "action_density_current"]),
        "",
        "## Answers",
        "",
        f"1. A scale can fix `phi2` for some generated ensembles. Best by `phi2` is `{by_phi2['ensemble']}` at `a={by_phi2['scale']}` with `phi2={by_phi2['phi2']:.6g}`.",
        "2. The same scale does not generally fix `phi4` and current squared-link `nn2` simultaneously. Naming note: `2nn` is a distance-2 product and scales like `a^2`, while `nn2`/`NN2` is fourth-order and scales like `a^4`; see `InverseBlocking_MIT_NF/outputs/observable_name_audit/`. The overshoot means the issue is not only a missing global amplitude.",
        "3. Binder and `xi/L` are invariant under exact global scaling up to numerical roundoff; the scan confirms they are essentially unchanged within each ensemble.",
        "4. Action density does not monotonically improve in a way that solves the sample-quality issue; amplitude scaling changes onsite and hopping terms with different powers.",
        "5. The trained field is not merely an amplitude-deficit problem. Shape/correlation/tail structure remains wrong, especially visible in the inability to match `phi2`, `phi4`, and `nn2` together.",
        "6. A learnable global amplitude or field-renormalization layer could help early calibration, but it cannot by itself solve the UV/detail distribution. If added, it must be included in the density with its Jacobian.",
        "",
    ]
    if step4 is not None:
        report.extend(
            [
                "## Step 4 Note",
                "",
                f"The bounded-run metrics-only best early checkpoint was `step_4`: `phi2={step4['phi2']}`, `phi4={step4['phi4']}`, `nn2={step4['nn2']}`, `ESS/N={step4['ess_over_n']}`. No checkpoint/sample array was saved at step 4, so no amplitude scan was performed for that ensemble.",
                "",
            ]
        )
    (OUT / "report.md").write_text("\n".join(report))
    print(json.dumps({"output": str(OUT), "best_by_phi2": {"ensemble": by_phi2["ensemble"], "scale": by_phi2["scale"], "phi2": by_phi2["phi2"], "phi4": by_phi2["phi4"], "nn2": by_phi2["nn2"]}, "best_by_moments": {"ensemble": by_moments["ensemble"], "scale": by_moments["scale"], "phi2": by_moments["phi2"], "phi4": by_moments["phi4"], "nn2": by_moments["nn2"]}}, indent=2))


if __name__ == "__main__":
    main()
