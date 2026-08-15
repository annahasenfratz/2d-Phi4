#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
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

from perfect_blocking_upsampling.actions import action_density  # noqa: E402
from perfect_blocking_upsampling.io import ActionSpec  # noqa: E402
from perfect_blocking_upsampling.kernels import KernelSpec, apply_kernel, inverse_kernel, kernel_stencil_from_spec, normalize_kernel  # noqa: E402
from prototype_lam0p5_patch_detail_parameterization import load_paired, stack_detail  # noqa: E402
from scan_lam0p5_patch_detail_operator_penalty import BASELINE_DELTA_S_STD, train_run  # noqa: E402

LAM = 0.5
KAPPA = 0.3426


def json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(type(obj).__name__)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_kernel_yaml(path: Path) -> KernelSpec:
    data: dict[str, Any] = {"orbits": {}}
    in_orbits = False
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        if raw.startswith("orbits:"):
            in_orbits = True
            continue
        if raw and not raw.startswith(" ") and ":" in raw:
            in_orbits = False
        if in_orbits and ":" in raw:
            k, v = raw.strip().split(":", 1)
            data["orbits"][k] = float(v.strip())
        elif ":" in raw and not raw.startswith(" "):
            k, v = raw.split(":", 1)
            key = k.strip()
            val = v.strip()
            if key in {"name", "type", "normalization"}:
                data[key] = val
            elif key in {"eta", "lambda", "kappa"}:
                data[key] = float(val)
            elif key == "scale_factor":
                data[key] = int(val)
    return KernelSpec(
        name=str(data.get("name", "best_lam0p5_5x5_kernel")),
        type=str(data.get("type", "orbit_kernel")),
        eta=float(data.get("eta", 0.25)),
        scale_factor=int(data.get("scale_factor", 2)),
        normalization=str(data.get("normalization", "sum_to_one")),
        orbits={str(k): float(v) for k, v in data["orbits"].items()},
        stencil=None,
    )


def load_phi(path: Path, expected_l: int) -> tuple[np.ndarray, dict[str, Any]]:
    with np.load(path) as z:
        arr = z["phi" if "phi" in z.files else z.files[0]].astype(np.float32)
        meta = {k: z[k].item() if np.asarray(z[k]).shape == () else z[k].tolist() for k in z.files if k != "phi"}
    if arr.shape[-2:] != (expected_l, expected_l):
        raise ValueError(f"unexpected shape for {path}: {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"nonfinite values in {path}")
    return arr, {"path": str(path), "shape": list(arr.shape), "sha256": sha256(path), "metadata": meta}


def obs_arrays(phi: np.ndarray, action: ActionSpec) -> dict[str, np.ndarray]:
    arr = phi.astype(np.float64)
    m = arr.mean(axis=(-2, -1))
    phi2 = np.mean(arr * arr, axis=(-2, -1))
    phi4 = np.mean(arr**4, axis=(-2, -1))
    nn = 0.5 * (np.mean(arr * np.roll(arr, -1, axis=-2), axis=(-2, -1)) + np.mean(arr * np.roll(arr, -1, axis=-1), axis=(-2, -1)))
    twonn = 0.5 * (np.mean(arr * np.roll(arr, -2, axis=-2), axis=(-2, -1)) + np.mean(arr * np.roll(arr, -2, axis=-1), axis=(-2, -1)))
    diag = np.mean(arr * np.roll(np.roll(arr, -1, axis=-2), -1, axis=-1), axis=(-2, -1))
    ad = action_density(arr, action).mean(axis=(-2, -1))
    return {"abs_m": np.abs(m), "m2": m * m, "m4": m**4, "phi2": phi2, "phi4": phi4, "NN": nn, "2nn": twonn, "diag": diag, "action_density": ad}


def obs_summary(phi: np.ndarray, action: ActionSpec) -> dict[str, float]:
    pc = obs_arrays(phi, action)
    out = {k: float(np.mean(v)) for k, v in pc.items()}
    out["Binder_U4"] = float(1.0 - out["m4"] / max(3.0 * out["m2"] * out["m2"], 1.0e-300))
    out["susceptibility"] = float(phi.shape[-1] * phi.shape[-1] * out["m2"])
    out["xi_over_L"] = float(math.sqrt(max(out["susceptibility"], 0.0) / max(out["phi2"], 1.0e-300)) / phi.shape[-1])
    return out


def construct_paired(args: argparse.Namespace, spec: KernelSpec, coarse_action: ActionSpec) -> dict[str, Any]:
    pair_dir = args.output_root / "paired_data"
    pair_dir.mkdir(parents=True, exist_ok=True)
    fine16, fine_meta = load_phi(args.fine16, 16)
    native8, native_meta = load_phi(args.native8, 8)
    psi = apply_kernel(fine16, spec).astype(np.float32)
    channels = np.stack([psi[:, 0::2, 0::2], psi[:, 1::2, 0::2], psi[:, 0::2, 1::2], psi[:, 1::2, 1::2]], axis=1).astype(np.float32)
    rng = np.random.default_rng(args.seed)
    idx = np.arange(len(fine16), dtype=np.int64)
    rng.shuffle(idx)
    n_train = int(round(args.train_fraction * len(idx)))
    train_idx = idx[:n_train]
    val_idx = idx[n_train:]
    paired = pair_dir / "paired_lam0p5_best5x5_L16_to_L8.npz"
    np.savez_compressed(
        paired,
        fine16=fine16,
        blocked_channels=channels,
        c00=channels[:, 0],
        edge_x=channels[:, 1],
        edge_y=channels[:, 2],
        corner=channels[:, 3],
        native_l8=native8,
        train_idx=train_idx,
        val_idx=val_idx,
        lambda_=np.array(LAM),
        kappa=np.array(KAPPA),
        eta=np.array(spec.eta),
    )
    np.savez_compressed(pair_dir / "split_indices.npz", train_idx=train_idx, val_idx=val_idx, seed=np.array(args.seed))
    phi_rt, inv_info = inverse_kernel(psi[:256], spec)
    psi_rt = apply_kernel(phi_rt, spec)
    roundtrip = {
        "phi_max_abs_error": float(np.max(np.abs(phi_rt - fine16[:256]))),
        "full_psi_reblocking_max_abs_error": float(np.max(np.abs(psi_rt - psi[:256]))),
        "ee_reblocking_max_abs_error": float(np.max(np.abs(psi_rt[:, 0::2, 0::2] - channels[:256, 0]))),
        **inv_info,
    }
    native_obs = obs_summary(native8, coarse_action)
    ee_obs = obs_summary(channels[:, 0], coarse_action)
    all_obs = obs_summary(channels.reshape(-1, 8, 8), coarse_action)
    rows = []
    for obs in ["phi2", "phi4", "NN", "2nn", "diag", "action_density", "Binder_U4", "xi_over_L", "susceptibility", "abs_m"]:
        rows.append({"observable": obs, "native_L8": native_obs[obs], "blocked_ee": ee_obs[obs], "blocked_all_sublattices": all_obs[obs], "delta_ee_minus_native": ee_obs[obs] - native_obs[obs], "delta_all_minus_native": all_obs[obs] - native_obs[obs]})
    write_csv(pair_dir / "blocked_vs_native_observables.csv", rows)
    manifest = {
        "lambda": LAM,
        "kappa": KAPPA,
        "kernel": str(args.kernel),
        "kernel_name": spec.name,
        "kernel_orbits": spec.orbits,
        "fine16": fine_meta,
        "native_l8": native_meta,
        "paired_file": paired,
        "split_seed": args.seed,
        "shapes": {"fine16": list(fine16.shape), "blocked_channels": list(channels.shape), "train": int(len(train_idx)), "validation": int(len(val_idx))},
        "roundtrip": roundtrip,
        "blocking_convention": "psi=apply_kernel(phi_L16,K); c00=psi[:,0::2,0::2]; detail channels=(10,01,11)",
    }
    write_json(pair_dir / "paired_data_manifest.json", manifest)
    with (pair_dir / "sha256_checksums.txt").open("w", encoding="utf-8") as f:
        for name in ["paired_lam0p5_best5x5_L16_to_L8.npz", "split_indices.npz", "paired_data_manifest.json", "blocked_vs_native_observables.csv"]:
            f.write(f"{sha256(pair_dir / name)}  {name}\n")
    lines = [
        "# Lambda=0.5 best-5x5 paired-data report",
        "",
        f"- kernel: `{args.kernel}`",
        f"- paired file: `{paired}`",
        f"- train/validation: `{len(train_idx)}/{len(val_idx)}`",
        f"- roundtrip phi max error: `{roundtrip['phi_max_abs_error']:.6g}`",
        f"- ee reblocking max error: `{roundtrip['ee_reblocking_max_abs_error']:.6g}`",
        "",
        "| observable | native L8 | blocked ee | blocked all | ee-native | all-native |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['observable']} | {row['native_L8']:.8g} | {row['blocked_ee']:.8g} | {row['blocked_all_sublattices']:.8g} | {row['delta_ee_minus_native']:.8g} | {row['delta_all_minus_native']:.8g} |")
    (pair_dir / "PAIRED_DATA_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def write_reports(args: argparse.Namespace, paired: dict[str, Any], result: dict[str, Any]) -> None:
    best = result["best"]
    improves = float(best["deltaS_std"]) < BASELINE_DELTA_S_STD
    clear = float(best["deltaS_std"]) < BASELINE_DELTA_S_STD - 0.25
    write_csv(args.output_root / "patch_detail_variance_training" / "best_by_run.csv", [best])
    summary = {"paired_data": paired, "best": best, "small3_reference_deltaS_std": BASELINE_DELTA_S_STD, "improves_small3_reference": improves, "clear_improvement": clear, "sampler_smoke_launched": False}
    write_json(args.output_root / "summary.json", summary)
    lines = [
        "# Lambda=0.5 5x5 Patch-Detail Training Report",
        "",
        "Bounded diagnostic only. No sampler smoke was launched.",
        "",
        f"- kernel: `{args.kernel}`",
        f"- paired data: `{args.output_root / 'paired_data/paired_lam0p5_best5x5_L16_to_L8.npz'}`",
        f"- loss: `NLL + {args.action_weight} * centered variance DeltaS`",
        f"- best checkpoint: `{best['checkpoint']}`",
        "",
        "## Diagnostics",
        "",
        f"- DeltaS std: `{best['deltaS_std']:.6g}`",
        f"- small3 patch-detail reference DeltaS std: `{BASELINE_DELTA_S_STD:.6g}`",
        f"- clear improvement: `{clear}`",
        f"- DeltaS q01/q50/q99: `{best['deltaS_q01']:.6g}`, `{best['deltaS_q50']:.6g}`, `{best['deltaS_q99']:.6g}`",
        f"- max |DeltaS-mean|: `{best['abs_centered_deltaS_max']:.6g}`",
        f"- reblocking error max: `{best['reblocking_error_max']:.6g}`",
        f"- logq/logdet mean/std: `{best.get('val_nll', float('nan')):.6g}` val NLL; see metrics CSV for per-epoch logq/logdet fields",
        f"- local shifts phi2/phi4/NN/2nn/diag: `{best['delta_phi2_mean']:.6g}` / `{best['delta_phi4_mean']:.6g}` / `{best['delta_NN_mean']:.6g}` / `{best.get('delta_2nn_mean', float('nan')):.6g}` / `{best.get('delta_diag_mean', float('nan')):.6g}`",
        "",
        "## Decision",
        "",
    ]
    if clear:
        lines.append("This checkpoint clearly improves on the small3 reference and is eligible for a separate sampler-smoke request.")
    elif improves:
        lines.append("This checkpoint improves slightly but does not pass the clear-improvement gate. Inspect tails before sampler smoke.")
    else:
        lines.append("This checkpoint does not improve on the small3 reference. Do not run sampler smoke.")
    (args.output_root / "LAM0P5_5X5_PATCH_DETAIL_TRAINING_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.output_root / "STATUS.md").write_text(
        "\n".join(
            [
                "# Lambda=0.5 5x5 Patch-Detail Status",
                "",
                "- paired data built: `yes`",
                "- old small3 paired data used: `no`",
                f"- best DeltaS std: `{best['deltaS_std']:.6g}`",
                f"- small3 reference DeltaS std: `{BASELINE_DELTA_S_STD:.6g}`",
                f"- clear improvement: `{clear}`",
                "- sampler smoke launched: `false`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", type=Path, default=PKG / "outputs" / "lam0p5_5x5_patch_detail_8to16")
    ap.add_argument("--kernel", type=Path, default=PKG / "outputs" / "lam0p5_5x5_kernel_search" / "kernel_candidates" / "best_lam0p5_5x5_kernel.yaml")
    ap.add_argument("--fine16", type=Path, default=PROJECT_ROOT / "phi4_phase-diagram" / "ensembles" / "lam0p5_kappa0p3426_L16_embedded_wolff_sign_cluster_plus_radial_heatbath_N5000" / "configs.npz")
    ap.add_argument("--native8", type=Path, default=PROJECT_ROOT / "phi4_phase-diagram" / "ensembles" / "lam0p5_kappa0p3426_L8_embedded_wolff_sign_cluster_plus_radial_heatbath_N5000" / "configs.npz")
    ap.add_argument("--seed", type=int, default=2026070719)
    ap.add_argument("--train-fraction", type=float, default=0.8)
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--n-coupling", type=int, default=16)
    ap.add_argument("--action-weight", type=float, default=5.0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--report-only", action="store_true", help="Regenerate summary/report from existing paired-data manifest and best_by_run.csv.")
    args = ap.parse_args()
    if args.output_root.exists() and args.overwrite:
        shutil.rmtree(args.output_root)
    for sub in ["paired_data", "scripts", "diagnostics", "patch_detail_variance_training"]:
        (args.output_root / sub).mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__), args.output_root / "scripts" / Path(__file__).name)
    shutil.copy2(args.kernel, args.output_root / "selected_kernel.yaml")
    spec = parse_kernel_yaml(args.kernel)
    fine_action = ActionSpec("phi4_nn", LAM, KAPPA)
    coarse_action = ActionSpec("phi4_nn", LAM, KAPPA)
    if args.report_only:
        manifest_path = args.output_root / "paired_data" / "paired_data_manifest.json"
        best_path = args.output_root / "patch_detail_variance_training" / "best_by_run.csv"
        if not manifest_path.exists() or not best_path.exists():
            raise FileNotFoundError(f"report-only requires {manifest_path} and {best_path}")
        paired = json.loads(manifest_path.read_text(encoding="utf-8"))
        with best_path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            raise ValueError(f"empty {best_path}")
        best: dict[str, Any] = dict(rows[0])
        for key, val in list(best.items()):
            if key == "checkpoint":
                continue
            if val in {"True", "False"}:
                best[key] = val == "True"
                continue
            try:
                best[key] = int(val)
            except ValueError:
                try:
                    best[key] = float(val)
                except ValueError:
                    pass
        write_reports(args, paired, {"best": best})
        print(json.dumps({"best_deltaS_std": best["deltaS_std"], "small3_reference_deltaS_std": BASELINE_DELTA_S_STD, "output_root": str(args.output_root), "report_only": True}, indent=2), flush=True)
        return 0
    paired = construct_paired(args, spec, coarse_action)
    arrays = load_paired(args.output_root / "paired_data" / "paired_lam0p5_best5x5_L16_to_L8.npz")
    c, detail, fine = stack_detail(arrays)
    train_idx = arrays["train_idx"].astype(np.int64)
    val_idx = arrays["val_idx"].astype(np.int64)
    phi_rt, _ = inverse_kernel(apply_kernel(fine[:128], spec), spec)
    write_json(args.output_root / "diagnostics" / "roundtrip_check.json", {"phi_roundtrip_max_abs_error": float(np.max(np.abs(phi_rt - fine[:128]))), "kernel": str(args.kernel)})
    ctx = {"kernel": spec, "fine_action": fine_action, "apply_kernel": lambda phi: apply_kernel(phi, spec)}
    sigmas = {
        "deltaS_total": BASELINE_DELTA_S_STD,
        "deltaS_hopping": 17.503842478135848,
        "deltaS_potential_shifted": 9.830771182533034,
        "delta_phi2": 0.04078191074020347,
        "delta_phi4": 0.11808250197982893,
        "delta_NN": 0.04989374246951119,
        "deltaS_project_quartic": 15.114560253418103,
    }
    result = train_run(
        "variance_dS",
        "control",
        0.0,
        args.output_root / "patch_detail_variance_training",
        c,
        detail,
        fine,
        train_idx,
        val_idx,
        ctx,
        sigmas,
        seed=args.seed,
        epochs=args.epochs,
        hidden=args.hidden,
        n_coupling=args.n_coupling,
        action_weight=args.action_weight,
        batch=args.batch,
    )
    write_reports(args, paired, result)
    print(json.dumps({"best_deltaS_std": result["best"]["deltaS_std"], "small3_reference_deltaS_std": BASELINE_DELTA_S_STD, "output_root": str(args.output_root)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
