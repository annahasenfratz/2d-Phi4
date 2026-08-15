#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PKG = PROJECT_ROOT / "perfect_blocking_upsampling"
sys.path.insert(0, str(PKG / "src"))

from perfect_blocking_upsampling.actions import ActionSpec  # noqa: E402
from perfect_blocking_upsampling.kernels import apply_kernel, load_kernel  # noqa: E402
from perfect_blocking_upsampling.observables import observables as ensemble_observables  # noqa: E402

LAM = 0.2
KAPPA = 0.323124
NATIVE_L32 = PKG / "outputs" / "lam0p2_kappa0p323124" / "native" / "L32" / "configs.npz"
NATIVE_L16 = PKG / "outputs" / "lam0p2_kappa0p323124" / "native" / "L16" / "configs.npz"
KERNEL_PATH = PKG / "outputs" / "controlled_patch_lam0p2" / "tail_aware_kernel_search_L16to32" / "rand5x5_0084_kernel.json"
DEFAULT_OUT_ROOT = PKG / "outputs" / "controlled_patch_lam0p2" / "L16to32_fixed_latent_generation_validation_detail_scan"
PREVIOUS_DETAIL1_ROOT = PKG / "outputs" / "controlled_patch_lam0p2" / "L16to32_fixed_latent_generation_validation"
NATIVE_REF = {"action_density": 0.095709, "Binder_U4": 0.612141, "xi_over_L": 0.659112}


def read_phi(path: Path, expected_l: int) -> np.ndarray:
    with np.load(path) as z:
        key = "phi" if "phi" in z.files else z.files[0]
        phi = z[key].astype(np.float32)
    if phi.ndim != 3 or phi.shape[1:] != (expected_l, expected_l):
        raise ValueError(f"expected {path} to contain (N,{expected_l},{expected_l}), got {phi.shape}")
    return phi


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=True, default=str) + "\n", encoding="utf-8")


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
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def prepare_blocked_native_l32_coarse(out_root: Path) -> Path:
    out_root.mkdir(parents=True, exist_ok=True)
    out = out_root / "blocked_native_L32_to_L16_coarse_rand5x5_0084.npz"
    manifest = out_root / "blocked_native_L32_to_L16_coarse_manifest.json"
    if out.exists():
        return out
    phi = read_phi(NATIVE_L32, 32)
    kernel, kernel_json = load_kernel(KERNEL_PATH)
    psi = apply_kernel(phi, kernel).astype(np.float32)
    coarse = psi[:, 0::2, 0::2].astype(np.float32)
    np.savez_compressed(
        out,
        phi=coarse,
        source_native_L32=str(NATIVE_L32),
        kernel_path=str(KERNEL_PATH),
        convention="psi=apply_kernel(phi_L32, rand5x5_0084); coarse=psi[:,0::2,0::2]",
    )
    write_json(
        manifest,
        {
            "file": str(out),
            "shape": list(coarse.shape),
            "source_native_L32": str(NATIVE_L32),
            "kernel_path": str(KERNEL_PATH),
            "convention": "psi=apply_kernel(phi_L32, rand5x5_0084); coarse=psi[:,0::2,0::2]",
            "kernel": kernel_json,
        },
    )
    return out


def parse_history(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def fval(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def mean(xs: list[float]) -> float:
    arr = np.asarray([x for x in xs if np.isfinite(x)], dtype=np.float64)
    return float(np.mean(arr)) if arr.size else float("nan")


def sem(xs: list[float]) -> float:
    arr = np.asarray([x for x in xs if np.isfinite(x)], dtype=np.float64)
    return float(np.std(arr, ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else float("nan")


def summarize_run(run_name: str, run_dir: Path, source: str, detail_passes: int | str, coarse_patch: int | str, detail_patch: int | str) -> list[dict[str, Any]]:
    hist = run_dir / "fixed_latent_coarse_patch_chain_history.csv"
    if not hist.exists():
        return []
    rows = parse_history(hist)
    if not rows:
        return []
    max_sweep = max(int(r["sweep"]) for r in rows)
    windows = [(0, 100), (100, 250), (250, 500), (500, 1000), (1000, 2000)]
    out = []
    for lo, hi in windows:
        if lo > max_sweep:
            continue
        rr = [r for r in rows if lo <= int(r["sweep"]) <= min(hi, max_sweep)]
        if not rr:
            continue
        row: dict[str, Any] = {
            "run": run_name,
            "source": source,
            "coarse_patch": coarse_patch,
            "detail_patch": detail_patch,
            "detail_passes": detail_passes,
            "run_dir": str(run_dir),
            "window": f"{lo}-{min(hi, max_sweep)}",
            "sweep_start": lo,
            "sweep_end": min(hi, max_sweep),
            "n_measurements": len(rr),
        }
        for key in ["action_density", "Binder_U4", "xi_over_L", "magnetization", "coarse_acceptance", "detail_acceptance"]:
            vals = [fval(r.get(key)) for r in rr]
            row[f"{key}_mean"] = mean(vals)
            row[f"{key}_time_sem"] = sem(vals)
            if key in NATIVE_REF:
                row[f"{key}_minus_native"] = row[f"{key}_mean"] - NATIVE_REF[key]
        row["abs_m_mean"] = mean([abs(fval(r.get("magnetization"))) for r in rr])
        row["abs_m_time_sem"] = sem([abs(fval(r.get("magnetization"))) for r in rr])
        row["nonfinite_count"] = int(max(fval(r.get("nonfinite_count_cumulative", 0)) for r in rr))
        row["wall_clock_sec"] = fval(rows[-1].get("runtime_sec"))
        out.append(row)
    return out


def source_stats() -> dict[str, Any]:
    action = ActionSpec("phi4_nn", LAM, KAPPA)
    kernel, _ = load_kernel(KERNEL_PATH)
    direct = read_phi(NATIVE_L16, 16)
    l32 = read_phi(NATIVE_L32, 32)
    blocked = apply_kernel(l32, kernel)[:, 0::2, 0::2].astype(np.float32)

    def stats(x: np.ndarray) -> dict[str, float]:
        obs = ensemble_observables(x.astype(np.float32), action)
        arr = x.astype(np.float64)
        m = arr.mean(axis=(1, 2))
        return {
            "n": int(len(x)),
            "abs_m": float(np.abs(m).mean()),
            "m2": float(np.mean(m * m)),
            "phi2": float(np.mean(arr * arr)),
            "phi4": float(np.mean(arr**4)),
            "NN": float(obs["NN"]),
            "Binder_U4": float(obs["Binder_U4"]),
            "xi_over_L": float(obs["xi_over_L"]),
            "action_density_L16_action": float(obs["action_density"]),
        }

    out = {"direct_native_L16": stats(direct), "blocked_native_L32_rand5x5_ee": stats(blocked)}
    for key in ["abs_m", "m2", "phi2", "phi4", "NN", "Binder_U4", "xi_over_L", "action_density_L16_action"]:
        out[f"delta_{key}_direct_minus_blocked"] = out["direct_native_L16"][key] - out["blocked_native_L32_rand5x5_ee"][key]
    return out


def read_submit_metadata(out_root: Path) -> dict[str, Any]:
    path = out_root / "detail_scan_submit_metadata.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_submit_metadata(out_root: Path, submit_command: str | None) -> None:
    if not submit_command:
        return
    meta = read_submit_metadata(out_root)
    meta["submit_command"] = submit_command
    write_json(out_root / "detail_scan_submit_metadata.json", meta)


def discover_current_runs(out_root: Path) -> dict[str, tuple[Path, str, int | str, int | str, int | str]]:
    found: dict[str, tuple[Path, str, int | str, int | str, int | str]] = {}
    if not out_root.exists():
        return found
    pat = re.compile(r"^(blocked_native_L32|direct_native_L16)(?:_patch(?P<patch>\d+))?(?:_detailpatch(?P<detailpatch>\d+))?_detail(?P<detail>\d+)$")
    for path in sorted(out_root.iterdir()):
        if not path.is_dir():
            continue
        match = pat.match(path.name)
        if not match:
            continue
        source = match.group(1)
        patch = int(match.group("patch")) if match.group("patch") is not None else 4
        detail_patch = int(match.group("detailpatch")) if match.group("detailpatch") is not None else 8
        detail = int(match.group("detail"))
        found[path.name] = (path, source, detail, patch, detail_patch)
    return found


def summarize(out_root: Path, submit_command: str | None = None) -> dict[str, Any]:
    write_submit_metadata(out_root, submit_command)
    submit_meta = read_submit_metadata(out_root)
    rows: list[dict[str, Any]] = []
    previous = {
        "blocked_native_L32_detail1_previous": (PREVIOUS_DETAIL1_ROOT / "blocked_native_L32_coarse_detail1", "blocked_native_L32", 1, 4, 8),
        "direct_native_L16_detail1_previous": (PREVIOUS_DETAIL1_ROOT / "direct_native_L16_coarse_detail1", "direct_native_L16", 1, 4, 8),
    }
    for name, (path, source, detail, coarse_patch, detail_patch) in {**previous, **discover_current_runs(out_root)}.items():
        rows.extend(summarize_run(name, path, source, detail, coarse_patch, detail_patch))
    write_csv(out_root / "detail_scan_summary.csv", rows)
    coarse_stats = source_stats()
    write_json(
        out_root / "detail_scan_summary.json",
        {
            "native_L32_reference": NATIVE_REF,
            "rows": rows,
            "coarse_marginal_check": coarse_stats,
            "submit_metadata": submit_meta,
        },
    )
    lines = [
        "# L16->L32 Fixed-latent Detail-pass Scan",
        "",
        "Exact kernel: plus-logJ fixed-latent coarse patch moves plus symmetric patchwise detail MH.",
        "",
        "Native L32 reference: action `0.095709`, Binder `0.612141`, xi/L `0.659112`.",
        "",
        "## Submission",
        "",
        "Exact submit command:",
        "",
        "```bash",
        submit_meta.get("submit_command", "not recorded"),
        "```",
        "",
        "## Window Summary",
        "",
        "| run | coarse patch | detail patch | detail passes | window | action | Binder | xi/L | |m| | coarse acc | detail acc | nonfinite | wall sec |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['run']} | {row['coarse_patch']} | {row['detail_patch']} | {row['detail_passes']} | {row['window']} | {row['action_density_mean']:.6f} | {row['Binder_U4_mean']:.6f} | {row['xi_over_L_mean']:.6f} | {row['abs_m_mean']:.6f} | {row['coarse_acceptance_mean']:.4f} | {row['detail_acceptance_mean']:.4f} | {row['nonfinite_count']} | {row['wall_clock_sec']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Coarse Marginal Check",
            "",
            "| quantity | direct native L16 | blocked native L32 ee | direct - blocked |",
            "|---|---:|---:|---:|",
        ]
    )
    for key in ["abs_m", "m2", "phi2", "phi4", "NN", "Binder_U4", "xi_over_L", "action_density_L16_action"]:
        lines.append(
            f"| {key} | {coarse_stats['direct_native_L16'][key]:.6f} | {coarse_stats['blocked_native_L32_rand5x5_ee'][key]:.6f} | {coarse_stats[f'delta_{key}_direct_minus_blocked']:.6f} |"
        )
    if not rows:
        lines.append("")
        lines.append("No completed scan runs found yet.")
    (out_root / "detail_scan_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "summary_csv": str(out_root / "detail_scan_summary.csv"),
        "summary_json": str(out_root / "detail_scan_summary.json"),
        "report_md": str(out_root / "detail_scan_report.md"),
        "n_rows": len(rows),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--prepare-blocked-coarse", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--submit-command", default=None)
    args = ap.parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    write_submit_metadata(args.out_root, args.submit_command)
    payload: dict[str, Any] = {}
    if args.prepare_blocked_coarse:
        payload["blocked_coarse"] = str(prepare_blocked_native_l32_coarse(args.out_root))
    if args.summarize:
        payload["summary"] = summarize(args.out_root, args.submit_command)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
