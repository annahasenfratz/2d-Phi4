#!/usr/bin/env python3
"""Resume only post-training diagnostics for the local multistage pilot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train_lam1p0_local_multistage_rqspline import (  # noqa: E402
    LocalMultistageFlow,
    check_locality,
    locality_tests,
    local_patch_mh,
    make_dataset,
    metrics_rows,
    action_tail_plot,
    model_physical_sample,
    write_csv,
    write_json,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PKG = PROJECT_ROOT / "perfect_blocking_upsampling"
import sys
sys.path.insert(0, str(PKG / "scripts"))
sys.path.insert(0, str(PKG / "src"))

from train_lam1p0_flow_detail_pilot import load_kernel_matrix, load_phi, split_pairs  # noqa: E402
from train_lam1p0_autoregressive_detail_flow import torch_kernel_fft  # noqa: E402


class TransferData:
    pass


def evaluate_transfer(model, stats, kernel, native: np.ndarray, label: str, run: Path, device: torch.device, seed: int) -> tuple[list[dict], TransferData, dict]:
    pairs = split_pairs(native, kernel)
    c_norm = ((pairs["coarse"] - stats["c"]["mean"]) / stats["c"]["std"]).astype(np.float32)
    fft = torch_kernel_fft(kernel, native.shape[1], device)
    with torch.no_grad():
        phi, _detail, _logs = model_physical_sample(model, torch.from_numpy(c_norm).to(device), stats, fft, torch.Generator(device=device).manual_seed(seed))
    generated = phi.cpu().numpy().astype(np.float32)
    rows = metrics_rows(native, generated, label, "whole") + metrics_rows(native, generated, label, "site")
    write_csv(run / "observables" / f"raw_metrics_{label}.csv", rows)
    action_tail_plot(native, generated, run / "plots" / f"action_tail_{label}", f"{label}: zero-shot local flow")
    data = TransferData(); data.c = c_norm; data.stats = stats; data.phi = native
    # Match the production whole-volume action convention without any global mode.
    phi2_n = np.square(native).mean(axis=(1, 2)); phi4_n = (native**4).mean(axis=(1, 2))
    nn_n = 0.5 * ((native * np.roll(native, -1, axis=1)).mean(axis=(1, 2)) + (native * np.roll(native, -1, axis=2)).mean(axis=(1, 2)))
    phi2_g = np.square(generated).mean(axis=(1, 2)); phi4_g = (generated**4).mean(axis=(1, 2))
    nn_g = 0.5 * ((generated * np.roll(generated, -1, axis=1)).mean(axis=(1, 2)) + (generated * np.roll(generated, -1, axis=2)).mean(axis=(1, 2)))
    action_n = 1.0 - 2.0 * phi2_n + phi4_n - 4.0 * 0.340301 * nn_n
    action_g = 1.0 - 2.0 * phi2_g + phi4_g - 4.0 * 0.340301 * nn_g
    tail = {"label": label, "native_q01": float(np.quantile(action_n, 0.01)), "native_q05": float(np.quantile(action_n, 0.05)), "generated_fraction_le_native_q01": float(np.mean(action_g <= np.quantile(action_n, 0.01))), "generated_fraction_le_native_q05": float(np.mean(action_g <= np.quantile(action_n, 0.05)))}
    return rows, data, tail


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--kernel-path", type=Path, default=Path("perfect_blocking/perfect_blocking_lam1p0/kernels/final/chosen_kernel.json"))
    ap.add_argument("--seed", type=int, default=2026072116)
    args = ap.parse_args()
    run = args.run_dir
    device = torch.device("cpu")
    kernel, _raw = load_kernel_matrix(args.kernel_path)
    ckpt = torch.load(run / "checkpoints" / "checkpoint_best.pt", map_location=device, weights_only=False)
    model = LocalMultistageFlow(hidden=32).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    split = json.loads((run / "dataset_split.json").read_text())
    phi16_all = load_phi(Path("data/configs_phi4_2d/lam1p0_kappac0p340301_L16/configs.npz"))
    phi16 = phi16_all[np.asarray(split["source_indices"], dtype=np.int64)]
    data = make_dataset(phi16, kernel, np.asarray(split["train_local_indices"], dtype=np.int64))
    test_idx = np.asarray(split["test_local_indices"], dtype=np.int64)
    all_rows, _, tail = evaluate_transfer(model, data.stats, kernel, data.phi[test_idx], "L8to16", run, device, args.seed + 16)
    tails = [tail]
    diag8 = local_patch_mh(model, data, test_idx, kernel, device, 16, 10, args.seed + 81)
    write_csv(run / "observables" / "local_patch_diagnostics_L8to16.csv", diag8["rows"])
    write_json(run / "N1000" / "local_patch_summary.json", {k: v for k, v in diag8.items() if k != "rows"})
    diagnostics = {"L8to16": {k: v for k, v in diag8.items() if k != "rows"}}
    for label, path, chains, sweeps in (("L16to32", Path("data/configs_phi4_2d/lam1p0_kappac0p340301_L32/configs.npz"), 16, 10), ("L32to64", Path("data/configs_phi4_2d/lam1p0_kappac0p340301_L64/configs.npz"), 8, 5)):
        if not path.exists():
            continue
        native = load_phi(path)[:100]
        rows, transfer, tail = evaluate_transfer(model, data.stats, kernel, native, label, run, device, args.seed + native.shape[1])
        all_rows.extend(rows)
        tails.append(tail)
        diag = local_patch_mh(model, transfer, np.arange(len(native)), kernel, device, chains, sweeps, args.seed + native.shape[1] + 81)
        write_csv(run / "observables" / f"local_patch_diagnostics_{label}.csv", diag["rows"])
        write_json(run / f"zero_shot_{label}" / "local_patch_summary.json", {k: v for k, v in diag.items() if k != "rows"})
        diagnostics[label] = {k: v for k, v in diag.items() if k != "rows"}
    locality = locality_tests(model, device)
    check_locality(locality)
    write_csv(run / "N1000" / "volume_independence_tests_after_training.csv", locality)
    write_csv(run / "observables" / "all_volume_metrics.csv", all_rows)
    write_csv(run / "observables" / "low_action_tail_occupancy.csv", tails)
    lines = ["# Volume Transfer", "", "The model is unchanged across L8->L16, L16->L32, and L32->L64.", "", "## Local Patch Acceptance", ""]
    for label, item in diagnostics.items():
        lines.append(f"- {label}: `{item['acceptance']:.6g}` (nonfinite `{item['nonfinite_count']}`)")
    (run / "summaries" / "volume_transfer_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    status = {"status": "completed", "post_training_evaluation": "completed", "N2000_started": False, "locality_passed": True, "local_patch_diagnostics": diagnostics}
    write_json(run / "status.json", status)
    print(json.dumps(status, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
