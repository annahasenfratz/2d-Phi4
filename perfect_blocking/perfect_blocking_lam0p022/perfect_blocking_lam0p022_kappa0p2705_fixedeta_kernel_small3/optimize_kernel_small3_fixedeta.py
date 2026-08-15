#!/usr/bin/env python3
"""Optimize a fixed-eta 3-orbit real-space kernel for lambda=0.022 criticality."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize


REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "perfect_blocking" / "perfect_blocking_lam0p022_kappa0p2705_fixedeta_kernel_small3"
FINE32 = REPO / "phi4_phase-diagram" / "ensembles" / "lam0p022_kappa0p271_L32_embedded_wolff_sign_cluster_plus_radial_heatbath" / "configs.npz"
DIRECT16 = REPO / "phi4_phase-diagram" / "ensembles" / "lam0p022_kappa0p2705_L16_embedded_wolff_sign_cluster_plus_radial_heatbath_N5000" / "configs.npz"
FINE16 = DIRECT16
DIRECT8 = REPO / "phi4_phase-diagram" / "ensembles" / "lam0p022_kappa0p2705_L8_embedded_wolff_sign_cluster_plus_radial_heatbath_N5000" / "configs.npz"
OLD5 = REPO / "perfect_blocking" / "perfect_blocking_lam0p022_kappa0p2705_fixedeta" / "kernel5x5_summary.json"
LARGE = REPO / "perfect_blocking" / "perfect_blocking_lam0p022_kappa0p2705_fixedeta_kernel_large" / "kernel_large_summary.json"

LAMBDA_VALUE = 0.022
KAPPA = 0.2705
ETA = 0.25
ETA_SCALE = float(2.0 ** (ETA / 2.0))
KEY_SCORE = ["phi2", "phi4", "NN", "nn2", "action_density", "m2", "m4"]
SEED = 20260626


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def jsonable(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    return x


def load_phi(path: Path, expected_l: int) -> tuple[np.ndarray, dict[str, Any]]:
    man = json.loads((path.parent / "manifest.json").read_text())
    if abs(float(man["lambda"]) - LAMBDA_VALUE) > 1e-12 or abs(float(man["kappa"]) - KAPPA) > 1e-12:
        raise RuntimeError(f"wrong parameters: {path}")
    if man.get("production_use") is not True or man.get("local_metropolis_used") is not False:
        raise RuntimeError(f"noncanonical metadata: {path}")
    with np.load(path) as z:
        phi = z["phi"].astype(np.float64)
    if phi.shape[1:] != (expected_l, expected_l):
        raise RuntimeError(f"wrong shape: {path} {phi.shape}")
    return phi, {"path": str(path), "manifest": man, "shape": list(phi.shape)}


def roll(phi: np.ndarray, dy: int, dx: int) -> np.ndarray:
    return np.roll(np.roll(phi, dy, axis=1), dx, axis=2)


def weights_from_x(x: np.ndarray) -> dict[str, float]:
    w10, w11 = float(x[0]), float(x[1])
    return {"w00": 1.0 - 4.0 * w10 - 4.0 * w11, "w10": w10, "w11": w11}


def norm_check(w: dict[str, float]) -> float:
    return w["w00"] + 4*w["w10"] + 4*w["w11"]


def kernel_matrix(w: dict[str, float], radius: int = 1) -> np.ndarray:
    mat = np.array([[w["w11"], w["w10"], w["w11"]], [w["w10"], w["w00"], w["w10"]], [w["w11"], w["w10"], w["w11"]]], dtype=np.float64)
    if radius == 1:
        return mat
    out = np.zeros((2 * radius + 1, 2 * radius + 1), dtype=np.float64)
    c = radius
    out[c-1:c+2, c-1:c+2] = mat
    return out


def shell_sums(phi: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "w00": phi,
        "w10": roll(phi, 1, 0) + roll(phi, -1, 0) + roll(phi, 0, 1) + roll(phi, 0, -1),
        "w11": roll(phi, 1, 1) + roll(phi, 1, -1) + roll(phi, -1, 1) + roll(phi, -1, -1),
    }


def convolve_shells(shells: dict[str, np.ndarray], w: dict[str, float]) -> np.ndarray:
    return w["w00"] * shells["w00"] + w["w10"] * shells["w10"] + w["w11"] * shells["w11"]


def block_all(shells: dict[str, np.ndarray], w: dict[str, float]) -> np.ndarray:
    psi = ETA_SCALE * convolve_shells(shells, w)
    return np.concatenate([psi[:, 0::2, 0::2], psi[:, 1::2, 0::2], psi[:, 0::2, 1::2], psi[:, 1::2, 1::2]], axis=0)


def action_density(phi: np.ndarray) -> np.ndarray:
    onsite = ((1.0 - 2.0 * LAMBDA_VALUE) * phi**2 + LAMBDA_VALUE * phi**4).mean(axis=(1, 2))
    nn = 0.5 * ((phi * np.roll(phi, -1, axis=1)).mean(axis=(1, 2)) + (phi * np.roll(phi, -1, axis=2)).mean(axis=(1, 2)))
    return onsite - 4.0 * KAPPA * nn


def xi_over_l(phi: np.ndarray) -> float:
    n, l, _ = phi.shape
    ft = np.fft.fftn(phi, axes=(1, 2))
    m = phi.mean(axis=(1, 2))
    chi = l * l * np.mean(m * m)
    f = 0.5 * (np.mean(np.abs(ft[:, 1, 0]) ** 2) + np.mean(np.abs(ft[:, 0, 1]) ** 2)) / (l * l)
    if f <= 0 or chi / f <= 1:
        return float("nan")
    return float((1.0 / (2.0 * math.sin(math.pi / l))) * math.sqrt(chi / f - 1.0) / l)


def op_arrays(phi: np.ndarray) -> dict[str, np.ndarray]:
    m = phi.mean(axis=(1, 2))
    nnx = phi * np.roll(phi, -1, axis=1)
    nny = phi * np.roll(phi, -1, axis=2)
    nn = 0.5 * (nnx.mean(axis=(1, 2)) + nny.mean(axis=(1, 2)))
    return {
        "phi2": np.mean(phi**2, axis=(1, 2)),
        "phi4": np.mean(phi**4, axis=(1, 2)),
        "NN": nn,
        "nn2": 0.5 * ((nnx**2).mean(axis=(1, 2)) + (nny**2).mean(axis=(1, 2))),
        "action_density": action_density(phi),
        "m": m,
        "abs_m": np.abs(m),
        "m2": m**2,
        "m4": m**4,
    }


def metrics(phi: np.ndarray) -> dict[str, float]:
    obs = op_arrays(phi)
    m2 = float(np.mean(obs["m2"]))
    m4 = float(np.mean(obs["m4"]))
    l = phi.shape[1]
    out = {k: float(np.mean(obs[k])) for k in ["phi2", "phi4", "NN", "nn2", "action_density", "m", "abs_m", "m2", "m4"]}
    out["Binder_U4"] = float(1.0 - m4 / (3.0 * m2**2))
    out["binder_B4"] = float(m4 / m2**2)
    out["binder_Q"] = float(m2**2 / m4)
    out["susceptibility"] = float(l * l * m2)
    out["xi_over_L"] = xi_over_l(phi)
    return out


def score(blocked: np.ndarray, direct: np.ndarray) -> dict[str, Any]:
    bobs, dobs = op_arrays(blocked), op_arrays(direct)
    b = np.vstack([bobs[k] for k in KEY_SCORE]).T
    d = np.vstack([dobs[k] for k in KEY_SCORE]).T
    bm, dm = b.mean(axis=0), d.mean(axis=0)
    diff = bm - dm
    cov = np.cov(b, rowvar=False) / b.shape[0] + np.cov(d, rowvar=False) / d.shape[0]
    reg = cov + 1e-6 * max(float(np.trace(cov)), 1e-12) / len(KEY_SCORE) * np.eye(len(KEY_SCORE))
    sem = np.sqrt(np.maximum(np.diag(cov), 0))
    z = np.divide(diff, sem, out=np.zeros_like(diff), where=sem > 0)
    scale = np.maximum(np.abs(dm), 1e-12)
    return {
        "D_op": float(diff @ np.linalg.pinv(reg, rcond=1e-12) @ diff),
        "normalized_rms": float(np.sqrt(np.mean((diff / scale) ** 2))),
        "rms_z": float(np.sqrt(np.mean(z**2))),
        "max_abs_z": float(np.max(np.abs(z))),
        "cov_condition": float(np.linalg.cond(reg)),
        "diff": diff,
        "sem": sem,
        "z": z,
        "b_mean": bm,
        "d_mean": dm,
    }


def fft_stats(w: dict[str, float], size: int) -> dict[str, float]:
    mat = np.zeros((size, size), dtype=np.float64)
    km = kernel_matrix(w)
    for iy, dy in enumerate(range(-1, 2)):
        for ix, dx in enumerate(range(-1, 2)):
            mat[dy % size, dx % size] = km[iy, ix]
    kt = np.fft.fft2(mat)
    a = np.abs(kt)
    return {"min_abs_Ktilde": float(a.min()), "max_abs_Ktilde": float(a.max()), "min_abs_Keta_tilde": float(ETA_SCALE*a.min()), "max_abs_Keta_tilde": float(ETA_SCALE*a.max()), "condition_number_abs": float(a.max()/max(a.min(), 1e-300))}


def inverse_roundtrip(phi: np.ndarray, w: dict[str, float]) -> dict[str, float]:
    l = phi.shape[1]
    shells = shell_sums(phi)
    psi = ETA_SCALE * convolve_shells(shells, w)
    mat = np.zeros((l, l), dtype=np.float64)
    km = kernel_matrix(w)
    for iy, dy in enumerate(range(-1, 2)):
        for ix, dx in enumerate(range(-1, 2)):
            mat[dy % l, dx % l] = km[iy, ix]
    kt = ETA_SCALE * np.fft.fft2(mat)
    rec = np.fft.ifft2(np.fft.fft2(psi, axes=(1, 2)) / kt[None], axes=(1, 2))
    return {"max_abs_real_error": float(np.max(np.abs(rec.real - phi))), "rms_real_error": float(np.sqrt(np.mean((rec.real - phi)**2))), "max_abs_imag": float(np.max(np.abs(rec.imag)))}


def evaluate_pair(label: str, fine: np.ndarray, direct: np.ndarray, w: dict[str, float], shells: dict[str, np.ndarray]) -> dict[str, Any]:
    blocked = block_all(shells, w)
    sc = score(blocked, direct)
    rows = []
    for i, op in enumerate(KEY_SCORE):
        rows.append({"match": label, "operator": op, "blocked_mean": float(sc["b_mean"][i]), "direct_mean": float(sc["d_mean"][i]), "delta": float(sc["diff"][i]), "combined_error": float(sc["sem"][i]), "z": float(sc["z"][i])})
    bmet, dmet = metrics(blocked), metrics(direct)
    for op in ["Binder_U4", "susceptibility", "xi_over_L"]:
        rows.append({"match": label, "operator": op, "blocked_mean": bmet[op], "direct_mean": dmet[op], "delta": bmet[op] - dmet[op], "combined_error": "", "z": ""})
    return {"blocked": blocked, "score": {k: v for k, v in sc.items() if k not in {"diff", "sem", "z", "b_mean", "d_mean"}}, "blocked_metrics": bmet, "direct_metrics": dmet, "operator_rows": rows, "fft": fft_stats(w, fine.shape[1]), "roundtrip": inverse_roundtrip(fine[:16], w)}


def old5_weights() -> dict[str, float]:
    ow = json.loads(OLD5.read_text())["weights_shells"]
    return {"w00": ow["w00"], "w10": ow["w10"], "w11": ow["w11"]}


def large_scores() -> dict[str, Any]:
    src = json.loads(LARGE.read_text())
    return {"L32_to_L16": src["primary_L32_to_L16"]["score"], "L16_to_L8": src["transfer_L16_to_L8"]["score"], "condition": src["primary_L32_to_L16"]["fft"]["condition_number_abs"]}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--compare-n", type=int, default=1000)
    p.add_argument("--opt-n", type=int, default=500)
    args = p.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fine32, fine32_meta = load_phi(FINE32, 32)
    direct16, direct16_meta = load_phi(DIRECT16, 16)
    fine16, fine16_meta = load_phi(FINE16, 16)
    direct8, direct8_meta = load_phi(DIRECT8, 8)
    n32 = min(args.compare_n, len(fine32))
    nopt = min(args.opt_n, n32)
    fine_obj_shells = shell_sums(fine32[:nopt])
    direct16_eval = direct16

    def objective(x: np.ndarray) -> float:
        w = weights_from_x(x)
        fs = fft_stats(w, 32)
        penalty = 0.0
        if fs["min_abs_Keta_tilde"] < 0.20:
            penalty += 1e6 * (0.20 - fs["min_abs_Keta_tilde"])**2
        if fs["condition_number_abs"] > 8:
            penalty += 1e4 * (fs["condition_number_abs"] - 8)**2
        blocked = block_all(fine_obj_shells, w)
        return score(blocked, direct16_eval)["D_op"] + penalty

    starts = [("identity", np.array([0.0, 0.0])), ("old5_projected", np.array([old5_weights()["w10"], old5_weights()["w11"]])), ("smooth", np.array([0.08, 0.02])), ("edge_only", np.array([0.10, 0.0])), ("negative_edge", np.array([-0.04, 0.02]))]
    rows, candidates = [], []
    for label, x0 in starts:
        res = minimize(objective, x0, method="Powell", bounds=[(-0.20, 0.24), (-0.16, 0.16)], options={"maxiter": 60, "maxfev": 350, "xtol": 2e-5, "ftol": 2e-6})
        w = weights_from_x(res.x)
        ev = evaluate_pair("L32_to_L16", fine32[:n32], direct16, w, shell_sums(fine32[:n32]))
        rows.append({"start": label, "success": bool(res.success), "fun": float(res.fun), "nfev": int(res.nfev), **w, **ev["score"], **ev["fft"]})
        candidates.append((ev["score"]["D_op"], label, w, ev))
    candidates.sort(key=lambda t: t[0])
    _dop, best_label, best_w, primary = candidates[0]
    transfer = evaluate_pair("L16_to_L8", fine16, direct8, best_w, shell_sums(fine16))
    old_primary = evaluate_pair("old5_projected_L32_to_L16", fine32[:n32], direct16, old5_weights(), shell_sums(fine32[:n32]))
    old_transfer = evaluate_pair("old5_projected_L16_to_L8", fine16, direct8, old5_weights(), shell_sums(fine16))
    lg = large_scores()

    write_csv(OUT_DIR / "optimization_log.csv", rows)
    write_csv(OUT_DIR / "operator_matching_table.csv", primary["operator_rows"])
    write_csv(OUT_DIR / "operator_matching_transfer_L16_to_L8.csv", transfer["operator_rows"])
    np.savetxt(OUT_DIR / "kernel_small3_3x3.csv", kernel_matrix(best_w), delimiter=",")
    np.savetxt(OUT_DIR / "kernel_small3_eta_scaled_3x3.csv", ETA_SCALE * kernel_matrix(best_w), delimiter=",")
    np.savez_compressed(OUT_DIR / "blocked_lam0p022_kappa0p2705_L32_to_L16_kernel_small3_fixedeta.npz", phi=primary["blocked"].astype(np.float32), weights=np.array([best_w["w00"], best_w["w10"], best_w["w11"]], dtype=np.float32), kernel_matrix=kernel_matrix(best_w).astype(np.float32), eta=np.array(ETA, dtype=np.float32), eta_scale=np.array(ETA_SCALE, dtype=np.float32))
    flags = []
    if primary["fft"]["min_abs_Keta_tilde"] < 0.25:
        flags.append("min |K_eta(q)| below 0.25")
    if primary["fft"]["condition_number_abs"] > 2 * old_primary["fft"]["condition_number_abs"]:
        flags.append("condition number more than twice old projected 3-orbit")
    if primary["roundtrip"]["max_abs_real_error"] > 1e-8 or primary["roundtrip"]["max_abs_imag"] > 1e-8:
        flags.append("inverse roundtrip above tolerance")
    summary = {
        "status": "complete",
        "diagnostic_only": True,
        "lambda": LAMBDA_VALUE,
        "kappa": KAPPA,
        "eta": ETA,
        "eta_scale": ETA_SCALE,
        "ansatz": ["w00", "w10", "w11"],
        "normalization": "sum K = 1 before eta scaling",
        "convention": "psi = 2^(eta/2) K phi; all four sublattices used as correlated blocked samples; no average",
        "fine32": fine32_meta,
        "direct16": direct16_meta,
        "fine16": fine16_meta,
        "direct8": direct8_meta,
        "compare_n_L32": int(n32),
        "opt_n_L32": int(nopt),
        "best_start": best_label,
        "weights_shells": best_w,
        "normalization_check": norm_check(best_w),
        "kernel_3x3": kernel_matrix(best_w),
        "eta_scaled_kernel_3x3": ETA_SCALE * kernel_matrix(best_w),
        "primary_L32_to_L16": {k: v for k, v in primary.items() if k != "blocked"},
        "transfer_L16_to_L8": {k: v for k, v in transfer.items() if k != "blocked"},
        "old5_projected_L32_to_L16": {k: v for k, v in old_primary.items() if k != "blocked"},
        "old5_projected_L16_to_L8": {k: v for k, v in old_transfer.items() if k != "blocked"},
        "large_kernel_scores": lg,
        "flags": flags,
    }
    (OUT_DIR / "kernel_small3_summary.json").write_text(json.dumps(jsonable(summary), indent=2) + "\n")
    (OUT_DIR / "kernel_coefficients.json").write_text(json.dumps(jsonable({"lambda": LAMBDA_VALUE, "kappa_cr": KAPPA, "eta": ETA, "eta_fixed": True, "eta_scale": ETA_SCALE, "weights_shells": best_w, "normalization_check": norm_check(best_w), "kernel_3x3": kernel_matrix(best_w), "eta_scaled_kernel_3x3": ETA_SCALE * kernel_matrix(best_w), **primary["fft"]}), indent=2) + "\n")
    report = f"""# Lambda=0.022 Critical Small 3-Orbit Fixed-Eta Kernel

Diagnostic only. The primary optimization used L32 -> L16 with all four
K_eta sublattices as correlated blocked samples.

## Selected Kernel

- best start: `{best_label}`
- eta: `{ETA}`
- sum K: `{norm_check(best_w):.16g}`
- min |K_eta(q)|: `{primary['fft']['min_abs_Keta_tilde']:.8g}`
- max |K_eta(q)|: `{primary['fft']['max_abs_Keta_tilde']:.8g}`
- condition number: `{primary['fft']['condition_number_abs']:.8g}`
- inverse max real roundtrip error: `{primary['roundtrip']['max_abs_real_error']:.3e}`
- inverse max imaginary residue: `{primary['roundtrip']['max_abs_imag']:.3e}`

Weights:
- w00 = `{best_w['w00']:.12g}`
- w10 = `{best_w['w10']:.12g}`
- w11 = `{best_w['w11']:.12g}`

## Matching Scores

| kernel | match | normalized RMS | D_op | rms z | max z | cond(K_eta) |
|---|---|---:|---:|---:|---:|---:|
| small3 | L32->L16 | {primary['score']['normalized_rms']:.6g} | {primary['score']['D_op']:.6g} | {primary['score']['rms_z']:.6g} | {primary['score']['max_abs_z']:.6g} | {primary['fft']['condition_number_abs']:.6g} |
| old 5x5 projected | L32->L16 | {old_primary['score']['normalized_rms']:.6g} | {old_primary['score']['D_op']:.6g} | {old_primary['score']['rms_z']:.6g} | {old_primary['score']['max_abs_z']:.6g} | {old_primary['fft']['condition_number_abs']:.6g} |
| large 8-orbit | L32->L16 | {lg['L32_to_L16']['normalized_rms']:.6g} | {lg['L32_to_L16']['D_op']:.6g} | {lg['L32_to_L16']['rms_z']:.6g} | {lg['L32_to_L16']['max_abs_z']:.6g} | {lg['condition']:.6g} |
| small3 | L16->L8 | {transfer['score']['normalized_rms']:.6g} | {transfer['score']['D_op']:.6g} | {transfer['score']['rms_z']:.6g} | {transfer['score']['max_abs_z']:.6g} | {transfer['fft']['condition_number_abs']:.6g} |

Flags: {flags if flags else 'none'}.
"""
    (OUT_DIR / "report.md").write_text(report)
    print(json.dumps({"status": "complete", "normalized_rms": primary["score"]["normalized_rms"], "condition": primary["fft"]["condition_number_abs"], "flags": flags}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
