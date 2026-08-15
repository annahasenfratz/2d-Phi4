#!/usr/bin/env python3
"""Optimize a fixed-eta larger real-space kernel for lambda=0.022 criticality.

Convention:
    psi = 2^(eta/2) K phi, eta = 0.25, sum K = 1
    all four sublattices of psi are used as correlated blocked samples
    no four-sublattice average
"""

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
OUT_DIR = REPO / "perfect_blocking" / "perfect_blocking_lam0p022_kappa0p2705_fixedeta_kernel_large"
FINE32 = REPO / "phi4_phase-diagram" / "ensembles" / "lam0p022_kappa0p271_L32_embedded_wolff_sign_cluster_plus_radial_heatbath" / "configs.npz"
DIRECT16 = REPO / "phi4_phase-diagram" / "ensembles" / "lam0p022_kappa0p2705_L16_embedded_wolff_sign_cluster_plus_radial_heatbath_N5000" / "configs.npz"
FINE16 = REPO / "phi4_phase-diagram" / "ensembles" / "lam0p022_kappa0p2705_L16_embedded_wolff_sign_cluster_plus_radial_heatbath_N5000" / "configs.npz"
DIRECT8 = REPO / "phi4_phase-diagram" / "ensembles" / "lam0p022_kappa0p2705_L8_embedded_wolff_sign_cluster_plus_radial_heatbath_N5000" / "configs.npz"
OLD5 = REPO / "perfect_blocking" / "perfect_blocking_lam0p022_kappa0p2705_fixedeta" / "kernel5x5_summary.json"

LAMBDA_VALUE = 0.022
KAPPA = 0.2705
ETA = 0.25
ETA_SCALE = float(2.0 ** (ETA / 2.0))
SEED = 20260626
ORBIT_NAMES = ["w00", "w01", "w11", "w20", "w21", "w22", "w30", "w31"]
OPERATORS = ["phi2", "phi4", "NN", "nn2", "action_density", "abs_m", "m2", "m4"]
KEY_SCORE = ["phi2", "phi4", "NN", "nn2", "action_density", "m2", "m4"]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
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
    manifest = json.loads((path.parent / "manifest.json").read_text())
    if abs(float(manifest["lambda"]) - LAMBDA_VALUE) > 1.0e-12 or abs(float(manifest["kappa"]) - KAPPA) > 1.0e-12:
        raise RuntimeError(f"wrong manifest parameters: {path}")
    if manifest.get("production_use") is not True or manifest.get("local_metropolis_used") is not False:
        raise RuntimeError(f"noncanonical production metadata: {path}")
    with np.load(path) as z:
        phi = z["phi"].astype(np.float64)
    if phi.shape[1:] != (expected_l, expected_l):
        raise RuntimeError(f"wrong shape for {path}: {phi.shape}")
    return phi, {"path": str(path), "manifest": manifest, "shape": list(phi.shape)}


def roll(phi: np.ndarray, dy: int, dx: int) -> np.ndarray:
    return np.roll(np.roll(phi, dy, axis=1), dx, axis=2)


def weights_from_x(x: np.ndarray) -> dict[str, float]:
    vals = {name: float(v) for name, v in zip(ORBIT_NAMES[1:], x)}
    vals["w00"] = 1.0 - (
        4.0 * vals["w01"]
        + 4.0 * vals["w11"]
        + 4.0 * vals["w20"]
        + 8.0 * vals["w21"]
        + 4.0 * vals["w22"]
        + 4.0 * vals["w30"]
        + 8.0 * vals["w31"]
    )
    return {name: vals[name] for name in ORBIT_NAMES}


def norm_check(w: dict[str, float]) -> float:
    return w["w00"] + 4*w["w01"] + 4*w["w11"] + 4*w["w20"] + 8*w["w21"] + 4*w["w22"] + 4*w["w30"] + 8*w["w31"]


def kernel_matrix(w: dict[str, float]) -> np.ndarray:
    mat = np.zeros((7, 7), dtype=np.float64)
    center = 3
    for dy in range(-3, 4):
        for dx in range(-3, 4):
            ady, adx = abs(dy), abs(dx)
            key = None
            if (ady, adx) == (0, 0):
                key = "w00"
            elif sorted((ady, adx)) == [0, 1]:
                key = "w01"
            elif (ady, adx) == (1, 1):
                key = "w11"
            elif sorted((ady, adx)) == [0, 2]:
                key = "w20"
            elif sorted((ady, adx)) == [1, 2]:
                key = "w21"
            elif (ady, adx) == (2, 2):
                key = "w22"
            elif sorted((ady, adx)) == [0, 3]:
                key = "w30"
            elif sorted((ady, adx)) == [1, 3]:
                key = "w31"
            if key is not None:
                mat[center + dy, center + dx] = w[key]
    return mat


def convolve(phi: np.ndarray, w: dict[str, float]) -> np.ndarray:
    out = w["w00"] * phi
    for dy, dx, key in [
        (1, 0, "w01"), (-1, 0, "w01"), (0, 1, "w01"), (0, -1, "w01"),
        (1, 1, "w11"), (1, -1, "w11"), (-1, 1, "w11"), (-1, -1, "w11"),
        (2, 0, "w20"), (-2, 0, "w20"), (0, 2, "w20"), (0, -2, "w20"),
        (2, 1, "w21"), (2, -1, "w21"), (-2, 1, "w21"), (-2, -1, "w21"),
        (1, 2, "w21"), (1, -2, "w21"), (-1, 2, "w21"), (-1, -2, "w21"),
        (2, 2, "w22"), (2, -2, "w22"), (-2, 2, "w22"), (-2, -2, "w22"),
        (3, 0, "w30"), (-3, 0, "w30"), (0, 3, "w30"), (0, -3, "w30"),
        (3, 1, "w31"), (3, -1, "w31"), (-3, 1, "w31"), (-3, -1, "w31"),
        (1, 3, "w31"), (1, -3, "w31"), (-1, 3, "w31"), (-1, -3, "w31"),
    ]:
        out = out + w[key] * roll(phi, dy, dx)
    return out


def shell_sums(phi: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "w00": phi,
        "w01": roll(phi, 1, 0) + roll(phi, -1, 0) + roll(phi, 0, 1) + roll(phi, 0, -1),
        "w11": roll(phi, 1, 1) + roll(phi, 1, -1) + roll(phi, -1, 1) + roll(phi, -1, -1),
        "w20": roll(phi, 2, 0) + roll(phi, -2, 0) + roll(phi, 0, 2) + roll(phi, 0, -2),
        "w21": roll(phi, 2, 1) + roll(phi, 2, -1) + roll(phi, -2, 1) + roll(phi, -2, -1) + roll(phi, 1, 2) + roll(phi, 1, -2) + roll(phi, -1, 2) + roll(phi, -1, -2),
        "w22": roll(phi, 2, 2) + roll(phi, 2, -2) + roll(phi, -2, 2) + roll(phi, -2, -2),
        "w30": roll(phi, 3, 0) + roll(phi, -3, 0) + roll(phi, 0, 3) + roll(phi, 0, -3),
        "w31": roll(phi, 3, 1) + roll(phi, 3, -1) + roll(phi, -3, 1) + roll(phi, -3, -1) + roll(phi, 1, 3) + roll(phi, 1, -3) + roll(phi, -1, 3) + roll(phi, -1, -3),
    }


def convolve_shells(shells: dict[str, np.ndarray], w: dict[str, float]) -> np.ndarray:
    out = np.zeros_like(shells["w00"])
    for key in ORBIT_NAMES:
        out = out + w[key] * shells[key]
    return out


def block_all_sublattices_from_shells(shells: dict[str, np.ndarray], w: dict[str, float]) -> np.ndarray:
    psi = (ETA_SCALE * convolve_shells(shells, w)).astype(np.float64)
    subs = [psi[:, 0::2, 0::2], psi[:, 1::2, 0::2], psi[:, 0::2, 1::2], psi[:, 1::2, 1::2]]
    return np.concatenate(subs, axis=0)


def block_all_sublattices(phi: np.ndarray, w: dict[str, float]) -> np.ndarray:
    return block_all_sublattices_from_shells(shell_sums(phi), w)


def action_density(phi: np.ndarray) -> np.ndarray:
    onsite = ((1.0 - 2.0 * LAMBDA_VALUE) * phi**2 + LAMBDA_VALUE * phi**4).mean(axis=(1, 2))
    nn = 0.5 * ((phi * np.roll(phi, -1, axis=1)).mean(axis=(1, 2)) + (phi * np.roll(phi, -1, axis=2)).mean(axis=(1, 2)))
    return onsite - 4.0 * KAPPA * nn


def xi_over_l(phi: np.ndarray) -> float:
    n, l, _ = phi.shape
    ft = np.fft.fftn(phi, axes=(1, 2))
    m = phi.mean(axis=(1, 2))
    chi = l * l * np.mean(m * m)
    fpx = np.mean(np.abs(ft[:, 1, 0]) ** 2) / (l * l)
    fpy = np.mean(np.abs(ft[:, 0, 1]) ** 2) / (l * l)
    f = 0.5 * (fpx + fpy)
    if f <= 0 or chi / f <= 1:
        return float("nan")
    xi = (1.0 / (2.0 * math.sin(math.pi / l))) * math.sqrt(chi / f - 1.0)
    return float(xi / l)


def op_arrays(phi: np.ndarray) -> dict[str, np.ndarray]:
    m = phi.mean(axis=(1, 2))
    nn = 0.5 * ((phi * np.roll(phi, -1, axis=1)).mean(axis=(1, 2)) + (phi * np.roll(phi, -1, axis=2)).mean(axis=(1, 2)))
    return {
        "phi2": np.mean(phi**2, axis=(1, 2)),
        "phi4": np.mean(phi**4, axis=(1, 2)),
        "NN": nn,
        "nn2": 0.5 * (np.mean((phi * np.roll(phi, -1, axis=1))**2, axis=(1, 2)) + np.mean((phi * np.roll(phi, -1, axis=2))**2, axis=(1, 2))),
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
    out["binder_B4"] = float(m4 / (m2**2))
    out["binder_Q"] = float((m2**2) / m4)
    out["susceptibility"] = float(l * l * m2)
    out["xi_over_L"] = xi_over_l(phi)
    return out


def stack_ops(phi: np.ndarray, names: list[str]) -> np.ndarray:
    obs = op_arrays(phi)
    return np.vstack([obs[k] for k in names]).T


def score(blocked: np.ndarray, direct: np.ndarray, names: list[str]) -> dict[str, Any]:
    b = stack_ops(blocked, names)
    d = stack_ops(direct, names)
    b_mean = b.mean(axis=0)
    d_mean = d.mean(axis=0)
    diff = b_mean - d_mean
    cov = np.cov(b, rowvar=False) / b.shape[0] + np.cov(d, rowvar=False) / d.shape[0]
    reg = cov + 1.0e-6 * max(float(np.trace(cov)), 1.0e-12) / len(names) * np.eye(len(names))
    inv = np.linalg.pinv(reg, rcond=1.0e-12)
    scale = np.maximum(np.abs(d_mean), 1.0e-12)
    normalized = diff / scale
    sem = np.sqrt(np.maximum(np.diag(cov), 0.0))
    z = np.divide(diff, sem, out=np.zeros_like(diff), where=sem > 0)
    return {
        "D_op": float(diff @ inv @ diff),
        "normalized_rms": float(np.sqrt(np.mean(normalized**2))),
        "rms_z": float(np.sqrt(np.mean(z**2))),
        "max_abs_z": float(np.max(np.abs(z))),
        "cov_condition": float(np.linalg.cond(reg)),
        "diff": diff,
        "sem": sem,
        "z": z,
        "b_mean": b_mean,
        "d_mean": d_mean,
    }


def fft_stats(w: dict[str, float], size: int) -> dict[str, float]:
    mat = np.zeros((size, size), dtype=np.float64)
    km = kernel_matrix(w)
    for iy, dy in enumerate(range(-3, 4)):
        for ix, dx in enumerate(range(-3, 4)):
            mat[dy % size, dx % size] = km[iy, ix]
    kt = np.fft.fft2(mat)
    abs_kt = np.abs(kt)
    return {
        "min_abs_Ktilde": float(abs_kt.min()),
        "max_abs_Ktilde": float(abs_kt.max()),
        "min_abs_Keta_tilde": float(ETA_SCALE * abs_kt.min()),
        "max_abs_Keta_tilde": float(ETA_SCALE * abs_kt.max()),
        "condition_number_abs": float(abs_kt.max() / max(abs_kt.min(), 1.0e-300)),
    }


def inverse_roundtrip_error(phi: np.ndarray, w: dict[str, float]) -> dict[str, float]:
    l = phi.shape[1]
    psi = ETA_SCALE * convolve(phi, w)
    mat = np.zeros((l, l), dtype=np.float64)
    km = kernel_matrix(w)
    for iy, dy in enumerate(range(-3, 4)):
        for ix, dx in enumerate(range(-3, 4)):
            mat[dy % l, dx % l] = km[iy, ix]
    kt = ETA_SCALE * np.fft.fft2(mat)
    phi_rec = np.fft.ifft2(np.fft.fft2(psi, axes=(1, 2)) / kt[None, :, :], axes=(1, 2))
    return {
        "max_abs_real_error": float(np.max(np.abs(phi_rec.real - phi))),
        "rms_real_error": float(np.sqrt(np.mean((phi_rec.real - phi) ** 2))),
        "max_abs_imag": float(np.max(np.abs(phi_rec.imag))),
    }


def objective_factory(fine_shells: dict[str, np.ndarray], direct: np.ndarray, names: list[str], size: int):
    cache: dict[tuple[float, ...], float] = {}
    def objective(x: np.ndarray) -> float:
        key = tuple(round(float(v), 10) for v in x)
        if key in cache:
            return cache[key]
        w = weights_from_x(x)
        if not np.all(np.isfinite(list(w.values()))) or max(abs(v) for v in w.values()) > 2.0:
            cache[key] = 1.0e12
            return cache[key]
        fs = fft_stats(w, size)
        penalty = 0.0
        if fs["min_abs_Keta_tilde"] < 0.20:
            penalty += 1.0e6 * (0.20 - fs["min_abs_Keta_tilde"]) ** 2
        if fs["condition_number_abs"] > 8.0:
            penalty += 1.0e4 * (fs["condition_number_abs"] - 8.0) ** 2
        blocked = block_all_sublattices_from_shells(fine_shells, w)
        if not np.all(np.isfinite(blocked)) or np.max(np.abs(blocked)) > 100.0:
            cache[key] = 1.0e12
            return cache[key]
        s = score(blocked, direct, names)
        cache[key] = float(s["D_op"] + penalty)
        return cache[key]
    return objective


def evaluate_pair(label: str, fine: np.ndarray, direct: np.ndarray, w: dict[str, float], size: int, fine_shells: dict[str, np.ndarray] | None = None) -> dict[str, Any]:
    blocked = block_all_sublattices_from_shells(fine_shells, w) if fine_shells is not None else block_all_sublattices(fine, w)
    sc = score(blocked, direct, KEY_SCORE)
    bmet = metrics(blocked)
    dmet = metrics(direct)
    rows = []
    for i, op in enumerate(KEY_SCORE):
        rows.append({
            "match": label,
            "operator": op,
            "blocked_mean": float(sc["b_mean"][i]),
            "direct_mean": float(sc["d_mean"][i]),
            "delta": float(sc["diff"][i]),
            "combined_error": float(sc["sem"][i]),
            "z": float(sc["z"][i]),
        })
    for op in ["Binder_U4", "susceptibility", "xi_over_L"]:
        rows.append({"match": label, "operator": op, "blocked_mean": bmet[op], "direct_mean": dmet[op], "delta": bmet[op] - dmet[op], "combined_error": "", "z": ""})
    return {"blocked": blocked, "score": {k: v for k, v in sc.items() if k not in {"diff", "sem", "z", "b_mean", "d_mean"}}, "blocked_metrics": bmet, "direct_metrics": dmet, "operator_rows": rows, "fft": fft_stats(w, size), "roundtrip": inverse_roundtrip_error(fine[: min(16, len(fine))], w)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--compare-n", type=int, default=1000)
    p.add_argument("--opt-n", type=int, default=300)
    p.add_argument("--maxiter", type=int, default=80)
    p.add_argument("--maxfev", type=int, default=900)
    p.add_argument("--random-starts", type=int, default=6)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fine32, fine32_meta = load_phi(FINE32, 32)
    direct16, direct16_meta = load_phi(DIRECT16, 16)
    fine16, fine16_meta = load_phi(FINE16, 16)
    direct8, direct8_meta = load_phi(DIRECT8, 8)
    n32 = min(args.compare_n, len(fine32))
    fine32_opt = fine32[:n32]
    direct16_opt = direct16
    nopt = min(args.opt_n, n32)
    fine32_obj = fine32[:nopt]
    fine32_obj_shells = shell_sums(fine32_obj)
    fine32_shells = shell_sums(fine32_opt)
    fine16_shells = shell_sums(fine16)

    old = json.loads(OLD5.read_text())
    ow = old["weights_shells"]
    starts = [
        ("identity", np.zeros(7)),
        ("old5_embed", np.array([ow["w10"], ow["w11"], ow["w20"], ow["w21"], ow["w22"], 0.0, 0.0], dtype=np.float64)),
        ("smooth_positive", np.array([0.045, 0.018, 0.018, 0.006, -0.002, 0.006, 0.002])),
        ("broad_alt", np.array([0.03, 0.015, 0.02, 0.01, -0.005, 0.012, 0.004])),
    ]
    rng = np.random.default_rng(SEED)
    for i in range(args.random_starts):
        starts.append((f"random_{i+1}", np.array([
            rng.uniform(-0.06, 0.10),
            rng.uniform(-0.05, 0.06),
            rng.uniform(-0.04, 0.07),
            rng.uniform(-0.03, 0.04),
            rng.uniform(-0.03, 0.03),
            rng.uniform(-0.025, 0.035),
            rng.uniform(-0.02, 0.025),
        ])))
    objective = objective_factory(fine32_obj_shells, direct16_opt, KEY_SCORE, 32)
    bounds = [(-0.16, 0.18), (-0.12, 0.12), (-0.10, 0.12), (-0.08, 0.08), (-0.08, 0.08), (-0.08, 0.08), (-0.06, 0.06)]
    opt_rows = []
    candidates = []
    for label, x0 in starts:
        res = minimize(objective, x0, method="Powell", bounds=bounds, options={"maxiter": args.maxiter, "maxfev": args.maxfev, "xtol": 2e-4, "ftol": 2e-5, "disp": False})
        w = weights_from_x(np.asarray(res.x, dtype=np.float64))
        ev = evaluate_pair("L32_to_L16", fine32_opt, direct16_opt, w, 32, fine32_shells)
        row = {"start": label, "success": bool(res.success), "fun": float(res.fun), "nfev": int(res.nfev), **w, **ev["score"], **ev["fft"]}
        opt_rows.append(row)
        candidates.append((float(ev["score"]["D_op"]), label, np.asarray(res.x, dtype=np.float64), w, ev))
    candidates.sort(key=lambda t: t[0])
    _dop, best_label, best_x, best_w, primary = candidates[0]
    transfer = evaluate_pair("L16_to_L8", fine16, direct8, best_w, 16, fine16_shells)
    old_w = {"w00": ow["w00"], "w01": ow["w10"], "w11": ow["w11"], "w20": ow["w20"], "w21": ow["w21"], "w22": ow["w22"], "w30": 0.0, "w31": 0.0}
    old_primary = evaluate_pair("old5_L32_to_L16", fine32_opt, direct16_opt, old_w, 32, fine32_shells)
    old_transfer = evaluate_pair("old5_L16_to_L8", fine16, direct8, old_w, 16, fine16_shells)

    write_csv(OUT_DIR / "optimization_log.csv", opt_rows)
    write_csv(OUT_DIR / "operator_matching_table.csv", primary["operator_rows"])
    write_csv(OUT_DIR / "operator_matching_transfer_L16_to_L8.csv", transfer["operator_rows"])
    write_csv(OUT_DIR / "old5_operator_matching_L32_to_L16.csv", old_primary["operator_rows"])
    np.savetxt(OUT_DIR / "kernel_large_7x7.csv", kernel_matrix(best_w), delimiter=",")
    np.savetxt(OUT_DIR / "kernel_large_eta_scaled_7x7.csv", ETA_SCALE * kernel_matrix(best_w), delimiter=",")
    np.savez_compressed(OUT_DIR / "blocked_lam0p022_kappa0p2705_L32_to_L16_kernel_large_fixedeta.npz", phi=primary["blocked"].astype(np.float32), weights=np.array([best_w[k] for k in ORBIT_NAMES], dtype=np.float32), kernel_matrix=kernel_matrix(best_w).astype(np.float32), eta=np.array(ETA, dtype=np.float32), eta_scale=np.array(ETA_SCALE, dtype=np.float32))

    flags = []
    if primary["fft"]["condition_number_abs"] > 2.0 * old_primary["fft"]["condition_number_abs"]:
        flags.append("condition number is more than twice old 5x5")
    if primary["fft"]["min_abs_Keta_tilde"] < 0.25:
        flags.append("min |K_eta(q)| below 0.25")
    if primary["roundtrip"]["max_abs_real_error"] > 1.0e-8 or primary["roundtrip"]["max_abs_imag"] > 1.0e-8:
        flags.append("inverse roundtrip above tolerance")
    summary = {
        "status": "complete",
        "diagnostic_only": True,
        "lambda": LAMBDA_VALUE,
        "kappa": KAPPA,
        "eta": ETA,
        "eta_scale": ETA_SCALE,
        "ansatz": ORBIT_NAMES,
        "normalization": "sum K = 1 before eta scaling",
        "convention": "psi = 2^(eta/2) K phi; all four sublattices used as correlated blocked samples; no sublattice average",
        "fine32": fine32_meta,
        "direct16": direct16_meta,
        "fine16": fine16_meta,
        "direct8": direct8_meta,
        "compare_n_L32": int(n32),
        "opt_n_L32": int(nopt),
        "statistics_note": "Primary L32 ensemble has N=1000; kernel optimization is statistics-limited.",
        "best_start": best_label,
        "weights_shells": best_w,
        "normalization_check": norm_check(best_w),
        "kernel_7x7": kernel_matrix(best_w),
        "eta_scaled_kernel_7x7": ETA_SCALE * kernel_matrix(best_w),
        "primary_L32_to_L16": {k: v for k, v in primary.items() if k != "blocked"},
        "transfer_L16_to_L8": {k: v for k, v in transfer.items() if k != "blocked"},
        "old5_L32_to_L16": {k: v for k, v in old_primary.items() if k != "blocked"},
        "old5_L16_to_L8": {k: v for k, v in old_transfer.items() if k != "blocked"},
        "flags": flags,
    }
    (OUT_DIR / "kernel_large_summary.json").write_text(json.dumps(jsonable(summary), indent=2) + "\n")
    (OUT_DIR / "kernel_coefficients.json").write_text(json.dumps(jsonable({
        "lambda": LAMBDA_VALUE,
        "kappa_cr": KAPPA,
        "eta": ETA,
        "eta_fixed": True,
        "eta_scale": ETA_SCALE,
        "weights_shells": best_w,
        "normalization_check": norm_check(best_w),
        "kernel_7x7": kernel_matrix(best_w),
        "eta_scaled_kernel_7x7": ETA_SCALE * kernel_matrix(best_w),
        **primary["fft"],
    }), indent=2) + "\n")
    report = f"""# Lambda=0.022 Critical Large-Kernel Fixed-Eta Optimization

Diagnostic only. Kernel ansatz has orbit weights `{', '.join(ORBIT_NAMES)}`.
The primary optimization used L32 -> L16 with all four K_eta sublattices as
correlated blocked samples. The L32 ensemble has N={n32}, so this is
statistics-limited.

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
{chr(10).join(f"- {k} = `{best_w[k]:.12g}`" for k in ORBIT_NAMES)}

## Matching Scores

| kernel | match | normalized RMS | D_op | rms z | max | cond(K_eta) |
|---|---|---:|---:|---:|---:|---:|
| large | L32->L16 | {primary['score']['normalized_rms']:.6g} | {primary['score']['D_op']:.6g} | {primary['score']['rms_z']:.6g} | {primary['score']['max_abs_z']:.6g} | {primary['fft']['condition_number_abs']:.6g} |
| old 5x5 | L32->L16 | {old_primary['score']['normalized_rms']:.6g} | {old_primary['score']['D_op']:.6g} | {old_primary['score']['rms_z']:.6g} | {old_primary['score']['max_abs_z']:.6g} | {old_primary['fft']['condition_number_abs']:.6g} |
| large | L16->L8 | {transfer['score']['normalized_rms']:.6g} | {transfer['score']['D_op']:.6g} | {transfer['score']['rms_z']:.6g} | {transfer['score']['max_abs_z']:.6g} | {transfer['fft']['condition_number_abs']:.6g} |
| old 5x5 | L16->L8 | {old_transfer['score']['normalized_rms']:.6g} | {old_transfer['score']['D_op']:.6g} | {old_transfer['score']['rms_z']:.6g} | {old_transfer['score']['max_abs_z']:.6g} | {old_transfer['fft']['condition_number_abs']:.6g} |

Flags: {flags if flags else 'none'}.
"""
    (OUT_DIR / "report.md").write_text(report)
    print(json.dumps({"status": "complete", "out_dir": str(OUT_DIR), "normalized_rms": primary["score"]["normalized_rms"], "condition": primary["fft"]["condition_number_abs"], "flags": flags}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
