#!/usr/bin/env python3
"""Run a self-contained Swendsen MCRG analysis on one ensemble."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

from mcrg.blocking import PRODUCTION_KERNEL, block
from mcrg.operators import names, measure
from mcrg.rg import bootstrap_rg, exponents, leading_real, solve_rg


def load_fields(path: Path, maximum: int | None) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        for key in ("phi", "configs", "arr_0"):
            if key in data:
                phi = data[key]
                break
        else:
            phi = data[data.files[0]]
    return np.asarray(phi[:maximum] if maximum else phi, dtype=np.float64)


def summarize(samples, sector):
    vals = []
    for r in samples:
        try: vals.append(leading_real(r)[0])
        except ValueError: pass
    if not vals: return {"n_valid_bootstrap": 0}
    a = np.asarray(vals)
    exponent_samples = np.log(2.0) / np.log(a) if sector == "even" else 4.0 - 2.0 * np.log(a) / np.log(2.0)
    exponent_name = "nu" if sector == "even" else "eta"
    out = {"lambda_median": float(np.median(a)), "lambda_std": float(np.std(a, ddof=1)), "central_68": [float(x) for x in np.quantile(a, [.16,.84])], "n_valid_bootstrap": len(a), "exponent_name": exponent_name, "exponent_median": float(np.median(exponent_samples)), "exponent_std": float(np.std(exponent_samples, ddof=1)), "exponent_central_68": [float(x) for x in np.quantile(exponent_samples, [.16,.84])]}
    if sector == "even":
        # Derived only: eta=1/4 is an input field normalization, not measured here.
        out["beta_derived_eta_input_0p25"] = {"median": float(np.median(exponent_samples / 8.0)), "std": float(np.std(exponent_samples / 8.0, ddof=1))}
        out["gamma_derived_eta_input_0p25"] = {"median": float(np.median(7.0 * exponent_samples / 4.0)), "std": float(np.std(7.0 * exponent_samples / 4.0, ddof=1))}
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--configs", type=Path, required=True)
    p.add_argument("--kernel", choices=["average", "perfect5", "perfect7"], required=True)
    p.add_argument("--average-normalization", choices=["literal", "matched"], default="matched")
    p.add_argument("--kernel-path", type=Path, default=PRODUCTION_KERNEL)
    p.add_argument("--max-configs", type=int)
    p.add_argument("--levels", type=int, default=4)
    p.add_argument("--bootstrap", type=int, default=200)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--svd-rtol", type=float, default=1e-10)
    p.add_argument("--svd-sensitivity", type=float, nargs="*", default=[1e-4, 1e-6, 1e-8, 1e-10, 1e-12])
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    fields = [load_fields(args.configs, args.max_configs)]
    for _ in range(args.levels): fields.append(block(fields[-1], args.kernel, args.average_normalization, args.kernel_path))
    kernel_sum = float(block(np.ones((1, 8, 8)), args.kernel, args.average_normalization, args.kernel_path)[0, 0, 0])
    out = {"input": str(args.configs), "n_configs": len(fields[0]), "kernel": args.kernel, "average_normalization": args.average_normalization, "kernel_path": str(args.kernel_path), "lattice_sizes": [int(x.shape[-1]) for x in fields], "svd_rtol": args.svd_rtol, "zero_mode_response": kernel_sum, "magnetic_normalization_warning": "If the blocker has field zero-mode response z, O1 alone gives lambda_h=4/z identically; eta inferred from O1 is then imposed by field normalization, not an independent universality test.", "results": []}
    for sector in ("even", "odd"):
        ordered = names(sector)
        cutoffs = ([3, 5, 7] if sector == "even" else [1, 2, 3, 4])
        for k in cutoffs:
            op = ordered[:k]; obs = [measure(x, op) for x in fields]
            for n in range(len(fields) - 1):
                result = solve_rg(obs[n], obs[n + 1], args.svd_rtol)
                boot = bootstrap_rg(obs[n], obs[n + 1], args.svd_rtol, args.bootstrap, args.seed + 1000*n + k)
                lam, idx = leading_real(result)
                sensitivity = {}
                for rtol in args.svd_sensitivity:
                    try:
                        sensitivity[str(rtol)] = leading_real(solve_rg(obs[n], obs[n + 1], rtol))[0]
                    except ValueError:
                        sensitivity[str(rtol)] = None
                row = {"sector": sector, "operators": op, "pair": f"{fields[n].shape[-1]}->{fields[n+1].shape[-1]}", "lambda": lam, "exponents": exponents(lambda_t=lam if sector == "even" else None, lambda_h=lam if sector == "odd" else None), "bootstrap": summarize(boot, sector), "condition_number": result.condition_number, "singular_values": result.singular_values.tolist(), "A_eigenvalues": result.a_eigenvalues.tolist(), "svd_sensitivity_lambda": sensitivity, "leading_right_eigenvector": result.right_eigenvectors[:, idx].real.tolist(), "leading_left_eigenvector": result.left_eigenvectors[:, idx].real.tolist()}
                out["results"].append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")

if __name__ == "__main__": main()
