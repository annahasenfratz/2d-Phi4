#!/usr/bin/env python3
"""Focused native-pilot diagnostics for algebraic magnetic normalization effects."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from mcrg.analyze import load_fields
from mcrg.blocking import PRODUCTION_KERNEL, average_block, perfect_block, perfect_kernel
from mcrg.operators import names, measure
from mcrg.rg import leading_real, solve_rg


PAIRS = ("128->64", "64->32", "32->16", "16->8")
CUTS = (1e-12, 1e-10, 1e-8, 1e-6)


def hierarchy(phi, kernel: str, factor: float = 1.0):
    fields = [phi]
    for _ in range(4):
        if kernel == "average":
            fields.append(factor * average_block(fields[-1], "matched"))
        else:
            fields.append(factor * perfect_block(fields[-1]))
    return fields


def parity_sums(kernel: str) -> tuple[float, dict[str, float], np.ndarray]:
    if kernel == "average":
        z = float(perfect_kernel().matrix.sum())
        matrix = np.full((2, 2), z / 4.0)
        offsets = range(0, 2)
    else:
        matrix = np.asarray(perfect_kernel().matrix, dtype=float)
        offsets = range(-(matrix.shape[0] // 2), matrix.shape[0] // 2 + 1)
    cs = {(a, b): 0.0 for a in range(2) for b in range(2)}
    for i, dx in enumerate(offsets):
        for j, dy in enumerate(offsets):
            cs[(dx % 2, dy % 2)] += float(matrix[i, j])
    return float(matrix.sum()), {f"C{a}{b}": cs[(a,b)] for a in range(2) for b in range(2)}, matrix


def unit_vector(v):
    v = np.asarray(v).real
    v = v / np.linalg.norm(v)
    return v if v[0] >= 0 else -v


def make_convergence(data, output: Path, field: str, target: float, ylabel: str):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), sharey=True, constrained_layout=True)
    for ax, (_, d) in zip(axes, data.items()):
        for k in (3, 5, 7):
            rows = [r for r in d["results"] if r["sector"] == "even" and len(r["operators"]) == k]
            ys = [r["lambda"] if field == "lambda" else r["exponents"]["nu"] for r in rows]
            es = [r["bootstrap"]["lambda_std"] if field == "lambda" else r["bootstrap"]["exponent_std"] for r in rows]
            ax.errorbar(range(4), ys, yerr=es, marker="o", capsize=2, label=f"{k} operators")
        ax.axhline(target, color="black", ls="--", lw=.8)
        ax.set(title=d["kernel"], xlabel="blocking pair", xticks=range(4), xticklabels=PAIRS)
        ax.grid(alpha=.25); ax.legend(fontsize=8)
    axes[0].set_ylabel(ylabel)
    output.parent.mkdir(parents=True, exist_ok=True); fig.savefig(output); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--configs", type=Path, required=True)
    p.add_argument("--average-json", type=Path, required=True)
    p.add_argument("--perfect-json", type=Path, required=True)
    p.add_argument("--max-configs", type=int, default=500)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--label", default="native_l128_n500", help="stem for diagnostic JSON/Markdown outputs")
    args = p.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    phi = load_fields(args.configs, args.max_configs)
    pilot = {"average": json.loads(args.average_json.read_text()), "perfect7": json.loads(args.perfect_json.read_text())}
    report = {"n_configs": int(len(phi)), "kernels": {}, "normalization_scan": {}, "even": {}}
    all_fields = {}
    for kernel in pilot:
        fields = hierarchy(phi, kernel); all_fields[kernel] = fields
        ksum, polyphase, matrix = parity_sums(kernel)
        residuals = []
        for n, pair in enumerate(PAIRS):
            mf, mc = fields[n].sum(axis=(1,2)), fields[n+1].sum(axis=(1,2))
            r = mc - (ksum / 4.0) * mf
            residuals.append({"pair": pair, "mean": float(r.mean()), "std": float(r.std(ddof=1)), "max_abs": float(np.max(np.abs(r))), "rms": float(np.sqrt(np.mean(r*r))), "std_over_std_Mprime": float(r.std(ddof=1) / mc.std(ddof=1))})
        report["kernels"][kernel] = {"Ksum": ksum, "Ksum_over_4": ksum/4, "polyphase_sums": polyphase, "polyphase_equal_Ksum_over_4": bool(np.allclose(list(polyphase.values()), ksum/4, rtol=1e-12, atol=1e-12)), "magnetization_residuals": residuals}
    # Scatter at the first blocking pair: strongest visual direct check.
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4), constrained_layout=True)
    for ax, kernel in zip(axes, pilot):
        ksum = report["kernels"][kernel]["Ksum"]; mf = all_fields[kernel][0].sum((1,2)); mc = all_fields[kernel][1].sum((1,2)); predicted = ksum/4*mf
        ax.scatter(predicted, mc, s=7, alpha=.55)
        lim = [min(predicted.min(),mc.min()), max(predicted.max(),mc.max())]; ax.plot(lim,lim,"k--",lw=.8)
        ax.set(title=kernel, xlabel=r"$(K_{sum}/4)M$", ylabel=r"$M'$", aspect="equal"); ax.grid(alpha=.2)
    fig.savefig(args.output_dir / "magnetization_relation_scatter.pdf"); plt.close(fig)
    # Scaling an entire blocker is not an eta measurement: show the exact O1 response.
    perfect_unit_sum_multiplier = 1.0 / report["kernels"]["perfect7"]["Ksum"]
    for c in (.9, perfect_unit_sum_multiplier, 1.0, 1.0905077, 1.2):
        fields = hierarchy(phi, "perfect7", c)
        obs = [measure(x, names("odd")[:1]) for x in fields]
        lams = [leading_real(solve_rg(obs[n], obs[n+1], 1e-12))[0] for n in range(4)]
        report["normalization_scan"][str(c)] = {"lambda_h_by_pair": dict(zip(PAIRS, lams)), "c_times_lambda_h": dict(zip(PAIRS, [c*x for x in lams]))}
    # T drift and reproducibly oriented leading eigenvectors, computed directly from pilot fields.
    for kernel, fields in all_fields.items():
        report["even"][kernel] = {}
        for k in (3,5,7):
            op = names("even")[:k]; obs = [measure(x, op) for x in fields]
            results = [solve_rg(obs[n], obs[n+1], 1e-10) for n in range(4)]
            vectors, rows = [], []
            for pair, r in zip(PAIRS, results):
                lam, idx = leading_real(r); v = unit_vector(r.right_eigenvectors[:,idx]); vectors.append(v)
                sensitivity = {str(cut): leading_real(solve_rg(obs[PAIRS.index(pair)], obs[PAIRS.index(pair)+1], cut))[0] for cut in CUTS}
                cutoff_values = np.asarray(list(sensitivity.values()))
                relative_span = float((cutoff_values.max() - cutoff_values.min()) / abs(lam))
                rows.append({"pair": pair, "lambda_t": lam, "operators": op, "right_eigenvector": v.tolist(), "singular_values_A": r.singular_values.tolist(), "condition_number_A": r.condition_number, "svd_cutoff_sensitivity": sensitivity, "svd_relative_span": relative_span, "svd_cutoff_flag": bool(relative_span > .01)})
            overlaps = [float(abs(np.dot(vectors[n], vectors[n+1]))) for n in range(3)]
            drift = [float(np.linalg.norm(results[n].T-results[n+1].T, 'fro') / np.linalg.norm(results[n].T, 'fro')) for n in range(3)]
            report["even"][kernel][str(k)] = {"pairs": rows, "successive_eigenvector_overlaps": dict(zip(("128->64 / 64->32", "64->32 / 32->16", "32->16 / 16->8"), overlaps)), "successive_T_relative_frobenius_differences": dict(zip(("128->64 / 64->32", "64->32 / 32->16", "32->16 / 16->8"), drift))}
    make_convergence(pilot, args.output_dir / "even_lambda_t_convergence.pdf", "lambda", 2., r"$\lambda_t$")
    make_convergence(pilot, args.output_dir / "even_nu_convergence.pdf", "nu", 1., r"$\nu$")
    (args.output_dir / f"{args.label}_diagnostics.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# Native L128, N=500 MCRG diagnostics", "", "## Kernel parity/polyphase sums", "", "| kernel | Ksum | C00 | C01 | C10 | C11 | all equal Ksum/4? |", "|---|---:|---:|---:|---:|---:|---|"]
    for key, x in report["kernels"].items():
        c=x["polyphase_sums"]; lines.append(f"| {key} | {x['Ksum']:.12g} | {c['C00']:.12g} | {c['C01']:.12g} | {c['C10']:.12g} | {c['C11']:.12g} | {x['polyphase_equal_Ksum_over_4']} |")
    lines += ["", "## Direct magnetization residuals", "", "| kernel | pair | mean(r) | std(r) | max |r| | RMS(r) | std(r)/std(M') |", "|---|---|---:|---:|---:|---:|---:|"]
    for key,x in report["kernels"].items():
        for q in x["magnetization_residuals"]: lines.append(f"| {key} | {q['pair']} | {q['mean']:.3g} | {q['std']:.3g} | {q['max_abs']:.3g} | {q['rms']:.3g} | {q['std_over_std_Mprime']:.3g} |")
    lines += ["", "## Perfect7 normalization scan: O1 lambda_h", "", "Here `c` multiplies every stored 7x7 coefficient; `c=1/Ksum` makes its total sum one.", "", "| multiplier c | " + " | ".join(PAIRS) + " |", "|---:|" + "---:|"*4]
    for c,x in report["normalization_scan"].items(): lines.append("| " + c + " | " + " | ".join(f"{x['lambda_h_by_pair'][pair]:.8g}" for pair in PAIRS) + " |")
    lines += ["", "## Leading even eigenvectors, overlaps, and matrix drift", ""]
    for kernel, bases in report["even"].items():
        lines += [f"### {kernel}", ""]
        for k, x in bases.items():
            lines += [f"#### {k} operators: `{'`, `'.join(x['pairs'][0]['operators'])}`", "", "| pair | lambda_t | right eigenvector (E1...) | cond(A) |", "|---|---:|---|---:|"]
            for q in x["pairs"]: lines.append(f"| {q['pair']} | {q['lambda_t']:.7g} | " + ", ".join(f"{v:.5f}" for v in q['right_eigenvector']) + f" | {q['condition_number_A']:.3g} |")
            lines += ["", f"overlaps: `{x['successive_eigenvector_overlaps']}`", "", f"relative Frobenius T drifts: `{x['successive_T_relative_frobenius_differences']}`", ""]
    lines += ["## SVD-cutoff sensitivity: leading even eigenvalue", "", "Flag means the span across relative cutoffs 1e-12, 1e-10, 1e-8, and 1e-6 exceeds 1% of the nominal eigenvalue.", "", "| kernel | basis | pair | λ(1e-12) | λ(1e-10) | λ(1e-8) | λ(1e-6) | relative span | flag |", "|---|---:|---|---:|---:|---:|---:|---:|---|"]
    for kernel, bases in report["even"].items():
        for k, x in bases.items():
            for q in x["pairs"]:
                s=q["svd_cutoff_sensitivity"]
                lines.append(f"| {kernel} | {k} | {q['pair']} | {s['1e-12']:.6g} | {s['1e-10']:.6g} | {s['1e-08']:.6g} | {s['1e-06']:.6g} | {q['svd_relative_span']:.3g} | {q['svd_cutoff_flag']} |")
    (args.output_dir / f"{args.label}_diagnostics.md").write_text("\n".join(lines)+"\n")

if __name__ == "__main__": main()
