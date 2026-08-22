#!/usr/bin/env python3
"""Make PDF iteration plots and a compact Markdown Swendsen-style table."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("inputs", nargs="+", type=Path)
    p.add_argument("--pdf", type=Path, required=True)
    p.add_argument("--table", type=Path, required=True)
    args = p.parse_args()
    datasets = [(x.stem, json.loads(x.read_text())) for x in args.inputs]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    table = ["| kernel | pair | sector | operators | lambda (bootstrap 1σ) | exponent (bootstrap 1σ) | cond(A) |", "|---|---|---|---:|---:|---:|---:|"]
    for label, data in datasets:
        for r in data["results"]:
            sector, pair, k = r["sector"], r["pair"], len(r["operators"])
            iteration = int(data["lattice_sizes"][0]).bit_length() - int(pair.split("->")[0]).bit_length()
            if sector == "even":
                values = (r["lambda"], r["exponents"].get("nu"))
                targets = (2.0, 1.0); target_axes=(0,1)
            else:
                values = (r["lambda"], r["exponents"].get("eta"))
                targets = (2**(15/8), .25); target_axes=(2,3)
            bootstrap = r["bootstrap"]
            errors = (bootstrap.get("lambda_std"), bootstrap.get("exponent_std"))
            for ax_i, value, target in zip(target_axes, values, targets):
                error = errors[0] if ax_i in (0, 2) else errors[1]
                axes.flat[ax_i].errorbar(iteration, value, yerr=error, marker="o", linestyle="none", capsize=2, label=f"{label}: {sector[0].upper()}{k}")
                axes.flat[ax_i].axhline(target, color="black", lw=.6, ls="--")
            exponent = values[1]
            table.append(f"| {data['kernel']} | {pair} | {sector} | {k} | {r['lambda']:.6g} ± {errors[0]:.2g} | {exponent:.6g} ± {errors[1]:.2g} | {r['condition_number']:.3g} |")
    for ax, title in zip(axes.flat, ("thermal $\\lambda_t$", "$\\nu$", "magnetic $\\lambda_h$", "$\\eta$")):
        ax.set(title=title, xlabel="blocking-pair iteration"); ax.grid(alpha=.25)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", fontsize=7); fig.subplots_adjust(right=.78)
    args.pdf.parent.mkdir(parents=True, exist_ok=True); fig.savefig(args.pdf); plt.close(fig)
    args.table.write_text("\n".join(table)+"\n")
if __name__ == "__main__": main()
