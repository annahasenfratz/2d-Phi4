#!/usr/bin/env python3
"""Block-replication fill diagnostic for inverse-blocking initialization."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
BASE = PROJECT / "outputs" / "inverse_blocking_step_by_step"
OUT = BASE / "block_replicate_fill"
PLOTS = OUT / "plots"

LAMBDA = 1.0
KAPPA_F = 0.320
BOOTSTRAP_N = 256
RNG_SEED = 20240623


def block_replicate(a: np.ndarray) -> np.ndarray:
    out = np.empty((a.shape[0], a.shape[1] * 2, a.shape[2] * 2), dtype=a.dtype)
    out[:, 0::2, 0::2] = a
    out[:, 1::2, 0::2] = a
    out[:, 0::2, 1::2] = a
    out[:, 1::2, 1::2] = a
    return out


def obs(configs: np.ndarray) -> dict[str, float]:
    _, ly, lx = configs.shape
    v = ly * lx
    m_cfg = configs.mean(axis=(-2, -1))
    nn_cfg = 0.5 * (
        (configs * np.roll(configs, -1, axis=-2)).mean(axis=(-2, -1))
        + (configs * np.roll(configs, -1, axis=-1)).mean(axis=(-2, -1))
    )
    diag_cfg = (configs * np.roll(np.roll(configs, -1, axis=-2), -1, axis=-1)).mean(axis=(-2, -1))
    twonn_cfg = 0.5 * (
        (configs * np.roll(configs, -2, axis=-2)).mean(axis=(-2, -1))
        + (configs * np.roll(configs, -2, axis=-1)).mean(axis=(-2, -1))
    )
    m2 = float(np.mean(m_cfg**2))
    m4 = float(np.mean(m_cfg**4))
    b4 = m4 / (m2 * m2) if m2 > 0 else math.nan
    u4 = 1.0 - b4 / 3.0 if math.isfinite(b4) else math.nan
    ft = np.fft.fft2(configs, axes=(-2, -1))
    chi = float(v * (np.mean(m_cfg**2) - np.mean(m_cfg) ** 2))
    fmin = float(0.5 * (np.mean(np.abs(ft[:, 1, 0]) ** 2) + np.mean(np.abs(ft[:, 0, 1]) ** 2)) / v)
    ratio = chi / fmin - 1.0 if fmin > 0 else math.nan
    xi = (1.0 / (2.0 * math.sin(math.pi / lx))) * math.sqrt(ratio) if ratio > 0 else math.nan
    return {
        "m": float(np.mean(m_cfg)),
        "|m|": float(np.mean(np.abs(m_cfg))),
        "phi2": float(np.mean(configs**2)),
        "phi4": float(np.mean(configs**4)),
        "NN": float(np.mean(nn_cfg)),
        "diag": float(np.mean(diag_cfg)),
        "2nn": float(np.mean(twonn_cfg)),
        "Binder_U4": float(u4),
        "Binder_B4": float(b4),
        "xi": float(xi) if math.isfinite(xi) else math.nan,
        "xi/L": float(xi / lx) if math.isfinite(xi) else math.nan,
    }


def bootstrap(configs: np.ndarray, rng: np.random.Generator) -> dict[str, dict[str, float]]:
    mean = obs(configs)
    reps = {k: [] for k in mean}
    n = configs.shape[0]
    for _ in range(BOOTSTRAP_N):
        idx = rng.integers(0, n, size=n)
        val = obs(configs[idx])
        for k in reps:
            reps[k].append(val[k])
    return {k: {"mean": mean[k], "error": float(np.nanstd(reps[k], ddof=1))} for k in mean}


def action_components(configs: np.ndarray) -> dict[str, float]:
    o = obs(configs)
    hopping = -4.0 * KAPPA_F * o["NN"]
    phi2_term = (1.0 - 2.0 * LAMBDA) * o["phi2"]
    phi4_term = LAMBDA * o["phi4"]
    return {
        "kappa": KAPPA_F,
        "lambda": LAMBDA,
        "NN": o["NN"],
        "hopping_density_minus_2kappa_sum_mu": hopping,
        "phi2_density_coeff_1_minus_2lambda": phi2_term,
        "phi4_density_lambda": phi4_term,
        "total_action_density_convention": hopping + phi2_term + phi4_term,
    }


def nn2_all_sites(configs: np.ndarray) -> float:
    return float(
        0.5
        * (
            np.mean((configs * np.roll(configs, -1, axis=-2)) ** 2)
            + np.mean((configs * np.roll(configs, -1, axis=-1)) ** 2)
        )
    )


def nn2_ee_start(configs: np.ndarray) -> float:
    ee = configs[:, 0::2, 0::2]
    down = configs[:, 1::2, 0::2]
    right = configs[:, 0::2, 1::2]
    return float(0.5 * (np.mean((ee * down) ** 2) + np.mean((ee * right) ** 2)))


def nn2_ee_to_ee(configs: np.ndarray) -> float:
    ee = configs[:, 0::2, 0::2]
    return float(
        0.5
        * (
            np.mean((ee * np.roll(ee, -1, axis=-2)) ** 2)
            + np.mean((ee * np.roll(ee, -1, axis=-1)) ** 2)
        )
    )


def nn2_compact(configs: np.ndarray) -> float:
    return float(
        0.5
        * (
            np.mean((configs * np.roll(configs, -1, axis=-2)) ** 2)
            + np.mean((configs * np.roll(configs, -1, axis=-1)) ** 2)
        )
    )


def write_nn2_even_even_comparison(
    fine: np.ndarray,
    a: np.ndarray,
    old_neighbor: np.ndarray,
    phi_block: np.ndarray,
    phi_oracle_block: np.ndarray,
) -> list[dict]:
    rows = []
    ensembles = {
        "original_fine": {"field": fine, "compact": fine[:, 0::2, 0::2]},
        "chi_alias_even_even_compact": {"field": None, "compact": a},
        "block_replicated_phi_block": {"field": phi_block, "compact": a},
        "oracle_block_replicated": {"field": phi_oracle_block, "compact": fine[:, 0::2, 0::2]},
        "old_neighbor_filled": {"field": old_neighbor, "compact": old_neighbor[:, 0::2, 0::2]},
    }
    original_all = nn2_all_sites(fine)
    original_ee_start = nn2_ee_start(fine)
    original_ee_to_ee = nn2_ee_to_ee(fine)
    original_compact = nn2_compact(fine[:, 0::2, 0::2])
    for name, parts in ensembles.items():
        field = parts["field"]
        compact = parts["compact"]
        if field is None:
            all_sites = math.nan
            ee_start = math.nan
            ee_to_ee = math.nan
        else:
            all_sites = nn2_all_sites(field)
            ee_start = nn2_ee_start(field)
            ee_to_ee = nn2_ee_to_ee(field)
        compact_val = nn2_compact(compact)
        reference = original_compact if name == "chi_alias_even_even_compact" else original_all
        rows.append(
            {
                "ensemble": name,
                "nn2_all_sites": all_sites,
                "nn2_ee_start": ee_start,
                "nn2_ee_to_ee": ee_to_ee,
                "nn2_compact_if_applicable": compact_val,
                "difference_vs_original": compact_val - reference if name == "chi_alias_even_even_compact" else all_sites - reference,
                "ratio_vs_original": compact_val / reference if name == "chi_alias_even_even_compact" else all_sites / reference,
                "original_reference_used": "original_fine_compact_even_even" if name == "chi_alias_even_even_compact" else "original_fine_all_sites",
                "original_nn2_all_sites": original_all,
                "original_nn2_ee_start": original_ee_start,
                "original_nn2_ee_to_ee": original_ee_to_ee,
                "original_nn2_compact_even_even": original_compact,
            }
        )
    with (OUT / "nn2_even_even_comparison.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    md = [
        "# Even-even nn2 / NN2 Comparison\n\n",
        "Definition used here: `nn2 = average over x,mu (phi_x phi_{x+mu})^2`, matching the requested nearest-neighbor squared-link convention. ",
        "`nn2_ee_start` restricts the starting site to even-even fine sites and uses one-lattice-step links. ",
        "`nn2_ee_to_ee` uses distance-2 fine links from even-even sites, equivalent to compact nearest-neighbor links on the even-even field.\n\n",
        "| ensemble | nn2_all_sites | nn2_ee_start | nn2_ee_to_ee | nn2_compact_if_applicable | diff vs original | ratio vs original |\n",
        "|---|---:|---:|---:|---:|---:|---:|\n",
    ]
    for row in rows:
        md.append(
            f"| {row['ensemble']} | {row['nn2_all_sites']:.12g} | {row['nn2_ee_start']:.12g} | "
            f"{row['nn2_ee_to_ee']:.12g} | {row['nn2_compact_if_applicable']:.12g} | "
            f"{row['difference_vs_original']:.12g} | {row['ratio_vs_original']:.12g} |\n"
        )
    (OUT / "nn2_even_even_comparison.md").write_text("".join(md))
    return rows


def write_operator_comparison(ensembles: dict[str, np.ndarray]) -> dict:
    rng = np.random.default_rng(RNG_SEED)
    stats = {name: bootstrap(arr, rng) for name, arr in ensembles.items()}
    base = stats["original_fine"]
    rows = []
    for name, vals in stats.items():
        for op, stat in vals.items():
            diff = stat["mean"] - base[op]["mean"]
            den = math.sqrt(stat["error"] ** 2 + base[op]["error"] ** 2)
            rows.append(
                {
                    "ensemble": name,
                    "L": int(ensembles[name].shape[-1]),
                    "operator": op,
                    "mean": stat["mean"],
                    "error": stat["error"],
                    "original_fine_mean": base[op]["mean"],
                    "original_fine_error": base[op]["error"],
                    "difference_vs_original_fine": diff,
                    "z_score_vs_original_fine": diff / den if den > 0 else math.nan,
                }
            )
    with (OUT / "operator_comparison.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    md = ["# Block-Replication Operator Comparison\n\nErrors are bootstrap standard deviations.\n"]
    for name in ensembles:
        md += [f"\n## {name}\n", "| operator | mean | error | diff vs original fine | z |\n", "|---|---:|---:|---:|---:|\n"]
        for row in rows:
            if row["ensemble"] == name:
                md.append(
                    f"| {row['operator']} | {row['mean']:.8g} | {row['error']:.3g} | "
                    f"{row['difference_vs_original_fine']:.8g} | {row['z_score_vs_original_fine']:.3g} |\n"
                )
    (OUT / "operator_comparison.md").write_text("".join(md))
    return {"stats": stats, "rows": rows}


def write_action_components(ensembles: dict[str, np.ndarray]) -> dict:
    rows = []
    for name, arr in ensembles.items():
        vals = action_components(arr)
        rows.append({"ensemble": name, **vals})
    with (OUT / "action_components.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {r["ensemble"]: r for r in rows}


def save_plots(fine: np.ndarray, alias: np.ndarray, block: np.ndarray, oracle: np.ndarray) -> None:
    PLOTS.mkdir(exist_ok=True)
    for i in range(min(4, fine.shape[0])):
        panels = [
            ("original fine", fine[i]),
            ("chi_alias", alias[i]),
            ("block replicated", block[i]),
            ("oracle block replicated", oracle[i]),
            ("block - original", block[i] - fine[i]),
        ]
        fig, axes = plt.subplots(1, 5, figsize=(15, 3.2), constrained_layout=True)
        for ax, (title, data) in zip(axes, panels):
            im = ax.imshow(data, origin="lower", cmap="coolwarm")
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, shrink=0.72)
        fig.savefig(PLOTS / f"config_{i:02d}_block_replicate.pdf")
        fig.savefig(PLOTS / f"config_{i:02d}_block_replicate.png", dpi=180)
        plt.close(fig)


def row_lookup(rows: list[dict], ensemble: str, operator: str) -> dict:
    return next(r for r in rows if r["ensemble"] == ensemble and r["operator"] == operator)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(exist_ok=True)

    fine = np.load(BASE / "input_fine_batch.npy")
    alias = np.load(BASE / "inverse_kernel_alias_field.npy")
    old_neighbor = np.load(BASE / "neighbor_filled_init.npy")
    a = alias[:, 0::2, 0::2]
    phi_block = block_replicate(a)
    a_oracle = fine[:, 0::2, 0::2]
    phi_oracle_block = block_replicate(a_oracle)

    np.save(OUT / "block_replicated_phi_block.npy", phi_block)
    np.save(OUT / "oracle_block_replicated_phi_oracle_block.npy", phi_oracle_block)

    preservation = {
        "max_abs_phi_block_even_even_minus_chi_alias_even_even": float(
            np.max(np.abs(phi_block[:, 0::2, 0::2] - a))
        ),
        "alias_off_even_even_max_abs": float(
            max(
                np.max(np.abs(alias[:, 1::2, 0::2])),
                np.max(np.abs(alias[:, 0::2, 1::2])),
                np.max(np.abs(alias[:, 1::2, 1::2])),
            )
        ),
    }
    (OUT / "preservation_check.json").write_text(json.dumps(preservation, indent=2) + "\n")

    moment_identity = {
        "chi_alias_even_even_phi2": float(np.mean(a**2)),
        "chi_alias_even_even_phi4": float(np.mean(a**4)),
        "phi_block_full_phi2": float(np.mean(phi_block**2)),
        "phi_block_full_phi4": float(np.mean(phi_block**4)),
        "abs_diff_phi2": float(abs(np.mean(phi_block**2) - np.mean(a**2))),
        "abs_diff_phi4": float(abs(np.mean(phi_block**4) - np.mean(a**4))),
    }
    (OUT / "moment_identity_check.json").write_text(json.dumps(moment_identity, indent=2) + "\n")
    (OUT / "moment_identity_check.md").write_text(
        "# Moment Identity Check\n\n"
        f"- chi_alias even-even phi2: {moment_identity['chi_alias_even_even_phi2']:.12g}\n"
        f"- phi_block full phi2: {moment_identity['phi_block_full_phi2']:.12g}\n"
        f"- absolute difference phi2: {moment_identity['abs_diff_phi2']:.12g}\n"
        f"- chi_alias even-even phi4: {moment_identity['chi_alias_even_even_phi4']:.12g}\n"
        f"- phi_block full phi4: {moment_identity['phi_block_full_phi4']:.12g}\n"
        f"- absolute difference phi4: {moment_identity['abs_diff_phi4']:.12g}\n"
    )

    oracle_check = {
        "original_even_even_phi2": float(np.mean(a_oracle**2)),
        "original_even_even_phi4": float(np.mean(a_oracle**4)),
        "phi_oracle_block_full_phi2": float(np.mean(phi_oracle_block**2)),
        "phi_oracle_block_full_phi4": float(np.mean(phi_oracle_block**4)),
        "abs_diff_phi2": float(abs(np.mean(phi_oracle_block**2) - np.mean(a_oracle**2))),
        "abs_diff_phi4": float(abs(np.mean(phi_oracle_block**4) - np.mean(a_oracle**4))),
    }
    (OUT / "oracle_block_check.json").write_text(json.dumps(oracle_check, indent=2) + "\n")

    oracle_rows = []
    for name, arr in {
        "original_fine": fine,
        "original_even_even_compact": a_oracle,
        "oracle_block_replicated": phi_oracle_block,
        "chi_alias_even_even_compact": a,
        "block_replicated_phi_block": phi_block,
    }.items():
        vals = obs(arr)
        for op in ["m", "|m|", "phi2", "phi4", "NN", "diag", "2nn", "Binder_U4", "Binder_B4", "xi", "xi/L"]:
            oracle_rows.append({"ensemble": name, "operator": op, "value": vals[op]})
    with (OUT / "oracle_block_observables.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(oracle_rows[0]))
        writer.writeheader()
        writer.writerows(oracle_rows)

    ensembles = {
        "original_fine": fine,
        "chi_alias_even_even_compact": a,
        "old_neighbor_filled": old_neighbor,
        "block_replicated_phi_block": phi_block,
        "oracle_block_replicated": phi_oracle_block,
    }
    comparison = write_operator_comparison(ensembles)
    actions = write_action_components(ensembles)
    nn2_rows = write_nn2_even_even_comparison(fine, a, old_neighbor, phi_block, phi_oracle_block)
    save_plots(fine, alias, phi_block, phi_oracle_block)

    block_phi2 = row_lookup(comparison["rows"], "block_replicated_phi_block", "phi2")
    block_phi4 = row_lookup(comparison["rows"], "block_replicated_phi_block", "phi4")
    old_phi2 = row_lookup(comparison["rows"], "old_neighbor_filled", "phi2")
    old_phi4 = row_lookup(comparison["rows"], "old_neighbor_filled", "phi4")
    nn_block = row_lookup(comparison["rows"], "block_replicated_phi_block", "NN")
    diag_block = row_lookup(comparison["rows"], "block_replicated_phi_block", "diag")
    twonn_block = row_lookup(comparison["rows"], "block_replicated_phi_block", "2nn")
    nn2_original = next(r for r in nn2_rows if r["ensemble"] == "original_fine")
    nn2_alias = next(r for r in nn2_rows if r["ensemble"] == "chi_alias_even_even_compact")
    nn2_block = next(r for r in nn2_rows if r["ensemble"] == "block_replicated_phi_block")
    nn2_oracle = next(r for r in nn2_rows if r["ensemble"] == "oracle_block_replicated")
    nn2_old = next(r for r in nn2_rows if r["ensemble"] == "old_neighbor_filled")

    summary = {
        "preservation_check": preservation,
        "moment_identity_check": moment_identity,
        "oracle_block_check": oracle_check,
        "selected_operator_rows": {
            "old_neighbor_phi2": old_phi2,
            "old_neighbor_phi4": old_phi4,
            "block_phi2": block_phi2,
            "block_phi4": block_phi4,
            "block_NN": nn_block,
            "block_diag": diag_block,
            "block_2nn": twonn_block,
        },
        "action_components": actions,
        "nn2_even_even_comparison": nn2_rows,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    report = f"""# Block-Replication Fill Diagnostic

## Answers

1. Did block replication preserve even-even sites exactly?

Yes. The maximum absolute preservation error is {preservation['max_abs_phi_block_even_even_minus_chi_alias_even_even']:.12g}.

2. Did full-volume phi2/phi4 equal even-even phi2/phi4 exactly?

Yes up to roundoff. For the inverse alias field:

- even-even phi2: {moment_identity['chi_alias_even_even_phi2']:.12g}
- block-replicated full-volume phi2: {moment_identity['phi_block_full_phi2']:.12g}
- absolute phi2 difference: {moment_identity['abs_diff_phi2']:.12g}
- even-even phi4: {moment_identity['chi_alias_even_even_phi4']:.12g}
- block-replicated full-volume phi4: {moment_identity['phi_block_full_phi4']:.12g}
- absolute phi4 difference: {moment_identity['abs_diff_phi4']:.12g}

3. Inverse alias vs original fine moments:

- inverse even-even phi2: {moment_identity['chi_alias_even_even_phi2']:.12g}
- inverse even-even phi4: {moment_identity['chi_alias_even_even_phi4']:.12g}
- block-replicated full phi2: {moment_identity['phi_block_full_phi2']:.12g}
- block-replicated full phi4: {moment_identity['phi_block_full_phi4']:.12g}
- original fine full phi2: {block_phi2['original_fine_mean']:.12g}
- original fine full phi4: {block_phi4['original_fine_mean']:.12g}

4. For oracle block replication, did phi2/phi4 equal original even-even moments?

Yes. Oracle phi2 difference is {oracle_check['abs_diff_phi2']:.12g}; oracle phi4 difference is {oracle_check['abs_diff_phi4']:.12g}.

5. How do NN, diag, 2nn change under block replication?

Relative to original fine:

- NN: {nn_block['mean']:.12g} vs {nn_block['original_fine_mean']:.12g}, difference {nn_block['difference_vs_original_fine']:.12g}
- diag: {diag_block['mean']:.12g} vs {diag_block['original_fine_mean']:.12g}, difference {diag_block['difference_vs_original_fine']:.12g}
- 2nn: {twonn_block['mean']:.12g} vs {twonn_block['original_fine_mean']:.12g}, difference {twonn_block['difference_vs_original_fine']:.12g}

6. Is block replication a better deterministic initializer than neighbor averaging?

For one-site moments, yes. Neighbor averaging gave phi2 {old_phi2['mean']:.12g} and phi4 {old_phi4['mean']:.12g}; block replication gives phi2 {block_phi2['mean']:.12g} and phi4 {block_phi4['mean']:.12g}. This exactly preserves the inverse even-even moments across the full volume instead of smoothing them down.

It is not uniformly better for all operators. Replication creates piecewise-constant 2x2 plateaus and therefore distorts link observables and UV structure in a different way.

7. What distortions remain for the conditional NF to learn?

The initializer still lacks within-block fluctuations and realistic short-distance variation. The conditional NF must learn residual UV noise, odd-sublattice fluctuations, and corrections to NN/diag/2nn and action-density components while preserving the coarse/block constraints.

## Even-even nn2 / NN2 comparison

Definition used here: `nn2 = average over x,mu (phi_x phi_{{x+mu}})^2`, using the nearest-neighbor squared-link convention requested for `NN2/nn2`.

1. Does chi_alias reproduce original fine nn2 on even-even-to-even-even distance-2 links?

Not exactly. The compact chi_alias nn2 is {nn2_alias['nn2_compact_if_applicable']:.12g}; the original fine even-even compact/distance-2 nn2 is {nn2_original['original_nn2_compact_even_even']:.12g}. The ratio is {nn2_alias['nn2_compact_if_applicable'] / nn2_original['original_nn2_compact_even_even']:.12g}.

2. Does block replication preserve the compact even-even nn2 exactly?

Yes. For block replication, `nn2_ee_to_ee = {nn2_block['nn2_ee_to_ee']:.12g}` and compact nn2 is `{nn2_block['nn2_compact_if_applicable']:.12g}`.

3. How badly does all-site nn2 change under block replication?

Original all-site nn2 is {nn2_original['nn2_all_sites']:.12g}; block-replicated all-site nn2 is {nn2_block['nn2_all_sites']:.12g}, difference {nn2_block['nn2_all_sites'] - nn2_original['nn2_all_sites']:.12g}, ratio {nn2_block['nn2_all_sites'] / nn2_original['nn2_all_sites']:.12g}. The old neighbor-filled all-site nn2 was {nn2_old['nn2_all_sites']:.12g}.

4. Is the remaining nn2 mismatch mainly from the inverse alias field or from the fill rule?

The even-even-to-even-even mismatch is already present in the inverse alias field: chi_alias compact nn2 is {nn2_alias['nn2_compact_if_applicable']:.12g} versus original compact nn2 {nn2_original['original_nn2_compact_even_even']:.12g}. Block replication preserves that compact structure on distance-2 even-even links, but it changes all-site nearest-neighbor nn2 by imposing constant 2x2 plateaus. The oracle block-replicated compact nn2 is {nn2_oracle['nn2_compact_if_applicable']:.12g}, matching the original even-even compact structure by construction.
"""
    (OUT / "report.md").write_text(report)


if __name__ == "__main__":
    main()
