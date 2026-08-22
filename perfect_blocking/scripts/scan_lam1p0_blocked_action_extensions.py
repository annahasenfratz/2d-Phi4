#!/usr/bin/env python3
"""Held-out one-family-at-a-time screen for missing blocked-action operators."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[2]
FIT_SCRIPT = ROOT / "perfect_blocking/scripts/fit_lam1p0_blocked_action_relative_entropy.py"
DEFAULT_OUT = ROOT / "perfect_blocking/perfect_blocking_lam1p0/tests/archive_superseded_kernel_explorations_20260818/softcond7_blocked_action_relative_entropy"


def fit_module():
    spec = importlib.util.spec_from_file_location("blocked_action_fit", FIT_SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def objective(z_direct: np.ndarray, z_blocked: np.ndarray, alpha: np.ndarray) -> float:
    return float(logsumexp(z_direct @ alpha) - np.log(len(z_direct)) - z_blocked.mean(axis=0) @ alpha)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--base-extra", default="", help="comma-separated already accepted extension operators")
    ap.add_argument("--candidates", default="", help="comma-separated candidate operators; defaults to the general extension bank")
    ap.add_argument("--label", default="", help="suffix for the output CSV")
    args = ap.parse_args()
    out = args.out if args.out.is_absolute() else ROOT / args.out
    summary = json.loads((out / "summary.json").read_text())
    m = fit_module()
    extras = [name for name in args.base_extra.split(",") if name]
    allowed = set(m.EXTENSION_CANDIDATES) | set(m.HIGH_FIELD_DIAGONAL_CANDIDATES) | set(m.MULTISITE_CANDIDATES)
    unknown = set(extras) - allowed
    if unknown:
        raise ValueError(f"unknown extension operators: {sorted(unknown)}")
    names0 = list(m.ACTION_BASIS) + extras
    rng = np.random.default_rng(20260809)
    n = int(summary["n_train"] + summary["n_validation"] + summary["n_test"])
    dp = m.load_configs(Path(summary["direct"])); fp = m.load_configs(Path(summary["fine"]))
    dp = dp[rng.permutation(len(dp))[:n]]; fp = fp[rng.permutation(len(fp))[:n]]
    bp = m.block_configs(fp, m.load_kernel(Path(summary["kernel"])))
    d, b = m.local_action_features(dp), m.local_action_features(bp)
    train = np.arange(int(summary["n_train"])); val = np.arange(int(summary["n_train"]), int(summary["n_train"] + summary["n_validation"])); test = np.arange(int(summary["n_train"] + summary["n_validation"]), n)

    def evaluate(names: list[str], source: np.ndarray, target: np.ndarray, evaluation: np.ndarray) -> tuple[float, float]:
        x, y = np.column_stack([d[k] for k in names]), np.column_stack([b[k] for k in names])
        combined = np.r_[x[source], y[source]]; center = combined.mean(0); scale = np.maximum(combined.std(0, ddof=1), 1e-12)
        z, t = (x - center) / scale, (y - center) / scale
        alpha = m.fit_tilt(z[source], t[source].mean(0), float(summary["ridge"]))
        w, _ = m.normalized_weights(z[evaluation], alpha)
        return objective(z[evaluation], t[evaluation], alpha), m.effective_sample_size(w) / len(evaluation)

    rows = []
    candidates = [name for name in args.candidates.split(",") if name] if args.candidates else list(m.EXTENSION_CANDIDATES)
    unknown_candidates = set(candidates) - allowed
    if unknown_candidates:
        raise ValueError(f"unknown candidate operators: {sorted(unknown_candidates)}")
    models = [("base", names0)] + [(candidate, names0 + [candidate]) for candidate in candidates if candidate not in extras]
    for label, names in models:
        validation, _ = evaluate(names, train, None, val)
        test_score, test_ess = evaluate(names, np.r_[train, val], None, test)
        rows.append({"added_operator": label, "n_terms": len(names), "validation_objective": validation,
                     "heldout_test_objective": test_score, "heldout_ess_fraction": test_ess})
    table = pd.DataFrame(rows)
    base = table.iloc[0]
    table["validation_improvement_over_base"] = base.validation_objective - table.validation_objective
    table["heldout_test_improvement_over_base"] = base.heldout_test_objective - table.heldout_test_objective
    suffix = f"_{args.label}" if args.label else ""
    table.sort_values("validation_improvement_over_base", ascending=False).to_csv(out / f"relative_entropy_extension_screen{suffix}.csv", index=False)
    print(table.sort_values("validation_improvement_over_base", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
