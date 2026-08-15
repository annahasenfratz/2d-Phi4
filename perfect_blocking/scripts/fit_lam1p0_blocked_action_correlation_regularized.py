#!/usr/bin/env python3
"""Fit S_b-S_c with relative entropy plus fixed-scale correlation penalties.

The correlation scales are measured once on the training ensembles and then
frozen.  This prevents the optimizer from reducing its loss by broadening the
reweighted distribution.  The loss remains a local-action ansatz, but it is
no longer a pure KL projection.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
FIT_SCRIPT = ROOT / "perfect_blocking/scripts/fit_lam1p0_blocked_action_relative_entropy.py"
SOURCE_DEFAULT = ROOT / "perfect_blocking/perfect_blocking_lam1p0/tests/softcond7_blocked_action_relative_entropy_highfield15"
OUT_DEFAULT = ROOT / "perfect_blocking/perfect_blocking_lam1p0/tests/softcond7_blocked_action_correlation_regularized"

# Selected from the significant held-out residual table, rather than all 105
# pairs.  Each entry is an explicit, physically interpretable target.
CORRELATION_PAIRS = [
    ("phi6", "diag"), ("phi6", "diag_phi2_neighbor"),
    ("phi6", "diag_phi2_bond_sq"), ("phi6", "diag_phi3phi3"),
    ("phi6", "NN"), ("phi6", "2nn"), ("phi6", "3nn"),
    ("phi6", "phi2_neighbor"), ("phi4", "diag"),
    ("phi4", "diag_phi2_neighbor"), ("phi4", "diag_phi3phi3"),
    ("NN", "phi2_neighbor"),
]


def module():
    spec = importlib.util.spec_from_file_location("blocked_action_fit", FIT_SCRIPT)
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def bootstrap_rho_se(x: np.ndarray, y: np.ndarray, rng: np.random.Generator, n_boot: int) -> float:
    take = rng.integers(len(x), size=(n_boot, len(x)))
    vals = np.array([np.corrcoef(x[t], y[t])[0, 1] for t in take])
    return float(max(vals.std(ddof=1), 1e-4))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=SOURCE_DEFAULT, help="completed highfield15 fit directory")
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--correlation-weight", type=float, default=0.01)
    ap.add_argument("--min-ess-fraction", type=float, default=0.12)
    ap.add_argument("--ess-penalty", type=float, default=50.0)
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--learning-rate", type=float, default=0.02)
    ap.add_argument("--n-bootstrap-scale", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260811)
    args = ap.parse_args()
    source = args.source if args.source.is_absolute() else ROOT / args.source
    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    summary = json.loads((source / "summary.json").read_text())
    coeff = pd.read_csv(source / "blocked_action_coefficients.csv")
    m = module(); names = list(summary["action_basis"])
    alpha0 = np.asarray([dict(zip(coeff.operator, coeff.standardized_log_ratio_coefficient))[name] for name in names], np.float64)
    rng = np.random.default_rng(args.seed)
    n = int(summary["n_train"] + summary["n_validation"] + summary["n_test"])
    direct_phi = m.load_configs(Path(summary["direct"])); fine_phi = m.load_configs(Path(summary["fine"]))
    sample_rng = np.random.default_rng(20260809)
    direct_phi = direct_phi[sample_rng.permutation(len(direct_phi))[:n]]
    fine_phi = fine_phi[sample_rng.permutation(len(fine_phi))[:n]]
    blocked_phi = m.block_configs(fine_phi, m.load_kernel(Path(summary["kernel"])))
    direct, blocked = m.local_action_features(direct_phi), m.local_action_features(blocked_phi)
    x, y = np.column_stack([direct[name] for name in names]), np.column_stack([blocked[name] for name in names])
    train = np.arange(int(summary["n_train"])); val = np.arange(int(summary["n_train"]), int(summary["n_train"] + summary["n_validation"])); test = np.arange(int(summary["n_train"] + summary["n_validation"]), n)
    joined = np.r_[x[train], y[train]]
    center, scale = joined.mean(0), np.maximum(joined.std(0, ddof=1), 1e-12)
    z, target = (x-center)/scale, (y-center)/scale
    pair_index = [(names.index(a), names.index(b)) for a, b in CORRELATION_PAIRS]
    target_rho = np.asarray([np.corrcoef(y[train,i], y[train,j])[0,1] for i,j in pair_index])
    # Fixed errors for a difference of two independent training ensembles.
    pair_scale = np.asarray([
        np.hypot(bootstrap_rho_se(x[train,i], x[train,j], rng, args.n_bootstrap_scale),
                 bootstrap_rho_se(y[train,i], y[train,j], rng, args.n_bootstrap_scale))
        for i,j in pair_index
    ])
    dtype = torch.float64
    z_train = torch.tensor(z[train], dtype=dtype); target_mean = torch.tensor(target[train].mean(0), dtype=dtype)
    target_rho_t = torch.tensor(target_rho, dtype=dtype); pair_scale_t = torch.tensor(pair_scale, dtype=dtype)
    alpha = torch.nn.Parameter(torch.tensor(alpha0, dtype=dtype))
    optimizer = torch.optim.Adam([alpha], lr=args.learning_rate)
    history = []
    ridge = float(summary["ridge"])
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        logw = z_train @ alpha; w = torch.softmax(logw, dim=0)
        base = torch.logsumexp(logw, dim=0) - np.log(len(train)) - alpha @ target_mean + .5 * ridge * (alpha @ alpha)
        mean = w @ z_train; centered = z_train - mean
        cov = centered.T @ (centered * w[:, None])
        rho = torch.stack([cov[i,j] / torch.sqrt(cov[i,i] * cov[j,j]) for i,j in pair_index])
        corr_chi2 = torch.mean(((rho-target_rho_t)/pair_scale_t)**2)
        ess_fraction = 1.0 / (len(train) * torch.sum(w*w))
        ess_guard = torch.relu(torch.tensor(args.min_ess_fraction, dtype=dtype) - ess_fraction)**2
        loss = base + args.correlation_weight * corr_chi2 + args.ess_penalty * ess_guard
        loss.backward(); optimizer.step()
        if epoch % 20 == 0 or epoch == args.epochs-1:
            history.append({"epoch": epoch, "loss": float(loss.detach()), "relative_entropy": float(base.detach()),
                            "correlation_chi2": float(corr_chi2.detach()), "ess_fraction": float(ess_fraction.detach())})
    pd.DataFrame(history).to_csv(out / "optimization_history.csv", index=False)
    alpha_np = alpha.detach().numpy()
    pd.DataFrame({"operator": names, "standardized_log_ratio_coefficient": alpha_np,
                  "deltaS_density_coefficient": -alpha_np/(scale*(direct_phi.shape[1]**2))}).to_csv(out / "correlation_regularized_coefficients.csv", index=False)

    def evaluate(indices: np.ndarray) -> tuple[float, float, list[dict[str,float]]]:
        zw = z[indices] @ alpha_np; ww = np.exp(zw - np.logaddexp.reduce(zw)); ww /= ww.sum()
        rows = []
        for (a,b),(i,j),rt,rs in zip(CORRELATION_PAIRS,pair_index,target_rho,pair_scale):
            rho_w = m.weighted_corr if False else float((ww @ ((z[indices,i]-ww@z[indices,i])*(z[indices,j]-ww@z[indices,j]))) / np.sqrt((ww @ (z[indices,i]-ww@z[indices,i])**2)*(ww @ (z[indices,j]-ww@z[indices,j])**2)))
            rho_b = float(np.corrcoef(target[indices,i], target[indices,j])[0,1])
            rows.append({"pair":f"{a} x {b}","reweighted_rho":rho_w,"blocked_rho":rho_b,"difference":rho_w-rho_b,"frozen_train_scale":float(rs)})
        return float(1/np.sum(ww*ww)/len(indices)), float(np.ptp(zw)), rows
    val_ess, val_span, val_rows = evaluate(val); test_ess, test_span, test_rows = evaluate(test)
    pd.DataFrame(val_rows).to_csv(out / "validation_correlations.csv", index=False)
    pd.DataFrame(test_rows).to_csv(out / "heldout_test_correlations.csv", index=False)
    result = {"source":str(source), "action_basis":names, "correlation_pairs":[f"{a} x {b}" for a,b in CORRELATION_PAIRS],
              "correlation_weight":args.correlation_weight,"min_ess_fraction":args.min_ess_fraction,
              "validation_ess_fraction":val_ess,"heldout_ess_fraction":test_ess,
              "validation_logweight_span":val_span,"heldout_logweight_span":test_span}
    (out / "summary.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
