# 2D phi4 kappa-lambda phase diagram notes

This directory collects the current finite-volume scans for the two-dimensional
phi4 model in the action convention

```text
S = sum_x [(1 - 2 lambda) phi_x^2 + lambda phi_x^4
           - 2 kappa sum_mu phi_x phi_{x+mu}]
```

The canonical data currently use `L = 16, 24, 32` and single-histogram
reweighting from the dated run folders.  The per-lambda reports include
susceptibilities, abs-centered susceptibilities, Binder cumulants, and ESS/N at
every refined kappa value.

## Current critical-kappa diagnostics

| lambda | linear Binder mean | linear Binder band | peak extrap. | chi exponent p | report |
| --- | --- | --- | --- | --- | --- |
| 1.0 | 0.339628 | 0.338497-0.340935 | 0.340296 | 1.725598 | [`lambda_1p0.md`](lambda_1p0.md) |
| 0.5 | 0.342570 | 0.341952-0.343278 | 0.342636 | 1.753487 | [`lambda_0p5.md`](lambda_0p5.md) |
| 0.1 | 0.302724 | 0.302465-0.303344 | 0.301586 | 1.462067 | [`lambda_0p1.md`](lambda_0p1.md) |
| 0.01 | 0.261115 | 0.260515-0.261565 | 0.261164 | 1.450868 | [`lambda_0p01.md`](lambda_0p01.md) |

## Reading the table

- `linear Binder mean` and `linear Binder band` summarize stored Binder
  crossings where the two Binder curves actually bracket a crossing.  Per-lambda
  reports still list closest-grid/no-sign-change diagnostics separately.
- `peak extrap.` is a linear extrapolation of susceptibility peak positions in
  `1/L`; it is a finite-volume diagnostic, not a precision infinite-volume fit.
- `chi exponent p` comes from `chi_abs,max ~ L^p`.  The 2D Ising value is
  `gamma/nu = 7/4 = 1.75`; lambda `1.0` and `0.5` are close, while lambda `0.1`
  and `0.01` are not yet clean.

## Plots

- [`plots/kappa_vs_lambda.svg`](plots/kappa_vs_lambda.svg)
- [`plots/kappa_vs_lambda.png`](plots/kappa_vs_lambda.png)
- [`plots/peak_exponent_vs_lambda.svg`](plots/peak_exponent_vs_lambda.svg)
- [`plots/peak_exponent_vs_lambda.png`](plots/peak_exponent_vs_lambda.png)

## Data products

Machine-readable refined tables are in [`data/`](data/).  The original dated
run folders are left in place so the provenance of each plot and summary remains
visible.
