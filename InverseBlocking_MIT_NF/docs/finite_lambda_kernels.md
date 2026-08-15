# Finite-lambda perfect-blocking kernels

Recovered from the prior **Finite lambda blocking** chat for the lambda=1.0, near-critical kappa≈0.3401 setup with eta=1/4 and N_eta=2^(1/8).

Important precision convention: **only the unit normalization is exact**. The shell coefficients below are approximate working values recovered from earlier diagnostics. Do not quote extra digits or interpret them as fitted uncertainties.

## Preferred starting kernel

Use `kernels/finite_lambda_lam1_L32_to_L16_5x5_KL.json`. This is the 5x5 KL-optimized L32→L16 kernel shape that outperformed the 3x3 kernel in the later D_op/operator comparison.

Rounded shell parameters:

- w00 ≈ 0.787
- w10 ≈ 0.027
- w11 ≈ 0.014
- w20 ≈ 0.014
- w21 ≈ 0.0024
- w22 is not independently quoted; it is fixed by unit normalization.

Unit-sum convention:

```text
w00 + 4 w10 + 4 w11 + 4 w20 + 8 w21 + 4 w22 = 1
```

With the rounded values above this gives w22 ≈ -0.00655, but that number should be regarded as normalization bookkeeping, not a measured coefficient.

## Reference kernels

- `finite_lambda_lam1_L32_to_L16_3x3_KL.json`: w00≈0.71, w10≈0.096, with w11 fixed by normalization.
- `finite_lambda_lam1_L16_to_L8_3x3_best_total.json`: w00≈0.48, w10≈0.21, with w11 fixed by normalization.

For the MIT-NF inverse-blocking pilot, start from the 5x5 kernel and keep the 3x3 kernels only as diagnostics/fallbacks.
