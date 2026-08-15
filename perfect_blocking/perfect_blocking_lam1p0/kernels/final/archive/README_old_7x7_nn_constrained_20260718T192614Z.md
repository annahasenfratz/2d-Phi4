# Lambda 1.0 Final Kernel

Selected final kernel: NN-constrained 7x7 no-corner.

Final operational files:

```text
perfect_blocking/perfect_blocking_lam1p0/kernels/final/chosen_kernel.json
perfect_blocking/perfect_blocking_lam1p0/kernels/final/chosen_kernel.txt
```

Source candidate:

```text
perfect_blocking/perfect_blocking_lam1p0/kernels/candidates/7x7_no33_nn_constrained/best_7x7_no33_nn_constrained_eta_included.json
```

The previous provisional 5x5 final was archived under:

```text
perfect_blocking/perfect_blocking_lam1p0/kernels/final/archive/
```

## Parameters And Eta Convention

- `lambda`: 1.0
- `kappa_cr`: 0.340301
- `kappa_f`: 0.340301
- `kappa_c`: 0.340301
- `eta`: 0.25
- `block_factor`: 2
- `eta_scale`: `2^0.125 = 1.0905077326652577`
- `kernel_coefficients_include_eta_scale`: true
- `sum(K)`: 1.0905077326652577
- convention: apply as stored; do not multiply by `eta_scale` again

Final recorded kernels used for blocking and upsampling include the eta normalization in their coefficients. For `b=2` and `eta=0.25`, the final stored kernel sum is `2^(eta/2)=2^0.125`, not 1.

## 7x7 No-Corner Structure

The final kernel excludes the far diagonal corner class:

```text
K33 = 0
```

Outer-shell base coefficients before eta scale:

```text
K30 = -0.00036894189276879484
K31 =  0.0022330433962489153
K32 =  0.0016781689520450665
K33 =  0.0
```

## Momentum Stability

Dense-grid Fourier diagnostics:

```text
min K(p)   = 0.7512347298361362
max K(p)   = 1.0905077326652584
min 1/K(p) = 0.9170040432046707
max 1/K(p) = 1.3311418658960643
```

## Promotion Rationale

Compared with the retrained 5x5, the final 7x7 improves `phi2` and `phi4`, slightly improves local kurtosis, keeps action density and long-distance sectors acceptable, protects `NN` compared with the previous unconstrained 7x7, and maintains good momentum-space conditioning.

| metric | retrained 5x5 | previous 7x7 | final NN-constrained 7x7 |
|---|---:|---:|---:|
| action_density KS | 0.0300 | 0.0290 | 0.0316 |
| phi2 KS | 0.1014 | 0.0648 | 0.0804 |
| phi4 KS | 0.0420 | 0.0322 | 0.0136 |
| local_kurtosis_ratio KS | 0.2404 | 0.2242 | 0.2388 |
| NN KS | 0.0224 | 0.0498 | 0.0336 |
| 2nn KS | 0.0280 | 0.0274 | 0.0274 |
| diag KS | 0.0222 | 0.0206 | 0.0210 |
| m2 KS | 0.0168 | 0.0182 | 0.0182 |
| m4 KS | 0.0168 | 0.0182 | 0.0182 |
| G_pmin_avg KS | 0.0242 | 0.0270 | 0.0250 |
| max 1/K(p) | 1.3074 | 1.3820 | 1.3311 |

The supporting NN-constrained refinement record is:

```text
perfect_blocking/perfect_blocking_lam1p0/tests/intermediate/7x7_no33_nn_constrained/recommendation.md
```
