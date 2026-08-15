# Lambda 1.0 Final Kernel

Selected final kernel: phi2-support-balanced 5x5.

Final operational files:

```text
perfect_blocking/perfect_blocking_lam1p0/kernels/final/chosen_kernel.json
perfect_blocking/perfect_blocking_lam1p0/kernels/final/chosen_kernel.txt
```

Source candidate:

```text
perfect_blocking/perfect_blocking_lam1p0/kernels/candidates/redo_phi2_support_balanced_2000/best_phi2_support_balanced_eta_included.json
```

The previous NN-constrained 7x7 final was archived as:

```text
perfect_blocking/perfect_blocking_lam1p0/kernels/final/archive/chosen_kernel_old_7x7_nn_constrained_eta_included_20260718T192614Z.json
perfect_blocking/perfect_blocking_lam1p0/kernels/final/archive/chosen_kernel_old_7x7_nn_constrained_eta_included_20260718T192614Z.txt
perfect_blocking/perfect_blocking_lam1p0/kernels/final/archive/README_old_7x7_nn_constrained_20260718T192614Z.md
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

## 5x5 Support-Balanced Structure

Eta-included final matrix:

```text
 -1.016291e-03   2.452469e-02   3.469901e-02   2.452469e-02  -1.016291e-03
  2.452469e-02  -8.869673e-03  -2.908963e-02  -8.869673e-03   2.452469e-02
  3.469901e-02  -2.908963e-02   9.114260e-01  -2.908963e-02   3.469901e-02
  2.452469e-02  -8.869673e-03  -2.908963e-02  -8.869673e-03   2.452469e-02
 -1.016291e-03   2.452469e-02   3.469901e-02   2.452469e-02  -1.016291e-03
```

Base unit-sum orbit coefficients before eta scale:

```text
K00 =  0.8357777588446743
K10 = -0.026676226898291144
K11 = -0.008133489062896272
K20 =  0.03181943592579118
K21 =  0.022488893524959825
K22 = -0.0009319467256919966
```

## Momentum Stability

Dense-grid Fourier diagnostics:

```text
min K(p)   = 0.7073660206381788
max K(p)   = 1.0934526085021135
min 1/K(p) = 0.9145343769126572
max 1/K(p) = 1.413695273484878
```

## Promotion Rationale

The previous lambda=1.0 selected/upscaling kernel over-preserved `phi4` while leaving the `phi2` and `local_kurtosis_ratio` coarse marginal mismatched. The support-balanced 5x5 was selected because it substantially improves distributional support for `phi2` and `local_kurtosis_ratio`, keeps `NN` and `G_pmin_avg` controlled, and maintains acceptable momentum conditioning.

The tradeoff is a modest degradation in `phi4` and action-density sharpness, which was judged preferable to missing important regions of the blocked distribution.

Full confirmation records:

```text
perfect_blocking/perfect_blocking_lam1p0/tests/final/final_confirm_phi2_support_balanced_kernel_full/
perfect_blocking/perfect_blocking_lam1p0/tests/final/final_confirm_phi2_support_balanced_kernel_L64_to_L32/
```
