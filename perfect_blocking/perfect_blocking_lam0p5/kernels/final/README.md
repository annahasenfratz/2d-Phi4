# Lambda 0.5 Final Kernel

Final kernel files:

```text
perfect_blocking/perfect_blocking_lam0p5/kernels/final/chosen_kernel.json
perfect_blocking/perfect_blocking_lam0p5/kernels/final/chosen_kernel.txt
```

The selected final 5x5 kernel is `refined_rand0084_0001_4_22_p`, promoted from:

```text
perfect_blocking_upsampling/outputs/lam0p5_5x5_kernel_search/kernel_candidates/best_lam0p5_5x5_kernel.json
```

It was selected because it had the lowest recorded score in the existing lambda=0.5 5x5 search and had stable conditioning and roundtrip diagnostics:

- score: 2.4254406070414483
- family: `coordinate_refined_5x5`
- full-BZ min abs K: 0.4787875523282582
- full-BZ condition number: 3.1514384422845176
- roundtrip phi max abs error: 2.38419e-07
- reblocking psi max abs error: 1.19209e-07

The source candidate metadata records kappa 0.3426. The promoted final kernel is used here for the current lambda=0.5 critical ensemble at kappa 0.343469.

## Eta Convention

The anomalous dimension/block normalization eta is part of the perfect blocking kernel convention. Final recorded kernels used for blocking and upsampling should include the eta normalization in their coefficients. For b=2 and eta=0.25 this factor is `eta_scale=2^(eta/2)=2^0.125`. Therefore the final stored kernel sum is `eta_scale`, not 1. Scripts must not multiply by `eta_scale` again when loading an eta-included final kernel.

During training/scanning, kernels may be represented as base normalized kernels with `sum(K)=1` plus an explicit `eta_scale`. When a kernel is promoted to final, write an eta-included copy to `kernels/final` and record both the base kernel and eta-included kernel metadata.

For this final kernel:

- `lambda`: 0.5
- `kappa_f`: 0.343469
- `kappa_c`: 0.343469
- `eta`: 0.25
- `block_factor`: 2
- `eta_scale`: `2^0.125`
- `eta_scale_numeric`: 1.0905077326652577
- `kernel_coefficients_include_eta_scale`: true
- `base_kernel_sum_before_eta_scale`: 1.0
- `final_kernel_sum_after_eta_scale`: 1.0905077326652577
- convention: `stored coefficients include eta_scale; do not multiply again on application`

The final operational orbit coefficients in `chosen_kernel.json` already include eta scale.
