# Lambda 1.0 Kernels Selected For Upscaling Tests

Updated UTC: `20260718T192614Z`

This directory provides stable paths for lambda=1.0 upscaling tests. The files here are copies of retained kernels, not new training outputs.

All listed kernels are eta-included operational kernels:

```text
eta = 0.25
block_factor = 2
eta_scale = 2^0.125 = 1.0905077326652577
kernel_coefficients_include_eta_scale = true
apply as stored; do not multiply by eta_scale again
```

## Recommended Production Kernel

Use this for lambda=1.0 production blocking/upscaling tests:

```text
best_phi2_support_balanced_eta_included.json
```

Source:

```text
perfect_blocking/perfect_blocking_lam1p0/kernels/candidates/redo_phi2_support_balanced_2000/best_phi2_support_balanced_eta_included.json
```

Final confirmation:

```text
perfect_blocking/perfect_blocking_lam1p0/tests/final/final_confirm_phi2_support_balanced_kernel_full/summary.md
perfect_blocking/perfect_blocking_lam1p0/tests/final/final_confirm_phi2_support_balanced_kernel_L64_to_L32/summary.md
```

The older `best_7x7_phi2_nn_guarded_eta_included.json` is retained for history and comparison, but it is no longer the recommended production kernel.

## Confirmation Summary

| observable | previous selected 7x7 L32->L16 KS | support-balanced L32->L16 KS | previous selected 7x7 L64->L32 KS | support-balanced L64->L32 KS |
|---|---:|---:|---:|---:|
| action_density | 0.0308 | 0.0373 | 0.0444 | 0.0408 |
| phi2 | 0.0708 | 0.0384 | 0.1216 | 0.0510 |
| phi4 | 0.0206 | 0.0318 | 0.0220 | 0.0404 |
| local_kurtosis_ratio | 0.2195 | 0.0863 | 0.4298 | 0.1562 |
| NN | 0.0262 | 0.0195 | 0.0534 | 0.0360 |
| G_pmin_avg | 0.0174 | 0.0116 | 0.0196 | 0.0206 |

## Retained Comparison Kernels

Retained best 5x5 baseline:

```text
best_5x5_retrained_full_objective_eta_included.json
```

Previous selected/upscaling 7x7:

```text
best_7x7_phi2_nn_guarded_eta_included.json
```

Previous operational final at cleanup time:

```text
current_final_7x7_no33_nn_constrained_eta_included.json
```

The support-balanced 5x5 has been promoted to:

```text
perfect_blocking/perfect_blocking_lam1p0/kernels/final/chosen_kernel.json
```
