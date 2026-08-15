# Finite-lambda kernel KL optimization: D4 5x5, L32 -> L16

This scan was performed for L_f = 32 -> L_c = 16.

## Setup
- test_name: `finite_lambda_kernel5x5_KL_etafit_L32_to_L16`
- volume_pair: `32 -> 16`
- L_f: `32`
- L_c: `16`
- lambda: `1.0`
- kappa: `0.34008`
- eta: exponent, tunable in this diagnostic and finally selected as `0.25`
- b: `2`
- spatial shell normalization: `w00 + 4*w10 + 4*w11 + 4*w20 + 8*w21 + 4*w22 = 1` before field rescaling
- block_norm: actual field-normalization factor, `block_norm = b^(eta/2) = 2^(eta/2)`
- block_norm_numeric at eta=0.25: `2^0.125 = 1.0905077326652577`
- K: normalized kernel shape with `sum_x K(x) = 1`
- B: full blocking operator, `B = block_norm * K`
- blocking rule: `psi_X = block_norm * [w00*phi_0 + w10*shell10 + w11*shell11 + w20*shell20 + w21*shell21 + w22*shell22], with spatial shell sum normalized to 1 first`
- action convention: `S = sum_x[(1-2*lambda)phi^2 + lambda phi^4] - 2*kappa*sum_{x,mu} phi_x phi_{x+mu}`

## Primary objective
The primary objective minimized is covariance-aware operator KL proxy:

`D_op(w) = 0.5 * DeltaO(w)^T C_reg^{-1} DeltaO(w)`

Operators used in the D_op objective:
`m2, m4, NN, diag, 2nn, NN2, diag2, 2nn2, action_density`

## Covariance details
- bootstrap replicates for optimization: `16`
- bootstrap replicates for final reporting: `256`
- regularization default epsilon: `1e-3`
- sensitivity reported for `eps in {1e-6, 1e-4, 1e-3, 1e-2}`
A shrinkage term `eps * trace(C)/Nop * I` is added before inversion.

## Objective behavior
- full covariance and cross-correlations are used
- no diagonal-only z-score fit is used as the objective
- the older RMS-z scores are reported only for comparison

### Summary table

| test_name | volume_pair | L_f | L_c | kernel_label | eta | N_eta | w00 | w10 | w11 | w20 | w21 | w22 | spatial_norm | effective_sum | score_name | score_value |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| finite_lambda_kernel5x5_KL_etafit_L32_to_L16 | 32 -> 16 | 32 | 16 | direct_generated_L16_reference | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 1.000000 | 1.000000 | D_op | 0.000000 |
| finite_lambda_kernel5x5_KL_etafit_L32_to_L16 | 32 -> 16 | 32 | 16 | direct_generated_L16_reference | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 1.000000 | 1.000000 | ir_score | 0.000000 |
| finite_lambda_kernel5x5_KL_etafit_L32_to_L16 | 32 -> 16 | 32 | 16 | direct_generated_L16_reference | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 1.000000 | 1.000000 | expanded_local_score | 0.000000 |
| finite_lambda_kernel5x5_KL_etafit_L32_to_L16 | 32 -> 16 | 32 | 16 | direct_generated_L16_reference | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 1.000000 | 1.000000 | expanded_total_score | 0.000000 |
| finite_lambda_kernel5x5_KL_etafit_L32_to_L16 | 32 -> 16 | 32 | 16 | fixed_eta_old_5x5_KL | 0.25 | 1.0905077326652577 | 0.7870554480116678 | 0.02707804843064735 | 0.013960878712339241 | 0.014384144266942532 | 0.002391343089368392 | -0.006969619591582884 | 1.000000 | 1.090508 | D_op | 6.225374 |
| finite_lambda_kernel5x5_KL_etafit_L32_to_L16 | 32 -> 16 | 32 | 16 | fixed_eta_old_5x5_KL | 0.25 | 1.0905077326652577 | 0.7870554480116678 | 0.02707804843064735 | 0.013960878712339241 | 0.014384144266942532 | 0.002391343089368392 | -0.006969619591582884 | 1.000000 | 1.090508 | ir_score | 0.823394 |
| finite_lambda_kernel5x5_KL_etafit_L32_to_L16 | 32 -> 16 | 32 | 16 | fixed_eta_old_5x5_KL | 0.25 | 1.0905077326652577 | 0.7870554480116678 | 0.02707804843064735 | 0.013960878712339241 | 0.014384144266942532 | 0.002391343089368392 | -0.006969619591582884 | 1.000000 | 1.090508 | expanded_local_score | 2.237887 |
| finite_lambda_kernel5x5_KL_etafit_L32_to_L16 | 32 -> 16 | 32 | 16 | fixed_eta_old_5x5_KL | 0.25 | 1.0905077326652577 | 0.7870554480116678 | 0.02707804843064735 | 0.013960878712339241 | 0.014384144266942532 | 0.002391343089368392 | -0.006969619591582884 | 1.000000 | 1.090508 | expanded_total_score | 2.025384 |
| finite_lambda_kernel5x5_KL_etafit_L32_to_L16 | 32 -> 16 | 32 | 16 | best_eta_tunable_5x5_KL | 0.25 | 1.0905077326652577 | 0.7870554480116678 | 0.02707804843064735 | 0.013960878712339241 | 0.014384144266942532 | 0.002391343089368392 | -0.006969619591582884 | 1.000000 | 1.090508 | D_op | 6.225374 |
| finite_lambda_kernel5x5_KL_etafit_L32_to_L16 | 32 -> 16 | 32 | 16 | best_eta_tunable_5x5_KL | 0.25 | 1.0905077326652577 | 0.7870554480116678 | 0.02707804843064735 | 0.013960878712339241 | 0.014384144266942532 | 0.002391343089368392 | -0.006969619591582884 | 1.000000 | 1.090508 | ir_score | 0.823394 |
| finite_lambda_kernel5x5_KL_etafit_L32_to_L16 | 32 -> 16 | 32 | 16 | best_eta_tunable_5x5_KL | 0.25 | 1.0905077326652577 | 0.7870554480116678 | 0.02707804843064735 | 0.013960878712339241 | 0.014384144266942532 | 0.002391343089368392 | -0.006969619591582884 | 1.000000 | 1.090508 | expanded_local_score | 2.237887 |
| finite_lambda_kernel5x5_KL_etafit_L32_to_L16 | 32 -> 16 | 32 | 16 | best_eta_tunable_5x5_KL | 0.25 | 1.0905077326652577 | 0.7870554480116678 | 0.02707804843064735 | 0.013960878712339241 | 0.014384144266942532 | 0.002391343089368392 | -0.006969619591582884 | 1.000000 | 1.090508 | expanded_total_score | 2.025384 |
| finite_lambda_kernel5x5_KL_etafit_L32_to_L16 | 32 -> 16 | 32 | 16 | best_IR_safe_eta_tunable_5x5_KL | 0.25 | 1.0905077326652577 | 0.7870554480116678 | 0.02707804843064735 | 0.013960878712339241 | 0.014384144266942532 | 0.002391343089368392 | -0.006969619591582884 | 1.000000 | 1.090508 | D_op | 6.225374 |
| finite_lambda_kernel5x5_KL_etafit_L32_to_L16 | 32 -> 16 | 32 | 16 | best_IR_safe_eta_tunable_5x5_KL | 0.25 | 1.0905077326652577 | 0.7870554480116678 | 0.02707804843064735 | 0.013960878712339241 | 0.014384144266942532 | 0.002391343089368392 | -0.006969619591582884 | 1.000000 | 1.090508 | ir_score | 0.823394 |
| finite_lambda_kernel5x5_KL_etafit_L32_to_L16 | 32 -> 16 | 32 | 16 | best_IR_safe_eta_tunable_5x5_KL | 0.25 | 1.0905077326652577 | 0.7870554480116678 | 0.02707804843064735 | 0.013960878712339241 | 0.014384144266942532 | 0.002391343089368392 | -0.006969619591582884 | 1.000000 | 1.090508 | expanded_local_score | 2.237887 |
| finite_lambda_kernel5x5_KL_etafit_L32_to_L16 | 32 -> 16 | 32 | 16 | best_IR_safe_eta_tunable_5x5_KL | 0.25 | 1.0905077326652577 | 0.7870554480116678 | 0.02707804843064735 | 0.013960878712339241 | 0.014384144266942532 | 0.002391343089368392 | -0.006969619591582884 | 1.000000 | 1.090508 | expanded_total_score | 2.025384 |
| finite_lambda_kernel5x5_KL_etafit_L32_to_L16 | 32 -> 16 | 32 | 16 | best_eta_only_old_spatial_5x5_KL | 0.25 | 1.0905077326652577 | 0.7870554480116678 | 0.02707804843064735 | 0.013960878712339241 | 0.014384144266942532 | 0.002391343089368392 | -0.006969619591582884 | 1.000000 | 1.090508 | D_op | 6.225374 |
| finite_lambda_kernel5x5_KL_etafit_L32_to_L16 | 32 -> 16 | 32 | 16 | best_eta_only_old_spatial_5x5_KL | 0.25 | 1.0905077326652577 | 0.7870554480116678 | 0.02707804843064735 | 0.013960878712339241 | 0.014384144266942532 | 0.002391343089368392 | -0.006969619591582884 | 1.000000 | 1.090508 | ir_score | 0.823394 |
| finite_lambda_kernel5x5_KL_etafit_L32_to_L16 | 32 -> 16 | 32 | 16 | best_eta_only_old_spatial_5x5_KL | 0.25 | 1.0905077326652577 | 0.7870554480116678 | 0.02707804843064735 | 0.013960878712339241 | 0.014384144266942532 | 0.002391343089368392 | -0.006969619591582884 | 1.000000 | 1.090508 | expanded_local_score | 2.237887 |
| finite_lambda_kernel5x5_KL_etafit_L32_to_L16 | 32 -> 16 | 32 | 16 | best_eta_only_old_spatial_5x5_KL | 0.25 | 1.0905077326652577 | 0.7870554480116678 | 0.02707804843064735 | 0.013960878712339241 | 0.014384144266942532 | 0.002391343089368392 | -0.006969619591582884 | 1.000000 | 1.090508 | expanded_total_score | 2.025384 |
| finite_lambda_kernel5x5_KL_etafit_L32_to_L16 | 32 -> 16 | 32 | 16 | best_spatial_only_eta0p25_5x5_KL | 0.25 | 1.0905077326652577 | 0.7833496919729274 | 0.027986023500293004 | 0.014144504958094738 | 0.01451170061242356 | 0.002295030848553372 | -0.007069713761149906 | 1.000000 | 1.090508 | D_op | 7.293097 |
| finite_lambda_kernel5x5_KL_etafit_L32_to_L16 | 32 -> 16 | 32 | 16 | best_spatial_only_eta0p25_5x5_KL | 0.25 | 1.0905077326652577 | 0.7833496919729274 | 0.027986023500293004 | 0.014144504958094738 | 0.01451170061242356 | 0.002295030848553372 | -0.007069713761149906 | 1.000000 | 1.090508 | ir_score | 0.825790 |
| finite_lambda_kernel5x5_KL_etafit_L32_to_L16 | 32 -> 16 | 32 | 16 | best_spatial_only_eta0p25_5x5_KL | 0.25 | 1.0905077326652577 | 0.7833496919729274 | 0.027986023500293004 | 0.014144504958094738 | 0.01451170061242356 | 0.002295030848553372 | -0.007069713761149906 | 1.000000 | 1.090508 | expanded_local_score | 2.295138 |
| finite_lambda_kernel5x5_KL_etafit_L32_to_L16 | 32 -> 16 | 32 | 16 | best_spatial_only_eta0p25_5x5_KL | 0.25 | 1.0905077326652577 | 0.7833496919729274 | 0.027986023500293004 | 0.014144504958094738 | 0.01451170061242356 | 0.002295030848553372 | -0.007069713761149906 | 1.000000 | 1.090508 | expanded_total_score | 2.069406 |

### Optimization method
- continuous optimizer: `scipy.optimize.minimize`
- method: `Powell`
- variables: `w10, w11, w20, w21, w22, eta` with `w00` determined by spatial normalization
- bounds: `eta in [-0.2, 0.8]`, `w10 in [-0.25, 0.35]`, `w11 in [-0.25, 0.25]`, `w20 in [-0.20, 0.25]`, `w21 in [-0.15, 0.15]`, `w22 in [-0.15, 0.15]`
- starting points include the old 5x5 KL kernel at several eta values, smooth positive kernels, and random starts

### Operator comparison for direct reference

| operator | direct L16 value | direct L16 error | blocked L32->L16 value | blocked error | difference | z-score | included_in_objective |
|---|---:|---:|---:|---:|---:|---:|---|
| m2 | 0.407945 | 0.009728 | 0.407945 | 0.009728 | +0.000000 | +0.000 | True |
| m4 | 0.191969 | 0.007664 | 0.191969 | 0.007664 | +0.000000 | +0.000 | True |
| NN | 0.566911 | 0.005223 | 0.566911 | 0.005223 | +0.000000 | +0.000 | True |
| diag | 0.509364 | 0.006094 | 0.509364 | 0.006094 | +0.000000 | +0.000 | True |
| 2nn | 0.478072 | 0.006800 | 0.478072 | 0.006800 | +0.000000 | +0.000 | True |
| NN2 | 0.328773 | 0.005928 | 0.328773 | 0.005928 | +0.000000 | +0.000 | True |
| diag2 | 0.269456 | 0.006203 | 0.269456 | 0.006203 | +0.000000 | +0.000 | True |
| 2nn2 | 0.241139 | 0.006525 | 0.241139 | 0.006525 | +0.000000 | +0.000 | True |
| action_density | -0.545915 | 0.004962 | -0.545915 | 0.004962 | +0.000000 | +0.000 | True |

### Operator comparison for selected best eta-tunable kernel

The following operators were used in the optimization objective:
`m2, m4, NN, diag, 2nn, NN2, diag2, 2nn2, action_density`

| operator | direct L16 value | direct L16 error | blocked L32->L16 value | blocked error | difference | covariance-normalized contribution | z-score | included_in_objective |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| m2 | 0.407945 | 0.009728 | 0.438383 | 0.010613 | +0.030438 | -12.399422 | +2.114 | True |
| m4 | 0.191969 | 0.007664 | 0.221103 | 0.008477 | +0.029133 | +24.828591 | +2.549 | True |
| NN | 0.566911 | 0.005223 | 0.583140 | 0.005480 | +0.016229 | +nan | +2.144 | True |
| diag | 0.509364 | 0.006094 | 0.529081 | 0.006345 | +0.019717 | +6.542916 | +2.241 | True |
| 2nn | 0.478072 | 0.006800 | 0.499317 | 0.007245 | +0.021245 | +7.726921 | +2.138 | True |
| NN2 | 0.328773 | 0.005928 | 0.347449 | 0.006418 | +0.018676 | +nan | +2.138 | True |
| diag2 | 0.269456 | 0.006203 | 0.290009 | 0.006676 | +0.020553 | -6.669575 | +2.255 | True |
| 2nn2 | 0.241139 | 0.006525 | 0.262275 | 0.007173 | +0.021136 | -17.767339 | +2.180 | True |
| action_density | -0.545915 | 0.004962 | -0.562910 | 0.005275 | -0.016995 | +1.333050 | -2.347 | True |
| |m| | 0.621451 | 0.008970 | 0.643313 | 0.009699 | +0.021862 | +nan | +1.655 | False |
| susceptibility | 104.396584 | 2.600279 | 112.135612 | 2.762794 | +7.739029 | +nan | +2.040 | False |
| Binder | 0.615490 | 0.005442 | 0.616500 | 0.005687 | +0.001009 | +nan | +0.128 | False |
| xi/L | 0.861749 | 0.037253 | 0.932748 | 0.048737 | +0.070998 | +nan | +1.157 | False |

### Operator comparison for fixed-eta old 5x5 KL kernel

The following operators were used in the optimization objective:
`m2, m4, NN, diag, 2nn, NN2, diag2, 2nn2, action_density`

| operator | direct L16 value | direct L16 error | blocked L32->L16 value | blocked error | difference | covariance-normalized contribution | z-score | included_in_objective |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| m2 | 0.407945 | 0.009728 | 0.438383 | 0.010613 | +0.030438 | -12.399422 | +2.114 | True |
| m4 | 0.191969 | 0.007664 | 0.221103 | 0.008477 | +0.029133 | +24.828591 | +2.549 | True |
| NN | 0.566911 | 0.005223 | 0.583140 | 0.005480 | +0.016229 | +nan | +2.144 | True |
| diag | 0.509364 | 0.006094 | 0.529081 | 0.006345 | +0.019717 | +6.542916 | +2.241 | True |
| 2nn | 0.478072 | 0.006800 | 0.499317 | 0.007245 | +0.021245 | +7.726921 | +2.138 | True |
| NN2 | 0.328773 | 0.005928 | 0.347449 | 0.006418 | +0.018676 | +nan | +2.138 | True |
| diag2 | 0.269456 | 0.006203 | 0.290009 | 0.006676 | +0.020553 | -6.669575 | +2.255 | True |
| 2nn2 | 0.241139 | 0.006525 | 0.262275 | 0.007173 | +0.021136 | -17.767339 | +2.180 | True |
| action_density | -0.545915 | 0.004962 | -0.562910 | 0.005275 | -0.016995 | +1.333050 | -2.347 | True |
| |m| | 0.621451 | 0.008970 | 0.643313 | 0.009699 | +0.021862 | +nan | +1.655 | False |
| susceptibility | 104.396584 | 2.600279 | 112.135612 | 2.762794 | +7.739029 | +nan | +2.040 | False |
| Binder | 0.615490 | 0.005442 | 0.616500 | 0.005687 | +0.001009 | +nan | +0.128 | False |
| xi/L | 0.861749 | 0.037253 | 0.932748 | 0.048737 | +0.070998 | +nan | +1.157 | False |

### Operator comparison for best eta-only old-spatial kernel

The following operators were used in the optimization objective:
`m2, m4, NN, diag, 2nn, NN2, diag2, 2nn2, action_density`

| operator | direct L16 value | direct L16 error | blocked L32->L16 value | blocked error | difference | covariance-normalized contribution | z-score | included_in_objective |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| m2 | 0.407945 | 0.009728 | 0.438383 | 0.010613 | +0.030438 | -12.399422 | +2.114 | True |
| m4 | 0.191969 | 0.007664 | 0.221103 | 0.008477 | +0.029133 | +24.828591 | +2.549 | True |
| NN | 0.566911 | 0.005223 | 0.583140 | 0.005480 | +0.016229 | +nan | +2.144 | True |
| diag | 0.509364 | 0.006094 | 0.529081 | 0.006345 | +0.019717 | +6.542916 | +2.241 | True |
| 2nn | 0.478072 | 0.006800 | 0.499317 | 0.007245 | +0.021245 | +7.726921 | +2.138 | True |
| NN2 | 0.328773 | 0.005928 | 0.347449 | 0.006418 | +0.018676 | +nan | +2.138 | True |
| diag2 | 0.269456 | 0.006203 | 0.290009 | 0.006676 | +0.020553 | -6.669575 | +2.255 | True |
| 2nn2 | 0.241139 | 0.006525 | 0.262275 | 0.007173 | +0.021136 | -17.767339 | +2.180 | True |
| action_density | -0.545915 | 0.004962 | -0.562910 | 0.005275 | -0.016995 | +1.333050 | -2.347 | True |
| |m| | 0.621451 | 0.008970 | 0.643313 | 0.009699 | +0.021862 | +nan | +1.655 | False |
| susceptibility | 104.396584 | 2.600279 | 112.135612 | 2.762794 | +7.739029 | +nan | +2.040 | False |
| Binder | 0.615490 | 0.005442 | 0.616500 | 0.005687 | +0.001009 | +nan | +0.128 | False |
| xi/L | 0.861749 | 0.037253 | 0.932748 | 0.048737 | +0.070998 | +nan | +1.157 | False |

### Operator comparison for best spatial-only eta0p25 kernel

The following operators were used in the optimization objective:
`m2, m4, NN, diag, 2nn, NN2, diag2, 2nn2, action_density`

| operator | direct L16 value | direct L16 error | blocked L32->L16 value | blocked error | difference | covariance-normalized contribution | z-score | included_in_objective |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| m2 | 0.407945 | 0.009728 | 0.438378 | 0.010612 | +0.030433 | -12.164816 | +2.114 | True |
| m4 | 0.191969 | 0.007664 | 0.221091 | 0.008475 | +0.029121 | +23.765287 | +2.549 | True |
| NN | 0.566911 | 0.005223 | 0.583240 | 0.005476 | +0.016329 | +nan | +2.158 | True |
| diag | 0.509364 | 0.006094 | 0.529052 | 0.006341 | +0.019688 | +6.183407 | +2.239 | True |
| 2nn | 0.478072 | 0.006800 | 0.499333 | 0.007243 | +0.021260 | +7.111958 | +2.140 | True |
| NN2 | 0.328773 | 0.005928 | 0.347553 | 0.006414 | +0.018780 | +nan | +2.150 | True |
| diag2 | 0.269456 | 0.006203 | 0.289966 | 0.006673 | +0.020510 | -7.515582 | +2.251 | True |
| 2nn2 | 0.241139 | 0.006525 | 0.262281 | 0.007171 | +0.021143 | -16.878964 | +2.181 | True |
| action_density | -0.545915 | 0.004962 | -0.566069 | 0.005270 | -0.020153 | +5.044818 | -2.784 | True |
| |m| | 0.621451 | 0.008970 | 0.643314 | 0.009698 | +0.021863 | +nan | +1.655 | False |
| susceptibility | 104.396584 | 2.600279 | 112.134506 | 2.762457 | +7.737923 | +nan | +2.040 | False |
| Binder | 0.615490 | 0.005442 | 0.616513 | 0.005686 | +0.001022 | +nan | +0.130 | False |
| xi/L | 0.861749 | 0.037253 | 0.932959 | 0.048752 | +0.071210 | +nan | +1.161 | False |

## Selected kernels

| kernel_label | volume_pair | L_f | L_c | w00 | w10 | w11 | w20 | w21 | w22 | normalization_check | D_op | ir_score | expanded_local_score | expanded_total_score | old_expanded_local_score | old_expanded_total_score |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| direct_generated_L16_reference | 32 -> 16 | 32 | 16 | n/a | n/a | n/a | n/a | n/a | n/a | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| fixed_eta_old_5x5_KL | 32 -> 16 | 32 | 16 | 0.7870554480116678 | 0.02707804843064735 | 0.013960878712339241 | 0.014384144266942532 | 0.002391343089368392 | -0.006969619591582884 | 1.000000 | 6.225374 | 0.823394 | 2.237887 | 2.025384 | 2.237887 | 2.025384 |
| best_eta_tunable_5x5_KL | 32 -> 16 | 32 | 16 | 0.7870554480116678 | 0.02707804843064735 | 0.013960878712339241 | 0.014384144266942532 | 0.002391343089368392 | -0.006969619591582884 | 1.000000 | 6.225374 | 0.823394 | 2.237887 | 2.025384 | 2.237887 | 2.025384 |
| best_IR_safe_eta_tunable_5x5_KL | 32 -> 16 | 32 | 16 | 0.7870554480116678 | 0.02707804843064735 | 0.013960878712339241 | 0.014384144266942532 | 0.002391343089368392 | -0.006969619591582884 | 1.000000 | 6.225374 | 0.823394 | 2.237887 | 2.025384 | 2.237887 | 2.025384 |
| best_eta_only_old_spatial_5x5_KL | 32 -> 16 | 32 | 16 | 0.7870554480116678 | 0.02707804843064735 | 0.013960878712339241 | 0.014384144266942532 | 0.002391343089368392 | -0.006969619591582884 | 1.000000 | 6.225374 | 0.823394 | 2.237887 | 2.025384 | 2.237887 | 2.025384 |
| best_spatial_only_eta0p25_5x5_KL | 32 -> 16 | 32 | 16 | 0.7833496919729274 | 0.027986023500293004 | 0.014144504958094738 | 0.01451170061242356 | 0.002295030848553372 | -0.007069713761149906 | 1.000000 | 7.293097 | 0.825790 | 2.295138 | 2.069406 | 2.295138 | 2.069406 |

## Top 10 candidate kernels

| rank | kernel_label | volume_pair | L_f | L_c | w00 | w10 | w11 | w20 | w21 | w22 | D_op | ir_score | expanded_local_score | expanded_total_score |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | old_5x5_eta0p25 | 32 -> 16 | 32 | 16 | 0.783224 | 0.027907 | 0.014067 | 0.014606 | 0.002304 | -0.006993 | 8.930752 | 0.000000 | 0.000000 | 0.000000 |
| 2 | seed_old_5x5_eta0p25 | 32 -> 16 | 32 | 16 | 0.787055 | 0.027078 | 0.013961 | 0.014384 | 0.002391 | -0.006970 | 9.573209 | 0.000000 | 0.000000 | 0.000000 |
| 3 | compact_positive_eta0.25 | 32 -> 16 | 32 | 16 | 0.787817 | 0.020288 | 0.020547 | 0.011346 | 0.001173 | -0.001481 | 30.060027 | 0.000000 | 0.000000 | 0.000000 |
| 4 | smooth_positive_eta0.25 | 32 -> 16 | 32 | 16 | 0.797156 | 0.016549 | 0.014401 | 0.014490 | 0.002943 | -0.000615 | 35.762277 | 0.000000 | 0.000000 | 0.000000 |
| 5 | smooth_positive_eta0.375 | 32 -> 16 | 32 | 16 | 0.732592 | 0.030816 | 0.014814 | 0.010273 | 0.004973 | 0.001003 | 48.231844 | 0.000000 | 0.000000 | 0.000000 |
| 6 | compact_positive_eta0.375 | 32 -> 16 | 32 | 16 | 0.715441 | 0.038206 | 0.018445 | 0.009114 | 0.002412 | 0.000551 | 58.087031 | 0.000000 | 0.000000 | 0.000000 |
| 7 | random_3 | 32 -> 16 | 32 | 16 | 0.692122 | 0.098482 | -0.028281 | -0.036546 | 0.032316 | -0.021319 | 71.973621 | 0.000000 | 0.000000 | 0.000000 |
| 8 | random_4 | 32 -> 16 | 32 | 16 | 0.716163 | -0.062782 | 0.072178 | 0.000210 | 0.037968 | -0.014582 | 83.248331 | 0.000000 | 0.000000 | 0.000000 |
| 9 | old_5x5_eta0.125 | 32 -> 16 | 32 | 16 | 0.875761 | 0.007552 | 0.012414 | 0.019095 | 0.000713 | -0.009427 | 108.836262 | 0.000000 | 0.000000 | 0.000000 |
| 10 | random_2 | 32 -> 16 | 32 | 16 | 0.624331 | 0.093008 | 0.006270 | -0.024842 | 0.007516 | 0.004450 | 117.981104 | 0.000000 | 0.000000 | 0.000000 |

## Covariance diagnostics
- condition number of C at best D_op kernel: `74148.4`
- condition number of C_reg at best D_op kernel: `7642.2`
- smallest/largest eigenvalues of C at best D_op kernel: `1.23877e-08`, `0.000918529`
- smallest/largest eigenvalues of C_reg at best D_op kernel: `1.20206e-07`, `0.000918637`
- covariance eigenvalues (C) at best D_op kernel: `1.239e-08, 3.855e-08, 5.385e-07, 2.209e-06, 2.925e-06, 6.706e-06, 8.88e-06, 3.052e-05, 0.0009185`
- covariance eigenvalues (C_reg) at best D_op kernel: `1.202e-07, 1.464e-07, 6.463e-07, 2.317e-06, 3.033e-06, 6.814e-06, 8.987e-06, 3.063e-05, 0.0009186`
- IR-unsafe flag at best D_op kernel: `False`

## Epsilon sensitivity

| eps | w00 | w10 | w11 | w20 | w21 | w22 | D_op | ir_score | expanded_local_score | expanded_total_score | IR unsafe |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1e-06 | 0.783224 | 0.027907 | 0.014067 | 0.014606 | 0.002304 | -0.006993 | 21.400595 | 0.000000 | 0.000000 | 0.000000 | False |
| 1e-04 | 0.783224 | 0.027907 | 0.014067 | 0.014606 | 0.002304 | -0.006993 | 14.203201 | 0.000000 | 0.000000 | 0.000000 | False |
| 1e-03 | 0.783224 | 0.027907 | 0.014067 | 0.014606 | 0.002304 | -0.006993 | 8.930752 | 0.000000 | 0.000000 | 0.000000 | False |
| 1e-02 | 0.787055 | 0.027078 | 0.013961 | 0.014384 | 0.002391 | -0.006970 | 6.725596 | 0.000000 | 0.000000 | 0.000000 | False |

## Interpretation

The primary objective is a covariance-aware operator KL proxy built from bootstrap mean differences and the full regularized covariance matrix.
The older RMS-z scores are included only as diagnostics.
This optimization reuses the saved L32 and L16 ensembles and does not regenerate Monte Carlo data.

Eta is the exponent, not the multiplicative blocking normalization. Eta was included as a continuous optimizer parameter. The final selected exponent was `eta=0.25`, giving `block_norm = 2^(eta/2) = 2^0.125`. This happened because common 256-bootstrap rescoring selected the old fixed-eta kernel. Optimization-stage candidates could move eta slightly, e.g. `eta≈0.24976`. The available fixed-kernel eta scan has a sharp minimum at `eta=0.25`, but no fine scan around `eta=0.25` was performed.
