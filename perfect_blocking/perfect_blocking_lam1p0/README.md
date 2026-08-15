# Perfect Blocking Lambda 1.0

This directory contains the lambda=1.0 perfect-blocking workflow using the shared scripts in `perfect_blocking/scripts/`.

Raw field configurations are not stored here. Native configurations remain under `data/configs_phi4_2d/`; generated configurations should go under `data/configs_phi4_2d/generated/`; blocked configurations are not saved by default and, if saved explicitly, should go under `data/configs_phi4_2d/blocked/`.

## Parameters

- `lambda`: 1.0
- `kappa_cr`: 0.340301
- `kappa_f`: 0.340301
- `kappa_c`: 0.340301
- `eta`: 0.25
- `block_factor`: 2
- `eta_scale`: `2^0.125 = 1.0905077326652577`
- final kernel family: phi2-support-balanced 5x5
- final kernel convention: eta-included; apply as stored

## Input Data

The current production kernel confirmation used:

- L16 direct coarse: `data/configs_phi4_2d/lam1p0_kappac0p340301_L16/configs.npz`
- L32 native fine: `data/configs_phi4_2d/lam1p0_kappac0p340301_L32/configs.npz`
- L64 native fine: `data/configs_phi4_2d/lam1p0_kappac0p340301_L64/configs.npz`

The manifest is:

```text
perfect_blocking/perfect_blocking_lam1p0/manifests/lam1p0_kernel_data_manifest.csv
```

## Final Production Kernel

The selected final production kernel is:

```text
perfect_blocking/perfect_blocking_lam1p0/kernels/final/chosen_kernel.json
perfect_blocking/perfect_blocking_lam1p0/kernels/final/chosen_kernel.txt
```

It was promoted from:

```text
perfect_blocking/perfect_blocking_lam1p0/kernels/candidates/redo_phi2_support_balanced_2000/best_phi2_support_balanced_eta_included.json
```

The same operational kernel is also available through the stable upscaling-selection path:

```text
perfect_blocking/perfect_blocking_lam1p0/kernels/selected_for_upscaling/best_phi2_support_balanced_eta_included.json
```

The previous NN-constrained 7x7 final was archived under:

```text
perfect_blocking/perfect_blocking_lam1p0/kernels/final/archive/chosen_kernel_old_7x7_nn_constrained_eta_included_20260718T192614Z.json
perfect_blocking/perfect_blocking_lam1p0/kernels/final/archive/chosen_kernel_old_7x7_nn_constrained_eta_included_20260718T192614Z.txt
perfect_blocking/perfect_blocking_lam1p0/kernels/final/archive/README_old_7x7_nn_constrained_20260718T192614Z.md
```

## Eta Convention

The anomalous dimension/block normalization eta is part of the perfect blocking kernel convention. Final recorded kernels used for blocking and upsampling include the eta normalization in their coefficients. For `b=2` and `eta=0.25`, this factor is:

```text
eta_scale = 2^(eta/2) = 2^0.125 = 1.0905077326652577
```

Therefore the final stored kernel sum is `eta_scale`, not 1. Scripts must not multiply by `eta_scale` again when loading an eta-included final kernel.

For this final lambda=1.0 kernel:

- base kernel sum before eta scale: 1.0
- final kernel sum after eta scale: 1.0905077326652577
- `kernel_coefficients_include_eta_scale`: true

## Kernel Matrix

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

Dense-grid momentum stability:

```text
min K(p)   = 0.7073660206381788
max K(p)   = 1.0934526085021135
min 1/K(p) = 0.9145343769126572
max 1/K(p) = 1.413695273484878
```

## Confirmation Summary

The support-balanced 5x5 replaces the previous selected/upscaling 7x7 because the 7x7 over-preserved `phi4` while leaving `phi2` and `local_kurtosis_ratio` mismatched. The promoted kernel intentionally balances support coverage: it improves `phi2` and local-kurtosis overlap strongly, keeps `NN` and `G_pmin_avg` controlled, and accepts a modest broadening/tradeoff in `phi4` and action-density.

L16 direct versus L32 blocked to L16:

| observable | previous selected 7x7 KS | support-balanced 5x5 KS |
|---|---:|---:|
| action_density | 0.0308 | 0.0373 |
| phi2 | 0.0708 | 0.0384 |
| phi4 | 0.0206 | 0.0318 |
| local_kurtosis_ratio | 0.2195 | 0.0863 |
| NN | 0.0262 | 0.0195 |
| G_pmin_avg | 0.0174 | 0.0116 |

L32 direct versus L64 blocked to L32:

| observable | previous selected 7x7 KS | support-balanced 5x5 KS |
|---|---:|---:|
| action_density | 0.0444 | 0.0408 |
| phi2 | 0.1216 | 0.0510 |
| phi4 | 0.0220 | 0.0404 |
| local_kurtosis_ratio | 0.4298 | 0.1562 |
| NN | 0.0534 | 0.0360 |
| G_pmin_avg | 0.0196 | 0.0206 |

Full confirmation records:

```text
perfect_blocking/perfect_blocking_lam1p0/tests/final/final_confirm_phi2_support_balanced_kernel_full/
perfect_blocking/perfect_blocking_lam1p0/tests/final/final_confirm_phi2_support_balanced_kernel_L64_to_L32/
```

## Blocked Observable Products

The production blocked-observable files generated with `kernels/final/chosen_kernel.json` are:

```text
perfect_blocking/perfect_blocking_lam1p0/observables/blocked/native_L32_blocked_to_L16_observables_per_config.csv
perfect_blocking/perfect_blocking_lam1p0/observables/blocked/native_L32_blocked_to_L16_Gk_per_config.csv
perfect_blocking/perfect_blocking_lam1p0/observables/blocked/native_L32_blocked_to_L16_Gk_summary_per_config.csv
perfect_blocking/perfect_blocking_lam1p0/observables/blocked/native_L64_blocked_to_L32_observables_per_config.csv
perfect_blocking/perfect_blocking_lam1p0/observables/blocked/native_L64_blocked_to_L32_Gk_per_config.csv
perfect_blocking/perfect_blocking_lam1p0/observables/blocked/native_L64_blocked_to_L32_Gk_summary_per_config.csv
```

Blocked field arrays were not saved under `perfect_blocking/`.

## Commands

Block L32 to L16 and measure:

```bash
../.venv/bin/python -B perfect_blocking/scripts/block_and_measure.py --config perfect_blocking/perfect_blocking_lam1p0/run_configs/kernel_training_lam1p0.yaml --mode blocked --fine-configs data/configs_phi4_2d/lam1p0_kappac0p340301_L32/configs.npz --kernel perfect_blocking/perfect_blocking_lam1p0/kernels/final/chosen_kernel.json --output-csv perfect_blocking/perfect_blocking_lam1p0/observables/blocked/native_L32_blocked_to_L16_observables_per_config.csv --gk-output-csv perfect_blocking/perfect_blocking_lam1p0/observables/blocked/native_L32_blocked_to_L16_Gk_per_config.csv --gk-summary-output-csv perfect_blocking/perfect_blocking_lam1p0/observables/blocked/native_L32_blocked_to_L16_Gk_summary_per_config.csv --fine-L 32 --coarse-L 16 --lambda 1.0 --kappa-f 0.340301 --kappa-c 0.340301 --block-factor 2 --source-prefix native_L32_blocked_to_L16 --ensemble-label native_L32_blocked_to_L16 --save-blocked-configs false
```

Block L64 to L32 and measure:

```bash
../.venv/bin/python -B perfect_blocking/scripts/block_and_measure.py --config perfect_blocking/perfect_blocking_lam1p0/run_configs/kernel_training_lam1p0.yaml --mode blocked --fine-configs data/configs_phi4_2d/lam1p0_kappac0p340301_L64/configs.npz --kernel perfect_blocking/perfect_blocking_lam1p0/kernels/final/chosen_kernel.json --output-csv perfect_blocking/perfect_blocking_lam1p0/observables/blocked/native_L64_blocked_to_L32_observables_per_config.csv --gk-output-csv perfect_blocking/perfect_blocking_lam1p0/observables/blocked/native_L64_blocked_to_L32_Gk_per_config.csv --gk-summary-output-csv perfect_blocking/perfect_blocking_lam1p0/observables/blocked/native_L64_blocked_to_L32_Gk_summary_per_config.csv --fine-L 64 --coarse-L 32 --lambda 1.0 --kappa-f 0.340301 --kappa-c 0.340301 --block-factor 2 --source-prefix native_L64_blocked_to_L32 --ensemble-label native_L64_blocked_to_L32 --save-blocked-configs false
```
