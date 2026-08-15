# Perfect Blocking Lambda 0.5

This directory contains the lambda=0.5 perfect-blocking workflow using the shared scripts in `perfect_blocking/scripts/`.

Raw field configurations are not stored here. Native configurations remain under `data/configs_phi4_2d/`; generated configurations should go under `data/configs_phi4_2d/generated/`; blocked configurations are not saved by default and, if saved explicitly, should go under `data/configs_phi4_2d/blocked/`.

## Parameters

- `lambda`: 0.5
- `kappa_cr`: 0.343469
- `kappa_f`: 0.343469
- `kappa_c`: 0.343469
- `eta`: 0.25
- `block_factor`: 2
- `eta_scale`: `2^0.125 = 1.0905077326652577`
- active kernel family: 5x5

## Input Data

The current analysis used:

- L8: `data/configs_phi4_2d/lam0p5_kappac0p343469_L8/configs.npz`
- L16 direct coarse: `data/configs_phi4_2d/lam0p5_kappac0p343469_L16/configs.npz`
- L32 native fine: `data/configs_phi4_2d/lam0p5_kappac0p343469_L32/configs.npz`

The manifest is:

```text
perfect_blocking/perfect_blocking_lam0p5/manifests/lam0p5_kernel_data_manifest.csv
```

## Final Kernel

The selected final kernel is:

```text
perfect_blocking/perfect_blocking_lam0p5/kernels/final/chosen_kernel.json
```

It was promoted from the existing lambda=0.5 5x5 candidate:

```text
perfect_blocking_upsampling/outputs/lam0p5_5x5_kernel_search/kernel_candidates/best_lam0p5_5x5_kernel.json
```

The selected candidate is `refined_rand0084_0001_4_22_p`. It was chosen because it had the lowest recorded 5x5 kernel-search score among the existing lambda=0.5 candidates, with stable Fourier conditioning and roundtrip diagnostics. The candidate source metadata records kappa 0.3426, while this final workflow analyzes the current kappa-cr ensemble at 0.343469.

## Eta Convention

The anomalous dimension/block normalization eta is part of the perfect blocking kernel convention. Final recorded kernels used for blocking and upsampling should include the eta normalization in their coefficients. For b=2 and eta=0.25 this factor is `eta_scale=2^(eta/2)=2^0.125`. Therefore the final stored kernel sum is `eta_scale`, not 1. Scripts must not multiply by `eta_scale` again when loading an eta-included final kernel.

During training/scanning, kernels may be represented as base normalized kernels with `sum(K)=1` plus an explicit `eta_scale`. When a kernel is promoted to final, write an eta-included copy to `kernels/final` and record both the base kernel and eta-included kernel metadata.

For this final lambda=0.5 kernel:

- base kernel sum before eta scale: 1.0
- final kernel sum after eta scale: 1.0905077326652577
- `kernel_coefficients_include_eta_scale`: true

## Commands

Measure direct L16:

```bash
../.venv/bin/python -B perfect_blocking/scripts/block_and_measure.py --config perfect_blocking/perfect_blocking_lam0p5/run_configs/kernel_training_lam0p5.yaml --mode native --configs data/configs_phi4_2d/lam0p5_kappac0p343469_L16/configs.npz --output-csv perfect_blocking/perfect_blocking_lam0p5/observables/native/direct_L16_all_observables_per_config.csv --gk-output-csv perfect_blocking/perfect_blocking_lam0p5/observables/native/direct_L16_Gk_per_config.csv --gk-summary-output-csv perfect_blocking/perfect_blocking_lam0p5/observables/native/direct_L16_Gk_summary_per_config.csv --source-prefix direct_L16
```

Measure native L32:

```bash
../.venv/bin/python -B perfect_blocking/scripts/block_and_measure.py --config perfect_blocking/perfect_blocking_lam0p5/run_configs/kernel_training_lam0p5.yaml --mode native --configs data/configs_phi4_2d/lam0p5_kappac0p343469_L32/configs.npz --output-csv perfect_blocking/perfect_blocking_lam0p5/observables/native/native_L32_all_observables_per_config.csv --gk-output-csv perfect_blocking/perfect_blocking_lam0p5/observables/native/native_L32_Gk_per_config.csv --gk-summary-output-csv perfect_blocking/perfect_blocking_lam0p5/observables/native/native_L32_Gk_summary_per_config.csv --source-prefix native_L32
```

Block L32 to L16 and measure, without saving blocked configurations:

```bash
../.venv/bin/python -B perfect_blocking/scripts/block_and_measure.py --config perfect_blocking/perfect_blocking_lam0p5/run_configs/kernel_training_lam0p5.yaml --mode blocked --fine-configs data/configs_phi4_2d/lam0p5_kappac0p343469_L32/configs.npz --kernel perfect_blocking/perfect_blocking_lam0p5/kernels/final/chosen_kernel.json --output-csv perfect_blocking/perfect_blocking_lam0p5/observables/blocked/native_L32_blocked_to_L16_observables_per_config.csv --gk-output-csv perfect_blocking/perfect_blocking_lam0p5/observables/blocked/native_L32_blocked_to_L16_Gk_per_config.csv --gk-summary-output-csv perfect_blocking/perfect_blocking_lam0p5/observables/blocked/native_L32_blocked_to_L16_Gk_summary_per_config.csv --source-prefix native_L32_blocked_to_L16 --save-blocked-configs false
```

Compare direct L16 against native L32 blocked to L16 and write histogram PDFs:

```bash
../.venv/bin/python -B perfect_blocking/scripts/test_kernel_observables.py --config perfect_blocking/perfect_blocking_lam0p5/run_configs/kernel_training_lam0p5.yaml --direct-csv perfect_blocking/perfect_blocking_lam0p5/observables/native/direct_L16_all_observables_per_config.csv --blocked-csv perfect_blocking/perfect_blocking_lam0p5/observables/blocked/native_L32_blocked_to_L16_observables_per_config.csv --output-csv perfect_blocking/perfect_blocking_lam0p5/observables/comparisons/histogram_quantification_direct_L16_vs_native_L32_blockedtoL16.csv --gk-direct-csv perfect_blocking/perfect_blocking_lam0p5/observables/native/direct_L16_Gk_summary_per_config.csv --gk-blocked-csv perfect_blocking/perfect_blocking_lam0p5/observables/blocked/native_L32_blocked_to_L16_Gk_summary_per_config.csv --gk-output-csv perfect_blocking/perfect_blocking_lam0p5/observables/comparisons/histogram_quantification_direct_L16_vs_native_L32_blockedtoL16_Gk.csv --plot-dir perfect_blocking/perfect_blocking_lam0p5/plots --plot-prefix histogram_direct_L16_vs_native_L32_blockedtoL16
```

Run the eta regression:

```bash
../.venv/bin/python -B perfect_blocking/scripts/regress_eta_convention_lam0p5.py
```

## Agreement Summary

The direct L16 ensemble covers the native L32 blocked-to-L16 target well for the current observables. The main comparison is:

```text
perfect_blocking/perfect_blocking_lam0p5/observables/comparisons/histogram_quantification_direct_L16_vs_native_L32_blockedtoL16.csv
```

Selected scalar summaries:

| observable | mean direct | mean blocked | standardized shift | std ratio direct/blocked | TV | JS | KS |
|---|---:|---:|---:|---:|---:|---:|---:|
| phi2 | 0.900408 | 0.901748 | 0.0155483 | 0.924669 | 0.0484 | 0.00337538 | 0.0248 |
| phi4 | 1.36026 | 1.37333 | 0.0632872 | 0.880914 | 0.0778 | 0.00648703 | 0.0474 |
| NN | 0.624665 | 0.632875 | 0.0734045 | 0.985369 | 0.0576 | 0.00369365 | 0.0318 |
| action_density | -0.178084 | -0.182826 | -0.0647681 | 1.14242 | 0.0762 | 0.00651099 | 0.0502 |
| m2 | 0.451677 | 0.455634 | 0.0201165 | 1.01059 | 0.0552 | 0.00254694 | 0.0234 |
| m4 | 0.243118 | 0.245894 | 0.0165969 | 1.00844 | 0.0506 | 0.00249415 | 0.0234 |

Selected G(k) summary:

| observable | mean direct | mean blocked | standardized shift | std ratio direct/blocked | TV | JS | KS |
|---|---:|---:|---:|---:|---:|---:|---:|
| G_pmin_avg | 3.92215 | 3.84034 | -0.0175162 | 0.997054 | 0.0308 | 0.00192434 | 0.0214 |

The small shifts and overlap scores indicate acceptable support/coverage for upscaling. The direct L16 distribution is not dangerously narrow relative to the blocked target in the checked observables.

## Regression

Eta convention regression passed:

```text
perfect_blocking/perfect_blocking_lam0p5/debug/eta_convention_regression_20260716T175200Z/report.md
```
