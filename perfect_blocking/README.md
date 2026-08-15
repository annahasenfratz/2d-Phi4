# Perfect Blocking

This tree contains reusable kernel-training and kernel-testing infrastructure for inverse-blocking/perfect-blocking studies.

Raw field configurations are not stored in this tree. Native configurations live
directly under data/configs_phi4_2d/. Generated and explicitly saved blocked
configurations, when needed, live under data/configs_phi4_2d/generated/ and
data/configs_phi4_2d/blocked/.

## Layout

- `scripts/`: reusable blocking, observable, comparison, and kernel utilities.
- `docs/`: workflow documentation.
- `perfect_blocking_lam*/`: lambda-specific run metadata, kernels, observable CSVs, comparisons, plots, logs, and manifests.

The lambda-specific directories should contain YAML run metadata that points to configuration files under `data/configs_phi4_2d/`. They should not contain raw `.npz` or `.npy` field configurations.

Each standard measurement now produces two observable products:

- `*_all_observables_per_config.csv`: one row per configuration for scalar observables.
- `*_Gk_per_config.csv`: long-format Fourier structure-factor rows for momenta from `gk_momenta`, plus `*_Gk_summary_per_config.csv` with `G_00`, `G_10`, `G_01`, and `G_pmin_avg` columns when available.

## Eta Convention

The anomalous dimension/block normalization eta is part of the perfect blocking kernel convention. Final recorded kernels used for blocking and upsampling should include the eta normalization in their coefficients. For b=2 and eta=0.25 this factor is `eta_scale=2^(eta/2)=2^0.125`. Therefore the final stored kernel sum is `eta_scale`, not 1. Scripts must not multiply by `eta_scale` again when loading an eta-included final kernel.

During training/scanning, kernels may be represented as base normalized kernels with `sum(K)=1` plus an explicit `eta_scale`. When a kernel is promoted to final, write an eta-included copy to `kernels/final` and record both the base kernel and eta-included kernel metadata.

## Lambda 0.2 End-To-End Test

From the repository root, regenerate the current lambda=0.2 observable workflow with:

```bash
../.venv/bin/python -B perfect_blocking/scripts/block_and_measure.py \
  --config perfect_blocking/perfect_blocking_lam0p2/run_configs/kernel_training_lam0p2.yaml \
  --mode native \
  --configs data/configs_phi4_2d/lam0p2_kappac0p323124_L32/configs.npz \
  --output-csv perfect_blocking/perfect_blocking_lam0p2/observables/native/native_L32_all_observables_per_config.csv \
  --gk-output-csv perfect_blocking/perfect_blocking_lam0p2/observables/native/native_L32_Gk_per_config.csv \
  --gk-summary-output-csv perfect_blocking/perfect_blocking_lam0p2/observables/native/native_L32_Gk_summary_per_config.csv \
  --source-prefix native_L32
```

```bash
../.venv/bin/python -B perfect_blocking/scripts/block_and_measure.py \
  --config perfect_blocking/perfect_blocking_lam0p2/run_configs/kernel_training_lam0p2.yaml \
  --mode native \
  --configs data/configs_phi4_2d/lam0p2_kappac0p323124_L16/configs.npz \
  --output-csv perfect_blocking/perfect_blocking_lam0p2/observables/native/direct_L16_all_observables_per_config.csv \
  --gk-output-csv perfect_blocking/perfect_blocking_lam0p2/observables/native/direct_L16_Gk_per_config.csv \
  --gk-summary-output-csv perfect_blocking/perfect_blocking_lam0p2/observables/native/direct_L16_Gk_summary_per_config.csv \
  --source-prefix direct_L16
```

```bash
../.venv/bin/python -B perfect_blocking/scripts/block_and_measure.py \
  --config perfect_blocking/perfect_blocking_lam0p2/run_configs/kernel_training_lam0p2.yaml \
  --mode blocked \
  --fine-configs data/configs_phi4_2d/lam0p2_kappac0p323124_L32/configs.npz \
  --kernel perfect_blocking/perfect_blocking_lam0p2/kernels/final/chosen_kernel.json \
  --output-csv perfect_blocking/perfect_blocking_lam0p2/observables/blocked/native_L32_blocked_to_L16_observables_per_config.csv \
  --gk-output-csv perfect_blocking/perfect_blocking_lam0p2/observables/blocked/native_L32_blocked_to_L16_Gk_per_config.csv \
  --gk-summary-output-csv perfect_blocking/perfect_blocking_lam0p2/observables/blocked/native_L32_blocked_to_L16_Gk_summary_per_config.csv \
  --source-prefix native_L32_blocked_to_L16 \
  --save-blocked-configs false
```

```bash
../.venv/bin/python -B perfect_blocking/scripts/test_kernel_observables.py \
  --config perfect_blocking/perfect_blocking_lam0p2/run_configs/kernel_training_lam0p2.yaml \
  --direct-csv perfect_blocking/perfect_blocking_lam0p2/observables/native/direct_L16_all_observables_per_config.csv \
  --blocked-csv perfect_blocking/perfect_blocking_lam0p2/observables/blocked/native_L32_blocked_to_L16_observables_per_config.csv \
  --output-csv perfect_blocking/perfect_blocking_lam0p2/observables/comparisons/histogram_quantification_direct_L16_vs_native_L32_blockedtoL16.csv \
  --gk-direct-csv perfect_blocking/perfect_blocking_lam0p2/observables/native/direct_L16_Gk_summary_per_config.csv \
  --gk-blocked-csv perfect_blocking/perfect_blocking_lam0p2/observables/blocked/native_L32_blocked_to_L16_Gk_summary_per_config.csv \
  --gk-output-csv perfect_blocking/perfect_blocking_lam0p2/observables/comparisons/histogram_quantification_direct_L16_vs_native_L32_blockedtoL16_Gk.csv \
  --plot-dir perfect_blocking/perfect_blocking_lam0p2/plots
```
