# Kernel Training Workflow

This workflow is meant to make future lambda studies repeatable instead of binding kernel choices to old `controlled_patch_*` paths.

## Storage Policy

Raw field configurations are not stored under `perfect_blocking/`.

Native 2D phi4 configurations live directly under:

```text
data/configs_phi4_2d/
```

Generated configurations, if explicitly saved, should live under:

```text
data/configs_phi4_2d/generated/
```

Blocked configurations, if explicitly saved, should live under:

```text
data/configs_phi4_2d/blocked/
```

The default for blocked kernel tests is to save only per-config observable CSVs under `perfect_blocking/perfect_blocking_lam*/observables/`.

Each measurement produces two observable products:

- `*_all_observables_per_config.csv`: one row per configuration for scalar observables.
- `*_Gk_per_config.csv`: long-format Fourier structure-factor measurements, plus `*_Gk_summary_per_config.csv` for convenient one-row-per-config comparison columns.

The current G(k) convention is the unconnected structure factor

```text
G_cfg(k) = |sum_x exp(i k.x) phi(x)|^2 / V
```

With `m = V^{-1} sum_x phi(x)`, this gives `G_cfg(0,0) = V m^2`. The default momentum index set is `(0,0)`, `(1,0)`, and `(0,1)`, where `(nx,ny)` means `kx = 2*pi*nx/L` and `ky = 2*pi*ny/L`. The summary file includes `G_00`, `G_10`, `G_01`, and `G_pmin_avg = 0.5 * (G_10 + G_01)`.

## Eta Convention

The anomalous dimension/block normalization eta is part of the perfect blocking kernel convention. Final recorded kernels used for blocking and upsampling should include the eta normalization in their coefficients. For b=2 and eta=0.25 this factor is `eta_scale=2^(eta/2)=2^0.125`. Therefore the final stored kernel sum is `eta_scale`, not 1. Scripts must not multiply by `eta_scale` again when loading an eta-included final kernel.

During training/scanning, kernels may be represented as base normalized kernels with `sum(K)=1` plus an explicit `eta_scale`. When a kernel is promoted to final, write an eta-included copy to `kernels/final` and record both the base kernel and eta-included kernel metadata.

## Generic Steps

1. Choose `lambda` and the target kappa range.
2. Generate or locate native fine configurations in `data/configs_phi4_2d/`.
3. Generate or locate direct coarse configurations in `data/configs_phi4_2d/`.
4. Choose the kernel family. The current active family is `5x5`.
5. Scan or train kernels using operator matching with explicit `kappa_f` and `kappa_c` fields.
6. Save candidates under `perfect_blocking/perfect_blocking_lam*/kernels/candidates/`.
7. Select one final kernel and copy it to `perfect_blocking/perfect_blocking_lam*/kernels/final/`.
8. Block fine configurations using the final kernel.
9. Compute per-config scalar observables and G(k) observables for native fine, direct coarse, and blocked-fine fields.
10. Compare direct coarse versus blocked-native scalar observables and G(k) summaries.
11. Inspect histogram PDFs and quantitative overlap scores.
12. Decide whether the kernel and coarse source have adequate support/coverage for upscaling.

## Mature Perfect-Kernel Training Protocol

This is the protocol that emerged from the successful lambda=0.2 `rand5x5_0084` workflow and was applied/extended for lambda=0.5. New lambda points should be judged against this standard before a kernel is called final.

### Kernel Family

Start from a general symmetric 5x5 kernel rather than a compact/simple ansatz. Preserve the lattice symmetries

```text
K(dx,dy) = K(-dx,dy) = K(dx,-dy) = K(dy,dx)
```

During training/scanning, use a base-normalized kernel with `sum(K_base)=1`. When a kernel is promoted to operational/final use, write an eta-included copy:

```text
K_final = 2^(eta/2) K_base
```

For eta=0.25 and block factor b=2, `2^(eta/2)=2^0.125=1.0905077326652577`. Therefore the final stored kernel satisfies `sum(K_final)=2^(eta/2)` and must be applied as stored with no further eta multiplication.

### Data And Comparisons

Compare direct coarse L16 against native L32 blocked to L16. Native configurations live directly under `data/configs_phi4_2d/`. Do not save blocked configurations by default; save blocked observables and G(k) summaries only.

### Observable Basis

Every mature comparison includes:

```text
phi2
phi4
local_kurtosis_ratio = phi4/(phi2)^2
NN
2nn
diag
action_density
m
m2
m4
Binder_U4_from_averages
xi_over_L
G_00
G_10
G_01
G_pmin_avg
```

### Distributional Diagnostics

For every observable, compute distributional metrics, not only central means:

```text
mean difference
standardized mean shift
width ratio
total variation distance
Jensen-Shannon divergence
Wasserstein-1 distance
Kolmogorov-Smirnov statistic and p-value
```

### Tail And Coverage Criteria

Do not select kernels by minimizing only central means. Check width ratios, tails, and sector coverage. For upscaling, support/coverage is essential. A somewhat broader direct coarse ensemble is acceptable if it covers the blocked-native target. A too-narrow ensemble, or one missing tails or magnetization/G(k) sectors, is dangerous.

### Guardrails

Every training or search objective must include hard guardrails, large penalties, or accept/reject filtering for:

```text
action_density
m2
m4
Binder_U4_from_averages
xi_over_L
G_pmin_avg
momentum-space positivity
inverse-kernel conditioning
```

A candidate that improves `phi2` or `local_kurtosis_ratio` by damaging `action_density` is a failed candidate, not an improved final kernel.

### Momentum-Space Checks

Compute `K(p)` on both a dense Brillouin-zone grid and the relevant discrete L16 grid. Require:

```text
K(p) > 0
no near-zero modes
bounded max 1/K(p)
```

Record:

```text
min K(p)
max K(p)
min 1/K(p)
max 1/K(p)
condition number = max K(p) / min K(p)
```

### Selection

The best kernel is not necessarily the one that minimizes one local observable. Select by a multi-objective score plus hard guardrails. The current final kernel must always be included as an explicit baseline in retraining studies. A larger family, such as 7x7, contains the embedded 5x5 kernel as a special case, so a valid search must be able to return the embedded 5x5 if no acceptable improvement exists.

## Interpretation

The direct coarse ensemble does not need to be an exactly identical blocked marginal. For upscaling, coverage/support is the key requirement.

A direct L16 ensemble that is somewhat broader than the blocked-native distribution is acceptable if it covers the target. A too-narrow ensemble, or one missing tails or sectors, is dangerous because the detail/upscaling step will be forced to extrapolate.

## Lambda 0.2 Current Setup

The active run metadata is:

```text
perfect_blocking/perfect_blocking_lam0p2/run_configs/kernel_training_lam0p2.yaml
```

The current final kernel is:

```text
perfect_blocking/perfect_blocking_lam0p2/kernels/final/chosen_kernel.json
```

The current final kernel is `rand5x5_0084`. A later 5x5 candidate, `lam0p2_candidate_5x5_114`, is preserved in `kernels/candidates/` as an intermediate candidate but has not been promoted to final.

## Lambda 1.0 Production Kernel Reopening

The lambda=1.0 kernel study was reopened after upscaling diagnostics showed that the previous phi2-priority/NN-guarded 7x7 kernel still left a direct-native L16 versus blocked-native L32-to-L16 coarse-marginal mismatch in `phi2`, `phi4`, and especially `local_kurtosis_ratio`. That mismatch invalidated the flow-training starting point because the detail flow was being asked to repair a coarse marginal that was already wrong.

The current lambda=1.0 production kernel is therefore the phi2-support-balanced 5x5:

```text
perfect_blocking/perfect_blocking_lam1p0/kernels/final/chosen_kernel.json
```

It is eta-included, has `sum(K)=2^0.125=1.0905077326652577`, and must be applied as stored. It was selected using direct-native L16 versus blocked-native L32-to-L16 confirmation, then repeated on direct-native L32 versus blocked-native L64-to-L32. The selection prioritizes distributional support and tail coverage over mean-only matching: a slightly broader blocked distribution is preferable to missing important regions of `phi2` or local-kurtosis support.

The current confirmation records are:

```text
perfect_blocking/perfect_blocking_lam1p0/tests/final/final_confirm_phi2_support_balanced_kernel_full/
perfect_blocking/perfect_blocking_lam1p0/tests/final/final_confirm_phi2_support_balanced_kernel_L64_to_L32/
```

## Reusable Commands

Measure native fine configs:

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

Measure direct coarse configs:

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

Block fine configs and measure the blocked fields without saving blocked configs:

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

Compare direct coarse and blocked-native observables:

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

## Lambda 0.5 Current Setup

The active run metadata is:

```text
perfect_blocking/perfect_blocking_lam0p5/run_configs/kernel_training_lam0p5.yaml
```

The current final kernel is:

```text
perfect_blocking/perfect_blocking_lam0p5/kernels/final/chosen_kernel.json
```

The current final 5x5 kernel is `refined_rand0084_0001_4_22_p`, promoted from:

```text
perfect_blocking_upsampling/outputs/lam0p5_5x5_kernel_search/kernel_candidates/best_lam0p5_5x5_kernel.json
```

The final stored coefficients include eta scale. For lambda=0.5, kappa 0.343469, eta=0.25, and b=2, the final kernel sum is `2^0.125 = 1.0905077326652577`.

Reusable lambda=0.5 commands:

```bash
../.venv/bin/python -B perfect_blocking/scripts/block_and_measure.py --config perfect_blocking/perfect_blocking_lam0p5/run_configs/kernel_training_lam0p5.yaml --mode native --configs data/configs_phi4_2d/lam0p5_kappac0p343469_L16/configs.npz --output-csv perfect_blocking/perfect_blocking_lam0p5/observables/native/direct_L16_all_observables_per_config.csv --gk-output-csv perfect_blocking/perfect_blocking_lam0p5/observables/native/direct_L16_Gk_per_config.csv --gk-summary-output-csv perfect_blocking/perfect_blocking_lam0p5/observables/native/direct_L16_Gk_summary_per_config.csv --source-prefix direct_L16
```

```bash
../.venv/bin/python -B perfect_blocking/scripts/block_and_measure.py --config perfect_blocking/perfect_blocking_lam0p5/run_configs/kernel_training_lam0p5.yaml --mode native --configs data/configs_phi4_2d/lam0p5_kappac0p343469_L32/configs.npz --output-csv perfect_blocking/perfect_blocking_lam0p5/observables/native/native_L32_all_observables_per_config.csv --gk-output-csv perfect_blocking/perfect_blocking_lam0p5/observables/native/native_L32_Gk_per_config.csv --gk-summary-output-csv perfect_blocking/perfect_blocking_lam0p5/observables/native/native_L32_Gk_summary_per_config.csv --source-prefix native_L32
```

```bash
../.venv/bin/python -B perfect_blocking/scripts/block_and_measure.py --config perfect_blocking/perfect_blocking_lam0p5/run_configs/kernel_training_lam0p5.yaml --mode blocked --fine-configs data/configs_phi4_2d/lam0p5_kappac0p343469_L32/configs.npz --kernel perfect_blocking/perfect_blocking_lam0p5/kernels/final/chosen_kernel.json --output-csv perfect_blocking/perfect_blocking_lam0p5/observables/blocked/native_L32_blocked_to_L16_observables_per_config.csv --gk-output-csv perfect_blocking/perfect_blocking_lam0p5/observables/blocked/native_L32_blocked_to_L16_Gk_per_config.csv --gk-summary-output-csv perfect_blocking/perfect_blocking_lam0p5/observables/blocked/native_L32_blocked_to_L16_Gk_summary_per_config.csv --source-prefix native_L32_blocked_to_L16 --save-blocked-configs false
```

```bash
../.venv/bin/python -B perfect_blocking/scripts/test_kernel_observables.py --config perfect_blocking/perfect_blocking_lam0p5/run_configs/kernel_training_lam0p5.yaml --direct-csv perfect_blocking/perfect_blocking_lam0p5/observables/native/direct_L16_all_observables_per_config.csv --blocked-csv perfect_blocking/perfect_blocking_lam0p5/observables/blocked/native_L32_blocked_to_L16_observables_per_config.csv --output-csv perfect_blocking/perfect_blocking_lam0p5/observables/comparisons/histogram_quantification_direct_L16_vs_native_L32_blockedtoL16.csv --gk-direct-csv perfect_blocking/perfect_blocking_lam0p5/observables/native/direct_L16_Gk_summary_per_config.csv --gk-blocked-csv perfect_blocking/perfect_blocking_lam0p5/observables/blocked/native_L32_blocked_to_L16_Gk_summary_per_config.csv --gk-output-csv perfect_blocking/perfect_blocking_lam0p5/observables/comparisons/histogram_quantification_direct_L16_vs_native_L32_blockedtoL16_Gk.csv --plot-dir perfect_blocking/perfect_blocking_lam0p5/plots --plot-prefix histogram_direct_L16_vs_native_L32_blockedtoL16
```

```bash
../.venv/bin/python -B perfect_blocking/scripts/regress_eta_convention_lam0p5.py
```

## Lambda 1.0 Current Setup

The active run metadata is:

```text
perfect_blocking/perfect_blocking_lam1p0/run_configs/kernel_training_lam1p0.yaml
```

The current final kernel is:

```text
perfect_blocking/perfect_blocking_lam1p0/kernels/final/chosen_kernel.json
```

The current operational final kernel is the NN-constrained 7x7 no-corner candidate, promoted from:

```text
perfect_blocking/perfect_blocking_lam1p0/kernels/candidates/7x7_no33_nn_constrained/best_7x7_no33_nn_constrained_eta_included.json
```

The old lambda=1.0 notes identified the KL eta-fit 5x5 kernel as the preferred starting point. The first active final was the nearby `spatial_only_eta0p25_5x5` candidate, validated at kappa=0.340301 and then treated as a provisional baseline. A systematic retraining pass produced a better 5x5; a 7x7 no-corner refinement improved local one-site moments but initially regressed `NN`; the final promoted 7x7 is the NN-constrained no-corner candidate.

```text
perfect_blocking/perfect_blocking_lam1p0/tests/intermediate/7x7_no33_nn_constrained/recommendation.md
```

The final stored coefficients include eta scale. For lambda=1.0, kappa 0.340301, eta=0.25, and b=2, the final kernel sum is `2^0.125 = 1.0905077326652577`.

The final lambda=1.0 kernel has `K33=0`, outer-shell base coefficients `K30=-0.00036894189276879484`, `K31=0.0022330433962489153`, `K32=0.0016781689520450665`, and dense-grid `max 1/K(p)=1.3311418658960643`.

Promotion comparison:

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

Reusable lambda=1.0 commands:

```bash
../.venv/bin/python -B perfect_blocking/scripts/block_and_measure.py --config perfect_blocking/perfect_blocking_lam1p0/run_configs/kernel_training_lam1p0.yaml --mode native --configs data/configs_phi4_2d/lam1p0_kappac0p340301_L16/configs.npz --output-csv perfect_blocking/perfect_blocking_lam1p0/observables/native/direct_L16_all_observables_per_config.csv --gk-output-csv perfect_blocking/perfect_blocking_lam1p0/observables/native/direct_L16_Gk_per_config.csv --gk-summary-output-csv perfect_blocking/perfect_blocking_lam1p0/observables/native/direct_L16_Gk_summary_per_config.csv --source-prefix direct_L16
```

```bash
../.venv/bin/python -B perfect_blocking/scripts/block_and_measure.py --config perfect_blocking/perfect_blocking_lam1p0/run_configs/kernel_training_lam1p0.yaml --mode native --configs data/configs_phi4_2d/lam1p0_kappac0p340301_L32/configs.npz --output-csv perfect_blocking/perfect_blocking_lam1p0/observables/native/native_L32_all_observables_per_config.csv --gk-output-csv perfect_blocking/perfect_blocking_lam1p0/observables/native/native_L32_Gk_per_config.csv --gk-summary-output-csv perfect_blocking/perfect_blocking_lam1p0/observables/native/native_L32_Gk_summary_per_config.csv --source-prefix native_L32
```

```bash
../.venv/bin/python -B perfect_blocking/scripts/block_and_measure.py --config perfect_blocking/perfect_blocking_lam1p0/run_configs/kernel_training_lam1p0.yaml --mode blocked --fine-configs data/configs_phi4_2d/lam1p0_kappac0p340301_L32/configs.npz --kernel perfect_blocking/perfect_blocking_lam1p0/kernels/final/chosen_kernel.json --output-csv perfect_blocking/perfect_blocking_lam1p0/observables/blocked/native_L32_blocked_to_L16_observables_per_config.csv --gk-output-csv perfect_blocking/perfect_blocking_lam1p0/observables/blocked/native_L32_blocked_to_L16_Gk_per_config.csv --gk-summary-output-csv perfect_blocking/perfect_blocking_lam1p0/observables/blocked/native_L32_blocked_to_L16_Gk_summary_per_config.csv --source-prefix native_L32_blocked_to_L16 --save-blocked-configs false
```

```bash
../.venv/bin/python -B perfect_blocking/scripts/test_kernel_observables.py --config perfect_blocking/perfect_blocking_lam1p0/run_configs/kernel_training_lam1p0.yaml --direct-csv perfect_blocking/perfect_blocking_lam1p0/observables/native/direct_L16_all_observables_per_config.csv --blocked-csv perfect_blocking/perfect_blocking_lam1p0/observables/blocked/native_L32_blocked_to_L16_observables_per_config.csv --output-csv perfect_blocking/perfect_blocking_lam1p0/observables/comparisons/histogram_quantification_direct_L16_vs_native_L32_blockedtoL16.csv --gk-direct-csv perfect_blocking/perfect_blocking_lam1p0/observables/native/direct_L16_Gk_summary_per_config.csv --gk-blocked-csv perfect_blocking/perfect_blocking_lam1p0/observables/blocked/native_L32_blocked_to_L16_Gk_summary_per_config.csv --gk-output-csv perfect_blocking/perfect_blocking_lam1p0/observables/comparisons/histogram_quantification_direct_L16_vs_native_L32_blockedtoL16_Gk.csv --plot-dir perfect_blocking/perfect_blocking_lam1p0/plots --plot-prefix histogram_direct_L16_vs_native_L32_blockedtoL16
```

```bash
../.venv/bin/python -B perfect_blocking/scripts/regress_eta_convention_lam1p0.py
```
