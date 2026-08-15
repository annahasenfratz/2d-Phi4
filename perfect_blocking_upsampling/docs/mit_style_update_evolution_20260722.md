# MIT-Style Inverse-Blocking Update Evolution

This note records the changes made to the lambda=1.0 wrapped-flow MIT reference drivers.  It is an implementation and acceptance-history summary, not a production-performance claim.

## 1. Direct MIT independence update: L8 to L16

The first reference implementation is `run_lam1p0_mit_style_inverse_blocking_L8to16.py`, submitted through `submit_mit_style`.

One recorded sweep is one global independence-MH proposal:

1. draw a new coarse configuration from the recorded native coarse pool;
2. draw fresh wrapped-flow latent fields for all three detail sectors;
3. reconstruct the complete fine configuration;
4. accept using the full independence ratio

   `logA = -S_f(new) - logq_full(new) + S_f(old) + logq_full(old)`.

`logq_full` includes the coarse density, all latent prior terms, the wrapped conditional-flow density/Jacobian, and the fixed inverse-blocking Jacobian convention.  The implementation saves sweep zero, preserves rejected states, and writes detailed old/new density decompositions to a debug CSV.

With the matching final 5x5 kernel and wrapped-flow checkpoint, the L8-to-L16 run

`mit_bL16_wrapped_RQS_cfg0_N128_S400_wrappedFlow_final5x5_directMIT_blockedNativeInit_20260722T092932Z`

had a global direct-MIT acceptance of **17.42%**.

## 2. Direct MIT independence update: L16 to L32

The same direct independence construction was generalized to arbitrary factor-two lattice sizes.  The L16-to-L32 run

`mit_bL32_wrapped_RQS_cfg0_N250_S400_wrappedFlow_final5x5_directMIT_blockedNativeInit_20260722T095414Z`

had acceptance **1.646%**.  Thus the global, all-sector independence proposal lost substantial overlap as the volume increased.  This was the motivation for separating the proposal into smaller exact MH transitions.

## 3. Four-substep state update

The new driver is:

`perfect_blocking_upsampling/scripts/run_lam1p0_mit_four_substep_L8to16.py`

and the editable top-level submit script is:

`submit_mit_four_substep`.

The chain state is represented as `(c, z01, z10, z11)`.  A four-substep cycle performs separate MH decisions for:

1. the coarse field;
2. `z01`;
3. `z10`;
4. `z11`.

Every substep reconstructs the autoregressive details and fine field, then uses the same full trusted expression:

`logA = -S_f(new) - logq_full(new) + S_f(old) + logq_full(old)`.

The decomposition is only a diagnostic; acceptance always uses the full density calculation.  Per-substep diagnostics include old/new action, coarse action, latent priors, flow Jacobian, full log density, full/decomposed `logA` agreement, and accept/reject state.

### Effect at L8 to L16

For the short L8-to-L16 four-substep smoke run, the independent latent refreshes accepted substantially more often than the original all-at-once global draw:

| Update | Acceptance |
|---|---:|
| Coarse substep | 22.0% |
| `z01` refresh | 44.625% |
| `z10` refresh | 43.75% |
| `z11` refresh | 46.375% |

The improvement comes from proposing one latent sector at a time.  The flow can retain the accepted sectors while changing a smaller part of the fine configuration, so a rejected sector refresh does not discard the entire state transition.

### Initial L16 to L32 result

With an independent coarse refresh, the L16-to-L32 coarse substep still had only about **2.17%** acceptance; the individual latent substeps were approximately **15.7–18.4%**.  Thus splitting the latent fields helped, but the independent whole-coarse draw remained the bottleneck at L16 to L32.

## 4. Reversible coarse Metropolis kernel

The coarse substep was then changed from an independent source-pool replacement to `coarse_mh_kernel`:

* a coarse transition consists of per-site Gaussian proposals with exact local `S_c` accept/reject decisions;
* sites are visited in a fresh uniform random permutation;
* the kernel is reversible with respect to `q_c(c) proportional to exp(-S_c(c))`;
* the proposed coarse field is passed through the retained wrapped-flow latents, then corrected by the same outer full-density fine-field MH ratio.

There are therefore two acceptance rates:

1. **inner coarse MH acceptance**, for the local transitions targeting the coarse proposal distribution;
2. **outer coarse A/R acceptance**, for the exact fine target correction after inverse blocking.

For the early L16-to-L32 `coarse_mh_kernel` run

`mit4_bL32_wrapped_RQS_cfg0_N128_S400_wrappedFlow_final5x5_fourSubstep_coarseMH_blockedNativeInit_20260722T162643Z`,

the observed rates were:

| Quantity | Acceptance |
|---|---:|
| Inner per-site coarse MH | 91.98% |
| Outer coarse fine-field A/R | 59.18% |
| `z01` substep | 19.73% |
| `z10` substep | 17.97% |
| `z11` substep | 19.53% |

This replaces the roughly 2% outer coarse acceptance of an independent whole-coarse refresh with a much more usable exact coarse transition.  It remains exact because the coarse proposal kernel has the same invariant proposal density; the outer A/R still evaluates the full old/new fine action and full model density.

## 5. Divided four-substep schedule

The four-substep driver now supports `--divide 1` and `--divide 2`.

For `divide=2`, each coarse and latent lattice is partitioned into the four residue classes

`(y mod 2, x mod 2) = (0,0), (0,1), (1,0), (1,1)`.

For each residue class, the driver performs, in order:

1. one coarse subset transition;
2. one `z01` subset refresh;
3. one `z10` subset refresh;
4. one `z11` subset refresh.

The four subset cycles are all recorded under one output sweep.  **Consequently, in the newest divided run, one reported sweep is four sequential coarse/detail sub-sweeps that would previously have been counted separately.**  It contains 16 sequential substeps per chain, covers every residue class once, and is not directly comparable to a single old full-lattice update without accounting for this changed sweep definition.

The current short divided L16-to-L32 run

`mit4_bL32_wrapped_RQS_cfg0_N128_S400_wrappedFlow_final5x5_fourSubstep_coarseMH_div2_blockedNativeInit_20260722T165803Z`

has completed three recorded sweeps.  Its cumulative rates are:

| Quantity | Acceptance |
|---|---:|
| Inner per-site coarse MH | 59.48% |
| Outer coarse subset A/R | 45.64% |
| `z01` subset refresh | 38.80% |
| `z10` subset refresh | 40.10% |
| `z11` subset refresh | 42.19% |

These values are preliminary because the run is still short, but the observed outer coarse rate is the approximately 45% rate visible in the run outputs.

## Current files and diagnostics

The main modified implementation files are:

* `perfect_blocking_upsampling/scripts/run_lam1p0_mit_style_inverse_blocking_L8to16.py`
* `perfect_blocking_upsampling/scripts/run_lam1p0_mit_four_substep_L8to16.py`
* `submit_mit_style`
* `submit_mit_four_substep`

The four-substep runs preserve the usual observable files and add:

* `observables/acceptance_history.csv` for sweep-level aggregates;
* `debug/four_substep_diagnostics.csv` for individual outer MH decisions;
* `debug/substep_acceptance_history.csv` for acceptance by sector and residue class;
* `debug/coarse_kernel_acceptance_history.csv` for inner per-site coarse MH acceptance.
