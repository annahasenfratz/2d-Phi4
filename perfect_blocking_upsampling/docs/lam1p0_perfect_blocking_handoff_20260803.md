# Lambda=1 perfect-blocking / upscaling handoff — 2026-08-03

Repository root:

```text
/Users/anna/Work/Research/Normalizing-flow/Inverse_RG/phi4_inverse_blocking
```

Use the shared interpreter `../../.venv/bin/python` from this repository (the
repository was moved; `../.venv/bin/python` may not exist).

## Current live job: L32 -> L64 fine-tune

Run directory:

```text
perfect_blocking_upsampling/runs/lam1p0/training/
lam1p0_L32toL64_current5x5_tailshape_finetune_N5000_20260803T170234Z
```

PID at handoff: `12870` (completed).

It is a matched-pair L64 -> L32 training run, used to improve the *independent*
direct L32 -> L64 proposal.  It starts from the L16 -> L32 epoch-2 checkpoint,
trains on 4,000 matched pairs, validates on 500, and tests on 500.  It uses:

- the promoted 5x5 eta-included kernel;
- 40% tail-stratified training draws from the union of the lowest action-density
  and local-kurtosis deciles;
- one-sided low-tail proposal coverage losses for action density and local
  kurtosis;
- a small local-kurtosis shape loss to reduce its excessive high tail.

The five epochs completed.  NLL selected epoch 5:

```text
validation NLL: epoch 0 = 1646.796; epoch 5 = 1644.232
best NLL checkpoint: checkpoint_best_nll.pt (epoch 5)
```

**Do not promote the epoch-5 checkpoint.**  On the held-out 500-configuration
raw conditional histogram check, the initial checkpoint (epoch 0) was better:

```text
                         epoch 0       epoch 5
action KS                  0.054         0.136
phi4 KS                    0.050         0.152
local-kurtosis KS          0.126         0.188
```

No trained epoch improved all three local observables.  The tail losses did act
on individual training batches, but did not yield a stable held-out shape
improvement.  Keep the prior epoch-2 L16->L32 checkpoint as the baseline;
do not spend the 5,000-config independent L32->L64 evaluation on this branch.
The next attempt should change the conditional architecture/conditioning or
train directly against the independent L32 coarse distribution, rather than
continue this loss-only fine-tune.

Useful commands:

```bash
RUN_DIR=perfect_blocking_upsampling/runs/lam1p0/training/lam1p0_L32toL64_current5x5_tailshape_finetune_N5000_20260803T170234Z
tail -80 "$RUN_DIR/logs/run.log"
cat "$RUN_DIR/status.json"
ls "$RUN_DIR/checkpoints"
```

If the computer reboots before completion, that process cannot survive.  Restart
cleanly from the saved latest checkpoint with a new run directory, using the
same command written in `RUN_DIR/submit_manifest.txt`, but replace
`--source-checkpoint` with:

```text
<old RUN_DIR>/checkpoints/checkpoint_latest.pt
```

Keep `--fine-config-source data/configs_phi4_2d/lam1p0_kappac0p340301_L64/configs.npz`,
`--coarse-lattice 32`, the same kernel, and the same tail/shape arguments.
The training driver restores the optimizer state from such a source checkpoint.

The launch script for a new clean run is:

```bash
bash perfect_blocking_upsampling/scripts/submit_lam1p0_l32to64_tailshape_finetune.sh --execute --background
```

## Promoted perfect-blocking kernel

Current kernel:

```text
perfect_blocking/perfect_blocking_lam1p0/kernels/final/chosen_kernel.json
```

It is the D4-symmetric eta-included 5x5 kernel from the final tail-refined
candidate.  `sum(K) = 2^0.125 = 1.0905077326652577`.  Its Fourier conditioning
is good: `min |K(p)| = 0.6847355`, `max |K(p)| = 1.1270750`, condition number
`1.6460` (and `max |1/K| = 1.4604`).

The blocking transformation was selected by random symmetry-preserving screening
followed by bounded derivative-free Powell optimization.  The fitted objective
contains local observables and the correct long-distance
`m2 = (mean_x phi_x)^2`; `m4 = (mean_x phi_x)^4` and IR structure factors were
held out for validation.

## L16 -> L32 upscaling result (good candidate)

Candidate checkpoint:

```text
perfect_blocking_upsampling/runs/lam1p0/training/
lam1p0_L16to32_current5x5_tailstratified_proposal_coverage_N5000_20260803T160559Z/
checkpoints/checkpoint_epoch002.pt
```

Five-thousand-configuration L16 -> L32 **conditional reconstruction diagnostic**:

```text
perfect_blocking_upsampling/runs/lam1p0/upscaling/
raw_L16toL32_current5x5_tailstratified_proposal_coverage_epoch002_N5000_20260803T162000Z
```

Key KS values: action `0.0486`, phi2 `0.0436`, phi4 `0.0786`, local kurtosis
`0.0782`.  The action and kurtosis low-tail coverages are approximately correct
(about 4.5% below the direct q05); the high tails are deliberately somewhat
broad for A/R rejection.  Important: this old diagnostic conditions on L16
fields obtained by blocking the corresponding native L32 configurations; it is
not an independent direct-L16 input test.  The L32 -> L64 evaluation below is
the first genuine independent-coarse comparison, using `--coarse-source`.

Notebook-ready CSVs in that run:

```text
observables/direct_L32_observables_per_config.csv
observables/upscaled_L16_to_L32_observables_per_config.csv
```

## L32 -> L64 evaluation before L64-specific fine-tuning (bad local shape)

This is the genuine independent comparison: direct L32 input ensemble upscaled
to L64 vs direct L64 reference, **not** reblocked native L64 configurations.

```text
perfect_blocking_upsampling/runs/lam1p0/upscaling/
raw_L32toL64_current5x5_tailstratified_proposal_coverage_epoch002_N5000_20260803T164638Z
```

The old L16->L32 flow transfers the infrared observables but not local shape:

```text
action KS 0.1276; only 3.28% below direct action q05 (target 5%)
phi4 KS 0.2046; only 1.52% below direct phi4 q05
local kurtosis KS 0.3284; only 1.06% below direct q05, 23.48% above direct q95
G_pmin KS 0.0162; m2 KS 0.0296
```

CSV files for its existing plots:

```text
observables/direct_L64_observables_per_config.csv
observables/upscaled_L32_to_L64_observables_per_config.csv
```

## Rejected mixture-envelope test

An exact proposal-density mixture was tested on 5,000 independent L32 -> L64
samples: 85% ordinary flow plus 15% with hotter latent Gaussian `T=1.10`.
The mixture density was evaluated exactly and is A/R-valid, but it broadens the
wrong directions and should not be used:

```text
perfect_blocking_upsampling/runs/lam1p0/upscaling/
raw_L32toL64_current5x5_mixture_envelope_w0p15_T1p10_N5000_20260803T190619Z

                         ordinary flow     mixture envelope
action KS                  0.1276             0.2268
phi4 KS                    0.2046             0.2774
local-kurtosis KS          0.3284             0.3688
```

In particular it raised the action standard-deviation ratio to 1.47 and made
the already excessive upper tails substantially worse, without recovering the
missing low tails.  Do not repeat simple symmetric latent-temperature widening.

## Current recommendation before restart

For L32 -> L64, retain the original epoch-2 L16 -> L32 flow as the primary
proposal:

```text
.../lam1p0_L16to32_current5x5_tailstratified_proposal_coverage_N5000_20260803T160559Z/
checkpoints/checkpoint_epoch002.pt
```

The small 5% cold-mixture envelope (`T=0.90`) moves action/phi4/kurtosis tails
in the desired direction, but only marginally improves their KS scores and
worsens several otherwise well-matched observables.  It is therefore optional
for future A/R tail-coverage tests, not a promoted replacement.  The original
flow has the better overall raw distribution; its suppressed low tails can be
handled by exact A/R, subject to effective-sample-size diagnostics.

## Important code changes

- `perfect_blocking_upsampling/scripts/train_lam1p0_l16to32_rqspline_finetune.py`
  is now volume-generic through `--coarse-lattice`; it has been smoke-tested
  for L32 -> L64.  The input fine lattice must be `2 * coarse_lattice`.
- The same trainer supports `--tail-stratified-train`,
  `--proposal-action-lowtail-weight`,
  `--proposal-kurtosis-lowtail-weight`, and the one-sided phi4-width controls.
- `compare_lam1p0_raw_upscaled_vs_native.py` now accepts `--coarse-source`.
  Supplying it performs the genuine independent-coarse comparison; omitting it
  retains the older conditional-reconstruction diagnostic.
- `split_prefixed_observables_csv.py` converts each paired observable table into
  the two standard direct/comparison CSVs used in Jupyter.

## Jupyter plotting reminder

Always reset all three paths for the selected volume:

```python
DIRECT_CSV = RUN_DIR / "observables" / "direct_L64_observables_per_config.csv"
COMPARISON_CSV = RUN_DIR / "observables" / "upscaled_L32_to_L64_observables_per_config.csv"
OUTPUT_DIR = RUN_DIR / "jupyter_figures"
```

For L16 -> L32 the filenames are `direct_L32_observables_per_config.csv` and
`upscaled_L16_to_L32_observables_per_config.csv`.  Do not call the old
`load_observable_comparison(RUN_DIR)` helper unless it is explicitly passed
filenames: its defaults may select the wrong volume.  Read `DIRECT_CSV` and
`COMPARISON_CSV` with `pandas.read_csv` directly, as in the current notebook.

## After the L32 -> L64 fine-tune finishes

1. Read `status.json`, `observables/checkpoint_comparison.csv`, and
   `observables/raw_histogram_metrics.csv`.  Prefer a trained epoch only if
   action/phi4/kurtosis tails improve; NLL alone is not the selection criterion.
2. Evaluate the selected trained checkpoint with direct L32 input and direct L64
   reference, using `compare_lam1p0_raw_upscaled_vs_native.py` with both
   `--native-source ...L64/configs.npz` and `--coarse-source ...L32/configs.npz`.
3. Save the generated L64 fields and split the paired CSV with
   `split_prefixed_observables_csv.py`; then plot in Jupyter.
