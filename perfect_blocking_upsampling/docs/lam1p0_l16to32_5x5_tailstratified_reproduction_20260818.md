# L16 -> L32 5x5 tail-stratified flow reproduction — 2026-08-18

## Purpose

Reproduce the August-3 5x5 L16 -> L32 flow recipe before applying the same
training procedure to Ethan's 7x7 kernel.  This run is a procedural control:
it checks that the reported good conditional-reconstruction histograms can be
recreated with the current code and data.

## Original successful tail-stratified stage

The archived launch script was
`scripts/submit_lam1p0_l16to32_tailstratified_proposal_coverage.sh`.  Its
settings were:

- eta-included promoted 5x5 kernel: `kernels/final/chosen_kernel.json`;
- native L32 matched pairs, split 4000 / 500 / 500 from 5000 configurations;
- four epochs, patience two, batch size 128, learning rate `5e-6`;
- observable weights: action `0.025`, phi2 `0.020`, phi4 `0.025`, local
  kurtosis `0.020`, NN `0.012`, 2NN and diagonal `0.004`;
- tail-stratified batches: 40% drawn from the union of the lowest action and
  local-kurtosis deciles;
- one-sided proposal-coverage weights: action `0.10`, local kurtosis `0.15`.

The selected historical checkpoint was epoch two.  Its five-thousand-sample
diagnostic was conditional reconstruction: L16 fields came from blocking the
corresponding native L32 fields.  It is not an independent native-L16 test.

## Reproduction command

Run from a persistent terminal:

```bash
bash perfect_blocking_upsampling/scripts/submit_lam1p0_l16to32_5x5_tailstratified_rebootstrap.sh --execute
```

The wrapper has two stages.

1. `stage_base_5x5` reproduces the one-epoch base 5x5 finetune, including its
   two-sided support and local-kurtosis shape guards.
2. `stage_tailstratified_5x5` applies the original four-epoch tail-stratified
   proposal-coverage procedure and writes `final_checkpoint.txt`.

## Unavoidable deviation

The original base checkpoint
`lam1p0_L16to32_current5x5_phi2_kurtosis_rqspline_N5000_20260803T121013Z`
and its normalization metadata are no longer in the workspace or mounted
backup.  The wrapper starts from the surviving compatible 5x5 L16 -> L32
RQ-spline checkpoint (`alternatingKL_highcorr5_r1_iteration5/flow5/stage_oo`)
and recomputes normalization from native L32 configurations blocked with the
chosen 5x5 kernel.  This is explicitly recorded in the run manifest.
No 7x7 training should be compared with this control until the 5x5 conditional
and independent-coarse raw evaluations have both been completed.

## Required evaluations after training

1. Conditional reconstruction: omit `--coarse-source` in
   `compare_lam1p0_raw_upscaled_vs_native.py`.
2. Independent initialization: supply native L16 as `--coarse-source` and
   native L32 as `--native-source`.
3. Select a checkpoint using histogram/tail metrics, not validation NLL alone.
