#!/usr/bin/env bash
# Conditional-flow refinement for the fixed alternating-KL iteration-5 kernel.
# NLL remains the main objective.  Modest generated-observable and low-tail
# coverage terms target the blocked-native+flow action mismatch.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="$ROOT/../../.venv/bin/python"
DRIVER="$ROOT/perfect_blocking_upsampling/scripts/train_lam1p0_l16to32_rqspline_finetune.py"

RUN_ROOT="${RUN_ROOT:?RUN_ROOT is required}"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:?SOURCE_CHECKPOINT is required}"
KERNEL="${KERNEL:?KERNEL is required}"
TOTAL_COUNT="${TOTAL_COUNT:-5000}"
TRAIN_COUNT="${TRAIN_COUNT:-4000}"
VAL_COUNT="${VAL_COUNT:-500}"
TEST_COUNT="${TEST_COUNT:-500}"
EPOCHS="${EPOCHS:-12}"
PATIENCE="${PATIENCE:-4}"
LR="${LR:-1e-5}"
BATCH_SIZE="${BATCH_SIZE:-128}"
SEED="${SEED:-2026081217}"
DEVICE="${DEVICE:-cpu}"

# These are deliberately small relative to the conditional NLL.  They are
# evaluated on fresh q(d|c) draws, so they address the failed conditional-flow
# control rather than fitting paired detail fields site-by-site.
OBS_WEIGHTS="action_density=0.025,phi2=0.008,phi4=0.012,local_kurtosis_ratio=0.020,NN=0.006,2nn=0,diag=0,G_pmin_avg=0"

run_stage() {
  local stage="$1" source="$2" normalization="$3" stage_dir="$RUN_ROOT/stage_$1"
  local cmd=("$PYTHON" -B "$DRIVER" --run-dir "$stage_dir"
    --source-checkpoint "$source" --kernel-path "$KERNEL"
    --source-start-index 0 --total-count "$TOTAL_COUNT" --train-count "$TRAIN_COUNT" --val-count "$VAL_COUNT" --test-count "$TEST_COUNT"
    --epochs "$EPOCHS" --patience "$PATIENCE" --eval-every 1 --exact-eval-every 2 --raw-eval-count 500
    --global-chains 4 --global-sweeps 5 --local-chains 4 --local-sweeps 2
    --batch-size "$BATCH_SIZE" --lr "$LR" --random-seed "$SEED" --device "$DEVICE" --train-stage "$stage"
    --obs-weights "$OBS_WEIGHTS"
    --tail-stratified-train --tail-stratified-quantile 0.10 --tail-stratified-tail-fraction 0.35
    --proposal-action-lowtail-weight 0.10 --proposal-kurtosis-lowtail-weight 0.10)
  [[ -z "$normalization" ]] || cmd+=(--normalization-metadata "$normalization")
  "${cmd[@]}"
}

mkdir -p "$RUN_ROOT"
if [[ ! -f "$RUN_ROOT/stage_eo/checkpoints/checkpoint_best_patch.pt" ]]; then run_stage eo "$SOURCE_CHECKPOINT" ""; fi
NORMALIZATION="$RUN_ROOT/stage_eo/normalization_metadata.json"
if [[ ! -f "$RUN_ROOT/stage_oe/checkpoints/checkpoint_best_patch.pt" ]]; then run_stage oe "$RUN_ROOT/stage_eo/checkpoints/checkpoint_best_patch.pt" "$NORMALIZATION"; fi
if [[ ! -f "$RUN_ROOT/stage_oo/checkpoints/checkpoint_best_patch.pt" ]]; then run_stage oo "$RUN_ROOT/stage_oe/checkpoints/checkpoint_best_patch.pt" "$NORMALIZATION"; fi
printf '%s\n' "$RUN_ROOT/stage_oo/checkpoints/checkpoint_best_patch.pt" > "$RUN_ROOT/final_checkpoint.txt"
