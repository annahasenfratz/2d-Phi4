#!/usr/bin/env bash
# EO -> OE -> OO conditional-flow training for the high-correlation 5x5 kernel.
# The objective is exactly conditional NLL: every observable-related weight is zero.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="$ROOT/../../.venv/bin/python"
DRIVER="$ROOT/perfect_blocking_upsampling/scripts/train_lam1p0_l16to32_rqspline_finetune.py"

RUN_ROOT="${RUN_ROOT:?RUN_ROOT is required}"; SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:?SOURCE_CHECKPOINT is required}"
KERNEL="${KERNEL:?KERNEL is required}"; TOTAL_COUNT="${TOTAL_COUNT:?TOTAL_COUNT is required}"
TRAIN_COUNT="${TRAIN_COUNT:?TRAIN_COUNT is required}"; VAL_COUNT="${VAL_COUNT:?VAL_COUNT is required}"; TEST_COUNT="${TEST_COUNT:?TEST_COUNT is required}"
EPOCHS="${EPOCHS:?EPOCHS is required}"; PATIENCE="${PATIENCE:?PATIENCE is required}"; LR="${LR:?LR is required}"
BATCH_SIZE="${BATCH_SIZE:?BATCH_SIZE is required}"; SEED="${SEED:?SEED is required}"; DEVICE="${DEVICE:?DEVICE is required}"
ZERO_OBS_WEIGHTS="action_density=0,phi2=0,phi4=0,local_kurtosis_ratio=0,NN=0,2nn=0,diag=0,G_pmin_avg=0"

run_stage() {
  local stage="$1" source="$2" normalization="$3" stage_dir="$RUN_ROOT/stage_$1"
  local cmd=("$PYTHON" -B "$DRIVER" --run-dir "$stage_dir" --source-checkpoint "$source" --kernel-path "$KERNEL"
    --source-start-index 0 --total-count "$TOTAL_COUNT" --train-count "$TRAIN_COUNT" --val-count "$VAL_COUNT" --test-count "$TEST_COUNT"
    --epochs "$EPOCHS" --patience "$PATIENCE" --eval-every 1 --raw-eval-count 500 --global-chains 4 --global-sweeps 5 --local-chains 4 --local-sweeps 2
    --batch-size "$BATCH_SIZE" --lr "$LR" --random-seed "$SEED" --device "$DEVICE" --train-stage "$stage" --obs-weights "$ZERO_OBS_WEIGHTS")
  [[ -z "$normalization" ]] || cmd+=(--normalization-metadata "$normalization")
  "${cmd[@]}"
}

mkdir -p "$RUN_ROOT"
if [[ ! -f "$RUN_ROOT/stage_eo/checkpoints/checkpoint_best_nll.pt" ]]; then run_stage eo "$SOURCE_CHECKPOINT" ""; fi
NORMALIZATION="$RUN_ROOT/stage_eo/normalization_metadata.json"
if [[ ! -f "$RUN_ROOT/stage_oe/checkpoints/checkpoint_best_nll.pt" ]]; then run_stage oe "$RUN_ROOT/stage_eo/checkpoints/checkpoint_best_nll.pt" "$NORMALIZATION"; fi
if [[ ! -f "$RUN_ROOT/stage_oo/checkpoints/checkpoint_best_nll.pt" ]]; then run_stage oo "$RUN_ROOT/stage_oe/checkpoints/checkpoint_best_nll.pt" "$NORMALIZATION"; fi
printf '%s\n' "$RUN_ROOT/stage_oo/checkpoints/checkpoint_best_nll.pt" > "$RUN_ROOT/final_checkpoint.txt"
