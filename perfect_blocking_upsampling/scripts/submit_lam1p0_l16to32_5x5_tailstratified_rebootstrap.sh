#!/usr/bin/env bash
# Rebuild the successful August-3 L16->L32 5x5 flow procedure.
#
# Stage 1 recreates the one-epoch 5x5 base finetune; stage 2 applies the
# four-epoch tail-stratified proposal-coverage finetune.  By default this
# script only prints the fully resolved run.  Invoke with --execute from a
# persistent terminal to run it.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="$ROOT/../../.venv/bin/python"
DRIVER="$ROOT/perfect_blocking_upsampling/scripts/train_lam1p0_l16to32_rqspline_finetune.py"
KERNEL="$ROOT/perfect_blocking/perfect_blocking_lam1p0/kernels/final/chosen_kernel.json"

# The original August-3 L16->L32 base checkpoint was pruned.  This surviving
# L16->L32 RQ-spline checkpoint has the current checkpoint schema and was
# trained with a 5x5 kernel.  Stage 1 recomputes normalization from the chosen
# 5x5 L32 pairs instead of reusing the lost normalization metadata.
SOURCE_CHECKPOINT="$ROOT/perfect_blocking_upsampling/runs/lam1p0/training/lam1p0_L16to32_alternatingKL_highcorr5_r1_iteration5/flow5/stage_oo/checkpoints/checkpoint_best_nll.pt"

TOTAL_COUNT="${TOTAL_COUNT:-5000}"
TRAIN_COUNT="${TRAIN_COUNT:-4000}"
VAL_COUNT="${VAL_COUNT:-500}"
TEST_COUNT="${TEST_COUNT:-500}"
BASE_EPOCHS="${BASE_EPOCHS:-1}"
TAIL_EPOCHS="${TAIL_EPOCHS:-4}"
RUN_LABEL="${RUN_LABEL:-lam1p0_L16to32_5x5_tailstratified_rebootstrap_N${TOTAL_COUNT}_$(date +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${RUN_ROOT:-$ROOT/perfect_blocking_upsampling/runs/lam1p0/training/$RUN_LABEL}"

EXECUTE=0
BACKGROUND=0
for arg in "$@"; do
  case "$arg" in
    --execute) EXECUTE=1 ;;
    --background) BACKGROUND=1 ;;
    *) echo "usage: $0 [--execute] [--background]" >&2; exit 2 ;;
  esac
done
[[ "$BACKGROUND" -eq 0 || "$EXECUTE" -eq 1 ]] || { echo "--background requires --execute" >&2; exit 2; }

[[ $((TRAIN_COUNT + VAL_COUNT + TEST_COUNT)) -eq "$TOTAL_COUNT" ]] || { echo "split counts must sum to TOTAL_COUNT" >&2; exit 2; }
for required in "$PYTHON" "$DRIVER" "$KERNEL" "$SOURCE_CHECKPOINT"; do
  [[ -e "$required" ]] || { echo "missing required path: $required" >&2; exit 1; }
done

BASE_RUN="$RUN_ROOT/stage_base_5x5"
TAIL_RUN="$RUN_ROOT/stage_tailstratified_5x5"
BASE_CMD=("$PYTHON" -u -B "$DRIVER"
  --run-dir "$BASE_RUN" --source-checkpoint "$SOURCE_CHECKPOINT" --kernel-path "$KERNEL" --source-start-index 0
  --total-count "$TOTAL_COUNT" --train-count "$TRAIN_COUNT" --val-count "$VAL_COUNT" --test-count "$TEST_COUNT"
  --epochs "$BASE_EPOCHS" --patience 1 --eval-every 1 --exact-eval-every 1 --raw-eval-count 500
  --global-chains 4 --global-sweeps 5 --local-chains 4 --local-sweeps 2 --batch-size 128 --lr 1e-5 --random-seed 2026080401 --device cpu
  --obs-weights action_density=0.025,phi2=0.020,phi4=0.025,local_kurtosis_ratio=0.040,NN=0.012,2nn=0.004,diag=0.004,G_pmin_avg=0
  --two-sided-tail-guard --action-support-weight 0.025 --phi2-support-weight 0.008 --phi4-support-weight 0.025
  --tail-guard-std-weight 0.05 --tail-guard-quantile-weight 0.25 --tail-guard-occupancy-weight 0.5
  --tail-guard-low-occupancy-weight 0.75 --tail-guard-high-occupancy-weight 0.5
  --local-kurtosis-shape-guard --local-kurtosis-shape-weight 0.012 --stop-after-eval-epoch 1)

TAIL_CMD=("$PYTHON" -u -B "$DRIVER"
  --run-dir "$TAIL_RUN" --source-checkpoint "$BASE_RUN/checkpoints/checkpoint_best_nll.pt" --kernel-path "$KERNEL" --normalization-metadata "$BASE_RUN/normalization_metadata.json" --source-start-index 0
  --total-count "$TOTAL_COUNT" --train-count "$TRAIN_COUNT" --val-count "$VAL_COUNT" --test-count "$TEST_COUNT"
  --epochs "$TAIL_EPOCHS" --patience 2 --eval-every 1 --exact-eval-every 2 --raw-eval-count 500
  --global-chains 4 --global-sweeps 5 --local-chains 4 --local-sweeps 2 --batch-size 128 --lr 5e-6 --random-seed 2026080417 --device cpu
  --obs-weights action_density=0.025,phi2=0.020,phi4=0.025,local_kurtosis_ratio=0.020,NN=0.012,2nn=0.004,diag=0.004,G_pmin_avg=0
  --tail-stratified-train --tail-stratified-quantile 0.10 --tail-stratified-tail-fraction 0.40
  --proposal-action-lowtail-weight 0.10 --proposal-kurtosis-lowtail-weight 0.15)

printf 'run_root=%s\nsource_checkpoint=%s\nkernel=%s\n' "$RUN_ROOT" "$SOURCE_CHECKPOINT" "$KERNEL"
printf 'base command='; printf '%q ' "${BASE_CMD[@]}"; printf '\n'
printf 'tail command='; printf '%q ' "${TAIL_CMD[@]}"; printf '\n'
if [[ "$EXECUTE" -eq 0 ]]; then
  echo 'Prepared only. Run with --execute from a persistent terminal; add --background to detach.'
  exit 0
fi
if [[ "$BACKGROUND" -eq 1 ]]; then
  [[ ! -e "$RUN_ROOT" ]] || { echo "refusing to overwrite existing run directory: $RUN_ROOT" >&2; exit 1; }
  mkdir -p "$RUN_ROOT/logs"
  nohup env \
    RUN_LABEL="$RUN_LABEL" RUN_ROOT="$RUN_ROOT" \
    TOTAL_COUNT="$TOTAL_COUNT" TRAIN_COUNT="$TRAIN_COUNT" VAL_COUNT="$VAL_COUNT" TEST_COUNT="$TEST_COUNT" \
    BASE_EPOCHS="$BASE_EPOCHS" TAIL_EPOCHS="$TAIL_EPOCHS" ALLOW_EXISTING_LAUNCH_ROOT=1 \
    bash "$0" --execute > "$RUN_ROOT/logs/run.log" 2>&1 < /dev/null &
  echo "$!" > "$RUN_ROOT/submit_pid.txt"
  printf 'background_pid=%s\nrun_root=%s\nlog=%s\n' "$!" "$RUN_ROOT" "$RUN_ROOT/logs/run.log"
  exit 0
fi
[[ ! -e "$RUN_ROOT" || "${ALLOW_EXISTING_LAUNCH_ROOT:-0}" == 1 ]] || { echo "refusing to overwrite existing run directory: $RUN_ROOT" >&2; exit 1; }
mkdir -p "$RUN_ROOT/logs"
{
  echo "run_id=$RUN_LABEL"
  echo 'procedure=5x5 base one-epoch finetune followed by 5x5 tail-stratified proposal-coverage finetune'
  echo 'deviation=original August-3 base checkpoint and normalization were pruned; stage_base recomputes 5x5 normalization from a surviving compatible 5x5 L16->L32 source checkpoint'
  printf 'base_command='; printf '%q ' "${BASE_CMD[@]}"; printf '\n'
  printf 'tail_command='; printf '%q ' "${TAIL_CMD[@]}"; printf '\n'
} > "$RUN_ROOT/submit_manifest.txt"

"${BASE_CMD[@]}" 2>&1 | tee "$RUN_ROOT/logs/stage_base.log"
"${TAIL_CMD[@]}" 2>&1 | tee "$RUN_ROOT/logs/stage_tailstratified.log"
printf '%s\n' "$TAIL_RUN/checkpoints/checkpoint_best_nll.pt" > "$RUN_ROOT/final_checkpoint.txt"
