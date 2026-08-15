#!/usr/bin/env bash
# Train the L16 -> L32 conditional RQ-spline detail flow for the current
# promoted perfect-blocking kernel.  By default this only prints the command.
# Use --execute to run it, and --background to submit it with nohup.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
if [[ -x "$REPO_ROOT/../../.venv/bin/python" ]]; then
  PYTHON="$REPO_ROOT/../../.venv/bin/python"
elif [[ -x "$REPO_ROOT/../.venv/bin/python" ]]; then
  PYTHON="$REPO_ROOT/../.venv/bin/python"
else
  printf 'Could not find the shared virtual-environment Python.\n' >&2
  exit 1
fi

DRIVER="$REPO_ROOT/perfect_blocking_upsampling/scripts/train_lam1p0_l16to32_rqspline_finetune.py"
KERNEL_PATH="$REPO_ROOT/perfect_blocking/perfect_blocking_lam1p0/kernels/final/chosen_kernel.json"
SOURCE_CHECKPOINT="$REPO_ROOT/perfect_blocking_upsampling/runs/lam1p0/training/lam1p0_L16to32_newkernel_rqspline_N2000_twosided_lowtail_action_phi2_phi4_kurtshape_from_N5000_bestnll_20260719T144834Z/checkpoints/checkpoint_best_patch.pt"
NORMALIZATION_METADATA="$REPO_ROOT/perfect_blocking_upsampling/runs/lam1p0/training/lam1p0_L16to32_newkernel_rqspline_N2000_twosided_lowtail_action_phi2_phi4_kurtshape_from_N5000_bestnll_20260719T144834Z/normalization_metadata.json"

TOTAL_COUNT=5000
TRAIN_COUNT=4000
VAL_COUNT=500
TEST_COUNT=500
EPOCHS=1
SEED=2026080401
OUTPUT_ROOT="$REPO_ROOT/perfect_blocking_upsampling/runs/lam1p0/training"
RUN_LABEL="lam1p0_L16to32_current5x5_phi2_kurtosis_rqspline_N${TOTAL_COUNT}_$(date +%Y%m%dT%H%M%SZ)"
RUN_DIR="$OUTPUT_ROOT/$RUN_LABEL"

EXECUTE=0
BACKGROUND=0
for arg in "$@"; do
  case "$arg" in
    --execute) EXECUTE=1 ;;
    --background) BACKGROUND=1 ;;
    *) printf 'usage: %s [--execute] [--background]\n' "$0" >&2; exit 2 ;;
  esac
done

for required in "$PYTHON" "$DRIVER" "$KERNEL_PATH" "$SOURCE_CHECKPOINT" "$NORMALIZATION_METADATA"; do
  [[ -e "$required" ]] || { printf 'missing required path: %s\n' "$required" >&2; exit 1; }
done

CMD=("$PYTHON" -B "$DRIVER"
  --run-dir "$RUN_DIR"
  --source-checkpoint "$SOURCE_CHECKPOINT"
  --kernel-path "$KERNEL_PATH"
  --normalization-metadata "$NORMALIZATION_METADATA"
  --source-start-index 0
  --total-count "$TOTAL_COUNT" --train-count "$TRAIN_COUNT" --val-count "$VAL_COUNT" --test-count "$TEST_COUNT"
  --epochs "$EPOCHS" --patience 1 --eval-every 1 --exact-eval-every 1 --raw-eval-count 500
  --global-chains 4 --global-sweeps 5 --local-chains 4 --local-sweeps 2
  --batch-size 128 --lr 1e-5 --random-seed "$SEED" --device cpu
  --obs-weights action_density=0.025,phi2=0.02,phi4=0.025,local_kurtosis_ratio=0.04,NN=0.012,2nn=0.004,diag=0.004,G_pmin_avg=0
  --two-sided-tail-guard --action-support-weight 0.025 --phi2-support-weight 0.008 --phi4-support-weight 0.025
  --tail-guard-std-weight 0.05 --tail-guard-quantile-weight 0.25 --tail-guard-occupancy-weight 0.5
  --tail-guard-low-occupancy-weight 0.75 --tail-guard-high-occupancy-weight 0.5
  --local-kurtosis-shape-guard --local-kurtosis-shape-weight 0.012
  --stop-after-eval-epoch 1)

printf 'run directory: %s\n' "$RUN_DIR"
printf 'command: '
printf '%q ' "${CMD[@]}"
printf '\n'
if [[ "$EXECUTE" -eq 0 ]]; then
  printf 'Prepared only. Re-run with --execute to start; add --background for nohup.\n'
  exit 0
fi

mkdir -p "$RUN_DIR/logs"
{
  printf 'run_id=%s\n' "$RUN_LABEL"
  printf 'kernel=%s\n' "$KERNEL_PATH"
  printf 'command='
  printf '%q ' "${CMD[@]}"
  printf '\n'
} > "$RUN_DIR/submit_manifest.txt"
if [[ "$BACKGROUND" -eq 1 ]]; then
  nohup "${CMD[@]}" > "$RUN_DIR/logs/run.log" 2>&1 &
  printf '%s\n' "$!" > "$RUN_DIR/submit_pid.txt"
  printf 'started background PID %s\n' "$!"
else
  exec "${CMD[@]}"
fi
