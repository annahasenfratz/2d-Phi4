#!/usr/bin/env bash
# Frozen base-highcorr 5x5 conditional-flow retraining.
# Usage: bash $0 [--execute] [--background]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
STAGE_DRIVER="$ROOT/perfect_blocking_upsampling/scripts/run_lam1p0_l16to32_highcorr5_pure_nll_stages.sh"
KERNEL="$ROOT/perfect_blocking/perfect_blocking_lam1p0/kernels/candidates/allL16_chi2_R2_corrW5000_highcorr_5x5_eta_included.json"
SOURCE_CHECKPOINT="$ROOT/perfect_blocking_upsampling/quarantine_legacy_gaussian_mh_20260813/training/lam1p0_L16to32_highcorr5_pureNLL_N5000_20260807T063341Z/stage_oo/checkpoints/checkpoint_best_nll.pt"

TOTAL_COUNT="${TOTAL_COUNT:-10000}"
TRAIN_COUNT="${TRAIN_COUNT:-8000}"
VAL_COUNT="${VAL_COUNT:-1000}"
TEST_COUNT="${TEST_COUNT:-1000}"
EPOCHS="${EPOCHS:-20}"
PATIENCE="${PATIENCE:-6}"
LR="${LR:-1e-5}"
BATCH_SIZE="${BATCH_SIZE:-128}"
EVAL_EVERY="${EVAL_EVERY:-5}"
SEED="${SEED:-2026081805}"
DEVICE="${DEVICE:-cpu}"
RUN_LABEL="lam1p0_L16to32_base_highcorr5_pureNLL_retrain_N${TOTAL_COUNT}_$(date +%Y%m%dT%H%M%SZ)"
RUN_DIR="${RUN_DIR:-$ROOT/perfect_blocking_upsampling/runs/lam1p0/training/$RUN_LABEL}"

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
[[ -f "$KERNEL" ]] || { echo "missing kernel: $KERNEL" >&2; exit 1; }
[[ -f "$SOURCE_CHECKPOINT" ]] || { echo "missing source checkpoint: $SOURCE_CHECKPOINT" >&2; exit 1; }

CMD=(env "RUN_ROOT=$RUN_DIR" "SOURCE_CHECKPOINT=$SOURCE_CHECKPOINT" "KERNEL=$KERNEL"
  "TOTAL_COUNT=$TOTAL_COUNT" "TRAIN_COUNT=$TRAIN_COUNT" "VAL_COUNT=$VAL_COUNT" "TEST_COUNT=$TEST_COUNT"
  "EPOCHS=$EPOCHS" "PATIENCE=$PATIENCE" "LR=$LR" "BATCH_SIZE=$BATCH_SIZE" "EVAL_EVERY=$EVAL_EVERY" "SEED=$SEED" "DEVICE=$DEVICE"
  bash "$STAGE_DRIVER")

printf 'run_dir=%s\n' "$RUN_DIR"
printf 'kernel=%s\nsource_checkpoint=%s\n' "$KERNEL" "$SOURCE_CHECKPOINT"
printf 'objective=pure conditional NLL (all observable and tail weights are zero)\n'
printf 'diagnostic schedule=every %s epochs, plus the final epoch\n' "$EVAL_EVERY"
printf 'command='; printf '%q ' "${CMD[@]}"; printf '\n'
[[ "$EXECUTE" -eq 1 ]] || exit 0

mkdir -p "$RUN_DIR/logs"
{
  printf 'kernel=%s\n' "$KERNEL"
  printf 'kernel_sha256='; shasum -a 256 "$KERNEL" | awk '{print $1}'
  printf 'source_checkpoint=%s\n' "$SOURCE_CHECKPOINT"
  printf 'objective=pure conditional NLL; all observable and tail weights zero\n'
  printf 'split=%s/%s/%s of %s; seed=%s\n' "$TRAIN_COUNT" "$VAL_COUNT" "$TEST_COUNT" "$TOTAL_COUNT" "$SEED"
  printf 'diagnostic_schedule=every %s epochs, plus final epoch\n' "$EVAL_EVERY"
  printf 'command='; printf '%q ' "${CMD[@]}"; printf '\n'
} > "$RUN_DIR/submit_manifest.txt"

if [[ "$BACKGROUND" -eq 1 ]]; then
  nohup "${CMD[@]}" > "$RUN_DIR/logs/run.log" 2>&1 < /dev/null &
  echo $! > "$RUN_DIR/submit_pid.txt"
  printf 'background_pid=%s\nlog=%s\n' "$!" "$RUN_DIR/logs/run.log"
else
  exec "${CMD[@]}"
fi
