#!/usr/bin/env bash
# Sequential EO -> OE -> OO pure conditional-NLL L16->L32 retraining with the
# promoted Aug-3 5x5 kernel.  This deliberately omits all generated-observable,
# tail, and width penalties.
# Usage: bash $0 [--execute] [--background]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
if [[ -x "$ROOT/../../.venv/bin/python" ]]; then
  PYTHON="$ROOT/../../.venv/bin/python"
elif [[ -x "$ROOT/../.venv/bin/python" ]]; then
  PYTHON="$ROOT/../.venv/bin/python"
else
  echo "shared Python not found" >&2
  exit 1
fi

DRIVER="$ROOT/perfect_blocking_upsampling/scripts/train_lam1p0_l16to32_rqspline_finetune.py"
STAGE_DRIVER="$ROOT/perfect_blocking_upsampling/scripts/run_lam1p0_l16to32_newkernel_pure_nll_stages.sh"
SOURCE_RUN="$ROOT/perfect_blocking_upsampling/runs/lam1p0/training/lam1p0_L16to32_current5x5_tailstratified_proposal_coverage_N5000_20260803T160559Z"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-$SOURCE_RUN/checkpoints/checkpoint_best_nll.pt}"
NORMALIZATION="$SOURCE_RUN/normalization_metadata.json"
# This is yesterday's promoted/new kernel, intentionally not the July archive.
KERNEL="$ROOT/perfect_blocking/perfect_blocking_lam1p0/kernels/final/chosen_kernel.json"

TOTAL_COUNT="${TOTAL_COUNT:-5000}"
TRAIN_COUNT="${TRAIN_COUNT:-4000}"
VAL_COUNT="${VAL_COUNT:-500}"
TEST_COUNT="${TEST_COUNT:-500}"
EPOCHS="${EPOCHS:-12}"
PATIENCE="${PATIENCE:-4}"
LR="${LR:-1e-5}"
BATCH_SIZE="${BATCH_SIZE:-128}"
SEED="${SEED:-2026080501}"
DEVICE="${DEVICE:-cpu}"
RUN_LABEL="lam1p0_L16to32_current5x5_pureNLL_from_tailstratifiedNLL_N${TOTAL_COUNT}_$(date +%Y%m%dT%H%M%SZ)"
RUN_DIR="${RESUME_RUN:-$ROOT/perfect_blocking_upsampling/runs/lam1p0/training/$RUN_LABEL}"

EXECUTE=0
BACKGROUND=0
for arg in "$@"; do
  case "$arg" in
    --execute) EXECUTE=1 ;;
    --background) BACKGROUND=1 ;;
    *) echo "usage: $0 [--execute] [--background]" >&2; exit 2 ;;
  esac
done
if [[ "$BACKGROUND" -eq 1 && "$EXECUTE" -eq 0 ]]; then
  echo "--background requires --execute" >&2
  exit 2
fi
for path in "$PYTHON" "$DRIVER" "$STAGE_DRIVER" "$SOURCE_CHECKPOINT" "$NORMALIZATION" "$KERNEL"; do
  [[ -e "$path" ]] || { echo "missing: $path" >&2; exit 1; }
done

CMD=(
  env
  "RUN_ROOT=$RUN_DIR"
  "SOURCE_CHECKPOINT=$SOURCE_CHECKPOINT"
  "NORMALIZATION=$NORMALIZATION"
  "KERNEL=$KERNEL"
  "TOTAL_COUNT=$TOTAL_COUNT" "TRAIN_COUNT=$TRAIN_COUNT" "VAL_COUNT=$VAL_COUNT" "TEST_COUNT=$TEST_COUNT"
  "EPOCHS=$EPOCHS" "PATIENCE=$PATIENCE" "LR=$LR" "BATCH_SIZE=$BATCH_SIZE" "SEED=$SEED" "DEVICE=$DEVICE"
  bash "$STAGE_DRIVER"
)

printf 'run_dir=%s\n' "$RUN_DIR"
printf 'command='; printf '%q ' "${CMD[@]}"; printf '\n'
if [[ "$EXECUTE" -eq 0 ]]; then
  echo "Prepared only. Add --execute to start; add --background for nohup."
  exit 0
fi

mkdir -p "$RUN_DIR/logs"
{
  echo "run_id=$(basename "$RUN_DIR")"
  echo "schedule=EO -> OE -> OO; each stage starts from the preceding best-NLL checkpoint"
  echo "objective=pure conditional NLL; all observable/tail/width penalties are explicitly zero"
  echo "kernel=promoted Aug-3 5x5 chosen_kernel.json"
  echo "source_checkpoint=$SOURCE_CHECKPOINT"
  printf 'command='; printf '%q ' "${CMD[@]}"; printf '\n'
} > "$RUN_DIR/submit_manifest.txt"
if [[ "$BACKGROUND" -eq 1 ]]; then
  nohup "${CMD[@]}" > "$RUN_DIR/logs/run.log" 2>&1 < /dev/null &
  echo "$!" > "$RUN_DIR/submit_pid.txt"
  printf 'background_pid=%s\nlog=%s\n' "$!" "$RUN_DIR/logs/run.log"
else
  exec "${CMD[@]}"
fi
