#!/usr/bin/env bash
# Clean, from-scratch control for Ethan's fixed 7x7 kernel.
# Usage: bash $0 [--execute] [--background]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="$ROOT/../../.venv/bin/python"
DRIVER="$ROOT/perfect_blocking_upsampling/scripts/train_lam1p0_l16to32_rqspline_finetune.py"
KERNEL="$ROOT/perfect_blocking/perfect_blocking_lam1p0/kernels/selected_for_upscaling/ethan_7x7_paper_objective_eta_included.json"

# This is deliberately a fresh model: no 5x5 or other-7x7 weights enter EO.
TOTAL_COUNT="${TOTAL_COUNT:-10000}"; TRAIN_COUNT="${TRAIN_COUNT:-8000}"
VAL_COUNT="${VAL_COUNT:-1000}"; TEST_COUNT="${TEST_COUNT:-1000}"
EPOCHS="${EPOCHS:-40}"; PATIENCE="${PATIENCE:-10}"; LR="${LR:-5e-5}"
BATCH_SIZE="${BATCH_SIZE:-128}"; EVAL_EVERY="${EVAL_EVERY:-5}"
SEED="${SEED:-2026081811}"; DEVICE="${DEVICE:-cpu}"
RUN_LABEL="lam1p0_L16to32_ethan7_fresh_pureNLL_control_N${TOTAL_COUNT}_$(date +%Y%m%dT%H%M%SZ)"
RUN_DIR="${RUN_DIR:-$ROOT/perfect_blocking_upsampling/runs/lam1p0/training/$RUN_LABEL}"
ZERO_OBS_WEIGHTS="action_density=0,phi2=0,phi4=0,local_kurtosis_ratio=0,NN=0,2nn=0,diag=0,G_pmin_avg=0"

EXECUTE=0; BACKGROUND=0
for arg in "$@"; do
  case "$arg" in
    --execute) EXECUTE=1 ;;
    --background) BACKGROUND=1 ;;
    *) echo "usage: $0 [--execute] [--background]" >&2; exit 2 ;;
  esac
done
[[ "$BACKGROUND" -eq 0 || "$EXECUTE" -eq 1 ]] || { echo "--background requires --execute" >&2; exit 2; }
for path in "$PYTHON" "$DRIVER" "$KERNEL"; do [[ -e "$path" ]] || { echo "missing: $path" >&2; exit 1; }; done

run_stage() {
  local stage="$1" mode="$2" source="$3" normalization="$4" stage_dir="$RUN_DIR/stage_$1"
  local cmd=("$PYTHON" -B "$DRIVER" --run-dir "$stage_dir" --kernel-path "$KERNEL"
    --initialization-mode "$mode" --source-start-index 0
    --total-count "$TOTAL_COUNT" --train-count "$TRAIN_COUNT" --val-count "$VAL_COUNT" --test-count "$TEST_COUNT"
    --epochs "$EPOCHS" --patience "$PATIENCE" --eval-every "$EVAL_EVERY" --raw-eval-count 500
    --global-chains 4 --global-sweeps 5 --local-chains 4 --local-sweeps 0
    --batch-size "$BATCH_SIZE" --lr "$LR" --random-seed "$SEED" --device "$DEVICE"
    --train-stage "$stage" --obs-weights "$ZERO_OBS_WEIGHTS")
  [[ "$mode" == fresh ]] || cmd+=(--source-checkpoint "$source")
  [[ -z "$normalization" ]] || cmd+=(--normalization-metadata "$normalization")
  "${cmd[@]}"
}

run_all() {
  mkdir -p "$RUN_DIR"
  if [[ ! -f "$RUN_DIR/stage_eo/checkpoints/checkpoint_best_nll.pt" ]]; then run_stage eo fresh "" ""; fi
  local normalization="$RUN_DIR/stage_eo/normalization_metadata.json"
  [[ -f "$normalization" ]] || { echo "EO normalization metadata is missing" >&2; exit 1; }
  if [[ ! -f "$RUN_DIR/stage_oe/checkpoints/checkpoint_best_nll.pt" ]]; then run_stage oe transferred "$RUN_DIR/stage_eo/checkpoints/checkpoint_best_nll.pt" "$normalization"; fi
  if [[ ! -f "$RUN_DIR/stage_oo/checkpoints/checkpoint_best_nll.pt" ]]; then run_stage oo transferred "$RUN_DIR/stage_oe/checkpoints/checkpoint_best_nll.pt" "$normalization"; fi
  printf '%s\n' "$RUN_DIR/stage_oo/checkpoints/checkpoint_best_nll.pt" > "$RUN_DIR/final_checkpoint.txt"
}

printf 'run_dir=%s\n' "$RUN_DIR"
printf 'kernel=%s\n' "$KERNEL"
printf 'objective=pure conditional NLL; all observable and tail weights zero\n'
printf 'initialization=fresh EO; OE/OO inherit only newly trained Ethan-7x7 stages\n'
printf 'split=%s/%s/%s of %s; epochs=%s; patience=%s; lr=%s\n' "$TRAIN_COUNT" "$VAL_COUNT" "$TEST_COUNT" "$TOTAL_COUNT" "$EPOCHS" "$PATIENCE" "$LR"
printf 'diagnostic_schedule=every %s epochs, plus final epoch\n' "$EVAL_EVERY"
[[ "$EXECUTE" -eq 1 ]] || { echo "Prepared only. Add --execute to start."; exit 0; }

mkdir -p "$RUN_DIR/logs"
{
  printf 'kernel=%s\n' "$KERNEL"
  printf 'kernel_sha256='; shasum -a 256 "$KERNEL" | awk '{print $1}'
  printf 'objective=pure conditional NLL; all observable and tail weights zero\n'
  printf 'initialization=fresh EO; OE/OO only inherit newly trained stages\n'
  printf 'split=%s/%s/%s of %s; epochs=%s; patience=%s; lr=%s; seed=%s\n' "$TRAIN_COUNT" "$VAL_COUNT" "$TEST_COUNT" "$TOTAL_COUNT" "$EPOCHS" "$PATIENCE" "$LR" "$SEED"
  printf 'diagnostic_schedule=every %s epochs, plus final epoch\n' "$EVAL_EVERY"
} > "$RUN_DIR/submit_manifest.txt"
if [[ "$BACKGROUND" -eq 1 ]]; then
  nohup env RUN_DIR="$RUN_DIR" TOTAL_COUNT="$TOTAL_COUNT" TRAIN_COUNT="$TRAIN_COUNT" VAL_COUNT="$VAL_COUNT" TEST_COUNT="$TEST_COUNT" \
    EPOCHS="$EPOCHS" PATIENCE="$PATIENCE" LR="$LR" BATCH_SIZE="$BATCH_SIZE" EVAL_EVERY="$EVAL_EVERY" SEED="$SEED" DEVICE="$DEVICE" \
    bash "$0" --execute > "$RUN_DIR/logs/run.log" 2>&1 < /dev/null &
  echo $! > "$RUN_DIR/submit_pid.txt"
  printf 'background_pid=%s\nlog=%s\n' "$!" "$RUN_DIR/logs/run.log"
else
  run_all
fi
