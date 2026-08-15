#!/usr/bin/env bash
# QUARANTINED 2026-08-04: contiguous-patch thermalization was superseded by
# checkerboard-coordinate MH. Historical code and outputs are retained only
# for provenance; use submit_lam1p0_coordinate_mh.sh for new runs.
echo "This contiguous-patch launcher is quarantined. Use: bash perfect_blocking_upsampling/scripts/submit_lam1p0_checkerboard_mh.sh r1 --execute --background" >&2
exit 2

# Prepare or execute one lambda=1.0 flow-initialized MH run.
# Usage: bash perfect_blocking_upsampling/scripts/submit_lam1p0_mh.sh 16 32 r1 [--execute] [--background]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
PYTHON="$REPO_ROOT/../../.venv/bin/python"
DRIVER="perfect_blocking_upsampling/scripts/run_flow_detail_coarse_detail.py"
SEED_BASE=2126072867

LC="${1:-}"
LF="${2:-}"
RUN_NUMBER="${3:-}"
if [[ ! "$LC" =~ ^(16|32)$ || ! "$LF" =~ ^(32|64)$ || $((2 * LC)) -ne "$LF" || ! "$RUN_NUMBER" =~ ^r([1-9][0-9]*)$ ]]; then
  echo "usage: $0 <Lc> <Lf> rN [--execute] [--background]" >&2
  echo "examples: $0 16 32 r1 --execute --background" >&2
  echo "          $0 32 64 r1 --execute --background" >&2
  exit 2
fi
shift 3

case "${LC}:${LF}" in
  16:32)
    CONFIG="perfect_blocking_upsampling/run_configs/lam1p0_L16to32_current5x5_epoch002_flowinit_mh.yaml"
    OUTPUT_ROOT="perfect_blocking_upsampling/outputs/controlled_patch_lam1p0/coarse_detail_L16to32"
    ;;
  32:64)
    CONFIG="perfect_blocking_upsampling/run_configs/lam1p0_L32to64_rqspline_newkernel_coarse_detail_production.yaml"
    OUTPUT_ROOT="perfect_blocking_upsampling/outputs/controlled_patch_lam1p0/coarse_detail_L32to64"
    ;;
esac

EXECUTE=()
BACKGROUND=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --execute) EXECUTE=(--execute) ;;
    --background) BACKGROUND=true ;;
    *) echo "usage: $0 <Lc> <Lf> rN [--execute] [--background]" >&2; exit 2 ;;
  esac
  shift
done
if [[ "$BACKGROUND" == true && ${#EXECUTE[@]} -eq 0 ]]; then
  echo "--background requires --execute" >&2
  exit 2
fi

# Trajectory parameters.  Edit these values for a different controlled run.
N_CHAINS=500
N_SWEEPS=500
START_INDEX=1000
DETAIL_PATCH=4
DETAIL_PASSES=1
DETAIL_STEP=0.15
COARSE_PATCH=4
COARSE_PASSES=1
COARSE_STEP=0.15
DETAIL_WARMUP=0
MEASURE_EVERY=5
CHECKPOINT_EVERY=100
SAVE_EVERY=100
BATCH_SIZE=50
SEED=$((SEED_BASE + ${RUN_NUMBER#r} - 1))

float_tag() { printf '%s' "$1" | sed 's/-/m/g; s/\./p/g'; }
RUN_ID="L${LC}toL${LF}_N${N_CHAINS}_S${N_SWEEPS}_start${START_INDEX}_Pd${DETAIL_PATCH}x${DETAIL_PASSES}_sd$(float_tag "$DETAIL_STEP")_Pc${COARSE_PATCH}x${COARSE_PASSES}_sc$(float_tag "$COARSE_STEP")_${RUN_NUMBER}"
RUN_DIR="$OUTPUT_ROOT/$RUN_ID"
if [[ -e "$RUN_DIR" ]]; then
  echo "refusing to overwrite existing run directory: $RUN_DIR" >&2
  exit 1
fi

CMD=(
  "$PYTHON" -B "$DRIVER"
  --config "$CONFIG"
  --run-id "$RUN_ID"
  --run-dir "$RUN_DIR"
  --set "n_chains=$N_CHAINS"
  --set "n_sweeps=$N_SWEEPS"
  --set "start_index=$START_INDEX"
  --set "random_seed=$SEED"
  --set "patch.detail_patch_size=$DETAIL_PATCH"
  --set "patch.detail_passes=$DETAIL_PASSES"
  --set "patch.fine_proposal_sigma=$DETAIL_STEP"
  --set "patch.coarse_patch_size=$COARSE_PATCH"
  --set "patch.coarse_passes=$COARSE_PASSES"
  --set "patch.coarse_step_size=$COARSE_STEP"
  --set "patch.initial_detail_only_sweeps=$DETAIL_WARMUP"
  --set "measure_every=$MEASURE_EVERY"
  --set "checkpoint_every=$CHECKPOINT_EVERY"
  --set "save_every=$SAVE_EVERY"
  --set "batch_size=$BATCH_SIZE"
  "${EXECUTE[@]}"
)

printf 'run_id=%s\nrun_dir=%s\n' "$RUN_ID" "$RUN_DIR"
printf 'command='; printf '%q ' "${CMD[@]}"; printf '\n'
if [[ "$BACKGROUND" == true ]]; then
  mkdir -p "$RUN_DIR/logs" "$RUN_DIR/manifests"
  nohup "${CMD[@]}" > "$RUN_DIR/logs/submit.nohup.log" 2>&1 < /dev/null &
  PID=$!
  printf '%s\n' "$PID" > "$RUN_DIR/manifests/submit_nohup.pid"
  printf 'background_pid=%s\nsubmit_log=%s\n' "$PID" "$RUN_DIR/logs/submit.nohup.log"
  exit 0
fi
"${CMD[@]}"
