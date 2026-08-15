#!/usr/bin/env bash
# Controlled L16->L32 HMC comparison.  Both runs start from exactly the same
# saved L16 fields and use the same flow, kernel, seed, divide, and chain set.
# Only the HMC trajectory length differs:
#   short: eps=.10, n=8  (tau=.8)
#   long:  eps=.10, n=20 (tau=2.0)
# Usage:
#   bash $0 r21 r22 [--execute] [--background]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="$ROOT/../../.venv/bin/python"
SHORT_RUN_NUMBER="${1:-}"; LONG_RUN_NUMBER="${2:-}"; shift 2 || true
[[ "$SHORT_RUN_NUMBER" =~ ^r[0-9]+$ && "$LONG_RUN_NUMBER" =~ ^r[0-9]+$ ]] || {
  echo "usage: $0 rSHORT rLONG [--execute] [--background]" >&2; exit 2;
}

EXEC=0; BG=0
for flag in "$@"; do
  case "$flag" in
    --execute) EXEC=1 ;;
    --background) BG=1 ;;
    *) echo "unknown argument: $flag" >&2; exit 2 ;;
  esac
done

N_CHAINS="${N_CHAINS:-1500}"
START_INDEX="${START_INDEX:-0}"
N_SWEEPS="${N_SWEEPS:-100}"
DIVIDE="${DIVIDE:-2}"
STEP_SIZE="${STEP_SIZE:-0.10}"       # deliberately identical in both arms
SHORT_STEPS="${SHORT_STEPS:-8}"
LONG_STEPS="${LONG_STEPS:-20}"
BATCH_SIZE="${BATCH_SIZE:-50}"
SAVE_EVERY="${SAVE_EVERY:-1}"
SEED="${SEED:-2026081321}"            # deliberately identical in both arms
L16_SOURCE="${L16_SOURCE:-$ROOT/perfect_blocking_upsampling/outputs/hmc_upscale_chain_lam1p0/L8toL64/L8toL64_N1500_start0_HMCtherm100_100_100_r1/levels/L8toL16/final_phi.npz}"
FLOW="$ROOT/perfect_blocking_upsampling/runs/lam1p0/training/lam1p0_L16to32_alternatingKL_highcorr5_r1_iteration5/flow5/stage_oo/checkpoints/checkpoint_best_nll.pt"
KERNEL="$ROOT/perfect_blocking/perfect_blocking_lam1p0/kernels/candidates/alternatingKL_L16to32_highcorr5_r1_iteration5/iteration5.json"
OUT_ROOT="$ROOT/perfect_blocking_upsampling/outputs/fine_hmc_lam1p0/L16toL32"

make_cmd() {
  local run="$1" steps="$2"
  env PYTHONUNBUFFERED=1 "$PY" -B "$ROOT/perfect_blocking_upsampling/scripts/run_lam1p0_fine_hmc.py" \
    --run-dir "$run" --native-source "$ROOT/data/configs_phi4_2d/lam1p0_kappac0p340301_L32/configs.npz" \
    --coarse-source "$L16_SOURCE" --flow-checkpoint "$FLOW" --kernel-path "$KERNEL" \
    --initialization direct_coarse_flow --n-chains "$N_CHAINS" --n-sweeps "$N_SWEEPS" \
    --save-every "$SAVE_EVERY" --batch-size "$BATCH_SIZE" --start-index "$START_INDEX" \
    --step-size "$STEP_SIZE" --leapfrog-steps "$steps" --divide "$DIVIDE" --seed "$SEED" \
    --level-name L16toL32
}

SHORT_RUN="$OUT_ROOT/L16toL32_N${N_CHAINS}_S${N_SWEEPS}_start${START_INDEX}_HMCd${DIVIDE}_n${SHORT_STEPS}_eps${STEP_SIZE/./p}_${SHORT_RUN_NUMBER}"
LONG_RUN="$OUT_ROOT/L16toL32_N${N_CHAINS}_S${N_SWEEPS}_start${START_INDEX}_HMCd${DIVIDE}_n${LONG_STEPS}_eps${STEP_SIZE/./p}_${LONG_RUN_NUMBER}"
printf 'shared: input_L16=%s; seed=%s; divide=%s; step_size=%s; N=%s; S=%s\n' "$L16_SOURCE" "$SEED" "$DIVIDE" "$STEP_SIZE" "$N_CHAINS" "$N_SWEEPS"
printf 'short: %s (trajectory length %s)\n' "$SHORT_RUN" "$(awk "BEGIN {print $STEP_SIZE * $SHORT_STEPS}")"
printf 'long:  %s (trajectory length %s)\n' "$LONG_RUN" "$(awk "BEGIN {print $STEP_SIZE * $LONG_STEPS}")"
[[ "$EXEC" -eq 1 ]] || exit 0
[[ -f "$L16_SOURCE" ]] || { echo "missing L16 source: $L16_SOURCE" >&2; exit 1; }
[[ ! -e "$SHORT_RUN" && ! -e "$LONG_RUN" ]] || { echo "a requested run directory already exists" >&2; exit 1; }

launch() {
  local run="$1" steps="$2"
  mkdir -p "$run/logs"
  { echo "controlled_pair=short_n${SHORT_STEPS}_vs_long_n${LONG_STEPS}"; echo "shared_seed=$SEED"; echo "input_L16=$L16_SOURCE"; } >"$run/logs/run.log"
  if [[ "$BG" -eq 1 ]]; then
    make_cmd "$run" "$steps" >>"$run/logs/run.log" 2>&1 </dev/null &
    echo "$!" >"$run/submit_pid.txt"; echo "background_pid=$! run=$run"
  else
    make_cmd "$run" "$steps" 2>&1 | tee -a "$run/logs/run.log"
  fi
}
launch "$SHORT_RUN" "$SHORT_STEPS"
launch "$LONG_RUN" "$LONG_STEPS"
