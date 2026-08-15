#!/usr/bin/env bash
# Start from the existing thermalized L8->L16 output, then test L16->L32 HMC.
# The L16->L32 HMC trajectory has tau = eps*n = .10*20 = 2.
# Usage: bash $0 r1 [--execute] [--background]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="$ROOT/../../.venv/bin/python"
RUN_NUMBER="${1:-}"; shift || true
[[ "$RUN_NUMBER" =~ ^r[0-9]+$ ]] || { echo "usage: $0 rNUMBER [--execute] [--background]" >&2; exit 2; }
EXEC=0; BG=0
for flag in "$@"; do case "$flag" in --execute) EXEC=1;; --background) BG=1;; *) echo "unknown argument: $flag" >&2; exit 2;; esac; done

N_CHAINS="${N_CHAINS:-1500}"
START_INDEX="${START_INDEX:-0}"
N_SWEEPS="${N_SWEEPS:-100}"
DIVIDE="${DIVIDE:-2}"
STEP_SIZE="${STEP_SIZE:-0.10}"
LEAPFROG_STEPS="${LEAPFROG_STEPS:-20}"
BATCH_SIZE="${BATCH_SIZE:-50}"
SAVE_EVERY="${SAVE_EVERY:-1}"
SEED="${SEED:-2026081319}"
L16_SOURCE="${L16_SOURCE:-$ROOT/perfect_blocking_upsampling/outputs/hmc_upscale_chain_lam1p0/L8toL64/L8toL64_N1500_start0_HMCtherm100_100_100_r1/levels/L8toL16/final_phi.npz}"
FLOW="$ROOT/perfect_blocking_upsampling/runs/lam1p0/training/lam1p0_L16to32_alternatingKL_highcorr5_r1_iteration5/flow5/stage_oo/checkpoints/checkpoint_best_nll.pt"
KERNEL="$ROOT/perfect_blocking/perfect_blocking_lam1p0/kernels/candidates/alternatingKL_L16to32_highcorr5_r1_iteration5/iteration5.json"
RUN="$ROOT/perfect_blocking_upsampling/outputs/fine_hmc_lam1p0/L16toL32/L16toL32_N${N_CHAINS}_S${N_SWEEPS}_start${START_INDEX}_HMCd${DIVIDE}_n${LEAPFROG_STEPS}_eps${STEP_SIZE/./p}_${RUN_NUMBER}"
CMD=(env PYTHONUNBUFFERED=1 "$PY" -B "$ROOT/perfect_blocking_upsampling/scripts/run_lam1p0_fine_hmc.py"
  --run-dir "$RUN" --native-source "$ROOT/data/configs_phi4_2d/lam1p0_kappac0p340301_L32/configs.npz"
  --coarse-source "$L16_SOURCE" --flow-checkpoint "$FLOW" --kernel-path "$KERNEL"
  --initialization direct_coarse_flow --n-chains "$N_CHAINS" --n-sweeps "$N_SWEEPS" --save-every "$SAVE_EVERY"
  --batch-size "$BATCH_SIZE" --start-index "$START_INDEX" --step-size "$STEP_SIZE"
  --leapfrog-steps "$LEAPFROG_STEPS" --divide "$DIVIDE" --seed "$SEED" --level-name L16toL32)
printf 'run_dir=%s\ninput_L16=%s\ntrajectory_length=%s\ncommand=' "$RUN" "$L16_SOURCE" "$(awk "BEGIN {print $STEP_SIZE * $LEAPFROG_STEPS}")"; printf '%q ' "${CMD[@]}"; printf '\n'
[[ $EXEC -eq 1 ]] || exit 0
[[ -f "$L16_SOURCE" ]] || { echo "missing completed L8->L16 source: $L16_SOURCE" >&2; exit 1; }
[[ ! -e "$RUN" ]] || { echo "run directory already exists: $RUN" >&2; exit 1; }
mkdir -p "$RUN/logs"
if [[ $BG -eq 1 ]]; then
  { echo "started_at=$(date -Iseconds)"; echo "source_L16=$L16_SOURCE"; printf 'command='; printf '%q ' "${CMD[@]}"; printf '\n'; } >"$RUN/logs/run.log"
  nohup "${CMD[@]}" >>"$RUN/logs/run.log" 2>&1 </dev/null &
  echo "$!" >"$RUN/submit_pid.txt"; echo "background_pid=$!"
else
  exec "${CMD[@]}"
fi
