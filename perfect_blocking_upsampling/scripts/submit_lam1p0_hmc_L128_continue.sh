#!/usr/bin/env bash
# Continue L128 fields with 100 more exact direct fine-HMC sweeps.
# Usage: bash $0 r1 [--execute] [--background]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="$ROOT/../../.venv/bin/python"
RUN_NUMBER="${1:-}"; shift || true
[[ "$RUN_NUMBER" =~ ^r[0-9]+$ ]] || { echo "usage: $0 rNUMBER [--execute] [--background]" >&2; exit 2; }
EXEC=0; BG=0
for flag in "$@"; do case "$flag" in --execute) EXEC=1;; --background) BG=1;; *) echo "unknown argument: $flag" >&2; exit 2;; esac; done

SOURCE_RUN="${SOURCE_RUN:-$ROOT/perfect_blocking_upsampling/outputs/hmc_upscale_chain_lam1p0/L8toL128/L8toL128_N1500_start0_HMCtherm100_100_100_100_d2_r2}"
INPUT="$SOURCE_RUN/final_phi.npz"
N_CHAINS="${N_CHAINS:-1500}"
N_SWEEPS="${N_SWEEPS:-100}"
SWEEP_OFFSET="${SWEEP_OFFSET:-100}"
DIVIDE="${DIVIDE:-2}"
STEP_SIZE="${STEP_SIZE:-0.08}"
LEAPFROG_STEPS="${LEAPFROG_STEPS:-10}"
SAVE_EVERY="${SAVE_EVERY:-1}"
SEED="${SEED:-2026081310}"
RUN="$ROOT/perfect_blocking_upsampling/outputs/hmc_upscale_chain_lam1p0/L8toL128/L8toL128_N${N_CHAINS}_start0_HMCtherm100_100_100_200_d${DIVIDE}_${RUN_NUMBER}"
CMD=(env PYTHONUNBUFFERED=1 "$PY" -B "$ROOT/perfect_blocking_upsampling/scripts/run_lam1p0_fine_hmc.py"
  --run-dir "$RUN" --native-source "$INPUT" --initialization native --n-chains "$N_CHAINS" --n-sweeps "$N_SWEEPS" --save-every "$SAVE_EVERY" --divide "$DIVIDE" --step-size "$STEP_SIZE" --leapfrog-steps "$LEAPFROG_STEPS" --seed "$SEED" --sweep-offset "$SWEEP_OFFSET" --level-name L64toL128)
printf 'run_dir=%s\ninput=%s\ncommand=' "$RUN" "$INPUT"; printf '%q ' "${CMD[@]}"; printf '\n'
[[ $EXEC -eq 1 ]] || exit 0
[[ -f "$INPUT" ]] || { echo "missing completed source field: $INPUT" >&2; exit 1; }
[[ ! -e "$RUN" ]] || { echo "run directory already exists: $RUN" >&2; exit 1; }
mkdir -p "$RUN/logs"
if [[ $BG -eq 1 ]]; then
  { echo "started_at=$(date -Iseconds)"; echo "continuation_of=$SOURCE_RUN"; printf 'command='; printf '%q ' "${CMD[@]}"; printf '\n'; } >"$RUN/logs/run.log"
  nohup "${CMD[@]}" >>"$RUN/logs/run.log" 2>&1 </dev/null &
  echo "$!" >"$RUN/submit_pid.txt"; echo "background_pid=$!"
else
  exec "${CMD[@]}"
fi
