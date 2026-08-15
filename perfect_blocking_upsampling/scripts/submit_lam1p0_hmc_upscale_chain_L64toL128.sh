#!/usr/bin/env bash
# Continue the completed L8->L64 chain: one L64->L128 flow, then exact fine HMC.
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
DIVIDE="${DIVIDE:-2}"
if [[ "$DIVIDE" == 2 ]]; then
  CONFIG_DEFAULT="$ROOT/perfect_blocking_upsampling/run_configs/lam1p0_hmc_upscale_chain_L64toL128_d2_streamed.json"
else
  CONFIG_DEFAULT="$ROOT/perfect_blocking_upsampling/run_configs/lam1p0_hmc_upscale_chain_L64toL128.json"
fi
CONFIG="${CONFIG:-$CONFIG_DEFAULT}"
INPUT_L64="${INPUT_L64:-$ROOT/perfect_blocking_upsampling/outputs/hmc_upscale_chain_lam1p0/L8toL64/L8toL64_N1500_start0_HMCtherm100_100_100_r1/levels/L32toL64/final_phi.npz}"
RUN="$ROOT/perfect_blocking_upsampling/outputs/hmc_upscale_chain_lam1p0/L8toL128/L8toL128_N${N_CHAINS}_start${START_INDEX}_HMCtherm100_100_100_100_d${DIVIDE}_${RUN_NUMBER}"
CMD=(env PYTHONUNBUFFERED=1 "$PY" -B "$ROOT/perfect_blocking_upsampling/scripts/run_lam1p0_hmc_upscale_chain.py" --config "$CONFIG" --run-dir "$RUN" --n-chains "$N_CHAINS" --start-index "$START_INDEX")

printf 'run_dir=%s\ninput_l64=%s\ncommand=' "$RUN" "$INPUT_L64"; printf '%q ' "${CMD[@]}"; printf '\n'
[[ $EXEC -eq 1 ]] || exit 0
[[ -f "$INPUT_L64" ]] || { echo "missing completed L64 fields: $INPUT_L64" >&2; exit 1; }
[[ ! -e "$RUN" ]] || { echo "run directory already exists: $RUN" >&2; exit 1; }
mkdir -p "$RUN/logs"
if [[ $BG -eq 1 ]]; then
  { echo "started_at=$(date -Iseconds)"; echo "input_l64=$INPUT_L64"; printf 'command='; printf '%q ' "${CMD[@]}"; printf '\n'; } >"$RUN/logs/run.log"
  nohup "${CMD[@]}" >>"$RUN/logs/run.log" 2>&1 </dev/null &
  echo "$!" >"$RUN/submit_pid.txt"
  echo "background_pid=$!"
else
  exec "${CMD[@]}"
fi
