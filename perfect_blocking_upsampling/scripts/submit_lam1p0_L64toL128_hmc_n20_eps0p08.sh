#!/usr/bin/env bash
# L64->L128: HMC trajectory eps=.08, n=20 (tau=1.6), fixed 16^2 active patch.
# Start this after the L32->L64 r24 run completes, or override INPUT_L64.
# Usage: bash $0 rNUMBER [--execute] [--background]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; PY="$ROOT/../../.venv/bin/python"
RUN_NUMBER="${1:-}"; shift || true
[[ "$RUN_NUMBER" =~ ^r[0-9]+$ ]] || { echo "usage: $0 rNUMBER [--execute] [--background]" >&2; exit 2; }
EXEC=0; BG=0; for x in "$@"; do case "$x" in --execute) EXEC=1;; --background) BG=1;; *) echo "unknown argument: $x" >&2; exit 2;; esac; done
N_CHAINS="${N_CHAINS:-1500}"; START_INDEX="${START_INDEX:-0}"
CONFIG="${CONFIG:-$ROOT/perfect_blocking_upsampling/run_configs/lam1p0_hmc_L64toL128_d8_n20_eps0p08.json}"
INPUT_L64="${INPUT_L64:-$ROOT/perfect_blocking_upsampling/outputs/hmc_upscale_chain_lam1p0/L32toL64/L32toL64_N1500_S100_start0_HMCd4_n20_eps0p08_r24/final_phi.npz}"
RUN="$ROOT/perfect_blocking_upsampling/outputs/hmc_upscale_chain_lam1p0/L64toL128/L64toL128_N${N_CHAINS}_S100_start${START_INDEX}_HMCd8_n20_eps0p08_${RUN_NUMBER}"
CMD=(env PYTHONUNBUFFERED=1 "$PY" -B "$ROOT/perfect_blocking_upsampling/scripts/run_lam1p0_hmc_upscale_chain.py" --config "$CONFIG" --run-dir "$RUN" --n-chains "$N_CHAINS" --start-index "$START_INDEX")
printf 'run_dir=%s\ninput_L64=%s\ntrajectory_length=1.6; active_sites=256\n' "$RUN" "$INPUT_L64"
[[ $EXEC -eq 1 ]] || exit 0
[[ -f "$INPUT_L64" ]] || { echo "missing completed L64 source: $INPUT_L64" >&2; exit 1; }
[[ ! -e "$RUN" ]] || { echo "run directory already exists: $RUN" >&2; exit 1; }
mkdir -p "$RUN/logs"
if [[ $BG -eq 1 ]]; then
  { echo "started_at=$(date -Iseconds)"; echo "input_L64=$INPUT_L64"; printf 'command='; printf '%q ' "${CMD[@]}"; printf '\n'; } >"$RUN/logs/run.log"
  nohup "${CMD[@]}" >>"$RUN/logs/run.log" 2>&1 </dev/null &
  echo "$!" >"$RUN/submit_pid.txt"; echo "background_pid=$!"
else exec "${CMD[@]}"; fi
