#!/usr/bin/env bash
# L8 -> L16 -> L32 -> L64 flow + fine-HMC chain.  Usage: bash $0 r1 [--execute] [--background]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; PY="$ROOT/../../.venv/bin/python"; RUN_NUMBER="${1:-}"; shift || true
[[ "$RUN_NUMBER" =~ ^r[0-9]+$ ]] || { echo "usage: $0 rNUMBER [--execute] [--background]" >&2; exit 2; }
EXEC=0; BG=0; for x in "$@"; do case "$x" in --execute) EXEC=1;; --background) BG=1;; *) exit 2;; esac; done
CONFIG="${CONFIG:-$ROOT/perfect_blocking_upsampling/run_configs/lam1p0_hmc_upscale_chain_L8to64.json}"
N_CHAINS="${N_CHAINS:-500}"; START_INDEX="${START_INDEX:-0}"; RUN="$ROOT/perfect_blocking_upsampling/outputs/hmc_upscale_chain_lam1p0/L8toL64/L8toL64_N${N_CHAINS}_start${START_INDEX}_HMCtherm10_10_100_${RUN_NUMBER}"
CMD=(env PYTHONUNBUFFERED=1 "$PY" -B "$ROOT/perfect_blocking_upsampling/scripts/run_lam1p0_hmc_upscale_chain.py" --config "$CONFIG" --run-dir "$RUN" --n-chains "$N_CHAINS" --start-index "$START_INDEX")
printf 'run_dir=%s\ncommand=' "$RUN"; printf '%q ' "${CMD[@]}"; printf '\n'; [[ $EXEC -eq 1 ]] || exit 0; [[ ! -e "$RUN" ]] || { echo "run directory already exists: $RUN" >&2; exit 1; }; mkdir -p "$RUN/logs"
if [[ $BG -eq 1 ]]; then { echo "started_at=$(date -Iseconds)"; printf 'command='; printf '%q ' "${CMD[@]}"; printf '\n'; } >"$RUN/logs/run.log"; nohup "${CMD[@]}" >>"$RUN/logs/run.log" 2>&1 </dev/null & echo "$!" >"$RUN/submit_pid.txt"; echo "background_pid=$!"; else exec "${CMD[@]}"; fi
