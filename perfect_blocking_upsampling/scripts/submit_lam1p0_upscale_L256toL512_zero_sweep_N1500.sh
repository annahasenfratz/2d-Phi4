#!/usr/bin/env bash
# Chained L256 -> L512 flow initialization in independently restartable chunks.
# Every completed chunk writes its own final_phi.npz and sweep-zero checkpoint.
# Usage: bash $0 BATCH_NUMBER [--execute] [--background]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${PYTHON:-$ROOT/../../.venv/bin/python}"
[[ -x "$PY" ]] || PY="$ROOT/../.venv/bin/python"
[[ -x "$PY" ]] || { echo "shared Python interpreter not found; set PYTHON=/path/to/python" >&2; exit 1; }
BATCH_NUMBER="${1:-}"; shift || true
[[ "$BATCH_NUMBER" =~ ^[0-9]+$ ]] || { echo "usage: $0 BATCH_NUMBER [--execute] [--background]" >&2; exit 2; }
EXECUTE=0; BACKGROUND=0
for arg in "$@"; do
  case "$arg" in --execute) EXECUTE=1 ;; --background) BACKGROUND=1 ;; *) echo "usage: $0 [--execute] [--background]" >&2; exit 2 ;; esac
done

CONFIG="$ROOT/perfect_blocking_upsampling/run_configs/lam1p0_upscale_L256toL512_zero_sweep_N1500.json"
INPUT="$ROOT/perfect_blocking_upsampling/outputs/fine_hmc_lam1p0/global/L128toL256_N1500_S100_full_tau2_n70_eps2over70_sweep0/checkpoints/checkpoint_sweep_0100.npz"
TOTAL_CHAINS="${TOTAL_CHAINS:-1500}"; CHUNK_SIZE="${CHUNK_SIZE:-25}"
START_INDEX=$(( BATCH_NUMBER * CHUNK_SIZE ))
[[ "$START_INDEX" -lt "$TOTAL_CHAINS" ]] || { echo "batch $BATCH_NUMBER starts beyond N=$TOTAL_CHAINS" >&2; exit 2; }
N_CHAINS=$(( TOTAL_CHAINS - START_INDEX )); (( N_CHAINS > CHUNK_SIZE )) && N_CHAINS="$CHUNK_SIZE"
RUN="$ROOT/perfect_blocking_upsampling/outputs/hmc_upscale_chain_lam1p0/L256toL512/chunks/L256toL512_N${N_CHAINS}_start${START_INDEX}_from_rethermed_L256_zero_sweep_flow"
CMD=(env PYTHONUNBUFFERED=1 "$PY" -B "$ROOT/perfect_blocking_upsampling/scripts/run_lam1p0_hmc_upscale_chain.py" --config "$CONFIG" --run-dir "$RUN" --n-chains "$N_CHAINS" --start-index "$START_INDEX")
printf 'L256->L512 sweep-zero flow initialization: chunk=%s, N=%s, start=%s, flow batch=2, no HMC\ninput=%s\noutput=%s\ncommand=' "$BATCH_NUMBER" "$N_CHAINS" "$START_INDEX" "$INPUT" "$RUN"; printf '%q ' "${CMD[@]}"; printf '\n'
[[ "$EXECUTE" -eq 1 ]] || exit 0
[[ -f "$INPUT" ]] || { echo "missing completed thermalized L256 input: $INPUT" >&2; exit 1; }
[[ ! -e "$RUN" ]] || { echo "refusing to overwrite existing output: $RUN" >&2; exit 1; }
mkdir -p "$RUN/logs"
if [[ "$BACKGROUND" -eq 1 ]]; then
  "${CMD[@]}" >"$RUN/logs/run.log" 2>&1 </dev/null & echo "$!" >"$RUN/submit_pid.txt"; echo "background_pid=$!"
else
  "${CMD[@]}" 2>&1 | tee "$RUN/logs/run.log"
fi
