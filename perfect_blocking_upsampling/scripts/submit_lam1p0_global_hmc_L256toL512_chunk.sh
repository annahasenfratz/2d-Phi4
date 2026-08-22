#!/usr/bin/env bash
# HMC-rethermalize one independently saved L256->L512 flow chunk.
# The completed chunk has final_phi.npz; intermediate checkpoints are saved
# every five sweeps and can be continued using the generic HMC append mode.
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
  case "$arg" in --execute) EXECUTE=1 ;; --background) BACKGROUND=1 ;; *) echo "usage: $0 BATCH_NUMBER [--execute] [--background]" >&2; exit 2 ;; esac
done

TOTAL_CHAINS="${TOTAL_CHAINS:-1500}"; CHUNK_SIZE="${CHUNK_SIZE:-25}"; N_SWEEPS="${N_SWEEPS:-100}"; SAVE_EVERY="${SAVE_EVERY:-5}"
START_INDEX=$(( BATCH_NUMBER * CHUNK_SIZE ))
[[ "$START_INDEX" -lt "$TOTAL_CHAINS" ]] || { echo "batch $BATCH_NUMBER starts beyond N=$TOTAL_CHAINS" >&2; exit 2; }
N_CHAINS=$(( TOTAL_CHAINS - START_INDEX )); (( N_CHAINS > CHUNK_SIZE )) && N_CHAINS="$CHUNK_SIZE"
FLOW_RUN="$ROOT/perfect_blocking_upsampling/outputs/hmc_upscale_chain_lam1p0/L256toL512/chunks/L256toL512_N${N_CHAINS}_start${START_INDEX}_from_rethermed_L256_zero_sweep_flow"
INPUT="$FLOW_RUN/levels/L256toL512/checkpoints/checkpoint_sweep_0000.npz"
RUN="$ROOT/perfect_blocking_upsampling/outputs/fine_hmc_lam1p0/global/L256toL512_chunks/L256toL512_N${N_CHAINS}_start${START_INDEX}_S${N_SWEEPS}_tau2_n100_eps2over100"
CMD=("$PY" -B "$ROOT/perfect_blocking_upsampling/scripts/run_lam1p0_fine_hmc.py"
  --run-dir "$RUN" --initialization input --input-source "$INPUT"
  --n-chains "$N_CHAINS" --n-sweeps "$N_SWEEPS" --save-every "$SAVE_EVERY" --start-index 0
  --step-size 0.02 --leapfrog-steps 100 --divide 1
  --batch-size 1 --hmc-batch-size 1 --measurement-batch-size 1
  --seed $(( 2026081541 + BATCH_NUMBER )) --level-name L256toL512)
printf 'L256->L512 HMC: chunk=%s, N=%s, source indices=%s..%s, tau=2, eps=0.02, n=100\ninput=%s\noutput=%s\n' "$BATCH_NUMBER" "$N_CHAINS" "$START_INDEX" "$(( START_INDEX + N_CHAINS - 1 ))" "$INPUT" "$RUN"
[[ "$EXECUTE" -eq 1 ]] || exit 0
[[ -f "$INPUT" ]] || { echo "missing sweep-zero flow chunk: run the matching upscale chunk first" >&2; exit 1; }
[[ ! -e "$RUN" ]] || { echo "refusing to overwrite existing output: $RUN" >&2; exit 1; }
mkdir -p "$RUN/logs"
if [[ "$BACKGROUND" -eq 1 ]]; then
  "${CMD[@]}" >"$RUN/logs/run.log" 2>&1 </dev/null & echo "$!" >"$RUN/submit_pid.txt"; echo "background_pid=$!"
else
  "${CMD[@]}" 2>&1 | tee "$RUN/logs/run.log"
fi
