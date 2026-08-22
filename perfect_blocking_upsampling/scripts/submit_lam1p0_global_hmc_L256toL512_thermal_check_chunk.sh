#!/usr/bin/env bash
# Thermalization check for one RAM-bounded L256->L512 chunk.
# Reads only this chunk from the disk-backed sweep-zero .npy and writes only
# final_phi.npz (plus small CSV observable histories) after 100 HMC sweeps.
# Usage: bash $0 CHUNK_NUMBER [--execute] [--background]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${PYTHON:-$ROOT/../../.venv/bin/python}"
[[ -x "$PY" ]] || PY="$ROOT/../.venv/bin/python"
[[ -x "$PY" ]] || { echo "shared Python interpreter not found; set PYTHON=/path/to/python" >&2; exit 1; }
CHUNK_NUMBER="${1:-}"; shift || true
[[ "$CHUNK_NUMBER" =~ ^[0-9]+$ ]] || { echo "usage: $0 CHUNK_NUMBER [--execute] [--background]" >&2; exit 2; }
EXECUTE=0; BACKGROUND=0
for arg in "$@"; do
  case "$arg" in --execute) EXECUTE=1 ;; --background) BACKGROUND=1 ;; *) echo "usage: $0 CHUNK_NUMBER [--execute] [--background]" >&2; exit 2 ;; esac
done

TOTAL_CHAINS="${TOTAL_CHAINS:-1500}"; CHUNK_SIZE="${CHUNK_SIZE:-25}"; N_SWEEPS="${N_SWEEPS:-100}"; MEASURE_EVERY="${MEASURE_EVERY:-5}"
START_INDEX=$(( CHUNK_NUMBER * CHUNK_SIZE ))
[[ "$START_INDEX" -lt "$TOTAL_CHAINS" ]] || { echo "chunk $CHUNK_NUMBER starts beyond N=$TOTAL_CHAINS" >&2; exit 2; }
N_CHAINS=$(( TOTAL_CHAINS - START_INDEX )); (( N_CHAINS > CHUNK_SIZE )) && N_CHAINS="$CHUNK_SIZE"
INPUT="$ROOT/perfect_blocking_upsampling/outputs/hmc_upscale_chain_lam1p0/L256toL512/L256toL512_N1500_from_rethermed_L256_zero_sweep_flow/levels/L256toL512/checkpoints/initialization_phi.npy"
RUN="$ROOT/perfect_blocking_upsampling/outputs/fine_hmc_lam1p0/global/L256toL512_thermal_check_chunks/L256toL512_N${N_CHAINS}_start${START_INDEX}_S${N_SWEEPS}_tau2_n100_eps2over100"
CMD=("$PY" -B "$ROOT/perfect_blocking_upsampling/scripts/run_lam1p0_fine_hmc.py"
  --run-dir "$RUN" --initialization input --input-source "$INPUT"
  --n-chains "$N_CHAINS" --start-index "$START_INDEX" --n-sweeps "$N_SWEEPS" --save-every "$MEASURE_EVERY"
  --step-size 0.02 --leapfrog-steps 100 --divide 1
  --batch-size 1 --hmc-batch-size 1 --measurement-batch-size 1 --final-config-only
  --seed $(( 2026081600 + CHUNK_NUMBER )) --level-name L256toL512)
printf 'L256->L512 thermalization check: chunk=%s, N=%s, source indices=%s..%s; tau=2, eps=0.02, n=100\ninput=%s\noutput=%s\n' "$CHUNK_NUMBER" "$N_CHAINS" "$START_INDEX" "$(( START_INDEX + N_CHAINS - 1 ))" "$INPUT" "$RUN"
[[ "$EXECUTE" -eq 1 ]] || exit 0
[[ -f "$INPUT" ]] || { echo "missing disk-backed sweep-zero fields: $INPUT" >&2; exit 1; }
[[ ! -e "$RUN" ]] || { echo "refusing to overwrite existing output: $RUN" >&2; exit 1; }
mkdir -p "$RUN/logs"
if [[ "$BACKGROUND" -eq 1 ]]; then
  "${CMD[@]}" >"$RUN/logs/run.log" 2>&1 </dev/null & echo "$!" >"$RUN/submit_pid.txt"; echo "background_pid=$!"
else
  "${CMD[@]}" 2>&1 | tee "$RUN/logs/run.log"
fi
