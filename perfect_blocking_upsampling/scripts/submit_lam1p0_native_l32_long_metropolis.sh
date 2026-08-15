#!/usr/bin/env bash
# Direct L32 local-Metropolis control. Usage: bash $0 r1 [--execute] [--background]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="$ROOT/../../.venv/bin/python"
DRIVER="$ROOT/perfect_blocking_upsampling/scripts/run_lam1p0_native_l32_long_metropolis.py"
RUN_NUMBER="${1:-}"
if [[ ! "$RUN_NUMBER" =~ ^r[0-9]+$ ]]; then
  echo "usage: $0 rNUMBER [--execute] [--background]" >&2
  exit 2
fi
shift

N_CHAINS="${N_CHAINS:-1}"
N_SWEEPS="${N_SWEEPS:-1000000}"
START_INDEX="${START_INDEX:-0}"
MEASURE_EVERY="${MEASURE_EVERY:-20}"
# A 200-sweep L32 pilot from native config 0 gave 0.598 acceptance at 0.94.
STEP_SIZE="${STEP_SIZE:-0.94}"
SEED="${SEED:-2026080601}"
RUN_TAG="L32_native_metropolis_N${N_CHAINS}_S${N_SWEEPS}_start${START_INDEX}_step${STEP_SIZE/./p}_meas${MEASURE_EVERY}_${RUN_NUMBER}"
RUN_DIR="$ROOT/perfect_blocking_upsampling/outputs/native_metropolis_lam1p0/L32/$RUN_TAG"
INPUT="$ROOT/data/configs_phi4_2d/lam1p0_kappac0p340301_L32/configs.npz"
CMD=("$PYTHON" -B "$DRIVER" --input-configs "$INPUT" --output-dir "$RUN_DIR" --n-chains "$N_CHAINS" --start-index "$START_INDEX" --sweeps "$N_SWEEPS" --measure-every "$MEASURE_EVERY" --step-size "$STEP_SIZE" --seed "$SEED")

EXECUTE=0
BACKGROUND=0
for arg in "$@"; do
  case "$arg" in
    --execute) EXECUTE=1 ;;
    --background) BACKGROUND=1 ;;
    *) echo "usage: $0 rNUMBER [--execute] [--background]" >&2; exit 2 ;;
  esac
done
[[ "$BACKGROUND" -eq 0 || "$EXECUTE" -eq 1 ]] || { echo "--background requires --execute" >&2; exit 2; }
for path in "$PYTHON" "$DRIVER" "$INPUT"; do [[ -e "$path" ]] || { echo "missing: $path" >&2; exit 1; }; done
[[ ! -e "$RUN_DIR" ]] || { echo "run directory already exists: $RUN_DIR" >&2; exit 1; }

printf 'run_dir=%s\n' "$RUN_DIR"
printf 'command='; printf '%q ' "${CMD[@]}"; printf '\n'
if [[ "$EXECUTE" -eq 0 ]]; then
  echo "Prepared only. Add --execute to start; add --background for nohup."
  exit 0
fi
mkdir -p "$RUN_DIR/logs"
printf 'run_id=%s\n' "$RUN_TAG" > "$RUN_DIR/submit_manifest.txt"
printf 'command=' >> "$RUN_DIR/submit_manifest.txt"; printf '%q ' "${CMD[@]}" >> "$RUN_DIR/submit_manifest.txt"; printf '\n' >> "$RUN_DIR/submit_manifest.txt"
if [[ "$BACKGROUND" -eq 1 ]]; then
  nohup "${CMD[@]}" > "$RUN_DIR/logs/run.log" 2>&1 < /dev/null &
  echo "$!" > "$RUN_DIR/submit_pid.txt"
  printf 'background_pid=%s\nlog=%s\n' "$!" "$RUN_DIR/logs/run.log"
else
  exec "${CMD[@]}"
fi
