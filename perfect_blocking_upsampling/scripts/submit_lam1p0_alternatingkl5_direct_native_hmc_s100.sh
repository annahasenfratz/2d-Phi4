#!/usr/bin/env bash
# Matched HMC test for the Aug alternating-KL iteration-5 5x5 flow/kernel.
# Input is the first 5,000 direct native L16 fields; this matches the new
# highcorr-5x5 and Ethan-7x7 end-to-end comparisons.
# Usage: bash $0 [--execute] [--background]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${PYTHON:-$ROOT/../../.venv/bin/python}"
[[ -x "$PY" ]] || { echo "Python interpreter not found: $PY" >&2; exit 1; }

EXECUTE=0
BACKGROUND=0
for arg in "$@"; do
  case "$arg" in
    --execute) EXECUTE=1 ;;
    --background) BACKGROUND=1 ;;
    *) echo "usage: $0 [--execute] [--background]" >&2; exit 2 ;;
  esac
done

N_CHAINS="${N_CHAINS:-5000}"
N_SWEEPS="${N_SWEEPS:-100}"
SAVE_EVERY="${SAVE_EVERY:-5}"
BATCH_SIZE="${BATCH_SIZE:-32}"
HMC_BATCH_SIZE="${HMC_BATCH_SIZE:-32}"
SEED="${SEED:-2026081817}"
INPUT="$ROOT/perfect_blocking_upsampling/outputs/flow_input_audit_lam1p0/alternatingKL5_direct_native_L16_N5000_20260818/sweep0_flow_phi.npz"
RUN="$ROOT/perfect_blocking_upsampling/outputs/fine_hmc_lam1p0/global/L16toL32_alternatingKL5_directnative_N${N_CHAINS}_S${N_SWEEPS}_tau2_n28_eps2over28_20260818"

CMD=(env PYTHONUNBUFFERED=1 "$PY" -B "$ROOT/perfect_blocking_upsampling/scripts/run_lam1p0_fine_hmc.py"
  --run-dir "$RUN" --initialization input --input-source "$INPUT"
  --n-chains "$N_CHAINS" --n-sweeps "$N_SWEEPS" --save-every "$SAVE_EVERY"
  --batch-size "$BATCH_SIZE" --hmc-batch-size "$HMC_BATCH_SIZE" --measurement-batch-size "$HMC_BATCH_SIZE"
  --step-size 0.07142857142857142 --leapfrog-steps 28 --divide 1 --seed "$SEED" --level-name L16toL32)

printf 'run_dir=%s\ninput=%s\nN=%s sweeps=%s save_every=%s\nfull-field HMC: eps=2/28, n=28, tau=2; calibrated acceptance about 0.86\n' \
  "$RUN" "$INPUT" "$N_CHAINS" "$N_SWEEPS" "$SAVE_EVERY"
[[ "$EXECUTE" -eq 1 ]] || exit 0
[[ -f "$INPUT" ]] || { echo "missing sweep-zero input: $INPUT" >&2; exit 1; }
[[ ! -e "$RUN" ]] || { echo "refusing to overwrite existing run: $RUN" >&2; exit 1; }
mkdir -p "$RUN/logs"
if [[ "$BACKGROUND" -eq 1 ]]; then
  { echo "started_at=$(date -Iseconds)"; printf 'command='; printf '%q ' "${CMD[@]}"; printf '\n'; } > "$RUN/logs/run.log"
  nohup "${CMD[@]}" >> "$RUN/logs/run.log" 2>&1 </dev/null &
  echo "$!" > "$RUN/submit_pid.txt"
  echo "background_pid=$!"
else
  "${CMD[@]}" 2>&1 | tee "$RUN/logs/run.log"
fi
