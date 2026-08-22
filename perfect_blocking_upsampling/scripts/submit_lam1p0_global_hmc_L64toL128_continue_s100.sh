#!/usr/bin/env bash
# Append 100 global-HMC sweeps to the completed L64 -> L128 run, starting at sweep 100.
# Usage: bash $0 [--execute] [--background]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${PYTHON:-$ROOT/../../.venv/bin/python}"
[[ -x "$PY" ]] || PY="$ROOT/../.venv/bin/python"
[[ -x "$PY" ]] || { echo "shared Python interpreter not found; set PYTHON=/path/to/python" >&2; exit 1; }
EXECUTE=0; BACKGROUND=0
for arg in "$@"; do
  case "$arg" in --execute) EXECUTE=1 ;; --background) BACKGROUND=1 ;; *) echo "usage: $0 [--execute] [--background]" >&2; exit 2 ;; esac
done

N_CHAINS="${N_CHAINS:-1500}"; N_SWEEPS="${N_SWEEPS:-100}"; SAVE_EVERY="${SAVE_EVERY:-5}"
RUN="$ROOT/perfect_blocking_upsampling/outputs/fine_hmc_lam1p0/global/L64toL128_N1500_S100_full_tau2_n50_eps2over50_sweep0"
INPUT="$RUN/checkpoints/checkpoint_sweep_0100.npz"
CMD=("$PY" -B "$ROOT/perfect_blocking_upsampling/scripts/run_lam1p0_fine_hmc.py"
  --run-dir "$RUN" --initialization input --input-source "$INPUT" --append --sweep-offset 100
  --n-chains "$N_CHAINS" --n-sweeps "$N_SWEEPS" --save-every "$SAVE_EVERY"
  --step-size 0.04 --leapfrog-steps 50 --divide 1
  --batch-size 50 --hmc-batch-size 50 --measurement-batch-size 50
  --seed 2026081540 --level-name L64toL128)
printf 'L64->L128 continuation: sweeps 101-%s, N=%s, eps=2/50 n=50 tau=2\ninput=%s\noutput=%s\n' "$((100 + N_SWEEPS))" "$N_CHAINS" "$INPUT" "$RUN"
[[ "$EXECUTE" -eq 1 ]] || exit 0
[[ -d "$RUN" ]] || { echo "missing original run directory: $RUN" >&2; exit 1; }
[[ -f "$INPUT" ]] || { echo "missing checkpoint: $INPUT" >&2; exit 1; }
if [[ "$BACKGROUND" -eq 1 ]]; then
  "${CMD[@]}" >>"$RUN/logs/continuation_s0100.log" 2>&1 </dev/null & echo "$!" >"$RUN/submit_pid_continue_s0100.txt"; echo "background_pid=$!"
else
  "${CMD[@]}" 2>&1 | tee -a "$RUN/logs/continuation_s0100.log"
fi
