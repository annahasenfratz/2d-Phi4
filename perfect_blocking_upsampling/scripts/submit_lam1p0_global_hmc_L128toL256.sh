#!/usr/bin/env bash
# Global fine-field HMC rethermalization, L128 -> L256, memory-batched.
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
INPUT="${INPUT_L256:-$ROOT/perfect_blocking_upsampling/outputs/hmc_upscale_chain_lam1p0/L8toL256/L8toL256_N1500_start0_HMCtherm100_100_100_200_100_d2_r2/levels/L128toL256/checkpoints/checkpoint_sweep_0000.npz}"
RUN="$ROOT/perfect_blocking_upsampling/outputs/fine_hmc_lam1p0/global/L128toL256_N${N_CHAINS}_S${N_SWEEPS}_full_tau2_n70_eps2over70_sweep0"
CMD=("$PY" -B "$ROOT/perfect_blocking_upsampling/scripts/run_lam1p0_fine_hmc.py"
  --run-dir "$RUN" --initialization input --input-source "$INPUT"
  --n-chains "$N_CHAINS" --n-sweeps "$N_SWEEPS" --save-every "$SAVE_EVERY"
  --step-size 0.02857142857142857 --leapfrog-steps 70 --divide 1
  --batch-size 8 --hmc-batch-size 8 --measurement-batch-size 8
  --seed 2026081538 --level-name L128toL256)
printf 'L128->L256: N=%s sweeps=%s global HMC eps=2/70 n=70 tau=2; HMC/measurement batch=8\ninput=%s\noutput=%s\n' "$N_CHAINS" "$N_SWEEPS" "$INPUT" "$RUN"
[[ "$EXECUTE" -eq 1 ]] || exit 0
[[ -f "$INPUT" ]] || { echo "missing sweep-zero input: $INPUT" >&2; exit 1; }
[[ ! -e "$RUN" ]] || { echo "refusing to overwrite existing output: $RUN" >&2; exit 1; }
mkdir -p "$RUN/logs"
if [[ "$BACKGROUND" -eq 1 ]]; then
  "${CMD[@]}" >"$RUN/logs/run.log" 2>&1 </dev/null & echo "$!" >"$RUN/submit_pid.txt"; echo "background_pid=$!"
else
  "${CMD[@]}" 2>&1 | tee "$RUN/logs/run.log"
fi
