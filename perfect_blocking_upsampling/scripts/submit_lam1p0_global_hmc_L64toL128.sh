#!/usr/bin/env bash
# Global fine-field HMC rethermalization, L64 -> L128.
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
INPUT="${INPUT_L128:-$ROOT/perfect_blocking_upsampling/outputs/hmc_upscale_chain_lam1p0/L64toL128/L64toL128_N1500_S100_start0_HMCd8_n20_eps0p08_r25/levels/L64toL128/checkpoints/checkpoint_sweep_0000.npz}"
RUN="$ROOT/perfect_blocking_upsampling/outputs/fine_hmc_lam1p0/global/L64toL128_N${N_CHAINS}_S${N_SWEEPS}_full_tau2_n50_eps2over50_sweep0"
CMD=("$PY" -B "$ROOT/perfect_blocking_upsampling/scripts/run_lam1p0_fine_hmc.py"
  --run-dir "$RUN" --initialization input --input-source "$INPUT"
  --n-chains "$N_CHAINS" --n-sweeps "$N_SWEEPS" --save-every "$SAVE_EVERY"
  --step-size 0.04 --leapfrog-steps 50 --divide 1
  --batch-size 50 --hmc-batch-size 50 --measurement-batch-size 50
  --seed 2026081534 --level-name L64toL128)
printf 'L64->L128: N=%s sweeps=%s global HMC eps=2/50 n=50 tau=2\ninput=%s\noutput=%s\n' "$N_CHAINS" "$N_SWEEPS" "$INPUT" "$RUN"
[[ "$EXECUTE" -eq 1 ]] || exit 0
[[ -f "$INPUT" ]] || { echo "missing sweep-zero input: $INPUT" >&2; exit 1; }
[[ ! -e "$RUN" ]] || { echo "refusing to overwrite existing output: $RUN" >&2; exit 1; }
mkdir -p "$RUN/logs"
if [[ "$BACKGROUND" -eq 1 ]]; then
  "${CMD[@]}" >"$RUN/logs/run.log" 2>&1 </dev/null & echo "$!" >"$RUN/submit_pid.txt"; echo "background_pid=$!"
else
  "${CMD[@]}" 2>&1 | tee "$RUN/logs/run.log"
fi
