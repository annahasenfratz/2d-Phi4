#!/usr/bin/env bash
# Ethan 7x7: streamed full-volume L128->L256 HMC, tau=2, 72 leapfrog steps.
# Usage: bash $0 [--execute] [--background]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${PYTHON:-$ROOT/../../.venv/bin/python}"
CONFIG="$ROOT/perfect_blocking_upsampling/run_configs/lam1p0_hmc_L128toL256_ethan7_global_tau2_n72_streamed.json"
INPUT="$ROOT/perfect_blocking_upsampling/outputs/hmc_upscale_chain_lam1p0/L64toL128/L64toL128_ethan7_N1500_S100_global_tau2_n51_eps2over51_r2/final_phi.npz"
RUN="$ROOT/perfect_blocking_upsampling/outputs/hmc_upscale_chain_lam1p0/L128toL256/L128toL256_ethan7_N1500_S100_global_tau2_n72_eps2over72_streamed_r1"
EXECUTE=0; BACKGROUND=0
for arg in "$@"; do case "$arg" in --execute) EXECUTE=1;; --background) BACKGROUND=1;; *) echo "usage: $0 [--execute] [--background]" >&2; exit 2;; esac; done
CMD=(env PYTHONUNBUFFERED=1 "$PY" -B "$ROOT/perfect_blocking_upsampling/scripts/run_lam1p0_hmc_upscale_chain.py" --config "$CONFIG" --run-dir "$RUN" --n-chains 1500 --start-index 0)
printf 'run_dir=%s\ninput_L128=%s\nflow=Ethan 7x7; HMC=global full volume, 100 sweeps, tau=2, n=72, eps=2/72\nstreaming: flow/HMC batches=10; checkpoints=0,25,50,75,100\n' "$RUN" "$INPUT"
[[ "$EXECUTE" -eq 1 ]] || exit 0
[[ -f "$INPUT" ]] || { echo "L64->L128 stage has not completed: $INPUT" >&2; exit 1; }
[[ ! -e "$RUN" ]] || { echo "run already exists: $RUN" >&2; exit 1; }
mkdir -p "$RUN/logs"
if [[ "$BACKGROUND" -eq 1 ]]; then nohup "${CMD[@]}" >"$RUN/logs/run.log" 2>&1 </dev/null & echo "$!" >"$RUN/submit_pid.txt"; echo "background_pid=$!"; else exec "${CMD[@]}"; fi
