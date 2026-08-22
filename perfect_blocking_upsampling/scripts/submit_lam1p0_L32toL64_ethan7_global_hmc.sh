#!/usr/bin/env bash
# Ethan 7x7: full-volume L32->L64 HMC, tau=2, 36 leapfrog steps.
# Usage: bash $0 [--execute] [--background]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${PYTHON:-$ROOT/../../.venv/bin/python}"
CONFIG="$ROOT/perfect_blocking_upsampling/run_configs/lam1p0_hmc_L32toL64_ethan7_global_tau2_n36.json"
RUN="$ROOT/perfect_blocking_upsampling/outputs/hmc_upscale_chain_lam1p0/L32toL64/L32toL64_ethan7_N1500_S100_global_tau2_n36_eps2over36_r1"
EXECUTE=0; BACKGROUND=0
for arg in "$@"; do case "$arg" in --execute) EXECUTE=1;; --background) BACKGROUND=1;; *) echo "usage: $0 [--execute] [--background]" >&2; exit 2;; esac; done
CMD=(env PYTHONUNBUFFERED=1 "$PY" -B "$ROOT/perfect_blocking_upsampling/scripts/run_lam1p0_hmc_upscale_chain.py" --config "$CONFIG" --run-dir "$RUN" --n-chains 1500 --start-index 0)
printf 'run_dir=%s\nflow=Ethan 7x7; HMC=global full volume, 100 sweeps, tau=2, n=36, eps=2/36\n' "$RUN"
[[ "$EXECUTE" -eq 1 ]] || exit 0
[[ ! -e "$RUN" ]] || { echo "run already exists: $RUN" >&2; exit 1; }
mkdir -p "$RUN/logs"
if [[ "$BACKGROUND" -eq 1 ]]; then nohup "${CMD[@]}" >"$RUN/logs/run.log" 2>&1 </dev/null & echo "$!" >"$RUN/submit_pid.txt"; echo "background_pid=$!"; else exec "${CMD[@]}"; fi
