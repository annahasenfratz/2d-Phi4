#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RUN_ROOT="${RUN_ROOT:-$ROOT/perfect_blocking_upsampling/runs/lam1p0/training/lam1p0_L8to16_alternatingKL_5x5_r1_continue}"
EXEC=0; BACKGROUND=0
for argument in "$@"; do case "$argument" in --execute) EXEC=1;; --background) BACKGROUND=1;; *) echo "Usage: $0 [--execute] [--background]" >&2; exit 2;; esac; done
printf 'run=%s\n' "$RUN_ROOT"; [[ $EXEC -eq 1 ]] || exit 0
mkdir -p "$RUN_ROOT/logs"
CMD=(env RUN_ROOT="$RUN_ROOT" bash "$ROOT/perfect_blocking_upsampling/scripts/run_lam1p0_l8to16_alternating_kl_5x5_continue.sh")
if [[ $BACKGROUND -eq 1 ]]; then nohup "${CMD[@]}" >"$RUN_ROOT/logs/run.log" 2>&1 < /dev/null & echo $! >"$RUN_ROOT/submit_pid.txt"; echo "background_pid=$!"; else exec "${CMD[@]}"; fi
