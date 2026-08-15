#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)";PY="$ROOT/../../.venv/bin/python";OUT="${OUT:-$ROOT/perfect_blocking_upsampling/outputs/global_ar_lam1p0/L16to32_alternatingKL_highcorr5_r1_iterations2to4}";N="${N:-5000}";EXEC=0;BACKGROUND=0
for x in "$@";do case "$x" in --execute)EXEC=1;;--background)BACKGROUND=1;;*)echo "Usage: $0 [--execute] [--background]" >&2;exit 2;;esac;done
CMD=("$PY" -B "$ROOT/perfect_blocking_upsampling/scripts/compare_lam1p0_l16to32_alternating_kl_continue_global_ar.py" --out "$OUT" --n "$N");printf 'out=%s\nn=%s\n' "$OUT" "$N";[[ $EXEC -eq 1 ]]||exit 0;mkdir -p "$OUT/logs"
if [[ $BACKGROUND -eq 1 ]];then nohup "${CMD[@]}" >"$OUT/logs/run.log" 2>&1 </dev/null &echo $! >"$OUT/submit_pid.txt";echo "background_pid=$!";else exec "${CMD[@]}";fi
