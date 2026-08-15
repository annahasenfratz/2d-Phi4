#!/usr/bin/env bash
# Regression smoke for append-only coordinate diagnostics at divide=2.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; PY="$ROOT/../../.venv/bin/python"
OUT="${OUT:-$ROOT/perfect_blocking_upsampling/outputs/smoke/coordinate_mh_append_div2}"
EXEC=0; BACKGROUND=0
for argument in "$@";do case "$argument" in --execute)EXEC=1;;--background)BACKGROUND=1;;*)echo "Usage: $0 [--execute] [--background]" >&2;exit 2;;esac;done
CMD=("$PY" -B "$ROOT/perfect_blocking_upsampling/scripts/run_lam1p0_mit_coordinate_mh_L8to16.py" --run-dir "$OUT" --n-chains 4 --n-sweeps 3 --divide 2 --update-mode coarse_detail --save-sweeps 0,1,2,3 --smoke)
printf 'out=%s\nexpected_coordinate_rows=192\n' "$OUT"; [[ $EXEC -eq 1 ]] || exit 0
if [[ -e "$OUT/debug/coordinate_mh_diagnostics.csv" ]]; then
  echo "refusing to append to an existing smoke output; choose OUT=..." >&2
  exit 2
fi
mkdir -p "$OUT/logs"
CHECK_CMD=("$PY" -B -c 'import csv,sys; from pathlib import Path; p=Path(sys.argv[1]); rows=list(csv.DictReader(p.open())); assert len(rows)==192, len(rows); assert len({tuple(r.items()) for r in rows})==192; print("append diagnostics OK: 192 unique rows")' "$OUT/debug/coordinate_mh_diagnostics.csv")
if [[ $BACKGROUND -eq 1 ]];then nohup bash -c "$(printf '%q ' "${CMD[@]}"); $(printf '%q ' "${CHECK_CMD[@]}")" >"$OUT/logs/run.log" 2>&1 < /dev/null & echo $! >"$OUT/submit_pid.txt";echo "background_pid=$!";else "${CMD[@]}"; "${CHECK_CMD[@]}";fi
