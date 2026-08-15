#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)";PY="$ROOT/../../.venv/bin/python"
RUN="${RUN_ROOT:-$ROOT/perfect_blocking_upsampling/runs/lam1p0/training/lam1p0_L16to32_directL16_observable_flow_r2}"
CK="$ROOT/perfect_blocking_upsampling/runs/lam1p0/training/lam1p0_L16to32_alternatingKL_highcorr5_r1_iteration5/flow5/stage_oo/checkpoints/checkpoint_best_nll.pt"
K="$ROOT/perfect_blocking/perfect_blocking_lam1p0/kernels/candidates/alternatingKL_L16to32_highcorr5_r1_iteration5/iteration5.json"
EXEC=0;BG=0;for x in "$@";do case "$x" in --execute)EXEC=1;;--background)BG=1;;*)echo "Usage: $0 [--execute] [--background]" >&2;exit 2;;esac;done
printf 'run=%s\n' "$RUN";[[ $EXEC -eq 1 ]]||exit 0;test ! -e "$RUN"||{ echo "run exists: $RUN" >&2;exit 1;};mkdir -p "$RUN/logs";CMD=("$PY" -B "$ROOT/perfect_blocking_upsampling/scripts/train_lam1p0_l16to32_directcoarse_observable_flow.py" --run-dir "$RUN" --source-checkpoint "$CK" --kernel "$K" --n 5000 --train-count 4000 --val-count 500 --epochs 20 --batch-size 256 --lr 2e-6 --seed 2026081219);if [[ $BG -eq 1 ]];then nohup "${CMD[@]}" >"$RUN/logs/run.log" 2>&1 </dev/null & echo $! >"$RUN/submit_pid.txt";echo "background_pid=$!";else exec "${CMD[@]}";fi
