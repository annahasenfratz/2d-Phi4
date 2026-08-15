#!/usr/bin/env bash
# Usage: bash $0 [--execute] [--background]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RUN_ROOT="${RUN_ROOT:-$ROOT/perfect_blocking_upsampling/runs/lam1p0/training/lam1p0_L16to32_alternatingKL5_flow_tailrefine_r1}"
SOURCE_CHECKPOINT="$ROOT/perfect_blocking_upsampling/runs/lam1p0/training/lam1p0_L16to32_alternatingKL_highcorr5_r1_iteration5/flow5/stage_oo/checkpoints/checkpoint_best_nll.pt"
KERNEL="$ROOT/perfect_blocking/perfect_blocking_lam1p0/kernels/candidates/alternatingKL_L16to32_highcorr5_r1_iteration5/iteration5.json"
EXEC=0; BACKGROUND=0
for arg in "$@"; do
  case "$arg" in
    --execute) EXEC=1 ;;
    --background) BACKGROUND=1 ;;
    *) echo "Usage: $0 [--execute] [--background]" >&2; exit 2 ;;
  esac
done
printf 'run=%s\nsource=%s\nkernel=%s\n' "$RUN_ROOT" "$SOURCE_CHECKPOINT" "$KERNEL"
[[ $EXEC -eq 1 ]] || exit 0
[[ -f "$SOURCE_CHECKPOINT" && -f "$KERNEL" ]] || { echo 'missing source checkpoint or kernel' >&2; exit 1; }
mkdir -p "$RUN_ROOT/logs"
CMD=(env RUN_ROOT="$RUN_ROOT" SOURCE_CHECKPOINT="$SOURCE_CHECKPOINT" KERNEL="$KERNEL"
  bash "$ROOT/perfect_blocking_upsampling/scripts/run_lam1p0_l16to32_alternatingkl5_flow_tailrefine_stages.sh")
if [[ $BACKGROUND -eq 1 ]]; then
  nohup "${CMD[@]}" >"$RUN_ROOT/logs/run.log" 2>&1 </dev/null &
  echo $! >"$RUN_ROOT/submit_pid.txt"
  echo "background_pid=$!"
else
  exec "${CMD[@]}"
fi
