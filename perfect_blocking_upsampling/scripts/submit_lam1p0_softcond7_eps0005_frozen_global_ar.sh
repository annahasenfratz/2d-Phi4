#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="$ROOT/../../.venv/bin/python"
CHECKPOINT="$ROOT/perfect_blocking_upsampling/runs/lam1p0/training/lam1p0_L16to32_softcond7_pureNLL_N5000_20260808T230212Z/stage_oo/checkpoints/checkpoint_best_nll.pt"
KERNEL="$ROOT/perfect_blocking_upsampling/outputs/global_ar_lam1p0/softcond7_frozen_flow_direction_scan/kernel_eps0p00050.json"
OUT="$ROOT/perfect_blocking_upsampling/outputs/global_ar_lam1p0/softcond7_eps0005_frozen_N5000"
N=5000
EXEC=0
BACKGROUND=0

for argument in "$@"; do
  case "$argument" in
    --execute) EXEC=1 ;;
    --background) BACKGROUND=1 ;;
    *) echo "Usage: $0 [--execute] [--background]" >&2; exit 2 ;;
  esac
done

CMD=("$PY" -B "$ROOT/perfect_blocking_upsampling/scripts/audit_lam1p0_softcond7_global_ar.py"
  --checkpoint "$CHECKPOINT" --kernel "$KERNEL" --out "$OUT" --n "$N")
printf 'checkpoint=%s\nkernel=%s\nout=%s\nn=%s\n' "$CHECKPOINT" "$KERNEL" "$OUT" "$N"
[[ $EXEC -eq 1 ]] || exit 0
mkdir -p "$OUT/logs"
if [[ $BACKGROUND -eq 1 ]]; then
  nohup "${CMD[@]}" >"$OUT/logs/run.log" 2>&1 < /dev/null &
  echo $! >"$OUT/submit_pid.txt"
  echo "background_pid=$!"
else
  exec "${CMD[@]}"
fi
