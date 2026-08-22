#!/usr/bin/env bash
# Standard Wolff+radial rethermalization of the clean native-L32 -> L64
# Ethan-7x7 sweep-zero ensemble at kappa_c=kappa_f=0.340301.
# Usage: bash $0 [--target-sweeps N] [--resume] [--execute] [--background]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${PYTHON:-$ROOT/../../.venv/bin/python}"
[[ -x "$PY" ]] || PY="$ROOT/../.venv/bin/python"
[[ -x "$PY" ]] || { echo "shared Python interpreter not found; set PYTHON=/path/to/python" >&2; exit 1; }
TARGET=100; RESUME=0; EXECUTE=0; BACKGROUND=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --target-sweeps) TARGET="${2:-}"; shift 2 ;;
    --resume) RESUME=1; shift ;;
    --execute) EXECUTE=1; shift ;;
    --background) BACKGROUND=1; shift ;;
    *) echo "usage: $0 [--target-sweeps N] [--resume] [--execute] [--background]" >&2; exit 2 ;;
  esac
done
[[ "$TARGET" =~ ^[0-9]+$ ]] || { echo '--target-sweeps must be non-negative' >&2; exit 2; }
SOURCE="$ROOT/perfect_blocking_upsampling/outputs/flow_only_lam1p0/L32toL64/L32toL64_ethan7_nativeL32_N5000_sweep0_r1/levels/L32toL64/checkpoints/checkpoint_sweep_0000.npz"
RUN="$ROOT/perfect_blocking_upsampling/outputs/wolff_rethermalization_lam1p0/L64_ethan7_nativeL32_sweep0_N5000_kc0p340301_kf0p340301_radial_wolff_fixed4_r1"
CMD=(env PYTHONUNBUFFERED=1 "$PY" -B "$ROOT/perfect_blocking_upsampling/scripts/run_lam1p0_l256_wolff_rethermalization.py" --source "$SOURCE" --run-dir "$RUN" --L 64 --start-index 0 --n-chains 5000 --target-sweeps "$TARGET" --checkpoint-every 25 --measurement-batch-size 50 --clusters-per-sweep 4 --seed 2026082536 --kappa 0.340301)
[[ "$RESUME" -eq 1 ]] && CMD+=(--resume)
printf 'source=%s\nrun_dir=%s\nalgorithm=one radial heat-bath sweep plus four embedded Wolff clusters per configuration\nkappa_c=kappa_f=0.340301; target_sweeps=%s\nrestart: --resume --target-sweeps NEW_TOTAL\n' "$SOURCE" "$RUN" "$TARGET"
[[ "$EXECUTE" -eq 1 ]] || exit 0
[[ -f "$SOURCE" ]] || { echo "missing L64 sweep-zero source: $SOURCE" >&2; exit 1; }
if [[ "$RESUME" -eq 0 && -e "$RUN" ]]; then echo "run already exists: $RUN" >&2; exit 1; fi
mkdir -p "$RUN/logs"
if [[ "$BACKGROUND" -eq 1 ]]; then
  nohup "${CMD[@]}" >"$RUN/logs/run.log" 2>&1 </dev/null &
  echo "$!" >"$RUN/submit_pid.txt"; echo "background_pid=$!"
else
  exec "${CMD[@]}"
fi
