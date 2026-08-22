#!/usr/bin/env bash
# Standard Wolff+radial rethermalization of the combined 3000-root clean
# L32(kappa_c=.340100)->L64 Ethan-7x7 sweep-zero ensemble at kappa_f=.340301.
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
SOURCE="$ROOT/perfect_blocking_upsampling/outputs/combined_initializations_lam1p0/L32k0p340100_replicas_N3000_toL64_ethan7_sweep0.npz"
RUN="$ROOT/perfect_blocking_upsampling/outputs/wolff_rethermalization_lam1p0/L64_ethan7_L32k0p340100_replicas_N3000_sweep0_kc0p340100_kf0p340301_radial_wolff_fixed4_r1"
CMD=(env PYTHONUNBUFFERED=1 "$PY" -B "$ROOT/perfect_blocking_upsampling/scripts/run_lam1p0_l256_wolff_rethermalization.py" --source "$SOURCE" --run-dir "$RUN" --L 64 --start-index 0 --n-chains 3000 --target-sweeps "$TARGET" --checkpoint-every 25 --measurement-batch-size 50 --clusters-per-sweep 4 --seed 2026082537 --kappa 0.340301)
[[ "$RESUME" -eq 1 ]] && CMD+=(--resume)
printf 'source=%s\nrun_dir=%s\nalgorithm=one radial heat-bath sweep plus four embedded Wolff clusters per configuration\nkappa_c=.340100, kappa_f=.340301; target_sweeps=%s\nrestart: --resume --target-sweeps NEW_TOTAL\n' "$SOURCE" "$RUN" "$TARGET"
[[ "$EXECUTE" -eq 1 ]] || exit 0
[[ -f "$SOURCE" ]] || { echo "missing combined L64 sweep-zero source: $SOURCE" >&2; exit 1; }
if [[ "$RESUME" -eq 0 && -e "$RUN" ]]; then echo "run already exists: $RUN" >&2; exit 1; fi
mkdir -p "$RUN/logs"
if [[ "$BACKGROUND" -eq 1 ]]; then
  nohup "${CMD[@]}" >"$RUN/logs/run.log" 2>&1 </dev/null &
  echo "$!" >"$RUN/submit_pid.txt"; echo "background_pid=$!"
else
  exec "${CMD[@]}"
fi
