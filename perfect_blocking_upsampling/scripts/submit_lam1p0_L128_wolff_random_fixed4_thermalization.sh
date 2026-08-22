#!/usr/bin/env bash
# Random-start control for the L128 fixed-four-cluster Wolff thermalization test.
# Usage: bash $0 [--target-sweeps N] [--resume] [--execute] [--background]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${PYTHON:-$ROOT/../../.venv/bin/python}"
RUN="$ROOT/perfect_blocking_upsampling/outputs/wolff_rethermalization_lam1p0/L128_random_N1500_radial_wolff_fixed4_thermalization_r1"
TARGET=100; RESUME=0; EXECUTE=0; BACKGROUND=0
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --target-sweeps) TARGET="${2:-}"; shift 2 ;;
    --resume) RESUME=1; shift ;;
    --execute) EXECUTE=1; shift ;;
    --background) BACKGROUND=1; shift ;;
    *) echo "usage: $0 [--target-sweeps N] [--resume] [--execute] [--background]" >&2; exit 2 ;;
  esac
done
[[ "$TARGET" =~ ^[0-9]+$ ]] || { echo "--target-sweeps must be non-negative" >&2; exit 2; }
CMD=(env PYTHONUNBUFFERED=1 "$PY" -B "$ROOT/perfect_blocking_upsampling/scripts/run_lam1p0_l256_wolff_rethermalization.py"
  --random-initialization --random-std 0.7 --run-dir "$RUN" --L 128 --start-index 0 --n-chains 1500
  --target-sweeps "$TARGET" --checkpoint-every 25 --measurement-batch-size 25 --clusters-per-sweep 4 --seed 2026082108)
[[ "$RESUME" -eq 1 ]] && CMD+=(--resume)
printf 'run_dir=%s\ninitialization=independent N(0,0.7^2) L128 fields\nalgorithm=one radial sweep plus four fixed Wolff clusters per sweep\ncomparison=matched against the existing L128 upscaled fixed4 Wolff rethermalization\nrestart: --resume --target-sweeps NEW_TOTAL\n' "$RUN"
[[ "$EXECUTE" -eq 1 ]] || exit 0
if [[ "$RESUME" -eq 0 && -e "$RUN" ]]; then echo "run already exists: $RUN" >&2; exit 1; fi
mkdir -p "$RUN/logs"
if [[ "$BACKGROUND" -eq 1 ]]; then nohup "${CMD[@]}" >"$RUN/logs/run.log" 2>&1 </dev/null & echo "$!" >"$RUN/submit_pid.txt"; echo "background_pid=$!"; else exec "${CMD[@]}"; fi
