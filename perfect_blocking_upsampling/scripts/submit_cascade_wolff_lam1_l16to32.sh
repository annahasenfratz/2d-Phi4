#!/usr/bin/env bash
# First level of CASCADE-WOLFF-LAM1.
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
[[ "$BACKGROUND" -eq 0 || "$EXECUTE" -eq 1 ]] || { echo '--background requires --execute' >&2; exit 2; }

# This file is the fresh pure-NLL Ethan-7x7 flow applied to direct native L16
# configurations.  We deliberately take its first 1500 fields; no new flow
# sample, kernel, or native configuration is generated.
SOURCE="$ROOT/perfect_blocking_upsampling/outputs/flow_input_audit_lam1p0/ethan7_fresh_pureNLL_direct_native_L16_N5000_20260818/sweep0_flow_phi.npz"
RUN="$ROOT/perfect_blocking_upsampling/outputs/cascade_wolff_lam1p0/CASCADE-WOLFF-LAM1/L16toL32_kc0p340301_kf0p340301/N1500_r1"
DRIVER="$ROOT/perfect_blocking_upsampling/scripts/run_lam1p0_l256_wolff_rethermalization.py"

# Consult the registry before a new execution.  The entry is pre-registered as
# WOLFF-CASCADE-L32-001 and this launcher refuses to duplicate a live run.
grep -q '^WOLFF-CASCADE-L32-001,' "$ROOT/registry/runs.csv" || { echo 'missing registry entry WOLFF-CASCADE-L32-001' >&2; exit 1; }
CMD=(env PYTHONUNBUFFERED=1 "$PY" -B "$DRIVER"
  --source "$SOURCE" --run-dir "$RUN" --L 32 --start-index 0 --n-chains 1500
  --target-sweeps "$TARGET" --checkpoint-every 25 --measurement-batch-size 50
  --clusters-per-sweep 4 --seed 2026082601 --kappa 0.340301)
[[ "$RESUME" -eq 1 ]] && CMD+=(--resume)

printf 'study=CASCADE-WOLFF-LAM1\nrun_id=WOLFF-CASCADE-L32-001\n'
printf 'source=%s\nrun_dir=%s\n' "$SOURCE" "$RUN"
printf 'L16 direct-native -> L32 Ethan-7x7 flow sweep zero; N=1500\n'
printf 'one Wolff sweep=one radial heat-bath sweep of every site plus four embedded sign-cluster updates per configuration; target_sweeps=%s\n' "$TARGET"
printf 'next-level source after completion=%s/checkpoints/checkpoint_sweep_%04d.npz\n' "$RUN" "$TARGET"
printf 'restart: --resume --target-sweeps NEW_TOTAL\n'
[[ "$EXECUTE" -eq 1 ]] || exit 0

[[ -f "$SOURCE" ]] || { echo "missing sweep-zero source: $SOURCE" >&2; exit 1; }
if [[ "$RESUME" -eq 0 && -e "$RUN" ]]; then echo "run already initialized: $RUN; use --resume" >&2; exit 1; fi
mkdir -p "$RUN/logs"
if [[ "$BACKGROUND" -eq 1 ]]; then
  nohup "${CMD[@]}" >"$RUN/logs/run.log" 2>&1 </dev/null &
  echo "$!" >"$RUN/submit_pid.txt"
  printf 'background_pid=%s\n' "$!"
else
  exec "${CMD[@]}"
fi
