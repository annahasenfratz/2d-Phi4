#!/usr/bin/env bash
# Build the shared clean 5000-root sweep-zero ensemble: direct L32 -> L64
# using the Ethan 7x7 flow/kernel, with no HMC at this stage.
# Usage: bash $0 [--execute] [--background]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${PYTHON:-$ROOT/../../.venv/bin/python}"
[[ -x "$PY" ]] || PY="$ROOT/../.venv/bin/python"
[[ -x "$PY" ]] || { echo "shared Python interpreter not found; set PYTHON=/path/to/python" >&2; exit 1; }
EXECUTE=0; BACKGROUND=0
for arg in "$@"; do case "$arg" in --execute) EXECUTE=1 ;; --background) BACKGROUND=1 ;; *) echo "usage: $0 [--execute] [--background]" >&2; exit 2 ;; esac; done
N_CHAINS=5000
CONFIG="$ROOT/perfect_blocking_upsampling/run_configs/lam1p0_flowonly_L32toL64_ethan7_nativeL32.json"
RUN="$ROOT/perfect_blocking_upsampling/outputs/flow_only_lam1p0/L32toL64/L32toL64_ethan7_nativeL32_N5000_sweep0_r1"
CMD=(env PYTHONUNBUFFERED=1 "$PY" -B "$ROOT/perfect_blocking_upsampling/scripts/run_lam1p0_hmc_upscale_chain.py" --config "$CONFIG" --run-dir "$RUN" --n-chains "$N_CHAINS" --start-index 0)
printf 'direct native L32 -> Ethan-7x7 flow-only L64: N=%s\noutput=%s\n' "$N_CHAINS" "$RUN"
[[ "$EXECUTE" -eq 1 ]] || exit 0
[[ ! -e "$RUN" ]] || { echo "refusing to overwrite existing output: $RUN" >&2; exit 1; }
mkdir -p "$RUN/logs"
if [[ "$BACKGROUND" -eq 1 ]]; then
  nohup "${CMD[@]}" >"$RUN/logs/run.log" 2>&1 </dev/null &
  echo "$!" >"$RUN/submit_pid.txt"; echo "background_pid=$!"
else
  exec "${CMD[@]}"
fi
