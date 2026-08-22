#!/usr/bin/env bash
# Flow-only L32 -> L64 initialization, cascade replica r2.
# Usage: bash $0 [--execute] [--background]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${PYTHON:-$ROOT/../../.venv/bin/python}"
[[ -x "$PY" ]] || PY="$ROOT/../.venv/bin/python"
[[ -x "$PY" ]] || { echo "shared Python interpreter not found; set PYTHON=/path/to/python" >&2; exit 1; }
EXECUTE=0; BACKGROUND=0
for arg in "$@"; do
  case "$arg" in
    --execute) EXECUTE=1 ;;
    --background) BACKGROUND=1 ;;
    *) echo "usage: $0 [--execute] [--background]" >&2; exit 2 ;;
  esac
done
[[ "$BACKGROUND" -eq 0 || "$EXECUTE" -eq 1 ]] || { echo '--background requires --execute' >&2; exit 2; }
SOURCE="$ROOT/perfect_blocking_upsampling/outputs/cascade_wolff_lam1p0/CASCADE-WOLFF-LAM1/L16toL32_kc0p340301_kf0p340301/N1500_r2/checkpoints/checkpoint_sweep_0100.npz"
RUN="$ROOT/perfect_blocking_upsampling/outputs/cascade_wolff_lam1p0/CASCADE-WOLFF-LAM1/L32toL64_kc0p340301_kf0p340301/N1500_r2/initialization"
CONFIG="$ROOT/perfect_blocking_upsampling/run_configs/cascade_wolff_lam1_L32toL64_kc0p340301_kf0p340301_r2.json"
DRIVER="$ROOT/perfect_blocking_upsampling/scripts/run_lam1p0_hmc_upscale_chain.py"
grep -q '^FLOW-CASCADE-L64-002,' "$ROOT/registry/runs.csv" || { echo 'missing registry entry FLOW-CASCADE-L64-002' >&2; exit 1; }
CMD=(env PYTHONUNBUFFERED=1 "$PY" -B "$DRIVER" --config "$CONFIG" --run-dir "$RUN" --n-chains 1500 --start-index 0)
printf 'study=CASCADE-WOLFF-LAM1\nrun_id=FLOW-CASCADE-L64-002\n'
printf 'source=%s\nrun_dir=%s\n' "$SOURCE" "$RUN"
printf 'completed L32 r2 fields -> L64 Ethan-7x7 flow-only initialization; N=1500, source indices 1500--2999\n'
printf 'next stage: submit_cascade_wolff_lam1_l32to64_r2_wolff.sh --execute --background\n'
[[ "$EXECUTE" -eq 1 ]] || exit 0
[[ -f "$SOURCE" ]] || { echo "missing completed L32 r2 source: $SOURCE" >&2; exit 1; }
# A failed launch can leave only logs/ and submit_pid.txt; no chain_config
# means the driver never initialized scientific output and retrying is safe.
[[ ! -e "$RUN/chain_config.json" && ! -e "$RUN/status.json" ]] || { echo "run already initialized: $RUN" >&2; exit 1; }
mkdir -p "$RUN/logs"
if [[ "$BACKGROUND" -eq 1 ]]; then
  nohup "${CMD[@]}" >"$RUN/logs/run.log" 2>&1 </dev/null &
  echo "$!" >"$RUN/submit_pid.txt"; printf 'background_pid=%s\n' "$!"
else
  exec "${CMD[@]}"
fi
