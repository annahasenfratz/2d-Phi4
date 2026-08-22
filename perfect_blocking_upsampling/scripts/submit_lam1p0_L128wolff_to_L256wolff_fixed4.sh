#!/usr/bin/env bash
# Flow-only L128->L256 upscaling from the corrected L128 Wolff endpoint,
# followed by 100 fixed-four-cluster Wolff rethermalization sweeps at L256.
# Usage: bash $0 [--execute] [--background]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${PYTHON:-$ROOT/../../.venv/bin/python}"
CONFIG="$ROOT/perfect_blocking_upsampling/run_configs/lam1p0_wolffL128_to_L256_ethan7_flowonly.json"
L128_SOURCE="$ROOT/perfect_blocking_upsampling/outputs/wolff_rethermalization_lam1p0/L128_ethan7_sweep0_N1500_radial_wolff_fixed4_r2/checkpoints/checkpoint_sweep_0100.npz"
FLOW_RUN="$ROOT/perfect_blocking_upsampling/outputs/hmc_upscale_chain_lam1p0/L128toL256/L128toL256_from_wolffL128S100_ethan7_flowonly_N1500_r1"
FLOW_SOURCE="$FLOW_RUN/levels/L128toL256/checkpoints/checkpoint_sweep_0000.npz"
WOLFF_RUN="$ROOT/perfect_blocking_upsampling/outputs/wolff_rethermalization_lam1p0/L256_ethan7_from_wolffL128S100_N1500_radial_wolff_fixed4_r1"
PIPE_LOG="$ROOT/perfect_blocking_upsampling/outputs/hmc_upscale_chain_lam1p0/L128toL256/L128toL256_from_wolffL128S100_pipeline_r1.log"
PIPE_PID="$ROOT/perfect_blocking_upsampling/outputs/hmc_upscale_chain_lam1p0/L128toL256/L128toL256_from_wolffL128S100_pipeline_r1.pid"
EXECUTE=0; BACKGROUND=0
for arg in "$@"; do
  case "$arg" in
    --execute) EXECUTE=1 ;;
    --background) BACKGROUND=1 ;;
    *) echo "usage: $0 [--execute] [--background]" >&2; exit 2 ;;
  esac
done
FLOW_CMD=(env PYTHONUNBUFFERED=1 "$PY" -B "$ROOT/perfect_blocking_upsampling/scripts/run_lam1p0_hmc_upscale_chain.py" --config "$CONFIG" --run-dir "$FLOW_RUN" --n-chains 1500 --start-index 0)
WOLFF_CMD=(env PYTHONUNBUFFERED=1 "$PY" -B "$ROOT/perfect_blocking_upsampling/scripts/run_lam1p0_l256_wolff_rethermalization.py" --source "$FLOW_SOURCE" --run-dir "$WOLFF_RUN" --L 256 --start-index 0 --n-chains 1500 --target-sweeps 100 --checkpoint-every 25 --measurement-batch-size 10 --clusters-per-sweep 4 --seed 2026082111)
run_pipeline() {
  if [[ ! -f "$FLOW_RUN/status.json" ]]; then
    [[ ! -e "$FLOW_RUN" ]] || { echo "incomplete flow-only run exists: $FLOW_RUN" >&2; exit 1; }
    mkdir -p "$FLOW_RUN/logs"
    "${FLOW_CMD[@]}" >"$FLOW_RUN/logs/run.log" 2>&1
  fi
  [[ -f "$FLOW_SOURCE" ]] || { echo "flow-only sweep-zero checkpoint missing: $FLOW_SOURCE" >&2; exit 1; }
  if [[ ! -f "$WOLFF_RUN/status.json" ]]; then
    [[ ! -e "$WOLFF_RUN" ]] || { echo "incomplete Wolff run exists: $WOLFF_RUN" >&2; exit 1; }
    mkdir -p "$WOLFF_RUN/logs"
    "${WOLFF_CMD[@]}" >"$WOLFF_RUN/logs/run.log" 2>&1
  fi
}
printf 'L128 Wolff source=%s\nflow-only run=%s\nL256 Wolff run=%s\nalgorithm=Ethan 7x7 flow once, then one radial sweep plus four fixed Wolff clusters for 100 sweeps\n' "$L128_SOURCE" "$FLOW_RUN" "$WOLFF_RUN"
[[ "$EXECUTE" -eq 1 ]] || exit 0
[[ -f "$L128_SOURCE" ]] || { echo "missing corrected L128 Wolff endpoint: $L128_SOURCE" >&2; exit 1; }
if [[ "$BACKGROUND" -eq 1 ]]; then
  nohup bash "$0" --execute >"$PIPE_LOG" 2>&1 </dev/null &
  echo "$!" >"$PIPE_PID"; echo "background_pid=$!"
else
  run_pipeline
fi
