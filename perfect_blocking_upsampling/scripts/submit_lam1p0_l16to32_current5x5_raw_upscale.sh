#!/usr/bin/env bash
# Generate and validate 5,000 raw L32 fields conditioned on direct native L16
# configurations, using the current perfect-blocking kernel and best-NLL flow.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
if [[ -x "$REPO_ROOT/../../.venv/bin/python" ]]; then
  PYTHON="$REPO_ROOT/../../.venv/bin/python"
elif [[ -x "$REPO_ROOT/../.venv/bin/python" ]]; then
  PYTHON="$REPO_ROOT/../.venv/bin/python"
else
  printf 'Could not find the shared virtual-environment Python.\n' >&2
  exit 1
fi

DRIVER="$REPO_ROOT/perfect_blocking_upsampling/scripts/compare_lam1p0_raw_upscaled_vs_native.py"
CHECKPOINT="$REPO_ROOT/perfect_blocking_upsampling/runs/lam1p0/training/lam1p0_L16to32_current5x5_phi2_kurtosis_rqspline_N5000_20260803T121013Z/checkpoints/checkpoint_best_nll.pt"
KERNEL="$REPO_ROOT/perfect_blocking/perfect_blocking_lam1p0/kernels/final/chosen_kernel.json"
NATIVE_L32="$REPO_ROOT/data/configs_phi4_2d/lam1p0_kappac0p340301_L32/configs.npz"

COUNT=5000
SEED=2026080402
STAMP="$(date +%Y%m%dT%H%M%SZ)"
RUN_NAME="raw_L16toL32_current5x5_bestnll_N${COUNT}_${STAMP}"
RUN_DIR="$REPO_ROOT/perfect_blocking_upsampling/runs/lam1p0/upscaling/$RUN_NAME"
UPSCALED_CONFIGS="$REPO_ROOT/data/configs_phi4_2d/upscaled/lam1p0_kappac0p340301_L16_to_L32_current5x5_bestnll_N${COUNT}_${STAMP}.npz"

EXECUTE=0
BACKGROUND=0
for arg in "$@"; do
  case "$arg" in
    --execute) EXECUTE=1 ;;
    --background) BACKGROUND=1 ;;
    *) printf 'usage: %s [--execute] [--background]\n' "$0" >&2; exit 2 ;;
  esac
done

for required in "$PYTHON" "$DRIVER" "$CHECKPOINT" "$KERNEL" "$NATIVE_L32"; do
  [[ -e "$required" ]] || { printf 'missing required path: %s\n' "$required" >&2; exit 1; }
done

CMD=("$PYTHON" -B "$DRIVER"
  --checkpoint "$CHECKPOINT" --kernel "$KERNEL" --native-l32 "$NATIVE_L32"
  --output-dir "$RUN_DIR" --count "$COUNT" --start-index 0 --seed "$SEED" --batch-size 128
  --label current5x5_bestnll_raw_upscaled
  --upscaled-config-output "$UPSCALED_CONFIGS")

printf 'run directory: %s\n' "$RUN_DIR"
printf 'generated fields: %s\n' "$UPSCALED_CONFIGS"
printf 'command: '
printf '%q ' "${CMD[@]}"
printf '\n'
if [[ "$EXECUTE" -eq 0 ]]; then
  printf 'Prepared only. Re-run with --execute to start; add --background for nohup.\n'
  exit 0
fi

mkdir -p "$RUN_DIR/logs"
{
  printf 'run_name=%s\n' "$RUN_NAME"
  printf 'generated_fields=%s\n' "$UPSCALED_CONFIGS"
  printf 'command='
  printf '%q ' "${CMD[@]}"
  printf '\n'
} > "$RUN_DIR/submit_manifest.txt"
if [[ "$BACKGROUND" -eq 1 ]]; then
  nohup "${CMD[@]}" > "$RUN_DIR/logs/run.log" 2>&1 &
  printf '%s\n' "$!" > "$RUN_DIR/submit_pid.txt"
  printf 'started background PID %s\n' "$!"
else
  exec "${CMD[@]}"
fi
