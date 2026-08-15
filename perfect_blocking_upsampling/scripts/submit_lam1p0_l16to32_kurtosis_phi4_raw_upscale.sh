#!/usr/bin/env bash
# Generate 5,000 raw L32 fields with the final gentle kurtosis/phi4 flow, then
# create notebook-ready direct/upscaled observable and structure-factor CSVs.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
if [[ -x "$REPO_ROOT/../../.venv/bin/python" ]]; then
  PYTHON="$REPO_ROOT/../../.venv/bin/python"
elif [[ -x "$REPO_ROOT/../.venv/bin/python" ]]; then
  PYTHON="$REPO_ROOT/../.venv/bin/python"
else
  printf 'Could not find the shared virtual-environment Python.\n' >&2; exit 1
fi

UPSCALE_DRIVER="$REPO_ROOT/perfect_blocking_upsampling/scripts/compare_lam1p0_raw_upscaled_vs_native.py"
MEASURE_DRIVER="$REPO_ROOT/perfect_blocking/scripts/block_and_measure.py"
MEASURE_CONFIG="$REPO_ROOT/perfect_blocking/perfect_blocking_lam1p0/run_configs/kernel_training_lam1p0.yaml"
CHECKPOINT="$REPO_ROOT/perfect_blocking_upsampling/runs/lam1p0/training/lam1p0_L16to32_current5x5_gentle_kurtosis_phi4_shape_N5000_20260803T131503Z/checkpoints/checkpoint_best_nll.pt"
KERNEL="$REPO_ROOT/perfect_blocking/perfect_blocking_lam1p0/kernels/final/chosen_kernel.json"
NATIVE_L32="$REPO_ROOT/data/configs_phi4_2d/lam1p0_kappac0p340301_L32/configs.npz"
COUNT=5000
SEED=2026080406
STAMP="$(date +%Y%m%dT%H%M%SZ)"
RUN_NAME="raw_L16toL32_current5x5_kurtosis_phi4_N${COUNT}_${STAMP}"
RUN_DIR="$REPO_ROOT/perfect_blocking_upsampling/runs/lam1p0/upscaling/$RUN_NAME"
UPSCALED_CONFIGS="$REPO_ROOT/data/configs_phi4_2d/upscaled/lam1p0_kappac0p340301_L16_to_L32_current5x5_kurtosis_phi4_N${COUNT}_${STAMP}.npz"

EXECUTE=0; BACKGROUND=0
for arg in "$@"; do
  case "$arg" in
    --execute) EXECUTE=1 ;;
    --background) BACKGROUND=1 ;;
    *) printf 'usage: %s [--execute] [--background]\n' "$0" >&2; exit 2 ;;
  esac
done
for required in "$PYTHON" "$UPSCALE_DRIVER" "$MEASURE_DRIVER" "$MEASURE_CONFIG" "$CHECKPOINT" "$KERNEL" "$NATIVE_L32"; do
  [[ -e "$required" ]] || { printf 'missing required path: %s\n' "$required" >&2; exit 1; }
done

UPSCALE_CMD=("$PYTHON" -B "$UPSCALE_DRIVER"
  --checkpoint "$CHECKPOINT" --kernel "$KERNEL" --native-l32 "$NATIVE_L32"
  --output-dir "$RUN_DIR" --count "$COUNT" --start-index 0 --seed "$SEED" --batch-size 128
  --label current5x5_kurtosis_phi4_raw_upscaled --upscaled-config-output "$UPSCALED_CONFIGS")
DIRECT_MEASURE_CMD=("$PYTHON" -B "$MEASURE_DRIVER" --config "$MEASURE_CONFIG" --mode native --configs "$NATIVE_L32"
  --output-csv "$RUN_DIR/observables/direct_L32_observables_per_config.csv"
  --gk-output-csv "$RUN_DIR/observables/direct_L32_Gk_per_config.csv"
  --gk-summary-output-csv "$RUN_DIR/observables/direct_L32_Gk_summary_per_config.csv"
  --fine-L 32 --coarse-L 32 --lambda 1.0 --kappa-f 0.340301 --kappa-c 0.340301 --max-configs "$COUNT"
  --source-prefix direct_L32 --ensemble-label direct_L32
  --manifest "$RUN_DIR/observables/direct_L32_measurement_manifest.json")
UPSCALED_MEASURE_CMD=("$PYTHON" -B "$MEASURE_DRIVER" --config "$MEASURE_CONFIG" --mode native --configs "$UPSCALED_CONFIGS"
  --output-csv "$RUN_DIR/observables/upscaled_L16_to_L32_observables_per_config.csv"
  --gk-output-csv "$RUN_DIR/observables/upscaled_L16_to_L32_Gk_per_config.csv"
  --gk-summary-output-csv "$RUN_DIR/observables/upscaled_L16_to_L32_Gk_summary_per_config.csv"
  --fine-L 32 --coarse-L 32 --lambda 1.0 --kappa-f 0.340301 --kappa-c 0.340301 --max-configs "$COUNT"
  --source-prefix upscaled_L16_to_L32_kurtosis_phi4 --ensemble-label upscaled_L16_to_L32_kurtosis_phi4
  --manifest "$RUN_DIR/observables/upscaled_L16_to_L32_measurement_manifest.json")

printf 'run directory: %s\n' "$RUN_DIR"
printf 'generated fields: %s\n' "$UPSCALED_CONFIGS"
if [[ "$EXECUTE" -eq 0 ]]; then
  printf 'Prepared only. Re-run with --execute to start; add --background for nohup.\n'
  exit 0
fi

mkdir -p "$RUN_DIR/logs" "$RUN_DIR/observables"
{
  printf 'run_name=%s\n' "$RUN_NAME"
  printf 'checkpoint=%s\n' "$CHECKPOINT"
  printf 'generated_fields=%s\n' "$UPSCALED_CONFIGS"
  printf 'upscale_command='; printf '%q ' "${UPSCALE_CMD[@]}"; printf '\n'
} > "$RUN_DIR/submit_manifest.txt"

run_all() {
  "${UPSCALE_CMD[@]}"
  "${DIRECT_MEASURE_CMD[@]}"
  "${UPSCALED_MEASURE_CMD[@]}"
}

if [[ "$BACKGROUND" -eq 1 ]]; then
  printf -v UPSCALE_SHELL '%q ' "${UPSCALE_CMD[@]}"
  printf -v DIRECT_SHELL '%q ' "${DIRECT_MEASURE_CMD[@]}"
  printf -v UPSCALED_SHELL '%q ' "${UPSCALED_MEASURE_CMD[@]}"
  nohup bash -c "$UPSCALE_SHELL && $DIRECT_SHELL && $UPSCALED_SHELL" > "$RUN_DIR/logs/run.log" 2>&1 &
  printf '%s\n' "$!" > "$RUN_DIR/submit_pid.txt"
  printf 'started background PID %s\n' "$!"
else
  run_all
fi
