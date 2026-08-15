#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
if [[ -x "$REPO_ROOT/../../.venv/bin/python" ]]; then PYTHON="$REPO_ROOT/../../.venv/bin/python"; elif [[ -x "$REPO_ROOT/../.venv/bin/python" ]]; then PYTHON="$REPO_ROOT/../.venv/bin/python"; else echo 'shared Python not found' >&2; exit 1; fi
UPSCALE="$REPO_ROOT/perfect_blocking_upsampling/scripts/compare_lam1p0_raw_upscaled_vs_native.py"
MEASURE="$REPO_ROOT/perfect_blocking/scripts/block_and_measure.py"
MEASURE_CFG="$REPO_ROOT/perfect_blocking/perfect_blocking_lam1p0/run_configs/kernel_training_lam1p0.yaml"
CHECKPOINT="$REPO_ROOT/perfect_blocking_upsampling/runs/lam1p0/training/lam1p0_L16to32_current5x5_stronger_phi4_action_stdmatch_N5000_20260803T135504Z/checkpoints/checkpoint_best_nll.pt"
KERNEL="$REPO_ROOT/perfect_blocking/perfect_blocking_lam1p0/kernels/final/chosen_kernel.json"
NATIVE="$REPO_ROOT/data/configs_phi4_2d/lam1p0_kappac0p340301_L32/configs.npz"
COUNT=5000; STAMP="$(date +%Y%m%dT%H%M%SZ)"
RUN_DIR="$REPO_ROOT/perfect_blocking_upsampling/runs/lam1p0/upscaling/raw_L16toL32_current5x5_stronger_stdmatch_N${COUNT}_${STAMP}"
FIELDS="$REPO_ROOT/data/configs_phi4_2d/upscaled/lam1p0_kappac0p340301_L16_to_L32_current5x5_stronger_stdmatch_N${COUNT}_${STAMP}.npz"
EXECUTE=0; BACKGROUND=0
for arg in "$@"; do case "$arg" in --execute) EXECUTE=1 ;; --background) BACKGROUND=1 ;; *) echo "usage: $0 [--execute] [--background]" >&2; exit 2 ;; esac; done
for p in "$PYTHON" "$UPSCALE" "$MEASURE" "$MEASURE_CFG" "$CHECKPOINT" "$KERNEL" "$NATIVE"; do [[ -e "$p" ]] || { echo "missing: $p" >&2; exit 1; }; done
UP=("$PYTHON" -B "$UPSCALE" --checkpoint "$CHECKPOINT" --kernel "$KERNEL" --native-l32 "$NATIVE" --output-dir "$RUN_DIR" --count "$COUNT" --start-index 0 --seed 2026080410 --batch-size 128 --label current5x5_stronger_stdmatch_raw_upscaled --upscaled-config-output "$FIELDS")
D=("$PYTHON" -B "$MEASURE" --config "$MEASURE_CFG" --mode native --configs "$NATIVE" --output-csv "$RUN_DIR/observables/direct_L32_observables_per_config.csv" --gk-output-csv "$RUN_DIR/observables/direct_L32_Gk_per_config.csv" --gk-summary-output-csv "$RUN_DIR/observables/direct_L32_Gk_summary_per_config.csv" --fine-L 32 --coarse-L 32 --lambda 1.0 --kappa-f 0.340301 --kappa-c 0.340301 --max-configs "$COUNT" --source-prefix direct_L32 --ensemble-label direct_L32 --manifest "$RUN_DIR/observables/direct_L32_measurement_manifest.json")
U=("$PYTHON" -B "$MEASURE" --config "$MEASURE_CFG" --mode native --configs "$FIELDS" --output-csv "$RUN_DIR/observables/upscaled_L16_to_L32_observables_per_config.csv" --gk-output-csv "$RUN_DIR/observables/upscaled_L16_to_L32_Gk_per_config.csv" --gk-summary-output-csv "$RUN_DIR/observables/upscaled_L16_to_L32_Gk_summary_per_config.csv" --fine-L 32 --coarse-L 32 --lambda 1.0 --kappa-f 0.340301 --kappa-c 0.340301 --max-configs "$COUNT" --source-prefix upscaled_L16_to_L32_stronger_stdmatch --ensemble-label upscaled_L16_to_L32_stronger_stdmatch --manifest "$RUN_DIR/observables/upscaled_L16_to_L32_measurement_manifest.json")
echo "run directory: $RUN_DIR"; echo "generated fields: $FIELDS"
if [[ "$EXECUTE" -eq 0 ]]; then echo 'Prepared only. Re-run with --execute to start; add --background for nohup.'; exit 0; fi
mkdir -p "$RUN_DIR/logs" "$RUN_DIR/observables"
{ echo "checkpoint=$CHECKPOINT"; echo "generated_fields=$FIELDS"; } > "$RUN_DIR/submit_manifest.txt"
if [[ "$BACKGROUND" -eq 1 ]]; then printf -v A '%q ' "${UP[@]}"; printf -v B '%q ' "${D[@]}"; printf -v C '%q ' "${U[@]}"; nohup bash -c "$A && $B && $C" > "$RUN_DIR/logs/run.log" 2>&1 & echo "$!" > "$RUN_DIR/submit_pid.txt"; echo "started background PID $!"; else "${UP[@]}"; "${D[@]}"; "${U[@]}"; fi
