#!/usr/bin/env bash
# Widen only phi4: the loss is zero once its proposal width reaches 1.05 times
# the native width, so it does not force the distribution back to equality.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
if [[ -x "$REPO_ROOT/../../.venv/bin/python" ]]; then PYTHON="$REPO_ROOT/../../.venv/bin/python"; elif [[ -x "$REPO_ROOT/../.venv/bin/python" ]]; then PYTHON="$REPO_ROOT/../.venv/bin/python"; else echo 'shared Python not found' >&2; exit 1; fi
DRIVER="$REPO_ROOT/perfect_blocking_upsampling/scripts/train_lam1p0_l16to32_rqspline_finetune.py"
BASE="$REPO_ROOT/perfect_blocking_upsampling/runs/lam1p0/training/lam1p0_L16to32_current5x5_phi2_kurtosis_rqspline_N5000_20260803T121013Z"
CHECKPOINT="$BASE/checkpoints/checkpoint_best_nll.pt"; NORMALIZATION="$BASE/normalization_metadata.json"; KERNEL="$REPO_ROOT/perfect_blocking/perfect_blocking_lam1p0/kernels/final/chosen_kernel.json"
RUN_LABEL="lam1p0_L16to32_current5x5_phi4_widthfloor_N5000_$(date +%Y%m%dT%H%M%SZ)"; RUN_DIR="$REPO_ROOT/perfect_blocking_upsampling/runs/lam1p0/training/$RUN_LABEL"
EXECUTE=0; BACKGROUND=0
for arg in "$@"; do case "$arg" in --execute) EXECUTE=1 ;; --background) BACKGROUND=1 ;; *) echo "usage: $0 [--execute] [--background]" >&2; exit 2 ;; esac; done
for p in "$PYTHON" "$DRIVER" "$CHECKPOINT" "$NORMALIZATION" "$KERNEL"; do [[ -e "$p" ]] || { echo "missing: $p" >&2; exit 1; }; done
CMD=("$PYTHON" -B "$DRIVER" --run-dir "$RUN_DIR" --source-checkpoint "$CHECKPOINT" --kernel-path "$KERNEL" --normalization-metadata "$NORMALIZATION" --source-start-index 0 --total-count 5000 --train-count 4000 --val-count 500 --test-count 500 --epochs 3 --patience 2 --eval-every 1 --exact-eval-every 1 --raw-eval-count 500 --global-chains 4 --global-sweeps 5 --local-chains 4 --local-sweeps 2 --batch-size 128 --lr 5e-6 --random-seed 2026080414 --device cpu --obs-weights action_density=0.025,phi2=0.020,phi4=0.030,local_kurtosis_ratio=0.04,NN=0.012,2nn=0.004,diag=0.004,G_pmin_avg=0 --proposal-phi4-min-std-ratio 1.05 --proposal-phi4-min-std-weight 0.20)
echo "run directory: $RUN_DIR"; printf 'command: '; printf '%q ' "${CMD[@]}"; printf '\n'
if [[ "$EXECUTE" -eq 0 ]]; then echo 'Prepared only. Re-run with --execute to start; add --background for nohup.'; exit 0; fi
mkdir -p "$RUN_DIR/logs"; { echo "run_id=$RUN_LABEL"; echo 'phi4_width_floor=native_std_ratio 1.05; weight 0.20'; printf 'command='; printf '%q ' "${CMD[@]}"; printf '\n'; } > "$RUN_DIR/submit_manifest.txt"
if [[ "$BACKGROUND" -eq 1 ]]; then nohup "${CMD[@]}" > "$RUN_DIR/logs/run.log" 2>&1 & echo "$!" > "$RUN_DIR/submit_pid.txt"; echo "started background PID $!"; else exec "${CMD[@]}"; fi
