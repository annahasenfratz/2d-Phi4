#!/usr/bin/env bash
# Multi-epoch distribution-level continuation from the original best-NLL flow.
# This deliberately uses stronger two-sided distribution losses than the prior
# one-epoch nudges; it is intended to change proposal shape, not only means.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
if [[ -x "$REPO_ROOT/../../.venv/bin/python" ]]; then PYTHON="$REPO_ROOT/../../.venv/bin/python"
elif [[ -x "$REPO_ROOT/../.venv/bin/python" ]]; then PYTHON="$REPO_ROOT/../.venv/bin/python"
else echo 'shared Python not found' >&2; exit 1; fi

DRIVER="$REPO_ROOT/perfect_blocking_upsampling/scripts/train_lam1p0_l16to32_rqspline_finetune.py"
BASE_RUN="$REPO_ROOT/perfect_blocking_upsampling/runs/lam1p0/training/lam1p0_L16to32_current5x5_phi2_kurtosis_rqspline_N5000_20260803T121013Z"
CHECKPOINT="$BASE_RUN/checkpoints/checkpoint_best_nll.pt"
NORMALIZATION="$BASE_RUN/normalization_metadata.json"
KERNEL="$REPO_ROOT/perfect_blocking/perfect_blocking_lam1p0/kernels/final/chosen_kernel.json"
RUN_LABEL="lam1p0_L16to32_current5x5_multiepoch_distribution_N5000_$(date +%Y%m%dT%H%M%SZ)"
RUN_DIR="$REPO_ROOT/perfect_blocking_upsampling/runs/lam1p0/training/$RUN_LABEL"

EXECUTE=0; BACKGROUND=0
for arg in "$@"; do case "$arg" in --execute) EXECUTE=1 ;; --background) BACKGROUND=1 ;; *) echo "usage: $0 [--execute] [--background]" >&2; exit 2 ;; esac; done
for p in "$PYTHON" "$DRIVER" "$CHECKPOINT" "$NORMALIZATION" "$KERNEL"; do [[ -e "$p" ]] || { echo "missing: $p" >&2; exit 1; }; done

CMD=("$PYTHON" -B "$DRIVER"
  --run-dir "$RUN_DIR" --source-checkpoint "$CHECKPOINT" --kernel-path "$KERNEL" --normalization-metadata "$NORMALIZATION"
  --source-start-index 0 --total-count 5000 --train-count 4000 --val-count 500 --test-count 500
  --epochs 6 --patience 3 --eval-every 1 --exact-eval-every 2 --raw-eval-count 500
  --global-chains 4 --global-sweeps 5 --local-chains 4 --local-sweeps 2
  --batch-size 128 --lr 8e-6 --random-seed 2026080411 --device cpu
  --obs-weights action_density=0.050,phi2=0.040,phi4=0.060,local_kurtosis_ratio=0.080,NN=0.015,2nn=0.006,diag=0.006,G_pmin_avg=0
  --two-sided-tail-guard --action-support-weight 0.100 --phi2-support-weight 0.060 --phi4-support-weight 0.120
  --tail-guard-std-weight 0.50 --tail-guard-quantile-weight 0.75 --tail-guard-occupancy-weight 1.00
  --tail-guard-low-occupancy-weight 1.00 --tail-guard-high-occupancy-weight 1.00
  --action-std-match-weight 0.080 --phi4-std-match-weight 0.120
  --local-kurtosis-shape-guard --local-kurtosis-shape-weight 0.080)

echo "run directory: $RUN_DIR"; printf 'command: '; printf '%q ' "${CMD[@]}"; printf '\n'
if [[ "$EXECUTE" -eq 0 ]]; then echo 'Prepared only. Re-run with --execute to start; add --background for nohup.'; exit 0; fi
mkdir -p "$RUN_DIR/logs"
{ echo "run_id=$RUN_LABEL"; echo 'distribution_loss=multi_epoch two_sided action/phi2/phi4 + kurtosis shape'; printf 'command='; printf '%q ' "${CMD[@]}"; printf '\n'; } > "$RUN_DIR/submit_manifest.txt"
if [[ "$BACKGROUND" -eq 1 ]]; then nohup "${CMD[@]}" > "$RUN_DIR/logs/run.log" 2>&1 & echo "$!" > "$RUN_DIR/submit_pid.txt"; echo "started background PID $!"; else exec "${CMD[@]}"; fi
