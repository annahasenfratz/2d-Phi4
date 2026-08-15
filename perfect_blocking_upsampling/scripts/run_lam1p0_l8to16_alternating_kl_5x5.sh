#!/usr/bin/env bash
# Two short alternating fixed-flow KL kernel updates and pure-NLL flow retunes.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="$ROOT/../../.venv/bin/python"
RUN_ROOT="${RUN_ROOT:-$ROOT/perfect_blocking_upsampling/runs/lam1p0/training/lam1p0_L8to16_alternatingKL_5x5_r1}"
KERNEL_ROOT="${KERNEL_ROOT:-$ROOT/perfect_blocking/perfect_blocking_lam1p0/kernels/candidates/alternatingKL_L8to16_5x5_r1}"
K0="$ROOT/perfect_blocking/perfect_blocking_lam1p0/kernels/selected_for_upscaling/best_5x5_retrained_full_objective_eta_included.json"
ZERO_WEIGHTS="action_density=0,phi2=0,phi4=0,local_kurtosis_ratio=0,NN=0,2nn=0,diag=0"

mkdir -p "$RUN_ROOT" "$KERNEL_ROOT"

train_fresh() {
  local run="$1" kernel="$2"
  "$PY" -B "$ROOT/perfect_blocking_upsampling/scripts/train_lam1p0_rqspline_detail_flow.py" \
    --run-dir "$run" --from-scratch --kernel-path "$kernel" \
    --epochs 15 --patience 5 --batch-size 128 --lr 5e-5 \
    --train-count 3000 --validation-count 1000 --split-seed 2026081001 \
    --obs-weights "$ZERO_WEIGHTS" --local-only \
    --eval-every 99 --checkpoint-every-epochs 5 \
    --patch-test-chains 32 --patch-test-sweeps 1 --generated-count 64 \
    --random-seed 2026081001 --device cpu
}

train_warm() {
  local run="$1" kernel="$2" checkpoint="$3"
  "$PY" -B "$ROOT/perfect_blocking_upsampling/scripts/train_lam1p0_rqspline_detail_flow.py" \
    --run-dir "$run" --resume-rqspline-checkpoint "$checkpoint" --kernel-path "$kernel" \
    --epochs 8 --patience 4 --batch-size 128 --lr 2e-5 \
    --obs-weights "$ZERO_WEIGHTS" --local-only \
    --eval-every 99 --checkpoint-every-epochs 4 \
    --patch-test-chains 32 --patch-test-sweeps 1 --generated-count 64 \
    --random-seed 2026081001 --device cpu
}

update_kernel() {
  local kernel="$1" checkpoint="$2" label="$3"
  "$PY" -B "$ROOT/perfect_blocking_upsampling/scripts/optimize_lam1p0_l8to16_kernel_kl.py" \
    --kernel "$kernel" --checkpoint "$checkpoint" \
    --out-kernel "$KERNEL_ROOT/${label}.json" \
    --out-summary "$RUN_ROOT/${label}_kernel_kl.json" \
    --steps 40 --batch-size 128 --n-coarse 2000 --lr 2e-4 \
    --min-k 0.50 --max-inv-k 2.0 --seed 20260810
}

echo "[0] Initial pure-NLL L8->L16 flow for the selected 5x5 kernel"
if [[ ! -f "$RUN_ROOT/flow0/checkpoints/checkpoint_best.pt" ]]; then
  train_fresh "$RUN_ROOT/flow0" "$K0"
else
  echo "reusing completed $RUN_ROOT/flow0"
fi

echo "[1] Fixed-flow KL kernel update"
update_kernel "$K0" "$RUN_ROOT/flow0/checkpoints/checkpoint_best.pt" iteration1

echo "[1] Short pure-NLL flow retune"
train_warm "$RUN_ROOT/flow1" "$KERNEL_ROOT/iteration1.json" "$RUN_ROOT/flow0/checkpoints/checkpoint_best.pt"

echo "[2] Fixed-flow KL kernel update"
update_kernel "$KERNEL_ROOT/iteration1.json" "$RUN_ROOT/flow1/checkpoints/checkpoint_best.pt" iteration2

echo "[2] Short pure-NLL flow retune"
train_warm "$RUN_ROOT/flow2" "$KERNEL_ROOT/iteration2.json" "$RUN_ROOT/flow1/checkpoints/checkpoint_best.pt"

echo "completed: $RUN_ROOT"
