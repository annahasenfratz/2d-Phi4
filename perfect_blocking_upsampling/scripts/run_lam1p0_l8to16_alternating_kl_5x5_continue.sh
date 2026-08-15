#!/usr/bin/env bash
# Continue the successful L8->L16 alternating-KL sequence for iterations 3 and 4.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="$ROOT/../../.venv/bin/python"
BASE_RUN="$ROOT/perfect_blocking_upsampling/runs/lam1p0/training/lam1p0_L8to16_alternatingKL_5x5_r1"
BASE_KERNEL="$ROOT/perfect_blocking/perfect_blocking_lam1p0/kernels/candidates/alternatingKL_L8to16_5x5_r1/iteration2.json"
RUN_ROOT="${RUN_ROOT:-$ROOT/perfect_blocking_upsampling/runs/lam1p0/training/lam1p0_L8to16_alternatingKL_5x5_r1_continue}"
KERNEL_ROOT="${KERNEL_ROOT:-$ROOT/perfect_blocking/perfect_blocking_lam1p0/kernels/candidates/alternatingKL_L8to16_5x5_r1_continue}"
ZERO_WEIGHTS="action_density=0,phi2=0,phi4=0,local_kurtosis_ratio=0,NN=0,2nn=0,diag=0"
mkdir -p "$RUN_ROOT" "$KERNEL_ROOT"

update_kernel() {
  local kernel="$1" checkpoint="$2" label="$3"
  "$PY" -B "$ROOT/perfect_blocking_upsampling/scripts/optimize_lam1p0_l8to16_kernel_kl.py" \
    --kernel "$kernel" --checkpoint "$checkpoint" \
    --out-kernel "$KERNEL_ROOT/${label}.json" \
    --out-summary "$RUN_ROOT/${label}_kernel_kl.json" \
    --steps 40 --batch-size 128 --n-coarse 2000 --lr 2e-4 \
    --min-k 0.50 --max-inv-k 2.0 --seed 20260810
}

retune_flow() {
  local run="$1" kernel="$2" checkpoint="$3"
  "$PY" -B "$ROOT/perfect_blocking_upsampling/scripts/train_lam1p0_rqspline_detail_flow.py" \
    --run-dir "$run" --resume-rqspline-checkpoint "$checkpoint" --kernel-path "$kernel" \
    --epochs 8 --patience 4 --batch-size 128 --lr 2e-5 \
    --obs-weights "$ZERO_WEIGHTS" --local-only \
    --eval-every 99 --checkpoint-every-epochs 4 \
    --patch-test-chains 32 --patch-test-sweeps 1 --generated-count 64 \
    --random-seed 2026081001 --device cpu
}

echo "[3] fixed-flow KL kernel update"
update_kernel "$BASE_KERNEL" "$BASE_RUN/flow2/checkpoints/checkpoint_best.pt" iteration3
echo "[3] short pure-NLL flow retune"
retune_flow "$RUN_ROOT/flow3" "$KERNEL_ROOT/iteration3.json" "$BASE_RUN/flow2/checkpoints/checkpoint_best.pt"

echo "[4] fixed-flow KL kernel update"
update_kernel "$KERNEL_ROOT/iteration3.json" "$RUN_ROOT/flow3/checkpoints/checkpoint_best.pt" iteration4
echo "[4] short pure-NLL flow retune"
retune_flow "$RUN_ROOT/flow4" "$KERNEL_ROOT/iteration4.json" "$RUN_ROOT/flow3/checkpoints/checkpoint_best.pt"
echo "completed: $RUN_ROOT"
