#!/usr/bin/env bash
# L16->L32 alternating-KL stopping test: iteration 6 from iteration 5.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="$ROOT/../../.venv/bin/python"
PREVIOUS_RUN="$ROOT/perfect_blocking_upsampling/runs/lam1p0/training/lam1p0_L16to32_alternatingKL_highcorr5_r1_iteration5"
PREVIOUS_KERNEL="$ROOT/perfect_blocking/perfect_blocking_lam1p0/kernels/candidates/alternatingKL_L16to32_highcorr5_r1_iteration5/iteration5.json"
PREVIOUS_CHECKPOINT="$PREVIOUS_RUN/flow5/stage_oo/checkpoints/checkpoint_best_nll.pt"
RUN_ROOT="${RUN_ROOT:-$ROOT/perfect_blocking_upsampling/runs/lam1p0/training/lam1p0_L16to32_alternatingKL_highcorr5_r1_iteration6}"
KDIR="${KDIR:-$ROOT/perfect_blocking/perfect_blocking_lam1p0/kernels/candidates/alternatingKL_L16to32_highcorr5_r1_iteration6}"

mkdir -p "$RUN_ROOT" "$KDIR"

echo '[6] fixed-flow KL kernel update'
"$PY" -B "$ROOT/perfect_blocking_upsampling/scripts/optimize_lam1p0_l16to32_kernel_kl.py" \
  --kernel "$PREVIOUS_KERNEL" --checkpoint "$PREVIOUS_CHECKPOINT" \
  --out-kernel "$KDIR/iteration6.json" --out-summary "$RUN_ROOT/iteration6_kernel_kl.json" \
  --steps 40 --batch-size 64 --n-coarse 2000 --lr 1e-4 \
  --min-k 0.5 --max-inv-k 2 --seed 20260810

echo '[6] EO -> OE -> OO pure-NLL retune'
env RUN_ROOT="$RUN_ROOT/flow6" \
  SOURCE_CHECKPOINT="$PREVIOUS_CHECKPOINT" KERNEL="$KDIR/iteration6.json" \
  TOTAL_COUNT=5000 TRAIN_COUNT=4000 VAL_COUNT=500 TEST_COUNT=500 \
  EPOCHS=6 PATIENCE=3 LR=2e-5 BATCH_SIZE=128 SEED=20260810 DEVICE=cpu \
  bash "$ROOT/perfect_blocking_upsampling/scripts/run_lam1p0_l16to32_highcorr5_pure_nll_stages.sh"

echo "completed: $RUN_ROOT"
