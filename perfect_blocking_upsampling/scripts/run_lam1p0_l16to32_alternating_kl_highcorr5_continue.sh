#!/usr/bin/env bash
# Final two short continuation cycles (iterations 3/4) of L16->L32 alternating KL.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; PY="$ROOT/../../.venv/bin/python"
BASE_RUN="$ROOT/perfect_blocking_upsampling/runs/lam1p0/training/lam1p0_L16to32_alternatingKL_highcorr5_r1"
BASE_K="$ROOT/perfect_blocking/perfect_blocking_lam1p0/kernels/candidates/alternatingKL_L16to32_highcorr5_r1/iteration2.json"
RUN_ROOT="${RUN_ROOT:-$ROOT/perfect_blocking_upsampling/runs/lam1p0/training/lam1p0_L16to32_alternatingKL_highcorr5_r1_continue}"
KDIR="${KDIR:-$ROOT/perfect_blocking/perfect_blocking_lam1p0/kernels/candidates/alternatingKL_L16to32_highcorr5_r1_continue}"
mkdir -p "$RUN_ROOT" "$KDIR"
update() { local k="$1" ck="$2" label="$3"; "$PY" -B "$ROOT/perfect_blocking_upsampling/scripts/optimize_lam1p0_l16to32_kernel_kl.py" --kernel "$k" --checkpoint "$ck" --out-kernel "$KDIR/$label.json" --out-summary "$RUN_ROOT/${label}_kernel_kl.json" --steps 40 --batch-size 64 --n-coarse 2000 --lr 1e-4 --min-k .5 --max-inv-k 2 --seed 20260810; }
retune() { local run="$1" k="$2" ck="$3"; env RUN_ROOT="$run" SOURCE_CHECKPOINT="$ck" KERNEL="$k" TOTAL_COUNT=5000 TRAIN_COUNT=4000 VAL_COUNT=500 TEST_COUNT=500 EPOCHS=6 PATIENCE=3 LR=2e-5 BATCH_SIZE=128 SEED=20260810 DEVICE=cpu bash "$ROOT/perfect_blocking_upsampling/scripts/run_lam1p0_l16to32_highcorr5_pure_nll_stages.sh"; }
echo '[3] fixed-flow KL update'; update "$BASE_K" "$BASE_RUN/flow2/stage_oo/checkpoints/checkpoint_best_nll.pt" iteration3
echo '[3] EO->OE->OO retune'; retune "$RUN_ROOT/flow3" "$KDIR/iteration3.json" "$BASE_RUN/flow2/stage_oo/checkpoints/checkpoint_best_nll.pt"
echo '[4] fixed-flow KL update'; update "$KDIR/iteration3.json" "$RUN_ROOT/flow3/stage_oo/checkpoints/checkpoint_best_nll.pt" iteration4
echo '[4] EO->OE->OO retune'; retune "$RUN_ROOT/flow4" "$KDIR/iteration4.json" "$RUN_ROOT/flow3/stage_oo/checkpoints/checkpoint_best_nll.pt"
echo "completed: $RUN_ROOT"
