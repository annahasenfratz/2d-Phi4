#!/usr/bin/env bash
# Two short alternating frozen-flow-KL kernel updates and EO->OE->OO NLL retunes.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; PY="$ROOT/../../.venv/bin/python"
BASE_KERNEL="$ROOT/perfect_blocking/perfect_blocking_lam1p0/kernels/candidates/allL16_chi2_R2_corrW5000_highcorr_5x5_eta_included.json"
BASE_CK="$ROOT/perfect_blocking_upsampling/runs/lam1p0/training/lam1p0_L16to32_highcorr5_pureNLL_N5000_20260807T063341Z/stage_oo/checkpoints/checkpoint_best_nll.pt"
RUN_ROOT="${RUN_ROOT:-$ROOT/perfect_blocking_upsampling/runs/lam1p0/training/lam1p0_L16to32_alternatingKL_highcorr5_r1}"
KERNEL_ROOT="${KERNEL_ROOT:-$ROOT/perfect_blocking/perfect_blocking_lam1p0/kernels/candidates/alternatingKL_L16to32_highcorr5_r1}"
mkdir -p "$RUN_ROOT" "$KERNEL_ROOT"
update_kernel() { local k="$1" ck="$2" label="$3"; "$PY" -B "$ROOT/perfect_blocking_upsampling/scripts/optimize_lam1p0_l16to32_kernel_kl.py" --kernel "$k" --checkpoint "$ck" --out-kernel "$KERNEL_ROOT/$label.json" --out-summary "$RUN_ROOT/${label}_kernel_kl.json" --steps 40 --batch-size 64 --n-coarse 2000 --lr 1e-4 --min-k .5 --max-inv-k 2 --seed 20260810; }
retune() { local run="$1" k="$2" ck="$3"; env RUN_ROOT="$run" SOURCE_CHECKPOINT="$ck" KERNEL="$k" TOTAL_COUNT=5000 TRAIN_COUNT=4000 VAL_COUNT=500 TEST_COUNT=500 EPOCHS=6 PATIENCE=3 LR=2e-5 BATCH_SIZE=128 SEED=20260810 DEVICE=cpu bash "$ROOT/perfect_blocking_upsampling/scripts/run_lam1p0_l16to32_highcorr5_pure_nll_stages.sh"; }
echo '[1] fixed-flow KL kernel update'; update_kernel "$BASE_KERNEL" "$BASE_CK" iteration1
echo '[1] EO->OE->OO pure-NLL retune'; retune "$RUN_ROOT/flow1" "$KERNEL_ROOT/iteration1.json" "$BASE_CK"
echo '[2] fixed-flow KL kernel update'; update_kernel "$KERNEL_ROOT/iteration1.json" "$RUN_ROOT/flow1/stage_oo/checkpoints/checkpoint_best_nll.pt" iteration2
echo '[2] EO->OE->OO pure-NLL retune'; retune "$RUN_ROOT/flow2" "$KERNEL_ROOT/iteration2.json" "$RUN_ROOT/flow1/stage_oo/checkpoints/checkpoint_best_nll.pt"
echo "completed: $RUN_ROOT"
