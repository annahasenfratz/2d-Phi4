#!/usr/bin/env bash
# Fixed iteration-5 kernel plus tail-refined conditional flow control.
# Native L32 -> c=K phi; discard native detail; draw d from refined flow; MH.
# Usage: bash $0 r1 [--execute] [--background]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
FLOW_CHECKPOINT="$ROOT/perfect_blocking_upsampling/runs/lam1p0/training/lam1p0_L16to32_alternatingKL5_flow_tailrefine_r1/stage_oo/checkpoints/checkpoint_best_patch.pt"
KERNEL_PATH="$ROOT/perfect_blocking/perfect_blocking_lam1p0/kernels/candidates/alternatingKL_L16to32_highcorr5_r1_iteration5/iteration5.json"

# Identical MH geometry and random seed to the unrefined blocked-native-flow run.
export LC="${LC:-16}"
export LF="${LF:-32}"
export N_CHAINS="${N_CHAINS:-500}"
export N_SWEEPS="${N_SWEEPS:-400}"
export START_INDEX="${START_INDEX:-0}"
export SEED="${SEED:-2026081215}"
export BATCH_SIZE="${BATCH_SIZE:-50}"
export DIVIDE="${DIVIDE:-2}"
export DETAIL_PASSES="${DETAIL_PASSES:-2}"
export COARSE_SIGMA="${COARSE_SIGMA:-0.40}"
export DETAIL_SIGMA="${DETAIL_SIGMA:-0.10}"
export COARSE_UPDATES="${COARSE_UPDATES:-1}"
export INITIAL_DETAIL_ONLY_SWEEPS="${INITIAL_DETAIL_ONLY_SWEEPS:-0}"
export SAVE_EVERY="${SAVE_EVERY:-5}"
export INITIALIZATION="blocked_native_flow"
export UPDATE_MODE="coarse_detail"
export FLOW_CHECKPOINT KERNEL_PATH
export OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/perfect_blocking_upsampling/outputs/checkerboard_mh_lam1p0/alternatingKL5_tailrefine_blockednativeflow_kernel}"

exec bash "$ROOT/perfect_blocking_upsampling/scripts/submit_lam1p0_checkerboard_mh.sh" "$@"
