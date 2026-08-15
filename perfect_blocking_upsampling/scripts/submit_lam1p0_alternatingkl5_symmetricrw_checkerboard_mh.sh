#!/usr/bin/env bash
# Test: same symmetric Gaussian checkerboard random walk for coarse and detail.
# Exact acceptance is -Delta S_f for both coordinate types.
# Usage: bash $0 r1 [--execute] [--background]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
FLOW_CHECKPOINT="$ROOT/perfect_blocking_upsampling/runs/lam1p0/training/lam1p0_L16to32_alternatingKL_highcorr5_r1_iteration5/flow5/stage_oo/checkpoints/checkpoint_best_nll.pt"
KERNEL_PATH="$ROOT/perfect_blocking/perfect_blocking_lam1p0/kernels/candidates/alternatingKL_L16to32_highcorr5_r1_iteration5/iteration5.json"
export LC="${LC:-16}" LF="${LF:-32}" N_CHAINS="${N_CHAINS:-500}" N_SWEEPS="${N_SWEEPS:-400}" START_INDEX="${START_INDEX:-0}" SEED="${SEED:-2026081221}" BATCH_SIZE="${BATCH_SIZE:-50}" DIVIDE="${DIVIDE:-2}" DETAIL_PASSES="${DETAIL_PASSES:-2}"
# Same random-walk scale for c and d, as requested.  Override COARSE_SIGMA to tune.
export COARSE_SIGMA="${COARSE_SIGMA:-0.10}" DETAIL_SIGMA="${DETAIL_SIGMA:-0.10}" COARSE_PROPOSAL_MODE="symmetric_rw" COARSE_UPDATES="${COARSE_UPDATES:-1}" INITIAL_DETAIL_ONLY_SWEEPS="${INITIAL_DETAIL_ONLY_SWEEPS:-0}" SAVE_EVERY="${SAVE_EVERY:-5}"
export FLOW_CHECKPOINT KERNEL_PATH
export OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/perfect_blocking_upsampling/outputs/checkerboard_mh_lam1p0/alternatingKL5_symmetricrw}"
exec bash "$ROOT/perfect_blocking_upsampling/scripts/submit_lam1p0_checkerboard_mh.sh" "$@"
