#!/usr/bin/env bash
# Exact detail-only counterpart of L16toL32_blockedNative_N1500_S400_start0_div2_D2_sc0p40_sd0p10_r1.
# Usage: bash $0 rNUMBER [--execute] [--background]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BASE="$ROOT/perfect_blocking_upsampling/scripts/submit_lam1p0_checkerboard_mh.sh"

# These are the saved parameters of the completed blocked-native N=1500 control.
export LC="${LC:-16}"
export LF="${LF:-32}"
export N_CHAINS="${N_CHAINS:-1500}"
export N_SWEEPS="${N_SWEEPS:-400}"
export START_INDEX="${START_INDEX:-0}"
export BATCH_SIZE="${BATCH_SIZE:-50}"
export SEED="${SEED:-2026080429}"
export DIVIDE="${DIVIDE:-2}"
export DETAIL_PASSES="${DETAIL_PASSES:-2}"
export DETAIL_SIGMA="${DETAIL_SIGMA:-0.10}"
export COARSE_SIGMA="${COARSE_SIGMA:-0.40}"
export COARSE_UPDATES="${COARSE_UPDATES:-1}"
export SAVE_EVERY="${SAVE_EVERY:-20}"
export INITIALIZATION="blocked_native"
export UPDATE_MODE="detail_only"
export FLOW_CHECKPOINT="$ROOT/perfect_blocking_upsampling/runs/lam1p0/training/lam1p0_L16to32_current5x5_pureNLL_from_tailstratifiedNLL_N5000_20260805T065543Z/stage_oo/checkpoints/checkpoint_best_nll.pt"
export KERNEL_PATH="$ROOT/perfect_blocking/perfect_blocking_lam1p0/kernels/final/chosen_kernel.json"

exec bash "$BASE" "$@"
